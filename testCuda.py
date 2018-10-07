#!/usr/bin/env python3
import numpy as np
import numba as nb
import numba.cuda
import numba.cuda.random
import math
import operator
from collections import OrderedDict
from itertools import repeat
import time
from compoundprism.glasscat import read_glasscat, calc_n
from reference import describe

warp_size = np.int32(32)
mask = np.uint32(0xffffffff)
packing = np.int32(0x1f)
rtype = nb.f4


@nb.cuda.jit(device=True)
def reduce(func, val, width):
    def ilog2(v):
        return 31 - nb.cuda.clz(np.int32(v))

    def is_pow_2(v):
        return not (v & (v - 1))

    if width <= warp_size and is_pow_2(width):
        for i in range(ilog2(width)):
            val = func(val, nb.cuda.shfl_xor_sync(mask, val, 1 << i))
        return val
    elif width <= warp_size:
        closest_pow2 = np.int32(1 << ilog2(width))
        diff = np.int32(width - closest_pow2)
        lid = nb.cuda.laneid
        temp = nb.cuda.shfl_down_sync(mask, val, closest_pow2)
        if lid < diff:
            val = func(val, temp)
        for i in range(ilog2(width)):
            val = func(val, nb.cuda.shfl_xor_sync(mask, val, 1 << i))
        return nb.cuda.shfl_sync(mask, val, 0)
    else:
        warp_count = int(math.ceil(width / warp_size))
        last_warp_size = width % warp_size
        nb.cuda.syncthreads()
        buffer = nb.cuda.shared.array(0, rtype)
        tid = nb.cuda.threadIdx.x
        lid = nb.cuda.laneid
        nb.cuda.syncthreads()
        if (last_warp_size == 0) or (tid < width - last_warp_size):
            for i in range(ilog2(warp_size)):
                val = func(val, nb.cuda.shfl_xor_sync(mask, val, 1 << i))
        elif is_pow_2(last_warp_size):
            for i in range(ilog2(last_warp_size)):
                val = func(val, nb.cuda.shfl_xor_sync(mask, val, 1 << i))
        else:
            closest_lpow2 = np.int32(1 << ilog2(last_warp_size))
            temp = nb.cuda.shfl_down_sync(mask, val, closest_lpow2)
            if lid < last_warp_size - closest_lpow2:
                val = func(val, temp)
            for i in range(ilog2(closest_lpow2)):
                val = func(val, nb.cuda.shfl_xor_sync(mask, val, 1 << i))
        if lid == 0:
            buffer[tid // warp_size] = val
        nb.cuda.syncthreads()
        val = buffer[0]
        for i in range(1, warp_count):
            val = func(val, buffer[i])
        return val


base_config_dict = OrderedDict(
    theta0=0,
    start=0.9,
    radius=0.5,
    height=1,
    max_size=40,
    deviation_target=0,
    dispersion_target=np.deg2rad(24),
    lower_bound=np.deg2rad(2),
    upper_bound=np.deg2rad(80),
    weight_deviation=25,
    weight_dispersion=250,
    weight_linearity=1000,
    weight_transmittance=30,
    weight_thinness=1
)


def create_optimizer(prism_count=3,
                     nwaves=64,
                     config_dict={},
                     steps=20,
                     dr=np.deg2rad(60)):
    dr = np.float32(dr)
    factor = np.float32(3 / steps)
    surface_count = prism_count + 1

    config_keys = list(base_config_dict.keys())
    config_values = tuple(config_dict.get(key, base_config_dict[key]) for key in config_keys)
    config_dtype = np.dtype(list(zip(config_keys, repeat('f4'))))
    config = np.rec.array([config_values], dtype=config_dtype)[0]

    out_dtype = np.dtype([('value', 'f4'), ('index', 'i4'), ('angles', 'f4', surface_count)])
    out_type = nb.from_dtype(out_dtype)

    @nb.cuda.jit(device=True)
    def nonlinearity(delta):
        """Calculate the nonlinearity of the given delta spectrum"""
        delta_spectrum = nb.cuda.shared.array(0, nb.f4)
        tid = nb.cuda.threadIdx.x
        # nwaves = nb.cuda.blockDim.x
        delta_spectrum[tid] = delta
        nb.cuda.syncthreads()
        if tid == 0:
            err = (2 * delta_spectrum[2] + 2 * delta - 4 * delta_spectrum[1])
        elif tid == nwaves - 1:
            err = (2 * delta - 4 * delta_spectrum[nwaves - 2] + 2 * delta_spectrum[nwaves - 3])
        elif tid == 1:
            err = (delta_spectrum[3] - 3 * delta + 2 * delta_spectrum[0])
        elif tid == nwaves - 2:
            err = (2 * delta_spectrum[nwaves - 1] - 3 * delta + delta_spectrum[nwaves - 4])
        else:
            err = (delta_spectrum[tid + 2] + delta_spectrum[tid - 2] - 2 * delta)
        return math.sqrt(reduce(operator.add, np.float32(err**2), nwaves)) / 4

    @nb.cuda.jit(device=True)
    def merit_error(n, angles, index, nglass):
        tid = nb.cuda.threadIdx.x
        # nwaves = nb.cuda.blockDim.x
        shared_delta = nb.cuda.shared.array(3, nb.f4)
        # Initial Values
        n1 = np.float32(1)
        n2 = n[index % nglass, tid]
        r = n1 / n2
        # Rotation of (-1, 0) by angle[0] CW
        norm = -math.cos(angles[0]), -math.sin(angles[0])
        size = config.height * abs(norm[1] / norm[0])
        ray_dir = math.cos(config.theta0), math.sin(config.theta0)
        ray_path = size - (config.height - config.start) * abs(norm[1] / norm[0]), config.start
        # Snell's Law
        ci = -ray_dir[0] * norm[0] - ray_dir[1] * norm[1]
        cr_sq = np.float32(1) - r * r * (np.float32(1) - ci * ci)
        if nb.cuda.syncthreads_or(cr_sq < 0):
            return
        cr = math.sqrt(cr_sq)
        inner = r * ci - cr
        ray_dir = ray_dir[0] * r + norm[0] * inner, ray_dir[1] * r + norm[1] * inner
        # Surface Transmittance / Fresnel Equation
        fresnel_rs = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))
        fresnel_rp = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))
        transmittance = np.float32(1) - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / np.float32(2)
        for i in range(1, surface_count):
            n1 = n2
            n2 = n[(index // (nglass**i)) % nglass, tid] if i < prism_count else np.float32(1)
            r = n1 / n2
            # Rotation of (-1, 0) by angle[i] CCW
            norm = -math.cos(angles[i]), -math.sin(angles[i])
            size += config.height * abs(norm[1] / norm[0])
            prism_y = np.float32(0) if i % 2 == 1 else config.height
            ci = -ray_dir[0] * norm[0] - ray_dir[1] * norm[1]
            # Line-Plane Intersection
            d = (norm[0]*(ray_path[0] - size) + norm[1]*(ray_path[1] - prism_y)) / ci
            ray_path = d * ray_dir[0] + ray_path[0], d * ray_dir[1] + ray_path[1]
            if nb.cuda.syncthreads_or(ray_path[1] <= 0 or ray_path[1] >= config.height):
                return
            # Snell's Law
            cr_sq = np.float32(1) - r * r * (np.float32(1) - ci * ci)
            if nb.cuda.syncthreads_or(cr_sq < 0):
                return
            cr = math.sqrt(cr_sq)
            inner = r * ci - cr
            ray_dir = ray_dir[0] * r + norm[0] * inner, ray_dir[1] * r + norm[1] * inner
            # Surface Transmittance / Fresnel Equation
            fresnel_rs = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))
            fresnel_rp = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))
            transmittance *= np.float32(1) - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / np.float32(2)
        delta = math.acos(ray_dir[0])
        if tid == 0:
            shared_delta[0] = delta
        elif tid == nwaves // 2:
            shared_delta[1] = delta
        elif tid == nwaves - 1:
            shared_delta[2] = delta
        mean_transmittance = reduce(operator.add, transmittance, nwaves) / np.float32(nwaves)  # implicit sync
        deviation = shared_delta[1]
        dispersion = abs(shared_delta[2] - shared_delta[0])
        nonlin = nonlinearity(delta)  # implicit sync
        merit_err = config.weight_deviation * (deviation - config.deviation_target) ** 2 \
                    + config.weight_dispersion * (dispersion - config.dispersion_target) ** 2 \
                    + config.weight_linearity * nonlin \
                    + config.weight_transmittance * (np.float32(1) - mean_transmittance) \
                    + config.weight_thinness * max(size - config.max_size, np.float32(0))
        return merit_err

    @nb.cuda.jit(device=True)
    def random_search(n, index, nglass, rng):
        tid = nb.cuda.threadIdx.x
        isodd = np.float32(-1 if (tid % 2 == 0) else 1)
        rid = nb.cuda.blockIdx.x * surface_count + tid
        best = np.float32(0)
        bestVal = np.float32(0)
        found = False
        trial = nb.cuda.shared.array(surface_count, nb.f4)
        if tid < surface_count:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            angle = config.lower_bound + rand * (config.upper_bound - config.lower_bound)
            trial[tid] = best = math.copysign(angle, isodd)
        nb.cuda.syncthreads()
        trialVal = merit_error(n, trial, index, nglass)
        if tid < surface_count and trialVal is not None:
            bestVal = trialVal
            found = True
        for rs in range(steps):
            if tid < surface_count:
                xi = nb.cuda.random.xoroshiro128p_normal_float32(rng, rid)
                sphere = xi / math.sqrt(reduce(operator.add, np.float32(xi**2), surface_count))
                test = best + sphere * dr * math.exp(-np.float32(rs) * factor)
                trial[tid] = math.copysign(max(min(abs(test), config.upper_bound), config.lower_bound), isodd)
            nb.cuda.syncthreads()
            trialVal = merit_error(n, trial, index, nglass)
            if tid < surface_count and trialVal is not None and (not found or trialVal < bestVal):
                bestVal = trialVal
                best = trial[tid]
                found = True
        return found, best, bestVal

    @nb.cuda.jit((nb.f4[:, :], out_type[:], nb.i8, nb.i8, nb.cuda.random.xoroshiro128p_type[:]), fastmath=True)
    def optimize(ns, out, start, stop, rng):
        tid = nb.cuda.threadIdx.x
        bid = nb.cuda.blockIdx.x
        bcount = nb.cuda.gridDim.x
        bestVal = np.float32(np.inf)
        nglass = ns.shape[0]
        for index in range(start + bid, stop, bcount):
            valid, xmin, fx = random_search(ns, index, nglass, rng)
            if tid < surface_count and valid and fx < bestVal:
                bestVal = fx
                if tid == 0:
                    out[bid].value = fx
                    out[bid].index = index
                out[bid].angles[tid] = xmin
            nb.cuda.syncthreads()

    return optimize, config, out_dtype


def subsect(end, n):
    start = 0
    step = int(np.ceil(end / n))
    while start + step < end:
        yield (start, start + step)
        start += step
    if start < end:
        yield (start, end)


prism_count = 3
angle_count = prism_count + 1
nwaves = 64
optimize, config, out_dtype = create_optimizer(prism_count, nwaves)

with open('prism.ll', 'w') as f:
    f.write(optimize.inspect_llvm())
with open('prism.ptx', 'w') as f:
    f.write(optimize.ptx)

w = np.linspace(650, 1000, nwaves, dtype=np.float64)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
nglass, names = len(gcat), list(gcat.keys())
glasses = np.stack(calc_n(gcat[name], w) for name in names).astype(np.float32)

blockCount = 512
gpus = nb.cuda.gpus
#gpus = [nb.cuda.gpus[0]]
ngpu = len(gpus)
output = nb.cuda.pinned_array(ngpu * blockCount, dtype=out_dtype)
streams = []
print('Starting')

t1 = time.time()
for gpu, bounds, sbounds in zip(gpus, subsect(nglass**prism_count, ngpu), subsect(ngpu * blockCount, ngpu)):
    with gpu:
        s = nb.cuda.stream()
        rng_states = nb.cuda.random.create_xoroshiro128p_states(blockCount * angle_count, seed=42, stream=s)
        d_ns = nb.cuda.to_device(glasses, stream=s)
        d_out = nb.cuda.device_array(blockCount, dtype=out_dtype, stream=s)
        optimize[blockCount, nwaves, s, 4 * nwaves](d_ns, d_out, *bounds, rng_states)
        d_out.copy_to_host(ary=output[slice(*sbounds)], stream=s)
        streams.append(s)
for gpu, s in zip(gpus, streams):
    with gpu:
        s.synchronize()
dt = time.time() - t1

value, ind, angles = output[np.argmin(output['value'])]
gs = [names[(ind // (nglass**i)) % nglass] for i in range(prism_count)]
print(value, *gs, *np.rad2deg(angles))
print(dt, 's')

ns = np.stack(calc_n(gcat[name], w) for name in gs).astype(np.float32)
ret = describe(ns, angles, config)
if isinstance(ret, int):
    print('Failed at interface', ret)
else:
    err, NL, dispersion, deviation, size, delta, transm = ret
    print(f"error: {err}\nNL: {NL}\nsize: {size}\ndelta: {np.rad2deg(delta)}\nT: {transm *100}")