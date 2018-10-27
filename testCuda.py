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
from ray import Ray, ray_intersect_surface, ray_intersect_lens, ray_intersect_spectrometer


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
        last_warp_size = width % warp_size
        warp_count = width // warp_size + (1 if last_warp_size else 0)
        nb.cuda.syncthreads()
        buffer = nb.cuda.shared.array(0, rtype)
        tid = nb.cuda.threadIdx.x + nb.cuda.threadIdx.y * nb.cuda.blockDim.x
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
    radius=0.01,
    sheight=3.2,
    max_size=30,
    lower_bound=np.deg2rad(2),
    upper_bound=np.deg2rad(80),
    weight_deviation=5,
    weight_dispersion=10,
    weight_linearity=1000,
    weight_transmittance=5,
    weight_spot=1
)


def create_optimizer(prism_count=3,
                     nwaves=64,
                     config_dict={},
                     steps=20,
                     dr=np.deg2rad(60)):
    dr = np.float32(dr)
    factor = np.float32(3 / steps)
    surface_count = prism_count + 1
    param_count = prism_count + 3

    config_keys = list(base_config_dict.keys())
    config_values = tuple(config_dict.get(key, base_config_dict[key]) for key in config_keys)
    config_dtype = np.dtype(list(zip(config_keys, repeat('f4'))))
    config = np.rec.array([config_values], dtype=config_dtype)[0]

    out_dtype = np.dtype([('value', 'f4'), ('index', 'i4'), ('curvature', 'f4'), ('sangle', 'f4'), ('angles', 'f4', surface_count)])
    out_type = nb.from_dtype(out_dtype)

    @nb.cuda.jit(device=True)
    def nonlinearity(vals):
        """Calculate the nonlinearity of the given delta spectrum"""
        tid = nb.cuda.threadIdx.x
        val = vals[tid]
        if tid == 0:
            err = (np.float32(2) * vals[2] + np.float32(2) * val - np.float32(4) * vals[1])
        elif tid == nwaves - 1:
            err = (np.float32(2) * val - np.float32(4) * vals[nwaves - 2] + np.float32(2) * vals[nwaves - 3])
        elif tid == 1:
            err = (vals[3] - np.float32(3) * val + np.float32(2) * vals[0])
        elif tid == nwaves - 2:
            err = (np.float32(2) * vals[nwaves - 1] - np.float32(3) * val + vals[nwaves - 4])
        else:
            err = (vals[tid + 2] + vals[tid - 2] - np.float32(2) * val)
        return math.sqrt(reduce(operator.add, np.float32(err**2), nwaves)) / np.float32(4)

    @nb.cuda.jit(device=True)
    def merit_error(n, params):
        nb.cuda.syncthreads()
        tix = nb.cuda.threadIdx.x
        tiy = nb.cuda.threadIdx.y
        # nwaves = nb.cuda.blockDim.x
        shared_data = nb.cuda.shared.array(12, nb.f4)
        shared_pos = nb.cuda.shared.array((3, nwaves), nb.f4)
        # Initial Surface
        n1, n2 = np.float32(1), n[0]
        normal = -math.cos(params[2]), -math.sin(params[2])
        size = abs(normal[1] / normal[0])
        start = config.start if tiy == 0 else \
            ((config.start + config.radius) if tiy == 1 else (config.start - config.radius))
        vertex = size, np.float32(1)
        inital = Ray((np.float32(0), start), (math.cos(config.theta0), math.sin(config.theta0)), np.float32(1))
        ray = ray_intersect_surface(inital, vertex, normal, n1, n2)
        if nb.cuda.syncthreads_or(ray is None):
            return
        for i in range(1, prism_count):
            n1, n2 = n2, n[i]
            normal = -math.cos(params[2 + i]), -math.sin(params[2 + i])
            size += abs(normal[1] / normal[0])
            vertex = size, np.float32((i + 1) % 2)
            ray = ray_intersect_surface(ray, vertex, normal, n1, n2)
            if nb.cuda.syncthreads_or(ray is None):
                return
        # Last / Convex Surface
        n1, n2 = n2, np.float32(1)
        normal = -math.cos(params[2 + prism_count]), -math.sin(params[2 + prism_count])
        diff = abs(normal[1] / normal[0])
        size += diff
        midpt = size - diff / np.float32(2), np.float32(0.5)
        curvature = params[0]
        ray = ray_intersect_lens(ray, midpt, normal, curvature, n1, n2)
        if nb.cuda.syncthreads_or(ray is None):
            return
        # Spectrometer
        if tix == nwaves // 2 and tiy == 0:
            shared_data[0] = ray.p[0]
            shared_data[1] = ray.p[1]
            shared_data[2] = ray.v[0]
            shared_data[3] = ray.v[1]
        elif tix == 0 and tiy == 0:
            shared_data[4] = ray.p[0]
            shared_data[5] = ray.p[1]
            shared_data[6] = ray.v[0]
            shared_data[7] = ray.v[1]
        elif tix == nwaves - 1 and tiy == 0:
            shared_data[8] = ray.p[0]
            shared_data[9] = ray.p[1]
            shared_data[10] = ray.v[0]
            shared_data[11] = ray.v[1]
        nb.cuda.syncthreads()
        upper_ray = Ray((shared_data[4], shared_data[5]), (shared_data[6], shared_data[7]), np.float32(1))
        lower_ray = Ray((shared_data[8], shared_data[9]), (shared_data[10], shared_data[11]), np.float32(1))
        if upper_ray.p[1] < lower_ray.p[1]:
            upper_ray, lower_ray = lower_ray, upper_ray
        spec_length = config.sheight
        spec_angle = params[1]
        out = ray_intersect_spectrometer(ray, upper_ray, lower_ray, spec_angle, spec_length)
        if out is None:
            return
        n_spec_pos, ray = out
        shared_pos[tiy, tix] = n_spec_pos
        mean_transmittance = reduce(operator.add, ray.T, nwaves) / np.float32(nwaves)  # implicit sync
        nonlin = nonlinearity(shared_pos[0])
        deviation = abs(shared_data[3])
        spot_size = abs(shared_pos[1, tix] - shared_pos[2, tix])
        mean_spot_size = reduce(operator.add, spot_size, nwaves) / np.float32(nwaves)  # implicit sync
        merit_err = config.weight_deviation * deviation\
                    + config.weight_linearity * nonlin \
                    + config.weight_transmittance * (np.float32(1) - mean_transmittance) \
                    + config.weight_spot * mean_spot_size
        return merit_err

    @nb.cuda.jit(device=True)
    def random_search(n, rng, lbound, ubound):
        tid = nb.cuda.threadIdx.x + nb.cuda.threadIdx.y * nb.cuda.blockDim.x
        rid = nb.cuda.blockIdx.x * param_count + tid
        best = np.float32(0)
        bestVal = np.float32(0)
        found = False
        trial = nb.cuda.shared.array(param_count, nb.f4)
        if tid < param_count:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            trial[tid] = best = lbound + rand * (ubound - lbound)
        nb.cuda.syncthreads()
        trialVal = merit_error(n, trial)
        if tid < param_count and trialVal is not None:
            bestVal = trialVal
            found = True
        for rs in range(steps):
            if tid < param_count:
                xi = nb.cuda.random.xoroshiro128p_normal_float32(rng, rid)
                sphere = xi / math.sqrt(reduce(operator.add, xi ** 2, param_count))
                test = best + sphere * dr * math.exp(-np.float32(rs) * factor)
                trial[tid] = max(min(test, ubound), lbound)
            nb.cuda.syncthreads()
            trialVal = merit_error(n, trial)
            if tid < param_count and trialVal is not None and (not found or trialVal < bestVal):
                bestVal = trialVal
                best = trial[tid]
                found = True
        return found, best, bestVal

    @nb.cuda.jit((nb.f4[:, :], out_type[:], nb.i8, nb.i8, nb.cuda.random.xoroshiro128p_type[:]), fastmath=False, debug=True)
    def optimize(ns, out, start, stop, rng):
        tid = nb.cuda.threadIdx.x + nb.cuda.threadIdx.y * nb.cuda.blockDim.x
        tix = nb.cuda.threadIdx.x
        bid = nb.cuda.blockIdx.x
        bcount = nb.cuda.gridDim.x
        lbound, ubound = config.lower_bound, config.upper_bound
        if tid == 0:  # curvature
            lbound, ubound = np.float32(0), np.float32(1)
        elif tid == 1:  # spectrometer angle
            lbound, ubound = np.float32(-np.pi / 2), np.float32(np.pi / 2)
        elif tid < param_count and tid % 2 == 0:  # Odd surfaces
            lbound, ubound = -ubound, -lbound
        bestVal = np.float32(np.inf)
        nglass = np.int64(ns.shape[0])
        n = nb.cuda.local.array(prism_count, nb.f4)
        for index in range(start + bid, stop, bcount):
            tot = np.int64(1)
            for i in range(prism_count):
                n[i] = ns[(index // tot) % nglass, tix]
                tot *= nglass
            valid, xmin, fx = random_search(n, rng, lbound, ubound)
            if tid < param_count and valid and fx < bestVal:
                bestVal = fx
                if tid == 0:
                    out[bid].value = fx
                    out[bid].index = index
                    out[bid].curvature = xmin
                elif tid == 1:
                    out[bid].sangle = xmin
                else:
                    out[bid].angles[tid-2] = xmin
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
param_count = prism_count + 3
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
        rng_states = nb.cuda.random.create_xoroshiro128p_states(blockCount * param_count, seed=42, stream=s)
        d_ns = nb.cuda.to_device(glasses, stream=s)
        d_out = nb.cuda.device_array(blockCount, dtype=out_dtype, stream=s)
        optimize[blockCount, (nwaves, 3), s, 0](d_ns, d_out, *bounds, rng_states)
        d_out.copy_to_host(ary=output[slice(*sbounds)], stream=s)
        streams.append(s)
for gpu, s in zip(gpus, streams):
    with gpu:
        s.synchronize()
dt = time.time() - t1

value, ind, curvature, sangle, angles = output[np.argmin(output['value'])]
gs = [names[(ind // (nglass**i)) % nglass] for i in range(prism_count)]
print(f"error: {value}, curvature: {curvature}, spec angle: {sangle}, glasses: {gs}, angles {np.rad2deg(angles)}")
print(dt, 's')

ns = np.stack(calc_n(gcat[name], w) for name in gs).astype(np.float32)
ret = describe(ns, angles, curvature, sangle, config)
if isinstance(ret, int):
    print('Failed at interface', ret)
else:
    err, NL, dispersion, deviation, size, spec_pos, transm = ret
    print(f"error: {err}\nNL: {NL}\nsize: {size}\nspectrometer position: {spec_pos}\nT: {transm * 100}")
