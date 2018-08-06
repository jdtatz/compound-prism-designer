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
from simple import describe

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
    start=1,
    radius=0.5,
    height=5,
    max_size=40,
    deltaC_target=0,
    deltaT_target=np.deg2rad(16),
    crit_angle_prop=0.999,
    lower_bound=np.deg2rad(2),
    upper_bound=np.deg2rad(120),
    weight_deviation=15,
    weight_dispersion=250,
    weight_linearity=1000,
    weight_transmission=30,
    weight_thinness=1
)


def create_optimizer(prism_count=3,
                     nwaves=64,
                     config_dict={},
                     steps=20,
                     dr=np.deg2rad(60)):
    dr = np.float32(dr)
    factor = np.float32(3 / steps)
    angle_count = prism_count + 1

    config_keys = list(base_config_dict.keys())
    config_values = tuple(config_dict.get(key, base_config_dict[key]) for key in config_keys)
    config_dtype = np.dtype(list(zip(config_keys, repeat('f4'))))
    config = np.rec.array([config_values], dtype=config_dtype)[0]

    out_dtype = np.dtype([('value', 'f4'), ('index', 'i4'), ('angles', 'f4', angle_count)])
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
        deltaC = nb.cuda.shared.array(1, nb.f4)
        n1 = np.float32(1)
        n2 = n[index % nglass, tid]
        path0 = (config.start - config.radius) / math.cos(angles[0])
        path1 = (config.start + config.radius) / math.cos(angles[0])
        sideL = config.height / math.cos(angles[0])
        incident = config.theta0 + angles[0]
        offAngle = angles[1]
        crit_angle = np.float32(math.pi / 2)
        if nb.cuda.syncthreads_or(abs(incident) >= crit_angle * config.crit_angle_prop):
            return
        refracted = math.asin((n1 / n2) * math.sin(incident))
        ci, cr = math.cos(incident), math.cos(refracted)
        T = np.float32(1) - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))**2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))**2) / np.float32(2)
        size = np.float32(0)
        for i in range(1, angle_count):
            n1 = n2
            n2 = n[(index // (nglass**i)) % nglass, tid] if i < prism_count else np.float32(1)
            alpha = angles[i] if i > 1 else (angles[1] + angles[0])
            incident = refracted - alpha

            crit_angle = math.asin(n2 / n1) if n2 < n1 else np.float32(np.pi / 2)
            if nb.cuda.syncthreads_or(abs(incident) >= crit_angle * config.crit_angle_prop):
                return

            if i > 1:
                offAngle += alpha
            sideR = config.height / math.cos(offAngle)
            t1 = np.float32(np.pi / 2) - refracted * math.copysign(np.float32(1), alpha)
            t2 = np.float32(np.pi) - abs(alpha) - t1
            los = math.sin(t1) / math.sin(t2)
            if alpha > 0:
                path0 *= los
                path1 *= los
            else:
                path0 = sideR - (sideL - path0) * los
                path1 = sideR - (sideL - path1) * los
            if nb.cuda.syncthreads_or(0 > path0 or path0 > sideR or 0 > path1 or path1 > sideR):
                return
            size += math.sqrt(sideL**2 + sideR**2 - np.float32(2) * sideL * sideR * math.cos(alpha))
            sideL = sideR

            refracted = math.asin((n1 / n2) * math.sin(incident))
            ci, cr = math.cos(incident), math.cos(refracted)
            T *= np.float32(1) - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))**2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))**2) / np.float32(2)
        delta = config.theta0 - (refracted + offAngle)
        if tid == nwaves // 2:
            deltaC[0] = delta
        deltaT = (reduce(max, delta, nwaves) - reduce(min, delta, nwaves))
        meanT = reduce(operator.add, T, nwaves) / np.float32(nwaves)
        NL = nonlinearity(delta)
        merit_err = config.weight_deviation * (deltaC[0] - config.deltaC_target) ** 2 \
                    + config.weight_dispersion * (deltaT - config.deltaT_target) ** 2 \
                    + config.weight_linearity * NL \
                    + config.weight_transmission * (np.float32(1) - meanT) \
                    + config.weight_thinness * max(size - config.max_size, np.float32(0))
        return merit_err

    @nb.cuda.jit(device=True)
    def random_search(n, index, nglass, rng):
        tid = nb.cuda.threadIdx.x
        isodd = np.float32(-1 if (tid > 1 and tid % 2 == 0) else 1)
        rid = nb.cuda.blockIdx.x * angle_count + tid
        best = np.float32(0)
        bestVal = np.float32(0)
        found = False
        trial = nb.cuda.shared.array(angle_count, nb.f4)
        lbound = config.lower_bound if (tid > 1) else (config.lower_bound / np.float32(2))
        ubound = config.upper_bound if (tid > 1) else (config.upper_bound / np.float32(2))
        if tid < angle_count:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            angle = lbound + rand * (ubound - lbound)
            trial[tid] = best = math.copysign(angle, isodd)
        nb.cuda.syncthreads()
        trialVal = merit_error(n, trial, index, nglass)
        if tid < angle_count and trialVal is not None:
            bestVal = trialVal
            found = True
        for rs in range(steps):
            if tid < angle_count:
                xi = nb.cuda.random.xoroshiro128p_normal_float32(rng, rid)
                sphere = xi / math.sqrt(reduce(operator.add, np.float32(xi**2), angle_count))
                test = best + sphere * dr * math.exp(-np.float32(rs) * factor) * np.float32(1 if tid > 1 else 0.5)
                trial[tid] = math.copysign(max(min(abs(test), ubound), lbound), isodd)
            nb.cuda.syncthreads()
            trialVal = merit_error(n, trial, index, nglass)
            if tid < angle_count and trialVal is not None and (not found or trialVal < bestVal):
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
            if tid < angle_count and valid and fx < bestVal:
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
    err, NL, deltaT, deltaC, size, delta, transm = ret
    print(err, NL, size, np.rad2deg(delta), transm * 100)
