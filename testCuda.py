#!/usr/bin/env python3
import os
import numpy as np
from numbaCudaFallbacks import syncthreads_or, reduce
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

os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'

base_config_dict = OrderedDict(theta0=0,
                               start=2,
                               radius=1.5,
                               height=25,
                               deltaC_target=0,
                               deltaT_target=np.deg2rad(48),
                               transmission_minimum=0.85,
                               crit_angle_prop=0.999,
                               lower_bound=np.deg2rad(10),
                               upper_bound=np.deg2rad(120),
                               weight_deviation=15,
                               weight_dispersion=25,
                               weight_linearity=1000,
                               weight_transmission=35
                               )


def create_optimizer(prism_count=3, nwaves=64, config_dict={}, sample_size=10, steps=5, dr=np.deg2rad(6)):
    dr = np.float32(dr)
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
        return math.sqrt(reduce(operator.add, np.float32(err ** 2), nwaves)) / 4

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
        if syncthreads_or(abs(incident) >= crit_angle * config.crit_angle_prop):
            return
        refracted = math.asin((n1 / n2) * math.sin(incident))
        ci, cr = math.cos(incident), math.cos(refracted)
        T = np.float32(1) - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 +
                             ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / np.float32(2)
        for i in range(1, angle_count):
            n1 = n2
            n2 = n[(index // (nglass ** i)) % nglass, tid] if i < prism_count else np.float32(1)
            alpha = angles[i] if i > 1 else (angles[1] + angles[0])
            incident = refracted - alpha

            crit_angle = math.asin(n2 / n1) if n2 < n1 else np.float32(np.pi / 2)
            if syncthreads_or(abs(incident) >= crit_angle * config.crit_angle_prop):
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
            if syncthreads_or(0 > path0 or path0 > sideR or 0 > path1 or path1 > sideR):
                return
            sideL = sideR

            refracted = math.asin((n1 / n2) * math.sin(incident))
            ci, cr = math.cos(incident), math.cos(refracted)
            T *= np.float32(1) - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 +
                                  ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / np.float32(2)
        delta = config.theta0 - (refracted + offAngle)
        if tid == nwaves // 2:
            deltaC[0] = delta
        deltaT = (reduce(max, delta, nwaves) - reduce(min, delta, nwaves))
        meanT = reduce(operator.add, T, nwaves) / np.float32(nwaves)
        transmission_err = max(config.transmission_minimum - meanT, np.float32(0))
        NL = nonlinearity(delta)
        merit_err = config.weight_deviation * (deltaC[0] - config.deltaC_target) ** 2 \
                    + config.weight_dispersion * (deltaT - config.deltaT_target) ** 2 \
                    + config.weight_linearity * NL \
                    + config.weight_transmission * transmission_err
        nb.cuda.syncthreads()
        return merit_err

    @nb.cuda.jit(device=True)
    def random_search(n, index, nglass, rng):
        tid = nb.cuda.threadIdx.x
        isodd = np.float32(-1 if (tid > 1 and tid % 2 == 0) else 1)
        rid = nb.cuda.blockIdx.x * angle_count + tid
        best = np.float32(0)
        bestVal = np.float32(np.inf)
        trial = nb.cuda.shared.array(angle_count, nb.f4)
        normed = nb.cuda.shared.array(1, nb.f4)
        lbound = config.lower_bound if (tid > 1) else (config.lower_bound / np.float32(2))
        ubound = config.upper_bound if (tid > 1) else (config.upper_bound / np.float32(2))
        for _ in range(sample_size):
            if tid < angle_count:
                rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
                angle = lbound + rand * (ubound - lbound)
                trial[tid] = math.copysign(angle, isodd)
            nb.cuda.syncthreads()
            trialVal = merit_error(n, trial, index, nglass)
            if tid < angle_count and trialVal is not None and trialVal < bestVal:
                bestVal = trialVal
                best = trial[tid]
        for _ in range(steps):
            if tid < angle_count:
                xi = nb.cuda.random.xoroshiro128p_normal_float32(rng, rid)
                if tid == 0:
                    rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
                    normed[0] = dr * rand ** (1 / angle_count)
                test = best + normed[0] * xi / math.sqrt(reduce(operator.add, np.float32(xi ** 2), angle_count))
                trial[tid] = math.copysign(max(min(abs(test), ubound), lbound), isodd)
            nb.cuda.syncthreads()
            trialVal = merit_error(n, trial, index, nglass)
            if tid < angle_count and trialVal is not None and trialVal < bestVal:
                bestVal = trialVal
                best = trial[tid]
        return True, best, bestVal

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
    step = int(np.ceil(end/n))
    while start+step < end:
        yield (start, start+step)
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
gpus = nb.cuda.gpus  # [nb.cuda.gpus[1]]
ngpu = len(gpus)
output = np.recarray(ngpu * blockCount, dtype=out_dtype)
outputs = [output[slice(*b)] for b in subsect(ngpu * blockCount, ngpu)]
streams = []
outs = []
print('Starting')

t1 = time.time()
for gpu, bounds in zip(gpus, subsect(nglass ** prism_count, ngpu)):
    with gpu:
        s = nb.cuda.stream()
        rng_states = nb.cuda.random.create_xoroshiro128p_states(blockCount * angle_count, seed=42, stream=s)
        d_ns = nb.cuda.to_device(glasses, stream=s)
        d_out = nb.cuda.device_array(blockCount, dtype=out_dtype, stream=s)
        optimize[blockCount, nwaves, s, 4*nwaves](d_ns, d_out, *bounds, rng_states)
        outs.append(d_out)
        streams.append(s)
for gpu, s, d_out, h_out in zip(gpus, streams, outs, outputs):
    with gpu:
        d_out.copy_to_host(ary=h_out, stream=s)
        s.synchronize()
dt = time.time() - t1

value, ind, angles = output[np.argmin(output['value'])]
gs = [names[(ind // (nglass ** i)) % nglass] for i in range(prism_count)]
print(value, *gs, *np.rad2deg(angles))
print(dt, 's')

ns = np.stack(calc_n(gcat[name], w) for name in gs).astype(np.float32)
ret = describe(ns, angles, config)
if isinstance(ret, int):
    print('Failed at interface', ret)
else:
    err, NL, deltaT, deltaC, delta, transm = ret
    print(err, NL, np.rad2deg(delta), transm * 100)
