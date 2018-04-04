#!/usr/bin/env python3
import os
import numpy as np
from numbaCudaFallbacks import syncthreads_or, create_reduce
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

count = 3
nwaves = 64

reduce_add_f32 = create_reduce(operator.add)
reduce_max_f32 = create_reduce(max)
reduce_min_f32 = create_reduce(min)

config_dict = OrderedDict(theta0=0,
                          start=2,
                          radius=1.5,
                          height=25,
                          deltaC_target=0,
                          deltaT_target=np.deg2rad(48),
                          transmission_minimum=0.85,
                          crit_angle_prop=0.999,
                          lower_bound=np.deg2rad(10),
                          upper_bound=np.deg2rad(120)
                         )
ks, vs = zip(*config_dict.items())
config_dtype = np.dtype(list(zip(ks, repeat('f4'))))
config = np.rec.array([tuple(vs)], dtype=config_dtype)[0]

base_weights = OrderedDict(thin=2.5, deviation=15, dispersion=25, linearity=1000, transm=35)
ks, vs = zip(*base_weights.items())
weights_dtype = np.dtype(list(zip(ks, repeat('f4'))))
weights = np.rec.array([tuple(vs)], dtype=weights_dtype)[0]

sample_size = 10
steps = 5
dr = np.float32(np.deg2rad(6))


@nb.cuda.jit(device=True)
def nonlinearity(delta):
    """Calculate the nonlinearity of the given delta spectrum"""
    nb.cuda.syncthreads()
    delta_spectrum = nb.cuda.shared.array(0, nb.f4)
    tid = nb.cuda.threadIdx.x
    # nwaves = nb.cuda.blockDim.x
    delta_spectrum[tid] = delta
    nb.cuda.syncthreads()
    if tid == 0:
        err = (2 * delta_spectrum[2] + 2 * delta - 4 * delta_spectrum[1])
    elif tid == nwaves - 1:
        err = (2 * delta - 4 * delta_spectrum[nwaves-2] + 2 * delta_spectrum[nwaves-3])
    elif tid == 1:
        err = (delta_spectrum[3] - 3 * delta + 2 * delta_spectrum[0])
    elif tid == nwaves - 2:
        err = (2 * delta_spectrum[nwaves-1] - 3 * delta + delta_spectrum[nwaves-4])
    else:
        err = (delta_spectrum[tid + 2] + delta_spectrum[tid - 2] - 2 * delta)
    nb.cuda.syncthreads()
    return math.sqrt(reduce_add_f32(err ** 2, nwaves)) / 4


@nb.cuda.jit(device=True)
def merit_error(n, angles, index, nglass):
    tid = nb.cuda.threadIdx.x
    # nwaves = nb.cuda.blockDim.x
    deltaC = nb.cuda.shared.array(1, nb.f4)
    mid = count // 2
    offAngle = angles[:mid].sum() + angles[mid] / np.float32(2)
    n1 = np.float32(1)
    n2 = n[index % nglass, tid]
    path0 = (config.start - config.radius) / math.cos(offAngle)
    path1 = (config.start + config.radius) / math.cos(offAngle)
    sideL = config.height / math.cos(offAngle)
    incident = config.theta0 + offAngle
    crit_angle = np.float32(math.pi / 2)
    if syncthreads_or(abs(incident) >= crit_angle * config.crit_angle_prop):
        return
    refracted = math.asin((n1 / n2) * math.sin(incident))
    ci, cr = math.cos(incident), math.cos(refracted)
    T = np.float32(1) - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / np.float32(2)
    for i in range(1, count + 1):
        n1 = n2
        n2 = n[(index // (nglass ** i)) % nglass, tid] if i < count else np.float32(1)
        alpha = angles[i - 1]
        incident = refracted - alpha

        crit_angle = math.asin(n2 / n1) if n2 < n1 else np.float32(np.pi / 2)
        if syncthreads_or(abs(incident) >= crit_angle * config.crit_angle_prop):
            return

        if i <= mid:
            offAngle -= alpha
        elif i > mid + 1:
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
        T *= np.float32(1) - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / np.float32(2)
    delta = config.theta0 - (refracted + offAngle)
    if tid == nwaves // 2:
        deltaC[0] = delta
    deltaT = (reduce_max_f32(delta, nwaves) - reduce_min_f32(delta, nwaves))
    meanT = reduce_add_f32(T, nwaves) / np.float32(nwaves)
    transm_err = max(config.transmission_minimum - meanT, np.float32(0))
    NL = nonlinearity(delta)
    merit_err = weights.deviation * (deltaC[0] - config.deltaC_target) ** 2 \
                + weights.dispersion * (deltaT - config.deltaT_target) ** 2 \
                + weights.linearity * NL \
                + weights.transm * transm_err
    nb.cuda.syncthreads()
    return merit_err


@nb.cuda.jit(device=True)
def random_search(n, index, nglass, rng):
    tid = nb.cuda.threadIdx.x
    isodd = np.float32(-1 if (tid % 2 == 1) else 1)
    rid = nb.cuda.blockIdx.x * count + tid
    best = np.float32(0)
    bestVal = np.float32(np.inf)
    trial = nb.cuda.shared.array(count, nb.f4)
    normed = nb.cuda.shared.array(1, nb.f4)
    for _ in range(sample_size):
        if tid < count:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            angle = config.lower_bound + rand * (config.upper_bound - config.lower_bound)
            trial[tid] = math.copysign(angle, isodd)
        nb.cuda.syncthreads()
        trialVal = merit_error(n, trial, index, nglass)
        if tid < count and trialVal is not None and trialVal < bestVal:
            bestVal = trialVal
            best = trial[tid]
    for _ in range(steps):
        if tid < count:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            xi = 2 * rand - 1
            if tid == 0:
                rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
                normed[0] = dr * rand ** (1 / count)
            test = best + normed[0] * xi / math.sqrt(reduce_add_f32(xi**2, count))
            trial[tid] = math.copysign(max(min(abs(test), config.upper_bound), config.lower_bound), isodd)
        nb.cuda.syncthreads()
        trialVal = merit_error(n, trial, index, nglass)
        if tid < count and trialVal is not None and trialVal < bestVal:
            bestVal = trialVal
            best = trial[tid]
    return True, best, bestVal


out_dtype = np.dtype([('value', 'f4'), ('index', 'i4'), ('angles', 'f4', count)])
out_type = nb.from_dtype(out_dtype)


@nb.cuda.jit((nb.f4[:, :], out_type[:], nb.i8, nb.i8, nb.cuda.random.xoroshiro128p_type[:]), fastmath=True)
def optimize(ns, out, start, stop, rng):
    tid = nb.cuda.threadIdx.x
    bid = nb.cuda.blockIdx.x
    bcount = nb.cuda.gridDim.x
    bestVal = np.float32(np.inf)
    nglass = ns.shape[0]
    for index in range(start + bid, stop, bcount):
        valid, xmin, fx = random_search(ns, index, nglass, rng)
        if tid < count and valid and fx < bestVal:
            bestVal = fx
            if tid == 0:
                out[bid].value = fx
                out[bid].index = index
            out[bid].angles[tid] = xmin
        nb.cuda.syncthreads()


with open('prism.ll', 'w') as f:
    f.write(optimize.inspect_llvm())
with open('prism.ptx', 'w') as f:
    f.write(optimize.ptx)

w = np.linspace(650, 1000, nwaves, dtype=np.float64)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
nglass, names = len(gcat), list(gcat.keys())
glasses = np.stack(calc_n(gcat[name], w) for name in names).astype(np.float32)

blockCount = 512
output = np.recarray(blockCount, out_dtype)
print('Starting')

t1 = time.time()
with nb.cuda.gpus[0]:
    rng_states = nb.cuda.random.create_xoroshiro128p_states(blockCount * count, seed=42)
    optimize[blockCount, nwaves, None, 4*nwaves](glasses, output, 0, nglass ** count, rng_states)
dt = time.time() - t1

value, ind, angles = output[np.argmin(output.value)]
gs = [names[(ind // (nglass ** i)) % nglass] for i in range(count)]
print(value, *gs, *np.rad2deg(angles))
print(dt, 's')

ns = np.stack(calc_n(gcat[name], w) for name in gs).astype(np.float32)
ret = describe(ns, angles, config, weights)
if isinstance(ret, int):
    print('Failed at interface', ret)
else:
    err, NL, deltaT, deltaC, delta, transm = ret
    print(err, NL, delta*180/np.pi, transm*100)
