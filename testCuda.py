#!/usr/bin/env python3.6
import os
import numpy as np
import numba as nb
import numba.cuda
import numba.cuda.random
from numbaCudaFallbacks import syncthreads_and, syncthreads_or, syncthreads_popc
import math
import operator
from collections import OrderedDict
import time
from compoundprism.glasscat import read_glasscat, calc_n
from simple import describe

os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'

count = 3
nwaves = 64

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
config_dtype = np.dtype([(k, 'f4') for k in ks])
config = np.rec.array([tuple(vs)], dtype=config_dtype)[0]

base_weights = OrderedDict(thin=2.5, deviation=15, dispersion=25, linearity=1000, transm=35)
ks, vs = zip(*base_weights.items())
weights_dtype = np.dtype([(k, 'f4') for k in ks])
weights = np.rec.array([tuple(vs)], dtype=weights_dtype)[0]

sample_size = 10
steps = 15
dr = np.float32(np.pi / 30)


@nb.cuda.jit(device=True)
def nonlinearity(delta):
    """Calculate the nonlinearity of the given delta spectrum"""
    g0 = (2 * delta[2] + 2 * delta[0] - 4 * delta[1]) ** 2
    gn1 = (2 * delta[-1] - 4 * delta[-2] + 2 * delta[-3]) ** 2
    g1 = (delta[3] - 3 * delta[1] + 2 * delta[0]) ** 2
    gn2 = (2 * delta[-1] - 3 * delta[-2] + delta[-4]) ** 2
    err = g0 + g1 + gn2 + gn1
    for i in range(2, delta.shape[0] - 2):
        err += (delta[i + 2] + delta[i - 2] - 2 * delta[i]) ** 2
    return math.sqrt(err) / 4


@nb.cuda.jit(device=True)
def merit_error(n, angles, index, nglass):
    tid = nb.cuda.threadIdx.x
    delta_spectrum = nb.cuda.shared.array(nwaves, nb.f4)
    transm_spectrum = nb.cuda.shared.array(nwaves, nb.f4)
    mid = count // 2
    offAngle = sum(angles[:mid]) + angles[mid] / np.float32(2)
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
    delta_spectrum[tid] = config.theta0 - (refracted + offAngle)
    transm_spectrum[tid] = T
    nb.cuda.syncthreads()
    deltaC = delta_spectrum[nwaves // 2]
    deltaT = (delta_spectrum.max() - delta_spectrum.min())
    meanT = sum(transm_spectrum) / np.float32(nwaves)
    transm_err = max(config.transmission_minimum - meanT, np.float32(0))
    NL = nonlinearity(delta_spectrum)
    merit_err = weights.deviation * (deltaC - config.deltaC_target) ** 2 \
                + weights.dispersion * (deltaT - config.deltaT_target) ** 2 \
                + weights.linearity * NL \
                + weights.transm * transm_err
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
        if tid == 0:
            normed[0] = 0
        nb.cuda.syncthreads()
        if tid < count:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            trial[tid] = v = 2 * rand - 1
            nb.cuda.atomic.add(normed, 0, v**2)
        nb.cuda.syncthreads()
        if tid == 0:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            normed[0] = math.sqrt(normed[0]) * dr * rand ** (1 / count)
        nb.cuda.syncthreads()
        if tid < count:
            trial[tid] = math.copysign(max(min(abs(best + normed[0] * trial[tid]), config.upper_bound), config.lower_bound), isodd)
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


w = np.linspace(650, 1000, nwaves, dtype=np.float64)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
nglass, names = len(gcat), list(gcat.keys())
glasses = np.stack(calc_n(gcat[name], w) for name in names).astype(np.float32)

blockCount = 512
output = np.recarray(blockCount, out_type)
print('Starting')

t1 = time.time()
with nb.cuda.gpus[1]:
    rng_states = nb.cuda.random.create_xoroshiro128p_states(blockCount * count, seed=42)
    optimize[blockCount, nwaves](glasses, output, 0, nglass ** count, rng_states)
dt = time.time() - t1

value, ind, angles = output[np.argmin(output.value)]
gs = [names[(ind // (nglass ** i)) % nglass] for i in range(count)]
print(value, *gs, *np.rad2deg(angles))
print(dt, 's')

ns = np.stack(calc_n(gcat[name], w) for name in gs).astype(np.float32)
status, *ret = describe(ns, angles, config, weights)
if status:
    err, NL, deltaT, deltaC, delta, transm = ret
    print(err, NL, delta*180/np.pi, transm*100)
else:
    print('Fail', *ret)