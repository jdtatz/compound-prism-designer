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
count_p1 = count + 1
start = np.float32(2)
radius = np.float32(1.5)
height = np.float32(25)
theta0 = np.float32(0)
nwaves = 64
deltaC_target = np.float32(0 * math.pi / 180)
deltaT_target = np.float32(48 * math.pi / 180)
transmission_minimum = np.float32(0.85)
crit_angle_prop = np.float32(0.999)
base_weights = OrderedDict(thin=2.5, deviation=15, dispersion=25, linearity=1000, transm=35)
ks, vs = zip(*base_weights.items())
weights_dtype = np.dtype([(k, 'f4') for k in ks])
weights = np.rec.array([tuple(vs)], dtype=weights_dtype)[0]

f_atol = np.float32(1e-2)
maxiter = 10
pop_size = 18
pop_size_m1 = pop_size - 1
crossover_probability = np.float32(0.6)
differential_weight = np.float32(0.8)
lower_bound = np.float32(np.pi/18)  # ~10 degrees
upper_bound = np.float32(2*np.pi/3)  # ~120 degrees


@nb.cuda.jit(nb.f4(nb.f4[:]), device=True)
def nonlinearity(delta):
    """Calculate the nonlinearity of the given delta spectrum"""
    g0 = (2 * delta[2] + 2 * delta[0] - 4 * delta[1]) ** 2
    gn1 = (2 * delta[-1] - 4 * delta[-2] + 2 * delta[-3]) ** 2
    g1 = (delta[3] - 3 * delta[1] + 2 * delta[0]) ** 2
    gn2 = (2 * delta[-1] - 3 * delta[-2] + delta[-4]) ** 2
    err = g0 + g1 + gn2 + gn1
    for i in range(2, nwaves - 2):
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
    path0 = (start - radius) / math.cos(offAngle)
    path1 = (start + radius) / math.cos(offAngle)
    sideL = height / math.cos(offAngle)
    incident = theta0 + offAngle
    crit_angle = np.float32(math.pi / 2)
    if syncthreads_or(abs(incident) >= crit_angle * crit_angle_prop):
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
        if syncthreads_or(abs(incident) >= crit_angle * crit_angle_prop):
            return

        if i <= mid:
            offAngle -= alpha
        elif i > mid + 1:
            offAngle += alpha
        sideR = height / math.cos(offAngle)
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
    delta_spectrum[tid] = theta0 - (refracted + offAngle)
    transm_spectrum[tid] = T
    nb.cuda.syncthreads()
    deltaC = delta_spectrum[nwaves // 2]
    deltaT = (delta_spectrum.max() - delta_spectrum.min())
    meanT = sum(transm_spectrum) / np.float32(nwaves)
    transm_err = max(transmission_minimum - meanT, np.float32(0))
    NL = nonlinearity(delta_spectrum)
    merit_err = weights.deviation * (deltaC - deltaC_target) ** 2 \
                + weights.dispersion * (deltaT - deltaT_target) ** 2 \
                + weights.linearity * NL \
                + weights.transm * transm_err
    return merit_err


@nb.cuda.jit((nb.f4[:, :], nb.i8, nb.i8, nb.cuda.random.xoroshiro128p_type[:]), device=True)
def diff_ev(n, index, nglass, rng):
    tid = nb.cuda.threadIdx.x
    rid = nb.cuda.blockIdx.x * count + tid
    population = nb.cuda.shared.array((pop_size, count), nb.f4)
    results = nb.cuda.shared.array(pop_size, nb.f4)
    y = nb.cuda.shared.array(count, nb.f4)
    fisher = nb.cuda.shared.array(pop_size_m1, nb.i4)
    minInd, minVal = 0, np.float32(np.inf)
    any_valid = False
    isodd = np.float32(-1 if (tid % 2 == 1) else 1)
    for i in range(pop_size):
        if tid < count:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            angle = lower_bound + rand * (upper_bound - lower_bound)
            population[i, tid] = math.copysign(angle, isodd)
        nb.cuda.syncthreads()
        fx = merit_error(n, population[i], index, nglass)
        if fx is not None:
            any_valid = True
            if fx < minVal:
                minInd, minVal = i, fx
            if tid == 0:
                results[i] = fx
        elif tid == 0:
            results[i] = np.float32(np.inf)
    if not any_valid:
        return False, population[0], np.float32(np.inf)
    for _ in range(maxiter):
        if syncthreads_and(tid >= pop_size or tid == minInd or abs(results[minInd] - results[tid]) <= f_atol):
            break
        for x in range(pop_size):
            if tid == 0:
                for i in range(0, pop_size_m1):
                    rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
                    j = np.int32(i * rand)
                    fisher[i], fisher[j] = fisher[j], (i if i < x else (i+1))
                rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
                fisher[0] = np.int32(count * rand)
            nb.cuda.syncthreads()
            R, a, b, c = fisher[:4]
            if tid < count:
                rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
                if tid == R or rand < crossover_probability:
                    trial = population[a, tid] + differential_weight * (population[b, tid] - population[c, tid])
                    y[tid] = math.copysign(max(min(abs(trial), upper_bound), lower_bound), isodd)
                else:
                    y[tid] = population[x, tid]
            fx = results[x]
            nb.cuda.syncthreads()
            fy = merit_error(n, y, index, nglass)
            if fy is not None and fy < fx:
                if tid == 0:
                    results[x] = fy
                if tid < count:
                    population[x, tid] = y[tid]
                if fy < minVal:
                    minInd, minVal = x, fy
    nb.cuda.syncthreads()
    return True, population[minInd], np.float32(minVal)


@nb.cuda.jit((nb.f4[:, :], nb.f4[:], nb.i8, nb.i8, nb.cuda.random.xoroshiro128p_type[:]), fastmath=True)
def optimize(ns, out, start, stop, rng):
    tid = nb.cuda.threadIdx.x
    bid = nb.cuda.blockIdx.x
    bcount = nb.cuda.gridDim.x
    best = nb.cuda.shared.array(count, nb.f4)
    bestVal, bestInd = np.float32(np.inf), 0
    nglass = ns.shape[0]
    for index in range(start + bid, stop, bcount):
        valid, xmin, fx = diff_ev(ns, index, nglass, rng)
        if tid == 0 and valid and fx < bestVal:
            bestVal = fx
            bestInd = index
            for i in range(count):
                best[i] = xmin[i]
        nb.cuda.syncthreads()
    if tid == 0:
        outi = bid * (count + 2)
        out[outi] = bestVal
        out[outi + 1] = bestInd
        for i in range(count):
            out[outi + i + 2] = best[i]


w = np.linspace(650, 1000, nwaves, dtype=np.float64)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
nglass, names = len(gcat), list(gcat.keys())
glasses = np.stack(calc_n(gcat[name], w) for name in names).astype(np.float32)

blockCount = 512
output = np.empty(blockCount * (count + 2), np.float32)
print('Starting')

t1 = time.time()
with nb.cuda.gpus[0]:
    rng_states = nb.cuda.random.create_xoroshiro128p_states(blockCount * count, seed=42)
    optimize[blockCount, nwaves](glasses, output, 0, nglass ** count, rng_states)
dt = time.time() - t1

indices = (count + 2) * np.where(output[::(count + 2)] == np.min(output[::(count + 2)]))[0][0]
gs = [names[(int(output[indices + 1]) // (nglass ** i)) % nglass] for i in range(count)]
angles = output[indices + 2:indices + (count + 2)]
print(output[indices], *gs, *(angles * 180 / np.pi))
print(dt, 's')

ns = np.stack(calc_n(gcat[name], w) for name in gs).astype(np.float32)
status, *ret = describe(ns, angles, weights, start, radius, height, theta0, deltaC_target, deltaT_target, transmission_minimum)
if status:
    err, NL, deltaT, deltaC, delta, transm = ret
    print(err, NL, delta*180/np.pi, transm*100)
else:
    print('Fail', *ret)