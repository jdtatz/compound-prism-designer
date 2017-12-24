#!/usr/bin/env python3.6
import os
import numpy as np
import numba as nb
import numba.cuda
from numbaCudaFallbacks import syncthreads_and, syncthreads_or, syncthreads_popc
import math
import operator
from collections import namedtuple, OrderedDict
import time
from compoundprism.glasscat import read_glasscat, calc_n

os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'

count = 4
count_p1 = count + 1
start = 2
radius = 1.5
height = 25
theta0 = 0
nwaves = 100
deltaC_target = 0 * math.pi / 180
deltaT_target = 32 * math.pi / 180
transmission_minimum = 0.85
inital_angles = np.full((count,), math.pi / 2, np.float32)
inital_angles[1::2] *= -1
base_weights = OrderedDict(tir=50, invalid=50, thin=2.5, deviation=5, dispersion=30, linearity=1000, transm=6)
MeritWeights = namedtuple('MeritWeights', base_weights.keys())
weights = MeritWeights(**base_weights)


@nb.cuda.jit(device=True)
def ilog2(n):
    return int(math.log(n, 2))


@nb.cuda.jit(device=True)
def isPow2(num):
    return not (num & (num - 1))


@nb.cuda.jit(device=True)
def roundToPow2(num):
    return 1 << ilog2(num)


def create_fold_func(func, warpSize=32):
    @nb.cuda.jit(device=True)
    def warpFold(value, foldLen):
        for i in range(1, 1 + ilog2(foldLen)):
            value = func(value, nb.cuda.shfl.xor(value, foldLen >> i, foldLen))
        return value

    @nb.cuda.jit(device=True)
    def partialWarpFold(value, warpId, foldLen):
        closestWarpSize = roundToPow2(foldLen)
        temp = nb.cuda.shfl.down(value, closestWarpSize)
        if warpId < foldLen - closestWarpSize:
            value = func(value, temp)
        value = warpFold(value, closestWarpSize)
        return nb.cuda.shfl.idx(value, 0)

    @nb.cuda.jit(device=True)
    def fold(value, sharedArr, foldLen):
        tID = nb.cuda.threadIdx.x
        if foldLen <= warpSize and isPow2(foldLen):
            return warpFold(value, foldLen)
        elif foldLen <= warpSize:
            return partialWarpFold(value, tID, foldLen)
        else:
            warpCount = (foldLen / warpSize + 1) if (foldLen % warpSize) else (foldLen / warpSize)
            warpCountPow2 = roundToPow2(warpCount)
            withinFullWarp = tID < foldLen - foldLen % warpSize
            warpID = tID % warpSize
            nb.cuda.syncthreads()
            if isPow2(foldLen) or withinFullWarp:
                value = warpFold(value, warpSize)
            elif isPow2(foldLen % warpSize):
                value = warpFold(value, foldLen % warpSize)
            else:
                value = partialWarpFold(value, warpID, foldLen % warpSize)
            if warpID == 0:
                sharedArr[tID // warpSize] = value
            nb.cuda.syncthreads()
            faster = (warpCount - 1) > (1 + ilog2(warpCountPow2) + (0 if isPow2(warpCount) else 1))
            allWarpsFull = foldLen % warpSize == 0
            partialBigEnough = foldLen % warpSize >= warpCountPow2

            if faster and (allWarpsFull or partialBigEnough or withinFullWarp):
                value = sharedArr[warpID % warpCount]
                if not isPow2(warpCount) and warpID < warpCount - warpCountPow2:
                    value = func(value, sharedArr[warpID + warpCountPow2])
                value = warpFold(value, warpCountPow2)
                value = nb.cuda.shfl.idx(value, 0)
            else:
                value = sharedArr[0]
                for i in range(1, warpCount):
                    value = func(value, sharedArr[i])
            return value

    return fold


foldSum = create_fold_func(operator.add)
foldMax = create_fold_func(max)
foldMin = create_fold_func(min)


@nb.cuda.jit(device=True)
def nonlinearity(delta):
    """Calculate the nonlinearity of the given delta spectrum"""
    g0 = (((delta[2] - delta[0]) / 2) - (delta[1] - delta[0])) ** 2
    gn1 = ((delta[-1] - delta[-2]) - ((delta[-1] - delta[-3]) / 2)) ** 2
    g1 = (((delta[3] - delta[1]) / 2) - (delta[1] - delta[0])) ** 2 / 4
    gn2 = ((delta[-1] - delta[-2]) - ((delta[-2] - delta[-4]) / 2)) ** 2 / 4
    err = g0 + g1 + gn2 + gn1
    for i in range(2, nwaves - 2):
        err += ((delta[i + 2] - delta[i]) - (delta[i] - delta[i - 2])) ** 2 / 16
    return math.sqrt(err)


'''
@nb.cuda.jit(device=True)
def nonlinearity(delta):
    """Calculate the nonlinearity of the given delta spectrum"""
    if tid == 0:
        err = (((delta[2] - delta[0]) / 2) - (delta[1] - delta[0])) ** 2
    elif tid == 1:
        err = (((delta[3] - delta[1]) / 2) - (delta[1] - delta[0])) ** 2 / 4
    elif tid == nwaves - 1:
        err = ((delta[-1] - delta[-2]) - ((delta[-1] - delta[-3]) / 2)) ** 2
    elif tid == nwaves - 2:
        err = ((delta[-1] - delta[-2]) - ((delta[-2] - delta[-4]) / 2)) ** 2 / 4
    else:
        err = ((delta[tid + 2] - delta[tid]) - (delta[tid] - delta[tid - 2])) ** 2 / 16
    return math.sqrt(foldSum(err))
'''


@nb.cuda.jit(device=True)
def merit_error(n, angles, ind, nglass):
    tid = nb.cuda.threadIdx.x
    nb.cuda.syncthreads()
    delta_spectrum = nb.cuda.shared.array(nwaves, nb.f8)
    transm_spectrum = nb.cuda.shared.array(nwaves, nb.f8)
    mid = count // 2
    beta = angles[mid] / 2
    for i in range(count):
        if i < mid:
            beta += angles[i]
    n1 = 1.0
    n2 = n[ind[0] % nglass, tid]
    path0 = (start - radius) / math.cos(beta)
    path1 = (start + radius) / math.cos(beta)
    offAngle = beta
    sideL = height / math.cos(offAngle)
    incident = theta0 + beta
    crit_angle = math.pi / 2
    crit_violation_count = syncthreads_popc(abs(incident) >= crit_angle * 0.999)
    if crit_violation_count > 0:
        return weights.tir * crit_violation_count, False
    refracted = math.asin((n1 / n2) * math.sin(incident))
    ci, cr = math.cos(incident), math.cos(refracted)
    T = 1 - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / 2
    for i in range(1, count + 1):
        n1 = n2
        n2 = n[(ind[0] // (nglass ** i)) % nglass, tid] if i < count else 1.0
        alpha = angles[i - 1]
        incident = refracted - alpha

        crit_angle = math.asin(n2 / n1) if n2 < n1 else (math.pi / 2)
        crit_violation_count = syncthreads_popc(abs(incident) >= crit_angle * 0.999)
        if crit_violation_count > 0:
            return weights.tir * crit_violation_count, False

        if i <= mid:
            offAngle -= alpha
        elif i > mid + 1:
            offAngle += alpha
        sideR = height / math.cos(offAngle)
        t1 = math.pi / 2 - refracted * math.copysign(1, alpha)
        t2 = math.pi - abs(alpha) - t1
        los = math.sin(t1) / math.sin(t2)
        if alpha > 0:
            path0 *= los
            path1 *= los
        else:
            path0 = sideR - (sideL - path0) * los
            path1 = sideR - (sideL - path1) * los
        invalid_count = syncthreads_popc(0 > path0 or path0 > sideR or 0 > path1 or path1 > sideR)
        if invalid_count > 0:
            return weights.invalid * invalid_count, False
        sideL = sideR

        refracted = math.asin((n1 / n2) * math.sin(incident))
        ci, cr = math.cos(incident), math.cos(refracted)
        T *= 1 - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / 2
    delta = theta0 - (refracted + offAngle)
    delta_spectrum[tid] = delta
    transm_spectrum[tid] = T
    nb.cuda.syncthreads()
    meanT = 0
    deltaC = delta_spectrum[nwaves // 2]
    minVal, maxVal = delta_spectrum[0], delta_spectrum[0]
    for i in range(nwaves):
        minVal = min(minVal, delta_spectrum[i])
        maxVal = max(maxVal, delta_spectrum[i])
        meanT += transm_spectrum[i]
    deltaT = (maxVal - minVal)
    meanT /= nwaves
    transm_err = min(transmission_minimum - meanT, 0)
    NL = nonlinearity(delta_spectrum)
    merit_err = weights.deviation * (deltaC - deltaC_target) ** 2 \
                + weights.dispersion * (deltaT - deltaT_target) ** 2 \
                + weights.linearity * NL \
                + weights.transm * transm_err
    return merit_err, True


@nb.cuda.jit(device=True)
def minimizer(n, index, nglass):
    const_init_angles = nb.cuda.const.array_like(inital_angles)
    points = nb.cuda.local.array((count_p1, count), nb.f8)
    results = nb.cuda.local.array(count_p1, nb.f8)
    # Share mutually exclusive used arrays to save space
    xr = xc = sorter = nb.cuda.local.array(count, nb.f8)
    xe = nb.cuda.local.array(count, nb.f8)

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    x_tolerance = 1e-4
    f_tolerance = 1e-4
    
    _, test = merit_error(n, const_init_angles, index, nglass)
    if not test:
        return points[0], np.inf

    for i in range(count):
        points[0, i] = const_init_angles[i]
    for k in range(count):
        for i in range(count):
            points[k + 1, i] = const_init_angles[i]
        points[k + 1, k] *= (1 + nonzdelt)

    max_iter = count * 200
    max_call = count * 200

    for k in range(count + 1):
        results[k], _ = merit_error(n, points[k], index, nglass)
    ncalls = count

    # sort
    for i in range(count + 1):
        fx = results[i]
        for k in range(count):
            sorter[k] = points[i, k]
        j = i - 1
        while j >= 0 and results[j] > fx:
            results[j + 1] = results[j]
            for k in range(count):
                points[j + 1, k] = points[j, k]
            j -= 1
        results[j + 1] = fx
        for k in range(count):
            points[j + 1, k] = sorter[k]

    iterations = 1
    while ncalls < max_call and iterations < max_iter:
        # Tolerence Check
        maxF, maxS = 0, 0
        for i in range(1, count + 1):
            maxF = max(maxF, abs(results[0] - results[i]))
            for j in range(count):
                maxS = max(maxS, abs(points[0, j] - points[i, j]))
        if maxS <= x_tolerance and maxF <= f_tolerance:
            break

        for i in range(count):
            centroid = 0
            for s in range(count):
                centroid += points[s, i]
            xr[i] = (1 + rho) * centroid / count - rho * points[-1, i]
        fxr, _ = merit_error(n, xr, index, nglass)
        ncalls += 1
        doshrink = False

        if fxr < results[0]:  # Reflection or Expansion
            for i in range(count):
                centroid = 0
                for s in range(count):
                    centroid += points[s, i]
                xe[i] = (1 + rho * chi) * centroid / count - rho * chi * points[-1, i]
            fxe, _ = merit_error(n, xe, index, nglass)
            ncalls += 1
            if fxe < fxr:  # Expansion
                for i in range(count):
                    points[-1, i] = xe[i]
                results[-1] = fxe
            else:  # Reflection
                for i in range(count):
                    points[-1, i] = xr[i]
                results[-1] = fxr
        elif fxr < results[-2]:  # Reflection
            for i in range(count):
                points[-1, i] = xr[i]
            results[-1] = fxr
        elif fxr < results[-1]:  # Contraction
            for i in range(count):
                centroid = 0
                for s in range(count):
                    centroid += points[s, i]
                xc[i] = (1 + psi * rho) * centroid / count - psi * rho * points[-1, i]
            fxc, _ = merit_error(n, xc, index, nglass)
            ncalls += 1
            if fxc <= fxr:
                for i in range(count):
                    points[-1, i] = xc[i]
                results[-1] = fxc
            else:
                doshrink = True
        else:  # Inside Contraction
            for i in range(count):
                centroid = 0
                for s in range(count):
                    centroid += points[s, i]
                xc[i] = (1 - psi) * centroid / count + psi * points[-1, i]
            fxcc, _ = merit_error(n, xc, index, nglass)
            ncalls += 1
            if fxcc < results[-1]:
                for i in range(count):
                    points[-1, i] = xc[i]
                results[-1] = fxcc
            else:
                doshrink = True
        if doshrink:
            for j in range(1, count + 1):
                for i in range(count):
                    points[j, i] = points[0, i] + sigma * (points[j, i] - points[0, i])
                results[j], _ = merit_error(n, points[j], index, nglass)
            ncalls += count
        # sort
        for i in range(count + 1):
            x = results[i]
            for k in range(count):
                sorter[k] = points[i, k]
            j = i - 1
            while j >= 0 and results[j] > x:
                results[j + 1] = results[j]
                for k in range(count):
                    points[j + 1, k] = points[j, k]
                j -= 1
            results[j + 1] = x
            for k in range(count):
                points[j + 1, k] = sorter[k]
        iterations += 1
    return points[0], results[0]


@nb.cuda.jit((nb.i4[:], nb.f8[:, :], nb.f8[:]), fastmath=False)
def optimize(counter, ns, out):
    tid = nb.cuda.threadIdx.x
    bid = nb.cuda.blockIdx.x
    index = nb.cuda.shared.array(1, nb.i4)
    best = nb.cuda.shared.array(count, nb.f8)
    bestVal, bestInd = np.inf, 0
    #n = nb.cuda.shared.array((count, nwaves), nb.f8)
    nglass = ns.shape[0]
    top = nglass ** count
    while True:
        if tid == 0:
            index[0] = nb.cuda.atomic.add(counter, 0, 1)
        nb.cuda.syncthreads()
        if index[0] >= top:
            break
        #for i in range(count):
        #    n[i, tid] = ns[(index[0] // (nglass ** i)) % nglass, tid]
        #nb.cuda.syncthreads()
        xmin, fx = minimizer(ns, index, nglass)
        if tid == 0 and fx < bestVal:
            bestVal = fx
            bestInd = index[0]
            for i in range(count):
                best[i] = xmin[i]
    if tid == 0:
        outi = bid * (count + 2)
        out[outi] = bestVal
        out[outi + 1] = bestInd
        for i in range(count):
            out[outi + i + 2] = best[i]


w = np.linspace(500, 820, nwaves, dtype=np.float64)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
nglass, names = len(gcat), list(gcat.keys())
glasses = np.stack(calc_n(gcat[name], w) for name in names).astype(np.float64)

blockCount = 512
output = np.empty(blockCount * (count + 2), np.float64)
counter = np.zeros((10,), np.int32)
print('Starting')

nb.cuda.profile_start()
t1 = time.time()
optimize[blockCount, nwaves](counter, glasses, output)
dt = time.time() - t1
nb.cuda.profile_stop()

indices = (count + 2) * np.where(output[::(count + 2)] == np.min(output[::(count + 2)]))[0][0]
gs = [names[(int(output[indices + 1]) // (nglass ** i)) % nglass] for i in range(count)]
print(output[indices], *gs, *(output[indices + 2:indices + (count + 2)] * 180 / np.pi))
print(dt, 's')
