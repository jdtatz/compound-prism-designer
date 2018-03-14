#!/usr/bin/env python3.6
import os
import numpy as np
import numba as nb
import numba.cuda
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
base_weights = OrderedDict(tir=50, invalid=50, thin=2.5, deviation=5, dispersion=30, linearity=1000, transm=20)
weights_dtype = np.dtype([(k, 'f4') for k in base_weights.keys()])
weights = np.rec.array([tuple(base_weights.values())], dtype=weights_dtype)[0]


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
def merit_error(n, angles, index, nglass):
    tid = nb.cuda.threadIdx.x
    nb.cuda.syncthreads()
    delta_spectrum = nb.cuda.shared.array(nwaves, nb.f4)
    transm_spectrum = nb.cuda.shared.array(nwaves, nb.f4)
    mid = count // 2
    offAngle = sum(angles[:mid]) + angles[mid] / 2
    n1 = 1.0
    n2 = n[index % nglass, tid]
    path0 = (start - radius) / math.cos(offAngle)
    path1 = (start + radius) / math.cos(offAngle)
    sideL = height / math.cos(offAngle)
    incident = theta0 + offAngle
    crit_angle = math.pi / 2
    crit_violation_count = syncthreads_popc(abs(incident) >= crit_angle * 0.999)
    if crit_violation_count > 0:
        return weights.tir * crit_violation_count, False
    refracted = math.asin((n1 / n2) * math.sin(incident))
    ci, cr = math.cos(incident), math.cos(refracted)
    T = 1 - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / 2
    for i in range(1, count + 1):
        n1 = n2
        n2 = n[(index // (nglass ** i)) % nglass, tid] if i < count else 1.0
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
    delta_spectrum[tid] = delta = theta0 - (refracted + offAngle)
    transm_spectrum[tid] = T
    nb.cuda.syncthreads()
    deltaC = delta_spectrum[nwaves // 2]
    deltaT = (delta_spectrum.max() - delta_spectrum.min())
    meanT = sum(transm_spectrum) / nwaves
    transm_err = max(transmission_minimum - meanT, 0)
    NL = nonlinearity(delta_spectrum)
    merit_err = weights.deviation * (deltaC - deltaC_target) ** 2 \
                + weights.dispersion * (deltaT - deltaT_target) ** 2 \
                + weights.linearity * NL \
                + weights.transm * transm_err
    return merit_err, True


@nb.cuda.jit(device=True)
def minimizer(n, index, nglass):
    const_init_angles = nb.cuda.const.array_like(inital_angles)
    points = nb.cuda.local.array((count_p1, count), nb.f4)
    results = nb.cuda.local.array(count_p1, nb.f4)
    # Share mutually exclusive used arrays to save space
    xr = xc = sorter = nb.cuda.local.array(count, nb.f4)
    centroid = xe = nb.cuda.local.array(count, nb.f4)

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    x_tolerance = 1e-4
    f_tolerance = 1e-4
    
    _, break_early_test = merit_error(n, const_init_angles, index, nglass)
    if not break_early_test:
        return points[0], np.inf

    for i in range(count):
        points[:, i] = const_init_angles[i]
        points[i + 1, i] *= (1 + nonzdelt)

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
        tolCheck = True
        for i in range(1, count + 1):
            tolCheck &= abs(results[0] - results[i]) <= f_tolerance
            for j in range(count):
                tolCheck &= abs(points[0, j] - points[i, j]) <= x_tolerance
        if tolCheck:
            break

        for i in range(count):
            centroid[i] = center = sum(points[:-1, i]) / count
            xr[i] = (1 + rho) * center - rho * points[-1, i]
        fxr, _ = merit_error(n, xr, index, nglass)
        ncalls += 1

        if fxr < results[0]:  # Maybe Expansion?
            for i in range(count):
                xe[i] = (1 + rho * chi) * centroid[i] - rho * chi * points[-1, i]
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
                xc[i] = (1 + psi * rho) * centroid[i] - psi * rho * points[-1, i]
            fxc, _ = merit_error(n, xc, index, nglass)
            ncalls += 1
            if fxc <= fxr:
                for i in range(count):
                    points[-1, i] = xc[i]
                results[-1] = fxc
            else:
                for j in range(1, count + 1):
                    for i in range(count):
                        points[j, i] = points[0, i] + sigma * (points[j, i] - points[0, i])
                    results[j], _ = merit_error(n, points[j], index, nglass)
                ncalls += count
        else:  # Inside Contraction
            for i in range(count):
                xc[i] = (1 - psi) * centroid[i] + psi * points[-1, i]
            fxcc, _ = merit_error(n, xc, index, nglass)
            ncalls += 1
            if fxcc < results[-1]:
                for i in range(count):
                    points[-1, i] = xc[i]
                results[-1] = fxcc
            else:
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


@nb.cuda.jit((nb.f4[:, :], nb.f4[:], nb.i8, nb.i8), fastmath=False)
def optimize(ns, out, start, stop):
    tid = nb.cuda.threadIdx.x
    bid = nb.cuda.blockIdx.x
    bcount = nb.cuda.gridDim.x
    best = nb.cuda.shared.array(count, nb.f4)
    bestVal, bestInd = np.inf, 0
    nglass = ns.shape[0]
    for index in range(start + bid, stop, bcount):
        xmin, fx = minimizer(ns, index, nglass)
        if tid == 0 and fx < bestVal:
            bestVal = fx
            bestInd = index
            for i in range(count):
                best[i] = xmin[i]
    if tid == 0:
        outi = bid * (count + 2)
        out[outi] = bestVal
        out[outi + 1] = bestInd
        for i in range(count):
            out[outi + i + 2] = best[i]


w = np.linspace(600, 1000, nwaves, dtype=np.float64)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
nglass, names = len(gcat), list(gcat.keys())
glasses = np.stack(calc_n(gcat[name], w) for name in names).astype(np.float32)

blockCount = 512
output = np.empty(blockCount * (count + 2), np.float32)
print('Starting')

t1 = time.time()
optimize[blockCount, nwaves](glasses, output, 0, nglass ** count)
dt = time.time() - t1

indices = (count + 2) * np.where(output[::(count + 2)] == np.min(output[::(count + 2)]))[0][0]
gs = [names[(int(output[indices + 1]) // (nglass ** i)) % nglass] for i in range(count)]
angles = output[indices + 2:indices + (count + 2)]
print(output[indices], *gs, *(angles * 180 / np.pi))
print(dt, 's')


ns = np.stack(calc_n(gcat[name], w) for name in gs)
status, err, NL, deltaT, deltaC, delta, transm = describe(ns, angles, weights, start, radius, height, theta0, deltaC_target, deltaT_target, transmission_minimum)
print(err, NL, delta*180/np.pi, transm*100)