import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
import numpy as np
import numba as nb
import numba.cuda
import math
from collections import namedtuple, OrderedDict


prism_count = 3
glass_count = prism_count
surf_count = prism_count+1
start = 6
radius = 5
height = 25
theta0 = 0
angle_limit = 65.0 * math.pi / 180.0
nwaves = 100
deltaC_target = 0
deltaT_target = 4.0 * math.pi / 180
initial_angles = np.array([90, -90, 90, -90, 90, -90]) * math.pi / 180

base_weights = OrderedDict(tir=0.1, valid=1.0, crit_angle=1.0, thin=0.25, deviation=1.0, dispersion=1.0, linearity=1.0)
MeritWeights = namedtuple('MeritWeights', base_weights.keys())
weights = MeritWeights(**base_weights)

@nb.cuda.jit(device=True)
def nonlinearity(delta):
    """Calculate the nonlinearity of the given delta spectrum"""
    g0 = (((delta[2] - delta[0])/2) - ((delta[1] - delta[0]))) ** 2
    gn1 = (((delta[-1] - delta[-2])) - ((delta[-1] - delta[-3])/2)) ** 2
    g1 = (((delta[3] - delta[1]) / 2) - ((delta[1] - delta[0]))) ** 2 / 4
    gn2 = (((delta[-1] - delta[-2])) - ((delta[-2] - delta[-4]) / 2)) ** 2 / 4
    err = g0 + g1 + gn2 + gn1
    for i in range(2, nwaves-2):
      err += (((delta[i + 2] - delta[i]) / 2) - ((delta[i] - delta[i - 2]) / 2)) ** 2 / 4
    return math.sqrt(err)

@nb.cuda.jit(device=True)
def merit_error(n, angles):
    tid = nb.cuda.threadIdx.x
    nb.cuda.syncthreads()
    sharedBlock = nb.cuda.shared.array(nwaves, nb.f8)
    nb.cuda.syncthreads()
    crit_count = 0
    invalid_count = 0
    mid = prism_count // 2
    midV2 = angles[mid] / 2
    beta = midV2
    for i in range(prism_count):
        if i < mid:
            beta += angles[i]
    path0 = (start - radius) / math.cos(beta)
    path1 = (start + radius) / math.cos(beta)
    offAngle = beta
    sideL = height / math.cos(offAngle)
    incident = theta0 + beta
    refracted = math.asin((1.0 / n[0, tid]) * math.sin(theta0 + beta))
    for i in range(1, prism_count + 1):
        alpha = angles[i - 1]
        incident = refracted - alpha
        if abs(incident > angle_limit):
            crit_count += 1
        
        if i <= mid:
            offAngle -= angles[i-1]
        elif i > mid+1:
            offAngle += angles[i-1]
        sideR = height / math.cos(offAngle)
        t1 = math.pi/2 - refracted*math.copysign(alpha, 1)
        t2 = math.pi - abs(alpha) - t1
        los = math.sin(t1)/math.sin(t2)
        if alpha > 0:
            path0 *= los
            path1 *= los
        else:
            path0 = sideR - (sideL - path0)*los
            path1 = sideR - (sideL - path1)*los
        if 0 > path0 or path0 > sideR:
            invalid_count += 1
        if 0 > path1 or path1 > sideR:
            invalid_count += 1
        sideL = sideR
            
        if i < prism_count:
            refracted = math.asin((n[i - 1, tid] / n[i, tid]) * math.sin(incident))
        else:
            refracted = math.asin(n[i - 1, tid] * math.sin(incident))
    delta = theta0 - (refracted + offAngle)
    sharedBlock[tid] = delta
    nb.cuda.syncthreads()
    
    nan_count = 0
    minVal, maxVal = sharedBlock[0], sharedBlock[0]
    for d in sharedBlock:
        if math.isnan(d):
            nan_count += 1
        else:
            minVal = min(minVal, d)
            maxVal = max(maxVal, d)
    nb.cuda.syncthreads()
    if nan_count > 0:
        return weights.tir * nan_count
    too_thin_err = 0
    for a in angles:
        t = abs(a)
        if t <= math.pi/180.0:
            too_thin_err += (t - 1.0)**2
    merit_err = weights.crit_angle * crit_count / (prism_count * nwaves) \
                + weights.valid * invalid_count / prism_count \
                + weights.thin * too_thin_err / prism_count \
                + weights.deviation * (sharedBlock[nwaves // 2] - deltaC_target) ** 2 \
                + weights.dispersion * ((maxVal - minVal) - deltaT_target) ** 2 \
                + weights.linearity * nonlinearity(sharedBlock)
    nb.cuda.syncthreads()
    return merit_err


@nb.cuda.jit(device=True)
def minimizer(n):
    sim = nb.cuda.local.array((surf_count, glass_count), nb.f8)
    fsim = nb.cuda.local.array(surf_count, nb.f8)
    xbar = nb.cuda.local.array(glass_count, nb.f8)
    xr = nb.cuda.local.array(glass_count, nb.f8)
    xe = nb.cuda.local.array(glass_count, nb.f8)
    xc = nb.cuda.local.array(glass_count, nb.f8)
    xcc = nb.cuda.local.array(glass_count, nb.f8)
    
    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    xatol = 1e-4
    fatol = 1e-4
    
    for i in range(glass_count):
        sim[0, i] = 90
    for k in range(glass_count):
        for i in range(glass_count):
            sim[k + 1, i] = 90*(-1)**i
        if sim[k + 1, k] != 0:
            sim[k + 1, k] *= (1 + nonzdelt)
        else:
            sim[k + 1, k] = zdelt
    
    maxiter = glass_count * 200
    maxfun = glass_count * 200
    
    for k in range(glass_count + 1):
        fsim[k] = merit_error(n, sim[k])
    ncalls = glass_count

    for i in range(glass_count):
        x, y, j = fsim[i], sim[i], i-1
        while j >= 0 and fsim[j] > x:
            fsim[j+1] = fsim[j]
            for k in range(glass_count):
                sim[j+1, k] = sim[j, k]
            j -= 1
        fsim[j+1] = x
        for k in range(glass_count):
            sim[j+1, k] = y[k]
    
    iterations = 1
    while ncalls < maxfun and iterations < maxiter:
        # Tolerence Check
        maxF, maxS = 0, 0
        for i in range(1, surf_count):
            maxF = max(maxF, abs(fsim[0] - fsim[i]))
            for j in range(glass_count):
                maxS = max(maxS, abs(sim[i, j] - sim[0, j]))
        if maxS <= xatol and maxF <= fatol:
            break

        # xbar = np.add.reduce(sim[:-1], 0) / glass_count
        xbar[:] = 0
        for s in range(glass_count):
            for i in range(glass_count):
                xbar[i] += sim[:-1][s, i]
        for i in range(glass_count):
            xbar[i] /= glass_count

        for i in range(glass_count):
            xr[i] = (1 + rho) * xbar[i] - rho * sim[-1, i]
        fxr = merit_error(n, xr)
        ncalls += 1
        doshrink = False

        if fxr < fsim[0]:
            for i in range(glass_count):
                xe[i] = (1 + rho * chi) * xbar[i] - rho * chi * sim[-1, i]
            fxe = merit_error(n, xe)
            ncalls += 1
            if fxe < fxr:
                for i in range(glass_count):
                    sim[-1, i] = xe[i]
                fsim[-1] = fxe
            else:
                for i in range(glass_count):
                    sim[-1, i] = xr[i]
                fsim[-1] = fxr
        else:
            if fxr < fsim[-2]:
                for i in range(glass_count):
                    sim[-1, i] = xr[i]
                fsim[-1] = fxr
            else:
                # Perform contraction
                if fxr < fsim[-1]:
                    for i in range(glass_count):
                        xc[i] = (1 + psi * rho) * xbar[i] - psi * rho * sim[-1, i]
                    fxc = merit_error(n, xc)
                    ncalls += 1
                    if fxc <= fxr:
                        for i in range(glass_count):
                            sim[-1, i] = xc[i]
                        fsim[-1] = fxc
                    else:
                        doshrink = True
                else:
                    # Perform an inside contraction
                    for i in range(glass_count):
                        xcc[i] = (1 - psi) * xbar[i] + psi * sim[-1, i]
                    fxcc = merit_error(n, xcc)
                    ncalls += 1
                    if fxcc < fsim[-1]:
                        for i in range(glass_count):
                            sim[-1, i] = xcc[i]
                        fsim[-1] = fxcc
                    else:
                        doshrink = True
                if doshrink:
                    for j in range(1, glass_count + 1):
                        for i in range(glass_count):
                            sim[j, i] = sim[0, i] + sigma * (sim[j, i] - sim[0, i])
                        fsim[j] = merit_error(n, sim[j])
                    ncalls += glass_count
        for i in range(glass_count):
            x, y, j = fsim[i], sim[i], i-1
            while j >= 0 and fsim[j] > x:
                fsim[j+1] = fsim[j]
                for k in range(glass_count):
                    sim[j+1, k] = sim[j, k]
                j -= 1
            fsim[j+1] = x
            for k in range(glass_count):
                sim[j+1, k] = y[k]
        iterations += 1
    return sim[0], fsim[0]


@nb.cuda.jit((nb.f8[:, :], nb.f8[:]), fastmath=True)
def optimize(ns, out):
    tid = nb.cuda.threadIdx.x
    ind = nb.cuda.blockIdx.x
    tn = ns.shape[0]
    nb.cuda.syncthreads()
    n = nb.cuda.shared.array((glass_count, nwaves), nb.f8)
    for i in range(glass_count):
        n[i, tid] = ns[(ind // tn**i)%tn, tid]
    nb.cuda.syncthreads()
    xm, fx = minimizer(ns)
    if tid == 0:
        out[ind*(1+glass_count)] = fx
        for i in range(glass_count):
            out[ind*(1+glass_count)+i+1] = xm[i]


import time
from compoundprism.glasscat import read_glasscat, calc_n
w = np.linspace(500, 820, nwaves, dtype=np.float64)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
nglass = len(gcat)
glasses = np.array([calc_n(g, w) for g in gcat.values()])

d_g = nb.cuda.to_device(glasses)
d_o = nb.cuda.device_array((4*nglass**glass_count), np.float64)
print('Starting')
t1 = time.time()
optimize[nglass**glass_count, nwaves](d_g, d_o)
output = d_o.copy_to_host()
dt = time.time() - t1
print(np.min(output[::4]), dt)

