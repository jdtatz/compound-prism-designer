#!/usr/bin/env python3.6
import os
os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
import numpy as np
import numba as nb
import numba.typing
import numba.typing.templates
import numba.cuda
import numba.cuda.cudaimpl
import numba.cuda.cudadecl
import numba.cuda.stubs
from llvmlite.llvmpy.core import Type
import math
import operator
from collections import namedtuple, OrderedDict

count = 4
count_p1 = count+1
start = 6
radius = 5
height = 20
theta0 = 0
angle_limit = 65.0 * math.pi / 180.0
nwaves = 100
deltaC_target = 0
deltaT_target = 45 * math.pi / 180
inital_angles = np.full((count,), math.pi/2, np.float32)
inital_angles[1::2] *= -1
base_weights = OrderedDict(tir=50.0, valid=5.0, crit_angle=1.0, thin=2.5, deviation=5.0, dispersion=30.0,
                           linearity=1000.0)
MeritWeights = namedtuple('MeritWeights', base_weights.keys())
weights = MeritWeights(**base_weights)


class shfl(nb.cuda.stubs.Stub):
    _description_ = '<shfl>'
    
    class idx(nb.cuda.stubs.Stub):
        """
        shfl from tid
        """
    
    class up(nb.cuda.stubs.Stub):
        """
        shfl from above
        """
        
    class down(nb.cuda.stubs.Stub):
        """
        shfl from below
        """
        
    class xor(nb.cuda.stubs.Stub):
        """
        shfl from xor
        """

class shfl_idx(nb.cuda.stubs.Stub):
    """
    shfl from tid
    """
    _description_ = '<shfl>'

@numba.cuda.cudaimpl.lower(shfl_idx, nb.i4, nb.i4, nb.i4)
def lower_shfl_idx(context, builder, sig, args):
    print('lower', sig, args)
    fname = 'llvm.nvvm.shfl.sync.idx.i32'
    lmod = builder.module
    fnty = Type.function(Type.int(32), (Type.int(32), Type.int(32), Type.int(32)))
    func = lmod.get_or_insert_function(fnty, name=fname)
    ret =  builder.call(func, args)
    print(lmod)
    return ret


@nb.cuda.cudadecl.intrinsic
class Cuda_shfl_idx(nb.typing.templates.AbstractTemplate):
    key = shfl_idx
    
    def generic(self, args, kws):
        print('resolve generic', args, kws)
        return nb.i4(nb.i4, nb.i4, nb.i4)

nb.cuda.cudadecl.intrinsic_global(shfl_idx, nb.types.Function(Cuda_shfl_idx))


class syncthreads_or(nb.cuda.stubs.Stub):
    _description_ = '<syncthreads_or()>'
@numba.cuda.cudaimpl.lower(syncthreads_or, nb.i4)
def lower_syncthreads_or(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.or'
    lmod = builder.module
    fnty = Type.function(Type.int(32), (Type.int(32),))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, args)
@nb.cuda.cudadecl.intrinsic
class Cuda_syncthreads_or(nb.typing.templates.AbstractTemplate):
    key = syncthreads_or
    def generic(self, args, kws):
        return nb.i4(nb.i4,)
nb.cuda.cudadecl.intrinsic_global(syncthreads_or, nb.types.Function(Cuda_syncthreads_or))


class syncthreads_and(nb.cuda.stubs.Stub):
    _description_ = '<syncthreads_and()>'
@numba.cuda.cudaimpl.lower(syncthreads_and, nb.i4)
def lower_syncthreads_and(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.and'
    lmod = builder.module
    fnty = Type.function(Type.int(32), (Type.int(32),))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, args)
@nb.cuda.cudadecl.intrinsic
class Cuda_syncthreads_and(nb.typing.templates.AbstractTemplate):
    key = syncthreads_and
    def generic(self, args, kws):
        return nb.i4(nb.i4,)
nb.cuda.cudadecl.intrinsic_global(syncthreads_and, nb.types.Function(Cuda_syncthreads_and))


class syncthreads_popc(nb.cuda.stubs.Stub):
    _description_ = '<syncthreads_popc()>'
@numba.cuda.cudaimpl.lower(syncthreads_popc, nb.i4)
def lower_syncthreads_popc(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.popc'
    lmod = builder.module
    fnty = Type.function(Type.int(32), (Type.int(32),))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, args)
@nb.cuda.cudadecl.intrinsic
class Cuda_syncthreads_popc(nb.typing.templates.AbstractTemplate):
    key = syncthreads_popc
    def generic(self, args, kws):
        return nb.i4(nb.i4,)
nb.cuda.cudadecl.intrinsic_global(syncthreads_popc, nb.types.Function(Cuda_syncthreads_popc))

"""
@nb.cuda.jit((nb.i4[:], ))
def test(arr):
    i = nb.cuda.threadIdx.x
    arr[i] = shfl_idx(syncthreads_popc(1), 0, 32)

testA = np.empty(64, np.int32)
test[2, 32](testA)
"""


@nb.cuda.jit(device=True)
def ilog2(n):
    return 0 if n < 2 else 1 + ilog2(n >> 1)


@nb.cuda.jit(device=True)
def isPow2(num):
    return not (num & (num - 1))


@nb.cuda.jit(device=True)
def roundToPow2(num):
    return 1 << ilog2(num)


def create_fold_func(func):
    @nb.cuda.jit(device=True)
    def warpFold(value, foldLen):
        i = foldLen / 2
        while i > 0:
            value = func(val, nb.cuda.shfl.xor(value, i, foldLen))
            i >>= 2
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
                value = partialWarpFold(value, warpID, foldLen % warpSize);
            if warpID == 0:
                sharedArr[tID // warpSize] = value
            nb.cuda.syncthreads()
            faster = (warpCount - 1) > (1 + log2(warpCountPow2) + (0 if isPow2(warpCount) else 1))
            allWarpsFull = foldLen % warpSize == 0
            partialBigEnough = foldLen % warpSize >= warpCountPow2
            
            if faster and (allWarpsFull or partialBigEnough or withinFullWarp):
              value = sharedArr[warpID % warpCount];
              if not isPow2(warpCount) and warpID < warpCount - warpCountPow2:
                value = func(value, sharedArr[warpID + warpCountPow2])
              value = warpFold(value, warpCountPow2);
              value = nb.cuda.shfl.idx(value, 0);
            else:
              value = sharedArr[0];
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


@nb.cuda.jit(device=True)
def merit_error(n, angles):
    tid = nb.cuda.threadIdx.x
    nb.cuda.syncthreads()
    sharedBlock = nb.cuda.shared.array(nwaves, nb.f4)
    crit_violation = False
    invalidity = False
    mid = count // 2
    beta = angles[mid] / 2
    for i in range(count):
        if i < mid:
            beta += angles[i]
    path0 = (start - radius) / math.cos(beta)
    path1 = (start + radius) / math.cos(beta)
    offAngle = beta
    sideL = height / math.cos(offAngle)
    incident = theta0 + beta
    refracted = math.asin((1.0 / n[0, tid]) * math.sin(incident))
    for i in range(1, count + 1):
        alpha = angles[i - 1]
        incident = refracted - alpha
        crit_violation |= abs(incident) > angle_limit

        if i <= mid:
            offAngle -= angles[i - 1]
        elif i > mid + 1:
            offAngle += angles[i - 1]
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
        invalidity |= 0 > path0 or path0 > sideR or 0 > path1 or path1 > sideR
        sideL = sideR
            
        if i < count:
            refracted = math.asin((n[i - 1, tid] / n[i, tid]) * math.sin(incident))
        else:
            refracted = math.asin(n[i - 1, tid] * math.sin(incident))
    delta = theta0 - (refracted + offAngle)
    sharedBlock[tid] = delta
    nan_count = syncthreads_popc(math.isnan(delta))
    if nan_count > 0:
        return weights.tir * nan_count
    crit_count = syncthreads_popc(crit_violation)
    invalid_count = syncthreads_popc(invalidity)
    
    minVal, maxVal = sharedBlock[0], sharedBlock[0]
    for i in range(nwaves):
        minVal = min(minVal, sharedBlock[i])
        maxVal = max(maxVal, sharedBlock[i])

    too_thin_err = 0
    for a in angles:
        t = abs(a)
        if t <= math.pi / 180.0:
            too_thin_err += (t - 1.0) ** 2
    merit_err = weights.crit_angle * crit_count / (count * nwaves) \
                + weights.valid * invalid_count / count \
                + weights.thin * too_thin_err / count \
                + weights.deviation * (sharedBlock[nwaves // 2] - deltaC_target) ** 2 \
                + weights.dispersion * ((maxVal - minVal) - deltaT_target) ** 2 \
                + weights.linearity * nonlinearity(sharedBlock)
    return merit_err


@nb.cuda.jit(device=True)
def minimizer(n, xmin):
    sim = nb.cuda.local.array((count_p1, count), nb.f4)
    fsim = nb.cuda.local.array(count_p1, nb.f4)
    xr = nb.cuda.local.array(count, nb.f4)
    xe = nb.cuda.local.array(count, nb.f4)
    xc = nb.cuda.local.array(count, nb.f4)
    xcc = nb.cuda.local.array(count, nb.f4)

    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    xatol = 1e-4
    fatol = 1e-4

    for i in range(count):
        sim[0, i] = xmin[i]
    for k in range(count):
        for i in range(count):
            sim[k + 1, i] = xmin[i]
        sim[k + 1, k] *= (1 + nonzdelt)

    maxiter = count * 200
    maxfun = count * 200

    for k in range(count + 1):
        fsim[k] = merit_error(n, sim[k])
    ncalls = count

    for i in range(count+1):
        x = fsim[i]
        for k in range(count):
            xmin[k] = sim[i, k]
        j = i - 1
        while j >= 0 and fsim[j] > x:
            fsim[j + 1] = fsim[j]
            for k in range(count):
                sim[j + 1, k] = sim[j, k]
            j -= 1
        fsim[j + 1] = x
        for k in range(count):
            sim[j + 1, k] = xmin[k]

    iterations = 1
    while ncalls < maxfun and iterations < maxiter:
        # Tolerence Check
        maxF, maxS = 0, 0
        for i in range(1, count+1):
            maxF = max(maxF, abs(fsim[0] - fsim[i]))
            for j in range(count):
                maxS = max(maxS, abs(sim[0, j] - sim[i, j]))
        if maxS <= xatol and maxF <= fatol:
            break

        for i in range(count):
            xbar = 0
            for s in range(count):
                xbar += sim[s, i]
            xr[i] = (1 + rho) * xbar / count - rho * sim[-1, i]
        fxr = merit_error(n, xr)
        ncalls += 1
        doshrink = False

        if fxr < fsim[0]:
            for i in range(count):
                xbar = 0
                for s in range(count):
                    xbar += sim[s, i]
                xe[i] = (1 + rho * chi) * xbar / count - rho * chi * sim[-1, i]
            fxe = merit_error(n, xe)
            ncalls += 1
            if fxe < fxr:
                for i in range(count):
                    sim[-1, i] = xe[i]
                fsim[-1] = fxe
            else:
                for i in range(count):
                    sim[-1, i] = xr[i]
                fsim[-1] = fxr
        elif fxr < fsim[-2]:
            for i in range(count):
                sim[-1, i] = xr[i]
            fsim[-1] = fxr
        else:
            # Perform contraction
            if fxr < fsim[-1]:
                for i in range(count):
                    xbar = 0
                    for s in range(count):
                        xbar += sim[s, i]
                    xc[i] = (1 + psi * rho) * xbar / count - psi * rho * sim[-1, i]
                fxc = merit_error(n, xc)
                ncalls += 1
                if fxc <= fxr:
                    for i in range(count):
                        sim[-1, i] = xc[i]
                    fsim[-1] = fxc
                else:
                    doshrink = True
            else:
                # Perform an inside contraction
                for i in range(count):
                    xbar = 0
                    for s in range(count):
                        xbar += sim[s, i]
                    xcc[i] = (1 - psi) * xbar / count + psi * sim[-1, i]
                fxcc = merit_error(n, xcc)
                ncalls += 1
                if fxcc < fsim[-1]:
                    for i in range(count):
                        sim[-1, i] = xcc[i]
                    fsim[-1] = fxcc
                else:
                    doshrink = True
            if doshrink:
                for j in range(1, count + 1):
                    for i in range(count):
                        sim[j, i] = sim[0, i] + sigma * (sim[j, i] - sim[0, i])
                    fsim[j] = merit_error(n, sim[j])
                ncalls += count
        for i in range(count+1):
            x = fsim[i]
            for k in range(count):
                xmin[k] = sim[i, k]
            j = i - 1
            while j >= 0 and fsim[j] > x:
                fsim[j + 1] = fsim[j]
                for k in range(count):
                    sim[j + 1, k] = sim[j, k]
                j -= 1
            fsim[j + 1] = x
            for k in range(count):
                sim[j + 1, k] = xmin[k]
        iterations += 1
    for i in range(count):
        xmin[i] = sim[0, i]
    return fsim[0]


@nb.cuda.jit((nb.f4[:, :], nb.f4[:]), fastmath=False)
def optimize(ns, out):
    tid = nb.cuda.threadIdx.x
    ind = nb.cuda.blockIdx.x
    tn = ns.shape[0]
    nb.cuda.syncthreads()
    n = nb.cuda.shared.array((count, nwaves), nb.f4)
    for i in range(count):
        n[i, tid] = ns[(ind // (tn ** i)) % tn, tid]
    nb.cuda.syncthreads()
    cinital_angles = nb.cuda.const.array_like(inital_angles)
    xmin = nb.cuda.local.array(count, nb.f4)
    for i in range(count):
        xmin[i] = cinital_angles[i]
    fx = minimizer(n, xmin)
    if tid == 0:
        out[ind * (1 + count)] = fx
        for i in range(count):
            out[ind * (1 + count) + i + 1] = xmin[i]


import time
from compoundprism.glasscat import read_glasscat, calc_n

w = np.linspace(500, 820, nwaves, dtype=np.float64)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
nglass = len(gcat)
names = list(gcat.keys())
glasses = np.array([calc_n(gcat[name], w) for name in names], np.float32)
nb.cuda.profile_start()
d_out = nb.cuda.device_array(((1 + count) * nglass ** count), np.float32)
print('Starting')
t1 = time.time()
optimize[nglass ** count, nwaves](glasses, d_out)
output = d_out.copy_to_host()
dt = time.time() - t1
mi, = np.where(output[::(count+1)] == np.min(output[::(count+1)]))
gs = [names[((mi[0]) // (nglass ** i)) % nglass] for i in range(count)]
mi = (count+1) * mi[0]
print(output[mi], *gs, *(output[mi+1:mi+(count+1)] * 180 / np.pi))
print(dt, 's')
nb.cuda.profile_stop()
"""
Best: ('N-SF66', 'N-LAF34', 75.6237857668015, -112.94657682536811, 1.8423067282937657, -17.32103143116662, 37.09139264883297, 50.430053463384056)
"""
