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

import numba.cgutils
import numba.cuda.cudaimpl
import numba.cuda.cudadecl
import numba.typing
import llvmlite.ir as ir

warp_size = np.int32(32)
mask = np.uint32(0xffffffff)
packing = np.int32(0x1f)
rtype = nb.f4


@numba.cuda.cudaimpl.lower('+', nb.types.BaseTuple, nb.types.BaseTuple)
def lower_tuple_add(context, builder, sig, args):
    t1 = nb.cgutils.unpack_tuple(builder, args[0])
    t2 = nb.cgutils.unpack_tuple(builder, args[1])
    to = [builder.fadd(a, b) for a, b in zip(t1, t2)]
    ret = nb.cgutils.pack_array(builder, to)
    return ret


@nb.cuda.cudadecl.intrinsic
class CudaTupleAdd(nb.typing.templates.AbstractTemplate):
    key = '+'

    def generic(self, args, kws):
        if isinstance(args[0], nb.types.BaseTuple) and isinstance(args[1], nb.types.BaseTuple)\
                and args[0].count == args[1].count and args[0].dtype == args[1].dtype:
            return args[0](*args)


@numba.cuda.cudaimpl.lower('-', nb.types.BaseTuple, nb.types.BaseTuple)
def lower_tuple_sub(context, builder, sig, args):
    t1 = nb.cgutils.unpack_tuple(builder, args[0])
    t2 = nb.cgutils.unpack_tuple(builder, args[1])
    to = [builder.fsub(a, b) for a, b in zip(t1, t2)]
    ret = nb.cgutils.pack_array(builder, to)
    return ret


@numba.cuda.cudaimpl.lower('-', nb.types.BaseTuple)
def lower_tuple_neg(context, builder, sig, args):
    t = nb.cgutils.unpack_tuple(builder, args[0])
    to = [builder.fmul(ir.values.Constant(sig.return_type.dtype.type, -1), a) for a in t]
    ret = nb.cgutils.pack_array(builder, to)
    return ret


@nb.cuda.cudadecl.intrinsic
class CudaTupleSub(nb.typing.templates.AbstractTemplate):
    key = '-'

    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], nb.types.BaseTuple):
            return args[0](args[0])
        if len(args) == 2 and isinstance(args[0], nb.types.BaseTuple) and isinstance(args[1], nb.types.BaseTuple)\
                and args[0].count == args[1].count and args[0].dtype == args[1].dtype:
            return args[0](*args)


@numba.cuda.cudaimpl.lower('*', nb.types.BaseTuple, nb.types.BaseTuple)
def lower_tuple_mul(context, builder, sig, args):
    t1 = nb.cgutils.unpack_tuple(builder, args[0])
    t2 = nb.cgutils.unpack_tuple(builder, args[1])
    to = [builder.fmul(a, b) for a, b in zip(t1, t2)]
    ret = nb.cgutils.pack_array(builder, to)
    return ret


@numba.cuda.cudaimpl.lower('*', nb.types.BaseTuple, nb.types.Number)
def lower_tuple_muls(context, builder, sig, args):
    t = nb.cgutils.unpack_tuple(builder, args[0])
    v = args[1]
    to = [builder.fmul(a, v) for a in t]
    ret = nb.cgutils.pack_array(builder, to)
    return ret


@nb.cuda.cudadecl.intrinsic
class CudaTupleMul(nb.typing.templates.AbstractTemplate):
    key = '*'

    def generic(self, args, kws):
        if isinstance(args[0], nb.types.BaseTuple) and isinstance(args[1], nb.types.BaseTuple)\
                and args[0].count == args[1].count and args[0].dtype == args[1].dtype:
            return args[0](*args)
        elif isinstance(args[0], nb.types.BaseTuple) and args[1] == args[0].dtype:
            return args[0](args[0], args[1])


@numba.cuda.cudaimpl.lower('/', nb.types.BaseTuple, nb.types.Number)
def lower_tuple_divs(context, builder, sig, args):
    t = nb.cgutils.unpack_tuple(builder, args[0])
    v = args[1]
    to = [builder.fdiv(a, v) for a in t]
    ret = nb.cgutils.pack_array(builder, to)
    return ret


@nb.cuda.cudadecl.intrinsic
class CudaTupleDiv(nb.typing.templates.AbstractTemplate):
    key = '/'

    def generic(self, args, kws):
        if isinstance(args[0], nb.types.BaseTuple) and args[1] == args[0].dtype:
            return args[0](args[0], args[1])


@numba.cuda.cudaimpl.lower('@', nb.types.BaseTuple, nb.types.BaseTuple)
def lower_tuple_dot(context, builder, sig, args):
    t1 = nb.cgutils.unpack_tuple(builder, args[0])
    t2 = nb.cgutils.unpack_tuple(builder, args[1])
    to = [builder.fmul(a, b) for a, b in zip(t1, t2)]
    ret = to[0]
    for i in range(1, len(to)):
        ret = builder.fadd(ret, to[i])
    return ret


@nb.cuda.cudadecl.intrinsic
class CudaTupleDot(nb.typing.templates.AbstractTemplate):
    key = '@'

    def generic(self, args, kws):
        if isinstance(args[0], nb.types.BaseTuple) and isinstance(args[1], nb.types.BaseTuple)\
                and args[0].count == args[1].count and args[0].dtype == args[1].dtype:
            return args[0].dtype(*args)


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

    out_dtype = np.dtype([('value', 'f4'), ('index', 'i4'), ('curvature', 'f4'), ('distance', 'f4'), ('angles', 'f4', surface_count)])
    out_type = nb.from_dtype(out_dtype)

    @nb.cuda.jit(device=True)
    def nonlinearity(vals):
        """Calculate the nonlinearity of the given delta spectrum"""
        tid = nb.cuda.threadIdx.x
        val = vals[tid]
        if tid == 0:
            err = (2 * vals[2] + 2 * val - 4 * vals[1])
        elif tid == nwaves - 1:
            err = (2 * val - 4 * vals[nwaves - 2] + 2 * vals[nwaves - 3])
        elif tid == 1:
            err = (vals[3] - 3 * val + 2 * vals[0])
        elif tid == nwaves - 2:
            err = (2 * vals[nwaves - 1] - 3 * val + vals[nwaves - 4])
        else:
            err = (vals[tid + 2] + vals[tid - 2] - 2 * val)
        return math.sqrt(reduce(operator.add, np.float32(err**2), nwaves)) / 4

    @nb.cuda.jit(device=True)
    def merit_error(n, params, index, nglass):
        curvature, distance, angles = params[0], params[1], params[2:]
        tix = nb.cuda.threadIdx.x
        tiy = nb.cuda.threadIdx.y
        # nwaves = nb.cuda.blockDim.x
        shared_data = nb.cuda.shared.array(4, nb.f4)
        shared_pos = nb.cuda.shared.array((3, nwaves), nb.f4)
        # Initial Surface
        n2 = n[index % nglass, tix]
        r = np.float32(1) / n2
        # Rotation of (-1, 0) by angle[0] CW
        norm = -math.cos(angles[0]), -math.sin(angles[0])
        size = abs(norm[1] / norm[0])
        ray_dir = math.cos(config.theta0), math.sin(config.theta0)
        start = config.start if tiy == 0 else \
            ((config.start + config.radius) if tiy == 1 else (config.start - config.radius))
        ray_path = size - (np.float32(1) - start) * abs(norm[1] / norm[0]), start
        # Snell's Law
        ci = -(ray_dir @ norm)
        cr_sq = np.float32(1) - r ** 2 * (np.float32(1) - ci ** 2)
        if nb.cuda.syncthreads_or(cr_sq < 0):
            return
        cr = math.sqrt(cr_sq)
        ray_dir = ray_dir * r + norm * (r * ci - cr)
        # Surface Transmittance / Fresnel Equation
        fresnel_rs = ((ci - n2 * cr) / (ci + n2 * cr))
        fresnel_rp = ((cr - n2 * ci) / (cr + n2 * ci))
        transmittance = np.float32(1) - (fresnel_rs ** 2 + fresnel_rp ** 2) / np.float32(2)
        # Inner Surfaces
        for i in range(1, prism_count):
            n1 = n2
            n2 = n[(index // (nglass**i)) % nglass, tix]
            r = n1 / n2
            # Rotation of (-1, 0) by angle[i] CCW
            norm = -math.cos(angles[i]), -math.sin(angles[i])
            size += abs(norm[1] / norm[0])
            ci = -(ray_dir @ norm)
            # Line-Plane Intersection
            vertex = size, np.float32((i + 1) % 2)
            d = ((ray_path - vertex) @ norm) / ci
            ray_path = ray_path + ray_dir * d
            if nb.cuda.syncthreads_or(ray_path[1] <= 0 or ray_path[1] >= 1):
                return
            # Snell's Law
            cr_sq = np.float32(1) - r ** 2 * (np.float32(1) - ci ** 2)
            if nb.cuda.syncthreads_or(cr_sq < 0):
                return
            cr = math.sqrt(cr_sq)
            ray_dir = ray_dir * r + norm * (r * ci - cr)
            # Surface Transmittance / Fresnel Equation
            fresnel_rs = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))
            fresnel_rp = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))
            transmittance *= np.float32(1) - (fresnel_rs ** 2 + fresnel_rp ** 2) / np.float32(2)
        # Last / Convex Surface
        n1 = n2
        r = n1
        norm = -math.cos(angles[prism_count]), -math.sin(angles[prism_count])
        diff = abs(norm[1] / norm[0])
        size += diff
        if size >= config.max_size:
            return
        diameter = np.float32(1) / abs(norm[0])
        midpt = size - diff / np.float32(2), np.float32(0.5)
        # Line-Sphere Intersection
        lens_radius = diameter / (np.float32(2) * curvature)
        rs = math.sqrt(lens_radius ** 2 - diameter ** 2 / np.float32(4))
        c = midpt + norm * rs
        delta = ray_path - c
        under = (ray_dir @ delta)**2 - (delta[0] ** 2 + delta[1] ** 2) + lens_radius ** 2
        if nb.cuda.syncthreads_or(under <= 0):
            return
        d = -(ray_dir @ delta) + math.sqrt(under)
        ray_path = ray_path + ray_dir * d
        rd = ray_path - midpt
        if nb.cuda.syncthreads_or(rd @ rd > (diameter ** 2 / np.float32(4))):
            return
        snorm = (c - ray_path) / lens_radius
        # Snell's Law
        ci = -(ray_dir @ snorm)
        cr_sq = np.float32(1) - r ** 2 * (np.float32(1) - ci ** 2)
        if nb.cuda.syncthreads_or(cr_sq < 0):
            return
        cr = math.sqrt(cr_sq)
        ray_dir = ray_dir * r + snorm * (r * ci - cr)
        # Surface Transmittance / Fresnel Equation
        fresnel_rs = (n1 * ci - cr) / (n1 * ci + cr)
        fresnel_rp = (n1 * cr - ci) / (n1 * cr + ci)
        transmittance *= np.float32(1) - (fresnel_rs ** 2 + fresnel_rp ** 2) / 2
        # Spectrometer
        if tix == nwaves // 2 and tiy == 0:
            shared_data[0] = ray_path[0]
            shared_data[1] = ray_path[1]
            shared_data[2] = ray_dir[0]
            shared_data[3] = ray_dir[1]
        nb.cuda.syncthreads()
        dist = distance * (config.max_size - shared_data[0])
        norm = -shared_data[2], -shared_data[3]
        vertex = (shared_data[0], shared_data[1]) - norm * dist
        ci = -(ray_dir @ norm)
        d = ((ray_path - vertex) @ norm) / ci
        end = ray_path + ray_dir * d
        vdiff = end - vertex
        spec_pos = math.copysign(math.sqrt(vdiff @ vdiff), vdiff[1])
        shared_pos[tiy, tix] = spec_pos
        if nb.cuda.syncthreads_or(abs(spec_pos) >= config.sheight / np.float32(2)):
            return
        nonlin = nonlinearity(shared_pos[0])
        mean_transmittance = reduce(operator.add, transmittance, nwaves) / np.float32(nwaves)  # implicit sync
        deviation = abs(shared_data[3])
        dispersion = abs(shared_pos[0, nwaves-1] - shared_pos[0, 0]) / config.sheight
        spot_size = abs(shared_pos[1, tix] - shared_pos[2, tix])
        mean_spot_size = reduce(operator.add, spot_size, nwaves) / np.float32(nwaves)  # implicit sync
        nb.cuda.syncthreads()
        merit_err = config.weight_deviation * deviation\
                    + config.weight_dispersion * (np.float32(1) - dispersion) \
                    + config.weight_linearity * nonlin \
                    + config.weight_transmittance * (np.float32(1) - mean_transmittance) \
                    + config.weight_spot * mean_spot_size
        return merit_err

    @nb.cuda.jit(device=True)
    def random_search(n, index, nglass, rng):
        tid = nb.cuda.threadIdx.x + nb.cuda.threadIdx.y * nb.cuda.blockDim.x
        isodd = np.float32(1) if (tid % 2 == 1 or tid < 2) else np.float32(-1)
        lbound, ubound = (config.lower_bound, config.upper_bound) if tid > 1 else (np.float32(0), np.float32(1))
        rid = nb.cuda.blockIdx.x * param_count + tid
        best = np.float32(0)
        bestVal = np.float32(0)
        found = False
        trial = nb.cuda.shared.array(param_count, nb.f4)
        if tid < param_count:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, rid)
            angle = lbound + rand * (ubound - lbound)
            trial[tid] = best = math.copysign(angle, isodd)
        nb.cuda.syncthreads()
        trialVal = merit_error(n, trial, index, nglass)
        if tid < param_count and trialVal is not None:
            bestVal = trialVal
            found = True
        for rs in range(steps):
            if tid < param_count:
                xi = nb.cuda.random.xoroshiro128p_normal_float32(rng, rid)
                sphere = xi / math.sqrt(reduce(operator.add, np.float32(xi**2), param_count))
                test = best + sphere * dr * math.exp(-np.float32(rs) * factor)
                trial[tid] = math.copysign(max(min(abs(test), ubound), lbound), isodd)
            nb.cuda.syncthreads()
            trialVal = merit_error(n, trial, index, nglass)
            if tid < param_count and trialVal is not None and (not found or trialVal < bestVal):
                bestVal = trialVal
                best = trial[tid]
                found = True
        return found, best, bestVal

    @nb.cuda.jit((nb.f4[:, :], out_type[:], nb.i8, nb.i8, nb.cuda.random.xoroshiro128p_type[:]), fastmath=False)
    def optimize(ns, out, start, stop, rng):
        tid = nb.cuda.threadIdx.x + nb.cuda.threadIdx.y * nb.cuda.blockDim.x
        bid = nb.cuda.blockIdx.x
        bcount = nb.cuda.gridDim.x
        bestVal = np.float32(np.inf)
        nglass = ns.shape[0]
        for index in range(start + bid, stop, bcount):
            valid, xmin, fx = random_search(ns, index, nglass, rng)
            if tid < param_count and valid and fx < bestVal:
                bestVal = fx
                if tid == 0:
                    out[bid].value = fx
                    out[bid].index = index
                    out[bid].curvature = xmin
                elif tid == 1:
                    out[bid].distance = xmin
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

value, ind, curvature, distance, angles = output[np.argmin(output['value'])]
gs = [names[(ind // (nglass**i)) % nglass] for i in range(prism_count)]
print(f"error: {value}, curvature: {curvature}, distance: {distance}, glasses: {gs}, angles {np.rad2deg(angles)}")
print(dt, 's')

ns = np.stack(calc_n(gcat[name], w) for name in gs).astype(np.float32)
ret = describe(ns, angles, curvature, distance, config)
if isinstance(ret, int):
    print('Failed at interface', ret)
else:
    err, NL, dispersion, deviation, size, spec_pos, transm = ret
    print(f"error: {err}\nNL: {NL}\nsize: {size}\nspectrometer position: {spec_pos}\nT: {transm * 100}")
