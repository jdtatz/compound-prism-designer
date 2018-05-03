import math
import numpy as np
import numba as nb
import numba.typing
import numba.typing.templates
import numba.cuda
import numba.cuda.cudaimpl
import numba.cuda.cudadecl
import numba.cuda.stubs
import llvmlite.llvmpy.core as lc


def _convert_type(t):
    if isinstance(t, nb.types.Integer):
        return lc.Type.int(t.bitwidth)
    elif isinstance(t, nb.types.Float):
        if t == nb.f4:
            return lc.Type.float()
        elif t == nb.f8:
            return lc.Type.double()
    elif t == nb.b1:
        return lc.Type.int(1)
    elif t == nb.void:
        return lc.Type.void()
    elif isinstance(t, nb.types.Tuple):
        return lc.Type.struct(tuple(map(_convert_type, t.types)))
    raise NotImplementedError


def add_nvvm_intrinsic(name, fname, signature, make_global=True):
    stub = type(name, (nb.cuda.stubs.Stub,), {'_description_': '<{}()>'.format(name)})
    llvm_signature = lc.Type.function(_convert_type(signature.return_type), list(map(_convert_type, signature.args)))

    @numba.cuda.cudaimpl.lower(stub, *signature.args)
    def lower_nvvm_intrinsic(context, builder, sig, args):
        lmod = builder.module
        func = nb.cgutils.insert_pure_function(lmod, llvm_signature, name=fname)
        return builder.call(func, args)

    @nb.cuda.cudadecl.intrinsic
    class NvvmIntrinsicTemplate(nb.typing.templates.AbstractTemplate):
        key = stub

        def generic(self, args, kws):
            return signature

    if make_global:
        nb.cuda.cudadecl.intrinsic_global(stub, nb.types.Function(NvvmIntrinsicTemplate))
    return stub

syncthreads_or = add_nvvm_intrinsic('syncthreads_or', 'llvm.nvvm.barrier0.or', nb.i4(nb.i4))
nb.cuda.syncthreads_or = syncthreads_or

laneid = add_nvvm_intrinsic('laneid', 'llvm.nvvm.read.ptx.sreg.laneid', nb.i4())
nb.cuda.laneid = laneid



class shfl_sync_intrinsic(nb.cuda.stubs.Stub):
    """https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-datamove"""
    _description_ = '<shfl_sync_intrinsic()>'

shfl_sync_llvm_sign = lc.Type.function(lc.Type.struct((lc.Type.int(32), lc.Type.int(1))), 5*[lc.Type.int(32)])


@numba.cuda.cudaimpl.lower(shfl_sync_intrinsic, nb.i4, nb.i4, nb.i4, nb.i4, nb.i4)
def lower_shfl_sync_intrinsic(context, builder, sig, args):
    fn = nb.cgutils.insert_pure_function(builder.module, shfl_sync_llvm_sign, 'llvm.nvvm.shfl.sync.i32')
    return builder.call(fn, args, tail=True)


@numba.cuda.cudaimpl.lower(shfl_sync_intrinsic, nb.i4, nb.i4, nb.f4, nb.i4, nb.i4)
def lower_shfl_sync_intrinsic(context, builder, sig, args):
    casted_args = (args[0], args[1], builder.bitcast(args[2], lc.Type.int(32)), args[3], args[4])
    fn = nb.cgutils.insert_pure_function(builder.module, shfl_sync_llvm_sign, 'llvm.nvvm.shfl.sync.i32')
    rstruct = builder.call(fn, casted_args, tail=True)
    ival, pred = builder.extract_value(rstruct, 0), builder.extract_value(rstruct, 1)
    fval = builder.bitcast(ival, lc.Type.float())
    return nb.cgutils.make_anonymous_struct(builder, (fval, pred))



@nb.cuda.cudadecl.intrinsic
class CudaShflSyncTemplate(nb.typing.templates.AbstractTemplate):
    key = shfl_sync_intrinsic
    def generic(self, args, kws):
        vty = args[2]
        if vty == nb.i4 or vty == nb.f4:
            return nb.types.Tuple((vty, nb.boolean))(nb.i4, nb.i4, vty, nb.i4, nb.i4)

nb.cuda.cudadecl.intrinsic_global(shfl_sync_intrinsic, nb.types.Function(CudaShflSyncTemplate))


@nb.cuda.jit(device=True)
def shfl_sync(mask, value, src_lane):
    return shfl_sync_intrinsic(mask, 0, value, src_lane, 0x1f)[0]
nb.cuda.shfl_sync = shfl_sync


@nb.cuda.jit(device=True)
def shfl_up_sync(mask, value, delta):
    return shfl_sync_intrinsic(mask, 1, value, delta, 0)[0]
nb.cuda.shfl_up_sync = shfl_up_sync


@nb.cuda.jit(device=True)
def shfl_down_sync(mask, value, delta):
    return shfl_sync_intrinsic(mask, 2, value, delta, 0x1f)[0]
nb.cuda.shfl_down_sync = shfl_down_sync


@nb.cuda.jit(device=True)
def shfl_xor_sync(mask, value, lane_mask):
    return shfl_sync_intrinsic(mask, 3, value, lane_mask, 0x1f)[0]
nb.cuda.shfl_xor_sync = shfl_xor_sync
