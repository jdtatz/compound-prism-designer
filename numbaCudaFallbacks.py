import numpy as np
import numba as nb
import numba.typing
import numba.typing.templates
import numba.cuda
import numba.cuda.cudaimpl
import numba.cuda.cudadecl
import numba.cuda.stubs
import llvmlite.llvmpy.core as lc


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


@numba.cuda.cudaimpl.lower(shfl.idx, nb.i4, nb.i4, nb.i4)
def lower_shfl_idx(context, builder, sig, args):
    fname = 'llvm.nvvm.shfl.idx.i32'
    lmod = builder.module
    fnty = lc.Type.function(lc.Type.int(32), (lc.Type.int(32), lc.Type.int(32), lc.Type.int(32)))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, args)


@nb.cuda.cudadecl.intrinsic
class Cuda_shfl_idx(nb.typing.templates.AbstractTemplate):
    key = shfl.idx

    def generic(self, args, kws):
        return nb.i4(nb.i4, nb.i4, nb.i4)


@nb.cuda.cudadecl.intrinsic_attr
class Cuda_shfls(nb.typing.templates.AttributeTemplate):
    key = nb.types.Module(shfl)

    def resolve_idx(self, mod):
        return nb.types.Function(Cuda_shfl_idx)


nb.cuda.cudadecl.intrinsic_global(shfl, nb.types.Module(shfl))


class syncthreads_or(nb.cuda.stubs.Stub):
    _description_ = '<syncthreads_or()>'


@numba.cuda.cudaimpl.lower(syncthreads_or, nb.i4)
def lower_syncthreads_or(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.or'
    lmod = builder.module
    fnty = lc.Type.function(lc.Type.int(32), (lc.Type.int(32),))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, args)


@nb.cuda.cudadecl.intrinsic
class Cuda_syncthreads_or(nb.typing.templates.AbstractTemplate):
    key = syncthreads_or

    def generic(self, args, kws):
        return nb.i4(nb.i4, )


nb.cuda.cudadecl.intrinsic_global(syncthreads_or, nb.types.Function(Cuda_syncthreads_or))


class syncthreads_and(nb.cuda.stubs.Stub):
    _description_ = '<syncthreads_and()>'


@numba.cuda.cudaimpl.lower(syncthreads_and, nb.i4)
def lower_syncthreads_and(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.and'
    lmod = builder.module
    fnty = lc.Type.function(lc.Type.int(32), (lc.Type.int(32),))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, args)


@nb.cuda.cudadecl.intrinsic
class Cuda_syncthreads_and(nb.typing.templates.AbstractTemplate):
    key = syncthreads_and

    def generic(self, args, kws):
        return nb.i4(nb.i4, )


nb.cuda.cudadecl.intrinsic_global(syncthreads_and, nb.types.Function(Cuda_syncthreads_and))


class syncthreads_popc(nb.cuda.stubs.Stub):
    _description_ = '<syncthreads_popc()>'


@numba.cuda.cudaimpl.lower(syncthreads_popc, nb.i4)
def lower_syncthreads_popc(context, builder, sig, args):
    fname = 'llvm.nvvm.barrier0.popc'
    lmod = builder.module
    fnty = lc.Type.function(lc.Type.int(32), (lc.Type.int(32),))
    func = lmod.get_or_insert_function(fnty, name=fname)
    return builder.call(func, args)


@nb.cuda.cudadecl.intrinsic
class Cuda_syncthreads_popc(nb.typing.templates.AbstractTemplate):
    key = syncthreads_popc

    def generic(self, args, kws):
        return nb.i4(nb.i4, )


nb.cuda.cudadecl.intrinsic_global(syncthreads_popc, nb.types.Function(Cuda_syncthreads_popc))


if __name__ == "__main__":
    @nb.cuda.jit((nb.i4[:],))
    def test(arr):
        i = nb.cuda.threadIdx.x
        # arr[i] = shfl.idx(syncthreads_popc(1), 0, 32)
        arr[i] = syncthreads_popc(1)


    testA = np.empty(64, np.int32)
    test[2, 32](testA)
