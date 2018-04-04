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



class _OverloadCudaFunctionTemplate(nb.typing.templates.AbstractTemplate):
    """
    A base class of templates for overload functions.
    """

    def generic(self, args, kws):
        """
        Type the overloaded function by compiling the appropriate
        implementation for the given args.
        """
        cache_key = self.context, args, tuple(kws.items())
        try:
            disp = self._impl_cache[cache_key]
        except KeyError:
            # Get the overload implementation for the given types
            pyfunc = self._overload_func(*args, **kws)
            if pyfunc is None:
                # No implementation => fail typing
                self._impl_cache[cache_key] = None
                return
            jitdecor = nb.cuda.jit(args, device=True, **self._jit_options)
            disp = self._impl_cache[cache_key] = jitdecor(pyfunc)
        else:
            if disp is None:
                return
        # Store the compiled overload for use in the lowering phase
        self._compiled_overloads[args] = disp
        return disp.cres.signature

    def get_impl_key(self, sig):
        """
        Return the key for looking up the implementation for the given
        signature on the target context.
        """
        return self._compiled_overloads[sig.args]



class _OverloadCudaAttributeTemplate(nb.typing.templates.AttributeTemplate):
    """
    A base class of templates for @overload_attribute functions.
    """

    def __init__(self, context):
        super(_OverloadCudaAttributeTemplate, self).__init__(context)
        self.context = context

    @classmethod
    def do_class_init(cls):
        """
        Register attribute implementation.
        """
        attr = cls._attr

        @numba.cuda.cudaimpl.lower(cls.key, attr)
        def getattr_impl(context, builder, typ, value):
            sig_args = (typ,)
            sig_kws = {}
            typing_context = context.typing_context
            disp = cls._get_dispatcher(typing_context, typ, attr, sig_args, sig_kws)
            sig = disp.cres.signature
            call = context.get_function(disp, sig)
            return call(builder, (value,))

    @classmethod
    def _get_dispatcher(cls, context, typ, attr, sig_args, sig_kws):
        """
        Get the compiled dispatcher implementing the attribute for
        the given formal signature.
        """
        cache_key = context, typ, attr
        try:
            disp = cls._impl_cache[cache_key]
        except KeyError:
            # Get the overload implementation for the given type
            pyfunc = cls._overload_func(*sig_args, **sig_kws)
            if pyfunc is None:
                # No implementation => fail typing
                cls._impl_cache[cache_key] = None
                return

            disp = cls._impl_cache[cache_key] = nb.cuda.jit(sig_args, device=True)(pyfunc)
        return disp

    def _resolve_impl_sig(self, typ, attr, sig_args, sig_kws):
        """
        Compute the actual implementation sig for the given formal argument types.
        """
        disp = self._get_dispatcher(self.context, typ, attr, sig_args, sig_kws)
        if disp is None:
            return None

        sig = disp.cres.signature
        return sig

    def _resolve(self, typ, attr):
        if self._attr != attr:
            return None

        sig = self._resolve_impl_sig(typ, attr, (typ,), {})
        return sig.return_type



class _OverloadCudaMethodTemplate(_OverloadCudaAttributeTemplate):
    """
    A base class of templates for @overload_method functions.
    """

    @classmethod
    def do_class_init(cls):
        """
        Register generic method implementation.
        """
        attr = cls._attr

        @numba.cuda.cudaimpl.lower((cls.key, attr), cls.key, nb.types.VarArg(nb.types.Any))
        def method_impl(context, builder, sig, args):
            typ = sig.args[0]
            typing_context = context.typing_context
            disp = cls._get_dispatcher(typing_context, typ, attr, sig.args, {})
            sig = disp.cres.signature
            call = context.get_function(disp, sig)
            # Link dependent library
            cg = context.codegen()
            for lib in getattr(call, 'libs', ()):
                cg.add_linking_library(lib)
            return call(builder, args)

    def _resolve(self, typ, attr):
        if self._attr != attr:
            return None

        assert isinstance(typ, self.key)

        class MethodTemplate(nb.typing.templates.AbstractTemplate):
            key = (self.key, attr)

            def generic(_, args, kws):
                args = (typ,) + args
                sig = self._resolve_impl_sig(typ, attr, args, kws)
                if sig is not None:
                    return sig.as_method()

        return nb.types.BoundFunction(MethodTemplate, typ)


def make_cuda_overload_template(func, overload_func, jit_options):
    """
    Make a template class for function *func* overloaded by *overload_func*.
    Compiler options are passed as a dictionary to *jit_options*.
    """
    func_name = getattr(func, '__name__', str(func))
    name = "OverloadCudaTemplate_%s" % (func_name,)
    base = _OverloadCudaFunctionTemplate
    dct = dict(key=func, _overload_func=staticmethod(overload_func),
               _impl_cache={}, _compiled_overloads={}, _jit_options=jit_options)
    return type(base)(name, (base,), dct)


def make_cuda_overload_attribute_template(typ, attr, overload_func, base=_OverloadCudaAttributeTemplate):
    """
    Make a template class for attribute *attr* of *typ* overloaded by
    *overload_func*.
    """
    assert isinstance(typ, nb.types.Type) or issubclass(typ, nb.types.Type)
    name = "OverloadTemplate_%s_%s" % (typ, attr)
    # Note the implementation cache is subclass-specific
    dct = dict(key=typ, _attr=attr, _impl_cache={},
               _overload_func=staticmethod(overload_func),
               )
    return type(base)(name, (base,), dct)


def make_cuda_overload_method_template(typ, attr, overload_func):
    """
    Make a template class for method *attr* of *typ* overloaded by
    *overload_func*.
    """
    return make_cuda_overload_attribute_template(typ, attr, overload_func, base=_OverloadCudaMethodTemplate)


def overload_cuda(func, opts={}):
    def decorate(overload_func):
        template = make_cuda_overload_template(func, overload_func, opts)
        nb.cuda.cudadecl.intrinsic(template)
        if hasattr(func, '__module__'):
            nb.cuda.cudadecl.intrinsic_global(func, nb.types.Function(template))
        return overload_func

    return decorate


def overload_cuda_attribute(typ, attr):
    def decorate(overload_func):
        template = make_cuda_overload_attribute_template(typ, attr, overload_func)
        nb.cuda.cudadecl.intrinsic_attr(template)
        return overload_func

    return decorate


def overload_cuda_method(typ, attr):
    def decorate(overload_func):
        template = make_cuda_overload_method_template(typ, attr, overload_func)
        nb.cuda.cudadecl.intrinsic_attr(template)
        return overload_func
    return decorate



@numba.cuda.cudaimpl.lower('in', nb.types.Any, nb.types.IterableType)
def lower_seq_in(context, builder, sig, args):
    def in_impl(v, arr):
        for elem in arr:
            if elem == v:
                return True
        return False
    return context.compile_internal(builder, in_impl, sig, args)


@nb.cuda.cudadecl.intrinsic
class Cuda_seq_in(nb.typing.templates.AbstractTemplate):
    key = 'in'

    def generic(self, args, kws):
        assert len(args) == 2 and isinstance(args[1], nb.types.IterableType)
        return nb.b1(*args)


@overload_cuda(sum)
def overload_cuda_sum(seq):
    if isinstance(seq, nb.types.Array):
        zero = seq.dtype(0)
        def sum_impl(arr):
            s = zero
            for v in arr:
                s += v
            return s
        return sum_impl


@overload_cuda_method(nb.types.Array, 'sum')
def overload_cuda_arr_sum(seq):
    if seq.ndim == 1:
        def sum_impl(arr):
            s = arr[0]
            for i in range(1, len(arr)):
                s += arr[i]
            return s
        return sum_impl


@overload_cuda_method(nb.types.Array, 'max')
def overload_cuda_arr_max(seq):
    if seq.ndim == 1:
        def max_impl(arr):
            m = arr[0]
            for i in range(1, len(arr)):
                m = max(m, arr[i])
            return m
        return max_impl


@overload_cuda_method(nb.types.Array, 'min')
def overload_cuda_arr_min(seq):
    if seq.ndim == 1:
        def min_impl(arr):
            m = arr[0]
            for i in range(1, len(arr)):
                m = min(m, arr[i])
            return m
        return min_impl


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

syncthreads_and = add_nvvm_intrinsic('syncthreads_and', 'llvm.nvvm.barrier0.and', nb.i4(nb.i4))
syncthreads_or = add_nvvm_intrinsic('syncthreads_or', 'llvm.nvvm.barrier0.or', nb.i4(nb.i4))
syncthreads_popc = add_nvvm_intrinsic('syncthreads_popc', 'llvm.nvvm.barrier0.popc', nb.i4(nb.i4))
laneid = add_nvvm_intrinsic('laneid', 'llvm.nvvm.read.ptx.sreg.laneid', nb.i4())
ctlz = add_nvvm_intrinsic('ctlz', 'llvm.ctlz.i32', nb.i4(nb.i4))
warp_sync = add_nvvm_intrinsic('warp_sync', 'llvm.nvvm.bar.warp.sync', nb.void(nb.i4))



class shfl_sync(nb.cuda.stubs.Stub):
    """https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#nvvm-intrin-warp-level-datamove"""
    _description_ = '<shfl_sync()>'

shfl_sync_llvm_sign = lc.Type.function(lc.Type.struct((lc.Type.int(32), lc.Type.int(1))), 5*[lc.Type.int(32)])


@numba.cuda.cudaimpl.lower(shfl_sync, nb.i4, nb.i4, nb.i4, nb.i4, nb.i4)
def lower_shfl_sync_intrinsic(context, builder, sig, args):
    fn = nb.cgutils.insert_pure_function(builder.module, shfl_sync_llvm_sign, 'llvm.nvvm.shfl.sync.i32')
    return builder.call(fn, args, tail=True)


@numba.cuda.cudaimpl.lower(shfl_sync, nb.i4, nb.i4, nb.f4, nb.i4, nb.i4)
def lower_shfl_sync_intrinsic(context, builder, sig, args):
    casted_args = (args[0], args[1], builder.bitcast(args[2], lc.Type.int(32)), args[3], args[4])
    fn = nb.cgutils.insert_pure_function(builder.module, shfl_sync_llvm_sign, 'llvm.nvvm.shfl.sync.i32')
    rstruct = builder.call(fn, casted_args, tail=True)
    ival, pred = builder.extract_value(rstruct, 0), builder.extract_value(rstruct, 1)
    fval = builder.bitcast(ival, lc.Type.float())
    return nb.cgutils.make_anonymous_struct(builder, (fval, pred))


@nb.cuda.cudadecl.intrinsic
class CudaShflSyncTemplate(nb.typing.templates.AbstractTemplate):
    key = shfl_sync
    def generic(self, args, kws):
        vty = args[2]
        if vty == nb.i4 or vty == nb.i8 or vty == nb.f4:
            return nb.types.Tuple((vty, nb.boolean))(nb.i4, nb.i4, vty, nb.i4, nb.i4)

nb.cuda.cudadecl.intrinsic_global(shfl_sync, nb.types.Function(CudaShflSyncTemplate))


@nb.cuda.jit(nb.i4(nb.i4), device=True)
def ilog2(v):
    """
    l2 = 0
        while v > 1:
            v >>= 1
            l2 += 1
        return l2
    """
    return 31 - ctlz(v)


def create_reduce(func, ntype=nb.f4):
    warp_size = np.int32(32)
    mask = 0xffffffff
    packing = 0x1f

    @nb.cuda.jit(ntype(ntype, nb.i4), device=True)
    def reduce(val, width):
        if width <= warp_size and not (width & (width - 1)):
            for i in range(ilog2(width)):
                val = func(val, shfl_sync(mask, 3, val, 1 << i, packing)[0])
            return val
        elif width <= warp_size:
            closest_pow2 = np.int32(1 << ilog2(width))
            diff = np.int32(width - closest_pow2)
            lid = laneid()
            temp = shfl_sync(mask, 2, val, closest_pow2, packing)[0]
            if lid < diff:
                val = func(val, temp)
            for i in range(ilog2(width)):
                val = func(val, shfl_sync(mask, 3, val, 1 << i, packing)[0])
            return shfl_sync(mask, 0, val, 0, packing)[0]
        else:
            warp_count = int(math.ceil(width / warp_size))
            last_warp_size = width % warp_size
            nb.cuda.syncthreads()
            buffer = nb.cuda.shared.array(0, ntype)
            tid = nb.cuda.threadIdx.x
            lid = laneid()
            nb.cuda.syncthreads()
            if (last_warp_size == 0) or (tid < width - last_warp_size):
                for i in range(ilog2(warp_size)):
                    val = func(val, shfl_sync(mask, 3, val, 1 << i, packing)[0])
            elif not (last_warp_size & (last_warp_size - 1)):
                for i in range(ilog2(last_warp_size)):
                    val = func(val, shfl_sync(mask, 3, val, 1 << i, packing)[0])
            else:
                closest_lpow2 = np.int32(1 << ilog2(last_warp_size))
                temp = shfl_sync(mask, 2, val, closest_lpow2, packing)[0]
                if lid < last_warp_size - closest_lpow2:
                    val = func(val, temp)
                for i in range(ilog2(closest_lpow2)):
                    val = func(val, shfl_sync(mask, 3, val, 1 << i, packing)[0])
            if lid == 0:
                buffer[tid // warp_size] = val
            nb.cuda.syncthreads()
            val = buffer[0]
            for i in range(1, warp_count):
                val = func(val, buffer[i])
            return val
    return reduce


if __name__ == "__main__":
    import os
    os.environ['NUMBAPRO_NVVM'] = '/usr/local/cuda/nvvm/lib64/libnvvm.so'
    os.environ['NUMBAPRO_LIBDEVICE'] = '/usr/local/cuda/nvvm/libdevice/'
    import operator

    block_size = 68
    summer = create_reduce(operator.add)

    @nb.cuda.jit((nb.f4[:],))
    def test(arr):
        tid = nb.cuda.threadIdx.x
        val = summer(tid, block_size)
        arr[tid] = val

    # testF = nb.cuda.jit(nb.f4[:])(test)
    # testA = np.empty(64, np.float32)
    # testF[1, 64](testA)
    # print(testA)

    testA = np.empty(block_size, np.float32)
    test[1, block_size, None, 4*block_size](testA)
    print(testA)
    print(test.ptx)