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
