from functools import wraps
import inspect
import operator
from types import MethodType, FunctionType

import numba
from numba import types
import numba.extending
from numba.extending import _overload_default_jit_options
import numba.typing as typing
from numba.typing.templates import make_overload_template, make_overload_attribute_template, \
    make_overload_method_template, infer, infer_global, infer_getattr, _EmptyImplementationEntry, Signature, \
    _OverloadFunctionTemplate, _OverloadAttributeTemplate, _OverloadMethodTemplate, _inline_info
from numba.targets.cpu_options import InlineOptions
import numba.cuda as cuda
from numba.cuda.cudadecl import intrinsic, intrinsic_global, intrinsic_attr
import numba.cuda.cudaimpl
from numba.cuda.cudaimpl import registry


annotations_to_numba = {
    int: types.int_,
    float: types.f8,
}


class _OverloadCudaFunctionTemplate(_OverloadFunctionTemplate):
    def generic(self, args, kws):
        """
        Type the overloaded function by compiling the appropriate
        implementation for the given args.
        """
        print('generic', self, args, kws)
        disp, new_args = self._get_impl(args, kws)
        if disp is None:
            return
        cres = disp.compile(new_args)
        if not self._inline.is_never_inline:
            raise NotImplementedError
        else:
            sig = cres.signature
            self._compiled_overloads[sig.args] = self.key
        print(sig, cres)
        return sig

    def _build_impl(self, cache_key, args, kws):
        """Build and cache the implementation.

        Given the positional (`args`) and keyword arguments (`kws`), obtains
        the `overload` implementation and wrap it in a Dispatcher object.
        The expected argument types are returned for use by type-inference.
        The expected argument types are only different from the given argument
        types if there is an imprecise type in the given argument types.

        Parameters
        ----------
        cache_key : hashable
            The key used for caching the implementation.
        args : Tuple[Type]
            Types of positional argument.
        kws : Dict[Type]
            Types of keyword argument.

        Returns
        -------
        disp, args :
            On success, returns `(Dispatcher, Tuple[Type])`.
            On failure, returns `(None, None)`.

        """
        print('buiild impl', self, cache_key, args, kws)
        # Get the overload implementation for the given types
        ovf_result = self._overload_func(*args, **kws)
        if ovf_result is None:
            # No implementation => fail typing
            self._impl_cache[cache_key] = None, None
            return None, None
        elif isinstance(ovf_result, tuple):
            # The implementation returned a signature that the type-inferencer
            # should be using.
            sig, pyfunc = ovf_result
            args = sig.args
            cache_key = None            # don't cache
        else:
            # Regular case
            pyfunc = ovf_result

        # Check type of pyfunc
        if not isinstance(pyfunc, FunctionType):
            msg = ("Implementator function returned by `@overload` "
                   "has an unexpected type.  Got {}")
            raise AssertionError(msg.format(pyfunc))

        # check that the typing and impl sigs match up
        if self._strict:
            self._validate_sigs(self._overload_func, pyfunc)
        # Make dispatcher
        print(self._jit_options)
        jitdecor = cuda.jit(device=True, **self._jit_options)
        disp = jitdecor(pyfunc)
        if cache_key is not None:
            self._impl_cache[cache_key] = disp, args
        return disp, args


class _OverloadCudaAttributeTemplate(_OverloadAttributeTemplate):
    """
    A base class of templates for @overload_attribute functions.
    """

    @classmethod
    def do_class_init(cls):
        """
        Register attribute implementation.
        """
        attr = cls._attr

        @registry.lower_getattr(cls.key, attr)
        def getattr_impl(context, builder, typ, value):
            print('getattr',  cls.key, attr, context, builder, typ, value)
            typingctx = context.typing_context
            fnty = cls._get_function_type(typingctx, typ)
            sig = cls._get_signature(typingctx, fnty, (typ,), {})
            call = context.get_function(fnty, sig)
            return call(builder, (value,))


class _OverloadCudaMethodTemplate(_OverloadMethodTemplate):
    """
    A base class of templates for @overload_method functions.
    """

    @classmethod
    def do_class_init(cls):
        """
        Register generic method implementation.
        """
        attr = cls._attr

        @registry.lower((cls.key, attr), cls.key, types.VarArg(types.Any))
        def method_impl(context, builder, sig, args):
            print('method', cls.key, attr, context, builder, sig, args)
            typ = sig.args[0]
            typing_context = context.typing_context
            fnty = cls._get_function_type(typing_context, typ)
            sig = cls._get_signature(typing_context, fnty, sig.args, {})
            call = context.get_function(fnty, sig)
            # Link dependent library
            context.add_linking_libs(getattr(call, 'libs', ()))
            return call(builder, args)


def make_cuda_overload_template(func, overload_func, jit_options, strict,
                                inline):
    """
    Make a template class for function *func* overloaded by *overload_func*.
    Compiler options are passed as a dictionary to *jit_options*.
    """
    func_name = getattr(func, '__name__', str(func))
    name = "OverloadCudaTemplate_%s" % (func_name,)
    base = _OverloadCudaFunctionTemplate
    dct = dict(key=func, _overload_func=staticmethod(overload_func),
               _impl_cache={}, _compiled_overloads={}, _jit_options=jit_options,
               _strict=strict, _inline=staticmethod(InlineOptions(inline)),
               _inline_overloads={})
    return type(base)(name, (base,), dct)


def overload(func, jit_options={}, strict=True, inline='never'):
    # set default options
    opts = _overload_default_jit_options.copy()
    opts.update(jit_options)  # let user options override

    def decorate(overload_func):
        template = make_overload_template(func, overload_func, opts, strict,
                                          inline)
        opts.pop("no_cpython_wrapper", None)
        opts.pop("boundscheck", None)
        cuda_template = make_cuda_overload_template(func, overload_func, opts, strict, inline)
        infer(template)
        intrinsic(cuda_template)
        if callable(func):
            infer_global(func, types.Function(template))
            intrinsic_global(func, types.Function(cuda_template))
        return overload_func

    return decorate


def overload_attribute(typ, attr, **kwargs):
    def decorate(overload_func):
        template = make_overload_attribute_template(
            typ, attr, overload_func,
            inline=kwargs.get('inline', 'never'),
        )
        cuda_template = make_overload_attribute_template(
            typ, attr, overload_func,
            inline=kwargs.get('inline', 'never'),
            base=_OverloadCudaAttributeTemplate,
        )
        infer_getattr(template)
        intrinsic_attr(cuda_template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func

    return decorate


def overload_method(typ, attr, **kwargs):
    def decorate(overload_func):
        template = make_overload_method_template(
            typ, attr, overload_func,
            inline=kwargs.get('inline', 'never'),
        )
        cuda_template = make_overload_attribute_template(
            typ, attr, overload_func,
            inline=kwargs.get('inline', 'never'),
            base=_OverloadCudaMethodTemplate,
        )
        infer_getattr(template)
        intrinsic_attr(cuda_template)
        overload(overload_func, **kwargs)(overload_func)
        return overload_func

    return decorate


def overload_helper(cls, builtin, fn):
    @overload(builtin)
    @wraps(fn)
    def _vector_overload(self, *args, **kwargs):
        if isinstance(self, types.BaseNamedTuple) and self.instance_class is cls:
            return fn


def overload_method_helper(cls, method_name, method_fn):
    @overload_method(types.BaseNamedTuple, method_name)
    @wraps(method_fn)
    def _vector_overload(self, *args, **kwargs):
        if isinstance(self, types.BaseNamedTuple) and self.instance_class is cls:
            return method_fn


def overload_getattr_helper(cls, attr_name, attr_fn):
    @overload_attribute(types.BaseNamedTuple, attr_name)
    @wraps(attr_fn)
    def _vector_overload(self, *args, **kwargs):
        if isinstance(self, types.BaseNamedTuple) and self.instance_class is cls:
            return attr_fn


def overload_named_tuple_subclass(cls):
    _prohibited = ('__new__', '__init__', '__slots__', '__getnewargs__',
                   '_fields', '_field_defaults', '_field_types',
                   '_make', '_replace', '_asdict', '_source')
    _special = ('__module__', '__name__', '__qualname__', '__annotations__')
    filtered = _prohibited + _special + ('__doc__', '_fields_defaults', '__repr__')
    for name, val in cls.__dict__.items():
        if name in filtered:
            continue
        if inspect.isroutine(val):
            if hasattr(operator, name) and inspect.isbuiltin(getattr(operator, name)):
                overload_helper(cls, getattr(operator, name), val)
            elif inspect.isfunction(val):
                overload_method_helper(cls, name, val)
            elif not (isinstance(val, classmethod) or isinstance(val, staticmethod)):
                overload_getattr_helper(cls, name, val)
    return cls
