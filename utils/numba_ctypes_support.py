from ctypes import *
import ctypes.util
import numba as nb
import numba.extending
import numba.typing
import numba.cgutils
from llvmlite import ir
from collections import OrderedDict


def cast_llvm(val, numba_type, llvm_type, builder):
    if val.type is llvm_type:
        return val
    elif isinstance(val.type, ir.IntType) and isinstance(llvm_type, ir.IntType):
        if val.type.width < llvm_type.width:
            if numba_type.signed:
                return builder.sext(val, llvm_type)
            else:
                return builder.zext(val, llvm_type)
        elif val.type.width > llvm_type.width:
            return builder.trunc(val, llvm_type)
        else:
            return builder.bitcast(val, llvm_type)
    elif isinstance(val.type, (ir.FloatType, ir.DoubleType)) and isinstance(llvm_type, (ir.FloatType, ir.DoubleType)):
        if isinstance(val.type, ir.FloatType):
            return builder.fpext(val, llvm_type)
        else:
            return builder.fptrunc(val, llvm_type)
    elif isinstance(val.type, ir.IntType) and isinstance(llvm_type, ir.PointerType):
        return builder.inttoptr(val, llvm_type)
    elif isinstance(val.type, ir.PointerType) and isinstance(llvm_type, ir.IntType):
        return builder.ptrtoint(val, llvm_type)
    raise NotImplementedError


def register_ctypes_type(ctypes_type, numba_type=None, llvm_type=None):
    if issubclass(ctypes_type, ctypes._SimpleCData):
        if numba_type is None or llvm_type is None:
            return NotImplementedError
        typ = CSimpleType(ctypes_type, numba_type, llvm_type)
        return register_simple_type(ctypes_type, typ)
    elif issubclass(ctypes_type, ctypes._Pointer):
        return register_pointer_type(ctypes_type)
    elif issubclass(ctypes_type, Structure):
        return register_struct(ctypes_type)
    else:
        raise NotImplementedError

"""
CSimple
"""


class CSimpleType(nb.types.Type):
    def __init__(self, ctyp, ntyp, primative):
        super().__init__(name="Ctypes({})".format(ctyp.__name__))
        self.ctypes_type = ctyp
        self.numba_type = ntyp
        self.primative = primative

    def cast_python_value(self, value):
        return self.ctypes_type(value)


@nb.extending.register_model(CSimpleType)
class CSimpleModel(nb.extending.models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        # print('model', fe_type, dmm)
        super().__init__(dmm, fe_type, fe_type.primative)


ctypes_typed_cache = {
    c_bool: CSimpleType(c_bool, nb.b1, ir.IntType(1)),
    c_uint8: CSimpleType(c_uint8, nb.u1, ir.IntType(8)),
    c_uint16: CSimpleType(c_uint16, nb.u2, ir.IntType(16)),
    c_uint32: CSimpleType(c_uint32, nb.u4, ir.IntType(32)),
    c_uint64: CSimpleType(c_uint64, nb.u8, ir.IntType(64)),
    c_int8: CSimpleType(c_int8, nb.i1, ir.IntType(8)),
    c_int16: CSimpleType(c_int16, nb.i2, ir.IntType(16)),
    c_int32: CSimpleType(c_int32, nb.i4, ir.IntType(32)),
    c_int64: CSimpleType(c_int64, nb.i8, ir.IntType(64)),
    c_float: CSimpleType(c_float, nb.f4, ir.FloatType()),
    c_double: CSimpleType(c_double, nb.f8, ir.DoubleType()),
    c_void_p: CSimpleType(c_void_p, nb.intp, ir.IntType(64))
}


def register_simple_type(ctyp, typ):
    if ctyp not in ctypes_typed_cache:
        ctypes_typed_cache[ctyp] = typ

    @nb.extending.type_callable(ctyp)
    def call_typ_ctyp(context):
        def typer(val):
            # print("typer", val)
            return ctypes_typed_cache[ctyp]
        # print('get_typer', ctyp, context)
        return typer

    @nb.extending.lower_builtin(typ.ctypes_type, nb.types.Number)
    def impl_csimple(context, builder, sig, args):
        value = args[0]
        # print("impl", sig.return_type, value, value.type)
        return cast_llvm(value, typ.numba_type, typ.primative, builder)

    @nb.extending.lower_cast(typ, typ.numba_type)
    def cast_csimple(context, builder, fromty, toty, val):
        print('CAST')
        return val


for simple_ctyp, simple_typ in ctypes_typed_cache.items():
    register_simple_type(simple_ctyp, simple_typ)


@nb.extending.typeof_impl.register(ctypes._SimpleCData)
def typeof_csimple(val, c):
    # print("reg", val, type(val), c)
    return ctypes_typed_cache[type(val)]


@nb.extending.unbox(CSimpleType)
def unbox_csimple(typ, obj, c):
    """
    Convert object to a native structure.
    """
    print('unbox', typ, obj, c.context.get_argument_type(typ))
    value_obj = c.pyapi.object_getattr_string(obj, "value")
    native_val = c.pyapi.to_native_value(typ.numba_type, value_obj)
    c.pyapi.decref(value_obj)
    return native_val


@nb.extending.box(CSimpleType)
def box_csimple(typ, val, c):
    """
    Convert a native structure to an object.
    """
    print('box', typ, val, typ.name)
    c_simple = c.pyapi.from_native_value(typ.numba_type, val)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.ctypes_type))
    obj = c.pyapi.call_function_objargs(class_obj, (c_simple, ))
    c.pyapi.decref(class_obj)
    return obj


@nb.extending.infer_getattr
class CSimpleAttributeTemplate(nb.typing.templates.AttributeTemplate):
    key = CSimpleType

    def generic_resolve(self, typ, attr):
        # print('resolve', typ, attr, self.key)
        if attr == 'value':
            return typ.numba_type


@nb.extending.lower_getattr(CSimpleType, 'value')
def get_value_csimple(context, builder, typ, val):
    # print('getValue', typ, val)
    return val

"""
CPointers
"""


class CPointerType(nb.types.Type):
    def __init__(self, base):
        super().__init__(name="CtypesPtr({})".format(base.ctypes_type.__name__))
        self.base = base
        self.ctypes_type = POINTER(base.ctypes_type)
        self.numba_type = nb.types.CPointer(base)
        self.primative = ir.PointerType(base.primative)

    def cast_python_value(self, value):
        return self.ctypes_type(value)


@nb.extending.register_model(CPointerType)
class CPointerModel(nb.extending.models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        # print('model', fe_type, dmm)
        super().__init__(dmm, fe_type, fe_type.primative)


def register_pointer_type(ptrTyp):
    baseTyp = ptrTyp._type_
    if ptrTyp in ctypes_typed_cache:
        return ptrTyp
    elif baseTyp not in ctypes_typed_cache:
        register_ctypes_type(baseTyp)
    typ = CPointerType(ctypes_typed_cache[baseTyp])
    globals()[ptrTyp.__name__] = ptrTyp
    ctypes_typed_cache[ptrTyp] = typ

    @nb.extending.type_callable(ptrTyp)
    def call_typ_cpointer(context):
        def typer(val):
            # print("typer", val)
            return typ
        # print('get_typer', typ.ctypes_type, context)
        return typer

    @nb.extending.lower_builtin(ptrTyp, typ.numba_type.key)
    def impl_cpointer(context, builder, sig, args):
        typ = sig.return_type
        value = args[0]
        # print("impl", typ, value.type, value)
        val = nb.cgutils.alloca_once_value(builder, value)
        return val

    return ptrTyp


@nb.extending.typeof_impl.register(ctypes._Pointer)
def typeof_cpointer(val, c):
    # print("reg", val, type(val), c)
    ptrTyp = type(val)
    if ptrTyp not in ctypes_typed_cache:
        register_pointer_type(ptrTyp)
    return ctypes_typed_cache[ptrTyp]


@nb.extending.unbox(CPointerType)
def unbox_cpointer(typ, obj, c):
    """
    Convert object to a native structure.
    """
    # print('unbox', typ, obj, c.context.get_argument_type(typ))
    contents_obj = c.pyapi.object_getattr_string(obj, "contents")
    contents = c.pyapi.to_native_value(typ.numba_type.key, contents_obj)
    c.pyapi.decref(contents_obj)
    val = nb.cgutils.alloca_once_value(c.builder, contents.value)
    is_error = nb.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return nb.extending.NativeValue(val, is_error=is_error)


@nb.extending.box(CPointerType)
def box_cpointer(typ, val, c):
    """
    Convert a native structure to an object.
    """
    # print('box', typ, val, typ.name, typ.numba_type.key)
    ptr = c.builder.load(val)
    inner = c.pyapi.from_native_value(typ.numba_type.key, ptr)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.ctypes_type))
    res = c.pyapi.call_function_objargs(class_obj, (inner, ))
    c.pyapi.decref(class_obj)
    c.pyapi.decref(inner)
    return res


@nb.extending.infer_getattr
class CPointerAttributeTemplate(nb.typing.templates.AttributeTemplate):
    key = CPointerType

    def generic_resolve(self, typ, attr):
        # print('resolve', typ, attr)
        if attr == 'contents':
            return typ.numba_type.key


@nb.extending.infer
class CPointerMethodTemplate(nb.typing.templates.FunctionTemplate):
    key = "getitem"

    def apply(self, args, kws):
        typ, index = args
        if isinstance(typ, CPointerType):
            retype = typ.numba_type.key
            if isinstance(retype, CSimpleType):
                retype = retype.numba_type
            sig = retype(*args)
            # print('genericIn', sig, self, args, kws)
            return sig


@nb.extending.lower_getattr(CPointerType, 'contents')
def get_contents_cpointer(context, builder, typ, val):
    # print('getContents', typ, val)
    return builder.load(val)


@nb.extending.lower_builtin('getitem', CPointerType, nb.types.Integer)
def get_item_cpointer(context, builder, sig, args):
    val, index = args
    # print('getItem', sig, val, index)
    ptr = nb.cgutils.pointer_add(builder, val, index)
    return builder.load(ptr)

"""
Ctypes Structure
"""

class CStructType(nb.types.Type):
    def __init__(self, struct):
        self.struct = struct
        self.spec = OrderedDict([(field, ctypes_typed_cache[ctyp]) for field, ctyp in struct._fields_])
        super().__init__(name="CtypesStruct({})".format(struct.__name__))


@nb.extending.register_model(CStructType)
class CStructModel(nb.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, fe_type.spec.items())


@nb.extending.infer_getattr
class CPointerAttributeTemplate(nb.typing.templates.AttributeTemplate):
    key = CStructType

    def generic_resolve(self, typ, attr):
        print('resolve', typ, attr, typ.spec)
        if attr in typ.spec:
            attrtyp = typ.spec[attr]
            if isinstance(attrtyp, CSimpleType):
                return attrtyp.numba_type
            return typ.spec[attr]


@nb.extending.lower_getattr_generic(CStructType)
def struct_getattr_impl(context, builder, typ, val, name):
    """ getattr for struct """
    struct = nb.cgutils.create_struct_proxy(typ)(context, builder, value=val)
    attrval = getattr(struct, name)
    # attrty = typ.spec[name]
    # return nb.targets.imputils.impl_ret_borrowed(context, builder, attrty, attrval)
    return attrval


@nb.extending.typeof_impl.register(Structure)
def typeof_struct(val, c):
    """ Find registered type of ctypes Structure, create one if not found """
    print("RegStruct", val, type(val), c)
    typ = type(val)
    if typ not in ctypes_typed_cache:
        register_struct(typ)
    return ctypes_typed_cache[typ]


@nb.extending.unbox(CStructType)
def unbox_struct(typ, obj, c):
    """
    Convert object to a native structure.
    """
    print("unbox", typ, obj)
    struct = nb.cgutils.create_struct_proxy(typ)(c.context, c.builder)
    for field, ftyp in typ.spec.items():
        field_obj = c.pyapi.object_getattr_string(obj, field)
        if isinstance(ftyp, CSimpleType):
            ftyp = ftyp.numba_type
        field_native = c.pyapi.to_native_value(ftyp, field_obj)
        val = field_native.value
        setattr(struct, field, val)
        c.pyapi.decref(field_obj)
    is_error = nb.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return nb.extending.NativeValue(struct._getvalue(), is_error=is_error)


@nb.extending.box(CStructType)
def box_struct(typ, val, c):
    """prism.db
    Convert a native structure to an object.
    """
    print('box', typ, val)
    struct = nb.cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(typ.struct))
    field_objs = [c.pyapi.from_native_value(ftyp, getattr(struct, field)) for field, ftyp in typ.spec.items()]
    res = c.pyapi.call_function_objargs(class_obj, field_objs)
    for field_obj in field_objs:
        c.pyapi.decref(field_obj)
    c.pyapi.decref(class_obj)
    return res


def register_struct(ctypes_struct):
    """
    Register a ctypes Struct
    """
    for field, ctyp in ctypes_struct._fields_:
        if ctyp not in ctypes_typed_cache:
            register_ctypes_type(ctyp)
    cstruct_type = CStructType(ctypes_struct)
    globals()[ctypes_struct.__name__] = ctypes_struct
    ctypes_typed_cache[ctypes_struct] = cstruct_type
    spec = cstruct_type.spec

    @nb.extending.type_callable(ctypes_struct)
    def call_typ_struct(context):
        """
        create call type
        """
        func_source = f"lambda {','.join(f'x{i}' for i in range(len(spec)))}: typ"
        return eval(func_source, {"typ": cstruct_type})

    @nb.extending.lower_builtin(ctypes_struct, nb.types.VarArg(nb.types.Any))
    def impl_struct(context, builder, sig, args):
        """
        make ctypes struct type declaration callable
        """
        typ = sig.return_type
        struct = nb.cgutils.create_struct_proxy(typ)(context, builder)
        for arg, field in zip(args, typ.spec):
            setattr(struct, field, arg)
        print('impl', typ, sig)
        return struct._getvalue()

    return CStructType


"""
Ctypes Util pointer
"""


@nb.extending.infer
class PointerTemplate(nb.typing.templates.FunctionTemplate):
    key = pointer

    def apply(self, args, kws):
        # print('APLY', args, kws)
        typ = args[0]
        ptrType = POINTER(typ.ctypes_type)
        if ptrType not in ctypes_typed_cache:
            register_pointer_type(ptrType)
        sig = ctypes_typed_cache[ptrType](typ)
        return sig

nb.typing.templates.infer_global(pointer, nb.types.Function(PointerTemplate))


@nb.extending.lower_builtin(pointer, nb.types.Any)
def nb_pointer(context, builder, sig, args):
    # print('ptr', sig, args)
    return nb.cgutils.alloca_once_value(builder, args[0])

"""
Function Ptr Code
"""


@nb.extending.typeof_impl.register(ctypes._CFuncPtr)
def test(cfnptr, c):
    if cfnptr.argtypes is None:
        raise TypeError("Need argtypes for CtypesFunctionPtr")
    args = []
    for arg in cfnptr.argtypes:
        if arg not in ctypes_typed_cache:
            register_ctypes_type(arg)
        args.append(ctypes_typed_cache[arg])
    if cfnptr.restype is not None:
        if cfnptr.restype not in ctypes_typed_cache:
            register_ctypes_type(cfnptr.restype)
        res = ctypes_typed_cache[cfnptr.restype]
        if isinstance(res, CSimpleType):
            res = res.numba_type
    else:
        res = nb.types.void
    sig = nb.typing.signature(res, *args)
    return nb.types.ExternalFunctionPointer(sig, cconv=None, get_pointer=lambda f: cast(f, c_void_p).value)

"""
Test Code
"""


@nb.extending.lower_cast(nb.types.Any, nb.types.Any)
def cast_sim(context, builder, fromty, toty, val):
    print('CAST', fromty, toty, val)
    return val


if __name__ == "__main__":

    class myStruct(Structure):
        _fields_ = [('x', c_int64), ('y', c_double), ('z', POINTER(c_int))]
    register_ctypes_type(myStruct)


    class myStruct2(Structure):
        _fields_ = [('x', c_int), ('t', c_float)]


    libc = CDLL(ctypes.util.find_library('c'))

    malloc = libc.malloc
    malloc.argtypes = [c_int64]
    malloc.restype = c_void_p

    free = libc.free
    free.argtypes = [c_void_p]
    free.restype = None


    @nb.jit(nopython=True)
    def test(x2):
        p = c_int64(16)
        t = malloc(p)
        print(t)
        #a = (c_int*5)(1, 2, 3, 4, 5)
        x = c_int(13)
        d = pointer(x)
        d2 = pointer(d)
        z = myStruct(10, 7.2, d)
        print("mix", t, x.value, d.contents, d2.contents.contents, x2, z, z.x, z.y)
        free(c_void_p(t))
        return t


    t = c_int(255)
    l = pointer(t)
    vp = cast(l, c_void_p)
    lt = pointer(l)
    z = myStruct2(5, 6.9)
    ret = test(z)
    print("mini", ret)
