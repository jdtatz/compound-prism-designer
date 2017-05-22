import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_double, cast, sizeof, POINTER, CFUNCTYPE
import numba as nb
import numpy as np
from functools import partial

jit = partial(nb.jit, nopython=True, nogil=True, cache=True)
cfunc = partial(nb.cfunc, nopython=True, nogil=True, cache=True)


@jit
def gradient(f: np.ndarray):
    """
    the gradient of a 1-dim numpy array
    """
    out = np.empty(f.shape)
    out[1:-1] = (f[2:] - f[:-2])/2.0
    out[0] = (f[1] - f[0])
    out[-1] = (f[-1] - f[-2])
    return out


def libfunc(lib, name, ret, *args):
    func = getattr(lib, name)
    func.argtypes = args
    func.restype = ret
    return func

libc = ctypes.CDLL(ctypes.util.find_library("c"))
libgslcblas = ctypes.CDLL('libgslcblas.so', mode=ctypes.RTLD_GLOBAL)
libgsl = ctypes.CDLL('libgsl.so')

malloc = libfunc(libc, 'malloc', c_void_p, c_size_t)
free = libfunc(libc, 'free', None, c_void_p)

vec_alloc = libfunc(libgsl, 'gsl_vector_alloc', c_void_p, c_size_t)
vec_free = libfunc(libgsl, 'gsl_vector_free', None, c_void_p)
vec_set = libfunc(libgsl, 'gsl_vector_set', None, c_void_p, c_size_t, c_double)
vec_set_all = libfunc(libgsl, 'gsl_vector_set_all', None, c_void_p, c_double)
vec_get = libfunc(libgsl, 'gsl_vector_get', c_double, c_void_p, c_size_t)
vec_ptr = libfunc(libgsl, 'gsl_vector_ptr', POINTER(c_double), c_void_p, c_size_t)
alloc_fmin = libfunc(libgsl, 'gsl_multimin_fminimizer_alloc', c_void_p, c_void_p, c_size_t)
set_fmin = libfunc(libgsl, 'gsl_multimin_fminimizer_set', c_int, c_void_p, c_void_p, c_void_p, c_void_p)
iter_fmin = libfunc(libgsl, 'gsl_multimin_fminimizer_iterate', c_int, c_void_p)
free_fmin = libfunc(libgsl, 'gsl_multimin_fminimizer_free', None, c_void_p)
getx_fmin = libfunc(libgsl, 'gsl_multimin_fminimizer_x', c_void_p, c_void_p)
getmin_fmin = libfunc(libgsl, 'gsl_multimin_fminimizer_minimum', c_double, c_void_p)
getsize_fmin = libfunc(libgsl, 'gsl_multimin_fminimizer_size', c_double, c_void_p)

nsimplex = c_void_p.in_dll(libgsl, 'gsl_multimin_fminimizer_nmsimplex2').value

gsl_multimin_function = type("gsl_multimin_function", (ctypes.Structure,),
    {"_fields_": [("f", c_void_p), ("n", c_size_t), ("params", c_void_p)]})


@CFUNCTYPE(c_void_p, c_void_p, c_size_t, c_void_p)
def create_func(f, n, p):
    buf = cast(malloc(sizeof(gsl_multimin_function)), POINTER(gsl_multimin_function))
    buf.contents.f = f
    buf.contents.n = n
    buf.contents.params = p
    return cast(buf, c_void_p).value


@jit
def minimizer(func, x0, params):
    size = x0.size
    fn = create_func(func, size, params.ctypes.data)
    x_vec, step_vec = vec_alloc(size), vec_alloc(size)
    for i in range(size):
        vec_set(x_vec, i, x0[i])
    vec_set_all(step_vec, 0.00025)
    mini = alloc_fmin(nsimplex, size)
    set_fmin(mini, fn, x_vec, step_vec)
    for it in range(200):
        if iter_fmin(mini):
            print("Failure to iterate")
            break
        elif getsize_fmin(mini) < 1e-4:
            break
    x = getx_fmin(mini)
    x_out = np.empty(size, dtype=np.float64)
    for i in range(size):
        x_out[i] = vec_get(x, i)
    free_fmin(mini)
    vec_free(x_vec)
    vec_free(step_vec)
    free(fn)
    return x_out
