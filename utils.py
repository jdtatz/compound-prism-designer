import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_double, POINTER
import numba as nb
import numba.extending
import numpy as np
from functools import partial

__all__ = ["jit", "get_poly_coeffs", "nonlinearity", "nonlinearity_error", "beam_compression", "spectral_sampling_ratio", "transmission"]

jit = partial(nb.jit, nopython=True, nogil=True, cache=False)
cfunc = partial(nb.cfunc, nopython=True, cache=True)


"""
Utility Functions
"""


@nb.extending.overload(np.gradient)
def np_gradient(arr):
    if isinstance(arr, nb.types.Array) and arr.ndim == 1:
        def gradient(f):
            out = np.empty_like(f)
            out[1:-1] = (f[2:] - f[:-2])/2.0
            out[0] = (f[1] - f[0])
            out[-1] = (f[-1] - f[-2])
            return out
        return gradient


@jit
def get_poly_coeffs(g, n):
    """Get the polynomial coefficients and remainder when treating the input as a n-polynomial function"""
    x = np.linspace(1.0, -1.0, g.size)
    poly = np.empty((g.size, n + 1))
    for i in range(n + 1):
        poly[:, i] = x ** i
    coeffs = np.linalg.pinv(poly) @ g
    remainder = np.sum((g - poly @ coeffs) ** 2)
    return coeffs, remainder


@jit
def nonlinearity(x):
    """Calculate the nonlinearity of the given delta spectrum"""
    g = np.gradient(np.gradient(x))
    return np.sqrt(np.sum(g ** 2))


@jit
def nonlinearity_error(x):
    """Calculate the nonlinearity error of the given delta spectrum"""
    grad = np.gradient(x)
    if np.max(grad) < 0.0 or np.min(grad) < 0.0:
        grad -= np.min(grad)
    return np.sum(np.gradient(grad) ** 2)


@jit
def beam_compression(thetas, nwaves):
    """Calculate the beam compression of the path taken by the central ray of light"""
    ts = np.abs(np.cos(thetas))
    K = np.prod(ts[1::2, nwaves//2]) / np.prod(ts[:-1:2, nwaves//2])
    return K


@jit
def spectral_sampling_ratio(w, delta_spectrum, sampling_domain_is_wavenumber):
    """Calculate the SSR of the given delta spectrum"""
    if sampling_domain_is_wavenumber:
        w = 1000.0 / w
    grad = np.gradient(delta_spectrum) / np.gradient(w)
    maxval, minval = np.max(grad), np.min(grad)
    # If both maxval and minval are negative, then we should swap the ratio to
    # get SSR > 1.
    return minval / maxval if abs(maxval) < abs(minval) else maxval / minval


@jit
def transmission(thetas, n, nwaves):
    """Calculate transmission percentage using Fresnel equations, unfinished"""
    ns = np.vstack((np.ones((1, nwaves)), n, np.ones((1, nwaves))))[:, nwaves//2]
    n1 = ns[:-1]
    n2 = ns[1:]
    # return 100 - 100*np.abs((n1 - n2) / (n1 + n2))**2
    ci = np.cos(thetas[:-1:2, nwaves // 2])
    ct = np.cos(thetas[1::2, nwaves // 2])
    rs = np.abs((n1*ci-n2*ct)/(n1*ci+n2*ct))**2
    rp = np.abs((n1*ct-n2*ci)/(n1*ct+n2*ci))**2
    return 100*np.prod(1 - (rs + rp) / 2)

"""
Ctypes Stuff
"""


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


class gsl_multimin_function(ctypes.Structure):
    _fields_ = [("f", c_void_p), ("n", c_size_t), ("params", c_void_p)]
fsize = ctypes.sizeof(gsl_multimin_function)


@jit
def minimizer(func, x0, modelPtr):
    size = x0.size
    fn = malloc(fsize)
    buffer = nb.carray(fn, (3,), np.uintp)
    buffer[0] = func
    buffer[1] = size
    buffer[2] = modelPtr
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
    merr = getmin_fmin(mini)
    free_fmin(mini)
    vec_free(x_vec)
    vec_free(step_vec)
    free(fn)
    return x_out, merr
