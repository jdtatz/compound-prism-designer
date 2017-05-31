import numba as nb
import numba.extending
import numpy as np
from functools import partial

__all__ = ["jit", "get_poly_coeffs", "nonlinearity", "nonlinearity_error", "beam_compression", "spectral_sampling_ratio"]

jit = partial(nb.jit, nopython=True, nogil=True, cache=False)

"""
Temporary Numpy overloads for Numba
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


@nb.extending.overload(np.tile)
def np_tile(arr, reps):
    if isinstance(arr, nb.types.Array) and arr.ndim == 1 and isinstance(reps, nb.types.Integer):
        def tile(a, r):
            out = np.empty(a.size * r, a.dtype)
            for i in range(0, out.size, a.size):
                out[i:i+a.size] = a
            return out
        return tile

"""
Utility Functions
"""


@jit
def get_poly_coeffs(g, n):
    """Get the polynomial coefficients and remainder when treating the input as a n-polynomial function"""
    x = np.linspace(1.0, -1.0, g.size)
    # poly = np.power.outer(x, np.arange(n + 1))
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
    # K = np.prod(ts[1::2], 0) / np.prod(ts[:-1:2], 0)
    # K = np.multiply.reduce(ts[1::2]) / np.multiply.reduce(ts[:-1:2])
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


def loadlibfunc(lib, name, ret, *args):
    func = getattr(lib, name)
    func.argtypes = args
    func.restype = ret
    return func
