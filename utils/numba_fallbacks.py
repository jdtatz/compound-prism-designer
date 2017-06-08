import numpy as np
import numba as nb
import numba.extending

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
