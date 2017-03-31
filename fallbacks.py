import numba
import numpy as np
from functools import partial
"""
Due to numba's incompleteness both numpy.gradient and scipy.optimize.fmin,
 need to be rewritten to work.
"""
jit = partial(numba.jit, nopython=True, nogil=True, cache=True)


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


def setup_nelder_mead(func, _maxiter=200, _maxfun=200):
    """
    function factory that replaces scipy.optimize.fmin,
    the factory is necessary due to numba limitations and
    the reimplementation is necessary due to numba not yet supporting scipy
    """
    @jit
    def nelder_mead(x0: np.ndarray, *args):
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5
        nonzdelt = 0.05
        zdelt = 0.00025
        xatol = 1e-4
        fatol = 1e-4

        x0.flatten()
        count = len(x0)
        ncalls = 0

        sim = np.zeros((count + 1, count), dtype=x0.dtype)
        sim[0] = x0
        for k in range(count):
            y = x0.copy()
            if y[k] != 0:
                y[k] *= (1 + nonzdelt)
            else:
                y[k] = zdelt
            sim[k + 1] = y

        maxiter = count * _maxiter
        maxfun = count * _maxfun

        one2np1 = list(range(1, count + 1))
        fsim = np.zeros((count + 1,))

        for k in range(count + 1):
            fsim[k] = func(sim[k], *args)
            ncalls += 1

        ind = np.argsort(fsim)
        sim = sim[ind]
        fsim = fsim[ind]
        iterations = 1
        while ncalls < maxfun and iterations < maxiter:
            if np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol:
                break

            # xbar = np.add.reduce(sim[:-1], 0) / N
            xbar = np.zeros(sim[0].size)
            for s in range(len(sim[:-1][0])):
                xbar += sim[:-1][s]
            xbar /= count

            xr = (1 + rho) * xbar - rho * sim[-1]
            fxr = func(xr, *args)
            ncalls += 1
            doshrink = False

            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                fxe = func(xe, *args)
                ncalls += 1
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        fxc = func(xc, *args)
                        ncalls += 1
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = True
                    else:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        fxcc = func(xcc, *args)
                        ncalls += 1
                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = True
                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            fsim[j] = func(sim[j], *args)
                            ncalls += 1
            ind = np.argsort(fsim)
            sim = sim[ind]
            fsim = fsim[ind]
            iterations += 1
        return sim[0]
    return nelder_mead


if __name__ == "__main__":
    from scipy.optimize import fmin


    @jit
    def rosenbrock(x0, a, b):
        return (a - x0[0]) ** 2 + b * (x0[1] - x0[0] ** 2) ** 2

    x = np.array([2.0, 5.0])

    nelder_mead = setup_nelder_mead(rosenbrock)
    assert np.allclose(fmin(rosenbrock, x, args=(1, 100), disp=False), nelder_mead(x, 1, 100))
    print(fmin(rosenbrock, x, args=(1, 100), disp=False))
    print(nelder_mead(x, 1, 100))

    rand = np.random.randn(100)
    assert np.allclose(np.gradient(rand), gradient(rand))
