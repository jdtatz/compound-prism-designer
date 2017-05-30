import numpy as np
from ctypes import CDLL, POINTER, c_void_p, c_int, c_double
from ctypes.util import find_library
from utils import loadlibfunc, jit
from itertools import repeat
import platform

"""
Load FitPack
"""

libc = CDLL(find_library("c"))
if platform.system() == 'Windows':
    libfit = CDLL("./fitpack/fitpack.dll")
else:
    libfit = CDLL("./fitpack/fitpack.so")

freal = c_double
fint = c_int
fpreal = POINTER(freal)
fpint = POINTER(fint)
npreal = np.dtype(freal)
npint = np.dtype(fint)

# curfit(iopt,m,x,y,w,xb,xe,k,s,nest,n,t,c,fp,wrk,lwrk,iwrk,ier)
curfit = loadlibfunc(libfit, 'curfit_', None, *repeat(c_void_p, 18))
# splev(t,n,c,k,x,y,m,e,ier)
splev = loadlibfunc(libfit, 'splev_', None, *repeat(c_void_p, 9))
# surfit(iopt,m,x,y,z,w,xb,xe,yb,ye,kx,ky,s,nxest,nyest,nmax,eps,nx,tx,ny,ty,c,fp,wrk1,lwrk1,wrk2,lwrk2,iwrk,kwrk,ier)
surfit = loadlibfunc(libfit, 'surfit_', None, *repeat(c_void_p, 30))
# bispev(tx,nx,ty,ny,c,kx,ky,x,mx,y,my,z,wrk,lwrk,iwrk,kwrk,ier)
bispev = loadlibfunc(libfit, 'bispev_', None, *repeat(c_void_p, 17))
# bispeu(tx,nx,ty,ny,c,kx,ky,x,y,z,m,wrk,lwrk, ier)
bispeu = loadlibfunc(libfit, 'bispeu_', None, *repeat(c_void_p, 14))


@jit
def create1d(iopt_, x_, y_, k_, s_):
    iopt = np.array([iopt_], npint)  # algo run type
    m = np.array([x_.size], npint)  # input size
    x = np.asfortranarray(x_).astype(npreal)  # x
    y = np.asfortranarray(y_).astype(npreal)  # y
    w = np.ones(x.size, npreal)  # weights of input
    xb = np.array([x.min()], npreal)  # min of input
    xe = np.array([x.max()], npreal)  # max of input
    k = np.array([k_], npint)  # order of interpolation
    s = np.array([s_], npreal)  # smoothness
    nest = k + m + 1  # over-estimate of the total number of knots
    n = np.empty(1, npint)  # number of knots
    t = np.empty(nest[0], npreal)  # knots
    c = np.empty(nest[0], npreal)  # coefficients
    fp = np.empty(1, npreal)  # weighted sum of squared residuals of the spline approximation
    wrk = np.empty((m * (k + 1) + nest * (7 + 3 * k))[0], npreal)  # workspace
    lwrk = np.array([wrk.size], npint)  # size of workspace
    iwrk = np.empty(nest[0], npint)  # workspace, but integers
    ier = np.empty(1, npint)  # errors
    curfit(iopt.ctypes.data, m.ctypes.data, x.ctypes.data, y.ctypes.data, w.ctypes.data, xb.ctypes.data, xe.ctypes.data, k.ctypes.data, s.ctypes.data, nest.ctypes.data, n.ctypes.data, t.ctypes.data, c.ctypes.data, fp.ctypes.data, wrk.ctypes.data, lwrk.ctypes.data, iwrk.ctypes.data, ier.ctypes.data)
    status = ier[0]
    if status == 1:
        raise Exception("required storage space exceeds the available storage space")
    elif status == 2:
        raise Exception("theoretically impossible result")
    elif status == 3:
        raise Exception("maximal number of iterations")
    elif status == 10:
        raise Exception("invalid input")
    return t[:n[0]], c, fp[0]


@jit
def create2d(iopt_, x_, y_, z_, kx_, ky_, s_):
    iopt = np.array([iopt_], npint)  # algo run type
    m = np.array([x_.size], npint)  # input size
    x = np.asfortranarray(x_).astype(npreal)  # x
    y = np.asfortranarray(y_).astype(npreal)  # y
    z = np.asfortranarray(z_).astype(npreal)  # z
    w = np.ones(x.size, npreal)  # weights of input
    xb = np.array([x.min()], npreal)  # min of input x
    xe = np.array([x.max()], npreal)  # max of input x
    yb = np.array([y.min()], npreal)  # min of input y
    ye = np.array([y.max()], npreal)  # max of input y
    kx = np.array([kx_], npint)  # order of interpolation x
    ky = np.array([ky_], npint)  # order of interpolation y
    s = np.array([s_], npreal)  # smoothness
    # over-estimate of the total number of knots for x
    nxest = np.maximum(2 * (kx + 1), kx + 1 + np.sqrt(m / 2)).astype(npint)
    # over-estimate of the total number of knots for y
    nyest = np.maximum(2 * (ky + 1), ky + 1 + np.sqrt(m / 2)).astype(npint)
    nmax = np.maximum(nxest, nyest)  # over-estimate of the total number of knots
    # threshold for determining the effective rank of an over-determined linear system of equations
    eps = np.array([1e-16], npreal)
    nx = np.empty(1, npint)  # number of knots for x
    tx = np.empty(nxest[0], npreal)  # knots for x
    ny = np.empty(1, npint)  # number of knots for y
    ty = np.empty(nyest[0], npreal)  # knots for y
    c = np.empty(((nxest - kx - 1) * (nyest - ky - 1))[0], npreal)  # coefficients
    fp = np.empty(1, npreal)  # weighted sum of squared residuals of the spline approximation
    # intermediate values
    u = nxest - kx - 1
    v = nyest - ky - 1
    km = np.maximum(kx, ky) + 1
    ne = np.maximum(nxest, nyest)
    bx = kx * v + ky + 1
    by = ky * u + kx + 1
    b1 = np.minimum(bx, by)
    b2 = b1 + ((v - ky) if bx[0] < by[0] else (u - kx))
    lwrk1 = u * v * (2 + b1 + b2) + 2 * (u + v + km * (m + ne) + ne - kx - ky) + b2 + 1  # size of workspace 1
    lwrk2 = u * v * (b2 + 1) + b2  # size of workspace 2
    wrk1 = np.empty(lwrk1[0], npreal)  # workspace 1
    wrk2 = np.empty(lwrk2[0], npreal)  # workspace 2
    kwrk = m + (nxest - 2 * kx - 1) * (nyest - 2 * ky - 1)  # size of integer workspace
    iwrk = np.empty(kwrk[0], npint)  # workspace, but integers
    ier = np.empty(1, npint)  # errors
    surfit(iopt.ctypes.data, m.ctypes.data, x.ctypes.data, y.ctypes.data, z.ctypes.data, w.ctypes.data, xb.ctypes.data, xe.ctypes.data, yb.ctypes.data, ye.ctypes.data, kx.ctypes.data, ky.ctypes.data, s.ctypes.data, nxest.ctypes.data, nyest.ctypes.data, nmax.ctypes.data, eps.ctypes.data, nx.ctypes.data, tx.ctypes.data, ny.ctypes.data, ty.ctypes.data, c.ctypes.data, fp.ctypes.data, wrk1.ctypes.data, lwrk1.ctypes.data, wrk2.ctypes.data, lwrk2.ctypes.data, iwrk.ctypes.data, kwrk.ctypes.data, ier.ctypes.data)
    status = ier[0]
    if status == 1:
        raise Exception("required storage space exceeds the available storage space")
    elif status == 2:
        raise Exception("theoretically impossible result")
    elif status == 3:
        raise Exception("maximal number of iterations")
    elif status == 4:
        raise Exception("no more knots can be added, either s or m too small")
    elif status == 5:
        raise Exception("no more knots can be added, s too small or too large a weight to an inaccurate data point")
    elif status == 10:
        raise Exception("invalid input")
    elif status > 10:
        raise Exception("lwrk2 is too small")
    return tx[:nx[0]], ty[:ny[0]], c[:((nx - kx - 1) * (ny - ky - 1))[0]], fp[0]


@jit
def eval1d(x_, t_, c_, k_):
    t = np.asfortranarray(t_)  # knots
    n = np.array([t.size], npint)  # number of knots
    c = np.asfortranarray(c_)  # coefficients
    k = np.array([k_], npint)  # order of interpolation
    x = np.asfortranarray(x_).astype(npreal)  # values to be interpolated
    y = np.empty_like(x)  # interpolated values
    m = np.array([x.size], npint)  # input size
    e = np.zeros(1, npint)  # extrapolation types
    ier = np.empty(1, npint)  # errors
    splev(t.ctypes.data, n.ctypes.data, c.ctypes.data, k.ctypes.data, x.ctypes.data, y.ctypes.data, m.ctypes.data, e.ctypes.data, ier.ctypes.data)
    if ier[0] == 0:
        return y
    raise Exception("invalid input data")


@jit
def eval2d_grid(x_, y_, tx_, ty_, c_, kx_, ky_):
    tx = np.asfortranarray(tx_)  # knots in x
    nx = np.array([tx.size], npint)  # number of knots in x
    ty = np.asfortranarray(ty_)  # knots in y
    ny = np.array([ty.size], npint)  # number of knots in y
    c = np.asfortranarray(c_)  # coefficients
    kx = np.array([kx_], npint)  # order of interpolation in x
    ky = np.array([ky_], npint)  # order of interpolation in y
    x = np.asfortranarray(x_).astype(npreal)  # values to be interpolated
    mx = np.array([x.size], npint)  # input size
    y = np.asfortranarray(y_).astype(npreal)  # values to be interpolated
    my = np.array([y.size], npint)  # input size
    z = np.asfortranarray(np.empty((x.size, y.size), npreal))  # interpolated values
    lwrk = mx * (kx + 1) + my * (ky + 1)  # real workspace size
    wrk = np.empty(lwrk[0], npreal)  # real workspace
    kwrk = mx + my  # integer workspace size
    iwrk = np.empty(kwrk[0], npreal)  # integer workspace
    ier = np.empty(1, npint)  # errors
    bispev(tx.ctypes.data, nx.ctypes.data, ty.ctypes.data, ny.ctypes.data, c.ctypes.data, kx.ctypes.data, ky.ctypes.data, x.ctypes.data, mx.ctypes.data, y.ctypes.data, my.ctypes.data, z.ctypes.data, wrk.ctypes.data, lwrk.ctypes.data, iwrk.ctypes.data, kwrk.ctypes.data, ier.ctypes.data)
    if ier[0] == 0:
        return z.T
    raise Exception("invalid input data")


@jit
def eval2d_pts(x_, y_, tx_, ty_, c_, kx_, ky_):
    tx = np.asfortranarray(tx_)  # knots in x
    nx = np.array([tx.size], npint)  # number of knots in x
    ty = np.asfortranarray(ty_)  # knots in y
    ny = np.array([ty.size], npint)  # number of knots in y
    c = np.asfortranarray(c_)  # coefficients
    kx = np.array([kx_], npint)  # order of interpolation in x
    ky = np.array([ky_], npint)  # order of interpolation in y
    x = np.asfortranarray(x_).astype(npreal)  # values to be interpolated
    y = np.asfortranarray(y_).astype(npreal)  # values to be interpolated
    m = np.array([x.size], npint)  # input size
    z = np.empty(x.size, npreal)  # interpolated values
    lwrk = kx + ky + 2  # real workspace size
    wrk = np.empty(lwrk[0], npreal)  # real workspace
    ier = np.empty(1, npint)  # errors
    bispeu(tx.ctypes.data, nx.ctypes.data, ty.ctypes.data, ny.ctypes.data, c.ctypes.data, kx.ctypes.data, ky.ctypes.data, x.ctypes.data, y.ctypes.data, z.ctypes.data, m.ctypes.data, wrk.ctypes.data, lwrk.ctypes.data, ier.ctypes.data)
    if ier[0] == 0:
        return z
    raise Exception("invalid input data")


"""
if __name__ == "__main__":
    print('verifying fitpack ctypes layer')
    try:
        import scipy.interpolate
        x = np.arange(21)
        y = np.linspace(0, 10, 21)
        t, c, fp = create1d(0, x, y, 3, 0)
        spline = scipy.interpolate.InterpolatedUnivariateSpline(x, y)
        xi = np.array([4, 4.5, 4.7])
        yo = eval1d(xi, t, c, 3)
        ys = spline(xi)
        assert np.allclose(yo, ys)
        count = 20
        x2 = np.random.rand(count)
        y2 = np.random.rand(count)
        z2 = np.random.rand(count)
        x2.sort(), y2.sort()
        tx, ty, c, fp = create2d(0, x2, y2, z2, 3, 3, count)
        bs = scipy.interpolate.SmoothBivariateSpline(x2, y2, z2)
        xo = np.random.rand(10)
        yo = np.random.rand(10)
        xo.sort(), yo.sort()
        d = eval2d_grid(xo, yo, tx, ty, c, 3, 3)
        d2 = bs(xo, yo)
        assert np.allclose(d, d2)
        d = eval2d_pts(xo, yo, tx, ty, c, 3, 3)
        d2 = bs(xo, yo, grid=False)
        assert np.allclose(d, d2)
        print('verified')
    except ImportError:
        print("scipy not installed")
"""

"""
PMT Code
"""


@jit
def power(waves, values, bounds, interp_pw2f):
    powered = np.zeros((len(bounds), len(waves)), np.float64)
    offset = np.array([1, -1], np.float64)
    tx, ty, c = interp_pw2f
    for i in range(len(bounds)):
        bound = bounds[i]
        l, ls, lt, rt, rs, r = bound
        lrs = np.array([l, r], np.float64)
        wps = np.abs(offset - eval2d_grid(lrs, waves, tx, ty, c, 3, 3)) / 2
        for j in range(len(waves)):
            w, val, wp = waves[j], values[j], wps[j]
            if lt <= w <= rt:
                powered[i, j] = val
            elif ls <= w <= rs:
                powered[i, j] = val * np.min(wp)
    return powered


@jit
def get_bounds(bins, fields, waves, positions):
    fbins = bins.flatten()
    bounds = np.empty((len(bins), 4), np.float64)
    ptw = np.empty((len(bins), 2, fields.size), np.float64)
    for i in range(fields.size):
        p2w = positions[i * waves.size:(i + 1) * waves.size]
        order = np.argsort(p2w)
        x, y = p2w[order], waves[order]
        t, c, res = create1d(0, x, y, 3, 0)
        wbins = eval1d(fbins, t, c, 3)
        ptw[:, :, i] = wbins.reshape((-1, 2))
    for i in range(len(bounds)):
        ls = np.min(ptw[i])
        lt = np.max(np.minimum(ptw[i, 0], ptw[i, 1]))
        rt = np.min(np.maximum(ptw[i, 0], ptw[i, 1]))
        rs = np.max(ptw[i])
        bounds[i] = (ls, lt, rt, rs)
    """
    bounds1 = np.transpose((
        np.minimum.reduce(np.minimum.reduce(ptw, 1), 1),
        np.maximum.reduce(np.minimum.reduce(ptw, 1), 1),
        np.minimum.reduce(np.maximum.reduce(ptw, 1), 1),
        np.maximum.reduce(np.maximum.reduce(ptw, 1), 1)
    ))
    assert np.all(bounds == bounds1)
    """
    return bounds


@jit
def pmt(bins, rays):
    fields = np.linspace(-1.0, 1.0, 101)
    # waves = np.empty(21, np.float64)
    waves = np.arange(500, 821, 16).astype(np.float64)
    # rays = np.empty(len(fields) * len(waves), np.float64)
    bounds = get_bounds(bins, fields, waves, rays)
    ls, lt, rt, rs = bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]
    som_in = np.abs(rs - ls)
    tot_in = np.abs(rt - lt)
    ideal = waves.max() - waves.min()
    print(np.sum(tot_in), np.sum(som_in), ideal)
    # twaves = np.tile(waves, fields.size)
    # tfields = np.tile(fields, waves.size)
    # tx, ty, c, res = create2d(0, rays, twaves, tfields, 3, 3, rays.size)
    return np.sum(som_in) / ideal, np.sum(tot_in) / ideal, np.sum(tot_in) / np.sum(som_in)


data = np.load("data.npy")
binLabels = list(range(32, 0, -1))
bbins = np.array([(b + 0.1, b + 0.9) for b in range(0, 32)], np.float64)
merit = pmt(bbins, data[:, 2])
print(merit)
