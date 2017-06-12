import math
import numpy as np
import ctypes
from ctypes import cast, c_void_p, c_uint64, c_char_p
from compoundprism.utils import loadlibfunc, jit
from collections import namedtuple, OrderedDict
from compoundprism.fitpack import *
from compoundprism.merit.registry import BaseMeritFunction
import numba as nb


def start_zemax(count, zemax_file):
    if 'zos' not in globals():
        global zos, wavesInSystem, pooledRayTraceSystem, optimzeSystem, changeSystem
        zos = ctypes.WinDLL("ZosLibrary.dll")
        wavesInSystem = loadlibfunc(zos, 'wavesInSystem', c_uint64, c_void_p, c_void_p)
        pooledRayTraceSystem = loadlibfunc(zos, 'pooledRayTraceSystem', None, c_void_p, c_void_p, c_void_p, c_uint64)
        optimzeSystem = loadlibfunc(zos, 'optimzeSystem', None, c_void_p)
        changeSystem = loadlibfunc(zos, 'changeSystem', None, c_void_p, c_void_p, c_void_p, c_void_p)
    print('connecting')
    conn = zos.connectZOS()
    if conn == 0:
        raise Exception("Failed to connect")
    print('connected')
    sys = zos.getPrimarySystem(conn)
    zos.loadFile(sys, zemax_file)
    print("Files Loaded")
    zos.setupSystem(sys, count)
    print('system setup')
    return sys


@jit
def rayTrace(sys, materials, alphas, h, fields):
    # convert Alphas
    count = len(alphas)
    ytans = np.empty(count+1, dtype=np.float64)
    thickness = np.empty(count, dtype=np.float64)
    for i in range(count):
        mid = count // 2
        beta = alphas[mid] / 2
        if i < mid:
            beta += np.sum(alphas[i:mid])
        elif i > mid:
            beta += np.sum(alphas[mid+1:i+1])
        gamma = alphas[i] - beta
        ytanbeta = math.tan(beta)
        ytangamma = math.tan(gamma)
        thickness[i] = ((h / 2) * (abs(ytanbeta) + abs(ytangamma)))
        if i <= mid:
            ytans[i] = (ytanbeta)
        if i >= mid:
            ytans[i+1] = (-ytanbeta)
    # Change system
    changeSystem(sys, materials, thickness.ctypes.data, ytans.ctypes.data)
    thickness[0] = ytans[0]
    # Optimize changed system
    optimzeSystem(sys)
    # Get Wave Data
    waveData = np.empty(25, dtype=np.float64)
    c = wavesInSystem(sys, waveData.ctypes.data)
    waveData = waveData[:c]*1000
    # Get Ray Data
    rays = np.empty(len(fields)*21, dtype=np.float64)
    pooledRayTraceSystem(sys, rays.ctypes.data, fields.ctypes.data, len(fields))
    return waveData, rays + 16


@jit
def power(waves, values, bounds, bins, tx, ty, c):
    powered = np.zeros((len(bounds), len(waves)), np.float64)
    for i in range(bounds.shape[0]):
        spots = np.digitize(waves, bounds[i])
        sl = np.where(spots == 1)[0]
        sr = np.where(spots == 3)[0]
        sin = np.where(spots == 2)[0]
        powered[i][sin] = values[sin]
        if sl.size:
            l = np.array(bins[i, 1])
            powered[i][sl] = values[sl] * np.abs(-1 - eval2d_grid(l, waves[sl], tx, ty, c, 3, 3)) / 2.0
        if sr.size:
            r = np.array(bins[i, 0])
            powered[i][sr] = values[sr] * np.abs(1 - eval2d_grid(r, waves[sr], tx, ty, c, 3, 3)) / 2.0
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
    for i in range(bounds.shape[0]):
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


class PMTMerit(BaseMeritFunction):
    name = 'pmt'
    weights = OrderedDict(tot_per=1.0, som_per=1.0, tot_som=0.5)
    model_params = OrderedDict(sys=nb.int64, bins=nb.types.Array(nb.float64, 2, 'C'), fields=nb.types.Array(nb.float64, 1, 'C'))

    def __init__(self, settings):
        self.sys = start_zemax(3, b"prism_compat_0p4_NA_125um.ZMX")

    def configure(self, configuration):
        new = {'bins': np.array([(b + 0.1, b + 0.9) for b in range(0, 32)], np.float64), 'fields': np.linspace(-1.0, 1.0, 101)}
        return {**configuration, **new}

    def configure_thread_model(self, model, tid):
        return model._replace(sys=zos.copySystem(self.sys))

    @staticmethod
    def merit(model, angles, delta, thetas):
            waves, rays = rayTrace(model.sys, model.glass, angles, 50, model.fields)
            bounds = get_bounds(model.bins, model.fields, waves, rays)
            ls, lt, rt, rs = bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]
            som_in = np.abs(rs - ls)
            tot_in = np.abs(rt - lt)
            ideal = waves.max() - waves.min()
            merr = model.weights.tot_per * np.sum(tot_in) / ideal
            merr += model.weights.som_per * np.sum(som_in) / ideal
            merr += model.weights.tot_som * np.sum(tot_in) / np.sum(som_in)
            return merr
    

@jit
def pmt_coverage(model, angles):
    """
    How much of the light is captured
    """
    fields = np.linspace(-1.0, 1.0, 101)
    waves, rays = rayTrace(model.sys, model.materials, angles, model.height, fields)
    bounds = get_bounds(model.bins, fields, waves, rays)
    ls, lt, rt, rs = bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]
    som_in = np.abs(rs - ls)
    tot_in = np.abs(rt - lt)
    ideal = waves.max() - waves.min()
    print(np.sum(tot_in), np.sum(som_in), ideal)
    return np.sum(som_in) / ideal, np.sum(tot_in) / ideal, np.sum(tot_in) / np.sum(som_in)


@jit
def pmt_spread(model, angles):
    """
    How spread out are the dyes along the pmt
    """
    bins, dyes = model.bins, model.dyes
    fields = np.linspace(-1.0, 1.0, 101)
    waves, rays = rayTrace(model.sys, model.materials, angles, model.height, fields)
    bounds = get_bounds(bins, fields, waves, rays)
    twaves = np.tile(waves, fields.size)
    tfields = np.tile(fields, waves.size)
    tx, ty, c, res = create2d(0, rays, twaves, tfields, 3, 3, rays.size)
    spread = np.zeros(dyes.shape[0])
    for i in range(dyes.shape[0]):
        waves, emission = dyes[i, 0], dyes[i, 1]
        p = power(waves, emission, bounds, bins, tx, ty, c)
        for j in range(bins.shape[0]):
            if p[j].nonzero()[0].size:
                spread[i] += 1
    return spread


@jit
def pmt_cross_talk(model, angles):
    """
    How much the dyes intersect
    """
    bins, dyes = model.bins, model.dyes 
    fields = np.linspace(-1.0, 1.0, 101)
    waves, rays = rayTrace(model.sys, model.materials, angles, model.height, fields)
    bounds = get_bounds(bins, fields, waves, rays)
    twaves = np.tile(waves, fields.size)
    tfields = np.tile(fields, waves.size)
    tx, ty, c, res = create2d(0, rays, twaves, tfields, 3, 3, rays.size)
    binned = np.empty((dyes.shape[0], bins.shape[0]), np.float64)
    for i in range(dyes.shape[0]):
        waves, emission = dyes[i, 0], dyes[i, 1]
        p = power(waves, emission, bounds, bins, tx, ty, c)
        for j in range(bins.shape[0]):
            binned[i, j] = p[j].sum() / max(p[j].nonzero()[0].size, 1)
    cross = np.empty((dyes.shape[0], dyes.shape[0]), np.float64)
    for i in range(dyes.shape[0]):
        for j in range(dyes.shape[0]):
            cross[i, j] = cross[j, i] = binned[i] @ binned[j]
    return cross


if __name__ == "__main__":
    g1 = b'N-FK58'
    g2 = b'P-SF68'
    g3 = b'LF5'

    a1 = 102.522545748073
    a2 = -48.2729540140781
    a3 = 17.502053458746

    h = 50
    materials = (g1, g2, g3, g2, g1)
    alphas = np.array((a1, a2, a3, a2, a1)) * np.pi / 180
    count = len(materials)

    zemaxFile = b"prism_compat_0p4_NA_125um.ZMX"

    sys = start_zemax(count, zemaxFile)

    mats = (c_char_p*count)(*materials)
    materialsPtr = cast(mats, c_void_p).value

    binLabels = list(range(32, 0, -1))
    bbins = np.array([(b + 0.1, b + 0.9) for b in range(0, 32)], np.float64)
    dyes = np.zeros((4, 2, 100), np.float64)
    dyes[:, 0] = np.linspace(500, 820, dyes.shape[2])
    dyes[:, 1, 20:80] = 100*np.random.rand(dyes.shape[0], 60)
    dyes[3, 1] = 0
    dyes[3, 1, 81:] = 100

    Model = namedtuple("Model", "sys, materials, height, bins, dyes")
    model = Model(sys, materialsPtr, h, bbins, dyes)

    cov = pmt_coverage(model, alphas)
    print(cov)
    pspread = pmt_spread(model, alphas)
    print(pspread)
    cross = pmt_cross_talk(model, alphas)
    print(cross)

