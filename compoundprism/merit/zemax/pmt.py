import math
import numpy as np
import ctypes
import os
from ctypes import c_void_p, c_uint64
from compoundprism.utils import loadlibfunc, jit
from collections import OrderedDict
from compoundprism.fitpack import *
from compoundprism.merit.registry import MeritFunctionRegistry
import numba as nb


class ZemaxPMTBase:
    def __init__(self, settings):
        base = os.path.join(os.path.dirname(__file__), 'prism_compat_0p4_NA_125um.ZMX')
        zemax_file = settings.get("zemax file", base).encode("UTF-8")
        self.fields = settings.get("zemax fields", np.linspace(-1.0, 1.0, 101))
        self.bins = settings.get("zemax bins",  np.array([(b + 0.1, b + 0.9) for b in range(0, 32)], np.float64))
        self.height = settings.get("zemax height", 50)
        if 'zos' not in globals():
            global zos, wavesInSystem, pooledRayTraceSystem, optimzeSystem, changeSystem
            zos = ctypes.WinDLL(os.path.join(os.path.dirname(__file__), 'ZosLibrary.dll'))
            wavesInSystem = loadlibfunc(zos, 'wavesInSystem', c_uint64, c_void_p, c_void_p)
            pooledRayTraceSystem = loadlibfunc(zos, 'pooledRayTraceSystem', None, c_void_p, c_void_p, c_void_p, c_uint64)
            optimzeSystem = loadlibfunc(zos, 'optimzeSystem', None, c_void_p)
            changeSystem = loadlibfunc(zos, 'changeSystem', None, c_void_p, c_void_p, c_void_p, c_void_p)
        print('connecting to Zemax OpticStudio')
        self.conn = zos.connectZOS()
        if self.conn == 0:
            raise ConnectionError("Failed to connect")
        print('connected')
        self.sys = zos.getPrimarySystem(self.conn)
        zos.loadFile(self.sys, zemax_file)

    def configure(self, configuration):
        zos.setupSystem(self.sys, configuration["prism_count"])
        return {**configuration, 'bins': self.bins, 'fields': self.fields, 'height': self.height}

    def configure_thread_model(self, model, tid):
        return model._replace(sys=zos.copySystem(self.sys))


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


@MeritFunctionRegistry.register
class PMTCoverageMerit(ZemaxPMTBase):
    """
    How much of the light is captured
    """
    name = 'pmt coverage'
    weights = OrderedDict(tot_per=1.0, som_per=1.0, tot_som=0.5)
    model_params = OrderedDict(sys=nb.int64, bins=nb.types.Array(nb.float64, 2, 'C'), fields=nb.types.Array(nb.float64, 1, 'C'), height=np.float64)

    @staticmethod
    def merit(model, angles, delta, thetas):
        waves, rays = rayTrace(model.sys, model.glass, angles, model.height, model.fields)
        bounds = get_bounds(model.bins, model.fields, waves, rays)
        ls, lt, rt, rs = bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]
        som_in = np.abs(rs - ls)
        tot_in = np.abs(rt - lt)
        ideal = waves.max() - waves.min()
        merr = model.weights.tot_per * np.sum(tot_in) / ideal
        merr += model.weights.som_per * np.sum(som_in) / ideal
        merr += model.weights.tot_som * np.sum(tot_in) / np.sum(som_in)
        return merr


@MeritFunctionRegistry.register
class PMTSpreadMerit(ZemaxPMTBase):
    """
    How spread out are the dyes along the pmt
    """
    name = 'pmt dye spread'
    weights = OrderedDict(spread=1.0)
    model_params = OrderedDict(sys=nb.int64, bins=nb.types.Array(nb.float64, 2, 'C'), fields=nb.types.Array(nb.float64, 1, 'C'), height=np.float64, dyes=nb.types.Array(nb.float64, 3, 'C'))

    def __init__(self, settings):
        super().__init__(settings)
        dyeFile = settings.get('zemax dye file', os.path.join(os.path.dirname(__file__), 'test_dyes.npy'))
        self.dyes = np.load(dyeFile)

    def configure(self, configuration):
        config = super().configure(configuration)
        return {**config, 'dyes': self.dyes}

    @staticmethod
    def merit(model, angles, delta, thetas):
        waves, rays = rayTrace(model.sys, model.materials, angles, model.height, model.fields)
        bounds = get_bounds(model.bins, model.fields, waves, rays)
        twaves = np.tile(waves, model.fields.size)
        tfields = np.tile(model.fields, waves.size)
        tx, ty, c, res = create2d(0, rays, twaves, tfields, 3, 3, rays.size)
        spread = np.zeros(model.dyes.shape[0])
        for i in range(model.dyes.shape[0]):
            waves, emission = model.dyes[i, 0], model.dyes[i, 1]
            p = power(waves, emission, bounds, model.bins, tx, ty, c)
            for j in range(model.bins.shape[0]):
                if p[j].nonzero()[0].size:
                    spread[i] += 1
        return model.weights.spread * spread


@MeritFunctionRegistry.register
class PMTCrossTalkMerit(ZemaxPMTBase):
    """
    How much the dyes intersect along the pmt
    """
    name = 'pmt dye cross talk'
    weights = OrderedDict(cross=0.5)
    model_params = OrderedDict(sys=nb.int64, bins=nb.types.Array(nb.float64, 2, 'C'), fields=nb.types.Array(nb.float64, 1, 'C'), height=np.float64, dyes=nb.types.Array(nb.float64, 3, 'C'))

    def __init__(self, settings):
        super().__init__(settings)
        dyeFile = settings.get('zemax dye file', os.path.join(os.path.dirname(__file__), 'test_dyes.npy'))
        self.dyes = np.load(dyeFile)

    def configure(self, configuration):
        config = super().configure(configuration)
        return {**config, 'dyes': self.dyes}

    @staticmethod
    def merit(model, angles, delta, thetas):
        waves, rays = rayTrace(model.sys, model.materials, angles, model.height, model.fields)
        bounds = get_bounds(model.bins, model.fields, waves, rays)
        twaves = np.tile(waves, model.fields.size)
        tfields = np.tile(model.fields, waves.size)
        tx, ty, c, res = create2d(0, rays, twaves, tfields, 3, 3, rays.size)
        binned = np.empty((model.dyes.shape[0], model.bins.shape[0]), np.float64)
        for i in range(model.dyes.shape[0]):
            waves, emission = model.dyes[i, 0], model.dyes[i, 1]
            p = power(waves, emission, bounds, model.bins, tx, ty, c)
            for j in range(model.bins.shape[0]):
                binned[i, j] = p[j].sum() / max(p[j].nonzero()[0].size, 1)
        cross = np.zeros((model.dyes.shape[0], model.dyes.shape[0]), np.float64)
        for i in range(model.dyes.shape[0]):
            for j in range(model.dyes.shape[0]):
                if i != j:
                    cross[i, j] = cross[j, i] = binned[i] @ binned[j]
        return cross.sum() * model.weights.cross


if __name__ == "__main__":
    from ctypes import cast, c_char_p
    from collections import namedtuple
    g1 = b'N-FK58'
    g2 = b'P-SF68'
    g3 = b'LF5'

    a1 = 102.522545748073
    a2 = -48.2729540140781
    a3 = 17.502053458746

    materials = (g1, g2, g3, g2, g1)
    alphas = np.array((a1, a2, a3, a2, a1)) * np.pi / 180
    count = len(materials)

    mats = (c_char_p*count)(*materials)
    materialsPtr = cast(mats, c_void_p).value

    dyes = np.zeros((4, 2, 100), np.float64)
    dyes[:, 0] = np.linspace(500, 820, dyes.shape[2])
    dyes[:, 1, 20:80] = 100*np.random.rand(dyes.shape[0], 60)
    dyes[3, 1] = 0
    dyes[3, 1, 81:] = 100
    np.save("test_dyes", dyes)

    Model = namedtuple("Model", "sys, materials, height, bins, fields, dyes")
    base_model = Model(None, materialsPtr, None, None, None, None)
    settings = {}

    Merits = [PMTCoverageMerit, PMTSpreadMerit, PMTCrossTalkMerit]
    for Merit in Merits:
        m = Merit(settings)
        m.configure({"prism_count": 3})
        model = m.configure_thread_model(base_model, 0)
        merit_error = m.merit(m, alphas, None, None)
        print(Merit, merit_error)
