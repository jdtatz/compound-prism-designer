import numba as nb
import numpy as np
from utils import *
from collections import namedtuple, OrderedDict

"""
Merit Functions
"""


def linearity(model, angles, delta_spectrum, thetas):
    """Linearity merit function"""
    NL_err = model.weights.NL * nonlinearity_error(delta_spectrum)
    return NL_err


def beam_comp(model, angles, delta_spectrum, thetas):
    """Beam Compression and Linearity merit function"""
    NL_err = model.weights.NL * nonlinearity_error(delta_spectrum)
    K = beam_compression(thetas, model.nwaves)
    K_err = model.weights.K * (K - 1.0) ** 2
    return NL_err + K_err


def chromaticity(model, angles, delta_spectrum, thetas):
    """Chromaticity merit function"""
    chromat = model.weights.chromat * np.abs(delta_spectrum.max() - delta_spectrum.min())
    return chromat


def thinness(model, angles, delta_spectrum, thetas):
    """Thinness merit function"""
    anglesum = model.weights.anglesum * model.deltaT_target * np.sum(angles ** 2)
    return anglesum


def second_order(model, angles, delta_spectrum, thetas):
    """Second Order Error merit function"""
    coeffs, remainder = get_poly_coeffs(delta_spectrum, 2)
    secondorder = model.weights.secondorder / coeffs[2] ** 2
    return secondorder


merit_funcs = {'linearity': linearity, 'beam compression': beam_comp, 'chromaticity': chromaticity,
               'thinness': thinness, 'second-order': second_order}

"""
Prism Code
"""


@jit
def snells(model, angles):
    thetas = np.empty((2 * model.prism_count + 3, model.nwaves))
    alphas = angles[model.glass_indices]
    beta = np.sum(alphas[:model.prism_count // 2]) + alphas[model.prism_count // 2] / 2
    gamma = np.sum(alphas[model.prism_count // 2 + 1:]) + alphas[model.prism_count // 2] / 2
    ns = model.n[model.glass_indices]

    thetas[0] = model.theta0 + beta  # theta 1
    np.arcsin((1.0 / ns[0]) * np.sin(model.theta0 + beta), thetas[1])  # theta 1 prime
    for i in range(1, model.prism_count + 1):
        thetas[2 * i] = thetas[2 * i - 1] - alphas[i - 1]  # theta i
        if i < model.prism_count:
            np.arcsin(ns[i - 1] / ns[i] * np.sin(thetas[2 * i]), thetas[2 * i + 1])  # theta i prime
        else:
            np.arcsin(ns[i - 1] * np.sin(thetas[2 * i]), thetas[2 * i + 1])  # theta (n-1) prime
    thetas[-1] = thetas[-2] + gamma  # theta n

    delta_spectrum = model.theta0 - thetas[-1]

    return delta_spectrum, thetas


@jit
def merit_error(model, angles):
    delta_spectrum, thetas = snells(model, angles)
    # If TIR occurs in the design (producing NaNs in the spectrum), then give a
    # hard error: return a large error which has nothing to do with the (invalid)
    # performance data.
    if np.any(np.isnan(delta_spectrum)):
        return model.weights.tir * np.sum(np.isnan(delta_spectrum))
    # enforces valid solution
    alphas = angles[model.glass_indices]
    refracted = thetas[model.refracted_indices]
    refs = np.abs(alphas) + refracted.T * np.sign(alphas)
    merit_err = model.weights.valid * np.sum(np.greater(refs, np.pi / 2)) / (model.prism_count * model.nwaves)
    # critical angle prevention
    incident = thetas[model.incident_indices]
    merit_err += model.weights.crit_angle * np.sum(np.greater(np.abs(incident), model.angle_limit)) / model.nwaves
    # Punish if the prism gets too small to be usable
    too_thin = np.abs(angles) - 1.0
    too_thin[np.where(np.abs(angles) > np.pi / 180.0)[0]] = 0.0
    merit_err += model.weights.thin * np.sum(too_thin ** 2)
    # deltaC and deltaT errors
    merit_err += model.weights.deviation * (delta_spectrum[model.nwaves // 2] - model.deltaC_target) ** 2
    merit_err += model.weights.dispersion * ((delta_spectrum.max() - delta_spectrum.min()) - model.deltaT_target) ** 2

    return merit_err + merit(model, angles, delta_spectrum, thetas)


@jit
def minimizer(model):
    rho = 1
    chi = 2
    psi = 0.5
    sigma = 0.5
    nonzdelt = 0.05
    zdelt = 0.00025
    xatol = 1e-4
    fatol = 1e-4

    count = model.glass_count
    ncalls = 0

    sim = np.zeros((count + 1, count), dtype=np.float64)
    sim[0] = model.initial_angles
    for k in range(count):
        y = model.initial_angles.copy()
        if y[k] != 0:
            y[k] *= (1 + nonzdelt)
        else:
            y[k] = zdelt
        sim[k + 1] = y

    maxiter = count * 200
    maxfun = count * 200

    one2np1 = list(range(1, count + 1))
    fsim = np.zeros((count + 1,))

    for k in range(count + 1):
        fsim[k] = merit_error(model, sim[k])
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
        fxr = merit_error(model, xr)
        ncalls += 1
        doshrink = False

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = merit_error(model, xe)
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
                    fxc = merit_error(model, xc)
                    ncalls += 1
                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = True
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = merit_error(model, xcc)
                    ncalls += 1
                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = True
                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = merit_error(model, sim[j])
                        ncalls += 1
        ind = np.argsort(fsim)
        sim = sim[ind]
        fsim = fsim[ind]
        iterations += 1
    return sim[0], fsim[0]


def prism(model):
    angles, merr = minimizer(model)
    delta_spectrum, thetas = snells(model, angles)
    if np.any(np.isnan(thetas)):
        return

    (delta0, delta1, delta2), remainder = get_poly_coeffs(delta_spectrum, 2)
    deltaM = np.mean(delta_spectrum) * 180.0 / np.pi
    deltaC = delta_spectrum[model.nwaves // 2] * 180.0 / np.pi
    deltaT = (delta_spectrum.max() - delta_spectrum.min()) * 180.0 / np.pi
    alphas = angles * 180.0 / np.pi
    NL = 10000.0 * nonlinearity(delta_spectrum)
    SSR = spectral_sampling_ratio(model.w, delta_spectrum, model.sampling_domain_is_wavenumber)
    K = beam_compression(thetas, model.nwaves)
    _, remainder = get_poly_coeffs(delta_spectrum, 1)
    dw = np.abs(np.mean(np.gradient(model.w)))
    nonlin = np.sqrt(remainder) * dw * 180.0 / np.pi
    chromat = 100.0 * abs(delta_spectrum.max() - delta_spectrum.min()) * 180.0 / np.pi
    delta1 *= 2.0 * 180.0 / np.pi
    delta2 *= 2.0 * 180.0 / np.pi
    # T = transmission(thetas, n, nwaves)

    return alphas, merr, deltaC, deltaT, NL, SSR, K, deltaM, delta1, delta2, nonlin, chromat

base = {'tir': 0.1, 'valid': 1.0, 'crit_angle': 1.0, 'thin': 0.25, 'deviation': 1.0, 'dispersion': 1.0}

default = {
    'linearity': {'NL': 25},
    'beam compression': {'NL': 25, 'K': 0.0025},
    'chromaticity': {'chromat': 1, 'dispersion': 0},
    'thinness': {'anglesum': 0.001},
    'second-order': {'secondorder': 0.001}
}


class CompoundPrism:
    """ Class to contain Compound Prism functionality """
    def __init__(self, merit_func, default_weights=None):
        global merit, prism, MeritWeights, Config
        if isinstance(merit_func, str):
            default_weights = default[merit_func]
            merit_func = merit_funcs[merit_func]
        self.merit = merit = jit()(merit_func)
        self.default_weights = {**base, **default_weights} if default_weights else base
        self.MeritWeights = MeritWeights = namedtuple('MeritWeights', self.default_weights.keys())
        MeritWeightType = nb.types.NamedUniTuple(nb.float64, len(self.default_weights), MeritWeights)
        spec = OrderedDict(
            prism_count=nb.int64,
            glass_count=nb.int64,
            glass_indices=nb.types.Array(nb.int64, 1, 'C'),
            incident_indices=nb.types.Array(nb.int64, 1, 'C'),
            refracted_indices=nb.types.Array(nb.int64, 1, 'C'),
            deltaC_target=nb.float64,
            deltaT_target=nb.float64,
            weights=MeritWeightType,
            nwaves=nb.int64,
            sampling_domain_is_wavenumber=nb.boolean,
            theta0=nb.float64,
            initial_angles=nb.types.Array(nb.float64, 1, 'C'),
            angle_limit=nb.float64,
            w=nb.types.Array(nb.float64, 1, 'C'),
            n=nb.types.Array(nb.float64, 2, 'C')
        )
        self.Config = Config = namedtuple("Config", spec.keys())
        ConfigType = nb.types.NamedTuple(spec.values(), Config)
        self.prism = prism = jit((ConfigType,))(prism)

    def configure(self, glass_indices, deltaC_target, deltaT_target, weights, sampling_domain, theta0, initial_angles, angle_limit, w, n=None):
        nwaves = w.size
        sampling_domain_is_wavenumber = sampling_domain == 'wavenumber'
        glass_indices = np.array(glass_indices)
        prism_count, glass_count = len(glass_indices), glass_indices.max() + 1
        incident_indices = np.arange(0, glass_count * 2 + 1, 2)
        refracted_indices = np.arange(1, prism_count * 2 + 1, 2)
        if initial_angles is None:
            initial_angles = np.ones((glass_count,)) * deltaT_target
            initial_angles[1::2] *= -1
            initial_angles += np.ones((glass_count,)) * deltaC_target / (2 ** glass_count)
        else:
            initial_angles = np.array(initial_angles) * np.pi / 180
        weights = self.MeritWeights(**{k: float(v) for k, v in {**self.default_weights, **weights}.items()})
        return self.Config(prism_count, glass_count, glass_indices, incident_indices, refracted_indices, deltaC_target, deltaT_target, weights, nwaves, sampling_domain_is_wavenumber, theta0, initial_angles, angle_limit, w, n)

    def __call__(self, model):
        return self.prism(model)
