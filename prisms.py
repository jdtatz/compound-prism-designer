import numba as nb
import numpy as np
from numpy import pi
from utils import jit, cfunc, gradient, vec_ptr, minimizer
from collections import namedtuple

"""
Utility Functions
"""


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
    g = gradient(gradient(x))
    return np.sqrt(np.sum(g ** 2))


@jit
def nonlinearity_error(x):
    """Calculate the nonlinearity error of the given delta spectrum"""
    grad = gradient(x)
    if np.max(grad) < 0.0 or np.min(grad) < 0.0:
        grad -= np.min(grad)
    return np.sum(gradient(grad) ** 2)


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
    grad = gradient(delta_spectrum) / gradient(w)
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
Prism Code
"""


class CompoundPrism(namedtuple("Prism", 'prism, error, snell, merit_functions, weights')):
    """ Class to contain Compound Prism functionality """
    def __call__(self, *args, **kwargs):
        return self.prism(*args, **kwargs)

base = {'tir': 0.1, 'valid': 1, 'crit_angle': 1, 'thin': 0.25, 'deviation': 1, 'dispersion': 1}

default = {
    'linearity': {'NL': 25},
    'beam compression': {'NL': 25, 'K': 0.0025},
    'chromaticity': {'chromat': 1},
    'thinness': {'anglesum': 0.001},
    'second-order': {'secondorder': 0.001}
}


def prism_setup(prism_count, glass_count, glass_indices, deltaC_target, deltaT_target, merit,
                weights, nwaves, sampling_domain, theta0, initial_angles, angle_limit):
    """
    Due to limitations in numba's abilities this function closure factory is used to generate the functions,
    instead of using a class as would be preferred.
    """
    sampling_domain_is_wavenumber = sampling_domain == 'wavenumber'
    incident_indices = np.arange(0, glass_count * 2 + 1, 2)
    refracted_indices = np.arange(1, prism_count * 2 + 1, 2)
    glass_indices = np.array(glass_indices)
    if initial_angles is None:
        initial_angles = np.ones((glass_count,)) * deltaT_target
        initial_angles[1::2] *= -1
        initial_angles += np.ones((glass_count,)) * deltaC_target / (2 ** glass_count)
    else:
        initial_angles = np.array(initial_angles) * pi / 180
    initial_angles = tuple(initial_angles)
    MeritWeights = namedtuple('MeritWeights', [*base, *default[merit]])
    weights = MeritWeights(**{**base, **default[merit], **weights})

    """
    Merit Functions
    """

    @jit
    def linearity(angles, n, delta_spectrum, thetas):
        """Linearity merit function"""
        deltaC_err = weights.deviation * (delta_spectrum[nwaves // 2] - deltaC_target) ** 2
        deltaT_err = weights.dispersion * ((delta_spectrum.max() - delta_spectrum.min()) - deltaT_target) ** 2
        NL_err = weights.NL * nonlinearity_error(delta_spectrum)
        return deltaC_err + deltaT_err + NL_err

    @jit
    def beam_comp(angles, n, delta_spectrum, thetas):
        """Beam Compression and Linearity merit function"""
        deltaC_err = weights.deviation * (delta_spectrum[nwaves // 2] - deltaC_target) ** 2
        deltaT_err = weights.dispersion * ((delta_spectrum.max() - delta_spectrum.min()) - deltaT_target) ** 2
        NL_err = weights.NL * nonlinearity_error(delta_spectrum)
        K = beam_compression(thetas, nwaves)
        K_err = weights.K * (K - 1.0) ** 2
        return deltaC_err + deltaT_err + NL_err + K_err

    @jit
    def chromaticity(angles, n, delta_spectrum, thetas):
        """Chromaticity merit function"""
        deltaC_err = weights.deviation * (delta_spectrum[nwaves // 2] - deltaC_target) ** 2
        chromat = weights.chromat * np.abs(delta_spectrum.max() - delta_spectrum.min())
        return deltaC_err + chromat

    @jit
    def thinness(angles, n, delta_spectrum, thetas):
        """Thinness merit function"""
        deltaC_err = weights.deviation * (delta_spectrum[nwaves // 2] - deltaC_target) ** 2
        deltaT_err = weights.dispersion * ((delta_spectrum.max() - delta_spectrum.min()) - deltaT_target) ** 2
        anglesum = weights.anglesum * deltaT_target * np.sum(angles ** 2)
        return deltaC_err + deltaT_err + anglesum

    @jit
    def second_order(angles, n, delta_spectrum, thetas):
        """Second Order Error merit function"""
        deltaC_err = weights.deviation * (delta_spectrum[nwaves // 2] - deltaC_target) ** 2
        deltaT_err = weights.dispersion * ((delta_spectrum.max() - delta_spectrum.min()) - deltaT_target) ** 2
        coeffs, remainder = get_poly_coeffs(delta_spectrum, 2)
        secondorder = weights.secondorder / coeffs[2] ** 2
        return deltaC_err + deltaT_err + secondorder

    merit_funcs = {'linearity': linearity, 'beam compression': beam_comp, 'chromaticity': chromaticity,
                   'thinness': thinness, 'second-order': second_order}
    merit_func = merit_funcs[merit]

    """
    Prism Code
    """

    @jit
    def snells(angles, n):
        thetas = np.empty((2 * prism_count + 3, nwaves))
        alphas = angles[glass_indices]
        beta = np.sum(alphas[:prism_count // 2]) + alphas[prism_count // 2] / 2
        gamma = np.sum(alphas[prism_count // 2 + 1:]) + alphas[prism_count // 2] / 2
        ns = n[glass_indices]

        thetas[0] = theta0 + beta  # theta 1
        np.arcsin((1.0 / ns[0]) * np.sin(theta0 + beta), thetas[1])  # theta 1 prime
        for i in range(1, prism_count + 1):
            thetas[2 * i] = thetas[2 * i - 1] - alphas[i - 1]  # theta i
            if i < prism_count:
                np.arcsin(ns[i - 1] / ns[i] * np.sin(thetas[2 * i]), thetas[2 * i + 1])  # theta i prime
            else:
                np.arcsin(ns[i - 1] * np.sin(thetas[2 * i]), thetas[2 * i + 1])  # theta (n-1) prime
        thetas[-1] = thetas[-2] + gamma  # theta n

        delta_spectrum = theta0 - thetas[-1]

        return delta_spectrum, thetas

    @jit
    def merit_error(angles, n):
        delta_spectrum, thetas = snells(angles, n)
        # If TIR occurs in the design (producing NaNs in the spectrum), then give a
        # hard error: return a large error which has nothing to do with the (invalid)
        # performance data.
        if np.any(np.isnan(delta_spectrum)):
            return weights.tir * np.sum(np.isnan(delta_spectrum))
        # enforces valid solution
        alphas = angles[glass_indices]
        refracted = thetas[refracted_indices]
        refs = np.abs(alphas) + refracted.T * np.sign(alphas)
        merit_err = weights.valid * np.sum(np.greater(refs, pi / 2)) / (prism_count * nwaves)
        # critical angle prevention
        incident = thetas[incident_indices]
        merit_err += weights.crit_angle * np.sum(np.greater(np.abs(incident), angle_limit)) / nwaves
        # Punish if the prism gets too small to be usable
        too_thin = np.abs(angles) - 1.0
        too_thin[np.where(np.abs(angles) > pi / 180.0)[0]] = 0.0
        merit_err += weights.thin * np.sum(too_thin ** 2)

        return merit_err + merit_func(angles, n, delta_spectrum, thetas)

    @cfunc(nb.float64(nb.types.voidptr, nb.types.voidptr))
    def prep(a0, n0):
        angles = nb.carray(vec_ptr(a0, 0), (glass_count,), np.float64)
        n = nb.carray(n0, (glass_count, nwaves), np.float64)
        return merit_error(angles, n)
    prep = prep.address

    @jit((nb.f8[:, :], nb.f8[:]))
    def prism(n, w):
        init = np.array(initial_angles)
        angles = minimizer(prep, init, n)
        merr = merit_error(angles, n)
        delta_spectrum, thetas = snells(angles, n)
        if np.any(np.isnan(thetas)):
            return

        (delta0, delta1, delta2), remainder = get_poly_coeffs(delta_spectrum, 2)
        deltaM = np.mean(delta_spectrum) * 180.0 / pi
        deltaC = delta_spectrum[nwaves // 2] * 180.0 / pi
        deltaT = (delta_spectrum.max() - delta_spectrum.min()) * 180.0 / pi
        alphas = angles * 180.0 / pi
        NL = 10000.0 * nonlinearity(delta_spectrum)
        SSR = spectral_sampling_ratio(w, delta_spectrum, sampling_domain_is_wavenumber)
        K = beam_compression(thetas, nwaves)
        _, remainder = get_poly_coeffs(delta_spectrum, 1)
        dw = np.abs(np.mean(gradient(w)))
        nonlin = np.sqrt(remainder) * dw * 180.0 / pi
        chromat = 100.0 * abs(delta_spectrum.max() - delta_spectrum.min()) * 180.0 / pi
        delta1 *= 2.0 * 180.0 / pi
        delta2 *= 2.0 * 180.0 / pi
        # T = transmission(thetas, n, nwaves)

        return alphas, merr, deltaC, deltaT, NL, SSR, K, deltaM, delta1, delta2, nonlin, chromat

    return CompoundPrism(prism, merit_error, snells, merit_funcs, weights)
