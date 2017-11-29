import numpy as np

from compoundprism.utils import *


registry = {}


def register(name):
    def decorator(func):
        registry[name] = func
        return func
    return decorator


@register('linearity')
def linearity(settings):
    NL = settings['weights']['linearity']
    @jit
    def merit(n, angles, delta_spectrum, thetas):
        NL_err = NL * nonlinearity(delta_spectrum)
        return NL_err
    return merit


@register('beam compression')
def beam_compression(settings):
    NL = settings['weights']['linearity']
    weight_K = settings['weights']['K']
    nwaves = settings['nwaves']
    @jit
    def merit(n, angles, delta_spectrum, thetas):
        NL_err = NL * nonlinearity_error(delta_spectrum)
        K = beam_compression(thetas, nwaves)
        K_err = weight_K * (K - 1.0) ** 2
        return NL_err + K_err
    return merit


@register('chromaticity')
def chromaticity(settings):
    weight_chromat = settings['weights']['chromat']
    @jit
    def merit(n, angles, delta_spectrum, thetas):
        chromat = weight_chromat * np.abs(delta_spectrum.max() - delta_spectrum.min())
        return chromat
    return merit


@register('thinness')
def thinness(settings):
    weight_anglesum = settings['weights']['anglesum']
    deltaT_target = settings['deltaT_target']
    @jit
    def merit(n, angles, delta_spectrum, thetas):
        anglesum = weight_anglesum * deltaT_target * np.sum(angles ** 2)
        return anglesum
    return merit


@register('second-order')
def secondorder(settings):
    weight_secondorder = settings['weights']['secondorder']
    @jit
    def merit(n, angles, delta_spectrum, thetas):
        coeffs, remainder = get_poly_coeffs(delta_spectrum, 2)
        secondorder = weight_secondorder / coeffs[2] ** 2
        return secondorder
    return merit
