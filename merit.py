import numpy as np
from utils import *

__all__ = ["get_merit_function", "register_merit_function"]

merit_functions = {}


def get_merit_function(merit_name):
    if not isinstance(merit_name, str):
        raise TypeError(f"Given merit name is not a string, is {type(merit_name)}")
    if merit_name not in merit_functions:
        raise Exception(f"Unregistered merit function called: {merit_name}")
    return merit_functions[merit_name]


def register_merit_function(name, func, init_func=None, **weights):
    if not isinstance(name, str):
        raise TypeError(f"Merit name is not a string, is {type(name)}")
    merit_functions[name] = (func, weights, init_func)


def linearity(model, angles, delta_spectrum, thetas):
    """Linearity merit function"""
    NL_err = model.weights.NL * nonlinearity_error(delta_spectrum)
    return NL_err
register_merit_function('linearity', linearity, NL=25)


def beam_comp(model, angles, delta_spectrum, thetas):
    """Beam Compression and Linearity merit function"""
    NL_err = model.weights.NL * nonlinearity_error(delta_spectrum)
    K = beam_compression(thetas, model.nwaves)
    K_err = model.weights.K * (K - 1.0) ** 2
    return NL_err + K_err
register_merit_function('beam compression', beam_comp, NL=25, K=0.0025)


def chromaticity(model, angles, delta_spectrum, thetas):
    """Chromaticity merit function"""
    chromat = model.weights.chromat * np.abs(delta_spectrum.max() - delta_spectrum.min())
    return chromat
register_merit_function('chromaticity', chromaticity, chromat=1, dispersion=0)


def thinness(model, angles, delta_spectrum, thetas):
    """Thinness merit function"""
    anglesum = model.weights.anglesum * model.deltaT_target * np.sum(angles ** 2)
    return anglesum
register_merit_function('thinness', thinness, anglesum=0.001)


def second_order(model, angles, delta_spectrum, thetas):
    """Second Order Error merit function"""
    coeffs, remainder = get_poly_coeffs(delta_spectrum, 2)
    secondorder = model.weights.secondorder / coeffs[2] ** 2
    return secondorder
register_merit_function('second-order', second_order, secondorder=0.001)
