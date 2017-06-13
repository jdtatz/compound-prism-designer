from collections import OrderedDict

import numpy as np

from .registry import MeritFunctionRegistry
from ..utils import *


@MeritFunctionRegistry.register
class LinearityMerit:
    name = 'linearity'
    weights = OrderedDict(NL=25)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        NL_err = model.weights.NL * nonlinearity_error(delta_spectrum)
        return NL_err


@MeritFunctionRegistry.register
class BeamCompressionMerit:
    name = 'beam compression'
    weights = OrderedDict(NL=25, K=0.0025)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        NL_err = model.weights.NL * nonlinearity_error(delta_spectrum)
        K = beam_compression(thetas, model.nwaves)
        K_err = model.weights.K * (K - 1.0) ** 2
        return NL_err + K_err


@MeritFunctionRegistry.register
class ChromaticityMerit:
    name = 'chromaticity'
    weights = OrderedDict(chromat=1, dispersion=0)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        chromat = model.weights.chromat * np.abs(delta_spectrum.max() - delta_spectrum.min())
        return chromat


@MeritFunctionRegistry.register
class ThinnessMerit:
    name = 'thinness'
    weights = OrderedDict(anglesum=0.001)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        anglesum = model.weights.anglesum * model.deltaT_target * np.sum(angles ** 2)
        return anglesum


@MeritFunctionRegistry.register
class SecondOrderMerit:
    name = 'second-order'
    weights = OrderedDict(secondorder=0.001)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        coeffs, remainder = get_poly_coeffs(delta_spectrum, 2)
        secondorder = model.weights.secondorder / coeffs[2] ** 2
        return secondorder
