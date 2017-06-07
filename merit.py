import numpy as np
from utils import *
from collections import OrderedDict

__all__ = ["MeritFunctionRegistry", "InvalidMeritFunction"]


class InvalidMeritFunction(Exception):
    pass


class MeritFunctionRegistry(type):
    registry = {}

    def __new__(cls, name, bases, namespace):
        if 'name' not in namespace:
            raise InvalidMeritFunction("No name given")
        reg_name = namespace["name"]
        if reg_name in cls.registry:
            raise InvalidMeritFunction(f"Name {reg_name} is already registered to {cls.registry[reg_name]}")
        if 'merit' not in namespace:
            raise InvalidMeritFunction("No merit function specfied")
        if not isinstance(namespace['merit'], staticmethod):
            raise InvalidMeritFunction("The merit function must be a static method")
        namespace['default_weights'] = OrderedDict(namespace.get('default_weights', {}))
        new_cls = super().__new__(cls, name, bases, namespace)
        cls.registry[reg_name] = new_cls
        return new_cls

    @classmethod
    def get(mcs, item):
        return mcs.registry[item]


class LinearityMerit(metaclass=MeritFunctionRegistry):
    name = 'linearity'
    default_weights = OrderedDict(NL=25)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        NL_err = model.weights.NL * nonlinearity_error(delta_spectrum)
        return NL_err


class BeamCompressionMerit(metaclass=MeritFunctionRegistry):
    name = 'beam compression'
    default_weights = OrderedDict(NL=25, K=0.0025)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        NL_err = model.weights.NL * nonlinearity_error(delta_spectrum)
        K = beam_compression(thetas, model.nwaves)
        K_err = model.weights.K * (K - 1.0) ** 2
        return NL_err + K_err


class ChromaticityMerit(metaclass=MeritFunctionRegistry):
    name = 'chromaticity'
    default_weights = OrderedDict(chromat=1, dispersion=0)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        chromat = model.weights.chromat * np.abs(delta_spectrum.max() - delta_spectrum.min())
        return chromat


class ThinnessMerit(metaclass=MeritFunctionRegistry):
    name = 'thinness'
    default_weights = OrderedDict(anglesum=0.001)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        anglesum = model.weights.anglesum * model.deltaT_target * np.sum(angles ** 2)
        return anglesum


class SecondOrderMerit(metaclass=MeritFunctionRegistry):
    name = 'second-order'
    default_weights = OrderedDict(secondorder=0.001)

    @staticmethod
    def merit(model, angles, delta_spectrum, thetas):
        coeffs, remainder = get_poly_coeffs(delta_spectrum, 2)
        secondorder = model.weights.secondorder / coeffs[2] ** 2
        return secondorder
