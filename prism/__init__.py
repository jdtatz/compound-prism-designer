from collections import namedtuple
import numpy as np
import pygmo as pg
import prism._prism as _prism
from prism._prism import RayTraceError, create_catalog, Prism, DetectorArray, GaussianBeam

Config = namedtuple("Config", "prism_count, wmin, wmax, prism_height, prism_width, det_arr_length, beam_width, det_arr_min_ci, bounds")
Params = namedtuple("Params", "glass_names, glasses, thetas, curvature, y_mean, det_arr_angle")
Soln = namedtuple("Soln", "params, objectives")
nobjective = 3


def to_params(config: Config, catalog, p: np.ndarray):
    gvalues = list(catalog.keys())
    glass_names = [gvalues[min(int(i), len(catalog) - 1)] for i in p[:config.prism_count]]
    glasses = [catalog[n] for n in glass_names]
    return Params(glass_names, glasses, np.asarray(p[config.prism_count:config.prism_count * 2 + 1]), *p[-3:])


def to_prism(config: Config, params: Params):
    return Prism(
        glasses=params.glasses,
        angles=params.thetas,
        curvature=params.curvature,
        height=config.prism_height,
        width=config.prism_width,
    )


def to_det_array(config: Config, params: Params):
    return DetectorArray(
        bins=config.bounds,
        min_ci=config.det_arr_min_ci,
        angle=params.det_arr_angle,
        length=config.det_arr_length,
    )


def to_beam(config: Config, params: Params):
    return GaussianBeam(
        width=config.beam_width,
        y_mean=params.y_mean,
        w_range=(config.wmin, config.wmax),
    )


def transmission_data(wavelengths: [float], config: Config, params: Params, det):
    return np.array(
        _prism.transmission(
            wavelengths,
            to_prism(config, params),
            to_det_array(config, params),
            to_beam(config, params),
            det
        )
    )


def detector_array_position(config: Config, params: Params):
    return np.array(
        _prism.detector_array_position(
            to_prism(config, params),
            to_det_array(config, params),
            to_beam(config, params),
        )
    )


def trace(wavelength: float, initial_y: float, config: Config, params: Params, det):
    return _prism.trace(
        wavelength,
        initial_y,
        to_prism(config, params),
        to_det_array(config, params),
        det
    )


def fitness(config: Config, params: Params):
    try:
        return _prism.fitness(
            to_prism(config, params),
            to_det_array(config, params),
            to_beam(config, params),
        )
    except RayTraceError:
        return [np.inf] * nobjective


class PyGmoPrismProblem:
    def __init__(self, config, catalog):
        self.config, self.catalog = config, catalog

    def fitness(self, v):
        params = to_params(self.config, self.catalog, v)
        val = fitness(self.config, params)
        if val[0] > 30 * self.config.prism_height:
            return [np.inf] * nobjective
        else:
            return val

    def get_bounds(self):
        nprism = self.config.prism_count
        glass_bounds = nprism * [(0, len(self.catalog) - 1)]
        l, u = np.deg2rad(0), np.deg2rad(90)
        angle_bounds = [(l, u) if i % 2 else (-u, -l) for i in range(nprism + 1)]
        curvature_bounds = 0.001, 1.0
        y_mean_bounds = 0, self.config.prism_height
        sec = np.deg2rad(180)
        det_arr_angle_bounds = -sec, sec
        return tuple(zip(*[
            *glass_bounds,
            *angle_bounds,
            curvature_bounds,
            y_mean_bounds,
            det_arr_angle_bounds,
        ]))

    @staticmethod
    def get_nobj():
        return nobjective

    @staticmethod
    def get_name():
        return "Compound Prism Optimizer"


def use_pygmo(iter_count, thread_count, pop_size, config: Config, catalog):
    if pop_size < 5 or pop_size % 4 != 0:
        pop_size = max(8, pop_size + 4 - pop_size % 4)
    prob = pg.problem(PyGmoPrismProblem(config, catalog))
    algo = pg.algorithm(pg.nsga2(gen=iter_count))
    archi = pg.archipelago(thread_count, algo=algo, prob=prob, pop_size=pop_size)
    archi.evolve()
    archi.wait_check()
    return [Soln(params=to_params(config, catalog, p), objectives=o) for isl in archi for (p, o) in zip(isl.get_population().get_x(), isl.get_population().get_f())]
