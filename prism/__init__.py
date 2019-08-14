import typing
import numpy as np
import pygmo as pg
import prism._prism as _prism
from prism._prism import RayTraceError, create_catalog, Prism, DetectorArray, GaussianBeam, PyGlass


class Config(typing.NamedTuple):
    prism_count: int
    wavelength_range: typing.Tuple[float, float]
    beam_width: float
    prism_height: float
    prism_width: float
    detector_array_length: float
    detector_array_min_ci: float
    detector_array_bin_bounds: np.ndarray
    glasses: typing.Sequence[typing.Tuple[str, PyGlass]]


class Params(typing.NamedTuple):
    glass_names: typing.Sequence[str]
    glasses: typing.Sequence[PyGlass]
    angles: typing.Sequence[float]
    curvature: float
    y_mean: float
    detector_array_angle: float


class Soln(typing.NamedTuple):
    params: Params
    objectives: typing.Tuple[float, float, float]


nobjective = 3


def to_params(config: Config, p: np.ndarray):
    glass_names, glasses = zip(*(config.glasses[min(int(i), len(config.glasses) - 1)] for i in p[:config.prism_count]))
    return Params(
        glass_names=glass_names,
        glasses=glasses,
        angles=np.asarray(p[config.prism_count:config.prism_count * 2 + 1]),
        curvature=p[-3],
        y_mean=p[-2],
        detector_array_angle=p[-1]
    )


def to_prism(config: Config, params: Params):
    return Prism(
        glasses=params.glasses,
        angles=params.angles,
        curvature=params.curvature,
        height=config.prism_height,
        width=config.prism_width,
    )


def to_det_array(config: Config, params: Params):
    return DetectorArray(
        bins=config.detector_array_bin_bounds,
        min_ci=config.detector_array_min_ci,
        angle=params.detector_array_angle,
        length=config.detector_array_length,
    )


def to_beam(config: Config, params: Params):
    return GaussianBeam(
        width=config.beam_width,
        y_mean=params.y_mean,
        w_range=config.wavelength_range,
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
    def __init__(self, config: Config):
        self.config = config

    def fitness(self, v: np.ndarray):
        params = to_params(self.config, v)
        val = fitness(self.config, params)
        if val[0] > 30 * self.config.prism_height or abs(val[1]) < 0.1:
            return [np.inf] * nobjective
        else:
            return val

    def batch_fitness(self, vs: np.ndarray):
        # from multiprocessing.dummy import Pool
        return np.concatenate(list(map(
            lambda v: self.fitness(v),
            vs.reshape(-1, 2 * self.config.prism_count + 4))), axis=None)

    def get_bounds(self):
        nprism = self.config.prism_count
        glass_bounds = nprism * [(0, len(self.config.glasses))]
        angle_bounds = (nprism + 1) * [(-np.pi / 2, np.pi / 2)]
        curvature_bounds = 0.001, 1.0
        y_mean_bounds = 0, self.config.prism_height
        det_arr_angle_bounds = -np.pi, np.pi
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


def use_pygmo(iter_count, thread_count, pop_size, config: Config):
    if pop_size < 5 or pop_size % 4 != 0:
        pop_size = max(8, pop_size + 4 - pop_size % 4)
    prob = pg.problem(PyGmoPrismProblem(config))
    bfe = pg.bfe(pg.member_bfe())
    a = pg.nsga2(gen=iter_count, m=0.05)
    a.set_bfe(bfe)
    algo = pg.algorithm(a)
    # w = np.array((1e-3, 1, 1e-4))
    # prob = pg.decompose(prob, weight=w / np.sum(w), z=(0, -np.log2(len(config.detector_array_bin_bounds)), 0), method='tchebycheff')
    # algo = pg.algorithm(pg.mbh(pg.gaco(gen=iter_count)))
    # pop = pg.population(prob=prob, size=pop_size)
    # algo.evolve(pop)
    archi = pg.archipelago(thread_count, algo=algo, prob=prob, pop_size=pop_size, b=bfe)
    archi.evolve()
    archi.wait_check()
    '''
    #for isl in archi:
        #pop = isl.get_population()
    p = to_params(config, catalog, pop.champion_x)
    v = fitness(config, p)
    print(pop.champion_f, v)
    print(p)
    '''
    return [Soln(params=to_params(config, p), objectives=prob.fitness(p)) for isl in archi for p in isl.get_population().get_x()]
