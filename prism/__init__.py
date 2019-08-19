import dataclasses
import typing
import numpy as np
import pygmo as pg
import prism.rffi as _prism
from prism.rffi import RayTraceError, create_glass_catalog, CompoundPrism, DetectorArray, GaussianBeam, PyGlass, DesignFitness


class Config(typing.NamedTuple):
    max_prism_count: int
    wavelength_range: typing.Tuple[float, float]
    beam_width: float
    prism_height: float
    prism_width: float
    detector_array_length: float
    detector_array_min_ci: float
    detector_array_bin_bounds: np.ndarray
    glasses: typing.Sequence[typing.Tuple[str, PyGlass]]


class Params(typing.NamedTuple):
    prism_count: int
    glass_names: typing.Sequence[str]
    glasses: typing.Sequence[PyGlass]
    angles: np.ndarray
    lengths: np.ndarray
    curvature: float
    y_mean: float
    detector_array_angle: float


class Soln(typing.NamedTuple):
    params: Params
    objectives: typing.Tuple[float, float, float]


nobjective = len(dataclasses.fields(DesignFitness))


def to_params(config: Config, p: np.ndarray):
    def slices():
        arr = np.asarray(p)
        sizes = 1, config.max_prism_count, config.max_prism_count + 1, config.max_prism_count, 1, 1, 1
        for i in sizes:
            if i == 1:
                yield arr[0]
            else:
                yield arr[:i]
            arr = arr[i:]
    params = list(slices())
    prism_count = int(np.clip(params[0], 1, config.max_prism_count))
    glass_names, glasses = zip(*(config.glasses[min(int(i), len(config.glasses) - 1)] for i in params[1][:prism_count]))
    return Params(
        prism_count=prism_count,
        glass_names=glass_names,
        glasses=glasses,
        angles=params[2][:prism_count + 1],
        lengths=params[3][:prism_count],
        curvature=params[4],
        y_mean=params[5],
        detector_array_angle=params[6]
    )


def to_prism(config: Config, params: Params):
    return CompoundPrism(
        glasses=params.glasses,
        angles=params.angles,
        lengths=params.lengths,
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
        wavelength_range=config.wavelength_range,
    )


def transmission_data(wavelengths: [float], config: Config, params: Params, det):
    return np.stack([
        _prism.p_dets_l_wavelength(
            w,
            to_prism(config, params),
            to_det_array(config, params),
            to_beam(config, params),
            det
        ) for w in wavelengths],
        axis=1
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
        return DesignFitness(size=np.inf, info=np.inf, deviation=np.inf)


class PyGmoPrismProblem:
    def __init__(self, config: Config):
        self.config = config

    def fitness(self, v: np.ndarray):
        params = to_params(self.config, v)
        val = fitness(self.config, params)
        if val.size > 30 * self.config.prism_height or abs(val.info) < 0.1:
            return [np.inf] * nobjective
        else:
            return dataclasses.astuple(val)

    def batch_fitness(self, vs: np.ndarray):
        # from multiprocessing.dummy import Pool
        return np.concatenate(list(map(
            lambda v: self.fitness(v),
            vs.reshape(-1, 3 * self.config.max_prism_count + 5))), axis=None)

    def get_bounds(self):
        nprism = self.config.max_prism_count
        prism_count_bounds = 1, 1 + self.config.max_prism_count
        glass_bounds = nprism * [(0, len(self.config.glasses))]
        angle_bounds = (nprism + 1) * [(-np.pi / 2, np.pi / 2)]
        length_bounds = nprism * [(0., self.config.prism_height)]
        curvature_bounds = 0.001, 1.0
        y_mean_bounds = 0, self.config.prism_height
        det_arr_angle_bounds = -np.pi, np.pi
        return tuple(zip(*[
            prism_count_bounds,
            *glass_bounds,
            *angle_bounds,
            *length_bounds,
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
    a = pg.nsga2(gen=iter_count, cr=0.98, m=0.1)
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
