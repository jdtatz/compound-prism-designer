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

    def params_bounds(self) -> typing.Tuple[typing.Iterable[float], typing.Iterable[float]]:
        nprism = self.max_prism_count
        prism_count_bounds = 1, 1 + self.max_prism_count
        glass_bounds = nprism * [(0, len(self.glasses))]
        angle_bounds = (nprism + 1) * [(-np.pi / 2, np.pi / 2)]
        length_bounds = nprism * [(0., self.prism_height)]
        curvature_bounds = 0.001, 1.0
        y_mean_bounds = 0, self.prism_height
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

    def array_to_params(self, p: typing.Sequence) -> (CompoundPrism, DetectorArray, GaussianBeam):
        def slices():
            arr = np.asarray(p)
            sizes = 1, self.max_prism_count, self.max_prism_count + 1, self.max_prism_count, 1, 1, 1
            for i in sizes:
                if i == 1:
                    yield arr[0]
                else:
                    yield arr[:i]
                arr = arr[i:]
        params = list(slices())
        prism_count = int(np.clip(params[0], 1, self.max_prism_count))
        glass_names, glasses = zip(*(self.glasses[min(int(i), len(self.glasses) - 1)] for i in params[1][:prism_count]))
        angles = params[2][:prism_count + 1]
        lengths = params[3][:prism_count]
        curvature = params[4]
        y_mean = params[5]
        detector_array_angle = params[6]

        prism = CompoundPrism(
            glasses=glasses,
            angles=angles,
            lengths=lengths,
            curvature=curvature,
            height=self.prism_height,
            width=self.prism_width,
        )
        detarr = DetectorArray(
            bins=self.detector_array_bin_bounds,
            min_ci=self.detector_array_min_ci,
            angle=detector_array_angle,
            length=self.detector_array_length,
        )
        beam = GaussianBeam(
            wavelength_range=self.wavelength_range,
            width=self.beam_width,
            y_mean=y_mean,
        )
        return prism, detarr, beam


class Soln(typing.NamedTuple):
    compound_prism: CompoundPrism
    detector_array: DetectorArray
    beam: GaussianBeam
    fitness: DesignFitness

    @staticmethod
    def from_config_and_array(config: Config, p: np.ndarray):
        compound_prism, detector_array, beam = config.array_to_params(p)
        return Soln(
            compound_prism=compound_prism,
            detector_array=detector_array,
            beam=beam,
            fitness=fitness(compound_prism, detector_array, beam)
        )


nobjective = len(dataclasses.fields(DesignFitness))


def transmission_data(wavelengths: [float], prism: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam, det):
    return np.stack([
        _prism.p_dets_l_wavelength(
            w,
            prism,
            detarr,
            beam,
            det
        ) for w in wavelengths],
        axis=1
    )


def detector_array_position(prism: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam):
    return np.array(
        _prism.detector_array_position(
            prism,
            detarr,
            beam,
        )
    )


def trace(wavelength: float, initial_y: float, prism: CompoundPrism, detarr: DetectorArray, det):
    return _prism.trace(
        wavelength,
        initial_y,
        prism,
        detarr,
        det
    )


def fitness(prism: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam):
    try:
        return _prism.fitness(
            prism,
            detarr,
            beam,
        )
    except RayTraceError:
        return DesignFitness(size=np.inf, info=0, deviation=np.inf)


class PyGmoPrismProblem:
    def __init__(self, config: Config):
        self.config = config

    def fitness(self, v: np.ndarray):
        prism, detarr, beam = self.config.array_to_params(v)
        val = fitness(prism, detarr, beam)
        if val.size > 30 * self.config.prism_height or abs(val.info) < 0.1:
            return [np.inf] * nobjective
        else:
            return val.size, -val.info, val.deviation

    def batch_fitness(self, vs: np.ndarray):
        # from multiprocessing.dummy import Pool
        return np.concatenate(list(map(
            lambda v: self.fitness(v),
            vs.reshape(-1, 3 * self.config.max_prism_count + 5))), axis=None)

    def get_bounds(self):
        return self.config.params_bounds()

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
    return [Soln.from_config_and_array(config, p) for isl in archi for p in isl.get_population().get_x()]
