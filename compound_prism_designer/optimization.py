import dataclasses
import typing
import numpy as np
import pygmo as pg
import platypus
from compound_prism_designer.rffi import RayTraceError, create_glass_catalog, CompoundPrism, DetectorArray, GaussianBeam, \
    PyGlass, DesignFitness, detector_array_position, trace, fitness, p_dets_l_wavelength


class Config(typing.NamedTuple):
    """Specification class for the configuration of the Spectrometer Designer."""
    max_prism_count: int
    wavelength_range: typing.Tuple[float, float]
    beam_width: float
    max_prism_height: float
    prism_width: float
    detector_array_length: float
    detector_array_min_ci: float
    detector_array_bin_bounds: np.ndarray
    glass_catalog: typing.Sequence[PyGlass]

    @property
    def params_count(self) -> int:
        return 3 * self.max_prism_count + 6


    @property
    def params_dtype(self) -> np.dtype:
        """The numpy dtype of the optimization parameters."""
        return np.dtype([
            ('nprism', 'f8'),
            ('prism_height', 'f8'),
            ('glass_indices', 'f8', self.max_prism_count),
            ('angles', 'f8', self.max_prism_count + 1),
            ('lengths', 'f8', self.max_prism_count),
            ('curvature', 'f8'),
            ('normalized_y_mean', 'f8'),
            ('detector_array_angle', 'f8'),
        ])

    def params_bounds(self) -> typing.Tuple[typing.Iterable[float], typing.Iterable[float]]:
        """The bounds of the optimization parameters."""
        nprism = self.max_prism_count
        prism_count_bounds = 1, 1 + self.max_prism_count
        prism_height_bounds = 0.001, self.max_prism_height
        glass_bounds = nprism * [(0, len(self.glass_catalog) - 1)]
        angle_bounds = (nprism + 1) * [(-np.pi / 2, np.pi / 2)]
        length_bounds = nprism * [(0., self.max_prism_height)]
        curvature_bounds = 0.001, 1.0
        normalized_y_mean_bounds = 0, 1
        det_arr_angle_bounds = -np.pi, np.pi
        lb, ub = np.transpose([
            prism_count_bounds,
            prism_height_bounds,
            *glass_bounds,
            *angle_bounds,
            *length_bounds,
            curvature_bounds,
            normalized_y_mean_bounds,
            det_arr_angle_bounds,
        ])
        return lb, ub

    def array_to_params(self, p: 'array_like') -> (CompoundPrism, DetectorArray, GaussianBeam):
        """Creates the spectrometer specification from the config & optimization parameters."""
        arr = np.ascontiguousarray(p, dtype=np.float64)
        assert len(arr) == self.params_count
        params = arr.view(self.params_dtype)[0]
        prism_count = int(np.clip(params['nprism'], 1, self.max_prism_count))
        prism_height = params['prism_height']
        prism = CompoundPrism(
            glasses=[self.glass_catalog[int(i)] for i in params['glass_indices'][:prism_count]],
            angles=params['angles'][:prism_count + 1],
            lengths=params['lengths'][:prism_count],
            curvature=params['curvature'],
            height=prism_height,
            width=self.prism_width,
        )
        detarr = DetectorArray(
            bins=self.detector_array_bin_bounds,
            min_ci=self.detector_array_min_ci,
            angle=params['detector_array_angle'],
            length=self.detector_array_length,
        )
        beam = GaussianBeam(
            wavelength_range=self.wavelength_range,
            width=self.beam_width,
            y_mean=params['normalized_y_mean'] * prism_height,
        )
        return prism, detarr, beam


class Soln(typing.NamedTuple):
    """A spectrometer design returned as a solution to the optimization problem."""
    compound_prism: CompoundPrism
    detector_array: DetectorArray
    beam: GaussianBeam
    fitness: DesignFitness

    @staticmethod
    def from_config_and_array(config: Config, p: 'array_like') -> typing.Optional['Soln']:
        compound_prism, detector_array, beam = config.array_to_params(p)
        try:
            return Soln(
                compound_prism=compound_prism,
                detector_array=detector_array,
                beam=beam,
                fitness=fitness(compound_prism, detector_array, beam)
            )
        except RayTraceError:
            return None


def transmission_data(wavelengths: [float], prism: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam, det):
    return np.stack([
        p_dets_l_wavelength(
            w,
            prism,
            detarr,
            beam,
            det
        ) for w in wavelengths],
        axis=1
    )


_nobjective = len(dataclasses.fields(DesignFitness))


class PyGmoPrismProblem:
    def __init__(self, config: Config):
        self.config = config

    def fitness(self, v: np.ndarray):
        prism, detarr, beam = self.config.array_to_params(v)
        try:
            val = fitness(
                prism,
                detarr,
                beam,
            )
            assert val.size <= 30 * self.config.max_prism_height and abs(val.info) >= 0.1
            return val.size, -val.info, val.deviation
        except (RayTraceError, AssertionError):
            return [np.inf] * _nobjective

    def get_bounds(self):
        return self.config.params_bounds()

    @staticmethod
    def get_nobj():
        return _nobjective

    @staticmethod
    def get_name():
        return "Compound Prism Optimizer"


def use_pygmo(iter_count, thread_count, pop_size, config: Config):
    if pop_size < 5 or pop_size % 4 != 0:
        pop_size = max(8, pop_size + 4 - pop_size % 4)
    prob = pg.problem(PyGmoPrismProblem(config))
    a = pg.nsga2(gen=iter_count, cr=0.98, m=0.1)
    algo = pg.algorithm(a)
    # w = np.array((1e-3, 1, 1e-4))
    # prob = pg.decompose(prob, weight=w / np.sum(w), z=(0, -np.log2(len(config.detector_array_bin_bounds)), 0), method='tchebycheff')
    # algo = pg.algorithm(pg.mbh(pg.gaco(gen=iter_count)))
    # pop = pg.population(prob=prob, size=pop_size)
    # algo.evolve(pop)
    archi = pg.archipelago(thread_count, algo=algo, prob=prob, pop_size=pop_size)
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
    solns = (Soln.from_config_and_array(config, x) for isl in archi for x in isl.get_population().get_x())
    solns = list(filter(lambda v: v is not None and v.fitness.size <= 30 * config.max_prism_height and v.fitness.info >= 0.1, solns))
    sorted_solns_idxs = pg.select_best_N_mo([
        (s.fitness.size, -s.fitness.info, s.fitness.deviation) for s in solns
    ], pop_size)
    return [solns[i] for i in sorted_solns_idxs]


class PlatypusPrismProblem(platypus.Problem):
    def __init__(self, config: Config):
        bounds = config.params_bounds()
        super(PlatypusPrismProblem, self).__init__(config.params_count, _nobjective, 1)
        self.types = [platypus.Real(l, u) for l, u in zip(*bounds)]
        self.constraints[:] = "!=0"
        self.directions[:] = platypus.Problem.MAXIMIZE, platypus.Problem.MINIMIZE, platypus.Problem.MAXIMIZE
        self.config = config

    def evaluate(self, solution: platypus.Solution):
        prism, detarr, beam = self.config.array_to_params(solution.variables[:])
        try:
            val = fitness(
                prism,
                detarr,
                beam,
            )
            assert val.size <= 30 * self.config.max_prism_height and val.info >= 0.1
            solution.objectives[:] = val.size, val.info, val.deviation
            solution.constraints[:] = 1
        except (RayTraceError, AssertionError):
            solution.constraints[:] = 0
            solution.objectives[:] = 30 * self.config.max_prism_height, 0, 1

    @staticmethod
    def get_name():
        return "Compound Prism Optimizer"
