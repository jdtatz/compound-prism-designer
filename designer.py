#!/usr/bin/env python3
# from __future__ import annotations
import itertools
from typing import Union, Sequence, List, Tuple, Callable, Optional, Iterator
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling, get_reference_directions
from pymoo.optimize import minimize
from multiprocessing.pool import ThreadPool
from compound_prism_designer import RayTraceError, Glass, BUNDLED_CATALOG, CompoundPrism, \
    DetectorArray, UniformWavelengthDistribution, GaussianBeam, FiberBeam, Spectrometer, AbstractGlass, new_catalog, position_detector_array
from compound_prism_designer.interactive import interactive_show, Design
from dataclasses import dataclass
from serde import serde
from serde.toml import from_toml
from pathlib import Path
from functools import reduce, partial
from operator import mul
rprod = partial(reduce, mul)


@serde(rename_all="spinalcase")
@dataclass
class CompoundPrismSpectrometerConfig:
    max_height: float
    prism_width: float
    bin_count: int
    bin_size: float
    linear_slope: float
    linear_intercept: float
    detector_array_length: float
    max_incident_angle: float
    angle_is_rad: bool
    wavelength_range: Tuple[float, float]
    ar_coated: bool
    # Gaussian Beam
    beam_width: Optional[float] = None
    # Fiber Beam
    fiber_core_radius: Optional[float] = None
    numerical_aperture: Optional[float] = None

    glass_catalog_path: Optional[Path] = None

    def __post_init__(self):
        if not self.angle_is_rad:
            self.max_incident_angle = np.deg2rad(self.max_incident_angle)
            self.angle_is_rad = True

    @property
    def glass_catalog(self) -> Iterator[Glass]:
        if self.glass_catalog_path is None:
            gcat: Iterator[AbstractGlass] = iter(BUNDLED_CATALOG)
        else:
            with open(self.glass_catalog_path) as f:
                gcat = new_catalog(f.read())
        return (g.into_glass(self.wavelength_range)[0] for g in gcat)


@serde(rename_all="spinalcase")
@dataclass
class OptimizerConfig:
    cpu_only: bool = False
    parallelize: bool = False
    optimize_size: bool = True
    optimize_deviation: bool = True


@serde(rename_all="spinalcase")
@dataclass
class CompoundPrismSpectrometerProblemConfig:
    spectrometer: CompoundPrismSpectrometerConfig
    optimizer: OptimizerConfig = OptimizerConfig()

    auto_position_detector_acceptance: Optional[float] = None
    nglass: Optional[int] = None
    glass_names: Optional[List[str]] = None

    def __post_init__(self):
        if self.nglass is None and self.glass_names is None:
            raise NotImplementedError("One of `nglass` or `glass-names` must be defined")

    @property
    def nglass_or_const_glasses(self) -> Union[int, Sequence[str]]:
        if self.nglass is None and self.glass_names is None:
            raise NotImplementedError("One of `nglass` or `glass-names` must be defined")
        elif self.glass_names is not None:
            return self.glass_names
        else:
            # MyPy complains if this assert is not included, but this assert should NEVER happen
            assert self.nglass is not None
            return self.nglass



class CompoundPrismSpectrometerProblem(ElementwiseProblem):
    def __init__(self, config: CompoundPrismSpectrometerProblemConfig, **kwargs):
        self.config = config
        self.cpu_only = config.optimizer.cpu_only
        self.glass_list = list(config.spectrometer.glass_catalog)
        self.glass_dict = { g.name: g for g in self.glass_list }
        catalog_bounds = 0, len(self.glass_list) - 1
        height_bounds = (0.0001 * config.spectrometer.max_height, config.spectrometer.max_height)
        normalized_y_mean_bounds = (0, 1)
        curvature_bounds = (0.00001, 1)
        det_arr_angle_bounds = (-np.pi, np.pi)
        angle_bounds = (-np.pi / 2, np.pi / 2)
        tanh_len_bounds = (0, 1)
        # FIXME Allows for some invalid designs where the array intersects the prisms
        position_bounds = [(self.config.spectrometer.max_height + config.spectrometer.detector_array_length, self.config.spectrometer.max_height * 40), (-10 * self.config.spectrometer.max_height, 10 * self.config.spectrometer.max_height)]
        nglass_or_const_glasses = config.nglass_or_const_glasses
        self.use_gaussian_beam = config.spectrometer.beam_width is not None
        self.use_fiber_beam = config.spectrometer.fiber_core_radius is not None and config.spectrometer.numerical_aperture is not None
        self.auto_position_detector_acceptance = config.auto_position_detector_acceptance
        assert self.use_gaussian_beam or self.use_fiber_beam
        assert self.auto_position_detector_acceptance is None or 0 < self.auto_position_detector_acceptance <= 1

        if isinstance(nglass_or_const_glasses, int):
            nglass = nglass_or_const_glasses
            self._glasses = None
            glass_dtype_fields = [("glass_indices", (np.int64, (nglass,))),]
            glass_bounds = {"glass_indices": [catalog_bounds] * nglass,}
        else:
            self._glasses = [self.glass_dict[n] for n in nglass_or_const_glasses]
            if not all(isinstance(g, Glass) for g in self._glasses):
                raise TypeError(f"{nglass_or_const_glasses} is not a sequence of Glass")
            nglass = len(self._glasses)
            glass_dtype_fields = []
            glass_bounds = {}
        dtype_fields = [
            *glass_dtype_fields,
            ("angles", (np.float64, (nglass + 1,))),
            ("tanh_lengths", (np.float64, (nglass,))),
            ("curvature", (np.float64, (1 if self.use_gaussian_beam else 4,))),
            ("height", np.float64),
            ("normalized_y_mean", np.float64),
            ("detector_array_angle", np.float64),
        ]
        bounds = {
            **glass_bounds,
            "angles": [angle_bounds] * (nglass + 1), 
            "tanh_lengths": [tanh_len_bounds] * nglass,
            "curvature": [curvature_bounds] * (1 if self.use_gaussian_beam else 4),
            "height": height_bounds,
            "normalized_y_mean": normalized_y_mean_bounds,
            "detector_array_angle": det_arr_angle_bounds,
        }
        if self.auto_position_detector_acceptance is None:
            dtype_fields.append(("position", (np.float64, (2,))))
            bounds["position"] = position_bounds
        self._numpy_dtype = np.dtype(dtype_fields)
        xl, xu = zip(*itertools.chain.from_iterable(map(lambda v: v if isinstance(v, list) else [v], bounds.values())))

        fix_indicies = np.add.accumulate([np.prod(self._numpy_dtype[n].shape, dtype=np.int64) for n in self._numpy_dtype.names])[:-1]
        self._fix_params = lambda *params: np.array(tuple(a.squeeze() for a in np.split(params, fix_indicies)), dtype=self._numpy_dtype)

        if config.optimizer.parallelize and "parallelization" not in kwargs:
            pool = ThreadPool()
            kwargs["parallelization"] = ('starmap', pool.starmap)

        super().__init__(
            n_var=len(xl),
            n_obj=1 + int(config.optimizer.optimize_size) + int(config.optimizer.optimize_deviation),
            n_constr=1,
            xl=xl,
            xu=xu,
            **kwargs
        )

    def create_spectrometer(self, params: np.ndarray) -> Spectrometer:
        params = (self._fix_params)(*params)
        if self.use_gaussian_beam:
            curvature = params["curvature"]
        else:
            ic0, ic1, fc0, fc1 = params["curvature"]
            curvature = ((-ic0, ic1), (fc0, fc1))
        if self._glasses is None:
            glasses = [self.glass_list[i] for i in params["glass_indices"]]
        else:
            glasses = self._glasses
        compound_prism = CompoundPrism(
            glasses=glasses,
            angles=params["angles"],
            lengths=np.arctanh(params["tanh_lengths"]) * self.config.spectrometer.max_height / 4,
            curvature=curvature,
            height=params["height"],
            width=self.config.spectrometer.prism_width,
            ar_coated=self.config.spectrometer.ar_coated
        )
        # print(compound_prism)
        wavelengths = UniformWavelengthDistribution(self.config.spectrometer.wavelength_range)
        if self.use_gaussian_beam:
            beam = GaussianBeam(
                width=self.config.spectrometer.beam_width,
                y_mean=params["height"] * params["normalized_y_mean"],
            )
        else:
            beam = FiberBeam(
                core_radius=self.config.spectrometer.fiber_core_radius,
                numerical_aperture=self.config.spectrometer.numerical_aperture,
                center_y=params["height"] * params["normalized_y_mean"]
            )
        if self.auto_position_detector_acceptance is None:
            x, y = params["position"]
            position = x, y
            flipped = False
        else:
            position, flipped = position_detector_array(
                length=self.config.spectrometer.detector_array_length,
                angle=params["detector_array_angle"],
                compound_prism=compound_prism,
                wavelengths=wavelengths,
                beam=beam,
                acceptance=self.auto_position_detector_acceptance
            )
        detector_array=DetectorArray(
            bin_count=self.config.spectrometer.bin_count,
            bin_size=self.config.spectrometer.bin_size,
            linear_slope=self.config.spectrometer.linear_slope,
            linear_intercept=self.config.spectrometer.linear_intercept,
            length=self.config.spectrometer.detector_array_length,
            max_incident_angle=self.config.spectrometer.max_incident_angle,
            angle=params["detector_array_angle"],
            position = position,
            flipped = flipped,
        )
        # print(detector_array)
        return Spectrometer(compound_prism, detector_array, wavelengths, beam)

    def _evaluate(self, x, out, *args, **kwargs):
        max_size = (self.config.spectrometer.max_height * 40) if self.config.optimizer.optimize_size else np.inf
        max_info = np.log2(self.config.spectrometer.bin_count)
        try:
            spectrometer = self.create_spectrometer(x)
            fit = None
            if self.cpu_only:
                fit = spectrometer.cpu_fitness()
            else:
                fit = spectrometer.gpu_fitness(seeds=np.random.rand(1), max_n=128, max_eval=16_384 // 2)
                if fit is None:
                    raise RayTraceError()
            fit_info = np.log2(self.config.spectrometer.bin_count) - fit.info
            assert fit_info > 0
            result = []
            if self.config.optimizer.optimize_size:
                result.append(fit.size)
            result.append(fit_info)
            if self.config.optimizer.optimize_deviation:
                result.append(fit.deviation)
            out["F"] = result
            feasable_size = np.logical_and(self.config.spectrometer.max_height / 2 < fit.size, fit.size < max_size)
            out["feasible"] = feasable_size
            out["G"] = np.where(feasable_size, 0, 1)
        except RayTraceError:
            result = []
            if self.config.optimizer.optimize_size:
                result.append(max_size * 10)
            result.append(max_info)
            if self.config.optimizer.optimize_deviation:
                result.append(1)
            out["F"] = result
            out["feasible"] = False
            out["G"] = 1


class MetaCompoundPrismSpectrometerProblem(ElementwiseProblem):
    def __init__(self, max_nglass: int, minimizer: Callable[[CompoundPrismSpectrometerProblem], Sequence[Design]], config: CompoundPrismSpectrometerProblemConfig):
        self.minimizer = minimizer
        self.config = config
        super().__init__(
            n_var=max_nglass,
            n_obj=3,
            n_constr=0,
            xl=0,
            xu=1,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pass


with open("spring.toml") as f:
    spring_config = from_toml(CompoundPrismSpectrometerProblemConfig, f.read())

# glass_names = "N-SF66", "N-SF14", "N-BAF4"
# spring_config.nglass = None
# spring_config.glass_names = glass_names

# n_threads = 8
# pool = ThreadPool(n_threads)
spring_config.optimizer.cpu_only = True
# spring_config.optimizer.optimize_size = False
# spring_config.optimizer.optimize_deviation = False
spring_config.auto_position_detector_acceptance = 0.98
problem = CompoundPrismSpectrometerProblem(spring_config) #, cpu_only=True, parallelization = ('starmap', pool.starmap))


from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover

nglass = problem.config.nglass or 0
mask = [*(["int"] * nglass), *(["real"] * (problem.n_var - nglass))]

sampling = MixedVariableSampling(mask, {
    "real": get_sampling("real_lhs"),
    "int": get_sampling("int_random")
})

crossover = MixedVariableCrossover(mask, {
    "real": get_crossover("real_sbx", eta=15, prob=0.9),
    "int": get_crossover("int_sbx", eta=15, prob=0.9)
})

mutation = MixedVariableMutation(mask, {
    "real": get_mutation("real_pm", prob=None, eta=20),
    "int": get_mutation("int_pm", prob=None, eta=20)
})

algorithm_kwargs = dict(sampling=sampling, crossover=crossover, mutation=mutation)
pop_size = 1000
ref_dirs = get_reference_directions("energy", problem.n_obj, 1000)

# 'ga', 'brkga', 'de', 'nelder-mead', 'pattern-search', 'cmaes', 'nsga2', 'rnsga2', 'nsga3', 'unsga3', 'rnsga3', 'moead', 'pso'
method = "age"


if method in {"nsga2", "de", "ga", "pso"}:
    algorithm = get_algorithm(method, pop_size=pop_size, **algorithm_kwargs)
elif method == "age":
    from pymoo.algorithms.moo.age import AGEMOEA

    algorithm = AGEMOEA(pop_size=pop_size, **algorithm_kwargs)
elif method == "age2":
    from pymoo.algorithms.moo.age2 import AGEMOEA2

    algorithm = AGEMOEA2(pop_size=pop_size, **algorithm_kwargs)
else:
    algorithm = get_algorithm(method, ref_dirs=ref_dirs, **algorithm_kwargs)


result = minimize(
    problem,
    algorithm,
    termination=('n_gen', 200),
    verbose=True,
    save_history=True
)

if problem.n_obj == 1:
    spec = problem.create_spectrometer(result.X)
    design = Design(spectrometer=spec, fitness=spec.cpu_fitness())
    print(f"Best solution found: {design}")
    print(result.opt.get("feasible"))
else:
    def create_designs():
        for x, f in zip(result.X, result.opt.get("feasible")):
            if f:
                spec = problem.create_spectrometer(x)
                yield Design(spectrometer=spec, fitness=spec.cpu_fitness())


    designs = list(create_designs())
    for design in designs:
        print(design)
    interactive_show(designs)
