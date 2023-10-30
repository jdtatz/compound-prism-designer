#!/usr/bin/env python3
from dataclasses import dataclass, field
from functools import partial, reduce
from multiprocessing.pool import ThreadPool
from operator import mul
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from pymoo.core.mixed import (
    MixedVariableDuplicateElimination,
    MixedVariableMating,
    MixedVariableSampling,
)
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Integer, Real
from pymoo.optimize import minimize
from serde import serde
from serde.toml import from_toml

from compound_prism_designer import (
    BUNDLED_CATALOG,
    AbstractGlass,
    CompoundPrism,
    DetectorArray,
    FiberBeam,
    GaussianBeam,
    Glass,
    RayTraceError,
    Spectrometer,
    UniformWavelengthDistribution,
    new_catalog,
    position_detector_array,
)
from compound_prism_designer.interactive import Design, interactive_show

rprod = partial(reduce, mul)


@serde(rename_all="kebabcase")
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


@serde(rename_all="kebabcase")
@dataclass
class OptimizerConfig:
    cpu_only: bool = False
    parallelize: bool = False
    optimize_size: bool = True
    optimize_deviation: bool = True


@serde(rename_all="kebabcase")
@dataclass
class CompoundPrismSpectrometerProblemConfig:
    spectrometer: CompoundPrismSpectrometerConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

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
        self.glass_dict = {g.name: g for g in self.glass_list}
        catalog_bounds = 0, len(self.glass_list) - 1
        height_bounds = (
            0.0001 * config.spectrometer.max_height,
            config.spectrometer.max_height,
        )
        normalized_y_mean_bounds = (0, 1)
        curvature_bounds = (0.00001, 1)
        det_arr_angle_bounds = (-np.pi, np.pi)
        angle_bounds = (-np.pi / 2, np.pi / 2)
        tanh_len_bounds = (0, 1)
        # FIXME Allows for some invalid designs where the array intersects the prisms
        position_bounds = [(0, self.config.spectrometer.max_height * 40), angle_bounds]
        nglass_or_const_glasses = config.nglass_or_const_glasses
        self.use_gaussian_beam = config.spectrometer.beam_width is not None
        self.use_fiber_beam = (
            config.spectrometer.fiber_core_radius is not None and config.spectrometer.numerical_aperture is not None
        )
        self.auto_position_detector_acceptance = config.auto_position_detector_acceptance
        assert self.use_gaussian_beam or self.use_fiber_beam
        assert self.auto_position_detector_acceptance is None or 0 < self.auto_position_detector_acceptance <= 1

        if isinstance(nglass_or_const_glasses, int):
            nglass = nglass_or_const_glasses
            self._glasses = None
            glass_vars = {f"glass_idx_{i}": Integer(bounds=catalog_bounds) for i in range(nglass)}
        else:
            self._glasses = [self.glass_dict[n] for n in nglass_or_const_glasses]
            if not all(isinstance(g, Glass) for g in self._glasses):
                raise TypeError(f"{nglass_or_const_glasses} is not a sequence of Glass")
            nglass = len(self._glasses)
            glass_vars = {}
        self.nglass = nglass
        if self.use_gaussian_beam:
            curvature_vars = {"curvature": Real(bounds=curvature_bounds)}
        else:
            curvature_vars = {f"curvature_{i}": Real(bounds=curvature_bounds) for i in range(4)}
        all_vars = {
            **glass_vars,
            **{f"angle_{i}": Real(bounds=angle_bounds) for i in range(1 + nglass)},
            **{f"tanh_len_{i}": Real(bounds=tanh_len_bounds) for i in range(nglass)},
            **curvature_vars,
            "height": Real(bounds=height_bounds),
            "normalized_y_mean": Real(bounds=normalized_y_mean_bounds),
            "detector_array_angle": Real(bounds=det_arr_angle_bounds),
        }
        if self.auto_position_detector_acceptance is None:
            all_vars["position_dv"], all_vars["position_da"] = (Real(bounds=b) for b in position_bounds)

        if config.optimizer.parallelize and "parallelization" not in kwargs:
            pool = ThreadPool()
            kwargs["parallelization"] = ("starmap", pool.starmap)

        for k, v in all_vars.items():
            print("   ", k, "=", type(v), v.bounds)
        super().__init__(
            vars=all_vars,
            n_obj=1 + int(config.optimizer.optimize_size) + int(config.optimizer.optimize_deviation),
            n_constr=1,
            **kwargs,
        )
        print(self)

    def create_spectrometer(self, params: np.ndarray) -> Spectrometer:
        if self.use_gaussian_beam:
            curvature = params["curvature"]
        else:
            ic0, ic1, fc0, fc1 = (params[f"curvature_{i}"] for i in range(4))
            curvature = ((-ic0, ic1), (fc0, fc1))
        if self._glasses is None:
            glasses = [self.glass_list[params[f"glass_idx_{i}"]] for i in range(self.nglass)]
        else:
            glasses = self._glasses
        compound_prism = CompoundPrism(
            glasses=glasses,
            angles=np.array([params[f"angle_{i}"] for i in range(1 + self.nglass)]),
            lengths=np.arctanh([params[f"tanh_len_{i}"] for i in range(self.nglass)])
            * self.config.spectrometer.max_height
            / 4,
            curvature=curvature,
            height=params["height"],
            width=self.config.spectrometer.prism_width,
            ar_coated=self.config.spectrometer.ar_coated,
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
                center_y=params["height"] * params["normalized_y_mean"],
            )
        if self.auto_position_detector_acceptance is None:
            dv, da = params["position_dv"], params["position_da"]
            c, s = np.cos(da), np.sin(da)
            R = np.array(((c, -s), (s, c)))
            exit_pos, exit_dir = compound_prism.final_midpt_and_normal()
            det_dv_dir = R @ (exit_dir.x, exit_dir.y)
            position = exit_pos.x + det_dv_dir[0] * dv, exit_pos.y + det_dv_dir[1] * dv
            flipped = False
        else:
            position, flipped = position_detector_array(
                length=self.config.spectrometer.detector_array_length,
                angle=params["detector_array_angle"],
                compound_prism=compound_prism,
                wavelengths=wavelengths,
                beam=beam,
                acceptance=self.auto_position_detector_acceptance,
            )
        detector_array = DetectorArray(
            bin_count=self.config.spectrometer.bin_count,
            bin_size=self.config.spectrometer.bin_size,
            linear_slope=self.config.spectrometer.linear_slope,
            linear_intercept=self.config.spectrometer.linear_intercept,
            length=self.config.spectrometer.detector_array_length,
            max_incident_angle=self.config.spectrometer.max_incident_angle,
            angle=params["detector_array_angle"],
            position=position,
            flipped=flipped,
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
            out["G"] = np.where(feasable_size, 0, 1)
        except RayTraceError:
            result = []
            if self.config.optimizer.optimize_size:
                result.append(max_size * 10)
            result.append(max_info)
            if self.config.optimizer.optimize_deviation:
                result.append(1)
            out["F"] = result
            out["G"] = 1


class MetaCompoundPrismSpectrometerProblem(ElementwiseProblem):
    def __init__(
        self,
        max_nglass: int,
        minimizer: Callable[[CompoundPrismSpectrometerProblem], Sequence[Design]],
        config: CompoundPrismSpectrometerProblemConfig,
    ):
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
# spring_config.auto_position_detector_acceptance = 0.98
problem = CompoundPrismSpectrometerProblem(
    spring_config
)  # , cpu_only=True, parallelization = ('starmap', pool.starmap))

sampling = MixedVariableSampling()
eliminate_duplicates = MixedVariableDuplicateElimination()
mating = MixedVariableMating(eliminate_duplicates=eliminate_duplicates)
pop_size = 1000


def get_algorithm(algorithm: str, **kwargs):
    # NOTE: The AGEMOEA algos require numba
    if algorithm == "age":
        from pymoo.algorithms.moo.age import AGEMOEA

        return AGEMOEA(**kwargs)
    elif algorithm == "age2":
        from pymoo.algorithms.moo.age2 import AGEMOEA2

        return AGEMOEA2(**kwargs)
    elif algorithm == "nsga2":
        from pymoo.algorithms.moo.nsga2 import NSGA2

        return NSGA2(**kwargs)
    else:
        raise NotImplementedError


# 'age', 'age2', 'nsga2'
method = "age2"

algorithm = get_algorithm(
    method,
    pop_size=pop_size,
    sampling=sampling,
    mating=mating,
    eliminate_duplicates=eliminate_duplicates,
)
result = minimize(problem, algorithm, termination=("n_gen", 200), verbose=True, save_history=True)

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
