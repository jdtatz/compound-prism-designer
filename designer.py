#!/usr/bin/env python3
import itertools
from typing import Union, Sequence, Tuple, NamedTuple, Callable
import numpy as np
from pymoo.model.problem import Problem
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
from pymoo.optimize import minimize
from compound_prism_designer import RayTraceError, Glass, BUNDLED_CATALOG, CompoundPrism, \
    DetectorArray, GaussianBeam, Spectrometer
from compound_prism_designer.interactive import interactive_show, Design


class CompoundPrismSpectrometerProblemConfig(NamedTuple):
    glass_catalog: Sequence[Glass] = BUNDLED_CATALOG
    max_height: float = 25
    prism_width: float = 10
    bin_count: int = 16
    bin_size: float = 0.8
    linear_slope: float = 1
    linear_intercept: float = 0.1
    detector_array_length: float = 16
    max_incident_angle: float = np.deg2rad(45)
    wavelength_range: Tuple[float, float] = (0.5, 1.0)
    beam_width: float = 3.2
    ar_coated: bool = False


class CompoundPrismSpectrometerProblem(Problem):
    def __init__(self, nglass_or_const_glasses: Union[int, Sequence[Glass]], config: CompoundPrismSpectrometerProblemConfig):
        self.config = config
        catalog_bounds = 0, len(config.glass_catalog)
        height_bounds = (0.0001 * config.max_height, config.max_height)
        normalized_y_mean_bounds = (0, 1)
        curvature_bounds = (0.00001, 1)
        det_arr_angle_bounds = (-np.pi, np.pi)
        angle_bounds = (-np.pi / 2, np.pi / 2)
        len_bounds = (0, 10)

        if isinstance(nglass_or_const_glasses, int):
            nglass = nglass_or_const_glasses
            self._glasses = None
            self._numpy_dtype = np.dtype([
                ("glass_find", (np.float64, nglass)),
                ("angles", (np.float64, nglass + 1)),
                ("lengths", (np.float64, nglass)),
                ("curvature", np.float64),
                ("height", np.float64),
                ("normalized_y_mean", np.float64),
                ("detector_array_angle", np.float64),
            ])
            bounds = {
                "glass_find": [catalog_bounds] * nglass,
                "angles": [angle_bounds] * (nglass + 1),
                "lengths": [len_bounds] * nglass,
                "curvature": curvature_bounds,
                "height": height_bounds,
                "normalized_y_mean": normalized_y_mean_bounds,
                "detector_array_angle": det_arr_angle_bounds,
            }
        else:
            glasses = tuple(nglass_or_const_glasses)
            if not all(isinstance(g, Glass) for g in glasses):
                raise TypeError(f"{nglass_or_const_glasses} is not a sequence of Glass")
            self._glasses = glasses
            nglass = len(self._glasses)
            self._numpy_dtype = np.dtype([
                ("angles", (np.float64, nglass + 1)),
                ("lengths", (np.float64, nglass)),
                ("curvature", np.float64),
                ("height", np.float64),
                ("normalized_y_mean", np.float64),
                ("detector_array_angle", np.float64),
            ])
            bounds = {
                "angles": [angle_bounds] * (nglass + 1),
                "lengths": [len_bounds] * nglass,
                "curvature": curvature_bounds,
                "height": height_bounds,
                "normalized_y_mean": normalized_y_mean_bounds,
                "detector_array_angle": det_arr_angle_bounds,
            }
        bounds = list(itertools.chain.from_iterable(map(lambda v: v if isinstance(v, list) else [v], (bounds[f] for f in self._numpy_dtype.fields))))
        bounds = list(zip(*bounds))
        bounds = np.array(bounds)

        super().__init__(
            n_var=bounds.shape[1],
            n_obj=3,
            n_constr=1,
            xl=bounds[0],
            xu=bounds[1],
            elementwise_evaluation=True,
        )

    def create_spectrometer(self, params: np.ndarray) -> Spectrometer:
        params = params.view(self._numpy_dtype)[0]
        return Spectrometer(
            CompoundPrism(
                glasses=self._glasses if self._glasses is not None else [self.config.glass_catalog[int(np.clip(i, 0, len(self.config.glass_catalog) - 1))] for i in params["glass_find"]],
                angles=params["angles"],
                lengths=params["lengths"],
                curvature=params["curvature"],
                height=params["height"],
                width=self.config.prism_width,
                ar_coated=self.config.ar_coated
            ), DetectorArray(
                bin_count=self.config.bin_count,
                bin_size=self.config.bin_size,
                linear_slope=self.config.linear_slope,
                linear_intercept=self.config.linear_intercept,
                length=self.config.detector_array_length,
                max_incident_angle=self.config.max_incident_angle,
                angle=params["detector_array_angle"]
            ),
            GaussianBeam(
                wavelength_range=self.config.wavelength_range,
                width=self.config.beam_width,
                y_mean=params["height"] * params["normalized_y_mean"]
            )
        )

    def _evaluate(self, x, out, *args, **kwargs):
        try:
            spectrometer = self.create_spectrometer(x)
            fit = spectrometer.gpu_fitness()
            if fit is None:
                fit = spectrometer.cpu_fitness()
            out["F"] = fit.size, np.log2(self.config.bin_count) - fit.info, fit.deviation
            out["feasible"] = fit.size < 800
            out["G"] = 0 if fit.size < 800 else 1
        except RayTraceError:
            out["F"] = 1e4, np.log2(self.config.bin_count), 1
            out["feasible"] = False
            out["G"] = 1


class MetaCompoundPrismSpectrometerProblem(Problem):
    def __init__(self, max_nglass: int, minimizer: Callable[[CompoundPrismSpectrometerProblem], Sequence[Design]], config: CompoundPrismSpectrometerProblemConfig):
        self.minimizer = minimizer
        self.config = config
        super().__init__(
            n_var=max_nglass,
            n_obj=3,
            n_constr=0,
            xl=0,
            xu=1,
            elementwise_evaluation=True,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        pass


spring_config = CompoundPrismSpectrometerProblemConfig(
    max_height=25,
    prism_width=7,
    bin_count=32,
    bin_size=0.8,
    linear_slope=1,
    linear_intercept=0.1,
    detector_array_length=32,
    max_incident_angle=np.deg2rad(45),
    wavelength_range=(0.5, 0.82),
    beam_width=3.2,
    ar_coated=True,
)
bch_config = CompoundPrismSpectrometerProblemConfig(
    max_height=20,
    prism_width=6.6,
    bin_count=64,
    bin_size=0.42,
    linear_slope=0.42,
    linear_intercept=0,
    detector_array_length=26.6,
    max_incident_angle=np.deg2rad(45),
    wavelength_range=(0.48, 1.0),
    beam_width=2,
    ar_coated=True,
)
# glass_cat = {g.name: g for g in BUNDLED_CATALOG}
# glass_names = "N-SF66", "N-SF14", "N-BAF4"
# glasses = [glass_cat[n] for n in glass_names]
# ndim = len(glasses)
problem = CompoundPrismSpectrometerProblem(3, bch_config)

ref_dirs = RieszEnergyReferenceDirectionFactory(n_dim=problem.n_obj, n_points=90).do()
# 'ga', 'brkga', 'de', 'nelder-mead', 'pattern-search', 'cmaes', 'nsga2', 'rnsga2', 'nsga3', 'unsga3', 'rnsga3', 'moead'
algorithm = get_algorithm("unsga3", ref_dirs, pop_size=100)

result = minimize(
    problem,
    algorithm,
    termination=('n_gen', 200),
    verbose=True
)


def create_designs():
    for x, f in zip(result.X, result.opt.get("feasible")):
        if f:
            spec = problem.create_spectrometer(x)
            yield Design(spectrometer=spec, fitness=spec.cpu_fitness())


designs = list(create_designs())
interactive_show(designs)
