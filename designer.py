#!/usr/bin/env python3
import pathlib
import gzip
import numpy as np
import matplotlib as mpl
mpl.rcParams["backend"] = "qt5agg"
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
try:
    from PySide2.QtWidgets import QFileDialog
except ImportError:
    from PyQt5.QtWidgets import QFileDialog
plt.switch_backend("qt5agg")


from compound_prism_designer import GlassCatalogError, RayTraceError, Glass, BUNDLED_CATALOG, CompoundPrism, \
    DetectorArray, GaussianBeam, Spectrometer, DesignFitness, draw_spectrometer, create_asap_macro
from pymoo.model.problem import Problem
from pymoo.factory import get_reference_directions
from pymoo.util.ref_dirs.energy import RieszEnergyReferenceDirectionFactory
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.algorithms.nsga3 import NSGA3
from pymoo.algorithms.moead import MOEAD
from pymoo.optimize import minimize
import numpy as np
from pickle import dump
import matplotlib.pyplot as plt
import itertools
from typing import Union, Sequence, Tuple, NamedTuple


class Design(NamedTuple):
    spectrometer: Spectrometer
    fitness: DesignFitness


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
            # nglass = nglass_or_const_glasses
            # self._numpy_dtype = np.dtype([])
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
            n_constr=0,
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
            out["F"] = (fit.size, np.log2(32) - fit.info, fit.deviation)
            out["feasible"] = fit.size < 800
        except RayTraceError:
            out["F"] = (1e4, np.log2(32), 1)
            out["feasible"] = False


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
glass_cat = {g.name: g for g in BUNDLED_CATALOG}
glass_names = "N-SF66", "N-SF14", "N-BAF4"
glasses = [glass_cat[n] for n in glass_names]
problem = CompoundPrismSpectrometerProblem(3, spring_config)

ref_dirs = RieszEnergyReferenceDirectionFactory(n_dim=3, n_points=90).do()
algorithm = NSGA2(pop_size=100)

result = minimize(
    problem,
    algorithm,
    termination=('n_gen', 200),
    verbose=True
)


def create_designs():
    for x in result.X:
        spec = problem.create_spectrometer(x)
        yield Design(spectrometer=spec, fitness=spec.cpu_fitness())


designs = list(create_designs())


class Interactive:
    def __init__(self, fig: Figure, designs: Sequence[Design]):
        self.fig = fig
        self.designs = designs
        nrows, ncols = 2, 4
        gs = fig.add_gridspec(nrows, ncols)
        self.pareto_ax = fig.add_subplot(gs[:, 0])

        x, y, c = np.array([(design.fitness.size, design.fitness.info, design.fitness.deviation)
                            for design in self.designs]).T
        c = np.rad2deg(np.arcsin(c))

        cm = mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 90, clip=True), 'PuRd')
        sc = self.pareto_ax.scatter(x, y, c=c, cmap=cm.cmap.reversed(), norm=cm.norm, picker=True)
        clb = fig.colorbar(sc)
        clb.ax.set_ylabel("deviation (deg)")
        self.pareto_ax.set_xlabel(f"size (a.u.)")
        self.pareto_ax.set_ylabel("mutual information (bits)")

        self.prism_ax = fig.add_subplot(gs[0, 1:3])
        self.det_ax = fig.add_subplot(gs[1, 1])
        self.trans_ax = fig.add_subplot(gs[1, 2])
        self.violin_ax = fig.add_subplot(gs[:, 3])
        self.prism_ax.axis('off')

        self.selected_design = None
        fig.canvas.mpl_connect('pick_event', self.pick_design)

    def pick_design(self, event):
        design = self.selected_design = sorted((self.designs[i] for i in event.ind), key=lambda s: -s.fitness.info)[0]
        spectrometer = design.spectrometer
        print("[Spectrometer]")
        print(create_asap_macro(spectrometer))
        waves = np.linspace(*spectrometer.gaussian_beam.wavelength_range, 200)

        transmission = spectrometer.transmission_probability(waves)
        p_det = transmission.sum(axis=0) * (1 / len(waves))
        p_w_l_D = transmission.sum(axis=1)

        # Draw Spectrometer
        beam = spectrometer.gaussian_beam
        wmin, wmax = beam.wavelength_range
        draw_spectrometer(
            self.prism_ax,
            spectrometer,
            zip((wmin, (wmin + wmax) / 2, wmax), ('r', 'g', 'b')),
            (beam.y_mean - beam.width, beam.y_mean, beam.y_mean + beam.width)
        )

        # Plot Design Results
        self.det_ax.clear()
        self.trans_ax.cla()
        self.violin_ax.cla()

        vpstats = [
            {
                "coords": waves,
                "vals": t,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }
            for t in transmission.T
        ]
        parts = self.violin_ax.violin(vpstats, showextrema=False, widths=1)
        for pc in parts['bodies']:
            pc.set_facecolor('black')
        self.violin_ax.plot([1, len(p_det)], spectrometer.gaussian_beam.wavelength_range, 'k--')
        self.det_ax.scatter(1 + np.arange(len(p_det)), p_det * 100, color='k')
        self.trans_ax.plot(waves, p_w_l_D * 100, 'k')
        self.trans_ax.axhline(np.mean(p_w_l_D) * 100, color='k', linestyle='--')

        self.det_ax.set_xlabel("detector bins")
        self.trans_ax.set_xlabel("wavelength (μm)")
        self.violin_ax.set_xlabel("detector bins")
        self.det_ax.set_ylabel("p (%)")
        self.trans_ax.set_ylabel("p (%)")
        self.violin_ax.set_ylabel("wavelength (μm)")
        self.det_ax.set_title("p(D=d|Λ)")
        self.trans_ax.set_title("p(D|Λ=λ)")
        self.violin_ax.set_title("p(D=d|Λ=λ) as a Pseudo Violin Plot")
        self.trans_ax.set_ylim(0, 100)
        self.det_ax.set_ylim(bottom=0)

        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_designs(self):
        filename, _ = QFileDialog.getSaveFileName(caption="Export saved designs", filter="Saved results (*.pkl)")
        path = pathlib.Path(filename)
        if path.is_dir():
            print(f"Warning: given '{path}' is not a file, using default")
            path = pathlib.Path.cwd().joinpath("results")

        with open(path.with_suffix('.pkl'), "wb") as f:
            dump(self.designs, f)


fig = plt.figure()
manager = plt.get_current_fig_manager()
manager.set_window_title("Compound Prism Spectrometer Designer")

interactive = Interactive(fig, designs)

mbar = manager.window.menuBar()

save_action = mbar.addAction("Save Designs")
save_action.triggered.connect(interactive.save_designs)

# export_action = mbar.addAction("Export Design as .zmx")
# export_action.triggered.connect(interactive.export_zemax)

plt.show()
