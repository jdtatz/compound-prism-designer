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
from .compound_prism_designer import create_glass_catalog, Glass, Design, DesignConfig, RayTraceError, \
    serialize_results, deserialize_results, config_from_toml
from .utils import draw_spectrometer
from .zemax import create_zmx


class Interactive:
    def __init__(self, fig: Figure, spec: DesignConfig, designs: [Design]):
        self.fig = fig
        self.spec = spec
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
        self.pareto_ax.set_xlabel(f"size ({spec.length_unit})")
        self.pareto_ax.set_ylabel("mutual information (bits)")

        self.prism_ax = fig.add_subplot(gs[0, 1:3])
        self.det_ax = fig.add_subplot(gs[1, 1])
        self.trans_ax = fig.add_subplot(gs[1, 2])
        self.violin_ax = fig.add_subplot(gs[:, 3])
        self.prism_ax.axis('off')

        self.waves = np.linspace(*spec.gaussian_beam.wavelength_range, 200)
        self.selected_design = None
        fig.canvas.mpl_connect('pick_event', self.pick_design)

    def pick_design(self, event):
        design = self.selected_design = sorted((self.designs[i] for i in event.ind), key=lambda s: -s.fitness.info)[0]

        transmission = design.transmission_probability(self.waves)
        p_det = transmission.sum(axis=0) * (1 / len(self.waves))
        p_w_l_D = transmission.sum(axis=1)
        zemax_design, zemax_file = create_zmx(design.compound_prism, design.detector_array, design.gaussian_beam)
        print(zemax_design)

        # Draw Spectrometer
        self.prism_ax.cla()
        draw_spectrometer(self.prism_ax, design.compound_prism, design.detector_array)

        wmin, wmax = design.gaussian_beam.wavelength_range
        for w, color in zip((wmin, (wmin + wmax) / 2, wmax), ('r', 'g', 'b')):
            for y in (design.gaussian_beam.y_mean - design.gaussian_beam.width, design.gaussian_beam.y_mean, design.gaussian_beam.y_mean + design.gaussian_beam.width):
                try:
                    ray = design.ray_trace(w, y)
                    poly = mpl.patches.Polygon(ray, closed=None, fill=None, edgecolor=color)
                    self.prism_ax.add_patch(poly)
                except RayTraceError:
                    pass
        self.prism_ax.axis('scaled')
        self.prism_ax.axis('off')

        # Plot Design Results
        self.det_ax.clear()
        self.trans_ax.cla()
        self.violin_ax.cla()

        vpstats = [
            {
                "coords": self.waves,
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
        self.violin_ax.plot([1, len(p_det)], self.spec.gaussian_beam.wavelength_range, 'k--')
        self.det_ax.scatter(1 + np.arange(len(p_det)), p_det * 100, color='k')
        self.trans_ax.plot(self.waves, p_w_l_D * 100, 'k')
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
        filename, _ = QFileDialog.getSaveFileName(caption="Export saved designs", filter="Saved results (*.cbor.gz)")
        path = pathlib.Path(filename)
        if path.is_dir():
            print(f"Warning: given '{path}' is not a file, using default")
            path = pathlib.Path.cwd().joinpath("results")

        with open(path.with_suffix('.cbor.gz'), "wb") as f:
            f.write(gzip.compress(serialize_results(self.spec, self.designs)))

    def export_zemax(self):
        filename, _ = QFileDialog.getSaveFileName(caption="Export zemax file", filter="Zemax Lens File (*.zmx)")
        path = pathlib.Path(filename)
        if self.selected_design is not None:
            if path.is_dir():
                print(f"Warning: given '{path}' is not a file, using default")
                path = pathlib.Path.cwd().joinpath("default.zmx")

            zemax_design, zemax_file = create_zmx(
                self.selected_design.compound_prism,
                self.selected_design.detector_array,
                self.selected_design.gaussian_beam
            )
            with open(path.with_suffix('.zmx'), "w") as f:
                f.write(zemax_file)
            return zemax_design


print(Design, dir(DesignConfig))
fig = plt.figure()
manager = plt.get_current_fig_manager()
manager.set_window_title("Compound Prism Spectrometer Designer")
spec_file = pathlib.Path(QFileDialog.getOpenFileName(
    caption="Select design specification Or previous results", filter="Configuration files (*.toml);;Saved results (*.cbor);;Compressed Saved results (*.cbor.gz);;All files (*)")[0])
if not spec_file.is_file():
    print(f"Warning: given '{spec_file}' is not a file, using default configuration file")
    spec_file = pathlib.Path.cwd().joinpath("design_config.toml")

if spec_file.suffix == ".cbor":
    with open(spec_file, "rb") as f:
        spec, designs = deserialize_results(f.read())
        print(spec, designs[0])
elif spec_file.suffix == ".gz":
    with open(spec_file, "rb") as f:
        spec, designs = deserialize_results(gzip.decompress(f.read()))
        print(spec, designs[0])
else:
    with open(spec_file, "r") as f:
        spec = config_from_toml(f.read())
    designs = spec.optimize()

interactive = Interactive(fig, spec, designs)

mbar = manager.window.menuBar()

save_action = mbar.addAction("Save Designs")
save_action.triggered.connect(interactive.save_designs)

export_action = mbar.addAction("Export Design as .zmx")
export_action.triggered.connect(interactive.export_zemax)

plt.show()
