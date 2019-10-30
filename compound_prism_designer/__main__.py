#!/usr/bin/env python3
import pathlib
import toml
import numpy as np
import matplotlib as mpl
mpl.rcParams["backend"] = "qt5agg"
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
QFileDialog = mpl.backends.qt_compat.QtWidgets.QFileDialog

from .compound_prism_designer import create_glass_catalog, Glass, Design, RayTraceError
from .utils import draw_spectrometer
from .zemax import create_zmx
from .optimizer import DesignConfig


class Interactive:
    def __init__(self, fig: Figure, spec: DesignConfig):
        self.fig = fig
        self.spec = spec
        self.designs = spec.optimize()
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
        self.pareto_ax.set_xlabel(f"size ({spec.units})")
        self.pareto_ax.set_ylabel("mutual information (bits)")

        self.prism_ax = fig.add_subplot(gs[0, 1:3])
        self.det_ax = fig.add_subplot(gs[1, 1])
        self.trans_ax = fig.add_subplot(gs[1, 2])
        self.violin_ax = fig.add_subplot(gs[:, 3])
        self.prism_ax.axis('off')

        self.waves = np.linspace(*spec.gaussian_beam.wavelength_range, 100)
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

    def export_zemax(self, filename):
        path = pathlib.Path(filename)
        if self.selected_design is not None and not path.is_dir():
            zemax_design, zemax_file = create_zmx(
                self.selected_design.compound_prism,
                self.selected_design.detector_array,
                self.selected_design.gaussian_beam
            )
            with open(path.with_suffix('.zmx'), "w") as f:
                f.write(zemax_file)
            return zemax_design


fig = plt.figure()
manager = plt.get_current_fig_manager()
manager.set_window_title("Compound Prism Spectrometer Designer")
spec_file = pathlib.Path(QFileDialog.getOpenFileName(
    caption="Select design specification", filter="Configuration files (*.toml);;All files (*)")[0])
assert spec_file.is_file()
spec = DesignConfig.from_dict(toml.load(spec_file))

interactive = Interactive(fig, spec)

mbar = manager.window.menuBar()
export_action = mbar.addAction("Export Design as .zmx")
export_action.triggered.connect(lambda:
                                interactive.export_zemax(
                                    pathlib.Path(QFileDialog.getSaveFileName(caption="Export zemax file", filter="Zemax Lens File (*.zmx)")[0])))
plt.show()
