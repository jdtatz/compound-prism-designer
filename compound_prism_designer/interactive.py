from __future__ import annotations
import numpy as np
from pickle import dump
from typing import Sequence, Optional
from dataclasses import dataclass, field
import matplotlib as mpl

mpl.rcParams["backend"] = "qt5agg"
from matplotlib.figure import Figure
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt

try:
    from PySide2.QtWidgets import QFileDialog
except ImportError:
    from PyQt5.QtWidgets import QFileDialog  # type:ignore
plt.switch_backend("qt5agg")
from pathlib import Path

from .compound_prism_designer import Spectrometer, DesignFitness
from .utils import draw_spectrometer
from .asap import create_asap_macro


@dataclass(frozen=True)
class Design:
    spectrometer: Spectrometer
    fitness: DesignFitness


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

        self.selected_design: Optional[Design] = None
        fig.canvas.mpl_connect('pick_event', self.pick_design)

    def pick_design(self, event):
        design = self.selected_design = sorted((self.designs[i] for i in event.ind), key=lambda s: -s.fitness.info)[0]
        spectrometer = design.spectrometer
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
        path = Path(filename)
        if path.is_dir():
            print(f"Warning: given '{path}' is not a file, using default")
            path = Path.cwd().joinpath("results")

        with open(path.with_suffix('.pkl'), "wb") as f:
            dump(self.designs, f)

    def export_asap(self):
        if self.selected_design is not None:
            filename, _ = QFileDialog.getSaveFileName(caption="Export design as ASAP .inr",
                                                      filter="ASAP script (*.inr)")
            path = Path(filename)
            if path.is_dir():
                print(f"Warning: given '{path}' is not a file, using default")
                path = Path.cwd().joinpath("results")

            with open(path.with_suffix('.inr'), "w") as f:
                asap = create_asap_macro(self.selected_design.spectrometer)
                f.write(asap)
            print("Saving ASAP script to", path.with_suffix('.inr'))
            print(asap)
        else:
            print("WARNING: No design selected")


def interactive_show(designs: Sequence[Design]):
    fig = plt.figure()
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Compound Prism Spectrometer Designer")

    interactive = Interactive(fig, designs)

    mbar = manager.window.menuBar()

    save_action = mbar.addAction("Save Designs")
    save_action.triggered.connect(interactive.save_designs)

    export_action = mbar.addAction("Export Design as ASAP .inr")
    export_action.triggered.connect(interactive.export_asap)

    plt.show()
