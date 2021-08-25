from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from pickle import dump
from typing import Callable, Optional, Sequence

import matplotlib as mpl
import numpy as np

mpl.rcParams["toolbar"] = "toolmanager"

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.backend_tools import ToolBase
from matplotlib.figure import Figure

from .asap import create_asap_macro
from .compound_prism_designer import DesignFitness, Spectrometer
from .utils import draw_spectrometer


@dataclass(frozen=True)
class Design:
    spectrometer: Spectrometer
    fitness: DesignFitness


class Interactive:
    def __init__(self, fig: Figure, designs: Sequence[Design]):
        self.fig = fig
        self.designs = designs
        self.fig.set_constrained_layout(True)
        nrows, ncols = 2, 4
        gs = fig.add_gridspec(nrows, ncols)
        self.pareto_ax = fig.add_subplot(gs[:, 0])

        x, y, c = np.array(
            [
                (design.fitness.size, design.fitness.info, design.fitness.deviation)
                for design in self.designs
            ]
        ).T
        c = np.rad2deg(np.arcsin(c))

        cm = mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 90, clip=True), "PuRd")
        sc = self.pareto_ax.scatter(
            x, y, c=c, cmap=cm.cmap.reversed(), norm=cm.norm, picker=True
        )
        clb = fig.colorbar(sc)
        clb.ax.set_ylabel("Deviation (deg)")
        self.pareto_ax.set_xlabel(f"Size (a.u.)")
        self.pareto_ax.set_ylabel("Mutual Information (bits)")

        max_info = max(
            np.log2(design.spectrometer.detector_array.bin_count)
            for design in self.designs
        )
        secondary_y = self.pareto_ax.secondary_yaxis(
            "right",
            functions=(lambda i: 100 * i / max_info, lambda e: e * max_info / 100),
        )
        secondary_y.set_ylabel("Information Efficiency (%)")

        self.prism_ax = fig.add_subplot(gs[0, 1:3])
        self.det_ax = fig.add_subplot(gs[1, 1])
        self.trans_ax = fig.add_subplot(gs[1, 2])
        self.violin_ax = fig.add_subplot(gs[:, 3])
        self.prism_ax.axis("off")

        self.selected_design: Optional[Design] = None
        fig.canvas.mpl_connect("pick_event", self.pick_design)

        fig.canvas.manager.toolmanager.add_tool(
            "SaveDesigns", SaveDesignsTool, designs=self.designs
        )
        fig.canvas.manager.toolbar.add_tool("SaveDesigns", "io", -1)

        fig.canvas.manager.toolmanager.add_tool(
            "ExportASAP", ExportDesignTool, get_selected=lambda: self.selected_design
        )
        fig.canvas.manager.toolbar.add_tool("ExportASAP", "io", -1)

    def pick_design(self, event):
        design = self.selected_design = sorted(
            (self.designs[i] for i in event.ind), key=lambda s: -s.fitness.info
        )[0]
        spectrometer = design.spectrometer
        waves = np.linspace(*spectrometer.wavelengths.bounds, 200)

        transmission = spectrometer.transmission_probability(waves)
        p_det = transmission.sum(axis=0) * (1 / len(waves))
        p_w_l_D = transmission.sum(axis=1)

        # Draw Spectrometer
        beam = spectrometer.beam
        wmin, wmax = spectrometer.wavelengths.bounds
        draw_spectrometer(
            self.prism_ax,
            spectrometer,
            zip((wmin, (wmin + wmax) / 2, wmax), ("r", "g", "b")),
            (beam.y_mean - beam.width, beam.y_mean, beam.y_mean + beam.width),
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
        for pc in parts["bodies"]:
            pc.set_facecolor("black")
        self.violin_ax.plot(
            [1, len(p_det)], spectrometer.wavelengths.bounds, "k--"
        )
        self.det_ax.scatter(1 + np.arange(len(p_det)), p_det * 100, color="k")
        self.trans_ax.plot(waves, p_w_l_D * 100, "k")
        self.trans_ax.axhline(np.mean(p_w_l_D) * 100, color="k", linestyle="--")

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

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def get_save_file_name(
    caption: str, filter_name: str, filter_ext: str, default_name: str,
) -> Optional[str]:
    # interactive frameworks = Qt, Gtk3, Wx, Tk, macosx, WebAgg, & nbAgg
    bknd = plt.get_backend().lower()
    default_fname = f"{default_name}.{filter_ext}"
    if "qt" in bknd:
        from matplotlib.backends.qt_compat import QtWidgets

        filter = f"{filter_name} (*.{filter_ext})"
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            caption=caption, filter=filter, dir=default_fname
        )
        return filename
    elif "gtk3" in bknd:
        from matplotlib.backends.backend_gtk3 import Gtk

        dialog = Gtk.FileChooserDialog(
            title=caption,
            parent=plt.gcf().canvas.get_toplevel(),
            action=Gtk.FileChooserAction.SAVE,
            buttons=(
                Gtk.STOCK_CANCEL,
                Gtk.ResponseType.CANCEL,
                Gtk.STOCK_SAVE,
                Gtk.ResponseType.OK,
            ),
        )
        ff = Gtk.FileFilter()
        ff.set_name(filter_name)
        ff.add_pattern("*." + filter_ext)
        dialog.add_filter(ff)

        filter_any = Gtk.FileFilter()
        filter_any.set_name("Any files")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)

        dialog.set_current_name(default_fname)
        dialog.set_do_overwrite_confirmation(True)

        response = dialog.run()
        if response != Gtk.ResponseType.OK:
            fname = None
        else:
            fname = dialog.get_filename()
        dialog.destroy()
        return fname
    else:
        # raise NotImplementedError("The only supported backends for saving are: Qt5 & Gtk3")
        import warnings
        warnings.warn("The only supported backends for choosing save path are: Qt5 & Gtk3")
        return default_fname


class SaveDesignsTool(ToolBase):
    description = "Save Spectrometer Designs"
    # image = "filesave"

    def __init__(self, *args, designs: Sequence[Design], **kwargs):
        self.designs = designs
        super().__init__(*args, **kwargs)

    def trigger(self, sender=None, event=None, data=None, *args, **kwargs):
        filename = get_save_file_name(
            caption="Export saved designs",
            filter_name="Saved results",
            filter_ext="pkl",
            default_name="results",
        )
        path = Path(filename)
        if path.is_dir():
            print(f"Warning: given '{path}' is not a file, using default")
            path = Path.cwd().joinpath("results")

        with open(path.with_suffix(".pkl"), "wb") as f:
            dump(self.designs, f)


class ExportDesignTool(ToolBase):
    description = "Export Selected Design as ASAP .inr"
    # image = "filesave"

    def __init__(self, *args, get_selected: Callable[[], Design], **kwargs):
        self.get_selected = get_selected
        super().__init__(*args, **kwargs)

    def trigger(self, sender=None, event=None, data=None, *args, **kwargs):
        selected_design = self.get_selected()
        if selected_design is not None:
            filename = get_save_file_name(
                caption="Export design as ASAP .inr",
                filter_name="ASAP script",
                filter_ext="inr",
                default_name="design",
            )
            path = Path(filename)
            if path.is_dir():
                print(f"Warning: given '{path}' is not a file, using default")
                path = Path.cwd().joinpath("results")

            with open(path.with_suffix(".inr"), "w") as f:
                asap = create_asap_macro(selected_design.spectrometer)
                f.write(asap)
            print("Saving ASAP script to", path.with_suffix(".inr"))
            print(asap)
        else:
            print("WARNING: No design selected")


def interactive_show(designs: Sequence[Design]):
    assert (
        mpl.rcParams["toolbar"] == "toolmanager"
    ), "Compound Prism Spectrometer Designer requires toolmanager"
    assert mpl.get_backend() in mpl.rcsetup.interactive_bk, "Compound Prism Spectrometer Designer requires interactive backend"
    fig = plt.figure(constrained_layout=True)
    fig.canvas.manager.set_window_title("Compound Prism Spectrometer Designer")
    interactive = Interactive(fig, designs)
    plt.show()
