#!/usr/bin/env python3
import typing
import pathlib
from dataclasses import dataclass
import toml
import numpy as np
import matplotlib as mpl
mpl.rcParams["backend"] = "qt5agg"
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
QFileDialog = mpl.backends.qt_compat.QtWidgets.QFileDialog

from .compound_prism_designer import optimize, create_glass_catalog, Glass, OptimizerSpecification, \
    CompoundPrismSpecification, GaussianBeamSpecification, DetectorArraySpecification, Design, RayTraceError
from .utils import draw_spectrometer
from .zemax import create_zmx


@dataclass
class Specification:
    units: str
    catalog: typing.Optional[Glass]
    optimizer: OptimizerSpecification
    compound_prism: CompoundPrismSpecification
    gaussian_beam: GaussianBeamSpecification
    detector_array: DetectorArraySpecification

    @staticmethod
    def from_toml(toml_path: str):
        toml_spec = toml.load(toml_path)
        catalog_path = toml_spec.get("catalog-path")
        if catalog_path is not None:
            with open(catalog_path) as f:
                catalog = create_glass_catalog(f.read())
        else:
            catalog = None
        units = toml_spec.get("length-unit", "a.u.")
        opt = toml_spec["optimizer"]
        cmpnd = toml_spec["compound-prism"]
        beam = toml_spec["gaussian-beam"]
        detarr = toml_spec["detector-array"]

        spec = Specification(
            units=units,
            catalog=catalog,
            optimizer=OptimizerSpecification(
                iteration_count=opt["iteration-count"],
                population_size=opt["population-size"],
                offspring_size=opt["offspring-size"],
                crossover_distribution_index=opt["crossover-distribution-index"],
                mutation_distribution_index=opt["mutation-distribution-index"],
                mutation_probability=opt["mutation-probability"],
                seed=opt.get("seed", np.random.rand(1)),
                epsilons=tuple(opt["epsilons"])
            ),
            compound_prism=CompoundPrismSpecification(
                max_count=cmpnd["max-count"],
                max_height=cmpnd["max-height"],
                width=cmpnd["width"]
            ),
            gaussian_beam=GaussianBeamSpecification(
                width=beam["width"],
                wavelength_range=tuple(beam["wavelength-range"]),
            ),
            detector_array=DetectorArraySpecification(
                length=detarr["length"],
                max_incident_angle=detarr["max-incident-angle"],
                bounds=detarr["bounds"]
            ),
        )
        return spec


def show_interactive(fig: Figure, spec: Specification, designs: [Design]):
    units = spec.units
    nrows, ncols = 2, 4
    gs = fig.add_gridspec(nrows, ncols)
    objectives_plt = fig.add_subplot(gs[:, 0])

    x, y, c = np.array([(design.fitness.size, design.fitness.info, design.fitness.deviation)
                        for design in designs]).T
    c = np.rad2deg(np.arcsin(c))

    cm = mpl.cm.ScalarMappable(mpl.colors.Normalize(0, 90, clip=True), 'PuRd')
    sc = objectives_plt.scatter(x, y, c=c, cmap=cm.cmap.reversed(), norm=cm.norm, picker=True)
    clb = fig.colorbar(sc)
    clb.ax.set_ylabel("deviation (deg)")
    objectives_plt.set_xlabel(f"size ({units})")
    objectives_plt.set_ylabel("mutual information (bits)")

    prism_plt = fig.add_subplot(gs[0, 1:3])
    det_plt = fig.add_subplot(gs[1, 1])
    trans_plt = fig.add_subplot(gs[1, 2])
    violin_plt = fig.add_subplot(gs[:, 3])
    prism_plt.axis('off')

    waves = np.linspace(*spec.gaussian_beam.wavelength_range, 100)
    selected_design = None

    def pick(event):
        nonlocal selected_design
        design = selected_design = sorted((designs[i] for i in event.ind), key=lambda s: -s.fitness.info)[0]

        transmission = design.transmission_probability(waves)
        p_det = transmission.sum(axis=0) * (1 / len(waves))
        p_w_l_D = transmission.sum(axis=1)
        zemax_design, zemax_file = create_zmx(design.compound_prism, design.detector_array, design.gaussian_beam)
        print(zemax_design)

        # Draw Spectrometer
        prism_plt.cla()
        draw_spectrometer(prism_plt, design.compound_prism, design.detector_array)

        wmin, wmax = design.gaussian_beam.wavelength_range
        for w, color in zip((wmin, (wmin + wmax) / 2, wmax), ('r', 'g', 'b')):
            for y in (design.gaussian_beam.y_mean - design.gaussian_beam.width, design.gaussian_beam.y_mean, design.gaussian_beam.y_mean + design.gaussian_beam.width):
                try:
                    ray = design.ray_trace(w, y)
                    poly = mpl.patches.Polygon(ray, closed=None, fill=None, edgecolor=color)
                    prism_plt.add_patch(poly)
                except RayTraceError:
                    pass
        prism_plt.axis('scaled')
        prism_plt.axis('off')

        # Plot Design Results
        det_plt.clear()
        trans_plt.cla()
        violin_plt.cla()

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
        parts = violin_plt.violin(vpstats, showextrema=False, widths=1)
        for pc in parts['bodies']:
            pc.set_facecolor('black')
        violin_plt.plot([1, len(p_det)], spec.gaussian_beam.wavelength_range, 'k--')
        det_plt.scatter(1 + np.arange(len(p_det)), p_det * 100, color='k')
        trans_plt.plot(waves, p_w_l_D * 100, 'k')
        trans_plt.axhline(np.mean(p_w_l_D) * 100, color='k', linestyle='--')

        det_plt.set_xlabel("detector bins")
        trans_plt.set_xlabel("wavelength (μm)")
        violin_plt.set_xlabel("detector bins")
        det_plt.set_ylabel("p (%)")
        trans_plt.set_ylabel("p (%)")
        violin_plt.set_ylabel("wavelength (μm)")
        det_plt.set_title("p(D=d|Λ)")
        trans_plt.set_title("p(D|Λ=λ)")
        violin_plt.set_title("p(D=d|Λ=λ) as a Pseudo Violin Plot")
        trans_plt.set_ylim(0, 100)
        det_plt.set_ylim(bottom=0)

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def export(filename):
        path = pathlib.Path(filename)
        if selected_design is not None and not path.is_dir():
            zemax_design, zemax_file = create_zmx(
                selected_design.compound_prism,
                selected_design.detector_array,
                selected_design.gaussian_beam
            )
            print(zemax_design)
            with open(path.with_suffix('.zmx'), "w") as f:
                f.write(zemax_file)

    fig.canvas.mpl_connect('pick_event', pick)
    return export


def main():
    fig = plt.figure()
    manager = plt.get_current_fig_manager()
    manager.set_window_title("Compound Prism Spectrometer Designer")
    spec_file = pathlib.Path(QFileDialog.getOpenFileName(
        caption="Select design specification", filter="Configuration files (*.toml);;All files (*)")[0])
    assert spec_file.is_file()
    spec = Specification.from_toml(spec_file)
    designs = optimize(spec.catalog, spec.optimizer, spec.compound_prism, spec.gaussian_beam, spec.detector_array)
    export = show_interactive(fig, spec, designs)
    mbar = manager.window.menuBar()
    export_action = mbar.addAction("Export Design as .zmx")
    export_action.triggered.connect(lambda: export(pathlib.Path(QFileDialog.getSaveFileName(caption="Export zemax file", filter="Zemax Lens File (*.zmx)")[0])))
    plt.show()
    return


if __name__ == "__main__":
    main()
