from os import cpu_count
import sys
from PySide2.QtCore import *
from PySide2.QtWidgets import *
from PySide2.QtGui import QIcon

import numpy as np
from multiprocessing import Process, Value, Pipe
import matplotlib as mpl
import matplotlib.patches
from matplotlib.backends.backend_qt5agg import (FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

import toml
import pathlib
import pygmo as pg
from compound_prism_designer import Config, Soln, RayTraceError, create_glass_catalog, \
    detector_array_position, trace, transmission_data, PyGmoPrismProblem
from compound_prism_designer.utils import draw_compound_prism
from compound_prism_designer.zemax import create_zmx


class DesignerConfigFormQt(QWidget):
    def __init__(self):
        super().__init__()
        layout = QFormLayout(self)

        self._choosen_file = None
        choose_cat_act = QAction(parent=self, icon=QIcon.fromTheme("document-open"), text="Choose glass catalog file")
        choose_cat_act.triggered.connect(self._choose_catalog_file)
        self._choose_cat = choose_cat = QLineEdit(self)
        choose_cat.addAction(choose_cat_act, QLineEdit.TrailingPosition)
        layout.addRow("Catalog File:", choose_cat)

    def _choose_catalog_file(self):
        cfile, _ = QFileDialog.getOpenFileName(
            caption="Select catalog file", filter="Glass Catalog files (*.agf);;All files (*)")
        self._choosen_file = cfile
        self._choose_cat.setText(cfile)


class DesignerAppQtWindow(QMainWindow):
    def __init__(self, config: Config, solns: [Soln], units: str):
        super().__init__()
        self._central_widget = QWidget()
        self.setCentralWidget(self._central_widget)
        layout = QGridLayout(self._central_widget)

        self._pareto_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self._pareto_canvas, 0, 0, 2, 1)
        self._pareto_ax = self._pareto_canvas.figure.subplots()
        self._pareto_toolbar = NavigationToolbar(self._pareto_canvas, self, False)
        self.addToolBar(self._pareto_toolbar)

        self._prism_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self._prism_canvas, 0, 1, 1, 2)
        self._prism_ax = self._prism_canvas.figure.add_axes([0, 0, 1, 1])
        self._prism_toolbar = NavigationToolbar(self._prism_canvas, self, False)
        self.addToolBar(self._prism_toolbar)
        self._prism_ax.axis('off')

        self._pdet_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self._pdet_canvas, 1, 1)
        self._pdet_ax = self._pdet_canvas.figure.subplots()

        self._ptrans_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self._ptrans_canvas, 1, 2)
        self._ptrans_ax = self._ptrans_canvas.figure.subplots()

        self._violin_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self._violin_canvas, 0, 3, 2, 1)
        self._violin_ax = self._violin_canvas.figure.subplots()

        self._config = config
        self._solns = solns
        self._units = units
        self._selected_solution = None

        x, y, c = np.array([(soln.fitness.size, soln.fitness.info, soln.fitness.deviation) for soln in solns]).T
        c = np.rad2deg(np.arcsin(c))
        sc = self._pareto_ax.scatter(x, y, c=c, cmap="viridis", picker=True)
        clb = self._pareto_canvas.figure.colorbar(sc)
        clb.ax.set_ylabel("deviation (deg)")
        self._pareto_ax.set_xlabel(f"size ({units})")
        self._pareto_ax.set_ylabel("mutual information (bits)")
        self._pareto_canvas.figure.canvas.mpl_connect('pick_event', self._pick_solution)

        mbar = self.menuBar()
        export_action = mbar.addAction("Export Design as .zmx")
        export_action.triggered.connect(self._export_solutin)

    def _export_solutin(self):
        if self._selected_solution is not None:
            zpath, _ = QFileDialog.getSaveFileName(
                caption="Select design specification", filter="Zemax Lens File (*.zmx)")
            zmx_path = pathlib.Path(zpath).with_suffix('.zmx')
            prism = self._selected_solution.compound_prism
            detarr = self._selected_solution.detector_array
            beam = self._selected_solution.beam
            zemax_design, zemax_file = create_zmx(prism, detarr, beam, self._config.detector_array_length)
            with open(zmx_path, "w") as f:
                f.write(zemax_file)

    def _pick_solution(self, event):
        soln = self._selected_solution = self._solns[event.ind[0]]
        prism = soln.compound_prism
        detarr = soln.detector_array
        beam = soln.beam
        det_arr_pos, det_arr_dir = det = detector_array_position(prism, detarr, beam)
        det_arr_end = det_arr_pos + det_arr_dir * self._config.detector_array_length
        # Draw Spectrometer
        draw_compound_prism(self._prism_ax, soln.compound_prism)
        spectro = mpl.patches.Polygon((det_arr_pos, det_arr_end), closed=None, fill=None, edgecolor='k')
        self._prism_ax.add_patch(spectro)
        wmin, wmax = self._config.wavelength_range
        for w, color in zip((wmin, (wmin + wmax) / 2, wmax), ('r', 'g', 'b')):
            for y in (beam.y_mean - self._config.beam_width, beam.y_mean, beam.y_mean + self._config.beam_width):
                try:
                    ray = np.stack(tuple(trace(w, y, prism, detarr, det)), axis=0)
                    poly = mpl.patches.Polygon(ray, closed=None, fill=None, edgecolor=color)
                    self._prism_ax.add_patch(poly)
                except RayTraceError:
                    pass
        self._prism_ax.axis('scaled')
        self._prism_ax.axis('off')
        # Plot solution results

        self._pdet_ax.clear()
        self._ptrans_ax.cla()
        self._violin_ax.cla()

        waves = np.linspace(self._config.wavelength_range[0], self._config.wavelength_range[1], 100)
        ts = transmission_data(waves, prism, detarr, beam, det)
        p_det = ts.sum(axis=1) * (1 / len(waves))
        p_w_l_D = ts.sum(axis=0)
        vpstats = [
            {
                "coords": waves,
                "vals": t,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }
            for t in ts
        ]
        parts = self._violin_ax.violin(vpstats, showextrema=False, widths=1)
        for pc in parts['bodies']:
            pc.set_facecolor('black')
        self._violin_ax.plot([1, len(p_det)], self._config.wavelength_range, 'k--')
        self._pdet_ax.scatter(1 + np.arange(len(p_det)), p_det * 100, color='k')
        self._ptrans_ax.plot(waves, p_w_l_D * 100, 'k')
        self._ptrans_ax.axhline(np.mean(p_w_l_D) * 100, color='k', linestyle='--')

        self._pdet_ax.set_xlabel("detector bins")
        self._ptrans_ax.set_xlabel("wavelength (μm)")
        self._violin_ax.set_xlabel("detector bins")
        self._pdet_ax.set_ylabel("p (%)")
        self._ptrans_ax.set_ylabel("p (%)")
        self._violin_ax.set_ylabel("wavelength (μm)")
        self._pdet_ax.set_title("p(D=d|Λ)")
        self._ptrans_ax.set_title("p(D|Λ=λ)")
        self._violin_ax.set_title("p(D=d|Λ=λ) as a Pseudo Violin Plot")
        self._ptrans_ax.set_ylim(0, 100)
        self._pdet_ax.set_ylim(bottom=0)

        self._prism_canvas.draw()
        self._pdet_canvas.draw()
        self._ptrans_canvas.draw()
        self._violin_canvas.draw()


def toml_to_config(toml_path: str):
    toml_spec = toml.load(toml_path)
    catalog_path = toml_spec["catalog-path"]
    with open(catalog_path) as f:
        catalog = create_glass_catalog(f.read())
    units = toml_spec.get("length-unit", "a.u.")

    config = Config(
        max_prism_count=toml_spec["compound-prism"]["max-count"],
        max_prism_height=toml_spec["compound-prism"]["max-height"],
        prism_width=toml_spec["compound-prism"]["width"],
        beam_width=toml_spec["gaussian-beam"]["width"],
        wavelength_range=(toml_spec["gaussian-beam"]["wmin"], toml_spec["gaussian-beam"]["wmax"]),
        detector_array_length=toml_spec["detector-array"]["length"],
        detector_array_min_ci=np.cos(np.deg2rad(toml_spec["detector-array"].get("max-incident-angle", 90))),
        detector_array_bin_bounds=np.array(toml_spec["detector-array"]["bounds"]),
        glass_catalog=catalog
    )
    opt_dict = {"iteration-count": 1000, "thread-count": 0, "pop-size": 20, **toml_spec.get("optimizer", {})}
    opt_dict['iteration-count'] = max(opt_dict["iteration-count"], 1)
    if opt_dict["thread-count"] < 1:
        opt_dict["thread-count"] = cpu_count()
    return config, units, opt_dict


def optimize(config, opt_dict, steps, send_soln):
    iter_count, thread_count, pop_size = opt_dict['iteration-count'], opt_dict['thread-count'], opt_dict['pop-size']
    if pop_size < 5 or pop_size % 4 != 0:
        pop_size = max(8, pop_size + 4 - pop_size % 4)
    prob = pg.problem(PyGmoPrismProblem(config))
    algo = pg.algorithm(pg.nsga2(gen=1, cr=0.98, m=0.1))
    archi = pg.archipelago(thread_count, algo=algo, prob=prob, pop_size=pop_size)
    for i in range(iter_count):
        archi.evolve()
        archi.wait_check()
        steps.value = i + 1
    solns = (Soln.from_config_and_array(config, x) for isl in archi for x in isl.get_population().get_x())
    solns = list(
        filter(lambda v: v is not None and v.fitness.size <= 30 * config.max_prism_height and v.fitness.info >= 0.1,
               solns))
    sorted_solns_idxs = pg.select_best_N_mo([
        (s.fitness.size, -s.fitness.info, s.fitness.deviation) for s in solns
    ], pop_size)
    solutions = [solns[i] for i in sorted_solns_idxs]
    send_soln.send(solutions)


def main():
    qapp = QApplication(sys.argv)
    qapp.setApplicationName("Compound Prism Spectrometer Designer")

    app = None
    steps = Value('i', 0)
    recv_soln, send_soln = Pipe(duplex=False)

    fname, _ = QFileDialog.getOpenFileName(
        caption="Select design specification", filter="Configuration files (*.toml);;All files (*)")
    config, units, opt_dict = toml_to_config(fname)

    progress = QProgressDialog("Design Optimization", "Cancel", 0, opt_dict['iteration-count'])
    progress.setLabelText("Optimization Progress")
    progress.setMinimumDuration(0)
    progress_timer = QTimer(progress)

    def update_progress():
        s = steps.value
        progress.setValue(s)
        if s >= progress.maximum() and recv_soln.poll():
            progress_timer.stop()
            solutions = recv_soln.recv()
            nonlocal app
            app = DesignerAppQtWindow(config, solutions, units)
            app.show()

    def cancel_timer():
        progress_timer.stop()
        progress_proc.kill()
        qapp.quit()

    progress.canceled.connect(cancel_timer)
    progress_timer.timeout.connect(update_progress)
    progress_timer.start(16)

    progress_proc = Process(target=optimize, args=(config, opt_dict, steps, send_soln))
    progress_proc.start()

    sys.exit(qapp.exec_())


if __name__ == "__main__":
    main()
