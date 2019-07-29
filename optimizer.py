#!/usr/bin/env python3
from collections import namedtuple
from os import cpu_count
from sys import argv
import numpy as np
import pygmo as pg
import toml
import prism
from prism import RayTraceError, create_catalog, Prism, DetectorArray, GaussianBeam

Config = namedtuple("Config", "prism_count, wmin, wmax, prism_height, prism_width, det_arr_length, beam_width, det_arr_min_ci, bounds")
Params = namedtuple("Params", "glass_names, glasses, thetas, curvature, y_mean, det_arr_angle")
Soln = namedtuple("Soln", "params, objectives")
nobjective = 3


def to_params(config: Config, catalog, p: np.ndarray):
    gvalues = list(catalog.keys())
    glass_names = [gvalues[min(int(i), len(catalog) - 1)] for i in p[:config.prism_count]]
    glasses = [catalog[n] for n in glass_names]
    return Params(glass_names, glasses, np.asarray(p[config.prism_count:config.prism_count * 2 + 1]), *p[-3:])


def to_prism(config: Config, params: Params):
    return Prism(
        glasses=params.glasses,
        angles=params.thetas,
        curvature=params.curvature,
        height=config.prism_height,
        width=config.prism_width,
    )


def to_det_array(config: Config, params: Params):
    return DetectorArray(
        bins=config.bounds,
        min_ci=config.det_arr_min_ci,
        angle=params.det_arr_angle,
        length=config.det_arr_length,
    )


def to_beam(config: Config, params: Params):
    return GaussianBeam(
        width=config.beam_width,
        y_mean=params.y_mean,
        w_range=(config.wmin, config.wmax),
    )


def transmission_data(wavelengths: [float], config: Config, params: Params, det):
    return np.array(
        prism.transmission(
            wavelengths,
            to_prism(config, params),
            to_det_array(config, params),
            to_beam(config, params),
            det
        )
    )


def detector_array_position(config: Config, params: Params):
    return np.array(
        prism.detector_array_position(
            to_prism(config, params),
            to_det_array(config, params),
            to_beam(config, params),
        )
    )


def trace(wavelength: float, initial_y: float, config: Config, params: Params, det):
    return prism.trace(
        wavelength,
        initial_y,
        to_prism(config, params),
        to_det_array(config, params),
        det
    )


def fitness(config: Config, params: Params):
    try:
        return prism.fitness(
            to_prism(config, params),
            to_det_array(config, params),
            to_beam(config, params),
        )
    except RayTraceError:
        return [1e6 * config.prism_height] * nobjective


def show_interactive(config: Config, solns: [Soln], units: str):
    import matplotlib as mpl
    import matplotlib.path
    import matplotlib.patches
    import matplotlib.pyplot as plt
    fig = plt.figure()
    nrows, ncols = 2, 4
    objectives_plt = fig.add_subplot(1, ncols, 1)

    x, y, c = np.array([soln.objectives for soln in solns]).T
    y = -y
    c = np.rad2deg(np.arcsin(c))

    sc = objectives_plt.scatter(x, y, c=c, cmap="viridis", picker=True)
    clb = fig.colorbar(sc)
    clb.ax.set_ylabel("deviation (deg)")
    objectives_plt.set_xlabel(f"size ({units})")
    objectives_plt.set_ylabel("mutual information (bits)")
    # ax.set_zlabel("deviation (deg)")
    """
    x, y, z, c = res.T
    c = 1 - c

    sc = ax.scatter(x, y, z, c=c, cmap="viridis", picker=True)
    clb = fig.colorbar(sc)
    clb.ax.set_ylabel('transmittance')
    """
    prism_plt = fig.add_subplot(nrows, ncols, 2)
    text_ax = fig.add_subplot(nrows, ncols, 6)
    det_plt = fig.add_subplot(nrows, ncols, 3)
    trans_plt = fig.add_subplot(nrows, ncols, 7)
    violin_plt = fig.add_subplot(1, ncols, 4)
    prism_plt.axis('off')
    text_ax.axis('off')

    def pick(event):
        soln = sorted((solns[i] for i in event.ind), key=lambda s: s.objectives[-1])[0]
        params = soln.params
        size, ninfo, dev = soln.objectives
        display = f"""Prism ({', '.join(params.glass_names)})
        angles (deg): {', '.join(f'{np.rad2deg(angle):.4}' for angle in params.thetas)}
        y_mean ({units}): {params.y_mean:.4}
        curvature: {params.curvature:.4}
        spectrometer angle (deg): {np.rad2deg(params.det_arr_angle):.4}
        objectives: (size={size:.4} ({units}), info: {-ninfo:.4} (bits), deviation: {np.rad2deg(np.arcsin(dev)):.4} (deg))"""
        print(display)
        text_ax.cla()
        text_ax.text(0, 0.5, display, horizontalalignment='left', verticalalignment='center',
                     transform=text_ax.transAxes)
        text_ax.axis('scaled')
        text_ax.axis('off')

        det_arr_pos, det_arr_dir = det = detector_array_position(config, params)
        det_arr_end = det_arr_pos + det_arr_dir * config.det_arr_length

        prism_vertices = np.empty((config.prism_count + 2, 2))
        prism_vertices[::2, 1] = 0
        prism_vertices[1::2, 1] = config.prism_height
        prism_vertices[0, 0] = 0
        prism_vertices[1:, 0] = np.add.accumulate(np.tan(np.abs(params.thetas)) * config.prism_height)
        triangles = np.stack((prism_vertices[1:-1], prism_vertices[:-2], prism_vertices[2:]), axis=1)

        midpt = prism_vertices[-2] + (prism_vertices[-1] - prism_vertices[-2]) / 2
        c, s = np.cos(params.thetas[-1]), np.sin(params.thetas[-1])
        R = np.array(((c, -s), (s, c)))
        normal = R @ (-1, 0)
        chord = config.prism_height / c
        lens_radius = chord / (2 * params.curvature)
        center = midpt + normal * np.sqrt(lens_radius ** 2 - chord ** 2 / 4)
        t1 = np.rad2deg(np.arctan2(prism_vertices[-1, 1] - center[1], prism_vertices[-1, 0] - center[0]))
        t2 = np.rad2deg(np.arctan2(prism_vertices[-2, 1] - center[1], prism_vertices[-2, 0] - center[0]))
        if config.prism_count % 2 == 0:
            t1, t2 = t2, t1

        prism_plt.cla()
        for i, tri in enumerate(triangles):
            poly = plt.Polygon(tri, edgecolor='k', facecolor=('gray' if i % 2 else 'white'), closed=False)
            prism_plt.add_patch(poly)
        arc = mpl.path.Path.arc(t1, t2)
        arc = mpl.path.Path(arc.vertices * lens_radius + center, arc.codes)
        arc = mpl.patches.PathPatch(arc, fill=None)
        prism_plt.add_patch(arc)

        spectro = plt.Polygon((det_arr_pos, det_arr_end), closed=None, fill=None, edgecolor='k')
        prism_plt.add_patch(spectro)

        for w, color in zip((config.wmin, (config.wmin + config.wmax) / 2, config.wmax), ('r', 'g', 'b')):
            try:
                ray = np.stack(tuple(trace(w, params.y_mean, config, params, det)), axis=0)
                poly = plt.Polygon(ray, closed=None, fill=None, edgecolor=color)
                prism_plt.add_patch(poly)
            except RayTraceError:
                pass

        prism_plt.axis('scaled')
        prism_plt.axis('off')

        det_plt.cla()
        trans_plt.cla()
        violin_plt.cla()
        det_plt.set_xlabel("bins")
        trans_plt.set_xlabel("wavelength (μm)")
        det_plt.set_ylabel("p (%)")
        trans_plt.set_ylabel("p (%)")
        det_plt.set_title("p(D=d|Λ)")
        trans_plt.set_title("p(D|Λ=λ)")
        trans_plt.set_ylim(0, 100)

        waves = np.linspace(config.wmin, config.wmax, 100)
        ts = transmission_data(waves, config, params, det)
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
        violin_plt.violin(vpstats, showextrema=False, widths=1)
        det_plt.plot(p_det * 100, 'k')
        trans_plt.plot(waves, p_w_l_D * 100, 'k')

        fig.canvas.draw()
        fig.canvas.flush_events()

    fig.canvas.mpl_connect('pick_event', pick)

    plt.show()


class PrismProblem:
    def __init__(self, config, catalog):
        self.config, self.catalog = config, catalog

    def fitness(self, v):
        params = to_params(self.config, self.catalog, v)
        val = fitness(self.config, params)
        if val[0] > 30 * self.config.prism_height:
            return [1e6 * self.config.prism_height] * nobjective
        else:
            return val

    def get_bounds(self):
        nprism = self.config.prism_count
        glass_bounds = nprism * [(0, len(self.catalog) - 1)]
        l, u = np.deg2rad(0), np.deg2rad(90)
        angle_bounds = [(l, u) if i % 2 else (-u, -l) for i in range(nprism + 1)]
        curvature_bounds = 0.01, 1.0
        y_mean_bounds = 0, self.config.prism_height
        sec = np.deg2rad(90)
        det_arr_angle_bounds = -sec, sec
        return tuple(zip(*[
            *glass_bounds,
            *angle_bounds,
            curvature_bounds,
            y_mean_bounds,
            det_arr_angle_bounds,
        ]))

    @staticmethod
    def get_nobj():
        return nobjective

    @staticmethod
    def get_name():
        return "Compound Prism Optimizer"


if __name__ == "__main__":
    toml_file = argv[1] if len(argv) > 1 else "design_config.toml"
    toml_spec = toml.load(toml_file)
    catalog_path = toml_spec["catalog-path"]
    with open(catalog_path) as f:
        catalog = create_catalog(f.read())
    units = toml_spec.get("length-unit", "a.u.")

    config = Config(
        prism_count=toml_spec["compound-prism"]["count"],
        prism_height=toml_spec["compound-prism"]["height"],
        prism_width=toml_spec["compound-prism"]["width"],
        beam_width=toml_spec["gaussian-beam"]["width"],
        wmin=toml_spec["gaussian-beam"]["wmin"],
        wmax=toml_spec["gaussian-beam"]["wmax"],
        det_arr_length=toml_spec["detector-array"]["length"],
        det_arr_min_ci=np.cos(np.deg2rad(toml_spec["detector-array"].get("max-incident-angle", 90))),
        bounds=np.array(toml_spec["detector-array"]["bounds"])
    )

    opt_dict = {"iteration-count": 1000, "thread-count": 0, "pop-size": 20, **toml_spec.get("optimizer", {})}
    iter_count = max(opt_dict["iteration-count"], 1)
    thread_count = opt_dict["thread-count"]
    if thread_count < 1:
        thread_count = cpu_count()
    pop_size = opt_dict["pop-size"]
    if pop_size < 5 or pop_size % 4 != 0:
        pop_size = max(8, pop_size + 4 - pop_size % 4)
    prob = pg.problem(PrismProblem(config, catalog))
    algo = pg.algorithm(pg.nsga2(gen=iter_count))
    archi = pg.archipelago(thread_count, algo=algo, prob=prob, pop_size=pop_size)
    archi.evolve()
    archi.wait_check()
    solutions = [Soln(params=to_params(config, catalog, p), objectives=o) for isl in archi for (p, o) in zip(isl.get_population().get_x(), isl.get_population().get_f())]

    print(len(solutions))

    show_interactive(config, solutions, units)
