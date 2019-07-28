from collections import namedtuple
from os import cpu_count
import numpy as np
import pygmo as pg
import prism
from prism import RayTraceError, create_catalog, Prism, PmtArray, GaussianBeam

Config = namedtuple("Config", "prism_count, wmin, wmax, prism_height, prism_width, spec_length, beam_width, spec_min_ci, bounds")
Params = namedtuple("Params", "glass_names, glasses, thetas, curvature, y_mean, spec_angle")
Soln = namedtuple("Soln", "params, objectives")
nobjective = 3


def to_params(config: Config, catalog, p: np.ndarray):
    gvalues = list(catalog.keys())
    glass_names = [gvalues[min(int(i), len(catalog) - 1)] for i in p[:config.prism_count]]
    glasses = [catalog[n] for n in glass_names]
    return Params(glass_names, glasses, np.asarray(p[config.prism_count:config.prism_count * 2 + 1]), *p[-3:])


def transmission_data(wavelengths: [float], config: Config, params: Params, det):
    return np.array(
        prism.transmission(
            wavelengths,
            Prism(
                glasses=params.glasses,
                angles=params.thetas,
                curvature=params.curvature,
                height=config.prism_height,
                width=config.prism_width,
            ),
            PmtArray(
                bins=config.bounds,
                min_ci=config.spec_min_ci,
                angle=params.spec_angle,
                length=config.spec_length,
            ),
            GaussianBeam(
                width=config.beam_width,
                y_mean=params.y_mean,
                w_range=(config.wmin, config.wmax),
            ),
            det
        )
    )


def spectrometer_position(config: Config, params: Params):
    return np.array(
        prism.spectrometer_position(
            Prism(
                glasses=params.glasses,
                angles=params.thetas,
                curvature=params.curvature,
                height=config.prism_height,
                width=config.prism_width,
            ),
            PmtArray(
                bins=config.bounds,
                min_ci=config.spec_min_ci,
                angle=params.spec_angle,
                length=config.spec_length,
            ),
            GaussianBeam(
                width=config.beam_width,
                y_mean=params.y_mean,
                w_range=(config.wmin, config.wmax),
            ),
        )
    )


def trace(wavelength: float, initial_y: float, config: Config, params: Params, det):
    return prism.trace(
        wavelength,
        initial_y,
        Prism(
            glasses=params.glasses,
            angles=params.thetas,
            curvature=params.curvature,
            height=config.prism_height,
            width=config.prism_width,
        ),
        PmtArray(
            bins=config.bounds,
            min_ci=config.spec_min_ci,
            angle=params.spec_angle,
            length=config.spec_length
        ),
        det
    )


def fitness(config: Config, params: Params):
    try:
        return prism.fitness(
            Prism(
                glasses=params.glasses,
                angles=params.thetas,
                curvature=params.curvature,
                height=config.prism_height,
                width=config.prism_width,
            ),
            PmtArray(
                bins=config.bounds,
                min_ci=config.spec_min_ci,
                angle=params.spec_angle,
                length=config.spec_length
            ),
            GaussianBeam(
                width=config.beam_width,
                y_mean=params.y_mean,
                w_range=(config.wmin, config.wmax),
            ),
        )
    except RayTraceError:
        return [1e6 * config.prism_height] * nobjective


def show_interactive(config: Config, solns: [Soln]):
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
    objectives_plt.set_xlabel("size")
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
        angles: {', '.join(f'{np.rad2deg(angle):.4}' for angle in params.thetas)}
        y_mean: {params.y_mean:.4}
        curvature: {params.curvature:.4}
        spectrometer angle: {np.rad2deg(params.spec_angle):.4}
        objectives: (size={size:.4}, info: {-ninfo:.4}, deviation: {np.rad2deg(np.arcsin(dev)):.4})"""
        print(display)
        text_ax.cla()
        text_ax.text(0, 0.5, display, horizontalalignment='left', verticalalignment='center',
                     transform=text_ax.transAxes)
        text_ax.axis('scaled')
        text_ax.axis('off')

        spec_pos, spec_dir = det = spectrometer_position(config, params)
        spec_end = spec_pos + spec_dir * config.spec_length

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

        spectro = plt.Polygon((spec_pos, spec_end), closed=None, fill=None, edgecolor='k')
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
        det_plt.set_xlabel("normalized bin bounds")
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
        spec_angle_bounds = -sec, sec
        return tuple(zip(*[
            *glass_bounds,
            *angle_bounds,
            curvature_bounds,
            y_mean_bounds,
            spec_angle_bounds,
        ]))

    @staticmethod
    def get_nobj():
        return nobjective

    @staticmethod
    def get_name():
        return "Compound Prism Optimizer"


if __name__ == "__main__":
    with open("catalog.agf") as f:
        catalog = create_catalog(f.read())

    nbins = 32
    bounds = np.stack((np.arange(nbins) + 0.1, np.arange(nbins) + 0.9), axis=1) / 10
    config = Config(
        prism_count=3,
        wmin=0.5,
        wmax=0.82,
        prism_height=2.5,
        prism_width=0.7,
        spec_length=3.2,
        spec_min_ci=np.cos(np.deg2rad(60)),
        beam_width=0.2,
        bounds=bounds
    )

    prob = pg.problem(PrismProblem(config, catalog))
    algo = pg.algorithm(pg.nsga2(gen=1000))
    count = cpu_count()
    archi = pg.archipelago(count, algo=algo, prob=prob, pop_size=6 * count)
    archi.evolve()
    archi.wait_check()
    solutions = [Soln(params=to_params(config, catalog, p), objectives=o) for isl in archi for (p, o) in zip(isl.get_population().get_x(), isl.get_population().get_f())]

    print(len(solutions))

    show_interactive(config, solutions)
