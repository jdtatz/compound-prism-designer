from collections import namedtuple
import numpy as np
import prism
from prism import PrismError, create_catalog, Prism, PmtArray, GaussianBeam

Config = namedtuple("Config", "prism_count, wmin, wmax, waves, prism_height, prism_width, spec_length, beam_width, spec_min_ci, bounds")
Params = namedtuple("Params", "glasses, thetas, curvature, y_mean, spec_angle")
Soln = namedtuple("Soln", "params, objectives")


def to_params(config: Config, catalog, p: np.ndarray):
    gvalues = list(catalog.values())
    glasses = tuple(gvalues[min(int(i), len(catalog) - 1)] for i in p[:config.prism_count])
    return Params(glasses, np.asarray(p[config.prism_count:config.prism_count * 2 + 1]), *p[-3:])


def transmission_data(config: Config, params: Params):
    return np.array(
        prism.transmission(
            config.waves,
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
            ),
        )
    )


def spectrometer_position(config: Config, params: Params):
    return np.array(
        prism.spectrometer_position(
            config.wmin,
            config.wmax,
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
            ),
        )
    )


def trace(wavelength: float, initial_y: float, config: Config, params: Params):
    return prism.trace(
        wavelength,
        config.wmin,
        config.wmax,
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
        GaussianBeam(
            width=config.beam_width,
            y_mean=params.y_mean,
        ),
    )


def fitness(config: Config, params: Params):
    try:
        return prism.fitness(
            config.waves,
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
            ),
        )
    except PrismError:
        return [1e6 * config.prism_height] * 2


def show_interactive(config: Config, solns: [Soln]):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(131)

    x, y = np.array([soln.objectives for soln in solns]).T
    y = -y

    sc = ax.scatter(x, y, picker=True)
    ax.set_xlabel("size")
    ax.set_ylabel("transinformation")
    """
    x, y, z, c = res.T
    c = 1 - c

    sc = ax.scatter(x, y, z, c=c, cmap="viridis", picker=True)
    ax.set_xlabel("size")
    ax.set_ylabel("nonlinearity")
    ax.set_zlabel("spot size")
    clb = fig.colorbar(sc)
    clb.ax.set_ylabel('transmittance')
    """

    px = fig.add_subplot(132)
    px.axis('off')
    gx = fig.add_subplot(233)
    tx = fig.add_subplot(236)

    def pick(event):
        soln = sorted((solns[i] for i in event.ind), key=lambda s: s.objectives[-1])[0]
        params = soln.params
        size, ninfo = soln.objectives
        print(params)
        print(f"size: {size:.2}, info: {-ninfo:.2}")

        spec_pos, spec_dir = spectrometer_position(config, params)
        spec_end = spec_pos + spec_dir

        prism_vertices = np.empty((config.prism_count + 2, 2))
        prism_vertices[::2, 1] = 0
        prism_vertices[1::2, 1] = 1
        prism_vertices[0, 0] = 0
        prism_vertices[1:, 0] = np.add.accumulate(np.tan(np.abs(params.thetas)))
        triangles = np.stack((prism_vertices[1:-1], prism_vertices[:-2], prism_vertices[2:]), axis=1)

        curvature = params.curvature
        ld = prism_vertices[-1] - prism_vertices[-2]
        norm = np.array((ld[1], -ld[0]))
        norm /= np.linalg.norm(norm)
        midpt = prism_vertices[-2] + (ld) / 2
        diameter = np.linalg.norm(ld)
        lradius = (diameter / 2) / curvature
        rs = diameter * np.sqrt(1 - curvature * curvature) / (2 * curvature)
        c = midpt[0] + norm[0] * rs, midpt[1] + norm[1] * rs

        px.cla()
        for i, tri in enumerate(triangles):
            poly = plt.Polygon(tri, edgecolor='k', facecolor=('gray' if i % 2 else 'white'), closed=False)
            px.add_patch(poly)
        t1 = np.rad2deg(np.arctan2(prism_vertices[-1, 1] - c[1], prism_vertices[-1, 0] - c[0]))
        t2 = np.rad2deg(np.arctan2(prism_vertices[-2, 1] - c[1], prism_vertices[-2, 0] - c[0]))
        arc = mpl.path.Path.arc(t1, t2)
        arc = mpl.path.Path(arc.vertices * lradius + c, arc.codes)
        arc = mpl.patches.PathPatch(arc, fill=None)
        px.add_patch(arc)

        spectro = plt.Polygon((spec_pos, spec_end), closed=None, fill=None, edgecolor='k')
        px.add_patch(spectro)

        for w, color in zip((config.wmin, (config.wmin + config.wmax) / 2, config.wmax), ('r', 'g', 'b')):
            y = params.y_mean
            # for y in (params.y_mean, params.y_mean + config.beam_width / config.prism_height, params.y_mean - config.beam_width / config.prism_height):
            try:
                ray = np.stack(tuple(trace(w, y, config, params)), axis=0)
                poly = plt.Polygon(ray, closed=None, fill=None, edgecolor=color)
                px.add_patch(poly)
            except PrismError:
                pass

        px.axis('scaled')
        px.axis('off')

        gx.cla()
        tx.cla()
        gx.set_xlabel("normalized bin bounds")
        tx.set_xlabel("wavelength (μm)")
        gx.set_ylabel("p (%)")
        tx.set_ylabel("p (%)")
        gx.set_title("p(D=d|Λ)")
        tx.set_title("p(D|Λ=λ)")
        tx.set_ylim(0, 100)

        ts = transmission_data(config, params)
        p_det = ts.sum(axis=1) * (1 / len(config.waves))
        p_w_l_D = ts.sum(axis=0)
        gx.plot([str(b) for b in config.bounds], p_det * 100, 'k')
        tx.plot(config.waves, p_w_l_D * 100, 'k')

        fig.canvas.draw()
        fig.canvas.flush_events()

    fig.canvas.mpl_connect('pick_event', pick)

    plt.show()


class PrismProblem:
    def __init__(self, config, catalog):
        self.config, self.catalog = config, catalog

    def fitness(self, v):
        try:
            params = to_params(self.config, self.catalog, v)
            return fitness(self.config, params)
        except PrismError:
            return [1e6 * self.config.prism_height] * 2

    def get_bounds(self):
        nprism = self.config.prism_count
        glass_bounds = nprism * [(0, len(self.catalog) - 1)]
        l, u = np.deg2rad(0), np.deg2rad(90)
        angle_bounds = [(l, u) if i % 2 else (-u, -l) for i in range(nprism + 1)]
        curvature_bounds = 0.1, 1.0
        y_mean_bounds = 0, 1
        sec = np.deg2rad(45)
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
        return 2

    @staticmethod
    def get_name():
        return "Compound Prism Optimizer"


if __name__ == "__main__":
    nprism = 2
    nwaves = 100
    nbin = 32

    with open("catalog.agf") as f:
        contents = f.read()
    catalog = create_catalog(contents)

    gnames, gvalues = zip(*catalog.items())

    bounds = np.linspace(0, 1, nbin + 1)
    bounds = [(l, u) for l, u in zip(bounds[:-1], bounds[1:])]
    config = Config(
        prism_count=nprism,
        wmin=0.5,
        wmax=0.82,
        waves=np.linspace(0.5, 0.82, nwaves),
        prism_height=2.5,
        prism_width=2.5,
        spec_length=3.2,
        spec_min_ci=np.cos(np.deg2rad(60)),
        beam_width=0.2,
        bounds=bounds
    )

    """
    from platypus import Problem, Real
    import platypus

    def fitness(p):
        try:
            size, info = fitness(config, to_params(p))
            return (size, info), [0 if size < 40 else 1]
        except prism.PrismError:
            return [1e6] * 2, [1]

    n_params = nprism * 2 + 4
    n_objectives = 2
    n_constraints = 1
    problem = Problem(n_params, n_objectives, n_constraints)
    problem.types[:nprism] = Real(0, len(catalog))
    l, u = np.deg2rad(0), np.deg2rad(90)
    problem.types[nprism:2*nprism+1] = [Real(l, u) if i % 2 else Real(-u, -l) for i in range(nprism + 1)]
    sac = np.deg2rad(60)
    problem.types[2*nprism+1:] = Real(0.2, 1.0), Real(0, 1), Real(-sac, sac)
    problem.constraints[:] = "==0"
    problem.function = fitness

    algorithm = platypus.NSGAII(problem, 100000)
    algorithm.run(1000)

    solutions = [s.objectives for s in algorithm.result if s.feasible]

    """
    import pygmo as pg

    prob = pg.problem(PrismProblem(config, catalog))
    algo = pg.algorithm(pg.moead(gen=10000))
    archi = pg.archipelago(16, algo=algo, prob=prob, pop_size=40)
    archi.evolve()
    archi.wait_check()
    solutions = [Soln(params=to_params(config, catalog, p), objectives=o) for isl in archi for (p, o) in zip(isl.get_population().get_x(), isl.get_population().get_f())]

    print(len(solutions))

    show_interactive(config, solutions)

