import numpy as np
from scipy.integrate import quad
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import namedtuple, OrderedDict
from platypus import NSGAII, Problem, Real

Ray = namedtuple("Ray", "p, v, t")
Config = namedtuple("Config", "prism_count, wmin, wmax, spec_length, beam_radius, spec_min_ci")
Params = namedtuple("Params", "glasses, thetas, curvature, initial_y, spec_angle")

glass_parametrization = {
    1: lambda cd: lambda w: np.sqrt(
        cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + cd[4] * w ** -6 + cd[5] * w ** -8),
    # Sellmeier1
    2: lambda cd: lambda w: np.sqrt(1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                                    cd[4] * w ** 2 / (w ** 2 - cd[5])),
    # Herzberger
    3: lambda cd: lambda w: cd[0] + cd[1] / (w ** 2 - 0.028) + cd[2] / (w ** 2 - 0.028) ** 2 + cd[3] * w ** 2 + cd[
        4] * w ** 4 + cd[5] * w ** 6,
    # Sellmeier2
    4: lambda cd: lambda w: np.sqrt(
        1.0 + cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2] ** 2) + cd[3] * w ** 2 / (w ** 2 - cd[4] ** 2)),
    # Conrady
    5: lambda cd: lambda w: cd[0] + cd[1] / w + cd[2] / w ** 3.5,
    # Sellmeier3
    6: lambda cd: lambda w: np.sqrt(1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                                    cd[4] * w ** 2 / (w ** 2 - cd[5]) + cd[6] * w ** 2 / (w ** 2 - cd[7])),
    # HandbookOfOptics1
    7: lambda cd: lambda w: np.sqrt(cd[0] + cd[1] / (w ** 2 - cd[2]) - cd[3] * w ** 2),
    # HandbookOfOptics2
    8: lambda cd: lambda w: np.sqrt(cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2]) - cd[3] * w ** 2),
    # Sellmeier4
    9: lambda cd: lambda w: np.sqrt(cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2]) + cd[3] * w ** 2 / (w ** 2 - cd[4])),
    # Extended1
    10: lambda cd: lambda w: np.sqrt(cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + cd[4] * w ** -6 + \
                                     cd[5] * w ** -8 + cd[6] * w ** -10 + cd[7] * w ** -12),
    # Sellmeier5
    11: lambda cd: lambda w: np.sqrt(1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                                     cd[4] * w ** 2 / (w ** 2 - cd[5]) + cd[6] * w ** 2 / (w ** 2 - cd[7]) + \
                                     cd[8] * w ** 2 / (w ** 2 - cd[9])),
    # Extended2
    12: lambda cd: lambda w: np.sqrt(cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + \
                                     cd[4] * w ** -6 + cd[5] * w ** -8 + cd[6] * w ** 4 + cd[7] * w ** 6),
}


def read_glasscat(catalog_filename):
    glasscat = OrderedDict()
    with open(catalog_filename, 'r') as f:
        for line in f:
            if line.startswith('NM'):
                nm = line.split()
                glassname = (nm[1]).upper()
                dispform = int(nm[2])
            elif line.startswith('CD'):
                glasscat[glassname] = glass_parametrization[dispform](tuple(map(float, line[2:].split())))
    return glasscat


def sqnorm(v):
    return v @ v


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def refract(n1, n2, normal, incident):
    r = n1 / n2
    ci = -(normal @ incident)
    assert ci > 0
    cr = np.sqrt(1 - r ** 2 * (1 - ci ** 2))
    assert np.isfinite(cr)
    rs = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2
    rp = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2
    return r * incident + (r * ci - cr) * normal, 1 - (rs + rp) / 2


def plane_intersection(p0, normal, origin, direction):
    d = (p0 - origin) @ normal / (direction @ normal)
    assert np.isfinite(d) and d > 0
    return origin + d * direction


def surface(n1, n2, theta, vertex, origin, direction):
    normal = rotation(theta) @ (-1, 0)
    v, t = refract(n1, n2, normal, direction)
    p = plane_intersection(vertex, normal, origin, direction)
    assert 0 <= p[1] <= 1
    return p, v, t


def lens_intersection(vertex, theta, curvature, origin, direction):
    normal = rotation(theta) @ (-1, 0)
    midpt = np.array((vertex[0] + np.tan(np.abs(theta)) / 2, 0.5))
    radius = 1 / (2 * np.cos(np.abs(theta)) * curvature)
    center = midpt + radius * normal
    d = -(direction @ (origin - center)) + np.sqrt(
        (direction @ (origin - center)) ** 2 - sqnorm(origin - center) + radius ** 2)
    p = origin + d * direction
    assert sqnorm(p - midpt) <= radius ** 2
    return p, center - p


def propagate_internal(wavelength, initial_y, glasses, thetas, curvature):
    n1 = 1
    vertex = np.array((0, 0))
    position = np.array((0, initial_y))
    direction = np.array((1, 0))
    transmittance = 1
    for glass, theta in zip(glasses, thetas):
        n2 = glass(wavelength)
        vertex = np.array((vertex[0] + np.tan(np.abs(theta)), 1 if vertex[1] == 0 else 0))
        position, direction, t = surface(n1, n2, theta, vertex, position, direction)
        transmittance *= t
        n1 = n2
    n2 = 1
    theta = thetas[-1]
    position, normal = lens_intersection(vertex, theta, curvature, position, direction)
    direction, t = refract(n1, n2, normal, direction)
    return position, direction, transmittance * t


def spectrometer_position(config: Config, params: Params):
    lp, lv, _ = propagate_internal(config.wmin, params.initial_y, params.glasses, params.thetas, params.curvature)
    up, uv, _ = propagate_internal(config.wmax, params.initial_y, params.glasses, params.thetas, params.curvature)
    spec = rotation(params.spec_angle) @ (0, 1) * config.spec_length
    dirM = np.transpose((uv, -lv))
    dist = np.linalg.inv(dirM) @ (spec - up + lp)
    if dist[1] > 0:
        return lp + dist[1] * lv
    else:
        return up - dist[0] * uv


def spectrometer_intersection(wavelength: float, initial_y: float, spec_pos: np.ndarray, config: Config,
                              params: Params):
    p, v, t = propagate_internal(wavelength, initial_y, params.glasses, params.thetas, params.curvature)
    spec_dir = rotation(params.spec_angle) @ (0, 1)
    normal = rotation(params.spec_angle) @ (-1, 0)
    assert -(normal @ v) >= config.spec_min_ci
    p = plane_intersection(spec_pos, normal, p, v)
    pos = ((p - spec_pos) @ spec_dir) / config.spec_length
    return pos, t


def trace(wavelength: float, spec_pos: np.ndarray, config: Config, params: Params):
    n1 = 1
    vertex = np.array((0, 0))
    position = np.array((0, params.initial_y))
    direction = np.array((1, 0))
    transmittance = 1
    yield position
    for glass, theta in zip(params.glasses, params.thetas):
        n2 = glass(wavelength)
        vertex = np.array((vertex[0] + np.tan(np.abs(theta)), 1 if vertex[1] == 0 else 0))
        position, direction, t = surface(n1, n2, theta, vertex, position, direction)
        transmittance *= t
        yield position
        n1 = n2
    n2 = 1
    theta = params.thetas[-1]
    position, normal = lens_intersection(vertex, theta, params.curvature, position, direction)
    direction, t = refract(n1, n2, normal, direction)
    transmittance *= t
    yield position
    normal = rotation(params.spec_angle) @ (-1, 0)
    assert -(normal @ direction) >= config.spec_min_ci
    position = plane_intersection(spec_pos, normal, position, direction)
    yield position


def merit(config: Config, params: Params):
    spec_pos = spectrometer_position(config, params)
    spec_dir = rotation(params.spec_angle) @ (0, 1)
    size = np.linalg.norm(spec_pos + spec_dir * config.spec_length / 2 - (0, 0.5))
    wlen = config.wmax - config.wmin
    nonlinearity = quad(
        lambda w: (spectrometer_intersection(w, params.initial_y, spec_pos, config, params)[0] ** 2
                   - ((w - config.wmin) / (config.wmax - config.wmin)) ** 2) ** 2, config.wmin, config.wmax)[0] / wlen
    spot_size = quad(
        lambda w: (spectrometer_intersection(w, params.initial_y + config.beam_radius, spec_pos, config, params)[0] -
                   spectrometer_intersection(w, params.initial_y - config.beam_radius, spec_pos, config, params)[0]
                   ) ** 2, config.wmin, config.wmax)[0] / wlen
    transmittance = 1 - quad(lambda w: spectrometer_intersection(w, params.initial_y, spec_pos, config, params)[1],
                         config.wmin, config.wmax)[0] / wlen
    return size, nonlinearity, spot_size, transmittance


if __name__ == "__main__":
    config = Config(3, 0.5, 0.82, 3.2, 0.01, 0.5)
    gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
    gcat = OrderedDict(sorted(gcat.items(), key=lambda p: p[1](config.wmin + (config.wmax - config.wmin) / 2)))
    gnames = list(gcat.keys())
    gfuncs = list(gcat.values())
    gs = [gcat[g] for g in "N-LASF41 SF57 N-SK11".split(" ")]
    angles = -17.469501, 77.99999, -49.67049, 35.29293
    angles = np.deg2rad(angles)
    ps = Params(gs, angles, 0.3, 0.9, 0)
    print(merit(config, ps))


    def fitness(p):
        try:
            glasses = [gfuncs[int(i)] for i in p[:config.prism_count]]
            return merit(config, Params(glasses, p[config.prism_count:config.prism_count * 2 + 1], *p[-3:])), [0]
        except AssertionError:
            return [np.inf] * 4, [1]

    problem = Problem(config.prism_count * 2 + 4, 4, 1)
    problem.types[:config.prism_count] = Real(0, len(gcat))
    l, u = np.deg2rad(2), np.deg2rad(80)
    problem.types[config.prism_count:2*config.prism_count+1] = [Real(l, u) if i % 2 else Real(-u, -l) for i in range(config.prism_count + 1)]
    sac = np.deg2rad(25)
    problem.types[2*config.prism_count+1:] = Real(0.1, 1.0), Real(config.beam_radius, 1-config.beam_radius), Real(-sac, sac)
    problem.constraints[:] = "==0"
    problem.function = fitness

    algorithm = NSGAII(problem)
    algorithm.run(1000)

    feasible_solutions = [s for s in algorithm.result if s.feasible]

    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    res = np.array([soln.objectives for soln in feasible_solutions])
    x, y, z, c = res.T
    c = 1 - c

    sc = ax.scatter(x, y, z, c=c, cmap="magma", picker=True)
    ax.set_xlabel("size")
    ax.set_xlim(0, 50)
    ax.set_ylabel("nonlinearity")
    ax.set_ylim(0, 1)
    ax.set_zlabel("spot size")
    ax.set_zlim(0, 0.1)
    clb = fig.colorbar(sc)
    clb.ax.set_ylabel('transmittance')

    px = fig.add_subplot(122)

    def pick(event):
        soln = sorted((feasible_solutions[i] for i in event.ind), key=lambda s: s.objectives[-1])[0]
        glasses = ','.join(gnames[int(i)] for i in soln.variables[:config.prism_count])
        angles = soln.variables[config.prism_count:config.prism_count * 2 + 1]
        curvature, initial_y, spec_angle = soln.variables[-3:]
        print(f"glasses: {glasses}, angles: {np.rad2deg(angles)}, curvature: {curvature:.2}, initial y: {initial_y:.2}, spec angle: {np.rad2deg(spec_angle):.2}")
        print("size: {:.2}, nonlinearity: {:.2}, spot_size: {:.2}, transmittance: {:.2}".format(*soln.objectives[:-1], 1 - soln.objectives[-1]))

        p = soln.variables
        gls = [gfuncs[int(i)] for i in p[:config.prism_count]]
        params = Params(gls, p[config.prism_count:config.prism_count * 2 + 1], *p[-3:])

        spec_pos = spectrometer_position(config, params)
        spec_dir = rotation(params.spec_angle) @ (0, 1)
        spec_end = spec_pos + spec_dir * config.spec_length

        prism_vertices = np.empty((config.prism_count + 2, 2))
        prism_vertices[::2, 1] = 0
        prism_vertices[1::2, 1] = 1
        prism_vertices[0, 0] = 0
        prism_vertices[1:, 0] = np.add.accumulate(np.tan(np.abs(angles)))
        triangles = np.stack((prism_vertices[1:-1], prism_vertices[:-2], prism_vertices[2:]), axis=1)

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
            ray = np.stack(tuple(trace(w, spec_pos, config, params)), axis=0)
            poly = plt.Polygon(ray, closed=None, fill=None, edgecolor=color)
            px.add_patch(poly)

        px.axis('scaled')
        px.axis('off')
        fig.canvas.draw()
        fig.canvas.flush_events()


    fig.canvas.mpl_connect('pick_event', pick)

    plt.show()
