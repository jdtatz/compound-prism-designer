import numpy as np
import numba as nb
import matplotlib as mpl
import matplotlib.pyplot as plt


@nb.guvectorize([
    "void(float32[:], float32[:], float32[:], float32, float32, float32[:, :], float32[:, :], float32[:])",
    "void(float64[:], float64[:], float64[:], float64, float64, float64[:, :], float64[:, :], float64[:])"
],
    "(p),(s),(d),(),()->(s, d),(s, d),(s)", nopython=True, cache=True, target='cpu')
def ray_trace(n, angles, init_dir, start, curvature, ray_path, ray_dir, transmittance):
    # First Surface
    n1 = 1
    n2 = n[0]
    r = n1 / n2
    norm = -np.cos(angles[0]), -np.sin(angles[0])
    size = abs(norm[1] / norm[0])
    start = np.float32(0.9)
    ray_path[0, 0] = size - (1 - start) * abs(norm[1] / norm[0])
    ray_path[0, 1] = start
    # Snell's Law
    ci = -(init_dir[0] * norm[0] + init_dir[1] * norm[1])
    cr_sq = 1 - r * r * (1 - ci * ci)
    assert cr_sq > 0, "Failed at Initial Surface"
    cr = np.sqrt(cr_sq)
    inner = r * ci - cr
    ray_dir[0, 0] = init_dir[0] * r + norm[0] * inner
    ray_dir[0, 1] = init_dir[1] * r + norm[1] * inner
    # Surface Transmittance / Fresnel Equation
    fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
    fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
    transmittance[0] = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2
    # Iterate over inner surfaces
    for i in range(1, n.shape[0]):
        # New Surface Values
        n1 = n2
        n2 = n[i]
        r = n1 / n2
        norm = -np.cos(angles[i]), -np.sin(angles[i])
        size += abs(norm[1] / norm[0])
        # Snell's Law
        ci = -(ray_dir[i - 1, 0] * norm[0] + ray_dir[i - 1, 1] * norm[1])
        cr_sq = 1 - r * r * (1 - ci * ci)
        assert cr_sq > 0, "Failed at Inner Surface"
        cr = np.sqrt(cr_sq)
        inner = r * ci - cr
        ray_dir[i, 0] = ray_dir[i - 1, 0] * r + norm[0] * inner
        ray_dir[i, 1] = ray_dir[i - 1, 1] * r + norm[1] * inner
        # Line-Plane Intersection
        vertex = size, 0 if i % 2 else 1
        d = ((ray_path[i - 1, 0] - vertex[0]) * norm[0] + (ray_path[i - 1, 1] - vertex[1]) * norm[1]) / ci
        ray_path[i, 0] = d * ray_dir[i - 1, 0] + ray_path[i - 1, 0]
        ray_path[i, 1] = d * ray_dir[i - 1, 1] + ray_path[i - 1, 1]
        assert 0 < ray_path[i, 1] < 1, "Escaped inside"
        # Surface Transmittance / Fresnel Equation
        fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
        fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
        transmittance[i] = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2
    # Last / Convex Surface
    n1 = n2
    n2 = 1
    r = n1
    norm = -np.cos(angles[-1]), -np.sin(angles[-1])
    diff = abs(norm[1] / norm[0])
    size += diff
    diameter = 1 / abs(norm[0])
    midpt = size - diff / 2, 1 / 2
    # Line-Sphere Intersection
    lens_radius = (diameter / 2) / curvature
    rs = diameter * np.sqrt(1 - curvature*curvature) / (2 * curvature)
    c = midpt[0] + norm[0] * rs, midpt[1] + norm[1] * rs
    o = ray_path[-2]
    l = ray_dir[-2]
    under = (l[0]*(o[0] - c[0]) + l[1]*(o[1] - c[1]))**2 - ((o[0] - c[0])**2+(o[1] - c[1])**2) + lens_radius * lens_radius
    assert under > 0, "ray doesn't intersect lens sphere"
    d = -(l[0]*(o[0] - c[0]) + l[1]*(o[1] - c[1])) + np.sqrt(under)
    x = o[0] + d * l[0], o[1] + d * l[1]
    ray_path[-1, 0], ray_path[-1, 1] = x
    assert ((x[0] - midpt[0])**2+(x[1] - midpt[1])**2) <= diameter * diameter / 4, "ray doesn't intersect lens surface"
    snorm = (c[0] - x[0]) / lens_radius, (c[1] - x[1]) / lens_radius
    # Snell's Law
    ci = -(ray_dir[-2, 0] * snorm[0] + ray_dir[-2, 1] * snorm[1])
    cr = np.sqrt(1 - r * r * (1 - ci * ci))
    inner = r * ci - cr
    ray_dir[-1, 0] = ray_dir[-2, 0] * r + snorm[0] * inner
    ray_dir[-1, 1] = ray_dir[-2, 1] * r + snorm[1] * inner
    # Surface Transmittance / Fresnel Equation
    fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
    fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
    transmittance[-1] = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2


def describe(n, angles, curvature, distance, config):
    assert np.all(np.abs(angles) < np.pi), 'Need Radians not Degrees'
    count, nwaves = n.shape
    # Ray Trace
    # * Prism
    ray_init_dir = np.cos(config["theta0"]), np.sin(config["theta0"])
    ray_path, ray_dir, transmittance = ray_trace(n.T, angles, ray_init_dir, config["start"], curvature)
    transmittance = np.prod(transmittance, axis=1)
    # Spectrometer
    dist = distance * (config["max_size"] - ray_path[nwaves//2, -1, 0])
    diameter = 1 / np.cos(np.abs(angles[-1]))
    norm = ray_dir[nwaves // 2, -1]
    vertex = ray_path[nwaves//2, -1] + dist * norm
    ci = -np.dot(ray_dir[:, -1], norm)
    d = np.dot(ray_path[:, -1] - vertex, norm) / ci
    end = np.stack((d * ray_dir[:, -1, 0] + ray_path[:, -1, 0], d * ray_dir[:, -1, 1] + ray_path[:, -1, 1]), 1)
    vdiff = end - vertex
    spec_pos = np.copysign(np.sqrt(vdiff[:, 0] * vdiff[:, 0] + vdiff[:, 1] * vdiff[:, 1]), vdiff[:, 1])
    plt.plot(spec_pos)
    plt.show()
    # Calc Error
    mean_transmittance = np.sum(transmittance) / nwaves
    deviation = abs(ray_dir[nwaves//2, -1, 1])
    dispersion = abs(spec_pos[-1] - spec_pos[0]) / config["sheight"]
    nonlin = np.sqrt(np.sum(np.gradient(np.gradient(spec_pos)) ** 2))
    size = np.sum(np.tan(np.abs(angles)))
    merit_err = config["weight_deviation"] * deviation \
                + config["weight_dispersion"] * (1 - dispersion) \
                + config["weight_linearity"] * nonlin \
                + config["weight_transmittance"] * (np.float32(1) - mean_transmittance)
    # Create SVG
    prism_vertices = np.empty((count + 2, 2))
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

    norm = ray_dir[nwaves // 2, -1]
    sc = ray_path[nwaves//2, -1] + dist * norm
    v = np.array((-norm[1], norm[0]))
    top = sc + v * config["sheight"] / 2
    bottom = sc - v * config["sheight"] / 2

    plt.axes()
    for i, tri in enumerate(triangles):
        poly = plt.Polygon(tri, edgecolor='k', facecolor=('gray' if i % 2 else 'white'), closed=False)
        plt.gca().add_patch(poly)
    t1 = np.rad2deg(np.arctan2(prism_vertices[-1, 1] - c[1], prism_vertices[-1, 0] - c[0]))
    t2 = np.rad2deg(np.arctan2(prism_vertices[-2, 1] - c[1], prism_vertices[-2, 0] - c[0]))
    arc = mpl.path.Path.arc(t1, t2)
    arc = mpl.path.Path(arc.vertices * lradius + c, arc.codes)
    arc = mpl.patches.PathPatch(arc, fill=None)
    plt.gca().add_patch(arc)
    spectro = plt.Polygon((top, bottom), closed=None, fill=None, edgecolor='k')
    plt.gca().add_patch(spectro)
    for ind, color in zip((0, nwaves//2, -1), ('r', 'g', 'b')):
        ray = np.stack((*ray_path[ind], end[ind]), axis=0)
        poly = plt.Polygon(ray, closed=None, fill=None, edgecolor=color)
        plt.gca().add_patch(poly)
    plt.axis('scaled')
    plt.axis('off')
    plt.show()
    return merit_err, nonlin, dispersion, deviation, size, spec_pos, transmittance
