import numpy as np
import numba as nb
import matplotlib as mpl
import matplotlib.pyplot as plt


@nb.guvectorize([
    "void(float32[:], float32[:], float32[:], float32, float32, float32[:, :], float32[:, :], float32[:])",
    "void(float64[:], float64[:], float64[:], float64, float64, float64[:, :], float64[:, :], float64[:])"
],
    "(p),(s),(d),(),()->(s, d),(s, d),(s)", nopython=True, cache=True, target='cpu')
def ray_trace(n, angles, init_dir, height, start, ray_path, ray_dir, transmittance):
    # First Surface
    n1 = 1
    n2 = n[0]
    r = n1 / n2
    norm = -np.cos(angles[0]), -np.sin(angles[0])
    size = height * abs(norm[1] / norm[0])
    ray_path[0, 0] = size - (height - start) * abs(norm[1] / norm[0])
    ray_path[0, 1] = start
    # Snell's Law
    ci = -(init_dir[0] * norm[0] + init_dir[1] * norm[1])
    cr = np.sqrt(1 - r * r * (1 - ci * ci))
    inner = r * ci - cr
    ray_dir[0, 0] = init_dir[0] * r + norm[0] * inner
    ray_dir[0, 1] = init_dir[1] * r + norm[1] * inner
    # Surface Transmittance / Fresnel Equation
    fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
    fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
    transmittance[0] = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2
    # Iterate over surfaces
    for i in range(1, n.shape[0] + 1):
        # New Surface Values
        n1 = n2
        n2 = n[i] if i < n.shape[0] else 1
        r = n1 / n2
        norm = -np.cos(angles[i]), -np.sin(angles[i])
        size += height * abs(norm[1] / norm[0])
        # Snell's Law
        ci = -(ray_dir[i - 1, 0] * norm[0] + ray_dir[i - 1, 1] * norm[1])
        cr = np.sqrt(1 - r * r * (1 - ci * ci))
        inner = r * ci - cr
        ray_dir[i, 0] = ray_dir[i - 1, 0] * r + norm[0] * inner
        ray_dir[i, 1] = ray_dir[i - 1, 1] * r + norm[1] * inner
        # Line-Plane Intersection
        vertex = size, 0 if i % 2 else height
        d = ((ray_path[i - 1, 0] - vertex[0]) * norm[0] + (ray_path[i - 1, 1] - vertex[1]) * norm[1]) / ci
        ray_path[i, 0] = d * ray_dir[i - 1, 0] + ray_path[i - 1, 0]
        ray_path[i, 1] = d * ray_dir[i - 1, 1] + ray_path[i - 1, 1]
        # Surface Transmittance / Fresnel Equation
        fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
        fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
        transmittance[i] = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2


@nb.guvectorize([
    "void(float64, float64[:], float64[:], float64[:], float64[:], float64, float64, float64[:, :], float64[:, :], float64[:])"
],
    "(),(d),(d),(d),(d),(),()->(d, d),(d, d),(d)", nopython=True, cache=True, target='cpu')
def spectro_ray_trace(lens_n, init_pos, init_dir, center, lens_dir, diameter, radius, ray_path, ray_dir, transmittance):
    # First Surface
    # Line-Sphere Intersection
    c = center + lens_dir * np.sqrt(radius * radius - diameter * diameter / 4)
    o = init_pos
    l = init_dir
    under = np.square(np.dot(l, o - c)) - np.sum(np.square(o - c)) + radius * radius
    if under <= 0:  # Check if passes through
        return
    d = -np.dot(l, o - c) - np.sqrt(under)
    ray_path[0] = x = o + d * l
    if np.sum(np.square(x - center)) > diameter * diameter / 4:
        return
    norm = (x - c) / radius
    # Snell's Law
    r = 1 / lens_n
    ci = -np.dot(l, norm)
    cr = np.sqrt(1 - r * r * (1 - ci * ci))
    inner = r * ci - cr
    ray_dir[0, 0] = l[0] * r + norm[0] * inner
    ray_dir[0, 1] = l[1] * r + norm[1] * inner
    # Surface Transmittance / Fresnel Equation
    fresnel_rs = (ci - lens_n * cr) / (ci + lens_n * cr)
    fresnel_rp = (cr - lens_n * ci) / (cr + lens_n * ci)
    transmittance[0] = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2
    # Second Surface
    # Line-Sphere Intersection
    c = center - lens_dir * np.sqrt(radius * radius - diameter * diameter / 4)
    o = x
    l = ray_dir[0]
    under = np.square(np.dot(l, o - c)) - np.sum(np.square(o - c)) + radius * radius
    if under <= 0:  # Check if passes through
        return
    d = -np.dot(l, o - c) + np.sqrt(under)
    ray_path[1] = x = o + d * l
    if np.sum(np.square(x - center)) > diameter * diameter / 4:
        return
    norm = (c - x) / radius
    # Snell's Law
    r = lens_n
    ci = -np.dot(l, norm)
    cr = np.sqrt(1 - r * r * (1 - ci * ci))
    inner = r * ci - cr
    ray_dir[1, 0] = ray_dir[0, 0] * r + norm[0] * inner
    ray_dir[1, 1] = ray_dir[0, 1] * r + norm[1] * inner
    # Surface Transmittance / Fresnel Equation
    fresnel_rs = (lens_n * ci - cr) / (lens_n * ci + cr)
    fresnel_rp = (lens_n * cr - ci) / (lens_n * cr + ci)
    transmittance[1] = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2


def describe(n, angles, config):
    assert np.all(np.abs(angles) < np.pi), 'Need Radians not Degrees'
    count, nwaves = n.shape
    # Ray Trace
    # * Prism
    ray_init_dir = np.cos(config["theta0"]), np.sin(config["theta0"])
    ray_path, ray_dir, transmittance = ray_trace(n.T, angles, ray_init_dir, config["height"], config["start"])
    transmittance = np.prod(transmittance, axis=1)
    if not (np.all(np.isfinite(ray_dir)) and
            np.all(np.logical_or(0 < ray_path[:, :, 1], ray_path[:, :, 1] < config["height"]))):
        return
    # * Lens
    radius = 10
    diameter = config["height"] + 5
    lens_dir = ray_dir[nwaves // 2, -1]
    center = ray_path[nwaves // 2, -1] + 2 * config["height"] * ray_dir[nwaves // 2, -1]
    c1 = center + lens_dir * np.sqrt(radius * radius - diameter * diameter / 4)
    c2 = center - lens_dir * np.sqrt(radius * radius - diameter * diameter / 4)
    r90 = np.array([[0, -1], [1, 0]])
    up = center + diameter / 2 * (r90 @ lens_dir)
    down = center - diameter / 2 * (r90 @ lens_dir)
    lens_path, lens_dir, lens_t = spectro_ray_trace(config["lens_n"], ray_path[:, -1], ray_dir[:, -1], center, lens_dir, diameter, radius)
    # * Spectrometer
    origin = center + 2 * config["height"] * ray_dir[nwaves // 2, -1]
    norm = -ray_dir[nwaves // 2, -1]
    ci = -np.dot(lens_dir[:, -1], norm)
    d = np.dot(lens_path[:, -1] - origin, norm) / ci
    spec_pos = np.stack((d * lens_dir[:, -1, 0] + lens_path[:, -1, 0], d * lens_dir[:, -1, 1] + lens_path[:, -1, 1]), 1)
    top = origin + config["height"] * -r90 @ norm
    bottom = origin + config["height"] * r90 @ norm
    # Calc Error
    delta_spectrum = np.arccos(ray_dir[:, -1, 0])
    deviation = delta_spectrum[nwaves // 2]
    dispersion = np.abs(delta_spectrum[-1] - delta_spectrum[0])
    mean_transmittance = np.sum(transmittance) / nwaves
    transmittance_err = np.max(1 - mean_transmittance, 0)
    nonlin = np.sqrt(np.sum(np.gradient(np.gradient(delta_spectrum)) ** 2))
    size = np.sum(config["height"] * np.tan(np.abs(angles)))
    merit_err = config["weight_deviation"] * (deviation - config["deviation_target"]) ** 2 \
                + config["weight_dispersion"] * (dispersion - config["dispersion_target"]) ** 2 \
                + config["weight_linearity"] * nonlin \
                + config["weight_transmittance"] * transmittance_err \
                + config["weight_thinness"] * max(size - config["max_size"], 0)
    # Create SVG
    prism_vertices = np.empty((count + 2, 2))
    prism_vertices[::2, 1] = 0
    prism_vertices[1::2, 1] = config["height"]
    prism_vertices[0, 0] = 0
    prism_vertices[1:, 0] = np.add.accumulate(config["height"] * np.tan(np.abs(angles)))
    triangles = np.stack((prism_vertices[:-2], prism_vertices[1:-1], prism_vertices[2:]), axis=1)

    plt.axes()
    for i, tri in enumerate(triangles):
        poly = plt.Polygon(tri, edgecolor='k', facecolor=('gray' if i % 2 else 'white'))
        plt.gca().add_patch(poly)
    t1 = np.rad2deg(np.arccos((up - c1)[0] / (np.linalg.norm(up - c1))))
    t2 = np.rad2deg(2 * np.pi - np.arccos((down - c1)[0] / (np.linalg.norm(down - c1))))
    t4 = np.rad2deg(np.arccos((up - c2)[0] / (np.linalg.norm(up - c2))))
    t3 = np.rad2deg(2 * np.pi - np.arccos((down - c2)[0] / (np.linalg.norm(down - c2))))
    arc1 = mpl.patches.Arc(c1, 2*radius, 2*radius, theta1=t1, theta2=t2)
    arc2 = mpl.patches.Arc(c2, 2*radius, 2*radius, theta1=t3, theta2=t4)
    plt.gca().add_patch(arc1)
    plt.gca().add_patch(arc2)
    spectro = plt.Polygon((top, bottom), closed=None, fill=None, edgecolor='k')
    plt.gca().add_patch(spectro)
    for ind, color in zip((0, nwaves//2, -1), ('r', 'g', 'b')):
        ray = np.stack((*ray_path[ind], *lens_path[ind], spec_pos[ind]), axis=0)
        poly = plt.Polygon(ray, closed=None, fill=None, edgecolor=color)
        plt.gca().add_patch(poly)
    plt.axis('scaled')
    plt.axis('off')
    plt.show()
    return merit_err, nonlin, dispersion, deviation, size, delta_spectrum, transmittance
