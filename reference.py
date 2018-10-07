import numpy as np
import numba as nb
from svgwrite import Drawing


@nb.guvectorize([
    "void(float32[:], float32[:], float32[:], float32, float32, float32[:, :], float32[:, :], float32[:])",
    "void(float64[:], float64[:], float64[:], float64, float64, float64[:, :], float64[:, :], float64[:])"
],
    "(p),(s),(d),(),()->(s, d),(s, d),(s)", cache=True)
def raytrace(n, angles, init_dir, height, start, ray_path, ray_dir, transmittance):
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
        d = ((ray_path[i - 1, 0] - size) * norm[0] + (ray_path[i - 1, 1] - (0 if i % 2 else height)) * norm[1]) / ci
        ray_path[i, 0] = d * ray_dir[i - 1, 0] + ray_path[i - 1, 0]
        ray_path[i, 1] = d * ray_dir[i - 1, 1] + ray_path[i - 1, 1]
        # Surface Transmittance / Fresnel Equation
        fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
        fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
        transmittance[i] = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2


def describe(n, angles, config):
    assert np.all(np.abs(angles) < np.pi), 'Need Radians not Degrees'
    count, nwaves = n.shape
    # Ray Trace
    ray_init_dir = np.array((np.cos(config["theta0"]), np.sin(config["theta0"])))
    ray_path, ray_dir, transmittance = raytrace(n.T, angles, ray_init_dir, config["height"], config["start"])
    transmittance = np.prod(transmittance, axis=1)
    if not (np.all(np.isfinite(ray_dir)) and
            np.all(np.logical_or(0 < ray_path[:, :, 1], ray_path[:, :, 1] < config["height"]))):
        return
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
    triangles[:, :, 1] *= -1
    triangles[:, :, 1] += config["height"]
    ray_path[:, :, 1] *= -1
    ray_path[:, :, 1] += config["height"]
    dwg = Drawing('figure.svg')
    for i, tri in enumerate(triangles):
        dwg.add(dwg.polygon((10 * tri).tolist(), fill=('gray' if i % 2 else 'white')))
    dwg.add(dwg.polyline((10 * ray_path[0]).tolist(), stroke='red', stroke_width=1, fill='none'))
    dwg.add(dwg.polyline((10 * ray_path[nwaves // 2]).tolist(), stroke='green', stroke_width=1, fill='none'))
    dwg.add(dwg.polyline((10 * ray_path[-1]).tolist(), stroke='blue', stroke_width=1, fill='none'))
    dwg.save()

    return merit_err, nonlin, dispersion, deviation, size, delta_spectrum, transmittance
