import numpy as np
import numba as nb
from svgwrite import Drawing


def describe(n, angles, config):
    assert np.all(np.abs(angles) < np.pi), 'Need Radians not Degrees'
    count, nwaves = n.shape
    # Set initial values
    prism_vertices = np.empty((count + 2, 2))
    prism_vertices[::2, 1] = 0
    prism_vertices[1::2, 1] = config["height"]
    prism_vertices[0, 0] = 0
    prism_vertices[1:, 0] = np.add.accumulate(config["height"] * np.tan(np.abs(angles)))
    normals = np.stack((-np.cos(angles), -np.sin(angles)), axis=1)  # Rotation of (-1, 0) by the angles CCW
    ray_path = np.empty((nwaves, count + 1, 2))
    ray_dir = np.empty((nwaves, count + 1, 2))
    ray_init_dir = np.array((np.cos(config["theta0"]), np.sin(config["theta0"])))
    ray_path[:, 0, 0] = prism_vertices[1, 0] - (config["height"] - config["start"]) * np.tan(np.abs(angles[0]))
    ray_path[:, 0, 1] = config["start"]
    # First Surface
    n1 = 1
    n2 = n[0]
    r = n1 / n2
    # Snell's Law
    ci = -(ray_init_dir @ normals[0])
    cr_sq = 1 - r * r * (1 - ci * ci)
    if np.any(cr_sq < 0):
        return
    cr = np.sqrt(cr_sq)
    inner = r * ci - cr
    ray_dir[:, 0, 0] = ray_init_dir[0] * r + normals[0, 0] * inner
    ray_dir[:, 0, 1] = ray_init_dir[1] * r + normals[0, 1] * inner
    # Surface Transmittance / Fresnel Equation
    fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
    fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
    transmittance = 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2
    # Iterate over surfaces
    for i in range(1, count + 1):
        # New Surface Values
        n1 = n2
        n2 = n[i] if i < count else 1
        r = n1 / n2
        # Snell's Law
        ci = -(ray_dir[:, i - 1] @ normals[i])
        cr_sq = 1 - r * r * (1 - ci * ci)
        if np.any(cr_sq < 0):
            return
        cr = np.sqrt(cr_sq)
        inner = r * ci - cr
        ray_dir[:, i, 0] = ray_dir[:, i - 1, 0] * r + normals[i, 0] * inner
        ray_dir[:, i, 1] = ray_dir[:, i - 1, 1] * r + normals[i, 1] * inner
        # Line-Plane Intersection
        d = ((ray_path[:, i - 1] - prism_vertices[i + 1]) @ normals[i]) / ci
        ray_path[:, i, 0] = d * ray_dir[:, i - 1, 0] + ray_path[:, i - 1, 0]
        ray_path[:, i, 1] = d * ray_dir[:, i - 1, 1] + ray_path[:, i - 1, 1]
        if np.any(np.logical_or(ray_path[:, i, 1] <= 0, ray_path[:, i, 1] >= config["height"])):
            return
        # Surface Transmittance / Fresnel Equation
        fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)
        fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)
        transmittance *= 1 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / 2
    delta_spectrum = np.arccos(ray_dir[:, -1, 0])
    deviation = delta_spectrum[nwaves // 2]
    dispersion = np.abs(delta_spectrum[-1] - delta_spectrum[0])
    mean_transmittance = np.sum(transmittance) / nwaves
    transmittance_err = np.max(1 - mean_transmittance, 0)
    nonlin = np.sqrt(np.sum(np.gradient(np.gradient(delta_spectrum)) ** 2))
    size = prism_vertices[-1, 0]
    merit_err = config["weight_deviation"] * (deviation - config["deviation_target"]) ** 2 \
                + config["weight_dispersion"] * (dispersion - config["dispersion_target"]) ** 2 \
                + config["weight_linearity"] * nonlin \
                + config["weight_transmittance"] * transmittance_err \
                + config["weight_thinness"] * max(size - config["max_size"], 0)

    triangles = np.stack((prism_vertices[:-2], prism_vertices[1:-1], prism_vertices[2:]), axis=1)
    # Mirror for svg
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
