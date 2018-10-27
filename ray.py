import math
import numpy as np
import numba as nb
from collections import namedtuple

Ray = namedtuple("Ray", "p, v, T")


@nb.njit
def ray_intersect_surface(ray, vertex, normal, n1, n2):
    r = n1 / n2
    ci = -(ray.v[0] * normal[0] + ray.v[1] * normal[1])
    # Line-Plane Intersection
    d = ((ray.p[0] - vertex[0]) * normal[0] + (ray.p[1] - vertex[1]) * normal[1]) / ci
    p = ray.p[0] + ray.v[0] * d, ray.p[1] + ray.v[1] * d
    if p[1] <= np.float32(0) or np.float32(1) <= p[1]:
        return
    # Snell's Law
    cr_sq = np.float32(1) - r ** 2 * (np.float32(1) - ci ** 2)
    if cr_sq < np.float32(0):
        return
    cr = math.sqrt(cr_sq)
    temp = r * ci - cr
    v = ray.v[0] * r + normal[0] * temp, ray.v[1] * r + normal[1] * temp
    # Surface Transmittance / Fresnel Equation
    fresnel_rs = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))
    fresnel_rp = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))
    transmittance = np.float32(1) - (fresnel_rs ** 2 + fresnel_rp ** 2) / np.float32(2)
    return Ray(p, v, ray.T * transmittance)


@nb.njit
def ray_intersect_lens(ray, midpt, normal, curvature, n1, n2):
    r = n1 / n2
    diameter = np.float32(1) / abs(normal[0])
    lens_radius = diameter / (np.float32(2) * curvature)
    rs = math.sqrt(lens_radius ** 2 - diameter ** 2 / np.float32(4))
    center = midpt[0] + normal[0] * rs, midpt[1] + normal[1] * rs
    delta = ray.p[0] - center[0], ray.p[1] - center[1]
    ud = ray.v[0] * delta[0] + ray.v[1] * delta[1]
    under = ud ** 2 - (delta[0] ** 2 + delta[1] ** 2) + lens_radius ** 2
    if under < np.float32(0):
        return
    d = -ud + math.sqrt(under)
    p = ray.p[0] + ray.v[0] * d, ray.p[1] + ray.v[1] * d
    rd2 = (ray.p[0] - midpt[0]) ** 2 + (ray.p[1] - midpt[1]) ** 2
    if rd2 > (diameter ** 2 / np.float32(4)):
        return
    snorm = (center[0] - p[0]) / lens_radius, (center[1] - p[1]) / lens_radius
    # Snell's Law
    ci = -(ray.v[0] * snorm[0] + ray.v[1] * snorm[1])
    cr_sq = np.float32(1) - r ** 2 * (np.float32(1) - ci ** 2)
    if cr_sq < np.float32(0):
        return
    cr = math.sqrt(cr_sq)
    temp = r * ci - cr
    v = ray.v[0] * r + snorm[0] * temp, ray.v[1] * r + snorm[1] * temp
    # Surface Transmittance / Fresnel Equation
    fresnel_rs = (n1 * ci - cr) / (n1 * ci + cr)
    fresnel_rp = (n1 * cr - ci) / (n1 * cr + ci)
    transmittance = np.float32(1) - (fresnel_rs ** 2 + fresnel_rp ** 2) / np.float32(2)
    return Ray(p, v, ray.T * transmittance)


@nb.njit
def ray_intersect_spectrometer(ray, upper_ray, lower_ray, spec_angle, spec_length):
    # Calc Spec Pos for maximal dispersion
    normal = -math.cos(spec_angle), -math.sin(spec_angle)
    det = upper_ray.v[1] * lower_ray.v[0] - upper_ray.v[0] * lower_ray.v[1]
    if det <= np.float32(0):
        return
    d2 = (upper_ray.v[0] * (lower_ray.p[1] - upper_ray.p[1] - spec_length * normal[0]) +
          upper_ray.v[1] * (upper_ray.p[0] - lower_ray.p[0] - spec_length * normal[1])) / det
    l_vertex = lower_ray.p[0] + lower_ray.v[0] * d2, lower_ray.p[1] + lower_ray.v[1] * d2
    # Calc ray's normalized spec pos
    ci = -(ray.v[0] * normal[0] + ray.v[1] * normal[1])
    d = ((ray.p[0] - l_vertex[0]) * normal[0] + (ray.p[1] - l_vertex[1]) * normal[1]) / ci
    p = ray.p[0] + ray.v[0] * d, ray.p[1] + ray.v[1] * d
    vdiff = p[0] - l_vertex[0], p[1] - l_vertex[1]
    spec_pos = math.copysign(math.sqrt(vdiff[0] ** 2 + vdiff[1] ** 2), vdiff[1])
    n_spec_pos = spec_pos / spec_length
    return n_spec_pos, Ray(p, ray.v, ray.T)
