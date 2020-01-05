from __future__ import annotations
import numpy as np
from numpy import sqrt, cos, sin, tan
from numpy.linalg import inv
from scipy import stats
from typing import NamedTuple, Tuple, Sequence, Callable
import numba as nb
import numba.extending
from numba import prange
import inspect
import operator
from functools import wraps, partial


def overload_helper(cls, builtin, fn):
    @nb.extending.overload(builtin)
    @wraps(fn)
    def _vector_overload(self, *args, **kwargs):
        if isinstance(self, nb.types.BaseNamedTuple) and self.instance_class is cls:
            return fn


def overload_method_helper(cls, method_name, method_fn):
    @nb.extending.overload_method(nb.types.BaseNamedTuple, method_name)
    @wraps(method_fn)
    def _vector_overload(self, *args, **kwargs):
        if isinstance(self, nb.types.BaseNamedTuple) and self.instance_class is cls:
            return method_fn


def overload_getattr_helper(cls, attr_name, attr_fn):
    @nb.extending.overload_attribute(nb.types.BaseNamedTuple, attr_name)
    @wraps(attr_fn)
    def _vector_overload(self, *args, **kwargs):
        if isinstance(self, nb.types.BaseNamedTuple) and self.instance_class is cls:
            return attr_fn


def overload_named_tuple_subclass(cls):
    _prohibited = ('__new__', '__init__', '__slots__', '__getnewargs__',
                   '_fields', '_field_defaults', '_field_types',
                   '_make', '_replace', '_asdict', '_source')
    _special = ('__module__', '__name__', '__qualname__', '__annotations__')
    filtered = _prohibited + _special + ('__doc__', '_fields_defaults', '__repr__')
    for name, val in cls.__dict__.items():
        if name in filtered:
            continue
        if inspect.isroutine(val):
            if hasattr(operator, name) and inspect.isbuiltin(getattr(operator, name)):
                overload_helper(cls, getattr(operator, name), val)
            elif inspect.isfunction(val):
                overload_method_helper(cls, name, val)
            elif not (isinstance(val, classmethod) or isinstance(val, staticmethod)):
                overload_getattr_helper(cls, name, val)
    return cls


@overload_named_tuple_subclass
class Vector(NamedTuple):
    x: float
    y: float

    def __neg__(self) -> Vector:
        return Vector(-self.x, -self.y)

    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> Vector:
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> Vector:
        return Vector(self.x / other, self.y / other)

    def __matmul__(self, other: Vector) -> float:
        return self.x * other.x + self.y * other.y

    def norm_squared(self) -> float:
        return self @ self

    def norm(self) -> float:
        return sqrt(self.norm_squared())


class Welford:
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.m2 = 0

    def next_sample(self, x):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def sample_variance(self):
        return self.m2 / (self.count - 1)


def phi(d):
    p = np.zeros(d + 2)
    p[0] = 1
    p[-1] = p[-2] = -1
    r = np.roots(p)
    return r[0].real


assert (np.allclose(phi(1), 1.6180339887498948482))
assert (np.allclose(phi(2), 1.32471795724474602596))
assert (np.allclose(phi(3), 1.22074408460575947536))
phi = (1.6180339887498948482, 1.32471795724474602596, 1.22074408460575947536)


def quasi(seed, dim):
    g = phi[dim]
    alpha = g ** -(1 + np.arange(dim))
    z = (seed + alpha) % 1
    while True:
        yield z
        z = (z + alpha) % 1


def quasi_monte_carlo_importance_sampling(f, distr, err, max_eval=100_000, seed=0.5):
    integrand = Welford()
    q = quasi(seed, 1)
    for _ in range(max_eval):
        x = distr.ppf(next(q))
        integrand.next_sample(f(*x))
        if integrand.count > 2:
            # err ^ 2 = sample_variance / N
            if np.all(integrand.sample_variance() < err * err * integrand.count):
                return integrand.mean
    print("Warning Reached max iter", integrand.sample_variance())
    return integrand.mean


def quasi_monte_carlo_importance_sampling_nd(f, distrs, err, max_eval=1_000_000, seed=0.5):
    integrand = Welford()
    q = quasi(seed, len(distrs))
    for _ in range(max_eval):
        x = [distr.ppf(v) for distr, v in zip(distrs, next(q))]
        integrand.next_sample(f(*x))
        if integrand.count > 2:
            var = integrand.sample_variance()
            # err = sample_variance / sqrt(N)
            if np.all(var * var < err * err * integrand.count):
                return integrand.mean
    print("Warning Reached max eval", integrand.sample_variance())
    return integrand.mean


def sellmeier1(*args):
    return tuple(map(float, args))


@nb.njit
def refractive_index(glass, w):
    b1, c1, b2, c2, b3, c3 = glass
    w2 = w * w
    return sqrt(1 + b1 * w2 / (w2 - c1) + b2 * w2 / (w2 - c2) + b3 * w2 / (w2 - c3))


def rotate(angle, vector):
    c, s = cos(angle), sin(angle)
    r = np.array(((c, -s), (s, c)))
    return Vector(*(r @ vector))


class RayTraceError(Exception):
    pass


class Surface(NamedTuple):
    """"""
    angle: float
    normal: Vector
    midpt: Vector

    @staticmethod
    def first_surface(angle, height):
        normal = rotate(angle, (-1, 0))
        midpt = Vector(abs(tan(angle)) * height / 2, height / 2)
        return Surface(angle=angle, normal=normal, midpt=midpt)

    def next_surface(self, angle, height, separation_length):
        normal = rotate(angle, (-1, 0))
        dx1 = abs(tan(self.angle)) * height / 2
        dx2 = abs(tan(angle)) * height / 2
        separation_distance = separation_length + \
                              (dx1 + dx2 if (self.normal[1] >= 0) != (normal[1] >= 0) else abs(dx1 - dx2))
        return Surface(angle=angle, normal=normal, midpt=self.midpt + Vector(separation_distance, 0))


@overload_named_tuple_subclass
class CurvedSurface(NamedTuple):
    midpt: Vector
    center: Vector
    radius: float
    max_dist_sq: float

    @staticmethod
    def from_chord(curvature, height, chord: Surface):
        chord_length = height / cos(chord.angle)
        radius = chord_length / 2 / curvature
        apothem = sqrt(radius*radius - chord_length*chord_length / 4)
        sagitta = radius - apothem
        center = chord.midpt + chord.normal * apothem
        midpt = chord.midpt - chord.normal * sagitta
        return CurvedSurface(midpt=midpt, center=center, radius=radius,
                             max_dist_sq=sagitta*sagitta + chord_length*chord_length / 4)

    def is_along_arc(self, pt):
        diff = pt - self.midpt
        return (diff @ diff) <= self.max_dist_sq


class GaussianBeam(NamedTuple):
    """Collimated Polychromatic Gaussian Beam

    Attributes:
        width (float): 1/e^2 beam width
        y_mean (float): Mean y coordinate
        wavelength_range ((float, float)): Range of wavelengths
    """
    width: float
    y_mean: float
    wavelength_range: Tuple[float, float]


class CompoundPrism(NamedTuple):
    prisms: Sequence[(Callable[[float], float], Surface)]
    lens: CurvedSurface
    height: float
    width: float

    @staticmethod
    def create(glasses, angles, lengths, curvature, height, width):
        prisms = []
        last_surface = Surface.first_surface(angles[0], height)
        for glass, angle, l in zip(glasses, angles[1::], lengths):
            next = last_surface.next_surface(angle, height, l)
            prisms.append((glass, last_surface))
            last_surface = next
        lens = CurvedSurface.from_chord(curvature, height, last_surface)
        return CompoundPrism(prisms=tuple(prisms), lens=lens, height=height, width=width)


class DetectorArray(NamedTuple):
    bins: np.ndarray
    min_ci: float
    angle: float
    length: float
    normal: Vector

    @staticmethod
    def create(bins, min_ci, angle, length):
        return DetectorArray(
            bins=bins,
            min_ci=min_ci,
            angle=angle,
            length=length,
            normal=rotate(angle, (-1, 0))
        )


class DetectorArrayPositioning(NamedTuple):
    position: Vector
    direction: Vector


@overload_named_tuple_subclass
class Ray(NamedTuple):
    origin: Vector
    direction: Vector
    s_transmittance: float
    p_transmittance: float

    @staticmethod
    def new_from_start(y):
        return Ray(
            origin=Vector(0., y),
            direction=Vector(1., 0.),
            s_transmittance=1.,
            p_transmittance=1.
        )

    def average_transmittance(self):
        return (self.s_transmittance + self.p_transmittance) / 2

    def refract(self, intersection, normal, ci, n1, n2):
        r = n1 / n2
        cr_sq = 1 - r*r * (1 - ci*ci)
        if cr_sq < 0:
            return None
            # raise RayTraceError("TotalInternalReflection")
        cr = sqrt(cr_sq)
        v = self.direction * r + normal * (r * ci - cr)
        s_reflection_sqrt = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))
        p_reflection_sqrt = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))
        s_transmittance = 1 - s_reflection_sqrt * s_reflection_sqrt
        p_transmittance = 1 - p_reflection_sqrt * p_reflection_sqrt
        return Ray(
            origin=intersection,
            direction=v,
            s_transmittance=self.s_transmittance * s_transmittance,
            p_transmittance=self.p_transmittance * p_transmittance,
        )

    def intersect_plane_interface(self, plane: Surface, n1, n2, prism_height):
        ci = -(self.direction @ plane.normal)
        if ci <= 0:
            return None
            # raise RayTraceError("OutOfBounds")
        d = (self.origin - plane.midpt) @ plane.normal / ci
        p = self.origin + self.direction * d
        if p[1] <= 0 or prism_height <= p[1]:
            return None
            # raise RayTraceError("OutOfBounds")
        return self.refract(p, plane.normal, ci, n1, n2)

    def intersect_curved_interface(self, lens: CurvedSurface, n1, n2):
        delta = self.origin - lens.center
        ud = self.direction @ delta
        under = ud*ud - delta.norm_squared() + lens.radius*lens.radius
        if under < 0:
            return None
            # raise RayTraceError("NoSurfaceIntersection")
        d = -ud + sqrt(under)
        if d <= 0:
            return None
            # raise RayTraceError("NoSurfaceIntersection")
        p = self.origin + self.direction * d
        if not lens.is_along_arc(p):
            return None
            # raise RayTraceError("NoSurfaceIntersection")
        snorm = (lens.center - p) / lens.radius
        return self.refract(p, snorm, -(self.direction @ snorm), n1, n2)

    def intersect_detector_array(self, detarr: DetectorArray, detpos: DetectorArrayPositioning):
        ci = -(self.direction @ detarr.normal)
        if ci <= detarr.min_ci:
            return None
            # raise RayTraceError("SpectrometerAngularResponseTooWeak")
        d = (self.origin - detpos.position) @ detarr.normal / ci
        p = self.origin + self.direction * d
        pos = (p - detpos.position) @ detpos.direction
        if pos < 0 or detarr.length < pos:
            return None
            # raise RayTraceError("NoSurfaceIntersection")
        return p, pos, self.average_transmittance()

    def propagate_internal(self, cmpnd: CompoundPrism, wavelength):
        ray, n1 = self, 1
        for glass, plane in cmpnd.prisms:
            n2 = refractive_index(glass, wavelength)
            ray = ray.intersect_plane_interface(plane, n1, n2, cmpnd.height)
            if ray is None:
                return None
            n1 = n2
        n2 = 1
        return ray.intersect_curved_interface(cmpnd.lens, n1, n2)

    def propagate(self, wavelength, cmpnd: CompoundPrism, detarr: DetectorArray, detpos: DetectorArrayPositioning):
        internal = self.propagate_internal(cmpnd, wavelength)
        if internal is None:
            return None
        else:
            return internal.intersect_detector_array(detarr, detpos)


def detector_array_positioning(cmpnd: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam):
    ray = Ray.new_from_start(beam.y_mean)
    wmin, wmax = beam.wavelength_range
    lower_ray = ray.propagate_internal(cmpnd, wmin)
    upper_ray = ray.propagate_internal(cmpnd, wmax)
    if lower_ray is None or upper_ray is None or lower_ray.average_transmittance() <= 1e-3 or upper_ray.average_transmittance() <= 1e-3:
        raise RayTraceError
    spec_dir = rotate(detarr.angle, (0, 1))
    spec = spec_dir * detarr.length
    mat = np.array((upper_ray.direction, -lower_ray.direction)).T
    imat = inv(mat)
    dists = imat @ (spec - upper_ray.origin + lower_ray.origin)
    d2 = dists[1]
    l_vertex = lower_ray.origin + lower_ray.direction * d2
    if d2 > 0:
        return DetectorArrayPositioning(l_vertex, spec_dir)
    dists = imat @ (-spec - upper_ray.origin + lower_ray.origin)
    d2 = dists[1]
    if d2 < 0:
        raise RayTraceError
    u_vertex = lower_ray.origin + lower_ray.direction * d2
    return DetectorArrayPositioning(u_vertex, -spec_dir)


@nb.vectorize
def plog2p(p):
    if p == 0.:
        return 0.
    else:
        return p * np.log2(p)


new_ray_from_start = nb.njit(Ray.new_from_start)


@nb.njit(nogil=True, error_model='numpy')
def propagate(cmpnd, detarr, detpos, w, y, positions, transmissions):
    for i in range(len(w)):
        for j in range(len(y)):
            r = new_ray_from_start(y[j]).propagate(w[i], cmpnd, detarr, detpos)
            if r is None:
                pos, t = np.nan, 0.
            else:
                _, pos, t = r
            positions[i, j] = pos
            transmissions[i, j] = t


@nb.njit(nogil=True, error_model='numpy')
def next_sample(means, v, count):
    means += (v - means) / np.float64(count)


@nb.njit(nogil=True, error_model='numpy')
def _mutual_info(cmpnd, detarr, detpos, w, y, pZ):
    p_dets = np.zeros(len(detarr.bins), dtype=np.float64)
    h_dets_l_w = np.zeros(len(detarr.bins), dtype=np.float64)
    for i in range(len(w)):
        p_dets_l_w = np.zeros(len(detarr.bins), dtype=np.float64)
        for j in range(len(y)):
            r = new_ray_from_start(y[j]).propagate(w[i], cmpnd, detarr, detpos)
            if r is None:
                next_sample(p_dets_l_w, 0, j + 1)
            else:
                _, pos, t = r
                # bin_test = np.array([detarr.bins[k][0] <= pos < detarr.bins[k][1] for k in range(len(detarr.bins))])
                bin_test = np.logical_and(detarr.bins[:, 0] <= pos, pos < detarr.bins[:, 1])
                p = np.where(bin_test, t * pZ, 0.)
                next_sample(p_dets_l_w, p, j + 1)
        next_sample(h_dets_l_w, plog2p(p_dets_l_w), i + 1)
        next_sample(p_dets, p_dets_l_w, i + 1)
    return np.sum(h_dets_l_w - plog2p(p_dets))


def fitness(cmpnd: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam):
    detpos = detector_array_positioning(cmpnd, detarr, beam)
    deviation_vector = detpos.position + detpos.direction * detarr.length / 2 - Vector(0, beam.y_mean)
    size = np.linalg.norm(deviation_vector)
    deviation = abs(deviation_vector[1]) / np.linalg.norm(deviation_vector)

    Y = stats.norm(loc=beam.y_mean, scale=beam.width / 2)
    Z = stats.norm(loc=0, scale=beam.width / 2)
    pZ = Z.cdf(cmpnd.width / 2) - Z.cdf(-cmpnd.width / 2)
    L = stats.uniform(loc=beam.wavelength_range[0], scale=beam.wavelength_range[1] - beam.wavelength_range[0])  # Λ

    def mutual_info(positions, transmissions, lb, ub):
        p = np.where(np.logical_and(lb <= positions, positions < ub), pZ * transmissions, 0)
        p_det_l_w = np.mean(p, axis=1)
        p_det = np.mean(p_det_l_w)
        # print(f"p(d=D) error: {100 * np.std(p, ddof=1) / np.sqrt(p.size):.3}%")
        # print(f"p(d=D|w=W) avg error: {100 * np.std(p, axis=1, ddof=1).mean() / np.sqrt(p.shape[1]):.3}%")
        return np.mean(plog2p(p_det_l_w)) - plog2p(p_det)

    ws = np.fromiter(map(L.ppf, quasi(0., 1)), np.float64, 1_000)
    ys = np.fromiter(map(Y.ppf, quasi(0., 1)), np.float64, 1_000)
    # print(ws, ys)
    '''
    print('pre prop')
    pos = np.empty((len(ws), len(ys)), dtype=np.float64)
    t = np.empty((len(ws), len(ys)), dtype=np.float64)
    propagate(cmpnd, detarr, detpos, ws, ys, pos, t)
    print('post prop')
    info = np.array([
        mutual_info(pos, t, lb, ub)
        for lb, ub in detarr.bins
    ])
    print(info)
    print(np.sum(info))
    print(sum(mutual_info(pos, t, lb, ub) for lb, ub in detarr.bins))
    '''
    info = _mutual_info(cmpnd, detarr, detpos, ws, ys, pZ)
    print(info)
    from timeit import timeit
    n = 20
    print(timeit(lambda: (propagate(cmpnd, detarr, detpos, ws, ys, pos, t), sum(mutual_info(pos, t, lb, ub) for lb, ub in detarr.bins)), number=n) / n)
    print(timeit(lambda: propagate(cmpnd, detarr, detpos, ws, ys, pos, t), number=n) / n)
    print(timeit(lambda: sum(mutual_info(pos, t, lb, ub) for lb, ub in detarr.bins), number=n) / n)
    print(timeit(lambda: _mutual_info(cmpnd, detarr, detpos, ws, ys, pZ), number=n) / n)
    '''
    sig = nb.typeof(cmpnd), nb.typeof(detarr), nb.typeof(detpos), nb.f8[:], nb.f8[:], nb.f8[:, ::], nb.f8[:, ::]
 
    with open('test2.ll', 'w') as f:
        f.write(propagate.inspect_llvm(sig))
    print(propagate.inspect_asm(sig))
    propagate.inspect_cfg(sig).display(view=True)
    # print(propagate.inspect_llvm())
    '''
    return size, deviation, info


def test():
    glasses = [
        sellmeier1(
            1.029607,
            0.00516800155,
            0.1880506,
            0.0166658798,
            0.736488165,
            138.964129
        ),
        sellmeier1(
            1.87543831,
            0.0141749518,
            0.37375749,
            0.0640509927,
            2.30001797,
            177.389795
        ),
        sellmeier1(
            0.738042712,
            0.00339065607,
            0.363371967,
            0.0117551189,
            0.989296264,
            212.842145
        ),
    ]
    angles = [-27.2712308, 34.16326141, -42.93207009, 1.06311416]
    angles = np.deg2rad(angles)
    lengths = (0, 0, 0)
    cmpnd = CompoundPrism.create(glasses, angles, lengths, curvature=0.21, height=2.5, width=2.)
    bins = np.array([(i + 0.1, i + 0.9) for i in range(32)]) / 10
    print(bins)
    nbin = 32
    pmt_length = 3.2
    bounds = np.array([i / nbin * pmt_length for i in range(nbin + 1)])
    bins = np.array((bounds[:-1], bounds[1:])).T
    print(bins)
    # bins = tuple(map(tuple, bins))
    detarr = DetectorArray.create(bins, min_ci=cos(np.deg2rad(60)), angle=0., length=3.2)
    beam = GaussianBeam(width=0.2, y_mean=0.95, wavelength_range=(0.5, 0.82))
    print(nb.typeof(cmpnd))
    print(nb.typeof(detarr))
    print(fitness(cmpnd, detarr, beam))


def test2():
    glasses = [
        sellmeier1(
            1.1273555,
            0.00720341707,
            0.124412303,
            0.0269835916,
            0.827100531,
            100.384588
        ),
        sellmeier1(
            1.42288601,
            0.00670283452,
            0.593661336,
            0.021941621,
            1.1613526,
            80.7407701
        ),
        sellmeier1(
            1.69022361,
            0.0130512113,
            0.288870052,
            0.061369188,
            1.7045187,
            149.517689
        ),
        sellmeier1(
            1.23697554,
            0.00747170505,
            0.153569376,
            0.0308053556,
            0.903976272,
            70.1731084
        ),
    ]
    angles = np.array([
        -0.23168624866497564,
        -0.15270363395467454,
        1.4031935147895553,
        -0.1992013814412539,
        -0.273581994447271
    ])
    lengths = np.array([
        0.0,
        0.0001156759279861445,
        0.6326213941932529,
        0.014635982099371865
    ])
    cmpnd = CompoundPrism.create(
        glasses,
        angles,
        lengths,
        curvature=0.042844548844556525,
        height=7.853605325124353,
        width=7.)
    bins = np.array([(i + 0.1, i + 0.9) for i in range(32)])
    detarr = DetectorArray.create(bins, min_ci=cos(np.deg2rad(90)), angle=0.8032206191368956, length=32.)
    beam = GaussianBeam(width=3.2, y_mean=6.687028954574839, wavelength_range=(0.5, 0.82))
    print(nb.typeof(cmpnd))
    print(nb.typeof(detarr))
    print(fitness(cmpnd, detarr, beam))


if __name__ == "__main__":
    test2()
