from __future__ import annotations
import numpy as np
from math import sqrt, cos, sin, tan
from numpy.linalg import inv
from scipy import stats
from typing import NamedTuple, Tuple, Sequence, Callable
import numba as nb
import numba.cuda


class Vector(NamedTuple):
    x: float
    y: float


@nb.njit
def vneg(v: Vector) -> Vector:
    return Vector(-v.x, -v.y)


@nb.njit
def vadd(lhs: Vector, rhs: Vector) -> Vector:
    return Vector(lhs.x + rhs.x, lhs.y + rhs.y)


@nb.njit
def vsub(lhs: Vector, rhs: Vector) -> Vector:
    return Vector(lhs.x - rhs.x, lhs.y - rhs.y)


@nb.njit
def vmul(lhs: Vector, rhs: float) -> Vector:
    return Vector(lhs.x * rhs, lhs.y * rhs)


@nb.njit
def vdiv(lhs: Vector, rhs: float) -> Vector:
    return Vector(lhs.x / rhs, lhs.y / rhs)


@nb.njit
def dot(lhs: Vector, rhs: Vector) -> float:
    return lhs.x * rhs.x + lhs.y * rhs.y


@nb.njit
def norm_squared(v: Vector) -> float:
    return dot(v, v)


@nb.njit
def norm(v: Vector) -> float:
    return sqrt(norm_squared(v))


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


def quasi(seed, dim):
    phi = (1.6180339887498948482, 1.32471795724474602596, 1.22074408460575947536)
    g = phi[dim]
    alpha = g ** -(1 + np.arange(dim))
    z = (seed + alpha) % 1
    while True:
        yield z
        z = (z + alpha) % 1


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


def first_surface(angle, height):
    normal = rotate(angle, (-1, 0))
    midpt = Vector(abs(tan(angle)) * height / 2, height / 2)
    return Surface(angle=angle, normal=normal, midpt=midpt)


def next_surface(prev, angle, height, separation_length):
    normal = rotate(angle, (-1, 0))
    dx1 = abs(tan(prev.angle)) * height / 2
    dx2 = abs(tan(angle)) * height / 2
    separation_distance = separation_length + \
                          (dx1 + dx2 if (prev.normal[1] >= 0) != (normal[1] >= 0) else abs(dx1 - dx2))
    return Surface(angle=angle, normal=normal, midpt=vadd(prev.midpt, Vector(separation_distance, 0)))


class CurvedSurface(NamedTuple):
    midpt: Vector
    center: Vector
    radius: float
    max_dist_sq: float


def from_chord(curvature: float, height: float, chord: Surface) -> CurvedSurface:
    chord_length = height / cos(chord.angle)
    radius = chord_length / 2 / curvature
    apothem = sqrt(radius*radius - chord_length*chord_length / 4)
    sagitta = radius - apothem
    center = vadd(chord.midpt, vmul(chord.normal, apothem))
    midpt = vsub(chord.midpt, vmul(chord.normal, sagitta))
    return CurvedSurface(midpt=midpt, center=center, radius=radius,
                         max_dist_sq=sagitta*sagitta + chord_length*chord_length / 4)


@nb.njit
def is_along_arc(csurf: CurvedSurface, pt: Vector) -> bool:
    diff = vsub(pt, csurf.midpt)
    return norm_squared(diff) <= csurf.max_dist_sq


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


def create_compound_prism(glasses, angles, lengths, curvature, height, width):
    prisms = []
    last_surface = first_surface(angles[0], height)
    for glass, angle, l in zip(glasses, angles[1::], lengths):
        next = next_surface(last_surface, angle, height, l)
        prisms.append((glass, last_surface))
        last_surface = next
    lens = from_chord(curvature, height, last_surface)
    return CompoundPrism(prisms=tuple(prisms), lens=lens, height=height, width=width)


class DetectorArray(NamedTuple):
    bins: np.ndarray
    min_ci: float
    angle: float
    length: float
    normal: Vector


def create_detector_array(bins, min_ci, angle, length):
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


class Ray(NamedTuple):
    origin: Vector
    direction: Vector
    s_transmittance: float
    p_transmittance: float


@nb.njit
def new_ray_from_start(y):
    return Ray(
        origin=Vector(0., y),
        direction=Vector(1., 0.),
        s_transmittance=1.,
        p_transmittance=1.
    )


@nb.njit
def average_transmittance(ray):
    return (ray.s_transmittance + ray.p_transmittance) / 2


@nb.njit
def refract(ray: Ray, intersection, normal, ci, n1, n2):
    r = n1 / n2
    cr_sq = 1 - r*r * (1 - ci*ci)
    if cr_sq < 0:
        return None
        # raise RayTraceError("TotalInternalReflection")
    cr = sqrt(cr_sq)
    v = vadd(vmul(ray.direction, r), vmul(normal, (r * ci - cr)))
    s_reflection_sqrt = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))
    p_reflection_sqrt = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))
    s_transmittance = 1 - s_reflection_sqrt * s_reflection_sqrt
    p_transmittance = 1 - p_reflection_sqrt * p_reflection_sqrt
    return Ray(
        origin=intersection,
        direction=v,
        s_transmittance=ray.s_transmittance * s_transmittance,
        p_transmittance=ray.p_transmittance * p_transmittance,
    )


@nb.njit
def intersect_plane_interface(ray, plane: Surface, n1, n2, prism_height):
    ci = -dot(ray.direction, plane.normal)
    if ci <= 0:
        return None
        # raise RayTraceError("OutOfBounds")
    d = dot(vsub(ray.origin, plane.midpt), plane.normal) / ci
    p = vadd(ray.origin, vmul(ray.direction, d))
    if p.y <= 0 or prism_height <= p.y:
        return None
        # raise RayTraceError("OutOfBounds")
    return refract(ray, p, plane.normal, ci, n1, n2)


@nb.njit
def intersect_curved_interface(ray: Ray, lens: CurvedSurface, n1, n2):
    delta = vsub(ray.origin, lens.center)
    ud = dot(ray.direction, delta)
    under = ud*ud - norm_squared(delta) + lens.radius*lens.radius
    if under < 0:
        return None
        # raise RayTraceError("NoSurfaceIntersection")
    d = -ud + sqrt(under)
    if d <= 0:
        return None
        # raise RayTraceError("NoSurfaceIntersection")
    p = vadd(ray.origin, vmul(ray.direction, d))
    if not is_along_arc(lens, p):
        return None
        # raise RayTraceError("NoSurfaceIntersection")
    snorm = vdiv(vsub(lens.center, p), lens.radius)
    return refract(ray, p, snorm, -dot(ray.direction, snorm), n1, n2)


@nb.njit
def intersect_detector_array(ray: Ray, detarr: DetectorArray, detpos: DetectorArrayPositioning):
    ci = -dot(ray.direction, detarr.normal)
    if ci <= detarr.min_ci:
        return None
        # raise RayTraceError("SpectrometerAngularResponseTooWeak")
    d = dot(vsub(ray.origin, detpos.position), detarr.normal) / ci
    p = vadd(ray.origin, vmul(ray.direction, d))
    pos = dot(vsub(p, detpos.position), detpos.direction)
    if pos < 0 or detarr.length < pos:
        return None
        # raise RayTraceError("NoSurfaceIntersection")
    return p, pos, average_transmittance(ray)


@nb.njit
def propagate_internal(ray: Ray, cmpnd: CompoundPrism, wavelength):
    n1 = 1.
    for glass, plane in cmpnd.prisms:
        n2 = refractive_index(glass, wavelength)
        ray = intersect_plane_interface(ray, plane, n1, n2, cmpnd.height)
        if ray is None:
            return None
        n1 = n2
    n2 = 1.
    return intersect_curved_interface(ray, cmpnd.lens, n1, n2)


@nb.njit
def propagate(ray: Ray, wavelength, cmpnd: CompoundPrism, detarr: DetectorArray, detpos: DetectorArrayPositioning):
    internal = propagate_internal(ray, cmpnd, wavelength)
    if internal is None:
        return None
    else:
        return intersect_detector_array(internal, detarr, detpos)


def detector_array_positioning(cmpnd: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam):
    ray = new_ray_from_start(beam.y_mean)
    wmin, wmax = beam.wavelength_range
    lower_ray = propagate_internal(ray, cmpnd, wmin)
    upper_ray = propagate_internal(ray, cmpnd, wmax)
    if lower_ray is None or upper_ray is None or average_transmittance(lower_ray) <= 1e-3 or average_transmittance(upper_ray) <= 1e-3:
        raise RayTraceError
    spec_dir = rotate(detarr.angle, (0, 1))
    spec = vmul(spec_dir, detarr.length)
    mat = np.array((upper_ray.direction, vneg(lower_ray.direction))).T
    imat = inv(mat)
    dists = imat @ vadd(vsub(spec, upper_ray.origin), lower_ray.origin)
    d2 = dists[1]
    l_vertex = vadd(lower_ray.origin, vmul(lower_ray.direction, d2))
    if d2 > 0:
        return DetectorArrayPositioning(l_vertex, spec_dir)
    dists = imat @ vadd(vsub(vneg(spec), upper_ray.origin), lower_ray.origin)
    d2 = dists[1]
    if d2 < 0:
        raise RayTraceError
    u_vertex = vadd(lower_ray.origin, vmul(lower_ray.direction, d2))
    return DetectorArrayPositioning(u_vertex, vneg(spec_dir))


@nb.vectorize
def plog2p(p):
    if p == 0.:
        return 0.
    else:
        return p * np.log2(p)


@nb.njit(nogil=True, error_model='numpy')
def vectorized_propagate(cmpnd, detarr, detpos, w, y, positions, transmissions):
    for i in range(len(w)):
        for j in range(len(y)):
            ray = new_ray_from_start(y[j])
            r = propagate(ray, w[i], cmpnd, detarr, detpos)
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
            ray = new_ray_from_start(y[j])
            r = propagate(ray, w[i], cmpnd, detarr, detpos)
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
    deviation_vector = vsub(vadd(detpos.position, vmul(detpos.direction, detarr.length / 2)), Vector(0., beam.y_mean))
    size = np.linalg.norm(deviation_vector)
    deviation = abs(deviation_vector[1]) / np.linalg.norm(deviation_vector)
    print(size, deviation)

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

    ws = np.fromiter(map(L.ppf, quasi(0., 1)), np.float64, 8192)
    ys = np.fromiter(map(Y.ppf, quasi(0., 1)), np.float64, 8192)
    # info = _mutual_info(cmpnd, detarr, detpos, ws, ys, pZ)
    # print('info', info)

    @nb.cuda.jit((nb.f8[:], nb.f8[:], nb.f8[:, ::], nb.f8[:, ::]))
    def cuda_propagate(ws, ys, positions, transmissions):
        i, j = nb.cuda.grid(2)
        w = ws[i]
        y = ys[j]
        ray = new_ray_from_start(y)
        r = propagate(ray, w, cmpnd, detarr, detpos)
        if r is None:
            pos, t = np.inf, 0.
        else:
            _, pos, t = r
        positions[i, j] = pos
        transmissions[i, j] = t

    @nb.cuda.jit((nb.f8[:], nb.f8[:], nb.f8[:, ::]))
    def cuda_p_det_l_w(ws, ys, prob):
        i = nb.cuda.grid(1)
        w = ws[i]
        prob[i, :] = 0.
        for j in range(prob.shape[1]):
            prob[i, j] = 0.
        count = 1.
        for y in ys:
            ray = new_ray_from_start(y)
            r = propagate(ray, w, cmpnd, detarr, detpos)
            if r is None:
                for j in range(prob.shape[1]):
                    prob[i, j] -= prob[i, j] / count
            else:
                _, pos, t = r
                for j in range(prob.shape[1]):
                    if detarr.bins[j, 0] <= pos < detarr.bins[j, 1]:
                        prob[i, j] += (t - prob[i, j]) / count
                    else:
                        prob[i, j] -= prob[i, j] / count
            count += 1.

    with open('test2.ptx', 'w') as f:
        f.write(cuda_p_det_l_w.ptx)
    # print(ws, ys)
    pos = np.empty((len(ws), len(ys)), dtype=np.float64)
    t = np.empty((len(ws), len(ys)), dtype=np.float64)
    assert 1024 * 8 == 8192
    cuda_propagate[(1024, 1024), (8, 8)](ws, ys, pos, t)
    info = np.array([
        mutual_info(pos, t, lb, ub)
        for lb, ub in detarr.bins
    ])
    print(info)
    print(np.sum(info))

    p_det_l_w = np.empty((len(ws), len(detarr.bins)), dtype=np.float64)
    cuda_p_det_l_w[8192 // 64, 64](ws, ys, p_det_l_w)
    p_det = np.mean(p_det_l_w, axis=1)
    i = np.mean(plog2p(p_det_l_w)) - np.mean(plog2p(p_det))
    print(i)

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
    '''
    print(timeit(lambda: (propagate(cmpnd, detarr, detpos, ws, ys, pos, t), sum(mutual_info(pos, t, lb, ub) for lb, ub in detarr.bins)), number=n) / n)
    print(timeit(lambda: propagate(cmpnd, detarr, detpos, ws, ys, pos, t), number=n) / n)
    print(timeit(lambda: sum(mutual_info(pos, t, lb, ub) for lb, ub in detarr.bins), number=n) / n)
    '''
    print(timeit(lambda: cuda_propagate[(1024, 1024), (8, 8)](ws, ys, pos, t), number=n) / n)
    print(timeit(lambda: (cuda_propagate[(1024, 1024), (8, 8)](ws, ys, pos, t), sum(mutual_info(pos, t, lb, ub)
                                                                                     for lb, ub in detarr.bins)), number=n) / n)
    n = 1
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
            1.46141885,
            0.0111826126,
            0.247713019,
            0.0508594669,
            0.949995832,
            112.041888
        ),
        sellmeier1(
            0.991463823,
            0.00522730467,
            0.495982121,
            0.0172733646,
            0.987393925,
            98.3594579
        ),
    ]
    angles = np.array([
        -0.14526140641191904,
        1.4195826230834718,
        -0.28592196648027723,
        -1.0246624248138436,
        -0.19580534765720792
    ])
    lengths = np.array([
        1.7352128646437137,
        0.2979322924387521,
        1.3590246624699227,
        0.02257389234062425
    ])
    cmpnd = create_compound_prism(
        glasses,
        angles,
        lengths,
        curvature=0.07702713427258588,
        height=13.507575346183698,
        width=7.)
    bins = np.array([(i + 0.1, i + 0.9) for i in range(32)])
    detarr = create_detector_array(
        bins,
        min_ci=cos(np.deg2rad(45)),
        angle=0.9379755338734997,
        length=32.)
    beam = GaussianBeam(width=3.2, y_mean=9.103647605649911, wavelength_range=(0.5, 0.82))
    print(nb.typeof(cmpnd))
    print(nb.typeof(detarr))
    print(fitness(cmpnd, detarr, beam))

'''
Design {
    compound_prism: CompoundPrismDesign {
        glasses: [
            (
                "N-LAK33B",
                Sellmeier1(
                    [
                        1.42288601,
                        0.00670283452,
                        0.593661336,
                        0.021941621,
                        1.1613526,
                        80.7407701,
                    ],
                ),
            ),
            (
                "N-SF14",
                Sellmeier1(
                    [
                        1.69022361,
                        0.0130512113,
                        0.288870052,
                        0.061369188,
                        1.7045187,
                        149.517689,
                    ],
                ),
            ),
            (
                "SF5",
                Sellmeier1(
                    [
                        1.46141885,
                        0.0111826126,
                        0.247713019,
                        0.0508594669,
                        0.949995832,
                        112.041888,
                    ],
                ),
            ),
            (
                "N-SK5",
                Sellmeier1(
                    [
                        0.991463823,
                        0.00522730467,
                        0.495982121,
                        0.0172733646,
                        0.987393925,
                        98.3594579,
                    ],
                ),
            ),
        ],
        angles: [
            -0.14526140641191904,
            1.4195826230834718,
            -0.28592196648027723,
            -1.0246624248138436,
            -0.19580534765720792,
        ],
        lengths: [
            1.7352128646437137,
            0.2979322924387521,
            1.3590246624699227,
            0.02257389234062425,
        ],
        curvature: 0.07702713427258588,
        height: 13.507575346183698,
        width: 7.0,
    },
    detector_array: DetectorArrayDesign {
        bins: [
            [
                0.1,
                0.9,
            ],
            [
                1.1,
                1.9,
            ],
            [
                2.1,
                2.9,
            ],
            [
                3.1,
                3.9,
            ],
            [
                4.1,
                4.9,
            ],
            [
                5.1,
                5.9,
            ],
            [
                6.1,
                6.9,
            ],
            [
                7.1,
                7.9,
            ],
            [
                8.1,
                8.9,
            ],
            [
                9.1,
                9.9,
            ],
            [
                10.1,
                10.9,
            ],
            [
                11.1,
                11.9,
            ],
            [
                12.1,
                12.9,
            ],
            [
                13.1,
                13.9,
            ],
            [
                14.1,
                14.9,
            ],
            [
                15.1,
                15.9,
            ],
            [
                16.1,
                16.9,
            ],
            [
                17.1,
                17.9,
            ],
            [
                18.1,
                18.9,
            ],
            [
                19.1,
                19.9,
            ],
            [
                20.1,
                20.9,
            ],
            [
                21.1,
                21.9,
            ],
            [
                22.1,
                22.9,
            ],
            [
                23.1,
                23.9,
            ],
            [
                24.1,
                24.9,
            ],
            [
                25.1,
                25.9,
            ],
            [
                26.1,
                26.9,
            ],
            [
                27.1,
                27.9,
            ],
            [
                28.1,
                28.9,
            ],
            [
                29.1,
                29.9,
            ],
            [
                30.1,
                30.9,
            ],
            [
                31.1,
                31.9,
            ],
        ],
        position: Pair {
            x: 211.34017386443577,
            y: 41.09153119458691,
        },
        direction: Pair {
            x: 0.8063624404692499,
            y: -0.5914216893219892,
        },
        length: 32.0,
        max_incident_angle: 45.0,
        angle: 0.9379755338734997,
    },
    gaussian_beam: GaussianBeamDesign {
        wavelength_range: (
            0.5,
            0.82,
        ),
        width: 3.2,
        y_mean: 9.103647605649911,
    },
    fitness: DesignFitness {
        size: 225.37045989321203,
        info: 3.8222556062078317,
        deviation: 0.09994715620875215,
    },
}
'''

if __name__ == "__main__":
    test2()
