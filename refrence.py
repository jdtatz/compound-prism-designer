import numpy as np
import sympy
import sympy.physics.optics
from sympy import symbols, assuming, sin, cos, tan, sqrt, Dummy, Function, And, Xor, Piecewise, \
    sympify, simplify, Lambda, MatrixSymbol
from sympy.core.basic import Basic
from sympy.matrices import Matrix
from sympy.stats import Uniform, Normal, density, cdf, E, quantile

x = symbols("x", positive=True)
print(sqrt(-abs(x)))
x_hat = Matrix((1, 0))
y_hat = Matrix((0, 1))


class Ray(Basic):
    __slots__ = Basic.__slots__ + ["origin", "direction", "s_transmittance", "p_transmittance"]

    def __new__(cls, origin, direction, s_transmittance, p_transmittance):
        args = [sympify(a) for a in (origin, direction, s_transmittance, p_transmittance)]
        obj = Basic.__new__(cls, *args)
        obj.origin, obj.direction, obj.s_transmittance, obj.p_transmittance = obj.args
        return obj

    @staticmethod
    def new_from_start(y):
        return Ray(
            origin=y*y_hat,
            direction=x_hat,
            s_transmittance=1,
            p_transmittance=1
        )

    @property
    def transmittance(self):
        return (self.s_transmittance + self.p_transmittance) / 2


origin = MatrixSymbol(Dummy('origin'), 2, 1)
direction = MatrixSymbol('direction', 2, 1)
s_transmittance = Dummy('s_transmittance')
p_transmittance = Dummy('p_transmittance')
ray = Ray(origin=origin, direction=direction, s_transmittance=s_transmittance, p_transmittance=p_transmittance)
intersection = MatrixSymbol('intersection', 2, 1)
normal = MatrixSymbol('normal', 2, 1)
ci = Dummy('ci')
n1 = Dummy('n1')
n2 = Dummy('n2')
r = n1 / n2
cr = sqrt(1 - r**2 * (1 - ci**2))
v = ray.direction * r + normal * (r * ci - cr)
s_transmittance = 1 - ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))**2
p_transmittance = 1 - ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))**2
refract = Lambda((ray.args, intersection, normal, ci, n1, n2),
                 Ray(
                     origin=intersection,
                     direction=v,
                     s_transmittance=ray.s_transmittance * s_transmittance,
                     p_transmittance=ray.p_transmittance * p_transmittance
                 )
                 )
print(refract)
print(simplify(refract))
print(sympy.cse(refract))
print(refract(Ray.new_from_start(7).args, intersection, normal, ci, n1, n2))

y0 = Dummy('y0')
init = Ray.new_from_start(y0)
print(init)
N = 3
refractive_indicies = [Dummy(f'n{1 + i}') for i in range(N)]

ray = init
n1 = 1
for n2 in refractive_indicies:
    ix = Dummy('ix')
    iy = Dummy('iy')
    t = Dummy('t')
    c, s = cos(t), sin(t)
    r = Matrix(((c, -s), (s, c)))
    n = r * Matrix([-1, 0])
    ray = refract(ray.args, Matrix([ix, iy]), n, Dummy('ci'), n1, n2)
    n1 = n2
propagate_internal = Lambda(
    (y0, tuple(refractive_indicies)),
    ray
)
print(propagate_internal)
print(simplify(propagate_internal))
print(sympy.cse(propagate_internal))


def sellmeier1(b1, c1, b2, c2, b3, c3):
    b1, c1, b2, c2, b3, c3 = (sympify(x) for x in (b1, c1, b2, c2, b3, c3))

    def _sellmeier1(w):
        w2 = w**2
        return sqrt(1 + b1 * w2 / (w2 - c1) + b2 * w2 / (w2 - c2) + b3 * w2 / (w2 - c3))
    return _sellmeier1


def rotate(angle, vector):
    c, s = cos(angle), sin(angle)
    r = Matrix(((c, -s), (s, c)))
    return r * vector


class RayTraceError(Exception):
    pass


class Surface:
    """"""
    def __init__(self, angle, normal, midpt):
        self.angle = angle
        self.normal = normal
        self.midpt = midpt

    @staticmethod
    def first_surface(angle, height):
        normal = rotate(angle, -x_hat)
        midpt = abs(tan(angle)) * height / 2 * x_hat + height / 2 * y_hat
        return Surface(angle=angle, normal=normal, midpt=midpt)

    def next_surface(self, angle, height, separation_length):
        normal = rotate(angle, -x_hat)
        dx1 = abs(tan(self.angle)) * height / 2
        dx2 = abs(tan(angle)) * height / 2
        separation_distance = separation_length + Piecewise(
            (dx1 + dx2, Xor((self.normal[1] >= 0), (normal[1] >= 0))),
            (abs(dx1 - dx2), True)
        )
        return Surface(angle=angle, normal=normal, midpt=self.midpt + separation_distance * x_hat)


class CurvedSurface:
    def __init__(self, curvature, height, chord: Surface):
        chord_length = height / cos(chord.angle)
        radius = chord_length / 2 / curvature
        apothem = sqrt(radius**2 - chord_length**2 / 4)
        sagitta = radius - apothem
        center = chord.midpt + chord.normal * apothem
        midpt = chord.midpt - chord.normal * sagitta
        self.midpt = midpt
        self.center = center
        self.radius = radius
        self.max_dist_sq = sagitta**2 + chord_length**2 / 4

    def is_along_arc(self, pt):
        return (pt - self.midpt).norm()**2 <= self.max_dist_sq


class GaussianBeam:
    """Collimated Polychromatic Gaussian Beam

    Attributes:
        width (float): 1/e^2 beam width
        y_mean (float): Mean y coordinate
        wavelength_range ((float, float)): Range of wavelengths
    """
    def __init__(self, width, y_mean, wavelength_range):
        self.width = width
        self.y_mean = y_mean
        self.wavelength_range = wavelength_range


class CompoundPrism:
    def __init__(self, glasses, angles, lengths, curvature, height, width):
        prisms = []
        last_surface = Surface.first_surface(angles[0], height)
        for glass, angle, l in zip(glasses, angles[1::], lengths):
            next = last_surface.next_surface(angle, height, l)
            prisms.append((glass, last_surface))
            last_surface = next
        lens = CurvedSurface(curvature, height, last_surface)
        self.prisms = prisms
        self.lens = lens
        self.height = height
        self.width = width


class DetectorArray:
    def __init__(self, bins, min_ci, angle, length):
        self.bins = bins
        self.min_ci = min_ci
        self.angle = angle
        self.length = length
        self.normal = rotate(angle, -x_hat)


class DetectorArrayPositioning:
    def __init__(self, position, direction):
        self.position = position
        self.direction = direction


class Ray(Basic):
    __slots__ = Basic.__slots__ + ["origin", "direction", "s_transmittance", "p_transmittance"]

    def __new__(cls, origin, direction, s_transmittance, p_transmittance):
        args = [sympify(a) for a in (origin, direction, s_transmittance, p_transmittance)]
        obj = Basic.__new__(cls, *args)
        obj.origin, obj.direction, obj.s_transmittance, obj.p_transmittance = obj.args
        return obj

    @staticmethod
    def new_from_start(y):
        return Ray(
            origin=y*y_hat,
            direction=x_hat,
            s_transmittance=1,
            p_transmittance=1
        )

    def average_transmittance(self):
        return (self.s_transmittance + self.p_transmittance) / 2

    def refract(self, intersection, normal, ci, n1, n2):
        r = n1 / n2
        cr_sq = 1 - r**2 * (1 - ci**2)
        #if cr_sq < 0:
        #    raise RayTraceError("TotalInternalReflection")
        cr = sqrt(cr_sq)
        v = self.direction * r + normal * (r * ci - cr)
        s_transmittance = 1 - ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr))**2
        p_transmittance = 1 - ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci))**2
        return Ray(
            origin=intersection,
            direction=v,
            s_transmittance=self.s_transmittance * s_transmittance,
            p_transmittance=self.p_transmittance * p_transmittance,
        )

    def intersect_plane_interface(self, plane: Surface, n1, n2, prism_height):
        ci = -self.direction.dot(plane.normal)
        #if ci <= 0:
        #    raise RayTraceError("OutOfBounds")
        d = (self.origin - plane.midpt).dot(plane.normal) / ci
        p = self.origin + self.direction * d
        #if p[1] <= 0 | prism_height <= p[1]:
        #    raise RayTraceError("OutOfBounds")
        # i, = intersection(self.ray, plane.plane)
        # print(p, i.evalf())
        return self.refract(p, plane.normal, ci, n1, n2)

    def intersect_curved_interface(self, lens: CurvedSurface, n1, n2):
        delta = self.origin - lens.center
        ud = self.direction.dot(delta)
        under = ud**2 - delta.norm()**2 + lens.radius**2
        #if under < 0:
        #    raise RayTraceError("NoSurfaceIntersection")
        d = -ud + sqrt(under)
        #if d <= 0:
        #    raise RayTraceError("NoSurfaceIntersection")
        p = self.origin + self.direction * d
        #if not lens.is_along_arc(p):
        #    raise RayTraceError("NoSurfaceIntersection")
        snorm = (lens.center - p) / lens.radius
        return self.refract(p, snorm, -self.direction.dot(snorm), n1, n2)

    def intersect_detector_array(self, detarr: DetectorArray, detpos: DetectorArrayPositioning):
        ci = -self.direction.dot(detarr.normal)
        #if ci <= detarr.min_ci:
        #    raise RayTraceError("SpectrometerAngularResponseTooWeak")
        d = (self.origin - detpos.position).dot(detarr.normal) / ci
        p = self.origin + self.direction * d
        pos = (p - detpos.position).dot(detpos.direction)
        #if pos < 0 | detarr.length < pos:
        #    raise RayTraceError("NoSurfaceIntersection")
        return p, pos, self.average_transmittance()

    def propagate_internal(self, cmpnd: CompoundPrism, wavelength):
        ray, n1 = self, 1
        for glass, plane in cmpnd.prisms:
            n2 = glass(wavelength)
            ray = ray.intersect_plane_interface(plane, n1, n2, cmpnd.height)
            n1 = n2
        n2 = 1
        return ray.intersect_curved_interface(cmpnd.lens, n1, n2)

    def propagate(self, wavelength, cmpnd: CompoundPrism, detarr: DetectorArray, detpos: DetectorArrayPositioning):
        return self.propagate_internal(cmpnd, wavelength).intersect_detector_array(detarr, detpos)


def detector_array_positioning(cmpnd: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam):
    ray = Ray.new_from_start(beam.y_mean)
    wmin, wmax = beam.wavelength_range
    lower_ray = ray.propagate_internal(cmpnd, wmin)
    upper_ray = ray.propagate_internal(cmpnd, wmax)
    #if lower_ray.average_transmittance() <= 1e-3 or upper_ray.average_transmittance() <= 1e-3:
    #    raise RayTraceError("SpectrometerAngularResponseTooWeak")
    spec_dir = rotate(detarr.angle, y_hat)
    spec = spec_dir * detarr.length
    mat = upper_ray.direction.row_join(-lower_ray.direction)
    # print(mat.det())
    imat = mat.inv()
    dists = imat * (spec - upper_ray.origin + lower_ray.origin)
    d2 = dists[1]
    l_vertex = lower_ray.origin + lower_ray.direction * d2
    if d2 > 0:
        return DetectorArrayPositioning(l_vertex, spec_dir)
    dists = imat * (-spec - upper_ray.origin + lower_ray.origin)
    d2 = dists[1]
    if d2 < 0:
        raise RayTraceError("NoSurfaceIntersection")
    u_vertex = lower_ray.origin + lower_ray.direction * d2
    return DetectorArrayPositioning(u_vertex, -spec_dir)


def fitness(cmpnd: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam):
    detpos = detector_array_positioning(cmpnd, detarr, beam)
    # print(detpos.__dict__)
    deviation_vector = detpos.position + detpos.direction * detarr.length / 2 - beam.y_mean * y_hat
    size = deviation_vector.norm()
    deviation = abs(deviation_vector[1]) / deviation_vector.norm()
    Y = Normal("Y", beam.y_mean, beam.width)
    invY = quantile(Y)
    Z = Normal("Z", 0, beam.width)
    pZ = sympy.stats.P(And(cmpnd.width / 2 > Z, Z > -cmpnd.width / 2))
    B = Normal("B", [beam.y_mean, 0], [[beam.width, 0], [0, beam.width]])
    L = Uniform("Λ", *beam.wavelength_range)
    y, w = symbols("y λ", positive=True)
    # print(quantile(L))
    # print(sympy.stats.P(And(cmpnd.width / 2 > Z, Z > -cmpnd.width / 2)))
    # print(Ray.new_from_start(Y).__dict__)
    _, pos, t = Ray.new_from_start(Y).propagate(w, cmpnd, detarr, detpos)
    # print("pos", pos)
    # print(t)
    # print(detarr.bins[0][0] <= pos, pos < detarr.bins[0][1])
    # print(And(detarr.bins[0][0] <= pos, pos < detarr.bins[0][1]))
    p_det_l_w = E(Piecewise((t, And(detarr.bins[0][0] <= pos, pos < detarr.bins[0][1])), (0, True)))
    return size, deviation, p_det_l_w


def test():
    prism_count = 3
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
    # glasses = symbols(f"g:{prism_count}", cls=Function, real=True, positive=True)
    angles = [-27.2712308, 34.16326141, -42.93207009, 1.06311416]
    angles = sympify(np.deg2rad(angles))
    lengths = sympify((0, 0, 0))
    cmpnd = CompoundPrism(glasses, angles, lengths, curvature=sympify(0.21), height=sympify(2.5), width=sympify(2))
    bins = sympify(np.array([(i + 0.1, i + 0.9) for i in range(32)]) / 10)
    detarr = DetectorArray(bins, min_ci=cos(sympy.rad(60)), angle=sympify(0), length=sympify(3.2))
    beam = GaussianBeam(width=sympify(0.2), y_mean=sympify(0.95), wavelength_range=(0.5, 0.82))
    print(fitness(cmpnd, detarr, beam))


def main(prism_count: int, bins):
    glasses = symbols(f"g:{prism_count}", cls=Function, real=True, positive=True)
    angles = symbols(f"θ:{1 + prism_count}", real=True)
    lengths = symbols(f"l:{prism_count}", positive=True)
    k, h, w = symbols("k h w", positive=True)
    cmpnd = CompoundPrism(
        glasses=glasses,
        angles=angles,
        lengths=lengths,
        curvature=k,
        height=h,
        width=w
    )
    alpha = symbols("α", real=True)
    ld, gamma = symbols("ld γ", positive=True)

    detarr = DetectorArray(
        bins=bins,
        angle=alpha,
        min_ci=cos(gamma),
        length=ld
    )

    sigma, ym, lmin, lmax = symbols("σ ym λmin λmax", positive=True)

    beam = GaussianBeam(
        width=sigma,
        y_mean=ym,
        wavelength_range=(lmin, lmax)
    )
    pi_2 = sympy.pi / 2
    with assuming(
            *(-pi_2 < t for t in angles),
            *(t < pi_2 for t in angles),
            0 < k,
            k <= 1,
            -pi_2 < alpha,
            alpha < pi_2,
            gamma <= pi_2,
            ym < h,
            lmin < lmax
    ):
        return fitness(cmpnd, detarr, beam)


if __name__ == "__main__":
    test()
    '''
    test_prism_count = 2
    test_bins = [[i + 0.1, i + 0.9] for i in range(32)]
    print(main(test_prism_count, test_bins))
    '''
