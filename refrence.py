import sympy
import sympy.physics.optics
from sympy import symbols, assuming, sin, cos, tan, sqrt, integrate, Function, And, Xor, Piecewise
from sympy.matrices import Matrix
from sympy.stats import Uniform, Normal, density, cdf

x_hat = Matrix((1, 0))
y_hat = Matrix((0, 1))


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
        self.plane = sympy.Plane(sympy.Point3D(*midpt, 0), normal_vector=(*normal, 0))

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
            (dx1 - dx2, abs(self.normal[1]) > abs(normal[1])),
            (dx2 - dx1, True)
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


class Ray:
    def __init__(self, origin, direction, s_transmittance, p_transmittance):
        self.origin = origin
        self.direction = direction
        self.s_transmittance = s_transmittance
        self.p_transmittance = p_transmittance
        p0 = sympy.Point3D(*self.origin, 0)
        v = sympy.Point3D(*self.direction, 0)
        self.ray = sympy.Ray(p0, p0 + v)

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
    print(mat.det())
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
    px, py, vx, vy = symbols("px, py, vx, vy", real=True)
    detpos = DetectorArrayPositioning(Matrix((px, py)), Matrix((vx, vy)))
    # detpos = detector_array_positioning(cmpnd, detarr, beam)
    deviation_vector = detpos.position + detpos.direction * detarr.length / 2 - beam.y_mean * y_hat
    size = deviation_vector.norm()
    deviation = abs(deviation_vector[1]) / deviation_vector.norm()
    Y = Normal("Y", beam.y_mean, beam.width)
    Z = Normal("Z", 0, beam.width)
    L = Uniform("Λ", *beam.wavelength_range)
    y, w = symbols("y λ", positive=True)
    _, pos, t = Ray.new_from_start(y).propagate(w, cmpnd, detarr, detpos)
    p_det_l_w = integrate(Piecewise((t, And(detarr.bins[0][0] <= pos, pos < detarr.bins[0][1])), (0, True)), (y, 0, cmpnd.height))
    return size, deviation, p_det_l_w


def main(prism_count: int, bins):
    glasses = symbols(f"g:{prism_count}", cls=Function, real=True)
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
    test_prism_count = 2
    test_bins = [[ 0.1,  0.9],
            [ 1.1,  1.9],
            [ 2.1,  2.9],
            [ 3.1,  3.9],
            [ 4.1,  4.9],
            [ 5.1,  5.9],
            [ 6.1,  6.9],
            [ 7.1,  7.9],
            [ 8.1,  8.9],
            [ 9.1,  9.9],
            [10.1, 10.9],
            [11.1, 11.9],
            [12.1, 12.9],
            [13.1, 13.9],
            [14.1, 14.9],
            [15.1, 15.9],
            [16.1, 16.9],
            [17.1, 17.9],
            [18.1, 18.9],
            [19.1, 19.9],
            [20.1, 20.9],
            [21.1, 21.9],
            [22.1, 22.9],
            [23.1, 23.9],
            [24.1, 24.9],
            [25.1, 25.9],
            [26.1, 26.9],
            [27.1, 27.9],
            [28.1, 28.9],
            [29.1, 29.9],
            [30.1, 30.9],
            [31.1, 31.9]]
    print(main(test_prism_count, test_bins))
