use crate::erf::{erf, erfc_inv};
use crate::glasscat::Glass;
use libm::{log2, sincos};
use std::f64::consts::*;
// Can't use libm::sqrt till https://github.com/rust-lang/libm/pull/222 is merged

#[derive(Debug, Display, Clone, Copy)]
pub enum RayTraceError {
    NoSurfaceIntersection,
    OutOfBounds,
    TotalInternalReflection,
    SpectrometerAngularResponseTooWeak,
}

impl Into<&'static str> for RayTraceError {
    fn into(self) -> &'static str {
        match self {
            RayTraceError::NoSurfaceIntersection => "NoSurfaceIntersection",
            RayTraceError::OutOfBounds => "OutOfBounds",
            RayTraceError::TotalInternalReflection => "TotalInternalReflection",
            RayTraceError::SpectrometerAngularResponseTooWeak => {
                "SpectrometerAngularResponseTooWeak"
            }
        }
    }
}

/// vector in R^2 represented as a 2-tuple
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy, From, Into, Neg, Add, Sub, Mul, Div)]
pub struct Pair {
    pub x: f64,
    pub y: f64,
}

impl Pair {
    /// dot product of two vectors, a • b
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// square of the vector norm, ||v||^2
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// vector norm, ||v||
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// is it a unit vector, ||v|| ≅? 1
    pub fn is_unit(self) -> bool {
        (self.norm() - 1_f64).abs() < 1e-3
    }
}

/// rotate `vector` by `angle` CCW
#[inline(always)]
fn rotate(angle: f64, vector: Pair) -> Pair {
    let (s, c) = sincos(angle);
    Pair {
        x: c * vector.x - s * vector.y,
        y: s * vector.x + c * vector.y,
    }
}

/// Matrix in R^(2x2)
#[derive(Debug, Clone, Copy)]
struct Mat2([f64; 4]);

impl Mat2 {
    /// New Matrix from the two given columns
    fn new_from_cols(col1: Pair, col2: Pair) -> Self {
        Self([col1.x, col2.x, col1.y, col2.y])
    }

    /// Matrix inverse if it exists
    fn inverse(self) -> Option<Self> {
        let [a, b, c, d] = self.0;
        let det = a * d - b * c;
        if det == 0. {
            None
        } else {
            Some(Self([d / det, -b / det, -c / det, a / det]))
        }
    }
}

impl core::ops::Mul<Pair> for Mat2 {
    type Output = Pair;

    /// Matrix x Vector -> Vector multiplication
    fn mul(self, rhs: Pair) -> Self::Output {
        let [a, b, c, d] = self.0;
        Pair {
            x: a * rhs.x + b * rhs.y,
            y: c * rhs.x + d * rhs.y,
        }
    }
}

/// Collimated Polychromatic Gaussian Beam
#[derive(Debug, Clone)]
pub struct GaussianBeam {
    /// 1/e^2 beam width
    pub width: f64,
    /// Mean y coordinate
    pub y_mean: f64,
    /// Range of wavelengths
    pub w_range: (f64, f64),
}

#[derive(Debug, Clone, Copy)]
pub struct Surface {
    pub normal: Pair,
    pub midpt: Pair,
}

impl Surface {
    fn first_surface(angle: f64, height: f64) -> Self {
        let normal = rotate(angle, (-1_f64, 0_f64).into());
        Self {
            normal,
            midpt: ((normal.y / normal.x).abs() * height * 0.5, height * 0.5).into(),
        }
    }

    fn next_surface(&self, height: f64, new_angle: f64, sep_length: f64) -> Self {
        let normal = rotate(new_angle, (-1_f64, 0_f64).into());
        let d1 = (self.normal.y / self.normal.x).abs() * height * 0.5;
        let d2 = (normal.y / normal.x).abs() * height * 0.5;
        let sep_dist = sep_length
            + if (self.normal.y >= 0.) != (normal.y >= 0.) {
                d1 + d2
            } else if self.normal.y.abs() > normal.y.abs() {
                d1 - d2
            } else {
                d2 - d1
            };
        Self {
            normal,
            midpt: self.midpt + (sep_dist, 0.).into(),
        }
    }

    pub fn end_points(&self, height: f64) -> (Pair, Pair) {
        let dx = self.normal.y / self.normal.x * height * 0.5;
        let ux = self.midpt.x - dx;
        let lx = self.midpt.x + dx;
        ((ux, height).into(), (lx, 0.).into())
    }
}

/// A Curved Surface, parameterized as a circular segment
#[derive(Debug, Clone, Copy)]
pub struct CurvedSurface {
    /// The midpt of the Curved Surface / circular segment
    midpt: Pair,
    /// The center of the circle
    center: Pair,
    /// The radius of the circle
    radius: f64,
    /// max_dist_sq = sagitta ^ 2 + (chord_length / 2) ^ 2
    max_dist_sq: f64,
}

impl CurvedSurface {
    fn new(curvature: f64, height: f64, chord: Surface) -> Self {
        let chord_length = height / chord.normal.x.abs();
        let radius = chord_length * 0.5 / curvature;
        let apothem = (radius * radius - chord_length * chord_length * 0.25).sqrt();
        let sagitta = radius - apothem;
        let center = chord.midpt + chord.normal * apothem;
        let midpt = chord.midpt - chord.normal * sagitta;
        Self {
            midpt,
            center,
            radius,
            max_dist_sq: sagitta * sagitta + chord_length * chord_length * 0.25,
        }
    }

    fn is_along_arc(&self, pt: Pair) -> bool {
        debug_assert!((pt - self.center).norm() < self.radius * 1.01);
        debug_assert!((pt - self.center).norm() > self.radius * 0.99);
        (pt - self.midpt).norm_squared() <= self.max_dist_sq
    }

    fn end_points(&self, height: f64) -> (Pair, Pair) {
        let theta_2 = 2. * (self.max_dist_sq.sqrt() / (2. * self.radius)).asin();
        let r = self.midpt - self.center;
        let u = self.center + rotate(theta_2, r);
        let l = self.center + rotate(-theta_2, r);
        debug_assert!((u.y - height).abs() < 1e-4, "{:?} {}", u, height);
        debug_assert!(l.y.abs() < 1e-4, "{:?}", l);
        debug_assert!(((u - r).norm_squared() - self.max_dist_sq).abs() / self.max_dist_sq < 1e-4);
        debug_assert!(((l - r).norm_squared() - self.max_dist_sq).abs() / self.max_dist_sq < 1e-4);
        ((u.x, height).into(), (l.x, 0.).into())
    }
}

/// Compound Prism Specification
#[derive(Debug, Clone)]
pub struct CompoundPrism<'a> {
    /// List of glasses the compound prism is composed of, in order.
    /// With their inter-media boundary surfaces
    prisms: Vec<(&'a Glass, Surface)>,
    /// The curved lens-like last inter-media boundary surface of the compound prism
    lens: CurvedSurface,
    /// Height of compound prism
    height: f64,
    /// Width of compound prism
    width: f64,
}

impl<'a> CompoundPrism<'a> {
    /// Create a new Compound Prism Specification
    ///
    /// # Arguments
    ///  * `glasses` - List of glasses the compound prism is composed of, in order
    ///  * `angles` - Angles that parameterize the shape of the compound prism
    ///  * `lengths` - Lengths that parameterize the trapezoidal shape of the compound prism
    ///  * `curvature` - Lens Curvature of last surface of compound prism
    ///  * `height` - Height of compound prism
    ///  * `width` - Width of compound prism
    pub fn new<I: IntoIterator<Item = &'a Glass>>(
        glasses: I,
        angles: &[f64],
        lengths: &[f64],
        curvature: f64,
        height: f64,
        width: f64,
    ) -> Self
    where
        I::IntoIter: ExactSizeIterator,
    {
        let glasses = glasses.into_iter();
        debug_assert!(glasses.len() > 0);
        debug_assert!(angles.len() - 1 == glasses.len());
        debug_assert!(lengths.len() == glasses.len());
        let mut prisms = Vec::with_capacity(glasses.len());
        let mut last_surface = Surface::first_surface(angles[0], height);
        for ((g, a), l) in glasses.zip(&angles[1..]).zip(lengths) {
            let next = last_surface.next_surface(height, *a, *l);
            prisms.push((g, last_surface));
            last_surface = next;
        }
        let lens = CurvedSurface::new(curvature, height, last_surface);
        Self {
            prisms,
            lens,
            height,
            width,
        }
    }

    pub fn polygons(&self) -> (Vec<[Pair; 4]>, [Pair; 4], Pair, f64) {
        let mut poly = Vec::with_capacity(self.prisms.len());
        let (mut u0, mut l0) = self.prisms[0].1.end_points(self.height);
        for (_, s) in self.prisms[1..].iter() {
            let (u1, l1) = s.end_points(self.height);
            poly.push([l0, u0, u1, l1]);
            u0 = u1;
            l0 = l1;
        }
        let (u1, l1) = self.lens.end_points(self.height);
        (poly, [l0, u0, u1, l1], self.lens.center, self.lens.radius)
    }
}

/// Array of detectors
#[derive(Debug, Clone)]
pub struct DetectorArray<'a> {
    /// Boundaries of detection bins
    bins: &'a [[f64; 2]],
    /// Minimum cosine of incident angle == cosine of maximum allowed incident angle
    min_ci: f64,
    /// CCW angle of the array from normal = Rot(θ) @ (0, 1)
    angle: f64,
    /// The normal of the array's surface, normal = Rot(θ) @ (-1, 0)
    normal: Pair,
    /// Length of the array
    length: f64,
}

impl<'a> DetectorArray<'a> {
    pub fn new(bins: &'a [[f64; 2]], min_ci: f64, angle: f64, length: f64) -> Self {
        Self {
            bins,
            min_ci,
            angle,
            normal: rotate(angle, (-1_f64, 0_f64).into()),
            length,
        }
    }

    pub fn end_points(&self, pos: &DetectorArrayPositioning) -> (Pair, Pair) {
        (pos.position, pos.position + pos.direction * self.length)
    }
}

/// Positioning of detector array
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DetectorArrayPositioning {
    /// Position vector of array
    pub position: Pair,
    /// Unit direction vector of array
    pub direction: Pair,
}

/// Light Ray
#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
struct Ray {
    /// Origin position vector
    origin: Pair,
    /// Unit normal direction vector
    direction: Pair,
    /// S-Polarization Transmittance probability
    s_transmittance: f64,
    /// P-Polarization Transmittance probability
    p_transmittance: f64,
}

impl Ray {
    /// Create a new unpolarized ray with full transmittance with a origin at (0, `y`) and a
    /// direction of (1, 0)
    ///
    /// # Arguments
    ///  * `y` - the initial y value of the ray's position
    fn new_from_start(y: f64) -> Self {
        Ray {
            origin: Pair { x: 0., y },
            direction: Pair { x: 1., y: 0. },
            s_transmittance: 1.,
            p_transmittance: 1.,
        }
    }

    /// The average of the S & P Polarizations transmittance's
    fn average_transmittance(self) -> f64 {
        (self.s_transmittance + self.p_transmittance) * 0.5
    }

    /// Refract ray through interface of two different media
    /// using vector form of snell's law
    ///
    /// # Arguments
    ///  * `intersection` - point of intersection between the media
    ///  * `normal` - the unit normal vector of the interface
    ///  * `ci` - cosine of incident angle
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    fn refract(
        self,
        intersection: Pair,
        normal: Pair,
        ci: f64,
        n1: f64,
        n2: f64,
    ) -> Result<Self, RayTraceError> {
        debug_assert!(n1 >= 1_f64);
        debug_assert!(n2 >= 1_f64);
        debug_assert!(normal.is_unit());
        let r = n1 / n2;
        let cr_sq = 1_f64 - r * r * (1_f64 - ci * ci);
        if cr_sq < 0_f64 {
            return Err(RayTraceError::TotalInternalReflection);
        }
        let cr = cr_sq.sqrt();
        let v = self.direction * r + normal * (r * ci - cr);
        let fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr);
        let fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci);
        Ok(Self {
            origin: intersection,
            direction: v,
            s_transmittance: self.s_transmittance * (1_f64 - fresnel_rs * fresnel_rs),
            p_transmittance: self.p_transmittance * (1_f64 - fresnel_rp * fresnel_rp),
        })
    }

    /// Find the intersection point of the ray with the interface
    /// of the current media and the next media. Using the line-plane intersection formula.
    /// Then refract the ray through the interface
    ///
    /// # Arguments
    ///  * `plane` - the inter-media interface plane
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    ///  * `prism_height` - the height of the prism
    fn intersect_plane_interface(
        self,
        plane: &Surface,
        n1: f64,
        n2: f64,
        prism_height: f64,
    ) -> Result<Self, RayTraceError> {
        let ci = -self.direction.dot(plane.normal);
        if ci <= 0_f64 {
            return Err(RayTraceError::OutOfBounds);
        }
        let d = (self.origin - plane.midpt).dot(plane.normal) / ci;
        let p = self.origin + self.direction * d;
        if p.y <= 0_f64 || prism_height <= p.y {
            return Err(RayTraceError::OutOfBounds);
        }
        self.refract(p, plane.normal, ci, n1, n2)
    }

    /// Find the intersection point of the ray with the lens-like interface
    /// of current media and the next media. Using the line-sphere intersection formula.
    /// Then refract the ray through the interface
    ///
    /// # Arguments
    ///  * `lens` - the parameterized curved surface of the lens
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    fn intersect_curved_interface(
        self,
        lens: &CurvedSurface,
        n1: f64,
        n2: f64,
    ) -> Result<Self, RayTraceError> {
        let delta = self.origin - lens.center;
        let ud = self.direction.dot(delta);
        let under = ud * ud - delta.norm_squared() + lens.radius * lens.radius;
        if under < 0_f64 {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let d = -ud + under.sqrt();
        if d <= 0_f64 {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let p = self.origin + self.direction * d;
        if !lens.is_along_arc(p) {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let snorm = (lens.center - p) / lens.radius;
        debug_assert!(snorm.is_unit());
        self.refract(p, snorm, -self.direction.dot(snorm), n1, n2)
    }

    /// Find the intersection position of the ray with the detector array
    /// and the ray's transmission probability. The intersection position is a
    /// scalar on the line defined by the detector array.
    ///
    /// # Arguments
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    fn intersect_detector_array(
        self,
        detarr: &DetectorArray,
        detpos: &DetectorArrayPositioning,
    ) -> Result<(Pair, f64, f64), RayTraceError> {
        let ci = -self.direction.dot(detarr.normal);
        if ci <= detarr.min_ci {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let d = (self.origin - detpos.position).dot(detarr.normal) / ci;
        debug_assert!(d > 0.);
        let p = self.origin + self.direction * d;
        debug_assert!((detpos.direction).is_unit());
        let pos = (p - detpos.position).dot(detpos.direction);
        if pos < 0_f64 || detarr.length < pos {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        Ok((p, pos, self.average_transmittance()))
    }

    /// Propagate a ray of `wavelength` through the compound prism
    ///
    /// # Arguments
    ///  * `cmpnd` - the compound prism specification
    ///  * `wavelength` - the wavelength of the light ray
    fn propagate_internal(
        self,
        cmpnd: &CompoundPrism,
        wavelength: f64,
    ) -> Result<Ray, RayTraceError> {
        let (ray, n1) =
            cmpnd
                .prisms
                .iter()
                .try_fold((self, 1_f64), |(ray, n1), (glass, plane)| {
                    let n2 = glass.calc_n(wavelength);
                    debug_assert!(n2 >= 1.);
                    let ray = ray.intersect_plane_interface(plane, n1, n2, cmpnd.height)?;
                    Ok((ray, n2))
                })?;
        let n2 = 1_f64;
        ray.intersect_curved_interface(&cmpnd.lens, n1, n2)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning the intersection scalar
    /// and the transmission probability.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `cmpnd` - the compound prism specification
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    fn propagate(
        self,
        wavelength: f64,
        cmpnd: &CompoundPrism,
        detarr: &DetectorArray,
        detpos: &DetectorArrayPositioning,
    ) -> Result<(Pair, f64, f64), RayTraceError> {
        self.propagate_internal(cmpnd, wavelength)?
            .intersect_detector_array(detarr, detpos)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning an iterator of the ray's origin position and
    /// all of the intersection positions.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `cmpnd` - the compound prism specification
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    fn trace<'s>(
        self,
        wavelength: f64,
        cmpnd: &'s CompoundPrism<'s>,
        detarr: &'s DetectorArray<'s>,
        detpos: &'s DetectorArrayPositioning,
    ) -> impl Iterator<Item = Result<Pair, RayTraceError>> + 's {
        let mut ray = self;
        let mut n1 = 1_f64;
        let mut prisms = cmpnd.prisms.iter().fuse();
        let mut internal = true;
        let mut done = false;
        let mut propagation_fn = move || -> Result<Option<Pair>, RayTraceError> {
            match prisms.next() {
                Some((glass, plane)) => {
                    let n2 = glass.calc_n(wavelength);
                    ray = ray.intersect_plane_interface(plane, n1, n2, cmpnd.height)?;
                    n1 = n2;
                    Ok(Some(ray.origin))
                }
                None if !done && internal => {
                    internal = false;
                    let n2 = 1_f64;
                    ray = ray.intersect_curved_interface(&cmpnd.lens, n1, n2)?;
                    Ok(Some(ray.origin))
                }
                None if !done && !internal => {
                    done = true;
                    let (pos, _, _) = ray.intersect_detector_array(detarr, detpos)?;
                    Ok(Some(pos))
                }
                _ if done => Ok(None),
                _ => unreachable!(),
            }
        };
        std::iter::once(Ok(self.origin))
            .chain(std::iter::from_fn(move || propagation_fn().transpose()).fuse())
    }
}

/// Find the position and orientation of the detector array,
/// parameterized by the minimum and maximum wavelengths of the input beam,
/// and its angle from the normal.
///
/// # Arguments
///  * `cmpnd` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
pub fn detector_array_positioning(
    cmpnd: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
) -> Result<DetectorArrayPositioning, RayTraceError> {
    let ray = Ray::new_from_start(beam.y_mean);
    let (wmin, wmax) = beam.w_range;
    let lower_ray = ray.propagate_internal(cmpnd, wmin)?;
    let upper_ray = ray.propagate_internal(cmpnd, wmax)?;
    if lower_ray.average_transmittance() <= (1e-3) || upper_ray.average_transmittance() <= (1e-3) {
        return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
    }
    debug_assert!(lower_ray.direction.is_unit());
    debug_assert!(upper_ray.direction.is_unit());
    let spec_dir = rotate(detarr.angle, (0_f64, 1_f64).into());
    let spec = spec_dir * detarr.length;
    let mat = Mat2::new_from_cols(upper_ray.direction, -lower_ray.direction);
    let imat = mat.inverse().ok_or(RayTraceError::NoSurfaceIntersection)?;
    let dists = imat * (spec - upper_ray.origin + lower_ray.origin);
    let d2 = dists.y;
    let l_vertex = lower_ray.origin + lower_ray.direction * d2;
    let (pos, dir) = if d2 > 0_f64 {
        (l_vertex, spec_dir)
    } else {
        let dists = imat * (-spec - upper_ray.origin + lower_ray.origin);
        let d2 = dists.y;
        if d2 < 0. {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let u_vertex = lower_ray.origin + lower_ray.direction * d2;
        (u_vertex, -spec_dir)
    };
    Ok(DetectorArrayPositioning {
        position: pos,
        direction: dir,
    })
}

/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
#[derive(Clone)]
struct Welford {
    count: f64,
    mean: f64,
    m2: f64,
}

impl Welford {
    fn new() -> Self {
        Welford {
            count: 0.,
            mean: 0.,
            m2: 0.,
        }
    }
    fn next_sample(&mut self, x: f64) {
        self.count += 1.;
        let delta = x - self.mean;
        self.mean += delta / self.count;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    #[allow(dead_code)]
    fn variance(&self) -> f64 {
        self.m2 / self.count
    }
    fn sample_variance(&self) -> f64 {
        self.m2 / (self.count - 1.)
    }
}

/// Conditional Probability of detection per detector given a wavelength
/// { p(D=d|Λ=λ) : d in D }
///
/// # Arguments
///  * `wavelength` - given wavelength
///  * `cmpnd` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
///  * `detpos` - the position and orientation of the detector array
pub fn p_dets_l_wavelength(
    wavelength: f64,
    cmpnd: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
    detpos: &DetectorArrayPositioning,
) -> impl Iterator<Item = f64> {
    const MAX_N: usize = 1000;
    let mut p_dets_l_w_stats = vec![Welford::new(); detarr.bins.len()];
    let p_z = erf(cmpnd.width * FRAC_1_SQRT_2 / beam.width);
    debug_assert!(0. <= p_z && p_z <= 1.);
    let mut qrng = quasirandom::Qrng::new(1);
    for u in std::iter::repeat_with(|| qrng.next::<f64>()).take(MAX_N) {
        // Inverse transform sampling-method: U[0, 1) => N(µ = beam.y_mean, σ = beam.width / 2)
        let y = beam.y_mean - beam.width * FRAC_1_SQRT_2 * erfc_inv(2. * u);
        if y <= 0. || cmpnd.height <= y {
            for stat in p_dets_l_w_stats.iter_mut() {
                stat.next_sample(0.);
            }
            continue;
        }
        let ray = Ray::new_from_start(y);
        if let Ok((_, pos, t)) = ray.propagate(wavelength, cmpnd, detarr, detpos) {
            debug_assert!(pos.is_finite());
            debug_assert!(0. <= pos && pos <= detarr.length);
            debug_assert!(t.is_finite());
            debug_assert!(0. <= t && t <= 1.);
            // What is actually being integrated is
            // pdf_t = p_z * t * pdf(y);
            // But because of importance sampling using the same distribution
            // pdf_t /= pdf(y);
            // the pdf(y) is cancelled, so.
            // pdf_t = p_z * t;
            let pdf_t = p_z * t;
            for (stat, &[l, u]) in p_dets_l_w_stats.iter_mut().zip(detarr.bins.iter()) {
                if l <= pos && pos < u {
                    stat.next_sample(pdf_t);
                } else {
                    stat.next_sample(0.);
                }
            }
        } else {
            for stat in p_dets_l_w_stats.iter_mut() {
                stat.next_sample(0.);
            }
        }
        if p_dets_l_w_stats.iter().all(|stat| {
            // err = sample_variance / sqrt(N)
            let var = stat.sample_variance();
            const MAX_ERR: f64 = 7.5e-3;
            const MAX_ERR_SQ: f64 = MAX_ERR * MAX_ERR;
            (var * var) < MAX_ERR_SQ * stat.count
        }) {
            break;
        }
    }
    debug_assert!(p_dets_l_w_stats
        .iter()
        .all(|s| 0. <= s.mean && s.mean <= 1.));
    p_dets_l_w_stats.into_iter().map(|w| w.mean)
}

/// The mutual information of Λ and D. How much information is gained about Λ by measuring D.
/// I(Λ; D) = H(D) - H(D|Λ)
///   = Sum(Integrate(p(Λ=λ) p(D=d|Λ=λ) log2(p(D=d|Λ=λ)), {λ, wmin, wmax}), d in D)
///      - Sum(p(D=d) log2(p(D=d)), d in D)
/// p(D=d) = Expectation_Λ(p(D=d|Λ=λ)) = Integrate(p(Λ=λ) p(D=d|Λ=λ), {λ, wmin, wmax})
/// p(Λ=λ) = 1 / (wmax - wmin) * step(wmin <= λ <= wmax)
/// H(Λ) is ill-defined because Λ is continuous, but I(Λ; D) is still well-defined for continuous variables.
/// https://en.wikipedia.org/wiki/Differential_entropy#Definition
fn mutual_information(
    cmpnd: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
    detpos: &DetectorArrayPositioning,
) -> f64 {
    const MAX_N: usize = 1000;
    let (wmin, wmax) = beam.w_range;
    let mut p_dets_stats = vec![Welford::new(); detarr.bins.len()];
    let mut info_stats = vec![Welford::new(); detarr.bins.len()];
    let mut qrng = quasirandom::Qrng::new(1);
    for u in std::iter::repeat_with(|| qrng.next::<f64>()).take(MAX_N) {
        // Inverse transform sampling-method: U[0, 1) => U[wmin, wmax)
        let w = wmin + u * (wmax - wmin);
        let p_dets_l_w = p_dets_l_wavelength(w, cmpnd, detarr, beam, detpos);
        for ((dstat, istat), p_det_l_w) in p_dets_stats
            .iter_mut()
            .zip(info_stats.iter_mut())
            .zip(p_dets_l_w)
        {
            debug_assert!(0. <= p_det_l_w && p_det_l_w <= 1.);
            dstat.next_sample(p_det_l_w);
            if p_det_l_w > 0. {
                istat.next_sample(p_det_l_w * log2(p_det_l_w));
            } else {
                istat.next_sample(0.);
            }
        }
        if p_dets_stats.iter().chain(info_stats.iter()).all(|stat| {
            // err = sample_variance / sqrt(N)
            let var = stat.sample_variance();
            const MAX_ERR: f64 = 2.5e-3;
            const MAX_ERR_SQ: f64 = MAX_ERR * MAX_ERR;
            (var * var) < MAX_ERR_SQ * stat.count
        }) {
            break;
        }
    }
    let mut info: f64 = info_stats.into_iter().map(|s| s.mean).sum();
    for stat in p_dets_stats {
        let p_det = stat.mean;
        debug_assert!(0. <= p_det && p_det <= 1.);
        if p_det > 0. {
            info -= p_det * log2(p_det);
        }
    }
    info
}

/// Trace the propagation of a ray of `wavelength` through the compound prism and
/// intersection the detector array. Returning an iterator of the ray's origin position and
/// all of the intersection positions.
///
/// # Arguments
///  * `wavelength` - the wavelength of the light ray
///  * `init_y` - the inital y value of the ray
///  * `cmpnd` - the compound prism specification
///  * `detarr` - detector array specification
///  * `detpos` - the position and orientation of the detector array
pub fn trace<'s>(
    wavelength: f64,
    init_y: f64,
    cmpnd: &'s CompoundPrism<'s>,
    detarr: &'s DetectorArray<'s>,
    detpos: &'s DetectorArrayPositioning,
) -> impl Iterator<Item = Result<Pair, RayTraceError>> + 's {
    let ray = Ray::new_from_start(init_y);
    ray.trace(wavelength, cmpnd, detarr, detpos)
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct DesignFitness {
    pub size: f64,
    pub info: f64,
    pub deviation: f64,
}

/// Return the fitness of the spectrometer design to be minimized by an optimizer.
/// The fitness objectives are
/// * size = the distance from the mean starting position of the beam to the center of detector array
/// * info = I(Λ; D)
/// * deviation = sin(abs(angle of deviation))
///
/// # Arguments
///  * `prism` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
pub fn fitness(
    cmpnd: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
) -> Result<DesignFitness, RayTraceError> {
    let detpos = detector_array_positioning(cmpnd, detarr, beam)?;
    let deviation_vector =
        detpos.position + detpos.direction * detarr.length * 0.5 - (0., beam.y_mean).into();
    let size = deviation_vector.norm();
    let deviation = deviation_vector.y.abs() / deviation_vector.norm();
    let info = mutual_information(cmpnd, detarr, beam, &detpos);
    Ok(DesignFitness {
        size,
        info,
        deviation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::ops::Deref;

    fn approx_eq(lhs: f64, rhs: f64, epsilon: f64) -> bool {
        let diff = f64::abs(lhs - rhs);

        if lhs.is_nan() || rhs.is_nan() {
            false
        } else if lhs == rhs {
            true
        } else if lhs == 0. || rhs == 0. || (lhs.abs() + rhs.abs() < std::f64::MIN_POSITIVE) {
            diff < epsilon
        } else {
            let sum = lhs.abs() + rhs.abs();
            if sum < std::f64::MAX {
                diff / sum < epsilon
            } else {
                diff / std::f64::MAX < epsilon
            }
        }
    }

    #[test]
    fn test_many() {
        let catalog = crate::glasscat::BUNDLED_CATALOG.deref();
        let nglass = catalog.len();
        let seed = 123456;
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed);
        let ntest = 500;
        let max_nprism = 6;
        let prism_height = 25.;
        let prism_width = 25.;
        let max_length = 0.5 * prism_height;
        let pmt_length = 3.2;
        const NBIN: usize = 32;
        let bounds: Box<[_]> = (0..=NBIN)
            .map(|i| (i as f64) / (NBIN as f64) * pmt_length)
            .collect();
        let bins: Box<[_]> = bounds.windows(2).map(|t| [t[0], t[1]]).collect();
        let spec_max_accepted_angle = (45_f64).to_radians();
        let beam_width = 0.2;
        let wavelegth_range = (0.5, 0.82);
        let mut nvalid = 0;
        while nvalid < ntest {
            let nprism: usize = rng.gen_range(1, 1 + max_nprism);
            let glasses = (0..nprism)
                .map(|_| &catalog[rng.gen_range(0, nglass)].1)
                .collect::<Vec<_>>();
            let angles = (0..nprism + 1)
                .map(|_| rng.gen_range(-FRAC_PI_2, FRAC_PI_2))
                .collect::<Vec<_>>();
            let lengths = (0..nprism)
                .map(|_| rng.gen_range(0., max_length))
                .collect::<Vec<_>>();
            let curvature = rng.gen_range(0., 1.);
            let prism = CompoundPrism::new(
                glasses,
                angles.as_ref(),
                lengths.as_ref(),
                curvature,
                prism_height,
                prism_width,
            );

            let detarr_angle = rng.gen_range(-PI, PI);
            let detarr = DetectorArray::new(
                bins.as_ref().into(),
                spec_max_accepted_angle.cos(),
                detarr_angle,
                pmt_length,
            );

            let y_mean = rng.gen_range(0., prism_height);
            let beam = GaussianBeam {
                width: beam_width,
                y_mean,
                w_range: wavelegth_range,
            };

            let detpos = match detector_array_positioning(&prism, &detarr, &beam) {
                Ok(d) => d,
                Err(_) => continue,
            };
            nvalid += 1;
            let nwlen = 25;
            for i in 0..nwlen {
                let w = wavelegth_range.0
                    + (wavelegth_range.1 - wavelegth_range.0) * ((i as f64) / ((nwlen - 1) as f64));
                let ps = p_dets_l_wavelength(w, &prism, &detarr, &beam, &detpos);
                for p in ps {
                    assert!(p.is_finite() && 0. <= p && p <= 1.);
                }
            }
            let v = fitness(&prism, &detarr, &beam).unwrap();
            assert!(v.size > 0.);
            assert!(0. <= v.info && v.info <= (NBIN as f64).log2());
            assert!(0. <= v.deviation && v.deviation < FRAC_PI_2);
        }
    }

    #[test]
    fn test_with_known_prism() {
        let glasses = [
            &Glass::Sellmeier1([
                1.029607,
                0.00516800155,
                0.1880506,
                0.0166658798,
                0.736488165,
                138.964129,
            ]),
            &Glass::Sellmeier1([
                1.87543831,
                0.0141749518,
                0.37375749,
                0.0640509927,
                2.30001797,
                177.389795,
            ]),
            &Glass::Sellmeier1([
                0.738042712,
                0.00339065607,
                0.363371967,
                0.0117551189,
                0.989296264,
                212.842145,
            ]),
        ];
        let angles = [-27.2712308, 34.16326141, -42.93207009, 1.06311416];
        let angles: Box<[f64]> = angles.iter().cloned().map(f64::to_radians).collect();
        let lengths = [0_f64; 3];
        let prism = CompoundPrism::new(
            glasses.iter().copied(),
            angles.as_ref(),
            lengths.as_ref(),
            0.21,
            2.5,
            2.,
        );

        const NBIN: usize = 32;
        let pmt_length = 3.2;
        let bounds: Box<[_]> = (0..=NBIN)
            .map(|i| (i as f64) / (NBIN as f64) * pmt_length)
            .collect();
        let bins: Box<[_]> = bounds.windows(2).map(|t| [t[0], t[1]]).collect();
        let spec_max_accepted_angle = (60_f64).to_radians();
        let detarr = DetectorArray::new(
            bins.as_ref().into(),
            spec_max_accepted_angle.cos(),
            0.,
            pmt_length,
        );

        let beam = GaussianBeam {
            width: 0.2,
            y_mean: 0.95,
            w_range: (0.5, 0.82),
        };

        let v = fitness(&prism, &detarr, &beam).expect("Merit function failed");
        assert!(
            approx_eq(v.size, 41.324065257329245, 1e-3),
            "Size is incorrect. {} ≉ 41.3",
            v.size
        );
        assert!(
            approx_eq(v.info, 1.444212905142612, 5e-3),
            "Mutual information is incorrect. {} ≉ 1.44",
            v.info
        );
        assert!(
            approx_eq(v.deviation, 0.37715870072898755, 1e-3),
            "Deviation is incorrect. {} ≉ 0.377",
            v.deviation
        );
    }
}
