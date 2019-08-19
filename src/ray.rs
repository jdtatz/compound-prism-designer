use crate::glasscat::Glass;
use statrs::function::erf::{erf, erfc_inv};
use std::borrow::Cow;

#[derive(Debug, Display, Clone, Copy)]
pub enum RayTraceError {
    NoSurfaceIntersection,
    OutOfBounds,
    TotalInternalReflection,
    SpectrometerAngularResponseTooWeak,
    Unknown,
}

impl RayTraceError {
    pub fn name(self) -> &'static str {
        match self {
            RayTraceError::NoSurfaceIntersection => "NoSurfaceIntersection",
            RayTraceError::OutOfBounds => "OutOfBounds",
            RayTraceError::TotalInternalReflection => "TotalInternalReflection",
            RayTraceError::SpectrometerAngularResponseTooWeak => {
                "SpectrometerAngularResponseTooWeak"
            }
            RayTraceError::Unknown => "Unknown",
        }
    }
}

impl std::error::Error for RayTraceError {}

/// vector in R^2 represented as a 2-tuple
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy, From, Neg, Add, Sub, Mul, Div)]
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
    debug_assert!(vector.is_unit());
    Pair {
        x: angle.cos() * vector.x - angle.sin() * vector.y,
        y: angle.sin() * vector.x + angle.cos() * vector.y,
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

/// Compound Prism Specification
#[derive(Debug, Clone)]
pub struct CompoundPrism<'a> {
    /// List of glasses the compound prism is composed of, in order
    pub glasses: Cow<'a, [Glass]>,
    /// Angles that parameterize the shape of the compound prism
    pub angles: Cow<'a, [f64]>,
    /// Lengths that parameterize the trapezoidal shape of the compound prism
    pub lengths: Cow<'a, [f64]>,
    /// Lens Curvature of last surface of compound prism
    pub curvature: f64,
    /// Height of compound prism
    pub height: f64,
    /// Width of compound prism
    pub width: f64,
}

impl<'a> CompoundPrism<'a> {
    /// Iterator over the midpts of each prism surface
    fn midpts<'s>(&'s self) -> impl Iterator<Item = Pair> + 's {
        let h2 = self.height * 0.5;
        std::iter::once(self.angles[0].abs().tan() * h2)
            .chain(
                self.angles
                    .windows(2)
                    .zip(self.lengths.iter())
                    .map(move |(win, len)| {
                        let last = win[0];
                        let angle = win[1];
                        if (last >= 0.) ^ (angle >= 0.) {
                            (last.abs().tan() + angle.abs().tan()) * h2 + len
                        } else if last.abs() > angle.abs() {
                            (last.abs().tan() - angle.abs().tan()) * h2 + len
                        } else {
                            (angle.abs().tan() - last.abs().tan()) * h2 + len
                        }
                    }),
            )
            .scan(0_f64, |x, width| {
                *x += width;
                Some(*x)
            })
            .map(move |x| Pair { x, y: h2 })
    }
}

/// Array of detectors
#[derive(Debug, Clone)]
pub struct DetectorArray<'a> {
    /// Boundaries of detection bins
    pub bins: Cow<'a, [[f64; 2]]>,
    /// Minimum cosine of incident angle == cosine of maximum allowed incident angle
    pub min_ci: f64,
    /// CCW angle of the array from normal = Rot(θ) @ (0, 1)
    pub angle: f64,
    /// Length of the array
    pub length: f64,
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
    pub origin: Pair,
    /// Unit normal direction vector
    pub direction: Pair,
    /// Transmittance probability
    pub transmittance: f64,
}

impl Ray {
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
        let transmittance = 1_f64 - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) * (0.5);
        Ok(Self {
            origin: intersection,
            direction: v,
            transmittance: self.transmittance * transmittance,
        })
    }

    /// Find the intersection point of the ray with the interface
    /// of the current media and the next media. Using the line-plane intersection formula.
    /// Then refract the ray through the interface
    ///
    /// # Arguments
    ///  * `vertex` - point on the interface
    ///  * `normal` - the unit normal vector of the interface
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    ///  * `prism_height` - the height of the prism
    fn intersect_plane_interface(
        self,
        vertex: Pair,
        normal: Pair,
        n1: f64,
        n2: f64,
        prism_height: f64,
    ) -> Result<Self, RayTraceError> {
        debug_assert!(normal.is_unit());
        let ci = -self.direction.dot(normal);
        if ci <= 0_f64 {
            return Err(RayTraceError::OutOfBounds);
        }
        let d = (self.origin - vertex).dot(normal) / ci;
        let p = self.origin + self.direction * d;
        if p.y <= 0_f64 || prism_height <= p.y {
            return Err(RayTraceError::OutOfBounds);
        }
        self.refract(p, normal, ci, n1, n2)
    }

    /// Find the intersection point of the ray with the lens-like interface
    /// of current media and the next media. Using the line-sphere intersection formula.
    /// Then refract the ray through the interface
    ///
    /// # Arguments
    ///  * `midpt` - midpoint of the len-like interface
    ///  * `normal` - the unit normal vector of the interface from the midpt
    ///  * `curvature` - the normalized curvature value of the interface
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    ///  * `prism_height` - the height of the prism
    fn intersect_curved_interface(
        self,
        midpt: Pair,
        normal: Pair,
        curvature: f64,
        n1: f64,
        n2: f64,
        prism_height: f64,
    ) -> Result<Self, RayTraceError> {
        debug_assert!(normal.is_unit());
        let chord = prism_height / normal.x.abs();
        let lens_radius = chord * (0.5) / curvature;
        let rs = (lens_radius * lens_radius - chord * chord * (0.25)).sqrt();
        let center = midpt + normal * rs;
        let delta = self.origin - center;
        let ud = self.direction.dot(delta);
        let under = ud * ud - delta.norm_squared() + lens_radius * lens_radius;
        if under < 0_f64 {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let d = -ud + under.sqrt();
        if d <= 0_f64 {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let p = self.origin + self.direction * d;
        let rd = p - midpt;
        if rd.norm_squared() > (chord * chord / (4.)) {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let snorm = (center - p) / lens_radius;
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
        detpos: DetectorArrayPositioning,
    ) -> Result<(Pair, f64, f64), RayTraceError> {
        let normal = rotate(detarr.angle, (-1_f64, 0_f64).into());
        debug_assert!(normal.is_unit());
        let ci = -self.direction.dot(normal);
        if ci <= detarr.min_ci {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let d = (self.origin - detpos.position).dot(normal) / ci;
        if d <= 0_f64 {
            // panic!("RayTraceError::Unknown");
            return Err(RayTraceError::Unknown);
        }
        let p = self.origin + self.direction * d;
        debug_assert!((detpos.direction).is_unit());
        let pos = (p - detpos.position).dot(detpos.direction);
        if pos < 0_f64 || detarr.length < pos {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        Ok((p, pos, self.transmittance))
    }

    /// Propagate a ray of `wavelength` through the compound prism
    ///
    /// # Arguments
    ///  * `prism` - the compound prism specification
    ///  * `wavelength` - the wavelength of the light ray
    fn propagate_internal(
        self,
        prism: &CompoundPrism,
        wavelength: f64,
    ) -> Result<Ray, RayTraceError> {
        let mut mids = prism.midpts();
        let (ray, n1) = prism
            .glasses
            .iter()
            .zip(prism.angles.iter())
            .zip(&mut mids)
            .try_fold((self, 1_f64), |(ray, n1), ((glass, angle), vertex)| {
                let n2 = glass.calc_n(wavelength);
                let normal = rotate(*angle, (-1_f64, 0_f64).into());
                debug_assert!(normal.is_unit());
                let ray = ray.intersect_plane_interface(vertex, normal, n1, n2, prism.height)?;
                Ok((ray, n2))
            })?;
        let midpt = mids.next().unwrap();
        let angle = prism.angles[prism.angles.len() - 1];
        let n2 = 1_f64;
        let normal = rotate(angle, (-1_f64, 0_f64).into());
        debug_assert!(normal.is_unit());
        ray.intersect_curved_interface(midpt, normal, prism.curvature, n1, n2, prism.height)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning the intersection scalar
    /// and the transmission probability.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `prism` - the compound prism specification
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    fn propagate(
        self,
        wavelength: f64,
        prism: &CompoundPrism,
        detarr: &DetectorArray,
        detpos: DetectorArrayPositioning,
    ) -> Result<(Pair, f64, f64), RayTraceError> {
        self.propagate_internal(prism, wavelength)?
            .intersect_detector_array(detarr, detpos)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning a list of the ray's origin position and
    /// all of the intersection positions.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `prism` - the compound prism specification
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    fn trace(
        self,
        wavelength: f64,
        prism: &CompoundPrism,
        detarr: &DetectorArray,
        detpos: DetectorArrayPositioning,
    ) -> Result<Vec<Pair>, RayTraceError> {
        let mut traced = Vec::new();
        let mut mids = prism.midpts();
        let (ray, n1) = prism
            .glasses
            .iter()
            .zip(prism.angles.iter())
            .zip(&mut mids)
            .try_fold((self, 1_f64), |(ray, n1), ((glass, angle), vertex)| {
                traced.push(ray.origin);
                let n2 = glass.calc_n(wavelength);
                let normal = rotate(*angle, (-1_f64, 0_f64).into());
                let ray = ray.intersect_plane_interface(vertex, normal, n1, n2, prism.height)?;
                Ok((ray, n2))
            })?;
        traced.push(ray.origin);
        let midpt = mids.next().unwrap();
        let angle = prism.angles[prism.angles.len() - 1];
        let n2 = 1_f64;
        let normal = rotate(angle, (-1_f64, 0_f64).into());
        let ray =
            ray.intersect_curved_interface(midpt, normal, prism.curvature, n1, n2, prism.height)?;
        traced.push(ray.origin);
        let (pos, _, _) = ray.intersect_detector_array(detarr, detpos)?;
        traced.push(pos);
        Ok(traced)
    }
}

/// Find the position and orientation of the detector array,
/// parameterized by the minimum and maximum wavelengths of the input beam,
/// and its angle from the normal.
///
/// # Arguments
///  * `prism` - the compound prism specification
///  * `detarr` - detector array specification
///  * `detarr` - input gaussian beam specification
pub fn detector_array_positioning(
    prism: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
) -> Result<DetectorArrayPositioning, RayTraceError> {
    let ray = Ray {
        origin: (0_f64, beam.y_mean).into(),
        direction: (1_f64, 0_f64).into(),
        transmittance: 1_f64,
    };
    let (wmin, wmax) = beam.w_range;
    let lower_ray = ray.propagate_internal(prism, wmin)?;
    let upper_ray = ray.propagate_internal(prism, wmax)?;
    if lower_ray.transmittance <= (1e-3) || upper_ray.transmittance <= (1e-3) {
        return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
    }
    debug_assert!(lower_ray.direction.is_unit());
    debug_assert!(upper_ray.direction.is_unit());
    let spec_dir = rotate(detarr.angle, (0_f64, 1_f64).into());
    let spec = spec_dir * detarr.length;
    let det = upper_ray.direction.y * lower_ray.direction.x
        - upper_ray.direction.x * lower_ray.direction.y;
    if det == 0_f64 {
        return Err(RayTraceError::NoSurfaceIntersection);
    }
    let v = Pair {
        x: -upper_ray.direction.y,
        y: upper_ray.direction.x,
    } / det;
    let d2 = v.dot(spec - upper_ray.origin + lower_ray.origin);
    let l_vertex = lower_ray.origin + lower_ray.direction * d2;
    let (pos, dir) = if d2 > 0_f64 {
        (l_vertex, spec_dir)
    } else {
        let d2 = v.dot(-spec - upper_ray.origin + lower_ray.origin);
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
///  * `prism` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
///  * `detpos` - the position and orientation of the detector array
pub fn p_dets_l_wavelength(
    wavelength: f64,
    prism: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
    detpos: DetectorArrayPositioning,
) -> impl IntoIterator<Item = f64> {
    const MAX_N: usize = 1000;
    // const FRAC_SQRT_2_SQRT_PI: f64 = core::f64::consts::FRAC_1_SQRT_2 * core::f64::consts::FRAC_2_SQRT_PI;
    let mut p_dets_l_w_stats = vec![Welford::new(); detarr.bins.len()];
    let p_z = erf(prism.width * core::f64::consts::FRAC_1_SQRT_2 / beam.width);
    let mut qrng = quasirandom::Qrng::new(0);
    for u in std::iter::repeat_with(|| qrng.next::<f64>()).take(MAX_N) {
        let y = beam.y_mean - beam.width * core::f64::consts::FRAC_1_SQRT_2 * erfc_inv(2. * u);
        // let y_bar = y - beam.y_mean;
        // let pdf_y = f64::exp(-2. * y_bar * y_bar / (beam.width * beam.width)) * FRAC_SQRT_2_SQRT_PI / beam.width;
        if y <= 0. || prism.height <= y {
            for stat in p_dets_l_w_stats.iter_mut() {
                stat.next_sample(0.);
            }
            continue;
        }
        let ray = Ray {
            origin: (0., y).into(),
            direction: (1., 0.).into(),
            transmittance: 1.,
        };
        if let Ok((_, pos, t)) = ray.propagate(wavelength, prism, detarr, detpos) {
            debug_assert!(pos.is_finite());
            debug_assert!(t.is_finite());
            debug_assert!(0. <= t && t <= 1.);
            let p_t = p_z * t;
            debug_assert!(0. <= p_t && p_t <= 1.);
            for (stat, &[l, u]) in p_dets_l_w_stats.iter_mut().zip(detarr.bins.iter()) {
                if l <= pos && pos < u {
                    stat.next_sample(p_t);
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
            let err = stat.sample_variance() / stat.count.sqrt();
            err < 3e-3
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
    prism: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
    detpos: DetectorArrayPositioning,
) -> f64 {
    const MAX_N: usize = 1000;
    let (wmin, wmax) = beam.w_range;
    let mut p_dets_stats = vec![Welford::new(); detarr.bins.len()];
    let mut info_stats = vec![Welford::new(); detarr.bins.len()];
    let mut qrng = quasirandom::Qrng::new(0);
    for u in std::iter::repeat_with(|| qrng.next::<f64>()).take(MAX_N) {
        let w = wmin + u * (wmax - wmin);
        let p_dets_l_w = p_dets_l_wavelength(w, prism, detarr, beam, detpos);
        for ((dstat, istat), p_det_l_w) in p_dets_stats
            .iter_mut()
            .zip(info_stats.iter_mut())
            .zip(p_dets_l_w)
        {
            debug_assert!(0. <= p_det_l_w && p_det_l_w <= 1.);
            dstat.next_sample(p_det_l_w);
            if p_det_l_w > 0. {
                istat.next_sample(p_det_l_w * p_det_l_w.log2());
            } else {
                istat.next_sample(0.);
            }
        }
        if p_dets_stats.iter().chain(info_stats.iter()).all(|stat| {
            let err = stat.sample_variance() / stat.count.sqrt();
            err < 3e-3
        }) {
            break;
        }
    }
    let mut info: f64 = info_stats.into_iter().map(|s| s.mean).sum();
    for stat in p_dets_stats {
        let p_det = stat.mean;
        debug_assert!(0. <= p_det && p_det <= 1.);
        if p_det > 0. {
            info -= p_det * p_det.log2();
        }
    }
    info
}

/// Trace the propagation of a ray of `wavelength` through the compound prism and
/// intersection the detector array. Returning a list of the ray's origin position and
/// all of the intersection positions.
///
/// # Arguments
///  * `wavelength` - the wavelength of the light ray
///  * `init_y` - the inital y value of the ray
///  * `prism` - the compound prism specification
///  * `detarr` - detector array specification
///  * `detpos` - the position and orientation of the detector array
pub fn trace(
    wavelength: f64,
    init_y: f64,
    prism: &CompoundPrism,
    detarr: &DetectorArray,
    detpos: DetectorArrayPositioning,
) -> Result<Vec<Pair>, RayTraceError> {
    let ray = Ray {
        origin: (0., init_y).into(),
        direction: (1., 0.).into(),
        transmittance: 1.,
    };
    ray.trace(wavelength, prism, detarr, detpos)
}

/// Return the fitness of the spectrometer design to be minimized by an optimizer.
/// The fitness objectives are
/// * size = the distance from the mean starting position of the beam to the center of detector array
/// * info = -I(Λ; D)
/// * deviation = sin(abs(angle of deviation))
///
/// # Arguments
///  * `prism` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
pub fn fitness(
    prism: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
) -> Result<[f64; 3], RayTraceError> {
    let detpos = detector_array_positioning(prism, detarr, beam)?;
    let deviation_vector =
        detpos.position + detpos.direction * detarr.length * 0.5 - (0., beam.y_mean).into();
    let size = deviation_vector.norm();
    let deviation = deviation_vector.y.abs() / deviation_vector.norm();
    let info = mutual_information(prism, detarr, beam, detpos);
    Ok([size, -info, deviation])
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_with_known_prism() {
        let glasses = [
            Glass::Sellmeier1([
                1.029607,
                0.00516800155,
                0.1880506,
                0.0166658798,
                0.736488165,
                138.964129,
            ]),
            Glass::Sellmeier1([
                1.87543831,
                0.0141749518,
                0.37375749,
                0.0640509927,
                2.30001797,
                177.389795,
            ]),
            Glass::Sellmeier1([
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
        let prism = CompoundPrism {
            glasses: glasses.as_ref().into(),
            angles: angles.as_ref().into(),
            lengths: lengths.as_ref().into(),
            curvature: 0.21,
            height: 2.5,
            width: 2.,
        };

        const nbin: usize = 32;
        let pmt_length = 3.2;
        let bounds: Box<[_]> = (0..=nbin)
            .map(|i| (i as f64) / (nbin as f64) * pmt_length)
            .collect();
        let bins: Box<[_]> = bounds.windows(2).map(|t| [t[0], t[1]]).collect();
        let spec_max_accepted_angle = (60_f64).to_radians();
        let detarr = DetectorArray {
            bins: bins.as_ref().into(),
            min_ci: spec_max_accepted_angle.cos(),
            angle: 0.,
            length: pmt_length,
        };

        let beam = GaussianBeam {
            width: 0.2,
            y_mean: 0.38 * prism.height,
            w_range: (0.5, 0.82),
        };

        let v = fitness(&prism, &detarr, &beam).expect("Merit function failed");
        assert!(
            approx_eq(v[0], 41.324065257329245, 1e-3),
            "Size is incorrect. {} ≉ 41.3",
            v[0]
        );
        assert!(
            approx_eq(v[1], -1.444212905142612, 1e-3),
            "Mutual information is incorrect. {} ≉ -1.44",
            v[1]
        );
        assert!(
            approx_eq(v[2], 0.37715870072898755, 1e-3),
            "Deviation is incorrect. {} ≉ 0.377",
            v[2]
        );
    }
}
