use std::borrow::Cow;
use derive_enum::Name;
use crate::glasscat::Glass;
use crate::quad::{Quadrature, KR21};

#[derive(Name, Debug, Display, Clone, Copy)]
pub enum RayTraceError {
    NoSurfaceIntersection,
    OutOfBounds,
    TotalInternalReflection,
    SpectrometerAngularResponseTooWeak,
    Unknown,
}

impl std::error::Error for RayTraceError {}

/// vector in R^2 represented as a 2-tuple
#[derive(Debug, PartialEq, Clone, Copy, From, Neg, Add, Sub, Mul, Div)]
pub struct Pair {
    pub x: f64,
    pub y: f64,
}

impl Pair {
    /// zero vector (0, 0)
    pub const ZERO: Self = Self { x: 0_f64, y: 0_f64 };

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
    /// Lens Curvature of last surface of compound prism
    pub curvature: f64,
    /// Height of compound prism
    pub height: f64,
    /// Width of compound prism
    pub width: f64,
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
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DetectorArrayPositioning {
    /// Position vector of array
    pub pos: Pair,
    /// Unit direction vector of array
    pub dir: Pair,
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
    ) -> Result<(f64, f64), RayTraceError> {
        let normal = rotate(detarr.angle, (-1_f64, 0_f64).into());
        debug_assert!(normal.is_unit());
        let ci = -self.direction.dot(normal);
        if ci <= detarr.min_ci {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let d = (self.origin - detpos.pos).dot(normal) / ci;
        if d <= 0_f64 {
            // panic!("RayTraceError::Unknown");
            return Err(RayTraceError::Unknown);
        }
        let p = self.origin + self.direction * d;
        debug_assert!((detpos.dir).is_unit());
        let pos = (p - detpos.pos).dot(detpos.dir);
        Ok((pos, self.transmittance))
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
        let (ray, n1, vertex) = prism.glasses.iter().zip(prism.angles.iter()).try_fold(
            (self, 1_f64, Pair::ZERO),
            |(ray, n1, vertex), (glass, angle)| {
                let n2 = glass.calc_n(wavelength);
                let normal = rotate(*angle, (-1_f64, 0_f64).into());
                debug_assert!(normal.is_unit());
                let vertex = Pair {
                    x: vertex.x + angle.abs().tan() * prism.height,
                    y: if vertex.y == 0_f64 {
                        prism.height
                    } else {
                        0_f64
                    },
                };
                let ray = ray.intersect_plane_interface(vertex, normal, n1, n2, prism.height)?;
                Ok((ray, n2, vertex))
            },
        )?;
        let angle = prism.angles[prism.glasses.len()];
        let n2 = 1_f64;
        let normal = rotate(angle, (-1_f64, 0_f64).into());
        debug_assert!(normal.is_unit());
        let midpt = Pair {
            x: vertex.x + angle.abs().tan() * prism.height * (0.5),
            y: prism.height * (0.5),
        };
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
    ) -> Result<(f64, f64), RayTraceError> {
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
        let (ray, n1, vertex) = prism.glasses.iter().zip(prism.angles.iter()).try_fold(
            (self, 1_f64, Pair::ZERO),
            |(ray, n1, vertex), (glass, angle)| {
                traced.push(ray.origin);
                let n2 = glass.calc_n(wavelength);
                let normal = rotate(*angle, (-1_f64, 0_f64).into());
                let vertex = Pair {
                    x: vertex.x + angle.abs().tan() * prism.height,
                    y: if vertex.y == 0_f64 {
                        prism.height
                    } else {
                        0_f64
                    },
                };
                let ray = ray.intersect_plane_interface(vertex, normal, n1, n2, prism.height)?;
                Ok((ray, n2, vertex))
            },
        )?;
        traced.push(ray.origin);
        let angle = prism.angles[prism.glasses.len()];
        let n2 = 1_f64;
        let normal = rotate(angle, (-1_f64, 0_f64).into());
        let midpt = Pair {
            x: vertex.x + angle.abs().tan() * prism.height * (0.5),
            y: prism.height * (0.5),
        };
        let ray = ray.intersect_curved_interface(midpt, normal, prism.curvature, n1, n2, prism.height)?;
        traced.push(ray.origin);
        let normal = rotate(detarr.angle, (-1_f64, 0_f64).into());
        let ci = -ray.direction.dot(normal);
        if ci <= detarr.min_ci {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let d = (ray.origin - detpos.pos).dot(normal) / ci;
        if d <= 0_f64 {
            return Err(RayTraceError::Unknown);
        }
        let p = ray.origin + ray.direction * d;
        traced.push(p);
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
    Ok(DetectorArrayPositioning { pos, dir })
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
fn p_dets_l_wavelength(
    wavelength: f64,
    prism: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
    detpos: DetectorArrayPositioning,
) -> Vec<f64> {
    let mut p_dets_l_w = vec![0_f64; detarr.bins.len()];
    // p(D=d|Λ=λ) = Integrate(T(λ, y) * g(y) * step(d_l <= S(λ, y) < d_u), {y, 0, prism.height})
    KR21::inplace_integrate(
        |y, integration_factor| {
            let ray = Ray {
                origin: (0., y).into(),
                direction: (1., 0.).into(),
                transmittance: 1.,
            };
            let y_bar = y - beam.y_mean;
            // sqrt(2 / π)
            const FRAC_SQRT_2_SQRT_PI: f64 =
                core::f64::consts::FRAC_1_SQRT_2 * core::f64::consts::FRAC_2_SQRT_PI;
            // circular gaussian beam pdf parameterized by 1/e2 beam width
            // f(x, y) = Exp[-2 (x^2 + y^2) / beam.width^2] * 2 / (π beam.width^2)
            // g(y) = Integrate[f(x, y), {x, -prism.width / 2, prism.width / 2}]
            // g(y) = Exp[-2 y^2 / beam.width^2] Erf[w / (Sqrt[2] beam.width)] Sqrt[2 / π] / beam_width
            let g_y = f64::exp(-2. * y_bar * y_bar / (beam.width * beam.width))
                * libm::erf(prism.width * core::f64::consts::FRAC_1_SQRT_2 / beam.width)
                * FRAC_SQRT_2_SQRT_PI
                / beam.width;
            debug_assert!(g_y.is_finite() && g_y >= 0.);
            if let Ok((pos, t)) = ray.propagate(wavelength, prism, detarr, detpos) {
                debug_assert!(pos.is_finite());
                debug_assert!(t.is_finite());
                debug_assert!(0. <= t && t <= 1.);
                let pdf = t * g_y * integration_factor;
                for (p_det_l_w, &[l, u]) in p_dets_l_w.iter_mut().zip(detarr.bins.iter()) {
                    if l <= pos && pos < u {
                        *p_det_l_w += pdf;
                    }
                }
            }
        },
        0.,
        prism.height,
        10,
    );
    debug_assert!(p_dets_l_w.iter().all(|&p| 0. <= p && p <= 1.));
    p_dets_l_w
}

/// The mutual information of Λ and D. How much information is gained about Λ by measuring D.
/// I(Λ; D) = H(D) - H(Λ|D)
///   = Sum(Integrate(p(Λ=λ) p(D=d|Λ=λ) log2(p(D=d|Λ=λ)), {λ, wmin, wmax}), d in D)
///      - Sum(p(D=d) log2(p(D=d)), d in D)
/// p(D=d) = Expectation_Λ(p(D=d|Λ=λ)) = Integrate(p(Λ=λ) p(D=d|Λ=λ), {λ, wmin, wmax})
/// p(Λ=λ) = 1 / (wmax - wmin) * step(wmin <= λ <= wmax)
fn mutual_information(
    prism: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
    detpos: DetectorArrayPositioning,
) -> f64 {
    let nbins = detarr.bins.len();
    let (wmin, wmax) = beam.w_range;
    let p_w = 1. / (wmax - wmin);
    let mut info = 0_f64;
    let mut p_dets = vec![0_f64; nbins];
    // p(D=d) = Integrate[p(D=d|Λ=λ), {λ, wmin, wmax}]
    KR21::inplace_integrate(
        |w, integration_factor| {
            let p_dets_l_w = p_dets_l_wavelength(w, prism, detarr, beam, detpos);
            for (p_det, p_det_l_w) in p_dets.iter_mut().zip(p_dets_l_w) {
                debug_assert!(0. <= p_det_l_w && p_det_l_w <= 1.);
                if p_det_l_w > 0. {
                    *p_det += p_w * p_det_l_w * integration_factor;
                    info += p_w * p_det_l_w * p_det_l_w.log2() * integration_factor;
                }
            }
        },
        wmin,
        wmax,
        5,
    );
    for p_det in p_dets {
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

/// Returns the matrix of transmission probabilities for the given `wavelengths` with the detectors
/// { { p(D=d|Λ=λ) : λ in `wavelengths` } : d in D }
///
/// # Arguments
///  * `wavelengths` - the wavelengths to find the transmission probabilities of
///  * `prism` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
///  * `detpos` - the position and orientation of the detector array
pub fn transmission(
    wavelengths: &[f64],
    prism: &CompoundPrism,
    detarr: &DetectorArray,
    beam: &GaussianBeam,
    detpos: DetectorArrayPositioning,
) -> Vec<Vec<f64>> {
    let mut ts = vec![vec![0_f64; wavelengths.len()]; detarr.bins.len()];
    for (w_idx, w) in wavelengths.iter().cloned().enumerate() {
        let p_dets_l_w = p_dets_l_wavelength(w, prism, detarr, beam, detpos);
        for (b_idx, p_det_l_w) in p_dets_l_w.into_iter().enumerate() {
            ts[b_idx][w_idx] = p_det_l_w;
        }
    }
    ts
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
    let deviation_vector = detpos.pos + detpos.dir * detarr.length * 0.5 - (0., beam.y_mean).into();
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
        let prism = CompoundPrism {
            glasses: std::borrow::Cow::Borrowed(&glasses),
            angles: std::borrow::Cow::Borrowed(&angles),
            curvature: 0.21,
            height: 2.5,
            width: 2.,
        };

        let nbin = 32;
        let pmt_length = 3.2;
        let bounds: Box<[_]> = (0..=nbin)
            .map(|i| f64::from(i) / f64::from(nbin) * pmt_length)
            .collect();
        let bins: Box<[_]> = bounds.windows(2).map(|t| [t[0], t[1]]).collect();
        let spec_max_accepted_angle = (60_f64).to_radians();
        let detarr = DetectorArray {
            bins: std::borrow::Cow::Borrowed(&bins),
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
            "Size is incorrect. {} ≉ 41.324",
            v[0]
        );
        assert!(
            approx_eq(v[1], -1.444212905142612, 1e-3),
            "Mutual information is incorrect. {} ≉ -1.444",
            v[1]
        );
        assert!(
            approx_eq(v[2], 0.37715870072898755, 1e-3),
            "Deviation is incorrect. {} ≉ 0.377",
            v[2]
        );
    }
}
