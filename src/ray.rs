use crate::glasscat::Glass;
use crate::quad::{Quadrature, KR21};
use alga::general::RealField;

#[derive(Debug, Display, Clone, Copy)]
pub enum RayTraceError {
    NoSurfaceIntersection,
    OutOfBounds,
    TotalInternalReflection,
    SpectrometerAngularResponseTooWeak,
    Unknown,
}

impl std::error::Error for RayTraceError {}

fn from_f64<N: RealField>(v: f64) -> N {
    N::from_subset(&v)
}

#[derive(Debug, PartialEq, Clone, Copy, From, Neg, Add, Sub, Mul, Div)]
pub struct Pair<N: RealField> {
    pub x: N,
    pub y: N,
}

impl<N: RealField> Pair<N> {
    pub fn zero() -> Self {
        Pair {
            x: N::zero(),
            y: N::zero(),
        }
    }

    pub fn dot(self, other: Self) -> N {
        self.x * other.x + self.y * other.y
    }

    pub fn norm_squared(self) -> N {
        self.dot(self)
    }

    pub fn norm(self) -> N {
        self.norm_squared().sqrt()
    }
}

fn rotate<N: RealField>(angle: N, vector: Pair<N>) -> Pair<N> {
    Pair {
        x: angle.cos() * vector.x - angle.sin() * vector.y,
        y: angle.sin() * vector.x + angle.cos() * vector.y,
    }
}

/// Collimated Polychromatic Gaussian Beam from Collimator
#[derive(Constructor, Debug, Clone, Copy)]
pub struct GaussianBeam<N: RealField> {
    /// 1/e^2 beam width
    pub width: N,
    /// mean normalized y coordinate
    pub y_mean: N,
    /// Range of wavelengths
    pub w_range: (N, N),
}

/// Compound Prism Specification
#[derive(Constructor, Debug, Clone, Copy)]
pub struct Prism<'a, N: RealField> {
    pub glasses: &'a [Glass<N>],
    pub angles: &'a [N],
    pub curvature: N,
    pub height: N,
    pub width: N,
}

#[derive(Constructor, Debug, Clone, Copy)]
pub struct PmtArray<'a, N: RealField> {
    /// Normalized Boundaries of pmt bins
    pub bins: &'a [[N; 2]],
    /// Minimum cosine of incident angle == cosine of maximum allowed incident angle
    pub min_ci: N,
    /// CCW angle of the array from normal = Rot(θ) @ (0, 1)
    pub angle: N,
    /// Length of the array
    pub length: N,
}

#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
pub struct DetectorPositioning<N: RealField> {
    pub pos: Pair<N>,
    pub dir: Pair<N>,
    pub angle: N,
    pub length: N,
}

#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
struct Ray<N: RealField> {
    pub origin: Pair<N>,
    pub direction: Pair<N>,
    pub transmittance: N,
}

impl<N: RealField> Ray<N> {
    fn refract(
        self,
        intersection: Pair<N>,
        normal: Pair<N>,
        ci: N,
        n1: N,
        n2: N,
    ) -> Result<Self, RayTraceError> {
        let r = n1 / n2;
        let cr_sq: N = N::one() - r * r * (N::one() - ci * ci);
        if cr_sq < N::zero() {
            return Err(RayTraceError::TotalInternalReflection);
        }
        let cr = cr_sq.sqrt();
        let v = self.direction * r + normal * (r * ci - cr);
        let fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr);
        let fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci);
        let transmittance =
            N::one() - (fresnel_rs * fresnel_rs + fresnel_rp * fresnel_rp) / from_f64(2.);
        Ok(Self {
            origin: intersection,
            direction: v,
            transmittance: self.transmittance * transmittance,
        })
    }

    fn intersect_surface(
        self,
        vertex: Pair<N>,
        normal: Pair<N>,
        n1: N,
        n2: N,
    ) -> Result<Self, RayTraceError> {
        let ci = -self.direction.dot(normal);
        if ci <= N::zero() {
            return Err(RayTraceError::OutOfBounds);
        }
        let d = (self.origin - vertex).dot(normal) / ci;
        let p = self.origin + self.direction * d;
        if p.y <= N::zero() || N::one() <= p.y {
            return Err(RayTraceError::OutOfBounds);
        }
        self.refract(p, normal, ci, n1, n2)
    }

    fn intersect_lens(
        self,
        midpt: Pair<N>,
        normal: Pair<N>,
        curvature: N,
        n1: N,
        n2: N,
    ) -> Result<Self, RayTraceError> {
        let chord = normal.x.abs().recip();
        let lens_radius = chord / (curvature * from_f64(2.));
        let rs = (lens_radius * lens_radius - chord * chord / from_f64(4.)).sqrt();
        let center = midpt + normal * rs;
        let delta = self.origin - center;
        let ud = self.direction.dot(delta);
        let under = ud * ud - delta.norm_squared() + lens_radius * lens_radius;
        if under < N::zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let d = -ud + under.sqrt();
        if d <= N::zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let p = self.origin + self.direction * d;
        let rd = p - midpt;
        if rd.norm_squared() > (chord * chord / from_f64(4.)) {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let snorm = (center - p) / lens_radius;
        self.refract(p, snorm, -self.direction.dot(snorm), n1, n2)
    }

    fn intersect_spectrometer(
        self,
        spec: DetectorPositioning<N>,
        spec_min_ci: N,
    ) -> Result<(N, N), RayTraceError> {
        let normal = rotate(spec.angle, (-N::one(), N::zero()).into());
        let ci = -self.direction.dot(normal);
        if ci <= spec_min_ci {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let d = (self.origin - spec.pos).dot(normal) / ci;
        if d <= N::zero() {
            // panic!("RayTraceError::Unknown");
            return Err(RayTraceError::Unknown);
        }
        let p = self.origin + self.direction * d;
        let pos = (p - spec.pos).dot(spec.dir) / (spec.length * spec.length);
        Ok((pos, self.transmittance))
    }

    fn propagate_internal(self, prism: Prism<N>, wavelength: N) -> Result<Ray<N>, RayTraceError> {
        let (ray, n1, vertex) = prism.glasses.iter().zip(prism.angles).try_fold(
            (self, N::one(), Pair::<N>::zero()),
            |(ray, n1, vertex), (glass, angle)| {
                let n2 = glass.calc_n(wavelength);
                let normal = rotate(*angle, (-N::one(), N::zero()).into());
                let vertex = Pair {
                    x: vertex.x + angle.abs().tan(),
                    y: if vertex.y.is_zero() {
                        N::one()
                    } else {
                        N::zero()
                    },
                };
                let ray = ray.intersect_surface(vertex, normal, n1, n2)?;
                Ok((ray, n2, vertex))
            },
        )?;
        let angle = prism.angles[prism.glasses.len()];
        let n2 = N::one();
        let normal = rotate(angle, (-N::one(), N::zero()).into());
        let diff = angle.abs().tan();
        let midpt = Pair {
            x: vertex.x + diff * from_f64(0.5),
            y: from_f64(0.5),
        };
        ray.intersect_lens(midpt, normal, prism.curvature, n1, n2)
    }

    fn spectrometer_position(
        y_mean: N,
        lower_wavelength: N,
        upper_wavelength: N,
        prism: Prism<N>,
        spec_angle: N,
        spec_length: N,
    ) -> Result<DetectorPositioning<N>, RayTraceError> {
        let ray = Ray {
            origin: (N::zero(), y_mean).into(),
            direction: (N::one(), N::zero()).into(),
            transmittance: N::one(),
        };
        let lower_ray = ray.propagate_internal(prism, lower_wavelength)?;
        let upper_ray = ray.propagate_internal(prism, upper_wavelength)?;
        if lower_ray.transmittance <= from_f64(1e-3) || upper_ray.transmittance <= from_f64(1e-3) {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let spec = rotate(spec_angle, (N::zero(), N::one()).into()) * spec_length;
        let det = upper_ray.direction.y * lower_ray.direction.x
            - upper_ray.direction.x * lower_ray.direction.y;
        if det.is_zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let v = Pair {
            x: -upper_ray.direction.y,
            y: upper_ray.direction.x,
        } / det;
        let d2 = v.dot(spec - upper_ray.origin + lower_ray.origin);
        let l_vertex = lower_ray.origin + lower_ray.direction * d2;
        let (pos, dir) = if d2 > N::zero() {
            (l_vertex, spec)
        } else {
            let d2 = v.dot(-spec - upper_ray.origin + lower_ray.origin);
            let u_vertex = lower_ray.origin + lower_ray.direction * d2;
            (u_vertex, -spec)
        };
        Ok(DetectorPositioning {
            pos,
            dir,
            angle: spec_angle,
            length: spec_length,
        })
    }
    fn propagate(
        self,
        wavelength: N,
        prism: Prism<N>,
        spec: DetectorPositioning<N>,
        spec_min_ci: N,
    ) -> Result<(N, N), RayTraceError> {
        self.propagate_internal(prism, wavelength)?
            .intersect_spectrometer(spec, spec_min_ci)
    }

    fn trace(
        self,
        wavelength: N,
        prism: Prism<N>,
        spec: DetectorPositioning<N>,
        spec_min_ci: N,
    ) -> Result<Vec<Pair<N>>, RayTraceError> {
        let mut traced = Vec::new();
        let (ray, n1, vertex) = prism.glasses.iter().zip(prism.angles).try_fold(
            (self, N::one(), Pair::<N>::zero()),
            |(ray, n1, vertex), (glass, angle)| {
                traced.push(ray.origin);
                let n2 = glass.calc_n(wavelength);
                let normal = rotate(*angle, (-N::one(), N::zero()).into());
                let vertex = Pair {
                    x: vertex.x + angle.abs().tan(),
                    y: if vertex.y.is_zero() {
                        N::one()
                    } else {
                        N::zero()
                    },
                };
                let ray = ray.intersect_surface(vertex, normal, n1, n2)?;
                Ok((ray, n2, vertex))
            },
        )?;
        traced.push(ray.origin);
        let angle = prism.angles[prism.glasses.len()];
        let n2 = N::one();
        let normal = rotate(angle, (-N::one(), N::zero()).into());
        let diff = angle.abs().tan();
        let midpt = Pair {
            x: vertex.x + diff * from_f64(0.5),
            y: from_f64(0.5),
        };
        let ray = ray.intersect_lens(midpt, normal, prism.curvature, n1, n2)?;
        traced.push(ray.origin);
        let normal = rotate(spec.angle, (-N::one(), N::zero()).into());
        let ci = -ray.direction.dot(normal);
        if ci <= spec_min_ci {
            return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
        }
        let d = (ray.origin - spec.pos).dot(normal) / ci;
        if d <= N::zero() {
            return Err(RayTraceError::Unknown);
        }
        let p = ray.origin + ray.direction * d;
        traced.push(p);
        Ok(traced)
    }
}

/// pdf((D=d|Λ=λ)|Y=y)
fn pdf_det_l_wavelength_y(
    y: f64,
    wavelength: f64,
    prism: Prism<f64>,
    spec: DetectorPositioning<f64>,
    spec_min_ci: f64,
    beam_width: f64,
    y_mean: f64,
) -> impl Fn(f64, f64) -> f64 {
    let w = prism.width / (2. * prism.height);
    let ray = Ray {
        origin: (0., y).into(),
        direction: (1., 0.).into(),
        transmittance: 1.,
    };
    let y_bar = y - y_mean;
    // circular gaussian beam pdf parameterized by 1/e2 beam width
    // f(x, y) = Exp[-2 (x^2 + y^2) / beam_width^2] * 2 / (pi beam_width^2)
    // g(y) = Integrate[f(x, y), {x, -w, w}]
    // g(y) = Exp[-2 y^2 / beam_width^2] Erf[Sqrt[2] w / beam_width] Sqrt[2 / pi] / beam_width
    const FRAC_SQRT_2_SQRT_PI: f64 =
        core::f64::consts::FRAC_1_SQRT_2 * core::f64::consts::FRAC_2_SQRT_PI;
    let g_y = f64::exp(-2. * y_bar * y_bar / (beam_width * beam_width))
        * libm::erf(core::f64::consts::SQRT_2 * w / beam_width)
        * FRAC_SQRT_2_SQRT_PI
        / beam_width;
    debug_assert!(g_y.is_finite());
    let (pos, t) = ray
        .propagate(wavelength, prism, spec, spec_min_ci)
        .unwrap_or((-1., 0.));
    debug_assert!(pos.is_finite());
    debug_assert!(t.is_finite());
    // pdf((D=d|Λ=λ)|Y=y) = T(λ, y) * g(y) * step(d_l <= S(λ, y) < d_u)
    let pdf = t * g_y;
    move |l, u| if l <= pos && pos < u { pdf } else { 0. }
}

/// I(Λ; D)
fn mutual_information(
    wmin: f64,
    wmax: f64,
    normalized_bounds: &[[f64; 2]],
    prism: Prism<f64>,
    spec: DetectorPositioning<f64>,
    spec_min_ci: f64,
    beam_width: f64,
    y_mean: f64,
) -> f64 {
    let nbins = normalized_bounds.len();
    let p_w = 1. / (wmax - wmin);
    let mut info = 0_f64;
    let mut p_dets = vec![0_f64; nbins];
    // p(D=d) = Integrate[p(D=d|Λ=λ), {λ, wmin, wmax}]
    KR21::inplace_integrate(
        |w, w_factor| {
            let mut p_det_l_ws = vec![0_f64; nbins];
            // p(D=d|Λ=λ) = Integrate[p((D=d|Λ=λ)|Y=y), {y, 0, 1}]
            KR21::inplace_integrate(
                |y, y_factor| {
                    let f =
                        pdf_det_l_wavelength_y(y, w, prism, spec, spec_min_ci, beam_width, y_mean);
                    for (p_det_l_w, &[l, u]) in p_det_l_ws.iter_mut().zip(normalized_bounds) {
                        *p_det_l_w += f(l, u) * y_factor;
                    }
                },
                0.,
                1.,
                10,
            );
            for (p_det, p_det_l_w) in p_dets.iter_mut().zip(p_det_l_ws) {
                debug_assert!(0. <= p_det_l_w && p_det_l_w <= 1.);
                if p_det_l_w > 0. {
                    *p_det += p_w * p_det_l_w * w_factor;
                    info += p_w * p_det_l_w * p_det_l_w.log2() * w_factor;
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

pub fn get_spectrometer_position(
    prism: Prism<f64>,
    pmts: PmtArray<f64>,
    beam: GaussianBeam<f64>,
) -> Result<DetectorPositioning<f64>, RayTraceError> {
    let (wmin, wmax) = beam.w_range;
    let spec_length = pmts.length / prism.height;
    Ray::spectrometer_position(beam.y_mean, wmin, wmax, prism, pmts.angle, spec_length)
}

pub fn trace(
    wavelength: f64,
    init_y: f64,
    prism: Prism<f64>,
    pmts: PmtArray<f64>,
    beam: GaussianBeam<f64>,
) -> Result<Vec<(f64, f64)>, RayTraceError> {
    let (wmin, wmax) = beam.w_range;
    let spec_length = pmts.length / prism.height;
    let ray = Ray {
        origin: (0., init_y).into(),
        direction: (1., 0.).into(),
        transmittance: 1.,
    };
    let spec = Ray::spectrometer_position(beam.y_mean, wmin, wmax, prism, pmts.angle, spec_length)?;
    let traced = ray.trace(wavelength, prism, spec, pmts.min_ci)?;
    Ok(traced.into_iter().map(|v| (v.x, v.y)).collect())
}

pub fn transmission(
    wavelengths: &[f64],
    prism: Prism<f64>,
    pmts: PmtArray<f64>,
    beam: GaussianBeam<f64>,
) -> Result<Vec<Vec<f64>>, RayTraceError> {
    let spec_length = pmts.length / prism.height;
    let beam_width = beam.width / prism.height;
    let (wmin, wmax) = beam.w_range;
    let spec = Ray::spectrometer_position(beam.y_mean, wmin, wmax, prism, pmts.angle, spec_length)?;
    let mut ts = vec![vec![0_f64; wavelengths.len()]; pmts.bins.len()];
    for (w_idx, w) in wavelengths.iter().cloned().enumerate() {
        KR21::inplace_integrate(
            |y, factor| {
                let f =
                    pdf_det_l_wavelength_y(y, w, prism, spec, pmts.min_ci, beam_width, beam.y_mean);
                for (b_idx, &[l, u]) in pmts.bins.iter().enumerate() {
                    let p = f(l, u);
                    ts[b_idx][w_idx] += p * factor;
                }
            },
            0.,
            1.,
            10,
        );
    }
    Ok(ts)
}

pub fn merit(
    prism: Prism<f64>,
    pmts: PmtArray<f64>,
    beam: GaussianBeam<f64>,
) -> Result<[f64; 3], RayTraceError> {
    let (wmin, wmax) = beam.w_range;
    let spec_length = pmts.length / prism.height;
    let beam_width = beam.width / prism.height;
    let spec = Ray::spectrometer_position(beam.y_mean, wmin, wmax, prism, pmts.angle, spec_length)?;
    let deviation_vector = spec.pos + spec.dir / 2.
        - Pair {
            x: 0.,
            y: beam.y_mean,
        };
    let size = deviation_vector.norm() * prism.height;
    let deviation = deviation_vector.y.abs() / deviation_vector.norm();
    let info = mutual_information(
        wmin,
        wmax,
        pmts.bins,
        prism,
        spec,
        pmts.min_ci,
        beam_width,
        beam.y_mean,
    );
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
    fn test_merit() {
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
        let prism = Prism {
            glasses: &glasses,
            angles: &angles,
            curvature: 0.21,
            height: 2.5,
            width: 2.,
        };

        let nbin = 32;
        let bounds: Box<[_]> = (0..=nbin).map(|i| f64::from(i) / f64::from(nbin)).collect();
        let bins: Box<[_]> = bounds.windows(2).map(|t| [t[0], t[1]]).collect();
        let spec_max_accepted_angle = (60_f64).to_radians();
        let pmts = PmtArray {
            bins: &bins,
            min_ci: spec_max_accepted_angle.cos(),
            angle: 0.,
            length: 3.2,
        };

        let beam = GaussianBeam {
            width: 0.2,
            y_mean: 0.38,
            w_range: (0.5, 0.82),
        };

        let v = merit(prism, pmts, beam).expect("Merit function failed");
        assert!(
            approx_eq(v[0], 41.324065257329245, 1e-3),
            "Size is incorrect. {} ≉ -41.324",
            v[0]
        );
        assert!(
            approx_eq(v[1], -1.444212905142612, 1e-3),
            "Mutual information is incorrect. {} ≉ -1.444",
            v[1]
        );
        assert!(
            approx_eq(v[2], 0.37715870072898755, 1e-3),
            "Deviation is incorrect. {} ≉ -0.377",
            v[2]
        );
    }
}
