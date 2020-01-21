use crate::erf::erf;
use crate::qrng::*;
use crate::ray::*;
use crate::utils::*;
use core::f64::consts::*;

const MAX_N: usize = 100_000;

fn vector_quasi_monte_carlo_integration<V, I, F>(
    max_err: V,
    vec_len: usize,
    vector_fn: F,
) -> Vec<Welford<V>>
where
    V: Float,
    I: ExactSizeIterator<Item = V>,
    F: Fn(V) -> Option<I>,
{
    let max_err_squared = max_err * max_err;
    let mut stats = vec![Welford::new(); vec_len];
    let qrng = Qrng::<V>::new(V::from_f64(0.5_f64));
    for u in qrng.take(MAX_N) {
        if let Some(vec) = vector_fn(u) {
            for (v, w) in vec.zip(stats.iter_mut()) {
                w.next_sample(v);
            }
        } else {
            for w in stats.iter_mut() {
                w.next_sample(V::zero());
            }
        }
        if stats
            .iter()
            .all(|stat| stat.sem_le_error_threshold(max_err_squared))
        {
            break;
        }
    }
    stats
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
pub fn p_dets_l_wavelength<F: Float>(
    wavelength: F,
    cmpnd: &CompoundPrism<F>,
    detarr: &DetectorArray<F>,
    beam: &GaussianBeam<F>,
    detpos: &DetectorArrayPositioning<F>,
) -> impl Iterator<Item = F> {
    let p_z = F::from_f64(erf(
        cmpnd.width.to_f64() * FRAC_1_SQRT_2 / beam.width.to_f64()
    ));
    debug_assert!(F::zero() <= p_z && p_z <= F::one());
    vector_quasi_monte_carlo_integration(F::from_f64(5e-3), detarr.bins.len(), move |u: F| {
        // Inverse transform sampling-method: U[0, 1) => N(µ = beam.y_mean, σ = beam.width / 2)
        let y = beam.inverse_cdf_initial_y(u);
        if y <= F::zero() || cmpnd.height <= y {
            return None;
        }
        let ray = Ray::new_from_start(y);
        if let Ok((_, pos, t)) = ray.propagate(wavelength, cmpnd, detarr, detpos) {
            debug_assert!(pos.is_finite());
            debug_assert!(F::zero() <= pos && pos <= detarr.length);
            debug_assert!(t.is_finite());
            debug_assert!(F::zero() <= t && t <= F::one());
            // What is actually being integrated is
            // pdf_t = p_z * t * pdf(y);
            // But because of importance sampling using the same distribution
            // pdf_t /= pdf(y);
            // the pdf(y) is cancelled, so.
            // pdf_t = p_z * t;
            let pdf_t = p_z * t;
            Some(detarr.bins.iter().map(move |&[l, u]| {
                if l <= pos && pos < u {
                    pdf_t
                } else {
                    F::zero()
                }
            }))
        } else {
            None
        }
    })
    .into_iter()
    .map(|w| w.mean)
}

/// The mutual information of Λ and D. How much information is gained about Λ by measuring D.
/// I(Λ; D) = H(D) - H(D|Λ)
///   = Sum(Integrate(p(Λ=λ) p(D=d|Λ=λ) log2(p(D=d|Λ=λ)), {λ, wmin, wmax}), d in D)
///      - Sum(p(D=d) log2(p(D=d)), d in D)
/// p(D=d) = Expectation_Λ(p(D=d|Λ=λ)) = Integrate(p(Λ=λ) p(D=d|Λ=λ), {λ, wmin, wmax})
/// p(Λ=λ) = 1 / (wmax - wmin) * step(wmin <= λ <= wmax)
/// H(Λ) is ill-defined because Λ is continuous, but I(Λ; D) is still well-defined for continuous variables.
/// https://en.wikipedia.org/wiki/Differential_entropy#Definition
fn mutual_information<F: Float>(
    cmpnd: &CompoundPrism<F>,
    detarr: &DetectorArray<F>,
    beam: &GaussianBeam<F>,
    detpos: &DetectorArrayPositioning<F>,
) -> F {
    let mut p_dets_stats = vec![Welford::new(); detarr.bins.len()];
    let mut info_stats = vec![Welford::new(); detarr.bins.len()];
    let qrng = Qrng::<F>::new(F::from_f64(0.5_f64));
    for u in qrng.take(MAX_N) {
        // Inverse transform sampling-method: U[0, 1) => U[wmin, wmax)
        let w = beam.inverse_cdf_wavelength(u);
        let p_dets_l_w = p_dets_l_wavelength(w, cmpnd, detarr, beam, detpos);
        for ((dstat, istat), p_det_l_w) in p_dets_stats
            .iter_mut()
            .zip(info_stats.iter_mut())
            .zip(p_dets_l_w)
        {
            debug_assert!(F::zero() <= p_det_l_w && p_det_l_w <= F::one());
            dstat.next_sample(p_det_l_w);
            if p_det_l_w > F::zero() {
                istat.next_sample(p_det_l_w * p_det_l_w.log2());
            } else {
                istat.next_sample(F::zero());
            }
        }
        if p_dets_stats.iter().chain(info_stats.iter()).all(|stat| {
            const MAX_ERR: f64 = 5e-3;
            const MAX_ERR_SQ: f64 = MAX_ERR * MAX_ERR;
            stat.sem_le_error_threshold(F::from_f64(MAX_ERR_SQ))
        }) {
            break;
        }
    }
    let mut info: F = info_stats
        .into_iter()
        .map(|s| s.mean)
        .fold(F::zero(), core::ops::Add::add);
    for stat in p_dets_stats {
        let p_det = stat.mean;
        debug_assert!(F::zero() <= p_det && p_det <= F::one());
        if p_det > F::zero() {
            info -= p_det * p_det.log2();
        }
    }
    info
}

#[derive(PartialEq, Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DesignFitness<F: Float> {
    pub size: F,
    pub info: F,
    pub deviation: F,
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
pub fn fitness<F: Float>(
    cmpnd: &CompoundPrism<F>,
    detarr: &DetectorArray<F>,
    beam: &GaussianBeam<F>,
) -> Result<DesignFitness<F>, RayTraceError> {
    let detpos = detector_array_positioning(cmpnd, detarr, beam)?;
    let deviation_vector = detpos.position + detpos.direction * detarr.length * F::from_f64(0.5)
        - Pair {
            x: F::zero(),
            y: beam.y_mean,
        };
    let size = deviation_vector.norm();
    let deviation = deviation_vector.y.abs() / deviation_vector.norm();
    let info = mutual_information(cmpnd, detarr, beam, &detpos);
    Ok(DesignFitness {
        size,
        info,
        deviation,
    })
}

impl<'a, F: Float> Spectrometer<'a, F> {
    pub fn p_dets_l_wavelength(&self, wavelength: F) -> impl Iterator<Item = F> {
        let p_z = self.probability_z_in_bounds();
        debug_assert!(F::zero() <= p_z && p_z <= F::one());
        let nbin = self.detector_array.bins.len();
        vector_quasi_monte_carlo_integration(F::from_f64(5e-3), nbin, move |u: F| {
            // Inverse transform sampling-method: U[0, 1) => N(µ = beam.y_mean, σ = beam.width / 2)
            let y = self.gaussian_beam.inverse_cdf_initial_y(u);
            if y <= F::zero() || self.compound_prism.height <= y {
                return None;
            }
            if let Ok((pos, t)) = self.propagate(wavelength, y) {
                debug_assert!(pos.is_finite());
                debug_assert!(F::zero() <= pos && pos <= self.detector_array.length);
                debug_assert!(t.is_finite());
                debug_assert!(F::zero() <= t && t <= F::one());
                // What is actually being integrated is
                // pdf_t = p_z * t * pdf(y);
                // But because of importance sampling using the same distribution
                // pdf_t /= pdf(y);
                // the pdf(y) is cancelled, so.
                // pdf_t = p_z * t;
                let pdf_t = p_z * t;
                Some(self.detector_array.bins.iter().map(move |&[l, u]| {
                    if l <= pos && pos < u {
                        pdf_t
                    } else {
                        F::zero()
                    }
                }))
            } else {
                None
            }
        })
            .into_iter()
            .map(|w| w.mean)
    }

    pub fn fitness(&self) -> DesignFitness<F> {
        let (size, deviation) = self.size_and_deviation();
        let info = mutual_information(
            &self.compound_prism,
            &self.detector_array,
            &self.gaussian_beam,
            &self.detector_array_position,
        );
        DesignFitness {
            size,
            info,
            deviation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glasscat::Glass;
    use crate::utils::almost_eq;
    use rand::prelude::*;
    use std::ops::Deref;

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
            // N-PK52A
            &Glass::Sellmeier1([
                1.029607,
                0.00516800155,
                0.1880506,
                0.0166658798,
                0.736488165,
                138.964129,
            ]),
            // N-SF57
            &Glass::Sellmeier1([
                1.87543831,
                0.0141749518,
                0.37375749,
                0.0640509927,
                2.30001797,
                177.389795,
            ]),
            // N-FK58
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
            almost_eq(v.size, 41.3241, 1e-3),
            "Size is incorrect. {} ≉ 41.3",
            v.size
        );
        assert!(
            almost_eq(v.info, 1.444212905142612, 5e-3),
            "Mutual information is incorrect. {} ≉ 1.44",
            v.info
        );
        assert!(
            almost_eq(v.deviation, 0.377159, 1e-3),
            "Deviation is incorrect. {} ≉ 0.377",
            v.deviation
        );
    }
}