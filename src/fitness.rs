use crate::geom::Vector;
use crate::qrng::*;
use crate::ray::*;
use crate::utils::*;

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
    let qrng = Qrng::<V>::new(V::from_u32_ratio(1, 2));
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

#[derive(PartialEq, Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(bound = "F: Float")]
pub struct DesignFitness<F: Float> {
    pub size: F,
    pub info: F,
    pub deviation: F,
}

impl<V: Vector> Spectrometer<V> {
    /// Conditional Probability of detection per detector given a `wavelength`
    /// { p(D=d|Λ=λ) : d in D }
    ///
    /// # Arguments
    ///  * `wavelength` - given wavelength
    pub fn p_dets_l_wavelength(&self, wavelength: V::Scalar) -> impl Iterator<Item = V::Scalar> {
        let p_z = self.probability_z_in_bounds();
        debug_assert!(V::Scalar::zero() <= p_z && p_z <= V::Scalar::one());
        let nbin = self.detector_array.bin_count as usize;
        vector_quasi_monte_carlo_integration(
            V::Scalar::from_u32_ratio(5, 1000),
            nbin,
            move |u: V::Scalar| {
                // Inverse transform sampling-method: U[0, 1) => N(µ = beam.y_mean, σ = beam.width / 2)
                let y = self.gaussian_beam.inverse_cdf_initial_y(u);
                if y <= V::Scalar::zero() || self.compound_prism.height <= y {
                    return None;
                }
                if let Ok((bin_idx, t)) = self.propagate(wavelength, y) {
                    debug_assert!(t.is_finite());
                    debug_assert!(V::Scalar::zero() <= t && t <= V::Scalar::one());
                    // What is actually being integrated is
                    // pdf_t = p_z * t * pdf(y);
                    // But because of importance sampling using the same distribution
                    // pdf_t /= pdf(y);
                    // the pdf(y) is cancelled, so.
                    // pdf_t = p_z * t;
                    let pdf_t = p_z * t;
                    Some((0..self.detector_array.bin_count).map(move |i| {
                        if i == bin_idx {
                            pdf_t
                        } else {
                            V::Scalar::zero()
                        }
                    }))
                } else {
                    None
                }
            },
        )
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
    pub fn mutual_information(&self) -> V::Scalar {
        let nbin = self.detector_array.bin_count as usize;
        // p(d=D)
        let mut p_dets = vec![Welford::new(); nbin];
        // -H(D|Λ)
        let mut h_det_l_w = Welford::new();
        let qrng = Qrng::new(V::Scalar::from_u32_ratio(1, 2));
        for u in qrng.take(MAX_N) {
            // Inverse transform sampling-method: U[0, 1) => U[wmin, wmax)
            let w = self.gaussian_beam.inverse_cdf_wavelength(u);
            // p(d=D|λ=Λ)
            let p_dets_l_w = self.p_dets_l_wavelength(w);
            // -H(D|λ=Λ)
            let mut h_det_l_ws = V::Scalar::zero();
            for (dstat, p_det_l_w) in p_dets.iter_mut().zip(p_dets_l_w) {
                debug_assert!(V::Scalar::zero() <= p_det_l_w && p_det_l_w <= V::Scalar::one());
                dstat.next_sample(p_det_l_w);
                h_det_l_ws += p_det_l_w.plog2p();
            }
            h_det_l_w.next_sample(h_det_l_ws);
            if p_dets.iter().all(|stat| {
                const MAX_ERR_N: u32 = 5;
                const MAX_ERR_D: u32 = 1000;
                stat.sem_le_error_threshold(V::Scalar::from_u32_ratio(
                    MAX_ERR_N * MAX_ERR_N,
                    MAX_ERR_D * MAX_ERR_D,
                ))
            }) {
                break;
            }
        }
        // -H(D)
        let h_det = p_dets
            .iter()
            .map(|s| s.mean.plog2p())
            .fold(V::Scalar::zero(), core::ops::Add::add);
        // I(Λ; D) = H(D) - H(D|Λ)
        h_det_l_w.mean - h_det
    }

    /// Return the fitness of the spectrometer design to be minimized by an optimizer.
    /// The fitness objectives are
    /// * size = the distance from the mean starting position of the beam to the center of detector array
    /// * info = I(Λ; D)
    /// * deviation = sin(abs(angle of deviation))
    pub fn fitness(&self) -> DesignFitness<V::Scalar> {
        let (size, deviation) = self.size_and_deviation();
        let info = self.mutual_information();
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
    use crate::geom::Pair;
    use crate::glasscat::{Glass, BUNDLED_CATALOG};
    use crate::utils::almost_eq;
    use rand::prelude::*;
    use std::f64::consts::*;

    #[test]
    fn test_many() {
        let nglass = BUNDLED_CATALOG.len();
        let seed = 123456;
        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(seed);
        let ntest = 500;
        let max_nprism = 6;
        let prism_height = 25.;
        let prism_width = 25.;
        let max_length = 0.5 * prism_height;
        let pmt_length = 3.2;
        const NBIN: usize = 32;
        let spec_max_accepted_angle = (45_f64).to_radians();
        let beam_width = 0.2;
        let wavelegth_range = (0.5, 0.82);
        let mut nvalid = 0;
        while nvalid < ntest {
            let nprism: usize = rng.gen_range(1, 1 + max_nprism);
            let glasses = (0..nprism)
                .map(|_| &BUNDLED_CATALOG[rng.gen_range(0, nglass)].1)
                .cloned()
                .collect::<Vec<_>>();
            let angles = (0..nprism + 1)
                .map(|_| rng.gen_range(-FRAC_PI_2, FRAC_PI_2))
                .collect::<Vec<_>>();
            let lengths = (0..nprism)
                .map(|_| rng.gen_range(0., max_length))
                .collect::<Vec<_>>();
            let curvature = rng.gen_range(0., 1.);
            let prism = CompoundPrism::new(
                glasses.into_iter(),
                angles.as_ref(),
                lengths.as_ref(),
                curvature,
                prism_height,
                prism_width,
                false,
            );

            let detarr_angle = rng.gen_range(-PI, PI);
            let detarr = LinearDetectorArray::new(
                NBIN as u32,
                0.1,
                0.1,
                0.0,
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

            let spec = match Spectrometer::<Pair<_>>::new(beam, prism, detarr) {
                Ok(s) => s,
                Err(_) => continue,
            };

            nvalid += 1;
            let nwlen = 25;
            for i in 0..nwlen {
                let w = wavelegth_range.0
                    + (wavelegth_range.1 - wavelegth_range.0) * ((i as f64) / ((nwlen - 1) as f64));
                let ps = spec.p_dets_l_wavelength(w);
                for p in ps {
                    assert!(p.is_finite() && 0_f64 <= p && p <= 1.);
                }
            }
            let v = spec.fitness();
            assert!(v.size > 0.);
            assert!(0. <= v.info && v.info <= (NBIN as f64).log2());
            assert!(0. <= v.deviation && v.deviation < FRAC_PI_2);
        }
    }

    #[test]
    fn test_with_known_prism() {
        let glasses = [
            // N-PK52A
            Glass::Sellmeier1([
                1.029607,
                0.00516800155,
                0.1880506,
                0.0166658798,
                0.736488165,
                138.964129,
            ]),
            // N-SF57
            Glass::Sellmeier1([
                1.87543831,
                0.0141749518,
                0.37375749,
                0.0640509927,
                2.30001797,
                177.389795,
            ]),
            // N-FK58
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
        let prism = CompoundPrism::<Pair<_>>::new(
            glasses.iter().cloned(),
            angles.as_ref(),
            lengths.as_ref(),
            0.21,
            2.5,
            2.,
            false,
        );

        const NBIN: usize = 32;
        let pmt_length = 3.2;
        let spec_max_accepted_angle = (60_f64).to_radians();
        let detarr = LinearDetectorArray::new(
            NBIN as u32,
            0.1,
            0.1,
            0.0,
            spec_max_accepted_angle.cos(),
            0.,
            pmt_length,
        );

        let beam = GaussianBeam {
            width: 0.2,
            y_mean: 0.95,
            w_range: (0.5, 0.82),
        };

        let spec =
            Spectrometer::new(beam, prism, detarr).expect("This is a valid spectrometer design.");

        let v = spec.fitness();
        assert!(
            almost_eq(v.size, 41.3241, 1e-3),
            "Size is incorrect. {} ≉ 41.3",
            v.size
        );
        assert!(
            almost_eq(v.info, 1.44, 1e-2),
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
