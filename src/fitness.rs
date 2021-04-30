use compound_prism_spectrometer::*;

// const MAX_N: usize = 100_000;

fn vector_quasi_monte_carlo_integration<V, Q, F>(
    max_n: usize,
    max_err: V,
    vec_len: usize,
    vector_fn: F,
) -> Vec<Welford<V>>
where
    V: FloatExt,
    Q: QuasiRandom<Scalar = V>,
    F: Fn(Q) -> Option<(usize, V)>,
{
    let max_err_squared = max_err * max_err;
    let mut count = V::zero();
    let mut stats = vec![Welford::new(); vec_len];
    let qrng = Qrng::new_from_scalar(V::lossy_from(0.5f64));
    for u in qrng.take(max_n) {
        count += V::one();
        if let Some((idx, v)) = vector_fn(u) {
            stats[idx].skip(count);
            stats[idx].next_sample(v);
            if stats[idx].sem_le_error_threshold(max_err_squared) {
                break;
            }
        }
    }
    for w in stats.iter_mut() {
        w.skip(count);
    }
    stats
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct DesignFitness<F> {
    pub size: F,
    pub info: F,
    pub deviation: F,
}

/// Conditional Probability of detection per detector given a `wavelength`
/// { p(D=d|Λ=λ) : d in D }
///
/// # Arguments
///  * `wavelength` - given wavelength
pub fn p_dets_l_wavelength<
    F: FloatExt,
    W: Distribution<F, Output = F>,
    B: Distribution<F, Output = Ray<F, D>>,
    S0: Copy + Surface<F, D>,
    SI: Copy + Surface<F, D>,
    SN: Copy + Surface<F, D>,
    const N: usize,
    const D: usize,
>(
    spectrometer: &Spectrometer<F, W, B, S0, SI, SN, N, D>,
    wavelength: F,
    max_n: usize,
) -> impl Iterator<Item = F> {
    let nbin = spectrometer.detector.bin_count() as usize;
    vector_quasi_monte_carlo_integration(max_n, F::lossy_from(5e-3f64), nbin, move |q: F| {
        // Inverse transform sampling-method
        let ray = spectrometer.beam.inverse_cdf(q);
        if let Ok((bin_idx, t)) = ray.propagate(
            wavelength,
            &spectrometer.compound_prism,
            &spectrometer.detector,
        ) {
            debug_assert!(t.is_finite());
            debug_assert!(F::zero() <= t && t <= F::one());
            // What is actually being integrated is
            // pdf_t = t * pdf(y, z);
            // But because of importance sampling using the same distribution
            // pdf_t /= pdf(y, z);
            // the pdf(y, z) is cancelled, so.
            // pdf_t = t;
            let pdf_t = t;
            Some((bin_idx as usize, pdf_t))
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
pub fn mutual_information<
    F: FloatExt,
    W: Distribution<F, Output = F>,
    B: Distribution<F, Output = Ray<F, D>>,
    S0: Copy + Surface<F, D>,
    SI: Copy + Surface<F, D>,
    SN: Copy + Surface<F, D>,
    const N: usize,
    const D: usize,
>(
    spectrometer: &Spectrometer<F, W, B, S0, SI, SN, N, D>,
    max_n: usize,
    max_m: usize,
) -> F {
    let nbin = spectrometer.detector.bin_count() as usize;
    // p(d=D)
    let mut p_dets = vec![Welford::new(); nbin];
    // -H(D|Λ)
    let mut h_det_l_w = Welford::new();
    let qrng = Qrng::new(F::lossy_from(0.5f64));
    for u in qrng.take(max_n) {
        // Inverse transform sampling-method: U[0, 1) => U[wmin, wmax)
        let w = spectrometer.wavelengths.inverse_cdf(u);
        // p(d=D|λ=Λ)
        let p_dets_l_w = p_dets_l_wavelength(spectrometer, w, max_m);
        // -H(D|λ=Λ)
        let mut h_det_l_ws = F::zero();
        for (dstat, p_det_l_w) in p_dets.iter_mut().zip(p_dets_l_w) {
            debug_assert!(F::zero() <= p_det_l_w && p_det_l_w <= F::one());
            dstat.next_sample(p_det_l_w);
            h_det_l_ws += p_det_l_w.plog2p();
        }
        h_det_l_w.next_sample(h_det_l_ws);
        if p_dets.iter().all(|stat| {
            const MAX_ERR: f64 = 5e-3;
            const MAX_ERR_SQR: f64 = MAX_ERR * MAX_ERR;
            stat.sem_le_error_threshold(F::lossy_from(MAX_ERR_SQR))
        }) {
            break;
        }
    }
    // -H(D)
    let h_det = p_dets
        .iter()
        .map(|s| s.mean.plog2p())
        .fold(F::zero(), core::ops::Add::add);
    // I(Λ; D) = H(D) - H(D|Λ)
    h_det_l_w.mean - h_det
}

/// Return the fitness of the spectrometer design to be minimized by an optimizer.
/// The fitness objectives are
/// * size = the distance from the mean starting position of the beam to the center of detector array
/// * info = I(Λ; D)
/// * deviation = sin(abs(angle of deviation))
pub fn fitness<
    F: FloatExt,
    W: Copy + Distribution<F, Output = F>,
    B: Copy + Distribution<F, Output = Ray<F, D>>,
    S0: Copy + Surface<F, D>,
    SI: Copy + Surface<F, D>,
    SN: Copy + Surface<F, D>,
    const N: usize,
    const D: usize,
>(
    spectrometer: &Spectrometer<F, W, B, S0, SI, SN, N, D>,
    max_n: usize,
    max_m: usize,
) -> DesignFitness<F> {
    let (size, deviation) = spectrometer.size_and_deviation();
    let info = mutual_information(spectrometer, max_n, max_m);
    DesignFitness {
        size,
        info,
        deviation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_known_prism() {
        let glasses = [
            // N-PK52A
            Glass {
                coefficents: [
                    -0.19660238,
                    0.85166448,
                    -1.49929414,
                    1.35438084,
                    -0.64424681,
                    1.62434799,
                ],
            },
            // N-SF57
            Glass {
                coefficents: [
                    -1.81746234,
                    7.71730927,
                    -13.2402884,
                    11.56821078,
                    -5.23836004,
                    2.82403194,
                ],
            },
            // N-FK58
            Glass {
                coefficents: [
                    -0.15938247,
                    0.69081086,
                    -1.21697038,
                    1.10021121,
                    -0.52409733,
                    1.55979703,
                ],
            },
        ];
        let [glass0, glasses @ ..] = glasses;
        let angles = [-27.2712308, 34.16326141, -42.93207009, 1.06311416];
        let angles = angles.map(f64::to_radians);
        let [first_angle, angles @ .., last_angle] = angles;
        let lengths = [0_f64; 3];
        let [first_length, lengths @ ..] = lengths;
        let prism = CompoundPrism::new(
            glass0,
            glasses,
            first_angle,
            angles,
            last_angle,
            first_length,
            lengths,
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

        let wavelengths = UniformDistribution {
            bounds: (0.5, 0.82),
        };
        let beam = GaussianBeam {
            width: 0.2,
            y_mean: 0.95,
            marker: core::marker::PhantomData,
        };
        // dbg!((&wavelengths, &beam, &prism, &detarr));
        let spec = Spectrometer::new(wavelengths, beam, prism, detarr)
            .expect("This is a valid spectrometer design.");

        let DesignFitness {
            size,
            info,
            deviation,
        } = fitness(&spec, 16_384, 16_384);
        float_eq::assert_float_eq!(size, 41.3, rmax <= 5e-3);
        float_eq::assert_float_eq!(info, 0.7407, rmax <= 1e-2);
        float_eq::assert_float_eq!(deviation, 0.377159, rmax <= 1e-3);
    }
}
