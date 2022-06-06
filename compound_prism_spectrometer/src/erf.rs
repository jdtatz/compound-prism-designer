#![allow(
    clippy::many_single_char_names,
    clippy::unreadable_literal,
    clippy::excessive_precision
)]
use crate::utils::*;

fn polynomial<F: FloatExt, const N: usize>(z: F, coeff: [F; N]) -> F {
    coeff
        .into_iter()
        // coeff
        //     .iter()
        //     .copied()
        .reduce(|s, c| s.mul_add(z, c))
        .expect("`polynomial` called with a zero-sized array")
}

macro_rules! polynomial_eval {
    ($z:ident ; [ $coeff0:expr , $($coeffs:expr),*  $(,)? ] ) => {
        $coeff0 $( .mul_add( $z , $coeffs ) )*
    };
    ($z:ident ; [ $($coeffs:literal),*  $(,)? ] as $t:ty ) => {
        polynomial_eval!( $z ; [ $( <$t as LossyFrom<f64>>::lossy_from($coeffs) ),* ] )
    };
}

const SHAW_P: [f64; 8] = [
    4.3304513840364031401e-8,
    0.000036313870818023761224,
    0.0034424140686962222423,
    0.085838533424158257377,
    0.73176759583280610539,
    2.3884158540184385711,
    3.0333178251950406994,
    1.2533141359896652729,
];

const SHAW_Q: [f64; 8] = [
    1.3141263119543315917e-6,
    0.00030617264753008793976,
    0.014494272424798068406,
    0.2168237095066675527,
    1.2356513216582148689,
    2.9373357991677046357,
    2.9202373175993672857,
    1.0,
];

/// Fast Non-branching Standard Normal inverse CDF
/// To transform into a normal distribution with stddev=a and mean=b
/// x = b +- a * norminv(u)
/// precision is ~1E-9 when 1E-15 <= u <= 1 - 1E-15
/// precision is ~1E-6 when 1E-22 <= u <= 1 - 1E-22
/// precision is ~1E-3 when 1E-30 <= u <= 1 - 1E-30
/// precision is ~1E-2 when 1E-60 <= u <= 1 - 1E-60
/// precision is ~2E-1 when 1E-100 <= u <= 1 - 1E-100
///
/// Source:
/// arXiv:0901.0638 [q-fin.CP]
/// Quantile Mechanics II: Changes of Variables in Monte Carlo methods and GPU-Optimized Normal Quantiles
/// William T. Shaw, Thomas Luu, Nick Brickman
pub fn norminv<F: FloatExt>(x: F) -> F {
    // F::from_f64(-core::f64::consts::SQRT_2 * statrs::function::erf::erfc_inv(2.0 * x.to_f64()))
    let u = if x > F::lossy_from(0.5f64) {
        F::one() - x
    } else {
        x
    };
    let v = -(F::lossy_from(2u32) * u).ln();
    let p = polynomial(v, LossyFrom::lossy_from(SHAW_P));
    let q = polynomial(v, LossyFrom::lossy_from(SHAW_Q));
    (v * p / q).copy_sign(x - F::lossy_from(0.5f64))
}

// const FAST_SHAW_P: [f64; 6] = [
//     1.1051591117060895699e-4,
//     0.011900603295838260268,
//     0.23753954196273241709,
//     1.3348090307272045436,
//     2.4101601285733391215,
//     1.2533141012558299407,
// ];

// const FAST_SHAW_Q: [f64; 7] = [
//     2.5996479253181457637e-6,
//     0.0010579909915338770381,
//     0.046292707412622896113,
//     0.50950202270351517687,
//     1.8481138350821456213,
//     2.4230267574304831865,
//     1.0,
// ];

/// Fast Non-branching Standard Normal inverse CDF
/// To transform into a normal distribution with stddev=a and mean=b
/// x = b +- a * norminv(u)
/// precision is ~2.98E-8 when 2.22E-10 <= u <= 1 - 2.22E-10
/// precision is ~3.1E-8 when 1E-11 <= u <= 1 - 1E-11
/// precision is ~1E-6 when 1E-12 <= u <= 1 - 1E-12
/// precision is ~1E-4 when 1E-17 <= u <= 1 - 1E-16
///
/// Source:
/// arXiv:0901.0638v5 [q-fin.CP]
/// Quantile Mechanics II: Changes of Variables in Monte Carlo methods and GPU-Optimized Normal Quantiles
/// William T. Shaw, Thomas Luu, Nick Brickman
pub fn fast_norminv<F: FloatExt>(u: F) -> F {
    let half_minus_u = F::lossy_from(0.5f64) - u;
    let mut x = (u * F::lossy_from(2u32)).copy_sign(half_minus_u);
    if half_minus_u < F::zero() {
        x += F::lossy_from(2u32);
    }
    let v = -F::ln(x);
    // let p = polynomial(v, LossyFrom::lossy_from(FAST_SHAW_P));
    // let q = polynomial(v, LossyFrom::lossy_from(FAST_SHAW_Q));
    let p = polynomial_eval!(v; [
        1.1051591117060895699e-4,
        0.011900603295838260268,
        0.23753954196273241709,
        1.3348090307272045436,
        2.4101601285733391215,
        1.2533141012558299407,
    ] as F);
    let q = polynomial_eval!(v; [
        2.5996479253181457637e-6_f64,
        0.0010579909915338770381,
        0.046292707412622896113,
        0.50950202270351517687,
        1.8481138350821456213,
        2.4230267574304831865,
        1.0,
    ] as F);
    (-(p / q)) * v.copy_sign(half_minus_u)
}

#[cfg(test)]
mod test {
    use super::*;
    use float_eq::assert_float_eq as assert_almost_eq;
    use statrs::distribution::{ContinuousCDF, Normal};

    #[test]
    fn test_shaw() {
        let n = Normal::new(0.0, 1.0).unwrap();

        assert_almost_eq!(norminv(1e-100), n.inverse_cdf(1e-100), rmax <= 2e-1);
        assert_almost_eq!(norminv(1e-60), n.inverse_cdf(1e-60), rmax <= 2e-2);
        assert_almost_eq!(norminv(1e-30), n.inverse_cdf(1e-30), rmax <= 1e-3);
        assert_almost_eq!(norminv(1e-20), n.inverse_cdf(1e-20), rmax <= 1e-5);
        assert_almost_eq!(norminv(1e-15), n.inverse_cdf(1e-15), rmax <= 5e-9);
        assert_almost_eq!(norminv(1e-10), n.inverse_cdf(1e-10), rmax <= 5e-9);
        assert_almost_eq!(norminv(1e-5), n.inverse_cdf(1e-5), rmax <= 2e-9);
        assert_almost_eq!(norminv(0.1), n.inverse_cdf(0.1), rmax <= 1e-9);
        assert_almost_eq!(norminv(0.2), n.inverse_cdf(0.2), rmax <= 1e-9);
        assert_almost_eq!(norminv(0.5), n.inverse_cdf(0.5), rmax <= 1e-9);
        assert_almost_eq!(norminv(0.7), n.inverse_cdf(0.7), rmax <= 1e-9);
        assert_almost_eq!(norminv(0.9), n.inverse_cdf(0.9), rmax <= 1e-9);
        assert_almost_eq!(norminv(0.99), n.inverse_cdf(0.99), rmax <= 2.04e-9);
        assert_almost_eq!(norminv(0.999), n.inverse_cdf(0.999), rmax <= 2.5e-9);
        assert_almost_eq!(norminv(0.9999), n.inverse_cdf(0.9999), rmax <= 4e-9);
    }

    #[test]
    fn test_fast_shaw() {
        let n = Normal::new(0.0, 1.0).unwrap();

        assert_almost_eq!(fast_norminv(1e-100), n.inverse_cdf(1e-100), rmax <= 1e-1);
        assert_almost_eq!(fast_norminv(1e-60), n.inverse_cdf(1e-60), rmax <= 1e-2);
        assert_almost_eq!(fast_norminv(1e-30), n.inverse_cdf(1e-30), rmax <= 1e-3);
        assert_almost_eq!(fast_norminv(1e-20), n.inverse_cdf(1e-20), rmax <= 1e-4);
        assert_almost_eq!(fast_norminv(1e-15), n.inverse_cdf(1e-15), rmax <= 5e-5);
        assert_almost_eq!(fast_norminv(1e-10), n.inverse_cdf(1e-10), rmax <= 5e-5);
        assert_almost_eq!(fast_norminv(1e-5), n.inverse_cdf(1e-5), rmax <= 2e-8);
        assert_almost_eq!(fast_norminv(0.1), n.inverse_cdf(0.1), rmax <= 3e-8);
        assert_almost_eq!(fast_norminv(0.2), n.inverse_cdf(0.2), rmax <= 3e-8);
        assert_almost_eq!(fast_norminv(0.5), n.inverse_cdf(0.5), rmax <= 3e-8);
        assert_almost_eq!(fast_norminv(0.7), n.inverse_cdf(0.7), rmax <= 3e-8);
        assert_almost_eq!(fast_norminv(0.9), n.inverse_cdf(0.9), rmax <= 3e-8);
        assert_almost_eq!(fast_norminv(0.99), n.inverse_cdf(0.99), rmax <= 3e-8);
        assert_almost_eq!(fast_norminv(0.999), n.inverse_cdf(0.999), rmax <= 3e-8);
        assert_almost_eq!(fast_norminv(0.9999), n.inverse_cdf(0.9999), rmax <= 3e-8);
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-5),
            n.inverse_cdf(1.0 - 1e-5),
            rmax <= 4e-8
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-10),
            n.inverse_cdf(1.0 - 1e-10),
            rmax <= 4e-8
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-11),
            n.inverse_cdf(1.0 - 1e-11),
            rmax <= 3e-7
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-15),
            n.inverse_cdf(1.0 - 1e-15),
            rmax <= 1e-5
        );
        assert_almost_eq!(
            fast_norminv(1.0 - 1e-16),
            n.inverse_cdf(1.0 - 1e-16),
            rmax <= 1e-4
        );
    }
}
