#![allow(clippy::many_single_char_names, clippy::unreadable_literal, clippy::excessive_precision)]
use crate::utils::Float;

fn polynomial<F: Float>(z: F, coeff: &[f64]) -> F {
    let n = coeff.len();
    if n == 0 {
        return F::zero();
    }
    let mut sum = F::from_f64(coeff[n - 1]);
    for i in (0..n - 1).rev() {
        sum = sum.mul_add(z, F::from_f64(coeff[i]));
    }
    sum
}

const SHAW_P: &[f64] = &[
    1.2533141359896652729,
    3.0333178251950406994,
    2.3884158540184385711,
    0.73176759583280610539,
    0.085838533424158257377 ,
    0.0034424140686962222423,
    0.000036313870818023761224,
    4.3304513840364031401e-8,
];

const SHAW_Q: &[f64] = &[
    1.0,
    2.9202373175993672857,
    2.9373357991677046357,
    1.2356513216582148689,
    0.2168237095066675527,
    0.014494272424798068406,
    0.00030617264753008793976,
    1.3141263119543315917e-6,
];


/// Fast Non-branching Standard Normal inverse CDF
/// To transform into a normal distribution with stddev=a and mean=b
/// x = b - a * norminv(u)
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
pub fn norminv<F: Float>(x: F) -> F {
    let u = if x > F::from_u32_ratio(1, 2) {
        F::one() - x
    } else {
        x
    };
    let v = -(F::from_u32(2) * u).ln();
    let p = polynomial(v, SHAW_P);
    let q = polynomial(v, SHAW_Q);
    (v * p / q).copy_sign(x - F::from_u32_ratio(1, 2))
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::assert_almost_eq;
    use core::f64::NAN;
    use statrs::distribution::{Normal, InverseCDF};

    #[test]
    fn test_shaw() {
        let n = Normal::new(0.0, 1.0).unwrap();

        assert_almost_eq!(norminv(1e-100), n.inverse_cdf(1e-100), 2e-1);
        assert_almost_eq!(norminv(1e-60), n.inverse_cdf(1e-60), 2e-2);
        assert_almost_eq!(norminv(1e-30), n.inverse_cdf(1e-30), 1e-3);
        assert_almost_eq!(norminv(1e-20), n.inverse_cdf(1e-20), 1e-5);
        assert_almost_eq!(norminv(1e-15), n.inverse_cdf(1e-15), 5e-9);
        assert_almost_eq!(norminv(1e-10), n.inverse_cdf(1e-10), 5e-9);
        assert_almost_eq!(norminv(1e-5), n.inverse_cdf(1e-5), 2e-9);
        assert_almost_eq!(norminv(0.1), n.inverse_cdf(0.1), 1e-9);
        assert_almost_eq!(norminv(0.2), n.inverse_cdf(0.2), 1e-9);
        assert_almost_eq!(norminv(0.5), n.inverse_cdf(0.5), 1e-9);
        assert_almost_eq!(norminv(0.7), n.inverse_cdf(0.7), 1e-9);
        assert_almost_eq!(norminv(0.9), n.inverse_cdf(0.9), 1e-9);
        assert_almost_eq!(norminv(0.99), n.inverse_cdf(0.99), 2.04e-9);
        assert_almost_eq!(norminv(0.999), n.inverse_cdf(0.999), 2.5e-9);
        assert_almost_eq!(norminv(0.9999), n.inverse_cdf(0.9999), 4e-9);
    }
}
