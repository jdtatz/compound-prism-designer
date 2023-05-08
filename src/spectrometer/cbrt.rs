#![doc = r#"\
Fast Calculation of Cube and Inverse Cube Roots Using a Magic Constant and Its Implementation on Microcontrollers
Leonid Moroz, Volodymyr Samotyy, Cezary J. Walczyk, and Jan L. Cieśliński
"#]
#![allow(clippy::many_single_char_names)]
use crate::FloatExt;

#[inline(always)]
fn bit_magic(x: f32) -> f32 {
    f32::from_bits(0x548c2b4b - f32::to_bits(x) / 3)
}

#[cfg(test)]
pub fn inv_cbrt_f32(x: f32) -> f32 {
    // Algorithm 5.
    const K1: f32 = 1.752319676_f32;
    const K2: f32 = 1.2509524245_f32;
    const K3: f32 = 0.5093818292_f32;
    let s = x.is_sign_positive();
    let x = x.abs();
    let y = bit_magic(x);
    let c = x * y * y * y;
    // let y = c.mul_add(-K3, K2).mul_add(-c, K1) * y;
    let y = y * (K1 - c * (K2 - K3 * c));
    let c = 1_f32 - x * y * y * y;
    let res = y * (1_f32 + 0.333333333333_f32 * c);
    // let res = c.mul_add(0.333333333333_f32, 1_f32) * y;
    if s { res } else { -res }
}

#[cfg(test)]
pub fn cbrt_f32(x: f32) -> f32 {
    // Algorithm 6.
    const K1: f32 = 1.752319676_f32;
    const K2: f32 = 1.2509524245_f32;
    const K3: f32 = 0.5093818292_f32;
    let s = x.is_sign_positive();
    let x = x.abs();
    let y = bit_magic(x);
    let c = x * y * y * y;
    // let y = c.mul_add(-K3, K2).mul_add(-c, K1) * y;
    let y = y * (K1 - c * (K2 - K3 * c));
    let d = x * y * y;
    let c = 1_f32 - d * y;
    let res = d * (1_f32 + 0.333333333333_f32 * c);
    if s { res } else { -res }
}

pub fn approx_cbrt<F: FloatExt>(x: F) -> F {
    // Algorithm 6.
    const K1: f32 = 1.752319676_f32;
    const K2: f32 = 1.2509524245_f32;
    const K3: f32 = 0.5093818292_f32;
    let k1 = F::lossy_from(K1);
    let k2 = F::lossy_from(K2);
    let k3 = F::lossy_from(K3);
    let one_third = F::lossy_from(0.333333333333_f32);
    let s = x.is_sign_positive();
    let x = x.abs();
    let y = F::lossy_from(bit_magic(x.lossy_into()));
    let c = x * y * y * y;
    // let y = c.mul_add(-K3, K2).mul_add(-c, K1) * y;
    let y = y * (k1 - c * (k2 - k3 * c));
    let d = x * y * y;
    let c = F::ONE - d * y;
    let res = d * (F::ONE + one_third * c);
    if s { res } else { -res }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use super::{approx_cbrt, cbrt_f32, inv_cbrt_f32};

    #[test]
    fn test_inv_cbrt_f32() {
        // const MAX_REL_ERR: f32 = 1.3301e-7_f32;
        const MAX_REL_ERR: f32 = 2.385e-7_f32;
        // let mut v = 1e-45_f32;
        let mut v = 1e-38_f32;
        while v < 128_f32 {
            v = libm::nextafterf(v, core::f32::INFINITY);
            let f_v = inv_cbrt_f32(v);
            let error = (f_v * (libm::cbrt(v as _) as f32)) - 1_f32;
            assert!(
                error.abs() < MAX_REL_ERR,
                "inv_cbrt({}) ≉ {}; error = {:e}",
                v,
                f_v,
                error
            );
        }
    }

    #[test]
    fn test_cbrt_f32() {
        const MAX_REL_ERR: f32 = 3e-5_f32;
        let mut v = 1e-5f32;
        while v < 1024_f32 {
            v = libm::nextafterf(v, core::f32::INFINITY);
            assert_float_eq!(
                cbrt_f32(v),
                libm::cbrt(v as f64) as f32,
                rmax <= MAX_REL_ERR,
                "approx_cbrt(v) ≉ true_cbrt(v) for v = {}",
                v
            );
            assert_eq!(cbrt_f32(v), approx_cbrt(v));
        }
    }
}
