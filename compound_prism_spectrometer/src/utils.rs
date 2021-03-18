#![allow(clippy::needless_return)]
use arrayvec::{Array, ArrayVec};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

pub trait LossyFrom<T>: Sized {
    fn lossy_from(_: T) -> Self;
}

pub trait LossyInto<T>: Sized {
    fn lossy_into(self) -> T;
}

impl<T, U: LossyFrom<T>> LossyInto<U> for T {
    fn lossy_into(self) -> U {
        LossyFrom::lossy_from(self)
    }
}

impl LossyFrom<f32> for f32 {
    fn lossy_from(v: f32) -> Self {
        v
    }
}

impl LossyFrom<f64> for f64 {
    fn lossy_from(v: f64) -> Self {
        v
    }
}

impl LossyFrom<f32> for f64 {
    fn lossy_from(v: f32) -> Self {
        v as f64
    }
}

impl LossyFrom<f64> for f32 {
    fn lossy_from(v: f64) -> Self {
        v as f32
    }
}

// impl<T, U: LossyFrom<T>, const N: usize> LossyFrom<[T; N]> for [U; N] {
//     fn lossy_from(t: [T; N]) -> Self {
//         t.map(LossyFrom::lossy_from)
//     }
// }

impl<T, U: LossyFrom<T>, const N: usize> LossyFrom<[T; N]> for [U; N]
where
    [T; N]: Array<Item = T>,
    [U; N]: Array<Item = U>,
{
    fn lossy_from(t: [T; N]) -> Self {
        ArrayVec::from(t)
            .into_iter()
            .map(LossyFrom::lossy_from)
            .collect::<ArrayVec<_>>()
            .into_inner()
            .unwrap_or_else(|_| unreachable!("How did this happend?"))
    }
}

impl<TA: Array, UA: Array> LossyFrom<ArrayVec<TA>> for ArrayVec<UA>
where
    UA::Item: LossyFrom<TA::Item>,
{
    fn lossy_from(a: ArrayVec<TA>) -> Self {
        a.into_iter().map(LossyFrom::lossy_from).collect()
    }
}

impl<T0, U0: LossyFrom<T0>> LossyFrom<(T0,)> for (U0,) {
    fn lossy_from(t: (T0,)) -> Self {
        (LossyFrom::lossy_from(t.0),)
    }
}

impl<T0, T1, U0: LossyFrom<T0>, U1: LossyFrom<T1>> LossyFrom<(T0, T1)> for (U0, U1) {
    fn lossy_from(t: (T0, T1)) -> Self {
        (LossyFrom::lossy_from(t.0), LossyFrom::lossy_from(t.1))
    }
}

pub trait Float:
    'static
    + Sized
    + Copy
    + Clone
    + core::fmt::Debug
    + core::fmt::Display
    + core::fmt::LowerExp
    + serde::Serialize
    + serde::de::DeserializeOwned
    + PartialEq
    + PartialOrd
    + Neg<Output = Self>
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Rem<Output = Self>
    + RemAssign
{
    fn from_f64(v: f64) -> Self;
    fn from_u32(v: u32) -> Self;
    fn from_u32_ratio(n: u32, d: u32) -> Self;
    fn to_f64(self) -> f64;
    fn to_u32(self) -> u32;
    fn zero() -> Self;
    fn one() -> Self;
    fn infinity() -> Self;
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_sign_positive(self) -> bool;
    fn copy_sign(self, sign: Self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqr(self) -> Self {
        self * self
    }
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn trunc(self) -> Self;
    fn fract(self) -> Self;
    fn euclid_div_rem(self, rhs: Self) -> (Self, Self) {
        let q = (self / rhs).trunc();
        let r = self - q * rhs;
        if r < Self::zero() {
            (
                if rhs > Self::zero() {
                    q - Self::one()
                } else {
                    q + Self::one()
                },
                r + rhs.abs(),
            )
        } else {
            (q, r)
        }
    }
    fn abs(self) -> Self;
    fn sincos(self) -> (Self, Self);
    fn tan(self) -> Self;
    fn asin(self) -> Self;
    fn erf(self) -> Self;
    fn floor(self) -> Self;
    fn min(self, other: Self) -> Self {
        if self <= other {
            self
        } else {
            other
        }
    }
    fn max(self, other: Self) -> Self {
        if self >= other {
            self
        } else {
            other
        }
    }
    fn plog2p(self) -> Self {
        if self > Self::zero() {
            self * self.log2()
        } else {
            Self::zero()
        }
    }
}

macro_rules! cuda_specific {
    ($cuda_expr:expr , $cpu_expr:expr) => {
        #[cfg(target_arch = "nvptx64")]
        {
            #[allow(unused_unsafe)]
            return unsafe { $cuda_expr };
        };
        #[cfg(not(target_arch = "nvptx64"))]
        {
            return $cpu_expr;
        };
    };
}

impl Float for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }

    fn from_u32(v: u32) -> Self {
        v as f32
    }

    fn from_u32_ratio(n: u32, d: u32) -> Self {
        (n as f32) / (d as f32)
    }

    fn to_f64(self) -> f64 {
        self as f64
    }

    fn to_u32(self) -> u32 {
        self as u32
    }

    fn zero() -> Self {
        0_f32
    }

    fn one() -> Self {
        1_f32
    }

    fn infinity() -> Self {
        core::f32::INFINITY
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn is_infinite(self) -> bool {
        self.is_infinite()
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn is_sign_positive(self) -> bool {
        self.is_sign_positive()
    }

    fn copy_sign(self, sign: Self) -> Self {
        cuda_specific!(
            core::intrinsics::copysignf32(self, sign),
            self.copysign(sign)
        );
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        cuda_specific!(core::intrinsics::fmaf32(self, a, b), self.mul_add(a, b));
    }

    fn sqrt(self) -> Self {
        cuda_specific!(core::intrinsics::sqrtf32(self), self.sqrt());
    }

    fn exp(self) -> Self {
        cuda_specific!(libm::expf(self), self.exp());
    }

    fn ln(self) -> Self {
        libm::logf(self)
    }

    fn log2(self) -> Self {
        libm::log2f(self)
    }

    fn trunc(self) -> Self {
        cuda_specific!(core::intrinsics::truncf32(self), self.trunc());
    }

    fn fract(self) -> Self {
        cuda_specific!(self - core::intrinsics::truncf32(self), self.fract());
    }

    fn abs(self) -> Self {
        cuda_specific!(core::intrinsics::fabsf32(self), self.abs());
    }

    fn sincos(self) -> (Self, Self) {
        libm::sincosf(self)
    }

    fn tan(self) -> Self {
        cuda_specific!(libm::tanf(self), self.tan());
    }

    fn asin(self) -> Self {
        cuda_specific!(libm::asinf(self), self.asin());
    }

    fn erf(self) -> Self {
        libm::erff(self)
    }

    fn floor(self) -> Self {
        cuda_specific!(core::intrinsics::floorf32(self), self.floor());
    }
}

impl Float for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }

    fn from_u32(v: u32) -> Self {
        v as f64
    }

    fn from_u32_ratio(n: u32, d: u32) -> Self {
        (n as f64) / (d as f64)
    }

    fn to_f64(self) -> f64 {
        self
    }

    fn to_u32(self) -> u32 {
        self as u32
    }

    fn zero() -> Self {
        0_f64
    }

    fn one() -> Self {
        1_f64
    }

    fn infinity() -> Self {
        core::f64::INFINITY
    }

    fn is_finite(self) -> bool {
        self.is_finite()
    }

    fn is_infinite(self) -> bool {
        self.is_infinite()
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn is_sign_positive(self) -> bool {
        self.is_sign_positive()
    }

    fn copy_sign(self, sign: Self) -> Self {
        cuda_specific!(
            core::intrinsics::copysignf64(self, sign),
            self.copysign(sign)
        );
    }

    fn mul_add(self, a: Self, b: Self) -> Self {
        cuda_specific!(core::intrinsics::fmaf64(self, a, b), self.mul_add(a, b));
    }

    fn sqrt(self) -> Self {
        cuda_specific!(core::intrinsics::sqrtf64(self), self.sqrt());
    }

    fn exp(self) -> Self {
        cuda_specific!(libm::exp(self), self.exp());
    }

    fn ln(self) -> Self {
        libm::log(self)
    }

    fn log2(self) -> Self {
        libm::log2(self)
    }

    fn trunc(self) -> Self {
        cuda_specific!(core::intrinsics::truncf64(self), self.trunc());
    }

    fn fract(self) -> Self {
        cuda_specific!(self - core::intrinsics::truncf64(self), self.fract());
    }

    fn abs(self) -> Self {
        cuda_specific!(core::intrinsics::fabsf64(self), self.abs());
    }

    fn sincos(self) -> (Self, Self) {
        libm::sincos(self)
    }

    fn tan(self) -> Self {
        cuda_specific!(libm::tan(self), self.tan());
    }

    fn asin(self) -> Self {
        cuda_specific!(libm::asin(self), self.asin());
    }

    fn erf(self) -> Self {
        libm::erf(self)
    }

    fn floor(self) -> Self {
        cuda_specific!(core::intrinsics::floorf64(self), self.floor());
    }
}

pub fn almost_eq<F: Float>(a: F, b: F, acc: F) -> bool {
    // only true if a and b are infinite with same
    // sign
    if a.is_infinite() && b.is_infinite() {
        return a.is_sign_positive() == b.is_sign_positive();
    }
    // NANs are never equal
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() < acc
}

#[macro_export]
macro_rules! assert_almost_eq {
    ($a:expr, $b:expr, $prec:expr) => {
        if !$crate::utils::almost_eq($a, $b, $prec) {
            panic!(
                "assertion failed: `abs(left - right) < {:e}`, (left: `{}`, right: `{}`), abs(left - right) = {:.2e}",
                $prec, $a, $b, ($a - $b).abs()
            );
        }
    };
    ($a:expr, $b:expr) => {
        $crate::assert_almost_eq!($a, $b, core::f64::EPSILON)
    };
}

#[macro_export]
macro_rules! debug_assert_almost_eq {
    ($($inner:tt)*) => {
        #[cfg(debug_assertions)] {
            $crate::assert_almost_eq!($($inner)*)
        }
    };
}
