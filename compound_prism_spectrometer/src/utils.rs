#![allow(clippy::needless_return)]
use core::mem::swap;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

pub fn array_prepend<T, const N: usize>(first: T, mut rest: [T; N]) -> ([T; N], T) {
    let mut swapped = first;
    for i in 0..N {
        swap(&mut rest[i], &mut swapped);
    }
    let last = swapped;
    (rest, last)
}

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

// For once `specialization` is stabilized
// impl<T, U: From<T>> LossyFrom<T> for U {
//     default fn lossy_from(v: T) -> Self {
//         From::from(v)
//     }
// }

macro_rules! primative_lossy_conv {
    ( $($t:ty),+ ) => {
        primative_lossy_conv! { { $($t),+ } => { $($t),+ } }
    };
    ( { $($from:ty),+ } => $right:tt ) => {
        $( primative_lossy_conv! { $from => $right } )+
    };
    ( $from:ty => { $($to:ty),+ } ) => {
        $( primative_lossy_conv! { $from => $to } )+
    };
    ( $from:ty => $to:ty ) => {
        impl LossyFrom<$from> for $to {
            fn lossy_from(v: $from) -> Self {
                v as $to
            }
        }
    };
}

primative_lossy_conv! { f32, f64, u32, i32 }

impl<T, U: LossyFrom<T>, const N: usize> LossyFrom<[T; N]> for [U; N] {
    fn lossy_from(t: [T; N]) -> Self {
        t.map(LossyFrom::lossy_from)
    }
}

wrapped_from_tuples! { LossyFrom::lossy_from for 0..12 }

pub trait Float:
    'static
    + Sized
    + Copy
    + Clone
    + LossyFrom<u32>
    + LossyInto<u32>
    + LossyFrom<f64>
    + LossyInto<f64>
    + core::fmt::Debug
    + core::fmt::Display
    + core::fmt::LowerExp
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
    fn from_f64(v: f64) -> Self {
        Self::lossy_from(v)
    }
    fn from_u32(v: u32) -> Self {
        Self::lossy_from(v)
    }
    fn from_u32_ratio(n: u32, d: u32) -> Self {
        Self::lossy_from(n) / Self::lossy_from(d)
    }
    fn to_f64(self) -> f64 {
        self.lossy_into()
    }
    fn to_u32(self) -> u32 {
        self.lossy_into()
    }
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
