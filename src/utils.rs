#![allow(clippy::needless_return)]
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
#[cfg(not(target_arch = "nvptx64"))]
use serde::{Deserialize, Serialize};

pub trait Float:
    'static
    + Sized
    + Copy
    + core::fmt::Debug
    + core::fmt::Display
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
    fn to_f64(self) -> f64;
    fn zero() -> Self;
    fn one() -> Self;
    fn infinity() -> Self;
    fn is_finite(self) -> bool;
    fn is_infinite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_sign_positive(self) -> bool;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqr(self) -> Self {
        self * self
    }
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn log2(self) -> Self;
    fn powf(self, n: Self) -> Self;
    fn fract(self) -> Self;
    fn abs(self) -> Self;
    fn sincos(self) -> (Self, Self);
    fn tan(self) -> Self;
    fn asin(self) -> Self;
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

    fn to_f64(self) -> f64 {
        self as f64
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

    fn mul_add(self, a: Self, b: Self) -> Self {
        cuda_specific!(core::intrinsics::fmaf32(self, a, b), self.mul_add(a, b));
    }

    fn sqrt(self) -> Self {
        cuda_specific!(core::intrinsics::sqrtf32(self), self.sqrt());
    }

    fn exp(self) -> Self {
        cuda_specific!(core::intrinsics::expf32(self), self.exp());
    }

    fn ln(self) -> Self {
        cuda_specific!(libm::logf(self), self.ln());
    }

    fn log2(self) -> Self {
        libm::log2f(self)
    }

    fn powf(self, n: Self) -> Self {
        cuda_specific!(libm::powf(self, n), self.powf(n));
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

    fn floor(self) -> Self {
        cuda_specific!(core::intrinsics::floorf32(self), self.floor());
    }
}

impl Float for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }

    fn to_f64(self) -> f64 {
        self
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

    fn mul_add(self, a: Self, b: Self) -> Self {
        cuda_specific!(core::intrinsics::fmaf64(self, a, b), self.mul_add(a, b));
    }

    fn sqrt(self) -> Self {
        cuda_specific!(core::intrinsics::sqrtf64(self), self.sqrt());
    }

    fn exp(self) -> Self {
        cuda_specific!(core::intrinsics::expf64(self), self.exp());
    }

    fn ln(self) -> Self {
        cuda_specific!(libm::log(self), self.ln());
    }

    fn log2(self) -> Self {
        libm::log2(self)
    }

    fn powf(self, n: Self) -> Self {
        cuda_specific!(libm::pow(self, n), self.powf(n));
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

    fn floor(self) -> Self {
        cuda_specific!(core::intrinsics::floorf64(self), self.floor());
    }
}

pub fn almost_eq(a: f64, b: f64, acc: f64) -> bool {
    // only true if a and b are infinite with same
    // sign
    if a.is_infinite() && b.is_infinite() {
        return a == b;
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

/// vector in R^2 represented as a 2-tuple
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy, Neg, Add, Sub, Mul, Div)]
#[cfg_attr(not(target_arch = "nvptx64"), derive(Serialize, Deserialize))]
pub struct Pair<F: Float> {
    pub x: F,
    pub y: F,
}

impl<F: Float> Pair<F> {
    /// dot product of two vectors, a • b
    pub fn dot(self, other: Self) -> F {
        self.x * other.x + self.y * other.y
    }

    /// square of the vector norm, ||v||^2
    pub fn norm_squared(self) -> F {
        self.dot(self)
    }

    /// vector norm, ||v||
    pub fn norm(self) -> F {
        self.norm_squared().sqrt()
    }

    /// is it a unit vector, ||v|| ≅? 1
    pub fn is_unit(self) -> bool {
        almost_eq(self.norm().to_f64(), 1_f64, 1e-3)
    }

    /// Fused multiply add of two vectors with a scalar, (self * a) + b
    pub fn mul_add(self, a: F, b: Self) -> Self {
        Pair {
            x: self.x.mul_add(a, b.x),
            y: self.y.mul_add(a, b.y),
        }
    }
}

/// rotate `vector` by `angle` CCW
#[inline(always)]
pub fn rotate<F: Float>(angle: F, vector: Pair<F>) -> Pair<F> {
    let (s, c) = angle.sincos();
    Pair {
        x: c * vector.x - s * vector.y,
        y: s * vector.x + c * vector.y,
    }
}

/// Matrix in R^(2x2) in row major order
#[derive(Debug, Clone, Copy)]
pub struct Mat2<F: Float>([F; 4]);

impl<F: Float> Mat2<F> {
    /// New Matrix from the two given columns
    pub fn new_from_cols(col1: Pair<F>, col2: Pair<F>) -> Self {
        Self([col1.x, col2.x, col1.y, col2.y])
    }

    /// Matrix inverse if it exists
    pub fn inverse(self) -> Option<Self> {
        let [a, b, c, d] = self.0;
        let det = a * d - b * c;
        if det == F::zero() {
            None
        } else {
            Some(Self([d / det, -b / det, -c / det, a / det]))
        }
    }
}

impl<F: Float> core::ops::Mul<Pair<F>> for Mat2<F> {
    type Output = Pair<F>;

    /// Matrix x Vector -> Vector multiplication
    fn mul(self, rhs: Pair<F>) -> Self::Output {
        let [a, b, c, d] = self.0;
        Pair {
            x: a * rhs.x + b * rhs.y,
            y: c * rhs.x + d * rhs.y,
        }
    }
}

/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
#[derive(Copy, Clone)]
pub struct Welford<F> {
    pub count: F,
    pub mean: F,
    pub m2: F,
}

impl<F: Float> Welford<F> {
    pub fn new() -> Self {
        Welford {
            count: F::zero(),
            mean: F::zero(),
            m2: F::zero(),
        }
    }
    pub fn next_sample(&mut self, x: F) {
        self.count += F::one();
        let delta = x - self.mean;
        self.mean += delta / self.count;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    pub fn sample_variance(&self) -> F {
        self.m2 / (self.count - F::one())
    }
    /// Standard Error of the Mean (SEM)
    pub fn sem(&self) -> F {
        (self.sample_variance() / self.count).sqrt()
    }
    /// Is the Standard Error of the Mean (SEM) less than the error threshold?
    /// Uses the square of the error for numerical stability (avoids sqrt)
    pub fn sem_le_error_threshold(&self, error_squared: F) -> bool {
        // SEM^2 = self.sample_variance() / self.count
        self.m2 < error_squared * (self.count * (self.count - F::one()))
    }
    pub fn combine(&mut self, other: Self) {
        let count = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean = (self.count * self.mean + other.count * other.mean) / count;
        self.m2 = self.m2 + other.m2 + delta.sqr() * self.count * other.count / count;
        self.count = count;
    }
}

impl<F: Float> Default for Welford<F> {
    fn default() -> Self {
        Self::new()
    }
}
