use libm::sincos;
#[cfg(not(target_arch ="nvptx64"))]
use serde::{Deserialize, Serialize};

#[cfg(target_arch ="nvptx64")]
pub trait F64Ext {
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, e: f64) -> Self;
    fn fract(self) -> Self;
    fn abs(self) -> Self;
    fn asin(self) -> Self;
}

#[cfg(target_arch ="nvptx64")]
impl F64Ext for f64 {
    fn mul_add(self, a: Self, b: Self) -> Self {
        unsafe { core::intrinsics::fmaf64(self, a, b) }
    }

    fn sqrt(self) -> Self {
        unsafe { core::intrinsics::sqrtf64(self) }
    }

    fn exp(self) -> Self {
        unsafe { core::intrinsics::expf64(self) }
    }

    fn ln(self) -> Self {
        libm::log(self)
    }

    fn powf(self, e: f64) -> Self {
        libm::pow(self, e)
    }

    fn fract(self) -> Self {
        self - unsafe { core::intrinsics::truncf64(self) }
    }

    fn abs(self) -> Self {
        unsafe { core::intrinsics::fabsf64(self) }
    }

    fn asin(self) -> Self {
        libm::asin(self)
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
#[derive(Debug, PartialEq, Clone, Copy, From, Into, Neg, Add, Sub, Mul, Div)]
#[cfg_attr(not(target_arch="nvptx64"), derive(Serialize, Deserialize))]
pub struct Pair {
    pub x: f64,
    pub y: f64,
}

impl Pair {
    /// dot product of two vectors, a • b
    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y
    }

    /// square of the vector norm, ||v||^2
    pub fn norm_squared(self) -> f64 {
        self.dot(self)
    }

    /// vector norm, ||v||
    pub fn norm(self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// is it a unit vector, ||v|| ≅? 1
    pub fn is_unit(self) -> bool {
        (self.norm() - 1_f64).abs() < 1e-3
    }

    /// Fused multiply add of two vectors with a scalar, (self * a) + b
    pub fn mul_add(self, a: f64, b: Pair) -> Pair {
        Pair {
            x: self.x.mul_add(a, b.x),
            y: self.y.mul_add(a, b.y),
        }
    }
}

/// rotate `vector` by `angle` CCW
#[inline(always)]
pub fn rotate(angle: f64, vector: Pair) -> Pair {
    let (s, c) = sincos(angle);
    Pair {
        x: c * vector.x - s * vector.y,
        y: s * vector.x + c * vector.y,
    }
}

/// Matrix in R^(2x2) in row major order
#[derive(Debug, Clone, Copy)]
pub struct Mat2([f64; 4]);

impl Mat2 {
    /// New Matrix from the two given columns
    pub fn new_from_cols(col1: Pair, col2: Pair) -> Self {
        Self([col1.x, col2.x, col1.y, col2.y])
    }

    /// Matrix inverse if it exists
    pub fn inverse(self) -> Option<Self> {
        let [a, b, c, d] = self.0;
        let det = a * d - b * c;
        if det == 0. {
            None
        } else {
            Some(Self([d / det, -b / det, -c / det, a / det]))
        }
    }
}

impl core::ops::Mul<Pair> for Mat2 {
    type Output = Pair;

    /// Matrix x Vector -> Vector multiplication
    fn mul(self, rhs: Pair) -> Self::Output {
        let [a, b, c, d] = self.0;
        Pair {
            x: a * rhs.x + b * rhs.y,
            y: c * rhs.x + d * rhs.y,
        }
    }
}

/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
#[derive(Clone)]
pub struct Welford {
    pub count: f64,
    pub mean: f64,
    m2: f64,
}

impl Welford {
    pub fn new() -> Self {
        Welford {
            count: 0.,
            mean: 0.,
            m2: 0.,
        }
    }
    pub fn next_sample(&mut self, x: f64) {
        self.count += 1.;
        let delta = x - self.mean;
        self.mean += delta / self.count;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    #[allow(dead_code)]
    pub fn variance(&self) -> f64 {
        self.m2 / self.count
    }
    #[allow(dead_code)]
    pub fn sample_variance(&self) -> f64 {
        self.m2 / (self.count - 1.)
    }
    pub fn sem(&self) -> f64 {
        (self.sample_variance() / self.count).sqrt()
    }
    /// Is the Standard Error of the Mean (SEM) less than the error threshold?
    /// Uses the square of the error for numerical stability (avoids sqrt)
    pub fn sem_le_error_threshold(&self, error_squared: f64) -> bool {
        // SEM^2 = self.sample_variance() / self.count
        self.m2 < error_squared * (self.count * (self.count - 1.))
    }
}
