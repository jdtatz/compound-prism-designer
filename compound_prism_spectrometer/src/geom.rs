use crate::utils::*;
use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Neg, Sub};

pub trait Vector:
    'static
    + Copy
    + Clone
    + Debug
    + PartialEq
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<<Self as Vector>::Scalar, Output = Self>
    + Div<<Self as Vector>::Scalar, Output = Self>
{
    type Scalar: Copy + FloatExt;

    fn x(self) -> Self::Scalar;

    fn y(self) -> Self::Scalar;

    fn z(self) -> Self::Scalar;

    fn zero() -> Self;

    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self;

    /// unit vector at angle `theta` relative to the x axis in the xy plane.
    fn angled_xy(theta: Self::Scalar) -> Self;

    fn rotate_xy(self, theta: Self::Scalar) -> Self;

    fn rot_90_xy(self) -> Self;

    fn rot_180_xy(self) -> Self;

    fn cos_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar;

    fn sec_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar;

    fn sin_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar;

    fn csc_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar;

    fn tan_xy(self) -> Self::Scalar;

    fn cot_xy(self) -> Self::Scalar;

    /// dot product of two vectors, a • b
    fn dot(self, other: Self) -> Self::Scalar;

    /// square of the vector norm, ||v||^2
    fn norm_squared(self) -> Self::Scalar {
        self.dot(self)
    }

    /// vector norm, ||v||
    fn norm(self) -> Self::Scalar {
        self.norm_squared().sqrt()
    }

    /// is it a unit vector, ||v|| ≅? 1
    fn check_unit(self) -> bool {
        float_eq::float_eq!(
            self.norm(),
            Self::Scalar::one(),
            rmax <= Self::Scalar::from_u32_ratio(1, 1000),
        )
    }

    /// Fused multiply add of two vectors with a scalar, (self * a) + b
    fn mul_add(self, a: Self::Scalar, b: Self) -> Self;
}

/// vector in R^2 represented as a 2-tuple
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy, Neg, Add, Sub, Mul, Div, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct Pair<T> {
    pub x: T,
    pub y: T,
}

impl<F: FloatExt> Vector for Pair<F> {
    type Scalar = F;

    fn x(self) -> Self::Scalar {
        self.x
    }

    fn y(self) -> Self::Scalar {
        self.y
    }

    fn z(self) -> Self::Scalar {
        Self::Scalar::zero()
    }

    fn zero() -> Self {
        Self {
            x: Self::Scalar::zero(),
            y: Self::Scalar::zero(),
        }
    }

    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self {
        Self { x, y }
    }

    /// unit vector at angle `theta` relative to the x axis.
    fn angled_xy(theta: F) -> Self {
        let (sin, cos) = theta.sincos();
        Self { x: cos, y: sin }
    }

    fn rotate_xy(self, theta: Self::Scalar) -> Self {
        let (s, c) = theta.sincos();
        Self {
            x: c * self.x - s * self.y,
            y: s * self.x + c * self.y,
        }
    }

    fn rot_90_xy(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }

    fn rot_180_xy(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }

    fn cos_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar {
        self.x / hypotenuse
    }

    fn sec_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar {
        hypotenuse / self.x
    }

    fn sin_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar {
        self.y / hypotenuse
    }

    fn csc_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar {
        hypotenuse / self.y
    }

    fn tan_xy(self) -> Self::Scalar {
        self.y / self.x
    }

    fn cot_xy(self) -> Self::Scalar {
        self.x / self.y
    }

    /// dot product of two vectors, a • b
    fn dot(self, other: Self) -> F {
        self.x * other.x + self.y * other.y
    }

    /// Fused multiply add of two vectors with a scalar, (self * a) + b
    fn mul_add(self, a: F, b: Self) -> Self {
        Self {
            x: self.x.mul_add(a, b.x),
            y: self.y.mul_add(a, b.y),
        }
    }
}

impl<F: FloatExt> Pair<F> {
    pub fn from_vector<V: Vector<Scalar = F>>(vector: V) -> Self {
        Pair {
            x: vector.x(),
            y: vector.y(),
        }
    }
}

#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy, Neg, Add, Sub, Mul, Div, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct Triplet<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<F: FloatExt> Vector for Triplet<F> {
    type Scalar = F;

    fn x(self) -> Self::Scalar {
        self.x
    }

    fn y(self) -> Self::Scalar {
        self.y
    }

    fn z(self) -> Self::Scalar {
        self.z
    }

    fn zero() -> Self {
        Self {
            x: Self::Scalar::zero(),
            y: Self::Scalar::zero(),
            z: Self::Scalar::zero(),
        }
    }

    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self {
        Self {
            x,
            y,
            z: Self::Scalar::zero(),
        }
    }

    /// unit vector at angle `theta` relative to the x axis.
    fn angled_xy(theta: F) -> Self {
        let (sin, cos) = theta.sincos();
        Self {
            x: cos,
            y: sin,
            z: Self::Scalar::zero(),
        }
    }

    fn rotate_xy(self, theta: Self::Scalar) -> Self {
        let (s, c) = theta.sincos();
        Self {
            x: c * self.x - s * self.y,
            y: s * self.x + c * self.y,
            z: self.z,
        }
    }

    fn rot_90_xy(self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
            z: self.z,
        }
    }

    fn rot_180_xy(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: self.z,
        }
    }

    fn cos_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar {
        self.x / hypotenuse
    }

    fn sec_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar {
        hypotenuse / self.x
    }

    fn sin_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar {
        self.y / hypotenuse
    }

    fn csc_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar {
        hypotenuse / self.y
    }

    fn tan_xy(self) -> Self::Scalar {
        self.y / self.x
    }

    fn cot_xy(self) -> Self::Scalar {
        self.x / self.y
    }

    /// dot product of two vectors, a • b
    fn dot(self, other: Self) -> F {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Fused multiply add of two vectors with a scalar, (self * a) + b
    fn mul_add(self, a: F, b: Self) -> Self {
        Self {
            x: self.x.mul_add(a, b.x),
            y: self.y.mul_add(a, b.y),
            z: self.z.mul_add(a, b.z),
        }
    }
}

/// Matrix in R^(2x2) in row major order
#[derive(Debug, Clone, Copy)]
pub struct Mat2<F: FloatExt>([F; 4]);

impl<F: FloatExt> Mat2<F> {
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

impl<F: FloatExt> core::ops::Mul<Pair<F>> for Mat2<F> {
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

pub trait Surface<Point: Vector, UnitVector: Vector = Point>: 'static + Copy {
    fn intersection(&self, ray: (Point, UnitVector)) -> Option<(Point, UnitVector)>;
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct Plane<V: Vector> {
    pub(crate) normal: V,
    pub(crate) midpt: V,
    height: V::Scalar,
}

#[cfg(not(target_arch = "nvptx64"))]
impl<V: Vector> Plane<V> {
    pub(crate) fn end_points(&self, height: V::Scalar) -> (V, V) {
        let dx = self.normal.tan_xy() * height * V::Scalar::from_u32_ratio(1, 2);
        let ux = self.midpt.x() - dx;
        let lx = self.midpt.x() + dx;
        (V::from_xy(ux, height), V::from_xy(lx, V::Scalar::zero()))
    }
}

impl<V: Vector> Surface<V> for Plane<V> {
    fn intersection(&self, ray: (V, V)) -> Option<(V, V)> {
        let (origin, direction) = ray;
        debug_assert!(direction.check_unit());
        let ci = -direction.dot(self.normal);
        if ci <= V::Scalar::zero() {
            return None;
        }
        let d = (origin - self.midpt).dot(self.normal) / ci;
        let p = direction.mul_add(d, origin);
        if p.y() <= V::Scalar::zero() || self.height <= p.y() {
            None
        } else {
            Some((p, self.normal))
        }
    }
}

pub(crate) fn create_joined_trapezoids<V: Vector, const N: usize>(
    height: V::Scalar,
    first_angle: V::Scalar,
    angles: [V::Scalar; N],
    last_angle: V::Scalar,
    first_sep_length: V::Scalar,
    sep_lengths: [V::Scalar; N],
) -> (Plane<V>, [Plane<V>; N], Plane<V>) {
    let (sep_lengths, last_sep_length) = array_prepend(first_sep_length, sep_lengths);
    let h2 = height * V::Scalar::from_u32_ratio(1, 2);
    let normal = V::angled_xy(first_angle).rot_180_xy();
    debug_assert_eq!(
        normal,
        V::from_xy(-V::Scalar::one(), V::Scalar::zero()).rotate_xy(first_angle)
    );
    #[cfg(all(test, debug_assertions))]
    float_eq::assert_float_eq!(
        normal.tan_xy().abs(),
        first_angle.tan().abs(),
        rmax <= LossyFrom::lossy_from(1e-5f64)
    );
    let first = Plane {
        height,
        normal,
        midpt: V::from_xy(normal.tan_xy().abs() * h2, h2),
    };
    let mut prev_sign = normal.y().is_sign_positive();
    let mut d1 = first.midpt.x();
    let mut mx = first.midpt.x();
    let mut next_plane = |(angle, sep_len)| {
        let normal = V::angled_xy(angle).rot_180_xy();
        let sign = normal.y().is_sign_positive();
        let d2 = normal.tan_xy().abs() * h2;
        let sep_dist = sep_len
            + if prev_sign != sign {
                d1 + d2
            } else {
                (d1 - d2).abs()
            };
        prev_sign = sign;
        d1 = d2;
        mx += sep_dist;
        Plane {
            height,
            normal,
            midpt: V::from_xy(mx, h2),
        }
    };
    let inter: [Plane<V>; N] = angles.zip(sep_lengths).map(&mut next_plane);
    let last = (&mut next_plane)((last_angle, last_sep_length));
    (first, inter, last)
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct CurvedPlane<V: Vector> {
    /// The midpt of the Curved Surface / circular segment
    pub(crate) midpt: V,
    /// The center of the circle
    pub(crate) center: V,
    /// The radius of the circle
    pub(crate) radius: V::Scalar,
    /// max_dist_sq = sagitta ^ 2 + (chord_length / 2) ^ 2
    max_dist_sq: V::Scalar,
    #[wrapped_from(skip)]
    direction: bool,
}

impl<V: Vector> CurvedPlane<V> {
    pub(crate) fn new(signed_curvature: V::Scalar, height: V::Scalar, chord: Plane<V>) -> Self {
        debug_assert!(
            V::Scalar::one() >= signed_curvature.abs()
                && signed_curvature.abs() > V::Scalar::zero()
        );
        debug_assert!(height > V::Scalar::zero());
        let chord_length = chord.normal.sec_xy(height).abs();
        let radius = chord_length * V::Scalar::from_u32_ratio(1, 2) / signed_curvature.abs();
        let apothem = (radius * radius
            - chord_length * chord_length * V::Scalar::from_u32_ratio(1, 4))
        .sqrt();
        let sagitta = radius - apothem;
        let (center, midpt) = if signed_curvature.is_sign_positive() {
            (
                chord.midpt + chord.normal * apothem,
                chord.midpt - chord.normal * sagitta,
            )
        } else {
            (
                chord.midpt - chord.normal * apothem,
                chord.midpt + chord.normal * sagitta,
            )
        };
        Self {
            midpt,
            center,
            radius,
            max_dist_sq: sagitta * sagitta
                + chord_length * chord_length * V::Scalar::from_u32_ratio(1, 4),
            direction: signed_curvature.is_sign_positive(),
        }
    }

    fn is_along_arc(&self, pt: V) -> bool {
        #[cfg(all(test, debug_assertions))]
        float_eq::assert_float_eq!(
            (pt - self.center).norm(),
            self.radius,
            rmax <= LossyFrom::lossy_from(1e-5f64)
        );
        (pt - self.midpt).norm_squared() <= self.max_dist_sq
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub(crate) fn end_points(&self, height: V::Scalar) -> (V, V) {
        let theta_2 = V::Scalar::lossy_from(2)
            * (self.max_dist_sq.sqrt() / (V::Scalar::lossy_from(2) * self.radius)).asin();
        let r = self.midpt - self.center;
        let u = self.center + r.rotate_xy(theta_2);
        let l = self.center + r.rotate_xy(-theta_2);
        debug_assert!(
            (u.y() - height).abs() < V::Scalar::from_u32_ratio(1, 10000),
            "{:?} {}",
            u,
            height
        );
        debug_assert!(l.y().abs() < V::Scalar::from_u32_ratio(1, 10000), "{:?}", l);
        debug_assert!(
            ((u - r).norm_squared() - self.max_dist_sq).abs() / self.max_dist_sq
                < V::Scalar::from_u32_ratio(1, 10000)
        );
        debug_assert!(
            ((l - r).norm_squared() - self.max_dist_sq).abs() / self.max_dist_sq
                < V::Scalar::from_u32_ratio(1, 10000)
        );
        (
            V::from_xy(u.x(), height),
            V::from_xy(l.x(), V::Scalar::zero()),
        )
    }
}

impl<V: Vector> Surface<V> for CurvedPlane<V> {
    fn intersection(&self, ray: (V, V)) -> Option<(V, V)> {
        let (origin, direction) = ray;
        debug_assert!(direction.check_unit());
        let delta = origin - self.center;
        let ud = direction.dot(delta);
        let discriminant = ud * ud - delta.norm_squared() + self.radius * self.radius;
        if discriminant <= V::Scalar::zero() {
            return None;
        }
        let d = if self.direction {
            -ud + discriminant.sqrt()
        } else {
            -ud - discriminant.sqrt()
        };
        if d <= V::Scalar::zero() {
            return None;
        }
        let p = direction.mul_add(d, origin);
        if self.is_along_arc(p) {
            let snorm = (self.center - p) / self.radius;
            debug_assert!(snorm.check_unit());
            Some((p, snorm))
        } else {
            None
        }
    }
}
