use super::utils::*;
use core::ops::*;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom, Deref, DerefMut)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct Vector<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> Vector<T, N> {
    pub const fn new(vector: [T; N]) -> Self {
        Self(vector)
    }

    pub fn map<R, F: FnMut(T) -> R>(self, f: F) -> Vector<R, N> {
        let Vector(array) = self;
        Vector(array.map(f))
    }

    pub fn zip<U>(self, other: Vector<U, N>) -> Vector<(T, U), N> {
        let Vector(lhs) = self;
        let Vector(rhs) = other;
        Vector(lhs.zip(rhs))
    }
}

impl<L, R, const N: usize> Vector<(L, R), N> {
    pub fn map2<T, F: FnMut(L, R) -> T>(self, mut f: F) -> Vector<T, N> {
        let Vector(array) = self;
        Vector(array.map(move |(l, r)| f(l, r)))
    }
}

impl<T: ConstZero, const N: usize> ConstZero for Vector<T, N> {
    const ZERO: Self = Vector([T::ZERO; N]);
}

impl<T: Mul<Output = T>, const N: usize> Vector<T, N> {
    pub fn hadamard_product(self, rhs: Self) -> Self {
        // Self(self.0.zip(rhs.0).map(|(l, r)| l * r))
        self.zip(rhs).map2(Mul::mul)
    }
}

impl<T: Copy + ConstZero + Add<Output = T> + Mul<Output = T>, const N: usize> Vector<T, N> {
    pub fn dot(self, rhs: Self) -> T {
        // core::array::IntoIter::new(self.hadamard_product(rhs).0).fold(T::ZERO, Add::add)
        self.hadamard_product(rhs)
            .0
            .iter()
            .copied()
            .fold(T::ZERO, Add::add)
    }
}

impl<T: Copy + ConstZero + Add<Output = T> + Mul<Output = T>, const N: usize> Vector<T, N> {
    pub fn norm_squared(self) -> T {
        self.dot(self)
    }
}

impl<T: Copy + ConstZero + ConstOne + Float, const N: usize> Vector<T, N> {
    pub fn norm(self) -> T {
        self.norm_squared().sqrt()
    }

    pub fn is_unit(self) -> bool
    where
        T: Float + ApproxEq + LossyFrom<f64>,
    {
        float_eq::float_eq!(
            self.norm(),
            T::ONE,
            rmax <= T::lossy_from(1e-3f64),
            // ulps <= 10u8,
        )
    }

    pub fn normalize(self) -> UnitVector<T, N> {
        UnitVector(self / self.norm())
    }
}

impl<T: Copy + Float, const N: usize> Vector<T, N> {
    pub fn mul_add(self, b: T, c: Self) -> Self {
        // Self(self.0.zip(c.0).map(|(a, c)| a.mul_add(b, c)))
        self.zip(c).map(|(a, c)| a.mul_add(b, c))
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Deref, DerefMut, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct UnitVector<T, const N: usize>(pub Vector<T, N>);

impl<T: Copy + FloatExt, const N: usize> UnitVector<T, N> {
    pub fn new(vector: Vector<T, N>) -> Self {
        debug_assert!(vector.is_unit());
        Self(vector)
    }

    pub fn try_new(vector: Vector<T, N>) -> Option<Self> {
        if vector.is_unit() {
            Some(Self(vector))
        } else {
            None
        }
    }
}

impl<T: Neg<Output = T>, const N: usize> Neg for Vector<T, N> {
    type Output = Vector<T, N>;

    fn neg(self) -> Self::Output {
        // Self(self.0.map(Neg::neg))
        self.map(Neg::neg)
    }
}

impl<T: Neg<Output = T>, const N: usize> Neg for UnitVector<T, N> {
    type Output = UnitVector<T, N>;

    fn neg(self) -> Self::Output {
        // Self(self.0.map(Neg::neg))
        Self(self.0.map(Neg::neg))
    }
}

impl<T: Add<Output = T>, const N: usize> Add<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn add(self, rhs: Vector<T, N>) -> Self::Output {
        // Self(self.0.zip(rhs.0).map(|(l, r)| l + r))
        self.zip(rhs).map2(Add::add)
    }
}

impl<T: Sub<Output = T>, const N: usize> Sub<Vector<T, N>> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn sub(self, rhs: Vector<T, N>) -> Self::Output {
        // Self(self.0.zip(rhs.0).map(|(l, r)| l - r))
        self.zip(rhs).map2(Sub::sub)
    }
}

impl<T: Copy + Mul<Output = T>, const N: usize> Mul<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        // Self(self.0.map(|lhs| lhs * rhs))
        self.map(|lhs| lhs * rhs)
    }
}

impl<T: Copy + Div<Output = T>, const N: usize> Div<T> for Vector<T, N> {
    type Output = Vector<T, N>;

    fn div(self, rhs: T) -> Self::Output {
        // Self(self.0.map(|lhs| lhs / rhs))
        self.map(|lhs| lhs / rhs)
    }
}

// pub fn proj<T: Copy + Ring, const N: usize>(
//     a: Vector<T, N>,
//     UnitVector(b): UnitVector<T, N>,
// ) -> Vector<T, N> {
//     b * (a.dot(b))
// }

pub fn oproj<T: Copy + Ring, const N: usize>(
    a: Vector<T, N>,
    UnitVector(b): UnitVector<T, N>,
) -> Vector<T, N> {
    a - b * (a.dot(b))
}

/// $\| a x b \|^2$
pub fn cross_prod_magnitude_sq<T: Copy + Ring, const N: usize>(
    a: Vector<T, N>,
    UnitVector(b): UnitVector<T, N>,
) -> T {
    a.norm_squared() - T::sqr(a.dot(b))
}

// Temp for back compat
impl<T: FloatExt, const N: usize> Vector<T, N> {
    pub(crate) fn x(self) -> T {
        self.0[0]
    }

    pub(crate) fn y(self) -> T {
        self.0[1]
    }

    // fn z(self) -> T {
    //     self.0[2]
    // }

    pub fn from_xy(x: T, y: T) -> Self {
        let mut v = Self::ZERO;
        v[0] = x;
        v[1] = y;
        v
    }

    /// unit vector at angle `theta` relative to the x axis in the xy plane.
    pub fn angled_xy(theta: T) -> Self {
        let (sin, cos) = theta.sin_cos();
        Self::from_xy(cos, sin)
    }

    pub(crate) fn rotate_xy(mut self, theta: T) -> Self {
        let (s, c) = theta.sin_cos();
        let x = self[0];
        let y = self[1];
        self[0] = c * x - s * y;
        self[1] = s * x + c * y;
        self
    }

    pub(crate) fn rot_90_xy(mut self) -> Self {
        let x = self[0];
        let y = self[1];
        self[0] = -y;
        self[1] = x;
        self
    }

    pub(crate) fn rot_180_xy(mut self) -> Self {
        let x = self[0];
        let y = self[1];
        self[0] = -x;
        self[1] = -y;
        self
    }

    pub(crate) fn rot_90_ccw_xy(mut self) -> Self {
        let x = self[0];
        let y = self[1];
        self[0] = y;
        self[1] = -x;
        self
    }

    // pub(crate) fn cos_xy(self, hypotenuse: T) -> T {
    //     self.x() / hypotenuse
    // }

    pub(crate) fn sec_xy(self, hypotenuse: T) -> T {
        hypotenuse / self.x()
    }

    pub(crate) fn sin_xy(self, hypotenuse: T) -> T {
        self.y() / hypotenuse
    }

    // pub(crate) fn csc_xy(self, hypotenuse: T) -> T {
    //     hypotenuse / self.y()
    // }

    pub(crate) fn tan_xy(self) -> T {
        self.y() / self.x()
    }

    // pub(crate) fn cot_xy(self) -> T {
    //     self.x() / self.y()
    // }
}
