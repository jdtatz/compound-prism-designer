use super::utils::*;
use core::ops::*;
use core::simd::*;

impl<T: ConstZero + SimdElement, const N: usize> ConstZero for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const ZERO: Self = Simd::from_array([T::ZERO; N]);
}

impl<T: ConstOne + SimdElement, const N: usize> ConstOne for Simd<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const ONE: Self = Simd::from_array([T::ONE; N]);
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(wrapped = "crate::LossyFrom::lossy_from")]
pub struct SimdVector<T: SimdElement, const N: usize>(Simd<T, N>)
where
    LaneCount<N>: SupportedLaneCount;

impl<T: ConstZero + SimdElement, const N: usize> ConstZero for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    const ZERO: Self = SimdVector(Simd::from_array([T::ZERO; N]));
}

impl<T: SimdElement, const N: usize> Neg for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: SimdFloat<Scalar = T> + Neg<Output = Simd<T, N>>,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl<T: SimdElement, const N: usize> Add for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: SimdFloat<Scalar = T> + Add<Output = Simd<T, N>>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl<T: SimdElement, const N: usize> Sub for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: SimdFloat<Scalar = T> + Sub<Output = Simd<T, N>>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl<T: SimdElement, const N: usize> Mul<T> for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: SimdFloat<Scalar = T> + Mul<Output = Simd<T, N>>,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self(self.0 * Simd::splat(rhs))
    }
}

impl<T: SimdElement, const N: usize> Div<T> for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
    Simd<T, N>: SimdFloat<Scalar = T> + Div<Output = Simd<T, N>>,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self(self.0 / Simd::splat(rhs))
    }
}

#[cfg(target_arch = "nvptx64")]
mod fast_nvptx {
    use super::*;
    use nvptx_sys::{FastFloat, FastNum};

    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub struct FastSimdVector<T: FastNum + SimdElement, const N: usize>(Simd<T, N>)
    where
        LaneCount<N>: SupportedLaneCount;

    impl<T: ConstZero + FastNum + SimdElement, const N: usize> ConstZero for FastSimdVector<T, N>
    where
        LaneCount<N>: SupportedLaneCount,
    {
        const ZERO: Self = FastSimdVector(Simd::from_array([T::ZERO; N]));
    }

    impl<T: FastNum + SimdElement, const N: usize> Neg for FastSimdVector<T, N>
    where
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: SimdFloat<Scalar = T> + Neg<Output = Simd<T, N>>,
    {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl<T: FastNum + SimdElement, const N: usize> Add for FastSimdVector<T, N>
    where
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: SimdFloat<Scalar = T> + Add<Output = Simd<T, N>>,
    {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            Self(self.0 + rhs.0)
        }
    }

    impl<T: FastNum + SimdElement, const N: usize> Sub for FastSimdVector<T, N>
    where
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: SimdFloat<Scalar = T> + Sub<Output = Simd<T, N>>,
    {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            Self(self.0 - rhs.0)
        }
    }

    impl<T: FastNum + SimdElement, const N: usize> Mul<FastFloat<T>> for FastSimdVector<T, N>
    where
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: SimdFloat<Scalar = T> + Mul<Output = Simd<T, N>>,
    {
        type Output = Self;

        fn mul(self, rhs: FastFloat<T>) -> Self::Output {
            Self(self.0 * Simd::splat(rhs.0))
        }
    }

    impl<T: FastNum + SimdElement, const N: usize> Div<FastFloat<T>> for FastSimdVector<T, N>
    where
        LaneCount<N>: SupportedLaneCount,
        Simd<T, N>: SimdFloat<Scalar = T> + Div<Output = Simd<T, N>>,
    {
        type Output = Self;

        fn div(self, rhs: FastFloat<T>) -> Self::Output {
            Self(self.0 / Simd::splat(rhs.0))
        }
    }

    extern "platform-intrinsic" {
        // fma
        fn simd_fma<T>(x: T, y: T, z: T) -> T;
    }

    impl<T: 'static + FastNum + SimdElement + Float + ConstZero + core::fmt::Debug> Vector<2>
        for FastSimdVector<T, 2>
    where
        Simd<T, 2>: SimdFloat<Scalar = T> + Ring + Div<Output = Simd<T, 2>>,
    {
        type Scalar = FastFloat<T>;

        fn new(vector: [Self::Scalar; 2]) -> Self {
            Self(Simd::from_array(vector.map(|v| v.0)))
        }

        fn to_array(self) -> [Self::Scalar; 2] {
            Simd::to_array(self.0).map(FastFloat)
        }

        fn hadamard_product(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }

        fn reduce_sum(self) -> Self::Scalar {
            FastFloat(<Simd<T, 2> as SimdFloat>::reduce_sum(self.0))
        }

        fn mul_add(self, b: Self::Scalar, c: Self) -> Self
        where
            Self::Scalar: Float,
        {
            // FastSimdVector(self.0.mul_add(Simd::splat(b.0), c.0))
            FastSimdVector(unsafe { simd_fma(self.0, Simd::splat(b.0), c.0) })
        }

        fn x(self) -> Self::Scalar {
            self.to_array()[0]
        }

        fn y(self) -> Self::Scalar {
            self.to_array()[1]
        }

        fn update_xy(self, x: Self::Scalar, y: Self::Scalar) -> Self {
            Self::new([x, y])
        }
    }

    impl<T: 'static + FastNum + SimdElement + Float + ConstZero + core::fmt::Debug> Vector<3>
        for FastSimdVector<T, 4>
    where
        Simd<T, 4>: SimdFloat<Scalar = T> + Ring + Div<Output = Simd<T, 4>>,
    {
        type Scalar = FastFloat<T>;

        fn new([x, y, z]: [Self::Scalar; 3]) -> Self {
            Self(Simd::from_array([x.0, y.0, z.0, T::ZERO]))
        }

        fn to_array(self) -> [Self::Scalar; 3] {
            let [x, y, z, _] = Simd::to_array(self.0);
            [FastFloat(x), FastFloat(y), FastFloat(z)]
        }

        fn hadamard_product(self, rhs: Self) -> Self {
            Self(self.0 * rhs.0)
        }

        fn reduce_sum(self) -> Self::Scalar {
            FastFloat(<Simd<T, 4> as SimdFloat>::reduce_sum(self.0))
        }

        fn mul_add(self, b: Self::Scalar, c: Self) -> Self
        where
            Self::Scalar: Float,
        {
            // FastSimdVector(self.0.mul_add(Simd::splat(b.0), c.0))
            FastSimdVector(unsafe { simd_fma(self.0, Simd::splat(b.0), c.0) })
        }

        fn x(self) -> Self::Scalar {
            self.to_array()[0]
        }

        fn y(self) -> Self::Scalar {
            self.to_array()[1]
        }

        fn update_xy(self, x: Self::Scalar, y: Self::Scalar) -> Self {
            let [_, _, z, _] = Simd::to_array(self.0);
            Self::new([x, y, FastFloat(z)])
        }
    }
}
#[cfg(target_arch = "nvptx64")]
pub use fast_nvptx::*;

pub trait Vector<const DIM: usize>:
    'static
    + Sized
    + Copy
    + core::fmt::Debug
    + core::cmp::PartialEq
    + ConstZero
    + Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self::Scalar, Output = Self>
    + Div<Self::Scalar, Output = Self>
{
    type Scalar;

    fn new(vector: [Self::Scalar; DIM]) -> Self;
    fn to_array(self) -> [Self::Scalar; DIM];
    fn hadamard_product(self, rhs: Self) -> Self;
    fn reduce_sum(self) -> Self::Scalar;
    fn dot(self, rhs: Self) -> Self::Scalar {
        self.hadamard_product(rhs).reduce_sum()
    }
    fn norm_squared(self) -> Self::Scalar {
        self.dot(self)
    }
    fn norm(self) -> Self::Scalar
    where
        Self::Scalar: Float,
    {
        self.norm_squared().sqrt()
    }

    fn is_unit(self) -> bool
    where
        Self::Scalar: Float + ConstOne + ApproxEq + LossyFrom<f64>,
    {
        float_eq::float_eq!(
            self.norm(),
            Self::Scalar::ONE,
            rmax <= Self::Scalar::lossy_from(1e-3f64),
            // ulps <= 10u8,
        )
    }

    fn normalize(self) -> UnitVector<Self>
    where
        Self::Scalar: Float,
    {
        UnitVector(self / self.norm())
    }

    fn mul_add(self, b: Self::Scalar, c: Self) -> Self
    where
        Self::Scalar: Float,
    {
        self * b + c
    }

    fn x(self) -> Self::Scalar;
    fn y(self) -> Self::Scalar;
    fn from_xy(x: Self::Scalar, y: Self::Scalar) -> Self {
        Self::ZERO.update_xy(x, y)
    }
    fn update_xy(self, x: Self::Scalar, y: Self::Scalar) -> Self;

    /// unit vector at angle `theta` relative to the x axis in the xy plane.
    fn angled_xy(theta: Self::Scalar) -> Self
    where
        Self::Scalar: Float,
    {
        let (sin, cos) = theta.sin_cos();
        Self::from_xy(cos, sin)
    }

    fn rotate_xy(self, theta: Self::Scalar) -> Self
    where
        Self::Scalar: Float,
    {
        let (s, c) = theta.sin_cos();
        let x = self.x();
        let y = self.y();
        self.update_xy(c * x - s * y, s * x + c * y)
    }

    fn rot_90_xy(self) -> Self
    where
        Self::Scalar: Neg<Output = Self::Scalar>,
    {
        let x = self.x();
        let y = self.y();
        self.update_xy(-y, x)
    }

    fn rot_180_xy(self) -> Self
    where
        Self::Scalar: Neg<Output = Self::Scalar>,
    {
        let x = self.x();
        let y = self.y();
        self.update_xy(-x, -y)
    }

    fn rot_90_ccw_xy(self) -> Self
    where
        Self::Scalar: Neg<Output = Self::Scalar>,
    {
        let x = self.x();
        let y = self.y();
        self.update_xy(y, -x)
    }

    // fn cos_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar where Self::Scalar: Div<Output=Self::Scalar> {
    //     self.x() / hypotenuse
    // }

    fn sec_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar
    where
        Self::Scalar: Div<Output = Self::Scalar>,
    {
        hypotenuse / self.x()
    }

    fn sin_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar
    where
        Self::Scalar: Div<Output = Self::Scalar>,
    {
        self.y() / hypotenuse
    }

    // fn csc_xy(self, hypotenuse: Self::Scalar) -> Self::Scalar where Self::Scalar: Div<Output=Self::Scalar> {
    //     hypotenuse / self.y()
    // }

    fn tan_xy(self) -> Self::Scalar
    where
        Self::Scalar: Div<Output = Self::Scalar>,
    {
        self.y() / self.x()
    }

    // fn cot_xy(self) -> Self::Scalar where Self::Scalar: Div<Output=Self::Scalar> {
    //     self.x() / self.y()
    // }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Deref, DerefMut, WrappedFrom)]
#[wrapped_from(wrapped = "crate::LossyFrom::lossy_from")]
pub struct UnitVector<V>(pub V);

impl<V> UnitVector<V> {
    pub fn new<const DIM: usize>(vector: V) -> Self
    where
        V: Vector<DIM>,
        <V as Vector<DIM>>::Scalar: Float + ConstOne + ApproxEq + LossyFrom<f64>,
    {
        debug_assert!(vector.is_unit());
        Self(vector)
    }

    pub fn try_new<const DIM: usize>(vector: V) -> Option<Self>
    where
        V: Vector<DIM>,
        <V as Vector<DIM>>::Scalar: Float + ConstOne + ApproxEq + LossyFrom<f64>,
    {
        if vector.is_unit() {
            Some(Self(vector))
        } else {
            None
        }
    }

    pub fn unit_x<const DIM: usize>() -> Self
    where
        V: Vector<DIM>,
        <V as Vector<DIM>>::Scalar: ConstZero + ConstOne,
    {
        Self(V::from_xy(V::Scalar::ONE, V::Scalar::ZERO))
    }

    pub fn unit_y<const DIM: usize>() -> Self
    where
        V: Vector<DIM>,
        <V as Vector<DIM>>::Scalar: ConstZero + ConstOne,
    {
        Self(V::from_xy(V::Scalar::ZERO, V::Scalar::ONE))
    }
}

impl<V: Neg<Output = V>> Neg for UnitVector<V> {
    type Output = UnitVector<V>;

    fn neg(self) -> Self::Output {
        Self(self.0.neg())
    }
}

#[cfg(feature = "std")]
use std::simd::StdFloat;
#[cfg(not(feature = "std"))]
pub trait StdFloat {}
#[cfg(not(feature = "std"))]
impl<T> StdFloat for T {}

impl<T: 'static + SimdElement + Float + ConstZero + core::fmt::Debug> Vector<2> for SimdVector<T, 2>
where
    Simd<T, 2>: SimdFloat<Scalar = T> + StdFloat + Ring + Div<Output = Simd<T, 2>>,
{
    type Scalar = T;

    fn new(vector: [Self::Scalar; 2]) -> Self {
        Self(Simd::from_array(vector))
    }

    fn to_array(self) -> [Self::Scalar; 2] {
        Simd::to_array(self.0)
    }

    fn hadamard_product(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }

    fn reduce_sum(self) -> Self::Scalar {
        <Simd<T, 2> as SimdFloat>::reduce_sum(self.0)
    }

    #[cfg(feature = "std")]
    fn mul_add(self, b: Self::Scalar, c: Self) -> Self
    where
        Self::Scalar: Float,
    {
        SimdVector(Simd::mul_add(self.0, Simd::splat(b), c.0))
    }

    fn x(self) -> Self::Scalar {
        self.to_array()[0]
    }

    fn y(self) -> Self::Scalar {
        self.to_array()[1]
    }

    fn update_xy(self, x: Self::Scalar, y: Self::Scalar) -> Self {
        Self::new([x, y])
    }
}

impl<T: 'static + SimdElement + Float + ConstZero + core::fmt::Debug> Vector<3> for SimdVector<T, 4>
where
    Simd<T, 4>: SimdFloat<Scalar = T> + StdFloat + Ring + Div<Output = Simd<T, 4>>,
{
    type Scalar = T;

    fn new([x, y, z]: [Self::Scalar; 3]) -> Self {
        Self(Simd::from_array([x, y, z, T::ZERO]))
    }

    fn to_array(self) -> [Self::Scalar; 3] {
        let [x, y, z, _] = Simd::to_array(self.0);
        [x, y, z]
    }

    fn hadamard_product(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }

    fn reduce_sum(self) -> Self::Scalar {
        <Simd<T, 4> as SimdFloat>::reduce_sum(self.0)
    }

    #[cfg(feature = "std")]
    fn mul_add(self, b: Self::Scalar, c: Self) -> Self
    where
        Self::Scalar: Float,
    {
        SimdVector(Simd::mul_add(self.0, Simd::splat(b), c.0))
    }

    fn x(self) -> Self::Scalar {
        self.to_array()[0]
    }

    fn y(self) -> Self::Scalar {
        self.to_array()[1]
    }

    fn update_xy(self, x: Self::Scalar, y: Self::Scalar) -> Self {
        let [_, _, z, _] = Simd::to_array(self.0);
        Self::new([x, y, z])
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom, Deref, DerefMut)]
#[wrapped_from(wrapped = "crate::LossyFrom::lossy_from")]
pub struct SimpleVector<T, const N: usize>(pub [T; N]);

impl<T, const N: usize> SimpleVector<T, N> {
    fn map<R, F: FnMut(T) -> R>(self, f: F) -> SimpleVector<R, N> {
        let SimpleVector(array) = self;
        SimpleVector(array.map(f))
    }

    fn zip<U>(self, other: SimpleVector<U, N>) -> SimpleVector<(T, U), N> {
        let SimpleVector(lhs) = self;
        let SimpleVector(rhs) = other;
        SimpleVector(lhs.zip(rhs))
    }
}

impl<L, R, const N: usize> SimpleVector<(L, R), N> {
    fn map2<T, F: FnMut(L, R) -> T>(self, mut f: F) -> SimpleVector<T, N> {
        let SimpleVector(array) = self;
        SimpleVector(array.map(move |(l, r)| f(l, r)))
    }
}

impl<T: ConstZero, const N: usize> ConstZero for SimpleVector<T, N> {
    const ZERO: Self = SimpleVector([T::ZERO; N]);
}

impl<
        T: 'static + Copy + core::fmt::Debug + core::cmp::PartialEq + Ring + Div<Output = T>,
        const DIM: usize,
    > Vector<DIM> for SimpleVector<T, DIM>
{
    type Scalar = T;

    fn new(vector: [Self::Scalar; DIM]) -> Self {
        SimpleVector(vector)
    }

    fn to_array(self) -> [Self::Scalar; DIM] {
        self.0
    }

    fn hadamard_product(self, rhs: Self) -> Self {
        self.zip(rhs).map2(Mul::mul)
    }

    fn reduce_sum(self) -> Self::Scalar {
        self.0.iter().copied().fold(T::ZERO, Add::add)
    }

    fn mul_add(self, b: Self::Scalar, c: Self) -> Self
    where
        Self::Scalar: Float,
    {
        self.zip(c).map(|(a, c)| a.mul_add(b, c))
    }

    fn x(self) -> Self::Scalar {
        self.0[0]
    }

    fn y(self) -> Self::Scalar {
        self.0[1]
    }

    fn update_xy(mut self, x: Self::Scalar, y: Self::Scalar) -> Self {
        self[0] = x;
        self[1] = y;
        self
    }
}

impl<T: Neg<Output = T>, const N: usize> Neg for SimpleVector<T, N> {
    type Output = SimpleVector<T, N>;

    fn neg(self) -> Self::Output {
        // Self(self.0.map(Neg::neg))
        self.map(Neg::neg)
    }
}

impl<T: Add<Output = T>, const N: usize> Add<SimpleVector<T, N>> for SimpleVector<T, N> {
    type Output = SimpleVector<T, N>;

    fn add(self, rhs: SimpleVector<T, N>) -> Self::Output {
        // Self(self.0.zip(rhs.0).map(|(l, r)| l + r))
        self.zip(rhs).map2(Add::add)
    }
}

impl<T: Sub<Output = T>, const N: usize> Sub<SimpleVector<T, N>> for SimpleVector<T, N> {
    type Output = SimpleVector<T, N>;

    fn sub(self, rhs: SimpleVector<T, N>) -> Self::Output {
        // Self(self.0.zip(rhs.0).map(|(l, r)| l - r))
        self.zip(rhs).map2(Sub::sub)
    }
}

impl<T: Copy + Mul<Output = T>, const N: usize> Mul<T> for SimpleVector<T, N> {
    type Output = SimpleVector<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        // Self(self.0.map(|lhs| lhs * rhs))
        self.map(|lhs| lhs * rhs)
    }
}

impl<T: Copy + Div<Output = T>, const N: usize> Div<T> for SimpleVector<T, N> {
    type Output = SimpleVector<T, N>;

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

pub fn oproj<T: Copy + Ring, V: Vector<N, Scalar = T>, const N: usize>(
    a: V,
    UnitVector(b): UnitVector<V>,
) -> V {
    a - b * (a.dot(b))
}

/// $\| a x b \|^2$
pub fn cross_prod_magnitude_sq<T: Copy + Ring, V: Vector<N, Scalar = T>, const N: usize>(
    a: V,
    UnitVector(b): UnitVector<V>,
) -> T {
    a.norm_squared() - T::sqr(a.dot(b))
}

unsafe impl<T: rustacuda_core::DeviceCopy + SimdElement, const N: usize> rustacuda_core::DeviceCopy
    for SimdVector<T, N>
where
    LaneCount<N>: SupportedLaneCount,
{
}

unsafe impl<T: rustacuda_core::DeviceCopy, const N: usize> rustacuda_core::DeviceCopy
    for SimpleVector<T, N>
{
}
