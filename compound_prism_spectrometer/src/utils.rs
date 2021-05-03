use core::mem::swap;
pub use num_traits::{Float, NumAssign, One, Zero};

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

impl<T, U> LossyFrom<core::marker::PhantomData<T>> for core::marker::PhantomData<U> {
    fn lossy_from(_: core::marker::PhantomData<T>) -> Self {
        core::marker::PhantomData
    }
}

impl<T, U: LossyFrom<T>, const N: usize> LossyFrom<[T; N]> for [U; N] {
    fn lossy_from(t: [T; N]) -> Self {
        t.map(LossyFrom::lossy_from)
    }
}

wrapped_from_tuples! { LossyFrom::lossy_from for 0..12 }

pub trait ConstZero {
    const ZERO: Self;
}

pub trait ConstOne {
    const ONE: Self;
}

pub trait Ring:
    Sized
    + ConstZero
    + ConstOne
    + core::ops::Neg<Output = Self>
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
{
    fn sqr(self) -> Self
    where
        Self: Copy,
    {
        self * self
    }
}
impl<T> Ring for T where
    T: Sized
        + ConstZero
        + ConstOne
        + core::ops::Neg<Output = Self>
        + core::ops::Add<Output = Self>
        + core::ops::Sub<Output = Self>
        + core::ops::Mul<Output = Self>
{
}

pub trait ApproxEq:
    float_eq::FloatEqUlpsTol
    + float_eq::FloatEq<Tol = Self>
    + float_eq::FloatEqDebugUlpsDiff<DebugUlpsDiff = Self::DebugUlpsDiffTy>
    + float_eq::AssertFloatEq<DebugAbsDiff = Self, DebugTol = Self>
{
    type DebugUlpsDiffTy: core::fmt::Debug;
}

impl<T> ApproxEq for T
where
    T: float_eq::FloatEqUlpsTol
        + float_eq::FloatEq<Tol = Self>
        + float_eq::FloatEqDebugUlpsDiff
        + float_eq::AssertFloatEq<DebugAbsDiff = Self, DebugTol = Self>,
    <T as float_eq::FloatEqDebugUlpsDiff>::DebugUlpsDiff: core::fmt::Debug,
{
    type DebugUlpsDiffTy = <T as float_eq::FloatEqDebugUlpsDiff>::DebugUlpsDiff;
}

pub trait FloatExt:
    'static
    + Float
    + NumAssign
    + LossyFrom<u32>
    + LossyInto<u32>
    + LossyFrom<f64>
    + LossyInto<f64>
    + core::fmt::Debug
    + core::fmt::Display
    + core::fmt::LowerExp
    + ConstZero
    + ConstOne
    + ApproxEq
{
    type BitRepr: 'static + Sized + Copy + core::fmt::Debug + num_traits::Unsigned;

    fn to_bits(self) -> Self::BitRepr;
    fn from_bits(bits: Self::BitRepr) -> Self;
    fn copy_sign(self, other: Self) -> Self;

    fn sincos(self) -> (Self, Self) {
        self.sin_cos()
    }
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
    fn sqr(self) -> Self {
        self * self
    }
    fn plog2p(self) -> Self {
        if self > Self::zero() {
            self * self.log2()
        } else {
            Self::zero()
        }
    }
}

impl ConstZero for f32 {
    const ZERO: Self = 0f32;
}

impl ConstOne for f32 {
    const ONE: Self = 1f32;
}

impl FloatExt for f32 {
    type BitRepr = u32;

    fn to_bits(self) -> Self::BitRepr {
        self.to_bits()
    }

    fn from_bits(bits: Self::BitRepr) -> Self {
        f32::from_bits(bits)
    }

    fn copy_sign(self, other: Self) -> Self {
        libm::copysignf(self, other)
    }
}

impl ConstZero for f64 {
    const ZERO: Self = 0f64;
}

impl ConstOne for f64 {
    const ONE: Self = 1f64;
}

impl FloatExt for f64 {
    type BitRepr = u64;

    fn to_bits(self) -> Self::BitRepr {
        self.to_bits()
    }

    fn from_bits(bits: Self::BitRepr) -> Self {
        f64::from_bits(bits)
    }

    fn copy_sign(self, other: Self) -> Self {
        libm::copysign(self, other)
    }
}

#[cfg(target_arch = "nvptx64")]
impl<T, F: LossyFrom<T>> LossyFrom<T> for nvptx_sys::FastFloat<F> {
    fn lossy_from(v: T) -> Self {
        Self(LossyFrom::lossy_from(v))
    }
}

#[cfg(target_arch = "nvptx64")]
impl<F: Copy + LossyInto<u32>> LossyInto<u32> for nvptx_sys::FastFloat<F> {
    fn lossy_into(self) -> u32 {
        LossyInto::lossy_into(*self)
    }
}

#[cfg(target_arch = "nvptx64")]
impl<F: Copy + LossyInto<f64>> LossyInto<f64> for nvptx_sys::FastFloat<F> {
    fn lossy_into(self) -> f64 {
        LossyInto::lossy_into(*self)
    }
}

#[cfg(target_arch = "nvptx64")]
impl<F: ConstZero> ConstZero for nvptx_sys::FastFloat<F> {
    const ZERO: Self = nvptx_sys::FastFloat(F::ZERO);
}

#[cfg(target_arch = "nvptx64")]
impl<F: ConstOne> ConstOne for nvptx_sys::FastFloat<F> {
    const ONE: Self = nvptx_sys::FastFloat(F::ONE);
}

#[cfg(target_arch = "nvptx64")]
impl<F: nvptx_sys::FastNum + FloatExt> FloatExt for nvptx_sys::FastFloat<F>
where
    <F as float_eq::FloatEqUlpsTol>::UlpsTol: Sized,
{
    type BitRepr = F::BitRepr;

    fn to_bits(self) -> Self::BitRepr {
        <F as FloatExt>::to_bits(self.0)
    }

    fn from_bits(bits: Self::BitRepr) -> Self {
        nvptx_sys::FastFloat(<F as FloatExt>::from_bits(bits))
    }

    fn copy_sign(self, other: Self) -> Self {
        self.copysign(other)
    }
}
