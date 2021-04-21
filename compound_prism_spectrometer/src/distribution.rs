use core::{marker::PhantomData, ops::*};

use crate::utils::*;

pub trait Distribution {
    type Item;
    fn inverse_cdf(&self, p: Self::Item) -> Self::Item;
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct DiracDeltaDistribution<T> {
    pub value: T,
}

impl<T: Copy> Distribution for DiracDeltaDistribution<T> {
    type Item = T;

    fn inverse_cdf(&self, _p: Self::Item) -> Self::Item {
        self.value
    }
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct UniformDistribution<T> {
    pub bounds: (T, T),
}

impl<T> Distribution for UniformDistribution<T>
where
    T: Sized + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Item = T;

    fn inverse_cdf(&self, p: Self::Item) -> Self::Item {
        self.bounds.0 + (self.bounds.1 - self.bounds.0) * p
    }
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct NormalDistribution<T> {
    pub mean: T,
    pub stddev: T,
}

impl<F: FloatExt> Distribution for NormalDistribution<F> {
    type Item = F;

    fn inverse_cdf(&self, p: Self::Item) -> Self::Item {
        self.mean + self.stddev * crate::erf::fast_norminv(p)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UserDistribution<T, F: Fn(T) -> T> {
    pub quantile: F,
    pub marker: PhantomData<T>,
}

impl<T, F: Fn(T) -> T> Distribution for UserDistribution<T, F> {
    type Item = T;

    fn inverse_cdf(&self, p: Self::Item) -> Self::Item {
        (self.quantile)(p)
    }
}
