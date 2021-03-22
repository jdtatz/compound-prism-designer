use core::{marker::PhantomData, ops::*};

use crate::utils::{Float, LossyFrom};

pub trait Distribution {
    type Item;
    fn inverse_cdf(&self, p: Self::Item) -> Self::Item;
}

#[derive(Debug, Clone, Copy)]
pub struct DiracDeltaDistribution<T> {
    pub value: T,
}

impl<T: Copy> Distribution for DiracDeltaDistribution<T> {
    type Item = T;

    fn inverse_cdf(&self, _p: Self::Item) -> Self::Item {
        self.value
    }
}

#[derive(Debug, Clone, Copy)]
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

#[derive(Debug, Clone, Copy)]
pub struct NormalDistribution<T> {
    pub mean: T,
    pub stddev: T,
}

impl<F: Float> Distribution for NormalDistribution<F> {
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

impl<T, U: LossyFrom<T>> LossyFrom<DiracDeltaDistribution<T>> for DiracDeltaDistribution<U> {
    fn lossy_from(v: DiracDeltaDistribution<T>) -> Self {
        Self {
            value: LossyFrom::lossy_from(v.value),
        }
    }
}

impl<T, U: LossyFrom<T>> LossyFrom<UniformDistribution<T>> for UniformDistribution<U> {
    fn lossy_from(v: UniformDistribution<T>) -> Self {
        Self {
            bounds: LossyFrom::lossy_from(v.bounds),
        }
    }
}

impl<T, U: LossyFrom<T>> LossyFrom<NormalDistribution<T>> for NormalDistribution<U> {
    fn lossy_from(v: NormalDistribution<T>) -> Self {
        Self {
            mean: LossyFrom::lossy_from(v.mean),
            stddev: LossyFrom::lossy_from(v.stddev),
        }
    }
}
