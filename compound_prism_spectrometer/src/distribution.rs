use core::{marker::PhantomData, ops::*};

use crate::{utils::Float, LossyInto};

pub trait Distribution {
    type Item;
    fn inverse_cdf(&self, p: Self::Item) -> Self::Item;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DiracDeltaDistribution<T> {
    pub value: T,
}

impl<T: Copy> Distribution for DiracDeltaDistribution<T> {
    type Item = T;

    fn inverse_cdf(&self, _p: Self::Item) -> Self::Item {
        self.value
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

impl<T: LossyInto<U>, U> LossyInto<DiracDeltaDistribution<U>> for DiracDeltaDistribution<T> {
    fn lossy_into(self) -> DiracDeltaDistribution<U> {
        DiracDeltaDistribution {
            value: self.value.lossy_into(),
        }
    }
}

impl<T: LossyInto<U>, U> LossyInto<UniformDistribution<U>> for UniformDistribution<T> {
    fn lossy_into(self) -> UniformDistribution<U> {
        UniformDistribution {
            bounds: self.bounds.lossy_into(),
        }
    }
}

impl<T: LossyInto<U>, U> LossyInto<NormalDistribution<U>> for NormalDistribution<T> {
    fn lossy_into(self) -> NormalDistribution<U> {
        NormalDistribution {
            mean: self.mean.lossy_into(),
            stddev: self.stddev.lossy_into(),
        }
    }
}
