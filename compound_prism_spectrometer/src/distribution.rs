use core::{marker::PhantomData, ops::*};

use crate::utils::*;

pub trait Distribution<T> {
    type Output;
    fn inverse_cdf(&self, p: T) -> Self::Output;
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct DiracDeltaDistribution<T> {
    pub value: T,
}

impl<T: Copy> Distribution<T> for DiracDeltaDistribution<T> {
    type Output = T;
    fn inverse_cdf(&self, _p: T) -> Self::Output {
        self.value
    }
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct UniformDistribution<T> {
    pub bounds: (T, T),
}

impl<T> Distribution<T> for UniformDistribution<T>
where
    T: Sized + Copy + Add<Output = T> + Sub<Output = T> + Mul<Output = T>,
{
    type Output = T;

    fn inverse_cdf(&self, p: T) -> Self::Output {
        self.bounds.0 + (self.bounds.1 - self.bounds.0) * p
    }
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct NormalDistribution<T> {
    pub mean: T,
    pub stddev: T,
}

impl<F: FloatExt> Distribution<F> for NormalDistribution<F> {
    type Output = F;

    fn inverse_cdf(&self, p: F) -> Self::Output {
        self.mean + self.stddev * crate::erf::fast_norminv(p)
    }
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct UniformDiscDistribution<T> {
    pub radius: T,
}

impl<F: FloatExt> Distribution<[F; 2]> for UniformDiscDistribution<F> {
    type Output = [F; 2];

    fn inverse_cdf(&self, p: [F; 2]) -> Self::Output {
        let [p_r, p_theta] = p;
        let (s, c) = (p_theta * F::lossy_from(core::f64::consts::TAU)).sin_cos();
        let r = self.radius * p_r.sqrt();
        [r * c, r * s]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UserDistribution<T, F: Fn(T) -> T> {
    pub quantile: F,
    pub marker: PhantomData<T>,
}

impl<T, F: Fn(T) -> T> Distribution<T> for UserDistribution<T, F> {
    type Output = T;

    fn inverse_cdf(&self, p: T) -> Self::Output {
        (self.quantile)(p)
    }
}
