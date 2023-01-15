use core::{marker::PhantomData, ops::*};

use crate::FloatExt;

pub trait Distribution<T, Output = T> {
    fn inverse_cdf(&self, p: T) -> Output;
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct DiracDeltaDistribution<T> {
    pub value: T,
}

impl<T: Copy> Distribution<T> for DiracDeltaDistribution<T> {
    fn inverse_cdf(&self, _p: T) -> T {
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
    fn inverse_cdf(&self, p: T) -> T {
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
    fn inverse_cdf(&self, p: F) -> F {
        self.mean + self.stddev * super::erf::fast_norminv(p)
    }
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct UniformDiscDistribution<T> {
    pub radius: T,
}

impl<F: FloatExt> Distribution<[F; 2]> for UniformDiscDistribution<F> {
    fn inverse_cdf(&self, p: [F; 2]) -> [F; 2] {
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
    fn inverse_cdf(&self, p: T) -> T {
        (self.quantile)(p)
    }
}
