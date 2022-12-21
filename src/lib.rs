#![feature(array_methods, ptr_metadata)]
#![allow(
    clippy::blocks_in_if_conditions,
    clippy::range_plus_one,
    clippy::excessive_precision
)]

#[cfg(feature = "pyext")]
#[macro_use]
extern crate derive_more;
#[cfg(feature = "pyext")]
#[macro_use]
extern crate derive_wrapped_from;

#[cfg(feature = "cuda")]
mod cuda_fitness;
mod fitness;
#[cfg(feature = "pyext")]
mod pylib;

#[cfg(feature = "cuda")]
pub use crate::cuda_fitness::*;
pub use crate::fitness::*;
pub use compound_prism_spectrometer::*;

pub trait SpectrometerFitness<T: FloatExt, const D: usize> {
    fn fitness(&self, max_n: usize, max_m: usize) -> DesignFitness<T>;

    #[cfg(feature = "cuda")]
    fn cuda_fitness(
        &self,
        seeds: &[f64],
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> rustacuda::error::CudaResult<Option<DesignFitness<T>>>;

    fn p_dets_l_wavelength(&self, wavelength: T, max_n: usize) -> Vec<T>;
    fn propagation_path(&self, ray: Ray<T, D>, wavelength: T) -> Vec<GeometricRay<T, D>>;
}

#[cfg(not(feature = "cuda"))]
impl<F: FloatExt, S: GenericSpectrometer<F, D>, const D: usize> SpectrometerFitness<F, D> for S {
    fn fitness(&self, max_n: usize, max_m: usize) -> DesignFitness<F> {
        crate::fitness::fitness(self, max_n, max_m)
    }

    fn p_dets_l_wavelength(&self, wavelength: F, max_n: usize) -> Vec<F> {
        crate::fitness::p_dets_l_wavelength(self, wavelength, max_n).collect()
    }

    fn propagation_path(&self, ray: Ray<F, D>, wavelength: F) -> Vec<GeometricRay<F, D>> {
        self.propagation_path(ray, wavelength).collect()
    }
}

#[cfg(feature = "cuda")]
impl<
        T: FloatExt + rustacuda::memory::DeviceCopy,
        S: ?Sized + GenericSpectrometer<T, D> + Kernel,
        const D: usize,
    > SpectrometerFitness<T, D> for S
{
    fn fitness(&self, max_n: usize, max_m: usize) -> DesignFitness<T> {
        crate::fitness::fitness(self, max_n, max_m)
    }

    fn cuda_fitness(
        &self,
        seeds: &[f64],
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> rustacuda::error::CudaResult<Option<DesignFitness<T>>> {
        crate::cuda_fitness::cuda_fitness(self, seeds, max_n, nwarp, max_eval)
    }

    fn p_dets_l_wavelength(&self, wavelength: T, max_n: usize) -> Vec<T> {
        crate::fitness::p_dets_l_wavelength(self, wavelength, max_n).collect()
    }

    fn propagation_path(&self, ray: Ray<T, D>, wavelength: T) -> Vec<GeometricRay<T, D>> {
        self.propagation_path(ray, wavelength).collect()
    }
}
