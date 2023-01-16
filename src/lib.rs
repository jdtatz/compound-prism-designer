#![cfg_attr(
    all(any(not(feature = "std"), target_arch = "nvptx64"), not(test)),
    no_std
)]
#![cfg_attr(
    target_arch = "nvptx64",
    feature(abi_ptx, platform_intrinsics, address_space)
)]
#![feature(portable_simd)]
#![feature(array_zip, type_alias_impl_trait)]
#![feature(array_methods, ptr_metadata)]
#![allow(
    clippy::blocks_in_if_conditions,
    clippy::range_plus_one,
    clippy::excessive_precision
)]

#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate derive_wrapped_from;

#[cfg(feature = "cuda")]
mod cuda_fitness;
#[cfg(feature = "std")]
mod fitness;
mod kernel;
#[cfg(feature = "pyext")]
mod pylib;
mod spectrometer;

#[cfg(feature = "cuda")]
pub use crate::cuda_fitness::*;
#[cfg(feature = "std")]
pub use crate::fitness::*;
pub use crate::spectrometer::*;

#[cfg(feature = "std")]
pub trait SpectrometerFitness<V: Vector<DIM, Scalar = Self::Scalar>, const DIM: usize>:
    GenericSpectrometer<V, DIM>
where
    <Self as GenericSpectrometer<V, DIM>>::Scalar: FloatExt,
{
    fn fitness(&self, max_n: usize, max_m: usize) -> DesignFitness<Self::Scalar>;

    #[cfg(feature = "cuda")]
    fn cuda_fitness(
        &self,
        seeds: &[f64],
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> rustacuda::error::CudaResult<Option<DesignFitness<Self::Scalar>>>;

    fn p_dets_l_wavelength(&self, wavelength: Self::Scalar, max_n: usize) -> Vec<Self::Scalar>;
    fn propagation_path(
        &self,
        ray: Ray<V, DIM>,
        wavelength: Self::Scalar,
    ) -> Vec<GeometricRay<V, DIM>>;
}

#[cfg(feature = "std")]
#[cfg(not(feature = "cuda"))]
impl<
        F: FloatExt,
        V: Vector<D, Scalar = F>,
        S: GenericSpectrometer<V, D, Scalar = F>,
        const D: usize,
    > SpectrometerFitness<V, D> for S
{
    fn fitness(&self, max_n: usize, max_m: usize) -> DesignFitness<F> {
        crate::fitness::fitness(self, max_n, max_m)
    }

    fn p_dets_l_wavelength(&self, wavelength: F, max_n: usize) -> Vec<F> {
        crate::fitness::p_dets_l_wavelength(self, wavelength, max_n).collect()
    }

    fn propagation_path(&self, ray: Ray<V, D>, wavelength: F) -> Vec<GeometricRay<V, D>> {
        self.propagation_path(ray, wavelength).collect()
    }
}

#[cfg(feature = "std")]
#[cfg(feature = "cuda")]
impl<
        T: FloatExt + rustacuda::memory::DeviceCopy,
        V: Vector<D, Scalar = T>,
        S: ?Sized + Kernel<V, D, Scalar = T>,
        const D: usize,
    > SpectrometerFitness<V, D> for S
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

    fn propagation_path(&self, ray: Ray<V, D>, wavelength: T) -> Vec<GeometricRay<V, D>> {
        self.propagation_path(ray, wavelength).collect()
    }
}
