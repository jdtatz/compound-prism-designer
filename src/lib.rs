#![cfg_attr(
    all(any(not(feature = "std"), target_arch = "nvptx64"), not(test)),
    no_std
)]
#![cfg_attr(
    target_arch = "nvptx64",
    feature(abi_ptx, platform_intrinsics, address_space, link_llvm_intrinsics)
)]
#![feature(portable_simd)]
#![feature(type_alias_impl_trait, impl_trait_in_assoc_type)]
#![feature(
    maybe_uninit_uninit_array,
    maybe_uninit_array_assume_init,
    array_methods,
    ptr_metadata
)]
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
pub trait SpectrometerFitness<V: Vector<DIM>, const DIM: usize>:
    GenericSpectrometer<V, DIM>
{
    fn fitness(&self, max_n: usize, max_m: usize) -> DesignFitness<V::Scalar>;

    #[cfg(feature = "cuda")]
    fn cuda_fitness(
        &self,
        seeds: &[f64],
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> rustacuda::error::CudaResult<Option<DesignFitness<V::Scalar>>>
    where
        Self: Kernel<V, DIM>;

    fn p_dets_l_wavelength(&self, wavelength: V::Scalar, max_n: usize) -> Vec<V::Scalar>;
    fn propagation_path(
        &self,
        ray: Ray<V, DIM>,
        wavelength: V::Scalar,
    ) -> Vec<GeometricRay<V, DIM>>;
}

#[cfg(feature = "std")]
impl<
    T: FloatExt + rustacuda::memory::DeviceCopy,
    V: Vector<D, Scalar = T>,
    S: ?Sized + GenericSpectrometer<V, D>,
    const D: usize,
> SpectrometerFitness<V, D> for S
{
    fn fitness(&self, max_n: usize, max_m: usize) -> DesignFitness<T> {
        crate::fitness::fitness(self, max_n, max_m)
    }

    #[cfg(feature = "cuda")]
    fn cuda_fitness(
        &self,
        seeds: &[f64],
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> rustacuda::error::CudaResult<Option<DesignFitness<T>>>
    where
        Self: Kernel<V, D>,
    {
        crate::cuda_fitness::cuda_fitness(self, seeds, max_n, nwarp, max_eval)
    }

    fn p_dets_l_wavelength(&self, wavelength: T, max_n: usize) -> Vec<T> {
        crate::fitness::p_dets_l_wavelength(self, wavelength, max_n).collect()
    }

    fn propagation_path(&self, ray: Ray<V, D>, wavelength: T) -> Vec<GeometricRay<V, D>> {
        self.propagation_path(ray, wavelength).collect()
    }
}
