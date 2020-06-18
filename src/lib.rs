#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(
    target_arch = "nvptx64",
    feature(abi_ptx, core_intrinsics, stdsimd, link_llvm_intrinsics)
)]
#![allow(clippy::blocks_in_if_conditions, clippy::range_plus_one, clippy::excessive_precision)]
#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate serde;

pub mod erf;
pub mod geom;
mod glasscat;
#[cfg(all(not(target_arch = "nvptx64"), feature = "pyext"))]
mod pylib;
pub mod qrng;
mod ray;
#[macro_use]
pub mod utils;
#[cfg(all(not(target_arch = "nvptx64"), feature = "cuda"))]
pub mod cuda_fitness;
#[cfg(not(target_arch = "nvptx64"))]
mod fitness;
#[cfg(not(target_arch = "nvptx64"))]
pub use crate::fitness::*;
pub use crate::glasscat::*;
pub use crate::ray::*;
#[cfg(target_arch = "nvptx64")]
mod kernel;
