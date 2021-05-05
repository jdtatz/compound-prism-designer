#![cfg_attr(
    all(any(not(feature = "std"), target_arch = "nvptx64"), not(test)),
    no_std
)]
#![feature(array_map, array_zip)]
#![allow(
    clippy::blocks_in_if_conditions,
    clippy::range_plus_one,
    clippy::excessive_precision
)]
#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate derive_wrapped_from;

mod cbrt;
mod distribution;
mod erf;
mod geometry;
mod glasscat;
mod qrng;
mod ray;
pub mod roots;
mod spectrometer;
mod toric;
mod utils;
mod vector;
mod welford;
pub use crate::distribution::{
    DiracDeltaDistribution, Distribution, NormalDistribution, UniformDistribution, UserDistribution,
};
pub use crate::erf::norminv;
pub use crate::geometry::*;
pub use crate::glasscat::Glass;
pub use crate::qrng::{Qrng, QuasiRandom};
pub use crate::ray::{CompoundPrism, DetectorArray, Ray, RayTraceError};
pub use crate::spectrometer::{
    detector_array_positioning, Beam, FiberBeam, GaussianBeam, LinearDetectorArray, Spectrometer,
};
pub use crate::toric::{ToricLens, ToricLensParametrization, ToricSurface};
pub use crate::utils::{Float, FloatExt, LossyFrom, LossyInto, NumAssign, One, Zero};
pub use crate::vector::{UnitVector, Vector};
pub use crate::welford::Welford;
