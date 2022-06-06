#![cfg_attr(
    all(
        any(not(feature = "std"), target_arch = "nvptx64", target_arch = "spirv"),
        not(test)
    ),
    no_std
)]
#![feature(array_zip, type_alias_impl_trait)]
#![feature(maybe_uninit_uninit_array)]
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
mod drawable;
mod erf;
mod geometry;
mod glasscat;
pub mod kernel;
mod qrng;
mod ray;
mod spectrometer;
mod toric;
mod utils;
mod vector;
mod welford;
pub use crate::distribution::{
    DiracDeltaDistribution, Distribution, NormalDistribution, UniformDistribution, UserDistribution,
};
pub use crate::drawable::{
    arc_as_2_cubic_béziers, arc_as_cubic_bézier, Drawable, Path, Point, Polygon,
};
pub use crate::erf::norminv;
pub use crate::geometry::*;
pub use crate::glasscat::Glass;
pub use crate::qrng::{Qrng, QuasiRandom};
pub use crate::ray::{CompoundPrism, DetectorArray, Ray, RayTraceError};
pub use crate::spectrometer::{
    detector_array_positioning, Beam, FiberBeam, GaussianBeam, GenericSpectrometer,
    LinearDetectorArray, Spectrometer,
};
pub use crate::toric::{ToricLens, ToricLensParametrization, ToricSurface};
pub use crate::utils::{Float, FloatExt, LossyFrom, LossyInto, NumAssign, One, Zero};
pub use crate::vector::{UnitVector, Vector};
pub use crate::welford::Welford;
