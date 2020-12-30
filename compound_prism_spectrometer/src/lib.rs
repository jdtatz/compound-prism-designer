#![cfg_attr(any(not(feature = "std"), target_arch = "nvptx64"), no_std)]
#![cfg_attr(target_arch = "nvptx64", feature(core_intrinsics))]
#![allow(
    clippy::blocks_in_if_conditions,
    clippy::range_plus_one,
    clippy::excessive_precision
)]
#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate serde;

mod erf;
mod geom;
mod glasscat;
mod qrng;
mod ray;
#[macro_use]
pub mod utils;
mod spectrometer;
mod welford;
pub use crate::erf::norminv;
pub use crate::geom::{CurvedPlane, Pair, Plane, Surface, Triplet, Vector};
pub use crate::glasscat::{new_catalog, CatalogError, Glass, BUNDLED_CATALOG};
pub use crate::qrng::{Qrng, QuasiRandom};
pub use crate::ray::{Beam, CompoundPrism, DetectorArray, Ray, RayTraceError};
pub use crate::spectrometer::{GaussianBeam, LinearDetectorArray, Spectrometer};
pub use crate::utils::{Float, LossyInto};
pub use crate::welford::Welford;