#![allow(
    clippy::blocks_in_if_conditions,
    clippy::range_plus_one,
    clippy::excessive_precision
)]

#[macro_use]
extern crate serde;

#[cfg(feature = "cuda")]
mod cuda_fitness;
mod fitness;
#[cfg(feature = "pyext")]
mod pylib;

#[cfg(feature = "cuda")]
pub use crate::cuda_fitness::*;
pub use crate::fitness::*;
