#![allow(clippy::block_in_if_condition_stmt, clippy::range_plus_one)]
#[macro_use]
extern crate derive_more;

mod erf;
mod glasscat;
pub mod optimizer;
#[cfg(feature = "pyext")]
mod pylib;
mod qrng;
mod ray;
#[macro_use]
mod utils;
pub use crate::glasscat::*;
pub use crate::ray::*;
