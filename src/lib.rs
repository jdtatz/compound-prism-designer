#![allow(clippy::block_in_if_condition_stmt)]
#[macro_use]
extern crate derive_more;

pub mod optimizer;
mod erf;
mod glasscat;
mod ray;
mod utils;
#[cfg(not(feature="pyext"))]
mod clib;
#[cfg(feature="pyext")]
mod pylib;
pub use crate::glasscat::*;
pub use crate::ray::*;
