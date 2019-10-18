#![allow(clippy::block_in_if_condition_stmt)]
#[macro_use]
extern crate derive_more;

mod clib;
mod erf;
mod glasscat;
mod ray;
pub use crate::glasscat::*;
pub use crate::ray::*;
