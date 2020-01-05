#![allow(clippy::block_in_if_condition_stmt, clippy::range_plus_one)]
#[macro_use]
extern crate derive_more;
use std::{
    error::Error,
    fs::{read, File},
};
use flate2::write::GzEncoder;
use flate2::Compression;


mod erf;
mod glasscat;
mod optimizer;
mod qrng;
mod ray;
#[macro_use]
mod utils;

use crate::optimizer::*;
use crate::ray::*;

#[derive(serde::Serialize)]
struct DesignOutput {
    specification: DesignConfig,
    designs: Vec<Design>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let file = read("design_config.toml")?;
    let config: DesignConfig = toml::from_slice(file.as_ref())?;
    let designs = config.optimize_designs(None);

    let out = (config, designs);
    let file = File::create("results.cbor.gz")?;
    let gz = GzEncoder::new(file, Compression::default());
    serde_cbor::to_writer(gz, &out)?;

    for design in out.1.iter() {
        println!("{:?}", design.fitness);
        if design.fitness.info > 3.7 {
            println!("{:#?}", design);
        }
    }
    Ok(())
}
