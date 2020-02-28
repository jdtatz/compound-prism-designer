#![allow(clippy::block_in_if_condition_stmt, clippy::range_plus_one)]
#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate serde;

use flate2::write::GzEncoder;
use flate2::Compression;
use std::{
    error::Error,
    fs::{read, File},
};

mod erf;
mod glasscat;
mod optimizer;
mod qrng;
mod ray;
#[macro_use]
mod utils;
#[cfg(feature = "cuda")]
mod cuda_fitness;
mod fitness;

use crate::glasscat::{BUNDLED_CATALOG, Glass};
use crate::optimizer::*;
use crate::ray::{CompoundPrism, GaussianBeam, LinearDetectorArray, Spectrometer};
use rand::{Rng, SeedableRng};
use std::f64::consts::*;
use crate::fitness::DesignFitness;
use crate::ray::RayTraceError;

#[derive(serde::Serialize)]
struct DesignOutput {
    specification: DesignConfig,
    designs: Vec<Design>,
}

const epsilons: [f64; 3] = [2.5, 0.02, 0.1];

struct DynamicTest(Vec<Glass<f64>>);

#[derive(Clone, Copy)]
struct DynamicParams<'s> {
    height: f64,
    normalized_y_mean: f64,
    curvature: f64,
    detector_array_angle: f64,
    lengths: &'s [f64],
    sine_angles: &'s [f64],
}

impl std::fmt::Debug for DynamicParams<'_> {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        let y_mean = self.height * self.normalized_y_mean;
        let detector_array_angle = self.detector_array_angle.to_degrees();
        let angles: Vec<_> = self.sine_angles.iter().map(|a| a.asin().to_degrees()).collect();
        fmt.debug_struct("DynamicParams")
            .field("height", &self.height)
            .field("y_mean", &y_mean)
            .field("curvature", &self.curvature)
            .field("detector_array_angle", &detector_array_angle)
            .field("lengths", &self.lengths)
            .field("angles", &angles)
            .finish()
    }
}

impl DynamicTest {
    fn slice_into_params<'s>(&self, params: &'s [f64]) -> DynamicParams<'s> {
        let (&height, params) = params.split_first().unwrap();
        let (&normalized_y_mean, params) = params.split_first().unwrap();
        let (&curvature, params) = params.split_first().unwrap();
        let (&detector_array_angle, params) = params.split_first().unwrap();
        let (lengths, params) = params.split_at(self.0.len());
        let sine_angles = params;
        DynamicParams {
            height,
            normalized_y_mean,
            curvature,
            detector_array_angle,
            lengths,
            sine_angles,
        }
    }

    fn params_into_spectrometer(&self, params: DynamicParams) -> Result<Spectrometer<f64>, RayTraceError> {
        let angles: Vec<_> = params.sine_angles.iter().map(|a| a.asin()).collect();
        let lengths: Vec<_> = params.lengths.iter().copied().map(|x| (params.height / 9.0) * x / (1.0 - x)).collect();
        let beam = GaussianBeam {
            width: 3.2,
            y_mean: params.normalized_y_mean * params.height,
            w_range: (0.5, 0.82),
        };
        let cmpnd = CompoundPrism::<f64>::new(
            self.0.iter().cloned(),
            &angles,
            &lengths,
            params.curvature,
            params.height,
            7.,
            true,
        );
        let detarr =
            LinearDetectorArray::new(
                32,
                0.8,
                1.,
                0.1,
                (45_f64).to_radians().cos(),
                params.detector_array_angle,
                32.);
        Spectrometer::new(beam, cmpnd, detarr)
    }
}

impl MultiObjectiveMinimizationProblem for DynamicTest {
    type Fitness = DesignFitness<f64>;

    fn epsilons(&self) -> &<Self::Fitness as MultiObjectiveFitness>::Epsilons {
        &epsilons
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        let nprism = self.0.len();
        let height_bounds = (0.001, 20.);
        let normalized_y_mean_bounds = (0., 1.);
        let curvature_bounds = (0.00001, 1.);
        let det_arr_angle_bounds = (-PI, PI);
        let len_bounds = (0., 10.);
        // let angle_bounds = (-FRAC_PI_2, FRAC_PI_2);
        let angle_bounds = (-1., 1.);
        let mut bounds = vec![
            height_bounds,
            normalized_y_mean_bounds,
            curvature_bounds,
            det_arr_angle_bounds,
        ];
        bounds.resize(bounds.len() + nprism, len_bounds);
        bounds.resize(bounds.len() + nprism + 1, angle_bounds);
        bounds
    }

    fn evaluate(&self, params: &[f64]) -> Option<Self::Fitness> {
        let params = self.slice_into_params(params);
        let spec = self.params_into_spectrometer(params).ok()?;
        let spec: Spectrometer<f32> = (&spec).into();
        spec.cuda_fitness()
            .map(|fit| DesignFitness {
                size: fit.size as f64,
                info: fit.info as f64,
                deviation: fit.deviation as f64,
            })
            .filter(|fit| fit.size <= 1800. && fit.info > 0.2)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // const glass_names: &'static [&'static str] = &["N-LAF21", "N-SF11", "N-SSK2", "N-FK5"];
    // const glass_names: &'static [&'static str] = &["N-LAF21", "N-SF11", "N-SSK2"];
    const glass_names: &'static [&'static str] = &["N-SF11", "N-SSK2"];
    let glasses = glass_names.iter().map(|name|
        BUNDLED_CATALOG.iter().find(|(n, _)| n == name).unwrap().1.clone())
        .collect::<Vec<_>>();
    let problem = DynamicTest(glasses);
    let solns = optimize(
        &problem,
        128,
        512,
        256,
        76432959324,
        16.,
        12.,
        0.08,
    );
    for soln in solns {
        let params = problem.slice_into_params(soln.params.as_slice());
        let spec = problem.params_into_spectrometer(params).unwrap();
        if soln.fitness.info > 3.7 {
            println!("{:.4?}\n{:.4?}\n{:.4?}\n{:.4?}\n",
                     spec,
                     params,
                     spec.fitness(),
                     soln.fitness,
            );
        }
    }
    return Ok(());
    // iDus 420  Bx-DD; 1024 x 256 pixels; 26 x 26 μm pixel size; 26.6 x 6.6 mm height

    /*
    let mut rng = rand_xoshiro::Xoshiro256Plus::seed_from_u64(90834257);
    const nglass: usize = BUNDLED_CATALOG.len();
    const nprism: usize = glass_names.len();
    let glasses = glass_names.iter().map(|name|
        BUNDLED_CATALOG.iter().find(|(n, _)| n == name).unwrap().1.clone())
        .collect::<Vec<_>>();
    for _ in 0..1000 {
        let angles = (0..=nprism).map(|_| rng.gen_range(-FRAC_PI_2, FRAC_PI_2)).collect::<Vec<_>>();
        let beam = GaussianBeam {
            width: 3.2,
            y_mean: 8.5,
            w_range: (0.5, 0.82),
        };
        let cmpnd = CompoundPrism::<f64>::new(
            glasses.iter().cloned(),
            &angles,
            &[0.; nprism],
            0.001,
            10.,
            7.,
        );
        let detarr =
            LinearDetectorArray::new(32, 0.8, 1., 0.1, (45_f64).to_radians().cos(), 0., 32.);
        if let Ok(spec) = Spectrometer::new(beam, cmpnd, detarr) {
            let angles = angles.into_iter().map(|a| a.to_degrees()).collect::<Vec<_>>();
            println!("{:.4?}\n{:.4?}\n", angles, spec.fitness())
        } else {
            // println!("Failed")
        }
    }
    return Ok(());*/

    let file = read("design_config.toml")?;
    let config: DesignConfig = toml::from_slice(file.as_ref())?;
    let designs = config.optimize_designs(None);

    let out = (config, designs);
    let file = File::create("results.cbor.gz")?;
    let gz = GzEncoder::new(file, Compression::default());
    serde_cbor::to_writer(gz, &out)?;

    for design in out.1.iter() {
        println!("{:?}", design.fitness);
        if design.fitness.info > 3.5 {
            println!("{:?} {:#?}", design.spectrometer.fitness(), design);
        }
    }
    Ok(())
}

/*
Ok(Spectrometer { gaussian_beam: GaussianBeam { width: 3.2000, y_mean: 9.0282, w_range: (0.5000, 0.8200) }, compound_prism: CompoundPrism { prisms: [(Sellmeier1([1.8713, 0.0093, 0.2508, 0.0346, 1.2205, 83.2405]), Surface { angle: -0.0605, normal: Pair { x: -0.9982, y: 0.0604 }, midpt: Pair { x: 0.4058, y: 6.7028 } }), (Sellmeier1([1.7376, 0.0132, 0.3137, 0.0623, 1.8988, 155.2363]), Surface { angle: 1.1874, normal: Pair { x: -0.3741, y: -0.9274 }, midpt: Pair { x: 27.2031, y: 6.7028 } }), (Sellmeier1([1.4306, 0.0082, 0.1532, 0.0334, 1.0139, 106.8708]), Surface { angle: -0.6878, normal: Pair { x: -0.7727, y: 0.6348 }, midpt: Pair { x: 51.3915, y: 6.7028 } })], lens: CurvedSurface { midpt: Pair { x: 65.1564, y: 6.7232 }, center: Pair { x: -252.9955, y: -85.0676 }, radius: 331.1286, max_dist_sq: 48.6732 }, height: 13.4057, width: 7.0000 }, detector_array: LinearDetectorArray { bin_count: 32, bin_size: 0.8000, linear_slope: 1.0000, linear_intercept: 0.1000, min_ci: 0.7071, angle: -0.5739, normal: Pair { x: -0.8398, y: 0.5430 }, length: 32.0000 }, detector_array_position: DetectorArrayPositioning { position: Pair { x: 503.0474, y: -24.3858 }, direction: Pair { x: -0.5430, y: -0.8398 } } })
DynamicParams { height: 13.4057, y_mean: 9.0282, curvature: 0.0211, detector_array_angle: -32.8848, lengths: [9.7761, 2.0660, 6.2534], angles: [-3.4648, 68.0302, -39.4064, 16.0935] }
DesignFitness { size: 496.5752, info: 3.5200, deviation: 0.0943 }

Ok(Spectrometer { gaussian_beam: GaussianBeam { width: 3.2000, y_mean: 11.8020, w_range: (0.5000, 0.8200) }, compound_prism: CompoundPrism { prisms: [(Sellmeier1([1.8713, 0.0093, 0.2508, 0.0346, 1.2205, 83.2405]), Surface { angle: -0.0027, normal: Pair { x: -1.0000, y: 0.0027 }, midpt: Pair { x: 0.0244, y: 8.9327 } }), (Sellmeier1([1.7376, 0.0132, 0.3137, 0.0623, 1.8988, 155.2363]), Surface { angle: 1.4401, normal: Pair { x: -0.1304, y: -0.9915 }, midpt: Pair { x: 70.7393, y: 8.9327 } }), (Sellmeier1([1.4306, 0.0082, 0.1532, 0.0334, 1.0139, 106.8708]), Surface { angle: -0.8380, normal: Pair { x: -0.6689, y: 0.7433 }, midpt: Pair { x: 158.3558, y: 8.9327 } })], lens: CurvedSurface { midpt: Pair { x: 169.3257, y: 8.8774 }, center: Pair { x: 71.5490, y: 22.0971 }, radius: 98.6663, max_dist_sq: 81.4213 }, height: 17.8653, width: 7.0000 }, detector_array: LinearDetectorArray { bin_count: 32, bin_size: 0.8000, linear_slope: 1.0000, linear_intercept: 0.1000, min_ci: 0.7071, angle: 0.5759, normal: Pair { x: -0.8387, y: -0.5446 }, length: 32.0000 }, detector_array_position: DetectorArrayPositioning { position: Pair { x: 289.8827, y: 51.8729 }, direction: Pair { x: 0.5446, y: -0.8387 } } })
DynamicParams { height: 17.8653, y_mean: 11.8020, curvature: 0.0914, detector_array_angle: 32.9977, lengths: [2.7550, 9.7554, 1.8430], angles: [-0.1565, 82.5093, -48.0142, -7.6999] }
DesignFitness { size: 299.7834, info: 3.4258, deviation: 0.0889 }

Spectrometer { gaussian_beam: GaussianBeam { width: 3.2000, y_mean: 8.0569, w_range: (0.5000, 0.8200) }, compound_prism: CompoundPrism { prisms: [(Sellmeier1([1.8713, 0.0093, 0.2508, 0.0346, 1.2205, 83.2405]), Surface { angle: -0.0556, normal: Pair { x: -0.9985, y: 0.0556 }, midpt: Pair { x: 0.3604, y: 6.4745 } }), (Sellmeier1([1.7376, 0.0132, 0.3137, 0.0623, 1.8988, 155.2363]), Surface { angle: -1.4417, normal: Pair { x: -0.1288, y: 0.9917 }, midpt: Pair { x: 59.8556, y: 6.4745 } }), (Sellmeier1([1.4306, 0.0082, 0.1532, 0.0334, 1.0139, 106.8708]), Surface { angle: 0.2050, normal: Pair { x: -0.9791, y: -0.2035 }, midpt: Pair { x: 116.2391, y: 6.4745 } })], lens: CurvedSurface { midpt: Pair { x: 123.5656, y: 6.4306 }, center: Pair { x: -89.9527, y: 101.8836 }, radius: 233.8832, max_dist_sq: 50.3079 }, height: 12.9489, width: 7.0000 }, detector_array: LinearDetectorArray { bin_count: 32, bin_size: 0.8000, linear_slope: 1.0000, linear_intercept: 0.1000, min_ci: 0.7071, angle: -0.2644, normal: Pair { x: -0.9653, y: 0.2613 }, length: 32.0000 }, detector_array_position: DetectorArrayPositioning { position: Pair { x: 302.1822, y: 37.7160 }, direction: Pair { x: 0.2613, y: 0.9653 } } }
DynamicParams { height: 12.9489, y_mean: 8.0569, curvature: 0.0303, detector_array_angle: -15.1480, lengths: [10.0000, 5.1818, 2.9880], angles: [-3.1858, -82.6007, 11.7444, -24.0869] }
DesignFitness { size: 309.6654, info: 3.3558, deviation: 0.1457 }
DesignFitness { size: 309.6654, info: 3.3640, deviation: 0.1457 }

*/