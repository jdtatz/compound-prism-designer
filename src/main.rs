#[macro_use]
extern crate derive_more;

mod erf;
mod glasscat;
mod qrng;
mod ray;
mod optimizer;

use crate::ray::*;
use crate::optimizer::*;

#[derive(serde::Serialize)]
struct TransmissionData {
    wavelengths: Vec<f64>,
    data: Vec<Vec<f64>>,
}

#[derive(serde::Serialize)]
struct DesignOutput {
    design: Design,
    transmission_data: TransmissionData,
}


fn main() {
    let config = DesignConfig {
        optimizer: OptimizationConfig {
            iteration_count: 1024,
            population_size: 100000,
            offspring_size: 256,
            crossover_distribution_index: 20.0,
            mutation_distribution_index: 12.0,
            mutation_probability: 0.05,
            seed: 745093248298,
            epsilons: [2.5, 0.02, 0.05]
        },
        compound_prism: CompoundPrismConfig {
            max_count: 6,
            max_height: 20.0,
            width: 7.0
        },
        detector_array: DetectorArrayConfig {
            length: 32.0,
            max_incident_angle: 45.0,
            bin_bounds: vec![[ 0.1,  0.9],
                              [ 1.1,  1.9],
                              [ 2.1,  2.9],
                              [ 3.1,  3.9],
                              [ 4.1,  4.9],
                              [ 5.1,  5.9],
                              [ 6.1,  6.9],
                              [ 7.1,  7.9],
                              [ 8.1,  8.9],
                              [ 9.1,  9.9],
                              [10.1, 10.9],
                              [11.1, 11.9],
                              [12.1, 12.9],
                              [13.1, 13.9],
                              [14.1, 14.9],
                              [15.1, 15.9],
                              [16.1, 16.9],
                              [17.1, 17.9],
                              [18.1, 18.9],
                              [19.1, 19.9],
                              [20.1, 20.9],
                              [21.1, 21.9],
                              [22.1, 22.9],
                              [23.1, 23.9],
                              [24.1, 24.9],
                              [25.1, 25.9],
                              [26.1, 26.9],
                              [27.1, 27.9],
                              [28.1, 28.9],
                              [29.1, 29.9],
                              [30.1, 30.9],
                              [31.1, 31.9]].into_boxed_slice()
        },
        gaussian_beam: GaussianBeamConfig {
            width: 3.2,
            wavelength_range: (0.5, 0.82)
        }
    };
    let designs = config.optimize(None);

    const N: usize = 100;
    let (l, u) = config.gaussian_beam.wavelength_range;
    let wavelengths: Vec<_> = (0..N).map(|i| l + (u - l) * (i as f64) / ((N - 1) as f64)).collect();

    let out: Vec<_> = designs.iter().map(|design| {
        let cmpnd: CompoundPrism = (&design.compound_prism).into();
        let detarr: DetectorArray = (&design.detector_array).into();
        let detpos: DetectorArrayPositioning = (&design.detector_array).into();
        let beam: GaussianBeam = (&design.gaussian_beam).into();
        let data = wavelengths.iter().map(|w| p_dets_l_wavelength(*w, &cmpnd, &detarr, &beam, &detpos).collect()).collect();
        DesignOutput {
            design: design.clone(),
            transmission_data: TransmissionData {
                wavelengths: wavelengths.clone(),
                data
            }
        }
    }).collect();

    let file = std::fs::File::create("results.cbor").unwrap();
    serde_cbor::to_writer(file, &out).unwrap();

    for design in designs.into_iter() {
        println!("{:?}", design.fitness);
        if design.fitness.info > 3.5 {
            println!("{:#?}", design);
        }
    }
}

/*DesignFitness { size: 163.54813057409118, info: 3.7427129701551016, deviation: 0.09547917696843032 }
Design {
    compound_prism: CompoundPrismDesign {
        glasses: [
            (
                "N-PK51",
                Sellmeier1(
                    [
                        1.15610775,
                        0.00585597402,
                        0.153229344,
                        0.0194072416,
                        0.785618966,
                        140.537046,
                    ],
                ),
            ),
            (
                "P-LAF37",
                Sellmeier1(
                    [
                        1.76003244,
                        0.00938006396,
                        0.248286745,
                        0.0360537464,
                        1.15935122,
                        86.4324693,
                    ],
                ),
            ),
            (
                "N-SF4",
                Sellmeier1(
                    [
                        1.67780282,
                        0.012679345,
                        0.282849893,
                        0.0602038419,
                        1.63539276,
                        145.760496,
                    ],
                ),
            ),
            (
                "P-LAK35",
                Sellmeier1(
                    [
                        1.3932426,
                        0.00715959695,
                        0.418882766,
                        0.0233637446,
                        1.043807,
                        88.3284426,
                    ],
                ),
            ),
        ],
        angles: [
            0.2771380326030153,
            -0.2545034335069508,
            -1.4134448014990648,
            1.077436999472814,
            0.19350125167211424,
        ],
        lengths: [
            0.006560118872936632,
            0.039125830376652945,
            0.10864671284028488,
            1.0268621834663,
        ],
        curvature: 0.024514461030329032,
        height: 4.790103712186817,
        width: 7.0,
    },
    detector_array: DetectorArrayDesign {
        bins: [
            [
                0.1,
                0.9,
            ],
            [
                1.1,
                1.9,
            ],
            [
                2.1,
                2.9,
            ],
            [
                3.1,
                3.9,
            ],
            [
                4.1,
                4.9,
            ],
            [
                5.1,
                5.9,
            ],
            [
                6.1,
                6.9,
            ],
            [
                7.1,
                7.9,
            ],
            [
                8.1,
                8.9,
            ],
            [
                9.1,
                9.9,
            ],
            [
                10.1,
                10.9,
            ],
            [
                11.1,
                11.9,
            ],
            [
                12.1,
                12.9,
            ],
            [
                13.1,
                13.9,
            ],
            [
                14.1,
                14.9,
            ],
            [
                15.1,
                15.9,
            ],
            [
                16.1,
                16.9,
            ],
            [
                17.1,
                17.9,
            ],
            [
                18.1,
                18.9,
            ],
            [
                19.1,
                19.9,
            ],
            [
                20.1,
                20.9,
            ],
            [
                21.1,
                21.9,
            ],
            [
                22.1,
                22.9,
            ],
            [
                23.1,
                23.9,
            ],
            [
                24.1,
                24.9,
            ],
            [
                25.1,
                25.9,
            ],
            [
                26.1,
                26.9,
            ],
            [
                27.1,
                27.9,
            ],
            [
                28.1,
                28.9,
            ],
            [
                29.1,
                29.9,
            ],
            [
                30.1,
                30.9,
            ],
            [
                31.1,
                31.9,
            ],
        ],
        position: Pair {
            x: 150.91164270045272,
            y: -24.729980747776754,
        },
        direction: Pair {
            x: 0.7430816484086802,
            y: 0.6692007649414626,
        },
        length: 32.0,
        max_incident_angle: 45.0,
        angle: -0.837663629259216,
    },
    gaussian_beam: GaussianBeamDesign {
        wavelength_range: (
            0.5,
            0.82,
        ),
        width: 3.2,
        y_mean: 1.5926723932262503,
    },
    fitness: DesignFitness {
        size: 163.54813057409118,
        info: 3.7427129701551016,
        deviation: 0.09547917696843032,
    },
}
*/