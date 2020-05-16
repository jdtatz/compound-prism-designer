use crate::fitness::*;
use crate::glasscat::Glass;
use crate::glasscat::BUNDLED_CATALOG;
use crate::ray::*;
use crate::utils::*;
use crate::geom::*;
#[cfg(feature = "pyext")]
use pyo3::prelude::{pyclass, PyObject};
use std::borrow::Cow;

use crate::optimizer::{
    ParetoDominance, MultiObjectiveFitness, MultiObjectiveMinimizationProblem, optimize
};

#[cfg_attr(feature = "pyext", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct OptimizationConfig {
    pub iteration_count: usize,
    pub population_size: usize,
    pub offspring_size: usize,
    pub crossover_distribution_index: f64,
    pub mutation_distribution_index: f64,
    pub mutation_probability: f64,
    pub seed: u64,
    pub epsilons: [f64; 3],
}

#[cfg_attr(feature = "pyext", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct CompoundPrismConfig {
    pub max_count: usize,
    pub max_height: f64,
    pub width: f64,
    #[serde(default)]
    pub ar_coating: bool,
}

#[cfg_attr(feature = "pyext", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct GaussianBeamConfig {
    pub width: f64,
    pub wavelength_range: (f64, f64),
}

#[cfg_attr(feature = "pyext", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct LinearDetectorArrayArrayConfig {
    pub bin_count: u32,
    pub bin_size: f64,
    pub linear_slope: f64,
    pub linear_intercept: f64,
    pub length: f64,
    pub max_incident_angle: f64,
}

/// Specification structure for the configuration of the Spectrometer Designer
#[cfg_attr(feature = "pyext", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct DesignConfig {
    pub length_unit: Cow<'static, str>,
    pub optimizer: OptimizationConfig,
    pub compound_prism: CompoundPrismConfig,
    pub detector_array: LinearDetectorArrayArrayConfig,
    pub gaussian_beam: GaussianBeamConfig,
}

impl DesignConfig {
    fn parameter_count(&self) -> usize {
        3 * self.compound_prism.max_count + 6
    }

    pub fn array_to_params(&self, params: &[f64]) -> Result<Spectrometer<f64>, RayTraceError> {
        let params = Params::from_slice(params, self.compound_prism.max_count);
        let cmpnd = CompoundPrism::new(
            params
                .glass_indices()
                .map(|i| &BUNDLED_CATALOG[i].1)
                .cloned(),
            &params.angles,
            &params.lengths,
            params.curvature,
            params.prism_height,
            self.compound_prism.width,
            self.compound_prism.ar_coating,
        );
        let detarr = LinearDetectorArray::new(
            self.detector_array.bin_count,
            self.detector_array.bin_size,
            self.detector_array.linear_slope,
            self.detector_array.linear_intercept,
            self.detector_array.max_incident_angle.to_radians().cos(),
            params.detector_array_angle,
            self.detector_array.length,
        );
        let beam = GaussianBeam {
            width: self.gaussian_beam.width,
            y_mean: params.y_mean,
            w_range: self.gaussian_beam.wavelength_range,
        };
        Spectrometer::new(beam, cmpnd, detarr)
    }

    pub fn array_to_design(
        &self,
        params: &[f64],
        fitness: Option<DesignFitness<f64>>,
    ) -> Result<Design, RayTraceError> {
        let spec = self.array_to_params(params)?;
        let params = Params::from_slice(params, self.compound_prism.max_count);
        let compound_prism = CompoundPrismDesign {
            glasses: params
                .glass_indices()
                .map(|i| {
                    let (n, g) = &BUNDLED_CATALOG[i];
                    (Cow::Borrowed(*n), g.clone())
                })
                .collect(),
            angles: params.angles.to_owned(),
            lengths: params.lengths.to_owned(),
            curvature: params.curvature,
            height: params.prism_height,
            width: self.compound_prism.width,
            ar_coated: self.compound_prism.ar_coating,
        };
        let detector_array = DetectorArrayDesign {
            bin_count: self.detector_array.bin_count,
            bin_size: self.detector_array.bin_size,
            linear_slope: self.detector_array.linear_slope,
            linear_intercept: self.detector_array.linear_intercept,
            position: spec.detector_array_position.position,
            direction: spec.detector_array_position.direction,
            length: self.detector_array.length,
            max_incident_angle: self.detector_array.max_incident_angle,
            angle: params.detector_array_angle,
        };
        let gaussian_beam = GaussianBeamDesign {
            wavelength_range: self.gaussian_beam.wavelength_range,
            width: self.gaussian_beam.width,
            y_mean: params.y_mean,
        };
        Ok(Design {
            compound_prism,
            detector_array,
            gaussian_beam,
            fitness: fitness.unwrap_or_else(|| spec.fitness()),
            spectrometer: spec,
        })
    }

    pub fn maximum_detector_information(&self) -> f64 {
        (self.detector_array.bin_count as f64).log2()
    }

    pub fn optimize_designs(&self, _glass_catalog: Option<&[(String, Glass<f64>)]>) -> Vec<Design> {
        if let Some(ref _c) = _glass_catalog {
            unimplemented!("Custom glass catalogs have not been implemented yet")
        }
        let mutation_probability = 1_f64 / (self.parameter_count() as f64);
        let archive = optimize(
            self,
            self.optimizer.iteration_count,
            self.optimizer.population_size,
            self.optimizer.offspring_size,
            self.optimizer.seed,
            self.optimizer.crossover_distribution_index,
            self.optimizer.mutation_distribution_index,
            mutation_probability,
        );
        archive
            .into_iter()
            .map(|s| {
                self.array_to_design(s.params.as_slice(), Some(s.fitness))
                    .expect("Only valid designs should result from optimization")
            })
            .collect()
    }
}

#[cfg_attr(feature="pyext", pyclass(name=CompoundPrism))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundPrismDesign {
    pub glasses: Vec<(Cow<'static, str>, Glass<f64>)>,
    pub angles: Vec<f64>,
    pub lengths: Vec<f64>,
    pub curvature: f64,
    pub height: f64,
    pub width: f64,
    #[serde(default)]
    pub ar_coated: bool,
}

impl Into<CompoundPrism<f64>> for &CompoundPrismDesign {
    fn into(self) -> CompoundPrism<f64> {
        CompoundPrism::new(
            self.glasses.iter().map(|(_, g)| g).cloned(),
            &self.angles,
            &self.lengths,
            self.curvature,
            self.height,
            self.width,
            self.ar_coated,
        )
    }
}

#[cfg_attr(feature="pyext", pyclass(name=DetectorArray))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorArrayDesign {
    pub bin_count: u32,
    pub bin_size: f64,
    pub linear_slope: f64,
    pub linear_intercept: f64,
    pub position: Pair<f64>,
    pub direction: Pair<f64>,
    pub length: f64,
    pub max_incident_angle: f64,
    pub angle: f64,
}

impl Into<LinearDetectorArray<f64>> for &DetectorArrayDesign {
    fn into(self) -> LinearDetectorArray<f64> {
        LinearDetectorArray::new(
            self.bin_count,
            self.bin_size,
            self.linear_slope,
            self.linear_intercept,
            self.max_incident_angle.to_radians().cos(),
            self.angle,
            self.length,
        )
    }
}

impl Into<DetectorArrayPositioning<f64>> for &DetectorArrayDesign {
    fn into(self) -> DetectorArrayPositioning<f64> {
        DetectorArrayPositioning {
            position: self.position,
            direction: self.direction,
        }
    }
}

#[cfg_attr(feature="pyext", pyclass(name=GaussianBeam))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianBeamDesign {
    pub wavelength_range: (f64, f64),
    pub width: f64,
    pub y_mean: f64,
}

impl Into<GaussianBeam<f64>> for &GaussianBeamDesign {
    fn into(self) -> GaussianBeam<f64> {
        GaussianBeam {
            width: self.width,
            y_mean: self.y_mean,
            w_range: self.wavelength_range,
        }
    }
}

#[cfg_attr(feature = "pyext", pyclass)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Design {
    pub compound_prism: CompoundPrismDesign,
    pub detector_array: DetectorArrayDesign,
    pub gaussian_beam: GaussianBeamDesign,
    pub fitness: DesignFitness<f64>,
    pub spectrometer: Spectrometer<f64>,
}

#[derive(Debug, Clone)]
struct Params<'s> {
    prism_count: usize,
    prism_height: f64,
    glass_findices: &'s [f64],
    angles: &'s [f64],
    lengths: &'s [f64],
    curvature: f64,
    y_mean: f64,
    detector_array_angle: f64,
}

impl<'s> Params<'s> {
    fn from_slice(s: &'s [f64], max_prism_count: usize) -> Self {
        assert_eq!(s.len(), 3 * max_prism_count + 6);
        let (&prism_count, s) = s.split_first().unwrap();
        let (&prism_height, s) = s.split_first().unwrap();
        let (glass_findices, s) = s.split_at(max_prism_count);
        let (angles, s) = s.split_at(max_prism_count + 1);
        let (lengths, s) = s.split_at(max_prism_count);
        let (&curvature, s) = s.split_first().unwrap();
        let (&normalized_y_mean, s) = s.split_first().unwrap();
        let (&detector_array_angle, _) = s.split_first().unwrap();
        let prism_count = (prism_count as usize).max(1).min(max_prism_count);
        Self {
            prism_count,
            prism_height,
            glass_findices: &glass_findices[..prism_count],
            angles: &angles[..prism_count + 1],
            lengths: &lengths[..prism_count],
            curvature,
            y_mean: prism_height * normalized_y_mean,
            detector_array_angle,
        }
    }

    fn glass_indices(&self) -> impl 's + ExactSizeIterator<Item = usize> {
        self.glass_findices.iter().map(|f| f.floor() as usize)
    }
}

impl<F: Float> MultiObjectiveFitness for DesignFitness<F> {
    type Epsilons = [F; 3];

    fn pareto_dominace(&self, other: &Self) -> Option<ParetoDominance> {
        if self.size <= other.size && self.info >= other.info && self.deviation <= other.deviation {
            Some(ParetoDominance::Dominates)
        } else if self.size >= other.size && self.info <= other.info && self.deviation >= other.deviation {
            Some(ParetoDominance::Dominated)
        } else {
            None
        }
    }

    fn epsilon_dominance(&self, other: &Self, epsilons: &Self::Epsilons) -> Option<ParetoDominance> {
        let box_id = |f: &DesignFitness<F>| {
            [
                (f.size / epsilons[0]).floor(),
                (-f.info / epsilons[1]).floor(),
                (f.deviation / epsilons[2]).floor(),
            ]
        };
        let box_dist = |f: &DesignFitness<F>, bid: &[F; 3]| {
            (f.size - bid[0] * epsilons[0]).sqr()
                + (f.info - bid[1] * epsilons[1]).sqr()
                + (f.deviation - bid[2] * epsilons[2]).sqr()
        };
        let l_box = box_id(self);
        let r_box = box_id(other);
        if l_box == r_box {
            let l_dist = box_dist(&self, &l_box);
            let r_dist = box_dist(&other, &r_box);
            if l_dist <= r_dist {
                Some(ParetoDominance::Dominates)
            } else {
                Some(ParetoDominance::Dominated)
            }
        } else if l_box[0] <= r_box[0] && l_box[1] <= r_box[1] && l_box[2] <= r_box[2] {
            Some(ParetoDominance::Dominates)
        } else if l_box[0] >= r_box[0] && l_box[1] >= r_box[1] && l_box[2] >= r_box[2] {
            Some(ParetoDominance::Dominated)
        } else {
            None
        }
    }
}

impl MultiObjectiveMinimizationProblem for DesignConfig {
    type Fitness = DesignFitness<f64>;

    fn epsilons(&self) -> &<Self::Fitness as MultiObjectiveFitness>::Epsilons {
        &self.optimizer.epsilons
    }

    fn parameter_bounds(&self) -> Vec<(f64, f64)> {
        let prism_count_bounds = (1., (1 + self.compound_prism.max_count) as f64);
        let prism_height_bounds = (0.001, self.compound_prism.max_height);
        let glass_bounds = (0., (BUNDLED_CATALOG.len() - 1) as f64);
        let angle_bounds = (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
        let length_bounds = (0., self.compound_prism.max_height);
        let curvature_bounds = (0.001, 1.0);
        let normalized_y_mean_bounds = (0., 1.0);
        let det_arr_angle_bounds = (-std::f64::consts::PI, std::f64::consts::PI);

        let mut bounds = Vec::with_capacity(self.parameter_count());
        bounds.push(prism_count_bounds);
        bounds.push(prism_height_bounds);
        for _ in 0..self.compound_prism.max_count {
            bounds.push(glass_bounds)
        }
        for _ in 0..self.compound_prism.max_count + 1 {
            bounds.push(angle_bounds)
        }
        for _ in 0..self.compound_prism.max_count {
            bounds.push(length_bounds)
        }
        bounds.push(curvature_bounds);
        bounds.push(normalized_y_mean_bounds);
        bounds.push(det_arr_angle_bounds);
        bounds
    }

    fn evaluate(&self, params: &[f64]) -> Option<Self::Fitness> {
        let spec = self.array_to_params(params).ok()?;
        #[cfg(feature = "cuda")]
            {
                let spec: Spectrometer<f32> = LossyInto::into(spec);
                spec.cuda_fitness()
                    .map(|fit| DesignFitness {
                        size: fit.size as f64,
                        info: fit.info as f64,
                        deviation: fit.deviation as f64,
                    })
                    .filter(|fit| fit.size <= 300. && fit.info > 0.2)
            }
        #[cfg(not(feature = "cuda"))]
            {
                let fit = spec.fitness();
                if fit.size <= 30. * self.compound_prism.max_height && fit.info > 0.2 {
                    Some(fit)
                } else {
                    None
                }
            }
    }
}
