use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::create_exception;
use pyo3::exceptions::Exception;
use pyo3::ObjectProtocol;
use pyo3::types::{PyAny};
use numpy::{PyArray1, PyArray2, IntoPyArray};
use crate::glasscat::*;
use crate::ray::{DesignFitness, detector_array_positioning, p_dets_l_wavelength};
use crate::optimizer::{Config, AGE, SBX, PM};
use std::sync::Arc;

create_exception!(compound_prism_designer, GlassCatalogError, Exception);
create_exception!(compound_prism_designer, RayTraceError, Exception);

impl From<CatalogError> for PyErr {
    fn from(err: CatalogError) -> PyErr {
        GlassCatalogError::py_err(<CatalogError as Into<&'static str>>::into(err))
    }
}

impl From<crate::ray::RayTraceError> for PyErr {
    fn from(err: crate::ray::RayTraceError) -> PyErr {
        RayTraceError::py_err(<crate::ray::RayTraceError as Into<&'static str>>::into(err))
    }
}

#[pyclass]
pub struct PyGlass {
    #[pyo3(get)]
    name: String,
    glass: Glass,
}

#[pymethods]
impl PyGlass {
    #[call]
    fn __call__(&self, w: f64) -> f64 {
        self.glass.calc_n(w)
    }
}

#[pyfunction]
fn create_glass_catalog(catalog_file_contents: &str) -> PyResult<Vec<PyGlass>> {
    new_catalog(catalog_file_contents)
        .map(|r| r.map(|(name, glass)| PyGlass {
            name: name.into(),
            glass
        }).map_err(|e| e.into()))
        .collect()
}

#[pyclass]
struct OptimizerSpecification {
    iteration_count: usize,
    population_size: usize,
    offspring_size: usize,
    crossover_distribution_index: f64,
    mutation_distribution_index: f64,
    mutation_probability: f64,
    seed: u64,
    epsilons: [f64; 3],
}

#[pymethods]
impl OptimizerSpecification {
    #[new]
    fn new(
        obj: &PyRawObject,
        iteration_count: usize,
        population_size: usize,
        offspring_size: usize,
        crossover_distribution_index: f64,
        mutation_distribution_index: f64,
        mutation_probability: f64,
        seed: u64,
        epsilons: (f64, f64, f64)
    ) {
        obj.init(OptimizerSpecification {
            iteration_count,
            population_size,
            offspring_size,
            crossover_distribution_index,
            mutation_distribution_index,
            mutation_probability,
            seed,
            epsilons: [epsilons.0, epsilons.1, epsilons.2]
        });
    }
}

#[pyclass]
struct CompoundPrismSpecification {
    max_count: usize,
    max_height: f64,
    width: f64,
}

#[pymethods]
impl CompoundPrismSpecification {
    #[new]
    fn new(
        obj: &PyRawObject,
        max_count: usize,
        max_height: f64,
        width: f64
    ) {
        obj.init(CompoundPrismSpecification {
            max_count,
            max_height,
            width
        });
    }
}

#[pyclass]
struct GaussianBeamSpecification {
    width: f64,
    wavelength_range: (f64, f64)
}

#[pymethods]
impl GaussianBeamSpecification {
    #[new]
    fn new(
        obj: &PyRawObject,
        width: f64,
        wavelength_range: (f64, f64)
    ) {
        obj.init(GaussianBeamSpecification {
            width,
            wavelength_range
        });
    }
}

#[pyclass]
struct DetectorArraySpecification {
    length: f64,
    max_incident_angle: f64,
    bounds: Box<[[f64; 2]]>,
}

#[pymethods]
impl DetectorArraySpecification {
    #[new]
    fn new(
        obj: &PyRawObject,
        length: f64,
        max_incident_angle: f64,
        bounds: &PyAny
    ) -> PyResult<()> {
        obj.init(DetectorArraySpecification {
            length,
            max_incident_angle,
            bounds: bounds.iter()?.map(|p| {
                let b = p?;
                let lb: f64 = b.get_item(0)?.extract()?;
                let ub: f64 = b.get_item(1)?.extract()?;
                Ok([lb, ub])
            }).collect::<PyResult<_>>()?,
        });
        Ok(())
    }
}

#[pyfunction]
fn optimize(
    py: Python,
    catalog: Option<&PyAny>,
    opt: &OptimizerSpecification,
    compound_prism: &CompoundPrismSpecification,
    gaussian_beam: &GaussianBeamSpecification,
    detector_array: &DetectorArraySpecification
) -> PyResult<Vec<PyDesign>> {
    let glass_catalog = if let Some(catalog) = catalog {
        catalog.iter()?.map(|p| {
            let pg = p?.downcast_ref::<PyGlass>()?;
            Ok((pg.name.clone(), pg.glass.clone()))
        }).collect::<PyResult<_>>()?
    } else {
        BUNDLED_CATALOG.iter().map(|(s, g)| (s.to_string(), g.clone())).collect()
    };
    let config = Config {
        max_prism_count: compound_prism.max_count,
        wavelength_range: gaussian_beam.wavelength_range,
        beam_width: gaussian_beam.width,
        max_prism_height: compound_prism.max_height,
        prism_width: compound_prism.width,
        detector_array_length: detector_array.length,
        detector_array_min_ci: detector_array.max_incident_angle.to_radians().cos(),
        detector_array_bin_bounds: detector_array.bounds.clone().to_vec().into(),
        glass_catalog,
    };
    let config = Arc::new(config);
    let mut optimizer = AGE::new(
        &config,
        opt.population_size,
        opt.offspring_size,
        opt.seed,
        opt.epsilons,
        SBX {
            distribution_index: opt.crossover_distribution_index
        },
        PM {
            distribution_index: opt.mutation_distribution_index,
            probability: opt.mutation_probability
        }
    );
    let iter_count = opt.iteration_count;
    py.allow_threads(|| {
        for _ in 0..iter_count {
            optimizer.iterate()
        }

    });
    let designs = optimizer.archive.into_iter()
        .map(|soln| {
            PyDesign {
                config: Arc::clone(&config),
                params: soln.params.to_vec(),
                fitness: soln.fitness.into()
            }
        }).collect();
    Ok(designs)
}

#[pyclass]
#[derive(Clone)]
pub struct PyDesignFitness {
    #[pyo3(get)]
    size: f64,
    #[pyo3(get)]
    info: f64,
    #[pyo3(get)]
    deviation: f64,
}

impl From<DesignFitness> for PyDesignFitness {
    fn from(fitness: DesignFitness) -> Self {
        PyDesignFitness {
            size: fitness.size,
            info: fitness.info,
            deviation: fitness.deviation
        }
    }
}

#[pyclass]
pub struct PyDesign {
    config: Arc<Config>,
    #[pyo3(get)]
    params: Vec<f64>,
    #[pyo3(get)]
    fitness: PyDesignFitness,
}

#[pymethods]
impl PyDesign {
    pub fn transmission_probability<'p>(&self, wavelengths: &PyArray1<f64>, py: Python<'p>) -> PyResult<&'p PyArray2<f64>> {
        let (cmpnd, _, detarr, beam) = self.config.array_to_params(&self.params);
        let detpos = detector_array_positioning(&cmpnd, &detarr, &beam)?;
        wavelengths
            .as_array()
            .into_iter()
            .flat_map(|w| p_dets_l_wavelength(*w, &cmpnd, &detarr, &beam, detpos))
            .collect::<Box<_>>()
            .into_pyarray(py)
            .reshape((wavelengths.len(), self.config.detector_array_bin_bounds.len()))
            .map_err(|e| e.into())
    }

    pub fn create_svg(&self) -> PyResult<String> {
        let (cmpnd, _, detarr, beam) = self.config.array_to_params(&self.params);
        let detpos = detector_array_positioning(&cmpnd, &detarr, &beam)?;
        Ok(crate::utils::create_svg(&cmpnd, &detarr, &detpos, &beam).to_string())
    }

    pub fn create_zemax_file(&self) -> PyResult<String> {
        let (cmpnd, glass_names, detarr, beam) = self.config.array_to_params(&self.params);
        let detpos = detector_array_positioning(&cmpnd, &detarr, &beam)?;
        let mut out = Vec::new();
        crate::utils::create_zmx(&cmpnd, &detarr, &detpos, &beam, glass_names, &mut out)?;
        String::from_utf8(out).map_err(|e| e.into())
    }
}

#[pymodule]
fn compound_prism_designer(_: Python, m: &PyModule) -> PyResult<()> {
    // TODO: Better way to expose exception types
    use pyo3::type_object::PyTypeObject;
    m.add("GlassCatalogError", GlassCatalogError::type_object())?;
    m.add("RayTraceError", RayTraceError::type_object())?;
    m.add_class::<PyGlass>()?;
    m.add_class::<OptimizerSpecification>()?;
    m.add_class::<CompoundPrismSpecification>()?;
    m.add_class::<GaussianBeamSpecification>()?;
    m.add_class::<DetectorArraySpecification>()?;
    m.add_class::<PyDesignFitness>()?;
    m.add_class::<PyDesign>()?;
    m.add_wrapped(wrap_pyfunction!(create_glass_catalog))?;
    m.add_wrapped(wrap_pyfunction!(optimize))?;
    Ok(())
}
