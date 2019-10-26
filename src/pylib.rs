use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::create_exception;
use pyo3::exceptions::Exception;
use pyo3::ObjectProtocol;
use pyo3::types::{PyAny};
use ndarray::prelude::{array, Array2};
use numpy::{PyArray1, PyArray2, IntoPyArray};
use crate::glasscat::*;
use crate::ray::{
    DesignFitness, CompoundPrism, DetectorArray, DetectorArrayPositioning, GaussianBeam,
    detector_array_positioning, p_dets_l_wavelength, trace
};
use crate::optimizer::{Config, AGE, SBX, PM, Params};

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

#[pyclass(name=Glass)]
#[derive(Clone)]
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

#[pyclass(name=CompoundPrism)]
#[derive(Clone)]
pub struct PyCompoundPrism {
    #[pyo3(get)]
    glasses: Vec<PyGlass>,
    #[pyo3(get)]
    angles: Vec<f64>,
    #[pyo3(get)]
    lengths: Vec<f64>,
    #[pyo3(get)]
    curvature: f64,
    #[pyo3(get)]
    height: f64,
    #[pyo3(get)]
    width: f64,
}

impl<'s> Into<CompoundPrism<'s>> for &'s PyCompoundPrism {
    fn into(self) -> CompoundPrism<'s> {
        CompoundPrism::new(
            self.glasses.iter().map(|pg| &pg.glass),
            &self.angles,
            &self.lengths,
            self.curvature,
            self.height,
            self.width
        )
    }
}

#[pymethods]
impl PyCompoundPrism {
    fn polygons<'p>(&self, py: Python<'p>) -> PyResult<(Vec<&'p PyArray2<f64>>, &'p PyArray2<f64>, &'p PyArray1<f64>, f64)> {
        let cmpnd: CompoundPrism = self.into();
        let (polys, lens_poly, lens_center, lens_radius) = cmpnd.polygons();
        let polys = polys
            .into_iter()
            .map(|[p0, p1, p2, p3]| (array![[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]]).into_pyarray(py))
            .collect();
        let [p0, p1, p2, p3] = lens_poly;
        let lens_poly = (array![[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]]).into_pyarray(py);
        let lens_center = PyArray1::from_slice(py, &[lens_center.x, lens_center.y]);
        Ok((polys, lens_poly, lens_center, lens_radius))
    }
}


#[pyclass(name=DetectorArray)]
#[derive(Clone)]
pub struct PyDetectorArray {
    bins: Vec<[f64; 2]>,
    #[pyo3(get)]
    position: (f64, f64),
    #[pyo3(get)]
    direction: (f64, f64),
    #[pyo3(get)]
    length: f64,
    #[pyo3(get)]
    min_ci: f64,
    #[pyo3(get)]
    angle: f64,
}

impl<'s> Into<DetectorArray<'s>> for &'s PyDetectorArray {
    fn into(self) -> DetectorArray<'s> {
        DetectorArray::new(
            self.bins.as_slice().into(),
            self.min_ci,
            self.angle,
            self.length
        )
    }
}

impl Into<DetectorArrayPositioning> for &PyDetectorArray {
    fn into(self) -> DetectorArrayPositioning {
        DetectorArrayPositioning {
            position: self.position.into(),
            direction: self.direction.into(),
        }
    }
}


#[pyclass(name=GaussianBeam)]
#[derive(Clone)]
pub struct PyGaussianBeam {
    #[pyo3(get)]
    wavelength_range: (f64, f64),
    #[pyo3(get)]
    width: f64,
    #[pyo3(get)]
    y_mean: f64,
}

impl Into<GaussianBeam> for &PyGaussianBeam {
    fn into(self) -> GaussianBeam {
        GaussianBeam {
            width: self.width,
            y_mean: self.y_mean,
            w_range: self.wavelength_range
        }
    }
}

#[pyclass]
struct OptimizerSpecification {
    #[pyo3(get, set)]
    iteration_count: usize,
    #[pyo3(get, set)]
    population_size: usize,
    #[pyo3(get, set)]
    offspring_size: usize,
    #[pyo3(get, set)]
    crossover_distribution_index: f64,
    #[pyo3(get, set)]
    mutation_distribution_index: f64,
    #[pyo3(get, set)]
    mutation_probability: f64,
    #[pyo3(get, set)]
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
    #[pyo3(get, set)]
    max_count: usize,
    #[pyo3(get, set)]
    max_height: f64,
    #[pyo3(get, set)]
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
    #[pyo3(get, set)]
    width: f64,
    #[pyo3(get, set)]
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
    #[pyo3(get, set)]
    length: f64,
    #[pyo3(get, set)]
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
) -> PyResult<Vec<Design>> {
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
            let params = Params::from_slice(&soln.params, config.max_prism_count);
            let (c, _, d, g) = config.array_to_params(&soln.params);
            let detpos = detector_array_positioning(&c, &d, &g).unwrap();
            Design {
                compound_prism: PyCompoundPrism {
                    glasses: params.glass_indices().map(|i| {
                        let g = &config.glass_catalog[i];
                        PyGlass {
                            name: g.0.clone(),
                            glass: g.1.clone(),
                        }
                    }).collect(),
                    angles: params.angles.to_vec(),
                    lengths: params.lengths.to_vec(),
                    curvature: params.curvature,
                    height: params.prism_height,
                    width: config.prism_width
                },
                detector_array: PyDetectorArray {
                    bins: config.detector_array_bin_bounds.to_vec(),
                    position: detpos.position.into(),
                    direction: detpos.direction.into(),
                    length: config.detector_array_length,
                    min_ci: config.detector_array_min_ci,
                    angle: params.detector_array_angle
                },
                gaussian_beam: PyGaussianBeam {
                    wavelength_range: config.wavelength_range,
                    width: config.beam_width,
                    y_mean: params.y_mean
                },
                fitness: soln.fitness.into(),
            }
        }).collect();
    Ok(designs)
}

#[pyclass(name=DesignFitness)]
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
pub struct Design {
    #[pyo3(get)]
    compound_prism: PyCompoundPrism,
    #[pyo3(get)]
    detector_array: PyDetectorArray,
    #[pyo3(get)]
    gaussian_beam: PyGaussianBeam,
    #[pyo3(get)]
    fitness: PyDesignFitness,
}

#[pymethods]
impl Design {
    pub fn transmission_probability<'p>(&self, wavelengths: &PyArray1<f64>, py: Python<'p>) -> PyResult<&'p PyArray2<f64>> {
        let cmpnd = (&self.compound_prism).into();
        let detarr = (&self.detector_array).into();
        let detpos = (&self.detector_array).into();
        let beam = (&self.gaussian_beam).into();
        py.allow_threads(|| wavelengths
            .as_array()
            .into_iter()
            .flat_map(|w| p_dets_l_wavelength(*w, &cmpnd, &detarr, &beam, &detpos))
            .collect::<Box<_>>())
            .into_pyarray(py)
            .reshape((wavelengths.len(), self.detector_array.bins.len()))
            .map_err(|e| e.into())
    }

    pub fn ray_trace<'p>(&self, wavelength: f64, inital_y: f64, py: Python<'p>) -> PyResult<&'p PyArray2<f64>> {
        let cmpnd = (&self.compound_prism).into();
        let detarr = (&self.detector_array).into();
        let detpos = (&self.detector_array).into();
        Ok(Array2::from(trace(wavelength, inital_y, &cmpnd, &detarr, &detpos)
            .map(|r| r.map(|p| [p.x, p.y]))
            .collect::<Result<Vec<_>, _>>()?)
            .into_pyarray(py))
    }
}

#[pymodule]
fn compound_prism_designer(_: Python, m: &PyModule) -> PyResult<()> {
    // TODO: Better way to expose exception types
    use pyo3::type_object::PyTypeObject;
    m.add("GlassCatalogError", GlassCatalogError::type_object())?;
    m.add("RayTraceError", RayTraceError::type_object())?;
    m.add_class::<PyGlass>()?;
    m.add_class::<PyCompoundPrism>()?;
    m.add_class::<PyDetectorArray>()?;
    m.add_class::<PyGaussianBeam>()?;
    m.add_class::<OptimizerSpecification>()?;
    m.add_class::<CompoundPrismSpecification>()?;
    m.add_class::<GaussianBeamSpecification>()?;
    m.add_class::<DetectorArraySpecification>()?;
    m.add_class::<PyDesignFitness>()?;
    m.add_class::<Design>()?;
    m.add_wrapped(wrap_pyfunction!(create_glass_catalog))?;
    m.add_wrapped(wrap_pyfunction!(optimize))?;
    Ok(())
}
