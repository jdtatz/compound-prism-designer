use crate::utils::Pair;
use crate::glasscat::*;
use crate::optimizer::*;
use crate::ray::{trace, CompoundPrism};
use crate::fitness::*;
use ndarray::prelude::{array, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::create_exception;
use pyo3::exceptions::Exception;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::wrap_pyfunction;
use pyo3::ObjectProtocol;
use pyo3::types::PyBytes;

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

impl<'src> FromPyObject<'src> for Pair {
    fn extract(obj: &'src PyAny) -> Result<Self, PyErr> {
        let (x, y) = obj.extract()?;
        Ok(Pair { x, y })
    }
}

impl IntoPy<PyObject> for Pair {
    fn into_py(self, py: Python) -> PyObject {
        (self.x, self.y).to_object(py)
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
        .map(|r| {
            r.map(|(name, glass)| PyGlass {
                name: name.into(),
                glass,
            })
            .map_err(|e| e.into())
        })
        .collect()
}

#[pymethods]
impl DesignFitness {
    #[getter]
    fn get_size(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.size)
    }

    #[getter]
    fn get_info(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.info)
    }

    #[getter]
    fn get_deviation(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.deviation)
    }
}

#[pymethods]
impl CompoundPrismDesign {
    #[new]
    fn create(obj: &PyRawObject, glasses: Vec<&PyGlass>, angles: &PyArray1<f64>, lengths: &PyArray1<f64>, curvature: f64, height: f64, width: f64) {
        obj.init({
            CompoundPrismDesign {
                glasses: glasses.into_iter().cloned().map(|pg| (pg.name.into(), pg.glass)).collect(),
                angles: angles.as_array().to_vec(),
                lengths: lengths.as_array().to_vec(),
                curvature,
                height,
                width,
            }
        })
    }

    #[getter]
    fn get_glasses(&self) -> PyResult<Vec<PyGlass>> {
        Ok(self.glasses.iter().map(|(s, g)| PyGlass {
            name: s.to_owned().to_string(),
            glass: g.clone(),
        }).collect())
    }

    #[getter]
    fn get_angles<'py>(&'py self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        Ok(PyArray1::from_slice(py, &self.angles))
    }

    #[getter]
    fn get_lengths<'py>(&'py self, py: Python<'py>) -> PyResult<&'py PyArray1<f64>> {
        Ok(PyArray1::from_slice(py, &self.lengths))
    }

    #[getter]
    fn get_curvature(&self) -> PyResult<f64> {
        Ok(self.curvature)
    }

    #[getter]
    fn get_height(&self) -> PyResult<f64> {
        Ok(self.height)
    }

    #[getter]
    fn get_width(&self) -> PyResult<f64> {
        Ok(self.width)
    }

    fn polygons<'p>(
        &self,
        py: Python<'p>,
    ) -> PyResult<(
        Vec<&'p PyArray2<f64>>,
        &'p PyArray2<f64>,
        &'p PyArray1<f64>,
        f64,
    )> {
        let cmpnd: CompoundPrism = self.into();
        let (polys, lens_poly, lens_center, lens_radius) = cmpnd.polygons();
        let polys = polys
            .into_iter()
            .map(|[p0, p1, p2, p3]| {
                (array![[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]]).into_pyarray(py)
            })
            .collect();
        let [p0, p1, p2, p3] = lens_poly;
        let lens_poly =
            (array![[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]]).into_pyarray(py);
        let lens_center = PyArray1::from_slice(py, &[lens_center.x, lens_center.y]);
        Ok((polys, lens_poly, lens_center, lens_radius))
    }
}

#[pymethods]
impl DetectorArrayDesign {
    #[new]
    fn create(obj: &PyRawObject, bins: &PyArray2<f64>, position: (f64, f64), direction: (f64, f64), length: f64, max_incident_angle: f64, angle: f64) -> PyResult<()> {
        let arr = bins.as_array();
        if arr.dim().1 != 2 {
            return Err(pyo3::exceptions::TypeError::py_err("bounds must have a shape of [_, 2]"));
        }
        obj.init({
            DetectorArrayDesign {
                bins: arr.genrows()
                    .into_iter()
                    .map(|r| [r[0], r[1]])
                    .collect(),
                position: position.into(),
                direction: direction.into(),
                length,
                max_incident_angle,
                angle
            }
        });
        Ok(())
    }

    #[getter]
    fn get_bins<'py>(&'py self, py: Python<'py>) -> PyResult<&'py PyArray2<f64>> {
        Ok(Array2::from(self.bins.clone()).into_pyarray(py))
    }

    #[getter]
    fn get_position(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.position)
    }

    #[getter]
    fn get_direction(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.direction)
    }

    #[getter]
    fn get_length(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.length)
    }

    #[getter]
    fn get_max_incident_angle(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.max_incident_angle)
    }

    #[getter]
    fn get_angle(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.angle)
    }
}

#[pymethods]
impl GaussianBeamDesign {
    #[new]
    fn create(obj: &PyRawObject, wavelength_range: (f64, f64), width: f64, y_mean: f64) {
        obj.init({
            GaussianBeamDesign {
                wavelength_range,
                width,
                y_mean
            }
        })
    }

    #[getter]
    fn get_wavelength_range(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.wavelength_range)
    }

    #[getter]
    fn get_width(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.width)
    }

    #[getter]
    fn get_y_mean(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.y_mean)
    }
}


#[pymethods]
impl Design {
    #[getter]
    fn get_compound_prism(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.compound_prism.clone())
    }

    #[getter]
    fn get_detector_array(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.detector_array.clone())
    }

    #[getter]
    fn get_gaussian_beam(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.gaussian_beam.clone())
    }

    #[getter]
    fn get_fitness(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.fitness)
    }
}

#[pymethods]
impl OptimizationConfig {
    #[getter]
    fn get_iteration_count(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.iteration_count)
    }

    #[getter]
    fn get_population_size(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.population_size)
    }

    #[getter]
    fn get_offspring_size(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.offspring_size)
    }

    #[getter]
    fn get_crossover_distribution_index(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.crossover_distribution_index)
    }

    #[getter]
    fn get_mutation_distribution_index(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.mutation_distribution_index)
    }

    #[getter]
    fn get_mutation_probability(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.mutation_probability)
    }

    #[getter]
    fn get_seed(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.seed)
    }

    #[getter]
    fn get_epsilons(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok((self.epsilons[0], self.epsilons[1], self.epsilons[2]))
    }
}

#[pymethods]
impl CompoundPrismConfig {
    #[getter]
    fn get_max_count(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.max_count)
    }

    #[getter]
    fn get_max_height(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.max_height)
    }

    #[getter]
    fn get_width(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.width)
    }
}

#[pymethods]
impl GaussianBeamConfig {
    #[getter]
    fn get_width(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.width)
    }

    #[getter]
    fn get_wavelength_range(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.wavelength_range)
    }
}


#[pymethods]
impl DetectorArrayConfig {
    #[getter]
    fn get_length(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.length)
    }

    #[getter]
    fn get_max_incident_angle(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.max_incident_angle)
    }

    #[getter]
    fn get_bounds<'p>(&self, py: Python<'p>) -> PyResult<impl 'p + IntoPy<PyObject>> {
        Ok(Array2::from(self.bounds.clone()).into_pyarray(py))
    }
}


#[pymethods]
impl DesignConfig {
    #[getter]
    fn get_length_unit(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.length_unit.clone().into_owned())
    }

    #[getter]
    fn get_compound_prism(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.compound_prism.clone())
    }

    #[getter]
    fn get_detector_array(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.detector_array.clone())
    }

    #[getter]
    fn get_gaussian_beam(&self) -> PyResult<impl IntoPy<PyObject>> {
        Ok(self.gaussian_beam.clone())
    }

    fn optimize(
        &self,
        py: Python,
        catalog: Option<&PyAny>,
    ) -> PyResult<Vec<Design>> {
        if let Some(_) = catalog {
            return Err(pyo3::exceptions::NotImplementedError::py_err(
                "Custom glass catalogs have not been implemented yet",
            ));
        }
        let designs = py.allow_threads(|| self.optimize_designs(None));
        Ok(designs)
    }
}

#[pyfunction]
fn evaluate(
    py: Python,
    compound_prism: &CompoundPrismDesign,
    detector_array: &DetectorArrayDesign,
    gaussian_beam: &GaussianBeamDesign,
) -> PyResult<Design> {
    let cmpnd = compound_prism.into();
    let detarr = detector_array.into();
    let beam = gaussian_beam.into();
    let fit = py.allow_threads(|| fitness(&cmpnd, &detarr, &beam))?;
    Ok(Design {
        compound_prism: compound_prism.clone(),
        detector_array: detector_array.clone(),
        gaussian_beam: gaussian_beam.clone(),
        fitness: fit
    })
}

#[pymethods]
impl Design {
    #[new]
    fn create(obj: &PyRawObject,
              compound_prism: &CompoundPrismDesign,
              detector_array: &DetectorArrayDesign,
              gaussian_beam: &GaussianBeamDesign,
              py: Python,
    ) -> PyResult<()> {
        obj.init({
             let cmpnd = compound_prism.into();
             let detarr = detector_array.into();
             let beam = gaussian_beam.into();
             let fit = py.allow_threads(|| fitness(&cmpnd, &detarr, &beam))?;
             Design {
                 compound_prism: compound_prism.clone(),
                 detector_array: detector_array.clone(),
                 gaussian_beam: gaussian_beam.clone(),
                 fitness: fit
             }
         });
        Ok(())
    }

    pub fn transmission_probability<'p>(
        &self,
        wavelengths: &PyArray1<f64>,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray2<f64>> {
        let cmpnd = (&self.compound_prism).into();
        let detarr = (&self.detector_array).into();
        let detpos = (&self.detector_array).into();
        let beam = (&self.gaussian_beam).into();
        py.allow_threads(|| {
            wavelengths
                .as_array()
                .into_iter()
                .flat_map(|w| p_dets_l_wavelength(*w, &cmpnd, &detarr, &beam, &detpos))
                .collect::<Box<_>>()
        })
        .into_pyarray(py)
        .reshape((wavelengths.len(), self.detector_array.bins.len()))
        .map_err(|e| e.into())
    }

    pub fn ray_trace<'p>(
        &self,
        wavelength: f64,
        inital_y: f64,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray2<f64>> {
        let cmpnd = (&self.compound_prism).into();
        let detarr = (&self.detector_array).into();
        let detpos = (&self.detector_array).into();
        Ok(Array2::from(
            trace(wavelength, inital_y, &cmpnd, &detarr, &detpos)
                .map(|r| r.map(|p| [p.x, p.y]))
                .collect::<Result<Vec<_>, _>>()?,
        )
        .into_pyarray(py))
    }
}

#[pyfunction]
fn serialize_results<'p>(
    py: Python<'p>,
    design_config: &DesignConfig,
    designs: Vec<&Design>,
) -> PyResult<&'p PyBytes> {
    serde_cbor::to_vec(&(design_config, designs))
        .map_err(|e| pyo3::exceptions::TypeError::py_err(e.to_string()))
        .map(|v| PyBytes::new(py, &v))
}

#[pyfunction]
fn deserialize_results(
    bytes: &[u8],
) -> PyResult<(DesignConfig, Vec<Design>)> {
    serde_cbor::from_slice(bytes)
        .map_err(|e| pyo3::exceptions::TypeError::py_err(e.to_string()))
}

#[pyfunction]
fn config_from_toml(
    toml_str: &str,
) -> PyResult<DesignConfig> {
    toml::from_str(toml_str)
        .map_err(|e| pyo3::exceptions::TypeError::py_err(e.to_string()))
}


#[pymodule]
fn compound_prism_designer(py: Python, m: &PyModule) -> PyResult<()> {
    // TODO: Better way to expose exception types
    use pyo3::type_object::PyTypeObject;
    m.add("GlassCatalogError", GlassCatalogError::type_object())?;
    m.add("RayTraceError", RayTraceError::type_object())?;
    m.add_class::<PyGlass>()?;
    m.add("BUNDLED_CATALOG", BUNDLED_CATALOG.iter().cloned().map(|(s, g)| Py::new(py, PyGlass {
        name: s.to_owned(),
        glass: g
    })).collect::<PyResult<Vec<_>>>()?)?;
    m.add_class::<CompoundPrismDesign>()?;
    m.add_class::<DetectorArrayDesign>()?;
    m.add_class::<GaussianBeamDesign>()?;
    m.add_class::<DesignFitness>()?;
    m.add_class::<Design>()?;
    m.add_class::<OptimizationConfig>()?;
    m.add_class::<CompoundPrismConfig>()?;
    m.add_class::<GaussianBeamConfig>()?;
    m.add_class::<DetectorArrayConfig>()?;
    m.add_class::<DesignConfig>()?;
    m.add_wrapped(wrap_pyfunction!(create_glass_catalog))?;
    m.add_wrapped(wrap_pyfunction!(evaluate))?;
    m.add_wrapped(wrap_pyfunction!(serialize_results))?;
    m.add_wrapped(wrap_pyfunction!(deserialize_results))?;
    m.add_wrapped(wrap_pyfunction!(config_from_toml))?;
    Ok(())
}
