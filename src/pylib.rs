use crate::glasscat::*;
use crate::optimizer::*;
use crate::ray::{
    p_dets_l_wavelength, trace, CompoundPrism, DesignFitness, DetectorArray,
    DetectorArrayPositioning, GaussianBeam,
};
use ndarray::prelude::{array, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::create_exception;
use pyo3::exceptions::Exception;
use pyo3::prelude::*;
use pyo3::types::PyAny;
use pyo3::wrap_pyfunction;
use pyo3::ObjectProtocol;

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
        .map(|r| {
            r.map(|(name, glass)| PyGlass {
                name: name.into(),
                glass,
            })
            .map_err(|e| e.into())
        })
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
            self.width,
        )
    }
}

impl From<CompoundPrismDesign> for PyCompoundPrism {
    fn from(design: CompoundPrismDesign) -> Self {
        PyCompoundPrism {
            glasses: design
                .glasses
                .into_vec()
                .into_iter()
                .map(|(name, glass)| PyGlass {
                    name: name.to_string(),
                    glass,
                })
                .collect(),
            angles: design.angles.into_vec(),
            lengths: design.lengths.into_vec(),
            curvature: design.curvature,
            height: design.height,
            width: design.width,
        }
    }
}

#[pymethods]
impl PyCompoundPrism {
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
    max_incident_angle: f64,
    #[pyo3(get)]
    angle: f64,
}

impl<'s> Into<DetectorArray<'s>> for &'s PyDetectorArray {
    fn into(self) -> DetectorArray<'s> {
        DetectorArray::new(
            self.bins.as_slice().into(),
            self.max_incident_angle.to_radians().cos(),
            self.angle,
            self.length,
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

impl From<DetectorArrayDesign> for PyDetectorArray {
    fn from(design: DetectorArrayDesign) -> Self {
        PyDetectorArray {
            bins: design.bins.to_vec(),
            position: design.position.into(),
            direction: design.direction.into(),
            length: design.length,
            max_incident_angle: design.max_incident_angle,
            angle: design.angle,
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
            w_range: self.wavelength_range,
        }
    }
}

impl From<GaussianBeamDesign> for PyGaussianBeam {
    fn from(design: GaussianBeamDesign) -> Self {
        PyGaussianBeam {
            wavelength_range: design.wavelength_range,
            width: design.width,
            y_mean: design.y_mean,
        }
    }
}

fn get<'p, O: ObjectProtocol, T: FromPyObject<'p>>(obj: &'p O, name: &'static str) -> PyResult<T> {
    if obj.hasattr(name)? {
        obj.getattr(name)?
    } else {
        obj.get_item(name)?
    }
    .extract()
}

impl<'source> FromPyObject<'source> for OptimizationConfig {
    fn extract(obj: &'source PyAny) -> Result<Self, PyErr> {
        let eps: (f64, f64, f64) = get(obj, "epsilons")?;
        Ok(Self {
            iteration_count: get(obj, "iteration_count")?,
            population_size: get(obj, "population_size")?,
            offspring_size: get(obj, "offspring_size")?,
            crossover_distribution_index: get(obj, "crossover_distribution_index")?,
            mutation_distribution_index: get(obj, "mutation_distribution_index")?,
            mutation_probability: get(obj, "mutation_probability")?,
            seed: get(obj, "seed")?,
            epsilons: [eps.0, eps.1, eps.2],
        })
    }
}

impl<'source> FromPyObject<'source> for CompoundPrismConfig {
    fn extract(obj: &'source PyAny) -> Result<Self, PyErr> {
        Ok(Self {
            max_count: get(obj, "max_count")?,
            max_height: get(obj, "max_height")?,
            width: get(obj, "width")?,
        })
    }
}

impl<'source> FromPyObject<'source> for GaussianBeamConfig {
    fn extract(obj: &'source PyAny) -> Result<Self, PyErr> {
        Ok(Self {
            width: get(obj, "width")?,
            wavelength_range: get(obj, "wavelength_range")?,
        })
    }
}

impl<'source> FromPyObject<'source> for DetectorArrayConfig {
    fn extract(obj: &'source PyAny) -> Result<Self, PyErr> {
        let bins: &PyArray2<f64> = get(obj, "bin_bounds")?;
        if bins.dims()[1] != 2 {
            return Err(pyo3::exceptions::TypeError::py_err(
                "bin_bounds must have a shape of [_, 2]",
            ));
        };
        Ok(Self {
            length: get(obj, "length")?,
            max_incident_angle: get(obj, "max_incident_angle")?,
            bin_bounds: bins
                .as_array()
                .genrows()
                .into_iter()
                .map(|r| [r[0], r[1]])
                .collect(),
        })
    }
}

impl<'source> FromPyObject<'source> for DesignConfig {
    fn extract(obj: &'source PyAny) -> Result<Self, PyErr> {
        Ok(Self {
            optimizer: get(obj, "optimizer")?,
            compound_prism: get(obj, "compound_prism")?,
            detector_array: get(obj, "detector_array")?,
            gaussian_beam: get(obj, "gaussian_beam")?,
        })
    }
}

#[pyfunction]
fn optimize(
    py: Python,
    catalog: Option<&PyAny>,
    design_config: DesignConfig,
) -> PyResult<Vec<PyDesign>> {
    if let Some(ref _c) = catalog {
        return Err(pyo3::exceptions::NotImplementedError::py_err(
            "Custom glass catalogs have not been implemented yet",
        ));
    }
    let glass_catalog = catalog
        .map(|catalog| {
            catalog
                .iter()?
                .map(|p| {
                    let pg = p?.downcast_ref::<PyGlass>()?;
                    Ok((pg.name.clone(), pg.glass.clone()))
                })
                .collect::<PyResult<_>>()
        })
        .transpose()?;
    let designs = py.allow_threads(|| design_config.optimize(glass_catalog));
    Ok(designs.into_vec().into_iter().map(|d| d.into()).collect())
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
            deviation: fitness.deviation,
        }
    }
}

#[pyclass(name=Design)]
pub struct PyDesign {
    #[pyo3(get)]
    compound_prism: PyCompoundPrism,
    #[pyo3(get)]
    detector_array: PyDetectorArray,
    #[pyo3(get)]
    gaussian_beam: PyGaussianBeam,
    #[pyo3(get)]
    fitness: PyDesignFitness,
}

impl From<Design> for PyDesign {
    fn from(design: Design) -> Self {
        PyDesign {
            compound_prism: design.compound_prism.into(),
            detector_array: design.detector_array.into(),
            gaussian_beam: design.gaussian_beam.into(),
            fitness: design.fitness.into(),
        }
    }
}

#[pymethods]
impl PyDesign {
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
    m.add_class::<PyDesignFitness>()?;
    m.add_class::<PyDesign>()?;
    m.add_wrapped(wrap_pyfunction!(create_glass_catalog))?;
    m.add_wrapped(wrap_pyfunction!(optimize))?;
    Ok(())
}
