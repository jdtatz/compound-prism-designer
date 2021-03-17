use compound_prism_spectrometer::*;
use core::convert::TryInto;
use ndarray::array;
use ndarray::prelude::Array2;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::{
    create_exception,
    gc::{PyGCProtocol, PyVisit},
    prelude::*,
    PyObjectProtocol, PyTraverseError,
};

create_exception!(
    compound_prism_designer,
    RayTraceError,
    pyo3::exceptions::PyException
);

#[pyclass(name = "Glass", module = "compound_prism_designer")]
#[derive(Clone, Debug)]
pub struct PyGlass {
    /// Glass Name : str
    #[pyo3(get)]
    name: String,
    glass: Glass<f64, 6>,
}

#[pymethods]
impl PyGlass {
    #[classattr]
    const ORDER: usize = 5;

    #[new]
    fn create(name: String, coefficents: &PyArray1<f64>) -> PyResult<Self> {
        Ok(PyGlass {
            name,
            glass: Glass {
                coefficents: coefficents.to_vec()?.try_into().map_err(|_| {
                    pyo3::exceptions::PyValueError::new_err("Incorrect coefficents length")
                })?,
            },
        })
    }

    fn __getnewargs__<'p>(&self, py: Python<'p>) -> PyResult<impl IntoPy<PyObject> + 'p> {
        Ok((
            self.name.clone(),
            PyArray1::from_slice(py, &self.glass.coefficents),
        ))
    }

    /// __call__(self, w, /)
    /// --
    ///
    /// Computes the index of refraction of the glass for the given wavelength
    ///
    /// Args:
    ///     w (float): wavelength given in units of micrometers
    #[call]
    fn __call__(&self, w: f64) -> f64 {
        self.glass.calc_n(w)
    }

    #[getter]
    fn get_glass<'p>(&self, py: Python<'p>) -> impl IntoPy<PyObject> + 'p {
        (
            self.name.clone(),
            PyArray1::from_slice(py, &self.glass.coefficents),
        )
    }
}

#[pyproto]
impl PyObjectProtocol for PyGlass {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}: {}", self.name, self.glass))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

#[pyclass(name = "DesignFitness", module = "compound_prism_designer")]
#[derive(Debug, Clone, Copy)]
struct PyDesignFitness {
    #[pyo3(get)]
    size: f64,
    #[pyo3(get)]
    info: f64,
    #[pyo3(get)]
    deviation: f64,
}

#[pymethods]
impl PyDesignFitness {
    #[new]
    fn create(size: f64, info: f64, deviation: f64) -> Self {
        Self {
            size,
            info,
            deviation,
        }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.size, self.info, self.deviation)
    }
}

#[pyproto]
impl PyObjectProtocol for PyDesignFitness {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

impl<F: Float> From<crate::DesignFitness<F>> for PyDesignFitness {
    fn from(fit: crate::DesignFitness<F>) -> Self {
        Self {
            size: fit.size.to_f64(),
            info: fit.info.to_f64(),
            deviation: fit.deviation.to_f64(),
        }
    }
}

impl<F: Float> Into<crate::DesignFitness<F>> for PyDesignFitness {
    fn into(self) -> crate::DesignFitness<F> {
        crate::DesignFitness {
            size: F::from_f64(self.size),
            info: F::from_f64(self.info),
            deviation: F::from_f64(self.deviation),
        }
    }
}

#[pyclass(name = "CompoundPrism", gc, module = "compound_prism_designer")]
#[derive(Debug, Clone)]
struct PyCompoundPrism {
    compound_prism: CompoundPrism<Pair<f64>>,
    #[pyo3(get)]
    glasses: Vec<Py<PyGlass>>,
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
    #[pyo3(get)]
    ar_coated: bool,
}

#[pymethods]
impl PyCompoundPrism {
    #[new]
    fn create(
        glasses: Vec<Py<PyGlass>>,
        angles: Vec<f64>,
        lengths: Vec<f64>,
        curvature: f64,
        height: f64,
        width: f64,
        ar_coated: bool,
        py: Python,
    ) -> Self {
        let compound_prism = CompoundPrism::new(
            glasses
                .iter()
                .map(|pg| pg.as_ref(py).borrow().glass.clone()),
            &angles,
            &lengths,
            curvature,
            height,
            width,
            ar_coated,
        );
        PyCompoundPrism {
            compound_prism,
            glasses,
            angles,
            lengths,
            curvature,
            height,
            width,
            ar_coated,
        }
    }

    fn __getnewargs__(&self, py: Python) -> PyResult<impl IntoPy<PyObject> + '_> {
        Ok((
            self.glasses
                .iter()
                .map(|p| p.as_ref(py).try_borrow().map(|v| v.clone()))
                .collect::<Result<Vec<PyGlass>, _>>()?,
            self.angles.clone(),
            self.lengths.clone(),
            self.curvature,
            self.height,
            self.width,
            self.ar_coated,
        ))
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
        let (polys, lens_poly, lens_center, lens_radius) = self.compound_prism.polygons();
        let polys = polys
            .into_iter()
            .map(|[p0, p1, p2, p3]| {
                (array![[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]]).to_pyarray(py)
            })
            .collect();
        let [p0, p1, p2, p3] = lens_poly;
        let lens_poly =
            (array![[p0.x, p0.y], [p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y]]).to_pyarray(py);
        let lens_center = PyArray1::from_slice(py, &[lens_center.x, lens_center.y]);
        Ok((polys, lens_poly, lens_center, lens_radius))
    }
}

#[pyproto]
impl PyGCProtocol for PyCompoundPrism {
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        for obj in self.glasses.iter() {
            visit.call(obj)?
        }
        Ok(())
    }

    fn __clear__(&mut self) {}
}

#[pyproto]
impl PyObjectProtocol for PyCompoundPrism {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

#[pyclass(name = "DetectorArray", module = "compound_prism_designer")]
#[derive(Debug, Clone)]
struct PyDetectorArray {
    detector_array: LinearDetectorArray<Pair<f64>>,
    #[pyo3(get)]
    bin_count: u32,
    #[pyo3(get)]
    bin_size: f64,
    #[pyo3(get)]
    linear_slope: f64,
    #[pyo3(get)]
    linear_intercept: f64,
    #[pyo3(get)]
    length: f64,
    #[pyo3(get)]
    max_incident_angle: f64,
    #[pyo3(get)]
    angle: f64,
}

#[pymethods]
impl PyDetectorArray {
    #[new]
    fn create(
        bin_count: u32,
        bin_size: f64,
        linear_slope: f64,
        linear_intercept: f64,
        length: f64,
        max_incident_angle: f64,
        angle: f64,
    ) -> Self {
        let detector_array = LinearDetectorArray::new(
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
            max_incident_angle.cos(),
            angle,
            length,
        );
        PyDetectorArray {
            detector_array,
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
            length,
            max_incident_angle,
            angle,
        }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (
            self.bin_count,
            self.bin_size,
            self.linear_slope,
            self.linear_intercept,
            self.length,
            self.max_incident_angle,
            self.angle,
        )
    }
}

#[pyproto]
impl PyObjectProtocol for PyDetectorArray {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

#[pyclass(name = "GaussianBeam", module = "compound_prism_designer")]
#[derive(Debug, Clone)]
struct PyGaussianBeam {
    gaussian_beam: GaussianBeam<f64, UniformDistribution<f64>>,
    #[pyo3(get)]
    wavelength_range: (f64, f64),
    #[pyo3(get)]
    width: f64,
    #[pyo3(get)]
    y_mean: f64,
}

#[pymethods]
impl PyGaussianBeam {
    #[new]
    fn create(wavelength_range: (f64, f64), width: f64, y_mean: f64) -> Self {
        let gaussian_beam = GaussianBeam {
            width,
            y_mean,
            wavelengths: UniformDistribution {
                bounds: wavelength_range,
            },
        };
        PyGaussianBeam {
            gaussian_beam,
            wavelength_range,
            width,
            y_mean,
        }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.wavelength_range, self.width, self.y_mean)
    }
}

#[pyproto]
impl PyObjectProtocol for PyGaussianBeam {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

/// Spectrometer(compound_prism, detector_array, gaussian_beam, /)
/// --
///
/// Compound Prism Spectrometer specification
///
/// Args:
///     compound_prism (CompoundPrism): spasefiowhpiueh
#[pyclass(name = "Spectrometer", gc, module = "compound_prism_designer")]
#[derive(Debug, Clone)]
struct PySpectrometer {
    spectrometer: Spectrometer<Pair<f64>, GaussianBeam<f64, UniformDistribution<f64>>>,
    /// compound prism specification : CompoundPrism
    #[pyo3(get)]
    compound_prism: Py<PyCompoundPrism>,
    /// linear detector array specification : DetectorArray
    #[pyo3(get)]
    detector_array: Py<PyDetectorArray>,
    /// input gaussian beam specification : GaussianBeam
    #[pyo3(get)]
    gaussian_beam: Py<PyGaussianBeam>,
    /// detector array position : (float, float)
    #[pyo3(get)]
    position: (f64, f64),
    /// detector array direction : (float, float)
    #[pyo3(get)]
    direction: (f64, f64),
}

#[pymethods]
impl PySpectrometer {
    #[new]
    fn create(
        compound_prism: Py<PyCompoundPrism>,
        detector_array: Py<PyDetectorArray>,
        gaussian_beam: Py<PyGaussianBeam>,
        py: Python,
    ) -> PyResult<Self> {
        let spectrometer = Spectrometer::new(
            gaussian_beam.as_ref(py).try_borrow()?.gaussian_beam.clone(),
            compound_prism
                .as_ref(py)
                .try_borrow()?
                .compound_prism
                .clone(),
            detector_array
                .as_ref(py)
                .try_borrow()?
                .detector_array
                .clone(),
        )
        .map_err(|err| {
            RayTraceError::new_err(<compound_prism_spectrometer::RayTraceError as Into<
                &'static str,
            >>::into(err))
        })?;
        Ok(PySpectrometer {
            compound_prism,
            detector_array,
            gaussian_beam,
            position: (
                spectrometer.detector.1.position.x,
                spectrometer.detector.1.position.y,
            ),
            direction: (
                spectrometer.detector.1.direction.x,
                spectrometer.detector.1.direction.y,
            ),
            spectrometer,
        })
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> + '_ {
        (
            &self.compound_prism,
            &self.detector_array,
            &self.gaussian_beam,
        )
    }

    /// cpu_fitness(self, /)
    /// --
    ///
    /// Computes the spectrometer fitness using on the cpu
    #[args("*", max_n = 16_384, max_m = 16_384)]
    fn cpu_fitness(&self, py: Python, max_n: usize, max_m: usize) -> PyDesignFitness {
        py.allow_threads(|| crate::fitness(&self.spectrometer, max_n, max_m).into())
    }

    #[args(wavelengths, "*", max_m = 16_384)]
    pub fn transmission_probability<'p>(
        &self,
        wavelengths: &PyArray1<f64>,
        py: Python<'p>,
        max_m: usize,
    ) -> PyResult<&'p PyArray2<f64>> {
        let spec = &self.spectrometer;
        let readonly = wavelengths.readonly();
        let wavelengths_array = readonly.as_array();
        py.allow_threads(|| {
            wavelengths_array
                .into_iter()
                .flat_map(|w| crate::p_dets_l_wavelength(&spec, *w, max_m))
                .collect::<Vec<_>>()
        })
        .to_pyarray(py)
        .reshape((
            wavelengths.len(),
            self.detector_array.as_ref(py).try_borrow()?.bin_count as usize,
        ))
        .map_err(|e| e.into())
    }

    pub fn ray_trace<'p>(
        &self,
        wavelength: f64,
        inital_y: f64,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray2<f64>> {
        let spec = &self.spectrometer;
        Ok(Array2::from(
            spec.trace_ray_path(wavelength, inital_y)
                .map(|r| r.map(|p| [p.x, p.y]))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|err| {
                    RayTraceError::new_err(<compound_prism_spectrometer::RayTraceError as Into<
                        &'static str,
                    >>::into(err))
                })?,
        )
        .to_pyarray(py))
    }

    pub fn to_string(&self) -> String {
        format!("{:#?}", self.spectrometer)
    }

    #[cfg(feature = "cuda")]
    #[args("*", max_n = 256, nwarp = 2, max_eval = 16_384)]
    fn gpu_fitness(
        &self,
        py: Python,
        seeds: &PyArray1<f64>,
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> Option<PyDesignFitness> {
        let seeds = seeds.readonly();
        let seeds = seeds.as_slice().unwrap();
        let spec: Spectrometer<Pair<f32>, GaussianBeam<f32, _>> =
            LossyInto::lossy_into(self.spectrometer.clone());
        let fit = py.allow_threads(|| crate::cuda_fitness(&spec, seeds, max_n, nwarp, max_eval))?;
        Some(fit.into())
    }

    #[cfg(feature = "cuda")]
    #[args("*", max_n = 256, nwarp = 2, max_eval = 16_384)]
    fn slow_gpu_fitness(
        &self,
        py: Python,
        seeds: &PyArray1<f64>,
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> Option<PyDesignFitness> {
        let seeds = seeds.readonly();
        let seeds = seeds.as_slice().unwrap();
        let fit = py.allow_threads(|| {
            crate::cuda_fitness(&self.spectrometer, seeds, max_n, nwarp, max_eval)
        })?;
        Some(fit.into())
    }
}

#[pyproto]
impl PyGCProtocol for PySpectrometer {
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        visit.call(&self.compound_prism)?;
        visit.call(&self.detector_array)?;
        visit.call(&self.gaussian_beam)?;
        Ok(())
    }

    fn __clear__(&mut self) {}
}

#[pyproto]
impl PyObjectProtocol for PySpectrometer {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

/// This module is implemented in Rust.
#[pymodule]
fn compound_prism_designer(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("RayTraceError", py.get_type::<RayTraceError>())?;
    m.add_class::<PyGlass>()?;
    m.add_class::<PyDesignFitness>()?;
    m.add_class::<PyCompoundPrism>()?;
    m.add_class::<PyDetectorArray>()?;
    m.add_class::<PyGaussianBeam>()?;
    m.add_class::<PySpectrometer>()?;
    Ok(())
}
