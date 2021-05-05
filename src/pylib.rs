use compound_prism_spectrometer::*;
use core::convert::TryInto;
use ndarray::{array, Array2};
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

fn map_ray_trace_err(err: compound_prism_spectrometer::RayTraceError) -> PyErr {
    RayTraceError::new_err(<compound_prism_spectrometer::RayTraceError as Into<
        &'static str,
    >>::into(err))
}

create_exception!(
    compound_prism_designer,
    CudaError,
    pyo3::exceptions::PyException
);

#[cfg(feature = "cuda")]
fn map_cuda_err(err: rustacuda::error::CudaError) -> PyErr {
    CudaError::new_err(err.to_string())
}

#[pyclass(name = "Glass", module = "compound_prism_designer")]
#[text_signature = "(name, coefficents)"]
#[derive(Debug, Display, Clone)]
#[display(fmt = "{}: {}", name, glass)]
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
        PyArray1::from_slice(py, &self.glass.coefficents)
    }
}

#[pyproto]
impl PyObjectProtocol for PyGlass {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

fn pyglasses_to_glasses<const N: usize>(
    py: Python,
    pyglasses: &[Py<PyGlass>],
) -> [Glass<f64, 6>; N] {
    let pyglasses: &[Py<PyGlass>; N] = pyglasses.try_into().unwrap();
    let pyglasses: [&Py<PyGlass>; N] = pyglasses.each_ref();
    pyglasses.map(|pg| pg.as_ref(py).borrow().glass)
}

#[pyclass(name = "DesignFitness", module = "compound_prism_designer")]
#[text_signature = "(size, info, deviation)"]
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

impl<F: LossyInto<f64>> From<crate::DesignFitness<F>> for PyDesignFitness {
    fn from(fit: crate::DesignFitness<F>) -> Self {
        Self {
            size: fit.size.lossy_into(),
            info: fit.info.lossy_into(),
            deviation: fit.deviation.lossy_into(),
        }
    }
}

impl<F: LossyFrom<f64>> From<PyDesignFitness> for crate::DesignFitness<F> {
    fn from(fit: PyDesignFitness) -> Self {
        Self {
            size: F::lossy_from(fit.size),
            info: F::lossy_from(fit.info),
            deviation: F::lossy_from(fit.deviation),
        }
    }
}

#[pyclass(
    name = "UniformWavelengthDistribution",
    module = "compound_prism_designer"
)]
#[text_signature = "(bounds)"]
#[derive(Debug, Clone, Copy)]
struct PyUniformWavelengthDistribution {
    distribution: UniformDistribution<f64>,
}

#[pymethods]
impl PyUniformWavelengthDistribution {
    #[new]
    fn create(bounds: (f64, f64)) -> PyResult<Self> {
        if bounds.1 < bounds.0 {
            Err(pyo3::exceptions::PyValueError::new_err(
                "lower bound > upper bound",
            ))
        } else {
            Ok(Self {
                distribution: UniformDistribution { bounds },
            })
        }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.distribution.bounds,)
    }

    #[getter]
    fn get_bounds(&self) -> impl IntoPy<PyObject> {
        self.distribution.bounds
    }
}

#[derive(Debug, Clone, Copy, FromPyObject)]
enum WavelengthDistributions {
    Uniform(PyUniformWavelengthDistribution),
}

macro_rules! map_wavelength_distributions {
    ($wavelength_distribution:expr => |$distribution:ident| $body:expr ) => {
        match $wavelength_distribution {
            WavelengthDistributions::Uniform(PyUniformWavelengthDistribution {
                distribution: $distribution,
            }) => $body,
        }
    };
}

impl IntoPy<PyObject> for WavelengthDistributions {
    fn into_py(self, py: pyo3::Python<'_>) -> PyObject {
        match self {
            WavelengthDistributions::Uniform(v) => v.into_py(py),
        }
    }
}

#[pyclass(name = "GaussianBeam", module = "compound_prism_designer")]
#[text_signature = "(width, y_mean)"]
#[derive(Debug, Clone, Copy)]
struct PyGaussianBeam {
    gaussian_beam: GaussianBeam<f64, 2>,
    #[pyo3(get)]
    width: f64,
    #[pyo3(get)]
    y_mean: f64,
}

#[pymethods]
impl PyGaussianBeam {
    #[new]
    fn create(width: f64, y_mean: f64) -> Self {
        let gaussian_beam = GaussianBeam {
            width,
            y_mean,
            marker: core::marker::PhantomData,
        };
        PyGaussianBeam {
            gaussian_beam,
            width,
            y_mean,
        }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.width, self.y_mean)
    }
}

#[pyproto]
impl PyObjectProtocol for PyGaussianBeam {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

#[pyclass(name = "FiberBeam", module = "compound_prism_designer")]
#[text_signature = "(core_radius, numerical_aperture, center_y)"]
#[derive(Debug, Clone, Copy)]
struct PyFiberBeam {
    fiber_beam: FiberBeam<f64, 3>,
    #[pyo3(get)]
    core_radius: f64,
    #[pyo3(get)]
    numerical_aperture: f64,
    #[pyo3(get)]
    center_y: f64,
}

#[pymethods]
impl PyFiberBeam {
    #[new]
    fn create(core_radius: f64, numerical_aperture: f64, center_y: f64) -> Self {
        let fiber_beam = FiberBeam::new(core_radius, numerical_aperture, center_y);
        PyFiberBeam {
            fiber_beam,
            core_radius,
            numerical_aperture,
            center_y,
        }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.core_radius, self.numerical_aperture, self.center_y)
    }
}

#[pyproto]
impl PyObjectProtocol for PyFiberBeam {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

#[derive(Debug, Clone, Copy, FromPyObject)]
enum BeamDistributions {
    Gaussian(PyGaussianBeam),
    Fiber(PyFiberBeam),
}

macro_rules! map_beam_distributions {
    ($beam_distribution:expr => |$distribution:ident| $body:expr ) => {
        match $beam_distribution {
            BeamDistributions::Gaussian(PyGaussianBeam {
                gaussian_beam: $distribution,
                ..
            }) => $body,
            BeamDistributions::Fiber(PyFiberBeam {
                fiber_beam: $distribution,
                ..
            }) => $body,
        }
    };
}

impl IntoPy<PyObject> for BeamDistributions {
    fn into_py(self, py: pyo3::Python<'_>) -> PyObject {
        match self {
            BeamDistributions::Gaussian(v) => v.into_py(py),
            BeamDistributions::Fiber(v) => v.into_py(py),
        }
    }
}

const MAX_GLASS: usize = 6;

macro_rules! call_sized_macro {
    ($cmacro:ident $($tt:tt)*) => {
        $cmacro! { [0, 1, 2, 3, 4, 5, 6] $($tt)* }
    };
}

macro_rules! define_sized_compound_prism {
    ([$($n:literal),+]) => {
        paste::paste! {
            #[derive(Debug, Clone, Copy, From)]
            enum SizedCompoundPrism<F: FloatExt, S0: Surface<F, 2>, SI: Surface<F, 2>, SN: Surface<F, 2>> {
                $( [<CompoundPrism $n>](CompoundPrism<F, S0, SI, SN, $n, 2>) ),*
            }
        }
    };
}

call_sized_macro! { define_sized_compound_prism }

macro_rules! map_sized_compound_prism {
    ([$($n:literal),*]; $sized_compound_prism:expr => |$compound_prism:ident| $body:expr) => {
        paste::paste! {
            match $sized_compound_prism {
                $( SizedCompoundPrism::[<CompoundPrism $n>]($compound_prism) => $body ),*
            }
        }
    };
    ($sized_compound_prism:expr => |$compound_prism:ident| $body:expr ) => {
        call_sized_macro! { map_sized_compound_prism ; $sized_compound_prism => |$compound_prism| $body }
    };
}

macro_rules! create_sized_compound_prism {
    ([$($n:literal),*]; $len:expr => $create:expr) => {
        paste::paste! {
            match ($len - 1) {
                $($n => SizedCompoundPrism::[<CompoundPrism $n>]($create) ,)*
                _ => unreachable!("Programmer Error during CompoundPrism creation"),
            }
        }
    };
    ($len:expr => $create:expr) => {
        call_sized_macro! {create_sized_compound_prism ; $len => $create}
    };
}

macro_rules! define_sized_spectrometer {
    ([$($n:literal),*]) => {
        paste::paste! {
            #[derive(Debug, Clone, Copy, From, WrappedFrom)]
            #[wrapped_from(trait = "LossyFrom", function = "lossy_from")]
            enum SizedSpectrometer<
                F: FloatExt,
                W: Distribution<F, Output = F>,
                B: Distribution<F, Output = Ray<F, 2>>,
                S0: Surface<F, 2>,
                SI: Surface<F, 2>,
                SN: Surface<F, 2>,
                > {
                $( [<Spectrometer $n>](Spectrometer<F, W, B, S0, SI, SN, $n, 2>) ),*
            }
        }
    };
}

call_sized_macro! { define_sized_spectrometer }

macro_rules! map_sized_spectrometer {
    ([$($n:literal),*]; $sized_spectrometer:expr => |$spectrometer:ident| $body:expr) => {
        paste::paste! {
            match $sized_spectrometer {
                $( SizedSpectrometer::[<Spectrometer $n>]($spectrometer) => $body ),*
            }
        }
    };
    ($sized_spectrometer:expr => |$spectrometer:ident| $body:expr ) => {
        call_sized_macro! { map_sized_spectrometer ; $sized_spectrometer => |$spectrometer| $body }
    };
}

macro_rules! create_sized_spectrometer {
    ([$($n:literal),*]; $sized_compound_prism:expr; $wavelengths:ident; $beam:ident; $detarr:ident) => {
        paste::paste! {
            match $sized_compound_prism {
                $(SizedCompoundPrism::[<CompoundPrism $n>](c) => SizedSpectrometer::[<Spectrometer $n>](Spectrometer::new($wavelengths, $beam, c, $detarr)),)*
            }
        }
    };
    ($sized_compound_prism:expr; $wavelengths:ident; $beam:ident; $detarr:ident) => {
        call_sized_macro! { create_sized_spectrometer ; $sized_compound_prism; $wavelengths; $beam; $detarr }
    }
}

#[pyclass(name = "CompoundPrism", gc, module = "compound_prism_designer")]
#[text_signature = "(glasses, angles, lengths, curvature, height, width, ar_coated)"]
#[derive(Debug, Clone)]
struct PyCompoundPrism {
    compound_prism: SizedCompoundPrism<f64, Plane<f64, 2>, Plane<f64, 2>, CurvedPlane<f64, 2>>,
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
    ) -> PyResult<Self> {
        let (first_angle, angles_rest) = angles.split_first().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("`angles` must have at least 2 elements")
        })?;
        if glasses.len() > MAX_GLASS {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Maximum of {} prisms are allowed",
                MAX_GLASS
            )));
        } else if glasses.len() != angles_rest.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "len(glasses) + 1 != len(angles)",
            ));
        } else if glasses.len() != lengths.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "len(glasses) != len(lengths)",
            ));
        }
        let sep_lengths = lengths.as_slice();
        let (last_angle, angles_rest) = angles_rest.split_last().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("`angles` must have at least 2 elements")
        })?;
        let initial_plane_parametrization = PlaneParametrization { height, width };
        let lens_parametrization = CurvedPlaneParametrization {
            signed_normalized_curvature: curvature,
            height,
        };
        let compound_prism = create_sized_compound_prism! { glasses.len() => CompoundPrism::new(
            glasses[0].as_ref(py).borrow().glass,
            pyglasses_to_glasses(py, &glasses[1..]),
            *first_angle,
            angles_rest.try_into().unwrap(),
            *last_angle,
            sep_lengths[0],
            sep_lengths[1..].try_into().unwrap(),
            initial_plane_parametrization,
            lens_parametrization,
            height,
            width,
            ar_coated,
        ) };

        Ok(PyCompoundPrism {
            compound_prism,
            glasses,
            angles,
            lengths,
            curvature,
            height,
            width,
            ar_coated,
        })
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
        map_sized_compound_prism!(self.compound_prism => |c| {
            let (polys, lens_poly, Vector(lens_center), lens_radius) = c.polygons();
            let polys = core::array::IntoIter::new(polys)
                .map(|[Vector(p0), Vector(p1), Vector(p2), Vector(p3)]| {
                    array![p0, p1, p2, p3].to_pyarray(py)
                })
                .collect();
            let [Vector(p0), Vector(p1), Vector(p2), Vector(p3)] = lens_poly;
            let lens_poly = array![p0, p1, p2, p3].to_pyarray(py);
            let lens_center = PyArray1::from_slice(py, &lens_center);
            Ok((polys, lens_poly, lens_center, lens_radius))
        })
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
#[text_signature = "(bin_count, bin_size, linear_slope, linear_intercept, length, max_incident_angle, angle)"]
#[derive(Debug, Clone, Copy)]
struct PyDetectorArray {
    detector_array: LinearDetectorArray<f64, 2>,
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
        compound_prism: Py<PyCompoundPrism>,
        wavelengths: WavelengthDistributions,
        beam: BeamDistributions,
        py: Python,
    ) -> PyResult<Self> {
        let (pos, flipped) = map_sized_compound_prism!(compound_prism.as_ref(py).try_borrow()?.compound_prism => |prism|
            map_beam_distributions!(beam => |beam|
                map_wavelength_distributions!(wavelengths => |ws|
                    detector_array_positioning(
                        prism,
                        length,
                        angle,
                        ws,
                        beam.median_y(),
                    ).map_err(map_ray_trace_err)?
                )
            )
        );
        let detector_array = LinearDetectorArray::new(
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
            max_incident_angle.cos(),
            angle,
            length,
            pos,
            flipped,
        );
        Ok(PyDetectorArray {
            detector_array,
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
            length,
            max_incident_angle,
            angle,
        })
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

/// Compound Prism Spectrometer specification
///
/// Args:
///     compound_prism (CompoundPrism): compound prism specification
///     detector_array (DetectorArray): linear detector array specification
///     wavelengths (UniformWavelengthDistribution): input wavelength distribution specification
///     gaussian_beam (GaussianBeam): input gaussian beam specification
#[pyclass(name = "Spectrometer", gc, module = "compound_prism_designer")]
#[text_signature = "(compound_prism, detector_array, wavelengths, gaussian_beam)"]
#[derive(Debug, Clone)]
struct PySpectrometer {
    spectrometer: SizedSpectrometer<
        f64,
        UniformDistribution<f64>,
        GaussianBeam<f64, 2>,
        Plane<f64, 2>,
        Plane<f64, 2>,
        CurvedPlane<f64, 2>,
    >,
    /// compound prism specification : CompoundPrism
    #[pyo3(get)]
    compound_prism: Py<PyCompoundPrism>,
    /// linear detector array specification : DetectorArray
    #[pyo3(get)]
    detector_array: Py<PyDetectorArray>,
    /// input wavelength distribution specification : UniformWavelengthDistribution
    #[pyo3(get)]
    wavelengths: WavelengthDistributions,
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
        wavelengths: WavelengthDistributions,
        gaussian_beam: Py<PyGaussianBeam>,
        py: Python,
    ) -> PyResult<Self> {
        let WavelengthDistributions::Uniform(PyUniformWavelengthDistribution { distribution: w }) =
            wavelengths;
        let gb = gaussian_beam.as_ref(py).try_borrow()?.gaussian_beam;
        let da = detector_array.as_ref(py).try_borrow()?.detector_array;
        let scp = compound_prism.as_ref(py).try_borrow()?.compound_prism;
        let spectrometer = create_sized_spectrometer!(scp; w; gb; da);
        Ok(PySpectrometer {
            compound_prism,
            detector_array,
            wavelengths,
            gaussian_beam,
            position: map_sized_spectrometer!(spectrometer => |s| (s.detector.position[0], s.detector.position[1])),
            direction: map_sized_spectrometer!(spectrometer => |s| (s.detector.direction[0], s.detector.direction[1])),
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

    /// Computes the spectrometer fitness using on the cpu
    #[text_signature = "($self, /, *, max_n = 16_384, max_m = 16_384)"]
    #[args("*", max_n = 16_384, max_m = 16_384)]
    fn cpu_fitness(&self, py: Python, max_n: usize, max_m: usize) -> PyDesignFitness {
        py.allow_threads(|| map_sized_spectrometer!(self.spectrometer => |s| crate::fitness(&s, max_n, max_m).into()))
    }

    #[text_signature = "($self, /, wavelengths, *, max_m = 16_384)"]
    #[args(wavelengths, "*", max_m = 16_384)]
    pub fn transmission_probability<'p>(
        &self,
        wavelengths: &PyArray1<f64>,
        py: Python<'p>,
        max_m: usize,
    ) -> PyResult<&'p PyArray2<f64>> {
        map_sized_spectrometer!(self.spectrometer => |spec| {
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
        })
    }

    #[text_signature = "($self, /, wavelength, inital_y)"]
    pub fn ray_trace<'p>(
        &self,
        wavelength: f64,
        inital_y: f64,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray2<f64>> {
        let spec = &self.spectrometer;
        Ok(Array2::from(
            map_sized_spectrometer!(spec => |s| s.trace_ray_path(wavelength, inital_y).map(|r| r.map(|Vector(p)| p))
            .collect::<Result<Vec<_>, _>>())
            .map_err(map_ray_trace_err)?,
        )
        .to_pyarray(py))
    }

    /// Computes the spectrometer fitness using on the gpu with float32
    #[cfg(feature = "cuda")]
    #[text_signature = "($self, /, seeds, *, max_n = 256, nwarp = 2, max_eval = 16_384)"]
    #[args("*", max_n = 256, nwarp = 2, max_eval = 16_384)]
    fn gpu_fitness(
        &self,
        py: Python,
        seeds: &PyArray1<f64>,
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> PyResult<Option<PyDesignFitness>> {
        let seeds = seeds.readonly();
        let seeds = seeds.as_slice().unwrap();
        let spec: SizedSpectrometer<f32, _, _, _, _, _> = LossyFrom::lossy_from(self.spectrometer);
        let fit = py.allow_threads(||
            map_sized_spectrometer!(spec => |spec| crate::cuda_fitness(&spec, seeds, max_n, nwarp, max_eval))).map_err(map_cuda_err)?;
        Ok(fit.map(Into::into))
    }

    /// Computes the spectrometer fitness using on the gpu with float64
    #[cfg(feature = "cuda")]
    #[text_signature = "($self, /, seeds, *, max_n = 256, nwarp = 2, max_eval = 16_384)"]
    #[args("*", max_n = 256, nwarp = 2, max_eval = 16_384)]
    fn slow_gpu_fitness(
        &self,
        py: Python,
        seeds: &PyArray1<f64>,
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> PyResult<Option<PyDesignFitness>> {
        let seeds = seeds.readonly();
        let seeds = seeds.as_slice().unwrap();
        let fit = py.allow_threads(|| {
            map_sized_spectrometer!(self.spectrometer => |s| crate::cuda_fitness(&s, seeds, max_n, nwarp, max_eval))
        }).map_err(map_cuda_err)?;
        Ok(fit.map(Into::into))
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
    #[cfg(feature = "cuda")]
    m.add("CudaError", py.get_type::<CudaError>())?;
    m.add_class::<PyGlass>()?;
    m.add_class::<PyDesignFitness>()?;
    m.add_class::<PyCompoundPrism>()?;
    m.add_class::<PyDetectorArray>()?;
    m.add_class::<PyGaussianBeam>()?;
    m.add_class::<PySpectrometer>()?;
    Ok(())
}
