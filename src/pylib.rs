use compound_prism_spectrometer::*;
use core::convert::TryInto;
use ndarray::Array2;
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::{create_exception, gc::PyVisit, prelude::*, wrap_pyfunction, PyTraverseError};

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
#[pyo3(text_signature = "(name, coefficents)")]
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
    fn __call__(&self, w: f64) -> f64 {
        self.glass.calc_n(w)
    }

    #[getter]
    fn get_glass<'p>(&self, py: Python<'p>) -> impl IntoPy<PyObject> + 'p {
        PyArray1::from_slice(py, &self.glass.coefficents)
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{}", self))
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

fn pyglasses_to_glasses<const N: usize>(pyglasses: &[PyGlass]) -> [Glass<f64, 6>; N] {
    let pyglasses: &[PyGlass; N] = pyglasses.try_into().unwrap();
    let pyglasses: [&PyGlass; N] = pyglasses.each_ref();
    pyglasses.map(|pg| pg.glass)
}

#[pyclass(name = "DesignFitness", module = "compound_prism_designer")]
#[pyo3(text_signature = "(size, info, deviation)")]
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

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
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

#[pyclass(name = "Vector2D", module = "compound_prism_designer")]
#[pyo3(text_signature = "(x, y)")]
#[derive(Debug, Clone, Copy)]
struct PyVector2D {
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
}

#[pymethods]
impl PyVector2D {
    #[new]
    fn create(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.x, self.y)
    }

    fn __iter__(&self) -> impl IntoPy<PyObject> {
        (self.x, self.y)
    }
}

impl<F: FloatExt, const D: usize> LossyFrom<Vector<F, D>> for PyVector2D {
    fn lossy_from(v: Vector<F, D>) -> Self {
        Self {
            x: v[0].lossy_into(),
            y: v[1].lossy_into(),
        }
    }
}

impl<F: FloatExt, const D: usize> LossyFrom<PyVector2D> for Vector<F, D> {
    fn lossy_from(PyVector2D { x, y }: PyVector2D) -> Self {
        Vector::from_xy(F::lossy_from(x), F::lossy_from(y))
    }
}

impl<F: FloatExt> LossyFrom<compound_prism_spectrometer::Point<F>> for PyVector2D {
    fn lossy_from(
        compound_prism_spectrometer::Point { x, y }: compound_prism_spectrometer::Point<F>,
    ) -> Self {
        Self {
            x: x.lossy_into(),
            y: y.lossy_into(),
        }
    }
}

#[pyclass(name = "UnitVector2D", module = "compound_prism_designer")]
#[pyo3(text_signature = "(x, y)")]
#[derive(Debug, Clone, Copy)]
struct PyUnitVector2D {
    #[pyo3(get)]
    x: f64,
    #[pyo3(get)]
    y: f64,
}

#[pymethods]
impl PyUnitVector2D {
    #[new]
    fn create(x: f64, y: f64) -> PyResult<Self> {
        if let Some(UnitVector(Vector([x, y]))) = UnitVector::try_new(Vector::new([x, y])) {
            Ok(Self { x, y })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err("x^2 + y^2 != 1"))
        }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.x, self.y)
    }

    fn __iter__(&self) -> impl IntoPy<PyObject> {
        (self.x, self.y)
    }
}

impl<F: FloatExt, const D: usize> LossyFrom<UnitVector<F, D>> for PyUnitVector2D {
    fn lossy_from(UnitVector(v): UnitVector<F, D>) -> Self {
        Self {
            x: v[0].lossy_into(),
            y: v[1].lossy_into(),
        }
    }
}

impl<F: FloatExt, const D: usize> LossyFrom<PyUnitVector2D> for UnitVector<F, D> {
    fn lossy_from(PyUnitVector2D { x, y }: PyUnitVector2D) -> Self {
        UnitVector::new(Vector::from_xy(F::lossy_from(x), F::lossy_from(y)))
    }
}

#[derive(Debug, Clone, Copy, From, FromPyObject)]
enum DimensionedRadius {
    Dim2(f64),
    Dim3(f64, f64),
}

impl IntoPy<PyObject> for DimensionedRadius {
    fn into_py(self, py: pyo3::Python<'_>) -> PyObject {
        match self {
            DimensionedRadius::Dim2(v) => v.into_py(py),
            DimensionedRadius::Dim3(v1, v2) => (v1, v2).into_py(py),
        }
    }
}

#[pyclass(name = "Surface", module = "compound_prism_designer")]
#[pyo3(text_signature = "(lower_pt, upper_pt, angle, radius)")]
#[derive(Debug, Clone, Copy)]
pub struct PySurface {
    #[pyo3(get)]
    lower_pt: PyVector2D,
    #[pyo3(get)]
    upper_pt: PyVector2D,
    #[pyo3(get)]
    angle: f64,
    #[pyo3(get)]
    radius: Option<DimensionedRadius>,
}

#[pymethods]
impl PySurface {
    #[new]
    fn create(
        lower_pt: PyVector2D,
        upper_pt: PyVector2D,
        angle: f64,
        radius: Option<DimensionedRadius>,
    ) -> PyResult<Self> {
        Ok(Self {
            lower_pt,
            upper_pt,
            angle,
            radius,
        })
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.lower_pt, self.upper_pt, self.angle, self.radius)
    }
}

#[pyclass(
    name = "UniformWavelengthDistribution",
    module = "compound_prism_designer"
)]
#[pyo3(text_signature = "(bounds)")]
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
#[pyo3(text_signature = "(width, y_mean)")]
#[derive(Debug, Clone, Copy)]
struct PyGaussianBeam {
    gaussian_beam: GaussianBeam<f64>,
    #[pyo3(get)]
    width: f64,
    #[pyo3(get)]
    y_mean: f64,
}

#[pymethods]
impl PyGaussianBeam {
    #[new]
    fn create(width: f64, y_mean: f64) -> Self {
        let gaussian_beam = GaussianBeam { width, y_mean };
        PyGaussianBeam {
            gaussian_beam,
            width,
            y_mean,
        }
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> {
        (self.width, self.y_mean)
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

#[pyclass(name = "FiberBeam", module = "compound_prism_designer")]
#[pyo3(text_signature = "(core_radius, numerical_aperture, center_y)")]
#[derive(Debug, Clone, Copy)]
struct PyFiberBeam {
    fiber_beam: FiberBeam<f64>,
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

    #[getter]
    fn get_y_mean(&self) -> impl IntoPy<PyObject> {
        self.center_y
    }

    #[getter]
    fn get_width(&self) -> impl IntoPy<PyObject> {
        self.core_radius
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

#[derive(Debug, Clone, Copy, FromPyObject)]
enum BeamDistributions {
    Gaussian(PyGaussianBeam),
    Fiber(PyFiberBeam),
}

macro_rules! map_beam_distributions {
    ($beam_distribution:expr => |$distribution:ident $(, $beam_ty:ident)?| $body:expr ) => {
        match $beam_distribution {
            BeamDistributions::Gaussian(PyGaussianBeam {
                gaussian_beam: $distribution,
                ..
            }) => {$(type $beam_ty<T> = GaussianBeam<T>;)? $body},
            BeamDistributions::Fiber(PyFiberBeam {
                fiber_beam: $distribution,
                ..
            }) => {$(type $beam_ty<T> = FiberBeam<T>;)? $body},
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
            #[derive(Debug, Clone, Copy, From, WrappedFrom)]
            #[wrapped_from(trait = "LossyFrom", function = "lossy_from")]
            enum SizedCompoundPrism<F: FloatExt, S0: Surface<F, D>, SI: Surface<F, D>, SN: Surface<F, D>, const D: usize> {
                $( [<CompoundPrism $n>](CompoundPrism<F, S0, SI, SN, $n, D>) ),*
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

impl<S0, SN, const D: usize> SizedCompoundPrism<f64, S0, Plane<f64, D>, SN, D>
where
    S0: Copy + Surface<f64, D> + FromParametrizedHyperPlane<f64, D> + Drawable<f64>,
    SN: Copy + Surface<f64, D> + FromParametrizedHyperPlane<f64, D> + Drawable<f64>,
    Plane<f64, D>: Surface<f64, D>,
{
    fn new(
        glasses: &[PyGlass],
        angles: &[f64],
        lengths: &[f64],
        p0: S0::Parametrization,
        pn: SN::Parametrization,
        height: f64,
        width: f64,
        ar_coated: bool,
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
        let sep_lengths = lengths;
        let (last_angle, angles_rest) = angles_rest.split_last().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("`angles` must have at least 2 elements")
        })?;
        Ok(
            create_sized_compound_prism! { glasses.len() => CompoundPrism::new(
                glasses[0].glass,
                pyglasses_to_glasses(&glasses[1..]),
                *first_angle,
                angles_rest.try_into().unwrap(),
                *last_angle,
                sep_lengths[0],
                sep_lengths[1..].try_into().unwrap(),
                p0,
                pn,
                height,
                width,
                ar_coated,
            ) },
        )
    }

    fn exit_ray(&self, y: f64, wavelength: f64) -> PyResult<(PyVector2D, PyUnitVector2D)> {
        let ray = Ray::new_from_start(y);
        match map_sized_compound_prism!(self => |c| ray.propagate_internal(c, wavelength)) {
            Ok(r) => Ok((r.origin.lossy_into(), r.direction.lossy_into())),
            Err(e) => Err(map_ray_trace_err(e)),
        }
    }

    fn final_midpt(&self) -> PyVector2D {
        map_sized_compound_prism!(self => |c| c.final_midpt().lossy_into())
    }

    fn polygons<'p>(&self, py: Python<'p>) -> PyResult<Vec<&'p PyAny>> {
        let matplotlib = py.import("matplotlib")?;
        let path_mod = matplotlib.getattr("path")?;
        let path_cls = path_mod.getattr("Path")?;
        let move_to = path_cls.getattr("MOVETO")?;
        let line_to = path_cls.getattr("LINETO")?;
        let curve_4 = path_cls.getattr("CURVE4")?;
        let close_p = path_cls.getattr("CLOSEPOLY")?;

        let point2array = |Point { x, y }: Point<f64>| (x, y);
        #[allow(non_snake_case)]
        let path2Path = |p: Path<f64>, start_code| match p {
            Path::Line { a, b } => (
                vec![point2array(a), point2array(b)],
                vec![start_code, line_to],
            ),
            Path::Arc {
                a,
                b,
                midpt,
                radius,
            } => {
                let curvature = 1.0 / radius;
                let [a, c0, c1, b] = arc_as_cubic_bézier(a, midpt, b, curvature);
                (
                    vec![
                        point2array(a),
                        point2array(c0),
                        point2array(c1),
                        point2array(b),
                    ],
                    vec![start_code, curve_4, curve_4, curve_4],
                )
            }
        };

        map_sized_compound_prism!(self => |c| {
            let (polys, last_poly) = c.polygons();
            polys.into_iter()
                .chain(core::iter::once(last_poly))
                .map(|Polygon([path_l, path_r])| {
                    let (vert_l, codes_l) = path2Path(path_l, move_to);
                    let (vert_r, codes_r) = path2Path(path_r, line_to);
                    let verts: Vec<_> = vert_l.into_iter().chain(vert_r).chain(core::iter::once(point2array(path_l.start()))).collect();
                    let codes: Vec<_> = codes_l.into_iter().chain(codes_r).chain(core::iter::once(close_p)).collect();
                    path_cls.call1((verts, codes))
                })
                .collect::<PyResult<Vec<_>>>()
        })
    }
}

#[derive(Debug, Clone, Copy, From, WrappedFrom)]
#[wrapped_from(trait = "LossyFrom", function = "lossy_from")]
enum DimensionedSizedCompoundPrism<F: FloatExt> {
    Dim2(SizedCompoundPrism<F, Plane<F, 2>, Plane<F, 2>, CurvedPlane<F, 2>, 2>),
    Dim3(SizedCompoundPrism<F, ToricLens<F, 3>, Plane<F, 3>, ToricLens<F, 3>, 3>),
}

macro_rules! map_dimensioned_sized_compound_prism {
    ($dimensioned_sized_compound_prism:expr => |$compound_prism:ident $(, $dim:ident)?| $body:expr ) => {
        match $dimensioned_sized_compound_prism {
            DimensionedSizedCompoundPrism::Dim2($compound_prism) => { $(const $dim: usize = 2;)? map_sized_compound_prism!($compound_prism => |$compound_prism| $body)},
            DimensionedSizedCompoundPrism::Dim3($compound_prism) => { $(const $dim: usize = 3;)? map_sized_compound_prism!($compound_prism => |$compound_prism| $body)},
        }
    };
}

#[derive(Debug, Clone, Copy, From, FromPyObject)]
enum DimensionedCurvature {
    Dim2(f64),
    Dim3((f64, f64), (f64, f64)),
}

impl IntoPy<PyObject> for DimensionedCurvature {
    fn into_py(self, py: pyo3::Python<'_>) -> PyObject {
        match self {
            DimensionedCurvature::Dim2(v) => v.into_py(py),
            DimensionedCurvature::Dim3(i, f) => (i, f).into_py(py),
        }
    }
}

#[pyclass(name = "CompoundPrism", module = "compound_prism_designer")]
#[pyo3(text_signature = "(glasses, angles, lengths, curvature, height, width, ar_coated)")]
#[derive(Debug, Clone)]
struct PyCompoundPrism {
    compound_prism: DimensionedSizedCompoundPrism<f64>,
    #[pyo3(get)]
    glasses: Vec<PyGlass>,
    #[pyo3(get)]
    angles: Vec<f64>,
    #[pyo3(get)]
    lengths: Vec<f64>,
    curvature: DimensionedCurvature,
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
        glasses: Vec<PyGlass>,
        angles: Vec<f64>,
        lengths: Vec<f64>,
        curvature: DimensionedCurvature,
        height: f64,
        width: f64,
        ar_coated: bool,
    ) -> PyResult<Self> {
        let compound_prism = match curvature {
            DimensionedCurvature::Dim2(curvature) => {
                let initial_parametrization = PlaneParametrization { height, width };
                let final_parametrization = CurvedPlaneParametrization {
                    signed_normalized_curvature: curvature,
                    height,
                };
                let compound_prism = SizedCompoundPrism::new(
                    &glasses,
                    &angles,
                    &lengths,
                    initial_parametrization,
                    final_parametrization,
                    height,
                    width,
                    ar_coated,
                )?;
                DimensionedSizedCompoundPrism::Dim2(compound_prism)
            }
            DimensionedCurvature::Dim3(initial_curvature, final_curvature) => {
                let initial_parametrization = ToricLensParametrization {
                    signed_normalized_poloidal_curvature: initial_curvature.0,
                    normalized_toroidal_curvature: initial_curvature.1,
                    height,
                    width,
                };
                let final_parametrization = ToricLensParametrization {
                    signed_normalized_poloidal_curvature: final_curvature.0,
                    normalized_toroidal_curvature: final_curvature.1,
                    height,
                    width,
                };
                let compound_prism = SizedCompoundPrism::new(
                    &glasses,
                    &angles,
                    &lengths,
                    initial_parametrization,
                    final_parametrization,
                    height,
                    width,
                    ar_coated,
                )?;
                DimensionedSizedCompoundPrism::Dim3(compound_prism)
            }
        };
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

    fn __getnewargs__(&self) -> PyResult<impl IntoPy<PyObject> + '_> {
        Ok((
            self.glasses.clone(),
            self.angles.clone(),
            self.lengths.clone(),
            self.curvature,
            self.height,
            self.width,
            self.ar_coated,
        ))
    }

    fn exit_ray<'p>(&self, y: f64, wavelength: f64) -> PyResult<(PyVector2D, PyUnitVector2D)> {
        match self.compound_prism {
            DimensionedSizedCompoundPrism::Dim2(c) => c.exit_ray(y, wavelength),
            DimensionedSizedCompoundPrism::Dim3(c) => c.exit_ray(y, wavelength),
        }
    }

    fn polygons<'p>(&self, py: Python<'p>) -> PyResult<Vec<&'p PyAny>> {
        match self.compound_prism {
            DimensionedSizedCompoundPrism::Dim2(c) => c.polygons(py),
            DimensionedSizedCompoundPrism::Dim3(c) => c.polygons(py),
        }
    }

    fn final_midpt_and_normal(&self) -> (PyVector2D, PyUnitVector2D) {
        let midpt = match self.compound_prism {
            DimensionedSizedCompoundPrism::Dim2(c) => c.final_midpt(),
            DimensionedSizedCompoundPrism::Dim3(c) => c.final_midpt(),
        };
        (
            midpt,
            UnitVector(Vector::<f64, 2>::angled_xy(*self.angles.last().unwrap())).lossy_into(),
        )
    }

    fn surfaces(&self) -> Vec<PySurface> {
        map_dimensioned_sized_compound_prism!(self.compound_prism => |c| {
            let (s_0, s_i, s_n) = c.surfaces();
            core::iter::once(s_0).chain(s_i).chain(core::iter::once(s_n)).zip(self.angles.as_slice()).map(|((start, end, radius), &angle)| {
                PySurface {
                    lower_pt: start.lossy_into(),
                    upper_pt: end.lossy_into(),
                    angle,
                    radius: radius.map(|r| DimensionedRadius::Dim2(r)),
                }
            }).collect()
        })
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

#[pyfunction]
fn position_detector_array(
    length: f64,
    angle: f64,
    compound_prism: &PyCompoundPrism,
    wavelengths: WavelengthDistributions,
    beam: BeamDistributions,
    acceptance: f64,
) -> PyResult<((f64, f64), bool)> {
    let (pos, flipped) = map_dimensioned_sized_compound_prism!(compound_prism.compound_prism => |prism|
        map_beam_distributions!(beam => |beam|
            map_wavelength_distributions!(wavelengths => |ws|
                detector_array_positioning(
                    prism,
                    length,
                    angle,
                    ws,
                    &beam,
                    acceptance,
                ).map(|(v, f)| ([v[0], v[1]], f)).map_err(map_ray_trace_err)?
            )
        )
    );
    Ok(((pos[0], pos[1]), flipped))
}

#[pyclass(name = "DetectorArray", module = "compound_prism_designer")]
#[pyo3(
    text_signature = "(bin_count, bin_size, linear_slope, linear_intercept, length, max_incident_angle, angle, position, flipped)"
)]
#[derive(Debug, Clone, Copy)]
struct PyDetectorArray {
    detector_array: LinearDetectorArray<f64, 3>,
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
    #[pyo3(get)]
    position: (f64, f64),
    #[pyo3(get)]
    flipped: bool,
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
        position: (f64, f64),
        flipped: bool,
    ) -> PyResult<Self> {
        let detector_array = LinearDetectorArray::new(
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
            max_incident_angle.cos(),
            angle,
            length,
            Vector::from_xy(position.0, position.1),
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
            position,
            flipped,
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
            self.position,
            self.flipped,
        )
    }

    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

impl<'p, const D: usize> From<&'p PyDetectorArray> for LinearDetectorArray<f64, D> {
    fn from(pda: &'p PyDetectorArray) -> Self {
        LinearDetectorArray::new(
            pda.bin_count,
            pda.bin_size,
            pda.linear_slope,
            pda.linear_intercept,
            pda.max_incident_angle.cos(),
            pda.angle,
            pda.length,
            Vector::from_xy(pda.position.0, pda.position.1),
            pda.flipped,
        )
    }
}

use crate::SpectrometerFitness;

trait PySpectrometerFitness<T: FloatExt, const D: usize>:
    SpectrometerFitness<T, D> + Send + 'static + core::fmt::Debug
{
}

impl<T: FloatExt, S, const D: usize> PySpectrometerFitness<T, D> for S where
    S: SpectrometerFitness<T, D> + Send + 'static + core::fmt::Debug
{
}

/// Compound Prism Spectrometer specification
///
/// Args:
///     compound_prism (CompoundPrism): compound prism specification
///     detector_array (DetectorArray): linear detector array specification
///     wavelengths (UniformWavelengthDistribution): input wavelength distribution specification
///     gaussian_beam (GaussianBeam): input gaussian beam specification
#[pyclass(name = "Spectrometer", module = "compound_prism_designer")]
#[pyo3(text_signature = "(compound_prism, detector_array, wavelengths, beam)")]
#[derive(Debug)]
struct PySpectrometer {
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
    beam: BeamDistributions,
}

#[pymethods]
impl PySpectrometer {
    #[new]
    fn create(
        compound_prism: Py<PyCompoundPrism>,
        detector_array: Py<PyDetectorArray>,
        wavelengths: WavelengthDistributions,
        beam: BeamDistributions,
    ) -> PyResult<Self> {
        Ok(PySpectrometer {
            compound_prism,
            detector_array,
            wavelengths,
            beam,
        })
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> + '_ {
        (
            &self.compound_prism,
            &self.detector_array,
            self.wavelengths,
            self.beam,
        )
    }

    /// Computes the spectrometer fitness using on the cpu
    #[pyo3(text_signature = "($self, /, *, max_n = 16_384, max_m = 16_384)")]
    #[args("*", max_n = 16_384, max_m = 16_384)]
    fn cpu_fitness(&self, py: Python, max_n: usize, max_m: usize) -> PyResult<PyDesignFitness> {
        let compound_prism = self.compound_prism.as_ref(py).try_borrow()?.compound_prism;
        let py_detarr = *self.detector_array.as_ref(py).try_borrow()?;
        Ok(py.allow_threads(|| {
            map_dimensioned_sized_compound_prism!(compound_prism => |compound_prism, D|
                map_wavelength_distributions!(self.wavelengths => |wavelengths|
                    map_beam_distributions!(self.beam => |beam| {
                        let detector = LinearDetectorArray::<f64, D>::from(&py_detarr);
                        let s = Spectrometer { wavelengths, beam, detector, compound_prism };
                        crate::fitness(&s, max_n, max_m).into()
                    }
                    )
                )
            )
        }))
    }

    #[pyo3(text_signature = "($self, /, wavelengths, *, max_m = 16_384)")]
    #[args(wavelengths, "*", max_m = 16_384)]
    pub fn transmission_probability<'p>(
        &self,
        wavelengths: &PyArray1<f64>,
        py: Python<'p>,
        max_m: usize,
    ) -> PyResult<&'p PyArray2<f64>> {
        let compound_prism = self.compound_prism.as_ref(py).try_borrow()?.compound_prism;
        let py_detarr = *self.detector_array.as_ref(py).try_borrow()?;
        map_dimensioned_sized_compound_prism!(compound_prism => |compound_prism, D|
            map_wavelength_distributions!(self.wavelengths => |w|
                map_beam_distributions!(self.beam => |beam| {
                    let detector = LinearDetectorArray::<f64, D>::from(&py_detarr);
                    let spec = Spectrometer { wavelengths: w, beam, detector, compound_prism };
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
            )
        )
    }

    #[pyo3(text_signature = "($self, /, wavelength, inital_y)")]
    pub fn ray_trace<'p>(
        &self,
        wavelength: f64,
        inital_y: f64,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray2<f64>> {
        let compound_prism = self.compound_prism.as_ref(py).try_borrow()?.compound_prism;
        let py_detarr = *self.detector_array.as_ref(py).try_borrow()?;
        map_dimensioned_sized_compound_prism!(compound_prism => |compound_prism, D|
            map_wavelength_distributions!(self.wavelengths => |wavelengths|
                map_beam_distributions!(self.beam => |beam| {
                    let detector = LinearDetectorArray::<f64, D>::from(&py_detarr);
                    let spec = Spectrometer { wavelengths, beam, detector, compound_prism,};
                    Ok(Array2::from(
                        GenericSpectrometer::propagation_path(&spec, Ray::new_from_start(inital_y), wavelength).map(|GeometricRay { origin: Vector([x, y, ..]), direction: UnitVector(Vector([ux, uy, ..])) } | [x, y, ux, uy])
                            .collect::<Vec<_>>()
                        )
                        .to_pyarray(py)
                    )
                })
            )
        )
    }

    /// Computes the spectrometer fitness using on the gpu with float32
    #[cfg(feature = "cuda")]
    #[pyo3(text_signature = "($self, /, seeds, *, max_n = 256, nwarp = 2, max_eval = 16_384)")]
    #[args("*", max_n = 256, nwarp = 2, max_eval = 16_384)]
    fn gpu_fitness(
        &self,
        py: Python,
        seeds: &PyArray1<f64>,
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> PyResult<Option<PyDesignFitness>> {
        let compound_prism = self.compound_prism.as_ref(py).try_borrow()?.compound_prism;
        let compound_prism: DimensionedSizedCompoundPrism<f32> = compound_prism.lossy_into();
        let py_detarr = *self.detector_array.as_ref(py).try_borrow()?;
        let seeds = seeds.readonly();
        let seeds = seeds.as_slice().unwrap();
        map_wavelength_distributions!(self.wavelengths => |w| {
            match compound_prism {
                DimensionedSizedCompoundPrism::Dim2(compound_prism) => map_sized_compound_prism!(compound_prism => |compound_prism| {
                    const D: usize = 2;
                    if let BeamDistributions::Gaussian(PyGaussianBeam { gaussian_beam: b, .. }) = self.beam {
                        type B<T> = GaussianBeam<T>;

                        let detector: LinearDetectorArray<f32, D> = LinearDetectorArray::<f64, D>::from(&py_detarr).lossy_into();
                        let beam: B<f32> = LossyFrom::lossy_from(b);
                        let spec = Spectrometer { wavelengths: w.lossy_into(), beam, detector, compound_prism };
                        let fit = py.allow_threads(|| crate::cuda_fitness(&spec, seeds, max_n, nwarp, max_eval)).map_err(map_cuda_err)?;
                        Ok(fit.map(Into::into))
                    } else {
                        Err(pyo3::exceptions::PyNotImplementedError::new_err("No kernel compiled for this Spectrometer"))
                    }
            }),
                DimensionedSizedCompoundPrism::Dim3(compound_prism) => map_sized_compound_prism!(compound_prism => |compound_prism| {
                    const D: usize = 3;
                    if let BeamDistributions::Fiber(PyFiberBeam { fiber_beam: b, .. }) = self.beam {
                        type B<T> = FiberBeam<T>;

                        let detector: LinearDetectorArray<f32, D> = LinearDetectorArray::<f64, D>::from(&py_detarr).lossy_into();
                        let beam: B<f32> = LossyFrom::lossy_from(b);
                        let spec = Spectrometer { wavelengths: w.lossy_into(), beam, detector, compound_prism };
                        let fit = py.allow_threads(|| crate::cuda_fitness(&spec, seeds, max_n, nwarp, max_eval)).map_err(map_cuda_err)?;
                        Ok(fit.map(Into::into))
                    } else {
                        Err(pyo3::exceptions::PyNotImplementedError::new_err("No kernel compiled for this Spectrometer"))
                    }
            }),
            }
        })
    }

    #[getter]
    fn get_position(&self, py: Python) -> PyResult<(f64, f64)> {
        let Vector([x, y, _]) = self
            .detector_array
            .as_ref(py)
            .try_borrow()?
            .detector_array
            .position;
        Ok((x, y))
    }

    #[getter]
    fn get_direction(&self, py: Python) -> PyResult<(f64, f64)> {
        let UnitVector(Vector([x, y, _])) = self
            .detector_array
            .as_ref(py)
            .try_borrow()?
            .detector_array
            .direction;
        Ok((x, y))
    }
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        visit.call(&self.compound_prism)?;
        visit.call(&self.detector_array)?;
        Ok(())
    }

    fn __clear__(&mut self) {}
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
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
    m.add_class::<PyVector2D>()?;
    m.add_class::<PyUnitVector2D>()?;
    m.add_class::<PySurface>()?;
    m.add_class::<PyUniformWavelengthDistribution>()?;
    m.add_class::<PyGaussianBeam>()?;
    m.add_class::<PyFiberBeam>()?;
    m.add_class::<PyCompoundPrism>()?;
    m.add_function(wrap_pyfunction!(position_detector_array, m)?)?;
    m.add_class::<PyDetectorArray>()?;
    m.add_class::<PySpectrometer>()?;
    Ok(())
}
