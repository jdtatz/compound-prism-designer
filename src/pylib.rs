use compound_prism_spectrometer::*;
use core::convert::TryInto;
use ndarray::{array, Array2};
use numpy::{PyArray1, PyArray2, ToPyArray};
use pyo3::{
    create_exception,
    gc::{PyGCProtocol, PyVisit},
    prelude::*,
    wrap_pyfunction, PyObjectProtocol, PyTraverseError,
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
}

#[pyproto]
impl PyObjectProtocol for PyDesignFitness {
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
}

#[pyproto]
impl PyObjectProtocol for PyGaussianBeam {
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
}

#[pyproto]
impl PyObjectProtocol for PyFiberBeam {
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

macro_rules! define_sized_spectrometer {
    ([$($n:literal),*]) => {
        paste::paste! {
            #[derive(Debug, Clone, Copy, From, WrappedFrom)]
            #[wrapped_from(trait = "LossyFrom", function = "lossy_from")]
            enum SizedSpectrometer<
                F: FloatExt,
                W: Distribution<F, Output = F>,
                B: Beam<F, D>,
                S0: Surface<F, D>,
                SI: Surface<F, D>,
                SN: Surface<F, D>,
                const D: usize,
                > {
                $( [<Spectrometer $n>](Spectrometer<F, W, B, S0, SI, SN, $n, D>) ),*
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

impl<S0, SN, const D: usize> SizedCompoundPrism<f64, S0, Plane<f64, D>, SN, D>
where
    S0: Copy + Surface<f64, D> + FromParametrizedHyperPlane<f64, D> + Drawable<f64>,
    SN: Copy + Surface<f64, D> + FromParametrizedHyperPlane<f64, D> + Drawable<f64>,
    Plane<f64, D>: Surface<f64, D>,
{
    fn new(
        py: Python,
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

    fn exit_ray<'p>(
        &self,
        y: f64,
        wavelength: f64,
        py: Python<'p>,
    ) -> PyResult<(PyVector2D, PyUnitVector2D)> {
        let ray = Ray::new_from_start(y);
        match map_sized_compound_prism!(self => |c| ray.propagate_internal(c, wavelength)) {
            Ok(r) => Ok((r.origin.lossy_into(), r.direction.lossy_into())),
            Err(e) => Err(map_ray_trace_err(e)),
        }
    }

    fn polygons<'p>(&self, py: Python<'p>) -> PyResult<Vec<&'p PyAny>> {
        let matplotlib = py.import("matplotlib")?;
        let path_mod = matplotlib.getattr("path")?;
        let transforms_mod = matplotlib.getattr("transforms")?;
        let path_cls = path_mod.getattr("Path")?;
        let affine_cls = transforms_mod.getattr("Affine2D")?;
        let move_to = path_cls.getattr("MOVETO")?;
        let line_to = path_cls.getattr("LINETO")?;
        let curve_4 = path_cls.getattr("CURVE4")?;
        let close_p = path_cls.getattr("CLOSEPOLY")?;

        let point2array = |Point { x, y }: Point<f64>| (x, y);
        // let createArc = |a: Point<f64>, b: Point<f64>, midpt: Point<f64>, center: Point<f64>, radius: f64| {
        //     let mut t1 = f64::to_degrees(f64::atan2(b.y - center.y, b.x - center.x));
        //     let mut t2 = f64::to_degrees(f64::atan2(a.y - center.y, a.x - center.x));
        //     let is_rightward = midpt.x >= center.x;
        //     if !is_rightward {
        //         core::mem::swap(&mut t1, &mut t2);
        //     }
        //     let arc = path_cls.call_method1("arc", (t1, t2))?;
        //     let transform = affine_cls
        //         .call0()?
        //         .call_method1("scale", (radius,))?
        //         .call_method1("translate", (center.x, center.y))?;
        //     arc.call_method1("transformed", (transform,))
        // };

        // let path2Path = |p: Path<f64>| match p {
        //     Path::Line { a, b } => path_cls.call1((array![[a.x, a.y], [b.x, b.y]].to_pyarray(py),)),
        //     Path::Arc {
        //         a,
        //         b,
        //         midpt,
        //         center,
        //         radius,
        //     } => {
        //         let mut t1 = f64::to_degrees(f64::atan2(b.y - center.y, b.x - center.x));
        //         let mut t2 = f64::to_degrees(f64::atan2(a.y - center.y, a.x - center.x));
        //         let is_rightward = midpt.x >= center.x;
        //         if !is_rightward {
        //             core::mem::swap(&mut t1, &mut t2);
        //         }
        //         let arc = path_cls.call_method1("arc", (t1, t2))?;
        //         let transform = affine_cls
        //             .call0()?
        //             .call_method1("scale", (radius,))?
        //             .call_method1("translate", (center.x, center.y))?;
        //         arc.call_method1("transformed", (transform,))
        //     }
        // };

        let path2Path = |p: Path<f64>, start_code| match p {
            Path::Line { a, b } => (
                vec![point2array(a), point2array(b)],
                vec![start_code, line_to],
            ),
            Path::Arc {
                a,
                b,
                midpt,
                // center,
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
                // let [[a, c0, c1, midpt], [_, c2, c3, b]] = arc_as_2_cubic_béziers(a, midpt, b, curvature);
                // (vec![point2array(a), point2array(c0), point2array(c1), point2array(midpt), point2array(c2), point2array(c3), point2array(b)], vec![start_code, curve_4, curve_4, curve_4, curve_4, curve_4, curve_4])
            }
        };

        map_sized_compound_prism!(self => |c| {
            let (polys, last_poly) = c.polygons();
            core::array::IntoIter::new(polys)
                .chain(core::iter::once(last_poly))
                .map(|Polygon([pathL, pathR])| {
                    let (vertL, codesL) = path2Path(pathL, move_to);
                    let (vertR, codesR) = path2Path(pathR, line_to);
                    let verts: Vec<_> = vertL.into_iter().chain(vertR).chain(core::iter::once(point2array(pathL.start()))).collect();
                    let codes: Vec<_> = codesL.into_iter().chain(codesR).chain(core::iter::once(close_p)).collect();
                    path_cls.call1((verts, codes))

                    // let pL = path2Path(pathL)?;
                    // let pR = path2Path(pathR)?;
                    // match (pathL, pathR) {
                    //     (Path::Line { a, b }, Path::Line { a: c, b: d }) => path_cls.call1(((point2array(a), point2array(b), point2array(c), point2array(d)),)),
                    //     (Path::Line { a, b }, Path::Arc { a: c, b: d, center, radius, midpt }) => path_cls.call_method1("make_compound_path", (createArc(c, d, midpt, center, radius)?, path_cls.call1(((point2array(c), point2array(b), point2array(a), point2array(d)),))?)),
                    //     (Path::Arc { a, b, center, radius, midpt }, Path::Line { a: c, b: d }) => path_cls.call_method1("make_compound_path", (createArc(b, a, midpt, center, radius)?, path_cls.call1(((point2array(b), point2array(c), point2array(d), point2array(a)),))?)),
                    //     (Path::Arc { a, b, center: cL, radius: rL, midpt: mL }, Path::Arc { a: c, b: d, center: cR, radius: rR, midpt: mR }) => path_cls.call_method1("make_compound_path", (createArc(b, a, mL, cL, rL)?, createArc(c, d, mR, cR, rR)?)),
                    // }
                    // let mid = (point2array(pathL.end()), point2array(pathR.start()));
                    // let fin = (point2array(pathR.end()), point2array(pathL.start()));
                    // let mid = path_cls.call1((mid,))?;
                    // let fin = path_cls.call1((fin,))?;
                    // path_cls.call_method1("make_compound_path", (pL, mid, pR, fin))
                })
                .collect::<PyResult<Vec<_>>>()
        })
    }
}

// #[pyclass(name = "CompoundPrism", gc, module = "compound_prism_designer")]
// #[pyo3(text_signature = "(glasses, angles, lengths, curvature, height, width, ar_coated)")]
// #[derive(Debug, Clone)]
// struct PyCompoundPrism2D {
//     compound_prism: SizedCompoundPrism<f64, Plane<f64, 2>, Plane<f64, 2>, CurvedPlane<f64, 2>, 2>,
//     #[pyo3(get)]
//     glasses: Vec<Py<PyGlass>>,
//     #[pyo3(get)]
//     angles: Vec<f64>,
//     #[pyo3(get)]
//     lengths: Vec<f64>,
//     #[pyo3(get)]
//     curvature: f64,
//     #[pyo3(get)]
//     height: f64,
//     #[pyo3(get)]
//     width: f64,
//     #[pyo3(get)]
//     ar_coated: bool,
// }

// #[pymethods]
// impl PyCompoundPrism2D {
//     #[new]
//     fn create(
//         glasses: Vec<Py<PyGlass>>,
//         angles: Vec<f64>,
//         lengths: Vec<f64>,
//         curvature: f64,
//         height: f64,
//         width: f64,
//         ar_coated: bool,
//         py: Python,
//     ) -> PyResult<Self> {
//         let initial_plane_parametrization = PlaneParametrization { height, width };
//         let lens_parametrization = CurvedPlaneParametrization {
//             signed_normalized_curvature: curvature,
//             height,
//         };
//         let compound_prism = SizedCompoundPrism::new(
//             py,
//             &glasses,
//             &angles,
//             &lengths,
//             initial_plane_parametrization,
//             lens_parametrization,
//             height,
//             width,
//             ar_coated,
//         )?;
//         Ok(PyCompoundPrism2D {
//             compound_prism,
//             glasses,
//             angles,
//             lengths,
//             curvature,
//             height,
//             width,
//             ar_coated,
//         })
//     }

//     fn __getnewargs__(&self, py: Python) -> PyResult<impl IntoPy<PyObject> + '_> {
//         Ok((
//             self.glasses
//                 .iter()
//                 .map(|p| p.as_ref(py).try_borrow().map(|v| v.clone()))
//                 .collect::<Result<Vec<PyGlass>, _>>()?,
//             self.angles.clone(),
//             self.lengths.clone(),
//             self.curvature,
//             self.height,
//             self.width,
//             self.ar_coated,
//         ))
//     }

//    fn polygons<'p>(&self, py: Python<'p>) -> PyResult<Vec<&'p PyAny>> {
//        self.compound_prism.polygons(py)
//    }
// }

// #[pyproto]
// impl PyGCProtocol for PyCompoundPrism2D {
//     fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
//         for obj in self.glasses.iter() {
//             visit.call(obj)?
//         }
//         Ok(())
//     }

//     fn __clear__(&mut self) {}
// }

// #[pyproto]
// impl PyObjectProtocol for PyCompoundPrism2D {
//     fn __repr__(&self) -> PyResult<String> {
//         Ok(format!("{:#?}", self))
//     }
// }

#[pyclass(name = "CompoundPrism", module = "compound_prism_designer")]
#[pyo3(
    text_signature = "(glasses, angles, lengths, initial_curvature, final_curvature, height, width, ar_coated)"
)]
#[derive(Debug, Clone)]
struct PyCompoundPrism {
    compound_prism: SizedCompoundPrism<f64, ToricLens<f64, 3>, Plane<f64, 3>, ToricLens<f64, 3>, 3>,
    #[pyo3(get)]
    glasses: Vec<PyGlass>,
    #[pyo3(get)]
    angles: Vec<f64>,
    #[pyo3(get)]
    lengths: Vec<f64>,
    #[pyo3(get)]
    initial_curvature: (f64, f64),
    #[pyo3(get)]
    final_curvature: (f64, f64),
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
        initial_curvature: (f64, f64),
        final_curvature: (f64, f64),
        height: f64,
        width: f64,
        ar_coated: bool,
        py: Python,
    ) -> PyResult<Self> {
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
            py,
            &glasses,
            &angles,
            &lengths,
            initial_parametrization,
            final_parametrization,
            height,
            width,
            ar_coated,
        )?;
        Ok(PyCompoundPrism {
            compound_prism,
            glasses,
            angles,
            lengths,
            initial_curvature,
            final_curvature,
            height,
            width,
            ar_coated,
        })
    }

    fn __getnewargs__(&self, py: Python) -> PyResult<impl IntoPy<PyObject> + '_> {
        Ok((
            self.glasses.clone(),
            self.angles.clone(),
            self.lengths.clone(),
            self.initial_curvature,
            self.final_curvature,
            self.height,
            self.width,
            self.ar_coated,
        ))
    }
    fn exit_ray<'p>(
        &self,
        y: f64,
        wavelength: f64,
        py: Python<'p>,
    ) -> PyResult<(PyVector2D, PyUnitVector2D)> {
        self.compound_prism.exit_ray(y, wavelength, py)
    }

    fn polygons<'p>(&self, py: Python<'p>) -> PyResult<Vec<&'p PyAny>> {
        self.compound_prism.polygons(py)
    }
}

#[pyproto]
impl PyObjectProtocol for PyCompoundPrism {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
    }
}

#[pyfunction]
fn position_detector_array(
    py: Python,
    length: f64,
    angle: f64,
    compound_prism: &PyCompoundPrism,
    wavelengths: WavelengthDistributions,
    beam: BeamDistributions,
) -> PyResult<((f64, f64), bool)> {
    let (pos, flipped) = map_sized_compound_prism!(compound_prism.compound_prism => |prism|
        map_beam_distributions!(beam => |beam|
            map_wavelength_distributions!(wavelengths => |ws|
                detector_array_positioning(
                    prism,
                    length,
                    angle,
                    ws,
                    &beam,
                ).map_err(map_ray_trace_err)?
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
}

#[pyproto]
impl PyObjectProtocol for PyDetectorArray {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:#?}", self))
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
#[pyo3(text_signature = "(compound_prism, detector_array, wavelengths, beam)")]
#[derive(Debug, Clone)]
struct PySpectrometer {
    spectrometer: SizedSpectrometer<
        f64,
        UniformDistribution<f64>,
        FiberBeam<f64>,
        ToricLens<f64, 3>,
        Plane<f64, 3>,
        ToricLens<f64, 3>,
        3,
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
    fiber_beam: Py<PyFiberBeam>,
}

#[pymethods]
impl PySpectrometer {
    #[new]
    fn create(
        compound_prism: Py<PyCompoundPrism>,
        detector_array: Py<PyDetectorArray>,
        wavelengths: WavelengthDistributions,
        fiber_beam: Py<PyFiberBeam>,
        py: Python,
    ) -> PyResult<Self> {
        let WavelengthDistributions::Uniform(PyUniformWavelengthDistribution { distribution: w }) =
            wavelengths;
        let gb = fiber_beam.as_ref(py).try_borrow()?.fiber_beam;
        let da = detector_array.as_ref(py).try_borrow()?.detector_array;
        let scp = compound_prism.as_ref(py).try_borrow()?.compound_prism;
        let spectrometer = create_sized_spectrometer!(scp; w; gb; da);
        Ok(PySpectrometer {
            compound_prism,
            detector_array,
            wavelengths,
            fiber_beam,
            spectrometer,
        })
    }

    fn __getnewargs__(&self) -> impl IntoPy<PyObject> + '_ {
        (
            &self.compound_prism,
            &self.detector_array,
            self.wavelengths,
            &self.fiber_beam,
        )
    }

    /// Computes the spectrometer fitness using on the cpu
    #[pyo3(text_signature = "($self, /, *, max_n = 16_384, max_m = 16_384)")]
    #[args("*", max_n = 16_384, max_m = 16_384)]
    fn cpu_fitness(&self, py: Python, max_n: usize, max_m: usize) -> PyDesignFitness {
        py.allow_threads(|| map_sized_spectrometer!(self.spectrometer => |s| crate::fitness(&s, max_n, max_m).into()))
    }

    #[pyo3(text_signature = "($self, /, wavelengths, *, max_m = 16_384)")]
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

    #[pyo3(text_signature = "($self, /, wavelength, inital_y)")]
    pub fn ray_trace<'p>(
        &self,
        wavelength: f64,
        inital_y: f64,
        py: Python<'p>,
    ) -> PyResult<&'p PyArray2<f64>> {
        let spec = &self.spectrometer;
        Ok(Array2::from(
            map_sized_spectrometer!(spec => |s| s.propagation_path(Ray::new_from_start(inital_y), wavelength).map(|GeometricRay { origin: Vector([x, y, _]), direction: UnitVector(Vector([ux, uy, _])) } | [x, y, ux, uy])
            .collect::<Vec<_>>()),
        )
        .to_pyarray(py))
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
        let seeds = seeds.readonly();
        let seeds = seeds.as_slice().unwrap();
        let spec: SizedSpectrometer<f32, _, _, _, _, _, 3> =
            LossyFrom::lossy_from(self.spectrometer);
        let fit = py.allow_threads(||
            map_sized_spectrometer!(spec => |spec| crate::cuda_fitness(&spec, seeds, max_n, nwarp, max_eval))).map_err(map_cuda_err)?;
        Ok(fit.map(Into::into))
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
}

#[pyproto]
impl PyGCProtocol for PySpectrometer {
    fn __traverse__(&self, visit: PyVisit) -> Result<(), PyTraverseError> {
        visit.call(&self.compound_prism)?;
        visit.call(&self.detector_array)?;
        visit.call(&self.fiber_beam)?;
        Ok(())
    }

    fn __clear__(&mut self) {}
}

#[pyproto]
impl PyObjectProtocol for PySpectrometer {
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
    m.add_class::<PyCompoundPrism>()?;
    m.add_class::<PyDetectorArray>()?;
    m.add_function(wrap_pyfunction!(position_detector_array, m)?)?;
    m.add_class::<PyUniformWavelengthDistribution>()?;
    m.add_class::<PyGaussianBeam>()?;
    m.add_class::<PyFiberBeam>()?;
    m.add_class::<PySpectrometer>()?;
    Ok(())
}
