#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate cpython;
use cpython::{
    FromPyObject, ObjectProtocol, PyObject, PyResult, PyTuple, Python, PythonObjectWithTypeObject,
    ToPyObject,
};
mod glasscat;
mod quad;
mod ray;

use crate::glasscat::{new_catalog, Glass};
use crate::ray::{
    get_spectrometer_position, merit, trace, transmission, GaussianBeam, PmtArray, Prism,
};
use alga::general::RealField;

py_exception!(prism, PrismError);

fn to_pyresult<T, E: std::error::Error>(py: Python, r: Result<T, E>) -> PyResult<T> {
    r.map_err(|e| {
        let err = format!("{}", e);
        PrismError::new(py, err)
    })
}

impl<N: RealField + ToPyObject> ToPyObject for Glass<N> {
    type ObjectType = PyTuple;

    fn to_py_object(&self, py: Python) -> Self::ObjectType {
        let pyglass: (i32, &[N]) = self.into();
        ToPyObject::into_py_object(pyglass, py)
    }
}

impl<N: RealField + for<'a> cpython::FromPyObject<'a>> FromPyObject<'_> for Glass<N> {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        let (glass_type, glass_descr): (i32, Vec<N>) = FromPyObject::extract(py, obj)?;
        to_pyresult(py, Glass::new(glass_type, &glass_descr))
    }
}

fn get_attr<T>(py: Python, obj: &PyObject, k: &'static str) -> PyResult<T>
where
    for<'a> T: FromPyObject<'a>,
{
    obj.getattr(py, k)?.extract(py)
}

macro_rules! obj_to_struct {
    (@field $py:ident $obj:ident => $field:ident) => {
        get_attr($py, $obj, stringify!($field))?
    };
    ($py:ident, $obj:ident => $stype:ident { $( $($fields:ident)+ ),* }) => {
        $stype::new(
            $( obj_to_struct!(@field $py $obj => $($fields)*) ),*
        )
    };
}

#[derive(Constructor, Debug, Clone)]
pub struct OwnedPrism<N: RealField> {
    pub glasses: Vec<Glass<N>>,
    pub angles: Vec<N>,
    pub curvature: N,
    pub height: N,
    pub width: N,
}

impl<'p, N: RealField> Into<Prism<'p, N>> for &'p OwnedPrism<N> {
    fn into(self) -> Prism<'p, N> {
        Prism {
            glasses: &self.glasses,
            angles: &self.angles,
            curvature: self.curvature,
            height: self.height,
            width: self.width,
        }
    }
}

#[derive(Constructor, Debug, Clone)]
pub struct OwnedPmtArray<N: RealField> {
    pub bins: Vec<(N, N)>,
    pub min_ci: N,
    pub angle: N,
    pub length: N,
}

impl<'p, N: RealField> Into<PmtArray<'p, N>> for &'p OwnedPmtArray<N> {
    fn into(self) -> PmtArray<'p, N> {
        PmtArray {
            bins: &self.bins,
            min_ci: self.min_ci,
            angle: self.angle,
            length: self.length,
        }
    }
}

impl<N: RealField + for<'a> cpython::FromPyObject<'a>> FromPyObject<'_> for OwnedPrism<N> {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        Ok(obj_to_struct!(py, obj => OwnedPrism{
            glasses,
            angles,
            curvature,
            height,
            width
        }))
    }
}

impl<N: RealField + for<'a> cpython::FromPyObject<'a>> FromPyObject<'_> for OwnedPmtArray<N> {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        Ok(obj_to_struct!(py, obj => OwnedPmtArray{
            bins,
            min_ci,
            angle,
            length
        }))
    }
}

impl<N: RealField + for<'a> cpython::FromPyObject<'a>> FromPyObject<'_> for GaussianBeam<N> {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        Ok(obj_to_struct!(py, obj => GaussianBeam{
            width,
            y_mean
        }))
    }
}

py_module_initializer!(prism, initprism, PyInit_prism, |py, m| {
    m.add(py, "__doc__", "Compound Prism Designer")?;
    m.add(py, "PrismError", PrismError::type_object(py))?;
    let collections = py.import("collections")?;
    let namedtuple = collections.get(py, "namedtuple")?;
    let prism_tuple = namedtuple.call(
        py,
        ("Prism", "glasses, angles, curvature, height, width"),
        None,
    )?;
    let pmt_tuple = namedtuple.call(py, ("PmtArray", "bins, min_ci, angle, length"), None)?;
    let beam_tuple = namedtuple.call(py, ("GaussianBeam", "width, y_mean"), None)?;
    m.add(py, "Prism", prism_tuple)?;
    m.add(py, "PmtArray", pmt_tuple)?;
    m.add(py, "GaussianBeam", beam_tuple)?;
    m.add(
        py,
        "create_catalog",
        py_fn!(py, create_catalog(file: &str) -> PyResult<impl ToPyObject> {
                to_pyresult(py, new_catalog::<f64>(file))
        }),
    )?;
    m.add(py, "fitness", py_fn!(py, fitness(wmin: f64, wmax: f64, prism: OwnedPrism<f64>, pmts: OwnedPmtArray<f64>, beam: GaussianBeam<f64>) -> PyResult<impl ToPyObject> {
        to_pyresult(py, py.allow_threads(|| merit(wmin, wmax, (&prism).into(), (&pmts).into(), beam).map(|arr| Vec::from(arr.as_ref()))))
    }))?;
    m.add(py, "trace", py_fn!(py, ray_trace(wavelength: f64, wmin: f64, wmax: f64, init_y: f64, prism: OwnedPrism<f64>, pmts: OwnedPmtArray<f64>, beam: GaussianBeam<f64>) -> PyResult<impl ToPyObject> {
        to_pyresult(py, py.allow_threads(|| trace(wavelength, wmin, wmax, init_y, (&prism).into(), (&pmts).into(), beam)))
    }))?;
    m.add(py, "transmission", py_fn!(py, ray_transmission(wavelengths: Vec<f64>, prism: OwnedPrism<f64>, pmts: OwnedPmtArray<f64>, beam: GaussianBeam<f64>) -> PyResult<impl ToPyObject> {
        to_pyresult(py, py.allow_threads(|| transmission(&wavelengths, (&prism).into(), (&pmts).into(), beam)))
    }))?;
    m.add(py, "spectrometer_position", py_fn!(py, spectrometer_position(wmin: f64, wmax: f64, prism: OwnedPrism<f64>, pmts: OwnedPmtArray<f64>, beam: GaussianBeam<f64>) -> PyResult<impl ToPyObject> {
        to_pyresult(py, py.allow_threads(|| get_spectrometer_position(wmin, wmax, (&prism).into(), (&pmts).into(), beam).map(|spec| ((spec.pos.x, spec.pos.y), (spec.dir.x, spec.dir.y)))))
    }))?;
    Ok(())
});
