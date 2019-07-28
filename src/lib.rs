#[macro_use]
extern crate derive_more;
#[macro_use]
extern crate cpython;
use cpython::{
    exc::TypeError, FromPyObject, ObjectProtocol, PyErr, PyObject, PyResult, Python, ToPyObject,
};
mod glasscat;
mod quad;
mod ray;

use crate::glasscat::{new_catalog, CatalogError as _CatalogError, Glass};
use crate::ray::{
    detector_array_positioning, fitness, trace, transmission, CompoundPrism, DetectorArray,
    DetectorArrayPositioning, GaussianBeam, Pair, RayTraceError as _RayTraceError,
};

trait IntoPyResult<T> {
    fn into_py_result(self, py: Python) -> PyResult<T>;
}

py_exception!(prism, RayTraceError);
py_exception!(prism, CatalogError);

impl<T> IntoPyResult<T> for Result<T, _RayTraceError> {
    fn into_py_result(self, py: Python) -> PyResult<T> {
        self.map_err(|e| RayTraceError::new(py, format!("{}", e)))
    }
}

impl<T> IntoPyResult<T> for Result<T, _CatalogError> {
    fn into_py_result(self, py: Python) -> PyResult<T> {
        self.map_err(|e| CatalogError::new(py, format!("{}", e)))
    }
}

impl ToPyObject for Glass {
    type ObjectType = cpython::PyTuple;

    fn to_py_object(&self, py: Python) -> Self::ObjectType {
        let pyglass: (i32, &[f64]) = self.into();
        ToPyObject::into_py_object(pyglass, py)
    }
}

impl FromPyObject<'_> for Glass {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        let (glass_type, glass_descr): (i32, Vec<f64>) = FromPyObject::extract(py, obj)?;
        Glass::new(glass_type, glass_descr).into_py_result(py)
    }
}

impl ToPyObject for Pair {
    type ObjectType = cpython::PyTuple;

    fn to_py_object(&self, py: Python) -> Self::ObjectType {
        ToPyObject::into_py_object((self.x, self.y), py)
    }
}

impl FromPyObject<'_> for Pair {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        let x = obj.get_item(py, 0)?.extract(py)?;
        let y = obj.get_item(py, 1)?.extract(py)?;
        Ok(Pair { x, y })
    }
}

impl ToPyObject for DetectorArrayPositioning {
    type ObjectType = cpython::PyTuple;

    fn to_py_object(&self, py: Python) -> Self::ObjectType {
        ToPyObject::into_py_object((self.pos, self.dir), py)
    }
}

impl FromPyObject<'_> for DetectorArrayPositioning {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        let pos = obj.get_item(py, 0)?.extract(py)?;
        let dir = obj.get_item(py, 1)?.extract(py)?;
        Ok(DetectorArrayPositioning { pos, dir })
    }
}

fn get_attr<T>(py: Python, obj: &PyObject, k: &'static str) -> PyResult<T>
where
    for<'a> T: FromPyObject<'a>,
{
    obj.getattr(py, k)?.extract(py)
}

fn get_buffer_attr<'p, I, T>(py: Python, obj: &'p PyObject, k: &'static str) -> PyResult<&'p [T]>
where
    I: cpython::buffer::Element + for<'a> cpython::FromPyObject<'a>,
    //      T: zerocopy::FromBytes,
{
    let obj = obj.getattr(py, k)?;
    let buffer = cpython::buffer::PyBuffer::get(py, &obj)?;
    let ptr = buffer.buf_ptr() as *const T;
    if !buffer.is_c_contiguous() {
        Err(PyErr::new::<TypeError, _>(
            py,
            "buffer must be C contiguous",
        ))
    } else if (ptr as usize) % core::mem::align_of::<T>() != 0 {
        Err(PyErr::new::<TypeError, _>(py, "Invalid buffer alignment"))
    } else if !I::is_compatible_format(buffer.format()) {
        Err(PyErr::new::<TypeError, _>(py, "Invalid buffer format"))
    } else if core::mem::size_of::<I>() != buffer.item_size()
        || buffer.len_bytes() % core::mem::size_of::<T>() != 0
    {
        Err(PyErr::new::<TypeError, _>(py, "Invalid buffer size"))
    } else {
        let len = buffer.len_bytes() / std::mem::size_of::<T>();
        Ok(unsafe { core::slice::from_raw_parts(ptr, len) })
    }
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
pub struct OwnedPrism {
    glasses: Vec<Glass>,
    angles: Vec<f64>,
    curvature: f64,
    height: f64,
    width: f64,
}

impl<'p> Into<CompoundPrism<'p>> for &'p OwnedPrism {
    fn into(self) -> CompoundPrism<'p> {
        CompoundPrism {
            glasses: &self.glasses,
            angles: &self.angles,
            curvature: self.curvature,
            height: self.height,
            width: self.width,
        }
    }
}

impl FromPyObject<'_> for OwnedPrism {
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

impl<'p> FromPyObject<'p> for DetectorArray<'p> {
    fn extract(py: Python, obj: &'p PyObject) -> PyResult<Self> {
        Ok(DetectorArray {
            bins: get_buffer_attr::<f64, [f64; 2]>(py, obj, "bins")?,
            min_ci: get_attr(py, obj, "min_ci")?,
            angle: get_attr(py, obj, "angle")?,
            length: get_attr(py, obj, "length")?,
        })
    }
}

impl FromPyObject<'_> for GaussianBeam {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        Ok(GaussianBeam {
            width: get_attr(py, obj, "width")?,
            y_mean: get_attr(py, obj, "y_mean")?,
            w_range: get_attr(py, obj, "w_range")?,
        })
    }
}

fn init_mod(py: Python, m: &cpython::PyModule) -> PyResult<()> {
    m.add(py, "__doc__", "Compound Prism Designer")?;
    m.add(py, "RayTraceError", py.get_type::<RayTraceError>())?;
    m.add(py, "CatalogError", py.get_type::<CatalogError>())?;
    let collections = py.import("collections")?;
    let namedtuple = collections.get(py, "namedtuple")?;
    let prism_tuple = namedtuple.call(
        py,
        ("Prism", "glasses, angles, curvature, height, width"),
        None,
    )?;
    let pmt_tuple = namedtuple.call(py, ("PmtArray", "bins, min_ci, angle, length"), None)?;
    let beam_tuple = namedtuple.call(py, ("GaussianBeam", "width, y_mean, w_range"), None)?;
    m.add(py, "Prism", prism_tuple)?;
    m.add(py, "PmtArray", pmt_tuple)?;
    m.add(py, "GaussianBeam", beam_tuple)?;
    m.add(
        py,
        "create_catalog",
        py_fn!(py, __create_catalog(file: &str) -> PyResult<impl ToPyObject> {
                new_catalog(file).into_py_result(py)
        }),
    )?;
    m.add(py, "fitness", py_fn!(py, __fitness(prism: OwnedPrism, pmts: DetectorArray, beam: GaussianBeam) -> PyResult<impl ToPyObject> {
        py.allow_threads(|| fitness((&prism).into(), pmts, beam).map(|arr| Vec::from(arr.as_ref()))).into_py_result(py)
    }))?;
    m.add(py, "trace", py_fn!(py, __trace(wavelength: f64, init_y: f64, prism: OwnedPrism, pmts: DetectorArray, det: DetectorArrayPositioning) -> PyResult<impl ToPyObject> {
        py.allow_threads(|| trace(wavelength, init_y, (&prism).into(), pmts, det)).into_py_result(py)
    }))?;
    m.add(py, "transmission", py_fn!(py, __transmission(wavelengths: Vec<f64>, prism: OwnedPrism, pmts: DetectorArray, beam: GaussianBeam, det: DetectorArrayPositioning) -> PyResult<impl ToPyObject> {
        py.allow_threads(|| Ok(transmission(&wavelengths, (&prism).into(), pmts, beam, det)))
    }))?;
    m.add(py, "spectrometer_position", py_fn!(py, __spectrometer_position(prism: OwnedPrism, pmts: DetectorArray, beam: GaussianBeam) -> PyResult<impl ToPyObject> {
        py.allow_threads(|| detector_array_positioning((&prism).into(), pmts, beam)).into_py_result(py)
    }))?;
    Ok(())
}

py_module_initializer!(prism, initprism, PyInit_prism, |py, m| { init_mod(py, m) });
