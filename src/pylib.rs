use crate::glasscat::{new_catalog, CatalogError as _CatalogError, Glass};
use crate::ray::{
    detector_array_positioning, fitness, trace, transmission, CompoundPrism, DetectorArray,
    DetectorArrayPositioning, GaussianBeam, Pair, RayTraceError as _RayTraceError,
};
use cpython::*;
use zerocopy::{FromBytes, LayoutVerified};

trait IntoPyResult<T> {
    fn into_py_result(self, py: Python) -> PyResult<T>;
}

py_exception!(_prism, RayTraceError);
py_exception!(_prism, CatalogError);

impl<T> IntoPyResult<T> for Result<T, _RayTraceError> {
    fn into_py_result(self, py: Python) -> PyResult<T> {
        self.map_err(|e| RayTraceError::new(py, e.name()))
    }
}

impl<T> IntoPyResult<T> for Result<T, _CatalogError> {
    fn into_py_result(self, py: Python) -> PyResult<T> {
        self.map_err(|e| CatalogError::new(py, e.name()))
    }
}

py_class!(class PyGlass |py| {
    data glass: Glass;
    def __new__(_cls, dispform: i32, cd: Vec<f64>) -> PyResult<PyGlass> {
        PyGlass::create_instance(py, Glass::new(dispform, cd).into_py_result(py)?)
    }
    def __getnewargs__(&self) -> PyResult<impl ToPyObject> {
        let (n, v): (i32, &[f64]) = self.glass(py).into();
        Ok((n, v.to_vec()))
    }
    def __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.glass(py)))
    }
});

impl FromPyObject<'_> for Glass {
    fn extract(py: Python, obj: &PyObject) -> PyResult<Self> {
        let pg: &PyGlass = obj.cast_as(py)?;
        Ok(pg.glass(py).clone())
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

fn get_buffer_attr<'p, T: FromBytes>(
    py: Python,
    obj: &'p PyObject,
    k: &'static str,
) -> PyResult<&'p [T]> {
    let obj = obj.getattr(py, k)?;
    let buffer = cpython::buffer::PyBuffer::get(py, &obj)?;
    if !buffer.is_c_contiguous() {
        Err(PyErr::new::<exc::BufferError, _>(
            py,
            "buffer must be C contiguous",
        ))
    } else {
        let bytes = unsafe {
            core::slice::from_raw_parts(buffer.buf_ptr() as *const u8, buffer.len_bytes())
        };
        LayoutVerified::new_slice(bytes)
            .map(LayoutVerified::into_slice)
            .ok_or_else(|| PyErr::new::<exc::BufferError, _>(py, "Invalid buffer"))
    }
}

impl<'p> FromPyObject<'p> for CompoundPrism<'p> {
    fn extract(py: Python, obj: &'p PyObject) -> PyResult<Self> {
        Ok(CompoundPrism {
            glasses: get_attr::<Vec<_>>(py, obj, "glasses")?.into(),
            angles: get_buffer_attr::<f64>(py, obj, "angles")?.into(),
            lengths: get_buffer_attr::<f64>(py, obj, "lengths")?.into(),
            curvature: get_attr(py, obj, "curvature")?,
            height: get_attr(py, obj, "height")?,
            width: get_attr(py, obj, "width")?,
        })
    }
}

impl<'p> FromPyObject<'p> for DetectorArray<'p> {
    fn extract(py: Python, obj: &'p PyObject) -> PyResult<Self> {
        Ok(DetectorArray {
            bins: get_buffer_attr::<[f64; 2]>(py, obj, "bins")?.into(),
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
        ("Prism", "glasses, angles, lengths, curvature, height, width"),
        None,
    )?;
    let det_tuple = namedtuple.call(py, ("DetectorArray", "bins, min_ci, angle, length"), None)?;
    let beam_tuple = namedtuple.call(py, ("GaussianBeam", "width, y_mean, w_range"), None)?;
    m.add_class::<PyGlass>(py)?;
    m.add(py, "Prism", prism_tuple)?;
    m.add(py, "DetectorArray", det_tuple)?;
    m.add(py, "GaussianBeam", beam_tuple)?;
    m.add(
        py,
        "create_catalog",
        py_fn!(py, __create_catalog(file: &str) -> PyResult<impl ToPyObject> {
                new_catalog(file)
                .map(|r| r
                    .into_py_result(py)
                    .and_then(|(k, v)|
                        Ok((k.to_owned(), PyGlass::create_instance(py, v)?))))
                .collect::<PyResult<std::collections::BTreeMap<_, _>>>()
        }),
    )?;
    m.add(py, "fitness", py_fn!(py, __fitness(prism: CompoundPrism, pmts: DetectorArray, beam: GaussianBeam) -> PyResult<impl ToPyObject> {
        py.allow_threads(|| fitness(&prism, &pmts, &beam).map(|arr| Vec::from(arr.as_ref()))).into_py_result(py)
    }))?;
    m.add(py, "trace", py_fn!(py, __trace(wavelength: f64, init_y: f64, prism: CompoundPrism, pmts: DetectorArray, det: DetectorArrayPositioning) -> PyResult<impl ToPyObject> {
        py.allow_threads(|| trace(wavelength, init_y, &prism, &pmts, det)).into_py_result(py)
    }))?;
    m.add(py, "transmission", py_fn!(py, __transmission(wavelengths: Vec<f64>, prism: CompoundPrism, pmts: DetectorArray, beam: GaussianBeam, det: DetectorArrayPositioning) -> PyResult<impl ToPyObject> {
        py.allow_threads(|| Ok(transmission(&wavelengths, &prism, &pmts, &beam, det)))
    }))?;
    m.add(py, "detector_array_position", py_fn!(py, __detector_array_position(prism: CompoundPrism, pmts: DetectorArray, beam: GaussianBeam) -> PyResult<impl ToPyObject> {
        py.allow_threads(|| detector_array_positioning(&prism, &pmts, &beam)).into_py_result(py)
    }))?;
    Ok(())
}

py_module_initializer!(_prism, init_prism, PyInit__prism, |py, m| {
    init_mod(py, m)
});
