use crate::glasscat::{new_catalog, Glass};
use crate::ray::{CompoundPrism, DetectorArray, DetectorArrayPositioning, GaussianBeam, Pair};

#[repr(C)]
pub struct FFiStr {
    str_len: usize,
    str_ptr: *const u8,
}

impl From<&str> for FFiStr {
    fn from(s: &str) -> Self {
        Self {
            str_len: s.len(),
            str_ptr: s.as_ptr(),
        }
    }
}

#[repr(C)]
pub struct FfiResult {
    succeeded: bool,
    err_str: FFiStr,
}

impl From<Option<&str>> for FfiResult {
    fn from(s: Option<&str>) -> Self {
        match s {
            Some(s) => Self {
                succeeded: false,
                err_str: s.into(),
            },
            None => Self {
                succeeded: true,
                err_str: FFiStr {
                    str_len: 0,
                    str_ptr: core::ptr::null(),
                },
            },
        }
    }
}

#[repr(C)]
pub struct GlassCatalogState {
    _private: [u8; 0],
}

#[no_mangle]
pub unsafe extern "C" fn update_glass_catalog(
    file: *const u8,
    file_len: usize,
    insert_item: fn(*mut GlassCatalogState, FFiStr, *mut Glass),
    state_ptr: *mut GlassCatalogState,
) -> FfiResult {
    let file = core::slice::from_raw_parts(file, file_len);
    if let Ok(file) = core::str::from_utf8(file) {
        new_catalog(file)
            .try_for_each(|r| {
                let (name, glass) = r.map_err(|e| e.name())?;
                insert_item(state_ptr, name.into(), Box::into_raw(Box::new(glass)));
                Ok(())
            })
            .err()
            .into()
    } else {
        Some("File has invalid utf-8 encoding").into()
    }
}

#[repr(C)]
pub struct SerializeBytesState { _private: [u8; 0], }

#[no_mangle]
pub unsafe extern "C" fn serialize_glass(
    glass: &Glass,
    append_next_byte: fn(*mut SerializeBytesState, u8),
    state_ptr: *mut SerializeBytesState,
) -> FfiResult {
    match serde_cbor::to_vec(glass) {
        Ok(v) => {
            for b in v {
                append_next_byte(state_ptr, b)
            }
            None
        }
        Err(_) => Some("Failed to serialize"),
    }
    .into()
}

#[no_mangle]
pub unsafe extern "C" fn deserialize_glass(
    glass: *mut *mut Glass,
    serialized_bytes: *const u8,
    serialized_bytes_length: usize,
) -> FfiResult {
    match serde_cbor::from_slice(core::slice::from_raw_parts(serialized_bytes, serialized_bytes_length)) {
        Ok(g) => {
            glass.write(Box::into_raw(Box::new(g)));
            None
        }
        Err(_) => Some("Failed to deserialize"),
    }
    .into()
}

#[no_mangle]
pub unsafe extern "C" fn free_glass(glass: *mut Glass) {
    drop(Box::from_raw(glass))
}

#[no_mangle]
pub unsafe extern "C" fn create_compound_prism(
    prism_count: usize,
    glasses: *const &'static Glass,
    angles: *const f64,
    lengths: *const f64,
    curvature: f64,
    height: f64,
    width: f64,
) -> *mut CompoundPrism<'static> {
    let glasses = core::slice::from_raw_parts(glasses, prism_count);
    let angles = core::slice::from_raw_parts(angles, 1 + prism_count);
    let lengths = core::slice::from_raw_parts(lengths, prism_count);
    Box::into_raw(Box::new(CompoundPrism {
        glasses: glasses.to_owned().into(),
        angles: angles.to_owned().into(),
        lengths: lengths.to_owned().into(),
        curvature,
        height,
        width,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn free_compound_prism(cmpnd: *mut CompoundPrism<'static>) {
    drop(Box::from_raw(cmpnd))
}

#[no_mangle]
pub unsafe extern "C" fn create_detector_array(
    bin_count: usize,
    bins: *const [f64; 2],
    min_ci: f64,
    angle: f64,
    length: f64,
) -> *mut DetectorArray<'static> {
    let bins = core::slice::from_raw_parts(bins, bin_count);
    Box::into_raw(Box::new(DetectorArray {
        bins: bins.to_owned().into(),
        min_ci,
        angle,
        length,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn free_detector_array(detarr: *mut DetectorArray<'static>) {
    drop(Box::from_raw(detarr))
}

#[no_mangle]
pub unsafe extern "C" fn create_gaussian_beam(
    width: f64,
    y_mean: f64,
    wmin: f64,
    wmax: f64,
) -> *mut GaussianBeam {
    Box::into_raw(Box::new(GaussianBeam {
        width,
        y_mean,
        w_range: (wmin, wmax),
    }))
}

#[no_mangle]
pub unsafe extern "C" fn free_gaussian_beam(beam: *mut GaussianBeam) {
    drop(Box::from_raw(beam))
}

#[repr(C)]
pub struct DesignFitness {
    size: f64,
    info: f64,
    deviation: f64,
}

#[no_mangle]
pub unsafe extern "C" fn fitness(
    prism: &CompoundPrism,
    detector_array: &DetectorArray,
    gaussian_beam: &GaussianBeam,
    design_fitness: *mut DesignFitness,
) -> FfiResult {
    match crate::ray::fitness(prism, detector_array, gaussian_beam) {
        Ok([size, info, deviation]) => {
            design_fitness.write(DesignFitness {
                size,
                info,
                deviation,
            });
            None
        }
        Err(e) => Some(e.name()),
    }
    .into()
}

#[no_mangle]
pub unsafe extern "C" fn detector_array_position(
    prism: &CompoundPrism,
    detector_array: &DetectorArray,
    gaussian_beam: &GaussianBeam,
    position: *mut DetectorArrayPositioning,
) -> FfiResult {
    match crate::ray::detector_array_positioning(prism, detector_array, gaussian_beam) {
        Ok(p) => {
            position.write(p);
            None
        }
        Err(e) => Some(e.name()),
    }
    .into()
}

#[repr(C)]
pub struct ProbabilitiesState { _private: [u8; 0], }

#[no_mangle]
pub unsafe extern "C" fn p_dets_l_wavelength(
    wavelength: f64,
    prism: &CompoundPrism,
    detector_array: &DetectorArray,
    gaussian_beam: &GaussianBeam,
    detpos: &DetectorArrayPositioning,
    append_next_detector_probability: fn(*mut ProbabilitiesState, f64),
    state_ptr: *mut ProbabilitiesState,
) {
    for p in crate::ray::p_dets_l_wavelength(wavelength, prism, detector_array, gaussian_beam, *detpos) {
        append_next_detector_probability(state_ptr, p)
    }
}

#[repr(C)]
pub struct TracedRayState { _private: [u8; 0], }

#[no_mangle]
pub unsafe extern "C" fn trace(
    wavelength: f64,
    inital_y: f64,
    prism: &CompoundPrism,
    detector_array: &DetectorArray,
    detpos: &DetectorArrayPositioning,
    append_next_ray_position: fn(*mut TracedRayState, Pair),
    state_ptr: *mut TracedRayState,
) -> FfiResult {
    match crate::ray::trace(wavelength, inital_y, prism, detector_array, *detpos) {
        Ok(t) => {
            for p in t {
                append_next_ray_position(state_ptr,  p)
            }
            None
        }
        Err(e) => Some(e.name()),
    }
    .into()
}
