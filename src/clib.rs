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

#[no_mangle]
pub unsafe extern "C" fn serialize_glass(
    glass: &Glass,
    buffer: *mut *mut u8,
    length: *mut usize,
) -> FfiResult {
    match serde_cbor::to_vec(glass) {
        Ok(mut v) => {
            v.shrink_to_fit();
            length.write(v.len());
            buffer.write(v.as_mut_ptr());
            core::mem::forget(v);
            None
        }
        Err(_) => Some("Failed to serialize"),
    }
    .into()
}

#[no_mangle]
pub unsafe extern "C" fn deserialize_glass(
    glass: *mut *mut Glass,
    buffer: *const u8,
    length: usize,
) -> FfiResult {
    match serde_cbor::from_slice(core::slice::from_raw_parts(buffer, length)) {
        Ok(g) => {
            glass.write(Box::into_raw(Box::new(g)));
            None
        }
        Err(_) => Some("Failed to deserialize"),
    }
    .into()
}

#[no_mangle]
pub unsafe extern "C" fn free_serialized_buffer(buffer: *mut u8, length: usize) {
    drop(Vec::from_raw_parts(buffer, length, length))
}

#[no_mangle]
pub unsafe extern "C" fn free_glass(glass: *mut Glass) {
    drop(Box::from_raw(glass))
}

#[no_mangle]
pub unsafe extern "C" fn create_compound_prism(
    prism_count: usize,
    glasses: *const &Glass,
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
        glasses: glasses
            .iter()
            .map(|g| (*g).clone())
            .collect::<Vec<Glass>>()
            .into(),
        angles: angles.into(),
        lengths: lengths.into(),
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
        bins: bins.into(),
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

#[no_mangle]
pub unsafe extern "C" fn p_dets_l_wavelength(
    wavelength: f64,
    prism: &CompoundPrism,
    detector_array: &DetectorArray,
    gaussian_beam: &GaussianBeam,
    detpos: &DetectorArrayPositioning,
    probabilities: *mut *mut f64,
    probabilities_len: *mut usize,
) {
    let mut ps =
        crate::ray::p_dets_l_wavelength(wavelength, prism, detector_array, gaussian_beam, *detpos)
            .into_iter()
            .collect::<Vec<_>>();
    ps.shrink_to_fit();
    probabilities.write(ps.as_mut_ptr());
    probabilities_len.write(ps.len());
    core::mem::forget(ps);
}

#[no_mangle]
pub unsafe extern "C" fn free_probabilities(probabilities: *mut f64, probabilities_length: usize) {
    drop(Vec::from_raw_parts(
        probabilities,
        probabilities_length,
        probabilities_length,
    ))
}

#[no_mangle]
pub unsafe extern "C" fn trace(
    wavelength: f64,
    inital_y: f64,
    prism: &CompoundPrism,
    detector_array: &DetectorArray,
    detpos: &DetectorArrayPositioning,
    traced_positions: *mut *mut Pair,
    traced_positions_len: *mut usize,
) -> FfiResult {
    match crate::ray::trace(wavelength, inital_y, prism, detector_array, *detpos) {
        Ok(mut t) => {
            t.shrink_to_fit();
            traced_positions.write(t.as_mut_ptr());
            traced_positions_len.write(t.len());
            core::mem::forget(t);
            None
        }
        Err(e) => Some(e.name()),
    }
    .into()
}

#[no_mangle]
pub unsafe extern "C" fn free_traced_positions(
    traced_positions: *mut Pair,
    traced_positions_length: usize,
) {
    drop(Vec::from_raw_parts(
        traced_positions,
        traced_positions_length,
        traced_positions_length,
    ))
}
