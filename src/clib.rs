use crate::glasscat::{new_catalog, Glass};
use crate::ray::{
    CompoundPrism, DesignFitness, DetectorArray, DetectorArrayPositioning, GaussianBeam, Pair,
};
use crate::optimizer::{Config, AGE, SBX, PM};

#[repr(C)]
pub struct FFiStr {
    str_len: usize,
    str_ptr: *const u8,
}

impl From<&'static str> for FFiStr {
    fn from(s: &str) -> Self {
        Self {
            str_len: s.len(),
            str_ptr: s.as_ptr(),
        }
    }
}

impl Into<String> for &FFiStr {
    fn into(self) -> String {
        unsafe {
            let utf8_slice = std::slice::from_raw_parts(self.str_ptr, self.str_len);
            String::from_utf8_lossy(utf8_slice).into_owned()
        }
    }
}

#[repr(C)]
pub struct FfiResult {
    succeeded: bool,
    err_str: FFiStr,
}

impl<E: Into<&'static str>> From<Result<(), E>> for FfiResult {
    fn from(r: Result<(), E>) -> Self {
        match r {
            Err(e) => Self {
                succeeded: false,
                err_str: e.into().into(),
            },
            Ok(()) => Self {
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
                r.map(|(name, glass)| {
                    insert_item(state_ptr, name.into(), Box::into_raw(Box::new(glass)))
                })
            })
            .into()
    } else {
        Err("File has invalid utf-8 encoding").into()
    }
}

#[no_mangle]
pub unsafe extern "C" fn glass_parametrization(
    glass: &Glass,
    dispersion_formula_number_ptr: *mut i32,
    dispersion_constants_ptr: *mut *const f64,
    dispersion_constants_len_ptr: *mut usize,
) {
    let (n, s): (i32, &[f64]) = glass.into();
    dispersion_formula_number_ptr.write(n);
    dispersion_constants_ptr.write(s.as_ptr());
    dispersion_constants_len_ptr.write(s.len());
}

#[no_mangle]
pub unsafe extern "C" fn new_glass_from_parametrization(
    dispersion_formula_number: i32,
    dispersion_constants: *const f64,
    dispersion_constants_len: usize,
    glass_ptr: *mut *mut Glass,
) -> FfiResult {
    let s = core::slice::from_raw_parts(dispersion_constants, dispersion_constants_len);
    Glass::new(dispersion_formula_number, s.iter().copied())
        .map(|glass| glass_ptr.write(Box::into_raw(Box::new(glass))))
        .into()
}

#[no_mangle]
pub unsafe extern "C" fn free_glass(glass: *mut Glass) {
    drop(Box::from_raw(glass))
}

#[no_mangle]
pub unsafe extern "C" fn glass_refractive_index(glass: &Glass, wavelength: f64) -> f64 {
    glass.calc_n(wavelength)
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
    Box::into_raw(Box::new(CompoundPrism::new(
        glasses.iter().copied(),
        angles,
        lengths,
        curvature,
        height,
        width,
    )))
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
    Box::into_raw(Box::new(DetectorArray::new(
        bins.to_owned().into(),
        min_ci,
        angle,
        length,
    )))
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

#[no_mangle]
pub unsafe extern "C" fn fitness(
    prism: &CompoundPrism,
    detector_array: &DetectorArray,
    gaussian_beam: &GaussianBeam,
    design_fitness: *mut DesignFitness,
) -> FfiResult {
    crate::ray::fitness(prism, detector_array, gaussian_beam)
        .map(|v| design_fitness.write(v))
        .into()
}

#[no_mangle]
pub unsafe extern "C" fn detector_array_position(
    prism: &CompoundPrism,
    detector_array: &DetectorArray,
    gaussian_beam: &GaussianBeam,
    position: *mut DetectorArrayPositioning,
) -> FfiResult {
    crate::ray::detector_array_positioning(prism, detector_array, gaussian_beam)
        .map(|p| position.write(p))
        .into()
}

#[repr(C)]
pub struct ProbabilitiesState {
    _private: [u8; 0],
}

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
    for p in
        crate::ray::p_dets_l_wavelength(wavelength, prism, detector_array, gaussian_beam, detpos)
    {
        append_next_detector_probability(state_ptr, p)
    }
}

#[repr(C)]
pub struct TracedRayState {
    _private: [u8; 0],
}

#[no_mangle]
pub unsafe extern "C" fn trace(
    wavelength: f64,
    inital_y: f64,
    prism: &'static CompoundPrism<'static>,
    detector_array: &'static DetectorArray<'static>,
    detpos: &DetectorArrayPositioning,
    append_next_ray_position: fn(*mut TracedRayState, Pair),
    state_ptr: *mut TracedRayState,
) -> FfiResult {
    crate::ray::trace(wavelength, inital_y, prism, detector_array, detpos)
        .try_for_each(|r| r.map(|p| append_next_ray_position(state_ptr, p)))
        .into()
}


#[repr(C)]
pub struct OptimizerSpecification {
    iteration_count: usize,
    population_size: usize,
    offspring_size: usize,
    crossover_distribution_index: f64,
    mutation_distribution_index: f64,
    mutation_probability: f64,
    seed: u64,
    epsilons: [f64; 3],
}

#[repr(C)]
pub struct CompoundPrismSpecification {
    max_count: usize,
    max_height: f64,
    width: f64,
}

#[repr(C)]
pub struct GaussianBeamSpecification {
    width: f64,
    wavelength_minimum: f64,
    wavelength_maximum: f64,
}

#[repr(C)]
pub struct DetectorArraySpecification {
    length: f64,
    max_incident_angle: f64,
    bin_bounds: *const [f64; 2],
    bin_count: usize,
}

#[repr(C)]
pub struct Specification {
    glass_count: usize,
    glass_names: *const FFiStr,
    glass_values: *const *const Glass,
    optimizer: OptimizerSpecification,
    compound_prism: CompoundPrismSpecification,
    gaussian_beam: GaussianBeamSpecification,
    detector_array: DetectorArraySpecification,
}

#[repr(C)]
pub struct Design {
    compound_prism: CompoundPrism<'static>,
    gaussian_beam: GaussianBeam,
    detector_array: DetectorArray<'static>,
    detector_array_position: DetectorArrayPositioning,
}

#[repr(C)]
pub struct DesignResult {
    design: *mut Design,
    fitness: DesignFitness,
}

#[no_mangle]
pub unsafe extern "C" fn optimize(
    spec: &Specification,
    results_ptr: *mut *const DesignResult,
) -> usize {
    let glass_catalog = if spec.glass_count > 0 {
        let glass_names = std::slice::from_raw_parts(spec.glass_names, spec.glass_count);
        let glass_values = std::slice::from_raw_parts(spec.glass_values as *const &Glass, spec.glass_count);
        glass_names.iter().zip(glass_values).map(|(s, &g)| (s.into(), g.clone())).collect()
    } else {
        crate::glasscat::BUNDLED_CATALOG.iter().map(|(s, g)| (s.to_string(), g.clone())).collect()
    };
    let config = Config {
        max_prism_count: spec.compound_prism.max_count,
        wavelength_range: (spec.gaussian_beam.wavelength_minimum, spec.gaussian_beam.wavelength_maximum),
        beam_width: spec.gaussian_beam.width,
        max_prism_height: spec.compound_prism.max_height,
        prism_width: spec.compound_prism.width,
        detector_array_length: spec.detector_array.length,
        detector_array_min_ci: spec.detector_array.max_incident_angle.to_radians().cos(),
        detector_array_bin_bounds: std::slice::from_raw_parts(spec.detector_array.bin_bounds, spec.detector_array.bin_count).into(),
        glass_catalog,
    };
    let mut optimizer = AGE::new(
        &config,
        spec.optimizer.population_size,
        spec.optimizer.offspring_size,
        spec.optimizer.seed,
        spec.optimizer.epsilons,
        SBX { distribution_index: spec.optimizer.crossover_distribution_index },
        PM { probability: spec.optimizer.mutation_probability, distribution_index: spec.optimizer.mutation_distribution_index }
    );
    for _ in 0..spec.optimizer.iteration_count {
        optimizer.iterate();
    }

    optimizer.archive.len()
}

#[no_mangle]
pub unsafe extern "C" fn free_design_result(design: *mut Design, count: usize) {
    drop(Box::from_raw(design))
}

#[no_mangle]
pub unsafe extern "C" fn trace_wavelength(
    wavelength: f64,
    inital_y: f64,
    design: &Design,
    append_next_ray_position: fn(*mut TracedRayState, Pair),
    state_ptr: *mut TracedRayState,
) -> FfiResult {
    crate::ray::trace(wavelength, inital_y, &design.compound_prism, &design.detector_array, &design.detector_array_position)
        .try_for_each(|r| r.map(|p| append_next_ray_position(state_ptr, p)))
        .into()
}

#[no_mangle]
pub unsafe extern "C" fn transmission_probability(
    wavelength: f64,
    design: &Design,
    append_next_detector_probability: fn(*mut ProbabilitiesState, f64),
    state_ptr: *mut ProbabilitiesState,
) {
    for p in
        crate::ray::p_dets_l_wavelength(wavelength, &design.compound_prism, &design.detector_array, &design.gaussian_beam, &design.detector_array_position)
        {
            append_next_detector_probability(state_ptr, p)
        }
}
