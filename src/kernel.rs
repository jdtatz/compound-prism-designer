use crate::{CompoundPrism, DetectorArray, DetectorArrayPositioning, GaussianBeam, Ray};
use crate::utils::F64Ext;
use core::{arch::nvptx::*, slice::{from_raw_parts, from_raw_parts_mut}, panic::PanicInfo};

#[panic_handler]
unsafe fn panic_handle(p: &PanicInfo) -> ! {
    let (file, line) = if let Some(loc) = p.location() {
        (loc.file().as_ptr(), loc.line())
    } else {
        (b"".as_ptr(), 0, )
    };
    if let Some(s) = p.payload().downcast_ref::<&str>() {
        __assert_fail(s.as_ptr(), file, line, b"".as_ptr());
    } else {
        __assert_fail(b"Panicked".as_ptr(), file, line, b"".as_ptr());
    }
    trap()
}

extern "C" {
    #[link_name="llvm.nvvm.ptr.shared.to.gen"]
    fn ptr_shared_to_gen(shared_ptr: *mut core::ffi::c_void) -> *mut core::ffi::c_void;

    #[link_name = "llvm.nvvm.read.ptx.sreg.laneid"]
    fn read_ptx_sreg_laneid() -> u32;

    #[link_name = "llvm.nvvm.shfl.sync.bfly.f32"]
    fn shfl_bfly_sync_f32(mask: u32, val: f32, lane_mask: u32, packing: u32) -> f32;

    #[link_name = "llvm.nvvm.shfl.sync.bfly.i32"]
    fn shfl_bfly_sync_i32(mask: u32, val: i32, lane_mask: u32, packing: u32) -> i32;
}


fn next_welford_sample(mean: &mut f64, m2: &mut f64, value: f64, count: f64) {
    let delta = value - *mean;
    *mean += delta / count;
    let delta2 = value - *mean;
    *m2 += delta * delta2;
}

fn parallel_welford_combine(mean_a: f64, m2_a: f64, count_a: f64, mean_b: f64, m2_b: f64, count_b: f64) -> [f64; 3] {
    let count = count_a + count_b;
    let delta = mean_b - mean_a;
    let mean = (count_a * mean_a + count_b * mean_b) / count;
    let m2 = m2_a + m2_b + (delta * delta) * count_a * count_b / count;
    [mean, m2, count]
}

/// Is the Standard Error of the Mean (SEM) less than the error threshold?
/// Uses the square of the error for numerical stability (avoids sqrt)
pub fn sem_le_error_threshold(m2: f64, count: f64, error_squared: f64) -> bool {
    // SEM^2 = self.sample_variance() / self.count
    m2 < error_squared * (count * (count - 1.))
}


unsafe fn cuda_memcpy_1d<T>(dest: *mut u8, src: &T) -> &T {
    let dest = dest as *mut u32;
    let src = src as *const T as *const u32;
    let count = (core::mem::size_of::<T>() / core::mem::size_of::<u32>()) as u32;
    // let count = (core::mem::size_of::<T>()) as u32;
    let mut id = _thread_idx_x() as u32;
    while id < count {
        dest.add(id as usize).write_volatile(src.add(id as usize).read_volatile());
        id += _block_dim_x() as u32;
    }
    &*(dest as *const T)
}

unsafe fn cuda_memcpy_slice_1d<T>(dest: *mut u8, src: &[T]) -> &[T] {
    let dest = dest as *mut u32;
    let len = src.len();
    let count = (core::mem::size_of::<T>() * len / core::mem::size_of::<u32>()) as u32;
    let src = src.as_ptr() as *const u32;
    let mut id = _thread_idx_x() as u32;
    while id < count {
        dest.add(id as usize).write_volatile(src.add(id as usize).read_volatile());
        id += _block_dim_x() as u32;
    }
    core::slice::from_raw_parts(dest as *const T, len)
}


#[no_mangle]
pub unsafe extern "ptx-kernel" fn prob_dets_given_wavelengths(
    seed: f64,
    max_evals: u32,
    spectrometer: &crate::Spectrometer,
    nbin: u32,
    bins: *const [f64; 2],
    prob: *mut f64,
) {
    if max_evals == 0 || nbin == 0 {
        return;
    }
    const MAX_ERR: f64 = 5e-3;
    const MAX_ERR_SQR: f64 = MAX_ERR * MAX_ERR;
    const PHI: f64 = 1.61803398874989484820458683436563;
    const ALPHA: f64 = 1_f64 / PHI;

    let laneid = read_ptx_sreg_laneid();
    let tid = _thread_idx_x() as u32;
    let warpid = tid / 32;
    let nwarps = _block_dim_x() as u32 / 32;
    let bins = from_raw_parts(bins, nbin as usize);

    let ptr: *mut u8 = ptr_shared_to_gen(0 as _) as _;
    let spec_ptr = ptr;
    let bins_ptr = spec_ptr.add(core::mem::size_of::<crate::Spectrometer>());
    let prob_ptr = bins_ptr.add(core::mem::size_of_val(bins));
    let ptr = prob_ptr as *mut [f64; 2];

    let shared = from_raw_parts_mut(ptr.add((warpid * nbin) as usize), nbin as usize);
    if laneid == 0 {
        for [mean, m2] in shared.iter_mut() {
            *mean = 0_f64;
            *m2 = 0_f64;
        }
    }
    let spectrometer = cuda_memcpy_1d(spec_ptr, spectrometer);
    let bins = cuda_memcpy_slice_1d(bins_ptr, bins);
    _syncthreads();
    if tid == 0 {
        let bins_ptr = (&spectrometer.detector_array.bins) as *const _ as *mut _;
        core::mem::replace(&mut*(bins_ptr), bins);
    }
    _syncthreads();
    let beam = &spectrometer.gaussian_beam;
    let cmpnd = &spectrometer.compound_prism;
    let detarr = &spectrometer.detector_array;
    let detpos = &spectrometer.detector_array_position;

    let id = (_block_idx_x() as u32) * nwarps + warpid;

    let u = (id as f64).mul_add(ALPHA, seed).fract();
    let wavelength = beam.w_range.0 + (beam.w_range.1 - beam.w_range.0) * u;

    let mut count = 0_f64;
    let mut index = laneid;
    let mut qrng_n = laneid as f64;
    let max_evals = max_evals - (max_evals % 32);
    while index < max_evals {
        use core::f64::consts::FRAC_1_SQRT_2;
        let u = qrng_n.mul_add(ALPHA, seed).fract();
        let y0 = beam.y_mean - beam.width * FRAC_1_SQRT_2 * crate::erf::erfc_inv(2. * u);
        let result = Ray::new_from_start(y0).propagate(wavelength, cmpnd, detarr, detpos);
        count += 1_f64;
        let prev_count = count;
        let mut finished = true;
        for ([lb, ub], [shared_mean, shared_m2]) in spectrometer.detector_array.bins.iter().copied().zip(shared.iter_mut()) {
            let mut mean = match result {
                Ok((_, pos, t)) if lb <= pos && pos < ub => t,
                _ => 0_f64
            };
            if laneid == 0 {
                next_welford_sample(shared_mean, shared_m2, mean, prev_count);
                mean = *shared_mean;
            }
            let m2 = if laneid == 0 { *shared_m2 } else { 0_f64 };
            let n = if laneid == 0 { prev_count } else { 1_f64 };
            let arr = share(mean, m2, n);
            if laneid == 0 {
                *shared_mean = arr[0];
                *shared_m2 = arr[1];
                count = arr[2];
            }
            finished = finished && sem_le_error_threshold(arr[1], arr[2], MAX_ERR_SQR);
        }
        if finished {
            break;
        }
        index += 32;
        qrng_n += 32.;
    }
    let prob = from_raw_parts_mut(prob.add((nbin * id) as usize), nbin as usize);
    if laneid == 0 {
        for (p, [mean, _m2]) in prob.iter_mut().zip(shared.iter().copied()) {
            *p = mean;
        }
    }
}

unsafe fn bfly_shfl_f64(v: f64, xor: u32) -> f64 {
    /*let out: f64;
    asm!(concat!("{ .reg .b32 lo, hi;",
    "mov.b64 { lo, hi }, $2;",
    "shfl.sync.bfly.b32 lo, lo, $1, 0x1F, 0xFFFFFFFF;",
    "shfl.sync.bfly.b32 hi, hi, $1, 0x1F, 0xFFFFFFFF;",
    "mov.b64 $0, { lo, hi };",
    "}") : "=d"(out) : "r"(xor),"d"(v));
    out*/
    let [lo, hi]: [i32; 2] = core::mem::transmute(v);
    let lo = shfl_bfly_sync_i32(0xFFFFFFFFu32, lo, xor, 0x1F);
    let hi = shfl_bfly_sync_i32(0xFFFFFFFFu32, hi, xor, 0x1F);
    core::mem::transmute([lo, hi])
}

unsafe fn share(mut mean: f64, mut m2: f64, mut count: f64) -> [f64; 3] {
    for xor in [16, 8, 4, 2, 1].iter().copied() {
        // let shfled = bfly_shfl_f64_v3([mean, m2, count], xor);
        // let arr = parallel_welford_combine(mean, m2, count, shfled[0], shfled[1], shfled[2]);
        let arr = parallel_welford_combine(mean, m2, count, bfly_shfl_f64(mean, xor), bfly_shfl_f64(m2, xor), bfly_shfl_f64(count, xor));
        mean = arr[0];
        m2 = arr[1];
        count = arr[2];
    }
    [mean, m2, count]
}
