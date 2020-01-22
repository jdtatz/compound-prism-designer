use crate::utils::{Float, Welford};
use crate::Spectrometer;
use core::{arch::nvptx::*, panic::PanicInfo, slice::from_raw_parts_mut};

#[panic_handler]
unsafe fn panic_handle(p: &PanicInfo) -> ! {
    let (file, line) = if let Some(loc) = p.location() {
        (loc.file().as_ptr(), loc.line())
    } else {
        (b"".as_ptr(), 0)
    };
    if let Some(s) = p.payload().downcast_ref::<&str>() {
        __assert_fail(s.as_ptr(), file, line, b"".as_ptr());
    } else {
        __assert_fail(b"Panicked".as_ptr(), file, line, b"".as_ptr());
    }
    trap()
}

extern "C" {
    #[link_name = "llvm.nvvm.ptr.shared.to.gen"]
    fn ptr_shared_to_gen(shared_ptr: *mut core::ffi::c_void) -> *mut core::ffi::c_void;

    #[link_name = "llvm.nvvm.read.ptx.sreg.laneid"]
    fn read_ptx_sreg_laneid() -> u32;

    #[link_name = "llvm.nvvm.shfl.sync.bfly.f32"]
    fn shfl_bfly_sync_f32(mask: u32, val: f32, lane_mask: u32, packing: u32) -> f32;

    #[link_name = "llvm.nvvm.shfl.sync.bfly.i32"]
    fn shfl_bfly_sync_i32(mask: u32, val: i32, lane_mask: u32, packing: u32) -> i32;
}

trait CudaFloat: Float {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self;
}

impl CudaFloat for f32 {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self {
        shfl_bfly_sync_f32(0xFFFFFFFFu32, self, lane_mask, 0x1F)
    }
}

impl CudaFloat for f64 {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self {
        /*let out: f64;
        asm!(concat!("{ .reg .b32 lo, hi;",
        "mov.b64 { lo, hi }, $2;",
        "shfl.sync.bfly.b32 lo, lo, $1, 0x1F, 0xFFFFFFFF;",
        "shfl.sync.bfly.b32 hi, hi, $1, 0x1F, 0xFFFFFFFF;",
        "mov.b64 $0, { lo, hi };",
        "}") : "=d"(out) : "r"(xor),"d"(self));
        out*/
        let [lo, hi]: [i32; 2] = core::mem::transmute(self);
        let lo = shfl_bfly_sync_i32(0xFFFFFFFFu32, lo, lane_mask, 0x1F);
        let hi = shfl_bfly_sync_i32(0xFFFFFFFFu32, hi, lane_mask, 0x1F);
        core::mem::transmute([lo, hi])
    }
}

unsafe fn share_welford<F: CudaFloat>(welford: &mut Welford<F>) {
    for xor in [16, 8, 4, 2, 1].iter().copied() {
        let other = Welford {
            count: welford.count.shfl_bfly_sync(xor),
            mean: welford.mean.shfl_bfly_sync(xor),
            m2: welford.m2.shfl_bfly_sync(xor),
        };
        welford.combine(other);
    }
}

unsafe fn cuda_memcpy_1d<T>(dest: *mut u8, src: &T) -> &T {
    let dest = dest as *mut u32;
    let src = src as *const T as *const u32;
    let count = (core::mem::size_of::<T>() / core::mem::size_of::<u32>()) as u32;
    // let count = (core::mem::size_of::<T>()) as u32;
    let mut id = _thread_idx_x() as u32;
    while id < count {
        dest.add(id as usize)
            .write_volatile(src.add(id as usize).read_volatile());
        id += _block_dim_x() as u32;
    }
    &*(dest as *const T)
}

unsafe fn kernel<F: CudaFloat>(
    seed: F,
    max_evals: u32,
    spectrometer: &Spectrometer<F>,
    prob: *mut F,
) {
    if max_evals == 0 {
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

    let ptr: *mut u8 = ptr_shared_to_gen(0 as _) as _;
    let spec_ptr = ptr;
    let prob_ptr = spec_ptr.add(core::mem::size_of::<Spectrometer<F>>());
    let ptr = prob_ptr as *mut [F; 2];

    let spectrometer = cuda_memcpy_1d(spec_ptr, spectrometer);
    _syncthreads();
    let nbin = spectrometer.detector_array.bin_count;
    let shared = from_raw_parts_mut(ptr.add((warpid * nbin) as usize), nbin as usize);
    if laneid == 0 {
        for [mean, m2] in shared.iter_mut() {
            *mean = F::zero();
            *m2 = F::zero();
        }
    }

    let id = (_block_idx_x() as u32) * nwarps + warpid;

    let u = (F::from_f64(id as f64))
        .mul_add(F::from_f64(ALPHA), seed)
        .fract();
    let wavelength = spectrometer.gaussian_beam.inverse_cdf_wavelength(u);

    let mut count = F::zero();
    let mut index = laneid;
    let mut qrng_n = F::from_f64(laneid as f64);
    let max_evals = max_evals - (max_evals % 32);
    while index < max_evals {
        let u = qrng_n.mul_add(F::from_f64(ALPHA), seed).fract();
        let y0 = spectrometer.gaussian_beam.inverse_cdf_initial_y(u);
        let result = spectrometer.propagate(wavelength, y0);
        let prev_count = count;
        let mut finished = true;
        for ([lb, ub], [shared_mean, shared_m2]) in
            spectrometer.detector_array.bounds().zip(shared.iter_mut())
        {
            let mut mean = match result {
                Ok((pos, t)) if lb <= pos && pos < ub => t,
                _ => F::zero(),
            };
            let mut welford = if laneid == 0 {
                let mut w = Welford {
                    count: prev_count,
                    mean: *shared_mean,
                    m2: *shared_m2,
                };
                w.next_sample(mean);
                w
            } else {
                Welford {
                    count: F::one(),
                    mean,
                    m2: F::zero(),
                }
            };
            share_welford(&mut welford);
            if laneid == 0 {
                count = welford.count;
                *shared_mean = welford.mean;
                *shared_m2 = welford.m2;
            }
            finished = finished && welford.sem_le_error_threshold(F::from_f64(MAX_ERR_SQR));
        }
        if finished {
            break;
        }
        index += 32;
        qrng_n += F::from_f64(32.);
    }
    let prob = from_raw_parts_mut(prob.add((nbin * id) as usize), nbin as usize);
    if laneid == 0 {
        for (p, [mean, _m2]) in prob.iter_mut().zip(shared.iter().copied()) {
            *p = mean;
        }
    }
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn prob_dets_given_wavelengths(
    seed: f32,
    max_evals: u32,
    spectrometer: &Spectrometer<f32>,
    prob: *mut f32,
) {
    kernel(seed, max_evals, spectrometer, prob)
}

#[no_mangle]
pub unsafe extern "ptx-kernel" fn prob_dets_given_wavelengths_f64(
    seed: f64,
    max_evals: u32,
    spectrometer: &Spectrometer<f64>,
    prob: *mut f64,
) {
    kernel(seed, max_evals, spectrometer, prob)
}
