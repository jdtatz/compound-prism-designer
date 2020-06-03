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

    #[link_name = "llvm.nvvm.vote.ballot.sync"]
    fn vote_ballot_sync(lane_mask: u32, pred: bool) -> u32;
}

unsafe fn warp_ballot(pred: bool) -> u32 {
    vote_ballot_sync(0xFFFFFFFFu32, pred).count_ones()
}

trait Shareable: 'static + Copy + Sized + Send {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self;
}

trait CudaFloat: Shareable + Float {}

impl<T: Shareable + Float> CudaFloat for T {}

impl Shareable for i32 {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self {
        shfl_bfly_sync_i32(0xFFFFFFFFu32, self, lane_mask, 0x1F)
    }
}

impl Shareable for u32 {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self {
        core::mem::transmute(shfl_bfly_sync_i32(
            0xFFFFFFFFu32,
            core::mem::transmute(self),
            lane_mask,
            0x1F,
        ))
    }
}

impl Shareable for f32 {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self {
        shfl_bfly_sync_f32(0xFFFFFFFFu32, self, lane_mask, 0x1F)
    }
}

impl Shareable for f64 {
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

impl<F: CudaFloat> Shareable for Welford<F> {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self {
        Self {
            count: self.count.shfl_bfly_sync(lane_mask),
            mean: self.mean.shfl_bfly_sync(lane_mask),
            m2: self.m2.shfl_bfly_sync(lane_mask),
        }
    }
}

unsafe fn warp_fold<F: Shareable, Op: Fn(F, F) -> F>(mut val: F, fold: Op) -> F {
    for xor in [16, 8, 4, 2, 1].iter().copied() {
        val = fold(val, val.shfl_bfly_sync(xor));
    }
    val
}

unsafe fn cuda_memcpy_1d<T>(dest: *mut u8, src: &T) -> &T {
    let dest = dest as *mut u32;
    let src = src as *const T as *const u32;
    let count = (core::mem::size_of::<T>() / core::mem::size_of::<u32>()) as u32;
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
    let ptr = prob_ptr as *mut Welford<F>;

    let spectrometer = cuda_memcpy_1d(spec_ptr, spectrometer);
    _syncthreads();
    let nbin = spectrometer.detector_array.bin_count;
    let shared = from_raw_parts_mut(ptr.add((warpid * nbin) as usize), nbin as usize);
    let mut i = laneid;
    while i < nbin {
        shared[i as usize] = Welford::new();
        i += 32;
    }

    let id = (_block_idx_x() as u32) * nwarps + warpid;

    let u = (F::from_u32(id)).mul_add(F::from_f64(ALPHA), seed).fract();
    let wavelength = spectrometer.gaussian_beam.inverse_cdf_wavelength(u);

    let mut count = F::zero();
    let mut index = laneid;
    let mut qrng_n = F::from_u32(laneid);
    let max_evals = max_evals - (max_evals % 32);
    while index < max_evals {
        let u = qrng_n.mul_add(F::from_f64(ALPHA), seed).fract();
        let y0 = spectrometer.gaussian_beam.inverse_cdf_initial_y(u);
        let (mut bin_index, t) = if let Ok((pos, t)) = spectrometer.propagate(wavelength, y0) {
            (spectrometer.detector_array.bin_index(pos), t)
        } else {
            (None, F::zero())
        };
        let mut det_count = warp_ballot(bin_index.is_some());
        let mut finished = det_count > 0;
        while det_count > 0 {
            let min_index = warp_fold(bin_index.unwrap_or(nbin), core::cmp::min);
            if min_index >= nbin {
                core::hint::unreachable_unchecked()
            }
            det_count -= warp_ballot(bin_index.map_or(false, |i| i == min_index));
            let bin_t = if bin_index.map_or(false, |i| i == min_index) {
                t
            } else {
                F::zero()
            };
            let welford = if laneid == 0 {
                let mut w = shared[min_index as usize];
                w.skip(count);
                w.next_sample(bin_t);
                w
            } else {
                Welford {
                    count: F::one(),
                    mean: bin_t,
                    m2: F::zero(),
                }
            };
            let welford = warp_fold(welford, |mut w1, w2| {
                w1.combine(w2);
                w1
            });
            if laneid == 0 {
                shared[min_index as usize] = welford;
            }
            finished = finished && welford.sem_le_error_threshold(F::from_f64(MAX_ERR_SQR));
            if bin_index.map_or(false, |i| i == min_index) {
                drop(bin_index.take());
            }
        }
        if finished {
            break;
        }
        index += 32;
        count += F::from_f64(32.);
        qrng_n += F::from_f64(32.);
    }
    let prob = from_raw_parts_mut(prob.add((nbin * id) as usize), nbin as usize);
    let mut i = laneid;
    while i < nbin {
        let w = &mut shared[i as usize];
        w.skip(count);
        prob[i as usize] = w.mean;
        i += 32;
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
