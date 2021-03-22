#![no_std]
#![no_main]
#![cfg(target_arch = "nvptx64")]
#![feature(abi_ptx, link_llvm_intrinsics, asm)]

use compound_prism_spectrometer::{
    Beam, DetectorArray, Float, GaussianBeam, Pair, Qrng, Spectrometer, UniformDistribution,
    Vector, Welford,
};
use core::slice::from_raw_parts_mut;
use nvptx_sys::{
    blockDim, blockIdx, dynamic_shared_memory, syncthreads, threadIdx, Shuffle, ALL_MEMBER_MASK,
};

extern "C" {
    #[link_name = "llvm.nvvm.vote.ballot.sync"]
    fn vote_ballot_sync(lane_mask: u32, pred: bool) -> u32;
    #[link_name = "llvm.nvvm.bar.warp.sync"]
    fn warp_sync(membermask: u32);
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
        Shuffle::shfl_bfly(self, ALL_MEMBER_MASK, lane_mask)
    }
}

impl Shareable for u32 {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self {
        core::mem::transmute(core::mem::transmute::<u32, i32>(self).shfl_bfly_sync(lane_mask))
    }
}

impl Shareable for f32 {
    unsafe fn shfl_bfly_sync(self, lane_mask: u32) -> Self {
        Shuffle::shfl_bfly(self, ALL_MEMBER_MASK, lane_mask)
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
        let lo = lo.shfl_bfly_sync(lane_mask);
        let hi = hi.shfl_bfly_sync(lane_mask);
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
    #[allow(clippy::cast_ptr_alignment)]
    let dest = dest as *mut u32;
    let src = src as *const T as *const u32;
    let count = (core::mem::size_of::<T>() / core::mem::size_of::<u32>()) as u32;
    let mut id = threadIdx::x() as u32;
    while id < count {
        dest.add(id as usize)
            .write_volatile(src.add(id as usize).read_volatile());
        id += blockDim::x() as u32;
    }
    syncthreads();
    &*(dest as *const T)
}

unsafe fn kernel<F: CudaFloat, V: Vector<Scalar = F>, B: Beam<Vector = V>, const N: usize>(
    seed: F,
    max_evals: u32,
    spectrometer: &Spectrometer<V, B, N>,
    probability_ptr: *mut F,
) {
    if max_evals == 0 {
        return;
    }
    const MAX_ERR: f64 = 5e-3;
    const MAX_ERR_SQR: f64 = MAX_ERR * MAX_ERR;
    const PHI: f64 = 1.61803398874989484820458683436563;
    const ALPHA: f64 = 1_f64 / PHI;

    let tid = threadIdx::x() as u32;
    let warpid = tid / 32;
    let nwarps = blockDim::x() as u32 / 32;
    let laneid = nvptx_sys::laneid() as u32;
    let shared_spectrometer_ptr;
    asm!(
        ".shared .align 16 .b8 shared_spectrometer[{size}]; cvta.shared.u64 {ptr}, shared_spectrometer;",
        ptr = out(reg64) shared_spectrometer_ptr,
        size = const core::mem::size_of::<Spectrometer<V, B, N>>(),
        options(readonly, nostack, preserves_flags)
    );
    let spectrometer = cuda_memcpy_1d(shared_spectrometer_ptr, spectrometer);

    let (ptr, _dyn_mem) = dynamic_shared_memory();
    let ptr = ptr as *mut Welford<F>;

    let nbin = spectrometer.detector.bin_count();
    let shared = from_raw_parts_mut(ptr.add((warpid * nbin) as usize), nbin as usize);
    let mut i = laneid;
    while i < nbin {
        shared[i as usize] = Welford::new();
        i += 32;
    }
    warp_sync(ALL_MEMBER_MASK);

    let id = (blockIdx::x() as u32) * nwarps + warpid;

    let u = (F::from_u32(id)).mul_add(F::from_f64(ALPHA), seed).fract();
    let wavelength = spectrometer.beam.inverse_cdf_wavelength(u);

    let mut count = F::zero();
    let mut index = laneid;
    let mut qrng = Qrng::new_from_scalar(seed);
    qrng.next_by(laneid);
    let max_evals = max_evals - (max_evals % 32);
    while index < max_evals {
        let q = qrng.next_by(32);
        let ray = spectrometer.beam.inverse_cdf_ray(q);
        let (mut bin_index, t) = ray
            .propagate(
                wavelength,
                &spectrometer.compound_prism,
                &spectrometer.detector,
            )
            .unwrap_or((nbin, F::zero()));
        let mut det_count = warp_ballot(bin_index < nbin);
        let mut finished = det_count > 0;
        while det_count > 0 {
            let min_index = warp_fold(bin_index, core::cmp::min);
            if min_index >= nbin {
                core::hint::unreachable_unchecked()
            }
            det_count -= warp_ballot(bin_index == min_index);
            let bin_t = if bin_index == min_index { t } else { F::zero() };
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
            warp_sync(ALL_MEMBER_MASK);
            finished = finished && welford.sem_le_error_threshold(F::from_f64(MAX_ERR_SQR));
            if bin_index == min_index {
                bin_index = nbin;
            }
        }
        // ensure convergence in the case of non-associative behavior in the floating point finished result
        if warp_ballot(finished) > 0 {
            break;
        }
        index += 32;
        count += F::from_u32(32);
    }
    warp_sync(ALL_MEMBER_MASK);
    let probability = from_raw_parts_mut(probability_ptr.add((nbin * id) as usize), nbin as usize);
    let mut i = laneid;
    while i < nbin {
        let w = &mut shared[i as usize];
        w.skip(count);
        probability[i as usize] = w.mean;
        i += 32;
    }
}

macro_rules! gen_kernel {
    (@inner $fty:ident $n:expr) => {
        paste::paste! {
            #[no_mangle]
            pub unsafe extern "ptx-kernel" fn  [<prob_dets_given_wavelengths_ $fty _ $n>] (
                seed: $fty,
                max_evals: u32,
                spectrometer: &Spectrometer<Pair<$fty>, GaussianBeam<$fty, UniformDistribution<$fty>>, $n>,
                prob: *mut $fty,
            ) {
                kernel(seed, max_evals, spectrometer, prob)
            }
        }
    };
    ([$($n:expr),*]) => {
        $( gen_kernel!(@inner f32 $n); )*
        $( gen_kernel!(@inner f64 $n); )*
    };
}

gen_kernel!([1, 2, 3, 4, 5, 6]);
