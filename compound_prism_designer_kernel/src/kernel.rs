#![no_std]
#![no_main]
#![cfg(target_arch = "nvptx64")]
#![feature(abi_ptx, asm)]

use compound_prism_spectrometer::{
    Beam, CurvedPlane, DetectorArray, Float, GaussianBeam, Pair, Plane, Qrng, Spectrometer,
    Surface, UniformDistribution, Vector, Welford,
};
use core::slice::from_raw_parts_mut;
use nvptx_sys::{
    blockDim, blockIdx, dynamic_shared_memory, syncthreads, threadIdx, vote_any, vote_ballot,
    warp_sync, Shuffle, ALL_MEMBER_MASK,
};

unsafe fn warp_ballot(pred: bool) -> u32 {
    vote_ballot(ALL_MEMBER_MASK, pred).count_ones()
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
        Shuffle::shfl_bfly(self, ALL_MEMBER_MASK, lane_mask)
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
    let mut id = threadIdx::x();
    while id < count {
        dest.add(id as usize)
            .write_volatile(src.add(id as usize).read_volatile());
        id += blockDim::x();
    }
    syncthreads();
    &*(dest as *const T)
}

unsafe fn kernel<
    F: CudaFloat,
    V: Vector<Scalar = F>,
    B: Beam<Vector = V>,
    S0: Surface<V>,
    SI: Surface<V>,
    SN: Surface<V>,
    const N: usize,
>(
    seed: F,
    max_evals: u32,
    spectrometer: &Spectrometer<V, B, S0, SI, SN, N>,
    probability_ptr: *mut F,
) {
    if max_evals == 0 {
        return;
    }
    const MAX_ERR: f64 = 5e-3;
    const MAX_ERR_SQR: f64 = MAX_ERR * MAX_ERR;
    const PHI: f64 = 1.61803398874989484820458683436563;
    const ALPHA: f64 = 1_f64 / PHI;

    let tid = threadIdx::x();
    let warpid = tid / 32;
    let nwarps = blockDim::x() / 32;
    let laneid = nvptx_sys::laneid();

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

    let id = (blockIdx::x()) * nwarps + warpid;

    let u = (F::from_u32(id)).mul_add(F::from_f64(ALPHA), seed).fract();
    let wavelength = spectrometer.beam.inverse_cdf_wavelength(u);

    let mut count = F::zero();
    let mut index = laneid;
    let mut qrng = Qrng::new_from_scalar(seed);
    qrng.next_by(laneid);
    let max_evals = max_evals - (max_evals % 32);
    while index < max_evals {
        count += F::from_u32(32);
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
            let mut welford = if laneid == 0 {
                shared[min_index as usize]
            } else {
                Welford::NEW
            };
            welford.next_sample(bin_t);
            welford = warp_fold(welford, |w1, w2| w1 + w2);
            welford.skip(count);
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
        if vote_any(ALL_MEMBER_MASK, finished) {
            break;
        }
        index += 32;
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
                spectrometer: &Spectrometer<Pair<$fty>, GaussianBeam<$fty, UniformDistribution<$fty>>, Plane<Pair<$fty>>, Plane<Pair<$fty>>, CurvedPlane<Pair<$fty>>, $n>,
                prob: *mut $fty,
            ) {
                let shared_spectrometer_ptr;
                asm!(
                    ".shared .align 16 .b8 shared_spectrometer[{size}]; cvta.shared.u64 {ptr}, shared_spectrometer;",
                    ptr = out(reg64) shared_spectrometer_ptr,
                    size = const core::mem::size_of::<Spectrometer<Pair<$fty>, GaussianBeam<$fty, UniformDistribution<$fty>>, Plane<Pair<$fty>>, Plane<Pair<$fty>>, CurvedPlane<Pair<$fty>>, $n>>(),
                    options(readonly, nostack, preserves_flags)
                );
                let spectrometer = cuda_memcpy_1d(shared_spectrometer_ptr, spectrometer);

                kernel(seed, max_evals, spectrometer, prob)
            }
        }
    };
    ([$($n:expr),*]) => {
        $( gen_kernel!(@inner f32 $n); )*
        $( gen_kernel!(@inner f64 $n); )*
    };
}

gen_kernel!([0, 1, 2, 3, 4, 5, 6]);
