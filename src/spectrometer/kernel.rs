use crate::{FloatExt, GenericSpectrometer, Qrng, Vector, Welford};
use core::ptr::NonNull;
use core::slice::from_raw_parts_mut;

pub trait GPU {
    /// Must be in the range `[0, 7]` as derived from the [GPU::warp_size] requirements
    fn warp_size_log2() -> u32;
    /// It must be a power-of-two & in the range `[1, 128]` [see the vulkan spec](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/SubgroupSize.html)
    /// * `WARP_SZ` in CUDA
    /// * `SubgroupSize` in SPIR-V
    fn warp_size() -> u32 {
        1 << Self::warp_size_log2()
    }
    /// * `threadIdx.x` in CUDA
    /// * `LocalInvocationId.x` or `LocalInvocationIndex` in SPIR-V
    fn thread_id() -> u32;
    /// * `blockDim.x` in CUDA
    /// * `WorkgroupSize.x`(deprecated) or `LocalSizeId.x` in SPIR-V
    fn block_dim() -> u32;
    /// * `blockIdx.x` in CUDA
    /// * `WorkgroupId.x` in SPIR-V
    fn block_id() -> u32;
    /// * `gridDim.x` in CUDA
    /// * `NumWorkgroups.x` in SPIR-V
    fn grid_dim() -> u32;

    /// `SubgroupLocalInvocationId` in SPIR-V
    fn lane_id() -> u32 {
        Self::thread_id() % Self::warp_size()
    }
    /// `SubgroupId` in SPIR-V
    fn warp_id() -> u32 {
        Self::thread_id() >> Self::warp_size_log2()
    }
    /// `SubgroupsPerWorkgroup` or `NumSubgroups` in SPIR-V
    fn nwarps() -> u32 {
        Self::block_dim() >> Self::warp_size_log2()
    }
    fn global_warp_id() -> u32 {
        Self::block_id() * Self::nwarps() + Self::warp_id()
    }
    /// `GlobalInvocationId.x` in SPIR-V
    fn global_thread_id() -> u32 {
        Self::block_id() * Self::block_dim() + Self::thread_id()
    }

    /// * `__syncwarp()` in CUDA
    /// * `OpControlBarrier` in SPIR-V
    fn sync_warp();
    /// * `__any_sync()` in CUDA
    /// * `OpGroupAny` or `OpGroupNonUniformAny` in SPIR-V
    fn warp_any(pred: bool) -> bool;
    /// * `__ballot_sync($pred).count_ones()`
    /// * `OpGroupNonUniformBallotBitCount(OpGroupNonUniformBallot())` in SPIR-V
    fn warp_ballot(pred: bool) -> u32;
}

pub trait GPUShuffle<T: Copy>: GPU {
    /// * `__shfl_xor_sync` in CUDA
    /// * `OpGroupNonUniformShuffleXor` in SPIR-V
    fn shfl_bfly_sync(val: T, lane_mask: u32) -> T;
    fn warp_fold<Op: Fn(T, T) -> T>(mut val: T, fold: Op) -> T {
        let mut xor = Self::warp_size_log2();
        while xor > 0 {
            xor -= 1;
            val = fold(val, Self::shfl_bfly_sync(val, 1 << xor));
        }
        val
    }
    /// * `__reduce_min_sync` in CUDA (compute capability >=8.x)
    /// * `OpGroupUMin` or `OpGroupNonUniformUMin` in SPIR-V
    fn warp_min(val: T) -> T
    where
        T: core::cmp::Ord,
    {
        Self::warp_fold(val, core::cmp::min)
    }
    /// * `__reduce_add_sync` in CUDA (compute capability >=8.x & `u32` only)
    /// * `OpGroupIAdd` / `OpGroupFAdd` or `OpGroupNonUniformIAdd` / `OpGroupNonUniformFAdd` in SPIR-V
    fn warp_sum(val: T) -> T
    where
        T: core::ops::Add<Output = T>,
    {
        Self::warp_fold(val, core::ops::Add::add)
    }
}

pub unsafe fn kernel<
    G: GPUShuffle<F> + GPUShuffle<u32>,
    F: FloatExt,
    V: Vector<D, Scalar = F>,
    GS: ?Sized + GenericSpectrometer<V, D>,
    const D: usize,
>(
    seed: F,
    max_evals: u32,
    spectrometer: &GS,
    probability_ptr: NonNull<F>,
    shared_ptr: NonNull<Welford<F>>,
) {
    if max_evals == 0 {
        return;
    }
    const MAX_ERR: f64 = 5e-3;
    const MAX_ERR_SQR: f64 = MAX_ERR * MAX_ERR;
    const PHI: f64 = 1.61803398874989484820458683436563;
    const ALPHA: f64 = 1_f64 / PHI;

    let warpid = G::warp_id();
    let laneid = G::lane_id();

    let nbin = spectrometer.detector_bin_count();
    let shared = core::ptr::slice_from_raw_parts_mut(
        shared_ptr.as_ptr().add((warpid * nbin) as usize),
        nbin as usize,
    );
    let mut i = laneid;
    while i < nbin {
        (*shared)[i as usize] = Welford::new();
        i += G::warp_size();
    }
    G::sync_warp();
    let shared = &mut *shared;

    let id = G::global_warp_id();

    let u = (F::lossy_from(id))
        .mul_add(F::lossy_from(ALPHA), seed)
        .fract();
    let wavelength = spectrometer.sample_wavelength(u);

    let mut count = F::zero();
    let mut index = laneid;
    let mut qrng = Qrng::new_from_scalar(seed);
    qrng.next_by(laneid);
    let max_evals = max_evals - (max_evals % G::warp_size());
    while index < max_evals {
        count += F::lossy_from(G::warp_size());
        let q = qrng.next_by(G::warp_size());
        let ray = spectrometer.sample_ray(q);
        let (mut bin_index, t) = spectrometer
            .propagate(ray, wavelength)
            .unwrap_or((nbin, F::zero()));
        let mut det_count = G::warp_ballot(bin_index < nbin);
        let mut finished = det_count > 0;
        while det_count > 0 {
            let min_index = G::warp_min(bin_index);
            if min_index >= nbin {
                core::hint::unreachable_unchecked()
            }
            det_count -= G::warp_ballot(bin_index == min_index);
            let welford = {
                let bin_t = if bin_index == min_index { t } else { F::zero() };

                let warp_mean = G::warp_sum(bin_t) / F::lossy_from(G::warp_size());
                let warp_m2 = G::warp_sum((bin_t - warp_mean).sqr());
                let warp_welford = Welford {
                    count: F::lossy_from(G::warp_size()),
                    mean: warp_mean,
                    m2: warp_m2,
                };
                let shared_welford = shared[min_index as usize];
                shared_welford + warp_welford
            };
            G::sync_warp();
            if laneid == 0 {
                shared[min_index as usize] = welford;
            }
            finished = finished && welford.sem_le_error_threshold(F::lossy_from(MAX_ERR_SQR));
            if bin_index == min_index {
                bin_index = nbin;
            }
        }
        // ensure convergence in the case of non-associative behavior in the floating point finished result
        if G::warp_any(finished) {
            break;
        }
        index += G::warp_size();
    }
    G::sync_warp();
    let probability = from_raw_parts_mut(
        probability_ptr.as_ptr().add((nbin * id) as usize),
        nbin as usize,
    );
    let mut i = laneid;
    while i < nbin {
        let w = &mut shared[i as usize];
        w.skip(count);
        probability[i as usize] = w.mean;
        i += G::warp_size();
    }
}

pub unsafe fn propagation_test_kernel<
    G: GPU,
    F: FloatExt,
    V: Vector<D, Scalar = F>,
    GS: ?Sized + GenericSpectrometer<V, D>,
    const D: usize,
>(
    spectrometer: &GS,
    wavelength_cdf_ptr: NonNull<F>,
    ray_cdf_ptr: NonNull<GS::Q>,
    bin_index_ptr: NonNull<u32>,
    probability_ptr: NonNull<F>,
) {
    let nbin = spectrometer.detector_bin_count();

    let id = G::global_thread_id();

    let u = wavelength_cdf_ptr.as_ptr().add(id as _).read();
    let wavelength = spectrometer.sample_wavelength(u);

    let q = ray_cdf_ptr.as_ptr().add(id as _).read();
    let ray = spectrometer.sample_ray(q);

    let (bin_index, t) = spectrometer
        .propagate(ray, wavelength)
        .unwrap_or((nbin, F::zero()));

    bin_index_ptr.as_ptr().add(id as _).write(bin_index);
    probability_ptr.as_ptr().add(id as _).write(t);
}
