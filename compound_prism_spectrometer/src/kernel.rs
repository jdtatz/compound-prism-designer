use crate::{
    Beam, DetectorArray, Distribution, Float, FloatExt, Qrng, Spectrometer, Surface, Welford,
};
use core::ptr::NonNull;
use core::slice::from_raw_parts_mut;

pub trait GPU {
    const ZERO_INITIALIZED_SHARED_MEMORY: bool;
    fn warp_size() -> u32;
    // fn thread_id() -> u32;
    // fn block_dim() -> u32;
    // fn block_id() -> u32;
    // fn grid_dim() -> u32;

    // fn lane_id() -> u32 {
    //     Self::thread_id() % Self::warp_size()
    // }
    // fn warp_id() -> u32 {
    //     Self::thread_id() / Self::warp_size()
    // }
    // fn nwarps() -> u32 {
    //     Self::block_dim() / Self::warp_size()
    // }
    // fn global_warp_id() -> u32 {
    //     Self::block_id() * Self::nwarps() + Self::warp_id()
    // }

    fn sync_warp();
    fn warp_any(pred: bool) -> bool;
    fn warp_ballot(pred: bool) -> u32;
}

pub trait GPUShuffle<T: Copy>: GPU {
    fn shfl_bfly_sync(val: T, lane_mask: u32) -> T;
    fn warp_fold<Op: Fn(T, T) -> T>(mut val: T, fold: Op) -> T {
        let mut xor = Self::warp_size();
        while xor > 1 {
            xor >>= 1;
            val = fold(val, Self::shfl_bfly_sync(val, xor));
        }
        val
    }
    fn warp_min(val: T) -> T
    where
        T: core::cmp::Ord,
    {
        Self::warp_fold(val, core::cmp::min)
    }
}

impl<F: Copy, G: GPUShuffle<F>> GPUShuffle<Welford<F>> for G {
    fn shfl_bfly_sync(val: Welford<F>, lane_mask: u32) -> Welford<F> {
        Welford {
            count: G::shfl_bfly_sync(val.count, lane_mask),
            mean: G::shfl_bfly_sync(val.mean, lane_mask),
            m2: G::shfl_bfly_sync(val.m2, lane_mask),
        }
    }
}

// #[inline(always)]
// pub unsafe fn get_warp_shared<T>(shared_ptr: NonNull<()>, warp_id: u32, n: u32) -> NonNull<[T]> {
//     unsafe {
//         NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(
//             shared_ptr.cast::<T>().as_ptr().add((warp_id * n) as usize),
//             n as usize,
//         ))
//     }
// }

// #[inline(always)]
// pub unsafe fn get_warp_global<T>(global_ptr: NonNull<()>, global_warp_id: u32, n: u32) -> NonNull<[T]> {
//     unsafe {
//         NonNull::new_unchecked(core::ptr::slice_from_raw_parts_mut(
//             global_ptr.cast::<T>().as_ptr().add((global_warp_id * n) as usize),
//             n as usize,
//         ))
//     }
// }

#[inline(always)]
pub unsafe fn kernel<
    G: GPUShuffle<F> + GPUShuffle<u32>,
    F: FloatExt,
    W: Copy + Distribution<F, Output = F>,
    B: Copy + Beam<F, D>,
    S0: Copy + Surface<F, D>,
    SI: Copy + Surface<F, D>,
    SN: Copy + Surface<F, D>,
    const N: usize,
    const D: usize,
>(
    seed: F,
    max_evals: u32,
    spectrometer: &Spectrometer<F, W, B, S0, SI, SN, N, D>,
    probability_ptr: NonNull<F>,
    shared_ptr: NonNull<Welford<F>>,
    warp_size: u32,
    lane_id: u32,
    warp_id: u32,
    global_warp_id: u32,
) {
    if max_evals == 0 {
        return;
    }
    const MAX_ERR: f64 = 5e-3;
    const MAX_ERR_SQR: f64 = MAX_ERR * MAX_ERR;
    const PHI: f64 = 1.61803398874989484820458683436563;
    const ALPHA: f64 = 1_f64 / PHI;

    let nbin = spectrometer.detector.bin_count();
    let shared = core::ptr::slice_from_raw_parts_mut(
        shared_ptr.as_ptr().add((warp_id * nbin) as usize),
        nbin as usize,
    );
    if !G::ZERO_INITIALIZED_SHARED_MEMORY {
        let mut i = lane_id;
        while i < nbin {
            (*shared)[i as usize] = Welford::NEW;
            i += G::warp_size();
        }
        G::sync_warp();
    }
    let shared = &mut *shared;

    let u = (F::lossy_from(global_warp_id))
        .mul_add(F::lossy_from(ALPHA), seed)
        .fract();
    let wavelength = spectrometer.wavelengths.inverse_cdf(u);
    let initial_refractive_index = spectrometer.compound_prism.initial_glass.calc_n(wavelength);
    // let refractive_indicies = spectrometer.compound_prism.glasses.map(|glass| glass.calc_n(wavelength));
    let mut refractive_indicies = [F::ZERO; N];
    for i in 0..N {
        refractive_indicies[i] = spectrometer.compound_prism.glasses[i].calc_n(wavelength);
    }

    let mut count = F::zero();
    let mut index = lane_id;
    let mut qrng = Qrng::new_from_scalar(seed);
    qrng.next_by(lane_id);
    let max_evals = max_evals - (max_evals % G::warp_size());
    while index < max_evals {
        count += F::lossy_from(G::warp_size());
        let q = qrng.next_by(G::warp_size());
        let ray = spectrometer.beam.inverse_cdf(q);
        let (mut bin_index, t) = ray
            .propagate(
                initial_refractive_index,
                refractive_indicies,
                spectrometer.compound_prism.initial_surface,
                spectrometer.compound_prism.inter_surfaces,
                spectrometer.compound_prism.final_surface,
                spectrometer.compound_prism.ar_coated,
                &spectrometer.detector,
            )
            .unwrap_or((nbin, F::zero()));
        let mut det_count = G::warp_ballot(bin_index < nbin);
        let mut finished = det_count > 0;
        while det_count > 0 {
            let min_index = G::warp_min(bin_index);
            if min_index >= nbin {
                core::hint::unreachable_unchecked()
            }
            det_count -= G::warp_ballot(bin_index == min_index);
            let bin_t = if bin_index == min_index { t } else { F::zero() };
            let mut welford = if lane_id == 0 {
                shared[min_index as usize]
            } else {
                Welford::NEW
            };
            welford.next_sample(bin_t);
            welford = G::warp_fold(welford, |w1, w2| w1 + w2);
            welford.skip(count);
            if lane_id == 0 {
                shared[min_index as usize] = welford;
            }
            G::sync_warp();
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
        probability_ptr
            .as_ptr()
            .add((nbin * global_warp_id) as usize),
        nbin as usize,
    );
    let mut i = lane_id;
    while i < nbin {
        let w = &mut shared[i as usize];
        w.skip(count);
        probability[i as usize] = w.mean;
        i += G::warp_size();
    }
}
