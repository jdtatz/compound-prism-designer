use crate::spectrometer::{
    kernel::*, ArrayFamily, Beam, CulminatingToricCompoundPrism, CurvedPlane, FastSimdVector,
    FiberBeam, FocusingPlanerCompoundPrism, GaussianBeam, GenericSpectrometer, Plane, SimpleVector,
    SliceFamily, Spectrometer, ToricLens, UniformDistribution, Welford,
};
use core::{arch::asm, cell::UnsafeCell, mem::MaybeUninit, ptr::NonNull};
use nvptx_sys::{
    blockDim, blockIdx, gridDim, laneid, syncthreads, threadIdx, vote_any, vote_ballot, warp_sync,
    warpsize, FastFloat, FastNum, Shuffle, ALL_MEMBER_MASK,
};

extern "C" {
    #[link_name = "llvm.nvvm.redux.sync.umin"]
    fn redux_sync_min_u32(src: u32, membermask: u32) -> u32;
}

const fn ilog2_u32(v: u32) -> u32 {
    (u32::BITS - 1) - v.leading_zeros()
}
const WARP_SIZE: u32 = 32;
const WARP_SIZE_LG2: u32 = ilog2_u32(WARP_SIZE);

struct CUDAGPU;
impl GPU for CUDAGPU {
    fn warp_size_log2() -> u32 {
        // ilog2_u32(warpsize())
        WARP_SIZE_LG2
    }
    fn thread_id() -> u32 {
        threadIdx::x()
    }
    fn block_dim() -> u32 {
        blockDim::x()
    }
    fn block_id() -> u32 {
        blockIdx::x()
    }
    fn grid_dim() -> u32 {
        gridDim::x()
    }

    fn lane_id() -> u32 {
        laneid()
    }

    fn sync_warp() {
        warp_sync(ALL_MEMBER_MASK)
    }
    fn warp_any(pred: bool) -> bool {
        vote_any(ALL_MEMBER_MASK, pred)
    }
    fn warp_ballot(pred: bool) -> u32 {
        vote_ballot(ALL_MEMBER_MASK, pred).count_ones()
    }
}

impl GPUShuffle<u32> for CUDAGPU {
    fn shfl_bfly_sync(val: u32, lane_mask: u32) -> u32 {
        Shuffle::shfl_bfly(val, ALL_MEMBER_MASK, lane_mask)
    }

    fn warp_min(val: u32) -> u32 {
        unsafe { redux_sync_min_u32(val, ALL_MEMBER_MASK) }
    }
}

impl GPUShuffle<f32> for CUDAGPU {
    fn shfl_bfly_sync(val: f32, lane_mask: u32) -> f32 {
        Shuffle::shfl_bfly(val, ALL_MEMBER_MASK, lane_mask)
    }
}

impl<F: Copy + Shuffle + FastNum> GPUShuffle<FastFloat<F>> for CUDAGPU {
    fn shfl_bfly_sync(val: FastFloat<F>, lane_mask: u32) -> FastFloat<F> {
        Shuffle::shfl_bfly(val, ALL_MEMBER_MASK, lane_mask)
    }
}

#[repr(transparent)]
pub struct StaticSyncWrapper<T>(UnsafeCell<MaybeUninit<T>>);

unsafe impl<T: Sync> Sync for StaticSyncWrapper<T> {}

impl<T> StaticSyncWrapper<T> {
    pub const UNINIT: Self = Self(UnsafeCell::new(MaybeUninit::uninit()));

    pub fn as_nonnull(&self) -> NonNull<T> {
        unsafe { NonNull::new_unchecked(self.0.get()) }.cast()
    }
}

extern "C" {
    #[address_space(3)]
    static DYN_SHARED: StaticSyncWrapper<[u128; 0]>;
}

macro_rules! gen_kernel {
    (@inner $fty:ty ; $v:ty ; $fname:ident $beam:ident $cmpnd:ident $d:literal) => {
        paste::paste! {
            #[no_mangle]
            pub unsafe extern "ptx-kernel" fn [<prob_dets_given_wavelengths_ $fname _ $beam:snake _ $cmpnd:snake>] (
                seed: $fty,
                max_evals: u32,
                spectrometer: &Spectrometer<$fty, $v, UniformDistribution<$fty>, $beam<$fty>, $cmpnd<$v, SliceFamily>>,
                prob: *mut $fty,
            ) {
                // let (ptr, _dyn_mem) = dynamic_shared_memory();
                // let shared_ptr: NonNull<Welford<$fty>> = ptr.cast();
                let shared_ptr: NonNull<Welford<$fty>> = DYN_SHARED.as_nonnull().cast();

                kernel::<CUDAGPU, _, _, _, $d>(seed, max_evals, spectrometer, NonNull::new_unchecked(prob), shared_ptr)
            }

            #[no_mangle]
            pub unsafe extern "ptx-kernel" fn [<propagation_test_kernel_ $fname _ $beam:snake _ $cmpnd:snake>] (
                spectrometer: &Spectrometer<$fty, $v, UniformDistribution<$fty>, $beam<$fty>, $cmpnd<$v, SliceFamily>>,
                wavelength_cdf_ptr: *const $fty,
                ray_cdf_ptr: *const <$beam<$fty> as Beam<$v, $d>>::Quasi,
                bin_index_ptr: *mut u32,
                probability_ptr: *mut $fty,
            ) {
                propagation_test_kernel::<CUDAGPU, _, _, _, $d>(spectrometer, NonNull::new_unchecked(wavelength_cdf_ptr as _), NonNull::new_unchecked(ray_cdf_ptr as _), NonNull::new_unchecked(bin_index_ptr), NonNull::new_unchecked(probability_ptr),)
            }
        }
    };
    (@inner $fty:ty ; $v:ty ; $fname:ident $beam:ident $cmpnd:ident $n:literal $d:literal) => {
        paste::paste! {
            #[no_mangle]
            pub unsafe extern "ptx-kernel" fn [<prob_dets_given_wavelengths_ $fname _ $beam:snake _ $cmpnd:snake _ $n>] (
                seed: $fty,
                max_evals: u32,
                spectrometer: &Spectrometer<$fty, $v, UniformDistribution<$fty>, $beam<$fty>, $cmpnd<$v, ArrayFamily<$n>>>,
                prob: *mut $fty,
            ) {
                // let (ptr, _dyn_mem) = dynamic_shared_memory();
                // let shared_ptr: NonNull<Welford<$fty>> = ptr.cast();
                let shared_ptr: NonNull<Welford<$fty>> = DYN_SHARED.as_nonnull().cast();

                kernel::<CUDAGPU, _, _, _, $d>(seed, max_evals, spectrometer, NonNull::new_unchecked(prob), shared_ptr)
            }

            #[no_mangle]
            pub unsafe extern "ptx-kernel" fn [<propagation_test_kernel_ $fname _ $beam:snake _ $cmpnd:snake _ $n>] (
                spectrometer: &Spectrometer<$fty, $v, UniformDistribution<$fty>, $beam<$fty>, $cmpnd<$v, ArrayFamily<$n>>>,
                wavelength_cdf_ptr: *const $fty,
                ray_cdf_ptr: *const <$beam<$fty> as Beam<$v, $d>>::Quasi,
                bin_index_ptr: *mut u32,
                probability_ptr: *mut $fty,
            ) {
                propagation_test_kernel::<CUDAGPU, _, _, _, $d>(spectrometer, NonNull::new_unchecked(wavelength_cdf_ptr as _), NonNull::new_unchecked(ray_cdf_ptr as _), NonNull::new_unchecked(bin_index_ptr), NonNull::new_unchecked(probability_ptr),)
            }
        }
    };
    ([$($n:literal),*]) => {
        gen_kernel!(@inner FastFloat<f32>; FastSimdVector<f32, 2> ; f32 GaussianBeam FocusingPlanerCompoundPrism 2);
        $( gen_kernel!(@inner FastFloat<f32>; FastSimdVector<f32, 2> ; f32 GaussianBeam FocusingPlanerCompoundPrism $n 2); )*
        gen_kernel!(@inner FastFloat<f32>; FastSimdVector<f32, 4> ; f32 FiberBeam    CulminatingToricCompoundPrism   3);
        $( gen_kernel!(@inner FastFloat<f32>; FastSimdVector<f32, 4> ; f32 FiberBeam    CulminatingToricCompoundPrism   $n 3); )*
        // gen_kernel!(@inner FastFloat<f32>; SimpleVector<FastFloat<f32>, 2> ; f32 GaussianBeam FocusingPlanerCompoundPrism 2);
        // $( gen_kernel!(@inner FastFloat<f32>; SimpleVector<FastFloat<f32>, 2> ; f32 GaussianBeam FocusingPlanerCompoundPrism $n 2); )*
        // gen_kernel!(@inner FastFloat<f32>; SimpleVector<FastFloat<f32>, 3> ; f32 FiberBeam    CulminatingToricCompoundPrism   3);
        // $( gen_kernel!(@inner FastFloat<f32>; SimpleVector<FastFloat<f32>, 3> ; f32 FiberBeam    CulminatingToricCompoundPrism   $n 3); )*
    };
}

gen_kernel!([0, 1, 2, 3, 4, 5, 6]);
