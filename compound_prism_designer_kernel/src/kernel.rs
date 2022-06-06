use compound_prism_spectrometer::{
    kernel::{kernel, GPUShuffle, GPU},
    CurvedPlane, FiberBeam, GaussianBeam, Plane, Spectrometer, ToricLens, UniformDistribution,
    Welford,
};
use core::{arch::asm, ptr::NonNull};
use nvptx_sys::{
    blockDim, blockIdx, dynamic_shared_memory, gridDim, syncthreads, threadIdx, vote_any,
    vote_ballot, warp_sync, FastFloat, FastNum, Shuffle, ALL_MEMBER_MASK,
};

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

struct CUDAGPU;
impl GPU for CUDAGPU {
    fn warp_size() -> u32 {
        32
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

// FIXME: orphan rules will be the death of me
// impl<T: Copy + Shuffle> GPUShuffle<T> for CUDAGPU {
//     fn shfl_bfly_sync(val: T, lane_mask: u32) -> T { Shuffle::shfl_bfly(val, ALL_MEMBER_MASK, lane_mask) }
// }

impl GPUShuffle<u32> for CUDAGPU {
    fn shfl_bfly_sync(val: u32, lane_mask: u32) -> u32 {
        Shuffle::shfl_bfly(val, ALL_MEMBER_MASK, lane_mask)
    }
}
impl GPUShuffle<f32> for CUDAGPU {
    fn shfl_bfly_sync(val: f32, lane_mask: u32) -> f32 {
        Shuffle::shfl_bfly(val, ALL_MEMBER_MASK, lane_mask)
    }
}
impl<F: FastNum + Shuffle> GPUShuffle<FastFloat<F>> for CUDAGPU {
    fn shfl_bfly_sync(val: FastFloat<F>, lane_mask: u32) -> FastFloat<F> {
        Shuffle::shfl_bfly(val, ALL_MEMBER_MASK, lane_mask)
    }
}

macro_rules! gen_kernel {
    (@inner $fty:ty ; $fname:ident $beam:ident $s0:ident $sn:ident $n:literal $d:literal) => {
        paste::paste! {
            #[no_mangle]
            pub unsafe extern "ptx-kernel" fn [<prob_dets_given_wavelengths_ $fname _ $beam:snake _ $s0:snake _ $sn:snake _ $n _ $d d>] (
                seed: $fty,
                max_evals: u32,
                spectrometer: &Spectrometer<$fty, UniformDistribution<$fty>, $beam<$fty>, $s0<$fty, $d>, Plane<$fty, $d>, $sn<$fty, $d>, $n, $d>,
                prob: *mut $fty,
            ) {
                let shared_spectrometer_ptr;
                asm!(
                    ".shared .align 16 .b8 shared_spectrometer[{size}]; cvta.shared.u64 {ptr}, shared_spectrometer;",
                    ptr = out(reg64) shared_spectrometer_ptr,
                    size = const core::mem::size_of::<Spectrometer<$fty, UniformDistribution<$fty>, $beam<$fty>, $s0<$fty, $d>, Plane<$fty, $d>, $sn<$fty, $d>, $n, $d>>(),
                    options(readonly, nostack, preserves_flags)
                );
                let spectrometer = cuda_memcpy_1d(shared_spectrometer_ptr, spectrometer);

                let (ptr, _dyn_mem) = dynamic_shared_memory();
                let shared_ptr: NonNull<Welford<$fty>> = ptr.cast();

                kernel::<CUDAGPU, _, _, _, _, _, _, $n, $d>(seed, max_evals, spectrometer, NonNull::new_unchecked(prob), shared_ptr)
            }
        }
    };
    ([$($n:literal),*]) => {
        $( gen_kernel!(@inner FastFloat<f32>; f32 GaussianBeam Plane     CurvedPlane $n 2); )*
        $( gen_kernel!(@inner FastFloat<f32>; f32 FiberBeam    ToricLens ToricLens   $n 3); )*
    };
}

gen_kernel!([0, 1, 2, 3, 4, 5, 6]);
