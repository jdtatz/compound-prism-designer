use compound_prism_spectrometer::{
    kernel::{kernel, GPUShuffle, GPU},
    CurvedPlane, FiberBeam, GaussianBeam, Plane, Spectrometer, ToricLens, UniformDistribution,
    Welford,
};
use core::mem::MaybeUninit;
use core::ptr::NonNull;
use spirv_std::vector::Vector;
use spirv_std::{arch::control_barrier, memory::Scope, memory::Semantics, scalar::Scalar};
use spirv_std::{glam::UVec3, integer::UnsignedInteger};

#[repr(u32)]
pub enum GroupOperation {
    Reduce = 0,
    InclusiveScan = 1,
    ExclusiveScan = 2,
    ClusteredReduce = 3,
    PartitionedReduceNV = 6,
    PartitionedInclusiveScanNV = 7,
    PartitionedExclusiveScanNV = 8,
}

// #[cfg(target_feature = "Groups")]
// #[doc(alias = "OpGroupAny")]
// fn group_any<const SCOPE: u32>(predicate: bool) -> bool {
//     let mut result = false;
//     unsafe {
//         asm! {
//             "%execution = OpConstant _ {execution}",
//             "%predicate = OpLoad _ {predicate}",
//             "%result = OpGroupAny %execution %predicate",
//             "OpStore {result} %result",
//             execution = const SCOPE,
//             predicate = in(reg) &predicate,
//             result = in(reg) &mut result
//         }
//     }
//     result
// }

// #[cfg(target_feature = "GroupNonUniformBallot")]
// #[doc(alias = "OpGroupNonUniformBallot")]
// fn group_non_uniform_ballot<T: UnsignedInteger, V: Vector<T, 4>, const SCOPE: u32>(predicate: bool) -> V {
//     let mut result = MaybeUninit::uninit();
//     unsafe {
//         asm! {
//             "%predicate = OpLoad _ %{predicate}",
//             "%result = OpGroupNonUniformBallot _ %{execution} %predicate",
//             "OpStore %{result} %result",
//             execution = const SCOPE,
//             predicate = in(reg) &predicate,
//             result = in(reg) result.as_mut_ptr(),
//         }
//     }
//     unsafe { result.assume_init() }
// }

// #[cfg(target_feature = "GroupNonUniformBallot")]
// #[doc(alias = "OpGroupNonUniformBallotBitCount")]
// fn group_non_uniform_ballot_bit_count<TV: UnsignedInteger, V: Vector<TV, 4>, TO: UnsignedInteger, const SCOPE: u32, const GROUP_OP: u32>(bitfields: V) -> TO {
//     let mut result = MaybeUninit::uninit();
//     unsafe {
//         asm! {
//             "%bitfields = OpLoad _ %{bitfields}",
//             "%result = OpGroupNonUniformBallotBitCount _ %{execution} %{operation} %bitfields",
//             "OpStore %{result} %result",
//             execution = const SCOPE,
//             operation = const GROUP_OP,
//             bitfields = in(reg) &bitfields,
//             result = in(reg) result.as_mut_ptr(),
//         }
//     }
//     unsafe { result.assume_init() }
// }

// fn group_ballot_count<T: UnsignedInteger, const SCOPE: u32, const GROUP_OP: u32>(predicate: bool) -> T {
//     let mut result = MaybeUninit::uninit();
//     unsafe {
//         asm! {
//             "%predicate = OpLoad _ %{predicate}",
//             "%bitfields = OpGroupNonUniformBallot _ %{execution} %predicate",
//             "%result = OpGroupNonUniformBallotBitCount _ %{execution} %{operation} %bitfields",
//             "OpStore %{result} %result",
//             execution = const SCOPE,
//             operation = const GROUP_OP,
//             predicate = in(reg) &predicate,
//             result = in(reg) result.as_mut_ptr(),
//         }
//     }
//     unsafe { result.assume_init() }
// }

// #[cfg(target_feature = "GroupNonUniformShuffle")]
// #[doc(alias = "OpGroupNonUniformShuffleXor")]
// fn group_non_uniform_shuffle_xor<T: Scalar, U: UnsignedInteger, const SCOPE: u32>(value: T, mask: U) -> T {
//     let mut result = MaybeUninit::uninit();
//     unsafe {
//         asm! {
//             "%value = OpLoad _ %{value}",
//             "%mask = OpLoad _ %{mask}",
//             "%result = OpGroupNonUniformShuffleXor _ %{execution} %value %mask",
//             "OpStore %{result} %result",
//             execution = const SCOPE,
//             value = in(reg) &value,
//             mask = in(reg) &mask,
//             result = in(reg) result.as_mut_ptr(),
//         }
//     }
//     unsafe { result.assume_init() }
// }

// #[cfg(target_feature = "Groups")]
// #[doc(alias = "OpGroupUMin")]
// fn group_umin<T: UnsignedInteger, const SCOPE: u32, const GROUP_OP: u32>(value: T) -> T {
//     let mut result = MaybeUninit::uninit();
//     unsafe {
//         asm! {
//             "%value = OpLoad _ {value}",
//             "%result = OpGroupUMin _ {execution} {operation} %value",
//             "OpStore {result} %result",
//             execution = const SCOPE,
//             operation = const GROUP_OP,
//             value = in(reg) &value,
//             result = in(reg) result.as_mut_ptr(),
//         }
//     }
//     unsafe { result.assume_init() }
// }

// https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#_a_id_group_a_group_and_subgroup_instructions
struct SPIRVGPU;
impl GPU for SPIRVGPU {
    const ZERO_INITIALIZED_SHARED_MEMORY: bool = true;

    fn warp_size() -> u32 {
        // SubgroupSize
        // Decorating a variable with the SubgroupSize builtin decoration will make that variable contain the implementation-dependent number of invocations in a subgroup. This value must be a power-of-two integer.
        // The maximum number of invocations that an implementation can support per subgroup is 128.
        // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#interfaces-builtin-variables-sgs

        32
    }
    // fn thread_id() -> u32 {
    //     // LocalInvocationId / LocalInvocationIndex
    //     threadIdx::x()
    // }
    // fn block_dim() -> u32 {
    //     // WorkgroupSize
    //     blockDim::x()
    // }
    // fn block_id() -> u32 {
    //     // WorkgroupId
    //     blockIdx::x()
    // }
    // fn grid_dim() -> u32 {
    //     // NumWorkgroups
    //     gridDim::x()
    // }

    // fn lane_id() -> u32 {
    //     // SubgroupLocalInvocationId = index of the invocation within the subgroup
    // }
    // fn warp_id() -> u32 {
    //     // SubgroupId = index of the subgroup within the local workgroup
    // }
    // fn nwarps() -> u32 {
    //     // NumSubgroups = number of subgroups in the local workgroup
    // }

    fn sync_warp() {
        // OpControlBarrier
        unsafe {
            control_barrier::<
                { Scope::Subgroup as u32 },
                { Scope::Subgroup as u32 },
                { Semantics::SUBGROUP_MEMORY.bits() | Semantics::WORKGROUP_MEMORY.bits() },
            >()
        }
    }
    fn warp_any(pred: bool) -> bool {
        // Which?
        // OpSubgroupAnyKHR
        // OpGroupAny
        // OpGroupNonUniformAny
        let mut result = false;
        unsafe {
            asm! {
                "%bool = OpTypeBool",
                "%u32 = OpTypeInt 32 0",
                "%execution = OpConstant %u32 {execution}",
                "%predicate = OpLoad _ {predicate}",
                "%result = OpGroupAny typeof*{result} %execution %predicate",
                "OpStore {result} %result",
                execution = const {Scope::Subgroup as u32},
                predicate = in(reg) &pred,
                result = in(reg) &mut result,
                options(nostack, nomem)
            }
        }
        result
    }
    fn warp_ballot(pred: bool) -> u32 {
        // OpSubgroupBallotKHR
        // OpGroupNonUniformBallot / OpGroupNonUniformBallotBitCount
        let result;
        unsafe {
            asm! {
                "%bool = OpTypeBool",
                "%u32 = OpTypeInt 32 0",
                "%u32vec4 = OpTypeVector %u32 4",
                "%execution = OpConstant %u32 {execution}",
                "%predicate = OpLoad _ {predicate}",
                "%bitfields = OpGroupNonUniformBallot %u32vec4 %execution %predicate",
                "{result} = OpGroupNonUniformBallotBitCount %u32 %execution Reduce %bitfields",
                execution = const {Scope::Subgroup as u32},
                // operation = const {GroupOperation::Reduce as u32},
                predicate = in(reg) &pred,
                result = out(reg) result,
                options(pure, nomem)
            }
        }
        result
    }
}

// https://www.khronos.org/registry/SPIR-V/specs/unified1/SPIRV.html#OpGroupNonUniformBroadcast

// OpGroupNonUniformShuffleXor
macro_rules! warp_shuffle_xor {
    ($value:ident, $mask:ident) => {{
        let result;
        unsafe {
            asm! {
                "%u32 = OpTypeInt 32 0",
                "%execution = OpConstant %u32 {execution}",
                "{result} = OpGroupNonUniformShuffleXor _ %execution {value} {mask}",
                execution = const {Scope::Subgroup as u32},
                value = in(reg) $value,
                mask = in(reg) $mask,
                result = out(reg) result,
                options(pure, nomem)
            }
        }
        result
    }};
}

impl GPUShuffle<u32> for SPIRVGPU {
    fn shfl_bfly_sync(val: u32, lane_mask: u32) -> u32 {
        warp_shuffle_xor!(val, lane_mask)
    }

    fn warp_min(val: u32) -> u32 {
        // OpGroupUMin
        // OpGroupNonUniformUMin
        // group_umin::<_, {Scope::Subgroup as u32}, {GroupOperation::Reduce as u32}>(val)
        let result;
        unsafe {
            asm! {
                "%u32 = OpTypeInt 32 0",
                "%execution = OpConstant %u32 {execution}",
                "{result} = OpGroupUMin %u32 %execution Reduce {value}",
                execution = const {Scope::Subgroup as u32},
                value = in(reg) val,
                result = out(reg) result,
                options(pure, nomem)
            }
        }
        result
    }
}
impl GPUShuffle<f32> for SPIRVGPU {
    fn shfl_bfly_sync(val: f32, lane_mask: u32) -> f32 {
        // group_non_uniform_shuffle_xor::<_, _, {Scope::Subgroup as u32}>(val, lane_mask)
        warp_shuffle_xor!(val, lane_mask)
    }
}

// #[spirv(compute(threads(128), entry_point_name="kernel"))]
// pub fn kernel0(
//     #[spirv(workgroup)] shared: &mut [Welford<f32>],
//     #[spirv(subgroup_local_invocation_id)] laneid: u32,
//     #[spirv(subgroup_size)] warp_size: u32,
//     #[spirv(num_subgroups)] nwarps: u32,
//     #[spirv(local_invocation_index)] thread_id: u32,

//     #[spirv(workgroup_id)] block_id_vec: UVec3,
// ) {
//     let warp_id = thread_id / warp_size;
//     let block_id = block_id_vec[0];
//     let global_warp_id = block_id * nwarps + warp_id;
// }

macro_rules! gen_kernel {
    (@inner $fname:ident $beam:ident $s0:ident $sn:ident $n:literal $d:literal) => {
        paste::paste! {
            #[no_mangle]
            #[spirv(compute(threads(128)))]
            pub fn [<prob_dets_given_wavelengths_ $fname _ $beam:snake _ $s0:snake _ $sn:snake _ $n _ $d d>] (
                seed: $fname,
                max_evals: u32,
                prob: *mut $fname,
                spectrometer: *const Spectrometer<$fname, UniformDistribution<$fname>, $beam<$fname>, $s0<$fname, $d>, Plane<$fname, $d>, $sn<$fname, $d>, $n, $d>,
                #[spirv(workgroup)] shared_ptr: &mut [Welford<$fname>; 64],

                #[spirv(subgroup_local_invocation_id)] laneid: u32,
                #[spirv(subgroup_size)] warp_size: u32,
                #[spirv(num_subgroups)] nwarps: u32,
                #[spirv(local_invocation_index)] thread_id: u32,

                #[spirv(workgroup_id)] block_id_vec: UVec3,
            ) {
                let warp_id = thread_id / warp_size;
                let block_id = block_id_vec[0];
                let global_warp_id = block_id * nwarps + warp_id;

                unsafe {
                    kernel::<SPIRVGPU, _, _, _, _, _, _, $n, $d>(seed, max_evals, &*spectrometer, NonNull::new_unchecked(prob), NonNull::new_unchecked(shared_ptr.as_mut_ptr()), warp_size, laneid, warp_id, global_warp_id)
                }
            }
        }
    };
    ([$($n:literal),*]) => {
        $( gen_kernel!(@inner f32 GaussianBeam Plane     CurvedPlane $n 2); )*
        $( gen_kernel!(@inner f32 FiberBeam    ToricLens ToricLens   $n 3); )*
    };
}

// gen_kernel!([0, 1, 2, 3, 4, 5, 6]);
gen_kernel!(@inner f32 GaussianBeam Plane     CurvedPlane 1 2);
