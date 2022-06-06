#![no_std]
#![cfg_attr(
    any(target_arch = "nvptx64", target_arch = "spirv"),
    feature(asm_experimental_arch, asm_const)
)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx))]
#![cfg_attr(target_arch = "spirv", feature(register_attr), register_attr(spirv))]

#[cfg(target_arch = "spirv")]
extern crate spirv_std;

#[cfg(target_arch = "nvptx64")]
mod kernel;
#[cfg(target_arch = "spirv")]
pub mod shader;
