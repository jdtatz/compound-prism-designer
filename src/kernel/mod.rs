#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(
    target_arch = "nvptx64",
    feature(abi_ptx, asm_experimental_arch, asm_const)
)]

#[cfg(target_arch = "nvptx64")]
mod kernel;
