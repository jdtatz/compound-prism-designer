#![cfg_attr(target_arch = "nvptx64", no_std)]
#![cfg_attr(target_arch = "nvptx64", feature(abi_ptx, asm))]

#[cfg(target_arch = "nvptx64")]
mod kernel;
