use std::env::var_os;
use std::process::{Command, Stdio};
use std::path::{PathBuf, Path};
use std::fs::copy;

const RUSTFLAGS: &str = concat!(
    "-Ccodegen-units=1 ",
    "-Clto ",
    "-Ctarget-cpu=sm_52 ",
    "-Ctarget-feature=+ptx60 ",
    "-Cllvm-args=--lto-embed-bitcode ",
    "-Cdefault-linker-libraries=no "
);

fn main() {
    let is_cuda = var_os("CARGO_FEATURE_CUDA").is_some();
    if is_cuda {
        let cargo = var_os("CARGO").unwrap();
        let out_dir_os = var_os("OUT_DIR").unwrap();
        let out_dir: &Path = out_dir_os.as_ref();
        let kernel_path = PathBuf::from(out_dir).join("kernel.ptx");

        let cmd = Command::new(cargo)
            .arg("rustc")
            .arg("--verbose")
            .arg("--manifest-path=compound_prism_designer_kernel/Cargo.toml")
            .arg("--release")
            .arg("--target=nvptx64-nvidia-cuda.json")
            .arg("-Z").arg("build-std=core")
            .arg("-Z").arg("features=all")
            .arg("--")
            .arg(format!("--emit=asm={}", kernel_path.display()))
            // .arg("--emit=asm=\"kernel.ptx\"")
            // .arg(format!("--out-dir=\"{}\"", out_dir.display()))
            .env("RUSTFLAGS", RUSTFLAGS)
            .stderr(Stdio::inherit())
            .output()
            .expect("cargo rustc kernel failed");
        if !cmd.status.success() {
            panic!("cargo rustc build kernel failed");
        }
        println!("cargo:rustc-env=KERNEL={}", kernel_path.display());
        let target = var_os("CARGO_MANIFEST_DIR").unwrap();
        let kpath = PathBuf::from(&target).join("kernel.ptx");
        copy(kernel_path, kpath).unwrap();
    }
}
