use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::C)
        .with_no_includes()
        .exclude_item("GlassCatalogState")
        .exclude_item("ProbabilitiesState")
        .exclude_item("TracedRayState")
        .with_header(
            r#"
typedef void GlassCatalogState;
typedef void ProbabilitiesState;
typedef void TracedRayState;
"#,
        )
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("target/bindings.h");
}
