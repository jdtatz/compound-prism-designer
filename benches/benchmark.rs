#[macro_use]
extern crate criterion;

use compound_prism_designer::*;

use criterion::profiler::Profiler;
use criterion::Criterion;
use std::path::Path;

use cpuprofiler::PROFILER;

struct GProf;

impl Profiler for GProf {
    fn start_profiling(&mut self, benchmark_id: &str, benchmark_dir: &Path) {
        std::fs::create_dir_all(benchmark_dir).unwrap();
        let profile = benchmark_dir.join(format!("{}.profile", benchmark_id));
        if profile.exists() {
            std::fs::remove_file(&profile).unwrap();
        }
        PROFILER
            .lock()
            .unwrap()
            .start(profile.to_string_lossy().into_owned())
            .unwrap();
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &Path) {
        PROFILER.lock().unwrap().stop().unwrap();

        let cwd = std::env::current_dir().unwrap();
        let fdir = cwd.join("flamegraphs");
        std::fs::create_dir_all(&fdir).unwrap();
        let flamegraph = fdir.join(format!("{}.svg", benchmark_id));
        if flamegraph.exists() {
            std::fs::remove_file(&flamegraph).unwrap();
        }
        let flamegraph = std::fs::File::create(flamegraph).unwrap();
        let binary = std::env::current_exe().unwrap();
        let profile = benchmark_dir.join(format!("{}.profile", benchmark_id));

        let pprof = std::process::Command::new("google-pprof")
            .arg("--collapsed")
            .arg(&binary)
            .arg(&profile)
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("Failed to process profiled benchmarks with google-pprof");
        let _inferno = std::process::Command::new("inferno-flamegraph")
            .stdin(pprof.stdout.unwrap())
            .stdout(flamegraph)
            .output()
            .expect("Failed to create flamegraphs of profiled benchmarks with inferno-flamegraph");
    }
}

fn profiled() -> Criterion {
    Criterion::default().with_profiler(GProf)
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let glasses = [
        &Glass::Sellmeier1([
            1.029607,
            0.00516800155,
            0.1880506,
            0.0166658798,
            0.736488165,
            138.964129,
        ]),
        &Glass::Sellmeier1([
            1.87543831,
            0.0141749518,
            0.37375749,
            0.0640509927,
            2.30001797,
            177.389795,
        ]),
        &Glass::Sellmeier1([
            0.738042712,
            0.00339065607,
            0.363371967,
            0.0117551189,
            0.989296264,
            212.842145,
        ]),
    ];
    let angles = [-27.2712308, 34.16326141, -42.93207009, 1.06311416];
    let angles: Box<[f64]> = angles.iter().cloned().map(f64::to_radians).collect();
    let lengths = [0_f64; 3];
    let prism = CompoundPrism::new(
        glasses.iter().copied(),
        angles.as_ref().into(),
        lengths.as_ref().into(),
        0.21,
        2.5,
        2.,
    );

    const NBIN: usize = 32;
    let pmt_length = 3.2;
    let bounds: Box<[_]> = (0..=NBIN)
        .map(|i| (i as f64) / (NBIN as f64) * pmt_length)
        .collect();
    let bins: Box<[_]> = bounds.windows(2).map(|t| [t[0], t[1]]).collect();
    let spec_max_accepted_angle = (60_f64).to_radians();
    let detarr = DetectorArray::new(
        bins.as_ref().into(),
        spec_max_accepted_angle.cos(),
        0.,
        pmt_length,
    );

    let beam = GaussianBeam {
        width: 0.2,
        y_mean: 0.95,
        w_range: (0.5, 0.82),
    };

    c.bench_function("known_design_example", |b| {
        b.iter(|| fitness(&prism, &detarr, &beam));
    });
}

criterion_group! {
    name = benches;
    config = profiled();
    targets = criterion_benchmark
}
criterion_main!(benches);
