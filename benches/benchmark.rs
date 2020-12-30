#![allow(clippy::excessive_precision)]
#[macro_use]
extern crate criterion;
use criterion::Criterion;

use compound_prism_designer::*;
use compound_prism_spectrometer::*;

#[cfg(unix)]
struct GProf(Option<pprof::ProfilerGuard<'static>>);

#[cfg(unix)]
impl criterion::profiler::Profiler for GProf {
    fn start_profiling(&mut self, benchmark_id: &str, benchmark_dir: &std::path::Path) {
        self.0.replace(pprof::ProfilerGuard::new(100).unwrap());
    }

    fn stop_profiling(&mut self, benchmark_id: &str, benchmark_dir: &std::path::Path) {
        use std::io::Write;
        use pprof::protos::Message;
        std::fs::create_dir_all(benchmark_dir).unwrap();
        let profile_path = benchmark_dir.join(format!("{}.pb", benchmark_id));
        if profile_path.exists() {
            std::fs::remove_file(&profile_path).unwrap();
        }
        let profile_file = std::fs::File::create(profile_path).unwrap();

        let cwd = std::env::current_dir().unwrap();
        let fdir = cwd.join("flamegraphs");
        std::fs::create_dir_all(&fdir).unwrap();
        let flamegraph_path = fdir.join(format!("{}.svg", benchmark_id));
        if flamegraph_path.exists() {
            std::fs::remove_file(&flamegraph_path).unwrap();
        }
        let flamegraph_file = std::fs::File::create(flamegraph_path).unwrap();

        let report = self.0.take().unwrap().report().build().unwrap();
        let profile = report.pprof().unwrap();
        let mut content = Vec::new();
        profile.encode(&mut content).unwrap();
        profile_file.write_all(&content).unwrap();

        report.flamegraph(flamegraph_file).unwrap();
    }
}

fn profiled() -> Criterion {
    #[cfg(unix)]
    {
        Criterion::default().with_profiler(GProf(None))
    }
    #[cfg(not(unix))]
    {
        Criterion::default()
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let glasses = [
        Glass::Sellmeier1([
            1.029607,
            0.00516800155,
            0.1880506,
            0.0166658798,
            0.736488165,
            138.964129,
        ]),
        Glass::Sellmeier1([
            1.87543831,
            0.0141749518,
            0.37375749,
            0.0640509927,
            2.30001797,
            177.389795,
        ]),
        Glass::Sellmeier1([
            0.738042712,
            0.00339065607,
            0.363371967,
            0.0117551189,
            0.989296264,
            212.842145,
        ]),
    ];
    let angles = [-27.2712308, 34.16326141, -42.93207009, 1.06311416];
    let angles: Box<[f32]> = angles.iter().cloned().map(f32::to_radians).collect();
    let lengths = [0_f32; 3];
    let prism = CompoundPrism::new(
        glasses.iter().cloned(),
        angles.as_ref(),
        lengths.as_ref(),
        0.21,
        2.5,
        2.,
        false,
    );

    const NBIN: usize = 32;
    let pmt_length = 3.2;
    let spec_max_accepted_angle = (60_f32).to_radians();
    let detarr = LinearDetectorArray::new(
        NBIN as u32,
        0.1,
        0.1,
        0.0,
        spec_max_accepted_angle.cos(),
        0.,
        pmt_length,
    );

    let beam = GaussianBeam {
        width: 0.2,
        y_mean: 0.95,
        w_range: (0.5, 0.82),
    };
    let spec = Spectrometer::<Pair<f32>, _>::new(beam, prism, detarr).unwrap();
    let cpu_fitness = fitness(&spec);
    let gpu_fitness = cuda_fitness(&spec).unwrap();
    assert_almost_eq!(cpu_fitness.info as f64, gpu_fitness.info as f64, 1e-2);

    c.bench_function("known_design_example", |b| {
        b.iter(|| fitness(&spec));
    });
    c.bench_function("cuda_known_design_example", |b| {
        b.iter(|| cuda_fitness(&spec).unwrap())
    });
    println!("cpu: {:?}", cpu_fitness);
    println!("gpu: {:?}", gpu_fitness);
}

criterion_group! {
    name = benches;
    config = profiled();
    targets = criterion_benchmark
}
criterion_main!(benches);
