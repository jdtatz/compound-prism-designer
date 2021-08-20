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
        use pprof::protos::Message;
        std::fs::create_dir_all(benchmark_dir).unwrap();
        let profile_path = benchmark_dir.join(format!("{}.pb", benchmark_id));
        if profile_path.exists() {
            std::fs::remove_file(&profile_path).unwrap();
        }

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
        std::fs::write(profile_path, &content).unwrap();
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
    // "N-PK52A", "N-SF57", "N-FK58"
    let glasses = [
        Glass {
            coefficents: [
                -0.19660238,
                0.85166448,
                -1.49929414,
                1.35438084,
                -0.64424681,
                1.62434799,
            ],
        },
        Glass {
            coefficents: [
                -1.81746234,
                7.71730927,
                -13.2402884,
                11.56821078,
                -5.23836004,
                2.82403194,
            ],
        },
        Glass {
            coefficents: [
                -0.15938247,
                0.69081086,
                -1.21697038,
                1.10021121,
                -0.52409733,
                1.55979703,
            ],
        },
    ];
    let [glass0, glasses @ ..] = glasses;
    let angles = [-27.2712308, 34.16326141, -42.93207009, 1.06311416];
    let angles = angles.map(f32::to_radians);
    let [first_angle, angles @ .., last_angle] = angles;
    let lengths = [0_f32; 3];
    let [first_length, lengths @ ..] = lengths;
    let height = 2.5;
    let width = 2.0;
    let prism = CompoundPrism::new(
        glass0,
        glasses,
        first_angle,
        angles,
        last_angle,
        first_length,
        lengths,
        PlaneParametrization { height, width },
        CurvedPlaneParametrization {
            signed_normalized_curvature: 0.21,
            height,
        },
        height,
        width,
        false,
    );

    let wavelengths = UniformDistribution {
        bounds: (0.5, 0.82),
    };
    let beam = GaussianBeam {
        width: 0.2,
        y_mean: 0.95,
    };

    const NBIN: usize = 32;
    let pmt_length = 3.2;
    let spec_max_accepted_angle = (60_f32).to_radians();
    let det_angle = 0.0;
    let (det_pos, det_flipped) =
        detector_array_positioning(prism, pmt_length, det_angle, wavelengths, &beam)
            .expect("This is a valid spectrometer design.");
    let detarr = LinearDetectorArray::new(
        NBIN as u32,
        0.1,
        0.1,
        0.0,
        spec_max_accepted_angle.cos(),
        0.,
        pmt_length,
        det_pos,
        det_flipped,
    );
    // dbg!((&wavelengths, &beam, &prism, &detarr));
    let spec = Spectrometer {
        wavelengths,
        beam,
        compound_prism: prism,
        detector: detarr,
    };
    let cpu_fitness = fitness(&spec, 16_384, 16_384);
    let gpu_fitness = cuda_fitness(&spec, &[0.798713], 256, 2, 16_384)
        .unwrap()
        .unwrap();
    float_eq::assert_float_eq!(
        cpu_fitness.info as f64,
        gpu_fitness.info as f64,
        rmax <= 1e-2
    );

    c.bench_function("known_design_example", |b| {
        b.iter(|| fitness(&spec, 16_384, 16_384));
    });
    c.bench_function("cuda_known_design_example", |b| {
        b.iter(|| {
            cuda_fitness(&spec, &[0.798713], 256, 2, 16_384)
                .unwrap()
                .unwrap()
        })
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
