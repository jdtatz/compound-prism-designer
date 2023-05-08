use std::ffi::CStr;
use std::mem::{self, ManuallyDrop};
use std::ptr::Pointee;
use std::slice;

use parking_lot::{const_mutex, Mutex};
use rustacuda::context::ContextStack;
use rustacuda::function::Function;
use rustacuda::memory::{DeviceBuffer, DeviceCopy, DevicePointer};
use rustacuda::prelude::*;
use rustacuda::{launch, quick_init};

use crate::fitness::DesignFitness;
use crate::spectrometer::*;

macro_rules! cstr {
    ($s:expr) => {
        unsafe { CStr::from_bytes_with_nul_unchecked(concat!($s, "\0").as_bytes()) }
    };
}

macro_rules! include_cstr {
    ($p:literal) => {
        unsafe { &*(concat!(include_str!($p), "\0").as_bytes() as *const [u8] as *const CStr) }
    };
}

// FIXME: error[E0080]: evaluation of constant value failed: exceeded interpreter step limit (see `#[const_eval_limit]`) inside `CStr::from_bytes_with_nul_unchecked::const_impl`
// const PTX: &CStr = cstr!(include_str!("kernel.ptx"));
const PTX: &CStr = include_cstr!("kernel.ptx");

// const MAX_N: usize = 256;
// const MAX_M: usize = 16_384;
// const NWARP: u32 = 2;

pub trait Kernel<V: Vector<DIM>, const DIM: usize>:
    DeviceCopy + GenericSpectrometer<V, DIM>
{
    const NAME: &'static CStr;

    unsafe fn launch(
        function: Function,
        grid: u32,
        block: u32,
        shared_memory_size: u32,
        stream: &Stream,
        seed: V::Scalar,
        max_eval: u32,
        dev_spec: DevicePointer<u8>,
        meta: <Self as Pointee>::Metadata,
        dev_probs: DevicePointer<V::Scalar>,
    ) -> rustacuda::error::CudaResult<()>;
}

pub trait PropagationKernel<V: Vector<DIM>, const DIM: usize>:
    DeviceCopy + GenericSpectrometer<V, DIM>
{
    const NAME: &'static CStr;

    unsafe fn launch(
        function: Function,
        grid: u32,
        block: u32,
        shared_memory_size: u32,
        stream: &Stream,
        dev_spec: DevicePointer<u8>,
        meta: <Self as Pointee>::Metadata,
        dev_wavelength_cdf: DevicePointer<V::Scalar>,
        dev_ray_cdf: DevicePointer<Self::Q>,
        dev_bin_index: DevicePointer<u32>,
        dev_probability: DevicePointer<V::Scalar>,
    ) -> rustacuda::error::CudaResult<()>;
}

macro_rules! kernel_impl {
    (@inner $fty:ty ; $v:ty ; $fname:ident $beam:ident $cmpnd:ident $d:literal) => {
        paste::paste! {
            impl Kernel<$v, $d> for Spectrometer<$fty, $v, UniformDistribution<$fty>, $beam<$fty>, $cmpnd<$v, SliceFamily>> {
                const NAME: &'static CStr = cstr!(stringify!([<prob_dets_given_wavelengths_ $fname _ $beam:snake _ $cmpnd:snake>]));

                unsafe fn launch(function: Function, grid: u32, block: u32, shared_memory_size: u32, stream: &Stream, seed: $fty, max_eval: u32, dev_spec: DevicePointer<u8>, meta: <Self as Pointee>::Metadata, dev_probs: DevicePointer<$fty>) -> rustacuda::error::CudaResult<()> {
                    launch!(function<<<grid, block, shared_memory_size, stream>>>(
                        seed,
                        max_eval,
                        dev_spec,
                        meta,
                        dev_probs
                    ))
                }

            }
            impl PropagationKernel<$v, $d> for Spectrometer<$fty, $v, UniformDistribution<$fty>, $beam<$fty>, $cmpnd<$v, SliceFamily>> {
                const NAME: &'static CStr = cstr!(stringify!([<propagation_test_kernel_ $fname _ $beam:snake _ $cmpnd:snake>]));

                unsafe fn launch(
                    function: Function,
                    grid: u32,
                    block: u32,
                    shared_memory_size: u32,
                    stream: &Stream,
                    dev_spec: DevicePointer<u8>,
                    meta: <Self as Pointee>::Metadata,
                    dev_wavelength_cdf: DevicePointer<$fty>,
                    dev_ray_cdf: DevicePointer<Self::Q>,
                    dev_bin_index: DevicePointer<u32>,
                    dev_probability: DevicePointer<$fty>,
                ) -> rustacuda::error::CudaResult<()> {
                    launch!(function<<<grid, block, shared_memory_size, stream>>>(
                        dev_spec,
                        meta,
                        dev_wavelength_cdf,
                        dev_ray_cdf,
                        dev_bin_index,
                        dev_probability
                    ))
                }
            }
        }
    };
    (@inner $fty:ty ; $v:ty ; $fname:ident $beam:ident $cmpnd:ident $n:literal $d:literal) => {
        paste::paste! {
            impl Kernel<$v, $d> for Spectrometer<$fty, $v, UniformDistribution<$fty>, $beam<$fty>, $cmpnd<$v, ArrayFamily<$n>>> {
                const NAME: &'static CStr = cstr!(stringify!([<prob_dets_given_wavelengths_ $fname _ $beam:snake _ $cmpnd:snake _ $n>]));

                unsafe fn launch(function: Function, grid: u32, block: u32, shared_memory_size: u32, stream: &Stream, seed: $fty, max_eval: u32, dev_spec: DevicePointer<u8>, _meta: <Self as Pointee>::Metadata, dev_probs: DevicePointer<$fty>) -> rustacuda::error::CudaResult<()> {
                    launch!(function<<<grid, block, shared_memory_size, stream>>>(
                        seed,
                        max_eval,
                        dev_spec,
                        dev_probs
                    ))
                }
            }
            impl PropagationKernel<$v, $d> for Spectrometer<$fty, $v, UniformDistribution<$fty>, $beam<$fty>, $cmpnd<$v, ArrayFamily<$n>>> {
                const NAME: &'static CStr = cstr!(stringify!([<propagation_test_kernel_ $fname _ $beam:snake _ $cmpnd:snake _ $n>]));

                unsafe fn launch(
                    function: Function,
                    grid: u32,
                    block: u32,
                    shared_memory_size: u32,
                    stream: &Stream,
                    dev_spec: DevicePointer<u8>,
                    _meta: <Self as Pointee>::Metadata,
                    dev_wavelength_cdf: DevicePointer<$fty>,
                    dev_ray_cdf: DevicePointer<Self::Q>,
                    dev_bin_index: DevicePointer<u32>,
                    dev_probability: DevicePointer<$fty>,
                ) -> rustacuda::error::CudaResult<()> {
                    launch!(function<<<grid, block, shared_memory_size, stream>>>(
                        dev_spec,
                        dev_wavelength_cdf,
                        dev_ray_cdf,
                        dev_bin_index,
                        dev_probability
                    ))
                }

            }
        }
    };
    ([$($n:literal),*]) => {
        kernel_impl!(@inner f32; SimdVector<f32, 2> ; f32 GaussianBeam FocusingPlanerCompoundPrism 2);
        $( kernel_impl!(@inner f32; SimdVector<f32, 2> ; f32 GaussianBeam FocusingPlanerCompoundPrism $n 2); )*
        kernel_impl!(@inner f32; SimdVector<f32, 4> ; f32 FiberBeam  CulminatingToricCompoundPrism 3);
        $( kernel_impl!(@inner f32; SimdVector<f32, 4> ; f32 FiberBeam   CulminatingToricCompoundPrism  $n 3); )*
    };
}
kernel_impl!([0, 1, 2, 3, 4, 5, 6]);

fn as_bytes_and_meta<T: ?Sized>(p: &T) -> (&[u8], <T as Pointee>::Metadata) {
    let (raw_ptr, metadata) = (p as *const T).to_raw_parts();
    let size = mem::size_of_val(p);
    // raw_ptr is over-aligned, so it's safe
    (
        unsafe { slice::from_raw_parts(raw_ptr as *const u8, size) },
        metadata,
    )
}

struct CudaFitnessContext {
    ctxt: Context,
    module: Module,
    stream: Stream,
}

impl CudaFitnessContext {
    fn new(ctxt: Context) -> rustacuda::error::CudaResult<Self> {
        ContextStack::push(&ctxt)?;
        let module = Module::load_from_string(PTX)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            ctxt,
            module,
            stream,
        })
    }

    fn launch_p_dets_l_ws<
        F: FloatExt + DeviceCopy,
        V: Vector<D, Scalar = F>,
        S: ?Sized + Kernel<V, D>,
        const D: usize,
    >(
        &mut self,
        seed: f64,
        spec: &S,
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> rustacuda::error::CudaResult<Vec<F>> {
        ContextStack::push(&self.ctxt)?;

        let nbin = spec.detector_bin_count();
        let nprobs = (max_n * nbin) as usize;
        // let mut dev_spec = ManuallyDrop::new(DeviceBox::new(spec)?);
        let (spec_bytes, spec_count) = as_bytes_and_meta(spec);
        let mut dev_spec = ManuallyDrop::new(DeviceBuffer::from_slice(spec_bytes)?);
        let mut dev_probs = ManuallyDrop::new(unsafe { DeviceBuffer::uninitialized(nprobs) }?);
        let function = self.module.get_function(<S as Kernel<V, D>>::NAME)?;
        let dynamic_shared_mem = nbin * nwarp * std::mem::size_of::<Welford<F>>() as u32;
        unsafe {
            <S as Kernel<V, D>>::launch(
                function,
                max_n / nwarp,
                32 * nwarp,
                dynamic_shared_mem,
                &self.stream,
                F::lossy_from(seed),
                max_eval,
                dev_spec.as_device_ptr(),
                spec_count,
                dev_probs.as_device_ptr(),
            )?;
        }
        self.stream.synchronize()?;
        // When the kernel fails to run correctly, the context becomes unusable
        // and the device memory that was allocated, can only be freed by destroying the context.
        // ManuallyDrop is used to prevent rust from trying to deallocate after kernel fail.
        let _dev_spec = ManuallyDrop::into_inner(dev_spec);
        let dev_probs = ManuallyDrop::into_inner(dev_probs);
        let mut p_dets_l_ws = vec![F::zero(); nprobs];
        dev_probs.copy_to(&mut p_dets_l_ws)?;
        Ok(p_dets_l_ws)
    }

    fn launch_propagation_test<
        F: FloatExt + DeviceCopy,
        V: Vector<D, Scalar = F>,
        S: ?Sized + PropagationKernel<V, D>,
        const D: usize,
    >(
        &mut self,
        spec: &S,

        wavelength_cdf: &[F],
        ray_cdf: &[S::Q],

        nwarp: u32,
    ) -> rustacuda::error::CudaResult<(Vec<u32>, Vec<F>)>
    where
        <S as GenericSpectrometer<V, D>>::Q: DeviceCopy,
    {
        assert_eq!(wavelength_cdf.len() % 32, 0);
        assert_eq!(wavelength_cdf.len(), ray_cdf.len());
        ContextStack::push(&self.ctxt)?;

        let total_nwarp = wavelength_cdf.len() as u32 / 32;
        let nblock = total_nwarp / nwarp + total_nwarp % nwarp;

        // let mut dev_spec = ManuallyDrop::new(DeviceBox::new(spec)?);
        let (spec_bytes, spec_count) = as_bytes_and_meta(spec);
        let mut dev_spec = ManuallyDrop::new(DeviceBuffer::from_slice(spec_bytes)?);
        let mut dev_wavelength_cdf =
            ManuallyDrop::new({ DeviceBuffer::from_slice(wavelength_cdf) }?);
        let mut dev_ray_cdf = ManuallyDrop::new({ DeviceBuffer::from_slice(ray_cdf) }?);
        let mut dev_bin_index =
            ManuallyDrop::new(unsafe { DeviceBuffer::uninitialized(wavelength_cdf.len()) }?);
        let mut dev_probability =
            ManuallyDrop::new(unsafe { DeviceBuffer::uninitialized(wavelength_cdf.len()) }?);

        let function = self
            .module
            .get_function(<S as PropagationKernel<V, D>>::NAME)?;
        let dynamic_shared_mem = 0u32;
        let stream = &self.stream;
        unsafe {
            <S as PropagationKernel<V, D>>::launch(
                function,
                nblock,
                32 * nwarp,
                dynamic_shared_mem,
                stream,
                dev_spec.as_device_ptr(),
                spec_count,
                dev_wavelength_cdf.as_device_ptr(),
                dev_ray_cdf.as_device_ptr(),
                dev_bin_index.as_device_ptr(),
                dev_probability.as_device_ptr(),
            )?;
        }
        self.stream.synchronize()?;
        // When the kernel fails to run correctly, the context becomes unusable
        // and the device memory that was allocated, can only be freed by destroying the context.
        // ManuallyDrop is used to prevent rust from trying to deallocate after kernel fail.
        let _drop = ManuallyDrop::into_inner(dev_spec);
        let _drop = ManuallyDrop::into_inner(dev_wavelength_cdf);
        let _drop = ManuallyDrop::into_inner(dev_ray_cdf);
        let dev_bin_index = ManuallyDrop::into_inner(dev_bin_index);
        let dev_probability = ManuallyDrop::into_inner(dev_probability);

        let mut bin_index = vec![0u32; wavelength_cdf.len()];
        let mut probability = vec![F::zero(); wavelength_cdf.len()];
        dev_bin_index.copy_to(&mut bin_index)?;
        dev_probability.copy_to(&mut probability)?;
        Ok((bin_index, probability))
    }
}

unsafe impl Send for CudaFitnessContext {}

static CACHED_CUDA_FITNESS_CONTEXT: Mutex<Option<CudaFitnessContext>> = const_mutex(None);

pub fn set_cached_cuda_context(ctxt: Context) -> rustacuda::error::CudaResult<()> {
    let new_ctxt = CudaFitnessContext::new(ctxt)?;
    let prev = CACHED_CUDA_FITNESS_CONTEXT.lock().replace(new_ctxt);
    drop(prev);
    Ok(())
}

fn get_or_try_insert_with<'o: 't, 't, T, F, E>(
    opt: &'o mut Option<T>,
    with_fn: F,
) -> Result<&'t mut T, E>
where
    F: FnOnce() -> Result<T, E>,
{
    match opt {
        Some(t) => Ok(t),
        o => {
            let t = with_fn()?;
            Ok(o.get_or_insert(t))
        }
    }
}

pub fn cuda_fitness<
    F: FloatExt + DeviceCopy,
    V: Vector<D, Scalar = F>,
    S: ?Sized + Kernel<V, D>,
    const D: usize,
>(
    spectrometer: &S,
    seeds: &[f64],
    max_n: u32,
    nwarp: u32,
    max_eval: u32,
) -> rustacuda::error::CudaResult<Option<DesignFitness<F>>> {
    let max_err = F::lossy_from(5e-3f64);
    let max_err_sq = max_err * max_err;

    let mut lock = CACHED_CUDA_FITNESS_CONTEXT.lock();
    let state = get_or_try_insert_with(&mut *lock, || CudaFitnessContext::new(quick_init()?))?;

    let nbin = spectrometer.detector_bin_count() as usize;
    // p(d=D)
    let mut p_dets = vec![Welford::new(); nbin];
    // -H(d=D|Λ)
    let mut h_det_l_w = Welford::new();

    for &seed in seeds {
        // p(d=D|λ=Λ)
        let p_dets_l_ws = state.launch_p_dets_l_ws(seed, spectrometer, max_n, nwarp, max_eval)?;
        for p_dets_l_w in p_dets_l_ws.chunks_exact(nbin) {
            // -H(D|λ=Λ)
            let mut h_det_l_ws = F::zero();
            for (p_det, p) in p_dets.iter_mut().zip(p_dets_l_w.iter().copied()) {
                p_det.next_sample(p);
                h_det_l_ws += p.plog2p();
            }
            h_det_l_w.next_sample(h_det_l_ws);
        }
        if p_dets.iter().all(|s| s.sem_le_error_threshold(max_err_sq)) {
            // -H(D)
            let h_det = p_dets
                .iter()
                .map(|s| s.mean.plog2p())
                .fold(F::zero(), core::ops::Add::add);
            // -H(D|Λ)
            let h_det_l_w = h_det_l_w.mean;
            // H(D) - H(D|Λ)
            let info = h_det_l_w - h_det;
            let (size, deviation) = spectrometer.size_and_deviation();

            return Ok(Some(DesignFitness {
                size,
                info,
                deviation,
            }));
        }
    }

    // -H(D)
    let h_det = p_dets
        .iter()
        .map(|s| s.mean.plog2p())
        .fold(F::zero(), core::ops::Add::add);
    // -H(D|Λ)
    let h_det_l_w = h_det_l_w.mean;
    // H(D) - H(D|Λ)
    let info = h_det_l_w - h_det;

    let errors: Vec<_> = p_dets
        .iter()
        .map(|s| s.sem())
        .filter(|e| !e.is_finite() || e >= &max_err)
        .collect();
    eprintln!(
        "cuda_fitness error values too large (>={}), consider raising max wavelength count: {:.3?}, {}",
        max_err, errors, info,
    );
    Ok(None)
}

#[cfg(test)]
mod tests {
    use std::ptr::NonNull;

    use rand::distributions::{Distribution, Standard};
    use rand::prelude::*;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256StarStar;

    use super::*;

    #[test]
    fn test_propagation_with_known_prism() {
        let glasses = [
            // N-PK52A
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
            // N-SF57
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
            // N-FK58
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
        let prism = CompoundPrism::<f32, Plane<_, 2>, _, CurvedPlane<_, 2>, _>::new(
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
            detector_array_positioning(&prism, pmt_length, det_angle, wavelengths, &beam, 1.0)
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

        let nwarp = 2;
        let nblock = 16;
        let nthread = nblock * nwarp * 32;

        let mut rng = Xoshiro256StarStar::seed_from_u64(0);
        let wavelength_cdf: Vec<f32> = Standard.sample_iter(&mut rng).take(nthread).collect();
        let ray_cdf: Vec<[f32; 1]> = Standard.sample_iter(&mut rng).take(nthread).collect();

        let mut lock = CACHED_CUDA_FITNESS_CONTEXT.lock();
        let state =
            get_or_try_insert_with(&mut *lock, || CudaFitnessContext::new(quick_init()?)).unwrap();

        let (gpu_bin_index, gpu_probability) = state
            .launch_propagation_test(&spec, &wavelength_cdf, &ray_cdf, nwarp as u32)
            .unwrap();

        struct MockGpu;

        impl crate::spectrometer::kernel::GPU for MockGpu {
            fn warp_size_log2() -> u32 {
                0
            }

            fn global_warp_id() -> u32 {
                0
            }

            fn lane_id() -> u32 {
                0
            }

            fn thread_id() -> u32 {
                todo!()
            }

            fn block_dim() -> u32 {
                todo!()
            }

            fn block_id() -> u32 {
                todo!()
            }

            fn grid_dim() -> u32 {
                todo!()
            }

            fn sync_warp() {
                todo!()
            }

            fn warp_any(pred: bool) -> bool {
                todo!()
            }

            fn warp_ballot(pred: bool) -> u32 {
                todo!()
            }
        }

        for i in 0..nthread {
            let mut cpu_bin_index = 0;
            let mut cpu_probability = 0.0;
            unsafe {
                crate::spectrometer::kernel::propagation_test_kernel::<MockGpu, _, _, _, 2>(
                    &spec,
                    NonNull::from(&wavelength_cdf[i]),
                    NonNull::from(&ray_cdf[i]),
                    NonNull::new(&mut cpu_bin_index).unwrap(),
                    NonNull::new(&mut cpu_probability).unwrap(),
                );
            }
            assert_eq!(cpu_bin_index, gpu_bin_index[i], "index {}", i);
            // println!("cpu {} ; gpu {}", cpu_probability, gpu_probability[i]);
            float_eq::assert_float_eq!(cpu_probability, gpu_probability[i], rmax <= 5e-3);
        }
    }
}
