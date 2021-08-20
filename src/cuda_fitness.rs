use crate::fitness::DesignFitness;
use compound_prism_spectrometer::*;
use parking_lot::{const_mutex, Mutex};
use rustacuda::context::ContextStack;
use rustacuda::memory::{DeviceBox, DeviceBuffer, DeviceCopy};
use rustacuda::prelude::*;
use rustacuda::{launch, quick_init};
use std::ffi::CStr;
use std::mem::ManuallyDrop;

const PTX_BYTES: &[u8] = concat!(include_str!("kernel.ptx"), "\0").as_bytes();
// const PTX: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(PTX_BYTES) };

// const MAX_N: usize = 256;
// const MAX_M: usize = 16_384;
// const NWARP: u32 = 2;

pub trait Kernel: DeviceCopy {
    const FNAME: &'static [u8];
}

macro_rules! kernel_impl {
    (@inner $fty:ty ; $fname:ident $beam:ident $s0:ident $sn:ident $n:literal $d:literal) => {
        paste::paste! {
            impl Kernel for Spectrometer<$fty, UniformDistribution<$fty>, $beam<$fty>, $s0<$fty, $d>, Plane<$fty, $d>, $sn<$fty, $d>, $n, $d> {
                const FNAME: &'static [u8] = concat!(stringify!([<prob_dets_given_wavelengths_ $fname _ $beam:snake _ $s0:snake _ $sn:snake _ $n _ $d d>]), "\0").as_bytes();
            }
        }
    };
    ([$($n:literal),*]) => {
        $( kernel_impl!(@inner f32; f32 GaussianBeam Plane     CurvedPlane $n 2); )*
        $( kernel_impl!(@inner f32; f32 FiberBeam    ToricLens ToricLens   $n 3); )*
    };
}
kernel_impl!([0, 1, 2, 3, 4, 5, 6]);

struct CudaFitnessContext {
    ctxt: Context,
    module: Module,
    stream: Stream,
}

impl CudaFitnessContext {
    fn new(ctxt: Context) -> rustacuda::error::CudaResult<Self> {
        ContextStack::push(&ctxt)?;
        let ptx_cstr = CStr::from_bytes_with_nul(PTX_BYTES)
            .unwrap_or_else(|_| unreachable!("Invalid PTX CStr bytes, this should never happen"));
        let module = Module::load_from_string(ptx_cstr)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            ctxt,
            module,
            stream,
        })
    }

    fn launch_p_dets_l_ws<
        F: FloatExt + DeviceCopy,
        S: GenericSpectrometer<F, D> + Kernel,
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
        let mut dev_spec = ManuallyDrop::new(DeviceBox::new(spec)?);
        let mut dev_probs = ManuallyDrop::new(unsafe { DeviceBuffer::uninitialized(nprobs) }?);
        let function = self
            .module
            .get_function(unsafe { CStr::from_bytes_with_nul_unchecked(<S as Kernel>::FNAME) })?;
        let dynamic_shared_mem = nbin * nwarp * std::mem::size_of::<Welford<F>>() as u32;
        let stream = &self.stream;
        unsafe {
            launch!(function<<<max_n / nwarp, 32 * nwarp, dynamic_shared_mem, stream>>>(
                F::lossy_from(seed),
                max_eval,
                dev_spec.as_device_ptr(),
                dev_probs.as_device_ptr()
            ))?;
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
    S: GenericSpectrometer<F, D> + Kernel,
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
