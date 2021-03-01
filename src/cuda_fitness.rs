use crate::fitness::DesignFitness;
use compound_prism_spectrometer::*;
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::Mutex;
use rustacuda::context::ContextStack;
use rustacuda::memory::{DeviceBox, DeviceBuffer, DeviceCopy};
use rustacuda::prelude::*;
use rustacuda::{launch, quick_init};
use std::ffi::{CStr, CString};
use std::mem::ManuallyDrop;

const PTX_STR: &str = include_str!("kernel.ptx");

// post-processed generated ptx
static PTX: Lazy<CString> = Lazy::new(|| {
    // switch from slow division into fast but still very accurate division
    // and flushes subnormal values to 0
    let s = PTX_STR.replace("div.rn.f32", "div.approx.ftz.f32");
    CString::new(s).unwrap()
});

// const MAX_N: usize = 256;
// const MAX_M: usize = 16_384;
// const NWARP: u32 = 2;

pub trait KernelFloat: Float + DeviceCopy {
    const FNAME: &'static [u8];
}

impl KernelFloat for f32 {
    const FNAME: &'static [u8] = b"prob_dets_given_wavelengths\0";
}

impl KernelFloat for f64 {
    const FNAME: &'static [u8] = b"prob_dets_given_wavelengths_f64\0";
}

struct CudaFitnessContext {
    ctxt: Context,
    module: Module,
    stream: Stream,
}

impl CudaFitnessContext {
    fn new(ctxt: Context) -> rustacuda::error::CudaResult<Self> {
        ContextStack::push(&ctxt)?;
        let module = Module::load_from_string(PTX.as_c_str())?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            ctxt,
            module,
            stream,
        })
    }

    fn launch_p_dets_l_ws<F: KernelFloat, V: Vector<Scalar = F>, B: Beam<Vector = V>>(
        &mut self,
        seed: f64,
        spec: &Spectrometer<V, B>,
        max_n: u32,
        nwarp: u32,
        max_eval: u32,
    ) -> rustacuda::error::CudaResult<Vec<F>> {
        ContextStack::push(&self.ctxt)?;

        let nbin = spec.detector.bin_count();
        let nprobs = (max_n * nbin) as usize;
        let mut dev_spec = ManuallyDrop::new(DeviceBox::new(spec)?);
        let mut dev_probs = ManuallyDrop::new(unsafe { DeviceBuffer::uninitialized(nprobs) }?);
        let function = self
            .module
            .get_function(unsafe { CStr::from_bytes_with_nul_unchecked(F::FNAME) })?;
        let dynamic_shared_mem = nbin * nwarp * std::mem::size_of::<Welford<F>>() as u32;
        let stream = &self.stream;
        unsafe {
            launch!(function<<<max_n / nwarp, 32 * nwarp, dynamic_shared_mem, stream>>>(
                F::from_f64(seed),
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

static CACHED_CUDA_FITNESS_CONTEXT: OnceCell<Mutex<CudaFitnessContext>> = OnceCell::new();

pub fn set_cached_cuda_context(ctxt: Context) -> rustacuda::error::CudaResult<()> {
    if let Some(mutex) = CACHED_CUDA_FITNESS_CONTEXT.get() {
        let mut guard = mutex.lock();
        let new_state = CudaFitnessContext::new(ctxt)?;
        *guard = new_state;
    } else {
        CACHED_CUDA_FITNESS_CONTEXT
            .get_or_try_init(|| CudaFitnessContext::new(ctxt).map(Mutex::new))?;
    }
    Ok(())
}

pub fn cuda_fitness<F: KernelFloat, V: Vector<Scalar = F>, B: Beam<Vector = V>>(
    spectrometer: &Spectrometer<V, B>,
    seeds: &[f64],
    max_n: u32,
    nwarp: u32,
    max_eval: u32,
) -> Option<DesignFitness<F>> {
    let max_err = F::from_u32_ratio(5, 1000);
    let max_err_sq = max_err * max_err;

    let mutex = CACHED_CUDA_FITNESS_CONTEXT
        .get_or_try_init(|| CudaFitnessContext::new(quick_init()?).map(Mutex::new))
        .expect("Failed to initialize Cuda Fitness Context");
    let mut state = mutex.lock();

    let nbin = spectrometer.detector.bin_count() as usize;
    // p(d=D)
    let mut p_dets = vec![Welford::new(); nbin];
    // -H(d=D|Λ)
    let mut h_det_l_w = Welford::new();

    for &seed in seeds {
        // p(d=D|λ=Λ)
        let p_dets_l_ws = state
            .launch_p_dets_l_ws(seed, spectrometer, max_n, nwarp, max_eval)
            .expect("Failed to launch Cuda fitness kernel");
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

            return Some(DesignFitness {
                size,
                info,
                deviation,
            });
        }
    }

    let errors: Vec<_> = p_dets
        .iter()
        .map(|s| s.sem())
        .filter(|e| !e.is_finite() || e >= &max_err)
        .collect();
    eprintln!(
        "cuda_fitness error values too large (>={}), consider raising max wavelength count: {:.3?}",
        max_err, errors
    );
    None
}
