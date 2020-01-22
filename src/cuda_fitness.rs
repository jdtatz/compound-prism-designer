use crate::fitness::DesignFitness;
use crate::ray::Spectrometer;
use crate::utils::{Float, Welford};
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use rustacuda::context::ContextStack;
use rustacuda::memory::{DeviceBox, DeviceBuffer, DeviceCopy};
use rustacuda::prelude::*;
use rustacuda::{launch, quick_init};
use std::ffi::CStr;

unsafe impl<F: Float + DeviceCopy> DeviceCopy for Spectrometer<F> {}

const PTX: &[u8] = concat!(
    include_str!("../target/nvptx64-nvidia-cuda/release/compound_prism_designer.ptx"),
    "\0"
)
.as_bytes();
const MAX_N: usize = 256;
const MAX_M: usize = 16_384;
const NWARP: u32 = 2;
#[allow(clippy::unreadable_literal)]
const SEEDS: &[f64] = &[
    0.4993455843,
    0.8171187913,
    0.8234955201,
    0.3911129692,
    0.2887278677,
    0.5004948416,
    0.6997901652,
    0.1544760953,
    0.8681075152,
    0.3130246465,
    0.0472957030,
    0.5421335682,
    0.5607025594,
    0.5911739281,
    0.7870769069,
];

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
        let module = Module::load_from_string(unsafe { CStr::from_bytes_with_nul_unchecked(PTX) })?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        Ok(Self {
            ctxt,
            module,
            stream,
        })
    }

    fn launch_p_dets_l_ws<F: KernelFloat>(
        &mut self,
        seed: f64,
        spec: &Spectrometer<F>,
    ) -> rustacuda::error::CudaResult<Vec<F>> {
        ContextStack::push(&self.ctxt)?;

        let nbin = spec.detector_array.bin_count as usize;
        let mut dev_spec = DeviceBox::new(spec)?;
        let mut dev_probs = unsafe { DeviceBuffer::zeroed(MAX_N * nbin) }?;
        let function = self
            .module
            .get_function(unsafe { CStr::from_bytes_with_nul_unchecked(F::FNAME) })?;
        let dynamic_shared_mem = std::mem::size_of::<Spectrometer<F>>() as u32
            + nbin as u32 * NWARP * std::mem::size_of::<[F; 2]>() as u32;
        let stream = &self.stream;
        unsafe {
            launch!(function<<<(MAX_N as u32) / NWARP, 32 * NWARP, dynamic_shared_mem, stream>>>(
                F::from_f64(seed),
                MAX_M as u32,
                dev_spec.as_device_ptr(),
                dev_probs.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;
        let mut p_dets_l_ws = vec![F::zero(); MAX_N * nbin];
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
        std::mem::replace(&mut *guard, new_state);
    } else {
        CACHED_CUDA_FITNESS_CONTEXT
            .get_or_try_init(|| CudaFitnessContext::new(ctxt).map(Mutex::new))?;
    }
    Ok(())
}

fn plog2p<F: Float>(p: F) -> F {
    if p == F::zero() {
        F::zero()
    } else {
        p * p.log2()
    }
}

#[derive(Debug, From)]
pub enum CudaFitnessError {
    IntegrationAccuracyIssue,
    Cuda(rustacuda::error::CudaError),
}

impl<F: KernelFloat> Spectrometer<F> {
    pub fn cuda_fitness(&self) -> Result<DesignFitness<F>, CudaFitnessError> {
        const MAX_ERR: f64 = 5e-3;
        const MAX_ERR_SQR: f64 = MAX_ERR * MAX_ERR;

        let mutex = CACHED_CUDA_FITNESS_CONTEXT
            .get_or_try_init(|| CudaFitnessContext::new(quick_init()?).map(Mutex::new))?;
        let mut state = mutex.lock();

        let nbin = self.detector_array.bin_count as usize;
        let mut p_dets = vec![Welford::new(); nbin];
        let mut plog2p_p_det_l_w = vec![Welford::new(); nbin];

        for &seed in SEEDS {
            let p_dets_l_ws = state.launch_p_dets_l_ws(seed, self)?;
            for p_dets_l_w in p_dets_l_ws.chunks_exact(nbin) {
                for ((p_det, stat), p) in p_dets
                    .iter_mut()
                    .zip(plog2p_p_det_l_w.iter_mut())
                    .zip(p_dets_l_w.iter().copied())
                {
                    p_det.next_sample(p);
                    stat.next_sample(plog2p(p));
                }
            }
            if p_dets
                .iter()
                .all(|s| s.sem_le_error_threshold(F::from_f64(MAX_ERR_SQR)))
            {
                let info = plog2p_p_det_l_w
                    .iter()
                    .map(|s| s.mean)
                    .fold(F::zero(), core::ops::Add::add)
                    - p_dets
                        .iter()
                        .map(|s| plog2p(s.mean))
                        .fold(F::zero(), core::ops::Add::add);
                let (size, deviation) = self.size_and_deviation();

                return Ok(DesignFitness {
                    size,
                    info,
                    deviation,
                });
            }
        }

        let errors: Vec<_> = p_dets
            .iter()
            .map(|s| s.sem().to_f64())
            .filter(|e| e >= &MAX_ERR)
            .collect();
        eprintln!("cuda_fitness error values too large (>={}), consider raising max wavelength count: {:.3?}", MAX_ERR, errors);
        Err(CudaFitnessError::IntegrationAccuracyIssue)
    }
}
