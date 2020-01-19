use crate::fitness::DesignFitness;
use crate::ray::{
    CompoundPrism, DetectorArray, DetectorArrayPositioning, GaussianBeam, Spectrometer,
};
use crate::utils::Welford;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;
use rustacuda::context::ContextStack;
use rustacuda::memory::{DeviceBox, DeviceBuffer};
use rustacuda::prelude::*;
use rustacuda::{launch, quick_init};
use std::ffi::CStr;

unsafe impl rustacuda::memory::DeviceCopy for CompoundPrism {}
unsafe impl<'a> rustacuda::memory::DeviceCopy for DetectorArray<'a> {}
unsafe impl rustacuda::memory::DeviceCopy for DetectorArrayPositioning {}
unsafe impl rustacuda::memory::DeviceCopy for GaussianBeam {}
unsafe impl<'a> rustacuda::memory::DeviceCopy for Spectrometer<'a> {}

const PTX: &[u8] = concat!(
    include_str!("../target/nvptx64-nvidia-cuda/release/compound_prism_designer.ptx"),
    "\0"
)
.as_bytes();
// const PTX_CSTR: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(PTX) };
const FNAME: &[u8] = b"prob_dets_given_wavelengths\0";
// const FNAME_CSTR: &CStr = unsafe { CStr::from_bytes_with_nul_unchecked(PTX) };
const MAX_N: usize = 128;
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

struct CudaFitnessContext {
    ctxt: Context,
    module: Module,
    stream: Stream,
    dev_spec: DeviceBox<Spectrometer<'static>>,
    dev_bins: Option<DeviceBuffer<[f64; 2]>>,
    dev_probs: Option<DeviceBuffer<f64>>,
}

impl CudaFitnessContext {
    fn new(ctxt: Context) -> rustacuda::error::CudaResult<Self> {
        ContextStack::push(&ctxt)?;
        let module = Module::load_from_string(unsafe { CStr::from_bytes_with_nul_unchecked(PTX) })?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        let dev_spec = unsafe { DeviceBox::<Spectrometer>::zeroed() }?;
        Ok(Self {
            ctxt,
            module,
            stream,
            dev_spec,
            dev_bins: None,
            dev_probs: None,
        })
    }

    fn launch_p_dets_l_ws<'a>(
        &mut self,
        seed: f64,
        spec: &Spectrometer<'a>,
    ) -> rustacuda::error::CudaResult<Vec<f64>> {
        ContextStack::push(&self.ctxt)?;

        let detarr = &spec.detector_array;
        self.dev_spec.copy_from(unsafe {
            std::mem::transmute::<&Spectrometer<'a>, &Spectrometer<'static>>(spec)
        })?; // FIXME immediately, very unsafe & dangerous
        let dev_bins = match &mut self.dev_bins {
            Some(ref mut dev_bins) if dev_bins.len() == detarr.bins.len() => {
                dev_bins.copy_from(detarr.bins)?;
                dev_bins
            }
            _ => {
                let dev_bins = DeviceBuffer::from_slice(&detarr.bins)?;
                drop(self.dev_bins.take());
                self.dev_bins.get_or_insert(dev_bins)
            }
        };
        let dev_probs = match &mut self.dev_probs {
            Some(ref mut dev_probs) if dev_probs.len() == MAX_N * detarr.bins.len() => dev_probs,
            _ => {
                let dev_probs = unsafe { DeviceBuffer::uninitialized(MAX_N * detarr.bins.len()) }?;
                drop(self.dev_probs.take());
                self.dev_probs.get_or_insert(dev_probs)
            }
        };
        let function = self
            .module
            .get_function(unsafe { CStr::from_bytes_with_nul_unchecked(FNAME) })?;
        let dynamic_shared_mem = std::mem::size_of::<Spectrometer>() as u32
            + std::mem::size_of_val(spec.detector_array.bins) as u32
            + detarr.bins.len() as u32 * NWARP * 2 * std::mem::size_of::<f64>() as u32;
        let stream = &self.stream;
        unsafe {
            launch!(function<<<(MAX_N as u32) / NWARP, 32 * NWARP, dynamic_shared_mem, stream>>>(
                seed,
                MAX_M as u32,
                self.dev_spec.as_device_ptr(),
                detarr.bins.len() as u32,
                dev_bins.as_device_ptr(),
                dev_probs.as_device_ptr()
            ))?;
        }
        self.stream.synchronize()?;
        let mut p_dets_l_ws = vec![0_f64; MAX_N * detarr.bins.len()];
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

fn plog2p(p: f64) -> f64 {
    if p == 0_f64 {
        0_f64
    } else {
        p * p.log2()
    }
}

#[derive(Debug, From)]
pub enum CudaFitnessError {
    IntegrationAccuracyIssue,
    Cuda(rustacuda::error::CudaError),
}

impl<'a> Spectrometer<'a> {
    pub fn cuda_fitness(&self) -> Result<DesignFitness, CudaFitnessError> {
        const MAX_ERR: f64 = 5e-3;
        const MAX_ERR_SQR: f64 = MAX_ERR * MAX_ERR;

        let beam = &self.gaussian_beam;
        let detarr = &self.detector_array;
        let detpos = &self.detector_array_position;
        let deviation_vector =
            detpos.position + detpos.direction * detarr.length * 0.5 - (0., beam.y_mean).into();
        let size = deviation_vector.norm();
        let deviation = deviation_vector.y.abs() / deviation_vector.norm();

        let mutex = CACHED_CUDA_FITNESS_CONTEXT
            .get_or_try_init(|| CudaFitnessContext::new(quick_init()?).map(Mutex::new))?;
        let mut state = mutex.lock();

        let mut p_dets = vec![Welford::new(); detarr.bins.len()];
        let mut plog2p_p_det_l_w = vec![Welford::new(); detarr.bins.len()];

        for &seed in SEEDS {
            let p_dets_l_ws = state.launch_p_dets_l_ws(seed, self)?;
            for p_dets_l_w in p_dets_l_ws.chunks_exact(detarr.bins.len()) {
                for ((p_det, stat), p) in p_dets
                    .iter_mut()
                    .zip(plog2p_p_det_l_w.iter_mut())
                    .zip(p_dets_l_w.iter().copied())
                {
                    p_det.next_sample(p);
                    stat.next_sample(plog2p(p));
                }
            }
            if p_dets.iter().all(|s| s.sem_le_error_threshold(MAX_ERR_SQR)) {
                let info = plog2p_p_det_l_w.iter().map(|s| s.mean).sum::<f64>()
                    - p_dets.iter().map(|s| plog2p(s.mean)).sum::<f64>();

                return Ok(DesignFitness {
                    size,
                    info,
                    deviation,
                });
            }
        }

        let errors: Vec<_> = p_dets
            .iter()
            .map(|s| s.sem())
            .filter(|e| e >= &MAX_ERR)
            .collect();
        eprintln!("cuda_fitness error values too large (>={}), consider raising max wavelength count: {:.3?}", MAX_ERR, errors);
        Err(CudaFitnessError::IntegrationAccuracyIssue)
    }
}
