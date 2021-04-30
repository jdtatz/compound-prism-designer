use crate::erf::norminv;
use crate::{distribution::Distribution, UnitVector, Vector};
// use crate::geom::{Mat2, Pair, Surface, Vector};
use crate::geometry::*;
use crate::qrng::QuasiRandom;
use crate::utils::*;
use crate::{CompoundPrism, DetectorArray, Ray, RayTraceError};

/// Collimated Polychromatic Gaussian Beam
#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct GaussianBeam<F, const D: usize> {
    /// 1/e^2 beam width
    pub width: F,
    /// Mean y coordinate
    pub y_mean: F,
    pub marker: core::marker::PhantomData<Vector<F, D>>,
}

impl<T: FloatExt, const D: usize> Distribution<T> for GaussianBeam<T, D> {
    type Output = Ray<T, D>;

    fn inverse_cdf(&self, p: T) -> Ray<T, D> {
        Ray::new_from_start(self.y_mean - self.width * norminv(p))
    }
}

/// Polychromatic Uniform Circular Multi-Mode Fiber Beam
#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct FiberBeam<F: FloatExt> {
    /// Radius of fiber core
    pub radius: F,
    /// Numerical apeature
    pub na: F,
    /// Mean y coordinate
    pub y_mean: F,
}

/// Linear Array of detectors
/// where the bins are defined by
/// for i in 0..bin_count
/// lower_bound = linear_slope * i + linear_intercept
/// upper_bound = linear_slope * i + linear_intercept + bin_size
#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct LinearDetectorArray<T, const D: usize> {
    /// The number of bins in the array
    #[wrapped_from(skip)]
    pub(crate) bin_count: u32,
    /// The size / length of the bins
    bin_size: T,
    /// The slope used in the linear equation find the bin bounds
    linear_slope: T,
    /// The intercept used in the linear equation find the bin bounds
    linear_intercept: T,
    /// Minimum cosine of incident angle == cosine of maximum allowed incident angle
    min_ci: T,
    /// CCW angle of the array from normal = Rot(θ) @ (0, 1)
    angle: T,
    /// The normal of the array's surface, normal = Rot(θ) @ (-1, 0)
    normal: UnitVector<T, D>,
    /// Length of the array
    pub(crate) length: T,
}

impl<T: FloatExt, const D: usize> LinearDetectorArray<T, D> {
    pub fn new(
        bin_count: u32,
        bin_size: T,
        linear_slope: T,
        linear_intercept: T,
        min_ci: T,
        angle: T,
        length: T,
    ) -> Self {
        debug_assert!(bin_count > 0);
        debug_assert!(bin_size > T::zero());
        debug_assert!(linear_slope > T::zero());
        debug_assert!(linear_intercept >= T::zero());
        Self {
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
            min_ci,
            angle,
            normal: UnitVector::new(Vector::angled_xy(angle).rot_180_xy()),
            length,
        }
    }

    pub fn bin_index(&self, pos: T) -> Option<u32> {
        let (bin, bin_pos) = (pos - self.linear_intercept).euclid_div_rem(self.linear_slope);
        let bin = bin.lossy_into();
        if bin < self.bin_count && bin_pos < self.bin_size {
            Some(bin)
        } else {
            None
        }
    }

    pub fn bounds(&self) -> impl ExactSizeIterator<Item = [T; 2]> + '_ {
        (0..self.bin_count).map(move |i| {
            let i = T::lossy_from(i);
            let lb = self.linear_intercept + self.linear_slope * i;
            let ub = lb + self.bin_size;
            [lb, ub]
        })
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub fn end_points(&self, pos: &DetectorArrayPositioning<T, D>) -> (Vector<T, D>, Vector<T, D>) {
        (
            pos.position,
            pos.direction.mul_add(self.length, pos.position),
        )
    }
}

/// Positioning of detector array
#[derive(Debug, PartialEq, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct DetectorArrayPositioning<T, const D: usize> {
    /// Position vector of array
    pub position: Vector<T, D>,
    /// Unit direction vector of array
    pub direction: UnitVector<T, D>,
}

impl<T: FloatExt, const D: usize> Surface<T, D>
    for (LinearDetectorArray<T, D>, DetectorArrayPositioning<T, D>)
{
    fn intersection(self, ray: GeometricRay<T, D>) -> Option<GeometricRayIntersection<T, D>> {
        let (detarr, detpos) = self;
        let ci = -ray.direction.dot(*detarr.normal);
        if ci <= detarr.min_ci {
            // RayTraceError::SpectrometerAngularResponseTooWeak
            return None;
        }
        let d = (ray.origin - detpos.position).dot(*detarr.normal) / ci;
        debug_assert!(d > T::zero());
        Some(GeometricRayIntersection {
            distance: d,
            normal: detarr.normal,
        })
    }
}

impl<T: FloatExt, const D: usize> DetectorArray<T, D>
    for (LinearDetectorArray<T, D>, DetectorArrayPositioning<T, D>)
{
    fn bin_count(&self) -> u32 {
        self.0.bin_count
    }

    fn length(&self) -> T {
        self.0.length
    }

    fn bin_index(&self, intersection: Vector<T, D>) -> Option<u32> {
        let (detarr, detpos) = self;
        let pos = (intersection - detpos.position).dot(*detpos.direction);
        if pos < T::zero() || detarr.length < pos {
            return None;
        }
        detarr.bin_index(pos)
    }
}

// TODO needs updating to new Vector impl
/// Find the position and orientation of the detector array,
/// parameterized by the minimum and maximum wavelengths of the input beam,
/// and its angle from the normal.
///
/// # Arguments
///  * `cmpnd` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
#[cfg(not(target_arch = "nvptx64"))]
pub(crate) fn detector_array_positioning<
    T: FloatExt,
    W: Distribution<T, Output = T>,
    Q: QuasiRandom<Scalar = T>,
    B: Distribution<Q, Output = Ray<T, D>>,
    S0: Copy + Surface<T, D>,
    SI: Copy + Surface<T, D>,
    SN: Copy + Surface<T, D>,
    const N: usize,
    const D: usize,
>(
    cmpnd: CompoundPrism<T, S0, SI, SN, N, D>,
    detarr: LinearDetectorArray<T, D>,
    wavelengths: W,
    beam: B,
) -> Result<DetectorArrayPositioning<T, D>, RayTraceError> {
    let ray = beam.inverse_cdf(Q::from_scalar(T::lossy_from(0.5f64)));
    // let wmin = wavelengths.inverse_cdf(T::lossy_from(0.01_f64));
    // let wmax = wavelengths.inverse_cdf(T::lossy_from(0.99_f64));
    let wmin = wavelengths.inverse_cdf(T::zero());
    let wmax = wavelengths.inverse_cdf(T::one());
    debug_assert!(wmin.is_finite());
    debug_assert!(wmax.is_finite());
    debug_assert!(wmin > T::zero());
    debug_assert!(wmax > wmin);
    let lower_ray = ray.propagate_internal(&cmpnd, wmin)?;
    let upper_ray = ray.propagate_internal(&cmpnd, wmax)?;
    if lower_ray.average_transmittance() <= T::lossy_from(1e-3f64)
        || upper_ray.average_transmittance() <= T::lossy_from(1e-3f64)
    {
        dbg!(ray, lower_ray, upper_ray);
        return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
    }
    debug_assert!(lower_ray.direction.is_unit());
    debug_assert!(upper_ray.direction.is_unit());
    let spec_dir = Vector::angled_xy(detarr.angle).rot_90_xy();
    let spec = spec_dir * detarr.length;

    /// Matrix inverse if it exists
    fn mat_inverse<T: FloatExt>(mat: [[T; 2]; 2]) -> Option<[[T; 2]; 2]> {
        let [[a, b], [c, d]] = mat;
        let det = a * d - b * c;
        if det == T::zero() {
            None
        } else {
            Some([[d / det, -b / det], [-c / det, a / det]])
        }
    }

    fn mat_mul<T: FloatExt>(mat: [[T; 2]; 2], vec: [T; 2]) -> [T; 2] {
        #![allow(clippy::many_single_char_names)]

        let [[a, b], [c, d]] = mat;
        let [x, y] = vec;
        [a * x + b * y, c * x + d * y]
    }

    let mat = [
        [upper_ray.direction.x(), -lower_ray.direction.x()],
        [upper_ray.direction.y(), -lower_ray.direction.y()],
    ];
    let imat = mat_inverse(mat).ok_or(RayTraceError::NoSurfaceIntersection)?;
    let temp = spec - upper_ray.origin + lower_ray.origin;
    let [_d1, d2] = mat_mul(imat, [temp.x(), temp.y()]);
    let l_vertex = lower_ray.direction.mul_add(d2, lower_ray.origin);
    let (pos, dir) = if d2 > T::zero() {
        (l_vertex, spec_dir)
    } else {
        let temp = -spec - upper_ray.origin + lower_ray.origin;
        let [_d1, d2] = mat_mul(imat, [temp.x(), temp.y()]);
        if d2 < T::zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let u_vertex = lower_ray.direction.mul_add(d2, lower_ray.origin);
        (u_vertex, -spec_dir)
    };
    Ok(DetectorArrayPositioning {
        position: pos,
        direction: UnitVector::new(dir),
    })
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct Spectrometer<
    T,
    W,
    B,
    S0: Surface<T, D>,
    SI: Surface<T, D>,
    SN: Surface<T, D>,
    const N: usize,
    const D: usize,
> {
    pub wavelengths: W,
    pub beam: B,
    pub compound_prism: CompoundPrism<T, S0, SI, SN, N, D>,
    pub detector: (LinearDetectorArray<T, D>, DetectorArrayPositioning<T, D>),
}

impl<
        T: FloatExt,
        W: Copy + Distribution<T, Output = T>,
        B: Copy,
        S0: Copy + Surface<T, D>,
        SI: Copy + Surface<T, D>,
        SN: Copy + Surface<T, D>,
        const N: usize,
        const D: usize,
    > Spectrometer<T, W, B, S0, SI, SN, N, D>
{
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn new<Q: QuasiRandom<Scalar = T>>(
        wavelengths: W,
        beam: B,
        compound_prism: CompoundPrism<T, S0, SI, SN, N, D>,
        detector_array: LinearDetectorArray<T, D>,
    ) -> Result<Self, RayTraceError>
    where
        B: Distribution<Q, Output = Ray<T, D>>,
    {
        let detector_array_position =
            detector_array_positioning(compound_prism, detector_array, wavelengths, beam)?;
        Ok(Self {
            wavelengths,
            beam,
            compound_prism,
            detector: (detector_array, detector_array_position),
        })
    }

    /// Propagate a ray of `wavelength` start `initial_y` through the spectrometer.
    /// Returning the intersection position on the detector array
    /// and the transmission probability.
    ///
    /// # Arguments
    ///  * `self` - spectrometer specification
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `initial_y` - the initial y value of the ray
    pub fn propagate(&self, wavelength: T, initial_y: T) -> Result<(u32, T), RayTraceError> {
        Ray::new_from_start(initial_y)
            .propagate(wavelength, &self.compound_prism, &self.detector)
            .map(|(idx, t)| (idx, t))
    }

    /// Trace the propagation of a ray of `wavelength` through the spectrometer.
    /// Returning an iterator of the ray's origin position and
    /// all of the intersection positions.
    ///
    /// # Arguments
    ///  * `self` - spectrometer specification
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `initial_y` - the initial y value of the ray
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn trace_ray_path(
        &self,
        wavelength: T,
        initial_y: T,
    ) -> impl Iterator<Item = Result<Vector<T, D>, RayTraceError>> {
        Ray::new_from_start(initial_y).trace(wavelength, self.compound_prism, self.detector)
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub fn size_and_deviation<Q: QuasiRandom<Scalar = T>>(&self) -> (T, T)
    where
        B: Distribution<Q, Output = Ray<T, D>>,
    {
        let deviation_vector = self.detector.1.direction.mul_add(
            self.detector.length() * T::lossy_from(0.5f64),
            self.detector.1.position,
        ) - self
            .beam
            .inverse_cdf(Q::from_scalar(T::lossy_from(0.5f64)))
            .origin;
        let size = deviation_vector.norm();
        let deviation = deviation_vector.sin_xy(size).abs();
        (size, deviation)
    }
}

unsafe impl<
        T: rustacuda_core::DeviceCopy,
        W,
        B,
        S0: Surface<T, D>,
        SI: Surface<T, D>,
        SN: Surface<T, D>,
        const N: usize,
        const D: usize,
    > rustacuda_core::DeviceCopy for Spectrometer<T, W, B, S0, SI, SN, N, D>
{
}
