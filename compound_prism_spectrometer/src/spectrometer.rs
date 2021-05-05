use crate::geometry::*;
use crate::qrng::QuasiRandom;
use crate::utils::*;
use crate::{distribution::Distribution, UnitVector, Vector};
use crate::{distribution::UniformDiscDistribution, erf::norminv};
use crate::{CompoundPrism, DetectorArray, Ray, RayTraceError};

pub trait Beam<T, Q: QuasiRandom<Scalar = T>, const D: usize>:
    Distribution<Q, Output = Ray<T, D>>
{
    fn median_y(&self) -> T;
}

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

impl<T: FloatExt, const D: usize> Beam<T, T, D> for GaussianBeam<T, D> {
    fn median_y(&self) -> T {
        self.y_mean
    }
}

/// Polychromatic Multi-Mode Fiber Beam
/// The exit position distribution is simplified from a 'Top Hat' profile to a Uniform Disc.
/// The exit direction distribution is uniform over the acceptance cone.
#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct FiberBeam<F, const D: usize> {
    /// Radius of fiber core
    pub radius: F,
    /// $ \cos_{min} = \cos(\theta_{max}) = \cos(\arcsin(NA)) = \sqrt{1 - NA^2}$
    pub min_cos: F,
    /// Center y coordinate
    pub center_y: F,
    pub marker: core::marker::PhantomData<Vector<F, D>>,
}

impl<T: FloatExt, const D: usize> FiberBeam<T, D> {
    /// Radius of fiber core
    /// Numerical aperture
    /// Center y coordinate
    pub fn new(radius: T, na: T, center_y: T) -> Self {
        Self {
            radius,
            min_cos: T::sqrt(T::ONE - na.sqr()),
            center_y,
            marker: core::marker::PhantomData,
        }
    }
}

impl<T: FloatExt> Distribution<[T; 4]> for FiberBeam<T, 3> {
    type Output = Ray<T, 3>;

    fn inverse_cdf(&self, p: [T; 4]) -> Ray<T, 3> {
        let [p_rho, p_theta, p_defl, p_rot] = p;
        let [y, z] = UniformDiscDistribution {
            radius: self.radius,
        }
        .inverse_cdf([p_rho, p_theta]);
        let origin = Vector([T::ZERO, y + self.center_y, z]);
        // let ux = UniformDistribution { bounds: (self.min_cos, T::ONE) }.inverse_cdf(p_defl);
        let ux = self.min_cos + p_defl * (T::ONE - self.min_cos);
        let s_ux = (T::ONE - ux.sqr()).sqrt();
        let (s, c) = (p_rot * T::lossy_from(core::f64::consts::TAU)).sin_cos();
        let uy = s_ux * c;
        let uz = s_ux * s;
        let direction = UnitVector::new(Vector([ux, uy, uz]));
        Ray::new_unpolarized(origin, direction)
    }
}

impl<T: FloatExt> Beam<T, [T; 4], 3> for FiberBeam<T, 3> {
    fn median_y(&self) -> T {
        self.center_y
    }
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
    /// Position vector of array
    pub position: Vector<T, D>,
    /// Unit direction vector of array
    pub direction: UnitVector<T, D>,
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
        position: Vector<T, D>,
        flipped: bool,
    ) -> Self {
        debug_assert!(bin_count > 0);
        debug_assert!(bin_size > T::zero());
        debug_assert!(linear_slope > T::zero());
        debug_assert!(linear_intercept >= T::zero());
        let normal = UnitVector::new(Vector::angled_xy(angle).rot_180_xy());
        let direction = if flipped {
            UnitVector::new(normal.rot_90_xy())
        } else {
            UnitVector::new(normal.rot_90_ccw_xy())
        };
        Self {
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
            min_ci,
            angle,
            normal,
            length,
            position,
            direction,
        }
    }

    // pub fn bin_index(&self, pos: T) -> Option<u32> {
    //     let (bin, bin_pos) = (pos - self.linear_intercept).euclid_div_rem(self.linear_slope);
    //     let bin = bin.lossy_into();
    //     if bin < self.bin_count && bin_pos < self.bin_size {
    //         Some(bin)
    //     } else {
    //         None
    //     }
    // }

    pub fn bounds(&self) -> impl ExactSizeIterator<Item = [T; 2]> + '_ {
        (0..self.bin_count).map(move |i| {
            let i = T::lossy_from(i);
            let lb = self.linear_intercept + self.linear_slope * i;
            let ub = lb + self.bin_size;
            [lb, ub]
        })
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub fn end_points(&self) -> (Vector<T, D>, Vector<T, D>) {
        (
            self.position,
            self.direction.mul_add(self.length, self.position),
        )
    }
}

// /// Positioning of detector array
// #[derive(Debug, PartialEq, Clone, Copy, WrappedFrom)]
// #[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
// pub struct DetectorArrayPositioning<T, const D: usize> {
//     /// Position vector of array
//     pub position: Vector<T, D>,
//     /// Unit direction vector of array
//     pub direction: UnitVector<T, D>,
// }

impl<T: FloatExt, const D: usize> Surface<T, D> for LinearDetectorArray<T, D> {
    fn intersection(self, ray: GeometricRay<T, D>) -> Option<GeometricRayIntersection<T, D>> {
        let ci = -ray.direction.dot(*self.normal);
        if ci <= self.min_ci {
            // RayTraceError::SpectrometerAngularResponseTooWeak
            return None;
        }
        let d = (ray.origin - self.position).dot(*self.normal) / ci;
        debug_assert!(d > T::zero());
        Some(GeometricRayIntersection {
            distance: d,
            normal: self.normal,
        })
    }
}

impl<T: FloatExt, const D: usize> DetectorArray<T, D> for LinearDetectorArray<T, D> {
    fn bin_count(&self) -> u32 {
        self.bin_count
    }

    fn length(&self) -> T {
        self.length
    }

    fn bin_index(&self, intersection: Vector<T, D>) -> Option<u32> {
        let pos = (intersection - self.position).dot(*self.direction);
        if pos < T::zero() || self.length < pos {
            return None;
        }
        let (bin, bin_pos) = (pos - self.linear_intercept).euclid_div_rem(self.linear_slope);
        let bin = bin.lossy_into();
        if bin < self.bin_count && bin_pos < self.bin_size {
            Some(bin)
        } else {
            None
        }
    }
}

// TODO needs updating to new Vector impl
/// Find the position and orientation of the detector array,
/// parameterized by the minimum and maximum wavelengths of the input beam,
/// and its angle from the normal.
///
/// # Arguments
///  * `cmpnd` - the compound prism specification
///  * `detector_array_length` - detector array length
///  * `detector_array_normal` - detector array normal unit vector
///  * `wavelengths` - input wavelength distribution
///  * `beam` - input gaussian beam specification
pub fn detector_array_positioning<
    T: FloatExt,
    W: Distribution<T, Output = T>,
    S0: Copy + Surface<T, D>,
    SI: Copy + Surface<T, D>,
    SN: Copy + Surface<T, D>,
    const N: usize,
    const D: usize,
>(
    cmpnd: CompoundPrism<T, S0, SI, SN, N, D>,
    detector_array_length: T,
    detector_array_angle: T,
    wavelengths: W,
    median_y: T,
) -> Result<(Vector<T, D>, bool), RayTraceError> {
    let ray = Ray::new_from_start(median_y);
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
        // dbg!(ray, lower_ray, upper_ray);
        return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
    }
    debug_assert!(lower_ray.direction.is_unit());
    debug_assert!(upper_ray.direction.is_unit());
    crate::geometry::fit_ray_difference_surface(
        lower_ray.into(),
        upper_ray.into(),
        detector_array_length,
        detector_array_angle,
    )
    .ok_or(RayTraceError::NoSurfaceIntersection)
}

#[derive(Constructor, Debug, Clone, Copy, WrappedFrom)]
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
    pub detector: LinearDetectorArray<T, D>,
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
    // #[cfg(not(target_arch = "nvptx64"))]
    // pub fn new<Q: QuasiRandom<Scalar = T>>(
    //     wavelengths: W,
    //     beam: B,
    //     compound_prism: CompoundPrism<T, S0, SI, SN, N, D>,
    //     detector_array: LinearDetectorArray<T, D>,
    // ) -> Result<Self, RayTraceError>
    // where
    //     B: Distribution<Q, Output = Ray<T, D>>,
    // {
    //     let detector_array_position =
    //         detector_array_positioning(compound_prism, detector_array.length, detector_array.normal, wavelengths, beam)?;
    //     Ok(Self {
    //         wavelengths,
    //         beam,
    //         compound_prism,
    //         detector: detector_array,
    //     })
    // }

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
        let deviation_vector = self.detector.direction.mul_add(
            self.detector.length() * T::lossy_from(0.5f64),
            self.detector.position,
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
