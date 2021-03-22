use crate::erf::norminv;
use crate::geom::{Mat2, Pair, Surface, Vector};
use crate::qrng::QuasiRandom;
use crate::utils::Float;
use crate::{distribution::Distribution, utils::LossyFrom};
use crate::{Beam, CompoundPrism, DetectorArray, Ray, RayTraceError};

/// Collimated Polychromatic Gaussian Beam
#[derive(Debug, Clone, Copy)]
pub struct GaussianBeam<F, D> {
    /// 1/e^2 beam width
    pub width: F,
    /// Mean y coordinate
    pub y_mean: F,
    /// Wavelengths distribution
    pub wavelengths: D,
}

impl<F: Float, D> GaussianBeam<F, D> {
    pub fn inverse_cdf_initial_y(&self, p: F) -> F {
        self.y_mean - self.width * norminv(p)
    }

    pub fn inverse_cdf_ray<V: Vector<Scalar = F>>(&self, p: F) -> Ray<V> {
        Ray::new_from_start(self.inverse_cdf_initial_y(p))
    }
}

impl<F: Float, D: Distribution<Item = F>> Beam for GaussianBeam<F, D> {
    type Vector = Pair<F>;
    // type Vector = impl Vector<Scalar=F>;
    type Quasi = F;

    fn inverse_cdf_wavelength(
        &self,
        p: <Self::Vector as Vector>::Scalar,
    ) -> <Self::Vector as Vector>::Scalar {
        self.wavelengths.inverse_cdf(p)
    }

    fn inverse_cdf_ray(&self, q: Self::Quasi) -> Ray<Self::Vector> {
        Ray::new_from_start(self.y_mean - self.width * norminv(q))
    }
}

/// Polychromatic Uniform Circular Multi-Mode Fiber Beam
#[derive(Debug, Clone, Copy)]
pub struct FiberBeam<F: Float> {
    /// Radius of fiber core
    pub radius: F,
    /// Numerical apeature
    pub na: F,
    /// Mean y coordinate
    pub y_mean: F,
    /// Range of wavelengths
    pub w_range: (F, F),
}
/// Linear Array of detectors
/// where the bins are defined by
/// for i in 0..bin_count
/// lower_bound = linear_slope * i + linear_intercept
/// upper_bound = linear_slope * i + linear_intercept + bin_size
#[derive(Debug, Clone, Copy)]
pub struct LinearDetectorArray<V: Vector> {
    /// The number of bins in the array
    pub(crate) bin_count: u32,
    /// The size / length of the bins
    bin_size: V::Scalar,
    /// The slope used in the linear equation find the bin bounds
    linear_slope: V::Scalar,
    /// The intercept used in the linear equation find the bin bounds
    linear_intercept: V::Scalar,
    /// Minimum cosine of incident angle == cosine of maximum allowed incident angle
    min_ci: V::Scalar,
    /// CCW angle of the array from normal = Rot(θ) @ (0, 1)
    angle: V::Scalar,
    /// The normal of the array's surface, normal = Rot(θ) @ (-1, 0)
    normal: V,
    /// Length of the array
    pub(crate) length: V::Scalar,
}

impl<V: Vector> LinearDetectorArray<V> {
    pub fn new(
        bin_count: u32,
        bin_size: V::Scalar,
        linear_slope: V::Scalar,
        linear_intercept: V::Scalar,
        min_ci: V::Scalar,
        angle: V::Scalar,
        length: V::Scalar,
    ) -> Self {
        debug_assert!(bin_count > 0);
        debug_assert!(bin_size > V::Scalar::zero());
        debug_assert!(linear_slope > V::Scalar::zero());
        debug_assert!(linear_intercept >= V::Scalar::zero());
        Self {
            bin_count,
            bin_size,
            linear_slope,
            linear_intercept,
            min_ci,
            angle,
            normal: V::angled_xy(angle).rot_180_xy(),
            length,
        }
    }

    pub fn bin_index(&self, pos: V::Scalar) -> Option<u32> {
        let (bin, bin_pos) = (pos - self.linear_intercept).euclid_div_rem(self.linear_slope);
        let bin = bin.to_u32();
        if bin < self.bin_count && bin_pos < self.bin_size {
            Some(bin)
        } else {
            None
        }
    }

    pub fn bounds<'s>(&'s self) -> impl ExactSizeIterator<Item = [V::Scalar; 2]> + 's {
        (0..self.bin_count).map(move |i| {
            let i = V::Scalar::from_u32(i);
            let lb = self.linear_intercept + self.linear_slope * i;
            let ub = lb + self.bin_size;
            [lb, ub]
        })
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub fn end_points(&self, pos: &DetectorArrayPositioning<V>) -> (V, V) {
        (pos.position, pos.position + pos.direction * self.length)
    }
}

/// Positioning of detector array
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct DetectorArrayPositioning<V: Vector> {
    /// Position vector of array
    pub position: V,
    /// Unit direction vector of array
    pub direction: V,
}

impl<V: Vector> Surface for (LinearDetectorArray<V>, DetectorArrayPositioning<V>) {
    type Point = V;
    type UnitVector = V;

    fn intersection(
        &self,
        ray: (Self::Point, Self::UnitVector),
    ) -> Option<(Self::Point, Self::UnitVector)> {
        let (detarr, detpos) = self;
        let ci = -ray.1.dot(detarr.normal);
        if ci <= detarr.min_ci {
            // RayTraceError::SpectrometerAngularResponseTooWeak
            return None;
        }
        let d = (ray.0 - detpos.position).dot(detarr.normal) / ci;
        debug_assert!(d > V::Scalar::zero());
        let p = ray.1.mul_add(d, ray.0);
        Some((p, detarr.normal))
    }
}

impl<V: Vector> DetectorArray for (LinearDetectorArray<V>, DetectorArrayPositioning<V>) {
    fn bin_count(&self) -> u32 {
        self.0.bin_count
    }

    fn length(&self) -> <Self::Point as Vector>::Scalar {
        self.0.length
    }

    fn bin_index(&self, intersection: Self::Point) -> Option<u32> {
        let (detarr, detpos) = self;
        let pos = (intersection - detpos.position).dot(detpos.direction);
        if pos < V::Scalar::zero() || detarr.length < pos {
            return None;
        }
        detarr.bin_index(pos)
    }
}

/// Find the position and orientation of the detector array,
/// parameterized by the minimum and maximum wavelengths of the input beam,
/// and its angle from the normal.
///
/// # Arguments
///  * `cmpnd` - the compound prism specification
///  * `detarr` - detector array specification
///  * `beam` - input gaussian beam specification
pub(crate) fn detector_array_positioning<V: Vector, B: Beam<Vector = V>, const N: usize>(
    cmpnd: CompoundPrism<V, N>,
    detarr: LinearDetectorArray<V>,
    beam: &B,
) -> Result<DetectorArrayPositioning<V>, RayTraceError> {
    let ray = beam.inverse_cdf_ray(B::Quasi::from_scalar(V::Scalar::from_u32_ratio(1, 2)));
    // let wmin = beam.inverse_cdf_wavelength(V::Scalar::from_u32_ratio(1, 100));
    // let wmax = beam.inverse_cdf_wavelength(V::Scalar::from_u32_ratio(99, 100));
    let wmin = beam.inverse_cdf_wavelength(V::Scalar::zero());
    let wmax = beam.inverse_cdf_wavelength(V::Scalar::one());
    let lower_ray = ray.propagate_internal(&cmpnd, wmin)?;
    let upper_ray = ray.propagate_internal(&cmpnd, wmax)?;
    if lower_ray.average_transmittance() <= V::Scalar::from_u32_ratio(1, 1000)
        || upper_ray.average_transmittance() <= V::Scalar::from_u32_ratio(1, 1000)
    {
        return Err(RayTraceError::SpectrometerAngularResponseTooWeak);
    }
    debug_assert!(lower_ray.direction.check_unit());
    debug_assert!(upper_ray.direction.check_unit());
    let spec_dir = V::angled_xy(detarr.angle).rot_90_xy();
    let spec = spec_dir * detarr.length;
    let mat = Mat2::new_from_cols(
        Pair::from_vector(upper_ray.direction),
        -Pair::from_vector(lower_ray.direction),
    );
    let imat = mat.inverse().ok_or(RayTraceError::NoSurfaceIntersection)?;
    let dists = imat * Pair::from_vector(spec - upper_ray.origin + lower_ray.origin);
    let d2 = dists.y;
    let l_vertex = lower_ray.direction.mul_add(d2, lower_ray.origin);
    let (pos, dir) = if d2 > V::Scalar::zero() {
        (l_vertex, spec_dir)
    } else {
        let dists = imat * Pair::from_vector(-spec - upper_ray.origin + lower_ray.origin);
        let d2 = dists.y;
        if d2 < V::Scalar::zero() {
            return Err(RayTraceError::NoSurfaceIntersection);
        }
        let u_vertex = lower_ray.direction.mul_add(d2, lower_ray.origin);
        (u_vertex, -spec_dir)
    };
    Ok(DetectorArrayPositioning {
        position: pos,
        direction: dir,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct Spectrometer<V: Vector, B: Beam<Vector = V>, const N: usize> {
    pub beam: B,
    pub compound_prism: CompoundPrism<V, N>,
    pub detector: (LinearDetectorArray<V>, DetectorArrayPositioning<V>),
}

impl<V: Vector, B: Beam<Vector = V>, const N: usize> Spectrometer<V, B, N> {
    pub fn new(
        beam: B,
        compound_prism: CompoundPrism<V, N>,
        detector_array: LinearDetectorArray<V>,
    ) -> Result<Self, RayTraceError> {
        let detector_array_position =
            detector_array_positioning(compound_prism, detector_array, &beam)?;
        Ok(Self {
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
    pub fn propagate(
        &self,
        wavelength: V::Scalar,
        initial_y: V::Scalar,
    ) -> Result<(u32, V::Scalar), RayTraceError> {
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
    pub fn trace_ray_path<'s>(
        &'s self,
        wavelength: V::Scalar,
        initial_y: V::Scalar,
    ) -> impl Iterator<Item = Result<V, RayTraceError>> + 's {
        Ray::new_from_start(initial_y).trace(wavelength, self.compound_prism, self.detector)
    }

    pub fn size_and_deviation(&self) -> (V::Scalar, V::Scalar) {
        let deviation_vector = self.detector.1.position
            + self.detector.1.direction * self.detector.length() * V::Scalar::from_u32_ratio(1, 2)
            - self
                .beam
                .inverse_cdf_ray(B::Quasi::from_scalar(V::Scalar::from_u32_ratio(1, 2)))
                .origin;
        let size = deviation_vector.norm();
        let deviation = deviation_vector.sin_xy(size).abs();
        (size, deviation)
    }
}

impl<F1, F2: LossyFrom<F1>, D1, D2: LossyFrom<D1>> LossyFrom<GaussianBeam<F1, D1>>
    for GaussianBeam<F2, D2>
{
    fn lossy_from(v: GaussianBeam<F1, D1>) -> Self {
        Self {
            width: LossyFrom::lossy_from(v.width),
            y_mean: LossyFrom::lossy_from(v.y_mean),
            wavelengths: LossyFrom::lossy_from(v.wavelengths),
        }
    }
}

impl<V1: Vector, V2: Vector + LossyFrom<V1>> LossyFrom<LinearDetectorArray<V1>>
    for LinearDetectorArray<V2>
where
    V2::Scalar: LossyFrom<V1::Scalar>,
{
    fn lossy_from(v: LinearDetectorArray<V1>) -> Self {
        Self {
            bin_count: v.bin_count,
            bin_size: LossyFrom::lossy_from(v.bin_size),
            linear_slope: LossyFrom::lossy_from(v.linear_slope),
            linear_intercept: LossyFrom::lossy_from(v.linear_intercept),
            min_ci: LossyFrom::lossy_from(v.min_ci),
            angle: LossyFrom::lossy_from(v.angle),
            normal: LossyFrom::lossy_from(v.normal),
            length: LossyFrom::lossy_from(v.length),
        }
    }
}

impl<V1: Vector, V2: Vector + LossyFrom<V1>> LossyFrom<DetectorArrayPositioning<V1>>
    for DetectorArrayPositioning<V2>
where
    V2::Scalar: LossyFrom<V1::Scalar>,
{
    fn lossy_from(v: DetectorArrayPositioning<V1>) -> Self {
        Self {
            position: LossyFrom::lossy_from(v.position),
            direction: LossyFrom::lossy_from(v.direction),
        }
    }
}

impl<
        V1: Vector,
        V2: Vector + LossyFrom<V1>,
        B1: Beam<Vector = V1>,
        B2: Beam<Vector = V2> + LossyFrom<B1>,
        const N: usize
    > LossyFrom<Spectrometer<V1, B1, N>> for Spectrometer<V2, B2, N>
where
    V2::Scalar: LossyFrom<V1::Scalar>,
    CompoundPrism<V2, N>: LossyFrom<CompoundPrism<V1, N>>,
{
    fn lossy_from(v: Spectrometer<V1, B1, N>) -> Self {
        Self {
            beam: LossyFrom::lossy_from(v.beam),
            compound_prism: LossyFrom::lossy_from(v.compound_prism),
            detector: LossyFrom::lossy_from(v.detector),
        }
    }
}

unsafe impl<F: Float + rustacuda_core::DeviceCopy, V: Vector<Scalar = F>, B: Beam<Vector = V>, const N: usize>
    rustacuda_core::DeviceCopy for Spectrometer<V, B, N>
{
}
