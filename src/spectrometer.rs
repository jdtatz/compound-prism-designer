use crate::erf::norminv;
use crate::geom::{Mat2, Pair, Surface, Vector};
use crate::utils::{Float, LossyInto};
use crate::{Beam, CompoundPrism, DetectorArray, Ray, RayTraceError};

/// Collimated Polychromatic Gaussian Beam
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Float")]
pub struct GaussianBeam<F: Float> {
    /// 1/e^2 beam width
    pub width: F,
    /// Mean y coordinate
    pub y_mean: F,
    /// Range of wavelengths
    pub w_range: (F, F),
}

impl<F: Float> GaussianBeam<F> {
    pub fn inverse_cdf_wavelength(&self, p: F) -> F {
        self.w_range.0 + (self.w_range.1 - self.w_range.0) * p
    }

    pub fn inverse_cdf_initial_y(&self, p: F) -> F {
        self.y_mean - self.width * norminv(p)
    }

    pub fn inverse_cdf_ray<V: Vector<Scalar = F>>(&self, p: F) -> Ray<V> {
        Ray::new_from_start(self.inverse_cdf_initial_y(p))
    }
}

impl<F: Float> Beam for GaussianBeam<F> {
    type Vector = Pair<F>;
    // type Vector = impl Vector<Scalar=F>;
    type Quasi = F;

    fn y_mean(&self) -> <Self::Vector as Vector>::Scalar {
        self.y_mean
    }

    fn wavelength_range(
        &self,
    ) -> (
        <Self::Vector as Vector>::Scalar,
        <Self::Vector as Vector>::Scalar,
    ) {
        self.w_range
    }

    fn inverse_cdf_wavelength(
        &self,
        p: <Self::Vector as Vector>::Scalar,
    ) -> <Self::Vector as Vector>::Scalar {
        self.w_range.0 + (self.w_range.1 - self.w_range.0) * p
    }

    fn inverse_cdf_ray(&self, q: Self::Quasi) -> Ray<Self::Vector> {
        Ray::new_from_start(self.y_mean - self.width * norminv(q))
    }
}

/// Polychromatic Uniform Circular Multi-Mode Fiber Beam
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "F: Float")]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "V: Vector")]
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
#[derive(Debug, PartialEq, Clone, Copy, Serialize, Deserialize)]
#[serde(bound = "V: Vector")]
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
pub(crate) fn detector_array_positioning<V: Vector, B: Beam<Vector = V>>(
    cmpnd: &CompoundPrism<V>,
    detarr: &LinearDetectorArray<V>,
    beam: &B,
) -> Result<DetectorArrayPositioning<V>, RayTraceError> {
    let ray = Ray::new_from_start(beam.y_mean());
    let (wmin, wmax) = beam.wavelength_range();
    let lower_ray = ray.propagate_internal(cmpnd, wmin)?;
    let upper_ray = ray.propagate_internal(cmpnd, wmax)?;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "V: Vector, B: serde::Serialize + serde::de::DeserializeOwned")]
pub struct Spectrometer<V: Vector, B: Beam<Vector = V>> {
    pub beam: B,
    pub compound_prism: CompoundPrism<V>,
    pub detector: (LinearDetectorArray<V>, DetectorArrayPositioning<V>),
}

impl<V: Vector, B: Beam<Vector = V>> Spectrometer<V, B> {
    pub fn new(
        beam: B,
        compound_prism: CompoundPrism<V>,
        detector_array: LinearDetectorArray<V>,
    ) -> Result<Self, RayTraceError> {
        let detector_array_position =
            detector_array_positioning(&compound_prism, &detector_array, &beam)?;
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
        Ray::new_from_start(initial_y).trace(wavelength, &self.compound_prism, &self.detector)
    }

    pub(crate) fn size_and_deviation(&self) -> (V::Scalar, V::Scalar) {
        let deviation_vector = self.detector.1.position
            + self.detector.1.direction * self.detector.length() * V::Scalar::from_u32_ratio(1, 2)
            - V::from_xy(V::Scalar::zero(), self.beam.y_mean());
        let size = deviation_vector.norm();
        let deviation = deviation_vector.sin_xy(size).abs();
        (size, deviation)
    }
}

// cpu => 1.434454269122527

// 5.6923 ms & approx.ftz => 1.4349899
// 6.5894 ms & full.ftz => 1.4349899
// 34.771 ms & div.f32 => 1.4349926

// 180.14 ms & div.f64 => 1.4339791804027775

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<GaussianBeam<F2>> for GaussianBeam<F1> {
    fn lossy_into(self) -> GaussianBeam<F2> {
        GaussianBeam {
            width: self.width.lossy_into(),
            y_mean: self.y_mean.lossy_into(),
            w_range: self.w_range.lossy_into(),
        }
    }
}

impl<V1: Vector + LossyInto<V2>, V2: Vector> LossyInto<LinearDetectorArray<V2>>
    for LinearDetectorArray<V1>
where
    V1::Scalar: LossyInto<V2::Scalar>,
{
    fn lossy_into(self) -> LinearDetectorArray<V2> {
        LinearDetectorArray {
            bin_count: self.bin_count,
            bin_size: self.bin_size.lossy_into(),
            linear_slope: self.linear_slope.lossy_into(),
            linear_intercept: self.linear_intercept.lossy_into(),
            min_ci: self.min_ci.lossy_into(),
            angle: self.angle.lossy_into(),
            normal: self.normal.lossy_into(),
            length: self.length.lossy_into(),
        }
    }
}

impl<V1: Vector + LossyInto<V2>, V2: Vector> LossyInto<DetectorArrayPositioning<V2>>
    for DetectorArrayPositioning<V1>
where
    V1::Scalar: LossyInto<V2::Scalar>,
{
    fn lossy_into(self) -> DetectorArrayPositioning<V2> {
        DetectorArrayPositioning {
            position: self.position.lossy_into(),
            direction: self.direction.lossy_into(),
        }
    }
}

impl<
        V1: Vector + LossyInto<V2>,
        V2: Vector,
        B1: Beam<Vector = V1> + LossyInto<B2>,
        B2: Beam<Vector = V2>,
    > LossyInto<Spectrometer<V2, B2>> for Spectrometer<V1, B1>
where
    V1::Scalar: LossyInto<V2::Scalar>,
{
    fn lossy_into(self) -> Spectrometer<V2, B2> {
        Spectrometer {
            beam: self.beam.lossy_into(),
            compound_prism: self.compound_prism.lossy_into(),
            detector: self.detector.lossy_into(),
        }
    }
}
