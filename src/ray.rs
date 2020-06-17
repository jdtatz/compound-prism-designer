use crate::erf::norminv;
use crate::geom::*;
use crate::glasscat::Glass;
use crate::qrng::QuasiRandom;
use crate::utils::*;

#[derive(Debug, Display, Clone, Copy)]
pub enum RayTraceError {
    NoSurfaceIntersection,
    OutOfBounds,
    TotalInternalReflection,
    SpectrometerAngularResponseTooWeak,
}

impl Into<&'static str> for RayTraceError {
    fn into(self) -> &'static str {
        match self {
            RayTraceError::NoSurfaceIntersection => "NoSurfaceIntersection",
            RayTraceError::OutOfBounds => "OutOfBounds",
            RayTraceError::TotalInternalReflection => "TotalInternalReflection",
            RayTraceError::SpectrometerAngularResponseTooWeak => {
                "SpectrometerAngularResponseTooWeak"
            }
        }
    }
}

pub trait Beam {
    type Scalar: Float;
    type Vector: Vector<Scalar = Self::Scalar>;
    type Quasi: QuasiRandom;

    fn y_mean(&self) -> Self::Scalar;
    fn wavelength_range(&self) -> (Self::Scalar, Self::Scalar);
    fn inverse_cdf_wavelength(&self, p: Self::Scalar) -> Self::Scalar;
    fn inverse_cdf_ray(&self, q: Self::Quasi) -> Ray<Self::Vector>;
}

pub trait DetectorArray: Surface {
    fn bin_count(&self) -> u32;
    fn length(&self) -> <<Self as Surface>::Point as Vector>::Scalar;
    fn bin_index(&self, intersection: Self::Point) -> Option<u32>;
}

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
    type Scalar = F;
    type Vector = Pair<F>;
    // type Vector = impl Vector<Scalar=F>;
    type Quasi = F;

    fn y_mean(&self) -> Self::Scalar {
        self.y_mean
    }

    fn wavelength_range(&self) -> (Self::Scalar, Self::Scalar) {
        self.w_range
    }

    fn inverse_cdf_wavelength(&self, p: Self::Scalar) -> Self::Scalar {
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

/// Compound Prism Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "V: Vector")]
pub struct CompoundPrism<V: Vector> {
    /// List of glasses the compound prism is composed of, in order.
    /// With their inter-media boundary surfaces
    prisms: arrayvec::ArrayVec<[(Glass<V::Scalar>, Plane<V>); 6]>,
    /// The curved lens-like last inter-media boundary surface of the compound prism
    lens: CurvedPlane<V>,
    /// Height of compound prism
    pub(crate) height: V::Scalar,
    /// Width of compound prism
    pub(crate) width: V::Scalar,
    /// Are the inter-media surfaces coated(anti-reflective)?
    ar_coated: bool,
}

impl<V: Vector> CompoundPrism<V> {
    /// Create a new Compound Prism Specification
    ///
    /// # Arguments
    ///  * `glasses` - List of glasses the compound prism is composed of, in order
    ///  * `angles` - Angles that parameterize the shape of the compound prism
    ///  * `lengths` - Lengths that parameterize the trapezoidal shape of the compound prism
    ///  * `curvature` - Lens Curvature of last surface of compound prism
    ///  * `height` - Height of compound prism
    ///  * `width` - Width of compound prism
    ///  * `coat` - Coat the outer compound prism surfaces with anti-reflective coating
    pub fn new<I: IntoIterator<Item = Glass<V::Scalar>>>(
        glasses: I,
        angles: &[V::Scalar],
        lengths: &[V::Scalar],
        curvature: V::Scalar,
        height: V::Scalar,
        width: V::Scalar,
        coat: bool,
    ) -> Self
    where
        I::IntoIter: ExactSizeIterator,
    {
        let glasses = glasses.into_iter();
        debug_assert!(glasses.len() > 0);
        debug_assert!(angles.len() - 1 == glasses.len());
        debug_assert!(lengths.len() == glasses.len());
        let mut prisms = arrayvec::ArrayVec::new();
        let (mut last_surface, rest) = create_joined_trapezoids(height, angles, lengths);
        for (g, next_surface) in glasses.zip(rest) {
            prisms.push((g, last_surface));
            last_surface = next_surface;
        }
        let lens = CurvedPlane::new(curvature, height, last_surface);
        Self {
            prisms,
            lens,
            height,
            width,
            ar_coated: coat,
        }
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub fn polygons(&self) -> (Vec<[V; 4]>, [V; 4], V, V::Scalar) {
        let mut poly = Vec::with_capacity(self.prisms.len());
        let (mut u0, mut l0) = self.prisms[0].1.end_points(self.height);
        for (_, s) in self.prisms[1..].iter() {
            let (u1, l1) = s.end_points(self.height);
            poly.push([l0, u0, u1, l1]);
            u0 = u1;
            l0 = l1;
        }
        let (u1, l1) = self.lens.end_points(self.height);
        (poly, [l0, u0, u1, l1], self.lens.center, self.lens.radius)
    }
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

impl<V: Vector> Surface for (&LinearDetectorArray<V>, &DetectorArrayPositioning<V>) {
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

impl<V: Vector> DetectorArray for (&LinearDetectorArray<V>, &DetectorArrayPositioning<V>) {
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

/// Light Ray
#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
pub struct Ray<V: Vector> {
    /// Origin position vector
    origin: V,
    /// Unit normal direction vector
    direction: V,
    /// S-Polarization Transmittance probability
    s_transmittance: V::Scalar,
    /// P-Polarization Transmittance probability
    p_transmittance: V::Scalar,
}

impl<V: Vector> Ray<V> {
    /// Create a new unpolarized ray with full transmittance with a origin at (0, `y`) and a
    /// direction of (1, 0)
    ///
    /// # Arguments
    ///  * `y` - the initial y value of the ray's position
    pub fn new_from_start(y: V::Scalar) -> Self {
        Ray {
            origin: V::from_xy(V::Scalar::zero(), y),
            direction: V::from_xy(V::Scalar::one(), V::Scalar::zero()),
            s_transmittance: V::Scalar::one(),
            p_transmittance: V::Scalar::one(),
        }
    }

    /// Create a new unpolarized ray with full transmittance with a origin at (0, `y`) and a
    /// direction of (angle_cosine, angle_sine)
    ///
    /// # Arguments
    ///  * `y` - the initial y value of the ray's position
    ///  * `angle_sine` - the sine of the angle that the ray is rotated from the axis
    pub fn new_from_start_at_angle(y: V::Scalar, angle_sine: V::Scalar) -> Self {
        Ray {
            origin: V::from_xy(V::Scalar::zero(), y),
            direction: V::from_xy((V::Scalar::one() - angle_sine.sqr()).sqrt(), angle_sine),
            s_transmittance: V::Scalar::one(),
            p_transmittance: V::Scalar::one(),
        }
    }

    /// The average of the S & P Polarizations transmittance's
    fn average_transmittance(self) -> V::Scalar {
        (self.s_transmittance + self.p_transmittance) * V::Scalar::from_u32_ratio(1, 2)
    }

    /// Refract ray through interface of two different media
    /// using vector form of snell's law
    ///
    /// # Arguments
    ///  * `intersection` - point of intersection between the media
    ///  * `normal` - the unit normal vector of the interface
    ///  * `ci` - cosine of incident angle
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    fn refract(
        self,
        intersection: V,
        normal: V,
        n1: V::Scalar,
        n2: V::Scalar,
        ar_coated: bool,
    ) -> Result<Self, RayTraceError> {
        debug_assert!(n1 >= V::Scalar::one());
        debug_assert!(n2 >= V::Scalar::one());
        debug_assert!(normal.check_unit());
        let ci = -self.direction.dot(normal);
        let r = n1 / n2;
        let cr_sq = V::Scalar::one() - r.sqr() * (V::Scalar::one() - ci.sqr());
        if cr_sq < V::Scalar::zero() {
            return Err(RayTraceError::TotalInternalReflection);
        }
        let cr = cr_sq.sqrt();
        let v = self.direction * r + normal * (r * ci - cr);
        let (s_transmittance, p_transmittance) =
            if ar_coated && ci > V::Scalar::from_u32_ratio(1, 2) {
                (
                    self.s_transmittance * V::Scalar::from_u32_ratio(99, 100),
                    self.p_transmittance * V::Scalar::from_u32_ratio(99, 100),
                )
            } else {
                let fresnel_rs = (n1 * ci - n2 * cr) / (n1 * ci + n2 * cr);
                let fresnel_rp = (n1 * cr - n2 * ci) / (n1 * cr + n2 * ci);
                (
                    self.s_transmittance * (V::Scalar::one() - fresnel_rs.sqr()),
                    self.p_transmittance * (V::Scalar::one() - fresnel_rp.sqr()),
                )
            };
        Ok(Self {
            origin: intersection,
            direction: v,
            s_transmittance,
            p_transmittance,
        })
    }

    /// Find the intersection position of the ray with the detector array
    /// and the ray's transmission probability. The intersection position is a
    /// scalar on the line defined by the detector array.
    ///
    /// # Arguments
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    fn intersect_detector_array(
        self,
        detarr: &LinearDetectorArray<V>,
        detpos: &DetectorArrayPositioning<V>,
    ) -> Result<(V, u32, V::Scalar), RayTraceError> {
        let det = (detarr, detpos);
        let (p, _n) = det
            .intersection((self.origin, self.direction))
            .ok_or(RayTraceError::SpectrometerAngularResponseTooWeak)?;
        let bin_idx = det
            .bin_index(p)
            .ok_or(RayTraceError::NoSurfaceIntersection)?;
        Ok((p, bin_idx, self.average_transmittance()))
    }

    /// Propagate a ray of `wavelength` through the compound prism
    ///
    /// # Arguments
    ///  * `cmpnd` - the compound prism specification
    ///  * `wavelength` - the wavelength of the light ray
    fn propagate_internal(
        self,
        cmpnd: &CompoundPrism<V>,
        wavelength: V::Scalar,
    ) -> Result<Self, RayTraceError> {
        let (ray, n1) = cmpnd.prisms.iter().try_fold(
            (self, V::Scalar::one()),
            |(ray, n1), (glass, plane)| {
                let n2 = glass.calc_n(wavelength);
                debug_assert!(n2 >= V::Scalar::one());
                let (p, normal) = plane
                    .intersection((ray.origin, ray.direction))
                    .ok_or(RayTraceError::NoSurfaceIntersection)?;
                let ray = ray.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
                Ok((ray, n2))
            },
        )?;
        let n2 = V::Scalar::one();
        let (p, normal) = cmpnd
            .lens
            .intersection((ray.origin, ray.direction))
            .ok_or(RayTraceError::NoSurfaceIntersection)?;
        ray.refract(p, normal, n1, n2, cmpnd.ar_coated)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning the intersection scalar
    /// and the transmission probability.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `cmpnd` - the compound prism specification
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    pub fn propagate(
        self,
        wavelength: V::Scalar,
        cmpnd: &CompoundPrism<V>,
        detarr: &LinearDetectorArray<V>,
        detpos: &DetectorArrayPositioning<V>,
    ) -> Result<(V, u32, V::Scalar), RayTraceError> {
        self.propagate_internal(cmpnd, wavelength)?
            .intersect_detector_array(detarr, detpos)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning an iterator of the ray's origin position and
    /// all of the intersection positions.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `cmpnd` - the compound prism specification
    ///  * `detarr` - detector array specification
    ///  * `detpos` - the position and orientation of the detector array
    #[cfg(not(target_arch = "nvptx64"))]
    fn trace<'s>(
        self,
        wavelength: V::Scalar,
        cmpnd: &'s CompoundPrism<V>,
        detarr: &'s LinearDetectorArray<V>,
        detpos: &'s DetectorArrayPositioning<V>,
    ) -> impl Iterator<Item = Result<V, RayTraceError>> + 's {
        let mut ray = self;
        let mut n1 = V::Scalar::one();
        let mut prisms = cmpnd.prisms.iter().fuse();
        let mut internal = true;
        let mut done = false;
        let mut propagation_fn = move || -> Result<Option<V>, RayTraceError> {
            match prisms.next() {
                Some((glass, plane)) => {
                    let n2 = glass.calc_n(wavelength);
                    let (p, normal) = plane
                        .intersection((ray.origin, ray.direction))
                        .ok_or(RayTraceError::NoSurfaceIntersection)?;
                    ray = ray.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
                    n1 = n2;
                    Ok(Some(ray.origin))
                }
                None if !done && internal => {
                    internal = false;
                    let n2 = V::Scalar::one();
                    let (p, normal) = cmpnd
                        .lens
                        .intersection((ray.origin, ray.direction))
                        .ok_or(RayTraceError::NoSurfaceIntersection)?;
                    ray = ray.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
                    Ok(Some(ray.origin))
                }
                None if !done && !internal => {
                    done = true;
                    let (pos, _, _) = ray.intersect_detector_array(detarr, detpos)?;
                    Ok(Some(pos))
                }
                _ if done => Ok(None),
                _ => unreachable!(),
            }
        };
        core::iter::once(Ok(self.origin))
            .chain(core::iter::from_fn(move || propagation_fn().transpose()).fuse())
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
pub(crate) fn detector_array_positioning<V: Vector>(
    cmpnd: &CompoundPrism<V>,
    detarr: &LinearDetectorArray<V>,
    beam: &GaussianBeam<V::Scalar>,
) -> Result<DetectorArrayPositioning<V>, RayTraceError> {
    let ray = Ray::new_from_start(beam.y_mean);
    let (wmin, wmax) = beam.w_range;
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
#[serde(bound = "V: Vector")]
pub struct Spectrometer<V: Vector> {
    pub gaussian_beam: GaussianBeam<V::Scalar>,
    pub compound_prism: CompoundPrism<V>,
    pub detector_array: LinearDetectorArray<V>,
    pub detector_array_position: DetectorArrayPositioning<V>,
}

impl<V: Vector> Spectrometer<V> {
    pub fn new(
        gaussian_beam: GaussianBeam<V::Scalar>,
        compound_prism: CompoundPrism<V>,
        detector_array: LinearDetectorArray<V>,
    ) -> Result<Self, RayTraceError> {
        let detector_array_position =
            detector_array_positioning(&compound_prism, &detector_array, &gaussian_beam)?;
        Ok(Self {
            gaussian_beam,
            compound_prism,
            detector_array,
            detector_array_position,
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
            .propagate(
                wavelength,
                &self.compound_prism,
                &self.detector_array,
                &self.detector_array_position,
            )
            .map(|(_p, idx, t)| (idx, t))
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
        Ray::new_from_start(initial_y).trace(
            wavelength,
            &self.compound_prism,
            &self.detector_array,
            &self.detector_array_position,
        )
    }

    pub(crate) fn size_and_deviation(&self) -> (V::Scalar, V::Scalar) {
        let deviation_vector = self.detector_array_position.position
            + self.detector_array_position.direction
                * self.detector_array.length
                * V::Scalar::from_u32_ratio(1, 2)
            - V::from_xy(V::Scalar::zero(), self.gaussian_beam.y_mean);
        let size = deviation_vector.norm();
        let deviation = deviation_vector.sin_xy(size).abs();
        (size, deviation)
    }

    pub(crate) fn probability_z_in_bounds(&self) -> V::Scalar {
        let p_z = (self.compound_prism.width
            * V::Scalar::from_f64(core::f64::consts::FRAC_1_SQRT_2)
            / self.gaussian_beam.width)
            .erf();
        debug_assert!(V::Scalar::zero() <= p_z && p_z <= V::Scalar::one());
        p_z
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
            w_range: LossyInto::lossy_into(self.w_range),
        }
    }
}

impl<V1: Vector + LossyInto<V2>, V2: Vector> LossyInto<CompoundPrism<V2>> for CompoundPrism<V1>
where
    V1::Scalar: LossyInto<V2::Scalar>,
{
    fn lossy_into(self) -> CompoundPrism<V2> {
        CompoundPrism {
            prisms: self.prisms.lossy_into(),
            lens: self.lens.lossy_into(),
            height: self.height.lossy_into(),
            width: self.width.lossy_into(),
            ar_coated: self.ar_coated,
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
            normal: LossyInto::lossy_into(self.normal),
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
            position: LossyInto::lossy_into(self.position),
            direction: LossyInto::lossy_into(self.direction),
        }
    }
}

impl<V1: Vector + LossyInto<V2>, V2: Vector> LossyInto<Spectrometer<V2>> for Spectrometer<V1>
where
    V1::Scalar: LossyInto<V2::Scalar>,
{
    fn lossy_into(self) -> Spectrometer<V2> {
        Spectrometer {
            gaussian_beam: LossyInto::lossy_into(self.gaussian_beam),
            compound_prism: LossyInto::lossy_into(self.compound_prism),
            detector_array: LossyInto::lossy_into(self.detector_array),
            detector_array_position: LossyInto::lossy_into(self.detector_array_position),
        }
    }
}
