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
    type Vector: Vector;
    type Quasi: QuasiRandom<Scalar = <Self::Vector as Vector>::Scalar>;

    fn y_mean(&self) -> <Self::Vector as Vector>::Scalar;
    fn wavelength_range(
        &self,
    ) -> (
        <Self::Vector as Vector>::Scalar,
        <Self::Vector as Vector>::Scalar,
    );
    fn inverse_cdf_wavelength(
        &self,
        p: <Self::Vector as Vector>::Scalar,
    ) -> <Self::Vector as Vector>::Scalar;
    fn inverse_cdf_ray(&self, q: Self::Quasi) -> Ray<Self::Vector>;
}

pub trait DetectorArray: Surface {
    fn bin_count(&self) -> u32;
    fn length(&self) -> <<Self as Surface>::Point as Vector>::Scalar;
    fn bin_index(&self, intersection: Self::Point) -> Option<u32>;
}

/// Compound Prism Specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "V: Vector")]
pub struct CompoundPrism<V: Vector> {
    /// List of glasses the compound prism is composed of, in order.
    /// With their inter-media boundary surfaces
    prisms: arrayvec::ArrayVec<[(Glass<V::Scalar, 6>, Plane<V>); 6]>,
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
    pub fn new<I: IntoIterator<Item = Glass<V::Scalar, 6>>>(
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

    #[cfg(feature = "std")]
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

/// Light Ray
#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
pub struct Ray<V: Vector> {
    /// Origin position vector
    pub origin: V,
    /// Unit normal direction vector
    pub direction: V,
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
    pub fn average_transmittance(self) -> V::Scalar {
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
    pub fn refract(
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
        detector: &impl DetectorArray<Point = V, UnitVector = V>,
    ) -> Result<(V, u32, V::Scalar), RayTraceError> {
        let (p, _n) = detector
            .intersection((self.origin, self.direction))
            .ok_or(RayTraceError::SpectrometerAngularResponseTooWeak)?;
        let bin_idx = detector
            .bin_index(p)
            .ok_or(RayTraceError::NoSurfaceIntersection)?;
        Ok((p, bin_idx, self.average_transmittance()))
    }

    /// Propagate a ray of `wavelength` through the compound prism
    ///
    /// # Arguments
    ///  * `cmpnd` - the compound prism specification
    ///  * `wavelength` - the wavelength of the light ray
    pub fn propagate_internal(
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
        detector: &impl DetectorArray<Point = V, UnitVector = V>,
    ) -> Result<(u32, V::Scalar), RayTraceError> {
        let (_, idx, t) = self
            .propagate_internal(cmpnd, wavelength)?
            .intersect_detector_array(detector)?;
        Ok((idx, t))
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
    pub fn trace<'s>(
        self,
        wavelength: V::Scalar,
        cmpnd: &'s CompoundPrism<V>,
        detector: &'s (impl DetectorArray<Point = V, UnitVector = V> + 's),
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
                    let (pos, _, _) = ray.intersect_detector_array(detector)?;
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
