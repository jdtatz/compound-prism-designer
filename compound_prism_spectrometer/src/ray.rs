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

impl From<RayTraceError> for &'static str {
    fn from(err: RayTraceError) -> &'static str {
        match err {
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

    fn inverse_cdf_wavelength(
        &self,
        p: <Self::Vector as Vector>::Scalar,
    ) -> <Self::Vector as Vector>::Scalar;
    fn inverse_cdf_ray(&self, q: Self::Quasi) -> Ray<Self::Vector>;
}

pub trait DetectorArray<Point: Vector, UnitVector: Vector = Point>:
    Surface<Point, UnitVector>
{
    fn bin_count(&self) -> u32;
    fn length(&self) -> <Point as Vector>::Scalar;
    fn bin_index(&self, intersection: Point) -> Option<u32>;
}

/// Compound Prism Specification
#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct CompoundPrism<V: Vector, S0: Surface<V>, SI: Surface<V>, SN: Surface<V>, const N: usize>
{
    /// First glass
    glass0: Glass<V::Scalar, 6>,
    /// First boundary surface
    surface0: S0,
    /// List of glasses the compound prism is composed of, in order
    glasses: [Glass<V::Scalar, 6>; N],
    /// Inter-media boundary surfaces
    isurfaces: [SI; N],
    /// Final boundary surface
    surfaceN: SN,
    /// Height of compound prism
    pub(crate) height: V::Scalar,
    /// Width of compound prism
    pub(crate) width: V::Scalar,
    /// Are the inter-media surfaces coated(anti-reflective)?
    #[wrapped_from(skip)]
    ar_coated: bool,
}

pub type PlanerCompoundPrism<V, const N: usize> = CompoundPrism<V, Plane<V>, Plane<V>, Plane<V>, N>;
pub type CompoundPrismLensLast<V, const N: usize> =
    CompoundPrism<V, Plane<V>, Plane<V>, CurvedPlane<V>, N>;
pub type CompoundPrismBothLens<V, const N: usize> =
    CompoundPrism<V, CurvedPlane<V>, Plane<V>, CurvedPlane<V>, N>;

impl<V: Vector, const N: usize> CompoundPrism<V, Plane<V>, Plane<V>, CurvedPlane<V>, N> {
    /// Create a new Compound Prism Specification
    ///
    /// # Arguments
    ///  * `glasses` - List of glasses the compound prism is composed of, in order
    ///  * `angles` - Angles that parameterize the shape of the compound prism
    ///  * `lengths` - Lengths that parameterize the trapezoidal shape of the compound prism
    ///  * `curvature` - Lens Curvature of last surface of compound prism
    ///  * `height` - Height of compound prism
    ///  * `width` - Width of compound prism
    ///  * `coat` - Coat the compound prism surfaces with anti-reflective coating
    pub fn new(
        glass0: Glass<V::Scalar, 6>,
        glasses: [Glass<V::Scalar, 6>; N],
        first_angle: V::Scalar,
        angles: [V::Scalar; N],
        last_angle: V::Scalar,
        first_length: V::Scalar,
        lengths: [V::Scalar; N],
        curvature: V::Scalar,
        height: V::Scalar,
        width: V::Scalar,
        coat: bool,
    ) -> Self {
        let (surface0, isurfaces, last_surface) = create_joined_trapezoids(
            height,
            first_angle,
            angles,
            last_angle,
            first_length,
            lengths,
        );
        let surfaceN = CurvedPlane::new(curvature, height, last_surface);
        Self {
            glass0,
            glasses,
            surface0,
            isurfaces,
            surfaceN,
            height,
            width,
            ar_coated: coat,
        }
    }

    #[cfg(feature = "std")]
    pub fn polygons(&self) -> (Vec<[V; 4]>, [V; 4], V, V::Scalar) {
        let mut poly = Vec::with_capacity(N);
        let (mut u0, mut l0) = self.surface0.end_points(self.height);
        for s in self.isurfaces.iter() {
            let (u1, l1) = s.end_points(self.height);
            poly.push([l0, u0, u1, l1]);
            u0 = u1;
            l0 = l1;
        }
        let (u1, l1) = self.surfaceN.end_points(self.height);
        (
            poly,
            [l0, u0, u1, l1],
            self.surfaceN.center,
            self.surfaceN.radius,
        )
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
        (self.s_transmittance + self.p_transmittance) * V::Scalar::lossy_from(0.5f64)
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
        let (s_transmittance, p_transmittance) = if ar_coated && ci > V::Scalar::lossy_from(0.5f64)
        {
            (
                self.s_transmittance * V::Scalar::lossy_from(0.99f64),
                self.p_transmittance * V::Scalar::lossy_from(0.99f64),
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
    ///  * `detector` - detector array specification
    fn intersect_detector_array(
        self,
        detector: &impl DetectorArray<V>,
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
    pub fn propagate_internal<S0: Surface<V>, SI: Surface<V>, SN: Surface<V>, const N: usize>(
        self,
        cmpnd: &CompoundPrism<V, S0, SI, SN, N>,
        wavelength: V::Scalar,
    ) -> Result<Self, RayTraceError> {
        let mut ray = self;
        let mut n1 = V::Scalar::one();

        let glass = cmpnd.glass0;
        let surf = cmpnd.surface0;
        let n2 = glass.calc_n(wavelength);
        debug_assert!(n2 >= V::Scalar::one());
        let (p, normal) = surf
            .intersection((ray.origin, ray.direction))
            .ok_or(RayTraceError::NoSurfaceIntersection)?;
        ray = ray.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
        n1 = n2;

        // for (glass, plane) in core::array::IntoIter::new(cmpnd.glasses.zip(cmpnd.isurfaces)) {
        for i in 0..N {
            let glass = cmpnd.glasses[i];
            let plane = cmpnd.isurfaces[i];
            let n2 = glass.calc_n(wavelength);
            debug_assert!(n2 >= V::Scalar::one());
            let (p, normal) = plane
                .intersection((ray.origin, ray.direction))
                .ok_or(RayTraceError::NoSurfaceIntersection)?;
            ray = ray.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
            n1 = n2;
        }
        let n2 = V::Scalar::one();
        let (p, normal) = cmpnd
            .surfaceN
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
    ///  * `detector` - detector array specification
    pub fn propagate<S0: Surface<V>, SI: Surface<V>, SN: Surface<V>, const N: usize>(
        self,
        wavelength: V::Scalar,
        cmpnd: &CompoundPrism<V, S0, SI, SN, N>,
        detector: &impl DetectorArray<V>,
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
    ///  * `detector` - detector array specification
    #[cfg(not(target_arch = "nvptx64"))]
    pub fn trace<S0: Surface<V>, SI: Surface<V>, SN: Surface<V>, const N: usize>(
        self,
        wavelength: V::Scalar,
        cmpnd: CompoundPrism<V, S0, SI, SN, N>,
        detector: impl DetectorArray<V> + Copy,
    ) -> impl Iterator<Item = Result<V, RayTraceError>> {
        let mut ray = self;
        let mut n1 = V::Scalar::one();
        let mut prism0 = Some((cmpnd.glass0, cmpnd.surface0));
        let mut prisms = core::array::IntoIter::new(cmpnd.glasses.zip(cmpnd.isurfaces));
        let mut internal = true;
        let mut done = false;
        let mut propagation_fn = move || -> Result<Option<V>, RayTraceError> {
            if let Some((glass, plane)) = prism0.take() {
                let n2 = glass.calc_n(wavelength);
                let (p, normal) = plane
                    .intersection((ray.origin, ray.direction))
                    .ok_or(RayTraceError::NoSurfaceIntersection)?;
                ray = ray.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
                n1 = n2;
                return Ok(Some(ray.origin));
            }
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
                        .surfaceN
                        .intersection((ray.origin, ray.direction))
                        .ok_or(RayTraceError::NoSurfaceIntersection)?;
                    ray = ray.refract(p, normal, n1, n2, cmpnd.ar_coated)?;
                    Ok(Some(ray.origin))
                }
                None if !done && !internal => {
                    done = true;
                    let (pos, _, _) = ray.intersect_detector_array(&detector)?;
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
