use crate::glasscat::Glass;
use crate::utils::*;
use crate::vector::{UnitVector, Vector};
use crate::{drawable::Polygon, geometry::*, Drawable};

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

pub trait DetectorArray<T, const D: usize>: Surface<T, D> {
    fn bin_count(&self) -> u32;
    fn length(&self) -> T;
    fn bin_index(&self, intersection: Vector<T, D>) -> Option<u32>;
    fn mid_pt(&self) -> Vector<T, D>;
}

/// Compound Prism Specification
#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct CompoundPrism<T, S0, SI, SN, const N: usize, const D: usize> {
    pub(crate) initial_glass: Glass<T, 6>,
    pub(crate) glasses: [Glass<T, 6>; N],
    /// initial prism surface
    pub(crate) initial_surface: S0,
    /// rest of the prism surfaces
    pub(crate) inter_surfaces: [SI; N],
    /// Final boundary surface
    pub(crate) final_surface: SN,
    /// Height of compound prism
    pub(crate) height: T,
    /// Width of compound prism
    pub(crate) width: T,
    /// Are the inter-media surfaces coated(anti-reflective)?
    #[wrapped_from(skip)]
    pub(crate) ar_coated: bool,
}

// pub type PlanerCompoundPrism<V, const N: usize> = CompoundPrism<V, Plane<V>, Plane<V>, Plane<V>, N>;
// pub type CompoundPrismLensLast<V, const N: usize> =
//     CompoundPrism<V, Plane<V>, Plane<V>, CurvedPlane<V>, N>;
// pub type CompoundPrismBothLens<V, const N: usize> =
//     CompoundPrism<V, CurvedPlane<V>, Plane<V>, CurvedPlane<V>, N>;

impl<T: FloatExt, S0, SN, const N: usize, const D: usize>
    CompoundPrism<T, S0, Plane<T, D>, SN, N, D>
where
    S0: Surface<T, D> + FromParametrizedHyperPlane<T, D>,
    SN: Surface<T, D> + FromParametrizedHyperPlane<T, D>,
{
    /// Create a new Compound Prism Specification
    ///
    /// # Arguments
    ///  * `glasses` - List of glasses the compound prism is composed of, in order
    ///  * `angles` - Angles that parameterize the shape of the compound prism
    ///  * `lengths` - Lengths that parameterize the trapezoidal shape of the compound prism
    ///  * `s0_parametrization` - First Surface Parametrization
    ///  * `sn_parametrization` - Final Surface Parametrization
    ///  * `height` - Height of compound prism
    ///  * `width` - Width of compound prism
    ///  * `coat` - Coat the compound prism surfaces with anti-reflective coating
    pub fn new(
        glass0: Glass<T, 6>,
        glasses: [Glass<T, 6>; N],
        first_angle: T,
        angles: [T; N],
        last_angle: T,
        first_length: T,
        lengths: [T; N],
        s0_parametrization: S0::Parametrization,
        sn_parametrization: SN::Parametrization,
        height: T,
        width: T,
        coat: bool,
    ) -> Self {
        let prism_bounds = PrismBounds {
            height,
            half_width: width * T::lossy_from(0.5f64),
        };
        let (surface0, isurfaces, last_surface) = create_joined_trapezoids(
            height,
            first_angle,
            angles,
            last_angle,
            first_length,
            lengths,
        );
        let surface0 = S0::from_hyperplane(surface0, s0_parametrization);
        let isurfaces = isurfaces.map(|s| Plane::new(s, prism_bounds));
        let final_surface = SN::from_hyperplane(last_surface, sn_parametrization);
        Self {
            final_surface,
            height,
            width,
            ar_coated: coat,
            initial_glass: glass0,
            glasses,
            initial_surface: surface0,
            inter_surfaces: isurfaces,
        }
    }
}

impl<T: FloatExt, S0, SI, SN, const N: usize, const D: usize> CompoundPrism<T, S0, SI, SN, N, D>
where
    S0: Surface<T, D> + Drawable<T>,
    SI: Copy + Surface<T, D> + Drawable<T>,
    SN: Surface<T, D> + Drawable<T>,
{
    pub fn surfaces(
        &self,
    ) -> (
        (crate::Point<T>, crate::Point<T>, Option<T>),
        [(crate::Point<T>, crate::Point<T>, Option<T>); N],
        (crate::Point<T>, crate::Point<T>, Option<T>),
    ) {
        let path2surface = |p: crate::Path<T>| match p {
            crate::Path::Line { a, b } => (a, b, None),
            crate::Path::Arc {
                a,
                b,
                midpt: _,
                radius,
            } => (a, b, Some(radius)),
        };
        let s_0 = path2surface(self.initial_surface.draw());
        let s_i = self.inter_surfaces.map(|s| path2surface(s.draw()));
        let s_n = path2surface(self.final_surface.draw());
        (s_0, s_i, s_n)
    }

    pub fn polygons(&self) -> ([Polygon<T>; N], Polygon<T>) {
        let mut path0 = self.initial_surface.draw();
        let polys = self.inter_surfaces.map(|s| {
            let path1 = s.draw();
            let poly = Polygon([path0.reverse(), path1]);
            path0 = path1;
            poly
        });
        let path1 = self.final_surface.draw();
        let final_poly = Polygon([path0.reverse(), path1]);
        (polys, final_poly)
    }
}

/// Light Ray
#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
pub struct Ray<T, const D: usize> {
    /// Origin position vector
    pub origin: Vector<T, D>,
    /// Unit normal direction vector
    pub direction: UnitVector<T, D>,
    /// S-Polarization Transmittance probability
    s_transmittance: T,
    /// P-Polarization Transmittance probability
    p_transmittance: T,
}

impl<T: FloatExt, const D: usize> Ray<T, D> {
    /// Create a new unpolarized ray with full transmittance
    ///
    /// # Arguments
    ///  * `origin` - the initial y value of the ray's position
    ///  * `direction` - the initial y value of the ray's position
    pub fn new_unpolarized(origin: Vector<T, D>, direction: UnitVector<T, D>) -> Self {
        Ray {
            origin,
            direction,
            s_transmittance: T::one(),
            p_transmittance: T::one(),
        }
    }

    /// Create a new unpolarized ray with full transmittance with a origin at (0, `y`) and a
    /// direction of (1, 0)
    ///
    /// # Arguments
    ///  * `y` - the initial y value of the ray's position
    pub fn new_from_start(y: T) -> Self {
        let mut origin = Vector::ZERO;
        origin[1] = y;
        let mut direction = Vector::ZERO;
        direction[0] = T::one();
        Ray {
            origin,
            direction: UnitVector::new(direction),
            s_transmittance: T::one(),
            p_transmittance: T::one(),
        }
    }

    pub fn translate(self, distance: T) -> Self {
        let Ray {
            origin,
            direction,
            s_transmittance,
            p_transmittance,
        } = self;
        Ray {
            origin: direction.mul_add(distance, origin),
            direction,
            s_transmittance,
            p_transmittance,
        }
    }

    /// The average of the S & P Polarizations transmittance's
    pub fn average_transmittance(self) -> T {
        (self.s_transmittance + self.p_transmittance) * T::lossy_from(0.5f64)
    }

    /// Refract ray through interface of two different media
    /// using vector form of snell's law
    ///
    /// # Arguments
    ///  * `normal` - the unit normal vector of the interface
    ///  * `ci` - cosine of incident angle
    ///  * `n1` - index of refraction of the current media
    ///  * `n2` - index of refraction of the new media
    pub fn refract(
        self,
        normal: UnitVector<T, D>,
        n1: T,
        n2: T,
        ar_coated: bool,
    ) -> Result<Self, RayTraceError> {
        debug_assert!(n1 >= T::one());
        debug_assert!(n2 >= T::one());
        debug_assert!(normal.is_unit());
        debug_assert!(self.direction.is_unit());
        let ci = -self.direction.dot(*normal);
        debug_assert!(
            T::zero() <= ci && ci <= T::one(),
            "-dot({:?}, {:?}) = {} is outside of valid range [0, 1]",
            self.direction,
            normal,
            ci
        );
        let r = n1 / n2;
        let cr_sq = T::one() - r.sqr() * (T::one() - ci.sqr());
        if cr_sq < T::zero() {
            return Err(RayTraceError::TotalInternalReflection);
        }
        let cr = cr_sq.sqrt();
        debug_assert!(T::zero() <= cr && cr <= T::one());
        let v = UnitVector::new((self.direction.0) * r + (normal.0) * (r * ci - cr));
        debug_assert!(v.x() >= T::one());
        let (s_transmittance, p_transmittance) = if ar_coated && ci > T::lossy_from(0.5f64) {
            (
                self.s_transmittance * T::lossy_from(0.99f64),
                self.p_transmittance * T::lossy_from(0.99f64),
            )
        } else {
            let fresnel_rs = ((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)).sqr();
            let fresnel_rp = ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)).sqr();
            debug_assert!(
                T::zero() <= fresnel_rs && fresnel_rs <= T::one(),
                "fresnel_rs = {} is outside of valid range [0, 1]",
                fresnel_rs
            );
            debug_assert!(
                T::zero() <= fresnel_rp && fresnel_rp <= T::one(),
                "fresnel_rp = {} is outside of valid range [0, 1]",
                fresnel_rp
            );
            (
                self.s_transmittance * (T::one() - fresnel_rs),
                self.p_transmittance * (T::one() - fresnel_rp),
            )
        };
        debug_assert!(T::zero() <= s_transmittance && s_transmittance <= T::one());
        debug_assert!(T::zero() <= p_transmittance && p_transmittance <= T::one());
        Ok(Self {
            origin: self.origin,
            direction: v,
            s_transmittance,
            p_transmittance,
        })
    }

    #[inline]
    pub fn surface_propagate<S: Surface<T, D>>(
        self,
        surface: S,
        n1: T,
        n2: T,
        ar_coated: bool,
    ) -> Result<Self, RayTraceError> {
        debug_assert!(n2 >= T::one());
        let GeometricRayIntersection { distance, normal } = surface
            .intersection(self.into())
            .ok_or(RayTraceError::NoSurfaceIntersection)?;
        self.translate(distance).refract(normal, n1, n2, ar_coated)
    }

    // /// Propagate a ray of `wavelength` through the compound prism
    // ///
    // /// # Arguments
    // ///  * `cmpnd` - the compound prism specification
    // ///  * `wavelength` - the wavelength of the light ray
    // pub fn propagate_internal<
    //     S0: Copy + Surface<T, D>,
    //     SI: Copy + Surface<T, D>,
    //     SN: Copy + Surface<T, D>,
    //     const N: usize,
    // >(
    //     self,
    //     cmpnd: &CompoundPrism<T, S0, SI, SN, N, D>,
    //     wavelength: T,
    // ) -> Result<Self, RayTraceError> {
    //     let mut ray = self;
    //     let mut n1 = T::one();

    //     let glass = cmpnd.glass0;
    //     let surface = cmpnd.surface0;
    //     let n2 = glass.calc_n(wavelength);
    //     ray = ray.surface_propagate(surface, n1, n2, cmpnd.ar_coated)?;
    //     n1 = n2;

    //     // for (glass, plane) in core::array::IntoIter::new(cmpnd.glasses.zip(cmpnd.isurfaces)) {
    //     for i in 0..N {
    //         let glass = cmpnd.glasses[i];
    //         let surface = cmpnd.isurfaces[i];
    //         let n2 = glass.calc_n(wavelength);
    //         ray = ray.surface_propagate(surface, n1, n2, cmpnd.ar_coated)?;
    //         n1 = n2;
    //     }
    //     let n2 = T::one();
    //     let surface = cmpnd.surfaceN;
    //     ray.surface_propagate(surface, n1, n2, cmpnd.ar_coated)
    // }

    /// Find the intersection position of the ray with the detector array
    /// and the ray's transmission probability. The intersection position is a
    /// scalar on the line defined by the detector array.
    ///
    /// # Arguments
    ///  * `detector` - detector array specification
    pub fn intersect_detector_array(
        self,
        detector: &(impl Copy + DetectorArray<T, D>),
    ) -> Result<(Vector<T, D>, u32, T), RayTraceError> {
        let GeometricRayIntersection { distance, .. } = detector
            .intersection(GeometricRay {
                origin: self.origin,
                direction: self.direction,
            })
            .ok_or(RayTraceError::SpectrometerAngularResponseTooWeak)?;
        let p = self.direction.mul_add(distance, self.origin);
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
    #[inline(always)]
    pub fn propagate_internal<
        S0: Copy + Surface<T, D>,
        SI: Copy + Surface<T, D>,
        SN: Copy + Surface<T, D>,
        const N: usize,
    >(
        self,
        initial_refractive_index: T,
        refractive_indicies: [T; N],
        initial_surface: S0,
        isurfaces: [SI; N],
        final_surface: SN,
        ar_coated: bool,
    ) -> Result<Self, RayTraceError> {
        let mut ray = self;
        let mut n1 = T::one();

        let n2 = initial_refractive_index;
        debug_assert!(n2 >= T::one());
        let GeometricRayIntersection { distance, normal } = initial_surface
            .intersection(ray.into())
            .ok_or(RayTraceError::NoSurfaceIntersection)?;
        ray = ray.translate(distance).refract(normal, n1, n2, ar_coated)?;
        n1 = n2;

        // for (n2, plane) in core::array::IntoIter::new(refractive_indicies.zip(cmpnd.isurfaces)) {
        for i in 0..N {
            let n2 = refractive_indicies[i];
            debug_assert!(n2 >= T::one());
            let GeometricRayIntersection { distance, normal } = isurfaces[i]
                .intersection(ray.into())
                .ok_or(RayTraceError::NoSurfaceIntersection)?;
            ray = ray.translate(distance).refract(normal, n1, n2, ar_coated)?;
            n1 = n2;
        }
        let n2 = T::one();
        let GeometricRayIntersection { distance, normal } = final_surface
            .intersection(ray.into())
            .ok_or(RayTraceError::NoSurfaceIntersection)?;
        ray.translate(distance).refract(normal, n1, n2, ar_coated)
    }

    /// Propagate a ray of `wavelength` through the compound prism and
    /// intersect the detector array. Returning the intersection scalar
    /// and the transmission probability.
    ///
    /// # Arguments
    ///  * `wavelength` - the wavelength of the light ray
    ///  * `cmpnd` - the compound prism specification
    ///  * `detector` - detector array specification
    #[inline(always)]
    pub fn propagate<
        S0: Copy + Surface<T, D>,
        SI: Copy + Surface<T, D>,
        SN: Copy + Surface<T, D>,
        const N: usize,
    >(
        self,
        initial_refractive_index: T,
        refractive_indicies: [T; N],
        initial_surface: S0,
        isurfaces: [SI; N],
        final_surface: SN,
        ar_coated: bool,
        detector: &(impl Copy + DetectorArray<T, D>),
    ) -> Result<(u32, T), RayTraceError> {
        let (_, idx, t) = self
            .propagate_internal(
                initial_refractive_index,
                refractive_indicies,
                initial_surface,
                isurfaces,
                final_surface,
                ar_coated,
            )?
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
    pub fn trace<
        S0: Copy + Surface<T, D>,
        SI: Copy + Surface<T, D>,
        SN: Copy + Surface<T, D>,
        const N: usize,
    >(
        self,
        wavelength: T,
        cmpnd: CompoundPrism<T, S0, SI, SN, N, D>,
        detector: impl DetectorArray<T, D> + Copy,
    ) -> impl Iterator<Item = GeometricRay<T, D>> {
        let mut ray = self;
        let mut n1 = T::one();
        let mut prism0 = Some((cmpnd.initial_glass, cmpnd.initial_surface));
        let mut prisms = core::array::IntoIter::new(cmpnd.glasses.zip(cmpnd.inter_surfaces));
        let mut internal = true;
        let mut done = false;
        let mut propagation_fn = move || -> Result<Option<_>, RayTraceError> {
            if let Some((glass, surf)) = prism0.take() {
                let n2 = glass.calc_n(wavelength);
                let GeometricRayIntersection { distance, normal } =
                    surf.intersection(ray.into())
                        .ok_or(RayTraceError::NoSurfaceIntersection)?;
                ray = ray
                    .translate(distance)
                    .refract(normal, n1, n2, cmpnd.ar_coated)?;
                n1 = n2;
                return Ok(Some(GeometricRay::from(ray)));
            }
            match prisms.next() {
                Some((glass, surf)) => {
                    let n2 = glass.calc_n(wavelength);
                    let GeometricRayIntersection { distance, normal } = surf
                        .intersection(ray.into())
                        .ok_or(RayTraceError::NoSurfaceIntersection)?;
                    ray = ray
                        .translate(distance)
                        .refract(normal, n1, n2, cmpnd.ar_coated)?;
                    n1 = n2;
                    Ok(Some(GeometricRay::from(ray)))
                }
                None if !done && internal => {
                    internal = false;
                    let n2 = T::one();
                    let GeometricRayIntersection { distance, normal } = cmpnd
                        .final_surface
                        .intersection(ray.into())
                        .ok_or(RayTraceError::NoSurfaceIntersection)?;
                    ray = ray
                        .translate(distance)
                        .refract(normal, n1, n2, cmpnd.ar_coated)?;
                    Ok(Some(GeometricRay::from(ray)))
                }
                None if !done && !internal => {
                    done = true;
                    let (pos, _, _) = ray.intersect_detector_array(&detector)?;
                    Ok(Some(GeometricRay {
                        origin: pos,
                        direction: ray.direction,
                    }))
                }
                _ if done => Ok(None),
                _ => unreachable!(),
            }
        };
        core::iter::once(GeometricRay::from(self))
            .chain(core::iter::from_fn(move || propagation_fn().ok().flatten()).fuse())
    }
}

impl<T, const D: usize> From<Ray<T, D>> for GeometricRay<T, D> {
    fn from(ray: Ray<T, D>) -> Self {
        let Ray {
            origin, direction, ..
        } = ray;
        Self { origin, direction }
    }
}

impl<
        T: FloatExt,
        S0: Copy + Surface<T, D>,
        SI: Copy + Surface<T, D>,
        SN: Copy + Surface<T, D>,
        const N: usize,
        const D: usize,
    > CompoundPrism<T, S0, SI, SN, N, D>
{
    pub fn propagate_internal_helper(
        &self,
        ray: Ray<T, D>,
        wavelength: T,
    ) -> Result<Ray<T, D>, RayTraceError> {
        let initial_refractive_index = self.initial_glass.calc_n(wavelength);
        let refractive_indicies = self.glasses.map(|glass| glass.calc_n(wavelength));
        ray.propagate_internal(
            initial_refractive_index,
            refractive_indicies,
            self.initial_surface,
            self.inter_surfaces.map(|surface| surface),
            self.final_surface,
            self.ar_coated,
        )
    }
}
