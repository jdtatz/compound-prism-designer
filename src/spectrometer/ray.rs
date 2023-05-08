use core::borrow::Borrow;

use super::drawable::Polygon;
use super::geometry::*;
use super::glasscat::Glass;
use super::utils::*;
use super::vector::{UnitVector, Vector};
use super::Drawable;

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

pub trait DetectorArray<V: Vector<DIM>, const DIM: usize>: Surface<V, DIM> {
    fn bin_count(&self) -> u32;
    fn length(&self) -> V::Scalar;
    fn bin_index(&self, intersection: V) -> Option<u32>;
    fn mid_pt(&self) -> V;
}

#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(wrapped = "crate::LossyFrom::lossy_from")]
pub struct PrismSurface<T, S> {
    /// glass
    pub glass: Glass<T, 6>,
    /// initial boundary surface
    pub surface: S,
}

pub trait ArrayLikeFamily {
    type Array<T>: ?Sized + Borrow<[T]>;

    fn as_slice<T>(array: &Self::Array<T>) -> &[T] {
        Borrow::borrow(array)
    }
}

pub struct SliceFamily;
pub struct ArrayFamily<const N: usize>;

impl ArrayLikeFamily for SliceFamily {
    type Array<T> = [T];
}

impl<const N: usize> ArrayLikeFamily for ArrayFamily<N> {
    type Array<T> = [T; N];
}

pub trait Array: Borrow<[Self::Item]> {
    type Item;

    fn as_slice(&self) -> &[Self::Item] {
        self.borrow()
    }
}

impl<T> Array for [T] {
    type Item = T;
}

impl<T, const N: usize> Array for [T; N] {
    type Item = T;
}

/// Compound Prism Specification
#[derive(Debug, Clone, Copy, WrappedFrom)]
#[wrapped_from(wrapped = "crate::LossyFrom::lossy_from")]
pub struct CompoundPrism<T, S0, SI, SN, L: ?Sized + Array<Item = PrismSurface<T, SI>>> {
    /// initial prism surface
    initial_prism: PrismSurface<T, S0>,
    /// Final boundary surface
    final_surface: SN,
    /// Height of compound prism
    pub(crate) height: T,
    /// Width of compound prism
    pub(crate) width: T,
    /// Are the inter-media surfaces coated(anti-reflective)?
    #[wrapped_from(skip)]
    ar_coated: bool,
    /// rest of the prism surfaces
    prisms: L,
}

pub type CompoundPrismTypeHelper<T, S0, SI, SN, A> =
    CompoundPrism<T, S0, SI, SN, <A as ArrayLikeFamily>::Array<PrismSurface<T, SI>>>;

pub type FocusingPlanerCompoundPrism<V, A> = CompoundPrismTypeHelper<
    <V as Vector<2>>::Scalar,
    Plane<V, 2>,
    Plane<V, 2>,
    CurvedPlane<V, 2>,
    A,
>;

pub type CulminatingToricCompoundPrism<V, A> = CompoundPrismTypeHelper<
    <V as Vector<3>>::Scalar,
    crate::ToricLens<V, 3>,
    Plane<V, 3>,
    crate::ToricLens<V, 3>,
    A,
>;

impl<T: FloatExt, V: Vector<DIM, Scalar = T>, S0, SN, const N: usize, const DIM: usize>
    CompoundPrism<T, S0, Plane<V, DIM>, SN, [PrismSurface<T, Plane<V, DIM>>; N]>
where
    S0: Surface<V, DIM> + FromParametrizedHyperPlane<V, DIM>,
    SN: Surface<V, DIM> + FromParametrizedHyperPlane<V, DIM>,
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
        let initial_prism = PrismSurface {
            glass: glass0,
            surface: surface0,
        };
        let isurfaces = isurfaces.map(|s| Plane::new(s, prism_bounds));
        let prisms = glasses
            .zip(isurfaces)
            .map(|(glass, surface)| PrismSurface { glass, surface });
        let final_surface = SN::from_hyperplane(last_surface, sn_parametrization);
        Self {
            initial_prism,
            prisms,
            final_surface,
            height,
            width,
            ar_coated: coat,
        }
    }
}

impl<T: FloatExt, S0: Drawable<T>, SI: Copy + Drawable<T>, SN: Drawable<T>, const N: usize>
    CompoundPrism<T, S0, SI, SN, [PrismSurface<T, SI>; N]>
{
    pub fn surfaces<V: Vector<DIM, Scalar = T>, const DIM: usize>(
        &self,
    ) -> (
        (crate::Point<T>, crate::Point<T>, Option<T>),
        [(crate::Point<T>, crate::Point<T>, Option<T>); N],
        (crate::Point<T>, crate::Point<T>, Option<T>),
    )
    where
        S0: Surface<V, DIM>,
        SI: Surface<V, DIM>,
        SN: Surface<V, DIM>,
    {
        let path2surface = |p: crate::Path<T>| match p {
            crate::Path::Line { a, b } => (a, b, None),
            crate::Path::Arc {
                a,
                b,
                midpt: _,
                radius,
            } => (a, b, Some(radius)),
        };
        let s_0 = path2surface(self.initial_prism.surface.draw());
        let s_i = self
            .prisms
            .map(|PrismSurface { surface: s, .. }| path2surface(s.draw()));
        let s_n = path2surface(self.final_surface.draw());
        (s_0, s_i, s_n)
    }

    pub fn polygons(&self) -> ([Polygon<T>; N], Polygon<T>) {
        let mut path0 = self.initial_prism.surface.draw();
        let polys = self.prisms.map(|PrismSurface { surface: s, .. }| {
            let path1 = s.draw();
            let poly = Polygon([path0.reverse(), path1]);
            path0 = path1;
            poly
        });
        let path1 = self.final_surface.draw();
        let final_poly = Polygon([path0.reverse(), path1]);
        (polys, final_poly)
    }

    pub fn final_midpt(&self) -> crate::Point<T> {
        let path = self.final_surface.draw();
        let midpt = match path {
            crate::Path::Line { a, b } => crate::Point {
                x: (a.x + b.x) * T::lossy_from(0.5),
                y: (a.y + b.y) * T::lossy_from(0.5),
            },
            crate::Path::Arc { midpt, .. } => midpt,
        };
        midpt
    }
}

/// Light Ray
#[derive(Constructor, Debug, PartialEq, Clone, Copy)]
pub struct Ray<V: Vector<DIM>, const DIM: usize> {
    /// Origin position vector
    pub origin: V,
    /// Unit normal direction vector
    pub direction: UnitVector<V>,
    /// S-Polarization Transmittance probability
    s_transmittance: V::Scalar,
    /// P-Polarization Transmittance probability
    p_transmittance: V::Scalar,
}

impl<T: FloatExt, V: Vector<DIM, Scalar = T>, const DIM: usize> Ray<V, DIM> {
    /// Create a new unpolarized ray with full transmittance
    ///
    /// # Arguments
    ///  * `origin` - the initial y value of the ray's position
    ///  * `direction` - the initial y value of the ray's position
    pub fn new_unpolarized(origin: V, direction: UnitVector<V>) -> Self {
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
        Self::new_unpolarized(V::from_xy(T::ZERO, y), UnitVector::unit_x())
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
        normal: UnitVector<V>,
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
    pub fn surface_propagate<S: Surface<V, DIM>>(
        self,
        surface: &S,
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

    /// Find the intersection position of the ray with the detector array
    /// and the ray's transmission probability. The intersection position is a
    /// scalar on the line defined by the detector array.
    ///
    /// # Arguments
    ///  * `detector` - detector array specification
    pub(crate) fn intersect_detector_array(
        self,
        detector: &(impl Copy + DetectorArray<V, DIM>),
    ) -> Result<(V, u32, T), RayTraceError> {
        let GeometricRayIntersection { distance, .. } = detector
            .intersection(self.into())
            .ok_or(RayTraceError::SpectrometerAngularResponseTooWeak)?;
        let p = self.direction.mul_add(distance, self.origin);
        let bin_idx = detector
            .bin_index(p)
            .ok_or(RayTraceError::NoSurfaceIntersection)?;
        Ok((p, bin_idx, self.average_transmittance()))
    }
}

pub trait CompoundPrismPropagatee<V: Vector<DIM>, const DIM: usize>: Sized
where
    <V as Vector<DIM>>::Scalar: Float,
{
    type Error;

    fn initial_surface<S: Surface<V, DIM>>(
        self,
        refraction_index: V::Scalar,
        surface: &S,
        ar_coated: bool,
    ) -> Result<Self, Self::Error> {
        self.next_surface(V::Scalar::one(), refraction_index, surface, ar_coated)
    }
    fn next_surface<S: Surface<V, DIM>>(
        self,
        prev_refraction_index: V::Scalar,
        refraction_index: V::Scalar,
        surface: &S,
        ar_coated: bool,
    ) -> Result<Self, Self::Error>;
    fn final_surface<S: Surface<V, DIM>>(
        self,
        prev_refraction_index: V::Scalar,
        surface: &S,
        ar_coated: bool,
    ) -> Result<Self, Self::Error> {
        self.next_surface(prev_refraction_index, V::Scalar::one(), surface, ar_coated)
    }
}

pub trait GenericCompoundPrism<V: Vector<DIM>, const DIM: usize>
where
    <V as Vector<DIM>>::Scalar: Float,
{
    type PropagateeTrace<'s, P: 's + Copy + CompoundPrismPropagatee<V, DIM>>: 's
        + Iterator<Item = P>
    where
        Self: 's;
    fn propagate<P: CompoundPrismPropagatee<V, DIM>>(
        &self,
        propagatee: P,
        wavelength: V::Scalar,
    ) -> Result<P, P::Error>;
    fn propagate_trace<'s, P: 's + Copy + CompoundPrismPropagatee<V, DIM>>(
        &'s self,
        propagatee: P,
        wavelength: V::Scalar,
    ) -> Self::PropagateeTrace<'s, P>;
}

impl<V: Vector<DIM>, const DIM: usize> CompoundPrismPropagatee<V, DIM> for Ray<V, DIM>
where
    <V as Vector<DIM>>::Scalar: FloatExt,
{
    type Error = RayTraceError;

    fn next_surface<S: Surface<V, DIM>>(
        self,
        prev_refraction_index: V::Scalar,
        refraction_index: V::Scalar,
        surface: &S,
        ar_coated: bool,
    ) -> Result<Self, Self::Error> {
        self.surface_propagate(surface, prev_refraction_index, refraction_index, ar_coated)
    }
}

impl<
    T: FloatExt,
    V: Vector<DIM, Scalar = T>,
    S0: Copy + Surface<V, DIM>,
    SI: Copy + Surface<V, DIM>,
    SN: Copy + Surface<V, DIM>,
    L: ?Sized + Array<Item = PrismSurface<T, SI>>,
    const DIM: usize,
> GenericCompoundPrism<V, DIM> for CompoundPrism<T, S0, SI, SN, L>
{
    type PropagateeTrace<'s, P: 's + Copy + CompoundPrismPropagatee<V, DIM>> = impl 's + Iterator<Item=P> where P: Sized, Self: 's;

    fn propagate<P: CompoundPrismPropagatee<V, DIM>>(
        &self,
        mut propagatee: P,
        wavelength: V::Scalar,
    ) -> Result<P, P::Error> {
        let mut n1 = self.initial_prism.glass.calc_n(wavelength);
        propagatee = propagatee.initial_surface(n1, &self.initial_prism.surface, self.ar_coated)?;

        for prism in self.prisms.as_slice() {
            let n2 = prism.glass.calc_n(wavelength);
            propagatee = propagatee.next_surface(n1, n2, &prism.surface, self.ar_coated)?;
            n1 = n2;
        }
        propagatee.final_surface(n1, &self.final_surface, self.ar_coated)
    }

    fn propagate_trace<'s, P: 's + Copy + CompoundPrismPropagatee<V, DIM>>(
        &'s self,
        propagatee: P,
        wavelength: V::Scalar,
    ) -> Self::PropagateeTrace<'s, P> {
        let mut ray = propagatee.clone();
        let mut n1 = self.initial_prism.glass.calc_n(wavelength);
        let mut prism0 = Some(self.initial_prism);
        let mut prisms = self.prisms.as_slice().iter();
        let mut done = false;
        let mut propagation_fn = move || -> Result<Option<_>, P::Error> {
            if let Some(PrismSurface {
                glass: _,
                surface: surf,
            }) = prism0.take()
            {
                ray = ray.initial_surface(n1, &surf, self.ar_coated)?;
                return Ok(Some(ray));
            }
            match prisms.next() {
                Some(PrismSurface {
                    glass,
                    surface: surf,
                }) => {
                    let n2 = glass.calc_n(wavelength);
                    ray = ray.next_surface(n1, n2, surf, self.ar_coated)?;
                    n1 = n2;
                    Ok(Some(ray))
                }
                None if !done => {
                    done = true;
                    ray = ray.final_surface(n1, &self.final_surface, self.ar_coated)?;
                    Ok(Some(ray))
                }
                _ if done => Ok(None),
                _ => unreachable!(),
            }
        };
        core::iter::once(propagatee)
            .chain(core::iter::from_fn(move || propagation_fn().ok().flatten()).fuse())
    }
}

impl<V: Vector<DIM>, const DIM: usize> From<Ray<V, DIM>> for GeometricRay<V, DIM> {
    fn from(ray: Ray<V, DIM>) -> Self {
        let Ray {
            origin, direction, ..
        } = ray;
        Self { origin, direction }
    }
}
