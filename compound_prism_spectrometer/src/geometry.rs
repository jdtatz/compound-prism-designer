use crate::utils::*;
use crate::vector::{UnitVector, Vector};
use core::ops::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricRay<T, const N: usize> {
    pub origin: Vector<T, N>,
    pub direction: UnitVector<T, N>,
}

impl<T: FloatExt, const N: usize> GeometricRay<T, N> {
    pub fn translate(self, distance: T) -> Self {
        let Self { origin, direction } = self;
        let translated = direction.mul_add(distance, origin);
        Self {
            origin: translated,
            direction,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricRayIntersection<T, const N: usize> {
    pub distance: T,
    pub normal: UnitVector<T, N>,
}

pub trait Bounds<T, const N: usize>: Sized {
    fn in_bounds(self, position: Vector<T, N>) -> bool;
}

pub trait HyperSurface<T, B: Bounds<T, N>, const N: usize>: Sized {
    /// If there are multiple intersections, the closest one is returned
    fn intersection(
        self,
        ray: GeometricRay<T, N>,
        bounds: B,
    ) -> Option<GeometricRayIntersection<T, N>>;
}

#[derive(Debug, Clone, Copy)]
pub struct NoBounds;

impl<T, const N: usize> Bounds<T, N> for NoBounds {
    fn in_bounds(self, _: Vector<T, N>) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct PrismBounds<T> {
    pub height: T,
    pub half_width: T,
}

impl<T: ConstZero + PartialOrd> Bounds<T, 2> for PrismBounds<T> {
    fn in_bounds(self, position: Vector<T, 2>) -> bool {
        let Vector([_, y]) = position;
        T::ZERO <= y && y <= self.height
    }
}

impl<T: Copy + ConstZero + Neg<Output = T> + PartialOrd> Bounds<T, 3> for PrismBounds<T> {
    fn in_bounds(self, position: Vector<T, 3>) -> bool {
        let Vector([_, y, z]) = position;
        T::ZERO <= y && y <= self.height && -self.half_width <= z && z <= self.half_width
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct RadialBounds<T, const N: usize> {
    pub center: Vector<T, N>,
    pub radius_squared: T,
}

impl<T: ConstZero + Float, const N: usize> Bounds<T, N> for RadialBounds<T, N> {
    fn in_bounds(self, position: Vector<T, N>) -> bool {
        (position - self.center).norm_squared() <= self.radius_squared
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct SandwichBounds<T, const N: usize> {
    pub center: Vector<T, N>,
    pub normal: UnitVector<T, N>,
    pub height: T,
}

impl<T: FloatExt, const N: usize> Bounds<T, N> for SandwichBounds<T, N> {
    fn in_bounds(self, position: Vector<T, N>) -> bool {
        let dp = self.normal.dot(position - self.center);
        T::ZERO <= dp && dp <= self.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct HyperPlane<T, const N: usize> {
    pub point: Vector<T, N>,
    pub normal: UnitVector<T, N>,
}

impl<T: Copy + ConstZero + Neg<Output = T> + Float, B: Bounds<T, N>, const N: usize>
    HyperSurface<T, B, N> for HyperPlane<T, N>
{
    fn intersection(
        self,
        ray: GeometricRay<T, N>,
        bounds: B,
    ) -> Option<GeometricRayIntersection<T, N>> {
        let GeometricRay { origin, direction } = ray;
        let ci = direction.dot(*self.normal);
        // maybe use float_eq::float_eq!(ci, T::ZERO, ulps <= 10)
        if ci == T::ZERO {
            return None;
        }
        let d = (self.point - origin).dot(*self.normal) / ci;
        if d >= T::ZERO && bounds.in_bounds(direction.mul_add(d, origin)) {
            Some(GeometricRayIntersection {
                distance: d,
                normal: self.normal,
            })
        } else {
            None
        }
    }
}

pub trait FromParametrizedHyperPlane<T, const N: usize> {
    type Parametrization;

    fn from_hyperplane(
        hyperplane: HyperPlane<T, N>,
        parametrization: Self::Parametrization,
    ) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuadricIntersection<T> {
    None,
    Tangent(T),
    Secant(T, T),
}

fn spherical_ray_intersection<T: FloatExt, const N: usize>(
    center: Vector<T, N>,
    radius_squared: T,
    ray: GeometricRay<T, N>,
) -> QuadricIntersection<T> {
    let GeometricRay { origin, direction } = ray;
    debug_assert!(direction.is_unit());
    let delta = origin - center;
    let ud = direction.dot(delta);
    let discriminant = ud.sqr() - delta.norm_squared() + radius_squared;
    if discriminant > T::ZERO {
        let sd = discriminant.sqrt();
        QuadricIntersection::Secant(-ud - sd, -ud + sd)
    } else if discriminant < T::ZERO {
        QuadricIntersection::None
    } else {
        QuadricIntersection::Tangent(-ud)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct SphericalLikeSurface<T, P, const N: usize> {
    pub center: Vector<T, N>,
    pub radius: T,
    marker: core::marker::PhantomData<P>,
}

impl<T: FloatExt, P, B: Copy + Bounds<T, 2>> HyperSurface<T, B, 2>
    for SphericalLikeSurface<T, P, 2>
{
    fn intersection(
        self,
        ray: GeometricRay<T, 2>,
        bounds: B,
    ) -> Option<GeometricRayIntersection<T, 2>> {
        let GeometricRay { origin, direction } = ray;
        let inter = spherical_ray_intersection(self.center, self.radius.sqr(), ray);
        let (distance, p) = if let QuadricIntersection::Secant(d1, d2) = inter {
            let p1 = direction.mul_add(d1, origin);
            let p2 = direction.mul_add(d2, origin);
            if d1.is_sign_positive() && bounds.in_bounds(p1) {
                (d1, p1)
            } else if d2.is_sign_positive() && bounds.in_bounds(p2) {
                (d2, p2)
            } else {
                return None;
            }
        } else {
            return None;
        };
        let normal = UnitVector::new((self.center - p) / self.radius);
        Some(GeometricRayIntersection { distance, normal })
    }
}

impl<T: FloatExt, const N: usize> FromParametrizedHyperPlane<T, N>
    for SphericalLikeSurface<T, (), N>
{
    type Parametrization = (T, T);

    fn from_hyperplane(
        hyperplane: HyperPlane<T, N>,
        parametrization: Self::Parametrization,
    ) -> Self {
        let (chord_length, signed_normalized_curvature) = parametrization;
        // let signed_curvature = signed_normalized_curvature / (chord_length * T::lossy_from(0.5f64));
        let radius = chord_length * T::lossy_from(0.5f64) / signed_normalized_curvature.abs();
        let apothem = (radius.sqr() - chord_length.sqr() * T::lossy_from(0.25f64)).sqrt();
        // let sagitta = radius - apothem;
        let center = hyperplane.normal.mul_add(
            apothem.copy_sign(signed_normalized_curvature),
            hyperplane.point,
        );
        // let midpt = hyperplane.normal.mul_add(-(sagitta.copy_sign(signed_normalized_curvature)), hyperplane.point);
        Self {
            center,
            radius,
            marker: core::marker::PhantomData,
        }
    }
}

impl<T: FloatExt> FromParametrizedHyperPlane<T, 2> for CurvedPlane<T> {
    type Parametrization = (T, T);

    fn from_hyperplane(
        hyperplane: HyperPlane<T, 2>,
        parametrization: Self::Parametrization,
    ) -> Self {
        let (chord_length, signed_normalized_curvature) = parametrization;
        // let signed_curvature = signed_normalized_curvature / (chord_length * T::lossy_from(0.5f64));
        let radius = chord_length * T::lossy_from(0.5f64) / signed_normalized_curvature.abs();
        let apothem = (radius.sqr() - chord_length.sqr() * T::lossy_from(0.25f64)).sqrt();
        let sagitta = radius - apothem;
        let center = hyperplane.normal.mul_add(
            apothem.copy_sign(signed_normalized_curvature),
            hyperplane.point,
        );
        let midpt = hyperplane.normal.mul_add(
            -(sagitta.copy_sign(signed_normalized_curvature)),
            hyperplane.point,
        );
        let max_dist_sq = sagitta.sqr() + chord_length.sqr() * T::lossy_from(0.25f64);
        Self {
            surface: SphericalLikeSurface {
                center,
                radius,
                marker: core::marker::PhantomData,
            },
            bounds: RadialBounds {
                center: midpt,
                radius_squared: max_dist_sq,
            },
            marker: core::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct BoundedHyperSurface<T, S, B, const N: usize> {
    pub surface: S,
    pub bounds: B,
    pub marker: core::marker::PhantomData<Vector<T, N>>,
}

impl<T, S, B, const N: usize> BoundedHyperSurface<T, S, B, N> {
    pub fn new(surface: S, bounds: B) -> Self {
        Self {
            surface,
            bounds,
            marker: core::marker::PhantomData,
        }
    }
}

pub type Plane<T, const N: usize> = BoundedHyperSurface<T, HyperPlane<T, N>, PrismBounds<T>, N>;
pub type CurvedPlane<T, P = ()> =
    BoundedHyperSurface<T, SphericalLikeSurface<T, P, 2>, RadialBounds<T, 2>, 2>;

pub trait Surface<T, const N: usize>: Sized {
    fn intersection(self, ray: GeometricRay<T, N>) -> Option<GeometricRayIntersection<T, N>>;
}

impl<T, S, B, const N: usize> Surface<T, N> for BoundedHyperSurface<T, S, B, N>
where
    S: HyperSurface<T, B, N>,
    B: Bounds<T, N>,
{
    fn intersection(self, ray: GeometricRay<T, N>) -> Option<GeometricRayIntersection<T, N>> {
        let Self {
            surface, bounds, ..
        } = self;
        surface.intersection(ray, bounds)
    }
}

// TODO needs updating to new Vector impl
impl<T: FloatExt, const N: usize> Plane<T, N> {
    pub(crate) fn end_points(self, height: T) -> [Vector<T, N>; 2] {
        let dx = self.surface.point.tan_xy() * height * T::lossy_from(0.5f64);
        let ux = self.surface.point.x() - dx;
        let lx = self.surface.point.x() + dx;
        [Vector::from_xy(ux, height), Vector::from_xy(lx, T::ZERO)]
    }
}

// TODO needs updating to new Vector impl
impl<T: FloatExt> CurvedPlane<T> {
    pub(crate) fn end_points(self, height: T) -> [Vector<T, 2>; 2] {
        let max_dist_sq = self.bounds.radius_squared;
        let theta_2 = T::lossy_from(2u32)
            * (max_dist_sq.sqrt() / (T::lossy_from(2u32) * self.surface.radius)).asin();
        let r = self.bounds.center - self.surface.center;
        let u = self.surface.center + r.rotate_xy(theta_2);
        let l = self.surface.center + r.rotate_xy(-theta_2);
        debug_assert!(
            (u.y() - height).abs() < T::lossy_from(1e-4f64),
            "{:?} {}",
            u,
            height
        );
        debug_assert!(l.y().abs() < T::lossy_from(1e-4f64), "{:?}", l);
        debug_assert!(
            ((u - r).norm_squared() - max_dist_sq).abs() / max_dist_sq < T::lossy_from(1e-4f64)
        );
        debug_assert!(
            ((l - r).norm_squared() - max_dist_sq).abs() / max_dist_sq < T::lossy_from(1e-4f64)
        );
        [
            Vector::from_xy(u.x(), height),
            Vector::from_xy(l.x(), T::ZERO),
        ]
    }
}

// TODO needs updating to new Vector impl
pub(crate) fn create_joined_trapezoids<T: FloatExt, const N: usize, const D: usize>(
    height: T,
    first_angle: T,
    angles: [T; N],
    last_angle: T,
    first_sep_length: T,
    sep_lengths: [T; N],
) -> (HyperPlane<T, D>, [HyperPlane<T, D>; N], HyperPlane<T, D>) {
    let (sep_lengths, last_sep_length) = array_prepend(first_sep_length, sep_lengths);
    let h2 = height * T::lossy_from(0.5f64);
    let normal = UnitVector::new(Vector::angled_xy(first_angle).rot_180_xy());
    debug_assert_eq!(
        normal,
        UnitVector::new(Vector::from_xy(-T::one(), T::ZERO).rotate_xy(first_angle))
    );
    #[cfg(debug_assertions)]
    float_eq::assert_float_eq!(
        normal.tan_xy().abs(),
        first_angle.tan().abs(),
        rmax <= LossyFrom::lossy_from(1e-5f64)
    );
    let first = HyperPlane {
        point: Vector::from_xy(normal.tan_xy().abs() * h2, h2),
        normal,
    };
    // let first = Plane {
    //     height,
    //     normal,
    //     midpt: V::from_xy(normal.tan_xy().abs() * h2, h2),
    // };
    let mut prev_sign = normal.y().is_sign_positive();
    let mut d1 = first.point.x();
    let mut mx = first.point.x();
    let mut next_plane = |(angle, sep_len): (T, T)| {
        let normal = UnitVector::new(Vector::angled_xy(angle).rot_180_xy());
        let sign = normal.y().is_sign_positive();
        let d2 = normal.tan_xy().abs() * h2;
        let sep_dist = sep_len
            + if prev_sign != sign {
                d1 + d2
            } else {
                (d1 - d2).abs()
            };
        prev_sign = sign;
        d1 = d2;
        mx += sep_dist;
        HyperPlane {
            point: Vector::from_xy(mx, h2),
            normal,
        }
    };
    let inter: [HyperPlane<T, D>; N] = angles.zip(sep_lengths).map(&mut next_plane);
    let last = (&mut next_plane)((last_angle, last_sep_length));
    (first, inter, last)
}
