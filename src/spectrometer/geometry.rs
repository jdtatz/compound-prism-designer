use super::drawable::{Drawable, Path, Point};
use super::utils::*;
use super::vector::{UnitVector, Vector};
use core::ops::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricRay<V, const DIM: usize> {
    pub origin: V,
    pub direction: UnitVector<V>,
}

impl<V: Vector<DIM>, const DIM: usize> GeometricRay<V, DIM>
where
    <V as Vector<DIM>>::Scalar: FloatExt,
{
    pub fn translate(self, distance: V::Scalar) -> Self {
        let Self { origin, direction } = self;
        let translated = direction.mul_add(distance, origin);
        Self {
            origin: translated,
            direction,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricRayIntersection<T, V> {
    pub distance: T,
    pub normal: UnitVector<V>,
}

pub trait Bounds<V, const DIM: usize>: Sized {
    fn in_bounds(&self, position: V) -> bool;
}

pub trait HyperSurface<V: Vector<DIM>, B: Bounds<V, DIM>, const DIM: usize>: Sized {
    /// If there are multiple intersections, the closest one is returned
    fn intersection(
        &self,
        ray: GeometricRay<V, DIM>,
        bounds: &B,
    ) -> Option<GeometricRayIntersection<V::Scalar, V>>;
}

#[derive(Debug, Clone, Copy)]
pub struct NoBounds;

impl<V: Vector<DIM>, const DIM: usize> Bounds<V, DIM> for NoBounds {
    fn in_bounds(&self, _: V) -> bool {
        true
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct PrismBounds<T> {
    pub height: T,
    pub half_width: T,
}

impl<T: ConstZero + PartialOrd, V: Vector<2, Scalar = T>> Bounds<V, 2> for PrismBounds<T> {
    fn in_bounds(&self, position: V) -> bool {
        let [_, y] = position.to_array();
        T::ZERO <= y && y <= self.height
    }
}

impl<T: Copy + ConstZero + Neg<Output = T> + PartialOrd, V: Vector<3, Scalar = T>> Bounds<V, 3>
    for PrismBounds<T>
{
    fn in_bounds(&self, position: V) -> bool {
        let [_, y, z] = position.to_array();
        T::ZERO <= y && y <= self.height && -self.half_width <= z && z <= self.half_width
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct RadialBounds<T, V> {
    pub center: V,
    pub radius_squared: T,
}

impl<T: ConstZero + Float, V: Vector<DIM, Scalar = T>, const DIM: usize> Bounds<V, DIM>
    for RadialBounds<T, V>
{
    fn in_bounds(&self, position: V) -> bool {
        (position - self.center).norm_squared() <= self.radius_squared
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct SandwichBounds<T, V> {
    pub center: V,
    pub normal: UnitVector<V>,
    pub height: T,
}

impl<T: FloatExt, V: Vector<DIM, Scalar = T>, const DIM: usize> Bounds<V, DIM>
    for SandwichBounds<T, V>
{
    fn in_bounds(&self, position: V) -> bool {
        let dp = self.normal.dot(position - self.center);
        T::ZERO <= dp && dp <= self.height
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct HyperPlane<V> {
    pub point: V,
    pub normal: UnitVector<V>,
}

impl<
        T: Copy + ConstZero + Neg<Output = T> + Float,
        B: Bounds<V, DIM>,
        V: Vector<DIM, Scalar = T>,
        const DIM: usize,
    > HyperSurface<V, B, DIM> for HyperPlane<V>
{
    fn intersection(
        &self,
        ray: GeometricRay<V, DIM>,
        bounds: &B,
    ) -> Option<GeometricRayIntersection<T, V>> {
        // https://en.wikipedia.org/wiki/Line–plane_intersection
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

pub trait FromParametrizedHyperPlane<V, const DIM: usize> {
    type Parametrization;

    fn from_hyperplane(hyperplane: HyperPlane<V>, parametrization: Self::Parametrization) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuadricIntersection<T> {
    None,
    Tangent(T),
    Secant(T, T),
}

fn spherical_ray_intersection<T: FloatExt, V: Vector<DIM, Scalar = T>, const DIM: usize>(
    center: V,
    radius_squared: T,
    ray: GeometricRay<V, DIM>,
) -> QuadricIntersection<T> {
    // https://en.wikipedia.org/wiki/Line–sphere_intersection
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
pub struct SphericalLikeSurface<T, V, P> {
    pub center: V,
    pub radius: T,
    marker: core::marker::PhantomData<P>,
}

impl<T: FloatExt, P, V: Vector<2, Scalar = T>, B: Copy + Bounds<V, 2>> HyperSurface<V, B, 2>
    for SphericalLikeSurface<T, V, P>
{
    fn intersection(
        &self,
        ray: GeometricRay<V, 2>,
        bounds: &B,
    ) -> Option<GeometricRayIntersection<T, V>> {
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

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct PlaneParametrization<T> {
    pub height: T,
    pub width: T,
}

impl<V: Vector<DIM>, const DIM: usize> FromParametrizedHyperPlane<V, DIM> for Plane<V, DIM>
where
    <V as Vector<DIM>>::Scalar: FloatExt,
{
    type Parametrization = PlaneParametrization<V::Scalar>;

    fn from_hyperplane(hyperplane: HyperPlane<V>, parametrization: Self::Parametrization) -> Self {
        let PlaneParametrization { height, width } = parametrization;
        let bounds = PrismBounds {
            height,
            half_width: width * V::Scalar::lossy_from(0.5_f64),
        };
        Self {
            surface: hyperplane,
            bounds,
            marker: core::marker::PhantomData,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct CurvedPlaneParametrization<T> {
    pub signed_normalized_curvature: T,
    pub height: T,
}

impl<T: FloatExt, V: Vector<DIM, Scalar = T>, const DIM: usize> FromParametrizedHyperPlane<V, DIM>
    for SphericalLikeSurface<T, V, ()>
{
    type Parametrization = CurvedPlaneParametrization<T>;

    fn from_hyperplane(hyperplane: HyperPlane<V>, parametrization: Self::Parametrization) -> Self {
        let CurvedPlaneParametrization {
            height,
            signed_normalized_curvature,
        } = parametrization;
        let chord_length = hyperplane.normal.sec_xy(height).abs();
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

impl<T: FloatExt, V: Vector<2, Scalar = T>> FromParametrizedHyperPlane<V, 2> for CurvedPlane<V, 2> {
    type Parametrization = CurvedPlaneParametrization<T>;

    fn from_hyperplane(hyperplane: HyperPlane<V>, parametrization: Self::Parametrization) -> Self {
        let CurvedPlaneParametrization {
            height,
            signed_normalized_curvature,
        } = parametrization;
        let chord_length = hyperplane.normal.sec_xy(height).abs();
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
pub struct BoundedHyperSurface<V, S, B, const DIM: usize> {
    pub surface: S,
    pub bounds: B,
    pub marker: core::marker::PhantomData<V>,
}

impl<V, S, B, const DIM: usize> BoundedHyperSurface<V, S, B, DIM> {
    pub fn new(surface: S, bounds: B) -> Self {
        Self {
            surface,
            bounds,
            marker: core::marker::PhantomData,
        }
    }
}

pub type Plane<V, const DIM: usize> =
    BoundedHyperSurface<V, HyperPlane<V>, PrismBounds<<V as Vector<DIM>>::Scalar>, DIM>;
pub type CurvedPlane<V, const DIM: usize> = BoundedHyperSurface<
    V,
    SphericalLikeSurface<<V as Vector<DIM>>::Scalar, V, ()>,
    RadialBounds<<V as Vector<DIM>>::Scalar, V>,
    DIM,
>;

pub trait Surface<V: Vector<DIM>, const DIM: usize>: Sized {
    fn intersection(
        &self,
        ray: GeometricRay<V, DIM>,
    ) -> Option<GeometricRayIntersection<V::Scalar, V>>;
}

impl<V: Vector<DIM>, S, B, const DIM: usize> Surface<V, DIM> for BoundedHyperSurface<V, S, B, DIM>
where
    S: HyperSurface<V, B, DIM>,
    B: Bounds<V, DIM>,
{
    fn intersection(
        &self,
        ray: GeometricRay<V, DIM>,
    ) -> Option<GeometricRayIntersection<V::Scalar, V>> {
        let Self {
            surface, bounds, ..
        } = self;
        surface.intersection(ray, bounds)
    }
}

impl<T: FloatExt, V: Vector<DIM, Scalar = T>, const DIM: usize> Drawable<T> for Plane<V, DIM> {
    fn draw(&self) -> Path<T> {
        let height = self.bounds.height;
        let dx = self.surface.normal.tan_xy() * height * T::lossy_from(0.5f64);
        let ux = self.surface.point.x() - dx;
        let lx = self.surface.point.x() + dx;
        Path::Line {
            a: Point { x: ux, y: height },
            b: Point { x: lx, y: T::ZERO },
        }
    }
}

impl<T: FloatExt, V: Vector<DIM, Scalar = T>, const DIM: usize> Drawable<T>
    for CurvedPlane<V, DIM>
{
    fn draw(&self) -> Path<T> {
        let max_dist_sq = self.bounds.radius_squared;
        let theta_2 = T::lossy_from(2u32)
            * (max_dist_sq.sqrt() / (T::lossy_from(2u32) * self.surface.radius)).asin();
        let r = self.bounds.center - self.surface.center;
        let mut u = self.surface.center + r.rotate_xy(theta_2);
        let mut l = self.surface.center + r.rotate_xy(-theta_2);
        if u.y() < l.y() {
            core::mem::swap(&mut l, &mut u);
        }
        let height = u.y();
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
        Path::Arc {
            a: Point {
                x: u.x(),
                y: height,
            },
            b: Point {
                x: l.x(),
                y: T::ZERO,
            },
            midpt: Point {
                x: self.bounds.center.x(),
                y: self.bounds.center.y(),
            },
            // center: Point {
            //     x: self.surface.center.x(),
            //     y: self.surface.center.y(),
            // },
            radius: self.surface.radius,
        }
    }
}

// TODO needs updating to new Vector impl
pub(crate) fn create_joined_trapezoids<
    T: FloatExt,
    V: Vector<DIM, Scalar = T>,
    const N: usize,
    const DIM: usize,
>(
    height: T,
    first_angle: T,
    angles: [T; N],
    last_angle: T,
    first_sep_length: T,
    sep_lengths: [T; N],
) -> (HyperPlane<V>, [HyperPlane<V>; N], HyperPlane<V>) {
    let (sep_lengths, last_sep_length) = array_prepend(first_sep_length, sep_lengths);
    let h2 = height * T::lossy_from(0.5f64);
    let normal = UnitVector::new(V::angled_xy(first_angle).rot_180_xy());
    debug_assert_eq!(
        normal,
        UnitVector::new(V::from_xy(-T::one(), T::ZERO).rotate_xy(first_angle))
    );
    #[cfg(debug_assertions)]
    float_eq::assert_float_eq!(
        normal.tan_xy().abs(),
        first_angle.tan().abs(),
        rmax <= LossyFrom::lossy_from(1e-5f64)
    );
    let first = HyperPlane {
        point: V::from_xy(normal.tan_xy().abs() * h2, h2),
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
        let normal = UnitVector::new(V::angled_xy(angle).rot_180_xy());
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
            point: V::from_xy(mx, h2),
            normal,
        }
    };
    let inter: [HyperPlane<V>; N] = angles.zip(sep_lengths).map(&mut next_plane);
    let last = (&mut next_plane)((last_angle, last_sep_length));
    (first, inter, last)
}

// TODO needs updating to new Vector impl
/// Find the position and orientation of the detector array,
/// parameterized by the minimum and maximum wavelengths of the input beam,
/// and its angle from the normal.
pub fn fit_ray_difference_surface<T: FloatExt, V: Vector<DIM, Scalar = T>, const DIM: usize>(
    lower_ray: GeometricRay<V, DIM>,
    upper_ray: GeometricRay<V, DIM>,
    d_length: T,
    d_angle: T,
) -> Option<(V, bool)> {
    let spec_dir = V::angled_xy(d_angle).rot_90_xy();
    let spec = spec_dir * d_length;

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
    let imat = mat_inverse(mat)?;
    let temp = spec - upper_ray.origin + lower_ray.origin;
    let [_d1, d2] = mat_mul(imat, [temp.x(), temp.y()]);
    let l_vertex = lower_ray.direction.mul_add(d2, lower_ray.origin);
    Some(if d2 > T::zero() {
        (l_vertex, false)
    } else {
        let temp = -spec - upper_ray.origin + lower_ray.origin;
        let [_d1, d2] = mat_mul(imat, [temp.x(), temp.y()]);
        if d2 < T::zero() {
            return None;
        }
        let u_vertex = lower_ray.direction.mul_add(d2, lower_ray.origin);
        (u_vertex, true)
    })
}
