use super::{cbrt::approx_cbrt, geometry::SandwichBounds, utils::Ring, *};

fn sqr<T: Copy + core::ops::Mul<T, Output = T>>(v: T) -> T {
    v * v
}

fn cube<T: Copy + core::ops::Mul<T, Output = T>>(v: T) -> T {
    v * v * v
}

/// The implicit equation of a torus $f(\vec{p}; R, r, \vec{c}, \hat{n}) = \left(\norm{\vec{p} - \vec{c}}^2 + R^2 - r^2 \right)^2 - 4 R^2 \left(\norm{\vec{p} - \vec{c}}^2 - \left((\vec{p} - \vec{c})\cdot \hat{n}\right)^2 \right)$
/// $\| \oproj_{\hat{n}}(\vec{p} - \vec{c}) \|^2$
/// poloidal direction is cylindrical
/// toroidal direction is spherical
#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(wrapped = "crate::LossyFrom::lossy_from")]
pub struct ToricSurface<V: Vector<3>> {
    center: V,
    toroidal_normal: UnitVector<V>,
    toroidal_radius: V::Scalar,
    poloidal_radius: V::Scalar,
}

impl<V: Vector<3>> ToricSurface<V>
where
    <V as Vector<3>>::Scalar: Copy + Ring + LossyFrom<u32>,
{
    pub fn implicit(self, p: V) -> V::Scalar {
        let u = p - self.center;
        sqr(u.norm_squared() + self.toroidal_radius.sqr() - self.poloidal_radius.sqr())
            - V::Scalar::lossy_from(4u32)
                * self.toroidal_radius.sqr()
                * super::vector::cross_prod_magnitude_sq(u, self.toroidal_normal)
    }
}

#[inline]
fn sympy_soln<F: FloatExt>(r_2: F, R_2: F, dp_magn_sq: F, dp_v: F, dp_n: F, n_v: F) -> [F; 4] {
    // from sympy import *
    // r_2, R_2 = symbols("r_2 R_2", positive=True)
    // dp_magn_sq, dp_v, dp_n, n_v = symbols("dp_magn_sq dp_v dp_n n_v", nonnegative=True)
    // _2, _4, _8 = 2, 4, 8
    // t = Symbol("t", real=True)
    // a = 1
    // b = _4 * dp_v
    // c = -_2 * (r_2 + R_2 - dp_magn_sq - _2 * dp_v**2 - _2 * R_2 * n_v**2)
    // d = -_4 * (r_2 + R_2 - dp_magn_sq) * dp_v + _8 * R_2 * dp_n * n_v
    // e = (r_2 - R_2)**2 - _2 * (R_2 + r_2) * dp_magn_sq + dp_magn_sq**2 + _4 * R_2 * dp_n**2
    // poly = a * t**4 + b * t**3 + c * t**2 + d * t + e
    // soln = solveset(Eq(poly, 0), t)
    // steps, fin = cse(soln)
    // for xi, v in steps:
    //     print(f"let {xi} = {v};")
    // print(fin[0])

    let _2 = F::lossy_from(2u32);
    let _3 = F::lossy_from(3u32);
    let _4 = F::lossy_from(4u32);
    let _6 = F::lossy_from(6u32);
    let _8 = F::lossy_from(8u32);
    let _12 = F::lossy_from(12u32);
    let _16 = F::lossy_from(16u32);
    let _27 = F::lossy_from(27u32);
    let _108 = F::lossy_from(108u32);
    let _216 = F::lossy_from(216u32);

    let x0 = R_2 * dp_v;
    let x1 = dp_magn_sq * dp_v;
    let x2 = dp_v * r_2;
    let x3 = dp_n * n_v;
    let x4 = R_2 * x3;
    let x5 = n_v.sqr();
    let x6 = _2 * R_2;
    let x7 = R_2 - dp_magn_sq + r_2 - x5 * x6;
    let x8 = _8 * dp_v * x7 - _8 * x0 + _8 * x1 - _8 * x2 + _16 * x4;
    let x9 = _2 * r_2;
    let x10 = dp_v.sqr();
    let x11 = R_2 * x5;
    let x12 = _2 * dp_magn_sq - _2 * x10 + _4 * x11 - x6 - x9;
    let x13 = cube(x12);
    let x14 = _4 * dp_v;
    let x15 = sqr(-_4 * x0 + _4 * x1 + x14 * x7 - _4 * x2 + _8 * x4);
    let x16 = R_2.sqr();
    let x17 = dp_magn_sq.sqr();
    let x18 = r_2.sqr();
    let x19 = dp_magn_sq * x6;
    let x20 = r_2 * x6;
    let x21 = dp_magn_sq * x9;
    let x22 = _4 * R_2 * dp_n.sqr();
    let x23 = x14
        * (-x0 + x1 + x14 * (R_2 / _8 - dp_magn_sq / _8 + r_2 / _8 - x10 / _16 - x11 / _4) - x2
            + x3 * x6);
    let x24 = x16 + x17 + x18 - x19 - x20 - x21 + x22 - x23;
    let x25 = x12 * x24;
    let x26 = -x13 / _108 - x15 / _8 + x25 / _3;
    let x27 = _2 * approx_cbrt(x26);
    let x28 = _8 * R_2 / _3;
    let x29 = _4 * R_2 / _3 - _4 * dp_magn_sq / _3 + _4 * r_2 / _3 + _4 * x10 / _3 - x28 * x5;
    let x30 = F::sqrt(-x27 + x29);
    let x31 = x8 / x30;
    let x32 = -_8 * dp_magn_sq / _3 + _8 * r_2 / _3 + _8 * x10 / _3 - _16 * x11 / _3 + x28;
    let x33 = x27 + x32;
    let x34 = F::sqrt(x31 + x33) / _2;
    let x35 = -dp_v;
    let x36 = x30 / _2;
    let x37 = x35 - x36;
    let x38 = x12.sqr() / _12;
    let x39 = x24 + x38 == F::ZERO;
    let x40 = -x16 - x17 - x18 + x19 + x20 + x21 - x22 + x23 - x38;
    let x41 =
        approx_cbrt(x13 / _216 + x15 / _16 - x25 / _6 + F::sqrt(sqr(x26) / _4 + cube(x40) / _27));
    let x42 = _2 * x41;
    let x43 = _2 * x40 / (_3 * x41);
    let x44 = F::sqrt(x29 + x42 - x43);
    let x45 = x8 / x44;
    let x46 = x32 - x42 + x43;
    let x47 = F::sqrt(x45 + x46) / _2;
    let x48 = x44 / _2;
    let x49 = x35 - x48;
    let x50 = F::sqrt(-x31 + x33) / _2;
    let x51 = x35 + x36;
    let x52 = F::sqrt(-x45 + x46) / _2;
    let x53 = x35 + x48;

    let r3 = if x39 { x34 + x37 } else { x47 + x49 };
    let r2 = if x39 { x50 + x51 } else { x52 + x53 };
    let r1 = if x39 { -x34 + x37 } else { -x47 + x49 };
    let r0 = if x39 { -x50 + x51 } else { -x52 + x53 };
    debug_assert!(
        r0 <= r1 && r1 <= r2 && r2 <= r3,
        "roots are unordered [{}, {}, {}, {}]",
        r0,
        r1,
        r2,
        r3
    );
    [r0, r1, r2, r3]
}

impl<T: FloatExt, V: Vector<3, Scalar = T>, B: Copy + Bounds<V, 3>> HyperSurface<V, B, 3>
    for ToricSurface<V>
{
    // $Assumptions =
    //  t \[Element] Reals && (p | c | n | v | u | dp) \[Element]
    //    Vectors[3, Reals] && n . n == 1 && v . v == 1
    // (u . u + \[ScriptCapitalR]^2 - r^2)^2 -
    //  4 \[ScriptCapitalR]^2 (u . u - (u . n)^2)
    // % /. u -> dp + v t
    // FunctionExpand[%] // TensorExpand
    // CoefficientList[%, t] // FullSimplify
    // MatrixForm[%]

    #[inline(always)]
    fn intersection(
        &self,
        ray: GeometricRay<V, 3>,
        bounds: &B,
    ) -> Option<GeometricRayIntersection<T, V>> {
        let GeometricRay { origin, direction } = ray;
        let dp = origin - self.center;
        let dp_magn_sq = dp.norm_squared();
        let dp_v = dp.dot(*direction);
        let dp_n = dp.dot(*self.toroidal_normal);
        let n_v = self.toroidal_normal.dot(*direction);
        let R_2 = self.toroidal_radius.sqr();
        let r_2 = self.poloidal_radius.sqr();
        // let _2 = T::lossy_from(2u32);
        // let _4 = T::lossy_from(4u32);
        // let _8 = T::lossy_from(8u32);
        // // quartic surface: a x^4 + b x^3 + c x^2 + d x + e == 0
        // let a = T::ONE;
        // let b = _4 * dp_v;
        // let c = -_2 * (r_2 + R_2 - dp_magn_sq - _2 * dp_v.sqr() - _2 * R_2 * n_v.sqr());
        // let d = -_4 * (r_2 + R_2 - dp_magn_sq) * dp_v + _8 * R_2 * dp_n * n_v;
        // let e = sqr(r_2 - R_2) - _2 * (R_2 + r_2) * dp_magn_sq
        //     + dp_magn_sq.sqr()
        //     + _4 * R_2 * dp_n.sqr();
        // let roots = crate::roots::find_roots_quartic(a, b, c, d, e);
        let check_root = |d| {
            if d < T::ZERO {
                None
            } else {
                let p = direction.mul_add(d, origin);
                if bounds.in_bounds(p) {
                    Some((d, p))
                } else {
                    None
                }
            }
        };
        // let (distance, p) = match roots {
        //     roots::Roots::No(_) | roots::Roots::One([_]) => None,
        //     roots::Roots::Two([sec1, sec2]) => check_root(sec1).or_else(|| check_root(sec2)),
        //     roots::Roots::Three([sec1, _tan, sec2]) => {
        //         check_root(sec1).or_else(|| check_root(sec2))
        //     }
        //     roots::Roots::Four([sec1, _sec2, _sec3, sec4]) => {
        //         check_root(sec1).or_else(|| check_root(sec4))
        //     }
        // }?;
        let [r0, r1, r2, r3] = sympy_soln(r_2, R_2, dp_magn_sq, dp_v, dp_n, n_v);
        float_eq::debug_assert_float_eq!(
            self.implicit(direction.mul_add(r0, origin)),
            T::ZERO,
            abs <= T::lossy_from(1e-5_f64)
        );
        float_eq::debug_assert_float_eq!(
            self.implicit(direction.mul_add(r1, origin)),
            T::ZERO,
            abs <= T::lossy_from(1e-5_f64)
        );
        float_eq::debug_assert_float_eq!(
            self.implicit(direction.mul_add(r2, origin)),
            T::ZERO,
            abs <= T::lossy_from(1e-5_f64)
        );
        float_eq::debug_assert_float_eq!(
            self.implicit(direction.mul_add(r3, origin)),
            T::ZERO,
            abs <= T::lossy_from(1e-5_f64)
        );
        let (distance, p) = check_root(r0)
            .or_else(|| check_root(r1))
            .or_else(|| check_root(r2))
            .or_else(|| check_root(r3))?;

        let t_center = super::vector::oproj(p - self.center, self.toroidal_normal)
            .normalize()
            .mul_add(self.toroidal_radius, self.center);
        let normal = UnitVector::new((t_center - p) / self.poloidal_radius);
        Some(GeometricRayIntersection { distance, normal })
    }
}

pub type ToricLens<V, const DIM: usize> =
    BoundedHyperSurface<V, ToricSurface<V>, SandwichBounds<<V as Vector<DIM>>::Scalar, V>, DIM>;

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(wrapped = "crate::LossyFrom::lossy_from")]
pub struct ToricLensParametrization<T> {
    pub signed_normalized_poloidal_curvature: T,
    pub normalized_toroidal_curvature: T,
    pub height: T,
    pub width: T,
}

impl<T: FloatExt, V: Vector<3, Scalar = T>> FromParametrizedHyperPlane<V, 3> for ToricLens<V, 3> {
    type Parametrization = ToricLensParametrization<T>;

    fn from_hyperplane(hyperplane: HyperPlane<V>, parametrization: Self::Parametrization) -> Self {
        let ToricLensParametrization {
            signed_normalized_poloidal_curvature,
            normalized_toroidal_curvature,
            height,
            width,
        } = parametrization;
        let chord_length = hyperplane.normal.sec_xy(height).abs();
        // let signed_curvature = signed_normalized_curvature / (chord_length * T::lossy_from(0.5f64));
        let poloidal_radius =
            chord_length * T::lossy_from(0.5f64) / signed_normalized_poloidal_curvature.abs();
        let toroidal_radius = width * T::lossy_from(0.5f64) / normalized_toroidal_curvature;

        let apothem = (poloidal_radius.sqr() - chord_length.sqr() * T::lossy_from(0.25f64)).sqrt();
        let sagitta = poloidal_radius - apothem;
        let center = hyperplane.normal.mul_add(
            (apothem + toroidal_radius).copy_sign(signed_normalized_poloidal_curvature),
            hyperplane.point,
        );
        Self {
            surface: ToricSurface {
                center,
                toroidal_normal: UnitVector::new(hyperplane.normal.rot_90_xy()),
                toroidal_radius,
                poloidal_radius,
            },
            bounds: SandwichBounds {
                center: hyperplane.point,
                normal: if signed_normalized_poloidal_curvature.is_sign_positive() {
                    -hyperplane.normal
                } else {
                    hyperplane.normal
                },
                height: sagitta,
            },
            marker: core::marker::PhantomData,
        }
    }
}

impl<T: FloatExt, V: Vector<3, Scalar = T>> Drawable<T> for ToricLens<V, 3> {
    fn draw(&self) -> Path<T> {
        let half_height = self.bounds.center.y();
        let dx = self.bounds.normal.tan_xy() * half_height;
        let ux = self.bounds.center.x() - dx;
        let lx = self.bounds.center.x() + dx;
        // let quasi_radius = (lx - self.surface.center.x()).hypot(self.surface.center.y());
        let quasi_radius = self.surface.poloidal_radius;
        let midpt = self
            .bounds
            .normal
            .mul_add(self.bounds.height, self.bounds.center);
        Path::Arc {
            a: Point {
                x: ux,
                y: half_height * T::lossy_from(2u32),
            },
            b: Point { x: lx, y: T::ZERO },
            midpt: Point {
                x: midpt.x(),
                y: midpt.y(),
            },
            // center: Point {
            //     x: self.surface.center.x(),
            //     y: self.surface.center.y(),
            // },
            radius: quasi_radius,
        }
    }
}
