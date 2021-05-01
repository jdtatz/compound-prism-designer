use crate::{geometry::SandwichBounds, *};

fn sqr<T: Copy + core::ops::Mul<T, Output = T>>(v: T) -> T {
    v * v
}

/// The implicit equation of a torus $f(\vec{p}; R, r, \vec{c}, \hat{n}) = \left(\norm{\vec{p} - \vec{c}}^2 + R^2 - r^2 \right)^2 - 4 R^2 \left(\norm{\vec{p} - \vec{c}}^2 - \left((\vec{p} - \vec{c})\cdot \hat{n}\right)^2 \right)$
/// $\| \oproj_{\hat{n}}(\vec{p} - \vec{c}) \|^2$
/// poloidal direction is cylindrical
/// toroidal direction is spherical
#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct ToricSurface<T> {
    center: Vector<T, 3>,
    toroidal_normal: UnitVector<T, 3>,
    toroidal_radius: T,
    poloidal_radius: T,
}

// impl<T: Copy + Ring + LossyFrom<u32>> ToricSurface<T> {
//     pub fn implicit(self, p: Vector<T, 3>) -> T {
//         let u = p - self.center;
//         sqr(u.norm_squared() + self.toroidal_radius.sqr() - self.poloidal_radius.sqr())
//             - T::lossy_from(4u32)
//                 * self.toroidal_radius.sqr()
//                 * crate::vector::cross_prod_magnitude_sq(u, self.toroidal_normal)
//     }
// }

impl<T: FloatExt, B: Copy + Bounds<T, 3>> HyperSurface<T, B, 3> for ToricSurface<T> {
    // $Assumptions =
    //  t \[Element] Reals && (p | c | n | v | u | dp) \[Element]
    //    Vectors[3, Reals] && n . n == 1 && v . v == 1
    // (u . u + \[ScriptCapitalR]^2 - r^2)^2 -
    //  4 \[ScriptCapitalR]^2 (u . u - (u . n)^2)
    // % /. u -> dp + v t
    // FunctionExpand[%] // TensorExpand
    // CoefficientList[%, t] // FullSimplify
    // MatrixForm[%]

    fn intersection(
        self,
        ray: GeometricRay<T, 3>,
        bounds: B,
    ) -> Option<GeometricRayIntersection<T, 3>> {
        let GeometricRay { origin, direction } = ray;
        let dp = origin - self.center;
        let dp_magn_sq = dp.norm_squared();
        let dp_v = dp.dot(*direction);
        let dp_n = dp.dot(*self.toroidal_normal);
        let n_v = self.toroidal_normal.dot(*direction);
        let R_2 = self.toroidal_radius.sqr();
        let r_2 = self.poloidal_radius.sqr();
        let _2 = T::lossy_from(2u32);
        let _4 = T::lossy_from(4u32);
        let _8 = T::lossy_from(8u32);
        // quartic surface: a x^4 + b x^3 + c x^2 + d x + e == 0
        let a = T::ONE;
        let b = _4 * dp_v;
        let c = -_2 * (r_2 + R_2 - dp_magn_sq - _2 * dp_v.sqr() - _2 * R_2 * n_v.sqr());
        let d = -_4 * (r_2 + R_2 - dp_magn_sq) * dp_v + _8 * R_2 * dp_n * n_v;
        let e = sqr(r_2 - R_2) - _2 * (R_2 + r_2) * dp_magn_sq
            + dp_magn_sq.sqr()
            + _4 * R_2 * dp_n.sqr();
        let roots = crate::roots::find_roots_quartic(a, b, c, d, e);
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
        let (distance, p) = match roots {
            roots::Roots::No(_) | roots::Roots::One([_]) => None,
            roots::Roots::Two([sec1, sec2]) => check_root(sec1).or_else(|| check_root(sec2)),
            roots::Roots::Three([sec1, _tan, sec2]) => {
                check_root(sec1).or_else(|| check_root(sec2))
            }
            roots::Roots::Four([sec1, _sec2, _sec3, sec4]) => {
                check_root(sec1).or_else(|| check_root(sec4))
            }
        }?;
        let t_center = crate::vector::oproj(p - self.center, self.toroidal_normal)
            .normalize()
            .mul_add(self.toroidal_radius, self.center);
        let normal = UnitVector::new((t_center - p) / self.poloidal_radius);
        Some(GeometricRayIntersection { distance, normal })
    }
}

pub type ToricLens<T> = BoundedHyperSurface<T, ToricSurface<T>, SandwichBounds<T, 3>, 3>;

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct ToricLensParametrization<T> {
    pub signed_normalized_poloidal_curvature: T,
    pub normalized_toroidal_curvature: T,
    pub height: T,
    pub width: T,
}

impl<T: FloatExt> FromParametrizedHyperPlane<T, 3> for ToricLens<T> {
    type Parametrization = ToricLensParametrization<T>;

    fn from_hyperplane(
        hyperplane: HyperPlane<T, 3>,
        parametrization: Self::Parametrization,
    ) -> Self {
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
                toroidal_normal: UnitVector::new(hyperplane.normal.0.rot_90_xy()),
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
