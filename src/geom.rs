use crate::{debug_assert_almost_eq, utils::{LossyInto, Float, almost_eq}};

/// vector in R^2 represented as a 2-tuple
#[repr(C)]
#[derive(Debug, PartialEq, Clone, Copy, Neg, Add, Sub, Mul, Div, Serialize, Deserialize)]
pub struct Pair<F: Float> {
    pub x: F,
    pub y: F,
}

impl<F: Float> Pair<F> {
    /// unit vector at angle `theta` relative to the x axis.
    pub fn angled(theta: F) -> Self {
        let (sin, cos) = theta.sincos();
        Self { x: cos, y: sin }
    }

    /// dot product of two vectors, a • b
    pub fn dot(self, other: Self) -> F {
        self.x * other.x + self.y * other.y
    }

    /// square of the vector norm, ||v||^2
    pub fn norm_squared(self) -> F {
        self.dot(self)
    }

    /// vector norm, ||v||
    pub fn norm(self) -> F {
        self.norm_squared().sqrt()
    }

    /// is it a unit vector, ||v|| ≅? 1
    pub fn is_unit(self) -> bool {
        almost_eq(self.norm().to_f64(), 1_f64, 1e-3)
    }

    /// Fused multiply add of two vectors with a scalar, (self * a) + b
    pub fn mul_add(self, a: F, b: Self) -> Self {
        Pair {
            x: self.x.mul_add(a, b.x),
            y: self.y.mul_add(a, b.y),
        }
    }
}

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<Pair<F2>> for Pair<F1> {
    fn into(self) -> Pair<F2> {
        Pair {
            x: self.x.into(),
            y: self.y.into()
        }
    }
}

/// rotate `vector` by `angle` CCW
#[inline(always)]
pub fn rotate<F: Float>(angle: F, vector: Pair<F>) -> Pair<F> {
    let (s, c) = angle.sincos();
    Pair {
        x: c * vector.x - s * vector.y,
        y: s * vector.x + c * vector.y,
    }
}

/// Matrix in R^(2x2) in row major order
#[derive(Debug, Clone, Copy)]
pub struct Mat2<F: Float>([F; 4]);

impl<F: Float> Mat2<F> {
    /// New Matrix from the two given columns
    pub fn new_from_cols(col1: Pair<F>, col2: Pair<F>) -> Self {
        Self([col1.x, col2.x, col1.y, col2.y])
    }

    /// Matrix inverse if it exists
    pub fn inverse(self) -> Option<Self> {
        let [a, b, c, d] = self.0;
        let det = a * d - b * c;
        if det == F::zero() {
            None
        } else {
            Some(Self([d / det, -b / det, -c / det, a / det]))
        }
    }
}

impl<F: Float> core::ops::Mul<Pair<F>> for Mat2<F> {
    type Output = Pair<F>;

    /// Matrix x Vector -> Vector multiplication
    fn mul(self, rhs: Pair<F>) -> Self::Output {
        let [a, b, c, d] = self.0;
        Pair {
            x: a * rhs.x + b * rhs.y,
            y: c * rhs.x + d * rhs.y,
        }
    }
}

pub trait Surface {
    type Point;
    type UnitVector;
    fn intersection(&self, ray: (Self::Point, Self::UnitVector)) -> Option<(Self::Point, Self::UnitVector)>;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Plane<F: Float> {
    pub(crate) normal: Pair<F>,
    pub(crate) midpt: Pair<F>,
    height: F,
}

impl<F: Float> Plane<F> {
    #[cfg(not(target_arch = "nvptx64"))]
    pub(crate) fn end_points(&self, height: F) -> (Pair<F>, Pair<F>) {
        let dx = self.normal.y / self.normal.x * height * F::from_f64(0.5);
        let ux = self.midpt.x - dx;
        let lx = self.midpt.x + dx;
        (
            Pair { x: ux, y: height },
            Pair {
                x: lx,
                y: F::zero(),
            },
        )
    }
}

impl<F: Float> Surface for Plane<F> {
    type Point = Pair<F>;
    type UnitVector = Pair<F>;

    fn intersection(&self, ray: (Self::Point, Self::UnitVector)) -> Option<(Self::Point, Self::UnitVector)> {
        let (origin, direction) = ray;
        debug_assert!(direction.is_unit());
        let ci = -direction.dot(self.normal);
        if ci <= F::zero() {
            return None;
        }
        let d = (origin - self.midpt).dot(self.normal) / ci;
        let p = direction.mul_add(d, origin);
        if p.y <= F::zero() || self.height <= p.y {
            None
        } else {
            Some((p, self.normal))
        }
    }
}

pub(crate) fn create_joined_trapezoids<'s, F: Float>(height: F, angles: &'s [F], sep_lengths: &'s [F])
    -> (Plane<F>, impl 's + ExactSizeIterator<Item=Plane<F>>) {
    debug_assert!(angles.len() >= 2);
    debug_assert!(angles.len() == sep_lengths.len() + 1);
    let h2 = height * F::from_f64(0.5);
    let normal = -Pair::angled(angles[0]);
    debug_assert_eq!(normal, rotate(angles[0], Pair { x: -F::one(), y: F::zero() }));
    debug_assert_almost_eq!(
            (normal.y / normal.x).abs().to_f64(),
            angles[0].tan().abs().to_f64(),
            1e-10
        );
    let first = Plane {
        height,
        normal,
        midpt: Pair {
            x: (normal.y / normal.x).abs() * h2,
            y: h2,
        },
    };
    let mut prev_sign = normal.y.is_sign_positive();
    let mut d1 = first.midpt.x;
    let mut mx = first.midpt.x;
    let rest = angles[1..].iter().copied().zip(sep_lengths.iter().copied())
        .map(move |(angle, sep_len)| {
            let normal = -Pair::angled(angle);
            let sign = normal.y.is_sign_positive();
            let d2 = (normal.y / normal.x).abs() * h2;
            let sep_dist = sep_len + if prev_sign != sign {
                d1 + d2
            } else {
                (d1 - d2).abs()
            };
            prev_sign = sign;
            d1 = d2;
            mx += sep_dist;
            Plane {
                height,
                normal,
                midpt: Pair {
                    x: mx,
                    y: h2,
                },
            }
        });
    (first, rest)
}

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<Plane<F2>> for Plane<F1> {
    fn into(self) -> Plane<F2> {
        Plane {
            height: self.height.into(),
            normal: LossyInto::into(self.normal),
            midpt: LossyInto::into(self.midpt),
        }
    }
}



#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct CurvedPlane<F: Float> {
    /// The midpt of the Curved Surface / circular segment
    pub(crate) midpt: Pair<F>,
    /// The center of the circle
    pub(crate) center: Pair<F>,
    /// The radius of the circle
    pub(crate) radius: F,
    /// max_dist_sq = sagitta ^ 2 + (chord_length / 2) ^ 2
    max_dist_sq: F,
}

impl<F: Float> CurvedPlane<F> {
    pub(crate) fn new(curvature: F, height: F, chord: Plane<F>) -> Self {
        debug_assert!(F::one() >= curvature && curvature > F::zero());
        debug_assert!(height > F::zero());
        let chord_length = height / chord.normal.x.abs();
        let radius = chord_length * F::from_f64(0.5) / curvature;
        let apothem = (radius * radius - chord_length * chord_length * F::from_f64(0.25)).sqrt();
        let sagitta = radius - apothem;
        let center = chord.midpt + chord.normal * apothem;
        let midpt = chord.midpt - chord.normal * sagitta;
        Self {
            midpt,
            center,
            radius,
            max_dist_sq: sagitta * sagitta + chord_length * chord_length * F::from_f64(0.25),
        }
    }

    fn is_along_arc(&self, pt: Pair<F>) -> bool {
        debug_assert_almost_eq!((pt - self.center).norm().to_f64(), self.radius.to_f64(), 1e-6);
        (pt - self.midpt).norm_squared() <= self.max_dist_sq
    }

    #[cfg(not(target_arch = "nvptx64"))]
    pub(crate) fn end_points(&self, height: F) -> (Pair<F>, Pair<F>) {
        let theta_2 =
            F::from_f64(2.) * (self.max_dist_sq.sqrt() / (F::from_f64(2.) * self.radius)).asin();
        let r = self.midpt - self.center;
        let u = self.center + rotate(theta_2, r);
        let l = self.center + rotate(-theta_2, r);
        debug_assert!(
            (u.y - height).abs() < F::from_f64(1e-4),
            "{:?} {}",
            u,
            height
        );
        debug_assert!(l.y.abs() < F::from_f64(1e-4), "{:?}", l);
        debug_assert!(
            ((u - r).norm_squared() - self.max_dist_sq).abs() / self.max_dist_sq
                < F::from_f64(1e-4)
        );
        debug_assert!(
            ((l - r).norm_squared() - self.max_dist_sq).abs() / self.max_dist_sq
                < F::from_f64(1e-4)
        );
        (
            Pair { x: u.x, y: height },
            Pair {
                x: l.x,
                y: F::zero(),
            },
        )
    }
}

impl<F: Float> Surface for CurvedPlane<F> {
    type Point = Pair<F>;
    type UnitVector = Pair<F>;

    fn intersection(&self, ray: (Self::Point, Self::UnitVector)) -> Option<(Self::Point, Self::UnitVector)> {
        let (origin, direction) = ray;
        debug_assert!(direction.is_unit());
        let delta = origin - self.center;
        let ud = direction.dot(delta);
        let discriminant = ud * ud - delta.norm_squared() + self.radius * self.radius;
        if discriminant <= F::zero() {
            return None;
        }
        let dl = -ud - discriminant.sqrt();
        let dr = -ud + discriminant.sqrt();
        if dl <= F::zero() && dr <= F::zero(){
            return None;
        }
        let pl = direction.mul_add(dl, origin);
        let pr = direction.mul_add(dr, origin);
        if self.is_along_arc(pl) {
            debug_assert!(!self.is_along_arc(pr));
            let snorm = (self.center - pl) / self.radius;
            debug_assert!(snorm.is_unit());
            Some((pl, snorm))
        } else if self.is_along_arc(pr) {
            let snorm = (self.center - pr) / self.radius;
            debug_assert!(snorm.is_unit());
            Some((pr, snorm))
        } else {
           None
        }
    }
}

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<CurvedPlane<F2>> for CurvedPlane<F1> {
    fn into(self) -> CurvedPlane<F2> {
        CurvedPlane {
            midpt: LossyInto::into(self.midpt),
            center: LossyInto::into(self.center),
            radius: self.radius.into(),
            max_dist_sq: self.max_dist_sq.into(),
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::almost_eq;
    use rand::prelude::*;
    use std::f64::consts::*;

    #[test]
    fn test_many() {

    }
}
