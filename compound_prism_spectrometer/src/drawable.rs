#![allow(clippy::just_underscores_and_digits)]
use crate::{FloatExt, Vector};

#[derive(Debug, Clone, Copy)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
}

impl<T> From<Vector<T, 2>> for Point<T> {
    fn from(v: Vector<T, 2>) -> Self {
        let Vector([x, y]) = v;
        Self { x, y }
    }
}

impl<T> From<Point<T>> for Vector<T, 2> {
    fn from(p: Point<T>) -> Self {
        let Point { x, y } = p;
        Vector([x, y])
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Path<T> {
    Line {
        a: Point<T>,
        b: Point<T>,
    },
    Arc {
        a: Point<T>,
        b: Point<T>,
        midpt: Point<T>,
        // center: Point<T>,
        radius: T,
    },
}

impl<T: Copy> Path<T> {
    pub fn reverse(self) -> Self {
        match self {
            Path::Line { a, b } => Path::Line { a: b, b: a },
            Path::Arc {
                a,
                b,
                midpt,
                // center,
                radius,
            } => Path::Arc {
                a: b,
                b: a,
                midpt,
                // center,
                radius,
            },
        }
    }

    pub fn start(&self) -> Point<T> {
        match self {
            Path::Line { a, .. } => *a,
            Path::Arc { a, .. } => *a,
        }
    }

    pub fn end(&self) -> Point<T> {
        match self {
            Path::Line { b, .. } => *b,
            Path::Arc { b, .. } => *b,
        }
    }

    // pub fn as_cubic_bézier(&self) -> [Point<T>; 4] where T: FloatExt {
    //     match *self {
    //         Path::Line { a, b } => {
    //             let delta = Vector::from(b) - Vector::from(a);
    //             let chord_len = delta.norm();
    //             let delta = delta / chord_len;
    //             let c0 = delta.mul_add(chord_len * T::lossy_from(0.25_f64), Vector::from(a));
    //             let c1 = delta.mul_add(chord_len * T::lossy_from(0.75_f64), Vector::from(a));
    //             [a, Point::from(c0), Point::from(c1), b]
    //         }
    //         Path::Arc { a, b, midpt, center, radius } => {
    //             let k = T::ONE / radius;
    //             let delta = Vector::from(b) - Vector::from(a);
    //             let chord_len = delta.norm();
    //             let chord_midpt = delta.mul_add(T::lossy_from(0.5_f64), Vector::from(a));
    //             let sagita_v = Vector::from(midpt) - chord_midpt;
    //             let sagita = sagita_v.norm();
    //             let sagita_v = sagita_v / sagita;
    //             let exsec_len = k * chord_len.sqr() / (T::lossy_from(2u32) * T::sqrt(T::lossy_from(4u32) - (k * chord_len).sqr())) - sagita;
    //             let tan_pt = sagita_v.mul_add(exsec_len, midpt.into());
    //             let tan_a = (tan_pt - Vector::from(a)).normalize();
    //             let tan_b = (tan_pt - Vector::from(b)).normalize();
    //             let alpha = k * chord_len * T::sqrt(T::lossy_from(4u32) - (k * chord_len).sqr()) * ( T::sqrt(T::ONE + T::lossy_from(3u32) / (T::lossy_from(2u32) + k * chord_len) - T::lossy_from(3u32) / (-T::lossy_from(2u32) + k * chord_len)) - T::ONE ) / T::lossy_from(6u32);
    //             let c0 = tan_a.mul_add(alpha, a.into());
    //             let c1 = tan_b.mul_add(alpha, b.into());
    //             [a, Point::from(c0), Point::from(c1), b]
    //         }
    //     }
    // }
}

pub fn arc_as_cubic_bézier<T: FloatExt>(
    a: Point<T>,
    midpt: Point<T>,
    b: Point<T>,
    curvature: T,
) -> [Point<T>; 4] {
    let frac_1_2 = T::lossy_from(0.5_f64);
    let _2 = T::lossy_from(2_u32);
    let _3 = T::lossy_from(3_u32);
    let _4 = T::lossy_from(4_u32);
    let _6 = T::lossy_from(6_u32);
    let k = curvature;
    let delta = Vector::from(b) - Vector::from(a);
    let chord_len = delta.norm();
    let chord_midpt = delta.mul_add(frac_1_2, Vector::from(a));
    let sagita_v = Vector::from(midpt) - chord_midpt;
    let sagita = sagita_v.norm();
    let sagita_v = sagita_v / sagita;
    let exsec_len = k * chord_len.sqr() / (_2 * T::sqrt(_4 - (k * chord_len).sqr())) - sagita;
    let tan_pt = sagita_v.mul_add(exsec_len, midpt.into());
    let tan_a = (tan_pt - Vector::from(a)).normalize();
    let tan_b = (tan_pt - Vector::from(b)).normalize();
    // let alpha = k * chord_len * T::sqrt(_4 - (k * chord_len).sqr()) * ( T::sqrt(T::ONE + _3 / (_2 + k * chord_len) - _3 / (-_2 + k * chord_len)) - T::ONE ) / _6;
    let alpha = chord_len
        * T::sqrt(_4 - (k * chord_len).sqr())
        * (T::sqrt(T::ONE + _3 / (_2 + k * chord_len) - _3 / (-_2 + k * chord_len)) - T::ONE)
        / _6;
    let c0 = tan_a.mul_add(alpha, a.into());
    let c1 = tan_b.mul_add(alpha, b.into());
    [a, Point::from(c0), Point::from(c1), b]
}

pub fn arc_as_2_cubic_béziers<T: FloatExt>(
    a: Point<T>,
    midpt: Point<T>,
    b: Point<T>,
    curvature: T,
) -> [[Point<T>; 4]; 2] {
    let frac_1_2 = T::lossy_from(0.5_f64);
    let _2 = T::lossy_from(2_u32);
    let _3 = T::lossy_from(3_u32);
    let _4 = T::lossy_from(4_u32);
    let _6 = T::lossy_from(6_u32);
    let _12 = T::lossy_from(12_u32);
    let k = curvature;
    let delta = Vector::from(b) - Vector::from(a);
    let chord_len = delta.norm();
    let chord_midpt = delta.mul_add(frac_1_2, Vector::from(a));
    let sagita_v = Vector::from(midpt) - chord_midpt;
    let sagita = sagita_v.norm();
    let sagita_v = sagita_v / sagita;
    let exsec_len = k * chord_len.sqr() / (_2 * T::sqrt(_4 - (k * chord_len).sqr())) - sagita;
    let tan_pt = sagita_v.mul_add(exsec_len, midpt.into());
    let tan_a = (tan_pt - Vector::from(a)).normalize();
    let tan_b = (tan_pt - Vector::from(b)).normalize();
    // let alpha = k * chord_len * T::sqrt(_4 - (k * chord_len).sqr()) * ( T::sqrt(T::ONE + _3 / (_2 + k * chord_len) - _3 / (-_2 + k * chord_len)) - T::ONE ) / _6;
    // let alpha = chord_len * T::sqrt(_4 - (k * chord_len).sqr()) * ( T::sqrt(T::ONE + _3 / (_2 + k * chord_len) - _3 / (-_2 + k * chord_len)) - T::ONE ) / _6;
    let alpha = chord_len
        * (T::sqrt(T::ONE + _12 / (_2 + T::sqrt(_4 - (k * chord_len).sqr()))) - T::ONE)
        / _6;
    let tan_mid = delta / chord_len;
    let c0 = tan_a.mul_add(alpha, a.into());
    let c1 = tan_mid.mul_add(-alpha, midpt.into());
    let c2 = tan_mid.mul_add(alpha, midpt.into());
    let c3 = tan_b.mul_add(alpha, b.into());
    [
        [a, Point::from(c0), Point::from(c1), midpt],
        [midpt, Point::from(c2), Point::from(c3), b],
    ]
}

#[derive(Debug, Clone, Copy)]
pub struct Polygon<T>(pub [Path<T>; 2]);

pub trait Drawable<T> {
    fn draw(&self) -> Path<T>;
}
