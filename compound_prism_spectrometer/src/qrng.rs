use crate::utils::Float;
/*
#![feature(const_fn_floating_point_arithmetic, min_const_generics)]

const fn powi(mut x: f64, mut n: usize) -> f64 {
    if n == 0 {
        return 1_f64;
    }
    let mut y = 1_f64;
    while n > 1 {
        if n % 2 == 0 {
            x *= x;
            n /= 2;
        } else {
            y *= x;
            x *= x;
            n = (n - 1) / 2;
        }
    }
    x * y
}

const fn phi(dim: usize) -> f64 {
    let mut lower = 1_f64;
    let mut upper = 2_f64;
    loop {
        let mid = (upper + lower) / 2_f64;
        if mid <= lower || mid >= upper {
            return mid;
        }
        let f_mid = powi(mid, 1 + dim) - mid - 1_f64;
        if f_mid < 0_f64 {
            lower = mid;
        } else if f_mid > 0_f64 {
            upper = mid;
        } else {
            return mid;
        }
    }
}

const fn create_alpha<const N: usize>() -> [f64; N] {
    let g = phi(N);
    let mut alpha = [0f64; N];
    let mut i = 0;
    while i < N {
        alpha[i] = 1_f64 / powi(g, i + 1);
        i += 1;
    }
    alpha
}


const fn array_cast_f64_f32<const N: usize>(arr: [f64; N]) -> [f32; N] {
    let mut out = [0f32; N];
    let mut i = 0;
    while i < N {
        out[i] = arr[i] as f32;
        i += 1;
    }
    out
}

pub struct Qrng32<const N: usize> {
    state: [f32; N],
}

impl<const N: usize> Qrng32<N> {
    const ALPHA: [f32; N] = array_cast_f64_f32(create_alpha::<N>());

    pub const fn new(seed: [f32; N]) -> Self {
        Self {
            state: seed,
        }
    }

    pub const fn from_scalar(seed: f32) -> Self {
        Self {
            state: [seed; N],
        }
    }

    pub fn next(&mut self) -> [f32; N] {
        let mut i = 0;
        while i < N {
            self.state[i] = (self.state[i] + Self::ALPHA[i]) % 1_f32;
            i += 1;
        }
        self.state
    }

    pub fn step(&mut self, step: u32) -> [f32; N] {
        let step = step as f32;
        let mut i = 0;
        while i < N {
            self.state[i] = (self.state[i] + step * Self::ALPHA[i]) % 1_f32;
            i += 1;
        }
        self.state
    }
}

pub struct Qrng64<const N: usize> {
    state: [f64; N],
}

impl<const N: usize> Qrng64<N> {
    const ALPHA: [f64; N] = create_alpha();

    pub const fn new(seed: [f64; N]) -> Self {
        Self {
            state: seed,
        }
    }

    pub const fn from_scalar(seed: f64) -> Self {
        Self {
            state: [seed; N],
        }
    }

    pub fn next(&mut self) -> [f64; N] {
        let mut i = 0;
        while i < N {
            self.state[i] = (self.state[i] + Self::ALPHA[i]) % 1_f64;
            i += 1;
        }
        self.state
    }

    pub fn step(&mut self, step: u32) -> [f64; N] {
        let step = step as f64;
        let mut i = 0;
        while i < N {
            self.state[i] = (self.state[i] + step * Self::ALPHA[i]) % 1_f64;
            i += 1;
        }
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_almost_eq;

    #[test]
    fn test_phi() {
        assert_almost_eq!(phi(1), 1.6180339887498948482045868343656381);
        assert_almost_eq!(phi(2), 1.3247179572447460259609088544780973, 1e-15);
        assert_almost_eq!(phi(16), 1.042917732301786579722142992093799834021, 1e-15);
    }
}
*/

#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
const PHI_1: f64 = 1.61803398874989484820458683436563;
const ALPHA_1: f64 = 1_f64 / PHI_1;
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
const PHI_2: f64 = 1.3247179572447460259609088544780973;
const ALPHA_2: [f64; 2] = [1_f64 / PHI_2, 1_f64 / (PHI_2 * PHI_2)];
#[allow(clippy::unreadable_literal, clippy::excessive_precision)]
const PHI_3: f64 = 1.2207440846057594753616853491088319;
const ALPHA_3: [f64; 3] = [
    1_f64 / PHI_3,
    1_f64 / (PHI_3 * PHI_3),
    1_f64 / (PHI_3 * PHI_3 * PHI_3),
];

pub trait QuasiRandom: Copy {
    type Scalar;
    fn alpha() -> Self;
    fn from_scalar(scalar: Self::Scalar) -> Self;
    fn iadd_mod_1(&mut self, rhs: Self);
    fn mul_by_int(self, rhs: u32) -> Self;
}

impl<F: Float> QuasiRandom for F {
    type Scalar = F;

    fn alpha() -> Self {
        F::from_f64(ALPHA_1)
    }

    fn from_scalar(scalar: Self::Scalar) -> Self {
        scalar
    }

    fn iadd_mod_1(&mut self, rhs: Self) {
        *self = (*self + rhs).fract();
    }

    fn mul_by_int(self, rhs: u32) -> Self {
        self * F::from_u32(rhs)
    }
}

impl<F: Float> QuasiRandom for [F; 2] {
    type Scalar = F;

    fn alpha() -> Self {
        [F::from_f64(ALPHA_2[0]), F::from_f64(ALPHA_2[1])]
    }

    fn from_scalar(scalar: Self::Scalar) -> Self {
        [scalar, scalar]
    }

    fn iadd_mod_1(&mut self, rhs: Self) {
        self[0] = (self[0] + rhs[0]).fract();
        self[1] = (self[1] + rhs[1]).fract();
    }

    fn mul_by_int(self, rhs: u32) -> Self {
        let rhs = F::from_u32(rhs);
        [self[0] * rhs, self[1] * rhs]
    }
}

impl<F: Float> QuasiRandom for [F; 3] {
    type Scalar = F;

    fn alpha() -> Self {
        [
            F::from_f64(ALPHA_3[0]),
            F::from_f64(ALPHA_3[1]),
            F::from_f64(ALPHA_3[2]),
        ]
    }

    fn from_scalar(scalar: Self::Scalar) -> Self {
        [scalar, scalar, scalar]
    }

    fn iadd_mod_1(&mut self, rhs: Self) {
        self[0] = (self[0] + rhs[0]).fract();
        self[1] = (self[1] + rhs[1]).fract();
        self[2] = (self[2] + rhs[2]).fract();
    }

    fn mul_by_int(self, rhs: u32) -> Self {
        let rhs = F::from_u32(rhs);
        [self[0] * rhs, self[1] * rhs, self[2] * rhs]
    }
}

pub struct Qrng<Q: QuasiRandom> {
    state: Q,
}

impl<Q: QuasiRandom> Qrng<Q> {
    pub fn new(seed: Q) -> Self {
        Self { state: seed }
    }

    pub fn new_from_scalar(scalar_seed: Q::Scalar) -> Self {
        Self {
            state: Q::from_scalar(scalar_seed),
        }
    }

    pub fn next_by(&mut self, step: u32) -> Q {
        self.state.iadd_mod_1(Q::alpha().mul_by_int(step));
        self.state
    }
}

impl<Q: QuasiRandom> Iterator for Qrng<Q> {
    type Item = Q;

    fn next(&mut self) -> Option<Self::Item> {
        self.state.iadd_mod_1(Q::alpha());
        Some(self.state)
    }
}