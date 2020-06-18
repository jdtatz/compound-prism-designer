use crate::utils::Float;
/*
#![feature(const_fn, const_loop, const_if_match, const_generics)]

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

struct Qrng<const N: usize> {
    state: [f64; N],
    alpha: [f64; N],
}

impl<const N: usize> Qrng<N> {
    pub const fn new(seed: [f64; N]) -> Self {
        let g = phi(N);
        let mut alpha = seed;
        let mut i = 0;
        while i < N {
            alpha[i] = 1_f64 / powi(g, i + 1);
            i += 1;
        }
        Self {
            state: seed,
            alpha,
        }
    }

    pub fn next(&mut self) -> [f64; N] {
        let mut i = 0;
        while i < N {
            self.state[i] = (self.state[i] + self.alpha[i]) % 1_f64;
            i += 1;
        }
        self.state
    }

    pub fn step(&mut self, step: u32) -> [f64; N] {
        let step = step as f64;
        let mut i = 0;
        while i < N {
            self.state[i] = (self.state[i] + step * self.alpha[i]) % 1_f64;
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
    fn alpha() -> Self;
    fn iadd_mod_1(&mut self, rhs: Self);
    fn mul_by_int(self, rhs: u32) -> Self;
}

impl<F: Float> QuasiRandom for F {
    fn alpha() -> Self {
        F::from_f64(ALPHA_1)
    }

    fn iadd_mod_1(&mut self, rhs: Self) {
        *self = (*self + rhs).fract();
    }

    fn mul_by_int(self, rhs: u32) -> Self {
        self * F::from_u32(rhs)
    }
}

impl<F: Float> QuasiRandom for [F; 2] {
    fn alpha() -> Self {
        [F::from_f64(ALPHA_2[0]), F::from_f64(ALPHA_2[1])]
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
    fn alpha() -> Self {
        [
            F::from_f64(ALPHA_3[0]),
            F::from_f64(ALPHA_3[1]),
            F::from_f64(ALPHA_3[2]),
        ]
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

#[cfg(not(target_arch = "nvptx64"))]
fn phi(dim: usize) -> f64 {
    let mut x = 2_f64;
    let pow = ((dim + 1) as f64).recip();
    for _ in 0..30 {
        x = (x + 1_f64).powf(pow);
    }
    x
}

#[cfg(not(target_arch = "nvptx64"))]
pub struct DynamicQrng {
    state: Vec<f64>,
    alphas: Vec<f64>,
}

#[cfg(not(target_arch = "nvptx64"))]
impl DynamicQrng {
    pub fn new(seed: Vec<f64>) -> Self {
        let dim = seed.len();
        let root = phi(dim);
        let alphas = (1..=dim).map(|i| root.powi(-(i as i32))).collect();
        DynamicQrng {
            state: seed,
            alphas,
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> &[f64] {
        for (s, alpha) in self.state.iter_mut().zip(self.alphas.iter().copied()) {
            *s = (*s + alpha).fract();
        }
        self.state.as_slice()
    }
}
