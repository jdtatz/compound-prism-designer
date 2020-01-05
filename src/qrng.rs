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


pub struct Qrng<T: Sized> {
    state: T,
    alpha: T,
}

const PHI_1: f64 = 1.61803398874989484820458683436563;
const PHI_2: f64 = 1.32471795724474602596090885447809;

impl Qrng<f64> {
    pub fn new(seed: f64) -> Self {
        Self {
            state: seed,
            alpha: 1_f64 / PHI_1
        }
    }

    pub fn next(&mut self) -> f64 {
        self.state = (self.state + self.alpha).fract();
        self.state
    }
}

impl Qrng<[f64; 2]> {
    pub fn new(seed: [f64; 2]) -> Self {
        Self {
            state: seed,
            alpha: [1_f64 / PHI_2, 1_f64 / (PHI_2 * PHI_2)]
        }
    }

    pub fn next(&mut self) -> [f64; 2] {
        self.state = [(self.state[0] + self.alpha[0]).fract(), (self.state[1] + self.alpha[1]).fract()];
        self.state
    }
}

fn phi(dim: usize) -> f64 {
    let mut x = 2_f64;
    let pow = ((dim + 1) as f64).recip();
    for _ in 0..30 {
        x = (x + 1_f64).powf(pow);
    }
    x
}

pub struct DynamicQrng {
    state: f64,
    alphas: Vec<f64>,
}

impl DynamicQrng {
    pub fn new(seed: f64, dim: usize) -> Self {
        let root = phi(dim);
        let alphas = (1..=dim).map(|i| root.powi(-(i as i32))).collect();
        DynamicQrng {
            state: seed,
            alphas,
        }
    }

    pub fn next(&mut self) -> impl ExactSizeIterator<Item = f64> + '_ {
        let state = self.state;
        self.state += 1_f64;
        self.alphas
            .iter()
            .copied()
            .map(move |c| (state * c).fract())
    }
}
