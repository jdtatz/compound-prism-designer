pub use quasirandom::Qrng;

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
    root: f64,
    alphas: Vec<f64>
}

impl DynamicQrng {
    pub fn new(seed: f64, dim: usize) -> Self {
        let root = phi(dim);
        let alphas = (1..=dim).map(|i| {
            root.powi(-(i as i32))
        }).collect();
        DynamicQrng {
            state: seed,
            root,
            alphas
        }
    }

    pub fn next(&mut self) -> impl ExactSizeIterator<Item=f64> + '_ {
        let state = self.state;
        self.state += 1_f64;
        self.alphas.iter().copied().map(move |c|
            (state * c).fract()
        )
    }
}
