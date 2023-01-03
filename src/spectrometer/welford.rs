use crate::FloatExt;

/// https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
#[derive(Copy, Clone)]
pub struct Welford<F> {
    pub count: F,
    pub mean: F,
    pub m2: F,
}

impl<F: FloatExt> Welford<F> {
    pub const NEW: Self = Welford {
        count: F::ZERO,
        mean: F::ZERO,
        m2: F::ZERO,
    };
    pub fn new() -> Self {
        Welford {
            count: F::zero(),
            mean: F::zero(),
            m2: F::zero(),
        }
    }
    pub fn next_sample(&mut self, x: F) {
        self.count += F::one();
        let delta = x - self.mean;
        self.mean += delta / self.count;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
    }
    pub fn skip(&mut self, new_count: F) {
        if self.count < new_count {
            let zero_count = new_count - self.count;
            let count = new_count;
            self.mean = (self.count * self.mean) / count;
            self.m2 += self.mean.sqr() * self.count * zero_count / count;
            self.count = count;
        }
    }
    pub fn sample_variance(&self) -> F {
        self.m2 / (self.count - F::one())
    }
    /// Standard Error of the Mean (SEM)
    pub fn sem(&self) -> F {
        (self.sample_variance() / self.count).sqrt()
    }
    /// Is the Standard Error of the Mean (SEM) less than the error threshold?
    /// Uses the square of the error for numerical stability (avoids sqrt)
    pub fn sem_le_error_threshold(&self, error_squared: F) -> bool {
        // SEM^2 = self.sample_variance() / self.count
        self.m2 < error_squared * (self.count * (self.count - F::one()))
    }
    pub fn combine(&mut self, other: Self) {
        let count = self.count + other.count;
        let delta = other.mean - self.mean;
        self.mean = (self.count * self.mean + other.count * other.mean) / count;
        self.m2 += other.m2 + delta.sqr() * self.count * other.count / count;
        self.count = count;
    }
}

impl<F: FloatExt> Default for Welford<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: FloatExt> core::ops::Add for Welford<F> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let count = self.count + rhs.count;
        let delta = rhs.mean - self.mean;
        let mean = (self.count * self.mean + rhs.count * rhs.mean) / count;
        let m2 = self.m2 + rhs.m2 + delta.sqr() * self.count * rhs.count / count;
        Welford { count, mean, m2 }
    }
}

impl<F: FloatExt> core::ops::AddAssign for Welford<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.combine(rhs)
    }
}
