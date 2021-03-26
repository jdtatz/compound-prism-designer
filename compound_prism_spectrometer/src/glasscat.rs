use core::fmt::{Display, Formatter, Result, Write};

use crate::utils::Float;

#[derive(Debug, Clone, Copy, PartialEq, WrappedFrom)]
#[wrapped_from(trait = "crate::LossyFrom", function = "lossy_from")]
pub struct Glass<F, const N: usize> {
    pub coefficents: [F; N],
}

impl<F: Float, const N: usize> Glass<F, N> {
    /// Return the index of refraction of the glass for the given wavelength
    ///
    /// # Arguments
    ///  * `w` - wavelength in micrometers
    #[inline(always)]
    pub fn calc_n(self, w: F) -> F {
        // core::array::IntoIter::new(self.coefficents)
        self.coefficents
            .iter()
            .copied()
            .reduce(|s, c| s.mul_add(w, c))
            .unwrap()
    }
}

impl<F: Float, const N: usize> Display for Glass<F, N> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        for _ in 0..N {
            f.write_char('(')?;
        }
        let mut it = core::array::IntoIter::new(self.coefficents);
        let cn = it.next().unwrap();
        f.write_fmt(format_args!("{})", cn))?;
        for c in it {
            f.write_fmt(format_args!(" λ + {})", c))?;
        }
        Ok(())
    }
}
