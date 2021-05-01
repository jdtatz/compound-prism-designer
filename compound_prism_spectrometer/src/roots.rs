// Root finding code adapted from the rust `roots` crate
// which is licensed under BSD-2-Clause
//
// Copyright (c) 2015, Mikhail Vorotilov
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
use crate::utils::FloatExt;

/// Sorted and unique list of roots of an equation.
#[derive(Debug, PartialEq)]
pub enum Roots<F: FloatExt> {
    /// Equation has no roots
    No([F; 0]),
    /// Equation has one root (or all roots of the equation are the same)
    One([F; 1]),
    /// Equation has two roots
    Two([F; 2]),
    /// Equation has three roots
    Three([F; 3]),
    /// Equation has four roots
    Four([F; 4]),
}

impl<F: FloatExt> AsRef<[F]> for Roots<F> {
    fn as_ref(&self) -> &[F] {
        match self {
            &Roots::No(ref x) => x,
            &Roots::One(ref x) => x,
            &Roots::Two(ref x) => x,
            &Roots::Three(ref x) => x,
            &Roots::Four(ref x) => x,
        }
    }
}

impl<F: FloatExt> Roots<F> {
    fn check_new_root(&self, new_root: F) -> (bool, usize) {
        let mut pos = 0;
        let mut exists = false;

        for x in self.as_ref().iter() {
            if *x == new_root {
                exists = true;
                break;
            }
            if *x > new_root {
                break;
            }
            pos = pos + 1;
        }

        (exists, pos)
    }

    /// Add a new root to existing ones keeping the list of roots ordered and unique.
    pub fn add_new_root(self, new_root: F) -> Self {
        match self {
            Roots::No(_) => Roots::One([new_root]),
            _ => {
                let (exists, pos) = self.check_new_root(new_root);

                if exists {
                    self
                } else {
                    let old_roots = self.as_ref();
                    match (old_roots.len(), pos) {
                        (1, 0) => Roots::Two([new_root, old_roots[0]]),
                        (1, 1) => Roots::Two([old_roots[0], new_root]),
                        (2, 0) => Roots::Three([new_root, old_roots[0], old_roots[1]]),
                        (2, 1) => Roots::Three([old_roots[0], new_root, old_roots[1]]),
                        (2, 2) => Roots::Three([old_roots[0], old_roots[1], new_root]),
                        (3, 0) => Roots::Four([new_root, old_roots[0], old_roots[1], old_roots[2]]),
                        (3, 1) => Roots::Four([old_roots[0], new_root, old_roots[1], old_roots[2]]),
                        (3, 2) => Roots::Four([old_roots[0], old_roots[1], new_root, old_roots[2]]),
                        (3, 3) => Roots::Four([old_roots[0], old_roots[1], old_roots[2], new_root]),
                        _ => panic!("Cannot add root"),
                    }
                }
            }
        }
    }
}

/// Solves a linear equation a1*x + a0 = 0.
///
/// # Examples
///
/// ```
/// use roots::Roots;
/// use roots::find_roots_linear;
///
/// // Returns Roots::No([]) as '0*x + 1 = 0' has no roots;
/// let no_root = find_roots_linear(0f32, 1f32);
/// assert_eq!(no_root, Roots::No([]));
///
/// // Returns Roots::Two([0f64]) as '1*x + 0 = 0' has the root 0
/// let root = find_roots_linear(1f64, 0f64);
/// assert_eq!(root, Roots::One([0f64]));
///
/// // Returns Roots::One([0f32]) as 0 is one of roots of '0*x + 0 = 0'
/// let zero_root = find_roots_linear(0f32, 0f32);
/// assert_eq!(zero_root, Roots::One([0f32]));
/// ```
pub fn find_roots_linear<F: FloatExt>(a1: F, a0: F) -> Roots<F> {
    if a1 == F::zero() {
        if a0 == F::zero() {
            Roots::One([F::zero()])
        } else {
            Roots::No([])
        }
    } else {
        Roots::One([-a0 / a1])
    }
}

/// Solves a quadratic equation a2*x^2 + a1*x + a0 = 0.
///
/// In case two roots are present, the first returned root is less than the second one.
///
/// # Examples
///
/// ```
/// use roots::Roots;
/// use roots::find_roots_quadratic;
///
/// let no_roots = find_roots_quadratic(1f32, 0f32, 1f32);
/// // Returns Roots::No([]) as 'x^2 + 1 = 0' has no roots
///
/// let one_root = find_roots_quadratic(1f64, 0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^2 = 0' has one root 0
///
/// let two_roots = find_roots_quadratic(1f32, 0f32, -1f32);
/// // Returns Roots::Two([-1f32,1f32]) as 'x^2 - 1 = 0' has roots -1 and 1
/// ```
pub fn find_roots_quadratic<F: FloatExt>(a2: F, a1: F, a0: F) -> Roots<F> {
    // Handle non-standard cases
    if a2 == F::zero() {
        // a2 = 0; a1*x+a0=0; solve linear equation
        find_roots_linear(a1, a0)
    } else {
        let _2 = F::lossy_from(2u32);
        let _4 = F::lossy_from(4u32);

        // Rust lacks a simple way to convert an integer constant to generic type F
        let discriminant = a1 * a1 - _4 * a2 * a0;
        if discriminant < F::zero() {
            Roots::No([])
        } else {
            let a2x2 = _2 * a2;
            if discriminant == F::zero() {
                Roots::One([-a1 / a2x2])
            } else {
                // To improve precision, do not use the smallest divisor.
                // See https://people.csail.mit.edu/bkph/articles/Quadratics.pdf
                let sq = discriminant.sqrt();

                let (same_sign, diff_sign) = if a1 < F::zero() {
                    (-a1 + sq, -a1 - sq)
                } else {
                    (-a1 - sq, -a1 + sq)
                };

                let (x1, x2) = if same_sign.abs() > a2x2.abs() {
                    let a0x2 = _2 * a0;
                    if diff_sign.abs() > a2x2.abs() {
                        // 2*a2 is the smallest divisor, do not use it
                        (a0x2 / same_sign, a0x2 / diff_sign)
                    } else {
                        // diff_sign is the smallest divisor, do not use it
                        (a0x2 / same_sign, same_sign / a2x2)
                    }
                } else {
                    // 2*a2 is the greatest divisor, use it
                    (diff_sign / a2x2, same_sign / a2x2)
                };

                // Order roots
                if x1 < x2 {
                    Roots::Two([x1, x2])
                } else {
                    Roots::Two([x2, x1])
                }
            }
        }
    }
}

/// Solves a bi-quadratic equation a4*x^4 + a2*x^2 + a0 = 0.
///
/// Returned roots are arranged in the increasing order.
///
/// # Examples
///
/// ```
/// use roots::find_roots_biquadratic;
///
/// let no_roots = find_roots_biquadratic(1f32, 0f32, 1f32);
/// // Returns Roots::No([]) as 'x^4 + 1 = 0' has no roots
///
/// let one_root = find_roots_biquadratic(1f64, 0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^4 = 0' has one root 0
///
/// let two_roots = find_roots_biquadratic(1f32, 0f32, -1f32);
/// // Returns Roots::Two([-1f32, 1f32]) as 'x^4 - 1 = 0' has roots -1 and 1
/// ```
pub fn find_roots_biquadratic<F: FloatExt>(a4: F, a2: F, a0: F) -> Roots<F> {
    // Handle non-standard cases
    if a4 == F::zero() {
        // a4 = 0; a2*x^2 + a0 = 0; solve quadratic equation
        find_roots_quadratic(a2, F::zero(), a0)
    } else if a0 == F::zero() {
        // a0 = 0; a4*x^4 + a2*x^2 = 0; solve quadratic equation and add zero root
        find_roots_quadratic(a4, F::zero(), a2).add_new_root(F::zero())
    } else {
        // solve the corresponding quadratic equation and order roots
        let mut roots = Roots::No([]);
        for x in find_roots_quadratic(a4, a2, a0).as_ref().iter() {
            if *x > F::zero() {
                let sqrt_x = x.sqrt();
                roots = roots.add_new_root(-sqrt_x).add_new_root(sqrt_x);
            } else if *x == F::zero() {
                roots = roots.add_new_root(F::zero());
            }
        }
        roots
    }
}

const FRAC_2_PI_3: f64 = 2f64 * core::f64::consts::FRAC_PI_3;

/// Solves a depressed cubic equation x^3 + a1*x + a0 = 0.
///
/// In case more than one roots are present, they are returned in the increasing order.
///
/// # Examples
///
/// ```
/// use roots::find_roots_cubic_depressed;
///
/// let one_root = find_roots_cubic_depressed(0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^3 = 0' has one root 0
///
/// let three_roots = find_roots_cubic_depressed(-1f32, 0f32);
/// // Returns Roots::Three([-1f32, -0f32, 1f32]) as 'x^3 - x = 0' has roots -1, 0, and 1
/// ```
pub fn find_roots_cubic_depressed<F: FloatExt>(a1: F, a0: F) -> Roots<F> {
    let _2 = F::lossy_from(2u32);
    let _3 = F::lossy_from(3u32);
    let _4 = F::lossy_from(4u32);
    let _9 = F::lossy_from(9u32);
    let _18 = F::lossy_from(18u32);
    let _27 = F::lossy_from(27u32);
    let _54 = F::lossy_from(54u32);

    if a1 == F::zero() {
        Roots::One([-a0.cbrt()])
    } else if a0 == F::zero() {
        find_roots_quadratic(F::one(), F::zero(), a1).add_new_root(F::zero())
    } else {
        let d = a0 * a0 / _4 + a1 * a1 * a1 / _27;
        if d < F::zero() {
            // n*a0^2 + m*a1^3 < 0 => a1 < 0
            let a = (-_4 * a1 / _3).sqrt();

            let phi = (-_4 * a0 / (a * a * a)).acos() / _3;
            Roots::One([a * phi.cos()])
                .add_new_root(a * (phi + F::lossy_from(FRAC_2_PI_3)).cos())
                .add_new_root(a * (phi - F::lossy_from(FRAC_2_PI_3)).cos())
        } else {
            let sqrt_d = d.sqrt();
            let a0_div_2 = a0 / _2;
            let x1 = (sqrt_d - a0_div_2).cbrt() - (sqrt_d + a0_div_2).cbrt();
            if d == F::zero() {
                // one real root and one double root
                Roots::One([x1]).add_new_root(a0_div_2)
            } else {
                // one real root
                Roots::One([x1])
            }
        }
    }
}

/// Solves a normalized cubic equation x^3 + a2*x^2 + a1*x + a0 = 0.
///
/// Trigonometric solution (arccos/cos) is implemented for three roots.
///
/// In case more than one roots are present, they are returned in the increasing order.
///
/// # Examples
///
/// ```
/// use roots::find_roots_cubic_normalized;
///
/// let one_root = find_roots_cubic_normalized(0f64, 0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^3 = 0' has one root 0
///
/// let three_roots = find_roots_cubic_normalized(0f32, -1f32, 0f32);
/// // Returns Roots::Three([-1f32, -0f32, 1f32]) as 'x^3 - x = 0' has roots -1, 0, and 1
/// ```
pub fn find_roots_cubic_normalized<F: FloatExt>(a2: F, a1: F, a0: F) -> Roots<F> {
    let _2 = F::lossy_from(2u32);
    let _3 = F::lossy_from(3u32);
    let _4 = F::lossy_from(4u32);
    let _9 = F::lossy_from(9u32);
    let _18 = F::lossy_from(18u32);
    let _27 = F::lossy_from(27u32);
    let _54 = F::lossy_from(54u32);

    let q = (_3 * a1 - a2 * a2) / _9;
    let r = (_9 * a2 * a1 - _27 * a0 - _2 * a2 * a2 * a2) / _54;
    let q3 = q * q * q;
    let d = q3 + r * r;
    let a2_div_3 = a2 / _3;

    if d < F::zero() {
        let phi_3 = (r / (-q3).sqrt()).acos() / _3;
        let sqrt_q_2 = _2 * (-q).sqrt();

        Roots::One([sqrt_q_2 * phi_3.cos() - a2_div_3])
            .add_new_root(sqrt_q_2 * (phi_3 - F::lossy_from(FRAC_2_PI_3)).cos() - a2_div_3)
            .add_new_root(sqrt_q_2 * (phi_3 + F::lossy_from(FRAC_2_PI_3)).cos() - a2_div_3)
    } else {
        let sqrt_d = d.sqrt();
        let s = (r + sqrt_d).cbrt();
        let t = (r - sqrt_d).cbrt();

        if s == t {
            if s + t == F::zero() {
                Roots::One([s + t - a2_div_3])
            } else {
                Roots::One([s + t - a2_div_3]).add_new_root(-(s + t) / _2 - a2_div_3)
            }
        } else {
            Roots::One([s + t - a2_div_3])
        }
    }
}

/// Solves a cubic equation a3*x^3 + a2*x^2 + a1*x + a0 = 0.
///
/// General formula (complex numbers) is implemented for three roots.
///
/// Note that very small values of a3 (comparing to other coefficients) will cause the loss of precision.
///
/// In case more than one roots are present, they are returned in the increasing order.
///
/// # Examples
///
/// ```
/// use roots::Roots;
/// use roots::find_roots_cubic;
///
/// let no_roots = find_roots_cubic(0f32, 1f32, 0f32, 1f32);
/// // Returns Roots::No([]) as 'x^2 + 1 = 0' has no roots
///
/// let one_root = find_roots_cubic(1f64, 0f64, 0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^3 = 0' has one root 0
///
/// let three_roots = find_roots_cubic(1f32, 0f32, -1f32, 0f32);
/// // Returns Roots::Three([-1f32, 0f32, 1f32]) as 'x^3 - x = 0' has roots -1, 0, and 1
///
/// let three_roots_less_precision = find_roots_cubic(
///            -0.000000000000000040410628481035f64,
///            0.0126298310280606f64,
///            -0.100896606408756f64,
///            0.0689539597036461f64);
/// // Returns Roots::Three([0.7583841816097057f64, 7.233267996296344f64, 312537357195212.9f64])
/// // while online math expects 0.7547108770537f64, 7.23404258961f64, 312537357195213f64
/// ```
pub fn find_roots_cubic<F: FloatExt>(a3: F, a2: F, a1: F, a0: F) -> Roots<F> {
    // Handle non-standard cases
    if a3 == F::zero() {
        // a3 = 0; a2*x^2+a1*x+a0=0; solve quadratic equation
        find_roots_quadratic(a2, a1, a0)
    } else if a2 == F::zero() {
        // a2 = 0; a3*x^3+a1*x+a0=0; solve depressed cubic equation
        find_roots_cubic_depressed(a1 / a3, a0 / a3)
    } else if a3 == F::one() {
        // solve normalized cubic expression
        find_roots_cubic_normalized(a2, a1, a0)
    } else {
        let _2 = F::lossy_from(2u32);
        let _3 = F::lossy_from(3u32);
        let _4 = F::lossy_from(4u32);
        let _9 = F::lossy_from(9u32);
        let _18 = F::lossy_from(18u32);
        let _27 = F::lossy_from(27u32);

        // standard case
        let d = _18 * a3 * a2 * a1 * a0 - _4 * a2 * a2 * a2 * a0 + a2 * a2 * a1 * a1
            - _4 * a3 * a1 * a1 * a1
            - _27 * a3 * a3 * a0 * a0;
        let d0 = a2 * a2 - _3 * a3 * a1;
        let d1 = _2 * a2 * a2 * a2 - _9 * a3 * a2 * a1 + _27 * a3 * a3 * a0;
        if d < F::zero() {
            // one real root
            let sqrt = (-_27 * a3 * a3 * d).sqrt();
            let c = F::cbrt(if d1 < F::zero() { d1 - sqrt } else { d1 + sqrt } / _2);
            let x = -(a2 + c + d0 / c) / (_3 * a3);
            Roots::One([x])
        } else if d == F::zero() {
            // multiple roots
            if d0 == F::zero() {
                // triple root
                Roots::One([-a2 / (a3 * _3)])
            } else {
                // single root and double root
                Roots::One([(_9 * a3 * a0 - a2 * a1) / (d0 * _2)]).add_new_root(
                    (_4 * a3 * a2 * a1 - _9 * a3 * a3 * a0 - a2 * a2 * a2) / (a3 * d0),
                )
            }
        } else {
            // three real roots
            let c3_img = F::sqrt(_27 * a3 * a3 * d) / _2;
            let c3_real = d1 / _2;
            let c3_module = F::sqrt(c3_img * c3_img + c3_real * c3_real);
            let c3_phase = _2 * F::atan(c3_img / (c3_real + c3_module));
            let c_module = F::cbrt(c3_module);
            let c_phase = c3_phase / _3;
            let c_real = c_module * F::cos(c_phase);
            let c_img = c_module * F::sin(c_phase);
            let x0_real = -(a2 + c_real + (d0 * c_real) / (c_module * c_module)) / (_3 * a3);

            let e_real = -F::one() / _2;
            let e_img = F::sqrt(_3) / _2;
            let c1_real = c_real * e_real - c_img * e_img;
            let c1_img = c_real * e_img + c_img * e_real;
            let x1_real = -(a2 + c1_real + (d0 * c1_real) / (c1_real * c1_real + c1_img * c1_img))
                / (_3 * a3);

            let c2_real = c1_real * e_real - c1_img * e_img;
            let c2_img = c1_real * e_img + c1_img * e_real;
            let x2_real = -(a2 + c2_real + (d0 * c2_real) / (c2_real * c2_real + c2_img * c2_img))
                / (_3 * a3);

            Roots::One([x0_real])
                .add_new_root(x1_real)
                .add_new_root(x2_real)
        }
    }
}

/// Solves a depressed quartic equation x^4 + a2*x^2 + a1*x + a0 = 0.
///
/// Returned roots are ordered. Precision is about 1e-14 for f64.
///
/// # Examples
///
/// ```
/// use roots::find_roots_quartic_depressed;
///
/// let one_root = find_roots_quartic_depressed(1f64, 0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^4 = 0' has one root 0
///
/// let two_roots = find_roots_quartic_depressed(1f32, 0f32, -1f32);
/// // Returns Roots::Two([-1f32, 1f32]) as 'x^4 - 1 = 0' has roots -1 and 1
/// ```
pub fn find_roots_quartic_depressed<F: FloatExt>(a2: F, a1: F, a0: F) -> Roots<F> {
    // Handle non-standard cases
    if a1 == F::zero() {
        // a1 = 0; x^4 + a2*x^2 + a0 = 0; solve biquadratic equation
        find_roots_biquadratic(F::one(), a2, a0)
    } else if a0 == F::zero() {
        // a0 = 0; x^4 + a2*x^2 + a1*x = 0; reduce to normalized cubic and add zero root
        find_roots_cubic_normalized(F::zero(), a2, a1).add_new_root(F::zero())
    } else {
        let _2 = F::lossy_from(2u32);
        let _5 = F::lossy_from(5u32);

        // Solve the auxiliary equation y^3 + (5/2)*a2*y^2 + (2*a2^2-a0)*y + (a2^3/2 - a2*a0/2 - a1^2/8) = 0
        let a2_pow_2 = a2 * a2;
        let a1_div_2 = a1 / _2;
        let b2 = a2 * _5 / _2;
        let b1 = _2 * a2_pow_2 - a0;
        let b0 = (a2_pow_2 * a2 - a2 * a0 - a1_div_2 * a1_div_2) / _2;

        // At least one root always exists. The last root is the maximal one.
        let resolvent_roots = dbg!(find_roots_cubic_normalized(dbg!(b2), dbg!(b1), dbg!(b0)));
        let y = resolvent_roots.as_ref().iter().last().unwrap();

        let _a2_plus_2y = a2 + _2 * *y;
        if _a2_plus_2y > F::zero() {
            let sqrt_a2_plus_2y = _a2_plus_2y.sqrt();
            let q0a = a2 + *y - a1_div_2 / sqrt_a2_plus_2y;
            let q0b = a2 + *y + a1_div_2 / sqrt_a2_plus_2y;

            let mut roots = find_roots_quadratic(F::one(), sqrt_a2_plus_2y, q0a);
            for x in find_roots_quadratic(F::one(), -sqrt_a2_plus_2y, q0b)
                .as_ref()
                .iter()
            {
                roots = roots.add_new_root(*x);
            }
            roots
        } else {
            Roots::No([])
        }
    }
}

/// Solves a quartic equation a4*x^4 + a4*x^3 + a2*x^2 + a1*x + a0 = 0.
/// pp, rr, and dd are already computed while searching for multiple roots
fn find_roots_via_depressed_quartic<F: FloatExt>(
    a4: F,
    a3: F,
    a2: F,
    a1: F,
    a0: F,
    pp: F,
    rr: F,
    dd: F,
) -> Roots<F> {
    // Depressed quartic
    // https://en.wikipedia.org/wiki/Quartic_function#Converting_to_a_depressed_quartic

    let _2 = F::lossy_from(2u32);
    let _3 = F::lossy_from(3u32);
    let _4 = F::lossy_from(4u32);
    let _6 = F::lossy_from(6u32);
    let _8 = F::lossy_from(8u32);
    let _12 = F::lossy_from(12u32);
    let _16 = F::lossy_from(16u32);
    let _256 = F::lossy_from(256u32);

    // a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0 => y^4 + p*y^2 + q*y + r.
    let a4_pow_2 = a4 * a4;
    let a4_pow_3 = a4_pow_2 * a4;
    let a4_pow_4 = a4_pow_2 * a4_pow_2;
    // Re-use pre-calculated values
    let p = pp / (_8 * dbg!(a4_pow_2));
    let q = rr / (_8 * dbg!(a4_pow_3));
    let r =
        (dd + _16 * a4_pow_2 * (_12 * a0 * a4 - _3 * a1 * a3 + a2 * a2)) / (_256 * dbg!(a4_pow_4));

    let mut roots = Roots::No([]);
    for y in find_roots_quartic_depressed(dbg!(p), dbg!(q), dbg!(r))
        .as_ref()
        .iter()
    {
        roots = roots.add_new_root(dbg!(*y) - a3 / (_4 * a4));
    }
    roots
}

/// Solves a quartic equation a4*x^4 + a3*x^3 + a2*x^2 + a1*x + a0 = 0.
///
/// Returned roots are ordered.
/// Precision is about 5e-15 for f64, 5e-7 for f32.
/// WARNING: f32 is often not enough to find multiple roots.
///
/// # Examples
///
/// ```
/// use roots::find_roots_quartic;
///
/// let one_root = find_roots_quartic(1f64, 0f64, 0f64, 0f64, 0f64);
/// // Returns Roots::One([0f64]) as 'x^4 = 0' has one root 0
///
/// let two_roots = find_roots_quartic(1f32, 0f32, 0f32, 0f32, -1f32);
/// // Returns Roots::Two([-1f32, 1f32]) as 'x^4 - 1 = 0' has roots -1 and 1
///
/// let multiple_roots = find_roots_quartic(-14.0625f64, -3.75f64, 29.75f64, 4.0f64, -16.0f64);
/// // Returns Roots::Two([-1.1016116464173349f64, 0.9682783130840016f64])
///
/// let multiple_roots_not_found = find_roots_quartic(-14.0625f32, -3.75f32, 29.75f32, 4.0f32, -16.0f32);
/// // Returns Roots::No([]) because of f32 rounding errors when trying to calculate the discriminant
/// ```
pub fn find_roots_quartic<F: FloatExt>(a4: F, a3: F, a2: F, a1: F, a0: F) -> Roots<F> {
    // Handle non-standard cases
    if a4 == F::zero() {
        // a4 = 0; a3*x^3 + a2*x^2 + a1*x + a0 = 0; solve cubic equation
        find_roots_cubic(a3, a2, a1, a0)
    } else if a0 == F::zero() {
        // a0 = 0; x^4 + a2*x^2 + a1*x = 0; reduce to cubic and arrange results
        find_roots_cubic(a4, a3, a2, a1).add_new_root(F::zero())
    } else if a1 == F::zero() && a3 == F::zero() {
        // a1 = 0, a3 =0; a4*x^4 + a2*x^2 + a0 = 0; solve bi-quadratic equation
        find_roots_biquadratic(a4, a2, a0)
    } else {
        let _3 = F::lossy_from(3u32);
        let _4 = F::lossy_from(4u32);
        let _6 = F::lossy_from(6u32);
        let _8 = F::lossy_from(8u32);
        let _9 = F::lossy_from(9u32);
        let _10 = F::lossy_from(10u32);
        let _12 = F::lossy_from(12u32);
        let _16 = F::lossy_from(16u32);
        let _18 = F::lossy_from(18u32);
        let _27 = F::lossy_from(27u32);
        let _64 = F::lossy_from(64u32);
        let _72 = F::lossy_from(72u32);
        let _80 = F::lossy_from(80u32);
        let _128 = F::lossy_from(128u32);
        let _144 = F::lossy_from(144u32);
        let _192 = F::lossy_from(192u32);
        let _256 = F::lossy_from(256u32);
        // Discriminant
        // https://en.wikipedia.org/wiki/Quartic_function#Nature_of_the_roots
        // Partially simplifed to keep intermediate values smaller (to minimize rounding errors).
        let discriminant = dbg!(
            a4 * a0
                * a4
                * (_256 * a4 * a0 * a0 + a1 * (_144 * a2 * dbg!(a1) - _192 * dbg!(a3) * a0))
        ) + dbg!(
            a4 * a0 * a2 * a2 * (_16 * a2 * a2 - _80 * a3 * a1 - _128 * a4 * a0)
        ) + dbg!(
            a3 * a3
                * (a4 * a0 * (_144 * a2 * a0 - _6 * a1 * a1)
                    + (a0 * (_18 * a3 * a2 * a1 - _27 * a3 * a3 * a0 - _4 * a2 * a2 * a2)
                        + a1 * a1 * (a2 * a2 - _4 * a3 * a1)))
        ) + dbg!(
            a4 * a1 * a1 * (_18 * a3 * a2 * a1 - _27 * a4 * a1 * a1 - _4 * a2 * a2 * a2)
        );
        let pp = _8 * a4 * a2 - _3 * a3 * a3;
        let rr = a3 * a3 * a3 + _8 * a4 * a4 * a1 - _4 * a4 * a3 * a2;
        let delta0 = a2 * a2 - _3 * a3 * a1 + _12 * a4 * a0;
        let dd = _64 * a4 * a4 * a4 * a0 - _16 * a4 * a4 * a2 * a2 + _16 * a4 * a3 * a3 * a2
            - _16 * a4 * a4 * a3 * a1
            - _3 * a3 * a3 * a3 * a3;

        // Handle special cases
        let double_root = dbg!(discriminant) == F::zero();
        if dbg!(double_root) {
            let triple_root = double_root && dbg!(delta0) == F::zero();
            let quadruple_root = triple_root && dbg!(dd) == F::zero();
            let no_roots = dd == F::zero() && dbg!(pp) > F::zero() && dbg!(rr) == F::zero();
            if dbg!(quadruple_root) {
                // Wiki: all four roots are equal
                Roots::One([-a3 / (_4 * a4)])
            } else if dbg!(triple_root) {
                // Wiki: At least three roots are equal to each other
                // x0 is the unique root of the remainder of the Euclidean division of the quartic by its second derivative
                //
                // Solved by SymPy (ra is the desired reminder)
                // a, b, c, d, e = symbols('a,b,c,d,e')
                // f=a*x**4+b*x**3+c*x**2+d*x+e     // Quartic polynom
                // g=6*a*x**2+3*b*x+c               // Second derivative
                // q, r = div(f, g)                 // SymPy only finds the highest power
                // simplify(f-(q*g+r)) == 0         // Verify the first division
                // qa, ra = div(r/a,g/a)            // Workaround to get the second division
                // simplify(f-((q+qa)*g+ra*a)) == 0 // Verify the second division
                // solve(ra,x)
                // ----- yields
                // (−72*a^2*e+10*a*c^2−3*b^2*c)/(9*(8*a^2*d−4*a*b*c+b^3))
                let x0 = (-_72 * a4 * a4 * a0 + _10 * a4 * a2 * a2 - _3 * a3 * a3 * a2)
                    / (_9 * (_8 * a4 * a4 * a1 - _4 * a4 * a3 * a2 + a3 * a3 * a3));
                let roots = dbg!(Roots::One([x0]));
                roots.add_new_root(dbg!(-(a3 / a4 + _3 * x0)))
            } else if dbg!(no_roots) {
                // Wiki: two complex conjugate double roots
                Roots::No([])
            } else {
                find_roots_via_depressed_quartic(a4, a3, a2, a1, a0, pp, dbg!(rr), dd)
            }
        } else {
            let no_roots = dbg!(pp) > F::zero() || dbg!(dd) > F::zero();
            if dbg!(no_roots) {
                // Wiki: two pairs of non-real complex conjugate roots
                Roots::No([])
            } else {
                find_roots_via_depressed_quartic(a4, a3, a2, a1, a0, pp, dbg!(rr), dd)
            }
        }
    }
}
