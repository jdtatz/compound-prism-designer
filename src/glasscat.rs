use arrayvec::ArrayVec;
use once_cell::sync::Lazy;

#[derive(Debug, Display, Clone, Copy)]
pub enum CatalogError {
    NameNotFound,
    GlassTypeNotFound,
    InvalidGlassDescription,
    GlassDescriptionNotFound,
    UnknownGlassType,
    DuplicateGlass,
}

impl Into<&'static str> for CatalogError {
    fn into(self) -> &'static str {
        match self {
            CatalogError::NameNotFound => "NameNotFound",
            CatalogError::GlassTypeNotFound => "GlassTypeNotFound",
            CatalogError::InvalidGlassDescription => "InvalidGlassDescription",
            CatalogError::GlassDescriptionNotFound => "GlassDescriptionNotFound",
            CatalogError::UnknownGlassType => "UnknownGlassType",
            CatalogError::DuplicateGlass => "DuplicateGlass",
        }
    }
}

/// Glass parametrization structure based off empirical glass dispersion formulae
/// The variant name is which glass dispersion formula it's parameterized by
/// The field of each variant is the array of coefficients that parameterize the glass
///
/// Glass Dispersion Formulae Source:
/// https://neurophysics.ucsd.edu/Manuals/Zemax/ZemaxManual.pdf#page=590
#[derive(Debug, Clone)]
pub enum Glass {
    Schott([f64; 6]),
    Sellmeier1([f64; 6]),
    Sellmeier2([f64; 5]),
    Sellmeier3([f64; 8]),
    Sellmeier4([f64; 5]),
    Sellmeier5([f64; 10]),
    Herzberger([f64; 6]),
    Conrady([f64; 3]),
    HandbookOfOptics1([f64; 4]),
    HandbookOfOptics2([f64; 4]),
    Extended([f64; 8]),
    Extended2([f64; 8]),
    Extended3([f64; 9]),
}

fn iter_to_array<A: arrayvec::Array<Item = f64>>(
    it: impl IntoIterator<Item = f64>,
) -> Result<A, CatalogError> {
    it.into_iter()
        .collect::<ArrayVec<_>>()
        .into_inner()
        .map_err(|_| CatalogError::InvalidGlassDescription)
}

impl Glass {
    /// Create Glass parametrization structure
    ///
    /// # Arguments
    ///  * `num` - dispersion formula number
    ///  * `cd` - dispersion formula coefficients
    pub fn new(num: i32, cd: impl IntoIterator<Item = f64>) -> Result<Self, CatalogError> {
        Ok(match num {
            1 => Glass::Schott(iter_to_array(cd)?),
            2 => Glass::Sellmeier1(iter_to_array(cd)?),
            3 => Glass::Herzberger(iter_to_array(cd)?),
            4 => Glass::Sellmeier2(iter_to_array(cd)?),
            5 => Glass::Conrady(iter_to_array(cd)?),
            6 => Glass::Sellmeier3(iter_to_array(cd)?),
            7 => Glass::HandbookOfOptics1(iter_to_array(cd)?),
            8 => Glass::HandbookOfOptics2(iter_to_array(cd)?),
            9 => Glass::Sellmeier4(iter_to_array(cd)?),
            10 => Glass::Extended(iter_to_array(cd)?),
            11 => Glass::Sellmeier5(iter_to_array(cd)?),
            12 => Glass::Extended2(iter_to_array(cd)?),
            // Unsure if formula Extended3's dispersion formula number is 13
            13 => Glass::Extended3(iter_to_array(cd)?),
            _ => return Err(CatalogError::UnknownGlassType),
        })
    }

    /// Return the index of refraction of the glass for the given wavelength
    ///
    /// # Arguments
    ///  * `w` - wavelength in micrometers
    #[allow(clippy::many_single_char_names)]
    pub fn calc_n(&self, w: f64) -> f64 {
        match self {
            Glass::Schott(cd) => {
                let &[a0, a1, a2, a3, a4, a5] = cd;
                let w2 = w * w;
                let w4 = w2 * w2;
                let w6 = w2 * w4;
                let w8 = w2 * w6;
                (a0 + a1 * w2 + a2 / w2 + a3 / w4 + a4 / w6 + a5 / w8).sqrt()
            }
            Glass::Sellmeier1(cd) => {
                let &[b1, c1, b2, c2, b3, c3] = cd;
                let w2 = w * w;
                (1_f64 + b1 * w2 / (w2 - c1) + b2 * w2 / (w2 - c2) + b3 * w2 / (w2 - c3)).sqrt()
            }
            Glass::Sellmeier2(cd) => {
                let &[a, b1, l1, b2, l2] = cd;
                let w2 = w * w;
                let l1 = l1 * l1;
                let l2 = l2 * l2;
                (1_f64 + a + b1 * w2 / (w2 - l1) + b2 / (w2 - l2)).sqrt()
            }
            Glass::Sellmeier3(cd) => {
                let &[k1, l1, k2, l2, k3, l3, k4, l4] = cd;
                let w2 = w * w;
                (1_f64
                    + k1 * w2 / (w2 - l1)
                    + k2 * w2 / (w2 - l2)
                    + k3 * w2 / (w2 - l3)
                    + k4 * w2 / (w2 - l4))
                    .sqrt()
            }
            Glass::Sellmeier4(cd) => {
                let &[a, b, c, d, e] = cd;
                let w2 = w * w;
                (a + b * w2 / (w2 - c) + d * w2 / (w2 - e)).sqrt()
            }
            Glass::Sellmeier5(cd) => {
                let &[k1, l1, k2, l2, k3, l3, k4, l4, k5, l5] = cd;
                let w2 = w * w;
                (1_f64
                    + k1 * w2 / (w2 - l1)
                    + k2 * w2 / (w2 - l2)
                    + k3 * w2 / (w2 - l3)
                    + k4 * w2 / (w2 - l4)
                    + k5 * w2 / (w2 - l5))
                    .sqrt()
            }
            Glass::Herzberger(cd) => {
                let &[a, b, c, d, e, f] = cd;
                let w2 = w * w;
                let w4 = w2 * w2;
                let w6 = w2 * w4;
                let l = 1_f64 / (w2 - 0.028_f64);
                let l2 = l * l;
                a + b * l + c * l2 + d * w2 + e * w4 + f * w6
            }
            Glass::Conrady(cd) => {
                let &[n0, a, b] = cd;
                n0 + a / w + b / w.powf(3.5)
            }
            Glass::HandbookOfOptics1(cd) => {
                let &[a, b, c, d] = cd;
                let w2 = w * w;
                (a + b / (w2 - c) - d * w2).sqrt()
            }
            Glass::HandbookOfOptics2(cd) => {
                let &[a, b, c, d] = cd;
                let w2 = w * w;
                (a + b * w2 / (w2 - c) - d * w2).sqrt()
            }
            Glass::Extended(cd) => {
                let &[a0, a1, a2, a3, a4, a5, a6, a7] = cd;
                let w2 = w * w;
                let w4 = w2 * w2;
                let w6 = w2 * w4;
                let w8 = w2 * w6;
                let w10 = w2 * w8;
                let w12 = w2 * w10;
                (a0 + a1 * w2 + a2 / w2 + a3 / w4 + a4 / w6 + a5 / w8 + a6 / w10 + a7 / w12).sqrt()
            }
            Glass::Extended2(cd) => {
                let &[a0, a1, a2, a3, a4, a5, a6, a7] = cd;
                let w2 = w * w;
                let w4 = w2 * w2;
                let w6 = w2 * w4;
                let w8 = w2 * w6;
                (a0 + a1 * w2 + a2 / w2 + a3 / w4 + a4 / w6 + a5 / w8 + a6 * w4 + a7 * w6).sqrt()
            }
            Glass::Extended3(cd) => {
                let &[a0, a1, a2, a3, a4, a5, a6, a7, a8] = cd;
                let w2 = w * w;
                let w4 = w2 * w2;
                let w6 = w2 * w4;
                let w8 = w2 * w6;
                let w10 = w2 * w8;
                let w12 = w2 * w10;
                (a0 + a1 * w2
                    + a2 * w4
                    + a3 / w2
                    + a4 / w4
                    + a5 / w6
                    + a6 / w8
                    + a7 / w10
                    + a8 / w12)
                    .sqrt()
            }
        }
    }
}

/// Break down a Glass parametrization structure
/// into its dispersion formula number and coefficients
impl<'g> Into<(i32, &'g [f64])> for &'g Glass {
    fn into(self) -> (i32, &'g [f64]) {
        match self {
            Glass::Schott(cd) => (1, cd.as_ref()),
            Glass::Sellmeier1(cd) => (2, cd.as_ref()),
            Glass::Herzberger(cd) => (3, cd.as_ref()),
            Glass::Sellmeier2(cd) => (4, cd.as_ref()),
            Glass::Conrady(cd) => (5, cd.as_ref()),
            Glass::Sellmeier3(cd) => (6, cd.as_ref()),
            Glass::HandbookOfOptics1(cd) => (7, cd.as_ref()),
            Glass::HandbookOfOptics2(cd) => (8, cd.as_ref()),
            Glass::Sellmeier4(cd) => (9, cd.as_ref()),
            Glass::Extended(cd) => (10, cd.as_ref()),
            Glass::Sellmeier5(cd) => (11, cd.as_ref()),
            Glass::Extended2(cd) => (12, cd.as_ref()),
            Glass::Extended3(cd) => (13, cd.as_ref()),
        }
    }
}

struct CatalogIter<'s> {
    file_lines: core::str::Lines<'s>,
    name: Option<&'s str>,
    dispersion_form: i32,
}

impl<'s> CatalogIter<'s> {
    fn new(file: &'s str) -> Self {
        CatalogIter {
            file_lines: file.lines(),
            name: None,
            dispersion_form: -1,
        }
    }

    fn next_result(&mut self) -> Result<Option<(&'s str, Glass)>, CatalogError> {
        while let Some(line) = self.file_lines.next() {
            if line.starts_with("NM") {
                let mut nm = line.split(' ');
                nm.next(); // == "NM"
                if self
                    .name
                    .replace(nm.next().ok_or(CatalogError::NameNotFound)?)
                    .is_some()
                {
                    return Err(CatalogError::GlassDescriptionNotFound);
                }
                self.dispersion_form = nm
                    .next()
                    .and_then(|d| {
                        d.parse()
                            .ok()
                            .or_else(|| d.parse::<f64>().ok().map(|f| f as _))
                    })
                    .ok_or(CatalogError::GlassTypeNotFound)?;
            } else if line.starts_with("CD") {
                let cd = line
                    .get(2..)
                    .map(|l| l.split(' ').filter_map(|s| s.parse().ok()))
                    .ok_or(CatalogError::GlassDescriptionNotFound)?;
                return Ok(Some((
                    self.name.take().ok_or(CatalogError::NameNotFound)?,
                    Glass::new(self.dispersion_form, cd)?,
                )));
            }
        }
        Ok(None)
    }
}

impl<'s> Iterator for CatalogIter<'s> {
    type Item = Result<(&'s str, Glass), CatalogError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_result().transpose()
    }
}

/// Create glass catalog from .agf file
pub fn new_catalog(file: &str) -> impl Iterator<Item = Result<(&str, Glass), CatalogError>> {
    CatalogIter::new(file)
}

pub const BUNDLED_CATALOG_FILE: &str = include_str!("../catalog.agf");

pub static BUNDLED_CATALOG: Lazy<Box<[(&'static str, Glass)]>> = Lazy::new(|| {
    new_catalog(BUNDLED_CATALOG_FILE)
        .collect::<Result<_, _>>()
        .unwrap()
});
