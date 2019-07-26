use alga::general::RealField;
use core::convert::TryInto;
use core::str::FromStr;
use std::collections::BTreeMap;

#[derive(Debug, Display, Clone, Copy)]
pub enum CatalogError {
    NameNotFound,
    GlassTypeNotFound,
    InvalidGlassDescription,
    GlassDescriptionNotFound,
    UnknownGlassType,
    DuplicateGlass,
}

impl std::error::Error for CatalogError {}

#[derive(Debug, Clone, Copy)]
pub enum Glass<N> {
    Schott([N; 6]),
    Sellmeier1([N; 6]),
}

impl<N: RealField> Glass<N> {
    pub fn new(form: i32, cd: &[N]) -> Result<Self, CatalogError> {
        macro_rules! slice_to_arr {
            ($n:literal) => {
                cd.get(..$n)
                    .and_then(|s| s.try_into().ok())
                    .ok_or(CatalogError::InvalidGlassDescription)?
            };
        }
        Ok(match form {
            1 => Glass::Schott(slice_to_arr!(6)),
            2 => Glass::Sellmeier1(slice_to_arr!(6)),
            _ => return Err(CatalogError::UnknownGlassType),
        })
    }

    pub fn calc_n(&self, w: N) -> N {
        // wavelength must be in micrometers
        match self {
            Glass::Schott(cd) => {
                let &[c1, c2, c3, c4, c5, c6] = cd;
                let w2 = w * w;
                let w4 = w2 * w2;
                let w6 = w2 * w4;
                let w8 = w2 * w6;
                (c1 + c2 * w2 + c3 / w2 + c4 / w4 + c5 / w6 + c6 / w8).sqrt()
            }
            Glass::Sellmeier1(cd) => {
                let &[b1, c1, b2, c2, b3, c3] = cd;
                let w2 = w * w;
                (N::one() + b1 * w2 / (w2 - c1) + b2 * w2 / (w2 - c2) + b3 * w2 / (w2 - c3)).sqrt()
            }
        }
        /*
        formula_rhs = 0
        if dispform == 1:
            formula_rhs = cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + cd[4] * w ** -6 + cd[5] * w ** -8
        elif dispform == 2:  # Sellmeier1
            formula_rhs = 1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                          cd[4] * w ** 2 / (w ** 2 - cd[5])
        elif dispform == 3:  # Herzberger
            L = 1.0 / (w ** 2 - 0.028)
            return cd[0] + cd[1] * L + cd[2] * L ** 2 + cd[3] * w ** 2 + cd[4] * w ** 4 + cd[5] * w ** 6
        elif dispform == 4:  # Sellmeier2
            formula_rhs = 1.0 + cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2] ** 2) + cd[3] * w ** 2 / (w ** 2 - cd[4] ** 2)
        elif dispform == 5:  # Conrady
            return cd[0] + cd[1] / w + cd[2] / w ** 3.5
        elif dispform == 6:  # Sellmeier3
            formula_rhs = 1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                          cd[4] * w ** 2 / (w ** 2 - cd[5]) + cd[6] * w ** 2 / (w ** 2 - cd[7])
        elif dispform == 7:  # HandbookOfOptics1
            formula_rhs = cd[0] + cd[1] / (w ** 2 - cd[2]) - cd[3] * w ** 2
        elif dispform == 8:  # HandbookOfOptics2
            formula_rhs = cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2]) - cd[3] * w ** 2
        elif dispform == 9:  # Sellmeier4
            formula_rhs = cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2]) + cd[3] * w ** 2 / (w ** 2 - cd[4])
        elif dispform == 10:  # Extended1
            formula_rhs = cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + cd[4] * w ** -6 + \
                          cd[5] * w ** -8 + cd[6] * w ** -10 + cd[7] * w ** -12
        elif dispform == 11:  # Sellmeier5
            formula_rhs = 1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                          cd[4] * w ** 2 / (w ** 2 - cd[5]) + cd[6] * w ** 2 / (w ** 2 - cd[7]) + \
                          cd[8] * w ** 2 / (w ** 2 - cd[9])
        elif dispform == 12:  # Extended2
            formula_rhs = cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + \
                          cd[4] * w ** -6 + cd[5] * w ** -8 + cd[6] * w ** 4 + cd[7] * w ** 6
        return np.sqrt(formula_rhs)
            */
    }
}

impl<'a, N: RealField> Into<(i32, &'a [N])> for &'a Glass<N> {
    fn into(self) -> (i32, &'a [N]) {
        match self {
            Glass::Schott(cd) => (1, cd.as_ref()),
            Glass::Sellmeier1(cd) => (2, cd.as_ref()),
        }
    }
}

pub fn new_catalog<N: RealField + FromStr>(
    file: &str,
) -> Result<BTreeMap<String, Glass<N>>, CatalogError> {
    let mut catalog = BTreeMap::new();
    let mut name = None;
    let mut dispform: i32 = -1;
    for line in file.lines() {
        if line.starts_with("NM") {
            let nm: Vec<_> = line.split(' ').collect();
            if name
                .replace(nm.get(1).ok_or(CatalogError::NameNotFound)?.to_uppercase())
                .is_some()
            {
                return Err(CatalogError::GlassDescriptionNotFound);
            }
            dispform = nm
                .get(2)
                .and_then(|d| d.parse().ok())
                .ok_or(CatalogError::GlassTypeNotFound)?;
        } else if line.starts_with("CD") {
            let cd = line
                .get(2..)
                .map(|l| {
                    l.split(' ')
                        .filter_map(|s| s.parse().ok())
                        .collect::<Vec<_>>()
                })
                .ok_or(CatalogError::GlassDescriptionNotFound)?;
            if catalog
                .insert(
                    name.take().ok_or(CatalogError::NameNotFound)?,
                    Glass::new(dispform, &cd)?,
                )
                .is_some()
            {
                return Err(CatalogError::DuplicateGlass);
            }
        }
    }
    Ok(catalog)
}
