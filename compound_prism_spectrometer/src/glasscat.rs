use crate::utils::{Float, LossyInto};
use arrayvec::ArrayVec;
use core::str::FromStr;

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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Glass<F> {
    Schott([F; 6]),
    Sellmeier1([F; 6]),
    Sellmeier2([F; 5]),
    #[cfg(feature="extended-glass")]
    Sellmeier3([F; 8]),
    Sellmeier4([F; 5]),
    #[cfg(feature="extended-glass")]
    Sellmeier5([F; 10]),
    Herzberger([F; 6]),
    Conrady([F; 3]),
    HandbookOfOptics1([F; 4]),
    HandbookOfOptics2([F; 4]),
    #[cfg(feature="extended-glass")]
    Extended([F; 8]),
    #[cfg(feature="extended-glass")]
    Extended2([F; 8]),
    #[cfg(feature="extended-glass")]
    Extended3([F; 9]),
}

fn iter_to_array<A: arrayvec::Array>(
    it: impl IntoIterator<Item = A::Item>,
) -> Result<A, CatalogError> {
    it.into_iter()
        .collect::<ArrayVec<_>>()
        .into_inner()
        .map_err(|_| CatalogError::InvalidGlassDescription)
}

impl<F: Float> Glass<F> {
    /// Create Glass parametrization structure
    ///
    /// # Arguments
    ///  * `num` - dispersion formula number
    ///  * `cd` - dispersion formula coefficients
    pub fn new(num: i32, cd: impl IntoIterator<Item = F>) -> Result<Self, CatalogError> {
        Ok(match num {
            1 => Glass::Schott(iter_to_array(cd)?),
            2 => Glass::Sellmeier1(iter_to_array(cd)?),
            3 => Glass::Herzberger(iter_to_array(cd)?),
            4 => Glass::Sellmeier2(iter_to_array(cd)?),
            5 => Glass::Conrady(iter_to_array(cd)?),
            #[cfg(feature="extended-glass")]
            6 => Glass::Sellmeier3(iter_to_array(cd)?),
            7 => Glass::HandbookOfOptics1(iter_to_array(cd)?),
            8 => Glass::HandbookOfOptics2(iter_to_array(cd)?),
            #[cfg(feature="extended-glass")]
            9 => Glass::Sellmeier4(iter_to_array(cd)?),
            #[cfg(feature="extended-glass")]
            10 => Glass::Extended(iter_to_array(cd)?),
            #[cfg(feature="extended-glass")]
            11 => Glass::Sellmeier5(iter_to_array(cd)?),
            #[cfg(feature="extended-glass")]
            12 => Glass::Extended2(iter_to_array(cd)?),
            #[cfg(feature="extended-glass")]
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
    pub fn calc_n(&self, w: F) -> F {
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
                (F::one() + b1 * w2 / (w2 - c1) + b2 * w2 / (w2 - c2) + b3 * w2 / (w2 - c3)).sqrt()
            }
            Glass::Sellmeier2(cd) => {
                let &[a, b1, l1, b2, l2] = cd;
                let w2 = w * w;
                let l1 = l1 * l1;
                let l2 = l2 * l2;
                (F::one() + a + b1 * w2 / (w2 - l1) + b2 / (w2 - l2)).sqrt()
            }
            #[cfg(feature="extended-glass")]
            Glass::Sellmeier3(cd) => {
                let &[k1, l1, k2, l2, k3, l3, k4, l4] = cd;
                let w2 = w * w;
                (F::one()
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
            #[cfg(feature="extended-glass")]
            Glass::Sellmeier5(cd) => {
                let &[k1, l1, k2, l2, k3, l3, k4, l4, k5, l5] = cd;
                let w2 = w * w;
                (F::one()
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
                let l = F::one() / (w2 - F::from_u32_ratio(28, 1000));
                let l2 = l * l;
                a + b * l + c * l2 + d * w2 + e * w4 + f * w6
            }
            Glass::Conrady(cd) => {
                let &[n0, a, b] = cd;
                let w_3_5 = w * w * w * w.sqrt();
                n0 + a / w + b / w_3_5
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
            #[cfg(feature="extended-glass")]
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
            #[cfg(feature="extended-glass")]
            Glass::Extended2(cd) => {
                let &[a0, a1, a2, a3, a4, a5, a6, a7] = cd;
                let w2 = w * w;
                let w4 = w2 * w2;
                let w6 = w2 * w4;
                let w8 = w2 * w6;
                (a0 + a1 * w2 + a2 / w2 + a3 / w4 + a4 / w6 + a5 / w8 + a6 * w4 + a7 * w6).sqrt()
            }
            #[cfg(feature="extended-glass")]
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

    pub fn decompose(&self) -> (&'static str, &[F]) {
        match self {
            Glass::Schott(cd) => ("Schott", cd),
            Glass::Sellmeier1(cd) => ("Sellmeier1", cd),
            Glass::Sellmeier2(cd) => ("Sellmeier2", cd),
            #[cfg(feature="extended-glass")]
            Glass::Sellmeier3(cd) => ("Sellmeier3", cd),
            Glass::Sellmeier4(cd) => ("Sellmeier4", cd),
            #[cfg(feature="extended-glass")]
            Glass::Sellmeier5(cd) => ("Sellmeier5", cd),
            Glass::Herzberger(cd) => ("Herzberger", cd),
            Glass::Conrady(cd) => ("Conrady", cd),
            Glass::HandbookOfOptics1(cd) => ("HandbookOfOptics1", cd),
            Glass::HandbookOfOptics2(cd) => ("HandbookOfOptics2", cd),
            #[cfg(feature="extended-glass")]
            Glass::Extended(cd) => ("Extended", cd),
            #[cfg(feature="extended-glass")]
            Glass::Extended2(cd) => ("Extended2", cd),
            #[cfg(feature="extended-glass")]
            Glass::Extended3(cd) => ("Extended3", cd),
        }
    }
}

fn from_iter_to_array<F1: Float + LossyInto<F2>, F2: Float, A: arrayvec::Array<Item = F2>>(
    it: &[F1],
) -> Result<A, CatalogError> {
    it.iter()
        .map(|v| (*v).lossy_into())
        .collect::<ArrayVec<_>>()
        .into_inner()
        .map_err(|_| CatalogError::InvalidGlassDescription)
}

impl<F1: Float + LossyInto<F2>, F2: Float> LossyInto<Glass<F2>> for Glass<F1> {
    fn lossy_into(self) -> Glass<F2> {
        match &self {
            Glass::Schott(cd) => Glass::Schott(from_iter_to_array(cd).unwrap()),
            Glass::Sellmeier1(cd) => Glass::Sellmeier1(from_iter_to_array(cd).unwrap()),
            Glass::Sellmeier2(cd) => Glass::Sellmeier2(from_iter_to_array(cd).unwrap()),
            #[cfg(feature="extended-glass")]
            Glass::Sellmeier3(cd) => Glass::Sellmeier3(from_iter_to_array(cd).unwrap()),
            Glass::Sellmeier4(cd) => Glass::Sellmeier4(from_iter_to_array(cd).unwrap()),
            #[cfg(feature="extended-glass")]
            Glass::Sellmeier5(cd) => Glass::Sellmeier5(from_iter_to_array(cd).unwrap()),
            Glass::Herzberger(cd) => Glass::Herzberger(from_iter_to_array(cd).unwrap()),
            Glass::Conrady(cd) => Glass::Conrady(from_iter_to_array(cd).unwrap()),
            Glass::HandbookOfOptics1(cd) => {
                Glass::HandbookOfOptics1(from_iter_to_array(cd).unwrap())
            }
            Glass::HandbookOfOptics2(cd) => {
                Glass::HandbookOfOptics2(from_iter_to_array(cd).unwrap())
            }
            #[cfg(feature="extended-glass")]
            Glass::Extended(cd) => Glass::Extended(from_iter_to_array(cd).unwrap()),
            #[cfg(feature="extended-glass")]
            Glass::Extended2(cd) => Glass::Extended2(from_iter_to_array(cd).unwrap()),
            #[cfg(feature="extended-glass")]
            Glass::Extended3(cd) => Glass::Extended3(from_iter_to_array(cd).unwrap()),
        }
    }
}

struct CatalogIter<'s, F: Float + FromStr> {
    file_lines: core::str::Lines<'s>,
    name: Option<&'s str>,
    dispersion_form: i32,
    marker: core::marker::PhantomData<F>,
}

impl<'s, F: Float + FromStr> CatalogIter<'s, F> {
    fn new(file: &'s str) -> Self {
        CatalogIter {
            file_lines: file.lines(),
            name: None,
            dispersion_form: -1,
            marker: core::marker::PhantomData,
        }
    }

    fn next_result(&mut self) -> Result<Option<(&'s str, Glass<F>)>, CatalogError> {
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

impl<'s, F: Float + FromStr> Iterator for CatalogIter<'s, F> {
    type Item = Result<(&'s str, Glass<F>), CatalogError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_result().transpose()
    }
}

/// Create glass catalog from .agf file
pub fn new_catalog<F: Float + FromStr>(
    file: &str,
) -> impl Iterator<Item = Result<(&str, Glass<F>), CatalogError>> {
    CatalogIter::new(file)
}

#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
pub const BUNDLED_CATALOG: &[(&str, Glass<f64>)] = &[
    ("F2", Glass::Sellmeier1([1.34533359, 0.00997743871, 0.209073176, 0.0470450767, 0.937357162, 111.886764])),
    ("F5", Glass::Sellmeier1([1.3104463, 0.00958633048, 0.19603426, 0.0457627627, 0.96612977, 115.011883])),
    ("K10", Glass::Sellmeier1([1.15687082, 0.00809424251, 0.0642625444, 0.0386051284, 0.872376139, 104.74773])),
    ("K7", Glass::Sellmeier1([1.1273555, 0.00720341707, 0.124412303, 0.0269835916, 0.827100531, 100.384588])),
    ("LAFN7", Glass::Sellmeier1([1.66842615, 0.0103159999, 0.298512803, 0.0469216348, 1.0774376, 82.5078509])),
    ("LF5", Glass::Sellmeier1([1.28035628, 0.00929854416, 0.163505973, 0.0449135769, 0.893930112, 110.493685])),
    ("LLF1", Glass::Sellmeier1([1.21640125, 0.00857807248, 0.13366454, 0.0420143003, 0.883399468, 107.59306])),
    ("N-BAF10", Glass::Sellmeier1([1.5851495, 0.00926681282, 0.143559385, 0.0424489805, 1.08521269, 105.613573])),
    ("N-BAF4", Glass::Sellmeier1([1.42056328, 0.00942015382, 0.102721269, 0.0531087291, 1.14380976, 110.278856])),
    ("N-BAF51", Glass::Sellmeier1([1.51503623, 0.00942734715, 0.153621958, 0.04308265, 1.15427909, 124.889868])),
    ("N-BAF52", Glass::Sellmeier1([1.43903433, 0.00907800128, 0.0967046052, 0.050821208, 1.09875818, 105.691856])),
    ("N-BAK1", Glass::Sellmeier1([1.12365662, 0.00644742752, 0.309276848, 0.0222284402, 0.881511957, 107.297751])),
    ("N-BAK2", Glass::Sellmeier1([1.01662154, 0.00592383763, 0.319903051, 0.0203828415, 0.937232995, 113.118417])),
    ("N-BAK4", Glass::Sellmeier1([1.28834642, 0.00779980626, 0.132817724, 0.0315631177, 0.945395373, 105.965875])),
    ("N-BALF4", Glass::Sellmeier1([1.31004128, 0.0079659645, 0.142038259, 0.0330672072, 0.964929351, 109.19732])),
    ("N-BALF5", Glass::Sellmeier1([1.28385965, 0.00825815975, 0.0719300942, 0.0441920027, 1.05048927, 107.097324])),
    ("N-BASF2", Glass::Sellmeier1([1.53652081, 0.0108435729, 0.156971102, 0.0562278762, 1.30196815, 131.3397])),
    ("N-BASF64", Glass::Sellmeier1([1.65554268, 0.0104485644, 0.17131977, 0.0499394756, 1.33664448, 118.961472])),
    ("N-BK10", Glass::Sellmeier1([0.888308131, 0.00516900822, 0.328964475, 0.0161190045, 0.984610769, 99.7575331])),
    ("N-BK7", Glass::Sellmeier1([1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653])),
    ("N-F2", Glass::Sellmeier1([1.39757037, 0.00995906143, 0.159201403, 0.0546931752, 1.2686543, 119.248346])),
    ("N-FK5", Glass::Sellmeier1([0.844309338, 0.00475111955, 0.344147824, 0.0149814849, 0.910790213, 97.8600293])),
    ("N-FK51A", Glass::Sellmeier1([0.971247817, 0.00472301995, 0.216901417, 0.0153575612, 0.904651666, 168.68133])),
    ("N-K5", Glass::Sellmeier1([1.08511833, 0.00661099503, 0.199562005, 0.024110866, 0.930511663, 111.982777])),
    ("N-KF9", Glass::Sellmeier1([1.19286778, 0.00839154696, 0.0893346571, 0.0404010786, 0.920819805, 112.572446])),
    ("N-KZFS11", Glass::Sellmeier1([1.3322245, 0.0084029848, 0.28924161, 0.034423972, 1.15161734, 88.4310532])),
    ("N-KZFS2", Glass::Sellmeier1([1.23697554, 0.00747170505, 0.153569376, 0.0308053556, 0.903976272, 70.1731084])),
    ("N-KZFS4", Glass::Sellmeier1([1.35055424, 0.0087628207, 0.197575506, 0.0371767201, 1.09962992, 90.3866994])),
    ("N-KZFS5", Glass::Sellmeier1([1.47460789, 0.00986143816, 0.193584488, 0.0445477583, 1.26589974, 106.436258])),
    ("N-KZFS8", Glass::Sellmeier1([1.62693651, 0.010880863, 0.24369876, 0.0494207753, 1.62007141, 131.009163])),
    ("N-LAF2", Glass::Sellmeier1([1.80984227, 0.0101711622, 0.15729555, 0.0442431765, 1.0930037, 100.687748])),
    ("N-LAF21", Glass::Sellmeier1([1.87134529, 0.0093332228, 0.25078301, 0.0345637762, 1.22048639, 83.2404866])),
    ("N-LAF33", Glass::Sellmeier1([1.79653417, 0.00927313493, 0.311577903, 0.0358201181, 1.15981863, 87.3448712])),
    ("N-LAF34", Glass::Sellmeier1([1.75836958, 0.00872810026, 0.313537785, 0.0293020832, 1.18925231, 85.1780644])),
    ("N-LAF35", Glass::Sellmeier1([1.51697436, 0.00750943203, 0.455875464, 0.0260046715, 1.07469242, 80.5945159])),
    ("N-LAF7", Glass::Sellmeier1([1.74028764, 0.010792558, 0.226710554, 0.0538626639, 1.32525548, 106.268665])),
    ("N-LAK10", Glass::Sellmeier1([1.72878017, 0.00886014635, 0.169257825, 0.0363416509, 1.19386956, 82.9009069])),
    ("N-LAK12", Glass::Sellmeier1([1.17365704, 0.00577031797, 0.588992398, 0.0200401678, 0.978014394, 95.4873482])),
    ("N-LAK14", Glass::Sellmeier1([1.50781212, 0.00746098727, 0.318866829, 0.0242024834, 1.14287213, 80.9565165])),
    ("N-LAK21", Glass::Sellmeier1([1.22718116, 0.00602075682, 0.420783743, 0.0196862889, 1.01284843, 88.4370099])),
    ("N-LAK22", Glass::Sellmeier1([1.14229781, 0.00585778594, 0.535138441, 0.0198546147, 1.04088385, 100.834017])),
    ("N-LAK33B", Glass::Sellmeier1([1.42288601, 0.00670283452, 0.593661336, 0.021941621, 1.1613526, 80.7407701])),
    ("N-LAK34", Glass::Sellmeier1([1.26661442, 0.00589278062, 0.665919318, 0.0197509041, 1.1249612, 78.8894174])),
    ("N-LAK7", Glass::Sellmeier1([1.23679889, 0.00610105538, 0.445051837, 0.0201388334, 1.01745888, 90.638038])),
    ("N-LAK8", Glass::Sellmeier1([1.33183167, 0.00620023871, 0.546623206, 0.0216465439, 1.19084015, 82.5827736])),
    ("N-LAK9", Glass::Sellmeier1([1.46231905, 0.00724270156, 0.344399589, 0.0243353131, 1.15508372, 85.4686868])),
    ("N-LASF31A", Glass::Sellmeier1([1.96485075, 0.00982060155, 0.475231259, 0.0344713438, 1.48360109, 110.739863])),
    ("N-LASF40", Glass::Sellmeier1([1.98550331, 0.010958331, 0.274057042, 0.0474551603, 1.28945661, 96.9085286])),
    ("N-LASF41", Glass::Sellmeier1([1.86348331, 0.00910368219, 0.413307255, 0.0339247268, 1.35784815, 93.3580595])),
    ("N-LASF43", Glass::Sellmeier1([1.93502827, 0.0104001413, 0.23662935, 0.0447505292, 1.26291344, 87.437569])),
    ("N-LASF44", Glass::Sellmeier1([1.78897105, 0.00872506277, 0.38675867, 0.0308085023, 1.30506243, 92.7743824])),
    ("N-LASF45", Glass::Sellmeier1([1.87140198, 0.011217192, 0.267777879, 0.0505134972, 1.73030008, 147.106505])),
    ("N-LASF46A", Glass::Sellmeier1([2.16701566, 0.0123595524, 0.319812761, 0.0560610282, 1.66004486, 107.047718])),
    ("N-LASF46B", Glass::Sellmeier1([2.17988922, 0.0125805384, 0.306495184, 0.0567191367, 1.56882437, 105.316538])),
    ("N-LASF9", Glass::Sellmeier1([2.00029547, 0.0121426017, 0.298926886, 0.0538736236, 1.80691843, 156.530829])),
    ("N-PK51", Glass::Sellmeier1([1.15610775, 0.00585597402, 0.153229344, 0.0194072416, 0.785618966, 140.537046])),
    ("N-PK52A", Glass::Sellmeier1([1.029607, 0.00516800155, 0.1880506, 0.0166658798, 0.736488165, 138.964129])),
    ("N-PSK3", Glass::Sellmeier1([0.88727211, 0.00469824067, 0.489592425, 0.0161818463, 1.04865296, 104.374975])),
    ("N-PSK53A", Glass::Sellmeier1([1.38121836, 0.00706416337, 0.196745645, 0.0233251345, 0.886089205, 97.4847345])),
    ("N-SF1", Glass::Sellmeier1([1.60865158, 0.0119654879, 0.237725916, 0.0590589722, 1.51530653, 135.521676])),
    ("N-SF10", Glass::Sellmeier1([1.62153902, 0.0122241457, 0.256287842, 0.0595736775, 1.64447552, 147.468793])),
    ("N-SF11", Glass::Sellmeier1([1.73759695, 0.013188707, 0.313747346, 0.0623068142, 1.89878101, 155.23629])),
    ("N-SF14", Glass::Sellmeier1([1.69022361, 0.0130512113, 0.288870052, 0.061369188, 1.7045187, 149.517689])),
    ("N-SF15", Glass::Sellmeier1([1.57055634, 0.0116507014, 0.218987094, 0.0597856897, 1.50824017, 132.709339])),
    ("N-SF2", Glass::Sellmeier1([1.47343127, 0.0109019098, 0.163681849, 0.0585683687, 1.36920899, 127.404933])),
    ("N-SF4", Glass::Sellmeier1([1.67780282, 0.012679345, 0.282849893, 0.0602038419, 1.63539276, 145.760496])),
    ("N-SF5", Glass::Sellmeier1([1.52481889, 0.011254756, 0.187085527, 0.0588995392, 1.42729015, 129.141675])),
    ("N-SF57", Glass::Sellmeier1([1.87543831, 0.0141749518, 0.37375749, 0.0640509927, 2.30001797, 177.389795])),
    ("N-SF6", Glass::Sellmeier1([1.77931763, 0.0133714182, 0.338149866, 0.0617533621, 2.08734474, 174.01759])),
    ("N-SF66", Glass::Sellmeier1([2.0245976, 0.0147053225, 0.470187196, 0.0692998276, 2.59970433, 161.817601])),
    ("N-SF8", Glass::Sellmeier1([1.55075812, 0.0114338344, 0.209816918, 0.0582725652, 1.46205491, 133.24165])),
    ("N-SK11", Glass::Sellmeier1([1.17963631, 0.00680282081, 0.229817295, 0.0219737205, 0.935789652, 101.513232])),
    ("N-SK14", Glass::Sellmeier1([0.936155374, 0.00461716525, 0.594052018, 0.016885927, 1.04374583, 103.736265])),
    ("N-SK16", Glass::Sellmeier1([1.34317774, 0.00704687339, 0.241144399, 0.0229005, 0.994317969, 92.7508526])),
    ("N-SK2", Glass::Sellmeier1([1.28189012, 0.0072719164, 0.257738258, 0.0242823527, 0.96818604, 110.377773])),
    ("N-SK4", Glass::Sellmeier1([1.32993741, 0.00716874107, 0.228542996, 0.0246455892, 0.988465211, 100.886364])),
    ("N-SK5", Glass::Sellmeier1([0.991463823, 0.00522730467, 0.495982121, 0.0172733646, 0.987393925, 98.3594579])),
    ("N-SSK2", Glass::Sellmeier1([1.4306027, 0.00823982975, 0.153150554, 0.0333736841, 1.01390904, 106.870822])),
    ("N-SSK5", Glass::Sellmeier1([1.59222659, 0.00920284626, 0.103520774, 0.0423530072, 1.05174016, 106.927374])),
    ("N-SSK8", Glass::Sellmeier1([1.44857867, 0.00869310149, 0.117965926, 0.0421566593, 1.06937528, 111.300666])),
    ("N-ZK7", Glass::Sellmeier1([1.07715032, 0.00676601657, 0.168079109, 0.0230642817, 0.851889892, 89.0498778])),
    ("P-LAF37", Glass::Sellmeier1([1.76003244, 0.00938006396, 0.248286745, 0.0360537464, 1.15935122, 86.4324693])),
    ("P-LAK35", Glass::Sellmeier1([1.3932426, 0.00715959695, 0.418882766, 0.0233637446, 1.043807, 88.3284426])),
    ("P-LASF47", Glass::Sellmeier1([1.85543101, 0.0100328203, 0.315854649, 0.0387095168, 1.28561839, 94.5421507])),
    ("P-LASF50", Glass::Sellmeier1([1.84910553, 0.00999234757, 0.329828674, 0.0387437988, 1.30400901, 95.8967681])),
    ("P-LASF51", Glass::Sellmeier1([1.84568806, 0.00988495571, 0.3390016, 0.0378097402, 1.32418921, 97.841543])),
    ("P-SF68", Glass::Sellmeier1([2.3330067, 0.0168838419, 0.452961396, 0.0716086325, 1.25172339, 118.707479])),
    ("P-SF69", Glass::Sellmeier1([1.62594647, 0.0121696677, 0.235927609, 0.0600710405, 1.67434623, 145.651908])),
    ("P-SF8", Glass::Sellmeier1([1.55370411, 0.011658267, 0.206332561, 0.0582087757, 1.39708831, 130.748028])),
    ("P-SK57", Glass::Sellmeier1([1.31053414, 0.00740877235, 0.169376189, 0.0254563489, 1.10987714, 107.751087])),
    ("P-SK58A", Glass::Sellmeier1([1.3167841, 0.00720717498, 0.171154756, 0.0245659595, 1.12501473, 102.739728])),
    ("P-SK60", Glass::Sellmeier1([1.40790442, 0.00784382378, 0.143381417, 0.0287769365, 1.16513947, 105.373397])),
    ("SF1", Glass::Sellmeier1([1.55912923, 0.0121481001, 0.284246288, 0.0534549042, 0.968842926, 112.174809])),
    ("SF10", Glass::Sellmeier1([1.61625977, 0.0127534559, 0.259229334, 0.0581983954, 1.07762317, 116.60768])),
    ("SF2", Glass::Sellmeier1([1.40301821, 0.0105795466, 0.231767504, 0.0493226978, 0.939056586, 112.405955])),
    ("SF4", Glass::Sellmeier1([1.61957826, 0.0125502104, 0.339493189, 0.0544559822, 1.02566931, 117.652222])),
    ("SF5", Glass::Sellmeier1([1.46141885, 0.0111826126, 0.247713019, 0.0508594669, 0.949995832, 112.041888])),
    ("SF56A", Glass::Sellmeier1([1.70579259, 0.0133874699, 0.344223052, 0.0579561608, 1.09601828, 121.616024])),
    ("SF57", Glass::Sellmeier1([1.81651371, 0.0143704198, 0.428893641, 0.0592801172, 1.07186278, 121.419942])),
    ("SF6", Glass::Sellmeier1([1.72448482, 0.0134871947, 0.390104889, 0.0569318095, 1.04572858, 118.557185])),
    ("SF11", Glass::Sellmeier1([1.73848403, 0.0136068604, 0.311168974, 0.0615960463, 1.17490871, 121.922711])),
    ("LASF35", Glass::Sellmeier1([2.45505861, 0.0135670404, 0.453006077, 0.054580302, 2.3851308, 167.904715])),
    ("N-FK58", Glass::Sellmeier1([0.738042712, 0.00339065607, 0.363371967, 0.0117551189, 0.989296264, 212.842145])),
];

#[cfg(test)]
mod tests {
    use super::*;
    const BUNDLED_CATALOG_FILE: &str = include_str!("catalog.agf");

    fn generate_bundled_catalog() {
        println!("pub const BUNDLED_CATALOG: &[(&str, Glass<f64>)] = &[");
        for r in new_catalog::<f64>(BUNDLED_CATALOG_FILE) {
            let (n, g) = r.unwrap();
            println!("  (\"{}\", Glass::{:?}),", n, g);
        }
        println!("];");
    }

    #[test]
    fn test_bundled_catalog() {
        for (e1, e2) in BUNDLED_CATALOG
            .iter()
            .zip(new_catalog(BUNDLED_CATALOG_FILE))
        {
            let e2 = e2.unwrap();
            if e1 != &e2 {
                // To output the new code for the new catalog
                generate_bundled_catalog();
                assert_eq!(e1, &e2);
            }
        }
    }
}
