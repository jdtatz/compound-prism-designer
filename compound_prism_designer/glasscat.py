"""
Glass parametrization structure based off empirical glass dispersion formulae
The variant name is which glass dispersion formula it's parameterized by
The field of each variant is the array of coefficients that parameterize the glass

Glass Dispersion Formulae Source:
https://neurophysics.ucsd.edu/Manuals/Zemax/ZemaxManual.pdf#page=590
"""
from __future__ import annotations

from abc import ABCMeta, abstractmethod
from enum import Enum, auto
from functools import reduce
from typing import ClassVar, Dict, Iterator, Optional, Tuple, Type

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from .compound_prism_designer import Glass

__all__ = [
    "GlassCatalogErrorReason",
    "GlassCatalogError",
    "AbstractGlass",
    "Schott",
    "Sellmeier1",
    "Sellmeier2",
    "Sellmeier3",
    "Sellmeier4",
    "Sellmeier5",
    "Herzberger",
    "Conrady",
    "HandbookOfOptics1",
    "HandbookOfOptics2",
    "Extended1",
    "Extended2",
    "Extended3",
    "new_catalog",
    "BUNDLED_CATALOG",
]


class GlassCatalogErrorReason(Exception, Enum):
    NameNotFound = auto()
    GlassTypeNotFound = auto()
    InvalidGlassDescription = auto()
    GlassDescriptionNotFound = auto()
    UnknownGlassType = auto()
    DuplicateGlass = auto()

    def __str__(self):
        return self.name


class GlassCatalogError(Exception):
    reason: GlassCatalogErrorReason

    def __init__(self, reason: GlassCatalogErrorReason):
        self.reason = reason
        super().__init__(reason)


_glass_registry: Dict[int, Type[AbstractGlass]] = {}


class AbstractGlass(metaclass=ABCMeta):
    catalog_index: ClassVar[int]
    ncoeff: ClassVar[int]

    name: str
    coefficents: Tuple[float, ...]
    wavelength_range: Tuple[float, float]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.catalog_index in _glass_registry:
            raise TypeError(
                f"Catalog Index Conflict, {cls.catalog_index} has already been registered to {_glass_registry[cls.catalog_index]}"
            )
        else:
            _glass_registry[cls.catalog_index] = cls

    def __init__(
        self,
        name: str,
        coefficents: Tuple[float, ...],
        wavelength_range: Tuple[float, float] = (0, np.inf),
    ):
        self.name = name
        if len(coefficents) < self.ncoeff or any(c != 0.0 for c in coefficents[self.ncoeff :]):
            raise GlassCatalogError(GlassCatalogErrorReason.InvalidGlassDescription)
        self.coefficents = coefficents[: self.ncoeff]
        self.wavelength_range = wavelength_range

    def __repr__(self):
        n = type(self).__name__
        return f'{type(self).__name__}(name="{self.name}", coefficents={self.coefficents}, wavelength_range={self.wavelength_range})'

    @abstractmethod
    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        pass

    def __call__(self, wavelength: ArrayLike) -> np.ndarray:
        wavelength = np.asarray(wavelength)
        assert (
            np.logical_and(
                self.wavelength_range[0] < wavelength,
                wavelength < self.wavelength_range[1],
            )
        ).all()
        return self.index_of_refraction(wavelength)

    def horner_polynomial_fit(self, limits: Tuple[float, float], n: int) -> Tuple[Tuple[float, ...], float]:
        assert limits[0] < limits[1]
        assert self.wavelength_range[0] < limits[0] and limits[1] < self.wavelength_range[1]
        x = np.linspace(*limits, 100)
        y = self(x)
        f = lambda x, *coeffs: reduce(lambda a, c: a * x + c, coeffs)
        popt, pcov = curve_fit(f, x, y, p0=np.ones(1 + n))
        r = y - f(x, *popt)
        chisq = np.sum(r**2)
        return popt, chisq

    def into_glass(self, limits: Tuple[float, float]) -> Tuple[Glass, float]:
        popt, err = self.horner_polynomial_fit(limits, Glass.ORDER)
        return Glass(self.name, np.asarray(popt)), err


class Schott(AbstractGlass):
    catalog_index = 1
    ncoeff = 6

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a0, a1, a2, a3, a4, a5 = self.coefficents
        w2 = w * w
        w4 = w2 * w2
        w6 = w2 * w4
        w8 = w2 * w6
        return np.sqrt(a0 + a1 * w2 + a2 / w2 + a3 / w4 + a4 / w6 + a5 / w8)


class Sellmeier1(AbstractGlass):
    catalog_index = 2
    ncoeff = 6

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        b1, c1, b2, c2, b3, c3 = self.coefficents
        w2 = w * w
        return np.sqrt(1 + b1 * w2 / (w2 - c1) + b2 * w2 / (w2 - c2) + b3 * w2 / (w2 - c3))


class Sellmeier2(AbstractGlass):
    catalog_index = 4
    ncoeff = 5

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a, b1, l1, b2, l2 = self.coefficents
        w2 = w * w
        l1 = l1 * l1
        l2 = l2 * l2
        return np.sqrt(1 + a + b1 * w2 / (w2 - l1) + b2 / (w2 - l2))


class Sellmeier3(AbstractGlass):
    catalog_index = 6
    ncoeff = 8

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        k1, l1, k2, l2, k3, l3, k4, l4 = self.coefficents
        w2 = w * w
        return np.sqrt(1 + k1 * w2 / (w2 - l1) + k2 * w2 / (w2 - l2) + k3 * w2 / (w2 - l3) + k4 * w2 / (w2 - l4))


class Sellmeier4(AbstractGlass):
    catalog_index = 9
    ncoeff = 5

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a, b, c, d, e = self.coefficents
        w2 = w * w
        return np.sqrt(a + b * w2 / (w2 - c) + d * w2 / (w2 - e))


class Sellmeier5(AbstractGlass):
    catalog_index = 11
    ncoeff = 10

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        k1, l1, k2, l2, k3, l3, k4, l4, k5, l5 = self.coefficents
        w2 = w * w
        return np.sqrt(
            1
            + k1 * w2 / (w2 - l1)
            + k2 * w2 / (w2 - l2)
            + k3 * w2 / (w2 - l3)
            + k4 * w2 / (w2 - l4)
            + k5 * w2 / (w2 - l5)
        )


class Herzberger(AbstractGlass):
    catalog_index = 3
    ncoeff = 6

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a, b, c, d, e, f = self.coefficents
        w2 = w * w
        w4 = w2 * w2
        w6 = w2 * w4
        l = 1 / (w2 - 0.028)
        l2 = l * l
        return a + b * l + c * l2 + d * w2 + e * w4 + f * w6


class Conrady(AbstractGlass):
    catalog_index = 5
    ncoeff = 3

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        n0, a, b = self.coefficents
        w_3_5 = w * w * w * np.sqrt(w)
        return n0 + a / w + b / w_3_5


class HandbookOfOptics1(AbstractGlass):
    catalog_index = 7
    ncoeff = 4

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a, b, c, d = self.coefficents
        w2 = w * w
        return np.sqrt(a + b / (w2 - c) - d * w2)


class HandbookOfOptics2(AbstractGlass):
    catalog_index = 8
    ncoeff = 4

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a, b, c, d = self.coefficents
        w2 = w * w
        return np.sqrt(a + b * w2 / (w2 - c) - d * w2)


class Extended1(AbstractGlass):
    catalog_index = 10
    ncoeff = 8

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a0, a1, a2, a3, a4, a5, a6, a7 = self.coefficents
        w2 = w * w
        w4 = w2 * w2
        w6 = w2 * w4
        w8 = w2 * w6
        w10 = w2 * w8
        w12 = w2 * w10
        return np.sqrt(a0 + a1 * w2 + a2 / w2 + a3 / w4 + a4 / w6 + a5 / w8 + a6 / w10 + a7 / w12)


class Extended2(AbstractGlass):
    catalog_index = 12
    ncoeff = 8

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a0, a1, a2, a3, a4, a5, a6, a7 = self.coefficents
        w2 = w * w
        w4 = w2 * w2
        w6 = w2 * w4
        w8 = w2 * w6
        return np.sqrt(a0 + a1 * w2 + a2 / w2 + a3 / w4 + a4 / w6 + a5 / w8 + a6 * w4 + a7 * w6)


class Extended3(AbstractGlass):
    catalog_index = 13
    ncoeff = 9

    def index_of_refraction(self, wavelength: ArrayLike) -> np.ndarray:
        w = np.asarray(wavelength)
        a0, a1, a2, a3, a4, a5, a6, a7, a8 = self.coefficents
        w2 = w * w
        w4 = w2 * w2
        w6 = w2 * w4
        w8 = w2 * w6
        w10 = w2 * w8
        w12 = w2 * w10
        return np.sqrt(a0 + a1 * w2 + a2 * w4 + a3 / w2 + a4 / w4 + a5 / w6 + a6 / w8 + a7 / w10 + a8 / w12)


def new_catalog(file: str) -> Iterator[AbstractGlass]:
    name_glcs: Optional[Tuple[str, Type[AbstractGlass]]] = None
    coeffs: Optional[Tuple[float, ...]] = None
    wavelength_range: Optional[Tuple[float, float]] = None
    for line in file.splitlines():
        line = line.strip()
        if line.startswith("NM"):
            if name_glcs is not None:
                raise GlassCatalogError(GlassCatalogErrorReason.GlassDescriptionNotFound)
            nm, name, dform, *_ = line.split(" ")
            dispersion_form = int(dform)
            gcls = _glass_registry.get(dispersion_form, None)
            if gcls is None:
                raise GlassCatalogError(GlassCatalogErrorReason.InvalidGlassDescription)
            name_glcs = name, gcls
        elif line.startswith("CD"):
            if coeffs is not None:
                raise GlassCatalogError(GlassCatalogErrorReason.NameNotFound)
            coeffs = tuple(map(float, line.split(" ")[1:]))
        elif line.startswith("LD"):
            if wavelength_range is not None:
                raise GlassCatalogError(GlassCatalogErrorReason.GlassDescriptionNotFound)
            _, l, u = line.split(" ")
            wavelength_range = float(l), float(u)
        if name_glcs is not None and coeffs is not None and wavelength_range is not None:
            name, gcls = name_glcs
            yield gcls(name, coeffs, wavelength_range)
            name_glcs = None
            coeffs = None
            wavelength_range = None


# with open("compound_prism_designer/catalog.agf") as f:
#     bundled = {g.name: g for g in new_catalog(f.read())}


# fmt: off
BUNDLED_CATALOG = (
    Sellmeier1(name="F2", coefficents=(1.34533359, 0.00997743871, 0.209073176, 0.0470450767, 0.937357162, 111.886764), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="F5", coefficents=(1.3104463, 0.00958633048, 0.19603426, 0.0457627627, 0.96612977, 115.011883), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="K10", coefficents=(1.15687082, 0.00809424251, 0.0642625444, 0.0386051284, 0.872376139, 104.74773), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="K7", coefficents=(1.1273555, 0.00720341707, 0.124412303, 0.0269835916, 0.827100531, 100.384588), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="LAFN7", coefficents=(1.66842615, 0.0103159999, 0.298512803, 0.0469216348, 1.0774376, 82.5078509), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="LF5", coefficents=(1.28035628, 0.00929854416, 0.163505973, 0.0449135769, 0.893930112, 110.493685), wavelength_range=(0.32, 2.325)),
    Sellmeier1(name="LLF1", coefficents=(1.21640125, 0.00857807248, 0.13366454, 0.0420143003, 0.883399468, 107.59306), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-BAF10", coefficents=(1.5851495, 0.00926681282, 0.143559385, 0.0424489805, 1.08521269, 105.613573), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-BAF4", coefficents=(1.42056328, 0.00942015382, 0.102721269, 0.0531087291, 1.14380976, 110.278856), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-BAF51", coefficents=(1.51503623, 0.00942734715, 0.153621958, 0.04308265, 1.15427909, 124.889868), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-BAF52", coefficents=(1.43903433, 0.00907800128, 0.0967046052, 0.050821208, 1.09875818, 105.691856), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-BAK1", coefficents=(1.12365662, 0.00644742752, 0.309276848, 0.0222284402, 0.881511957, 107.297751), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="N-BAK2", coefficents=(1.01662154, 0.00592383763, 0.319903051, 0.0203828415, 0.937232995, 113.118417), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="N-BAK4", coefficents=(1.28834642, 0.00779980626, 0.132817724, 0.0315631177, 0.945395373, 105.965875), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-BALF4", coefficents=(1.31004128, 0.0079659645, 0.142038259, 0.0330672072, 0.964929351, 109.19732), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-BALF5", coefficents=(1.28385965, 0.00825815975, 0.0719300942, 0.0441920027, 1.05048927, 107.097324), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-BASF2", coefficents=(1.53652081, 0.0108435729, 0.156971102, 0.0562278762, 1.30196815, 131.3397), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-BASF64", coefficents=(1.65554268, 0.0104485644, 0.17131977, 0.0499394756, 1.33664448, 118.961472), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-BK10", coefficents=(0.888308131, 0.00516900822, 0.328964475, 0.0161190045, 0.984610769, 99.7575331), wavelength_range=(0.29, 2.5)),
    Sellmeier1(name="N-BK7", coefficents=(1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="N-F2", coefficents=(1.39757037, 0.00995906143, 0.159201403, 0.0546931752, 1.2686543, 119.248346), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-FK5", coefficents=(0.844309338, 0.00475111955, 0.344147824, 0.0149814849, 0.910790213, 97.8600293), wavelength_range=(0.26, 2.5)),
    Sellmeier1(name="N-FK51A", coefficents=(0.971247817, 0.00472301995, 0.216901417, 0.0153575612, 0.904651666, 168.68133), wavelength_range=(0.29, 2.5)),
    Sellmeier1(name="N-K5", coefficents=(1.08511833, 0.00661099503, 0.199562005, 0.024110866, 0.930511663, 111.982777), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-KF9", coefficents=(1.19286778, 0.00839154696, 0.0893346571, 0.0404010786, 0.920819805, 112.572446), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-KZFS11", coefficents=(1.3322245, 0.0084029848, 0.28924161, 0.034423972, 1.15161734, 88.4310532), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-KZFS2", coefficents=(1.23697554, 0.00747170505, 0.153569376, 0.0308053556, 0.903976272, 70.1731084), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-KZFS4", coefficents=(1.35055424, 0.0087628207, 0.197575506, 0.0371767201, 1.09962992, 90.3866994), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-KZFS5", coefficents=(1.47460789, 0.00986143816, 0.193584488, 0.0445477583, 1.26589974, 106.436258), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-KZFS8", coefficents=(1.62693651, 0.010880863, 0.24369876, 0.0494207753, 1.62007141, 131.009163), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-LAF2", coefficents=(1.80984227, 0.0101711622, 0.15729555, 0.0442431765, 1.0930037, 100.687748), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-LAF21", coefficents=(1.87134529, 0.0093332228, 0.25078301, 0.0345637762, 1.22048639, 83.2404866), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-LAF33", coefficents=(1.79653417, 0.00927313493, 0.311577903, 0.0358201181, 1.15981863, 87.3448712), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="N-LAF34", coefficents=(1.75836958, 0.00872810026, 0.313537785, 0.0293020832, 1.18925231, 85.1780644), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-LAF35", coefficents=(1.51697436, 0.00750943203, 0.455875464, 0.0260046715, 1.07469242, 80.5945159), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-LAF7", coefficents=(1.74028764, 0.010792558, 0.226710554, 0.0538626639, 1.32525548, 106.268665), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-LAK10", coefficents=(1.72878017, 0.00886014635, 0.169257825, 0.0363416509, 1.19386956, 82.9009069), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-LAK12", coefficents=(1.17365704, 0.00577031797, 0.588992398, 0.0200401678, 0.978014394, 95.4873482), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="N-LAK14", coefficents=(1.50781212, 0.00746098727, 0.318866829, 0.0242024834, 1.14287213, 80.9565165), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-LAK21", coefficents=(1.22718116, 0.00602075682, 0.420783743, 0.0196862889, 1.01284843, 88.4370099), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="N-LAK22", coefficents=(1.14229781, 0.00585778594, 0.535138441, 0.0198546147, 1.04088385, 100.834017), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-LAK33B", coefficents=(1.42288601, 0.00670283452, 0.593661336, 0.021941621, 1.1613526, 80.7407701), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="N-LAK34", coefficents=(1.26661442, 0.00589278062, 0.665919318, 0.0197509041, 1.1249612, 78.8894174), wavelength_range=(0.29, 2.5)),
    Sellmeier1(name="N-LAK7", coefficents=(1.23679889, 0.00610105538, 0.445051837, 0.0201388334, 1.01745888, 90.638038), wavelength_range=(0.29, 2.5)),
    Sellmeier1(name="N-LAK8", coefficents=(1.33183167, 0.00620023871, 0.546623206, 0.0216465439, 1.19084015, 82.5827736), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-LAK9", coefficents=(1.46231905, 0.00724270156, 0.344399589, 0.0243353131, 1.15508372, 85.4686868), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="N-LASF31A", coefficents=(1.96485075, 0.00982060155, 0.475231259, 0.0344713438, 1.48360109, 110.739863), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-LASF40", coefficents=(1.98550331, 0.010958331, 0.274057042, 0.0474551603, 1.28945661, 96.9085286), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-LASF41", coefficents=(1.86348331, 0.00910368219, 0.413307255, 0.0339247268, 1.35784815, 93.3580595), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-LASF43", coefficents=(1.93502827, 0.0104001413, 0.23662935, 0.0447505292, 1.26291344, 87.437569), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-LASF44", coefficents=(1.78897105, 0.00872506277, 0.38675867, 0.0308085023, 1.30506243, 92.7743824), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-LASF45", coefficents=(1.87140198, 0.011217192, 0.267777879, 0.0505134972, 1.73030008, 147.106505), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-LASF46A", coefficents=(2.16701566, 0.0123595524, 0.319812761, 0.0560610282, 1.66004486, 107.047718), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-LASF46B", coefficents=(2.17988922, 0.0125805384, 0.306495184, 0.0567191367, 1.56882437, 105.316538), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-LASF9", coefficents=(2.00029547, 0.0121426017, 0.298926886, 0.0538736236, 1.80691843, 156.530829), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-PK51", coefficents=(1.15610775, 0.00585597402, 0.153229344, 0.0194072416, 0.785618966, 140.537046), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="N-PK52A", coefficents=(1.029607, 0.00516800155, 0.1880506, 0.0166658798, 0.736488165, 138.964129), wavelength_range=(0.29, 2.5)),
    Sellmeier1(name="N-PSK3", coefficents=(0.88727211, 0.00469824067, 0.489592425, 0.0161818463, 1.04865296, 104.374975), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="N-PSK53A", coefficents=(1.38121836, 0.00706416337, 0.196745645, 0.0233251345, 0.886089205, 97.4847345), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="N-SF1", coefficents=(1.60865158, 0.0119654879, 0.237725916, 0.0590589722, 1.51530653, 135.521676), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-SF10", coefficents=(1.62153902, 0.0122241457, 0.256287842, 0.0595736775, 1.64447552, 147.468793), wavelength_range=(0.38, 2.5)),
    Sellmeier1(name="N-SF11", coefficents=(1.73759695, 0.013188707, 0.313747346, 0.0623068142, 1.89878101, 155.23629), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-SF14", coefficents=(1.69022361, 0.0130512113, 0.288870052, 0.061369188, 1.7045187, 149.517689), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-SF15", coefficents=(1.57055634, 0.0116507014, 0.218987094, 0.0597856897, 1.50824017, 132.709339), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-SF2", coefficents=(1.47343127, 0.0109019098, 0.163681849, 0.0585683687, 1.36920899, 127.404933), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-SF4", coefficents=(1.67780282, 0.012679345, 0.282849893, 0.0602038419, 1.63539276, 145.760496), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="N-SF5", coefficents=(1.52481889, 0.011254756, 0.187085527, 0.0588995392, 1.42729015, 129.141675), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-SF57", coefficents=(1.87543831, 0.0141749518, 0.37375749, 0.0640509927, 2.30001797, 177.389795), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-SF6", coefficents=(1.77931763, 0.0133714182, 0.338149866, 0.0617533621, 2.08734474, 174.01759), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-SF66", coefficents=(2.0245976, 0.0147053225, 0.470187196, 0.0692998276, 2.59970433, 161.817601), wavelength_range=(0.39, 2.5)),
    Sellmeier1(name="N-SF8", coefficents=(1.55075812, 0.0114338344, 0.209816918, 0.0582725652, 1.46205491, 133.24165), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-SK11", coefficents=(1.17963631, 0.00680282081, 0.229817295, 0.0219737205, 0.935789652, 101.513232), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="N-SK14", coefficents=(0.936155374, 0.00461716525, 0.594052018, 0.016885927, 1.04374583, 103.736265), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-SK16", coefficents=(1.34317774, 0.00704687339, 0.241144399, 0.0229005, 0.994317969, 92.7508526), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-SK2", coefficents=(1.28189012, 0.0072719164, 0.257738258, 0.0242823527, 0.96818604, 110.377773), wavelength_range=(0.31, 2.5)),
    Sellmeier1(name="N-SK4", coefficents=(1.32993741, 0.00716874107, 0.228542996, 0.0246455892, 0.988465211, 100.886364), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="N-SK5", coefficents=(0.991463823, 0.00522730467, 0.495982121, 0.0172733646, 0.987393925, 98.3594579), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="N-SSK2", coefficents=(1.4306027, 0.00823982975, 0.153150554, 0.0333736841, 1.01390904, 106.870822), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-SSK5", coefficents=(1.59222659, 0.00920284626, 0.103520774, 0.0423530072, 1.05174016, 106.927374), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-SSK8", coefficents=(1.44857867, 0.00869310149, 0.117965926, 0.0421566593, 1.06937528, 111.300666), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="N-ZK7", coefficents=(1.07715032, 0.00676601657, 0.168079109, 0.0230642817, 0.851889892, 89.0498778), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="P-LAF37", coefficents=(1.76003244, 0.00938006396, 0.248286745, 0.0360537464, 1.15935122, 86.4324693), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="P-LAK35", coefficents=(1.3932426, 0.00715959695, 0.418882766, 0.0233637446, 1.043807, 88.3284426), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="P-LASF47", coefficents=(1.85543101, 0.0100328203, 0.315854649, 0.0387095168, 1.28561839, 94.5421507), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="P-LASF50", coefficents=(1.84910553, 0.00999234757, 0.329828674, 0.0387437988, 1.30400901, 95.8967681), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="P-LASF51", coefficents=(1.84568806, 0.00988495571, 0.3390016, 0.0378097402, 1.32418921, 97.841543), wavelength_range=(0.334, 2.5)),
    Sellmeier1(name="P-SF68", coefficents=(2.3330067, 0.0168838419, 0.452961396, 0.0716086325, 1.25172339, 118.707479), wavelength_range=(0.42, 2.5)),
    Sellmeier1(name="P-SF69", coefficents=(1.62594647, 0.0121696677, 0.235927609, 0.0600710405, 1.67434623, 145.651908), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="P-SF8", coefficents=(1.55370411, 0.011658267, 0.206332561, 0.0582087757, 1.39708831, 130.748028), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="P-SK57", coefficents=(1.31053414, 0.00740877235, 0.169376189, 0.0254563489, 1.10987714, 107.751087), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="P-SK58A", coefficents=(1.3167841, 0.00720717498, 0.171154756, 0.0245659595, 1.12501473, 102.739728), wavelength_range=(0.32, 2.5)),
    Sellmeier1(name="P-SK60", coefficents=(1.40790442, 0.00784382378, 0.143381417, 0.0287769365, 1.16513947, 105.373397), wavelength_range=(0.3, 2.5)),
    Sellmeier1(name="SF1", coefficents=(1.55912923, 0.0121481001, 0.284246288, 0.0534549042, 0.968842926, 112.174809), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="SF10", coefficents=(1.61625977, 0.0127534559, 0.259229334, 0.0581983954, 1.07762317, 116.60768), wavelength_range=(0.38, 2.5)),
    Sellmeier1(name="SF2", coefficents=(1.40301821, 0.0105795466, 0.231767504, 0.0493226978, 0.939056586, 112.405955), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="SF4", coefficents=(1.61957826, 0.0125502104, 0.339493189, 0.0544559822, 1.02566931, 117.652222), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="SF5", coefficents=(1.46141885, 0.0111826126, 0.247713019, 0.0508594669, 0.949995832, 112.041888), wavelength_range=(0.35, 2.5)),
    Sellmeier1(name="SF56A", coefficents=(1.70579259, 0.0133874699, 0.344223052, 0.0579561608, 1.09601828, 121.616024), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="SF57", coefficents=(1.81651371, 0.0143704198, 0.428893641, 0.0592801172, 1.07186278, 121.419942), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="SF6", coefficents=(1.72448482, 0.0134871947, 0.390104889, 0.0569318095, 1.04572858, 118.557185), wavelength_range=(0.365, 2.5)),
    Sellmeier1(name="SF11", coefficents=(1.73848403, 0.0136068604, 0.311168974, 0.0615960463, 1.17490871, 121.922711), wavelength_range=(0.39, 2.5)),
    Sellmeier1(name="LASF35", coefficents=(2.45505861, 0.0135670404, 0.453006077, 0.054580302, 2.3851308, 167.904715), wavelength_range=(0.37, 2.5)),
    Sellmeier1(name="N-FK58", coefficents=(0.738042712, 0.00339065607, 0.363371967, 0.0117551189, 0.989296264, 212.842145), wavelength_range=(0.26, 2.5))
)
