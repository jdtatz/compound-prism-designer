import numpy as np
from itertools import count
# from compound_prism_designer import Spectrometer
from typing import NamedTuple, Sequence, Tuple, Optional


class Glass(NamedTuple):
    name: str


class CompoundPrism(NamedTuple):
    glasses: Sequence[Glass]
    angles: Sequence[float]
    lengths: Sequence[float]
    curvature: float
    height: float
    width: float
    ar_coated: bool


class LinearDetectorArray(NamedTuple):
    bin_count: int
    bin_size: float
    linear_slope: float
    linear_intercept: float
    length: float
    max_incident_angle: float
    angle: float


class GaussianBeam(NamedTuple):
    wavelength_range: Tuple[float, float]
    width: float
    y_mean: float


class Spectrometer(NamedTuple):
    compound_prism: CompoundPrism
    detector_array: LinearDetectorArray
    gaussian_beam: GaussianBeam
    position: Tuple[float, float]
    direction: Tuple[float, float]


def create_asap_macro(spectrometer: Spectrometer) -> str:
    polys, lens_poly, lens_center, lens_radius = spectrometer.compound_prism.polygons()
    midpts = [(l0 + u0) / 2 for l0, u0, _, _ in polys] + [(lens_poly[0] + lens_poly[1]) / 2]
    lens_midpt = (lens_poly[-2] + lens_poly[-1]) / 2
    glass_names = ["AIR"] + [g.name for g in spectrometer.compound_prism.glasses] + ["AIR"]
    planes = '\n'.join(define_plane(
        angle=angle,
        midpt=midpt,
        height=spectrometer.compound_prism.height,
        width=spectrometer.compound_prism.width,
        g1=g1,
        g2=g2,
        coated=spectrometer.compound_prism.ar_coated,
    ) for (angle, midpt, g1, g2) in zip(spectrometer.compound_prism.angles, midpts, glass_names[:-1], glass_names[1:]))
    lens = define_plane(
        angle=spectrometer.compound_prism.angles[-1],
        midpt=lens_midpt,
        height=spectrometer.compound_prism.height,
        width=spectrometer.compound_prism.width,
        g1=glass_names[-2],
        g2=glass_names[-1],
        coated=spectrometer.compound_prism.ar_coated,
        radius=lens_radius,
    )
    det_midpt = np.asarray(spectrometer.position) + np.asarray(spectrometer.direction) * spectrometer.detector_array.length / 2
    wi, wf = spectrometer.gaussian_beam.wavelength_range
    wm = (wi + wf) / 2
    media = "\n".join(f"    {g(wi)} {g(wm)} {g(wf)} '{g.name}'" for g in set(spectrometer.compound_prism.glasses))
    sample = f"""\
SYSTEM NEW
RESET
AXIS X
UNITS MM
WAVELENGTHS {wi} {wm} {wf} UM
MEDIA
{media}

COATING PROPERTIES
    0 1 'AR'
    0 0 'ABSORB'

ENT OBJECT

{planes}

{lens}

SURFACE
    RECTANGLE X {det_midpt[0]} {spectrometer.detector_array.length} {spectrometer.compound_prism.width}
OBJECT 'DETECTOR'
    SKEW X {np.rad2deg(spectrometer.detector_array.angle)} Y
    SHIFT Y {det_midpt[1]}
    INTERFACE COATING 'ABSORB'

GAUSSIAN X 0 0 1000 {spectrometer.gaussian_beam.width}

WINDOW Y X
PLOT FACETS 5 5
"""
    return sample


def define_plane(angle: float, midpt: Tuple[float, float], height: float, width: float, g1: str, g2: str, coated: bool, radius: Optional[float] = None) -> str:
    s, c = np.sin(angle), np.cos(angle)
    # R = np.array(((c, -s), (s, c)))
    # ux, uy = R @ (-1, 0)
    ux, uy = -c, -s
    if radius is not None:
        chord_length = height / c
        apothem = np.sqrt(radius**2 - (chord_length / 2)**2)
        fs = f"    SAG X 0 SPHERE {apothem}\n"
    else:
        fs = ""
    return f"""\
SURFACE
    PLANE NORMAL {ux},{uy},0 {midpt[0]} {midpt[1]} 0
    LOCAL 0 1000000 0 {height} {-width/2} {width/2}
OBJECT
{fs}\
    INTERFACE COATING {"AR" if coated else "BARE"} '{g1}' '{g2}'
"""



