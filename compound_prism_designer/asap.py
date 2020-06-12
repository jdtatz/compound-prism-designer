import numpy as np
from .type_hints import Spectrometer


def create_asap_macro(spectrometer: Spectrometer) -> str:
    polys, lens_poly, lens_center, lens_radius = spectrometer.compound_prism.polygons()
    midpts = [(l0 + u0) / 2 for l0, u0, _, _ in polys] + [(lens_poly[0] + lens_poly[1]) / 2]
    lengths = [np.linalg.norm(u0 - l0) for l0, u0, _, _ in polys] + [np.linalg.norm(lens_poly[1] - lens_poly[0])]
    lens_midpt = (lens_poly[-2] + lens_poly[-1]) / 2
    lens_len = np.linalg.norm(lens_poly[-1] - lens_poly[-2])
    glass_names = ["AIR"] + [g.name.replace("-", "_") for g in spectrometer.compound_prism.glasses] + ["AIR"]
    planes = '\n'.join(f"""\
SURFACE
    PLANE X {midpt[0]} RECT {l/2} {spectrometer.compound_prism.width/2}
OBJECT
    ROTATE Z {np.rad2deg(angle)}
    INTERFACE COATING {"AR" if spectrometer.compound_prism.ar_coated else "BARE"} {g1} {g2}
""" for (angle, midpt, l, g1, g2) in zip(spectrometer.compound_prism.angles, midpts, lengths, glass_names[:-1], glass_names[1:]))
    det_midpt = np.asarray(spectrometer.position) + np.asarray(spectrometer.direction) * spectrometer.detector_array.length / 2
    wi, wf = spectrometer.gaussian_beam.wavelength_range
    wm = (wi + wf) / 2
    media = "\n".join(f"    {g(wi)} {g(wm)} {g(wf)} '{g.name.replace('-', '_')}'" for g in set(spectrometer.compound_prism.glasses))
    return f"""\
SYSTEM NEW
RESET
AXIS X
UNITS MM
WAVELENGTHS {wi} {wm} {wf} UM
MEDIA
{media}

COATING PROPERTIES
    0 1 0 1 0 1 'AR'
    0 0 0 0 0 0 'ASB'

{planes}

SURFACE
    BICONIC X {lens_midpt[0]} {-lens_radius} 0 0 0 RECT {lens_len/2} {spectrometer.compound_prism.width/2}                                                   
OBJECT
    ROTATE Z {np.rad2deg(spectrometer.compound_prism.angles[-1])}
    INTERFACE COATING {"AR" if spectrometer.compound_prism.ar_coated else "BARE"} {glass_names[-2]} {glass_names[-1]}

SURFACE
    PLANE X {det_midpt[0]} RECT {spectrometer.detector_array.length/2} {spectrometer.compound_prism.width/2}
OBJECT 'DETECTOR'
    ROTATE Z {np.rad2deg(spectrometer.detector_array.angle)}
    SHIFT Y {det_midpt[1] - spectrometer.compound_prism.height/2}
    INTERFACE COATING ASB AIR AIR

WAVELENGTH {wi}
GAUSSIAN X 0 0 1000 {spectrometer.gaussian_beam.width}
SHIFT Y {spectrometer.gaussian_beam.y_mean - spectrometer.compound_prism.height / 2}

WAVELENGTH {wm}
GAUSSIAN X 0 0 1000 {spectrometer.gaussian_beam.width}
SHIFT Y {spectrometer.gaussian_beam.y_mean - spectrometer.compound_prism.height / 2}

WAVELENGTH {wf}
GAUSSIAN X 0 0 1000 {spectrometer.gaussian_beam.width}
SHIFT Y {spectrometer.gaussian_beam.y_mean - spectrometer.compound_prism.height / 2}

RETURN

$IO VECTOR REWIND
$IO PLOT CANCEL
$PLOT OFF
WINDOW Y X
PLOT FACETS 5 5 OVERLAY
SELECT ALL
TRACE PLOT 1000
$VIEW
$PLOT NORM
$IO PLOT
$IO INPUT CLOSE

WINDOW Z Y
IRRADIANCE Z
AXIS LOCAL 'DETECTOR'
SPOTS POSITION EVERY 1000
"""
