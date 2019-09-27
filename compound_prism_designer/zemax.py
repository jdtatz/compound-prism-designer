import numpy as np
from itertools import count
from compound_prism_designer import CompoundPrism, DetectorArray, GaussianBeam


def _thickness_fn(prism: CompoundPrism):
    h = prism.height / 2
    for last, angle, l in zip(prism.angles[:-1], prism.angles[1:], prism.lengths):
        if (last >= 0) ^ (angle >= 0):
            yield (np.tan(np.abs(last)) + np.tan(np.abs(angle))) * h + l
        elif np.abs(last) > np.abs(angle):
            yield (np.tan(np.abs(last)) - np.tan(np.abs(angle))) * h + l
        else:
            yield (np.tan(np.abs(angle)) - np.tan(np.abs(last))) * h + l


def create_zmx(prism: CompoundPrism, detarr: DetectorArray, beam: GaussianBeam, detarr_offset):
    incrementer = count(1)
    wmin, wmax = beam.wavelength_range

    angles = prism.angles
    ytans = np.tan(angles)
    thickness = list(_thickness_fn(prism))

    c, s = np.cos(prism.angles[-1]), np.sin(prism.angles[-1])
    chord = prism.height / c
    # curvature = R_max / R_lens
    # R_max = chord / 2
    # R_lens = chord / (2 curvature)
    lens_radius = chord / (2 * prism.curvature)

    waves = np.linspace(wmin, wmax, 24)
    waves = "\n".join(f"WAVM {1 + i} {w} 1" for i, w in enumerate(waves))
    newline = "\n"

    zemax_design = f"""\
    Zemax Rows with Aperture pupil diameter = {beam.width} & Wavelengths from {wmin} to {wmax}: 
        Coord break: decenter y = {prism.height / 2 - beam.y_mean}
        {f"{newline}        ".join(f"Tilted: thickness = {t} material = {g} semi-diameter = {prism.height / 2} x tan = 0 y tan = {-y}" for g, y, t in zip((g.name for g in prism.glasses), ytans, thickness))}
        Coord break: tilt about x = {-np.rad2deg(prism.angles[-1])}
        Biconic: radius = {-lens_radius} semi-diameter = {chord / 2} conic = 0 x_radius = 0 x_conic = 0
        Coord break: tilt about x = {np.rad2deg(prism.angles[-1])}
        Coord break: thickness: {detarr_offset[0]} decenter y: {detarr_offset[1]}
        Coord break: tilt about x = {-np.rad2deg(detarr.angle)}
        Image (Standard): semi-diameter = {detarr.length / 2}\
    """
    zemax_file = f"""\
VERS 181119 693 105780 L105780
MODE SEQ
NAME 
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
ENPD {beam.width}
ENVD 20 1 0
GFAC 0 0
GCAT SCHOTT 
RAIM 0 0 1 1 0 0 0 0 0 1
PUSH 19.88873953731639 117.51261512638798 0 0 0 0
SDMA 0 0 0
OMMA 0 0
FTYP 0 0 1 24 0 0 0 1
ROPD 2
HYPR 1
PICB 1
XFLN 0
YFLN 0
FWGN 1
VDXN 0
VDYN 0
VCXN 0
VCYN 0
VANN 0
{waves}
PWAV 13
POLS 1 0 1 0 0 1 0
GLRS 2 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 0.001 10 9.9999999999999995e-07 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
  TYPE STANDARD
  FIMP 

  CURV 0.0 0 0 0 0 ""
  HIDE 0 0 0 0 0 0 0 0 0 0
  MIRR 2 0
  SLAB 12
  DISZ INFINITY
  DIAM 0 0 0 0 1 ""
  MEMA 0 0 0 0 1 ""
  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
{create_coord_break(next(incrementer), decenter_y=prism.height / 2 - beam.y_mean)}
{f"{newline}".join(create_tilt(next(incrementer), thickness=t, semi_diameter=prism.height / 2, glass=g, y_tangent=-y) for i, (g, y, t) in enumerate(zip((g.name for g in prism.glasses), ytans, thickness)))}
{create_coord_break(next(incrementer), tilt_about_x=-np.rad2deg(prism.angles[-1]))}
{create_biconic(next(incrementer), radius=-lens_radius, semi_diameter=chord / 2)}
{create_coord_break(next(incrementer), tilt_about_x=np.rad2deg(prism.angles[-1]))}
{create_coord_break(next(incrementer), thickness=detarr_offset[0], decenter_y=detarr_offset[1])}
{create_coord_break(next(incrementer), tilt_about_x=-np.rad2deg(detarr.angle))}
SURF {next(incrementer)}
  TYPE STANDARD
  FIMP 

  CURV 0.0 0 0 0 0 ""
  HIDE 0 0 0 0 0 0 0 0 0 0
  MIRR 2 0
  SLAB 0
  DISZ 0
  DIAM {detarr.length / 2} 1 0 0 1 ""
  MEMA {detarr.length / 2} 0 0 0 1 ""
  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
BLNK
BLNK 
TOL TOFF   0   0 0.0000000000000000E+00 0.0000000000000000E+00   0 0 0 0 0
MNUM 1 1
MOFF   0   1 "" 0 0 0 1 1 0 0.0 "" 0
"""
    return zemax_design, zemax_file


def create_standard(
        surface_number: int,
        thickness: float = 0,
        semi_diameter: float = 0,
        radius: float = 0,
        glass: str = None
):
    return new_surface(
        surface_number,
        "STANDARD",
        thickness=thickness,
        semi_diameter=semi_diameter,
        radius=radius,
        glass=glass
    )


def create_coord_break(
        surface_number: int,
        thickness: float = 0,
        decenter_x: float = 0,
        decenter_y: float = 0,
        tilt_about_x: float = 0,
        tilt_about_y: float = 0,
        tilt_about_z: float = 0,
        order: int = 0,
):
    return new_surface(
        surface_number,
        "COORDBRK",
        params=[
            decenter_x,
            decenter_y,
            tilt_about_x,
            tilt_about_y,
            tilt_about_z,
            order
        ],
        thickness=thickness
    )


def create_tilt(
        surface_number: int,
        thickness: float = 0,
        semi_diameter: float = 0,
        radius: float = 0,
        glass: str = None,
        x_tangent: float = 0,
        y_tangent: float = 0,
):
    return new_surface(
        surface_number,
        "TILTSURF",
        params=[
            x_tangent,
            y_tangent
        ],
        thickness=thickness,
        glass=glass,
        radius=radius,
        semi_diameter=semi_diameter,
    )


def create_biconic(
        surface_number: int,
        thickness: float = 0,
        semi_diameter: float = 0,
        radius: float = 0,
        glass: str = None,
        x_radius: float = 0,
):
    return new_surface(
        surface_number,
        "BICONICX",
        params=[
            x_radius,
        ],
        thickness=thickness,
        glass=glass,
        radius=radius,
        semi_diameter=semi_diameter,
    )


def new_surface(
        surface_number: int,
        surface_type: str,
        params: [float] = None,
        thickness: float = 0,
        semi_diameter: float = 0,
        radius: float = 0,
        glass: str = None
):
    if params is None:
        params = []
    params = (params + [0] * 6)[:6]
    params = "\n  ".join(f"PARM {1 + i} {v}" for i, v in enumerate(params) if i < 6)
    glass = "\n  " + f"GLAS {glass} 0 0 1.5 40 0 0 0 0 0 0" if glass is not None else ""
    flap = ("\n  " + f"FLAP 0 {semi_diameter} 0") if semi_diameter > 0 else ""
    surface = f"""\
SURF {surface_number}
  TYPE {surface_type.upper()}
  FIMP 

  CURV {radius} 0 0.0 0.0 0
  HIDE 0 0 0 0 0 0 0 0 0 0
  MIRR 2 0
  SLAB 0
  {params}
  DISZ {thickness}{glass}
  DIAM {semi_diameter} 1 0 0 1 ""
  MEMA {semi_diameter} 0 0 0 1 ""
  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0{flap}\
"""
    return surface
