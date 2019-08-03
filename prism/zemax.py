import numpy as np
from itertools import count


def create_zmx(config, params, ytans, thickness, lens_radius, chord, detarr_offset):
    incrementer = count(1)
    waves = np.linspace(config.wmin, config.wmax, 24)
    waves = "\n".join(f"WAVM {1 + i} {w} 1" for i, w in enumerate(waves))
    newline = "\n"
    return f"""\
VERS 181119 693 105780 L105780
MODE SEQ
NAME 
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
ENPD {config.beam_width}
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
{create_coord_break(next(incrementer), decenter_y=config.prism_height / 2 - params.y_mean)}
{f"{newline}".join(create_tilt(next(incrementer), t, config.prism_height / 2, glass=g, y_tangent=y) for i, (g, y, t) in enumerate(zip(params.glass_names, ytans, thickness)))}
{create_coord_break(next(incrementer), tilt_about_x=-np.rad2deg(params.thetas[-1]))}
{create_biconic(next(incrementer), radius=-lens_radius, semi_diameter=chord / 2)}
{create_coord_break(next(incrementer), tilt_about_x=np.rad2deg(params.thetas[-1]))}
{create_coord_break(next(incrementer), thickness=detarr_offset[0], decenter_y=detarr_offset[1])}
{create_coord_break(next(incrementer), tilt_about_x=-np.rad2deg(params.det_arr_angle))}
SURF {next(incrementer)}
  TYPE STANDARD
  FIMP 

  CURV 0.0 0 0 0 0 ""
  HIDE 0 0 0 0 0 0 0 0 0 0
  MIRR 2 0
  SLAB 0
  DISZ 0
  DIAM {config.det_arr_length / 2} 1 0 0 1 ""
  MEMA {config.det_arr_length / 2} 0 0 0 1 ""
  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0
BLNK
BLNK 
TOL TOFF   0   0 0.0000000000000000E+00 0.0000000000000000E+00   0 0 0 0 0
MNUM 1 1
MOFF   0   1 "" 0 0 0 1 1 0 0.0 "" 0
"""


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
        "TILTSURF",
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
  DIAM {semi_diameter} 0 0 0 1 ""
  MEMA {semi_diameter} 0 0 0 1 ""
  POPS 0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0{flap}\
"""
    return surface