from __future__ import annotations

from platform import system
from typing import Iterable, Optional

import numpy as np

from .compound_prism_designer import Spectrometer, Vector2D

if system() == "Windows":
    from win32com.client import CastTo, constants
    from win32com.client.gencache import EnsureDispatch, EnsureModule


class ZemaxException(Exception):
    pass


def create_zemax_file(spec: Spectrometer, zemax_file: str):
    cmpnd = spec.compound_prism
    beam = spec.beam
    detarr = spec.detector_array

    EnsureModule("ZOSAPI_Interfaces", 0, 1, 0)
    connection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
    if connection is None:
        raise ZemaxException("Unable to intialize COM connection to ZOSAPI")
    application = connection.CreateNewApplication()
    if application is None:
        raise ZemaxException("Unable to acquire ZOSAPI application")
    if not application.IsValidLicenseForAPI:
        raise ZemaxException("License is not valid for ZOSAPI use")
    system = application.CreateNewSystem(constants.SystemType_Sequential)
    if system is None:
        raise ZemaxException("Unable to acquire Primary system")

    system.SaveAs(zemax_file)

    aperture = system.SystemData.Aperture
    aperture.ApertureType = constants.ZemaxApertureType_EntrancePupilDiameter
    aperture.ApertureValue = beam.width

    waves = system.SystemData.Wavelengths
    for i in range(waves.NumberOfWavelengths):
        waves.RemoveWavelength(1)
    for w in np.linspace(*beam.wavelength_range, 24):
        waves.AddWavelength(w, 1)

    # fields = system.SystemData.Fields

    surfaces = cmpnd.surfaces()
    assert (
        all(s.radius is None for s in surfaces[:-1]) and surfaces[-1].radius is not None
    ), "Only 2D CompoundPrism designs can be translated to Zemax"
    midpts = [np.array(((s.lower_pt.x + s.upper_pt.x) / 2, (s.lower_pt.y + s.upper_pt.y) / 2)) for s in surfaces]
    thickness = [m1[0] - m0[0] for m0, m1 in zip(midpts[:-1], midpts[1:])]
    angles = cmpnd.angles
    ytans = np.tan(angles)
    dpos = np.array(spec.position)
    ddir = np.array(spec.direction)
    detmid = dpos + ddir * detarr.length / 2
    detarr_offset = detmid - midpts[-1]

    c, s = np.cos(angles[-1]), np.sin(angles[-1])
    chord = cmpnd.height / c
    lens_radius = surfaces[-1].radius

    start = _create_standard(system)
    _create_coord_break(system, decenter_y=cmpnd.height / 2 - beam.y_mean)
    for g, y, t in zip(cmpnd.glasses, ytans, thickness):
        _create_tilt(
            system,
            thickness=t,
            semi_diameter=cmpnd.height / 2,
            glass=g.name,
            y_tangent=-y,
        )

    _create_coord_break(system, tilt_about_x=-np.rad2deg(angles[-1]))
    _create_biconic(system, radius=-lens_radius, semi_diameter=chord / 2)
    _create_coord_break(system, tilt_about_x=np.rad2deg(angles[-1]))

    _create_coord_break(system, thickness=detarr_offset[0], decenter_y=detarr_offset[1])
    _create_coord_break(system, tilt_about_x=-np.rad2deg(detarr.angle))

    last = _create_standard(system, semi_diameter=detarr.length / 2)
    # last.IsStop = True

    system.close()
    application.CloseApplication()


def _create_standard(
    system,
    thickness: float = 0,
    semi_diameter: float = 0,
    radius: float = 0,
    glass: Optional[str] = None,
):
    return _new_surface(
        system,
        "Standard",
        thickness=thickness,
        semi_diameter=semi_diameter,
        radius=radius,
        glass=glass,
    )


def _create_coord_break(
    system,
    thickness: float = 0,
    decenter_x: float = 0,
    decenter_y: float = 0,
    tilt_about_x: float = 0,
    tilt_about_y: float = 0,
    tilt_about_z: float = 0,
    order: int = 0,
):
    return _new_surface(
        system,
        "CoordinateBreak",
        params=[
            decenter_x,
            decenter_y,
            tilt_about_x,
            tilt_about_y,
            tilt_about_z,
            order,
        ],
        thickness=thickness,
    )


def _create_tilt(
    system,
    thickness: float = 0,
    semi_diameter: float = 0,
    radius: float = 0,
    glass: Optional[str] = None,
    x_tangent: float = 0,
    y_tangent: float = 0,
):
    return _new_surface(
        system,
        "Tilted",
        params=[x_tangent, y_tangent],
        thickness=thickness,
        glass=glass,
        radius=radius,
        semi_diameter=semi_diameter,
    )


def _create_biconic(
    system,
    thickness: float = 0,
    semi_diameter: float = 0,
    radius: float = 0,
    glass: Optional[str] = None,
    x_radius: float = 0,
):
    return _new_surface(
        system,
        "Biconic",
        params=[
            x_radius,
        ],
        thickness=thickness,
        glass=glass,
        radius=radius,
        semi_diameter=semi_diameter,
    )


def _new_surface(
    system,
    surface_type: str,
    params: Optional[Iterable[float]] = None,
    thickness: float = 0,
    semi_diameter: float = 0,
    radius: float = 0,
    glass: Optional[str] = None,
):
    if params is None:
        params = []

    surf = system.LDE.AddSurface()
    surface_type = getattr(constants, f"SurfaceType_{surface_type}")
    surf.ChangeType(surf.GetSurfaceTypeSettings(surface_type))
    for i, v in enumerate(params):
        surf.GetSurfaceCell(getattr(constants, f"SurfaceColumn_Par{1 + i}")).Value = v
    if thickness:
        surf.Thickness = thickness
    if semi_diameter:
        surf.SemiDiameter = semi_diameter
        surf.MechanicalSemiDiameter = semi_diameter
    if radius:
        surf.Radius = radius
    if glass:
        surf.Material = glass
    assert CastTo(surf, "IEditorRow").IsValidRow
    return surf
