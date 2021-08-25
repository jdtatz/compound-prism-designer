from __future__ import annotations
import numpy as np
from typing import Optional, Iterable
from .compound_prism_designer import Spectrometer
from platform import system

if system() == "Windows":
    from win32com.client.gencache import EnsureDispatch, EnsureModule
    from win32com.client import CastTo, constants


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

    polys, lens_poly, _, _ = cmpnd.polygons()
    thickness = [
        (l1 + u1) / 2 - (l0 + u0) / 2 for (l0, u0, l1, u1) in [*polys, lens_poly]
    ]
    angles = cmpnd.angles
    ytans = np.tan(angles)
    dpos = np.array(spec.position)
    ddir = np.array(spec.direction)
    detmid = dpos + ddir * detarr.length / 2
    [_, _, l, u] = lens_poly
    lmidpt = np.array([(l + u) / 2, cmpnd.height / 2])
    detarr_offset = detmid - lmidpt

    c, s = np.cos(angles[-1]), np.sin(angles[-1])
    chord = cmpnd.height / c
    # curvature = R_max / R_lens
    # R_max = chord / 2
    # R_lens = chord / (2 curvature)
    lens_radius = chord / (2 * cmpnd.curvature)

    start = _create_standard(system)
    _create_coord_break(system, decenter_y=cmpnd.height / 2 - beam.y_mean)
    for g, y, t in zip(cmpnd.glasses, ytans, thickness):
        _create_tilt(
            system, thickness=t, semi_diameter=cmpnd.height / 2, glass=g.name, y_tangent=-y
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
