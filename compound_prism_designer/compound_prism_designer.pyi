from typing import Any, ClassVar, Optional, Sequence

import numpy as np
from matplotlib.path import Path

class RayTraceError(Exception):
    pass

class CudaError(Exception):
    pass

class Glass:
    ORDER: ClassVar[int]

    name: str
    glass: list[float]
    def __new__(cls, name: str, coefficents: np.ndarray) -> Glass: ...
    def __call__(self, wavelength: float) -> float: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class Vector2D:
    x: float
    y: float
    def __new__(cls, x: float, y: float) -> Vector2D: ...
    def __iter__(self) -> Sequence[float]: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class UnitVector2D:
    x: float
    y: float
    def __new__(cls, x: float, y: float) -> UnitVector2D: ...
    def __iter__(self) -> Sequence[float]: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class Surface:
    lower_pt: Vector2D
    upper_pt: Vector2D
    angle: float
    # radius: Optional[float | tuple[float, float]]
    radius: Optional[float]
    # def __new__(
    #     cls,
    #     lower_pt: Vector2D,
    #     upper_pt: Vector2D,
    #     angle: float,
    #     radius: Optional[float | tuple[float, float]],
    # ) -> Surface: ...
    def __new__(
        cls,
        lower_pt: Vector2D,
        upper_pt: Vector2D,
        angle: float,
        radius: Optional[float],
    ) -> Surface: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class WavelengthDistribution:
    @property
    def bounds(self) -> tuple[float, float]: ...

class UniformWavelengthDistribution(WavelengthDistribution):
    bounds: tuple[float, float]
    def __new__(cls, bounds: tuple[float, float]) -> UniformWavelengthDistribution: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class BeamDistribution:
    @property
    def width(self) -> float: ...
    @property
    def y_mean(self) -> float: ...

class GaussianBeam(BeamDistribution):
    width: float
    y_mean: float
    def __new__(cls, width: float, y_mean: float) -> GaussianBeam: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class FiberBeam(BeamDistribution):
    core_radius: float
    numerical_aperture: float
    center_y: float
    def __new__(
        cls, core_radius: float, numerical_aperture: float, center_y: float
    ) -> FiberBeam: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class CompoundPrism:
    glasses: Sequence[Glass]
    angles: Sequence[float]
    lengths: Sequence[float]
    curvature: float | tuple[tuple[float, float], tuple[float, float]]
    height: float
    width: float
    ar_coated: bool
    def __new__(
        cls,
        glasses: Sequence[Glass],
        angles: Sequence[float],
        lengths: Sequence[float],
        curvature: float | tuple[tuple[float, float], tuple[float, float]],
        height: float,
        width: float,
        ar_coated: bool,
    ) -> CompoundPrism: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...
    def exit_ray(
        self, y: float, wavelength: float
    ) -> tuple[Vector2D, UnitVector2D]: ...
    def surfaces(self) -> Sequence[Surface]: ...
    def polygons(self) -> Sequence[Path]: ...

def position_detector_array(
    length: float,
    angle: float,
    compound_prism: CompoundPrism,
    wavelengths: WavelengthDistribution,
    beam: BeamDistribution,
) -> tuple[tuple[float, float], bool]: ...

class DetectorArray:
    bin_count: int
    bin_size: float
    linear_slope: float
    linear_intercept: float
    length: float
    max_incident_angle: float
    angle: float
    position: tuple[float, float]
    flipped: bool
    def __new__(
        cls,
        bin_count: int,
        bin_size: float,
        linear_slope: float,
        linear_intercept: float,
        length: float,
        max_incident_angle: float,
        angle: float,
        position: tuple[float, float],
        flipped: bool,
    ) -> DetectorArray: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class DesignFitness:
    size: float
    info: float
    deviation: float
    def __new__(cls, size: float, info: float, deviation: float) -> DesignFitness: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...

class Spectrometer:
    compound_prism: CompoundPrism
    detector_array: DetectorArray
    wavelengths: WavelengthDistribution
    beam: BeamDistribution
    def __new__(
        cls,
        compound_prism: CompoundPrism,
        detector_array: DetectorArray,
        wavelengths: WavelengthDistribution,
        beam: BeamDistribution,
    ) -> Spectrometer: ...
    def __getnewargs__(self) -> tuple[Any, ...]: ...
    @property
    def position(self) -> tuple[float, float]: ...
    @property
    def direction(self) -> tuple[float, float]: ...
    def transmission_probability(
        self, wavelengths: np.ndarray, *, max_m: int = ...
    ) -> np.ndarray: ...
    def ray_trace(self, wavelength: float, inital_y: float) -> np.ndarray: ...
    def cpu_fitness(self, max_n: int = ..., max_m: int = ...) -> DesignFitness: ...
    def gpu_fitness(
        self, seeds: np.ndarray, max_n: int = ..., nwarp: int = ..., max_eval: int = ...
    ) -> Optional[DesignFitness]: ...
