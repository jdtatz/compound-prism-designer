from __future__ import annotations

from typing import Any, ClassVar, Optional, Sequence, Tuple

import numpy as np


class RayTraceError(Exception):
    pass

class Glass:
    ORDER: ClassVar[int]

    name: str
    def __new__(cls, name: str, coefficents: np.ndarray) -> Glass: ...
    def __call__(self, wavelength: float) -> float: ...

class DesignFitness:
    size: float
    info: float
    deviation: float
    def __new__(cls, size: float, info: float, deviation: float) -> DesignFitness: ...
    def __getnewargs__(self) -> Tuple[Any, ...]: ...

class CompoundPrism:
    glasses: Sequence[Glass]
    angles: Sequence[float]
    lengths: Sequence[float]
    curvature: float
    height: float
    width: float
    ar_coated: bool
    def __new__(
        cls,
        glasses: Sequence[Glass],
        angles: Sequence[float],
        lengths: Sequence[float],
        curvature: float,
        height: float,
        width: float,
        ar_coated: bool,
    ) -> CompoundPrism: ...
    def __getnewargs__(self) -> Tuple[Any, ...]: ...
    def polygons(
        self,
    ) -> Tuple[Sequence[np.ndarray], np.ndarray, np.ndarray, np.ndarray]: ...

class DetectorArray:
    bin_count: int
    bin_size: float
    linear_slope: float
    linear_intercept: float
    length: float
    max_incident_angle: float
    angle: float
    def __new__(
        cls,
        bin_count: int,
        bin_size: float,
        linear_slope: float,
        linear_intercept: float,
        length: float,
        max_incident_angle: float,
        angle: float,
    ) -> DetectorArray: ...
    def __getnewargs__(self) -> Tuple[Any, ...]: ...

class GaussianBeam:
    wavelength_range: Tuple[float, float]
    width: float
    y_mean: float
    def __new__(
        cls, wavelength_range: Tuple[float, float], width: float, y_mean: float
    ) -> GaussianBeam: ...
    def __getnewargs__(self) -> Tuple[Any, ...]: ...

class Spectrometer:
    compound_prism: CompoundPrism
    detector_array: DetectorArray
    gaussian_beam: GaussianBeam
    def __new__(
        cls,
        compound_prism: CompoundPrism,
        detector_array: DetectorArray,
        gaussian_beam: GaussianBeam,
    ) -> Spectrometer: ...
    def __getnewargs__(self) -> Tuple[Any, ...]: ...
    @property
    def position(self) -> Tuple[float, float]: ...
    @property
    def direction(self) -> Tuple[float, float]: ...
    def transmission_probability(self, wavelengths: np.ndarray, max_m: int = ...): ...
    def ray_trace(self, wavelength: float, inital_y: float) -> np.ndarray: ...
    def cpu_fitness(self, max_n: int = ..., max_m: int = ...) -> DesignFitness: ...
    def gpu_fitness(
        self, seeds: np.ndarray, max_n: int = ..., nwarp: int = ..., max_eval: int = ...
    ) -> Optional[DesignFitness]: ...
    def slow_gpu_fitness(
        self, max_n: int = ..., nwarp: int = ..., max_eval: int = ...
    ) -> Optional[DesignFitness]: ...
