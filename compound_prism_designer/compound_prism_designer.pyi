from __future__ import annotations
from typing import NamedTuple, Optional, Sequence, Tuple
import numpy as np


class GlassCatalogError(Exception):
    pass

class RayTraceError(Exception):
    pass

class Glass(NamedTuple):
    name: str
    def __call__(self, wavelength: float) -> float: ...

BUNDLED_CATALOG: Sequence[Glass] = ...

def create_glass_catalog(str) -> Sequence[Glass]: ...

class DesignFitness(NamedTuple):
    size: float
    info: float
    deviation: float

class CompoundPrism(NamedTuple):
    glasses: Sequence[Glass]
    angles: Sequence[float]
    lengths: Sequence[float]
    curvature: float
    height: float
    width: float
    ar_coated: bool
    def polygons(
        self,
    ) -> Tuple[Sequence[np.ndarray], np.ndarray, np.ndarray, np.ndarray]: ...

class DetectorArray(NamedTuple):
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
    detector_array: DetectorArray
    gaussian_beam: GaussianBeam
    @property
    def position(self) -> Tuple[float, float]: ...
    @property
    def direction(self) -> Tuple[float, float]: ...
    def to_string(self) -> str: ...
    def transmission_probability(self, wavelengths: np.ndarray, max_m: int = ...): ...
    def ray_trace(self, wavelength: float, inital_y: float) -> np.ndarray: ...
    def cpu_fitness(self, max_n: int = ..., max_m: int = ...) -> DesignFitness: ...
    def gpu_fitness(
        self, max_n: int = ..., nwarp: int = ..., max_eval: int = ...
    ) -> Optional[DesignFitness]: ...
    def slow_gpu_fitness(
        self, max_n: int = ..., nwarp: int = ..., max_eval: int = ...
    ) -> Optional[DesignFitness]: ...
