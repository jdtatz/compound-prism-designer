from typing import NamedTuple, Sequence, Tuple
import numpy as np
try:
    from typing import type_check_only
except ImportError:
    type_check_only = lambda x: None


@type_check_only
class Glass(NamedTuple):
    name: str

    def __call__(self, wavelength: float) -> float:
        raise NotImplementedError


@type_check_only
class CompoundPrism(NamedTuple):
    glasses: Sequence[Glass]
    angles: Sequence[float]
    lengths: Sequence[float]
    curvature: float
    height: float
    width: float
    ar_coated: bool

    def polygons(self) -> Tuple[Sequence[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError


@type_check_only
class LinearDetectorArray(NamedTuple):
    bin_count: int
    bin_size: float
    linear_slope: float
    linear_intercept: float
    length: float
    max_incident_angle: float
    angle: float


@type_check_only
class GaussianBeam(NamedTuple):
    wavelength_range: Tuple[float, float]
    width: float
    y_mean: float


@type_check_only
class Spectrometer(NamedTuple):
    compound_prism: CompoundPrism
    detector_array: LinearDetectorArray
    gaussian_beam: GaussianBeam
    position: Tuple[float, float]
    direction: Tuple[float, float]

    def ray_trace(self, wavelength: float, starting_y: float) -> np.ndarray:
        raise NotImplementedError
