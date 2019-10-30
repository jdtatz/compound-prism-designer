import typing
from dataclasses import dataclass
import numpy as np
from .compound_prism_designer import optimize as _optimize, Glass, Design


@dataclass
class OptimizationConfig:
    iteration_count: int
    population_size: int
    offspring_size: int
    crossover_distribution_index: float
    mutation_distribution_index: float
    mutation_probability: float
    seed: int
    epsilons: typing.Tuple[float, float, float]

    @staticmethod
    def from_dict(d: typing.Dict):
        return OptimizationConfig(
            iteration_count=d['iteration-count'],
            population_size=d['population-size'],
            offspring_size=d['offspring-size'],
            crossover_distribution_index=d['crossover-distribution-index'],
            mutation_distribution_index=d['mutation-distribution-index'],
            mutation_probability=d['mutation-probability'],
            seed=d['seed'],
            epsilons=tuple(d['epsilons']),
        )


@dataclass
class CompoundPrismConfig:
    max_count: int
    max_height: float
    width: float

    @staticmethod
    def from_dict(d: typing.Dict):
        return CompoundPrismConfig(
            max_count=d['max-count'],
            max_height=d['max-height'],
            width=d['width'],
        )


@dataclass
class GaussianBeamConfig:
    width: float
    wavelength_range: typing.Tuple[float, float]

    @staticmethod
    def from_dict(d: typing.Dict):
        return GaussianBeamConfig(
            width=d['width'],
            wavelength_range=tuple(d['wavelength-range']),
        )


@dataclass
class DetectorArrayConfig:
    length: float
    max_incident_angle: float
    bin_bounds: np.ndarray

    @staticmethod
    def from_dict(d: typing.Dict):
        return DetectorArrayConfig(
            length=d['length'],
            max_incident_angle=d['max-incident-angle'],
            bin_bounds=np.array(d['bounds']),
        )


@dataclass
class DesignConfig:
    units: str
    catalog: typing.Optional[typing.List[Glass]]
    optimizer: OptimizationConfig
    compound_prism: CompoundPrismConfig
    detector_array: DetectorArrayConfig
    gaussian_beam: GaussianBeamConfig

    @staticmethod
    def from_dict(d: typing.Dict):
        return DesignConfig(
            units=d["length-unit"],
            catalog=None,
            optimizer=OptimizationConfig.from_dict(d['optimizer']),
            compound_prism=CompoundPrismConfig.from_dict(d['compound-prism']),
            detector_array=DetectorArrayConfig.from_dict(d['detector-array']),
            gaussian_beam=GaussianBeamConfig.from_dict(d['gaussian-beam']),
        )

    def optimize(self) -> Design:
        print(self)
        return _optimize(None, self)
