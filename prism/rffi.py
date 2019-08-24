from prism._native import ffi, lib
import typing
import dataclasses
import numpy as np


class RayTraceError(Exception):
    pass


class CatalogError(Exception):
    pass


def _from_ffi_str(ffi_str: ffi.CData) -> str:
    return bytes(ffi.unpack(ffi_str.str_ptr, ffi_str.str_len)).decode("utf-8")


def _from_ffi_result(ffi_result: ffi.CData, exception_type=Exception):
    if ffi_result.succeeded is False:
        raise exception_type(_from_ffi_str(ffi_result.err_str))


class PyGlass:
    def __init__(self, name: str, ptr: ffi.CData):
        self.name = name
        self._ptr = ffi.gc(ptr, lib.free_glass)

    def __getstate__(self):
        serialized_bytes = []
        @ffi.callback("void(void*, uint8_t)")
        def append(_state, byte):
            serialized_bytes.append(byte)
        _from_ffi_result(lib.serialize_glass(self._ptr, append, ffi.NULL))
        return {"name": self.name, "serialized": bytes(serialized_bytes)}

    def __setstate__(self, state):
        self.name = state["name"]
        glass_ptr = ffi.new("Glass**")
        _from_ffi_result(lib.deserialize_glass(glass_ptr, state["serialized"], len(state["serialized"])))
        self._ptr = ffi.gc(glass_ptr[0], lib.free_glass)


def create_glass_catalog(file_contents: str) -> typing.Mapping[str, PyGlass]:
    catalog = {}
    @ffi.callback("void(void*, FFiStr, Glass*)")
    def update(_state, ffi_str, glass):
        name = _from_ffi_str(ffi_str)
        catalog[name] = PyGlass(name, glass)
    file_contents = file_contents.encode("UTF-8")
    _from_ffi_result(lib.update_glass_catalog(file_contents, len(file_contents), update, ffi.NULL), CatalogError)
    return catalog


@dataclasses.dataclass
class CompoundPrism:
    glasses: typing.Sequence[PyGlass]
    angles: np.ndarray
    lengths: np.ndarray
    curvature: float
    height: float
    width: float
    cffi_ptr: ffi.CData = dataclasses.field(init=False)

    def __post_init__(self):
        assert len(self.glasses) + 1 == len(self.angles)
        assert len(self.glasses) == len(self.lengths)
        assert self.angles.dtype == np.float64
        assert self.lengths.dtype == np.float64
        assert self.angles.ndim == 1
        assert self.lengths.ndim == 1
        assert 0 <= self.curvature <= 1
        assert 0 < self.height
        assert 0 < self.width
        ptr = lib.create_compound_prism(
            len(self.glasses),
            [g._ptr for g in self.glasses],
            ffi.cast("double *", self.angles.ctypes.data),
            ffi.cast("double *", self.lengths.ctypes.data),
            self.curvature,
            self.height,
            self.width,
        )
        self.cffi_ptr = ffi.gc(ptr, lib.free_compound_prism)


@dataclasses.dataclass
class DetectorArray:
    bins: np.ndarray
    min_ci: float
    angle: float
    length: float
    cffi_ptr: ffi.CData = dataclasses.field(init=False)

    def __post_init__(self):
        assert self.bins.dtype == np.float64
        assert self.bins.ndim == 2
        assert self.bins.shape[1] == 2
        assert 0 <= self.min_ci <= 1
        assert -np.pi <= self.angle <= np.pi
        assert 0 < self.length

        ptr = lib.create_detector_array(
            len(self.bins),
            ffi.cast("double(*)[2]", self.bins.ctypes.data),
            self.min_ci,
            self.angle,
            self.length,
        )
        self.cffi_ptr = ffi.gc(ptr, lib.free_detector_array)


@dataclasses.dataclass
class GaussianBeam:
    width: float
    y_mean: float
    wavelength_range: typing.Tuple[float, float]
    cffi_ptr: ffi.CData = dataclasses.field(init=False)

    def __post_init__(self):
        assert 0 < self.width
        assert 0 <= self.y_mean
        assert 0 <= self.wavelength_range[0] < self.wavelength_range[1]

        ptr = lib.create_gaussian_beam(
            self.width,
            self.y_mean,
            self.wavelength_range[0],
            self.wavelength_range[1],
        )
        self.cffi_ptr = ffi.gc(ptr, lib.free_gaussian_beam)


@dataclasses.dataclass(order=True)
class DesignFitness:
    size: float
    info: float
    deviation: float


def fitness(cmpnd_prism: CompoundPrism, detector_array: DetectorArray, gaussian_beam: GaussianBeam) -> DesignFitness:
    ptr = ffi.new("DesignFitness*")
    _from_ffi_result(lib.fitness(
        cmpnd_prism.cffi_ptr,
        detector_array.cffi_ptr,
        gaussian_beam.cffi_ptr,
        ptr,
    ), RayTraceError)
    return DesignFitness(size=ptr.size, info=ptr.info, deviation=ptr.deviation)


def detector_array_position(cmpnd_prism: CompoundPrism, detector_array: DetectorArray, gaussian_beam: GaussianBeam) -> typing.Tuple[np.ndarray, np.ndarray]:
    ptr = ffi.new("DetectorArrayPositioning*")
    _from_ffi_result(lib.detector_array_position(
        cmpnd_prism.cffi_ptr,
        detector_array.cffi_ptr,
        gaussian_beam.cffi_ptr,
        ptr,
    ), RayTraceError)
    return np.array((ptr.position.x, ptr.position.y)), np.array((ptr.direction.x, ptr.direction.y))


def trace(wavelength: float, inital_y: float, cmpnd_prism: CompoundPrism, detector_array: DetectorArray, detpos: typing.Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    ptr = ffi.new("DetectorArrayPositioning*")
    ptr.position.x = detpos[0][0]
    ptr.position.y = detpos[0][1]
    ptr.direction.x = detpos[1][0]
    ptr.direction.y = detpos[1][1]
    array = []
    @ffi.callback("void(void*, Pair)")
    def append_next_ray_position(_state, pos):
        array.append((pos.x, pos.y))
    _from_ffi_result(lib.trace(
        wavelength,
        inital_y,
        cmpnd_prism.cffi_ptr,
        detector_array.cffi_ptr,
        ptr,
        append_next_ray_position,
        ffi.NULL,
    ), RayTraceError)
    return np.array(array)


def p_dets_l_wavelength(wavelength: float, cmpnd_prism: CompoundPrism, detector_array: DetectorArray, gaussian_beam: GaussianBeam, detpos: typing.Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    ptr = ffi.new("DetectorArrayPositioning*")
    ptr.position.x = detpos[0][0]
    ptr.position.y = detpos[0][1]
    ptr.direction.x = detpos[1][0]
    ptr.direction.y = detpos[1][1]
    array = []
    @ffi.callback("void(void*, double)")
    def append_next_detector_probability(_state, p):
        array.append(p)
    lib.p_dets_l_wavelength(
        wavelength,
        cmpnd_prism.cffi_ptr,
        detector_array.cffi_ptr,
        gaussian_beam.cffi_ptr,
        ptr,
        append_next_detector_probability,
        ffi.NULL,
    )
    return np.array(array)
