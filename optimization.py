from compound_prism_designer import BUNDLED_CATALOG, CompoundPrism, DetectorArray, \
    GaussianBeam, DesignFitness, Design, RayTraceError, GlassCatalogError, evaluate
import numpy as np
import scipy.optimize

catalog = BUNDLED_CATALOG
catalog_dict = {g.name: g for g in catalog}
max_nprism = 4
max_height = 20
prism_width = 7
bins = np.array([(i + 0.1, i + 0.9) for i in range(32)])
detarr_length = 32
max_incident_angle = 45
wavelength_range = 0.5, 0.82
beam_width = 3.6

param_dtype = np.dtype(list({
    'nprism': np.float64,
    'prism_height': np.float64,
    'glass_indicies': (np.float64, max_nprism),
    'angles': (np.float64, max_nprism + 1),
    'lengths': (np.float64, max_nprism),
    'curvature': np.float64,
    'normalized_y_mean': np.float64,
    'det_arr_angle': np.float64,
}.items()))

print(param_dtype)

prism_count_bounds = 1, 1 + max_nprism
prism_height_bounds = max_height * 1e-2, max_height
glass_bounds = 0, len(catalog)
angle_bounds = -np.pi / 2, np.pi / 2
length_bounds = 0, max_height
curvature_bounds = 1e-3, 1
normalized_y_mean_bounds = 0, 1
det_arr_angle_bounds = -np.pi, np.pi
bounds = np.array((
    prism_count_bounds,
    prism_height_bounds,
    *(glass_bounds for _ in range(max_nprism)),
    *(angle_bounds for _ in range(1 + max_nprism)),
    *(length_bounds for _ in range(max_nprism)),
    curvature_bounds,
    normalized_y_mean_bounds,
    det_arr_angle_bounds,
), dtype=np.float64)

lb = np.ascontiguousarray(bounds[:, 0])
ub = np.ascontiguousarray(bounds[:, 1])

print(*bounds)
print(np.ascontiguousarray(bounds[:, 1]).view(param_dtype).view(np.recarray))

print(np.array([(i + 0.1, i + 0.9) for i in range(32)]))
example_glasses = "K7", "N-LAK33B", "N-SF14", "N-KZFS2"
example = CompoundPrism(
    glasses=[catalog_dict[g] for g in example_glasses],
    angles=np.array([
        -0.23168624866497564,
        -0.15270363395467454,
        1.4031935147895553,
        -0.1992013814412539,
        -0.273581994447271
    ]),
    lengths=np.array([
        0.0,
        0.0001156759279861445,
        0.6326213941932529,
        0.014635982099371865
    ]),
    curvature=0.042844548844556525,
    height=7.853605325124353,
    width=7.0
), DetectorArray(
    bins=np.array([(i + 0.1, i + 0.9) for i in range(32)]),
    position=(177.893440699849, 27.127233556048136),
    direction=(0.7195961936369738, -0.6943927693338828),
    length=32.,
    max_incident_angle=45,
    angle=0.8032206191368956,
), GaussianBeam(
    wavelength_range=(0.5, 0.82),
    width=3.2,
    y_mean=6.687028954574839,
)
example_design = evaluate(*example)
print(example_design.fitness.size, example_design.fitness.info, example_design.fitness.deviation)


def optimizer(params: np.ndarray):
    params = params.view(param_dtype).view(np.recarray)[0]
    nprism = min(int(np.floor(params.nprism)), max_nprism)
    cmpnd = CompoundPrism(
        [catalog[min(int(np.floor(i)), len(catalog) - 1)] for i in params.glass_indicies[:nprism]],
        params.angles[:nprism],
        params.lengths[:nprism],
        params.curvature,
        params.prism_height,
        prism_width,
    )
    detarr = DetectorArray(
        bins,
        (np.nan, np.nan),  # FIXME
        (np.nan, np.nan),  # FIXME
        detarr_length,
        max_incident_angle,
        params.det_arr_angle,
    )
    beam = GaussianBeam(
        wavelength_range,
        beam_width,
        params.normalized_y_mean * params.prism_height
    )
    return evaluate(cmpnd, detarr, beam)


def eval_opt(params: np.ndarray):
    try:
        d = optimizer(params)
        return -d.fitness.info
    except RayTraceError:
        return 0


r = np.random.rand(len(lb))
p = lb + r * (ub - lb)
print(p.view(param_dtype))
# print(optimizer(p))
o = scipy.optimize.shgo(eval_opt, bounds, options={'maxfev': 10_000})
print(o)
