import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functools import partial
from compound_prism_designer import config_from_toml, RayTraceError, CompoundPrism, DetectorArray, GaussianBeam, Design, BUNDLED_CATALOG
from platypus import *
import pymoo.model.problem
import pymoo.algorithms.nsga2
import pymoo.optimize
from pymoo.factory import get_sampling, get_crossover, get_mutation, get_termination
from pymoo.visualization.scatter import Scatter
import pymoo.performance_indicator.hv


with open("design_config.toml") as f:
    contents = f.read()

# config = config_from_toml(contents)
# nbin = config.detector_array.bin_count
nbin = 32
max_info = np.log2(nbin)
max_size = 300
max_fit = max_size, 1, 1

gnames = "N-SF11", "N-SSK2"
nprism = len(gnames)
glasses = [pg for n in gnames for pg in BUNDLED_CATALOG if pg.name == n]
ginds = [i for n in gnames for i, pg in enumerate(BUNDLED_CATALOG) if pg.name == n]
is_arcoated = True
design_dtype = np.dtype(list({
    # "glass_findicies": (np.float64, nprism),
    "height": np.float64,
    "angles": (np.float64, nprism + 1),
    "extension_lengths": (np.float64, nprism),
    "curvature": np.float64,
    "normalized_y_mean": np.float64,
    "det_arr_angle": np.float64,
}.items()))

bounds = np.array([
    (1e-3, 25),
    *([(-np.pi / 2, np.pi / 2)] * (1 + nprism)),
    *([(0, 25)] * nprism),
    (1e-3, 1),
    (0, 1),
    (-np.pi, np.pi)
], dtype=np.float64)
lb, ub = bounds.T


def fitness(params: np.ndarray):
    params = params.view(design_dtype)[0]
    cmpnd = CompoundPrism(
        glasses=glasses,
        angles=params["angles"],
        lengths=params["extension_lengths"],
        curvature=float(params["curvature"]),
        height=float(params["height"]),
        width=7,
        ar_coated=True
    )
    detarr = DetectorArray(
        bin_count=32,
        bin_size=0.8,
        linear_slope=1,
        linear_intercept=0.1,
        position=(np.nan, np.nan),
        direction=(np.nan, np.nan),
        length=32,
        max_incident_angle=45,
        angle=params["det_arr_angle"]
    )
    beam = GaussianBeam(
        wavelength_range=(0.5, 0.82),
        width=3.2,
        y_mean=params["normalized_y_mean"] * params["height"]
    )
    try:
        design = Design(
            compound_prism=cmpnd,
            detector_array=detarr,
            gaussian_beam=beam,
        )
        fit = design.fitness
        info_loss = (max_info - fit.info) / max_info
        if fit.size <= max_size and info_loss < 0.95:
            return (fit.size, info_loss, fit.deviation), 0
    except RayTraceError:
        pass
    return max_fit, 1


x0 = lb + (ub - lb) * 0.5
mini = opt.minimize(lambda x: fitness(x)[0][1], x0=x0, bounds=bounds)
# mini = opt.minimize(lambda x: soln_eval(x)[0][1], x0=x0, bounds=bounds)
print(mini)
print(fitness(mini.x))

mini = opt.differential_evolution(lambda x: fitness(x)[0][1], bounds=bounds)
print(mini)
print(fitness(mini.x))

mini = opt.shgo(lambda x: fitness(x)[0][1], bounds=bounds)
print(mini)
print(fitness(mini.x))

mini = opt.dual_annealing(lambda x: fitness(x)[0][1], bounds=bounds)
print(mini)
print(fitness(mini.x))

exit(0)

class SpectrometerDesignMoo(pymoo.model.problem.Problem):

    def __init__(self):
        super().__init__(n_var=len(lb),
                         n_obj=3,
                         n_constr=1,
                         xl=lb,
                         xu=ub,
                         elementwise_evaluation=True,
                         )

    def _evaluate(self, x, out, **kwargs):
        assert np.all(np.logical_and(lb <= x, x < ub))
        f, g = fitness(x)
        out['F'], out['G'] = f, g
        if out['G'] == 0:
            print(x.view(design_dtype), out)


problem = SpectrometerDesignMoo()
algorithm = pymoo.algorithms.nsga2.NSGA2(
    pop_size=10000,
    n_offsprings=100,
    # sampling=get_sampling("real_lhs"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", prob=0.05, eta=12),
    eliminate_duplicates=True
)
termination = get_termination("n_gen", 400)
res = pymoo.optimize.minimize(problem,
                              algorithm,
                              termination,
                              seed=982734398,
                              pf=problem.pareto_front(use_cache=False),
                              save_history=True,
                              verbose=True)
print(res)

'''
metric = pymoo.performance_indicator.hv.Hypervolume(ref_point=max_fit)
# collect the population in each generation
pop_each_gen = [a.pop for a in res.history]

# receive the population in each generation
obj_and_feasible_each_gen = [pop[pop.get("feasible")[:, 0]].get("F") for pop in pop_each_gen]

# calculate for each generation the HV metric
hv = [metric.calc(f) for f in obj_and_feasible_each_gen]

# visualze the convergence curve
plt.plot(np.arange(len(hv)), hv, '-o')
plt.title("Convergence")
plt.xlabel("Generation")
plt.ylabel("Hypervolume")
plt.show()
'''

# get the pareto-set and pareto-front for plotting
ps = problem.pareto_set(use_cache=False, flatten=False)
pf = problem.pareto_front(use_cache=False, flatten=False)
print(pf)

print(res.X, ps)
# Design Space
'''
plot = Scatter(title = "Design Space", axis_labels="x")
plot.add(res.X, s=30, facecolors='none', edgecolors='r')
plot.add(ps, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.show()
'''

# Objective Space
plot = Scatter(title = "Objective Space")
plot.add(res.F)
plot.add(pf, plot_type="line", color="black", alpha=0.7)
plot.do()
plot.show()

exit(0)


class SpectrometerDesign(Problem):

    def __init__(self):
        bounds = np.array(config.param_bounds())
        nvar = len(bounds)
        nobj = 3
        ncon = 1
        super().__init__(nvar, nobj, ncon)
        self.types[:] = [Real(l, u) for l, u in bounds]
        self.directions[:] = Problem.MINIMIZE, Problem.MINIMIZE, Problem.MINIMIZE
        self.constraints[:] = "<=0"

    def evaluate(self, solution):
        solution.objectives[:], solution.constraints[:] = soln_eval(solution.variables)


problem = SpectrometerDesign()

epsilons = 2.5, 0.005, 0.1

algorithms = [NSGAII,
              (NSGAIII, {"divisions_outer": 12}),
              (EpsNSGAII, {"epsilons": epsilons}),
              # (CMAES, {"epsilons": epsilons}),
              # GDE3,
              # IBEA,
              # (MOEAD, {"weight_generator": normal_boundary_weights, "divisions_outer": 12}),
              (OMOPSO, {"epsilons": epsilons}),
              # SMPSO,
              # SPEA2,
              (EpsMOEA, {"epsilons": epsilons})]

hyp = Hypervolume(minimum=[0, 0, 0], maximum=[30 * config.compound_prism.max_height, 1, 1])

results = {}
for algo in algorithms:
    print(algo)
    if isinstance(algo, tuple):
        algo, kwargs = algo
        algo = partial(algo, **kwargs)
    algo = algo(problem)
    algo.run(10000)
    feasible = [s for s in algo.result if s.feasible]
    results[algo.__class__.__name__] = {"SpectrometerDesign": [feasible]}
    print(len(feasible))
    if len(feasible) > 0:
        print(min(tuple(s.objectives)[1] for s in feasible))
        print(hyp.calculate(feasible))

hyp_result = calculate(results, hyp)
display(hyp_result, ndigits=3)

# display the results
fig = plt.figure()

for i, algorithm in enumerate((results).keys()):
    result = results[algorithm]["SpectrometerDesign"][0]

    ax = fig.add_subplot(3, 2, i + 1)
    ax.set_title(algorithm)
    sc = ax.scatter([s.objectives[0] for s in result],
                    [s.objectives[1] for s in result],
                    c=[s.objectives[2] for s in result])
    fig.colorbar(sc)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    # ax.set_xlim([0, 1.1])
    # ax.set_ylim([0, 1.1])
    # ax.set_zlim([0, 1.1])
    # ax.view_init(elev=30.0, azim=15.0)
    # ax.locator_params(nbins=4)
plt.show()
