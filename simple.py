import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import compound_prism_designer
from compound_prism_designer import config_from_toml, RayTraceError
from platypus import *

with open("design_config.toml") as f:
    contents = f.read()

config = config_from_toml(contents)
# nbin = config.detector_array.bin_count
nbin = 32


class SpectrometerDesign(Problem):

    def __init__(self):
        bounds = np.array(config.param_bounds())
        nvar = len(bounds)
        nobj = 3
        ncon = 1
        super().__init__(nvar, nobj, ncon)
        self.types[:] = [Real(l, u) for l, u in bounds]
        self.directions[:] = Problem.MINIMIZE, Problem.MAXIMIZE, Problem.MINIMIZE
        self.constraints[:] = "<=0"

    def evaluate(self, solution):
        try:
            fit = config.param_fitness(np.asanyarray(solution.variables))
            solution.objectives[:] = fit.size, fit.info, fit.deviation
            solution.constraints[:] = 0 if fit.size <= 30 * config.compound_prism.max_height and fit.info > 0.2 else 1
        except RayTraceError:
            solution.objectives[:] = 30 * config.compound_prism.max_height, np.log2(nbin), 1
            solution.constraints[:] = 1,


problem = SpectrometerDesign()

algorithms = [NSGAII,
              # (NSGAIII, {"divisions_outer":12}),
              (CMAES, {"epsilons": [0.05]}),
              GDE3,
              IBEA,
              (MOEAD, {"weight_generator": normal_boundary_weights, "divisions_outer": 12}),
              (OMOPSO, {"epsilons": [0.05]}),
              SMPSO,
              SPEA2,
              (EpsMOEA, {"epsilons": [0.05]})]


algo = NSGAII(problem)
print(algo.run(10000))
print([s for s in algo.result if s.feasible])


results = experiment(algorithms, problem, seeds=1, nfe=100)

print(results)

hyp = Hypervolume(minimum=[0, 0, 0], maximum=[30 * config.compound_prism.max_height, np.log2(nbin), 1])

hyp_result = calculate(results, hyp, evaluator=evaluator)
display(hyp_result, ndigits=3)

# display the results
fig = plt.figure()

for i, algorithm in enumerate((results).keys()):
    result = results[algorithm]["SpectrometerDesign"][0]

    ax = fig.add_subplot(2, 5, i + 1, projection='3d')
    ax.scatter([s.objectives[0] for s in result],
               [s.objectives[1] for s in result],
               [s.objectives[2] for s in result])
    ax.set_title(algorithm)
    # ax.set_xlim([0, 1.1])
    # ax.set_ylim([0, 1.1])
    # ax.set_zlim([0, 1.1])
    ax.view_init(elev=30.0, azim=15.0)
    ax.locator_params(nbins=4)

plt.show()
