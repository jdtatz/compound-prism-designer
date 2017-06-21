import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

qe_curve = np.loadtxt("h7260_20_qe.csv", delimiter=',')
qe = InterpolatedUnivariateSpline(qe_curve[:, 0], qe_curve[:, 1] / 100)

basisCount = 100
basis = InterpolatedUnivariateSpline(np.linspace(500, 820, basisCount), np.ones(basisCount))


def mu(lam):
    return qe(lam)*1024*basis(lam)


def var(lam):
    return np.sqrt(2*mu(lam))


def pos(lam):
    return np.random.poisson(mu(lam), (1000, lam.size)).T

waves = np.linspace(500, 820, 10)
print(mu(waves))
print(var(waves))
print(pos(waves))
