import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

qe_curve = np.loadtxt("h7260_20_qe.csv", delimiter=',')
qe = InterpolatedUnivariateSpline(qe_curve[:, 0], qe_curve[:, 1] / 100, k=1)

theortical = np.load("dyes.npz")
dyes = {k: theortical[k] for k in theortical.keys()}
dye = 'AF633'
basis = InterpolatedUnivariateSpline(dyes[dye][0], dyes[dye][2] / np.max(dyes[dye][2]), k=1)


def mu(lam):
    return qe(lam)*1024*basis(lam)


def var(lam):
    return np.sqrt(2*mu(lam))


def pos(lam, amount):
    mean = mu(lam)
    variance = var(lam)
    alpha = variance / np.where(mean != 0, mean, 1.0)
    return (np.random.poisson(mean / np.where(alpha != 0, alpha, 1.0), (amount, lam.size)) * alpha).T

waves = np.linspace(500, 820, 6)
poisson = pos(waves, 1000000)
print(mu(waves))
print(np.mean(poisson, axis=1))
print(var(waves))
print(np.var(poisson, axis=1))
