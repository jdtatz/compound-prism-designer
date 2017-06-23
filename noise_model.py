import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

qe_curve = np.loadtxt("h7260_20_qe.csv", delimiter=',')
qe = InterpolatedUnivariateSpline(qe_curve[:, 0], qe_curve[:, 1] / 100)

theortical = np.load("dyes.npz")
dyes = {k: theortical[k] for k in theortical.keys()}
dye = 'AF633'
basis = InterpolatedUnivariateSpline(dyes[dye][0], dyes[dye][2] / np.max(dyes[dye][2]))


def mu(lam):
    return qe(lam)*1024*np.maximum(0, basis(lam))


def var(lam):
    return np.sqrt(2*mu(lam))


def pos(lam, amount):
    return np.random.poisson(mu(lam), (amount, lam.size)).T

waves = np.linspace(500, 820, 6)
poisson = pos(waves, 1000000)
print(np.mean(poisson, axis=1))
print(np.var(poisson, axis=1))
