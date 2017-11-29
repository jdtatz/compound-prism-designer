from collections import namedtuple, OrderedDict
import numba as nb
import numpy as np
from compoundprism.utils import jit, get_poly_coeffs, nonlinearity, spectral_sampling_ratio, beam_compression

"""
Prism Code
"""


def create(merit, glass_indices, deltaC_target, deltaT_target, weights,
           sampling_domain, theta0, initial_angles, angle_limit, w):
    nwaves = w.size
    dw = np.abs(np.mean(np.gradient(w)))
    radius = 4.8
    start = 5
    height = 25
    sampling_domain_is_wavenumber = sampling_domain == 'wavenumber'
    glass_indices = np.asarray(glass_indices, np.int64)
    prism_count, glass_count = len(glass_indices), glass_indices.max() + 1
    if initial_angles is None:
        initial_angles = np.full((glass_count,), np.pi / 2)
        initial_angles[1::2] *= -1
    else:
        initial_angles = np.asarray(initial_angles, np.float64) * np.pi / 180
    base_weights = OrderedDict(tir=0.1, valid=1.0, crit_angle=1.0, thin=0.25, deviation=1.0, dispersion=1.0, linearity=1.0)
    MeritWeights = namedtuple('MeritWeights', base_weights.keys())
    weights = MeritWeights(**{k: v for k, v in {**base_weights, **weights}.items() if k in base_weights})

    @jit((nb.f8[:, :], nb.f8[:]))
    def snells(n, angles):
        incident = np.empty((prism_count+1, nwaves), np.float64)
        refracted = np.empty((prism_count+1, nwaves), np.float64)
        path = np.empty((prism_count+1, 2, nwaves), np.float64)

        mid = prism_count // 2
        midV2 = angles[glass_indices[mid]] / 2
        beta = np.sum(angles[glass_indices[:mid]]) + midV2
        gamma = np.sum(angles[glass_indices[mid + 1:]]) + midV2
        sides = height / np.cos(midV2 + np.array([np.sum(angles[glass_indices[i:mid]]) for i in range(mid+1)] + [np.sum(angles[glass_indices[mid+1:i+1]]) for i in range(mid, prism_count)]))

        path[0, 0, :] = (start - radius) / np.cos(beta)
        path[0, 1, :] = (start + radius) / np.cos(beta)
        incident[0] = theta0 + beta
        np.arcsin((1.0 / n[glass_indices[0]]) * np.sin(theta0 + beta), refracted[0])
        for i in range(1, prism_count + 1):
            alpha = angles[glass_indices[i - 1]]
            incident[i] = refracted[i-1] - alpha
            t1 = np.pi/2 - refracted[i-1]*np.sign(alpha)
            t2 = np.pi - np.abs(alpha) - t1
            if alpha > 0:
                path[i] = path[i-1]*np.sin(t1)/np.sin(t2)
            else:
                path[i] = sides[i] - (sides[i-1] - path[i-1])*np.sin(t1)/np.sin(t2)
            if i < prism_count:
                np.arcsin((n[glass_indices[i - 1]] / n[glass_indices[i]]) * np.sin(incident[i]), refracted[i])
            else:
                np.arcsin(n[glass_indices[i - 1]] * np.sin(incident[i]), refracted[i])
        delta = theta0 - (refracted[prism_count] + gamma)
        return delta, incident, refracted, path
        
    @jit((nb.f8[:, :], nb.f8[:]))
    def merit_error(n, angles):
        crit_count = 0
        invalid_count = 0
        mid = prism_count // 2
        midV2 = angles[glass_indices[mid]] / 2
        beta = np.sum(angles[glass_indices[:mid]]) + midV2
        gamma = np.sum(angles[glass_indices[mid + 1:]]) + midV2
        sides = height / np.cos(midV2 + np.array([np.sum(angles[glass_indices[i:mid]]) for i in range(mid+1)] + [np.sum(angles[glass_indices[mid+1:i+1]]) for i in range(mid, prism_count)]))

        path = np.empty((2, nwaves), np.float64)
        path[0, :] = (start - radius) / np.cos(beta)
        path[1, :] = (start + radius) / np.cos(beta)
        refracted = np.arcsin((1.0 / n[glass_indices[0]]) * np.sin(theta0 + beta))
        for i in range(1, prism_count + 1):
            alpha = angles[glass_indices[i - 1]]
            incident = refracted - alpha
            crit_count += np.sum(np.abs(incident) > angle_limit)
            # TODO: Triple-Check this
            t1 = np.pi/2 - refracted*np.sign(alpha)
            t2 = np.pi - np.abs(alpha) - t1
            if alpha > 0:
                path = path*np.sin(t1)/np.sin(t2)
            else:
                path = sides[i] - (sides[i-1] - path)*np.sin(t1)/np.sin(t2)
            invalid_count += np.sum(np.logical_or(path < 0, path > sides[i]))
            if i < prism_count:
                refracted = np.arcsin((n[glass_indices[i - 1]] / n[glass_indices[i]]) * np.sin(incident))
            else:
                refracted = np.arcsin(n[glass_indices[i - 1]] * np.sin(incident))
        delta = theta0 - (refracted + gamma)

        if np.any(np.isnan(delta)):
            return weights.tir * np.sum(np.isnan(delta))
        too_thin = np.abs(angles) - 1.0
        too_thin_err = np.sum(np.array([t**2 for t in too_thin if t+1.0 <= np.pi / 180]))
        merit_err = weights.crit_angle * crit_count / (prism_count * nwaves) \
                    + weights.valid * invalid_count / prism_count \
                    + weights.thin * too_thin_err / glass_count \
                    + weights.dispersion * ((delta.max() - delta.min()) - deltaT_target) ** 2 \
                    + weights.linearity * nonlinearity(delta)
        return merit_err

    @jit((nb.f8[:, :],))
    def minimizer(n):
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5
        nonzdelt = 0.05
        zdelt = 0.00025
        xatol = 1e-4
        fatol = 1e-4

        count = glass_count
        ncalls = 0

        sim = np.zeros((count + 1, count), dtype=np.float64)
        sim[0] = initial_angles
        for k in range(count):
            y = initial_angles.copy()
            if y[k] != 0:
                y[k] *= (1 + nonzdelt)
            else:
                y[k] = zdelt
            sim[k + 1] = y

        maxiter = count * 200
        maxfun = count * 200

        one2np1 = list(range(1, count + 1))
        fsim = np.zeros((count + 1,))

        for k in range(count + 1):
            fsim[k] = merit_error(n, sim[k])
            ncalls += 1

        ind = np.argsort(fsim)
        sim = sim[ind]
        fsim = fsim[ind]
        iterations = 1
        while ncalls < maxfun and iterations < maxiter:
            if np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and np.max(np.abs(fsim[0] - fsim[1:])) <= fatol:
                break

            # xbar = np.add.reduce(sim[:-1], 0) / N
            xbar = np.zeros(sim[0].size)
            for s in range(len(sim[:-1][0])):
                xbar += sim[:-1][s]
            xbar /= count

            xr = (1 + rho) * xbar - rho * sim[-1]
            fxr = merit_error(n, xr)
            ncalls += 1
            doshrink = False

            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                fxe = merit_error(n, xe)
                ncalls += 1
                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        fxc = merit_error(n, xc)
                        ncalls += 1
                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = True
                    else:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        fxcc = merit_error(n, xcc)
                        ncalls += 1
                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = True
                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            fsim[j] = merit_error(n, sim[j])
                            ncalls += 1
            ind = np.argsort(fsim)
            sim = sim[ind]
            fsim = fsim[ind]
            iterations += 1
        return sim[0], fsim[0]

    @jit((nb.f8[:, :],))
    def optimize(n):
        angles, merr = minimizer(n)
        delta_spectrum, incident, refracted, path = snells(n, angles)
        if np.any(np.isnan(delta_spectrum)):
            return

        (delta0, delta1, delta2), remainder = get_poly_coeffs(delta_spectrum, 2)
        deltaM = np.mean(delta_spectrum) * 180.0 / np.pi
        deltaC = delta_spectrum[nwaves // 2] * 180.0 / np.pi
        deltaT = (delta_spectrum.max() - delta_spectrum.min()) * 180.0 / np.pi
        alphas = angles * 180.0 / np.pi
        NL = 10000.0 * nonlinearity(delta_spectrum)
        SSR = 0  # spectral_sampling_ratio(w, delta_spectrum, sampling_domain_is_wavenumber)
        K = 0  # beam_compression(thetas, nwaves)
        _, remainder = get_poly_coeffs(delta_spectrum, 1)
        nonlin = np.sqrt(remainder) * dw * 180.0 / np.pi
        chromat = 100.0 * abs(delta_spectrum.max() - delta_spectrum.min()) * 180.0 / np.pi
        delta1 *= 2.0 * 180.0 / np.pi
        delta2 *= 2.0 * 180.0 / np.pi
        # T = transmission(thetas, n, nwaves)

        return alphas, merr, deltaC, deltaT, NL, SSR, K, deltaM, delta1, delta2, nonlin, chromat
    return optimize
