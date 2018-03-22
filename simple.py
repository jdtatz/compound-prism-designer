import numpy as np


def describe(n, angles, weights, start, radius, height, theta0, deltaC_target, deltaT_target, transmission_minimum):
    count, nwaves = n.shape
    mid = count // 2
    offAngle = np.sum(angles[:mid]) + angles[mid] / 2
    n1 = 1.0
    n2 = n[0]
    path0 = (start - radius) / np.cos(offAngle)
    path1 = (start + radius) / np.cos(offAngle)
    sideL = height / np.cos(offAngle)
    incident = theta0 + offAngle
    crit_angle = np.pi / 2
    crit_violation_count = np.sum(np.abs(incident) >= crit_angle * 0.999)
    if crit_violation_count > 0:
        return False, weights.tir * crit_violation_count
    refracted = np.arcsin((n1 / n2) * np.sin(incident))
    ci, cr = np.cos(incident), np.cos(refracted)
    T = 1 - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / 2
    for i in range(1, count + 1):
        n1 = n2
        n2 = n[i] if i < count else 1.0
        alpha = angles[i - 1]
        incident = refracted - alpha
        
        crit_angle = np.arcsin(np.minimum(n2 / n1, 1))
        crit_violation_count = np.sum(np.abs(incident) >= crit_angle * 0.999)
        if crit_violation_count > 0:
            return False, weights.tir * crit_violation_count

        if i <= mid:
            offAngle -= alpha
        elif i > mid + 1:
            offAngle += alpha
        sideR = height / np.cos(offAngle)
        t1 = np.pi / 2 - refracted * np.copysign(1, alpha)
        t2 = np.pi - np.abs(alpha) - t1
        los = np.sin(t1) / np.sin(t2)
        if alpha > 0:
            path0 *= los
            path1 *= los
        else:
            path0 = sideR - (sideL - path0) * los
            path1 = sideR - (sideL - path1) * los
        invalid_count = np.sum(np.logical_or(np.logical_or(0 > path0, path0 > sideR), np.logical_or(0 > path1, path1 > sideR)))
        if invalid_count > 0:
            return False, weights.invalid * invalid_count
        sideL = sideR

        refracted = np.arcsin((n1 / n2) * np.sin(incident))
        ci, cr = np.cos(incident), np.cos(refracted)
        T *= 1 - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / 2
    delta_spectrum = theta0 - (refracted + offAngle)
    transm_spectrum = T
    deltaC = delta_spectrum[nwaves // 2]
    deltaT = (delta_spectrum.max() - delta_spectrum.min())
    meanT = np.sum(transm_spectrum) / nwaves
    transm_err = np.max(transmission_minimum - meanT, 0)
    NL = np.sqrt(np.sum(np.gradient(np.gradient(delta_spectrum)) ** 2))
    merit_err = weights.deviation * (deltaC - deltaC_target) ** 2 \
                + weights.dispersion * (deltaT - deltaT_target) ** 2 \
                + weights.linearity * NL \
                + weights.transm * transm_err
    return True, merit_err, NL, deltaT, deltaC, delta_spectrum, transm_spectrum
