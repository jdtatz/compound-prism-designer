import numpy as np
from svgwrite import Drawing


def describe(n, angles, config):
    assert np.all(np.abs(angles) < np.pi), 'Need Radians not Degrees'
    count, nwaves = n.shape
    n1 = 1
    n2 = n[0]
    pts = np.zeros((count+2, 2), dtype=np.float32)
    pts[::2, 1] = config.height
    pts[1, 0] = config.height * np.tan(angles[0])
    path0 = (config.start - config.radius) / np.cos(angles[0])
    path1 = (config.start + config.radius) / np.cos(angles[0])
    ptsL = np.empty((nwaves, count + 1, 2), dtype=np.float32)
    ptsU = np.empty((nwaves, count + 1, 2), dtype=np.float32)
    ptsL[:, 0, 0] = pts[1, 0] - (config.start - config.radius) * np.tan(angles[0])
    ptsL[:, 0, 1] = config.start - config.radius
    ptsU[:, 0, 0] = pts[1, 0] - (config.start + config.radius) * np.tan(angles[0])
    ptsU[:, 0, 1] = config.start + config.radius
    sideL = config.height / np.cos(angles[0])
    incident = config.theta0 + angles[0]
    offAngle = angles[1]
    crit_angle = np.pi / 2
    crit_violation_count = np.sum(np.abs(incident) >= crit_angle * config.crit_angle_prop)
    if crit_violation_count > 0:
        return 0
    refracted = np.arcsin((n1 / n2) * np.sin(incident))
    ci, cr = np.cos(incident), np.cos(refracted)
    T = 1 - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / 2
    size = 0
    for i in range(1, count + 1):
        n1 = n2
        n2 = n[i] if i < count else 1
        alpha = angles[i] if i > 1 else (angles[1] + angles[0])
        incident = refracted - alpha
        
        crit_angle = np.arcsin(np.minimum(n2 / n1, 1))
        crit_violation_count = np.sum(np.abs(incident) >= crit_angle * config.crit_angle_prop)
        if crit_violation_count > 0:
            return i

        if i > 1:
            offAngle += alpha
        sideR = config.height / np.cos(offAngle)
        t1 = np.pi / 2 - refracted * np.copysign(1, alpha)
        t2 = np.pi - np.abs(alpha) - t1
        los = np.sin(t1) / np.sin(t2)
        if alpha > 0:
            path0 *= los
            path1 *= los
            ptsL[:, i, 1] = np.cos(offAngle) * path0
            ptsU[:, i, 1] = np.cos(offAngle) * path1
        else:
            path0 = sideR - (sideL - path0) * los
            path1 = sideR - (sideL - path1) * los
            ptsL[:, i, 1] = config.height - np.cos(offAngle) * (sideR - path0)
            ptsU[:, i, 1] = config.height - np.cos(offAngle) * (sideR - path1)
        ptsL[:, i, 0] = pts[i, 0] + np.sin(abs(offAngle)) * path0
        ptsU[:, i, 0] = pts[i, 0] + np.sin(abs(offAngle)) * path1

        invalid_count = np.sum(np.logical_or(np.logical_or(0 > path0, path0 > sideR), np.logical_or(0 > path1, path1 > sideR)))
        if invalid_count > 0:
            return i
        size += np.sqrt(sideL**2 + sideR**2 - 2*sideL*sideR*np.cos(alpha))
        sideL = sideR
        pts[i+1, 0] = size

        refracted = np.arcsin((n1 / n2) * np.sin(incident))
        ci, cr = np.cos(incident), np.cos(refracted)
        T *= 1 - (((n1 * ci - n2 * cr) / (n1 * ci + n2 * cr)) ** 2 + ((n1 * cr - n2 * ci) / (n1 * cr + n2 * ci)) ** 2) / 2
    delta_spectrum = config.theta0 - (refracted + offAngle)
    transmission_spectrum = T
    deltaC = delta_spectrum[nwaves // 2]
    deltaT = (delta_spectrum.max() - delta_spectrum.min())
    meanT = np.sum(transmission_spectrum) / nwaves
    transmission_err = np.max(1 - meanT, 0)
    NL = np.sqrt(np.sum(np.gradient(np.gradient(delta_spectrum)) ** 2))
    merit_err = config.weight_deviation * (deltaC - config.deltaC_target) ** 2 \
                + config.weight_dispersion * (deltaT - config.deltaT_target) ** 2 \
                + config.weight_linearity * NL \
                + config.weight_transmission * transmission_err \
                + config.weight_thinness * max(size - config.max_size, 0)
    triangles = np.stack((pts[:-2], pts[1:-1], pts[2:]), axis=1)

    dwg = Drawing('figure.svg')
    for i, tri in enumerate(triangles):
        dwg.add(dwg.polygon(tri.tolist(), fill=('gray' if i%2 else 'white')))
    dwg.add(dwg.polyline(ptsU[-1].tolist(), stroke='red', stroke_width=0.1, fill='none'))
    dwg.save()

    return merit_err, NL, deltaT, deltaC, size, delta_spectrum, transmission_spectrum

