import numpy as np
from compoundprism.glasscat import read_glasscat, calc_n
from reference import describe

names = "N-LASF41 SF57 N-SK11".split()
angles = -17.469501, 77.99999, -49.67049, 35.29293
angles = np.deg2rad(angles)
nwaves = 16

w = np.linspace(650, 1000, nwaves)
gcat = read_glasscat('Glasscat/schott_positive_glass_trimmed_oct2015.agf')
glasses = np.stack(calc_n(gcat[name], w) for name in names)


config = {
    "height": 10,
    "start": 9,
    "theta0": 0,
    "deviation_target": 0,
    "dispersion_target": np.deg2rad(16),
    "max_size": 60,
    "weight_deviation": 25,
    "weight_dispersion": 250,
    "weight_linearity": 1000,
    "weight_transmittance": 30,
    "weight_thinness": 1
}

ret = describe(glasses, angles, 0.3, config)
if ret:
    err, NL, dispersion, deviation, size, spec_pos, transm = ret
    print(f"error: {err}\nNL: {NL}\nsize: {size}\nspectrometer position: {spec_pos)}\nT: {transm *100}")
else:
    print('Failed')
