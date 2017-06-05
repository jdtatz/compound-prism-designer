from itertools import permutations, starmap
import numpy as np


def calc_n(glass_dict, wm):
    dispform = glass_dict['dispform']
    cd = glass_dict['cd']
    w = wm / 1000.0
    formula_rhs = 0
    if dispform == 1:
        formula_rhs = cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + cd[4] * w ** -6 + cd[5] * w ** -8
    elif dispform == 2:  # Sellmeier1
        formula_rhs = 1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                      cd[4] * w ** 2 / (w ** 2 - cd[5])
    elif dispform == 3:  # Herzberger
        L = 1.0 / (w ** 2 - 0.028)
        return cd[0] + cd[1] * L + cd[2] * L ** 2 + cd[3] * w ** 2 + cd[4] * w ** 4 + cd[5] * w ** 6
    elif dispform == 4:  # Sellmeier2
        formula_rhs = 1.0 + cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2] ** 2) + cd[3] * w ** 2 / (w ** 2 - cd[4] ** 2)
    elif dispform == 5:  # Conrady
        return cd[0] + cd[1] / w + cd[2] / w ** 3.5
    elif dispform == 6:  # Sellmeier3
        formula_rhs = 1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                      cd[4] * w ** 2 / (w ** 2 - cd[5]) + cd[6] * w ** 2 / (w ** 2 - cd[7])
    elif dispform == 7:  # HandbookOfOptics1
        formula_rhs = cd[0] + cd[1] / (w ** 2 - cd[2]) - cd[3] * w ** 2
    elif dispform == 8:  # HandbookOfOptics2
        formula_rhs = cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2]) - cd[3] * w ** 2
    elif dispform == 9:  # Sellmeier4
        formula_rhs = cd[0] + cd[1] * w ** 2 / (w ** 2 - cd[2]) + cd[3] * w ** 2 / (w ** 2 - cd[4])
    elif dispform == 10:  # Extended1
        formula_rhs = cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + cd[4] * w ** -6 + \
                      cd[5] * w ** -8 + cd[6] * w ** -10 + cd[7] * w ** -12
    elif dispform == 11:  # Sellmeier5
        formula_rhs = 1.0 + cd[0] * w ** 2 / (w ** 2 - cd[1]) + cd[2] * w ** 2 / (w ** 2 - cd[3]) + \
                      cd[4] * w ** 2 / (w ** 2 - cd[5]) + cd[6] * w ** 2 / (w ** 2 - cd[7]) + \
                      cd[8] * w ** 2 / (w ** 2 - cd[9])
    elif dispform == 12:  # Extended2
        formula_rhs = cd[0] + cd[1] * w ** 2 + cd[2] * w ** -2 + cd[3] * w ** -4 + \
                      cd[4] * w ** -6 + cd[5] * w ** -8 + cd[6] * w ** 4 + cd[7] * w ** 6
    return np.sqrt(formula_rhs)


def read_glasscat(catalog_filename):
    glasscat = {}
    with open(catalog_filename, 'r') as f:
        for line in f:
            if line.startswith('NM'):
                nm = line.split()
                glassname = (nm[1]).upper()
                glasscat[glassname] = {'name': glassname, 'catalog': catalog_filename, 'dispform': int(nm[2])}
            elif line.startswith('CD'):
                glasscat[glassname]['cd'] = tuple(map(float, line[2:].split()))
            elif line.startswith('LD'):
                glasscat[glassname]['ld'] = tuple(map(float, line[2:].split()))
    return glasscat


def glass_combos(gcat, count, w):
    transmission = lambda g: (gcat[g]['ld'][0] < w.min() / 1000.0) and (gcat[g]['ld'][1] > w.max() / 1000.0)
    glasses = list(map(lambda g: (g, calc_n(gcat[g], w)), filter(transmission, gcat)))
    return starmap(zip, permutations(glasses, count))


def glass_paired(gcat, combos, w):
    ns = {g: calc_n(gcat[g], w) for g in gcat}
    for combination in combos:
        yield combination, [ns[g] for g in combination]