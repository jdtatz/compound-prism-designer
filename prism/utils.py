import numpy as np
from prism import Config, Params


def midpts_gen(h, angles, lengths):
    h = h / 2
    x = np.tan(np.abs(angles[0])) * h
    yield x, h
    for last, angle, l in zip(angles[:-1], angles[1:], lengths):
        if (last >= 0) ^ (angle >= 0):
            x += (np.tan(np.abs(last)) + np.tan(np.abs(angle))) * h + l
        elif np.abs(last) > np.abs(angle):
            x += (np.tan(np.abs(last)) - np.tan(np.abs(angle))) * h + l
        else:
            x += (np.tan(np.abs(angle)) - np.tan(np.abs(last))) * h + l
        yield x, h


def lines_gen(h, angles, lengths):
    for midpt, angle in zip(midpts_gen(h, angles, lengths), angles):
        mid_x = midpt[0]
        diff = np.tan(np.abs(angle)) * (h / 2)
        l, r = mid_x - diff, mid_x + diff
        if angle > 0:
            yield (l, h), (r, 0)
        else:
            yield (l, 0), (r, h)


def draw_compound_prism(ax, config: Config, params: Params):
    import matplotlib as mpl
    import matplotlib.path
    import matplotlib.patches

    angles = params.angles
    lines = np.array(list(lines_gen(config.prism_height, params.angles, params.lengths)))

    def polygon_gen():
        for (a, b), (c, d), last, angle in zip(lines[:-1], lines[1:], angles[:-1], angles[1:]):
            if (last >= 0) ^ (angle >= 0):
                yield np.array((d, a, b, c))
            elif np.abs(last) > np.abs(angle):
                yield np.array((d, b, a, c))
            else:
                yield np.array((c, a, b, d))
    polygons = list(polygon_gen())

    midpt = list(midpts_gen(config.prism_height, params.angles, params.lengths))[-1]
    c, s = np.cos(params.angles[-1]), np.sin(params.angles[-1])
    R = np.array(((c, -s), (s, c)))
    normal = R @ (-1, 0)
    chord = config.prism_height / c
    # curvature = R_max / R_lens
    # R_max = chord / 2
    # R_lens = chord / (2 curvature)
    lens_radius = chord / (2 * params.curvature)
    center = midpt + normal * np.sqrt(lens_radius ** 2 - chord ** 2 / 4)
    t1 = np.rad2deg(np.arctan2(lines[-1, 0, 1] - center[1], lines[-1, 0, 0] - center[0]))
    t2 = np.rad2deg(np.arctan2(lines[-1, 1, 1] - center[1], lines[-1, 1, 0] - center[0]))
    if params.angles[-1] > 0:
        t1, t2 = t2, t1

    ax.cla()
    for i, poly in enumerate(polygons):
        poly = mpl.patches.Polygon(poly, edgecolor='k', facecolor=('gray' if i % 2 else 'white'), closed=False)
        ax.add_patch(poly)
    arc = mpl.path.Path.arc(t1, t2)
    arc = mpl.path.Path(arc.vertices * lens_radius + center, arc.codes)
    arc = mpl.patches.PathPatch(arc, fill=None)
    ax.add_patch(arc)

