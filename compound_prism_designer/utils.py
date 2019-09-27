import numpy as np
from compound_prism_designer import CompoundPrism


def midpts_gen(prism: CompoundPrism):
    h = prism.height / 2
    x = np.tan(np.abs(prism.angles[0])) * h
    yield x, h
    for last, angle, l in zip(prism.angles[:-1], prism.angles[1:], prism.lengths):
        if (last >= 0) ^ (angle >= 0):
            x += (np.tan(np.abs(last)) + np.tan(np.abs(angle))) * h + l
        elif np.abs(last) > np.abs(angle):
            x += (np.tan(np.abs(last)) - np.tan(np.abs(angle))) * h + l
        else:
            x += (np.tan(np.abs(angle)) - np.tan(np.abs(last))) * h + l
        yield x, h


def lines_gen(prism: CompoundPrism):
    h = prism.height
    for midpt, angle in zip(midpts_gen(prism), prism.angles):
        mid_x = midpt[0]
        diff = np.tan(np.abs(angle)) * (h / 2)
        l, r = mid_x - diff, mid_x + diff
        if angle > 0:
            yield (l, h), (r, 0)
        else:
            yield (l, 0), (r, h)


def draw_compound_prism(ax, prism: CompoundPrism):
    import matplotlib as mpl
    import matplotlib.path
    import matplotlib.patches

    angles = prism.angles
    lines = np.array(list(lines_gen(prism)))

    def polygon_gen():
        for (a, b), (c, d), last, angle in zip(lines[:-1], lines[1:], angles[:-1], angles[1:]):
            if (last >= 0) ^ (angle >= 0):
                yield np.array((d, a, b, c))
            elif np.abs(last) > np.abs(angle):
                yield np.array((d, b, a, c))
            else:
                yield np.array((c, a, b, d))

    polygons = list(polygon_gen())

    midpt = list(midpts_gen(prism))[-1]
    c, s = np.cos(prism.angles[-1]), np.sin(prism.angles[-1])
    R = np.array(((c, -s), (s, c)))
    normal = R @ (-1, 0)
    chord = prism.height / c
    # curvature = R_max / R_lens
    # R_max = chord / 2
    # R_lens = chord / (2 curvature)
    lens_radius = chord / (2 * prism.curvature)
    center = midpt + normal * np.sqrt(lens_radius ** 2 - chord ** 2 / 4)
    t1 = np.rad2deg(np.arctan2(lines[-1, 0, 1] - center[1], lines[-1, 0, 0] - center[0]))
    t2 = np.rad2deg(np.arctan2(lines[-1, 1, 1] - center[1], lines[-1, 1, 0] - center[0]))
    if prism.angles[-1] > 0:
        t1, t2 = t2, t1

    ax.cla()
    for i, poly in enumerate(polygons[:-1]):
        poly = mpl.patches.Polygon(poly, edgecolor='k', facecolor=('gray' if i % 2 else 'white'), closed=False)
        ax.add_patch(poly)
    arc = mpl.path.Path.arc(t1, t2)
    arc = mpl.path.Path(arc.vertices * lens_radius + center, arc.codes)
    if np.allclose(arc.vertices[-1], polygons[-1][0]):
        last = mpl.path.Path.make_compound_path(arc, mpl.path.Path(polygons[-1]))
    else:
        last = mpl.path.Path.make_compound_path(arc, mpl.path.Path(polygons[-1][::-1]))
    last = mpl.patches.PathPatch(last, fill=True, edgecolor='k', facecolor=('white' if len(polygons) % 2 else 'gray'))
    ax.add_patch(last)
