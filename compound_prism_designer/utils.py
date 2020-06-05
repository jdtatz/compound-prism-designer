from typing import Iterable, Sequence, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.path
import matplotlib.patches
from .compound_prism_designer import Spectrometer, RayTraceError


def draw_spectrometer(ax, spectrometer: Spectrometer, draw_wavelengths: Iterable[Tuple[float, str]], starting_ys: Sequence[float]):
    ax.cla()

    prism = spectrometer.compound_prism
    polys, lens_poly, center, radius = prism.polygons()

    t1 = np.rad2deg(np.arctan2(*(lens_poly[3] - center)[::-1]))
    t2 = np.rad2deg(np.arctan2(*(lens_poly[2] - center)[::-1]))

    ax.cla()
    for i, poly in enumerate(polys):
        poly = mpl.patches.Polygon(poly, edgecolor='k', facecolor=('gray' if i % 2 else 'white'), closed=True)
        ax.add_patch(poly)
    arc = mpl.path.Path.arc(t1, t2)
    arc = mpl.path.Path(arc.vertices * radius + center, arc.codes)
    lens_like = mpl.path.Path.make_compound_path(arc, mpl.path.Path(lens_poly[[2, 1, 0, 3]], closed=False))
    lens_like = mpl.patches.PathPatch(lens_like, fill=True, edgecolor='k', facecolor=('gray' if len(polys) % 2 else 'white'))
    ax.add_patch(lens_like)

    det_arr_pos = np.array(spectrometer.position)
    det_arr_dir = np.array(spectrometer.direction)
    detector_array_length = spectrometer.detector_array.length
    det_arr_end = det_arr_pos + det_arr_dir * detector_array_length

    spectro = mpl.patches.Polygon((det_arr_pos, det_arr_end), closed=None, fill=None, edgecolor='k')
    ax.add_patch(spectro)

    for w, color in draw_wavelengths:
        for y in starting_ys:
            try:
                ray = spectrometer.ray_trace(w, y)
                poly = mpl.patches.Polygon(ray, closed=None, fill=None, edgecolor=color)
                ax.add_patch(poly)
            except RayTraceError:
                pass
    ax.axis('scaled')
    ax.axis('off')
