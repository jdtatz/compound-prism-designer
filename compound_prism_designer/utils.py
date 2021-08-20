from __future__ import annotations
from typing import Iterable, Sequence, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.path
import matplotlib.patches
from .compound_prism_designer import RayTraceError
from .compound_prism_designer import Spectrometer


def draw_spectrometer(ax, spectrometer: Spectrometer, draw_wavelengths: Iterable[Tuple[float, str]], starting_ys: Sequence[float]):
    ax.cla()

    prism = spectrometer.compound_prism
    polys = prism.polygons()
    for i, poly in enumerate(polys):
        poly = mpl.patches.PathPatch(poly, edgecolor='k', facecolor=('gray' if i % 2 else 'white'))
        ax.add_patch(poly)
    det_arr_pos = np.array(spectrometer.position)
    det_arr_dir = np.array(spectrometer.direction)
    detector_array_length = spectrometer.detector_array.length
    det_arr_end = det_arr_pos + det_arr_dir * detector_array_length

    spectro = mpl.patches.Polygon((det_arr_pos, det_arr_end), closed=None, fill=None, edgecolor='k')
    ax.add_patch(spectro)

    for w, color in draw_wavelengths:
        for y in starting_ys:
            try:
                ray_path, _ray_dir = np.split(spectrometer.ray_trace(w, y), 2, axis=1)
                poly = mpl.patches.Polygon(ray_path, closed=None, fill=None, edgecolor=color)
                ax.add_patch(poly)
            except RayTraceError:
                pass
    ax.axis('scaled')
    ax.axis('off')
