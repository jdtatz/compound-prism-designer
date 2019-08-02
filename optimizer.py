#!/usr/bin/env python3
from os import cpu_count
from sys import argv
import numpy as np
import toml
from prism import Config, Params, Soln, RayTraceError, create_catalog, detector_array_position, trace, transmission_data, use_pygmo


def show_interactive(config: Config, solns: [Soln], units: str):
    import matplotlib as mpl
    import matplotlib.path
    import matplotlib.patches
    import matplotlib.pyplot as plt
    fig = plt.figure()
    nrows, ncols = 2, 4
    objectives_plt = fig.add_subplot(1, ncols, 1)

    x, y, c = np.array([soln.objectives for soln in solns]).T
    y = -y
    c = np.rad2deg(np.arcsin(c))

    sc = objectives_plt.scatter(x, y, c=c, cmap="viridis", picker=True)
    clb = fig.colorbar(sc)
    clb.ax.set_ylabel("deviation (deg)")
    objectives_plt.set_xlabel(f"size ({units})")
    objectives_plt.set_ylabel("mutual information (bits)")
    # ax.set_zlabel("deviation (deg)")
    """
    x, y, z, c = res.T
    c = 1 - c

    sc = ax.scatter(x, y, z, c=c, cmap="viridis", picker=True)
    clb = fig.colorbar(sc)
    clb.ax.set_ylabel('transmittance')
    """
    prism_plt = fig.add_subplot(nrows, ncols, 2)
    text_ax = fig.add_subplot(nrows, ncols, 6)
    det_plt = fig.add_subplot(nrows, ncols, 3)
    trans_plt = fig.add_subplot(nrows, ncols, 7)
    violin_plt = fig.add_subplot(1, ncols, 4)
    prism_plt.axis('off')
    text_ax.axis('off')

    def pick(event):
        soln = sorted((solns[i] for i in event.ind), key=lambda s: s.objectives[-1])[0]
        params: Params = soln.params
        size, ninfo, dev = soln.objectives
        display = f"""Prism ({', '.join(params.glass_names)})
    Parameters:
        angles (deg): {', '.join(f'{np.rad2deg(angle):.4}' for angle in params.thetas)}
        y_mean ({units}): {params.y_mean:.4}
        curvature: {params.curvature:.4}
        detector array angle (deg): {np.rad2deg(params.det_arr_angle):.4}
        objectives: (size={size:.4} ({units}), info: {-ninfo:.4} (bits), deviation: {np.rad2deg(np.arcsin(dev)):.4} (deg))"""
        print(display)
        text_ax.cla()
        text_ax.text(0, 0.5, display, horizontalalignment='left', verticalalignment='center',
                     transform=text_ax.transAxes)
        text_ax.axis('scaled')
        text_ax.axis('off')

        det_arr_pos, det_arr_dir = det = detector_array_position(config, params)
        det_arr_end = det_arr_pos + det_arr_dir * config.det_arr_length

        prism_vertices = np.empty((config.prism_count + 2, 2))
        prism_vertices[::2, 1] = 0
        prism_vertices[1::2, 1] = config.prism_height
        prism_vertices[0, 0] = 0
        prism_vertices[1:, 0] = np.add.accumulate(np.tan(np.abs(params.thetas)) * config.prism_height)
        triangles = np.stack((prism_vertices[1:-1], prism_vertices[:-2], prism_vertices[2:]), axis=1)

        midpt = prism_vertices[-2] + (prism_vertices[-1] - prism_vertices[-2]) / 2
        c, s = np.cos(params.thetas[-1]), np.sin(params.thetas[-1])
        R = np.array(((c, -s), (s, c)))
        normal = R @ (-1, 0)
        chord = config.prism_height / c
        # curvature = R_max / R_lens
        # R_max = chord / 2
        # R_lens = chord / (2 curvature)
        lens_radius = chord / (2 * params.curvature)
        center = midpt + normal * np.sqrt(lens_radius ** 2 - chord ** 2 / 4)
        t1 = np.rad2deg(np.arctan2(prism_vertices[-1, 1] - center[1], prism_vertices[-1, 0] - center[0]))
        t2 = np.rad2deg(np.arctan2(prism_vertices[-2, 1] - center[1], prism_vertices[-2, 0] - center[0]))
        if config.prism_count % 2 == 0:
            t1, t2 = t2, t1

        prism_plt.cla()
        for i, tri in enumerate(triangles):
            poly = plt.Polygon(tri, edgecolor='k', facecolor=('gray' if i % 2 else 'white'), closed=False)
            prism_plt.add_patch(poly)
        arc = mpl.path.Path.arc(t1, t2)
        arc = mpl.path.Path(arc.vertices * lens_radius + center, arc.codes)
        arc = mpl.patches.PathPatch(arc, fill=None)
        prism_plt.add_patch(arc)

        spectro = plt.Polygon((det_arr_pos, det_arr_end), closed=None, fill=None, edgecolor='k')
        prism_plt.add_patch(spectro)

        for w, color in zip((config.wmin, (config.wmin + config.wmax) / 2, config.wmax), ('r', 'g', 'b')):
            try:
                ray = np.stack(tuple(trace(w, params.y_mean, config, params, det)), axis=0)
                poly = plt.Polygon(ray, closed=None, fill=None, edgecolor=color)
                prism_plt.add_patch(poly)
            except RayTraceError:
                pass

        prism_plt.axis('scaled')
        prism_plt.axis('off')

        ytans = np.tan(params.thetas)
        thickness = [abs(t0) * config.prism_height / 2 + abs(t1) * config.prism_height / 2 for t0, t1 in zip(ytans[:-1], ytans[1:])]
        newline = "\n        "

        detarr_offset = (det_arr_pos + det_arr_dir * config.det_arr_length / 2) - midpt
        zemax_info = (f"""\
    Zemax Rows with Apeature pupil diameter = {config.beam_width} & Wavelengths from {config.wmin} to {config.wmax}: 
        Coord break: decenter y = {config.prism_height / 2 - params.y_mean}
        {f"{newline}".join(f"Tilted: thickness = {t} material = {g} semi-dimater = {config.prism_height / 2} x tan = 0 y tan = {-y}" for g, y, t in zip(params.glass_names, ytans, thickness))}
        Coord break: tilt about x = {-np.rad2deg(params.thetas[-1])}
        Biconic: radius = {-lens_radius} semi-dimater = {chord / 2} conic = 0 x_radius = 0 x_conic = 0
        Coord break: tilt about x = {np.rad2deg(params.thetas[-1])}
        Coord break: thickness: {detarr_offset[0]} decenter y: {detarr_offset[1]}
        Coord break: tilt about x = {-np.rad2deg(params.det_arr_angle)}
        Image (Standard): semi-dimater = {config.det_arr_length / 2}
""")
        print(zemax_info)
        """c_x = 1 / R_x
            c_y = 1 / R_y
            z = (c_x x^2 + c_y y^2 ) / (1 + Sqrt[1 - (1 + k_x)c_x^2 x^2 - (1 + k_y) c_y^2 y^2])
            """


        det_plt.clear()
        trans_plt.cla()
        violin_plt.cla()

        waves = np.linspace(config.wmin, config.wmax, 100)
        ts = transmission_data(waves, config, params, det)
        p_det = ts.sum(axis=1) * (1 / len(waves))
        p_w_l_D = ts.sum(axis=0)
        vpstats = [
            {
                "coords": waves,
                "vals": t,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }
            for t in ts
        ]
        parts = violin_plt.violin(vpstats, showextrema=False, widths=1)
        for pc in parts['bodies']:
            pc.set_facecolor('black')
        violin_plt.plot([1, len(p_det)], [config.wmin, config.wmax], 'k--')
        det_plt.scatter(1 + np.arange(len(p_det)), p_det * 100, color='k')
        trans_plt.plot(waves, p_w_l_D * 100, 'k')
        trans_plt.axhline(np.mean(p_w_l_D) * 100, color='k', linestyle='--')

        det_plt.set_xlabel("detector bins")
        trans_plt.set_xlabel("wavelength (μm)")
        violin_plt.set_xlabel("detector bins")
        det_plt.set_ylabel("p (%)")
        trans_plt.set_ylabel("p (%)")
        violin_plt.set_ylabel("wavelength (μm)")
        det_plt.set_title("p(D=d|Λ)")
        trans_plt.set_title("p(D|Λ=λ)")
        violin_plt.set_title("p(D=d|Λ=λ) as a Pseudo Violin Plot")
        trans_plt.set_ylim(0, 100)
        det_plt.set_ylim(bottom=0)

        fig.canvas.draw()
        fig.canvas.flush_events()

    fig.canvas.mpl_connect('pick_event', pick)

    plt.show()


if __name__ == "__main__":
    toml_file = argv[1] if len(argv) > 1 else "design_config.toml"
    toml_spec = toml.load(toml_file)
    catalog_path = toml_spec["catalog-path"]
    with open(catalog_path) as f:
        catalog = create_catalog(f.read())
    units = toml_spec.get("length-unit", "a.u.")

    config = Config(
        prism_count=toml_spec["compound-prism"]["count"],
        prism_height=toml_spec["compound-prism"]["height"],
        prism_width=toml_spec["compound-prism"]["width"],
        beam_width=toml_spec["gaussian-beam"]["width"],
        wmin=toml_spec["gaussian-beam"]["wmin"],
        wmax=toml_spec["gaussian-beam"]["wmax"],
        det_arr_length=toml_spec["detector-array"]["length"],
        det_arr_min_ci=np.cos(np.deg2rad(toml_spec["detector-array"].get("max-incident-angle", 90))),
        bounds=np.array(toml_spec["detector-array"]["bounds"])
    )

    opt_dict = {"iteration-count": 1000, "thread-count": 0, "pop-size": 20, **toml_spec.get("optimizer", {})}
    iter_count = max(opt_dict["iteration-count"], 1)
    thread_count = opt_dict["thread-count"]
    if thread_count < 1:
        thread_count = cpu_count()
    pop_size = opt_dict["pop-size"]

    solutions = use_pygmo(iter_count, thread_count, pop_size, config, catalog)
    print(len(solutions))

    show_interactive(config, solutions, units)
