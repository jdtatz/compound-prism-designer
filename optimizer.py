from os import cpu_count
from sys import argv
import numpy as np
import toml
from compound_prism_designer import Config, Soln, RayTraceError, create_glass_catalog, \
    detector_array_position, trace, transmission_data, use_pygmo
from compound_prism_designer.utils import draw_compound_prism, midpts_gen
from compound_prism_designer.zemax import create_zmx


def show_interactive(config: Config, solns: [Soln], units: str):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    nrows, ncols = 2, 4
    objectives_plt = fig.add_subplot(1, ncols, 1)

    x, y, c = np.array([(soln.fitness.size, soln.fitness.info, soln.fitness.deviation) for soln in solns]).T
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
        soln = sorted((solns[i] for i in event.ind), key=lambda s: -s.fitness.info)[0]
        prism = soln.compound_prism
        detarr = soln.detector_array
        beam = soln.beam
        size, info, dev = soln.fitness.size, soln.fitness.info, soln.fitness.deviation

        det_arr_pos, det_arr_dir = det = detector_array_position(prism, detarr, beam)
        det_arr_end = det_arr_pos + det_arr_dir * config.detector_array_length

        midpt = list(midpts_gen(prism))[-1]
        detarr_offset = (det_arr_pos + det_arr_dir * config.detector_array_length / 2) - midpt

        draw_compound_prism(prism_plt, prism)
        spectro = plt.Polygon((det_arr_pos, det_arr_end), closed=None, fill=None, edgecolor='k')
        prism_plt.add_patch(spectro)

        wmin, wmax = config.wavelength_range
        for w, color in zip((wmin, (wmin + wmax) / 2, wmax), ('r', 'g', 'b')):
            for y in (beam.y_mean - config.beam_width, beam.y_mean, beam.y_mean + config.beam_width):
                try:
                    ray = np.stack(tuple(trace(w, y, prism, detarr, det)), axis=0)
                    poly = plt.Polygon(ray, closed=None, fill=None, edgecolor=color)
                    prism_plt.add_patch(poly)
                except RayTraceError:
                    pass

        prism_plt.axis('scaled')
        prism_plt.axis('off')

        display = f"""CompoundPrism ({', '.join(g.name for g in prism.glasses)})
    Parameters:
        height ({units}): {prism.height:.4}
        angles (deg): {', '.join(f'{np.rad2deg(angle):.4}' for angle in prism.angles)}
        lengths ({units}): {', '.join(f'{l:.4}' for l in prism.lengths)}
        y_mean ({units}): {beam.y_mean:.4}
        curvature: {prism.curvature:.4}
        detector array angle (deg): {np.rad2deg(detarr.angle):.4}
        objectives: (size={size:.4} ({units}), info: {info:.4} (bits), deviation: {np.rad2deg(np.arcsin(dev)):.4} (deg))
"""
        zemax_design, zemax_file = create_zmx(prism, detarr, beam, detarr_offset)
        display += zemax_design
        print(display)
        with open("spec.zmx", "w") as f:
            f.write(zemax_file)
        text_ax.cla()
        text_ax.text(0, 0.5, display, horizontalalignment='left', verticalalignment='center',
                     transform=text_ax.transAxes)
        text_ax.axis('scaled')
        text_ax.axis('off')

        det_plt.clear()
        trans_plt.cla()
        violin_plt.cla()

        waves = np.linspace(config.wavelength_range[0], config.wavelength_range[1], 100)
        ts = transmission_data(waves, prism, detarr, beam, det)
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
        violin_plt.plot([1, len(p_det)], config.wavelength_range, 'k--')
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
        catalog = create_glass_catalog(f.read())
    units = toml_spec.get("length-unit", "a.u.")

    config = Config(
        max_prism_count=toml_spec["compound-prism"]["max-count"],
        max_prism_height=toml_spec["compound-prism"]["max-height"],
        prism_width=toml_spec["compound-prism"]["width"],
        beam_width=toml_spec["gaussian-beam"]["width"],
        wavelength_range=(toml_spec["gaussian-beam"]["wmin"], toml_spec["gaussian-beam"]["wmax"]),
        detector_array_length=toml_spec["detector-array"]["length"],
        detector_array_min_ci=np.cos(np.deg2rad(toml_spec["detector-array"].get("max-incident-angle", 90))),
        detector_array_bin_bounds=np.array(toml_spec["detector-array"]["bounds"]),
        glass_catalog=catalog
    )

    opt_dict = {"iteration-count": 1000, "thread-count": 0, "pop-size": 20, **toml_spec.get("optimizer", {})}
    iter_count = max(opt_dict["iteration-count"], 1)
    thread_count = opt_dict["thread-count"]
    if thread_count < 1:
        thread_count = cpu_count()
    pop_size = opt_dict["pop-size"]

    solutions = use_pygmo(iter_count, thread_count, pop_size, config)
    print(len(solutions))

    show_interactive(config, solutions, units)
