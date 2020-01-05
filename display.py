import numpy as np
import matplotlib.pyplot as plt
import cbor2
import json
from compound_prism_designer import designs_from_json

with open("results.cbor", "rb") as f:
    results = cbor2.load(f)

designs = designs_from_json(json.dumps([result["design"] for result in results]))
print(designs)
print(results[0])
ws = None
for result in results:
    t = result["transmission_data"]
    ws = np.array(t["wavelengths"])
    result["transmission_data"] = d = np.array(t["data"])
    print(result["design"]["fitness"])
    if result["design"]["fitness"]["info"] > 3.6:
        pt = 100 * np.sum(d, axis=1)
        pdet = 100 * np.sum(d, axis=0) / len(pt)
        fig, axs = plt.subplots(1, 3)
        axs[0].plot(ws, pt)
        axs[0].set_ylim(0, 100)
        axs[1].scatter(1 + np.arange(len(pdet)), pdet)

        vpstats = [
            {
                "coords": ws,
                "vals": t,
                "mean": None,
                "median": None,
                "min": None,
                "max": None,
            }
            for t in d.T
        ]
        parts = axs[2].violin(vpstats, showextrema=False, widths=1)
        for pc in parts['bodies']:
            pc.set_facecolor('black')
        axs[2].plot([1, len(pdet)], (0.5, 0.82), 'k--')
        print(result["design"])
        plt.show()
