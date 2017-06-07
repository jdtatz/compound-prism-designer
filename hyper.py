from glasscat import read_glasscat, glass_paired
from multiprocessing import cpu_count
from threading import Thread, Lock
import time, sys, json, sqlite3
from prisms import CompoundPrism
from itertools import repeat
from uuid import uuid4
from math import pi
import numpy as np


def hyper(**settings):
    prism_name = settings.get('prism name', 'prism')
    count = settings.get('glass count', 0)
    indices = settings.get('glass indices', None)
    catalog_filename = settings["catalog"]
    merit = settings["merit function"]
    weights = settings.get("weights", {})
    sampling_domain = settings.get("sampling domain", "wavelength")
    wavemin = settings.get("wave minimum", 500.0)
    wavemax = settings.get("wave maximum", 820.0)
    nwaves = settings.get("number of waves", 100)
    w = settings.get("w", None)
    deviation_target = settings.get("deviation target", 0.0)
    dispersion_target = settings.get("dispersion target", 1.0)
    angle_lim = settings.get("incident angle limit", 65.0)
    theta0 = settings.get("theta0", 0.0)
    iangles = settings.get("initial angles", None)
    database_file = settings["database file"]
    input_table_name = settings["input table name"]
    output_table_name = settings.get("output table name", "t" + uuid4().hex)
    thread_count = settings.get("thread count", cpu_count())

    if indices is not None:
        count = max(indices) + 1
    elif count > 0:
        indices = tuple(range(count))
    else:
        raise Exception("Invalid prism parametrization")

    if w is not None:
        w = np.asarray(w, np.float64)
    elif sampling_domain == "wavenumber":
        w = 1.0 / np.linspace(1.0 / wavemax, 1.0 / wavemin, nwaves, dtype=np.float64)
    else:
        w = np.linspace(wavemin, wavemax, nwaves, dtype=np.float64)

    gcat = read_glasscat(catalog_filename)
    nglass = len(gcat)
    # in radians
    deltaC_target = deviation_target * pi / 180.0
    deltaT_target = dispersion_target * pi / 180.0
    angle_lim *= pi / 180.0

    print('input table:', input_table_name)
    print('output table:', output_table_name)
    print('# glasses in catalog =', nglass)
    print(f'targets: deltaC = {deviation_target} deg, deltaT = {dispersion_target} deg')
    print('thread count:', thread_count, '\n')
    cmpnd = CompoundPrism(merit, settings)
    model = cmpnd.configure(indices, deltaC_target, deltaT_target, weights, sampling_domain, theta0, iangles, angle_lim, w)

    conn = sqlite3.connect(database_file, check_same_thread=False)
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS runs (
        "table id" TEXT PRIMARY KEY,
        "prism name" TEXT,
        "merit function" TEXT,
        "deviation target" REAL,
        "dispersion target" REAL,
        json TEXT
        )""")

    c.execute(f"""CREATE TABLE {output_table_name} (
        {", ".join(f"glass{i} TEXT" for i in range(1, count+1))},
        {", ".join(f"alpha{i} REAL" for i in range(1, count+1))},
        meritError REAL,
        deviation REAL,
        dispersion REAL,
        NL REAL,
        SSR REAL,
        K REAL,
        deltaM REAL,
        delta1 REAL,
        delta2 REAL,
        nonlin REAL,
        chromat REAL
        )""")

    c.execute(f"""SELECT {", ".join(f"glass{i} TEXT" for i in range(1, count+1))} FROM {input_table_name}""")

    t1 = time.time()
    items = glass_paired(gcat, c.fetchall(), w)
    itemLock, outputLock = Lock(), Lock()
    insert_stmt = f"INSERT INTO {output_table_name} VALUES ({', '.join(repeat('?', 2 * count + 11))})"
    amounts = []

    def process(tid):
        count = 0
        thread_model = cmpnd.configure_thread(model, tid)
        while True:
            try:
                with itemLock:
                    glass, n = next(items)
            except StopIteration:
                break
            data = cmpnd(thread_model._replace(n=np.stack(n)))
            if data is not None:
                count += 1
                with outputLock:
                    c.execute(insert_stmt, (*glass, *data[0], *data[1:]))
                    if tid == 0 and count % 10000 == 0:
                        conn.commit()
                        print("Saved")
        amounts.append(count)

    threads = [Thread(target=process, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    amount = sum(amounts)

    dt = time.time() - t1
    print('Total glass combinations considered =', amount)
    print('Elapsed time for solution search in the full catalog =', dt, 'sec')
    print('Elapsed time per glass combination =', 1000 * dt / amount, 'ms')

    c.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?)",
              (output_table_name, prism_name, merit, deviation_target, dispersion_target, str(settings)))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    print('Starting Compound Prism Hyper Parameter Designer')
    with open(sys.argv[1], 'r') as f:
        settings = json.load(f)
    hyper(**settings)
