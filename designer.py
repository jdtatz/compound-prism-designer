from glasscat import calc_w, read_glasscat, glass_combos
from multiprocessing import cpu_count
from threading import Thread, Lock
import time, sys, json, sqlite3
from prisms import prism_setup
from queue import Queue
from uuid import uuid4
from math import pi
import numpy as np

print('Starting Compound Prism Designer')

with open(sys.argv[1], 'r') as f:
    settings = json.load(f)
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
    angle_limit = settings.get("incident angle limit", 65.0)
    theta0 = settings.get("theta0", 0.0)
    initial_angles = settings.get("initial angles", None)
    database_file = settings.get("database file", "prism.db")
    table_name = settings.get("table name", "t" + uuid4().hex)
    thread_count = settings.get("thread count", cpu_count())

if count == 0:
    count = max(indices) + 1
    prism_count = len(indices)
elif indices is None:
    prism_count = count
    indices = tuple(range(count))
else:
    raise Exception("Invalid prism parametrization")

if w is not None:
    w = np.array(w)
    wavemin = w.min()
    wavemax = w.max()
    nwaves = w.size
else:
    w = calc_w(wavemin, wavemax, nwaves, sampling_domain)

gcat = read_glasscat(catalog_filename)
nglass = len(gcat)
# in radians
deltaC_target = deviation_target * pi / 180.0
deltaT_target = dispersion_target * pi / 180.0

print('table:', table_name)
print('# glasses =', nglass)
print('targets: deltaC = %5.2f deg, deltaT = %5.2f deg' % (deviation_target, dispersion_target))
print('thread count:', thread_count, '\n')
cmpnd = prism_setup(prism_count, count, indices, deltaC_target, deltaT_target, merit, weights,
                    nwaves, sampling_domain, theta0, initial_angles, angle_limit * pi / 180.0)

conn = sqlite3.connect(database_file)
c = conn.cursor()

c.execute("""CREATE TABLE IF NOT EXISTS setups (
    "table id" TEXT PRIMARY KEY,
    "prism name" TEXT,
    "merit function" TEXT,
    "deviation target" REAL,
    "dispersion target" REAL,
    "sampling domain" TEXT,
    "wave minimum" REAL,
    "wave maximum" REAL,
    "number of waves" INTEGER,
    "angle limit" REAL,
    catalog TEXT
    )""")

c.execute("""CREATE TABLE {} (
    {}
    {}
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
    )""".format(table_name,
                (count * "glass{} TEXT, ").format(*range(1, count + 1)),
                (count * "alpha{} REAL, ").format(*range(1, count + 1))))

t1 = time.time()
outQueue = Queue()
items = glass_combos(gcat, count, w, wavemin, wavemax)
lock = Lock()


def process():
    while True:
        try:
            with lock:
                p = next(items)
        except StopIteration:
            break
        glass, n = p
        outQueue.put((glass, cmpnd(np.array(n), w)))
    outQueue.put(None)

for t in (Thread(target=process) for _ in range(thread_count)):
    t.start()

insert_stmt = "INSERT INTO {} VALUES ({} ?)".format(table_name, "?, " * (2 * count + 10))
amount, ncount = 0, 0
while ncount < thread_count:
    item = outQueue.get()
    if item is None:
        ncount += 1
    else:
        gls, data = item
        if data is not None:
            c.execute(insert_stmt, (*gls, *data[0], *data[1:]))
        amount += 1
    if amount % 500000 == 0:
        conn.commit()
        print("Saved at", amount, "items")

dt = time.time() - t1
print('Total glass combinations considered =', amount)
print('Elapsed time for solution search in the full catalog =', dt, 'sec')
print('Elapsed time per glass combination =', 1000 * dt / amount, 'ms')

c.execute("INSERT INTO setups VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
          (table_name, prism_name, merit, deviation_target, dispersion_target,
           sampling_domain, wavemin, wavemax, nwaves, angle_limit, catalog_filename))
conn.commit()
conn.close()
