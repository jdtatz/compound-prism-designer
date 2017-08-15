import ctypes as ct
from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from scipy.interpolate import InterpolatedUnivariateSpline, SmoothBivariateSpline
from scipy.optimize import nnls
import matplotlib.pyplot as plt
import numpy as np
import math

# setup values
h = 50
'''
g1 = 'N-FK51A'
g2 = 'P-SF68'

a1 = 99.1128327540148
a2 = -84.8784614978487

materials = (g1, g2, g1)
alphas = np.array((a1, a2, a1)) * math.pi / 180
'''

g1 = 'N-FK58'
g2 = 'P-SF68'
g3 = 'LF5'

a1 = 102.522545748073
a2 = -48.2729540140781
a3 = 17.502053458746

materials = (g1, g2, g3, g2, g1)
alphas = np.array((a1, a2, a3, a2, a1)) * math.pi / 180

dataDir = 'C:\\Users\\Admin\\Desktop\\prism\\'
baseZmx = 'prism_compat_0p4_NA_125um.ZMX'
outZmx = 'prism.ZMX'
outFile = 'out.txt'

# calculations
count = len(materials)
ytans, thickness = [], []

for i in range(count):
    mid = count // 2
    beta = alphas[mid] / 2
    if i < mid:
        beta += np.sum(alphas[i:mid])
    elif i > mid:
        beta += np.sum(alphas[mid+1:i+1])
    gamma = alphas[i] - beta
    ytanbeta = math.tan(beta)
    ytangamma = math.tan(gamma)
    thickness.append((h / 2) * (abs(ytanbeta) + abs(ytangamma)))
    if i <= mid:
        ytans.append(ytanbeta)
    if i >= mid:
        ytans.append(-ytanbeta)
print(ytans)
print(thickness)

# ZOS-API Connection
EnsureModule('ZOSAPI_Interfaces', 0, 1, 0)
connection = EnsureDispatch("ZOSAPI.ZOSAPI_Connection")
if connection is None:
    raise Exception("Unable to intialize COM connection to ZOSAPI")
application = connection.CreateNewApplication()
if application is None:
    raise Exception("Unable to acquire ZOSAPI application")
if not application.IsValidLicenseForAPI:
    raise Exception("License is not valid for ZOSAPI use")
system = application.PrimarySystem
if system is None:
    raise Exception("Unable to acquire Primary system")
print('connected')

# Load base file and mark save file
print(dataDir + baseZmx)
system.LoadFile(dataDir + baseZmx, False)
system.SaveAs(dataDir + outZmx)

# Wave data
waves = system.SystemData.Wavelengths
fields = system.SystemData.Fields

# Find and create prism surfaces
start, scount = 10000, 0
for i in range(1, 1+system.LDE.NumberOfSurfaces):
    surf = system.LDE.GetSurfaceAt(i)
    if surf.TypeName == 'Tilted':
        start = min(start, i)
        scount += 1

assert scount > 1, "Need at least two tilted surfaces to start"

if count + 1 < scount:
    print("Removing Excess Surfaces")
    assert scount-count-1 == system.LDE.RemoveSurfacesAt(start+1, scount-count-1), 'Surfaces not removed correctly'
elif count +1 > scount:
    print("Adding More Surfaces")
    for i in range(1+count-scount):
        assert 1 == system.LDE.CopySurfaces(start, 1, start+1), 'Surface not copied correctly'
else:
    print("Correct Number of Surfaces")
    
# Lens data
xTanCell = constants.SurfaceColumn_Par1
yTanCell = constants.SurfaceColumn_Par2

for i in range(count):
    surf = system.LDE.GetSurfaceAt(start+i)
    surf.Thickness = thickness[i]
    surf.Material = materials[i]
    surf.SemiDiameter = h / 2
    surf.GetSurfaceCell(xTanCell).Value = 0
    surf.GetSurfaceCell(yTanCell).Value = ytans[i]
    assert CastTo(surf, 'IEditorRow').IsValidRow

lastSurf = system.LDE.GetSurfaceAt(start+count)
lastSurf.SemiDiameter = h / 2
lastSurf.GetSurfaceCell(xTanCell).Value = 0
lastSurf.GetSurfaceCell(yTanCell).Value = ytans[-1]
assert CastTo(lastSurf, 'IEditorRow').IsValidRow

# Optimize
optim = system.Tools.OpenLocalOptimization()
print("Initial Merit:", optim.InitialMeritFunction)
CastTo(optim, 'ISystemTool').RunAndWaitForCompletion()
print("Final Merit:", optim.CurrentMeritFunction)
CastTo(optim, 'ISystemTool').Close()

# Ray Trace data

# * Pooled Ray Trace
wnum = waves.NumberOfWavelengths
nfs = np.linspace(-1, 1, 101)
ray = system.Tools.OpenBatchRayTrace()
pool = ray.CreateNormUnpol(len(nfs) * wnum, 0, system.LDE.NumberOfSurfaces)
pool.ClearData()
assert all(pool.AddRay(i, 0, hy, 0, 1, False) for i in range(1, 1+wnum) for hy in nfs), "Ray not batched correctly"
assert CastTo(ray, 'ISystemTool').RunAndWaitForCompletion(), "Bat Ray processing failed"
wdata = np.array([(1000*waves.GetWavelength(i).Wavelength, j, pool.ReadNextResult()[5]) for i in range(1, 1+wnum) for j in nfs])
CastTo(ray, 'ISystemTool').Close()
np.savetxt(outFile, wdata, fmt=('%g', '%g', '%g'), delimiter=',', newline='\r\n', header='wavelength(nm),field pos,pmt position(mm)', comments='')

# How well it unmixes
test = np.linspace(500, 820, 100)
# * How much is abosrbed
qe_curve = np.loadtxt("h7260_20_qe.csv", delimiter=',')
qe = InterpolatedUnivariateSpline(qe_curve[:, 0], qe_curve[:, 1] / 100, k=1)
# * How is it binned
pw2f = SmoothBivariateSpline(wdata[:, 2], wdata[:, 0], wdata[:, 1])
dFields = np.array([np.where(wdata[:, 1] == f)[0] for f in nfs])
ws, ps = wdata[dFields, 0], wdata[dFields, 2]
p2w = {f: InterpolatedUnivariateSpline(p[np.argsort(p)], w[np.argsort(p)], k=3) for f, w, p in zip(nfs, ws, ps)}
bins = np.array([(b+0.1, b+0.9) for b in range(-16, 16)])
wbins = np.array([p2w[f](bins) for f in nfs]).T
tot_in = np.transpose((np.maximum.reduce(np.minimum.reduce(wbins, 2), 0), np.minimum.reduce(np.maximum.reduce(wbins, 2), 0)))
som_in = np.transpose((np.maximum.reduce(np.maximum.reduce(wbins, 2), 0), np.minimum.reduce(np.minimum.reduce(wbins, 2), 0)))
bounds = np.hstack((bins, tot_in, som_in))

@np.vectorize
def power(w, val, l, r, lt, rt, ls, rs):
    if min(lt, rt) < w < max(lt, rt):
        return val
    elif min(ls, lt) < w < max(ls, lt):
        return val * float(np.abs(1 - pw2f(l, w)) / 2)
    elif min(rs, rt) < w < max(rs, rt):
        return val * float(np.abs(-1 - pw2f(r, w)) / 2)
    else:
        return 0.0

binned = np.apply_along_axis(lambda b: power(test, qe(test), *b), 1, bounds)
# * How much was emitted
dyes = np.load("dyes.npz")
basis = {dye: InterpolatedUnivariateSpline(dyes[dye][0], dyes[dye][2]/dyes[dye][2].max(), k=1) for dye in dyes.keys()}
mix = {}
for dye in dyes.keys():
    mixed = np.apply_along_axis(lambda b: b[b!=0].mean() if np.any(b!=0) else 0, 1, binned * basis[dye](test))
    mix[dye] = mixed / (mixed.max() or 1)
A = np.transpose([*mix.values()])


def poissionNoise(waves, basis, amount):
    mean = qe(waves)*basis(waves)*1024
    variance = np.sqrt(2*mean)
    alpha = variance / np.where(mean != 0, mean, 1.0)
    return (np.random.poisson(mean / np.where(alpha != 0, alpha, 1.0), (amount, lam.size)) * alpha).T


# Save and close
system.Save()
application.CloseApplication()

# Plot
wdata = wdata[np.argsort(wdata[:, 0])]
plt.plot(wdata[:, 2], wdata[:, 0])
plt.ylabel('Wavelength(nm)')
plt.xlabel('Spot Position(mm)')
plt.title('Linearity')
plt.savefig("LinearityFigure.png")
plt.show()
