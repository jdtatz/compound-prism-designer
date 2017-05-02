from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
import matplotlib.pyplot as plt
import numpy as np
import math

def temp(line):
    '''ignore'''
    t = line.split()
    l = len(t)//2
    return (*t[:l], *map(float, t[l:]))

def calc(materials, alphas):
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
    return ytans, thickness

# setup values
h = 50

g1 = 'N-PK52A'
g2 = 'P-SF68'
g3 = 'N-FK58'

a1 = 107.02325732201899
a2 = -63.57538439773579
a3 = 54.3416288226025

materials = (g1, g2, g3, g2, g1)
alphas = np.array((a1, a2, a3, a2, a1)) * math.pi / 180

dataDir = 'C:\\Users\\Aki\\Desktop\\prism\\'
baseZmx = 'prism_compat_0p4_NA.ZMX'
outZmx = 'prism.ZMX'
outFile = 'out.txt'

# calculations
ytans, thickness = calc(materials, alphas)
count = len(materials)
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
system.LoadFile(dataDir + baseZmx, False)
system.SaveAs(dataDir + outZmx)

# Wave data
waves = system.SystemData.Wavelengths
fields = system.SystemData.Fields

# Find and create prism surfaces
start, scount = 10000, 0
for i in range(system.LDE.NumberOfSurfaces):
    surf = system.LDE.GetSurfaceAt(i+1)
    if surf.TypeName == 'Tilted':
        start = min(start, i)
        scount += 1

assert scount > 1, "Need at least two tilted surfaces to start"

if count + 1 < scount:
    assert scount-count+1 == system.LDE.RemoveSurfacesAt(start+2, scount-count+1), 'Surfaces not removed correctly'
elif count +1 > scount:
    for i in range(1+count-scount):
        assert 1 == system.LDE.CopySurfaces(start+1, 1, start+2), 'Surface not copied correctly'

# Lens data
xTanCell = constants.SurfaceColumn_Par1
yTanCell = constants.SurfaceColumn_Par2

for i in range(count):
    surf = system.LDE.GetSurfaceAt(start+1+i)
    surf.Thickness = thickness[i]
    surf.Material = materials[i]
    surf.SemiDiameter = h / 2
    surf.GetSurfaceCell(xTanCell).Value = 0
    surf.GetSurfaceCell(yTanCell).Value = ytans[i]

lastSurf = system.LDE.GetSurfaceAt(start+1+count)
lastSurf.SemiDiameter = h / 2
lastSurf.GetSurfaceCell(xTanCell).Value = 0
lastSurf.GetSurfaceCell(yTanCell).Value = ytans[-1]

# Spot data
wdata = []
wave = waves.AddWavelength(0.500, 1.0)
wnum = waves.NumberOfWavelengths
for w in np.arange(500, 815, 0.1):
    wave.Wavelength = w/1000
    spot = system.Analyses.New_StandardSpot()
    spot_settings = CastTo(spot.GetSettings(), 'IAS_Spot')
    spot_settings.Wavelength.SetWavelengthNumber(wnum)
    for j in range(1, 1 + fields.NumberOfFields):
        spot_settings.Field.SetFieldNumber(j)
        ret = spot.ApplyAndWaitForCompletion()
        results = spot.GetResults()
        y = float(next(filter(lambda l: l.startswith('Image coord'), results.HeaderData.Lines)).split()[-1])
        f = fields.GetField(j).FieldNumber
        wdata.append((w, f, y))
    spot.Close()
wdata = np.array(wdata)
np.savetxt(outFile, wdata, fmt=('%g', '%u', '%g'), delimiter=',', newline='\r\n', header='wavelength(nm),field#,pmt position(mm)', comments='')

# PMT bins
bins = np.arange(math.floor(wdata[:, 2].min()), math.ceil(wdata[:, 2].max()))
digi = np.digitize(wdata[:, 2], bins, right=True)
binned = {b: wdata[digi==i+1, ::2] for i, b in enumerate(bins)}
inbin = {b: data[(b+0.1 < data[:, 1]) & (data[:, 1] < b+0.9)] for b, data in binned.items()}
with open("pmtBins.txt", 'w') as fp:
    fp.write("Bin# (pos1(mm), pos2(mm)): from wave1(nm) to wave2(nm)\n")
    for i, b in enumerate(range(-16, 16)):
        w0, wl = inbin[b][[0, -1], 0]
        fp.write(f"Bin{i+1} ({b+0.1:.1f}, {b+0.9:.1f}): from {wl:.1f} to {w0:.1f}\n")

# Save and close
system.Save()
application.CloseApplication()

# Plot
plt.plot(wdata[:, 2], wdata[:, 0])
plt.ylabel('Wavelength(nm)')
plt.xlabel('Spot Position(mm)')
plt.title('Linearity')
plt.show()
