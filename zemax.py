from win32com.client.gencache import EnsureDispatch, EnsureModule
from win32com.client import CastTo, constants
from scipy.interpolate import InterpolatedUnivariateSpline, SmoothBivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import math

# Initial Varibles
prism_count = 3
prism_height = 15
thickness = []
materials = []
ytans = []

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
system = application.CreateNewSystem(constants.SystemType_Sequential)
if system is None:
    raise Exception("Unable to create new system")
print('connected')
system.SaveAs("cmpndprism.zmx")
xTanCell = constants.SurfaceColumn_Par1
yTanCell = constants.SurfaceColumn_Par2

# Create Optical System
lde = system.LDE
for _ in range(prism_count):
    lde.InsertNewSurfaceAt(1)
assert len(thickness) == len(materials) == len(ytans) - 1
for (i, (thick, glass, ytan)) in enumerate(zip(thickness, materials, ytans)):
    # Prism Surfaces
    surf = lde.GetSurfaceAt(1 + i)
    surf.Thickness = thick
    surf.Material = glass
    surf.SemiDiameter = prism_height / 2
    surf.GetSurfaceCell(xTanCell).Value = 0
    surf.GetSurfaceCell(yTanCell).Value = ytan
# Sphereical Prism Surface

# Clean up
system.Save()
application.CloseApplication()
