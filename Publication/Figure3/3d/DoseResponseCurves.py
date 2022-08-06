import matplotlib.pyplot as plt
import colorsys
import numpy as np
from pathlib import Path
import pandas
import scipy.optimize
from Publication.PlottingUtils import DoseReponsePlot

fontsize = 10
corrColor = [x / 255 for x in (0, 205, 108)]
meanColor = [x / 255 for x in (0, 154, 222)]
lodColor = [x / 255 for x in (255, 31, 91)]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
hue = 343 / 360
colors = [colorsys.hsv_to_rgb(hue, sat, 1) for sat in np.linspace(0.1, 1, 4)] + \
         [colorsys.hsv_to_rgb(hue, 1, val) for val in np.linspace(0.8, 0, 3)]

t = 52
mtsData = pandas.read_csv(r"Publication\Figure3\MTSAssay.csv").set_index(["Dosage", "Replicate"])
organoidData = pandas.read_csv(r"Publication\Figure3\OrganoidMeasurements.csv").set_index(
    ["Time", "Dosage", "Replicate", "Organoid ID"])
piData = pandas.read_csv(r"Publication\Figure3\PIMeasurements.csv").set_index(
    ["Time", "Dosage", "Replicate"])

initialFeatures = organoidData.reset_index().sort_values("Time").groupby(
    ["Dosage", "Replicate", "Organoid ID"]).first().drop(columns="Time")
trackedOrganoidData = organoidData / initialFeatures
trackedOrganoidData = trackedOrganoidData.reset_index().set_index(
    ["Time", "Dosage", "Replicate", "Organoid ID"])
trackedFPA = trackedOrganoidData["Fluorescence"] / trackedOrganoidData["Area"]
trackedFPA = trackedFPA.rename("Organoid fluorescence/area (FC)")
trackedFPA = trackedFPA.loc[t]


def FoldChange(df: pandas.Series, from_max: bool = True):
    baseResponse = df.loc[1000 if from_max else 0].mean()
    return (100 * df / baseResponse).rename(
        str(df.name) + " (%% %s. control)" % ("pos" if from_max else "neg"))


organoidData = organoidData.loc[t]
piData = piData.loc[t]
fluorescence = piData["Fluorescence"]
mtsViability = mtsData["Viability"]

organoidArea = organoidData.groupby(["Dosage", "Replicate"]).sum()["Area"]
organoidCount = organoidData.groupby(["Dosage", "Replicate"]).count()["Area"].rename("Count")
maskedFluorescence = organoidData.groupby(["Dosage", "Replicate"]).sum()[
    "Fluorescence"].rename(
    "Masked fluorescence")
mtsNormalized = (piData["Fluorescence"] / mtsData["Viability"]).rename(
    "MTS-normalized fluorescence")
areaNormalized = (maskedFluorescence / organoidArea).rename("Area-normalized fluorescence")


def SubtractNegativeControl(df: pandas.Series):
    negativeResponse = df.reset_index().query("Dosage == 0").groupby("Time").mean()[df.name]
    responses = df[df.index.get_level_values("Dosage") != 0]
    return responses - negativeResponse


fig, axes = plt.subplots(4, 1, sharex='all')
axes = axes.flatten()
DoseReponsePlot(FoldChange(mtsViability, False), axes[0], [0, .7, 0])
DoseReponsePlot(FoldChange(organoidArea, False), axes[1], (0, .7, 0))
DoseReponsePlot(FoldChange(fluorescence), axes[2])
DoseReponsePlot(FoldChange(trackedFPA), axes[3])
axes[3].set_xlabel("Gemcitabine dosage (nM)")

_, axes = plt.subplots(1, 3)
axes = axes.flatten()
DoseReponsePlot(FoldChange(organoidData["Circularity"], False), axes[0], [0, .7, 0])
DoseReponsePlot(FoldChange(organoidData["Solidity"], False), axes[1], (0, .7, 0))
DoseReponsePlot(FoldChange(organoidData["Eccentricity"]), axes[2])

[ax.set_xlabel("Gemcitabine dosage (nM)") for ax in axes]
plt.show()
