import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import pandas
from Publication.PlottingUtils import TimecoursePlot

fontsize = 10
corrColor = [x / 255 for x in (0, 205, 108)]
lodColor = [x / 255 for x in (0, 154, 222)]
meanColor = [x / 255 for x in (0, 205, 108)]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

mtsData = pandas.read_csv(r"Publication\Figure3\MTSAssay.csv").set_index(["Dosage", "Replicate"])
organoidData = pandas.read_csv(r"Publication\Figure3\OrganoidMeasurements.csv").set_index(
    ["Dosage", "Replicate", "Organoid ID", "Time"])
piData = pandas.read_csv(r"Publication\Figure3\PIMeasurements.csv").set_index(
    ["Dosage", "Replicate", "Time"])


def FoldChangeFromInitial(df: pandas.Series):
    initialResponse = df.reset_index().query("Time == 0").set_index(
        ["Dosage", "Replicate"])[df.name]
    return (df / initialResponse).rename(str(df.name) + " (fold change)")


def SubtractNegativeControl(df: pandas.Series):
    negativeResponse = df.reset_index().query("Dosage == 0").groupby("Time").mean()[df.name]
    responses = df[df.index.get_level_values("Dosage") != 0]
    return responses - negativeResponse


initialFeatures = organoidData.reset_index().sort_values("Time").groupby(
    ["Dosage", "Replicate", "Organoid ID"]).first().drop(columns="Time")

organoidCount = organoidData.groupby(["Dosage", "Replicate", "Time"]).count()["Area"].rename(
    "Count")
trackedOrganoidData = organoidData / initialFeatures

maskedFluorescence = organoidData["Fluorescence"].groupby(piData.index.names).sum().rename(
    "Masked fluorescence")
organoidArea = organoidData["Area"].groupby(piData.index.names).sum().rename("Total organoid area")
maskedFluorescencePerArea = (maskedFluorescence / organoidArea).rename(
    "Masked fluorescence per total area")

trackedFPA = trackedOrganoidData["Fluorescence"] / trackedOrganoidData["Area"]
trackedFPA = trackedFPA.rename("Organoid fluorescence/area (FC)")
fig, axes = plt.subplots(2, 2, sharex='all')
axes = axes.flatten()
TimecoursePlot(trackedOrganoidData["Area"].rename("Area (FC)"), axes[0])
TimecoursePlot(SubtractNegativeControl(piData["Fluorescence"] / 100000), axes[1])
TimecoursePlot(FoldChangeFromInitial(organoidCount), axes[2])
TimecoursePlot(SubtractNegativeControl(trackedFPA), axes[3])
axes[1].legend()
[ax.set_xlabel("") for ax in axes[:2]]
plt.show()
