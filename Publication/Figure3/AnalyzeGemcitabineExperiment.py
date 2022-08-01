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
plt.rcParams['font.sans-serif'] =q ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

mtsData = pandas.read_csv(r"Publication\Figure3\MTSAssay.csv").set_index(["Dosage", "Replicate"])
organoidData = pandas.read_csv(r"Publication\Figure3\OrganoidData.csv").set_index(
    ["Dosage", "Replicate", "Organoid ID", "Time"])
piData = pandas.read_csv(r"Publication\Figure3\PIData.csv").set_index(
    ["Dosage", "Replicate", "Time"])


def FoldChangeFromInitial(df: pandas.Series):
    initialResponse = df.reset_index().query("Time == 0").set_index(
        ["Dosage", "Replicate"])[df.name]
    return (df / initialResponse).rename(str(df.name) + " (fold change)")


def SubtractNegativeControl(df: pandas.Series):
    negativeResponse = df.reset_index().query("Dosage == 0").groupby("Time").mean()[df.name]
    responses = df[df.index.get_level_values("Dosage") != 0]
    return (responses - negativeResponse).rename(str(df.name) + " (change from control)")


maskedFluorescence = organoidData["Fluorescence"].groupby(piData.index.names).sum().rename(
    "Masked fluorescence")
organoidArea = organoidData["Area"].groupby(piData.index.names).sum().rename("Total organoid area")
maskedFluorescencePerArea = (maskedFluorescence / organoidArea).rename(
    "Masked fluorescence per total area")

singleOrganoidFluorescence = organoidData["Fluorescence"].rename("Organoid fluorescence")
singleOrganoidArea = organoidData["Area"].rename("Organoid area")
singleOrganoidFluorescencePerArea = (singleOrganoidFluorescence / singleOrganoidArea).rename(
    "Organoid fluorescence per area")

fig, axes = plt.subplots(2, 4, sharex='all')
(ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7) = axes.flatten()
TimecoursePlot(SubtractNegativeControl(piData["Fluorescence"]), ax0)
TimecoursePlot(maskedFluorescence, ax1)
TimecoursePlot(FoldChangeFromInitial(organoidArea), ax2)
TimecoursePlot(SubtractNegativeControl(maskedFluorescencePerArea), ax3)
TimecoursePlot(singleOrganoidFluorescence, ax4)
TimecoursePlot(singleOrganoidArea, ax5)
TimecoursePlot(SubtractNegativeControl(singleOrganoidFluorescencePerArea), ax6)
# TimecoursePlot(Prep(fpa), ax3)
# TimecoursePlot(Prep(maskedFPA), ax4)
# TimecoursePlot(normalizedTrackedOrganoidData["Area"], ax5)
ax0.legend()
[ax.set_xlabel("")
 for ax in [ax0, ax1, ax2]]
# TimecoursePlot(piData["Norm fluorescence"], axes[1], foldChange)
# TimecoursePlot(piData["Fold area"], axes[2], foldChange)
# TimecoursePlot(piData["Fold fluorescence"], axes[3], foldChange)
# TimecoursePlot((piData["Fluorescence"] / piData["Area"]).rename("FPA"), axes[2], foldChange)
# TimecoursePlot((piData["Masked fluorescence"] / piData["Area"]).rename("Masked FPA"), axes[3],
#                foldChange)
# TimecoursePlot(organoidData["Circularity"], axes[0], foldChange)
# TimecoursePlot(organoidData["Eccentricity"], axes[1], foldChange)
# TimecoursePlot(organoidData["Solidity"], axes[2], foldChange)
# TimecoursePlot(organoidData["Area"], axes[3], foldChange)
# TimecoursePlot(normalizedOrganoidData["Circularity"], axes[0], foldChange)
# TimecoursePlot(normalizedOrganoidData["Eccentricity"], axes[1], foldChange)
# TimecoursePlot(normalizedOrganoidData["Solidity"], axes[2], foldChange)
# TimecoursePlot(normalizedOrganoidData["Area"], axes[3], foldChange)
# TimecoursePlot(foldChangeOrganoidData["Circularity"], axes[0], foldChange)
# TimecoursePlot(foldChangeOrganoidData["Eccentricity"], axes[1], foldChange)
# TimecoursePlot(foldChangeOrganoidData["Solidity"], axes[2], foldChange)
# TimecoursePlot(foldChangeOrganoidData["Area"], axes[3], foldChange)
plt.show()
