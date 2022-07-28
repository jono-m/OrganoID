import matplotlib.pyplot as plt
import colorsys
import numpy as np
from pathlib import Path

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

dataFile = open(Path(r"Publication\Figure3\3b\Data.csv"), mode="r")
data = np.asarray([[float(y) for y in x.split(",")] for x in dataFile.read().split("\n")[1:]])

dosages = np.unique(data[:, 0])
replicates = np.unique(data[:, 1])
times = np.unique(data[:, 2])


def GetData(d: np.ndarray, column, dosage):
    dosageData = d[d[:, 0] == dosage]
    return np.stack(
        [dosageData[dosageData[:, 1] == replicate][:, column] for replicate in replicates], axis=1)


def Plot():
    fig, axes = plt.subplots(2, 2, sharex='all')
    for color, dosage in zip(colors[1:], dosages[1:]):
        label = "%d nM" % dosage
        counts = GetData(data, 3, dosage)
        areas = GetData(data, 4, dosage)
        totalFluorescences = GetData(data, 5, dosage)
        maskedTotalFluorescences = GetData(data, 6, dosage)

        negativeControlFPA = GetData(data, 6, 0) / GetData(data, 4, 0)
        fpa = maskedTotalFluorescences / areas - np.mean(negativeControlFPA, axis=1)[:, None]

        totalFluorescences = totalFluorescences - np.mean(GetData(data, 5, 0), axis=1)[:, None]
        totalFluorescences = totalFluorescences / 100000

        areas = areas / areas[:1, :]
        counts = counts / counts[:1, :]
        plotArrangement = [(axes[0, 0], areas, "Organoid area\n(fold change)"),
                           (axes[1, 0], counts, "Number of organoids\n(fold change)"),
                           (axes[0, 1], totalFluorescences, "Fluorescence intensity\n(a.u.)"),
                           (axes[1, 1], fpa,
                            "Fluorescence intensity per area\n(a.u.)")]
        for axis, toPlot, yLabel in plotArrangement:
            axis.errorbar(np.arange(0, 73, 4), np.mean(toPlot, axis=1),
                          yerr=np.std(toPlot, axis=1) / np.sqrt(toPlot.shape[1]),
                          label=label, color=color)
            axis.set_ylabel(yLabel)

    plt.xticks(np.arange(0, 73, 12))
    axes[0, 1].legend()
    plt.xlabel("Time (hours)")
    plt.show()


Plot()
