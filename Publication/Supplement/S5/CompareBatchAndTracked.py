import matplotlib.pyplot as plt
import colorsys
import numpy as np
from pathlib import Path
import pandas
import seaborn
import scipy.optimize

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

soData = pandas.read_csv(Path(r"Publication\Figure3\3c\Data.csv"))

dosages = np.unique(soData["Dosage"])
soData = soData.set_index(["Dosage", "Replicate", "Organoid ID", "Time"])
soData["Fluorescence Per Area"] = soData["Masked fluorescence"] / soData["Area"]


def CompareBatchAndTracked(column):
    columnData = soData[column]

    plt.subplot(1, 2, 1)
    batchSEMs = []
    for color, dosage in zip(colors, dosages):
        dosageData = columnData[columnData.index.get_level_values("Dosage") == dosage]
        batchData = dosageData / dosageData.groupby("Time").mean()[0]
        batchData = batchData.groupby("Time")
        plt.errorbar(batchData.mean().index, batchData.mean(), yerr=batchData.sem(),
                     color=color, label="%d nM" % dosage)
        batchSEMs += list(batchData.sem())
    plt.xlabel("Time")
    plt.ylabel(column + " (Fold change)")

    plt.subplot(1, 2, 2)
    trackedSEMs = []
    for color, dosage in zip(colors, dosages):
        dosageData = columnData[columnData.index.get_level_values("Dosage") == dosage]
        trackedData = dosageData / dosageData.groupby(
            ["Dosage", "Replicate", "Organoid ID"]).first()
        trackedData = trackedData.groupby("Time")
        plt.subplot(1, 2, 2)
        plt.errorbar(trackedData.mean().index, trackedData.mean(), yerr=trackedData.sem(),
                     color=color, label="%d nM" % dosage)
        trackedSEMs += list(trackedData.sem())
    plt.xlabel("Time")
    plt.ylabel(column + " (Fold change)")
    print("Batch SEM: %f\nTracked: %f\n" % (np.mean(batchSEMs), np.mean(trackedSEMs)))


CompareBatchAndTracked("Circularity")
plt.show()
