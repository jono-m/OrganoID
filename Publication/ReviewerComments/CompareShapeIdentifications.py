import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import scipy.stats
from Publication.Statistics import linr_ci, pearsonr_ci
import pandas
from Publication.PlottingUtils import CorrelationPlot

fontsize = 10
corrColor = [x / 255 for x in (0, 205, 108)]
lodColor = [x / 255 for x in (0, 154, 222)]
meanColor = [x / 255 for x in (0, 205, 108)]
hue = 343 / 360
colors = [colorsys.hsv_to_rgb(hue, sat, 1) for sat in np.linspace(0.1, 1, 4)] + \
         [colorsys.hsv_to_rgb(hue, 1, val) for val in np.linspace(0.8, 0, 3)]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

countData = pandas.read_csv(r"Publication\Figure2\2b-c\CountComparison.csv")
organoidData = pandas.read_csv(r"Publication\Figure2\2b-c\OrganoidComparison.csv")

PCimages = ["PDAC1", "PDAC3", "PDAC7", "PDAC8", "PDAC9", "PDAC10", "Lung3"]

PCorganoidData = organoidData[organoidData["Filename"].isin(PCimages)]
BForganoidData = organoidData[~organoidData["Filename"].isin(PCimages)]

toPlot = [("Circularity", "All")]

for i, feature in enumerate(["Circularity", "Solidity", "Eccentricity"]):
    ax = plt.subplot(1, 3, i + 1)
    ax.set_aspect('equal', adjustable='box')
    for di, (name, data, color) in enumerate(
            [("All", organoidData, "black"), ("BF", BForganoidData, lodColor),
             ("PC", PCorganoidData, colors[4])]):
        featuresData = data.pivot_table(index=["Filename", "ID"], columns=["Feature"])
        featureData = featuresData.xs(feature, axis=1, level=1)
        # For circularity, center 0-1 scores with x^n:
        if feature != "Eccentricity":
            factor = np.log(0.5) / np.log(featureData.mean().mean())
            featureData = np.power(featureData, factor)
        else:
            factor = 1
        featuresX = featureData["Manual"]
        featuresY = featureData["Automated"]
        ccc, lo, hi = linr_ci(featuresX, featuresY)
        plt.text(0.05, 0.95 - di * 0.05, "CCC %s=%.2f [95%% CI %.2f-%.2f]" % (name, ccc, lo, hi),
                 color=color)
        if name != "All":
            plt.scatter(featuresX, featuresY, color=color, label=name, s=5)
        plt.xlabel("Manual %s %d" % (feature, factor))
        plt.ylabel("Automated %s %d" % (feature, factor))
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.plot([0, 1], [0, 1], color=meanColor)
plt.show()
