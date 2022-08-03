import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
from Publication.Statistics import pearsonr_ci, linr_ci
import matplotlib.pyplot as plt
import numpy as np

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


def BuildCorrPlot(data: np.ndarray, label0: str, label1: str, pointName: str,
                  metricName: str, units: str):
    if units != "":
        units = " (" + units + ")"
    ccc, loc, hic = linr_ci(data[:, 0], data[:, 1])
    r, p, lo, hi = pearsonr_ci(data[:, 0], data[:, 1])
    print(metricName + str((r, p, lo, hi)))
    maxData = np.max(data)
    plt.subplots(1, 2, figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(data[:, 1], data[:, 0], marker='o', s=5, color='k')
    plt.xlabel(label1 + units)
    plt.ylabel(label0 + units)
    plt.text(0, maxData, "$CCC=%.2f$\n[%.2f-%.2f]" % (ccc, loc, hic),
             verticalalignment='top',
             size=fontsize,
             color=corrColor)
    plt.plot([0, maxData], [0, maxData], "-", color=corrColor)

    plt.subplot(1, 2, 2)
    means = np.mean(data, axis=1)
    differences = data[:, 1] - data[:, 0]
    plt.scatter(means, differences, marker='o', s=5, color='k')
    plt.legend([pointName])

    meanDifference = np.mean(differences)
    stdDifference = np.std(differences)
    labels = [(meanDifference, "Mean=%.2f" % meanDifference, meanColor),
              (meanDifference - stdDifference * 1.96,
               "-1.96\u03C3=%.2f" % (meanDifference - stdDifference * 1.96), lodColor),
              (meanDifference + stdDifference * 1.96,
               "+1.96\u03C3=%.2f" % (meanDifference + stdDifference * 1.96), lodColor)]
    plt.axhline(y=meanDifference, color=meanColor, linestyle="solid")
    plt.axhline(y=meanDifference + stdDifference * 1.96, color=lodColor, linestyle="dashed")
    plt.axhline(y=meanDifference - stdDifference * 1.96, color=lodColor, linestyle="dashed")
    plt.xlabel(metricName + " average" + units)
    plt.ylabel(metricName + " difference" + units)
    ylim = np.max(np.abs([min(differences) - 2, max(differences) + 2]))
    plt.ylim([-ylim, ylim])
    for (y, text, color) in labels:
        plt.text(np.max(means), y, text, verticalalignment='bottom', horizontalalignment='right',
                 color=color,
                 size=fontsize)


countFile = open(r"Publication\ReviewerComments\MouseOrganoids\CountComparison.csv", "r")
counts = np.asarray(
    [[float(x) for x in line.split(",")[1:]] for line in countFile.read().split("\n")[1:]])

areasFile = open(r"Publication\ReviewerComments\MouseOrganoids\AreaComparison.csv", "r")
areas = np.asarray(
    [[float(x) for x in line.split(",")] for line in areasFile.read().split("\n")[1:]]) / 1000

BuildCorrPlot(counts, "Manual count", "OrganoID count", "Organoid image", "Count", "")
BuildCorrPlot(areas, r"Manual area", "OrganoID area", "Organoid", "Area", r"x $10^3 \mu m^2$")
plt.show()
