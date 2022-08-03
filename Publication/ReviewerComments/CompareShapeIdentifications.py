import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from Publication.Statistics import linr_ci
import pandas
from Publication.PlottingUtils import CorrelationPlot

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

countData = pandas.read_csv(r"Publication\Figure2\2b-c\CountComparison.csv")
organoidData = pandas.read_csv(r"Publication\Figure2\2b-c\OrganoidComparison.csv")

PCimages = ["PDAC1", "PDAC3", "PDAC7", "PDAC8", "PDAC9", "PDAC10", "Lung3"]

PCorganoidData = organoidData[organoidData["Filename"].isin(PCimages)]
BForganoidData = organoidData[~organoidData["Filename"].isin(PCimages)]

CorrelationPlot(countData, "Manual count", "OrganoID count", "Image", "Count", "")

toPlot = [("Area", "All")]
for name, data in [("All", organoidData), ("PC", PCorganoidData), ("BF", BForganoidData)]:
    print("%s:" % name)
    featuresData = data.pivot_table(index=["Filename", "ID"], columns=["Feature"])
    for feature in ["Circularity", "Solidity", "Eccentricity", "Perimeter", "Area"]:
        featureData = featuresData.xs(feature, axis=1, level=1)
        featuresX = featureData["Manual"]
        featuresY = featureData["Automated"]
        ccc, loc, hic = linr_ci(featuresX, featuresY)
        print("\t%s\t%f (95%% CI %f-%f)" % (feature, ccc, loc, hic))

        if (feature, name) in toPlot:
            CorrelationPlot(featureData, "Manual " + feature, "OrganoID " + feature, "Organoid",
                            feature, "")
plt.show()
