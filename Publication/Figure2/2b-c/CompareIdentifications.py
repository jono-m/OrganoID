import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
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
CorrelationPlot(countData, "Manual count", "OrganoID count", "Image", "Count", "")

featuresData = organoidData.pivot_table(index=["Filename", "ID"], columns=["Feature"])
featureData = featuresData.xs("Area", axis=1, level=1) * (6.8644 / 1000)
CorrelationPlot(featureData, "Manual area", "OrganoID area", "Organoid", "Area", r"x $10^3 \mu m^2$")
plt.show()
