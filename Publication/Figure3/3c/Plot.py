from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import colorsys
import pandas
import seaborn
import numpy as np

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
colors = [colorsys.hsv_to_rgb(hue, sat, 1) for sat in np.linspace(0.2, 1, 2)] + \
         [colorsys.hsv_to_rgb(hue, 1, val) for val in np.linspace(0, 0, 1)]

dataFile = open(Path(r"Publication\Figure3\3c\Data.csv"), mode="r")
data = np.asarray([[float(y) for y in x.split(",")] for x in dataFile.read().split("\n")[1:]])
dosagesToUse = [0, 10, 100]
data = data[np.in1d(data[:, 0], dosagesToUse), :]

allData = pandas.DataFrame(data, columns=["Dose", "Replicate", "Time", "Organoid ID",
                                          "Fluorescence", "Area", "Circularity"])
allData["Fluorescence intensity per area"] = allData["Fluorescence"] / allData["Area"]

images = []
t = 72
d = allData[allData["Time"] == t]

palette = {dose: color for color, dose in zip(colors, dosagesToUse)}
jp = seaborn.JointGrid(data=d, y="Fluorescence intensity per area", x="Circularity", hue="Dose",
                       palette=palette,
                       xlim=(np.min(d["Circularity"]),
                             np.max(d["Circularity"])),
                       ylim=(np.min(d["Fluorescence intensity per area"]),
                             np.max(d["Fluorescence intensity per area"])))
jp.plot_marginals(seaborn.kdeplot, common_norm=False, fill=True, alpha=0.7)
jp.plot_joint(seaborn.scatterplot, s=10, legend=False)
jp.ax_joint.axvline(1.0, color="black")
for color, dose in zip(colors, dosagesToUse):
    jp.ax_marg_x.axvline(d[d["Dose"] == dose]["Circularity"].median(), color=color)
    jp.ax_marg_y.axhline(d[d["Dose"] == dose]["Fluorescence intensity per area"].median(),
                         color=color)

plt.show()
