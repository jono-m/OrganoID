from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
import pandas
from Publication.PlottingUtils import DoseReponsePlot
import numpy as np


bulkData = pandas.read_csv(Path(r"Publication\Figure3\3b\Data.csv"))
bulkData = bulkData.set_index(["Dosage", "Replicate", "Time"])
bulkData["FPA"] = bulkData["Total fluorescence"] / bulkData["Organoid area"]

mtsData = pandas.read_csv(Path(r"Publication\Supplement\S5\MTSAssay.csv"))
fig, axes = plt.subplots(4, 1, sharex='all')
mtsData = mtsData.set_index("Dosage")
DoseReponsePlot(mtsData, "Viability", axes[0])
DoseReponsePlot(bulkData.query("Time == 48"), "Total fluorescence", axes[1])
DoseReponsePlot(bulkData.query("Time == 48"), "Organoid area", axes[2])
DoseReponsePlot(bulkData.query("Time == 48"), "FPA", axes[3])
axes[3].set_xlabel("Gemcitabine concentration (nM)")
# data = np.asarray([[float(y) for y in x.split(",")] for x in dataFile.read().split("\n")[1:]])
# dosagesToUse = [0, 10, 100]
# data = data[np.in1d(data[:, 0], dosagesToUse), :]
#
# allData = pandas.DataFrame(data, columns=["Dose", "Replicate", "Time", "Organoid ID",
#                                           "Fluorescence", "Area", "Circularity"])
# allData["Fluorescence intensity per area"] = allData["Fluorescence"] / allData["Area"]
#
# images = []
# t = 72
# d = allData[allData["Time"] == t]
#
# palette = {dose: color for color, dose in zip(colors, dosagesToUse)}
# jp = seaborn.JointGrid(data=d, y="Fluorescence intensity per area", x="Circularity", hue="Dose",
#                        palette=palette,
#                        xlim=(np.min(d["Circularity"]),
#                              np.max(d["Circularity"])),
#                        ylim=(np.min(d["Fluorescence intensity per area"]),
#                              np.max(d["Fluorescence intensity per area"])))
# jp.plot_marginals(seaborn.kdeplot, common_norm=False, fill=True, alpha=0.7)
# jp.plot_joint(seaborn.scatterplot, s=10, legend=False)
# jp.ax_joint.axvline(1.0, color="black")
# for color, dose in zip(colors, dosagesToUse):
#     jp.ax_marg_x.axvline(d[d["Dose"] == dose]["Circularity"].median(), color=color)
#     jp.ax_marg_y.axhline(d[d["Dose"] == dose]["Fluorescence intensity per area"].median(),
#                          color=color)

plt.show()
