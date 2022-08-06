import matplotlib.pyplot as plt
import colorsys
import numpy as np
from pathlib import Path
import pandas
import seaborn
import scipy.optimize
import scipy.stats

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

soData = pandas.read_csv(Path(r"Publication\Figure3\OrganoidMeasurements.csv"))

dosages = np.unique(soData["Dosage"])
soData = soData.set_index(["Dosage", "Replicate", "Organoid ID", "Time"])
soData = soData.drop(columns=["X", "Y"])

initialBatch = soData.query("Time == 0").groupby(["Dosage", "Replicate"]).mean()
batchFoldChange = soData / initialBatch
batchFoldChange = batchFoldChange.groupby(["Dosage", "Time"])
batchCovs = (batchFoldChange.std() / np.abs(batchFoldChange.mean()))

initialTracked = soData.reset_index().sort_values("Time").groupby(
    ["Dosage", "Replicate", "Organoid ID"]).first().drop(columns="Time")
trackedFoldChange = soData / initialTracked
trackedFoldChange = trackedFoldChange.groupby(["Dosage", "Time"])
trackedCovs = (trackedFoldChange.std() / np.abs(trackedFoldChange.mean()))

trackedCovsMelted = trackedCovs.melt()
trackedCovsMelted["Strategy"] = "Tracked"
batchCovsMelted = batchCovs.melt()
batchCovsMelted["Strategy"] = "Batch"
covs = pandas.concat([trackedCovsMelted, batchCovsMelted])
seaborn.boxplot(x="variable", y="value", data=covs, hue="Strategy", palette={"Tracked": corrColor,
                                                                             "Batch": lodColor})
plt.ylabel("CV across organoid replicates")
plt.xlabel("Organoid feature (fold change)")

print((batchCovs.mean() - trackedCovs.mean()) / ((batchCovs.mean() + trackedCovs.mean()) / 2))
print(scipy.stats.ttest_ind(batchCovs, trackedCovs, axis=0, alternative='less',
                            equal_var=False).pvalue)
plt.show()
