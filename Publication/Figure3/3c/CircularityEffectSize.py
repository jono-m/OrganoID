import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn


fontsize = 10
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
organoidData = pandas.read_csv(r"Publication\Figure3\OrganoidMeasurements.csv").set_index(
    ["Time", "Dosage", "Replicate", "Organoid ID"])
organoidData["Organoid fluorescence"] = organoidData["Fluorescence"] / organoidData["Area"]
t = 72
organoidData = organoidData.loc[t]

features = ["Circularity", "Eccentricity", "Solidity", "Organoid fluorescence"]
results = []
for feature in features:
    featureData = organoidData[feature]
    means = featureData.groupby("Dosage").mean().rename("Mean")
    sds = featureData.groupby("Dosage").std().rename("SD")
    featureData = pandas.concat([means, sds], axis=1)

    result = pd.DataFrame(index=featureData.index, columns=featureData.index, dtype=float)

    for dosageA in featureData.index:
        for dosageB in featureData.index:
            if dosageA < dosageB:
                continue
            d = ((featureData.loc[dosageA, "Mean"] - featureData.loc[dosageB, "Mean"]) /
                 np.sqrt((featureData.loc[dosageB, "SD"] ** 2 +
                          featureData.loc[dosageA, "SD"] ** 2) / 2))
            result.loc[dosageA, dosageB] = float(d)
    results.append(result.iloc[::-1])
    print(feature)
    print(result)

dMin = -1
dMax = 1
for i, (feature, result) in enumerate(zip(features, results)):
    plt.subplot(2, 2, i + 1)
    plt.title(feature)
    labelY = result.index.astype(int)
    labelX = result.columns.astype(int)
    seaborn.heatmap(result, vmin=dMin, vmax=dMax, center=0, square=True, xticklabels=labelX, yticklabels=labelY,
                    cmap=seaborn.cm.icefire_r)
    plt.ylabel("Dosage (nM)")
    plt.xlabel("Dosage (nM)")
plt.tight_layout()
plt.show()
