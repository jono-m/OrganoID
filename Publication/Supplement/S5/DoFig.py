import matplotlib.pyplot as plt
import colorsys
import numpy as np
from pathlib import Path
import pandas
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

allData = pandas.read_csv(Path(r"Publication\Figure3\3b\Data.csv"))
mtsData = pandas.read_csv(Path(r"Publication\Supplement\S5\MTSAssay.csv"))
mtsMean = mtsData.groupby('Dosage').mean()
endpointTime = np.unique(allData["Time"])[-1]
endpointData = allData.loc[allData["Time"] == endpointTime].copy()
fluorescenceColumn = 'Total fluorescence'
backgroundFluorescence = endpointData.groupby('Dosage').mean().loc[0, fluorescenceColumn]
endpointData[fluorescenceColumn] -= backgroundFluorescence
endpointData['Fluorescence (area-normalized)'] = endpointData[fluorescenceColumn] / endpointData[
    'Organoid area']
endpointData['Fluorescence (area-normalized)'] = endpointData[fluorescenceColumn] / endpointData[
    'Organoid area']
endpointData['Fluroescence (MTS-normalized)'] = endpointData[fluorescenceColumn] / list(
    mtsMean.loc[endpointData['Dosage'], 'Viability'])





plt.subplot(2, 2, 1)
DoseReponsePlot(endpointData, fluorescenceColumn, 1000)
plt.subplot(2, 2, 2)
DoseReponsePlot(endpointData, "Fluroescence (MTS-normalized)", 1000)
plt.subplot(2, 2, 3)
# DoseReponsePlot(endpointData, "Fluorescence (area-normalized)", 1000)
DoseReponsePlot(endpointData, "Organoid area", None)
plt.xlabel("Gemcitabine concentration (nM)")
plt.subplot(2, 2, 4)
DoseReponsePlot(mtsData, "Viability", 0)
plt.xlabel("Gemcitabine concentration (nM)")
plt.tight_layout()
# DoseReponsePlot(endpointData, "Masked fluorescence (area-normalized)")
plt.show()
