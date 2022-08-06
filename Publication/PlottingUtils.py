import pandas
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import scipy.optimize
from Publication.Statistics import linr_ci

fontsize = 10
corrColor = [x / 255 for x in (0, 205, 108)]
meanColor = [x / 255 for x in (0, 205, 108)]
lodColor = [x / 255 for x in (0, 154, 222)]
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
hue = 343 / 360
piColors = [colorsys.hsv_to_rgb(hue, sat, 1) for sat in np.linspace(0.1, 1, 4)] + \
           [colorsys.hsv_to_rgb(hue, 1, val) for val in np.linspace(0.8, 0, 4)]


def TimecoursePlot(df: pandas.Series, axes: plt.Axes):
    responseLabel = str(df.name)
    dosages = np.unique(df.index.get_level_values("Dosage"))
    for color, dosage in zip(reversed(piColors), reversed(dosages)):
        if dosage == 0:
            continue
        response = df[df.index.get_level_values("Dosage") == dosage].groupby("Time")
        axes.errorbar(response.mean().index, response.mean(), response.sem(),
                      label="%d nM" % dosage, color=color)
    n = df.groupby(["Dosage", "Time"]).count()
    axes.set_ylabel(responseLabel)
    axes.set_xlabel("Time (hours)")


def DoseReponsePlot(df: pandas.Series, axes: plt.Axes, color=piColors[4]):
    def fsigmoid(x, a, b, c, d):
        return c / (1.0 + np.exp(a * (np.log(x) - b))) + d

    def FitCurve(xdata, ydata):
        initialGuess = [0, np.mean(np.log(xdata)), np.max(ydata) - np.min(ydata), np.min(ydata)]
        popt, pcov = scipy.optimize.curve_fit(fsigmoid, xdata, ydata, p0=initialGuess,
                                              method='dogbox')
        return popt

    responseLabel = str(df.name)
    df = df[df.index.get_level_values("Dosage") != 0]
    df = df.groupby("Dosage")
    axes.errorbar(df.mean().index, df.mean(), yerr=df.sem(), fmt="o", color=color, label=responseLabel)
    axes.set_xscale('log')
    axes.set_ylabel(responseLabel)
    try:
        fit = FitCurve(df.mean().index, df.mean())
        ec50 = np.exp(fit[1])
        x = np.linspace(np.min(df.mean().index), np.max(df.mean().index), 10000)
        axes.plot(x, fsigmoid(x, *fit), color=color)
        yec50 = fsigmoid(ec50, *fit)
        axes.axvline(ec50, yec50 - 100, yec50 + 100, color="black", linestyle="--")
        axes.text(ec50 + 10, yec50, "EC50: %.2f nM" % ec50)
    except Exception as e:
        print(e)


def CorrelationPlot(features: pandas.DataFrame, labelX: str, labelY: str,
                    pointName: str, featureName: str, units: str):
    featuresX = features["Manual"]
    featuresY = features["Automated"]
    if units != "":
        units = " (" + units + ")"
    ccc, loc, hic = linr_ci(featuresX, featuresY)
    maxData = max(np.max(featuresX), np.max(featuresY))
    plt.subplots(1, 2, figsize=(9, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(featuresX, featuresY, marker='o', s=5, color='k')
    plt.xlabel(labelX + units)
    plt.ylabel(labelY + units)
    minData = min(np.min(featuresX), np.min(featuresY))
    plt.text(minData, maxData, "$CCC=%.2f$\n[%.2f-%.2f]" % (ccc, loc, hic),
             verticalalignment='top',
             size=fontsize,
             color=corrColor)
    plt.plot([minData, maxData], [minData, maxData], "-", color=corrColor)

    plt.subplot(1, 2, 2)
    data = np.stack([featuresX, featuresY])
    means = np.mean(data, axis=0)
    differences = featuresY - featuresX
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
    plt.xlabel(featureName + " average" + units)
    plt.ylabel(featureName + " difference" + units)
    ylim = np.max(np.abs([min(differences), max(differences)])) * 1.2
    plt.ylim([-ylim, ylim])
    for (y, text, color) in labels:
        plt.text(np.max(means), y, text, verticalalignment='bottom', horizontalalignment='right',
                 color=color,
                 size=fontsize)
    plt.title("Comparison of " + featureName)
