import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline
import numpy as np
import pandas


def SmoothCurve(x, y, resolution):
    # 300 represents number of points to make between T.min and T.max
    xSmooth = np.linspace(0, max(x) - min(x), (max(x) - min(x)) * resolution + 1) + min(x)

    spline = make_interp_spline(x, y, k=3)  # type: BSpline
    ySmooth = spline(xSmooth)

    return xSmooth, ySmooth


labelsToHighlight = {1: (0, 154, 222),
                     2: (255, 198, 30),
                     10: (175, 88, 186),
                     33: (0, 205, 108)}


def DoPlotScatterEnvelope(lab, x, y, c, z):
    smoothX, smoothY = SmoothCurve(x, y, 10)
    plt.plot(smoothX, smoothY, '-', color=c, label="_nolabel", zorder=z, linewidth=2)
    plt.scatter(x, y, marker='o', s=30, facecolor="white", edgecolor=c, label=lab, zorder=z)


areasData = pandas.read_csv(r"Publication\Figure2\2d-e\Areas.csv").set_index(["ID", "Time"])

plt.rcParams['svg.fonttype'] = 'none'
for x in labelsToHighlight:
    data = areasData.loc[x]
    color = [y / 255 for y in labelsToHighlight[x]]
    zIndex = 2
    label = "ID %d" % x
    DoPlotScatterEnvelope(label, data.index, data, color, zIndex)

plt.legend()
plt.xlabel("Time (hours)")
plt.ylabel(r"Organoid Area (x $10^3 \mu m^2$)")
plt.xlim([-1, 90])
plt.show()
