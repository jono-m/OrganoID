import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import matplotlib.pyplot as plt
from Core.ImageHandling import LoadPILImages, GetFrames
import numpy as np
import skimage.measure
import skimage.morphology
import skimage.filters
from scipy.interpolate import make_interp_spline, BSpline
from PIL import ImageFont, ImageDraw, Image


def DrawAndHighlight(labeled, image, labelsAndColorToHighlight, savePath):
    overlay = Image.new("RGBA", tuple(reversed(image.shape)), (255, 255, 255, 0))
    drawer = ImageDraw.Draw(overlay)
    font = ImageFont.truetype("arial.ttf", 24)
    for rp in skimage.measure.regionprops(labeled):
        if rp.label in labelsAndColorToHighlight:
            color = labelsAndColorToHighlight[rp.label]
        else:
            color = (255, 255, 255)
        drawer.point(list(rp.coords[:, (1, 0)].flatten()), color + tuple([50]))
        outline = skimage.morphology.binary_dilation(
            skimage.filters.sobel(rp.image, mode='constant') > 0)
        outlineCoords = np.argwhere(outline) + rp.bbox[:2]
        drawer.point(list(outlineCoords[:, (1, 0)].flatten()), color + tuple([255]))
        xC, yC = reversed(rp.centroid)
        drawer.text((xC, yC), str(rp.label), anchor="ms", fill=(255, 255, 255, 255), font=font)
    image = Image.fromarray(image).convert("RGBA")
    image = Image.alpha_composite(image, overlay)
    image.save(savePath)


def SmoothCurve(x, y, resolution):
    # 300 represents number of points to make between T.min and T.max
    xSmooth = np.linspace(0, max(x) - min(x), (max(x) - min(x)) * resolution + 1) + min(x)

    spline = make_interp_spline(x, y, k=3)  # type: BSpline
    ySmooth = spline(xSmooth)

    return xSmooth, ySmooth


def DoPlotScatterEnvelope(lab, x, y, c, z):
    smoothX, smoothY = SmoothCurve(x, y, 10)
    plt.plot(smoothX, smoothY, '-', color=c, label="_nolabel", zorder=z, linewidth=2)
    plt.scatter(x, y, marker='o', s=30, facecolor="white", edgecolor=c, label=lab, zorder=z)


def DoPlot():
    plt.rcParams['svg.fonttype'] = 'none'
    squareMicronsPerSquarePixel = 6.8644
    areasPerFrame = [{rp.label: rp.area * squareMicronsPerSquarePixel for rp in
                      skimage.measure.regionprops(i)} for i in labeledImages]
    for x in labelsToHighlight:
        areas = np.asarray([a[x] for a in areasPerFrame if x in a]) / 1000
        times = np.asarray([i * 2 for i, a in enumerate(areasPerFrame) if x in a])
        color = [y / 255 for y in labelsToHighlight[x]]
        zIndex = 2
        label = "ID %d" % x
        DoPlotScatterEnvelope(label, times, areas, color, zIndex)

    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel(r"Organoid Area (x $10^3 \mu m^2$)")
    plt.xlim([-1, len(areasPerFrame) * 2 - 1])
    plt.show()


originalImages = [np.asarray(i) / 255 for i in
                  GetFrames(LoadPILImages(
                      Path(r"Publication\Dataset\OriginalData\testing\Timelapse.tif"))[0])]
labeledImages = [np.asarray(i) for i in
                 GetFrames(
                     LoadPILImages(Path(r"Publication\Figure2\2d\Images\Timelapse_labeled.tif"))[
                         0])]

labelsToHighlight = {1: (0, 154, 222),
                     2: (255, 198, 30),
                     10: (175, 88, 186),
                     33: (0, 205, 108)}

DrawAndHighlight(labeledImages[0], originalImages[0], labelsToHighlight,
                 Path(r"Publication\Figure2\2d\Images\Timelapse_0hr.png"))
DrawAndHighlight(labeledImages[23], originalImages[23], labelsToHighlight,
                 Path(r"Publication\Figure2\2d\Images\Timelapse_46hr.png"))
DrawAndHighlight(labeledImages[46], originalImages[46], labelsToHighlight,
                 Path(r"Publication\Figure2\2d\Images\Timelapse_92hr.png"))
