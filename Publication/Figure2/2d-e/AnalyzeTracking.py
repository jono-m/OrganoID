import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
from Core.ImageHandling import LoadPILImages, GetFrames
from Core.Config import ARIAL_FONT_PATH
import numpy as np
import skimage.measure
import skimage.morphology
import skimage.filters
from PIL import ImageFont, ImageDraw, Image


def DrawAndHighlight(labeled, image, labelsAndColorToHighlight, savePath):
    overlay = Image.new("RGBA", tuple(reversed(image.shape)), (255, 255, 255, 0))
    drawer = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(ARIAL_FONT_PATH, 24)
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


originalImages = [np.asarray(i) / 255 for i in
                  GetFrames(LoadPILImages(
                      Path(r"Publication\Dataset\OriginalData\testing\Timelapse.tif"))[0])]
labeledImages = [np.asarray(i) for i in
                 GetFrames(
                     LoadPILImages(Path(r"Publication\Figure2\2d-e\Images\Timelapse_labeled.tif"))[
                         0])]

squareMicronsPerSquarePixel = 1.31 ** 2
areasFile = open(r"Publication\Figure2\2d-e\Areas.csv", "w+")
areasFile.write("Time,ID,Area")
for i, image in enumerate(labeledImages):
    for rp in skimage.measure.regionprops(image):
        label = rp.label
        area = rp.area * squareMicronsPerSquarePixel
        areasFile.write("\n%d,%d,%f" % (i * 2, label, area))
areasFile.close()

labelsToHighlight = {1: (0, 154, 222),
                     2: (255, 198, 30),
                     10: (175, 88, 186),
                     33: (0, 205, 108)}

DrawAndHighlight(labeledImages[0], originalImages[0], labelsToHighlight,
                 Path(r"Publication\Figure2\2d-e\Images\Timelapse_0hr.png"))
DrawAndHighlight(labeledImages[23], originalImages[23], labelsToHighlight,
                 Path(r"Publication\Figure2\2d-e\Images\Timelapse_46hr.png"))
DrawAndHighlight(labeledImages[46], originalImages[46], labelsToHighlight,
                 Path(r"Publication\Figure2\2d-e\Images\Timelapse_92hr.png"))
