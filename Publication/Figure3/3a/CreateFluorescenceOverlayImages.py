import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))
import numpy as np
import re
from Core.ImageHandling import LoadPILImages, ComputeOutline
from PIL import Image

dosages = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3]


def GetXY(name: str):
    matches = re.findall(r".*XY(\d+).*", name)
    if matches:
        return int(matches[0])
    return None


def GetDosage(xY: int):
    return dosages[int((xY - 1) / 6)]


brightfieldImages = LoadPILImages(Path(r"Publication\Dataset\GemcitabineScreen\BF"))
piImages = LoadPILImages(Path(r"Publication\Dataset\GemcitabineScreen\PI"))
labeledImages = LoadPILImages(
    Path(r"Publication\Dataset\GemcitabineScreen\OrganoIDProcessed\*.tif"))

outputPath = Path(r"Publication\Figure3\3a")
outputPath.mkdir(parents=True, exist_ok=True)

dosagesToShow = [0, 30]
timesToShow = [0, 9, 17]

for piImage, labeledImage, brightfieldImage in zip(piImages, labeledImages, brightfieldImages):
    xy = GetXY(Path(piImage.filename).stem)
    if xy != GetXY(Path(labeledImage.filename).stem) or xy != GetXY(
            Path(brightfieldImage.filename).stem):
        raise Exception("Paths are not consistent.")
    dosage = GetDosage(xy)
    if dosage not in dosagesToShow:
        continue
    if (xy - 1) % 6 > 0:
        continue
    for time in timesToShow:
        piImage.seek(time)
        labeledImage.seek(time)
        brightfieldImage.seek(time)

        piNormalized = np.asarray(piImage)
        piNormalized = 2 * (piNormalized - np.min(piNormalized)) / (
                np.max(piNormalized) - np.min(piNormalized))
        piNormalized = np.clip(piNormalized, 0, 1)
        piAlphaImage = np.full(list(piNormalized.shape) + [4], [255, 31, 91, 255])
        piAlphaImage[:, :, 3] = piNormalized * 255
        piAlphaImage = piAlphaImage.astype(np.uint8)

        outlineImage = np.where(ComputeOutline(np.asarray(labeledImage))[:, :, None],
                                [255, 255, 255, 255],
                                [0, 0, 0, 255]).astype(np.uint8)

        brightfieldImagePrepared = (np.asarray(brightfieldImage) / 255).astype(np.uint8)

        piAlphaImage = Image.fromarray(piAlphaImage).convert(mode="RGBA")
        outlineImage = Image.fromarray(outlineImage).convert(mode="RGBA")
        brightfieldImagePrepared = Image.fromarray(brightfieldImagePrepared).convert(mode="RGBA")
        Image.alpha_composite(brightfieldImagePrepared, piAlphaImage).save(
            outputPath / ("overlay_%dnM_%dhr.png" % (dosage, time * 4)))
        Image.alpha_composite(outlineImage, piAlphaImage).save(
            outputPath / ("segmented_%dnM_%dhr.png" % (dosage, time * 4)))
