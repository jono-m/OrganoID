from pathlib import Path
import sys
import re

sys.path.append(str(Path(".").resolve()))

import numpy as np
from Core.ImageHandling import LoadPILImages, GetFrames
from Core.HelperFunctions import printRep
import skimage.measure

dosages = [np.inf, 0, None, 1000, 300, 100, 30, 10, 3]


def GetXY(name: str):
    matches = re.findall(r".*XY(\d+).*", name)
    if matches:
        return int(matches[0])
    return None


def Replicate(xY: int):
    return xY % 6


def GetDosage(xY: int):
    return dosages[int((xY - 1) / 6)]


def circularity(rp):
    return (4 * np.pi * rp.area) / (rp.perimeter * rp.perimeter)


piImages = LoadPILImages(Path(r"Publication\Dataset\GemcitabineScreen\PI"))
labeledImages = LoadPILImages(
    Path(r"Publication\Dataset\GemcitabineScreen\OrganoIDProcessed\*.tif"))

organoidMeasurements = ""
piMeasurements = ""
for piImage, labeledImage in zip(piImages, labeledImages):
    xy = GetXY(Path(piImage.filename).stem)
    if xy != GetXY(Path(labeledImage.filename).stem):
        raise Exception("Paths are not consistent.")
    dosage = GetDosage(xy)
    replicate = Replicate(xy)
    for t, (piFrame, labeledFrame) in enumerate(zip(GetFrames(piImage), GetFrames(labeledImage))):
        printRep(piImage.filename + " (" + str(t) + ")")
        time = t * 4
        labeledFrame = np.asarray(labeledFrame)
        piFrame = np.asarray(piFrame) / 1000
        piMeasurements += "\n%d,%d,%d,%d" % (dosage, replicate, time, int(np.sum(piFrame)))
        rps = skimage.measure.regionprops(labeledFrame)
        organoidCount = len(rps)
        for rp in rps:
            coords = rp.coords
            maskedFluorescence = int(np.sum(piFrame[coords[:, 0], coords[:, 1]]))
            organoidMeasurements += "\n%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f" % \
                                    (dosage, replicate, time, rp.label, maskedFluorescence, rp.area,
                                     rp.perimeter, rp.solidity, rp.eccentricity, circularity(rp),
                                     rp.centroid[0],
                                     rp.centroid[1])
printRep(None)

organoidMeasurementsFile = open(Path(r"Publication\Figure3\OrganoidMeasurements.csv"), mode="w+")
organoidMeasurementsFile.write(
    "Dosage,Replicate,Time,Organoid ID,Fluorescence,"
    "Area,Perimeter,Solidity,Eccentricity,Circularity,X,Y")
organoidMeasurementsFile.write(organoidMeasurements)
organoidMeasurementsFile.close()

piMeasurementsFile = open(Path(r"Publication\Figure3\PIMeasurements.csv"), mode="w+")
piMeasurementsFile.write(
    "Dosage,Replicate,Time,Fluorescence")
piMeasurementsFile.write(piMeasurements)
piMeasurementsFile.close()
