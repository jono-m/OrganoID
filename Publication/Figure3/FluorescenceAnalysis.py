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


piImages = LoadPILImages(Path(r"Publication\Dataset\GemcitabineScreen\PI"))
labeledImages = LoadPILImages(Path(r"Publication\Dataset\GemcitabineScreen\OrganoIDProcessed\*.tif"))

bulkData = ""
singleData = ""
for piImage, labeledImage in zip(piImages, labeledImages):
    xy = GetXY(Path(piImage.filename).stem)
    if xy != GetXY(Path(labeledImage.filename).stem):
        raise Exception("Paths are not consistent.")
    dosage = GetDosage(xy)
    replicate = Replicate(xy)
    for i, (piFrame, labeledFrame) in enumerate(zip(GetFrames(piImage), GetFrames(labeledImage))):
        printRep(piImage.filename + " (" + str(i) + ")")
        time = i * 4
        labeledFrame = np.asarray(labeledFrame)
        piFrame = np.asarray(piFrame) / 1000
        totalFluorescence = int(np.sum(piFrame))
        organoidArea = np.count_nonzero(labeledFrame) / 1000
        maskedPiFrame = np.where(labeledFrame > 0, piFrame, 0)
        maskedTotalFluorescence = int(np.sum(maskedPiFrame))
        rps = skimage.measure.regionprops(labeledFrame)
        organoidCount = len(rps)
        bulkData += "\n%d,%d,%d,%d,%d,%d,%d" % (dosage, replicate, time, organoidCount,
                                                organoidArea, totalFluorescence,
                                                maskedTotalFluorescence)
        for rp in rps:
            circularity = (np.pi * rp.equivalent_diameter_area) / rp.perimeter
            if circularity > 1:
                print("Bad")
            organoidID = rp.label
            area = rp.area
            coords = rp.coords
            maskedFluorescence = int(np.sum(piFrame[coords[:, 0], coords[:, 1]]))
            singleData += "\n%d,%d,%d,%d,%d,%d,%f" % \
                          (dosage, replicate, time, organoidID, maskedFluorescence, area,
                           circularity)
printRep(None)

bulkMeasurementFile = open(Path(r"Publication\Figure3\3b\Data.csv"), mode="w+")
singleOrganoidMeasurementFile = open(Path(r"Publication\Figure3\3c\Data.csv"), mode="w+")
bulkMeasurementFile.write(
    "Gemcitabine Dosage (nM), Replicate, Time (hr), Organoid count, Organoid area (px), "
    "Total fluorescence (au), Masked total fluorescence (au)")
bulkMeasurementFile.write(bulkData)
singleOrganoidMeasurementFile.write(
    "Gemcitabine Dosage (nM), Replicate, Time (hr), Organoid ID, Masked fluorescence, Area, "
    "Circularity")
singleOrganoidMeasurementFile.write(singleData)
bulkMeasurementFile.close()
singleOrganoidMeasurementFile.close()
