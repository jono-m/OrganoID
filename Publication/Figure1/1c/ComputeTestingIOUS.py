from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))

from Core.Model import LoadLiteModel, ComputeIOUs
from Core.ImageHandling import LoadPILImages
import numpy as np

datasets = ["PDAC", "ACC", "C", "Lung"]
names = ["PDAC", "ACC", "Colon", "Lung"]

liteModel = LoadLiteModel(Path(r"OptimizedModel"))

iousFile = open(Path(r"Publication\Figure1\1c\IOUs.csv"), mode="w+")
for dataset, name in zip(datasets, names):
    images = LoadPILImages(Path(r"Publication\Dataset\OriginalData\testing\images\%s*" % dataset))
    segmentations = LoadPILImages(
        Path(r"Publication\Dataset\OriginalData\testing\segmentations\%s*" % dataset))
    liteIOUs = ComputeIOUs(liteModel, images, segmentations)
    iousFile.write(name + ": " + str(np.mean(liteIOUs)) + " (SD: " + str(np.std(liteIOUs)) + ")\n")

images = LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\testing\images")) + \
         LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\training\pre_augmented\images")) + \
         LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\validation\images"))
segmentations = LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\testing\segmentations")) + \
                LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\training\pre_augmented\segmentations")) + \
                LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\validation\segmentations"))
mouseIOUs = ComputeIOUs(liteModel, images, segmentations)
iousFile.write("Mouse: " + str(np.mean(mouseIOUs)) + " (SD: " + str(np.std(mouseIOUs)) + ")\n")
iousFile.close()
