from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))

from Core.Model import LoadLiteModel, ComputeIOUs, LoadModel
from Core.ImageHandling import LoadPILImages
import numpy as np

liteModel = LoadLiteModel(Path(r"OptimizedModel"))
model = LoadModel(Path(r"Publication\Figure4\RetrainedModel"))

images = LoadPILImages(Path(r"Publication\Dataset\MouseOrganoids\testing\images"))
segmentations = LoadPILImages(
    Path(r"Publication\Dataset\MouseOrganoids\testing\segmentations"))

print("Mouse Data:")
ious = ComputeIOUs(model, images, segmentations, 0.5)
iousLite = ComputeIOUs(liteModel, images, segmentations, 0.5)
print("Filename\tOldModel\tNewModel")
for iou, iouLite, image in zip(ious, iousLite, images):
    print("%s\t%f\t%f\t" % (Path(image.filename).name, iouLite, iou))
print("Old model: %f (%f SD)" % (float(np.mean(iousLite)), float(np.std(iousLite))))
print("New model: %f (%f SD)" % (float(np.mean(ious)), float(np.std(ious))))

images = LoadPILImages(Path(r"Publication\Dataset\OriginalData\testing\images"))
segmentations = LoadPILImages(Path(r"Publication\Dataset\OriginalData\testing\segmentations"))
print("Original Data:")
ious = ComputeIOUs(model, images, segmentations, 0.5)
iousLite = ComputeIOUs(liteModel, images, segmentations, 0.5)
print("Filename\tOldModel\tNewModel")
for iou, iouLite, image in zip(ious, iousLite, images):
    print("%s\t%f\t%f\t" % (Path(image.filename).name, iouLite, iou))
print("Old model: %f (%f SD)" % (float(np.mean(iousLite)), float(np.std(iousLite))))
print("New model: %f (%f SD)" % (float(np.mean(ious)), float(np.std(ious))))
