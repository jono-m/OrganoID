import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(".").resolve()))
from PIL import Image
from Core.ImageHandling import LoadPILImages
from Core.Model import Detect, LoadLiteModel, PrepareSegmentationsForModel, PrepareImagesForModel

exampleImageNames = ["ACC3", "C3", "Lung5", "PDAC9"]
outPath = Path(r"Publication\Figure1\1c\ExampleImages")
outPath.mkdir(parents=True, exist_ok=True)

model = LoadLiteModel(Path(r"OptimizedModel"))
images = PrepareImagesForModel(LoadPILImages(
    [Path(r"Publication\Dataset\OriginalData\testing\images\%s*" % name) for name in
     exampleImageNames]) + LoadPILImages(
    Path(r"Publication\Dataset\MouseOrganoids\validation\images\M13.png")),
                               model)
segmentations = PrepareSegmentationsForModel(LoadPILImages(
    [Path(r"Publication\Dataset\OriginalData\testing\segmentations\%s*" % name) for name in
     exampleImageNames]) + LoadPILImages(
    Path(r"Publication\Dataset\MouseOrganoids\validation\segmentations\M13.png")),
                                             model)

detections = Detect(model, images)

for i, name in enumerate(exampleImageNames + ["M13"]):
    actualImage = segmentations[i]
    predictionImage = detections[i]
    groundTruthColored = np.zeros([512, 512, 3], dtype=np.uint8)
    detectedColored = np.zeros([512, 512, 3], dtype=np.uint8)
    merged = np.zeros([512, 512, 3], dtype=np.uint8)

    groundTruthColored[np.where(actualImage)] = [174, 205, 236]
    merged[np.where(actualImage)] = [174, 205, 236]

    detectedColored[np.where(predictionImage >= 0.5)] = [243, 213, 123]
    merged[np.where(predictionImage >= 0.5)] = [243, 213, 123]

    merged[np.where(np.bitwise_and(predictionImage >= 0.5, actualImage))] = [112, 200, 120]

    Image.fromarray(merged).save(outPath / (name + "_merged.png"))
    Image.fromarray(detectedColored).save(outPath / (name + "_detected.png"))
    Image.fromarray(groundTruthColored).save(outPath / (name + "_gt.png"))
