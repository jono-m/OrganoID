from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))
import tensorflow as tf
from Core.Model import LoadLiteModel, Detect, PrepareImagesForModel, PrepareSegmentationsForModel
from Core.ImageHandling import LoadPILImages

modelPaths = list(Path(r"Publication\Training\OriginalModelEpochs").iterdir())[:-1]
trainingImages = LoadPILImages(Path(r"Publication\Dataset\OriginalData\training\images"))
trainingSegmentations = LoadPILImages(
    Path(r"Publication\Dataset\OriginalData\training\segmentations"))
validationImages = LoadPILImages(Path(r"Publication\Dataset\OriginalData\validation\images"))
validationSegmentations = LoadPILImages(
    Path(r"Publication\Dataset\OriginalData\validation\segmentations"))

trainingImagesPrep = None
trainingSegPrep = None
validationImagesPrep = None
validationSegPrep = None

lossFunction = tf.keras.losses.BinaryCrossentropy()

data = ""
for modelPath in modelPaths:
    model = LoadLiteModel(modelPath)
    epoch = int(modelPath.name.split("_")[1])
    if trainingImagesPrep is None:
        trainingImagesPrep = PrepareImagesForModel(trainingImages, model)
        trainingSegPrep = PrepareSegmentationsForModel(trainingSegmentations, model)
        validationImagesPrep = PrepareImagesForModel(validationImages, model)
        validationSegPrep = PrepareSegmentationsForModel(validationSegmentations, model)
    trainingRes = Detect(model, trainingImagesPrep, 16)
    validationRes = Detect(model, validationImagesPrep, 16)
    trainingLoss = lossFunction(trainingSegPrep, trainingRes).numpy()
    validationLoss = lossFunction(validationSegPrep, validationRes).numpy()
    data += "\n%d,%f,%f" % (epoch, trainingLoss, validationLoss)

outFile = open(Path(r"Publication\Supplement\S1\Losses.csv"), "w+")
outFile.write("Epoch, Training Loss, Validation Loss")
outFile.write(data)
outFile.close()
