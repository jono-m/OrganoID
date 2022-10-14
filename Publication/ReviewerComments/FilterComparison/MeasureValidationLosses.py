from pathlib import Path
import sys

sys.path.append(str(Path(".").resolve()))
import tensorflow as tf
from Core.Model import LoadFullModel, LoadLiteModel, Detect, PrepareImagesForModel, PrepareSegmentationsForModel
from Core.ImageHandling import LoadPILImages

model4 = LoadFullModel(Path(r"Publication\ReviewerComments\FilterComparison\Filter4_BEST"))
model8 = LoadLiteModel(Path(r"OptimizedModel"))
model16 = LoadFullModel(Path(r"Publication\ReviewerComments\FilterComparison\Filter16_BEST"))
validationImages = LoadPILImages(Path(r"Publication\Dataset\OriginalData\validation\images"))
validationSegmentations = LoadPILImages(
    Path(r"Publication\Dataset\OriginalData\validation\segmentations"))

lossFunction = tf.keras.losses.BinaryCrossentropy()

for name, model in [("4", model4),
                    ("8", model8),
                    ("16", model16)]:
    validationImagesPrep = PrepareImagesForModel(validationImages, model)
    validationSegPrep = PrepareSegmentationsForModel(validationSegmentations, model)
    validationRes = Detect(model, validationImagesPrep, 16)
    validationLoss = lossFunction(validationSegPrep, validationRes).numpy()

    print(name + ": %f" % validationLoss)
