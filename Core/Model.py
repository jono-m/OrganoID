# Training.py -- uses Keras/TensorFlow to train a U-Net convolutional neural network.
import math
import pathlib

import tensorflow as tf
import numpy as np
from PIL import Image
from typing import List, Tuple, Union
from Core.ImageHandling import NumFrames, GetFrames
from Core.HelperFunctions import printRep
from pathlib import Path


# Builds a U-Net model for a given image size and filter count.
def BuildModel(imageSize: Tuple[int, int], dropoutRate: float, firstLayerFilterCount: int):
    # First layer in the network is each pixel in the grayscale image
    inputs = tf.keras.layers.Input((imageSize[0], imageSize[1], 1))

    # Contracting path identifies features at increasing levels of detail
    # (i.e. intensity, edges, shapes, texture...)
    currentLayer = inputs
    contractingLayers = []
    for i in range(5):
        if i > 0:
            currentLayer = tf.keras.layers.MaxPooling2D((2, 2))(currentLayer)

        size = firstLayerFilterCount * 2 ** i
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation='elu',
                                              kernel_initializer='he_normal',
                                              padding='same')(
            currentLayer)
        currentLayer = tf.keras.layers.Dropout(dropoutRate * i)(currentLayer)
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation='elu',
                                              kernel_initializer='he_normal',
                                              padding='same')(
            currentLayer)
        contractingLayers.append(currentLayer)

    # Expanding path spatially places recognized features.
    for i in reversed(range(4)):
        size = firstLayerFilterCount * 2 ** i

        currentLayer = tf.keras.layers.Conv2DTranspose(size, (2, 2), strides=(2, 2),
                                                       padding='same')(
            currentLayer)
        currentLayer = tf.keras.layers.concatenate([currentLayer, contractingLayers[i]])
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation='elu',
                                              kernel_initializer='he_normal',
                                              padding='same')(
            currentLayer)
        currentLayer = tf.keras.layers.Dropout(dropoutRate * i)(currentLayer)
        currentLayer = tf.keras.layers.Conv2D(size, (3, 3), activation='elu',
                                              kernel_initializer='he_normal',
                                              padding='same')(
            currentLayer)

    # Last layer is sigmoid to produce probability map.
    final = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(currentLayer)

    model = tf.keras.Model(inputs=[inputs], outputs=[final])
    return model


def TrainModel(model: tf.keras.Model, learningRate, patience, epochs, batchSize,
               trainingData: List['GroundTruth'], validationData: List['GroundTruth'],
               saveDirectory: Path, saveNamePrefix: str, saveLite, saveAll: bool):
    savePath = saveDirectory / (saveNamePrefix + "_E{epoch:03d}")

    # Adam optimizer is used for SGD. Binary cross-entropy for loss.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learningRate),
                  loss=tf.keras.losses.binary_crossentropy)

    # Stop training once performance on validation dataset has reached a local minimum
    # (window size is "patience"). We also will save a copy of the model after every epoch.
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=patience, verbose=1, restore_best_weights=True),
    ]
    if saveAll:
        callbacks.append(ModelCheckpoint(savePath.absolute(), model, saveLite))

    model.fit(x=ImageGenerator(trainingData, batchSize, model),
              validation_data=LoadGroundTruths(validationData, model),
              batch_size=batchSize,
              shuffle=True,
              verbose=1,
              epochs=epochs,
              callbacks=callbacks)

    if saveLite:
        SaveLiteModel(saveDirectory / "model.tflite", model)
    else:
        model.save(saveDirectory / (saveNamePrefix + "_BEST"))


def SaveLiteModel(path: Path, model: tf.keras.Model):
    savePath = path
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    liteModel = converter.convert()
    with open(savePath, "wb") as f:
        f.write(liteModel)


ModelType = Union[tf.lite.Interpreter, tf.keras.Model]


class ModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, path: Path, model: ModelType, saveLite: bool):
        super().__init__(path, verbose=True)
        self.path = path
        self.model = model
        self.saveLite = saveLite

    def on_epoch_end(self, epoch: int, logs=None):
        if self.saveLite:
            path = self.path.parent / "epochs" / ("epoch_" + str(epoch) + ".tflite")
            SaveLiteModel(path, self.model)
        else:
            super().on_epoch_end(epoch, logs)


class ImageGenerator(tf.keras.utils.Sequence):
    def __init__(self, data: List['GroundTruth'], batchSize: int, model: ModelType):
        super().__init__()
        self.data = data
        self.batchSize = batchSize
        self.model = model
        self.Shuffle()

    def __len__(self):
        return math.ceil(float(len(self.data)) / self.batchSize)

    def __getitem__(self, index):
        return LoadGroundTruths(self.data[(self.batchSize * index):(self.batchSize * (index + 1))],
                                self.model)

    def on_epoch_end(self):
        self.Shuffle()

    def Shuffle(self):
        self.data = np.random.permutation(self.data)


class GroundTruth:
    def __init__(self, imagePath: pathlib.Path, segmentationPath: pathlib.Path):
        self.imagePath = imagePath
        self.segmentationPath = segmentationPath
        if imagePath.stem != segmentationPath.stem:
            raise Exception("Ground truth names do not match!\n\t" +
                            str(imagePath.absolute()) + "\n\t" +
                            str(segmentationPath.absolute()) + "\n\t")


def LoadGroundTruths(groundTruths: List[GroundTruth], model: ModelType):
    images = PrepareImagesForModel([Image.open(x.imagePath) for x in groundTruths], model,
                                   verbose=False)
    segmentations = PrepareSegmentationsForModel(
        [Image.open(x.segmentationPath) for x in groundTruths], model)

    return images, segmentations


def LoadModel(path: pathlib.Path):
    return tf.keras.models.load_model(path)


def LoadLiteModel(path: pathlib.Path):
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=str(path.absolute()))
    interpreter.allocate_tensors()
    return interpreter


def InputSize(model: ModelType) -> Tuple[int, int]:
    if isinstance(model, tf.lite.Interpreter):
        input_details = model.get_input_details()
        return input_details[0]['shape'][1:3]
    return model.get_layer(index=0).input_shape[0][1:3]


def PrepareImagesForModel(images: List[Image.Image], model: ModelType, verbose=True):
    modelInputSize = InputSize(model)
    totalFrameCount = sum([NumFrames(image) for image in images])
    if verbose:
        print("Loading images...", end="", flush=True)
    loadedImages = np.zeros([totalFrameCount, modelInputSize[0], modelInputSize[1]],
                            dtype=np.uint8)
    i = 0
    for image in images:
        for frame in GetFrames(image):
            if verbose:
                printRep(str(i) + "/" + str(totalFrameCount))
            if frame.mode[0] == "I":
                frame = frame.convert(mode="I")
                frame = frame.point(lambda x: x * (1 / 255))
            frame = np.asarray(frame.convert(mode="L").resize(modelInputSize), dtype=np.uint8)
            frame = 255 * ((frame - np.min(frame)) / (np.max(frame) - np.min(frame)))
            loadedImages[i] = frame
            i += 1
    if verbose:
        printRep(None)
        print("Done")
    return loadedImages


def PrepareSegmentationsForModel(images: List[Image.Image], model: ModelType):
    totalFrameCount = sum([NumFrames(image) for image in images])
    modelInputSize = InputSize(model)
    loadedImages = np.zeros([totalFrameCount, modelInputSize[0], modelInputSize[1]],
                            dtype=bool)
    i = 0
    for image in images:
        for frame in GetFrames(image):
            frame = frame.resize(modelInputSize).convert(mode="1")
            loadedImages[i] = np.asarray(frame)
            i += 1
    return loadedImages


def Detect(model: ModelType, prepared: np.ndarray, batchSize=None):
    print("Running model...")
    print("-" * 100)
    prepared = np.reshape(prepared, list(prepared.shape[:3]) + [1]).astype(np.float32)
    if isinstance(model, tf.lite.Interpreter):
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        if batchSize is None:
            batchSize = prepared.shape[0]
        batchShape = [batchSize] + list(prepared.shape[1:])
        model.resize_tensor_input(input_details[0]['index'], batchShape, strict=True)
        model.allocate_tensors()
        outputImages = np.zeros_like(prepared, dtype=np.float32)
        batchCount = math.ceil(float(prepared.shape[0]) / batchSize)
        for batchNumber in range(batchCount):
            printRep("Batch %d/%d" % (batchNumber + 1, batchCount))
            start = batchNumber * batchSize
            end = (batchNumber + 1) * batchSize
            batchData = prepared[start:end]
            actualBatchSize = batchData.shape[0]
            if actualBatchSize < batchSize:
                paddedData = np.zeros(batchShape, dtype=np.float32)
                paddedData[:actualBatchSize] = batchData
                batchData = paddedData
            model.set_tensor(input_details[0]['index'], batchData)
            model.invoke()
            outputImages[start:end] = model.get_tensor(output_details[0]['index'])[:actualBatchSize]
        printRep(None)
    else:
        if batchSize is not None:
            outputImages = np.zeros_like(prepared, dtype=np.float32)
            batchCount = math.ceil(float(prepared.shape[0]) / batchSize)
            for batchNumber in range(batchCount):
                start = batchNumber * batchSize
                end = (batchNumber + 1) * batchSize
                outputImages[start:end] = model.predict(prepared[start:end])
        else:
            outputImages = model.predict(prepared)
    print("-" * 100)
    print("Done.")
    return outputImages[:, :, :, 0]


def ComputeIOUs(model: Union[tf.keras.Model, tf.lite.Interpreter], images: List[Image.Image],
                segmentations: List[Image.Image], threshold=0.5):
    outputImages = Detect(model, PrepareImagesForModel(images, model)) >= threshold
    segmentations = PrepareSegmentationsForModel(segmentations, model)
    true = segmentations.astype(dtype=bool)
    predicted = outputImages.astype(dtype=bool)
    intersection = np.count_nonzero(np.bitwise_and(predicted, true), axis=(1, 2))
    union = np.count_nonzero(np.bitwise_or(predicted, true), axis=(1, 2))
    return list(intersection / union)
