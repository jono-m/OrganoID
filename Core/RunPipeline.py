from typing import List, Optional
from pathlib import Path
from Core.Model import LoadFullModel, Detect, LoadLiteModel, PrepareImagesForModel
from Core.Identification import Cleanup, SeparateContours, DetectEdges, Label
from Core.ImageHandling import LoadPILImages, ImagesToHeatmaps, \
    LabeledImagesToColoredImages, DrawRegionsOnImages, ConvertImagesToStacks, SaveAsGIF
from Core.Tracking import Track, Inverse, Overlap


def SaveImages(data, suffix, pilImages, outputPath):
    from Core.ImageHandling import ConvertImagesToPILImageStacks, SavePILImageStack
    stacks = ConvertImagesToPILImageStacks(data, pilImages)

    for stack, pilImage in zip(stacks, pilImages):
        p = Path(pilImage.filename)
        SavePILImageStack(stack, outputPath / (p.stem + suffix + p.suffix))


def MakeDirectory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise Exception("Could not find or create directory '" + str(path.absolute()) + "'.")


def LoadModel(modelPath: Path):
    print("Loading model...")
    print("-" * 100)
    if modelPath.is_file():
        model = LoadLiteModel(modelPath)
    else:
        model = LoadFullModel(modelPath)
    print("-" * 100)
    print("Model loaded.")
    return model


def RunPipeline(modelPath: Path, imagePaths: List[Path], outputPath: Optional[Path],
                threshold: float, batchSize: int, edgeSigma: float, edgeMin: float,
                edgeMax: float, minimumArea: int, fillHoles: bool, removeBorder: bool,
                detectionOutput: bool, binaryOutput: bool, separateContours: bool,
                edges: bool, colorLabeledOutput: bool, idLabeledOutput: bool,
                track: bool, overlay: bool, gif: bool, batch: bool, computeProps: bool):
    model = LoadModel(modelPath)
    # Load the images
    pilImages = LoadPILImages(imagePaths)
    preparedImages = PrepareImagesForModel(pilImages, model)
    detectionImages = Detect(model, preparedImages, batchSize)

    outputImages = {'Prepared Input': preparedImages}

    def Output(name: str, data):
        if outputPath is not None:
            MakeDirectory(outputPath)
            SaveImages(data, "_" + name.lower(), pilImages, outputPath)
            if gif:
                outputImages[name] = data
        else:
            outputImages[name] = data

    if detectionOutput:
        Output('Detection', ImagesToHeatmaps(detectionImages))
    if binaryOutput:
        Output('Binary', detectionImages > float(threshold))
    if separateContours:
        edgeImages = DetectEdges(detectionImages, edgeSigma, edgeMin, edgeMax, threshold)
        if edges:
            Output('Edges', edgeImages)
        labeledImages = SeparateContours(detectionImages, edgeImages, threshold, edgeSigma)
    else:
        labeledImages = Label(detectionImages, threshold)

    cleanedImages = Cleanup(labeledImages, minimumArea, removeBorder, fillHoles)

    if track:
        i = 0
        stacks = ConvertImagesToStacks(cleanedImages, pilImages) if batch else [cleanedImages]
        for stack in stacks:
            stack = Track(stack, 1, Inverse(Overlap))
            cleanedImages[i:(i + stack.shape[0])] = stack
            i += stack.shape[0]

    if overlay:
        overlayImages = DrawRegionsOnImages(cleanedImages, preparedImages, (255, 255, 255), 16,
                                            (0, 255, 0))
        Output('Overlay', overlayImages)

    if gif and outputPath is not None:
        MakeDirectory(outputPath)
        for name in outputImages:
            stacks = ConvertImagesToStacks(outputImages[name], pilImages) if batch else [
                outputImages[name]]
            for stack, original in zip(stacks, pilImages):
                path = Path(original.filename)
                SaveAsGIF(stack, outputPath / (path.stem + "_" + name.lower() + ".gif"))

    if colorLabeledOutput:
        Output('Color-Labeled', LabeledImagesToColoredImages(cleanedImages))

    if idLabeledOutput:
        Output('ID-Labeled', cleanedImages)

    if outputPath is not None and computeProps:
        from Core.Analyze import AnalyzeAndExport
        stacks = ConvertImagesToStacks(cleanedImages, pilImages) if batch else [cleanedImages]
        for stack, original in zip(stacks, pilImages):
            path = Path(original.filename)
            AnalyzeAndExport(stack, outputPath / (path.stem + "_data.xlsx"))

    return outputImages
