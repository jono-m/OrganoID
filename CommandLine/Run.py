from CommandLine.Program import Program
import argparse
import pathlib


class Run(Program):
    def Name(self):
        return "run"

    def Description(self):
        return "Identifies and analyzes organoids in microscopy images."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("modelPath", help="Path to trained OrganoID model.", type=pathlib.Path)
        parser.add_argument("imagesPath", help="Path to images to analyze. Accepts image files, "
                                               "TIFF stacks, directories, and regular expressions.",
                            type=pathlib.Path)
        parser.add_argument("outputPath", help="Directory where images will be saved.",
                            type=pathlib.Path)
        parser.add_argument("-T", dest="threshold", default=0.5,
                            help="Belief threshold to consider a pixel an organoid.",
                            type=float)
        parser.add_argument("-BS", dest="batchSize", default=16,
                            help="Number of images to process at a time.",
                            type=float)
        parser.add_argument("-ES", dest="edgeSigma", default=2,
                            help="Standard deviation of Gaussian kernel used for edge "
                                 "detection.",
                            type=float)
        parser.add_argument("-EL", dest="edgeMin", default=0.005,
                            help="Low threshold for edge detection hysteresis filter.",
                            type=float)
        parser.add_argument("-EH", dest="edgeMax", default=0.05,
                            help="High threshold for edge detection hysteresis filter.",
                            type=float)
        parser.add_argument("-A", dest="minimumArea", default=100,
                            help="Ignore objects with an area smaller than this number of "
                                 "pixels.",
                            type=int)
        parser.add_argument("--nofill", action="store_true",
                            help="If set, holes in detected objects will not be filled.")
        parser.add_argument("--border", action="store_true",
                            help="If set, objects in contact with image borders will be kept.")
        parser.add_argument("--belief", action="store_true",
                            help="If set, network belief images will be produced.")
        parser.add_argument("--binary", action="store_true",
                            help="If set, thresholded images will be produced.")
        parser.add_argument("--no_separation", action="store_true",
                            help="If set, contours in physical contact will not be separated.")
        parser.add_argument("--edges", action="store_true",
                            help="If set, edge detection images will be produced.")
        parser.add_argument("--colorize", action="store_true",
                            help="If set, colored labeled images will be produced.")
        parser.add_argument("--track", action="store_true",
                            help="If set, images will be processed as timelapse data and labeled "
                                 "according to individual organoid ID.")
        parser.add_argument("--overlay", action="store_true",
                            help="If this is set, a label-overlay image will also be produced for "
                                 "each frame.")
        parser.add_argument("--gif", action="store_true",
                            help="Only used if. If this is set, overlay images will"
                                 "be produced as a .gif")
        parser.add_argument("--show", action="store_true",
                            help="If this is set, the last image to be produced will be shown.")
        parser.add_argument("--batch", action="store_true",
                            help="If this is set, image stacks will be separately tracked.")

    @staticmethod
    def SaveImages(data, suffix, pilImages, outputPath):
        from Core.ImageHandling import ConvertImagesToPILImageStacks, SavePILImageStack
        stacks = ConvertImagesToPILImageStacks(data, pilImages)

        for stack, pilImage in zip(stacks, pilImages):
            p = pathlib.Path(pilImage.filename)
            SavePILImageStack(stack, outputPath / (p.stem + suffix + p.suffix))

    def RunProgram(self, parserArgs: argparse.Namespace):
        import numpy as np
        from Core.Model import LoadModel, Detect, LoadLiteModel, PrepareImagesForModel
        from Core.Identification import Cleanup, SeparateContours, DetectEdges, Label
        from Core.ImageHandling import LoadPILImages, ImagesToHeatmaps, \
            LabeledImagesToColoredImages, DrawRegionsOnImages, SaveAsGIF, ConvertImagesToStacks
        from Core.Tracking import Track, Inverse, Overlap

        # Load the images
        print("Loading model...")
        print("-" * 100)
        if parserArgs.modelPath.is_file():
            model = LoadLiteModel(parserArgs.modelPath)
        else:
            model = LoadModel(parserArgs.modelPath)
        print("-" * 100)
        print("Model loaded.")
        pilImages = LoadPILImages(parserArgs.imagesPath)
        self.MakeDirectory(parserArgs.outputPath)
        images = PrepareImagesForModel(pilImages, model)
        detections = Detect(model, images, parserArgs.batchSize)
        if parserArgs.belief:
            heatmaps = ImagesToHeatmaps(detections)
            self.SaveImages(heatmaps, "_belief", pilImages, parserArgs.outputPath)
        if parserArgs.binary:
            thresholded: np.ndarray = detections > float(parserArgs.threshold)
            self.SaveImages(thresholded, "_binary", pilImages, parserArgs.outputPath)
        if parserArgs.no_separation:
            labeled = Label(detections, parserArgs.threshold)
        else:
            edges = DetectEdges(detections, parserArgs.edgeSigma, parserArgs.edgeMin,
                                parserArgs.edgeMax, parserArgs.threshold)
            if parserArgs.edges:
                self.SaveImages(edges, "_edges", pilImages, parserArgs.outputPath)
            labeled = SeparateContours(detections, edges, parserArgs.threshold,
                                       parserArgs.edgeSigma)
        final = Cleanup(labeled, parserArgs.minimumArea, not parserArgs.border,
                        not parserArgs.nofill)
        if parserArgs.track:
            i = 0
            stacks = ConvertImagesToStacks(final, pilImages) if parserArgs.batch else [final]
            for stack in stacks:
                stack = Track(stack, 1, Inverse(Overlap))
                final[i:(i + stack.shape[0])] = stack
                i += stack.shape[0]

        if parserArgs.overlay or parserArgs.gif:
            overlay = DrawRegionsOnImages(final, images, (255, 255, 255), 16, (0, 255, 0))

            if parserArgs.overlay:
                self.SaveImages(overlay, "_overlay", pilImages, parserArgs.outputPath)

            if parserArgs.gif:
                if parserArgs.batch:
                    stacks = ConvertImagesToStacks(overlay, pilImages)
                    for stack, original in zip(stacks, pilImages):
                        path = pathlib.Path(original.filename)
                        SaveAsGIF(stack, parserArgs.outputPath / (path.stem + "_overlay.gif"))
                else:
                    SaveAsGIF(overlay, parserArgs.outputPath / "overlay.gif")

        if parserArgs.colorize:
            rgb = LabeledImagesToColoredImages(final)
            self.SaveImages(rgb, "_color", pilImages, parserArgs.outputPath)

        self.SaveImages(final, "_labeled", pilImages, parserArgs.outputPath)
