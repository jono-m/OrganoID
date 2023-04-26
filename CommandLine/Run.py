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
        parser.add_argument("--batch", action="store_true",
                            help="If this is set, image stacks will be separately tracked.")
        parser.add_argument("--analyze", action="store_true",
                            help="If this is set, images will be analyzed to an Excel file.")

    @staticmethod
    def SaveImages(data, suffix, pilImages, outputPath):
        from Core.ImageHandling import ConvertImagesToPILImageStacks, SavePILImageStack
        stacks = ConvertImagesToPILImageStacks(data, pilImages)

        for stack, pilImage in zip(stacks, pilImages):
            p = pathlib.Path(pilImage.filename)
            SavePILImageStack(stack, outputPath / (p.stem + suffix + p.suffix))

    def RunProgram(self, parserArgs: argparse.Namespace):
        from Core.RunPipeline import RunPipeline

        RunPipeline(parserArgs.modelPath, parserArgs.imagesPath, parserArgs.outputPath,
                    parserArgs.threshold, parserArgs.batchSize, parserArgs.edgeSigma,
                    parserArgs.edgeMin, parserArgs.edgeMax, parserArgs.minimumArea,
                    not parserArgs.nofill, not parserArgs.border, parserArgs.belief,
                    parserArgs.binary, not parserArgs.no_separation, parserArgs.edges,
                    parserArgs.colorize, True, parserArgs.track, parserArgs.overlay,
                    parserArgs.gif, parserArgs.batch, parserArgs.analyze)
