# Split.py -- a sub-program that randomly splits ground-truth data into training and validation
# sets.

from CommandLine.Program import Program
import argparse
import pathlib


class Split(Program):
    def Name(self):
        return "split"

    def Description(self):
        return "Split up ground-truth data for model training and validation."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("inputPath", help="Path to image and segmentation data. "
                                              "Directory should have the subfolders images/"
                                              " and segmentations/",
                            type=pathlib.Path)
        parser.add_argument("outputPath", type=pathlib.Path,
                            help="Path where the split images will be saved.")
        parser.add_argument("-VS", dest='validationSplit', default=0.2,
                            help="Fraction of images to split for validation (0.0-1.0).",
                            type=float)
        parser.add_argument("-TS", dest='testSplit', default=0.2,
                            help="Fraction of images to split for testing (0.0-1.0).", type=float)

    def RunProgram(self, parserArgs: argparse.Namespace):
        from Core.DataPreparation import SplitData
        [self.AssertDirectoryExists(x) for x in [parserArgs.inputPath,
                                                 parserArgs.inputPath / "images",
                                                 parserArgs.inputPath / "segmentations"]]
        self.MakeDirectory(parserArgs.outputPath / "training" / "images")
        self.MakeDirectory(parserArgs.outputPath / "training" / "segmentations")
        self.MakeDirectory(parserArgs.outputPath / "validation" / "images")
        self.MakeDirectory(parserArgs.outputPath / "validation" / "segmentations")
        self.MakeDirectory(parserArgs.outputPath / "testing" / "images")
        self.MakeDirectory(parserArgs.outputPath / "testing" / "segmentations")
        SplitData(list((parserArgs.inputPath / "images").iterdir()),
                  list((parserArgs.inputPath / "segmentations").iterdir()),
                  parserArgs.validationSplit, parserArgs.testSplit, parserArgs.outputPath)
