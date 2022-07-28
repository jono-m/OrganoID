# Augment.py -- Sub-program to augment training data.

from CommandLine.Program import Program
import argparse
import pathlib


class Augment(Program):
    def Name(self):
        return "augment"

    def Description(self):
        return "Augment images for training."

    def SetupParser(self, parser: argparse.ArgumentParser):
        parser.add_argument("inputPath", help="Path to image and segmentation data. "
                                              "Directory should have the subfolders images/ and "
                                              "segmentations/",
                            type=pathlib.Path)
        parser.add_argument("outputPath", type=pathlib.Path,
                            help="Path where the augmented images will be saved.")
        parser.add_argument("augmentCount", help="Number of augmented images to produce.", type=int)

    def RunProgram(self, parserArgs: argparse.Namespace):
        from Core.DataPreparation import AugmentImages

        self.MakeDirectory(parserArgs.outputPath / "images")
        self.MakeDirectory(parserArgs.outputPath / "segmentations")
        [self.AssertDirectoryExists(x) for x in [parserArgs.outputPath,
                                                 parserArgs.outputPath / "images",
                                                 parserArgs.outputPath / "segmentations",
                                                 parserArgs.inputPath,
                                                 parserArgs.inputPath / "images",
                                                 parserArgs.inputPath / "segmentations"]]
        AugmentImages(parserArgs.inputPath / "images",
                      parserArgs.inputPath / "segmentations",
                      parserArgs.outputPath,
                      parserArgs.augmentCount)
