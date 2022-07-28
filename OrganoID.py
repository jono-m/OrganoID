# OrganoID.py -- Entry-point for the OrganoID suite.

import argparse
from CommandLine.Augment import Augment
from CommandLine.Run import Run
from CommandLine.Split import Split
from CommandLine.Train import Train

# List of sub-programs.
programs = [Augment, Run, Split, Train]
# Parse sub-program selection
parser = argparse.ArgumentParser(
    description="OrganoID: deep learning for organoid image analysis.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(dest="subparser_name")

# Instantiate sub-programs
programs = [program() for program in programs]

# Load sub-program arguments
for program in programs:
    program.SetupParser(
        subparsers.add_parser(program.Name(),
                              help=program.Description(),
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter))

# Parse all command-line arguments.
args = parser.parse_args()

# Run the selected sub-program with the parsed arguments.
for program in programs:
    if program.Name() == args.subparser_name:
        program.RunProgram(args)
        break
