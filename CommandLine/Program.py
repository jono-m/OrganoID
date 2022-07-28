# Program.py -- a generic class for a sub-program of OrganoID.

import argparse
import pathlib
from abc import ABC, abstractmethod


class Program(ABC):
    def __init__(self):
        self._lastPrint = ""

    @abstractmethod
    def Name(self):
        pass

    @abstractmethod
    def Description(self):
        pass

    def SetupParser(self, parser: argparse.ArgumentParser):
        pass

    def RunProgram(self, parserArgs: argparse.Namespace):
        pass

    @staticmethod
    def AssertDirectoryExists(path: pathlib.Path):
        if not path.is_dir():
            raise Exception("Could not find required directory '" + str(path.absolute()) + "'.")
    
    @staticmethod
    def MakeDirectory(path: pathlib.Path):
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise Exception("Could not find or create directory '" + str(path.absolute()) + "'.")
