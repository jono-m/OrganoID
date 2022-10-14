from PySide6 import QtWidgets
from UI.MainWindow import MainWindow
import multiprocessing
import sys

if __name__ == '__main__':
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()

    window.show()

    app.exec()
