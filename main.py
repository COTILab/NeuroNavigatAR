import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nnar.main_gui import MainWindow
from PyQt5 import QtWidgets

if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication(sys.argv)

    mainWin = MainWindow()
    mainWin.show()
    app.exec_()
