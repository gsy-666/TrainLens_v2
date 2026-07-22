"""PyQt6 + numpy probe."""
import sys
from PyQt6.QtCore import QT_VERSION_STR
from PyQt6.QtWidgets import QApplication, QLabel
import numpy as _numpy

app = QApplication(sys.argv)
label = QLabel(
    f"Qt {QT_VERSION_STR}\n"
    f"numpy {_numpy.__version__}\n"
    "OK"
)
label.resize(360, 140)
label.show()
raise SystemExit(app.exec())
