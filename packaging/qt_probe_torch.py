"""PyQt6 + torch probe."""
import sys
from PyQt6.QtCore import QT_VERSION_STR
from PyQt6.QtWidgets import QApplication, QLabel
import torch as _t
app = QApplication(sys.argv)
label = QLabel(f"Qt {QT_VERSION_STR}\ntorch {_t.__version__}\nOK"); label.resize(360,140); label.show()
raise SystemExit(app.exec())
