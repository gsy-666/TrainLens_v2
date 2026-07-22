"""Minimal PyQt6 probe — tests if PyQt6 can be frozen and run."""
import sys
from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
from PyQt6.QtWidgets import QApplication, QLabel

app = QApplication(sys.argv)
label = QLabel(
    f"Qt {QT_VERSION_STR}\n"
    f"PyQt {PYQT_VERSION_STR}\n"
    "QtCore loaded successfully"
)
label.resize(360, 140)
label.show()
raise SystemExit(app.exec())
