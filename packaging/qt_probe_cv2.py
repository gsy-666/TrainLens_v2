"""PyQt6 + cv2 probe."""
import sys
from PyQt6.QtCore import QT_VERSION_STR
from PyQt6.QtWidgets import QApplication, QLabel
import cv2
app = QApplication(sys.argv)
label = QLabel(f"Qt {QT_VERSION_STR}\ncv2 {cv2.__version__}\nOK"); label.resize(360,140); label.show()
raise SystemExit(app.exec())
