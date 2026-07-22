"""PyQt6 + paramiko probe."""
import sys
from PyQt6.QtCore import QT_VERSION_STR
from PyQt6.QtWidgets import QApplication, QLabel
import paramiko as _p
app = QApplication(sys.argv)
label = QLabel(f"Qt {QT_VERSION_STR}\nparamiko {_p.__version__}\nOK"); label.resize(360,140); label.show()
raise SystemExit(app.exec())
