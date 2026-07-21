"""Remote Profile Editor Widget."""

import logging
import os
import uuid
from typing import Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QSpinBox, QTextEdit,
    QVBoxLayout, QWidget, QProgressBar, QApplication,
)

from anylabeling.services.training_center.remote.models import (
    RemoteProfile, AuthMethod, DiagnosticItem, DiagnosticStatus,
)
from anylabeling.services.training_center.remote.storage import get_profile_store
from anylabeling.services.training_center.remote.diagnostics_worker import DiagnosticsWorker


_log = logging.getLogger(__name__)


class HostKeyDialog(QDialog):
    """Dialog asking user to trust a new SSH host key."""

    def __init__(self, hostname: str, fingerprint: str, key_type: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SSH Host Key Verification")
        self.setMinimumWidth(450)
        self._trust = False
        self._save = False

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            f"The authenticity of host '{hostname}' can't be established.\n\n"
            f"Key type: {key_type}\n"
            f"Fingerprint: {fingerprint}\n\n"
            f"Are you sure you want to continue connecting?"
        ))

        btn_layout = QHBoxLayout()
        trust_once_btn = QPushButton("Trust Once")
        trust_once_btn.clicked.connect(self._trust_once)
        trust_save_btn = QPushButton("Trust && Save")
        trust_save_btn.clicked.connect(self._trust_save)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(trust_once_btn)
        btn_layout.addWidget(trust_save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _trust_once(self):
        self._trust = True
        self._save = False
        self.accept()

    def _trust_save(self):
        self._trust = True
        self._save = True
        self.accept()

    @property
    def trust(self) -> bool:
        return self._trust

    @property
    def save(self) -> bool:
        return self._save


class RemoteProfileWidget(QGroupBox):
    """Widget for editing a single RemoteProfile and running diagnostics."""

    profile_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Remote SSH Configuration", parent)
        self._store = get_profile_store()
        self._profile: Optional[RemoteProfile] = None
        self._password = ""  # session-only
        self._worker: Optional[DiagnosticsWorker] = None
        self._thread: Optional[QThread] = None
        self._results: list = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Profile selector
        top = QHBoxLayout()
        top.addWidget(QLabel("Profile:"))
        self._combo = QComboBox()
        self._combo.currentIndexChanged.connect(self._on_profile_selected)
        top.addWidget(self._combo, 1)

        self._new_btn = QPushButton("New")
        self._new_btn.clicked.connect(self._new_profile)
        top.addWidget(self._new_btn)

        self._delete_btn = QPushButton("Delete")
        self._delete_btn.clicked.connect(self._delete_profile)
        top.addWidget(self._delete_btn)
        layout.addLayout(top)

        # Form
        form = QFormLayout()

        self._name_edit = QLineEdit()
        form.addRow("Profile Name:", self._name_edit)

        self._host_edit = QLineEdit()
        form.addRow("Host:", self._host_edit)

        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(22)
        form.addRow("Port:", self._port_spin)

        self._user_edit = QLineEdit()
        form.addRow("Username:", self._user_edit)

        self._auth_combo = QComboBox()
        self._auth_combo.addItem("SSH Key", AuthMethod.SSH_KEY.value)
        self._auth_combo.addItem("Password", AuthMethod.PASSWORD.value)
        self._auth_combo.currentIndexChanged.connect(self._on_auth_changed)
        form.addRow("Authentication:", self._auth_combo)

        # SSH Key path
        key_layout = QHBoxLayout()
        self._key_edit = QLineEdit()
        key_layout.addWidget(self._key_edit)
        self._browse_key_btn = QPushButton("Browse")
        self._browse_key_btn.clicked.connect(self._browse_key)
        key_layout.addWidget(self._browse_key_btn)
        self._key_row = form.addRow("Private Key:", key_layout)

        # Password (session-only)
        self._pw_edit = QLineEdit()
        self._pw_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._pw_edit.setVisible(False)
        self._pw_label = QLabel("Password:")
        self._pw_label.setVisible(False)
        form.addRow(self._pw_label, self._pw_edit)

        self._workspace_edit = QLineEdit()
        form.addRow("Remote Workspace:", self._workspace_edit)

        self._python_edit = QLineEdit()
        self._python_edit.setPlaceholderText("e.g. python3 or /path/to/python")
        form.addRow("Remote Python:", self._python_edit)

        layout.addLayout(form)

        # Buttons
        btn_layout = QHBoxLayout()
        self._save_btn = QPushButton("Save Profile")
        self._save_btn.clicked.connect(self._save_profile)
        btn_layout.addWidget(self._save_btn)

        self._test_btn = QPushButton("Test Connection")
        self._test_btn.clicked.connect(self._test_connection)
        self._test_btn.setStyleSheet("font-weight: bold;")
        btn_layout.addWidget(self._test_btn)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._cancel_test)
        self._cancel_btn.setVisible(False)
        btn_layout.addWidget(self._cancel_btn)

        layout.addLayout(btn_layout)

        # Progress
        self._progress = QProgressBar()
        self._progress.setRange(0, 0)  # indeterminate
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._stage_label = QLabel("")
        self._stage_label.setVisible(False)
        layout.addWidget(self._stage_label)

        # Results
        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMaximumHeight(200)
        self._results_text.setVisible(False)
        layout.addWidget(self._results_text)

        # Load profiles
        self._refresh_profiles()

    def _refresh_profiles(self):
        self._combo.blockSignals(True)
        self._combo.clear()
        profiles = self._store.list_all()
        for p in profiles:
            self._combo.addItem(p.name or p.host, p.profile_id)
        self._combo.blockSignals(False)
        self._combo.setCurrentIndex(-1)
        self._clear_form()

    def _on_profile_selected(self, idx: int):
        if idx < 0:
            self._profile = None
            self._clear_form()
            return
        pid = self._combo.itemData(idx)
        self._profile = self._store.get(pid)
        if self._profile:
            self._populate_form(self._profile)

    def _new_profile(self):
        self._profile = RemoteProfile(
            profile_id=f"rp-{uuid.uuid4().hex[:12]}",
            name="New Profile",
        )
        self._populate_form(self._profile)

    def _delete_profile(self):
        if not self._profile:
            return
        reply = QMessageBox.question(
            self, "Delete Profile",
            f"Delete profile '{self._profile.name}'?",
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._store.delete(self._profile.profile_id)
            self._profile = None
            self._refresh_profiles()

    def _populate_form(self, p: RemoteProfile):
        self._name_edit.setText(p.name)
        self._host_edit.setText(p.host)
        self._port_spin.setValue(p.port)
        self._user_edit.setText(p.username)
        idx = self._auth_combo.findData(p.auth_method.value)
        if idx >= 0:
            self._auth_combo.setCurrentIndex(idx)
        self._key_edit.setText(p.private_key_path)
        self._workspace_edit.setText(p.remote_workspace)
        self._python_edit.setText(p.remote_python)

    def _clear_form(self):
        for w in [self._name_edit, self._host_edit, self._user_edit,
                   self._key_edit, self._python_edit, self._workspace_edit]:
            w.clear()
        self._port_spin.setValue(22)
        self._auth_combo.setCurrentIndex(0)
        self._pw_edit.clear()
        self._results_text.clear()
        self._results_text.setVisible(False)
        self._progress.setVisible(False)
        self._stage_label.setVisible(False)

    def _read_form(self) -> RemoteProfile:
        if not self._profile:
            self._profile = RemoteProfile(
                profile_id=f"rp-{uuid.uuid4().hex[:12]}",
            )
        p = self._profile
        p.name = self._name_edit.text().strip()
        p.host = self._host_edit.text().strip()
        p.port = self._port_spin.value()
        p.username = self._user_edit.text().strip()
        p.auth_method = AuthMethod(self._auth_combo.currentData())
        p.private_key_path = self._key_edit.text().strip()
        p.remote_workspace = self._workspace_edit.text().strip()
        p.remote_python = self._python_edit.text().strip()
        return p

    def _on_auth_changed(self):
        is_key = self._auth_combo.currentData() == AuthMethod.SSH_KEY.value
        self._key_edit.setVisible(is_key)
        self._browse_key_btn.setVisible(is_key)
        self._pw_edit.setVisible(not is_key)
        self._pw_label.setVisible(not is_key)

    def _browse_key(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH Private Key", os.path.expanduser("~/.ssh"),
            "All Files (*)"
        )
        if path:
            self._key_edit.setText(path)

    def _save_profile(self):
        p = self._read_form()
        if not p.name or not p.host:
            QMessageBox.warning(self, "Missing Fields", "Profile name and host are required.")
            return
        self._password = self._pw_edit.text()  # session-only
        self._store.save(p)
        self._profile = p
        self._refresh_profiles()
        # Re-select
        for i in range(self._combo.count()):
            if self._combo.itemData(i) == p.profile_id:
                self._combo.setCurrentIndex(i)
                break
        self.profile_changed.emit()
        QMessageBox.information(self, "Saved", f"Profile '{p.name}' saved.")

    def _test_connection(self):
        p = self._read_form()
        if not p.host:
            QMessageBox.warning(self, "Missing Fields", "Host is required.")
            return
        self._password = self._pw_edit.text()

        self._test_btn.setVisible(False)
        self._cancel_btn.setVisible(True)
        self._progress.setVisible(True)
        self._stage_label.setVisible(True)
        self._results_text.clear()
        self._results_text.setVisible(True)

        self._worker = DiagnosticsWorker(p, password=self._password)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)

        self._worker.stage_changed.connect(self._stage_label.setText)
        self._worker.item_found.connect(self._on_item)
        self._worker.finished.connect(self._on_diagnostics_finished)
        self._worker.error.connect(self._on_diagnostics_error)
        self._worker.host_key_prompt.connect(self._on_host_key_prompt)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    def _cancel_test(self):
        if self._worker:
            self._worker.cancel()
        self._reset_test_ui()

    def _on_host_key_prompt(self, hostname, fingerprint, key_type):
        dlg = HostKeyDialog(hostname, fingerprint, key_type, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._worker.confirm_host_key(dlg.trust, dlg.save)
            if dlg.save and self._profile:
                self._profile.known_host_fingerprint = fingerprint
                self._store.save(self._profile)
        else:
            self._worker.confirm_host_key(False, False)

    def _on_item(self, item: DiagnosticItem):
        _log.info("Diagnostic item received by UI: [%s] %s", item.status.value, item.label)
        self._render_item(item)

    def _render_item(self, item: DiagnosticItem):
        color = {
            DiagnosticStatus.PASS: "green",
            DiagnosticStatus.WARNING: "orange",
            DiagnosticStatus.ERROR: "red",
            DiagnosticStatus.PENDING: "gray",
        }.get(item.status, "black")
        self._results_text.append(
            f'<span style="color:{color}; font-weight:bold">[{item.status.value}]</span> '
            f'<b>{item.label}</b>: {item.message}'
        )

    def _on_diagnostics_finished(self, results: list):
        self._results = results
        _log.info("Diagnostics finished: %d items received", len(results))

        # Fallback: render any items not yet added to the text widget
        # (handles cases where item_found signals were missed)
        current_text = self._results_text.toPlainText()
        for item in results:
            # Check if this item's label already appears in the output
            if item.label not in current_text:
                _log.info("Rendering missed item: [%s] %s", item.status.value, item.label)
                self._render_item(item)

        self._results_text.append(f"\n--- Diagnostics Complete ({len(results)} items) ---")
        self._cleanup_thread()
        self._cleanup_thread()

    def _on_diagnostics_error(self, err: str):
        self._results_text.append(f'<span style="color:red">Error: {err}</span>')
        self._cleanup_thread()

    def _cleanup_thread(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
            self._thread = None
        self._worker = None
        self._reset_test_ui()

    def _reset_test_ui(self):
        self._test_btn.setVisible(True)
        self._cancel_btn.setVisible(False)
        self._progress.setVisible(False)
        self._stage_label.setVisible(False)

    def get_profile(self) -> Optional[RemoteProfile]:
        return self._profile

    def get_session_password(self) -> str:
        """Return current session password (never persisted)."""
        return self._pw_edit.text()
