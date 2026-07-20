"""Background worker for remote connection test + diagnostics."""

import logging
from typing import Optional, List

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .models import RemoteProfile, DiagnosticItem, DiagnosticStatus
from .ssh_service import SSHConnectionService
from .diagnostics import run_remote_diagnostics

_log = logging.getLogger(__name__)


class DiagnosticsWorker(QObject):
    """Runs SSH connection test + remote diagnostics on a QThread."""

    # Progress signals
    stage_changed = pyqtSignal(str)  # current stage description
    item_found = pyqtSignal(object)  # DiagnosticItem

    # Completion signals
    finished = pyqtSignal(list)  # List[DiagnosticItem]
    error = pyqtSignal(str)  # fatal error
    host_key_prompt = pyqtSignal(str, str, str)  # hostname, fingerprint, key_type

    def __init__(self, profile: RemoteProfile, password: str = "", parent=None):
        super().__init__(parent)
        self._profile = profile
        self._password = password
        self._cancelled = False
        self._ssh: Optional[SSHConnectionService] = None
        # Host key confirmation
        self._host_key_result: Optional[tuple] = None  # (trust: bool, save: bool)

    def cancel(self):
        self._cancelled = True
        if self._ssh:
            self._ssh.close()

    def confirm_host_key(self, trust: bool, save: bool):
        """Called from GUI thread to confirm/reject host key."""
        self._host_key_result = (trust, save)

    def run(self):
        """Main worker entry point (runs on QThread)."""
        results: List[DiagnosticItem] = []

        try:
            self._ssh = SSHConnectionService()

            # ── Connect ──
            self.stage_changed.emit("Connecting via SSH...")

            # Host key callback (will block worker until user confirms)
            self._host_key_pending = False

            def on_host_key(hostname, fingerprint, key_type):
                self._host_key_pending = True
                self.host_key_prompt.emit(hostname, fingerprint, key_type)
                # Wait for GUI to call confirm_host_key()
                while self._host_key_result is None and not self._cancelled:
                    QThread.msleep(100)
                if self._cancelled:
                    return (False, False)
                result = self._host_key_result
                self._host_key_result = None
                self._host_key_pending = False
                return result

            ok, msg = self._ssh.connect(
                self._profile,
                password=self._password,
                on_host_key=on_host_key,
            )
            if self._cancelled:
                results.append(DiagnosticItem(
                    stage="connection", label="Connection",
                    status=DiagnosticStatus.ERROR, message="Cancelled by user",
                ))
                self.finished.emit(results)
                return

            if not ok:
                results.append(DiagnosticItem(
                    stage="connection", label="Connection",
                    status=DiagnosticStatus.ERROR, message=msg,
                ))
                self.finished.emit(results)
                return

            # Save fingerprint
            fingerprint = msg if ":" in msg else self._ssh.get_fingerprint()
            results.append(DiagnosticItem(
                stage="connection", label="SSH Connection",
                status=DiagnosticStatus.PASS,
                message=f"Connected to {self._profile.host}:{self._profile.port} · {fingerprint}",
                details={"fingerprint": fingerprint},
            ))

            # ── Run diagnostics ──
            diag_results = run_remote_diagnostics(
                self._profile, self._ssh, self._password,
                on_progress=lambda label: self.stage_changed.emit(label),
            )
            for item in diag_results:
                if self._cancelled:
                    break
                results.append(item)
                self.item_found.emit(item)

        except Exception as e:
            _log.error("Diagnostics failed: %s", e, exc_info=True)
            results.append(DiagnosticItem(
                stage="fatal", label="Fatal Error",
                status=DiagnosticStatus.ERROR, message=str(e),
            ))
        finally:
            if self._ssh:
                try:
                    self._ssh.close()
                except Exception:
                    pass
            self._ssh = None

        self.finished.emit(results)
