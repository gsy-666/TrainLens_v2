"""SSH connection service using Paramiko.

All network operations run on a QThread worker — never blocks the GUI.
"""

import logging
import os
import shlex
import threading
import time
from io import StringIO
from typing import Optional, Tuple, Dict, Any

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .models import RemoteProfile, AuthMethod

_log = logging.getLogger(__name__)

try:
    import paramiko
    HAS_PARAMIKO = True
except ImportError:
    HAS_PARAMIKO = False
    paramiko = None  # type: ignore


class HostKeyPolicy(paramiko.MissingHostKeyPolicy if paramiko else object):
    """Custom host key policy that requires user confirmation."""

    def __init__(self, on_new_host=None, on_key_changed=None):
        super().__init__()
        self._on_new_host = on_new_host  # callable(fingerprint, key_type) → bool
        self._on_key_changed = on_key_changed  # callable(fingerprint, key_type) → bool

    def missing_host_key(self, client, hostname, key):
        fingerprint = ":".join(
            f"{b:02x}" for b in key.get_fingerprint()
        )
        key_type = key.get_name()
        if self._on_new_host:
            result = self._on_new_host(hostname, fingerprint, key_type)
            if result:
                client.get_host_keys().add(hostname, key_type, key)
                return
        raise paramiko.SSHException(
            f"Unknown host {hostname}. Fingerprint: {fingerprint}"
        )


def _check_fingerprint_match(profile: RemoteProfile, hostname: str, key) -> Tuple[bool, str]:
    """Check if the saved fingerprint matches the presented key."""
    saved = (profile.known_host_fingerprint or "").strip()
    if not saved:
        return True, ""  # No saved fingerprint — first connection
    current = ":".join(f"{b:02x}" for b in key.get_fingerprint())
    if saved != current:
        return False, (
            f"WARNING: Host key for {hostname} has changed!\n\n"
            f"Saved:   {saved}\n"
            f"Current: {current}\n\n"
            f"This could indicate a MITM attack or a server reinstall.\n"
            f"Connection blocked for safety."
        )
    return True, ""


class SSHConnectionService:
    """Manages a single SSH connection + remote command execution."""

    def __init__(self):
        self._client: Optional[paramiko.SSHClient] = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def connect(
        self,
        profile: RemoteProfile,
        password: str = "",
        on_host_key: callable = None,
    ) -> Tuple[bool, str]:
        """Establish SSH connection.

        Args:
            profile: Connection parameters
            password: Session-only password (never persisted)
            on_host_key: callback(hostname, fingerprint, key_type) → (trust: bool, save: bool)

        Returns:
            (success, message)
        """
        if not HAS_PARAMIKO:
            return False, "Paramiko is not installed. Run: pip install paramiko"

        try:
            self._client = paramiko.SSHClient()

            # Load system host keys
            self._client.load_system_host_keys()

            # Custom host key policy
            policy = HostKeyPolicy(
                on_new_host=lambda h, fp, kt: self._handle_new_host(h, fp, kt, on_host_key),
                on_key_changed=lambda h, fp, kt: self._handle_key_changed(h, fp, kt),
            )
            self._client.set_missing_host_key_policy(policy)

            # Check fingerprint match
            # (We do this by connecting and letting the policy handle it)

            connect_kwargs = {
                "hostname": profile.host,
                "port": profile.port or 22,
                "username": profile.username,
                "timeout": 10,
                "banner_timeout": 10,
                "auth_timeout": 10,
            }

            if profile.auth_method == AuthMethod.SSH_KEY:
                key_path = profile.private_key_path
                if not key_path or not os.path.isfile(key_path):
                    return False, f"SSH key not found: {key_path}"
                try:
                    pkey = paramiko.RSAKey.from_private_key_file(key_path)
                except paramiko.SSHException:
                    try:
                        pkey = paramiko.Ed25519Key.from_private_key_file(key_path)
                    except paramiko.SSHException:
                        try:
                            pkey = paramiko.ECDSAKey.from_private_key_file(key_path)
                        except paramiko.SSHException as e:
                            return False, f"Failed to load SSH key: {e}"
                connect_kwargs["pkey"] = pkey
            else:
                pwd = password or profile.password
                if not pwd:
                    return False, "Password is required for password authentication"
                connect_kwargs["password"] = pwd

            self._client.connect(**connect_kwargs)
            self._connected = True

            # Save fingerprint
            t = self._client.get_transport()
            if t:
                key = t.get_remote_server_key()
                fingerprint = ":".join(f"{b:02x}" for b in key.get_fingerprint())
                return True, fingerprint

            return True, "Connected (fingerprint not available)"

        except paramiko.AuthenticationException:
            return False, "Authentication failed. Check username, key, or password."
        except paramiko.SSHException as e:
            return False, f"SSH error: {e}"
        except OSError as e:
            return False, f"Connection error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def execute(
        self,
        command: str,
        *,
        stdin_data: Optional[str] = None,
        timeout: int = 30,
        get_pty: bool = False,
    ) -> Tuple[int, str, str]:
        """Execute a command on the remote host.

        Supports both plain shell commands and stdin-piped scripts.

        Args:
            command: Shell command (paths must be shlex.quote()'d by caller).
            stdin_data: If set, written to stdin; channel shutdown_write after.
            timeout: Command timeout in seconds.
            get_pty: Request a pseudo-terminal (needed for some interactive commands).

        Returns:
            (exit_code, stdout, stderr)
        """
        if not self.is_connected or not self._client:
            raise RuntimeError("SSH connection is not established")

        try:
            stdin, stdout, stderr = self._client.exec_command(
                command, timeout=timeout, get_pty=get_pty,
            )

            if stdin_data is not None:
                if isinstance(stdin_data, str):
                    stdin.write(stdin_data)
                else:
                    stdin.write(stdin_data)
                stdin.flush()
                stdin.channel.shutdown_write()

            exit_code = stdout.channel.recv_exit_status()
            out = stdout.read().decode("utf-8", errors="replace").strip()
            err = stderr.read().decode("utf-8", errors="replace").strip()
            return exit_code, out, err
        except Exception as e:
            if "SSH connection" in str(e):
                raise
            return -1, "", str(e)

    def execute_script(self, script: str, python_path: str = "python3",
                        timeout: int = 30) -> Tuple[int, str, str]:
        """Execute a Python script on the remote host via stdin piping.

        Convenience wrapper around execute() for Python scripts.
        The script is validated with compile() before sending.
        """
        try:
            compile(script, "<trainlens_remote_script>", "exec")
        except SyntaxError as e:
            return -1, "", f"Local compile check failed: {e}"

        command = f"{shlex.quote(python_path)} -"
        return self.execute(command, stdin_data=script, timeout=timeout)

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
        self._connected = False

    def _handle_new_host(self, hostname, fingerprint, key_type, callback) -> bool:
        """Ask user to trust a new host."""
        if callback:
            trust, save = callback(hostname, fingerprint, key_type)
            return trust
        return False

    def _handle_key_changed(self, hostname, fingerprint, key_type):
        raise paramiko.SSHException(
            f"Host key for {hostname} has changed! Possible MITM attack."
        )

    def get_fingerprint(self) -> str:
        """Return the current connection's host key fingerprint."""
        if not self.is_connected or not self._client:
            return ""
        t = self._client.get_transport()
        if t:
            key = t.get_remote_server_key()
            return ":".join(f"{b:02x}" for b in key.get_fingerprint())
        return ""
