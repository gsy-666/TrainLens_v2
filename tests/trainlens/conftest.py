"""Global test fixtures — isolate HistoryStore to prevent user-data pollution.

Strategy:
1. At session start: replace history module's _store singleton with a temp-dir store.
2. Before EACH test: re-reset _store (other tests may pollute it via get_history_store(dir=...)).
3. All JobManager instances automatically use the isolated store via _get_history_store().
4. At session end: restore original singleton; temp records are discarded.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


# ── session-scoped state ──────────────────────────────────────────────────

_original_store = None
_session_temp_dir = None


def pytest_configure(config):
    """Before any test: replace history singleton with temp-dir store."""
    global _original_store, _session_temp_dir

    from anylabeling.services.training_center import history as hist_mod

    # Save original singleton
    _original_store = hist_mod._store

    # Create isolated store pointed at temp dir
    _session_temp_dir = Path(tempfile.mkdtemp(prefix="trainlens-test-"))
    from anylabeling.services.training_center.history import HistoryStore
    hist_mod._store = HistoryStore(history_dir=_session_temp_dir)


def pytest_unconfigure(config):
    """After all tests: restore original history singleton."""
    global _original_store, _session_temp_dir

    from anylabeling.services.training_center import history as hist_mod

    # Restore original singleton
    hist_mod._store = _original_store

    # Clean up temp dir
    if _session_temp_dir and _session_temp_dir.exists():
        shutil.rmtree(_session_temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _reset_history_singleton():
    """Before each test: ensure _store points to the session-isolated temp dir.

    Some tests (e.g. test_get_history_store_with_custom_dir_creates_new) call
    get_history_store(history_dir=...) which overwrites the global _store.
    Also resets JobManager singleton's _history_store (JobManager is a singleton
    and test fixtures inject _history_store directly).
    """
    from anylabeling.services.training_center import history as hist_mod
    from anylabeling.services.training_center.history import HistoryStore
    from anylabeling.services.training_center.job_manager import JobManager

    global _session_temp_dir
    if _session_temp_dir and _session_temp_dir.exists():
        hist_mod._store = HistoryStore(history_dir=_session_temp_dir)

    # Reset JobManager singleton's _history_store to force lazy re-init
    # from the (now-reset) global singleton.
    try:
        jm = JobManager()
        jm._history_store = None
    except Exception:
        pass

    yield
