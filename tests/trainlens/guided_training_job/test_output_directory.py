"""Tests for Guided Training output directory config passing and write-back."""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp(prefix="trainlens_test_output_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. project_readonly defaults to False
# ---------------------------------------------------------------------------


class TestProjectReadonlyDefault:

    def test_readonly_defaults_to_false(self, qapp):
        from anylabeling.views.training.guided_training_widget import (
            GuidedTrainingWidget,
        )
        with patch(
            "anylabeling.views.training.guided_training_widget.get_config",
            side_effect=Exception("no config"),
        ):
            widget = GuidedTrainingWidget(
                parent=None, image_list=[], output_dir="", supported_shape=[]
            )
            assert widget.project_readonly is False
            widget.shutdown()


# ---------------------------------------------------------------------------
# 2. train_args contains user-specified project/name (via config widgets)
# ---------------------------------------------------------------------------


def _make_widget_with_config(qapp, temp_dir, project, name, task_type="Detect"):
    from anylabeling.views.training.guided_training_widget import (
        GuidedTrainingWidget,
    )
    widget = GuidedTrainingWidget(
        parent=None, image_list=[], output_dir=temp_dir, supported_shape=[]
    )
    widget.selected_task_type = task_type
    widget.names = []
    widget.ensure_config_tab_initialized()
    widget.config_widgets["project"].setText(project)
    widget.config_widgets["name"].setText(name)
    widget.config_widgets["model"].setText("yolo11n.pt")
    widget.config_widgets["data"].setText(os.path.join(temp_dir, "data.yaml"))
    return widget


def _make_data_yaml(temp_dir):
    p = os.path.join(temp_dir, "data.yaml")
    with open(p, "w") as f:
        f.write("path: .\n")
    return p


class TestTrainArgsProjectName:

    def test_project_and_name_in_train_args(self, qapp, temp_dir):
        _make_data_yaml(temp_dir)
        custom_project = os.path.join(temp_dir, "my_runs")
        custom_name = "my_exp"
        widget = _make_widget_with_config(qapp, temp_dir, custom_project, custom_name)

        with patch(
            "anylabeling.views.training.guided_training_widget.create_yolo_dataset",
            return_value=temp_dir,
        ):
            train_args = widget.get_training_args(widget.get_current_config())

        assert train_args["project"] == custom_project
        assert train_args["name"] == custom_name
        widget.shutdown()

    def test_project_not_overwritten_by_default(self, qapp, temp_dir):
        _make_data_yaml(temp_dir)
        custom_project = os.path.join(temp_dir, "my_runs")
        custom_name = "gui_smoke"
        widget = _make_widget_with_config(qapp, temp_dir, custom_project, custom_name)

        with patch(
            "anylabeling.views.training.guided_training_widget.create_yolo_dataset",
            return_value=temp_dir,
        ):
            train_args = widget.get_training_args(widget.get_current_config())

        assert "xanylabeling_data" not in train_args["project"]
        assert train_args["project"] == custom_project
        widget.shutdown()


# ---------------------------------------------------------------------------
# 3. TrainingManager receives project/name via payload serialization
# ---------------------------------------------------------------------------


class TestTrainingManagerReceivesConfig:

    def test_manager_receives_project_and_name(self, temp_dir):
        from anylabeling.services.auto_training.ultralytics.trainer import (
            TrainingManager,
        )
        custom_project = os.path.join(temp_dir, "runs")
        custom_name = "test_exp"
        manager = TrainingManager()

        with patch(
            "anylabeling.services.auto_training.ultralytics.trainer.create_training_payload",
        ) as mock_payload:
            mock_payload.return_value = os.path.join(temp_dir, "payload.json")
            with patch("threading.Thread"):
                train_args = {
                    "model": "yolo11n.pt", "project": custom_project,
                    "name": custom_name, "epochs": 1,
                    "data": "data.yaml", "device": "cpu",
                }
                manager.start_training(train_args)
                mock_payload.assert_called_once()
                payload_args = mock_payload.call_args[0][0]
                assert payload_args["project"] == custom_project
                assert payload_args["name"] == custom_name


# ---------------------------------------------------------------------------
# 4. Event save_dir write-back to current_project_path
# ---------------------------------------------------------------------------


class TestSaveDirWriteBack:

    def _make_ready_widget(self, qapp, temp_dir):
        from anylabeling.views.training.guided_training_widget import (
            GuidedTrainingWidget,
        )
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir=temp_dir, supported_shape=[]
        )
        widget.names = []
        widget.current_project_path = None
        widget.training_status = "idle"
        # Mock UI attributes needed by on_training_event
        widget.status_label = MagicMock()
        widget.progress_bar = MagicMock()
        widget.log_display = MagicMock()
        widget.start_training_button = MagicMock()
        widget.stop_training_button = MagicMock()
        widget.export_button = MagicMock()
        widget.previous_button = MagicMock()
        widget.progress_timer = MagicMock()
        widget.image_timer = MagicMock()
        widget.image_labels = []
        widget.total_epochs = 1
        widget.current_epochs = 0
        return widget

    def test_completed_event_updates_project_path(self, qapp, temp_dir):
        widget = self._make_ready_widget(qapp, temp_dir)
        real_save_dir = os.path.join(temp_dir, "runs", "gui_smoke")
        os.makedirs(real_save_dir, exist_ok=True)
        widget.on_training_event("training_completed", {
            "results": "done", "save_dir": real_save_dir,
        })
        assert widget.current_project_path == real_save_dir
        widget.shutdown()

    def test_stopped_event_updates_with_save_dir(self, qapp, temp_dir):
        widget = self._make_ready_widget(qapp, temp_dir)
        widget.training_status = "training"
        stopped_dir = os.path.join(temp_dir, "runs", "stopped_exp")
        os.makedirs(stopped_dir, exist_ok=True)
        widget.on_training_event("training_stopped", {"save_dir": stopped_dir})
        assert widget.current_project_path == stopped_dir
        widget.shutdown()

    def test_no_save_dir_keeps_original(self, qapp, temp_dir):
        widget = self._make_ready_widget(qapp, temp_dir)
        original = os.path.join(temp_dir, "original")
        os.makedirs(original, exist_ok=True)
        widget.current_project_path = original
        widget.on_training_event("training_completed", {"results": "done"})
        assert widget.current_project_path == original
        widget.shutdown()

    def test_error_event_updates_with_save_dir(self, qapp, temp_dir):
        widget = self._make_ready_widget(qapp, temp_dir)
        error_dir = os.path.join(temp_dir, "runs", "error_exp")
        os.makedirs(error_dir, exist_ok=True)
        widget.on_training_event("training_error", {
            "error": "bad", "save_dir": error_dir,
        })
        assert widget.current_project_path == error_dir
        widget.shutdown()


# ---------------------------------------------------------------------------
# 5. Worker emits save_dir in completed event
# ---------------------------------------------------------------------------


class TestWorkerEmitsSaveDir:

    def test_emit_save_dir(self, temp_dir):
        from anylabeling.services.auto_training.ultralytics.trainer import (
            emit_training_worker_event,
        )
        import io
        out = io.StringIO()
        expected_dir = os.path.join(temp_dir, "runs", "exp")
        emit_training_worker_event(
            "training_completed", output_stream=out,
            results="ok", save_dir=expected_dir,
        )
        out.seek(0)
        raw = out.read()
        assert "__XANYLABELING_TRAIN_EVENT__" in raw
        payload = json.loads(raw.split("=", 1)[1])
        assert payload.get("save_dir") == expected_dir

    def test_worker_passes_project_name(self, temp_dir):
        """Verify run_training_worker_command passes project/name to model.train()."""
        payload_path = os.path.join(temp_dir, "payload.json")
        with open(payload_path, "w") as f:
            json.dump({
                "model": "dummy.pt", "data": "dummy.yaml",
                "epochs": 1, "device": "cpu",
                "project": os.path.join(temp_dir, "runs"),
                "name": "test_exp",
            }, f)

        mock_result = MagicMock()
        mock_result.save_dir = os.path.join(temp_dir, "runs", "test_exp")
        mock_model = MagicMock()
        mock_model.train.return_value = mock_result

        # The function imports YOLO inside, so patch the import target
        with patch("ultralytics.YOLO", return_value=mock_model):
            with patch("matplotlib.use"):
                import sys
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = MagicMock()
                sys.stderr = MagicMock()
                try:
                    from anylabeling.services.auto_training.ultralytics.trainer import (
                        run_training_worker_command,
                    )
                    from argparse import Namespace
                    try:
                        run_training_worker_command(Namespace(payload=payload_path))
                    except SystemExit:
                        pass
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

        mock_model.train.assert_called_once()
        call_kwargs = mock_model.train.call_args[1]
        assert call_kwargs.get("project") == os.path.join(temp_dir, "runs")
        assert call_kwargs.get("name") == "test_exp"


# ---------------------------------------------------------------------------
# 6. Default behavior preserved
# ---------------------------------------------------------------------------


class TestDefaultBehaviorPreserved:

    def test_empty_project_does_not_crash(self, qapp, temp_dir):
        _make_data_yaml(temp_dir)
        widget = _make_widget_with_config(qapp, temp_dir, "", "exp")
        with patch(
            "anylabeling.views.training.guided_training_widget.create_yolo_dataset",
            return_value=temp_dir,
        ):
            train_args = widget.get_training_args(widget.get_current_config())
        assert "project" in train_args
        assert "name" in train_args
        widget.shutdown()

    def test_absolute_path_preserved(self, qapp, temp_dir):
        _make_data_yaml(temp_dir)
        abs_project = os.path.join(temp_dir, "my", "absolute", "runs")
        widget = _make_widget_with_config(qapp, temp_dir, abs_project, "my_exp")
        with patch(
            "anylabeling.views.training.guided_training_widget.create_yolo_dataset",
            return_value=temp_dir,
        ):
            train_args = widget.get_training_args(widget.get_current_config())
        assert train_args["project"] == abs_project
        widget.shutdown()


# ---------------------------------------------------------------------------
# 7. Adapter forwards save_dir in unified events
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 8. JobManager writes save_dir to job.output_directory
# ---------------------------------------------------------------------------


class TestJobManagerOutputDir:

    def test_complete_job_sets_output_directory(self, temp_dir):
        from anylabeling.services.training_center.job_manager import (
            JobManager,
        )
        from anylabeling.services.training_center.models import (
            TrainingJob, TrainingMode, TrainingStatus,
        )
        from anylabeling.services.training_center.event_protocol import (
            TrainingEvent, TrainingEventType,
        )

        mgr = JobManager()
        job = TrainingJob(
            job_id="test-job",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING,
        )
        mgr._current_job = job
        mgr._current_adapter = MagicMock()

        save_dir = os.path.join(temp_dir, "runs", "real_exp")
        mgr.complete_job("test-job", ended_at=123456.0, metadata={
            "results": "ok", "save_dir": save_dir,
        })

        assert job.output_directory == Path(save_dir), (
            f"Expected {save_dir}, got {job.output_directory}"
        )

    def test_fail_job_sets_output_directory(self, temp_dir):
        from anylabeling.services.training_center.job_manager import (
            JobManager,
        )
        from anylabeling.services.training_center.models import (
            TrainingJob, TrainingMode, TrainingStatus,
        )

        mgr = JobManager()
        job = TrainingJob(
            job_id="test-job",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING,
        )
        mgr._current_job = job
        mgr._current_adapter = MagicMock()

        save_dir = os.path.join(temp_dir, "runs", "fail_exp")
        mgr.fail_job("test-job", error="bad", ended_at=123456.0, metadata={
            "save_dir": save_dir,
        })

        assert job.output_directory == Path(save_dir)

    def test_stop_job_sets_output_directory(self, temp_dir):
        from anylabeling.services.training_center.job_manager import (
            JobManager,
        )
        from anylabeling.services.training_center.models import (
            TrainingJob, TrainingMode, TrainingStatus,
        )

        mgr = JobManager()
        job = TrainingJob(
            job_id="test-job",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING,
        )
        mgr._current_job = job
        mgr._current_adapter = MagicMock()

        save_dir = os.path.join(temp_dir, "runs", "stop_exp")
        mgr.stop_job("test-job", ended_at=123456.0, metadata={
            "save_dir": save_dir,
        })

        assert job.output_directory == Path(save_dir)

    def test_no_save_dir_does_not_overwrite(self, temp_dir):
        from anylabeling.services.training_center.job_manager import (
            JobManager,
        )
        from anylabeling.services.training_center.models import (
            TrainingJob, TrainingMode, TrainingStatus,
        )
        from pathlib import Path

        mgr = JobManager()
        original_dir = Path(os.path.join(temp_dir, "original"))
        job = TrainingJob(
            job_id="test-job",
            mode=TrainingMode.GUIDED_ULTRALYTICS,
            status=TrainingStatus.RUNNING,
            output_directory=original_dir,
        )
        mgr._current_job = job
        mgr._current_adapter = MagicMock()

        mgr.complete_job("test-job", ended_at=123456.0, metadata={
            "results": "ok",
            # No save_dir
        })

        assert job.output_directory == original_dir


# ---------------------------------------------------------------------------
# 9. HistoryStore records output_directory
# ---------------------------------------------------------------------------


class TestHistoryStoreOutputDir:

    def test_history_finalize_stores_output_dir(self, temp_dir):
        from anylabeling.services.training_center.history import (
            HistoryStore, JobHistoryRecord,
        )
        from anylabeling.services.training_center.models import (
            TrainingStatus,
        )
        from datetime import datetime

        store = HistoryStore(
            history_dir=str(temp_dir),
        )

        record = JobHistoryRecord(
            job_id="test-job",
            mode="guided_ultralytics",
            status=TrainingStatus.RUNNING.value,
            created_at=datetime.now().isoformat(),
            display_name="test",
        )
        store.append_job(record)

        save_dir = os.path.join(temp_dir, "runs", "real_exp")
        store.finalize_job(
            "test-job",
            status=TrainingStatus.COMPLETED,
            ended_at=datetime.now(),
            output_directory=save_dir,
        )

        updated = store.get_job("test-job")
        assert updated.output_directory == save_dir, (
            f"Expected {save_dir}, got {updated.output_directory}"
        )


# ---------------------------------------------------------------------------
# 10. Project Browse button
# ---------------------------------------------------------------------------


class TestProjectBrowseButton:

    def test_browse_project_dir_exists(self, qapp, temp_dir):
        from anylabeling.views.training.guided_training_widget import (
            GuidedTrainingWidget,
        )
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir=temp_dir, supported_shape=[]
        )
        widget.names = []
        widget.ensure_config_tab_initialized()

        assert hasattr(widget, 'browse_project_dir'), (
            "browse_project_dir method missing"
        )
        assert callable(widget.browse_project_dir)

        widget.shutdown()

    def test_browse_cancel_preserves_original_value(self, qapp, temp_dir):
        from anylabeling.views.training.guided_training_widget import (
            GuidedTrainingWidget,
        )
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir=temp_dir, supported_shape=[]
        )
        widget.names = []
        widget.ensure_config_tab_initialized()

        original = os.path.join(temp_dir, "my_project")
        widget.config_widgets["project"].setText(original)

        with patch(
            "anylabeling.views.training.guided_training_widget.QFileDialog.getExistingDirectory",
            return_value="",
        ):
            widget.browse_project_dir()
            assert widget.config_widgets["project"].text() == original

        widget.shutdown()

    def test_browse_select_updates_project(self, qapp, temp_dir):
        from anylabeling.views.training.guided_training_widget import (
            GuidedTrainingWidget,
        )
        widget = GuidedTrainingWidget(
            parent=None, image_list=[], output_dir=temp_dir, supported_shape=[]
        )
        widget.names = []
        widget.ensure_config_tab_initialized()

        new_dir = os.path.join(temp_dir, "selected_project")
        os.makedirs(new_dir, exist_ok=True)

        with patch(
            "anylabeling.views.training.guided_training_widget.QFileDialog.getExistingDirectory",
            return_value=new_dir,
        ):
            widget.browse_project_dir()
            assert widget.config_widgets["project"].text() == new_dir

        widget.shutdown()


class TestAdapterForwardsSaveDir:

    def test_adapter_completed_event_has_save_dir(self):
        from anylabeling.services.training_center.adapters.ultralytics_adapter import (
            UltralyticsAdapter,
        )
        from anylabeling.services.training_center.event_protocol import (
            TrainingEventType,
        )

        adapter = UltralyticsAdapter.__new__(UltralyticsAdapter)
        adapter._callbacks = []
        adapter._original_callbacks = []
        adapter.manager = MagicMock()
        adapter.manager.callbacks = [adapter._on_training_event]
        adapter.manager.is_training = False
        adapter._current_job_id = "test-job-1"
        received = []
        adapter.subscribe(lambda e: received.append(e))
        adapter._on_training_event("training_completed", {
            "results": "ok", "save_dir": "/fake/runs/exp",
        })

        assert len(received) == 1
        assert received[0].event_type == TrainingEventType.COMPLETED
        assert received[0].payload.get("save_dir") == "/fake/runs/exp"
