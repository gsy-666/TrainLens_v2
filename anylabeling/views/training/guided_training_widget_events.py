"""Unified TrainingEvent handler for GuidedTrainingWidget

This module handles the mapping from unified TrainingEvents to widget UI updates.
Replaces direct TrainingManager signal connections.
"""


def handle_unified_training_event(widget, event):
    """Handle unified TrainingEvent and update widget UI

    Args:
        widget: GuidedTrainingWidget instance
        event: TrainingEvent from JobManager
    """
    from anylabeling.services.training_center.event_protocol import TrainingEventType

    # Only process events for this widget's current job
    if event.job_id != widget._current_job_id:
        return

    event_type = event.event_type

    if event_type == TrainingEventType.PREPARING:
        widget.training_status = "preparing"
        widget.update_training_status_display()
        widget.start_training_button.setEnabled(False)
        widget.append_training_log(widget.tr("Preparing training..."))

    elif event_type == TrainingEventType.PROCESS_STARTED:
        widget.training_status = "training"
        total_epochs = event.payload.get("total_epochs", 100)
        widget.total_epochs = total_epochs
        widget.current_epochs = 0
        widget.progress_bar.setValue(0)
        widget.progress_bar.setFormat(f"0/{total_epochs}")
        widget.update_training_status_display()
        widget.start_training_button.setVisible(False)
        widget.stop_training_button.setVisible(True)
        widget.stop_training_button.setEnabled(True)
        widget.export_button.setVisible(False)
        widget.previous_button.setVisible(False)
        widget.progress_timer.start(1000)
        widget.image_timer.start(5000)
        widget.append_training_log(widget.tr("Training started..."))

    elif event_type == TrainingEventType.CONSOLE_OUTPUT:
        message = event.payload.get("message", "")
        if message:
            widget.append_training_log(message)

    elif event_type == TrainingEventType.COMPLETED:
        widget.training_status = "completed"
        widget.update_training_status_display()
        widget.stop_training_button.setVisible(False)
        widget.start_training_button.setVisible(False)
        widget.previous_button.setVisible(True)
        widget.export_button.setVisible(True)
        widget.progress_timer.stop()
        widget.image_timer.stop()
        widget.update_training_progress()
        widget.update_training_images()
        widget.append_training_log(widget.tr("Training completed successfully!"))

        # Save to history
        _save_to_history(widget, "completed", None)
        widget._current_job_id = None

    elif event_type == TrainingEventType.FAILED:
        widget.training_status = "error"
        widget.update_training_status_display()
        widget.start_training_button.setVisible(False)
        widget.previous_button.setVisible(True)
        widget.stop_training_button.setVisible(False)
        widget.export_button.setVisible(False)
        widget.progress_timer.stop()
        widget.image_timer.stop()
        error_msg = event.payload.get("error", "Unknown error occurred")
        widget.append_training_log(f"ERROR: {error_msg}")

        # Save to history
        _save_to_history(widget, "failed", error_msg)
        widget._current_job_id = None

    elif event_type == TrainingEventType.STOPPED:
        widget.training_status = "stop"
        widget.update_training_status_display()
        widget.start_training_button.setVisible(False)
        widget.previous_button.setVisible(True)
        widget.stop_training_button.setVisible(False)
        widget.stop_training_button.setEnabled(True)  # Re-enable for next time
        widget.export_button.setVisible(False)
        widget.progress_timer.stop()
        widget.image_timer.stop()
        widget.append_training_log(widget.tr("Training stopped by user"))

        # Save to history
        _save_to_history(widget, "stopped", None)
        widget._current_job_id = None


def _save_to_history(widget, final_status, error_message):
    """Save training run to history store

    Args:
        widget: GuidedTrainingWidget instance
        final_status: "completed", "failed", or "stopped"
        error_message: Error message if failed, None otherwise
    """
    try:
        current_job = widget.job_manager.get_current_job()
        if not current_job:
            return

        from datetime import datetime

        record = {
            "job_id": current_job.job_id,
            "mode": "guided_ultralytics",
            "task": current_job.metadata.get("task", "detect"),
            "model": current_job.metadata.get("model", ""),
            "data": current_job.metadata.get("data", ""),
            "project": current_job.metadata.get("project", ""),
            "name": current_job.metadata.get("name", ""),
            "started_at": current_job.started_at.isoformat() if current_job.started_at else datetime.now().isoformat(),
            "ended_at": current_job.ended_at.isoformat() if current_job.ended_at else datetime.now().isoformat(),
            "status": final_status,
            "error_message": error_message,
            "output_dir": current_job.metadata.get("output_dir", ""),
            "metadata": current_job.metadata
        }

        widget.history_store.save_record(record)
    except Exception as e:
        widget.append_training_log(f"Failed to save history: {str(e)}")
