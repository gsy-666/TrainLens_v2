"""Training Metrics Dashboard — cards + tabbed charts."""

import csv
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFileDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from anylabeling.services.training_center.metrics import MetricRunData, MetricStore
from .metrics_chart_widget import MetricsChartWidget


class TrainingMetricsDashboard(QWidget):
    """Dashboard: metric cards + tabbed line charts + export buttons.

    Shared by Guided Training, Custom Project, and History detail view.
    Uses a dirty flag + 200ms refresh timer for UI update throttling.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._store = MetricStore()
        self._current_job_id: Optional[str] = None
        self._is_history_mode: bool = False

        # CSV poll timer (1s)
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_csv)
        self._poll_timer.setInterval(1000)

        # UI refresh throttle (dirty flag + 200ms timer)
        self._dirty = False
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._do_refresh)
        self._refresh_timer.setInterval(200)
        self._refresh_timer.setSingleShot(True)

        self._build_ui()
        self._show_empty()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        cards_group = QGroupBox("")
        cards_layout = QGridLayout(cards_group)
        self._epoch_card = self._make_card("Epoch", "- / -")
        self._current_metric_card = self._make_card("Current", "--")
        self._best_metric_card = self._make_card("Best", "--")
        self._loss_card = self._make_card("Loss", "--")
        cards_layout.addWidget(self._epoch_card, 0, 0)
        cards_layout.addWidget(self._current_metric_card, 0, 1)
        cards_layout.addWidget(self._best_metric_card, 0, 2)
        cards_layout.addWidget(self._loss_card, 0, 3)
        layout.addWidget(cards_group)

        self._tabs = QTabWidget()
        self._loss_chart = MetricsChartWidget()
        self._quality_chart = MetricsChartWidget()
        self._lr_chart = MetricsChartWidget()
        self._other_chart = MetricsChartWidget()
        self._tabs.addTab(self._loss_chart, "Loss")
        self._tabs.addTab(self._quality_chart, "Accuracy && mAP")
        self._tabs.addTab(self._lr_chart, "Learning Rate")
        self._tabs.addTab(self._other_chart, "Other")
        layout.addWidget(self._tabs, stretch=1)

        btn_row = QHBoxLayout()
        self._export_csv_btn = QPushButton("Export CSV")
        self._export_csv_btn.clicked.connect(self._export_csv)
        self._export_csv_btn.setEnabled(False)
        btn_row.addWidget(self._export_csv_btn)
        self._export_img_btn = QPushButton("Save Chart Image")
        self._export_img_btn.clicked.connect(self._export_image)
        self._export_img_btn.setEnabled(False)
        btn_row.addWidget(self._export_img_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Empty state label (shown when no data)
        self._empty_label = QLabel(
            "No structured metrics detected.\n\n"
            "Supported sources:\n"
            "  TrainLens EPOCH_METRICS events\n"
            "  metrics.jsonl\n"
            "  Ultralytics results.csv"
        )
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: gray; font-size: 10pt;")
        self._empty_label.setWordWrap(True)
        self._empty_label.hide()
        layout.addWidget(self._empty_label)

    def _make_card(self, title: str, value: str) -> QGroupBox:
        gb = QGroupBox(title)
        ly = QVBoxLayout(gb)
        label = QLabel(value)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        label.setObjectName(f"card_{title.lower().replace(' ', '_').replace('&&','').strip()}_value")
        ly.addWidget(label)
        return gb

    # ── public API ────────────────────────────────────────────────────

    def bind_job(self, job_id: str, output_dir: Optional[str] = None):
        if self._current_job_id and self._current_job_id != job_id:
            self.clear()
        self._current_job_id = job_id
        self._is_history_mode = False
        self._store.start_run(job_id, output_dir)
        if output_dir and not self._is_history_mode:
            self._poll_timer.start()
        self._schedule_refresh()

    def on_metric_event(self, job_id: str, payload: dict):
        if job_id != self._current_job_id:
            return
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            return
        values = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                values[str(k)] = float(v)
        if not values:
            return
        from anylabeling.services.training_center.metrics.models import MetricSample
        sample = MetricSample(
            job_id=job_id,
            epoch=payload.get("epoch"),
            step=payload.get("step"),
            total_epochs=payload.get("total_epochs"),
            values=values,
            timestamp=payload.get("timestamp", time.time()),
        )
        self._store.add_sample(sample)
        self._schedule_refresh()

    def on_run_completed(self, job_id: str):
        """Training completed — do final CSV read BEFORE stopping polling."""
        if job_id != self._current_job_id:
            return
        # 1. Final poll while still active
        data = self._store.poll_csv()
        if not data:
            # Fallback: load directly from stored output_dir
            run = self._store.get_run(job_id)
            if run and run.output_dir:
                data = self._store.load_from_output_dir(job_id, run.output_dir)
        # 2. Stop live polling (data preserved)
        self._poll_timer.stop()
        # 3. Mark finished (clears active_job_id but keeps data + csv_path)
        self._store.finish_run(job_id)
        # 4. Refresh UI
        self._schedule_refresh()

    def on_run_stopped(self, job_id: str):
        """Training stopped — same as completed: final read, stop poll, keep data."""
        if job_id != self._current_job_id:
            return
        # Same flow as completed
        data = self._store.poll_csv()
        if not data:
            run = self._store.get_run(job_id)
            if run and run.output_dir:
                data = self._store.load_from_output_dir(job_id, run.output_dir)
        self._poll_timer.stop()
        self._store.finish_run(job_id)
        self._schedule_refresh()

    def update_output_dir(self, job_id: str, output_dir: str):
        """Update the output directory after training starts (real save_dir callback).

        Called when the real ultralytics save_dir becomes known (differs from
        the predicted project/name path). Re-binds the store's CSV path.
        """
        if job_id != self._current_job_id:
            return
        self._store.update_csv_path(job_id, output_dir)
        # Force a poll immediately
        self._poll_csv()
        self._schedule_refresh()

    def load_history(self, job_id: str, output_dir: str):
        self.clear()
        self._current_job_id = job_id
        self._is_history_mode = True
        self._poll_timer.stop()
        data = self._store.load_from_output_dir(job_id, output_dir)
        if data:
            self._schedule_refresh()
        else:
            self._show_empty()

    def clear(self):
        self._current_job_id = None
        self._is_history_mode = False
        self._poll_timer.stop()
        self._refresh_timer.stop()
        self._dirty = False
        self._loss_chart.clear()
        self._quality_chart.clear()
        self._lr_chart.clear()
        self._other_chart.clear()
        self._show_empty()

    def cleanup(self):
        self._poll_timer.stop()
        self._refresh_timer.stop()

    # ── internal ──────────────────────────────────────────────────────

    def _schedule_refresh(self):
        """Set dirty flag, schedule deferred UI update (max ~5 FPS)."""
        self._dirty = True
        if not self._refresh_timer.isActive():
            self._refresh_timer.start()

    def _do_refresh(self):
        if not self._dirty:
            return
        self._dirty = False
        self._refresh()

    def _poll_csv(self):
        data = self._store.poll_csv()
        if data:
            self._schedule_refresh()

    def _refresh(self):
        run = self._store.get_run(self._current_job_id) if self._current_job_id else None
        if not run or not run.samples:
            self._show_empty()
            return

        self._empty_label.hide()
        for tab_idx in range(self._tabs.count()):
            self._tabs.widget(tab_idx).show()

        series_list = run.to_series()
        by_group: dict = {"loss": [], "quality": [], "learning_rate": [], "other": []}
        for s in series_list:
            by_group.setdefault(s.group, []).append((s.name, s.display_name, s.points))

        self._loss_chart.set_series(by_group.get("loss", []))
        self._quality_chart.set_series(by_group.get("quality", []))
        self._lr_chart.set_series(by_group.get("learning_rate", []))
        other = by_group.get("other", [])
        if other:
            self._tabs.setTabVisible(3, True)
            self._other_chart.set_series(other)
        else:
            self._tabs.setTabVisible(3, False)

        self._update_cards(run)
        self._export_csv_btn.setEnabled(True)
        self._export_img_btn.setEnabled(True)

    def _update_cards(self, run: MetricRunData):
        samples = run.samples
        if samples:
            last = samples[-1]
            ep = f"{int(last.epoch or 0)} / {run.total_epochs or '?'}"
            self._set_card("epoch", ep)

            cur_val = "--"
            best_val = "--"
            best_ep = "--"
            qual_kvs = {}
            for s in samples:
                for k, v in s.values.items():
                    kl = k.lower()
                    if any(w in kl for w in ("map", "accuracy", "precision")):
                        qual_kvs[(s.epoch or 0, k)] = v
            if qual_kvs:
                last_items = sorted(qual_kvs.keys(), key=lambda x: x[0])
                cur_val = f"{qual_kvs[last_items[-1]]:.4f}"
                best_key = max(qual_kvs, key=lambda x: qual_kvs[x])
                best_val = f"{qual_kvs[best_key]:.4f}"
                best_ep = f"Epoch {int(best_key[0])}"
            self._set_card("current", cur_val)
            self._set_card("best", f"{best_val} ({best_ep})")

            loss_vals = []
            for s in samples:
                for k, v in s.values.items():
                    if "loss" in k.lower():
                        loss_vals.append(v)
            self._set_card("loss", f"{loss_vals[-1]:.4f}" if loss_vals else "--")

    def _set_card(self, name: str, value: str):
        card = self.findChild(QLabel, f"card_{name}_value")
        if card:
            card.setText(value)

    def _show_empty(self):
        for chart in (self._loss_chart, self._quality_chart, self._lr_chart, self._other_chart):
            chart.clear()
        for name in ("epoch", "current", "best", "loss"):
            self._set_card(name, "--")
        self._export_csv_btn.setEnabled(False)
        self._export_img_btn.setEnabled(False)
        self._empty_label.show()
        for tab_idx in range(self._tabs.count()):
            self._tabs.widget(tab_idx).hide()

    def _export_csv(self):
        run = self._store.get_run(self._current_job_id) if self._current_job_id else None
        if not run or not run.samples:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                all_keys = set()
                for s in run.samples:
                    all_keys.update(s.values.keys())
                keys = sorted(all_keys)
                w = csv.writer(f)
                w.writerow(["epoch", "step"] + keys)
                for s in run.samples:
                    row = [s.epoch, s.step] + [s.values.get(k, "") for k in keys]
                    w.writerow(row)
        except OSError as e:
            QMessageBox.critical(self, "Error", f"Failed to export CSV: {e}")

    def _export_image(self):
        chart = self._tabs.currentWidget()
        if not isinstance(chart, MetricsChartWidget):
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Chart Image", "", "PNG (*.png)")
        if not path:
            return
        pixmap = chart.grab()
        if not pixmap.save(path, "PNG"):
            QMessageBox.critical(self, "Error", "Failed to save chart image")
