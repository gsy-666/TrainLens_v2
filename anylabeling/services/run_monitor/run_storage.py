"""Run storage for Run Monitor

Persists run metadata and events to disk.
"""

import json
from pathlib import Path
from typing import Optional

from .models import Run, TrainingEvent, Workspace


class RunStorage:
    """Handles persistence of run data"""

    TRAINLENS_DIR = ".trainlens"
    RUNS_DIR = "runs"

    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.trainlens_dir = workspace_path / self.TRAINLENS_DIR
        self.runs_dir = self.trainlens_dir / self.RUNS_DIR

    def initialize(self) -> bool:
        """
        Initialize storage directories.

        Returns:
            True if successful, False if failed
        """
        try:
            self.trainlens_dir.mkdir(exist_ok=True)
            self.runs_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"Failed to initialize storage: {e}")
            return False

    def save_workspace(self, workspace: Workspace) -> bool:
        """Save workspace metadata"""
        try:
            workspace_file = self.trainlens_dir / "workspace.json"
            with open(workspace_file, "w", encoding="utf-8") as f:
                json.dump(workspace.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save workspace: {e}")
            return False

    def save_run(self, run: Run) -> bool:
        """Save run metadata to <workspace>/.trainlens/runs/<run_id>/run.json"""
        try:
            run_dir = self.runs_dir / run.run_id
            run_dir.mkdir(exist_ok=True)

            run_file = run_dir / "run.json"
            with open(run_file, "w", encoding="utf-8") as f:
                json.dump(run.to_dict(), f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save run: {e}")
            return False

    def save_event(self, event: TrainingEvent) -> bool:
        """Append event to <workspace>/.trainlens/runs/<run_id>/events.jsonl"""
        try:
            run_dir = self.runs_dir / event.run_id
            run_dir.mkdir(exist_ok=True)

            events_file = run_dir / "events.jsonl"
            with open(events_file, "a", encoding="utf-8") as f:
                json.dump(event.to_dict(), f)
                f.write("\n")
            return True
        except Exception as e:
            print(f"Failed to save event: {e}")
            return False

    def save_console_line(self, run_id: str, line: str) -> bool:
        """Append log line to <workspace>/.trainlens/runs/<run_id>/console.log"""
        try:
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(exist_ok=True)

            console_file = run_dir / "console.log"
            with open(console_file, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
            return True
        except Exception as e:
            print(f"Failed to save console line: {e}")
            return False

    def save_resource_sample(self, run_id: str, sample: dict) -> bool:
        """Append resource sample to <workspace>/.trainlens/runs/<run_id>/resources.jsonl"""
        try:
            run_dir = self.runs_dir / run_id
            run_dir.mkdir(exist_ok=True)

            resources_file = run_dir / "resources.jsonl"
            with open(resources_file, "a", encoding="utf-8") as f:
                json.dump(sample, f)
                f.write("\n")
            return True
        except Exception as e:
            print(f"Failed to save resource sample: {e}")
            return False

    def load_run(self, run_id: str) -> Optional[Run]:
        """Load run metadata"""
        try:
            run_file = self.runs_dir / run_id / "run.json"
            if not run_file.exists():
                return None

            with open(run_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Run.from_dict(data)
        except Exception as e:
            print(f"Failed to load run: {e}")
            return None
