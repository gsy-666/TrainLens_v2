"""Script detection for Run Monitor

Analyzes Python files to determine if they are training scripts.
"""

import re
from pathlib import Path
from typing import Optional

from .models import DetectedScript


class ScriptDetector:
    """Detects training scripts through static analysis"""

    # Filename patterns that suggest training scripts
    TRAINING_FILENAMES = {
        "train.py": 0.4,
        "main.py": 0.3,
        "run.py": 0.3,
        "fit.py": 0.4,
        "trainer.py": 0.4,
        "run_train.py": 0.4,
        "train_model.py": 0.4,
    }

    # Framework import patterns
    FRAMEWORK_IMPORTS = {
        "ultralytics": [
            r"from\s+ultralytics\s+import\s+YOLO",
            r"import\s+ultralytics",
        ],
        "pytorch": [
            r"import\s+torch",
            r"from\s+torch\s+import",
            r"from\s+torch\.utils\.data\s+import\s+DataLoader",
            r"from\s+torch\.optim\s+import",
        ],
        "lightning": [
            r"import\s+pytorch_lightning",
            r"from\s+pytorch_lightning\s+import\s+Trainer",
            r"import\s+lightning",
        ],
        "tensorflow": [
            r"import\s+tensorflow",
            r"from\s+tensorflow\s+import",
            r"import\s+keras",
        ],
    }

    # Training code patterns
    TRAINING_PATTERNS = [
        r"\.train\(",
        r"Trainer\.fit\(",
        r"model\.fit\(",
        r"for\s+epoch\s+in\s+range\(",
        r"optimizer\.step\(",
        r"loss\.backward\(",
    ]

    # Entry point patterns
    ENTRY_POINT_PATTERN = r'if\s+__name__\s*==\s*["\']__main__["\']'

    # CLI framework patterns
    CLI_PATTERNS = [
        r"import\s+argparse",
        r"import\s+click",
        r"import\s+typer",
    ]

    def detect(self, script_path: Path) -> Optional[DetectedScript]:
        """
        Analyze a Python file to determine if it's a training script.

        Args:
            script_path: Path to Python file

        Returns:
            DetectedScript if identified as training script, None otherwise
        """
        if not script_path.exists() or not script_path.is_file():
            return None

        if script_path.suffix != ".py":
            return None

        try:
            content = script_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return None

        confidence = 0.0
        reasons = []
        detected_framework = None

        # Check filename
        filename = script_path.name.lower()
        filename_match = False
        for train_filename in self.TRAINING_FILENAMES:
            if filename == train_filename.lower():
                filename_score = self.TRAINING_FILENAMES[train_filename]
                confidence += filename_score
                reasons.append("filename_match")
                filename_match = True
                break

        # Check framework imports
        for framework, patterns in self.FRAMEWORK_IMPORTS.items():
            for pattern in patterns:
                if re.search(pattern, content):
                    confidence += 0.4
                    reasons.append(f"imports_{framework}")
                    if detected_framework is None:
                        detected_framework = framework
                    break

        # Check training patterns
        training_pattern_found = False
        for pattern in self.TRAINING_PATTERNS:
            if re.search(pattern, content):
                if not training_pattern_found:
                    confidence += 0.3
                    reasons.append("contains_training_code")
                    training_pattern_found = True
                break

        # Check entry point
        if re.search(self.ENTRY_POINT_PATTERN, content):
            confidence += 0.1
            reasons.append("has_main_entry_point")

        # Check CLI frameworks
        for pattern in self.CLI_PATTERNS:
            if re.search(pattern, content):
                confidence += 0.05
                reasons.append("has_cli_framework")
                break

        # Threshold for detection with ML framework signals
        if confidence >= 0.5:
            return DetectedScript(
                path=script_path,
                framework=detected_framework or "unknown",
                confidence=min(confidence, 1.0),
                reasons=reasons,
            )

        # Fallback: common entry filenames are always runnable even without framework signals
        if filename_match:
            return DetectedScript(
                path=script_path,
                framework="generic_python",
                confidence=min(confidence, 1.0),
                reasons=reasons + ["common_entry_filename_fallback"],
            )

        return None
