from .app_info import __appdescription__, __appname__, __version__

# Defer view imports to avoid loading UI code during service/test imports
import os
if os.environ.get('ANYLABELING_SKIP_VIEW_IMPORTS') != '1':
    try:
        from anylabeling.views.common.checks import run_checks as checks
        from anylabeling.views.common.converter import (
            SUPPORTED_TASKS,
            run_conversion as convert,
            list_supported_tasks,
        )
    except ImportError:
        # Allow package to be imported even if view dependencies are missing
        checks = None
        convert = None
        list_supported_tasks = None
        SUPPORTED_TASKS = None
else:
    checks = None
    convert = None
    list_supported_tasks = None
    SUPPORTED_TASKS = None

__all__ = (
    "__version__",
    "__appname__",
    "__appdescription__",
    "checks",
    "convert",
    "list_supported_tasks",
    "SUPPORTED_TASKS",
)
