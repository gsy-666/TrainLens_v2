import os

# Temporary fix for: bus error
# Source: https://stackoverflow.com/questions/73072612/
# why-does-np-linalg-solve-raise-bus-error-when-running-on-its-own-thread-mac-m1
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Suppress ICC profile warnings
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.gui.icc=false"

import argparse
import codecs
import logging
import multiprocessing

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

# ── Early-intercept: training-worker mode (must not import Qt) ──
if "--training-worker" in sys.argv:
    # Parse just enough to get the payload path
    _worker_parser = argparse.ArgumentParser(add_help=False)
    _worker_parser.add_argument("--training-worker", action="store_true")
    _worker_parser.add_argument("--payload", type=str, required=True)
    _worker_args, _ = _worker_parser.parse_known_args()
    if _worker_args.training_worker and _worker_args.payload:
        from anylabeling.services.auto_training.ultralytics.training_worker import main as worker_main
        # Override sys.argv so the worker's own argparse works
        sys.argv = [sys.argv[0], "--payload", _worker_args.payload]
        worker_main()
        sys.exit(0)

# ── Early-intercept: packaging self-check (no GUI) ──
if "--packaging-self-check" in sys.argv:
    # Prevent anylabeling.__init__ from importing Qt-dependent view modules
    os.environ["ANYLABELING_SKIP_VIEW_IMPORTS"] = "1"

    import json as _json
    import traceback as _tb
    _result = {"status": "starting", "frozen": bool(getattr(sys, "frozen", False))}
    _exit_code = 0

    def _sc_write():
        try:
            _ud = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "TrainLens"
            _ud.mkdir(parents=True, exist_ok=True)
            (_ud / "selfcheck_result.json").write_text(_json.dumps(_result, indent=2))
        except Exception:
            pass

    _sc_write()
    try:
        # Executable path
        _result["executable"] = sys.executable
        _sc_write()
        # QtCore — real import test. PyInstaller's PyQt6 runtime hook
        # (which sets QT_PLUGIN_PATH, PATH, etc.) must run first.
        try:
            if getattr(sys, "frozen", False):
                try:
                    from pyi_rth_pyqt6 import _pyi_rthook
                    _pyi_rthook()
                except Exception:
                    pass
            from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR
            _result["qtcore_import"] = True
            _result["qt_version"] = QT_VERSION_STR
            _result["pyqt_version"] = PYQT_VERSION_STR
        except Exception as e:
            _result["qtcore_import"] = False
            _result["qtcore_error"] = repr(e)
            _result["status"] = "error"
            _exit_code = 1
        _sc_write()
        # Torch
        try:
            import torch
            _result["torch"] = torch.__version__
            _result["torch_cuda"] = torch.cuda.is_available()
        except Exception as e:
            _result["torch"] = str(e)
        _sc_write()
        # Ultralytics
        try:
            import ultralytics
            _result["ultralytics"] = ultralytics.__version__
        except Exception as e:
            _result["ultralytics"] = str(e)
        _sc_write()
        # Paramiko
        try:
            import paramiko
            _result["paramiko"] = getattr(paramiko, "__version__", "ok")
        except Exception as e:
            _result["paramiko"] = str(e)
        _sc_write()
        # OpenCV
        try:
            import cv2
            _result["opencv"] = cv2.__version__
        except Exception as e:
            _result["opencv"] = str(e)
        _sc_write()
        # Worker resource: training_worker.py is bundled as a data file (raw .py)
        try:
            _base = Path(getattr(sys, "_MEIPASS", str(Path(__file__).resolve().parent.parent)))
            _wp = _base / "anylabeling" / "services" / "auto_training" / "ultralytics" / "training_worker.py"
            _result["worker_resource"] = _wp.exists()
        except Exception as e:
            _result["worker_resource"] = str(e)
        _sc_write()
        # User data: %LOCALAPPDATA%\TrainLens must be writable
        try:
            _ud = Path(os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))) / "TrainLens"
            _ud.mkdir(parents=True, exist_ok=True)
            (_ud_test := _ud / ".write_test").write_text("ok")
            _ud_test.unlink()
            _result["userdata_writable"] = True
        except Exception as e:
            _result["userdata_writable"] = str(e)
        _sc_write()
        if _result.get("status") != "error":
            _result["status"] = "ok"
    except Exception:
        _result["status"] = "error"
        _result["error"] = _tb.format_exc()
        _exit_code = 1
    _sc_write()
    sys.exit(_exit_code)

import yaml
from PyQt6 import QtCore, QtGui, QtWidgets

from anylabeling.app_info import (
    __appname__,
    __version__,
    __url__,
    CLI_HELP_MSG,
)
from anylabeling.config import (
    get_config,
    set_work_directory,
    get_work_directory,
)
from anylabeling import config as anylabeling_config


def is_wsl_environment():
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True

    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def get_default_qt_platform():
    if os.environ.get("QT_QPA_PLATFORM"):
        return None

    if is_wsl_environment() and os.environ.get("WAYLAND_DISPLAY"):
        return "xcb"

    return None


def main():
    multiprocessing.freeze_support()

    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command", help="available commands"
    )
    subparsers.add_parser("help", help="show help message")
    subparsers.add_parser(
        "checks", help="display system and package information"
    )
    subparsers.add_parser("version", help="show version information")
    subparsers.add_parser("config", help="show config file path")
    train_worker_parser = subparsers.add_parser(
        "train-worker", help=argparse.SUPPRESS
    )
    train_worker_parser.add_argument(
        "--payload", required=True, help=argparse.SUPPRESS
    )

    convert_parser = subparsers.add_parser(
        "convert", help="run conversion tasks"
    )
    convert_parser.add_argument(
        "--task",
        type=str,
        help="conversion task name (e.g., yolo2xlabel, xlabel2yolo)",
    )
    convert_parser.add_argument(
        "--images", type=str, help="image directory path"
    )
    convert_parser.add_argument(
        "--labels", type=str, help="label directory path"
    )
    convert_parser.add_argument(
        "--output", type=str, help="output directory path"
    )
    convert_parser.add_argument(
        "--classes", type=str, help="classes file path"
    )
    convert_parser.add_argument(
        "--pose-cfg", type=str, help="pose configuration file path"
    )
    convert_parser.add_argument("--mode", type=str, help="conversion mode")
    convert_parser.add_argument(
        "--mapping", type=str, help="mapping table file path"
    )
    convert_parser.add_argument(
        "--skip-empty-files",
        action="store_true",
        help="skip creating empty output files, only support `xlabel2yolo` and `xlabel2voc` tasks",
    )

    parser.add_argument(
        "--reset-config", action="store_true", help="reset qt config"
    )
    parser.add_argument(
        "--logger-level",
        default="info",
        choices=["debug", "info", "warning", "fatal", "error"],
        help="logger level",
    )
    parser.add_argument(
        "--no-auto-update-check",
        action="store_true",
        help="disable automatic update check on startup",
    )
    parser.add_argument(
        "--qt-platform",
        help=(
            "Force Qt platform plugin (e.g., 'xcb', 'wayland'). "
            "If not specified, Qt will auto-detect the platform."
        ),
        default=None,
    )
    parser.add_argument(
        "--qt-image-allocation-limit",
        type=int,
        help=(
            "Override Qt image allocation limit in MB. "
            "Qt default is 256 MB. Use 0 to disable the limit."
        ),
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--filename",
        nargs="?",
        help=(
            "image or label filename; "
            "If a directory path is passed in, the folder will be loaded automatically"
        ),
    )
    parser.add_argument(
        "--output",
        "-O",
        "-o",
        help=(
            "output file or directory (if it ends with .json it is "
            "recognized as file, else as directory)"
        ),
    )
    parser.add_argument(
        "--config",
        dest="config",
        help="config file or yaml-format string",
        default=None,
    )
    # config for the gui
    parser.add_argument(
        "--nodata",
        dest="store_data",
        action="store_false",
        help="stop storing image data to JSON file",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--autosave",
        dest="auto_save",
        action="store_true",
        help="auto save",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--nosortlabels",
        dest="sort_labels",
        action="store_false",
        help="stop sorting labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--flags",
        help="comma separated list of flags OR file containing flags",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labelflags",
        dest="label_flags",
        help=r"yaml string of label specific flags OR file containing json "
        r"string of label specific flags (ex. {person-\d+: [male, tall], "
        r"dog-\d+: [black, brown, white], .*: [occluded]})",  # NOQA
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--labels",
        help="comma separated list of labels OR file containing labels",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--validatelabel",
        dest="validate_label",
        choices=["exact"],
        help="label validation types",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--keep-prev",
        action="store_true",
        help="keep annotation of previous frame",
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        help="working directory for configuration and data files",
        default=os.path.expanduser("~"),
    )
    args = parser.parse_args()

    set_work_directory(args.work_dir)

    special = {
        "help": lambda args: print(CLI_HELP_MSG),
        "checks": lambda args: __import__(
            "anylabeling.views.common.checks", fromlist=["run_checks"]
        ).run_checks(),
        "version": lambda args: print(__version__),
        "config": lambda args: print(
            os.path.join(get_work_directory(), ".xanylabelingrc")
        ),
        "convert": lambda args: __import__(
            "anylabeling.views.common.converter",
            fromlist=["handle_convert_command"],
        ).handle_convert_command(args),
        "train-worker": lambda args: __import__(
            "anylabeling.services.auto_training.ultralytics.trainer",
            fromlist=["run_training_worker_command"],
        ).run_training_worker_command(args),
    }

    if args.command and args.command in special:
        special[args.command](args)
        return

    from anylabeling.views.mainwindow import MainWindow
    from anylabeling.views.labeling.logger import logger
    from anylabeling.views.labeling.utils import new_icon, gradient_text
    from anylabeling.views.labeling.utils.theme import (
        init_theme,
        get_app_stylesheet,
        get_dark_palette,
    )
    from anylabeling.views.labeling.utils.update_checker import (
        check_for_updates_async,
    )

    # NOTE: Do not remove this import, it is required for loading translations
    from anylabeling.resources import resources  # noqa: F401

    if hasattr(args, "flags"):
        if os.path.isfile(args.flags):
            with codecs.open(args.flags, "r", encoding="utf-8") as f:
                args.flags = [line.strip() for line in f if line.strip()]
        else:
            args.flags = [line for line in args.flags.split(",") if line]

    if hasattr(args, "labels"):
        if os.path.isfile(args.labels):
            with codecs.open(args.labels, "r", encoding="utf-8") as f:
                args.labels = [line.strip() for line in f if line.strip()]
        else:
            args.labels = [line for line in args.labels.split(",") if line]

    if hasattr(args, "label_flags"):
        if os.path.isfile(args.label_flags):
            with codecs.open(args.label_flags, "r", encoding="utf-8") as f:
                args.label_flags = yaml.safe_load(f)
        else:
            args.label_flags = yaml.safe_load(args.label_flags)

    config_from_args = args.__dict__
    config_from_args.pop("command", None)
    config_from_args.pop("work_dir")
    reset_config = config_from_args.pop("reset_config")
    filename = config_from_args.pop("filename")
    output = config_from_args.pop("output")
    config_file_or_yaml = config_from_args.pop("config")
    if config_file_or_yaml is None:
        config_file_or_yaml = os.path.join(
            get_work_directory(), ".xanylabelingrc"
        )
    logger_level = config_from_args.pop("logger_level")
    no_auto_update_check = config_from_args.pop("no_auto_update_check", False)
    qt_platform = config_from_args.pop("qt_platform", None)

    logger.setLevel(getattr(logging, logger_level.upper()))

    # ── File logging (survives even without console) ──
    try:
        from anylabeling.services.training_center.build_info import (
            get_log_dir, ensure_user_dirs, get_build_info,
        )
        ensure_user_dirs()
        _log_dir = get_log_dir()
        _file_handler = logging.FileHandler(
            str(_log_dir / "trainlens.log"),
            encoding="utf-8", delay=True,
        )
        _file_handler.setLevel(logging.DEBUG)
        _file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logging.getLogger().addHandler(_file_handler)
        _bi = get_build_info()
        logging.getLogger("anylabeling").info(
            "TrainLens %s (frozen=%s) | Python %s | %s",
            _bi["trainlens_version"], _bi["frozen"],
            _bi["python_version"], _bi["platform"],
        )
        logging.getLogger("anylabeling").info(
            "User data: %s", _bi["user_data_dir"],
        )
        # Also log uncaught exceptions to file
        def _log_uncaught(exc_type, exc_value, exc_tb):
            import traceback
            logging.getLogger("anylabeling").critical(
                "Uncaught exception:\n%s",
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb)),
            )
            sys.__excepthook__(exc_type, exc_value, exc_tb)
        sys.excepthook = _log_uncaught
    except Exception:
        pass  # File logging is best-effort; never prevent GUI launch

    logger.info(
        f"🚀 {gradient_text(f'X-AnyLabeling v{__version__} launched!')}"
    )
    logger.info(f"⭐ If you like it, give us a star: {__url__}")
    if qt_platform:
        os.environ["QT_QPA_PLATFORM"] = qt_platform
        logger.info(f"🖥️ Using Qt platform: {qt_platform}")
    else:
        default_qt_platform = get_default_qt_platform()
        if default_qt_platform:
            os.environ["QT_QPA_PLATFORM"] = default_qt_platform
            logger.info(
                "🖥️ Detected WSL/Wayland; using Qt platform: "
                f"{default_qt_platform}"
            )

    anylabeling_config.current_config_file = config_file_or_yaml
    config = get_config(config_file_or_yaml, config_from_args, show_msg=True)

    if not config["labels"] and config["validate_label"]:
        logger.error(
            "--labels must be specified with --validatelabel or "
            "validate_label: exact in the config file "
            "(ex. ~/.xanylabelingrc)."
        )
        sys.exit(1)

    output_file = None
    output_dir = None
    if output is not None:
        if output.endswith(".json"):
            output_file = output
        else:
            output_dir = output

    language = config.get("language", QtCore.QLocale.system().name())
    translator = QtCore.QTranslator()
    loaded_language = translator.load(
        ":/languages/translations/" + language + ".qm"
    )
    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts
    )
    qt_image_allocation_limit = config.get("qt_image_allocation_limit")
    if qt_image_allocation_limit is not None:
        QtGui.QImageReader.setAllocationLimit(qt_image_allocation_limit)
        if qt_image_allocation_limit == 0:
            logger.info("🖼️ Disabled Qt image allocation limit")
        else:
            logger.info(
                "🖼️ Set Qt image allocation limit to "
                f"{qt_image_allocation_limit} MB"
            )

    app = QtWidgets.QApplication(sys.argv)
    init_theme(config.get("theme", "light"))
    _dark_palette = get_dark_palette()
    if _dark_palette is not None:
        app.setStyle("Fusion")
        app.setPalette(_dark_palette)
    app.setStyleSheet(get_app_stylesheet())
    app.processEvents()

    app.setApplicationName(__appname__)
    app.setApplicationVersion(__version__)
    app.setWindowIcon(new_icon("icon"))
    if loaded_language:
        app.installTranslator(translator)
    else:
        logger.warning(
            f"Failed to load translation for {language}. "
            "Using default language.",
        )
    if reset_config:
        settings = QtCore.QSettings("anylabeling", "anylabeling")
        logger.info(f"Resetting Qt config: {settings.fileName()}")
        settings.clear()
        settings.sync()
        return

    win = MainWindow(
        app,
        config=config,
        filename=filename,
        output_file=output_file,
        output_dir=output_dir,
    )

    if not no_auto_update_check:

        def delayed_update_check():
            check_for_updates_async(timeout=5)

        QtCore.QTimer.singleShot(2000, delayed_update_check)

    win.showMaximized()
    win.raise_()
    sys.exit(app.exec())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()
