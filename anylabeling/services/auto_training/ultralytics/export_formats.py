"""Format definitions, capabilities, and output path mapping for model export.

This module centralises all supported export formats, their capabilities,
output path conventions, and environment status metadata.
"""

import os
import sys
import platform
from dataclasses import dataclass, field
from typing import List, Optional

from .utils import check_package_installed


# ── Format Capability ────────────────────────────────────────────────────

@dataclass
class FormatCapability:
    """Declares what an export format supports."""

    supports_half: bool = False
    supports_int8: bool = False
    supports_dynamic: bool = False
    supports_simplify: bool = False
    requires_dataset: bool = False
    requires_gpu: bool = False
    supported_platforms: List[str] = field(
        default_factory=lambda: ["linux", "windows", "darwin"]
    )


@dataclass
class FormatInfo:
    """Complete metadata for one export format."""

    format_code: str
    display_name: str
    category: str  # "Generic", "Mobile & Edge", "Hardware-Specific", "Other Frameworks"
    output_path_template: str  # relative to format output dir, e.g. "best.onnx"
    extension: str  # file extension or "" for directory
    is_directory: bool = False
    description: str = ""
    capability: FormatCapability = field(default_factory=FormatCapability)
    required_packages: List[str] = field(default_factory=list)
    pip_package_mapping: dict = field(default_factory=dict)
    large_dependencies: List[str] = field(default_factory=list)


# ── Large dependency list (must confirm before installing) ─────────────────

LARGE_DEPENDENCIES = {
    "tensorflow",
    "paddlepaddle",
    "paddlepaddle-gpu",
    "openvino",
    "tensorrt",
    "rknn-toolkit2",
    "coremltools",
}


# ── Format Registry ──────────────────────────────────────────────────────

EXPORT_FORMATS: List[FormatInfo] = [
    # ── Generic ──────────────────────────────────────────────────────────
    FormatInfo(
        format_code="onnx",
        display_name="ONNX",
        category="Generic",
        output_path_template="best.onnx",
        extension=".onnx",
        is_directory=False,
        description="Universal deployment and cross-conversion",
        capability=FormatCapability(
            supports_half=True,
            supports_int8=True,
            supports_dynamic=True,
            supports_simplify=True,
        ),
        required_packages=["onnx", "onnxslim", "onnxruntime"],
        pip_package_mapping={
            "onnx": "onnx>=1.15.0",
            "onnxslim": "onnxslim>=0.1.59",
            "onnxruntime": "onnxruntime",
        },
    ),
    FormatInfo(
        format_code="torchscript",
        display_name="TorchScript",
        category="Generic",
        output_path_template="best.torchscript",
        extension=".torchscript",
        is_directory=False,
        description="PyTorch JIT trace for C++ deployment",
        capability=FormatCapability(supports_half=True),
        required_packages=[],
    ),
    FormatInfo(
        format_code="openvino",
        display_name="OpenVINO",
        category="Generic",
        output_path_template="best_openvino_model/",
        extension="",
        is_directory=True,
        description="Intel OpenVINO IR for CPU/GPU/iGPU/VPU",
        capability=FormatCapability(
            supports_half=True, supports_int8=True, supports_dynamic=True
        ),
        required_packages=["openvino"],
        pip_package_mapping={"openvino": "openvino>=2024.0.0"},
        large_dependencies=["openvino"],
    ),
    # ── Mobile & Edge ────────────────────────────────────────────────────
    FormatInfo(
        format_code="litert",
        display_name="LiteRT",
        category="Mobile & Edge",
        output_path_template="best.litert",
        extension=".litert",
        is_directory=False,
        description="Google LiteRT for mobile/edge (replaces TFLite/TF.js)",
        capability=FormatCapability(
            supports_half=True, supports_int8=True, requires_dataset=True
        ),
        required_packages=[],
    ),
    FormatInfo(
        format_code="mnn",
        display_name="MNN",
        category="Mobile & Edge",
        output_path_template="best_mnn_model/",
        extension="",
        is_directory=True,
        description="Alibaba MNN lightweight inference",
        capability=FormatCapability(supports_half=True, supports_int8=True),
        required_packages=["MNN"],
        pip_package_mapping={"MNN": "MNN>=2.9.6"},
    ),
    FormatInfo(
        format_code="ncnn",
        display_name="NCNN",
        category="Mobile & Edge",
        output_path_template="best_ncnn_model/",
        extension="",
        is_directory=True,
        description="Tencent NCNN mobile-optimised inference",
        capability=FormatCapability(supports_half=True, supports_int8=True),
        required_packages=["ncnn"],
        pip_package_mapping={"ncnn": "ncnn"},
    ),
    # ── Hardware-Specific ────────────────────────────────────────────────
    FormatInfo(
        format_code="engine",
        display_name="TensorRT",
        category="Hardware-Specific",
        output_path_template="best.engine",
        extension=".engine",
        is_directory=False,
        description="NVIDIA TensorRT GPU inference engine",
        capability=FormatCapability(
            supports_half=True,
            supports_int8=True,
            requires_dataset=True,
            requires_gpu=True,
        ),
        required_packages=["tensorrt"],
        pip_package_mapping={"tensorrt": "tensorrt>7.0.0,!=10.1.0"},
        large_dependencies=["tensorrt"],
    ),
    FormatInfo(
        format_code="rknn",
        display_name="RKNN",
        category="Hardware-Specific",
        output_path_template="best_rknn_model/",
        extension="",
        is_directory=True,
        description="Rockchip NPU (RK3588 etc.)",
        capability=FormatCapability(
            supports_int8=True,
            requires_dataset=True,
            supported_platforms=["linux"],
        ),
        required_packages=["rknn-toolkit2"],
        pip_package_mapping={"rknn-toolkit2": "rknn-toolkit2"},
        large_dependencies=["rknn-toolkit2"],
    ),
    FormatInfo(
        format_code="imx",
        display_name="IMX500",
        category="Hardware-Specific",
        output_path_template="best_imx_model/",
        extension="",
        is_directory=True,
        description="Sony IMX500 AI sensor",
        capability=FormatCapability(supports_int8=True),
        required_packages=["imx500-converter", "mct-quantizers"],
        pip_package_mapping={
            "imx500-converter": "imx500-converter[pt]>=3.16.1",
            "mct-quantizers": "mct-quantizers>=1.6.0",
        },
        large_dependencies=["imx500-converter", "mct-quantizers"],
    ),
    FormatInfo(
        format_code="edgetpu",
        display_name="Edge TPU",
        category="Hardware-Specific",
        output_path_template="best_full_integer_quant_edgetpu.tflite",
        extension=".tflite",
        is_directory=False,
        description="Google Coral Edge TPU",
        capability=FormatCapability(
            supports_int8=True, requires_dataset=True
        ),
        required_packages=["tensorflow"],
        pip_package_mapping={"tensorflow": "tensorflow>=2.0.0"},
        large_dependencies=["tensorflow"],
    ),
    # ── Other Frameworks ─────────────────────────────────────────────────
    FormatInfo(
        format_code="saved_model",
        display_name="TensorFlow SavedModel",
        category="Other Frameworks",
        output_path_template="best_saved_model/",
        extension="",
        is_directory=True,
        description="TensorFlow SavedModel directory format",
        capability=FormatCapability(supports_int8=True),
        required_packages=["tensorflow"],
        pip_package_mapping={"tensorflow": "tensorflow>=2.0.0"},
        large_dependencies=["tensorflow"],
    ),
    FormatInfo(
        format_code="paddle",
        display_name="PaddlePaddle",
        category="Other Frameworks",
        output_path_template="best_paddle_model/",
        extension="",
        is_directory=True,
        description="Baidu PaddlePaddle inference model",
        capability=FormatCapability(),
        required_packages=["paddlepaddle", "x2paddle"],
        pip_package_mapping={
            "paddlepaddle": "paddlepaddle-gpu",
            "x2paddle": "x2paddle",
        },
        large_dependencies=["paddlepaddle", "paddlepaddle-gpu"],
    ),
    FormatInfo(
        format_code="coreml",
        display_name="CoreML",
        category="Hardware-Specific",
        output_path_template="best.mlpackage",
        extension=".mlpackage",
        is_directory=True,
        description="Apple Core ML for iOS/macOS",
        capability=FormatCapability(
            supports_half=True,
            supports_int8=True,
            supported_platforms=["darwin"],
        ),
        required_packages=["coremltools"],
        pip_package_mapping={"coremltools": "coremltools>=8.0"},
        large_dependencies=["coremltools"],
    ),
]


# ── Fast lookups ─────────────────────────────────────────────────────────

FORMAT_BY_CODE = {f.format_code: f for f in EXPORT_FORMATS}
FORMAT_BY_CATEGORY = {}
for f in EXPORT_FORMATS:
    FORMAT_BY_CATEGORY.setdefault(f.category, []).append(f)

# Categories in display order
CATEGORY_ORDER = ["Generic", "Mobile & Edge", "Hardware-Specific", "Other Frameworks"]


# ── Deprecated formats that should not appear in UI ─────────────────────

DEPRECATED_FORMATS = {"tflite", "tfjs"}
DEPRECATED_REDIRECT = {"tflite": "litert", "tfjs": "litert"}


# ── Environment status helpers ───────────────────────────────────────────

class FormatStatus:
    READY = "ready"
    MISSING_DEPENDENCY = "missing_dependency"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    UNSUPPORTED_DEVICE = "unsupported_device"
    NOT_IMPLEMENTED = "not_implemented"


def get_format_status(info: FormatInfo) -> str:
    """Determine the environment readiness for a format."""
    # Platform check
    current_platform = sys.platform
    if current_platform == "win32":
        current_platform = "windows"
    if current_platform == "linux2":
        current_platform = "linux"

    if current_platform not in info.capability.supported_platforms:
        return FormatStatus.UNSUPPORTED_PLATFORM

    # GPU check
    if info.capability.requires_gpu:
        try:
            import torch

            if not torch.cuda.is_available():
                return FormatStatus.UNSUPPORTED_DEVICE
        except ImportError:
            return FormatStatus.UNSUPPORTED_DEVICE

    # TorchScript special case: validate torch + model loadable
    if info.format_code == "torchscript":
        try:
            import torch
        except ImportError:
            return FormatStatus.MISSING_DEPENDENCY

    # Package check
    for pkg in info.required_packages:
        if not check_package_installed(pkg):
            return FormatStatus.MISSING_DEPENDENCY

    return FormatStatus.READY


def get_missing_pip_packages(info: FormatInfo) -> List[str]:
    """Get the list of missing packages in pip-installable format."""
    missing = []
    for pkg in info.required_packages:
        if not check_package_installed(pkg):
            pkg_spec = info.pip_package_mapping.get(pkg, pkg)
            missing.append(pkg_spec)
    return missing


def get_large_missing_packages(info: FormatInfo) -> List[str]:
    """Return which missing packages are 'large' (require user confirmation)."""
    missing = get_missing_pip_packages(info)
    large = []
    for m in missing:
        # extract base package name from spec
        base = m.split(">=")[0].split(">")[0].split("!=")[0].split("[")[0].strip()
        if base in LARGE_DEPENDENCIES:
            large.append(m)
    return large


def has_any_large_missing(formats: List[FormatInfo]) -> bool:
    """Check if any of the selected formats has large missing deps."""
    for info in formats:
        if get_large_missing_packages(info):
            return True
    return False
