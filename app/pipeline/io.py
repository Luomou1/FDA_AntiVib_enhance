from __future__ import annotations
"""输入数据加载模块。

提供两类基础能力：
- 按自然顺序收集图像文件
- 把图像序列加载成三维强度立方体 (height, width, samples)
- 从 MAT 文件中提取三维强度立方体
"""

import re
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy.io import loadmat

SUPPORTED_SUFFIXES = {".jpg", ".tif", ".tiff", ".png", ".bmp"}
ImageIntensityMode = Literal["legacy_8bit", "mono12_uint16"]


def _natural_key(path: Path) -> list[object]:
    """构造自然排序键，让 file2 排在 file10 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.findall(r"\d+|\D+", path.name)]


def collect_image_files(folder: Path) -> list[Path]:
    """扫描目录并返回受支持后缀的图像文件列表（自然排序）。"""
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files, key=_natural_key)


def load_intensity_cube(files: list[Path], image_intensity_mode: ImageIntensityMode = "legacy_8bit") -> np.ndarray:
    """把图像序列加载为 float32 数据立方体。"""
    if not files:
        raise ValueError("At least one image is required.")

    reference = _load_frame(files[0], image_intensity_mode)
    height, width = reference.shape
    cube = np.empty((height, width, len(files)), dtype=np.float32)
    cube[:, :, 0] = reference

    # 后续帧若尺寸不一致，按首帧尺寸重采样，保证立方体维度一致。
    for index, path in enumerate(files[1:], start=1):
        frame = _load_frame(path, image_intensity_mode)
        if frame.shape != (height, width):
            if image_intensity_mode == "mono12_uint16":
                raise ValueError(f"Mono12 image size mismatch: {path.name}.")
            image = Image.fromarray(frame.astype(np.uint8, copy=False)).resize((width, height))
            frame = np.asarray(image, dtype=np.float32)
        cube[:, :, index] = frame

    return cube


def _load_frame(path: Path, image_intensity_mode: ImageIntensityMode) -> np.ndarray:
    """按图像强度模式读取单帧，并统一返回二维 float32 强度图。"""
    if image_intensity_mode == "legacy_8bit":
        return np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    if image_intensity_mode == "mono12_uint16":
        return _load_mono12_uint16_frame(path)
    raise ValueError(f"Unsupported image intensity mode: {image_intensity_mode}.")


def _load_mono12_uint16_frame(path: Path) -> np.ndarray:
    """读取左对齐保存的 Mono12-in-uint16 单通道图，并还原到 0..4095。"""
    if path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        raise ValueError(f"Mono12 mode only supports PNG/TIFF images: {path.name}.")

    array = np.asarray(Image.open(path))
    if array.ndim != 2:
        raise ValueError(f"Mono12 image must be single-channel: {path.name}.")
    if array.dtype != np.uint16:
        raise ValueError(f"Mono12 image must be stored as uint16: {path.name}.")
    restored = np.right_shift(array, 4)
    return restored.astype(np.float32, copy=False)


def _coerce_mat_cube(candidate: object, name: str) -> np.ndarray:
    """把 MAT 变量规范成可直接分析的三维强度立方体。"""
    array = np.asarray(candidate)
    if array.ndim != 3:
        raise ValueError(f"MAT variable '{name}' is not a 3D array.")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"MAT variable '{name}' must contain numeric values.")
    return array.astype(np.float32, copy=False)


def load_mat_intensity_cube(path: Path) -> np.ndarray:
    """从 MAT 文件提取强度立方体，优先读取 `inten_mtr`。"""
    if not path.exists():
        raise FileNotFoundError(f"MAT file not found: {path}")

    try:
        payload = loadmat(path)
    except NotImplementedError as exc:
        raise ValueError("Unsupported MAT format; please convert the file to MATLAB v7.2 or earlier.") from exc

    if "inten_mtr" in payload:
        return _coerce_mat_cube(payload["inten_mtr"], "inten_mtr")

    # 优先复用与示例文件一致的变量名；若用户的 MAT 命名不同，再退回到第一个三维数值矩阵。
    for key, value in payload.items():
        if key.startswith("__"):
            continue
        try:
            return _coerce_mat_cube(value, key)
        except ValueError:
            continue

    raise ValueError("No 3D numeric matrix found in MAT file.")
