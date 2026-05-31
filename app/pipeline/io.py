from __future__ import annotations
"""输入数据加载模块。

提供两类基础能力：
- 按自然顺序收集图像文件
- 把图像序列加载成三维强度立方体 (height, width, samples)
- 从 MAT 文件中提取三维强度立方体
"""

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat

SUPPORTED_SUFFIXES = {".jpg", ".tif", ".png", ".bmp"}


def _natural_key(path: Path) -> list[object]:
    """构造自然排序键，让 file2 排在 file10 前面。"""
    return [int(part) if part.isdigit() else part.lower() for part in re.findall(r"\d+|\D+", path.name)]


def collect_image_files(folder: Path) -> list[Path]:
    """扫描目录并返回受支持后缀的图像文件列表（自然排序）。"""
    files = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES]
    return sorted(files, key=_natural_key)


def load_intensity_cube(files: list[Path]) -> np.ndarray:
    """把图像序列加载为 float32 数据立方体。"""
    if not files:
        raise ValueError("At least one image is required.")

    reference = np.asarray(Image.open(files[0]).convert("L"), dtype=np.float32)
    height, width = reference.shape
    cube = np.empty((height, width, len(files)), dtype=np.float32)
    cube[:, :, 0] = reference

    # 后续帧若尺寸不一致，按首帧尺寸重采样，保证立方体维度一致。
    for index, path in enumerate(files[1:], start=1):
        image = Image.open(path).convert("L")
        if image.size != (width, height):
            image = image.resize((width, height))
        cube[:, :, index] = np.asarray(image, dtype=np.float32)

    return cube


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
