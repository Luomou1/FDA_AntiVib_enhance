from __future__ import annotations
"""scan_log 解析模块。

负责从设备导出的 scan_log 文本中提取 actual displacement 序列，
并进行编码兼容和严格单调性校验。
"""

import re
from pathlib import Path

import numpy as np

_DATA_LINE_PATTERN = re.compile(r"^\s*\d+\s*,")
_SUPPORTED_ENCODINGS = ("utf-8-sig", "gb18030", "utf-16", "utf-16-le", "utf-16-be")


def _read_scan_log_text(path: Path) -> str:
    """按预设编码列表逐个尝试读取 scan_log 文本。"""
    raw = path.read_bytes()
    for encoding in _SUPPORTED_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode scan log: {path}")


def load_actual_positions_um(path: Path) -> np.ndarray:
    """解析 scan_log 并返回实际位移数组（单位 um）。"""
    text = _read_scan_log_text(path)
    positions: list[float] = []

    # 只处理数据行；标题、说明、空行会被自动跳过。
    for line in text.splitlines():
        if not _DATA_LINE_PATTERN.match(line):
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            raise ValueError("Malformed scan log row; expected index, target displacement, actual displacement.")
        positions.append(float(parts[2]))

    if not positions:
        raise ValueError("Scan log does not contain any actual displacement rows.")

    values = np.asarray(positions, dtype=np.float32)
    # 后续频谱计算要求位移严格递增，否则会破坏非均匀采样频域映射。
    if not np.all(np.isfinite(values)):
        raise ValueError("Sample positions must be finite.")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("Sample positions must be strictly increasing.")
    return values
