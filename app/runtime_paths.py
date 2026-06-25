from __future__ import annotations

"""运行时资源路径工具，兼容源码运行与 PyInstaller 打包。"""

import sys
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """返回资源文件路径；打包后从 PyInstaller 临时目录读取。"""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    return base / relative_path
