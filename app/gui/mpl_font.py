from __future__ import annotations

"""Matplotlib 中文字体辅助工具。"""

import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager

_PREFERRED_SANS_SERIF = (
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
)


def _installed_font_names() -> set[str]:
    """收集当前 Matplotlib 已识别到的字体名。"""
    return {font.name for font in fontManager.ttflist}


def resolve_sans_serif_fonts() -> list[str]:
    """
    解析当前环境里真正可用的 sans-serif 字体列表。

    这里先按项目偏好的中文字体顺序筛一遍，再在完全缺失时回退到
    `DejaVu Sans`，避免把不存在的字体名直接塞给 Matplotlib 后
    在每次绘图时反复打印 `findfont` 告警。
    """
    installed = _installed_font_names()
    resolved = [name for name in _PREFERRED_SANS_SERIF if name in installed]
    if resolved:
        return resolved
    return ["DejaVu Sans"]


def configure_matplotlib_fonts() -> FontProperties:
    """
    配置 Matplotlib 的中文字体回退链，并返回同源 `FontProperties`。

    这样标题、坐标轴和图例都走同一套字体选择，避免界面不同区域
    因为各自单独选字而出现不一致或额外告警。
    """
    families = resolve_sans_serif_fonts()
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = families
    mpl.rcParams["axes.unicode_minus"] = False
    return FontProperties(family=families)
