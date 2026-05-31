from __future__ import annotations
"""
结果模型模块。

这个模块的任务是把分析结果收拢成一个统一结构，
让 GUI、导出、测试都能用同一套接口访问结果。

这里特别保留了两类兼容：
1. canonical 键：`h` / `h_prime` / `phi0`
2. legacy 键：`heightMap` / `heightMap_prime` / `phi0_map`

同时，像 `theta_map`、`phase_gap_raw`、`active_range` 这类诊断图层
会通过 `extras` 原样保留下来。
"""

# `dataclass` 用来把结果对象写成更清晰的结构化类型。
from dataclasses import dataclass, field
# `Any` 和 `Mapping` 用来描述“输入可以是任意值的字典映射”。
from typing import Any, Mapping

# 统一使用 NumPy 数组作为结果载体。
import numpy as np


# 这个字典定义了“标准键”和“旧键”的一一对应关系。
# 之所以放成常量，是为了后面提取/回写时都走同一份映射，不要到处手写字符串。
_CANONICAL_ARRAY_KEYS = {
    # `h` 的标准键是 `h`，旧代码里也可能叫 `heightMap`。
    "h": ("h", "heightMap"),
    # `h_prime` 的标准键是 `h_prime`，旧代码里也可能叫 `heightMap_prime`。
    "h_prime": ("h_prime", "heightMap_prime"),
    # `phi0` 的标准键是 `phi0`，旧代码里也可能叫 `phi0_map`。
    "phi0": ("phi0", "phi0_map"),
}


def _extract_array(result: Mapping[str, Any], *keys: str) -> np.ndarray:
    """按候选键顺序提取数组结果。"""
    # 依次尝试所有候选键，谁先存在就用谁。
    for key in keys:
        # 只要当前键在结果字典里，就立即取出并返回。
        if key in result:
            # 统一转成 `float32`，这样 GUI、导出、测试都能用稳定的数据类型。
            return np.asarray(result[key], dtype=np.float32)
    # 如果一个候选键都没找到，就抛出明确错误，避免后面静默出错。
    raise KeyError(f"Missing required result key(s): {', '.join(keys)}")


def _extract_float(result: Mapping[str, Any], *keys: str, default: float = float("nan")) -> float:
    """按候选键顺序提取浮点标量。"""
    # 和数组提取一样，按顺序查找可用键。
    for key in keys:
        # 找到后立即转成 Python `float`。
        if key in result:
            return float(result[key])
    # 一个都没找到时返回默认值，通常是 `nan`。
    return float(default)


def _extract_int(result: Mapping[str, Any], *keys: str, default: int = -1) -> int:
    """按候选键顺序提取整数标量。"""
    # 顺序扫描所有候选键。
    for key in keys:
        # 找到后统一转成 Python `int`。
        if key in result:
            return int(result[key])
    # 如果没找到，返回调用方给的默认整数。
    return int(default)


@dataclass(slots=True)
class AnalysisResult:
    """
    分析结果的统一封装对象。

    - `h`：斜率高度图
    - `h_prime`：当前工作流输出的最终高度图
    - `phi0`：`k0` 处相位图
    - `extras`：其余诊断图层，如 `theta_map`、`phase_gap_raw`、`g0_map`、`active_range` 等
    """
    # `h`：对外约定的斜率高度图。
    h: np.ndarray
    # `h_prime`：对外约定的当前工作流最终高度图。
    h_prime: np.ndarray
    # `phi0`：名义波数 `k0` 处的相位图。
    phi0: np.ndarray
    # `k0_index`：频谱网格里对应 `k0` 的索引，默认未知时记为 -1。
    k0_index: int = -1
    # `k0_value`：名义波数本身，默认未知时记为 `nan`。
    k0_value: float = float("nan")
    # `extras`：其余未进入主字段的图层和标量，全部原样保留。
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, result: Mapping[str, Any]) -> "AnalysisResult":
        """
        从字典结果构造 `AnalysisResult`。

        主字段提成固定属性，其余未知字段全部进入 `extras`，
        这样就不会因为后端新增诊断图层而把 GUI/导出层搞崩。
        """
        # 先把任意映射对象复制成普通字典，便于后面多次读取。
        mapping = dict(result)
        # 这两个标量字段也属于“主字段”，不应该被扔进 extras。
        canonical_keys = {"k0_index", "k0_value"}
        # 把所有标准键和旧键都登记到主字段集合里。
        for keys in _CANONICAL_ARRAY_KEYS.values():
            # `keys` 是一个元组，比如 `("h", "heightMap")`。
            canonical_keys.update(keys)
        # 任何不在主字段集合里的内容，都自动进入 extras。
        extras = {key: value for key, value in mapping.items() if key not in canonical_keys}
        # 构造最终对象。
        return cls(
            # 从标准键/旧键里提取 `h`。
            h=_extract_array(mapping, *_CANONICAL_ARRAY_KEYS["h"]),
            # 从标准键/旧键里提取 `h_prime`。
            h_prime=_extract_array(mapping, *_CANONICAL_ARRAY_KEYS["h_prime"]),
            # 从标准键/旧键里提取 `phi0`。
            phi0=_extract_array(mapping, *_CANONICAL_ARRAY_KEYS["phi0"]),
            # 提取 `k0_index`。
            k0_index=_extract_int(mapping, "k0_index"),
            # 提取 `k0_value`。
            k0_value=_extract_float(mapping, "k0_value"),
            # 写入剩余诊断图层。
            extras=extras,
        )

    @classmethod
    def coerce(cls, result: "AnalysisResult | Mapping[str, Any]") -> "AnalysisResult":
        """把已有对象或映射统一转换成 `AnalysisResult`。"""
        # 如果本来就是 `AnalysisResult`，直接返回，不做重复包装。
        if isinstance(result, AnalysisResult):
            return result
        # 如果是普通映射对象，就走 `from_mapping()` 统一解析。
        if isinstance(result, Mapping):
            return cls.from_mapping(result)
        # 其他类型都不支持，直接报错。
        raise TypeError(f"Unsupported analysis result type: {type(result)!r}")

    def to_mapping(self) -> dict[str, Any]:
        """把结果对象重新转回兼容 canonical + legacy 的字典。"""
        # 先复制 extras，保证所有诊断图层都会保留下来。
        result = dict(self.extras)
        # 再把主字段写回标准键和旧键。
        result.update(
            {
                # 标准键 `h`。
                "h": self.h,
                # 标准键 `h_prime`。
                "h_prime": self.h_prime,
                # 标准键 `phi0`。
                "phi0": self.phi0,
                # 旧键 `heightMap`，继续兼容老调用方。
                "heightMap": self.h,
                # 旧键 `heightMap_prime`，继续兼容老调用方。
                "heightMap_prime": self.h_prime,
                # 旧键 `phi0_map`，继续兼容老调用方。
                "phi0_map": self.phi0,
                # 写回 `k0_index`。
                "k0_index": self.k0_index,
                # 写回 `k0_value`。
                "k0_value": self.k0_value,
            }
        )
        # 返回完整字典。
        return result

    def text_exports(self) -> dict[str, np.ndarray]:
        """定义文本导出时允许直接导出的主结果。"""
        # 文本导出只放最核心的主结果，不默认把所有诊断图层都写出去。
        return {
            # 导出斜率高度。
            "h": self.h,
            # 导出最终高度。
            "h_prime": self.h_prime,
            # 导出 `phi0` 相位图。
            "phi0": self.phi0,
        }

    def figure_exports(self) -> dict[str, np.ndarray]:
        """定义图片导出时允许直接导出的主结果。"""
        # 图片导出当前也只保留三个主图层。
        return {
            # 导出斜率高度图。
            "h": self.h,
            # 导出最终高度图。
            "h_prime": self.h_prime,
            # 导出 `phi0` 相位图。
            "phi0": self.phi0,
        }
