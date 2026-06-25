from __future__ import annotations
"""扫描步长诊断与坐标整理。

GFDA 软件估计模块会先从图像序列中估计每帧扫描坐标；本模块只负责把
这些软件坐标显式整理成可导出的步长结果，并额外给出一个最小改动的
单调投影版本，供诊断或低置信修复参考。scan_log 的设备反馈路径不经过这里。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ScanStepResult:
    """一次扫描坐标整理的结构化结果。"""

    raw_positions_um: np.ndarray
    analysis_positions_um: np.ndarray
    monotone_positions_um: np.ndarray
    raw_increments_um: np.ndarray
    monotone_increments_um: np.ndarray
    reversal_mask: np.ndarray
    correction_um: np.ndarray
    nominal_step_um: float
    min_increment_um: float
    strategy: str

    @property
    def reversal_count(self) -> int:
        return int(np.count_nonzero(self.reversal_mask))

    @property
    def max_abs_correction_um(self) -> float:
        if self.correction_um.size == 0:
            return 0.0
        return float(np.max(np.abs(self.correction_um)))

    def to_mapping(self, prefix: str = "scan") -> dict[str, np.ndarray | float | int | str]:
        """转换为核心结果可直接携带的诊断字段。"""
        return {
            f"{prefix}_positions_raw_um": self.raw_positions_um.astype(np.float32),
            f"{prefix}_positions_used_um": self.analysis_positions_um.astype(np.float32),
            f"{prefix}_positions_monotone_um": self.monotone_positions_um.astype(np.float32),
            f"{prefix}_step_raw_um": self.raw_increments_um.astype(np.float32),
            f"{prefix}_step_monotone_um": self.monotone_increments_um.astype(np.float32),
            f"{prefix}_step_reversal_mask": self.reversal_mask.astype(np.float32),
            f"{prefix}_position_correction_um": self.correction_um.astype(np.float32),
            f"{prefix}_nominal_step_um": float(self.nominal_step_um),
            f"{prefix}_min_increment_um": float(self.min_increment_um),
            f"{prefix}_reversal_count": int(self.reversal_count),
            f"{prefix}_max_abs_correction_um": float(self.max_abs_correction_um),
            f"{prefix}_strategy": self.strategy,
        }


def _validate_positions(positions_um: np.ndarray) -> np.ndarray:
    values = np.asarray(positions_um, dtype=np.float32)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("Sample positions must be a 1D array with at least two samples.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Sample positions must be finite.")
    return values


def _infer_nominal_step_um(positions_um: np.ndarray, fallback_step_um: float | None) -> float:
    diffs = np.diff(positions_um.astype(np.float64))
    positive = diffs[diffs > 0.0]
    if positive.size > 0:
        return float(np.median(positive))
    if fallback_step_um is not None and float(fallback_step_um) > 0.0:
        return float(fallback_step_um)
    nonzero = np.abs(diffs[np.abs(diffs) > 0.0])
    if nonzero.size > 0:
        return float(np.median(nonzero))
    return 1.0


def _resolve_min_increment_um(positions_um: np.ndarray, nominal_step_um: float, min_increment_um: float | None) -> float:
    if min_increment_um is not None:
        value = float(min_increment_um)
        if value <= 0.0:
            raise ValueError("min_increment_um must be positive.")
        return value
    scale = max(abs(float(nominal_step_um)), float(np.ptp(positions_um)) / max(1, positions_um.size - 1), 1.0)
    # 只用于把单调投影从 non-decreasing 推成 strictly increasing；取很小的自适应间隔，
    # 避免把真实倒退段过度拉直。
    return max(scale * 1e-6, 1e-9)


def _pava_non_decreasing(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators algorithm，求加权最小二乘单调投影。"""
    y = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if y.shape != w.shape:
        raise ValueError("values and weights must have the same shape.")
    if np.any(w <= 0.0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and positive.")

    block_values: list[float] = []
    block_weights: list[float] = []
    block_lengths: list[int] = []
    for value, weight in zip(y, w):
        block_values.append(float(value))
        block_weights.append(float(weight))
        block_lengths.append(1)
        while len(block_values) >= 2 and block_values[-2] > block_values[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / merged_weight
            merged_length = block_lengths[-2] + block_lengths[-1]
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_lengths[-2:] = [merged_length]

    projected = np.empty_like(y)
    cursor = 0
    for value, length in zip(block_values, block_lengths):
        projected[cursor : cursor + length] = value
        cursor += length
    return projected


def strict_monotone_projection_um(
    positions_um: np.ndarray,
    *,
    nominal_step_um: float | None = None,
    min_increment_um: float | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """把任意扫描坐标投影成严格递增坐标，且保持最小加权改动。"""
    raw = _validate_positions(positions_um).astype(np.float64)
    nominal = _infer_nominal_step_um(raw, nominal_step_um)
    min_increment = _resolve_min_increment_um(raw, nominal, min_increment_um)
    sample_index = np.arange(raw.size, dtype=np.float64)
    shifted = raw - min_increment * sample_index
    if weights is None:
        pava_weights = np.ones_like(shifted, dtype=np.float64)
    else:
        pava_weights = np.asarray(weights, dtype=np.float64)
    projected = _pava_non_decreasing(shifted, pava_weights) + min_increment * sample_index
    return projected.astype(np.float32)


def build_scan_step_result(
    positions_um: np.ndarray,
    *,
    nominal_step_um: float | None = None,
    min_increment_um: float | None = None,
    use_monotone_for_analysis: bool = False,
) -> ScanStepResult:
    """生成软件估计扫描坐标的显式步长诊断。"""
    raw = _validate_positions(positions_um)
    nominal = _infer_nominal_step_um(raw, nominal_step_um)
    min_increment = _resolve_min_increment_um(raw, nominal, min_increment_um)
    monotone = strict_monotone_projection_um(
        raw,
        nominal_step_um=nominal,
        min_increment_um=min_increment,
    )
    raw_increments = np.diff(raw).astype(np.float32)
    monotone_increments = np.diff(monotone).astype(np.float32)
    reversal_mask = raw_increments <= 0.0
    analysis_positions = monotone if use_monotone_for_analysis else raw
    strategy = "adaptive_monotone_projection" if use_monotone_for_analysis else "software_nonuniform_gfda"
    if not np.any(reversal_mask) and not use_monotone_for_analysis:
        strategy = "software_monotonic_gfda"
    return ScanStepResult(
        raw_positions_um=raw.astype(np.float32),
        analysis_positions_um=analysis_positions.astype(np.float32),
        monotone_positions_um=monotone.astype(np.float32),
        raw_increments_um=raw_increments,
        monotone_increments_um=monotone_increments,
        reversal_mask=reversal_mask,
        correction_um=(monotone - raw).astype(np.float32),
        nominal_step_um=float(nominal),
        min_increment_um=float(min_increment),
        strategy=strategy,
    )
