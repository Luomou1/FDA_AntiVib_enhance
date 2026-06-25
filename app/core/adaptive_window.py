from __future__ import annotations
"""频谱分析窗函数构造。

该模块把静态窗和自适应窗收敛到同一个入口，避免 FDA、自动 K0、
PhaseGap 等流程各自实现一套窗函数逻辑。
"""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import hilbert


_HAMMING_ALPHA = 0.54
_HANN_ALPHA = 0.50
_EDGE_RATIO = 0.10
_BASELINE_PERCENTILE = 20.0
_DYNAMIC_EPS = 1e-6


def normalize_window_name(window_name: str) -> str:
    """统一窗函数名称，兼容界面文案和历史参数写法。"""
    normalized = str(window_name).strip().lower().replace("-", "_")
    if normalized == "hanning":
        return "hann"
    return normalized


def is_adaptive_window(window_name: str) -> bool:
    """判断当前窗是否需要根据每条曲线单独估计中心和宽度。"""
    return normalize_window_name(window_name) in {"adaptive_hann", "adaptive_hamming"}


def build_analysis_window(curves: np.ndarray, window_name: str, window_alpha: float = _HANN_ALPHA) -> np.ndarray:
    """
    为一批曲线构造窗矩阵。

    返回形状始终为 `(curve_count, sample_count)`，这样调用方可以直接
    与曲线逐点相乘；静态窗会自动广播成每条曲线同一份窗口。
    """
    data = _validate_curves(curves)
    normalized = normalize_window_name(window_name)
    if normalized in {"none", "hann", "hamming"}:
        static_window = build_static_window(normalized, data.shape[1], window_alpha=window_alpha)
        return np.broadcast_to(static_window[None, :], data.shape).astype(np.float32, copy=True)
    if normalized == "adaptive_hann":
        return _build_adaptive_window(data, alpha=_HANN_ALPHA)
    if normalized == "adaptive_hamming":
        return _build_adaptive_window(data, alpha=_HAMMING_ALPHA)
    raise ValueError(f"Unsupported window: {window_name}")


def build_static_window(window_name: str, sample_count: int, window_alpha: float = _HANN_ALPHA) -> np.ndarray:
    """构造不随信号中心变化的一维静态窗。"""
    normalized = normalize_window_name(window_name)
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1.")
    if normalized == "none":
        return np.ones(sample_count, dtype=np.float32)
    if normalized == "hann":
        return np.hanning(sample_count).astype(np.float32)
    if normalized == "hamming":
        return np.hamming(sample_count).astype(np.float32)
    if normalized == "general_hamming":
        return _general_hamming(sample_count, float(window_alpha))
    raise ValueError(f"Unsupported window: {window_name}")


def _validate_curves(curves: np.ndarray) -> np.ndarray:
    """把输入统一成二维有限浮点数组。"""
    data = np.asarray(curves, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.ndim != 2:
        raise ValueError("curves must be a 1D or 2D array.")
    if data.shape[1] < 1:
        raise ValueError("curves must contain at least one sample.")
    if not np.all(np.isfinite(data)):
        raise ValueError("curves must contain only finite values.")
    return data


def _general_hamming(sample_count: int, alpha: float) -> np.ndarray:
    """按 generalized Hamming 公式构造一维窗。"""
    if sample_count == 1:
        return np.ones(1, dtype=np.float32)
    index = np.arange(sample_count, dtype=np.float32)
    window = alpha - (1.0 - alpha) * np.cos(2.0 * np.pi * index / float(sample_count - 1))
    return window.astype(np.float32)


def _build_adaptive_window(curves: np.ndarray, alpha: float) -> np.ndarray:
    """基于 Hilbert 包络为每条曲线构造中心对齐的非对称余弦窗。"""
    sample_count = curves.shape[1]
    if sample_count < 3:
        return np.ones_like(curves, dtype=np.float32)

    raw_envelope, smooth_envelope = _estimate_envelopes(curves)
    centers, left_widths, right_widths, valid = _estimate_window_geometry(raw_envelope, smooth_envelope)
    adaptive = _asymmetric_general_hamming(sample_count, centers, left_widths, right_widths, alpha)

    # 无明显调制的曲线不强行自适应，回退到同长度静态窗，避免噪声峰决定窗口中心。
    fallback_name = "hann" if abs(alpha - _HANN_ALPHA) <= 1e-9 else "hamming"
    static_window = build_static_window(fallback_name, sample_count)
    adaptive[~valid, :] = static_window[None, :]
    return adaptive.astype(np.float32)


def _estimate_envelopes(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用解析信号估计包络，并返回原始包络和平滑包络。"""
    centered = curves - np.mean(curves, axis=1, keepdims=True)
    raw_envelope = np.abs(hilbert(centered, axis=1)).astype(np.float32)
    smooth_envelope = raw_envelope
    smooth_size = _resolve_smooth_size(curves.shape[1])
    if smooth_size > 1:
        smooth_envelope = uniform_filter1d(raw_envelope, size=smooth_size, axis=1, mode="nearest")
    return raw_envelope.astype(np.float32), smooth_envelope.astype(np.float32)


def _resolve_smooth_size(sample_count: int) -> int:
    """根据采样长度选择奇数平滑窗口，避免过度抹平窄包络。"""
    size = max(3, int(round(sample_count * 0.05)))
    size = min(size, max(1, sample_count // 4))
    if size % 2 == 0:
        size += 1
    return max(1, min(size, sample_count))


def _estimate_window_geometry(
    raw_envelope: np.ndarray,
    smooth_envelope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从包络估计窗中心、左右边界宽度和调制有效性。"""
    curve_count, sample_count = smooth_envelope.shape
    centers = np.argmax(raw_envelope, axis=1).astype(np.float32)
    left_widths = np.empty(curve_count, dtype=np.float32)
    right_widths = np.empty(curve_count, dtype=np.float32)
    valid = np.zeros(curve_count, dtype=bool)
    min_half_width = max(2, sample_count // 20)

    for row in range(curve_count):
        row_envelope = smooth_envelope[row]
        center_index = int(centers[row])
        baseline = float(np.nanpercentile(row_envelope, _BASELINE_PERCENTILE))
        peak = float(row_envelope[center_index])
        dynamic = peak - baseline
        if not np.isfinite(dynamic) or dynamic <= _DYNAMIC_EPS:
            left_widths[row] = max(float(center_index), float(min_half_width))
            right_widths[row] = max(float(sample_count - 1 - center_index), float(min_half_width))
            continue

        threshold = baseline + _EDGE_RATIO * dynamic
        left_edge, right_edge = _find_threshold_segment(row_envelope, center_index, threshold)
        left_edge = max(0, min(left_edge, center_index - min_half_width))
        right_edge = min(sample_count - 1, max(right_edge, center_index + min_half_width))
        left_widths[row] = max(float(center_index - left_edge), 1.0)
        right_widths[row] = max(float(right_edge - center_index), 1.0)
        valid[row] = True

    return centers, left_widths, right_widths, valid


def _find_threshold_segment(envelope: np.ndarray, center_index: int, threshold: float) -> tuple[int, int]:
    """从包络峰向左右找到同一有效包络段的弱阈值边界。"""
    left = center_index
    while left > 0 and float(envelope[left - 1]) >= threshold:
        left -= 1
    right = center_index
    last_index = envelope.shape[0] - 1
    while right < last_index and float(envelope[right + 1]) >= threshold:
        right += 1
    return left, right


def _asymmetric_general_hamming(
    sample_count: int,
    centers: np.ndarray,
    left_widths: np.ndarray,
    right_widths: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """按左右独立宽度生成非对称 generalized Hamming 窗矩阵。"""
    index = np.arange(sample_count, dtype=np.float32)[None, :]
    center = centers[:, None].astype(np.float32)
    left = np.maximum(left_widths[:, None].astype(np.float32), 1.0)
    right = np.maximum(right_widths[:, None].astype(np.float32), 1.0)
    normalized = np.where(index <= center, (index - center) / left, (index - center) / right)
    inside = np.abs(normalized) <= 1.0
    window = alpha + (1.0 - alpha) * np.cos(np.pi * normalized)
    return np.where(inside, window, 0.0).astype(np.float32)

