from __future__ import annotations
"""有效扫描范围检测。

该模块只依赖时间维度调制强度，不使用样品类型规则。
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class ActiveRangeResult:
    """有效范围检测结果，帧号字段使用 1-based，便于和采集文件名对照。"""

    start_frame: int
    end_frame: int
    ranges: list[tuple[int, int]]
    score: np.ndarray
    threshold: float
    is_valid: bool
    reason: str


def detect_active_range(cube: np.ndarray, max_pixels: int = 120_000, window: int = 9) -> ActiveRangeResult:
    """根据 z 向时间调制强度自动确定有效扫描范围。"""
    data = np.asarray(cube, dtype=np.float32)
    if data.ndim != 3 or data.shape[2] < 3:
        return _invalid_result(data.shape[2] if data.ndim == 3 else 0, "数据立方体帧数不足")

    sampled = _spatial_sample(data, max_pixels=max_pixels)
    score = _temporal_score(sampled, window=window)
    ranges, threshold = _select_ranges(score)
    if not ranges:
        return ActiveRangeResult(1, int(data.shape[2]), [], score, threshold, False, "未检测到有效时间调制")

    start_frame = min(start for start, _ in ranges)
    end_frame = max(end for _, end in ranges)
    return ActiveRangeResult(start_frame, end_frame, ranges, score, threshold, True, "已检测到有效时间调制")


def apply_active_range(
    cube: np.ndarray,
    positions: np.ndarray | None = None,
    left_expansion_frames: int = 0,
    right_expansion_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, ActiveRangeResult]:
    """检测并裁剪 cube；若传入非均匀采样位置，同步裁剪位置数组。"""
    result = detect_active_range(cube)
    if not result.is_valid:
        return cube, positions, result
    left_margin = max(0, int(left_expansion_frames))
    right_margin = left_margin if right_expansion_frames is None else max(0, int(right_expansion_frames))
    expanded_range = expand_active_range(
        (result.start_frame, result.end_frame),
        frame_count=int(cube.shape[2]),
        left_expansion_frames=left_margin,
        right_expansion_frames=right_margin,
    )
    if expanded_range != (result.start_frame, result.end_frame):
        if left_margin == right_margin:
            reason = f"{result.reason}，左右各扩展 {left_margin} 帧"
        else:
            reason = f"{result.reason}，左扩展 {left_margin} 帧，右扩展 {right_margin} 帧"
        result = ActiveRangeResult(
            start_frame=expanded_range[0],
            end_frame=expanded_range[1],
            ranges=[expanded_range],
            score=result.score,
            threshold=result.threshold,
            is_valid=result.is_valid,
            reason=reason,
        )
    return apply_known_active_range(cube, positions, expanded_range, result)


def expand_active_range(
    active_range: tuple[int, int],
    frame_count: int,
    left_expansion_frames: int,
    right_expansion_frames: int | None = None,
) -> tuple[int, int]:
    """按 1-based 帧号独立扩展有效范围左右边界，并自动裁剪到数据边界内。"""
    left_margin = max(0, int(left_expansion_frames))
    right_margin = left_margin if right_expansion_frames is None else max(0, int(right_expansion_frames))
    start_frame = max(1, int(active_range[0]) - left_margin)
    end_frame = min(int(frame_count), int(active_range[1]) + right_margin)
    return start_frame, max(start_frame, end_frame)


def apply_known_active_range(
    cube: np.ndarray,
    positions: np.ndarray | None,
    active_range: tuple[int, int],
    result: ActiveRangeResult | None = None,
) -> tuple[np.ndarray, np.ndarray | None, ActiveRangeResult]:
    """
    按已经确认的 1-based 有效帧范围裁剪数据。

    自动 K0 已经完成有效范围确认时，正式分析会走这里复用同一段帧，
    避免同一批数据在两个阶段出现范围不一致。
    """
    frame_count = int(cube.shape[2])
    start_frame = max(1, min(int(active_range[0]), frame_count))
    end_frame = max(start_frame, min(int(active_range[1]), frame_count))
    active_result = result
    if active_result is None:
        active_result = ActiveRangeResult(
            start_frame=start_frame,
            end_frame=end_frame,
            ranges=[(start_frame, end_frame)],
            score=np.zeros(frame_count, dtype=np.float32),
            threshold=float("nan"),
            is_valid=True,
            reason="复用自动 K0 已确认范围",
        )

    start_index = start_frame - 1
    end_index = end_frame
    cropped_cube = cube[:, :, start_index:end_index]
    if positions is None:
        return cropped_cube, None, active_result
    return cropped_cube, np.asarray(positions)[start_index:end_index], active_result


def _spatial_sample(cube: np.ndarray, max_pixels: int) -> np.ndarray:
    """等步长空间抽样，降低大图计算量但不改变帧号。"""
    pixel_count = int(cube.shape[0] * cube.shape[1])
    if pixel_count <= max_pixels:
        return cube
    stride = int(np.ceil(np.sqrt(pixel_count / float(max_pixels))))
    return cube[::stride, ::stride, :]


def _temporal_score(cube: np.ndarray, window: int) -> np.ndarray:
    """把每帧转换成全局时间调制分数。"""
    diff_stack = np.diff(cube, axis=2)
    energy = _local_rms(diff_stack, window=window)
    baseline, noise = _pixel_noise_model(energy)
    score = np.zeros(cube.shape[2], dtype=np.float32)
    for index in range(energy.shape[2]):
        snr_frame = np.maximum((energy[:, :, index] - baseline) / noise, 0.0)
        score[index + 1] = float(np.nanpercentile(snr_frame, 95.0))
    score[0] = score[1]
    return score


def _local_rms(diff_stack: np.ndarray, window: int) -> np.ndarray:
    """沿 z 方向计算局部 RMS，表示局部时间调制能量。"""
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    half = window // 2
    power = diff_stack * diff_stack
    output = np.empty_like(diff_stack, dtype=np.float32)
    for index in range(diff_stack.shape[2]):
        left = max(0, index - half)
        right = min(diff_stack.shape[2], index + half + 1)
        output[:, :, index] = np.sqrt(np.mean(power[:, :, left:right], axis=2))
    return output


def _pixel_noise_model(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用每个像素低能量段估计自己的背景和噪声，减少反射率差异影响。"""
    sorted_energy = np.sort(energy, axis=2)
    low_count = max(1, int(round(sorted_energy.shape[2] * 0.2)))
    low_energy = sorted_energy[:, :, :low_count]
    baseline = np.median(low_energy, axis=2)
    mad = np.median(np.abs(low_energy - np.median(low_energy, axis=2, keepdims=True)), axis=2)
    noise = np.maximum(1.4826 * mad, np.finfo(np.float32).eps * np.maximum(np.abs(baseline), 1.0))
    return baseline.astype(np.float32), noise.astype(np.float32)


def _select_ranges(score: np.ndarray) -> tuple[list[tuple[int, int]], float]:
    """在全局时间调制曲线上选择一个或多个有效区间。"""
    baseline = float(np.nanpercentile(score, 20.0))
    peak = float(np.nanpercentile(score, 95.0))
    if not np.isfinite(peak) or peak <= baseline:
        return [], float("nan")

    low_values = score[score <= np.nanpercentile(score, 20.0)]
    noise = 1.4826 * float(np.median(np.abs(low_values - np.median(low_values)))) if low_values.size else 0.0
    span = peak - baseline
    threshold = baseline + max(0.35 * noise, 0.01 * span)
    strong_threshold = baseline + max(1.5 * noise, 0.06 * span)
    mask = _fill_small_gaps(score >= threshold, max_gap=2)
    return _valid_runs(mask, score, strong_threshold, min_length=3), threshold


def _fill_small_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """填补有效区间内部的短缺口，避免弱包络边缘被单帧噪声切断。"""
    filled = np.asarray(mask, dtype=bool).copy()
    starts, ends = _true_runs(~filled)
    for start, end in zip(starts, ends):
        touches_edge = start == 0 or end == filled.size - 1
        if not touches_edge and (end - start + 1) <= max_gap:
            filled[start : end + 1] = True
    return filled


def _valid_runs(mask: np.ndarray, score: np.ndarray, strong_threshold: float, min_length: int) -> list[tuple[int, int]]:
    """保留包含强调制锚点的弱阈值连续区间。"""
    ranges: list[tuple[int, int]] = []
    starts, ends = _true_runs(mask)
    for start, end in zip(starts, ends):
        if end - start + 1 >= min_length and float(np.nanmax(score[start : end + 1])) >= strong_threshold:
            ranges.append((start + 1, end + 1))
    return ranges


def _true_runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.diff(np.concatenate([[False], np.asarray(mask, dtype=bool), [False]]).astype(np.int8))
    return np.where(edges == 1)[0], np.where(edges == -1)[0] - 1


def _invalid_result(frame_count: int, reason: str) -> ActiveRangeResult:
    return ActiveRangeResult(1, max(1, int(frame_count)), [], np.zeros(max(0, frame_count), dtype=np.float32), float("nan"), False, reason)
