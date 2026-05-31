from __future__ import annotations
"""
Phase-Gap 核心分析模块。

这个模块对应的是两篇文献里真正“拉开差距”的那部分：
- 先构造 wrapped phase gap
- 再做质量评估
- 再做 quality-guided connect
- 再做相干图平滑与 phase-gap 拟合
- 最后通过 final gap 恢复 fringe order，并得到最终高度 `h`

如果把 `fda_baseline.py` 看成“基线输入层”，
这里就是“文献算法的主体实现层”。
"""

import heapq

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

TAU = 2.0 * np.pi
SUPPORTED_PHASE_GAP_METHODS = {
    "quality_guided",
    "circular_average",
    "constant",
    "constant_phase_gap",
    "robust_model_fit",
    "branch_cut",
    "weighted_least_squares",
    "weighted_ls",
    "minimum_lp",
    "minimum_l_p",
}


def _wrap_to_pi(values: np.ndarray) -> np.ndarray:
    """把任意相位值包裹回 `[-pi, pi)` 区间。"""
    return np.angle(np.exp(1j * np.asarray(values, dtype=np.float64))).astype(np.float32)


def _unwrap_relative_to_reference(value: float, reference: float) -> float:
    """
    以参考值为基准，把单个值移动到“最接近参考值的 2π 分支”上。

    这个小函数是 connect 阶段的基础工具。
    """
    return float(value + TAU * np.rint((reference - value) / TAU))


def _normalize_map(values: np.ndarray) -> np.ndarray:
    """把有效值线性压到 `[0, 1]`，供后续置信度权重使用。"""
    values = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)
    valid = values[finite]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if abs(hi - lo) < 1e-12:
        return np.ones_like(values, dtype=np.float32)
    normalized = (values - lo) / (hi - lo)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _iter_neighbors(y: int, x: int, height: int, width: int):
    """枚举 8 邻域像素。"""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width:
                yield ny, nx


def _weighted_circular_mean(field: np.ndarray, weights: np.ndarray, size: int) -> np.ndarray:
    """
    在局部窗口里做加权圆均值。

    这里不能直接对相位做普通平均，否则在 `pi/-pi` 附近会出错。
    所以先转到复平面，再取角度。
    """
    complex_field = weights * np.exp(1j * field)
    sum_real = uniform_filter(np.real(complex_field), size=size, mode="nearest")
    sum_imag = uniform_filter(np.imag(complex_field), size=size, mode="nearest")
    sum_weights = uniform_filter(weights, size=size, mode="nearest")
    mean = np.zeros_like(field, dtype=np.float32)
    valid = sum_weights > 1e-12
    mean[valid] = np.angle(sum_real[valid] + 1j * sum_imag[valid]).astype(np.float32)
    if np.any(weights > 0):
        global_mean = float(np.angle(np.sum(complex_field[weights > 0])))
    else:
        global_mean = 0.0
    mean[~valid] = global_mean
    return mean.astype(np.float32)


def _global_circular_average(field: np.ndarray) -> float:
    """
    计算整幅相位场的圆均值，并恢复到主导的 `2π` 分支。

    这里为什么不能直接做普通平均：
    - phase gap 本质上是圆周量，`-pi` 和 `pi` 在几何上很近
    - 普通平均会把这两者错误地平均到 0 附近

    所以我们先在复平面上求圆均值，再通过全场中位数估一个主导分支，
    把均值拉回到“最常见的那个 2π 周期”上。
    """
    values = np.asarray(field, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return 0.0
    wrapped_mean = float(np.angle(np.sum(np.exp(1j * values[finite]))))
    cycle_offset = float(np.rint(np.median((values[finite] - wrapped_mean) / TAU)))
    return wrapped_mean + cycle_offset * TAU


def compute_merit_map(wrapped_gap: np.ndarray) -> np.ndarray:
    """
    计算 wrapped phase gap 的局部质量图。

    merit 越大，说明该像素与邻域不一致越严重，
    一般意味着噪声更大、局部不稳定或更可能是 branch error。
    """
    wrapped_gap = np.asarray(wrapped_gap, dtype=np.float32)
    height, width = wrapped_gap.shape
    merit = np.full((height, width), np.nan, dtype=np.float32)

    for y in range(height):
        for x in range(width):
            center = float(wrapped_gap[y, x])
            if not np.isfinite(center):
                continue
            local_sum = 0.0
            local_count = 0
            for ny, nx in _iter_neighbors(y, x, height, width):
                neighbor = float(wrapped_gap[ny, nx])
                if not np.isfinite(neighbor):
                    continue
                aligned = _unwrap_relative_to_reference(neighbor, center)
                local_sum += abs(aligned - center)
                local_count += 1
            if local_count > 0:
                merit[y, x] = local_sum / local_count

    return merit


_compute_merit_map = compute_merit_map


def quality_guided_connect(
    wrapped_gap: np.ndarray,
    merit_map: np.ndarray,
    merit_threshold: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    对 wrapped phase gap 做质量引导的连接。

    策略是：
    - 先从 merit 低的区域开始连
    - 优先扩展高质量连通区
    - 质量太差的区域先留空，后面交给 fill / fit 处理

    这里的 connect 只负责“局部连续性”：
    - 它会让相邻像素尽量落在同一个连续分支上
    - 但它不会保证整个连通区已经回到和 raw gap 同一个绝对 `2π` 分支

    因此后面还需要一次“相对 raw gap 的逐像素分支对齐”，
    否则在相位整体跨过 `pi` 时，很容易整片掉到错误分支。
    """
    wrapped_gap = np.asarray(wrapped_gap, dtype=np.float32)
    merit_map = np.asarray(merit_map, dtype=np.float32)
    height, width = wrapped_gap.shape
    connected = np.full((height, width), np.nan, dtype=np.float32)
    connected_mask = np.zeros((height, width), dtype=bool)
    valid = np.isfinite(wrapped_gap) & np.isfinite(merit_map)
    if not np.any(valid):
        return connected, connected_mask

    merit_values = merit_map[valid]
    if merit_threshold is None:
        q1 = float(np.quantile(merit_values, 0.25))
        q3 = float(np.quantile(merit_values, 0.75))
        merit_threshold = max(q3 + 1.5 * (q3 - q1), float(np.median(merit_values)) + 1e-6)

    priority: list[tuple[float, int, int]] = []

    def push_neighbors(y0: int, x0: int) -> None:
        for ny, nx in _iter_neighbors(y0, x0, height, width):
            if valid[ny, nx] and not connected_mask[ny, nx]:
                heapq.heappush(priority, (float(merit_map[ny, nx]), ny, nx))

    while True:
        remaining = valid & ~connected_mask & (merit_map <= merit_threshold)
        if not np.any(remaining):
            break
        seed_index = int(np.argmin(np.where(remaining, merit_map, np.inf)))
        sy, sx = np.unravel_index(seed_index, merit_map.shape)
        connected[sy, sx] = wrapped_gap[sy, sx]
        connected_mask[sy, sx] = True
        priority.clear()
        push_neighbors(sy, sx)

        while priority:
            _, y, x = heapq.heappop(priority)
            if connected_mask[y, x] or merit_map[y, x] > merit_threshold or not valid[y, x]:
                continue

            references = [
                float(connected[ny, nx])
                for ny, nx in _iter_neighbors(y, x, height, width)
                if connected_mask[ny, nx] and np.isfinite(connected[ny, nx])
            ]
            if not references:
                continue

            reference = float(np.mean(references))
            connected[y, x] = _unwrap_relative_to_reference(float(wrapped_gap[y, x]), reference)
            connected_mask[y, x] = True
            push_neighbors(y, x)

    return connected, connected_mask


_quality_guided_connect = quality_guided_connect


def _estimate_cycle_offset(raw_gap: np.ndarray, connected_gap: np.ndarray, connected_mask: np.ndarray) -> float:
    """估计 raw gap 与 connected gap 之间整体差了多少个 `2π` 周期。"""
    if not np.any(connected_mask):
        return 0.0
    wrapped = _wrap_to_pi(connected_gap)
    cycle_terms = (raw_gap[connected_mask] - wrapped[connected_mask]) / TAU
    return float(np.rint(np.median(cycle_terms)))


def _fill_unconnected_regions(
    raw_gap: np.ndarray,
    connected_gap: np.ndarray,
    connected_mask: np.ndarray,
) -> np.ndarray:
    """
    对未连接区域做填补。

    文献/专利里的保守做法是：对高变化或未连接区域先退回
    一个全场圆均值常量，后续再由拟合与置信度融合去细化。

    这样做看起来“粗暴”，但它有两个现实好处：
    1. 不会把局部坏点的错误分支继续扩散到更大区域
    2. 不会在低置信区域凭空制造看起来很平滑、其实完全错误的表面形状
    """
    filled = connected_gap.copy()
    finite_raw = np.isfinite(raw_gap)
    fill_value = _global_circular_average(raw_gap)
    fill_mask = finite_raw & ~connected_mask
    filled[fill_mask] = float(fill_value)
    return filled.astype(np.float32)


def smooth_coherence_profile(coherence_proxy: np.ndarray, size: int, limit: float | None = None) -> np.ndarray:
    """
    对 coherence / theta 图做有限度平滑。

    目标不是“把图整体磨平”，而是尽量压制文献中提到的
    diffraction spike 或局部尖峰，同时保留真实结构。
    """
    coherence_proxy = np.asarray(coherence_proxy, dtype=np.float32)
    smooth = uniform_filter(coherence_proxy, size=max(3, int(size)), mode="nearest")
    delta = coherence_proxy - smooth
    if limit is None:
        spread = float(np.nanstd(coherence_proxy))
        limit_value = max(spread * 0.5, 1.0)
    else:
        limit_value = float(limit)
    delta = np.clip(delta, -limit_value, limit_value)
    return (coherence_proxy - delta).astype(np.float32)


_smooth_coherence_proxy = smooth_coherence_profile


def _merit_to_confidence(merit_map: np.ndarray) -> np.ndarray:
    """把 merit 图转换成 `[0, 1]` 置信度图。"""
    merit_map = np.asarray(merit_map, dtype=np.float32)
    finite = np.isfinite(merit_map)
    if not np.any(finite):
        return np.zeros_like(merit_map, dtype=np.float32)
    valid = merit_map[finite]
    lo = float(np.min(valid))
    hi = float(np.max(valid))
    if abs(hi - lo) < 1e-12:
        confidence = np.ones_like(merit_map, dtype=np.float32)
        confidence[~finite] = 0.0
        return confidence
    normalized = (merit_map - lo) / (hi - lo)
    confidence = 1.0 - np.clip(normalized, 0.0, 1.0)
    confidence[~finite] = 0.0
    return confidence.astype(np.float32)


def _build_phase_gap_confidence(
    merit_map: np.ndarray,
    amplitude_map: np.ndarray,
    trusted_mask: np.ndarray | None = None,
) -> np.ndarray:
    """融合谱峰幅值与 merit，得到 strategy 通用置信度图。"""
    amplitude_weights = _normalize_map(amplitude_map)
    amplitude_weights = np.clip(amplitude_weights, 0.05, 1.0).astype(np.float32)
    merit_penalty = np.clip(_merit_to_confidence(merit_map), 0.1, 1.0)
    confidence = (amplitude_weights * merit_penalty).astype(np.float32)
    if trusted_mask is not None:
        # 未被当前策略直接信任的区域只保留弱置信度，后续更多依赖模型或填补。
        confidence = np.where(trusted_mask, confidence, amplitude_weights * 0.05).astype(np.float32)
    return confidence


def _blend_observation_with_model(
    raw_gap: np.ndarray,
    observed_gap: np.ndarray,
    observed_mask: np.ndarray,
    confidence_map: np.ndarray,
    theta_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把局部观测、保守填补和低阶模型融合成 final phase gap。"""
    filled_gap = _fill_unconnected_regions(raw_gap, observed_gap, observed_mask)
    fit_weights = np.where(observed_mask, confidence_map, 0.0).astype(np.float32)
    fit_gap = fit_phase_gap_surface(observed_gap, fit_weights, theta_map)
    finite_raw = np.isfinite(raw_gap)
    # 高置信区更信任实际观测，低置信区退回低阶模型，避免坏点扩散。
    final_gap = np.where(
        finite_raw,
        fit_gap + (filled_gap - fit_gap) * confidence_map,
        np.nan,
    ).astype(np.float32)
    return filled_gap.astype(np.float32), fit_gap.astype(np.float32), final_gap


def fit_phase_gap_surface(
    gap_map: np.ndarray,
    confidence_map: np.ndarray,
    coherence_proxy: np.ndarray,
) -> np.ndarray:
    """
    拟合文献中使用的低阶 phase-gap 曲面。

    当前实现使用：
    `c0 + c1*x + c2*y + c3*theta`

    这里的 `theta` 不是普通高度图，而是与相位同单位的相干等效图。
    """
    gap_map = np.asarray(gap_map, dtype=np.float64)
    confidence_map = np.asarray(confidence_map, dtype=np.float64)
    coherence_proxy = np.asarray(coherence_proxy, dtype=np.float64)
    height, width = gap_map.shape
    yy, xx = np.mgrid[0:height, 0:width]

    x_norm = (xx - (width - 1) / 2.0) / max(width - 1, 1)
    y_norm = (yy - (height - 1) / 2.0) / max(height - 1, 1)
    coherence_norm = coherence_proxy.copy()
    finite = np.isfinite(coherence_norm)
    if np.any(finite):
        centered = coherence_norm[finite]
        spread = float(np.std(centered))
        center = float(np.mean(centered))
        if spread > 1e-12:
            coherence_norm = (coherence_norm - center) / spread
        else:
            coherence_norm = coherence_norm - center
    else:
        coherence_norm = np.zeros_like(coherence_norm)

    # 设计矩阵的四列分别对应：
    # 1. 常数项
    # 2. x 方向 tip
    # 3. y 方向 tilt
    # 4. 与 coherence/theta 相关的形状基函数
    design = np.stack([np.ones_like(x_norm), x_norm, y_norm, coherence_norm], axis=-1).reshape(-1, 4)
    response = gap_map.reshape(-1)
    weights = np.clip(confidence_map.reshape(-1), 1e-3, 1.0)
    # 这里一定要同时检查 response、weight、design 三者都有效。
    # 之前如果 design 里混进 NaN，`lstsq` 会直接在 SVD 阶段崩掉，
    # 在真实数据的边缘缺失或低信噪区域上尤其容易出现。
    valid = np.isfinite(response) & np.isfinite(weights) & np.all(np.isfinite(design), axis=1)
    if np.count_nonzero(valid) < 4:
        return np.where(np.isfinite(gap_map), gap_map, 0.0).astype(np.float32)

    sqrt_weights = np.sqrt(weights[valid])
    aw = design[valid] * sqrt_weights[:, None]
    bw = response[valid] * sqrt_weights
    coeffs, *_ = np.linalg.lstsq(aw, bw, rcond=None)
    fitted = design @ coeffs
    fitted_map = fitted.reshape(height, width)
    fallback = np.where(np.isfinite(gap_map), gap_map, 0.0)
    # 对拟合后仍然无效的点，用原始 gap（若存在）或 0 回填，
    # 保证后续 blend 不会再因为 NaN 把整张图拖坏。
    fitted_map = np.where(np.isfinite(fitted_map), fitted_map, fallback)
    return fitted_map.astype(np.float32)


_fit_gap_surface = fit_phase_gap_surface


def _quality_guided_phase_gap_solution(
    raw_gap: np.ndarray,
    wrapped_gap: np.ndarray,
    merit_map: np.ndarray,
    theta_map: np.ndarray,
    amplitude_map: np.ndarray,
) -> dict[str, np.ndarray | dict[str, np.ndarray | str]]:
    """使用当前项目既有 quality-guided 路径生成统一 strategy 输出。"""
    connected_gap, connected_mask = quality_guided_connect(wrapped_gap, merit_map)
    connected_gap = np.where(
        connected_mask,
        connected_gap + TAU * np.rint((raw_gap - connected_gap) / TAU),
        np.nan,
    ).astype(np.float32)
    confidence_map = _build_phase_gap_confidence(merit_map, amplitude_map, connected_mask)
    filled_gap, fit_gap, final_gap = _blend_observation_with_model(
        raw_gap=raw_gap,
        observed_gap=connected_gap,
        observed_mask=connected_mask,
        confidence_map=confidence_map,
        theta_map=theta_map,
    )
    return {
        "phase_gap_final": final_gap,
        "phase_gap_connected": connected_gap.astype(np.float32),
        "connected_mask": connected_mask,
        "confidence_map": confidence_map,
        "diagnostic_maps": {
            "method": "quality_guided",
            "phase_gap_filled": filled_gap,
            "phase_gap_fit": fit_gap,
        },
    }


def _constant_phase_gap_solution(
    raw_gap: np.ndarray,
    merit_map: np.ndarray,
    amplitude_map: np.ndarray,
) -> dict[str, np.ndarray | dict[str, np.ndarray | str | float]]:
    """用全场圆均值作为常量 phase-gap 基线。"""
    finite_raw = np.isfinite(raw_gap)
    constant_value = float(_global_circular_average(raw_gap))
    final_gap = np.where(finite_raw, constant_value, np.nan).astype(np.float32)
    confidence_map = _build_phase_gap_confidence(merit_map, amplitude_map, finite_raw)
    return {
        "phase_gap_final": final_gap,
        "phase_gap_connected": final_gap.copy(),
        "connected_mask": finite_raw,
        "confidence_map": confidence_map,
        "diagnostic_maps": {
            "method": "circular_average",
            "phase_gap_filled": final_gap.copy(),
            "phase_gap_fit": final_gap.copy(),
            "phase_gap_constant": np.full_like(final_gap, constant_value, dtype=np.float32),
        },
    }


def _robust_model_phase_gap_solution(
    raw_gap: np.ndarray,
    merit_map: np.ndarray,
    theta_map: np.ndarray,
    amplitude_map: np.ndarray,
    iterations: int = 5,
) -> dict[str, np.ndarray | dict[str, np.ndarray | str]]:
    """迭代分支对齐后做 Huber 加权低阶模型拟合。"""
    finite_raw = np.isfinite(raw_gap)
    confidence_map = _build_phase_gap_confidence(merit_map, amplitude_map, finite_raw)
    model = np.where(finite_raw, _global_circular_average(raw_gap), np.nan).astype(np.float32)
    aligned_gap = model.copy()
    robust_weights = confidence_map.copy()

    for _ in range(max(1, int(iterations))):
        # 先把 raw gap 拉到当前模型附近的 2π 分支，再用残差估计 Huber 权重。
        aligned_gap = np.where(
            finite_raw,
            raw_gap - TAU * np.rint((raw_gap - model) / TAU),
            np.nan,
        ).astype(np.float32)
        residual = aligned_gap - model
        valid = finite_raw & np.isfinite(residual)
        if not np.any(valid):
            break
        median_abs = float(np.median(np.abs(residual[valid] - np.median(residual[valid]))))
        scale = max(1.4826 * median_abs, 1e-3)
        huber_limit = 1.345 * scale
        huber = np.minimum(1.0, huber_limit / np.maximum(np.abs(residual), 1e-6))
        robust_weights = (confidence_map * np.where(np.isfinite(huber), huber, 0.0)).astype(np.float32)
        next_model = fit_phase_gap_surface(aligned_gap, robust_weights, theta_map)
        if np.allclose(next_model[valid], model[valid], atol=1e-5, rtol=1e-5):
            model = next_model
            break
        model = next_model

    return {
        "phase_gap_final": np.where(finite_raw, model, np.nan).astype(np.float32),
        "phase_gap_connected": aligned_gap.astype(np.float32),
        "connected_mask": finite_raw,
        "confidence_map": robust_weights.astype(np.float32),
        "diagnostic_maps": {
            "method": "robust_model_fit",
            "phase_gap_filled": aligned_gap.astype(np.float32),
            "phase_gap_fit": model.astype(np.float32),
            "phase_gap_robust_weight": robust_weights.astype(np.float32),
        },
    }


def _compute_residue_map(wrapped_gap: np.ndarray) -> np.ndarray:
    """计算每个 2x2 小环的 wrapped-gradient residue。"""
    top_left = wrapped_gap[:-1, :-1]
    top_right = wrapped_gap[:-1, 1:]
    bottom_right = wrapped_gap[1:, 1:]
    bottom_left = wrapped_gap[1:, :-1]
    loop_sum = (
        _wrap_to_pi(top_right - top_left)
        + _wrap_to_pi(bottom_right - top_right)
        + _wrap_to_pi(bottom_left - bottom_right)
        + _wrap_to_pi(top_left - bottom_left)
    )
    residue = np.rint(loop_sum / TAU).astype(np.int32)
    finite = np.isfinite(top_left) & np.isfinite(top_right) & np.isfinite(bottom_right) & np.isfinite(bottom_left)
    return np.where(finite, residue, 0).astype(np.int32)


def _draw_branch_cut(cut_mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    """在像素网格上画一条简化 Manhattan branch cut。"""
    y0, x0 = start
    y1, x1 = end
    step_y = 1 if y1 >= y0 else -1
    for y in range(y0, y1 + step_y, step_y):
        cut_mask[min(max(y, 0), cut_mask.shape[0] - 1), min(max(x0, 0), cut_mask.shape[1] - 1)] = True
    step_x = 1 if x1 >= x0 else -1
    for x in range(x0, x1 + step_x, step_x):
        cut_mask[min(max(y1, 0), cut_mask.shape[0] - 1), min(max(x, 0), cut_mask.shape[1] - 1)] = True


def _build_simple_branch_cut_mask(residue_map: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """把正负 residue 贪心配对，未配对 residue 连接到最近边界。"""
    cut_mask = np.zeros(shape, dtype=bool)
    positives = [tuple(pos) for pos in np.argwhere(residue_map > 0)]
    negatives = [tuple(pos) for pos in np.argwhere(residue_map < 0)]

    while positives and negatives:
        source = positives.pop(0)
        target_index = int(np.argmin([abs(source[0] - y) + abs(source[1] - x) for y, x in negatives]))
        target = negatives.pop(target_index)
        _draw_branch_cut(cut_mask, source, target)

    height, width = shape
    for source in positives + negatives:
        y, x = source
        boundary_targets = [(0, x), (height - 1, x), (y, 0), (y, width - 1)]
        target = min(boundary_targets, key=lambda item: abs(item[0] - y) + abs(item[1] - x))
        _draw_branch_cut(cut_mask, source, target)
    return cut_mask


def _branch_cut_phase_gap_solution(
    raw_gap: np.ndarray,
    wrapped_gap: np.ndarray,
    merit_map: np.ndarray,
    theta_map: np.ndarray,
    amplitude_map: np.ndarray,
) -> dict[str, np.ndarray | dict[str, np.ndarray | str]]:
    """用简化 Goldstein 风格 residue 隔离后再局部 flood-fill 展开。"""
    residue_map = _compute_residue_map(wrapped_gap)
    cut_mask = _build_simple_branch_cut_mask(residue_map, raw_gap.shape)
    valid = np.isfinite(wrapped_gap) & np.isfinite(merit_map) & ~cut_mask
    connected = np.full(raw_gap.shape, np.nan, dtype=np.float32)
    connected_mask = np.zeros(raw_gap.shape, dtype=bool)
    height, width = raw_gap.shape

    while np.any(valid & ~connected_mask):
        seed = int(np.argmin(np.where(valid & ~connected_mask, merit_map, np.inf)))
        sy, sx = np.unravel_index(seed, raw_gap.shape)
        connected[sy, sx] = raw_gap[sy, sx]
        connected_mask[sy, sx] = True
        queue: list[tuple[float, int, int]] = [(float(merit_map[sy, sx]), sy, sx)]
        while queue:
            _, y, x = heapq.heappop(queue)
            for ny, nx in _iter_neighbors(y, x, height, width):
                if not valid[ny, nx] or connected_mask[ny, nx]:
                    continue
                connected[ny, nx] = _unwrap_relative_to_reference(float(wrapped_gap[ny, nx]), float(connected[y, x]))
                connected[ny, nx] += TAU * np.rint((raw_gap[ny, nx] - connected[ny, nx]) / TAU)
                connected_mask[ny, nx] = True
                heapq.heappush(queue, (float(merit_map[ny, nx]), ny, nx))

    confidence_map = _build_phase_gap_confidence(merit_map, amplitude_map, connected_mask)
    filled_gap, fit_gap, final_gap = _blend_observation_with_model(raw_gap, connected, connected_mask, confidence_map, theta_map)
    return {
        "phase_gap_final": final_gap,
        "phase_gap_connected": connected,
        "connected_mask": connected_mask,
        "confidence_map": confidence_map,
        "diagnostic_maps": {
            "method": "branch_cut",
            "phase_gap_filled": filled_gap,
            "phase_gap_fit": fit_gap,
            "phase_gap_residue": residue_map.astype(np.float32),
            "phase_gap_branch_cut": cut_mask.astype(np.float32),
        },
    }


def _collect_gradient_edges(
    wrapped_gap: np.ndarray,
    confidence_map: np.ndarray,
) -> tuple[list[tuple[int, int, float, float]], np.ndarray]:
    """收集有限像素之间的水平/垂直 wrapped-gradient 约束。"""
    finite = np.isfinite(wrapped_gap)
    index_map = np.full(wrapped_gap.shape, -1, dtype=np.int32)
    index_map[finite] = np.arange(int(np.count_nonzero(finite)), dtype=np.int32)
    edges: list[tuple[int, int, float, float]] = []
    height, width = wrapped_gap.shape
    for y in range(height):
        for x in range(width):
            current = int(index_map[y, x])
            if current < 0:
                continue
            for ny, nx in ((y, x + 1), (y + 1, x)):
                neighbor = int(index_map[ny, nx]) if ny < height and nx < width else -1
                if neighbor < 0:
                    continue
                delta = float(_wrap_to_pi(wrapped_gap[ny, nx] - wrapped_gap[y, x]))
                weight = float(max(confidence_map[y, x] * confidence_map[ny, nx], 1e-4))
                edges.append((current, neighbor, delta, weight))
    return edges, index_map


def _solve_weighted_gradient_field(
    raw_gap: np.ndarray,
    wrapped_gap: np.ndarray,
    confidence_map: np.ndarray,
    p_norm: float = 2.0,
    iterations: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """用稀疏 LS/IRLS 求连续 phase-gap 场。"""
    edges, index_map = _collect_gradient_edges(wrapped_gap, confidence_map)
    pixel_count = int(np.max(index_map)) + 1 if np.any(index_map >= 0) else 0
    if pixel_count == 0:
        return np.full_like(raw_gap, np.nan, dtype=np.float32), np.zeros_like(raw_gap, dtype=np.float32)

    solution = np.full(pixel_count, _global_circular_average(raw_gap), dtype=np.float64)
    edge_scale = np.ones(len(edges), dtype=np.float64)
    for _ in range(max(1, int(iterations))):
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        rhs: list[float] = []
        for row, (left, right, delta, base_weight) in enumerate(edges):
            weight = np.sqrt(base_weight * edge_scale[row])
            rows.extend([row, row])
            cols.extend([right, left])
            data.extend([weight, -weight])
            rhs.append(weight * delta)
        anchor_row = len(edges)
        anchor_coord = np.argwhere(index_map >= 0)[0]
        anchor_col = int(index_map[int(anchor_coord[0]), int(anchor_coord[1])])
        anchor_weight = 1e-2
        rows.append(anchor_row)
        cols.append(anchor_col)
        data.append(anchor_weight)
        rhs.append(anchor_weight * _global_circular_average(raw_gap))
        matrix = coo_matrix((data, (rows, cols)), shape=(len(rhs), pixel_count)).tocsr()
        solution = lsqr(matrix, np.asarray(rhs, dtype=np.float64), atol=1e-6, btol=1e-6, iter_lim=200)[0]
        if p_norm >= 1.999:
            break
        residuals = np.array([solution[right] - solution[left] - delta for left, right, delta, _ in edges])
        edge_scale = np.clip((np.abs(residuals) + 1e-3) ** (p_norm - 2.0), 0.05, 20.0)

    field = np.full(raw_gap.shape, np.nan, dtype=np.float32)
    field[index_map >= 0] = solution[index_map[index_map >= 0]].astype(np.float32)
    residual_map = np.zeros_like(raw_gap, dtype=np.float32)
    counts = np.zeros_like(raw_gap, dtype=np.float32)
    coords = np.argwhere(index_map >= 0)
    flat_to_coord = {int(index_map[y, x]): (int(y), int(x)) for y, x in coords}
    for left, right, delta, _ in edges:
        residual = abs(float(solution[right] - solution[left] - delta))
        for node in (left, right):
            y, x = flat_to_coord[node]
            residual_map[y, x] += residual
            counts[y, x] += 1.0
    residual_map = np.where(counts > 0, residual_map / np.maximum(counts, 1.0), 0.0).astype(np.float32)
    return field, residual_map


def _weighted_solver_phase_gap_solution(
    raw_gap: np.ndarray,
    wrapped_gap: np.ndarray,
    merit_map: np.ndarray,
    amplitude_map: np.ndarray,
    method: str,
) -> dict[str, np.ndarray | dict[str, np.ndarray | str]]:
    """生成 weighted LS 或 minimum Lp 的统一 strategy 输出。"""
    finite_raw = np.isfinite(raw_gap)
    confidence_map = _build_phase_gap_confidence(merit_map, amplitude_map, finite_raw)
    p_norm = 1.2 if method == "minimum_lp" else 2.0
    iterations = 5 if p_norm < 2.0 else 1
    field, residual_map = _solve_weighted_gradient_field(raw_gap, wrapped_gap, confidence_map, p_norm, iterations)
    return {
        "phase_gap_final": field.astype(np.float32),
        "phase_gap_connected": field.astype(np.float32),
        "connected_mask": np.isfinite(field),
        "confidence_map": confidence_map,
        "diagnostic_maps": {
            "method": method,
            "phase_gap_filled": field.astype(np.float32),
            "phase_gap_fit": field.astype(np.float32),
            "phase_gap_gradient_residual": residual_map,
        },
    }


def resolve_phase_gap(
    raw_gap: np.ndarray,
    wrapped_gap: np.ndarray,
    merit_map: np.ndarray,
    theta_map: np.ndarray,
    amplitude_map: np.ndarray,
    baseline_order: np.ndarray | None = None,
    method: str = "quality_guided",
) -> dict[str, np.ndarray | dict[str, np.ndarray | str]]:
    """
    统一 phase-gap finalization strategy 入口。

    所有方法都接收同一组物理/诊断输入，并返回固定字段；是否存在
    “连接”过程由具体 strategy 自己决定，主流程只消费统一输出。
    """
    del baseline_order
    method_name = str(method).strip().lower()
    method_aliases = {
        "constant": "circular_average",
        "constant_phase_gap": "circular_average",
        "weighted_ls": "weighted_least_squares",
        "minimum_l_p": "minimum_lp",
    }
    method_name = method_aliases.get(method_name, method_name)
    if method_name not in SUPPORTED_PHASE_GAP_METHODS:
        raise ValueError(f"Unsupported phase gap method: {method}")
    if method_name == "quality_guided":
        return _quality_guided_phase_gap_solution(raw_gap, wrapped_gap, merit_map, theta_map, amplitude_map)
    if method_name == "circular_average":
        return _constant_phase_gap_solution(raw_gap, merit_map, amplitude_map)
    if method_name == "robust_model_fit":
        return _robust_model_phase_gap_solution(raw_gap, merit_map, theta_map, amplitude_map)
    if method_name == "branch_cut":
        return _branch_cut_phase_gap_solution(raw_gap, wrapped_gap, merit_map, theta_map, amplitude_map)
    if method_name in {"weighted_least_squares", "minimum_lp"}:
        return _weighted_solver_phase_gap_solution(raw_gap, wrapped_gap, merit_map, amplitude_map, method_name)
    raise ValueError(f"Unsupported phase gap method: {method}")


def analyze_phase_gap_maps(
    baseline_result: dict[str, np.ndarray | float | int],
    smoothing_size: int = 3,
    phase_gap_method: str = "quality_guided",
) -> dict[str, object]:
    """
    执行完整的 phase-gap 核心处理。

    处理链条如下：
    1. 读取 baseline 提供的 `phase_gap_raw`、`theta_map`、`h_prime`
    2. 构造 wrapped gap
    3. 计算 merit 图
    4. 通过 `resolve_phase_gap` 选择 phase-gap finalization strategy
    5. 恢复 fringe order
    6. 得到最终 `h`
    """
    baseline_raw_gap = np.asarray(baseline_result["phase_gap_raw"], dtype=np.float32)
    h_prime = np.asarray(baseline_result["h_prime"], dtype=np.float32)
    h_coarse = np.asarray(baseline_result.get("h_coarse", h_prime), dtype=np.float32)
    # 优先使用 baseline 显式提供的 theta_map；没有时才回退到旧 proxy。
    # 这里保留回退逻辑只是为了兼容旧调用方，不代表它在物理上更优。
    theta_map = np.asarray(baseline_result.get("theta_map", h_coarse), dtype=np.float32)
    phi0 = np.asarray(baseline_result["phi0_map"], dtype=np.float32)
    peak_amplitude = np.asarray(
        baseline_result.get("peak_amplitude_map", np.ones_like(baseline_raw_gap)),
        dtype=np.float32,
    )
    k0_value = float(baseline_result["k0_value"])
    baseline_order_source = np.asarray(
        baseline_result.get("baseline_fringe_order_map", np.rint(baseline_raw_gap / TAU)),
        dtype=np.float32,
    )
    # baseline fringe order 允许输入里出现 NaN。
    # 这里先把它规整成整数图，避免后面在高度重建时把 NaN 直接 cast 成巨大的异常整数。
    baseline_fringe_order = np.where(
        np.isfinite(baseline_order_source),
        np.rint(baseline_order_source),
        0.0,
    ).astype(np.int32)

    smoothed_theta = smooth_coherence_profile(theta_map, size=smoothing_size, limit=np.pi / 2.0)
    # 严格按文献主线，experimental gap 由 phase 与平滑后的 coherence 共同构造。
    # 也就是说，真正参与 connect / fit / fringe-order 恢复的 gap
    # 不再是 baseline 直接给的 `phi0 - theta`，而是 `phi0 - theta_smoothed`。
    raw_gap = np.where(
        np.isfinite(phi0) & np.isfinite(smoothed_theta),
        phi0 - smoothed_theta,
        baseline_raw_gap,
    ).astype(np.float32)
    finite_raw = np.isfinite(raw_gap)
    wrapped_gap = _wrap_to_pi(raw_gap)
    merit_map = compute_merit_map(wrapped_gap)

    phase_gap_solution = resolve_phase_gap(
        raw_gap=raw_gap,
        wrapped_gap=wrapped_gap,
        merit_map=merit_map,
        theta_map=smoothed_theta,
        amplitude_map=peak_amplitude,
        baseline_order=baseline_fringe_order,
        method=phase_gap_method,
    )
    phase_gap_final = np.asarray(phase_gap_solution["phase_gap_final"], dtype=np.float32)
    connected_gap = np.asarray(phase_gap_solution["phase_gap_connected"], dtype=np.float32)
    connected_mask = np.asarray(phase_gap_solution["connected_mask"], dtype=bool)
    confidence_map = np.asarray(phase_gap_solution["confidence_map"], dtype=np.float32)
    diagnostic_maps = dict(phase_gap_solution["diagnostic_maps"])
    filled_gap = np.asarray(diagnostic_maps.get("phase_gap_filled", phase_gap_final), dtype=np.float32)
    fit_gap = np.asarray(diagnostic_maps.get("phase_gap_fit", phase_gap_final), dtype=np.float32)

    # 文献核心：用 raw gap 与 final gap 的差恢复“相对条纹级次修正”。
    # 注意这里恢复出来的是“要修正多少个 2π”，不是新的绝对级次。
    fringe_order_float = np.where(finite_raw, np.rint((raw_gap - phase_gap_final) / TAU), 0.0)
    fringe_order_map = fringe_order_float.astype(np.int32)
    # 最终高度仍然从 baseline 已有的级次起步，再叠加/扣除 phase-gap 给出的相对修正。
    # 如果这里错误地改成“直接用 rewrapped experimental order 作为新的绝对级次”，
    # 很容易把本来连续的倾斜面重新切成大面积平行条带。
    corrected_total_order = baseline_fringe_order - fringe_order_map
    corrected_phase = phi0 + TAU * corrected_total_order.astype(np.float32)
    # 最终 `h_prime` 与 baseline 高度链保持同一标尺，使用 `2*k0` 做相位到高度的换算。
    # 这里保留 `corrected_phase` 原值，便于诊断图继续直接展示最终相位场本身。
    h = (-(corrected_phase / (2.0 * k0_value)) * 1000.0).astype(np.float32)

    # 返回图层分两类：
    # - 中间诊断层（phase_gap_*、theta_map_smoothed、merit_map、confidence_map）
    # - 最终输出层（fringe_order_map、h）
    result: dict[str, object] = {
        "phase_gap_raw": raw_gap,
        "phase_gap_baseline_raw": baseline_raw_gap,
        "phase_gap_merit": merit_map.astype(np.float32),
        "merit_map": merit_map.astype(np.float32),
        "phase_gap_connected": connected_gap.astype(np.float32),
        "phase_gap_filled": filled_gap.astype(np.float32),
        "phase_gap_fit": fit_gap.astype(np.float32),
        "phase_gap_final": phase_gap_final,
        "phase_gap_method": str(diagnostic_maps.get("method", phase_gap_method)),
        "theta_map": theta_map.astype(np.float32),
        "theta_map_smoothed": smoothed_theta.astype(np.float32),
        "fringe_order_map": fringe_order_map,
        "confidence_map": confidence_map,
        # 这张图保留“最终高度对应的相位量纲”结果，供 GUI 直接显示
        # phase-gap 修正后的绝对相位分布，不必再从高度反推回相位。
        "final_height_phase_map": corrected_phase.astype(np.float32),
        "h": h,
    }
    for key, value in diagnostic_maps.items():
        if key not in result and isinstance(value, np.ndarray):
            result[key] = value.astype(np.float32)
    return result
