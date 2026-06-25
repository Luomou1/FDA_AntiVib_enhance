from __future__ import annotations

"""平面与台阶高度矩阵的独立后处理算法。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


@dataclass(frozen=True)
class PlaneAnalysisResult:
    """平面分析输出。"""

    original: np.ndarray
    calibrated: np.ndarray
    processed: np.ndarray
    fitted_surface: np.ndarray
    coefficients: np.ndarray
    method: str
    outlier_count: int
    noise_count: int
    stats: dict[str, float]


@dataclass(frozen=True)
class StepAnalysisResult:
    """台阶分析输出。"""

    original: np.ndarray
    leveled: np.ndarray
    processed: np.ndarray
    fitted_surface: np.ndarray
    cluster_map: np.ndarray
    coefficients: np.ndarray
    threshold: float
    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]
    denoised: bool
    noise_count: int
    layer_stats: dict[str, dict[str, float]]


@dataclass(frozen=True)
class StepMeasurement:
    """两个矩形区域的台阶高度测量结果。"""

    region_one_mean: float
    region_two_mean: float
    step_height: float
    region_one: tuple[int, int, int, int]
    region_two: tuple[int, int, int, int]


def _as_height_matrix(data: np.ndarray | Iterable[Iterable[float]]) -> np.ndarray:
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError("高度数据必须是至少 2×2 的二维矩阵。")
    if np.isinf(matrix).any():
        raise ValueError("高度数据包含无穷值，请先检查源文件。")
    return matrix.copy()


def fill_nan_local_median(data: np.ndarray) -> np.ndarray:
    """用逐级扩大的局部中值修复 NaN，与 MATLAB 程序的 5/9/13 窗口一致。"""
    matrix = _as_height_matrix(data)
    nan_positions = np.argwhere(np.isnan(matrix))
    if nan_positions.size == 0:
        return matrix

    finite_values = matrix[np.isfinite(matrix)]
    if finite_values.size == 0:
        raise ValueError("高度数据全部为 NaN，无法分析。")
    global_median = float(np.median(finite_values))
    rows, cols = matrix.shape

    for row, col in nan_positions:
        replacement: float | None = None
        for radius in (2, 4, 6):
            row_start = max(0, int(row) - radius)
            row_stop = min(rows, int(row) + radius + 1)
            col_start = max(0, int(col) - radius)
            col_stop = min(cols, int(col) + radius + 1)
            neighborhood = matrix[row_start:row_stop, col_start:col_stop]
            valid = neighborhood[np.isfinite(neighborhood)]
            if valid.size:
                replacement = float(np.median(valid))
                break
        matrix[row, col] = global_median if replacement is None else replacement
    return matrix


def load_height_matrix(path: str | Path, conversion_factor: float = 1.0) -> np.ndarray:
    """读取空白分隔的二维高度文本矩阵并转换到纳米单位。"""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"高度文件不存在：{source}")
    try:
        matrix = np.loadtxt(source, dtype=np.float64)
    except (OSError, ValueError) as exc:
        raise ValueError(f"无法读取高度文件：{exc}") from exc
    if not np.isfinite(conversion_factor) or conversion_factor == 0:
        raise ValueError("高度换算系数必须是非零有限值。")
    return fill_nan_local_median(_as_height_matrix(matrix) * float(conversion_factor))


def _coordinate_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = shape
    y, x = np.mgrid[0:rows, 0:cols]
    return x.astype(np.float64), y.astype(np.float64)


def _mad(values: np.ndarray) -> float:
    flattened = np.asarray(values, dtype=np.float64).ravel()
    median = float(np.median(flattened))
    return float(np.median(np.abs(flattened - median)))


def _fit_plane(data: np.ndarray, robust: bool = False) -> tuple[np.ndarray, np.ndarray]:
    x, y = _coordinate_grid(data.shape)
    design = np.column_stack((x.ravel(), y.ravel(), np.ones(data.size)))
    values = data.ravel()
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)

    if robust:
        residuals = values - design @ coefficients
        mad_value = _mad(residuals)
        threshold = max(2.5 * mad_value, np.finfo(float).eps)
        inliers = np.abs(residuals - np.median(residuals)) <= threshold
        if np.count_nonzero(inliers) > values.size * 0.5:
            coefficients, *_ = np.linalg.lstsq(design[inliers], values[inliers], rcond=None)

    surface = coefficients[0] * x + coefficients[1] * y + coefficients[2]
    return coefficients.astype(np.float64), surface


def _fit_quadratic(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y = _coordinate_grid(data.shape)
    design = np.column_stack(
        (
            x.ravel() ** 2,
            y.ravel() ** 2,
            (x * y).ravel(),
            x.ravel(),
            y.ravel(),
            np.ones(data.size),
        )
    )
    coefficients, *_ = np.linalg.lstsq(design, data.ravel(), rcond=None)
    surface = (
        coefficients[0] * x**2
        + coefficients[1] * y**2
        + coefficients[2] * x * y
        + coefficients[3] * x
        + coefficients[4] * y
        + coefficients[5]
    )
    return coefficients.astype(np.float64), surface


def _component_window(area: int, sizes: tuple[int, int, int]) -> int:
    if area <= 4:
        return sizes[0]
    if area <= 16:
        return sizes[1]
    return sizes[2]


def _repair_masked_values(
    data: np.ndarray,
    mask: np.ndarray,
    *,
    valid_domain: np.ndarray | None = None,
    fallback: np.ndarray | float | None = None,
    window_sizes: tuple[int, int, int] = (3, 5, 7),
    large_area: int = 30,
    gaussian_sigma: float = 1.0,
    connectivity: int = 1,
    gaussian_kernel: np.ndarray | None = None,
    large_margin: int | None = None,
    defer_large_filter: bool = False,
    column_major_components: bool = False,
) -> np.ndarray:
    """按连通区域大小选择窗口，只使用当前有效域内的非异常邻点修复。"""
    repaired = np.asarray(data, dtype=np.float64).copy()
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return repaired
    domain = np.ones_like(mask, dtype=bool) if valid_domain is None else np.asarray(valid_domain, dtype=bool)
    structure = ndimage.generate_binary_structure(2, connectivity)
    labels, component_count = ndimage.label(mask, structure=structure)
    rows, cols = data.shape
    component_slices = ndimage.find_objects(labels)
    components: list[np.ndarray] = []
    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        row_slice, col_slice = component_slice
        local_positions = np.argwhere(labels[component_slice] == component_id)
        if local_positions.size == 0:
            continue
        local_positions[:, 0] += row_slice.start
        local_positions[:, 1] += col_slice.start
        components.append(local_positions)
    if column_major_components:
        components.sort(key=lambda positions: int(np.min(positions[:, 0] + positions[:, 1] * rows)))

    def filter_large_component(positions: np.ndarray, radius: int) -> None:
        nonlocal repaired
        area = int(positions.shape[0])
        if area <= large_area:
            return
        margin = radius if large_margin is None else large_margin
        row_start = max(0, int(np.min(positions[:, 0])) - margin)
        row_stop = min(rows, int(np.max(positions[:, 0])) + margin + 1)
        col_start = max(0, int(np.min(positions[:, 1])) - margin)
        col_stop = min(cols, int(np.max(positions[:, 1])) + margin + 1)
        region = repaired[row_start:row_stop, col_start:col_stop]
        if gaussian_kernel is None:
            filtered = ndimage.gaussian_filter(region, sigma=gaussian_sigma, mode="nearest")
        else:
            filtered = ndimage.convolve(region, gaussian_kernel, mode="nearest")
        region_domain = domain[row_start:row_stop, col_start:col_stop]
        repaired[row_start:row_stop, col_start:col_stop] = np.where(
            region_domain,
            filtered,
            region,
        )

    for positions in components:
        area = int(positions.shape[0])
        window_size = _component_window(area, window_sizes)
        radius = window_size // 2
        for row, col in positions:
            row_start = max(0, int(row) - radius)
            row_stop = min(rows, int(row) + radius + 1)
            col_start = max(0, int(col) - radius)
            col_stop = min(cols, int(col) + radius + 1)
            local_data = data[row_start:row_stop, col_start:col_stop]
            local_valid = domain[row_start:row_stop, col_start:col_stop] & ~mask[
                row_start:row_stop, col_start:col_stop
            ]
            values = local_data[local_valid]
            if values.size:
                repaired[row, col] = float(np.median(values))
            elif isinstance(fallback, np.ndarray):
                repaired[row, col] = float(fallback[row, col])
            elif fallback is not None:
                repaired[row, col] = float(fallback)
            else:
                domain_values = data[domain & ~mask]
                repaired[row, col] = float(np.median(domain_values)) if domain_values.size else 0.0

        if not defer_large_filter:
            filter_large_component(positions, radius)

    if defer_large_filter:
        for positions in components:
            area = int(positions.shape[0])
            radius = _component_window(area, window_sizes) // 2
            filter_large_component(positions, radius)
    return repaired


def _surface_stats(data: np.ndarray) -> dict[str, float]:
    values = np.asarray(data, dtype=np.float64)
    mean = float(np.mean(values))
    return {
        "height_range": float(np.max(values) - np.min(values)),
        "maximum": float(np.max(values)),
        "minimum": float(np.min(values)),
        "mean": mean,
        "std": float(np.std(values)),
        "rms": float(np.sqrt(np.mean((values - mean) ** 2))),
    }


def analyze_plane(data: np.ndarray, method: str = "simple") -> PlaneAnalysisResult:
    """执行单个平面高度矩阵的异常点修复、表面校准和残余去噪。"""
    matrix = fill_nan_local_median(data)

    initial_coefficients, initial_surface = _fit_plane(matrix)
    residuals = matrix - initial_surface
    mad_value = _mad(residuals)
    outlier_threshold = max(8.0 * mad_value, np.finfo(float).eps)
    outlier_mask = np.abs(residuals - np.median(residuals)) > outlier_threshold
    cleaned = _repair_masked_values(
        matrix,
        outlier_mask,
        fallback=initial_surface,
        window_sizes=(5, 7, 9),
        large_area=30,
        gaussian_sigma=2.0,
    )

    normalized_method = method.strip().lower()
    if normalized_method == "simple":
        coefficients, fitted_surface = _fit_plane(cleaned)
    elif normalized_method == "robust":
        coefficients, fitted_surface = _fit_plane(cleaned, robust=True)
    elif normalized_method == "quadratic":
        coefficients, fitted_surface = _fit_quadratic(cleaned)
    else:
        raise ValueError(f"未知平面校准方法：{method}")

    calibrated = cleaned - fitted_surface
    calibrated_mean = float(np.mean(calibrated))
    calibrated_std = float(np.std(calibrated))
    if calibrated_std <= np.finfo(float).eps:
        noise_mask = np.zeros_like(calibrated, dtype=bool)
    else:
        noise_mask = np.abs(calibrated - calibrated_mean) > 5.0 * calibrated_std
    denoised = _repair_masked_values(
        calibrated,
        noise_mask,
        fallback=calibrated_mean,
        window_sizes=(3, 5, 7),
        large_area=30,
        gaussian_sigma=1.5,
    )

    # MATLAB 程序在去倾斜后恢复原始平均高度；这样校准只改变形貌，不改变绝对高度基准。
    denoised += float(np.mean(cleaned) - np.mean(denoised))
    return PlaneAnalysisResult(
        original=cleaned,
        calibrated=calibrated,
        processed=denoised,
        fitted_surface=fitted_surface,
        coefficients=coefficients,
        method=normalized_method,
        outlier_count=int(np.count_nonzero(outlier_mask)),
        noise_count=int(np.count_nonzero(noise_mask)),
        stats=_surface_stats(denoised),
    )


def _normalize_points(
    points: Iterable[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    normalized = tuple((int(row), int(col)) for row, col in points)
    if len(normalized) != 3:
        raise ValueError("三点调平必须选择 3 个参考点。")
    rows, cols = shape
    if any(row < 0 or row >= rows or col < 0 or col >= cols for row, col in normalized):
        raise ValueError("三点调平参考点超出高度矩阵范围。")
    return normalized  # type: ignore[return-value]


def _three_point_level(
    data: np.ndarray,
    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 点采用 (row, col) 零基索引；平面方程仍按 x=col、y=row 构造。
    design = np.array([[col, row, 1.0] for row, col in points], dtype=np.float64)
    heights = np.array([data[row, col] for row, col in points], dtype=np.float64)
    if abs(float(np.linalg.det(design))) < 1e-12:
        raise ValueError("三点调平参考点退化，请重新选择三个不共线的点。")
    raw_coefficients = np.linalg.solve(design, heights)
    x, y = _coordinate_grid(data.shape)
    raw_surface = raw_coefficients[0] * x + raw_coefficients[1] * y + raw_coefficients[2]
    fitted_surface = raw_surface - float(np.mean(raw_surface))
    coefficients = raw_coefficients.copy()
    coefficients[2] -= float(np.mean(raw_surface))
    return coefficients, fitted_surface, data - fitted_surface


def _disk(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return x * x + y * y <= radius * radius


def _matlab_hist(values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """复刻 MATLAB legacy hist：分箱边界上的值归入左侧箱。"""
    flattened = np.asarray(values, dtype=np.float64).ravel()
    minimum = float(np.min(flattened))
    maximum = float(np.max(flattened))
    if minimum == maximum:
        centers = np.full(bins, minimum, dtype=np.float64)
        counts = np.zeros(bins, dtype=np.float64)
        counts[bins // 2] = flattened.size
        return counts, centers

    edges = np.linspace(minimum, maximum, bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    assignments = np.searchsorted(edges[1:-1], flattened, side="left")
    counts = np.bincount(assignments, minlength=bins).astype(np.float64)
    return counts, centers


def _matlab_smooth_three(values: np.ndarray) -> np.ndarray:
    """复刻 smooth(values, 3) 的三点移动平均及端点行为。"""
    source = np.asarray(values, dtype=np.float64)
    smoothed = source.copy()
    if source.size >= 3:
        smoothed[1:-1] = (source[:-2] + source[1:-1] + source[2:]) / 3.0
    return smoothed


def _sample_std(values: np.ndarray) -> float:
    source = np.asarray(values, dtype=np.float64)
    if source.size <= 1:
        return 0.0
    return float(np.std(source, ddof=1))


def _matlab_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    radius = (size - 1) / 2.0
    coordinates = np.arange(size, dtype=np.float64) - radius
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    kernel = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    return kernel / np.sum(kernel)


def _matlab_binary_open_close(mask: np.ndarray, structure: np.ndarray) -> np.ndarray:
    """复刻 imclose(imopen(mask, se), se) 的扩展域边界行为。"""
    opened = ndimage.binary_erosion(mask, structure, border_value=1)
    opened = ndimage.binary_dilation(opened, structure, border_value=0)

    row_margin = structure.shape[0] // 2
    col_margin = structure.shape[1] // 2
    padded = np.pad(
        opened,
        ((row_margin, row_margin), (col_margin, col_margin)),
        mode="constant",
        constant_values=False,
    )
    closed = ndimage.binary_dilation(padded, structure, border_value=0)
    closed = ndimage.binary_erosion(closed, structure, border_value=0)
    return closed[
        row_margin : row_margin + mask.shape[0],
        col_margin : col_margin + mask.shape[1],
    ]


def _segment_step(data: np.ndarray) -> tuple[np.ndarray, float]:
    values = data.ravel()
    counts, centers = _matlab_hist(values, bins=50)
    smoothed = _matlab_smooth_three(counts)
    minimum_height = float(np.max(smoothed) * 0.05) if smoothed.size else 0.0
    peak_indices, properties = find_peaks(smoothed, height=minimum_height, distance=5)

    if peak_indices.size >= 2:
        order = np.argsort(properties["peak_heights"])[::-1][:2]
        selected = peak_indices[order]
        selected_heights = properties["peak_heights"][order]
        height_one, height_two = centers[selected]
        peak_one, peak_two = selected_heights
        if height_one < height_two:
            height_one, height_two = height_two, height_one
        threshold = float((height_one * peak_one + height_two * peak_two) / (peak_one + peak_two))
        if abs(float(height_one - height_two)) < float(np.ptp(values)) * 0.1:
            threshold = float(np.mean(values))
    else:
        threshold = float((np.percentile(values, 30) + np.percentile(values, 70)) / 2.0)

    high_mask = data > threshold
    low_mask = ~high_mask
    total = data.size
    if np.count_nonzero(high_mask) / total < 0.05 or np.count_nonzero(low_mask) / total < 0.05:
        threshold = float(np.median(values))
        high_mask = data > threshold
        low_mask = ~high_mask

    structure = _disk(2)
    high_cleaned = _matlab_binary_open_close(high_mask, structure)
    low_cleaned = _matlab_binary_open_close(low_mask, structure)
    overlap = high_cleaned & low_cleaned
    low_cleaned &= ~overlap
    unassigned = ~(high_cleaned | low_cleaned)
    high_cleaned[unassigned & (data > threshold)] = True
    low_cleaned[unassigned & (data <= threshold)] = True

    if not np.any(high_cleaned) or not np.any(low_cleaned):
        high_cleaned = data > threshold
        low_cleaned = ~high_cleaned

    cluster_map = np.ones(data.shape, dtype=np.uint8)
    cluster_map[high_cleaned] = 2
    cluster_map[low_cleaned] = 1
    return cluster_map, threshold


def _layer_statistics(data: np.ndarray, cluster_map: np.ndarray) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for layer, name in ((1, "low"), (2, "high")):
        values = data[cluster_map == layer]
        if values.size == 0:
            result[name] = {"flatness": float("nan"), "std": float("nan"), "area_ratio": 0.0}
            continue
        result[name] = {
            "flatness": float(np.max(values) - np.min(values)),
            "std": _sample_std(values),
            "area_ratio": float(values.size / data.size),
        }
    return result


def _denoise_step_layers(data: np.ndarray, cluster_map: np.ndarray) -> tuple[np.ndarray, int]:
    processed = data.copy()
    total_noise = 0
    for layer in (1, 2):
        layer_mask = cluster_map == layer
        layer_values = data[layer_mask]
        if layer_values.size == 0:
            continue
        layer_mean = float(np.mean(layer_values))
        layer_std = _sample_std(layer_values)
        if layer_std <= np.finfo(float).eps:
            continue
        noise_mask = layer_mask & (np.abs(data - layer_mean) > 3.0 * layer_std)
        total_noise += int(np.count_nonzero(noise_mask))
        processed = _repair_masked_values(
            processed,
            noise_mask,
            valid_domain=layer_mask,
            fallback=layer_mean,
            window_sizes=(3, 5, 7),
            large_area=20,
            gaussian_sigma=0.8,
            connectivity=2,
            gaussian_kernel=_matlab_gaussian_kernel(3, 0.8),
            large_margin=1,
            defer_large_filter=True,
            column_major_components=True,
        )
    return processed, total_noise


def analyze_step(
    data: np.ndarray,
    *,
    points: Iterable[tuple[int, int]],
    denoise: bool,
) -> StepAnalysisResult:
    """执行三点调平、双层分割，并按选择决定是否进行逐层去噪。"""
    matrix = fill_nan_local_median(data)
    normalized_points = _normalize_points(points, matrix.shape)
    coefficients, fitted_surface, leveled = _three_point_level(matrix, normalized_points)
    cluster_map, threshold = _segment_step(leveled)
    if denoise:
        processed, noise_count = _denoise_step_layers(leveled, cluster_map)
    else:
        processed = leveled.copy()
        noise_count = 0
    processed += float(np.mean(matrix) - np.mean(processed))
    layer_stats = _layer_statistics(processed, cluster_map)

    return StepAnalysisResult(
        original=matrix,
        leveled=leveled,
        processed=processed,
        fitted_surface=fitted_surface,
        cluster_map=cluster_map,
        coefficients=coefficients,
        threshold=threshold,
        points=normalized_points,
        denoised=bool(denoise),
        noise_count=noise_count,
        layer_stats=layer_stats,
    )


def _clip_region(
    region: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (int(round(value)) for value in region)
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    rows, cols = shape
    left = max(0, min(left, cols - 1))
    right = max(left + 1, min(right, cols))
    top = max(0, min(top, rows - 1))
    bottom = max(top + 1, min(bottom, rows))
    return left, top, right, bottom


def compute_step_height(
    data: np.ndarray,
    region_one: tuple[int, int, int, int],
    region_two: tuple[int, int, int, int],
) -> StepMeasurement:
    """计算两个矩形区域的均值差，矩形格式为 (x0, y0, x1, y1)。"""
    matrix = _as_height_matrix(data)
    clipped_one = _clip_region(region_one, matrix.shape)
    clipped_two = _clip_region(region_two, matrix.shape)

    def region_mean(region: tuple[int, int, int, int]) -> float:
        left, top, right, bottom = region
        values = matrix[top:bottom, left:right]
        if values.size == 0:
            raise ValueError("矩形区域为空，请重新框选。")
        return float(np.mean(values))

    mean_one = region_mean(clipped_one)
    mean_two = region_mean(clipped_two)
    return StepMeasurement(
        region_one_mean=mean_one,
        region_two_mean=mean_two,
        step_height=mean_one - mean_two,
        region_one=clipped_one,
        region_two=clipped_two,
    )
