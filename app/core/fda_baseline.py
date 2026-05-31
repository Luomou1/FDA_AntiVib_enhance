from __future__ import annotations
"""
FDA 基线分析模块。

这个模块负责完成两件事：
1. 从逐像素扫描信号中提取 `g0` 和 `phi0`；
2. 根据 `phi0 + 2πN` 估计 FDA 修正高度 `h_prime`。

其中 `theta_map`、`phase_gap_raw` 和 `baseline_fringe_order_map` 仍然保留为
FDA 级次估计的中间量，便于排查 `h_prime` 的整周期选择是否合理：
   - `theta_map`
   - `phase_gap_raw`
   - `baseline_fringe_order_map`
"""

import time

import numpy as np

from app.core.kernel import (
    _build_k_axis,
    _compute_windowed_spectrum,
    _custom_phase_unwrap_batch,
    _fit_quadratic_batch,
    _fit_simple_batch,
    _fit_weighted_batch,
    _prepare_local_phase_window,
    _unwrap_phase_itoh,
    _remove_dc_batch,
    _resolve_block_rows,
    estimate_global_k0,
    resolve_fft_length,
)


def build_phase_gap_baseline_maps(
    phi0_map: np.ndarray,
    g0_map: np.ndarray,
    k0_value: float,
) -> dict[str, np.ndarray]:
    """
    根据 FDA 基线拟合结果构造级次估计所需的基础图层。

    参数说明：
    - `phi0_map`：在名义波数 `k0` 处的相位图
    - `g0_map`：相位对波数的一阶导近似，来自局部拟合
    - `k0_value`：当前分析使用的名义波数

    返回值说明：
    - `theta_map`：与相位同单位的相干等效相位图
    - `phase_gap_raw`：保留下来的相位差诊断量，当前实现取 `phi0 - theta`
    - `baseline_fringe_order_map`：用相位差直接得到的 FDA 整数级次

    这里的 `theta_map` 只服务于 FDA 级次估计和诊断，不再进入其他后处理。
    """
    phi0 = np.asarray(phi0_map, dtype=np.float32)
    g0 = np.asarray(g0_map, dtype=np.float32)
    # `g0` 是相位对波数的局部斜率。当前项目把它乘以 `k0`
    # 映射到“与相位同单位”的 coherence-equivalent profile，
    # 后续级次判断统一在“弧度”这个量纲下进行。
    theta_map = (float(k0_value) * g0).astype(np.float32)
    # baseline 原始 phase gap 直接定义为 `phi0 - theta`。
    # 注意这里不再接全场 connect/fill，只作为 FDA 阶段
    # 判断 `phi0` 需要补几个 2π 周期的中间量。
    phase_gap_raw = (phi0 - theta_map).astype(np.float32)
    # 这里保持与项目原先更稳定的 fringe-order 公式一致：
    # 条纹级次来自 `theta - phi0` 的整周期差，而不是 `phi0 - theta`。
    # 这个符号方向在真实倾斜面上非常关键，反过来会直接导致整场条带。
    baseline_fringe_order_map = np.rint((theta_map - phi0) / (2.0 * np.pi)).astype(np.float32)
    return {
        "theta_map": theta_map,
        "phase_gap_raw": phase_gap_raw,
        "baseline_fringe_order_map": baseline_fringe_order_map,
    }


def _compute_baseline_block(
    block: np.ndarray,
    step_size: float,
    window_size: int,
    fitting_method: str,
    unwrap_method: str,
    global_peak_index: int,
    fixed_k0_value: float,
    sample_positions_um: np.ndarray | None = None,
    window_name: str = "hamming",
    window_alpha: float = 0.5,
    fft_length: int | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    对一个分块图像做 FDA 基线分析。

    分块处理的目的不是改变算法，而是降低内存占用，让大尺寸数据也能跑。
    返回值里既包含最终的 `h_prime`，也包含 FDA 级次估计的中间图。
    """
    # `block` 维度约定：
    # - 第 0 维：行（当前分块内的行号）
    # - 第 1 维：列（图像宽度方向）
    # - 第 2 维：逐像素扫描采样点（z 扫描或时间采样）
    row_count, width, sample_count = block.shape
    pixel_count = row_count * width
    # 拉平成 (pixel_count, sample_count) 便于批处理 FFT、unwrap、拟合。
    curves = block.reshape(pixel_count, sample_count)
    # 去直流后，主频峰通常更容易稳定定位，减少低频泄漏对局部拟合的干扰。
    curves = _remove_dc_batch(curves)

    # 对每个像素的扫描曲线做频谱分析，得到幅值、相位和对应的 k 轴。
    # 这一步仍然只是“逐像素局部信息”，不会再进入全场非 FDA 约束。
    amplitude, phase, k = _compute_windowed_spectrum(
        curves,
        step_size=step_size,
        sample_positions_um=sample_positions_um,
        window_name=window_name,
        window_alpha=window_alpha,
        fft_length=fft_length,
    )

    # 只截取 k0 附近的小窗口做局部拟合。
    # 这样做的含义是：我们不试图解释整段频谱，只关心名义波数 k0
    # 附近那一小段最能代表“相位截距 phi0”和“局部斜率 g0”的数据。
    # 在全局峰值索引两侧构建对称窗口，窗口总长度为 2*window_size+1。
    offsets = np.arange(-window_size, window_size + 1, dtype=np.int32)
    idx_range_row = np.clip(global_peak_index + offsets, 0, k.size - 1)
    # 为每个像素复制同一组索引窗口，实现完全向量化的批量取样。
    idx_range = np.broadcast_to(idx_range_row, (pixel_count, idx_range_row.shape[0]))
    pixel_index = np.arange(pixel_count)[:, None]

    k_masked = k[idx_range].astype(np.float64)
    amplitude_masked = amplitude[pixel_index, idx_range].astype(np.float64)
    # 峰值幅值图常用于后续质量评估或置信度融合。
    peak_amplitude = amplitude[pixel_index, global_peak_index].astype(np.float64)
    k_center = np.full(pixel_count, float(fixed_k0_value), dtype=np.float64)

    # 这里的 unwrap 只服务于局部拟合窗口，不等同于最终表面上的 2D unwrap。
    # baseline 的目标是稳定估出 `phi0` 和 `g0`，再由当前 FDA 公式完成整数级次修正。
    if unwrap_method == "global":
        phi_unwrapped = np.unwrap(phase, axis=1)
        phi_masked = phi_unwrapped[pixel_index, idx_range].astype(np.float64)
    elif unwrap_method == "itoh":
        phi_masked = _unwrap_phase_itoh(phase[pixel_index, idx_range]).astype(np.float64)
    elif unwrap_method in {"local", "gr", "pda", "branch_search"}:
        phase_window = phase[pixel_index, idx_range].astype(np.float64)
        phi_masked = np.empty_like(phase_window, dtype=np.float64)
        for idx in range(pixel_count):
            local_unwrapped, _, _, _ = _prepare_local_phase_window(
                k_masked=k_masked[idx],
                phase_wrapped=phase_window[idx],
                amplitude_masked=amplitude_masked[idx],
                k_center=float(k_center[idx]),
                unwrap_method=unwrap_method,
                fitting_method=fitting_method,
            )
            phi_masked[idx] = local_unwrapped.astype(np.float64)
    else:
        raise ValueError(f"Unsupported unwrap method: {unwrap_method}")
    # 统一输出两类局部参数：
    # - `g0`: 局部斜率（dphi/dk）
    # - `phi0`: 在 k0 处的相位截距
    if fitting_method == "simple":
        g0, phi0 = _fit_simple_batch(k_masked, phi_masked, k_center)
    elif fitting_method == "quadratic":
        g0, phi0 = _fit_quadratic_batch(k_masked, phi_masked, k_center)
    elif fitting_method == "weighted":
        g0, phi0 = _fit_weighted_batch(k_masked, phi_masked, amplitude_masked, k_center)
    else:
        raise ValueError(f"Unsupported fitting method: {fitting_method}")

    # `h_coarse = -g0/2` 是最粗的 slope height，
    # 只反映相位-波数斜率对应的连续高度，不做条纹级次修正。
    # 由 FDA 基线近似可得：h_coarse = -g0/2（单位通常为微米量纲下的等效高度）。
    h_coarse = -g0 / 2.0
    baseline_maps = build_phase_gap_baseline_maps(
        phi0_map=phi0.reshape(row_count, width),
        g0_map=g0.reshape(row_count, width),
        k0_value=fixed_k0_value,
    )
    theta_map = baseline_maps["theta_map"].reshape(pixel_count)
    phase_gap_raw = baseline_maps["phase_gap_raw"].reshape(pixel_count)
    fringe_order = baseline_maps["baseline_fringe_order_map"].reshape(pixel_count)
    # `h_prime` 是 baseline/FDA 阶段给出的“已带基线级次”的高度。
    # 它通常比 `h_coarse` 更接近真实高度，但仍只使用逐像素信息；
    # 在大斜率、衍射尖峰、coherence-phase mismatch 明显时仍可能出现整周期错误。
    # h_prime 把相位截距 phi0 与基线条纹级次 N 合并，形式为
    # h' = -(phi0 + 2πN) / (2k0)。
    h_prime = np.full_like(h_coarse, np.nan)
    valid_center = np.abs(k_center) > 1e-12
    h_prime[valid_center] = -(1.0 / (2.0 * k_center[valid_center])) * (
        phi0[valid_center] + 2.0 * np.pi * fringe_order[valid_center]
    )

    # 返回顺序固定，调用方按该顺序写回全场缓存。
    # 高度类输出统一乘以 1000 转成 nm（与当前工程其余模块约定一致）。
    return (
        (h_coarse.reshape(row_count, width) * 1000.0).astype(np.float32),
        (h_prime.reshape(row_count, width) * 1000.0).astype(np.float32),
        phi0.reshape(row_count, width).astype(np.float32),
        g0.reshape(row_count, width).astype(np.float32),
        phase_gap_raw.reshape(row_count, width).astype(np.float32),
        fringe_order.reshape(row_count, width).astype(np.float32),
        theta_map.reshape(row_count, width).astype(np.float32),
        peak_amplitude.reshape(row_count, width).astype(np.float32),
    )


def analyze_cube_baseline(
    intensity_data: np.ndarray,
    step_size: float,
    window_size: int,
    fitting_method: str,
    unwrap_method: str,
    fixed_k0_value: float | None = None,
    sample_positions_um: np.ndarray | None = None,
    window_name: str = "hamming",
    window_alpha: float = 0.5,
    zero_padding_mode: str = "next_power_of_two",
    show_progress: bool = False,
    progress_callback=None,
) -> dict[str, np.ndarray]:
    """
    对完整数据立方体执行 FDA 基线分析。

    这一步的职责很明确：
    - 求出 `h_prime`
    - 求出 `theta_map`
    - 求出用于级次判断的相位差诊断量

    它不执行 connect/fill 等非 FDA 后处理。
    """
    # 统一输入类型，避免后续混入 float64 导致额外内存和拷贝开销。
    intensity_data = np.asarray(intensity_data, dtype=np.float32)
    height, width, _ = intensity_data.shape
    # 预分配输出图层，按“整图尺寸”一次性分配，块内结果再逐块写回。
    coarse_map = np.full((height, width), np.nan, dtype=np.float32)
    height_map_prime = np.full((height, width), np.nan, dtype=np.float32)
    phi0_map = np.full((height, width), np.nan, dtype=np.float32)
    g0_map = np.full((height, width), np.nan, dtype=np.float32)
    phase_gap_raw = np.full((height, width), np.nan, dtype=np.float32)
    fringe_order = np.full((height, width), np.nan, dtype=np.float32)
    theta_map = np.full((height, width), np.nan, dtype=np.float32)
    peak_amplitude = np.full((height, width), np.nan, dtype=np.float32)
    start_time = time.time()
    fft_length = resolve_fft_length(intensity_data.shape[2], zero_padding_mode)

    # 构建全局统一的 k 轴：后续所有像素都使用同一个峰值索引窗口。
    k_axis = _build_k_axis(
        sample_count=intensity_data.shape[2],
        step_size=step_size,
        sample_positions_um=sample_positions_um,
        fft_length=fft_length,
    )
    if fixed_k0_value is None:
        # 如果用户没有给定 K0，就先自动估计一个全局 K0。
        estimated = estimate_global_k0(
            intensity_data=intensity_data,
            step_size=step_size,
            candidate_ratio=1.0,
            window_name=window_name,
            window_alpha=window_alpha,
            zero_padding_mode=zero_padding_mode,
            sample_positions_um=sample_positions_um,
        )
        global_k0 = float(estimated["k0_value"])
    else:
        global_k0 = float(fixed_k0_value)
    # 把连续值 k0 映射到离散频谱索引，用于局部拟合窗口定位。
    global_peak_index = int(np.argmin(np.abs(k_axis - global_k0)))

    # 自动选择分块行数：在大图上节省内存，在小图上减少分块开销。
    block_rows = _resolve_block_rows(height, width)
    if show_progress:
        print(
            f"[Python] Starting baseline analysis: shape={intensity_data.shape}, "
            f"fitting={fitting_method}, unwrap={unwrap_method}, "
            f"block_rows={block_rows}, fft_length={fft_length}, global_k0={global_k0:.6f}",
            flush=True,
        )

    for start_row in range(0, height, block_rows):
        end_row = min(height, start_row + block_rows)
        block = intensity_data[start_row:end_row, :, :]
        # 每个分块独立完成 baseline 分析，再写回全场图层。
        (
            block_coarse,
            block_h_prime,
            block_phi0,
            block_g0,
            block_phase_gap_raw,
            block_fringe_order,
            block_theta_map,
            block_peak_amplitude,
        ) = _compute_baseline_block(
            block=block,
            step_size=step_size,
            window_size=int(window_size),
            fitting_method=fitting_method,
            unwrap_method=unwrap_method,
            global_peak_index=global_peak_index,
            fixed_k0_value=global_k0,
            sample_positions_um=sample_positions_um,
            window_name=window_name,
            window_alpha=window_alpha,
            fft_length=fft_length,
        )
        coarse_map[start_row:end_row, :] = block_coarse
        height_map_prime[start_row:end_row, :] = block_h_prime
        phi0_map[start_row:end_row, :] = block_phi0
        g0_map[start_row:end_row, :] = block_g0
        phase_gap_raw[start_row:end_row, :] = block_phase_gap_raw
        fringe_order[start_row:end_row, :] = block_fringe_order
        theta_map[start_row:end_row, :] = block_theta_map
        peak_amplitude[start_row:end_row, :] = block_peak_amplitude

        # 进度按照“已完成行数/总行数”计算，便于 GUI 进度条直接消费。
        percent = int(end_row / height * 100.0)
        if progress_callback is not None:
            progress_callback(percent)
        if show_progress:
            elapsed = time.time() - start_time
            print(
                f"[Python] Baseline progress: {end_row}/{height} rows ({percent:.1f}%), elapsed {elapsed:.1f}s",
                flush=True,
            )

    if show_progress:
        print(f"[Python] Baseline analysis complete in {time.time() - start_time:.1f}s", flush=True)

    # 返回中同时保留新旧字段名（例如 h_prime 与 heightMap_prime），
    # 目的是兼容项目中不同阶段/不同入口的调用代码。
    return {
        "heightMap_coarse": coarse_map,
        "heightMap_prime": height_map_prime,
        "h_coarse": coarse_map,
        "h_prime": height_map_prime,
        "phi0_map": phi0_map,
        "g0_map": g0_map,
        "theta_map": theta_map,
        "phase_gap_raw": phase_gap_raw,
        "baseline_fringe_order_map": fringe_order,
        "peak_amplitude_map": peak_amplitude,
        "k0_index": global_peak_index,
        "k0_value": global_k0,
        "fft_length": int(fft_length),
        "zero_padding_mode": str(zero_padding_mode),
    }
