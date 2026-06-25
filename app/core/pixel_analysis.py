from __future__ import annotations
"""
单像素分析模块。

用途：
- 给 GUI 的像素点检查窗口提供完整诊断数据
- 复用主流程中的频谱、unwrap、局部拟合策略

注意：
这个模块输出的是“单点解释信息”，并不执行全场 connect/fit。
"""

import numpy as np

from app.core.adaptive_window import build_analysis_window
from app.core.kernel import (
    _build_k_axis,
    _compute_nonuniform_uniform_grid_spectrum,
    _prepare_local_phase_window,
    _resolve_optical_positions_um,
    _unwrap_phase_itoh,
    _validate_sample_positions,
    resolve_fft_length,
)


def _custom_phase_unwrap(phi_raw: np.ndarray) -> np.ndarray:
    """旧版局部解包裹兼容入口，保留给现有测试使用。"""
    phi_raw = np.asarray(phi_raw, dtype=np.float32)
    n_phase = phi_raw.shape[0]
    wrap_count = 0
    steps = np.zeros(n_phase, dtype=np.float32)

    for index in range(n_phase - 1):
        if phi_raw[index] > phi_raw[index + 1]:
            wrap_count += 1
        steps[index + 1] = wrap_count

    return phi_raw + 2.0 * np.pi * steps


def _fit_phase_segment(
    k_masked: np.ndarray,
    phi_masked: np.ndarray,
    amplitude_masked: np.ndarray,
    k_center: float,
    fitting_method: str,
) -> np.ndarray:
    """对单像素局部窗口生成拟合相位曲线（仅返回曲线，不返回参数）。"""
    if fitting_method == "simple":
        coeffs = np.polyfit(k_masked, phi_masked, deg=1)
        return np.polyval(coeffs, k_masked).astype(np.float32)

    if fitting_method == "weighted":
        centered = k_masked - float(k_center)
        weights = amplitude_masked * amplitude_masked
        sw = np.sum(weights)
        swx = np.sum(weights * centered)
        swy = np.sum(weights * phi_masked)
        swxx = np.sum(weights * centered * centered)
        swxy = np.sum(weights * centered * phi_masked)
        denom = sw * swxx - swx * swx
        if sw <= 0.0 or abs(float(denom)) <= 1e-12:
            return np.full_like(k_masked, np.nan, dtype=np.float32)
        g0 = (sw * swxy - swx * swy) / denom
        phi0 = (swy - g0 * swx) / sw
        return (g0 * centered + phi0).astype(np.float32)

    if fitting_method == "quadratic":
        coeffs = np.polyfit(k_masked, phi_masked, deg=2)
        return np.polyval(coeffs, k_masked).astype(np.float32)

    raise ValueError(f"Unsupported fitting method: {fitting_method}")


def build_pixel_analysis(
    intensity_data: np.ndarray,
    x: int,
    y: int,
    step_size: float,
    start_height: float = 0.0,
    unwrap_method: str = "itoh",
    window_size: int = 9,
    fitting_method: str = "weighted",
    global_k0_index: int | None = None,
    global_k0_value: float | None = None,
    sample_positions_um: np.ndarray | None = None,
    window_name: str = "hamming",
    window_alpha: float = 0.5,
    zero_padding_mode: str = "fixed_512",
) -> dict[str, np.ndarray | int]:
    """
    生成某个像素的完整分析剖面。

    返回内容覆盖四层信息：
    1. 原始/去直流信号
    2. 频谱幅值与相位
    3. `k0` 附近局部窗口
    4. unwrap 后拟合曲线
    """
    cube = np.asarray(intensity_data, dtype=np.float32)
    # 注意索引顺序：输入是 (height, width, samples)，所以取值用 [y, x, :]
    signal_raw = cube[y, x, :].astype(np.float32)
    signal_dc = signal_raw - np.mean(signal_raw)

    if sample_positions_um is None:
        sample_index = np.arange(signal_raw.shape[0], dtype=np.float32)
        signal_x = float(start_height) + sample_index * float(step_size)
    else:
        signal_x = _validate_sample_positions(sample_positions_um, signal_raw.shape[0]).astype(np.float32)

    # 单点频谱同样做加窗，减少泄漏带来的峰值偏移。
    window = build_analysis_window(signal_dc[None, :], window_name, window_alpha=window_alpha)[0]
    signal_windowed = signal_dc * window
    fft_length = resolve_fft_length(signal_windowed.shape[0], zero_padding_mode)
    if sample_positions_um is None:
        spectrum = np.fft.rfft(signal_windowed, n=fft_length)
    else:
        optical_positions = _resolve_optical_positions_um(signal_windowed.shape[0], step_size, sample_positions_um)
        spectrum, k_x = _compute_nonuniform_uniform_grid_spectrum(
            signal_windowed[None, :],
            optical_positions,
            fft_length=fft_length,
        )
        spectrum = spectrum[0]
    if sample_positions_um is None:
        amplitude = np.abs(spectrum).astype(np.float32) / float(fft_length)
        if amplitude.shape[0] > 2:
            amplitude[1:-1] *= 2.0
    else:
        amplitude = np.abs(spectrum).astype(np.float32) / float(signal_windowed.shape[0])
    phase_raw = np.angle(spectrum).astype(np.float32)
    # 这里的 `phase_unwrapped_y` 主要用于可视化；
    # 局部拟合时会按分支策略再次取局部窗口处理。
    if unwrap_method == "global":
        phase_unwrapped = np.unwrap(phase_raw).astype(np.float32)
    elif unwrap_method in {"itoh", "gr", "pda", "branch_search", "local"}:
        phase_unwrapped = phase_raw.copy()
    else:
        raise ValueError(f"Unsupported unwrap method: {unwrap_method}")

    if sample_positions_um is None:
        dz_um = 2.0 * float(step_size)
        fs_um = 1.0 / dz_um
        f_x = np.arange(amplitude.shape[0], dtype=np.float32) * (fs_um / float(fft_length))
        k_x = (2.0 * np.pi * f_x).astype(np.float32)
    # `k0` 选取优先级：
    # 1) 指定 `global_k0_value`
    # 2) 指定 `global_k0_index`
    # 3) 当前像素幅值主峰
    if global_k0_value is not None:
        peak_index = int(np.argmin(np.abs(k_x - float(global_k0_value))))
        k0_x = float(global_k0_value)
    elif global_k0_index is not None:
        peak_index = int(global_k0_index)
        k0_x = float(k_x[peak_index])
    else:
        peak_index = int(np.argmax(amplitude))
        k0_x = float(k_x[peak_index])

    # 构造 `k0` 两侧局部窗口，供相位拟合与对比显示。
    offsets = np.arange(-int(window_size), int(window_size) + 1, dtype=np.int32)
    idx_range = np.clip(peak_index + offsets, 0, k_x.shape[0] - 1)
    fit_mask_k_x = k_x[idx_range].astype(np.float32)
    fit_mask_amplitude_y = amplitude[idx_range].astype(np.float32)
    if unwrap_method == "global":
        fit_mask_phase_y = phase_unwrapped[idx_range].astype(np.float32)
        fit_phase_y = _fit_phase_segment(
            k_masked=fit_mask_k_x.astype(np.float64),
            phi_masked=fit_mask_phase_y.astype(np.float64),
            amplitude_masked=fit_mask_amplitude_y.astype(np.float64),
            k_center=float(k0_x),
            fitting_method=fitting_method,
        )
    elif unwrap_method == "itoh":
        fit_mask_phase_y = _unwrap_phase_itoh(phase_raw[idx_range]).astype(np.float32)
        fit_phase_y = _fit_phase_segment(
            k_masked=fit_mask_k_x.astype(np.float64),
            phi_masked=fit_mask_phase_y.astype(np.float64),
            amplitude_masked=fit_mask_amplitude_y.astype(np.float64),
            k_center=float(k0_x),
            fitting_method=fitting_method,
        )
    else:
        # `local` 保留为兼容别名，内部按参考项目映射到 `pda`；
        # `gr` 则走当前项目正式接入的 GR 局部解包裹分支。
        local_method = "pda" if unwrap_method == "local" else unwrap_method
        fit_mask_phase_y, fit_phase_y, _, _ = _prepare_local_phase_window(
            k_masked=fit_mask_k_x.astype(np.float64),
            phase_wrapped=phase_raw[idx_range].astype(np.float64),
            amplitude_masked=fit_mask_amplitude_y.astype(np.float64),
            k_center=float(k0_x),
            unwrap_method=local_method,
            fitting_method=fitting_method,
        )

    # 字段命名尽量与 GUI 历史代码保持兼容（如 `*_y` 后缀）。
    return {
        "x": int(x),
        "y": int(y),
        "signal_x": signal_x,
        "signal_raw_y": signal_raw,
        "signal_dc_y": signal_dc,
        "window_y": window.astype(np.float32),
        "signal_windowed_y": signal_windowed.astype(np.float32),
        "k_x": k_x,
        "amplitude_y": amplitude,
        "k0_x": k0_x,
        "k0_y": float(amplitude[peak_index]),
        "k0_index": peak_index,
        "fft_length": int(fft_length),
        "phase_raw_y": phase_raw,
        "phase_unwrapped_y": phase_unwrapped,
        "fit_k_x": fit_mask_k_x,
        "fit_phase_y": fit_phase_y,
        "fit_mask_k_x": fit_mask_k_x,
        "fit_mask_phase_y": fit_mask_phase_y,
    }
