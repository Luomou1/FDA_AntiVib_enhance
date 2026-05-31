from __future__ import annotations
"""
数值核心入口模块。

这个文件主要承担三类职责：
1. 基础数值工具：去直流、频谱、K0 估计、局部拟合
2. 正式分析总入口：只调度 FDA baseline 主链
3. 为 GUI / worker / pixel analysis 提供公共底层函数

注意：
真正与 FDA 分析步骤对应的实现主要分布在：
- `app.core.fda_baseline`

本文件更偏向于“公共底层 + 正式总调度”。
"""

import time

import numpy as np
import finufft
from scipy.fft import rfft
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, peak_prominences
FFT_SIZE = 512
NUFFT_EPS = 1e-6


def resolve_fft_length(sample_count: int, zero_padding_mode: str = "next_power_of_two") -> int:
    """
    根据补零模式确定 FFT 长度。

    - `none`：不补零，FFT 长度等于原始信号点数。
    - `pow2_1`：补到第 1 个严格大于 N 的 2 的幂，例如 130->256、280->512。
    - `pow2_2` / `pow2_3`：继续取后面的 2 的幂，例如 130->512/1024。
    - `factor_2` / `factor_4` / `factor_8`：按固定倍数补零。
    """
    count = int(sample_count)
    if count < 2:
        raise ValueError("sample_count must be at least 2.")
    mode = str(zero_padding_mode).strip().lower()
    if mode in {"none", "no", "off", "raw"}:
        return count

    # 兼容旧配置名：默认的 next_power_of_two 等价于第 1 个大于 N 的 2 的幂。
    pow2_aliases = {
        "next_power_of_two": 1,
        "next_pow2": 1,
        "pow2": 1,
        "power2": 1,
        "pow2_1": 1,
        "pow2_2": 2,
        "pow2_3": 3,
    }
    if mode in pow2_aliases:
        base_power = int(count.bit_length())
        return 1 << (base_power + pow2_aliases[mode] - 1)

    factor_aliases = {
        "factor_2": 2,
        "2x": 2,
        "factor_4": 4,
        "4x": 4,
        "factor_8": 8,
        "8x": 8,
    }
    if mode in factor_aliases:
        return count * factor_aliases[mode]
    raise ValueError(f"Unsupported zero padding mode: {zero_padding_mode}")


def _resolve_block_rows(height: int, width: int) -> int:
    """根据图像尺寸估算一个合适的分块行数，控制单次处理的像素量。"""
    target_pixels = 32768
    return max(1, min(height, target_pixels // max(1, width)))


def _remove_dc_batch(intensity_curves: np.ndarray) -> np.ndarray:
    """
    对一批扫描曲线做去直流/去慢变趋势。

    这里尽量兼顾三种估计：
    - 全局平均
    - 边缘估计
    - 平滑趋势估计
    目的是让后续频谱分析更稳定。
    """
    smooth_curves = np.asarray(intensity_curves, dtype=np.float32)
    dc_mean = smooth_curves.mean(axis=1)
    edge_length = min(5, smooth_curves.shape[1] // 10)

    if edge_length > 0:
        edge_values = np.concatenate(
            [smooth_curves[:, :edge_length], smooth_curves[:, -edge_length:]],
            axis=1,
        )
        dc_edge = np.median(edge_values, axis=1)
    else:
        dc_edge = dc_mean

    if smooth_curves.shape[1] > 10:
        window_size = max(3, smooth_curves.shape[1] // 5)
        trend = uniform_filter1d(smooth_curves, size=window_size, axis=1, mode="nearest")
        dc_trend = np.mean(np.stack([trend[:, 0], trend[:, -1]], axis=1), axis=1)
    else:
        dc_trend = dc_mean

    signal_std = np.std(smooth_curves, axis=1)
    dc_component = np.where(
        np.abs(dc_edge - dc_mean) < 0.1 * signal_std,
        dc_edge,
        np.where(
            np.abs(dc_trend - dc_mean) < 0.2 * signal_std,
            dc_trend,
            dc_mean,
        ),
    )

    return smooth_curves - dc_component[:, None]


def _custom_phase_unwrap_batch(phi_raw: np.ndarray) -> np.ndarray:
    """
    当前项目使用的 GR 局部解包裹辅助函数。

    它不是全场 2D unwrap，而是针对拟合窗口内的一维相位序列，
    用“相位下降就加一个 2π 周期”的累计规则做本地展开。

    这里同时兼容两种输入：
    - 1D：单个像素的局部频谱相位窗口
    - 2D：一批像素的局部频谱相位窗口

    这样 `GR解包裹` 既可以接入当前 baseline 主流程，
    也可以保留旧兼容路径里的批量调用方式。
    """
    phi_raw = np.asarray(phi_raw, dtype=np.float32)
    squeeze_result = False
    if phi_raw.ndim == 1:
        phi_raw = phi_raw[None, :]
        squeeze_result = True
    elif phi_raw.ndim != 2:
        raise ValueError("phi_raw must be 1D or 2D.")

    phi_unwrapped = np.zeros_like(phi_raw)
    phi_unwrapped[:, 0] = phi_raw[:, 0]
    wrap_counts = np.zeros(phi_raw.shape[0], dtype=np.float32)

    for index in range(phi_raw.shape[1] - 1):
        wrap_counts += (phi_raw[:, index] > phi_raw[:, index + 1]).astype(np.float32)
        phi_unwrapped[:, index + 1] = phi_raw[:, index + 1] + 2 * np.pi * wrap_counts

    if squeeze_result:
        return phi_unwrapped[0]
    return phi_unwrapped


def _unwrap_phase_itoh(phi_raw: np.ndarray) -> np.ndarray:
    """
    Itoh 一维相位解包裹。

    它按相邻相位差的主值来累计恢复连续相位，
    比简单的“只要下降就加 2π”更符合经典局部相位展开思路。
    """
    phi_raw = np.asarray(phi_raw, dtype=np.float64)
    if phi_raw.ndim == 1:
        result = np.empty_like(phi_raw)
        result[0] = phi_raw[0]
        if phi_raw.shape[0] > 1:
            delta = np.angle(np.exp(1j * np.diff(phi_raw)))
            result[1:] = result[0] + np.cumsum(delta)
        return result.astype(np.float32)
    if phi_raw.ndim == 2:
        result = np.empty_like(phi_raw)
        result[:, 0] = phi_raw[:, 0]
        if phi_raw.shape[1] > 1:
            delta = np.angle(np.exp(1j * np.diff(phi_raw, axis=1)))
            result[:, 1:] = result[:, [0]] + np.cumsum(delta, axis=1)
        return result.astype(np.float32)
    raise ValueError("phi_raw must be 1D or 2D.")


def _fit_phase_segment_single(
    k_masked: np.ndarray,
    phi_masked: np.ndarray,
    amplitude_masked: np.ndarray,
    k_center: float,
    fitting_method: str,
) -> tuple[np.ndarray, float, float]:
    """
    对单个局部相位窗口做拟合，同时返回：
    - 拟合后的整段相位曲线
    - `g0`
    - `phi0`
    """
    k_masked = np.asarray(k_masked, dtype=np.float64)
    phi_masked = np.asarray(phi_masked, dtype=np.float64)
    amplitude_masked = np.asarray(amplitude_masked, dtype=np.float64)

    if fitting_method == "simple":
        coeffs = np.polyfit(k_masked, phi_masked, deg=1)
        predicted = np.polyval(coeffs, k_masked)
        g0 = float(coeffs[0])
        phi0 = float(np.polyval(coeffs, k_center))
        return predicted.astype(np.float32), g0, phi0

    if fitting_method == "weighted":
        centered = k_masked - float(k_center)
        weights = amplitude_masked * amplitude_masked
        sw = float(np.sum(weights))
        swx = float(np.sum(weights * centered))
        swy = float(np.sum(weights * phi_masked))
        swxx = float(np.sum(weights * centered * centered))
        swxy = float(np.sum(weights * centered * phi_masked))
        denom = sw * swxx - swx * swx
        if sw <= 0.0 or abs(denom) <= 1e-12:
            predicted = np.full_like(k_masked, np.nan, dtype=np.float32)
            return predicted, float("nan"), float("nan")
        g0 = (sw * swxy - swx * swy) / denom
        phi0 = (swy - g0 * swx) / sw
        predicted = g0 * centered + phi0
        return predicted.astype(np.float32), float(g0), float(phi0)

    if fitting_method == "quadratic":
        coeffs = np.polyfit(k_masked, phi_masked, deg=2)
        predicted = np.polyval(coeffs, k_masked)
        g0 = float(2.0 * coeffs[0] * k_center + coeffs[1])
        phi0 = float(np.polyval(coeffs, k_center))
        return predicted.astype(np.float32), g0, phi0

    raise ValueError(f"Unsupported fitting method: {fitting_method}")


def _reconstruct_phase_window_pda(
    k_masked: np.ndarray,
    phase_wrapped: np.ndarray,
    amplitude_masked: np.ndarray,
    k_center: float,
) -> np.ndarray:
    """
    PDA（phase derivative approximation）局部重构。

    思路来自参考项目：
    - 先通过频谱相邻相位差估一个初始 slope
    - 再构造预测相位
    - 最后把每个点归到最接近预测值的 `2π` 分支
    """
    complex_spectrum = amplitude_masked.astype(np.float64) * np.exp(1j * phase_wrapped.astype(np.float64))
    if complex_spectrum.shape[0] < 2:
        return phase_wrapped.astype(np.float32)

    delta_phase = np.angle(complex_spectrum[1:] * np.conj(complex_spectrum[:-1]))
    delta_k = np.diff(k_masked.astype(np.float64))
    diff_weights = np.maximum(amplitude_masked[:-1] * amplitude_masked[1:], 1e-12).astype(np.float64)
    denom = float(np.sum(diff_weights * delta_k * delta_k))
    if denom <= 1e-12:
        return _unwrap_phase_itoh(phase_wrapped)

    g0_init = float(np.sum(diff_weights * delta_k * delta_phase) / denom)
    detrended = complex_spectrum * np.exp(-1j * g0_init * (k_masked.astype(np.float64) - float(k_center)))
    phi0_init = float(np.angle(np.sum(np.maximum(amplitude_masked, 1e-12) * detrended)))
    predicted = g0_init * (k_masked.astype(np.float64) - float(k_center)) + phi0_init
    branch = np.rint((predicted - phase_wrapped.astype(np.float64)) / (2.0 * np.pi))
    return (phase_wrapped.astype(np.float64) + 2.0 * np.pi * branch).astype(np.float32)


def _reconstruct_phase_window_branch_search(
    k_masked: np.ndarray,
    phase_wrapped: np.ndarray,
    amplitude_masked: np.ndarray,
    k_center: float,
    fitting_method: str,
    iterations: int = 4,
) -> np.ndarray:
    """
    branch-search 局部重构。

    先用 PDA 给一个初值，再反复：
    - 用当前展开结果拟合预测曲线
    - 把原始包裹相位重新分配到最接近预测曲线的 `2π` 分支
    """
    current = _reconstruct_phase_window_pda(k_masked, phase_wrapped, amplitude_masked, k_center).astype(np.float64)
    wrapped = phase_wrapped.astype(np.float64)
    for _ in range(max(1, int(iterations))):
        predicted, _, _ = _fit_phase_segment_single(
            k_masked=k_masked,
            phi_masked=current,
            amplitude_masked=amplitude_masked,
            k_center=k_center,
            fitting_method=fitting_method,
        )
        branch = np.rint((predicted.astype(np.float64) - wrapped) / (2.0 * np.pi))
        updated = wrapped + 2.0 * np.pi * branch
        if np.allclose(updated, current, atol=1e-8):
            break
        current = updated
    return current.astype(np.float32)


def _prepare_local_phase_window(
    k_masked: np.ndarray,
    phase_wrapped: np.ndarray,
    amplitude_masked: np.ndarray,
    k_center: float,
    unwrap_method: str,
    fitting_method: str,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    统一准备局部相位窗口。

    当前支持：
    - `itoh`
    - `gr`
    - `pda`
    - `branch_search`
    - `local`：兼容旧参数，内部映射到 `pda`
    """
    method = "pda" if unwrap_method == "local" else unwrap_method
    if method == "itoh":
        phi_unwrapped = _unwrap_phase_itoh(phase_wrapped)
    elif method == "gr":
        # `GR解包裹` 直接使用项目历史上的“单调下降补 2π”规则，
        # 保持旧局部策略的行为特征，同时把它接入正式主流程。
        phi_unwrapped = _custom_phase_unwrap_batch(phase_wrapped)
    elif method == "pda":
        phi_unwrapped = _reconstruct_phase_window_pda(k_masked, phase_wrapped, amplitude_masked, k_center)
    elif method == "branch_search":
        phi_unwrapped = _reconstruct_phase_window_branch_search(
            k_masked=k_masked,
            phase_wrapped=phase_wrapped,
            amplitude_masked=amplitude_masked,
            k_center=k_center,
            fitting_method=fitting_method,
        )
    else:
        raise ValueError(f"Unsupported unwrap method: {unwrap_method}")

    predicted, g0, phi0 = _fit_phase_segment_single(
        k_masked=k_masked,
        phi_masked=phi_unwrapped,
        amplitude_masked=amplitude_masked,
        k_center=k_center,
        fitting_method=fitting_method,
    )
    return phi_unwrapped.astype(np.float32), predicted.astype(np.float32), float(g0), float(phi0)


def _fit_simple_batch(k_masked: np.ndarray, phi_masked: np.ndarray, k_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对每个像素在 `k0` 附近做简单线性拟合。"""
    count = float(k_masked.shape[1])
    sx = np.sum(k_masked, axis=1)
    sy = np.sum(phi_masked, axis=1)
    sxx = np.sum(k_masked * k_masked, axis=1)
    sxy = np.sum(k_masked * phi_masked, axis=1)
    denom = count * sxx - sx * sx

    g0 = np.full_like(k_center, np.nan, dtype=np.float64)
    phi0 = np.full_like(k_center, np.nan, dtype=np.float64)
    valid = np.abs(denom) > 1e-12
    g0[valid] = (count * sxy[valid] - sx[valid] * sy[valid]) / denom[valid]
    intercept = np.full_like(k_center, np.nan, dtype=np.float64)
    intercept[valid] = (sy[valid] - g0[valid] * sx[valid]) / count
    phi0[valid] = g0[valid] * k_center[valid] + intercept[valid]
    return g0, phi0


def _fit_weighted_batch(k_masked: np.ndarray, phi_masked: np.ndarray, amplitude: np.ndarray, k_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对每个像素在 `k0` 附近做加权线性拟合，权重来自频谱幅值。"""
    centered = k_masked - k_center[:, None]
    weights = amplitude * amplitude
    sw = np.sum(weights, axis=1)
    swx = np.sum(weights * centered, axis=1)
    swy = np.sum(weights * phi_masked, axis=1)
    swxx = np.sum(weights * centered * centered, axis=1)
    swxy = np.sum(weights * centered * phi_masked, axis=1)
    denom = sw * swxx - swx * swx

    g0 = np.full_like(k_center, np.nan, dtype=np.float64)
    phi0 = np.full_like(k_center, np.nan, dtype=np.float64)
    valid = (sw > 0) & (np.abs(denom) > 1e-12)
    g0[valid] = (sw[valid] * swxy[valid] - swx[valid] * swy[valid]) / denom[valid]
    phi0[valid] = (swy[valid] - g0[valid] * swx[valid]) / sw[valid]
    return g0, phi0


def _fit_quadratic_batch(k_masked: np.ndarray, phi_masked: np.ndarray, k_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对每个像素在 `k0` 附近做二次拟合。"""
    x0 = k_masked * k_masked
    x1 = k_masked
    ones = np.ones_like(k_masked)
    design = np.stack([x0, x1, ones], axis=2)
    xtx = np.einsum("pwi,pwj->pij", design, design, optimize=True)
    xty = np.einsum("pwi,pw->pi", design, phi_masked, optimize=True)
    coeffs = np.einsum("pij,pj->pi", np.linalg.pinv(xtx), xty, optimize=True)

    a = coeffs[:, 0]
    b = coeffs[:, 1]
    c = coeffs[:, 2]
    g0 = 2 * a * k_center + b
    phi0 = a * k_center * k_center + b * k_center + c
    return g0, phi0


def _demean_curves(curves: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float32)
    return curves - np.mean(curves, axis=1, keepdims=True)


def _build_window(window_name: str, sample_count: int, window_alpha: float = 0.5) -> np.ndarray:
    normalized = window_name.strip().lower()
    if normalized == "none":
        return np.ones(sample_count, dtype=np.float32)
    if normalized == "hamming":
        return np.hamming(sample_count).astype(np.float32)
    raise ValueError(f"Unsupported window: {window_name}")


def _validate_sample_positions(sample_positions_um: np.ndarray, sample_count: int) -> np.ndarray:
    positions = np.asarray(sample_positions_um, dtype=np.float32)
    if positions.ndim != 1 or positions.shape[0] != sample_count:
        raise ValueError("sample_positions_um must be a 1D array with one entry per sample.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("sample_positions_um must contain only finite values.")
    if np.any(np.diff(positions) <= 0.0):
        raise ValueError("sample_positions_um must be strictly increasing.")
    return positions


def _resolve_optical_positions_um(
    sample_count: int,
    step_size: float,
    sample_positions_um: np.ndarray | None,
) -> np.ndarray:
    """
    把机械位移位置转换成光程位移位置。

    对反射式干涉来说，光程通常是机械位移的两倍，所以这里乘以 2。
    """
    if sample_positions_um is None:
        positions = np.arange(sample_count, dtype=np.float32) * float(step_size)
    else:
        positions = _validate_sample_positions(sample_positions_um, sample_count)
        positions = positions - positions[0]
    return (2.0 * positions).astype(np.float32)


def _build_k_axis(
    sample_count: int,
    step_size: float,
    sample_positions_um: np.ndarray | None,
    fft_length: int,
) -> np.ndarray:
    """
    构造一侧频谱对应的波数轴。

    非均匀采样模式下，这里使用保守的最大采样间隔来构造统一网格，
    保证后续频谱解释和测试约束一致。
    """
    if sample_positions_um is None:
        dz_um = 2.0 * float(step_size)
    else:
        optical_positions = _resolve_optical_positions_um(sample_count, step_size, sample_positions_um)
        if optical_positions.shape[0] < 2:
            raise ValueError("At least two samples are required.")
        dz_um = float(np.max(np.diff(optical_positions)))
    fs_um = 1.0 / dz_um
    frequencies = np.arange(0, fft_length // 2 + 1, dtype=np.float32) * (fs_um / float(fft_length))
    return (2.0 * np.pi * frequencies).astype(np.float32)


def _compute_nonuniform_spectrum(
    curves: np.ndarray,
    optical_positions_um: np.ndarray,
    k_axis: np.ndarray,
) -> np.ndarray:
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim == 1:
        curves = curves[None, :]
    x = np.asarray(optical_positions_um, dtype=np.float64)
    s = np.asarray(k_axis, dtype=np.float64)
    return finufft.nufft1d3(x, curves.astype(np.complex128), s, isign=-1, eps=NUFFT_EPS)


def _compute_nonuniform_uniform_grid_spectrum(
    curves: np.ndarray,
    optical_positions_um: np.ndarray,
    fft_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    把非均匀采样曲线投影到统一频率网格上。

    这里底层使用 FINUFFT，把非均匀位置上的采样转换为统一一侧频谱。
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim == 1:
        curves = curves[None, :]
    optical_positions_um = np.asarray(optical_positions_um, dtype=np.float64)
    if optical_positions_um.ndim != 1 or optical_positions_um.shape[0] != curves.shape[1]:
        raise ValueError("optical_positions_um must match the sample dimension of curves.")
    if optical_positions_um.shape[0] < 2:
        raise ValueError("At least two samples are required.")

    dz_ref = float(np.max(np.diff(optical_positions_um)))
    period_um = float(fft_length) * dz_ref
    x = (2.0 * np.pi * optical_positions_um / period_um).astype(np.float64)
    x = ((x + np.pi) % (2.0 * np.pi)) - np.pi

    n_modes = int(fft_length) + 1
    spectrum_full = finufft.nufft1d1(
        x,
        curves.astype(np.complex128),
        n_modes=(n_modes,),
        isign=-1,
        eps=NUFFT_EPS,
        modeord=1,
    )

    positive_count = int(fft_length) // 2 + 1
    spectrum = spectrum_full[:, :positive_count]
    k_axis = (2.0 * np.pi / period_um) * np.arange(positive_count, dtype=np.float64)
    return spectrum, k_axis.astype(np.float32)


def _compute_windowed_spectrum(
    curves: np.ndarray,
    step_size: float,
    sample_positions_um: np.ndarray | None = None,
    window_name: str = "hamming",
    window_alpha: float = 0.5,
    fft_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算当前批像素的窗口化频谱。

    - 均匀采样：直接走 FFT
    - 非均匀采样：先转成光程位置，再走统一网格频谱
    """
    fft_length = resolve_fft_length(curves.shape[1]) if fft_length is None else int(fft_length)
    window = _build_window(window_name, curves.shape[1], window_alpha=window_alpha)
    windowed = curves * window[None, :]
    if sample_positions_um is None:
        spectrum = rfft(windowed, n=fft_length, axis=1)
    else:
        optical_positions = _resolve_optical_positions_um(curves.shape[1], step_size, sample_positions_um)
        spectrum, k_axis = _compute_nonuniform_uniform_grid_spectrum(windowed, optical_positions, fft_length)

    if sample_positions_um is None:
        amplitude = np.abs(spectrum) / float(fft_length)
        if amplitude.shape[1] > 2:
            amplitude[:, 1:-1] *= 2.0
    else:
        amplitude = np.abs(spectrum) / float(curves.shape[1])
    phase = np.angle(spectrum)
    if sample_positions_um is None:
        k_axis = _build_k_axis(curves.shape[1], step_size, sample_positions_um=None, fft_length=fft_length)
    return amplitude.astype(np.float32), phase.astype(np.float32), k_axis.astype(np.float32)


def _compute_one_sided_amplitude(
    curves: np.ndarray,
    step_size: float,
    window_name: str,
    fft_length: int,
    sample_positions_um: np.ndarray | None = None,
    window_alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    计算一侧幅值谱和对应的 k 轴。

    返回值：
    - `amplitude`：形状 `(pixel_count, fft_length//2+1)`
    - `k_axis`：与频谱列一一对应的波数坐标

    这个函数主要给全局 K0 估计使用，因此不返回相位。
    """
    window = _build_window(window_name, curves.shape[1], window_alpha=window_alpha)
    windowed = curves * window[None, :]
    if sample_positions_um is None:
        spectrum = rfft(windowed, n=fft_length, axis=1)
    else:
        optical_positions = _resolve_optical_positions_um(curves.shape[1], step_size, sample_positions_um)
        spectrum, k_axis = _compute_nonuniform_uniform_grid_spectrum(windowed, optical_positions, fft_length)
    if sample_positions_um is None:
        amplitude = np.abs(spectrum) / float(fft_length)
        if amplitude.shape[1] > 2:
            amplitude[:, 1:-1] *= 2.0
    else:
        amplitude = np.abs(spectrum) / float(curves.shape[1])
    if sample_positions_um is None:
        k_axis = _build_k_axis(curves.shape[1], step_size, sample_positions_um=None, fft_length=fft_length)
    return amplitude.astype(np.float32), k_axis.astype(np.float32)


def _select_candidate_curves(intensity_data: np.ndarray, candidate_ratio: float) -> np.ndarray:
    """
    从整幅数据中选取一批“质量更高”的曲线作为 K0 候选。

    当前质量指标：`(max-min)/|mean|`，直观上更偏向调制度高、
    直流占比低的像素曲线。
    """
    curves = np.asarray(intensity_data, dtype=np.float32).reshape(-1, intensity_data.shape[2])
    pv = np.max(curves, axis=1) - np.min(curves, axis=1)
    mean = np.mean(curves, axis=1)
    quality = pv / np.maximum(np.abs(mean), 1e-6)
    valid = np.isfinite(quality) & (quality > 0)
    valid_curves = curves[valid]
    valid_quality = quality[valid]
    if valid_curves.shape[0] == 0:
        raise ValueError("No valid pixels available for global K0 estimation.")

    ratio = float(candidate_ratio)
    if not (0.0 < ratio <= 1.0):
        raise ValueError("candidate_ratio must be within (0, 1].")

    # 只保留前 `candidate_ratio` 的高质量曲线，抑制坏点对全局谱峰的影响。
    candidate_count = max(1, int(np.ceil(valid_curves.shape[0] * ratio)))
    order = np.argsort(valid_quality)[::-1][:candidate_count]
    return valid_curves[order]


def estimate_global_k0(
    intensity_data: np.ndarray,
    step_size: float,
    candidate_ratio: float,
    window_name: str = "hamming",
    window_alpha: float = 0.5,
    zero_padding_mode: str = "next_power_of_two",
    sample_positions_um: np.ndarray | None = None,
) -> dict[str, np.ndarray | float | int | str]:
    """
    自动估计全局 K0。

    做法是：
    1. 选出一部分质量较高的像素曲线
    2. 统计它们的频谱中值
    3. 在聚合谱上找主峰，作为 K0
    """
    cube = np.asarray(intensity_data, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError("intensity_data must be a 3D array.")

    candidate_curves = _select_candidate_curves(cube, candidate_ratio)
    demeaned = _demean_curves(candidate_curves)
    fft_length = resolve_fft_length(cube.shape[2], zero_padding_mode)
    amplitude, k_axis = _compute_one_sided_amplitude(
        demeaned,
        step_size=step_size,
        window_name=window_name,
        fft_length=fft_length,
        sample_positions_um=sample_positions_um,
        window_alpha=window_alpha,
    )
    aggregated = np.median(amplitude, axis=0).astype(np.float32)

    search = aggregated.copy()
    if search.shape[0] > 2:
        search[:2] = 0.0
    peaks, _ = find_peaks(search)
    peaks = peaks[peaks >= 2]
    if peaks.size > 0:
        prominences = peak_prominences(search, peaks)[0]
        peak_choice = int(np.argmax(prominences))
        peak_index = int(peaks[peak_choice])
        peak_prominence = float(prominences[peak_choice])
    else:
        if search.shape[0] > 2:
            peak_index = int(np.argmax(search[2:]) + 2)
        else:
            peak_index = int(np.argmax(search))
        peak_prominence = 0.0

    return {
        "k_axis": k_axis,
        "spectrum": aggregated,
        "peak_index": peak_index,
        "k0_value": float(k_axis[peak_index]),
        "peak_value": float(aggregated[peak_index]),
        "peak_prominence": peak_prominence,
        "candidate_count": int(candidate_curves.shape[0]),
        "window_name": window_name.strip().lower(),
        "window_alpha": float(window_alpha),
        "zero_padding_mode": str(zero_padding_mode),
        "fft_length": int(fft_length),
    }


def _compute_block(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    旧版单块处理入口（保留用于兼容/回归）。

    与 `fda_baseline.py` 的新版分块逻辑相比，这里更精简：
    - 只返回 `h`、`h_prime`、`phi0`
    - unwrap 只支持 `global/local`
    """
    # `block` 约定为 `(rows, width, sample_count)`。
    row_count, width, sample_count = block.shape
    pixel_count = row_count * width
    curves = block.reshape(pixel_count, sample_count)
    curves = _remove_dc_batch(curves)

    amplitude, phase, k = _compute_windowed_spectrum(
        curves,
        step_size=step_size,
        sample_positions_um=sample_positions_um,
        window_name=window_name,
        window_alpha=window_alpha,
        fft_length=fft_length,
    )

    offsets = np.arange(-window_size, window_size + 1, dtype=np.int32)
    idx_range_row = np.clip(global_peak_index + offsets, 0, k.size - 1)
    idx_range = np.broadcast_to(idx_range_row, (pixel_count, idx_range_row.shape[0]))
    pixel_index = np.arange(pixel_count)[:, None]

    k_masked = k[idx_range].astype(np.float64)
    amplitude_masked = amplitude[pixel_index, idx_range].astype(np.float64)
    if unwrap_method == "global":
        phi_unwrapped = np.unwrap(phase, axis=1)
        phi_masked = phi_unwrapped[pixel_index, idx_range].astype(np.float64)
    elif unwrap_method == "local":
        phi_masked = _custom_phase_unwrap_batch(phase[pixel_index, idx_range]).astype(np.float64)
    else:
        raise ValueError(f"Unsupported unwrap method: {unwrap_method}")

    k_center = np.full(pixel_count, float(fixed_k0_value), dtype=np.float64)
    if fitting_method == "simple":
        g0, phi0 = _fit_simple_batch(k_masked, phi_masked, k_center)
    elif fitting_method == "quadratic":
        g0, phi0 = _fit_quadratic_batch(k_masked, phi_masked, k_center)
    elif fitting_method == "weighted":
        g0, phi0 = _fit_weighted_batch(k_masked, phi_masked, amplitude_masked, k_center)
    else:
        raise ValueError(f"Unsupported fitting method: {fitting_method}")

    # 连续高度近似：h = -g0/2。
    h = -g0 / 2.0
    kz = g0 / 2.0
    h_prime = np.full_like(h, np.nan)
    valid_center = np.abs(k_center) > 1e-12
    # 旧版 `h_prime` 构造公式，保留以兼容历史输出。
    h_prime[valid_center] = -(1.0 / (2.0 * k_center[valid_center])) * (
        phi0[valid_center]
        + 2
        * np.pi
        * np.round((2 * k_center[valid_center] * kz[valid_center] - phi0[valid_center]) / (2 * np.pi))
    )

    return (
        (h.reshape(row_count, width) * 1000.0).astype(np.float32),
        (h_prime.reshape(row_count, width) * 1000.0).astype(np.float32),
        phi0.reshape(row_count, width).astype(np.float32),
    )


def _run_formal_analysis_pipeline(
    intensity_data: np.ndarray,
    step_size: float,
    window_size: int,
    fitting_method: str,
    unwrap_method: str,
    fixed_k0_value: float | None = None,
    sample_positions_um: np.ndarray | None = None,
    phase_gap_method: str = "FDA",
    window_name: str = "hamming",
    window_alpha: float = 0.5,
    zero_padding_mode: str = "next_power_of_two",
    show_progress: bool = False,
    progress_callback=None,
) -> dict[str, np.ndarray]:
    """执行 FDA / PhaseGap 统一分析入口。"""
    from app.core.fda_baseline import analyze_cube_baseline

    baseline_result = analyze_cube_baseline(
        intensity_data=intensity_data,
        step_size=step_size,
        window_size=window_size,
        fitting_method=fitting_method,
        unwrap_method=unwrap_method,
        fixed_k0_value=fixed_k0_value,
        sample_positions_um=sample_positions_um,
        window_name=window_name,
        window_alpha=window_alpha,
        zero_padding_mode=zero_padding_mode,
        show_progress=show_progress,
        progress_callback=progress_callback,
    )

    method_name = str(phase_gap_method).strip()
    if method_name.lower() == "fda":
        merged = dict(baseline_result)
        # FDA 模式直接采用 baseline 层结果：
        # `h` 是斜率高度，`h_prime` 是基于 phi0 + 2πN 的整数级次修正高度。
        merged["h"] = np.asarray(baseline_result["h_coarse"], dtype=np.float32)
        merged["h_prime"] = np.asarray(baseline_result["h_prime"], dtype=np.float32)
        merged["heightMap"] = merged["h"]
        merged["heightMap_prime"] = merged["h_prime"]
        merged["phi0"] = merged["phi0_map"]
        merged["phase_gap_method"] = "FDA"
        merged["analysis_mode"] = "FDA"
        return merged

    from app.core.phase_gap import analyze_phase_gap_maps

    phase_gap_result = analyze_phase_gap_maps(baseline_result, phase_gap_method=method_name)

    merged = dict(baseline_result)
    merged.update(phase_gap_result)
    # PhaseGap 模式保留 FDA baseline 的 `h_coarse`，但对外把最终修正高度放到 `h_prime`。
    # 这样主界面对 FDA 和 PhaseGap 两条工作流使用同一套显示/导出字段。
    merged["h"] = np.asarray(baseline_result["h_coarse"], dtype=np.float32)
    merged["h_prime"] = np.asarray(phase_gap_result["h"], dtype=np.float32)
    merged["heightMap"] = merged["h"]
    merged["heightMap_prime"] = merged["h_prime"]
    merged["phi0"] = merged["phi0_map"]
    merged["h_coarse"] = np.asarray(baseline_result["h_coarse"], dtype=np.float32)
    merged["h_phase_gap"] = np.asarray(phase_gap_result["h"], dtype=np.float32)
    merged["analysis_mode"] = "PhaseGap"
    return merged


def analyze_cube_fast(
    intensity_data: np.ndarray,
    step_size: float,
    window_size: int,
    fitting_method: str,
    unwrap_method: str,
    fixed_k0_value: float | None = None,
    sample_positions_um: np.ndarray | None = None,
    phase_gap_method: str = "FDA",
    window_name: str = "hamming",
    window_alpha: float = 0.5,
    zero_padding_mode: str = "next_power_of_two",
    show_progress: bool = False,
    progress_callback=None,
) -> dict[str, np.ndarray]:
    """
    对外快速入口：当前只是正式主流程的直通包装。

    保留这个函数名的意义是兼容既有 GUI/脚本调用路径。
    """
    return _run_formal_analysis_pipeline(
        intensity_data=intensity_data,
        step_size=step_size,
        window_size=window_size,
        fitting_method=fitting_method,
        unwrap_method=unwrap_method,
        fixed_k0_value=fixed_k0_value,
        sample_positions_um=sample_positions_um,
        phase_gap_method=phase_gap_method,
        window_name=window_name,
        window_alpha=window_alpha,
        zero_padding_mode=zero_padding_mode,
        show_progress=show_progress,
        progress_callback=progress_callback,
    )
