from __future__ import annotations
"""GFDA 软件扫描步长估计。

这里实现两条软件路径：
- spatial carrier phase：用倾斜条纹的空间载波相位估计每帧相位偏移。
- intensity-height scatter：先用 FDA baseline 得到初始相位/高度信息，再按专利散点思路拟合每帧相位偏移。
"""

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter1d

from app.core.fda_baseline import analyze_cube_baseline
from app.core.scan_steps import ScanStepResult, build_scan_step_result

TAU = 2.0 * np.pi


@dataclass(frozen=True, slots=True)
class SoftwareScanStepEstimate:
    """软件估计得到的一组扫描坐标及诊断。"""

    scan_steps: ScanStepResult
    confidence: np.ndarray
    phase_rad: np.ndarray
    raw_positions_um: np.ndarray
    method: str
    diagnostics: dict[str, np.ndarray | float | int | str]

    def to_mapping(self) -> dict[str, np.ndarray | float | int | str]:
        mapping = self.scan_steps.to_mapping()
        mapping.update(
            {
                "scan_step_method": self.method,
                "scan_step_confidence": self.confidence.astype(np.float32),
                "scan_phase_estimated_rad": self.phase_rad.astype(np.float32),
                "scan_positions_estimated_raw_um": self.raw_positions_um.astype(np.float32),
            }
        )
        mapping.update(self.diagnostics)
        return mapping


def _sample_pixels(height: int, width: int, max_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(np.ceil(np.sqrt((height * width) / max(1, int(max_pixels))))))
    y = np.arange(0, height, stride, dtype=np.int32)
    x = np.arange(0, width, stride, dtype=np.int32)
    return np.meshgrid(y, x, indexing="ij")


def _robust_normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = np.percentile(arr, [1.0, 99.0])
    scale = max(float(hi - lo), 1e-6)
    return np.clip((arr - lo) / scale, 0.0, 1.0).astype(np.float32)


def _phase_to_stage_positions_um(phase_rad: np.ndarray, k0_value: float, nominal_step_um: float) -> np.ndarray:
    phase = np.asarray(phase_rad, dtype=np.float64)
    if abs(float(k0_value)) <= 1e-12:
        raise ValueError("GFDA software scan-step estimation requires a positive K0.")
    centered = phase - phase[0]
    # 反射式系统中 OPD = 2 * stage_position，故机械位置 = phase / (2*k0)。
    candidate = centered / (2.0 * float(k0_value))
    nominal = np.arange(phase.size, dtype=np.float64) * float(nominal_step_um)
    if candidate.size >= 2 and np.corrcoef(candidate, nominal)[0, 1] < 0.0:
        candidate = -candidate
    candidate -= candidate[0]
    return candidate.astype(np.float32)


def _anchor_to_nominal_span(positions_um: np.ndarray, nominal_step_um: float) -> np.ndarray:
    """用已知计划行程约束软件相位坐标的全局尺度，保留局部非均匀扰动。"""
    raw = np.asarray(positions_um, dtype=np.float64)
    anchored = raw - raw[0]
    nominal_span = float(nominal_step_um) * float(raw.size - 1)
    estimated_span = float(anchored[-1])
    if raw.size >= 2 and nominal_span > 0.0 and np.isfinite(estimated_span) and abs(estimated_span) > 1e-12:
        anchored *= nominal_span / estimated_span
    return anchored.astype(np.float32)


def _adaptive_increment_consistency_um(
    positions_um: np.ndarray,
    nominal_step_um: float,
) -> tuple[np.ndarray, float, str]:
    """按步长偏差分布压制不可信的局部倒退/离群扰动。"""
    positions = np.asarray(positions_um, dtype=np.float64)
    nominal = np.arange(positions.size, dtype=np.float64) * float(nominal_step_um)
    diffs = np.diff(positions)
    deviations = diffs - float(nominal_step_um)
    abs_dev = np.abs(deviations)
    if abs_dev.size == 0:
        return positions.astype(np.float32), 1.0, "increment_consistency_passthrough"
    reversal_count = int(np.count_nonzero(diffs <= 0.0))
    p95 = float(np.percentile(abs_dev, 95.0))
    median = float(np.median(abs_dev))
    if reversal_count == 0 and p95 <= 0.25 * abs(float(nominal_step_um)):
        return positions.astype(np.float32), 1.0, "increment_consistency_passthrough"
    gain = (median / max(p95, 1e-12)) ** 4
    gain = float(np.clip(gain, 0.03, 1.0))
    corrected = nominal + gain * (positions - nominal)
    corrected -= corrected[0]
    return corrected.astype(np.float32), gain, "adaptive_increment_consistency"


def _finalize_estimate(
    raw_positions_um: np.ndarray,
    confidence: np.ndarray,
    phase_rad: np.ndarray,
    *,
    method: str,
    nominal_step_um: float,
    diagnostics: dict[str, np.ndarray | float | int | str] | None = None,
) -> SoftwareScanStepEstimate:
    raw = np.asarray(raw_positions_um, dtype=np.float32)
    anchored = _anchor_to_nominal_span(raw, nominal_step_um)
    regularized, consistency_gain, consistency_strategy = _adaptive_increment_consistency_um(
        anchored,
        nominal_step_um,
    )
    nominal = np.arange(raw.size, dtype=np.float32) * float(nominal_step_um)
    nominal_span = max(float(nominal[-1] - nominal[0]), 1e-12)
    raw_span_ratio = abs(float(raw[-1] - raw[0])) / nominal_span
    median_confidence = float(np.median(np.asarray(confidence, dtype=np.float32)))
    reliability_gain = 1.0
    if method == "software_scatter_fit" and (raw_span_ratio < 0.65 or median_confidence < 1e-3):
        reliability_gain = 0.0
        regularized = nominal
    elif method == "software_carrier_phase":
        reliability_gain = 0.0 if raw_span_ratio < 0.95 else 0.5
        regularized = (nominal + reliability_gain * (regularized - nominal)).astype(np.float32)
    adaptive_correction = (regularized - raw).astype(np.float32)
    scan_steps = build_scan_step_result(
        regularized,
        nominal_step_um=nominal_step_um,
        use_monotone_for_analysis=False,
    )
    merged_diagnostics: dict[str, np.ndarray | float | int | str] = {
        "scan_positions_adaptive_correction_um": adaptive_correction.astype(np.float32),
        "scan_adaptive_strategy": f"endpoint_span_anchor+{consistency_strategy}",
        "scan_adaptive_consistency_gain": float(consistency_gain),
        "scan_adaptive_reliability_gain": float(reliability_gain),
        "scan_raw_span_ratio": float(raw_span_ratio),
        "scan_median_confidence": float(median_confidence),
    }
    if diagnostics:
        merged_diagnostics.update(diagnostics)
    return SoftwareScanStepEstimate(
        scan_steps=scan_steps,
        confidence=np.asarray(confidence, dtype=np.float32),
        phase_rad=np.asarray(phase_rad, dtype=np.float32),
        raw_positions_um=raw,
        method=method,
        diagnostics=merged_diagnostics,
    )


def _carrier_mask(
    shape: tuple[int, int],
    center_y: int,
    center_x: int,
    radius: int,
) -> tuple[slice, slice, np.ndarray]:
    y0 = max(0, int(center_y) - int(radius))
    y1 = min(shape[0], int(center_y) + int(radius) + 1)
    x0 = max(0, int(center_x) - int(radius))
    x1 = min(shape[1], int(center_x) + int(radius) + 1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dist2 = (yy - float(center_y)) ** 2 + (xx - float(center_x)) ** 2
    sigma = max(float(radius) / 2.0, 1.0)
    mask = np.exp(-0.5 * dist2 / (sigma * sigma)).astype(np.float32)
    return slice(y0, y1), slice(x0, x1), mask


def _carrier_positions_from_peak(
    spectra: np.ndarray,
    *,
    peak_y: int,
    peak_x: int,
    search_radius: int,
    k0_value: float,
    nominal_step_um: float,
    peak_frame_map: np.ndarray | None = None,
    envelope_radius_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按专利空间载波路径，从滤波复图的逐像素相位差估计扫描坐标。"""
    frame_count, height, width = spectra.shape
    y_slice, x_slice, mask = _carrier_mask((height, width), peak_y, peak_x, search_radius)
    phase = np.zeros(frame_count, dtype=np.float64)
    confidence = np.zeros(frame_count, dtype=np.float32)
    mean_amplitude = np.zeros(frame_count, dtype=np.float32)
    previous_field: np.ndarray | None = None
    previous_amplitude: np.ndarray | None = None
    for frame_index in range(frame_count):
        filtered = np.zeros((height, width), dtype=np.complex64)
        filtered[y_slice, x_slice] = spectra[frame_index, y_slice, x_slice] * mask
        field = np.fft.ifft2(np.fft.ifftshift(filtered)).astype(np.complex64)
        amplitude = np.abs(field).astype(np.float32)
        mean_amplitude[frame_index] = float(np.mean(amplitude))
        if previous_field is not None and previous_amplitude is not None:
            product = field * np.conj(previous_field)
            weights = amplitude * previous_amplitude
            threshold = float(np.percentile(weights, 60.0))
            valid = np.isfinite(weights) & (weights > threshold)
            if peak_frame_map is not None:
                local = np.abs(peak_frame_map - int(frame_index)) <= int(envelope_radius_frames)
                valid &= local
            if np.count_nonzero(valid) < 256:
                valid = np.isfinite(weights) & (weights > 0.0)
                if peak_frame_map is not None:
                    local = np.abs(peak_frame_map - int(frame_index)) <= max(int(envelope_radius_frames) * 2, 1)
                    valid &= local
            if np.count_nonzero(valid) < 256:
                valid = np.isfinite(weights) & (weights > threshold)
            if np.count_nonzero(valid) < 256:
                valid = np.isfinite(weights) & (weights > 0.0)
            angles = np.angle(product[valid]).astype(np.float32)
            valid_weights = weights[valid].astype(np.float32)
            unit = np.exp(1j * angles).astype(np.complex64)
            initial = np.sum(valid_weights * unit, dtype=np.complex128)
            residual = np.angle(unit * np.exp(-1j * float(np.angle(initial)))).astype(np.float32)
            scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
            keep_limit = max(2.5 * scale, float(np.percentile(np.abs(residual), 70.0)))
            keep = np.abs(residual) <= keep_limit
            if np.count_nonzero(keep) < 256:
                keep = np.ones_like(residual, dtype=bool)
            phasor = np.sum(valid_weights[keep] * unit[keep], dtype=np.complex128)
            denom = float(np.sum(valid_weights[keep], dtype=np.float64)) + 1e-12
            delta = float(np.angle(phasor))
            phase[frame_index] = phase[frame_index - 1] + delta
            confidence[frame_index] = float(np.abs(phasor) / denom)
        previous_field = field
        previous_amplitude = amplitude
    if frame_count > 1:
        confidence[0] = confidence[1]
    positions = _phase_to_stage_positions_um(phase.astype(np.float32), k0_value, nominal_step_um)
    return positions, phase.astype(np.float32), confidence, mean_amplitude


def estimate_positions_carrier_phase(
    cube: np.ndarray,
    *,
    k0_value: float,
    nominal_step_um: float,
    max_pixels: int = 160_000,
    search_radius: int = 5,
) -> SoftwareScanStepEstimate:
    """用空间载波相位估计 GFDA 扫描坐标。"""
    data = np.asarray(cube, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("cube must be a 3D array.")
    height, width, frame_count = data.shape
    yy, xx = _sample_pixels(height, width, max_pixels)
    sampled = data[yy, xx, :]
    frame_indices = np.unique(np.linspace(0, frame_count - 1, num=min(frame_count, 32), dtype=np.int32))
    sigma = max(1.0, min(sampled.shape[:2]) / 64.0)
    centered_sampled = sampled - np.mean(sampled, axis=2, keepdims=True)
    envelope_window = max(5, int(round(np.pi / max(abs(float(k0_value)) * float(nominal_step_um), 1e-6))))
    envelope_energy = uniform_filter1d(centered_sampled * centered_sampled, size=envelope_window, axis=2, mode="nearest")
    peak_frame_map = np.argmax(envelope_energy, axis=2).astype(np.int32)
    spectra = np.empty((frame_count, sampled.shape[0], sampled.shape[1]), dtype=np.complex64)
    power = np.zeros(sampled.shape[:2], dtype=np.float64)
    frame_index_set = set(int(index) for index in frame_indices)
    for frame_index in range(frame_count):
        frame = sampled[:, :, frame_index]
        high_pass = frame - gaussian_filter(frame, sigma=sigma)
        spectrum = np.fft.fftshift(np.fft.fft2(high_pass)).astype(np.complex64)
        spectra[frame_index] = spectrum
        if frame_index in frame_index_set:
            power += np.abs(spectrum) ** 2
    power = power / float(frame_indices.size)
    cy, cx = np.array(power.shape) // 2
    guard = max(4, min(power.shape) // 32)
    power[cy - guard : cy + guard + 1, cx - guard : cx + guard + 1] = 0.0
    candidate_centers: list[tuple[int, int]] = []
    candidate_power = power.copy()
    suppress_radius = max(2, int(search_radius))
    for _ in range(16):
        peak_y, peak_x = np.unravel_index(int(np.argmax(candidate_power)), candidate_power.shape)
        if float(candidate_power[peak_y, peak_x]) <= 0.0:
            break
        candidate_centers.append((int(peak_y), int(peak_x)))
        y0 = max(0, peak_y - suppress_radius)
        y1 = min(candidate_power.shape[0], peak_y + suppress_radius + 1)
        x0 = max(0, peak_x - suppress_radius)
        x1 = min(candidate_power.shape[1], peak_x + suppress_radius + 1)
        candidate_power[y0:y1, x0:x1] = 0.0
    if not candidate_centers:
        raise ValueError("Carrier-phase scan-step estimation did not find a spatial carrier peak.")

    nominal = np.arange(frame_count, dtype=np.float64) * float(nominal_step_um)
    nominal_span = max(float(nominal[-1] - nominal[0]), 1e-12)
    best_score = -np.inf
    best_index = 0
    best_phase = np.zeros(frame_count, dtype=np.float32)
    best_positions = nominal.astype(np.float32)
    best_confidence = np.zeros(frame_count, dtype=np.float32)
    best_amplitude = np.zeros(frame_count, dtype=np.float32)
    candidate_scores: list[float] = []
    candidate_spans: list[float] = []
    candidate_residual_stds: list[float] = []
    candidate_reversals: list[int] = []
    candidate_median_confidences: list[float] = []
    for candidate_index, (peak_y, peak_x) in enumerate(candidate_centers[:8]):
        candidate_positions, candidate_phase, candidate_confidence, candidate_amplitude = _carrier_positions_from_peak(
            spectra,
            peak_y=peak_y,
            peak_x=peak_x,
            search_radius=search_radius,
            k0_value=k0_value,
            nominal_step_um=nominal_step_um,
            peak_frame_map=peak_frame_map,
            envelope_radius_frames=3,
        )
        if not np.all(np.isfinite(candidate_positions)):
            continue
        span = float(candidate_positions[-1] - candidate_positions[0])
        corr = float(np.corrcoef(candidate_positions, nominal)[0, 1]) if frame_count >= 2 else 0.0
        if not np.isfinite(corr):
            corr = 0.0
        span_error = abs(span / nominal_span - 1.0)
        score = corr - span_error + 0.2 * float(np.nanmedian(candidate_confidence))
        candidate_scores.append(float(score))
        candidate_spans.append(float(span))
        candidate_residual_stds.append(float(np.std(candidate_positions.astype(np.float64) - nominal)))
        candidate_reversals.append(int(np.count_nonzero(np.diff(candidate_positions) <= 0.0)))
        candidate_median_confidences.append(float(np.nanmedian(candidate_confidence)))
        if score > best_score:
            best_score = score
            best_index = candidate_index
            best_phase = candidate_phase
            best_positions = candidate_positions
            best_confidence = candidate_confidence
            best_amplitude = candidate_amplitude

    peak_y, peak_x = candidate_centers[best_index]
    ky = TAU * (float(peak_y) - cy) / float(sampled.shape[0])
    kx = TAU * (float(peak_x) - cx) / float(sampled.shape[1])
    phase = best_phase
    amplitude = best_amplitude.astype(np.float32)
    confidence = _robust_normalize(best_confidence)
    confidence = uniform_filter1d(confidence, size=3, mode="nearest")
    positions = best_positions
    return _finalize_estimate(
        positions,
        confidence,
        phase,
        method="software_carrier_phase",
        nominal_step_um=nominal_step_um,
        diagnostics={
            "scan_carrier_kx": float(kx),
            "scan_carrier_ky": float(ky),
            "scan_carrier_candidate_count": int(min(len(candidate_centers), 8)),
            "scan_carrier_candidate_score": float(best_score),
            "scan_carrier_amplitude": amplitude.astype(np.float32),
            "scan_carrier_candidate_scores": np.asarray(candidate_scores, dtype=np.float32),
            "scan_carrier_candidate_spans_um": np.asarray(candidate_spans, dtype=np.float32),
            "scan_carrier_candidate_residual_std_um": np.asarray(candidate_residual_stds, dtype=np.float32),
            "scan_carrier_candidate_reversals": np.asarray(candidate_reversals, dtype=np.float32),
            "scan_carrier_candidate_median_confidence": np.asarray(candidate_median_confidences, dtype=np.float32),
        },
    )


def estimate_positions_scatter_fit(
    cube: np.ndarray,
    *,
    k0_value: float,
    nominal_step_um: float,
    window_size: int,
    fitting_method: str,
    unwrap_method: str,
    window_name: str,
    window_alpha: float,
    zero_padding_mode: str,
    max_pixels: int = 180_000,
    envelope_radius_frames: int = 3,
) -> SoftwareScanStepEstimate:
    """用强度-高度散点拟合法估计 GFDA 扫描坐标。"""
    data = np.asarray(cube, dtype=np.float32)
    if data.ndim != 3:
        raise ValueError("cube must be a 3D array.")
    baseline = analyze_cube_baseline(
        intensity_data=data,
        step_size=nominal_step_um,
        window_size=window_size,
        fitting_method=fitting_method,
        unwrap_method=unwrap_method,
        fixed_k0_value=k0_value,
        sample_positions_um=None,
        window_name=window_name,
        window_alpha=window_alpha,
        zero_padding_mode=zero_padding_mode,
    )
    theta = np.asarray(baseline["theta_map"], dtype=np.float32)
    weights_map = np.asarray(baseline["peak_amplitude_map"], dtype=np.float32)
    valid = np.isfinite(theta) & np.isfinite(weights_map) & (weights_map > 0.0)
    if not np.any(valid):
        raise ValueError("Scatter-fit scan-step estimation found no valid FDA baseline pixels.")
    valid_y_all, valid_x_all = np.nonzero(valid)
    quality_all = weights_map[valid_y_all, valid_x_all]
    keep = min(int(max_pixels), valid_y_all.size)
    if keep < valid_y_all.size:
        chosen = np.argpartition(quality_all, -keep)[-keep:]
        valid_y_all = valid_y_all[chosen]
        valid_x_all = valid_x_all[chosen]
        quality_all = quality_all[chosen]

    curves = data[valid_y_all, valid_x_all, :].astype(np.float32)
    centered_curves = curves - np.mean(curves, axis=1, keepdims=True)
    # 原始强度峰会被多周期条纹误导；用局部能量近似白光包络峰来选择散点邻域。
    envelope_window = max(5, int(round(np.pi / max(abs(float(k0_value)) * float(nominal_step_um), 1e-6))))
    envelope_energy = uniform_filter1d(centered_curves * centered_curves, size=envelope_window, axis=1, mode="nearest")
    peak_frame = np.argmax(envelope_energy, axis=1).astype(np.int32)
    theta_all = theta[valid_y_all, valid_x_all].astype(np.float64)
    theta_center = float(np.median(theta_all))
    theta_scale = max(float(np.percentile(theta_all, 95.0) - np.percentile(theta_all, 5.0)), 1e-6)
    theta_norm_all = (theta_all - theta_center) / theta_scale
    cos_t_all = np.cos(theta_all)
    sin_t_all = np.sin(theta_all)
    base_weights_all = quality_all.astype(np.float64)
    base_weights_all = base_weights_all / max(float(np.nanmedian(base_weights_all)), 1e-12)
    base_weights_all = np.clip(base_weights_all, 0.05, 10.0)

    frame_count = data.shape[2]
    pair_phasors_by_frame = np.empty((frame_count, 4), dtype=np.complex128)
    residual_confidence = np.empty(frame_count, dtype=np.float32)
    for frame_index in range(frame_count):
        local = np.abs(peak_frame - int(frame_index)) <= int(envelope_radius_frames)
        if np.count_nonzero(local) < 256:
            local = np.ones_like(peak_frame, dtype=bool)
        theta_norm = theta_norm_all[local]
        cos_t = cos_t_all[local]
        sin_t = sin_t_all[local]
        weights = base_weights_all[local]
        # Eq. 7: U = A + B'(theta) cos(theta + phi)，
        # 用二次包络 B'=B+Cθ+Dθ² 展开成线性最小二乘。
        design = np.column_stack(
            [
                np.ones(theta_norm.size),
                cos_t,
                theta_norm * cos_t,
                theta_norm * theta_norm * cos_t,
                -sin_t,
                -theta_norm * sin_t,
                -(theta_norm * theta_norm) * sin_t,
            ]
        )
        weighted_design = design * np.sqrt(weights)[:, None]
        values = curves[local, frame_index].astype(np.float64)
        rhs = values * np.sqrt(weights)
        coeffs, *_ = np.linalg.lstsq(weighted_design, rhs, rcond=None)
        # 常数/线性/二次包络项分别对应专利中的 d/a、e/b、f/c 相位估计。
        # 第 4 列是幅值加权合成候选，最终在整段扫描上统一选择稳定通道。
        cos_coeff = coeffs[1:4]
        sin_coeff = coeffs[4:7]
        pair_phasors = cos_coeff + 1j * sin_coeff
        pair_amplitudes = np.maximum(np.abs(pair_phasors), 1e-12)
        pair_phasors_by_frame[frame_index, :3] = pair_phasors
        pair_phasors_by_frame[frame_index, 3] = np.sum(pair_phasors * pair_amplitudes)
        model_amp = float(np.max(pair_amplitudes))
        residual = values - design @ coeffs
        noise = float(np.sqrt(np.mean(residual**2))) + 1e-12
        residual_confidence[frame_index] = model_amp / noise

    nominal = np.arange(frame_count, dtype=np.float64) * float(nominal_step_um)
    nominal_span = max(float(nominal[-1] - nominal[0]), 1e-12)
    best_score = -np.inf
    best_channel = 0
    best_phase = np.zeros(frame_count, dtype=np.float32)
    best_positions = nominal.astype(np.float32)
    for channel in range(pair_phasors_by_frame.shape[1]):
        channel_phase = np.unwrap(np.angle(pair_phasors_by_frame[:, channel])).astype(np.float32)
        channel_positions = _phase_to_stage_positions_um(channel_phase, k0_value, nominal_step_um)
        span = float(channel_positions[-1] - channel_positions[0])
        corr = float(np.corrcoef(channel_positions, nominal)[0, 1]) if frame_count >= 2 else 0.0
        if not np.isfinite(corr):
            corr = 0.0
        span_error = abs(span / nominal_span - 1.0)
        amplitude_score = float(np.nanmedian(np.abs(pair_phasors_by_frame[:, channel])))
        score = corr - span_error + 1e-6 * amplitude_score
        if score > best_score:
            best_score = score
            best_channel = channel
            best_phase = channel_phase
            best_positions = channel_positions

    phase = best_phase
    confidence = _robust_normalize(residual_confidence)
    confidence = uniform_filter1d(confidence, size=3, mode="nearest")
    positions = best_positions
    return _finalize_estimate(
        positions,
        confidence,
        phase,
        method="software_scatter_fit",
        nominal_step_um=nominal_step_um,
        diagnostics={
            "scan_scatter_pixel_count": int(theta_all.size),
            "scan_scatter_envelope_radius_frames": int(envelope_radius_frames),
            "scan_scatter_confidence": confidence.astype(np.float32),
            "scan_scatter_phase_channel": int(best_channel),
            "scan_scatter_channel_score": float(best_score),
        },
    )


def estimate_software_scan_positions(
    cube: np.ndarray,
    *,
    method: str,
    k0_value: float,
    nominal_step_um: float,
    window_size: int,
    fitting_method: str,
    unwrap_method: str,
    window_name: str,
    window_alpha: float,
    zero_padding_mode: str,
) -> SoftwareScanStepEstimate:
    """GFDA 软件步长估计统一入口。"""
    method_key = str(method).strip().lower()
    if method_key == "gfda_carrier_phase":
        return estimate_positions_carrier_phase(
            cube,
            k0_value=k0_value,
            nominal_step_um=nominal_step_um,
        )
    if method_key == "gfda_scatter_fit":
        return estimate_positions_scatter_fit(
            cube,
            k0_value=k0_value,
            nominal_step_um=nominal_step_um,
            window_size=window_size,
            fitting_method=fitting_method,
            unwrap_method=unwrap_method,
            window_name=window_name,
            window_alpha=window_alpha,
            zero_padding_mode=zero_padding_mode,
        )
    raise ValueError(f"Unsupported GFDA software scan-step method: {method}")
