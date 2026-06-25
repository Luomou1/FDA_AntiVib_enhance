from __future__ import annotations

import numpy as np

from app.core.adaptive_window import build_analysis_window, normalize_window_name
from app.core.kernel import _compute_windowed_spectrum


def _make_shifted_interferogram(sample_count: int, center: float, left_sigma: float, right_sigma: float) -> np.ndarray:
    """生成一个中心可偏移、左右衰减可不同的合成干涉包络。"""
    index = np.arange(sample_count, dtype=np.float32)
    sigma = np.where(index <= center, left_sigma, right_sigma)
    envelope = np.exp(-0.5 * ((index - center) / sigma) ** 2)
    carrier = np.cos(0.42 * index + 0.3)
    return (envelope * carrier).astype(np.float32)


def test_adaptive_hann_window_peak_tracks_interferogram_center() -> None:
    """自适应 Hann 窗的最大权重应落在干涉包络中心，而不是采样段几何中心。"""
    sample_count = 96
    signal_center = 70.0
    curve = _make_shifted_interferogram(sample_count, signal_center, left_sigma=8.0, right_sigma=8.0)

    static_window = build_analysis_window(curve[None, :], "hann")[0]
    adaptive_window = build_analysis_window(curve[None, :], "adaptive_hann")[0]

    assert abs(int(np.argmax(static_window)) - sample_count // 2) <= 1
    assert abs(int(np.argmax(adaptive_window)) - int(signal_center)) <= 1


def test_adaptive_hamming_uses_different_left_and_right_widths() -> None:
    """不对称包络应得到不对称窗宽，避免短边过宽或长边被截断。"""
    sample_count = 128
    signal_center = 52.0
    curve = _make_shifted_interferogram(sample_count, signal_center, left_sigma=5.0, right_sigma=20.0)

    window = build_analysis_window(curve[None, :], "adaptive_hamming")[0]
    strong = np.flatnonzero(window > 0.2)
    left_width = int(signal_center) - int(strong[0])
    right_width = int(strong[-1]) - int(signal_center)

    assert right_width > left_width * 2
    assert abs(int(np.argmax(window)) - int(signal_center)) <= 2


def test_static_window_names_keep_legacy_behavior() -> None:
    """新增自适应窗不应破坏既有静态窗名称和 hanning 别名。"""
    sample_count = 17
    curves = np.ones((2, sample_count), dtype=np.float32)

    assert normalize_window_name("Hanning") == "hann"
    np.testing.assert_allclose(build_analysis_window(curves, "none"), np.ones_like(curves))
    np.testing.assert_allclose(build_analysis_window(curves, "hamming")[0], np.hamming(sample_count).astype(np.float32))
    np.testing.assert_allclose(build_analysis_window(curves, "hann")[0], np.hanning(sample_count).astype(np.float32))


def test_adaptive_window_is_used_by_windowed_spectrum_entry() -> None:
    """频谱公共入口应能直接使用自适应窗，保证 FDA 和自动 K0 共用同一实现。"""
    sample_count = 64
    first = _make_shifted_interferogram(sample_count, center=20.0, left_sigma=6.0, right_sigma=8.0)
    second = _make_shifted_interferogram(sample_count, center=46.0, left_sigma=10.0, right_sigma=5.0)
    curves = np.stack([first, second], axis=0).astype(np.float32)

    amplitude, phase, k_axis = _compute_windowed_spectrum(
        curves,
        step_size=0.05,
        window_name="adaptive_hann",
        fft_length=128,
    )

    assert amplitude.shape == (2, 65)
    assert phase.shape == amplitude.shape
    assert k_axis.shape == (65,)
    assert np.all(np.isfinite(amplitude))
