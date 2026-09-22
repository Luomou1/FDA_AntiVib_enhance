import numpy as np
import pytest

from app.core.fda_baseline import analyze_cube_baseline
from app.core.metro_unwrap import unwrap_metro
from app.core.pixel_analysis import build_pixel_analysis


@pytest.mark.parametrize("length", [128, 256, 512])
def test_metro_recovers_height_beyond_itoh_half_record(length):
    z = np.arange(128) * 0.05
    height = 4.0
    signal = 100 + 40 * np.exp(-0.5 * ((z - height) / 0.35) ** 2) * np.cos(2 * 11.15 * (z - height))
    result = analyze_cube_baseline(
        signal[None, None, :], 0.05, 3, "simple", "metro",
        fixed_k0_value=11.15, window_name="none", zero_padding_mode=str(length),
    )
    np.testing.assert_allclose(result["h_prime"], [[4000]], atol=0.1)


def test_odd_length_and_clipped_bins_restore_slope():
    bins = np.array([0, 0, 1, 2, 3, 4, 5])
    k = bins * np.pi / (129 * 0.05)
    expected = -2 * 4.0 * k + 0.2
    result = unwrap_metro(np.angle(np.exp(1j * expected)), np.array([1, 1, 2, 5, 2, 1, 1]), bins)
    np.testing.assert_allclose(np.diff(expected), np.diff(result), atol=1e-12)


def test_invalid_amplitude_stops_outward_tracking():
    result = unwrap_metro(np.zeros(5), np.array([1, 0, 5, 2, 0]), np.arange(5))
    assert np.array_equal([False, False, True, True, False], np.isfinite(result))


def test_pixel_diagnostics_accept_metro():
    z = np.arange(128) * 0.05
    signal = np.exp(-0.5 * ((z - 4) / 0.35) ** 2) * np.cos(2 * 11.15 * (z - 4))
    result = build_pixel_analysis(
        signal[None, None, :], 0, 0, 0.05, unwrap_method="metro",
        window_size=3, fitting_method="simple", global_k0_value=11.15,
        window_name="none", zero_padding_mode="none",
    )
    slope = np.polyfit(result["fit_mask_k_x"], result["fit_mask_phase_y"], 1)[0]
    assert abs(slope + 8) < 0.001


@pytest.mark.parametrize("method", ["simple", "weighted", "quadratic"])
def test_zero_signal_is_invalid_without_crashing(method):
    result = analyze_cube_baseline(
        np.zeros((1, 1, 128)), 0.05, 3, method, "metro",
        fixed_k0_value=11.15, window_name="none", zero_padding_mode="none",
    )
    assert np.isnan(result["h_prime"]).all()


def test_exact_pi_keeps_original_boundary_direction():
    result = unwrap_metro(np.array([0., 0.]), np.array([2., 1.]), np.array([0, 1]))
    np.testing.assert_allclose([0., 0.], result, atol=1e-12)
