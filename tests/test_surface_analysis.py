from __future__ import annotations

import numpy as np
import pytest

import app.core.surface_analysis as surface_analysis
from app.core.surface_analysis import (
    analyze_plane,
    analyze_step,
    compute_step_height,
    load_height_matrix,
)


def test_load_height_matrix_repairs_nan_with_local_values(tmp_path) -> None:
    path = tmp_path / "height.txt"
    matrix = np.arange(25, dtype=float).reshape(5, 5)
    matrix[2, 2] = np.nan
    np.savetxt(path, matrix)

    loaded = load_height_matrix(path)

    assert loaded.shape == (5, 5)
    assert np.isfinite(loaded).all()
    assert loaded[2, 2] == pytest.approx(12.0)


def test_load_height_matrix_rejects_non_matrix_input(tmp_path) -> None:
    path = tmp_path / "height.txt"
    np.savetxt(path, np.arange(8, dtype=float))

    with pytest.raises(ValueError, match="二维"):
        load_height_matrix(path)


@pytest.mark.parametrize("method", ["simple", "robust", "quadratic"])
def test_plane_analysis_removes_known_tilt(method: str) -> None:
    rows, cols = 42, 54
    y, x = np.mgrid[:rows, :cols]
    local_shape = 2.5 * np.exp(-((x - 27.0) ** 2 + (y - 20.0) ** 2) / 90.0)
    height = 0.8 * x - 0.35 * y + 120.0 + local_shape
    height[5, 7] += 500.0

    result = analyze_plane(height, method=method)

    assert result.original.shape == height.shape
    assert result.processed.shape == height.shape
    assert np.isfinite(result.processed).all()
    assert result.stats["height_range"] < 20.0
    assert result.stats["std"] < 3.0
    assert result.outlier_count >= 1


def _build_step_surface() -> np.ndarray:
    rows, cols = 64, 80
    y, x = np.mgrid[:rows, :cols]
    step = np.where(x >= 40, 85.0, 0.0)
    ripple = 0.7 * np.sin(x / 7.0) + 0.4 * np.cos(y / 9.0)
    data = 0.22 * x - 0.14 * y + 30.0 + step + ripple
    data[15, 20] += 220.0
    data[48, 62] -= 180.0
    return data


def test_step_analysis_levels_segments_and_supports_both_modes() -> None:
    data = _build_step_surface()
    points = ((8, 8), (50, 12), (20, 32))

    raw_result = analyze_step(data, points=points, denoise=False)
    denoised_result = analyze_step(data, points=points, denoise=True)

    assert set(np.unique(raw_result.cluster_map)) == {1, 2}
    assert np.array_equal(raw_result.cluster_map, denoised_result.cluster_map)
    assert denoised_result.noise_count >= 2
    assert np.std(denoised_result.processed[denoised_result.cluster_map == 1]) < np.std(
        raw_result.processed[raw_result.cluster_map == 1]
    )

    region_one = (48, 10, 72, 38)
    region_two = (8, 10, 32, 38)
    measurement = compute_step_height(denoised_result.processed, region_one, region_two)

    assert measurement.step_height == pytest.approx(85.0, abs=5.0)
    assert measurement.region_one_mean > measurement.region_two_mean


def test_step_analysis_rejects_collinear_reference_points() -> None:
    data = _build_step_surface()

    with pytest.raises(ValueError, match="不共线"):
        analyze_step(data, points=((5, 5), (10, 10), (15, 15)), denoise=False)


def test_matlab_hist_and_smooth_match_legacy_matlab_behavior() -> None:
    counts, centers = surface_analysis._matlab_hist(np.arange(11, dtype=float), bins=5)
    smoothed = surface_analysis._matlab_smooth_three(counts)

    assert centers == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0])
    assert counts == pytest.approx([3.0, 2.0, 2.0, 2.0, 2.0])
    assert smoothed == pytest.approx([3.0, 7.0 / 3.0, 2.0, 2.0, 2.0])


def test_denoised_layer_statistics_use_processed_sample_standard_deviation() -> None:
    data = _build_step_surface()
    result = analyze_step(data, points=((8, 8), (50, 12), (20, 32)), denoise=True)

    low_values = result.processed[result.cluster_map == 1]
    high_values = result.processed[result.cluster_map == 2]

    assert result.layer_stats["low"]["std"] == pytest.approx(np.std(low_values, ddof=1))
    assert result.layer_stats["high"]["std"] == pytest.approx(np.std(high_values, ddof=1))
