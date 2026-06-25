from __future__ import annotations

import numpy as np
from PIL import Image

from app.pipeline.io import collect_image_files, load_intensity_cube


def _save_uint16(path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint16, copy=False)).save(path)


def test_collect_image_files_includes_tiff(tmp_path) -> None:
    _save_uint16(tmp_path / "frame2.tiff", np.zeros((2, 2), dtype=np.uint16))
    _save_uint16(tmp_path / "frame10.tif", np.zeros((2, 2), dtype=np.uint16))
    _save_uint16(tmp_path / "frame1.png", np.zeros((2, 2), dtype=np.uint16))

    files = collect_image_files(tmp_path)

    assert [path.name for path in files] == ["frame1.png", "frame2.tiff", "frame10.tif"]


def test_load_intensity_cube_mono12_uint16_restores_left_aligned_12bit_values(tmp_path) -> None:
    first = np.array([[0, 17], [2048, 4095]], dtype=np.uint16)
    second = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    first_path = tmp_path / "frame1.png"
    second_path = tmp_path / "frame2.tiff"
    _save_uint16(first_path, first << 4)
    _save_uint16(second_path, second << 4)

    cube = load_intensity_cube([first_path, second_path], image_intensity_mode="mono12_uint16")

    assert cube.dtype == np.float32
    assert cube.shape == (2, 2, 2)
    np.testing.assert_array_equal(cube[:, :, 0], first.astype(np.float32))
    np.testing.assert_array_equal(cube[:, :, 1], second.astype(np.float32))


def test_load_intensity_cube_mono12_uint16_uses_high_12_bits(tmp_path) -> None:
    path = tmp_path / "frame1.png"
    _save_uint16(path, np.array([[0x001F, 0xFFF0]], dtype=np.uint16))

    cube = load_intensity_cube([path], image_intensity_mode="mono12_uint16")

    np.testing.assert_array_equal(cube[:, :, 0], np.array([[1.0, 4095.0]], dtype=np.float32))


def test_load_intensity_cube_legacy_mode_keeps_8bit_compatibility(tmp_path) -> None:
    path = tmp_path / "frame1.png"
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(path)

    cube = load_intensity_cube([path])

    assert cube.dtype == np.float32
    np.testing.assert_array_equal(cube[:, :, 0], np.array([[0.0, 255.0]], dtype=np.float32))
