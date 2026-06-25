from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget

from app.gui.fringe_profile_window import FringeProfileWindow
from app.gui.pixel_window import PixelAnalysisWindow


def test_phasegap_profile_titles_and_ylabels_are_chinese(qtbot) -> None:
    window = FringeProfileWindow()
    qtbot.addWidget(window)
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagnostic_maps = {
        "phi0_map": data,
        "theta_map": data,
        "theta_map_smoothed": data,
        "final_height_phase_map": data,
        "phase_gap_raw": data,
        "phase_gap_final": data,
        "merit_map": data,
        "confidence_map": data,
        "peak_amplitude_map": data,
    }

    window.update_profiles(
        source_map=data,
        x=1,
        y=1,
        layer_key="phi0_map",
        diagnostic_maps=diagnostic_maps,
    )

    tab_titles = [window.tabs.tabText(index) for index in range(window.tabs.count())]
    assert "相位剖面" in tab_titles
    assert "相干剖面" in tab_titles
    assert "最终高度剖面（以相位为单位）" in tab_titles
    assert window._single_profile_axes[("phi0_map", "x")].get_ylabel() == "相位 (cycles)"
    assert window._single_profile_axes[("theta_map", "x")].get_ylabel() == "相位 (cycles)"
    assert window._single_profile_axes[("final_height_phase_map", "x")].get_ylabel() == "相位 (cycles)"
    assert window._single_profile_axes[("phase_gap_final", "x")].get_ylabel() == "相位差 (cycles)"
    assert window._diagnostic_axes["phi0_map"].get_xlabel() == "X位置(像素)"
    assert window._diagnostic_axes["phi0_map"].get_ylabel() == "Y位置(像素)"
    assert window.x_axes.get_xlabel() == "X位置(像素)"
    assert window.y_axes.get_xlabel() == "Y位置(像素)"
    assert window._single_profile_axes[("phi0_map", "x")].get_xlabel() == "X位置(像素)"
    assert window._single_profile_axes[("phi0_map", "y")].get_xlabel() == "Y位置(像素)"
    assert window._multi_profile_axes[("phi0_map", "x")].get_xlabel() == "X位置(像素)"
    assert window._multi_profile_axes[("phi0_map", "y")].get_xlabel() == "Y位置(像素)"
    assert window.x_axes.get_xlim() == (0.0, 3.0)
    assert window.y_axes.get_xlim() == (0.0, 2.0)
    assert window._single_profile_axes[("phi0_map", "x")].get_xlim() == (0.0, 3.0)
    assert window._single_profile_axes[("phi0_map", "y")].get_xlim() == (0.0, 2.0)
    assert window._multi_profile_axes[("phi0_map", "x")].get_xlim() == (0.0, 3.0)
    assert window._multi_profile_axes[("phi0_map", "y")].get_xlim() == (0.0, 2.0)


def test_auxiliary_plot_windows_are_parented_tool_windows(qtbot) -> None:
    parent = QWidget()
    fringe_window = FringeProfileWindow(parent)
    pixel_window = PixelAnalysisWindow(parent)
    qtbot.addWidget(parent)
    qtbot.addWidget(fringe_window)
    qtbot.addWidget(pixel_window)

    assert fringe_window.parent() is parent
    assert pixel_window.parent() is parent
    assert fringe_window.windowFlags() & Qt.Tool
    assert pixel_window.windowFlags() & Qt.Tool


def test_phasegap_profile_window_can_jump_to_typed_pixel_without_wheel_changes(qtbot) -> None:
    class IgnoredWheelEvent:
        def __init__(self) -> None:
            self.ignored = False

        def ignore(self) -> None:
            self.ignored = True

    window = FringeProfileWindow()
    qtbot.addWidget(window)
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagnostic_maps = {
        "phi0_map": data,
        "theta_map": data,
        "theta_map_smoothed": data,
        "final_height_phase_map": data,
        "phase_gap_raw": data,
        "phase_gap_final": data,
        "merit_map": data,
        "confidence_map": data,
        "peak_amplitude_map": data,
    }
    window.update_profiles(
        source_map=data,
        x=1,
        y=1,
        layer_key="phi0_map",
        diagnostic_maps=diagnostic_maps,
    )

    assert window.x_coordinate_spin.maximum() == 3
    assert window.y_coordinate_spin.maximum() == 2
    assert window.x_coordinate_spin.value() == 1
    assert window.y_coordinate_spin.value() == 1

    window.x_coordinate_spin.setValue(3)
    window.y_coordinate_spin.setValue(2)
    window.jump_to_pixel_button.click()

    assert "x=3, y=2" in window.header_label.text()
    assert "X=3, Y=2" in window.coordinate_status_label.text()
    assert "value=11.000000" in window.coordinate_status_label.text()

    wheel_event = IgnoredWheelEvent()
    window.x_coordinate_spin.wheelEvent(wheel_event)

    assert wheel_event.ignored
    assert window.x_coordinate_spin.value() == 3


def test_pixel_popup_ylabels_are_chinese_and_keep_units(qtbot) -> None:
    window = PixelAnalysisWindow()
    qtbot.addWidget(window)
    sample_axis = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    payload = {
        "x": 2,
        "y": 3,
        "signal_x": sample_axis,
        "signal_raw_y": sample_axis,
        "signal_dc_y": sample_axis,
        "k_x": sample_axis,
        "amplitude_y": sample_axis + 1.0,
        "k0_x": 0.5,
        "k0_y": 1.5,
        "phase_raw_y": sample_axis,
        "phase_unwrapped_y": sample_axis,
    }

    window.update_analysis(payload, layer_name="phi0", fitting_method="simple", unwrap_method="global")

    assert window.raw_axes.get_title() == "相干剖面"
    assert window.dc_axes.get_title() == "相干剖面（已去除 DC）"
    assert window.phase_unwrapped_axes.get_title() == "相位剖面"
    assert window.raw_axes.get_ylabel() == "强度 (a.u.)"
    assert window.dc_axes.get_ylabel() == "强度 (a.u.)"
    assert window.amp_axes.get_ylabel() == "相对振幅 (a.u.)"
    assert window.phase_raw_axes.get_ylabel() == "相位 (cycles)"
    assert window.phase_unwrapped_axes.get_ylabel() == "相位 (cycles)"
    assert window.raw_axes.get_xlim() == (0.0, 1.0)
    assert window.dc_axes.get_xlim() == (0.0, 1.0)
    assert window.amp_axes.get_xlim() == (0.0, 1.0)
    assert window.phase_raw_axes.get_xlim() == (0.0, 1.0)
    assert window.phase_unwrapped_axes.get_xlim() == (0.0, 1.0)


def test_local_unwrap_phase_profile_uses_full_data_range_without_padding(qtbot) -> None:
    window = PixelAnalysisWindow()
    qtbot.addWidget(window)
    sample_axis = np.linspace(0.0, 4.0, 5, dtype=np.float32)
    payload = {
        "x": 2,
        "y": 3,
        "signal_x": sample_axis,
        "signal_raw_y": sample_axis,
        "signal_dc_y": sample_axis,
        "k_x": sample_axis,
        "amplitude_y": sample_axis + 1.0,
        "k0_x": 2.0,
        "k0_y": 3.0,
        "phase_raw_y": sample_axis,
        "phase_unwrapped_y": sample_axis,
        "fit_mask_k_x": sample_axis[1:4],
        "fit_mask_phase_y": sample_axis[1:4],
        "fit_k_x": sample_axis[1:4],
        "fit_phase_y": sample_axis[1:4],
    }

    window.update_analysis(payload, layer_name="phi0", fitting_method="simple", unwrap_method="itoh")

    assert window.phase_unwrapped_axes.get_xlim() == (0.0, 4.0)
