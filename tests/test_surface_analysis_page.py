from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.gui.surface_analysis_page import (
    PlaneAnalysisPage,
    ResultFigureCanvas,
    Surface3DCanvas,
    StepAnalysisPage,
    StepSelectionCanvas,
    _downsample_surface,
)


def test_plane_and_step_pages_keep_independent_file_inputs(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    plane_page.file_edit.setText("C:/data/plane.txt")
    step_page.file_edit.setText("C:/data/step.txt")

    assert plane_page.file_edit.text().endswith("plane.txt")
    assert step_page.file_edit.text().endswith("step.txt")
    assert plane_page.file_edit is not step_page.file_edit


def test_step_page_exposes_denoise_and_no_denoise_modes(qtbot) -> None:
    page = StepAnalysisPage()
    qtbot.addWidget(page)

    assert page.mode_combo.count() == 2
    assert page.mode_combo.itemData(0) == "denoise"
    assert page.mode_combo.itemData(1) == "raw"
    assert not page.analyze_button.isEnabled()


def test_analysis_pages_expose_accessible_primary_controls(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    assert plane_page.choose_button.accessibleName() == "选择平面高度文件"
    assert plane_page.analyze_button.accessibleName() == "开始平面分析"
    assert step_page.choose_button.accessibleName() == "选择台阶高度文件"
    assert step_page.analyze_button.accessibleName() == "开始台阶分析"


def test_result_charts_use_one_tab_per_chart(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    assert [plane_page.tabs.tabText(index) for index in range(plane_page.tabs.count())] == [
        "原始三维",
        "处理后三维",
        "二维高度图",
        "一维轮廓",
    ]
    assert [step_page.tabs.tabText(index) for index in range(step_page.tabs.count())] == [
        "交互选择",
        "原始三维",
        "分层结果",
        "处理后三维",
        "分层标准差",
        "一维轮廓",
    ]


def test_surface_preview_is_capped_to_a_small_render_grid() -> None:
    data = np.zeros((1024, 1280), dtype=float)

    reduced, x, y = _downsample_surface(data)

    assert reduced.shape[0] <= 160
    assert reduced.shape[1] <= 160
    assert x.size == reduced.shape[1]
    assert y.size == reduced.shape[0]


def test_surface_canvas_can_reuse_original_z_limits(qtbot) -> None:
    canvas = Surface3DCanvas(render_enabled=False)
    qtbot.addWidget(canvas)

    canvas.draw_surface(np.ones((8, 10), dtype=float) * 12.0, "处理后三维", z_limits=(0.0, 100.0))

    assert canvas._last_display_bounds is not None
    assert canvas._last_display_bounds[4:] == (0.0, 100.0)


def test_profile_canvas_reports_two_clicked_points_delta(qtbot) -> None:
    canvas = ResultFigureCanvas()
    qtbot.addWidget(canvas)
    data = np.array([[0.0, 2.0, 5.0, 9.0, 13.0]], dtype=float)

    canvas.draw_profile(data, "row", 0)
    axes = canvas.figure.axes[0]
    canvas._on_profile_click(SimpleNamespace(inaxes=axes, xdata=1.0, ydata=2.0))
    canvas._on_profile_click(SimpleNamespace(inaxes=axes, xdata=4.0, ydata=13.0))

    overlay_text = "\n".join(
        artist.get_text() for artist in canvas._profile_artists if hasattr(artist, "get_text")
    )
    assert "ΔH=11.000 nm" in overlay_text


def test_step_selection_canvas_displays_first_matrix_row_at_top(qtbot) -> None:
    canvas = StepSelectionCanvas()
    qtbot.addWidget(canvas)
    canvas.start_point_selection(np.arange(24, dtype=float).reshape(4, 6))

    assert canvas.axes.images[0].origin == "upper"


def test_profile_spinboxes_are_configured_for_fast_adjustment(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    assert plane_page.profile_index.isAccelerated()
    assert step_page.profile_index.isAccelerated()
    assert not plane_page.profile_index.keyboardTracking()
    assert not step_page.profile_index.keyboardTracking()
    assert plane_page.profile_index.buttonSymbols().name == "NoButtons"
    assert step_page.profile_index.buttonSymbols().name == "NoButtons"
    assert plane_page.profile_index.minimumWidth() >= 112
    assert step_page.profile_index.minimumWidth() >= 112


def test_plane_and_step_use_pyvista_for_3d_surfaces(qtbot) -> None:
    plane_page = PlaneAnalysisPage()
    step_page = StepAnalysisPage()
    qtbot.addWidget(plane_page)
    qtbot.addWidget(step_page)

    assert isinstance(plane_page.raw_surface_canvas, Surface3DCanvas)
    assert isinstance(plane_page.processed_surface_canvas, Surface3DCanvas)
    assert isinstance(step_page.raw_surface_canvas, Surface3DCanvas)
    assert isinstance(step_page.processed_surface_canvas, Surface3DCanvas)


def test_plane_page_runs_one_independent_height_file(qtbot, tmp_path) -> None:
    y, x = np.mgrid[:24, :30]
    data = 0.4 * x - 0.2 * y + 50.0 + 0.5 * np.sin(x / 4.0)
    path = tmp_path / "plane.txt"
    np.savetxt(path, data)

    page = PlaneAnalysisPage()
    qtbot.addWidget(page)
    page.file_edit.setText(str(path))
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)

    assert page._result is not None
    assert page._result.original.shape == data.shape
    assert "高度范围" in page.metrics.toPlainText()
    assert page._rendered_tabs == {2}


def test_step_page_runs_and_measures_regions_with_same_three_points(qtbot, tmp_path) -> None:
    y, x = np.mgrid[:36, :48]
    data = 0.1 * x - 0.08 * y + np.where(x >= 24, 40.0, 0.0)
    path = tmp_path / "step.txt"
    np.savetxt(path, data)

    page = StepAnalysisPage()
    qtbot.addWidget(page)
    page.file_edit.setText(str(path))
    page._load_for_selection()
    page.selection_canvas._points = [(5, 5), (28, 8), (12, 20)]
    page._on_points_changed(page.selection_canvas.points)
    page._run_analysis()
    qtbot.waitUntil(lambda: page._result is not None and page._analysis_thread is None, timeout=8000)

    assert page._result is not None
    assert page.analyze_button.isEnabled()
    assert page._rendered_tabs == set()

    page._on_regions_changed(((28, 5, 42, 28), (4, 5, 18, 28)))

    assert page._measurement is not None
    assert page._measurement.step_height > 30.0
    assert "台阶平均高度差" in page.metrics.toPlainText()
    page._render_current_tab(2)
    assert len(page.layer_map_canvas.figure.axes[0].collections) > 0
