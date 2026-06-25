from __future__ import annotations

"""平面分析和台阶分析一级页面。"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from pyvistaqt import QtInteractor
from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.surface_analysis import (
    PlaneAnalysisResult,
    StepAnalysisResult,
    StepMeasurement,
    analyze_plane,
    analyze_step,
    compute_step_height,
    load_height_matrix,
)
from app.gui.mpl_font import configure_matplotlib_fonts
from app.gui.widgets import SectionHeader
from app.plotting.preview import build_normalized_surface_grid

FONT_PROP = configure_matplotlib_fonts()


def _downsample_surface(data: np.ndarray, max_points: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = data.shape
    row_step = max(1, int(np.ceil(rows / max_points)))
    col_step = max(1, int(np.ceil(cols / max_points)))
    reduced = data[::row_step, ::col_step]
    y = np.arange(0, rows, row_step, dtype=np.float64)[: reduced.shape[0]]
    x = np.arange(0, cols, col_step, dtype=np.float64)[: reduced.shape[1]]
    return reduced, x, y


def _profile_data(
    data: np.ndarray,
    orientation: str,
    index: int,
) -> tuple[np.ndarray, str, str]:
    rows, cols = data.shape
    if orientation == "column":
        resolved = max(0, min(int(index), cols - 1))
        return data[:, resolved], "行位置（像素）", f"第 {resolved + 1} 列轮廓"
    resolved = max(0, min(int(index), rows - 1))
    return data[resolved, :], "列位置（像素）", f"第 {resolved + 1} 行轮廓"


def _height_limits(data: np.ndarray) -> tuple[float, float]:
    return float(np.min(data)), float(np.max(data))


class ResultFigureCanvas(FigureCanvasQTAgg):
    """单个标签页只绘制一张结果图，避免多子图拥挤和重复渲染。"""

    def __init__(self) -> None:
        self.figure = Figure(figsize=(10, 7), dpi=100, constrained_layout=True)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._profile_values: np.ndarray | None = None
        self._profile_axes = None
        self._profile_points: list[int] = []
        self._profile_artists: list[object] = []
        self._profile_click_connection = self.mpl_connect("button_press_event", self._on_profile_click)
        self.show_message("请选择高度文件并开始分析。")

    def show_message(self, message: str) -> None:
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        axes.axis("off")
        axes.text(0.5, 0.5, message, ha="center", va="center", color="#64748b", fontsize=12)
        self.draw_idle()

    @staticmethod
    def _draw_surface(
        axes,
        data: np.ndarray,
        title: str,
        z_limits: tuple[float, float] | None = None,
    ) -> None:
        reduced, x, y = _downsample_surface(data)
        xx, yy = np.meshgrid(x, y)
        axes.plot_surface(
            xx,
            yy,
            reduced,
            cmap="viridis",
            linewidth=0,
            edgecolor="none",
            antialiased=True,
            shade=True,
        )
        axes.set_title(title)
        axes.set_xlabel("X（像素）")
        axes.set_ylabel("Y（像素）")
        axes.set_zlabel("高度（nm）")
        if z_limits is not None and z_limits[0] < z_limits[1]:
            axes.set_zlim(z_limits)
        axes.invert_yaxis()
        axes.view_init(elev=32, azim=-105)

    def draw_surface(
        self,
        data: np.ndarray,
        title: str,
        z_limits: tuple[float, float] | None = None,
    ) -> None:
        self.figure.clear()
        axes = self.figure.add_subplot(111, projection="3d")
        self._draw_surface(axes, data, title, z_limits)
        self.draw_idle()

    def draw_height_map(self, data: np.ndarray, title: str) -> None:
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        image = axes.imshow(data, cmap="viridis", origin="upper", aspect="equal")
        axes.set_title(title)
        axes.set_xlabel("X（像素）")
        axes.set_ylabel("Y（像素）")
        self.figure.colorbar(image, ax=axes, fraction=0.035, pad=0.025, label="高度（nm）")
        self.draw_idle()

    def draw_layer_map(self, result: StepAnalysisResult) -> None:
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        layer_image = axes.imshow(
            result.cluster_map,
            cmap="coolwarm",
            origin="upper",
            aspect="equal",
            vmin=1,
            vmax=2,
        )
        axes.set_title("台阶高低层分割")
        axes.set_xlabel("X（像素）")
        axes.set_ylabel("Y（像素）")
        axes.contour(result.cluster_map, levels=[1.5], colors="black", linewidths=1.0, origin="upper")
        colorbar = self.figure.colorbar(layer_image, ax=axes, fraction=0.035, pad=0.025)
        colorbar.set_ticks([1.25, 1.75], labels=["低层", "高层"])
        self.draw_idle()

    def draw_layer_std(self, result: StepAnalysisResult) -> None:
        self.figure.clear()
        axes = self.figure.add_subplot(111)
        values = [
            result.layer_stats["low"]["std"],
            result.layer_stats["high"]["std"],
        ]
        bars = axes.bar(["下表面", "上表面"], values, color=["#2563eb", "#ea580c"], width=0.58)
        axes.set_title("分层表面标准差")
        axes.set_ylabel("标准差（nm）")
        axes.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values, strict=True):
            axes.text(
                bar.get_x() + bar.get_width() / 2.0,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )
        self.draw_idle()

    def draw_profile(
        self,
        data: np.ndarray,
        orientation: str,
        index: int,
        *,
        step_height: float | None = None,
    ) -> None:
        profile, x_label, title = _profile_data(data, orientation, index)
        self._profile_values = profile
        self._profile_points.clear()
        self._profile_artists.clear()
        mean = float(np.mean(profile))
        pv = float(np.max(profile) - np.min(profile))
        std = float(np.std(profile))
        rms = float(np.sqrt(np.mean((profile - mean) ** 2)))
        maximum_index = int(np.argmax(profile))
        minimum_index = int(np.argmin(profile))

        self.figure.clear()
        axes = self.figure.add_subplot(111)
        self._profile_axes = axes
        axes.plot(profile, color="#b91c1c", linewidth=1.2)
        axes.scatter([maximum_index], [profile[maximum_index]], color="#15803d", label="最高点", zorder=3)
        axes.scatter([minimum_index], [profile[minimum_index]], color="#7e22ce", label="最低点", zorder=3)
        suffix = "" if step_height is None else f"，区域台阶高度差={abs(step_height):.3f} nm"
        axes.set_title(f"{title}（PV={pv:.3f} nm，Std={std:.3f} nm，RMS={rms:.3f} nm{suffix}）")
        axes.set_xlabel(x_label)
        axes.set_ylabel("高度（nm）")
        axes.grid(alpha=0.25)
        axes.legend(loc="best")
        axes.set_xlim(0, max(1, profile.size - 1))
        axes.text(
            0.01,
            0.02,
            "提示：在曲线上点击任意两点，可显示两点高度差",
            transform=axes.transAxes,
            color="#475569",
            fontsize=9,
            ha="left",
            va="bottom",
        )
        self.draw_idle()

    def _on_profile_click(self, event) -> None:
        if self._profile_values is None or self._profile_axes is None or event.inaxes is not self._profile_axes:
            return
        if event.xdata is None or event.ydata is None:
            return
        index = int(np.clip(round(event.xdata), 0, self._profile_values.size - 1))
        if len(self._profile_points) >= 2:
            self._profile_points.clear()
        self._profile_points.append(index)
        self._draw_profile_delta_overlay()

    def _draw_profile_delta_overlay(self) -> None:
        if self._profile_values is None or self._profile_axes is None:
            return
        for artist in self._profile_artists:
            artist.remove()
        self._profile_artists.clear()

        for point_index in self._profile_points:
            marker = self._profile_axes.scatter(
                [point_index],
                [self._profile_values[point_index]],
                color="#0284c7",
                s=42,
                zorder=4,
            )
            line = self._profile_axes.axvline(point_index, color="#0284c7", linestyle="--", linewidth=0.9, alpha=0.6)
            self._profile_artists.extend([marker, line])

        if len(self._profile_points) == 2:
            first, second = self._profile_points
            first_height = float(self._profile_values[first])
            second_height = float(self._profile_values[second])
            delta_height = second_height - first_height
            segment = self._profile_axes.plot(
                [first, second],
                [first_height, second_height],
                color="#0284c7",
                linewidth=1.4,
                zorder=3,
            )[0]
            annotation = self._profile_axes.text(
                0.99,
                0.98,
                f"点1={first}: {first_height:.3f} nm\n"
                f"点2={second}: {second_height:.3f} nm\n"
                f"ΔH={delta_height:.3f} nm，|ΔH|={abs(delta_height):.3f} nm",
                transform=self._profile_axes.transAxes,
                ha="right",
                va="top",
                color="#0f172a",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#0284c7", "alpha": 0.92},
            )
            self._profile_artists.extend([segment, annotation])
        self.draw_idle()


class Surface3DCanvas(QWidget):
    """基于 PyVista/VTK 的三维预览画布，替代 Matplotlib 3D 以提高交互流畅度。"""

    def __init__(self, *, render_enabled: bool = True) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._render_enabled = render_enabled
        self._stack = QStackedLayout(self)
        self._message_label = QLabel("请选择高度文件并开始分析。")
        self._message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message_label.setWordWrap(True)
        self._message_label.setStyleSheet("color: #64748b; font-size: 12px;")
        self._plotter: QtInteractor | None = None
        self._last_display_bounds: tuple[float, float, float, float, float, float] | None = None
        self._closed = False
        self._stack.addWidget(self._message_label)
        self.show_message("请选择高度文件并开始分析。")

    def _ensure_plotter(self) -> QtInteractor:
        if self._plotter is None:
            self._plotter = QtInteractor(self, auto_update=False)
            self._stack.addWidget(self._plotter)
        return self._plotter

    def show_message(self, message: str) -> None:
        if self._plotter is not None:
            self._plotter.clear()
        self._message_label.setText(message)
        self._stack.setCurrentWidget(self._message_label)

    def _render_if_visible(self) -> None:
        if self._plotter is None or self._closed or not self.isVisible():
            return
        try:
            self._plotter.render()
        except (AttributeError, RuntimeError):
            return

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._render_if_visible()

    def draw_surface(
        self,
        data: np.ndarray,
        title: str,
        z_limits: tuple[float, float] | None = None,
    ) -> None:
        if self._plotter is not None:
            self._plotter.hide()
            self._plotter.setUpdatesEnabled(False)
        self._message_label.setText("正在渲染三维图...")
        self._stack.setCurrentWidget(self._message_label)
        if self.isVisible():
            self._message_label.repaint()

        grid, display_values, x_coords, y_coords, z_bounds = build_normalized_surface_grid(
            data,
            max_points=256,
            z_bounds=z_limits,
        )
        self._last_display_bounds = (
            float(x_coords[0]),
            float(x_coords[-1]),
            float(y_coords[0]),
            float(y_coords[-1]),
            float(z_bounds[0]),
            float(z_bounds[1]),
        )

        if not self._render_enabled:
            self._message_label.setText(title)
            self._stack.setCurrentWidget(self._message_label)
            return

        plotter = self._ensure_plotter()
        plotter.suppress_rendering = True
        try:
            plotter.clear()
            plotter.set_background("#ffffff")
            plotter.add_mesh(
                grid,
                scalars=display_values.ravel(order="F"),
                cmap="viridis",
                smooth_shading=True,
                show_scalar_bar=True,
                scalar_bar_args={
                    "title": "高度（nm）",
                    "vertical": True,
                    "position_x": 0.88,
                    "position_y": 0.20,
                    "width": 0.045,
                    "height": 0.58,
                    "label_font_size": 9,
                    "title_font_size": 10,
                    "color": "#111827",
                },
                clim=z_bounds,
            )
            plotter.add_text(title, position="upper_edge", font_size=10, color="#111827")
            plotter.show_bounds(
                bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
                axes_ranges=self._last_display_bounds,
                xtitle="X（像素）",
                ytitle="Y（像素）",
                ztitle="高度（nm）",
                grid="back",
                location="outer",
                all_edges=True,
                color="#374151",
                font_size=10,
                n_xlabels=5,
                n_ylabels=5,
                n_zlabels=5,
            )
            plotter.camera_position = [(1.55, -1.70, 1.12), (0.5, 0.5, 0.46), (0.0, 0.0, 1.0)]
            plotter.camera.zoom(0.92)
        finally:
            plotter.suppress_rendering = False
            plotter.setUpdatesEnabled(True)
        self._stack.setCurrentWidget(plotter)
        plotter.show()
        self._render_if_visible()

    def shutdown(self) -> None:
        if not self._closed and self._plotter is not None:
            render_timer = getattr(self._plotter, "render_timer", None)
            if render_timer is not None:
                render_timer.stop()
            self._plotter.close()
            self._closed = True

    def closeEvent(self, event) -> None:
        self.shutdown()
        super().closeEvent(event)


class StepSelectionCanvas(FigureCanvasQTAgg):
    """台阶分析的三点点击与双矩形框选画布。"""

    points_changed = Signal(object)
    regions_changed = Signal(object)

    def __init__(self) -> None:
        self.figure = Figure(figsize=(8, 6), dpi=100, constrained_layout=True)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._data: np.ndarray | None = None
        self._mode = "idle"
        self._points: list[tuple[int, int]] = []
        self._regions: list[tuple[int, int, int, int]] = []
        self._selector: RectangleSelector | None = None
        self._click_connection = self.mpl_connect("button_press_event", self._on_click)
        self.show_message("选择台阶高度文件后，在此处选取三个同层参考点。")

    @property
    def points(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._points)

    @property
    def regions(self) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(self._regions)

    def show_message(self, message: str) -> None:
        self.axes.clear()
        self.axes.axis("off")
        self.axes.text(0.5, 0.5, message, ha="center", va="center", color="#64748b", fontsize=12)
        self.draw_idle()

    def _draw_height_map(self, title: str) -> None:
        if self._data is None:
            return
        self.axes.clear()
        self.axes.imshow(self._data, cmap="viridis", origin="upper", aspect="equal")
        self.axes.set_title(title)
        self.axes.set_xlabel("X（像素）")
        self.axes.set_ylabel("Y（像素）")

    def start_point_selection(self, data: np.ndarray) -> None:
        self._disable_selector()
        self._data = np.asarray(data, dtype=np.float64)
        self._points.clear()
        self._regions.clear()
        self._mode = "points"
        self._draw_height_map("请依次点击三个同层、分布较开的非共线参考点")
        self.draw_idle()
        self.points_changed.emit(tuple())

    def _on_click(self, event) -> None:
        if (
            self._mode != "points"
            or self._data is None
            or event.inaxes is not self.axes
            or event.xdata is None
            or event.ydata is None
            or len(self._points) >= 3
        ):
            return
        col = int(np.clip(round(event.xdata), 0, self._data.shape[1] - 1))
        row = int(np.clip(round(event.ydata), 0, self._data.shape[0] - 1))
        self._points.append((row, col))
        self.axes.plot(col, row, marker="o", markerfacecolor="none", markeredgecolor="white", markersize=9)
        self.axes.plot(col, row, marker="x", color="black", markersize=8)
        self.axes.text(col + 1, row + 1, str(len(self._points)), color="white", weight="bold")
        if len(self._points) == 3:
            self._mode = "idle"
            self.axes.set_title("三个参考点已选择，可开始台阶分析")
        self.draw_idle()
        self.points_changed.emit(tuple(self._points))

    def start_region_selection(self, data: np.ndarray) -> None:
        self._disable_selector()
        self._data = np.asarray(data, dtype=np.float64)
        self._regions.clear()
        self._mode = "regions"
        self._draw_height_map("请框选区域一，然后框选区域二")
        self._selector = RectangleSelector(
            self.axes,
            self._on_rectangle,
            useblit=True,
            button=[1],
            minspanx=2,
            minspany=2,
            spancoords="data",
            interactive=True,
        )
        self.draw_idle()
        self.regions_changed.emit(tuple())

    def _on_rectangle(self, press_event, release_event) -> None:
        if (
            self._mode != "regions"
            or self._data is None
            or press_event.xdata is None
            or press_event.ydata is None
            or release_event.xdata is None
            or release_event.ydata is None
        ):
            return
        x0, x1 = sorted((press_event.xdata, release_event.xdata))
        y0, y1 = sorted((press_event.ydata, release_event.ydata))
        region = (
            int(np.floor(x0)),
            int(np.floor(y0)),
            int(np.ceil(x1)) + 1,
            int(np.ceil(y1)) + 1,
        )
        self._regions.append(region)
        color = "#111827" if len(self._regions) == 1 else "#ffffff"
        self.axes.add_patch(
            Rectangle(
                (region[0], region[1]),
                region[2] - region[0],
                region[3] - region[1],
                fill=False,
                edgecolor=color,
                linewidth=2.0,
            )
        )
        self.axes.text(region[0] + 1, region[1] + 1, f"区域{len(self._regions)}", color=color, weight="bold")
        if len(self._regions) >= 2:
            self._mode = "idle"
            self._disable_selector()
            self.axes.set_title("两个测量区域已选择")
        else:
            self.axes.set_title("区域一已选择，请继续框选区域二")
        self.draw_idle()
        self.regions_changed.emit(tuple(self._regions))

    def _disable_selector(self) -> None:
        if self._selector is not None:
            self._selector.set_active(False)
            self._selector = None

    def shutdown(self) -> None:
        self._disable_selector()
        self.mpl_disconnect(self._click_connection)


def _build_canvas_tab(canvas: QWidget) -> QWidget:
    page = QWidget()
    layout = QVBoxLayout(page)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    if isinstance(canvas, FigureCanvasQTAgg):
        layout.addWidget(NavigationToolbar2QT(canvas, page))
    layout.addWidget(canvas, 1)
    return page


def _build_file_row(edit: QLineEdit, button: QPushButton) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    layout.addWidget(edit, 1)
    layout.addWidget(button, 0)
    return row


def _configure_profile_spinbox(spinbox: QSpinBox) -> None:
    spinbox.setRange(1, 1)
    spinbox.setAccelerated(True)
    spinbox.setKeyboardTracking(False)
    spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
    spinbox.setMinimumWidth(112)
    spinbox.setMinimumHeight(32)
    spinbox.setToolTip("可直接输入行/列序号；按回车或移出焦点后更新轮廓。")


class _PlaneAnalysisWorker(QObject):
    """平面分析后台任务，避免大矩阵加载和拟合阻塞 GUI 线程。"""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, path: str, conversion_factor: float, method: str) -> None:
        super().__init__()
        self._path = path
        self._conversion_factor = float(conversion_factor)
        self._method = method

    @Slot()
    def run(self) -> None:
        try:
            matrix = load_height_matrix(self._path, self._conversion_factor)
            self.finished.emit(analyze_plane(matrix, method=self._method))
        except Exception as exc:  # pragma: no cover - failure path is exercised through GUI state.
            self.failed.emit(str(exc))


class _StepAnalysisWorker(QObject):
    """台阶分析后台任务，避免调平、分层和逐层去噪阻塞 GUI 线程。"""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, matrix: np.ndarray, points: tuple[tuple[int, int], ...], denoise: bool) -> None:
        super().__init__()
        self._matrix = np.asarray(matrix, dtype=np.float64)
        self._points = tuple(points)
        self._denoise = bool(denoise)

    @Slot()
    def run(self) -> None:
        try:
            self.finished.emit(analyze_step(self._matrix, points=self._points, denoise=self._denoise))
        except Exception as exc:  # pragma: no cover - failure path is exercised through GUI state.
            self.failed.emit(str(exc))


class PlaneAnalysisPage(QFrame):
    """单个平面高度文件分析页面。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AnalysisPage")
        self._result: PlaneAnalysisResult | None = None
        self._rendered_tabs: set[int] = set()
        self._analysis_thread: QThread | None = None
        self._analysis_worker: _PlaneAnalysisWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        self.raw_surface_canvas = Surface3DCanvas()
        self.processed_surface_canvas = Surface3DCanvas()
        self.height_map_canvas = ResultFigureCanvas()
        self.profile_canvas = ResultFigureCanvas()
        self.tabs.addTab(_build_canvas_tab(self.raw_surface_canvas), "原始三维")
        self.tabs.addTab(_build_canvas_tab(self.processed_surface_canvas), "处理后三维")
        self.tabs.addTab(_build_canvas_tab(self.height_map_canvas), "二维高度图")
        self.tabs.addTab(_build_canvas_tab(self.profile_canvas), "一维轮廓")
        self.tabs.currentChanged.connect(self._render_current_tab)
        root.addWidget(self.tabs, 1)

        controls = QFrame()
        controls.setObjectName("AnalysisControlPanel")
        controls.setMinimumWidth(320)
        controls.setMaximumWidth(380)
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(SectionHeader("平面分析", "选择一个独立的二维高度文本文件。"))

        form = QFormLayout()
        form.setSpacing(8)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("请选择平面高度 .txt 文件")
        self.choose_button = QPushButton("选择")
        self.choose_button.setAccessibleName("选择平面高度文件")
        self.choose_button.clicked.connect(self._choose_file)
        form.addRow("高度文件", _build_file_row(self.file_edit, self.choose_button))

        self.conversion_factor = QDoubleSpinBox()
        self.conversion_factor.setDecimals(6)
        self.conversion_factor.setRange(-1_000_000.0, 1_000_000.0)
        self.conversion_factor.setValue(1.0)
        self.conversion_factor.setSuffix(" nm/单位")
        form.addRow("换算系数", self.conversion_factor)

        self.method_combo = QComboBox()
        self.method_combo.addItem("简单平面校准", "simple")
        self.method_combo.addItem("鲁棒平面校准", "robust")
        self.method_combo.addItem("二次曲面校准", "quadratic")
        form.addRow("校准方法", self.method_combo)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem("行轮廓", "row")
        self.profile_combo.addItem("列轮廓", "column")
        self.profile_combo.currentIndexChanged.connect(self._refresh_profile_limits)
        form.addRow("轮廓方向", self.profile_combo)
        self.profile_index = QSpinBox()
        _configure_profile_spinbox(self.profile_index)
        self.profile_index.valueChanged.connect(self._draw_profile)
        form.addRow("轮廓序号", self.profile_index)
        layout.addLayout(form)

        self.analyze_button = QPushButton("开始平面分析")
        self.analyze_button.setObjectName("PrimaryButton")
        self.analyze_button.setAccessibleName("开始平面分析")
        self.analyze_button.clicked.connect(self._run_analysis)
        layout.addWidget(self.analyze_button)

        self.status_label = QLabel("等待选择平面高度文件。")
        self.status_label.setObjectName("AnalysisStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(SectionHeader("分析指标", "数值与图表同步使用处理后的单个平面数据。"))
        self.metrics = QPlainTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setPlaceholderText("分析完成后显示高度范围、标准差和拟合参数。")
        layout.addWidget(self.metrics, 1)
        root.addWidget(controls, 0)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择平面高度文件", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.file_edit.setText(path)
            self.status_label.setText("文件已选择，可开始平面分析。")

    def _run_analysis(self) -> None:
        if self._analysis_thread is not None:
            return
        self.analyze_button.setEnabled(False)
        self.status_label.setText("正在后台执行平面分析，界面可继续响应。")
        self._analysis_thread = QThread(self)
        self._analysis_worker = _PlaneAnalysisWorker(
            self.file_edit.text().strip(),
            self.conversion_factor.value(),
            str(self.method_combo.currentData()),
        )
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._handle_analysis_finished)
        self._analysis_worker.failed.connect(self._handle_analysis_failed)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.failed.connect(self._analysis_thread.quit)
        self._analysis_worker.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.failed.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._finalize_analysis_thread)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)
        self._analysis_thread.start()

    def _handle_analysis_finished(self, result: object) -> None:
        self._result = result if isinstance(result, PlaneAnalysisResult) else None
        if self._result is None:
            self._handle_analysis_failed("平面分析返回了无效结果。")
            return
        self._rendered_tabs.clear()
        self._refresh_profile_limits()
        self._update_metrics()
        self.tabs.setCurrentIndex(2)
        self._render_current_tab(2)
        self.status_label.setText("平面分析完成。二维图已显示，三维图点击对应标签页后按需渲染。")

    def _handle_analysis_failed(self, message: str) -> None:
        self.status_label.setText(f"分析失败：{message} 请检查文件和参数后重试。")
        QMessageBox.critical(self, "平面分析失败", f"{message}\n\n请检查高度文件格式和校准参数。")

    def _finalize_analysis_thread(self) -> None:
        self._analysis_thread = None
        self._analysis_worker = None
        self.analyze_button.setEnabled(True)

    def _refresh_profile_limits(self) -> None:
        if self._result is None:
            return
        orientation = str(self.profile_combo.currentData())
        maximum = self._result.processed.shape[1] if orientation == "column" else self._result.processed.shape[0]
        self.profile_index.blockSignals(True)
        self.profile_index.setRange(1, maximum)
        self.profile_index.setValue(maximum // 2 + 1)
        self.profile_index.blockSignals(False)
        self._draw_profile()

    def _draw_profile(self) -> None:
        if self._result is None:
            return
        self._rendered_tabs.discard(3)
        if self.tabs.currentIndex() == 3:
            self._render_current_tab(3)

    def _render_current_tab(self, index: int) -> None:
        if self._result is None or index in self._rendered_tabs:
            return
        if index == 0:
            self.raw_surface_canvas.draw_surface(self._result.original, "原始三维表面形貌")
        elif index == 1:
            self.processed_surface_canvas.draw_surface(
                self._result.processed,
                "校正并去噪后三维形貌",
                z_limits=_height_limits(self._result.original),
            )
        elif index == 2:
            self.height_map_canvas.draw_height_map(self._result.processed, "处理后二维高度图")
        elif index == 3:
            self.profile_canvas.draw_profile(
                self._result.processed,
                str(self.profile_combo.currentData()),
                self.profile_index.value() - 1,
            )
        else:
            return
        self._rendered_tabs.add(index)

    def _update_metrics(self) -> None:
        if self._result is None:
            return
        stats = self._result.stats
        coefficients = ", ".join(f"{value:.6f}" for value in self._result.coefficients)
        self.metrics.setPlainText(
            "\n".join(
                (
                    f"文件：{Path(self.file_edit.text()).name}",
                    f"校准方法：{self.method_combo.currentText()}",
                    f"高度范围：{stats['height_range']:.6f} nm",
                    f"最大高度：{stats['maximum']:.6f} nm",
                    f"最小高度：{stats['minimum']:.6f} nm",
                    f"平均高度：{stats['mean']:.6f} nm",
                    f"标准差：{stats['std']:.6f} nm",
                    f"RMS：{stats['rms']:.6f} nm",
                    f"初步异常点：{self._result.outlier_count}",
                    f"残余噪点：{self._result.noise_count}",
                    f"拟合参数：{coefficients}",
                )
            )
        )

    def shutdown(self) -> None:
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.quit()
            self._analysis_thread.wait(1000)
        for canvas in (self.raw_surface_canvas, self.processed_surface_canvas):
            if hasattr(canvas, "shutdown"):
                canvas.shutdown()


class StepAnalysisPage(QFrame):
    """单个台阶高度文件分析页面。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("AnalysisPage")
        self._matrix: np.ndarray | None = None
        self._result: StepAnalysisResult | None = None
        self._measurement: StepMeasurement | None = None
        self._rendered_tabs: set[int] = set()
        self._analysis_thread: QThread | None = None
        self._analysis_worker: _StepAnalysisWorker | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        self.tabs = QTabWidget()
        self.selection_canvas = StepSelectionCanvas()
        self.raw_surface_canvas = Surface3DCanvas()
        self.layer_map_canvas = ResultFigureCanvas()
        self.processed_surface_canvas = Surface3DCanvas()
        self.layer_std_canvas = ResultFigureCanvas()
        self.profile_canvas = ResultFigureCanvas()
        self.tabs.addTab(_build_canvas_tab(self.selection_canvas), "交互选择")
        self.tabs.addTab(_build_canvas_tab(self.raw_surface_canvas), "原始三维")
        self.tabs.addTab(_build_canvas_tab(self.layer_map_canvas), "分层结果")
        self.tabs.addTab(_build_canvas_tab(self.processed_surface_canvas), "处理后三维")
        self.tabs.addTab(_build_canvas_tab(self.layer_std_canvas), "分层标准差")
        self.tabs.addTab(_build_canvas_tab(self.profile_canvas), "一维轮廓")
        self.tabs.currentChanged.connect(self._render_current_tab)
        root.addWidget(self.tabs, 1)

        controls = QFrame()
        controls.setObjectName("AnalysisControlPanel")
        controls.setMinimumWidth(330)
        controls.setMaximumWidth(390)
        layout = QVBoxLayout(controls)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        layout.addWidget(SectionHeader("台阶分析", "选择一个独立台阶文件，再完成三点调平与双区域测量。"))

        form = QFormLayout()
        form.setSpacing(8)
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("请选择台阶高度 .txt 文件")
        self.file_edit.textChanged.connect(self._invalidate_input)
        self.choose_button = QPushButton("选择")
        self.choose_button.setAccessibleName("选择台阶高度文件")
        self.choose_button.clicked.connect(self._choose_file)
        form.addRow("高度文件", _build_file_row(self.file_edit, self.choose_button))

        self.conversion_factor = QDoubleSpinBox()
        self.conversion_factor.setDecimals(6)
        self.conversion_factor.setRange(-1_000_000.0, 1_000_000.0)
        self.conversion_factor.setValue(1.0)
        self.conversion_factor.setSuffix(" nm/单位")
        self.conversion_factor.valueChanged.connect(lambda _value: self._invalidate_input())
        form.addRow("换算系数", self.conversion_factor)

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("去噪", "denoise")
        self.mode_combo.addItem("不去噪", "raw")
        form.addRow("处理模式", self.mode_combo)

        self.profile_combo = QComboBox()
        self.profile_combo.addItem("行轮廓", "row")
        self.profile_combo.addItem("列轮廓", "column")
        self.profile_combo.currentIndexChanged.connect(self._refresh_profile_limits)
        form.addRow("轮廓方向", self.profile_combo)
        self.profile_index = QSpinBox()
        _configure_profile_spinbox(self.profile_index)
        self.profile_index.valueChanged.connect(self._draw_profile)
        form.addRow("轮廓序号", self.profile_index)
        layout.addLayout(form)

        self.load_button = QPushButton("载入并选择三点")
        self.load_button.setObjectName("SecondaryButton")
        self.load_button.clicked.connect(self._load_for_selection)
        layout.addWidget(self.load_button)

        self.analyze_button = QPushButton("开始台阶分析")
        self.analyze_button.setObjectName("PrimaryButton")
        self.analyze_button.setAccessibleName("开始台阶分析")
        self.analyze_button.setEnabled(False)
        self.analyze_button.clicked.connect(self._run_analysis)
        layout.addWidget(self.analyze_button)

        reset_row = QHBoxLayout()
        self.reset_points_button = QPushButton("重选三点")
        self.reset_points_button.clicked.connect(self._reset_points)
        self.reset_regions_button = QPushButton("重选区域")
        self.reset_regions_button.clicked.connect(self._reset_regions)
        self.reset_regions_button.setEnabled(False)
        reset_row.addWidget(self.reset_points_button)
        reset_row.addWidget(self.reset_regions_button)
        layout.addLayout(reset_row)

        self.status_label = QLabel("等待选择台阶高度文件。")
        self.status_label.setObjectName("AnalysisStatus")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        layout.addWidget(SectionHeader("分析指标", "区域一减区域二；正负号保留用户框选顺序。"))
        self.metrics = QPlainTextEdit()
        self.metrics.setReadOnly(True)
        self.metrics.setPlaceholderText("三点调平、分层和区域高度差将在此显示。")
        layout.addWidget(self.metrics, 1)
        root.addWidget(controls, 0)

        self.selection_canvas.points_changed.connect(self._on_points_changed)
        self.selection_canvas.regions_changed.connect(self._on_regions_changed)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择台阶高度文件", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.file_edit.setText(path)
            self._load_for_selection()

    def _invalidate_input(self) -> None:
        self._matrix = None
        self._result = None
        self._measurement = None
        self.analyze_button.setEnabled(False)
        self.reset_regions_button.setEnabled(False)

    def _load_for_selection(self) -> None:
        try:
            self._matrix = load_height_matrix(self.file_edit.text().strip(), self.conversion_factor.value())
            self._result = None
            self._measurement = None
            self._rendered_tabs.clear()
            self.selection_canvas.start_point_selection(self._matrix)
            self.tabs.setCurrentIndex(0)
            self.status_label.setText("请在图中依次点击三个同层、分布较开的非共线参考点。")
            self.metrics.clear()
            self._show_result_placeholders("选择三个参考点并开始分析后显示结果。")
        except Exception as exc:
            self.status_label.setText(f"载入失败：{exc} 请检查文件后重试。")
            QMessageBox.critical(self, "台阶文件载入失败", f"{exc}\n\n请确认文件为规则二维数值矩阵。")

    def _on_points_changed(self, points: tuple[tuple[int, int], ...]) -> None:
        self.analyze_button.setEnabled(self._matrix is not None and len(points) == 3)
        if len(points) < 3 and self._matrix is not None:
            self.status_label.setText(f"已选择 {len(points)}/3 个参考点。")
        elif len(points) == 3:
            self.status_label.setText("三个参考点已选择，可开始台阶分析。")

    def _run_analysis(self) -> None:
        if self._matrix is None or len(self.selection_canvas.points) != 3 or self._analysis_thread is not None:
            return
        self.analyze_button.setEnabled(False)
        self.status_label.setText("正在后台执行台阶调平、分层和去噪，界面可继续响应。")
        self._analysis_thread = QThread(self)
        self._analysis_worker = _StepAnalysisWorker(
            self._matrix,
            self.selection_canvas.points,
            str(self.mode_combo.currentData()) == "denoise",
        )
        self._analysis_worker.moveToThread(self._analysis_thread)
        self._analysis_thread.started.connect(self._analysis_worker.run)
        self._analysis_worker.finished.connect(self._handle_analysis_finished)
        self._analysis_worker.failed.connect(self._handle_analysis_failed)
        self._analysis_worker.finished.connect(self._analysis_thread.quit)
        self._analysis_worker.failed.connect(self._analysis_thread.quit)
        self._analysis_worker.finished.connect(self._analysis_worker.deleteLater)
        self._analysis_worker.failed.connect(self._analysis_worker.deleteLater)
        self._analysis_thread.finished.connect(self._finalize_analysis_thread)
        self._analysis_thread.finished.connect(self._analysis_thread.deleteLater)
        self._analysis_thread.start()

    def _handle_analysis_finished(self, result: object) -> None:
        self._result = result if isinstance(result, StepAnalysisResult) else None
        if self._result is None:
            self._handle_analysis_failed("台阶分析返回了无效结果。")
            return
        self._measurement = None
        self._rendered_tabs.clear()
        self._refresh_profile_limits()
        self.selection_canvas.start_region_selection(self._result.processed)
        self.reset_regions_button.setEnabled(True)
        self._update_metrics()
        self.tabs.setCurrentIndex(0)
        self.status_label.setText("调平和分层完成。请在图中先框选区域一，再框选区域二。")

    def _handle_analysis_failed(self, message: str) -> None:
        self.status_label.setText(f"分析失败：{message} 请重新选择参考点后重试。")
        QMessageBox.critical(self, "台阶分析失败", f"{message}\n\n请重新选择三个同层且不共线的参考点。")

    def _finalize_analysis_thread(self) -> None:
        self._analysis_thread = None
        self._analysis_worker = None
        self.analyze_button.setEnabled(self._matrix is not None and len(self.selection_canvas.points) == 3)

    def _on_regions_changed(self, regions: tuple[tuple[int, int, int, int], ...]) -> None:
        if self._result is None:
            return
        if len(regions) < 2:
            if len(regions) == 1:
                self.status_label.setText("区域一已选择，请继续框选区域二。")
            return
        try:
            self._measurement = compute_step_height(self._result.processed, regions[0], regions[1])
            self._update_metrics()
            self._draw_profile()
            self.status_label.setText("两个区域已选择，台阶高度差已计算。")
            self.tabs.setCurrentIndex(2)
        except Exception as exc:
            self.status_label.setText(f"区域计算失败：{exc} 请重新框选。")
            QMessageBox.critical(self, "区域计算失败", f"{exc}\n\n请重新框选两个有效矩形区域。")

    def _reset_points(self) -> None:
        if self._matrix is not None:
            self._result = None
            self._measurement = None
            self.selection_canvas.start_point_selection(self._matrix)
            self.tabs.setCurrentIndex(0)
            self.reset_regions_button.setEnabled(False)
            self._rendered_tabs.clear()
            self._show_result_placeholders("重新选择三个参考点后显示结果。")

    def _reset_regions(self) -> None:
        if self._result is not None:
            self._measurement = None
            self.selection_canvas.start_region_selection(self._result.processed)
            self.tabs.setCurrentIndex(0)
            self.status_label.setText("请重新框选区域一和区域二。")
            self._update_metrics()
            self._draw_profile()

    def _refresh_profile_limits(self) -> None:
        if self._result is None:
            return
        orientation = str(self.profile_combo.currentData())
        maximum = self._result.processed.shape[1] if orientation == "column" else self._result.processed.shape[0]
        self.profile_index.blockSignals(True)
        self.profile_index.setRange(1, maximum)
        self.profile_index.setValue(maximum // 2 + 1)
        self.profile_index.blockSignals(False)
        self._draw_profile()

    def _draw_profile(self) -> None:
        if self._result is None:
            return
        self._rendered_tabs.discard(5)
        if self.tabs.currentIndex() == 5:
            self._render_current_tab(5)

    def _render_current_tab(self, index: int) -> None:
        if self._result is None or index == 0 or index in self._rendered_tabs:
            return
        if index == 1:
            self.raw_surface_canvas.draw_surface(self._result.original, "原始三维表面形貌")
        elif index == 2:
            self.layer_map_canvas.draw_layer_map(self._result)
        elif index == 3:
            title = "分层去噪后三维形貌" if self._result.denoised else "三点调平后三维形貌（未去噪）"
            self.processed_surface_canvas.draw_surface(self._result.processed, title)
        elif index == 4:
            self.layer_std_canvas.draw_layer_std(self._result)
        elif index == 5:
            step_height = None if self._measurement is None else self._measurement.step_height
            self.profile_canvas.draw_profile(
                self._result.processed,
                str(self.profile_combo.currentData()),
                self.profile_index.value() - 1,
                step_height=step_height,
            )
        else:
            return
        self._rendered_tabs.add(index)

    def _show_result_placeholders(self, message: str) -> None:
        for canvas in (
            self.raw_surface_canvas,
            self.layer_map_canvas,
            self.processed_surface_canvas,
            self.layer_std_canvas,
            self.profile_canvas,
        ):
            canvas.show_message(message)

    def _update_metrics(self) -> None:
        if self._result is None:
            return
        low = self._result.layer_stats["low"]
        high = self._result.layer_stats["high"]
        lines = [
            f"文件：{Path(self.file_edit.text()).name}",
            f"处理模式：{'去噪' if self._result.denoised else '不去噪'}",
            f"分割阈值：{self._result.threshold:.6f} nm",
            f"三点调平 X 斜率：{self._result.coefficients[0]:.6f}",
            f"三点调平 Y 斜率：{self._result.coefficients[1]:.6f}",
            f"下表面标准差：{low['std']:.6f} nm",
            f"上表面标准差：{high['std']:.6f} nm",
            f"下表面面积占比：{low['area_ratio'] * 100:.2f}%",
            f"上表面面积占比：{high['area_ratio'] * 100:.2f}%",
            f"修复噪点：{self._result.noise_count}",
        ]
        if self._measurement is None:
            lines.append("区域高度差：等待框选两个区域")
        else:
            lines.extend(
                (
                    f"区域一平均高度：{self._measurement.region_one_mean:.6f} nm",
                    f"区域二平均高度：{self._measurement.region_two_mean:.6f} nm",
                    f"台阶平均高度差：{self._measurement.step_height:.6f} nm（区域一 - 区域二）",
                )
            )
        self.metrics.setPlainText("\n".join(lines))

    def shutdown(self) -> None:
        if self._analysis_thread is not None and self._analysis_thread.isRunning():
            self._analysis_thread.quit()
            self._analysis_thread.wait(1000)
        self.selection_canvas.shutdown()
        for canvas in (self.raw_surface_canvas, self.processed_surface_canvas):
            if hasattr(canvas, "shutdown"):
                canvas.shutdown()
