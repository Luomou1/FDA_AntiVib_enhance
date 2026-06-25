from __future__ import annotations

from typing import Callable

import numpy as np
import pyvista as pv
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtGui import QCloseEvent, QGuiApplication, QShowEvent
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor


def downsample_for_preview(data: np.ndarray, max_points: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = data.shape
    row_step = max(1, int(np.ceil(rows / max_points)))
    col_step = max(1, int(np.ceil(cols / max_points)))
    reduced = data[::row_step, ::col_step]
    y_coords = np.arange(0, rows, row_step, dtype=np.float64)[: reduced.shape[0]]
    x_coords = np.arange(0, cols, col_step, dtype=np.float64)[: reduced.shape[1]]
    return reduced, x_coords, y_coords


def _zoom_limits(left: float, right: float, center: float, scale: float) -> tuple[float, float]:
    span = (right - left) * scale
    new_left = center - span / 2.0
    new_right = center + span / 2.0
    return new_left, new_right


def _clamp_limits(left: float, right: float, bound_left: float, bound_right: float) -> tuple[float, float]:
    span = right - left
    bound_span = bound_right - bound_left
    if span >= bound_span:
        return bound_left, bound_right
    if left < bound_left:
        return bound_left, bound_left + span
    if right > bound_right:
        return bound_right - span, bound_right
    return left, right


def _box_aspect_from_limits(
    x_left: float,
    x_right: float,
    y_bottom: float,
    y_top: float,
    z_bottom: float,
    z_top: float,
) -> tuple[float, float, float]:
    x_span = max(float(x_right - x_left), 1.0)
    y_span = max(float(y_top - y_bottom), 1.0)
    z_span = max(float(z_top - z_bottom), 1.0)
    return (x_span, y_span, z_span)


def _ensure_nonzero_limits(lower: float, upper: float, minimum_span: float = 1.0) -> tuple[float, float]:
    """确保坐标轴范围始终有非零跨度，避免 Matplotlib 在平坦数据上报警告。"""
    lower = float(lower)
    upper = float(upper)
    span = upper - lower
    if abs(span) > 1e-12:
        return lower, upper

    # 当整张图是平面时，Matplotlib 会因为上下界完全相同而发出警告。
    # 这里按“数值量级 + 最小兜底跨度”补一个很小的 padding，让显示稳定但不改数据本身。
    half_span = max(abs(lower) * 0.05, abs(upper) * 0.05, float(minimum_span) / 2.0, 1e-6)
    return lower - half_span, upper + half_span


def _combined_finite_bounds(*arrays: np.ndarray) -> tuple[float, float]:
    """计算多组表面的公共有限值范围，平面数据也返回稳定的非零跨度。"""
    minima: list[float] = []
    maxima: list[float] = []
    for array in arrays:
        finite_values = np.asarray(array, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            minima.append(float(np.min(finite_values)))
            maxima.append(float(np.max(finite_values)))
    if not minima:
        return -0.5, 0.5
    return _ensure_nonzero_limits(min(minima), max(maxima))


def build_normalized_surface_grid(
    data: np.ndarray,
    max_points: int = 96,
    z_bounds: tuple[float, float] | None = None,
) -> tuple[pv.StructuredGrid, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """将表面归一化到固定单位坐标盒，同时保留原始坐标和值用于刻度和着色。"""
    reduced, x_coords, y_coords = downsample_for_preview(np.asarray(data), max_points=max_points)
    resolved_z_bounds = _combined_finite_bounds(reduced) if z_bounds is None else _ensure_nonzero_limits(*z_bounds)
    z_lower, z_upper = resolved_z_bounds
    display_values = np.nan_to_num(
        reduced.astype(np.float32),
        nan=z_lower,
        posinf=z_upper,
        neginf=z_lower,
    )
    z_normalized = np.clip((display_values - z_lower) / (z_upper - z_lower), 0.0, 1.0)
    x_normalized = np.linspace(0.0, 1.0, display_values.shape[1], dtype=np.float32)
    y_normalized = np.linspace(0.0, 1.0, display_values.shape[0], dtype=np.float32)
    xx, yy = np.meshgrid(x_normalized, y_normalized)
    grid = pv.StructuredGrid(xx, yy, z_normalized)
    return grid, display_values, x_coords, y_coords, resolved_z_bounds


class _InteractiveCanvas(FigureCanvasQTAgg):
    def __init__(self, projection: str | None = None) -> None:
        # Qt 页签里的隐藏画布在重绘时可能还没有有效尺寸；constrained_layout
        # 会把这种 0 尺寸状态当成布局输入，导致退出或刷新时反复报警并拖慢交互。
        self.figure = Figure(figsize=(8, 6), dpi=100, constrained_layout=False)
        self.figure.subplots_adjust(left=0.10, right=0.92, bottom=0.11, top=0.90)
        self.axes = self.figure.add_subplot(111, projection=projection)
        super().__init__(self.figure)
        self._info_callback: Callable[[str], None] | None = None
        self._pixel_callback: Callable[[dict[str, float]], None] | None = None
        self._default_limits: tuple | None = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def set_info_callback(self, callback: Callable[[str], None]) -> None:
        self._info_callback = callback

    def set_pixel_callback(self, callback: Callable[[dict[str, float]], None]) -> None:
        self._pixel_callback = callback

    def _emit_info(self, text: str) -> None:
        if self._info_callback is not None:
            self._info_callback(text)

    def _emit_pixel(self, payload: dict[str, float]) -> None:
        if self._pixel_callback is not None:
            self._pixel_callback(payload)

    def copy_current_view_to_clipboard(self) -> None:
        pixmap = self.grab()
        QGuiApplication.clipboard().setPixmap(pixmap)
        self._emit_info("当前视图已复制到剪贴板")

    def reset_view(self) -> None:
        if self._default_limits is None:
            return


class HeatmapCanvas(_InteractiveCanvas):
    def __init__(self) -> None:
        super().__init__(projection=None)
        self._data: np.ndarray | None = None
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("scroll_event", self._on_scroll)

    def draw_map(self, data: np.ndarray, title: str) -> None:
        self._data = np.asarray(data)
        self.axes.clear()
        image = self.axes.imshow(
            self._data,
            cmap="viridis",
            origin="lower",
            extent=(0, self._data.shape[1] - 1, 0, self._data.shape[0] - 1),
            aspect="equal",
        )
        self.axes.set_title(title)
        self.axes.set_xlabel("X (pixels)")
        self.axes.set_ylabel("Y (pixels)")
        if len(self.figure.axes) > 1:
            for extra_axes in self.figure.axes[1:]:
                self.figure.delaxes(extra_axes)
        self.figure.colorbar(image, ax=self.axes, fraction=0.035, pad=0.02)
        self._default_limits = (
            (0.0, float(self._data.shape[1] - 1)),
            (0.0, float(self._data.shape[0] - 1)),
        )
        self.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axes or self._data is None or event.xdata is None or event.ydata is None:
            return
        col = int(np.clip(round(event.xdata), 0, self._data.shape[1] - 1))
        row = int(np.clip(round(event.ydata), 0, self._data.shape[0] - 1))
        value = float(self._data[row, col])
        self._emit_info(f"x={col}, y={row}, value={value:.6f}")
        self._emit_pixel({"x": float(col), "y": float(row), "value": value})

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.axes:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x_left, x_right = self.axes.get_xlim()
        y_bottom, y_top = self.axes.get_ylim()
        x_center = event.xdata if event.xdata is not None else (x_left + x_right) / 2.0
        y_center = event.ydata if event.ydata is not None else (y_bottom + y_top) / 2.0
        new_x = _clamp_limits(*_zoom_limits(x_left, x_right, x_center, scale), 0.0, float(self._data.shape[1] - 1))
        new_y = _clamp_limits(*_zoom_limits(y_bottom, y_top, y_center, scale), 0.0, float(self._data.shape[0] - 1))
        self.axes.set_xlim(*new_x)
        self.axes.set_ylim(*new_y)
        self.draw_idle()

    def reset_view(self) -> None:
        if self._default_limits is None:
            return
        self.axes.set_xlim(*self._default_limits[0])
        self.axes.set_ylim(*self._default_limits[1])
        self.draw_idle()


class SpectrumCanvas(_InteractiveCanvas):
    def __init__(self) -> None:
        super().__init__(projection=None)
        self._k_axis: np.ndarray | None = None
        self._spectrum: np.ndarray | None = None
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("scroll_event", self._on_scroll)

    def draw_spectrum(self, k_axis: np.ndarray, spectrum: np.ndarray, peak_index: int, title: str) -> None:
        self._k_axis = np.asarray(k_axis, dtype=np.float64)
        self._spectrum = np.asarray(spectrum, dtype=np.float64)
        peak_index = int(np.clip(peak_index, 0, self._k_axis.shape[0] - 1))
        peak_k = float(self._k_axis[peak_index])
        peak_value = float(self._spectrum[peak_index])

        self.axes.clear()
        self.axes.plot(self._k_axis, self._spectrum, color="black", linewidth=1.4)
        self.axes.axvline(peak_k, color="#d94841", linestyle="--", linewidth=1.2)
        self.axes.scatter([peak_k], [peak_value], color="#d94841", zorder=3)
        self.axes.annotate(
            f"K0={peak_k:.4f}",
            (peak_k, peak_value),
            xytext=(8, 8),
            textcoords="offset points",
            color="#d94841",
        )
        self.axes.set_title(title)
        self.axes.set_xlabel("Wave Number k (rad/um)")
        self.axes.set_ylabel("Aggregated Amplitude (a.u.)")
        y_max = max(float(np.nanmax(self._spectrum)), 1e-6)
        self.axes.set_xlim(float(self._k_axis[0]), float(self._k_axis[-1]))
        self.axes.set_ylim(0.0, y_max * 1.1)
        self._default_limits = (
            (float(self._k_axis[0]), float(self._k_axis[-1])),
            (0.0, y_max * 1.1),
        )
        self.draw_idle()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.axes or self._k_axis is None or self._spectrum is None or event.xdata is None:
            return
        index = int(np.argmin(np.abs(self._k_axis - event.xdata)))
        self._emit_info(f"k={self._k_axis[index]:.6f}, value={self._spectrum[index]:.6f}")

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.axes or self._default_limits is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x_left, x_right = self.axes.get_xlim()
        y_bottom, y_top = self.axes.get_ylim()
        x_center = event.xdata if event.xdata is not None else (x_left + x_right) / 2.0
        y_center = event.ydata if event.ydata is not None else (y_bottom + y_top) / 2.0
        new_x = _clamp_limits(*_zoom_limits(x_left, x_right, x_center, scale), *self._default_limits[0])
        new_y = _clamp_limits(*_zoom_limits(y_bottom, y_top, y_center, scale), *self._default_limits[1])
        self.axes.set_xlim(*new_x)
        self.axes.set_ylim(*new_y)
        self.draw_idle()

    def reset_view(self) -> None:
        if self._default_limits is None:
            return
        self.axes.set_xlim(*self._default_limits[0])
        self.axes.set_ylim(*self._default_limits[1])
        self.draw_idle()


class _PyVistaPreview(QWidget):
    """封装 PyVista Qt 预览控件，保持主窗口调用接口稳定。"""

    def __init__(self) -> None:
        """创建嵌入式 PyVista 场景，并准备统一的回调字段。"""
        super().__init__()
        self._info_callback: Callable[[str], None] | None = None
        self._pixel_callback: Callable[[dict[str, float]], None] | None = None
        self._shutdown = False
        # QtInteractor 默认每 200ms 自动刷新一次。Windows 下窗口隐藏、显示器切换或
        # OpenGL 上下文短暂失效后，该定时器仍会调用 MakeCurrent，导致日志持续刷屏。
        self.plotter = QtInteractor(self, auto_update=False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # pyvistaqt 不同版本暴露的 Qt 控件字段略有差异，这里优先使用官方示例里的 interactor。
        layout.addWidget(getattr(self.plotter, "interactor", self.plotter))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_info_callback(self, callback: Callable[[str], None]) -> None:
        """记录读数回调，供复制视图和绘制完成状态同步使用。"""
        self._info_callback = callback

    def set_pixel_callback(self, callback: Callable[[dict[str, float]], None]) -> None:
        """保留与二维画布一致的接口，后续需要点选三维表面时可直接接入。"""
        self._pixel_callback = callback

    def _emit_info(self, text: str) -> None:
        """把 PyVista 场景状态推送到主窗口读数区。"""
        if self._info_callback is not None:
            self._info_callback(text)

    def copy_current_view_to_clipboard(self) -> None:
        """复制当前 PyVista Qt 控件截图，命令行为与 Matplotlib 画布一致。"""
        pixmap = self.grab()
        QGuiApplication.clipboard().setPixmap(pixmap)
        self._emit_info("当前视图已复制到剪贴板")

    def reset_view(self) -> None:
        """恢复固定单位坐标盒的默认等轴测视角。"""
        if self._shutdown:
            return
        self._set_default_camera()
        self._render_if_visible()

    def _set_default_camera(self) -> None:
        """只更新相机参数，不强制隐藏页签立即渲染。"""
        self.plotter.view_isometric()
        self.plotter.reset_camera(bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0), render=False)
        self.plotter.camera.zoom(1.16)

    def _render_if_visible(self) -> None:
        """仅在控件可见且渲染器仍有效时执行一次显式渲染。"""
        if self._shutdown or not self.isVisible():
            return
        try:
            if getattr(self.plotter, "_closed", False) or getattr(self.plotter, "render_window", None) is None:
                return
            self.plotter.render()
        except (AttributeError, RuntimeError):
            # Qt/VTK 对象可能在窗口关闭事件中先于 Python 包装对象销毁。
            return

    def showEvent(self, event: QShowEvent) -> None:
        """页签真正显示时再渲染，避免隐藏页签争用 OpenGL 上下文。"""
        super().showEvent(event)
        self._render_if_visible()

    def shutdown(self) -> None:
        """停止所有后台渲染并幂等释放 VTK/Qt 资源。"""
        if self._shutdown:
            return
        self._shutdown = True
        render_timer = getattr(self.plotter, "render_timer", None)
        if render_timer is not None:
            render_timer.stop()
        try:
            self.plotter.suppress_rendering = True
            self.plotter.close()
        except (AttributeError, RuntimeError):
            return

    def closeEvent(self, event: QCloseEvent) -> None:
        """控件关闭时显式释放 OpenGL 上下文。"""
        self.shutdown()
        super().closeEvent(event)

    def _prepare_scene(self, title: str) -> None:
        """清空并设置统一的浅色技术预览场景。"""
        self.plotter.clear()
        self.plotter.set_background("#f8fafc")
        self.plotter.add_text(title, position="upper_left", font_size=10, color="#111827")

    def _make_surface_grid(
        self,
        data: np.ndarray,
        max_points: int = 96,
        z_bounds: tuple[float, float] | None = None,
    ) -> tuple[pv.StructuredGrid, np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
        """把高度矩阵降采样并放入固定坐标盒，不改变原始结果数据。"""
        return build_normalized_surface_grid(data, max_points=max_points, z_bounds=z_bounds)

    def _show_fixed_bounds(
        self,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_bounds: tuple[float, float],
    ) -> None:
        """显示固定大小的坐标盒，并用原始数据范围标注刻度。"""
        self.plotter.show_bounds(
            bounds=(0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
            axes_ranges=(
                float(x_coords[0]),
                float(x_coords[-1]),
                float(y_coords[0]),
                float(y_coords[-1]),
                float(z_bounds[0]),
                float(z_bounds[1]),
            ),
            xtitle="X (pixels)",
            ytitle="Y (pixels)",
            ztitle="Height",
            grid="back",
            location="outer",
            all_edges=True,
            color="#64748b",
            font_size=10,
            n_xlabels=4,
            n_ylabels=4,
            n_zlabels=5,
        )


class SurfaceCanvas(_PyVistaPreview):
    """使用 PyVista 绘制单张三维高度表面。"""

    def __init__(self) -> None:
        """初始化单表面预览所需的缓存字段。"""
        super().__init__()
        self._x_coords: np.ndarray | None = None
        self._y_coords: np.ndarray | None = None
        self._z_values: np.ndarray | None = None

    def draw_surface(self, data: np.ndarray, title: str) -> None:
        """将高度矩阵渲染为 PyVista 三维表面预览。"""
        if self._shutdown:
            return
        grid, display_values, x_coords, y_coords, z_bounds = self._make_surface_grid(data)
        self.plotter.suppress_rendering = True
        try:
            self._prepare_scene(title)
            self.plotter.add_mesh(
                grid,
                scalars=display_values.ravel(order="F"),
                cmap="viridis",
                smooth_shading=True,
                show_scalar_bar=True,
                scalar_bar_args={
                    "vertical": False,
                    "position_x": 0.22,
                    "position_y": 0.03,
                    "width": 0.58,
                    "height": 0.08,
                    "title": "",
                },
            )
            self.plotter.show_axes()
            self._show_fixed_bounds(x_coords, y_coords, z_bounds)
            self._set_default_camera()
        finally:
            self.plotter.suppress_rendering = False
        self._render_if_visible()
        self._x_coords = x_coords
        self._y_coords = y_coords
        self._z_values = display_values
        self._emit_info(f"x范围={int(x_coords[0])}-{int(x_coords[-1])}, y范围={int(y_coords[0])}-{int(y_coords[-1])}")


class ComparisonCanvas(_PyVistaPreview):
    """使用 PyVista 叠加显示校正前后的两张高度表面。"""

    def __init__(self) -> None:
        """初始化双表面对比预览所需的缓存字段。"""
        super().__init__()
        self._x_coords: np.ndarray | None = None
        self._y_coords: np.ndarray | None = None
        self._h_values: np.ndarray | None = None
        self._hp_values: np.ndarray | None = None

    def draw_comparison(self, height_map: np.ndarray, height_map_prime: np.ndarray, title: str) -> None:
        """将 h 与 h_prime 以半透明 PyVista 表面叠加预览。"""
        if self._shutdown:
            return
        z_bounds = _combined_finite_bounds(height_map, height_map_prime)
        h_grid, h_values, x_coords, y_coords, _ = self._make_surface_grid(height_map, z_bounds=z_bounds)
        hp_grid, hp_values, _, _, _ = self._make_surface_grid(height_map_prime, z_bounds=z_bounds)
        self.plotter.suppress_rendering = True
        try:
            self._prepare_scene(title)
            self.plotter.add_mesh(
                h_grid,
                color="#2563eb",
                opacity=0.62,
                smooth_shading=True,
                label="h",
            )
            self.plotter.add_mesh(
                hp_grid,
                color="#dc2626",
                opacity=0.58,
                smooth_shading=True,
                label="h_prime",
            )
            self.plotter.add_legend(face="rectangle", bcolor="#ffffff", border=True)
            self.plotter.show_axes()
            self._show_fixed_bounds(x_coords, y_coords, z_bounds)
            self._set_default_camera()
        finally:
            self.plotter.suppress_rendering = False
        self._render_if_visible()
        self._x_coords = x_coords
        self._y_coords = y_coords
        self._h_values = h_values
        self._hp_values = hp_values
        self._emit_info("h 与 h_prime 三维对比已更新")
