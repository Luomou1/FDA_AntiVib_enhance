from __future__ import annotations

from typing import Callable

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QSizePolicy


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


class SurfaceCanvas(_InteractiveCanvas):
    def __init__(self) -> None:
        super().__init__(projection="3d")
        self._x_coords: np.ndarray | None = None
        self._y_coords: np.ndarray | None = None
        self._z_values: np.ndarray | None = None
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("scroll_event", self._on_scroll)

    def draw_surface(self, data: np.ndarray, title: str) -> None:
        reduced, x_coords, y_coords = downsample_for_preview(np.asarray(data), max_points=64)
        self.axes.clear()
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.axes.plot_surface(xx, yy, reduced, cmap="viridis", linewidth=0, antialiased=False)
        self.axes.set_title(title)
        self.axes.set_xlabel("X (pixels)")
        self.axes.set_ylabel("Y (pixels)")
        self.axes.set_zlabel("Height (nm)")
        z_min = float(np.nanmin(reduced))
        z_max = float(np.nanmax(reduced))
        # 对平坦表面先扩成一个极小但非零的 z 轴范围，避免 identical zlims warning。
        z_limits = _ensure_nonzero_limits(z_min, z_max)
        self.axes.set_box_aspect(
            _box_aspect_from_limits(
                float(x_coords[0]),
                float(x_coords[-1]),
                float(y_coords[0]),
                float(y_coords[-1]),
                z_limits[0],
                z_limits[1],
            )
        )
        self.axes.set_xlim3d(float(x_coords[0]), float(x_coords[-1]))
        self.axes.set_ylim3d(float(y_coords[0]), float(y_coords[-1]))
        self.axes.set_zlim3d(*z_limits)
        self._default_limits = (
            (float(x_coords[0]), float(x_coords[-1])),
            (float(y_coords[0]), float(y_coords[-1])),
            z_limits,
            (self.axes.elev, self.axes.azim),
        )
        self._x_coords = x_coords
        self._y_coords = y_coords
        self._z_values = reduced
        self.draw_idle()

    def _on_click(self, event) -> None:
        if (
            event.inaxes is not self.axes
            or self._x_coords is None
            or self._y_coords is None
            or self._z_values is None
            or event.xdata is None
            or event.ydata is None
        ):
            return
        x_index = int(np.argmin(np.abs(self._x_coords - event.xdata)))
        y_index = int(np.argmin(np.abs(self._y_coords - event.ydata)))
        x_value = float(self._x_coords[x_index])
        y_value = float(self._y_coords[y_index])
        z_value = float(self._z_values[y_index, x_index])
        self._emit_info(f"x={int(round(x_value))}, y={int(round(y_value))}, z={z_value:.6f}")
        self._emit_pixel({"x": x_value, "y": y_value, "value": z_value})

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.axes:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x_left, x_right = self.axes.get_xlim3d()
        y_bottom, y_top = self.axes.get_ylim3d()
        z_bottom, z_top = self.axes.get_zlim3d()
        x_center = (x_left + x_right) / 2.0
        y_center = (y_bottom + y_top) / 2.0
        z_center = (z_bottom + z_top) / 2.0
        new_x = _clamp_limits(*_zoom_limits(x_left, x_right, x_center, scale), *self._default_limits[0])
        new_y = _clamp_limits(*_zoom_limits(y_bottom, y_top, y_center, scale), *self._default_limits[1])
        new_z = _clamp_limits(*_zoom_limits(z_bottom, z_top, z_center, scale), *self._default_limits[2])
        self.axes.set_xlim3d(*new_x)
        self.axes.set_ylim3d(*new_y)
        self.axes.set_zlim3d(*new_z)
        self.axes.set_box_aspect(_box_aspect_from_limits(*new_x, *new_y, *new_z))
        self.draw_idle()

    def reset_view(self) -> None:
        if self._default_limits is None:
            return
        self.axes.set_xlim3d(*self._default_limits[0])
        self.axes.set_ylim3d(*self._default_limits[1])
        self.axes.set_zlim3d(*self._default_limits[2])
        self.axes.view_init(*self._default_limits[3])
        self.axes.set_box_aspect(_box_aspect_from_limits(*self._default_limits[0], *self._default_limits[1], *self._default_limits[2]))
        self.draw_idle()


class ComparisonCanvas(_InteractiveCanvas):
    def __init__(self) -> None:
        super().__init__(projection="3d")
        self._x_coords: np.ndarray | None = None
        self._y_coords: np.ndarray | None = None
        self._h_values: np.ndarray | None = None
        self._hp_values: np.ndarray | None = None
        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("scroll_event", self._on_scroll)

    def draw_comparison(self, height_map: np.ndarray, height_map_prime: np.ndarray, title: str) -> None:
        reduced_h, x_coords, y_coords = downsample_for_preview(np.asarray(height_map), max_points=64)
        reduced_hp, _, _ = downsample_for_preview(np.asarray(height_map_prime), max_points=64)
        self.axes.clear()
        xx, yy = np.meshgrid(x_coords, y_coords)
        self.axes.plot_surface(xx, yy, reduced_h, color="#1f77b4", alpha=0.6, linewidth=0, antialiased=False)
        self.axes.plot_surface(xx, yy, reduced_hp, color="#d94841", alpha=0.6, linewidth=0, antialiased=False)
        self.axes.set_title(title)
        self.axes.set_xlabel("X (pixels)")
        self.axes.set_ylabel("Y (pixels)")
        self.axes.set_zlabel("Height (nm)")
        z_min = min(float(np.nanmin(reduced_h)), float(np.nanmin(reduced_hp)))
        z_max = max(float(np.nanmax(reduced_h)), float(np.nanmax(reduced_hp)))
        # 对比图也可能出现两张表面都完全平坦的情况，这里复用同一套兜底逻辑。
        z_limits = _ensure_nonzero_limits(z_min, z_max)
        self.axes.set_xlim3d(float(x_coords[0]), float(x_coords[-1]))
        self.axes.set_ylim3d(float(y_coords[0]), float(y_coords[-1]))
        self.axes.set_zlim3d(*z_limits)
        self.axes.set_box_aspect(
            _box_aspect_from_limits(
                float(x_coords[0]),
                float(x_coords[-1]),
                float(y_coords[0]),
                float(y_coords[-1]),
                z_limits[0],
                z_limits[1],
            )
        )
        self._default_limits = (
            (float(x_coords[0]), float(x_coords[-1])),
            (float(y_coords[0]), float(y_coords[-1])),
            z_limits,
            (self.axes.elev, self.axes.azim),
        )
        self._x_coords = x_coords
        self._y_coords = y_coords
        self._h_values = reduced_h
        self._hp_values = reduced_hp
        self.draw_idle()

    def _on_click(self, event) -> None:
        if (
            event.inaxes is not self.axes
            or self._x_coords is None
            or self._y_coords is None
            or self._h_values is None
            or self._hp_values is None
            or event.xdata is None
            or event.ydata is None
        ):
            return
        x_index = int(np.argmin(np.abs(self._x_coords - event.xdata)))
        y_index = int(np.argmin(np.abs(self._y_coords - event.ydata)))
        x_value = float(self._x_coords[x_index])
        y_value = float(self._y_coords[y_index])
        h_value = float(self._h_values[y_index, x_index])
        hp_value = float(self._hp_values[y_index, x_index])
        self._emit_info(
            f"x={int(round(x_value))}, y={int(round(y_value))}, h={h_value:.6f}, h_prime={hp_value:.6f}"
        )
        self._emit_pixel({"x": x_value, "y": y_value, "h": h_value, "h_prime": hp_value})

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.axes:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x_left, x_right = self.axes.get_xlim3d()
        y_bottom, y_top = self.axes.get_ylim3d()
        z_bottom, z_top = self.axes.get_zlim3d()
        x_center = (x_left + x_right) / 2.0
        y_center = (y_bottom + y_top) / 2.0
        z_center = (z_bottom + z_top) / 2.0
        new_x = _clamp_limits(*_zoom_limits(x_left, x_right, x_center, scale), *self._default_limits[0])
        new_y = _clamp_limits(*_zoom_limits(y_bottom, y_top, y_center, scale), *self._default_limits[1])
        new_z = _clamp_limits(*_zoom_limits(z_bottom, z_top, z_center, scale), *self._default_limits[2])
        self.axes.set_xlim3d(*new_x)
        self.axes.set_ylim3d(*new_y)
        self.axes.set_zlim3d(*new_z)
        self.axes.set_box_aspect(_box_aspect_from_limits(*new_x, *new_y, *new_z))
        self.draw_idle()

    def reset_view(self) -> None:
        if self._default_limits is None:
            return
        self.axes.set_xlim3d(*self._default_limits[0])
        self.axes.set_ylim3d(*self._default_limits[1])
        self.axes.set_zlim3d(*self._default_limits[2])
        self.axes.view_init(*self._default_limits[3])
        self.axes.set_box_aspect(_box_aspect_from_limits(*self._default_limits[0], *self._default_limits[1], *self._default_limits[2]))
        self.draw_idle()
