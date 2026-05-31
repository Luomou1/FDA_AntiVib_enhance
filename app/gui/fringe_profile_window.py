from __future__ import annotations
"""条纹级次剖面窗口。

该窗口专门服务于 `fringe_order_map` 的点击诊断：
- 点击条纹级次图中的任意一点
- 弹出独立窗口
- 同时显示该点所在行的 X 向级次剖面、所在列的 Y 向级次剖面
"""

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QLabel, QMainWindow, QTabWidget, QVBoxLayout, QWidget

from app.gui.mpl_font import configure_matplotlib_fonts

FONT_PROP = configure_matplotlib_fonts()
TAU = 2.0 * np.pi


class FringeProfileWindow(QMainWindow):
    """承载条纹级次 X/Y 双向剖面的独立窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("条纹级次剖面")
        self._tab_pages: dict[str, QWidget] = {}
        self._single_profile_colors: dict[str, str] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        """初始化标题区、页签和诊断图组。"""
        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        self.header_label = QLabel("请先在条纹级次图上点击一个像素点")
        layout.addWidget(self.header_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)

        profile_page = QWidget()
        profile_layout = QVBoxLayout(profile_page)
        self.figure = Figure(figsize=(10, 8), dpi=100, constrained_layout=True)
        gridspec = self.figure.add_gridspec(2, 1)
        self.x_axes = self.figure.add_subplot(gridspec[0, 0])
        self.y_axes = self.figure.add_subplot(gridspec[1, 0])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        profile_layout.addWidget(self.toolbar)
        profile_layout.addWidget(self.canvas, 1)
        self._tab_pages["fringe_order_map"] = profile_page
        self.tabs.addTab(profile_page, "条纹级次剖面")

        self._single_profile_figures: dict[str, Figure] = {}
        self._single_profile_canvases: dict[str, FigureCanvasQTAgg] = {}
        self._single_profile_axes: dict[tuple[str, str], object] = {}
        self._single_profile_summary_labels: dict[str, QLabel] = {}
        for key, tab_title, color in (
            ("phi0_map", "phase profile", "#6a4c93"),
            ("theta_map", "Coherence Profile", "#2c7a7b"),
            ("final_height_phase_map", "Final Height Profile (in units of phase)", "#4f772d"),
            ("phase_gap_raw", "Disconnected Phase Gap", "#1f77b4"),
            ("phase_gap_final", "Final Phase Gap", "#d97706"),
        ):
            self._build_single_profile_tab(key=key, tab_title=tab_title, color=color)

        diagnostic_page = QWidget()
        diagnostic_layout = QVBoxLayout(diagnostic_page)
        self.diagnostic_summary_label = QLabel("等待诊断图层")
        self.diagnostic_summary_label.setWordWrap(True)
        diagnostic_layout.addWidget(self.diagnostic_summary_label)

        self.diagnostic_figure = Figure(figsize=(10, 12), dpi=100, constrained_layout=True)
        diagnostic_gridspec = self.diagnostic_figure.add_gridspec(4, 2)
        self._diagnostic_axes = {
            "phase_gap_raw": self.diagnostic_figure.add_subplot(diagnostic_gridspec[0, 0]),
            "phase_gap_final": self.diagnostic_figure.add_subplot(diagnostic_gridspec[0, 1]),
            "merit_map": self.diagnostic_figure.add_subplot(diagnostic_gridspec[1, 0]),
            "confidence_map": self.diagnostic_figure.add_subplot(diagnostic_gridspec[1, 1]),
            "theta_map": self.diagnostic_figure.add_subplot(diagnostic_gridspec[2, 0]),
            "theta_map_smoothed": self.diagnostic_figure.add_subplot(diagnostic_gridspec[2, 1]),
            "phi0_map": self.diagnostic_figure.add_subplot(diagnostic_gridspec[3, 0]),
            "peak_amplitude_map": self.diagnostic_figure.add_subplot(diagnostic_gridspec[3, 1]),
        }
        self.diagnostic_canvas = FigureCanvasQTAgg(self.diagnostic_figure)
        self.diagnostic_toolbar = NavigationToolbar2QT(self.diagnostic_canvas, self)
        diagnostic_layout.addWidget(self.diagnostic_toolbar)
        diagnostic_layout.addWidget(self.diagnostic_canvas, 1)
        self.tabs.addTab(diagnostic_page, "坏点联查")

        multi_profile_page = QWidget()
        multi_profile_layout = QVBoxLayout(multi_profile_page)
        self.multi_profile_summary_label = QLabel("等待多图层剖面")
        self.multi_profile_summary_label.setWordWrap(True)
        multi_profile_layout.addWidget(self.multi_profile_summary_label)

        self.multi_profile_figure = Figure(figsize=(12, 14), dpi=100, constrained_layout=True)
        multi_profile_gridspec = self.multi_profile_figure.add_gridspec(4, 2)
        self._multi_profile_axes = {
            ("phi0_map", "x"): self.multi_profile_figure.add_subplot(multi_profile_gridspec[0, 0]),
            ("phi0_map", "y"): self.multi_profile_figure.add_subplot(multi_profile_gridspec[0, 1]),
            ("theta_map", "x"): self.multi_profile_figure.add_subplot(multi_profile_gridspec[1, 0]),
            ("theta_map", "y"): self.multi_profile_figure.add_subplot(multi_profile_gridspec[1, 1]),
            ("phase_gap_raw", "x"): self.multi_profile_figure.add_subplot(multi_profile_gridspec[2, 0]),
            ("phase_gap_raw", "y"): self.multi_profile_figure.add_subplot(multi_profile_gridspec[2, 1]),
            ("phase_gap_final", "x"): self.multi_profile_figure.add_subplot(multi_profile_gridspec[3, 0]),
            ("phase_gap_final", "y"): self.multi_profile_figure.add_subplot(multi_profile_gridspec[3, 1]),
        }
        self.multi_profile_canvas = FigureCanvasQTAgg(self.multi_profile_figure)
        self.multi_profile_toolbar = NavigationToolbar2QT(self.multi_profile_canvas, self)
        multi_profile_layout.addWidget(self.multi_profile_toolbar)
        multi_profile_layout.addWidget(self.multi_profile_canvas, 1)
        self.tabs.addTab(multi_profile_page, "多图层剖面")

    def _build_single_profile_tab(self, key: str, tab_title: str, color: str) -> None:
        """为单个图层创建独立的 X/Y 双剖面页。"""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        summary_label = QLabel(f"等待 {tab_title}")
        summary_label.setWordWrap(True)
        page_layout.addWidget(summary_label)

        figure = Figure(figsize=(10, 8), dpi=100, constrained_layout=True)
        gridspec = figure.add_gridspec(2, 1)
        x_axes = figure.add_subplot(gridspec[0, 0])
        y_axes = figure.add_subplot(gridspec[1, 0])
        canvas = FigureCanvasQTAgg(figure)
        toolbar = NavigationToolbar2QT(canvas, self)
        page_layout.addWidget(toolbar)
        page_layout.addWidget(canvas, 1)

        self._single_profile_figures[key] = figure
        self._single_profile_canvases[key] = canvas
        self._single_profile_axes[(key, "x")] = x_axes
        self._single_profile_axes[(key, "y")] = y_axes
        self._single_profile_summary_labels[key] = summary_label
        self._single_profile_colors[key] = color
        self._tab_pages[key] = page
        self.tabs.addTab(page, tab_title)

    def _extract_local_patch(
        self,
        data: np.ndarray,
        x: int,
        y: int,
        radius: int,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """截取以当前点为中心的局部窗口，供坏点联查使用。"""
        row_start = max(0, y - radius)
        row_end = min(data.shape[0], y + radius + 1)
        col_start = max(0, x - radius)
        col_end = min(data.shape[1], x + radius + 1)
        patch = data[row_start:row_end, col_start:col_end]
        return patch, (row_start, row_end, col_start, col_end)

    def _format_value(self, value: float) -> str:
        """统一诊断摘要里的数值格式。"""
        if not np.isfinite(value):
            return "nan"
        return f"{value:.6f}"

    def _convert_map_for_display(self, key: str, data: np.ndarray) -> np.ndarray:
        """把需要以 cycles 展示的相位类图层从 rad 换算为 cycles。"""
        phase_like_keys = {
            "phi0_map",
            "theta_map",
            "theta_map_smoothed",
            "final_height_phase_map",
            "phase_gap_raw",
            "phase_gap_final",
        }
        array = np.asarray(data, dtype=np.float32)
        if key in phase_like_keys:
            return (array / TAU).astype(np.float32)
        return array

    def _draw_diagnostic_map(
        self,
        key: str,
        title: str,
        diagnostic_maps: dict[str, np.ndarray] | None,
        x: int,
        y: int,
        patch_radius: int,
    ) -> str:
        """绘制某一张局部诊断图，并返回当前点的文字摘要。"""
        axes = self._diagnostic_axes[key]
        axes.clear()

        if diagnostic_maps is None or key not in diagnostic_maps:
            axes.set_title(f"{title}（缺失）", fontproperties=FONT_PROP)
            axes.text(0.5, 0.5, "N/A", transform=axes.transAxes, ha="center", va="center")
            axes.set_xticks([])
            axes.set_yticks([])
            return f"{key}=N/A"

        data = self._convert_map_for_display(key, diagnostic_maps[key])
        patch, (row_start, row_end, col_start, col_end) = self._extract_local_patch(data, x=x, y=y, radius=patch_radius)
        local_x = x - col_start
        local_y = y - row_start

        # 这里展示的是局部窗口而不是整图，目的是把坏点附近证据压缩到一次点击能读完的范围。
        image = axes.imshow(
            patch,
            cmap="viridis",
            origin="lower",
            aspect="equal",
        )
        axes.scatter([local_x], [local_y], color="#d94841", s=28, zorder=3)
        axes.set_title(
            f"{title} | x[{col_start}:{col_end}) y[{row_start}:{row_end})",
            fontproperties=FONT_PROP,
            fontsize=10,
        )
        axes.set_xlabel("Local X (pixels)", fontproperties=FONT_PROP)
        axes.set_ylabel("Local Y (pixels)", fontproperties=FONT_PROP)
        self.diagnostic_figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03)

        point_value = float(data[y, x])
        return f"{key}={self._format_value(point_value)}"

    def _draw_map_profile(
        self,
        key: str,
        axis_name: str,
        title: str,
        data_maps: dict[str, np.ndarray] | None,
        x: int,
        y: int,
        color: str,
    ) -> str:
        """绘制指定图层在 X 或 Y 方向上的剖面。"""
        axes = self._multi_profile_axes[(key, axis_name)]
        axes.clear()

        if data_maps is None or key not in data_maps:
            axes.set_title(f"{title}（缺失）", fontproperties=FONT_PROP)
            axes.text(0.5, 0.5, "N/A", transform=axes.transAxes, ha="center", va="center")
            axes.set_xticks([])
            axes.set_yticks([])
            return f"{key}_{axis_name}=N/A"

        data = np.asarray(data_maps[key], dtype=np.float32)
        if axis_name == "x":
            coords = np.arange(data.shape[1], dtype=np.int32)
            values = data[y, :]
            focus_index = x
            xlabel = "X (pixels)"
            direction_label = f"X 向（第 {y} 行）"
        else:
            coords = np.arange(data.shape[0], dtype=np.int32)
            values = data[:, x]
            focus_index = y
            xlabel = "Y (pixels)"
            direction_label = f"Y 向（第 {x} 列）"

        point_value = float(values[focus_index])
        # 这些图层是连续场，不适合用 step；这里用普通折线更容易看局部趋势和背景漂移。
        axes.plot(coords, values, color=color, linewidth=1.3)
        axes.scatter([focus_index], [point_value], color="#d94841", zorder=3, s=28)
        axes.axvline(focus_index, color="#d94841", linestyle="--", linewidth=1.0)
        axes.set_title(f"{title} {direction_label}", fontproperties=FONT_PROP, fontsize=10)
        axes.set_xlabel(xlabel, fontproperties=FONT_PROP)
        ylabel_map = {
            "phi0_map": "Phase (cycles)",
            "theta_map": "Coherence (cycles)",
            "phase_gap_raw": "Phase Gap (cycles)",
            "phase_gap_final": "Phase Gap (cycles)",
        }
        axes.set_ylabel(ylabel_map.get(key, title), fontproperties=FONT_PROP)
        axes.grid(True, linestyle="--", alpha=0.35)
        return f"{key}_{axis_name}={self._format_value(point_value)}"

    def _draw_primary_profile(
        self,
        layer_key: str,
        source_map: np.ndarray,
        x: int,
        y: int,
    ) -> None:
        """刷新顶部主剖面区，用于展示当前被点击的图层。"""
        data = self._convert_map_for_display(layer_key, source_map)
        x_coords = np.arange(data.shape[1], dtype=np.int32)
        y_coords = np.arange(data.shape[0], dtype=np.int32)
        row_profile = data[y, :]
        col_profile = data[:, x]
        point_value = float(data[y, x])

        if layer_key == "fringe_order_map":
            title = "Fringe Order Map"
            ylabel = "Fringe Order (order)"
            color_x = "#1f77b4"
            color_y = "#2c7a7b"
            use_step = True
        else:
            title_map = {
                "phi0_map": "phase profile",
                "theta_map": "Coherence Profile",
                "final_height_phase_map": "Final Height Profile (in units of phase)",
                "phase_gap_raw": "Disconnected Phase Gap",
                "phase_gap_final": "Final Phase Gap",
            }
            ylabel_map = {
                "phi0_map": "Phase (cycles)",
                "theta_map": "Coherence (cycles)",
                "final_height_phase_map": "Phase (cycles)",
                "phase_gap_raw": "Phase Gap (cycles)",
                "phase_gap_final": "Phase Gap (cycles)",
            }
            title = title_map.get(layer_key, layer_key)
            ylabel = ylabel_map.get(layer_key, title)
            color_x = self._single_profile_colors.get(layer_key, "#1f77b4")
            color_y = color_x
            use_step = False

        self.x_axes.clear()
        self.y_axes.clear()

        if use_step:
            self.x_axes.step(x_coords, row_profile, where="mid", color=color_x, linewidth=1.4)
            self.y_axes.step(y_coords, col_profile, where="mid", color=color_y, linewidth=1.4)
        else:
            self.x_axes.plot(x_coords, row_profile, color=color_x, linewidth=1.3)
            self.y_axes.plot(y_coords, col_profile, color=color_y, linewidth=1.3)

        self.x_axes.scatter([x], [point_value], color="#d94841", zorder=3, s=36)
        self.x_axes.axvline(x, color="#d94841", linestyle="--", linewidth=1.0)
        self.x_axes.set_title(f"{title} X 向剖面（第 {y} 行）", fontproperties=FONT_PROP)
        self.x_axes.set_xlabel("X (pixels)", fontproperties=FONT_PROP)
        self.x_axes.set_ylabel(ylabel, fontproperties=FONT_PROP)
        self.x_axes.grid(True, linestyle="--", alpha=0.35)

        self.y_axes.scatter([y], [point_value], color="#d94841", zorder=3, s=36)
        self.y_axes.axvline(y, color="#d94841", linestyle="--", linewidth=1.0)
        self.y_axes.set_title(f"{title} Y 向剖面（第 {x} 列）", fontproperties=FONT_PROP)
        self.y_axes.set_xlabel("Y (pixels)", fontproperties=FONT_PROP)
        self.y_axes.set_ylabel(ylabel, fontproperties=FONT_PROP)
        self.y_axes.grid(True, linestyle="--", alpha=0.35)
        self.canvas.draw_idle()

    def _draw_single_profile_tab(
        self,
        key: str,
        title: str,
        data_maps: dict[str, np.ndarray] | None,
        x: int,
        y: int,
        color: str,
    ) -> None:
        """刷新某一张独立图层剖面页。"""
        summary_label = self._single_profile_summary_labels[key]
        x_axes = self._single_profile_axes[(key, "x")]
        y_axes = self._single_profile_axes[(key, "y")]
        x_axes.clear()
        y_axes.clear()

        if data_maps is None or key not in data_maps:
            summary_label.setText(f"{title} 缺失")
            for axes in (x_axes, y_axes):
                axes.text(0.5, 0.5, "N/A", transform=axes.transAxes, ha="center", va="center")
                axes.set_xticks([])
                axes.set_yticks([])
            self._single_profile_canvases[key].draw_idle()
            return

        data = self._convert_map_for_display(key, data_maps[key])
        x_coords = np.arange(data.shape[1], dtype=np.int32)
        y_coords = np.arange(data.shape[0], dtype=np.int32)
        row_profile = data[y, :]
        col_profile = data[:, x]
        point_value = float(data[y, x])

        # 这里给每张图层单独开页，是为了避免用户在综合页里反复找同一颜色对应哪张图。
        x_axes.plot(x_coords, row_profile, color=color, linewidth=1.3)
        x_axes.scatter([x], [point_value], color="#d94841", zorder=3, s=28)
        x_axes.axvline(x, color="#d94841", linestyle="--", linewidth=1.0)
        x_axes.set_title(f"{title} X 向剖面（第 {y} 行）", fontproperties=FONT_PROP)
        x_axes.set_xlabel("X (pixels)", fontproperties=FONT_PROP)
        ylabel_map = {
            "phi0_map": "Phase (cycles)",
            "theta_map": "Coherence (cycles)",
            "final_height_phase_map": "Phase (cycles)",
            "phase_gap_raw": "Phase Gap (cycles)",
            "phase_gap_final": "Phase Gap (cycles)",
        }
        x_axes.set_ylabel(ylabel_map.get(key, title), fontproperties=FONT_PROP)
        x_axes.grid(True, linestyle="--", alpha=0.35)

        y_axes.plot(y_coords, col_profile, color=color, linewidth=1.3)
        y_axes.scatter([y], [point_value], color="#d94841", zorder=3, s=28)
        y_axes.axvline(y, color="#d94841", linestyle="--", linewidth=1.0)
        y_axes.set_title(f"{title} Y 向剖面（第 {x} 列）", fontproperties=FONT_PROP)
        y_axes.set_xlabel("Y (pixels)", fontproperties=FONT_PROP)
        y_axes.set_ylabel(ylabel_map.get(key, title), fontproperties=FONT_PROP)
        y_axes.grid(True, linestyle="--", alpha=0.35)

        summary_label.setText(
            f"{title}: 点值={self._format_value(point_value)} | "
            f"X向范围=[{self._format_value(float(np.nanmin(row_profile)))}, {self._format_value(float(np.nanmax(row_profile)))}] | "
            f"Y向范围=[{self._format_value(float(np.nanmin(col_profile)))}, {self._format_value(float(np.nanmax(col_profile)))}]"
        )
        self._single_profile_canvases[key].draw_idle()

    def update_profiles(
        self,
        source_map: np.ndarray,
        x: int,
        y: int,
        layer_key: str = "fringe_order_map",
        diagnostic_maps: dict[str, np.ndarray] | None = None,
    ) -> None:
        """根据点击坐标刷新剖面窗口，并同步更新当前保留的所有诊断页。"""
        data = np.asarray(source_map, dtype=np.float32)
        x = int(np.clip(x, 0, data.shape[1] - 1))
        y = int(np.clip(y, 0, data.shape[0] - 1))
        point_value = float(data[y, x])

        self.header_label.setText(
            f"当前图层: {layer_key} | 当前点: x={x}, y={y}, value={self._format_value(point_value)}"
        )
        self._draw_primary_profile(layer_key=layer_key, source_map=data, x=x, y=y)

        # Matplotlib 的 colorbar 会额外创建坐标轴；这里在每次刷新前先删掉旧 colorbar，
        # 避免用户连续点击多个坏点时诊断页不断叠加新的辅助坐标轴。
        main_axes = set(self._diagnostic_axes.values())
        for axes in list(self.diagnostic_figure.axes):
            if axes not in main_axes:
                self.diagnostic_figure.delaxes(axes)

        # 这里额外把与级次判定直接相关的 8 张证据图同步出来，便于区分：
        # - 是 raw/final gap 差值把像素推过了 ±pi 阈值
        # - 还是 phi0 单点错支 / 峰值幅值过低导致局部拟合不稳定
        summary_parts = [
            self._draw_diagnostic_map("phase_gap_raw", "phase_gap_raw", diagnostic_maps, x=x, y=y, patch_radius=10),
            self._draw_diagnostic_map("phase_gap_final", "phase_gap_final", diagnostic_maps, x=x, y=y, patch_radius=10),
            self._draw_diagnostic_map("merit_map", "merit_map", diagnostic_maps, x=x, y=y, patch_radius=10),
            self._draw_diagnostic_map("confidence_map", "confidence_map", diagnostic_maps, x=x, y=y, patch_radius=10),
            self._draw_diagnostic_map("theta_map", "theta_map", diagnostic_maps, x=x, y=y, patch_radius=10),
            self._draw_diagnostic_map(
                "theta_map_smoothed",
                "theta_map_smoothed",
                diagnostic_maps,
                x=x,
                y=y,
                patch_radius=10,
            ),
            self._draw_diagnostic_map("phi0_map", "phi0_map", diagnostic_maps, x=x, y=y, patch_radius=10),
            self._draw_diagnostic_map(
                "peak_amplitude_map",
                "peak_amplitude_map",
                diagnostic_maps,
                x=x,
                y=y,
                patch_radius=10,
            ),
        ]
        self.diagnostic_summary_label.setText(" | ".join(summary_parts))
        self.diagnostic_canvas.draw_idle()

        multi_profile_summary = [
            self._draw_map_profile("phi0_map", "x", "phi0_map", diagnostic_maps, x=x, y=y, color="#6a4c93"),
            self._draw_map_profile("phi0_map", "y", "phi0_map", diagnostic_maps, x=x, y=y, color="#6a4c93"),
            self._draw_map_profile("theta_map", "x", "theta_map", diagnostic_maps, x=x, y=y, color="#2c7a7b"),
            self._draw_map_profile("theta_map", "y", "theta_map", diagnostic_maps, x=x, y=y, color="#2c7a7b"),
            self._draw_map_profile("phase_gap_raw", "x", "phase_gap_raw", diagnostic_maps, x=x, y=y, color="#1f77b4"),
            self._draw_map_profile("phase_gap_raw", "y", "phase_gap_raw", diagnostic_maps, x=x, y=y, color="#1f77b4"),
            self._draw_map_profile(
                "phase_gap_final",
                "x",
                "phase_gap_final",
                diagnostic_maps,
                x=x,
                y=y,
                color="#d97706",
            ),
            self._draw_map_profile(
                "phase_gap_final",
                "y",
                "phase_gap_final",
                diagnostic_maps,
                x=x,
                y=y,
                color="#d97706",
            ),
        ]
        self.multi_profile_summary_label.setText(" | ".join(multi_profile_summary))
        self.multi_profile_canvas.draw_idle()

        self._draw_single_profile_tab("phi0_map", "phase profile", diagnostic_maps, x=x, y=y, color="#6a4c93")
        self._draw_single_profile_tab("theta_map", "Coherence Profile", diagnostic_maps, x=x, y=y, color="#2c7a7b")
        self._draw_single_profile_tab(
            "final_height_phase_map",
            "Final Height Profile (in units of phase)",
            diagnostic_maps,
            x=x,
            y=y,
            color="#4f772d",
        )
        self._draw_single_profile_tab("phase_gap_raw", "Disconnected Phase Gap", diagnostic_maps, x=x, y=y, color="#1f77b4")
        self._draw_single_profile_tab("phase_gap_final", "Final Phase Gap", diagnostic_maps, x=x, y=y, color="#d97706")
        self.tabs.setCurrentWidget(self._tab_pages.get(layer_key, self._tab_pages["fringe_order_map"]))
