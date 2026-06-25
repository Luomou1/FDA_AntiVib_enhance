from __future__ import annotations
"""单像素分析可视化窗口。

该窗口用于展示用户点击某个像素后得到的完整频谱与相位诊断图，
是主窗口中的热图读数功能的下游详情页。
"""

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from app.gui.mpl_font import configure_matplotlib_fonts

FONT_PROP = configure_matplotlib_fonts()
TAU = 2.0 * np.pi


def _set_data_xlim_from_zero(axes, x_values: np.ndarray) -> None:
    """让数据横轴从 0 延伸到最后的数据坐标，不保留 Matplotlib 自动边距。"""
    values = np.asarray(x_values, dtype=np.float64)
    finite_values = values[np.isfinite(values)]
    upper = float(np.max(finite_values)) if finite_values.size else 1.0
    axes.set_xlim(0.0, max(upper, 1e-9))


class PixelAnalysisWindow(QMainWindow):
    """承载单像素信号与频谱分析图组的独立窗口。"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("\u50cf\u7d20\u5206\u6790")
        self.setWindowFlag(Qt.Tool, True)
        self._build_ui()

    def _build_ui(self) -> None:
        """初始化窗口布局和 5 个核心子图坐标轴。"""
        root = QWidget()
        layout = QVBoxLayout(root)
        self.setCentralWidget(root)

        self.header_label = QLabel("\u8bf7\u9009\u62e9\u4e00\u4e2a\u50cf\u7d20\u70b9")
        layout.addWidget(self.header_label)

        # 弹窗首次刷新时可能尚未真正显示，手动布局比 constrained_layout
        # 更适合嵌入式 Qt 画布，避免零尺寸 canvas 触发布局警告。
        self.figure = Figure(figsize=(12, 10), dpi=100, constrained_layout=False)
        self.figure.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.94, hspace=0.48, wspace=0.30)
        gridspec = self.figure.add_gridspec(3, 2)
        self.raw_axes = self.figure.add_subplot(gridspec[0, 0])
        self.dc_axes = self.figure.add_subplot(gridspec[0, 1])
        self.amp_axes = self.figure.add_subplot(gridspec[1, 0])
        self.phase_raw_axes = self.figure.add_subplot(gridspec[1, 1])
        self.phase_unwrapped_axes = self.figure.add_subplot(gridspec[2, :])
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def update_analysis(self, payload: dict, layer_name: str, fitting_method: str, unwrap_method: str) -> None:
        """用新像素分析结果刷新整套图形与标题说明。"""
        self.header_label.setText(
            f"\u56fe\u5c42: {layer_name} | x={payload['x']}, y={payload['y']} | "
            f"\u62df\u5408={fitting_method} | \u89e3\u5305\u88f9={unwrap_method}"
        )
        phase_raw_cycles = np.asarray(payload["phase_raw_y"], dtype=np.float32) / TAU
        phase_unwrapped_cycles = np.asarray(payload["phase_unwrapped_y"], dtype=np.float32) / TAU
        fit_phase_cycles = np.asarray(payload.get("fit_phase_y"), dtype=np.float32) / TAU if payload.get("fit_phase_y") is not None else None
        fit_mask_phase_cycles = (
            np.asarray(payload.get("fit_mask_phase_y"), dtype=np.float32) / TAU
            if payload.get("fit_mask_phase_y") is not None
            else None
        )

        self.raw_axes.clear()
        self.raw_axes.plot(payload["signal_x"], payload["signal_raw_y"], color="#1f77b4", linewidth=1.2)
        self.raw_axes.set_title("相干剖面", fontproperties=FONT_PROP)
        self.raw_axes.set_xlabel("\u626b\u63cf\u4f4d\u7f6e z (um)", fontproperties=FONT_PROP)
        self.raw_axes.set_ylabel("强度 (a.u.)", fontproperties=FONT_PROP)
        _set_data_xlim_from_zero(self.raw_axes, payload["signal_x"])

        self.dc_axes.clear()
        self.dc_axes.plot(payload["signal_x"], payload["signal_dc_y"], color="#d94841", linewidth=1.2)
        self.dc_axes.set_title("相干剖面（已去除 DC）", fontproperties=FONT_PROP)
        self.dc_axes.set_xlabel("\u626b\u63cf\u4f4d\u7f6e z (um)", fontproperties=FONT_PROP)
        self.dc_axes.set_ylabel("强度 (a.u.)", fontproperties=FONT_PROP)
        _set_data_xlim_from_zero(self.dc_axes, payload["signal_x"])

        self.amp_axes.clear()
        self.amp_axes.plot(payload["k_x"], payload["amplitude_y"], color="black", linewidth=1.2)
        self.amp_axes.scatter([payload["k0_x"]], [payload["k0_y"]], color="red", zorder=3)
        self.amp_axes.annotate(
            f"k0={payload['k0_x']:.3f}",
            (payload["k0_x"], payload["k0_y"]),
            xytext=(8, 8),
            textcoords="offset points",
            color="red",
        )
        self.amp_axes.set_title("振幅频谱", fontproperties=FONT_PROP)
        self.amp_axes.set_xlabel("\u6ce2\u6570 k (rad/um)", fontproperties=FONT_PROP)
        self.amp_axes.set_ylabel("相对振幅 (a.u.)", fontproperties=FONT_PROP)
        _set_data_xlim_from_zero(self.amp_axes, payload["k_x"])

        self.phase_raw_axes.clear()
        self.phase_raw_axes.plot(payload["k_x"], phase_raw_cycles, color="#9467bd", linewidth=1.2)
        self.phase_raw_axes.set_title("相位频谱", fontproperties=FONT_PROP)
        self.phase_raw_axes.set_xlabel("\u6ce2\u6570 k (rad/um)", fontproperties=FONT_PROP)
        self.phase_raw_axes.set_ylabel("相位 (cycles)", fontproperties=FONT_PROP)
        _set_data_xlim_from_zero(self.phase_raw_axes, payload["k_x"])

        self.phase_unwrapped_axes.clear()
        if unwrap_method == "global":
            self.phase_unwrapped_axes.plot(
                payload["k_x"],
                phase_unwrapped_cycles,
                color="#d94841",
                linewidth=1.2,
            )
            if "fit_mask_k_x" in payload and fit_mask_phase_cycles is not None:
                self.phase_unwrapped_axes.scatter(
                    payload["fit_mask_k_x"],
                    fit_mask_phase_cycles,
                    color="#1f77b4",
                    s=18,
                    zorder=3,
                    label="拟合区间",
                )
            if "fit_k_x" in payload and fit_phase_cycles is not None:
                self.phase_unwrapped_axes.plot(
                    payload["fit_k_x"],
                    fit_phase_cycles,
                    color="black",
                    linewidth=1.4,
                    linestyle="--",
                    label="拟合曲线",
                )
        else:
            fit_k_x = payload.get("fit_k_x")
            fit_phase_y = fit_phase_cycles
            fit_mask_k_x = payload.get("fit_mask_k_x")
            fit_mask_phase_y = fit_mask_phase_cycles
            if fit_mask_k_x is not None and fit_mask_phase_y is not None:
                self.phase_unwrapped_axes.scatter(
                    fit_mask_k_x,
                    fit_mask_phase_y,
                    color="black",
                    s=42,
                    zorder=3,
                    label="Unwrapped Phase",
                )
            if fit_k_x is not None and fit_phase_y is not None:
                self.phase_unwrapped_axes.plot(
                    fit_k_x,
                    fit_phase_y,
                    color="red",
                    linewidth=1.6,
                    linestyle="-",
                    label="Best-fit-line",
                )
            self.phase_unwrapped_axes.axvline(
                payload["k0_x"],
                color="#ff6b6b",
                linewidth=1.4,
                linestyle="--",
                label="K0",
            )
            if fit_k_x is not None and fit_phase_y is not None and len(fit_k_x) >= 2:
                slope = float(np.polyfit(fit_k_x, fit_phase_y, deg=1)[0])
                self.phase_unwrapped_axes.text(
                    0.60,
                    0.12,
                    f"G0 = {slope:.6f} cycles/(rad/um)",
                    transform=self.phase_unwrapped_axes.transAxes,
                    color="red",
                    fontsize=12,
                    bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.9},
                )
                p_min = float(min(np.min(fit_mask_phase_y), np.min(fit_phase_y)))
                p_max = float(max(np.max(fit_mask_phase_y), np.max(fit_phase_y)))
                p_span = max(1e-6, p_max - p_min)
                self.phase_unwrapped_axes.set_ylim(p_min - 0.15 * p_span, p_max + 0.15 * p_span)
        self.phase_unwrapped_axes.set_title("相位剖面", fontproperties=FONT_PROP)
        self.phase_unwrapped_axes.set_xlabel("\u6ce2\u6570 k (rad/um)", fontproperties=FONT_PROP)
        self.phase_unwrapped_axes.set_ylabel("相位 (cycles)", fontproperties=FONT_PROP)
        _set_data_xlim_from_zero(self.phase_unwrapped_axes, payload["k_x"])
        if len(self.phase_unwrapped_axes.lines) > 1 or len(self.phase_unwrapped_axes.collections) > 0:
            self.phase_unwrapped_axes.legend(prop=FONT_PROP, loc="best")

        self.canvas.draw_idle()
