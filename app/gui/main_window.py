from __future__ import annotations
"""主窗口模块。

该文件负责把参数面板、结果图层、后台线程 worker 和导出功能整合到一个
统一的桌面界面中，是 GUI 端的主要控制器。
"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.pixel_analysis import build_pixel_analysis
from app.core.kernel import resolve_fft_length
from app.core.result_model import AnalysisResult
from app.gui.fringe_profile_window import FringeProfileWindow
from app.gui.pixel_window import PixelAnalysisWindow
from app.pipeline.io import collect_image_files
from app.gui.worker import AnalysisWorker, GlobalK0Worker
from app.pipeline.scan_log import load_actual_positions_um
from app.pipeline.session import AnalysisParams, AnalysisSession
from app.plotting.paper import save_paper_figures
from app.plotting.preview import ComparisonCanvas, HeatmapCanvas, SpectrumCanvas, SurfaceCanvas

TAU = 2.0 * np.pi


class PlotToolbar(NavigationToolbar2QT):
    """为绘图工具栏补充统一 Home 行为。"""

    def home(self, *args) -> None:
        """优先调用画布自定义复位逻辑，缺失时退回默认 Home。"""
        if hasattr(self.canvas, "reset_view"):
            self.canvas.reset_view()
        else:
            super().home(*args)


class ExportSelectionDialog(QDialog):
    """导出选择对话框，允许用户勾选要导出的键。"""

    def __init__(self, title: str, options: list[tuple[str, str]], parent=None) -> None:
        """根据给定选项动态生成多选框。"""
        super().__init__(parent)
        self.setWindowTitle(title)
        self._checkboxes: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("请选择要导出的内容："))

        for key, label in options:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True)
            self._checkboxes[key] = checkbox
            layout.addWidget(checkbox)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def selected_keys(self) -> list[str]:
        """返回当前被勾选的导出键。"""
        return [key for key, checkbox in self._checkboxes.items() if checkbox.isChecked()]


class AutoK0Dialog(QDialog):
    """自动估计 K0 参数配置对话框。"""

    def __init__(self, parent=None) -> None:
        """初始化候选像素比例控件。"""
        super().__init__(parent)
        self.setWindowTitle("自动定 K0")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.candidate_ratio = QDoubleSpinBox()
        self.candidate_ratio.setDecimals(1)
        self.candidate_ratio.setRange(0.1, 100.0)
        self.candidate_ratio.setValue(10.0)
        self.candidate_ratio.setSuffix("%")
        form.addRow("候选像素比例", self.candidate_ratio)

        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, float | int | str]:
        """导出当前对话框配置值，供 worker 启动时使用。"""
        return {
            "candidate_ratio": float(self.candidate_ratio.value()) / 100.0,
        }


def _populate_zero_padding_combo(combo: QComboBox, frame_count: int | None = None) -> None:
    """填充统一补零选项；已知帧数时直接显示具体 FFT 长度。"""
    current_data = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    n = int(frame_count) if frame_count is not None and int(frame_count) > 1 else None
    combo.addItem("1x", "none")
    if n is None:
        combo.addItem("2^+1", "pow2_1")
        combo.addItem("2^+2", "pow2_2")
        combo.addItem("2^+3", "pow2_3")
    else:
        combo.addItem(str(resolve_fft_length(n, "pow2_1")), "pow2_1")
        combo.addItem(str(resolve_fft_length(n, "pow2_2")), "pow2_2")
        combo.addItem(str(resolve_fft_length(n, "pow2_3")), "pow2_3")
    combo.addItem("2x", "factor_2")
    combo.addItem("4x", "factor_4")
    combo.addItem("8x", "factor_8")
    if current_data is not None:
        index = combo.findData(current_data)
        if index >= 0:
            combo.setCurrentIndex(index)
    combo.blockSignals(False)


class MainWindow(QMainWindow):
    """应用主窗口：参数输入、分析调度、可视化与导出统一入口。"""

    def __init__(self) -> None:
        """初始化状态字段并构建整套界面。"""
        super().__init__()
        self.setWindowTitle("FDA_Antivib")
        self._thread: QThread | None = None
        self._worker: AnalysisWorker | None = None
        self._k0_worker: GlobalK0Worker | None = None
        self._result: AnalysisResult | None = None
        self._auto_k0_result: dict | None = None
        self._analysis_cube = None
        self._analysis_params: AnalysisParams | None = None
        self._pixel_window: PixelAnalysisWindow | None = None
        self._fringe_profile_window: FringeProfileWindow | None = None
        self._plot_info_labels: dict[str, QLabel] = {}
        self._plot_tabs: dict[str, QWidget] = {}
        self._diagnostic_tab: QWidget | None = None
        self._diagnostic_summary: QPlainTextEdit | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        """构建根布局并挂接左中右三栏与底部状态区。"""
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)
        self.setCentralWidget(root)

        top_splitter = QSplitter()
        root_layout.addWidget(top_splitter, 1)

        top_splitter.addWidget(self._build_left_panel())
        top_splitter.addWidget(self._build_center_panel())
        top_splitter.addWidget(self._build_right_panel())
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setStretchFactor(2, 0)
        top_splitter.setSizes([280, 980, 280])

        root_layout.addWidget(self._build_bottom_panel(), 0)

    def _build_left_panel(self) -> QWidget:
        """构建参数设置与任务控制面板。"""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        panel.setMinimumWidth(270)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("分析参数")
        title.setStyleSheet("font-weight: 600; font-size: 15px;")
        layout.addWidget(title)

        form = QFormLayout()

        self.data_source = QComboBox()
        self.data_source.addItem("图像文件夹", "image_folder")
        self.data_source.addItem("MAT 数据", "mat_file")
        self.data_source.currentIndexChanged.connect(self._update_data_source_ui)
        form.addRow("数据来源", self.data_source)

        self.folder_edit = QLineEdit()
        self.folder_button = QPushButton("选择文件夹")
        self.folder_button.clicked.connect(self._choose_folder)
        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.addWidget(self.folder_edit, 1)
        folder_layout.addWidget(self.folder_button)
        form.addRow("图像文件夹", folder_row)

        self.mat_edit = QLineEdit()
        self.mat_button = QPushButton("选择 MAT")
        self.mat_button.clicked.connect(self._choose_mat_file)
        mat_row = QWidget()
        mat_layout = QHBoxLayout(mat_row)
        mat_layout.setContentsMargins(0, 0, 0, 0)
        mat_layout.addWidget(self.mat_edit, 1)
        mat_layout.addWidget(self.mat_button)
        form.addRow("MAT 文件", mat_row)

        self.start_height = QDoubleSpinBox()
        self.start_height.setDecimals(4)
        self.start_height.setRange(-1e6, 1e6)
        self.start_height.setValue(0.0)
        form.addRow("起始高度（um）", self.start_height)

        self.sampling_mode = QComboBox()
        self.sampling_mode.addItem("均匀采样", "uniform")
        self.sampling_mode.addItem("非均匀采样", "nonuniform")
        self.sampling_mode.currentIndexChanged.connect(self._update_sampling_mode_ui)
        form.addRow("采样模式", self.sampling_mode)

        self.step_size = QDoubleSpinBox()
        self.step_size.setDecimals(4)
        self.step_size.setRange(0.0001, 1e6)
        self.step_size.setValue(0.056)
        form.addRow("步长（um）", self.step_size)

        self.scan_log_edit = QLineEdit()
        self.scan_log_button = QPushButton("选择 scan_log")
        self.scan_log_button.clicked.connect(self._choose_scan_log)
        self.scan_log_edit.textChanged.connect(self._refresh_scan_log_summary)
        scan_log_row = QWidget()
        scan_log_layout = QHBoxLayout(scan_log_row)
        scan_log_layout.setContentsMargins(0, 0, 0, 0)
        scan_log_layout.addWidget(self.scan_log_edit, 1)
        scan_log_layout.addWidget(self.scan_log_button)
        form.addRow("scan_log", scan_log_row)

        self.scan_log_summary_label = QLabel("未加载 scan_log")
        self.scan_log_summary_label.setWordWrap(True)
        form.addRow("位移摘要", self.scan_log_summary_label)

        self.fixed_k0 = QDoubleSpinBox()
        self.fixed_k0.setDecimals(6)
        self.fixed_k0.setRange(0.0, 1e6)
        self.fixed_k0.setValue(0.0)
        form.addRow("固定 K0", self.fixed_k0)

        self.window_size = QSpinBox()
        self.window_size.setRange(1, 200)
        self.window_size.setValue(9)
        form.addRow("窗口大小", self.window_size)

        self.analysis_window_name = QComboBox()
        self.analysis_window_name.addItem("Hamming", "hamming")
        self.analysis_window_name.addItem("None", "none")
        form.addRow("分析窗函数", self.analysis_window_name)

        self.analysis_zero_padding_mode = QComboBox()
        _populate_zero_padding_combo(self.analysis_zero_padding_mode, frame_count=None)
        self.analysis_zero_padding_mode.setCurrentIndex(1)
        form.addRow("正式分析补零", self.analysis_zero_padding_mode)

        self.expand_active_range_checkbox = QCheckBox("扩大有效范围")
        self.expand_active_range_checkbox.setChecked(False)
        form.addRow("有效范围扩展", self.expand_active_range_checkbox)

        self.active_range_expansion_frames = QSpinBox()
        self.active_range_expansion_frames.setRange(0, 10000)
        self.active_range_expansion_frames.setValue(35)
        self.active_range_expansion_frames.setEnabled(False)
        self.expand_active_range_checkbox.toggled.connect(self.active_range_expansion_frames.setEnabled)
        form.addRow("左右扩展帧数", self.active_range_expansion_frames)

        self.fitting = QComboBox()
        self.fitting.addItem("简单拟合", "simple")
        self.fitting.addItem("二次拟合", "quadratic")
        self.fitting.addItem("加权拟合", "weighted")
        form.addRow("拟合方式", self.fitting)

        self.unwrap = QComboBox()
        self.unwrap.addItem("全局解包裹", "global")
        self.unwrap.addItem("Itoh 局部解包裹", "itoh")
        self.unwrap.addItem("GR 解包裹", "gr")
        self.unwrap.addItem("PDA 局部解包裹", "pda")
        self.unwrap.addItem("Branch Search 局部解包裹", "branch_search")
        form.addRow("解包裹方式", self.unwrap)

        self.analysis_mode_label = QLabel("FDA")
        form.addRow("方法", self.analysis_mode_label)

        self.phase_gap_method = QComboBox()
        self.phase_gap_method.addItem("FDA", "FDA")
        self.phase_gap_method.addItem("PhaseGap - Quality guided", "quality_guided")
        self.phase_gap_method.addItem("PhaseGap - Circular average", "circular_average")
        self.phase_gap_method.addItem("PhaseGap - Robust model fit", "robust_model_fit")
        self.phase_gap_method.addItem("PhaseGap - Branch cut", "branch_cut")
        self.phase_gap_method.addItem("PhaseGap - Weighted least squares", "weighted_least_squares")
        self.phase_gap_method.addItem("PhaseGap - Minimum Lp", "minimum_lp")
        self.phase_gap_method.currentIndexChanged.connect(self._update_analysis_mode_ui)
        form.addRow("工作流", self.phase_gap_method)
        layout.addLayout(form)

        self.auto_k0_button = QPushButton("自动定 K0")
        self.auto_k0_button.clicked.connect(self._start_auto_k0)
        self.start_button = QPushButton("开始分析")
        self.start_button.clicked.connect(self._start_analysis)
        self.export_text_button = QPushButton("导出数据")
        self.export_text_button.clicked.connect(self._export_text)
        self.export_text_button.setEnabled(False)
        self.export_figures_button = QPushButton("导出图片")
        self.export_figures_button.clicked.connect(self._export_figures)
        self.export_figures_button.setEnabled(False)

        layout.addWidget(self.auto_k0_button)
        layout.addWidget(self.start_button)
        layout.addWidget(self.export_text_button)
        layout.addWidget(self.export_figures_button)
        layout.addStretch(1)
        self._update_data_source_ui()
        self._update_sampling_mode_ui()
        return panel

    def _build_center_panel(self) -> QWidget:
        """
        构建中间主视图区。

        主结果对 FDA 和 PhaseGap 共用；PhaseGap 模式下额外显示中间诊断图层。
        """
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.tabs = QTabWidget()
        self.h_prime_canvas = HeatmapCanvas()
        self.h_canvas = HeatmapCanvas()
        self.phi0_canvas = HeatmapCanvas()
        self.theta_canvas = HeatmapCanvas()
        self.final_height_phase_canvas = HeatmapCanvas()
        self.merit_canvas = HeatmapCanvas()
        self.phase_gap_raw_canvas = HeatmapCanvas()
        self.phase_gap_final_canvas = HeatmapCanvas()
        self.fringe_order_canvas = HeatmapCanvas()
        self.confidence_canvas = HeatmapCanvas()
        self.surface_canvas = SurfaceCanvas()
        self.h_prime_surface_canvas = SurfaceCanvas()
        self.comparison_canvas = ComparisonCanvas()
        self.k0_canvas = SpectrumCanvas()

        self._plot_tabs = {
            "h": self._build_plot_tab(self.h_canvas, "h"),
            "h_prime": self._build_plot_tab(self.h_prime_canvas, "h_prime"),
            "phi0": self._build_plot_tab(self.phi0_canvas, "phi0"),
            "theta": self._build_plot_tab(self.theta_canvas, "theta"),
            "final_height_phase": self._build_plot_tab(self.final_height_phase_canvas, "final_height_phase"),
            "merit": self._build_plot_tab(self.merit_canvas, "merit"),
            "phase_gap_raw": self._build_plot_tab(self.phase_gap_raw_canvas, "phase_gap_raw"),
            "phase_gap_final": self._build_plot_tab(self.phase_gap_final_canvas, "phase_gap_final"),
            "fringe_order": self._build_plot_tab(self.fringe_order_canvas, "fringe_order"),
            "confidence": self._build_plot_tab(self.confidence_canvas, "confidence"),
            "surface": self._build_plot_tab(self.surface_canvas, "surface"),
            "h_prime_surface": self._build_plot_tab(self.h_prime_surface_canvas, "h_prime_surface"),
            "comparison": self._build_plot_tab(self.comparison_canvas, "comparison"),
            "k0": self._build_plot_tab(self.k0_canvas, "k0"),
        }
        self.k0_tab = self._plot_tabs["k0"]
        self._diagnostic_tab = self._build_diagnostic_tab()
        self._update_analysis_mode_ui()
        layout.addWidget(self.tabs, 1)
        return panel

    def _build_right_panel(self) -> QWidget:
        """构建右侧数据读数区。"""
        self.value_panel = QFrame()
        self.value_panel.setFrameShape(QFrame.StyledPanel)
        self.value_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.value_panel.setMinimumWidth(270)
        layout = QVBoxLayout(self.value_panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("数据")
        title.setStyleSheet("font-weight: 600; font-size: 15px;")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.addWidget(QLabel("当前图层"), 0, 0)
        self.layer_label = QLabel("-")
        grid.addWidget(self.layer_label, 0, 1)
        grid.addWidget(QLabel("X"), 1, 0)
        self.x_label = QLabel("-")
        grid.addWidget(self.x_label, 1, 1)
        grid.addWidget(QLabel("Y"), 2, 0)
        self.y_label = QLabel("-")
        grid.addWidget(self.y_label, 2, 1)
        grid.addWidget(QLabel("值"), 3, 0)
        self.value_label = QLabel("-")
        grid.addWidget(self.value_label, 3, 1)
        grid.addWidget(QLabel("有效帧范围"), 4, 0)
        self.active_range_label = QLabel("未确认")
        grid.addWidget(self.active_range_label, 4, 1)
        grid.addWidget(QLabel("有效帧数"), 5, 0)
        self.active_frame_count_label = QLabel("-")
        grid.addWidget(self.active_frame_count_label, 5, 1)
        layout.addLayout(grid)

        # 这些标签继续保留为状态字段，避免自动 K0 完成回调访问空属性，
        # 但不再挂到界面上，以满足“删除摘要和说明部分”的要求。
        self.auto_k0_value_label = QLabel("-")
        self.auto_k0_prominence_label = QLabel("-")
        self.auto_k0_window_label = QLabel("-")
        self.auto_k0_pad_label = QLabel("-")
        self.auto_k0_candidates_label = QLabel("-")
        self.readout_label: QLabel | None = None
        layout.addStretch(1)
        return self.value_panel

    def _build_bottom_panel(self) -> QWidget:
        """构建底部状态栏：状态文本、进度条、日志输出。"""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.status_label = QLabel("空闲")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)

        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.log)
        return panel

    def _build_plot_tab(self, canvas, key: str) -> QWidget:
        """为单个画布创建带工具栏和提示文本的标准 tab 容器。"""
        container = QWidget()
        tab_layout = QVBoxLayout(container)
        tab_layout.setContentsMargins(2, 2, 2, 2)
        tab_layout.setSpacing(2)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar = PlotToolbar(canvas, self)
        copy_button = QPushButton("复制图片")
        copy_button.clicked.connect(canvas.copy_current_view_to_clipboard)
        toolbar_row.addWidget(toolbar, 1)
        toolbar_row.addWidget(copy_button, 0)

        info_label = QLabel("点击图像后显示数值。")
        self._plot_info_labels[key] = info_label
        canvas.set_info_callback(lambda text, layer=key: self._update_readout(layer, text))
        if key in {"h", "h_prime"}:
            canvas.set_pixel_callback(lambda payload, layer=key: self._handle_pixel_click(layer, payload))
        elif key in {"phi0", "theta", "phase_gap_raw", "phase_gap_final", "fringe_order", "final_height_phase"}:
            canvas.set_pixel_callback(lambda payload, layer=key: self._handle_profile_click(layer, payload))

        tab_layout.addLayout(toolbar_row)
        tab_layout.addWidget(canvas, 1)
        tab_layout.addWidget(info_label)
        return container

    def _convert_map_for_display(self, key: str, data: np.ndarray) -> np.ndarray:
        """把前台以相位量纲展示的图层统一从 rad 换算为 cycles。"""
        phase_like_keys = {
            "phi0",
            "theta_map",
            "final_height_phase_map",
            "phase_gap_raw",
            "phase_gap_final",
        }
        array = np.asarray(data, dtype=np.float32)
        if key in phase_like_keys:
            return (array / TAU).astype(np.float32)
        return array

    def _build_diagnostic_tab(self) -> QWidget:
        """构建结果摘要文本页。"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("结果摘要")
        title.setStyleSheet("font-weight: 600; font-size: 14px;")
        layout.addWidget(title)

        self._diagnostic_summary = QPlainTextEdit()
        self._diagnostic_summary.setReadOnly(True)
        self._diagnostic_summary.setPlaceholderText("分析完成后显示 h（斜率高度）与 h_prime（最终高度）的对比摘要。")
        layout.addWidget(self._diagnostic_summary, 1)
        return container

    def _update_readout(self, layer: str, text: str) -> None:
        """同步更新右侧读数区和当前图层提示。"""
        if self.readout_label is not None:
            self.readout_label.setText(text)
        self.layer_label.setText(layer)
        self._plot_info_labels[layer].setText(text)

        parts = [part.strip() for part in text.split(",")]
        parsed: dict[str, str] = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key.strip()] = value.strip()
        self.x_label.setText(parsed.get("x", parsed.get("k", "-")))
        self.y_label.setText(parsed.get("y", "-"))
        self.value_label.setText(parsed.get("value", parsed.get("z", parsed.get("h_prime", parsed.get("h", "-")))))

    def _choose_folder(self) -> None:
        """弹出目录选择器并写入图像目录输入框。"""
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if folder:
            self.folder_edit.setText(folder)
            self._refresh_zero_padding_labels(self._estimate_frame_count_for_padding_labels())

    def _choose_scan_log(self) -> None:
        """弹出文件选择器并写入 scan_log 路径。"""
        path, _ = QFileDialog.getOpenFileName(self, "选择 scan_log.txt", "", "Text Files (*.txt);;All Files (*)")
        if path:
            self.scan_log_edit.setText(path)

    def _choose_mat_file(self) -> None:
        """弹出文件选择器并写入 MAT 文件路径。"""
        path, _ = QFileDialog.getOpenFileName(self, "选择 MAT 文件", "", "MAT Files (*.mat);;All Files (*)")
        if path:
            self.mat_edit.setText(path)
            self._refresh_zero_padding_labels(None)

    def _update_data_source_ui(self) -> None:
        """根据数据来源切换图像目录与 MAT 文件控件。"""
        is_mat_file = str(self.data_source.currentData()) == "mat_file"
        self.folder_edit.setEnabled(not is_mat_file)
        self.folder_button.setEnabled(not is_mat_file)
        self.mat_edit.setEnabled(is_mat_file)
        self.mat_button.setEnabled(is_mat_file)

    def _refresh_scan_log_summary(self) -> None:
        """读取 scan_log 并在面板中展示点数与首尾样本摘要。"""
        path_text = self.scan_log_edit.text().strip()
        if not path_text:
            self.scan_log_summary_label.setText("未加载 scan_log")
            return
        try:
            positions = load_actual_positions_um(Path(path_text))
        except Exception as exc:
            self.scan_log_summary_label.setText(f"读取失败: {exc}")
            return

        head = ", ".join(f"{value:.3f}" for value in positions[:3])
        tail = ", ".join(f"{value:.3f}" for value in positions[-3:])
        self.scan_log_summary_label.setText(
            f"点数: {positions.shape[0]}\n前3个: {head}\n后3个: {tail}"
        )

    def _update_sampling_mode_ui(self) -> None:
        """根据采样模式切换步长/scan_log 相关控件可用状态。"""
        is_nonuniform = str(self.sampling_mode.currentData()) == "nonuniform"
        self.step_size.setEnabled(not is_nonuniform)
        self.scan_log_edit.setEnabled(is_nonuniform)
        self.scan_log_button.setEnabled(is_nonuniform)

    def _is_diagnostic_mode(self) -> bool:
        """当前选择 PhaseGap 工作流时显示后处理诊断页。"""
        return str(self.phase_gap_method.currentData()).strip().lower() != "fda"

    def _refresh_result_tabs(self) -> None:
        """根据当前工作流刷新结果页签。"""
        current_widget = self.tabs.currentWidget() if hasattr(self, "tabs") else None
        self.tabs.clear()
        self.tabs.addTab(self._plot_tabs["h"], "Height")
        self.tabs.addTab(self._plot_tabs["h_prime"], "height_prime")
        self.tabs.addTab(self._plot_tabs["phi0"], "phi0")
        self.tabs.addTab(self._plot_tabs["surface"], "三维图")
        self.tabs.addTab(self._plot_tabs["h_prime_surface"], "h_prime三维图")
        self.tabs.addTab(self._plot_tabs["comparison"], "对比图")
        self.tabs.addTab(self._plot_tabs["k0"], "全局 K0 频谱")
        if self._is_diagnostic_mode():
            self.tabs.addTab(self._plot_tabs["theta"], "Coherece")
            self.tabs.addTab(self._plot_tabs["final_height_phase"], "Final Height Map (phase)")
            self.tabs.addTab(self._plot_tabs["merit"], "merit 质量图")
            self.tabs.addTab(self._plot_tabs["phase_gap_raw"], "Disconnected Phase Gap")
            self.tabs.addTab(self._plot_tabs["phase_gap_final"], "Final Phase Gap")
            self.tabs.addTab(self._plot_tabs["fringe_order"], "Fringe Order Map")
            self.tabs.addTab(self._plot_tabs["confidence"], "置信度图")
        if self._is_diagnostic_mode() and self._diagnostic_tab is not None:
            self.tabs.addTab(self._diagnostic_tab, "诊断摘要")
        if current_widget is not None:
            index = self.tabs.indexOf(current_widget)
            if index >= 0:
                self.tabs.setCurrentIndex(index)

    def _update_analysis_mode_ui(self) -> None:
        """刷新当前工作流对应的结果页签。"""
        self._refresh_result_tabs()
        if hasattr(self, "analysis_mode_label"):
            self.analysis_mode_label.setText("FDA" if not self._is_diagnostic_mode() else "PhaseGap")
        if self._result is not None:
            self._update_diagnostic_summary(self._result)

    def _update_diagnostic_summary(self, result: AnalysisResult) -> None:
        """
        刷新结果摘要文字。

        这个摘要默认不挂在主界面上，但保留函数可以让后续需要时直接复用。
        """
        if self._diagnostic_summary is None:
            return
        delta = np.asarray(result.h, dtype=np.float32) - np.asarray(result.h_prime, dtype=np.float32)
        finite_delta = delta[np.isfinite(delta)]
        if finite_delta.size > 0:
            delta_mean = float(np.mean(finite_delta))
            delta_abs_mean = float(np.mean(np.abs(finite_delta)))
            delta_min = float(np.min(finite_delta))
            delta_max = float(np.max(finite_delta))
        else:
            delta_mean = float("nan")
            delta_abs_mean = float("nan")
            delta_min = float("nan")
            delta_max = float("nan")

        available_maps = [
            key
            for key in (
                "theta_map",
                "theta_map_smoothed",
                "merit_map",
                "phase_gap_raw",
                "phase_gap_baseline_raw",
                "phase_gap_connected",
                "phase_gap_filled",
                "phase_gap_fit",
                "phase_gap_final",
                "phase_gap_residue",
                "phase_gap_branch_cut",
                "phase_gap_gradient_residual",
                "baseline_fringe_order_map",
                "fringe_order_map",
                "confidence_map",
                "g0_map",
                "peak_amplitude_map",
                "active_range",
                "active_ranges",
            )
            if key in result.extras
        ]
        phase_gap_method = str(result.extras.get("phase_gap_method", "unknown"))
        result_label = "FDA级次修正" if phase_gap_method.upper() == "FDA" else "PhaseGap最终高度"
        lines = [
            f"结果模型: h（斜率高度） / h_prime（{result_label}） / phi0（k0处相位）",
            f"工作流: {phase_gap_method}",
            "中间图层: " + (", ".join(available_maps) if available_maps else "无"),
            f"h 形状: {result.h.shape}",
            f"h_prime 形状: {result.h_prime.shape}",
            f"phi0 形状: {result.phi0.shape}",
            f"k0_index: {result.k0_index}",
            f"k0_value: {result.k0_value:.6f}",
            f"h - h_prime 均值: {delta_mean:.6f}",
            f"|h - h_prime| 均值: {delta_abs_mean:.6f}",
            f"h - h_prime 范围: [{delta_min:.6f}, {delta_max:.6f}]",
        ]
        self._diagnostic_summary.setPlainText("\n".join(lines))

    def _draw_optional_diagnostic_map(self, result: AnalysisResult, key: str, canvas: HeatmapCanvas, title: str) -> None:
        """
        绘制可选 FDA 中间图层。

        如果当前结果里没有这个键，就画一张 NaN 图，
        这样 GUI 结构稳定，不会因为个别图层缺失直接报错。
        """
        data = result.extras.get(key)
        if data is None:
            data = np.full_like(result.h, np.nan, dtype=np.float32)
        canvas.draw_map(self._convert_map_for_display(key, np.asarray(data, dtype=np.float32)), title)

    def _current_source_key(self) -> tuple[str, str, str, str, bool, int]:
        """生成当前数据源签名，用于判断自动 K0 的有效范围缓存是否仍可复用。"""
        data_source = str(self.data_source.currentData())
        source_text = self.mat_edit.text().strip() if data_source == "mat_file" else self.folder_edit.text().strip()
        sampling_mode = str(self.sampling_mode.currentData())
        scan_log_text = self.scan_log_edit.text().strip() if sampling_mode == "nonuniform" else ""
        # Windows 路径大小写不敏感，这里统一 casefold，避免同一路径不同大小写导致缓存误判失效。
        return (
            data_source,
            str(Path(source_text)).casefold(),
            sampling_mode,
            str(Path(scan_log_text)).casefold() if scan_log_text else "",
            bool(self.expand_active_range_checkbox.isChecked()),
            int(self.active_range_expansion_frames.value()),
        )

    def _extract_active_range_pair(self, payload: object) -> tuple[int, int] | None:
        """从 worker 结果或缓存字典里提取 1-based 有效帧范围。"""
        if payload is None:
            return None
        array = np.asarray(payload).reshape(-1)
        if array.size < 2:
            return None
        start_frame = int(array[0])
        end_frame = int(array[1])
        if start_frame <= 0 or end_frame < start_frame:
            return None
        return start_frame, end_frame

    def _cached_active_range_for_current_source(self) -> tuple[int, int] | None:
        """若自动 K0 缓存仍匹配当前数据源，则返回已确认的有效帧范围。"""
        if not isinstance(self._auto_k0_result, dict):
            return None
        if self._auto_k0_result.get("_source_key") != self._current_source_key():
            return None
        return self._extract_active_range_pair(self._auto_k0_result.get("active_range"))

    def _set_active_range_display(self, active_range: tuple[int, int] | None) -> None:
        """刷新右侧有效帧范围显示。"""
        if active_range is None:
            self.active_range_label.setText("未确认")
            self.active_frame_count_label.setText("-")
            self._refresh_zero_padding_labels(self._estimate_frame_count_for_padding_labels())
            return
        start_frame, end_frame = active_range
        self.active_range_label.setText(f"{start_frame}-{end_frame}")
        frame_count = end_frame - start_frame + 1
        self.active_frame_count_label.setText(str(frame_count))
        self._refresh_zero_padding_labels(frame_count)

    def _estimate_frame_count_for_padding_labels(self) -> int | None:
        """尽量用当前图像文件数生成补零选项标签；MAT 数据等未知情况返回 None。"""
        if hasattr(self, "active_frame_count_label"):
            active_count_text = self.active_frame_count_label.text().strip()
            if active_count_text.isdigit() and int(active_count_text) > 1:
                return int(active_count_text)
        if str(self.data_source.currentData()) != "image_folder":
            return None
        folder_text = self.folder_edit.text().strip()
        if not folder_text:
            return None
        try:
            return len(collect_image_files(Path(folder_text)))
        except Exception:
            return None

    def _refresh_zero_padding_labels(self, frame_count: int | None) -> None:
        """根据当前已知帧数刷新补零下拉显示，不改变用户已选模式。"""
        if hasattr(self, "analysis_zero_padding_mode"):
            _populate_zero_padding_combo(self.analysis_zero_padding_mode, frame_count=frame_count)

    def _build_params(self) -> AnalysisParams:
        """从界面控件读取并校验一次分析所需参数。"""
        data_source = str(self.data_source.currentData())
        folder_text = self.folder_edit.text().strip()
        mat_text = self.mat_edit.text().strip()
        if data_source == "mat_file":
            if not mat_text:
                raise ValueError("请先选择 MAT 文件。")
            mat_path = Path(mat_text)
            folder = mat_path.parent
        else:
            if not folder_text:
                raise ValueError("请先选择图像文件夹。")
            mat_path = None
            folder = Path(folder_text)
        fixed_k0 = float(self.fixed_k0.value())
        if fixed_k0 <= 0.0:
            raise ValueError("请先设置固定 K0，或先执行自动定 K0。")
        sampling_mode = str(self.sampling_mode.currentData())
        scan_log_path = None
        if sampling_mode == "nonuniform":
            scan_log_text = self.scan_log_edit.text().strip()
            if not scan_log_text:
                raise ValueError("非均匀采样模式下必须选择 scan_log.txt。")
            scan_log_path = Path(scan_log_text)
        return AnalysisParams(
            folder=folder,
            start_height=float(self.start_height.value()),
            step_size=float(self.step_size.value()),
            fixed_k0=fixed_k0,
            window_size=int(self.window_size.value()),
            fitting_method=str(self.fitting.currentData()),
            unwrap_method=str(self.unwrap.currentData()),
            window_name=str(self.analysis_window_name.currentData()),
            window_alpha=0.5,
            zero_padding_mode=str(self.analysis_zero_padding_mode.currentData()),
            data_source=data_source,
            mat_path=mat_path,
            phase_gap_method=str(self.phase_gap_method.currentData()),
            sampling_mode=sampling_mode,
            scan_log_path=scan_log_path,
            sample_positions_um=None,
            active_range=self._cached_active_range_for_current_source(),
            expand_active_range=bool(self.expand_active_range_checkbox.isChecked()),
            active_range_expansion_frames=int(self.active_range_expansion_frames.value()),
        )

    def _set_running_state(self, running: bool) -> None:
        """统一控制任务按钮的启停状态。"""
        self.auto_k0_button.setEnabled(not running)
        self.start_button.setEnabled(not running)

    def _finalize_thread(self) -> None:
        """在线程结束后清理 worker/thread 引用并恢复按钮状态。"""
        self._thread = None
        self._worker = None
        self._k0_worker = None
        self._set_running_state(False)

    def _start_analysis(self) -> None:
        """创建分析线程与 AnalysisWorker，并连接全部信号槽。"""
        if self._thread is not None and self._thread.isRunning():
            return
        try:
            params = self._build_params()
        except Exception as exc:
            QMessageBox.critical(self, "参数错误", str(exc))
            return

        self._result = None
        self._analysis_cube = None
        self._analysis_params = params
        self.export_text_button.setEnabled(False)
        self.export_figures_button.setEnabled(False)
        self.progress.setValue(0)
        self.log.clear()
        self._set_active_range_display(params.active_range)
        self.status_label.setText("正在分析")
        self._set_running_state(True)

        self._thread = QThread(self)
        self._worker = AnalysisWorker(params)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self.progress.setValue)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._handle_finished)
        self._worker.failed.connect(self._handle_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._finalize_thread)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _start_auto_k0(self) -> None:
        """创建自动 K0 线程并启动估计任务。"""
        if self._thread is not None and self._thread.isRunning():
            return
        data_source = str(self.data_source.currentData())
        folder = self.folder_edit.text().strip()
        mat_text = self.mat_edit.text().strip()
        if data_source == "mat_file":
            if not mat_text:
                QMessageBox.critical(self, "参数错误", "请先选择 MAT 文件。")
                return
            folder_path = Path(mat_text).parent
            mat_path = Path(mat_text)
        else:
            if not folder:
                QMessageBox.critical(self, "参数错误", "请先选择图像文件夹。")
                return
            folder_path = Path(folder)
            mat_path = None
        sampling_mode = str(self.sampling_mode.currentData())
        scan_log_path = None
        if sampling_mode == "nonuniform":
            scan_log_text = self.scan_log_edit.text().strip()
            if not scan_log_text:
                QMessageBox.critical(self, "参数错误", "非均匀采样模式下必须选择 scan_log.txt。")
                return
            scan_log_path = Path(scan_log_text)

        dialog = AutoK0Dialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        settings = dialog.values()

        self.progress.setValue(0)
        self.log.clear()
        self._set_active_range_display(None)
        self.status_label.setText("正在自动估计 K0")
        self._set_running_state(True)

        self._thread = QThread(self)
        self._k0_worker = GlobalK0Worker(
            folder=folder_path,
            step_size=float(self.step_size.value()),
            candidate_ratio=float(settings["candidate_ratio"]),
            # 自动 K0 与正式 FDA 分析共用主界面的窗函数配置，避免两处参数不一致。
            window_name=str(self.analysis_window_name.currentData()),
            window_alpha=0.5,
            zero_padding_mode=str(self.analysis_zero_padding_mode.currentData()),
            expand_active_range=bool(self.expand_active_range_checkbox.isChecked()),
            active_range_expansion_frames=int(self.active_range_expansion_frames.value()),
            data_source=data_source,
            mat_path=mat_path,
            sampling_mode=sampling_mode,
            scan_log_path=scan_log_path,
        )
        self._k0_worker.moveToThread(self._thread)
        self._thread.started.connect(self._k0_worker.run)
        self._k0_worker.progress.connect(self.progress.setValue)
        self._k0_worker.log.connect(self._append_log)
        self._k0_worker.finished.connect(self._handle_auto_k0_finished)
        self._k0_worker.failed.connect(self._handle_auto_k0_failed)
        self._k0_worker.finished.connect(self._thread.quit)
        self._k0_worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._finalize_thread)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _append_log(self, message: str) -> None:
        """向底部日志框追加一行文本。"""
        self.log.appendPlainText(message)

    def _present_aux_window(self, window: QWidget) -> None:
        """仅在需要时显式呈现辅助窗口，避免重复点击时反复抢焦点。"""
        # 已经打开且处于正常显示状态时，只刷新内容，不再反复 raise/activate；
        # 这样可以明显减少主窗口与弹窗之间的焦点来回切换，降低“闪一下”的体感。
        if window.isVisible() and not window.isMinimized():
            return

        if window.isMinimized():
            window.showNormal()
        else:
            window.show()
        window.raise_()
        window.activateWindow()

    def _handle_finished(self, result: object) -> None:
        """
        分析完成后的 GUI 刷新入口。

        FDA_Antivib 只刷新 FDA 主结果图层，非 FDA 后处理图层不会进入主界面。
        """
        analysis_result = AnalysisResult.coerce(result)
        self._result = analysis_result
        self._analysis_cube = self._worker.last_cube if self._worker is not None else None
        self.status_label.setText("分析完成")
        self.export_text_button.setEnabled(True)
        self.export_figures_button.setEnabled(True)
        self.h_prime_canvas.draw_map(analysis_result.h_prime, "height_prime")
        self.h_canvas.draw_map(analysis_result.h, "Height")
        self.phi0_canvas.draw_map(self._convert_map_for_display("phi0", analysis_result.phi0), "Phase Profile (cycles)")
        if self._is_diagnostic_mode():
            self._draw_optional_diagnostic_map(analysis_result, "theta_map", self.theta_canvas, "Coherece(in units of phase, cycles)")
            self._draw_optional_diagnostic_map(
                analysis_result,
                "final_height_phase_map",
                self.final_height_phase_canvas,
                "Final Height Map (in units of phase, cycles)",
            )
            self._draw_optional_diagnostic_map(analysis_result, "merit_map", self.merit_canvas, "merit_map（局部质量）")
            self._draw_optional_diagnostic_map(
                analysis_result,
                "phase_gap_raw",
                self.phase_gap_raw_canvas,
                "Disconnected Phase Gap (cycles)",
            )
            self._draw_optional_diagnostic_map(
                analysis_result,
                "phase_gap_final",
                self.phase_gap_final_canvas,
                "Final Phase Gap (cycles)",
            )
            self._draw_optional_diagnostic_map(
                analysis_result,
                "fringe_order_map",
                self.fringe_order_canvas,
                "Fringe Order Map",
            )
            self._draw_optional_diagnostic_map(
                analysis_result,
                "confidence_map",
                self.confidence_canvas,
                "confidence_map（置信度）",
            )
        self.surface_canvas.draw_surface(analysis_result.h, "h（三维，斜率高度）")
        self.h_prime_surface_canvas.draw_surface(analysis_result.h_prime, "h_prime（三维，最终高度）")
        comparison_title = "h（斜率高度） vs h_prime（FDA级次修正）"
        if self._is_diagnostic_mode():
            comparison_title = "h（斜率高度） vs h_prime（PhaseGap最终）"
        self.comparison_canvas.draw_comparison(
            analysis_result.h,
            analysis_result.h_prime,
            comparison_title,
        )
        self._update_diagnostic_summary(analysis_result)
        self._set_active_range_display(self._extract_active_range_pair(analysis_result.extras.get("active_range")))
        self._append_log("结果预览已更新。")

    def _handle_auto_k0_finished(self, result: dict) -> None:
        """处理自动 K0 成功结果并刷新相关 UI。"""
        result["_source_key"] = self._current_source_key()
        self._auto_k0_result = result
        active_range = self._extract_active_range_pair(result.get("active_range"))
        self._set_active_range_display(active_range)
        self.fixed_k0.setValue(float(result["k0_value"]))
        self.auto_k0_value_label.setText(f"{float(result['k0_value']):.6f}")
        self.auto_k0_prominence_label.setText(f"{float(result['peak_prominence']):.6f}")
        self.auto_k0_window_label.setText(str(result["window_name"]))
        self.auto_k0_pad_label.setText(str(result.get("zero_padding_mode", "-")))
        self.auto_k0_candidates_label.setText(str(int(result["candidate_count"])))
        self.k0_canvas.draw_spectrum(result["k_axis"], result["spectrum"], int(result["peak_index"]), "全局 K0 频谱")
        self.tabs.setCurrentWidget(self.k0_tab)
        self.status_label.setText("全局 K0 已就绪")
        self._append_log(f"自动 K0 = {float(result['k0_value']):.6f}")

    def _handle_pixel_click(self, layer: str, payload: dict[str, float]) -> None:
        """响应热图点击，打开并刷新单像素分析窗口。"""
        if self._analysis_cube is None or self._analysis_params is None:
            return
        x = int(round(payload.get("x", 0.0)))
        y = int(round(payload.get("y", 0.0)))
        x = max(0, min(x, self._analysis_cube.shape[1] - 1))
        y = max(0, min(y, self._analysis_cube.shape[0] - 1))

        global_k0_value = None
        if self._result is not None:
            if isinstance(self._result, AnalysisResult):
                global_k0_value = float(self._result.k0_value)
            elif isinstance(self._result, dict) and "k0_value" in self._result:
                global_k0_value = float(self._result["k0_value"])

        if self._pixel_window is None:
            self._pixel_window = PixelAnalysisWindow()
        self._present_aux_window(self._pixel_window)

        analysis = build_pixel_analysis(
            intensity_data=self._analysis_cube,
            x=x,
            y=y,
            step_size=self._analysis_params.step_size,
            start_height=self._analysis_params.start_height,
            unwrap_method=self._analysis_params.unwrap_method,
            window_size=self._analysis_params.window_size,
            fitting_method=self._analysis_params.fitting_method,
            global_k0_value=global_k0_value,
            sample_positions_um=self._analysis_params.sample_positions_um,
            window_name=self._analysis_params.window_name,
            window_alpha=self._analysis_params.window_alpha,
            zero_padding_mode=self._analysis_params.zero_padding_mode,
        )
        self._pixel_window.update_analysis(
            payload=analysis,
            layer_name=layer,
            fitting_method=self._analysis_params.fitting_method,
            unwrap_method=self._analysis_params.unwrap_method,
        )

    def _handle_profile_click(self, layer: str, payload: dict[str, float]) -> None:
        """响应 PhaseGap 诊断图层点击，弹出剖面窗口联查相位与级次。"""
        if self._result is None:
            return

        layer_sources: dict[str, np.ndarray | None] = {
            "fringe_order": self._result.extras.get("fringe_order_map"),
            "phi0": self._result.phi0,
            "theta": self._result.extras.get("theta_map"),
            "final_height_phase": self._result.extras.get("final_height_phase_map"),
            "phase_gap_raw": self._result.extras.get("phase_gap_raw"),
            "phase_gap_final": self._result.extras.get("phase_gap_final"),
        }
        source_map = layer_sources.get(layer)
        if source_map is None:
            return

        layer_key_map = {
            "fringe_order": "fringe_order_map",
            "phi0": "phi0_map",
            "theta": "theta_map",
            "final_height_phase": "final_height_phase_map",
            "phase_gap_raw": "phase_gap_raw",
            "phase_gap_final": "phase_gap_final",
        }

        data = np.asarray(source_map, dtype=np.float32)
        x = int(round(payload.get("x", 0.0)))
        y = int(round(payload.get("y", 0.0)))
        x = max(0, min(x, data.shape[1] - 1))
        y = max(0, min(y, data.shape[0] - 1))

        diagnostic_maps = {
            key: np.asarray(self._result.extras[key], dtype=np.float32)
            for key in (
                "phase_gap_raw",
                "phase_gap_final",
                "merit_map",
                "confidence_map",
                "theta_map",
                "theta_map_smoothed",
                "final_height_phase_map",
                "peak_amplitude_map",
            )
            if key in self._result.extras
        }
        # `phi0` 是结果对象主字段，不在 extras 里；这里显式补进去，便于和 phase gap 图层联查。
        diagnostic_maps["phi0_map"] = np.asarray(self._result.phi0, dtype=np.float32)

        if self._fringe_profile_window is None:
            self._fringe_profile_window = FringeProfileWindow()
        self._fringe_profile_window.update_profiles(
            data,
            x=x,
            y=y,
            layer_key=layer_key_map.get(layer, "fringe_order_map"),
            diagnostic_maps=diagnostic_maps,
        )
        self._present_aux_window(self._fringe_profile_window)

    def _handle_failed(self, message: str) -> None:
        """处理分析失败：更新状态、弹窗并记录日志。"""
        self.status_label.setText("分析失败")
        QMessageBox.critical(self, "分析失败", message)
        self._append_log(f"错误：{message}")

    def _handle_auto_k0_failed(self, message: str) -> None:
        """处理自动 K0 失败：更新状态、弹窗并记录日志。"""
        self.status_label.setText("自动 K0 失败")
        QMessageBox.critical(self, "自动 K0 失败", message)
        self._append_log(f"错误：{message}")

    def _export_text(self) -> None:
        """弹出数据导出对话框并执行文本导出。"""
        # 导出依赖的是“分析结果 + 参数快照”，不应该依赖已结束线程里的 worker 实例。
        # 否则线程一收尾把 `_worker` 清空后，导出按钮虽然还亮着，点击却会直接失效。
        if self._analysis_params is None or self._result is None:
            return
        options = [
            ("h", "h 数据"),
            ("h_prime", "h_prime 数据"),
            ("phi0", "phi0 数据"),
        ]
        dialog = ExportSelectionDialog(
            "选择导出数据",
            options,
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        selected_keys = dialog.selected_keys()
        if not selected_keys:
            return
        default_filenames = {
            "h": "h.txt",
            "h_prime": "h_prime.txt",
            "phi0": "phi0.txt",
        }
        display_names = dict(options)
        target_paths: dict[str, Path] = {}
        # 先逐个收集文件保存路径；只有所有目标都确认后才真正落盘，避免半途取消留下半套结果。
        for selected_key in selected_keys:
            default_name = default_filenames.get(selected_key, f"{selected_key}.txt")
            suggested_path = str(self._analysis_params.folder / default_name)
            save_path, _ = QFileDialog.getSaveFileName(
                self,
                f"保存 {display_names.get(selected_key, selected_key)}",
                suggested_path,
                "Text Files (*.txt);;All Files (*)",
            )
            if not save_path:
                return
            target_paths[selected_key] = Path(save_path)
        # 每次导出时基于保存下来的参数快照重建 session，避免依赖线程生命周期。
        session = AnalysisSession(self._analysis_params)
        paths = session.export_text_results(
            self._result,
            self._analysis_params.folder,
            selected_keys=selected_keys,
            target_paths=target_paths,
        )
        self._append_log(f"数据已导出：{', '.join(str(path) for path in paths.values())}")

    def _export_figures(self) -> None:
        """弹出图片导出对话框并执行图片导出。"""
        # 图片导出同样只依赖当前结果本身，不需要仍然持有后台 worker。
        if self._result is None:
            return
        figure_options = [
            ("h_2d", "h 二维图"),
            ("h_prime_2d", "h_prime 二维图"),
            ("h_prime_3d", "h_prime 三维图"),
            ("phi0_2d", "phi0 二维图"),
        ]
        dialog = ExportSelectionDialog(
            "选择导出图片",
            figure_options,
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        selected_keys = dialog.selected_keys()
        if not selected_keys:
            return
        folder = QFileDialog.getExistingDirectory(self, "选择图片输出文件夹")
        if not folder:
            return
        # 直接调用统一图片导出函数，让导出能力与线程 worker 解耦。
        paths = save_paper_figures(self._result, Path(folder), selected_keys=selected_keys)
        self._append_log(f"图片已导出：{', '.join(str(path) for path in paths.values())}")
