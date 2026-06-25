from __future__ import annotations
"""主窗口模块。

该文件负责把参数面板、结果图层、后台线程 worker 和导出功能整合到一个
统一的桌面界面中，是 GUI 端的主要控制器。
"""

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QEvent, QProcess, Qt, QThread, QUrl
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSizePolicy,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app import APP_NAME, __version__
from app.core.pixel_analysis import build_pixel_analysis
from app.core.result_model import AnalysisResult
from app.gui.fringe_profile_window import FringeProfileWindow
from app.gui.pixel_window import PixelAnalysisWindow
from app.gui.surface_analysis_page import PlaneAnalysisPage, StepAnalysisPage
from app.gui.theme import ThemeMode, build_stylesheet, resolve_theme_mode
from app.gui.widgets import SectionHeader, StatusPill, StepButton
from app.pipeline.io import collect_image_files
from app.gui.worker import AnalysisWorker, GlobalK0Worker, UpdateCheckWorker, UpdateDownloadWorker
from app.pipeline.scan_log import load_actual_positions_um
from app.pipeline.session import AnalysisParams, AnalysisSession
from app.plotting.paper import save_paper_figures
from app.plotting.preview import ComparisonCanvas, HeatmapCanvas, SpectrumCanvas, SurfaceCanvas
from app.update_checker import UpdateInfo

TAU = 2.0 * np.pi
GFDA_SOFTWARE_METHODS = {"gfda_carrier_phase", "gfda_scatter_fit"}
DEFAULT_FIXED_K0_VALUE = 11.15


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
    """填充统一补零选项；固定长度选项不随有效帧数改名，避免用户选择跳变。"""
    current_data = combo.currentData()
    combo.blockSignals(True)
    combo.clear()
    combo.addItem("1x", "none")
    combo.addItem("256", "fixed_256")
    combo.addItem("512", "fixed_512")
    combo.addItem("1024", "fixed_1024")
    combo.addItem("2x", "factor_2")
    combo.addItem("4x", "factor_4")
    combo.addItem("8x", "factor_8")
    if current_data is not None:
        index = combo.findData(current_data)
        if index >= 0:
            combo.setCurrentIndex(index)
    elif combo.findData("fixed_512") >= 0:
        combo.setCurrentIndex(combo.findData("fixed_512"))
    combo.blockSignals(False)


class MainWindow(QMainWindow):
    """应用主窗口：参数输入、分析调度、可视化与导出统一入口。"""

    def __init__(self) -> None:
        """初始化状态字段并构建整套界面。"""
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self._thread: QThread | None = None
        self._update_thread: QThread | None = None
        self._update_worker: object | None = None
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
        self._preview_step_labels: list[QLabel] = []
        self._diagnostic_tab: QWidget | None = None
        self._diagnostic_summary: QPlainTextEdit | None = None
        self._theme_mode: ThemeMode = "light"
        self._result_host_layout: QVBoxLayout | None = None
        self._algorithm_config_touched = False
        self.setMinimumSize(1180, 760)
        self._build_ui()
        self._install_wheel_guards()
        self._apply_theme()
        style_hints = QGuiApplication.styleHints()
        if hasattr(style_hints, "colorSchemeChanged"):
            style_hints.colorSchemeChanged.connect(self._handle_system_color_scheme_changed)

    def _build_ui(self) -> None:
        """构建工业分析控制台根布局。"""
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        self.main_pages = QStackedWidget()
        self.main_pages.addWidget(self._build_workbench_page())
        self.plane_analysis_page = PlaneAnalysisPage()
        self.step_analysis_page = StepAnalysisPage()
        self.main_pages.addWidget(self.plane_analysis_page)
        self.main_pages.addWidget(self.step_analysis_page)
        self.main_pages.addWidget(self._build_settings_page())
        root_layout.addWidget(self._build_top_command_bar(), 0)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(6)
        body_layout.addWidget(self.main_pages, 1)
        root_layout.addWidget(body, 1)
        root_layout.addWidget(self._build_bottom_panel(), 0)

        self._select_main_page(0)
        self._select_step(0)

    def _build_top_command_bar(self) -> QWidget:
        """构建扁平顶部栏：只保留品牌和一级页面切换。"""
        bar = QFrame()
        bar.setObjectName("TopCommandBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        brand = QLabel(APP_NAME)
        brand.setObjectName("BrandText")
        layout.addWidget(brand, 0)

        self.global_nav_group = QButtonGroup(self)
        self.global_nav_group.setExclusive(True)
        self.global_nav_buttons: list[QPushButton] = []
        for index, title in enumerate(("工作台", "平面分析", "台阶分析", "设置")):
            button = QPushButton(title)
            button.setObjectName("TopNavButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, page=index: self._select_main_page(page))
            self.global_nav_group.addButton(button, index)
            self.global_nav_buttons.append(button)
            layout.addWidget(button, 0)

        layout.addStretch(1)
        self.update_button = QPushButton("检查更新")
        self.update_button.setObjectName("TopNavButton")
        self.update_button.setToolTip(f"当前版本 {__version__}，检查 GitHub Releases 中的新版本。")
        self.update_button.clicked.connect(self._check_for_updates)
        layout.addWidget(self.update_button, 0)
        return bar

    def _build_workbench_page(self) -> QWidget:
        """构建步骤导航、结果画布和当前参数三栏工作台。"""
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.step_rail = self._build_step_rail()
        parameter_panel = self._build_parameter_panel()
        self.result_workspace = self._build_center_panel()

        self.workbench_result_host = QWidget()
        self.workbench_result_host.setMinimumSize(0, 0)
        self.workbench_result_host.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.workbench_result_layout = QVBoxLayout(self.workbench_result_host)
        self.workbench_result_layout.setContentsMargins(0, 0, 0, 0)
        self.workbench_result_layout.setSpacing(0)
        self.workbench_result_layout.addWidget(self.result_workspace)
        self._result_host_layout = self.workbench_result_layout

        layout.addWidget(self.step_rail, 0)
        layout.addWidget(self.workbench_result_host, 1)
        layout.addWidget(parameter_panel, 0)
        return page

    def _build_log_page(self) -> QWidget:
        """构建完整运行日志页面。"""
        page = QFrame()
        page.setObjectName("ResultWorkspace")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        title = QLabel("运行日志")
        title.setObjectName("PageTitle")
        hint = QLabel("记录自动 K0、分析、结果刷新和导出过程。")
        hint.setObjectName("MutedLabel")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(self.log, 1)
        return page

    def _build_settings_page(self) -> QWidget:
        """构建主题与全局显示设置页面。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title = QLabel("界面设置")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        card = QFrame()
        card.setObjectName("SettingsCard")
        card_layout = QFormLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(10)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("跟随系统", "system")
        self.theme_combo.addItem("浅色", "light")
        self.theme_combo.addItem("深色", "dark")
        self.theme_combo.setCurrentIndex(self.theme_combo.findData("light"))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        card_layout.addRow("主题模式", self.theme_combo)
        card_layout.addRow("界面密度", QLabel("均衡"))
        card_layout.addRow("目标尺寸", QLabel("1400 × 900（最低 1180 × 760）"))
        layout.addWidget(card, 0)
        layout.addStretch(1)
        return page

    def _build_step_rail(self) -> QWidget:
        """构建三步分析流程导航。"""
        rail = QFrame()
        rail.setObjectName("StepRail")
        rail.setMinimumWidth(220)
        rail.setMaximumWidth(240)
        layout = QVBoxLayout(rail)
        layout.setContentsMargins(10, 12, 10, 12)
        layout.setSpacing(8)

        title = QLabel("分析流程")
        title.setObjectName("SectionTitle")
        layout.addWidget(title)
        self.step_button_group = QButtonGroup(self)
        self.step_button_group.setExclusive(True)
        self.step_buttons: list[StepButton] = []
        for index, (number, label) in enumerate(
            (
                ("01", "数据与扫描"),
                ("02", "算法与运行"),
            )
        ):
            button = StepButton(number, label)
            button.clicked.connect(lambda checked=False, step=index: self._select_step(step))
            self.step_button_group.addButton(button, index)
            self.step_buttons.append(button)
            layout.addWidget(button)

        layout.addSpacing(8)
        range_title = QLabel("有效范围")
        range_title.setObjectName("SectionTitle")
        layout.addWidget(range_title)
        range_grid = QGridLayout()
        range_grid.setContentsMargins(0, 0, 0, 0)
        range_grid.setHorizontalSpacing(8)
        range_grid.setVerticalSpacing(4)
        range_grid.addWidget(QLabel("帧范围"), 0, 0)
        self.active_range_label = QLabel("未确认")
        range_grid.addWidget(self.active_range_label, 0, 1)
        range_grid.addWidget(QLabel("帧数"), 1, 0)
        self.active_frame_count_label = QLabel("-")
        range_grid.addWidget(self.active_frame_count_label, 1, 1)
        layout.addLayout(range_grid)

        k0_title = QLabel("自动 K0 摘要")
        k0_title.setObjectName("SectionTitle")
        layout.addWidget(k0_title)
        k0_grid = QGridLayout()
        k0_grid.setContentsMargins(0, 0, 0, 0)
        k0_grid.setHorizontalSpacing(8)
        k0_grid.setVerticalSpacing(4)
        self.auto_k0_value_label = QLabel("-")
        self.auto_k0_prominence_label = QLabel("-")
        self.auto_k0_window_label = QLabel("-")
        self.auto_k0_pad_label = QLabel("-")
        self.auto_k0_candidates_label = QLabel("-")
        for row, (label, value) in enumerate(
            (
                ("K0", self.auto_k0_value_label),
                ("峰值显著性", self.auto_k0_prominence_label),
                ("窗函数", self.auto_k0_window_label),
                ("补零", self.auto_k0_pad_label),
                ("候选数", self.auto_k0_candidates_label),
            )
        ):
            k0_grid.addWidget(QLabel(label), row, 0)
            k0_grid.addWidget(value, row, 1)
        layout.addLayout(k0_grid)

        readout_title = QLabel("当前读数")
        readout_title.setObjectName("SectionTitle")
        layout.addWidget(readout_title)
        readout_grid = QGridLayout()
        readout_grid.setContentsMargins(0, 0, 0, 0)
        readout_grid.setHorizontalSpacing(8)
        readout_grid.setVerticalSpacing(4)
        self.layer_label = QLabel("-")
        self.x_label = QLabel("-")
        self.y_label = QLabel("-")
        self.value_label = QLabel("-")
        for row, (label, value) in enumerate(
            (
                ("图层", self.layer_label),
                ("X", self.x_label),
                ("Y", self.y_label),
                ("值", self.value_label),
            )
        ):
            readout_grid.addWidget(QLabel(label), row, 0)
            readout_grid.addWidget(value, row, 1)
        layout.addLayout(readout_grid)
        self.readout_label: QLabel | None = None

        self.export_text_button = QPushButton("导出数据")
        self.export_text_button.setObjectName("ExportButton")
        self.export_text_button.clicked.connect(self._export_text)
        self.export_text_button.setEnabled(False)
        self.export_figures_button = QPushButton("导出图片")
        self.export_figures_button.setObjectName("ExportButton")
        self.export_figures_button.clicked.connect(self._export_figures)
        self.export_figures_button.setEnabled(False)
        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(6)
        export_row.addWidget(self.export_text_button)
        export_row.addWidget(self.export_figures_button)
        layout.addLayout(export_row)
        layout.addStretch(1)

        log_title = QLabel("运行日志")
        log_title.setObjectName("SectionTitle")
        layout.addWidget(log_title)
        self.rail_log = QPlainTextEdit()
        self.rail_log.setObjectName("RailLog")
        self.rail_log.setFrameShape(QFrame.NoFrame)
        self.rail_log.setReadOnly(True)
        self.rail_log.setMaximumBlockCount(120)
        self.rail_log.setFixedHeight(112)
        self.rail_log.setPlaceholderText("等待任务")
        layout.addWidget(self.rail_log, 0)
        self.log = self.rail_log
        return rail

    def _build_parameter_panel(self) -> QWidget:
        """构建只显示当前步骤的参数面板。"""
        panel = QFrame()
        panel.setObjectName("ParameterPanel")
        panel.setMinimumWidth(310)
        panel.setMaximumWidth(360)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(7)

        self.parameter_title = QLabel("数据与扫描")
        self.parameter_title.setObjectName("PageTitle")
        self.parameter_hint = QLabel("选择数据并设置扫描采样。")
        self.parameter_hint.setObjectName("MutedLabel")
        self.parameter_hint.setWordWrap(True)
        layout.addWidget(self.parameter_title)
        layout.addWidget(self.parameter_hint)

        self.parameter_stack = QStackedWidget()
        self.parameter_stack.addWidget(self._build_data_parameter_page())
        self.parameter_stack.addWidget(self._build_algorithm_parameter_page())
        layout.addWidget(self.parameter_stack, 1)

        self.folder_edit.textChanged.connect(self._refresh_step_states)
        self.mat_edit.textChanged.connect(self._refresh_step_states)
        self.fixed_k0.valueChanged.connect(self._refresh_step_states)
        self.scan_log_edit.textChanged.connect(self._refresh_step_states)
        self._update_data_source_ui()
        self._update_sampling_mode_ui()
        self._refresh_step_states()
        return panel

    def _wrap_parameter_content(self, content: QWidget) -> QWidget:
        """当前参数页使用固定布局，不创建滚动容器。"""
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        return content

    def _install_wheel_guards(self) -> None:
        """禁止滚轮修改下拉框和数值框，避免浏览页面时误改参数。"""
        guarded_widgets = [
            *self.findChildren(QComboBox),
            *self.findChildren(QAbstractSpinBox),
            *self.findChildren(QPlainTextEdit),
        ]
        for widget in guarded_widgets:
            widget.setProperty("wheelGuard", True)
            widget.installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:
        """拦截受保护参数控件的滚轮事件。"""
        if event.type() == QEvent.Type.Wheel and watched.property("wheelGuard"):
            event.ignore()
            return True
        return super().eventFilter(watched, event)

    def _build_data_parameter_page(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        layout.addWidget(SectionHeader("输入数据", "支持图像文件夹与 MAT 体数据。"))

        form = QFormLayout()
        form.setSpacing(8)
        self.data_source = QComboBox()
        self.data_source.addItem("图像文件夹", "image_folder")
        self.data_source.addItem("MAT 数据", "mat_file")
        self.data_source.currentIndexChanged.connect(self._update_data_source_ui)
        form.addRow("来源", self.data_source)

        self.image_intensity_mode = QComboBox()
        self.image_intensity_mode.addItem("8-bit 灰度兼容", "legacy_8bit")
        self.image_intensity_mode.addItem("12-bit 左对齐 Mono16", "mono12_uint16")
        mono12_index = self.image_intensity_mode.findData("mono12_uint16")
        if mono12_index >= 0:
            self.image_intensity_mode.setCurrentIndex(mono12_index)
        self.image_intensity_mode.currentIndexChanged.connect(self._refresh_step_states)
        form.addRow("图像灰度", self.image_intensity_mode)

        self.folder_edit = QLineEdit()
        self.folder_button = QPushButton("选择")
        self.folder_button.clicked.connect(self._choose_folder)
        form.addRow("图像目录", self._build_file_row(self.folder_edit, self.folder_button))

        self.mat_edit = QLineEdit()
        self.mat_button = QPushButton("选择")
        self.mat_button.clicked.connect(self._choose_mat_file)
        form.addRow("MAT 文件", self._build_file_row(self.mat_edit, self.mat_button))
        layout.addLayout(form)

        layout.addWidget(SectionHeader("扫描采样", "上传数据后直接设置扫描起点、步长和非均匀采样日志。"))
        sampling_form = QFormLayout()
        sampling_form.setSpacing(8)
        self.start_height = QDoubleSpinBox()
        self.start_height.setDecimals(4)
        self.start_height.setRange(-1e6, 1e6)
        sampling_form.addRow("起始高度 um", self.start_height)

        self.sampling_mode = QComboBox()
        self.sampling_mode.addItem("均匀采样", "uniform")
        self.sampling_mode.addItem("非均匀采样", "nonuniform")
        self.sampling_mode.currentIndexChanged.connect(self._update_sampling_mode_ui)
        sampling_form.addRow("采样模式", self.sampling_mode)

        self.step_size = QDoubleSpinBox()
        self.step_size.setDecimals(4)
        self.step_size.setRange(0.0001, 1e6)
        self.step_size.setValue(0.05)
        sampling_form.addRow("步长 um", self.step_size)

        self.scan_log_edit = QLineEdit()
        self.scan_log_button = QPushButton("选择")
        self.scan_log_button.clicked.connect(self._choose_scan_log)
        self.scan_log_edit.textChanged.connect(self._refresh_scan_log_summary)
        sampling_form.addRow("scan_log", self._build_file_row(self.scan_log_edit, self.scan_log_button))

        self.scan_log_summary_label = QLabel("未加载 scan_log")
        self.scan_log_summary_label.setObjectName("MutedLabel")
        self.scan_log_summary_label.setWordWrap(True)
        sampling_form.addRow("位移摘要", self.scan_log_summary_label)
        layout.addLayout(sampling_form)
        layout.addStretch(1)
        return self._wrap_parameter_content(content)

    def _build_sampling_parameter_page(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        layout.addWidget(SectionHeader("扫描采样", "配置轴向采样模型和位移信息。"))

        form = QFormLayout()
        form.setSpacing(8)
        self.start_height = QDoubleSpinBox()
        self.start_height.setDecimals(4)
        self.start_height.setRange(-1e6, 1e6)
        form.addRow("起始高度 um", self.start_height)

        self.sampling_mode = QComboBox()
        self.sampling_mode.addItem("均匀采样", "uniform")
        self.sampling_mode.addItem("非均匀采样", "nonuniform")
        self.sampling_mode.currentIndexChanged.connect(self._update_sampling_mode_ui)
        form.addRow("采样模式", self.sampling_mode)

        self.step_size = QDoubleSpinBox()
        self.step_size.setDecimals(4)
        self.step_size.setRange(0.0001, 1e6)
        self.step_size.setValue(0.05)
        form.addRow("步长 um", self.step_size)

        self.scan_log_edit = QLineEdit()
        self.scan_log_button = QPushButton("选择")
        self.scan_log_button.clicked.connect(self._choose_scan_log)
        self.scan_log_edit.textChanged.connect(self._refresh_scan_log_summary)
        form.addRow("scan_log", self._build_file_row(self.scan_log_edit, self.scan_log_button))

        self.scan_log_summary_label = QLabel("未加载 scan_log")
        self.scan_log_summary_label.setObjectName("MutedLabel")
        self.scan_log_summary_label.setWordWrap(True)
        form.addRow("位移摘要", self.scan_log_summary_label)
        layout.addLayout(form)
        layout.addStretch(1)
        return self._wrap_parameter_content(content)

    def _build_algorithm_parameter_page(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        layout.addWidget(SectionHeader("算法配置", "选择 K0、频谱、拟合与工作流参数。"))

        form = QFormLayout()
        form.setSpacing(8)
        self.fixed_k0 = QDoubleSpinBox()
        self.fixed_k0.setDecimals(6)
        self.fixed_k0.setRange(0.0, 1e6)
        self.fixed_k0.setValue(DEFAULT_FIXED_K0_VALUE)
        form.addRow("固定 K0", self.fixed_k0)

        self.window_size = QSpinBox()
        self.window_size.setRange(1, 200)
        self.window_size.setValue(9)
        form.addRow("窗口大小", self.window_size)

        self.analysis_window_name = QComboBox()
        self.analysis_window_name.addItem("Adaptive Hann", "adaptive_hann")
        self.analysis_window_name.addItem("Adaptive Hamming", "adaptive_hamming")
        self.analysis_window_name.addItem("Hamming", "hamming")
        self.analysis_window_name.addItem("Hann", "hann")
        self.analysis_window_name.addItem("None", "none")
        none_window_index = self.analysis_window_name.findData("none")
        if none_window_index >= 0:
            self.analysis_window_name.setCurrentIndex(none_window_index)
        form.addRow("窗函数", self.analysis_window_name)

        self.analysis_zero_padding_mode = QComboBox()
        _populate_zero_padding_combo(self.analysis_zero_padding_mode)
        form.addRow("补零", self.analysis_zero_padding_mode)

        self.expand_active_range_checkbox = QCheckBox("扩大有效范围")
        form.addRow("范围扩展", self.expand_active_range_checkbox)
        self.active_range_left_expansion_frames = QSpinBox()
        self.active_range_left_expansion_frames.setRange(0, 10000)
        self.active_range_left_expansion_frames.setValue(35)
        self.active_range_left_expansion_frames.setEnabled(False)
        self.expand_active_range_checkbox.toggled.connect(
            self.active_range_left_expansion_frames.setEnabled
        )
        form.addRow("左扩展帧", self.active_range_left_expansion_frames)
        self.active_range_right_expansion_frames = QSpinBox()
        self.active_range_right_expansion_frames.setRange(0, 10000)
        self.active_range_right_expansion_frames.setValue(35)
        self.active_range_right_expansion_frames.setEnabled(False)
        self.expand_active_range_checkbox.toggled.connect(
            self.active_range_right_expansion_frames.setEnabled
        )
        form.addRow("右扩展帧", self.active_range_right_expansion_frames)

        self.fitting = QComboBox()
        self.fitting.addItem("简单拟合", "simple")
        self.fitting.addItem("二次拟合", "quadratic")
        self.fitting.addItem("加权拟合", "weighted")
        form.addRow("拟合", self.fitting)

        self.unwrap = QComboBox()
        self.unwrap.addItem("全局解包裹", "global")
        self.unwrap.addItem("Itoh 局部解包裹", "itoh")
        self.unwrap.addItem("GR 解包裹", "gr")
        self.unwrap.addItem("PDA 局部解包裹", "pda")
        self.unwrap.addItem("Branch Search 局部解包裹", "branch_search")
        self.unwrap.setCurrentIndex(self.unwrap.findData("itoh"))
        form.addRow("解包裹", self.unwrap)

        self.phase_gap_method = QComboBox()
        self.phase_gap_method.addItem("FDA", "FDA")
        self.phase_gap_method.addItem("GFDA - Software carrier phase", "gfda_carrier_phase")
        self.phase_gap_method.addItem("GFDA - Software scatter fit", "gfda_scatter_fit")
        self.phase_gap_method.addItem("PhaseGap - Quality guided", "quality_guided")
        self.phase_gap_method.addItem("PhaseGap - Circular average", "circular_average")
        self.phase_gap_method.addItem("PhaseGap - Robust model fit", "robust_model_fit")
        self.phase_gap_method.addItem("PhaseGap - Branch cut", "branch_cut")
        self.phase_gap_method.addItem("PhaseGap - Weighted least squares", "weighted_least_squares")
        self.phase_gap_method.addItem("PhaseGap - Minimum Lp", "minimum_lp")
        self.phase_gap_method.currentIndexChanged.connect(self._update_analysis_mode_ui)
        self.phase_gap_method.currentIndexChanged.connect(self._mark_algorithm_configured)
        form.addRow("工作流", self.phase_gap_method)
        self.analysis_mode_label = QLabel("FDA")
        form.addRow("当前方法", self.analysis_mode_label)
        layout.addLayout(form)

        layout.addWidget(SectionHeader("运行", "算法参数确认后，在这里直接自动定 K0 或开始分析。"))
        self.auto_k0_button = QPushButton("自动定 K0")
        self.auto_k0_button.setObjectName("SecondaryButton")
        self.auto_k0_button.clicked.connect(self._start_auto_k0)
        self.start_button = QPushButton("开始分析")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._start_analysis)

        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        run_row.addWidget(self.auto_k0_button)
        run_row.addWidget(self.start_button)
        layout.addLayout(run_row)
        self.run_hint_label = QLabel("完成数据与扫描、算法配置后启用运行。")
        self.run_hint_label.setObjectName("MutedLabel")
        self.run_hint_label.setWordWrap(True)
        layout.addWidget(self.run_hint_label)
        layout.addStretch(1)
        return self._wrap_parameter_content(content)

    def _build_run_parameter_page(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(10)
        layout.addWidget(SectionHeader("结果动作", "确认状态后执行分析或导出结果。"))

        self.auto_k0_button = QPushButton("自动定 K0")
        self.auto_k0_button.setObjectName("SecondaryButton")
        self.auto_k0_button.clicked.connect(self._start_auto_k0)
        self.start_button = QPushButton("开始分析")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._start_analysis)

        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        run_row.addWidget(self.auto_k0_button)
        run_row.addWidget(self.start_button)
        layout.addLayout(run_row)
        self.run_hint_label = QLabel("完成数据与扫描、算法配置后启用运行。")
        self.run_hint_label.setObjectName("MutedLabel")
        self.run_hint_label.setWordWrap(True)
        layout.addWidget(self.run_hint_label)

        range_group, range_layout = self._build_section("有效范围")
        range_grid = QGridLayout()
        range_grid.addWidget(QLabel("帧范围"), 0, 0)
        self.active_range_label = QLabel("未确认")
        range_grid.addWidget(self.active_range_label, 0, 1)
        range_grid.addWidget(QLabel("帧数"), 1, 0)
        self.active_frame_count_label = QLabel("-")
        range_grid.addWidget(self.active_frame_count_label, 1, 1)
        range_layout.addLayout(range_grid)
        layout.addWidget(range_group)

        k0_group, k0_layout = self._build_section("自动 K0 摘要")
        k0_form = QFormLayout()
        self.auto_k0_value_label = QLabel("-")
        self.auto_k0_prominence_label = QLabel("-")
        self.auto_k0_window_label = QLabel("-")
        self.auto_k0_pad_label = QLabel("-")
        self.auto_k0_candidates_label = QLabel("-")
        k0_form.addRow("K0", self.auto_k0_value_label)
        k0_form.addRow("峰值显著性", self.auto_k0_prominence_label)
        k0_form.addRow("窗函数", self.auto_k0_window_label)
        k0_form.addRow("补零", self.auto_k0_pad_label)
        k0_form.addRow("候选数", self.auto_k0_candidates_label)
        k0_layout.addLayout(k0_form)
        layout.addWidget(k0_group)

        readout_group, readout_layout = self._build_section("当前读数")
        readout_grid = QGridLayout()
        self.layer_label = QLabel("-")
        self.x_label = QLabel("-")
        self.y_label = QLabel("-")
        self.value_label = QLabel("-")
        for row, (label, value) in enumerate(
            (
                ("图层", self.layer_label),
                ("X", self.x_label),
                ("Y", self.y_label),
                ("值", self.value_label),
            )
        ):
            readout_grid.addWidget(QLabel(label), row, 0)
            readout_grid.addWidget(value, row, 1)
        readout_layout.addLayout(readout_grid)
        layout.addWidget(readout_group)
        self.readout_label: QLabel | None = None

        self.export_text_button = QPushButton("导出数据")
        self.export_text_button.setObjectName("ExportButton")
        self.export_text_button.clicked.connect(self._export_text)
        self.export_text_button.setEnabled(False)
        self.export_figures_button = QPushButton("导出图片")
        self.export_figures_button.setObjectName("ExportButton")
        self.export_figures_button.clicked.connect(self._export_figures)
        self.export_figures_button.setEnabled(False)
        export_row = QHBoxLayout()
        export_row.addWidget(self.export_text_button)
        export_row.addWidget(self.export_figures_button)
        layout.addLayout(export_row)
        layout.addStretch(1)
        return self._wrap_parameter_content(content)

    def _select_step(self, index: int) -> None:
        """切换当前参数步骤并保持控件实例和值不变。"""
        index = max(0, min(index, self.parameter_stack.count() - 1))
        titles = ("数据与扫描", "算法与运行")
        hints = (
            "选择数据并设置扫描采样。",
            "设置算法参数，并在同一页直接自动定 K0 或开始分析。",
        )
        if index == 1:
            self._algorithm_config_touched = True
        self.parameter_stack.setCurrentIndex(index)
        self.parameter_title.setText(titles[index])
        self.parameter_hint.setText(hints[index])
        self.step_buttons[index].setChecked(True)
        self._refresh_step_states()

    def _select_result_tab(self, key: str) -> None:
        """从顶部工具带快速切换到指定结果视图。"""
        self._select_main_page(0)
        if hasattr(self, "result_content_stack") and hasattr(self, "tabs"):
            self.result_content_stack.setCurrentWidget(self.tabs)
            target = self._plot_tabs.get(key)
            if target is not None:
                self.tabs.setCurrentWidget(target)

    def _refresh_step_states(self, *_args) -> None:
        """根据当前输入给步骤导航提供轻量完成提示。"""
        data_scan_ready = self._is_data_scan_ready()
        states = (
            "complete" if data_scan_ready else "idle",
            "complete" if self._is_algorithm_ready() else "idle",
        )
        for button, state in zip(self.step_buttons, states):
            button.set_step_state(state)
        self._update_run_controls()

    def _mark_algorithm_configured(self, *_args) -> None:
        """记录用户已经进入或调整算法配置，作为运行前置条件之一。"""
        self._algorithm_config_touched = True
        self._refresh_step_states()

    def _has_selected_source(self) -> bool:
        """判断当前数据来源是否已有路径输入。"""
        if not hasattr(self, "data_source"):
            return False
        data_source = str(self.data_source.currentData())
        if data_source == "mat_file":
            return bool(self.mat_edit.text().strip())
        return bool(self.folder_edit.text().strip())

    def _requires_scan_log(self) -> bool:
        """只有显式非均匀采样模式需要 scan_log。"""
        if not hasattr(self, "sampling_mode") or not hasattr(self, "phase_gap_method"):
            return False
        return str(self.sampling_mode.currentData()) == "nonuniform"

    def _is_data_scan_ready(self) -> bool:
        """数据路径和采样前置条件均满足时返回 True。"""
        if not self._has_selected_source():
            return False
        return bool(self.scan_log_edit.text().strip()) if self._requires_scan_log() else True

    def _is_algorithm_ready(self) -> bool:
        """用户确认过算法页后才允许进入运行动作。"""
        return bool(self._algorithm_config_touched)

    def _is_auto_k0_ready(self) -> bool:
        """自动 K0 的最小前置条件：数据/扫描就绪并完成算法配置。"""
        return self._is_data_scan_ready() and self._is_algorithm_ready()

    def _is_analysis_ready(self) -> bool:
        """正式分析需要固定 K0 或已完成自动 K0。"""
        if not self._is_auto_k0_ready():
            return False
        has_k0 = float(self.fixed_k0.value()) > 0.0
        return has_k0

    def _update_run_controls(self) -> None:
        """按工作流前置条件统一控制运行按钮，避免右侧操作提前出现可点状态。"""
        if not hasattr(self, "auto_k0_button") or not hasattr(self, "start_button"):
            return
        running = self._thread is not None and self._thread.isRunning()
        auto_ready = (not running) and self._is_auto_k0_ready()
        analysis_ready = (not running) and self._is_analysis_ready()
        self.auto_k0_button.setEnabled(auto_ready)
        self.start_button.setEnabled(analysis_ready)
        if not hasattr(self, "run_hint_label"):
            return
        if running:
            hint = "任务运行中，参数已锁定。"
        elif not self._has_selected_source():
            hint = "先在“数据与扫描”中选择图像目录或 MAT 文件。"
        elif self._requires_scan_log() and not self.scan_log_edit.text().strip():
            hint = "非均匀采样需要先选择 scan_log。"
        elif not self._is_algorithm_ready():
            hint = "进入“算法配置”确认参数后，才启用自动 K0 和分析。"
        elif not self._is_analysis_ready():
            hint = "自动定 K0 或填写固定 K0 后，才启用正式分析。"
        else:
            hint = "前置条件已满足，可自动定 K0 或开始分析。"
        self.run_hint_label.setText(hint)

    def _select_main_page(self, index: int) -> None:
        """切换工作台、日志和设置页面。"""
        index = max(0, min(index, self.main_pages.count() - 1))
        self.main_pages.setCurrentIndex(index)
        self.global_nav_buttons[index].setChecked(True)

    def _on_theme_changed(self, _index: int = -1) -> None:
        self._theme_mode = str(self.theme_combo.currentData())
        self._apply_theme()

    def _handle_system_color_scheme_changed(self, _scheme) -> None:
        if self._theme_mode == "system":
            self._apply_theme()

    def _apply_theme(self) -> None:
        """应用显式主题或当前系统主题。"""
        color_scheme = QGuiApplication.styleHints().colorScheme()
        resolved = resolve_theme_mode(self._theme_mode, color_scheme)
        stylesheet = build_stylesheet(resolved)
        application = QApplication.instance()
        if application is not None:
            application.setStyleSheet(stylesheet)
        else:
            self.setStyleSheet(stylesheet)

    def _apply_visual_style(self) -> None:
        """集中设置桌面工作台的视觉风格，避免样式散落在各个控件创建点。"""
        self.setMinimumSize(1180, 760)
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #eef2f6;
                color: #172033;
                font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
                font-size: 13px;
            }
            QFrame#HeaderPanel, QFrame#SidePanel, QFrame#CenterPanel, QFrame#BottomPanel {
                background: #ffffff;
                border: 1px solid #d8dee8;
                border-radius: 8px;
            }
            QGroupBox {
                background: #fbfcfe;
                border: 1px solid #dbe2ea;
                border-radius: 8px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #253247;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: #ffffff;
                border: 1px solid #cfd7e3;
                border-radius: 6px;
                padding: 5px 7px;
                selection-background-color: #2563eb;
            }
            QPushButton {
                background: #f8fafc;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: 7px 10px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #edf2f7;
                border-color: #94a3b8;
            }
            QPushButton:disabled {
                color: #94a3b8;
                background: #f1f5f9;
            }
            QPushButton#PrimaryButton {
                background: #1f6feb;
                border-color: #1f6feb;
                color: #ffffff;
            }
            QPushButton#PrimaryButton:hover {
                background: #1d5fd2;
            }
            QPushButton#SecondaryButton {
                background: #0f766e;
                border-color: #0f766e;
                color: #ffffff;
            }
            QPushButton#ExportButton {
                background: #fff7ed;
                border-color: #fdba74;
                color: #9a3412;
            }
            QTabWidget::pane {
                border: 1px solid #d8dee8;
                border-radius: 8px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                background: #eef2f6;
                border: 1px solid #d8dee8;
                border-bottom: none;
                padding: 7px 12px;
                margin-right: 2px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #1f6feb;
            }
            QProgressBar {
                border: 1px solid #cfd7e3;
                border-radius: 6px;
                background: #f8fafc;
                height: 12px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #1f6feb;
                border-radius: 5px;
            }
            """
        )

    def _build_section(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        """创建统一参数分组，保持左侧流程面板排版一致。"""
        section = QGroupBox(title)
        layout = QVBoxLayout(section)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        return section, layout

    def _build_file_row(self, edit: QLineEdit, button: QPushButton) -> QWidget:
        """创建路径输入与选择按钮的标准行，减少文件来源控件的重复布局。"""
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(edit, 1)
        layout.addWidget(button, 0)
        return row

    def _build_left_panel_redesign(self) -> QWidget:
        """构建新版左侧流程面板，只重排控件，不改变参数字段与业务含义。"""
        panel = QFrame()
        panel.setObjectName("SidePanel")
        panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        panel.setMinimumWidth(320)
        panel.setMaximumWidth(390)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("流程参数")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #101828;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(10)

        data_section, data_layout = self._build_section("1. 数据来源")
        data_form = QFormLayout()
        data_form.setLabelAlignment(Qt.AlignLeft)
        self.data_source = QComboBox()
        self.data_source.addItem("图像文件夹", "image_folder")
        self.data_source.addItem("MAT 数据", "mat_file")
        self.data_source.currentIndexChanged.connect(self._update_data_source_ui)
        data_form.addRow("来源", self.data_source)

        self.folder_edit = QLineEdit()
        self.folder_button = QPushButton("选择")
        self.folder_button.clicked.connect(self._choose_folder)
        data_form.addRow("图像目录", self._build_file_row(self.folder_edit, self.folder_button))

        self.mat_edit = QLineEdit()
        self.mat_button = QPushButton("选择")
        self.mat_button.clicked.connect(self._choose_mat_file)
        data_form.addRow("MAT 文件", self._build_file_row(self.mat_edit, self.mat_button))
        data_layout.addLayout(data_form)
        content_layout.addWidget(data_section)

        sampling_section, sampling_layout = self._build_section("2. 扫描采样")
        sampling_form = QFormLayout()
        sampling_form.setLabelAlignment(Qt.AlignLeft)
        self.start_height = QDoubleSpinBox()
        self.start_height.setDecimals(4)
        self.start_height.setRange(-1e6, 1e6)
        self.start_height.setValue(0.0)
        sampling_form.addRow("起始高度 um", self.start_height)

        self.sampling_mode = QComboBox()
        self.sampling_mode.addItem("均匀采样", "uniform")
        self.sampling_mode.addItem("非均匀采样", "nonuniform")
        self.sampling_mode.currentIndexChanged.connect(self._update_sampling_mode_ui)
        sampling_form.addRow("采样模式", self.sampling_mode)

        self.step_size = QDoubleSpinBox()
        self.step_size.setDecimals(4)
        self.step_size.setRange(0.0001, 1e6)
        self.step_size.setValue(0.05)
        sampling_form.addRow("步长 um", self.step_size)

        self.scan_log_edit = QLineEdit()
        self.scan_log_button = QPushButton("选择")
        self.scan_log_button.clicked.connect(self._choose_scan_log)
        self.scan_log_edit.textChanged.connect(self._refresh_scan_log_summary)
        sampling_form.addRow("scan_log", self._build_file_row(self.scan_log_edit, self.scan_log_button))

        self.scan_log_summary_label = QLabel("未加载 scan_log")
        self.scan_log_summary_label.setWordWrap(True)
        self.scan_log_summary_label.setStyleSheet("color: #667085;")
        sampling_form.addRow("位移摘要", self.scan_log_summary_label)
        sampling_layout.addLayout(sampling_form)
        content_layout.addWidget(sampling_section)

        algorithm_section, algorithm_layout = self._build_section("3. 算法配置")
        algorithm_form = QFormLayout()
        algorithm_form.setLabelAlignment(Qt.AlignLeft)
        self.fixed_k0 = QDoubleSpinBox()
        self.fixed_k0.setDecimals(6)
        self.fixed_k0.setRange(0.0, 1e6)
        self.fixed_k0.setValue(DEFAULT_FIXED_K0_VALUE)
        algorithm_form.addRow("固定 K0", self.fixed_k0)

        self.window_size = QSpinBox()
        self.window_size.setRange(1, 200)
        self.window_size.setValue(9)
        algorithm_form.addRow("窗口大小", self.window_size)

        self.analysis_window_name = QComboBox()
        self.analysis_window_name.addItem("Adaptive Hann", "adaptive_hann")
        self.analysis_window_name.addItem("Adaptive Hamming", "adaptive_hamming")
        self.analysis_window_name.addItem("Hamming", "hamming")
        self.analysis_window_name.addItem("Hann", "hann")
        self.analysis_window_name.addItem("None", "none")
        none_window_index = self.analysis_window_name.findData("none")
        if none_window_index >= 0:
            self.analysis_window_name.setCurrentIndex(none_window_index)
        algorithm_form.addRow("窗函数", self.analysis_window_name)

        self.analysis_zero_padding_mode = QComboBox()
        _populate_zero_padding_combo(self.analysis_zero_padding_mode, frame_count=None)
        self.analysis_zero_padding_mode.setCurrentIndex(self.analysis_zero_padding_mode.findData("fixed_512"))
        algorithm_form.addRow("补零", self.analysis_zero_padding_mode)

        self.expand_active_range_checkbox = QCheckBox("扩大有效范围")
        self.expand_active_range_checkbox.setChecked(False)
        algorithm_form.addRow("范围扩展", self.expand_active_range_checkbox)

        self.active_range_left_expansion_frames = QSpinBox()
        self.active_range_left_expansion_frames.setRange(0, 10000)
        self.active_range_left_expansion_frames.setValue(35)
        self.active_range_left_expansion_frames.setEnabled(False)
        self.expand_active_range_checkbox.toggled.connect(self.active_range_left_expansion_frames.setEnabled)
        algorithm_form.addRow("左扩展帧", self.active_range_left_expansion_frames)

        self.active_range_right_expansion_frames = QSpinBox()
        self.active_range_right_expansion_frames.setRange(0, 10000)
        self.active_range_right_expansion_frames.setValue(35)
        self.active_range_right_expansion_frames.setEnabled(False)
        self.expand_active_range_checkbox.toggled.connect(self.active_range_right_expansion_frames.setEnabled)
        algorithm_form.addRow("右扩展帧", self.active_range_right_expansion_frames)

        self.fitting = QComboBox()
        self.fitting.addItem("简单拟合", "simple")
        self.fitting.addItem("二次拟合", "quadratic")
        self.fitting.addItem("加权拟合", "weighted")
        algorithm_form.addRow("拟合", self.fitting)

        self.unwrap = QComboBox()
        self.unwrap.addItem("全局解包裹", "global")
        self.unwrap.addItem("Itoh 局部解包裹", "itoh")
        self.unwrap.addItem("GR 解包裹", "gr")
        self.unwrap.addItem("PDA 局部解包裹", "pda")
        self.unwrap.addItem("Branch Search 局部解包裹", "branch_search")
        self.unwrap.setCurrentIndex(self.unwrap.findData("itoh"))
        algorithm_form.addRow("解包裹", self.unwrap)

        self.analysis_mode_label = QLabel("FDA")
        algorithm_form.addRow("方法", self.analysis_mode_label)

        self.phase_gap_method = QComboBox()
        self.phase_gap_method.addItem("FDA", "FDA")
        self.phase_gap_method.addItem("GFDA - Software carrier phase", "gfda_carrier_phase")
        self.phase_gap_method.addItem("GFDA - Software scatter fit", "gfda_scatter_fit")
        self.phase_gap_method.addItem("PhaseGap - Quality guided", "quality_guided")
        self.phase_gap_method.addItem("PhaseGap - Circular average", "circular_average")
        self.phase_gap_method.addItem("PhaseGap - Robust model fit", "robust_model_fit")
        self.phase_gap_method.addItem("PhaseGap - Branch cut", "branch_cut")
        self.phase_gap_method.addItem("PhaseGap - Weighted least squares", "weighted_least_squares")
        self.phase_gap_method.addItem("PhaseGap - Minimum Lp", "minimum_lp")
        self.phase_gap_method.currentIndexChanged.connect(self._update_analysis_mode_ui)
        algorithm_form.addRow("工作流", self.phase_gap_method)
        algorithm_layout.addLayout(algorithm_form)
        content_layout.addWidget(algorithm_section)

        command_section, command_layout = self._build_section("4. 命令")
        self.auto_k0_button = QPushButton("自动定 K0")
        self.auto_k0_button.setObjectName("SecondaryButton")
        self.auto_k0_button.clicked.connect(self._start_auto_k0)
        self.start_button = QPushButton("开始分析")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self._start_analysis)
        self.export_text_button = QPushButton("导出数据")
        self.export_text_button.setObjectName("ExportButton")
        self.export_text_button.clicked.connect(self._export_text)
        self.export_text_button.setEnabled(False)
        self.export_figures_button = QPushButton("导出图片")
        self.export_figures_button.setObjectName("ExportButton")
        self.export_figures_button.clicked.connect(self._export_figures)
        self.export_figures_button.setEnabled(False)

        command_layout.addWidget(self.auto_k0_button)
        command_layout.addWidget(self.start_button)
        export_row = QHBoxLayout()
        export_row.setContentsMargins(0, 0, 0, 0)
        export_row.setSpacing(8)
        export_row.addWidget(self.export_text_button)
        export_row.addWidget(self.export_figures_button)
        command_layout.addLayout(export_row)
        content_layout.addWidget(command_section)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        self._update_data_source_ui()
        self._update_sampling_mode_ui()
        return panel

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
        self.step_size.setValue(0.05)
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
        self.fixed_k0.setValue(DEFAULT_FIXED_K0_VALUE)
        form.addRow("固定 K0", self.fixed_k0)

        self.window_size = QSpinBox()
        self.window_size.setRange(1, 200)
        self.window_size.setValue(9)
        form.addRow("窗口大小", self.window_size)

        self.analysis_window_name = QComboBox()
        self.analysis_window_name.addItem("Adaptive Hann", "adaptive_hann")
        self.analysis_window_name.addItem("Adaptive Hamming", "adaptive_hamming")
        self.analysis_window_name.addItem("Hamming", "hamming")
        self.analysis_window_name.addItem("Hann", "hann")
        self.analysis_window_name.addItem("None", "none")
        none_window_index = self.analysis_window_name.findData("none")
        if none_window_index >= 0:
            self.analysis_window_name.setCurrentIndex(none_window_index)
        form.addRow("分析窗函数", self.analysis_window_name)

        self.analysis_zero_padding_mode = QComboBox()
        _populate_zero_padding_combo(self.analysis_zero_padding_mode, frame_count=None)
        self.analysis_zero_padding_mode.setCurrentIndex(self.analysis_zero_padding_mode.findData("fixed_512"))
        form.addRow("正式分析补零", self.analysis_zero_padding_mode)

        self.expand_active_range_checkbox = QCheckBox("扩大有效范围")
        self.expand_active_range_checkbox.setChecked(False)
        form.addRow("有效范围扩展", self.expand_active_range_checkbox)

        self.active_range_left_expansion_frames = QSpinBox()
        self.active_range_left_expansion_frames.setRange(0, 10000)
        self.active_range_left_expansion_frames.setValue(35)
        self.active_range_left_expansion_frames.setEnabled(False)
        self.expand_active_range_checkbox.toggled.connect(self.active_range_left_expansion_frames.setEnabled)
        form.addRow("左扩展帧数", self.active_range_left_expansion_frames)

        self.active_range_right_expansion_frames = QSpinBox()
        self.active_range_right_expansion_frames.setRange(0, 10000)
        self.active_range_right_expansion_frames.setValue(35)
        self.active_range_right_expansion_frames.setEnabled(False)
        self.expand_active_range_checkbox.toggled.connect(self.active_range_right_expansion_frames.setEnabled)
        form.addRow("右扩展帧数", self.active_range_right_expansion_frames)

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
        self.unwrap.setCurrentIndex(self.unwrap.findData("itoh"))
        form.addRow("解包裹方式", self.unwrap)

        self.analysis_mode_label = QLabel("FDA")
        form.addRow("方法", self.analysis_mode_label)

        self.phase_gap_method = QComboBox()
        self.phase_gap_method.addItem("FDA", "FDA")
        self.phase_gap_method.addItem("GFDA - Software carrier phase", "gfda_carrier_phase")
        self.phase_gap_method.addItem("GFDA - Software scatter fit", "gfda_scatter_fit")
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
        panel.setObjectName("ResultWorkspace")
        panel.setMinimumSize(0, 0)
        panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        preview_title = QLabel("结果预览")
        preview_title.setObjectName("PageTitle")
        title_row.addWidget(preview_title, 0)
        title_row.addStretch(1)
        layout.addLayout(title_row)
        self.tabs = QTabWidget()
        self.tabs.setMinimumSize(0, 0)
        self.tabs.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.tabBar().setElideMode(Qt.ElideRight)
        self.tabs.tabBar().setExpanding(False)
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
        self.result_content_stack = QStackedWidget()
        self.result_content_stack.setMinimumSize(0, 0)
        self.result_content_stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.result_content_stack.addWidget(self._build_empty_results_page())
        self.result_content_stack.addWidget(self.tabs)
        layout.addWidget(self.result_content_stack, 1)
        return panel

    def _build_empty_results_page(self) -> QWidget:
        """在尚无结果时提供明确的下一步提示。"""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(7)
        layout.addStretch(1)

        title = QLabel("准备开始扫描分析")
        title.setObjectName("EmptyStateTitle")
        title.setAlignment(Qt.AlignCenter)
        description = QLabel("完成数据与扫描、算法配置后，在右侧运行自动 K0 或正式分析。")
        description.setObjectName("MutedLabel")
        description.setAlignment(Qt.AlignCenter)
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)
        layout.addStretch(1)
        return page

    def _build_preview_flow_strip(self) -> QHBoxLayout:
        """构建预览流程条，让数据、K0、分析、预览、导出形成明确顺序。"""
        flow_layout = QHBoxLayout()
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setSpacing(6)
        self._preview_step_labels = []
        for label in ("1 数据", "2 K0", "3 分析", "4 预览", "5 导出"):
            step_label = StatusPill(label)
            step_label.setMinimumHeight(28)
            self._preview_step_labels.append(step_label)
            flow_layout.addWidget(step_label)
        return flow_layout

    def _set_preview_stage(self, active_index: int) -> None:
        """兼容旧调用；当前界面不再显示第二套流程条。"""
        for index, label in enumerate(self._preview_step_labels):
            if index < active_index:
                label.set_status("success")
            elif index == active_index:
                label.set_status("running")
            else:
                label.set_status("idle")

    def _build_right_panel(self) -> QWidget:
        """构建右侧数据读数区。"""
        self.value_panel = QFrame()
        self.value_panel.setObjectName("SidePanel")
        self.value_panel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.value_panel.setMinimumWidth(280)
        self.value_panel.setMaximumWidth(340)
        layout = QVBoxLayout(self.value_panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        title = QLabel("预览读数")
        title.setStyleSheet("font-size: 16px; font-weight: 700; color: #101828;")
        layout.addWidget(title)

        readout_section, readout_layout = self._build_section("当前点")
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        grid.addWidget(QLabel("图层"), 0, 0)
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
        readout_layout.addLayout(grid)
        layout.addWidget(readout_section)

        range_section, range_layout = self._build_section("有效范围")
        range_grid = QGridLayout()
        range_grid.setHorizontalSpacing(10)
        range_grid.setVerticalSpacing(8)
        range_grid.addWidget(QLabel("帧范围"), 0, 0)
        self.active_range_label = QLabel("未确认")
        range_grid.addWidget(self.active_range_label, 0, 1)
        range_grid.addWidget(QLabel("帧数"), 1, 0)
        self.active_frame_count_label = QLabel("-")
        range_grid.addWidget(self.active_frame_count_label, 1, 1)
        range_layout.addLayout(range_grid)
        layout.addWidget(range_section)

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
        """构建紧凑底部状态栏。"""
        panel = QFrame()
        panel.setObjectName("BottomStatusBar")
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(12, 7, 12, 7)
        layout.setSpacing(10)

        self.status_label = QLabel("空闲")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMaximumWidth(260)
        self.latest_log_label = QLabel("等待任务")
        self.latest_log_label.setObjectName("LatestLogLabel")
        self.latest_log_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.status_label, 0)
        layout.addWidget(self.progress, 0)
        layout.addStretch(1)
        return panel

    def _build_plot_tab(self, canvas, key: str) -> QWidget:
        """为单个画布创建带工具栏和提示文本的标准 tab 容器。"""
        container = QWidget()
        container.setMinimumSize(0, 0)
        container.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        tab_layout = QVBoxLayout(container)
        tab_layout.setContentsMargins(8, 8, 8, 8)
        tab_layout.setSpacing(6)

        toolbar_row = QHBoxLayout()
        toolbar_row.setContentsMargins(0, 0, 0, 0)
        toolbar_row.setSpacing(6)
        # Matplotlib 与 PyVista 的工具栏体系不同，这里只给 Matplotlib 画布挂导航栏。
        if hasattr(canvas, "figure"):
            toolbar = PlotToolbar(canvas, self)
            toolbar_row.addWidget(toolbar, 1)
        else:
            reset_button = QPushButton("重置视角")
            reset_button.clicked.connect(canvas.reset_view)
            toolbar_row.addWidget(reset_button, 0)
            toolbar_row.addStretch(1)
        copy_button = QPushButton("复制视图")
        copy_button.clicked.connect(canvas.copy_current_view_to_clipboard)
        toolbar_row.addWidget(copy_button, 0)

        info_label = QLabel("预览就绪后显示当前读数。")
        info_label.setStyleSheet("color: #667085; padding: 2px 0;")
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
        self.image_intensity_mode.setEnabled(not is_mat_file)
        self.folder_edit.setEnabled(not is_mat_file)
        self.folder_button.setEnabled(not is_mat_file)
        self.mat_edit.setEnabled(is_mat_file)
        self.mat_button.setEnabled(is_mat_file)
        self._refresh_step_states()

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
            f"点数: {positions.shape[0]}\n"
            f"前3个: {head}\n后3个: {tail}"
        )

    def _update_sampling_mode_ui(self) -> None:
        """根据采样模式切换步长/scan_log 相关控件可用状态。"""
        is_nonuniform = self._requires_scan_log()
        self.sampling_mode.setEnabled(True)
        self.step_size.setEnabled(not is_nonuniform)
        self.scan_log_edit.setEnabled(is_nonuniform)
        self.scan_log_button.setEnabled(is_nonuniform)
        if is_nonuniform:
            self._refresh_scan_log_summary()
        else:
            self.scan_log_summary_label.setText("均匀采样模式不使用 scan_log")
        self._refresh_step_states()

    def _is_diagnostic_mode(self) -> bool:
        """当前选择非基础 FDA 工作流时显示诊断摘要页。"""
        return str(self.phase_gap_method.currentData()).strip().lower() != "fda"

    def _is_phase_gap_mode(self) -> bool:
        """只有 PhaseGap 系列工作流需要显示 phase-gap 专属图层。"""
        method = str(self.phase_gap_method.currentData()).strip().lower()
        return method not in {"fda", *GFDA_SOFTWARE_METHODS}

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
        if self._is_phase_gap_mode():
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
            method = str(self.phase_gap_method.currentData()).strip().lower()
            if method == "fda":
                label = "FDA"
                auto_k0_label = "自动定 K0"
            elif method in GFDA_SOFTWARE_METHODS:
                label = "GFDA"
                auto_k0_label = "GFDA 自动定 K0"
            else:
                label = "PhaseGap"
                auto_k0_label = "自动定 K0"
            self.analysis_mode_label.setText(label)
            self.auto_k0_button.setText(auto_k0_label)
        self._update_sampling_mode_ui()
        self._refresh_step_states()
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
                "scan_positions_raw_um",
                "scan_positions_used_um",
                "scan_positions_monotone_um",
                "scan_step_raw_um",
                "scan_step_monotone_um",
                "scan_step_reversal_mask",
                "scan_position_correction_um",
            )
            if key in result.extras
        ]
        phase_gap_method = str(result.extras.get("phase_gap_method", "unknown"))
        if phase_gap_method.upper() == "FDA":
            result_label = "FDA级次修正"
        elif phase_gap_method.upper() == "GFDA":
            result_label = "GFDA非均匀级次修正"
        else:
            result_label = "PhaseGap最终高度"
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

    def _current_source_key(self) -> tuple[str, str, str, str, str, bool, int, int]:
        """生成当前数据源签名，用于判断自动 K0 的有效范围缓存是否仍可复用。"""
        data_source = str(self.data_source.currentData())
        source_text = self.mat_edit.text().strip() if data_source == "mat_file" else self.folder_edit.text().strip()
        intensity_mode = str(self.image_intensity_mode.currentData()) if data_source == "image_folder" else "mat"
        sampling_mode = str(self.sampling_mode.currentData())
        scan_log_text = self.scan_log_edit.text().strip() if sampling_mode == "nonuniform" else ""
        # Windows 路径大小写不敏感，这里统一 casefold，避免同一路径不同大小写导致缓存误判失效。
        return (
            data_source,
            str(Path(source_text)).casefold(),
            intensity_mode,
            sampling_mode,
            str(Path(scan_log_text)).casefold() if scan_log_text else "",
            bool(self.expand_active_range_checkbox.isChecked()),
            int(self.active_range_left_expansion_frames.value()),
            int(self.active_range_right_expansion_frames.value()),
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
        workflow = str(self.phase_gap_method.currentData())
        fixed_k0_value = float(self.fixed_k0.value())
        fixed_k0 = fixed_k0_value if fixed_k0_value > 0.0 else None
        if fixed_k0 is None:
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
            image_intensity_mode=str(self.image_intensity_mode.currentData()) if data_source == "image_folder" else "legacy_8bit",
            mat_path=mat_path,
            phase_gap_method=workflow,
            sampling_mode=sampling_mode,
            scan_log_path=scan_log_path,
            sample_positions_um=None,
            active_range=self._cached_active_range_for_current_source(),
            expand_active_range=bool(self.expand_active_range_checkbox.isChecked()),
            active_range_left_expansion_frames=int(self.active_range_left_expansion_frames.value()),
            active_range_right_expansion_frames=int(self.active_range_right_expansion_frames.value()),
        )

    def _set_running_state(self, running: bool) -> None:
        """统一控制任务按钮的启停状态。"""
        self.parameter_stack.setEnabled(not running)
        self.step_rail.setEnabled(not running)
        self.auto_k0_button.setText("自动定 K0")
        self.start_button.setText("开始分析")
        if running:
            self.auto_k0_button.setEnabled(False)
            self.start_button.setEnabled(False)
            if hasattr(self, "run_hint_label"):
                self.run_hint_label.setText("任务运行中，参数已锁定。")
        else:
            self._update_run_controls()

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
        if not self._is_analysis_ready():
            error_hint = self.run_hint_label.text()
            self._select_main_page(0)
            self._select_step(1 if self._is_data_scan_ready() else 0)
            QMessageBox.critical(self, "参数错误", error_hint)
            return
        try:
            params = self._build_params()
        except Exception as exc:
            self._focus_parameter_error(str(exc))
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
        self._set_preview_stage(3)
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
        if not self._is_auto_k0_ready():
            error_hint = self.run_hint_label.text()
            self._select_main_page(0)
            self._select_step(1 if self._is_data_scan_ready() else 0)
            QMessageBox.critical(self, "参数错误", error_hint)
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
        workflow = str(self.phase_gap_method.currentData())
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
        self._set_preview_stage(1)
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
            active_range_left_expansion_frames=int(self.active_range_left_expansion_frames.value()),
            active_range_right_expansion_frames=int(self.active_range_right_expansion_frames.value()),
            data_source=data_source,
            image_intensity_mode=str(self.image_intensity_mode.currentData()) if data_source == "image_folder" else "legacy_8bit",
            mat_path=mat_path,
            sampling_mode=sampling_mode,
            scan_log_path=scan_log_path,
            phase_gap_method=workflow,
            window_size=int(self.window_size.value()),
            fitting_method=str(self.fitting.currentData()),
            unwrap_method=str(self.unwrap.currentData()),
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
        """追加完整日志，并同步刷新左侧日志与最新消息状态。"""
        self.log.appendPlainText(message)
        if hasattr(self, "rail_log") and self.rail_log is not self.log:
            self.rail_log.appendPlainText(message)
        self.latest_log_label.setText(message)

    def _focus_parameter_error(self, message: str) -> None:
        """根据错误内容定位到最可能需要修正的参数步骤。"""
        lowered = message.lower()
        if any(token in lowered for token in ("文件夹", "mat 文件", "数据源")):
            step = 0
        elif any(token in lowered for token in ("scan_log", "采样", "步长")):
            step = 0
        else:
            step = 1
        self._select_main_page(0)
        self._select_step(step)
        self.step_buttons[step].set_step_state("error")

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

    def _check_for_updates(self) -> None:
        """启动后台更新检查，避免网络请求阻塞 GUI。"""
        if self._update_thread is not None and self._update_thread.isRunning():
            QMessageBox.information(self, "检查更新", "已有更新任务正在运行。")
            return
        self.update_button.setEnabled(False)
        self.update_button.setText("检查中")
        self._append_log(f"正在检查更新，当前版本 {__version__}。")
        self._start_update_worker(UpdateCheckWorker(__version__), "check")

    def _start_update_worker(self, worker: object, task: str) -> None:
        """创建更新线程；检查与下载共用同一套生命周期管理。"""
        thread = QThread(self)
        self._update_thread = thread
        self._update_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        if task == "check":
            worker.finished.connect(self._handle_update_check_finished)
        else:
            worker.progress.connect(self._handle_update_download_progress)
            worker.finished.connect(self._handle_update_download_finished)
        worker.failed.connect(self._handle_update_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finalize_update_thread)
        thread.start()

    def _finalize_update_thread(self) -> None:
        """恢复更新按钮状态并释放 worker 引用。"""
        self._update_thread = None
        self._update_worker = None
        if hasattr(self, "update_button"):
            self.update_button.setEnabled(True)
            self.update_button.setText("检查更新")

    def _handle_update_check_finished(self, info: object) -> None:
        """展示更新检查结果，并在用户确认后下载更新包。"""
        update_info = info if isinstance(info, UpdateInfo) else None
        if update_info is None:
            QMessageBox.warning(self, "检查更新", "更新结果格式异常。")
            return
        if not update_info.update_available:
            QMessageBox.information(self, "检查更新", f"当前已是最新版本：{update_info.current_version}")
            self._append_log(f"当前已是最新版本：{update_info.current_version}")
            return
        if not update_info.download_url or not update_info.asset_name:
            button = QMessageBox.question(
                self,
                "发现新版本",
                f"发现新版本 {update_info.latest_version}，但该 Release 没有可自动下载的安装包。\n是否打开发布页面？",
            )
            if button == QMessageBox.Yes and update_info.release_url:
                QDesktopServices.openUrl(QUrl(update_info.release_url))
            return

        cached_hint = "\n本机已有完整缓存，将直接复用。" if update_info.cached_path is not None else ""
        button = QMessageBox.question(
            self,
            "发现新版本",
            (
                f"当前版本：{update_info.current_version}\n"
                f"最新版本：{update_info.latest_version}\n"
                f"安装包：{update_info.asset_name}{cached_hint}\n\n"
                "是否现在下载并安装？"
            ),
        )
        if button != QMessageBox.Yes:
            self._append_log(f"发现新版本 {update_info.latest_version}，用户暂不更新。")
            return
        self.update_button.setEnabled(False)
        self.update_button.setText("下载中")
        self._append_log(f"开始下载更新包：{update_info.asset_name}")
        self._start_update_worker(UpdateDownloadWorker(update_info), "download")

    def _handle_update_download_progress(self, percent: int) -> None:
        """刷新更新下载进度。"""
        value = max(0, min(int(percent), 100))
        self.update_button.setText(f"下载 {value}%")

    def _handle_update_download_finished(self, path: object) -> None:
        """下载完成后运行安装器。"""
        installer_path = Path(path)
        self._append_log(f"更新包已就绪：{installer_path}")
        button = QMessageBox.question(
            self,
            "更新包已就绪",
            "安装器已下载完成。是否现在运行安装器并关闭当前程序？",
        )
        if button == QMessageBox.Yes:
            if QProcess.startDetached(str(installer_path), []):
                QApplication.quit()
            else:
                QMessageBox.warning(self, "启动失败", f"无法启动安装器：{installer_path}")

    def _handle_update_failed(self, message: str) -> None:
        """展示更新失败原因。"""
        QMessageBox.warning(self, "检查更新失败", message)
        self._append_log(f"检查更新失败：{message}")

    def closeEvent(self, event) -> None:
        """关闭主窗口前停止 PyVista 渲染并释放 VTK/OpenGL 资源。"""
        for canvas_name in ("surface_canvas", "h_prime_surface_canvas", "comparison_canvas"):
            canvas = getattr(self, canvas_name, None)
            if canvas is not None and hasattr(canvas, "shutdown"):
                canvas.shutdown()
        for page_name in ("plane_analysis_page", "step_analysis_page"):
            page = getattr(self, page_name, None)
            if page is not None and hasattr(page, "shutdown"):
                page.shutdown()
        super().closeEvent(event)

    def _handle_finished(self, result: object) -> None:
        """
        分析完成后的 GUI 刷新入口。

        FDA_Antivib 只刷新 FDA 主结果图层，非 FDA 后处理图层不会进入主界面。
        """
        analysis_result = AnalysisResult.coerce(result)
        self._result = analysis_result
        self._analysis_cube = self._worker.last_cube if self._worker is not None else None
        self.result_content_stack.setCurrentWidget(self.tabs)
        self.status_label.setText("分析完成")
        self.export_text_button.setEnabled(True)
        self.export_figures_button.setEnabled(True)
        self.h_prime_canvas.draw_map(analysis_result.h_prime, "height_prime")
        self.h_canvas.draw_map(analysis_result.h, "Height")
        self.phi0_canvas.draw_map(self._convert_map_for_display("phi0", analysis_result.phi0), "Phase Profile (cycles)")
        if self._is_phase_gap_mode():
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
        if str(analysis_result.extras.get("phase_gap_method", "")).upper() == "GFDA":
            comparison_title = "h（斜率高度） vs h_prime（GFDA非均匀级次修正）"
        if self._is_phase_gap_mode():
            comparison_title = "h（斜率高度） vs h_prime（PhaseGap最终）"
        self.comparison_canvas.draw_comparison(
            analysis_result.h,
            analysis_result.h_prime,
            comparison_title,
        )
        self._set_preview_stage(4)
        self._update_diagnostic_summary(analysis_result)
        self._set_active_range_display(self._extract_active_range_pair(analysis_result.extras.get("active_range")))
        self._refresh_step_states()
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
        self.result_content_stack.setCurrentWidget(self.tabs)
        self.tabs.setCurrentWidget(self.k0_tab)
        self.status_label.setText("全局 K0 已就绪")
        self._set_preview_stage(2)
        self._refresh_step_states()
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
            self._pixel_window = PixelAnalysisWindow(self)
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
            self._fringe_profile_window = FringeProfileWindow(self)
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
        scan_export_labels = {
            "scan_positions_raw_um": "GFDA 软件位置",
            "scan_positions_used_um": "GFDA 分析使用位置",
            "scan_positions_monotone_um": "GFDA 单调投影位置",
            "scan_step_raw_um": "GFDA 软件步长",
            "scan_step_monotone_um": "GFDA 单调投影步长",
            "scan_step_reversal_mask": "GFDA 倒退标记",
            "scan_position_correction_um": "GFDA 单调修正量",
            "scan_step_confidence": "GFDA 步长置信度",
            "scan_phase_estimated_rad": "GFDA 估计相位",
            "scan_positions_estimated_raw_um": "GFDA 原始估计位置",
            "scan_positions_adaptive_correction_um": "GFDA 自适应修正量",
        }
        options.extend(
            (key, label)
            for key, label in scan_export_labels.items()
            if key in self._result.extras
        )
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
            "scan_positions_raw_um": "scan_positions_raw_um.txt",
            "scan_positions_used_um": "scan_positions_used_um.txt",
            "scan_positions_monotone_um": "scan_positions_monotone_um.txt",
            "scan_step_raw_um": "scan_step_raw_um.txt",
            "scan_step_monotone_um": "scan_step_monotone_um.txt",
            "scan_step_reversal_mask": "scan_step_reversal_mask.txt",
            "scan_position_correction_um": "scan_position_correction_um.txt",
            "scan_step_confidence": "scan_step_confidence.txt",
            "scan_phase_estimated_rad": "scan_phase_estimated_rad.txt",
            "scan_positions_estimated_raw_um": "scan_positions_estimated_raw_um.txt",
            "scan_positions_adaptive_correction_um": "scan_positions_adaptive_correction_um.txt",
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
        self._set_preview_stage(5)
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
        self._set_preview_stage(5)
        self._append_log(f"图片已导出：{', '.join(str(path) for path in paths.values())}")
