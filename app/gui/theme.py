from __future__ import annotations

"""应用级主题令牌与 QSS 生成。"""

from typing import Literal

from PySide6.QtCore import Qt

ThemeMode = Literal["system", "light", "dark"]
ResolvedTheme = Literal["light", "dark"]


_PALETTES: dict[ResolvedTheme, dict[str, str]] = {
    "light": {
        "app": "#e7e9ec",
        "surface": "#f7f7f8",
        "surface_alt": "#ededf0",
        "surface_muted": "#e2e5e9",
        "chrome": "#303237",
        "chrome_alt": "#3b3e45",
        "chrome_deep": "#25272c",
        "rail": "#f2f3f5",
        "rail_hover": "#e8ebef",
        "canvas": "#fbfaf7",
        "canvas_text": "#111418",
        "canvas_muted": "#666e7a",
        "text": "#111418",
        "text_muted": "#56606d",
        "text_faint": "#929aa6",
        "chrome_text": "#f6f7f8",
        "border": "#b8bec8",
        "border_strong": "#8f98a6",
        "primary": "#1d6f9f",
        "primary_hover": "#155b84",
        "primary_soft": "#dce8f1",
        "accent": "#236b63",
        "success": "#237044",
        "success_soft": "#e2eee8",
        "danger": "#b91c1c",
        "danger_soft": "#fdecec",
        "input": "#fbfbfc",
        "scroll": "#a9b0ba",
    },
    "dark": {
        "app": "#111827",
        "surface": "#1f2937",
        "surface_alt": "#273244",
        "surface_muted": "#172033",
        "chrome": "#111827",
        "chrome_alt": "#1f2937",
        "chrome_deep": "#111827",
        "rail": "#111827",
        "rail_hover": "#273244",
        "canvas": "#f8fafc",
        "canvas_text": "#111827",
        "canvas_muted": "#64748b",
        "text": "#f8fafc",
        "text_muted": "#cbd5e1",
        "text_faint": "#94a3b8",
        "chrome_text": "#ffffff",
        "border": "#334155",
        "border_strong": "#475569",
        "primary": "#3b82f6",
        "primary_hover": "#60a5fa",
        "primary_soft": "#1e3a5f",
        "accent": "#2dd4bf",
        "success": "#86efac",
        "success_soft": "#123524",
        "danger": "#fca5a5",
        "danger_soft": "#471d1d",
        "input": "#111827",
        "scroll": "#64748b",
    },
}


def resolve_theme_mode(
    mode: ThemeMode,
    color_scheme: Qt.ColorScheme | None = None,
) -> ResolvedTheme:
    """解析主题模式；无法识别系统主题时稳定回退到浅色。"""
    if mode == "light":
        return "light"
    if mode == "dark":
        return "dark"
    if color_scheme == Qt.ColorScheme.Dark:
        return "dark"
    return "light"


def build_stylesheet(mode: ResolvedTheme) -> str:
    """按调色板生成整套桌面工作台样式。"""
    colors = _PALETTES[mode]
    return f"""
    QWidget {{
        background: transparent;
        color: {colors["text"]};
        font-family: "Microsoft YaHei UI", "Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "Segoe UI Variable", "Segoe UI", sans-serif;
        font-size: 12px;
    }}
    QMainWindow {{
        background: {colors["app"]};
    }}
    QFrame#TopCommandBar {{
        background: {colors["chrome_deep"]};
        border: none;
        border-bottom: 1px solid {colors["border"]};
    }}
    QFrame#ResultWorkspace, QFrame#ParameterPanel,
    QFrame#BottomStatusBar, QFrame#SettingsCard,
    QFrame#AnalysisPage, QFrame#AnalysisControlPanel {{
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-radius: 1px;
    }}
    QFrame#AnalysisPage {{
        border: none;
    }}
    QFrame#AnalysisControlPanel {{
        background: {colors["surface"]};
    }}
    QFrame#ResultWorkspace {{
        background: {colors["canvas"]};
    }}
    QFrame#ResultWorkspace QLabel {{
        color: {colors["canvas_text"]};
    }}
    QFrame#ResultWorkspace QLabel#MutedLabel,
    QFrame#ResultWorkspace QLabel#LatestLogLabel {{
        color: {colors["canvas_muted"]};
    }}
    QFrame#ParameterPanel, QFrame#StepRail, QFrame#BottomStatusBar {{
        background: {colors["surface"]};
    }}
    QDialog, QMessageBox, QFileDialog {{
        color: {colors["text"]};
        background: {colors["surface"]};
    }}
    QDialog QLabel, QMessageBox QLabel, QFileDialog QLabel {{
        color: {colors["text"]};
        background: transparent;
    }}
    QDialogButtonBox {{
        background: transparent;
    }}
    QFrame#StepRail {{
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-radius: 1px;
    }}
    QLabel#MenuText, QLabel#BrandText {{
        color: {colors["chrome_text"]};
        background: transparent;
    }}
    QLabel#BrandText {{
        font-size: 14px;
        font-weight: 700;
    }}
    QLabel#AppTitle {{
        color: {colors["text"]};
        font-size: 17px;
        font-weight: 700;
    }}
    QLabel#PageTitle {{
        color: {colors["text"]};
        font-size: 14px;
        font-weight: 700;
    }}
    QLabel#SectionTitle {{
        color: {colors["text"]};
        font-size: 13px;
        font-weight: 700;
    }}
    QLabel#EmptyStateTitle {{
        color: {colors["text"]};
        font-size: 16px;
        font-weight: 700;
    }}
    QLabel#MutedLabel, QLabel#LatestLogLabel {{
        color: {colors["text_muted"]};
    }}
    QLabel#AnalysisStatus {{
        padding: 8px;
        color: {colors["text_muted"]};
        background: {colors["surface_alt"]};
        border: 1px solid {colors["border"]};
        border-radius: 1px;
    }}
    QPushButton {{
        min-height: 26px;
        padding: 0 9px;
        background: {colors["surface_muted"]};
        color: {colors["text"]};
        border: 1px solid {colors["border_strong"]};
        border-radius: 1px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {colors["surface_alt"]};
        border-color: {colors["primary"]};
    }}
    QPushButton:pressed {{
        background: {colors["primary_soft"]};
    }}
    QPushButton:disabled {{
        color: {colors["text_faint"]};
        background: {colors["surface_alt"]};
        border-color: {colors["border"]};
    }}
    QPushButton#PrimaryButton {{
        min-height: 30px;
        background: {colors["primary"]};
        color: #ffffff;
        border-color: {colors["primary"]};
    }}
    QPushButton#PrimaryButton:hover {{
        background: {colors["primary_hover"]};
        border-color: {colors["primary_hover"]};
    }}
    QPushButton#SecondaryButton {{
        min-height: 30px;
        background: {colors["surface_alt"]};
        color: {colors["text"]};
        border-color: {colors["primary"]};
    }}
    QPushButton#TopNavButton {{
        min-height: 26px;
        padding: 0 12px;
        color: {colors["chrome_text"]};
        background: transparent;
        border: 1px solid transparent;
        border-radius: 1px;
    }}
    QPushButton#TopNavButton:hover {{
        background: {colors["chrome_alt"]};
        border-color: {colors["chrome_alt"]};
    }}
    QPushButton#TopNavButton:checked {{
        background: {colors["chrome_alt"]};
        color: {colors["chrome_text"]};
        border-color: {colors["border_strong"]};
    }}
    QPushButton#ExportButton {{
        background: {colors["surface_muted"]};
        color: {colors["accent"]};
        border-color: {colors["accent"]};
    }}
    QPushButton[navRole="step"] {{
        min-height: 42px;
        padding: 4px 9px;
        text-align: left;
        color: {colors["text_muted"]};
        background: transparent;
        border: none;
        border-left: 3px solid {colors["border_strong"]};
        border-radius: 0;
        font-weight: 600;
    }}
    QPushButton[navRole="step"]:hover {{
        color: {colors["text"]};
        background: transparent;
    }}
    QPushButton[navRole="step"]:checked {{
        color: {colors["primary"]};
        background: transparent;
        border-left-color: {colors["primary"]};
    }}
    QPushButton[stepState="complete"] {{
        border-left-color: {colors["success"]};
    }}
    QPushButton[stepState="error"] {{
        color: {colors["danger"]};
        border-left-color: {colors["danger"]};
    }}
    QLabel[status="idle"], QLabel[status="running"], QLabel[status="success"],
    QLabel[status="error"] {{
        padding: 3px 8px;
        border-radius: 1px;
        font-weight: 700;
    }}
    QLabel[status="idle"] {{
        color: {colors["text_muted"]};
        background: {colors["surface_alt"]};
    }}
    QLabel[status="running"] {{
        color: {colors["primary"]};
        background: {colors["primary_soft"]};
    }}
    QLabel[status="success"] {{
        color: {colors["success"]};
        background: {colors["success_soft"]};
    }}
    QLabel[status="error"] {{
        color: {colors["danger"]};
        background: {colors["danger_soft"]};
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
        min-height: 27px;
        padding: 0 8px;
        color: {colors["text"]};
        background: {colors["input"]};
        border: 1px solid {colors["border_strong"]};
        border-radius: 1px;
        selection-background-color: {colors["primary"]};
    }}
    QPlainTextEdit {{
        padding: 8px;
        font-family: "Cascadia Mono", "Consolas", "Microsoft YaHei", monospace;
    }}
    QPlainTextEdit#RailLog {{
        min-height: 0;
        padding: 2px 0;
        background: transparent;
        border: none;
        font-size: 11px;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QPlainTextEdit:focus {{
        border: 1px solid {colors["primary"]};
    }}
    QComboBox::drop-down {{
        width: 24px;
        border: none;
    }}
    QComboBox QAbstractItemView {{
        color: {colors["text"]};
        background: {colors["surface"]};
        border: 1px solid {colors["border_strong"]};
        selection-color: #ffffff;
        selection-background-color: {colors["primary"]};
    }}
    QCheckBox {{
        spacing: 7px;
        color: {colors["text"]};
    }}
    QGroupBox {{
        margin-top: 11px;
        padding: 12px 9px 9px 9px;
        color: {colors["text"]};
        background: {colors["surface_muted"]};
        border: 1px solid {colors["border"]};
        border-radius: 1px;
        font-weight: 700;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 9px;
        padding: 0 4px;
    }}
    QTabWidget::pane {{
        top: -1px;
        background: {colors["surface"]};
        border: 1px solid {colors["border"]};
        border-radius: 1px;
    }}
    QTabBar::tab {{
        min-height: 28px;
        padding: 0 11px;
        margin-right: 3px;
        color: {colors["text_muted"]};
        background: {colors["surface_alt"]};
        border: 1px solid {colors["border"]};
        border-bottom: none;
        border-top-left-radius: 1px;
        border-top-right-radius: 1px;
    }}
    QTabBar::tab:selected {{
        color: {colors["primary"]};
        background: {colors["surface"]};
    }}
    QProgressBar {{
        min-height: 8px;
        max-height: 8px;
        color: transparent;
        background: {colors["surface_alt"]};
        border: none;
        border-radius: 4px;
    }}
    QProgressBar::chunk {{
        background: {colors["primary"]};
        border-radius: 2px;
    }}
    QScrollArea {{
        background: transparent;
        border: none;
    }}
    QScrollBar:vertical {{
        width: 9px;
        margin: 2px;
        background: transparent;
    }}
    QScrollBar::handle:vertical {{
        min-height: 28px;
        background: {colors["scroll"]};
        border-radius: 4px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QSplitter::handle {{
        background: transparent;
    }}
    QToolBar, QToolButton {{
        color: {colors["text"]};
        background: transparent;
        border: none;
    }}
    QToolButton:hover {{
        background: {colors["surface_alt"]};
        border-radius: 5px;
    }}
    QToolTip {{
        color: {colors["text"]};
        background: {colors["surface"]};
        border: 1px solid {colors["border_strong"]};
        padding: 5px;
    }}
    """
