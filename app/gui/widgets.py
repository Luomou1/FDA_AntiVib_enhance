from __future__ import annotations

"""Guided Analysis Console 使用的轻量展示控件。"""

from PySide6.QtCore import QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


def _refresh_style(widget: QWidget) -> None:
    """动态属性变化后立即刷新当前控件样式。"""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def _draw_nav_pixmap(kind: str, color: str) -> QPixmap:
    """绘制统一的高分屏线性导航图标，避免系统图标在不同平台上风格漂移。"""
    pixmap = QPixmap(40, 40)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(QColor(color), 2.4, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)

    if kind == "workbench":
        for rect in (
            QRectF(5, 5, 12, 12),
            QRectF(23, 5, 12, 12),
            QRectF(5, 23, 12, 12),
            QRectF(23, 23, 12, 12),
        ):
            painter.drawRoundedRect(rect, 2.5, 2.5)
    elif kind == "logs":
        for y in (9.0, 20.0, 31.0):
            painter.drawEllipse(QRectF(5, y - 1.5, 3, 3))
            painter.drawLine(13, int(y), 35, int(y))
    elif kind == "settings":
        knob_positions = (14.0, 27.0, 19.0)
        for y, knob_x in zip((9.0, 20.0, 31.0), knob_positions, strict=True):
            painter.drawLine(5, int(y), 35, int(y))
            painter.setBrush(QColor(color))
            painter.drawEllipse(QRectF(knob_x - 3, y - 3, 6, 6))
            painter.setBrush(Qt.NoBrush)
    else:
        raise ValueError(f"未知导航图标类型：{kind}")

    painter.end()
    pixmap.setDevicePixelRatio(2.0)
    return pixmap


def build_nav_icon(kind: str) -> QIcon:
    """构建带正常、悬停和选中状态的导航图标。"""
    icon = QIcon()
    normal = _draw_nav_pixmap(kind, "#a7afbb")
    active = _draw_nav_pixmap(kind, "#ffffff")
    disabled = _draw_nav_pixmap(kind, "#68717e")
    icon.addPixmap(normal, QIcon.Normal, QIcon.Off)
    icon.addPixmap(active, QIcon.Active, QIcon.Off)
    icon.addPixmap(active, QIcon.Normal, QIcon.On)
    icon.addPixmap(disabled, QIcon.Disabled, QIcon.Off)
    return icon


class NavButton(QPushButton):
    """全局一级导航按钮。"""

    def __init__(self, icon: QIcon, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIcon(icon)
        self.setIconSize(QSize(20, 20))
        self.setCheckable(True)
        self.setToolTip(title)
        self.setAccessibleName(title)
        self.setProperty("navRole", "global")
        self.setCursor(Qt.PointingHandCursor)


class StepButton(QPushButton):
    """分析步骤按钮，使用动态属性表达完成和错误状态。"""

    def __init__(self, number: str, title: str, parent: QWidget | None = None) -> None:
        super().__init__(f"{number}  {title}", parent)
        self.setFlat(True)
        self.setCheckable(True)
        self.setAccessibleName(title)
        self.setProperty("navRole", "step")
        self.setProperty("stepState", "idle")
        self.setCursor(Qt.PointingHandCursor)

    def set_step_state(self, state: str) -> None:
        self.setProperty("stepState", state)
        _refresh_style(self)


class StatusPill(QLabel):
    """统一任务状态标签。"""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setProperty("status", "idle")
        self.setAlignment(Qt.AlignCenter)

    def set_status(self, status: str, text: str | None = None) -> None:
        self.setProperty("status", status)
        if text is not None:
            self.setText(text)
        _refresh_style(self)


class SectionHeader(QWidget):
    """参数页标题与说明。"""

    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")
        description_label = QLabel(description)
        description_label.setObjectName("MutedLabel")
        description_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(description_label)
