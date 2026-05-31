from __future__ import annotations
"""应用主入口。

职责很简单：
- 创建 Qt 应用对象
- 初始化主窗口
- 启动事件循环
"""

from PySide6.QtWidgets import QApplication

from app.gui.main_window import MainWindow


def main() -> int:
    """创建并启动 GUI，返回 Qt 事件循环退出码。"""
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
