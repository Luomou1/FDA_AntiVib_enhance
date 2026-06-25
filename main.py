from __future__ import annotations
"""应用主入口。

职责很简单：
- 创建 Qt 应用对象
- 初始化主窗口
- 启动事件循环
"""

import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from app import APP_NAME, __version__
from app.runtime_paths import resource_path
from app.update_checker import cleanup_cached_installers


def main(argv: list[str] | None = None) -> int:
    """创建并启动 GUI，返回 Qt 事件循环退出码。"""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--smoke-test-imports" in args:
        from app.gui.main_window import MainWindow

        _ = MainWindow
        return 0

    cleanup_cached_installers(__version__)

    from app.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon(str(resource_path("assets/app_icon.ico"))))
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
