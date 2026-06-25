from __future__ import annotations

from PySide6.QtWidgets import QWidget

import app.plotting.preview as preview_module


class _FakeTimer:
    def __init__(self) -> None:
        self.stop_calls = 0

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeCamera:
    def zoom(self, value: float) -> None:
        self.zoom_value = value


class _FakeQtInteractor(QWidget):
    def __init__(self, parent=None, auto_update=True) -> None:
        super().__init__(parent)
        self.interactor = self
        self.auto_update = auto_update
        self.render_timer = _FakeTimer()
        self.camera = _FakeCamera()
        self.render_window = object()
        self.suppress_rendering = False
        self._closed = False
        self.render_calls = 0
        self.close_calls = 0

    def render(self) -> None:
        self.render_calls += 1

    def view_isometric(self) -> None:
        return None

    def reset_camera(self, **kwargs) -> None:
        return None

    def close(self) -> None:
        self.close_calls += 1
        self._closed = True
        self.render_window = None


def test_pyvista_preview_disables_timer_and_shutdown_is_idempotent(qtbot, monkeypatch) -> None:
    monkeypatch.setattr(preview_module, "QtInteractor", _FakeQtInteractor)
    canvas = preview_module.SurfaceCanvas()
    qtbot.addWidget(canvas)

    assert canvas.plotter.auto_update is False
    canvas._render_if_visible()
    assert canvas.plotter.render_calls == 0

    canvas.show()
    assert canvas.plotter.render_calls == 1

    canvas.shutdown()
    canvas.shutdown()

    assert canvas.plotter.render_timer.stop_calls == 1
    assert canvas.plotter.close_calls == 1
