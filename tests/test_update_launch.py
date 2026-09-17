from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QMessageBox

import app.gui.main_window as main_window_module
from app.gui.main_window import MainWindow


@pytest.mark.parametrize("started", [True, False])
def test_update_launch_resets_frozen_environment_and_only_quits_on_success(monkeypatch, tmp_path, started):
    installer = tmp_path / "setup.exe"
    monkeypatch.setenv("_PYI_APPLICATION_HOME_DIR", str(tmp_path / "removed-extraction"))
    monkeypatch.setenv("UPDATE_TEST_SENTINEL", "preserved")
    launches = []
    quits = []
    warnings = []

    class RecordingProcess(QProcess):
        def startDetached(self):
            launches.append((self.program(), self.processEnvironment()))
            return started

    monkeypatch.setattr(main_window_module, "QProcess", RecordingProcess)
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(main_window_module.QApplication, "quit", lambda: quits.append(True))
    window = SimpleNamespace(_append_log=lambda message: None)

    MainWindow._handle_update_download_finished(window, installer)

    assert str(installer) == launches[0][0]
    environment = launches[0][1]
    assert environment is not None, "安装器必须使用独立的子进程环境"
    assert "1" == environment.value("PYINSTALLER_RESET_ENVIRONMENT")
    assert "preserved" == environment.value("UPDATE_TEST_SENTINEL")
    assert started == bool(quits)
    assert (not started) == bool(warnings)


def test_update_launch_missing_installer_keeps_application_open(monkeypatch, tmp_path):
    quits = []
    warnings = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.Yes)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args))
    monkeypatch.setattr(main_window_module.QApplication, "quit", lambda: quits.append(True))
    window = SimpleNamespace(_append_log=lambda message: None)

    MainWindow._handle_update_download_finished(window, tmp_path / "missing-installer.exe")

    assert not quits
    assert len(warnings) == 1
