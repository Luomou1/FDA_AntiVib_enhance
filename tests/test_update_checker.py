from __future__ import annotations

from app.update_checker import _pick_release_asset, cleanup_cached_installers, is_newer_version


def test_is_newer_version_handles_v_prefix_and_suffix() -> None:
    assert is_newer_version("v0.1.1", "0.1.0") is True
    assert is_newer_version("v0.1.0", "0.1.0") is False
    assert is_newer_version("v0.1.0-beta", "0.1.1") is False


def test_pick_release_asset_prefers_installer_exe() -> None:
    asset = _pick_release_asset(
        [
            {"name": "source.zip", "browser_download_url": "https://example.test/source.zip"},
            {"name": "数据分析-0.1.1-setup.exe", "browser_download_url": "https://example.test/setup.exe"},
            {"name": "数据分析.zip", "browser_download_url": "https://example.test/app.zip"},
        ]
    )

    assert asset is not None
    assert asset["name"] == "数据分析-0.1.1-setup.exe"


def test_cleanup_cached_installers_removes_all_app_installers(tmp_path) -> None:
    current = tmp_path / "DataAnalysis-0.1.0-setup.exe"
    old = tmp_path / "DataAnalysis-0.0.9-setup.exe"
    partial = tmp_path / "数据分析-0.1.0-setup.exe.part"
    unrelated = tmp_path / "other-tool-0.0.1-setup.exe"
    for path in (current, old, partial, unrelated):
        path.write_text("placeholder", encoding="utf-8")

    removed = cleanup_cached_installers(roots=[tmp_path])

    assert unrelated.exists()
    assert not current.exists()
    assert not old.exists()
    assert not partial.exists()
    assert {path.name for path in removed} == {current.name, old.name, partial.name}
