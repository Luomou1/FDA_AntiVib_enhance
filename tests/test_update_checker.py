from __future__ import annotations

from app.update_checker import _pick_release_asset, is_newer_version


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
