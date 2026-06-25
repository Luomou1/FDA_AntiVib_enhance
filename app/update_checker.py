from __future__ import annotations

"""GitHub Releases 在线更新检查与下载。"""

import json
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app import APP_NAME

RELEASE_API_URL = "https://api.github.com/repos/Luomou1/FDA_AntiVib_enhance/releases/latest"
USER_AGENT = "FDA-AntiVib-enhance-updater"


@dataclass(frozen=True)
class UpdateInfo:
    """最新版本信息。"""

    current_version: str
    latest_version: str
    release_url: str
    release_name: str
    asset_name: str | None
    download_url: str | None
    asset_size: int | None
    update_available: bool
    cached_path: Path | None = None


def _request_json(url: str, timeout: float = 12.0) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _version_parts(version: str) -> tuple[int, ...]:
    """提取版本数字，支持 `v1.2.3`、`1.2.3-beta` 等常见 tag。"""
    numbers = re.findall(r"\d+", version)
    if not numbers:
        return (0,)
    return tuple(int(part) for part in numbers[:4])


def is_newer_version(latest: str, current: str) -> bool:
    latest_parts = _version_parts(latest)
    current_parts = _version_parts(current)
    length = max(len(latest_parts), len(current_parts))
    return latest_parts + (0,) * (length - len(latest_parts)) > current_parts + (0,) * (length - len(current_parts))


def _pick_release_asset(assets: list[dict]) -> dict | None:
    candidates = [
        asset
        for asset in assets
        if str(asset.get("name", "")).lower().endswith((".exe", ".msi", ".zip"))
        and asset.get("browser_download_url")
    ]
    if not candidates:
        return None

    def score(asset: dict) -> tuple[int, int]:
        name = str(asset.get("name", "")).lower()
        installer_score = 2 if any(token in name for token in ("setup", "installer", "install", "安装")) else 0
        exe_score = 1 if name.endswith(".exe") else 0
        return (installer_score, exe_score)

    return max(candidates, key=score)


def update_cache_dir() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    root = Path(local_app_data) if local_app_data else Path(tempfile.gettempdir())
    path = root / APP_NAME / "updates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_latest_update(current_version: str) -> UpdateInfo:
    """查询 GitHub 最新 Release，并返回是否需要更新。"""
    try:
        release = _request_json(RELEASE_API_URL)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError("未找到公开 Release；若仓库是私有仓库，在线更新需要公开 Release 或配置公开更新源。") from exc
        raise RuntimeError(f"检查更新失败：HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"检查更新失败：{exc.reason}") from exc

    latest_version = str(release.get("tag_name") or release.get("name") or "").strip()
    if not latest_version:
        raise RuntimeError("最新 Release 缺少版本号。")

    asset = _pick_release_asset(list(release.get("assets") or []))
    asset_name = str(asset.get("name")) if asset else None
    asset_size = int(asset["size"]) if asset and asset.get("size") is not None else None
    cached_path = None
    if asset_name and asset_size is not None:
        candidate = update_cache_dir() / asset_name
        if candidate.exists() and candidate.stat().st_size == asset_size:
            cached_path = candidate

    return UpdateInfo(
        current_version=current_version,
        latest_version=latest_version,
        release_url=str(release.get("html_url") or ""),
        release_name=str(release.get("name") or latest_version),
        asset_name=asset_name,
        download_url=str(asset.get("browser_download_url")) if asset else None,
        asset_size=asset_size,
        update_available=is_newer_version(latest_version, current_version),
        cached_path=cached_path,
    )


def download_update(info: UpdateInfo, progress_callback: Callable[[int], None] | None = None) -> Path:
    """下载更新包；若缓存文件大小匹配则直接复用。"""
    if info.cached_path is not None and info.cached_path.exists():
        if progress_callback is not None:
            progress_callback(100)
        return info.cached_path
    if not info.download_url or not info.asset_name:
        raise RuntimeError("当前 Release 没有可下载的安装包。")

    target = update_cache_dir() / info.asset_name
    part = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(info.download_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, part.open("wb") as output:
            total = int(response.headers.get("Content-Length") or info.asset_size or 0)
            downloaded = 0
            while True:
                chunk = response.read(1024 * 512)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                if total and progress_callback is not None:
                    progress_callback(min(99, int(downloaded * 100 / total)))
        if info.asset_size is not None and part.stat().st_size != info.asset_size:
            raise RuntimeError("安装包下载不完整，请稍后重试。")
        shutil.move(str(part), str(target))
        if progress_callback is not None:
            progress_callback(100)
        return target
    finally:
        if part.exists():
            part.unlink(missing_ok=True)
