# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


ROOT = Path(SPECPATH).resolve().parent


hiddenimports = [
    "finufft",
    "matplotlib.backends.backend_qtagg",
    "pyvistaqt",
]

# Qt6Core 需要未版本化的 ICU 入口；Poppler 的 ICU 78 DLL 使用同名文件但只导出
# `icu_78` 版本化符号，不能让 PyInstaller 把它误收进 Qt 应用。
excluded_binaries = {"icuuc.dll", "icuin.dll", "icudt78.dll"}

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[],
    binaries=collect_dynamic_libs("finufft"),
    datas=[(str(ROOT / "assets" / "app_icon.ico"), "assets")],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "IPython",
        "jupyter",
        "matplotlib.tests",
        "numba",
        "pandas",
        "pytest",
        "torch",
    ],
    noarchive=False,
    optimize=0,
)
# 必须在依赖分析后过滤：冲突 DLL 是扫描 Qt6Core 的传递依赖时引入的。
# Qt 使用 Windows 自带的 ICU，不将构建机器 PATH 上的第三方 ICU 随包分发。
a.binaries = [item for item in a.binaries if Path(item[0]).name.lower() not in excluded_binaries]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="数据分析",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "app_icon.ico"),
)
