# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


ROOT = Path(SPECPATH).resolve().parent


hiddenimports = [
    "finufft",
    "matplotlib.backends.backend_qtagg",
    "pyvistaqt",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[],
    binaries=[],
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
