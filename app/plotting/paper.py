from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np

from app.core.result_model import AnalysisResult

_DEFAULT_PAPER_EXPORT_KEYS = ("heightMap_prime_2d", "h_prime_3d")
TAU = 2.0 * np.pi


def _apply_paper_style() -> None:
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _phase_to_cycles(values: np.ndarray) -> np.ndarray:
    """把相位量纲数组从 rad 统一换算到 cycles。"""
    return (np.asarray(values, dtype=np.float32) / TAU).astype(np.float32)


def _h_prime_title_suffix(result: AnalysisResult) -> str:
    """根据当前工作流决定 h_prime 图题后缀。"""
    method = str(result.extras.get("phase_gap_method", "FDA"))
    return "FDA级次修正" if method.upper() == "FDA" else "PhaseGap最终"


def save_paper_figures(
    result: AnalysisResult | Mapping[str, Any],
    output_dir: Path,
    selected_keys: list[str] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _apply_paper_style()
    analysis_result = AnalysisResult.coerce(result)
    h_prime_suffix = _h_prime_title_suffix(analysis_result)
    figure_paths: dict[str, Path] = {}
    keys_to_export = set(_DEFAULT_PAPER_EXPORT_KEYS if selected_keys is None else selected_keys)

    if "h_2d" in keys_to_export:
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=240)
        image = ax.imshow(analysis_result.h, cmap="viridis", origin="lower", aspect="auto")
        ax.set_title("h 二维图（斜率高度）")
        ax.set_xlabel("X（像素）")
        ax.set_ylabel("Y（像素）")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        path_2d = output_dir / "h_2d.png"
        fig.savefig(path_2d, bbox_inches="tight")
        plt.close(fig)
        figure_paths["h_2d"] = path_2d

    if "h_prime_2d" in keys_to_export:
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=240)
        image = ax.imshow(analysis_result.h_prime, cmap="viridis", origin="lower", aspect="auto")
        ax.set_title(f"h_prime 二维图（{h_prime_suffix}）")
        ax.set_xlabel("X（像素）")
        ax.set_ylabel("Y（像素）")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        path_2d = output_dir / "h_prime_2d.png"
        fig.savefig(path_2d, bbox_inches="tight")
        plt.close(fig)
        figure_paths["h_prime_2d"] = path_2d

    if "heightMap_prime_2d" in keys_to_export:
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=240)
        image = ax.imshow(analysis_result.h_prime, cmap="viridis", origin="lower", aspect="auto")
        ax.set_title(f"h_prime 二维图（{h_prime_suffix}）")
        ax.set_xlabel("X（像素）")
        ax.set_ylabel("Y（像素）")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        path_2d = output_dir / "heightMap_prime_2d.png"
        fig.savefig(path_2d, bbox_inches="tight")
        plt.close(fig)
        figure_paths["heightMap_prime_2d"] = path_2d

    if "phi0_2d" in keys_to_export:
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=240)
        image = ax.imshow(_phase_to_cycles(analysis_result.phi0), cmap="viridis", origin="lower", aspect="auto")
        ax.set_title("Phase Profile 2D (cycles)")
        ax.set_xlabel("X（像素）")
        ax.set_ylabel("Y（像素）")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Phase (cycles)")
        path_2d = output_dir / "phi0_2d.png"
        fig.savefig(path_2d, bbox_inches="tight")
        plt.close(fig)
        figure_paths["phi0_2d"] = path_2d

    if {"h_prime_3d", "comparison_3d"} & keys_to_export:
        fig = plt.figure(figsize=(7.0, 5.2), dpi=240)
        ax = fig.add_subplot(111, projection="3d")
        y_coords = np.arange(analysis_result.h.shape[0])
        x_coords = np.arange(analysis_result.h.shape[1])
        xx, yy = np.meshgrid(x_coords, y_coords)
        ax.plot_surface(xx, yy, analysis_result.h_prime, cmap="viridis", linewidth=0, antialiased=True)
        ax.set_title(f"h_prime 三维图（{h_prime_suffix}）")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("高度（nm）")
        path_3d = output_dir / "h_prime_3d.png"
        fig.savefig(path_3d, bbox_inches="tight")
        plt.close(fig)
        figure_paths["h_prime_3d"] = path_3d
        if "comparison_3d" in keys_to_export:
            figure_paths["comparison_3d"] = path_3d

    return figure_paths
