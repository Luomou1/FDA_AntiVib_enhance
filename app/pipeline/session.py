from __future__ import annotations
"""分析会话与导出模块。

定义：
- AnalysisParams：一次分析任务的参数快照
- AnalysisSession：基于参数执行结果导出
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from app.core.result_model import AnalysisResult


@dataclass(slots=True)
class AnalysisParams:
    """分析参数数据结构，供 GUI 到 worker 的参数传递使用。"""

    folder: Path
    start_height: float
    step_size: float
    fixed_k0: float
    window_size: int
    fitting_method: str
    unwrap_method: str
    window_name: str = "hamming"
    window_alpha: float = 0.5
    zero_padding_mode: str = "next_power_of_two"
    data_source: str = "image_folder"
    mat_path: Path | None = None
    phase_gap_method: str = "FDA"
    sampling_mode: str = "uniform"
    scan_log_path: Path | None = None
    sample_positions_um: np.ndarray | None = None
    active_range: tuple[int, int] | None = None
    expand_active_range: bool = False
    active_range_expansion_frames: int = 35


@dataclass(slots=True)
class AnalysisSession:
    """分析会话对象，封装导出策略。"""

    params: AnalysisParams

    def export_text_results(
        self,
        result: AnalysisResult | Mapping[str, Any],
        output_dir: Path,
        selected_keys: list[str] | None = None,
        target_paths: Mapping[str, Path] | None = None,
    ) -> dict[str, Path]:
        """把核心结果按约定键导出为制表符分隔文本。"""
        analysis_result = AnalysisResult.coerce(result)
        mapping = {
            "h": "h.txt",
            "h_prime": "h_prime.txt",
            "phi0": "phi0.txt",
        }
        alias = {
            "heightMap": "h",
            "heightMap_prime": "h_prime",
            "phi0_map": "phi0",
        }
        output_paths: dict[str, Path] = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        # 若未指定导出键，则默认导出三个主结果图层。
        requested_keys = list(mapping.keys()) if selected_keys is None else list(selected_keys)
        exports = analysis_result.text_exports()
        for requested_key in requested_keys:
            key = alias.get(requested_key, requested_key)
            if key not in mapping:
                raise KeyError(f"Unsupported export key: {requested_key}")
            # 允许 GUI 在导出前逐个收集用户自定义文件名；若未提供，就退回默认命名。
            if target_paths is not None and requested_key in target_paths:
                path = Path(target_paths[requested_key])
            elif target_paths is not None and key in target_paths:
                path = Path(target_paths[key])
            else:
                filename = mapping[key]
                path = output_dir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(path, exports[key], delimiter="\t")
            output_paths[requested_key] = path
        return output_paths
