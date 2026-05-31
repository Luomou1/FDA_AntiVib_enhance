from __future__ import annotations
"""
参考分析入口。

这个文件故意保持很薄：
- 不放算法细节
- 只把参数透传到正式主流程

这样测试或外部调用可以固定依赖一个稳定入口。
"""

import numpy as np

from app.core.kernel import _run_formal_analysis_pipeline


def analyze_cube_reference(
    intensity_data: np.ndarray,
    step_size: float,
    window_size: int,
    fitting_method: str,
    unwrap_method: str,
) -> dict[str, np.ndarray]:
    """以无进度输出模式调用正式分析管线，返回统一结果字典。"""
    return _run_formal_analysis_pipeline(
        intensity_data=intensity_data,
        step_size=step_size,
        window_size=window_size,
        fitting_method=fitting_method,
        unwrap_method=unwrap_method,
        show_progress=False,
    )
