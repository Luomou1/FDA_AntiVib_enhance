"""MetroPro FDA 中心化及峰值双向展开规则，输出恢复为原扫描参考。"""

import numpy as np
from numpy.typing import NDArray


def unwrap_metro(
    phase: NDArray[np.floating],
    amplitude: NDArray[np.floating],
    bin_indices: NDArray[np.integer],
) -> NDArray[np.float64]:
    """在指定连续 FFT 频点窗口中展开；非正幅值阻断该侧并标为 NaN。

    未公开的 MetroPro 噪声阈值不作猜测；这里仅排除非正/非有限幅值。
    使用真实频点编号而非窗口内序号，兼容奇数 FFT 长度及边界重复点。
    """
    raw = np.asarray(phase, dtype=np.float64)
    amps = np.asarray(amplitude, dtype=np.float64)
    bins = np.asarray(bin_indices, dtype=np.int64)
    if raw.ndim not in (1, 2) or raw.shape != amps.shape or bins.shape != (raw.shape[-1],):
        raise ValueError("Metro phase/amplitude shapes and frequency indices must agree.")
    single = raw.ndim == 1
    raw = np.atleast_2d(raw)
    amps = np.atleast_2d(amps)
    valid = np.isfinite(raw) & np.isfinite(amps) & (amps > 0)
    # 奇数频点乘 -1，等价于增加 pi*j 的相位，移除半记录长度的延迟。
    centered = np.angle(np.exp(1j * raw) * np.where(bins % 2, -1.0, 1.0))
    peaks = np.argmax(np.where(valid, amps, -np.inf), axis=1)
    result = np.full_like(raw, np.nan)
    rows = np.arange(raw.shape[0])
    result[rows, peaks] = np.where(valid[rows, peaks], centered[rows, peaks], np.nan)
    for direction in (-1, 1):
        alive = valid[rows, peaks].copy()
        for distance in range(1, raw.shape[1]):
            columns = peaks + direction * distance
            inside = (columns >= 0) & (columns < raw.shape[1])
            selected = rows[alive & inside]
            target = columns[selected]
            good = valid[selected, target]
            alive[selected[~good]] = False
            selected, target = selected[good], target[good]
            previous = result[selected, target - direction]
            current = centered[selected, target]
            # 最近整数校正允许累计多个周期，与逐步 ±pi 连续化的分支一致。
            delta = current - previous
            turns = np.where(delta > np.pi, np.ceil((delta - np.pi) / (2 * np.pi)),
                             np.where(delta < -np.pi, np.floor((delta + np.pi) / (2 * np.pi)), 0.0))
            result[selected, target] = current - 2 * np.pi * turns
    # 后续 FDA 公式仍以原扫描起点为参考，必须恢复已知线性相位。
    result -= np.pi * bins
    return result[0] if single else result
