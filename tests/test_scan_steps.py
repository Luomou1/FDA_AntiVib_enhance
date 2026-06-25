from __future__ import annotations

import numpy as np
import pytest

from app.core.scan_steps import build_scan_step_result, strict_monotone_projection_um
from app.pipeline.scan_log import load_actual_positions_um


def test_scan_log_rejects_local_reversal(tmp_path) -> None:
    path = tmp_path / "scan_log.txt"
    path.write_text(
        "index,target,actual\n"
        "1,0.00,0.00\n"
        "2,0.05,0.05\n"
        "3,0.10,0.04\n"
        "4,0.15,0.16\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strictly increasing"):
        load_actual_positions_um(path)


def test_strict_monotone_projection_repairs_reversal_with_small_correction() -> None:
    raw = np.array([0.0, 0.05, 0.04, 0.16], dtype=np.float32)

    projected = strict_monotone_projection_um(raw, nominal_step_um=0.05)

    assert np.all(np.diff(projected) > 0.0)
    np.testing.assert_allclose(projected[[0, -1]], np.array([0.0, 0.16], dtype=np.float32))


def test_scan_step_result_keeps_raw_positions_for_gfda_and_exports_diagnostics() -> None:
    raw = np.array([0.0, 0.05, 0.04, 0.16], dtype=np.float32)

    result = build_scan_step_result(raw, nominal_step_um=0.05)
    mapping = result.to_mapping()

    np.testing.assert_allclose(result.analysis_positions_um, raw)
    np.testing.assert_allclose(
        mapping["scan_step_raw_um"],
        np.array([0.05, -0.01, 0.12], dtype=np.float32),
        atol=1e-8,
    )
    np.testing.assert_allclose(mapping["scan_step_reversal_mask"], np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert result.reversal_count == 1
    assert mapping["scan_strategy"] == "software_nonuniform_gfda"
