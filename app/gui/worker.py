from __future__ import annotations
"""GUI 后台 worker 模块。

包含两个 QObject worker：
- AnalysisWorker：执行完整分析与导出
- GlobalK0Worker：执行全局 K0 自动估计

两者都在独立线程运行，通过信号把进度和结果回传给主线程 UI。
"""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from app.core.kernel import analyze_cube_fast, estimate_global_k0
from app.core.result_model import AnalysisResult
from app.core.software_scan_steps import estimate_software_scan_positions
from app.pipeline.active_range import apply_active_range, apply_known_active_range
from app.pipeline.io import collect_image_files, load_intensity_cube, load_mat_intensity_cube
from app.pipeline.scan_log import load_actual_positions_um
from app.pipeline.session import AnalysisParams, AnalysisSession
from app.plotting.paper import save_paper_figures

class AnalysisWorker(QObject):
    """分析任务 worker：负责加载数据、运行主算法、回传结果。"""

    progress = Signal(int)
    log = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, params: AnalysisParams) -> None:
        super().__init__()
        self.params = params
        self.last_cube: np.ndarray | None = None
        # 记录“分析阶段”最近一次已经发出的 GUI 进度。
        # 这样可以避免底层回调重复上报同一个百分比，导致进度条抖动或倒退。
        self._last_analysis_progress = 35

    def _is_gfda_software_workflow(self) -> bool:
        workflow = str(self.params.phase_gap_method).strip().lower()
        return workflow in {"gfda_carrier_phase", "gfda_scatter_fit"}

    def _resolve_sample_positions(self, sample_count: int) -> np.ndarray | None:
        """按采样模式解析并校验 sample positions。"""
        if self._is_gfda_software_workflow():
            self.params.sample_positions_um = None
            return None
        if self.params.sampling_mode != "nonuniform":
            return None
        if self.params.scan_log_path is None:
            raise ValueError("Non-uniform sampling requires a scan_log.txt path.")
        positions = load_actual_positions_um(self.params.scan_log_path)
        if positions.shape[0] != sample_count:
            raise ValueError(
                f"Sample position count ({positions.shape[0]}) does not match image count ({sample_count})."
            )
        self.params.sample_positions_um = positions
        return positions

    def _load_input_cube(self) -> np.ndarray:
        """按数据来源加载输入立方体，并输出对应日志。"""
        if self.params.data_source == "mat_file":
            if self.params.mat_path is None:
                raise ValueError("MAT data source requires a MAT file path.")
            self.progress.emit(10)
            self.log.emit(f"正在加载 MAT 数据: {self.params.mat_path.name}")
            cube = load_mat_intensity_cube(self.params.mat_path)
            self.log.emit(f"MAT 数据已加载，尺寸={cube.shape}")
            return cube

        self.log.emit("\u6b63\u5728\u6536\u96c6\u56fe\u50cf\u6587\u4ef6...")
        files = collect_image_files(self.params.folder)
        if len(files) < 1:
            raise ValueError("\u81f3\u5c11\u9700\u8981\u4e00\u5f20\u56fe\u50cf\u3002")

        self.progress.emit(10)
        self.log.emit(f"\u627e\u5230 {len(files)} \u5f20\u56fe\u50cf\uff0c\u6b63\u5728\u8f7d\u5165\u6570\u636e\u7acb\u65b9\u4f53...")
        return load_intensity_cube(files, image_intensity_mode=self.params.image_intensity_mode)

    def _emit_analysis_progress(self, percent: int) -> None:
        """
        把底层分析进度映射到 GUI 进度条。

        当前 worker 把整个流程大致分成三段：
        - 0 ~ 35：文件收集、数据加载、采样位置解析
        - 35 ~ 90：真正的数值分析主体
        - 90 ~ 100：结果封装、发信号、收尾

        底层 `analyze_cube_fast()` / `analyze_cube_baseline()` 的回调仍然使用 0~100。
        所以这里需要把它线性映射到 GUI 的 35~90 区间，进度条才会在分析过程中持续前进。
        """
        # 先把底层输入规整到 0~100，防止异常值把进度条拉出范围。
        clamped = max(0, min(int(percent), 100))
        # 线性映射：0 -> 35，100 -> 90。
        mapped = 35 + int(round(clamped * 55 / 100))
        # 为了保持进度条单调前进，只在新值更大时才发射信号。
        if mapped > self._last_analysis_progress:
            self._last_analysis_progress = mapped
            self.progress.emit(mapped)

    @Slot()
    def run(self) -> None:
        """worker 主执行函数，在子线程中被触发。"""
        try:
            # 每次新任务开始时，把“分析阶段最后进度”重置回分析起点。
            self._last_analysis_progress = 35
            self.progress.emit(0)
            cube = self._load_input_cube()
            sample_positions = self._resolve_sample_positions(cube.shape[2])
            # 自动 K0 阶段已经确认过有效范围时，正式分析直接复用同一段帧；
            # 只有用户跳过自动 K0 或数据源变化导致没有缓存时，才重新检测。
            if self.params.active_range is not None:
                cube, sample_positions, active_range = apply_known_active_range(
                    cube,
                    sample_positions,
                    self.params.active_range,
                )
            else:
                left_expansion_frames = self.params.active_range_left_expansion_frames if self.params.expand_active_range else 0
                right_expansion_frames = self.params.active_range_right_expansion_frames if self.params.expand_active_range else 0
                cube, sample_positions, active_range = apply_active_range(
                    cube,
                    sample_positions,
                    left_expansion_frames=left_expansion_frames,
                    right_expansion_frames=right_expansion_frames,
                )
            self.last_cube = cube
            self.params.active_range = (active_range.start_frame, active_range.end_frame)
            self.log.emit(
                f"有效扫描范围：{active_range.start_frame}-{active_range.end_frame}，"
                f"有效帧数={active_range.end_frame - active_range.start_frame + 1}，"
                f"来源={active_range.reason}"
            )
            scan_step_mapping = None
            analysis_method = self.params.phase_gap_method
            if self._is_gfda_software_workflow():
                if self.params.fixed_k0 is None:
                    raise ValueError("GFDA 软件步长估计需要先设置固定 K0，或先执行自动定 K0。")
                self.log.emit(f"正在估计 GFDA 软件步长：{self.params.phase_gap_method}")
                estimate = estimate_software_scan_positions(
                    cube,
                    method=self.params.phase_gap_method,
                    k0_value=float(self.params.fixed_k0),
                    nominal_step_um=self.params.step_size,
                    window_size=self.params.window_size,
                    fitting_method=self.params.fitting_method,
                    unwrap_method=self.params.unwrap_method,
                    window_name=self.params.window_name,
                    window_alpha=self.params.window_alpha,
                    zero_padding_mode=self.params.zero_padding_mode,
                )
                sample_positions = estimate.scan_steps.analysis_positions_um
                self.params.sample_positions_um = sample_positions
                scan_step_mapping = estimate.to_mapping()
                analysis_method = "GFDA"
                self.log.emit(
                    f"GFDA 软件步长完成：方法={estimate.method}，"
                    f"倒退步数={estimate.scan_steps.reversal_count}，"
                    f"策略={estimate.scan_steps.strategy}/{scan_step_mapping.get('scan_adaptive_strategy')}。"
                )
            elif sample_positions is not None:
                self.params.sample_positions_um = sample_positions
                self.log.emit(f"已加载 {sample_positions.shape[0]} 个 scan_log 非均匀采样位置。")

            self.progress.emit(35)
            self.log.emit(
                f"\u6b63\u5728\u5206\u6790\uff1a\u62df\u5408={self.params.fitting_method}\uff0c"
                f"\u89e3\u5305\u88f9={self.params.unwrap_method}\uff0c"
                f"工作流={analysis_method}\uff0c"
                f"\u7a97\u51fd\u6570={self.params.window_name}\uff0c"
                f"补零={self.params.zero_padding_mode}"
            )
            result = analyze_cube_fast(
                intensity_data=cube,
                step_size=self.params.step_size,
                window_size=self.params.window_size,
                fitting_method=self.params.fitting_method,
                unwrap_method=self.params.unwrap_method,
                fixed_k0_value=self.params.fixed_k0,
                sample_positions_um=sample_positions,
                phase_gap_method=analysis_method,
                window_name=self.params.window_name,
                window_alpha=self.params.window_alpha,
                zero_padding_mode=self.params.zero_padding_mode,
                progress_callback=self._emit_analysis_progress,
            )
            if scan_step_mapping is not None:
                result.update(scan_step_mapping)
            result["active_range"] = np.asarray([active_range.start_frame, active_range.end_frame], dtype=np.int32)
            result["active_ranges"] = np.asarray(active_range.ranges, dtype=np.int32)
            result["active_range_score"] = active_range.score.astype(np.float32)
            analysis_result = AnalysisResult.from_mapping(result)
            self.progress.emit(90)
            self.log.emit(f"分析完成，FFT长度={int(result.get('fft_length', cube.shape[2]))}。")
            self.finished.emit(analysis_result)
            self.progress.emit(100)
        except Exception as exc:
            self.failed.emit(str(exc))

    def export_text(self, result: AnalysisResult | dict[str, np.ndarray], output_dir: Path) -> dict[str, Path]:
        """导出默认文本结果集合。"""
        session = AnalysisSession(self.params)
        return session.export_text_results(result, output_dir)

    def export_text_selected(
        self,
        result: AnalysisResult | dict[str, np.ndarray],
        output_dir: Path,
        selected_keys: list[str],
    ) -> dict[str, Path]:
        """按指定键导出文本结果。"""
        session = AnalysisSession(self.params)
        return session.export_text_results(result, output_dir, selected_keys=selected_keys)

    def export_figures(self, result: AnalysisResult | dict[str, np.ndarray], output_dir: Path) -> dict[str, Path]:
        """导出默认图像结果集合。"""
        return save_paper_figures(result, output_dir)

    def export_figures_selected(
        self,
        result: AnalysisResult | dict[str, np.ndarray],
        output_dir: Path,
        selected_keys: list[str],
    ) -> dict[str, Path]:
        """按指定键导出图像结果。"""
        return save_paper_figures(result, output_dir, selected_keys=selected_keys)


class GlobalK0Worker(QObject):
    """全局 K0 自动估计 worker。"""

    progress = Signal(int)
    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str)

    def __init__(
        self,
        folder: Path,
        step_size: float,
        candidate_ratio: float,
        window_name: str,
        window_alpha: float,
        zero_padding_mode: str,
        expand_active_range: bool = False,
        active_range_left_expansion_frames: int = 35,
        active_range_right_expansion_frames: int = 35,
        data_source: str = "image_folder",
        image_intensity_mode: str = "mono12_uint16",
        mat_path: Path | None = None,
        sampling_mode: str = "uniform",
        scan_log_path: Path | None = None,
        phase_gap_method: str = "FDA",
        window_size: int = 2,
        fitting_method: str = "weighted",
        unwrap_method: str = "itoh",
    ) -> None:
        super().__init__()
        self.folder = folder
        self.step_size = step_size
        self.candidate_ratio = candidate_ratio
        self.window_name = window_name
        self.window_alpha = window_alpha
        self.zero_padding_mode = zero_padding_mode
        self.expand_active_range = bool(expand_active_range)
        self.active_range_left_expansion_frames = int(active_range_left_expansion_frames)
        self.active_range_right_expansion_frames = int(active_range_right_expansion_frames)
        self.data_source = data_source
        self.image_intensity_mode = image_intensity_mode
        self.mat_path = mat_path
        self.sampling_mode = sampling_mode
        self.scan_log_path = scan_log_path
        self.phase_gap_method = phase_gap_method
        self.window_size = int(window_size)
        self.fitting_method = fitting_method
        self.unwrap_method = unwrap_method
        self.last_cube: np.ndarray | None = None

    def _is_gfda_software_workflow(self) -> bool:
        workflow = str(self.phase_gap_method).strip().lower()
        return workflow in {"gfda_carrier_phase", "gfda_scatter_fit"}

    def _resolve_sample_positions(self, sample_count: int) -> np.ndarray | None:
        """在非均匀采样模式下加载并校验 scan_log 位移数据。"""
        if self._is_gfda_software_workflow():
            return None
        if self.sampling_mode != "nonuniform":
            return None
        if self.scan_log_path is None:
            raise ValueError("Non-uniform sampling requires a scan_log.txt path.")
        positions = load_actual_positions_um(self.scan_log_path)
        if positions.shape[0] != sample_count:
            raise ValueError(
                f"Sample position count ({positions.shape[0]}) does not match image count ({sample_count})."
            )
        return positions

    def _load_input_cube(self) -> np.ndarray:
        """按数据来源加载 K0 估计所需的输入立方体。"""
        if self.data_source == "mat_file":
            if self.mat_path is None:
                raise ValueError("MAT data source requires a MAT file path.")
            self.progress.emit(15)
            self.log.emit(f"正在加载 MAT 数据: {self.mat_path.name}")
            cube = load_mat_intensity_cube(self.mat_path)
            self.log.emit(f"MAT 数据已加载，尺寸={cube.shape}")
            return cube

        self.log.emit("\u6b63\u5728\u6536\u96c6\u56fe\u50cf\u6587\u4ef6...")
        files = collect_image_files(self.folder)
        if len(files) < 1:
            raise ValueError("\u81f3\u5c11\u9700\u8981\u4e00\u5f20\u56fe\u50cf\u3002")

        self.progress.emit(15)
        self.log.emit(f"\u627e\u5230 {len(files)} \u5f20\u56fe\u50cf\uff0c\u6b63\u5728\u8f7d\u5165\u6570\u636e\u7acb\u65b9\u4f53...")
        return load_intensity_cube(files, image_intensity_mode=self.image_intensity_mode)

    @Slot()
    def run(self) -> None:
        """执行 K0 自动估计流程并通过信号回传结果。"""
        try:
            self.progress.emit(0)
            cube = self._load_input_cube()
            sample_positions = self._resolve_sample_positions(cube.shape[2])
            left_expansion_frames = self.active_range_left_expansion_frames if self.expand_active_range else 0
            right_expansion_frames = self.active_range_right_expansion_frames if self.expand_active_range else 0
            cube, sample_positions, active_range = apply_active_range(
                cube,
                sample_positions,
                left_expansion_frames=left_expansion_frames,
                right_expansion_frames=right_expansion_frames,
            )
            self.last_cube = cube
            self.log.emit(
                f"自动 K0 使用有效扫描范围：{active_range.start_frame}-{active_range.end_frame}，"
                f"有效帧数={active_range.end_frame - active_range.start_frame + 1}"
            )
            if sample_positions is not None:
                self.log.emit(f"自动 K0 已加载 {sample_positions.shape[0]} 个 scan_log 非均匀采样位置。")

            self.progress.emit(55)
            self.log.emit(
                f"\u6b63\u5728\u81ea\u52a8\u4f30\u8ba1 K0\uff1a\u5019\u9009={self.candidate_ratio * 100.0:.1f}%\uff0c"
                f"\u7a97\u51fd\u6570={self.window_name}\uff0c"
                f"\u8865\u96f6={self.zero_padding_mode}"
            )
            result = estimate_global_k0(
                intensity_data=cube,
                step_size=self.step_size,
                candidate_ratio=self.candidate_ratio,
                window_name=self.window_name,
                window_alpha=self.window_alpha,
                zero_padding_mode=self.zero_padding_mode,
                sample_positions_um=sample_positions,
            )
            result["active_range"] = np.asarray([active_range.start_frame, active_range.end_frame], dtype=np.int32)
            result["active_ranges"] = np.asarray(active_range.ranges, dtype=np.int32)
            result["active_range_score"] = active_range.score.astype(np.float32)
            self.progress.emit(90)
            self.log.emit(f"\u5168\u5c40 K0 \u4f30\u8ba1\u5b8c\u6210\uff0cFFT长度={int(result.get('fft_length', cube.shape[2]))}\u3002")
            self.finished.emit(result)
            self.progress.emit(100)
        except Exception as exc:
            self.failed.emit(str(exc))
