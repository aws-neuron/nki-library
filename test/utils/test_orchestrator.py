# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import multiprocessing
import os
import pathlib
import pickle
import re
import shutil
import subprocess
import traceback
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from _pytest.config import Config

from . import feature_flag_helper
from .bir_neff_cache import CompiledKernel
from .common_dataclasses import (
    INF_ARTIFACT_DIR_NAME,
    KernelArgs,
    LazyGoldenGenerator,
    NeuronDeviceInfo,
    NKICompilationMode,
    PerRankLazyInputGenerator,
    Platforms,
    SeparationPassMode,
    TraceMode,
    UploadProfileMode,
    normalize_golden_output,
)
from .determinism_checker import DeterminismChecker
from .exceptions import CompilationException, InferenceException, TestStatus, ValidationException
from .host_management import Host, HostManager
from .metrics_collector import IMetricsCollector, MetricName
from .negative_test_helpers import is_in_negative_test_context
from .output_validator import OutputValidator
from .persistent_input_cache import link_raw_memmap
from .profiler_utils import (
    NEURON_RT_DBG_SEQ_IRAM_BLOCK_SIZES_KB,
    NEURON_RT_ENABLE_DGE_NOTIFICATIONS,
    NEURON_RT_INSTR_FETCH_ON_H2D,
    NEURON_RT_UCODE_LIB_PATH,
    ProfilerCommands,
    extract_and_filter_output_files,
)
from .s3_utils import parse_s3_uri


def _resolve_neuronx_cc_jobs(config) -> int | None:
    """Return --jobs N to pass to neuronx-cc, or None to leave it unset.

    'auto' divides available CPUs by the number of xdist workers so concurrent
    compilations share cores without thrashing.  '0' disables the flag entirely.
    """
    raw = feature_flag_helper.get_feature_flag(config, "neuronx_cc_jobs", "auto")
    if raw == "0":
        return None
    if raw != "auto":
        return int(raw)
    cpu_count = multiprocessing.cpu_count()
    workerinput = getattr(config, "workerinput", None)
    if workerinput:
        num_workers = workerinput.get("workercount", 1)
    else:
        num_workers = getattr(config.option, "numprocesses", None) or 1
        if num_workers == "auto":
            num_workers = cpu_count
    jobs = max(1, math.ceil(cpu_count / int(num_workers)))
    logging.debug("neuronx-cc jobs: cpu_count=%d num_workers=%s -> --jobs=%d", cpu_count, num_workers, jobs)
    return jobs


_BIRSIM_BOOL_FLAG_RE = re.compile(
    r"--enable-birsim(?:-after-all|-at-begin|-at-end|-with-kernel-inline|-sync-only)?"
    r"(?:\s*=\s*(\S+))?(?=\s|$)",
    re.IGNORECASE,
)
_BIRSIM_CHECKER_FLAG_RE = re.compile(r"--enable-checker-after(?=[=\s]|$)", re.IGNORECASE)
_BIRSIM_BOOL_TRUE_VALUES = {"true", "1"}
_BIRSIM_BOOL_FALSE_VALUES = {"false", "0"}


def _has_user_birsim_flag(additional_cmd_args: list[str]) -> bool:
    """Return True if the user-supplied compiler args contain an affirmative birsim opt-in.

    Recognized forms:
      - --enable-birsim is the master switch; --enable-birsim-{after-all,at-begin,
        at-end,sync-only,with-kernel-inline} are timing variants. All are
        cl::opt<bool>, so LLVM accepts the bare flag (=true) or =true/=1/=false/=0
        (case-insensitive).
      - --enable-checker-after=<passes> is a cl::list<std::string>; with the master
        flag also enabled, the named passes run birsim/birverifier. Treat any presence
        of the flag as opt-in: the compiler asserts if a checker variant is set
        without the master, so a user who passes only --enable-checker-after almost
        certainly intends birsim to run too.
    """
    for arg in additional_cmd_args:
        for match in _BIRSIM_BOOL_FLAG_RE.finditer(arg):
            value = match.group(1)
            if value is None:
                # bare flag, equivalent to =true under cl::opt<bool>
                return True
            value_lower = value.lower()
            if value_lower in _BIRSIM_BOOL_TRUE_VALUES:
                return True
            if value_lower not in _BIRSIM_BOOL_FALSE_VALUES:
                # cl::opt<bool> rejects unrecognized values at parse time, but be
                # conservative here so the dump is created if the compiler accepts.
                return True
        if _BIRSIM_CHECKER_FLAG_RE.search(arg):
            return True
    return False


@dataclass
class FilesystemArgs:
    base_output_directory_path: str
    host_manager: HostManager
    skip_remote_cleanup: bool
    force_local_cleanup: bool = False
    test_directory_name: Optional[str] = None
    artifacts_output_directory_path: Optional[str] = None


def run_separated_perf_analysis(test_dir: str, target_instance_family: str, profiled_file: str = "profiler_db"):
    """Run performance analysis on separation pass trace files.

    The separation pass splits a kernel's execution trace into distinct memory (DMA) and
    compute sections. By analyzing these separately, we can determine whether a kernel is
    memory-bounded or compute-bounded, quantify DMA bandwidth utilization vs compute
    throughput (e.g. matmul initiation intervals) for each section, and estimate the
    maximum achievable performance.

    This function analyzes one trace per NeuronCore (nc00/nc01), correlating postscheduler
    trace events with profiled runtime data from infer_result artifacts.

    Args:
        test_dir: Path to the test output directory containing trace files and infer_result/.
        target_instance_family: Instance family (e.g. 'trn2', 'trn3') for DMA bandwidth selection.
    """
    test_root = Path(test_dir)
    artifacts_dir = test_root / "artifacts"
    trace_configs = [
        # vnc_2 (LNC2): one trace per NeuronCore (files directly in artifacts/)
        # BB name varies by compiler version (e.g. "bb", "Block1"), so use glob.
        ("perf_sim_at_end_trace.nc00_sg00.sg0000.*.json", "sg00", "analysis_nc00.log"),
        ("perf_sim_at_end_trace.nc01_sg00.sg0000.*.json", "sg01", "analysis_nc01.log"),
        # vnc_1 (LNC1): single module-level trace (file in artifacts/sg00/)
        ("sg00/perf_sim_at_end_trace.module.sg0000.*.json", "sg00", "analysis_nc00.log"),
    ]
    for trace_pattern, subgraph, output_filename in trace_configs:
        matches = sorted(artifacts_dir.glob(trace_pattern))
        if not matches:
            logging.debug(f"Perf analysis trace file not found: {artifacts_dir / trace_pattern}")
            continue
        input_file = matches[0]
        try:
            from .perf_analysis_private import analyze_trace

            analyze_trace(
                input_file=input_file,
                profiled_file=test_root / INF_ARTIFACT_DIR_NAME / profiled_file,
                subgraph=subgraph,
                base_dir=test_root,
                output_filename=output_filename,
                target_instance_family=target_instance_family,
            )
        except Exception as e:
            logging.exception(e)


class Orchestrator:
    def __init__(
        self,
        config: Config,
        trace_mode: TraceMode,
        host_manager: HostManager,
        collector: IMetricsCollector,
        nki_compilation_mode: NKICompilationMode,
        perf_analysis_enabled: bool = False,
        hw_profile_enabled: bool = True,
        kernel_name: str | None = None,
    ):
        # Perf analysis reads NTFF artifacts, so it requires HW profiling enabled.
        if perf_analysis_enabled and not hw_profile_enabled:
            raise ValueError(
                "--enable-perf-analysis requires --enable-hw-profile=True; perf analysis "
                "reads NTFF artifacts produced by neuron-explorer capture."
            )

        self.fs_config: FilesystemArgs = FilesystemArgs(
            base_output_directory_path=feature_flag_helper.resolve_base_output_directory(config),
            host_manager=host_manager,
            skip_remote_cleanup=feature_flag_helper.get_feature_flag(config, "skip_remote_cleanup"),
            force_local_cleanup=feature_flag_helper.get_feature_flag(config, "force_local_cleanup", False),
        )
        self.trace_mode: TraceMode = trace_mode
        self.collector = collector
        self.kernel_under_test: Optional[KernelArgs] = None
        self._compiled_kernel: Optional[CompiledKernel] = None
        self.perf_analysis_enabled: bool = perf_analysis_enabled
        self.hw_profile_enabled: bool = hw_profile_enabled
        self.kernel_name: str | None = kernel_name

        self.profiler_binary_path: str = self.__get_neuron_binary_path_for__(config, "neuron-explorer")
        self.explorer_binary_path: str = self.__get_neuron_binary_path_for__(config, "neuron-explorer")
        self.neuron_ls_binary_path: str = self.__get_neuron_binary_path_for__(config, "neuron-ls")
        self.enable_kernel_debugging: bool = feature_flag_helper.get_feature_flag(config, "debug_kernels", False)
        self.skip_core_reset: bool = feature_flag_helper.get_feature_flag(config, "skip_core_reset", False)
        self.enable_dge_notifs: bool = feature_flag_helper.get_feature_flag(config, "enable_dge_notifs", False)
        self.enable_validation_histograms: bool = feature_flag_helper.get_feature_flag(
            config, "validation_histograms", False
        )
        self.nki_compilation_mode = nki_compilation_mode
        self.debugger_interactive: bool = feature_flag_helper.get_feature_flag(config, "debugger_interactive", False)
        self.debugger_core_id: int = feature_flag_helper.get_feature_flag(config, "debugger_core_id", 0)
        self.debugger_replay: bool = feature_flag_helper.get_feature_flag(config, "debugger_replay", False)
        self.neuronx_cc_jobs: int | None = _resolve_neuronx_cc_jobs(config)
        self.neuronx_cc_cache_path: str | None = feature_flag_helper.get_feature_flag(
            config, "s3_neuronx_cc_cache_path"
        )
        self.torch_ref_cache_path: str | None = feature_flag_helper.get_feature_flag(config, "s3_torch_ref_cache_path")
        # Validate the URI up front so a malformed path fails loudly instead of
        # silently becoming a cache miss. Empty/unset disables caching.
        if self.torch_ref_cache_path:
            parse_s3_uri(self.torch_ref_cache_path)
        self.upload_profile_to_explorer: Optional[UploadProfileMode] = (
            UploadProfileMode.from_str(val)
            if (val := feature_flag_helper.get_feature_flag(config, "upload_profile_to_explorer", None))
            else None
        )
        self.ucode_lib_path: Optional[str] = feature_flag_helper.get_feature_flag(config, "ucode_lib_path", None)
        if self.ucode_lib_path and not os.path.isfile(self.ucode_lib_path):
            raise ValueError(f"--ucode-lib-path does not exist: {self.ucode_lib_path}")
        if self.upload_profile_to_explorer and not hw_profile_enabled:
            raise ValueError(
                "--upload-profile-to-explorer requires --enable-hw-profile=True; "
                "explorer upload bundles the NTFF trace from neuron-explorer capture."
            )

    def execute(self, kernel_under_test: KernelArgs):
        kernel_under_test.compiler_input.enable_debugging = self.enable_kernel_debugging
        # Assign current collector to kernel_args (used by output_validator)
        kernel_under_test.collector = self.collector

        # Debugger mode: force device dump and disable breakpoints during compile+infer
        is_debugger = self.trace_mode == TraceMode.Debugger
        saved_breakpoint = None
        if is_debugger:
            kernel_under_test.compiler_input.enable_device_dump = True
            saved_breakpoint = os.environ.get("PYTHONBREAKPOINT")
            os.environ["PYTHONBREAKPOINT"] = "0"

        # Inject perf sim backend option when perf analysis is enabled
        if self.perf_analysis_enabled:
            perf_sim_flag = "--internal-backend-options=--enable-perf-sim"
            if perf_sim_flag not in kernel_under_test.compiler_input.additional_cmd_args:
                kernel_under_test.compiler_input.additional_cmd_args.append(perf_sim_flag)

        # Cap neuronx-cc thread count so xdist workers share CPUs without thrashing
        if self.neuronx_cc_jobs is not None:
            jobs_flag = f"--jobs={self.neuronx_cc_jobs}"
            if not any(a.startswith("--jobs") for a in kernel_under_test.compiler_input.additional_cmd_args):
                kernel_under_test.compiler_input.additional_cmd_args.append(jobs_flag)
                logging.info("neuronx-cc jobs capped to %d", self.neuronx_cc_jobs)

        self.__prepare_output_directory()
        assert self.fs_config.artifacts_output_directory_path

        # Start timing right before compilation stage
        self.collector.start_test()

        status = TestStatus.SUCCESS

        try:
            # Debugger replay: skip compile+infer, reuse artifacts from previous run
            if is_debugger and self.debugger_replay:
                local_artifact_download_path = os.path.join(
                    self.fs_config.artifacts_output_directory_path, INF_ARTIFACT_DIR_NAME
                )
                if not os.path.isdir(local_artifact_download_path):
                    raise InferenceException(
                        f"--debugger-replay requires existing inference artifacts at "
                        f"{local_artifact_download_path} from a previous debugger run"
                    )
                logging.info(f"Debugger replay: reusing artifacts from {local_artifact_download_path}")
                self._run_debugger_inference(kernel_under_test, local_artifact_download_path)
                return

            output_names = None
            if kernel_under_test.validation_args is not None:
                golden = normalize_golden_output(kernel_under_test.validation_args.golden_output)
                output_names = list(golden.keys())

            # Run Compilation
            self._run_compilation(self.collector, kernel_under_test, output_names)

            # Return early if trace-only or compile-only mode
            if self.trace_mode in (TraceMode.TraceOnly, TraceMode.CompileOnly):
                return

            # Create inputs
            with self.collector.timer(MetricName.INPUT_DUMP_TIME):
                input_file_paths = self.__dump_kernel_inputs__(
                    self.fs_config.artifacts_output_directory_path, kernel_under_test
                )
            # Run Inference
            local_artifact_download_path = self._run_inference(kernel_under_test, input_file_paths)
            self._rename_neff_outputs_to_python_names(local_artifact_download_path, output_names)

            # Inference (incl. any host-rotation retries) has succeeded, so no
            # further re-upload of the input tensors can occur. Only now is it
            # safe to free their local disk under --force-local-cleanup; deleting
            # earlier would make a rotation retry ship an archive missing the
            # inputs (neuron-explorer "open inp-*.bin: no such file or directory").
            if self.fs_config.force_local_cleanup:
                from .host_io import cleanup_input_bins

                cleanup_input_bins(self.fs_config.artifacts_output_directory_path)

            # Run performance analysis if perf analysis is enabled
            if self.perf_analysis_enabled:
                logging.info("Running performance analysis")
                run_separated_perf_analysis(
                    self.fs_config.artifacts_output_directory_path,
                    kernel_under_test.compiler_input.platform_target.value,
                )
                if kernel_under_test.compiler_input.separation_pass_mode != SeparationPassMode.NONE:
                    self._record_separation_pass_metrics(self.collector, self.fs_config.artifacts_output_directory_path)

            # single compile separation pass: hoist DMAs from the compiled unseparated NEFF and re-run on hardware, without needing to re-compile to get a separated NEFF
            if self.trace_mode == TraceMode.CompileAndInferAndSeparate:
                try:
                    from .single_compile_separation_private import run_single_compile_separation_pass
                except ImportError:
                    logging.warning(
                        "single_compile_separation_private not available, skipping single compile separation pass"
                    )
                    return

                run_single_compile_separation_pass(self, kernel_under_test, input_file_paths)
                logging.info("Skipping output validation in compile-and-infer-and-separate mode")
            elif is_debugger:
                # Restore breakpoints after compile+infer, before nki.debug()
                self._restore_breakpoint(saved_breakpoint)
                self._run_debugger_inference(kernel_under_test, local_artifact_download_path)
            else:
                self._run_validation(
                    kernel_under_test,
                    local_artifact_download_path,
                )

        except (CompilationException, InferenceException, ValidationException) as e:
            # Use EXPECTED_FAILURE for negative tests, otherwise use actual failure status
            status = TestStatus.EXPECTED_FAILURE if is_in_negative_test_context() else e.status
            raise
        finally:
            # Restore breakpoints if disabled for debugger mode (safety net for error path)
            if is_debugger:
                self._restore_breakpoint(saved_breakpoint)

            self.collector.add_dimension(
                {
                    "KernelAPI": kernel_under_test.kernel_func.__name__,
                    "LNCCores": str(kernel_under_test.compiler_input.logical_nc_config),
                    "Status": status.value,
                    "IsSuccessful": "true" if status == TestStatus.SUCCESS else "false",
                }
            )
            with self.collector.timer(MetricName.ARTIFACT_PARSE_TIME):
                self.collector.parse_artifacts(
                    artifact_dir=self.fs_config.artifacts_output_directory_path,
                    inference_artifact_dir=INF_ARTIFACT_DIR_NAME,
                    target=kernel_under_test.compiler_input.platform_target.value,
                )

            # Record -1 for phases that didn't run (must happen AFTER parse_artifacts)
            self._record_missing_phase_metrics(self.collector)

            # Upload profile to Neuron Explorer if requested. A separation run in ALWAYS
            # mode has already uploaded the separated pair (and disabled further uploads),
            # so this block only fires when no separation upload occurred.
            if self.upload_profile_to_explorer:
                should_upload = self.upload_profile_to_explorer == UploadProfileMode.ALWAYS or (
                    self.upload_profile_to_explorer == UploadProfileMode.ON_FAIL_ONLY and status != TestStatus.SUCCESS
                )
                if should_upload:
                    self._upload_profile_to_explorer()

    def get_test_artifact_output_path(self) -> str:
        assert self.fs_config.artifacts_output_directory_path, "Test has to be executed first"

        return self.fs_config.artifacts_output_directory_path

    def _record_missing_phase_metrics(self, collector: IMetricsCollector) -> None:
        """
        Record -1 values for metrics of phases that didn't execute.
        Ensures all tests have consistent metric structure
        """
        timing_metrics = [
            MetricName.HOST_LOCK_TIME,
            MetricName.FILE_TRANSFER_UPLOAD_TIME,
            MetricName.FILE_TRANSFER_DOWNLOAD_TIME,
            MetricName.INFERENCE_TIME_TOTAL,
            MetricName.PROFILE_JSON_GENERATION_TIME,
            MetricName.NEURON_PROFILE_CAPTURE_TIME,
            MetricName.NEURON_PROFILE_SHOW_TIME,
            MetricName.CORE_ALLOCATION_TIME,
            MetricName.CORE_LOCK_HOLD_TIME,
            MetricName.INFERENCE_TIME,
            MetricName.VALIDATION_TIME,
            MetricName.SIMULATION_TIME,
            MetricName.HOST_ARCH_VALIDATION_TIME,
            MetricName.FRONTEND_TRACE_TIME,
            MetricName.MLIR_TO_BIR_TIME,
            MetricName.BIR_TO_NEFF_TIME,
            MetricName.INPUT_DUMP_TIME,
            MetricName.ARTIFACT_PARSE_TIME,
            MetricName.DETERMINISM_CHECK_TIME,
            MetricName.VALIDATION_OUTPUT_LOAD_TIME,
            MetricName.VALIDATION_COMPARE_TIME,
            MetricName.GOLDEN_ACQUISITION_TIME,
            MetricName.GOLDEN_COMPUTATION_TIME,
        ]

        for metric_name in timing_metrics:
            if not collector.has_metric(metric_name):
                collector.record_metric(metric_name, -1.0, "Seconds")

        if not collector.has_metric(MetricName.ACCURACY_HW):
            collector.record_metric(MetricName.ACCURACY_HW, -1.0, "None")

    @staticmethod
    def _record_separation_pass_metrics(collector: IMetricsCollector, test_dir: str) -> None:
        """Parse analysis_nc*.log files and record max compute/memory time across cores."""
        compute_times = []
        memory_times = []
        for log_name in ("analysis_nc00.log", "analysis_nc01.log"):
            log_path = Path(test_dir) / log_name
            if not log_path.exists():
                continue
            content = log_path.read_text()
            m = re.search(r"Duration excluding overlap \(profiled\):\s+(\d+)", content)
            if m:
                compute_times.append(float(m.group(1)))
            m = re.search(r"DMA duration.*?:\s+(\d+)", content)
            if m:
                memory_times.append(float(m.group(1)))
        if compute_times:
            collector.record_metric(MetricName.SEPARATED_COMPUTE_TIME, max(compute_times) / 1e9, "Seconds")
        if memory_times:
            collector.record_metric(MetricName.SEPARATED_MEMORY_TIME, max(memory_times) / 1e9, "Seconds")

    def _upload_profile_to_explorer(self) -> None:
        """Upload NEFF and NTFF profiling artifacts to Neuron Explorer."""
        try:
            from .explorer_upload_private import upload_profile_to_explorer
        except ImportError:
            logging.warning("explorer_upload_private not available, skipping explorer upload")
            return

        artifact_dir = self.fs_config.artifacts_output_directory_path
        if artifact_dir is None:
            logging.warning("No artifact directory available, skipping explorer upload")
            return

        profile_url = upload_profile_to_explorer(artifact_dir)
        if profile_url:
            self.collector.add_dimension({MetricName.EXPLORER_PROFILE_URL: profile_url})

    def _run_compilation(
        self,
        collector: IMetricsCollector,
        kernel_under_test: KernelArgs,
        output_names: list[str] | None,
    ) -> None:
        """Run kernel compilation phase."""
        # Skip compilation for simulator mode
        if self.trace_mode == TraceMode.Simulator:
            logging.info("Skipping compilation in simulator mode")
            return

        # Lazy import to avoid loading nki.compiler.backends in simulation mode
        from .kernel_tracer import trace_kernel

        logging.info(f"Running compilation for {self.fs_config.artifacts_output_directory_path=}")
        try:
            with collector.timer(MetricName.COMPILATION_TIME):
                # Dump birsim inputs/goldens when the user opts in via the framework flag or
                # via raw compiler args. See _has_user_birsim_flag for the set of accepted forms.
                additional_cmd_args = kernel_under_test.compiler_input.additional_cmd_args or []
                if kernel_under_test.compiler_input.enable_birsim or _has_user_birsim_flag(additional_cmd_args):
                    self._dump_birsim_artifacts(kernel_under_test)

                # A test may pin the frontend via CompilerArgs.nki_compilation_mode;
                # otherwise fall back to the session/CLI default (--nki-compilation-mode).
                frontend_mode = kernel_under_test.compiler_input.nki_compilation_mode or self.nki_compilation_mode
                self._compiled_kernel = trace_kernel(
                    kernel_under_test=kernel_under_test,
                    mode=self.trace_mode,
                    output_directory=self.fs_config.artifacts_output_directory_path,
                    output_names=output_names,
                    frontendMode=frontend_mode,
                    neuronx_cc_cache_path=self.neuronx_cc_cache_path,
                    collector=collector,
                )

            # Record MLIR→BIR and BIR→NEFF sub-phase timings from the compiled kernel
            if self._compiled_kernel is not None:
                collector.record_timer(MetricName.MLIR_TO_BIR_TIME, self._compiled_kernel.mlir_time)
                collector.record_timer(MetricName.BIR_TO_NEFF_TIME, self._compiled_kernel.neuronx_cc_time)
        except Exception as e:
            # Print full exception details including stdout, stderr, and stacktrace
            error_msg = f"Compilation failed with exception: {type(e).__name__}: {str(e)}\n"

            # If it's a CalledProcessError, include stdout and stderr
            if isinstance(e, subprocess.CalledProcessError):
                error_msg += f"\nReturn code: {e.returncode}\n"
                error_msg += f"\nStdout:\n{e.stdout}\n" if e.stdout else "\nStdout: (empty)\n"
                error_msg += f"\nStderr:\n{e.stderr}\n" if e.stderr else "\nStderr: (empty)\n"

            # Add full stacktrace
            error_msg += f"\nFull traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"

            logging.error(error_msg)
            raise CompilationException(error_msg) from e

    def _run_inference(self, kernel_under_test: KernelArgs, input_file_paths: dict[str, str]) -> str | None:
        """Run compiled kernel on hardware using neuron-explorer."""
        assert self.fs_config.artifacts_output_directory_path

        logging.info(f"Running inference for {self.fs_config.artifacts_output_directory_path=}")

        if self.trace_mode == TraceMode.Simulator:
            return self._run_simulator_inference(kernel_under_test)

        # Stage uCode lib into the artifacts dir
        if self.ucode_lib_path:
            dest = os.path.join(self.fs_config.artifacts_output_directory_path, os.path.basename(self.ucode_lib_path))
            shutil.copy(self.ucode_lib_path, dest)
            logging.info(f"Staged uCode lib for upload: {dest}")

        try:
            with closing(
                self.fs_config.host_manager.get_host_assignment_with_retry(
                    platform_target=kernel_under_test.compiler_input.platform_target,
                    collector=self.collector,
                    collective_ranks=kernel_under_test.inference_args.collective_ranks,
                    lnc_config=kernel_under_test.compiler_input.logical_nc_config,
                )
            ) as host_assignment_generator:
                for host_assignment_attempt in host_assignment_generator:
                    with host_assignment_attempt as execution_host:
                        with self.collector.timer(MetricName.INFERENCE_TIME_TOTAL):
                            with self.collector.timer(MetricName.HOST_ARCH_VALIDATION_TIME):
                                self.__confirm_host_arch__(
                                    execution_host,
                                    kernel_under_test.compiler_input.platform_target,
                                )

                            # Soft-join the core-allocation FIFO queue before
                            # uploading artifacts so upload latency does not cost
                            # this attempt its FIFO position. The
                            # later get_core_allocation reuses the same cached
                            # manager/entry_id to commit; if the slot is
                            # pruned/bumped during a long upload, the first
                            # ready=True poll transparently re-enqueues (poll
                            # verb), so no resume code is needed. The poll itself
                            # is best-effort (genuine failures are swallowed), but
                            # if the joined slot's queue ETA already exceeds
                            # patience it raises to rotate to another host before
                            # this attempt pays the upload cost.
                            execution_host.soft_join_queue(
                                collector=self.collector,
                                collective_ranks=kernel_under_test.inference_args.collective_ranks,
                                lnc_config=kernel_under_test.compiler_input.logical_nc_config,
                            )

                            with execution_host.prepare_host(
                                target_directory=self.fs_config.artifacts_output_directory_path,
                                skip_remote_cleanup=self.fs_config.skip_remote_cleanup,
                                collector=self.collector,
                                force_local_cleanup=self.fs_config.force_local_cleanup,
                            ):
                                return self.__run_profiler_on_host__(
                                    execution_host,
                                    kernel_under_test,
                                    input_file_paths,
                                    self.collector,
                                )
        except Exception as e:
            if isinstance(e, InferenceException):
                raise e
            raise InferenceException(e) from e

    def __run_profiler_on_host__(
        self,
        execution_host: Host,
        kernel_under_test: KernelArgs,
        input_file_paths: dict[str, str],
        collector,
    ) -> str | None:
        """Run neuron-explorer on the prepared host."""
        kernel_input_args = self.__format_profiler_kernel_input_args__(kernel_under_test, input_file_paths)

        env_vars: Optional[dict[str, str]] = kernel_under_test.inference_args.env_vars
        separation_pass_enabled = kernel_under_test.compiler_input.separation_pass_mode != SeparationPassMode.NONE
        single_compile_separation_enabled = self.trace_mode == TraceMode.CompileAndInferAndSeparate
        if (
            self.enable_kernel_debugging
            or self.enable_dge_notifs
            or separation_pass_enabled
            or single_compile_separation_enabled
            or self.perf_analysis_enabled
        ):
            if env_vars is None:
                env_vars = {}

            if NEURON_RT_ENABLE_DGE_NOTIFICATIONS not in env_vars:
                env_vars[NEURON_RT_ENABLE_DGE_NOTIFICATIONS] = "1"

        if kernel_under_test.compiler_input.platform_target.value == "trn3_a0":
            if env_vars is None:
                env_vars = {}

            env_vars["NEURON_RT_ALLOW_LEGACY_NEFF"] = "1"

        # Point the runtime at the uploaded uCode lib
        if self.ucode_lib_path:
            if env_vars is None:
                env_vars = {}
            env_vars[NEURON_RT_UCODE_LIB_PATH] = f"./{os.path.basename(self.ucode_lib_path)}"

        # Forward the IRAM cache block size config to the execution host if set locally.
        # Value-only passthrough — no file to stage.
        iram_block_size = os.environ.get(NEURON_RT_DBG_SEQ_IRAM_BLOCK_SIZES_KB)
        if iram_block_size:
            if env_vars is None:
                env_vars = {}
            env_vars[NEURON_RT_DBG_SEQ_IRAM_BLOCK_SIZES_KB] = iram_block_size

        # Forward the instruction-fetch-on-H2D toggle to the execution host if set locally.
        # Value-only passthrough — no file to stage.
        instr_fetch_on_h2d = os.environ.get(NEURON_RT_INSTR_FETCH_ON_H2D)
        if instr_fetch_on_h2d is not None:
            if env_vars is None:
                env_vars = {}
            env_vars[NEURON_RT_INSTR_FETCH_ON_H2D] = instr_fetch_on_h2d

        # Save and download all outputs when determinism check or profile all runs is enabled.
        # Otherwise, only save/download the last execution's outputs.
        save_all_outputs = (
            kernel_under_test.inference_args.enable_determinism_check
            or kernel_under_test.inference_args.profile_all_runs
        )

        force_clean_input_writes: bool = (
            kernel_under_test.inference_args.num_runs > 1
            if kernel_under_test.inference_args.num_runs is not None
            else False
        )

        profiler_cmds = ProfilerCommands(
            num_runs=kernel_under_test.inference_args.resolved_num_runs,
            profile_all_runs=kernel_under_test.inference_args.profile_all_runs,
            profiler_binary_path=self.profiler_binary_path,
            kernel_input_args=kernel_input_args,
            metrics_enabled=self.collector.metrics_enabled
            or separation_pass_enabled
            or single_compile_separation_enabled,
            collective_ranks=kernel_under_test.inference_args.collective_ranks,
            profile_all_ranks=kernel_under_test.inference_args.profile_all_ranks,
            env_vars=env_vars,
            perf_analysis_enabled=self.perf_analysis_enabled,
            hw_profile_enabled=self.hw_profile_enabled,
            save_all_outputs=save_all_outputs,
            force_clean_input_writes=force_clean_input_writes,
            separation_pass_enabled=separation_pass_enabled or single_compile_separation_enabled,
            explorer_binary_path=self.explorer_binary_path,
        )

        def get_list_of_files_to_copy(stdout: str) -> list[str]:
            outputs_to_copy = extract_and_filter_output_files(stdout, save_all_outputs)

            # Build complete list of files to copy from remote host
            files_to_copy = []
            files_to_copy.extend(outputs_to_copy)
            files_to_copy.extend(profiler_cmds.expected_ntff_files)
            files_to_copy.append("log-infer.txt")
            files_to_copy.extend(profiler_cmds.expected_profiler_view_json_files)
            files_to_copy.extend(profiler_cmds.expected_detailed_parquet_dirs)
            files_to_copy.extend(profiler_cmds.expected_show_session_json_files)
            files_to_copy.append("debug_output")
            return files_to_copy

        hardware_cmd = profiler_cmds.get_hardware_command()
        post_lock_cmd = profiler_cmds.get_post_lock_command()

        assert self.fs_config.artifacts_output_directory_path

        return execution_host.execute_command(
            command=f"( {hardware_cmd} ) 2>&1 | tee log-infer.txt",
            target_directory=self.fs_config.artifacts_output_directory_path,
            collective_ranks=kernel_under_test.inference_args.collective_ranks,
            lnc_config=kernel_under_test.compiler_input.logical_nc_config,
            do_copy_artifacts=True,
            get_list_of_files_to_copy=get_list_of_files_to_copy,
            collector=collector,
            post_lock_command=(f"( {post_lock_cmd} ) 2>&1 | tee -a log-infer.txt" if post_lock_cmd else None),
            skip_core_reset=self.skip_core_reset,
        )

    def _dump_output_tensors(self, output_tensors: dict[str, np.ndarray]) -> str:
        """Dump output tensors to inference artifact directory.

        Returns path to output directory.
        """
        assert self.fs_config.artifacts_output_directory_path, "Test has to be executed first"
        output_path = os.path.join(self.fs_config.artifacts_output_directory_path, INF_ARTIFACT_DIR_NAME)
        self.__dump_tensors__(output_path, output_tensors, lambda name: name)
        return output_path

    def _run_simulator_inference(self, kernel_under_test: KernelArgs) -> Optional[str]:
        """Run kernel using nki.simulate and dump outputs for validation."""
        from .simulation_setup import run_simulator_inference

        try:
            with self.collector.timer(MetricName.SIMULATION_TIME):
                output_tensors = run_simulator_inference(kernel_under_test)
            if not output_tensors:
                return None
            return self._dump_output_tensors(output_tensors)
        except Exception as e:
            raise InferenceException(str(e)) from e

    @staticmethod
    def _restore_breakpoint(saved_breakpoint: Optional[str]) -> None:
        """Restore PYTHONBREAKPOINT env var to its original value."""
        if saved_breakpoint is None:
            os.environ.pop("PYTHONBREAKPOINT", None)
        else:
            os.environ["PYTHONBREAKPOINT"] = saved_breakpoint

    def _run_debugger_inference(
        self, kernel_under_test: KernelArgs, local_artifact_download_path: Optional[str]
    ) -> None:
        """Run nki.debug on device dumps produced by inference."""
        if local_artifact_download_path is None:
            raise InferenceException("Debugger mode requires inference artifacts but none were produced")

        from .debugger_setup import run_debugger_inference

        dump_dir = os.path.join(local_artifact_download_path, "debug_output")

        rtol, atol = 1e-2, 1e-1
        if kernel_under_test.validation_args is not None:
            rtol = kernel_under_test.validation_args.relative_accuracy
            atol = kernel_under_test.validation_args.absolute_accuracy

        lnc = kernel_under_test.compiler_input.logical_nc_config
        platform_target = kernel_under_test.compiler_input.platform_target.value

        logging.info(f"Running nki.debug for {dump_dir}")
        run_debugger_inference(
            kernel_under_test.kernel_func,
            self._single_core_kernel_input(kernel_under_test),
            dump_dir,
            core_id=self.debugger_core_id,
            interactive=self.debugger_interactive,
            rtol=rtol,
            atol=atol,
            lnc=lnc,
            platform_target=platform_target,
        )

    def _rename_neff_outputs_to_python_names(self, artifact_path: str | None, output_names: list[str] | None) -> None:
        """Rename NEFF output files (output_N) to Python-level names.

        When using neuron-explorer with a CompiledKernel that has output_specs,
        the NEFF uses generic names like output_0 but the validator expects
        the Python-level names (e.g. 'y'). This renames the files to match.
        """
        if artifact_path is None or self._compiled_kernel is None or not output_names:
            # The simulation mode does not compile the kernel and produces no
            # downloaded artifact directory, so there is nothing to rename.
            return

        aliases = self._compiled_kernel.input_output_aliases or {}
        for i, python_name in enumerate(output_names):
            neff_name = aliases[i] if i in aliases else python_name
            if neff_name == python_name:
                continue
            src = os.path.join(artifact_path, neff_name)
            dst = os.path.join(artifact_path, python_name)
            if os.path.exists(src):
                os.rename(src, dst)
                logging.info(f"Renamed output {neff_name} -> {python_name}")
            else:
                logging.warning(f"Expected output file not found: {src}")

    def _run_validation(
        self,
        kernel_under_test: KernelArgs,
        local_artifact_download_path: Optional[str],
    ) -> None:
        """Run output validation phase."""
        if local_artifact_download_path is None:
            logging.warning("Skipping output validation due to missing output artifact path")
            return

        # Skip validation if separation pass is enabled (invalidates output)
        if kernel_under_test.compiler_input.separation_pass_mode != SeparationPassMode.NONE:
            logging.info("Skipping output validation because separation_pass_mode is enabled")
            return

        logging.info(f"Running validation for {self.fs_config.artifacts_output_directory_path=}")
        # Time the validation phase
        assert self.fs_config.artifacts_output_directory_path
        with self.collector.timer(MetricName.VALIDATION_TIME):
            try:
                abs_out_file_paths = []
                output_path = pathlib.Path(local_artifact_download_path)
                for file_name in output_path.iterdir():
                    if file_name.is_file():
                        abs_out_file_paths.append(file_name.absolute().as_posix())
                    elif file_name.is_dir() and file_name.name.startswith("output_worker_"):
                        # Include files from per-rank output directories
                        for rank_file in file_name.iterdir():
                            if rank_file.is_file():
                                abs_out_file_paths.append(rank_file.absolute().as_posix())

                validation_log_filepath = os.path.join(
                    self.fs_config.artifacts_output_directory_path,
                    INF_ARTIFACT_DIR_NAME,
                    "log-validate.txt",
                )

                # Run determinism check if enabled and not simulating
                if kernel_under_test.inference_args.enable_determinism_check:
                    if self.trace_mode == TraceMode.Simulator:
                        logging.warning("Skipping determinism check; simulation does not produce multiple outputs.")
                    else:
                        # Build list of paths to check (per-rank for collectives, single path otherwise)
                        if kernel_under_test.inference_args.collective_ranks > 1:
                            check_paths = [
                                (
                                    rank,
                                    os.path.join(local_artifact_download_path, f"output_worker_{rank}"),
                                )
                                for rank in range(kernel_under_test.inference_args.collective_ranks)
                            ]
                        else:
                            check_paths = [(None, local_artifact_download_path)]

                        for rank, path in check_paths:
                            checker = DeterminismChecker(
                                kernel_under_test,
                                path,
                                kernel_under_test.inference_args.resolved_num_runs,
                                self.collector,
                                logfile_path=validation_log_filepath,
                                rank_id=rank,
                            )
                            checker.check()

                OutputValidator(
                    kernel_under_test,
                    abs_out_file_paths,
                ).validate(
                    logfile_path=validation_log_filepath,
                    enable_histograms=self.enable_validation_histograms,
                )

            except Exception as e:
                raise ValidationException(str(e)) from e

    def __dump_simulation_artifacts__(
        self, base_path: str, kernel_output: list[Any], golden_output: dict[str, Any]
    ) -> str:
        assert len(kernel_output) == 1 and len(golden_output) == 1, (
            "Simulator does not label outputs with tensor names, so it's ambigious when more than a single output is present"
        )

        golden_output_key = next(iter(golden_output))

        simulation_output_path = os.path.join(base_path, "simulation_output")
        os.makedirs(simulation_output_path, exist_ok=True)

        with open(os.path.join(simulation_output_path, golden_output_key), "wb") as f:
            kernel_output[0].tofile(f)

        return simulation_output_path

    def __get_neuron_binary_path_for__(self, config: Config, binary: str):
        neuron_base_binary_path: str = feature_flag_helper.get_feature_flag(config, "neuron_tools_bin_path")
        if os.path.isabs(neuron_base_binary_path):
            return os.path.join(neuron_base_binary_path, binary)
        else:
            return binary  # assume that binary is going to be inside PATH

    def __confirm_host_arch__(self, host: Host, platform_target: Platforms):
        # TODO: Make host architecture validation more robust by adding the ability to query the host type from the host
        attached_neuron_devices: list[NeuronDeviceInfo] = host.get_neuron_device_info()

        if platform_target in (
            Platforms.TRN2,
            Platforms.TRN3,
            Platforms.TRN3_A0,
            Platforms.TRN3_PDS,
            Platforms.TRN3_PDS_A0,
        ):
            assert sum(neuron_info.nc_count for neuron_info in attached_neuron_devices) in [
                4,
                8,
                64,
                128,
            ]
        else:
            raise Exception(f"{platform_target} is currently unsupported by this test framework!")

    def __format_profiler_kernel_input_args__(self, kernel_input: KernelArgs, input_file_paths: dict[str, str]) -> str:
        """Format kernel input arguments for neuron-explorer command.

        Args:
            kernel_input: Kernel arguments
            input_file_paths: Dict mapping arg names to file paths
                For single input: {"input": "/path/inp-input-000.bin", "weights": "/path/inp-weights-000.bin"}
                For per-rank inputs: {"--multi-input": "4rank_inputs.txt"}

        Returns:
            Formatted string for neuron-explorer command:
                For single input: "input inp-input-000.bin weights inp-weights-000.bin"
                For per-rank inputs: "--multi-input 4rank_inputs.txt"
        """
        if kernel_input.kernel_input is None:
            return ""
        return " ".join(f"{arg_name} {os.path.basename(file_path)}" for arg_name, file_path in input_file_paths.items())

    @staticmethod
    def _single_core_kernel_input(kernel_under_test: KernelArgs) -> dict[str, Any]:
        """Kernel inputs as a plain mapping, for paths that inspect a single core.

        Per-rank inputs are materialised for rank 0, the rank these paths look at.
        """
        kernel_input = kernel_under_test.kernel_input
        if isinstance(kernel_input, PerRankLazyInputGenerator):
            return kernel_input.for_rank(0)
        return kernel_input or {}

    def __dump_tensors__(
        self,
        target_directory: str,
        tensors: dict[str, Any],
        name_fn: Callable[[str], str],
        force_numpy_arrays: bool = False,
    ) -> dict[str, str]:
        """
        Dump tensors to target directory using provided naming function.

        For numpy arrays, saves as raw .bin files for neuron-explorer compatibility.
        """
        os.makedirs(target_directory, exist_ok=True)
        dumped_files = {}
        for name, value in tensors.items():
            if isinstance(value, np.ndarray):
                if force_numpy_arrays:
                    file_name = name_fn(name) + ".npy"
                    file_path = os.path.join(target_directory, file_name)
                    np.save(file_path, value)
                else:
                    # Save as raw binary for neuron-explorer
                    file_name = name_fn(name) + ".bin"
                    file_path = os.path.join(target_directory, file_name)
                    if not link_raw_memmap(value, file_path):
                        with open(file_path, "wb") as f:
                            value.tofile(f)

                dumped_files[name] = file_path
            else:
                file_name = name_fn(name)
                file_path = os.path.join(target_directory, file_name)
                with open(file_path, "wb") as f:
                    pickle.dump(value, f)
                dumped_files[name] = file_path
        return dumped_files

    def _dump_birsim_artifacts(self, kernel_under_test: KernelArgs) -> None:
        """Dump kernel inputs and golden outputs in birsim format for debugging."""
        assert self.fs_config.artifacts_output_directory_path
        logical_nc_config = kernel_under_test.compiler_input.logical_nc_config
        artifacts_root = os.path.join(self.fs_config.artifacts_output_directory_path, "artifacts")
        for nc_idx in range(logical_nc_config):
            nc_dir = f"nc{nc_idx:02d}"
            birsim_dir = os.path.join(artifacts_root, nc_dir, "sg00")
            # Birsim requires naming convention: value_{name}.npy
            if kernel_under_test.kernel_input:
                _ = self.__dump_tensors__(
                    birsim_dir, self._single_core_kernel_input(kernel_under_test), lambda name: f"value_{name}", True
                )
            if kernel_under_test.validation_args:
                if (
                    isinstance(kernel_under_test.validation_args.golden_output, LazyGoldenGenerator)
                    and kernel_under_test.validation_args.golden_output.golden is not None
                ):
                    output_golden = kernel_under_test.validation_args.golden_output.golden
                    # Drop CustomValidatorWithOutputTensorData entries: birsim has no golden to
                    # compare against, but input dumps still feed sanitizer/race/barrier checks.
                    output_golden = {k: v for k, v in output_golden.items() if isinstance(v, np.ndarray)}
                    if output_golden:
                        # Cast goldens to the kernel's output dtype (torch_ref often returns fp32
                        # but birsim expects the dtype to match the kernel output).
                        output_ndarray = kernel_under_test.validation_args.golden_output.output_ndarray
                        for k in output_golden:
                            if k in output_ndarray and output_golden[k].dtype != output_ndarray[k].dtype:
                                output_golden[k] = output_golden[k].astype(output_ndarray[k].dtype)
                        # birsim is looking for files with specific naming pattern. moreover, they
                        # have to be numpy files, not binary files
                        _ = self.__dump_tensors__(birsim_dir, output_golden, lambda name: f"value_{name}", True)
                else:
                    raise AssertionError(
                        "Birsim does not support custom validator as golden output! Please disable bir sim or switch to a different golden generator"
                    )

    def __dump_kernel_inputs__(self, target_directory: str, kernel_under_test: KernelArgs) -> dict[str, str]:
        """
        Dump kernel inputs to target directory.

        Returns:
            Dict mapping arg names to file paths.
            For single input: {"input": "/path/inp-input-000.bin", "weights": "/path/inp-weights-000.bin"}
            For per-rank inputs: {"--multi-input": "4rank_inputs.txt"}
        """
        logging.info(f"Dumping kernel arguments for {self.fs_config.artifacts_output_directory_path=}")

        if kernel_under_test.kernel_input is None:
            return {}

        if isinstance(kernel_under_test.kernel_input, PerRankLazyInputGenerator):
            return self.__dump_per_rank_inputs__(target_directory, kernel_under_test)

        # Standard single-input case: {"input": "/path/inp-input-000.bin", ...}
        return self.__dump_tensors__(target_directory, kernel_under_test.kernel_input, lambda name: f"inp-{name}-000")

    def __dump_per_rank_inputs__(self, target_directory: str, kernel_under_test: KernelArgs) -> dict[str, str]:
        """
        Dump per-rank inputs and generate multi-input file for neuron-explorer.

        The --multi-input neuron-explorer flag is used for collectives with per-rank inputs.
        It specifies a file where each line provides inputs for one rank (line 1 = rank 0,
        line 2 = rank 1, etc.). Each line has space-separated pairs: "arg_name filename ..."

        Example 2rank_inputs.txt for 2 ranks with 'input' and 'replica_group' args:
            input inp-input-rank0.bin replica_group inp-replica_group-rank0.bin
            input inp-input-rank1.bin replica_group inp-replica_group-rank1.bin

        Returns:
            Dict with "--multi-input" key mapping to the multi-input filename.
            This gets formatted to "--multi-input 4rank_inputs.txt" by __format_profiler_kernel_input_args__.
        """
        assert isinstance(kernel_under_test.kernel_input, PerRankLazyInputGenerator)
        num_ranks = kernel_under_test.inference_args.collective_ranks

        if source_directory := kernel_under_test.kernel_input.input_artifacts_directory:
            return self.__reuse_per_rank_inputs__(target_directory, source_directory, num_ranks)

        multi_input_lines = []

        for rank_id in range(num_ranks):
            rank_inputs = kernel_under_test.kernel_input.for_rank(rank_id)

            # Dump inputs for this rank with rank suffix
            rank_input_paths = self.__dump_tensors__(
                target_directory, rank_inputs, lambda name, r=rank_id: f"inp-{name}-rank{r}"
            )

            # Build line for this rank: "arg_name filename arg_name filename ..."
            line_parts = []
            for arg_name, file_path in rank_input_paths.items():
                line_parts.append(f"{arg_name} {os.path.basename(file_path)}")
            multi_input_lines.append(" ".join(line_parts))

        # Write multi-input file
        multi_input_filename = f"{num_ranks}rank_inputs.txt"
        multi_input_file = os.path.join(target_directory, multi_input_filename)
        with open(multi_input_file, "w") as f:
            f.write("\n".join(multi_input_lines) + "\n")

        logging.info(f"Created multi-input file: {multi_input_file}")
        return {"--multi-input": multi_input_filename}

    @staticmethod
    def __reuse_per_rank_inputs__(
        target_directory: str,
        source_directory: str,
        num_ranks: int,
    ) -> dict[str, str]:
        """Reuse a prior profile's inputs without recreating FSx hard links."""
        source_directory = os.path.abspath(source_directory)
        source_manifest = os.path.join(source_directory, f"{num_ranks}rank_inputs.txt")
        with open(source_manifest) as f:
            source_lines = [line.strip() for line in f if line.strip()]
        if len(source_lines) != num_ranks:
            raise ValueError(
                f"Expected {num_ranks} entries in reused input manifest, got {len(source_lines)}: {source_manifest}"
            )

        output_lines = []
        for line in source_lines:
            parts = line.split()
            if len(parts) % 2:
                raise ValueError(f"Invalid reused input manifest line: {line}")
            for index in range(1, len(parts), 2):
                if not os.path.isabs(parts[index]):
                    parts[index] = os.path.join(source_directory, parts[index])
            output_lines.append(" ".join(parts))

        output_filename = f"{num_ranks}rank_inputs.txt"
        output_manifest = os.path.join(target_directory, output_filename)
        with open(output_manifest, "w") as f:
            f.write("\n".join(output_lines) + "\n")
        logging.info("Reused per-rank inputs from %s", source_manifest)
        return {"--multi-input": output_filename}

    def __prepare_output_directory(self):
        self.fs_config.test_directory_name = feature_flag_helper.construct_test_output_directory_name()
        self.fs_config.artifacts_output_directory_path = os.path.join(
            self.fs_config.base_output_directory_path,
            self.fs_config.test_directory_name,
        )
        # Do not clean output directory if the user wants to replay device prints from a previous run
        if not self.debugger_replay:
            shutil.rmtree(self.fs_config.artifacts_output_directory_path, ignore_errors=True)
        os.makedirs(self.fs_config.artifacts_output_directory_path, exist_ok=True)

        self.collector.set_output_dir(self.fs_config.artifacts_output_directory_path)
