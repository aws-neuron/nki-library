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
from .common_dataclasses import (
    INF_ARTIFACT_DIR_NAME,
    KernelArgs,
    LazyGoldenGenerator,
    NeuronDeviceInfo,
    PerRankLazyInputGenerator,
    Platforms,
    SeparationPassMode,
    TraceMode,
)
from .determinism_checker import DeterminismChecker
from .exceptions import CompilationException, InferenceException, TestStatus, ValidationException
from .host_management import Host, HostManager
from .metrics_collector import IMetricsCollector, MetricName
from .metrics_emitter import IMetricsEmitter, OutputMode
from .output_validator import OutputValidator
from .param_extractor import normalize_params_with_type_hints
from .perf_analysis_private import analyze_trace
from .profiler_utils import NEURON_RT_ENABLE_DGE_NOTIFICATIONS, ProfilerCommands, extract_and_filter_output_files
from .ranged_test_harness import is_in_negative_test_context


@dataclass
class FilesystemArgs:
    base_output_directory_path: str
    host_manager: HostManager
    skip_remote_cleanup: bool
    force_local_cleanup: bool = False
    test_directory_name: Optional[str] = None
    artifacts_output_directory_path: Optional[str] = None


def run_separated_perf_analysis(test_dir: str, target_instance_family: str, profiled_file: str = "ntff_detailed.json"):
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
    test_dir = Path(test_dir)
    try:
        analyze_trace(
            input_file=test_dir / "perf_sim_at_end_trace.nc00_sg00.sg0000.Block1.json",
            profiled_file=test_dir / INF_ARTIFACT_DIR_NAME / profiled_file,
            subgraph='sg00',
            base_dir=test_dir,
            output_filename='analysis_nc00.log',
            target_instance_family=target_instance_family,
        )
        analyze_trace(
            input_file=test_dir / "perf_sim_at_end_trace.nc01_sg00.sg0000.Block1.json",
            profiled_file=test_dir / INF_ARTIFACT_DIR_NAME / profiled_file,
            subgraph='sg01',
            base_dir=test_dir,
            output_filename='analysis_nc01.log',
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
        emitter: IMetricsEmitter,
        perf_analysis_enabled: bool = False,
    ):
        self.fs_config: FilesystemArgs = FilesystemArgs(
            base_output_directory_path=feature_flag_helper.resolve_base_output_directory(config),
            host_manager=host_manager,
            skip_remote_cleanup=feature_flag_helper.get_feature_flag(config, "skip_remote_cleanup"),
            force_local_cleanup=feature_flag_helper.get_feature_flag(config, "force_local_cleanup", False),
        )
        self.trace_mode: TraceMode = trace_mode
        self.emitter = emitter
        self.kernel_under_test: Optional[KernelArgs] = None
        self.perf_analysis_enabled: bool = perf_analysis_enabled

        self.profiler_binary_path: str = self.__get_neuron_binary_path_for__(config, "neuron-profile")
        self.neuron_ls_binary_path: str = self.__get_neuron_binary_path_for__(config, "neuron-ls")
        self.enable_kernel_debugging: bool = feature_flag_helper.get_feature_flag(config, "debug_kernels", False)
        self.enable_dge_notifs: bool = feature_flag_helper.get_feature_flag(config, "enable_dge_notifs", False)
        self.enable_validation_histograms: bool = feature_flag_helper.get_feature_flag(
            config, "validation_histograms", False
        )

    def execute(self, kernel_under_test: KernelArgs):
        kernel_under_test.compiler_input.enable_debugging = self.enable_kernel_debugging
        # Assign current emitter to kernel_args
        kernel_under_test.emitter = self.emitter

        # Inject perf sim backend option when perf analysis is enabled
        if self.perf_analysis_enabled:
            perf_sim_flag = "--internal-backend-options=--enable-perf-sim"
            if perf_sim_flag not in kernel_under_test.compiler_input.additional_cmd_args:
                kernel_under_test.compiler_input.additional_cmd_args.append(perf_sim_flag)

        self.__prepare_output_directory()
        assert self.fs_config.artifacts_output_directory_path

        # Initialize MetricsCollector for this test
        test_name = feature_flag_helper.derive_pytest_test_id()
        collector = self.emitter.get_collector()
        collector.set_test_name(test_name)
        # Start timing right before compilation stage
        collector.start_test()

        # Normalize pytest-captured params using kernel function's type hints
        # This converts integers to enum names for cleaner dashboard display
        params = collector.get_kernel_params()
        if params:
            normalized_params = normalize_params_with_type_hints(params, kernel_under_test.kernel_func)
            collector.set_kernel_params(normalized_params)

        status = TestStatus.SUCCESS

        try:
            # Run Compilation
            self._run_compilation(collector, kernel_under_test)

            # Return early if trace-only or compile-only mode
            if self.trace_mode in (TraceMode.TraceOnly, TraceMode.CompileOnly):
                return

            # Create inputs
            with collector.timer(MetricName.INPUT_DUMP_TIME):
                input_file_paths = self.__dump_kernel_inputs__(
                    self.fs_config.artifacts_output_directory_path, kernel_under_test
                )
            # Run Inference
            local_artifact_download_path = self._run_inference(kernel_under_test, input_file_paths)

            # Run performance analysis if perf analysis is enabled
            if self.perf_analysis_enabled:
                logging.info("Running performance analysis")
                run_separated_perf_analysis(
                    self.fs_config.artifacts_output_directory_path,
                    kernel_under_test.compiler_input.platform_target.value,
                )
                if kernel_under_test.compiler_input.separation_pass_mode != SeparationPassMode.NONE:
                    self._record_separation_pass_metrics(collector, self.fs_config.artifacts_output_directory_path)

            # Run validation
            self._run_validation(
                kernel_under_test,
                local_artifact_download_path,
            )

        except (CompilationException, InferenceException, ValidationException) as e:
            # Use EXPECTED_FAILURE for negative tests, otherwise use actual failure status
            status = TestStatus.EXPECTED_FAILURE if is_in_negative_test_context() else e.status
            raise
        finally:
            collector.add_dimension(
                {
                    "TestName": test_name,
                    # Use KERNEL_NAME env var (set by CI) if available, otherwise fall back to function name
                    "KernelName": os.environ.get("KERNEL_NAME") or kernel_under_test.kernel_func.__name__,
                    "Target": kernel_under_test.compiler_input.platform_target.value,
                    "LNCCores": str(kernel_under_test.compiler_input.logical_nc_config),
                    "Status": status.value,
                    "IsSuccessful": "true" if status == TestStatus.SUCCESS else "false",
                    "TraceMode": self.trace_mode.value,
                }
            )
            with collector.timer(MetricName.ARTIFACT_PARSE_TIME):
                collector.parse_artifacts(
                    artifact_dir=self.fs_config.artifacts_output_directory_path,
                    inference_artifact_dir=INF_ARTIFACT_DIR_NAME,
                )

            # Record -1 for phases that didn't run (must happen AFTER parse_artifacts)
            self._record_missing_phase_metrics(collector)

            # Emitter reads from collector and writes metrics
            self.emitter.emit()

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
            MetricName.INFERENCE_TIME,
            MetricName.VALIDATION_TIME,
            MetricName.SIMULATION_TIME,
            MetricName.HOST_ARCH_VALIDATION_TIME,
            MetricName.INPUT_DUMP_TIME,
            MetricName.ARTIFACT_PARSE_TIME,
            MetricName.DETERMINISM_CHECK_TIME,
            MetricName.VALIDATION_OUTPUT_LOAD_TIME,
            MetricName.VALIDATION_COMPARE_TIME,
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

    def _run_compilation(self, collector: IMetricsCollector, kernel_under_test: KernelArgs) -> None:
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
                if kernel_under_test.compiler_input.enable_birsim:
                    # Dump inputs and golden outputs early for debugging in birsim format
                    self._dump_birsim_artifacts(kernel_under_test)

                trace_kernel(
                    kernel_under_test=kernel_under_test,
                    mode=self.trace_mode,
                    output_directory=self.fs_config.artifacts_output_directory_path,
                )
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

    def _run_inference(self, kernel_under_test: KernelArgs, input_file_paths: dict[str, str]) -> Optional[str]:
        """Run compiled kernel on hardware using neuron-profile."""
        assert self.fs_config.artifacts_output_directory_path

        logging.info(f"Running inference for {self.fs_config.artifacts_output_directory_path=}")

        if self.trace_mode == TraceMode.Simulator:
            return self._run_simulator_inference(kernel_under_test)

        collector = self.emitter.get_collector()

        try:
            with closing(
                self.fs_config.host_manager.get_host_assignment_with_retry(
                    platform_target=kernel_under_test.compiler_input.platform_target,
                    collector=collector,
                )
            ) as host_assignment_generator:
                for host_assignment_attempt in host_assignment_generator:
                    with host_assignment_attempt as execution_host:
                        with collector.timer(MetricName.INFERENCE_TIME_TOTAL):
                            with collector.timer(MetricName.HOST_ARCH_VALIDATION_TIME):
                                self.__confirm_host_arch__(
                                    execution_host,
                                    kernel_under_test.compiler_input.platform_target,
                                )

                            with execution_host.prepare_host(
                                target_directory=self.fs_config.artifacts_output_directory_path,
                                skip_remote_cleanup=self.fs_config.skip_remote_cleanup,
                                collector=collector,
                                force_local_cleanup=self.fs_config.force_local_cleanup,
                            ):
                                return self.__run_profiler_on_host__(
                                    execution_host, kernel_under_test, input_file_paths, collector
                                )
        except Exception as e:
            if isinstance(e, InferenceException):
                raise e
            raise InferenceException(e) from e

    def __run_profiler_on_host__(
        self,
        execution_host,
        kernel_under_test: KernelArgs,
        input_file_paths: dict[str, str],
        collector,
    ) -> str:
        """Run neuron-profile on the prepared host."""
        kernel_input_args = self.__format_profiler_kernel_input_args__(kernel_under_test, input_file_paths)

        env_vars: Optional[dict[str, str]] = kernel_under_test.inference_args.env_vars
        separation_pass_enabled = kernel_under_test.compiler_input.separation_pass_mode != SeparationPassMode.NONE
        if (
            self.enable_kernel_debugging
            or self.enable_dge_notifs
            or separation_pass_enabled
            or self.perf_analysis_enabled
        ):
            if env_vars is None:
                env_vars = {}

            if NEURON_RT_ENABLE_DGE_NOTIFICATIONS not in env_vars:
                env_vars[NEURON_RT_ENABLE_DGE_NOTIFICATIONS] = "1"

        profiler_cmds = ProfilerCommands(
            num_runs=kernel_under_test.inference_args.num_runs,
            profile_all_runs=kernel_under_test.inference_args.profile_all_runs,
            profiler_binary_path=self.profiler_binary_path,
            kernel_input_args=kernel_input_args,
            metrics_enabled=self.emitter.get_metrics_enabled() or separation_pass_enabled,
            collective_ranks=kernel_under_test.inference_args.collective_ranks,
            profile_all_ranks=kernel_under_test.inference_args.profile_all_ranks,
            env_vars=env_vars,
            perf_analysis_enabled=self.perf_analysis_enabled,
        )

        def get_list_of_files_to_copy(stdout: str) -> list[str]:
            # if determinism check or profile all runs is enabled, download all outputs
            download_all = (
                kernel_under_test.inference_args.enable_determinism_check
                or kernel_under_test.inference_args.profile_all_runs
            )

            outputs_to_copy = extract_and_filter_output_files(stdout, download_all)

            # Build complete list of files to copy from remote host
            files_to_copy = []
            files_to_copy.extend(outputs_to_copy)
            files_to_copy.extend(profiler_cmds.expected_ntff_files)
            files_to_copy.append("log-infer.txt")
            files_to_copy.extend(profiler_cmds.expected_profiler_view_json_files)
            files_to_copy.extend(profiler_cmds.expected_detailed_json_files)
            files_to_copy.extend(profiler_cmds.expected_show_session_json_files)
            files_to_copy.append("debug_output")
            return files_to_copy

        return execution_host.execute_command(
            command=f"({profiler_cmds.get_complete_command()}) 2>&1 | tee log-infer.txt",
            target_directory=self.fs_config.artifacts_output_directory_path,
            collective_ranks=kernel_under_test.inference_args.collective_ranks,
            lnc_config=kernel_under_test.compiler_input.logical_nc_config,
            do_copy_artifacts=True,
            get_list_of_files_to_copy=get_list_of_files_to_copy,
            collector=collector,
        )

    def _dump_output_tensors(self, output_tensors: dict[str, np.ndarray]) -> str:
        """Dump output tensors to inference artifact directory.

        Returns path to output directory.
        """
        output_path = os.path.join(self.fs_config.artifacts_output_directory_path, INF_ARTIFACT_DIR_NAME)
        self.__dump_tensors__(output_path, output_tensors, lambda name: name)
        return output_path

    def _run_simulator_inference(self, kernel_under_test: KernelArgs) -> Optional[str]:
        """Run kernel using NkiCpuSimulator and dump outputs for validation."""
        from .simulation_setup import run_simulator_inference

        collector = self.emitter.get_collector()
        try:
            with collector.timer(MetricName.SIMULATION_TIME):
                output_tensors = run_simulator_inference(kernel_under_test)
            if not output_tensors:
                return None
            return self._dump_output_tensors(output_tensors)
        except Exception as e:
            raise InferenceException(str(e)) from e

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
        collector = self.emitter.get_collector()
        assert self.fs_config.artifacts_output_directory_path
        with collector.timer(MetricName.VALIDATION_TIME):
            try:
                abs_out_file_paths = []
                output_path = pathlib.Path(local_artifact_download_path)
                for file_name in output_path.iterdir():
                    if file_name.is_file():
                        abs_out_file_paths.append(file_name.absolute().as_posix())
                    elif file_name.is_dir() and file_name.name.startswith('output_worker_'):
                        # Include files from per-rank output directories
                        for rank_file in file_name.iterdir():
                            if rank_file.is_file():
                                abs_out_file_paths.append(rank_file.absolute().as_posix())

                validation_log_filepath = os.path.join(
                    self.fs_config.artifacts_output_directory_path,
                    INF_ARTIFACT_DIR_NAME,
                    'log-validate.txt',
                )

                # Run determinism check if enabled and not simulating
                if kernel_under_test.inference_args.enable_determinism_check:
                    if self.trace_mode == TraceMode.Simulator:
                        logging.warning("Skipping determinism check; simulation does not produce multiple outputs.")
                    else:
                        # Build list of paths to check (per-rank for collectives, single path otherwise)
                        if kernel_under_test.inference_args.collective_ranks > 1:
                            check_paths = [
                                (rank, os.path.join(local_artifact_download_path, f"output_worker_{rank}"))
                                for rank in range(kernel_under_test.inference_args.collective_ranks)
                            ]
                        else:
                            check_paths = [(None, local_artifact_download_path)]

                        for rank, path in check_paths:
                            checker = DeterminismChecker(
                                kernel_under_test,
                                path,
                                kernel_under_test.inference_args.num_runs,
                                collector,
                                logfile_path=validation_log_filepath,
                                rank_id=rank,
                            )
                            checker.check()

                OutputValidator(
                    kernel_under_test,
                    abs_out_file_paths,
                ).validate(logfile_path=validation_log_filepath, enable_histograms=self.enable_validation_histograms)

            except Exception as e:
                raise ValidationException(str(e)) from e

    def __dump_simulation_artifacts__(
        self, base_path: str, kernel_output: list[Any], golden_output: dict[str, Any]
    ) -> str:
        assert (
            len(kernel_output) == 1 and len(golden_output) == 1
        ), "Simulator does not label outputs with tensor names, so it's ambigious when more than a single output is present"

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

        if platform_target in (Platforms.TRN2, Platforms.TRN3, Platforms.TRN3_A0):
            assert sum(neuron_info.nc_count for neuron_info in attached_neuron_devices) in [
                4,
                8,
                64,
                128,
            ]
        else:
            raise Exception(f"{platform_target} is currently unsupported by this test framework!")

    def __format_profiler_kernel_input_args__(self, kernel_input: KernelArgs, input_file_paths: dict[str, str]) -> str:
        """Format kernel input arguments for neuron-profile command.

        Args:
            kernel_input: Kernel arguments
            input_file_paths: Dict mapping arg names to file paths
                For single input: {"input": "/path/inp-input-000.bin", "weights": "/path/inp-weights-000.bin"}
                For per-rank inputs: {"--multi-input": "4rank_inputs.txt"}

        Returns:
            Formatted string for neuron-profile command:
                For single input: "input inp-input-000.bin weights inp-weights-000.bin"
                For per-rank inputs: "--multi-input 4rank_inputs.txt"
        """
        if kernel_input.kernel_input is None:
            return ""
        return " ".join(f"{arg_name} {os.path.basename(file_path)}" for arg_name, file_path in input_file_paths.items())

    def __dump_tensors__(
        self,
        target_directory: str,
        tensors: dict[str, Any],
        name_fn: Callable[[str], str],
        force_numpy_arrays: bool = False,
    ) -> dict[str, str]:
        """
        Dump tensors to target directory using provided naming function.

        For numpy arrays, saves as raw .bin files for neuron-profile compatibility.
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
                    # Save as raw binary for neuron-profile
                    file_name = name_fn(name) + ".bin"
                    file_path = os.path.join(target_directory, file_name)
                    with open(file_path, 'wb') as f:
                        f.write(value.tobytes())

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
        for nc_idx in range(logical_nc_config):
            nc_dir = f"nc{nc_idx:02d}"
            birsim_dir = os.path.join(self.fs_config.artifacts_output_directory_path, nc_dir, "sg00")
            # Birsim requires naming convention: value_{name}.npy
            if kernel_under_test.kernel_input:
                _ = self.__dump_tensors__(
                    birsim_dir, kernel_under_test.kernel_input, lambda name: f"value_{name}", True
                )
            if kernel_under_test.validation_args:
                if (
                    isinstance(kernel_under_test.validation_args.golden_output, LazyGoldenGenerator)
                    and kernel_under_test.validation_args.golden_output.golden is not None
                ):
                    output_golden = kernel_under_test.validation_args.golden_output.golden
                else:
                    assert False, "Birsim does not support custom validator as golden output! Please disable bir sim or switch to a different golden generator"

                # birsim is looking for files with specific naming pattern. moreover, they have to
                # be numpy files, not binary files.
                _ = self.__dump_tensors__(birsim_dir, output_golden, lambda name: f"value_{name}", True)

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
        Dump per-rank inputs and generate multi-input file for neuron-profile.

        The --multi-input neuron-profile flag is used for collectives with per-rank inputs.
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
        with open(multi_input_file, 'w') as f:
            f.write("\n".join(multi_input_lines) + "\n")

        logging.info(f"Created multi-input file: {multi_input_file}")
        return {"--multi-input": multi_input_filename}

    def __prepare_output_directory(self):
        self.fs_config.test_directory_name = feature_flag_helper.construct_test_output_directory_name()
        self.fs_config.artifacts_output_directory_path = os.path.join(
            self.fs_config.base_output_directory_path,
            self.fs_config.test_directory_name,
        )
        shutil.rmtree(self.fs_config.artifacts_output_directory_path, ignore_errors=True)
        os.makedirs(self.fs_config.artifacts_output_directory_path, exist_ok=True)

        # Set output directory for file mode metrics
        if self.emitter.get_output_mode() == OutputMode.FILE:
            # Create metrics subdirectory and point emitter to it
            metrics_dir = os.path.join(self.fs_config.artifacts_output_directory_path, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            self.emitter.set_output_dir(metrics_dir)
