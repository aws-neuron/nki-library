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
"""
Metrics collector for NKL tests - actively measures and records metrics during test execution.


Architecture:
1. MetricsCollector: In-memory metric storage and measurement
2. MetricsEmitter: Output formatting and delivery
3. Relationship: Emitter reads from Collector (via get_metrics_context())
"""

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, final

import numpy as np
from aws_embedded_metrics.logger.metrics_context import MetricsContext
from typing_extensions import override

from ..utils.exceptions import UnimplementedException


class MetricName:
    """
    Centralized metric name constants.
    """

    # Timing metrics (recorded during test execution)
    GOLDEN_COMPUTATION_TIME = "GoldenComputationTime"
    COMPILATION_TIME = "CompilationTime"
    HOST_LOCK_TIME = "HostLockTime"
    FILE_TRANSFER_TIME = "FileTransferTime"
    INFERENCE_TIME_TOTAL = "InferenceTimeTotal"
    FILE_TRANSFER_NETWORK_TIME = "FileTransferNetworkTime"
    FILE_TRANSFER_COMPRESSION_TIME = "FileTransferCompressionTime"
    FILE_TRANSFER_BYTES_COMPRESSED = "FileTransferBytesCompressed"
    FILE_TRANSFER_BYTES_UNCOMPRESSED = "FileTransferBytesUncompressed"
    PROFILE_JSON_GENERATION_TIME = "ProfileJsonGenerationTime"
    NEURON_PROFILE_CAPTURE_TIME = "NeuronProfileCaptureTime"
    NEURON_PROFILE_SHOW_TIME = "NeuronProfileShowTime"
    CORE_ALLOCATION_TIME = "CoreAllocationTime"
    INFERENCE_TIME = "InferenceTime"
    VALIDATION_TIME = "ValidationTime"
    ELAPSED_ALL_SEC = "ElapsedAllSec"
    DETERMINISM_CHECK_TIME = "DeterminismCheckTime"

    # Profiler metrics (from artifacts)
    MBU_ESTIMATED_PERCENT = "MbuEstimatedPercent"
    PROFILER_MFU = "ProfilerMFU"
    TPB_SG_CYCLES_SUM = "TpbSgCyclesSum"

    # Per-core cycle metrics (dynamically named with zero-based core index suffix)
    CYCLE_OUTLIERS_PREFIX = "CycleOutliers_CIdx_"
    CYCLE_QCD_PREFIX = "CycleQCD_CIdx_"

    # Compilation metrics (from artifacts)
    TPB_COUNT = "TPBCount"

    # Validation metrics
    ACCURACY_HW = "AccuracyHw"
    IS_VALIDATION_SKIPPED = "IsValidationSkipped"

    # Host management metrics
    FAILED_HOSTS_COUNT = "FailedHostsCount"


class IMetricsCollector(ABC):
    """
    Collects metrics during test execution using timers and measurements.
    Example Usage:
        collector = MetricsCollector()

        # Start test
        collector.start_test()

        # Measure compilation
        with collector.timer(MetricName.COMPILATION_TIME):
            compile_kernel()

        # Record validation result
        collector.record_metric(MetricName.ACCURACY_HW, 0.001, unit="Percent")

        # Set dimensions
        collector.set_dimensions({"TestName": "test_rmsnorm", "Target": "trn2"})
    """

    @abstractmethod
    def set_namespace(self, namespace: str) -> None:
        raise UnimplementedException()

    @abstractmethod
    def start_test(self) -> None:
        raise UnimplementedException()

    @abstractmethod
    def timer(self, name: str) -> "_TimerContext":
        """
        Context manager for timing a code block.

        Usage:
            with collector.timer(MetricName.COMPILATION_TIME):
                compile_kernel()
        """
        raise UnimplementedException()

    @abstractmethod
    def record_timer(self, name: str, duration_seconds: float) -> None:
        """
        Record a timed duration

        Args:
            name: Timer name
            duration_seconds: Duration in seconds
        """
        raise UnimplementedException()

    @abstractmethod
    def record_metric(self, name: str, value: float, unit: str = "None") -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            unit: CloudWatch unit
        """
        raise UnimplementedException()

    def has_metric(self, name: str) -> bool:
        """
        Check if a metric has been recorded.

        Args:
            name: Metric name to check

        Returns:
            True if metric exists, False otherwise
        """
        raise UnimplementedException()

    @abstractmethod
    def get_finalized_metrics_context(self) -> MetricsContext | None:
        """
        Finalize and return the metrics context for emission.
        Returns:
            MetricsContext containing all collected metrics
        """
        raise UnimplementedException()

    @abstractmethod
    def parse_artifacts(self, artifact_dir: str, inference_artifact_dir: str) -> None:
        """
        Parse test artifacts to extract metrics (compilation time, latency, MBU, etc.)
        """
        raise UnimplementedException()

    @abstractmethod
    def set_test_name(self, test_name: str) -> None:
        raise UnimplementedException()

    @abstractmethod
    def match_and_add_metadata_dimensions(
        self, test_metadata_key: dict[str, Any], metadata_list: list[dict[str, Any]]
    ) -> None:
        raise UnimplementedException()

    @abstractmethod
    def add_dimension(self, dimensions: dict[str, str]) -> None:
        raise UnimplementedException()

    @abstractmethod
    def set_kernel_params(self, params: dict[str, Any]) -> None:
        """
        Store kernel parameters for metrics emission.

        Args:
            params: Dict of kernel parameters (scalars only, no tensors)
        """
        raise UnimplementedException()

    @abstractmethod
    def get_kernel_params(self) -> dict[str, Any]:
        """
        Get stored kernel parameters.
        """
        raise UnimplementedException()


@dataclass
class MetricsCollector(IMetricsCollector):
    """
    Collects metrics during test execution using timers and measurements.
    Example Usage:
        collector = MetricsCollector()

        # Start test
        collector.start_test()

        # Measure compilation
        with collector.timer(MetricName.COMPILATION_TIME):
            compile_kernel()

        # Record validation result
        collector.record_metric(MetricName.ACCURACY_HW, 0.001, unit="Percent")

        # Set dimensions
        collector.set_dimensions({"TestName": "test_rmsnorm", "Target": "trn2"})
    """

    test_name: str = ""
    """Test name for logging purposes"""

    dimensions: dict[str, str] = field(default_factory=dict)
    """Common dimensions to apply to all metrics (e.g., TestName, Target, LNCCores)"""

    kernel_params: dict[str, Any] = field(default_factory=dict)
    """Kernel test parameters (scalars only) for metrics emission"""

    _start_time: float = field(default=0.0, init=False)
    """Test start timestamp"""

    _metrics_context: MetricsContext | None = field(default=None, init=False)
    """AWS EMF MetricsContext for metric storage"""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__), init=False)

    def __post_init__(self):
        """Initialize MetricsContext immediately so it's available for early measurements in test code (e.g. golden compilation time)"""
        self._metrics_context = MetricsContext.empty()

    @override
    def set_namespace(self, namespace: str) -> None:
        """
        Set the CloudWatch namespace for metrics.
        """
        assert self._metrics_context
        self._metrics_context.namespace = namespace

    @override
    def start_test(self) -> None:
        """Mark the start of test execution (record start time)."""
        self._start_time = time.time()

    @override
    def timer(self, name: str) -> "_TimerContext":
        """
        Context manager for timing a code block.

        Usage:
            with collector.timer(MetricName.COMPILATION_TIME):
                compile_kernel()
        """
        return _TimerContext(self, name)

    @override
    def record_timer(self, name: str, duration_seconds: float) -> None:
        """
        Record a timed duration

        Args:
            name: Timer name
            duration_seconds: Duration in seconds
        """
        assert self._metrics_context
        self._metrics_context.put_metric(name, duration_seconds, "Seconds")

    @override
    def record_metric(self, name: str, value: float, unit: str = "None") -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            unit: CloudWatch unit
        """
        assert self._metrics_context
        self._metrics_context.put_metric(name, value, unit)

    @override
    def add_dimension(self, dimensions: dict[str, str]) -> None:
        """
        Add dimension(s) to metrics context.

        Args:
            dimensions: Dict of dimensions to add (e.g., {"TestName": "test_foo", "Target": "trn2"})

        Examples:
            collector.add_dimension({"TestName": "test_foo", "Target": "trn2"})
            collector.add_dimension({"ConfigSupported": "true"})
        """
        self.dimensions.update(dimensions)

    @override
    def match_and_add_metadata_dimensions(
        self, test_metadata_key: dict[str, Any], metadata_list: list[dict[str, Any]]
    ) -> None:
        """
        Match test configuration to metadata and add model dimensions.

        Searches metadata_list for entries where test_settings matches test_metadata_key.
        Adds all keys from matched model_settings as dimensions
        """
        for entry in metadata_list:
            test_settings = entry.get("test_settings", {})
            # Check if all keys in test_metadata_key match test_settings
            if all(test_settings.get(k) == v for k, v in test_metadata_key.items()):
                model_settings = entry.get("model_settings", {})
                # Add all model_settings as dimensions
                for key, value in model_settings.items():
                    if value is not None:
                        self.add_dimension({key: str(value)})
                return  # Found match, stop searching

    @override
    def has_metric(self, name: str) -> bool:
        """
        Check if a metric has been recorded.

        Args:
            name: Metric name to check

        Returns:
            True if metric exists, False otherwise
        """
        assert self._metrics_context
        return name in self._metrics_context.metrics

    @override
    def get_finalized_metrics_context(self) -> MetricsContext | None:
        """
        Finalize and return the metrics context for emission.
        Returns:
            MetricsContext containing all collected metrics
        """
        assert self._metrics_context
        # Add dimensions
        if self.dimensions:
            self._metrics_context.set_dimensions([self.dimensions])

        # Add total elapsed time
        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            self._metrics_context.put_metric(MetricName.ELAPSED_ALL_SEC, elapsed, "Seconds")

        return self._metrics_context

    @override
    def parse_artifacts(self, artifact_dir: str, inference_artifact_dir: str) -> None:
        """
        Parse test artifacts to extract metrics (compilation time, latency, MBU, etc.)
        """
        artifact_path = Path(artifact_dir)

        # Parse info.json for NEFF metadata
        info_json = artifact_path / "info.json"
        self._parse_neff_info(info_json)

        # Parse log-infer.txt for timing metrics
        log_infer = artifact_path / inference_artifact_dir / "log-infer.txt"
        if log_infer.exists():
            self._parse_log_infer(log_infer)
        else:
            self.logger.warning("log-infer.txt not found")

        # Parse show-session JSON files for cycle counts
        self._parse_show_session_json_files(artifact_path / inference_artifact_dir)

        # Parse ntff.json for profiler metrics (MBU, MFU, latency)
        ntff_path = artifact_path / inference_artifact_dir / "ntff.json"
        if ntff_path.exists():
            self._parse_ntff_json(ntff_path)
        else:
            self.logger.debug(f"ntff.json not found")
            # Always record profiler metrics for consistency
            self.record_metric(MetricName.MBU_ESTIMATED_PERCENT, -1.0, "Percent")
            self.record_metric(MetricName.PROFILER_MFU, -1.0, "Percent")

    @override
    def set_test_name(self, test_name: str) -> None:
        self.test_name = test_name

    @override
    def set_kernel_params(self, params: dict[str, Any]) -> None:
        self.kernel_params = params

    @override
    def get_kernel_params(self) -> dict[str, Any]:
        return self.kernel_params

    def _parse_neff_info(self, info_path) -> None:
        """Parse info.json for NEFF metadata."""
        try:
            with open(info_path, "r") as f:
                info = json.load(f)

            if "num_tpb" in info:
                self.record_metric(MetricName.TPB_COUNT, float(info["num_tpb"]), "Count")
        except Exception as e:
            self.logger.warning(f"Failed to parse info.json: {e}")
            self.record_metric(MetricName.TPB_COUNT, -1.0, "Count")

    def _parse_and_record_time_metric(self, log_content: str, metric_name: str, pattern_prefix: str) -> None:
        """Parse a timing metric from log content and record it. Uses findall to capture the last run."""
        matches = re.findall(rf"{pattern_prefix}(?:_RUN_\d+)?:\s+(\d+\.?\d*)", log_content)
        if matches:
            # Take the last run's timing
            time_value = float(matches[-1])
            self.record_metric(metric_name, time_value, "Seconds")

    def _parse_log_infer(self, log_path) -> None:
        """Parse log-infer.txt for timing metrics."""
        try:
            with open(log_path, "r") as f:
                log_content = f.read()

            # Parse timing metrics from neuron-profile commands
            timing_metrics = [
                (MetricName.PROFILE_JSON_GENERATION_TIME, "PROFILE_JSON_GENERATION_TIME"),
                (MetricName.NEURON_PROFILE_CAPTURE_TIME, "NEURON_PROFILE_CAPTURE_TIME"),
                (MetricName.NEURON_PROFILE_SHOW_TIME, "NEURON_PROFILE_SHOW_TIME"),
            ]
            for metric_name, pattern in timing_metrics:
                self._parse_and_record_time_metric(log_content, metric_name, pattern)

        except Exception as e:
            self.logger.warning(f"Failed to parse log-infer.txt: {e}")

    def _parse_show_session_json_files(self, inference_artifact_dir) -> None:
        """
        Parse show-session JSON files to extract cycle counts per physical core.

        Collects cycle counts from all runs, groups by physical core (NC), applies IQR
        filtering per core, then takes the max of the filtered averages.
        Records -1 if any errors occur during parsing or calculation.
        """
        try:
            inference_path = Path(inference_artifact_dir)

            # Find all show_session_*.json files
            show_session_files = sorted(inference_path.glob("show_session_*.json"))
            if not show_session_files:
                self.logger.warning(f"No show_session_*.json files found in {inference_artifact_dir}")
                self.record_metric(MetricName.TPB_SG_CYCLES_SUM, -1.0, "Count")
                return

            # Collect cycle counts per physical core across all runs
            # Key: physical core ID (NC), Value: list of cycle counts from each run
            cycles_per_core: dict[int, list[float]] = {}

            for json_file in show_session_files:
                with open(json_file, "r") as f:
                    data = json.load(f)

                # Extract cycle counts from each graph (physical core)
                for neff_node in data.get("NeffNodes", []):
                    for graph in neff_node.get("NodeInfo", {}).get("Graphs", []):
                        nc = graph.get("NC")
                        cycle_count = graph.get("CycleCount")
                        if nc is None:
                            self.logger.warning(f"Missing NC (physical core ID) in graph in {json_file}")
                            continue
                        if cycle_count is None:
                            self.logger.warning(f"Missing CycleCount for NC={nc} in {json_file}")
                            continue
                        if nc not in cycles_per_core:
                            cycles_per_core[nc] = []
                        cycles_per_core[nc].append(float(cycle_count))

            if not cycles_per_core:
                self.logger.warning(f"No cycle counts found in show-session JSON files in {inference_artifact_dir}")
                self.record_metric(MetricName.TPB_SG_CYCLES_SUM, -1.0, "Count")
                return

            # Apply IQR filtering per core and compute filtered average for each
            # Use zero-based index for metric names (sorted by core ID)
            filtered_averages: dict[int, float] = {}
            for idx, (nc, cycles_list) in enumerate(sorted(cycles_per_core.items())):
                filtered_avg, outliers_count, qcd = self._iqr_filtered_average(cycles_list, core_id=nc)
                filtered_averages[nc] = filtered_avg
                self.logger.info(f"Physical core NC={nc} (index {idx}): filtered average = {filtered_avg:.0f}")

                # Record per-core metrics using zero-based index
                self.record_metric(f"{MetricName.CYCLE_OUTLIERS_PREFIX}{idx}", float(outliers_count), "Count")
                self.record_metric(f"{MetricName.CYCLE_QCD_PREFIX}{idx}", qcd, "None")

            # Final cycle count is the max of all filtered averages
            max_cycles = max(filtered_averages.values())
            self.logger.info(f"Final cycle count (max of filtered averages): {max_cycles:.0f}")
            self.record_metric(MetricName.TPB_SG_CYCLES_SUM, max_cycles, "Count")

        except Exception as e:
            self.logger.warning(f"Failed to parse show-session JSON files: {e}")
            self.record_metric(MetricName.TPB_SG_CYCLES_SUM, -1.0, "Count")

    def _get_hardware_specs(self, target_instance_family: str) -> tuple[float, int]:
        """
        Get hardware specifications for a target instance family.

        Args:
            target_instance_family: Instance family

        Returns:
            Tuple of (pe_frequency_hz, tensor_engine_size)

        """
        if any(target_instance_family.startswith(family) for family in ("trn2", "trn3")):
            peFreq = 2.4e9
            tensor_engine_size = 128

            return (peFreq, tensor_engine_size)
        else:
            raise ValueError(f"Unknown target instance family: {target_instance_family}")

    def _parse_ntff_json(self, ntff_path) -> None:
        """Parse ntff.json for profiler metrics"""
        try:
            # Get hardware specs for the target platform
            target = self.dimensions["Target"]
            pe_freq, tensor_engine_size = self._get_hardware_specs(target)

            with open(ntff_path, "r") as f:
                profiler_data = json.load(f)
            profiler_summary = profiler_data.get("summary", [{}])[0]

            infer_sec = profiler_summary.get("total_time")

            # Record InferenceTime metric
            if infer_sec is not None and infer_sec > 0:
                self.record_metric(MetricName.INFERENCE_TIME, float(infer_sec), "Seconds")
            else:
                self.record_metric(MetricName.INFERENCE_TIME, -1.0, "Seconds")

            # Getting the MBU from profiler directly
            mbu = (
                profiler_summary.get("mbu_estimated_percent")
                if profiler_summary.get("mbu_estimated_percent") >= 0
                else -1
            )
            self.record_metric(MetricName.MBU_ESTIMATED_PERCENT, float(mbu * 100), "Percent")

            # MFU Calculation
            hw_flops = float(profiler_summary.get("hardware_flops") or 0)
            tr_flops = float(profiler_summary.get("transpose_flops") or 0)

            if not infer_sec or infer_sec <= 0 or hw_flops < 0:
                self.record_metric(MetricName.PROFILER_MFU, -1.0, "None")
                self.logger.warning("Invalid data for MFU calculation")
                return

            num_lnc = int(self.dimensions["LNCCores"])
            actual_flops = hw_flops - tr_flops

            pe_ops_per_sec = (
                2 * tensor_engine_size * tensor_engine_size * pe_freq  # 2 ops per PE cycle
            )
            max_flops_per_core = pe_ops_per_sec * infer_sec
            mfu = actual_flops / (max_flops_per_core * num_lnc) if max_flops_per_core > 0 else -1

            self.record_metric(MetricName.PROFILER_MFU, float(mfu * 100), "Percent")

        except Exception as e:
            self.logger.warning(f"Failed to parse ntff.json: {e}")
            self.record_metric(MetricName.MBU_ESTIMATED_PERCENT, -1.0, "Percent")
            self.record_metric(MetricName.PROFILER_MFU, -1.0, "Percent")

    def _iqr_filtered_average(self, cycle_list, core_id: int | None = None) -> tuple[float, int, float]:
        """
        Filter outliers from a list of cycle counts using the Interquartile Range (IQR) method
        and return the average of the remaining items along with statistics.

        Args:
          cycle_list: List of cycle count values
          core_id: Optional physical core ID for logging context

        Returns:
          tuple: (filtered_average, outliers_count, qcd)
            - filtered_average: Average cycle count after removing outliers
            - outliers_count: Number of outliers removed
            - qcd: Quartile Coefficient of Dispersion (IQR / Median)
        """
        core_prefix = f"NC={core_id}: " if core_id is not None else ""
        self.logger.info(f'{core_prefix}Measured cycle counts: {cycle_list}')

        # Calculate quartiles, IQR, and median
        Q1 = np.percentile(cycle_list, 25)
        Q3 = np.percentile(cycle_list, 75)
        IQR = Q3 - Q1
        median = np.median(cycle_list)

        # Calculate QCD (Quartile Coefficient of Dispersion)
        qcd = ((IQR / median) * 100.0) if median > 0 else -1.0

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter outliers
        filtered = [x for x in cycle_list if lower_bound <= x <= upper_bound]
        outliers_count = len(cycle_list) - len(filtered)

        # Calculate average of filtered values, fallback to original if all filtered out
        if filtered:
            avg_cycles = sum(filtered) // len(filtered)
        else:
            avg_cycles = sum(cycle_list) // len(cycle_list)

        self.logger.info(
            f'{core_prefix}Filtered out {outliers_count} outliers (bounds: {lower_bound:.2f} - {upper_bound:.2f}), '
            f'QCD: {qcd:.6f}'
        )

        return avg_cycles, outliers_count, qcd


@final
class _TimerContext:
    """Context manager for timing code blocks."""

    def __init__(self, collector: IMetricsCollector, name: str):
        self.collector = collector
        self.name = name
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.collector.record_timer(self.name, duration)
        return False  # Don't suppress exceptions


class NoopMetricsCollector(IMetricsCollector):
    """
    Metrics Collector that does nothing. Useful when metrics emissions needs to be disabled
    """

    @override
    def set_namespace(self, namespace: str) -> None:
        pass

    @override
    def start_test(self) -> None:
        pass

    @override
    def timer(self, name: str) -> "_TimerContext":
        """
        Context manager for timing a code block.

        Usage:
            with collector.timer(MetricName.COMPILATION_TIME):
                compile_kernel()
        """
        return _TimerContext(self, name)

    @override
    def record_timer(self, name: str, duration_seconds: float) -> None:
        """
        Record a timed duration

        Args:
            name: Timer name
            duration_seconds: Duration in seconds
        """
        pass

    @override
    def record_metric(self, name: str, value: float, unit: str = "None") -> None:
        """
        Record a metric value.

        Args:
            name: Metric name
            value: Metric value
            unit: CloudWatch unit
        """
        pass

    @override
    def add_dimension(self, dimensions: dict[str, str]) -> None:
        """No-op: add dimension(s) to metrics context"""
        pass

    @override
    def match_and_add_metadata_dimensions(
        self, test_metadata_key: dict[str, Any], metadata_list: list[dict[str, Any]]
    ) -> None:
        """
        No-op implementation
        """
        pass

    @override
    def has_metric(self, name: str) -> bool:
        """
        Check if a metric has been recorded.

        Args:
            name: Metric name to check

        Returns:
            True if metric exists, False otherwise
        """
        return False

    @override
    def get_finalized_metrics_context(self) -> MetricsContext | None:
        """
        Finalize and return the metrics context for emission.
        Returns:
            MetricsContext containing all collected metrics
        """
        return None

    @override
    def parse_artifacts(self, artifact_dir: str, inference_artifact_dir: str) -> None:
        """
        Parse test artifacts to extract metrics (compilation time, latency, MBU, etc.)
        """
        pass

    @override
    def set_test_name(self, test_name: str) -> None:
        pass

    @override
    def set_kernel_params(self, params: dict[str, Any]) -> None:
        pass

    @override
    def get_kernel_params(self) -> dict[str, Any]:
        return {}
