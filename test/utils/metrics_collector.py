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

from .exceptions import UnimplementedException
from .metadata_loader import match_model_config_id


class MetricName:
    """
    Centralized metric name constants.
    """

    # ==========================================================================
    # Top-level timing metrics
    # ==========================================================================

    ELAPSED_ALL_SEC = "ElapsedAllSec"
    COMPILATION_TIME = "CompilationTime"
    # Wall-clock time spent in the NKI front-end (compile_to_bir): Python->MLIR
    # trace, MLIR emission/serialization, and first-call backend imports. This
    # runs on every execution regardless of the NEFF cache, since the cache key
    # is derived from the traced BIR. Distinct from MLIR_TO_BIR_TIME, which is
    # the compiler's self-reported inner MLIR->BIR pass time.
    FRONTEND_TRACE_TIME = "FrontendTraceTime"
    MLIR_TO_BIR_TIME = "MlirToBirTime"
    BIR_TO_NEFF_TIME = "BirToNeffTime"
    INFERENCE_TIME_TOTAL = "InferenceTimeTotal"
    VALIDATION_TIME = "ValidationTime"
    INPUT_DUMP_TIME = "InputDumpTime"
    ARTIFACT_PARSE_TIME = "ArtifactParseTime"

    # ==========================================================================
    # Simulation metrics
    # ==========================================================================
    SIMULATION_TIME = "SimulationTime"

    # ==========================================================================
    # Host management metrics
    # ==========================================================================
    HOST_LOCK_TIME = "HostLockTime"
    HOST_ARCH_VALIDATION_TIME = "HostArchValidationTime"
    CORE_ALLOCATION_TIME = "CoreAllocationTime"
    # Core allocation sub-metrics (components of CORE_ALLOCATION_TIME)
    CORE_LOCK_INIT_TIME = "CoreLockInitTime"
    CORE_LOCK_DEPLOY_TIME = "CoreLockDeployTime"
    CORE_LOCK_ACQUIRE_TIME = "CoreLockAcquireTime"
    CORE_LOCK_VERSION_CHECK_TIME = "CoreLockVersionCheckTime"
    CORE_LOCK_RELEASE_FAILED_COUNT = "CoreLockReleaseFailedCount"
    # Core lock contention metrics
    CORE_LOCK_NO_CORES_COUNT = "CoreLockNoCoresCount"
    CORE_LOCK_CONTENTION_WAIT_TIME = "CoreLockContentionWaitTime"
    CORE_LOCK_HOLD_TIME = "CoreLockHoldTime"
    FAILED_HOSTS_COUNT = "FailedHostsCount"
    RECOVERABLE_HOST_WAIT_TIME = "RecoverableHostWaitTime"
    FLEET_STARTUP_WAIT_TIME = "FleetStartupWaitTime"
    INSTANCE_TYPE = "InstanceType"
    # Core lock FIFO-queue fairness metrics
    CORE_LOCK_QUEUE_WAIT_TIME = "CoreLockQueueWaitTime"
    CORE_LOCK_QUEUE_POSITION_AT_ENQUEUE = "CoreLockQueuePositionAtEnqueue"
    CORE_LOCK_ETA_AT_ENQUEUE = "CoreLockEtaAtEnqueue"
    # How much longer the caller actually waited in the queue than the
    # worst-case ETA it was first quoted at enqueue (clamped to >= 0). A
    # non-zero value signals scheduling inefficiency: the optimistic ETA
    # under-estimated the real wait.
    CORE_LOCK_ETA_OVERRUN_TIME = "CoreLockEtaOverrunTime"
    CORE_LOCK_BUMP_COUNT = "CoreLockBumpCount"
    CORE_LOCK_REENQUEUE_COUNT = "CoreLockReenqueueCount"
    CORE_LOCK_HOST_ROTATION_COUNT = "CoreLockHostRotationCount"
    CORE_LOCK_DRAIN_WAIT_TIME = "CoreLockDrainWaitTime"

    # ==========================================================================
    # File transfer metrics
    # ==========================================================================
    FILE_TRANSFER_UPLOAD_TIME = "FileTransferUploadTime"
    FILE_TRANSFER_DOWNLOAD_TIME = "FileTransferDownloadTime"

    # ==========================================================================
    # SFTP file transfer metrics
    # ==========================================================================
    FILE_TRANSFER_COMPRESSION_TIME = "FileTransferCompressionTime"
    FILE_TRANSFER_BYTES_COMPRESSED = "FileTransferBytesCompressed"
    FILE_TRANSFER_BYTES_UNCOMPRESSED = "FileTransferBytesUncompressed"

    # ==========================================================================
    # SFTP transfer timing metrics
    # ==========================================================================
    SFTP_UPLOAD_TIME = "SftpUploadTime"
    SFTP_DOWNLOAD_TIME = "SftpDownloadTime"

    # ==========================================================================
    # S3 transfer timing metrics
    # ==========================================================================
    S3_UPLOAD_LOCAL_ARCHIVE_TIME = "S3UploadLocalArchiveTime"
    S3_UPLOAD_LOCAL_TO_S3_TIME = "S3UploadLocalToS3Time"
    S3_UPLOAD_S3_TO_REMOTE_TIME = "S3UploadS3ToRemoteTime"
    S3_UPLOAD_REMOTE_EXTRACT_TIME = "S3UploadRemoteExtractTime"
    S3_DOWNLOAD_REMOTE_ARCHIVE_TIME = "S3DownloadRemoteArchiveTime"
    S3_DOWNLOAD_REMOTE_TO_S3_TIME = "S3DownloadRemoteToS3Time"
    S3_DOWNLOAD_S3_TO_LOCAL_TIME = "S3DownloadS3ToLocalTime"
    S3_DOWNLOAD_LOCAL_EXTRACT_TIME = "S3DownloadLocalExtractTime"

    # ==========================================================================
    # S3 transfer bytes and rates
    # ==========================================================================
    S3_UPLOAD_BYTES = "S3UploadBytes"
    S3_DOWNLOAD_BYTES = "S3DownloadBytes"
    S3_UPLOAD_LOCAL_TO_S3_RATE = "S3UploadLocalToS3Rate"
    S3_UPLOAD_S3_TO_REMOTE_RATE = "S3UploadS3ToRemoteRate"
    S3_DOWNLOAD_REMOTE_TO_S3_RATE = "S3DownloadRemoteToS3Rate"
    S3_DOWNLOAD_S3_TO_LOCAL_RATE = "S3DownloadS3ToLocalRate"

    # ==========================================================================
    # Neuron profiler timing metrics (parsed from log-infer.txt)
    # ==========================================================================
    NEURON_PROFILE_CAPTURE_TIME = "NeuronProfileCaptureTime"
    NEURON_PROFILE_SHOW_TIME = "NeuronProfileShowTime"
    PROFILE_JSON_GENERATION_TIME = "ProfileJsonGenerationTime"

    # ==========================================================================
    # Profiler performance metrics (parsed from ntff.json)
    # ==========================================================================
    INFERENCE_TIME = "InferenceTime"
    ACTIVE_INFERENCE_TIME = "ActiveInferenceTime"
    ACTIVE_INFERENCE_TIME_OUTLIERS = "ActiveInferenceTimeOutliers"
    ACTIVE_INFERENCE_TIME_QCD = "ActiveInferenceTimeQCD"
    ACTIVE_INFERENCE_TIME_SAMPLES = "ActiveInferenceTimeSamples"
    MBU_ESTIMATED_PERCENT = "MbuEstimatedPercent"
    PROFILER_MFU = "ProfilerMFU"

    # ==========================================================================
    # Cycle count metrics (parsed from show_session_*.json)
    # ==========================================================================
    TPB_SG_CYCLES_SUM = "TpbSgCyclesSum"
    CYCLE_OUTLIERS_PREFIX = "CycleOutliers_CIdx_"
    CYCLE_QCD_PREFIX = "CycleQCD_CIdx_"

    # ==========================================================================
    # Compilation metrics (parsed from info.json)
    # ==========================================================================
    TPB_COUNT = "TPBCount"

    # ==========================================================================
    # Validation metrics
    # ==========================================================================
    # Total wall time to acquire the golden at validation time, from any source
    # (a fresh reference compute, or a torch-ref cache download). Parent of
    # GOLDEN_COMPUTATION_TIME — do not sum the two.
    GOLDEN_ACQUISITION_TIME = "GoldenAcquisitionTime"
    # Wall time of the reference compute itself. Recorded only when the golden is
    # actually computed (cache miss or caching disabled); absent on a cache hit.
    GOLDEN_COMPUTATION_TIME = "GoldenComputationTime"
    VALIDATION_OUTPUT_LOAD_TIME = "ValidationOutputLoadTime"
    VALIDATION_COMPARE_TIME = "ValidationCompareTime"
    ACCURACY_HW = "AccuracyHw"
    IS_VALIDATION_SKIPPED = "IsValidationSkipped"
    DETERMINISM_CHECK_TIME = "DeterminismCheckTime"

    # ==========================================================================
    # Separation pass metrics (parsed from analysis_nc*.log)
    # ==========================================================================
    SEPARATED_MEMORY_TIME = "SeparatedMemoryTime"
    SEPARATED_COMPUTE_TIME = "SeparatedComputeTime"

    # ==========================================================================
    # BIR-to-NEFF S3 cache metrics
    # ==========================================================================
    NEFF_CACHE_LOOKUP_TIME = "NeffCacheLookupTime"
    NEFF_CACHE_STORE_TIME = "NeffCacheStoreTime"
    NEFF_CACHE_HIT = "NeffCacheHit"

    TORCH_REF_CACHE_LOOKUP_TIME = "TorchRefCacheLookupTime"
    TORCH_REF_CACHE_STORE_TIME = "TorchRefCacheStoreTime"
    TORCH_REF_CACHE_HIT = "TorchRefCacheHit"

    # ==========================================================================
    # Explorer upload metrics
    # ==========================================================================
    EXPLORER_PROFILE_URL = "ExplorerProfileURL"


# Truncate the captured failure reason to keep it queryable as a doc field.
MAX_FAILURE_REASON_LEN = 240


def sanitize_dimension_value(value: str) -> str:
    """Coerce an arbitrary string into a valid CloudWatch/EMF dimension value.

    The EMF library rejects a dimension value that is non-ASCII or empty/whitespace-only
    (aws_embedded_metrics.validator.validate_dimension_set), and a raised
    InvalidDimensionError aborts the whole metrics emit — which would drop the real
    FailureReason and spam errors. Failure messages routinely contain non-ASCII (e.g. an
    em-dash) and newlines, so normalize here: collapse whitespace, drop non-ASCII, truncate.
    Returns "" if nothing usable remains (caller then skips the dimension)."""
    ascii_only = value.encode("ascii", "ignore").decode("ascii")
    collapsed = " ".join(ascii_only.split())  # also strips newlines/leading/trailing ws
    return collapsed[:MAX_FAILURE_REASON_LEN]


def add_rerun_dimensions(
    collector: "IMetricsCollector",
    attempt_number: int,
    failed: bool,
    failure_reason: str | None = None,
) -> None:
    """Record per-attempt rerun dimensions on the metrics record.

    Adds AttemptNumber (1-based; a rerun is AttemptNumber > 1) and, on a failed
    attempt, a short FailureReason.

    Takes primitives (not pytest objects) so this stays free of pytest types;
    the makereport hook extracts attempt_number/failed/failure_reason.
    """
    collector.add_dimension({"AttemptNumber": str(attempt_number)})
    if failed:
        reason = sanitize_dimension_value(failure_reason or "")
        if reason:  # skip if nothing usable remains — an empty value would fail EMF validation
            collector.add_dimension({"FailureReason": reason})


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

    @property
    @abstractmethod
    def metrics_enabled(self) -> bool:
        """Whether metrics collection is enabled."""
        raise UnimplementedException()

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
    def parse_artifacts(self, artifact_dir: str, inference_artifact_dir: str, target: str) -> None:
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

    @abstractmethod
    def set_pytest_marks(self, marks: list[str]) -> None:
        """Store resolved pytest marker names for metrics emission."""
        raise UnimplementedException()

    @abstractmethod
    def get_pytest_marks(self) -> list[str]:
        """Get resolved pytest marker names."""
        raise UnimplementedException()

    @abstractmethod
    def set_output_dir(self, output_dir: str) -> None:
        """Set the output directory of the test."""
        raise UnimplementedException()

    @abstractmethod
    def get_output_dir(self) -> str | None:
        """Get the output directory of the test."""
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

    pytest_marks: list[str] = field(default_factory=list)
    """Resolved pytest marker names for the test"""

    output_dir: str | None = field(default=None)
    """Output directory of the test"""

    _start_time: float = field(default=0.0, init=False)
    """Test start timestamp"""

    _metrics_context: MetricsContext | None = field(default=None, init=False)
    """AWS EMF MetricsContext for metric storage"""

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__), init=False)

    def __post_init__(self):
        """Initialize MetricsContext immediately so it's available for early measurements in test code (e.g. golden compilation time)"""
        self._metrics_context = MetricsContext.empty()

    @property
    @override
    def metrics_enabled(self) -> bool:
        return True

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

        Searches metadata_list for entries where all keys in test_metadata_key
        match test_settings. On match, computes a ModelConfigId (SHA256 hash)
        for association.
        """
        config_id = match_model_config_id(test_metadata_key, metadata_list)
        if config_id:
            self.add_dimension({"ModelConfigId": config_id})
        else:
            self.logger.warning(f"No model config match found in {len(metadata_list)} entries")

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

        # Add kernel parameters as properties
        if self.kernel_params:
            for key, value in self.kernel_params.items():
                self._metrics_context.set_property(key, value)

        self._metrics_context.set_property("PytestMarks", self.pytest_marks)

        # Add total elapsed time
        if self._start_time > 0:
            elapsed = time.time() - self._start_time
            self._metrics_context.put_metric(MetricName.ELAPSED_ALL_SEC, elapsed, "Seconds")

        return self._metrics_context

    @override
    def parse_artifacts(self, artifact_dir: str, inference_artifact_dir: str, target: str) -> None:
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

        # Parse ntff*.json for profiler metrics (MBU, MFU, latency)
        # Single run produces ntff.json; multiple profiled runs produce ntff_0.json, ntff_1.json, etc.
        # Use the last file to match the profile_all_runs=False behavior (last execution).
        inference_path = artifact_path / inference_artifact_dir
        ntff_path = inference_path / "ntff.json"
        if not ntff_path.exists():
            # Multiple profiled runs: find ntff_N.json files and use the last one
            ntff_numbered = sorted(inference_path.glob("ntff_[0-9]*.json"))
            ntff_path = ntff_numbered[-1] if ntff_numbered else None
        if ntff_path:
            self._parse_ntff_json(ntff_path, target)
        else:
            self.logger.debug("No ntff*.json files found")
            # Always record profiler metrics for consistency
            self.record_metric(MetricName.MBU_ESTIMATED_PERCENT, -1.0, "Percent")
            self.record_metric(MetricName.PROFILER_MFU, -1.0, "Percent")

        # Parse total_exec_time from ntff*.json for ActiveInferenceTime
        self._parse_active_inference_time_from_ntff(artifact_path / inference_artifact_dir)

    @override
    def set_test_name(self, test_name: str) -> None:
        self.test_name = test_name
        self.add_dimension({"TestName": test_name})

    @override
    def set_kernel_params(self, params: dict[str, Any]) -> None:
        self.kernel_params = params

    @override
    def get_kernel_params(self) -> dict[str, Any]:
        return self.kernel_params

    @override
    def set_pytest_marks(self, marks: list[str]) -> None:
        self.pytest_marks = list(marks)

    @override
    def get_pytest_marks(self) -> list[str]:
        return self.pytest_marks

    @override
    def set_output_dir(self, output_dir: str) -> None:
        self.output_dir = output_dir

    @override
    def get_output_dir(self) -> str | None:
        return self.output_dir

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

            # Parse timing metrics from neuron-explorer commands
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
                filtered_averages[nc] = int(filtered_avg)
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

    def _parse_ntff_json(self, ntff_path, target: str) -> None:
        """Parse ntff.json for profiler metrics (summary-json format)."""
        try:
            # Get hardware specs for the target platform
            pe_freq, tensor_engine_size = self._get_hardware_specs(target)

            with open(ntff_path, "r") as f:
                profiler_data = json.load(f)

            # summary-json format: {"n_hash...": {...}} - get first value
            profiler_summary = next(iter(profiler_data.values()), {})

            infer_sec = profiler_summary.get("total_time")

            # Record InferenceTime metric
            if infer_sec is not None and infer_sec > 0:
                self.record_metric(MetricName.INFERENCE_TIME, float(infer_sec), "Seconds")
            else:
                self.record_metric(MetricName.INFERENCE_TIME, -1.0, "Seconds")

            # Getting the MBU from profiler directly
            mbu_estimated = profiler_summary.get("mbu_estimated_percent")
            mbu = mbu_estimated if mbu_estimated is not None and mbu_estimated >= 0 else -1
            self.record_metric(MetricName.MBU_ESTIMATED_PERCENT, float(mbu * 100), "Percent")

            # MFU Calculation
            actual_flops = float(profiler_summary.get("adjusted_hardware_flops") or 0) - float(
                profiler_summary.get("adjusted_transpose_flops") or 0
            )

            if not infer_sec or infer_sec <= 0 or actual_flops < 0:
                self.record_metric(MetricName.PROFILER_MFU, -1.0, "None")
                self.logger.warning("Invalid data for MFU calculation")
                return

            num_lnc = int(self.dimensions["LNCCores"])

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

    def _parse_active_inference_time_from_ntff(self, inference_path) -> None:
        """Extract ActiveInferenceTime from total_exec_time in ntff summary-json files.

        Reads all ntff*.json files, extracts total_exec_time from each, applies
        IQR filtering across samples, and records the filtered average.
        Falls back to -1.0 with a warning if total_exec_time is not found.
        """
        active_inference_time = -1.0
        outliers_count = None
        qcd = None
        num_samples = None
        try:
            inference_dir = Path(inference_path)
            ntff_files = sorted(inference_dir.glob("ntff*.json"))
            if not ntff_files:
                return

            active_times = []
            for ntff_file in ntff_files:
                with open(ntff_file, "r") as f:
                    data = json.load(f)
                summary = next(iter(data.values()), {})
                total_exec = summary.get("total_exec_time")
                if total_exec is not None and total_exec > 0:
                    active_times.append(float(total_exec))

            if not active_times:
                self.logger.warning(
                    "total_exec_time not found in ntff summary-json; "
                    "ActiveInferenceTime requires neuron-explorer with summary-json v2+"
                )
                return

            active_inference_time, outliers_count, qcd = self._iqr_filtered_average(active_times)
            num_samples = len(active_times)
        except Exception as e:
            self.logger.warning(f"Failed to parse ntff json for ActiveInferenceTime: {e}")
        finally:
            self.record_metric(MetricName.ACTIVE_INFERENCE_TIME, active_inference_time, "Seconds")
            if outliers_count is not None:
                self.record_metric(MetricName.ACTIVE_INFERENCE_TIME_OUTLIERS, float(outliers_count), "Count")
            if qcd is not None:
                self.record_metric(MetricName.ACTIVE_INFERENCE_TIME_QCD, qcd, "None")
            if num_samples is not None:
                self.record_metric(MetricName.ACTIVE_INFERENCE_TIME_SAMPLES, num_samples, "Count")

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
            avg_cycles = sum(filtered) / len(filtered)
        else:
            avg_cycles = sum(cycle_list) / len(cycle_list)

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
        self.duration = 0.0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        self.collector.record_timer(self.name, self.duration)
        return False  # Don't suppress exceptions


class NoopMetricsCollector(IMetricsCollector):
    """
    Metrics Collector that does nothing. Useful when metrics emissions needs to be disabled
    """

    test_name: str = ""

    @property
    @override
    def metrics_enabled(self) -> bool:
        return False

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
    def parse_artifacts(self, artifact_dir: str, inference_artifact_dir: str, target: str) -> None:
        """
        Parse test artifacts to extract metrics (compilation time, latency, MBU, etc.)
        """
        pass

    @override
    def set_test_name(self, test_name: str) -> None:
        self.test_name = test_name

    @override
    def set_kernel_params(self, params: dict[str, Any]) -> None:
        pass

    @override
    def get_kernel_params(self) -> dict[str, Any]:
        return {}

    @override
    def set_pytest_marks(self, marks: list[str]) -> None:
        pass

    @override
    def get_pytest_marks(self) -> list[str]:
        return []

    @override
    def set_output_dir(self, output_dir: str) -> None:
        pass

    @override
    def get_output_dir(self) -> str | None:
        return None


# Default no metrics collector
NOOP_METRICS_COLLECTOR = NoopMetricsCollector()
