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
Metrics emission library for NKL tests.

Emits metrics in AWS EMF (Embedded Metric Format) to either:
1. Local JSON files in test artifact directory (file mode - default)
2. stdout for log ingestion (stdout mode)

Architecture:
- Session-level data (Target, etc.) is baked into emitters at session start.
- Per-test data (TestName, Metrics, Params) lives in a MetricsCollector created per test.
- Emitters receive the collector at emit time: emit(collector).
"""

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, final

from aws_embedded_metrics.serializers.log_serializer import LogSerializer
from typing_extensions import override

from .exceptions import UnimplementedException
from .feature_flag_helper import truncate_name_for_filesystem
from .metrics_collector import IMetricsCollector


@dataclass(frozen=True)
class CoverageData:
    """Coverage metrics extracted from pytest-cov's in-memory data."""

    BranchRate: float
    LineRate: float
    CoveragePercent: float  # Combined line+branch % (same as pytest-cov terminal report)
    BranchesCovered: int
    BranchesValid: int


@dataclass(frozen=True)
class SessionContext:
    """Immutable session-level data shared across all tests in a pytest session."""

    target: str
    trace_mode: str
    nki_compilation_mode: str
    kernel_name: str | None = None
    run_type: str | None = None
    is_release: bool = False
    sqs_queue_url: str | None = None
    username: str | None = None
    version_set_eid: str | None = None

    def to_dimensions(self) -> dict:
        """Convert to CloudWatch-style dimension dict for payloads."""
        fields = {
            "Target": self.target,
            "TraceMode": self.trace_mode,
            "NkiCompilationMode": self.nki_compilation_mode,
            "IsRelease": self.is_release,
        }
        if self.kernel_name is not None:
            fields["KernelName"] = self.kernel_name
        if self.run_type is not None:
            fields["RunType"] = self.run_type
        if self.username is not None:
            fields["Username"] = self.username
        if self.version_set_eid is not None:
            fields["VersionSetEid"] = self.version_set_eid
        return fields


class OutputMode(str, Enum):
    """Output mode for metrics emission."""

    FILE = "file"
    STDERR = "stderr"
    STDOUT = "stdout"


RUN_TYPE_USER = "user"


class IMetricsEmitter(ABC):
    @abstractmethod
    def get_metrics_enabled(self) -> bool:
        """
        Check if metrics emission is enabled.
        Returns True if output_mode is not None.
        """
        raise UnimplementedException()

    @abstractmethod
    def get_output_mode(self) -> OutputMode | None:
        """
        Get the output mode.

        Returns: OutputMode.FILE, OutputMode.STDOUT, or None
        """
        raise UnimplementedException()

    @abstractmethod
    def emit(self, collector: IMetricsCollector) -> None:
        """
        Emit all metrics from the given collector.
        """
        raise UnimplementedException()


@final
class MetricsEmitter(IMetricsEmitter):
    """
    Emits metrics in EMF format by reading from MetricsCollector.

    Two output modes:
    1. File mode (default): Writes to JSON files in test artifact directory
    2. Stdout mode: Writes to stdout for logs to capture

    The emitter reads from the collector and handles all I/O operations.
    """

    def __init__(
        self,
        output_mode: OutputMode | None = None,
    ):
        """
        Initialize metrics emitter.

        Args:
            output_mode: Where to write metrics ("file", "stdout", or None to disable)
        """
        self._output_mode: OutputMode | None = output_mode

        self.logger = logging.getLogger(__name__)
        self.serializer = LogSerializer()

    @override
    def get_metrics_enabled(self) -> bool:
        """
        Check if metrics emission is enabled.
        Returns True if output_mode is not None.
        """
        return self._output_mode is not None

    @override
    def get_output_mode(self) -> OutputMode | None:
        """
        Get the output mode.

        Returns: OutputMode.FILE, OutputMode.STDOUT, or None
        """
        return self._output_mode

    @override
    def emit(self, collector: IMetricsCollector) -> None:
        """
        Emit all metrics from the given collector.

        This method:
        1. Gets finalized metrics from collector
        2. Writes to configured output destination
        """
        metrics_context = collector.get_finalized_metrics_context()
        self._emit_metrics_context(metrics_context, collector.get_output_dir())

    def _emit_metrics_context(self, metrics_context, output_dir: str | None) -> None:
        """
        Emit metrics using AWS MetricsContext.
        """
        # Serialize to EMF JSON using AWS library (returns List[str] for batching)
        emf_json_list = self.serializer.serialize(metrics_context)

        for emf_json_str in emf_json_list:
            if not emf_json_str:
                continue

            emf_data = json.loads(emf_json_str)
            # writing to stdout or stderr
            if self._output_mode == OutputMode.STDERR:
                json.dump(emf_data, sys.stderr)
                _ = sys.stderr.write("\n")
                _ = sys.stderr.flush()
            elif self._output_mode == OutputMode.STDOUT:
                json.dump(emf_data, sys.stdout)
                _ = sys.stdout.write("\n")
                _ = sys.stdout.flush()
            # writing to local file
            else:
                self._write_to_file(emf_data, output_dir)

    def _write_to_file(self, emf_data: dict[str, Any], output_dir: str | None) -> None:
        """
        Write EMF JSON to metrics subdirectory under the test artifact directory.
        Filename format: <test_name>_<timestamp>.json
        """
        assert output_dir, "output_dir must be set on collector before emitting in file mode"

        metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)

        # Generate filename with test name and readable timestamp
        test_name = emf_data.get("TestName", "unknown")
        readable_time = datetime.now(timezone.utc).strftime("%m-%d_%H-%M-%S-UTC")
        test_name = truncate_name_for_filesystem(test_name)
        filename = f"{test_name}_{readable_time}.json"
        filepath = os.path.join(metrics_dir, filename)

        with open(filepath, "w") as f:
            json.dump(emf_data, f, indent=2)


class NoopMetricsEmitter(IMetricsEmitter):
    """
    Metrics emitter that does not do anything. Useful when metrics need to be disabled for whatever
    reason.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize noop metrics emitter.

        Args:
            verbose: If True, log a warning when emit is called
        """
        super().__init__()

        self.verbose: bool = verbose
        self.logger = logging.getLogger(__name__)

    @override
    def get_metrics_enabled(self) -> bool:
        """
        Check if metrics emission is enabled.
        Returns True if output_mode is not None.
        """
        return False

    @override
    def get_output_mode(self) -> OutputMode | None:
        """
        Get the output mode.

        Returns: OutputMode.FILE, OutputMode.STDOUT, or None
        """
        return None

    @override
    def emit(self, collector: IMetricsCollector) -> None:
        """
        Emit all metrics collected by the collector.

        This method:
        1. Gets finalized metrics from collector
        2. Writes to configured output destination
        """
        if self.verbose:
            self.logger.warning("Skipping metric emission as it's disabled")
