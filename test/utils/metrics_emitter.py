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
- MetricsEmitter reads from MetricsCollector (doesn't own the metrics)
- MetricsCollector stores metrics in memory
- MetricsEmitter formats and writes to output destinations
"""

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, final

from aws_embedded_metrics.serializers.log_serializer import LogSerializer
from typing_extensions import override

from ..utils.exceptions import UnimplementedException
from ..utils.feature_flag_helper import truncate_name_for_filesystem
from ..utils.metrics_collector import IMetricsCollector, NoopMetricsCollector


class OutputMode(str, Enum):
    """Output mode for metrics emission."""

    FILE = "file"
    STDERR = "stderr"
    STDOUT = "stdout"


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
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set the output directory for file mode.
        """
        raise UnimplementedException()

    @abstractmethod
    def emit(self) -> None:
        """
        Emit all metrics collected by the collector.
        """
        raise UnimplementedException()

    @abstractmethod
    def get_collector(self) -> IMetricsCollector:
        """
        Emit all metrics collected by the collector.
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
        collector: IMetricsCollector,
        output_mode: OutputMode | None = None,
        output_dir: str | None = None,
    ):
        """
        Initialize metrics emitter.

        Args:
            collector: MetricsCollector instance to read metrics from
            output_mode: Where to write metrics ("file", "stdout", or None to disable)
            output_dir: Directory for file mode (can be set later via set_output_dir)
        """
        self.collector = collector
        self._output_mode: OutputMode | None = output_mode
        self._output_dir = output_dir

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
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set the output directory for file mode.
        """
        self._output_dir = output_dir

    @override
    def emit(self) -> None:
        """
        Emit all metrics collected by the collector.

        This method:
        1. Gets finalized metrics from collector
        2. Writes to configured output destination
        """
        metrics_context = self.collector.get_finalized_metrics_context()

        # Emit the metrics context
        self._emit_metrics_context(metrics_context)

    @override
    def get_collector(self) -> IMetricsCollector:
        return self.collector

    def _emit_metrics_context(self, metrics_context) -> None:
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
                self._write_to_file(emf_data)

    def _write_to_file(self, emf_data: dict[str, Any]) -> None:
        """
        Write EMF JSON to local file in test artifact directory.
        Filename format: <test_name>_<timestamp>.json
        """
        assert self._output_dir

        # Generate filename with test name and readable timestamp
        test_name = emf_data.get("TestName", "unknown")
        readable_time = datetime.now(timezone.utc).strftime("%m-%d_%H-%M-%S-UTC")
        test_name = truncate_name_for_filesystem(test_name)
        filename = f"{test_name}_{readable_time}.json"
        filepath = os.path.join(self._output_dir, filename)

        with open(filepath, "w") as f:
            json.dump(emf_data, f, indent=2)


class NoopMetricsEmitter(IMetricsEmitter):
    """
    Metrics emitter that does not do anything. Useful when metrics need to be disabled for whatever
    reason.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize metrics emitter.

        Args:
            collector: MetricsCollector instance to read metrics from
            output_mode: Where to write metrics ("file", "stdout", or None to disable)
            output_dir: Directory for file mode (can be set later via set_output_dir)
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
    def set_output_dir(self, output_dir: str) -> None:
        """
        Set the output directory for file mode.
        """
        pass

    @override
    def emit(self) -> None:
        """
        Emit all metrics collected by the collector.

        This method:
        1. Gets finalized metrics from collector
        2. Writes to configured output destination
        """
        if self.verbose:
            self.logger.warning("Skipping metric emission as it's disabled")

    @override
    def get_collector(self) -> IMetricsCollector:
        return NoopMetricsCollector()
