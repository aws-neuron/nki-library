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
Integration test for metrics collection system.

Tests that metrics are properly collected and emitted during test execution.
"""

import json
import tempfile
from pathlib import Path

from ..utils.metrics_collector import MetricName, MetricsCollector
from ..utils.metrics_emitter import MetricsEmitter, OutputMode


def test_metrics_collector_with_timers():
    """Test that MetricsCollector properly times operations and emits metrics."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = MetricsCollector()
        collector.set_namespace("NeuronCompiler")
        collector.set_pytest_marks(["fast", "mlp"])

        emitter = MetricsEmitter(
            output_mode=OutputMode.FILE,
        )
        collector.test_name = "test_example"
        collector.set_output_dir(tmpdir)

        # Start test
        collector.start_test()

        # Simulate compilation stage
        with collector.timer(MetricName.COMPILATION_TIME):
            import time

            time.sleep(0.01)  # Simulate work

        # Simulate execution stage
        with collector.timer("ExecutionTime"):
            time.sleep(0.01)

        # Record additional metrics
        collector.record_metric(MetricName.TPB_COUNT, 2.0, "Count")
        collector.record_metric(MetricName.MBU_ESTIMATED_PERCENT, 78.5, "Percent")

        # Build dimensions (simulating orchestrator behavior)
        collector.add_dimension(
            {
                "TestName": "test_example",
                "KernelName": "test_kernel",
                "Target": "trn2",
                "LNCCores": "2",
                "Status": "SUCCESS",
                "IsSuccessful": "true",
            }
        )

        # Emit all metrics
        emitter.emit(collector)

        # Verify metrics file was created
        files = list(Path(tmpdir).joinpath('metrics').glob("*.json"))
        assert len(files) == 1

        # Verify metrics content
        with open(files[0], "r") as f:
            data = json.load(f)

        # Check timers were recorded
        assert MetricName.COMPILATION_TIME in data
        assert "ExecutionTime" in data
        assert data[MetricName.COMPILATION_TIME] > 0.0
        assert data["ExecutionTime"] > 0.0

        # Check additional metrics
        assert data[MetricName.TPB_COUNT] == 2.0
        assert data[MetricName.MBU_ESTIMATED_PERCENT] == 78.5

        # Check dimensions
        assert data["TestName"] == "test_example"
        assert data["Target"] == "trn2"
        assert data["Status"] == "SUCCESS"
        assert data["IsSuccessful"] == "true"
        assert data["KernelName"] == "test_kernel"
        assert data["LNCCores"] == "2"
        assert data["PytestMarks"] == ["fast", "mlp"]

        # Check elapsed time was added
        assert MetricName.ELAPSED_ALL_SEC in data
        assert data[MetricName.ELAPSED_ALL_SEC] > 0.0


def test_metrics_collector_failure_status():
    """Test that failure status is properly tracked."""
    with tempfile.TemporaryDirectory() as tmpdir:
        collector = MetricsCollector()
        collector.set_namespace("NeuronCompiler")

        emitter = MetricsEmitter(
            output_mode=OutputMode.FILE,
        )
        collector.test_name = "test_failure"
        collector.set_output_dir(tmpdir)

        collector.start_test()

        # Only compilation timer (simulating execution failure)
        with collector.timer(MetricName.COMPILATION_TIME):
            pass

        # Build dimensions with failure status
        collector.add_dimension(
            {
                "TestName": "test_failure",
                "KernelName": "test_kernel",
                "Target": "trn2",
                "LNCCores": "2",
                "Status": "EXECUTION_FAILURE",
                "IsSuccessful": "false",
            }
        )

        # Emit with failure status
        emitter.emit(collector)

        files = list(Path(tmpdir).joinpath('metrics').glob("*.json"))
        assert len(files) == 1

        with open(files[0], "r") as f:
            data = json.load(f)

        assert data["Status"] == "EXECUTION_FAILURE"
        assert data["IsSuccessful"] == "false"
        assert MetricName.COMPILATION_TIME in data
        assert "ExecutionTime" not in data  # Didn't reach this stage
