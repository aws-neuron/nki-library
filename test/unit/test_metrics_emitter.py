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
Unit tests for metrics_emitter module.
"""

import json
import tempfile
from pathlib import Path

from aws_embedded_metrics.logger.metrics_context import MetricsContext

from ..utils.metrics_collector import MetricName, MetricsCollector
from ..utils.metrics_emitter import MetricsEmitter, OutputMode


class TestMetricsEmitter:
    """Test MetricsEmitter functionality."""

    def test_file_mode_creates_metrics_files(self):
        """File mode: Metrics files created in local directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector()
            collector.set_namespace("NeuronCompiler")

            emitter = MetricsEmitter(
                collector=collector,
                output_mode=OutputMode.FILE,
                output_dir=tmpdir,
            )

            # Create MetricsContext and add metric
            context = MetricsContext.empty()
            context.namespace = "NeuronCompiler"
            context.set_dimensions([{"TestName": "test_foo", "Target": "trn2"}])
            context.put_metric(MetricName.COMPILATION_TIME, 5.234, "Seconds")

            # Emit directly
            emitter._emit_metrics_context(context)

            # Verify file created
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1

            # Verify file content
            with open(files[0], "r") as f:
                data = json.load(f)

            assert "_aws" in data
            assert data["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "NeuronCompiler"
            assert data[MetricName.COMPILATION_TIME] == 5.234
            assert data["TestName"] == "test_foo"
            assert data["Target"] == "trn2"

    def test_batch_metrics(self):
        """Test recording multiple metrics in a batch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector()
            collector.set_namespace("NeuronCompiler")

            emitter = MetricsEmitter(
                collector=collector,
                output_mode=OutputMode.FILE,
                output_dir=tmpdir,
            )

            # Create MetricsContext with multiple metrics
            context = MetricsContext.empty()
            context.namespace = "NeuronCompiler"
            context.set_dimensions([{"TestName": "test_foo"}])
            context.put_metric(MetricName.COMPILATION_TIME, 5.234, "Seconds")
            context.put_metric(MetricName.MBU_ESTIMATED_PERCENT, 78.5, "Percent")

            # Emit
            emitter._emit_metrics_context(context)

            # Verify single file created with both metrics
            files = list(Path(tmpdir).glob("*.json"))
            assert len(files) == 1

            with open(files[0], "r") as f:
                data = json.load(f)

            # Both metrics should be present
            assert data[MetricName.COMPILATION_TIME] == 5.234
            assert data[MetricName.MBU_ESTIMATED_PERCENT] == 78.5

            # Check EMF structure
            metrics = data["_aws"]["CloudWatchMetrics"][0]["Metrics"]
            metric_names = [m["Name"] for m in metrics]
            assert MetricName.COMPILATION_TIME in metric_names
            assert MetricName.MBU_ESTIMATED_PERCENT in metric_names

    def test_emf_format_structure(self):
        """Verify EMF (Embedded Metric Format) structure is correct."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector()
            collector.set_namespace("NeuronCompiler")

            emitter = MetricsEmitter(
                collector=collector,
                output_mode=OutputMode.FILE,
                output_dir=tmpdir,
            )

            # Create MetricsContext
            context = MetricsContext.empty()
            context.namespace = "NeuronCompiler"
            context.set_dimensions([{"KernelName": "rmsnorm", "Platform": "trn2"}])
            context.put_metric("Latency", 123.45, "Milliseconds")

            # Emit
            emitter._emit_metrics_context(context)

            files = list(Path(tmpdir).glob("*.json"))
            with open(files[0], "r") as f:
                data = json.load(f)

            # Verify EMF structure
            assert "_aws" in data
            assert "Timestamp" in data["_aws"]
            assert "CloudWatchMetrics" in data["_aws"]

            cw_metrics = data["_aws"]["CloudWatchMetrics"][0]
            assert cw_metrics["Namespace"] == "NeuronCompiler"
            assert len(cw_metrics["Dimensions"]) == 1
            assert set(cw_metrics["Dimensions"][0]) == {"KernelName", "Platform"}

            # Verify metric definition
            assert len(cw_metrics["Metrics"]) == 1
            assert cw_metrics["Metrics"][0]["Name"] == "Latency"
            assert cw_metrics["Metrics"][0]["Unit"] == "Milliseconds"

            # Verify dimension values
            assert data["KernelName"] == "rmsnorm"
            assert data["Platform"] == "trn2"

            # Verify metric value
            assert data["Latency"] == 123.45

    def test_custom_namespace(self):
        """Test custom CloudWatch namespace."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector()
            collector.set_namespace("CustomNamespace")

            emitter = MetricsEmitter(
                collector=collector,
                output_mode=OutputMode.FILE,
                output_dir=tmpdir,
            )

            # Create MetricsContext
            context = MetricsContext.empty()
            context.namespace = "CustomNamespace"
            context.put_metric("TestMetric", 42.0, "None")

            # Emit
            emitter._emit_metrics_context(context)

            files = list(Path(tmpdir).glob("*.json"))
            with open(files[0], "r") as f:
                data = json.load(f)

            assert data["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "CustomNamespace"
