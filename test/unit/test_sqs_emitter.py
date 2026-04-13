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
"""Unit tests for SQS emitter module."""

import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from ..utils.metrics_collector import MetricName, MetricsCollector
from ..utils.sqs_emitter import SQSEmitter


class TestSQSEmitter:
    """Test SQSEmitter functionality."""

    @pytest.fixture
    def mock_sqs(self):
        """Mock boto3 SQS client."""
        with patch("boto3.client") as mock_client:
            mock_sqs = MagicMock()
            mock_client.return_value = mock_sqs
            yield mock_sqs

    @pytest.fixture
    def collector_with_metrics(self):
        """Create a collector with sample metrics."""
        collector = MetricsCollector()
        collector.set_namespace("NeuronCompiler")
        collector.add_dimension(
            {
                "TestName": "test_rmsnorm[batch=32-seq=1024]",
                "KernelName": "rmsnorm",
                "Target": "trn2",
                "LNCCores": "2",
                "Status": "PASSED",
            }
        )
        collector.set_kernel_params(
            {
                "batch_size": 32,
                "seq_len": 1024,
            }
        )
        collector.record_metric(MetricName.COMPILATION_TIME, 3.74, "Seconds")
        collector.record_metric(MetricName.MBU_ESTIMATED_PERCENT, 85.5, "Percent")
        collector.record_metric(MetricName.INFERENCE_TIME, 1.2, "Milliseconds")
        return collector

    def test_emit_sends_test_result_message(self, mock_sqs, collector_with_metrics, monkeypatch):
        """Verify emit() sends correctly formatted test_result message to SQS."""
        monkeypatch.setenv("EMIT_TEST_RESULTS", "true")
        emitter = SQSEmitter(
            collector=collector_with_metrics,
            queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue",
            run_id="test-run-123",
        )

        emitter.emit()

        # Verify send_message was called
        mock_sqs.send_message.assert_called_once()
        call_args = mock_sqs.send_message.call_args

        # Verify queue URL
        assert call_args.kwargs["QueueUrl"] == "https://sqs.us-east-1.amazonaws.com/123456789/test-queue"

        # Parse and verify message body
        message = json.loads(call_args.kwargs["MessageBody"])
        print("\n=== test_result message ===")
        print(json.dumps(message, indent=2))

        assert message["type"] == "test_result"
        payload = message["payload"]

        # All fields use PascalCase
        assert payload["RunId"] == "test-run-123"
        assert payload["TestName"] == "test_rmsnorm[batch=32-seq=1024]"
        assert payload["KernelName"] == "rmsnorm"
        assert payload["Status"] == "PASSED"
        assert payload["Target"] == "trn2"
        assert "Timestamp" in payload

        # Verify params from collector.set_kernel_params()
        assert payload["Params"]["batch_size"] == 32
        assert payload["Params"]["seq_len"] == 1024

        # Verify metrics
        assert payload["Metrics"][MetricName.COMPILATION_TIME] == 3.74
        assert payload["Metrics"][MetricName.MBU_ESTIMATED_PERCENT] == 85.5
        assert payload["Metrics"][MetricName.INFERENCE_TIME] == 1.2

    def test_emit_run_complete_message(self, mock_sqs):
        """Verify emit_run_complete() sends correctly formatted message."""
        collector = MetricsCollector()
        emitter = SQSEmitter(
            collector=collector,
            queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue",
            run_id="test-run-456",
        )

        emitter.emit_run_complete(kernel_name="attention_cte", tests_passed=95, tests_total=100)

        mock_sqs.send_message.assert_called_once()
        call_args = mock_sqs.send_message.call_args

        message = json.loads(call_args.kwargs["MessageBody"])
        print("\n=== run_complete message ===")
        print(json.dumps(message, indent=2))

        assert message["type"] == "run_complete"
        payload = message["payload"]

        assert payload["RunId"] == "test-run-456"
        assert payload["KernelName"] == "attention_cte"
        assert payload["TestsPassed"] == 95
        assert payload["TestsTotal"] == 100
        assert "Timestamp" in payload

    def test_emit_with_no_queue_url_does_nothing(self, mock_sqs):
        """Verify emit() does nothing when queue_url is None."""
        collector = MetricsCollector()
        emitter = SQSEmitter(collector=collector, queue_url=None, run_id="test-run")

        # Should not raise
        emitter.emit()
        emitter.emit_run_complete("kernel", 10, 10)

    def test_get_metrics_enabled(self, mock_sqs):
        """Verify get_metrics_enabled() returns correct value."""
        collector = MetricsCollector()

        emitter_enabled = SQSEmitter(collector=collector, queue_url="https://sqs.example.com/queue", run_id="run-1")
        assert emitter_enabled.get_metrics_enabled() is True

        emitter_disabled = SQSEmitter(collector=collector, queue_url=None, run_id="run-2")
        assert emitter_disabled.get_metrics_enabled() is False

    def test_emit_handles_sqs_error_gracefully(self, mock_sqs, collector_with_metrics):
        """Verify emit() logs error but doesn't raise on SQS failure."""
        mock_sqs.send_message.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailable", "Message": "Service unavailable"}},
            "SendMessage",
        )

        emitter = SQSEmitter(
            collector=collector_with_metrics,
            queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue",
            run_id="test-run-123",
        )

        # Should not raise
        emitter.emit()

    def test_payload_structure_matches_opensearch_schema(self, mock_sqs, collector_with_metrics, monkeypatch):
        """Verify payload structure matches expected OpenSearch document schema."""
        monkeypatch.setenv("EMIT_TEST_RESULTS", "true")
        emitter = SQSEmitter(
            collector=collector_with_metrics,
            queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue",
            run_id="pipeline-event-6414929928",
        )

        emitter.emit()

        message = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])
        payload = message["payload"]

        # Required fields per OpenSearch schema (all PascalCase)
        required_fields = [
            "RunId",
            "TestName",
            "KernelName",
            "Timestamp",
            "Status",
            "Params",
            "Metrics",
        ]
        for field in required_fields:
            assert field in payload, f"Missing required field: {field}"

        # Verify types
        assert isinstance(payload["RunId"], str)
        assert isinstance(payload["TestName"], str)
        assert isinstance(payload["KernelName"], str)
        assert isinstance(payload["Timestamp"], str)
        assert isinstance(payload["Status"], str)
        assert isinstance(payload["Params"], dict)
        assert isinstance(payload["Metrics"], dict)
