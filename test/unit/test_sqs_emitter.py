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
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from ..utils.metrics_collector import MetricName, MetricsCollector
from ..utils.metrics_emitter import RUN_TYPE_USER, CoverageData, SessionContext
from ..utils.sqs_emitter import SQSEmitter

_SESSION = SessionContext(
    target="trn2",
    trace_mode="compile_and_infer",
    nki_compilation_mode="parser",
    sqs_queue_url="https://sqs.us-east-1.amazonaws.com/123456789/test-queue",
)


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
        collector.set_pytest_marks(["fast", "rmsnorm"])
        collector.record_metric(MetricName.COMPILATION_TIME, 3.74, "Seconds")
        collector.record_metric(MetricName.MBU_ESTIMATED_PERCENT, 85.5, "Percent")
        collector.record_metric(MetricName.INFERENCE_TIME, 1.2, "Milliseconds")
        return collector

    def test_emit_sends_test_result_message(self, mock_sqs, collector_with_metrics):
        """Verify emit() sends correctly formatted test_result message to SQS."""
        emitter = SQSEmitter(session=_SESSION)

        emitter.emit(collector_with_metrics)

        # Verify send_message was called
        mock_sqs.send_message.assert_called_once()
        call_args = mock_sqs.send_message.call_args

        # Verify queue URL
        assert call_args.kwargs["QueueUrl"] == _SESSION.sqs_queue_url

        # Parse and verify message body
        message = json.loads(call_args.kwargs["MessageBody"])
        print("\n=== test_result message ===")
        print(json.dumps(message, indent=2))

        assert message["type"] == "test_result"
        payload = message["payload"]

        # Session-level fields from emitter
        assert payload["Target"] == "trn2"
        assert payload["TraceMode"] == "compile_and_infer"
        assert payload["NkiCompilationMode"] == "parser"
        assert "RunType" not in payload
        assert "Username" not in payload
        assert "VersionSetEid" not in payload
        assert payload["IsRelease"] is False
        assert "Timestamp" in payload

        # Per-test fields from collector dimensions
        assert payload["TestName"] == "test_rmsnorm[batch=32-seq=1024]"
        assert payload["KernelName"] == "rmsnorm"
        assert payload["Status"] == "PASSED"
        assert payload["PytestMarks"] == ["fast", "rmsnorm"]

        # Verify params from collector.set_kernel_params()
        assert payload["Params"]["batch_size"] == 32
        assert payload["Params"]["seq_len"] == 1024

        # Verify metrics
        assert payload["Metrics"][MetricName.COMPILATION_TIME] == 3.74
        assert payload["Metrics"][MetricName.MBU_ESTIMATED_PERCENT] == 85.5
        assert payload["Metrics"][MetricName.INFERENCE_TIME] == 1.2

    def test_emit_run_complete_message(self, mock_sqs):
        """Verify emit_run_complete() sends correctly formatted message."""
        emitter = SQSEmitter(session=replace(_SESSION, kernel_name="attention_cte"))

        emitter.emit_run_complete(
            tests_passed=95,
            tests_total=100,
            run_duration_sec=1234.56,
        )

        mock_sqs.send_message.assert_called_once()
        call_args = mock_sqs.send_message.call_args

        message = json.loads(call_args.kwargs["MessageBody"])
        print("\n=== run_complete message ===")
        print(json.dumps(message, indent=2))

        assert message["type"] == "run_complete"
        payload = message["payload"]

        assert payload["KernelName"] == "attention_cte"
        assert payload["Target"] == "trn2"
        assert payload["TraceMode"] == "compile_and_infer"
        assert payload["NkiCompilationMode"] == "parser"
        assert "RunType" not in payload
        assert "Username" not in payload
        assert payload["IsRelease"] is False
        assert payload["TestsPassed"] == 95
        assert payload["TestsTotal"] == 100
        assert payload["RunDurationSec"] == pytest.approx(1234.56)
        assert "Timestamp" in payload

    def test_emit_with_custom_run_type_and_is_release(self, mock_sqs, collector_with_metrics):
        """Verify non-default run_type and is_release are included in payloads."""
        emitter = SQSEmitter(
            session=replace(_SESSION, run_type="model", is_release=True),
        )

        # Verify test_result payload
        emitter.emit(collector_with_metrics)
        call_args = mock_sqs.send_message.call_args
        message = json.loads(call_args.kwargs["MessageBody"])
        payload = message["payload"]

        assert payload["RunType"] == "model"
        assert payload["IsRelease"] is True

        # Verify run_complete payload
        mock_sqs.send_message.reset_mock()
        emitter.emit_run_complete(
            tests_passed=10,
            tests_total=10,
            run_duration_sec=42.0,
        )
        message = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])
        payload = message["payload"]

        assert payload["RunType"] == "model"
        assert payload["IsRelease"] is True

    def test_emit_with_user_run_type_and_username(self, mock_sqs, collector_with_metrics):
        """User runs include RunType=user and the developer's Username in both payloads."""
        emitter = SQSEmitter(
            session=replace(
                _SESSION,
                run_type=RUN_TYPE_USER,
                username="alice",
                kernel_name="rmsnorm",
            ),
        )

        emitter.emit(collector_with_metrics)
        test_result_payload = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])["payload"]
        assert test_result_payload["RunType"] == "user"
        assert test_result_payload["Username"] == "alice"
        assert "VersionSetEid" not in test_result_payload

        mock_sqs.send_message.reset_mock()
        emitter.emit_run_complete(tests_passed=3, tests_total=3, run_duration_sec=1.0)
        run_complete_payload = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])["payload"]
        assert run_complete_payload["RunType"] == "user"
        assert run_complete_payload["Username"] == "alice"

    def test_emit_with_version_set_eid(self, mock_sqs, collector_with_metrics):
        """Pipeline runs include VersionSetEid as a keyword-typed identifier."""
        emitter = SQSEmitter(
            session=replace(
                _SESSION,
                version_set_eid="6446448844",
                kernel_name="rmsnorm",
            ),
        )

        emitter.emit(collector_with_metrics)
        test_result_payload = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])["payload"]
        assert test_result_payload["VersionSetEid"] == "6446448844"

        mock_sqs.send_message.reset_mock()
        emitter.emit_run_complete(tests_passed=5, tests_total=5, run_duration_sec=1.0)
        run_complete_payload = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])["payload"]
        assert run_complete_payload["VersionSetEid"] == "6446448844"

    def test_emit_with_no_queue_url_does_nothing(self, mock_sqs):
        """Verify emit() does nothing when queue_url is None."""
        collector = MetricsCollector()
        emitter = SQSEmitter(session=replace(_SESSION, sqs_queue_url=None))

        # Should not raise
        emitter.emit(collector)
        emitter.emit_run_complete(10, 10, run_duration_sec=1.0)

    def test_get_metrics_enabled(self, mock_sqs):
        """Verify get_metrics_enabled() returns correct value."""
        emitter_enabled = SQSEmitter(session=_SESSION)
        assert emitter_enabled.get_metrics_enabled() is True

        emitter_disabled = SQSEmitter(session=replace(_SESSION, sqs_queue_url=None))
        assert emitter_disabled.get_metrics_enabled() is False

    def test_emit_handles_sqs_error_gracefully(self, mock_sqs, collector_with_metrics):
        """Verify emit() logs error but doesn't raise on SQS failure."""
        mock_sqs.send_message.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailable", "Message": "Service unavailable"}},
            "SendMessage",
        )

        emitter = SQSEmitter(session=_SESSION)

        # Should not raise
        emitter.emit(collector_with_metrics)

    def test_payload_structure_matches_opensearch_schema(self, mock_sqs, collector_with_metrics):
        """Verify payload structure matches expected OpenSearch document schema."""
        emitter = SQSEmitter(session=_SESSION)

        emitter.emit(collector_with_metrics)

        message = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])
        payload = message["payload"]

        # Required fields per OpenSearch schema (all PascalCase)
        required_fields = [
            "TestName",
            "KernelName",
            "Timestamp",
            "Status",
            "IsRelease",
            "Params",
            "Metrics",
            "Target",
            "TraceMode",
            "NkiCompilationMode",
            "PytestMarks",
        ]
        for field in required_fields:
            assert field in payload, f"Missing required field: {field}"

        # Verify types
        assert isinstance(payload["TestName"], str)
        assert isinstance(payload["KernelName"], str)
        assert isinstance(payload["Timestamp"], str)
        assert isinstance(payload["Status"], str)
        assert isinstance(payload["Params"], dict)
        assert isinstance(payload["Metrics"], dict)
        assert isinstance(payload["PytestMarks"], list)

    def test_emit_includes_empty_pytest_marks(self, mock_sqs, collector_with_metrics):
        collector_with_metrics.set_pytest_marks([])

        SQSEmitter(session=_SESSION).emit(collector_with_metrics)

        payload = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])["payload"]
        assert payload["PytestMarks"] == []

    def test_emit_skipped_test_via_collector(self, mock_sqs):
        """Verify skipped tests emit via emit(collector) with Status=skipped dimension."""
        emitter = SQSEmitter(session=replace(_SESSION, kernel_name="mlp_tkg"))

        collector = MetricsCollector()
        collector.set_namespace("NeuronCompiler")
        collector.set_test_name("test_optimal_256_1_128")
        collector.set_kernel_params({"batch": 256, "hidden": 128})
        collector.add_dimension({"Status": "skipped", "SkipReason": "MX quantization not supported"})

        emitter.emit(collector)

        mock_sqs.send_message.assert_called_once()
        message = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])

        assert message["type"] == "test_result"
        payload = message["payload"]
        assert payload["Status"] == "skipped"
        assert payload["SkipReason"] == "MX quantization not supported"
        assert payload["KernelName"] == "mlp_tkg"
        assert payload["Target"] == "trn2"
        assert payload["Params"] == {"batch": 256, "hidden": 128}

    def test_emit_run_complete_with_coverage_data(self, mock_sqs):
        """Verify coverage_data fields are merged into the run_complete SQS payload."""
        emitter = SQSEmitter(session=replace(_SESSION, kernel_name="attention_cte"))

        coverage_data = CoverageData(
            BranchRate=0.85,
            LineRate=0.92,
            CoveragePercent=89.12,
            BranchesCovered=170,
            BranchesValid=200,
        )

        emitter.emit_run_complete(
            tests_passed=80,
            tests_total=100,
            run_duration_sec=987.65,
            tests_skipped=15,
            tests_xfailed=5,
            coverage_data=coverage_data,
        )

        mock_sqs.send_message.assert_called_once()
        message = json.loads(mock_sqs.send_message.call_args.kwargs["MessageBody"])

        assert message["type"] == "run_complete"
        payload = message["payload"]

        # Core run counts
        assert payload["TestsPassed"] == 80
        assert payload["TestsTotal"] == 100
        assert payload["TestsSkipped"] == 15
        assert payload["TestsXfailed"] == 5
        assert payload["RunDurationSec"] == pytest.approx(987.65)

        # Coverage fields must be merged into the top-level payload
        assert payload["BranchRate"] == pytest.approx(0.85)
        assert payload["LineRate"] == pytest.approx(0.92)
        assert payload["BranchesCovered"] == 170
        assert payload["BranchesValid"] == 200
        assert payload["CoveragePercent"] == pytest.approx(89.12)

        # Session fields still present
        assert payload["KernelName"] == "attention_cte"
        assert payload["Target"] == "trn2"
        assert payload["TraceMode"] == "compile_and_infer"
        assert "Timestamp" in payload
