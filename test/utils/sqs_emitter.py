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
"""SQS emitter for sending test metrics to OpenSearch via SQS queue."""

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from typing import final

import boto3
from botocore.exceptions import ClientError
from typing_extensions import override

from .metrics_collector import IMetricsCollector
from .metrics_emitter import CoverageData, IMetricsEmitter, OutputMode, SessionContext


@final
class SQSEmitter(IMetricsEmitter):
    """Emits test metrics to SQS for OpenSearch ingestion.

    Session-scoped: created once per pytest session and reused for all messages.
    Owns all session-level fields (RunId, RunType, Target, TraceMode, etc.)
    so they are set in one place rather than split across orchestrator and conftest.
    """

    def __init__(
        self,
        session: SessionContext,
    ):
        """
        Initialize SQS emitter.

        Args:
            session: Session-level context (RunId, Target, queue URL, etc.)
        """
        self._session = session
        self._sqs = None
        self._logger = logging.getLogger(__name__)

    def _get_sqs_client(self):
        """Lazy-initialize SQS client on first use."""
        if self._sqs is None:
            self._sqs = boto3.client("sqs")
        return self._sqs

    @override
    def get_metrics_enabled(self) -> bool:
        return self._session.sqs_queue_url is not None

    @override
    def get_output_mode(self) -> OutputMode | None:
        return None

    @override
    def emit(self, collector: IMetricsCollector) -> None:
        """Send test_result message to SQS."""
        if not self._session.sqs_queue_url:
            return

        # Skip emission if no kernel params - test is likely not parametrized and not useful for dashboard
        if not collector.get_kernel_params():
            self._logger.warning("Skipping SQS emission: no kernel params found")
            return

        try:
            metrics_context = collector.get_finalized_metrics_context()
            payload = self._build_payload(metrics_context, collector)
            message = {"type": "test_result", "payload": payload}

            # Log the SQS message being sent for debugging
            self._logger.info(f"SQS Message: {json.dumps(message)}")
            self._get_sqs_client().send_message(QueueUrl=self._session.sqs_queue_url, MessageBody=json.dumps(message))
        except ClientError as e:
            self._logger.error(f"Failed to send message to SQS: {e}")
        except Exception as e:
            self._logger.error(f"SQSEmitter.emit failed: {e}")

    def emit_run_complete(
        self,
        tests_passed: int,
        tests_total: int,
        run_duration_sec: float,
        tests_skipped: int = 0,
        tests_xfailed: int = 0,
        coverage_data: CoverageData | None = None,
    ) -> None:
        """Send run_complete message at session end."""
        if not self._session.sqs_queue_url:
            return

        try:
            payload = {
                **self._session.to_dimensions(),
                "TestsPassed": tests_passed,
                "TestsTotal": tests_total,
                "TestsSkipped": tests_skipped,
                "TestsXfailed": tests_xfailed,
                "RunDurationSec": run_duration_sec,
                "Timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if coverage_data:
                payload.update(asdict(coverage_data))

            message = {
                "type": "run_complete",
                "payload": payload,
            }

            # Log the SQS message being sent for debugging
            self._logger.info(f"SQS Message: {json.dumps(message)}")
            self._get_sqs_client().send_message(QueueUrl=self._session.sqs_queue_url, MessageBody=json.dumps(message))
        except ClientError as e:
            self._logger.error(f"Failed to send run_complete to SQS: {e}")
        except Exception as e:
            self._logger.error(f"SQSEmitter.emit_run_complete failed: {e}")

    def _build_payload(self, metrics_context, collector: IMetricsCollector) -> dict:
        """Build test_result payload from metrics context."""
        dimensions = {}
        metrics = {}

        for dim in metrics_context.dimensions:
            dimensions.update(dim)

        for metric_name, metric in metrics_context.metrics.items():
            metrics[metric_name] = metric.values[0] if metric.values else None

        params = collector.get_kernel_params()

        return {
            **self._session.to_dimensions(),
            **dimensions,
            "PytestMarks": collector.get_pytest_marks(),
            "Params": params,
            "Metrics": metrics,
            "Timestamp": datetime.now(timezone.utc).isoformat(),
        }
