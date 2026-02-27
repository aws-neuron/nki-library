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

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import final

import boto3
from botocore.exceptions import ClientError
from typing_extensions import override

from .metrics_collector import IMetricsCollector
from .metrics_emitter import IMetricsEmitter, OutputMode

# SQS batch send configuration
SQS_BATCH_SIZE = 10  # SQS maximum messages per batch
SQS_MAX_RETRIES = 3


@final
class SQSEmitter(IMetricsEmitter):
    """Emits test metrics to SQS for OpenSearch ingestion."""

    def __init__(self, collector: IMetricsCollector, queue_url: str | None, run_id: str):
        """
        Initialize SQS emitter.

        Args:
            collector: MetricsCollector instance to read metrics from
            queue_url: SQS Standard queue URL
            run_id: Pipeline run identifier
        """
        self._collector = collector
        self._queue_url = queue_url
        self._run_id = run_id
        self._sqs = None  # Lazy init to avoid overhead when disabled
        self._logger = logging.getLogger(__name__)

    def _get_sqs_client(self):
        """Lazy-initialize SQS client on first use."""
        if self._sqs is None:
            self._sqs = boto3.client("sqs")
        return self._sqs

    @override
    def get_metrics_enabled(self) -> bool:
        return self._queue_url is not None

    @override
    def get_output_mode(self) -> OutputMode | None:
        return None

    @override
    def set_output_dir(self, output_dir: str) -> None:
        pass

    @override
    def emit(self) -> None:
        """Send test_result message to SQS."""
        if not self._queue_url:
            return

        # Skip emission if no kernel params - test is likely not parametrized and not useful for dashboard
        if not self._collector.get_kernel_params():
            self._logger.warning("Skipping SQS emission: no kernel params found")
            return

        try:
            metrics_context = self._collector.get_finalized_metrics_context()
            payload = self._build_payload(metrics_context)
            message = {"type": "test_result", "payload": payload}

            self._get_sqs_client().send_message(QueueUrl=self._queue_url, MessageBody=json.dumps(message))
        except ClientError as e:
            self._logger.error(f"Failed to send message to SQS: {e}")
        except Exception as e:
            self._logger.error(f"SQSEmitter.emit failed: {e}")

    @override
    def get_collector(self) -> IMetricsCollector:
        return self._collector

    def emit_run_complete(self, kernel_name: str, tests_passed: int, tests_total: int) -> None:
        """Send run_complete message at session end."""
        if not self._queue_url:
            return

        # Only emit targeted model configs if flag is set (for model test approval steps)
        if os.environ.get("EMIT_TARGETED_MODEL", "").lower() == "true":
            self._emit_targeted_model_configs()

        try:
            message = {
                "type": "run_complete",
                "payload": {
                    "run_id": self._run_id,
                    "kernel_name": kernel_name,
                    "tests_passed": tests_passed,
                    "tests_total": tests_total,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            self._get_sqs_client().send_message(QueueUrl=self._queue_url, MessageBody=json.dumps(message))
        except ClientError as e:
            self._logger.error(f"Failed to send run_complete to SQS: {e}")
        except Exception as e:
            self._logger.error(f"SQSEmitter.emit_run_complete failed: {e}")

    def _emit_targeted_model_configs(self) -> int:
        """
        Emit all targeted model configs from configuration/test_vector_metadata.

        Uses SQS batch send (up to 10 messages per batch) for efficiency.
        Each config is emitted with a content-based version_id for deduplication.
        Returns the number of configs emitted.
        """
        if not self._queue_url:
            return 0

        metadata_dir = Path(__file__).parent.parent.parent / "configuration" / "test_vector_metadata"
        if not metadata_dir.exists():
            self._logger.warning(f"Metadata directory not found: {metadata_dir}")
            return 0

        # Collect all messages first
        messages = []
        for config_file in metadata_dir.glob("*.json"):
            try:
                with open(config_file, "r") as f:
                    configs = json.load(f)

                # Handle both list and single object
                if not isinstance(configs, list):
                    configs = [configs]

                for config in configs:
                    version_id = self._compute_config_version_id(config)
                    message = {
                        "type": "targeted_model",
                        "payload": {
                            "version_id": version_id,
                            "source_file": config_file.name,
                            "indexed_at": datetime.now(timezone.utc).isoformat(),
                            **config,
                        },
                    }
                    messages.append((f"{config_file.name}:{version_id[:8]}", message))
            except Exception as e:
                self._logger.error(f"Failed to parse config file {config_file.name}: {e}")

        # Send in batches of 10 (SQS limit) with retry
        emitted_count = 0
        failed_configs = []

        for i in range(0, len(messages), SQS_BATCH_SIZE):
            batch = messages[i : i + SQS_BATCH_SIZE]
            entries = [{"Id": str(idx), "MessageBody": json.dumps(msg)} for idx, (_, msg) in enumerate(batch)]

            for attempt in range(SQS_MAX_RETRIES):
                try:
                    response = self._get_sqs_client().send_message_batch(QueueUrl=self._queue_url, Entries=entries)
                    emitted_count += len(response.get("Successful", []))
                    for failure in response.get("Failed", []):
                        failed_configs.append(batch[int(failure["Id"])][0])
                    break
                except Exception as e:
                    if attempt < SQS_MAX_RETRIES - 1:
                        self._logger.warning(f"Batch send attempt {attempt + 1}/{SQS_MAX_RETRIES} failed: {e}")
                    else:
                        self._logger.error(f"Batch send failed after {SQS_MAX_RETRIES} retries: {e}")
                        failed_configs.extend([key for key, _ in batch])

        self._logger.info(f"Emitted {emitted_count} targeted model configs to SQS")
        if failed_configs:
            self._logger.error(f"Failed to emit {len(failed_configs)} configs: {failed_configs[:10]}...")
        return emitted_count

    def _compute_config_version_id(self, config: dict) -> str:
        """Compute stable version ID from config content."""
        content = json.dumps(config, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()  # Full 64-char SHA256 for collision resistance

    def _build_payload(self, metrics_context) -> dict:
        """Build test_result payload from metrics context."""
        dimensions = {}
        metrics = {}

        for dim in metrics_context.dimensions:
            dimensions.update(dim)

        for metric_name, metric in metrics_context.metrics.items():
            metrics[metric_name] = metric.values[0] if metric.values else None

        # Get kernel params from collector (extracted scalar params from kernel_input)
        params = self._collector.get_kernel_params()

        test_name = dimensions.get("TestName", "unknown")
        return {
            "run_id": self._run_id,
            "test_name": test_name,
            "kernel_name": dimensions.get("KernelName", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform_target": dimensions.get("Target", "unknown"),
            "status": dimensions.get("Status", "UNKNOWN"),
            "params": params,
            "metrics": metrics,
        }
