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
QoR (Quality of Results) data collector for NKL tests.

Collects metrics from test runs and aggregates them into a CSV file that persists
beyond test cleanup.
"""

import csv
import fcntl
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# QoR metrics to collect
QOR_METRICS = [
    "TpbSgCyclesSum",
    "MbuEstimatedPercent",
    "ProfilerMFU",
    "InferenceTime",
    "ActiveInferenceTime",
    "SeparatedComputeTime",
    "SeparatedMemoryTime",
]

# Dimension fields to include in CSV
DIMENSION_FIELDS = [
    "TestName",
]


def collect_qor_from_test_dir(test_dir: str, qor_output_dir: str, session_id: str | None = None) -> bool:
    """
    Collect QoR data from metrics JSON file (created by --metric-output) and append to CSV.

    Args:
        test_dir: Test output directory
        qor_output_dir: Directory for the QoR CSV file
        session_id: Session ID for filename (shared across parallel workers)

    Returns:
        True if data was collected, False otherwise
    """
    test_path = Path(test_dir)
    test_name = test_path.name

    metrics_dir = test_path / "metrics"
    if not metrics_dir.exists():
        return False

    for json_file in metrics_dir.glob("*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            record: dict[str, Any] = {"TestName": data.get("TestName", test_name)}
            for metric in QOR_METRICS:
                record[metric] = data.get(metric, -1.0)

            _append_qor_record(record, qor_output_dir, session_id)
            return True
        except Exception as e:
            logger.warning(f"Failed to parse metrics JSON {json_file}: {e}")

    return False


def _append_qor_record(record: dict[str, Any], qor_output_dir: str, session_id: str | None) -> None:
    """Append a single QoR record to the CSV file (thread-safe)."""
    output_dir = Path(qor_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if session_id:
        filename = f"qor_data_{session_id}.csv"
    else:
        filename = f"qor_data_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
    filepath = output_dir / filename
    fieldnames = DIMENSION_FIELDS + QOR_METRICS

    with open(filepath, "a", newline="") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            if f.tell() == 0:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            else:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(record)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
