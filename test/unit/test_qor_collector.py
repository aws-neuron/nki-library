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
"""Unit tests for QoR collector."""

import csv
import json
import tempfile
from pathlib import Path

from ..utils.qor_collector import collect_qor_from_test_dir


def test_qor_collector_from_metrics_json():
    """Test QoR collection from metrics JSON file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_output" / "test_example"
        metrics_dir = test_dir / "metrics"
        metrics_dir.mkdir(parents=True)

        # Create metrics JSON (as produced by --metric-output)
        metrics_data = {
            "TestName": "test_example",
            "TpbSgCyclesSum": 12345.0,
            "MbuEstimatedPercent": 75.5,
            "ProfilerMFU": 0.5,
            "InferenceTime": 0.001,
        }
        with open(metrics_dir / "test_example_01-01_00-00-00-UTC.json", "w") as f:
            json.dump(metrics_data, f)

        qor_dir = Path(tmpdir) / "qor_output"
        result = collect_qor_from_test_dir(str(test_dir), str(qor_dir))

        assert result is True

        csv_files = list(qor_dir.glob("*.csv"))
        assert len(csv_files) == 1

        with open(csv_files[0], "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["TestName"] == "test_example"
        assert float(rows[0]["TpbSgCyclesSum"]) == 12345.0
        assert float(rows[0]["MbuEstimatedPercent"]) == 75.5
        assert float(rows[0]["ProfilerMFU"]) == 0.5
        assert float(rows[0]["InferenceTime"]) == 0.001


def test_qor_collector_no_metrics_dir():
    """Test QoR collector with no metrics directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test_output" / "test_empty"
        test_dir.mkdir(parents=True)

        qor_dir = Path(tmpdir) / "qor"
        result = collect_qor_from_test_dir(str(test_dir), str(qor_dir))

        assert result is False
        assert not qor_dir.exists()
