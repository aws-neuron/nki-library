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
"""Unit tests for CSVMonitor."""

import pytest

from ..utils.common_dataclasses import PYTEST_XDIST_WORKER_ENV
from ..utils.csv_monitor import CSVMonitor


@pytest.fixture
def monitor():
    """A simple CSVMonitor for testing — sorts by first numeric field descending."""
    return CSVMonitor(
        prefix="test_monitor",
        header="name,value",
        sort_field=1,
        sort_descending=True,
        format_row=lambda test_id, fields: f"{fields[0]} :: {test_id}",
        title_template="Test: Top {top_n} ({total} total)",
    )


class TestAppend:
    def test_creates_output_dir(self, tmp_path, monitor, monkeypatch):
        monkeypatch.delenv(PYTEST_XDIST_WORKER_ENV, raising=False)
        output_dir = tmp_path / "nested" / "output"
        monitor.append(output_dir, "foo,1.0")
        assert output_dir.exists()

    def test_appends_rows_to_worker_file(self, tmp_path, monitor, monkeypatch):
        monkeypatch.setenv(PYTEST_XDIST_WORKER_ENV, "gw3")
        monitor.append(tmp_path, "a,1.0")
        monitor.append(tmp_path, "b,2.0")
        content = (tmp_path / "test_monitor_gw3.csv").read_text()
        assert content == "a,1.0\nb,2.0\n"


class TestCleanupStale:
    def test_removes_only_worker_csvs(self, tmp_path, monitor):
        (tmp_path / "test_monitor.csv").write_text("merged\n")
        (tmp_path / "test_monitor_gw0.csv").write_text("old data\n")
        (tmp_path / "test_monitor_gw1.csv").write_text("old data\n")
        monitor.cleanup_stale(tmp_path)
        assert list(tmp_path.glob("test_monitor_*.csv")) == []
        assert (tmp_path / "test_monitor.csv").read_text() == "merged\n"

    def test_no_error_when_nothing_to_clean(self, tmp_path, monitor):
        monitor.cleanup_stale(tmp_path)
        monitor.cleanup_stale(tmp_path / "nonexistent")


class TestMergeAndReport:
    def test_merges_sorts_cleans_and_prints_summary(self, tmp_path, monitor, capsys):
        (tmp_path / "test_monitor_gw0.csv").write_text("a,3.0\nc,1.0\n")
        (tmp_path / "test_monitor_gw1.csv").write_text("b,5.0\nd,2.0\n")

        monitor.merge_and_report(tmp_path, top_n=2)

        merged = (tmp_path / "test_monitor.csv").read_text()
        lines = merged.strip().split("\n")
        assert lines[0] == "name,value"
        assert lines[1] == "b,5.0"
        assert lines[2] == "a,3.0"
        assert lines[3] == "d,2.0"
        assert lines[4] == "c,1.0"

        assert list(tmp_path.glob("test_monitor_*.csv")) == []

        output = capsys.readouterr().out
        assert "Test: Top 2 (4 total)" in output
        assert "5.0 :: b" in output
        assert "3.0 :: a" in output
        assert "2.0 :: d" not in output
        assert f"Full results: {tmp_path / 'test_monitor.csv'}" in output

    def test_noop_when_no_data(self, tmp_path, monitor, capsys):
        monitor.merge_and_report(tmp_path, top_n=5)
        (tmp_path / "test_monitor_gw0.csv").write_text("")
        monitor.merge_and_report(tmp_path, top_n=5)

        assert not (tmp_path / "test_monitor.csv").exists()
        assert capsys.readouterr().out == ""
