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
"""Per-worker CSV accumulation, merge, and top-N summary reporting.

Used by monitor fixtures (duration, memory) that collect one row per test across
xdist workers and merge into a single sorted CSV at session end.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .common_dataclasses import PYTEST_XDIST_WORKER_ENV


def _parse_fields(row: str, num_fields: int) -> tuple[str, list[str]]:
    """Split a CSV row into (test_id, [field1, field2, ...]).

    Uses rsplit to handle test IDs that contain commas (e.g. parametrized names).
    """
    parts = row.rsplit(",", num_fields)
    return parts[0], parts[1:]


@dataclass(frozen=True)
class CSVMonitor:
    """Configuration for a per-worker CSV monitor.

    Attributes:
        prefix: filename prefix — per-worker files are ``{prefix}_{worker_id}.csv``,
            merged output is ``{prefix}.csv``.
        header: CSV header line (without trailing newline).
        sort_field: column index to sort by (0-indexed into the full header).
        sort_descending: whether to sort largest-first.
        format_row: function from ``(test_id, [field_values...])`` to a summary string.
        title_template: f-string-ready template with ``{top_n}`` and ``{total}`` placeholders.
    """

    prefix: str
    header: str
    sort_field: int
    sort_descending: bool
    format_row: Callable[[str, list[str]], str]
    title_template: str

    @property
    def num_fields(self) -> int:
        return len(self.header.split(",")) - 1

    def _sort_key(self, row: str) -> float:
        _, fields = _parse_fields(row, self.num_fields)
        value = float(fields[self.sort_field - 1])
        return -value if self.sort_descending else value

    def append(self, output_dir: Path, row: str) -> None:
        """Append a raw CSV row (no header, no newline) to the current worker's file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        worker_id = os.environ.get(PYTEST_XDIST_WORKER_ENV, "master")
        with open(output_dir / f"{self.prefix}_{worker_id}.csv", "a") as f:
            f.write(row + "\n")

    def cleanup_stale(self, output_dir: Path) -> None:
        """Remove per-worker CSVs left over from a previous run."""
        for csv_path in output_dir.glob(f"{self.prefix}_*.csv"):
            csv_path.unlink()

    def merge_and_report(self, output_dir: Path, top_n: int) -> None:
        """Merge per-worker CSVs, write sorted output, and print a top-N summary."""
        worker_csvs = sorted(output_dir.glob(f"{self.prefix}_*.csv"))
        lines: list[str] = []
        for csv_path in worker_csvs:
            lines.extend(csv_path.read_text().splitlines())
        if not lines:
            return

        rows = sorted(lines, key=self._sort_key)

        merged_path = output_dir / f"{self.prefix}.csv"
        with open(merged_path, "w") as f:
            f.write(self.header + "\n")
            f.writelines(row + "\n" for row in rows)
        for csv_path in worker_csvs:
            csv_path.unlink()

        print(f"\n=== {self.title_template.format(top_n=top_n, total=len(rows))} ===")
        for row in rows[:top_n]:
            test_id, fields = _parse_fields(row, self.num_fields)
            print(f"  {self.format_row(test_id, fields)}")
        print(f"  Full results: {merged_path}")


DURATION_MONITOR = CSVMonitor(
    prefix="duration_monitor",
    header="test_id,cpu_time_s,wall_time_s",
    sort_field=1,
    sort_descending=True,
    format_row=lambda test_id, fields: f"{float(fields[0]):8.1f}s cpu  {float(fields[1]):8.1f}s wall  {test_id}",
    title_template="Duration Monitor: Top {top_n} by CPU Time ({total} total)",
)

MEMORY_MONITOR = CSVMonitor(
    prefix="memory_monitor",
    header="test_id,peak_mb,delta_mb",
    sort_field=2,
    sort_descending=True,
    format_row=lambda test_id, fields: f"{float(fields[1]):8.1f} MB delta ({float(fields[0]):8.1f} MB peak)  {test_id}",
    title_template="Memory Monitor: Top {top_n} by Delta ({total} tests)",
)
