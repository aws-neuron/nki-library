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
"""Unit tests for test/utils/test_set_minimizer.py — cost-weighted
greedy set-cover that minimizes a test set while preserving branch
coverage, using peak memory only as a tiebreaker."""

import textwrap
from pathlib import Path

import pytest

from ..utils.test_set_minimizer import (
    MinimizationResult,
    _lookup,
    load_csv_metric,
    load_per_test_arcs,
    minimize_test_set,
)

# ---------------------------------------------------------------------------
# minimize_test_set: algorithm core
# ---------------------------------------------------------------------------


class TestGreedyAlgorithm:
    """Compile-time-weighted greedy picks the highest arcs/sec each iteration."""

    def test_picks_higher_arcs_per_second(self):
        # Both cover the same number of arcs but A is 2x faster.
        test_arcs = {"A": {(1, 1, 2), (1, 2, 3), (1, 3, 4)}, "B": {(1, 4, 5)}}
        durations = {"A": 1.0, "B": 2.0}
        result = minimize_test_set(test_arcs, durations)
        # A picked first (3 arcs / 1s vs 1 arc / 2s).
        assert result.chosen[0].nodeid == "A"

    def test_drops_redundant_test(self):
        # B is a strict subset of A — never picked.
        test_arcs = {"A": {(1, 1, 2), (1, 2, 3)}, "B": {(1, 1, 2)}}
        durations = {"A": 1.0, "B": 1.0}
        result = minimize_test_set(test_arcs, durations)
        chosen_ids = {c.nodeid for c in result.chosen}
        assert chosen_ids == {"A"}
        assert not result.uncovered_arcs

    def test_keeps_one_slow_when_uniquely_covers(self):
        # Slow test owns the only path to arc (1, 9, 9).
        test_arcs = {
            "fast_a": {(1, 1, 2)},
            "slow_unique": {(1, 9, 9)},
        }
        durations = {"fast_a": 1.0, "slow_unique": 60.0}
        result = minimize_test_set(test_arcs, durations)
        chosen_ids = {c.nodeid for c in result.chosen}
        assert chosen_ids == {"fast_a", "slow_unique"}

    def test_replaces_one_slow_with_many_fast(self):
        # A=60s covers {1,2,3}; B+C=20s covers {1,2,3} together — pick B+C.
        test_arcs = {
            "A": {(1, 1, 2), (1, 2, 3), (1, 3, 4)},
            "B": {(1, 1, 2), (1, 2, 3)},
            "C": {(1, 3, 4)},
        }
        durations = {"A": 60.0, "B": 10.0, "C": 10.0}
        result = minimize_test_set(test_arcs, durations)
        chosen_ids = {c.nodeid for c in result.chosen}
        assert chosen_ids == {"B", "C"}
        assert result.total_duration_s == 20.0

    def test_missing_duration_excluded_and_reported(self):
        # B has no duration entry — most likely errored/killed. It is excluded
        # from the analysis (neither chosen nor redundant) and reported. Its
        # arcs leave the universe entirely, so uncovered_arcs stays empty —
        # the missing_duration set is the signal to rerun.
        test_arcs = {"A": {(1, 1, 2)}, "B": {(1, 2, 3)}}
        durations = {"A": 1.0}  # B missing
        result = minimize_test_set(test_arcs, durations)
        assert result.missing_duration == {"B"}
        assert {c.nodeid for c in result.chosen} == {"A"}
        assert "B" not in result.redundant
        assert not result.uncovered_arcs

    def test_redundant_tests_reported(self):
        # B is a strict subset of A and has a duration → redundant, not chosen.
        test_arcs = {"A": {(1, 1, 2), (1, 2, 3)}, "B": {(1, 1, 2)}}
        durations = {"A": 1.0, "B": 1.0}
        result = minimize_test_set(test_arcs, durations)
        assert {c.nodeid for c in result.chosen} == {"A"}
        assert result.redundant == {"B"}
        assert not result.missing_duration

    def test_total_duration_sums_chosen(self):
        test_arcs = {"A": {(1, 1, 2)}, "B": {(1, 2, 3)}, "C": {(1, 3, 4)}}
        durations = {"A": 1.0, "B": 2.0, "C": 4.0}
        result = minimize_test_set(test_arcs, durations)
        assert result.total_duration_s == sum(c.duration_s for c in result.chosen)

    def test_empty_test_arcs_returns_empty_result(self):
        result = minimize_test_set({}, {})
        assert result.chosen == []
        assert not result.uncovered_arcs
        assert result.total_duration_s == 0.0

    def test_returns_minimization_result_instance(self):
        result = minimize_test_set({"A": {(1, 1, 2)}}, {"A": 1.0})
        assert isinstance(result, MinimizationResult)

    def test_chosen_set_and_order_stable_across_dict_orderings(self):
        # Output must be reproducible across re-runs; tie-breaks must
        # not depend on dict insertion order. A, B, C all have score =
        # 1 arc / 1 sec.
        arcs_in_order_1 = {
            "A": {(1, 1, 2)},
            "B": {(1, 2, 3)},
            "C": {(1, 3, 4)},
        }
        arcs_in_order_2 = {
            "C": {(1, 3, 4)},
            "A": {(1, 1, 2)},
            "B": {(1, 2, 3)},
        }
        durations = {"A": 1.0, "B": 1.0, "C": 1.0}
        r1 = minimize_test_set(arcs_in_order_1, durations)
        r2 = minimize_test_set(arcs_in_order_2, durations)
        # Both the membership AND the pick order must match.
        assert [c.nodeid for c in r1.chosen] == [c.nodeid for c in r2.chosen]


# ---------------------------------------------------------------------------
# minimize_test_set: memory tiebreak behavior
# ---------------------------------------------------------------------------


class TestMemoryTiebreak:
    """Peak memory only breaks ties between equal-score tests; it never excludes."""

    def test_lighter_test_wins_a_tie(self):
        # Both cover the same single arc at the same cost — equal score.
        # The lower-memory test should be chosen.
        test_arcs = {"heavy": {(1, 1, 2)}, "light": {(1, 1, 2)}}
        durations = {"heavy": 1.0, "light": 1.0}
        peak = {"heavy": 3000.0, "light": 500.0}
        result = minimize_test_set(test_arcs, durations, peak_memory_mb=peak)
        chosen_ids = {c.nodeid for c in result.chosen}
        assert chosen_ids == {"light"}

    def test_heavy_test_still_eligible_and_chosen_when_unique(self):
        # A high-memory test that uniquely covers an arc is NOT excluded —
        # memory is not a cap. Both arcs must be covered.
        test_arcs = {"light": {(1, 1, 2)}, "heavy_uniq": {(1, 4, 5)}}
        durations = {"light": 1.0, "heavy_uniq": 1.0}
        peak = {"light": 500.0, "heavy_uniq": 3000.0}
        result = minimize_test_set(test_arcs, durations, peak_memory_mb=peak)
        chosen_ids = {c.nodeid for c in result.chosen}
        assert chosen_ids == {"light", "heavy_uniq"}
        assert not result.uncovered_arcs

    def test_memory_does_not_override_score(self):
        # A heavier test with a strictly better arcs/sec score still wins —
        # memory only matters on an exact score tie.
        test_arcs = {"heavy_better": {(1, 1, 2), (1, 2, 3)}, "light_worse": {(1, 1, 2)}}
        durations = {"heavy_better": 1.0, "light_worse": 1.0}
        peak = {"heavy_better": 3000.0, "light_worse": 100.0}
        result = minimize_test_set(test_arcs, durations, peak_memory_mb=peak)
        # heavy_better covers both arcs in one pick; light_worse is redundant.
        assert {c.nodeid for c in result.chosen} == {"heavy_better"}

    def test_no_peak_memory_falls_back_to_nodeid_order(self):
        # Without peak data, an exact tie breaks by sorted nodeid (the
        # deterministic default) — "A" before "B".
        test_arcs = {"B": {(1, 1, 2)}, "A": {(1, 1, 2)}}
        durations = {"A": 1.0, "B": 1.0}
        result = minimize_test_set(test_arcs, durations, peak_memory_mb=None)
        assert {c.nodeid for c in result.chosen} == {"A"}

    def test_missing_memory_entry_does_not_exclude(self):
        # A test absent from the peak dict is still fully eligible; it just
        # doesn't win ties on memory.
        test_arcs = {"measured": {(1, 1, 2)}, "unmeasured": {(1, 4, 5)}}
        durations = {"measured": 1.0, "unmeasured": 1.0}
        peak = {"measured": 500.0}  # unmeasured intentionally absent
        result = minimize_test_set(test_arcs, durations, peak_memory_mb=peak)
        assert {c.nodeid for c in result.chosen} == {"measured", "unmeasured"}
        assert not result.uncovered_arcs


# ---------------------------------------------------------------------------
# minimize_test_set: mandatory inclusion
# ---------------------------------------------------------------------------


class TestMandatoryInclusion:
    """`mandatory` tests are always picked, before the greedy runs."""

    def test_mandatory_test_force_included(self):
        # B is strictly redundant with A but mandatory keeps it.
        test_arcs = {"A": {(1, 1, 2), (1, 2, 3)}, "B": {(1, 1, 2)}}
        durations = {"A": 1.0, "B": 1.0}
        result = minimize_test_set(test_arcs, durations, mandatory={"B"})
        chosen_ids = {c.nodeid for c in result.chosen}
        assert "B" in chosen_ids

    def test_mandatory_arcs_removed_before_greedy(self):
        # After mandatory adds A, the greedy shouldn't pick C (already covered).
        test_arcs = {
            "A": {(1, 1, 2), (1, 2, 3)},
            "C": {(1, 1, 2)},  # subset of A
            "D": {(1, 9, 9)},  # genuinely new arc
        }
        durations = {"A": 1.0, "C": 0.5, "D": 1.0}
        result = minimize_test_set(test_arcs, durations, mandatory={"A"})
        chosen_ids = {c.nodeid for c in result.chosen}
        assert chosen_ids == {"A", "D"}

    def test_unknown_mandatory_nodeid_ignored(self):
        # Nodeid in mandatory but not in test_arcs — silently skipped.
        test_arcs = {"A": {(1, 1, 2)}}
        durations = {"A": 1.0}
        result = minimize_test_set(test_arcs, durations, mandatory={"does_not_exist"})
        chosen_ids = {c.nodeid for c in result.chosen}
        assert chosen_ids == {"A"}


# ---------------------------------------------------------------------------
# load_per_test_arcs: real coverage SQLite db
# ---------------------------------------------------------------------------


def _build_coverage_db(tmp_path: Path, files_to_arcs: dict[str, list[tuple[int, int, int]]]) -> str:
    """Build a real .coverage SQLite db with two test contexts.

    Splits the arcs of each file roughly half-and-half between two test
    contexts so that load_per_test_arcs has interesting per-test data.
    """
    import coverage

    db_path = tmp_path / "test.coverage"
    cov_data = coverage.CoverageData(basename=str(db_path))

    # Split arcs per file into two test contexts.
    for ctx in ("test_a|run", "test_b|run"):
        cov_data.set_context(ctx)
        per_file: dict[str, list[tuple[int, int]]] = {}
        for fpath, arcs in files_to_arcs.items():
            # Each context picks a different slice — overlap is fine.
            picks = arcs if ctx == "test_a|run" else arcs[1:] + arcs[:1]
            # `arc` storage is (file, [(fromno, tono), ...])
            per_file[fpath] = [(fr, to) for (_fid, fr, to) in picks]
        cov_data.add_arcs(per_file)
    cov_data.write()
    return str(db_path)


class TestLoadPerTestArcs:
    """End-to-end SQLite read using a real coverage.CoverageData db."""

    def test_loads_arcs_from_real_sqlite(self, tmp_path):
        db = _build_coverage_db(
            tmp_path,
            {
                "/src/nkilib/core/widget/widget.py": [(0, 10, 11), (0, 11, 12)],
            },
        )
        cov = load_per_test_arcs(db, "widget")
        test_arcs, all_arcs, _ = cov.test_arcs, cov.all_arcs, cov.test_outcomes
        assert len(test_arcs) == 2  # test_a and test_b
        assert all(nodeid in test_arcs for nodeid in ("test_a", "test_b"))
        assert all_arcs  # something landed
        assert all(arc[1] in (10, 11) for arc in all_arcs)

    def test_filters_torch_files_out(self, tmp_path):
        db = _build_coverage_db(
            tmp_path,
            {
                "/src/nkilib/core/widget/widget.py": [(0, 10, 11)],
                "/src/nkilib/core/widget/widget_torch.py": [(0, 99, 100)],
            },
        )
        all_arcs = load_per_test_arcs(db, "widget").all_arcs
        # The torch file's arcs should not appear.
        assert all(arc[1] != 99 for arc in all_arcs)

    def test_raises_when_no_files_match(self, tmp_path):
        db = _build_coverage_db(
            tmp_path,
            {"/src/nkilib/core/widget/widget.py": [(0, 10, 11)]},
        )
        with pytest.raises(ValueError, match="match"):
            load_per_test_arcs(db, "no_such_kernel")

    def test_kernel_filter_substring_match_is_loose(self, tmp_path):
        # `kernel_filter in p` is a plain substring check — it will match
        # any path containing the substring, not just path-component
        # boundaries. Pin the behavior so a future tightening is a
        # deliberate change, not a silent regression. Callers should use
        # specific filters like "core/rope/rope.py" rather than "rope".
        db = _build_coverage_db(
            tmp_path,
            {
                "/src/nkilib/core/rope/rope.py": [(0, 10, 11)],
                "/src/nkilib/core/some_prope_helper.py": [(0, 99, 100)],
            },
        )
        all_arcs = load_per_test_arcs(db, "rope").all_arcs
        # Both files match — arc 99 from the unrelated file slipped in.
        assert any(arc[1] == 99 for arc in all_arcs)

    def test_skips_non_test_contexts(self, tmp_path):
        # Build a db where one context has no "|" — the loader should skip it.
        import coverage

        db_path = tmp_path / "test.coverage"
        cov_data = coverage.CoverageData(basename=str(db_path))
        cov_data.set_context("test_a|run")
        cov_data.add_arcs({"/src/nkilib/core/widget/widget.py": [(10, 11)]})
        cov_data.set_context("")  # empty context — no "|"
        cov_data.add_arcs({"/src/nkilib/core/widget/widget.py": [(20, 21)]})
        cov_data.write()

        test_arcs = load_per_test_arcs(str(db_path), "widget").test_arcs
        assert "test_a" in test_arcs
        # "" context should NOT have produced an entry.
        assert "" not in test_arcs


# ---------------------------------------------------------------------------
# load_csv_metric
# ---------------------------------------------------------------------------


class TestLoadCsvMetric:
    """Same loader handles durations CSV (col 2) and memory CSV (col 1)."""

    def test_loads_durations_csv_value_column_2(self, tmp_path):
        csv_path = tmp_path / "durations.csv"
        csv_path.write_text(
            textwrap.dedent(
                """\
                nodeid,phase,duration
                test_x,call,1.5
                test_y,call,3.0
                """
            )
        )
        result = load_csv_metric(str(csv_path), value_column=2)
        assert result == {"test_x": 1.5, "test_y": 3.0}

    def test_loads_memory_csv_value_column_1(self, tmp_path):
        csv_path = tmp_path / "memory.csv"
        csv_path.write_text(
            textwrap.dedent(
                """\
                test_id,peak_mb,delta_mb
                test_x,1500.0,800.0
                test_y,2200.0,1900.0
                """
            )
        )
        result = load_csv_metric(str(csv_path), value_column=1)
        assert result == {"test_x": 1500.0, "test_y": 2200.0}

    def test_returns_empty_dict_when_file_missing(self, tmp_path):
        nonexistent = tmp_path / "nope.csv"
        assert load_csv_metric(str(nonexistent)) == {}

    def test_skips_unparseable_rows(self, tmp_path):
        csv_path = tmp_path / "durations.csv"
        csv_path.write_text(
            textwrap.dedent(
                """\
                nodeid,phase,duration
                test_x,call,1.5
                test_bad,call,not-a-number
                test_y,call,3.0
                """
            )
        )
        result = load_csv_metric(str(csv_path), value_column=2)
        assert result == {"test_x": 1.5, "test_y": 3.0}


# ---------------------------------------------------------------------------
# _lookup: nodeid normalization
# ---------------------------------------------------------------------------


class TestLookup:
    """`|run` suffix stripping and substring fallback; ``None`` on a miss."""

    def test_strips_run_suffix_from_nodeid(self):
        table = {"test_x": 5.0}
        assert _lookup("test_x|run", table) == 5.0

    def test_substring_fallback(self):
        # Coverage capture may produce "test/foo/bar.py::test_x" while the
        # durations CSV records just "bar.py::test_x" — substring fallback.
        table = {"bar.py::test_x": 5.0}
        assert _lookup("test/foo/bar.py::test_x", table) == 5.0

    def test_returns_none_on_miss(self):
        table = {"test_x": 5.0}
        assert _lookup("test_z", table) is None

    def test_exact_match_preferred_over_substring(self):
        table = {"bar.py::test_x": 5.0, "test_x": 1.0}
        assert _lookup("test_x", table) == 1.0
