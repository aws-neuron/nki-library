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
"""Cost-weighted greedy set-cover for minimizing a test set under
branch-coverage constraints.

Given per-test branch arcs and per-test costs (e.g. compile time), find
the smallest-cost subset that preserves the union of arcs covered by
the full input set. The algorithm is generic: any test tier (fast,
slow, model, etc.) works as long as its arcs and costs come from a
``coverage.py`` SQLite db and a costs CSV.

The greedy iteratively picks the test with the highest
``new_arcs_covered / cost`` ratio until all arcs are covered. Peak
memory, when provided, is used only as a tiebreaker between
otherwise-equal candidates (prefer the lighter test) — it does not
exclude any test. Memory is not a hard constraint here because the
dry-run already kills any fast test that exceeds its own
``--memory-limit``; this just nudges the chosen set toward the lighter
of two equivalent tests at no extra cost (the memory data piggybacks
on the coverage run).
"""

import csv
import os
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from typing import NamedTuple


class Arc(NamedTuple):
    """A branch arc as stored by coverage.py: an edge from line ``fromno`` to
    line ``tono`` within file ``file_id``. A NamedTuple so it stays hashable
    and usable in set operations while its fields are self-describing."""

    file_id: int
    fromno: int
    tono: int


@dataclass(frozen=True)
class ChosenTest:
    """A test picked for the minimum set, in pick order."""

    nodeid: str
    duration_s: float
    new_arcs: int  # arcs this test newly covered when it was picked


@dataclass
class PerTestArcs:
    """Per-test branch-arc coverage for one source file, from a coverage db."""

    test_arcs: dict[str, set[Arc]] = field(default_factory=dict)
    """``{nodeid: set[Arc]}`` — arcs each test covered. Feed to :func:`minimize_test_set`."""

    all_arcs: set[Arc] = field(default_factory=set)
    """Union of all arcs across tests (total branch count for sanity checks)."""

    test_outcomes: dict[str, str] = field(default_factory=dict)
    """``{nodeid: outcome_token}`` parsed from the coverage context name
    (e.g. ``"run"``). Not the pytest pass/fail outcome — see
    :func:`load_per_test_arcs`."""


@dataclass
class MinimizationResult:
    chosen: list[ChosenTest] = field(default_factory=list)
    """Tests picked for the minimum set, in pick order (set 1: cover unique arcs)."""

    redundant: set[str] = field(default_factory=set)
    """Tests not picked — their arcs are covered by the chosen set, or covered
    more cheaply by others (set 2: safe to drop)."""

    missing_duration: set[str] = field(default_factory=set)
    """Tests excluded from the analysis because they had no duration — most
    likely they errored, were killed, or skipped (a clean test always records
    a call-phase duration). Reported, not analyzed (set 3: investigate).
    Because they are excluded before the arc universe is built, an excluded
    test that was the *sole* cover for some branch silently removes that
    branch from the analysis — which is why this set must be surfaced and the
    flow rerun once the underlying failures are fixed."""

    uncovered_arcs: set[Arc] = field(default_factory=set)
    """Arcs left uncovered by the chosen set. Always empty by construction —
    the universe is built only from analyzed tests, each of which stays a
    candidate until picked — so a non-empty value indicates a solver bug.
    Exposed as a defensive invariant check."""

    total_duration_s: float = 0.0


def load_per_test_arcs(coverage_db_path: str, source_path_filter: str) -> PerTestArcs:
    """Read a coverage SQLite db and group per-test arcs by source-file match.

    Call once per source file of interest: a single ``.coverage`` db can
    hold arcs from many files (the test suite often imports several),
    so the caller filters down to the source whose tests they want to
    minimize.

    Args:
        coverage_db_path: Path to ``.coverage`` SQLite produced by
            ``pytest --cov-context=test --cov-branch``.
        source_path_filter: Substring matched against ``file.path``.
            Files whose path contains the filter and do NOT contain
            ``"torch"`` are considered the source of interest. **Use a
            path-distinguishing substring** like
            ``"core/cumsum/cumsum.py"`` rather than a bare module name
            like ``"cumsum"``: the match is a plain ``str.__contains__``
            so any path containing the substring is included
            (e.g. ``source_path_filter="rope"`` would also match
            ``some_prope_helper.py``).

    Returns:
        A :class:`PerTestArcs` with ``test_arcs``, ``all_arcs``, and
        ``test_outcomes``. The outcome token is parsed from the trailing
        part of the coverage context name (typically ``"run"``); it is
        **not** the pytest pass/fail outcome. For that, cross-reference
        an external manifest. Tests that failed mid-run with
        non-validation errors should be passed as ``mandatory=`` to
        :func:`minimize_test_set` since their recorded arcs are partial.

    Raises:
        ValueError: if no file in the db matches ``source_path_filter``.
        sqlite3.DatabaseError: if the db is missing, unreadable, or not a
            valid coverage database.
    """
    try:
        # URI mode=ro so a missing/locked file fails fast instead of
        # silently creating an empty db.
        uri = f"file:{coverage_db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, context FROM context")
            contexts = {r[0]: r[1] for r in cur.fetchall()}
            cur.execute("SELECT context_id, file_id, fromno, tono FROM arc")
            ctx_arcs: dict[int, set[Arc]] = defaultdict(set)
            for ctx_id, fid, fr, to in cur.fetchall():
                ctx_arcs[ctx_id].add(Arc(fid, fr, to))
            cur.execute("SELECT id, path FROM file")
            files = {r[0]: r[1] for r in cur.fetchall()}
    except sqlite3.Error as e:
        raise sqlite3.DatabaseError(f"Failed to read coverage db {coverage_db_path!r}: {e}") from e

    target_ids = {fid for fid, p in files.items() if source_path_filter in p and "torch" not in p}
    if not target_ids:
        raise ValueError(f"No files in {coverage_db_path} match {source_path_filter!r}")

    test_arcs: dict[str, set[Arc]] = defaultdict(set)
    test_outcomes: dict[str, str] = {}
    all_arcs: set[Arc] = set()
    for cid, name in contexts.items():
        if "|" not in name:
            continue
        nodeid = name.split("|")[0]
        outcome = name.rsplit("|", 1)[-1]
        test_outcomes.setdefault(nodeid, outcome)
        for arc in ctx_arcs[cid]:
            if arc.file_id in target_ids:
                test_arcs[nodeid].add(arc)
                all_arcs.add(arc)

    return PerTestArcs(test_arcs=dict(test_arcs), all_arcs=all_arcs, test_outcomes=test_outcomes)


def load_csv_metric(path: str, value_column: int = 2) -> dict[str, float]:
    """Load a ``test_id,...,<float_value>`` CSV into a ``{nodeid: value}`` dict.

    Used for both the durations CSV (``test_id,phase,duration``) and the
    memory CSV (``test_id,peak_mb,delta_mb``).
    """
    out: dict[str, float] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) > value_column:
                try:
                    out[row[0]] = float(row[value_column])
                except ValueError:
                    pass
    return out


def _lookup(nodeid: str, table: dict[str, float]) -> float | None:
    """Match a nodeid against a metrics dict, returning ``None`` on a miss.

    Coverage contexts often have a ``|run`` suffix; metric CSVs do not.
    Different prefixes between coverage capture and duration capture are
    also common, so fall through to substring matching after the exact
    lookup misses.
    """
    nodeid = nodeid.replace("|run", "")
    if nodeid in table:
        return table[nodeid]
    for k, v in table.items():
        if k.endswith(nodeid) or nodeid.endswith(k):
            return v
    return None


def minimize_test_set(
    test_arcs: dict[str, set[Arc]],
    durations: dict[str, float],
    peak_memory_mb: dict[str, float] | None = None,
    mandatory: set[str] | None = None,
) -> MinimizationResult:
    """Run cost-weighted greedy set cover, partitioning tests into three sets.

    Every test that recorded coverage lands in exactly one of:

    1. **chosen** — picked because it covers arcs nothing cheaper does.
    2. **redundant** — its arcs are covered by the chosen set (safe to drop).
    3. **missing_duration** — excluded from the analysis because it has no
       duration, which means it most likely errored, was killed, or skipped
       (a clean test always records a call-phase duration). These are
       reported for the caller to rerun/investigate, not analyzed — their
       coverage can't be trusted, so they are neither kept nor called
       redundant.

    Args:
        test_arcs: ``{nodeid: set[Arc]}`` from :func:`load_per_test_arcs`.
        durations: ``{nodeid: cost_seconds}`` from the durations CSV
            (typically compile time when minimizing for a compile-only
            tier). A test with no entry here is treated as set 3.
        peak_memory_mb: ``{nodeid: peak_memory_megabytes}`` from the
            memory CSV (the ``peak_mb`` column — PSS, proportional set
            size). Used **only as a tiebreaker**: when two candidates
            have the same arcs/sec score, the lower-memory one is picked.
            Never excludes a test — the dry-run's own ``--memory-limit``
            is the real guard against memory-hungry tests. ``None``
            disables the tiebreak (falls back to nodeid order).
        mandatory: Tests forced into the chosen set regardless of cost or
            duration (e.g. tests that uniquely cover branches but fail in
            simulation). They bypass the missing-duration exclusion.

    Returns:
        :class:`MinimizationResult` with ``chosen``, ``redundant``,
        ``missing_duration``, ``uncovered_arcs``, and ``total_duration_s``.
    """
    mandatory = set(mandatory or ())

    # Partition out set 3: tests with coverage but no duration (excluded from
    # analysis). Mandatory tests bypass this — they are forced in regardless.
    missing_duration = {
        nodeid for nodeid in test_arcs if nodeid not in mandatory and _lookup(nodeid, durations) is None
    }
    candidates: dict[str, set[Arc]] = {
        nodeid: arcs for nodeid, arcs in test_arcs.items() if nodeid not in missing_duration
    }

    chosen: list[ChosenTest] = []
    total_time = 0.0

    # Universe = arcs from analyzed (mandatory + eligible) tests only. Excluded
    # missing-duration tests contribute no arcs, so uncovered_arcs stays empty
    # by construction.
    all_arcs: set[Arc] = set()
    for arcs in candidates.values():
        all_arcs |= arcs
    remaining = set(all_arcs)

    # 1. Force-include mandatory tests (sorted for determinism).
    for nodeid in sorted(mandatory):
        if nodeid not in candidates:
            continue
        d = _lookup(nodeid, durations) or 0.0  # mandatory may lack a duration
        new_arcs = candidates[nodeid] & remaining
        chosen.append(ChosenTest(nodeid=nodeid, duration_s=d, new_arcs=len(new_arcs)))
        remaining -= candidates[nodeid]
        del candidates[nodeid]
        total_time += d

    # 2. Greedy weighted set cover over the eligible pool (all have durations).
    # Iterate in sorted nodeid order so ties break deterministically and the
    # same inputs always yield the same chosen set + order. When peak memory
    # is available, prefer the lighter test among equal-score candidates
    # (a soft tiebreak, never an exclusion).
    while remaining and candidates:
        best, best_score, best_new, best_mem = None, -1.0, 0, None
        for t, arcs in sorted(candidates.items()):
            new = len(arcs & remaining)
            if new == 0:
                continue
            d = _lookup(t, durations)  # guaranteed present (eligible pool)
            score = new / max(d, 0.1)
            mem = _lookup(t, peak_memory_mb) if peak_memory_mb is not None else None
            if score > best_score:
                best, best_score, best_new, best_mem = t, score, new, mem
            elif score == best_score and mem is not None and best_mem is not None and mem < best_mem:
                # Tie on arcs/sec: prefer the lighter test.
                best, best_new, best_mem = t, new, mem
        if best is None:
            break
        d = _lookup(best, durations)
        chosen.append(ChosenTest(nodeid=best, duration_s=d, new_arcs=best_new))
        remaining -= candidates[best]
        del candidates[best]
        total_time += d

    # Set 2: every eligible test that wasn't chosen is redundant.
    chosen_ids = {c.nodeid for c in chosen}
    redundant = {nodeid for nodeid in test_arcs if nodeid not in chosen_ids and nodeid not in missing_duration}

    return MinimizationResult(
        chosen=chosen,
        redundant=redundant,
        missing_duration=missing_duration,
        uncovered_arcs=remaining,
        total_duration_s=total_time,
    )
