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
"""Selection logic for ``--run-relevant-tests``.

Lives in a leaf module (not conftest.py) so it can be unit-tested in isolation
with a trivial fake ``config`` -- no pytest session, no xdist, no import of the
heavy conftest. conftest.py wires these into the pytest hooks.

The trap this module encapsulates: the relevant-test selection is computed once
on the xdist master, but collection (where filtering happens) runs on separate
worker processes that do NOT inherit the master's ``config`` attributes. The
master ships the result via ``workerinput``; each worker must adopt it. Keeping
the compute/adopt/apply steps here makes that whole contract directly testable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Attribute name under which the resolved selection is stored on pytest's config.
RELEVANT_TEST_DIRS_KEY = "_relevant_test_dirs"

# Key under which the master ships the selection to xdist workers via workerinput.
WORKERINPUT_KEY = "relevant_test_dirs"

# Sentinel distinguishing "attribute never set" (flag off) from an explicit None
# ("flag on, but run everything"). Both mean "do not filter", but only the former
# should be silent.
_UNSET = "NOT_SET"


def resolve_relevant_test_dirs(config: Any, repo_root: Path, get_option) -> None:
    """Resolve the selection onto ``config`` so readers are process-role-agnostic.

    Sets ``config._relevant_test_dirs`` to a collection of dirs (flag on, narrowed),
    ``None`` (flag on, run all), or leaves it unset (flag off).

    - xdist worker: adopt the value the master shipped via ``workerinput``.
    - master / non-xdist: run the finder when ``--run-relevant-tests`` is set.

    Args:
        config: pytest Config (or any object; only attribute/``workerinput`` access
            is used, which keeps this unit-testable with a fake).
        repo_root: repository root passed to the finder.
        get_option: callable ``(config, "run_relevant_tests") -> commit spec | None``
            (injected so tests don't need pytest's option machinery).
    """
    if hasattr(config, "workerinput"):
        # xdist worker: collection happens here, but the finder ran on the master.
        # Adopt its shipped result. Absent key => flag was off => leave attr unset.
        if WORKERINPUT_KEY in config.workerinput:
            setattr(config, RELEVANT_TEST_DIRS_KEY, config.workerinput[WORKERINPUT_KEY])
        return

    commit_ids = get_option(config, "run_relevant_tests")
    if not commit_ids:
        return

    # Imported lazily: keeps this module import-light and avoids a cycle if the
    # finder ever imports selection helpers.
    from .finder import RelevantTestFinder

    relevant_dirs = RelevantTestFinder(repo_root=repo_root).get_relevant_test_dirs(commit_ids)
    setattr(config, RELEVANT_TEST_DIRS_KEY, relevant_dirs)
    if relevant_dirs is not None:
        logger.info("Relevant test directories: %s", relevant_dirs)
    else:
        logger.info("Running all tests (infrastructure change or fallback)")


def ship_relevant_test_dirs_to_worker(node) -> None:
    """Copy the master's resolved selection into a worker's ``workerinput``.

    Called from ``pytest_configure_node`` (runs on the master, per worker). The
    set is converted to a list because xdist serializes ``workerinput`` and only
    handles simple types. A ``None`` (run-all) is shipped as-is.
    """
    if hasattr(node.config, RELEVANT_TEST_DIRS_KEY):
        value = getattr(node.config, RELEVANT_TEST_DIRS_KEY)
        node.workerinput[WORKERINPUT_KEY] = list(value) if value is not None else None


def partition_items_by_relevance(config: Any, items: list) -> tuple[list, list] | None:
    """Split collected ``items`` into (kept, deselected) per the resolved selection.

    Returns ``None`` when no filtering should happen (flag off, or run-all None),
    so the caller can no-op. Reads only ``config._relevant_test_dirs`` -- the same
    attribute in every process, since resolve_relevant_test_dirs already normalized it.
    """
    relevant_dirs = getattr(config, RELEVANT_TEST_DIRS_KEY, _UNSET)
    if relevant_dirs == _UNSET or relevant_dirs is None:
        return None

    kept, deselected = [], []
    for item in items:
        if _item_is_relevant(str(item.fspath), relevant_dirs):
            kept.append(item)
        else:
            deselected.append(item)
    return kept, deselected


def _item_is_relevant(fspath: str, relevant_dirs: Iterable[str]) -> bool:
    return any(fspath.startswith(d) for d in relevant_dirs)
