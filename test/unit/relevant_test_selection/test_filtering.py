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
"""Standalone unit tests for the --run-relevant-tests selection logic.

Depends only on test.utils.relevant_test_selection.filtering -- no conftest import, no
pytest session, no xdist, no on-disk test tree. A ``_FakeConfig`` stands in for
pytest's Config: presence of ``workerinput`` is what distinguishes an xdist
worker from the master, which is the whole crux of the bug these tests guard.
"""

from pathlib import Path

from test.utils.relevant_test_selection.filtering import (
    RELEVANT_TEST_DIRS_KEY,
    WORKERINPUT_KEY,
    partition_items_by_relevance,
    resolve_relevant_test_dirs,
    ship_relevant_test_dirs_to_worker,
)

_REL = "/repo/test/integration/nkilib/experimental/mxfp_subkernels"
_OTHER = "/repo/test/integration/nkilib/core/attention"


class _FakeConfig:
    """Minimal Config stand-in. ``workerinput`` present => xdist worker."""

    def __init__(self, workerinput=None):
        if workerinput is not None:
            self.workerinput = workerinput


class _FakeNode:
    """Stand-in for the xdist node passed to pytest_configure_node."""

    def __init__(self, config):
        self.config = config
        self.workerinput = {}


class _FakeItem:
    def __init__(self, path):
        self.path = path


def _never_called(config, name):  # get_option that must not run on the worker path
    raise AssertionError("finder/get_option must not be consulted on an xdist worker")


def test_worker_adopts_selection_from_master():
    # Master resolved a narrowed selection (set directly, standing in for the finder).
    master = _FakeConfig()
    setattr(master, RELEVANT_TEST_DIRS_KEY, {_REL})

    # Master ships it to a worker.
    node = _FakeNode(master)
    ship_relevant_test_dirs_to_worker(node)
    assert node.workerinput[WORKERINPUT_KEY] == [_REL]

    # Fresh worker process: config has workerinput, NOT the attribute.
    worker = _FakeConfig(workerinput=node.workerinput)
    assert not hasattr(worker, RELEVANT_TEST_DIRS_KEY)

    resolve_relevant_test_dirs(worker, Path("/repo"), _never_called)

    # The bug (no adoption) leaves this unset -> no filtering downstream.
    assert getattr(worker, RELEVANT_TEST_DIRS_KEY) == [_REL]


def test_worker_adopts_run_all_sentinel():
    """Master's run-all decision (None) must reach the worker as None, not 'unset'."""
    master = _FakeConfig()
    setattr(master, RELEVANT_TEST_DIRS_KEY, None)
    node = _FakeNode(master)
    ship_relevant_test_dirs_to_worker(node)
    assert node.workerinput[WORKERINPUT_KEY] is None

    worker = _FakeConfig(workerinput=node.workerinput)
    resolve_relevant_test_dirs(worker, Path("/repo"), _never_called)
    assert getattr(worker, RELEVANT_TEST_DIRS_KEY) is None


def test_worker_stays_unset_when_flag_off():
    """Flag off: master ships nothing, worker sets no attribute."""
    master = _FakeConfig()  # no attribute set (flag off)
    node = _FakeNode(master)
    ship_relevant_test_dirs_to_worker(node)
    assert WORKERINPUT_KEY not in node.workerinput

    worker = _FakeConfig(workerinput={})  # worker, no shipped key
    resolve_relevant_test_dirs(worker, Path("/repo"), _never_called)
    assert not hasattr(worker, RELEVANT_TEST_DIRS_KEY)


def test_master_flag_off_sets_nothing():
    config = _FakeConfig()  # master
    resolve_relevant_test_dirs(config, Path("/repo"), lambda c, n: None)  # flag off
    assert not hasattr(config, RELEVANT_TEST_DIRS_KEY)


def _items():
    return [_FakeItem(f"{_REL}/test_mxfp_load.py"), _FakeItem(f"{_OTHER}/test_attention_cte.py")]


def test_partition_keeps_only_relevant():
    config = _FakeConfig()
    setattr(config, RELEVANT_TEST_DIRS_KEY, {_REL})
    partition = partition_items_by_relevance(config, _items())
    assert partition is not None, "a relevant-dirs set was just installed, so filtering must apply"
    kept, deselected = partition
    assert [str(i.path) for i in kept] == [f"{_REL}/test_mxfp_load.py"]
    assert [str(i.path) for i in deselected] == [f"{_OTHER}/test_attention_cte.py"]


def test_partition_noop_when_unset():
    config = _FakeConfig()  # attribute absent -> flag off
    assert partition_items_by_relevance(config, _items()) is None


def test_partition_noop_when_run_all():
    config = _FakeConfig()
    setattr(config, RELEVANT_TEST_DIRS_KEY, None)
    assert partition_items_by_relevance(config, _items()) is None
