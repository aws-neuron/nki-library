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

"""Unit tests for distributed adapter and SimDistRunner thread safety."""

import threading
import time
from unittest.mock import MagicMock, patch

import nkilib_src.nkilib.experimental.collectives.distributed_adapter as adapter_mod
import numpy as np
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.distributed_adapter import (
    SimDistAdapter,
    TorchDistAdapter,
    _get_adapter,
    get_pg,
    get_rank,
    set_adapter,
)

from test.utils.unit_test_collective_framework import SimDistRunner


class TestDistributedAdapterThreadSafety:
    """Verify no race conditions in get_rank()/get_pg() under concurrent SimDistRunner usage."""

    def test_concurrent_get_rank_isolation(self):
        """Each thread sees its own rank, not another thread's."""
        num_ranks = 4
        rg = ReplicaGroup([list(range(num_ranks))])
        runner = SimDistRunner(num_ranks=num_ranks, replica_groups=[rg])

        def torch_ref(replica_groups=None):
            rank = get_rank()
            # Sleep briefly to increase chance of interleaving

            time.sleep(0.01)
            # Verify rank hasn't changed
            assert get_rank() == rank, f"Rank changed mid-execution: was {rank}, now {get_rank()}"
            return {"rank": np.array([rank])}

        per_rank_inputs = {r: {"replica_groups": rg} for r in range(num_ranks)}
        results = runner.run(torch_ref, per_rank_inputs)

        # Each rank should report its own rank
        for r in range(num_ranks):
            assert results[r]["rank"][0] == r

    def test_concurrent_get_pg_isolation(self):
        """Each thread gets the correct process group for its rank."""
        num_ranks = 4
        rg = ReplicaGroup([list(range(num_ranks))])
        runner = SimDistRunner(num_ranks=num_ranks, replica_groups=[rg])

        def torch_ref(replica_groups: ReplicaGroup):
            rank = get_rank()
            pg = get_pg(replica_groups)
            assert pg.rank() == rank
            assert pg.size() == num_ranks
            return {"rank": np.array([rank]), "pg_rank": np.array([pg.rank()])}

        per_rank_inputs = {r: {"replica_groups": rg} for r in range(num_ranks)}
        results = runner.run(torch_ref, per_rank_inputs)

        for r in range(num_ranks):
            assert results[r]["pg_rank"][0] == r

    def test_multi_group_isolation(self):
        """Multiple sub-groups don't interfere with each other."""
        # 4 ranks, 2 groups: [0,1] and [2,3]
        rg = ReplicaGroup([[0, 1], [2, 3]])
        runner = SimDistRunner(num_ranks=4, replica_groups=[rg])

        def torch_ref(replica_groups: ReplicaGroup):
            rank = get_rank()
            pg = get_pg(replica_groups)
            # Each sub-group has size 2
            assert pg.size() == 2
            # pg.rank() is position within sub-group
            expected_pg_rank = rank % 2
            assert pg.rank() == expected_pg_rank
            return {"rank": np.array([rank]), "pg_rank": np.array([pg.rank()])}

        per_rank_inputs = {r: {"replica_groups": rg} for r in range(4)}
        results = runner.run(torch_ref, per_rank_inputs)

        assert results[0]["pg_rank"][0] == 0
        assert results[1]["pg_rank"][0] == 1
        assert results[2]["pg_rank"][0] == 0
        assert results[3]["pg_rank"][0] == 1

    def test_sequential_runs_no_stale_adapter(self):
        """After SimDistRunner.run() completes, threads don't have stale adapters."""
        num_ranks = 2
        rg = ReplicaGroup([list(range(num_ranks))])
        runner = SimDistRunner(num_ranks=num_ranks, replica_groups=[rg])

        def torch_ref(replica_groups=None):
            return {"rank": np.array([get_rank()])}

        per_rank_inputs = {r: {"replica_groups": rg} for r in range(num_ranks)}

        # Run twice — second run should not see stale state from first
        results1 = runner.run(torch_ref, per_rank_inputs)
        results2 = runner.run(torch_ref, per_rank_inputs)

        for r in range(num_ranks):
            assert results1[r]["rank"][0] == r
            assert results2[r]["rank"][0] == r

    def test_two_runners_concurrent(self):
        """Two SimDistRunners running concurrently on different thread pools don't interfere."""
        rg1 = ReplicaGroup([[0, 1]])
        rg2 = ReplicaGroup([[0, 1, 2, 3]])

        runner1 = SimDistRunner(num_ranks=2, replica_groups=[rg1])
        runner2 = SimDistRunner(num_ranks=4, replica_groups=[rg2])

        results = {}

        def run1():
            def torch_ref(replica_groups: ReplicaGroup):
                import time

                time.sleep(0.02)
                return {"rank": np.array([get_rank()]), "size": np.array([get_pg(replica_groups).size()])}

            results["r1"] = runner1.run(torch_ref, {r: {"replica_groups": rg1} for r in range(2)})

        def run2():
            def torch_ref(replica_groups: ReplicaGroup):
                import time

                time.sleep(0.02)
                return {"rank": np.array([get_rank()]), "size": np.array([get_pg(replica_groups).size()])}

            results["r2"] = runner2.run(torch_ref, {r: {"replica_groups": rg2} for r in range(4)})

        t1 = threading.Thread(target=run1)
        t2 = threading.Thread(target=run2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Runner1: 2 ranks, size 2
        for r in range(2):
            assert results["r1"][r]["rank"][0] == r
            assert results["r1"][r]["size"][0] == 2

        # Runner2: 4 ranks, size 4
        for r in range(4):
            assert results["r2"][r]["rank"][0] == r
            assert results["r2"][r]["size"][0] == 4

    def test_allgather_correctness_under_concurrency(self):
        """all_gather produces correct results when multiple ranks run concurrently."""
        num_ranks = 4
        rg = ReplicaGroup([list(range(num_ranks))])
        runner = SimDistRunner(num_ranks=num_ranks, replica_groups=[rg])

        def torch_ref(replica_groups: ReplicaGroup, data=None):
            get_rank()
            pg = get_pg(replica_groups)
            assert data is not None, "the reference requires its input array"
            t = torch.from_numpy(data.astype(np.float32))
            gathered = [torch.zeros_like(t) for _ in range(num_ranks)]
            dist.all_gather(gathered, t, group=pg)
            return {"gathered": torch.cat(gathered).numpy()}

        per_rank_inputs = {
            r: {"data": np.array([r * 10.0, r * 10.0 + 1.0]), "replica_groups": rg} for r in range(num_ranks)
        }
        results = runner.run(torch_ref, per_rank_inputs)

        expected = np.array([0.0, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0])
        for r in range(num_ranks):
            np.testing.assert_array_equal(results[r]["gathered"], expected)

    def test_read_after_cleanup_raises(self):
        """get_rank() raises after adapter is cleared (no stale reads)."""
        num_ranks = 2
        rg = ReplicaGroup([list(range(num_ranks))])
        runner = SimDistRunner(num_ranks=num_ranks, replica_groups=[rg])

        errors_from_post_run = []

        def torch_ref(replica_groups=None):
            return {"rank": np.array([get_rank()])}

        per_rank_inputs = {r: {"replica_groups": rg} for r in range(num_ranks)}
        runner.run(torch_ref, per_rank_inputs)

        # After run completes, calling get_rank() on a new thread should fail
        # (no adapter set, dist not initialized)
        def try_get_rank():
            try:
                get_rank()
                errors_from_post_run.append("Should have raised")
            except RuntimeError:
                pass  # Expected

        t = threading.Thread(target=try_get_rank)
        t.start()
        t.join()
        assert not errors_from_post_run, errors_from_post_run

    def test_rapid_sequential_runs_no_bleed(self):
        """Rapid sequential runs don't bleed adapter state between runs."""
        rg = ReplicaGroup([[0, 1]])
        runner = SimDistRunner(num_ranks=2, replica_groups=[rg])

        def torch_ref(replica_groups=None, run_id=None):
            rank = get_rank()
            return {"rank": np.array([rank]), "run_id": np.array([run_id])}

        for run_id in range(20):
            per_rank_inputs = {r: {"replica_groups": rg, "run_id": run_id} for r in range(2)}
            results = runner.run(torch_ref, per_rank_inputs)
            assert results[0]["rank"][0] == 0
            assert results[1]["rank"][0] == 1
            assert results[0]["run_id"][0] == run_id
            assert results[1]["run_id"][0] == run_id

    def test_write_write_same_thread_last_wins(self):
        """If set_adapter is called twice on same thread, last write wins."""
        ReplicaGroup([[0, 1, 2, 3]])

        # Manually test that overwriting adapter on same thread works correctly
        results = {}

        def worker():
            # First write
            set_adapter(SimDistAdapter(0, {}))
            assert get_rank() == 0
            # Overwrite with different rank
            set_adapter(SimDistAdapter(3, {}))
            assert get_rank() == 3
            results["final_rank"] = get_rank()
            set_adapter(None)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert results["final_rank"] == 3


# ==================== Real dist backend tests ====================
# These mock torch.distributed to test TorchDistAdapter dispatch logic
# without requiring gloo/nccl.


class TestTorchDistAdapter:
    """Tests for TorchDistAdapter dispatch path using mocked torch.distributed."""

    def test_auto_detects_dist_backend(self):
        """_get_adapter() returns TorchDistAdapter when dist.is_initialized() is True."""

        with patch("nkilib_src.nkilib.experimental.collectives.distributed_adapter.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 5
            # Clear any cached _dist_backend

            old = adapter_mod._dist_backend
            adapter_mod._dist_backend = None
            try:
                adapter = _get_adapter()
                assert isinstance(adapter, TorchDistAdapter)
            finally:
                adapter_mod._dist_backend = old

    def test_get_rank_uses_dist_when_initialized(self):
        """get_rank() delegates to dist.get_rank() when dist is initialized."""

        with patch("nkilib_src.nkilib.experimental.collectives.distributed_adapter.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 7

            old = adapter_mod._dist_backend
            adapter_mod._dist_backend = None
            try:
                assert get_rank() == 7
            finally:
                adapter_mod._dist_backend = old

    def test_sim_adapter_takes_priority_over_dist(self):
        """Thread-local SimDistAdapter takes priority even when dist is initialized."""

        with patch("nkilib_src.nkilib.experimental.collectives.distributed_adapter.dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 99

            # Set sim adapter on this thread
            set_adapter(SimDistAdapter(3, {}))
            try:
                assert get_rank() == 3  # sim wins over dist
            finally:
                set_adapter(None)

    def test_dist_backend_singleton_under_concurrent_access(self):
        """_dist_backend is created only once even with concurrent threads."""

        with patch.object(adapter_mod, "dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 0
            adapter_mod._dist_backend = None

            adapters = []
            barrier = threading.Barrier(8)

            def get_adapter_concurrently():
                barrier.wait()  # all threads start at the same time
                adapters.append(_get_adapter())

            threads = [threading.Thread(target=get_adapter_concurrently) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All threads should get the same singleton instance
            assert all(a is adapters[0] for a in adapters)
            assert isinstance(adapters[0], TorchDistAdapter)
            adapter_mod._dist_backend = None

    def test_get_pg_creates_process_group_via_dist(self):
        """TorchDistAdapter.get_pg() calls dist.new_group and caches the result."""

        with patch.object(adapter_mod, "dist") as mock_dist:
            mock_dist.is_initialized.return_value = True
            mock_dist.get_rank.return_value = 0
            mock_pg = MagicMock()
            mock_dist.new_group.return_value = mock_pg
            adapter_mod._dist_backend = None

            try:
                rg = ReplicaGroup([[0, 1, 2]])
                pg = get_pg(rg)
                mock_dist.new_group.assert_called_once_with(ranks=[0, 1, 2])
                assert pg is mock_pg

                # Second call should use cache, not call new_group again
                pg2 = get_pg(rg)
                assert pg2 is mock_pg
                assert mock_dist.new_group.call_count == 1
            finally:
                adapter_mod._dist_backend = None
