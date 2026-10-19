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
"""Unit tests for the host_state.json store and its mutators."""

import os
import random
import tempfile
from dataclasses import replace

import pytest

from test.utils.common_dataclasses import PYTEST_XDIST_WORKER_ENV, Platforms, ResolvedHost
from test.utils.exceptions import FleetEmptyError, RequestTooLargeError
from test.utils.host_state import (
    MULTINODE_CORE_THRESHOLD,
    HostGroup,
    HostRecord,
    HostState,
    HostStateStore,
    OwnerHostStateStore,
    temporary_random_seed,
)

pytestmark = pytest.mark.usefixtures("controller_environment")


def _resolved(*aliases: str, num_physical_cores: int = 64) -> list[ResolvedHost]:
    """Build a ResolvedHost list (all TRN2). These are good-to-use hosts — sources
    filter to reachable ones before the store ever sees them. Defaults to 64 cores so
    the hosts are eligible for typical (small) capacity requests."""
    return [ResolvedHost(ssh_host=a, host_type=Platforms.TRN2, num_physical_cores=num_physical_cores) for a in aliases]


def _record(
    alias: str,
    host_type: Platforms = Platforms.TRN2,
    *,
    work_queue_depth: int = 0,
    available: bool = True,
    num_physical_cores: int = 0,
) -> HostRecord:
    """Build a HostRecord — a ResolvedHost (alias/platform/cores) augmented with the store's
    runtime bookkeeping (work_queue_depth, available)."""
    resolved = ResolvedHost(ssh_host=alias, host_type=host_type, num_physical_cores=num_physical_cores)
    return HostRecord(resolved=resolved, work_queue_depth=work_queue_depth, available=available)


def _member(
    alias: str,
    logical_host_id: str,
    rank: int | None,
    *,
    size: int = 2,
    num_physical_cores: int = 128,
    work_queue_depth: int = 0,
    available: bool = True,
    host_type: Platforms = Platforms.TRN2,
) -> HostRecord:
    """Build a HostRecord belonging to multi-node deployment ``logical_host_id``."""
    resolved = ResolvedHost(
        ssh_host=alias,
        host_type=host_type,
        num_physical_cores=num_physical_cores,
        logical_host_id=logical_host_id,
        logical_host_rank=rank,
        logical_host_size=size,
    )
    return HostRecord(resolved=resolved, work_queue_depth=work_queue_depth, available=available)


def _depths(store) -> dict[str, int]:
    return {h.resolved.ssh_host: h.work_queue_depth for h in store.read().hosts}


@pytest.fixture
def store():
    # OwnerHostStateStore so these tests can exercise the controller-only mutations
    # (reset/apply_resolved/set_poisoned) alongside the base participant ops it inherits.
    # The controller_environment fixture permits owner construction even under xdist.
    with tempfile.TemporaryDirectory() as d:
        yield OwnerHostStateStore(d)


class TestUpdatePrimitive:
    """The private _update primitive that the named operations are built on."""

    def test_update_creates_and_persists(self, store):
        store._update(lambda s: HostState([_record("h1", Platforms.TRN2)]))
        assert store.read().hosts[0].resolved.ssh_host == "h1"

    def test_update_none_skips_write(self, store):
        store._update(lambda s: HostState([_record("h1", Platforms.TRN2)]))
        store._update(lambda s: None)  # no-op
        assert {h.resolved.ssh_host for h in store.read().hosts} == {"h1"}

    def test_write_is_atomic_no_temp_left(self, store):
        store.initialize(_resolved("h1"))
        d = os.path.dirname(store.state_path)
        assert [f for f in os.listdir(d) if f.endswith(".tmp")] == []


class TestHostStateStore:
    def test_unavailable_count(self, store):
        assert store.unavailable_count() == 0  # no file yet
        store._update(
            lambda s: HostState(
                [
                    _record("a", Platforms.TRN2, available=True),
                    _record("b", Platforms.TRN2, available=False),
                    _record("c", Platforms.TRN2, available=False),
                ]
            )
        )
        assert store.unavailable_count() == 2  # deduplicated by host, available=False only

    def test_unknown_host_type_row_is_dropped(self):
        # A row with an unknown/missing host_type is dropped, not admitted as a
        # typeless host the assignment filter could never match.
        assert HostRecord.from_json({"host_alias": "h1", "host_type": "bogus"}) is None

    def test_missing_host_alias_row_is_dropped(self):
        # A row missing host_alias is dropped (not a KeyError): from_json tolerates a
        # corrupt/partial file so one bad row can't crash every reader.
        assert HostRecord.from_json({"host_type": "trn2"}) is None
        assert HostRecord.from_json({"host_alias": "", "host_type": "trn2"}) is None

    def test_corrupt_row_does_not_break_whole_file_read(self):
        # A file with one malformed row (no host_alias) still reads: the bad row is
        # dropped and the good rows survive.
        state = HostState.from_json({"hosts": [{"host_type": "trn2"}, {"host_alias": "ok", "host_type": "trn2"}]})
        assert [h.resolved.ssh_host for h in state.hosts] == ["ok"]

    def test_reset_removes_file(self, store):
        store.initialize(_resolved("h1"))
        store.reset()
        assert store.read() is None

    def test_is_initialized_ignores_leftover_file(self, store):
        # A fresh store instance pointed at a directory with a leftover
        # host_state.json (e.g. a crashed prior run) must NOT read as initialized:
        # is_initialized tracks our own writes, not file existence.
        store.initialize(_resolved("h1"))  # leaves a file on disk
        fresh = HostStateStore(os.path.dirname(store.state_path))
        assert os.path.exists(fresh.state_path)  # file is there...
        assert fresh.is_initialized() is False  # ...but this instance never wrote it


class TestInitialize:
    def test_adds_missing_hosts_preserving_existing(self, store):
        store._update(lambda s: HostState([_record("h1", Platforms.TRN2, work_queue_depth=3, available=False)]))
        store.initialize(_resolved("h1", "h2"))
        by_alias = {h.resolved.ssh_host: h for h in store.read().hosts}
        # h1 untouched (depth + availability preserved — initialize only adds)
        assert by_alias["h1"].work_queue_depth == 3
        assert by_alias["h1"].available is False
        assert by_alias["h2"].work_queue_depth == 0

    def test_initialize_adds_all_hosts_regardless_of_shuffle(self, store):
        # initialize shuffles the bootstrap list, but membership must be complete and
        # exactly-once no matter the order.
        store.initialize(_resolved("h1", "h2", "h3", "h4"))
        assert sorted(h.resolved.ssh_host for h in store.read().hosts) == ["h1", "h2", "h3", "h4"]

    def test_initialize_randomizes_append_order_across_runs(self):
        # The shuffle exists so the claim tie-break (append order, when equal-size hosts sit
        # at equal load) doesn't deterministically hammer the first-listed host every run.
        # Initialize many independent stores from the SAME input and assert the persisted
        # order isn't always identical. 20 hosts over 15 stores makes a false failure
        # (all-identical by chance) ~15/20! — effectively impossible — so no seeding needed.
        aliases = [f"h{i:02d}" for i in range(20)]
        orders = set()
        with tempfile.TemporaryDirectory() as base:
            for i in range(15):
                store = HostStateStore(os.path.join(base, f"run{i}"))
                store.initialize(_resolved(*aliases))
                persisted = store.read()
                assert persisted is not None, "initialize must persist a state file"
                orders.add(tuple(h.resolved.ssh_host for h in persisted.hosts))
        assert len(orders) > 1, "initialize should randomize host append order across runs"


class TestMultiNodeDeploymentPersistence:
    """The multi-node deployment a probe discovers must survive a round trip through
    host_state.json, so every worker sees which hosts belong together."""

    def _deployed(self, alias: str, **kwargs) -> ResolvedHost:
        fields = {"logical_host_id": "us-1", "logical_host_rank": 2, "logical_host_size": 4, **kwargs}
        return ResolvedHost(ssh_host=alias, host_type=Platforms.TRN2, num_physical_cores=64, **fields)

    def test_round_trips_through_json(self):
        row = HostRecord(resolved=self._deployed("h1")).to_json()
        assert (row["logical_host_id"], row["logical_host_rank"], row["logical_host_size"]) == ("us-1", 2, 4)
        restored = HostRecord.from_json(row)
        assert restored is not None
        assert restored.resolved == self._deployed("h1")

    def test_single_node_row_omits_the_deployment_keys(self):
        # A single-node fleet's rows are byte-identical to before this field existed.
        assert set(HostRecord(resolved=_resolved("h1")[0]).to_json()) == {
            "host_alias",
            "host_type",
            "work_queue_depth",
            "available",
            "num_physical_cores",
        }

    def test_row_written_before_the_field_existed_reads_back_as_unknown(self):
        record = HostRecord.from_json({"host_alias": "h1", "host_type": "trn2", "num_physical_cores": 64})
        assert record is not None
        assert record.resolved.logical_host_id is None
        assert (record.resolved.logical_host_rank, record.resolved.logical_host_size) == (None, None)

    def test_apply_resolved_records_a_newly_discovered_deployment(self, store):
        store._update(lambda s: HostState([_record("h1", Platforms.TRN2, num_physical_cores=64)]))
        store.apply_resolved([self._deployed("h1")])
        resolved = store.read().hosts[0].resolved
        assert (resolved.logical_host_id, resolved.logical_host_rank, resolved.logical_host_size) == ("us-1", 2, 4)

    def test_apply_resolved_keeps_a_known_deployment_when_none_is_reported(self, store):
        # A cycle that resolves the host without deployment info (e.g. a legacy neuron-ls, or a
        # source that never probed it) must not erase what the store already knew.
        store._update(lambda s: HostState([HostRecord(resolved=self._deployed("h1"))]))
        store.apply_resolved(_resolved("h1"))
        resolved = store.read().hosts[0].resolved
        assert (resolved.logical_host_id, resolved.logical_host_rank) == ("us-1", 2)

    def test_apply_resolved_refreshes_a_changed_deployment(self, store):
        store._update(lambda s: HostState([HostRecord(resolved=self._deployed("h1"))]))
        store.apply_resolved([self._deployed("h1", logical_host_id="us-2", logical_host_rank=0, logical_host_size=2)])
        resolved = store.read().hosts[0].resolved
        assert (resolved.logical_host_id, resolved.logical_host_rank, resolved.logical_host_size) == ("us-2", 0, 2)


class TestApplyResolved:
    def test_resolved_host_is_available_reactivating_an_unavailable_one(self, store):
        # Pure membership: a host present in the resolved (good-to-use) set is available
        # again, even if it was previously marked unavailable; depth is preserved.
        store._update(lambda s: HostState([_record("h1", Platforms.TRN2, work_queue_depth=2, available=False)]))
        store.apply_resolved(_resolved("h1"))
        rec = store.read().hosts[0]
        assert rec.available is True
        assert rec.work_queue_depth == 2  # depth preserved across reactivation

    def test_host_absent_from_resolved_is_marked_unavailable(self, store):
        # The other half of membership: a host NOT in the resolved set goes unavailable
        # (its source dropped it — vanished or no longer reachable). Row + depth kept.
        store._update(lambda s: HostState([_record("h1", Platforms.TRN2, work_queue_depth=2, available=True)]))
        store.apply_resolved(_resolved("h2"))  # h1 absent
        by_alias = {h.resolved.ssh_host: h for h in store.read().hosts}
        assert by_alias["h1"].available is False
        assert by_alias["h1"].work_queue_depth == 2  # row + depth kept for posterity
        assert by_alias["h2"].available is True  # newly added

    def test_three_sets_together(self, store):
        # keep: in both (available, depth kept); gone: only in state (unavailable);
        # new: only in resolved (added fresh).
        store._update(
            lambda s: HostState(
                [
                    _record("keep", Platforms.TRN2, work_queue_depth=3),
                    _record("gone", Platforms.TRN2, work_queue_depth=5),
                ]
            )
        )
        store.apply_resolved(_resolved("keep", "new"))
        by_alias = {h.resolved.ssh_host: h for h in store.read().hosts}
        assert (by_alias["keep"].available, by_alias["keep"].work_queue_depth) == (True, 3)
        assert by_alias["gone"].available is False
        assert (by_alias["new"].available, by_alias["new"].work_queue_depth) == (True, 0)

    def test_empty_resolution_marks_all_unavailable(self, store):
        # A successful empty resolution is a real state (pool scaled to zero), not a
        # failure: applied normally, it marks every existing host unavailable.
        store.initialize(_resolved("h1"))
        store.apply_resolved([])
        assert store.read().hosts[0].available is False

    def test_empty_resolution_then_claim_returns_none_end_to_end(self, store):
        # End-to-end: a scale-to-zero resolution marks every host unavailable, and a
        # subsequent claim then yields None — exactly what HostManager turns into a
        # "no hosts available" / recoverable wait.
        store.initialize(_resolved("h1", "h2"))
        store.apply_resolved([])
        assert store.claim_least_busy(Platforms.TRN2, 1) is None


class TestClaimReleaseMarkUnavailable:
    def test_claim_returns_least_busy_and_increments(self, store):
        store._update(
            lambda s: HostState(
                [
                    _record("busy", Platforms.TRN2, work_queue_depth=5, num_physical_cores=64),
                    _record("idle", Platforms.TRN2, work_queue_depth=0, num_physical_cores=64),
                ]
            )
        )
        claimed = store.claim_least_busy(Platforms.TRN2, 1)
        assert claimed.host_id == "idle"
        # Returned record reflects the post-increment depth, and is persisted.
        assert claimed.primary.work_queue_depth == 1
        assert _depths(store) == {"busy": 5, "idle": 1}

    def test_claim_skips_unavailable_and_wrong_platform(self, store):
        store._update(
            lambda s: HostState(
                [
                    _record("down", Platforms.TRN2, available=False, num_physical_cores=64),
                    _record("other", Platforms.TRN3_A0, num_physical_cores=64),
                ]
            )
        )
        assert store.claim_least_busy(Platforms.TRN2, 1) is None

    def test_release_decrements_to_zero(self, store):
        store._update(lambda s: HostState([_record("h1", Platforms.TRN2, work_queue_depth=1, num_physical_cores=64)]))
        store.release("h1", 1)
        assert store.read().hosts[0].work_queue_depth == 0

    def test_release_at_zero_floors_and_warns(self, store, caplog):
        # Unbalanced release (depth already 0): stays floored at 0 and logs a bug signal.
        store._update(lambda s: HostState([_record("h1", Platforms.TRN2, work_queue_depth=0, num_physical_cores=64)]))
        with caplog.at_level("WARNING"):
            store.release("h1", 1)
        assert store.read().hosts[0].work_queue_depth == 0
        assert any("unbalanced release" in r.message for r in caplog.records)

    def test_mark_unavailable_keeps_row(self, store):
        store.initialize(_resolved("h1"))
        store.mark_unavailable("h1")
        rec = store.read().hosts[0]
        assert rec.resolved.ssh_host == "h1"
        assert rec.available is False

    def test_marked_unavailable_host_is_reactivated_when_resolved_again(self, store):
        # The store contract: a worker marks a host unavailable, and a later resolution
        # that includes it (a source confirmed it reachable again) brings it back and
        # makes it claimable.
        store.initialize(_resolved("h1"))
        store.mark_unavailable("h1")
        assert store.claim_least_busy(Platforms.TRN2, 1) is None  # out of the pool while down
        store.apply_resolved(_resolved("h1"))  # good-to-use again
        rec = store.read().hosts[0]
        assert rec.available is True
        assert store.claim_least_busy(Platforms.TRN2, 1).host_id == "h1"

    def test_host_aliases_for_platform(self, store):
        store._update(
            lambda s: HostState([_record("a", Platforms.TRN2, available=False), _record("b", Platforms.TRN3_A0)])
        )
        # Includes unavailable hosts; filtered by platform.
        assert store.host_aliases_for(Platforms.TRN2) == ["a"]
        assert store.host_aliases_for(Platforms.TRN3_A0) == ["b"]


class TestHostUnitInterface:
    """HostRecord and HostGroup expose the same HostUnit interface over their members."""

    def test_snapshot_is_detached(self):
        group = HostGroup(members=(_member("m0", "L", 0), _member("m1", "L", 1)))
        copy = group.snapshot()
        group.members[0].work_queue_depth = 4
        assert [rec.work_queue_depth for rec in copy] == [0, 0]


class TestMultiNodeClaim:
    """Above MULTINODE_CORE_THRESHOLD a multi-node deployment is claimed as one HostGroup and
    every member is charged the full request; at or below it, members are claimed individually."""

    # Above the threshold and divisible by 2, 3 and 4, so it splits evenly over every group here.
    ABOVE = 132

    def test_at_threshold_members_are_claimed_individually(self, store):
        store._update(lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1, work_queue_depth=10)]))
        claimed = store.claim_least_busy(Platforms.TRN2, MULTINODE_CORE_THRESHOLD)
        assert isinstance(claimed, HostRecord)
        assert claimed.host_id == "m0"
        assert _depths(store) == {"m0": MULTINODE_CORE_THRESHOLD, "m1": 10}

    def test_above_threshold_claims_the_group_and_charges_every_member(self, store):
        store._update(
            lambda s: HostState(
                [_record("single", num_physical_cores=128), _member("m1", "L", 1), _member("m0", "L", 0)]
            )
        )
        claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE)
        assert isinstance(claimed, HostGroup)
        assert claimed.host_id == "m0"  # ordered by rank, not by row order
        assert [rec.work_queue_depth for rec in claimed] == [self.ABOVE, self.ABOVE]
        assert _depths(store) == {"single": 0, "m0": self.ABOVE, "m1": self.ABOVE}

    @staticmethod
    def _mixed_pool() -> HostState:
        return HostState(
            [
                _member("a0", "A", 0),
                _member("a1", "A", 1),
                _member("b0", "B", 0),
                _member("b1", "B", 1, available=False),
                _record("big", num_physical_cores=192),
                _record("small", num_physical_cores=64),
            ]
        )

    def test_eligible_host_aliases_lists_one_id_per_eligible_unit_above_threshold(self, store):
        store._update(lambda s: self._mixed_pool())
        assert store.eligible_host_aliases(Platforms.TRN2, self.ABOVE) == {"a0", "big"}

    def test_eligible_host_aliases_lists_members_individually_at_threshold(self, store):
        store._update(lambda s: self._mixed_pool())
        assert store.eligible_host_aliases(Platforms.TRN2, MULTINODE_CORE_THRESHOLD) == {"a0", "a1", "b0", "big"}

    def test_release_via_non_primary_member_unreserves_every_member(self, store, caplog):
        store._update(lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1)]))
        store.claim_least_busy(Platforms.TRN2, self.ABOVE)
        with caplog.at_level("WARNING"):
            store.release("m1", self.ABOVE)
        assert _depths(store) == {"m0": 0, "m1": 0}
        assert not any("unbalanced release" in r.message for r in caplog.records)

    def test_above_threshold_members_are_not_offered_individually(self, store):
        # Offered alone, m0 would score 132/256=0.516 and beat the group mean (0.516 + 332/256) / 2 = 0.906.
        store._update(
            lambda s: HostState(
                [
                    _member("m0", "L", 0, num_physical_cores=256),
                    _member("m1", "L", 1, num_physical_cores=256, work_queue_depth=200),
                ]
            )
        )
        claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE)
        assert isinstance(claimed, HostGroup)
        assert claimed.member_ids == ("m0", "m1")
        assert _depths(store) == {"m0": self.ABOVE, "m1": 200 + self.ABOVE}

    def test_large_single_host_competes_with_groups_and_releases_alone(self, store):
        # big scores 132/192=0.688, the idle 2x128 group 132/128=1.031.
        store._update(
            lambda s: HostState([_record("big", num_physical_cores=192), _member("m0", "L", 0), _member("m1", "L", 1)])
        )
        claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE)
        assert isinstance(claimed, HostRecord)
        assert claimed.host_id == "big"
        assert claimed.member_ids == ("big",)
        assert _depths(store) == {"big": self.ABOVE, "m0": 0, "m1": 0}
        store.release("big", self.ABOVE)
        assert _depths(store) == {"big": 0, "m0": 0, "m1": 0}

    @pytest.mark.parametrize("with_member_ids", [True, False], ids=["member_ids", "rederived"])
    def test_individual_and_group_claims_on_one_deployment_balance(self, store, caplog, with_member_ids):
        store._update(lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1)]))
        single = store.claim_least_busy(Platforms.TRN2, 64)
        group = store.claim_least_busy(Platforms.TRN2, self.ABOVE)
        assert isinstance(single, HostRecord) and single.host_id == "m0"
        assert isinstance(group, HostGroup) and group.host_id == "m0"
        assert _depths(store) == {"m0": 64 + self.ABOVE, "m1": self.ABOVE}
        with caplog.at_level("WARNING"):
            store.release("m0", 64)
            store.release(group.host_id, self.ABOVE, member_ids=group.member_ids if with_member_ids else None)
        assert _depths(store) == {"m0": 0, "m1": 0}
        assert not any("unbalanced release" in r.message for r in caplog.records)

    def test_group_members_with_unknown_rank_sort_last(self, store):
        store._update(lambda s: HostState([_member("m_none", "L", None), _member("m0", "L", 0)]))
        claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE)
        assert claimed.host_id == "m0"
        assert claimed.member_ids == ("m0", "m_none")

    def test_release_with_member_ids_survives_deployment_id_change(self, store):
        store._update(lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1)]))
        claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE)

        def _move_m1(state: HostState) -> HostState:
            for rec in state.hosts:
                if rec.resolved.ssh_host == "m1":
                    rec.resolved = replace(rec.resolved, logical_host_id="OTHER")
            return state

        store._update(_move_m1)
        store.release(claimed.host_id, self.ABOVE, member_ids=claimed.member_ids)
        assert _depths(store) == {"m0": 0, "m1": 0}

    def test_release_with_host_id_outside_member_ids_logs_and_releases_members(self, store, caplog):
        store._update(
            lambda s: HostState([_member("m0", "L", 0, work_queue_depth=5), _member("m1", "L", 1, work_queue_depth=5)])
        )
        with caplog.at_level("ERROR"):
            store.release("m0", 5, member_ids=("m1",))
        assert _depths(store) == {"m0": 5, "m1": 0}
        assert any(r.levelname == "ERROR" and "m0" in r.getMessage() for r in caplog.records)

    def test_rederive_release_of_unknown_host_warns(self, store, caplog):
        store._update(lambda s: HostState([_record("h0", num_physical_cores=64, work_queue_depth=3)]))
        with caplog.at_level("WARNING"):
            store.release("nope", 1)
        assert _depths(store) == {"h0": 3}
        assert any(r.levelname == "WARNING" and "nope" in r.getMessage() for r in caplog.records)

    def test_release_with_member_ids_skips_and_warns_on_missing_rows(self, store, caplog):
        store._update(lambda s: HostState([_member("m0", "L", 0, work_queue_depth=5)]))
        with caplog.at_level("WARNING"):
            store.release("m0", 5, member_ids=("m0", "gone"))
        assert _depths(store) == {"m0": 0}
        assert any(r.levelname == "WARNING" and "gone" in r.getMessage() for r in caplog.records)

    def test_release_with_no_matching_member_rows_writes_nothing(self, store, caplog):
        store._update(
            lambda s: HostState([_member("m0", "L", 0, work_queue_depth=5), _record("h0", work_queue_depth=3)])
        )
        before = os.stat(store.state_path)
        with open(store.state_path, "rb") as f:
            content_before = f.read()
        with caplog.at_level("WARNING"):
            store.release("x", 1, member_ids=("x", "y"))
        after = os.stat(store.state_path)
        with open(store.state_path, "rb") as f:
            assert f.read() == content_before
        assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)
        assert _depths(store) == {"m0": 5, "h0": 3}
        assert any(
            r.levelname == "WARNING" and "'x'" in r.getMessage() and "'y'" in r.getMessage() for r in caplog.records
        )

    @pytest.mark.parametrize(
        ("depths_by_group", "winner"),
        [
            # A: 232/128=1.812, 132/128=1.031 -> mean 1.422; B: 192/128=1.5 each -> mean 1.5.
            pytest.param({"A": (100, 0), "B": (60, 60)}, "A", id="max_or_first_member_would_pick_B"),
            # A: 1.812, 1.031 -> mean 1.422; B: 172/128=1.344 each -> mean 1.344.
            pytest.param({"A": (100, 0), "B": (40, 40)}, "B", id="min_would_pick_A"),
            # C: 1.031 x3 -> mean 1.031 (sum 3.094); D: 142/128=1.109 x2 -> mean 1.109 (sum 2.219).
            pytest.param({"C": (0, 0, 0), "D": (10, 10)}, "C", id="sum_would_pick_D"),
        ],
    )
    def test_group_ranked_by_mean_member_load(self, store, depths_by_group, winner):
        hosts = [
            _member(f"{group.lower()}{rank}", group, rank, size=len(depths), work_queue_depth=depth)
            for group, depths in reversed(depths_by_group.items())
            for rank, depth in enumerate(depths)
        ]
        store._update(lambda s: HostState(hosts))
        claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE)
        assert claimed.host_id == f"{winner.lower()}0"
        for rank, depth in enumerate(depths_by_group[winner]):
            assert _depths(store)[f"{winner.lower()}{rank}"] == depth + self.ABOVE

    def test_group_with_any_excluded_member_is_deprioritized(self, store):
        store._update(
            lambda s: HostState(
                [
                    _member("a0", "A", 0),
                    _member("a1", "A", 1),
                    _member("b0", "B", 0, work_queue_depth=50),
                    _member("b1", "B", 1, work_queue_depth=50),
                ]
            )
        )
        assert store.claim_least_busy(Platforms.TRN2, self.ABOVE, exclude={"a1"}).host_id == "b0"

    def test_every_group_excluded_falls_back_to_full_eligible_set(self, store):
        store._update(
            lambda s: HostState(
                [
                    _member("b0", "B", 0, work_queue_depth=50),
                    _member("b1", "B", 1, work_queue_depth=50),
                    _member("a0", "A", 0),
                    _member("a1", "A", 1),
                ]
            )
        )
        claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE, exclude={"a1", "b0"})
        assert claimed.host_id == "a0"
        assert _depths(store) == {"a0": self.ABOVE, "a1": self.ABOVE, "b0": 50, "b1": 50}

    def test_group_with_unavailable_member_is_not_claimable(self, store):
        store._update(lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1, available=False)]))
        assert store.claim_least_busy(Platforms.TRN2, self.ABOVE) is None

    # 258 is the next request that splits over 2 hosts; its 129-core share exceeds each 128-core member.
    @pytest.mark.parametrize(
        ("needed", "claimable"), [(256, True), (258, False)], ids=["exact_total", "next_split_over"]
    )
    def test_group_total_core_boundary(self, store, needed, claimable):
        store._update(lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1)]))
        claimed = store.claim_least_busy(Platforms.TRN2, needed)
        if claimable:
            assert claimed is not None and claimed.host_id == "m0"
            assert _depths(store) == {"m0": needed, "m1": needed}
        else:
            assert claimed is None
            assert _depths(store) == {"m0": 0, "m1": 0}

    def test_explain_unclaimable_reports_group_capacity(self, store):
        store._update(lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1)]))
        assert "1 too small (largest available 256 cores)" in store.explain_unclaimable(Platforms.TRN2, 300)

    @staticmethod
    def _mixed_platform_group(*, available: bool = True) -> HostState:
        return HostState(
            [
                _member("t2", "M", 0),
                _member("t3", "M", 1, host_type=Platforms.TRN3_A0, available=available),
            ]
        )

    def test_mixed_platform_group_does_not_block_other_platforms(self, store):
        store._update(lambda s: self._mixed_platform_group())
        assert store.claim_least_busy(Platforms.TRN1, self.ABOVE) is None

    def test_mixed_platform_group_raises_through_claim(self, store):
        store._update(lambda s: self._mixed_platform_group())
        with pytest.raises(ValueError, match="mixes platforms"):
            store.claim_least_busy(Platforms.TRN2, self.ABOVE)
        assert store.claim_least_busy(Platforms.TRN2, 64).host_id == "t2"

    def test_mixed_platform_group_raises_even_with_unavailable_member(self, store):
        store._update(lambda s: self._mixed_platform_group(available=False))
        with pytest.raises(ValueError, match="mixes platforms"):
            store.claim_least_busy(Platforms.TRN2, self.ABOVE)

    def test_explain_unclaimable_tolerates_unavailable_mixed_group(self, store):
        store._update(lambda s: self._mixed_platform_group(available=False))
        assert "1 unavailable" in store.explain_unclaimable(Platforms.TRN2, 300)

    def test_group_with_zero_core_member_is_not_claimable(self, store):
        store._update(
            lambda s: HostState(
                [_member("z0", "Z", 0, num_physical_cores=256), _member("z1", "Z", 1, num_physical_cores=0)]
            )
        )
        assert store.claim_least_busy(Platforms.TRN2, self.ABOVE) is None
        store._update(lambda s: replace(s, hosts=[*s.hosts, _member("h0", "H", 0), _member("h1", "H", 1)]))
        assert store.claim_least_busy(Platforms.TRN2, self.ABOVE).host_id == "h0"

    def test_explain_unclaimable_counts_zero_core_member_group_reports_unknown_capacity(self, store):
        store._update(
            lambda s: HostState(
                [_member("z0", "Z", 0, num_physical_cores=256), _member("z1", "Z", 1, num_physical_cores=0)]
            )
        )
        message = store.explain_unclaimable(Platforms.TRN2, self.ABOVE)
        assert "1 with unknown capacity" in message
        assert "too small" not in message

    def test_reset_clears_incomplete_warning_memory(self, store, caplog):
        def _pool(s: HostState | None = None) -> HostState:
            return HostState([_member("m0", "L", 0, size=3), _member("m1", "L", 1, size=3)])

        def _warnings() -> int:
            return sum(1 for r in caplog.records if r.levelname == "WARNING" and "members in the pool" in r.message)

        with caplog.at_level("WARNING"):
            store._update(_pool)
            assert store.claim_least_busy(Platforms.TRN2, self.ABOVE) is not None
            assert _warnings() == 1
            store.reset()
            store._update(_pool)
            assert store.claim_least_busy(Platforms.TRN2, self.ABOVE) is not None
            assert _warnings() == 2

    def test_incomplete_group_warns_once_per_membership(self, store, caplog):
        def _warnings() -> int:
            return sum(1 for r in caplog.records if r.levelname == "WARNING" and "members in the pool" in r.message)

        store._update(lambda s: HostState([_member("m0", "L", 0, size=4), _member("m1", "L", 1, size=4)]))
        with caplog.at_level("WARNING"):
            for _ in range(2):
                claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE)
                store.release(claimed.host_id, self.ABOVE, member_ids=claimed.member_ids)
            assert claimed.host_id == "m0"
            assert any(r.levelname == "WARNING" and "has 2 of 4 members" in r.message for r in caplog.records)
            assert _warnings() == 1
            store._update(lambda s: replace(s, hosts=[*s.hosts, _member("m2", "L", 2, size=4)]))
            for _ in range(2):
                claimed = store.claim_least_busy(Platforms.TRN2, self.ABOVE)
                store.release(claimed.host_id, self.ABOVE, member_ids=claimed.member_ids)
            assert _warnings() == 2


class TestMultiNodeRankSplit:
    """A deployment serves a request only if its logical ranks (physical cores / lnc_config) split
    evenly across the pooled members and every member fits its share; the full physical request is
    still charged to every member."""

    @staticmethod
    def _state_bytes(store) -> bytes:
        with open(store.state_path, "rb") as f:
            return f.read()

    def test_three_of_four_members_cannot_split_128_ranks(self, store):
        # 128 logical ranks at LNC2 = 256 physical; 128 ranks do not divide over 3 hosts even though
        # the 3 x 128 = 384 pooled cores exceed the request.
        store._update(lambda s: HostState([_member(f"m{rank}", "L", rank, size=4) for rank in range(3)]))
        before = self._state_bytes(store)
        assert store.claim_least_busy(Platforms.TRN2, 256, lnc_config=2) is None
        assert self._state_bytes(store) == before
        assert _depths(store) == {"m0": 0, "m1": 0, "m2": 0}
        assert store.eligible_host_aliases(Platforms.TRN2, 256, lnc_config=2) == set()
        message = store.explain_unclaimable(Platforms.TRN2, 256, lnc_config=2)
        assert "1 whose 3 hosts cannot evenly split 128 logical ranks" in message
        assert "reason unknown" not in message

    def test_two_of_four_members_serve_an_even_split(self, store, caplog):
        store._update(lambda s: HostState([_member(f"m{rank}", "L", rank, size=4) for rank in range(2)]))
        with caplog.at_level("WARNING"):
            claimed = store.claim_least_busy(Platforms.TRN2, 256, lnc_config=2)
        assert isinstance(claimed, HostGroup)
        assert claimed.member_ids == ("m0", "m1")
        assert _depths(store) == {"m0": 256, "m1": 256}  # full request charged to every member
        assert any("has 2 of 4 members" in r.message for r in caplog.records)

    def test_full_homogeneous_deployment_is_claimed(self, store):
        store._update(lambda s: HostState([_member(f"m{rank}", "L", rank, size=4) for rank in range(4)]))
        assert store.eligible_host_aliases(Platforms.TRN2, 512, lnc_config=2) == {"m0"}
        claimed = store.claim_least_busy(Platforms.TRN2, 512, lnc_config=2)
        assert isinstance(claimed, HostGroup)
        assert claimed.member_ids == ("m0", "m1", "m2", "m3")
        assert _depths(store) == {"m0": 512, "m1": 512, "m2": 512, "m3": 512}

    def test_even_physical_split_with_uneven_logical_ranks_is_rejected(self, store):
        # 258 physical / LNC2 = 129 ranks: the physical count divides over 2 hosts, the ranks do not.
        store._update(
            lambda s: HostState(
                [_member("m0", "L", 0, num_physical_cores=256), _member("m1", "L", 1, num_physical_cores=256)]
            )
        )
        assert store.claim_least_busy(Platforms.TRN2, 258, lnc_config=2) is None
        assert store.eligible_host_aliases(Platforms.TRN2, 258, lnc_config=2) == set()
        assert _depths(store) == {"m0": 0, "m1": 0}
        assert "cannot evenly split 129 logical ranks" in store.explain_unclaimable(Platforms.TRN2, 258, lnc_config=2)
        # The same physical count at LNC1 is 258 ranks, which split 129 per host.
        assert store.claim_least_busy(Platforms.TRN2, 258, lnc_config=1).host_id == "m0"

    def test_odd_request_on_two_members_is_rejected_at_lnc1(self, store):
        store._update(lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1)]))
        assert store.claim_least_busy(Platforms.TRN2, MULTINODE_CORE_THRESHOLD + 1) is None
        assert _depths(store) == {"m0": 0, "m1": 0}

    def test_member_below_its_share_is_rejected_despite_sufficient_aggregate(self, store):
        # 256 + 64 = 320 pooled cores cover 256, but each host must host 128.
        store._update(
            lambda s: HostState(
                [_member("m0", "L", 0, num_physical_cores=256), _member("m1", "L", 1, num_physical_cores=64)]
            )
        )
        before = self._state_bytes(store)
        assert store.claim_least_busy(Platforms.TRN2, 256) is None
        assert self._state_bytes(store) == before
        assert store.eligible_host_aliases(Platforms.TRN2, 256) == set()
        message = store.explain_unclaimable(Platforms.TRN2, 256)
        assert "1 with a host below its 128-core share" in message
        assert "too small (largest" not in message
        assert "reason unknown" not in message

    @pytest.mark.parametrize(
        ("needed", "lnc_config", "match"),
        [
            pytest.param(0, 1, "positive", id="zero_request"),
            pytest.param(-2, 1, "positive", id="negative_request"),
            pytest.param(4, 0, "lnc_config", id="zero_lnc"),
            pytest.param(257, 2, "divisible", id="physical_not_divisible_by_lnc"),
        ],
    )
    def test_invalid_request_raises_without_mutation(self, store, needed, lnc_config, match):
        # A single host beside the deployment: validation precedes placement for both.
        store._update(
            lambda s: HostState([_member("m0", "L", 0), _member("m1", "L", 1), _record("h0", num_physical_cores=64)])
        )
        before = self._state_bytes(store)
        for call in (store.claim_least_busy, store.eligible_host_aliases, store.explain_unclaimable):
            with pytest.raises(ValueError, match=match):
                call(Platforms.TRN2, needed, lnc_config=lnc_config)
        assert self._state_bytes(store) == before

    def test_single_host_request_divisible_by_lnc_is_claimed(self, store):
        store._update(lambda s: HostState([_record("h0", num_physical_cores=64)]))
        assert store.claim_least_busy(Platforms.TRN2, 4, lnc_config=2).host_id == "h0"


class TestPoisoned:
    """The per-platform poisoned set written by the background re-resolver and read by waiting workers."""

    def test_absent_file_is_not_poisoned(self, store):
        assert store.is_poisoned(Platforms.TRN2) is False  # no file yet

    def test_set_and_read_poison(self, store):
        store.initialize(_resolved("h1"))
        assert store.is_poisoned(Platforms.TRN2) is False  # default
        store.set_poisoned(Platforms.TRN2, True)
        assert store.is_poisoned(Platforms.TRN2) is True
        store.set_poisoned(Platforms.TRN2, False)
        assert store.is_poisoned(Platforms.TRN2) is False

    def test_poison_is_per_platform(self, store):
        # Poisoning one platform must not poison another.
        store.set_poisoned(Platforms.TRN2, True)
        assert store.is_poisoned(Platforms.TRN2) is True
        assert store.is_poisoned(Platforms.TRN3_A0) is False

    def test_poison_round_trips_through_disk(self, store):
        store.initialize(_resolved("h1"))
        store.set_poisoned(Platforms.TRN2, True)
        # A fresh store over the same dir reads the persisted set.
        reopened = HostStateStore(os.path.dirname(store.state_path))
        assert reopened.is_poisoned(Platforms.TRN2) is True

    def test_poison_is_preserved_across_host_mutations(self, store):
        # Poison is fleet metadata: per-host mutations (apply_resolved/mark_unavailable)
        # must not silently clear it — only set_poisoned(..., False) does.
        store.initialize(_resolved("h1"))
        store.set_poisoned(Platforms.TRN2, True)
        store.mark_unavailable("h1")
        store.apply_resolved(_resolved("h1", "h2"))
        assert store.is_poisoned(Platforms.TRN2) is True

    def test_no_op_set_skips_write(self, store):
        store.initialize(_resolved("h1"))
        before = os.path.getmtime(store.state_path)
        store.set_poisoned(Platforms.TRN2, False)  # not poisoned -> no write
        assert os.path.getmtime(store.state_path) == before

    def test_claim_raises_when_poisoned_and_no_host(self, store):
        # A claim with nothing to give AND a poisoned platform fails fast (unretriable),
        # rather than returning None — the caller must not keep retrying a dead platform.
        store.initialize(_resolved("h1"))
        store.mark_unavailable("h1")
        store.set_poisoned(Platforms.TRN2, True)
        with pytest.raises(FleetEmptyError, match="perpetually empty"):
            store.claim_least_busy(Platforms.TRN2, 1)

    def test_claim_returns_host_even_when_poisoned_if_one_is_available(self, store):
        # Recovery race: hosts are back but the poison flag isn't cleared yet (the
        # re-resolver sets hosts available and clears poison as two writes). A claimable
        # host wins over raising, so the worker uses the recovered host.
        store.initialize(_resolved("h1"))
        store.set_poisoned(Platforms.TRN2, True)
        assert store.claim_least_busy(Platforms.TRN2, 1).host_id == "h1"

    def test_claim_unaffected_by_other_platforms_poison(self, store):
        # A TRN2 claim must not raise just because TRN3_A0 is poisoned.
        store.initialize(_resolved("h1"))
        store.mark_unavailable("h1")
        store.set_poisoned(Platforms.TRN3_A0, True)
        assert store.claim_least_busy(Platforms.TRN2, 1) is None  # waits, does not raise

    def test_claim_returns_none_when_empty_but_not_poisoned(self, store):
        # Not poisoned + nothing claimable -> None (the caller waits), never raises.
        store.initialize(_resolved("h1"))
        store.mark_unavailable("h1")
        assert store.claim_least_busy(Platforms.TRN2, 1) is None


class TestUnderCapacity:
    """The per-platform under-capacity set, applied by ``apply_resolved`` in the same locked
    write as membership (so a host is never made available with a stale gate). An
    under-capacity platform has its available hosts withheld from claims, so work isn't piled
    onto the few that came up while the fleet is still scaling."""

    def test_absent_file_is_not_under_capacity(self, store):
        assert store.is_under_capacity(Platforms.TRN2) is False  # no file yet

    def test_apply_resolved_sets_and_clears_the_verdict(self, store):
        store.apply_resolved(_resolved("h1"), under_capacity={Platforms.TRN2})
        assert store.is_under_capacity(Platforms.TRN2) is True
        # A later cycle with the platform healthy clears it (the verdict is the whole set each time).
        store.apply_resolved(_resolved("h1"), under_capacity=set())
        assert store.is_under_capacity(Platforms.TRN2) is False

    def test_round_trips_through_disk(self, store):
        store.apply_resolved(_resolved("h1"), under_capacity={Platforms.TRN2})
        reopened = HostStateStore(os.path.dirname(store.state_path))
        assert reopened.is_under_capacity(Platforms.TRN2) is True

    def test_membership_only_apply_preserves_the_verdict(self, store):
        # apply_resolved without an under_capacity arg reconciles membership only, leaving the
        # existing verdict untouched — the standalone-membership use must not clear the gate.
        store.apply_resolved(_resolved("h1"), under_capacity={Platforms.TRN2})
        store.apply_resolved(_resolved("h1"))  # membership-only
        assert store.is_under_capacity(Platforms.TRN2) is True

    def test_claim_withheld_even_with_an_available_host(self, store):
        # The whole point: a healthy, claimable host is still withheld while its platform is
        # under-capacity — so the lone host that came up isn't bombarded.
        store.apply_resolved(_resolved("h1"), under_capacity={Platforms.TRN2})
        assert store.claim_least_busy(Platforms.TRN2, 1) is None

    def test_eligible_aliases_empty_when_under_capacity(self, store):
        # The startup wait reads eligibility through this; under-capacity reads as "no eligible
        # host" so a non-recoverable run keeps waiting instead of releasing workers.
        store.apply_resolved(_resolved("h1"), under_capacity={Platforms.TRN2})
        assert store.eligible_host_aliases(Platforms.TRN2, 1) == set()

    def test_clearing_under_capacity_restores_claims(self, store):
        # Once enough hosts come up and the refresher clears the flag, the withheld hosts
        # become claimable again.
        store.apply_resolved(_resolved("h1"), under_capacity={Platforms.TRN2})
        assert store.claim_least_busy(Platforms.TRN2, 1) is None
        store.apply_resolved(_resolved("h1"), under_capacity=set())
        assert store.claim_least_busy(Platforms.TRN2, 1).host_id == "h1"

    def test_poison_still_raises_when_under_capacity_and_poisoned(self, store):
        # Under-capacity withholds (returns None), but a poisoned platform with nothing
        # claimable still fails fast — poison's unretriable verdict wins over a plain wait.
        store.apply_resolved(_resolved("h1"), under_capacity={Platforms.TRN2})
        store.set_poisoned(Platforms.TRN2, True)
        with pytest.raises(FleetEmptyError, match="perpetually empty"):
            store.claim_least_busy(Platforms.TRN2, 1)

    def test_under_capacity_preserved_across_host_mutations(self, store):
        # Under-capacity is fleet metadata: a per-host mutation (mark_unavailable) must not
        # silently clear it — only another apply_resolved verdict does.
        store.apply_resolved(_resolved("h1"), under_capacity={Platforms.TRN2})
        store.mark_unavailable("h1")
        assert store.is_under_capacity(Platforms.TRN2) is True


class TestProvisionableCeiling:
    """The per-platform capacity ceiling (largest single host the fleet could ever provision),
    written by ``apply_resolved`` and enforced by ``claim_least_busy``: a request that exceeds
    it fails fast with RequestTooLargeError instead of returning None (waiting) for a host that
    can never appear. A platform absent from the map is unbounded — the guard is off."""

    def test_absent_file_and_default_is_unbounded(self, store):
        assert store.provisionable_ceiling_for(Platforms.TRN2) is None  # no file yet
        store.initialize(_resolved("h1"))
        assert store.provisionable_ceiling_for(Platforms.TRN2) is None  # default: no ceiling set

    def test_apply_resolved_sets_and_round_trips_through_disk(self, store):
        store.apply_resolved(_resolved("h1"), provisionable_ceiling={Platforms.TRN2: 8})
        assert store.provisionable_ceiling_for(Platforms.TRN2) == 8
        reopened = HostStateStore(os.path.dirname(store.state_path))
        assert reopened.provisionable_ceiling_for(Platforms.TRN2) == 8

    def test_membership_only_apply_preserves_the_ceiling(self, store):
        # apply_resolved without a ceiling arg reconciles membership only, leaving the existing
        # ceiling untouched (symmetric with the under_capacity contract).
        store.apply_resolved(_resolved("h1"), provisionable_ceiling={Platforms.TRN2: 8})
        store.apply_resolved(_resolved("h1"))  # membership-only
        assert store.provisionable_ceiling_for(Platforms.TRN2) == 8

    def test_ceiling_is_rewritten_whole_each_verdict(self, store):
        # Passing a ceiling replaces the whole map (a reshape is reflected), like under_capacity.
        store.apply_resolved(_resolved("h1"), provisionable_ceiling={Platforms.TRN2: 8})
        store.apply_resolved(_resolved("h1"), provisionable_ceiling={Platforms.TRN2: 128})
        assert store.provisionable_ceiling_for(Platforms.TRN2) == 128

    def test_claim_over_ceiling_fails_fast(self, store):
        # A request larger than the ceiling can never be served -> RequestTooLargeError, even
        # though a host IS available (it's just too small). This is the fail-fast the bug needed.
        store.apply_resolved(_resolved("h1", num_physical_cores=8), provisionable_ceiling={Platforms.TRN2: 8})
        with pytest.raises(RequestTooLargeError, match="Request needs 32 physical cores"):
            store.claim_least_busy(Platforms.TRN2, 32)

    def test_claim_at_or_under_ceiling_is_unaffected(self, store):
        # A request that fits the ceiling behaves normally: exactly-at-ceiling claims the host.
        store.apply_resolved(_resolved("h1", num_physical_cores=8), provisionable_ceiling={Platforms.TRN2: 8})
        assert store.claim_least_busy(Platforms.TRN2, 8).host_id == "h1"

    def test_no_ceiling_means_no_fast_fail(self, store):
        # Unbounded platform (no ceiling recorded): a too-big request falls through to the normal
        # wait (returns None), never RequestTooLargeError — the guard fires only on positive
        # knowledge. This is the backward-compatible default for stores that never set a ceiling.
        store.initialize(_resolved("h1", num_physical_cores=8))
        assert store.claim_least_busy(Platforms.TRN2, 32) is None

    def test_over_ceiling_wins_over_poison(self, store):
        # An impossible-size request should report the SIZING cause, not a generic poison error,
        # even when the platform is also poisoned — the ceiling check precedes the poison check.
        store.apply_resolved(_resolved("h1", num_physical_cores=8), provisionable_ceiling={Platforms.TRN2: 8})
        store.mark_unavailable("h1")
        store.set_poisoned(Platforms.TRN2, True)
        with pytest.raises(RequestTooLargeError, match="largest host the fleet can provision"):
            store.claim_least_busy(Platforms.TRN2, 32)

    @staticmethod
    def _add_busy_deployment(store) -> None:
        # Added after apply_resolved (which would mark rows absent from `resolved` unavailable);
        # members fit the ceiling and one is unavailable, so nothing can serve the request.
        members = [
            _member("m0", "L", 0, num_physical_cores=64),
            _member("m1", "L", 1, num_physical_cores=64, available=False),
        ]
        store._update(lambda s: replace(s, hosts=[*s.hosts, *members]))

    def test_ceiling_skipped_above_multinode_threshold(self, store):
        # Known limitation: the ceiling bounds a single host, so a request a multi-node deployment
        # on the platform may serve never fails fast on size.
        store.apply_resolved(_resolved("h1", num_physical_cores=128), provisionable_ceiling={Platforms.TRN2: 128})
        self._add_busy_deployment(store)
        assert store.claim_least_busy(Platforms.TRN2, MULTINODE_CORE_THRESHOLD + 1) is None

    def test_ceiling_applies_when_only_another_platform_has_a_deployment(self, store):
        store.apply_resolved(_resolved("h1", num_physical_cores=128), provisionable_ceiling={Platforms.TRN2: 128})
        members = [
            _member("t0", "T3", 0, host_type=Platforms.TRN3_A0),
            _member("t1", "T3", 1, host_type=Platforms.TRN3_A0),
        ]
        store._update(lambda s: replace(s, hosts=[*s.hosts, *members]))
        with pytest.raises(RequestTooLargeError):
            store.claim_least_busy(Platforms.TRN2, MULTINODE_CORE_THRESHOLD + 1)

    def test_claim_over_ceiling_fails_fast_above_threshold_without_deployments(self, store):
        store.apply_resolved(_resolved("h1", num_physical_cores=128), provisionable_ceiling={Platforms.TRN2: 128})
        with pytest.raises(RequestTooLargeError):
            store.claim_least_busy(Platforms.TRN2, MULTINODE_CORE_THRESHOLD + 1)

    def test_claim_at_threshold_over_ceiling_still_fails_fast(self, store):
        store.apply_resolved(_resolved("h1", num_physical_cores=64), provisionable_ceiling={Platforms.TRN2: 64})
        self._add_busy_deployment(store)
        with pytest.raises(RequestTooLargeError):
            store.claim_least_busy(Platforms.TRN2, MULTINODE_CORE_THRESHOLD)

    def test_ceiling_preserved_across_host_mutations(self, store):
        # The ceiling is fleet metadata: a per-host mutation (mark_unavailable) must not clear it.
        store.apply_resolved(_resolved("h1"), provisionable_ceiling={Platforms.TRN2: 8})
        store.mark_unavailable("h1")
        assert store.provisionable_ceiling_for(Platforms.TRN2) == 8


class TestTemporaryRandomSeed:
    """Tests for the temporary_random_seed context manager."""

    def setup_method(self) -> None:
        """Set a deterministic seed before each test."""
        random.seed(42)

    def teardown_method(self) -> None:
        """Reseed from system entropy so tests don't leak deterministic state."""
        random.seed()

    def test_restores_original_state(self) -> None:
        """RNG state before and after the context manager should be identical."""
        state_before = random.getstate()
        with temporary_random_seed(999):
            random.random()
        assert random.getstate() == state_before

    def test_choice_deterministic_with_same_seed(self) -> None:
        """Same temporary seed produces the same random.choice result."""
        items = ["host_a", "host_b", "host_c"]

        with temporary_random_seed(123):
            first = random.choice(items)

        with temporary_random_seed(123):
            second = random.choice(items)

        assert first == second

    def test_outer_sequence_unchanged_with_choice(self) -> None:
        """random.choice inside the context manager should not alter the outer RNG sequence."""
        expected_next = random.choice(["a", "b", "c"])

        random.seed(42)
        with temporary_random_seed(999):
            random.choice(["a", "b", "c"])
        assert random.choice(["a", "b", "c"]) == expected_next


class TestOwnerVsParticipantSplit:
    """The authoritative whole-fleet mutations live only on OwnerHostStateStore, which only
    the controller (no PYTEST_XDIST_WORKER in env) may construct. Workers get the base store,
    which simply lacks those methods."""

    _OWNER_ONLY = ("reset", "apply_resolved", "set_poisoned")

    @pytest.mark.parametrize("method", _OWNER_ONLY)
    def test_base_store_lacks_owner_mutations(self, method):
        # A worker holding a base store can't perform a controller-only mutation: the method
        # isn't there at all (AttributeError in review), not a silent cross-process write.
        with tempfile.TemporaryDirectory() as d:
            assert not hasattr(HostStateStore(d), method)

    @pytest.mark.parametrize("method", _OWNER_ONLY)
    def test_owner_store_has_owner_mutations(self, method):
        with tempfile.TemporaryDirectory() as d:
            assert callable(getattr(OwnerHostStateStore(d), method))

    def test_owner_store_construction_in_worker_raises(self, monkeypatch):
        # In an xdist worker (PYTEST_XDIST_WORKER set), constructing the owner store is a hard
        # error — only the controller owns fleet-wide reset/reconcile/poison.
        monkeypatch.setenv(PYTEST_XDIST_WORKER_ENV, "gw3")
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(RuntimeError, match="controller-only"):
                OwnerHostStateStore(d)

    def test_owner_is_a_host_state_store(self):
        # Owner extends base, so anything typed for the participant store accepts an owner one
        # (e.g. the refresher passing its owner store where reads/claims also happen).
        with tempfile.TemporaryDirectory() as d:
            assert isinstance(OwnerHostStateStore(d), HostStateStore)

    def test_worker_can_still_construct_base_store(self, monkeypatch):
        # The guard is owner-only: a worker builds its participant store normally.
        monkeypatch.setenv(PYTEST_XDIST_WORKER_ENV, "gw3")
        with tempfile.TemporaryDirectory() as d:
            assert isinstance(HostStateStore(d), HostStateStore)


class TestRecordProbedHosts:
    """The participant-side probe write: capacity and deployment land on existing rows only,
    with every other piece of bookkeeping left untouched."""

    @staticmethod
    def _probed(
        alias: str,
        num_physical_cores: int = 128,
        logical_host_id: str | None = None,
        logical_host_rank: int | None = None,
        logical_host_size: int | None = None,
    ) -> ResolvedHost:
        return ResolvedHost(
            alias, Platforms.TRN2, num_physical_cores, logical_host_id, logical_host_rank, logical_host_size
        )

    def test_a_worker_store_can_record(self, monkeypatch):
        # Plain-target probing runs in every xdist worker, so the base store must expose it.
        monkeypatch.setenv(PYTEST_XDIST_WORKER_ENV, "gw3")
        with tempfile.TemporaryDirectory() as d:
            worker_store = HostStateStore(d)
            worker_store.initialize([ResolvedHost("a", Platforms.TRN2)])
            worker_store.record_probed_hosts([self._probed("a", num_physical_cores=64)])
            state = worker_store.read()
            assert state is not None
            assert state.hosts[0].resolved.num_physical_cores == 64

    def test_records_capacity_and_deployment_preserving_bookkeeping(self, store):
        store._update(
            lambda s: HostState(
                [
                    _record("a", work_queue_depth=4, available=False),
                    _record("b", work_queue_depth=2),
                    _record("other", work_queue_depth=7, num_physical_cores=32),
                ],
                poisoned={Platforms.TRN3},
                under_capacity={Platforms.TRN2},
                provisionable_ceiling={Platforms.TRN2: 128},
            )
        )
        store.record_probed_hosts([self._probed("a", 128, "dep-1", 1, 2), self._probed("b", 128, "dep-1", 0, 2)])
        state = store.read()
        by_alias = {rec.resolved.ssh_host: rec for rec in state.hosts}
        assert [rec.resolved.ssh_host for rec in state.hosts] == ["a", "b", "other"]
        assert by_alias["a"].resolved == _member("a", "dep-1", 1).resolved
        assert by_alias["b"].resolved == _member("b", "dep-1", 0).resolved
        # Availability and depth are the store's, never the probe's.
        assert (by_alias["a"].available, by_alias["a"].work_queue_depth) == (False, 4)
        assert (by_alias["b"].available, by_alias["b"].work_queue_depth) == (True, 2)
        # A row that was not probed is untouched (and not marked unavailable).
        assert by_alias["other"] == _record("other", work_queue_depth=7, num_physical_cores=32)
        assert state.poisoned == {Platforms.TRN3}
        assert state.under_capacity == {Platforms.TRN2}
        assert state.provisionable_ceiling == {Platforms.TRN2: 128}

    def test_zero_cores_and_missing_deployment_keep_known_values(self, store):
        known = _member("a", "dep-1", 3, size=4, num_physical_cores=128, work_queue_depth=5)
        store._update(lambda s: HostState([known]))
        store.record_probed_hosts([self._probed("a", num_physical_cores=0)])
        assert store.read().hosts == [known]

    def test_positive_cores_refresh_without_erasing_a_known_deployment(self, store):
        store._update(lambda s: HostState([_member("a", "dep-1", 3, size=4, num_physical_cores=64)]))
        store.record_probed_hosts([self._probed("a", num_physical_cores=128)])
        assert store.read().hosts[0].resolved == _member("a", "dep-1", 3, size=4, num_physical_cores=128).resolved

    def test_unknown_alias_is_not_added(self, store):
        store._update(lambda s: HostState([_record("a", num_physical_cores=64)]))
        store.record_probed_hosts([self._probed("ghost", logical_host_id="dep-1", logical_host_rank=0)])
        assert [rec.resolved.ssh_host for rec in store.read().hosts] == ["a"]

    def test_store_platform_wins_over_the_probe(self, store):
        store._update(lambda s: HostState([_record("a", Platforms.TRN3)]))
        store.record_probed_hosts([self._probed("a", num_physical_cores=64)])
        resolved = store.read().hosts[0].resolved
        assert (resolved.host_type, resolved.num_physical_cores) == (Platforms.TRN3, 64)

    def test_nothing_probed_skips_the_write(self, store):
        store.record_probed_hosts([])
        assert store.read() is None  # no file was created
