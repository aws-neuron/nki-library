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

import pytest

from test.utils.common_dataclasses import PYTEST_XDIST_WORKER_ENV, Platforms, ResolvedHost
from test.utils.exceptions import FleetEmptyError, RequestTooLargeError
from test.utils.host_state import (
    HostRecord,
    HostState,
    HostStateStore,
    OwnerHostStateStore,
    temporary_random_seed,
)


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


@pytest.fixture
def store():
    # OwnerHostStateStore so these tests can exercise the controller-only mutations
    # (reset/apply_resolved/set_poisoned) alongside the base participant ops it inherits.
    # Unit tests run serially (no PYTEST_XDIST_WORKER), so the owner guard permits it.
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
        assert claimed.resolved.ssh_host == "idle"
        # Returned record reflects the post-increment depth, and is persisted.
        assert claimed.work_queue_depth == 1
        assert {h.resolved.ssh_host: h.work_queue_depth for h in store.read().hosts} == {"busy": 5, "idle": 1}

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
        assert store.claim_least_busy(Platforms.TRN2, 1).resolved.ssh_host == "h1"

    def test_host_aliases_for_platform(self, store):
        store._update(
            lambda s: HostState([_record("a", Platforms.TRN2, available=False), _record("b", Platforms.TRN3_A0)])
        )
        # Includes unavailable hosts; filtered by platform.
        assert store.host_aliases_for(Platforms.TRN2) == ["a"]
        assert store.host_aliases_for(Platforms.TRN3_A0) == ["b"]


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
        assert store.claim_least_busy(Platforms.TRN2, 1).resolved.ssh_host == "h1"

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
        assert store.claim_least_busy(Platforms.TRN2, 1).resolved.ssh_host == "h1"

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
        assert store.claim_least_busy(Platforms.TRN2, 8).resolved.ssh_host == "h1"

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
