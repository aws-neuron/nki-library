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
"""Single source of truth for the remote host pool, persisted to a state file.

Every pytest worker reads the file on each host acquisition and updates the chosen host's ``work_queue_depth``.
All access goes through ``HostStateStore``, which serializes read-modify-write
under a file lock and publishes via temp-file + ``os.replace`` so readers never
see a partial file.

Schema::

    {
      "hosts": [
        {"host_alias": "1.2.3.4", "host_type": "trn2",
         "work_queue_depth": 0, "available": true},
        ...
      ]
    }

A host leaves the pool by being marked ``available: false`` (recoverable) — by a worker
on a host-level failure (connection fault or exhausted core-lock acquisition), or by
dropping out of a poll's resolved set. It becomes available again when a later
resolution re-lists it (see ``apply_resolved``).

Rows are kept (not deleted). The controller resets this file once at session start,
so workers never inherit stale prior-session state.
"""

import contextlib
import json
import logging
import math
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Callable, Generator, Iterator, Self

from filelock import FileLock

from .common_dataclasses import PYTEST_XDIST_WORKER_ENV, Platforms, ResolvedHost, is_xdist_worker
from .exceptions import FleetEmptyError, RequestTooLargeError

logger = logging.getLogger(__name__)

# A request needing more physical cores than this is also offered multi-node deployments, each
# claimed as one unit; at or below it, every host (multi-node member or not) is claimed alone.
MULTINODE_CORE_THRESHOLD = 128


class PlacementRejection(Enum):
    """Capacity constraints that prevent placing a request on a host unit."""

    UNSPLITTABLE = "unsplittable"
    TOO_SMALL = "too_small"
    MEMBER_TOO_SMALL = "member_too_small"


def validate_core_request(num_physical_cores_needed: int, lnc_config: int) -> None:
    """Raise ``ValueError`` unless the request is a positive whole number of logical ranks."""
    if num_physical_cores_needed <= 0:
        raise ValueError(f"Requested physical cores must be positive, got {num_physical_cores_needed}.")
    if lnc_config <= 0:
        raise ValueError(f"lnc_config must be positive, got {lnc_config}.")
    if num_physical_cores_needed % lnc_config:
        raise ValueError(
            f"Requested {num_physical_cores_needed} physical cores is not divisible by lnc_config={lnc_config}."
        )


def per_member_physical_cores(num_physical_cores_needed: int, member_count: int, *, lnc_config: int = 1) -> int | None:
    """Physical cores each of ``member_count`` hosts must provide when the request's logical ranks
    are split evenly over them, or None if the ranks do not divide by ``member_count``."""
    validate_core_request(num_physical_cores_needed, lnc_config)
    ranks = num_physical_cores_needed // lnc_config
    if ranks % member_count:
        return None
    return ranks // member_count * lnc_config


@contextlib.contextmanager
def temporary_random_seed(seed: int) -> Generator[None, None, None]:
    """Temporarily reseed the global RNG, restoring the original state on exit."""
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


class HostUnit(ABC):
    """One claimable unit of the pool: a single host (``HostRecord``) or every member of a
    multi-node deployment (``HostGroup``). Iterating a unit yields its ``HostRecord`` members,
    and the shared behaviour below is defined over that iteration, so the store and its callers
    handle both kinds the same way."""

    @abstractmethod
    def __iter__(self) -> Iterator["HostRecord"]: ...

    @property
    @abstractmethod
    def primary(self) -> "HostRecord":
        """The member that identifies the unit (the lowest-ranked pooled member of a multi-node
        deployment)."""

    @abstractmethod
    def snapshot(self) -> Self:
        """A detached copy, safe to hand out after the store lock is released."""

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @property
    def host_id(self) -> str:
        return self.primary.resolved.ssh_host

    @property
    def member_ids(self) -> tuple[str, ...]:
        """Aliases of the members in iteration order; pass to ``HostStateStore.release`` so a
        release touches exactly the hosts this claim reserved."""
        return tuple(rec.resolved.ssh_host for rec in self)

    @property
    def host_type(self) -> Platforms:
        """The members' common platform. Raises ``ValueError`` if members disagree."""
        host_types = {rec.resolved.host_type for rec in self}
        if len(host_types) != 1:
            raise ValueError(
                f"Host unit {self.host_id} mixes platforms {sorted(p.value for p in host_types)}; "
                f"a multi-node deployment must be homogeneous."
            )
        return host_types.pop()

    @property
    def total_physical_cores(self) -> int:
        return sum(rec.resolved.num_physical_cores for rec in self)

    @property
    def is_claimable(self) -> bool:
        return all(rec.available for rec in self)

    def can_serve(self, platform: Platforms, num_physical_cores_needed: int, *, lnc_config: int = 1) -> bool:
        """True if the unit can take the request. A unit with no member on ``platform`` is simply
        not a candidate; one that has a member there but mixes platforms raises ``ValueError``
        (see ``host_type``), whatever the members' availability. Every member must be available
        and report a known (positive) core count, and the placement must be legal (see
        ``placement_rejection``)."""
        if not any(rec.resolved.host_type == platform for rec in self):
            return False
        return (
            self.host_type == platform
            and self.is_claimable
            and all(rec.resolved.num_physical_cores > 0 for rec in self)
            and self.placement_rejection(num_physical_cores_needed, lnc_config=lnc_config) is None
        )

    def placement_rejection(self, num_physical_cores_needed: int, *, lnc_config: int = 1) -> PlacementRejection | None:
        """Why the request cannot be placed on this unit's capacity, or None if it can. The
        request's logical ranks (``num_physical_cores_needed // lnc_config``) are split evenly over
        the members, so each member must host ``num_physical_cores_needed // len(self)`` physical
        cores. Returns ``UNSPLITTABLE`` if the ranks do not divide by the member count,
        ``TOO_SMALL`` if the members' total capacity is below the request, and
        ``MEMBER_TOO_SMALL`` if the total suffices but some member is below its share. Ignores
        availability and unknown capacity (see ``can_serve``)."""
        share = per_member_physical_cores(num_physical_cores_needed, len(self), lnc_config=lnc_config)
        if share is None:
            return PlacementRejection.UNSPLITTABLE
        if self.total_physical_cores < num_physical_cores_needed:
            return PlacementRejection.TOO_SMALL
        if any(rec.resolved.num_physical_cores < share for rec in self):
            return PlacementRejection.MEMBER_TOO_SMALL
        return None

    def post_placement_load(self, num_physical_cores_needed: int) -> float:
        """Mean over members of ``(work_queue_depth + needed) / num_physical_cores`` — the load
        each member WOULD carry after this request. A member of unknown (0) capacity counts as
        infinitely loaded."""

        def _member_load(rec: "HostRecord") -> float:
            if rec.resolved.num_physical_cores <= 0:
                return math.inf
            return (rec.work_queue_depth + num_physical_cores_needed) / rec.resolved.num_physical_cores

        return sum(_member_load(rec) for rec in self) / len(self)


@dataclass
class HostRecord(HostUnit):
    """One host row in host_state.json: a :class:`ResolvedHost` (the host's identity and
    probed capacity) augmented with the store's runtime bookkeeping (``available`` and
    ``work_queue_depth``).

    ``resolved.host_type`` is stored as a string on disk but kept as a Platforms in memory
    (converted at the JSON boundary). Every writer supplies a real platform, so it is
    mandatory; a row read back with a missing or unknown one is dropped (see ``from_json``).

    ``available`` means "claimable right now". A host marked unavailable (available=false)
    is recoverable: a later resolution that lists it makes it claimable again.

    ``work_queue_depth`` accumulates the physical cores reserved by in-flight claims (a
    claim adds the request's core count, release subtracts it). ``resolved.num_physical_cores``
    is the host's total physical-core capacity, probed when the host enters the pool; a host
    with fewer cores than a request needs is ineligible for it, and the two together give
    the post-placement load ratio used to route work by headroom. A host whose capacity is
    unknown (never probed / stale row) reads back as 0, which makes it ineligible for any
    request (needed cores are always >= 1) rather than risking a divide-by-zero in ranking.

    ``resolved.logical_host_*`` records the multi-node deployment the host belongs to when its
    tooling reports one; those keys are absent from the row for a single-node host."""

    resolved: ResolvedHost
    work_queue_depth: int = 0
    available: bool = True

    def to_json(self) -> dict:
        row = {
            "host_alias": self.resolved.ssh_host,
            "host_type": self.resolved.host_type.value,
            "work_queue_depth": self.work_queue_depth,
            "available": self.available,
            "num_physical_cores": self.resolved.num_physical_cores,
        }
        # Written only for a multi-node host, so a single-node fleet's rows are unchanged.
        if self.resolved.logical_host_id is not None:
            row["logical_host_id"] = self.resolved.logical_host_id
            row["logical_host_rank"] = self.resolved.logical_host_rank
            row["logical_host_size"] = self.resolved.logical_host_size
        return row

    @classmethod
    def from_json(cls, raw: dict) -> "HostRecord | None":
        """Parse one row, or return None to drop it. A row whose ``host_type`` is
        missing or unparseable is dropped rather than admitted as a typeless host
        the assignment filter could never match (tolerates a stale/corrupt file)."""
        host_alias = raw.get("host_alias")
        if not host_alias:
            logger.warning("Dropping host row with missing host_alias: %r", raw)
            return None
        raw_type = raw.get("host_type")
        host_type = Platforms.from_str_safe(raw_type) if raw_type else None
        if host_type is None:
            logger.warning("Dropping host row %r: missing or unknown host_type %r", host_alias, raw_type)
            return None
        return cls(
            resolved=ResolvedHost(
                ssh_host=host_alias,
                host_type=host_type,
                num_physical_cores=raw.get("num_physical_cores", 0),
                logical_host_id=raw.get("logical_host_id"),
                logical_host_rank=raw.get("logical_host_rank"),
                logical_host_size=raw.get("logical_host_size"),
            ),
            work_queue_depth=raw.get("work_queue_depth", 0),
            available=raw.get("available", False),
        )

    def __iter__(self) -> Iterator["HostRecord"]:
        yield self

    @property
    def primary(self) -> "HostRecord":
        return self

    def snapshot(self) -> Self:
        return replace(self)


@dataclass
class HostGroup(HostUnit):
    """Every pooled member of one multi-node deployment, ordered by ``logical_host_rank``, claimed
    and released as a single unit. The members are the store's own rows while under the lock
    (so a reservation updates the state in place) and copies once handed out."""

    members: tuple[HostRecord, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError("A HostGroup needs at least one member.")

    def __iter__(self) -> Iterator[HostRecord]:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    @property
    def primary(self) -> HostRecord:
        return self.members[0]

    def snapshot(self) -> Self:
        return replace(self, members=tuple(replace(rec) for rec in self.members))


def _reserve(unit: HostUnit, num_physical_cores: int) -> None:
    """Add ``num_physical_cores`` to every member's ``work_queue_depth``."""
    for rec in unit:
        rec.work_queue_depth += num_physical_cores


def _unreserve(unit: HostUnit, num_physical_cores: int) -> None:
    """Subtract ``num_physical_cores`` from every member's ``work_queue_depth``, floored at 0.
    Going below 0 is an unbalanced release (a claim/release bug), so it is logged."""
    for rec in unit:
        if rec.work_queue_depth - num_physical_cores < 0:
            logger.warning(
                "Releasing %d cores from host %s whose work_queue_depth is %d "
                "(unbalanced release - likely a claim/release bug)",
                num_physical_cores,
                rec.resolved.ssh_host,
                rec.work_queue_depth,
            )
        rec.work_queue_depth = max(0, rec.work_queue_depth - num_physical_cores)


def _host_units(hosts: list[HostRecord], num_physical_cores_needed: int) -> list[HostUnit]:
    """Partition ``hosts`` into the units a ``num_physical_cores_needed`` request selects from.

    At or below ``MULTINODE_CORE_THRESHOLD`` every host is its own unit. Above it, hosts sharing a
    ``logical_host_id`` form one ``HostGroup`` (their members are not offered individually) and the
    rest stay single. A group with fewer pooled members than its reported ``logical_host_size`` is
    still offered (the store logs it; see ``HostStateStore._units``)."""
    if num_physical_cores_needed <= MULTINODE_CORE_THRESHOLD:
        return list(hosts)

    units: list[HostUnit] = []
    deployments: dict[str, list[HostRecord]] = defaultdict(list)
    for rec in hosts:
        if rec.resolved.logical_host_id is None:
            units.append(rec)
        else:
            deployments[rec.resolved.logical_host_id].append(rec)

    for members in deployments.values():
        members.sort(key=lambda rec: (rec.resolved.logical_host_rank is None, rec.resolved.logical_host_rank or 0))
        units.append(HostGroup(members=tuple(members)))
    return units


@dataclass
class HostState:
    """Whole-file contents: the host records, plus the sets of poisoned and under-capacity
    platforms.

    ``poisoned`` holds the platforms that are perpetually empty; a claim for a poisoned
    platform fails fast instead of waiting.

    ``under_capacity`` holds the platforms where too few of the expected hosts are currently
    available (below the healthy-ratio floor the refresher enforces). A claim for an
    under-capacity platform is withheld — the same as no host being available — so work isn't
    piled onto the handful that came up while the rest of the fleet is still scaling.

    ``provisionable_ceiling`` maps a platform to the physical cores of the LARGEST single host
    the fleet could ever provision for it (the biggest per-host size the fleet can bring up).
    A claim needing more cores than this can never be served, so it fails fast.

    Limitation: the ceiling bounds a single host only, so it is not applied to a request above
    ``MULTINODE_CORE_THRESHOLD`` when the platform has a multi-node deployment in the pool (which
    may serve it); such a request never fails fast on size and instead waits out the claim
    deadline if nothing can serve it.
    """

    hosts: list[HostRecord]
    poisoned: set[Platforms] = field(default_factory=set)
    under_capacity: set[Platforms] = field(default_factory=set)
    provisionable_ceiling: dict[Platforms, int] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "hosts": [h.to_json() for h in self.hosts],
            "poisoned": sorted(p.value for p in self.poisoned),
            "under_capacity": sorted(p.value for p in self.under_capacity),
            "provisionable_ceiling": {p.value: cores for p, cores in self.provisionable_ceiling.items()},
        }

    @classmethod
    def from_json(cls, raw: dict) -> "HostState":
        parsed = (HostRecord.from_json(h) for h in raw.get("hosts", []))
        poisoned = {p for raw_p in raw.get("poisoned", []) if (p := Platforms.from_str_safe(raw_p)) is not None}
        under_capacity = {
            p for raw_p in raw.get("under_capacity", []) if (p := Platforms.from_str_safe(raw_p)) is not None
        }
        provisionable_ceiling = {
            p: cores
            for raw_p, cores in raw.get("provisionable_ceiling", {}).items()
            if (p := Platforms.from_str_safe(raw_p)) is not None
        }
        return cls(
            hosts=[rec for rec in parsed if rec is not None],
            poisoned=poisoned,
            under_capacity=under_capacity,
            provisionable_ceiling=provisionable_ceiling,
        )


class HostStateStore:
    """Atomic, file-locked accessor for the host-state file

    This class only allows access to operations that should be accessible in
    parallel. Functionality that should only be controlled by a single source
    is available in `OwnerHostStateStore`.

    Concurrency model:
      * A file lock serializes every read-modify-write across processes.
      * Writes go to a temp file then ``os.replace`` (atomic on POSIX), so a
        concurrent reader either sees the old file or the new one, never a
        partial write.
    """

    # The on-disk filename
    HOST_STATE_FILENAME = "host_state.json"

    def __init__(self, base_dir: str, lock_timeout_seconds: int = 10) -> None:
        self.base_dir = base_dir
        self.state_path = os.path.join(base_dir, self.HOST_STATE_FILENAME)
        self._lock = FileLock(f"{self.state_path}.lock", timeout=lock_timeout_seconds)
        self._initialized = False
        # (logical_host_id, pooled members, expected size) already reported as incomplete.
        self._warned_incomplete: set[tuple[str, int, int]] = set()

    def _units(self, hosts: list[HostRecord], num_physical_cores_needed: int) -> list[HostUnit]:
        """``_host_units``, logging each incomplete deployment once per distinct membership."""
        units = _host_units(hosts, num_physical_cores_needed)
        for unit in units:
            if not isinstance(unit, HostGroup):
                continue
            logical_host_id = unit.primary.resolved.logical_host_id
            expected_size = unit.primary.resolved.logical_host_size
            if logical_host_id is None or expected_size is None or len(unit) >= expected_size:
                continue
            key = (logical_host_id, len(unit), expected_size)
            if key in self._warned_incomplete:
                continue
            self._warned_incomplete.add(key)
            logger.warning(
                "Multi-node deployment %s has %d of %d members in the pool",
                logical_host_id,
                len(unit),
                expected_size,
            )
        return units

    # --- low-level -------------------------------------------------

    def _write_unlocked(self, state: HostState) -> None:
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp_path = f"{self.state_path}.{os.getpid()}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(state.to_json(), f)
        os.replace(tmp_path, self.state_path)
        self._initialized = True

    def _update(self, mutator: Callable[[HostState], HostState | None]) -> HostState:
        """Locked read-modify-write primitive that every named operation builds on.

        Private: the store exposes a fixed vocabulary of operations built on this,
        rather than letting callers run arbitrary code under the lock.

        ``mutator`` receives the current ``HostState`` (an empty one if the file
        is absent) and returns the new state to write; returning ``None`` means
        "no change" and skips the write. Returns the resulting state.
        """
        with self._lock.acquire():
            # read() under the lock here is the read-half of this read-modify-write.
            state = self.read() or HostState(hosts=[])
            new_state = mutator(state)
            if new_state is None:
                return state
            self._write_unlocked(new_state)
            return new_state

    # --- public reads / lifecycle --------------------------------------------

    def read(self) -> HostState | None:
        """Read the current state. Lock-free: readers tolerate a stale snapshot, and
        os.replace guarantees each snapshot is internally consistent (never partial)."""
        try:
            with open(self.state_path, "r") as f:
                return HostState.from_json(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def unavailable_count(self) -> int:
        """Number of hosts currently marked unavailable (``available=False``).

        Deduplicated by host (one row per alias), so this is the authoritative
        "failed host" gauge — unlike counting per-failure events. Zero when the
        state file is absent."""
        state = self.read()
        return sum(1 for rec in state.hosts if not rec.available) if state else 0

    def is_poisoned(self, platform: Platforms) -> bool:
        """True if ``platform`` has been declared perpetually empty (poisoned), so
        callers should fail fast rather than wait for a host of that type. Lock-free;
        False when the state file is absent."""
        state = self.read()
        return platform in state.poisoned if state else False

    def is_under_capacity(self, platform: Platforms) -> bool:
        """True if ``platform`` has too few of its expected hosts available (below the
        healthy-ratio floor), so its hosts are being withheld from claims. Lock-free;
        False when the state file is absent."""
        state = self.read()
        return platform in state.under_capacity if state else False

    def provisionable_ceiling_for(self, platform: Platforms) -> int | None:
        """The cores of the largest host the fleet could ever provision for ``platform``, or
        ``None`` if unbounded (no known ceiling — the fast-fail guard is off for it). Lock-free;
        None when the state file is absent."""
        state = self.read()
        return state.provisionable_ceiling.get(platform) if state else None

    def is_initialized(self) -> bool:
        """True once THIS store has persisted state this run, letting a caller skip
        a redundant bootstrap. Deliberately tracks our own writes, not file
        existence: a leftover host_state.json from a crashed prior run must not
        read as initialized. OwnerHostStateStore.reset() returns this to False."""
        return self._initialized

    # --- named operations (the only sanctioned mutations) --------------------

    def initialize(self, resolved: list[ResolvedHost]) -> None:
        """Add any bootstrap hosts missing from the current state, at depth 0.

        Only *adds* — never marks anything unavailable, because only a full
        re-resolution (see ``apply_resolved``) is the authority on which hosts have
        disappeared. The state file is reset once at session start, so there is no
        stale prior-session state.

        Bootstrap hosts are shuffled before insertion so their append order — the
        tie-break ``claim_least_busy`` falls back on when several equal-size hosts sit at
        equal load (notably every host at depth 0 right after startup) — differs from run
        to run. Without it every suite would deterministically claim the first-listed host
        first and hammer it. ``temporary_random_seed`` seeds from the wall clock for the
        shuffle then restores the global RNG, so the harness's reproducible xdist collection
        seed is left intact.
        """
        shuffled = list(resolved)
        with temporary_random_seed(time.time_ns()):
            random.shuffle(shuffled)
        self._update(lambda state: _add_new_hosts(state, shuffled))

    def claim_least_busy(
        self,
        platform: Platforms,
        num_physical_cores_needed: int,
        exclude: set[str] | None = None,
        *,
        lnc_config: int = 1,
    ) -> HostUnit | None:
        """Atomically select the best-fit available unit for ``platform`` that can serve a
        request needing ``num_physical_cores_needed`` physical cores, reserve those cores on
        every member (``work_queue_depth += num_physical_cores_needed``), and return a copy of
        the claimed unit. Selection + reservation are one locked read-modify-write so
        concurrent workers can't both claim the same headroom.

        Units: at or below ``MULTINODE_CORE_THRESHOLD`` every host is a candidate on its own
        (``HostRecord``); above it, each multi-node deployment is one candidate (``HostGroup``)
        alongside the single-node hosts (see ``_host_units``).

        Eligibility: every member available, matching platform, every member's capacity known
        (positive), the request's logical ranks (``num_physical_cores_needed // lnc_config``)
        divisible by the member count, and every member's ``num_physical_cores`` at least its
        even share of the request (see ``HostUnit.placement_rejection``). A deployment with fewer
        pooled members than its reported size is still a candidate when those rules hold.
        Every member is charged the full request, not its share.

        Ranking: minimum POST-placement load ratio ``(work_queue_depth + needed) /
        num_physical_cores``, averaged over the unit's members — the ratio the unit WOULD have
        after taking this request. This routes toward the unit with the most proportional
        headroom, so a larger host absorbs proportionally more work than a smaller one instead
        of an equal per-host share, while a small idle host still wins until a large host's
        resulting ratio climbs past it.

        ``exclude`` holds hosts already tried-and-busy during THIS allocation; a unit with any
        excluded member is deprioritized so a deterministic pick rotates across the fleet
        instead of re-picking the same unit. If every eligible unit is excluded, the full
        eligible set is used so a unit is still returned.

        Returns None if no eligible unit matches (the caller waits/retries). When nothing is
        claimable, raises ``RequestTooLargeError`` if the request needs more cores than the
        fleet's largest provisionable host for the platform (an impossible ask no wait can
        satisfy), else ``FleetEmptyError`` if the platform is poisoned — both unretriable. The
        ceiling check is skipped above ``MULTINODE_CORE_THRESHOLD`` when the platform has a
        multi-node deployment in the pool (see ``HostState``).

        Raises ``ValueError`` if a candidate multi-node deployment with a member on ``platform``
        mixes platforms (see ``HostUnit.can_serve``), or if the request is not a positive whole
        number of ``lnc_config`` logical ranks (see ``validate_core_request``)."""
        validate_core_request(num_physical_cores_needed, lnc_config)
        excluded: set[str] = exclude or set()
        claimed: list[HostUnit] = []  # used as Box for return

        def _claim(state: HostState) -> HostState | None:
            if platform in state.under_capacity:
                return None  # too few of the expected hosts up; withhold as if none available

            eligible = [
                unit
                for unit in self._units(state.hosts, num_physical_cores_needed)
                if unit.can_serve(platform, num_physical_cores_needed, lnc_config=lnc_config)
            ]
            if not eligible:
                return None  # no claimable unit; no write

            # Prefer units not busy this allocation; fall back to all eligible if every one
            # has already been tried (so the deadline loop keeps cycling rather than stalling).
            selectable = [
                unit for unit in eligible if not any(rec.resolved.ssh_host in excluded for rec in unit)
            ] or eligible
            chosen = min(selectable, key=lambda unit: unit.post_placement_load(num_physical_cores_needed))
            _reserve(chosen, num_physical_cores_needed)
            claimed.append(chosen.snapshot())
            return state

        state = self._update(_claim)
        if claimed:
            return claimed[0]

        # TODO: bound requests a multi-node deployment may serve too; the ceiling only describes
        # the largest single host, so it is skipped when such a deployment exists.
        ceiling = state.provisionable_ceiling.get(platform)
        deployment_may_serve = num_physical_cores_needed > MULTINODE_CORE_THRESHOLD and any(
            rec.resolved.logical_host_id is not None and rec.resolved.host_type == platform for rec in state.hosts
        )
        if ceiling is not None and not deployment_may_serve and num_physical_cores_needed > ceiling:
            raise RequestTooLargeError(
                f"Request needs {num_physical_cores_needed} physical cores, but the largest host "
                f"the fleet can provision for platform {platform.value} is {ceiling} cores; "
                f"no host will ever be large enough."
            )
        if platform in state.poisoned:
            raise FleetEmptyError(f"Fleet is perpetually empty for platform {platform.value}.")
        return None

    def get_all_linked_hosts(self, logical_node_id: str) -> list[HostRecord] | None:
        all_host_records = self.read()

        if not all_host_records:
            return None

        return [
            host_record
            for host_record in all_host_records.hosts
            if host_record.resolved.logical_host_id == logical_node_id
        ]

    def release(self, host_id: str, num_physical_cores: int, member_ids: tuple[str, ...] | None = None) -> None:
        """Release ``num_physical_cores`` reserved by a claim of ``num_physical_cores`` on the
        unit containing ``host_id`` (decrement every member's work_queue_depth, floored at 0).

        Prefer passing ``member_ids`` from the claimed unit (``HostUnit.member_ids``): exactly
        those rows are released, so a deployment gaining, losing, or renaming members since the
        claim does not change what is released. A ``host_id`` outside them is logged as an error
        and exactly ``member_ids`` is still released (never raises, so it is safe in a caller's
        finally); an alias with no row is skipped and logged. Without ``member_ids`` the unit is
        re-derived from the current state with the claim's core count, which can mis-attribute
        the release if deployment membership changed since the claim; a ``host_id`` in no unit
        is logged and nothing is released.

        Every claim is balanced by exactly one release of the same core count (in the caller's
        finally). A release that would drive a depth below 0 is an unbalanced release — a
        double-release or a release of a never-claimed host — so it is floored at 0 and logged,
        since it points to a claim/release bug rather than corrupting state."""
        if member_ids is not None and host_id not in member_ids:
            logger.error(
                "Released host %s is not among the claimed members %s; releasing the members",
                host_id,
                sorted(member_ids),
            )

        def _release_members(state: HostState, ids: tuple[str, ...]) -> HostState | None:
            rows = [rec for rec in state.hosts if rec.resolved.ssh_host in ids]
            missing = set(ids) - {rec.resolved.ssh_host for rec in rows}
            if missing:
                logger.warning("Releasing claim on %s: no host row for %s", host_id, sorted(missing))
            if not rows:
                return None
            for rec in rows:
                _unreserve(rec, num_physical_cores)
            return state

        def _rederive(state: HostState) -> HostState | None:
            for unit in self._units(state.hosts, num_physical_cores):
                if any(rec.resolved.ssh_host == host_id for rec in unit):
                    _unreserve(unit, num_physical_cores)
                    return state
            logger.warning("Releasing claim on %s: no host row matches; nothing released", host_id)
            return None

        if member_ids is not None:
            ids = member_ids
            self._update(lambda state: _release_members(state, ids))
        else:
            self._update(_rederive)

    def set_physical_cores(self, cores_by_alias: dict[str, int]) -> None:
        """Persist probed physical-core counts onto matching host rows. Only positive
        counts are written (a 0/unknown probe never clobbers a known capacity), so a host
        whose probe failed stays at whatever it had — and an unprobed host stays at 0
        (ineligible) until a real count arrives."""

        def _mutate(state: HostState) -> HostState | None:
            changed = False
            for rec in state.hosts:
                cores = cores_by_alias.get(rec.resolved.ssh_host, 0)
                if cores > 0 and rec.resolved.num_physical_cores != cores:
                    rec.resolved = replace(rec.resolved, num_physical_cores=cores)
                    changed = True
            return state if changed else None

        self._update(_mutate)

    def record_probed_hosts(self, probed: list[ResolvedHost]) -> None:
        """Persist probed capacity and multi-node deployment onto the EXISTING rows they name,
        in one locked write. Each field follows ``apply_resolved``'s refresh rule: a positive core
        count replaces the stored one (0/unknown never clobbers it), and a reported deployment
        replaces the stored one (an unreported one leaves it intact). The row's platform comes
        from the store, not the probe.

        Participant-safe: unlike ``apply_resolved`` this never adds or removes hosts, nor touches
        availability, ``work_queue_depth`` or fleet metadata, so a host that was not probed (or
        whose probe failed) keeps its row exactly as it was."""
        probed_by_alias = {host.ssh_host: host for host in probed}

        def _mutate(state: HostState) -> HostState | None:
            changed = False
            for rec in state.hosts:
                match = probed_by_alias.get(rec.resolved.ssh_host)
                if match is not None:
                    changed |= _refresh_probed_fields(rec, match)
            return state if changed else None

        self._update(_mutate)

    def mark_unavailable(self, host_id: str) -> None:
        """Mark a host unavailable after a host-level failure; the row is kept (available=false). A later
        resolution re-adds it once a source confirms it's available again. Each source defines availability."""

        def _mutate(state: HostState) -> HostState | None:
            changed = False
            for rec in state.hosts:
                if rec.resolved.ssh_host == host_id and rec.available:
                    rec.available = False
                    changed = True
            return state if changed else None

        self._update(_mutate)

    def host_aliases_for(self, platform: Platforms) -> list[str]:
        """All host aliases matching ``platform`` (regardless of availability).
        Used to distinguish 'no such hosts' from 'all busy/unavailable' for errors."""
        state = self.read()
        return [rec.resolved.ssh_host for rec in state.hosts if rec.resolved.host_type == platform] if state else []

    def eligible_host_aliases(
        self, platform: Platforms, num_physical_cores_needed: int, *, lnc_config: int = 1
    ) -> set[str]:
        """One identifier per unit currently eligible to serve a ``num_physical_cores_needed``
        request on ``platform``: the unit's ``host_id`` (its primary's alias). Uses the same
        partition (``_host_units``) and predicate (``can_serve``, platform not under-capacity)
        that ``claim_least_busy`` selects from. The retry loop records a busy claim by this id,
        which excludes the whole unit from the next claim, and compares its busy set against
        this to know when every eligible unit has been tried-and-busy so it can re-rotate.

        Raises ``ValueError`` like ``claim_least_busy`` for a relevant mixed-platform
        deployment or an invalid request."""
        validate_core_request(num_physical_cores_needed, lnc_config)
        state = self.read()
        if state is None or platform in state.under_capacity:
            return set()
        return {
            unit.host_id
            for unit in self._units(state.hosts, num_physical_cores_needed)
            if unit.can_serve(platform, num_physical_cores_needed, lnc_config=lnc_config)
        }

    def explain_unclaimable(self, platform: Platforms, num_physical_cores_needed: int, *, lnc_config: int = 1) -> str:
        """Human-readable reason no host of ``platform`` could serve a
        ``num_physical_cores_needed`` request, for the terminal error. Distinguishes a
        genuine fleet-wide failure from a request simply too large for any host:
          * no host of this platform exists in the pool at all; else
          * a per-reason breakdown over matching units (any member on ``platform``) — the same
            candidates ``claim_least_busy`` selects from: unavailable / with unknown capacity
            (a member reporting 0 cores) / whose host count cannot evenly split the request's
            logical ranks / too small in total (with the largest available core count) / with a
            member below its even per-host share. Each unit is counted under exactly one reason.

        Raises ``ValueError`` for an invalid request (see ``validate_core_request``)."""
        validate_core_request(num_physical_cores_needed, lnc_config)
        state = self.read()
        units = self._units(state.hosts, num_physical_cores_needed) if state else []
        matching = [unit for unit in units if any(rec.resolved.host_type == platform for rec in unit)]
        if not matching:
            return f"no host of platform {platform.value} exists in the pool"
        unavailable = [unit for unit in matching if not unit.is_claimable]
        claimable = [unit for unit in matching if unit.is_claimable]
        unknown_capacity = [unit for unit in claimable if any(rec.resolved.num_physical_cores <= 0 for rec in unit)]
        rejected: dict[PlacementRejection, list[HostUnit]] = defaultdict(list)
        for unit in claimable:
            if any(rec.resolved.num_physical_cores <= 0 for rec in unit):
                continue
            reason = unit.placement_rejection(num_physical_cores_needed, lnc_config=lnc_config)
            if reason is not None:
                rejected[reason].append(unit)
        num_ranks = num_physical_cores_needed // lnc_config
        reasons: list[str] = []
        if unavailable:
            reasons.append(f"{len(unavailable)} unavailable")
        if unknown_capacity:
            reasons.append(f"{len(unknown_capacity)} with unknown capacity")
        for size in sorted({len(unit) for unit in rejected[PlacementRejection.UNSPLITTABLE]}):
            count = sum(1 for unit in rejected[PlacementRejection.UNSPLITTABLE] if len(unit) == size)
            reasons.append(f"{count} whose {size} hosts cannot evenly split {num_ranks} logical ranks")
        if rejected[PlacementRejection.TOO_SMALL]:
            largest = max(unit.total_physical_cores for unit in rejected[PlacementRejection.TOO_SMALL])
            reasons.append(
                f"{len(rejected[PlacementRejection.TOO_SMALL])} too small (largest available {largest} cores)"
            )
        for size in sorted({len(unit) for unit in rejected[PlacementRejection.MEMBER_TOO_SMALL]}):
            count = sum(1 for unit in rejected[PlacementRejection.MEMBER_TOO_SMALL] if len(unit) == size)
            share = num_physical_cores_needed // size
            reasons.append(f"{count} with a host below its {share}-core share")
        breakdown = ", ".join(reasons) if reasons else "reason unknown"
        return (
            f"needing {num_physical_cores_needed} physical cores - of {len(matching)} matching candidates: {breakdown}"
        )


class OwnerHostStateStore(HostStateStore):
    """The controller-only view of the store, adding the authoritative whole-fleet
    mutations (``reset``, ``apply_resolved``, ``set_poisoned``).

    These reconcile or wipe the *entire* pool, so exactly one process — the controller,
    which owns the fleet lifecycle and the refresher — may perform them. Constructing this
    in an xdist worker raises: a worker reconciling against its partial view would wrongly
    mark its peers' hosts unavailable, and two processes resetting/poisoning would race on
    the shared file. Workers use the base :class:`HostStateStore`, which simply does not
    expose these methods (calling one is an ``AttributeError``, caught in review, not a
    silent corruption in a shared-fleet run)."""

    def __init__(self, base_dir: str, lock_timeout_seconds: int = 10) -> None:
        if is_xdist_worker():
            worker_id = os.environ.get(PYTEST_XDIST_WORKER_ENV)
            raise RuntimeError(
                f"OwnerHostStateStore constructed in xdist worker {worker_id!r}; the "
                "authoritative store (reset/apply_resolved/set_poisoned) is controller-only. "
                "Workers must use HostStateStore."
            )
        super().__init__(base_dir, lock_timeout_seconds)

    def reset(self) -> None:
        """Delete the state file, discarding any stale prior-session contents.

        Called once by the controller at session start so workers begin from a
        clean slate without needing a session id to detect staleness.
        """
        with self._lock.acquire():
            try:
                os.remove(self.state_path)
            except FileNotFoundError:
                pass
        self._initialized = False
        self._warned_incomplete.clear()

    def apply_resolved(
        self,
        resolved: list[ResolvedHost],
        under_capacity: set[Platforms] | None = None,
        provisionable_ceiling: dict[Platforms, int] | None = None,
    ) -> None:
        """Reconcile the current state against the freshly-resolved host set by pure
        membership: a host in ``resolved`` is available; a host absent from it is
        unavailable; a host not yet known is added (available, depth 0). Rows kept for
        posterity; depth preserved.

        ``resolved`` is the set of good-to-use hosts — sources are responsible for
        confirming reachability before yielding a host — so the store does not judge
        liveness itself; it just tracks membership. This means a host a worker marked
        unavailable comes back the moment it re-appears in a resolution (i.e. once a
        source confirms it reachable again).

        ``under_capacity`` is a set of platforms that should not yield despite having
        hosts to prevent the small amount of hosts available from being flooded.

        ``provisionable_ceiling`` maps a platform to the cores of the largest host the fleet
        could ever provision for it; a claim exceeding it fails fast (see ``claim_least_busy``).
        Like ``under_capacity`` it is fleet metadata refreshed whole each cycle — passed
        ``None`` (a membership-only apply) leaves the existing map untouched.

        Controller-only: it reconciles against the WHOLE resolved set, so a worker (which
        sees only its own claims) running it would mark every host it didn't resolve
        unavailable."""

        def _apply_resolved_hosts(state: HostState, resolved: list[ResolvedHost]) -> HostState:
            """In-place membership reconcile backing ``apply_resolved`` (see there for
            semantics): availability follows presence in ``resolved``; new hosts are added.
            work_queue_depth is never touched here (owned by the workers).

            A host's ``num_physical_cores`` is refreshed from the resolution when the source
            reports a positive value (a re-probe can fill in a previously-unknown capacity), but a
            0/unknown report never clobbers an already-known count. Its multi-node deployment is
            refreshed on the same terms: a reported one wins, an unreported one leaves the stored
            one intact.
            """
            resolved_by_alias = {host.ssh_host: host for host in resolved}

            for rec in state.hosts:
                rec.available = rec.resolved.ssh_host in resolved_by_alias
                match = resolved_by_alias.get(rec.resolved.ssh_host)
                if match is not None:
                    _refresh_probed_fields(rec, match)

            # In resolved but not in state: add fresh (available, depth 0).
            _add_new_hosts(state, resolved)
            if under_capacity is not None:
                state.under_capacity = set(under_capacity)
            if provisionable_ceiling is not None:
                state.provisionable_ceiling = dict(provisionable_ceiling)
            return state

        self._update(lambda state: _apply_resolved_hosts(state, resolved))

    def set_poisoned(self, platform: Platforms, poisoned: bool) -> None:
        """Add or remove ``platform`` from the poisoned set. A poisoned platform
        throws an error when a claim is attempted.

        Controller-only: poisoning is the refresher's fleet-wide "this platform is
        perpetually empty" verdict, reached only after watching every source over many
        polls — not a call any single worker is positioned to make."""

        def _mutate(state: HostState) -> HostState | None:
            if (platform in state.poisoned) == poisoned:
                return None  # already in the desired state — skip the write
            if poisoned:
                state.poisoned.add(platform)
            else:
                state.poisoned.discard(platform)
            return state

        self._update(_mutate)


def _refresh_probed_fields(rec: HostRecord, probed: ResolvedHost) -> bool:
    """Refresh ``rec``'s capacity and multi-node deployment from ``probed`` (in place) and return
    whether anything changed. A positive core count replaces the stored one; a reported deployment
    replaces the stored one. A 0/unknown count or an unreported deployment leaves the stored value
    intact. Never touches platform, availability or depth."""
    current = rec.resolved
    updated = current
    if probed.num_physical_cores > 0:
        updated = replace(updated, num_physical_cores=probed.num_physical_cores)
    if probed.logical_host_id is not None:
        updated = replace(
            updated,
            logical_host_id=probed.logical_host_id,
            logical_host_rank=probed.logical_host_rank,
            logical_host_size=probed.logical_host_size,
        )
    rec.resolved = updated
    return updated != current


def _add_new_hosts(state: HostState, resolved: list[ResolvedHost]) -> HostState:
    """Append rows for aliases in ``resolved`` not already in ``state`` (in place),
    at depth 0.

    Purely additive: never touches an existing row's availability or depth. This
    is what makes it safe for the static multi-worker path, where every worker
    initializes the same file and must not reactivate a host another worker marked
    unavailable.
    """
    existing = {h.resolved.ssh_host for h in state.hosts}
    for host in resolved:
        if host.ssh_host not in existing:
            state.hosts.append(HostRecord(resolved=host))
    return state
