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
"""Remote lock helper functions for atomic core locking.

This file is deployed to remote hosts and executed under flock for atomic
read-modify-write operations on locks.json. The functions are documented
and tested locally, ensuring correctness before deployment.

Communication protocol:
- Caller passes --result-file <path> argument
- Script writes JSON result to that file
- stdout/stderr are free for logging/debugging
- Caller reads the result file after script completes

Result format (written to result file):
    {"status": "ALLOCATED", "cores": [0, 1, 2, 3]}
    {"status": "NO_CORES"}
    {"status": "DRAINING"}
    {"status": "RELEASED"}
    {"status": "DRAINED", "max_lock_expiry": 1234567890}
    {"status": "UNDRAINED"}
    {"status": "ERROR", "message": "..."}

Exit code protocol:
    The script communicates status via exit codes to avoid stdout dependency.
    The caller reads the exit code to determine the result status, and only
    reads the result file when core IDs are needed (ALLOCATED).

    Exit codes (10+ to avoid collision with flock exit code 1):
        0  = ALLOCATED (result file contains core IDs)
        10 = NO_CORES
        11 = DRAINING
        12 = RELEASED
        13 = DRAINED
        14 = UNDRAINED
        99 = ERROR
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, NamedTuple

# Bump this whenever the helper's public verbs/API change (e.g. adding
# poll/probe/dequeue). The remote deploy is version-gated: a client only
# overwrites the deployed helper when its SCRIPT_VERSION is strictly newer than
# the version already on the host, so a stale client can never downgrade a
# newer deployed helper out from under concurrent newer clients.
SCRIPT_VERSION = 5
EVENTS_LOG_PREFIX = "events-"

# When this env var is truthy on the host, reconcile-driven grants emit a
# verbose RESERVE event (region cores + assembly time) so an offline pipeline
# can reconstruct idle-held and fragmentation metrics. Default off so the
# event log stays small in production; benchmarking turns it on.
DEBUG_EVENTS_ENV = "CORE_LOCK_DEBUG_EVENTS"

# Shared timing constants for the FIFO core-allocation queue. This helper is the
# single source of truth for these timings: because it is redeployed from this
# exact file each run, the client imports identical values via core_lock_client.
# COMMIT_WINDOW and STALE_THRESHOLD are derived from POLL_PERIOD so the
# derivation is encoded, not hard-coded.
POLL_PERIOD = 5
# Ceiling of the jitter added to each poll-loop sleep (random.uniform(0,
# POLL_JITTER_MAX) in host_management). The real inter-refresh gap is therefore
# POLL_PERIOD + up to POLL_JITTER_MAX (plus RPC latency), not POLL_PERIOD alone.
POLL_JITTER_MAX = 0.5
# A ready head gets >=1 poll inside its commit window (POLL_PERIOD < COMMIT_WINDOW).
COMMIT_WINDOW = 2 * POLL_PERIOD + 2
# Must exceed the poll loop's full retry budget *including jitter* so an
# actively-retrying client is never pruned out from under itself mid-budget.
# Budget is max_retryable_errors (10) refreshes spaced at POLL_PERIOD +
# POLL_JITTER_MAX, i.e. 10 * (5 + 0.5) = 55s; 12 * POLL_PERIOD = 60s clears it
# with headroom for RPC latency.
STALE_THRESHOLD = 12 * POLL_PERIOD
# Upper bound on branch-and-bound nodes the cooperative placement solver
# explores before falling back to a greedy (min-assembly) completion of the
# remaining entries. The contended large-request set is tiny in practice, so
# the solver terminates well under this cap; it only guards pathological queue
# depth from making a reconcile pass expensive.
SOLVER_NODE_CAP: int = 5000
# Max ready, region-less entries placed via recursive branch-and-bound in one
# reconcile pass. Beyond this, fall back to the iterative greedy completion so a
# very deep ready-queue degrades gracefully instead of exceeding the interpreter
# recursion limit. Kept well under sys.getrecursionlimit() (default 1000).
SOLVER_DEPTH_CAP = 256


class EventType(str, Enum):
    """Event-log event names.

    Subclasses ``str`` so members are wire-compatible: ``json.dumps`` serializes
    a member to its plain string value and ``EventType.COMMIT == "COMMIT"`` is
    True. Never use ``str(EventType.X)`` (yields ``"EventType.X"``); pass the
    member itself so the on-disk JSON ``event`` field stays the plain string.
    """

    ENQUEUE = "ENQUEUE"
    COMMIT = "COMMIT"
    DEQUEUE = "DEQUEUE"
    RELEASE = "RELEASE"
    PRUNE = "PRUNE"
    RESERVE = "RESERVE"
    RESERVE_EXPIRED = "RESERVE_EXPIRED"
    BUMP = "BUMP"
    DRAIN = "DRAIN"
    UNDRAIN = "UNDRAIN"
    SOLVER_OVERFLOW = "SOLVER_OVERFLOW"


class ReconcileEvent(NamedTuple):
    """A single reconcile-driven event for the caller to log.

    ``extra`` carries optional fields (e.g. the verbose RESERVE payload). It is
    read-only and never mutated by consumers.
    """

    event: EventType
    entry_id: str
    extra: dict[str, Any] = {}


class LockStatus(Enum):
    """Status codes returned by lock operations."""

    # (string_value, exit_code) — exit codes 10+ avoid collision with reserved unix exit codes
    ALLOCATED = ("ALLOCATED", 0)
    FLOCK_TIMEOUT = ("FLOCK_TIMEOUT", 1)
    NO_CORES = ("NO_CORES", 10)
    DRAINING = ("DRAINING", 11)
    RELEASED = ("RELEASED", 12)
    DRAINED = ("DRAINED", 13)
    UNDRAINED = ("UNDRAINED", 14)
    IN_QUEUE = ("IN_QUEUE", 15)
    ERROR = ("ERROR", 99)

    def __new__(cls, value: str, exit_code: int) -> LockStatus:
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, value: str, exit_code: int) -> None:
        self.exit_code = exit_code

    @classmethod
    def from_exit_code(cls, exit_code: int) -> LockStatus | None:
        """Look up a LockStatus by exit code, or None if not found."""
        for member in cls:
            if member.exit_code == exit_code:
                return member
        return None


@dataclass
class LockResult:
    """Result of a lock operation."""

    status: LockStatus
    cores: list[int] | None = None
    message: str | None = None
    max_lock_expiry: int | None = None
    expiry: int | None = None
    position: int | None = None
    # Relative seconds-from-now until the caller is estimated to acquire (not an
    # absolute timestamp): converted at the verb boundary via ``_relative`` so
    # the client compares it directly to its patience budget without depending
    # on client/server clock skew.
    worst_case_eta: int | None = None
    # Additive-optional bump signal: True when reconcile bumped THIS caller's
    # entry to the tail this cycle (drives sole-entry livelock detection that a
    # position delta cannot see -- a sole entry bumped 0->0 never moves).
    # Serialized only when True; absent/False defaults keep the wire backward
    # compatible (no locking-protocol version change).
    bumped: bool = False
    # Additive-optional re-enqueue signal: True when THIS caller's entry was
    # pruned this cycle (its last_seen_ts went stale, e.g. during a long
    # soft-join upload) and then transparently re-appended at the tail by the
    # Step-3 enqueue path -- silently losing FIFO position. Distinct from a bump
    # (entry present and moved) and from a first-ever enqueue (no prior PRUNE).
    # Serialized only when True; absent/False defaults keep the wire backward
    # compatible (no locking-protocol version change).
    re_enqueued: bool = False
    # Core reset can take up to ~7s of active inference time.
    # We want to opportunistically skip it to reduce inference time.
    # Due to NRT limitation we can only do so when the LNC config is the same across runs on all cores.
    should_reset_cores: bool = True

    def to_json(self) -> str:
        """Convert to JSON string for file output."""
        result: dict[str, Any] = {"status": self.status.value}
        if self.cores is not None:
            result["cores"] = self.cores
        if self.message is not None:
            result["message"] = self.message
        if self.max_lock_expiry is not None:
            result["max_lock_expiry"] = self.max_lock_expiry
        if self.expiry is not None:
            result["expiry"] = self.expiry
        if self.position is not None:
            result["position"] = self.position
        if self.worst_case_eta is not None:
            result["worst_case_eta"] = self.worst_case_eta
        if self.bumped:
            result["bumped"] = self.bumped
        if self.re_enqueued:
            result["re_enqueued"] = self.re_enqueued
        if not self.should_reset_cores:
            result["should_reset_cores"] = self.should_reset_cores
        return json.dumps(result)

    @classmethod
    def from_json(cls, json_str: str) -> LockResult:
        """Parse from JSON string."""
        data = json.loads(json_str)
        return cls(
            status=LockStatus[data["status"]],
            cores=data.get("cores"),
            message=data.get("message"),
            max_lock_expiry=data.get("max_lock_expiry"),
            expiry=data.get("expiry"),
            position=data.get("position"),
            worst_case_eta=data.get("worst_case_eta"),
            bumped=data.get("bumped", False),
            re_enqueued=data.get("re_enqueued", False),
            should_reset_cores=data.get("should_reset_cores", True),
        )


@dataclass
class QueueEntry:
    """A single waiter in the FIFO core-allocation queue.

    Persisted as part of LockState.queue. ``commit_deadline_ts`` is only set
    once an entry reaches the head and is granted a commit window; it is
    omitted from the serialized dict while None.
    """

    entry_id: str
    request_cores: int
    enqueued_ts: int
    last_seen_ts: int
    commit_deadline_ts: int | None = None
    reserved_region: list[int] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict (omits commit_deadline_ts when None)."""
        result: dict[str, Any] = {
            "entry_id": self.entry_id,
            "request_cores": self.request_cores,
            "enqueued_ts": self.enqueued_ts,
            "last_seen_ts": self.last_seen_ts,
        }
        if self.commit_deadline_ts is not None:
            result["commit_deadline_ts"] = self.commit_deadline_ts
        if self.reserved_region:
            result["reserved_region"] = self.reserved_region
        if self.ready:
            result["ready"] = self.ready
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QueueEntry:
        """Parse a queue entry from a JSON dict."""
        return cls(
            entry_id=data["entry_id"],
            request_cores=data["request_cores"],
            enqueued_ts=data["enqueued_ts"],
            last_seen_ts=data["last_seen_ts"],
            commit_deadline_ts=data.get("commit_deadline_ts"),
            reserved_region=data.get("reserved_region") or data.get("reserved_cores", []),
            ready=data.get("ready", False),
        )


@dataclass
class LockState:
    """Represents the state of the core locking system."""

    version: int
    draining_enabled_with_timeout: int | None  # Unix timestamp or None
    physical_neuron_cores_lock_timeout: dict[str, int]  # core_id -> expiry timestamp
    queue: list[QueueEntry] = field(default_factory=list)  # FIFO waiters
    last_lnc_config_per_core: dict[int, int] = field(default_factory=dict)

    @classmethod
    def empty(cls, version: int) -> LockState:
        """Create an empty lock state with given protocol version."""
        return cls(
            version=version,
            draining_enabled_with_timeout=None,
            physical_neuron_cores_lock_timeout={},
            last_lnc_config_per_core={},
            queue=[],
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], default_version: int) -> LockState:
        """Parse lock state from JSON dict."""
        return cls(
            version=data.get("version", default_version),
            draining_enabled_with_timeout=data.get("draining_enabled_with_timeout"),
            physical_neuron_cores_lock_timeout=data.get("physical_neuron_cores_lock_timeout", {}),
            queue=[QueueEntry.from_dict(e) for e in data.get("queue", [])],
            last_lnc_config_per_core={int(k): v for k, v in data.get("last_lnc_config_per_core", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert lock state to JSON-serializable dict."""
        result: dict[str, Any] = {
            "version": self.version,
            "physical_neuron_cores_lock_timeout": self.physical_neuron_cores_lock_timeout,
        }
        if self.draining_enabled_with_timeout is not None:
            result["draining_enabled_with_timeout"] = self.draining_enabled_with_timeout
        if self.queue:
            result["queue"] = [e.to_dict() for e in self.queue]
        if self.last_lnc_config_per_core:
            result["last_lnc_config_per_core"] = self.last_lnc_config_per_core
        return result

    def is_draining(self, now: int) -> bool:
        """Check if system is currently draining (blocking new acquisitions)."""
        if self.draining_enabled_with_timeout is None:
            return False
        return now < self.draining_enabled_with_timeout

    def get_available_cores(self, now: int, total_cores: int) -> list[int]:
        """Get list of available (unlocked or expired) core IDs."""
        available = []
        for core_id in range(total_cores):
            expiry = self.physical_neuron_cores_lock_timeout.get(str(core_id))
            if expiry is None or now >= expiry:
                available.append(core_id)
        return available

    def get_free_tiles(self, now: int, total_cores: int) -> list[int]:
        """Cores free of both live locks and any queue entry's reservation."""
        reserved = {c for e in self.queue for c in e.reserved_region}
        return sorted(set(self.get_available_cores(now, total_cores)) - reserved)

    def get_locked_cores(self, now: int) -> list[int]:
        """Get list of currently locked (non-expired) core IDs."""
        locked = []
        for core_id_str, expiry in self.physical_neuron_cores_lock_timeout.items():
            if now < expiry:
                locked.append(int(core_id_str))
        return locked

    def lock_cores(self, core_ids: list[int], expiry: int) -> None:
        """Lock the given cores with expiry timestamp."""
        for c in core_ids:
            self.physical_neuron_cores_lock_timeout[str(c)] = expiry

    def unlock_cores(self, core_ids: list[int]) -> None:
        """Unlock the given cores."""
        for c in core_ids:
            self.physical_neuron_cores_lock_timeout.pop(str(c), None)

    def set_draining(self, expiry: int) -> None:
        """Enable drain mode with expiry timestamp."""
        self.draining_enabled_with_timeout = expiry


def append_event(locks_file: str, event: str | EventType, caller_id: str | None = None, **fields: Any) -> None:
    """Append an event to the event log file.

    Events are written to daily log files (``events-YYYY-MM-DD.log``) stored
    alongside locks.json.  Each line is a self-contained JSON object with
    timestamp, event type, and additional fields.
    Must be called under flock for safe concurrent access.

    Transitions-only logging rule: events are emitted on actual
    state transitions (ENQUEUE/COMMIT/BUMP/DEQUEUE/PRUNE and ACQUIRE/RELEASE)
    only -- never per-poll or per-refresh. A polling caller that merely
    refreshes its ``last_seen_ts`` without changing state logs nothing. This
    bounds the size of the daily-rotated event log under heavy polling.

    Args:
        locks_file: Path to locks.json (event log is in the same directory)
        event: Event type (e.g., ACQUIRE, RELEASE, DRAIN, UNDRAIN)
        caller_id: Optional caller identifier (e.g., "gw7:test_qkv_tkg_sweep")
        **fields: Additional fields to include in the event

    """
    log_dir = str(Path(locks_file).parent)
    today = datetime.date.today().isoformat()
    events_file = f"{log_dir}/{EVENTS_LOG_PREFIX}{today}.log"
    entry: dict[str, Any] = {"ts": int(time.time()), "event": event}
    if caller_id:
        entry["caller"] = caller_id
    entry.update(fields)
    with open(events_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _debug_events_enabled() -> bool:
    """True when the host env opts into verbose RESERVE events."""
    return os.environ.get(DEBUG_EVENTS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _flush_events(locks_file: str, caller_id: str | None, events: list[ReconcileEvent]) -> None:
    """Transition-only logging of reconcile-driven events.

    Emits PRUNE/RESERVE_EXPIRED/BUMP (plus the verbose RESERVE event with its
    extra payload when debug is on). Must be called under flock.
    """
    for evt in events:
        # For reconcile-driven events the caller (caller_id) is the poller that
        # triggered this pass, not necessarily the owner of the affected entry;
        # the affected entry is identified by entry_id.
        append_event(locks_file, evt.event, caller_id, entry_id=evt.entry_id, **evt.extra)


def load_state(locks_file: str, default_version: int) -> LockState:
    """Load lock state from JSON file.

    If the file is missing or corrupted, returns an empty state.
    This provides fault tolerance - corrupted state resets to empty.

    Corruption->reset is a deliberately accepted tradeoff. Paired
    with the non-atomic ``save_state`` below, a torn write self-heals here on
    the next load by resetting to an empty queue (waiters simply re-enqueue).
    This is preferred over the added complexity of atomic rename under the
    existing flock; the cost is a rare fairness reset, never a deadlock.

    Args:
        locks_file: Path to locks.json
        default_version: Default version if file doesn't exist

    Returns:
        LockState object

    """
    try:
        with open(locks_file) as f:
            data = json.load(f)
            return LockState.from_dict(data, default_version)
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        # File missing or corrupted - return empty state
        return LockState.empty(default_version)


def save_state(locks_file: str, state: LockState) -> None:
    """Save lock state to JSON file.

    Intentionally a plain in-place write, not an atomic write-then-rename.
    A process killed mid-write can leave ``locks.json`` corrupt;
    that torn write self-heals on the next ``load_state``, which resets to an
    empty queue. The simplicity is preferred over atomic-rename complexity
    under the existing flock, accepting a rare best-effort fairness reset.

    Args:
        locks_file: Path to locks.json
        state: LockState object to save

    """
    with open(locks_file, "w") as f:
        json.dump(state.to_dict(), f)


def write_result(result_file: str, result: LockResult) -> None:
    """Write lock result to output file.

    Args:
        result_file: Path to write result JSON
        result: LockResult to write

    """
    Path(result_file).write_text(result.to_json())


def find_contiguous_cores(
    available: list[int],
    num_physical: int,
    total_cores: int,
) -> list[int] | None:
    """Find contiguous physical cores from available cores.

    Start positions are shuffled for load balancing to avoid always allocating
    from the same cores (which would cause uneven wear).

    Collectives require start positions aligned to num_physical.
    Since num_physical is always a multiple of lnc_config, aligning to
    num_physical automatically satisfies the lnc_config alignment requirement.

    Args:
        available: List of available core IDs
        num_physical: Number of physical cores needed
        total_cores: Total physical cores on the host

    Returns:
        List of contiguous physical core IDs, or None if not found

    """
    if len(available) < num_physical:
        return None

    available_set = set(available)

    # Generate start positions aligned to num_physical and shuffle for load balancing
    starts = list(range(0, total_cores - num_physical + 1, num_physical))
    random.shuffle(starts)

    for start in starts:
        block = list(range(start, start + num_physical))
        if all(c in available_set for c in block):
            return block

    return None


def _assert_reservation_invariant(state: LockState, now: int) -> None:
    """Reservations are pairwise disjoint among entries; a region MAY overlap the live locks it is waiting on.

    Guards the concurrent per-entry reservation model against a future change
    silently double-allocating the same physical core to two waiters.
    """
    all_reserved = [c for e in state.queue for c in e.reserved_region]
    if len(all_reserved) != len(set(all_reserved)):
        raise RuntimeError(f"reconcile invariant violated: overlapping reservations {all_reserved}")


def _build_free_at(state: LockState, now: int, total_cores: int) -> dict[int, int]:
    """Earliest-free time per core for cores eligible for NEW region placement.

    Cores already held by an entry's reserved_region are fixed and excluded.
    For each remaining core: now if free or its lock expired, else the live
    lock expiry. Pure: now is injected; no time.time() and no I/O.

    Args:
        state: Lock state to read (not mutated).
        now: Current unix timestamp (injected for determinism).
        total_cores: Total physical cores on the host.

    Returns:
        Mapping from placement-eligible core id to the timestamp at which it
        becomes free. Cores held by an entry's reserved_region are omitted.

    """
    held = {c for e in state.queue for c in e.reserved_region}
    free_at: dict[int, int] = {}
    for core_id in range(total_cores):
        if core_id in held:
            continue
        expiry = state.physical_neuron_cores_lock_timeout.get(str(core_id))
        free_at[core_id] = expiry if expiry is not None and expiry > now else now
    return free_at


def _candidate_blocks(
    k: int,
    free_at: dict[int, int],
    used: set[int],
    total_cores: int,
) -> list[tuple[list[int], int]]:
    """Enumerate aligned candidate blocks of size ``k`` for one request.

    Aligned offsets are multiples of ``k`` (the alignment NRT requires for a
    collective). Every core in a block must be placement-eligible (present in
    ``free_at``) and not already taken by an earlier pick (``used``). A block
    may include cores that are still locked: such a block is a *future* region
    that assembles once those locks expire. Pure: no clock or I/O.

    Args:
        k: Number of contiguous aligned cores the request needs.
        free_at: Earliest-free time per placement-eligible core.
        used: Cores already claimed by earlier picks this pass.
        total_cores: Total physical cores on the host.

    Returns:
        List of ``(block, assembly_time)`` where ``assembly_time`` is the time
        the whole block is free (``max free_at`` over the block); a fully-free
        block assembles at ``now``.

    """
    out: list[tuple[list[int], int]] = []
    if k <= 0:
        # Defensive: a non-positive request size has no valid aligned block and
        # would make range() step zero. Treat as unplaceable rather than crash
        # the whole reconcile pass for every poller on the host.
        return out
    for start in range(0, total_cores - k + 1, k):
        block = list(range(start, start + k))
        if all(c in free_at and c not in used for c in block):
            out.append((block, max(free_at[c] for c in block)))
    return out


def _solve_placement(
    reqs: list[int],
    free_at: dict[int, int],
    total_cores: int,
    now: int,
    node_cap: int = SOLVER_NODE_CAP,
    stats: dict[str, Any] | None = None,
) -> list[list[int] | None]:
    """Lexicographic-FIFO joint placement of ready, region-less entries.

    ``reqs`` is the per-entry ``request_cores`` of the ready, region-less
    waiters in FIFO order. Returns a list aligned with ``reqs`` of a chosen
    block (``list[int]``) or ``None``. Once an entry is unplaceable, it and all
    entries after it are ``None`` (FIFO no-overtake: an earlier waiter is never
    sacrificed to place a later one).

    Objective, minimized lexicographically: the per-entry assembly time (the
    ``max free_at`` over the chosen block). All fully-free blocks tie at
    ``now``, so the common case packs several FIFO-leading entries into disjoint
    free-now blocks at once (co-scheduling). Ties are broken by least idle-held
    (count of cores that are free now but sit idle inside a placed future block
    until it assembles), then by a wear-leveling shuffle among equal candidates.

    Bounded branch-and-bound DFS: candidates are explored earliest-assembly
    first and branches whose assembly prefix already loses to the incumbent are
    pruned. On exceeding ``node_cap`` the remaining entries are completed
    greedily (each takes its min-assembly disjoint block), guaranteeing a valid
    disjoint, aligned assignment. Pure: ``now`` injected, no ``time.time()`` and
    no I/O (the only nondeterminism is the wear-leveling shuffle).
    """
    n = len(reqs)
    # Sentinel assembly for an unplaced entry: dominates any real assembly so the
    # lexicographic comparison prefers placing earlier entries sooner, then
    # placing more entries.
    sentinel = now + 10**9

    def idle_of(block: list[int], assembly: int) -> int:
        # Cores free now held inside a future block sit idle until it assembles.
        if assembly <= now:
            return 0
        return sum(1 for c in block if free_at.get(c, now) <= now)

    def ordered_candidates(used: set[int], k: int) -> list[tuple[list[int], int]]:
        groups: dict[tuple[int, int], list[tuple[list[int], int]]] = {}
        for block, asm in _candidate_blocks(k, free_at, used, total_cores):
            groups.setdefault((asm, idle_of(block, asm)), []).append((block, asm))
        out: list[tuple[list[int], int]] = []
        for key in sorted(groups):
            grp = groups[key]
            random.shuffle(grp)  # wear-leveling among equal candidates
            out.extend(grp)
        return out

    best: dict[str, Any] = {"key": None, "assign": [None] * n}
    progress: dict[str, Any] = {"nodes": 0, "overflow": False}

    def consider(assign: list[list[int] | None], assembly: list[int], idle_total: int) -> None:
        key = (tuple(assembly), idle_total)
        if best["key"] is None or key < best["key"]:
            best["key"] = key
            best["assign"] = list(assign)

    def greedy_complete(
        i: int,
        used: set[int],
        assign: list[list[int] | None],
        assembly: list[int],
        idle_total: int,
    ) -> None:
        assign = list(assign)
        assembly = list(assembly)
        for j in range(i, n):
            cands = _candidate_blocks(reqs[j], free_at, used, total_cores)
            if not cands:
                assign.extend([None] * (n - j))
                assembly.extend([sentinel] * (n - j))
                consider(assign, assembly, idle_total)
                return
            block, asm = min(cands, key=lambda bc: (bc[1], idle_of(bc[0], bc[1]), bc[0][0]))
            assign.append(block)
            assembly.append(asm)
            idle_total += idle_of(block, asm)
            used = used | set(block)
        consider(assign, assembly, idle_total)

    def dfs(i: int, used: set[int], assign: list[list[int] | None], assembly: list[int], idle_total: int) -> None:
        if progress["overflow"]:
            return
        if i == n:
            consider(assign, assembly, idle_total)
            return
        # Prune: an assembly prefix already worse than the incumbent's cannot win.
        if best["key"] is not None and tuple(assembly) > best["key"][0][:i]:
            return
        candidates = ordered_candidates(used, reqs[i])
        if not candidates:
            # No eligible block: this entry and all behind it are unplaced.
            consider(assign + [None] * (n - i), assembly + [sentinel] * (n - i), idle_total)
            return
        for block, asm in candidates:
            progress["nodes"] += 1
            if progress["nodes"] > node_cap:
                progress["overflow"] = True
                greedy_complete(i, used, assign, assembly, idle_total)
                return
            dfs(
                i + 1,
                used | set(block),
                assign + [block],
                assembly + [asm],
                idle_total + idle_of(block, asm),
            )

    if n > SOLVER_DEPTH_CAP:
        # Too many entries for recursive B&B; complete greedily (iterative, FIFO,
        # min-assembly per entry) to stay within the interpreter recursion limit.
        greedy_complete(0, set(), [], [], 0)
        if stats is not None:
            stats["greedy_fallback"] = True
            stats["reason"] = "depth_cap"
            stats["queue_len"] = n
    else:
        dfs(0, set(), [], [], 0)
        if stats is not None:
            stats["greedy_fallback"] = progress["overflow"]
            stats["queue_len"] = n
            stats["nodes"] = progress["nodes"]
            stats["reason"] = "node_cap" if progress["overflow"] else None
    return best["assign"]


def reconcile(
    state: LockState,
    now: int,
    total_cores: int,
    *,
    draining: bool,
    debug: bool = False,
    node_cap: int = SOLVER_NODE_CAP,
) -> list[ReconcileEvent]:
    """Side-effect-free FIFO-queue reconcile pass with concurrent reservations.

    Mutates ``state`` in place (consistent with ``release``) but is pure with
    respect to wall-clock time and I/O: ``now`` is injected and no
    ``time.time()`` call or file access happens here. Given the same
    ``(state, now, total_cores, draining)`` it always produces the same mutation
    and the same ordered event list.

    Rather than a single head commit window, every ready waiter is placed
    cooperatively into its OWN disjoint aligned ``reserved_region`` in FIFO
    order. A region may include cores that are still locked (a *future* region
    that assembles once those locks expire, bounded by the lock cap); a fully
    free region opens its commit window immediately. A reserved entry commits
    its own block on a later poll; an entry that holds a reservation but never
    commits before its deadline has the tiles reclaimed.

    Steps, in order:
      1. Expire stale core locks (reusing the existing expiry predicate) and
         prune any queue entry not seen within ``STALE_THRESHOLD`` (emit
         ``PRUNE``). Pruning drops the entry's reservation automatically, since
         ``get_free_tiles`` unions over the surviving entries.

      (drain freeze, early return) While draining, preserve FIFO order and every
         reservation; only clear an already-expired ``commit_deadline_ts`` so a
         stale timer does not fire on undrain. No reclaim, bump, or packing.

      2. Reclaim expired reservations: an entry that holds ``reserved_region``
         whose ``commit_deadline_ts`` has passed releases its tiles (emit
         ``RESERVE_EXPIRED``) but keeps its FIFO position; packing may re-reserve
         it the same pass.
      3. Bump every not-ready entry whose one-shot readiness grace expired
         (emit ``BUMP``) so the queue cannot wedge behind stuck uploaders.
      4. Cooperative aligned-block placement: a joint solver assigns each ready,
         region-less entry its own disjoint aligned region (free-now or future),
         minimizing assembly time lexicographically in FIFO order. Held regions
         stay fixed; a held region opens its commit window the first pass it is
         fully assembled. A not-ready entry reserves nothing (transparent to
         backfill); each not-ready entry a hypothetical free-now allocation would
         place gets a one-shot readiness grace window. A large front entry takes
         a future region instead of blocking, so smaller entries behind backfill
         disjoint free cores; the only true barrier is capacity exhaustion (no
         eligible aligned block remains for an entry).

    Args:
        state: Lock state to reconcile (mutated in place).
        now: Current unix timestamp (injected for determinism).
        total_cores: Total physical cores on the host.
        draining: Whether the host is currently draining (freezes the queue).
        debug: When True, each first-time region assignment also appends a
            verbose ``RESERVE`` 3-tuple ``(event, entry_id, {cores, assembly_ts})``
            for offline analysis. Default off keeps the event list byte-for-byte
            unchanged. This flag is the only switch: ``reconcile`` never reads
            the environment, preserving its purity.

    Returns:
        Ordered list of ``ReconcileEvent`` for the caller to log (PRUNE,
        RESERVE_EXPIRED, BUMP, plus a verbose RESERVE carrying region cores and
        assembly time in ``extra`` when ``debug`` is True).

    """
    events: list[ReconcileEvent] = []

    # Step 1: expire stale core locks. Reuse the expiry predicate in
    # get_locked_cores (non-expired cores) instead of duplicating the
    # ``now >= expiry`` comparison; anything else in the dict has expired.
    locked_now = set(state.get_locked_cores(now))
    expired_core_ids = [int(c) for c in state.physical_neuron_cores_lock_timeout if int(c) not in locked_now]
    state.unlock_cores(expired_core_ids)

    # Step 1 (cont.): prune dead queue entries. Pruning drops an entry's
    # reserved_region automatically (get_free_tiles unions over the survivors).
    pruned_ids = {e.entry_id for e in state.queue if now - e.last_seen_ts > STALE_THRESHOLD}
    if pruned_ids:
        for entry in state.queue:
            if entry.entry_id in pruned_ids:
                events.append(ReconcileEvent(EventType.PRUNE, entry.entry_id))
        state.queue = [e for e in state.queue if e.entry_id not in pruned_ids]

    if draining:
        # Freeze: preserve FIFO order and every reservation across the drain.
        # Only clear an already-expired commit_deadline_ts so a stale timer does
        # not fire on undrain; do not bump, pack, or reclaim while draining.
        # Poll withholds all grants during a drain, so an entry cannot commit
        # through no fault of its own; freezing keeps every waiter's turn so
        # FIFO order is fully preserved across the drain.
        for e in state.queue:
            if e.commit_deadline_ts is not None and now > e.commit_deadline_ts:
                e.commit_deadline_ts = None
        _assert_reservation_invariant(state, now)
        return events

    # Step 2: reclaim expired reservations. An entry granted a reservation that
    # did not commit before its deadline (e.g. its client died) releases the
    # tiles but keeps its FIFO position; packing below may re-reserve it.
    for e in state.queue:
        if e.reserved_region and e.commit_deadline_ts is not None and now > e.commit_deadline_ts:
            e.reserved_region = []
            e.commit_deadline_ts = None
            events.append(ReconcileEvent(EventType.RESERVE_EXPIRED, e.entry_id))

    # Step 3: bump every not-ready entry whose readiness grace expired. Grace
    # timers are opened in Step 4 for each not-ready entry a hypothetical FIFO
    # allocation would place; an entry that misses its window loses its anchor
    # and moves to the tail so the queue cannot wedge behind stuck uploaders.
    # Survivors keep FIFO order; bumped entries are appended in their original
    # relative order.
    expired = [
        e for e in state.queue if (not e.ready) and e.commit_deadline_ts is not None and now > e.commit_deadline_ts
    ]
    if expired:
        expired_ids = {e.entry_id for e in expired}
        survivors = [e for e in state.queue if e.entry_id not in expired_ids]
        for e in expired:
            e.commit_deadline_ts = None
            events.append(ReconcileEvent(EventType.BUMP, e.entry_id))
        state.queue[:] = survivors + expired

    # Step 4: cooperative aligned-block placement. Held regions are fixed; the
    # solver only places ready, region-less entries. A placed region may include
    # still-locked cores (a future region) and assembles when they expire.
    free_at = _build_free_at(state, now, total_cores)
    locked_now = set(state.get_locked_cores(now))
    ready_regionless = [e for e in state.queue if e.ready and not e.reserved_region]
    solver_stats: dict[str, Any] = {}
    blocks = _solve_placement(
        [e.request_cores for e in ready_regionless],
        free_at,
        total_cores,
        now,
        node_cap=node_cap,
        stats=solver_stats,
    )
    assignment = {id(e): block for e, block in zip(ready_regionless, blocks, strict=True)}

    if debug and solver_stats.get("greedy_fallback"):
        # Pass-level diagnostic: the solver fell back to greedy completion (lost
        # B&B optimality). entry_id is empty because this is not tied to one
        # waiter; the detail is carried in extra.
        events.append(
            ReconcileEvent(
                EventType.SOLVER_OVERFLOW,
                "",
                {k: solver_stats[k] for k in ("reason", "queue_len", "nodes") if k in solver_stats},
            ),
        )

    # Free-now pool used only to decide which not-ready entries earn a readiness
    # grace window. It is independent of the solver's reservations; a placed
    # ready block (free-now or future) is subtracted as the FIFO scan reaches it
    # so a not-ready entry never counts cores a ready entry ahead already took.
    simulated_free = set(state.get_free_tiles(now, total_cores))
    for e in state.queue:
        if e.reserved_region:
            # Drop a stale held region so the entry re-solves cleanly: a
            # readiness downgrade (not-ready entries hold nothing) or a changed
            # request size (the held block no longer matches the requested core
            # count) invalidates the region. Falling through leaves the entry
            # region-less this pass; it is re-placed next pass (if still ready)
            # or handled by the W-lazy path below (if not ready).
            if not e.ready or len(e.reserved_region) != e.request_cores:
                e.reserved_region = []
                e.commit_deadline_ts = None
            else:
                # Held region is fixed. Open its commit window the first pass it is
                # fully assembled (no locked cores remain); a partial region keeps
                # commit_deadline_ts None so Step 2 never reclaims it.
                if e.commit_deadline_ts is None and set(e.reserved_region).isdisjoint(locked_now):
                    e.commit_deadline_ts = now + COMMIT_WINDOW
                continue
        if not e.ready:
            # W-lazy: a not-ready entry reserves nothing and is transparent to
            # backfill. If a free-now contiguous block would fit it in the
            # simulation, it is imminent: give it a one-shot readiness grace so
            # Step 3 can bump it if it never becomes ready.
            simulated_block = find_contiguous_cores(sorted(simulated_free), e.request_cores, total_cores)
            if simulated_block:
                if e.commit_deadline_ts is None:
                    e.commit_deadline_ts = now + COMMIT_WINDOW
                # this not-ready entry consumes simulated cores for entries behind it
                simulated_free -= set(simulated_block)
            continue
        # ready + region-less -> take the solver's block, if any. None means no
        # eligible aligned block this pass (capacity exhausted by earlier picks);
        # leave it unreserved but do NOT break -- not-ready entries behind it may
        # still earn a readiness grace.
        block = assignment.get(id(e))
        if block is None:
            continue
        e.reserved_region = block
        if debug:
            # Transition-only: emit the verbose grant event the single pass a
            # region is first assigned. Held regions skip this branch, so a
            # region never re-emits on a later pass.
            events.append(
                ReconcileEvent(
                    EventType.RESERVE,
                    e.entry_id,
                    {"cores": list(block), "assembly_ts": max(free_at[c] for c in block)},
                ),
            )
        if max(free_at[c] for c in block) <= now:
            # fully free now -> open the commit window immediately.
            e.commit_deadline_ts = now + COMMIT_WINDOW
        else:
            # partial (future) region -> no window until it fully assembles.
            e.commit_deadline_ts = None
        # a placed block leaves the readiness-grace pool for entries behind it
        simulated_free -= set(block)

    _assert_reservation_invariant(state, now)
    return events


def estimate_eta(
    state: LockState,
    request_cores: int,
    now: int,
    total_cores: int,
    hold_window: int,
) -> int:
    """Optimistic, best-effort worst-case ETA forward-simulation.

    Seeds a per-core "becomes free" timeline from
    ``state.physical_neuron_cores_lock_timeout`` (live locks) AND from every
    queue entry's ``reserved_region`` (committed-imminent blocks held for
    ``hold_window``), then walks the existing FIFO queue placing only the
    entries the caller actually waits behind, and finally returns the earliest
    time at which the caller's ``request_cores`` cores are free. Worst case =
    everyone the caller waits behind holds to their full window.

    Seeding/skip rules (matching how ``reconcile`` actually grants):
      - Live locks seed ``free_at`` from their expiry.
      - A queued entry's ``reserved_region`` is held until it fully assembles:
        every core is marked busy until ``max(free_at[c] for c in region) +
        hold_window`` (the region commits when its latest core frees, then holds
        for the window). For a free-now region this reduces to ``now +
        hold_window``; a future region defers the hold to its assembly time.
      - The FIFO walk SKIPS entries that already hold ``reserved_region`` (their
        block is in the seed -- placing them again would double-count) and
        entries with ``ready is False`` (W-lazy: a ready caller backfills past a
        not-ready entry, so it does not delay the caller). Only ready,
        unreserved entries are placed; each such entry is exactly a FIFO/barrier
        predecessor the caller must wait behind.

    Purity: ``now`` is injected and the function performs no ``time.time()``
    call and no I/O. ``state`` is treated as read-only -- the timeline is copied
    into a local list and only the copy is advanced. Deterministic given the
    same inputs.

    ``hold_window`` is a parameter, not a constant: the uniform per-acquire max
    hold is a hard cap and callers pass their own lock timeout.

    Two under-estimations are deliberately accepted,
    making the estimate optimistic:
      1. Fragmentation: placement uses a free-core *count* check, NOT aligned
         contiguous-block availability. A request needing a contiguous aligned
         block may actually wait longer than reported when enough *total* cores
         are free but no aligned block is. Justified by homogeneous-pool routing.
      2. Handoff latency: the per-handoff ``POLL_PERIOD`` gap between one entry
         releasing its cores and the next noticing/committing
         (``~position * POLL_PERIOD`` for a deep queue) is NOT modeled. This term
         is largely common-mode across hosts, so it does not distort *relative*
         host selection; it only makes the *absolute* ETA optimistic.

    Args:
        state: Lock state to simulate over (read-only; not mutated).
        request_cores: Number of cores the caller wants.
        now: Current unix timestamp (injected for determinism).
        total_cores: Total physical cores on the host.
        hold_window: Uniform per-acquire max hold in seconds (the lock timeout).

    Returns:
        Earliest unix timestamp at which ``request_cores`` cores are estimated
        to be free for the caller. Empty queue with enough free cores now
        returns ``now``.

    """
    if request_cores <= 0:
        return now

    # Seed a local copy of the timeline: when each physical core becomes free.
    # A core with no entry, or whose expiry is at/before ``now``, is free now.
    free_at: list[int] = []
    for core_id in range(total_cores):
        expiry = state.physical_neuron_cores_lock_timeout.get(str(core_id))
        free_at.append(expiry if expiry is not None and expiry > now else now)

    # Seed reservations as occupied. A queued entry commits only once its whole
    # region assembles (its latest core frees), then holds for hold_window, so
    # mark every reserved core busy until max(free_at over the region) +
    # hold_window. For a free-now region this is just now + hold_window; for a
    # future region (some cores still locked) it correctly defers the hold to
    # the assembly time. (A region MAY include still-locked cores it is waiting
    # on.)
    for entry in state.queue:
        # ETA-only clamp: tolerate a reserved_region read from older or foreign
        # host state whose core ids may exceed THIS host's total_cores. ETA is
        # advisory, so out-of-range ids are simply dropped here. _build_free_at
        # and reconcile deliberately do NOT clamp: they operate on the
        # authoritative current-host total_cores and a well-formed region.
        region = [c for c in entry.reserved_region if 0 <= c < total_cores]
        if not region:
            continue
        busy_until = max(free_at[c] for c in region) + hold_window
        for c in region:
            free_at[c] = max(free_at[c], busy_until)

    def earliest_free(k: int) -> int:
        # Earliest time at which k cores are simultaneously free, by COUNT: the
        # k-th smallest free time (at that instant >= k cores have become free).
        return sorted(free_at)[k - 1]

    def occupy(k: int, start: int) -> None:
        # Occupy the k earliest-available cores for hold_window from ``start``.
        order = sorted(range(len(free_at)), key=lambda i: free_at[i])
        for i in order[:k]:
            free_at[i] = start + hold_window

    # Walk queued entries in FIFO order, placing only the ones the ready caller
    # actually waits behind. Skip entries already holding reserved_region (in the
    # seed -- placing again would double-count) and not-ready entries (W-lazy:
    # the caller backfills past them). Only ready+unreserved entries are placed.
    for entry in state.queue:
        if entry.reserved_region:
            continue
        if not entry.ready:
            continue
        n = entry.request_cores
        if n <= 0 or n > total_cores:
            continue
        start = earliest_free(n)
        occupy(n, start)

    if request_cores > total_cores:
        # Cannot ever be satisfied on this host; best-effort optimism returns now.
        return now
    return earliest_free(request_cores)


def _relative(eta: int, now: int) -> int:
    """Convert an absolute ETA timestamp to a relative seconds-from-now wait.

    ``estimate_eta`` returns an absolute unix timestamp; the client compares
    ``worst_case_eta`` directly against a relative patience budget. Converting
    here, at the wire boundary, keeps client/server clock skew non-load-bearing
    (the client never has to subtract its own clock). Clamped at 0 so a stale
    past timestamp never surfaces as a negative wait.
    """
    return max(0, int(eta - now))


def poll(
    locks_file: str,
    total_cores: int,
    num_physical: int,
    timeout_seconds: int,
    version: int,
    entry_id: str,
    ready: bool,
    lnc_config: int,
    caller_id: str | None = None,
) -> LockResult:
    """Single mutating FIFO-queue verb: enqueue/refresh/commit.

    Thin composition of the pure helpers ``reconcile`` (FIFO maintenance) and
    ``estimate_eta`` (worst-case wait) plus ``acquire``'s allocation mechanics.
    ``now`` is read once here at the verb boundary; the pure helpers receive it
    as a parameter and stay clock-free.

    Flow:
      1. Load state, refresh THIS caller's queue entry (readiness, last_seen,
         request size) BEFORE running ``reconcile`` so cooperative placement uses
         fresh readiness, then run ``reconcile(..., draining=draining)``,
         collecting its transition events.
      2. Not yet queued: append a tail ``QueueEntry`` carrying its readiness
         (ENQUEUE) and return IN_QUEUE (or DRAINING while draining) with
         position + worst-case ETA. A new entry never commits same-exec
         (two-phase: ENQUEUE now, COMMIT on a later poll once reconcile has
         reserved it a block).
      3. Already queued: commit its OWN ``reserved_region`` block when NOT
         draining and ``ready`` -- at ANY position, since the reservation is
         disjoint and free by the reconcile invariant. A ``ready=false`` entry
         holds no reservation and never commits; no entry commits while
         draining. Otherwise return IN_QUEUE (or DRAINING).

    While draining the verb withholds all grants (no commit;
    reconcile skips window-opening) but preserves the queue and the caller's
    position, surfacing the distinct ``DRAINING`` status so the client can
    stay-and-probe. On undrain the head's next poll opens a
    window and commits in FIFO order.

    ``position`` is the 0-based index of the entry in the queue (head = 0).

    Args:
        locks_file: Path to locks.json
        total_cores: Total physical cores on the host
        num_physical: Number of contiguous physical cores requested
        timeout_seconds: Lock hold window (also the ETA hold_window)
        version: Lock protocol version
        entry_id: Stable per-attempt queue identity for this caller
        ready: Whether the caller is ready to commit (false never grabs cores)
        caller_id: Optional caller identifier for event logging

    Returns:
        LockResult: ALLOCATED (with cores/expiry) on commit, else
        IN_QUEUE with position + worst_case_eta (relative seconds from now).

    """
    # Read wall-clock once at the verb boundary; pure helpers receive it as a param.
    now = int(time.time())

    state = load_state(locks_file, version)
    draining = state.is_draining(now)

    # Refresh THIS caller's readiness and request size BEFORE reconcile so
    # cooperative placement uses fresh values. last_seen_ts is intentionally NOT
    # refreshed yet: reconcile must still see a stale timestamp so a long-absent
    # entry is pruned (and re-enqueued below with re_enqueued=True).
    existing = next((e for e in state.queue if e.entry_id == entry_id), None)
    if existing is not None:
        existing.ready = ready
        # Refresh the requested size so reconcile's packing, ETA, and commit all
        # use the current request rather than the stale enqueue-time value.
        existing.request_cores = num_physical

    # Step 1: FIFO maintenance (prune/reclaim/bump/pack). While draining,
    # reconcile freezes the queue so no grants are handed out. The env var is
    # read here at the verb boundary and passed in; reconcile stays pure.
    debug = _debug_events_enabled()
    reconcile_events = reconcile(state, now, total_cores, draining=draining, debug=debug)

    # Did reconcile bump THIS caller's entry to the tail this cycle? Derived from
    # the events reconcile already returns (no state re-scan). Surfaced on the
    # queued LockResult so the client can detect sole-entry bumps (0->0) that a
    # position delta is blind to.
    bumped_this_cycle = any(evt.event == EventType.BUMP and evt.entry_id == entry_id for evt in reconcile_events)

    # Did reconcile prune THIS caller's entry this cycle? If so and the entry is
    # absent below, the Step-3 enqueue re-appends it at the tail -- a silent FIFO
    # position loss (e.g. a long soft-join upload outlasting STALE_THRESHOLD).
    # Surfaced on the re-enqueue LockResult so the client can distinguish this
    # from a bump or a first-ever enqueue.
    re_enqueued_this_cycle = any(evt.event == EventType.PRUNE and evt.entry_id == entry_id for evt in reconcile_events)

    # Reconcile may have bumped this entry to the tail; re-find it. If it
    # survived (was not pruned as stale), refresh its last_seen_ts now (never
    # logged: transition-only logging) so it stays alive, including across a drain.
    existing = next((e for e in state.queue if e.entry_id == entry_id), None)
    if existing is not None:
        existing.last_seen_ts = now

    def _flush_reconcile_events() -> None:
        # Transition-only logging of reconcile-driven events. Must run under flock.
        _flush_events(locks_file, caller_id, reconcile_events)

    def _eta_for_position(position: int) -> int:
        # Worst-case ETA considers only the entries AHEAD of the caller.
        ahead = LockState(
            version=state.version,
            draining_enabled_with_timeout=state.draining_enabled_with_timeout,
            physical_neuron_cores_lock_timeout=state.physical_neuron_cores_lock_timeout,
            last_lnc_config_per_core=state.last_lnc_config_per_core,
            queue=state.queue[:position],
        )
        return estimate_eta(ahead, num_physical, now, total_cores, hold_window=timeout_seconds)

    def _lock_and_save(block: list[int], event: str | EventType) -> LockResult:
        # Reuse acquire's expiry/lock/append_event mechanics for commit.
        expiry = now + timeout_seconds
        state.lock_cores(block, expiry)
        # Skip reset only when every allocated core previously ran at this lnc_config.
        # Any LNC change (up or down) or first use (None) requires reset.
        should_reset = not all(state.last_lnc_config_per_core.get(core) == lnc_config for core in block)
        state.last_lnc_config_per_core.update(dict.fromkeys(block, lnc_config))
        save_state(locks_file, state)
        _flush_reconcile_events()
        append_event(locks_file, event, caller_id, cores=block, expiry=expiry)
        return LockResult(status=LockStatus.ALLOCATED, cores=block, expiry=expiry, should_reset_cores=should_reset)

    # Step 2: not yet queued -> enqueue a NEW tail entry WITH its readiness. A
    # new entry never commits same-exec (two-phase: ENQUEUE now, COMMIT on a
    # later poll once reconcile has reserved it a block).
    if existing is None:
        entry = QueueEntry(
            entry_id=entry_id,
            request_cores=num_physical,
            enqueued_ts=now,
            last_seen_ts=now,
            ready=ready,
        )
        state.queue.append(entry)
        position = len(state.queue) - 1
        eta = _eta_for_position(position)
        save_state(locks_file, state)
        _flush_reconcile_events()
        append_event(locks_file, EventType.ENQUEUE, caller_id, entry_id=entry_id, position=position)
        queued_status = LockStatus.DRAINING if draining else LockStatus.IN_QUEUE
        return LockResult(
            status=queued_status,
            position=position,
            worst_case_eta=_relative(eta, now),
            bumped=bumped_this_cycle,
            re_enqueued=re_enqueued_this_cycle,
        )

    # Step 3: already queued -> commit its OWN reserved block when ready and not
    # draining, but ONLY once the block is fully free (no core still locked). A
    # partial (future) region is held until its locked cores expire and it
    # assembles; committing it early would double-allocate a core another caller
    # still holds. A ready=false entry holds no reservation and never commits; no
    # entry commits while draining (reconcile granted none).
    if (
        not draining
        and existing.reserved_region
        and ready
        and set(existing.reserved_region).isdisjoint(state.get_locked_cores(now))
    ):
        block = list(existing.reserved_region)
        state.queue.remove(existing)
        return _lock_and_save(block, EventType.COMMIT)

    position = state.queue.index(existing)
    eta = _eta_for_position(position)
    save_state(locks_file, state)
    _flush_reconcile_events()
    queued_status = LockStatus.DRAINING if draining else LockStatus.IN_QUEUE
    return LockResult(
        status=queued_status,
        position=position,
        worst_case_eta=_relative(eta, now),
        bumped=bumped_this_cycle,
    )


def probe(
    locks_file: str,
    total_cores: int,
    num_physical: int,
    timeout_seconds: int,
    version: int,
) -> LockResult:
    """Strictly read-only worst-case ETA peek.

    Lets a caller peek a host's worst-case ETA *before* committing to enqueue,
    so an obviously-bad host can be skipped without enqueue/dequeue churn.

    Unlike ``poll`` this verb is purely read-only: it runs no ``reconcile``
    pass, never calls ``save_state``, never mutates ``state`` or any queue
    entry (no ``last_seen_ts`` refresh), and appends no event. Wall-clock is
    read once at the boundary; the pure ``estimate_eta`` receives it as a param.

    The caller is not in the queue, so every currently-queued entry is treated
    as ahead of it (worst case = they all hold to their full window).

    While draining the host hands out no cores, so probe surfaces
    the distinct ``DRAINING`` status (still carrying the worst-case ETA) so the
    client can stay-and-probe. probe remains strictly read-only either way: only
    the returned status changes, never any state.

    Args:
        locks_file: Path to locks.json
        total_cores: Total physical cores on the host
        num_physical: Number of contiguous physical cores the caller wants
        timeout_seconds: Lock hold window (also the ETA hold_window)
        version: Lock protocol version

    Returns:
        LockResult with DRAINING status while draining else IN_QUEUE, carrying
        ``worst_case_eta`` (relative seconds from now until estimated acquire).

    """
    # Read wall-clock once at the verb boundary; estimate_eta stays clock-free.
    now = int(time.time())
    state = load_state(locks_file, version)
    # The returned ETA is a read-only snapshot of the queue at this instant and
    # may be stale by the caller's next poll: entries ahead can
    # commit, release early, or be pruned in between. Accepted -- the caller
    # re-evaluates after enqueue, so probe staleness only affects host pre-screening.
    eta = estimate_eta(state, num_physical, now, total_cores, hold_window=timeout_seconds)
    # Read-only: only the surfaced status changes during a drain; no mutation.
    status = LockStatus.DRAINING if state.is_draining(now) else LockStatus.IN_QUEUE
    return LockResult(status=status, worst_case_eta=_relative(eta, now))


def dequeue(
    locks_file: str,
    total_cores: int,
    version: int,
    entry_id: str,
    caller_id: str | None = None,
) -> LockResult:
    """Graceful FIFO-queue removal for an abandoning caller.

    Used when a caller gives up on a host (patience rotation, or a test error
    before acquire). Removing the entry by ``entry_id`` is head-safe: dropping
    the head (along with any commit window it carried, since the whole entry is
    gone) leaves a consistent state, and the *next* reconcile/poll promotes the
    new head and opens its window. ``now`` is read once here at the verb
    boundary; the pure ``reconcile`` receives it as a param.

    Removing a non-existent ``entry_id`` is a no-op: reconcile still runs and
    state is saved, but no DEQUEUE event is logged for a missing entry.

    Args:
        locks_file: Path to locks.json
        total_cores: Total physical cores on the host
        version: Lock protocol version
        entry_id: Queue identity to remove
        caller_id: Optional caller identifier for event logging

    Returns:
        LockResult with RELEASED status.

    """
    # Read wall-clock once at the verb boundary; reconcile stays clock-free.
    now = int(time.time())

    state = load_state(locks_file, version)

    # FIFO maintenance (prune/bump/open-window) before removal. Derive drain
    # state the same single way poll does, so we never open a commit window
    # during a drain that poll will never grant. The env var is read here at the
    # verb boundary and passed in; reconcile stays pure.
    debug = _debug_events_enabled()
    reconcile_events = reconcile(state, now, total_cores, draining=state.is_draining(now), debug=debug)

    # Remove the abandoning entry, if present. Removing the head (and its
    # window) is safe: the next reconcile promotes + windows the new head.
    existing = next((e for e in state.queue if e.entry_id == entry_id), None)
    if existing is not None:
        state.queue = [e for e in state.queue if e.entry_id != entry_id]

    save_state(locks_file, state)

    # Transition-only logging: emit reconcile's PRUNE/BUMP (and verbose RESERVE
    # when debug is on), then DEQUEUE only when an entry was actually removed.
    _flush_events(locks_file, caller_id, reconcile_events)
    if existing is not None:
        append_event(locks_file, EventType.DEQUEUE, caller_id, entry_id=entry_id)

    return LockResult(status=LockStatus.RELEASED)


def release(
    locks_file: str,
    core_ids: list[int],
    version: int,
    caller_id: str | None = None,
    expected_expiry: int | None = None,
) -> LockResult:
    """Release previously acquired cores.

    If expected_expiry is provided, only cores whose current expiry matches
    will be released. Cores that have been reallocated (different expiry) are
    skipped to prevent corrupting another caller's lock.

    Args:
        locks_file: Path to locks.json
        core_ids: List of physical core IDs to release
        version: Lock protocol version
        caller_id: Optional caller identifier for event logging
        expected_expiry: Expiry timestamp from the original acquire, used to
            verify ownership before releasing.

    Returns:
        LockResult with RELEASED status

    """
    state = load_state(locks_file, version)

    if expected_expiry is not None:
        cores_to_unlock = []
        skipped_cores = []
        for core_id in core_ids:
            current_expiry = state.physical_neuron_cores_lock_timeout.get(str(core_id))
            if current_expiry == expected_expiry:
                cores_to_unlock.append(core_id)
            else:
                skipped_cores.append(core_id)
    else:
        cores_to_unlock = core_ids
        skipped_cores = []

    state.unlock_cores(cores_to_unlock)
    save_state(locks_file, state)

    if skipped_cores:
        append_event(locks_file, EventType.RELEASE, caller_id, cores=cores_to_unlock, skipped=skipped_cores)
    else:
        append_event(locks_file, EventType.RELEASE, caller_id, cores=cores_to_unlock)

    return LockResult(status=LockStatus.RELEASED)


def drain(locks_file: str, timeout_seconds: int, version: int) -> LockResult:
    """Enable drain mode to block new acquisitions.

    Args:
        locks_file: Path to locks.json
        timeout_seconds: How long drain should last after all existing locks expire
        version: Lock protocol version

    Returns:
        LockResult with DRAINED status

    """
    # Use local time on the inference host to avoid clock drift issues
    now = int(time.time())

    state = load_state(locks_file, version)

    # Compute max active lock expiry so caller knows how long to wait
    active_expiries = [v for v in state.physical_neuron_cores_lock_timeout.values() if v > now]
    max_lock_expiry = max(active_expiries) if active_expiries else now

    expiry = max_lock_expiry + timeout_seconds
    state.set_draining(expiry)
    save_state(locks_file, state)
    append_event(locks_file, EventType.DRAIN, expiry=expiry, max_lock_expiry=max_lock_expiry)

    return LockResult(status=LockStatus.DRAINED, max_lock_expiry=max_lock_expiry)


def undrain(locks_file: str, version: int) -> LockResult:
    """Disable drain mode to allow new acquisitions.

    Args:
        locks_file: Path to locks.json
        version: Lock protocol version

    Returns:
        LockResult with UNDRAINED status

    """
    state = load_state(locks_file, version)
    state.draining_enabled_with_timeout = None
    save_state(locks_file, state)
    append_event(locks_file, EventType.UNDRAIN)
    return LockResult(status=LockStatus.UNDRAINED)


def is_host_draining(locks_file: str, version: int) -> LockResult:
    """Check if the host is currently in drain mode.

    Read-only operation — does not modify lock state.

    Args:
        locks_file: Path to locks.json
        version: Lock protocol version

    Returns:
        LockResult with DRAINING if draining, UNDRAINED otherwise

    """
    now = int(time.time())
    state = load_state(locks_file, version)
    if state.is_draining(now):
        return LockResult(status=LockStatus.DRAINING)
    return LockResult(status=LockStatus.UNDRAINED)


def main(args: list[str]) -> LockResult:
    """CLI entry point for remote execution.

    Args:
        args: Command line arguments (excluding script name)

    Returns:
        LockResult from the command (also written to result file if specified)

    """
    parser = argparse.ArgumentParser(prog="lock_helpers.py", description="Core lock management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # poll subcommand
    p_poll = subparsers.add_parser("poll", help="Poll the FIFO queue (enqueue/refresh/commit)")
    p_poll.add_argument("--result-file", dest="result_file", help="Path to write JSON result")
    p_poll.add_argument("locks_file", help="Path to locks.json")
    p_poll.add_argument("total_cores", type=int, help="Total physical cores on host")
    p_poll.add_argument("num_physical", type=int, help="Number of cores requested")
    p_poll.add_argument("timeout_seconds", type=int, help="Lock timeout in seconds")
    p_poll.add_argument("version", type=int, help="Lock protocol version")
    p_poll.add_argument("entry_id", help="Stable per-attempt queue identity")
    p_poll.add_argument("--ready", dest="ready", action="store_true", help="Caller is ready to commit")
    p_poll.add_argument("--lnc-config", dest="lnc_config", type=int, default=2, help="LNC configuration (1 or 2)")
    p_poll.add_argument("--caller-id", dest="caller_id", help="Caller identifier for event logging")

    # probe subcommand
    p_probe = subparsers.add_parser("probe", help="Read-only worst-case ETA peek (no enqueue/mutation)")
    p_probe.add_argument("--result-file", dest="result_file", help="Path to write JSON result")
    p_probe.add_argument("locks_file", help="Path to locks.json")
    p_probe.add_argument("total_cores", type=int, help="Total physical cores on host")
    p_probe.add_argument("num_physical", type=int, help="Number of cores requested")
    p_probe.add_argument("timeout_seconds", type=int, help="Lock timeout in seconds")
    p_probe.add_argument("version", type=int, help="Lock protocol version")

    # dequeue subcommand
    p_dequeue = subparsers.add_parser("dequeue", help="Gracefully remove an entry from the FIFO queue")
    p_dequeue.add_argument("--result-file", dest="result_file", help="Path to write JSON result")
    p_dequeue.add_argument("locks_file", help="Path to locks.json")
    p_dequeue.add_argument("total_cores", type=int, help="Total physical cores on host")
    p_dequeue.add_argument("version", type=int, help="Lock protocol version")
    p_dequeue.add_argument("entry_id", help="Queue identity to remove")
    p_dequeue.add_argument("--caller-id", dest="caller_id", help="Caller identifier for event logging")

    # release subcommand
    p_release = subparsers.add_parser("release", help="Release cores")
    p_release.add_argument("--result-file", dest="result_file", help="Path to write JSON result")
    p_release.add_argument("locks_file", help="Path to locks.json")
    p_release.add_argument("version", type=int, help="Lock protocol version")
    p_release.add_argument("core_ids", type=int, nargs="+", help="Core IDs to release")
    p_release.add_argument("--caller-id", dest="caller_id", help="Caller identifier for event logging")
    p_release.add_argument(
        "--expected-expiry",
        dest="expected_expiry",
        type=int,
        help="Expected expiry timestamp for ownership verification",
    )

    # drain subcommand
    p_drain = subparsers.add_parser("drain", help="Enable drain mode")
    p_drain.add_argument("--result-file", dest="result_file", help="Path to write JSON result")
    p_drain.add_argument("locks_file", help="Path to locks.json")
    p_drain.add_argument("timeout_seconds", type=int, help="Drain timeout in seconds")
    p_drain.add_argument("version", type=int, help="Lock protocol version")

    # undrain subcommand
    p_undrain = subparsers.add_parser("undrain", help="Disable drain mode")
    p_undrain.add_argument("--result-file", dest="result_file", help="Path to write JSON result")
    p_undrain.add_argument("locks_file", help="Path to locks.json")
    p_undrain.add_argument("version", type=int, help="Lock protocol version")

    # check-draining subcommand
    p_check = subparsers.add_parser("check-draining", help="Check if host is draining")
    p_check.add_argument("--result-file", dest="result_file", help="Path to write JSON result")
    p_check.add_argument("locks_file", help="Path to locks.json")
    p_check.add_argument("version", type=int, help="Lock protocol version")

    try:
        parsed = parser.parse_args(args)
    except SystemExit:
        return LockResult(status=LockStatus.ERROR, message="Invalid arguments")

    try:
        if parsed.command == "poll":
            result = poll(
                parsed.locks_file,
                parsed.total_cores,
                parsed.num_physical,
                parsed.timeout_seconds,
                parsed.version,
                parsed.entry_id,
                parsed.ready,
                parsed.lnc_config,
                caller_id=parsed.caller_id,
            )
        elif parsed.command == "probe":
            result = probe(
                parsed.locks_file,
                parsed.total_cores,
                parsed.num_physical,
                parsed.timeout_seconds,
                parsed.version,
            )
        elif parsed.command == "dequeue":
            result = dequeue(
                parsed.locks_file,
                parsed.total_cores,
                parsed.version,
                parsed.entry_id,
                caller_id=parsed.caller_id,
            )
        elif parsed.command == "release":
            result = release(
                parsed.locks_file,
                parsed.core_ids,
                parsed.version,
                caller_id=parsed.caller_id,
                expected_expiry=parsed.expected_expiry,
            )
        elif parsed.command == "drain":
            result = drain(parsed.locks_file, parsed.timeout_seconds, parsed.version)
        elif parsed.command == "undrain":
            result = undrain(parsed.locks_file, parsed.version)
        elif parsed.command == "check-draining":
            result = is_host_draining(parsed.locks_file, parsed.version)
        else:
            result = LockResult(status=LockStatus.ERROR, message=f"Unknown command: {parsed.command}")
    except Exception as e:
        result = LockResult(status=LockStatus.ERROR, message=str(e))

    if parsed.result_file:
        write_result(parsed.result_file, result)

    return result


# Entry point for remote execution
if __name__ == "__main__":
    import sys

    sys.exit(main(sys.argv[1:]).status.exit_code)
