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

"""Distributed adapter: unified interface for rank/process-group resolution.

Provides get_rank() and get_pg() that work transparently across:
- Real torch.distributed via TorchDistAdapter (NCCL/gloo on hardware/GPU)
- Simulated environment via SimDistAdapter (CPU threading with SimProcessGroup)

Torch refs import get_rank/get_pg from here. They never need to know
which backend is active. Backend selection is automatic:
- If a SimDistAdapter is set on the current thread (by SimDistRunner), use it.
- Otherwise if torch.distributed is initialized, use TorchDistAdapter.
"""

import torch.distributed as dist
from nki.collectives import ReplicaGroup
from torch.distributed import ProcessGroup

# ==================== Adapter classes ====================


class DistributedAdapter:
    """Base interface for rank/process-group resolution."""

    def get_rank(self) -> int:
        raise NotImplementedError

    def get_pg(self, replica_group: ReplicaGroup) -> ProcessGroup:
        raise NotImplementedError


class TorchDistAdapter(DistributedAdapter):
    """Real torch.distributed backend (NCCL/gloo)."""

    def __init__(self):
        self._pg_cache: dict = {}

    def get_rank(self) -> int:
        return dist.get_rank()

    def get_pg(self, replica_group: ReplicaGroup) -> ProcessGroup:
        key = replica_group_key(replica_group)
        if key not in self._pg_cache:
            rank = dist.get_rank()
            for group_ranks in replica_group._value:
                pg = dist.new_group(ranks=list(group_ranks))
                if rank in group_ranks:
                    self._pg_cache[key] = pg
        return self._pg_cache[key]


class SimDistAdapter(DistributedAdapter):
    """Simulated backend using pre-built process group map."""

    def __init__(self, rank: int, pg_map: dict):
        self._rank = rank
        self._pg_map = pg_map

    def get_rank(self) -> int:
        return self._rank

    def get_pg(self, replica_group: ReplicaGroup) -> ProcessGroup:
        key = replica_group_key(replica_group)
        return self._pg_map[key]


# ==================== Dispatch ====================

_sim_adapter = None  # Set per-process by SimDistRunner before running torch_ref
_dist_backend = None


def _get_adapter() -> DistributedAdapter:
    """Get the active adapter for the current process."""
    if _sim_adapter is not None:
        return _sim_adapter
    if dist.is_initialized():
        global _dist_backend
        if _dist_backend is None:
            _dist_backend = TorchDistAdapter()
        return _dist_backend
    raise RuntimeError("No distributed backend: torch.distributed not initialized and no sim adapter set")


def set_adapter(adapter: DistributedAdapter):
    """Set the sim adapter for the current process.

    Called by SimDistRunner per-process before running a torch_ref.
    """
    global _sim_adapter
    _sim_adapter = adapter


# ==================== Public API ====================


def get_rank() -> int:
    """Get the current rank (like ncc.rank_id() on hardware)."""
    return _get_adapter().get_rank()


def get_pg(replica_group: ReplicaGroup) -> ProcessGroup:
    """Get the process group for a replica group (like ncc collective context on hardware)."""
    return _get_adapter().get_pg(replica_group)


def replica_group_key(replica_group: ReplicaGroup) -> tuple:
    """Convert a ReplicaGroup to a hashable key for pg_map lookup."""
    return tuple(tuple(g) for g in replica_group._value)
