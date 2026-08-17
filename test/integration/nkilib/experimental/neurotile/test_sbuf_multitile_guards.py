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

"""Device tests for SBUF interleaved/block sharding access contract.

Validates that slice-based sharding on a SBUF view addresses owned tiles
correctly under both LNC=1 and LNC=2:

  Single-tile access (after indexing the view, e.g. `view[0, i]`) works
  for both `.data` (ISA ops consume a contiguous tile) and `.data`
  (DMA store/load of one owned tile).

  Multi-tile `.data` on the sharded view emits a stacked AP pattern that
  walks only owned tiles; the DMA engine skips gap bytes.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental import neurotile as nt

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Module-level constants so the NKI tracer can see them.
H0 = 128
TS = 128
S_LNC1 = 512  # 4 slots, single-core sharding
S_LNC2 = 1024  # 8 slots, 4 per core under LNC=2

N_SLOTS_LNC1 = S_LNC1 // TS  # 4
N_LOCAL_LNC2 = (S_LNC2 // TS) // 2  # 4 owned slots per core

BLOCK_SIZE_F = 2  # tiles per block on F dim


# ---------------------------------------------------------------------------
# Single-tile .data: writes + stores to HBM work on LNC=1 sharded view
# ---------------------------------------------------------------------------


@nki.jit
def _lnc1_single_tile_kernel():
    """Per-tile writes via k_own[0, i].data on LNC=1 interleaved SBUF view."""
    out = nl.ndarray((H0, S_LNC1), dtype=nl.float32, buffer=nl.shared_hbm)
    k_full = nl.ndarray((H0, S_LNC1), dtype=nl.float32, buffer=nl.sbuf)

    total_tiles = nt.ceiling_div(S_LNC1, TS)
    own = nt.interleaved_range(rank=nl.program_id(0), num_shards=nl.num_programs(0), total=total_tiles)
    k_own = nt.tiles(k_full, tile_size=(H0, TS))[:, own]

    for i in range(N_SLOTS_LNC1):
        nisa.memset(k_own[0, i].data, float(i + 1))

    out_tiles = nt.tiles(out, tile_size=(H0, TS))
    for i in range(N_SLOTS_LNC1):
        out_tiles[0, i].store(k_own[0, i].data)
    return out


def _lnc1_single_tile_inputs(_):
    return {}


def _lnc1_single_tile_output(_):
    return {"out": np.zeros((H0, S_LNC1), dtype=np.float32)}


def _lnc1_single_tile_ref() -> torch.Tensor:
    """Each LNC1 slot i is filled with constant float(i+1)."""
    out = torch.zeros((H0, S_LNC1), dtype=torch.float32)
    for i in range(N_SLOTS_LNC1):
        out[:, i * TS : (i + 1) * TS] = float(i + 1)
    return out


# ---------------------------------------------------------------------------
# Single-tile .data: LNC=2 own + peer views address disjoint owned slots
# ---------------------------------------------------------------------------


@nki.jit
def _lnc2_own_view_kernel():
    """LNC=2 own view: each core writes its owned slots (even/odd)."""
    out = nl.ndarray((H0, S_LNC2), dtype=nl.float32, buffer=nl.shared_hbm)
    k_full = nl.ndarray((H0, S_LNC2), dtype=nl.float32, buffer=nl.sbuf)

    rank = nl.program_id(0)
    num_shards = nl.num_programs(0)
    total_tiles = nt.ceiling_div(S_LNC2, TS)

    own = nt.interleaved_range(rank=rank, num_shards=num_shards, total=total_tiles)
    k_own = nt.tiles(k_full, tile_size=(H0, TS))[:, own]

    for i in range(N_LOCAL_LNC2):
        nisa.memset(k_own[0, i].data, float(100 + i))

    out_own = nt.tiles(out, tile_size=(H0, TS))[:, own]
    for i in range(N_LOCAL_LNC2):
        out_own[0, i].store(k_own[0, i].data)
    return out


def _lnc2_own_inputs(_):
    return {}


def _lnc2_own_output(_):
    return {"out": np.zeros((H0, S_LNC2), dtype=np.float32)}


def _lnc2_own_ref() -> torch.Tensor:
    """LNC2 interleaved: slot g maps to local index g//2 on its rank."""
    out = torch.zeros((H0, S_LNC2), dtype=torch.float32)
    for g in range(S_LNC2 // TS):
        local_i = g // 2
        out[:, g * TS : (g + 1) * TS] = float(100 + local_i)
    return out


@nki.jit
def _lnc2_peer_view_kernel():
    """LNC=2 peer view: each core writes the OTHER core's owned slots."""
    out = nl.ndarray((H0, S_LNC2), dtype=nl.float32, buffer=nl.shared_hbm)
    k_full = nl.ndarray((H0, S_LNC2), dtype=nl.float32, buffer=nl.sbuf)

    rank = nl.program_id(0)
    num_shards = nl.num_programs(0)
    total_tiles = nt.ceiling_div(S_LNC2, TS)

    peer = nt.interleaved_range(rank=1 - rank, num_shards=num_shards, total=total_tiles)
    k_peer = nt.tiles(k_full, tile_size=(H0, TS))[:, peer]

    for i in range(N_LOCAL_LNC2):
        nisa.memset(k_peer[0, i].data, float(200 + i))

    out_peer = nt.tiles(out, tile_size=(H0, TS))[:, peer]
    for i in range(N_LOCAL_LNC2):
        out_peer[0, i].store(k_peer[0, i].data)
    return out


def _lnc2_peer_ref() -> torch.Tensor:
    out = torch.zeros((H0, S_LNC2), dtype=torch.float32)
    for g in range(S_LNC2 // TS):
        local_i = g // 2
        out[:, g * TS : (g + 1) * TS] = float(200 + local_i)
    return out


# ---------------------------------------------------------------------------
# .data on multi-tile sharded view: stacked pattern walks only owned tiles
# ---------------------------------------------------------------------------


@nki.jit
def _lnc2_multi_tile_ap_kernel():
    """Multi-tile .data on sharded SBUF view drives a correct strided DMA."""
    out = nl.ndarray((H0, S_LNC2), dtype=nl.float32, buffer=nl.shared_hbm)
    k_full = nl.ndarray((H0, S_LNC2), dtype=nl.float32, buffer=nl.sbuf)

    rank = nl.program_id(0)
    num_shards = nl.num_programs(0)
    total_tiles = nt.ceiling_div(S_LNC2, TS)

    own = nt.interleaved_range(rank=rank, num_shards=num_shards, total=total_tiles)
    k_own = nt.tiles(k_full, tile_size=(H0, TS))[:, own]

    for i in range(N_LOCAL_LNC2):
        nisa.memset(k_own[0, i].data, float(300 + i))

    out_own = nt.tiles(out, tile_size=(H0, TS))[:, own]
    out_own.store(k_own.data)
    return out


def _lnc2_multi_tile_ref() -> torch.Tensor:
    out = torch.zeros((H0, S_LNC2), dtype=torch.float32)
    for g in range(S_LNC2 // TS):
        local_i = g // 2
        out[:, g * TS : (g + 1) * TS] = float(300 + local_i)
    return out


# ---------------------------------------------------------------------------
# Block-granular SBUF shard: nt.blocks(block_size>1) + block_range on block dim
# ---------------------------------------------------------------------------


@nki.jit
def _lnc2_block_view_shard_kernel():
    """Block-granular sharded SBUF view: each rank owns contiguous block range."""
    out = nl.ndarray((H0, S_LNC2), dtype=nl.float32, buffer=nl.shared_hbm)
    k_full = nl.ndarray((H0, S_LNC2), dtype=nl.float32, buffer=nl.sbuf)

    rank = nl.program_id(0)
    num_shards = nl.num_programs(0)
    n_blocks_f = (S_LNC2 // TS) // BLOCK_SIZE_F
    own_blocks = nt.block_range(rank=rank, num_shards=num_shards, total=n_blocks_f)

    k_blocks = nt.blocks(
        k_full,
        tile_size=(H0, TS),
        block_size=(1, BLOCK_SIZE_F),
    )[:, own_blocks]

    k_tiles = nt.tiles(k_blocks)
    for i in range(N_LOCAL_LNC2):
        nisa.memset(k_tiles[0, i].data, float(500 + i))

    out_blocks = nt.blocks(
        out,
        tile_size=(H0, TS),
        block_size=(1, BLOCK_SIZE_F),
    )[:, own_blocks]
    out_tiles = nt.tiles(out_blocks)
    for i in range(N_LOCAL_LNC2):
        out_tiles[0, i].store(k_tiles[0, i].data)
    return out


def _lnc2_block_view_ref() -> torch.Tensor:
    out = torch.zeros((H0, S_LNC2), dtype=torch.float32)
    for g in range(S_LNC2 // TS):
        local_i = g % N_LOCAL_LNC2
        out[:, g * TS : (g + 1) * TS] = float(500 + local_i)
    return out


# ---------------------------------------------------------------------------
# Sliced SBUF source: nt.tiles accepts a sliced SBUF view directly
# (the slice self-addresses; no root= needed)
# ---------------------------------------------------------------------------


@nki.jit
def _slice_root_kernel():
    """Write distinct values to a sub-region of k_full via nt.tiles on a slice."""
    out = nl.ndarray((H0, 2 * S_LNC1), dtype=nl.float32, buffer=nl.shared_hbm)
    k_full = nl.ndarray((H0, 2 * S_LNC1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(k_full, 0.0)

    sub = k_full[:, 0:S_LNC1]
    sub_tiles = nt.tiles(sub, tile_size=(H0, TS))
    for i in range(N_SLOTS_LNC1):
        nisa.memset(sub_tiles[0, i].data, float(10 + i))

    out_tiles = nt.tiles(out, tile_size=(H0, TS))
    for i in range(2 * N_SLOTS_LNC1):
        out_tiles[0, i].store(nt.tiles(k_full, tile_size=(H0, TS))[0, i].data)
    return out


def _slice_root_inputs(_):
    return {}


def _slice_root_output(_):
    return {"out": np.zeros((H0, 2 * S_LNC1), dtype=np.float32)}


def _slice_root_ref() -> torch.Tensor:
    """Slice region [0:S_LNC1] gets float(10+i) per tile; remainder stays zero."""
    out = torch.zeros((H0, 2 * S_LNC1), dtype=torch.float32)
    for i in range(N_SLOTS_LNC1):
        out[:, i * TS : (i + 1) * TS] = float(10 + i)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest_marks(["neurotile"])
class TestSBUFMultiTileGuardsLNC1:
    """LNC=1 single-tile and slice+root tests."""

    @pytest.mark.fast
    def test_sbuf_sharded_single_tile_data_lnc1(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_lnc1_single_tile_kernel,
            torch_ref=torch_ref_wrapper(_lnc1_single_tile_ref),
            kernel_input_generator=_lnc1_single_tile_inputs,
            output_tensor_descriptor=_lnc1_single_tile_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-5,
            atol=1e-5,
        )

    @pytest.mark.fast
    def test_sbuf_slice_plus_root_addresses_slice_region(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_slice_root_kernel,
            torch_ref=torch_ref_wrapper(_slice_root_ref),
            kernel_input_generator=_slice_root_inputs,
            output_tensor_descriptor=_slice_root_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest_marks(["neurotile"])
class TestSBUFMultiTileGuardsLNC2:
    """LNC=2 own/peer/multi-tile/block-view sharding tests."""

    @pytest.mark.fast
    def test_sbuf_sharded_own_view_lnc2(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_lnc2_own_view_kernel,
            torch_ref=torch_ref_wrapper(_lnc2_own_ref),
            kernel_input_generator=_lnc2_own_inputs,
            output_tensor_descriptor=_lnc2_own_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_sbuf_sharded_peer_view_lnc2(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_lnc2_peer_view_kernel,
            torch_ref=torch_ref_wrapper(_lnc2_peer_ref),
            kernel_input_generator=_lnc2_own_inputs,
            output_tensor_descriptor=_lnc2_own_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_sbuf_sharded_multi_tile_ap_dma_lnc2(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_lnc2_multi_tile_ap_kernel,
            torch_ref=torch_ref_wrapper(_lnc2_multi_tile_ref),
            kernel_input_generator=_lnc2_own_inputs,
            output_tensor_descriptor=_lnc2_own_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_sbuf_block_view_shard_lnc2(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_lnc2_block_view_shard_kernel,
            torch_ref=torch_ref_wrapper(_lnc2_block_view_ref),
            kernel_input_generator=_lnc2_own_inputs,
            output_tensor_descriptor=_lnc2_own_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=1e-5,
            atol=1e-5,
        )
