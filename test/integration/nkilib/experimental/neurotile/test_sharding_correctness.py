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

"""End-to-end DMA correctness under sharding.

Pins per-core tile ownership for block + interleaved sharding patterns
against a hand-computed reference.
"""

import ml_dtypes
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

TILE_P = 128
TILE_F = 512


@nki.jit
def _write_shard_id_block(dummy):
    """Fill each owned tile with value = shard_id using block sharding."""
    M, N = 2048, TILE_F
    out = nl.ndarray((M, N), dtype=dummy.dtype, buffer=nl.shared_hbm)
    shard_id = nl.program_id(0)
    num_shards = nl.num_programs(0)

    own = nt.block_range(rank=shard_id, num_shards=num_shards, total=nt.ceiling_div(M, TILE_P))
    out_tiles = nt.tiles(out, tile_size=(TILE_P, N))[own, :]

    fill_sb = nl.ndarray((TILE_P, N), dtype=out.dtype, buffer=nl.sbuf)
    nisa.memset(fill_sb, value=0)
    nisa.tensor_scalar(dst=fill_sb, data=fill_sb, op0=nl.add, operand0=shard_id)
    fill_view = nt.tiles(fill_sb, tile_size=(TILE_P, N))

    for out_view in out_tiles.tolist():
        out_view.store(fill_view[0, 0].data)

    return out


@nki.jit
def _write_shard_id_interleaved(dummy):
    """Fill each owned tile with value = shard_id using interleaved sharding."""
    M, N = 2048, TILE_F
    out = nl.ndarray((M, N), dtype=dummy.dtype, buffer=nl.shared_hbm)
    shard_id = nl.program_id(0)
    num_shards = nl.num_programs(0)

    own = nt.interleaved_range(rank=shard_id, num_shards=num_shards, total=nt.ceiling_div(M, TILE_P))
    out_tiles = nt.tiles(out, tile_size=(TILE_P, N))[own, :]

    fill_sb = nl.ndarray((TILE_P, N), dtype=out.dtype, buffer=nl.sbuf)
    nisa.memset(fill_sb, value=0)
    nisa.tensor_scalar(dst=fill_sb, data=fill_sb, op0=nl.add, operand0=shard_id)
    fill_view = nt.tiles(fill_sb, tile_size=(TILE_P, N))

    for out_view in out_tiles.tolist():
        out_view.store(fill_view[0, 0].data)

    return out


@nki.jit
def _identity_block_sharded(src):
    """Load -> store with block sharding on single core -- owns everything."""
    P, F = src.shape[0], src.shape[1]
    out = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.shared_hbm)

    own = nt.block_range(rank=0, num_shards=1, total=nt.ceiling_div(P, TILE_P))
    src_tiles = nt.tiles(src, tile_size=(TILE_P, F))[own, :]
    out_tiles = nt.tiles(out, tile_size=(TILE_P, F))[own, :]

    for i in range(src_tiles.shape[0]):
        out_tiles[i].store(src_tiles[i].load().data)

    return out


def _dummy_inputs(_):
    return {"dummy": np.zeros((1,), dtype=ml_dtypes.bfloat16)}


def _shard_output(_):
    return {"out": np.zeros((2048, TILE_F), dtype=ml_dtypes.bfloat16)}


def _shard_id_block_ref(dummy: torch.Tensor) -> torch.Tensor:
    """LNC=2 block: rows [0:M/2) get 0, rows [M/2:M) get 1."""
    M, N = 2048, TILE_F
    half = M // 2
    out = torch.zeros((M, N), dtype=dummy.dtype)
    out[half:M, :] = 1.0
    return out


def _shard_id_interleaved_ref(dummy: torch.Tensor) -> torch.Tensor:
    """LNC=2 interleaved: tile g gets owner = g % 2."""
    M, N = 2048, TILE_F
    out = torch.zeros((M, N), dtype=dummy.dtype)
    for g in range(M // TILE_P):
        owner = g % 2
        out[g * TILE_P : (g + 1) * TILE_P, :] = float(owner)
    return out


def _identity_inputs(_):
    np.random.seed(42)
    return {"src": np.arange(512 * TILE_F, dtype=np.float32).reshape(512, TILE_F)}


def _identity_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _identity_ref(src: torch.Tensor) -> torch.Tensor:
    return src.clone()


@pytest_marks(["neurotile"])
class TestBlockShardMultiCore:
    """Block sharding across 2 cores must partition rows contiguously."""

    @pytest.mark.fast
    def test_half_half_partition(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_write_shard_id_block,
            torch_ref=torch_ref_wrapper(_shard_id_block_ref),
            kernel_input_generator=_dummy_inputs,
            output_tensor_descriptor=_shard_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=0.01,
            atol=0.01,
        )


@pytest_marks(["neurotile"])
class TestInterleavedShardMultiCore:
    """Interleaved sharding across 2 cores -- definitive fix check."""

    @pytest.mark.fast
    def test_stride_2_tile_ownership(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_write_shard_id_interleaved,
            torch_ref=torch_ref_wrapper(_shard_id_interleaved_ref),
            kernel_input_generator=_dummy_inputs,
            output_tensor_descriptor=_shard_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=0.01,
            atol=0.01,
        )


@pytest_marks(["neurotile"])
class TestBlockShardSingleCore:
    """Single-core block shard (num_shards=1) degenerates to identity."""

    @pytest.mark.fast
    def test_identity_copy(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_identity_block_sharded,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=_identity_inputs,
            output_tensor_descriptor=_identity_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=0.01,
            atol=0.01,
        )
