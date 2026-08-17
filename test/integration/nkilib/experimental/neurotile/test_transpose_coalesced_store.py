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

"""Coalesced transpose-load paired with a single-DMA store, and per-tile
indexing on the preserved transpose grid.

A .load(transpose=True) over a tile-row is one coalesced DMA whose result keeps
its tile grid. These tests cover the two things you can do with that grid:

1. Store it back in ONE DMA. The transpose swapped axes on load, so the packed
   chunks lie along the SBUF free axis while their destinations are partition
   row-bands of out -- a plain store cannot express that, but a pattern_override
   remaps free chunk -> row-band, making the whole out-tile column one strided DMA.
2. Index each transposed chunk (packed[0, ni]) to run a per-tile compute op.
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


@nki.jit
def _xpose_coalesced_single_store(src):
    """Coalesced transpose-load (ONE DMA) + coalesced store (ONE DMA).

    packed.data[p, ni*128+q] == src[mi*128+q, ni*128+p] == out[ni*128+p, mi*128+q].
    out is row-major [N, M] (row stride M), so the SBUF read order (p, ni, q) maps
    to HBM levels [[M,128],[128*M,nc],[1,128]] at base out[0, mi*128] -- one DMA."""
    m, n = src.shape
    out = nl.ndarray((n, m), dtype=src.dtype, buffer=nl.shared_hbm)
    src_tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))
    nc = src_tiles.shape[1]
    for mi in range(src_tiles.shape[0]):
        packed = src_tiles[mi, :].load(transpose=True)  # ONE DMA -> (128, N), grid (1, nc)
        out_tiles[0, mi].store(packed.data, pattern_override=[[m, 128], [128 * m, nc], [1, 128]])
    return out


@nki.jit
def _xpose_coalesced_store_via_rearrange(src):
    """Same coalesced single-DMA store, but express the HBM destination regroup
    with the einops-style NDSlice.rearrange instead of reshape_dim+permute or a
    raw pattern_override.

    out_tiles[:, mi] is the full (nc*128, 128) column, indexed (ni p) q -- nc
    row-bands of 128, then the tile's 128 columns. Rearranging to p ni q makes
    the view iterate (p, ni, q), matching packed.data's (128, nc*128) =
    (p, [ni, q]) read order, so the whole column stores in ONE DMA."""
    m, n = src.shape
    out = nl.ndarray((n, m), dtype=src.dtype, buffer=nl.shared_hbm)
    src_tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))
    for mi in range(src_tiles.shape[0]):
        packed = src_tiles[mi, :].load(transpose=True)
        dst = out_tiles[:, mi].rearrange((("ni", "p"), "q"), ("p", "ni", "q"), {"p": 128})
        dst.store(packed.data)
    return out


@nki.jit
def _xpose_coalesced_scaled(src):
    """Per-tile indexing on the preserved transpose grid: coalesce the tile-row
    into ONE transpose-load DMA, then index each transposed chunk packed[0, ni]
    and scale it by (ni + 1) before storing -- one compute + store per chunk."""
    m, n = src.shape
    out = nl.ndarray((n, m), dtype=src.dtype, buffer=nl.shared_hbm)
    src_tiles = nt.tiles(src, tile_size=(128, 128))
    out_tiles = nt.tiles(out, tile_size=(128, 128))
    for mi in range(src_tiles.shape[0]):
        packed = src_tiles[mi, :].load(transpose=True)
        for ni in range(packed.shape[1]):
            chunk = packed[0, ni]  # per-tile index into the (1, nc) grid
            nisa.tensor_scalar(chunk.data, chunk.data, op0=nl.multiply, operand0=float(ni + 1))
            out_tiles[ni, mi].store(chunk.data)
    return out


def _single_store_ref(src: torch.Tensor) -> torch.Tensor:
    return src.t().contiguous()


def _scaled_ref(src: torch.Tensor) -> torch.Tensor:
    out = src.t().contiguous()  # [N, M]
    nc = src.shape[1] // 128
    for ni in range(nc):
        out[ni * 128 : (ni + 1) * 128, :] = out[ni * 128 : (ni + 1) * 128, :] * (ni + 1)
    return out


def _inputs_for(m, n):
    def _gen(_):
        np.random.seed(42)
        return {"src": np.random.randn(m, n).astype(ml_dtypes.bfloat16)}

    return _gen


def _out_for(m, n):
    def _out(_):
        return {"out": np.zeros((n, m), dtype=ml_dtypes.bfloat16)}

    return _out


@pytest_marks(["neurotile"])
class TestTransposeCoalescedStore:
    """A coalesced transpose-load can be stored in one DMA and indexed per-tile."""

    @pytest.mark.fast
    def test_transpose_coalesced_single_store(self, test_manager: Orchestrator, platform_target: Platforms):
        """(384,256) transpose-loaded per tile-row and stored back in ONE DMA == src.T.
        Shape differs from the other coalesced-store tests so concurrent compiles
        do not collide on the shared NKI intermediate cache."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_coalesced_single_store,
            torch_ref=torch_ref_wrapper(_single_store_ref),
            kernel_input_generator=_inputs_for(384, 256),
            output_tensor_descriptor=_out_for(384, 256),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_transpose_coalesced_store_via_rearrange(self, test_manager: Orchestrator, platform_target: Platforms):
        """(256,384) coalesced single-DMA store expressed via NDSlice.rearrange == src.T.
        Distinct shape from the example kernel so the two do not race on the
        shared NKI intermediate cache under parallel compilation."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_coalesced_store_via_rearrange,
            torch_ref=torch_ref_wrapper(_single_store_ref),
            kernel_input_generator=_inputs_for(256, 384),
            output_tensor_descriptor=_out_for(256, 384),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_transpose_coalesced_scaled(self, test_manager: Orchestrator, platform_target: Platforms):
        """(256,512) transpose-loaded, each transposed chunk ni scaled by (ni+1)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_xpose_coalesced_scaled,
            torch_ref=torch_ref_wrapper(_scaled_ref),
            kernel_input_generator=_inputs_for(256, 512),
            output_tensor_descriptor=_out_for(256, 512),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
