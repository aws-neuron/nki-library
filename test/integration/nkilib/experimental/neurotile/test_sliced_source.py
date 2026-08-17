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

"""Integration tests for sliced HBM sources in neurotile.

Covers the case where source is a rank-reducing slice of a higher-rank
tensor (e.g. weights[expert] from a 3D tensor) passed directly to
nt.tiles() / nt.blocks(): the slice self-addresses, so the DMA strides
match the slice's rank, not the parent's rank.
"""

import ml_dtypes
import nki
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental import neurotile as nt

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Constants
# =============================================================================

E, K, N = 4, 512, 512
TILE_K, TILE_N = 128, 512
K_BLK = nt.largest_divisor(K // TILE_K, max_val=8)

B, H, S, D = 2, 4, 256, 128
TILE_S, TILE_D = 128, 128

# =============================================================================
# Kernels: leading-index slice with nt.tiles()
# =============================================================================


@nki.jit
def _kernel_tiles_leading_index_slice(weights):
    """Load a 2D slice weights[1] from a 3D tensor using nt.tiles (self-addressing)."""
    w_slice = weights[1]
    out = nl.ndarray((K, N), dtype=weights.dtype, buffer=nl.shared_hbm)
    w_tiles = nt.tiles(w_slice, tile_size=(TILE_K, TILE_N))
    out_tiles = nt.tiles(out, tile_size=(TILE_K, TILE_N))
    for i in range(w_tiles.shape[0]):
        for j in range(w_tiles.shape[1]):
            tile = w_tiles[i, j].load()
            out_tiles[i, j].store(tile.data)
    return out


@nki.jit
def _kernel_tiles_leading_index_slice_stream(weights):
    """Stream a 2D slice weights[2] from a 3D tensor using nt.tiles (self-addressing)."""
    w_slice = weights[2]
    out = nl.ndarray((K, N), dtype=weights.dtype, buffer=nl.shared_hbm)
    w_tiles = nt.tiles(w_slice, tile_size=(TILE_K, TILE_N))
    out_tiles = nt.tiles(out, tile_size=(TILE_K, TILE_N))
    for j in range(w_tiles.shape[1]):
        stream = w_tiles[:, j].stream(buffer_count=2)
        for i in nl.affine_range(w_tiles.shape[0]):
            tile = stream.load(i)
            out_tiles[i, j].store(tile.data)
    return out


# =============================================================================
# Kernels: leading-index slice with nt.blocks()
# =============================================================================


@nki.jit
def _kernel_blocks_leading_index_slice(weights):
    """Block-load a 2D slice weights[1] from a 3D tensor using nt.blocks (self-addressing)."""
    w_slice = weights[1]
    out = nl.ndarray((K, N), dtype=weights.dtype, buffer=nl.shared_hbm)
    w_blocks = nt.blocks(w_slice, tile_size=(TILE_K, TILE_N), block_size=(K_BLK, 1))
    out_blocks = nt.blocks(out, tile_size=(TILE_K, TILE_N), block_size=(K_BLK, 1))
    for bi in range(w_blocks.shape[0]):
        for bj in range(w_blocks.shape[1]):
            block = w_blocks[bi, bj].load()
            out_blocks[bi, bj].store(block.data)
    return out


@nki.jit
def _kernel_blocks_leading_index_slice_stream(weights):
    """Stream blocks from a 2D slice weights[0] of a 3D tensor."""
    w_slice = weights[0]
    out = nl.ndarray((K, N), dtype=weights.dtype, buffer=nl.shared_hbm)
    w_blocks = nt.blocks(w_slice, tile_size=(TILE_K, TILE_N), block_size=(K_BLK, 1))
    out_blocks = nt.blocks(out, tile_size=(TILE_K, TILE_N), block_size=(K_BLK, 1))
    stream = w_blocks[:, 0].stream(buffer_count=2)
    for bi in nl.affine_range(w_blocks.shape[0]):
        block = stream.load(bi)
        out_blocks[bi, 0].store(block.data)
    return out


# =============================================================================
# Kernels: 4D -> 2D/3D leading-index slice
# =============================================================================


@nki.jit
def _kernel_tiles_4d_double_leading_index(weights):
    """Load from weights[0, 1] -- a 2D slice of a 4D tensor."""
    w_slice = weights[0, 1]
    out = nl.ndarray((S, D), dtype=weights.dtype, buffer=nl.shared_hbm)
    w_tiles = nt.tiles(w_slice, tile_size=(TILE_S, TILE_D))
    out_tiles = nt.tiles(out, tile_size=(TILE_S, TILE_D))
    for i in range(w_tiles.shape[0]):
        for j in range(w_tiles.shape[1]):
            tile = w_tiles[i, j].load()
            out_tiles[i, j].store(tile.data)
    return out


@nki.jit
def _kernel_tiles_4d_single_leading_index(weights):
    """Load from weights[1] -- a 3D slice of a 4D [B,H,S,D] tensor, tiled as 3D.

    tile_size=(TILE_S, TILE_D) on a 3D source creates 1 batch dim (H).
    Batch dims are consumed first via indexing before tile-level iteration.
    """
    w_slice = weights[1]
    out = nl.ndarray((H, S, D), dtype=weights.dtype, buffer=nl.shared_hbm)
    w_tiles = nt.tiles(w_slice, tile_size=(TILE_S, TILE_D))
    out_tiles = nt.tiles(out, tile_size=(TILE_S, TILE_D))
    for h in range(H):
        w_slab = w_tiles[h]
        out_slab = out_tiles[h]
        for i in range(w_slab.shape[0]):
            for j in range(w_slab.shape[1]):
                tile = w_slab[i, j].load()
                out_slab[i, j].store(tile.data)
    return out


# =============================================================================
# Torch references
# =============================================================================


def _ref_expert_slice_1(weights: torch.Tensor) -> torch.Tensor:
    return weights[1].clone()


def _ref_expert_slice_2(weights: torch.Tensor) -> torch.Tensor:
    return weights[2].clone()


def _ref_expert_slice_0(weights: torch.Tensor) -> torch.Tensor:
    return weights[0].clone()


def _ref_4d_slice_0_1(weights: torch.Tensor) -> torch.Tensor:
    return weights[0, 1].clone()


def _ref_4d_slice_1(weights: torch.Tensor) -> torch.Tensor:
    return weights[1].clone()


# =============================================================================
# Input/output generators
# =============================================================================


def _weights_3d_inputs(_):
    np.random.seed(42)
    return {"weights": np.random.randn(E, K, N).astype(ml_dtypes.bfloat16)}


def _output_2d_KN(kernel_input):
    return {"out": np.zeros((K, N), dtype=kernel_input["weights"].dtype)}


def _weights_4d_inputs(_):
    np.random.seed(42)
    return {"weights": np.random.randn(B, H, S, D).astype(ml_dtypes.bfloat16)}


def _output_2d_SD(kernel_input):
    return {"out": np.zeros((S, D), dtype=kernel_input["weights"].dtype)}


def _output_3d_HSD(kernel_input):
    return {"out": np.zeros((H, S, D), dtype=kernel_input["weights"].dtype)}


# =============================================================================
# Tests
# =============================================================================


@pytest_marks(["neurotile"])
class TestSlicedSourceLeadingIndex:
    """Tests for leading-index slices (rank-reducing), tiled directly."""

    @pytest.mark.fast
    def test_tiles_3d_leading_index(self, test_manager: Orchestrator, platform_target: Platforms):
        """nt.tiles on weights[1] from [E, K, N], tiled directly."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_tiles_leading_index_slice,
            torch_ref=torch_ref_wrapper(_ref_expert_slice_1),
            kernel_input_generator=_weights_3d_inputs,
            output_tensor_descriptor=_output_2d_KN,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    def test_tiles_3d_leading_index_stream(self, test_manager: Orchestrator, platform_target: Platforms):
        """nt.tiles streaming on weights[2] from [E, K, N], tiled directly."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_tiles_leading_index_slice_stream,
            torch_ref=torch_ref_wrapper(_ref_expert_slice_2),
            kernel_input_generator=_weights_3d_inputs,
            output_tensor_descriptor=_output_2d_KN,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    def test_blocks_3d_leading_index(self, test_manager: Orchestrator, platform_target: Platforms):
        """nt.blocks on weights[1] from [E, K, N], tiled directly."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_blocks_leading_index_slice,
            torch_ref=torch_ref_wrapper(_ref_expert_slice_1),
            kernel_input_generator=_weights_3d_inputs,
            output_tensor_descriptor=_output_2d_KN,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    def test_blocks_3d_leading_index_stream(self, test_manager: Orchestrator, platform_target: Platforms):
        """nt.blocks streaming on weights[0] from [E, K, N], tiled directly."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_blocks_leading_index_slice_stream,
            torch_ref=torch_ref_wrapper(_ref_expert_slice_0),
            kernel_input_generator=_weights_3d_inputs,
            output_tensor_descriptor=_output_2d_KN,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    def test_tiles_4d_double_leading_index(self, test_manager: Orchestrator, platform_target: Platforms):
        """nt.tiles on weights[0, 1] from [B, H, S, D], tiled directly."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_tiles_4d_double_leading_index,
            torch_ref=torch_ref_wrapper(_ref_4d_slice_0_1),
            kernel_input_generator=_weights_4d_inputs,
            output_tensor_descriptor=_output_2d_SD,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    def test_tiles_4d_single_leading_index(self, test_manager: Orchestrator, platform_target: Platforms):
        """nt.tiles on weights[1] from [B, H, S, D], tiled directly -- 3D result."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_tiles_4d_single_leading_index,
            torch_ref=torch_ref_wrapper(_ref_4d_slice_1),
            kernel_input_generator=_weights_4d_inputs,
            output_tensor_descriptor=_output_3d_HSD,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0,
            atol=0,
        )
