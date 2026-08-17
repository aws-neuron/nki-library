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

"""Sliced / transformed HBM sources tiled directly.

A modern NkiTensor self-describes its layout, so nt.tiles()/nt.blocks() tile a
sliced or transformed source directly. The previously-broken case was a
NONZERO-OFFSET slice (e.g. data[:, D/2:D]): the offset was applied twice (once in
the slice handle, once re-applied by the factory), walking past the slice extent
(exit 70 on a tiled load) or returning wrong rows (silently, on a gather). These
tests use arange fill so a wrong offset/stride yields visibly wrong data, and
assert exact equality against numpy.
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
# Constants -- M rows, N cols; the column slice halves at N/2.
# =============================================================================

M, N = 128, 256
HALF = N // 2
TILE = 128

# =============================================================================
# Kernels -- each transforms/slices the source and tiles it directly.
# =============================================================================


@nki.jit
def _kernel_offset_slice_no_root(src):
    """The fix target: tile src[:, HALF:N] (nonzero offset) with no root=."""
    out = nl.ndarray((M, HALF), dtype=src.dtype, buffer=nl.shared_hbm)
    view = src[:, HALF:N]
    src_tiles = nt.tiles(view, tile_size=(TILE, HALF))
    out_tiles = nt.tiles(out, tile_size=(TILE, HALF))
    out_tiles[0, 0].store(src_tiles[0, 0].load().data)
    return out


@nki.jit
def _kernel_offset_slice_multitile(src):
    """Multi-tile offset slice src[:, HALF:N] split into 2 F-tiles of 64 -- the
    stride+offset interaction must hold across more than one tile."""
    out = nl.ndarray((M, HALF), dtype=src.dtype, buffer=nl.shared_hbm)
    view = src[:, HALF:N]
    src_tiles = nt.tiles(view, tile_size=(TILE, 64))
    out_tiles = nt.tiles(out, tile_size=(TILE, 64))
    for j in range(src_tiles.shape[1]):
        out_tiles[0, j].store(src_tiles[0, j].load().data)
    return out


@nki.jit
def _kernel_partition_offset_slice(tall):
    """Offset on the partition dim: tall[64:192, :] (no root=). `tall` is
    (256, N) so the 128-row window starting at row 64 is in bounds."""
    out = nl.ndarray((128, N), dtype=tall.dtype, buffer=nl.shared_hbm)
    view = tall[64:192, :]
    src_tiles = nt.tiles(view, tile_size=(128, N))
    out_tiles = nt.tiles(out, tile_size=(128, N))
    out_tiles[0, 0].store(src_tiles[0, 0].load().data)
    return out


@nki.jit
def _kernel_reshape_no_root(src):
    """Regression: a contiguous reshape tiles with no root (already worked).

    (128, 256) -> (256, 128); tiled at P=128 so the 256-row result is 2 P-tiles
    (the partition dim caps at 128)."""
    out = nl.ndarray((N, M), dtype=src.dtype, buffer=nl.shared_hbm)
    view = src.reshape((N, M))
    src_tiles = nt.tiles(view, tile_size=(TILE, M))
    out_tiles = nt.tiles(out, tile_size=(TILE, M))
    for i in range(src_tiles.shape[0]):
        out_tiles[i, 0].store(src_tiles[i, 0].load().data)
    return out


# Gather guard: this case previously returned WRONG rows with NO error (the
# double-applied offset shifted the gather base by one slice width). Assert
# VALUES, not just "compiles".
G_K, G_TD = 16, HALF  # 16 gathered rows, full HALF-wide columns


@nki.jit
def _kernel_gather_offset_slice(data, indices):
    """Gather G_K rows of the offset slice data[:, HALF:N] (no root=)."""
    out = nl.ndarray((G_K, HALF), dtype=data.dtype, buffer=nl.shared_hbm)
    view = data[:, HALF:N]
    data_iter = nt.tiles(view, tile_size=(M, G_TD))
    idx_iter = nt.tiles(indices, tile_size=(G_K, 1))
    out_iter = nt.tiles(out, tile_size=(G_K, G_TD))
    idx_tile = idx_iter[0, 0].load()
    gathered = data_iter[idx_tile, 0].load()
    out_iter[0, 0].store(gathered.data)
    return out


def _ref_gather_offset_slice(data: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    sliced = data[:, HALF:N]
    idx = indices[:, 0].to(torch.int64)
    return sliced[idx].clone()


def _gather_inputs(_):
    np.random.seed(42)
    data = np.arange(M * N, dtype=np.float32).reshape(M, N).astype(ml_dtypes.bfloat16)
    # Distinct, non-identity row order so a wrong base is visibly wrong.
    rows = np.array([5, 0, 12, 3, 9, 1, 7, 2, 11, 4, 15, 6, 13, 8, 14, 10], dtype=np.int32)
    return {"data": data, "indices": rows.reshape(G_K, 1)}


def _out_gather(_):
    return {"out": np.zeros((G_K, HALF), dtype=ml_dtypes.bfloat16)}


# =============================================================================
# Torch references
# =============================================================================


def _ref_offset_slice(src: torch.Tensor) -> torch.Tensor:
    return src[:, HALF:N].clone()


def _ref_partition_offset_slice(tall: torch.Tensor) -> torch.Tensor:
    return tall[64:192, :].clone()


def _ref_reshape(src: torch.Tensor) -> torch.Tensor:
    return src.reshape(N, M).clone()


# =============================================================================
# Input / output generators -- arange so a wrong offset is visibly wrong.
# =============================================================================


def _arange_src(_):
    np.random.seed(42)
    return {"src": np.arange(M * N, dtype=np.float32).reshape(M, N).astype(ml_dtypes.bfloat16)}


def _arange_tall(_):
    """A (256, N) source so a partition-dim window at row 64 is in bounds."""
    np.random.seed(42)
    return {"tall": np.arange(256 * N, dtype=np.float32).reshape(256, N).astype(ml_dtypes.bfloat16)}


def _out_M_half(_):
    return {"out": np.zeros((M, HALF), dtype=ml_dtypes.bfloat16)}


def _out_128_N(_):
    return {"out": np.zeros((128, N), dtype=ml_dtypes.bfloat16)}


def _out_N_M(_):
    return {"out": np.zeros((N, M), dtype=ml_dtypes.bfloat16)}


# =============================================================================
# Tests
# =============================================================================


@pytest_marks(["neurotile"])
class TestTransformedSourceNoRoot:
    """Sliced / transformed sources tile correctly without root=."""

    @pytest.mark.fast
    def test_offset_slice_no_root(self, test_manager: Orchestrator, platform_target: Platforms):
        """src[:, HALF:N] tiled with no root= -- the fix target."""
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_offset_slice_no_root,
            torch_ref=torch_ref_wrapper(_ref_offset_slice),
            kernel_input_generator=_arange_src,
            output_tensor_descriptor=_out_M_half,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=0, atol=0)

    def test_offset_slice_multitile(self, test_manager: Orchestrator, platform_target: Platforms):
        """Offset slice across 2 F-tiles -- stride+offset interaction per tile."""
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_offset_slice_multitile,
            torch_ref=torch_ref_wrapper(_ref_offset_slice),
            kernel_input_generator=_arange_src,
            output_tensor_descriptor=_out_M_half,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=0, atol=0)

    def test_partition_offset_slice(self, test_manager: Orchestrator, platform_target: Platforms):
        """Offset on the partition dim: src[64:192, :] with no root=."""
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_partition_offset_slice,
            torch_ref=torch_ref_wrapper(_ref_partition_offset_slice),
            kernel_input_generator=_arange_tall,
            output_tensor_descriptor=_out_128_N,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=0, atol=0)

    @pytest.mark.fast
    def test_reshape_no_root_regression(self, test_manager: Orchestrator, platform_target: Platforms):
        """A contiguous reshape tiles with no root (regression -- already worked)."""
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_reshape_no_root,
            torch_ref=torch_ref_wrapper(_ref_reshape),
            kernel_input_generator=_arange_src,
            output_tensor_descriptor=_out_N_M,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=0, atol=0)

    def test_gather_offset_slice_values(self, test_manager: Orchestrator, platform_target: Platforms):
        """Gather on an offset-sliced source: previously returned WRONG rows with
        NO error. Asserts exact values (rtol=atol=0) against numpy."""
        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_gather_offset_slice,
            torch_ref=torch_ref_wrapper(_ref_gather_offset_slice),
            kernel_input_generator=_gather_inputs,
            output_tensor_descriptor=_out_gather,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=0, atol=0)
