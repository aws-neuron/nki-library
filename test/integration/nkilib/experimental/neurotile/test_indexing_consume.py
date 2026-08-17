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

"""Behavioral tests for the locked-in indexing model (kernel-dispatch tests).

CPU-only tests for indexing-consume rules live in
test/unit/.../test_indexing_validation.py. This file holds the on-device
kernels that exercise load/store with the cleaned-up indexing semantics.
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
def _kernel_load_tile_row(src):
    """Load a row of tiles via [i, :] then iterate the row's columns; scale by 2."""
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    ov = nt.tiles(out, tile_size=(128, 128))
    for i in range(v.shape[0]):
        row = v[i, :].load()
        for j in range(row.shape[0]):
            tile = row[j]
            nisa.tensor_scalar(tile.data, tile.data, nl.multiply, 2.0)
        ov[i, :].store(row.data)
    return out


@nki.jit
def _kernel_load_tile_column(src):
    """Load a column of tiles via [:, j]; scale by 3."""
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    ov = nt.tiles(out, tile_size=(128, 128))
    for j in range(v.shape[1]):
        col = v[:, j].load()
        for i in range(col.shape[0]):
            nisa.tensor_scalar(col[i].data, col[i].data, nl.multiply, 3.0)
        ov[:, j].store(col.data)
    return out


@nki.jit
def _kernel_load_block_row(src):
    """Load a row of blocks via [bi, :]; scale by 2."""
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.blocks(src, tile_size=(128, 128), block_size=(2, 2))
    ov = nt.blocks(out, tile_size=(128, 128), block_size=(2, 2))
    for bi in range(v.shape[0]):
        block_row = v[bi, :].load()
        for bj in range(block_row.shape[0]):
            block = block_row[bj]
            for ti in range(block.shape[0]):
                for tj in range(block.shape[1]):
                    nisa.tensor_scalar(block[ti, tj].data, block[ti, tj].data, nl.multiply, 2.0)
        ov[bi, :].store(block_row.data)
    return out


@nki.jit
def _kernel_load_subgrid(src):
    """Pass-through with inner 2x2 sub-grid scaled 4x."""
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    ov = nt.tiles(out, tile_size=(128, 128))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            t = v[i, j].load()
            ov[i, j].store(t.data)
    sub = v[1:3, 1:3].load()
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            nisa.tensor_scalar(sub[i, j].data, sub[i, j].data, nl.multiply, 4.0)
    ov[1:3, 1:3].store(sub.data)
    return out


@nki.jit
def _kernel_3d_subtile_slice(src):
    """Per-F1-slice scaling on a 3D loaded tile; F1=0 by 2, F1=1 by 3."""
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 2, 32))
    ov = nt.tiles(out, tile_size=(128, 2, 32))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                tile = v[i, j, k].load()
                slice0 = tile[:, 0, :]
                slice1 = tile[:, 1, :]
                nisa.tensor_scalar(slice0.data, slice0.data, nl.multiply, 2.0)
                nisa.tensor_scalar(slice1.data, slice1.data, nl.multiply, 3.0)
                ov[i, j, k].store(tile.data)
    return out


def _src_512_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(512, 512).astype(ml_dtypes.bfloat16)}


def _src_512_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _src_1024_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(1024, 512).astype(ml_dtypes.bfloat16)}


def _src_1024_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _src_3d_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(256, 4, 64).astype(ml_dtypes.bfloat16)}


def _src_3d_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _scale_2_ref(src: torch.Tensor) -> torch.Tensor:
    return src * 2.0


def _scale_3_ref(src: torch.Tensor) -> torch.Tensor:
    return src * 3.0


def _subgrid_ref(src: torch.Tensor) -> torch.Tensor:
    out = src.clone()
    out[128:384, 128:384] *= 4.0
    return out


def _subtile_3d_ref(src: torch.Tensor) -> torch.Tensor:
    out = src.clone()
    out[:, ::2, :] *= 2.0
    out[:, 1::2, :] *= 3.0
    return out


@pytest_marks(["neurotile"])
class TestLoadPreservesGrid:
    """Load through indexing preserves iteration grid + per-tile compute."""

    @pytest.mark.fast
    def test_load_tile_row(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_load_tile_row,
            torch_ref=torch_ref_wrapper(_scale_2_ref),
            kernel_input_generator=_src_512_inputs,
            output_tensor_descriptor=_src_512_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_load_tile_column(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_load_tile_column,
            torch_ref=torch_ref_wrapper(_scale_3_ref),
            kernel_input_generator=_src_512_inputs,
            output_tensor_descriptor=_src_512_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_load_block_row(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_load_block_row,
            torch_ref=torch_ref_wrapper(_scale_2_ref),
            kernel_input_generator=_src_1024_inputs,
            output_tensor_descriptor=_src_1024_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_load_subgrid(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_load_subgrid,
            torch_ref=torch_ref_wrapper(_subgrid_ref),
            kernel_input_generator=_src_512_inputs,
            output_tensor_descriptor=_src_512_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest_marks(["neurotile"])
class TestSubTileSliceOnHigherRankTile:
    """Sub-tile slicing on N-D SBUF source: tile[:, f1, :] flattens correctly."""

    @pytest.mark.fast
    def test_3d_subtile_per_slice_scale(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_3d_subtile_slice,
            torch_ref=torch_ref_wrapper(_subtile_3d_ref),
            kernel_input_generator=_src_3d_inputs,
            output_tensor_descriptor=_src_3d_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
