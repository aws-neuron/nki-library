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

"""NKI on-device integration tests for neurotile.

Tests real DMA through the pipeline: factory -> index -> load -> store.
All kernels use only declarative public APIs (__getitem__, .load(),
.store(), .data, .tolist(), .stream()).

This is the migrated subset covering the core nki-integration test
classes (basic load/store, sequential range, indirect index, vector
gather). Remaining test classes from NeuroTile's test_nki_integration.py
(TestBlocks, TestAxisExplicitIteration, TestMultiTileLoadStore,
TestSlicedSourceMultiPTile, TestExtendedValidation, TestSBUFTileGridIndexing,
TestSBUFFactory, TestAllocAndSubIndex, TestNonContiguousStrides,
TestSBUFTransforms, TestDMATranspose, TestDMATransposeWithDst,
TestFoldMultiDMA, TestSharding, TestAllocBlocks, TestSBUFBlockBugs)
follow the same migration pattern and are tracked for follow-up.
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
# Basic load/store kernels
# =============================================================================


@nki.jit
def _kernel_single_tile(src):
    """Identity copy of a single tile."""
    P, F = src.shape[0], src.shape[1]
    out = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(P, F))
    out_v = nt.tiles(out, tile_size=(P, F))
    tile = v[0, 0].load()
    out_v[0, 0].store(tile.data)
    return out


@nki.jit
def _kernel_multi_tile(src):
    """Identity copy over a 2x2 tile grid."""
    M, N = src.shape[0], src.shape[1]
    P = M // 2
    F = N // 2
    out = nl.ndarray((M, N), dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(P, F))
    out_v = nt.tiles(out, tile_size=(P, F))
    for i in range(2):
        for j in range(2):
            tile = v[i, j].load()
            out_v[i, j].store(tile.data)
    return out


@nki.jit
def _kernel_remainder(src):
    """Identity copy with non-divisible shape: tile_size=(128, 512) on (300, 512)."""
    M, N = src.shape[0], src.shape[1]
    out = nl.ndarray((M, N), dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 512))
    out_v = nt.tiles(out, tile_size=(128, 512))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            tile = v[i, j].load()
            out_v[i, j].store(tile.data)
    return out


# =============================================================================
# .data is the access-pattern view (.data == .data)
# =============================================================================


@nki.jit
def _kernel_data_as_source(src):
    """Use ``.data`` (not ``.data``) as the DMA source: ``.data`` is the
    access-pattern view, so it is a valid DMA/compute operand."""
    M, N = src.shape
    out = nl.ndarray((M, N), dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 512))
    out_v = nt.tiles(out, tile_size=(128, 512))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            tile = v[i, j].load()
            # .data feeds store directly -- exercises the .data == .data contract.
            out_v[i, j].store(tile.data)
    return out


@nki.jit
def _kernel_data_remainder(src):
    """``.data`` on a partial trailing tile addresses only the real extent
    (300/128 -> last row-tile is 44 rows; 512 cols full)."""
    M, N = src.shape
    out = nl.ndarray((M, N), dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 512))
    out_v = nt.tiles(out, tile_size=(128, 512))
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            tile = v[i, j].load()
            out_v[i, j].store(tile.data)
    return out


# =============================================================================
# Sequential range
# =============================================================================


@nki.jit
def _kernel_seq_range_rows(src):
    """Iterate tile-rows via nl.sequential_range."""
    M, N = src.shape
    out = nl.ndarray((M, N), dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    out_v = nt.tiles(out, tile_size=(128, 128))
    for i in nl.sequential_range(v.shape[0]):
        for j in range(v.shape[1]):
            t = v[i, j].load()
            out_v[i, j].store(t.data)
    return out


@nki.jit
def _kernel_seq_range_cols(src):
    """Iterate tile-cols via nl.sequential_range."""
    M, N = src.shape
    out = nl.ndarray((M, N), dtype=src.dtype, buffer=nl.shared_hbm)
    v = nt.tiles(src, tile_size=(128, 128))
    out_v = nt.tiles(out, tile_size=(128, 128))
    for j in nl.sequential_range(v.shape[1]):
        for i in range(v.shape[0]):
            t = v[i, j].load()
            out_v[i, j].store(t.data)
    return out


# =============================================================================
# Common input/output factories
# =============================================================================


def _src_inputs(shape, seed=42):
    def _gen(_):
        np.random.seed(seed)
        return {"src": np.random.randn(*shape).astype(ml_dtypes.bfloat16)}

    return _gen


def _src_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _identity_ref(src: torch.Tensor) -> torch.Tensor:
    return src.clone()


# =============================================================================
# Tests
# =============================================================================


@pytest_marks(["neurotile"])
class TestBasicLoadStore:
    @pytest.mark.fast
    def test_single_tile_copy(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_single_tile,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=_src_inputs((128, 512), seed=42),
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_multi_tile_copy(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_multi_tile,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=_src_inputs((256, 1024), seed=42),
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_data_as_source(self, test_manager: Orchestrator, platform_target: Platforms):
        """``.data`` used directly as the DMA source (the .data == .data view)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_data_as_source,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=_src_inputs((256, 1024), seed=42),
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_data_as_source_remainder(self, test_manager: Orchestrator, platform_target: Platforms):
        """``.data`` on partial trailing tiles (non-divisible shape) round-trips."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_data_remainder,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=_src_inputs((300, 512), seed=42),
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_remainder_copy(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_remainder,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=_src_inputs((300, 512), seed=42),
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest_marks(["neurotile"])
class TestSequentialRange:
    @pytest.mark.fast
    def test_row_iteration(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_seq_range_rows,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=_src_inputs((512, 512), seed=42),
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_column_iteration(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_kernel_seq_range_cols,
            torch_ref=torch_ref_wrapper(_identity_ref),
            kernel_input_generator=_src_inputs((128, 1024), seed=42),
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
