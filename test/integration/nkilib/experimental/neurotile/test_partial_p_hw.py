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

"""Hardware validation for a multi-P-tile trailing partial (element_shape=(300,256),
tile (128,128) => 3 P-tiles: 128 + 128 + 44).

A CPU-sim investigation showed the trailing P-tile's .data reporting (128,128)
instead of (44,128); the same descent on a pure Grid (outside the tracer) reported
(44,128) correctly, pointing at an AxisLabel enum-identity artifact of the sim
process rather than a Grid-logic bug. These tests decide it on-device:

  - _partial_p_roundtrip_kernel: numeric round-trip through a (300,256) partial-P
    SBUF staging alloc, iterating all 3 P-tiles incl. the 44-row trailing partial.
    Wrong trailing extent => wrong numbers or an out-of-bounds copy.
  - _partial_p_shape_assert_kernel: trace-time asserts pinning the trailing tile to
    (44,128)/is_remainder -- probes what the compile-path trace actually sees.
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

_P = 300  # 2 full P-tiles (128) + a 44-row trailing partial
_F = 256  # 2 F-tiles of 128


@nki.jit
def _partial_p_roundtrip_kernel(src):
    """Copy (300,256) HBM->HBM through a partial-P SBUF staging alloc.

    Each of the 3 P-tiles (incl. the 44-row trailing partial) is loaded, staged,
    and stored using the loaded tile's reported partition extent. If the trailing
    tile mis-reports 128 rows the load reads past the 300-row source.
    """
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    src_tiles = nt.tiles(src, tile_size=(128, 128))
    staging = nt.alloc_tiles(
        tile_size=(128, 128),
        buffer_type=nl.sbuf,
        dtype=src.dtype,
        element_shape=(_P, _F),
    )

    for pi in range(src_tiles.shape[0]):
        for fi in range(src_tiles.shape[1]):
            loaded = src_tiles[pi, fi].load()
            rows = loaded.data.shape[0]  # 44 on the trailing P-tile
            nisa.tensor_copy(staging[pi, fi].data[nl.ds(0, rows), :], loaded.data)
            nisa.dma_copy(
                dst[nl.ds(pi * 128, rows), nl.ds(fi * 128, 128)],
                staging[pi, fi].data[nl.ds(0, rows), :],
            )
    return dst


@nki.jit
def _partial_p_shape_assert_kernel(src):
    """Pin the trailing P-tile shape at trace time.

    Passes only if the compile-path trace resolves the trailing tile to its true
    44-row extent (matching the pure-Grid descent), which is the on-device answer
    to whether the sim's (128,128) reading was an environment artifact.
    """
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    staging = nt.alloc_tiles(
        tile_size=(128, 128),
        buffer_type=nl.sbuf,
        dtype=src.dtype,
        element_shape=(_P, _F),
    )
    assert staging.shape == (3, 2), f"grid expected (3,2), got {staging.shape}"

    for pi in range(2):
        full = staging[pi, 0]
        assert full.data.shape == (128, 128), f"P-tile [{pi},0] expected (128,128), got {full.data.shape}"

    # The addressable extent is the load-bearing fact and is consistent across
    # trace and compile paths. (is_remainder on a folded-P trailing tile is not
    # asserted here: it diverges by trace path -- False under nki.simulate, True
    # on the compile path -- so pinning it would make this kernel platform-bound.)
    rem = staging[2, 0]
    assert rem.element_shape == (44, 128), f"trailing P-tile expected (44,128), got {rem.element_shape}"
    assert rem.data.shape == (44, 128), f"trailing P-tile data expected (44,128), got {rem.data.shape}"

    nisa.dma_copy(dst, src)
    return dst


@nki.jit
def _partial_p_blocks_shape_assert_kernel(src):
    """alloc_blocks analogue: a block spanning the 3 P-tiles must resolve the
    trailing tile to (44,128) too -- the block path folds the partition axis the
    same way alloc_tiles does."""
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)

    blocks = nt.alloc_blocks(
        tile_size=(128, 128),
        block_size=(3, 1),
        buffer_type=nl.sbuf,
        dtype=src.dtype,
        element_shape=(_P, _F),
    )
    interior = blocks[0, 0]  # (3, 2) tile grid inside the block
    for pi in range(2):
        full = interior[pi, 0]
        assert full.data.shape == (128, 128), f"block P-tile [{pi},0] expected (128,128), got {full.data.shape}"

    rem = interior[2, 0]
    assert rem.element_shape == (44, 128), f"block trailing P-tile expected (44,128), got {rem.element_shape}"
    assert rem.data.shape == (44, 128), f"block trailing P-tile data expected (44,128), got {rem.data.shape}"

    nisa.dma_copy(dst, src)
    return dst


def _src_inputs(_):
    np.random.seed(42)
    return {"src": np.random.randn(_P, _F).astype(ml_dtypes.bfloat16)}


def _src_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["src"])}


def _passthrough_ref(src: torch.Tensor) -> torch.Tensor:
    return src.clone()


@pytest_marks(["neurotile"])
class TestPartialPartitionTrailing:
    """Multi-P-tile trailing partial (128+128+44) on hardware."""

    @pytest.mark.fast
    def test_partial_p_roundtrip(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_partial_p_roundtrip_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_src_inputs,
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_partial_p_shape_assert(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_partial_p_shape_assert_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_src_inputs,
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_partial_p_blocks_shape_assert(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_partial_p_blocks_shape_assert_kernel,
            torch_ref=torch_ref_wrapper(_passthrough_ref),
            kernel_input_generator=_src_inputs,
            output_tensor_descriptor=_src_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=1e-2,
            atol=1e-2,
        )
