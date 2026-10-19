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
"""Load / compute / store over an extent that is not a multiple of the block.

One computation, three ways of sizing the SBUF buffer:

* ``symmetric_block`` -- sized from what ``load()`` returned, so both extents
  agree with no adjustment. The pattern to reach for.
* ``uniform_buffer_per_tile_store`` -- sized independently of the region (a
  whole-block accumulator), so it cannot match a partial block; stores per tile,
  reading the valid extent from the destination.
* ``output_stream`` -- ``BlockStream.store(index)``, where the slot's uniform
  size is the library's own and it narrows internally. Used to fail for *any*
  unaligned extent.

Validated numerically: a store whose element *count* matches but whose *order*
differs would pass every size assert, so the input is a position-dependent ramp
rather than a constant fill.
"""

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental import neurotile as nt

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator

_TILE_P = 128
_TILES_PER_BLK = 4


@nki.jit
def symmetric_block(src):
    """Buffer sized from the load; load and store extents agree on their own."""
    P, F = _TILE_P, src.shape[1]
    src_blks = nt.blocks(src, tile_size=(P, F), block_size=(_TILES_PER_BLK, 1))
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    out_blks = nt.blocks(out, tile_size=(P, F), block_size=(_TILES_PER_BLK, 1))

    for b in range(src_blks.shape[0]):
        loaded = src_blks[b, 0].load()
        res = nl.ndarray(tuple(loaded.data.shape), dtype=src.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(res, loaded.data, nl.multiply, 2.0)
        out_blks[b, 0].store(res)

    return out


@nki.jit
def uniform_buffer_per_tile_store(src):
    """Whole-block accumulator, uniform across blocks, stored a tile at a time.

    ``acc`` is the same size for every block, so it cannot match a partial one.
    The per-tile store sidesteps that, and ``valid_p`` is read from the
    destination tile -- never recomputed from the shape.
    """
    P, F = _TILE_P, src.shape[1]
    src_blks = nt.blocks(src, tile_size=(P, F), block_size=(_TILES_PER_BLK, 1))
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    out_blks = nt.blocks(out, tile_size=(P, F), block_size=(_TILES_PER_BLK, 1))

    for b in range(src_blks.shape[0]):
        loaded = src_blks[b, 0].load()
        acc = nt.alloc_tiles(tile_size=(P, F), grid=(_TILES_PER_BLK, 1), buffer_type=nl.sbuf, dtype=src.dtype)
        o_blk = out_blks[b, 0]
        for t in range(o_blk.shape[0]):
            hbm_tile = o_blk[t, 0]
            valid_p = hbm_tile.element_shape[0]
            nisa.tensor_scalar(acc[t, 0].data[0:valid_p, :], loaded[t, 0].data, nl.multiply, 2.0)
            hbm_tile.store(acc[t, 0].data[0:valid_p, :])

    return out


@nki.jit
def output_stream(src):
    """Output ``BlockStream``: the rotating slot is the library's own sizing.

    The two views are indexed by *their own* grids, which differ:
    ``src_blks[b, 0]`` is the HBM block's tile grid, while ``out_stream[b]`` is
    the slot's, whose P-tiles are packed along the free axis. Mixing them up
    silently leaves part of the slot unwritten, so the source tile count comes
    from the block view and the destination position from the slot.
    """
    P, F = _TILE_P, src.shape[1]
    src_blks = nt.blocks(src, tile_size=(P, F), block_size=(_TILES_PER_BLK, 1))
    out = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    out_blks = nt.blocks(out, tile_size=(P, F), block_size=(_TILES_PER_BLK, 1))

    out_stream = out_blks[:, 0].stream(buffer_count=2)

    for b in range(src_blks.shape[0]):
        block = src_blks[b, 0]
        loaded = block.load()
        slot = out_stream[b]
        for t in range(block.shape[0]):
            src_tile = loaded[t, 0]
            valid_p = src_tile.element_shape[0]
            nisa.tensor_scalar(slot[0, t].data[0:valid_p, :], src_tile.data, nl.multiply, 2.0)
        out_stream.store(b)

    return out


def scale_by_two_torch_ref(src):
    return {"out": src * 2.0}


def _shapes():
    """Aligned baselines plus the ways an extent can be unaligned.

    ``_TILES_PER_BLK * _TILE_P == 512`` elements per block on the partition axis.

    Deliberately absent: a trailing block holding a full tile *plus* a partial one
    (e.g. 704 rows = 5 whole tiles + 64). On the partition axis that region is not
    a rectangle, so no single access pattern can describe it, and NeuroTile rejects
    it at trace time rather than emitting a pattern that reads past the tensor.
    That rejection is covered in
    ``test/unit/.../test_unaligned_block_extents.py::TestRaggedTileRegionRejected``.
    """
    return [
        # Exactly 1 / 2 whole blocks -- proves no regression on the aligned path.
        pytest.param(512, 128, id="1blk_aligned"),
        pytest.param(1024, 128, id="2blk_aligned"),
        # Short on tile *count*: the trailing block holds whole tiles, just fewer.
        pytest.param(640, 128, id="1blk_plus_1tile"),
        pytest.param(896, 128, id="1blk_plus_3tiles"),
        # Short *within* a tile: the trailing block is a single partial tile.
        pytest.param(576, 128, id="1blk_plus_partial_tile"),
        # Smaller than a single block -- the whole grid is one partial block.
        pytest.param(128, 128, id="single_tile_only"),
        pytest.param(64, 128, id="sub_tile_only"),
        # A wider free axis, so the store is not a single contiguous run.
        pytest.param(576, 512, id="partial_tile_wide_F"),
    ]


@pytest_test_metadata(name="NeurotileUnalignedBlockStore")
class TestNeurotileUnalignedBlockStore:
    """All three buffer-sizing patterns, over aligned and unaligned extents."""

    def _run(self, test_manager, platform_target, kernel_entry, rows, cols):
        def input_generator(test_config):
            np.random.seed(42)
            # A position-dependent ramp: any mis-ordered or mis-offset store shows
            # up as a value mismatch, which a constant fill would hide. fp32 keeps
            # the ramp exact and finite after the x2 (the largest case reaches
            # 576 * 512 * 2, well past what fp16 can represent).
            ramp = np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)
            noise = np.random.randn(rows, cols).astype(np.float32)
            return {"src": (ramp + noise).astype(np.float32)}

        def output_tensors(kernel_input):
            return {"out": np.zeros_like(kernel_input["src"])}

        from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_entry,
            torch_ref=torch_ref_wrapper(scale_by_two_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            inference_args=InferenceArgs(),
        )

    @pytest_parametrize("rows, cols", _shapes())
    @pytest.mark.fast
    def test_symmetric_block(self, test_manager: Orchestrator, platform_target: Platforms, rows, cols):
        self._run(test_manager, platform_target, symmetric_block, rows, cols)

    @pytest_parametrize("rows, cols", _shapes())
    @pytest.mark.fast
    def test_uniform_buffer_per_tile_store(self, test_manager: Orchestrator, platform_target: Platforms, rows, cols):
        self._run(test_manager, platform_target, uniform_buffer_per_tile_store, rows, cols)

    @pytest_parametrize("rows, cols", _shapes())
    @pytest.mark.fast
    def test_output_stream(self, test_manager: Orchestrator, platform_target: Platforms, rows, cols):
        self._run(test_manager, platform_target, output_stream, rows, cols)
