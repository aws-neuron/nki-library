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
"""Test that BlockStream correctly handles a partial (remainder) last block.

When the total number of tiles is not divisible by the block_size, the last
block has fewer tiles. The stream must issue a correctly-sized DMA for that
block (not the full block size). This test would fail with:
    AssertionError: dma_copy requires src and dst to have the same number of elements
without the truncate_to_source fix in BlockStream._child / _bound_child.
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


@nki.jit
def stream_blocks_with_remainder(src):
    """Stream blocks with a partial last block.

    Tiles src into (128, 128) tiles and groups them into blocks of 16 along
    dim 0. When src rows aren't a multiple of 16*128=2048, the last block is
    partial. Exercises the BlockStream partial-block path.
    """
    P, F = 128, 128
    BLOCK_P = 16  # tiles per block along dim 0

    src_blocks = nt.blocks(src, tile_size=(P, F), block_size=(BLOCK_P, 1))
    dst = nl.ndarray(src.shape, dtype=src.dtype, buffer=nl.shared_hbm)
    dst_tiles = nt.tiles(dst, tile_size=(P, F))

    num_blocks = src_blocks.shape[0]
    # Stream along dim 0 (the block-row axis), select dim 1 = 0 (singleton)
    stream = src_blocks[:, 0].stream(buffer_count=2)

    global_tile = 0
    for block_idx in range(num_blocks):
        loaded = stream.load(block_idx)
        block = loaded[0]  # descend past the singleton block-column dim
        tiles_in_block = block.shape[0]
        for local_tile in range(tiles_in_block):
            tile = block[local_tile, 0]
            # Simple operation: scale by 2
            result = nl.ndarray((P, F), dtype=src.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(result, tile.data, nl.multiply, 2.0)
            dst_tiles[global_tile, 0].store(result)
            global_tile += 1

    return dst


def stream_blocks_with_remainder_torch_ref(src):
    """Torch reference: multiply by 2."""
    return {"out": src * 2.0}


@pytest_test_metadata(name="NeurotileBlockStreamRemainder")
class TestNeurotileBlockStreamRemainder:
    """Test that BlockStream handles partial last blocks correctly."""

    @pytest_parametrize(
        "rows, cols",
        [
            # 16 tiles along dim 0: exactly 1 full block, no remainder
            pytest.param(2048, 128, id="16tiles_aligned"),
            # 32 tiles: exactly 2 full blocks, no remainder
            pytest.param(4096, 128, id="32tiles_aligned"),
            # 17 tiles along dim 0: 1 full block (16) + 1 remainder tile
            pytest.param(2176, 128, id="17tiles_1rem"),
            # 19 tiles along dim 0: 1 full block (16) + 3 remainder tiles
            pytest.param(2432, 128, id="19tiles_3rem"),
            # 33 tiles: 2 full blocks (16+16) + 1 remainder
            pytest.param(4224, 128, id="33tiles_1rem"),
        ],
    )
    @pytest.mark.fast
    def test_stream_blocks_remainder(self, test_manager: Orchestrator, platform_target: Platforms, rows, cols):
        def input_generator(test_config):
            np.random.seed(42)
            return {"src": np.random.randn(rows, cols).astype(np.float16)}

        def output_tensors(kernel_input):
            return {"out": np.zeros_like(kernel_input["src"])}

        from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=stream_blocks_with_remainder,
            torch_ref=torch_ref_wrapper(stream_blocks_with_remainder_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            inference_args=InferenceArgs(),
        )
