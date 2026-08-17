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

"""Integration tests for rotation_matrix utility.

Tests build_rotation_matrix for circular shift correctness across various
block_size, num_blocks, shifts, and negate configurations. Validates both
unsigned permutations (topk-style) and signed permutations (RoPE rotate-half).
"""

from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.rotation_matrix import build_rotation_matrix

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Test Kernel
# =============================================================================


@nki.jit
def kernel_rotation_matrix(
    input_data,
    output,
    block_size: int,
    num_blocks: int,
    shifts: int,
    negate: bool,
    use_scalar_engine: bool,
):
    """Test kernel that builds rotation matrix, applies it, and returns result."""
    N = block_size * num_blocks
    free_dim = input_data.shape[1]

    data = nl.ndarray((N, free_dim), dtype=input_data.dtype, buffer=nl.sbuf)
    nisa.dma_copy(data, input_data)

    engine = nisa.scalar_engine if use_scalar_engine else nisa.vector_engine
    rot = build_rotation_matrix(block_size, num_blocks, shifts, nl.float16, negate=negate, engine=engine)

    # Apply via nc_matmul
    f_max = nl.tile_size.gemm_moving_fmax
    n_tiles = (free_dim + f_max - 1) // f_max
    result = nl.ndarray((N, free_dim), dtype=nl.float32, buffer=nl.sbuf)
    for tile_idx in nl.affine_range(n_tiles):
        tile_size = min(f_max, free_dim - tile_idx * f_max)
        tile_slice = nl.ds(tile_idx * f_max, tile_size)
        psum_buf = nl.ndarray((N, tile_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(psum_buf, rot, data[:, tile_slice])
        nisa.tensor_copy(result[:, tile_slice], psum_buf, engine=nisa.vector_engine)

    out_sbuf = nl.ndarray((N, free_dim), dtype=input_data.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(out_sbuf, result, engine=nisa.vector_engine)
    nisa.dma_copy(output, out_sbuf)
    return output


# =============================================================================
# Torch Reference
# =============================================================================


def rotation_matrix_torch_ref(
    input_data,
    output,
    block_size,
    num_blocks,
    shifts,
    negate,
    use_scalar_engine,
):
    """Torch reference: apply circular shift with optional negation per block."""
    result = torch.zeros_like(input_data)
    for block_idx in range(num_blocks):
        base = block_idx * block_size
        for i in range(block_size):
            src_row = (i - shifts) % block_size
            sign = 1.0
            if negate and src_row >= (block_size - (shifts % block_size)):
                sign = -1.0
            result[base + i] = sign * input_data[base + src_row]
    return {"output": result}


# =============================================================================
# Tests
# =============================================================================


@pytest_test_metadata(name="RotationMatrix")
@pytest_marks(["rotation_matrix"])
@final
class TestRotationMatrix:
    """Tests for build_rotation_matrix with various shifts and block configs."""

    _ABBREVS = {
        "block_size": "bs",
        "num_blocks": "nb",
        "shifts": "s",
        "negate": "neg",
        "use_scalar_engine": "scalar",
    }

    @pytest_parametrize(
        "block_size,num_blocks,shifts,negate,use_scalar_engine",
        [
            # --- Identity and boundary shifts (vector engine) ---
            pytest.param(128, 1, 0, False, False, marks=pytest.mark.fast),
            (128, 1, 32, False, False),
            (128, 1, 64, False, False),
            (128, 1, 96, False, False),
            pytest.param(128, 1, 127, False, False, marks=pytest.mark.fast),
            # --- Primes crossing quadrant boundaries ---
            (128, 1, 31, False, False),
            (128, 1, 33, False, False),
            (128, 1, 97, False, False),
            # --- RoPE: Llama (d_head=128, shift=64) ---
            pytest.param(128, 1, 64, True, False, marks=pytest.mark.fast),
            # --- RoPE: GPT-NeoX/GPT-J (d_head=64, 2 packed heads) ---
            pytest.param(64, 2, 32, True, False, marks=pytest.mark.fast),
            # --- Negate crossing quadrant boundaries ---
            (128, 1, 33, True, False),
            (128, 1, 97, True, False),
            # --- Multi-block: power-of-2 ---
            (32, 4, 17, True, False),
            (16, 8, 9, False, False),
            # --- Multi-block: prime block sizes and counts ---
            (5, 3, 2, False, False),
            pytest.param(7, 15, 13, True, False, marks=pytest.mark.fast),
            (7, 5, 3, False, False),
            (7, 5, 3, True, False),
            # --- Scalar engine: unsigned permutation ---
            pytest.param(128, 1, 64, False, True, marks=pytest.mark.fast),
            # --- Scalar engine: RoPE with negation ---
            pytest.param(128, 1, 64, True, True, marks=pytest.mark.fast),
        ],
        abbrevs=_ABBREVS,
    )
    def test_rotation_matrix(
        self, test_manager, platform_target, block_size, num_blocks, shifts, negate, use_scalar_engine
    ):
        """Test build_rotation_matrix + nc_matmul application."""
        N = block_size * num_blocks
        free_dim = 512

        def input_generator(test_config):
            np.random.seed(42)
            input_data = np.random.randn(N, free_dim).astype(np.float16)
            return {
                "input_data": input_data,
                "output.must_alias_input": np.zeros_like(input_data),
                "block_size": block_size,
                "num_blocks": num_blocks,
                "shifts": shifts,
                "negate": negate,
                "use_scalar_engine": use_scalar_engine,
            }

        def output_tensors(kernel_input):
            return {"output": np.zeros((N, free_dim), dtype=np.float16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_rotation_matrix,
            torch_ref=torch_ref_wrapper(rotation_matrix_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )
