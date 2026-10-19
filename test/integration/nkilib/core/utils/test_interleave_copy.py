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

"""Integration tests for interleave_copy utility.

interleave_copy drains a source tile into a destination tile (optionally fused
with a per-partition or full-shape scale/bias) while alternating between the
Scalar (even index) and Vector (odd index) engines. The tests sweep every
scale/bias shape combination on both engines to exercise each instruction path.

scale_mode / bias_mode encoding:
  0 = absent (None)
  1 = per-partition, shape (P, 1)
  2 = full-shape, shape (P, F)
"""

from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.interleave_copy import interleave_copy

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Test Kernel
# =============================================================================


@nki.jit
def kernel_interleave_copy(input_data, scale_data, bias_data, output, scale_mode, bias_mode, index):
    """Load src (and optional scale/bias per mode), run interleave_copy, return dst."""
    num_par, free_dim = input_data.shape
    src = nl.ndarray((num_par, free_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=src, src=input_data)
    dst = nl.ndarray((num_par, free_dim), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=dst, value=0.0)

    scale = None
    if scale_mode == 1:
        scale = nl.ndarray((num_par, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=scale, src=scale_data[:, 0:1])
    elif scale_mode == 2:
        scale = nl.ndarray((num_par, free_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=scale, src=scale_data)

    bias = None
    if bias_mode == 1:
        bias = nl.ndarray((num_par, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias, src=bias_data[:, 0:1])
    elif bias_mode == 2:
        bias = nl.ndarray((num_par, free_dim), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bias, src=bias_data)

    interleave_copy(dst, src, scale, bias, index=index)  # ty: ignore[invalid-argument-type]
    nisa.dma_copy(dst=output, src=dst)
    return output


# =============================================================================
# Torch Reference
# =============================================================================


def interleave_copy_torch_ref(input_data, scale_data, bias_data, output, scale_mode, bias_mode, index):
    """Reference: dst = src * scale + bias, broadcasting per the selected shapes."""
    result = input_data.to(torch.float32).clone()
    if scale_mode == 1:
        result = result * scale_data[:, 0:1]
    elif scale_mode == 2:
        result = result * scale_data
    if bias_mode == 1:
        result = result + bias_data[:, 0:1]
    elif bias_mode == 2:
        result = result + bias_data
    return {"output": result}


# =============================================================================
# Tests
# =============================================================================


@final
@pytest_test_metadata(name="InterleaveCopy")
@pytest_marks(["interleave_copy"])
class TestInterleaveCopy:
    """Tests for interleave_copy across all scale/bias shapes and both engines."""

    _ABBREVS = {
        "scale_mode": "scmode",
        "bias_mode": "bimode",
        "index": "idx",
    }

    @pytest_parametrize(
        "scale_mode,bias_mode,index",
        [
            # --- Scalar engine (even index, all operands per-partition or absent) ---
            pytest.param(1, 1, 0, marks=pytest.mark.fast),  # activation(scale, bias)
            (1, 0, 0),  # activation(scale)
            (0, 1, 0),  # activation(bias)
            (0, 0, 0),  # activation(copy)
            # --- Even index but full-shape operand forces the Vector engine ---
            (2, 0, 0),
            # --- Vector engine (odd index): full scale/bias shape matrix ---
            pytest.param(1, 1, 1, marks=pytest.mark.fast),  # fused tensor_scalar(mul, add)
            (1, 2, 1),  # scalar_tensor_tensor(mul, add)
            (2, 1, 1),  # tensor_tensor(mul) + tensor_scalar(add)
            (2, 2, 1),  # tensor_tensor(mul) + tensor_tensor(add)
            (1, 0, 1),  # tensor_scalar(mul)
            (2, 0, 1),  # tensor_tensor(mul)
            (0, 1, 1),  # tensor_scalar(add)
            (0, 2, 1),  # tensor_tensor(add)
            pytest.param(0, 0, 1, marks=pytest.mark.fast),  # tensor_copy
        ],
        abbrevs=_ABBREVS,
    )
    def test_interleave_copy(self, test_manager, platform_target, scale_mode, bias_mode, index):
        """Exercise each Scalar/Vector engine instruction path for the given operand shapes."""
        num_par, free_dim = 64, 16

        def input_generator(test_config):
            np.random.seed(42)
            input_data = np.random.randn(num_par, free_dim).astype(np.float32)
            scale_data = (np.random.randn(num_par, free_dim) + 1.0).astype(np.float32)
            bias_data = np.random.randn(num_par, free_dim).astype(np.float32)
            return {
                "input_data": input_data,
                "scale_data": scale_data,
                "bias_data": bias_data,
                "output.must_alias_input": np.zeros_like(input_data),
                "scale_mode": scale_mode,
                "bias_mode": bias_mode,
                "index": index,
            }

        def output_tensors(kernel_input):
            return {"output": np.zeros((num_par, free_dim), dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_interleave_copy,
            torch_ref=torch_ref_wrapper(interleave_copy_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )
