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

"""Integration tests for ve_transpose_broadcast utility.

Transposes a partition-dim column into the free dimension via the vector-engine
nc_transpose (in 32-element chunks), then broadcasts partition 0 across all
destination partitions.
"""

from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.ve_transpose_broadcast import ve_transpose_broadcast

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Test Kernel
# =============================================================================


@nki.jit
def kernel_ve_transpose_broadcast(input_data, output, num_free, num_par):
    """Transpose a (num_free, 1) column to the free dim and broadcast to num_par partitions."""
    src = nl.ndarray((num_free, 1), dtype=input_data.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=src, src=input_data)
    dst = nl.ndarray((num_par, num_free), dtype=input_data.dtype, buffer=nl.sbuf)
    nisa.memset(dst=dst, value=0.0)
    ve_transpose_broadcast(src=src[:, :], dst=dst, num_free=num_free, num_par=num_par)
    nisa.dma_copy(dst=output, src=dst)
    return output


# =============================================================================
# Torch Reference
# =============================================================================


def ve_transpose_broadcast_torch_ref(input_data, output, num_free, num_par):
    """Transpose the column into the free dim, then broadcast across partitions."""
    row = input_data[:num_free, 0].reshape(1, num_free)
    result = row.expand(num_par, -1).contiguous()
    return {"output": result}


# =============================================================================
# Tests
# =============================================================================


@final
@pytest_test_metadata(name="VeTransposeBroadcast")
@pytest_marks(["ve_transpose_broadcast"])
class TestVeTransposeBroadcast:
    """Tests for ve_transpose_broadcast utility with full data validation."""

    @pytest.mark.parametrize(
        "num_free,num_par",
        [
            pytest.param(8, 16, marks=pytest.mark.fast),
            (32, 8),
            (40, 8),
            (64, 128),
        ],
    )
    def test_ve_transpose_broadcast(self, test_manager, platform_target, num_free, num_par):
        """Transpose+broadcast for single-chunk and multi-chunk free dimensions."""

        def input_generator(test_config):
            np.random.seed(42)
            input_data = np.random.randn(num_free, 1).astype(np.float32)
            return {
                "input_data": input_data,
                "output.must_alias_input": np.zeros((num_par, num_free), dtype=np.float32),
                "num_free": num_free,
                "num_par": num_par,
            }

        def output_tensors(kernel_input):
            return {"output": np.zeros((num_par, num_free), dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_ve_transpose_broadcast,
            torch_ref=torch_ref_wrapper(ve_transpose_broadcast_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )
