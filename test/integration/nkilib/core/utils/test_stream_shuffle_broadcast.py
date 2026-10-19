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

"""Integration tests for stream_shuffle_broadcast utility.

Broadcasts the first partition of a 2D SBUF tensor across all partitions of the
destination via nc_stream_shuffle, processing the partition dim in 32-partition
chunks.
"""

from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Test Kernel
# =============================================================================


@nki.jit
def kernel_stream_shuffle_broadcast(input_data, output, dst_npar):
    """Load a single-partition row and broadcast it across dst_npar partitions."""
    free_dim = input_data.shape[1]
    src = nl.ndarray((1, free_dim), dtype=input_data.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=src, src=input_data)
    dst = nl.ndarray((dst_npar, free_dim), dtype=input_data.dtype, buffer=nl.sbuf)
    nisa.memset(dst=dst, value=0.0)
    stream_shuffle_broadcast(src=src, dst=dst)
    nisa.dma_copy(dst=output, src=dst)
    return output


# =============================================================================
# Torch Reference
# =============================================================================


def stream_shuffle_broadcast_torch_ref(input_data, output, dst_npar):
    """Broadcast the first row across dst_npar partitions."""
    result = input_data[0:1, :].expand(dst_npar, -1).contiguous()
    return {"output": result}


# =============================================================================
# Tests
# =============================================================================


@final
@pytest_test_metadata(name="StreamShuffleBroadcast")
@pytest_marks(["stream_shuffle_broadcast"])
class TestStreamShuffleBroadcast:
    """Tests for stream_shuffle_broadcast utility with full data validation."""

    @pytest.mark.parametrize(
        "dst_npar,free_dim",
        [
            pytest.param(1, 8, marks=pytest.mark.fast),
            (32, 4),
            (33, 8),
            (64, 16),
            (128, 32),
        ],
    )
    def test_stream_shuffle_broadcast(self, test_manager, platform_target, dst_npar, free_dim):
        """Broadcast first partition across a range of partition counts (multi-chunk)."""

        def input_generator(test_config):
            np.random.seed(42)
            input_data = np.random.randn(1, free_dim).astype(np.float32)
            return {
                "input_data": input_data,
                "output.must_alias_input": np.zeros((dst_npar, free_dim), dtype=np.float32),
                "dst_npar": dst_npar,
            }

        def output_tensors(kernel_input):
            return {"output": np.zeros((dst_npar, free_dim), dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_stream_shuffle_broadcast,
            torch_ref=torch_ref_wrapper(stream_shuffle_broadcast_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )
