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

"""Performance sweep test for load_and_quantize_mxfp_mk (swizzle + quantize from HBM).

Sweeps K dimension from 1024 to 4096 with M=128.
"""

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.mx_utils import quantize_mx_golden
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import final

import ml_dtypes
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.mxfp_subkernels.mxfp_load_utils import (
    mxfp_load_performance_wrapper,
)

M = 128
P_MAX = 128
K_BLOCK_SIZE = 512
X4_PACK_FACTOR = 4

# quantize_mx writes 4 rows of scales at each of these partition offsets (16 rows total)
SCALE_PARTITION_OFFSETS = [0, 32, 64, 96]
SCALE_ROWS_PER_GROUP = 4


def generate_kernel_inputs(K):
    """Generate Gaussian-distributed BF16 input tensor with fixed seed."""
    np.random.seed(42)
    generate_tensor = gaussian_tensor_generator()
    return {"tensor": generate_tensor(name="tensor", shape=(M, K), dtype=ml_dtypes.bfloat16)}


def golden_mxfp_load(inputs):
    """Golden reference: swizzle + quantize per 512-element K block.

    For each K block of 512 BF16 elements (= [M=128, 512]):
        Swizzle: [M, 512] -> reshape [M, 128, 4] -> transpose(1, 0, 2) -> [128, M*4]
        Quantize: [128, M*4] -> mx_data [128, M] x4, mx_scale [16, M] uint8

    Scales are placed at partition offsets [0, 32, 64, 96] with 4 rows each,
    matching the hardware layout of quantize_mx. Remaining rows are zero-padded.

    Concatenate all K blocks along free dim -> [128, M * K_blocks]
    """
    tensor = inputs["tensor"]
    M_dim, K_dim = tensor.shape
    k_block_count = K_dim // K_BLOCK_SIZE
    out_free_dim = M_dim * k_block_count

    all_data = []
    scale_buf = np.zeros((P_MAX, out_free_dim), dtype=np.uint8)

    for k_block_idx in range(k_block_count):
        block = tensor[:, k_block_idx * K_BLOCK_SIZE : (k_block_idx + 1) * K_BLOCK_SIZE].astype(np.float32)

        # Swizzle: [M, 512] -> [128, M*4]
        swizzled = block.reshape(M_dim, P_MAX, X4_PACK_FACTOR).transpose(1, 0, 2).reshape(P_MAX, M_dim * X4_PACK_FACTOR)

        mx_data, mx_scale = quantize_mx_golden(swizzled, nl.float8_e4m3fn_x4)
        all_data.append(mx_data)

        # Place scale rows at hardware partition offsets
        free_slice = slice(k_block_idx * M_dim, (k_block_idx + 1) * M_dim)
        for grp_idx, partition_offset in enumerate(SCALE_PARTITION_OFFSETS):
            src = slice(grp_idx * SCALE_ROWS_PER_GROUP, (grp_idx + 1) * SCALE_ROWS_PER_GROUP)
            dst = slice(partition_offset, partition_offset + SCALE_ROWS_PER_GROUP)
            scale_buf[dst, free_slice] = mx_scale[src]

    mx_data_full = np.concatenate(all_data, axis=1)

    return {
        "out_data_hbm": mx_data_full.view(np.float32),
        "out_scale_hbm": scale_buf,
    }


@pytest_test_metadata(
    name="MxfpLoadPerformance",
    pytest_marks=["mxfp", "performance"],
    tags=[],
)
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMxfpLoadPerformance:
    @pytest.mark.fast
    @pytest.mark.parametrize("K", [1024, 1536, 2048, 2560, 3072, 3584, 4096])
    def test_mxfp_load_performance(self, test_manager: Orchestrator, K):
        """Sweep K dimension for load_and_quantize_mxfp_mk performance."""
        kernel_input = generate_kernel_inputs(K)
        out_free_dim = M * K // K_BLOCK_SIZE

        test_manager.execute(
            KernelArgs(
                kernel_func=mxfp_load_performance_wrapper,
                compiler_input=CompilerArgs(platform_target=Platforms.TRN3_A0),
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=lambda: golden_mxfp_load(kernel_input),
                        output_ndarray={
                            "out_data_hbm": np.zeros((P_MAX, out_free_dim), dtype=np.float32),
                            "out_scale_hbm": np.zeros((P_MAX, out_free_dim), dtype=np.uint8),
                        },
                    ),
                    absolute_accuracy=0.0,
                    relative_accuracy=0.0,
                ),
            ),
        )
