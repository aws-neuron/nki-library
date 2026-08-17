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

"""Tests for topk_reduce subkernel using UnitTestFramework."""

from typing import final

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.subkernels.topk_reduce import topk_reduce
from nkilib_src.nkilib.experimental.subkernels.topk_reduce_torch import topk_reduce_torch_ref

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_topk_reduce_inputs(T, K, H, src_ranks, variable_rows=False):
    """Generate scattered input buffer with packed 1-indexed global token indices.

    Args:
        T, K, H: token count, top-K, hidden dim.
        src_ranks: source rank count; padded input has src_ranks * T rows.
        variable_rows: if True, each token has a random row count in [1, K] instead
            of exactly K. Simulates the case where multiple experts for the same
            token sit on the same rank and are pre-reduced on that rank, so fewer
            than K rows survive into the scattered buffer.
    """
    np.random.seed(42)
    padded_input_size = src_ranks * T

    # Determine row count per token: [1, K] when variable_rows, else exactly K.
    if variable_rows:
        rows_per_token = np.random.randint(1, K + 1, size=T).astype(np.int32)
        # Pin deterministic edge cases: at least one token with 1 row and one with K.
        rows_per_token[0] = 1
        rows_per_token[-1] = K
    else:
        rows_per_token = np.full(T, K, dtype=np.int32)

    total_rows = int(rows_per_token.sum())

    # Padded input buffer filled with -1
    input_buf = np.full((padded_input_size, H + 2), -1, dtype=np.dtype('bfloat16'))

    # Real hidden values
    real_values = np.random.randn(total_rows, H).astype(np.dtype('bfloat16'))

    # Pack 1-indexed global token index as int32 viewed as 2 bf16 columns.
    # Token t (0-indexed) appears in the buffer with packed id t+1.
    token_ids = np.repeat(np.arange(1, T + 1, dtype=np.int32), rows_per_token).reshape(total_rows, 1)
    token_ids_bf16 = token_ids.view(np.dtype('bfloat16'))  # (total_rows, 2)

    real_with_idx = np.concatenate([real_values, token_ids_bf16], axis=1)

    # Randomly scatter into padded buffer
    scatter_indices = np.random.permutation(padded_input_size)[:total_rows]
    input_buf[scatter_indices] = real_with_idx

    return input_buf


# fmt: off
TOPK_REDUCE_PARAM_NAMES = "lnc_degree, src_ranks, T, K, H, variable_rows"
TOPK_REDUCE_PARAMS = [
    # Exactly K rows per token (every routed expert on a distinct rank).
    # 8r, lnc1
    (1, 8, 4, 4, 2880, False),
    # 8r, 32 - 1K global tokens
    (2, 8, 4, 4, 2880, False),
    (2, 8, 4, 8, 4096, False),
    (2, 8, 128, 4, 2880, False),
    (2, 8, 128, 8, 4096, False),
    # 128r, 512 - 16K global tokens
    (2, 128, 4, 4, 2880, False),
    (2, 128, 4, 8, 4096, False),
    (2, 128, 128, 4, 2880, False),
    (2, 128, 128, 8, 4096, False),
    # Fewer than K rows per token (multiple routed experts share a rank and get
    # pre-reduced on that rank before landing in the scattered buffer).
    (1, 8, 4, 4, 2880, True),
    (2, 8, 4, 4, 2880, True),
    (2, 8, 4, 8, 4096, True),
    (2, 8, 128, 4, 2880, True),
    (2, 8, 128, 8, 4096, True),
    (2, 128, 4, 4, 2880, True),
    (2, 128, 4, 8, 4096, True),
    (2, 128, 128, 4, 2880, True),
    (2, 128, 128, 8, 4096, True),
]
# fmt: on


@pytest_test_metadata(name="TopkReduce")
@pytest_marks(["topk_reduce", "subkernels"])
@final
class TestTopkReduceKernel:
    """Test class for topk_reduce subkernel."""

    @pytest.mark.fast
    @pytest.mark.parametrize(TOPK_REDUCE_PARAM_NAMES, TOPK_REDUCE_PARAMS)
    def test_topk_reduce(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree: int,
        src_ranks: int,
        T: int,
        K: int,
        H: int,
        variable_rows: bool,
    ) -> None:
        input_buf = generate_topk_reduce_inputs(T, K, H, src_ranks, variable_rows=variable_rows)

        def input_generator(test_config):
            return {
                "input": input_buf,
                "T": T,
                "K": K,
            }

        def output_tensors(kernel_input):
            return {
                "out": np.zeros((T, H), dtype=np.dtype('bfloat16')),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=topk_reduce,
            torch_ref=torch_ref_wrapper(topk_reduce_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=lnc_degree),
            inference_args=InferenceArgs(enable_determinism_check=True, num_runs=10),
            rtol=5e-2,
            atol=5e-2,
        )
