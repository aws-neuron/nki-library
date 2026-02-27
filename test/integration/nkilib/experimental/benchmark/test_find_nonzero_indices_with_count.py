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

"""Integration tests for find_nonzero_indices_with_count kernel."""

from test.integration.nkilib.utils.tensor_generators import sparse_nonzero_tensor_generator
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import Any, final

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.benchmark.find_nonzero_indices_with_count import (
    PADDING_VALUE,
    find_nonzero_indices_with_count,
)


def build_output_placeholder(inp: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Build placeholder outputs for LazyGoldenGenerator and golden_output_validator.

    Args:
        inp (dict[str, Any]): Input dictionary containing 'input_tensor'

    Returns:
        dict[str, np.ndarray]: Dictionary with 'output' placeholder array
    """
    T = inp['input_tensor'].shape[-1]
    return {"output": np.full((1, T + 1), PADDING_VALUE, dtype=np.int32)}


def golden_output_validator(inp: dict[str, Any]) -> dict[str, np.ndarray]:
    """
    Compute expected output for find_nonzero_indices_with_count.

    Output format: [idx1, idx2, ..., -1, -1, ..., count] where indices of nonzero
    elements come first, followed by padding (-1), and the count is the LAST element.

    Args:
        inp (dict[str, Any]): Input dictionary containing 'input_tensor'

    Returns:
        dict[str, np.ndarray]: Dictionary with 'output' containing expected result
    """
    input_tensor = inp['input_tensor']

    nonzero_indices = np.nonzero(input_tensor[0])[0]
    count = len(nonzero_indices)

    output = build_output_placeholder(inp)
    output["output"][0, :count] = nonzero_indices
    output["output"][0, -1] = count

    return output


def build_find_nonzero_indices_kernel_input(T: int, tensor_gen) -> dict[str, np.ndarray]:
    """
    Build input tensor for find_nonzero_indices_with_count kernel.

    Args:
        T (int): Sequence length
        tensor_gen: Tensor generator used to build input tensor

    Returns:
        dict[str, np.ndarray]: Dictionary with 'input_tensor' key
    """
    input_tensor = tensor_gen(shape=(1, T), dtype=np.float32)
    return {"input_tensor": input_tensor}


@pytest_test_metadata(
    name="FindNonzeroIndicesWithCount",
    pytest_marks=["find_nonzero_indices_with_count", "moe"],
)
@final
class TestFindNonzeroIndicesWithCountKernel:
    def run_find_nonzero_indices_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: MetricsCollector,
        T: int,
        pct_nonzero: float,
        min_val: float = 0.0,
        max_val: float = 1.0,
        seed: int = 0,
    ):
        tensor_gen = sparse_nonzero_tensor_generator(
            num_nonzero=int(T * pct_nonzero),
            value_range=(min_val, max_val),
            seed=seed,
        )

        kernel_input = build_find_nonzero_indices_kernel_input(
            T=T,
            tensor_gen=tensor_gen,
        )

        # Build golden generator + placeholder output
        def create_lazy_golden():
            return golden_output_validator(inp=kernel_input)

        output_placeholder = build_output_placeholder(inp=kernel_input)

        test_manager.execute(
            KernelArgs(
                kernel_func=find_nonzero_indices_with_count,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_placeholder,
                    ),
                    relative_accuracy=0.0,  # Exact match for indices
                    absolute_accuracy=0.0,
                ),
            )
        )

    # fmt: off
    find_nonzero_indices_params = "T, pct_nonzero, tpbSgCyclesSum"
    find_nonzero_indices_perms = [
        # All zero (no tokens routed to local expert)
        (32, 0.0, None),
        (64, 0.0, None),
        (128, 0.0, None),
        (256, 0.0, None),
        (512, 0.0, None),
        (1024, 0.0, None),
        (2048, 0.0, None),

        # 1/32 Nonzero (average for GPT-OSS 120B with E=128 and TopK=4)
        (32, 1/32, None),
        (64, 1/32, None),
        (128, 1/32, None),
        (256, 1/32, None),
        (512, 1/32, None),
        (1024, 1/32, None),
        (2048, 1/32, None),

        # 5/32 Nonzero (5x worse than average for GPT-OSS 120B with E=128 and TopK=4)
        (32, 5/32, None),
        (64, 5/32, None),
        (128, 5/32, None),
        (256, 5/32, None),
        (512, 5/32, None),
        (1024, 5/32, None),
        (2048, 5/32, None),

        # No zeros (all tokens routed to local expert)
        (32, 1.0, None),
        (64, 1.0, None),
        (128, 1.0, None),
        (256, 1.0, None),
        (512, 1.0, None),
        (1024, 1.0, None),
        (2048, 1.0, None),

    ]
    # fmt: on

    @pytest.mark.parametrize(find_nonzero_indices_params, find_nonzero_indices_perms)
    def test_find_nonzero_indices_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        T,
        pct_nonzero,
        platform_target: Platforms,
        tpbSgCyclesSum,
    ):
        compiler_args = CompilerArgs(platform_target=platform_target)
        self.run_find_nonzero_indices_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            T=T,
            pct_nonzero=pct_nonzero,
        )
