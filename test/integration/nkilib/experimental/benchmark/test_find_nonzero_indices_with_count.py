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

import numpy as np
from nkilib_src.nkilib.experimental.benchmark.find_nonzero_indices_with_count import (
    PADDING_VALUE,
    find_nonzero_indices_with_count,
)
from nkilib_src.nkilib.experimental.benchmark.find_nonzero_indices_with_count_torch import (
    find_nonzero_indices_with_count_torch_ref,
)

from test.integration.nkilib.utils.tensor_generators import sparse_nonzero_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# fmt: off
PARAM_NAMES = "T, pct_nonzero"
TEST_PARAMS = [
    # All zero (no tokens routed to local expert)
    (32,   0.0),
    (64,   0.0),
    (128,  0.0),
    (256,  0.0),
    (512,  0.0),
    (1024, 0.0),
    (2048, 0.0),

    # 1/32 Nonzero (average for GPT-OSS 120B with E=128 and TopK=4)
    (32,   1/32),
    (64,   1/32),
    (128,  1/32),
    (256,  1/32),
    (512,  1/32),
    (1024, 1/32),
    (2048, 1/32),

    # 5/32 Nonzero (5x worse than average for GPT-OSS 120B with E=128 and TopK=4)
    (32,   5/32),
    (64,   5/32),
    (128,  5/32),
    (256,  5/32),
    (512,  5/32),
    (1024, 5/32),
    (2048, 5/32),

    # No zeros (all tokens routed to local expert)
    (32,   1.0),
    (64,   1.0),
    (128,  1.0),
    (256,  1.0),
    (512,  1.0),
    (1024, 1.0),
    (2048, 1.0),
]
# fmt: on


@pytest_test_metadata(name="FindNonzeroIndicesWithCount")
@pytest_marks(["find_nonzero_indices_with_count", "moe"])
class TestFindNonzeroIndicesWithCountKernel:
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_find_nonzero_indices_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        T: int,
        pct_nonzero: float,
    ):
        seed = 0
        tensor_gen = sparse_nonzero_tensor_generator(
            num_nonzero=int(T * pct_nonzero),
            value_range=(0.0, 1.0),
            seed=seed,
        )

        def input_generator(test_config):
            return {"input_tensor": tensor_gen(shape=(1, T), dtype=np.float32)}

        def output_tensors(kernel_input):
            T_val = kernel_input["input_tensor"].shape[-1]
            return {"output": np.full((1, T_val + 1), PADDING_VALUE, dtype=np.int32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=find_nonzero_indices_with_count,
            torch_ref=torch_ref_wrapper(find_nonzero_indices_with_count_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.0,
            atol=0.0,
        )
