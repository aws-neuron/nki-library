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

"""Integration tests for scatter_add kernel."""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.misc.scatter_add import scatter_add
from nkilib_src.nkilib.experimental.misc.scatter_add_torch import scatter_add_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_inputs(bs_slen, dim_size, top_k, dtype):
    """Generate randomized inputs for scatter_add.

    The kernel requires unique destination indices within each 128-row tile,
    so indices are built as a concatenation of per-tile permutations of the
    destination row range. This requires bs_slen >= 128.
    """
    np.random.seed(42)
    src_rows = bs_slen * top_k

    indices = []
    for start in range(0, src_rows, 128):
        tile_len = min(128, src_rows - start)
        indices.append(np.random.permutation(bs_slen)[:tile_len])

    return {
        "input.must_alias_input": np.random.randn(bs_slen, dim_size).astype(dtype),
        "dim": 0,
        "index": np.concatenate(indices).astype(np.int32),
        "src": np.random.randn(src_rows, dim_size).astype(dtype),
    }


def _output_tensors(kernel_input):
    return {"output": kernel_input["input.must_alias_input"]}


PARAM_NAMES = "bs_slen, dim_size, top_k, dtype"
TEST_PARAMS = [
    (128, 512, 4, nl.float32),
    (128, 512, 2, nl.bfloat16),
    (256, 2048, 8, nl.bfloat16),
    (256, 2048, 8, nl.float32),
    (512, 1024, 4, nl.bfloat16),
    (512, 1024, 4, nl.float32),
    (1024, 4096, 2, nl.bfloat16),
    (1024, 4096, 2, nl.float32),
    (4096, 2880, 4, nl.bfloat16),
    (4096, 2880, 4, nl.float32),
]


@pytest_test_metadata(name="ScatterAdd")
@pytest_marks(["scatter_add"])
class TestScatterAddKernel:
    """Test class for scatter_add kernel."""

    @pytest.mark.fast
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_scatter_add(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        bs_slen,
        dim_size,
        top_k,
        dtype,
    ):
        is_bf16 = dtype == nl.bfloat16

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=scatter_add,
            torch_ref=torch_ref_wrapper(scatter_add_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(bs_slen, dim_size, top_k, dtype),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
                dump_after_lowering=False,
                logical_nc_config=2,
            ),
            atol=1e-1 if is_bf16 else 1e-4,
            rtol=1e-2 if is_bf16 else 1e-5,
        )
