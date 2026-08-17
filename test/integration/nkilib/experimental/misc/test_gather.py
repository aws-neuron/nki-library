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

"""Integration tests for gather kernel."""

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.misc.gather import gather
from nkilib_src.nkilib.experimental.misc.gather_torch import gather_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_inputs(bs_slen, dim_size, top_k, dtype):
    np.random.seed(42)
    output_rows = bs_slen * top_k
    return {
        "input": np.random.randn(bs_slen, dim_size).astype(dtype),
        "dim": 0,
        "index": np.random.randint(0, bs_slen, size=(output_rows,)).astype(np.int32),
    }


def _output_tensors(kernel_input):
    k = kernel_input["index"].shape[0]
    d = kernel_input["input"].shape[1]
    return {"output_0": np.zeros((k, d), dtype=kernel_input["input"].dtype)}


PARAM_NAMES = "bs_slen, dim_size, top_k, dtype"
TEST_PARAMS = [
    (128, 512, 2, nl.bfloat16),
    (128, 512, 2, nl.float32),
    (256, 2048, 8, nl.bfloat16),
    (512, 1024, 4, nl.float32),
    (1024, 4096, 2, nl.bfloat16),
    (4096, 2880, 4, nl.bfloat16),
    (4096, 2880, 4, nl.float32),
]


@pytest_test_metadata(name="Gather")
@pytest_marks(["gather"])
class TestGatherKernel:
    """Test class for gather kernel."""

    @pytest.mark.fast
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS)
    def test_gather(self, test_manager: Orchestrator, platform_target: Platforms, bs_slen, dim_size, top_k, dtype):
        is_bf16 = dtype == nl.bfloat16

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gather,
            torch_ref=torch_ref_wrapper(gather_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(bs_slen, dim_size, top_k, dtype),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=1e-2 if is_bf16 else 1e-4,
            rtol=1e-2 if is_bf16 else 1e-5,
        )
