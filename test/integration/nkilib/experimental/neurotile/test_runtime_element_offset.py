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

"""Runtime element-offset indexing tests for NeuroTile."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _07_runtime_element_offset as runtime_offset_example,
)
from nkilib_src.nkilib.experimental.neurotile.examples._05_indirect import (
    _07_runtime_element_offset_torch as runtime_offset_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

P_MAX = runtime_offset_example.P_TILE


def _inputs(_):
    np.random.seed(42)
    m_dim = 384
    h_dim = 1024
    input_a = np.random.randn(m_dim, h_dim).astype(ml_dtypes.bfloat16)
    input_b = np.random.randn(m_dim, h_dim).astype(ml_dtypes.bfloat16)
    num_m_tiles = np.array([[m_dim // P_MAX]], dtype=np.int32)
    return {"input_a": input_a, "input_b": input_b, "num_m_tiles": num_m_tiles}


def _outputs(kernel_input):
    return {"out": np.zeros(kernel_input["input_a"].shape, dtype=kernel_input["input_a"].dtype)}


@pytest_marks(["neurotile"])
class TestRuntimeElementOffset:
    @pytest.mark.fast
    def test_runtime_element_offset_load_store(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=runtime_offset_example.dynamic_elementwise_add,
            torch_ref=torch_ref_wrapper(runtime_offset_ref.dynamic_elementwise_add_torch_ref),
            kernel_input_generator=_inputs,
            output_tensor_descriptor=_outputs,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-2,
            atol=1e-3,
        )
