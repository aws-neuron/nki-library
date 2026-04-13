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

"""Integration tests for indexed_flatten kernel."""

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import numpy as np
import pytest
from nkilib_src.nkilib.core.subkernels.indexed_flatten import indexed_flatten
from nkilib_src.nkilib.core.subkernels.indexed_flatten_torch import indexed_flatten_torch_ref


def _generate_input_tensor(T: int, E: int) -> np.ndarray:
    """Generate deterministic input tensor with sequential int32 values.

    Args:
        T (int): Number of elements per row
        E (int): Number of rows

    Returns:
        np.ndarray: [E, T] input tensor with sequential int32 values
    """
    return np.arange(E * T, dtype=np.int32).reshape(E, T)


def _output_tensor_descriptor(kernel_input):
    """Generate output tensor descriptor from kernel inputs."""
    return {"flattened_array": np.zeros(kernel_input["output_len"], dtype=np.int32)}


# fmt: off
FAST_PARAM_NAMES = \
    "T, E, f_len"
FAST_TEST_PARAMS = [
    (4096,  1,   128),
    (4096,  2,   128),
    (4096,  2,   256),
    (4096,  3,   128),
    (10240, 16,  128),
    (10240, 17,  128),
    (65536, 4,   128),
    (65536, 4,   256),
    (4096,  64,  128),
    (4096,  128, 128),
    (4096,  256, 128),
]

ROW_OFFSET_PARAM_NAMES = \
    "T, E, f_len, N, row_offsets_start"
ROW_OFFSET_TEST_PARAMS = [
    (4096,  2, 128, 4,  2),
    (4096,  3, 128, 5,  1),
    (10240, 8, 128, 16, 8),
]
# fmt: on


@pytest_test_metadata(
    name="IndexedFlatten",
    pytest_marks=["indexed_flatten"],
)
class TestIndexedFlattenKernel:
    """Test class for indexed_flatten kernel."""

    def _run_test(self, test_manager: Orchestrator, input_generator):
        """Run indexed_flatten test with the given input generator."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=indexed_flatten,
            torch_ref=torch_ref_wrapper(indexed_flatten_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2),
            atol=0,
            rtol=0,
        )

    @pytest.mark.fast
    @pytest_parametrize(FAST_PARAM_NAMES, FAST_TEST_PARAMS)
    def test_indexed_flatten_fast(
        self,
        test_manager: Orchestrator,
        T: int,
        E: int,
        f_len: int,
    ):
        """Fast compile-only tests with minimal coverage."""

        def input_generator(test_config):
            partitions_per_row = T // f_len
            row_offsets_list = [partitions_per_row * idx for idx in range(E)]
            output_len = E * T + 1024
            output_len = ((output_len + 127) // 128) * 128
            return {
                "input_tensor": _generate_input_tensor(T, E),
                "f_len": f_len,
                "output_len": output_len,
                "row_offsets": np.array(row_offsets_list, dtype=np.int32),
                "row_offsets_start": None,
                "padding_val": -1,
            }

        self._run_test(test_manager, input_generator)

    @pytest.mark.fast
    @pytest_parametrize(ROW_OFFSET_PARAM_NAMES, ROW_OFFSET_TEST_PARAMS)
    def test_indexed_flatten_row_offsets_start(
        self,
        test_manager: Orchestrator,
        T: int,
        E: int,
        f_len: int,
        N: int,
        row_offsets_start: int,
    ):
        """Test row_offsets_start parameter for tensor-parallel MoE use cases."""

        def input_generator(test_config):
            partitions_per_row = T // f_len
            row_offsets_list = [partitions_per_row * idx for idx in range(N)]
            output_len = N * T + 1024
            output_len = ((output_len + 127) // 128) * 128
            return {
                "input_tensor": _generate_input_tensor(T, E),
                "f_len": f_len,
                "output_len": output_len,
                "row_offsets": np.array(row_offsets_list, dtype=np.int32),
                "row_offsets_start": np.array([row_offsets_start], dtype=np.int32),
                "padding_val": -1,
            }

        self._run_test(test_manager, input_generator)
