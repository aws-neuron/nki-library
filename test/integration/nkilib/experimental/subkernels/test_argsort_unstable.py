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

"""Tests for argsort_unstable subkernel using UnitTestFramework."""

from typing import final

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.subkernels.argsort_unstable import argsort_unstable
from nkilib_src.nkilib.experimental.subkernels.argsort_unstable_torch import argsort_unstable_torch_ref

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# fmt: off
ARGSORT_PARAM_NAMES = "lnc_degree, N, descending, num_unique"
ARGSORT_PARAMS = [
    # Descending
    (2, 8, True, 8),
    (2, 16, True, 4),
    # Ascending
    (2, 8, False, 4),
    (2, 8, False, 8),
    (2, 16, False, 4),
    (2, 16, False, 16),
    (2, 32, False, 8),
    (2, 32, False, 32),
    (2, 64, False, 8),
    (2, 64, False, 64),
    (2, 128, False, 16),
    (2, 128, False, 128),
    (2, 256, False, 128),
    (2, 512, False, 128),
    (2, 1024, False, 256),
]
# fmt: on


@pytest_test_metadata(
    name="ArgsortUnstable",
    pytest_marks=["argsort_unstable", "subkernels"],
)
@final
class TestArgsortUnstableKernel:
    """Test class for argsort_unstable subkernel."""

    @pytest.mark.fast
    @pytest.mark.parametrize(ARGSORT_PARAM_NAMES, ARGSORT_PARAMS)
    def test_argsort_unstable(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree: int,
        N: int,
        descending: bool,
        num_unique: int,
    ) -> None:
        np.random.seed(42)
        unique_vals = np.random.randint(0, 100, size=num_unique).astype(np.int32)
        data = np.random.choice(unique_vals, size=N).astype(np.int32).reshape(1, N)

        def input_generator(test_config):
            return {"data": data, "descending": descending}

        def output_tensors(kernel_input):
            return {"out": np.zeros((1, N), dtype=np.int32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=argsort_unstable,
            torch_ref=torch_ref_wrapper(argsort_unstable_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=lnc_degree),
            inference_args=InferenceArgs(enable_determinism_check=True, num_runs=10),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("N", [16, 64])
    def test_argsort_unstable_1d(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        N: int,
    ) -> None:
        """Test argsort_unstable with 1D input (used by permute_routed_tokens)."""
        np.random.seed(42)
        data = np.random.randint(0, 10, size=(N,)).astype(np.int32)

        def input_generator(test_config):
            return {"data": data, "descending": False}

        def output_tensors(kernel_input):
            return {"out": np.zeros((N,), dtype=np.uint32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=argsort_unstable,
            torch_ref=torch_ref_wrapper(argsort_unstable_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            inference_args=InferenceArgs(enable_determinism_check=True, num_runs=10),
            rtol=0,
            atol=0,
        )
