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

from typing import final

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.mxfp_subkernels.mxfp_load_utils import (
    mxfp_load_performance_wrapper,
)
from nkilib_src.nkilib.experimental.mxfp_subkernels.mxfp_load_utils_torch import (
    mxfp_load_performance_wrapper_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework

M = 128
P_MAX = 128
K_BLOCK_SIZE = 512


@pytest_test_metadata(name="MxfpLoadPerformance")
@pytest_marks(["mx", "mxfp8", "performance"])
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestMxfpLoadPerformance:
    @pytest.mark.fast
    @pytest.mark.parametrize("K", [1024, 1536, 2048, 2560, 3072, 3584, 4096])
    def test_mxfp_load_performance(self, test_manager: Orchestrator, platform_target: Platforms, K):
        """Sweep K dimension for load_and_quantize_mxfp_mk performance."""

        def input_generator(test_config):
            np.random.seed(42)
            return {"tensor": np.random.randn(M, K).astype(ml_dtypes.bfloat16)}

        def output_tensors(kernel_input):
            out_free_dim = M * K // K_BLOCK_SIZE
            return {
                "out_data_hbm": np.zeros((P_MAX, out_free_dim), dtype=np.float32),
                "out_scale_hbm": np.zeros((P_MAX, out_free_dim), dtype=np.uint8),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mxfp_load_performance_wrapper,
            torch_ref=mxfp_load_performance_wrapper_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
            ),
            rtol=0.0,
            atol=0.0,
        )
