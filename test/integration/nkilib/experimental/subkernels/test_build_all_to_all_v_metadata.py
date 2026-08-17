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

"""Tests for build_all_to_all_v_metadata subkernel using UnitTestFramework."""

from typing import final

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.subkernels.build_all_to_all_v_metadata import build_all_to_all_v_metadata
from nkilib_src.nkilib.experimental.subkernels.build_all_to_all_v_metadata_torch import (
    build_all_to_all_v_metadata_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# fmt: off
PARAM_NAMES = "lnc_degree, T, K, E, replica_group_size"
PARAMS = [
    # RG size = E
    (2, 1,  8, 256, 128),
    (2, 4,  4, 128, 128),
    (2, 8,  2, 16,  16),
    (2, 16, 2, 8,   8),
    (2, 32, 4, 16,  16),
    (2, 32, 4, 128, 128),
    (2, 512, 8, 1024, 1024),
    # RG size < E
    (2, 1,  8, 256, 16),
    (2, 32, 4, 128, 16),
    (2, 512, 8, 1024, 256),
]
# fmt: on


@pytest_test_metadata(
    name="BuildAllToAllVMetadata",
    pytest_marks=["build_all_to_all_v_metadata", "subkernels"],
)
@final
class TestBuildAllToAllVMetadataKernel:
    """Test class for build_all_to_all_v_metadata subkernel."""

    @pytest.mark.fast
    @pytest.mark.parametrize(PARAM_NAMES, PARAMS)
    def test_build_all_to_all_v_metadata(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree: int,
        T: int,
        K: int,
        E: int,
        replica_group_size: int,
    ) -> None:
        np.random.seed(42)
        expert_index = np.stack([np.random.choice(E, size=K, replace=False) for _ in range(T)]).astype(np.int32)

        def input_generator(test_config):
            return {
                "expert_index": expert_index,
                "replica_group_size": replica_group_size,
                "E": E,
            }

        def output_tensors(kernel_input):
            # FIXME: change to np.uint32 when nkilib testing infra supports it
            return {"out": np.zeros((3, replica_group_size), dtype=np.int32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=build_all_to_all_v_metadata,
            torch_ref=torch_ref_wrapper(build_all_to_all_v_metadata_torch_ref),
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
