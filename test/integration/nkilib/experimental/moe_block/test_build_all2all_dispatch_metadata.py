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

"""Integration test for build_all2all_dispatch_metadata NKI kernel."""

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.moe_block.build_all2all_dispatch_metadata import (
    build_all2all_dispatch_metadata,
)
from nkilib_src.nkilib.experimental.moe_block.build_all2all_dispatch_metadata_torch import (
    build_all2all_dispatch_metadata_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _torch_ref(expert_index, num_experts, num_elements_per_token, replica_group_size):
    """Wrapper for torch reference that returns dict of numpy arrays."""
    import torch

    if isinstance(expert_index, np.ndarray):
        expert_index_t = torch.from_numpy(expert_index).to(torch.int32)
    else:
        expert_index_t = expert_index.to(torch.int32)
    result = build_all2all_dispatch_metadata_torch_ref(
        expert_index=expert_index_t,
        num_experts=num_experts,
        num_elements_per_token=num_elements_per_token,
        replica_group_size=replica_group_size,
    )
    # Return as float32 (framework stores golden as float32)
    return {"metadata": result.to(torch.float32).numpy()}


# fmt: off
PARAMS = "tokens, K, num_experts, replica_group_size, num_elements_per_token"
TEST_CASES = [
    # Exact E2E config from MoEE2EModelFused (test_moe_e2e_a2av.py):
    # MoETestConfig(T=8, H=8192, I=2048, E=16, top_k=4), NNODES=2
    # E_L=8, H_CONCAT=H+E_L+BF16_PER_INT32=8192+8+2=8202
    (8, 4, 16, 2, 8202),     # PRIMARY: matches production E2E
    (8, 2, 16, 2, 8202),     # K=2 variant of production config
    # Other configs
    (8, 2, 2, 2, 8194),      # E_L=1
    (8, 4, 32, 4, 8200),     # 4 ranks
    (1, 4, 16, 2, 8202),     # single token
]
# fmt: on


@pytest_test_metadata(name="Build All2All Dispatch Metadata")
@pytest_marks(["moe", "a2av", "metadata"])
class TestBuildAll2allDispatchMetadata:
    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN1])
    @pytest.mark.parametrize(PARAMS, TEST_CASES)
    def test_build_all2all_dispatch_metadata(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        tokens,
        K,
        num_experts,
        replica_group_size,
        num_elements_per_token,
    ):
        np.random.seed(42)

        def input_generator(_):
            # Generate random expert indices in [0, num_experts)
            expert_index = np.random.randint(0, num_experts, size=(tokens, K)).astype(np.int32)
            return {
                "expert_index": expert_index,
                "num_experts": num_experts,
                "num_elements_per_token": num_elements_per_token,
                "replica_group_size": replica_group_size,
            }

        def output_descriptor(ki):
            R = ki["replica_group_size"]
            return {
                "metadata": np.zeros((4, R), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=build_all2all_dispatch_metadata,
            torch_ref=torch_ref_wrapper(_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            rtol=0,
            atol=0,  # Exact integer match
        )
