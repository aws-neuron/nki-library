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
"""Tests for fine-grained ring-based all-gather kernel."""

from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    PerRankLazyGoldenGenerator,
    PerRankLazyInputGenerator,
    ValidationArgs,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.collectives.fg_allgather import (
    fine_grained_allgather,
)


@pytest_test_metadata(
    name="FgAllgather",
    pytest_marks=["collectives", "FgAllgather"],
)
class TestFgAllgather:
    """Test class for fine-grained ring-based all-gather kernel."""

    @pytest.mark.parametrize(
        "m, K, dtype, tp_degree, lnc, force_hbm_cc",
        [
            # Basic LNC2 tests
            pytest.param(1024, 4096, nl.bfloat16, 4, 2, False, id="m1024_K4096_bf16_tp4_lnc2"),
            # FIXME: The following tests doesn't work and suffers from time out
            # NKILIB-795
            #
            # pytest.param(256, 2048, nl.bfloat16, 16, 2, False, id="m256_K2048_bf16_tp16_lnc2"),
            # pytest.param(512, 8192, nl.bfloat16, 4, 2, False, id="m512_K8192_bf16_tp4_lnc2"),
            # # Force HBM mode
            # pytest.param(512, 8192, nl.bfloat16, 4, 2, True, id="m512_K8192_bf16_tp4_lnc2_hbm"),
            # # Float32
            # pytest.param(1024, 4096, nl.float32, 16, 2, False, id="m1024_K4096_f32_tp4_lnc2"),
            # # LNC1
            # pytest.param(1024, 4096, nl.bfloat16, 4, 1, False, id="m1024_K4096_bf16_tp4_lnc1"),
        ],
    )
    def test_fine_grained_allgather(
        self,
        test_manager: Orchestrator,
        m: int,
        K: int,
        dtype: np.dtype,
        tp_degree: int,
        lnc: int,
        force_hbm_cc: bool,
    ):
        """Test fine-grained ring-based all-gather kernel."""
        np.random.seed(42)
        num_groups = 1
        M = m * tp_degree

        # Global tensor: [M, K] — each rank owns rows [rank*m : (rank+1)*m]
        lhs_global = np.random.randn(M, K).astype(dtype)

        def create_inputs(rank_id: int):
            return {
                "lhs": lhs_global[rank_id * m : (rank_id + 1) * m, :],
                "tp_degree": tp_degree,
                "num_groups": num_groups,
                "force_hbm_cc": force_hbm_cc,
            }

        def create_golden(rank_id: int):
            # All ranks produce the full gathered tensor
            return {"result": lhs_global.astype(dtype)}

        test_manager.execute(
            KernelArgs(
                kernel_func=fine_grained_allgather,
                compiler_input=CompilerArgs(logical_nc_config=lnc),
                kernel_input=PerRankLazyInputGenerator(create_inputs),
                inference_args=InferenceArgs(collective_ranks=tp_degree),
                validation_args=ValidationArgs(
                    golden_output=PerRankLazyGoldenGenerator(create_golden),
                    absolute_accuracy=1e-3,
                    relative_accuracy=1e-3,
                ),
            )
        )
