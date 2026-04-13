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
"""Tests for fused all-gather + compute matmul (FGCC) kernel."""

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
from nkilib_src.nkilib.experimental.collectives.fgcc import (
    allgather_compute_matmul,
)


@pytest_test_metadata(
    name="FGCC",
    pytest_marks=["collectives", "fgcc"],
)
class TestFgcc:
    """Test class for fused all-gather + compute matmul kernel."""

    @pytest.mark.parametrize(
        "m, K, N, dtype, tp_degree, lnc, force_hbm_cc",
        [
            # FIXME: All but one test cases are timing out onthe pipeline for both beta2 and beta3
            # NKILIB-796
            #
            # Basic LNC2 tests
            # pytest.param(256, 1024, 1024, nl.bfloat16, 4, 2, False, id="m256_K1024_N1024_bf16_tp4_lnc2"),
            # pytest.param(512, 2048, 2048, nl.bfloat16, 4, 2, False, id="m512_K2048_N2048_bf16_tp4_lnc2"),
            pytest.param(1024, 4096, 4096, nl.bfloat16, 4, 2, False, id="m1024_K4096_N4096_bf16_tp4_lnc2"),
            # TP16 tests
            # pytest.param(512, 1024, 1024, nl.bfloat16, 16, 2, False, id="m512_K1024_N1024_bf16_tp16_lnc2"),
            # pytest.param(2048, 4096, 8192, nl.bfloat16, 16, 2, False, id="m2048_K4096_N8192_bf16_tp16_lnc2"),
            # # TP64 tests
            # pytest.param(512, 1024, 1024, nl.bfloat16, 64, 2, False, id="m512_K1024_N1024_bf16_tp64_lnc2"),
            # pytest.param(512, 1024, 1024, nl.bfloat16, 64, 1, False, id="m512_K1024_N1024_bf16_tp64_lnc1"),
            # pytest.param(512, 8192, 32768, nl.bfloat16, 64, 2, False, id="m512_K8192_N32768_bf16_tp64_lnc2"),
            # # Force HBM mode
            # pytest.param(1024, 2048, 2048, nl.bfloat16, 4, 2, True, id="m1024_K2048_N2048_bf16_tp4_lnc2_hbm"),
            # pytest.param(256, 1024, 1024, nl.bfloat16, 4, 2, True, id="m256_K1024_N1024_bf16_tp4_lnc2_hbm"),
            # pytest.param(256, 1024, 1024, nl.bfloat16, 4, 1, True, id="m256_K1024_N1024_bf16_tp4_lnc1_hbm"),
            # # Float32 tests
            # pytest.param(256, 1024, 1024, nl.float32, 4, 2, False, id="m256_K1024_N1024_f32_tp4_lnc2"),
            # # LNC1 tests
            # pytest.param(256, 1024, 1024, nl.bfloat16, 4, 1, False, id="m256_K1024_N1024_bf16_tp4_lnc1"),
        ],
    )
    def test_allgather_compute_matmul(
        self,
        test_manager: Orchestrator,
        m: int,
        K: int,
        N: int,
        dtype: np.dtype,
        tp_degree: int,
        lnc: int,
        force_hbm_cc: bool,
    ):
        """Test fused all-gather + compute matmul (FGCC) kernel."""
        np.random.seed(42)
        num_groups = 1

        # lhs: (m * tp_degree, K) row-sharded -> each rank gets (m, K)
        # rhs: (K, N) column-sharded -> each rank gets (K, N // tp_degree)
        lhs_global = np.random.randn(m * tp_degree, K).astype(dtype)
        rhs_global = np.random.randn(K, N).astype(dtype)

        # Golden: full matmul
        result_global = (lhs_global.astype(np.float32) @ rhs_global.astype(np.float32)).astype(dtype)

        def create_inputs(rank_id: int):
            return {
                "lhs": lhs_global[rank_id * m : (rank_id + 1) * m, :],
                "rhs": rhs_global[:, rank_id * (N // tp_degree) : (rank_id + 1) * (N // tp_degree)],
                "tp_degree": tp_degree,
                "num_groups": num_groups,
                "force_hbm_cc": force_hbm_cc,
            }

        def create_golden(rank_id: int):
            # Each rank gets column-sharded result
            return {"result": result_global[:, rank_id * (N // tp_degree) : (rank_id + 1) * (N // tp_degree)]}

        test_manager.execute(
            KernelArgs(
                kernel_func=allgather_compute_matmul,
                compiler_input=CompilerArgs(logical_nc_config=lnc),
                kernel_input=PerRankLazyInputGenerator(create_inputs),
                inference_args=InferenceArgs(collective_ranks=tp_degree),
                validation_args=ValidationArgs(
                    golden_output=PerRankLazyGoldenGenerator(create_golden),
                    absolute_accuracy=1e-2,
                    relative_accuracy=1e-2,
                ),
            )
        )
