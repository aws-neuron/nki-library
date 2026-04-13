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
"""Tests for SBUF-to-SBUF All-Gather kernels."""

from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    PerRankLazyGoldenGenerator,
    PerRankLazyInputGenerator,
    ValidationArgs,
)
from test.utils.test_orchestrator import Orchestrator

import nki.language as nl
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.sb2sb_allgather import (
    allgather_sb2sb,
    allgather_sb2sb_tiled,
)


class TestSb2sbAllgather:
    """Test class for SBUF-to-SBUF all-gather kernels."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "m, k, dtype, tp_degree",
        [
            # Basic tests
            pytest.param(128, 512, nl.bfloat16, 8, id="m128_k512_bf16_tp8"),
            pytest.param(64, 1024, nl.bfloat16, 8, id="m64_k1024_bf16_tp8"),
            pytest.param(128, 2048, nl.bfloat16, 8, id="m128_k2048_bf16_tp8"),
            pytest.param(96, 512, nl.bfloat16, 16, id="m96_k512_bf16_tp16"),
            # dtype variations
            pytest.param(128, 512, np.float32, 8, id="m128_k512_fp32_tp8"),
            pytest.param(64, 1024, np.float16, 8, id="m64_k1024_fp16_tp8"),
            # Different TP degrees
            pytest.param(128, 256, nl.bfloat16, 64, id="m128_k256_bf16_tp64"),
            pytest.param(128, 256, nl.bfloat16, 32, id="m128_k256_bf16_tp32"),
            # Non-power-of-2 k
            pytest.param(128, 384, nl.bfloat16, 16, id="m128_k384_bf16_tp16"),
        ],
    )
    def test_allgather_sb2sb(
        self,
        test_manager: Orchestrator,
        m: int,
        k: int,
        dtype: np.dtype,
        tp_degree: int,
    ):
        """Test basic SBUF-to-SBUF all-gather kernel."""
        np.random.seed(42)
        # Each rank has different input data
        x_global = np.random.randn(tp_degree, m, k).astype(dtype)
        replica_groups = ReplicaGroup([list(range(tp_degree))])

        def create_inputs(rank_id: int):
            return {
                "inp": x_global[rank_id],
                "replica_groups": replica_groups,
                "tp_degree": tp_degree,
            }

        def create_golden(rank_id: int):
            # all_gather concatenates all ranks' data along k dimension
            gathered = np.concatenate([x_global[r] for r in range(tp_degree)], axis=1)
            return {"out": gathered}

        test_manager.execute(
            KernelArgs(
                kernel_func=allgather_sb2sb,
                compiler_input=CompilerArgs(logical_nc_config=1),
                kernel_input=PerRankLazyInputGenerator(create_inputs),
                inference_args=InferenceArgs(collective_ranks=tp_degree),
                validation_args=ValidationArgs(
                    golden_output=PerRankLazyGoldenGenerator(create_golden),
                    absolute_accuracy=1e-3,
                    relative_accuracy=1e-3,
                ),
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "m, k, dtype, tp_degree, lnc",
        [
            # Single tile cases (m <= 128)
            pytest.param(128, 512, nl.bfloat16, 4, 2, id="m128_k512_bf16_tp4_lnc2"),
            pytest.param(64, 1024, nl.bfloat16, 4, 2, id="m64_k1024_bf16_tp4_lnc2"),
            # Multi-tile cases (m > 128, m % 128 == 0)
            pytest.param(256, 512, nl.bfloat16, 8, 1, id="m256_k512_bf16_tp8_lnc1"),
            pytest.param(256, 512, nl.bfloat16, 4, 2, id="m256_k512_bf16_tp4_lnc2"),
            pytest.param(512, 1024, nl.bfloat16, 8, 1, id="m512_k1024_bf16_tp8_lnc1"),
            pytest.param(512, 1024, nl.bfloat16, 8, 2, id="m512_k1024_bf16_tp8_lnc2"),
            # dtype variations
            pytest.param(256, 512, np.float32, 8, 1, id="m256_k512_fp32_tp8_lnc1"),
            pytest.param(512, 1024, np.float16, 4, 2, id="m512_k1024_fp16_tp4_lnc2"),
            pytest.param(256, 512, nl.bfloat16, 8, 2, id="m256_k512_bf16_tp8_lnc2"),
        ],
    )
    def test_allgather_sb2sb_tiled(
        self,
        test_manager: Orchestrator,
        m: int,
        k: int,
        dtype: np.dtype,
        tp_degree: int,
        lnc: int,
    ):
        """Test tiled SBUF-to-SBUF all-gather kernel with LNC support."""
        np.random.seed(42)
        # Each rank has different input data
        x_global = np.random.randn(tp_degree, m, k).astype(dtype)
        replica_groups = ReplicaGroup([list(range(tp_degree))])

        def create_inputs(rank_id: int):
            return {
                "inp": x_global[rank_id],
                "replica_groups": replica_groups,
                "tp_degree": tp_degree,
            }

        def create_golden(rank_id: int):
            # all_gather concatenates all ranks' data along k dimension
            gathered = np.concatenate([x_global[r] for r in range(tp_degree)], axis=1)
            return {"result": gathered}

        test_manager.execute(
            KernelArgs(
                kernel_func=allgather_sb2sb_tiled,
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
