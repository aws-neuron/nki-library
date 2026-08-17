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
"""Tests for ring attention unpermute kernel (striped -> contiguous)."""

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.attention.ring_attention_unpermute import ring_attention_unpermute
from nkilib_src.nkilib.experimental.attention.ring_attention_unpermute_torch import ring_attention_unpermute_torch_ref

from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    Platforms,
)
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework


class TestRingAttentionUnpermute:
    """Integration tests for ring attention unpermute kernel (striped -> contiguous)."""

    @pytest.mark.parametrize(
        "batch, d, seqlen_per_rank, cp_degree, lnc",
        [
            pytest.param(2, 128, 4096, 4, 2, id="bs2_d128_seq4096_cp4_lnc2"),
            pytest.param(2, 128, 1024, 4, 2, id="bs2_d128_seq1024_cp4_lnc2"),
            pytest.param(1, 128, 4096, 4, 1, id="bs1_d128_seq4096_cp4_lnc1"),
            pytest.param(4, 64, 2048, 2, 2, id="bs4_d64_seq2048_cp2_lnc2"),
        ],
    )
    def test_ring_attention_unpermute(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        d: int,
        seqlen_per_rank: int,
        cp_degree: int,
        lnc: int,
    ):
        """Test striped-to-contiguous unpermute against reference.

        Generates global data in natural order, stripes it across ranks,
        then verifies the kernel reconstructs the correct contiguous chunk
        for each rank.
        """
        np.random.seed(42)

        seqlen = seqlen_per_rank * cp_degree

        # Generate global data in natural position order
        global_data = np.random.randn(batch, seqlen, d).astype(np.float32)

        # Stripe-slice: rank r gets positions [r, r+cp, r+2*cp, ...]
        x_per_rank = [global_data[:, r::cp_degree, :].copy() for r in range(cp_degree)]

        replica_groups = tuple(tuple(range(cp_degree)) for _ in range(1))

        def create_inputs(rank_id: int):
            return {
                "x": x_per_rank[rank_id].astype(np.float16),
                "replica_groups": replica_groups,
                "num_workers": cp_degree,
            }

        env_vars = {"NEURON_RT_ULTRASERVER_MODE": "4"} if (platform_target.is_trn3() and cp_degree > 1) else None
        framework = CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=ring_attention_unpermute,
            torch_ref=ring_attention_unpermute_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=cp_degree,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            inference_args=InferenceArgs(collective_ranks=cp_degree, env_vars=env_vars),
            output_keys=["out_x"],
        )
