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

# ty: ignore — vendored GPT-OSS test; generate_inputs optional-arg idioms trip ty's type mapping
"""Multi-rank EP unit test for the GPT-OSS MXFP4 MoE decode block with collectives.

Exercises ``moe_block_dp_ep_kernel`` — the MoE block wrapped with its dense
cross-DP EP dispatch/combine collectives (AllGather -> moe_block_tkg ->
ReduceScatter) — across **8 ranks at LNC=1, one expert per rank**.

Config (EP8, GPT-OSS-120B MoE shapes):
  * 8 ranks, num_global_experts=8, num_local_experts=1 (rank r owns expert r),
    top_k=4, H=3072 (padded from 2880), I=3072, MXFP4 experts.
  * T=128 decode tokens, token-sharded 16/rank on entry (SP), gathered to the
    full 128 for the expert compute, reduce-scattered back to 16/rank on exit.

The reference (``moe_block_dp_ep_torch_ref``) runs the identical collective
structure via torch.distributed (SimDistAdapter in sim / real backend on
hardware), so the golden validates the *combined* collective+compute result, not
just the local MoE math. On ``trn3pds-pdx10-1`` all 8 ranks run and the summed,
re-sharded output is diffed at the MXFP4 low-precision tolerance.
"""

import nki.language as nl
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.moe_block_dp_ep import (
    moe_block_dp_ep_kernel,
)
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.moe_block_dp_ep_torch import (
    moe_block_dp_ep_torch_ref,
)

# Reuse the maintained MoE MXFP4 input builder.
from test.integration.nkilib.core.moe_block.test_moe_block_tkg import generate_inputs
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

# ── EP8 GPT-OSS MoE constants ────────────────────────────────────────────────
_NUM_RANKS = 8  # EP degree == number of experts (1 expert per rank)
_T = 128  # decode tokens (gathered), 16 per rank on entry/exit
_H = 3072  # hidden, padded from 2880 (kernel needs H % 512 == 0)
_H_ACTUAL = 2880  # true GPT-OSS hidden width (RMSNorm divisor)
_I = 3072  # intermediate, padded from 2880
_TOP_K = 4
_EPS = 1e-5
_GATE_CLAMP_UPPER = 7.0  # GPT-OSS SwiGLU: gate <= 7
_UP_CLAMP_UPPER = 8.0  # up in [-6, 8] == clamp(up, -7, 7) + 1 (bias-before-clamp)
_UP_CLAMP_LOWER = -6.0
_MOE_WEIGHT_DTYPE = nl.float4_e2m1fn_x4  # MXFP4
_INPUT_DTYPE = nl.float16

RANKS_LNC_PARAM_NAMES = "collective_ranks, logical_nc_config"
RANKS_LNC = [(_NUM_RANKS, 1)]  # 8 ranks, LNC=1
_RANKS_LNC_ABBREVS = {"collective_ranks": "ranks", "logical_nc_config": "lnc"}


def _build_global_moe_inputs():
    """Build one global 8-expert MXFP4 MoE input set (shared across ranks).

    Uses the maintained ``generate_inputs`` all-expert MXFP4 path with
    num_global==num_local==8, then applies the GPT-OSS "+1" up-bias fold. Returns
    the dict; per-rank slicing happens in the input generator.
    """
    inputs = generate_inputs(
        batch=_T,
        seqlen=1,
        hidden=_H,
        hidden_actual=_H_ACTUAL,
        intermediate=_I,
        num_global_experts=_NUM_RANKS,
        num_local_experts=_NUM_RANKS,  # build all 8 experts; slice 1 per rank below
        top_k=_TOP_K,
        router_fn=None,  # set per-call in the kernel/ref (SOFTMAX)
        hidden_act_fn=None,
        expert_affinities_scaling_mode=None,
        moe_weight_dtype=_MOE_WEIGHT_DTYPE,
        input_dtype=_INPUT_DTYPE,
        has_bias=True,
        has_clamp=True,
        router_act_first=False,
        norm_topk_prob=False,
        skip_router_logits=True,
        router_mm_dtype=nl.float16,
        is_all_expert=True,
    )
    # GPT-OSS "+1" on the up branch (axis-2 index 1 = up in the MXFP4 fused bias).
    inputs["expert_gate_up_bias"][:, :, 1, ...] += 1.0
    return inputs


# Build once at import (deterministic — generate_inputs seeds np.random / rng).
_GLOBAL = _build_global_moe_inputs()


@pytest_test_metadata(name="GPT-OSS MXFP4 MoE DP-EP - collectives")
@pytest_marks(["collectives"])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
class TestMoeBlockDpEpGptOss:
    """EP8 MXFP4 MoE decode block with AllGather/ReduceScatter collectives (LNC=1)."""

    @pytest.mark.fast
    @pytest_parametrize(RANKS_LNC_PARAM_NAMES, RANKS_LNC, abbrevs=_RANKS_LNC_ABBREVS)
    def test_moe_block_dp_ep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collective_ranks: int,
        logical_nc_config: int,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MXFP4 (float4_e2m1fn_x4) is only supported on TRN3.")

        replica_group = ReplicaGroup([list(range(collective_ranks))])
        T_shard = _T // collective_ranks

        # Full [T, H] token set, sharded 16/rank (SP). float16 model dtype.
        full_tokens = _GLOBAL["inp"].reshape(_T, _H)

        def create_inputs(rank_id: int):
            # This rank's token shard (SP layout).
            inp_shard = full_tokens[rank_id * T_shard : (rank_id + 1) * T_shard].copy()
            # This rank's single local expert = expert index rank_id.
            gu_w = _GLOBAL["expert_gate_up_weights"][rank_id : rank_id + 1]
            dn_w = _GLOBAL["expert_down_weights"][rank_id : rank_id + 1]
            gu_s = _GLOBAL["expert_gate_up_weights_scale"][rank_id : rank_id + 1]
            dn_s = _GLOBAL["expert_down_weights_scale"][rank_id : rank_id + 1]
            gu_b = _GLOBAL["expert_gate_up_bias"][rank_id : rank_id + 1]
            dn_b = _GLOBAL["expert_down_bias"][rank_id : rank_id + 1]
            # moe_block_tkg accepts float4_e2m1fn_x4 weights natively (the uint16
            # NxD hand-off is not required here), so both kernel and torch ref use
            # the MX-typed weights directly — no bitcast needed.
            return {
                "inp_shard": inp_shard,
                "gamma": _GLOBAL["gamma"],
                "router_weights": _GLOBAL["router_weights"],
                "expert_gate_up_weights": gu_w,
                "expert_down_weights": dn_w,
                "expert_gate_up_weights_scale": gu_s,
                "expert_down_weights_scale": dn_s,
                "router_bias": _GLOBAL["router_bias"],
                "expert_gate_up_bias": gu_b,
                "expert_down_bias": dn_b,
                "rank_id": np.array([[rank_id]], dtype=np.uint32),
                "replica_group": replica_group,
                "num_ranks": collective_ranks,
                "top_k": _TOP_K,
                "eps": _EPS,
                "hidden_actual": _H_ACTUAL,
                "gate_clamp_upper_limit": _GATE_CLAMP_UPPER,
                "up_clamp_upper_limit": _UP_CLAMP_UPPER,
                "up_clamp_lower_limit": _UP_CLAMP_LOWER,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_block_dp_ep_kernel,
            torch_ref=moe_block_dp_ep_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=logical_nc_config,
                platform_target=platform_target,
                additional_cmd_args=["--enable-ocp-compliant-scale-computation"],
            ),
            output_keys=["out"],
            rtol=5e-2,  # MXFP4 low-precision tolerance
            atol=1e-2,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
