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

"""Multi-rank TP unit test for the GPT-OSS attention decode block with collectives.

Exercises ``attention_block_tp_kernel`` — the attention block wrapped with its TP
dispatch/combine collectives (AllGather -> attention_block_tkg -> ReduceScatter) —
across a **TP8 group at LNC=1** for two data-parallel degrees:

  * **DP1TP8**  — 8 ranks, one TP group ``[[0..7]]``.
  * **DP2TP8**  — 16 ranks, two independent TP groups ``[[0..7],[8..15]]``. DP is
    pure batch data-parallel (disjoint requests, private KV, no cross-DP
    collective), so the second replica gets independently-seeded token/KV data
    while its TP8 AllGather/ReduceScatter stay within its own subgroup.

Config (GPT-OSS-120B attention shard per TP rank):
  * q_heads=8 (1/8 of 64), kv_heads=1 (1/8 of 8, replicated per rank), d_head=64,
    H=3072/2880, S_ctx=256, batch=8, S_tkg=1 (B*S_tkg=8 divisible by TP=8).
  * Non-transposed layout (transposed_in requires LNC>=2), attention sinks on.

Each rank's O-proj is row-parallel over its 8 heads, so ``attention_block_tkg``
emits a per-rank **partial**; the ReduceScatter sums the 8 partials into the true
multi-head output and re-shards over tokens. The golden runs the identical
collective structure via torch.distributed, so the summed/re-sharded result is
validated end-to-end on ``trn3pds-pdx10-1``.
"""

import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.attention_block_tp import (
    attention_block_tp_kernel,
)
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.attention_block_tp_torch import (
    attention_block_tp_torch_ref,
)

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg import (
    generate_kernel_inputs,
)
from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_utils import (
    AttnBlkTestConfig,
)
from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

# ── GPT-OSS-120B TP8 attention shard config (per rank) ───────────────────────
_TP = 8
_CFG = AttnBlkTestConfig(
    batch=8,  # B*S_tkg = 8, divisible by TP=8 (SP shard / RS chunk)
    q_heads=8,  # TP8 shard of 64 q heads
    kv_heads=1,  # TP8 shard of 8 kv heads (replicated per rank)
    d_head=64,
    H=3072,
    H_actual=2880,
    S_ctx=256,  # short context keeps the multi-rank run fast
    S_max_ctx=256,
    S_tkg=1,
    lnc=1,
    rmsnorm_X=False,  # attention isolated from the block's input_layernorm
    test_sink=True,  # GPT-OSS per-head attention sinks
)

# DP degree -> (collective_ranks, replica-group subgroups).
DP_PARAM_NAMES = "dp, collective_ranks, logical_nc_config"
DP_PARAMS = [
    (1, _TP, 1),  # DP1TP8: 8 ranks, one TP group
    (2, 2 * _TP, 1),  # DP2TP8: 16 ranks, two independent TP groups
]
_DP_ABBREVS = {"dp": "dp", "collective_ranks": "ranks", "logical_nc_config": "lnc"}


# Keys of generate_kernel_inputs that attention_block_tp_kernel/_torch_ref consume.
_KERNEL_KEYS = (
    "W_qkv",
    "bias_qkv",
    "W_out",
    "bias_out",
    "cos",
    "sin",
    "sink",
    "K_cache",
    "V_cache",
    "active_blocks_table",
    "attention_mask",
    "kv_cache_update_idx",
    "pos_ids",
    "swa_start_pos_ids",
)


@pytest_test_metadata(name="GPT-OSS Attention TP - collectives")
@pytest_marks(["collectives"])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
class TestAttentionBlockTpGptOss:
    """TP8 attention decode block with AllGather/ReduceScatter collectives (LNC=1)."""

    @pytest.mark.fast
    @pytest_parametrize(DP_PARAM_NAMES, DP_PARAMS, abbrevs=_DP_ABBREVS)
    def test_attention_block_tp(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        dp: int,
        collective_ranks: int,
        logical_nc_config: int,
    ):
        # Two independent TP8 subgroups for DP2; one for DP1.
        subgroups = [list(range(d * _TP, (d + 1) * _TP)) for d in range(dp)]
        replica_group = ReplicaGroup(subgroups)
        T = _CFG.batch * _CFG.S_tkg
        T_shard = T // _TP
        H = _CFG.H
        softmax_scale = float(_CFG.d_head**-0.5)

        def create_inputs(rank_id: int):
            # tp_rank within this rank's DP replica; dp_replica selects the data seed.
            tp_rank = rank_id % _TP
            dp_replica = rank_id // _TP
            # Per-rank sharded weights + KV/mask. Seed by tp_rank so all TP ranks in
            # a replica share the SAME token set (X/cos/sin) while owning DIFFERENT
            # head slices; offset by dp_replica so the two DP replicas get different
            # request data (pure batch DP).
            ki = generate_kernel_inputs(_CFG, seed=tp_rank + 1000 * dp_replica)
            X_full = ki["X"].reshape(T, H)
            # This rank's SP token shard (all TP ranks reconstruct the full set via AllGather).
            X_shard = X_full[tp_rank * T_shard : (tp_rank + 1) * T_shard].copy()

            out = {"X_shard": X_shard}
            for k in _KERNEL_KEYS:
                out[k] = ki[k]
            # update_cache=False (see kernel): no in-place KV write-back, so K_cache/
            # V_cache are plain read-only inputs — no .must_alias_input needed.
            out.update(
                replica_group=replica_group,
                num_ranks=_TP,
                B=_CFG.batch,
                S_tkg=_CFG.S_tkg,
                X_hidden_dim_actual=_CFG.H_actual,
                softmax_scale=softmax_scale,
            )
            return out

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_block_tp_kernel,
            torch_ref=attention_block_tp_torch_ref,
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
            rtol=2e-2,
            atol=1e-2,
            inference_args=InferenceArgs(collective_ranks=collective_ranks, enable_determinism_check=False),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
