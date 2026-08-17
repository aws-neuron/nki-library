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

"""Direct: attention kernel-WITH-COLLECTIVES vs the HF golden, multi-rank on HW.

Diffs ``attention_block_tp_kernel`` (AllGather -> attention_block_tkg ->
ReduceScatter) DIRECTLY against a golden-backed distributed reference that runs
the identical collective structure around the HF-bit-validated golden oracle
``gptoss_mxfp4_golden.kernels.attention_decode`` (see
``attention_block_tp_golden_torch_ref``).

This is the piece the other attention tests each cover only half of:
  * ``test_attention_golden_direct_gptoss.py`` — bare kernel vs golden, SINGLE rank
    (validates the math, not the collectives).
  * ``test_attention_block_tp_gptoss.py`` — kernel+collectives vs nkilib's OWN ref
    (validates the collectives, but against nkilib's ref, not the golden).
Here the kernel+collectives is validated against the GOLDEN math directly, at
DP1TP8 (8 ranks) and DP2TP8 (16 ranks) on ``trn3pds-pdx10-1``.

The golden is single-device; the golden-backed ref wraps it in the SAME TP
AllGather/ReduceScatter as the kernel, so the summed/re-sharded output is the
true multi-head attention output. Config is the per-rank GPT-OSS TP8 shard
(q_heads=8, kv_heads=1, d_head=64) with paged block KV (the golden requires a
block_table) and per-head sinks, ``update_cache=False`` to isolate the attention
math from the KV write.
"""

import numpy as np
import pytest
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.distributed_adapter import get_pg
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.attention_block_tp import (
    attention_block_tp_kernel,
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

# The vendored HF-bit-validated golden oracle (test tree, relative import).
from .gptoss_mxfp4_golden import kernels as _golden


def attention_block_tp_golden_torch_ref(
    X_shard,
    W_qkv,
    bias_qkv,
    W_out,
    bias_out,
    cos,
    sin,
    sink,
    K_cache,
    V_cache,
    active_blocks_table,
    attention_mask,
    kv_cache_update_idx,
    pos_ids,
    swa_start_pos_ids,
    replica_group,
    num_ranks,
    B,
    S_tkg,
    X_hidden_dim_actual,
    softmax_scale,
):
    """GOLDEN-backed distributed ref: same signature as attention_block_tp_kernel.

    Mirrors the kernel's TP collectives (AllGather -> golden.attention_decode(this
    rank's head shard) -> ReduceScatter) but the per-rank math is the HF golden
    oracle. Each rank's row-parallel W_out slice makes the golden's O-proj a
    per-rank partial; the ReduceScatter sums them into the full multi-head output.
    """
    pg = get_pg(replica_group)
    dtype = X_shard.dtype
    _, H = X_shard.shape

    def _t(a):
        if a is None:
            return None
        if isinstance(a, torch.Tensor):
            return a.float()
        s = str(a.dtype)
        if "bfloat16" in s or "float8" in s or "float16" in s:
            return torch.from_numpy(a.astype(np.float32))
        if a.dtype == np.uint32:
            return torch.from_numpy(a.astype(np.int64))
        return torch.from_numpy(a.copy())

    # Dispatch: AllGather[TP] the SP token shards -> full [T, H].
    shard_t = torch.from_numpy(X_shard.astype(np.float32))
    gathered = [torch.zeros_like(shard_t) for _ in range(num_ranks)]
    dist.all_gather(gathered, shard_t, group=pg)
    Xt = torch.cat(gathered, dim=0).reshape(B, S_tkg, H)

    # Adapt this rank's tensors to the golden attention_decode layout.
    Wqkv = _t(W_qkv)
    bqkv = _t(bias_qkv).reshape(-1) if bias_qkv is not None else None
    d_head = K_cache.shape[-1]
    Kc, Vc = _t(K_cache), _t(V_cache)
    if Kc.dim() == 3:  # nkilib [nb, BL, D] -> golden [nb, kv_heads=1, BL, D]
        Kc, Vc = Kc.unsqueeze(1), Vc.unsqueeze(1)
    kv_heads = Kc.shape[1]
    num_q_heads = Wqkv.shape[1] // d_head - 2 * kv_heads
    cos_g = _t(cos).permute(1, 2, 0).reshape(B * S_tkg, d_head // 2)
    sin_g = _t(sin).permute(1, 2, 0).reshape(B * S_tkg, d_head // 2)
    pos_flat = _t(pos_ids).reshape(-1).long()
    sliding_window = None
    if swa_start_pos_ids is not None:
        start = _t(swa_start_pos_ids).reshape(-1).long()
        sliding_window = int((pos_flat - start).max().item()) + 1
    btab = _t(active_blocks_table).int()
    slot = torch.zeros(B * S_tkg, dtype=torch.long)
    sink_g = _t(sink).reshape(-1) if sink is not None else None
    scale = float(softmax_scale) if softmax_scale is not None else float(d_head**-0.5)

    # Compute this rank's head-shard attention via the GOLDEN oracle (partial).
    partial = _golden.attention_decode(
        X=Xt,
        W_qkv=Wqkv,
        bias_qkv=bqkv,
        num_q_heads=num_q_heads,
        num_kv_heads=kv_heads,
        head_dim=d_head,
        cos=cos_g,
        sin=sin_g,
        K_cache=Kc.clone(),
        V_cache=Vc.clone(),
        block_table=btab,
        slot_mapping=slot,
        pos_ids=pos_flat,
        sliding_window=sliding_window,
        sink=sink_g,
        softmax_scale=scale,
        W_out=_t(W_out),
        bias_out=_t(bias_out).reshape(-1) if bias_out is not None else None,
        update_cache=False,
    ).to(torch.float32)  # [T, H] partial

    # Combine: ReduceScatter[TP] the partials -> [T_shard, H].
    chunks = list(partial.chunk(num_ranks, dim=0))
    out = torch.zeros_like(chunks[0])
    dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM, group=pg)
    return {"out": out.numpy().astype(dtype)}


# ── GPT-OSS-120B TP8 attention shard (per rank), paged block KV. ─────────────
_TP = 8


def _cfg(sliding_window):
    return AttnBlkTestConfig(
        batch=8,
        q_heads=8,
        kv_heads=1,
        d_head=64,
        H=3072,
        H_actual=2880,
        S_ctx=256,
        S_max_ctx=256,
        S_tkg=1,
        lnc=1,
        block_len=32,  # paged block KV (golden oracle requires a block_table)
        rmsnorm_X=False,
        test_sink=True,
        use_pos_id=True,  # golden builds its mask from pos_ids; match that path
        sliding_window=sliding_window if sliding_window is not None else 0,
        update_cache=False,  # isolate attention math from the post-attn KV write
    )


# (dp, collective_ranks, lnc, sliding_window)
DP_PARAM_NAMES = "dp, collective_ranks, logical_nc_config, sliding_window"
DP_PARAMS = [
    (1, _TP, 1, None),  # DP1TP8, full attention
    (2, 2 * _TP, 1, None),  # DP2TP8, full attention
    (1, _TP, 1, 128),  # DP1TP8, sliding-window
]
_DP_ABBREVS = {"dp": "dp", "collective_ranks": "ranks", "logical_nc_config": "lnc", "sliding_window": "sw"}

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


@pytest_test_metadata(name="GPT-OSS Attention TP vs golden - collectives")
@pytest_marks(["collectives"])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
class TestAttentionBlockTpGolden:
    """TP8 attention+collectives validated directly against the HF golden (LNC=1)."""

    @pytest.mark.fast
    @pytest_parametrize(DP_PARAM_NAMES, DP_PARAMS, abbrevs=_DP_ABBREVS)
    def test_attention_block_tp_vs_golden(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        dp: int,
        collective_ranks: int,
        logical_nc_config: int,
        sliding_window,
    ):
        cfg = _cfg(sliding_window)
        subgroups = [list(range(d * _TP, (d + 1) * _TP)) for d in range(dp)]
        replica_group = ReplicaGroup(subgroups)
        T = cfg.batch * cfg.S_tkg
        T_shard = T // _TP
        H = cfg.H
        softmax_scale = float(cfg.d_head**-0.5)

        def create_inputs(rank_id: int):
            tp_rank = rank_id % _TP
            dp_replica = rank_id // _TP
            # Seed by tp_rank so all TP ranks in a replica share the SAME tokens
            # (X/cos/sin) while owning DIFFERENT head slices; offset by dp_replica
            # so the two DP replicas get independent request data (pure batch DP).
            ki = generate_kernel_inputs(cfg, seed=tp_rank + 1000 * dp_replica)
            X_full = ki["X"].reshape(T, H)
            X_shard = X_full[tp_rank * T_shard : (tp_rank + 1) * T_shard].copy()

            out = {"X_shard": X_shard}
            for k in _KERNEL_KEYS:
                out[k] = ki[k]
            out.update(
                replica_group=replica_group,
                num_ranks=_TP,
                B=cfg.batch,
                S_tkg=cfg.S_tkg,
                X_hidden_dim_actual=cfg.H_actual,
                softmax_scale=softmax_scale,
            )
            return out

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attention_block_tp_kernel,
            torch_ref=attention_block_tp_golden_torch_ref,  # <-- the GOLDEN, wrapped in TP collectives
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
