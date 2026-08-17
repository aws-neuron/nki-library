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

# ty: ignore — vendored GPT-OSS test; golden/optional-arg idioms trip ty's type mapping
"""Full non-SWA GPT-OSS decoder layer vs the HF golden, multi-rank on hardware.

Diffs the fused ``gpt_oss_decode_layer_dp_tp_ep_kernel`` (AllGather[TP] -> attn
(input_layernorm) -> RS[TP] -> +residual -> AllGather[EP] -> MoE (post-attn norm)
-> RS[EP] -> +residual) DIRECTLY against the vendored HF-bit-validated **whole-
layer** oracle ``gptoss_mxfp4_golden.gptoss_mxfp4_decode_layer`` — no hand-rolled
RMSNorm or residual adds: the entire layer body (both RMSNorms, attention_decode,
moe_block_tkg, and both residual adds) is the golden's own ``forward_decode``.

Sharding: **DP2 x TP8 attention + EP16 MoE, 16 experts**, LNC=1, on 16 ranks.
  * tp_replica_group = [[0..7],[8..15]] (per-DP-replica TP group)
  * ep_replica_group = [[0..15]] (wide-EP over all ranks)

Why one whole-layer oracle call *per DP replica* reproduces the fused kernel:
decode attention and MoE are both **token-independent per row** — each query
token attends only to its own KV cache (S_tkg=1, private sequence), and the MoE
routes/experts each token independently (no cross-token op). So:
  * running the oracle over a DP replica's B_attn tokens gives the same per-token
    result as the kernel's per-replica attention, and
  * the oracle's MoE over those B_attn tokens (with ALL ep_size experts) equals
    the kernel's wide-EP MoE for those tokens (the kernel gathers all global
    tokens only so each rank's single sharded expert can contribute; the per-token
    output is unchanged).
The ref reconstructs the replica's tokens (AllGather[TP]), the FULL 64-head
attention weights (AllGather[TP] of the per-rank Q-head slices), and the full
expert set (AllGather[EP], MXFP4->dense via the CPU-verified adapter), calls the
oracle once, and slices this rank's token (its TP position within the replica).

Genuine 64-head attention: each TP rank owns a DISTINCT 8-Q-head shard (its own Q
columns of W_qkv, W_out rows, sinks, and the head-broadcast mask heads) sharing
the single KV head + cache + tokens. The kernel's RS[TP] sums the ``tp_size``
distinct 8-head O-proj partials into the true 64-head attention output. The ref
rebuilds the full 64-head W_qkv/W_out/sinks and runs one coherent 64-head
attention. This O-proj row-parallel identity (full 64-head == sum of per-rank
8-head partials) is CPU-proven to rel-L2 ~1e-6 (bias-free here: test_bias=False,
so no once-vs-tp_size O-proj-bias ambiguity). No W_out scaling.

The oracle runs in fp32 (``config.torch_dtype=float32`` + fp32 inputs), matching
the fp32 reference math the half-tests validated against.

MXFP4 experts -> TRN3 only. Kernel stream is fp16 (matches the kernel's
router_mm_dtype and the proven MoE MXFP4 path; attention is dtype-generic).
"""

import nki.language as nl
import numpy as np
import pytest
import torch
import torch.distributed as dist
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.distributed_adapter import get_pg
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.gpt_oss_decode_layer_dp_tp_ep import (
    gpt_oss_decode_layer_dp_tp_ep_kernel,
)

from test.integration.nkilib.core.moe_block.test_moe_block_tkg import generate_inputs
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

from .gptoss_mxfp4_golden import GptOssConfig, gptoss_mxfp4_decode_layer
from .mxfp4_dense_adapter import (
    packed_mx_to_dense_down,
    packed_mx_to_dense_gate_up,
    unpack_gate_up_bias,
)

# ── Config: shared H for attention + MoE (H%128 for attn, H%512 for MX MoE). ──
_TP = 8
_H = 512
_I = 512
_D_HEAD = 64
_Q_HEADS = 8  # per rank (TP8 head shard)
_KV_HEADS = 1
_S_CTX = 256
_S_TKG = 1
_TOP_K = 4
_EPS = 1e-5
_SWIGLU_LIMIT = 7.0
_SWIGLU_ALPHA = 1.702
_GATE_CLAMP_UPPER = 7.0
_UP_CLAMP_UPPER = 8.0
_UP_CLAMP_LOWER = -6.0
_BLOCK_LEN = 32
_MX = nl.float4_e2m1fn_x4
_DTYPE = nl.float16

# (dp, ep, collective_ranks). DP1TP8EP8 smoke first, then target DP2TP8EP16,
# then DP8TP8EP64 (max EP on a 128-core LNC=1 box; ep_group [[0..63]], 8 DP
# replicas each an 8-way TP attention group). EP128 is infeasible on this host
# (nd5 is bad -> only 120 of 128 cores healthy).
DP_PARAM_NAMES = "dp, n_experts, collective_ranks, logical_nc_config"
DP_PARAMS = [
    (1, 8, 8, 1),  # DP1TP8EP8 smoke: tp==ep==[[0..7]], isolates fused wiring
    (2, 16, 16, 1),  # DP2TP8EP16 target: tp=[[0..7],[8..15]], ep=[[0..15]]
    (4, 32, 32, 1),  # DP4TP8EP32: GPT-OSS-20B (32 experts). DP=ranks/TP=32/8=4.
    (8, 64, 64, 1),  # DP8TP8EP64 scale: tp=[[0..7],...,[56..63]], ep=[[0..63]]
]
_ABBREVS = {"dp": "dp", "n_experts": "e", "collective_ranks": "ranks", "logical_nc_config": "lnc"}


_FULL_Q_HEADS = _TP * _Q_HEADS  # 64: the whole model's Q-head count (TP8 x 8/rank)


def _attn_cfg_full(batch):
    # ONE coherent full-model attention set per DP replica: 64 Q-heads sharing a
    # single KV head, cache, tokens, cos/sin, pos_ids, and block table. Each TP
    # rank slices its own 8-Q-head shard (distinct Q columns / W_out rows / sinks)
    # while the shared KV state stays identical across the replica -- exactly the
    # physical setup a genuine 64-head attention decodes. batch == B_attn (tokens
    # per DP replica); each rank takes its 1/TP SP token shard.
    return AttnBlkTestConfig(
        batch=batch,
        q_heads=_FULL_Q_HEADS,
        kv_heads=_KV_HEADS,
        d_head=_D_HEAD,
        H=_H,
        H_actual=_H,
        S_ctx=_S_CTX,
        S_max_ctx=_S_CTX,
        S_tkg=_S_TKG,
        lnc=1,
        block_len=_BLOCK_LEN,
        rmsnorm_X=True,
        test_sink=True,
        use_pos_id=True,
        sliding_window=0,
        update_cache=False,
        dtype=_DTYPE,
    )


def _to_t(a):
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


def make_golden_ref(tp_group, ep_group, tp_size, ep_size):
    """Golden-backed distributed ref that calls the WHOLE-LAYER oracle.

    Per DP replica: AllGather[TP] the token shards into the replica's B_attn
    pre-norm tokens, AllGather[TP] the distinct per-rank Q-head weight slices into
    the FULL 64-head W_qkv/W_out/sinks (the shared single KV head is identical on
    every rank), AllGather[EP] the MXFP4->dense expert set into all ep_size
    experts, then call ``gptoss_mxfp4_decode_layer`` ONCE (its own two RMSNorms,
    attention_decode, moe_block_tkg, and both residual adds) and slice out this
    rank's token. No RMSNorm / residual math is reimplemented here.

    Step B (genuine 64-head attention): each TP rank owns a DISTINCT 8-Q-head
    slice (its own Q columns of W_qkv, W_out rows, and sinks) sharing the single
    KV head. The kernel's RS[TP] sums the ``tp_size`` distinct 8-head O-proj
    partials into the true 64-head output; the ref reproduces that by rebuilding
    the full 64-head weights (concatenate the Q slices along the head axis) and
    running one coherent 64-head attention. This exact O-proj row-parallel
    identity (full == sum of per-rank partials, bias_out added once) is CPU-proven
    to rel-L2 ~1e-6. No ``tp_size`` W_out scaling (that was the Step A stand-in
    for identical partials).
    """

    def ref(
        x_shard,
        input_layernorm_weight,
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
        post_attn_layernorm_weight,
        router_weights,
        router_bias,
        expert_gate_up_weights,
        expert_down_weights,
        expert_gate_up_weights_scale,
        expert_down_weights_scale,
        expert_gate_up_bias,
        expert_down_bias,
        rank_id,
        tp_replica_group,
        ep_replica_group,
        tp_size,
        ep_size,
        B_attn,
        S_tkg,
        top_k,
        eps,
        hidden_actual,
        softmax_scale,
        gate_clamp_upper_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
    ):
        tp_pg = get_pg(tp_replica_group)
        ep_pg = get_pg(ep_replica_group)
        dtype = x_shard.dtype
        T_shard, H = x_shard.shape
        I = _I

        # ── AllGather[TP]: reconstruct this replica's B_attn pre-norm tokens. ──
        x_t = torch.from_numpy(x_shard.astype(np.float32))
        ag_tp = [torch.zeros_like(x_t) for _ in range(tp_size)]
        dist.all_gather(ag_tp, x_t, group=tp_pg)
        hidden = torch.cat(ag_tp, dim=0)  # [B_attn, H] fp32, pre-norm

        # ── Attention tensors (adapt kernel layout to the oracle's contract). ──
        d_head = K_cache.shape[-1]
        Wqkv_r = _to_t(W_qkv)  # [H, 8*D + 2*kv_size] this rank's shard
        Kc, Vc = _to_t(K_cache), _to_t(V_cache)
        if Kc.dim() == 3:  # [nb, BL, D] -> [nb, 1, BL, D]
            Kc, Vc = Kc.unsqueeze(1), Vc.unsqueeze(1)
        kv_heads = Kc.shape[1]
        q_per_rank = Wqkv_r.shape[1] // d_head - 2 * kv_heads  # 8
        qd_r = q_per_rank * d_head
        cos_g = _to_t(cos).permute(1, 2, 0).reshape(B_attn * S_tkg, d_head // 2)
        sin_g = _to_t(sin).permute(1, 2, 0).reshape(B_attn * S_tkg, d_head // 2)
        pos_flat = _to_t(pos_ids).reshape(-1).long()
        slot = torch.zeros(B_attn * S_tkg, dtype=torch.long)  # cache write is discarded (clone)
        btab = _to_t(active_blocks_table).int()

        # ── AllGather[TP]: rebuild the FULL 64-head W_qkv / W_out / sinks. ──
        # Each rank holds its Q slice (columns [:qd_r]) + the shared KV columns; the
        # full model's Q columns are the ranks' Q slices concatenated head-major.
        def _ag_tp(local):  # torch tensor -> list over TP
            buf = [torch.zeros_like(local) for _ in range(tp_size)]
            dist.all_gather(buf, local.contiguous(), group=tp_pg)
            return buf

        kv_size = kv_heads * d_head
        q_slices = _ag_tp(Wqkv_r[:, :qd_r].contiguous())  # tp_size x [H, qd_r]
        Wout_r = _to_t(W_out)  # [qd_r, H] this rank's O-proj rows
        wout_slices = _ag_tp(Wout_r)
        Kcol, Vcol = Wqkv_r[:, qd_r : qd_r + kv_size], Wqkv_r[:, qd_r + kv_size :]  # shared KV cols
        Wqkv = torch.cat(q_slices + [Kcol, Vcol], dim=1)  # [H, 64*D + 2*kv_size]
        Wout = torch.cat(wout_slices, dim=0)  # [64*D, H]
        num_q_heads = tp_size * q_per_rank  # 64
        q_size = num_q_heads * d_head
        # QKV bias: same Q-slice/shared-KV structure; O-proj bias added once (real full model bias).
        if bias_qkv is not None:
            bqkv_r = _to_t(bias_qkv).reshape(-1)
            bq_slices = _ag_tp(bqkv_r[:qd_r].contiguous())
            bqkv = torch.cat(bq_slices + [bqkv_r[qd_r : qd_r + kv_size], bqkv_r[qd_r + kv_size :]], dim=0)
        else:
            bqkv = torch.zeros(q_size + 2 * kv_size)
        bout = _to_t(bias_out).reshape(-1) if bias_out is not None else torch.zeros(H)
        # sinks: one per Q head; gather the per-rank 8-head slices into 64.
        if sink is not None:
            sink_slices = _ag_tp(_to_t(sink).reshape(-1))
            sink_g = torch.cat(sink_slices, dim=0)  # [64]
        else:
            sink_g = None

        # ── MoE tensors: AllGather[EP] the MXFP4->dense expert set (all ep_size). ──
        gu = packed_mx_to_dense_gate_up(expert_gate_up_weights, expert_gate_up_weights_scale, H, I)
        dn = packed_mx_to_dense_down(expert_down_weights, expert_down_weights_scale, H, I)
        gub = unpack_gate_up_bias(expert_gate_up_bias, H, I)
        dnb = _to_t(expert_down_bias)

        def _ag_e(local):
            buf = [torch.zeros_like(local) for _ in range(ep_size)]
            dist.all_gather(buf, local.contiguous(), group=ep_pg)
            return torch.cat(buf, dim=0)

        all_gu, all_dn, all_gub, all_dnb = _ag_e(gu), _ag_e(dn), _ag_e(gub), _ag_e(dnb)
        router_w = _to_t(router_weights).t().contiguous()  # [H,E] -> [E,H]
        router_b = _to_t(router_bias).reshape(-1)

        # ── One whole-layer oracle call over this replica's B_attn tokens. ──
        # fp32 config so the oracle runs pure-fp32 (matches the reference math).
        cfg = GptOssConfig(
            hidden_size=H,
            intermediate_size=I,
            num_attention_heads=num_q_heads,
            num_key_value_heads=kv_heads,
            head_dim=d_head,
            num_local_experts=ep_size,
            num_experts_per_tok=top_k,
            rms_norm_eps=eps,
            swiglu_limit=gate_clamp_upper_limit,
            swiglu_alpha=_SWIGLU_ALPHA,
            torch_dtype=torch.float32,
            sliding_window=None,
        )
        out_full = gptoss_mxfp4_decode_layer(
            cfg,
            hidden_states=hidden,
            positions=pos_flat,
            cos=cos_g,
            sin=sin_g,
            input_layernorm_weight=_to_t(input_layernorm_weight).reshape(H),
            qkv_proj_weight=Wqkv,
            qkv_proj_bias=bqkv,
            o_proj_weight=Wout,
            o_proj_bias=bout,
            sinks=sink_g,
            post_attention_layernorm_weight=_to_t(post_attn_layernorm_weight).reshape(H),
            router_weight=router_w,
            router_bias=router_b,
            gate_up_weight=all_gu,
            gate_up_bias=all_gub,
            down_weight=all_dn,
            down_bias=all_dnb,
            k_cache=Kc.clone(),
            v_cache=Vc.clone(),
            block_table=btab,
            slot_mapping=slot,
            sliding_window=None,
        ).to(torch.float32)  # [B_attn, H]

        # ── Slice this rank's token (its TP position within the replica). ──
        tp_rank = int(rank_id[0, 0]) % tp_size
        out = out_full[tp_rank * T_shard : (tp_rank + 1) * T_shard, :]
        return {"out": out.numpy().astype(dtype)}

    return ref


@pytest_test_metadata(name="GPT-OSS full decode layer DP-TP-EP vs golden")
@pytest_marks(["collectives"])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
class TestGptOssDecodeLayerGolden:
    """Fused non-SWA full layer (DPxTP attn + EP MoE) vs the HF golden, LNC=1."""

    @pytest.mark.fast
    @pytest_parametrize(DP_PARAM_NAMES, DP_PARAMS, abbrevs=_ABBREVS)
    def test_full_layer_vs_golden(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        dp: int,
        n_experts: int,
        collective_ranks: int,
        logical_nc_config: int,
    ):
        if not platform_target.is_trn3():
            pytest.skip("MXFP4 (float4_e2m1fn_x4) is only supported on TRN3.")

        tp_subgroups = [list(range(d * _TP, (d + 1) * _TP)) for d in range(dp)]
        tp_group = ReplicaGroup(tp_subgroups)
        ep_group = ReplicaGroup([list(range(collective_ranks))])
        T_shard = 1  # tokens per rank
        B_g = T_shard * collective_ranks  # global tokens
        B_attn = T_shard * _TP  # tokens per DP replica (attention gather)
        softmax_scale = float(_D_HEAD**-0.5)

        acfg = _attn_cfg_full(B_attn)  # ONE full 64-head attention set per replica
        _QD_R = _Q_HEADS * _D_HEAD  # 8*64 columns of Q per rank
        _Q_ALL = _FULL_Q_HEADS * _D_HEAD  # 64*64 total Q columns
        _KV = _KV_HEADS * _D_HEAD  # shared KV column width
        # One global MXFP4 MoE weight set (E=n_experts), sliced per rank.
        moe_global = generate_inputs(
            batch=B_g,
            seqlen=1,
            hidden=_H,
            hidden_actual=_H,
            intermediate=_I,
            num_global_experts=n_experts,
            num_local_experts=n_experts,
            top_k=_TOP_K,
            router_fn=None,
            hidden_act_fn=None,
            expert_affinities_scaling_mode=None,
            moe_weight_dtype=_MX,
            input_dtype=_DTYPE,
            has_bias=True,
            has_clamp=True,
            router_act_first=False,
            norm_topk_prob=False,
            skip_router_logits=True,
            router_mm_dtype=nl.float16,
            is_all_expert=True,
        )
        moe_global["expert_gate_up_bias"][:, :, 1, ...] += 1.0  # GPT-OSS +1

        def create_inputs(rank_id: int):
            dp_replica = rank_id // _TP
            tp_rank = rank_id % _TP
            # ONE full 64-Q-head attention set per DP replica (seeded per replica);
            # every TP rank shares the KV head/cache/tokens/cos/sin/mask, and slices
            # its OWN distinct 8-Q-head shard: Q columns of W_qkv, W_out rows, sinks,
            # and the (head-broadcast) mask heads. This makes the kernel's RS[TP] a
            # genuine 64-head attention (sum of 8 DISTINCT head-partials), not
            # tp_size copies of one 8-head partial.
            aki = generate_kernel_inputs(acfg, seed=1000 * dp_replica)
            # x_shard: this rank's SP token shard of the replica's B_attn tokens.
            X_full = aki["X"].reshape(B_attn, _H)
            x_shard = X_full[tp_rank * T_shard : (tp_rank + 1) * T_shard].copy()
            # W_qkv: this rank's 8 Q-head columns + the shared single KV head columns.
            Wqkv_full = aki["W_qkv"]  # [H, 64*D + 2*KV]
            Wqkv_r = np.ascontiguousarray(
                np.concatenate(
                    [
                        Wqkv_full[:, tp_rank * _QD_R : (tp_rank + 1) * _QD_R],
                        Wqkv_full[:, _Q_ALL : _Q_ALL + _KV],
                        Wqkv_full[:, _Q_ALL + _KV : _Q_ALL + 2 * _KV],
                    ],
                    axis=1,
                )
            )
            # W_out: this rank's O-proj rows [8*D, H]; sink: this rank's 8 heads.
            Wout_r = np.ascontiguousarray(aki["W_out"][tp_rank * _QD_R : (tp_rank + 1) * _QD_R, :])
            sink_r = np.ascontiguousarray(aki["sink"][tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS, :])
            # attention_mask: [S_tkg, B, 64, S_tkg] head-broadcast -> take this rank's 8 heads.
            mask_full = aki["attention_mask"]
            mask_r = np.ascontiguousarray(mask_full[:, :, tp_rank * _Q_HEADS : (tp_rank + 1) * _Q_HEADS, :])
            # This rank's expert (EP): global expert index == rank_id.
            e = rank_id
            return {
                "x_shard": x_shard,
                "input_layernorm_weight": aki["rmsnorm_X_gamma"],
                "W_qkv": Wqkv_r,
                "bias_qkv": aki["bias_qkv"],  # bias_qkv is None (test_bias=False)
                "W_out": Wout_r,
                "bias_out": aki["bias_out"],  # bias_out is None
                "cos": aki["cos"],
                "sin": aki["sin"],
                "sink": sink_r,
                "K_cache": aki["K_cache"],
                "V_cache": aki["V_cache"],
                "active_blocks_table": aki["active_blocks_table"],
                "attention_mask": mask_r,
                "kv_cache_update_idx": aki["kv_cache_update_idx"],
                "pos_ids": aki["pos_ids"],
                "swa_start_pos_ids": aki["swa_start_pos_ids"],
                "post_attn_layernorm_weight": moe_global["gamma"],
                "router_weights": moe_global["router_weights"],
                "router_bias": moe_global["router_bias"],
                "expert_gate_up_weights": moe_global["expert_gate_up_weights"][e : e + 1],
                "expert_down_weights": moe_global["expert_down_weights"][e : e + 1],
                "expert_gate_up_weights_scale": moe_global["expert_gate_up_weights_scale"][e : e + 1],
                "expert_down_weights_scale": moe_global["expert_down_weights_scale"][e : e + 1],
                "expert_gate_up_bias": moe_global["expert_gate_up_bias"][e : e + 1],
                "expert_down_bias": moe_global["expert_down_bias"][e : e + 1],
                "rank_id": np.array([[rank_id]], dtype=np.uint32),
                "tp_replica_group": tp_group,
                "ep_replica_group": ep_group,
                "tp_size": _TP,
                "ep_size": collective_ranks,
                "B_attn": B_attn,
                "S_tkg": _S_TKG,
                "top_k": _TOP_K,
                "eps": _EPS,
                "hidden_actual": _H,
                "softmax_scale": softmax_scale,
                "gate_clamp_upper_limit": _GATE_CLAMP_UPPER,
                "up_clamp_upper_limit": _UP_CLAMP_UPPER,
                "up_clamp_lower_limit": _UP_CLAMP_LOWER,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gpt_oss_decode_layer_dp_tp_ep_kernel,
            torch_ref=make_golden_ref(tp_group, ep_group, _TP, collective_ranks),
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
            rtol=7e-2,
            atol=1e-2,
            inference_args=InferenceArgs(collective_ranks=collective_ranks, enable_determinism_check=False),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
