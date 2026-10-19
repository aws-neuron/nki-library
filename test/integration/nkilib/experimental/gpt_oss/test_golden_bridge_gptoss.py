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

"""CPU bridge tests: nkilib torch refs == the HF-validated GPT-OSS golden oracles.

The kernel tests in this directory diff each NKI kernel against nkilib's *own*
torch reference (``moe_block_tkg_torch_ref`` / ``AttentionBlockTkgTorchRef``).
Those refs are a separate implementation from the vendored ``gptoss_mxfp4_golden``
(which is bit-validated against HuggingFace). These tests close that gap: they
feed one identical input set to BOTH the golden oracle and the nkilib ref and
assert they agree, so transitively

    NKI kernel  ==(kernel tests, on HW)==  nkilib ref  ==(here)==  golden  ==  HuggingFace.

Everything runs on CPU in fp32 at the golden's natural *unpadded* dims (H=2880),
with dense (non-MXFP4) weights, so the comparison isolates pure math semantics
(RMSNorm, router top-k-then-softmax, SwiGLU clamp convention, sinks, RoPE, paged
KV, GQA) from low-precision / packing noise.
"""

import numpy as np
import pytest
import torch

# nkilib's own torch references (what the kernel tests diff against).
from nkilib_src.nkilib.core.moe_block.moe_block_tkg_torch import moe_block_tkg_torch_ref
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    DtypeMode,
    ExpertAffinityScaleMode,
    QuantizationType,
    RouterActFnType,
)
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg_torch import (
    AttentionBlockTkgTorchRef,
)

from test.integration.nkilib.core.attention.test_attention_tkg_utils import (
    build_active_attention_mask,
)

# The HF-validated golden oracles (vendored).
from .gptoss_mxfp4_golden import kernels as golden

_SEED = 0
_EPS = 1e-5
_SWIGLU_LIMIT = 7.0
_SWIGLU_ALPHA = 1.702


def _rng():
    g = torch.Generator()
    g.manual_seed(_SEED)
    return g


# ─────────────────────────────────────────────────────────────────────────────
# MoE bridge
# ─────────────────────────────────────────────────────────────────────────────
def test_moe_golden_matches_nkilib_ref():
    """golden.moe_block_tkg == moe_block_tkg_torch_ref (dense fp32, unpadded H).

    Both take dense ``[E,H,2,I]`` gate/up + ``[E,I,H]`` down weights with the +1
    baked into the up bias and the same SwiGLU clamps; the golden runs top-k then
    softmax (pre_norm=False) with POST_SCALE affinity, which is exactly nkilib's
    ``RouterActFnType.SOFTMAX`` + ``router_pre_norm=False`` all-expert path.
    """
    g = _rng()
    # Small but GPT-OSS-shaped: unpadded H=I, a few experts, a handful of tokens.
    T, H, I, E, top_k = 16, 256, 256, 8, 4

    hidden = torch.randn(T, H, generator=g, dtype=torch.float32) * 0.1
    gamma = torch.randn(H, generator=g, dtype=torch.float32) * 0.1 + 1.0
    router_w = torch.randn(E, H, generator=g, dtype=torch.float32) * 0.1
    router_b = torch.randn(E, generator=g, dtype=torch.float32) * 0.1
    gate_up_w = torch.randn(E, H, 2, I, generator=g, dtype=torch.float32) * (H**-0.5)
    gate_up_b = torch.randn(E, 2, I, generator=g, dtype=torch.float32) * 0.1
    gate_up_b[:, 1, :] += 1.0  # +1 baked into the up bias (GPT-OSS convention)
    down_w = torch.randn(E, I, H, generator=g, dtype=torch.float32) * (I**-0.5)
    down_b = torch.randn(E, H, generator=g, dtype=torch.float32) * 0.1

    # ── Golden oracle (HF-validated). ──
    golden_out = golden.moe_block_tkg(
        hidden_states=hidden,
        gamma=gamma,
        eps=_EPS,
        router_weight=router_w,
        router_bias=router_b,
        gate_up_weight=gate_up_w,
        gate_up_bias=gate_up_b,
        down_weight=down_w,
        down_bias=down_b,
        top_k=top_k,
        swiglu_limit=_SWIGLU_LIMIT,
        swiglu_alpha=_SWIGLU_ALPHA,
    ).to(torch.float32)

    # ── nkilib torch ref (all-expert, dense, SOFTMAX-after-topk, POST_SCALE). ──
    # Signature matches the moe_block_tkg kernel: inp is [B, S, H], weights [E,...].
    # is_all_expert + rank_id=0 with num_local==num_global==E processes all experts.
    nkilib_out_dict = moe_block_tkg_torch_ref(
        inp=hidden.reshape(1, T, H),
        gamma=gamma.reshape(1, H),
        router_weights=router_w.t().contiguous(),  # nkilib router_weights is [H, E]
        expert_gate_up_weights=gate_up_w,
        expert_down_weights=down_w,
        router_bias=router_b.reshape(1, E),
        expert_gate_up_bias=gate_up_b,
        expert_down_bias=down_b,
        eps=_EPS,
        top_k=top_k,
        router_act_fn=RouterActFnType.SOFTMAX,
        router_pre_norm=False,  # softmax AFTER top-k
        norm_topk_prob=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        hidden_act_fn=ActFnType.Swish,  # x * sigmoid(1.702 x)
        gate_clamp_upper_limit=_SWIGLU_LIMIT,  # gate <= 7
        up_clamp_upper_limit=_SWIGLU_LIMIT + 1,  # up in [-6, 8]
        up_clamp_lower_limit=-_SWIGLU_LIMIT + 1,
        skip_router_logits=True,
        is_all_expert=True,
        rank_id=np.array([[0]], dtype=np.uint32),
    )
    nkilib_out = nkilib_out_dict["out"]
    if isinstance(nkilib_out, torch.Tensor):
        nkilib_out = nkilib_out.to(torch.float32).numpy()
    nkilib_out = np.asarray(nkilib_out, dtype=np.float32).reshape(T, H)

    np.testing.assert_allclose(
        nkilib_out,
        golden_out.numpy(),
        rtol=1e-4,
        atol=1e-4,
        err_msg="nkilib MoE torch ref diverged from the HF-validated golden oracle",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Attention bridge
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sliding_window", [None, 128], ids=["full", "swa"])
def test_attention_golden_matches_nkilib_ref(sliding_window):
    """golden.attention_decode == AttentionBlockTkgTorchRef (dense fp32, paged KV).

    One canonical fp32 input set (per-rank GPT-OSS TP shape: q_heads=8, kv_heads=1,
    head_dim=64, paged block KV) is fed to BOTH. The golden builds its mask from
    ``pos_ids`` (+ ``sliding_window``); the nkilib ref reproduces the identical
    mask via its ``use_pos_id`` path (``pos_ids`` + ``swa_start_pos_ids``). RoPE
    uses the contiguous half-split (``rope_contiguous_layout=True``). ``kv_heads=1``
    makes the paged layouts line up 1:1. Runs in fp32 so the comparison is pure
    semantics (bf16 would only add rounding noise). ``update_cache=False`` isolates
    the attention math from the (post-attention) KV write.
    """
    torch.manual_seed(42)
    B, S, H = 2, 1, 256
    q_heads, kv_heads, D = 8, 1, 64
    BL, MB = 16, 8
    S_ctx = BL * MB
    num_blocks = B * MB
    I = (q_heads + 2 * kv_heads) * D
    scale = float(D**-0.5)
    dt = torch.float32  # fp32 => tight semantic comparison (no bf16 rounding)

    X = torch.randn(B, S, H, dtype=dt)
    Wqkv = (torch.randn(H, I) / H**0.5).to(dt)
    bqkv = (torch.randn(I) * 0.1).to(dt)
    cosg = (torch.rand(B * S, D // 2) * 2 - 1).to(dt)  # golden layout [B*S, D//2]
    sing = (torch.rand(B * S, D // 2) * 2 - 1).to(dt)
    Kc_g = (torch.rand(num_blocks, kv_heads, BL, D) * 2 - 1).to(dt)  # golden paged layout
    Vc_g = (torch.rand(num_blocks, kv_heads, BL, D) * 2 - 1).to(dt)
    # block_table: valid blocks packed as prefix (here fully populated), -1 pad convention.
    btab = torch.arange(B * MB, dtype=torch.int32).reshape(B, MB)
    pos = torch.full((B * S,), S_ctx - S, dtype=torch.long)  # each seq near full context
    sink = torch.rand(q_heads).float()
    Wout = (torch.randn(q_heads * D, H) * 0.5).to(dt)
    bout = (torch.randn(H) * 0.1).to(dt)
    slot = torch.zeros(B * S, dtype=torch.long)  # unused (update_cache=False)

    # ── Golden oracle (HF-validated). ──
    out_g = golden.attention_decode(
        X=X.clone(),
        W_qkv=Wqkv,
        bias_qkv=bqkv,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        head_dim=D,
        cos=cosg,
        sin=sing,
        K_cache=Kc_g.clone(),
        V_cache=Vc_g.clone(),
        block_table=btab,
        slot_mapping=slot,
        pos_ids=pos,
        sliding_window=sliding_window,
        sink=sink,
        softmax_scale=scale,
        W_out=Wout,
        bias_out=bout,
        update_cache=False,
    ).to(torch.float32)  # [B*S, H]

    # ── Adapt the SAME tensors to nkilib's layout. ──
    nk_cos = cosg.reshape(B, S, D // 2).permute(2, 0, 1).contiguous()  # [D//2, B, S]
    nk_sin = sing.reshape(B, S, D // 2).permute(2, 0, 1).contiguous()
    nk_K = Kc_g[:, 0, :, :].clone()  # drop size-1 kv-head axis -> [nb, BL, D]
    nk_V = Vc_g[:, 0, :, :].clone()
    nk_pos = pos.reshape(B, S).float()
    nk_swa = (nk_pos - sliding_window + 1).clamp(min=0) if sliding_window is not None else None
    nk_mask = build_active_attention_mask(B, q_heads, S, transposed=True).to(torch.uint8)  # [S,B,N,S]
    kv_upd = torch.zeros(B, S, dtype=torch.int64)  # unused (update_cache=False)

    ref = AttentionBlockTkgTorchRef(lnc=1)
    res = ref(
        X=X.clone(),
        X_hidden_dim_actual=None,
        rmsnorm_X_enabled=False,
        rmsnorm_X_eps=None,
        rmsnorm_X_gamma=None,
        W_qkv=Wqkv,
        bias_qkv=bqkv.reshape(1, I),
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=0.0,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        cos=nk_cos,
        sin=nk_sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        skip_attention=False,
        K_cache_transposed=False,
        active_blocks_table=btab,
        K_cache=nk_K,
        V_cache=nk_V,
        attention_mask=nk_mask,
        sink=sink.reshape(q_heads, 1),
        softmax_scale=scale,
        k_scale=None,
        v_scale=None,
        update_cache=False,
        kv_cache_update_idx=kv_upd,
        W_out=Wout,
        bias_out=bout.reshape(1, H),
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        transposed_in=False,
        out_in_sb=False,
        fp8_packed=False,
        pos_ids=nk_pos,
        swa_start_pos_ids=nk_swa,
        S_ctx=None,
        is_h_transposed_by_4=False,
        dtype_mode=DtypeMode.NON_OCP,
        KVDP=1,
        CP=1,
    )
    out_nk = res["X_out"]
    if isinstance(out_nk, torch.Tensor):
        out_nk = out_nk.to(torch.float32)
    else:
        out_nk = torch.as_tensor(np.asarray(out_nk, dtype=np.float32))
    out_nk = out_nk.reshape(B * S, H)

    torch.testing.assert_close(
        out_nk,
        out_g,
        rtol=1e-3,
        atol=1e-3,
        msg="nkilib attention torch ref diverged from the HF-validated golden oracle",
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
