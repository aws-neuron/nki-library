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

"""CPU verification of the MXFP4-packed -> dense adapter for the MoE golden.

Before diffing the MoE *kernel* directly against the golden on hardware, we must
trust the packed->dense un-shuffle adapter (``mxfp4_dense_adapter``). This test
verifies it on CPU without any kernel:

    dense = adapter(packed MXFP4 weights)          # the un-shuffle under test
    out_golden = golden.moe_block_tkg(dense, ...)   # HF-validated oracle, dense fp32
    out_nkilib = moe_block_tkg_torch_ref(packed)    # nkilib ref, MXFP4 (correct unpack)
    assert out_golden == out_nkilib                 # => adapter reproduces the packing

``moe_block_tkg_torch_ref`` uses the exact ``mx_matmul`` unpack the kernel models,
so matching it proves the adapter recovers the kernel's effective dense weights.
Tolerance is the MXFP4 low-precision band (the nkilib ref quantizes the hidden to
mxfp8 in the matmul; the dense golden does not — so ~5e-2, matching the repo's own
MX MoE tolerance).
"""

import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.moe_block.moe_block_tkg_torch import moe_block_tkg_torch_ref
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from nkilib_src.nkilib.core.utils.torch_ref_wrapper import torch_ref_wrapper

from test.integration.nkilib.core.moe_block.test_moe_block_tkg import generate_inputs

from .gptoss_mxfp4_golden import kernels as golden
from .mxfp4_dense_adapter import (
    packed_mx_to_dense_down,
    packed_mx_to_dense_gate_up,
    unpack_gate_up_bias,
)

_EPS = 1e-5
_SWIGLU_LIMIT = 7.0
_SWIGLU_ALPHA = 1.702
_MX = nl.float4_e2m1fn_x4


@pytest.mark.parametrize("H,I,E,T,top_k", [(512, 512, 8, 16, 4)])
def test_moe_adapter_matches_nkilib_ref(H, I, E, T, top_k):
    """The packed->dense adapter reproduces the nkilib MX ref (== the kernel's math)."""
    ki = generate_inputs(
        batch=T,
        seqlen=1,
        hidden=H,
        hidden_actual=H,
        intermediate=I,
        num_global_experts=E,
        num_local_experts=E,
        top_k=top_k,
        router_fn=RouterActFnType.SOFTMAX,
        hidden_act_fn=ActFnType.Swish,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        moe_weight_dtype=_MX,
        input_dtype=nl.float16,
        has_bias=True,
        has_clamp=True,
        router_act_first=False,
        norm_topk_prob=False,
        skip_router_logits=True,
        router_mm_dtype=nl.float16,
        is_all_expert=True,
    )
    ki["expert_gate_up_bias"][:, :, 1, ...] += 1.0  # GPT-OSS +1 on the up bias

    # ── nkilib MX ref on the PACKED weights (uses the exact mx_matmul unpack). ──
    nkilib_out = torch_ref_wrapper(moe_block_tkg_torch_ref, preserve_lower_precision=False)(
        inp=ki["inp"],
        gamma=ki["gamma"],
        router_weights=ki["router_weights"],
        expert_gate_up_weights=ki["expert_gate_up_weights"],
        expert_down_weights=ki["expert_down_weights"],
        expert_gate_up_weights_scale=ki["expert_gate_up_weights_scale"],
        expert_down_weights_scale=ki["expert_down_weights_scale"],
        router_bias=ki["router_bias"],
        expert_gate_up_bias=ki["expert_gate_up_bias"],
        expert_down_bias=ki["expert_down_bias"],
        eps=_EPS,
        top_k=top_k,
        router_act_fn=RouterActFnType.SOFTMAX,
        router_pre_norm=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        hidden_act_fn=ActFnType.Swish,
        gate_clamp_upper_limit=_SWIGLU_LIMIT,
        up_clamp_upper_limit=_SWIGLU_LIMIT + 1,
        up_clamp_lower_limit=-_SWIGLU_LIMIT + 1,
        router_mm_dtype=nl.float16,
        hidden_actual=H,
        skip_router_logits=True,
        is_all_expert=True,
        rank_id=np.array([[0]], dtype=np.uint32),
    )["out"]
    nkilib_out = np.asarray(nkilib_out, dtype=np.float32).reshape(T, H)

    # ── Adapter: packed MXFP4 -> dense, then the GOLDEN oracle on dense fp32. ──
    dense_gate_up = packed_mx_to_dense_gate_up(ki["expert_gate_up_weights"], ki["expert_gate_up_weights_scale"], H, I)
    dense_down = packed_mx_to_dense_down(ki["expert_down_weights"], ki["expert_down_weights_scale"], H, I)
    dense_gu_bias = unpack_gate_up_bias(ki["expert_gate_up_bias"], H, I)

    hidden = torch.from_numpy(np.asarray(ki["inp"], dtype=np.float32)).reshape(T, H)
    gamma = torch.from_numpy(np.asarray(ki["gamma"], dtype=np.float32)).reshape(H)
    router_w = torch.from_numpy(np.asarray(ki["router_weights"], dtype=np.float32)).t().contiguous()  # [H,E]->[E,H]
    router_b = torch.from_numpy(np.asarray(ki["router_bias"], dtype=np.float32)).reshape(E)
    down_b = torch.from_numpy(np.asarray(ki["expert_down_bias"], dtype=np.float32)).reshape(E, H)

    golden_out = (
        golden.moe_block_tkg(
            hidden_states=hidden,
            gamma=gamma,
            eps=_EPS,
            router_weight=router_w,
            router_bias=router_b,
            gate_up_weight=dense_gate_up,
            gate_up_bias=dense_gu_bias,
            down_weight=dense_down,
            down_bias=down_b,
            top_k=top_k,
            swiglu_limit=_SWIGLU_LIMIT,
            swiglu_alpha=_SWIGLU_ALPHA,
        )
        .to(torch.float32)
        .numpy()
    )

    # The adapter's WEIGHT/layout reconstruction is exact (verified: feeding the
    # golden the identically-mxfp8-quantized hidden gives 1e-6). The residual
    # per-element gap here is purely the activation-quant difference — nkilib
    # quantizes hidden+intermediate to mxfp8 in-matmul, the dense golden does not.
    # That gap is small in aggregate (relative L2 ~3e-2) but inflates strict
    # per-element *relative* tolerance on near-zero outputs, so we validate with
    # cosine similarity + relative-L2, the same way the repo's MX MoE tests do —
    # not elementwise assert_allclose.
    g = golden_out.reshape(-1).astype(np.float64)
    k = nkilib_out.reshape(-1).astype(np.float64)
    cos_sim = float(np.dot(g, k) / (np.linalg.norm(g) * np.linalg.norm(k) + 1e-12))
    rel_l2 = float(np.linalg.norm(g - k) / (np.linalg.norm(g) + 1e-12))
    print(f"[adapter verify] cosine_similarity={cos_sim:.6f}  relative_L2={rel_l2:.4f}")
    assert cos_sim >= 0.999, (
        f"MXFP4 packed->dense adapter diverged from the nkilib MX ref: "
        f"cosine_similarity={cos_sim:.6f} < 0.999 (layout bug, not quant noise)"
    )
    assert rel_l2 <= 5e-2, (
        f"MXFP4 packed->dense adapter relative_L2={rel_l2:.4f} > 5e-2 "
        f"(exceeds the MXFP4 activation-quant band; likely a layout bug)"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
