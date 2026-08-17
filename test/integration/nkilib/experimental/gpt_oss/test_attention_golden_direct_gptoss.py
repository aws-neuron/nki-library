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

"""Direct kernel-vs-golden attention test on hardware.

Diffs the NKI ``attention_block_tkg`` kernel DIRECTLY against the vendored,
HF-bit-validated golden oracle ``gptoss_mxfp4_golden.kernels.attention_decode``
(a single unbroken kernel->golden link on real hardware), rather than against
nkilib's own torch ref. The reference here IS the golden: a torch_ref that takes
the kernel's input dict, adapts the tensors to the golden's layout (cos/sin
[B*S,D//2], paged KV without the size-1 kv-head axis, pos_id-derived mask), and
returns the golden's ``attention_decode`` output.

Attention weights are bf16 (no MXFP4 packing), so this needs no dequant adapter.
Config is the per-rank GPT-OSS TP8 shard (q_heads=8, kv_heads=1, d_head=64) at
LNC=1, full-attention and sliding-window variants. Tolerance is the bf16 decode
tolerance the attention suite already uses (kernel bf16 vs golden bf16 compute).
"""

import nki
import numpy as np
import pytest
import torch

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg import (
    attention_block_tkg_kernel_test_wrapper,
    generate_kernel_inputs,
    make_cosine_similarity_validator,
)
from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_utils import (
    AttnBlkTestConfig,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework

from .gptoss_mxfp4_golden import kernels as golden


class _AlwaysPass(CustomValidator):
    """Validator for the placeholder K_tkg/V_tkg outputs (not diffed vs golden)."""

    def validate(self, inference_output) -> bool:  # noqa: ARG002
        return True


def _golden_attention_ref(
    # Signature MUST match attention_block_tkg_kernel_test_wrapper exactly
    # (the framework's validate_torch_ref_signature requires an exact key-set
    # match). We accept every param but use only the ones the golden needs.
    X,
    X_in_sb,
    X_hidden_dim_actual,
    rmsnorm_X_enabled,
    rmsnorm_X_eps,
    rmsnorm_X_gamma,
    W_qkv,
    bias_qkv,
    quantization_type_qkv,
    weight_dequant_scale_qkv,
    input_dequant_scale_qkv,
    rmsnorm_QK_pre_rope_enabled,
    rmsnorm_QK_pre_rope_eps,
    rmsnorm_QK_pre_rope_W_Q,
    rmsnorm_QK_pre_rope_W_K,
    cos,
    sin,
    rope_contiguous_layout,
    rmsnorm_QK_post_rope_enabled,
    rmsnorm_QK_post_rope_eps,
    rmsnorm_QK_post_rope_W_Q,
    rmsnorm_QK_post_rope_W_K,
    skip_attention,
    K_cache_transposed,
    active_blocks_table,
    K_cache,
    V_cache,
    attention_mask,
    sink,
    softmax_scale,
    enable_fa_s_prior_tiling,
    fp8_packed,
    k_scale,
    v_scale,
    update_cache,
    kv_cache_update_idx,
    W_out,
    bias_out,
    quantization_type_out,
    weight_dequant_scale_out,
    input_dequant_scale_out,
    transposed_out,
    transposed_in,
    out_in_sb,
    sbm=None,
    KVDP=1,
    KVDP_replica_group=None,
    KVDP_collective_mode=None,
    KVDP_rank=None,
    pos_ids=None,
    swa_start_pos_ids=None,
    S_ctx=None,
    is_h_transposed_by_4=False,
    max_context_len=None,
    dtype_mode=None,
    CP=1,
    CP_replica_group=None,
    CP_collective_mode=None,
):
    """torch_ref backed by the golden ``attention_decode`` oracle.

    Signature mirrors ``attention_block_tkg_kernel_test_wrapper`` exactly (framework
    requirement). Adapts nkilib-layout inputs to the golden's layout, then returns
    ``{"X_out": [B*S, H]}``.
    """

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

    Xt = _t(X)  # [B, S, H]
    B, S, H = Xt.shape
    Wqkv = _t(W_qkv)
    # nkilib bias is [1, I]; golden wants [I].
    bqkv = _t(bias_qkv).reshape(-1) if bias_qkv is not None else None
    d_head = K_cache.shape[-1]
    # nkilib block-KV cache is [num_blocks, block_size, d_head] (kv_heads==1 folds
    # the head axis out); golden wants [num_blocks, kv_heads=1, block_size, d_head].
    Kc = _t(K_cache)
    Vc = _t(V_cache)
    if Kc.dim() == 3:
        Kc = Kc.unsqueeze(1)
        Vc = Vc.unsqueeze(1)
    kv_heads = Kc.shape[1]
    I = Wqkv.shape[1]
    num_q_heads = I // d_head - 2 * kv_heads
    # nkilib cos/sin are [D//2, B, S]; golden wants [B*S, D//2].
    cos_g = _t(cos).permute(1, 2, 0).reshape(B * S, d_head // 2)
    sin_g = _t(sin).permute(1, 2, 0).reshape(B * S, d_head // 2)
    # pos_ids [B, S] -> [B*S]; sliding window inferred from swa_start_pos_ids presence.
    pos_flat = _t(pos_ids).reshape(-1).long()
    sliding_window = None
    if swa_start_pos_ids is not None:
        start = _t(swa_start_pos_ids).reshape(-1).long()
        # golden recomputes start = clamp(pos - window + 1, 0); recover window.
        win = int((pos_flat - start).max().item()) + 1
        sliding_window = win
    btab = _t(active_blocks_table).int()
    slot = torch.zeros(B * S, dtype=torch.long)  # unused (update_cache=False)
    sink_g = _t(sink).reshape(-1) if sink is not None else None
    # softmax_scale may be None (kernel default = d_head**-0.5); the golden needs a float.
    scale = float(softmax_scale) if softmax_scale is not None else float(d_head**-0.5)

    out = golden.attention_decode(
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
    )
    # Return X_out in the kernel's IO dtype (bf16) so the validator's byte-view
    # reshape matches the kernel output element count. neuron_dtypes.static_cast
    # handles fp32 -> bf16 numpy.
    import neuron_dtypes as ndt

    x_dtype = X.dtype if isinstance(X, np.ndarray) else np.float32
    x_out = ndt.static_cast(out.to(torch.float32).numpy(), x_dtype)
    # The kernel wrapper returns (X_out, K_tkg, V_tkg) with update_cache=False. We
    # only validate X_out against the golden; K_tkg/V_tkg are the post-RoPE active
    # K/V — filled here to satisfy the 3-output contract but not tightly checked
    # (the custom comparator validates X_out only).
    S_tkg = X.shape[1]
    kv_placeholder_k = ndt.static_cast(np.zeros((B, kv_heads, S_tkg, d_head), dtype=np.float32), x_dtype)
    kv_placeholder_v = ndt.static_cast(np.zeros((B, kv_heads, S_tkg, d_head), dtype=np.float32), x_dtype)
    return {"X_out": x_out, "K_tkg": kv_placeholder_k, "V_tkg": kv_placeholder_v}


# Per-rank GPT-OSS TP8 attention shard at LNC=1 (non-transposed; kv_heads=1).
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


_PARAMS = [
    pytest.param(None, id="full"),
    pytest.param(128, id="swa"),
]


@pytest.mark.fast
@pytest.mark.parametrize("sliding_window", _PARAMS)
def test_attention_kernel_matches_golden(
    test_manager: Orchestrator,
    platform_target: Platforms,
    sliding_window,
):
    """The NKI attention kernel matches the HF-validated golden oracle directly."""
    cfg = _cfg(sliding_window)
    kernel_input = generate_kernel_inputs(cfg)

    def input_generator(test_config):
        return kernel_input

    # The wrapper returns (X_out, K_tkg, V_tkg) with update_cache=False. Declare all
    # three; only X_out is validated against the golden (K_tkg/V_tkg are placeholders).
    B, S = cfg.batch, cfg.S_tkg
    d_head, kv_heads = cfg.d_head, cfg.kv_heads
    x_dtype = kernel_input["X"].dtype

    def output_tensors(ki):
        return {
            "X_out": np.zeros((B * S, cfg.H), dtype=x_dtype),
            "K_tkg": np.zeros((B, kv_heads, S, d_head), dtype=x_dtype),
            "V_tkg": np.zeros((B, kv_heads, S, d_head), dtype=x_dtype),
        }

    def comparator(golden_dict, output_tensors_ignored=None):
        # Validate X_out tightly against the golden; K_tkg/V_tkg are placeholders
        # (always pass) since the golden oracle does not expose the active K/V.
        validators = {}
        x_golden = golden_dict["X_out"]
        validators["X_out"] = CustomValidatorWithOutputTensorData(
            validator=make_cosine_similarity_validator(
                x_golden,
                rtol=2e-2,
                atol=1e-2,
                min_cosine_similarity=0.99,
                min_pass_rate=0.99,
                name="X_out",
            ),
            output_ndarray=x_golden,
        )
        for name in ("K_tkg", "V_tkg"):
            g = golden_dict[name]
            validators[name] = CustomValidatorWithOutputTensorData(
                validator=_AlwaysPass,
                output_ndarray=g,
            )
        return validators

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=nki.jit(attention_block_tkg_kernel_test_wrapper),
        torch_ref=_golden_attention_ref,
        kernel_input_generator=input_generator,
        output_tensor_descriptor=output_tensors,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(
            logical_nc_config=cfg.lnc,
            enable_birsim=False,
            platform_target=platform_target,
            additional_cmd_args=["--enable-ocp-compliant-scale-computation"],
        ),
        rtol=2e-2,
        atol=1e-2,
        custom_comparator=comparator,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
