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

# ty: ignore — vendored GPT-OSS golden/test; heterogeneous helper dicts trip ty's type mapping
"""Integration test for the GPT-OSS MXFP4 decode-layer kernel (placeholder stub).

The reference oracle is the pure-torch golden ``gptoss_mxfp4_decode_layer``,
**vendored** alongside this test in the ``gptoss_mxfp4_golden`` subpackage (a
verbatim copy of ``private-vllm-neuron/gptoss_mxfp4_golden`` — see
``gptoss_mxfp4_golden/VENDORED.md``). That golden constructs a
``GptOssDecoderLayer`` and runs its ``forward_decode``, so the executed math is
the structured GPT-OSS decode block — bit-validated against HuggingFace on real
120B weights. We diff the NKI kernel against it via ``UnitTestFramework``. The
vendoring makes this test self-contained: no cross-package ``sys.path`` bridge
and no ``private-vllm-neuron`` checkout required.

Two tests:

  * ``test_gpt_oss_mxfp4_decode_layer_traces`` — drives the kernel through
    ``UnitTestFramework``. It validates that the kernel and the torch-ref
    oracle have matching signatures and that the kernel traces/compiles. The
    kernel body is still a placeholder that returns zeros, so it is marked
    ``compile_only``: the framework traces + compiles the kernel but skips the
    numeric diff against the golden (which zeros would fail), including on
    hardware. (Under ``--test-mode=simulation`` the marker instead skips the
    test outright, since a mode marker is incompatible with simulation.) Drop
    the ``compile_only`` marker to re-enable the numeric comparison once the
    real kernel body lands. The test also pins the tracer frontend via
    ``CompilerArgs(nki_compilation_mode=...)``, so it traces without a Neuron
    device regardless of the suite-wide ``--nki-compilation-mode``.

  * ``test_torch_ref_bridge_matches_golden`` — pure CPU, no NKI. Proves the
    torch-ref adapter (flat scalars -> ``GptOssConfig`` -> golden) reproduces the
    golden's own flat entry bit-for-bit, so the oracle wiring is trustworthy
    independent of the kernel.
"""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.models.gpt_oss.c128_giga_kernel.experimental.gpt_oss.gpt_oss_mxfp4_decode_layer import (
    gpt_oss_mxfp4_decode_layer,
)

from test.utils.common_dataclasses import CompilerArgs, NKICompilationMode, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Reference oracle: the vendored pure-torch golden (see gptoss_mxfp4_golden/VENDORED.md).
from .gptoss_mxfp4_golden.config import GptOssConfig
from .gptoss_mxfp4_golden.layer_entry import gptoss_mxfp4_decode_layer as golden_layer
from .gptoss_mxfp4_golden.model import GptOssRotaryEmbedding
from .gptoss_mxfp4_golden.paged_kv import PagedKVManager

_GOLDEN = {
    "GptOssConfig": GptOssConfig,
    "golden_layer": golden_layer,
    "GptOssRotaryEmbedding": GptOssRotaryEmbedding,
    "PagedKVManager": PagedKVManager,
}


# ── Tiny GPT-OSS-shaped config (fast to trace / run on CPU) ──────────────────
# Ratios follow the 120B (GQA 8:1, SWA on even layers, top-k MoE) at toy sizes.
_H = 128  # hidden
_HEAD_DIM = 16
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 2
_INTERMEDIATE = 64  # I
_NUM_EXPERTS = 8
_TOP_K = 2
_EPS = 1e-5
_SWIGLU_LIMIT = 7.0
_SWIGLU_ALPHA = 1.702
_BLOCK_SIZE = 8
_NUM_BLOCKS = 64


def _build_case(sliding_window, seed, dtype):
    """Build one seeded single-layer decode case with the golden's helpers.

    Returns a dict of torch tensors (one per kernel weight/cache/input) plus the
    ``PagedKVManager`` whose caches back ``k_cache``/``v_cache``. Two calls with
    the same ``seed`` produce identical starting caches, so a reference and a
    kernel can be diffed on independent-but-equal state.
    """
    import torch

    GptOssConfig = _GOLDEN["GptOssConfig"]
    GptOssRotaryEmbedding = _GOLDEN["GptOssRotaryEmbedding"]
    PagedKVManager = _GOLDEN["PagedKVManager"]
    from .gptoss_mxfp4_golden.model import GptOssDecoderLayer

    torch.manual_seed(seed)
    cfg = GptOssConfig(
        hidden_size=_H,
        num_attention_heads=_NUM_Q_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        intermediate_size=_INTERMEDIATE,
        rms_norm_eps=_EPS,
        num_local_experts=_NUM_EXPERTS,
        num_experts_per_tok=_TOP_K,
        sliding_window=sliding_window if sliding_window is not None else 32,
        swiglu_limit=_SWIGLU_LIMIT,
        swiglu_alpha=_SWIGLU_ALPHA,
        torch_dtype=dtype,
    )
    layer_idx = 0 if sliding_window is not None else 1
    lcfg = cfg.tiny(num_hidden_layers=1, sliding_window=cfg.sliding_window)
    layer = GptOssDecoderLayer(lcfg, layer_idx=layer_idx)
    for _, p in layer.named_parameters():
        if p.dtype.is_floating_point:
            torch.nn.init.normal_(p, std=0.02)
    layer.eval()

    mgr = PagedKVManager(
        num_blocks=_NUM_BLOCKS,
        block_size=_BLOCK_SIZE,
        kv_heads=_NUM_KV_HEADS,
        head_dim=_HEAD_DIM,
        num_layers=1,
        dtype=dtype,
    )
    seq_ids = [0, 1]
    prefill_lens = [20, 13]
    g = torch.Generator().manual_seed(seed + 5)
    for i, s in enumerate(seq_ids):
        mgr.allocate_sequence(s, prefill_lens[i])
        n = prefill_lens[i]
        mgr.prefill_write(
            0,
            s,
            torch.randn(n, _NUM_KV_HEADS, _HEAD_DIM, generator=g),
            torch.randn(n, _NUM_KV_HEADS, _HEAD_DIM, generator=g),
        )
    positions = mgr.current_positions(seq_ids).to(torch.int32)
    max_blocks = mgr.max_blocks_per_seq(seq_ids)
    block_table = mgr.block_table(seq_ids, max_blocks).to(torch.int32)
    slot_mapping = mgr.decode_slot_mapping(seq_ids).to(torch.int32)

    T = len(seq_ids)
    hidden = torch.randn(T, _H, generator=g).to(dtype)
    rot = GptOssRotaryEmbedding(lcfg)
    cos, sin = rot(positions, dtype=hidden.dtype)

    sd = layer.state_dict()
    tensors = {
        "hidden_states": hidden,
        "positions": positions,
        "cos": cos,
        "sin": sin,
        "input_layernorm_weight": sd["input_layernorm.weight"],
        "qkv_proj_weight": sd["self_attn.qkv_proj_weight"],
        "qkv_proj_bias": sd["self_attn.qkv_proj_bias"],
        "o_proj_weight": sd["self_attn.o_proj_weight"],
        "o_proj_bias": sd["self_attn.o_proj_bias"],
        "sinks": sd["self_attn.sinks"],
        "post_attention_layernorm_weight": sd["mlp.experts.post_attention_layernorm.weight"],
        "router_weight": sd["mlp.experts.router_weight"],
        "router_bias": sd["mlp.experts.router_bias"],
        "gate_up_weight": sd["mlp.experts.gate_up_weight"],
        "gate_up_bias": sd["mlp.experts.gate_up_bias"],
        "down_weight": sd["mlp.experts.down_weight"],
        "down_bias": sd["mlp.experts.down_bias"],
        "k_cache": mgr.k_caches[0],
        "v_cache": mgr.v_caches[0],
        "block_table": block_table,
        "slot_mapping": slot_mapping,
    }
    return tensors, mgr


# ── Torch-ref oracle: flat kernel scalars -> GptOssConfig -> golden layer ────
def _gpt_oss_mxfp4_decode_layer_torch_ref(
    hidden_states,
    positions,
    cos,
    sin,
    input_layernorm_weight,
    qkv_proj_weight,
    qkv_proj_bias,
    o_proj_weight,
    o_proj_bias,
    sinks,
    post_attention_layernorm_weight,
    router_weight,
    router_bias,
    gate_up_weight,
    gate_up_bias,
    down_weight,
    down_bias,
    k_cache,
    v_cache,
    block_table,
    slot_mapping,
    head_dim,
    num_q_heads,
    num_kv_heads,
    num_experts,
    top_k,
    eps,
    softmax_scale,  # noqa: ARG001 — golden derives head_dim**-0.5; kept for signature parity
    swiglu_limit,
    swiglu_alpha,
    sliding_window=None,
):
    """Reference oracle matching ``gpt_oss_mxfp4_decode_layer``'s signature.

    Reconstructs a ``GptOssConfig`` from the flat scalars (H and I come from the
    tensor shapes) and calls the standalone golden's single-layer flat entry,
    returning ``{"layer_output": [T, H]}``. The golden writes the new K/V into
    ``k_cache``/``v_cache`` in place, exactly as the kernel will.
    """
    GptOssConfig = _GOLDEN["GptOssConfig"]
    golden_layer = _GOLDEN["golden_layer"]

    H = hidden_states.shape[-1]
    I = gate_up_weight.shape[-1]
    cfg = GptOssConfig(
        hidden_size=H,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        intermediate_size=I,
        rms_norm_eps=eps,
        num_local_experts=num_experts,
        num_experts_per_tok=top_k,
        sliding_window=sliding_window if sliding_window is not None else 32,
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        torch_dtype=hidden_states.dtype,
    )
    out = golden_layer(
        cfg,
        hidden_states,
        positions,
        cos,
        sin,
        input_layernorm_weight,
        qkv_proj_weight,
        qkv_proj_bias,
        o_proj_weight,
        o_proj_bias,
        sinks,
        post_attention_layernorm_weight,
        router_weight,
        router_bias,
        gate_up_weight,
        gate_up_bias,
        down_weight,
        down_bias,
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        slot_mapping=slot_mapping,
        sliding_window=sliding_window,
    )
    return {"layer_output": out}


# ── numpy conversion for the framework ───────────────────────────────────────
def _to_numpy_inputs(tensors, np_float_dtype):
    """Convert a case's torch tensors to the numpy dict the framework passes to
    both the kernel and (via torch_ref_wrapper) the torch ref."""
    import torch

    out = {}
    for k, v in tensors.items():
        arr = v.detach().cpu()
        if arr.dtype.is_floating_point:
            out[k] = arr.to(dtype=torch.float32).numpy().astype(np_float_dtype)
        else:
            out[k] = arr.numpy().astype(np.int32)
    return out


def _scalar_inputs(sliding_window):
    return {
        "head_dim": _HEAD_DIM,
        "num_q_heads": _NUM_Q_HEADS,
        "num_kv_heads": _NUM_KV_HEADS,
        "num_experts": _NUM_EXPERTS,
        "top_k": _TOP_K,
        "eps": _EPS,
        "softmax_scale": _HEAD_DIM**-0.5,
        "swiglu_limit": _SWIGLU_LIMIT,
        "swiglu_alpha": _SWIGLU_ALPHA,
        "sliding_window": sliding_window,
    }


PARAM_NAMES = "sliding_window, lnc, rel_tol"
TEST_PARAMS = [
    pytest.param(8, 1, 5.0, marks=pytest.mark.fast),  # even layer: sliding-window attention
    pytest.param(None, 1, 5.0, marks=pytest.mark.fast),  # odd layer: full attention
]
_ABBREVS = {"sliding_window": "sw", "rel_tol": "rt"}


@pytest_test_metadata(name="GPT-OSS MXFP4 Decode Layer")
class TestGptOssMxfp4DecodeLayer:
    """Integration tests for the gpt_oss_mxfp4_decode_layer kernel (stub)."""

    @staticmethod
    def generate_inputs(sliding_window, dtype_np):
        import torch

        tensors, _ = _build_case(sliding_window, seed=0, dtype=torch.float32)
        kernel_input = _to_numpy_inputs(tensors, dtype_np)
        kernel_input.update(_scalar_inputs(sliding_window))
        return kernel_input

    # Pins CompileOnly trace mode over the suite default (the trace_mode fixture
    # honours the marker), skipping inference + the numeric diff a zero-returning
    # stub cannot pass. See the module docstring; drop it when the real body lands.
    @pytest.mark.compile_only
    @pytest_parametrize(PARAM_NAMES, TEST_PARAMS, abbrevs=_ABBREVS)
    def test_gpt_oss_mxfp4_decode_layer_traces(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        sliding_window,
        lnc: int,
        rel_tol: float,
    ):
        """Trace/compile the kernel; the numeric diff is gated off while it is a stub.

        The stub returns zeros, so it only passes numeric validation once the
        real body is implemented. The ``compile_only`` marker pins this test to
        CompileOnly trace mode, so the run traces + compiles the kernel (and
        validates the kernel/torch-ref signatures) but skips inference and the
        output comparison, including on hardware. (Under ``--test-mode=simulation``
        the marker instead skips the test outright.) The kernel also pins the
        tracer frontend via ``CompilerArgs(nki_compilation_mode=NKICompilationMode.tracer)``
        so it traces the same way regardless of the suite-wide ``--nki-compilation-mode``.
        """
        T = 2
        dtype_np = ml_dtypes.bfloat16

        def input_generator(test_config):
            return self.generate_inputs(sliding_window, dtype_np)

        def output_tensors(kernel_input):
            return {"layer_output": np.zeros((T, _H), dtype=dtype_np)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gpt_oss_mxfp4_decode_layer,
            torch_ref=torch_ref_wrapper(_gpt_oss_mxfp4_decode_layer_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc,
                platform_target=platform_target,
                nki_compilation_mode=NKICompilationMode.tracer,
            ),
            rtol=rel_tol / 100.0,
            atol=1e-2,
        )

    @pytest.mark.parametrize("sliding_window", [8, None])
    def test_torch_ref_bridge_matches_golden(self, sliding_window):
        """The flat torch-ref adapter reproduces the golden's own flat entry.

        Pure CPU, no NKI. Guards that the scalar->config reconstruction and the
        numpy<->torch bridge are faithful, so the oracle the kernel is diffed
        against is exactly the validated golden.
        """
        import torch

        golden_layer = _GOLDEN["golden_layer"]

        # Reference: call the golden directly on a seeded case (fresh cache A).
        tensors_a, _ = _build_case(sliding_window, seed=0, dtype=torch.float32)
        cfg = _GOLDEN["GptOssConfig"](
            hidden_size=_H,
            num_attention_heads=_NUM_Q_HEADS,
            num_key_value_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            intermediate_size=_INTERMEDIATE,
            rms_norm_eps=_EPS,
            num_local_experts=_NUM_EXPERTS,
            num_experts_per_tok=_TOP_K,
            sliding_window=sliding_window if sliding_window is not None else 32,
            swiglu_limit=_SWIGLU_LIMIT,
            swiglu_alpha=_SWIGLU_ALPHA,
            torch_dtype=torch.float32,
        )
        weight_keys = [
            "hidden_states",
            "positions",
            "cos",
            "sin",
            "input_layernorm_weight",
            "qkv_proj_weight",
            "qkv_proj_bias",
            "o_proj_weight",
            "o_proj_bias",
            "sinks",
            "post_attention_layernorm_weight",
            "router_weight",
            "router_bias",
            "gate_up_weight",
            "gate_up_bias",
            "down_weight",
            "down_bias",
        ]
        expected = golden_layer(
            cfg,
            *[tensors_a[k] for k in weight_keys],
            k_cache=tensors_a["k_cache"],
            v_cache=tensors_a["v_cache"],
            block_table=tensors_a["block_table"],
            slot_mapping=tensors_a["slot_mapping"],
            sliding_window=sliding_window,
        )

        # Adapter path: rebuild an identical case (fresh cache B), pass through the
        # numpy<->torch wrapper exactly as the framework does.
        tensors_b, _ = _build_case(sliding_window, seed=0, dtype=torch.float32)
        kernel_input = _to_numpy_inputs(tensors_b, np.float32)
        kernel_input.update(_scalar_inputs(sliding_window))
        wrapped = torch_ref_wrapper(_gpt_oss_mxfp4_decode_layer_torch_ref)
        actual = wrapped(**kernel_input)["layer_output"]

        assert actual.shape == (2, _H)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-x"])
