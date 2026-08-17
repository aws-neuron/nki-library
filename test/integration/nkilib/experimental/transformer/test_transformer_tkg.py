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
"""Integration tests for transformer_tkg kernel."""

import math
import os

os.environ["NKI_FRONTEND"] = "beta2"

from typing import List, Optional, final

import ml_dtypes
import neuron_dtypes as dt
import nki
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib.core.attention.gen_mask_tkg_torch import build_full_attention_mask
from nkilib_src.nkilib.experimental.transformer.transformer_tkg import transformer_tkg
from nkilib_src.nkilib.experimental.transformer.transformer_tkg_torch import llama3_transformer_fwd_tkg_torch

from test.integration.nkilib.core.attention.test_attention_tkg_utils import generate_cache_lens
from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def generate_llama3_transformer_tkg_combinations(
    tp_values: list[int] | None = None,
    global_batch_values: list[int] | None = None,
    S_tkg: int = 1,
    S_ctx_values: list[int] | None = None,
    q_heads: int = 64,
    d_head: int = 128,
    H: int = 8192,
    I: int = 28672,
    lnc: int = 2,
    rel_diff_tolerance: int = 2.5,
    enable_separation_pass_values: list[bool] | None = None,
) -> list[list]:
    """Generate test combinations for transformer TKG kernel.

    Uses Llama 70B model dimensions.

    Returns list of [tp, batch, S_tkg, S_ctx, q_heads, d_head, H, I, lnc, rel_tol]
    """
    tp_values = tp_values if tp_values is not None else [8, 16, 32]
    global_batch_values = global_batch_values if global_batch_values is not None else [8, 16, 32, 64, 128]
    S_ctx_values = S_ctx_values if S_ctx_values is not None else [1024, 10240, 36864]
    enable_separation_pass_values = (
        enable_separation_pass_values if enable_separation_pass_values is not None else [False]
    )

    combinations = []

    for tp in tp_values:
        for global_batch in global_batch_values:
            dp_degree = 64 // tp
            batch = global_batch // dp_degree

            for S_ctx in S_ctx_values:
                for _enable_separation_pass in enable_separation_pass_values:
                    combination = [
                        tp,
                        batch,
                        S_tkg,
                        S_ctx,
                        q_heads,
                        d_head,
                        H,
                        I,
                        lnc,
                        rel_diff_tolerance,
                    ]

                    combinations.append(combination)

    return combinations


# ── Variance-preserving tensor generators ──────────────────────────────────
# bf16 has ~0.4% worst-case quantization error. With N(0,1) weights, QKV
# projections have std ≈ √H, making attention scores O(H). When two positions
# score nearly identically, 0.4% noise can flip the softmax argmax, picking
# wrong V rows. Using 1/√fan_in scaling keeps scores O(1) where softmax is
# stable. See commit 50ec39f for full analysis.
_rng = np.random.default_rng(42)


def _uniform_activation(shape, dtype):
    """Uniform[-1, 1]"""
    return np.ascontiguousarray(dt.static_cast(_rng.uniform(-1.0, 1.0, shape).astype(np.float32), dtype))


def _gaussian(shape, dtype, std):
    """N(0, std)"""
    return np.ascontiguousarray(dt.static_cast(_rng.normal(0.0, std, shape).astype(np.float32), dtype))


def _fan_in_projection(shape, dtype, fan_in):
    """N(0, 1/√fan_in). Keeps matmul output variance ≈ input variance."""
    return _gaussian(shape, dtype, std=1.0 / np.sqrt(fan_in))


def _near_unity(shape, dtype):
    """Uniform[0.5, 1.5]. RMSNorm gammas are ~1.0 in trained models."""
    return np.ascontiguousarray(dt.static_cast(_rng.uniform(0.5, 1.5, shape).astype(np.float32), dtype))


def transformer_tkg_tp4_4layer_wrapper(
    X: nl.ndarray,
    W_qkv_0: nl.ndarray,
    W_qkv_1: nl.ndarray,
    W_qkv_2: nl.ndarray,
    W_qkv_3: nl.ndarray,
    W_out_0: nl.ndarray,
    W_out_1: nl.ndarray,
    W_out_2: nl.ndarray,
    W_out_3: nl.ndarray,
    W_gate_0: nl.ndarray,
    W_gate_1: nl.ndarray,
    W_gate_2: nl.ndarray,
    W_gate_3: nl.ndarray,
    W_up_0: nl.ndarray,
    W_up_1: nl.ndarray,
    W_up_2: nl.ndarray,
    W_up_3: nl.ndarray,
    W_down_0: nl.ndarray,
    W_down_1: nl.ndarray,
    W_down_2: nl.ndarray,
    W_down_3: nl.ndarray,
    W_gamma_qkv_0: nl.ndarray,
    W_gamma_qkv_1: nl.ndarray,
    W_gamma_qkv_2: nl.ndarray,
    W_gamma_qkv_3: nl.ndarray,
    W_gamma_mlp_0: nl.ndarray,
    W_gamma_mlp_1: nl.ndarray,
    W_gamma_mlp_2: nl.ndarray,
    W_gamma_mlp_3: nl.ndarray,
    K_cache_0: nl.ndarray,
    K_cache_1: nl.ndarray,
    K_cache_2: nl.ndarray,
    K_cache_3: nl.ndarray,
    V_cache_0: nl.ndarray,
    V_cache_1: nl.ndarray,
    V_cache_2: nl.ndarray,
    V_cache_3: nl.ndarray,
    RoPE_cos: nl.ndarray,
    RoPE_sin: nl.ndarray,
    attention_mask: nl.ndarray,
    position_ids: Optional[nl.ndarray],
    eps: float,
    sbuf_residual_and_cc: bool,
    W_gate_scale_0: nl.ndarray,
    W_gate_scale_1: nl.ndarray,
    W_gate_scale_2: nl.ndarray,
    W_gate_scale_3: nl.ndarray,
    W_up_scale_0: nl.ndarray,
    W_up_scale_1: nl.ndarray,
    W_up_scale_2: nl.ndarray,
    W_up_scale_3: nl.ndarray,
    W_down_scale_0: nl.ndarray,
    W_down_scale_1: nl.ndarray,
    W_down_scale_2: nl.ndarray,
    W_down_scale_3: nl.ndarray,
    replica_groups: List[List[int]],
):
    """Wrapper for TP=4, 4-layer transformer (quantized)."""
    return transformer_tkg(
        X=X,
        W_qkvs=[W_qkv_0, W_qkv_1, W_qkv_2, W_qkv_3],
        W_outs=[W_out_0, W_out_1, W_out_2, W_out_3],
        W_gates=[W_gate_0, W_gate_1, W_gate_2, W_gate_3],
        W_ups=[W_up_0, W_up_1, W_up_2, W_up_3],
        W_downs=[W_down_0, W_down_1, W_down_2, W_down_3],
        W_gamma_qkvs=[W_gamma_qkv_0, W_gamma_qkv_1, W_gamma_qkv_2, W_gamma_qkv_3],
        W_gamma_mlps=[W_gamma_mlp_0, W_gamma_mlp_1, W_gamma_mlp_2, W_gamma_mlp_3],
        K_caches=[K_cache_0, K_cache_1, K_cache_2, K_cache_3],
        V_caches=[V_cache_0, V_cache_1, V_cache_2, V_cache_3],
        RoPE_cos=RoPE_cos,
        RoPE_sin=RoPE_sin,
        attention_mask=attention_mask,
        position_ids=position_ids,
        num_layers=4,
        eps=eps,
        replica_groups=replica_groups,
        sbuf_residual_and_cc=sbuf_residual_and_cc,
        W_gate_scales=[W_gate_scale_0, W_gate_scale_1, W_gate_scale_2, W_gate_scale_3],
        W_up_scales=[W_up_scale_0, W_up_scale_1, W_up_scale_2, W_up_scale_3],
        W_down_scales=[W_down_scale_0, W_down_scale_1, W_down_scale_2, W_down_scale_3],
    )


def transformer_tkg_tp4_4layer_torch_ref(
    X,
    W_qkv_0,
    W_qkv_1,
    W_qkv_2,
    W_qkv_3,
    W_out_0,
    W_out_1,
    W_out_2,
    W_out_3,
    W_gate_0,
    W_gate_1,
    W_gate_2,
    W_gate_3,
    W_up_0,
    W_up_1,
    W_up_2,
    W_up_3,
    W_down_0,
    W_down_1,
    W_down_2,
    W_down_3,
    W_gamma_qkv_0,
    W_gamma_qkv_1,
    W_gamma_qkv_2,
    W_gamma_qkv_3,
    W_gamma_mlp_0,
    W_gamma_mlp_1,
    W_gamma_mlp_2,
    W_gamma_mlp_3,
    K_cache_0,
    K_cache_1,
    K_cache_2,
    K_cache_3,
    V_cache_0,
    V_cache_1,
    V_cache_2,
    V_cache_3,
    RoPE_cos,
    RoPE_sin,
    attention_mask,
    position_ids,
    eps,
    sbuf_residual_and_cc,
    W_gate_scale_0,
    W_gate_scale_1,
    W_gate_scale_2,
    W_gate_scale_3,
    W_up_scale_0,
    W_up_scale_1,
    W_up_scale_2,
    W_up_scale_3,
    W_down_scale_0,
    W_down_scale_1,
    W_down_scale_2,
    W_down_scale_3,
    replica_groups,
):
    """Torch reference matching transformer_tkg_tp4_4layer_wrapper signature."""
    result = llama3_transformer_fwd_tkg_torch(
        X=X,
        W_qkvs=[W_qkv_0, W_qkv_1, W_qkv_2, W_qkv_3],
        W_outs=[W_out_0, W_out_1, W_out_2, W_out_3],
        W_gates=[W_gate_0, W_gate_1, W_gate_2, W_gate_3],
        W_gate_scales=[W_gate_scale_0, W_gate_scale_1, W_gate_scale_2, W_gate_scale_3],
        W_ups=[W_up_0, W_up_1, W_up_2, W_up_3],
        W_up_scales=[W_up_scale_0, W_up_scale_1, W_up_scale_2, W_up_scale_3],
        W_downs=[W_down_0, W_down_1, W_down_2, W_down_3],
        W_down_scales=[W_down_scale_0, W_down_scale_1, W_down_scale_2, W_down_scale_3],
        W_gamma_qkvs=[W_gamma_qkv_0, W_gamma_qkv_1, W_gamma_qkv_2, W_gamma_qkv_3],
        W_gamma_mlps=[W_gamma_mlp_0, W_gamma_mlp_1, W_gamma_mlp_2, W_gamma_mlp_3],
        RoPE_cos=RoPE_cos,
        RoPE_sin=RoPE_sin,
        attention_mask=attention_mask,
        position_ids=position_ids,
        K_caches=[K_cache_0, K_cache_1, K_cache_2, K_cache_3],
        V_caches=[V_cache_0, V_cache_1, V_cache_2, V_cache_3],
        num_layers=4,
        replica_groups=replica_groups,
        eps=eps,
        mlp_down_proj_layout_enabled=False,
    )
    return {"layer_output": result}


# Test vectors generated using generate_llama3_transformer_tkg_combinations()
# Uses Llama 70B model dimensions (H=8192, I=28672, q_heads=64)
# Format: [tp, batch, S_tkg, S_ctx, q_heads, d_head, H, I, lnc, rel_tol]
PARAM_NAMES = "tp, batch, S_tkg, S_ctx, q_heads, d_head, H, I, lnc, rel_tol"
TEST_PARAMS = generate_llama3_transformer_tkg_combinations()
_ABBREVS = {"batch": "B", "S_tkg": "St", "S_ctx": "Sc", "q_heads": "qh", "d_head": "dh", "rel_tol": "rt"}

# (tp, batch, S_tkg, S_ctx) keys for full-only tests (excluded from fast suite)
_FULL_ONLY_KEYS = {
    (8, 8, 1, 1024),
    (8, 8, 1, 10240),
    (16, 2, 1, 1024),
    (16, 2, 1, 10240),
    (16, 2, 1, 36864),
    (16, 4, 1, 1024),
    (16, 4, 1, 10240),
    (16, 4, 1, 36864),
    (16, 8, 1, 1024),
    (16, 8, 1, 10240),
    (16, 16, 1, 1024),
    (16, 32, 1, 1024),
    (32, 4, 1, 1024),
    (32, 4, 1, 10240),
    (32, 4, 1, 36864),
    (32, 8, 1, 1024),
    (32, 8, 1, 10240),
    (32, 16, 1, 1024),
    (32, 16, 1, 10240),
    (32, 32, 1, 1024),
    (32, 64, 1, 1024),
    (32, 64, 1, 36864),
    (16, 32, 1, 36864),
    (32, 32, 1, 36864),
    (8, 16, 1, 36864),
    (16, 16, 1, 36864),
    (32, 16, 1, 36864),
    (32, 64, 1, 10240),
    (8, 8, 1, 36864),
    (8, 16, 1, 10240),
    (8, 4, 1, 36864),
    (16, 32, 1, 10240),
    (16, 8, 1, 36864),
    (8, 2, 1, 36864),
    (32, 8, 1, 36864),
    (32, 32, 1, 10240),
    (8, 2, 1, 10240),
    (8, 1, 1, 36864),
    (8, 4, 1, 10240),
    (16, 16, 1, 10240),
    (8, 16, 1, 1024),
}

ALL_PARAMS = [
    pytest.param(*c, marks=pytest.mark.fast) if tuple(c[:4]) not in _FULL_ONLY_KEYS else c for c in TEST_PARAMS
]


@pytest_test_metadata(name="Transformer TKG")
@pytest_marks(["transformer_tkg", "transformer"])
@final
@pytest.mark.high_rank
class TestTransformerTKG:
    """Integration tests for transformer_tkg kernel."""

    @staticmethod
    def generate_inputs(tp, batch, S_tkg, S_ctx, q_heads, d_head, H, I, lnc):
        """Generate all input tensors for the transformer_tkg kernel."""
        np.random.seed(42)
        B = batch
        num_kv_heads = 1
        num_layers = 4
        eps = 1e-6
        dtype = nl.bfloat16

        # Compute per-core dimensions (sharded across TP)
        q_heads_per_core = q_heads // tp
        fd_per_core = I // tp
        # Round up to the nearest 128 - MLP only supports I that are multiples of 128
        fd_per_core = math.ceil(fd_per_core / 128) * 128

        qkv_dim = d_head * (q_heads_per_core + 2 * num_kv_heads)

        # First and last layers are non-quantized
        nonquantized_layers = {0, num_layers - 1}

        def mlp_w_dtype(layer):
            if layer in nonquantized_layers:
                return dtype
            return nl.float8_e4m3

        # Generate per-layer tensors with variance-preserving distributions
        W_qkvs = [_fan_in_projection((H, qkv_dim), dtype, fan_in=H) for _ in range(num_layers)]
        W_outs = [_gaussian((q_heads_per_core * d_head, H), dtype, std=0.5) for _ in range(num_layers)]
        W_gates = [_fan_in_projection((H, fd_per_core), mlp_w_dtype(layer), fan_in=H) for layer in range(num_layers)]
        W_ups = [_fan_in_projection((H, fd_per_core), mlp_w_dtype(layer), fan_in=H) for layer in range(num_layers)]
        W_downs = [
            _fan_in_projection((fd_per_core, H), mlp_w_dtype(layer), fan_in=fd_per_core) for layer in range(num_layers)
        ]
        W_gamma_qkvs = [_near_unity((1, H), dtype) for _ in range(num_layers)]
        W_gamma_mlps = [_near_unity((1, H), dtype) for _ in range(num_layers)]
        K_caches = [_uniform_activation((B, num_kv_heads, d_head, S_ctx), dtype) for _ in range(num_layers)]
        V_caches = [_uniform_activation((B, num_kv_heads, S_ctx, d_head), dtype) for _ in range(num_layers)]

        # Generate attention mask and position_ids
        cache_len = generate_cache_lens(B, S_ctx, S_tkg, mode="normal")
        assert cache_len.max() <= (S_ctx - S_tkg)

        cache_lens_torch = torch.from_numpy(cache_len.flatten()).to(torch.float32)
        attention_mask = build_full_attention_mask(
            cache_lens=cache_lens_torch,
            batch=B,
            num_heads=q_heads_per_core,
            s_active=S_tkg,
            s_ctx=S_ctx,
            lnc=lnc,
            block_len=0,
            include_active_mask=True,
            transposed=True,
        ).numpy()  # (S_ctx, B, q_heads_per_core, S_tkg)
        attention_mask = dt.static_cast(np.ascontiguousarray(attention_mask), dtype=np.uint8)

        position_ids = cache_len + np.arange(S_tkg)  # (B, S_tkg)

        # Generate MLP scales: first and last layers are non-quantized (no scales)
        scale_rng = np.random.default_rng(0)
        W_gate_scales = []
        W_up_scales = []
        W_down_scales = []
        for layer in range(num_layers):
            if layer in nonquantized_layers:
                W_gate_scales.append(None)
                W_up_scales.append(None)
                W_down_scales.append(None)
            else:
                W_gate_scales.append(np.full((128, fd_per_core), scale_rng.random(), dtype=np.float32))
                W_up_scales.append(np.full((128, fd_per_core), scale_rng.random(), dtype=np.float32))
                W_down_scales.append(np.full((128, H), scale_rng.random(), dtype=np.float32))

        # Build kernel_input with individual keys (required by test framework)
        kernel_input = {"X": _uniform_activation((B, S_tkg, H), dtype)}
        for i in range(num_layers):
            kernel_input[f"W_qkv_{i}"] = W_qkvs[i]
        for i in range(num_layers):
            kernel_input[f"W_out_{i}"] = W_outs[i]
        for i in range(num_layers):
            kernel_input[f"W_gate_{i}"] = W_gates[i]
        for i in range(num_layers):
            kernel_input[f"W_up_{i}"] = W_ups[i]
        for i in range(num_layers):
            kernel_input[f"W_down_{i}"] = W_downs[i]
        for i in range(num_layers):
            kernel_input[f"W_gamma_qkv_{i}"] = W_gamma_qkvs[i]
        for i in range(num_layers):
            kernel_input[f"W_gamma_mlp_{i}"] = W_gamma_mlps[i]
        for i in range(num_layers):
            kernel_input[f"K_cache_{i}"] = K_caches[i]
        for i in range(num_layers):
            kernel_input[f"V_cache_{i}"] = V_caches[i]
        kernel_input["RoPE_cos"] = _uniform_activation((d_head // 2, B, S_tkg), dtype)
        kernel_input["RoPE_sin"] = _uniform_activation((d_head // 2, B, S_tkg), dtype)
        kernel_input["attention_mask"] = attention_mask
        kernel_input["position_ids"] = position_ids.astype(np.uint32)
        kernel_input["eps"] = eps
        kernel_input["sbuf_residual_and_cc"] = False
        for i in range(num_layers):
            kernel_input[f"W_gate_scale_{i}"] = W_gate_scales[i]
            kernel_input[f"W_up_scale_{i}"] = W_up_scales[i]
            kernel_input[f"W_down_scale_{i}"] = W_down_scales[i]

        return kernel_input

    @pytest_parametrize(PARAM_NAMES, ALL_PARAMS, abbrevs=_ABBREVS)
    def test_transformer_tkg(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        tp: int,
        batch: int,
        S_tkg: int,
        S_ctx: int,
        q_heads: int,
        d_head: int,
        H: int,
        I: int,
        lnc: int,
        rel_tol: float,
    ):
        """Test transformer_tkg with various configurations."""
        B = batch

        def input_generator(test_config):
            kernel_input = self.generate_inputs(tp, batch, S_tkg, S_ctx, q_heads, d_head, H, I, lnc)
            kernel_input["replica_groups"] = (tuple(range(tp)),)
            # Disable cache update in separated pass: indirect-addressed DMA scatters
            # trigger OOB errors under the separated scheduler, causing collective deadlocks.
            if os.environ.get("NKILIB_ENABLE_SEPARATION_ANALYSIS") == "1":
                kernel_input["position_ids"] = None
            return kernel_input

        def output_tensors(kernel_input):
            return {"layer_output": np.zeros((B, S_tkg, H), dtype=ml_dtypes.bfloat16)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=nki.jit(transformer_tkg_tp4_4layer_wrapper),
            torch_ref=torch_ref_wrapper(transformer_tkg_tp4_4layer_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        inference_args = InferenceArgs(collective_ranks=tp)
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc, platform_target=platform_target),
            inference_args=inference_args,
            rtol=rel_tol / 100.0,
            atol=1e-2,
        )
