# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from test.integration.nkilib.experimental.transformer.test_attention_block_tkg_model_config import (
    attention_block_tkg_model_configs,
)
from test.integration.nkilib.utils.comparators import maxAllClose
from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
    np_random_sample_static_quantize_inp,
)
from test.utils.common_dataclasses import (
    MODEL_TEST_TYPE,
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.metadata_loader import load_model_configs
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import Any, Optional, final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.utils.allocator import SbufManager
from nkilib_src.nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from nkilib_src.nkilib.core.utils.kernel_helpers import get_program_sharding_info, is_hbm_buffer
from nkilib_src.nkilib.core.utils.tensor_view import TensorView
from nkilib_src.nkilib.experimental.transformer.attention_block_tkg import (
    attention_block_tkg,
)
from typing_extensions import override

# Define test parameters (example values in comments)
TEST_CONFIG = "config"
BATCH_CFG = "batch"  # e.g., 8
NUM_HEADS_CFG = "num_heads"  # e.g., 8
D_HEAD_CFG = "d_head"  # e.g., 64
H_CFG = "H"  # e.g., 6144
H_ACTUAL_CFG = "H_actual"  # e.g., 2880
S_CTX_CFG = "S_ctx"  # e.g., 11264
S_MAX_CTX_CFG = "S_max_ctx"  # e.g., 11264
S_TKG_CFG = "S_tkg"  # e.g., 1
BLOCK_LEN_CFG = "block_len"  # e.g., 0
UPDATE_CACHE_CFG = "update_cache"  # e.g., True
K_CACHE_TRANSPOSED_CFG = "K_cache_transposed"  # e.g., False
RMSNORM_X_CFG = "rmsnorm_X"  # e.g., True
SKIP_ROPE_CFG = "skip_rope"  # e.g., False
ROPE_CONTIGUOUS_LAYOUT_CFG = "rope_contiguous_layout"  # e.g., True
RMSNORM_QK_CFG = "rmsnorm_qk"  # e.g., False
QK_NORM_POST_ROPE_CFG = "qk_norm"  # e.g., False
QK_NORM_POST_ROPE_GAMMA_CFG = "qk_norm_gamma_post_rope"  # e.g., False
DTYPE_CFG = "dtype"  # e.g., nl.bfloat16
LNC_CFG = "lnc"  # e.g., 2
SKIP_OUTPUT_PROJECTION_CFG = "skip_output_projection"
TRANSPOSED_OUT_CFG = "transposed_out"  # e.g., False
TEST_BIAS_CFG = "test_bias"  # e.g., True
OUT_IN_SBUF_CFG = "out_in_sbuf"  # e.g., False"
INPUT_IN_SBUF_CFG = "input_in_sbuf"  # e.g., False"
QUANTIZATION_TYPE = "quantization_type"  # e.g. QuantizationType.NONE
SOFTMAX_SCALE_CFG = "softmax_scale"  # e.g., None - Optional custom softmax scale
KV_QUANT_CFG = "kv_quant"  # e.g., False - FP8 KV cache quantization


def generate_kernel_inputs(test_cfg: dict, skip_attention: bool):
    from test.integration.nkilib.core.attention.test_attention_tkg import (
        gen_deterministic_active_block_table,
        numpy_gen_attention_cache_mask,
    )

    dtype = test_cfg[DTYPE_CFG]
    batch = test_cfg[BATCH_CFG]
    num_heads = test_cfg[NUM_HEADS_CFG]
    d_head = test_cfg[D_HEAD_CFG]
    H = test_cfg[H_CFG]
    H_actual = test_cfg[H_ACTUAL_CFG] if test_cfg[H_ACTUAL_CFG] is not None else H
    S_ctx = test_cfg[S_CTX_CFG]
    S_max_ctx = test_cfg[S_MAX_CTX_CFG]
    S_tkg = test_cfg[S_TKG_CFG]
    out_in_sbuf = test_cfg[OUT_IN_SBUF_CFG]
    num_q_heads = num_heads
    num_kv_heads = 1
    I = d_head * (num_q_heads + 2 * num_kv_heads)

    quantization_type = test_cfg[QUANTIZATION_TYPE]
    lnc = test_cfg[LNC_CFG]
    test_bias = bool(test_cfg[TEST_BIAS_CFG])
    eps = 1e-5 if dtype == np.float32 else 1e-3

    # FP8 KV cache quantization scales - initialized later with generate_quant_tensor
    kv_quant = bool(test_cfg.get(KV_QUANT_CFG, False))

    # Use uniform [-1, 1] for kv_quant (FP8 stability), gaussian otherwise (original behavior)
    if kv_quant:

        def _tensor_generator(shape, dtype, name=None):
            np.random.seed(hash(name) % (2**32) if name else 0)
            arr = np.random.uniform(-1.0, 1.0, shape)
            return dt.static_cast(arr, dtype)
    else:
        _tensor_generator = gaussian_tensor_generator()

    generate_quant_tensor = np_random_sample_static_quantize_inp()

    # Wrap to ensure C-contiguous arrays
    def generate_tensor(name, shape, dtype):
        return np.ascontiguousarray(_tensor_generator(shape, dtype, name))

    # -- input
    X = generate_tensor(name="X", shape=(batch, S_tkg, H), dtype=dtype)
    X[:, :, H_actual:] = 0.0

    # -- rmsnorm X
    rmsnorm_X = bool(test_cfg[RMSNORM_X_CFG])
    rmsnorm_X_gamma = generate_tensor(name="rmsnorm_X_gamma", shape=(1, H), dtype=dtype) if rmsnorm_X else None

    # -- qkv projections, optional bias
    if quantization_type == QuantizationType.NONE:
        W_qkv = generate_tensor(name="W_qkv", shape=(H, (num_q_heads + 2 * num_kv_heads) * d_head), dtype=dtype)
        weight_dequant_scale_qkv = None
        input_dequant_scale_qkv = None
    else:
        W_q, w_scale_q, input_dequant_scale_qkv = generate_quant_tensor(
            shape=(H, num_q_heads * d_head), dtype=nl.float8_e4m3
        )
        W_k, w_scale_k, _ = generate_quant_tensor(shape=(H, num_kv_heads * d_head), dtype=nl.float8_e4m3)
        W_v, w_scale_v, _ = generate_quant_tensor(shape=(H, num_kv_heads * d_head), dtype=nl.float8_e4m3)
        W_qkv = np.concatenate([W_q, W_k, W_v], axis=1)
        weight_dequant_scale_qkv = np.array([[w_scale_q, w_scale_k, w_scale_v]])
        weight_dequant_scale_qkv = np.broadcast_to(weight_dequant_scale_qkv, (128, 3))
        input_dequant_scale_qkv = np.broadcast_to(input_dequant_scale_qkv, (128, 1))
    bias_qkv = (
        generate_tensor(name="bias_qkv", shape=(1, (num_q_heads + 2 * num_kv_heads) * d_head), dtype=dtype)
        if test_bias
        else None
    )

    # -- rmsnorm QK
    rmsnorm_qk = bool(test_cfg[RMSNORM_QK_CFG])
    # -- RoPE
    skip_rope = bool(test_cfg[SKIP_ROPE_CFG])
    rope_contiguous_layout = bool(test_cfg[ROPE_CONTIGUOUS_LAYOUT_CFG])
    cos = None if skip_rope else generate_tensor(name="cos", shape=(d_head // 2, batch, S_tkg), dtype=dtype)
    sin = None if skip_rope else generate_tensor(name="sin", shape=(d_head // 2, batch, S_tkg), dtype=dtype)

    # -- rmsnorm QK post RoPE
    qk_norm_post_rope = bool(test_cfg[QK_NORM_POST_ROPE_CFG])
    qk_norm_gamma_post_rope = bool(test_cfg[QK_NORM_POST_ROPE_GAMMA_CFG])
    W_rmsnorm_Q_post_rope = (
        generate_tensor(name="W_rmsnorm_Q_post_rope", shape=(1, d_head), dtype=dtype)
        if qk_norm_gamma_post_rope
        else None
    )
    W_rmsnorm_K_post_rope = (
        generate_tensor(name="W_rmsnorm_K_post_rope", shape=(1, d_head), dtype=dtype)
        if qk_norm_gamma_post_rope
        else None
    )

    # -- Attention (and KV cache)
    block_len = test_cfg[BLOCK_LEN_CFG]
    is_block_kv = block_len > 0

    # Determine cache shapes
    update_cache = bool(test_cfg[UPDATE_CACHE_CFG])
    K_cache_transposed = bool(test_cfg[K_CACHE_TRANSPOSED_CFG])

    if is_block_kv:
        assert not K_cache_transposed
        assert S_ctx % block_len == 0
        assumed_num_cache_blocks = batch * S_ctx // block_len
        K_cache_shape = V_cache_shape = (assumed_num_cache_blocks, block_len, d_head)
    else:
        assumed_num_cache_blocks = 0
        K_cache_shape = (batch, 1, d_head, S_max_ctx) if K_cache_transposed else (batch, 1, S_max_ctx, d_head)
        V_cache_shape = (batch, 1, S_max_ctx, d_head)

    # Generate KV cache in FP8 when kv_quant=True
    kv_cache_dtype = nl.float8_e4m3 if kv_quant else dtype
    if kv_quant:
        # FP8 scale for K/V from QKV projection (K = X @ W where X,W ~ uniform[-1,1])
        # Var(uniform[-1,1]) = (1-(-1))²/12 = 1/3
        # Var(K) = H * Var(X) * Var(W) = H/9  =>  std(K) = sqrt(H)/3
        # Max ≈ 3.5*std (99.95% coverage), scale = FP8_max / max ≈ 240 / (3.5*sqrt(H)/3)
        # Simplified to 240/sqrt(H) for full FP8 range with minimal clipping
        k_scale_scalar = 240.0 / np.sqrt(H)
        v_scale_scalar = 240.0 / np.sqrt(H)
        # Use different shapes to test both broadcast (1,1) and per-partition (PMAX,1) paths
        k_scale = np.full((1, 1), k_scale_scalar, dtype=np.float32)
        v_scale = np.full((128, 1), v_scale_scalar, dtype=np.float32)

        # Generate KV cache matching the distribution of scaled K/V:
        # K/V are ~Gaussian (CLT: sum of H products) with std(K)*scale = (sqrt(H)/3)*(240/sqrt(H)) = 80
        # H cancels out, so kv_std=80 works for any hidden dimension
        np.random.seed(42)
        kv_std = 80.0
        K_cache_f32 = np.random.normal(0, kv_std, K_cache_shape).astype(np.float32)
        V_cache_f32 = np.random.normal(0, kv_std, V_cache_shape).astype(np.float32)
        K_cache = dt.static_cast(np.clip(K_cache_f32, -240, 240), kv_cache_dtype)
        V_cache = dt.static_cast(np.clip(V_cache_f32, -240, 240), kv_cache_dtype)
    else:
        K_cache = generate_tensor(name="K_cache", shape=K_cache_shape, dtype=kv_cache_dtype)
        V_cache = generate_tensor(name="V_cache", shape=V_cache_shape, dtype=kv_cache_dtype)
        k_scale = None
        v_scale = None

    # pos_id (shape=(batch, 1)) defines the first position to append new KV to cache, per batch element
    cache_len = ((np.arange(batch) * 3 + (S_ctx // 4 * 3)) % (S_ctx - S_tkg))[:, np.newaxis]
    assert cache_len.max() < (S_ctx - S_tkg)  # Make sure not to go out of bound.
    attention_mask = numpy_gen_attention_cache_mask(
        cache_len, batch, num_heads, S_tkg, S_ctx, lnc, block_len, unify_for_cascaded=True
    )  # mask: (S_ctx, batch, num_heads, S_tkg)
    attention_mask = dt.static_cast(np.ascontiguousarray(attention_mask), dtype=np.uint8)

    active_blocks_table = (
        gen_deterministic_active_block_table(
            batch, S_ctx, S_tkg, cache_len, block_len, batch * S_ctx // block_len
        ).astype(np.uint32)
        if is_block_kv
        else None
    )  # (B, S_ctx // block_len)

    # kv_cache_update_idx is (batch,) containing the start position for consecutive writes
    def generate_kv_cache_update_idx():
        if block_len == 0:
            return cache_len.astype(np.uint32)

        # Block KV: translate logical position to physical slot_mapping
        logical_blks = cache_len // block_len
        offset_in_blk = cache_len % block_len
        physical_blks = active_blocks_table[np.arange(batch)[:, None], logical_blks]
        physical_kv_cache_update_idx = physical_blks * block_len + offset_in_blk
        # Mask update for last batch element to test scenario when it is just padding
        if batch > 1:
            physical_kv_cache_update_idx[-1] = -1
        return physical_kv_cache_update_idx.astype(np.uint32)

    kv_cache_update_idx = generate_kv_cache_update_idx()

    # Output projection
    skip_output_projection = test_cfg[SKIP_OUTPUT_PROJECTION_CFG]

    weight_dequant_scale_out = None
    input_dequant_scale_out = None
    if skip_output_projection:
        W_out = None
    elif quantization_type == QuantizationType.NONE:
        W_out = generate_tensor(name="W_out", shape=(num_heads * d_head, H), dtype=dtype)
    else:
        W_out, weight_dequant_scale_out, input_dequant_scale_out = generate_quant_tensor(
            shape=(num_heads * d_head, H), dtype=nl.float8_e4m3
        )
        weight_dequant_scale_out = np.broadcast_to(weight_dequant_scale_out, (128, 1))
        input_dequant_scale_out = np.broadcast_to(input_dequant_scale_out, (128, 1))

    bias_out = generate_tensor(name="bias_out", shape=(1, H), dtype=dtype) if test_bias else None
    transposed_out = bool(test_cfg[TRANSPOSED_OUT_CFG])

    return {
        # -- input
        "X": X,
        "X_in_sb": test_cfg[INPUT_IN_SBUF_CFG],
        "X_hidden_dim_actual": H_actual,
        # -- rmsnorm X
        "rmsnorm_X_enabled": rmsnorm_X,
        "rmsnorm_X_eps": eps,
        "rmsnorm_X_gamma": rmsnorm_X_gamma,
        # -- qkv projections
        "W_qkv": W_qkv,
        "bias_qkv": bias_qkv,
        "quantization_type_qkv": quantization_type,
        "weight_dequant_scale_qkv": weight_dequant_scale_qkv,
        "input_dequant_scale_qkv": input_dequant_scale_qkv,
        # -- QK rmsnorm pre RoPE
        "rmsnorm_QK_enabled": rmsnorm_qk,
        "rmsnorm_QK_eps": eps,
        # -- RoPE
        "cos": cos,
        "sin": sin,
        "rope_contiguous_layout": rope_contiguous_layout,
        # -- QK rmsnorm post RoPE
        "rmsnorm_QK_post_rope_enabled": qk_norm_post_rope,
        "rmsnorm_QK_post_rope_eps": eps,
        "W_rmsnorm_Q_post_rope": W_rmsnorm_Q_post_rope,
        "W_rmsnorm_K_post_rope": W_rmsnorm_K_post_rope,
        # -- attention
        "skip_attention": skip_attention,
        "K_cache_transposed": K_cache_transposed,
        "active_blocks_table": active_blocks_table,
        "K_cache": K_cache,
        "V_cache": V_cache,
        "attention_mask": attention_mask,
        "sink": None,
        "softmax_scale": test_cfg.get(SOFTMAX_SCALE_CFG),
        # -- FP8 KV cache quantization
        "k_scale": k_scale,
        "v_scale": v_scale,
        # -- KV cache update
        "update_cache": update_cache,
        "kv_cache_update_idx": kv_cache_update_idx,
        # -- output projection
        "W_out": W_out,
        "bias_out": bias_out,
        "quantization_type_out": quantization_type,
        "weight_dequant_scale_out": weight_dequant_scale_out,
        "input_dequant_scale_out": input_dequant_scale_out,
        # -- output
        "transposed_out": transposed_out,
        "out_in_sb": out_in_sbuf,
    }


def golden_ref_attn_blk(
    inp_np,
    dtype,
    d_head,
    num_q_heads,
    block_len,
    S_ctx,
    S_max_ctx,
    lnc,
    quantization_type,
):
    from test.integration.nkilib.core.attention.test_attention_tkg import P_MAX, AttnTKGConfig, golden_attention_tkg_fwd
    from test.integration.nkilib.core.embeddings.test_rope import rope_single_head
    from test.integration.nkilib.core.qkv.test_qkv_tkg import golden_func_fused_add_qkv

    from nkilib_src.nkilib.core.attention.attention_tkg import resize_cache_block_len_for_attention_tkg_kernel

    # -- input
    X = inp_np['X'].astype(np.float32)
    hidden_actual = inp_np['X_hidden_dim_actual']
    batch, S_tkg, H = X.shape
    num_heads = num_q_heads
    skip_attention = inp_np['skip_attention']

    # -- rmsnorm X and qkv projections
    rmsnorm_X = inp_np['rmsnorm_X_enabled']
    rmsnorm_X_eps = inp_np['rmsnorm_X_eps']
    rmsnorm_X_gamma = (
        inp_np['rmsnorm_X_gamma'].astype(np.float32) if rmsnorm_X and inp_np['rmsnorm_X_gamma'] is not None else None
    )

    W_qkv = inp_np['W_qkv'].astype(np.float32)
    bias_qkv = inp_np['bias_qkv'].astype(np.float32) if inp_np['bias_qkv'] is not None else None

    qkv_input_np = {
        "input": X,
        "mlp_prev": None,
        "attention_prev": None,
        "fused_qkv_weights": W_qkv,
        "gamma_norm_weights": rmsnorm_X_gamma,
        "bias": bias_qkv,
        "layer_norm_bias": None,
        "qkv_w_scale": inp_np["weight_dequant_scale_qkv"],
        "qkv_in_scale": inp_np["input_dequant_scale_qkv"],
    }

    # Note: result in NBSd here, to fit the RoPE golden.
    #       The kernel outputs qkv in B_SxI though (where I=d*(q_heads+k_heads+v_heads))
    QKV_dict = golden_func_fused_add_qkv(
        inp_np=qkv_input_np,
        dtype=np.float32,
        fused_add=False,
        norm_type=NormType.RMS_NORM if rmsnorm_X else NormType.NO_NORM,
        eps=rmsnorm_X_eps,
        head_dim=d_head,
        n_kv_heads=1,
        n_q_heads=num_q_heads,
        output_layout=QKVOutputLayout.NBSd,
        hidden_actual=hidden_actual,
        quantization_type=quantization_type,
    )
    QKV = QKV_dict['out'].astype(np.float32)

    # -- QK rmsnorm pre RoPE
    def rms_norm(x, axis, eps, w=None):
        normalized = x / np.sqrt((np.mean((x**2), axis=axis, keepdims=True) + eps))
        if w is not None:
            normalized = normalized * w
        return normalized

    rmsnorm_qk = inp_np['rmsnorm_QK_enabled']
    rmsnorm_qk_eps = inp_np['rmsnorm_QK_eps']
    if rmsnorm_qk:
        for i in range(num_heads + 1):
            QKV[i] = rms_norm(QKV[i], axis=-1, eps=rmsnorm_qk_eps)

    # -- RoPE
    cos = inp_np['cos'].astype(np.float32) if inp_np['cos'] is not None else None
    sin = inp_np['sin'].astype(np.float32) if inp_np['sin'] is not None else None
    rope_contiguous_layout = inp_np['rope_contiguous_layout']

    # RoPE on Q and K in NBSd layout --> dBNS layout
    skip_rope = cos is None or sin is None
    QK = np.zeros((d_head, batch, num_heads + 1, S_tkg))  # Do Q & K together.
    for b in range(batch):
        for h in range(num_heads + 1):
            cur_head = QKV[h, b, :, :].transpose(1, 0)  # d_S
            QK[:, b, h, :] = (
                cur_head
                if skip_rope
                else rope_single_head(cur_head, cos[:, b, :], sin[:, b, :], contiguous_layout=rope_contiguous_layout)
            )

    Q = QK[:, :, :num_heads, :]  # d_B_N_S
    K = QK[:, :, num_heads, :]  # d_B_S

    # -- QK rmsnorm post RoPE
    qk_norm_post_rope = inp_np['rmsnorm_QK_post_rope_enabled']
    rmsnorm_QK_post_rope_eps = inp_np['rmsnorm_QK_post_rope_eps']
    W_rmsnorm_Q_post_rope = (
        inp_np['W_rmsnorm_Q_post_rope'].astype(np.float32)
        if qk_norm_post_rope and inp_np['W_rmsnorm_Q_post_rope'] is not None
        else None
    )
    W_rmsnorm_K_post_rope = (
        inp_np['W_rmsnorm_K_post_rope'].astype(np.float32)
        if qk_norm_post_rope and inp_np['W_rmsnorm_K_post_rope'] is not None
        else None
    )

    if qk_norm_post_rope:
        w_Q = W_rmsnorm_Q_post_rope.reshape((d_head, 1, 1, 1)) if W_rmsnorm_Q_post_rope is not None else None
        w_K = W_rmsnorm_K_post_rope.reshape((d_head, 1, 1)) if W_rmsnorm_K_post_rope is not None else None
        Q = rms_norm(Q, axis=0, eps=rmsnorm_QK_post_rope_eps, w=w_Q)
        K = rms_norm(K, axis=0, eps=rmsnorm_QK_post_rope_eps, w=w_K)

    output = Q.copy()  # may be later overriden
    # Last head of QKV which is NBSd where N is q_heads + 2 * kv_heads and we assume k and v are one head
    V = QKV[-1]  # BSd

    # FP8 KV cache quantization
    kv_quant = inp_np.get('k_scale') is not None and inp_np.get('v_scale') is not None
    if kv_quant:

        def quantize_to_fp8(x, scale, target_dtype=nl.float8_e4m3):
            FP8_E4M3_MAX = 240.0
            x_scaled = np.multiply(x, scale[0, 0])
            clipped_ratio = np.mean(np.abs(x_scaled) > FP8_E4M3_MAX)
            assert clipped_ratio <= 0.01, f"Too many values clipped ({clipped_ratio:.1%}), scale may be inappropriate"
            return dt.static_cast(np.clip(x_scaled, -FP8_E4M3_MAX, FP8_E4M3_MAX), target_dtype)

        k_scale = inp_np['k_scale'].astype(np.float32)
        v_scale = inp_np['v_scale'].astype(np.float32)
        # Cast to bf16 to match kernel precision before quantization
        K = dt.static_cast(K, nl.bfloat16).astype(np.float32)
        V = dt.static_cast(V, nl.bfloat16).astype(np.float32)
        K = quantize_to_fp8(K, k_scale)
        V = quantize_to_fp8(V, v_scale)
        K_cache = inp_np['K_cache'].astype(nl.float8_e4m3).copy()
        V_cache = inp_np['V_cache'].astype(nl.float8_e4m3).copy()
    else:
        K_cache = inp_np['K_cache'].astype(np.float32).copy()
        V_cache = inp_np['V_cache'].astype(np.float32).copy()

    # Save K, V before transpose for return
    K_new = K.copy()
    V_new = V.copy()

    # -- attention
    is_block_kv = block_len > 0
    K_cache_transposed = inp_np['K_cache_transposed']
    active_blocks_table = (
        inp_np['active_blocks_table'].astype(np.int32) if inp_np['active_blocks_table'] is not None else None
    )
    update_cache = inp_np['update_cache']
    out_in_sb = inp_np['out_in_sb']

    q_attn = Q.reshape(d_head, batch, num_heads, S_tkg).transpose(1, 2, 3, 0)  # q_attn.shape=(B,N,S,d)
    k_attn_active = K.reshape(d_head, batch, 1, S_tkg).transpose(1, 2, 3, 0)  # k_attn_active.shape=(B,1,S,d)
    v_attn_active = V.reshape(batch, 1, S_tkg, d_head)  # (b,1,S,d)

    if skip_attention:
        attn_out_B_N_d_S = Q.reshape(d_head, batch, num_heads, S_tkg).transpose(1, 2, 0, 3)
    else:
        _softmax_scale = inp_np.get('softmax_scale')
        if _softmax_scale is not None:
            q_attn = q_attn * _softmax_scale
        else:
            q_attn = q_attn / np.sqrt(d_head)

        attention_mask = dt.static_cast(inp_np['attention_mask'], nl.uint8)

        attention_tkg_input = {
            'q': q_attn,
            'k_active': k_attn_active,
            'v_active': v_attn_active,
            'k_prior': K_cache,
            'v_prior': V_cache,
            'active_mask': attention_mask,
            'active_blocks_table': active_blocks_table,
        }

        AttnCfg = AttnTKGConfig(
            bs=batch,
            q_head=num_heads,
            s_active=S_tkg,
            curr_sprior=S_ctx,
            full_sprior=S_max_ctx,
            d_head=d_head,
            block_len=block_len,
            tp_k_prior=not K_cache_transposed,
            strided_mm1=not is_block_kv,
            use_pos_id=False,
            fuse_rope=False,
        )

        attn_golden_output = golden_attention_tkg_fwd(
            inp_np=attention_tkg_input,
            cfg=AttnCfg,
            is_block_kv=is_block_kv,
            test_sink=False,
            dtype=np.float32,  # avoids cast here to lower precision
            lnc=lnc,
            attn_out_shape=(batch, num_heads, d_head, S_tkg),
            k_out_shape=None,
            relative_tolerance=0,  # unused. This doesn't belong here!
            absolute_tolerance=0,  # unused. This doesn't belong here!
            DBG=False,
        )
        attn_out_B_N_d_S = attn_golden_output["golden_out"].astype(np.float32)
        output = attn_out_B_N_d_S.transpose(2, 0, 1, 3) if out_in_sb else attn_out_B_N_d_S  # may be overriden

    # -- update KV cache
    kv_cache_update_idx = inp_np['kv_cache_update_idx']
    if update_cache:
        if is_block_kv:
            # Update to the original block KV.
            num_blocks = K_cache.shape[0]  # K_cache.shape=(blocks, block_len, d)
            K_cache = K_cache.reshape((num_blocks * block_len, d_head))
            V_cache = V_cache.reshape((num_blocks * block_len, d_head))
            for b in range(batch):
                if kv_cache_update_idx[b] == np.uint32(-1):
                    continue

                start_pos = kv_cache_update_idx[b, 0]
                K_cache[start_pos : start_pos + S_tkg, :] = K_new[:, b, :].T
                V_cache[start_pos : start_pos + S_tkg, :] = V_new[b, :, :]
            K_out = K_cache.reshape(num_blocks, block_len, d_head)
            V_out = V_cache.reshape(num_blocks, block_len, d_head)
        else:  # not block_kv
            for b in range(batch):
                # Skip batch elements with invalid kv_cache_update_idx (marked as uint32(-1)).
                # This logic is not in the kernel. It doesn't work for unclear reason
                # if kv_cache_update_idx[b] == np.uint32(-1):
                #     continue

                start_pos = kv_cache_update_idx[b, 0]
                # V cache is in (B,1,S,d). K cache is in (B,1,d,S) if transposed else (B,1,S,d)
                if K_cache_transposed:
                    K_cache[b, 0, :, start_pos : start_pos + S_tkg] = K_new[:, b, :]  # K_new is (d,B,S_tkg)
                else:
                    K_cache[b, 0, start_pos : start_pos + S_tkg, :] = K_new[:, b, :].T  # K_new is (d,B,S_tkg)
                V_cache[b, 0, start_pos : start_pos + S_tkg, :] = V_new[b, :, :]  # V_new is (B,S_tkg,d)
            K_out, V_out = K_cache, V_cache
    else:  # no cache update
        K_out, V_out = K_new, V_new

    # -- output projection
    skip_output_projection = inp_np['W_out'] is None
    W_out = None if skip_output_projection else inp_np['W_out'].astype(np.float32)
    weight_dequant_scale_out = (
        None
        if quantization_type == QuantizationType.NONE
        else inp_np["weight_dequant_scale_out"][0, 0].astype(np.float32)
    )
    input_dequant_scale_out = (
        None
        if quantization_type == QuantizationType.NONE
        else inp_np["input_dequant_scale_out"][0, 0].astype(np.float32)
    )
    bias_out = inp_np['bias_out'].astype(np.float32) if inp_np['bias_out'] is not None else None
    transposed_out = inp_np['transposed_out']

    if not skip_output_projection:
        attn_out_BS_Nd = attn_out_B_N_d_S.transpose(0, 3, 1, 2).reshape((batch * S_tkg, num_heads * d_head))
        if quantization_type == QuantizationType.STATIC:
            attn_out_BS_Nd /= input_dequant_scale_out
            attn_out_BS_Nd = attn_out_BS_Nd.clip(-240, 240)
        output = (attn_out_BS_Nd @ W_out).astype(dtype)  # -> (B*S_tkg, H)
        if quantization_type == QuantizationType.STATIC:
            output *= weight_dequant_scale_out * input_dequant_scale_out
        if bias_out is not None:
            output += bias_out
        if transposed_out:
            # relayout to (B*S_tkg, H) --> (PMAX, lnc, H // lnc // PMAX, B*S_tkg)
            output = output.reshape(batch * S_tkg, lnc, 128, H // lnc // 128).transpose((2, 1, 3, 0))

    # -- Outputs
    kv_dtype = nl.float8_e4m3 if kv_quant else dtype
    if update_cache:
        return {
            "X_out": dt.static_cast(output, dtype),
            "K_cache_updated": dt.static_cast(K_out, kv_dtype),  # updated K cache
            "V_cache_updated": dt.static_cast(V_out, kv_dtype),  # update V cache
        }

    return {
        "X_out": dt.static_cast(output, dtype),
        "K_tkg": dt.static_cast(K_out, kv_dtype),  # generated tokens
        "V_tkg": dt.static_cast(V_out, kv_dtype),  # generated tokens
    }


# wrapper to test SBUF IO
def attention_block_tkg_kernel_test_wrapper(
    # -- input
    X: nl.ndarray,
    *,
    X_in_sb: bool,
    X_hidden_dim_actual: Optional[int],
    # -- rmsnorm X
    rmsnorm_X_enabled: bool,
    rmsnorm_X_eps: Optional[float],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    # -- qkv projections
    W_qkv: nl.ndarray,
    bias_qkv: Optional[nl.ndarray],
    quantization_type_qkv: QuantizationType,
    weight_dequant_scale_qkv: Optional[nl.ndarray],
    input_dequant_scale_qkv: Optional[nl.ndarray],
    # -- QK rmsnorm pre RoPE
    rmsnorm_QK_enabled: bool,
    rmsnorm_QK_eps: Optional[float],
    # -- RoPE embeddings
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    # -- QK rmsnorm post RoPE
    rmsnorm_QK_post_rope_enabled: bool,
    rmsnorm_QK_post_rope_eps: float,
    W_rmsnorm_Q_post_rope: Optional[nl.ndarray],
    W_rmsnorm_K_post_rope: Optional[nl.ndarray],
    # -- attention
    skip_attention: bool,
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    sink: Optional[nl.ndarray],
    softmax_scale: Optional[float],
    # -- FP8 KV cache quantization
    k_scale: Optional[nl.ndarray],
    v_scale: Optional[nl.ndarray],
    # -- KV cache update
    update_cache: bool,
    kv_cache_update_idx: nl.ndarray,
    # -- output projection
    W_out: Optional[nl.ndarray],
    bias_out: Optional[nl.ndarray],
    quantization_type_out: QuantizationType,
    weight_dequant_scale_out: Optional[nl.ndarray],
    input_dequant_scale_out: Optional[nl.ndarray],
    # -- output
    transposed_out: bool,
    out_in_sb: bool,
    sbm: Optional[SbufManager] = None,
):
    B, S_tkg, H = X.shape
    if X_in_sb:
        # QKV_tkg requires the input shape to be (pmax, B*S, H // pmax)
        assert H % 128 == 0, "H must be divisible by 128"
        H0 = nl.tile_size.pmax
        H1 = H // 128
        BxS = B * S_tkg

        # Check program dimensionality
        _, lnc, _ = get_program_sharding_info()
        assert H1 % lnc == 0

        X_sb = nl.ndarray((H0, BxS, H1), X.dtype, nl.sbuf, name="X_sb")
        X_hbm = X.reshape((BxS, lnc, H0, H1 // lnc))

        """
        Note how X@HBM is read to SBUF: The full H dimension is divided into (lnc, H0=128, H1//lnc).
        Per SBUF partition (the H0=128 dim), we read H1//lnc values from each of the lnc chunks,
        interleaving them to reconstruct the full H1 dimension in SBUF while transposing the layout
        from (BxS, lnc, H0, H1//lnc) to (H0, BxS, H1). This matches how qkv_tkg() kernel expects
        SBUF input and constrains attention_block_tkg() SBUF input layout.
        """
        nisa.dma_copy(
            dst=TensorView(X_sb).reshape_dim(2, (lnc, -1)).get_view(),
            src=TensorView(X_hbm)
            .rearrange(('BS', 'lnc', 'H0', 'H1 // lnc'), ('H0', 'BS', 'lnc', 'H1 // lnc'))
            .get_view(),
        )

        X = X_sb

    kernel_output, K_hbm_out, V_hbm_out = attention_block_tkg(
        X=X,
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=rmsnorm_X_enabled,
        rmsnorm_X_eps=rmsnorm_X_eps,
        rmsnorm_X_gamma=rmsnorm_X_gamma,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=quantization_type_qkv,
        weight_dequant_scale_qkv=weight_dequant_scale_qkv,
        input_dequant_scale_qkv=input_dequant_scale_qkv,
        rmsnorm_QK_pre_rope_enabled=rmsnorm_QK_enabled,
        rmsnorm_QK_pre_rope_eps=rmsnorm_QK_eps if rmsnorm_QK_eps else 1e-5,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled=rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps=rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q=W_rmsnorm_Q_post_rope,
        rmsnorm_QK_post_rope_W_K=W_rmsnorm_K_post_rope,
        skip_attention=skip_attention,
        K_cache_transposed=K_cache_transposed,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        attention_mask=attention_mask,
        sink=sink,
        softmax_scale=softmax_scale,
        update_cache=update_cache,
        kv_cache_update_idx=kv_cache_update_idx,
        k_scale=k_scale,
        v_scale=v_scale,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=quantization_type_out,
        weight_dequant_scale_out=weight_dequant_scale_out,
        input_dequant_scale_out=input_dequant_scale_out,
        transposed_out=transposed_out,
        out_in_sb=out_in_sb,
        sbm=sbm,
    )

    assert is_hbm_buffer(K_hbm_out)
    assert is_hbm_buffer(V_hbm_out)

    if not out_in_sb:
        return kernel_output, K_hbm_out, V_hbm_out

    assert kernel_output.buffer == nl.sbuf, "Expecting output on SBUF"

    # copy output to HBM
    skip_output_projection = W_out is None
    if skip_output_projection:
        kernel_output_hbm = nl.ndarray(kernel_output.shape, kernel_output.dtype, nl.hbm, name="kernel_output_hbm")
        nisa.dma_copy(kernel_output_hbm, kernel_output)
    else:
        kernel_output_hbm = relayout_sbuf_to_hbm_for_output_projection(kernel_output, transposed_out, B, S_tkg, H)

    return kernel_output_hbm, K_hbm_out, V_hbm_out


def relayout_sbuf_to_hbm_for_output_projection(kernel_output, transposed_out, B, S_tkg, H):
    # if transposed: SBUF.layout=(PMAX, H // lnc // PMAX, B*S_tkg) and HBM.layout=(PMAX, lnc, H // lnc // PMAX, B*S_tkg)
    # else: SBUF.layout=(B*S_tkg, H // lnc) and HBM.layout=(B*S_tkg, H)

    # Note: this code is based on the output_projection_tkg() logic
    _, n_prgs, prg_id = get_program_sharding_info()
    if transposed_out:
        H0, H1, H2 = n_prgs, nl.tile_size.pmax, H // n_prgs // nl.tile_size.pmax
        kernel_output_hbm = nl.ndarray(
            (H1, H0, H2, B * S_tkg), kernel_output.dtype, nl.shared_hbm, name="kernel_output_hbm"
        )
        nisa.dma_copy(
            dst=kernel_output_hbm.ap(
                pattern=[
                    [H0 * H2 * B * S_tkg, H1],
                    [B * S_tkg, H2],
                    [1, B * S_tkg],
                ],
                offset=prg_id * H2 * B * S_tkg,
            ),
            src=kernel_output,
        )
        return kernel_output_hbm

    # Else, not transposed out
    kernel_output_hbm = nl.ndarray((B * S_tkg, H), kernel_output.dtype, nl.shared_hbm, name="kernel_output_hbm")
    H_sharded = H // n_prgs
    nisa.dma_copy(kernel_output_hbm[:, nl.ds(prg_id * H_sharded, H_sharded)], kernel_output)
    return kernel_output_hbm


# FP8 KV cache validation: cosine similarity catches directional drift from mixed-precision
# (fp8/bf16 kernel vs fp32 golden), while allclose with min_pass_rate catches per-element errors.
# Both are needed because cosine similarity alone misses uniform scaling errors, and allclose
# alone is too strict for the accumulated rounding from FP8 quantization boundaries.
def make_cosine_similarity_validator(
    golden: npt.NDArray[Any], rtol: float, atol: float, min_cosine_similarity: float, min_pass_rate: float, name: str
) -> type[CustomValidator]:
    """Create a validator that checks cosine similarity and allclose with a minimum pass rate."""
    _golden = golden
    _rtol = rtol
    _atol = atol
    _min_cos = min_cosine_similarity
    _min_pass_rate = min_pass_rate
    _name = name
    _shape = golden.shape
    _dtype = golden.dtype

    class CosineValidator(CustomValidator):
        @override
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            actual = inference_output.view(_dtype).reshape(_shape).astype(np.float32)
            expected = _golden.astype(np.float32)

            # Cosine similarity on flattened vectors
            a, b = actual.flatten(), expected.flatten()
            cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

            # Allclose with min_pass_rate
            allclose_pass = maxAllClose(
                actual, expected, rtol=_rtol, atol=_atol, verbose=1, logfile=self.logfile, min_pass_rate=_min_pass_rate
            )

            self._print_with_log(
                f"Validating {_name}: cosine_similarity={cos_sim:.6f} (min={_min_cos}), "
                f"allclose(pass_rate>={_min_pass_rate})={allclose_pass}"
            )
            return cos_sim >= _min_cos and allclose_pass

    return CosineValidator


def _run_attention_block_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    batch: int,
    num_heads: int,
    d_head: int,
    H: int,
    H_actual: int,
    S_ctx: int,
    S_max_ctx: int,
    S_tkg: int,
    block_len: int,
    update_cache: bool,
    K_cache_transposed: bool,
    rmsnorm_X: bool,
    skip_rope: bool,
    rope_contiguous_layout: bool,
    rmsnorm_QK: bool,
    qk_norm_post_rope: bool,
    qk_norm_post_rope_gamma: bool,
    dtype,
    quantization_type: QuantizationType,
    lnc: int,
    transposed_out: bool,
    test_bias: bool,
    input_in_sb: bool,
    output_in_sb: bool,
    softmax_scale,
    kv_quant: bool,
    skip_output_projection: bool = False,
    skip_attention: bool = False,
):
    """Shared test execution logic for attention block TKG kernel."""
    test_cfg = {
        BATCH_CFG: batch,
        NUM_HEADS_CFG: num_heads,
        D_HEAD_CFG: d_head,
        H_CFG: H,
        H_ACTUAL_CFG: H_actual,
        S_CTX_CFG: S_ctx,
        S_MAX_CTX_CFG: S_max_ctx,
        S_TKG_CFG: S_tkg,
        BLOCK_LEN_CFG: block_len,
        UPDATE_CACHE_CFG: update_cache,
        K_CACHE_TRANSPOSED_CFG: K_cache_transposed,
        RMSNORM_X_CFG: rmsnorm_X,
        SKIP_ROPE_CFG: skip_rope,
        ROPE_CONTIGUOUS_LAYOUT_CFG: rope_contiguous_layout,
        RMSNORM_QK_CFG: rmsnorm_QK,
        QK_NORM_POST_ROPE_CFG: qk_norm_post_rope,
        QK_NORM_POST_ROPE_GAMMA_CFG: qk_norm_post_rope_gamma,
        DTYPE_CFG: dtype,
        LNC_CFG: lnc,
        SKIP_OUTPUT_PROJECTION_CFG: skip_output_projection,
        TRANSPOSED_OUT_CFG: transposed_out,
        TEST_BIAS_CFG: test_bias,
        OUT_IN_SBUF_CFG: output_in_sb,
        INPUT_IN_SBUF_CFG: input_in_sb,
        QUANTIZATION_TYPE: quantization_type,
        SOFTMAX_SCALE_CFG: softmax_scale,
        KV_QUANT_CFG: kv_quant,
    }

    kernel_input = generate_kernel_inputs(test_cfg, skip_attention)

    def create_golden():
        return golden_ref_attn_blk(
            kernel_input, dtype, d_head, num_heads, block_len, S_ctx, S_max_ctx, lnc, quantization_type
        )

    golden_outputs = create_golden()
    output_placeholder = {k: np.zeros_like(v) for k, v in golden_outputs.items()}

    rtol = 0.07 if kv_quant or quantization_type != QuantizationType.NONE else 0.015
    atol = 16.0 if kv_quant or quantization_type != QuantizationType.NONE else 1e-5

    if kv_quant:
        validation_args = ValidationArgs(
            golden_output={
                name: CustomValidatorWithOutputTensorData(
                    validator=make_cosine_similarity_validator(
                        golden,
                        rtol=rtol,
                        atol=atol,
                        min_cosine_similarity=0.97,
                        min_pass_rate=0.97 if name == 'X_out' else 1.0,
                        name=name,
                    ),
                    output_ndarray=output_placeholder[name],
                )
                for name, golden in golden_outputs.items()
            }
        )
    else:
        validation_args = ValidationArgs(
            golden_output=LazyGoldenGenerator(
                lazy_golden_generator=create_golden,
                output_ndarray=output_placeholder,
            ),
            relative_accuracy=rtol,
            absolute_accuracy=atol,
        )

    test_manager.execute(
        KernelArgs(
            kernel_func=nki.jit(attention_block_tkg_kernel_test_wrapper),
            compiler_input=CompilerArgs(logical_nc_config=lnc, enable_birsim=False, platform_target=platform_target),
            kernel_input=kernel_input,
            validation_args=validation_args,
            inference_args=TKG_INFERENCE_ARGS,
        )
    )


@pytest_test_metadata(
    name="Attention Block TKG",
    pytest_marks=["attention", "tkg", "experimental"],
    tags=["model", "trn2", "trn3"],
)
@final
class TestRangeAttnBlk:
    # fmt: off
    @pytest.mark.parametrize(
            "batch, num_heads,  d_head, H,      H_actual,   S_ctx,  S_max_ctx,  S_tkg,  block_len,  update_cache,   K_cache_transposed, rmsnorm_X,  skip_rope,  rope_contiguous_layout, rmsnorm_QK, qk_norm_post_rope,  qk_norm_post_rope_gamma,    dtype,   quantization_type,       lnc,    transposed_out, test_bias,  input_in_sb,    output_in_sb,   softmax_scale,  kv_quant",
        [
            # SBUF IO
            (4,     8,          64,     6144,   2880,       11264,  11264,      1,      0,          True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE,  2,      False,          False,      False,          True,           None,           False),
            (4,     8,          64,     6144,   2880,       11264,  11264,      1,      0,          True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE,  2,      True,           False,      False,          True,           None,           False),
            (4,     8,          64,     6144,   2880,       11264,  11264,      2,      0,          False,          False,              False,      True,       True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE,  2,      False,          False,      True,           True,           None,           False),
            # HBM IO
            (4,     8,          64,     6144,   2880,       11264,  11264,      1,      0,          True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           False),
            (4,     8,          128,    6144,   2880,       11264,  11264,      1,      0,          True,           False,              True,       False,      True,                   True,       True,               True,                       nl.bfloat16,   QuantizationType.NONE,  2,      False,          True,       False,          False,          None,           False),
            (4,     1,          128,     6144,   2880,       10240,  10240,      5,      0,         False,          False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           False),
            # GPT OSS RIV'25
            # (8,     8,          64,     6144,   2880,       11264,  11264,      1,      0,          True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE,  2,      False,          True,       False,          False),     # FAIL (accuracy ~35% for hidden-out. Others valid)
            (8,     8,          64,     3072,   2880,       11264,  11264,      5,      0,          True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (8,     8,          64,     3072,   2880,       10240,  10240,      5,      0,          True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (4,     8,          64,     3072,   None,       10240,  10240,      4,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (4,     8,          64,     3072,   2880,       10240,  10240,      4,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            # Qwen3
            (16,    1,          128,    4096,   None,       10240,  10240,      1,      0,          True,           False,              True,       False,      True,                   True,       False,              False,                      nl.bfloat16,   QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           False),
            # New model, 2025-Jul
            (32,    1,          64,     3072,   None,       8192,   8192,       1,      0,          True,           False,              False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (32,    1,          64,     3072,   None,       8192,   8192,       1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (64,    1,          64,     3072,   None,       8192,   8192,       1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (32,    1,          64,     3072,   None,       128,    128,        1,      0,          True,           False,              False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (32,    1,          64,     3072,   None,       128,    128,        1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (64,    1,          64,     3072,   None,       128,    128,        1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (1,     1,          64,     3072,   None,       128,    128,        1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (4,     8,          64,     3072,   None,       8192,   8192,       1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (8,     8,          64,     3072,   None,       8192,   8192,       1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (16,    8,          64,     3072,   None,       8192,   8192,       1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          True,       False,          False,          None,           False),
            (32,    1,          64,     3072,   None,       8192,   8192,       3,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      True,           True,       False,          False,          None,           False),
            (4,     8,          64,     3072,   None,       8192,   8192,       2,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      True,           True,       False,          False,          None,           False),
            # secret text
            (1,     1,          128,    7168,   None,       256,    256,        1,      0,          True,           True,               True,       False,      True,                   False,      True,               True,                       nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            # llama
            (1,     1,          128,    8192,   None,       8192,   8192,       1,      0,          True,           True,               False,      False,      True,                   False,      True,               False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (1,     1,          128,    8192,   None,       8192,   8192,       1,      0,          True,           True,               False,      True,       True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (1,     1,          128,    8192,   None,       8192,   8192,       1,      0,          True,           True,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (1,     1,          128,    8192,   None,       8192,   8192,       1,      0,          True,           True,               False,      False,      False,                  False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (1,     1,          128,    5120,   None,       8192,   8192,       1,      0,          True,           True,               True,       False,      False,                  False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (1,     1,          128,    5120,   None,       8192,   8192,       1,      0,          True,           True,               True,       True,       False,                  False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (4,     1,          128,    8192,   None,       10240,  16384,      5,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (4,     1,          128,    8192,   None,       10240,  10240,      5,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (8,     2,          128,    16384,  None,       2048,   2048,       7,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            # (4,     2,          128,    8192,   None,       16384,  16640,      5,      0,          False,          False,              False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 1,      False,          False,      False,          False),     # FAIL (LNC=1 not yet supported)
            (1,     16,         128,    16384,  None,       4096,   8192,       7,      0,          True,           False,              False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           False),
            # (1,     16,         128,    16384,  None,       4096,   8192,       7,      0,          True,           False,              False,      False,      False,                  False,      False,              False,                      nl.bfloat16,  QuantizationType.NONE,  2,      False,          False,      False,          False),     # FAIL (accuracy ~35% for hidden-out. Others valid)
            (8,     2,          128,    16384,  None,       2048,   2048,       7,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      True,           False,      False,          False,          None,           False),
            # # Test vectors for block KV
            (4,     1,          128,    8192,   None,       256,    256,        5,      16,         True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (4,     1,          128,    8192,   None,       8192,   8192,       5,      16,         True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (4,     1,          128,    8192,   None,       12288,  12288,      5,      16,         True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (4,     1,          128,    8192,   None,       10240,  10240,      5,      16,         True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            # Test vectors to verify functionality of different num_heads, d_head and H dimensions
            (2,     1,          128,    2048,   None,       10240,  16384,      5,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (2,     1,          64,     2048,   None,       10240,  16384,      5,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (2,     2,          64,     3072,   None,       10240,  16384,      5,      0,          True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (2,     3,          64,     4096,   None,       10240,  16384,      5,      0,          False,          False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (2,     4,          128,    6144,   None,       10240,  16384,      5,      0,          False,          True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (2,     3,          128,    20480,  None,       10240,  16384,      5,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            (4,     1,          128,    8192,   None,       10240,  10240,      5,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          None,           False),
            # static quantization tests
            # TODO: random input causing numerical instability for quantized weights, more tests will be added
            # after better fp8 random generator is implemented
            # E2E inference tests shows good accuracy
            (8,     1,          128,    8192,   None,       2048,  2048,      5,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.STATIC,  2,      False,          False,      False,          False,          None,           False),
            # softmax_scale tests (Gemma model support)
            (4,     8,          64,     3072,   2880,       10240,  10240,      4,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,   QuantizationType.NONE, 2,      False,          True,       False,          False,          0.05,           False),
            (1,     1,          128,    7168,   None,       256,    256,        1,      0,          True,           True,               True,       False,      True,                   False,      True,               True,                       nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          0.09,           False),
            (1,     1,          128,    5120,   None,       8192,   8192,       1,      0,          True,           True,               True,       True,       False,                  False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          0.13,           False),
            (4,     1,          128,    8192,   None,       8192,   8192,       5,      16,         True,           False,              True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          0.17,           False),
            (4,     1,          128,    8192,   None,       10240,  10240,      5,      0,          True,           True,               True,       False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE, 2,      False,          False,      False,          False,          0.21,           False),
            # llama FP8 KV Cache Tests
            (2,     1,          128,    8192,   None,       8192,   8192,       1,      0,          True,            False,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           True),
            (37,    1,          128,    8192,   None,       8192,   8192,       1,      0,          True,            True,                False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           True),
            (96,    1,          128,    8192,   None,       8192,   8192,       1,      0,          True,            True,                False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           True),
            # llama FP8 KV Cache Tests - batched cache update
            (32,    1,          128,    8192,   None,       4096,   4096,       1,      0,          True,            False,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           True),
            (128,   1,          128,    8192,   None,       2048,   2048,       1,      0,          True,            False,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           True),
            # llama FP8 KV Cache Tests - block KV cache
            (16,    1,          128,    8192,   None,       2048,   2048,       1,      16,         True,            False,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           True),
            (32,    1,          128,    8192,   None,       2048,   2048,       1,      16,         True,            False,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           True),
            (64,    1,          128,    8192,   None,       2048,   2048,       1,      16,         True,            False,               False,      False,      True,                   False,      False,              False,                      nl.bfloat16,    QuantizationType.NONE,  2,      False,          False,      False,          False,          None,           True),
        ],
    # fmt: on
    )
    def test_attn_blk_megakernel(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        num_heads: int,
        d_head: int,
        H: int,
        H_actual: int,
        S_ctx: int,
        S_max_ctx: int,
        S_tkg: int,
        block_len: int,
        update_cache: bool,
        K_cache_transposed: bool,
        rmsnorm_X: bool,
        skip_rope: bool,
        rope_contiguous_layout: bool,
        rmsnorm_QK: bool,
        qk_norm_post_rope: bool,
        qk_norm_post_rope_gamma: bool,
        dtype,
        quantization_type: QuantizationType,
        lnc: int,
        transposed_out: bool,
        test_bias: bool,
        input_in_sb: bool,
        output_in_sb: bool,
        softmax_scale,
        kv_quant: bool,
        skip_output_projection: bool = False,
        skip_attention: bool = False,
    ):
        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            num_heads=num_heads,
            d_head=d_head,
            H=H,
            H_actual=H_actual,
            S_ctx=S_ctx,
            S_max_ctx=S_max_ctx,
            S_tkg=S_tkg,
            block_len=block_len,
            update_cache=update_cache,
            K_cache_transposed=K_cache_transposed,
            rmsnorm_X=rmsnorm_X,
            skip_rope=skip_rope,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_QK=rmsnorm_QK,
            qk_norm_post_rope=qk_norm_post_rope,
            qk_norm_post_rope_gamma=qk_norm_post_rope_gamma,
            dtype=dtype,
            quantization_type=quantization_type,
            lnc=lnc,
            transposed_out=transposed_out,
            test_bias=test_bias,
            input_in_sb=input_in_sb,
            output_in_sb=output_in_sb,
            softmax_scale=softmax_scale,
            kv_quant=kv_quant,
            skip_output_projection=skip_output_projection,
            skip_attention=skip_attention,
        )

    # Pre-generate test IDs with MODEL_WIP prefix for pytest -k filtering
    ATTENTION_BLOCK_TKG_MODEL_TEST_IDS = [
        f"{MODEL_TEST_TYPE}_" + "-".join(str(p.value) if hasattr(p, 'value') else str(p) for p in params)
        for params in attention_block_tkg_model_configs
    ]

    @pytest.mark.parametrize(
        "batch, num_heads, d_head, H, H_actual, S_ctx, S_max_ctx, S_tkg, block_len, update_cache, K_cache_transposed, rmsnorm_X, skip_rope, rope_contiguous_layout, rmsnorm_QK, qk_norm_post_rope, qk_norm_post_rope_gamma, dtype, quantization_type, lnc, transposed_out, test_bias, input_in_sb, output_in_sb, softmax_scale, kv_quant",
        attention_block_tkg_model_configs,
        ids=ATTENTION_BLOCK_TKG_MODEL_TEST_IDS,
    )
    def test_attn_blk_model(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        request,
        batch: int,
        num_heads: int,
        d_head: int,
        H: int,
        H_actual: int,
        S_ctx: int,
        S_max_ctx: int,
        S_tkg: int,
        block_len: int,
        update_cache: bool,
        K_cache_transposed: bool,
        rmsnorm_X: bool,
        skip_rope: bool,
        rope_contiguous_layout: bool,
        rmsnorm_QK: bool,
        qk_norm_post_rope: bool,
        qk_norm_post_rope_gamma: bool,
        dtype,
        quantization_type: QuantizationType,
        lnc: int,
        transposed_out: bool,
        test_bias: bool,
        input_in_sb: bool,
        output_in_sb: bool,
        softmax_scale,
        kv_quant: bool,
    ):
        """Test Attention Block TKG kernel with model configurations (weekly regression)."""
        attn_blk_metadata_list = load_model_configs("test_attention_block")

        # Apply xfail and add metadata dimensions for model coverage tracking
        request.node.add_marker(pytest.mark.xfail(strict=False, reason="Model coverage test"))
        test_metadata_key = {
            BATCH_CFG: batch,
            NUM_HEADS_CFG: num_heads,
            D_HEAD_CFG: d_head,
            H_CFG: H,
            S_CTX_CFG: S_ctx,
            S_TKG_CFG: S_tkg,
        }
        collector.match_and_add_metadata_dimensions(test_metadata_key, attn_blk_metadata_list)

        _run_attention_block_test(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            num_heads=num_heads,
            d_head=d_head,
            H=H,
            H_actual=H_actual,
            S_ctx=S_ctx,
            S_max_ctx=S_max_ctx,
            S_tkg=S_tkg,
            block_len=block_len,
            update_cache=update_cache,
            K_cache_transposed=K_cache_transposed,
            rmsnorm_X=rmsnorm_X,
            skip_rope=skip_rope,
            rope_contiguous_layout=rope_contiguous_layout,
            rmsnorm_QK=rmsnorm_QK,
            qk_norm_post_rope=qk_norm_post_rope,
            qk_norm_post_rope_gamma=qk_norm_post_rope_gamma,
            dtype=dtype,
            quantization_type=quantization_type,
            lnc=lnc,
            transposed_out=transposed_out,
            test_bias=test_bias,
            input_in_sb=input_in_sb,
            output_in_sb=output_in_sb,
            softmax_scale=softmax_scale,
            kv_quant=kv_quant,
        )
