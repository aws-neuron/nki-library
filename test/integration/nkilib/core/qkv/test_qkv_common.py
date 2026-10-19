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
import functools
from typing import Literal, Optional

import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.qkv.qkv import qkv
from nkilib_src.nkilib.core.qkv.qkv_torch import qkv_torch_ref
from nkilib_src.nkilib.core.utils.common_types import (
    DtypeMode,
    NormType,
    QKNormConfig,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
)
from nkilib_src.nkilib.core.utils.kernel_helpers import get_max_positive_value_for_dtype

from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
    generate_stabilized_mx_data,
    update_func_str,
)
from test.integration.nkilib.utils.test_kernel_common import resolve_dtype_mode_for_torch_ref
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Constructed once at module load and shared across all callers that rely on the default,
# matching the previous behavior where the default argument expression was evaluated once.
_DEFAULT_TENSOR_GEN = gaussian_tensor_generator()

DUMMY_TENSOR_NAME = "dummy"

VNC_DEGREE_DIM_NAME = "vnc_degree"
BATCH_DIM_NAME = "batch"
SEQUENCE_LEN_DIM_NAME = "seqlen"
HIDDEN_DIM_NAME = "hidden_dim"
N_Q_HEADS_DIM_NAME = "n_q_heads"
N_KV_HEADS_DIM_NAME = "n_kv_heads"
D_HEAD_DIM_NAME = "d_head"
NORM_TYPE_DIM_NAME = "norm_type"
FUSED_ADD_DIM_NAME = "fused_add"
OUTPUT_LAYOUT_DIM_NAME = "output_layout"
QUANTIZATION_TYPE_DIM_NAME = "quantization_type"

_q_width = 4
p_max = 128


def build_noncontiguous_slot_mapping(seqlen: int, batch: int, block_size: int, num_blocks: int):
    """Build a shuffled slot_mapping where consecutive logical blocks map to scattered physical blocks.

    Handles partial last block: if seqlen is not a multiple of block_size, the remaining
    tokens are assigned to sequential slots within an additional physical block.

    Note: Caller is responsible for setting np.random.seed before calling this function.
    """
    num_full_blocks = seqlen // block_size
    remainder = seqlen % block_size
    num_seq_blocks = num_full_blocks + (1 if remainder > 0 else 0)
    # Create a shuffled physical block index array
    physical_blocks = np.random.choice(num_blocks, size=num_seq_blocks, replace=False)
    # Build slot_mapping: for logical block i, physical slots are
    # physical_blocks[i] * block_size + 0, 1, ..., block_size-1
    slot_mapping = np.zeros(seqlen, dtype=np.int32)
    for i in range(num_full_blocks):
        start = i * block_size
        slot_mapping[start : start + block_size] = physical_blocks[i] * block_size + np.arange(
            block_size, dtype=np.int32
        )
    if remainder > 0:
        start = num_full_blocks * block_size
        slot_mapping[start : start + remainder] = physical_blocks[num_full_blocks] * block_size + np.arange(
            remainder, dtype=np.int32
        )
    return slot_mapping.reshape(batch, seqlen)


def build_qkv_input(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    fused_qkv_dim: int,
    dtype,
    d_head: Optional[int] = None,
    eps: Optional[float] = None,
    norm_type: NormType = NormType.RMS_NORM,
    fused_add: bool = True,
    lnc_degree: int = 1,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    quantization_type: QuantizationType = QuantizationType.NONE,
    use_dma_transpose: bool = True,
    qkv_bias: Optional[bool] = False,
    norm_bias: Optional[bool] = False,
    hidden_actual: Optional[int] = None,
    fused_rope: Optional[bool] = False,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    tensor_gen=_DEFAULT_TENSOR_GEN,
    fp8_kv_cache: bool = False,
    bf16_kv_cache: bool = False,
    transpose_k_cache: bool = False,
    max_seq_len: Optional[int] = None,
    k_scale_val: Optional[float] = None,
    v_scale_val: Optional[float] = None,
    fp8_max: float = 240.0,
    fp8_min: float = -240.0,
    use_block_kv: bool = False,
    fp8_packed: bool = False,
    num_blocks: Optional[int] = None,
    block_size: Optional[int] = None,
    slot_mapping: Optional[np.ndarray] = None,
    is_h_dim_4h_transposed: bool = False,
    qk_norm_pre_rope_config: Optional[dict] = None,
    qk_norm_post_rope_config: Optional[dict] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
):
    qkv_w_scale = None
    qkv_in_scale = None
    if quantization_type in (QuantizationType.MX, QuantizationType.STATIC_MX, QuantizationType.ROW_MX):
        np.random.seed(0)
        dequant_weights, fused_qkv_weights_x4, qkv_w_scale = generate_stabilized_mx_data(
            nl.float8_e4m3fn_x4, (hidden_dim // _q_width, fused_qkv_dim * _q_width), val_range=5
        )
        # Convert x4-packed [H//4, I] to unpacked fp8 [H//4, I, 4]
        fused_qkv_weights = fused_qkv_weights_x4.view(nl.float8_e4m3fn).reshape(
            hidden_dim // _q_width, fused_qkv_dim, _q_width
        )

        if quantization_type == QuantizationType.STATIC_MX:
            # STATIC_MX: per-Q/K/V weight dequant scale, single input scale
            base_scale = 1.0 / 448.0
            np.random.seed(42)
            qkv_w_scale = base_scale * np.random.uniform(0.995, 1.005, (1, 3)).astype(np.float32)
            qkv_in_scale = base_scale * np.random.uniform(0.995, 1.005, (1, 1)).astype(np.float32)
        elif quantization_type == QuantizationType.ROW_MX:
            # ROW_MX: per-column weight dequant scale [1, I], no input scale
            np.random.seed(42)
            base_scale = 1.0 / 448.0
            qkv_w_scale = base_scale * np.random.uniform(0.995, 1.005, (1, fused_qkv_dim)).astype(np.float32)
        else:
            qkv_w_scale = qkv_w_scale.reshape(-1, fused_qkv_dim)
        input_tensor = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="input")
        if is_h_dim_4h_transposed:
            input_tensor = (
                input_tensor.reshape(batch, seqlen, hidden_dim // (p_max * _q_width), p_max, _q_width)
                .transpose(0, 1, 4, 2, 3)
                .reshape(batch, seqlen, hidden_dim)
            )
    else:
        input_tensor = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="input")
        # FP8 weight dtype: nl.float8_e4m3fn when DtypeMode.OCP,
        # nl.float8_e4m3 otherwise.
        _fp8_weight_dtype = nl.float8_e4m3fn if dtype_mode == DtypeMode.OCP else nl.float8_e4m3
        weight_dtype = dtype if quantization_type == QuantizationType.NONE else _fp8_weight_dtype
        fused_qkv_weights = tensor_gen(shape=(hidden_dim, fused_qkv_dim), dtype=weight_dtype, name="fused_qkv_weights")
        if quantization_type == QuantizationType.STATIC:
            # Clip input and weights to [-1, 1] for proper quantization.
            # Use (1, N) shape: uniform across partitions, matching torch ref.
            input_tensor = np.clip(input_tensor, -1.0, 1.0).astype(input_tensor.dtype)
            fused_qkv_weights = np.clip(fused_qkv_weights, -1.0, 1.0).astype(fused_qkv_weights.dtype)
            # Set scales to 1/240 with small deviation
            base_scale = 1.0 / 240.0
            # Add small per-element deviation (±0.5%), then broadcast to partition dim.
            np.random.seed(42)  # For reproducibility
            qkv_w_scale = np.broadcast_to(
                base_scale * np.random.uniform(0.995, 1.005, (1, 3)).astype(np.float32), (128, 3)
            ).copy()
            qkv_in_scale = np.broadcast_to(
                base_scale * np.random.uniform(0.995, 1.005, (1, 1)).astype(np.float32), (128, 1)
            ).copy()
        elif quantization_type == QuantizationType.ROW:
            # For row quantization: per-output-channel weight scales, no input quantization
            FP8_E4M3_MAX = get_max_positive_value_for_dtype(_fp8_weight_dtype)
            # Compute per-column (per-output-channel) scale from weight magnitudes
            w_max = np.abs(fused_qkv_weights).max(axis=0, keepdims=True)
            w_scale = (w_max / FP8_E4M3_MAX).astype(np.float32)
            # Quantize weights to FP8
            fused_qkv_weights = (fused_qkv_weights / w_scale).astype(_fp8_weight_dtype)
            # Broadcast scale to (128, fused_qkv_dim) shape
            qkv_w_scale = np.broadcast_to(w_scale, (128, fused_qkv_dim)).astype(np.float32)
            qkv_in_scale = None

    mlp_prev = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="mlp_prev") if fused_add else None
    attention_prev = (
        tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="attention_prev") if fused_add else None
    )
    # If is_h_dim_4h_transposed, we must shuffle gamma tensor as well: otherwise non-shuffled rms_norm is applied to shuffled input.
    # Note that in real model:
    # -> Shuffled H in input comes from offline pre-shuffle of upstream weights, e.g. down-projection.
    # -> Shuffled H in gamma comes from direct offline pre-shuffle of gamma weights.
    if quantization_type in (QuantizationType.MX, QuantizationType.STATIC_MX, QuantizationType.ROW_MX):
        gamma_norm_weights = (
            tensor_gen(shape=(1, hidden_dim), dtype=dtype, name="gamma_norm_weights")
            if norm_type in [NormType.RMS_NORM, NormType.LAYER_NORM]
            else None
        )

        if is_h_dim_4h_transposed and gamma_norm_weights is not None:
            gamma_norm_weights = (
                gamma_norm_weights.reshape(1, hidden_dim // (p_max * _q_width), p_max, _q_width)
                .transpose(0, 3, 1, 2)
                .reshape(1, hidden_dim)
            )
    else:
        gamma_norm_weights = (
            tensor_gen(shape=(1, hidden_dim), dtype=dtype, name="gamma_norm_weights")
            if norm_type in [NormType.RMS_NORM, NormType.LAYER_NORM]
            else None
        )
    bias = tensor_gen(shape=(1, fused_qkv_dim), dtype=dtype, name="bias") if qkv_bias else None
    layer_norm_bias = tensor_gen(shape=(1, hidden_dim), dtype=dtype, name="layer_norm_bias") if norm_bias else None

    cos_cache_t = tensor_gen(shape=(batch, seqlen, d_head), dtype=dtype, name="cos_cache") if fused_rope else None
    sin_cache_t = tensor_gen(shape=(batch, seqlen, d_head), dtype=dtype, name="sin_cache") if fused_rope else None

    k_cache = None
    v_cache = None
    k_scale = None
    v_scale = None
    kv_dtype = None
    kv_dim = num_kv_heads * d_head if num_kv_heads and d_head else None
    if fp8_kv_cache or bf16_kv_cache:
        assert num_kv_heads is not None and d_head is not None and kv_dim is not None, (
            "KV cache requires num_kv_heads and d_head"
        )
        # MX weights are OCP float8_e4m3fn; the compiler rejects mixing OCP and
        # legacy (float8_e4m3) fp8 in one kernel (NCC_EOCP001), so the KV cache
        # must use the OCP dtype whenever the projection is MX.
        _fp8_kv_dtype = nl.float8_e4m3fn if quantization_type.is_mx() else nl.float8_e4m3
        cache_np_dtype = nl.bfloat16 if bf16_kv_cache else _fp8_kv_dtype
        kv_dtype = nl.bfloat16 if bf16_kv_cache else _fp8_kv_dtype
        if not bf16_kv_cache:
            k_scale = np.full((128, 1), k_scale_val, dtype=np.float32)
            v_scale = np.full((128, 1), v_scale_val, dtype=np.float32)
        if use_block_kv:
            assert num_blocks is not None and block_size is not None, (
                "block KV cache requires num_blocks and block_size"
            )
            if fp8_packed:
                k_cache = np.zeros((num_blocks, num_kv_heads, block_size // 2, d_head, 2), dtype=cache_np_dtype)
            elif transpose_k_cache:
                k_cache = np.zeros((num_blocks * num_kv_heads, d_head, block_size), dtype=cache_np_dtype)
            else:
                k_cache = np.zeros((num_blocks, block_size, kv_dim), dtype=cache_np_dtype)
            if fp8_packed:
                # Head-split V cache: [num_blocks, num_kv_heads, block_size, d_head].
                v_cache = np.zeros((num_blocks, num_kv_heads, block_size, d_head), dtype=cache_np_dtype)
            else:
                v_cache = np.zeros((num_blocks, block_size, kv_dim), dtype=cache_np_dtype)
        else:
            assert max_seq_len is not None, "contiguous KV cache requires max_seq_len"
            if transpose_k_cache:
                k_cache = np.zeros((batch, kv_dim, max_seq_len), dtype=cache_np_dtype)
            else:
                k_cache = np.zeros((batch, max_seq_len, kv_dim), dtype=cache_np_dtype)
            v_cache = np.zeros((batch, max_seq_len, kv_dim), dtype=cache_np_dtype)

    result = {
        "input": input_tensor,
        "fused_qkv_weights": fused_qkv_weights,
        "output_layout": output_layout,
        "bias": bias,
        "quantization_type": quantization_type,
        "qkv_w_scale": qkv_w_scale,
        "qkv_in_scale": qkv_in_scale,
        "fused_residual_add": fused_add,
        "mlp_prev": mlp_prev,
        "attention_prev": attention_prev,
        "fused_norm_type": norm_type,
        "gamma_norm_weights": gamma_norm_weights,
        "layer_norm_bias": layer_norm_bias,
        "norm_eps": eps,
        "hidden_actual": hidden_actual,
        "fused_rope": fused_rope,
        "cos_cache": cos_cache_t,
        "sin_cache": sin_cache_t,
        "d_head": d_head,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "store_output_in_sbuf": False,
        "sbm": None,
        "use_auto_allocation": False,
        "load_input_with_DMA_transpose": use_dma_transpose,
        "weight_layout": QKVWeightLayout.MX_CONTIGUOUS
        if quantization_type in (QuantizationType.MX, QuantizationType.STATIC_MX, QuantizationType.ROW_MX)
        else QKVWeightLayout.CONTIGUOUS,
    }

    if fp8_kv_cache or bf16_kv_cache:
        result["k_cache.must_alias_input"] = k_cache
        result["v_cache.must_alias_input"] = v_cache
        result["kv_dtype"] = kv_dtype
        result["transpose_k_cache"] = transpose_k_cache
        result["fp8_packed"] = fp8_packed
        if not bf16_kv_cache:
            result["k_scale"] = k_scale
            result["v_scale"] = v_scale
            result["fp8_max"] = fp8_max
            result["fp8_min"] = fp8_min
        if use_block_kv:
            result["use_block_kv"] = True
            result["block_size"] = block_size
            result["slot_mapping"] = slot_mapping

    if qk_norm_pre_rope_config is not None:
        result["qk_norm_pre_rope"] = QKNormConfig(
            q_norm=qk_norm_pre_rope_config.get("q_norm", NormType.RMS_NORM),
            k_norm=qk_norm_pre_rope_config.get("k_norm", NormType.RMS_NORM),
            eps=qk_norm_pre_rope_config["eps"],
            gamma_fused_in_rope_caches=qk_norm_pre_rope_config.get("gamma_fused_in_rope_caches", False),
        )
        result["qk_norm_pre_rope_q_gamma"] = qk_norm_pre_rope_config.get("q_gamma")
        result["qk_norm_pre_rope_k_gamma"] = qk_norm_pre_rope_config.get("k_gamma")
    if qk_norm_post_rope_config is not None:
        result["qk_norm_post_rope"] = QKNormConfig(
            q_norm=qk_norm_post_rope_config.get("q_norm", NormType.RMS_NORM),
            k_norm=qk_norm_post_rope_config.get("k_norm", NormType.RMS_NORM),
            eps=qk_norm_post_rope_config["eps"],
        )
        result["qk_norm_post_rope_q_gamma"] = qk_norm_post_rope_config.get("q_gamma")
        result["qk_norm_post_rope_k_gamma"] = qk_norm_post_rope_config.get("k_gamma")

    return result


def _swizzle_mla_cols(arr, dim):
    # Reorder columns from [I//512, 4, 128] to [4, I//512, 128] so the matmul
    # output has layout [S, 4_I, I//512, 128_I] enabling contiguous nc_transpose
    # reads (pre-swizzled transpose optimization).
    num_512_tiles = dim // (p_max * _q_width)
    idx = np.arange(dim).reshape(num_512_tiles, p_max, _q_width).transpose(2, 0, 1).reshape(dim)
    return arr[..., idx]


_DS_SCALE_BLOCK = 128


def _reduce_mx_scale_to_compact_block128(mx_scale, K, N):
    """Reduce an MX-format ``[K//32, N]`` scale to DeepSeek compact ``[K//128, ceil(N/128)]``.

    Picks the top-left value of each 4-row x 128-col block as the canonical
    block-128 scale. Callers should subsequently re-broadcast this to the MX
    format if they want a "uniform within 128x128 block" full-MX scale.
    """
    K_blocks = K // _DS_SCALE_BLOCK
    N_blocks = (N + _DS_SCALE_BLOCK - 1) // _DS_SCALE_BLOCK
    compact = np.empty((K_blocks, N_blocks), dtype=np.uint8)
    for ki in range(K_blocks):
        for ni in range(N_blocks):
            n_start = ni * _DS_SCALE_BLOCK
            compact[ki, ni] = mx_scale[ki * 4, n_start]
    return compact


def build_qkv_mla_input(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    qk_lora_rank: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    n_heads: int,
    *,
    variant: Literal["v32", "v4"] = "v32",
    qk_nope_head_dim: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    head_dim: Optional[int] = None,
):
    """Build kernel inputs for an MLA QKV CTE kernel.

    Two variants are supported:
        - ``v32``: matches :func:`qkv_mla_mx`. Requires ``qk_nope_head_dim`` and
          ``v_head_dim``. Returns ``wqkv_a_hbm``/``wkv_b_hbm`` for the two-stage
          KV path.
        - ``v4``: matches :func:`qkv_mla_v4_mx`. Requires ``head_dim``. Returns
          a single fused ``wqkv_hbm`` (no wkv_b second matmul).
    """
    if variant == "v32":
        if qk_nope_head_dim is None or v_head_dim is None:
            raise ValueError("variant='v32' requires qk_nope_head_dim and v_head_dim")
        q_out_dim = n_heads * (qk_nope_head_dim + qk_rope_head_dim)
        kv_b_out_dim = n_heads * (qk_nope_head_dim + v_head_dim)
    elif variant == "v4":
        if head_dim is None:
            raise ValueError("variant='v4' requires head_dim")
        q_out_dim = n_heads * head_dim
        kv_b_out_dim = 0  # v4 has no second KV matmul
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    np.random.seed(42)
    tensor_gen = rope_gaussian_tensor_generator()

    # ---- Common: input tensor x ----
    x = tensor_gen((batch, seqlen, hidden_dim), nl.bfloat16, "input")
    x = (
        x.reshape(batch * seqlen, hidden_dim // (p_max * _q_width), p_max, _q_width)
        .transpose(0, 3, 1, 2)
        .reshape(batch, seqlen, hidden_dim)
    )

    # ---- Common: wq_a (first Q projection) and wq_b (second Q projection) ----
    _, wq_a, wq_a_scale = generate_stabilized_mx_data(
        nl.float8_e4m3fn_x4, (hidden_dim // 4, qk_lora_rank * 4), val_range=5
    )
    _, wq_b, wq_b_scale = generate_stabilized_mx_data(
        nl.float8_e4m3fn_x4, (qk_lora_rank // 4, q_out_dim * 4), val_range=5
    )

    # ---- Common: norm gammas and RoPE caches ----
    q_norm_gamma = tensor_gen((1, qk_lora_rank), nl.bfloat16, "q_norm")
    cos_cache = tensor_gen((batch, seqlen, qk_rope_head_dim), nl.bfloat16, "cos_cache")
    sin_cache = tensor_gen((batch, seqlen, qk_rope_head_dim), nl.bfloat16, "sin_cache")

    if variant == "v32":
        kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
        fused_qkv_dim = qk_lora_rank + kv_a_out_dim

        _, wkv_a, wkv_a_scale = generate_stabilized_mx_data(
            nl.float8_e4m3fn_x4, (hidden_dim // 4, kv_a_out_dim * 4), val_range=5
        )
        _, wkv_b, wkv_b_scale = generate_stabilized_mx_data(
            nl.float8_e4m3fn_x4, (kv_lora_rank // 4, kv_b_out_dim * 4), val_range=5
        )

        wqkv_a = np.concatenate([wq_a, wkv_a], axis=-1)
        wqkv_a_scale = np.concatenate([wq_a_scale, wkv_a_scale], axis=-1)

        # Swizzle qr and kv portions, leave k_pe unswizzled (k_pe goes to RoPE, not matmul).
        wqkv_a_qr = wqkv_a[..., :qk_lora_rank]
        wqkv_a_kv = wqkv_a[..., qk_lora_rank : qk_lora_rank + kv_lora_rank]
        wqkv_a_kpe = wqkv_a[..., qk_lora_rank + kv_lora_rank :]
        wqkv_a = np.concatenate(
            [_swizzle_mla_cols(wqkv_a_qr, qk_lora_rank), _swizzle_mla_cols(wqkv_a_kv, kv_lora_rank), wqkv_a_kpe],
            axis=-1,
        )
        wqkv_a_scale_qr = wqkv_a_scale[..., :qk_lora_rank]
        wqkv_a_scale_kv = wqkv_a_scale[..., qk_lora_rank : qk_lora_rank + kv_lora_rank]
        wqkv_a_scale_kpe = wqkv_a_scale[..., qk_lora_rank + kv_lora_rank :]
        wqkv_a_scale = np.concatenate(
            [
                _swizzle_mla_cols(wqkv_a_scale_qr, qk_lora_rank),
                _swizzle_mla_cols(wqkv_a_scale_kv, kv_lora_rank),
                wqkv_a_scale_kpe,
            ],
            axis=-1,
        ).reshape(-1, fused_qkv_dim)

        kv_norm_gamma = tensor_gen((1, kv_lora_rank), nl.bfloat16, "kv_norm")

        # Reduce to DeepSeek compact block-128 scales. The kernel consumes the
        # compact form directly; the torch reference broadcasts it back to MX
        # layout before its mx_matmul.
        wqkv_a_scale_compact = _reduce_mx_scale_to_compact_block128(wqkv_a_scale, hidden_dim, fused_qkv_dim)
        wq_b_scale_compact = _reduce_mx_scale_to_compact_block128(
            wq_b_scale.reshape(-1, q_out_dim), qk_lora_rank, q_out_dim
        )
        wkv_b_scale_compact = _reduce_mx_scale_to_compact_block128(
            wkv_b_scale.reshape(-1, kv_b_out_dim), kv_lora_rank, kv_b_out_dim
        )

        return {
            "x_hbm": x,
            "wqkv_a_hbm": wqkv_a,
            "wqkv_a_scale_hbm": wqkv_a_scale_compact,
            "wq_b_hbm": wq_b,
            "wq_b_scale_hbm": wq_b_scale_compact,
            "q_norm_gamma_hbm": q_norm_gamma,
            "wkv_b_hbm": wkv_b,
            "wkv_b_scale_hbm": wkv_b_scale_compact,
            "kv_norm_gamma_hbm": kv_norm_gamma,
            "cos_cache_hbm": cos_cache,
            "sin_cache_hbm": sin_cache,
            "n_heads": n_heads,
            "qk_nope_head_dim": qk_nope_head_dim,
            "qk_rope_head_dim": qk_rope_head_dim,
            "v_head_dim": v_head_dim,
            "kv_lora_rank": kv_lora_rank,
            "qk_lora_rank": qk_lora_rank,
            "norm_eps": 1e-6,
        }

    # variant == "v4"
    kv_dim = kv_lora_rank + qk_rope_head_dim
    fused_out_dim = qk_lora_rank + kv_dim

    _, wkv, wkv_scale = generate_stabilized_mx_data(nl.float8_e4m3fn_x4, (hidden_dim // 4, kv_dim * 4), val_range=5)

    wq_a = _swizzle_mla_cols(wq_a, qk_lora_rank)
    wq_a_scale = _swizzle_mla_cols(wq_a_scale, qk_lora_rank)

    # Fuse wq_a and wkv into single weight matrix (qr cols pre-swizzled, kv cols as-is).
    wqkv = np.concatenate([wq_a, wkv], axis=-1)
    wqkv_scale = np.concatenate([wq_a_scale, wkv_scale], axis=-1).reshape(-1, fused_out_dim)

    kv_norm_gamma = tensor_gen((1, kv_dim), nl.bfloat16, "kv_norm")

    # Reduce to DeepSeek compact block-128 scales (kernel consumes compact;
    # torch ref broadcasts internally).
    wqkv_scale_compact = _reduce_mx_scale_to_compact_block128(wqkv_scale, hidden_dim, fused_out_dim)
    wq_b_scale_compact = _reduce_mx_scale_to_compact_block128(
        wq_b_scale.reshape(-1, q_out_dim), qk_lora_rank, q_out_dim
    )

    return {
        "x_hbm": x,
        "wqkv_hbm": wqkv,
        "wqkv_scale_hbm": wqkv_scale_compact,
        "wq_b_hbm": wq_b,
        "wq_b_scale_hbm": wq_b_scale_compact,
        "q_norm_gamma_hbm": q_norm_gamma,
        "kv_norm_gamma_hbm": kv_norm_gamma,
        "cos_cache_hbm": cos_cache,
        "sin_cache_hbm": sin_cache,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "qk_rope_head_dim": qk_rope_head_dim,
        "kv_lora_rank": kv_lora_rank,
        "qk_lora_rank": qk_lora_rank,
        "norm_eps": 1e-6,
    }


def build_qkv_mla_v4_input(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    qk_lora_rank: int,
    head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    n_heads: int,
):
    return build_qkv_mla_input(
        batch=batch,
        seqlen=seqlen,
        hidden_dim=hidden_dim,
        qk_lora_rank=qk_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        n_heads=n_heads,
        variant="v4",
        head_dim=head_dim,
    )


def rope_gaussian_tensor_generator(mean=0.0, std=1.0):
    """Create a Gaussian tensor generator with special handling for RoPE cache tensors.

    Args:
        mean (float, optional): The mean (center) of the Gaussian distribution.
                               Defaults to 0.0.
        std (float, optional): The standard deviation (spread) of the Gaussian
                              distribution. Defaults to 1.0.

    Returns:
        callable: A tensor generator function with signature (shape, dtype, name) -> np.ndarray
                 that generates Gaussian-distributed tensors with special RoPE cache handling.

    Behavior:
        - For cos_cache and sin_cache tensors:
          * Generates tensor with last dimension halved
          * Tiles the tensor to duplicate values (required for RoPE implementation)
          * Final shape matches requested shape
        - For all other tensors:
          * Standard Gaussian distribution with specified mean/std

    Example:
        >>> generator = rope_gaussian_tensor_generator(mean=0.0, std=1.0)
        >>> cos_cache = generator(shape=(128, 64), dtype=np.float32, name="cos_cache")
        >>> # Generates (128, 32) tensor, then tiles to (128, 64)
    """
    rng = np.random.default_rng(0)

    @update_func_str(mean=mean, std=std)
    def tensor_generator(shape, dtype, name):
        guessed_dtype = dtype
        if name in ["cos_cache", "sin_cache"]:
            tensor_template_shape = list(shape)
            tensor_template_shape[-1] = tensor_template_shape[-1] // 2
            tensor = (rng.normal(size=tensor_template_shape) * std + mean).astype(guessed_dtype)
            tensor = np.tile(tensor, 2)
            return tensor
        else:
            return (rng.normal(size=shape) * std + mean).astype(guessed_dtype)

    return tensor_generator


def generate_rope_frequencies(seqlen, d_head, theta=10000.0):
    """Generate standard RoPE cos/sin caches with real frequencies."""
    d_half = d_head // 2
    freqs = 1.0 / (theta ** (np.arange(0, d_half, dtype=np.float64) / d_half))
    positions = np.arange(seqlen, dtype=np.float64)
    angles = np.outer(positions, freqs)
    cos_half = np.cos(angles).astype(np.float32)
    sin_half = np.sin(angles).astype(np.float32)
    return np.tile(cos_half, (1, 2)), np.tile(sin_half, (1, 2))


def real_rope_tensor_generator(seed=42, theta=10000.0):
    """Tensor generator that uses real RoPE frequencies for cos/sin caches."""
    base_gen = gaussian_tensor_generator(seed=seed)

    def tensor_generator(shape, dtype, name):
        if name in ("cos_cache", "sin_cache"):
            batch, seqlen, d_head = shape
            cos_2d, sin_2d = generate_rope_frequencies(seqlen, d_head, theta)
            cache = cos_2d if name == "cos_cache" else sin_2d
            return np.broadcast_to(cache[np.newaxis], shape).copy().astype(dtype)
        return base_gen(shape, dtype, name)

    return tensor_generator


def run_qkv_test(
    test_manager,
    compiler_args,
    B: int,
    H: int,
    S: int,
    fused_qkv_dim: int,
    lnc_degree: int,
    dtype,
    eps: float,
    norm_type: NormType,
    fused_add: bool = True,
    norm_bias: bool = False,
    qkv_bias: bool = False,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    hidden_actual: int | None = None,
    n_q_heads: int | None = None,
    n_kv_heads: int | None = None,
    d_head: int | None = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    is_h_dim_4h_transposed: bool = False,
    transposed_in: bool = False,
    # --- CTE-specific params (ignored by TKG path) ---
    use_dma_transpose: bool = True,
    fused_rope: bool = False,
    tensor_gen=None,
    fp8_kv_cache: bool = False,
    bf16_kv_cache: bool = False,
    transpose_k_cache: bool = False,
    max_seq_len: int | None = None,
    k_scale_val: float | None = None,
    v_scale_val: float | None = None,
    fp8_max: float = 240.0,
    fp8_min: float = -240.0,
    use_block_kv: bool = False,
    fp8_packed: bool = False,
    num_blocks: int | None = None,
    block_size: int | None = None,
    slot_mapping: np.ndarray | None = None,
    qkv_in_scale_for_mx: np.ndarray | None = None,
    qkv_w_scale_for_mx: np.ndarray | None = None,
    preserve_lower_precision: bool = False,
    # --- QK-Norm params ---
    qk_norm_pre_rope_config: dict | None = None,
    qk_norm_post_rope_config: dict | None = None,
    # --- Test execution params ---
    rtol: float = 2e-2,
    atol: float = 1e-5,
    is_negative_test: bool = False,
    inference_args=None,
    expect_fused_hidden_output: bool = False,
    # --- FP8 E4M3 dtype mode. See DtypeMode enum.
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
):
    """Shared test helper for QKV CTE and TKG tests using UnitTestFramework.

    Both CTE and TKG tests use the same kernel entry (qkv) and torch ref
    (qkv_torch_ref), so the boilerplate for input generation, output tensor
    descriptors, and framework setup is identical. This function consolidates
    that shared logic.

    The expect_fused_hidden_output flag controls whether fused_hidden is
    included in output tensors when fused_add=True (TKG returns it; CTE
    does not).
    """
    if tensor_gen is None:
        tensor_gen = gaussian_tensor_generator()

    # Pre-resolve DtypeMode.AUTO once and use it for:
    #   (1) the weight allocator in build_qkv_input → ensures the HBM weight
    #       tensor has a concrete FP8 dtype matching the platform.
    #   (2) the torch ref → torch ref runs on CPU and can't query hardware.
    # The kernel itself receives the original dtype_mode (possibly AUTO).
    # It uses weight.dtype (always concrete) for STATIC/ROW allocations,
    # so AUTO becomes a no-op there.
    torch_ref_dtype_mode = resolve_dtype_mode_for_torch_ref(dtype_mode, compiler_args.platform_target)

    def input_generator(test_config):
        kernel_input = build_qkv_input(
            batch=B,
            seqlen=S,
            hidden_dim=H,
            fused_qkv_dim=fused_qkv_dim,
            dtype=dtype,
            eps=eps,
            d_head=d_head,
            norm_type=norm_type,
            fused_add=fused_add,
            output_layout=output_layout,
            lnc_degree=lnc_degree,
            use_dma_transpose=use_dma_transpose,
            qkv_bias=qkv_bias,
            norm_bias=norm_bias,
            hidden_actual=hidden_actual,
            fused_rope=fused_rope,
            num_q_heads=n_q_heads,
            num_kv_heads=n_kv_heads,
            quantization_type=quantization_type,
            tensor_gen=tensor_gen,
            fp8_kv_cache=fp8_kv_cache,
            bf16_kv_cache=bf16_kv_cache,
            max_seq_len=max_seq_len,
            k_scale_val=k_scale_val,
            v_scale_val=v_scale_val,
            fp8_max=fp8_max,
            fp8_min=fp8_min,
            use_block_kv=use_block_kv,
            fp8_packed=fp8_packed,
            num_blocks=num_blocks,
            block_size=block_size,
            transpose_k_cache=transpose_k_cache,
            slot_mapping=slot_mapping,
            is_h_dim_4h_transposed=is_h_dim_4h_transposed,
            qk_norm_pre_rope_config=qk_norm_pre_rope_config,
            qk_norm_post_rope_config=qk_norm_post_rope_config,
            dtype_mode=torch_ref_dtype_mode,
        )
        kernel_input["is_h_dim_4h_transposed"] = is_h_dim_4h_transposed
        # Only pass transposed_in when True to avoid signature mismatch with torch refs
        if transposed_in:
            kernel_input["transposed_in"] = True
            # Convert input to transposed layout
            X = kernel_input["input"]
            BxS = B * S
            H0 = 128
            H1_shard = H // lnc_degree // H0
            X_transposed = X.reshape(BxS, lnc_degree, H0, H1_shard).transpose(2, 1, 3, 0)
            kernel_input["input"] = np.ascontiguousarray(X_transposed)
        # Pass MX static dequant scales to kernel if provided
        if qkv_in_scale_for_mx is not None:
            kernel_input["qkv_in_scale"] = qkv_in_scale_for_mx
        if qkv_w_scale_for_mx is not None:
            kernel_input["qkv_w_scale"] = qkv_w_scale_for_mx
        # Pass the original (possibly AUTO) dtype_mode to the kernel. The
        # kernel uses the caller-allocated weight dtype as the source of
        # truth and only resolves dtype_mode if the caller passed an opaque
        # "float8e4" sentinel.
        kernel_input["dtype_mode"] = dtype_mode
        return kernel_input

    def output_tensor_descriptor(kernel_input):
        if fp8_kv_cache or bf16_kv_cache:
            assert n_q_heads is not None and n_kv_heads is not None and d_head is not None, (
                "KV cache requires n_q_heads, n_kv_heads and d_head"
            )
            q_dim = n_q_heads * d_head
            kv_dim = n_kv_heads * d_head
            cache_dtype = nl.bfloat16 if bf16_kv_cache else nl.float8_e4m3
            if use_block_kv:
                assert num_blocks is not None and block_size is not None, (
                    "block KV cache requires num_blocks and block_size"
                )
                if fp8_packed:
                    k_cache = np.zeros((num_blocks, n_kv_heads, block_size // 2, d_head, 2), dtype=cache_dtype)
                elif not transpose_k_cache:
                    k_cache = np.zeros((num_blocks, block_size, kv_dim), dtype=cache_dtype)
                else:
                    k_cache = np.zeros((num_blocks * n_kv_heads, d_head, block_size), dtype=cache_dtype)
                if fp8_packed:
                    v_cache = np.zeros((num_blocks, n_kv_heads, block_size, d_head), dtype=cache_dtype)
                else:
                    v_cache = np.zeros((num_blocks, block_size, kv_dim), dtype=cache_dtype)
                return {
                    "q_tensor_hbm": np.zeros((B, S, q_dim), dtype=dtype),
                    "k_cache": k_cache,
                    "v_cache": v_cache,
                }
            else:
                assert max_seq_len is not None, "contiguous KV cache requires max_seq_len"
                if not transpose_k_cache:
                    k_cache = np.zeros((B, max_seq_len, kv_dim), dtype=cache_dtype)
                else:
                    k_cache = np.zeros((B, kv_dim, max_seq_len), dtype=cache_dtype)
                return {
                    "q_tensor_hbm": np.zeros((B, S, q_dim), dtype=dtype),
                    "k_cache": k_cache,
                    "v_cache": np.zeros((B, max_seq_len, kv_dim), dtype=cache_dtype),
                }
        output_dtype = nl.bfloat16 if dtype in (nl.float8_e4m3, nl.float8_e4m3fn) else dtype
        result = {
            "out": np.zeros((B * S, fused_qkv_dim) if transposed_in else (B, S, fused_qkv_dim), dtype=output_dtype)
        }
        if fused_add and expect_fused_hidden_output:
            result["fused_hidden"] = np.zeros((B, S, H), dtype=output_dtype)
        return result

    _is_fp8_dtype = dtype in (nl.float8_e4m3, nl.float8_e4m3fn)

    @functools.wraps(qkv_torch_ref)
    def _qkv_torch_ref_with_resolved_dtype_mode(**kwargs):
        kwargs["dtype_mode"] = torch_ref_dtype_mode
        if _is_fp8_dtype and quantization_type == QuantizationType.STATIC:
            # FP8 input is already quantized. The torch ref will divide by in_scale,
            # so pre-multiply input to compensate: hidden * in_scale / in_scale = hidden.
            in_scale = kwargs["qkv_in_scale"]
            if in_scale is not None:
                scale_scalar = float(in_scale.flatten()[0])
                kwargs["input"] = kwargs["input"] * scale_scalar
        return qkv_torch_ref(**kwargs)

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=qkv,
        torch_ref=torch_ref_wrapper(
            _qkv_torch_ref_with_resolved_dtype_mode, preserve_lower_precision=preserve_lower_precision
        ),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=output_tensor_descriptor,
        check_unused_params=True,
    )
    framework.run_test(
        test_config=None,
        compiler_args=compiler_args,
        rtol=rtol,
        atol=atol,
        is_negative_test=is_negative_test,
        inference_args=inference_args,
    )


def fuse_gamma_into_rope_caches(cos_cache, sin_cache, q_gamma, k_gamma, d_head):
    """Pre-multiply gamma weights into RoPE cos/sin caches for gamma fusion.

    Returns (q_cos, q_sin, k_cos, k_sin) with gamma baked in.
    """
    d_half = d_head // 2
    sin_half = sin_cache[:, :, :d_half]
    q_cos = cos_cache * q_gamma
    q_sin = np.concatenate([sin_half * q_gamma[:, :d_half], sin_half * q_gamma[:, d_half:]], axis=-1)
    k_cos = cos_cache * k_gamma
    k_sin = np.concatenate([sin_half * k_gamma[:, :d_half], sin_half * k_gamma[:, d_half:]], axis=-1)
    return q_cos, q_sin, k_cos, k_sin


def gamma_unfused_ref(wrapped_torch_ref, original_cos, original_sin, q_gamma, k_gamma, eps):
    """Wrap a torch ref to substitute unfused (original) inputs for golden computation.

    The kernel receives gamma-fused caches with no explicit gamma weights.
    This wrapper intercepts those inputs and replaces them with the original
    unfused caches + explicit gamma, so the torch ref computes the unfused
    path as the golden reference.
    """

    @functools.wraps(wrapped_torch_ref)
    def ref(**kwargs):
        kwargs["cos_cache"] = original_cos
        kwargs["sin_cache"] = original_sin
        kwargs.pop("k_cos_cache", None)
        kwargs.pop("k_sin_cache", None)
        kwargs["qk_norm_pre_rope"] = QKNormConfig(eps=eps)
        kwargs["qk_norm_pre_rope_q_gamma"] = q_gamma
        kwargs["qk_norm_pre_rope_k_gamma"] = k_gamma
        return wrapped_torch_ref(**kwargs)

    return ref


def row_mx_unpack_ref(wrapped_torch_ref, unpacked_fp8_data, row_scales):
    """Wrap a torch ref to unpack ROW_MX tail-scale input for golden computation.

    The kernel receives [B, S, H+4] FP8 with tail-packed float32 scale and
    qkv_in_scale=None. This wrapper intercepts the packed input and replaces
    it with the unpacked [B, S, H] FP8 data + separate [B, S, 1] row scale
    as qkv_in_scale, so the torch ref can compute the golden reference.
    """

    @functools.wraps(wrapped_torch_ref)
    def ref(**kwargs):
        kwargs["input"] = unpacked_fp8_data
        kwargs["qkv_in_scale"] = row_scales
        return wrapped_torch_ref(**kwargs)

    return ref


def strided_gather_ref(wrapped_torch_ref, gathered_input):
    """Wrap a torch ref to substitute a pre-gathered input for the strided kernel path.

    The kernel receives full [B, S_full, H] input plus a StridedInputConfig and writes
    [B, num_local_tokens, I] output. This wrapper substitutes the pre-gathered
    [B, num_local_tokens, H] input so the torch ref computes the golden reference
    against the same tokens the kernel processes, and remaps the ref's ``"out"``
    key to ``"output_hbm"`` to match the caller-provided output_hbm tensor.
    """

    @functools.wraps(wrapped_torch_ref)
    def ref(**kwargs):
        kwargs["input"] = gathered_input
        ref_result = wrapped_torch_ref(**kwargs)
        if "out" in ref_result:
            ref_result = {"output_hbm": ref_result.pop("out"), **ref_result}
        return ref_result

    return ref
