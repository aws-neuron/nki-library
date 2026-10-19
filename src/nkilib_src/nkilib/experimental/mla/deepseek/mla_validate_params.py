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

"""Shared validators for the MLA CTE kernels params (MX qkv, attention, MX V-up + MX o_proj)"""

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from .mla_common_cte import _DS_SCALE_BLOCK, _H_PACK, _K_CHUNK, _MM1_TILE, _MX_SCALE_BLOCK, _P_MAX, MlaPrecision


def _validate_mla_qkv_inputs(
    x_hbm_mx: nl.NkiTensor,
    wqkv_a_hbm: nl.NkiTensor,
    wqkv_a_scale_hbm: nl.NkiTensor,
    wq_b_hbm: nl.NkiTensor,
    wq_b_scale_hbm: nl.NkiTensor,
    q_norm_gamma_hbm: nl.NkiTensor,
    kv_norm_gamma_hbm: nl.NkiTensor,
    wuk_hbm: nl.NkiTensor,
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    compact_scales: bool = True,
    precision: MlaPrecision = MlaPrecision.MX,
) -> None:
    """Validate all inputs to the MLA QKV CTE kernel.

    Performs comprehensive validation of tensor shapes, dtypes, and dimension
    parameters before kernel execution begins. Ensures:
      - The activation has the correct layout and dtype for ``precision``.
      - Weights are fp8x4-packed (MX) or bf16 ``[K, N]`` (BF16) with correct shapes.
      - Compact block-128 scales have the expected shape (MX only).
      - Norm gammas are bf16 with the correct shape.
      - The absorption weight W_uk is bf16 with the correct shape.
      - RoPE caches are bf16 with the correct shape.
      - Dimension parameters satisfy hardware constraints.

    Raises:
        AssertionError: If any validation check fails.
    """
    if precision.is_bf16():
        _validate_mla_qkv_inputs_bf16(
            x_hbm=x_hbm_mx,
            wqkv_a_hbm=wqkv_a_hbm,
            wq_b_hbm=wq_b_hbm,
            q_norm_gamma_hbm=q_norm_gamma_hbm,
            kv_norm_gamma_hbm=kv_norm_gamma_hbm,
            wuk_hbm=wuk_hbm,
            cos_cache_hbm=cos_cache_hbm,
            sin_cache_hbm=sin_cache_hbm,
            n_heads=n_heads,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            kv_lora_rank=kv_lora_rank,
            qk_lora_rank=qk_lora_rank,
        )
        return

    P_MAX = nl.tile_size.pmax
    _PREFIX = "[MLA QKV CTE]"

    # ---- Dimension parameter constraints ----
    kernel_assert(
        qk_nope_head_dim == 128,
        f"{_PREFIX} qk_nope_head_dim must be 128 (full-partition bf16 absorption contraction), got "
        f"{qk_nope_head_dim}. MlaPrecision.BF16 tiles this contraction and accepts any value.",
    )
    kernel_assert(
        qk_rope_head_dim > 0 and qk_rope_head_dim % 2 == 0,
        f"{_PREFIX} qk_rope_head_dim must be positive and even (interleaved RoPE), got {qk_rope_head_dim}.",
    )
    kernel_assert(
        n_heads >= 2 and n_heads % 2 == 0,
        f"{_PREFIX} n_heads must be >= 2 and even (head-group loop requires even divisor), got {n_heads}.",
    )
    kernel_assert(
        kv_lora_rank > 0 and kv_lora_rank % (P_MAX * _H_PACK) == 0,
        f"{_PREFIX} kv_lora_rank must be a positive multiple of {P_MAX * _H_PACK}, got {kv_lora_rank}.",
    )
    kernel_assert(
        qk_lora_rank > 0 and qk_lora_rank % (P_MAX * _H_PACK) == 0,
        f"{_PREFIX} qk_lora_rank must be a positive multiple of {P_MAX * _H_PACK}, got {qk_lora_rank}.",
    )

    # ---- Derive dimensions from weight shapes ----
    kernel_assert(
        len(wqkv_a_hbm.shape) == 2,
        f"{_PREFIX} wqkv_a_hbm must be 2D [H//4, qkv_out_dim], got shape {wqkv_a_hbm.shape}.",
    )
    H = wqkv_a_hbm.shape[0] * _H_PACK
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
    qkv_out_dim = qk_lora_rank + kv_a_out_dim

    kernel_assert(
        H % P_MAX == 0,
        f"{_PREFIX} Hidden dimension H must be a multiple of {P_MAX}, got {H}.",
    )
    kernel_assert(
        H % (P_MAX * _H_PACK) == 0,
        f"{_PREFIX} Hidden dimension H must be a multiple of {P_MAX * _H_PACK} (MX 512-tile), got {H}.",
    )

    # ---- Packed MX input (rmsnorm_mx_prefill pack_scales=True) ----
    kernel_assert(
        len(x_hbm_mx.shape) == 3,
        f"{_PREFIX} x_hbm_mx must be 3D [B, S, H + scale_region], got shape {x_hbm_mx.shape}.",
    )
    B, S, mx_row = x_hbm_mx.shape
    n_H512 = H // (P_MAX * _H_PACK)
    n_packed = div_ceil(n_H512, 4)
    expected_mx_row = H + n_packed * P_MAX
    kernel_assert(
        mx_row == expected_mx_row,
        f"{_PREFIX} x_hbm_mx last dimension must be H + ceil(H/2048)*128 = {expected_mx_row}, got {mx_row}. "
        f"Ensure the input is produced by rmsnorm_mx_prefill with pack_scales=True.",
    )
    kernel_assert(
        x_hbm_mx.dtype in [nl.float8_e4m3, nl.float8_e4m3fn],
        f"{_PREFIX} x_hbm_mx must have fp8 dtype (packed MX activation), got {x_hbm_mx.dtype}.",
    )

    # ---- wqkv_a (stage 1 first projection) ----
    kernel_assert(
        wqkv_a_hbm.shape == (H // _H_PACK, qkv_out_dim),
        f"{_PREFIX} wqkv_a_hbm shape must be [H//4, qkv_out_dim] = [{H // _H_PACK}, {qkv_out_dim}], "
        f"got {wqkv_a_hbm.shape}.",
    )
    # Torch has no native 4-packed fp8 dtype, so callers may pass the packed weights as
    # uint32 (same 4-byte word); the MX loaders reinterpret to fp8x4 via .ap(dtype=...).
    kernel_assert(
        wqkv_a_hbm.dtype in [nl.float8_e4m3fn_x4, nl.uint32],
        f"{_PREFIX} wqkv_a_hbm must have dtype float8_e4m3fn_x4 or uint32 (MX 4-packed), got {wqkv_a_hbm.dtype}.",
    )

    # ---- wqkv_a scale: compact block-128 [K//128, ceil(N/128)] or native block-32 [K//32, N] ----
    if compact_scales:
        expected_a_scale_shape = (H // _DS_SCALE_BLOCK, div_ceil(qkv_out_dim, _DS_SCALE_BLOCK))
        _a_scale_desc = "[H//128, ceil(qkv_out_dim/128)]"
    else:
        expected_a_scale_shape = (H // _MX_SCALE_BLOCK, qkv_out_dim)
        _a_scale_desc = "[H//32, qkv_out_dim] (native block-32)"
    kernel_assert(
        wqkv_a_scale_hbm.shape == expected_a_scale_shape,
        f"{_PREFIX} wqkv_a_scale_hbm shape must be {_a_scale_desc} = "
        f"{expected_a_scale_shape}, got {wqkv_a_scale_hbm.shape}.",
    )
    kernel_assert(
        wqkv_a_scale_hbm.dtype == nl.uint8,
        f"{_PREFIX} wqkv_a_scale_hbm must have dtype uint8 (compact block-128 scale), got {wqkv_a_scale_hbm.dtype}.",
    )

    # ---- wq_b (stage 2 Q second projection) ----
    q_out_dim = n_heads * qk_head_dim
    expected_wq_b_shape = (qk_lora_rank // _H_PACK, q_out_dim)
    kernel_assert(
        len(wq_b_hbm.shape) == 2,
        f"{_PREFIX} wq_b_hbm must be 2D [qk_lora_rank//4, n_heads*qk_head_dim], got shape {wq_b_hbm.shape}.",
    )
    kernel_assert(
        wq_b_hbm.shape == expected_wq_b_shape,
        f"{_PREFIX} wq_b_hbm shape must be [qk_lora_rank//4, q_out_dim] = {expected_wq_b_shape}, got {wq_b_hbm.shape}.",
    )
    kernel_assert(
        wq_b_hbm.dtype in [nl.float8_e4m3fn_x4, nl.uint32],
        f"{_PREFIX} wq_b_hbm must have dtype float8_e4m3fn_x4 or uint32 (MX 4-packed), got {wq_b_hbm.dtype}.",
    )

    # ---- wq_b scale: compact block-128 [K//128, ceil(N/128)] or native block-32 [K//32, N] ----
    if compact_scales:
        expected_b_scale_shape = (qk_lora_rank // _DS_SCALE_BLOCK, div_ceil(q_out_dim, _DS_SCALE_BLOCK))
        _b_scale_desc = "[qk_lora_rank//128, ceil(q_out_dim/128)]"
    else:
        expected_b_scale_shape = (qk_lora_rank // _MX_SCALE_BLOCK, q_out_dim)
        _b_scale_desc = "[qk_lora_rank//32, q_out_dim] (native block-32)"
    kernel_assert(
        wq_b_scale_hbm.shape == expected_b_scale_shape,
        f"{_PREFIX} wq_b_scale_hbm shape must be {_b_scale_desc} = "
        f"{expected_b_scale_shape}, got {wq_b_scale_hbm.shape}.",
    )
    kernel_assert(
        wq_b_scale_hbm.dtype == nl.uint8,
        f"{_PREFIX} wq_b_scale_hbm must have dtype uint8 (compact block-128 scale), got {wq_b_scale_hbm.dtype}.",
    )

    # ---- q_norm_gamma ----
    kernel_assert(
        q_norm_gamma_hbm.shape == (1, qk_lora_rank),
        f"{_PREFIX} q_norm_gamma_hbm shape must be (1, qk_lora_rank) = (1, {qk_lora_rank}), "
        f"got {q_norm_gamma_hbm.shape}.",
    )

    # ---- kv_norm_gamma ----
    kernel_assert(
        kv_norm_gamma_hbm.shape == (1, kv_lora_rank),
        f"{_PREFIX} kv_norm_gamma_hbm shape must be (1, kv_lora_rank) = (1, {kv_lora_rank}), "
        f"got {kv_norm_gamma_hbm.shape}.",
    )
    kernel_assert(
        kv_norm_gamma_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} kv_norm_gamma_hbm must have dtype bfloat16, got {kv_norm_gamma_hbm.dtype}.",
    )

    # ---- wuk (absorption weight, bf16) ----
    expected_wuk_shape = (qk_nope_head_dim, n_heads * kv_lora_rank)
    kernel_assert(
        wuk_hbm.shape == expected_wuk_shape,
        f"{_PREFIX} wuk_hbm shape must be (qk_nope_head_dim, n_heads*kv_lora_rank) = "
        f"{expected_wuk_shape}, got {wuk_hbm.shape}.",
    )

    # ---- RoPE caches ----
    expected_rope_shape = (B, S, qk_rope_head_dim)
    kernel_assert(
        cos_cache_hbm.shape == expected_rope_shape,
        f"{_PREFIX} cos_cache_hbm shape must be (B, S, qk_rope_head_dim) = {expected_rope_shape}, "
        f"got {cos_cache_hbm.shape}.",
    )
    kernel_assert(
        sin_cache_hbm.shape == expected_rope_shape,
        f"{_PREFIX} sin_cache_hbm shape must be (B, S, qk_rope_head_dim) = {expected_rope_shape}, "
        f"got {sin_cache_hbm.shape}.",
    )


_FP8_CACHE_DTYPES = (nl.float8_e4m3, nl.float8_e4m3fn)
# Bytes reserved per fp8 cache row for the per-token scale; mirrors mla_qkv_cte._KV_SCALE_REGION.
_KV_SCALE_REGION = 16


def _validate_paged_kv_store(
    kv_cache_hbm: nl.NkiTensor,
    slot_mapping_hbm: nl.NkiTensor,
    kv_cache_fp8_dtype,
    kv_row_bf16: int,
    B: int,
    S: int,
    precision: MlaPrecision = MlaPrecision.MX,
) -> None:
    """Validate the optional fused paged KV cache store args of the MLA QKV kernel."""
    _PREFIX = "[QKV MLA paged KV store]"
    if kv_cache_hbm is None and slot_mapping_hbm is None:
        return
    kernel_assert(
        kv_cache_hbm is not None and slot_mapping_hbm is not None,
        f"{_PREFIX} kv_cache_hbm and slot_mapping_hbm must be given together.",
    )
    kernel_assert(
        precision.is_bf16(),
        f"{_PREFIX} the paged KV cache store is only supported with MlaPrecision.BF16 (the GLM "
        f"config), got {precision}. The MX path would also need an OCP-compliant fp8 dtype for the "
        f"latent span (its weights are float8_e4m3fn_x4, and NCC_EOCP001 forbids mixing OCP and "
        f"legacy fp8 in one kernel); add it when an MX consumer needs it.",
    )
    kernel_assert(
        kv_cache_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} kv_cache_hbm must be bfloat16, got {kv_cache_hbm.dtype}. bf16 is the row's "
        f"CONTAINER, not the latent's format: it keeps the row a single 2-byte dtype so the "
        f"consumer gathers a whole token in one descriptor and dma_transpose stays legal, while "
        f"the latent span is bitcast to fp8.",
    )
    kernel_assert(
        kv_cache_fp8_dtype in _FP8_CACHE_DTYPES,
        f"{_PREFIX} kv_cache_fp8_dtype must be one of "
        f"{', '.join(str(d) for d in _FP8_CACHE_DTYPES)}, got {kv_cache_fp8_dtype}.",
    )
    kernel_assert(
        len(kv_cache_hbm.shape) >= 2 and kv_cache_hbm.shape[-1] == kv_row_bf16,
        f"{_PREFIX} kv_cache_hbm's most-minor dim must be {kv_row_bf16} bf16 elements (the fp8 "
        f"latent bitcast 2-per-bf16, then the scale, then k_pe, padded to a 16-byte row stride), "
        f"got shape {kv_cache_hbm.shape}.",
    )

    n_slot_entries = 1
    for d in slot_mapping_hbm.shape:
        n_slot_entries *= d
    kernel_assert(
        n_slot_entries == B * S,
        f"{_PREFIX} slot_mapping_hbm must hold one slot per token, i.e. B * S = {B * S} entries, "
        f"got shape {slot_mapping_hbm.shape}.",
    )
    kernel_assert(
        slot_mapping_hbm.dtype in (nl.int32, nl.uint32),
        f"{_PREFIX} slot_mapping_hbm must be int32/uint32, got {slot_mapping_hbm.dtype}.",
    )


def _validate_mla_qkv_inputs_bf16(
    x_hbm: nl.NkiTensor,
    wqkv_a_hbm: nl.NkiTensor,
    wq_b_hbm: nl.NkiTensor,
    q_norm_gamma_hbm: nl.NkiTensor,
    kv_norm_gamma_hbm: nl.NkiTensor,
    wuk_hbm: nl.NkiTensor,
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
) -> None:
    """Validate the BF16 (``MlaPrecision.BF16``) inputs to the MLA QKV CTE kernel.

    Every tensor is plain bf16 with the contraction on rows and NATURAL column order; there
    are no scale tensors, no 4-packing and no output-column swizzle. Dimension constraints are
    correspondingly looser than the MX path's:

      - K dimensions must be multiples of 128 (the partition cap), not 512 (MX's packed tile).
      - ``qk_nope_head_dim`` is unconstrained -- the absorption contraction is K tiled, so
        GLM-MoE-DSA's 192 is legal where the MX path demands exactly 128.

    Raises:
        AssertionError: If any validation check fails.
    """
    _PREFIX = "[MLA QKV CTE bf16]"
    _K_TILE = 128

    # ---- Dimension parameter constraints ----
    kernel_assert(
        qk_nope_head_dim > 0,
        f"{_PREFIX} qk_nope_head_dim must be positive, got {qk_nope_head_dim}.",
    )
    kernel_assert(
        qk_rope_head_dim > 0 and qk_rope_head_dim % 2 == 0,
        f"{_PREFIX} qk_rope_head_dim must be positive and even (interleaved RoPE), got {qk_rope_head_dim}.",
    )
    kernel_assert(
        n_heads >= 2 and n_heads % 2 == 0,
        f"{_PREFIX} n_heads must be >= 2 and even (head-group loop requires an even divisor), got {n_heads}.",
    )
    kernel_assert(
        kv_lora_rank > 0 and kv_lora_rank % _K_TILE == 0,
        f"{_PREFIX} kv_lora_rank must be a positive multiple of {_K_TILE}, got {kv_lora_rank}.",
    )
    kernel_assert(
        qk_lora_rank > 0 and qk_lora_rank % _K_TILE == 0,
        f"{_PREFIX} qk_lora_rank must be a positive multiple of {_K_TILE}, got {qk_lora_rank}.",
    )

    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_out_dim = n_heads * qk_head_dim
    qkv_out_dim = qk_lora_rank + kv_lora_rank + qk_rope_head_dim

    # ---- wqkv_a (stage 1 fused projection): [H, qkv_out_dim] ----
    kernel_assert(
        len(wqkv_a_hbm.shape) == 2,
        f"{_PREFIX} wqkv_a_hbm must be 2D [H, qkv_out_dim], got shape {wqkv_a_hbm.shape}.",
    )
    H = wqkv_a_hbm.shape[0]
    kernel_assert(
        H % _K_TILE == 0,
        f"{_PREFIX} hidden dimension H must be a multiple of {_K_TILE}, got {H}.",
    )
    kernel_assert(
        wqkv_a_hbm.shape == (H, qkv_out_dim),
        f"{_PREFIX} wqkv_a_hbm shape must be [H, qk_lora_rank + kv_lora_rank + qk_rope_head_dim] = "
        f"[{H}, {qkv_out_dim}], got {wqkv_a_hbm.shape}.",
    )
    kernel_assert(
        wqkv_a_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} wqkv_a_hbm must have dtype bfloat16, got {wqkv_a_hbm.dtype}.",
    )

    # ---- Activation: plain bf16 [B, S, H], no packed scale region ----
    kernel_assert(
        len(x_hbm.shape) == 3,
        f"{_PREFIX} x_hbm must be 3D [B, S, H], got shape {x_hbm.shape}.",
    )
    B, S, x_h = x_hbm.shape
    kernel_assert(
        x_h == H,
        f"{_PREFIX} x_hbm last dimension must equal H = {H} (bf16 activations carry no scale region), got {x_h}.",
    )
    kernel_assert(
        x_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} x_hbm must have dtype bfloat16, got {x_hbm.dtype}. Pass the packed fp8 activation "
        f"only with MlaPrecision.MX.",
    )

    # ---- wq_b (stage 2 Q second projection): [qk_lora_rank, n_heads * qk_head_dim] ----
    expected_wq_b_shape = (qk_lora_rank, q_out_dim)
    kernel_assert(
        wq_b_hbm.shape == expected_wq_b_shape,
        f"{_PREFIX} wq_b_hbm shape must be [qk_lora_rank, n_heads*qk_head_dim] = {expected_wq_b_shape}, "
        f"got {wq_b_hbm.shape}.",
    )
    kernel_assert(
        wq_b_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} wq_b_hbm must have dtype bfloat16, got {wq_b_hbm.dtype}.",
    )

    # ---- Norm gammas ----
    kernel_assert(
        q_norm_gamma_hbm.shape == (1, qk_lora_rank),
        f"{_PREFIX} q_norm_gamma_hbm shape must be (1, qk_lora_rank) = (1, {qk_lora_rank}), "
        f"got {q_norm_gamma_hbm.shape}.",
    )
    kernel_assert(
        kv_norm_gamma_hbm.shape == (1, kv_lora_rank),
        f"{_PREFIX} kv_norm_gamma_hbm shape must be (1, kv_lora_rank) = (1, {kv_lora_rank}), "
        f"got {kv_norm_gamma_hbm.shape}.",
    )
    kernel_assert(
        q_norm_gamma_hbm.dtype == nl.bfloat16 and kv_norm_gamma_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} norm gammas must have dtype bfloat16, got q={q_norm_gamma_hbm.dtype}, "
        f"kv={kv_norm_gamma_hbm.dtype}.",
    )

    # ---- wuk (absorption weight): [qk_nope_head_dim, n_heads * kv_lora_rank] ----
    expected_wuk_shape = (qk_nope_head_dim, n_heads * kv_lora_rank)
    kernel_assert(
        wuk_hbm.shape == expected_wuk_shape,
        f"{_PREFIX} wuk_hbm shape must be (qk_nope_head_dim, n_heads*kv_lora_rank) = "
        f"{expected_wuk_shape}, got {wuk_hbm.shape}.",
    )
    kernel_assert(
        wuk_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} wuk_hbm must have dtype bfloat16, got {wuk_hbm.dtype}.",
    )

    # ---- RoPE caches ----
    expected_rope_shape = (B, S, qk_rope_head_dim)
    kernel_assert(
        cos_cache_hbm.shape == expected_rope_shape,
        f"{_PREFIX} cos_cache_hbm shape must be (B, S, qk_rope_head_dim) = {expected_rope_shape}, "
        f"got {cos_cache_hbm.shape}.",
    )
    kernel_assert(
        sin_cache_hbm.shape == expected_rope_shape,
        f"{_PREFIX} sin_cache_hbm shape must be (B, S, qk_rope_head_dim) = {expected_rope_shape}, "
        f"got {sin_cache_hbm.shape}.",
    )


def _validate_mla_attention_inputs(
    q_lift_hbm: nl.NkiTensor,
    q_pe_hbm: nl.NkiTensor,
    c_kv_hbm: nl.NkiTensor,
    k_pe_hbm: nl.NkiTensor,
    topk_indices_hbm: nl.NkiTensor = None,
    topk_tiled: bool = False,
    dense: bool = False,
    prior_tokens_hbm: nl.NkiTensor = None,
    cp_rank_hbm: nl.NkiTensor = None,
    cp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    kv_cache_packed: bool = False,
) -> None:
    """Validate all inputs to the sparse MLA latent + RoPE attention kernel (kernel A).

    Performs comprehensive validation of tensor shapes, dtypes, and the topk-index
    layout before kernel execution begins. Ensures:
      - q_lift / q_pe / c_kv / k_pe are bf16 with matching latent (L) and RoPE (R) dims.
      - The single-batch, single-latent-tile (L == 512) constraints hold.
      - H satisfies the transpose xbar-alignment constraint (multiple of 16, <= P_MAX).
      - The topk index tensor matches the selected layout and yields a legal K.
      - The optional CP scalar tensors (prior_tokens / cp_rank) are single elements and the CP
        interleave params are powers of two that partition S_kv.

    Raises:
        AssertionError: If any validation check fails.
    """
    _PREFIX = "[MLA Sparse Attention CTE]"

    # ---- q_lift is the shape source of truth: [B, S, H, L] ----
    kernel_assert(
        len(q_lift_hbm.shape) == 4,
        f"{_PREFIX} q_lift_hbm must be 4D [B, S, H, L], got shape {q_lift_hbm.shape}.",
    )
    B, S, H, L = q_lift_hbm.shape
    kernel_assert(
        len(q_pe_hbm.shape) == 4,
        f"{_PREFIX} q_pe_hbm must be 4D [B, S, H, R], got shape {q_pe_hbm.shape}.",
    )
    R = q_pe_hbm.shape[3]
    # Dense mode ignores topk_indices; sparse mode REQUIRES it (it drives the
    # per-query gather). Guard here so a None/omitted index in sparse mode fails
    # with a clear message instead of a downstream shape/attr error.
    kernel_assert(
        dense or topk_indices_hbm is not None,
        f"{_PREFIX} topk_indices_hbm is required when dense=False (sparse "
        f"path); pass dense=True for the full-prefix/no-indexer case.",
    )
    # Dense (no-indexer) attends all S_kv keys: K == S_kv, topk_indices is an unused dummy.
    if dense:
        # Dense attends every cache slot, so K == S_kv. A packed cache is [S_kv, row], so S_kv is
        # the major dim there; unpacked is [B, S_kv, L].
        K = c_kv_hbm.shape[0] if kv_cache_packed else c_kv_hbm.shape[1]
    else:
        K = (topk_indices_hbm.shape[3] * 16) if topk_tiled else topk_indices_hbm.shape[2]

    # ---- Dimension constraints ----
    kernel_assert(B == 1, f"{_PREFIX} only B == 1 is supported, got B={B}.")
    kernel_assert(0 < H <= _P_MAX, f"{_PREFIX} H must be in (0, {_P_MAX}], got {H}.")
    # H % 16: the q_lift transpose output has H as its step axis; H*bf16 must be 32B-aligned
    # (else NCC_IBIR155 xbar-alignment failure).
    kernel_assert(
        H % 16 == 0,
        f"{_PREFIX} H must be a multiple of 16 (q_lift transpose output step H*bf16 must be 32B-aligned), got {H}.",
    )
    # L == 512 (DeepSeek kv_lora_rank): the MM2 4-pack permute + o_proj consumer + ref all
    # assume the single 512-wide latent tile.
    kernel_assert(
        L == _P_MAX * _H_PACK,
        f"{_PREFIX} L must be {_P_MAX * _H_PACK} (DeepSeek kv_lora_rank; MM2 4-pack permute assumes "
        f"single-latent-tile), got {L}.",
    )
    kernel_assert(0 < R <= _P_MAX, f"{_PREFIX} R must be in (0, {_P_MAX}], got {R}.")
    kernel_assert(K % _K_CHUNK == 0, f"{_PREFIX} K must be a multiple of {_K_CHUNK}, got {K}.")
    kernel_assert(K % 16 == 0, f"{_PREFIX} K must be a multiple of 16 (tensor-indirection gather), got {K}.")

    # ---- q_pe / c_kv / k_pe shape + dtype agreement ----
    kernel_assert(
        q_pe_hbm.shape[0] == B and q_pe_hbm.shape[1] == S and q_pe_hbm.shape[2] == H,
        f"{_PREFIX} q_pe_hbm leading dims must match q_lift [B, S, H] = [{B}, {S}, {H}], got {q_pe_hbm.shape[:3]}.",
    )
    if kv_cache_packed:
        # PACKED: one fused bf16 row per token carrying the fp8 latent, its scale and k_pe, so
        # c_kv_hbm is [S_kv, row] and k_pe_hbm is unused.
        _row = div_ceil(L // 2 + 1 + R, 8) * 8
        kernel_assert(
            len(c_kv_hbm.shape) == 2 and c_kv_hbm.shape[1] == _row,
            f"{_PREFIX} with kv_cache_packed the KV cache must be [S_kv, {_row}] (L//2 bf16 fp8 "
            f"containers + scale + k_pe, padded to a 16-byte row), got {c_kv_hbm.shape}.",
        )
        kernel_assert(
            c_kv_hbm.dtype == nl.bfloat16,
            f"{_PREFIX} the packed KV cache must be bfloat16 (the row's 2-byte CONTAINER; the "
            f"latent span is bitcast to fp8), got {c_kv_hbm.dtype}.",
        )
    else:
        kernel_assert(
            len(c_kv_hbm.shape) == 3 and c_kv_hbm.shape[0] == B and c_kv_hbm.shape[2] == L,
            f"{_PREFIX} c_kv_hbm must be [B, S_kv, L] with B={B}, L={L}, got {c_kv_hbm.shape}.",
        )
        kernel_assert(
            len(k_pe_hbm.shape) == 3 and k_pe_hbm.shape[0] == B and k_pe_hbm.shape[2] == R,
            f"{_PREFIX} k_pe_hbm must be [B, S_kv, R] with B={B}, R={R}, got {k_pe_hbm.shape}.",
        )
        kernel_assert(
            c_kv_hbm.shape[1] == k_pe_hbm.shape[1],
            f"{_PREFIX} c_kv and k_pe must share S_kv, got {c_kv_hbm.shape[1]} vs {k_pe_hbm.shape[1]}.",
        )
    kernel_assert(
        q_lift_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} q_lift_hbm must have dtype bfloat16, got {q_lift_hbm.dtype}.",
    )
    kernel_assert(
        q_pe_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} q_pe_hbm must have dtype bfloat16, got {q_pe_hbm.dtype}.",
    )
    kernel_assert(
        c_kv_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} c_kv_hbm must have dtype bfloat16, got {c_kv_hbm.dtype}.",
    )
    kernel_assert(
        k_pe_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} k_pe_hbm must have dtype bfloat16, got {k_pe_hbm.dtype}.",
    )

    # ---- topk indices layout (skipped in dense: no gather, topk_indices is an unused dummy) ----
    if not dense:
        if topk_tiled:
            kernel_assert(
                len(topk_indices_hbm.shape) == 4,
                f"{_PREFIX} topk_indices_hbm (tiled) must be 4D "
                f"[num_s_tiles, NUM_TOPK_BATCHES, P_MAX, K//16], got shape {topk_indices_hbm.shape}.",
            )
            kernel_assert(
                topk_indices_hbm.shape[2] == _P_MAX,
                f"{_PREFIX} topk_indices_hbm (tiled) partition dim must be {_P_MAX}, got {topk_indices_hbm.shape[2]}.",
            )
        else:
            kernel_assert(
                len(topk_indices_hbm.shape) == 3 and topk_indices_hbm.shape[0] == B and topk_indices_hbm.shape[1] == S,
                f"{_PREFIX} topk_indices_hbm (flat) must be [B, S, K] = [{B}, {S}, K], got {topk_indices_hbm.shape}.",
            )
        kernel_assert(
            topk_indices_hbm.dtype == nl.int32,
            f"{_PREFIX} topk_indices_hbm must have dtype int32, got {topk_indices_hbm.dtype}.",
        )

    # ---- CP interleave params + runtime scalar tensors (defaults = non-CP identity) ----
    # cp_world_size == 1 is the non-CP case; > 1 means the KV cache is stored rank-major
    # interleaved and topk indices are physical slots (see the kernel's owned() conversion).
    # A packed cache is [S_kv, row], so tokens are the MAJOR dim there; unpacked is [B, S_kv, L].
    S_kv = c_kv_hbm.shape[0] if kv_cache_packed else c_kv_hbm.shape[1]
    kernel_assert(
        cp_world_size >= 1 and (cp_world_size & (cp_world_size - 1)) == 0,
        f"{_PREFIX} cp_world_size must be a positive power of two, got {cp_world_size}.",
    )
    kernel_assert(
        cp_kv_cache_interleave_size >= 1 and (cp_kv_cache_interleave_size & (cp_kv_cache_interleave_size - 1)) == 0,
        f"{_PREFIX} cp_kv_cache_interleave_size must be a positive power of two, got {cp_kv_cache_interleave_size}.",
    )
    if cp_world_size > 1:
        kernel_assert(
            S_kv % cp_world_size == 0,
            f"{_PREFIX} S_kv ({S_kv}) must be divisible by cp_world_size ({cp_world_size}).",
        )
        S_kv_local = S_kv // cp_world_size
        kernel_assert(
            (S_kv_local & (S_kv_local - 1)) == 0,
            f"{_PREFIX} S_kv // cp_world_size ({S_kv_local}) must be a power of two (interleave "
            f"decode uses it as a shift).",
        )
        kernel_assert(
            S_kv_local % cp_kv_cache_interleave_size == 0,
            f"{_PREFIX} cp_kv_cache_interleave_size ({cp_kv_cache_interleave_size}) must divide "
            f"S_kv // cp_world_size ({S_kv_local}).",
        )
    for _name, _t in (("prior_tokens_hbm", prior_tokens_hbm), ("cp_rank_hbm", cp_rank_hbm)):
        if _t is not None:
            kernel_assert(
                tuple(_t.shape) == (1, 1),
                f"{_PREFIX} {_name} must be a [1, 1] scalar tensor, got shape {_t.shape}.",
            )
            kernel_assert(
                _t.dtype in [nl.int32, nl.uint32],
                f"{_PREFIX} {_name} must have dtype int32 or uint32, got {_t.dtype}.",
            )


def _validate_mla_vupmx_oproj_inputs(
    out_attn_hbm: nl.NkiTensor,
    wuv_qtz_hbm: nl.NkiTensor,
    wuv_scale_hbm: nl.NkiTensor,
    wo_qtz_hbm: nl.NkiTensor,
    wo_scale_hbm: nl.NkiTensor,
    compact_scales: bool = True,
    precision: MlaPrecision = MlaPrecision.MX,
    kv_lora_rank: int = _P_MAX * _H_PACK,
) -> None:
    """Validate all inputs to the V-up + o_proj kernel (kernel B).

    Performs comprehensive validation of tensor shapes and dtypes before kernel
    execution begins. Ensures:
      - The latent attention input is bf16 with a single-batch layout.
      - V-up / o_proj weights are fp8x4-packed (MX) or bf16 [K, N] (BF16), correctly shaped.
      - Compact block-128 scales have the expected shapes (MX only).
      - Dimension parameters (d_v, H, Hdv, HID) satisfy hardware constraints.

    H is recovered from tensor shapes as ``H*L / L``: wuv_qtz_hbm is [H*L // 4, d_v] (MX) or
    [H*L, d_v] (BF16). H is NOT a runtime scalar. L is 512 on the MX path and
    ``kv_lora_rank`` on the BF16 path.

    Raises:
        AssertionError: If any validation check fails.
    """
    if precision.is_bf16():
        _validate_mla_vup_oproj_inputs_bf16(
            out_attn_hbm=out_attn_hbm,
            wuv_hbm=wuv_qtz_hbm,
            wo_hbm=wo_qtz_hbm,
            kv_lora_rank=kv_lora_rank,
        )
        return

    _PREFIX = "[MLA V-up + O-proj CTE]"
    L = _P_MAX * _H_PACK  # 512

    # ---- Latent attention input [B, S, H*L] ----
    kernel_assert(
        len(out_attn_hbm.shape) == 3,
        f"{_PREFIX} out_attn_hbm must be 3D [B, S, H*L], got shape {out_attn_hbm.shape}.",
    )
    B, _S, HL = out_attn_hbm.shape
    kernel_assert(B == 1, f"{_PREFIX} only B == 1 is supported, got B={B}.")
    kernel_assert(
        HL % L == 0 and HL > 0,
        f"{_PREFIX} out_attn last dimension H*L ({HL}) must be a positive multiple of L={L}.",
    )
    H = HL // L
    kernel_assert(
        H % _H_PACK == 0,
        f"{_PREFIX} H (heads/rank = H*L // {L}) must be a multiple of {_H_PACK} "
        f"(MX o_proj tiles {_H_PACK} heads per 512-block), got H={H}.",
    )
    kernel_assert(
        0 < H <= _P_MAX,
        f"{_PREFIX} H (heads/rank) must be in (0, {_P_MAX}], got {H}.",
    )
    kernel_assert(
        out_attn_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} out_attn_hbm must have dtype bfloat16, got {out_attn_hbm.dtype}.",
    )

    # ---- wuv (MX V-up weight): [H*L // 4, d_v] ----
    kernel_assert(
        len(wuv_qtz_hbm.shape) == 2,
        f"{_PREFIX} wuv_qtz_hbm must be 2D [H*L//4, d_v], got shape {wuv_qtz_hbm.shape}.",
    )
    kernel_assert(
        wuv_qtz_hbm.shape[0] * _H_PACK == HL,
        f"{_PREFIX} wuv_qtz_hbm rows*4 ({wuv_qtz_hbm.shape[0] * _H_PACK}) must equal H*L ({HL}).",
    )
    d_v = wuv_qtz_hbm.shape[1]
    kernel_assert(
        d_v == _P_MAX,
        f"{_PREFIX} d_v (per-head V dim) must be {_P_MAX} (pre-swizzled transpose requirement), got {d_v}.",
    )
    kernel_assert(
        wuv_qtz_hbm.dtype in [nl.float8_e4m3fn_x4, nl.uint32],
        f"{_PREFIX} wuv_qtz_hbm must have dtype float8_e4m3fn_x4 or uint32 (MX 4-packed), got {wuv_qtz_hbm.dtype}.",
    )

    # ---- wuv scale: compact block-128 [H*L//128, ceil(d_v/128)] or native block-32 [H*L//32, d_v] ----
    if compact_scales:
        expected_wuv_scale_shape = (HL // _DS_SCALE_BLOCK, div_ceil(d_v, _DS_SCALE_BLOCK))
        _wuv_scale_desc = "[H*L//128, ceil(d_v/128)]"
    else:
        expected_wuv_scale_shape = (HL // _MX_SCALE_BLOCK, d_v)
        _wuv_scale_desc = "[H*L//32, d_v] (native block-32)"
    kernel_assert(
        wuv_scale_hbm.shape == expected_wuv_scale_shape,
        f"{_PREFIX} wuv_scale_hbm shape must be {_wuv_scale_desc} = "
        f"{expected_wuv_scale_shape}, got {wuv_scale_hbm.shape}.",
    )
    kernel_assert(
        wuv_scale_hbm.dtype == nl.uint8,
        f"{_PREFIX} wuv_scale_hbm must have dtype uint8 (compact block-128 scale), got {wuv_scale_hbm.dtype}.",
    )

    # ---- wo (MX o_proj weight): [H*d_v // 4, HID] ----
    Hdv = H * d_v
    kernel_assert(
        len(wo_qtz_hbm.shape) == 2,
        f"{_PREFIX} wo_qtz_hbm must be 2D [H*d_v//4, HID], got shape {wo_qtz_hbm.shape}.",
    )
    kernel_assert(
        wo_qtz_hbm.shape[0] * _H_PACK == Hdv,
        f"{_PREFIX} wo_qtz_hbm rows*4 ({wo_qtz_hbm.shape[0] * _H_PACK}) must equal H*d_v ({Hdv}).",
    )
    HID = wo_qtz_hbm.shape[1]
    kernel_assert(
        Hdv % (_P_MAX * _H_PACK) == 0,
        f"{_PREFIX} H*d_v ({Hdv}) must be a multiple of {_P_MAX * _H_PACK} (MX 512-tile).",
    )
    kernel_assert(
        HID > 0 and HID % _MM1_TILE == 0,
        f"{_PREFIX} HID ({HID}) must be a positive multiple of {_MM1_TILE} (o_proj HID-tile).",
    )
    kernel_assert(
        wo_qtz_hbm.dtype in [nl.float8_e4m3fn_x4, nl.uint32],
        f"{_PREFIX} wo_qtz_hbm must have dtype float8_e4m3fn_x4 or uint32 (MX 4-packed), got {wo_qtz_hbm.dtype}.",
    )

    # ---- wo scale: compact block-128 [H*d_v//128, ceil(HID/128)] or native block-32 [H*d_v//32, HID] ----
    if compact_scales:
        expected_wo_scale_shape = (Hdv // _DS_SCALE_BLOCK, div_ceil(HID, _DS_SCALE_BLOCK))
        _wo_scale_desc = "[H*d_v//128, ceil(HID/128)]"
    else:
        expected_wo_scale_shape = (Hdv // _MX_SCALE_BLOCK, HID)
        _wo_scale_desc = "[H*d_v//32, HID] (native block-32)"
    kernel_assert(
        wo_scale_hbm.shape == expected_wo_scale_shape,
        f"{_PREFIX} wo_scale_hbm shape must be {_wo_scale_desc} = {expected_wo_scale_shape}, got {wo_scale_hbm.shape}.",
    )
    kernel_assert(
        wo_scale_hbm.dtype == nl.uint8,
        f"{_PREFIX} wo_scale_hbm must have dtype uint8 (compact block-128 scale), got {wo_scale_hbm.dtype}.",
    )


def _validate_mla_vup_oproj_inputs_bf16(
    out_attn_hbm: nl.NkiTensor,
    wuv_hbm: nl.NkiTensor,
    wo_hbm: nl.NkiTensor,
    kv_lora_rank: int,
) -> None:
    """Validate the BF16 (``MlaPrecision.BF16``) inputs to the V-up + o_proj kernel.

    Every tensor is plain bf16 with the contraction on rows and NATURAL column order: no 4-pack,
    no Hdv output-column swizzle, no scale tensors. Constraints are correspondingly looser than
    the MX path's:

      - ``d_v`` need only be a multiple of 128, not exactly 128 (GLM-MoE-DSA uses 256).
      - ``H`` need not be a multiple of 4 (that was the MX 4-heads-per-512-block tiling).
      - ``L`` comes from ``kv_lora_rank`` instead of being pinned to 512.

    Raises:
        AssertionError: If any validation check fails.
    """
    _PREFIX = "[MLA V-up + O-proj CTE bf16]"
    _K_TILE = 128
    L = kv_lora_rank

    kernel_assert(
        L > 0 and L % _K_TILE == 0,
        f"{_PREFIX} kv_lora_rank (L) must be a positive multiple of {_K_TILE}, got {L}.",
    )

    # ---- Latent attention input [B, S, H*L], natural latent column order ----
    kernel_assert(
        len(out_attn_hbm.shape) == 3,
        f"{_PREFIX} out_attn_hbm must be 3D [B, S, H*L], got shape {out_attn_hbm.shape}.",
    )
    B, _S, HL = out_attn_hbm.shape
    kernel_assert(B == 1, f"{_PREFIX} only B == 1 is supported, got B={B}.")
    kernel_assert(
        HL % L == 0 and HL > 0,
        f"{_PREFIX} out_attn last dimension H*L ({HL}) must be a positive multiple of L={L}.",
    )
    H = HL // L
    kernel_assert(0 < H <= _P_MAX, f"{_PREFIX} H (heads/rank) must be in (0, {_P_MAX}], got {H}.")
    kernel_assert(
        out_attn_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} out_attn_hbm must have dtype bfloat16, got {out_attn_hbm.dtype}.",
    )

    # ---- W_uv: [H*L, d_v] bf16 ----
    kernel_assert(
        len(wuv_hbm.shape) == 2,
        f"{_PREFIX} wuv_hbm must be 2D [H*L, d_v], got shape {wuv_hbm.shape}.",
    )
    kernel_assert(
        wuv_hbm.shape[0] == HL,
        f"{_PREFIX} wuv_hbm first dimension must be H*L = {HL} (bf16 weights are unpacked), got {wuv_hbm.shape[0]}.",
    )
    d_v = wuv_hbm.shape[1]
    kernel_assert(
        d_v > 0 and d_v % _K_TILE == 0,
        f"{_PREFIX} d_v ({d_v}) must be a positive multiple of {_K_TILE} so each head owns whole "
        f"partition tiles of the K-major attn_v buffer.",
    )
    kernel_assert(
        wuv_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} wuv_hbm must have dtype bfloat16, got {wuv_hbm.dtype}.",
    )

    # ---- W_o: [H*d_v, HID] bf16 ----
    Hdv = H * d_v
    kernel_assert(
        len(wo_hbm.shape) == 2,
        f"{_PREFIX} wo_hbm must be 2D [H*d_v, HID], got shape {wo_hbm.shape}.",
    )
    kernel_assert(
        wo_hbm.shape[0] == Hdv,
        f"{_PREFIX} wo_hbm first dimension must be H*d_v = {Hdv}, got {wo_hbm.shape[0]}.",
    )
    HID = wo_hbm.shape[1]
    kernel_assert(
        HID > 0 and HID % _MM1_TILE == 0,
        f"{_PREFIX} HID ({HID}) must be a positive multiple of {_MM1_TILE} (PSUM bank width).",
    )
    kernel_assert(
        wo_hbm.dtype == nl.bfloat16,
        f"{_PREFIX} wo_hbm must have dtype bfloat16, got {wo_hbm.dtype}.",
    )
