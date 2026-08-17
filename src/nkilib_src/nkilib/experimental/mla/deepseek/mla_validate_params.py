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
from .mla_common_cte import _DS_SCALE_BLOCK, _H_PACK, _K_CHUNK, _MM1_TILE, _P_MAX


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
) -> None:
    """Validate all inputs to the MLA QKV CTE kernel.

    Performs comprehensive validation of tensor shapes, dtypes, and dimension
    parameters before kernel execution begins. Ensures:
      - The packed MX input has the correct layout and fp8 dtype.
      - MX weights are fp8x4-packed with correct shapes.
      - Compact block-128 scales have the expected shape.
      - Norm gammas are bf16 with the correct shape.
      - The absorption weight W_uk is bf16 with the correct shape.
      - RoPE caches are bf16 with the correct shape.
      - Dimension parameters satisfy hardware constraints.

    Raises:
        AssertionError: If any validation check fails.
    """
    P_MAX = nl.tile_size.pmax
    _PREFIX = "[MLA QKV CTE]"

    # ---- Dimension parameter constraints ----
    kernel_assert(
        qk_nope_head_dim == 128,
        f"{_PREFIX} qk_nope_head_dim must be 128 (full-partition bf16 absorption contraction), got {qk_nope_head_dim}.",
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

    # ---- wqkv_a scale (compact block-128) ----
    expected_a_scale_shape = (H // _DS_SCALE_BLOCK, div_ceil(qkv_out_dim, _DS_SCALE_BLOCK))
    kernel_assert(
        wqkv_a_scale_hbm.shape == expected_a_scale_shape,
        f"{_PREFIX} wqkv_a_scale_hbm shape must be [H//128, ceil(qkv_out_dim/128)] = "
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

    # ---- wq_b scale (compact block-128) ----
    expected_b_scale_shape = (qk_lora_rank // _DS_SCALE_BLOCK, div_ceil(q_out_dim, _DS_SCALE_BLOCK))
    kernel_assert(
        wq_b_scale_hbm.shape == expected_b_scale_shape,
        f"{_PREFIX} wq_b_scale_hbm shape must be [qk_lora_rank//128, ceil(q_out_dim/128)] = "
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


def _validate_mla_attention_inputs(
    q_lift_hbm: nl.NkiTensor,
    q_pe_hbm: nl.NkiTensor,
    c_kv_hbm: nl.NkiTensor,
    k_pe_hbm: nl.NkiTensor,
    topk_indices_hbm: nl.NkiTensor,
    topk_tiled: bool = False,
) -> None:
    """Validate all inputs to the sparse MLA latent + RoPE attention kernel (kernel A).

    Performs comprehensive validation of tensor shapes, dtypes, and the topk-index
    layout before kernel execution begins. Ensures:
      - q_lift / q_pe / c_kv / k_pe are bf16 with matching latent (L) and RoPE (R) dims.
      - The single-batch, single-latent-tile (L == 512) constraints hold.
      - H satisfies the transpose xbar-alignment constraint (multiple of 16, <= P_MAX).
      - The topk index tensor matches the selected layout and yields a legal K.

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

    # ---- topk indices layout ----
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


def _validate_mla_vupmx_oproj_inputs(
    out_attn_hbm: nl.NkiTensor,
    wuv_qtz_hbm: nl.NkiTensor,
    wuv_scale_hbm: nl.NkiTensor,
    wo_qtz_hbm: nl.NkiTensor,
    wo_scale_hbm: nl.NkiTensor,
) -> None:
    """Validate all inputs to the MX V-up + MX o_proj kernel (kernel B).

    Performs comprehensive validation of tensor shapes and dtypes before kernel
    execution begins. Ensures:
      - The latent attention input is bf16 with a single-batch, L==512 layout.
      - MX V-up / o_proj weights are fp8x4-packed with correct shapes.
      - Compact block-128 scales have the expected shapes.
      - Dimension parameters (d_v, H, Hdv, HID) satisfy hardware constraints.

    H and L are recovered from tensor shapes (L fixed at 512 = P_MAX * H_PACK):
    wuv_qtz_hbm is [H*L // 4, d_v]. H is NOT a runtime scalar.

    Raises:
        AssertionError: If any validation check fails.
    """
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

    # ---- wuv scale (compact block-128): [H*L // 128, ceil(d_v/128)] ----
    expected_wuv_scale_shape = (HL // _DS_SCALE_BLOCK, div_ceil(d_v, _DS_SCALE_BLOCK))
    kernel_assert(
        wuv_scale_hbm.shape == expected_wuv_scale_shape,
        f"{_PREFIX} wuv_scale_hbm shape must be [H*L//128, ceil(d_v/128)] = "
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

    # ---- wo scale (compact block-128): [H*d_v // 128, ceil(HID/128)] ----
    expected_wo_scale_shape = (Hdv // _DS_SCALE_BLOCK, div_ceil(HID, _DS_SCALE_BLOCK))
    kernel_assert(
        wo_scale_hbm.shape == expected_wo_scale_shape,
        f"{_PREFIX} wo_scale_hbm shape must be [H*d_v//128, ceil(HID/128)] = "
        f"{expected_wo_scale_shape}, got {wo_scale_hbm.shape}.",
    )
    kernel_assert(
        wo_scale_hbm.dtype == nl.uint8,
        f"{_PREFIX} wo_scale_hbm must have dtype uint8 (compact block-128 scale), got {wo_scale_hbm.dtype}.",
    )
