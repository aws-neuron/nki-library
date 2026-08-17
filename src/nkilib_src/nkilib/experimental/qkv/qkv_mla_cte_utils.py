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

"""Shared helpers for MLA QKV CTE kernels (v32 and v4).

Contains:

- Input validators with capacity/shape assertions for both kernels.
- MX-format weight/scale loading (full prologue load + K-slab streaming).
- MX matmul (single output, split output, and K-range body for slab loops).
- Quantization, RMSNorm primitives, RoPE primitives, and the pre-swizzled
  transpose used by both kernels' second-stage matmuls.
- SBUF cost models that drive each kernel's dispatch:
  ``_v32_compute_k_slab_size_512_tiles`` for v32's K-slabbing of wqkv_a,
  ``_v4_compute_heads_per_chunk`` for v4's N-chunking of wq_b.
- Post-projection helpers (``_v4_apply_*``) shared between v4's fast and
  chunked paths.
"""

from typing import List, Tuple

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ...core.qkv.qkv_cte import _get_psum_bank_size
from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil

NUM_HW_PSUM_BANKS = 8
H_PACK = 4


def _validate_mla_inputs(
    x_hbm: nl.NkiTensor,
    wqkv_a_hbm: nl.NkiTensor,
    wqkv_a_scale_hbm: nl.NkiTensor,
    wq_b_hbm: nl.NkiTensor,
    wq_b_scale_hbm: nl.NkiTensor,
    q_norm_gamma_hbm: nl.NkiTensor,
    wkv_b_hbm: nl.NkiTensor,
    wkv_b_scale_hbm: nl.NkiTensor,
    kv_norm_gamma_hbm: nl.NkiTensor,
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    norm_eps: float,
) -> None:
    """
    Validate all inputs to the MLA QKV kernel.

    Checks tensor shapes, dimension divisibility requirements, and consistency
    between dimension parameters and weight shapes.

    Args:
        All parameters match the qkv_mla_mx kernel signature.

    Raises:
        AssertionError: If any validation check fails with descriptive error message.
    """
    P_MAX = nl.tile_size.pmax
    H_PACK = 4
    MX_WEIGHT_PACK = 4
    DS_SCALE_BLOCK = 128  # DeepSeek block-128 scales

    B, S, H = x_hbm.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    qkv_a_out_dim = qk_lora_rank + kv_lora_rank + qk_rope_head_dim
    q_out_dim = n_heads * qk_head_dim
    kv_b_out_dim = n_heads * (qk_nope_head_dim + v_head_dim)

    qkv_a_n_blocks = (qkv_a_out_dim + DS_SCALE_BLOCK - 1) // DS_SCALE_BLOCK
    q_out_n_blocks = (q_out_dim + DS_SCALE_BLOCK - 1) // DS_SCALE_BLOCK
    kv_b_out_n_blocks = (kv_b_out_dim + DS_SCALE_BLOCK - 1) // DS_SCALE_BLOCK

    # Input shape
    kernel_assert(
        len(x_hbm.shape) == 3,
        f"[QKV MLA Kernel] x_hbm must be 3D [B, S, H], got shape {x_hbm.shape}",
    )

    # H divisibility: must be divisible by P_MAX * H_PACK = 512
    kernel_assert(
        H % (P_MAX * H_PACK) == 0,
        f"[QKV MLA Kernel] H must be divisible by {P_MAX * H_PACK} (P_MAX * H_PACK), got {H=}",
    )

    # Lora rank divisibility: must be divisible by P_MAX * H_PACK = 512
    kernel_assert(
        qk_lora_rank % (P_MAX * H_PACK) == 0,
        f"[QKV MLA Kernel] qk_lora_rank must be divisible by {P_MAX * H_PACK}, got {qk_lora_rank=}",
    )
    kernel_assert(
        kv_lora_rank % (P_MAX * H_PACK) == 0,
        f"[QKV MLA Kernel] kv_lora_rank must be divisible by {P_MAX * H_PACK}, got {kv_lora_rank=}",
    )

    # RoPE head dim must be even (for half-dim split)
    kernel_assert(
        qk_rope_head_dim % 2 == 0,
        f"[QKV MLA Kernel] qk_rope_head_dim must be even for RoPE half-dim split, got {qk_rope_head_dim=}",
    )

    # Fused first-stage weight shape: [H // MX_WEIGHT_PACK, qk_lora_rank + kv_lora_rank + qk_rope_head_dim]
    kernel_assert(
        wqkv_a_hbm.shape == (H // MX_WEIGHT_PACK, qkv_a_out_dim),
        f"[QKV MLA Kernel] wqkv_a_hbm shape must be ({H // MX_WEIGHT_PACK}, {qkv_a_out_dim}), got {wqkv_a_hbm.shape}",
    )
    kernel_assert(
        wqkv_a_scale_hbm.shape == (H // DS_SCALE_BLOCK, qkv_a_n_blocks),
        f"[QKV MLA Kernel] wqkv_a_scale_hbm shape must be ({H // DS_SCALE_BLOCK}, {qkv_a_n_blocks})"
        f" (compact DeepSeek block-128 scales), got {wqkv_a_scale_hbm.shape}",
    )

    # Second-stage Q weight shape: [qk_lora_rank // MX_WEIGHT_PACK, n_heads * qk_head_dim]
    kernel_assert(
        wq_b_hbm.shape == (qk_lora_rank // MX_WEIGHT_PACK, q_out_dim),
        f"[QKV MLA Kernel] wq_b_hbm shape must be ({qk_lora_rank // MX_WEIGHT_PACK}, {q_out_dim}),"
        f" got {wq_b_hbm.shape}",
    )
    kernel_assert(
        wq_b_scale_hbm.shape == (qk_lora_rank // DS_SCALE_BLOCK, q_out_n_blocks),
        f"[QKV MLA Kernel] wq_b_scale_hbm shape must be ({qk_lora_rank // DS_SCALE_BLOCK}, {q_out_n_blocks})"
        f" (compact DeepSeek block-128 scales), got {wq_b_scale_hbm.shape}",
    )

    # Q norm gamma shape: [1, qk_lora_rank]
    kernel_assert(
        q_norm_gamma_hbm.shape == (1, qk_lora_rank),
        f"[QKV MLA Kernel] q_norm_gamma_hbm shape must be (1, {qk_lora_rank}), got {q_norm_gamma_hbm.shape}",
    )

    # Second-stage KV weight shape: [kv_lora_rank // MX_WEIGHT_PACK, n_heads * (qk_nope_head_dim + v_head_dim)]
    kernel_assert(
        wkv_b_hbm.shape == (kv_lora_rank // MX_WEIGHT_PACK, kv_b_out_dim),
        f"[QKV MLA Kernel] wkv_b_hbm shape must be ({kv_lora_rank // MX_WEIGHT_PACK}, {kv_b_out_dim}),"
        f" got {wkv_b_hbm.shape}",
    )
    kernel_assert(
        wkv_b_scale_hbm.shape == (kv_lora_rank // DS_SCALE_BLOCK, kv_b_out_n_blocks),
        f"[QKV MLA Kernel] wkv_b_scale_hbm shape must be ({kv_lora_rank // DS_SCALE_BLOCK}, {kv_b_out_n_blocks})"
        f" (compact DeepSeek block-128 scales), got {wkv_b_scale_hbm.shape}",
    )

    # KV norm gamma shape: [1, kv_lora_rank]
    kernel_assert(
        kv_norm_gamma_hbm.shape == (1, kv_lora_rank),
        f"[QKV MLA Kernel] kv_norm_gamma_hbm shape must be (1, {kv_lora_rank}), got {kv_norm_gamma_hbm.shape}",
    )

    # RoPE cache shapes: [B, S, qk_rope_head_dim]
    kernel_assert(
        cos_cache_hbm.shape == (B, S, qk_rope_head_dim),
        f"[QKV MLA Kernel] cos_cache_hbm shape must be ({B}, {S}, {qk_rope_head_dim}), got {cos_cache_hbm.shape}",
    )
    kernel_assert(
        sin_cache_hbm.shape == (B, S, qk_rope_head_dim),
        f"[QKV MLA Kernel] sin_cache_hbm shape must be ({B}, {S}, {qk_rope_head_dim}), got {sin_cache_hbm.shape}",
    )

    # Epsilon must be positive
    kernel_assert(
        norm_eps > 0,
        f"[QKV MLA Kernel] norm_eps must be positive, got {norm_eps=}",
    )

    # Capacity ceilings derived empirically from the v32 test matrix.
    # The supported envelope is non-monotonic in n_heads: per-tile working set
    # grows with n_heads (q_tile + k_tile + v_tile + matmul outputs + transposes)
    # and v32 has no N-tile dispatch to compensate; the K-slab dispatch reduces
    # prologue weights enough that some larger-n_heads configs still fit.
    # Validated combinations (n_heads -> max H): 1->10240, 2->16384, 3->9216,
    # 8->7168. n_heads in {4,5,6,7} OOMs at standard H.
    if n_heads == 1:
        _v32_max_h = 10240
        _v32_n_heads_ok = True
    elif n_heads == 2:
        _v32_max_h = 16384
        _v32_n_heads_ok = True
    elif n_heads == 3:
        _v32_max_h = 9216
        _v32_n_heads_ok = True
    elif n_heads == 8:
        _v32_max_h = 7168
        _v32_n_heads_ok = True
    else:
        _v32_max_h = 0
        _v32_n_heads_ok = False
    kernel_assert(
        _v32_n_heads_ok,
        f"[QKV MLA Kernel] n_heads={n_heads} is not in the validated set "
        f"(1, 2, 3, 8). Per-tile working set scales with n_heads; n_heads in "
        f"(4, 5, 6, 7) overflows SBUF on standard head dimensions and the "
        f"kernel has no N-tile dispatch to compensate. n_heads=8 happens to "
        f"pass only because K-slab dispatch saves enough prologue space at "
        f"H<=7168. To extend support, port the v4 head-chunk dispatch to v32, "
        f"validate, then update this allow-list.",
    )
    kernel_assert(
        H <= _v32_max_h,
        f"[QKV MLA Kernel] H={H} exceeds the largest validated H for "
        f"n_heads={n_heads} ({_v32_max_h}). Larger H increases the wqkv_a slab "
        f"footprint and input buffer cost.",
    )


def _validate_mla_v4_inputs(
    x_hbm: nl.NkiTensor,
    wqkv_hbm: nl.NkiTensor,
    wqkv_scale_hbm: nl.NkiTensor,
    wq_b_hbm: nl.NkiTensor,
    wq_b_scale_hbm: nl.NkiTensor,
    q_norm_gamma_hbm: nl.NkiTensor,
    kv_norm_gamma_hbm: nl.NkiTensor,
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    norm_eps: float,
) -> None:
    """
    Validate all inputs to the v4 MLA QKV kernel.

    Checks tensor shapes, dimension divisibility requirements, and consistency
    between dimension parameters and weight shapes for the fused-projection v4
    variant (no wkv_b, kv latent passed through directly).

    Args:
        All parameters match the qkv_mla_v4_mx kernel signature.

    Raises:
        AssertionError: If any validation check fails with descriptive error message.
    """
    P_MAX = nl.tile_size.pmax
    H_PACK = 4
    MX_WEIGHT_PACK = 4
    DS_SCALE_BLOCK = 128  # DeepSeek block-128 scales

    B, S, H = x_hbm.shape
    kv_dim = kv_lora_rank + qk_rope_head_dim
    qkv_out_dim = qk_lora_rank + kv_dim
    q_out_dim = n_heads * head_dim
    qkv_n_blocks = (qkv_out_dim + DS_SCALE_BLOCK - 1) // DS_SCALE_BLOCK
    q_out_n_blocks = (q_out_dim + DS_SCALE_BLOCK - 1) // DS_SCALE_BLOCK

    # Input shape
    kernel_assert(
        len(x_hbm.shape) == 3,
        f"[QKV MLA v4 Kernel] x_hbm must be 3D [B, S, H], got shape {x_hbm.shape}",
    )

    # H divisibility: must be divisible by P_MAX * H_PACK = 512
    kernel_assert(
        H % (P_MAX * H_PACK) == 0,
        f"[QKV MLA v4 Kernel] H must be divisible by {P_MAX * H_PACK} (P_MAX * H_PACK), got {H=}",
    )

    # Lora rank divisibility: qk_lora_rank must be divisible by P_MAX * H_PACK = 512
    kernel_assert(
        qk_lora_rank % (P_MAX * H_PACK) == 0,
        f"[QKV MLA v4 Kernel] qk_lora_rank must be divisible by {P_MAX * H_PACK}, got {qk_lora_rank=}",
    )

    # head_dim must exceed qk_rope_head_dim (nope portion = head_dim - rope_dim)
    kernel_assert(
        head_dim > qk_rope_head_dim,
        f"[QKV MLA v4 Kernel] head_dim ({head_dim}) must exceed qk_rope_head_dim ({qk_rope_head_dim})",
    )

    # RoPE head dim must be even (for half-dim split)
    kernel_assert(
        qk_rope_head_dim % 2 == 0,
        f"[QKV MLA v4 Kernel] qk_rope_head_dim must be even for RoPE half-dim split, got {qk_rope_head_dim=}",
    )

    # Fused first-stage weight shape: [H // MX_WEIGHT_PACK, qk_lora_rank + kv_dim]
    kernel_assert(
        wqkv_hbm.shape == (H // MX_WEIGHT_PACK, qkv_out_dim),
        f"[QKV MLA v4 Kernel] wqkv_hbm shape must be ({H // MX_WEIGHT_PACK}, {qkv_out_dim}), got {wqkv_hbm.shape}",
    )
    kernel_assert(
        wqkv_scale_hbm.shape == (H // DS_SCALE_BLOCK, qkv_n_blocks),
        f"[QKV MLA v4 Kernel] wqkv_scale_hbm shape must be ({H // DS_SCALE_BLOCK}, {qkv_n_blocks})"
        f" (compact DeepSeek block-128 scales), got {wqkv_scale_hbm.shape}",
    )

    # Second-stage Q weight shape: [qk_lora_rank // MX_WEIGHT_PACK, n_heads * head_dim]
    kernel_assert(
        wq_b_hbm.shape == (qk_lora_rank // MX_WEIGHT_PACK, q_out_dim),
        f"[QKV MLA v4 Kernel] wq_b_hbm shape must be ({qk_lora_rank // MX_WEIGHT_PACK}, {q_out_dim}),"
        f" got {wq_b_hbm.shape}",
    )
    kernel_assert(
        wq_b_scale_hbm.shape == (qk_lora_rank // DS_SCALE_BLOCK, q_out_n_blocks),
        f"[QKV MLA v4 Kernel] wq_b_scale_hbm shape must be ({qk_lora_rank // DS_SCALE_BLOCK}, {q_out_n_blocks})"
        f" (compact DeepSeek block-128 scales), got {wq_b_scale_hbm.shape}",
    )

    # Q norm gamma shape: [1, qk_lora_rank]
    kernel_assert(
        q_norm_gamma_hbm.shape == (1, qk_lora_rank),
        f"[QKV MLA v4 Kernel] q_norm_gamma_hbm shape must be (1, {qk_lora_rank}), got {q_norm_gamma_hbm.shape}",
    )

    # KV norm gamma shape: [1, kv_dim] (norm applied to full kv latent + rope tail)
    kernel_assert(
        kv_norm_gamma_hbm.shape == (1, kv_dim),
        f"[QKV MLA v4 Kernel] kv_norm_gamma_hbm shape must be (1, {kv_dim}), got {kv_norm_gamma_hbm.shape}",
    )

    # RoPE cache shapes: [B, S, qk_rope_head_dim]
    kernel_assert(
        cos_cache_hbm.shape == (B, S, qk_rope_head_dim),
        f"[QKV MLA v4 Kernel] cos_cache_hbm shape must be ({B}, {S}, {qk_rope_head_dim}), got {cos_cache_hbm.shape}",
    )
    kernel_assert(
        sin_cache_hbm.shape == (B, S, qk_rope_head_dim),
        f"[QKV MLA v4 Kernel] sin_cache_hbm shape must be ({B}, {S}, {qk_rope_head_dim}), got {sin_cache_hbm.shape}",
    )

    # Epsilon must be positive
    kernel_assert(
        norm_eps > 0,
        f"[QKV MLA v4 Kernel] norm_eps must be positive, got {norm_eps=}",
    )

    # Capacity ceilings derived empirically from the v4 test matrix.
    # v4 has the head-chunk dispatch (handles n_heads pressure) AND K-slab
    # dispatch on the fused first projection. Ceilings represent validated
    # values; configurations beyond may work but are untested.
    _V4_MAX_N_HEADS_SUPPORTED = 8
    _V4_MAX_HIDDEN_DIM_SUPPORTED = 16384
    _V4_MAX_HEAD_DIM_SUPPORTED = 512
    _V4_MAX_QK_LORA_RANK_SUPPORTED = 2048
    _V4_MAX_KV_LORA_RANK_SUPPORTED = 1024
    kernel_assert(
        n_heads <= _V4_MAX_N_HEADS_SUPPORTED,
        f"[QKV MLA v4 Kernel] n_heads={n_heads} exceeds the largest validated value ({_V4_MAX_N_HEADS_SUPPORTED}).",
    )
    kernel_assert(
        H <= _V4_MAX_HIDDEN_DIM_SUPPORTED,
        f"[QKV MLA v4 Kernel] H={H} exceeds the largest validated value ({_V4_MAX_HIDDEN_DIM_SUPPORTED}).",
    )
    kernel_assert(
        head_dim <= _V4_MAX_HEAD_DIM_SUPPORTED,
        f"[QKV MLA v4 Kernel] head_dim={head_dim} exceeds the largest validated value ({_V4_MAX_HEAD_DIM_SUPPORTED}).",
    )
    kernel_assert(
        qk_lora_rank <= _V4_MAX_QK_LORA_RANK_SUPPORTED,
        f"[QKV MLA v4 Kernel] qk_lora_rank={qk_lora_rank} exceeds the largest "
        f"validated value ({_V4_MAX_QK_LORA_RANK_SUPPORTED}).",
    )
    kernel_assert(
        kv_lora_rank <= _V4_MAX_KV_LORA_RANK_SUPPORTED,
        f"[QKV MLA v4 Kernel] kv_lora_rank={kv_lora_rank} exceeds the largest "
        f"validated value ({_V4_MAX_KV_LORA_RANK_SUPPORTED}).",
    )


def _v32_compute_k_slab_size_512_tiles(
    H: int,
    qkv_out_dim: int,
    q_out_dim: int,
    kv_b_out_dim: int,
    qk_lora_rank: int,
    kv_lora_rank: int,
    n_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    qk_rope_head_dim: int,
    sbuf_budget_bytes: int,
    num_slab_buffers: int,
) -> int:
    """Pick the largest slab size (in 512-tiles, dividing H_512) that fits SBUF.

    Mirrors the qkv_cte look-ahead pattern used elsewhere in this package: pure-Python
    cost model that conservatively estimates per-partition SBUF usage of the v32 MLA
    kernel, varying the number of K (H) 512-tiles loaded simultaneously for
    ``wqkv_a``/``wqkv_a_scale``. Returns the largest divisor of ``H_512`` such that
    ``fixed + num_slab_buffers * slab_cost <= sbuf_budget_bytes``. Returns ``H_512``
    when the fast (single-slab, prologue-loaded) path fits, so existing configs see
    no perf hit.

    Args:
        H: Hidden dimension of input.
        qkv_out_dim: Combined out width of the first projection (qk_lora + kv_lora + rope).
        q_out_dim: ``n_heads * qk_head_dim`` (second-stage Q output width).
        kv_b_out_dim: ``n_heads * (qk_nope + v_head_dim)`` (second-stage KV output width).
        qk_lora_rank, kv_lora_rank: LoRA ranks for Q and KV.
        n_heads: Number of attention heads.
        qk_head_dim, v_head_dim: Head dimensions for Q/K and V.
        qk_rope_head_dim: RoPE portion of qk_head_dim.
        sbuf_budget_bytes: Per-partition SBUF capacity available to the kernel.
        num_slab_buffers: 1 for single-buffered slabs, 2 for double-buffered (prefetch).

    Returns:
        Slab size in 512-tiles, in [1, H_512] and dividing H_512.
    """
    P_MAX = nl.tile_size.pmax
    H_PACK_LOCAL = 4
    H_512 = H // (P_MAX * H_PACK_LOCAL)

    fp8x4 = sizeinbytes(nl.float8_e4m3fn_x4)
    u8 = sizeinbytes(nl.uint8)
    bf16 = sizeinbytes(nl.bfloat16)
    f32 = sizeinbytes(nl.float32)

    qk_lora_512 = qk_lora_rank // (P_MAX * H_PACK_LOCAL)
    kv_lora_512 = kv_lora_rank // (P_MAX * H_PACK_LOCAL)

    # Tile-scope intermediates (qr_sb, transposes, quants, q/k/v assembly, etc.)
    # all live within nested SBUF scopes that share storage with the slab buffer
    # via the SbufManager stack. The peak SBUF in the FAST path is empirically
    # bounded by:
    #   prologue (norms + eps + zero + wq_b + wkv_b + 2x x_sb + shfl + 2x x_qtz)
    #   + wqkv_a + wqkv_a_scale (the streamable weights)
    #   + a constant per-tile working set that's the same in fast and slabbed paths.
    # We model it as: prologue_fixed + slab_cost + per_tile_const.
    # The slabbed path reduces the slab cost (proportional to slab_size_512); the
    # per_tile_const is identical between paths, so this comparison is what we want.
    # Scale buffers are padded to whole 128-column blocks on N (compact
    # block-128 scales materialize to ``ceil(N/128) * 128``).
    SCALE_BLOCK = 128
    q_out_dim_padded = ((q_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    kv_b_out_dim_padded = ((kv_b_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK

    prologue_fixed = 0
    prologue_fixed += qk_lora_512 * H_PACK_LOCAL * f32
    prologue_fixed += kv_lora_512 * H_PACK_LOCAL * f32
    prologue_fixed += 2 * bf16
    # wq_b/wkv_b weights + materialized scales (DMA-broadcast directly).
    prologue_fixed += qk_lora_512 * q_out_dim * fp8x4 + qk_lora_512 * q_out_dim_padded * u8
    prologue_fixed += kv_lora_512 * kv_b_out_dim * fp8x4 + kv_lora_512 * kv_b_out_dim_padded * u8
    prologue_fixed += 2 * H * bf16
    prologue_fixed += H_512 * P_MAX * H_PACK_LOCAL * bf16
    prologue_fixed += 2 * H_512 * P_MAX * (fp8x4 + u8)

    # Per-tile working set that exists at peak (qr_sb + kv_raw + qr_transpose +
    # qr_qtz/scale + kv_transpose + kv_qtz/scale + q/k/v assembly buffers +
    # RoPE scratch). Conservatively estimated; same in both paths.
    per_tile_const = 0
    per_tile_const += qk_lora_rank * bf16
    per_tile_const += (kv_lora_rank + qk_rope_head_dim) * bf16
    per_tile_const += qk_lora_512 * P_MAX * H_PACK_LOCAL * bf16
    per_tile_const += qk_lora_512 * P_MAX * (fp8x4 + u8)
    per_tile_const += kv_lora_512 * P_MAX * H_PACK_LOCAL * bf16
    per_tile_const += kv_lora_512 * P_MAX * (fp8x4 + u8)
    per_tile_const += q_out_dim * bf16
    per_tile_const += kv_b_out_dim * bf16
    per_tile_const += n_heads * qk_head_dim * bf16  # q_tile_sb
    per_tile_const += n_heads * qk_head_dim * bf16  # k_tile_sb
    per_tile_const += n_heads * v_head_dim * bf16  # v_tile_sb
    per_tile_const += qk_rope_head_dim * 4 * bf16  # rope scratch + cos/sin

    # Slab cost: weights + materialized scales (DMA-broadcast directly).
    slab_byte_per_512_tile = qkv_out_dim * fp8x4 + qkv_out_dim_padded * u8

    for divisor in range(1, H_512 + 1):
        if H_512 % divisor != 0:
            continue
        slab_size_512 = H_512 // divisor
        total = prologue_fixed + num_slab_buffers * slab_size_512 * slab_byte_per_512_tile + per_tile_const
        if total <= sbuf_budget_bytes:
            return slab_size_512
    return 1


def _load_mx_weights_k_slab(
    weights_hbm: nl.NkiTensor,
    scales_hbm: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    scales_sb: nl.NkiTensor,
    in_dim_full: int,
    out_dim: int,
    k_tile_start: int,
    k_tile_count: int,
    sbm: SbufManager,
    name: str = "slab",
) -> None:
    """Load a K-slab ``[k_tile_start : k_tile_start + k_tile_count]`` of MX weights.

    Companion to :func:`_load_mx_weights` for K-streaming. Same two-stage load
    used to materialize the full MX scale layout: compact DMA from HBM into an
    inner-scope scratch (no free-dim broadcast at DMA time, since the engine
    falls back to per-element when stride-0 hits the free dim), then a
    vector-engine ``tensor_copy`` with stride-0 source broadcast to expand
    into ``scales_sb``. The compact scratch is freed before the function
    returns so callers see no extra SBUF pressure.

    Args:
        weights_hbm: ``[in_dim_full // 4, out_dim]`` fp8x4 weights on HBM.
        scales_hbm: ``[in_dim_full // 128, ceil(out_dim / 128)]`` uint8 compact
            block-128 scales on HBM.
        weights_sb: ``[P_MAX, k_tile_count, out_dim]`` slab destination.
        scales_sb: ``[P_MAX, k_tile_count, ceil(out_dim/128) * 128]`` slab
            full-MX scales destination consumed by ``nc_matmul_mx``.
        in_dim_full: Full K dimension of the HBM tensor.
        out_dim: N dimension.
        k_tile_start: Index of the first 512-tile in this slab.
        k_tile_count: Number of 512-tiles in this slab.
        sbm: SBUF memory manager used for the inner-scope compact scratch.
    """
    P_MAX = nl.tile_size.pmax
    SCALE_P_PER_QUAD = 4
    SCALE_BLOCK = 128
    QUADS_PER_TILE = 4

    in_dim_packed = in_dim_full // 4
    n_blocks = (out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK

    for h_local in nl.affine_range(k_tile_count):
        h_global = k_tile_start + h_local
        h_tile_sz = min(P_MAX, in_dim_packed - h_global * P_MAX)
        nisa.dma_copy(
            dst=weights_sb[0:h_tile_sz, h_local, 0:out_dim],
            src=weights_hbm.ap(
                pattern=[[out_dim, h_tile_sz], [1, out_dim]],
                offset=h_global * P_MAX * out_dim,
                dtype=nl.float8_e4m3fn_x4,
            ),
            dge_mode=dge_mode.swdge,
        )

    # ---- Stage 1: DMA compact scales (no free-dim broadcast at DMA time). ----
    sbm.open_scope()
    compact_scales_sb = sbm.alloc_stack(
        (P_MAX, k_tile_count, n_blocks),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name}_scales_compact",
    )
    for quad_idx in nl.affine_range(QUADS_PER_TILE):
        slab_start_compact_row = k_tile_start * QUADS_PER_TILE + quad_idx
        nisa.dma_copy(
            dst=compact_scales_sb[nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), 0:k_tile_count, 0:n_blocks],
            src=scales_hbm.ap(
                pattern=[
                    [0, SCALE_P_PER_QUAD],
                    [QUADS_PER_TILE * n_blocks, k_tile_count],
                    [1, n_blocks],
                ],
                offset=slab_start_compact_row * n_blocks,
                dtype=nl.uint8,
            ),
            dge_mode=dge_mode.swdge,
        )

    # ---- Stage 2: Vector-engine broadcast into the materialized scales_sb. ----
    src_view = compact_scales_sb.expand_dim(dim=3).broadcast(dim=3, size=SCALE_BLOCK)
    dst_view = scales_sb.reshape_dim(dim=2, shape=(n_blocks, SCALE_BLOCK))
    nisa.tensor_copy(dst=dst_view, src=src_view)

    sbm.close_scope()  # Frees compact_scales_sb.


def _mx_matmul_split_k_range(
    input_qtz_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    weights_slab_sb: nl.NkiTensor,
    weights_slab_scale_sb: nl.NkiTensor,
    k_tile_start: int,
    k_tile_count: int,
    m_dim: int,
    n_dim: int,
    output_psum: list,
) -> None:
    """Run nc_matmul_mx for one K-slab, accumulating into the caller-provided PSUM banks.

    Mirrors the matmul body of :func:`_mx_matmul_split` restricted to a K range.
    Reads input data/scale from indices ``[k_tile_start : k_tile_start + k_tile_count]``
    of the kernel-wide quantized input, and weight data/scale from the slab-local
    indices ``[0 : k_tile_count]``. PSUM accumulates implicitly across calls because
    each call writes to the same destination banks.

    Args:
        input_qtz_sb: ``[P_MAX, num_k_tiles_full, m_dim]`` fp8x4 stationary input
            (kernel-wide; this call slices ``[k_tile_start : k_tile_start + k_tile_count]``).
        input_scale_sb: ``[P_MAX, num_k_tiles_full, m_dim]`` uint8 stationary scale.
        weights_slab_sb: ``[P_MAX, k_tile_count, n_dim]`` fp8x4 slab weights (slab-local indexing).
        weights_slab_scale_sb: ``[P_MAX, k_tile_count, n_dim]`` uint8 slab weight scales.
        k_tile_start: Slab's starting global K-tile index (used to index input).
        k_tile_count: Number of K-tiles in this slab.
        m_dim: M dimension of the matmul (typically s_tile_pad).
        n_dim: N dimension (full output width).
        output_psum: List of PSUM bank ndarrays, one per N-tile. Caller allocates and
            issues the post-loop PSUM->SBUF copy.
    """
    P_MAX = nl.tile_size.pmax
    F_MAX = 512
    num_n_tiles = div_ceil(n_dim, F_MAX)

    for k_local in nl.affine_range(k_tile_count):
        k_global = k_tile_start + k_local
        for i_n_tile in nl.affine_range(num_n_tiles):
            n_tile_sz = min(F_MAX, n_dim - i_n_tile * F_MAX)
            nisa.nc_matmul_mx(
                dst=output_psum[i_n_tile][0:m_dim, 0:n_tile_sz],
                stationary=input_qtz_sb[0:P_MAX, k_global, nl.ds(0, m_dim)],
                moving=weights_slab_sb[0:P_MAX, k_local, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
                stationary_scale=input_scale_sb[0:P_MAX, k_global, nl.ds(0, m_dim)],
                moving_scale=weights_slab_scale_sb[0:P_MAX, k_local, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
            )


def _v4_per_chunk_bytes(
    heads_per_chunk: int,
    head_dim: int,
    qk_rope_head_dim: int,
    qk_lora_512_tiles: int,
    fp8x4: int,
    u8: int,
    bf16: int,
    f32: int,
) -> int:
    """Bytes-per-partition that scale with ``heads_per_chunk`` in v4 stage 2."""
    SCALE_BLOCK = 128
    chunk_q_out_dim = heads_per_chunk * head_dim
    chunk_q_out_dim_padded = ((chunk_q_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    # wq_b weights + materialized scales for the chunk.
    per_chunk = qk_lora_512_tiles * chunk_q_out_dim * fp8x4
    per_chunk += qk_lora_512_tiles * chunk_q_out_dim_padded * u8
    # q_sb (matmul output for the chunk)
    per_chunk += chunk_q_out_dim * bf16
    # rope_tmp_sb scales with heads_per_chunk
    per_chunk += heads_per_chunk * qk_rope_head_dim * bf16
    # sum_sq_sb (negligible) + act_temp scratch
    per_chunk += heads_per_chunk * f32 + 1 * f32
    return per_chunk


def _v4_compute_heads_per_chunk(
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    qk_lora_rank: int,
    kv_lora_rank: int,
    H: int,
    sbuf_budget_bytes: int,
) -> int:
    """Pick the largest divisor of ``n_heads`` whose worst-case live SBUF fits.

    Mirrors the qkv_cte.py "look-ahead" pattern: pre-computes the per-partition
    bytes of every tensor live across the post-fused-projection portion of
    ``qkv_mla_v4_mx`` for a given ``heads_per_chunk``, and returns the largest
    chunk size (dividing ``n_heads``) such that the projected total fits the
    budget. Returns ``n_heads`` when the fast (single-chunk) path fits, so
    callers see no perf hit on configs that already work today.

    The cost model accounts only for terms that scale with ``heads_per_chunk``;
    a fixed overhead term covers tensors invariant under chunking. Both terms
    are conservative upper bounds, so a "fits" decision is safe.

    Args:
        n_heads: Number of attention heads.
        head_dim: Q/K head dimension (nope + rope).
        qk_rope_head_dim: RoPE portion of head_dim.
        qk_lora_rank: Q latent dimension (after first projection).
        kv_lora_rank: KV latent dimension.
        H: Hidden dimension of input.
        sbuf_budget_bytes: Per-partition SBUF capacity available to the kernel.

    Returns:
        Heads-per-chunk in [1, n_heads], dividing n_heads.
    """
    P_MAX = nl.tile_size.pmax
    H_PACK_LOCAL = 4

    qk_lora_512_tiles = qk_lora_rank // (P_MAX * H_PACK_LOCAL)
    fp8x4 = sizeinbytes(nl.float8_e4m3fn_x4)
    u8 = sizeinbytes(nl.uint8)
    bf16 = sizeinbytes(nl.bfloat16)
    f32 = sizeinbytes(nl.float32)

    # ---- Per-partition costs invariant under chunking ----
    kv_dim = kv_lora_rank + qk_rope_head_dim
    qkv_out_dim = qk_lora_rank + kv_dim
    H_512_tiles = H // (P_MAX * H_PACK_LOCAL)

    SCALE_BLOCK = 128
    qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK

    fixed = 0
    # wqkv weights + materialized block-128 scales (fused first projection).
    fixed += H_512_tiles * qkv_out_dim * fp8x4
    fixed += H_512_tiles * qkv_out_dim_padded * u8
    # qr_sb + kv_sb (fused projection outputs)
    fixed += (qk_lora_rank + kv_dim) * bf16
    # qr_transposed + qr quantized + qr scale
    fixed += qk_lora_512_tiles * P_MAX * H_PACK_LOCAL * bf16
    fixed += qk_lora_512_tiles * P_MAX * (fp8x4 + u8)
    # q_norm gamma, kv gamma, eps/zero scratch, RoPE caches, kv rope scratch
    fixed += qk_lora_512_tiles * H_PACK_LOCAL * f32
    fixed += kv_dim * bf16
    fixed += 2 * bf16
    fixed += qk_rope_head_dim * bf16
    fixed += (qk_rope_head_dim // 2) * bf16
    fixed += qk_rope_head_dim * bf16
    # Double-buffered input + quantized input (2x each)
    fixed += 2 * H * bf16
    fixed += 2 * H_512_tiles * (fp8x4 + u8)
    # Slack for unaccounted scratch + alignment padding (~10%).
    fixed = (fixed * 11) // 10

    # Pick the largest divisor of n_heads that fits.
    for divisor in range(1, n_heads + 1):
        if n_heads % divisor != 0:
            continue
        heads_per_chunk = n_heads // divisor
        total = fixed + _v4_per_chunk_bytes(
            heads_per_chunk,
            head_dim,
            qk_rope_head_dim,
            qk_lora_512_tiles,
            fp8x4,
            u8,
            bf16,
            f32,
        )
        if total <= sbuf_budget_bytes:
            return heads_per_chunk
    return 1


def _v4_wqkv_bytes_at_slab(
    slab_size_512: int,
    qkv_out_dim: int,
    qkv_out_dim_padded: int,
    fp8x4: int,
    u8: int,
) -> int:
    """Per-partition bytes used by v4's fused first projection at a given K-slab size."""
    return slab_size_512 * qkv_out_dim * fp8x4 + slab_size_512 * qkv_out_dim_padded * u8


def _v4_fixed_bytes(
    qkv_first_proj_bytes: int,
    qk_lora_rank: int,
    kv_dim: int,
    qk_rope_head_dim: int,
    H: int,
    qk_lora_512_tiles: int,
    P_MAX: int,
    H_PACK_LOCAL: int,
    fp8x4: int,
    u8: int,
    bf16: int,
    f32: int,
    H_512_tiles: int,
) -> int:
    """Fixed (non-chunk) per-partition SBUF cost for v4, parameterized by the
    bytes used for the fused first projection (wqkv weights+scales).

    Used by both :func:`_v4_compute_heads_per_chunk` (which takes the full
    prologue cost) and :func:`_v4_compute_dispatch` (which substitutes a
    slab-sized cost when K-slabbing). Returns bytes including the ~10% slack.
    """
    fixed = 0
    # Caller-provided wqkv (full prologue load OR slabbed buffer).
    fixed += qkv_first_proj_bytes
    # qr_sb + kv_sb (fused projection outputs)
    fixed += (qk_lora_rank + kv_dim) * bf16
    # qr_transposed + qr quantized + qr scale
    fixed += qk_lora_512_tiles * P_MAX * H_PACK_LOCAL * bf16
    fixed += qk_lora_512_tiles * P_MAX * (fp8x4 + u8)
    # q_norm gamma, kv gamma, eps/zero scratch, RoPE caches, kv rope scratch
    fixed += qk_lora_512_tiles * H_PACK_LOCAL * f32
    fixed += kv_dim * bf16
    fixed += 2 * bf16
    fixed += qk_rope_head_dim * bf16
    fixed += (qk_rope_head_dim // 2) * bf16
    fixed += qk_rope_head_dim * bf16
    # Double-buffered input + quantized input (2x each)
    fixed += 2 * H * bf16
    fixed += 2 * H_512_tiles * (fp8x4 + u8)
    # Slack for unaccounted scratch + alignment padding (~10%).
    return (fixed * 11) // 10


def _v4_compute_dispatch(
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    qk_lora_rank: int,
    kv_lora_rank: int,
    H: int,
    sbuf_budget_bytes: int,
) -> Tuple[int, int]:
    """Unified v4 dispatch: pick (heads_per_chunk, k_slab_size_512) for SBUF fit.

    Searches a 4-mode space and returns the configuration that maximizes
    SBUF utilization without overflow (i.e., the lightest chunking/slabbing
    that fits):

    1. Fast path: ``heads_per_chunk = n_heads, k_slab_size = H_512`` (no chunking
       or slabbing, full prologue load of wqkv and full second matmul).
    2. N-only: chunk wq_b along N (heads), keep wqkv full prologue.
    3. K-only: slab wqkv along K, keep wq_b full second matmul.
    4. Both: chunk N AND slab K.

    Among configurations that fit, prefers (1) > (2)/(3) > (4) so the kernel
    only chunks/slabs when strictly necessary, preserving baseline perf for
    configs that fit at full size. When (2) and (3) both fit, picks the one
    with greater residual SBUF (i.e., larger budget left over) which means
    less aggressive chunking on the chosen axis.

    Args:
        n_heads, head_dim, qk_rope_head_dim, qk_lora_rank, kv_lora_rank, H:
            Same as the kernel parameters.
        sbuf_budget_bytes: Per-partition SBUF capacity available.

    Returns:
        Tuple ``(heads_per_chunk, k_slab_size_512)``. The kernel's USE_N_CHUNK
        flag is ``heads_per_chunk < n_heads``; USE_K_SLAB is
        ``k_slab_size_512 < H_512``.
    """
    P_MAX = nl.tile_size.pmax
    H_PACK_LOCAL = 4
    SCALE_BLOCK = 128

    fp8x4 = sizeinbytes(nl.float8_e4m3fn_x4)
    u8 = sizeinbytes(nl.uint8)
    bf16 = sizeinbytes(nl.bfloat16)
    f32 = sizeinbytes(nl.float32)

    qk_lora_512_tiles = qk_lora_rank // (P_MAX * H_PACK_LOCAL)
    kv_dim = kv_lora_rank + qk_rope_head_dim
    qkv_out_dim = qk_lora_rank + kv_dim
    qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    H_512_tiles = H // (P_MAX * H_PACK_LOCAL)

    full_wqkv_bytes = _v4_wqkv_bytes_at_slab(H_512_tiles, qkv_out_dim, qkv_out_dim_padded, fp8x4, u8)
    fixed_full = _v4_fixed_bytes(
        full_wqkv_bytes,
        qk_lora_rank,
        kv_dim,
        qk_rope_head_dim,
        H,
        qk_lora_512_tiles,
        P_MAX,
        H_PACK_LOCAL,
        fp8x4,
        u8,
        bf16,
        f32,
        H_512_tiles,
    )

    # ---- Mode 1: fast path (no chunking, no slabbing) ----
    full_chunk_bytes = _v4_per_chunk_bytes(
        n_heads,
        head_dim,
        qk_rope_head_dim,
        qk_lora_512_tiles,
        fp8x4,
        u8,
        bf16,
        f32,
    )
    if fixed_full + full_chunk_bytes <= sbuf_budget_bytes:
        return n_heads, H_512_tiles

    # ---- Mode 2: N-only (chunk heads, full wqkv prologue) ----
    # Walk hpc largest-first; first that fits wins.
    best_n_only_hpc = 0
    best_n_only_total = 0
    for divisor in range(2, n_heads + 1):
        if n_heads % divisor != 0:
            continue
        hpc = n_heads // divisor
        chunk_bytes = _v4_per_chunk_bytes(
            hpc,
            head_dim,
            qk_rope_head_dim,
            qk_lora_512_tiles,
            fp8x4,
            u8,
            bf16,
            f32,
        )
        total = fixed_full + chunk_bytes
        if total <= sbuf_budget_bytes:
            best_n_only_hpc = hpc
            best_n_only_total = total
            break

    # ---- Mode 3: K-only (slab wqkv, full N) ----
    best_k_only_slab = 0
    best_k_only_total = 0
    for divisor in range(2, H_512_tiles + 1):
        if H_512_tiles % divisor != 0:
            continue
        slab = H_512_tiles // divisor
        fixed_slab = _v4_fixed_bytes(
            _v4_wqkv_bytes_at_slab(slab, qkv_out_dim, qkv_out_dim_padded, fp8x4, u8),
            qk_lora_rank,
            kv_dim,
            qk_rope_head_dim,
            H,
            qk_lora_512_tiles,
            P_MAX,
            H_PACK_LOCAL,
            fp8x4,
            u8,
            bf16,
            f32,
            H_512_tiles,
        )
        total = fixed_slab + full_chunk_bytes
        if total <= sbuf_budget_bytes:
            best_k_only_slab = slab
            best_k_only_total = total
            break

    # Prefer the single-axis that fits with greater SBUF utilization
    # (= less aggressive chunking).
    if best_n_only_hpc != 0 and best_k_only_slab != 0:
        if best_n_only_total >= best_k_only_total:
            return best_n_only_hpc, H_512_tiles
        return n_heads, best_k_only_slab
    if best_n_only_hpc != 0:
        return best_n_only_hpc, H_512_tiles
    if best_k_only_slab != 0:
        return n_heads, best_k_only_slab

    # ---- Mode 4: both axes ----
    for n_div in range(2, n_heads + 1):
        if n_heads % n_div != 0:
            continue
        hpc = n_heads // n_div
        chunk_bytes = _v4_per_chunk_bytes(
            hpc,
            head_dim,
            qk_rope_head_dim,
            qk_lora_512_tiles,
            fp8x4,
            u8,
            bf16,
            f32,
        )
        for k_div in range(2, H_512_tiles + 1):
            if H_512_tiles % k_div != 0:
                continue
            slab = H_512_tiles // k_div
            fixed_slab = _v4_fixed_bytes(
                _v4_wqkv_bytes_at_slab(slab, qkv_out_dim, qkv_out_dim_padded, fp8x4, u8),
                qk_lora_rank,
                kv_dim,
                qk_rope_head_dim,
                H,
                qk_lora_512_tiles,
                P_MAX,
                H_PACK_LOCAL,
                fp8x4,
                u8,
                bf16,
                f32,
                H_512_tiles,
            )
            total = fixed_slab + chunk_bytes
            if total <= sbuf_budget_bytes:
                return hpc, slab

    # Fallback: most aggressive chunking on both axes. May still OOM at compile;
    # the validators' capacity ceilings should reject the config first in practice.
    return 1, 1


def _load_mx_weights(
    weights_hbm: nl.NkiTensor,
    scales_hbm: nl.NkiTensor,
    in_dim: int,
    out_dim: int,
    sbm: SbufManager,
    name: str = "mx",
    full_out_dim: int = None,
    out_col_offset: int = 0,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Load MX weights and DeepSeek-style block-128 scales from HBM to SBUF.

    Loads fp8x4 packed weights and per-(128 K x 128 N) block uint8 scales
    (DeepSeek V3.2 ``scale_fmt=ue8m0`` format). The scales tensor on HBM is
    compact: one byte per 128x128 weight block. On SBUF the scales are
    expanded into the MX hardware layout that ``nisa.nc_matmul_mx`` expects.

    Two-stage load:

    1. DMA compact scales to a small SBUF scratch ``[P_MAX, num_512_tiles,
       out_n_blocks]``. The DMA pattern uses stride-0 only on the partition
       axis (within a quadrant, broadcasting one source byte across 4 partition
       rows) — which the DMA engine natively supports. Free-dim stride-0 fan-out
       is NOT done here; the DMA engine doesn't support it and falls back to
       per-element copies, which serializes the prologue.
    2. Vector-engine ``tensor_copy`` materializes the broadcast layout from the
       compact scratch into the final ``scales_sb`` (``[P_MAX, num_512_tiles,
       padded_out_dim]``). Vector engine permits stride-0 source addressing,
       so the broadcast is one wide compute-engine op instead of a DMA fallback.

    The compact scratch lives in an inner SBUF scope and is released before the
    function returns, so callers see no extra SBUF pressure.

    Args:
        weights_hbm: ``[in_dim // 4, full_out_dim]`` fp8x4 weights on HBM.
        scales_hbm: ``[in_dim // 128, ceil(full_out_dim / 128)]`` uint8 compact
            block-128 scales on HBM.
        in_dim: K dimension of the matmul.
        out_dim: N dimension to ALLOCATE and load (slice width if slicing).
        sbm: SBUF memory manager.
        name: Name prefix for allocated buffers.
        full_out_dim: Full N dimension of the HBM tensor; defaults to ``out_dim``.
        out_col_offset: Column index of the slice start (multiple of 128).

    Returns:
        ``(weights_sb, scales_sb)``. ``weights_sb`` is
        ``[P_MAX, num_512_tiles, out_dim]``; ``scales_sb`` is
        ``[P_MAX, num_512_tiles, ceil(out_dim/128) * 128]`` with each block
        scale replicated SCALE_BLOCK times along the trailing dim.
    """
    if full_out_dim is None:
        full_out_dim = out_dim
    P_MAX = nl.tile_size.pmax
    SCALE_P_PER_QUAD = 4
    SCALE_BLOCK = 128
    QUADS_PER_TILE = 4

    kernel_assert(
        out_col_offset % SCALE_BLOCK == 0,
        f"out_col_offset ({out_col_offset}) must be a multiple of {SCALE_BLOCK} for compact block-128 scales.",
    )

    in_dim_packed = in_dim // 4
    num_512_tiles = in_dim // (P_MAX * H_PACK)

    full_n_blocks = (full_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK
    out_n_blocks = (out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK
    n_block_offset = out_col_offset // SCALE_BLOCK
    padded_out_dim = out_n_blocks * SCALE_BLOCK

    weights_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, out_dim),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
        name=f"{name}_weights",
    )
    scales_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, padded_out_dim),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name}_scales",
    )

    for h_tile_idx in nl.affine_range(num_512_tiles):
        h_tile_sz = min(P_MAX, in_dim_packed - h_tile_idx * P_MAX)
        nisa.dma_copy(
            dst=weights_sb[0:h_tile_sz, h_tile_idx, 0:out_dim],
            src=weights_hbm.ap(
                pattern=[[full_out_dim, h_tile_sz], [1, out_dim]],
                offset=h_tile_idx * P_MAX * full_out_dim + out_col_offset,
                dtype=nl.float8_e4m3fn_x4,
            ),
            dge_mode=dge_mode.swdge,
        )

    # ---- Stage 1: DMA compact scales (no free-dim broadcast at DMA time). ----
    # Inner scope: compact_scales_sb is released after stage 2 so it doesn't
    # hold SBUF for the kernel duration.
    sbm.open_scope()
    compact_scales_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, out_n_blocks),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name}_scales_compact",
    )
    # Source pattern: 4-row partition broadcast within each quadrant (stride 0
    # on partition axis is legal for DMA), then the K-tile and N-block walks.
    # No [0, SCALE_BLOCK] free-dim broadcast — that fanout is done in stage 2.
    for quad_idx in nl.affine_range(QUADS_PER_TILE):
        nisa.dma_copy(
            dst=compact_scales_sb[nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), 0:num_512_tiles, 0:out_n_blocks],
            src=scales_hbm.ap(
                pattern=[
                    [0, SCALE_P_PER_QUAD],
                    [QUADS_PER_TILE * full_n_blocks, num_512_tiles],
                    [1, out_n_blocks],
                ],
                offset=quad_idx * full_n_blocks + n_block_offset,
                dtype=nl.uint8,
            ),
            dge_mode=dge_mode.swdge,
        )

    # ---- Stage 2: Vector-engine broadcast into the final scales_sb. ----
    # Source view: [P_MAX, num_512_tiles, n_blocks, 1] -> broadcast(dim=3, size=128).
    # Destination view: [P_MAX, num_512_tiles, n_blocks, 128] (reshape of the 3D scales_sb).
    src_view = compact_scales_sb.expand_dim(dim=3).broadcast(dim=3, size=SCALE_BLOCK)
    dst_view = scales_sb.reshape_dim(dim=2, shape=(out_n_blocks, SCALE_BLOCK))
    nisa.tensor_copy(dst=dst_view, src=src_view)

    sbm.close_scope()  # Frees compact_scales_sb.

    return weights_sb, scales_sb


def _load_norm_weights_for_mx(
    norm_weights_hbm: nl.NkiTensor,
    dim: int,
    sbm: SbufManager,
    name: str = "norm_gamma",
) -> nl.NkiTensor:
    """
    Load norm weights [1, dim] to SBUF in swizzled format for MX path.

    Gathers elements with stride-4 to produce a layout where each column contains
    one element per 128-element sub-tile, matching the MX quantization layout.

    Args:
        norm_weights_hbm (nl.NkiTensor): [1, dim] bf16, Norm gamma weights on HBM
        dim (int): Hidden dimension size
        sbm (SbufManager): SBUF memory manager
        name (str): Name for the allocated buffer

    Returns:
        nl.NkiTensor: [P_MAX, num_512_tiles * H_PACK] float32, Swizzled gamma weights in SBUF
    """
    P_MAX = nl.tile_size.pmax
    num_512_tiles = dim // (P_MAX * H_PACK)

    norm_weights_hbm = norm_weights_hbm.reshape((1, dim))
    gamma_sb = sbm.alloc_stack((P_MAX, num_512_tiles * H_PACK), dtype=nl.float32, buffer=nl.sbuf, name=name)

    for h_tile_idx in range(num_512_tiles):
        for h_sub_idx in range(H_PACK):
            src_offset = h_tile_idx * P_MAX * H_PACK + h_sub_idx
            dst_col = h_tile_idx * H_PACK + h_sub_idx
            nisa.dma_copy(
                dst=gamma_sb[0:P_MAX, nl.ds(dst_col, 1)],
                src=norm_weights_hbm.ap(
                    pattern=[[H_PACK, P_MAX], [1, 1]],
                    offset=src_offset,
                ),
                dge_mode=dge_mode.swdge,
            )
    return gamma_sb


def _quantize_mx(
    transposed_sb: nl.NkiTensor,
    num_512_tiles: int,
    s_tile_sz: int,
    sbm: SbufManager,
    name: str = "qtz",
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Quantize bf16 tensor to MX format (fp8x4 + uint8 scales).

    Args:
        transposed_sb (nl.NkiTensor): [P_MAX, num_512_tiles, s_tile_sz * H_PACK], Input tensor in SBUF
        num_512_tiles (int): Number of 512-element tiles
        s_tile_sz (int): Number of active sequence positions in the tile
        sbm (SbufManager): SBUF memory manager
        name (str): Name prefix for allocated buffers

    Returns:
        Tuple[nl.NkiTensor, nl.NkiTensor]:
            - qtz_sb (nl.NkiTensor): [P_MAX, num_512_tiles, s_tile_sz] fp8x4, Quantized values
            - scale_sb (nl.NkiTensor): [P_MAX, num_512_tiles, s_tile_sz] uint8, Per-block scales
    """
    P_MAX = nl.tile_size.pmax

    qtz_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, s_tile_sz),
        dtype=nl.float8_e4m3fn_x4,
        buffer=nl.sbuf,
        name=f"{name}_data",
    )
    scale_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, s_tile_sz),
        dtype=nl.uint8,
        buffer=nl.sbuf,
        name=f"{name}_scale",
    )
    nisa.quantize_mx(
        src=transposed_sb[0:P_MAX, 0:num_512_tiles, 0 : s_tile_sz * H_PACK],
        dst=qtz_sb[0:P_MAX, 0:num_512_tiles, 0:s_tile_sz],
        dst_scale=scale_sb[0:P_MAX, 0:num_512_tiles, 0:s_tile_sz],
    )

    return qtz_sb, scale_sb


def _mx_matmul(
    input_qtz_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    weights_scale_sb: nl.NkiTensor,
    num_k_tiles: int,
    m_dim: int,
    n_dim: int,
    sbm: SbufManager,
    name: str = "matmul",
) -> nl.NkiTensor:
    """
    MX matrix multiplication: input @ weights.

    Args:
        input_qtz_sb (nl.NkiTensor): [P_MAX, num_k_tiles, m_dim] fp8x4, Quantized input (stationary)
        input_scale_sb (nl.NkiTensor): [P_MAX, num_k_tiles, m_dim] uint8, Input scales
        weights_sb (nl.NkiTensor): [P_MAX, num_k_tiles, n_dim] fp8x4, Quantized weights (moving)
        weights_scale_sb (nl.NkiTensor): [P_MAX, num_k_tiles, n_dim] uint8, Weight scales
        num_k_tiles (int): Number of K-dimension tiles to accumulate over
        m_dim (int): M dimension (sequence tile size)
        n_dim (int): N dimension (total output width)
        sbm (SbufManager): SBUF memory manager
        name (str): Name prefix for allocated buffers

    Returns:
        nl.NkiTensor: [m_dim, n_dim] bf16, Matrix multiplication result in SBUF
    """
    P_MAX = nl.tile_size.pmax
    F_MAX = 512
    PSUM_BANK_SIZE = _get_psum_bank_size()

    num_n_tiles = div_ceil(n_dim, F_MAX)

    output_psum = []
    for bank_id in nl.affine_range(num_n_tiles):
        output_psum.append(
            nl.ndarray(
                (P_MAX, F_MAX),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, bank_id * PSUM_BANK_SIZE),
            )
        )

    for i_k_tile in nl.affine_range(num_k_tiles):
        for i_n_tile in nl.affine_range(num_n_tiles):
            n_tile_sz = min(F_MAX, n_dim - i_n_tile * F_MAX)
            nisa.nc_matmul_mx(
                dst=output_psum[i_n_tile][0:m_dim, 0:n_tile_sz],
                stationary=input_qtz_sb[0:P_MAX, i_k_tile, nl.ds(0, m_dim)],
                moving=weights_sb[0:P_MAX, i_k_tile, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
                stationary_scale=input_scale_sb[0:P_MAX, i_k_tile, nl.ds(0, m_dim)],
                moving_scale=weights_scale_sb[0:P_MAX, i_k_tile, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
            )

    output_sb = sbm.alloc_stack((P_MAX, n_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_output")
    for i_n_tile in nl.affine_range(num_n_tiles):
        n_tile_sz = min(F_MAX, n_dim - i_n_tile * F_MAX)
        nisa.tensor_copy(
            dst=output_sb[0:m_dim, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
            src=output_psum[i_n_tile][0:m_dim, 0:n_tile_sz],
        )

    return output_sb


def _mx_matmul_split(
    input_qtz_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    weights_scale_sb: nl.NkiTensor,
    num_k_tiles: int,
    m_dim: int,
    n_dim: int,
    split_points: List[int],
    sbm: SbufManager,
) -> List[nl.NkiTensor]:
    """
    MX matmul with split output: input @ weights -> multiple output buffers.

    Split happens during PSUM->SBUF copy to avoid extra memory movement.

    Args:
        input_qtz_sb (nl.NkiTensor): [P_MAX, num_k_tiles, m_dim] fp8x4, Quantized input (stationary)
        input_scale_sb (nl.NkiTensor): [P_MAX, num_k_tiles, m_dim] uint8, Input scales
        weights_sb (nl.NkiTensor): [P_MAX, num_k_tiles, n_dim] fp8x4, Quantized weights (moving)
        weights_scale_sb (nl.NkiTensor): [P_MAX, num_k_tiles, n_dim] uint8, Weight scales
        num_k_tiles (int): Number of K-dimension tiles to accumulate over
        m_dim (int): M dimension (sequence tile size)
        n_dim (int): N dimension (total output width)
        split_points (List[int]): Column indices at which to split the output
        sbm (SbufManager): SBUF memory manager

    Returns:
        List[nl.NkiTensor]: List of SBUF tensors, one per split segment
    """
    P_MAX = nl.tile_size.pmax
    F_MAX = 512
    PSUM_BANK_SIZE = _get_psum_bank_size()

    num_n_tiles = div_ceil(n_dim, F_MAX)

    output_psum = []
    for bank_id in nl.affine_range(num_n_tiles):
        output_psum.append(
            nl.ndarray(
                (P_MAX, F_MAX),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, bank_id * PSUM_BANK_SIZE),
            )
        )

    for i_k_tile in nl.affine_range(num_k_tiles):
        for i_n_tile in nl.affine_range(num_n_tiles):
            n_tile_sz = min(F_MAX, n_dim - i_n_tile * F_MAX)
            nisa.nc_matmul_mx(
                dst=output_psum[i_n_tile][0:m_dim, 0:n_tile_sz],
                stationary=input_qtz_sb[0:P_MAX, i_k_tile, nl.ds(0, m_dim)],
                moving=weights_sb[0:P_MAX, i_k_tile, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
                stationary_scale=input_scale_sb[0:P_MAX, i_k_tile, nl.ds(0, m_dim)],
                moving_scale=weights_scale_sb[0:P_MAX, i_k_tile, nl.ds(i_n_tile * F_MAX, n_tile_sz)],
            )

    all_splits = [0] + split_points + [n_dim]
    outputs = []

    for split_idx in range(len(all_splits) - 1):
        start_col = all_splits[split_idx]
        end_col = all_splits[split_idx + 1]
        width = end_col - start_col

        out_sb = sbm.alloc_stack(
            (P_MAX, width),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
            name=f"matmul_split_out_{split_idx}",
        )

        col = 0
        while col < width:
            global_col = start_col + col
            bank_idx, bank_offset = divmod(global_col, F_MAX)
            copy_width = min(F_MAX - bank_offset, width - col)

            nisa.tensor_copy(
                dst=out_sb[0:m_dim, nl.ds(col, copy_width)],
                src=output_psum[bank_idx][0:m_dim, nl.ds(bank_offset, copy_width)],
                engine=nisa.scalar_engine,
            )
            col += copy_width

        outputs.append(out_sb)

    return outputs


def _transpose_preswizzled_for_mx(
    input_sb: nl.NkiTensor,
    s_tile_sz: int,
    hidden_dim: int,
    num_512_tiles: int,
    sbm: SbufManager,
    gamma_sb: nl.NkiTensor = None,
    rsqrt_scale_sb: nl.NkiTensor = None,
    name: str = "transpose_preswizzled",
) -> nl.NkiTensor:
    """Transpose pre-swizzled [S, H] -> [H, S] for MX quantization using all 8 PSUM banks.

    Processes two h_tiles simultaneously using 8 PSUM banks (4 per h_tile).
    When ``rsqrt_scale_sb`` and ``gamma_sb`` are both provided, fuses
    rsqrt_scale * gamma into a combined scale buffer up-front so the
    PSUM->SBUF eviction performs a single multiply instead of two ops.

    Args:
        input_sb (nl.NkiTensor): [s_tile_sz, hidden_dim], Pre-swizzled input in SBUF.
        s_tile_sz (int): Number of active sequence positions in the tile.
        hidden_dim (int): Hidden dimension size.
        num_512_tiles (int): Number of 512-element tiles along H.
        sbm (SbufManager): SBUF memory manager.
        gamma_sb (nl.NkiTensor, optional): [P_MAX, num_512_tiles * H_PACK] gamma weights,
            applied during the eviction multiply if provided.
        rsqrt_scale_sb (nl.NkiTensor, optional): [s_tile_sz, 1] rsqrt scale; if also
            ``gamma_sb`` is provided, the two are pre-multiplied into a fused gamma.
        name (str): Name prefix for allocated buffers.

    Returns:
        nl.NkiTensor: [P_MAX, num_512_tiles, s_tile_sz * H_PACK], Transposed tensor in SBUF.
    """
    if rsqrt_scale_sb is not None and gamma_sb is not None:
        fused_gamma_sb = sbm.alloc_stack(gamma_sb.shape, dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_fused_gamma")
        nisa.activation(
            dst=fused_gamma_sb,
            op=nl.copy,
            data=gamma_sb,
            scale=rsqrt_scale_sb,
        )
        gamma_sb = fused_gamma_sb

    P_MAX = nl.tile_size.pmax
    PSUM_BANK_SIZE = _get_psum_bank_size()
    TILES_PER_GROUP = NUM_HW_PSUM_BANKS // H_PACK  # 2 h_tiles at once

    transposed_sb = sbm.alloc_stack(
        (P_MAX, num_512_tiles, s_tile_sz * H_PACK),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name=name,
    )

    transpose_psum = []
    for bank_id in range(NUM_HW_PSUM_BANKS):
        transpose_psum.append(
            nl.ndarray(
                (P_MAX, P_MAX),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, bank_id * PSUM_BANK_SIZE),
            )
        )

    num_groups = num_512_tiles // TILES_PER_GROUP
    remainder = num_512_tiles % TILES_PER_GROUP

    for group_idx in nl.affine_range(num_groups):
        for tile_in_group in nl.affine_range(TILES_PER_GROUP):
            h_tile_idx = group_idx * TILES_PER_GROUP + tile_in_group
            bank_base = tile_in_group * H_PACK
            for h_sub_idx in nl.affine_range(H_PACK):
                src_offset = (h_sub_idx * num_512_tiles + h_tile_idx) * P_MAX
                nisa.nc_transpose(
                    data=input_sb[0:s_tile_sz, src_offset : src_offset + P_MAX],
                    dst=transpose_psum[bank_base + h_sub_idx][0:P_MAX, 0:s_tile_sz],
                )

        for tile_in_group in nl.affine_range(TILES_PER_GROUP):
            h_tile_idx = group_idx * TILES_PER_GROUP + tile_in_group
            bank_base = tile_in_group * H_PACK
            for h_sub_idx in nl.affine_range(H_PACK):
                dst_ap = transposed_sb.ap(
                    pattern=[[num_512_tiles * s_tile_sz * H_PACK, P_MAX], [H_PACK, s_tile_sz]],
                    offset=h_tile_idx * s_tile_sz * H_PACK + h_sub_idx,
                )
                if gamma_sb is not None:
                    gamma_tile_index = h_tile_idx * H_PACK + h_sub_idx
                    nisa.tensor_scalar(
                        dst=dst_ap,
                        data=transpose_psum[bank_base + h_sub_idx][0:P_MAX, 0:s_tile_sz],
                        op0=nl.multiply,
                        operand0=gamma_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                        engine=nisa.scalar_engine,
                    )
                else:
                    nisa.tensor_copy(
                        dst=dst_ap,
                        src=transpose_psum[bank_base + h_sub_idx][0:P_MAX, 0:s_tile_sz],
                    )

    for rem_idx in nl.affine_range(remainder):
        h_tile_idx = num_groups * TILES_PER_GROUP + rem_idx
        for h_sub_idx in nl.affine_range(H_PACK):
            src_offset = (h_sub_idx * num_512_tiles + h_tile_idx) * P_MAX
            nisa.nc_transpose(
                data=input_sb[0:s_tile_sz, src_offset : src_offset + P_MAX],
                dst=transpose_psum[h_sub_idx][0:P_MAX, 0:s_tile_sz],
            )
        for h_sub_idx in nl.affine_range(H_PACK):
            dst_ap = transposed_sb.ap(
                pattern=[[num_512_tiles * s_tile_sz * H_PACK, P_MAX], [H_PACK, s_tile_sz]],
                offset=h_tile_idx * s_tile_sz * H_PACK + h_sub_idx,
            )
            if gamma_sb is not None:
                gamma_tile_index = h_tile_idx * H_PACK + h_sub_idx
                nisa.tensor_scalar(
                    dst=dst_ap,
                    data=transpose_psum[h_sub_idx][0:P_MAX, 0:s_tile_sz],
                    op0=nl.multiply,
                    operand0=gamma_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                    engine=nisa.scalar_engine,
                )
            else:
                nisa.tensor_copy(dst=dst_ap, src=transpose_psum[h_sub_idx][0:P_MAX, 0:s_tile_sz])

    return transposed_sb


def _compute_rms_norm_scale(
    input_sb: nl.NkiTensor,
    zero_bias_sb: nl.NkiTensor,
    norm_eps_sb: nl.NkiTensor,
    s_tile_sz: int,
    hidden_dim: int,
    sbm: SbufManager,
    name: str = "rms",
) -> nl.NkiTensor:
    """Compute rsqrt(mean(x²) + eps) scale factor without applying it.

    Returns the [s_tile_sz, 1] scale factor to be fused into a later step.

    Args:
        input_sb (nl.NkiTensor): [s_tile_sz, hidden_dim], Input tensor in SBUF.
        zero_bias_sb (nl.NkiTensor): [P_MAX, 1], Pre-zeroed bias tensor for activation_reduce.
        norm_eps_sb (nl.NkiTensor): [P_MAX, 1], Pre-filled epsilon tensor.
        s_tile_sz (int): Number of active sequence positions.
        hidden_dim (int): Hidden dimension size.
        sbm (SbufManager): SBUF memory manager.
        name (str): Name prefix for allocated buffers.

    Returns:
        nl.NkiTensor: [P_MAX, 1] float32, rsqrt scale factor in SBUF.
    """
    P_MAX = nl.tile_size.pmax

    square_sum_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_square_sum")
    act_temp_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"{name}_act_temp")

    nisa.activation_reduce(
        dst=act_temp_sb.ap(pattern=[[1, s_tile_sz], [0, hidden_dim]]),
        op=nl.square,
        data=input_sb[0:s_tile_sz, 0:hidden_dim],
        reduce_op=nl.add,
        reduce_res=square_sum_sb[0:s_tile_sz, 0:1],
        bias=zero_bias_sb[0:s_tile_sz, 0:1],
    )

    nisa.activation(
        dst=square_sum_sb[0:s_tile_sz, 0:1],
        op=nl.rsqrt,
        data=square_sum_sb[0:s_tile_sz, 0:1],
        bias=norm_eps_sb[0:s_tile_sz, 0:1],
        scale=float(1.0 / hidden_dim),
    )

    return square_sum_sb


def _apply_rms_norm_inplace(
    input_sb: nl.NkiTensor,
    zero_bias_sb: nl.NkiTensor,
    norm_eps_sb: nl.NkiTensor,
    s_tile_sz: int,
    hidden_dim: int,
    sbm: SbufManager,
    name: str = "rms",
) -> None:
    """Apply RMSNorm in-place: ``x = x / sqrt(mean(x²) + eps)``.

    Computes the reciprocal RMS and multiplies the input in-place. Gamma is
    NOT applied here; callers that fuse gamma into the subsequent
    ``_transpose_preswizzled_for_mx`` should pass ``gamma_sb`` to that
    function instead.

    Args:
        input_sb: [s_tile_sz, hidden_dim], modified in-place.
        zero_bias_sb: [P_MAX, 1] pre-zeroed bias for ``activation_reduce``.
        norm_eps_sb: [P_MAX, 1] pre-filled epsilon tensor.
        s_tile_sz: Number of active sequence positions.
        hidden_dim: Hidden dimension size.
        sbm: SBUF memory manager.
        name: Name prefix for allocated buffers.
    """
    square_sum_sb = _compute_rms_norm_scale(input_sb, zero_bias_sb, norm_eps_sb, s_tile_sz, hidden_dim, sbm, name=name)

    nisa.tensor_scalar(
        dst=input_sb[0:s_tile_sz, 0:hidden_dim],
        data=input_sb[0:s_tile_sz, 0:hidden_dim],
        op0=nl.multiply,
        operand0=square_sum_sb[0:s_tile_sz, 0:1],
        engine=nisa.scalar_engine,
    )


def _apply_rope_to_tensor(
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    s_tile_sz: int,
    rope_dim: int,
    sbm: SbufManager,
    name: str = "rope",
) -> nl.NkiTensor:
    """
    Apply Rotary Position Embedding (RoPE) to a tensor, returning a new buffer.

    Computes: output = [x1, x2] * cos + [-x2, x1] * sin
    where x1 and x2 are the first and second halves of the input along rope_dim.

    Args:
        x_sb (nl.NkiTensor): [s_tile_sz, rope_dim], Input tensor in SBUF.
        cos_sb (nl.NkiTensor): [s_tile_sz, rope_dim], Cosine frequencies in SBUF.
        sin_sb (nl.NkiTensor): [s_tile_sz, rope_dim // 2], Sine frequencies in SBUF.
        s_tile_sz (int): Number of active sequence positions.
        rope_dim (int): RoPE dimension (must be even).
        sbm (SbufManager): SBUF memory manager.
        name (str): Name prefix for allocated buffers.

    Returns:
        nl.NkiTensor: [s_tile_sz, rope_dim], New tensor with RoPE applied in SBUF.
    """
    half_dim = rope_dim // 2

    output_sb = sbm.alloc_stack((s_tile_sz, rope_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_out")
    temp_sb = sbm.alloc_stack((s_tile_sz, rope_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"{name}_tmp")

    nisa.tensor_tensor(
        dst=output_sb[0:s_tile_sz, 0:rope_dim],
        data1=x_sb[0:s_tile_sz, 0:rope_dim],
        data2=cos_sb[0:s_tile_sz, 0:rope_dim],
        op=nl.multiply,
    )

    nisa.tensor_tensor(
        dst=temp_sb[0:s_tile_sz, 0:half_dim],
        data1=x_sb[0:s_tile_sz, half_dim:rope_dim],
        data2=sin_sb[0:s_tile_sz, 0:half_dim],
        op=nl.multiply,
    )
    nisa.tensor_scalar(
        dst=temp_sb[0:s_tile_sz, 0:half_dim],
        data=temp_sb[0:s_tile_sz, 0:half_dim],
        op0=nl.multiply,
        operand0=-1.0,
        engine=nisa.scalar_engine,
    )

    nisa.tensor_tensor(
        dst=temp_sb[0:s_tile_sz, half_dim:rope_dim],
        data1=x_sb[0:s_tile_sz, 0:half_dim],
        data2=sin_sb[0:s_tile_sz, 0:half_dim],
        op=nl.multiply,
    )

    nisa.tensor_tensor(
        dst=output_sb[0:s_tile_sz, 0:rope_dim],
        data1=output_sb[0:s_tile_sz, 0:rope_dim],
        data2=temp_sb[0:s_tile_sz, 0:rope_dim],
        op=nl.add,
    )

    return output_sb


def _apply_rope_inplace(
    x_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    s_tile_sz: int,
    rope_dim: int,
) -> None:
    """
    Apply RoPE in-place using scratch space in the same buffer.

    Computes: x = [x1, x2] * cos + [-x2, x1] * sin, where x1 and x2 are the
    first and second halves of the input along rope_dim. Uses
    ``x_sb[rope_dim:rope_dim*2]`` as scratch space.

    Args:
        x_sb (nl.NkiTensor): [s_tile_sz, rope_dim * 2], Input modified in-place.
            First rope_dim elements are the input; second rope_dim are scratch.
        cos_sb (nl.NkiTensor): [s_tile_sz, rope_dim], Cosine frequencies.
        sin_sb (nl.NkiTensor): [s_tile_sz, rope_dim // 2], Sine frequencies.
        s_tile_sz (int): Number of active sequence positions.
        rope_dim (int): RoPE dimension (must be even).
    """
    half_dim = rope_dim // 2

    nisa.tensor_tensor(
        dst=x_sb[0:s_tile_sz, rope_dim : rope_dim + half_dim],
        data1=x_sb[0:s_tile_sz, half_dim:rope_dim],
        data2=sin_sb[0:s_tile_sz, 0:half_dim],
        op=nl.multiply,
    )
    nisa.tensor_scalar(
        dst=x_sb[0:s_tile_sz, rope_dim : rope_dim + half_dim],
        data=x_sb[0:s_tile_sz, rope_dim : rope_dim + half_dim],
        op0=nl.multiply,
        operand0=-1.0,
    )

    nisa.tensor_tensor(
        dst=x_sb[0:s_tile_sz, rope_dim + half_dim : rope_dim * 2],
        data1=x_sb[0:s_tile_sz, 0:half_dim],
        data2=sin_sb[0:s_tile_sz, 0:half_dim],
        op=nl.multiply,
    )

    nisa.tensor_tensor(
        dst=x_sb[0:s_tile_sz, 0:rope_dim],
        data1=x_sb[0:s_tile_sz, 0:rope_dim],
        data2=cos_sb[0:s_tile_sz, 0:rope_dim],
        op=nl.multiply,
    )

    nisa.tensor_tensor(
        dst=x_sb[0:s_tile_sz, 0:rope_dim],
        data1=x_sb[0:s_tile_sz, 0:rope_dim],
        data2=x_sb[0:s_tile_sz, rope_dim : rope_dim * 2],
        op=nl.add,
    )


# ====================================================================
# v4 post-projection helpers
# ====================================================================
#
# Shared between the v4 fast path (single chunk over all heads) and the
# chunked path (per-head-chunk loop). The fast path passes ``n_heads``
# for the head-count argument; the chunked path passes ``heads_per_chunk``.


def _v4_load_rope_caches(
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    sbm: SbufManager,
    s_tile_sz: int,
    qk_rope_head_dim: int,
    rope_offset: int,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Allocate cos/sin SBUF buffers and DMA them in from HBM.

    The full ``cos`` cache is stored ``[batch * S, qk_rope_head_dim]`` and
    needed in full per S-tile; ``sin`` only needs the first half of
    ``qk_rope_head_dim`` (the half-rope split is reused for the second).
    """
    P_MAX = nl.tile_size.pmax
    cos_sb = sbm.alloc_stack((P_MAX, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_cache")
    sin_sb = sbm.alloc_stack(
        (P_MAX, qk_rope_head_dim // 2),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="sin_cache",
    )
    nisa.dma_copy(
        dst=cos_sb[0:s_tile_sz, 0:qk_rope_head_dim],
        src=cos_cache_hbm.ap(
            pattern=[[qk_rope_head_dim, s_tile_sz], [1, qk_rope_head_dim]],
            offset=rope_offset,
        ),
        dge_mode=dge_mode.swdge,
    )
    nisa.dma_copy(
        dst=sin_sb[0:s_tile_sz, 0 : qk_rope_head_dim // 2],
        src=sin_cache_hbm.ap(
            pattern=[[qk_rope_head_dim, s_tile_sz], [1, qk_rope_head_dim // 2]],
            offset=rope_offset,
        ),
        dge_mode=dge_mode.swdge,
    )
    return cos_sb, sin_sb


def _v4_apply_kv_norm_inplace(
    kv_sb: nl.NkiTensor,
    kv_gamma_sb: nl.NkiTensor,
    zero_bias_sb: nl.NkiTensor,
    norm_eps_sb: nl.NkiTensor,
    sbm: SbufManager,
    s_tile_sz: int,
    kv_dim: int,
) -> None:
    """KV path: rsqrt(mean(x²)+eps) * gamma in-place on ``kv_sb[0:s_tile_sz, 0:kv_dim]``.

    Kept as three separate ops (rsqrt-scale compute, scalar-multiply,
    tensor-multiply) rather than fusing rsqrt and gamma into a single
    multiply, so the compiler can interleave them with the surrounding
    Q-path and RoPE work.
    """
    kv_rsqrt_scale_sb = _compute_rms_norm_scale(
        kv_sb,
        zero_bias_sb,
        norm_eps_sb,
        s_tile_sz,
        kv_dim,
        sbm,
        name="kv_rms",
    )
    nisa.tensor_scalar(
        dst=kv_sb[0:s_tile_sz, 0:kv_dim],
        data=kv_sb[0:s_tile_sz, 0:kv_dim],
        op0=nl.multiply,
        operand0=kv_rsqrt_scale_sb[0:s_tile_sz, 0:1],
        engine=nisa.scalar_engine,
    )
    nisa.tensor_tensor(
        dst=kv_sb[0:s_tile_sz, 0:kv_dim],
        data1=kv_sb[0:s_tile_sz, 0:kv_dim],
        data2=kv_gamma_sb[0:s_tile_sz, 0:kv_dim],
        op=nl.multiply,
    )


def _v4_apply_q_post_rsqrt_norm(
    q_sb: nl.NkiTensor,
    zero_bias_sb: nl.NkiTensor,
    norm_eps_sb: nl.NkiTensor,
    sbm: SbufManager,
    s_tile_pad: int,
    n_heads: int,
    head_dim: int,
    name: str = "q_post",
) -> None:
    """Apply ``q *= rsqrt(mean(q², dim=-1) + eps)`` per head, in-place.

    Two-step: per-head sum-of-squares via ``activation_reduce`` accumulating
    into ``sum_sq_sb[..., i_head]``, then a single ``activation`` op with
    ``op=rsqrt`` and ``scale=1/head_dim`` produces the per-head scale, which
    is broadcast-multiplied across ``head_dim`` via ``nl.NkiTensor``.

    Used by both fast and chunked paths; the chunked path passes
    ``n_heads=heads_per_chunk``.
    """
    P_MAX = nl.tile_size.pmax
    sum_sq_sb = sbm.alloc_stack(
        (P_MAX, n_heads),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=f"{name}_sum_sq",
    )
    act_temp_sb = sbm.alloc_stack(
        (P_MAX, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
        name=f"{name}_act_temp",
    )
    for i_head in nl.affine_range(n_heads):
        head_offset = i_head * head_dim
        nisa.activation_reduce(
            dst=act_temp_sb.ap(pattern=[[1, s_tile_pad], [0, head_dim]]),
            op=nl.square,
            data=q_sb[0:s_tile_pad, nl.ds(head_offset, head_dim)],
            reduce_op=nl.add,
            reduce_res=sum_sq_sb[0:s_tile_pad, nl.ds(i_head, 1)],
            bias=zero_bias_sb[0:s_tile_pad, 0:1],
        )
    nisa.activation(
        dst=sum_sq_sb[0:s_tile_pad, 0:n_heads],
        op=nl.rsqrt,
        data=sum_sq_sb[0:s_tile_pad, 0:n_heads],
        bias=norm_eps_sb[0:s_tile_pad, 0:1],
        scale=float(1.0 / head_dim),
    )
    q_3d = q_sb.reshape_dim(dim=1, shape=(n_heads, head_dim))
    scale_broadcast = sum_sq_sb.reshape_dim(dim=1, shape=(n_heads, 1)).broadcast(dim=2, size=head_dim)
    nisa.tensor_tensor(
        dst=q_3d,
        data1=q_3d,
        data2=scale_broadcast,
        op=nl.multiply,
    )


def _v4_apply_q_rope_inplace(
    q_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    sbm: SbufManager,
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
) -> None:
    """Vectorized RoPE on ``q_sb[..., -qk_rope_head_dim:]`` across heads.

    Reshapes ``q_sb`` to ``[s, n_heads, head_dim]``, slices the last
    ``qk_rope_head_dim`` columns, broadcasts ``cos``/``sin`` across heads,
    and applies the rotary transform in-place. Used by both paths
    (chunked path passes ``n_heads=heads_per_chunk``).
    """
    P_MAX = nl.tile_size.pmax
    qk_nope_head_dim = head_dim - qk_rope_head_dim
    half_rope = qk_rope_head_dim // 2

    q_3d = q_sb.reshape_dim(dim=1, shape=(n_heads, head_dim))
    q_pe_view = q_3d.slice(dim=2, start=qk_nope_head_dim, end=head_dim)

    cos_3d = cos_sb.expand_dim(dim=1).broadcast(dim=1, size=n_heads)
    sin_half = sin_sb.expand_dim(dim=1).broadcast(dim=1, size=n_heads)

    rope_tmp_sb = sbm.alloc_stack(
        (P_MAX, n_heads * qk_rope_head_dim),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="q_rope_tmp",
    )
    rope_tmp_3d = rope_tmp_sb.reshape_dim(dim=1, shape=(n_heads, qk_rope_head_dim))

    q_pe_first = q_pe_view.slice(dim=2, start=0, end=half_rope)
    q_pe_second = q_pe_view.slice(dim=2, start=half_rope, end=qk_rope_head_dim)
    cos_first = cos_3d.slice(dim=2, start=0, end=half_rope)
    cos_second = cos_3d.slice(dim=2, start=half_rope, end=qk_rope_head_dim)
    rope_tmp_first = rope_tmp_3d.slice(dim=2, start=0, end=half_rope)
    rope_tmp_second = rope_tmp_3d.slice(dim=2, start=half_rope, end=qk_rope_head_dim)

    nisa.tensor_tensor(
        dst=rope_tmp_first,
        data1=q_pe_second,
        data2=sin_half,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=rope_tmp_second,
        data1=q_pe_first,
        data2=sin_half,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=q_pe_first,
        data1=q_pe_first,
        data2=cos_first,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=q_pe_first,
        data1=q_pe_first,
        data2=rope_tmp_first,
        op=nl.subtract,
    )
    nisa.tensor_tensor(
        dst=q_pe_second,
        data1=q_pe_second,
        data2=cos_second,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=q_pe_second,
        data1=q_pe_second,
        data2=rope_tmp_second,
        op=nl.add,
    )


def _v4_apply_kv_rope_inplace(
    kv_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    sbm: SbufManager,
    kv_dim: int,
    qk_rope_head_dim: int,
) -> None:
    """Apply RoPE in-place to the last ``qk_rope_head_dim`` columns of ``kv_sb``.

    Same rotary transform as ``_v4_apply_q_rope_inplace`` but on a 1-D row
    layout (no head dim) and using the half-rope sin slice directly.
    """
    P_MAX = nl.tile_size.pmax
    half_rope = qk_rope_head_dim // 2
    kv_rope_offset = kv_dim - qk_rope_head_dim

    kv_pe_first = kv_sb.slice(dim=1, start=kv_rope_offset, end=kv_rope_offset + half_rope)
    kv_pe_second = kv_sb.slice(dim=1, start=kv_rope_offset + half_rope, end=kv_dim)

    kv_rope_tmp_sb = sbm.alloc_stack(
        (P_MAX, qk_rope_head_dim),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
        name="kv_rope_tmp",
    )
    kv_tmp_first = kv_rope_tmp_sb.slice(dim=1, start=0, end=half_rope)
    kv_tmp_second = kv_rope_tmp_sb.slice(dim=1, start=half_rope, end=qk_rope_head_dim)

    cos_first_1d = cos_sb.slice(dim=1, start=0, end=half_rope)
    cos_second_1d = cos_sb.slice(dim=1, start=half_rope, end=qk_rope_head_dim)
    sin_half_1d = sin_sb

    nisa.tensor_tensor(
        dst=kv_tmp_first,
        data1=kv_pe_second,
        data2=sin_half_1d,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=kv_tmp_second,
        data1=kv_pe_first,
        data2=sin_half_1d,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=kv_pe_first,
        data1=kv_pe_first,
        data2=cos_first_1d,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=kv_pe_first,
        data1=kv_pe_first,
        data2=kv_tmp_first,
        op=nl.subtract,
    )
    nisa.tensor_tensor(
        dst=kv_pe_second,
        data1=kv_pe_second,
        data2=cos_second_1d,
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=kv_pe_second,
        data1=kv_pe_second,
        data2=kv_tmp_second,
        op=nl.add,
    )
