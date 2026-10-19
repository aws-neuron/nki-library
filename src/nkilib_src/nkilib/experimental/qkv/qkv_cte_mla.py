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

"""MLA QKV projection kernels for Context Encoding (CTE).

Two MLA kernel variants share helpers from :mod:`qkv_mla_cte_utils`:

- :func:`qkv_mla_mx` (v32): full MLA with separate Q/KV second projections,
  per-head RoPE on q_pe and shared k_pe broadcast across heads.
- :func:`qkv_mla_v4_mx` (v4): fused first projection, no wkv_b second matmul
  (kv latent passed through directly), post-wq_b rsqrt norm on q output, and
  RoPE applied in-place on the last rope_dim elements of both q and kv.
"""

from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ...core.mlp.mlp_tkg.mlp_tkg_utils import _layout_adapter_sb
from ...core.qkv.qkv_cte import _get_psum_bank_size
from ...core.utils.allocator import SbufManager, get_logger
from ...core.utils.kernel_helpers import div_ceil, get_program_sharding_info
from .qkv_mla_cte_utils import (
    _apply_rms_norm_inplace,
    _apply_rope_inplace,
    _apply_rope_to_tensor,
    _load_mx_weights,
    _load_mx_weights_k_slab,
    _load_norm_weights_for_mx,
    _mx_matmul,
    _mx_matmul_split,
    _mx_matmul_split_k_range,
    _quantize_mx,
    _transpose_preswizzled_for_mx,
    _v4_apply_kv_norm_inplace,
    _v4_apply_kv_rope_inplace,
    _v4_apply_q_post_rsqrt_norm,
    _v4_apply_q_rope_inplace,
    _v4_compute_dispatch,
    _v4_load_rope_caches,
    _v32_compute_k_slab_size_512_tiles,
    _validate_mla_inputs,
    _validate_mla_v4_inputs,
)


@nki.jit
def qkv_mla_mx(
    # Input
    x_hbm: nl.NkiTensor,
    wqkv_a_hbm: nl.NkiTensor,
    wqkv_a_scale_hbm: nl.NkiTensor,
    wq_b_hbm: nl.NkiTensor,
    wq_b_scale_hbm: nl.NkiTensor,
    q_norm_gamma_hbm: nl.NkiTensor,
    # KV path weights
    wkv_b_hbm: nl.NkiTensor,
    wkv_b_scale_hbm: nl.NkiTensor,
    kv_norm_gamma_hbm: nl.NkiTensor,
    # RoPE caches
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    # Dimension parameters
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    norm_eps: float = 1e-6,
) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    DeepSeek MLA QKV projection with MX quantization for Context Encoding.

    Implements the full QKV projection pipeline for Multi-head Latent Attention (MLA)
    with MX (fp8) quantization. Includes two-stage low-rank projections for both Q and
    KV paths, fused RMSNorm, and Rotary Position Embedding (RoPE). Supports LNC sharding
    on the sequence dimension.

    - Sequence length: tiled at 128
    - Batch size: 1 (CTE processes single context)

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension (input)
        qk_lora_rank: Q latent dimension (e.g., 1536)
        kv_lora_rank: KV latent dimension (e.g., 512)
        qk_head_dim: qk_nope_head_dim + qk_rope_head_dim (e.g., 128 + 64 = 192)
        qk_nope_head_dim: Non-rotary part of Q/K (e.g., 128)
        qk_rope_head_dim: Rotary part of Q/K (e.g., 64)
        v_head_dim: Value head dimension (e.g., 128)
        n_heads: Number of attention heads

    Args:
        x_hbm (nl.NkiTensor): [B, S, H] bf16, Input hidden states
        wqkv_a_hbm (nl.NkiTensor): [H//4, qk_lora_rank] fp8x4, First combined Q/KV projection weights
        wqkv_a_scale_hbm (nl.NkiTensor): [H//128, ceil((qk_lora_rank + kv_lora_rank + qk_rope_head_dim)/128)] uint8,
            DeepSeek block-128 compact scales for wqkv_a
        wq_b_hbm (nl.NkiTensor): [qk_lora_rank//4, n_heads * qk_head_dim] fp8x4, Second Q projection weights
        wq_b_scale_hbm (nl.NkiTensor): [qk_lora_rank//128, ceil(n_heads * qk_head_dim / 128)] uint8,
            DeepSeek block-128 compact scales for wq_b
        q_norm_gamma_hbm (nl.NkiTensor): [1, qk_lora_rank] bf16, RMSNorm gamma for Q intermediate
        wkv_b_hbm (nl.NkiTensor): [kv_lora_rank//4, n_heads * (qk_nope_head_dim + v_head_dim)] fp8x4,
            Second KV projection weights
        wkv_b_scale_hbm (nl.NkiTensor): [kv_lora_rank//128, ceil(n_heads * (qk_nope_head_dim + v_head_dim) / 128)] uint8,
            MX scales for wkv_b
        kv_norm_gamma_hbm (nl.NkiTensor): [1, kv_lora_rank] bf16, RMSNorm gamma for KV intermediate
        cos_cache_hbm (nl.NkiTensor): [B, S, qk_rope_head_dim] bf16, Cosine RoPE frequencies
        sin_cache_hbm (nl.NkiTensor): [B, S, qk_rope_head_dim] bf16, Sine RoPE frequencies
        n_heads (int): Number of attention heads
        qk_nope_head_dim (int): Non-RoPE portion of Q/K head dimension
        qk_rope_head_dim (int): RoPE portion of Q/K head dimension
        v_head_dim (int): Value head dimension
        kv_lora_rank (int): Latent dimension for KV compression
        qk_lora_rank (int): Latent dimension for Q compression
        norm_eps (float): RMSNorm epsilon. Defaults to 1e-6

    Returns:
        Q (nl.NkiTensor): [B, S, n_heads, qk_head_dim] bf16, Query projections with RoPE applied
        K (nl.NkiTensor): [B, S, n_heads, qk_head_dim] bf16, Key projections with RoPE applied
        V (nl.NkiTensor): [B, S, n_heads, v_head_dim] bf16, Value projections

    Notes:
        - Matmul shapes:
          Combined Q/KV Path Stage 1:
            x[B,S,H] @ wqkv_a[H, qk_lora_rank + kv_lora_rank + qk_rope_head_dim]
                -> qkv_a_out[B,S,qk_lora_rank + kv_lora_rank + qk_rope_head_dim]
            Split: qr[B,S,qk_lora_rank], kv[B,S,kv_lora_rank], k_pe[B,S,qk_rope_head_dim]
          Q Path Stage 2:
            norm(qr)[B,S,qk_lora_rank] @ wq_b[qk_lora_rank, n_heads*qk_head_dim]
                -> q[B,S,n_heads*qk_head_dim]
          KV Path Stage 2:
            norm(kv)[B,S,kv_lora_rank] @ wkv_b[kv_lora_rank, n_heads*(qk_nope_head_dim+v_head_dim)]
                -> kv_out[B,S,n_heads*(qk_nope_head_dim+v_head_dim)]
            Split: k_nope[B,S,n_heads,qk_nope_head_dim], v[B,S,n_heads,v_head_dim]
          Final assembly:
            q_pe = q[..., qk_nope_head_dim:] -> RoPE -> q[..., qk_nope_head_dim:]
            k_pe -> RoPE -> broadcast to all heads -> concat with k_nope -> K
    """

    _validate_mla_inputs(
        x_hbm=x_hbm,
        wqkv_a_hbm=wqkv_a_hbm,
        wqkv_a_scale_hbm=wqkv_a_scale_hbm,
        wq_b_hbm=wq_b_hbm,
        wq_b_scale_hbm=wq_b_scale_hbm,
        q_norm_gamma_hbm=q_norm_gamma_hbm,
        wkv_b_hbm=wkv_b_hbm,
        wkv_b_scale_hbm=wkv_b_scale_hbm,
        kv_norm_gamma_hbm=kv_norm_gamma_hbm,
        cos_cache_hbm=cos_cache_hbm,
        sin_cache_hbm=sin_cache_hbm,
        n_heads=n_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_lora_rank=qk_lora_rank,
        norm_eps=norm_eps,
    )

    B, S, H = x_hbm.shape
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    # ==================== LNC Sharding Setup ====================
    _, num_shards, shard_id = get_program_sharding_info()

    S_shard_base = S // num_shards
    S_shard = S_shard_base
    if S % num_shards != 0 and shard_id == num_shards - 1:
        S_shard = S // num_shards + (S % num_shards)

    S_shard_offset = shard_id * S_shard_base

    q_out_dim = n_heads * qk_head_dim
    kv_a_out_dim = kv_lora_rank + qk_rope_head_dim
    kv_b_out_dim = n_heads * (qk_nope_head_dim + v_head_dim)
    qkv_out_dim = qk_lora_rank + kv_a_out_dim

    # Hardware constants
    P_MAX = nl.tile_size.pmax  # 128
    H_TILE_SIZE = P_MAX
    H_TILE_COUNT = H // H_TILE_SIZE
    H_PACK = 4  # MX packs 4 fp8 elements

    q_hbm = nl.ndarray((B, S, n_heads, qk_head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    k_hbm = nl.ndarray((B, S, n_heads, qk_head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    v_hbm = nl.ndarray((B, S, n_heads, v_head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        use_auto_alloc=False,
        logger=get_logger("mla"),
    )
    sbm.open_scope()

    # ==================== Load norm weights ====================
    q_norm_gamma_sb = _load_norm_weights_for_mx(q_norm_gamma_hbm, qk_lora_rank, sbm, name="q_norm_gamma")
    kv_norm_gamma_sb = _load_norm_weights_for_mx(kv_norm_gamma_hbm, kv_lora_rank, sbm, name="kv_norm_gamma")

    norm_eps_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="norm_eps")
    nisa.memset(dst=norm_eps_sb, value=norm_eps)

    zero_bias_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="zero_bias")
    nisa.memset(dst=zero_bias_sb, value=0.0)

    # ==================== Tiling setup ====================
    S_TILE_SIZE = s_tile_pad = P_MAX

    H_512_TILE_COUNT = H // (P_MAX * H_PACK)
    QK_LORA_128_TILE_COUNT = qk_lora_rank // (P_MAX * H_PACK)
    KV_LORA_128_TILE_COUNT = kv_lora_rank // (P_MAX * H_PACK)

    # K-slab dispatch for the first (Q/KV stage 1) projection.
    _CALIBRATED_SBUF_BUDGET_BYTES = 240 * 1024
    _sbuf_budget = min(int(nl.tile_size.total_available_sbuf_size), _CALIBRATED_SBUF_BUDGET_BYTES)

    # Single-buffered slabs
    _NUM_WQKV_A_SLAB_BUFFERS = 1
    K_SLAB_SIZE_512 = _v32_compute_k_slab_size_512_tiles(
        H=H,
        qkv_out_dim=qkv_out_dim,
        q_out_dim=q_out_dim,
        kv_b_out_dim=kv_b_out_dim,
        qk_lora_rank=qk_lora_rank,
        kv_lora_rank=kv_lora_rank,
        n_heads=n_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        sbuf_budget_bytes=_sbuf_budget,
        num_slab_buffers=_NUM_WQKV_A_SLAB_BUFFERS,
    )
    NUM_K_SLABS = H_512_TILE_COUNT // K_SLAB_SIZE_512
    USE_K_SLAB = NUM_K_SLABS > 1

    if not USE_K_SLAB:
        # Fast path: prologue-load all of wqkv_a once.
        wqkv_a_sb, wqkv_a_scale_sb = _load_mx_weights(wqkv_a_hbm, wqkv_a_scale_hbm, H, qkv_out_dim, sbm, name="wqkv_a")
    wq_b_sb, wq_b_scale_sb = _load_mx_weights(wq_b_hbm, wq_b_scale_hbm, qk_lora_rank, q_out_dim, sbm, name="wq_b")
    wkv_b_sb, wkv_b_scale_sb = _load_mx_weights(
        wkv_b_hbm, wkv_b_scale_hbm, kv_lora_rank, kv_b_out_dim, sbm, name="wkv_b"
    )

    SCALE_BLOCK = 128
    qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    if USE_K_SLAB:
        # Slabbed path: allocate slab-sized buffers ONCE up front and reuse
        # them across all S-tiles. Each S-tile drives a slab loop that
        # DMA-loads a K-slice of wqkv_a into these buffers and runs that
        # slab's matmul. The scale buffer is padded up to a whole 128-N-block;
        # the materialization happens inside the per-quadrant DMA via a
        # stride-0 free-dim broadcast access pattern (no separate scratch).
        wqkv_a_slab_bufs = []
        wqkv_a_scale_slab_bufs = []
        for buf_idx in range(_NUM_WQKV_A_SLAB_BUFFERS):
            wqkv_a_slab_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, K_SLAB_SIZE_512, qkv_out_dim),
                    dtype=nl.float8_e4m3fn_x4,
                    buffer=nl.sbuf,
                    name=f"wqkv_a_slab_buf_{buf_idx}",
                )
            )
            wqkv_a_scale_slab_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, K_SLAB_SIZE_512, qkv_out_dim_padded),
                    dtype=nl.uint8,
                    buffer=nl.sbuf,
                    name=f"wqkv_a_scale_slab_buf_{buf_idx}",
                )
            )

    # ==================== Multi-buffering setup ====================
    NUM_INPUT_BUFFERS = 2
    S_BLOCK_SIZE = NUM_INPUT_BUFFERS * S_TILE_SIZE
    num_S_blocks = div_ceil(S_shard, S_BLOCK_SIZE)

    hidden_for_transpose = x_hbm.reshape((S * H_TILE_COUNT, 1, 1, H_TILE_SIZE))

    for s_block_idx in nl.affine_range(num_S_blocks):
        sbm.open_scope()
        sbm.set_name_prefix(f"sb{s_block_idx}_")

        s_block_offset = s_block_idx * S_BLOCK_SIZE
        s_block_sz = min(S_BLOCK_SIZE, S_shard - s_block_offset)
        num_tiles_in_block = div_ceil(s_block_sz, S_TILE_SIZE)

        # ================================================================
        # STEP 1: Pre-allocate and load input for all tiles in the block.
        # ================================================================
        x_sb_bufs = []
        for buf_idx in range(num_tiles_in_block):
            x_sb_bufs.append(
                sbm.alloc_stack(
                    (H_TILE_SIZE, s_tile_pad, H_TILE_COUNT),
                    dtype=x_hbm.dtype,
                    buffer=nl.sbuf,
                    align=32,
                    name=f"x_sb_buf_{buf_idx}",
                )
            )

        # Block-scope layout-adapter destination. Hoisting out of tile scope
        # prevents aliasing with tile-scope matmul-side buffers (e.g.
        # qr_transpose), which previously serialized the next tile's
        # layout_adapter tensor_copy behind the previous tile's nc_transpose.
        shfl_sb_buf = sbm.alloc_stack(
            (P_MAX, H_512_TILE_COUNT, s_tile_pad, H_PACK),
            dtype=x_hbm.dtype,
            buffer=nl.sbuf,
            name="shfl_sb_buf",
        )

        # x_qtz / x_scale must be double-buffered in block scope: matmul_mx
        # reads them as stationary inputs across all K/N tile iterations.
        x_qtz_sb_bufs = []
        x_scale_sb_bufs = []
        for buf_idx in range(num_tiles_in_block):
            x_qtz_sb_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, H_512_TILE_COUNT, s_tile_pad),
                    dtype=nl.float8_e4m3fn_x4,
                    buffer=nl.sbuf,
                    name=f"x_qtz_buf_{buf_idx}",
                )
            )
            x_scale_sb_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, H_512_TILE_COUNT, s_tile_pad),
                    dtype=nl.uint8,
                    buffer=nl.sbuf,
                    name=f"x_scale_buf_{buf_idx}",
                )
            )

        for i_tile in range(num_tiles_in_block):
            s_tile_local_offset = s_block_offset + i_tile * S_TILE_SIZE
            s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)
            s_tile_global_offset = S_shard_offset + s_tile_local_offset

            hidden_sb_flat = x_sb_bufs[i_tile].reshape((H_TILE_SIZE, 1, 1, s_tile_pad * H_TILE_COUNT))

            nisa.dma_transpose(
                dst=hidden_sb_flat[0:H_TILE_SIZE, 0:1, 0:1, 0 : s_tile_sz * H_TILE_COUNT],
                src=hidden_for_transpose[
                    s_tile_global_offset * H_TILE_COUNT : (s_tile_global_offset + s_tile_sz) * H_TILE_COUNT,
                    0:1,
                    0:1,
                    0:H_TILE_SIZE,
                ],
            )

        # ================================================================
        # STEP 2-12: Process each tile in the block.
        # ================================================================
        for i_tile in nl.affine_range(num_tiles_in_block):
            sbm.open_scope()
            sbm.set_name_prefix(f"sb{s_block_idx}_t{i_tile}_")

            s_tile_local_offset = s_block_offset + i_tile * S_TILE_SIZE
            s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)
            s_tile_global_offset = S_shard_offset + s_tile_local_offset

            x_qtz_sb = x_qtz_sb_bufs[i_tile]
            x_scale_sb = x_scale_sb_bufs[i_tile]

            x_transposed_sb = _layout_adapter_sb(src=x_sb_bufs[i_tile], n_prgs=1, prg_id=0).reshape(
                (P_MAX, H_512_TILE_COUNT, s_tile_pad * H_PACK)
            )
            nisa.quantize_mx(
                src=x_transposed_sb[0:P_MAX, 0:H_512_TILE_COUNT, 0 : s_tile_pad * H_PACK],
                dst=x_qtz_sb[0:P_MAX, 0:H_512_TILE_COUNT, 0:s_tile_pad],
                dst_scale=x_scale_sb[0:P_MAX, 0:H_512_TILE_COUNT, 0:s_tile_pad],
            )

            # ============================================================
            # Prefetch RoPE caches early (overlaps with matmul compute)
            # ============================================================
            cos_sb = sbm.alloc_stack((P_MAX, qk_rope_head_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="cos_cache")
            sin_sb = sbm.alloc_stack(
                (P_MAX, qk_rope_head_dim // 2),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name="sin_cache",
            )

            rope_offset = s_tile_global_offset * qk_rope_head_dim
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

            # ============================================================
            # Combined Q/KV Stage 1 - x @ wqkv_a -> qr, kv + k_pe
            # ============================================================
            if not USE_K_SLAB:
                qr_sb, kv_raw_sb = _mx_matmul_split(
                    x_qtz_sb,
                    x_scale_sb,
                    wqkv_a_sb,
                    wqkv_a_scale_sb,
                    H_512_TILE_COUNT,
                    s_tile_pad,
                    qkv_out_dim,
                    [qk_lora_rank],
                    sbm,
                )
            else:
                # Slabbed path: allocate PSUM banks, then loop over slabs
                # streaming wqkv_a from HBM. PSUM accumulates implicitly
                # across slabs (each call targets the same dst banks).
                F_MAX = 512
                PSUM_BANK_SIZE = _get_psum_bank_size()
                num_n_tiles = (qkv_out_dim + F_MAX - 1) // F_MAX
                output_psum = []
                for bank_id in range(num_n_tiles):
                    output_psum.append(
                        nl.ndarray(
                            (P_MAX, F_MAX),
                            dtype=nl.bfloat16,
                            buffer=nl.psum,
                            address=(0, bank_id * PSUM_BANK_SIZE),
                        )
                    )

                for slab_id in range(NUM_K_SLABS):
                    cur_buf_idx = slab_id % _NUM_WQKV_A_SLAB_BUFFERS
                    slab_buf = wqkv_a_slab_bufs[cur_buf_idx]
                    slab_scale_buf = wqkv_a_scale_slab_bufs[cur_buf_idx]
                    _load_mx_weights_k_slab(
                        wqkv_a_hbm,
                        wqkv_a_scale_hbm,
                        slab_buf,
                        slab_scale_buf,
                        in_dim_full=H,
                        out_dim=qkv_out_dim,
                        k_tile_start=slab_id * K_SLAB_SIZE_512,
                        k_tile_count=K_SLAB_SIZE_512,
                        sbm=sbm,
                        name=f"slab{slab_id}",
                    )
                    _mx_matmul_split_k_range(
                        input_qtz_sb=x_qtz_sb,
                        input_scale_sb=x_scale_sb,
                        weights_slab_sb=slab_buf,
                        weights_slab_scale_sb=slab_scale_buf,
                        k_tile_start=slab_id * K_SLAB_SIZE_512,
                        k_tile_count=K_SLAB_SIZE_512,
                        m_dim=s_tile_pad,
                        n_dim=qkv_out_dim,
                        output_psum=output_psum,
                    )

                # Post-loop: split PSUM into qr_sb (qk_lora_rank) and
                # kv_raw_sb (qkv_out_dim - qk_lora_rank), copying out per-bank.
                qr_sb = sbm.alloc_stack(
                    (P_MAX, qk_lora_rank),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                    name="matmul_split_out_0",
                )
                kv_raw_width = qkv_out_dim - qk_lora_rank
                kv_raw_sb = sbm.alloc_stack(
                    (P_MAX, kv_raw_width),
                    dtype=nl.bfloat16,
                    buffer=nl.sbuf,
                    name="matmul_split_out_1",
                )

                # Split-0: PSUM[0 : qk_lora_rank] -> qr_sb
                width = qk_lora_rank
                col = 0
                while col < width:
                    bank_idx, bank_offset = divmod(col, F_MAX)
                    copy_width = min(F_MAX - bank_offset, width - col)
                    nisa.tensor_copy(
                        dst=qr_sb[0:s_tile_pad, nl.ds(col, copy_width)],
                        src=output_psum[bank_idx][0:s_tile_pad, nl.ds(bank_offset, copy_width)],
                        engine=nisa.scalar_engine,
                    )
                    col += copy_width

                # Split-1: PSUM[qk_lora_rank : qkv_out_dim] -> kv_raw_sb
                col = 0
                while col < kv_raw_width:
                    global_col = qk_lora_rank + col
                    bank_idx, bank_offset = divmod(global_col, F_MAX)
                    copy_width = min(F_MAX - bank_offset, kv_raw_width - col)
                    nisa.tensor_copy(
                        dst=kv_raw_sb[0:s_tile_pad, nl.ds(col, copy_width)],
                        src=output_psum[bank_idx][0:s_tile_pad, nl.ds(bank_offset, copy_width)],
                        engine=nisa.scalar_engine,
                    )
                    col += copy_width

            # ============================================================
            # Q Path - Apply RMSNorm to qr (gamma fused into transpose below)
            # ============================================================
            _apply_rms_norm_inplace(
                qr_sb,
                zero_bias_sb,
                norm_eps_sb,
                s_tile_pad,
                qk_lora_rank,
                sbm,
                name="q_rms",
            )

            # ============================================================
            # Q Path Stage 2 - qr_normed @ wq_b -> q
            # ============================================================
            qr_transposed_sb = _transpose_preswizzled_for_mx(
                qr_sb,
                s_tile_pad,
                qk_lora_rank,
                QK_LORA_128_TILE_COUNT,
                sbm,
                gamma_sb=q_norm_gamma_sb,
                name="qr_transpose",
            )
            qr_qtz_sb, qr_scale_sb = _quantize_mx(
                qr_transposed_sb, QK_LORA_128_TILE_COUNT, s_tile_pad, sbm, name="qr_qtz"
            )
            q_sb = _mx_matmul(
                qr_qtz_sb,
                qr_scale_sb,
                wq_b_sb,
                wq_b_scale_sb,
                QK_LORA_128_TILE_COUNT,
                s_tile_pad,
                q_out_dim,
                sbm,
                name="q_matmul",
            )

            # ============================================================
            # Split kv_raw into kv and k_pe
            # ============================================================
            kv_sb = sbm.alloc_stack((P_MAX, kv_lora_rank), dtype=nl.bfloat16, buffer=nl.sbuf, name="kv_split")
            nisa.tensor_copy(
                dst=kv_sb[0:s_tile_sz, 0:kv_lora_rank],
                src=kv_raw_sb[0:s_tile_sz, 0:kv_lora_rank],
                engine=nisa.scalar_engine,
            )

            k_pe_sb = sbm.alloc_stack(
                (P_MAX, qk_rope_head_dim),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name="k_pe_split",
            )
            nisa.tensor_copy(
                dst=k_pe_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                src=kv_raw_sb[0:s_tile_sz, kv_lora_rank : kv_lora_rank + qk_rope_head_dim],
                engine=nisa.scalar_engine,
            )

            # ============================================================
            # KV Path - Apply RMSNorm to kv (gamma fused into transpose below)
            # ============================================================
            _apply_rms_norm_inplace(
                kv_sb,
                zero_bias_sb,
                norm_eps_sb,
                s_tile_sz,
                kv_lora_rank,
                sbm,
                name="kv_rms",
            )

            # ============================================================
            # KV Path Stage 2 - kv_normed @ wkv_b -> kv_out
            # ============================================================
            kv_transposed_sb = _transpose_preswizzled_for_mx(
                kv_sb,
                s_tile_pad,
                kv_lora_rank,
                KV_LORA_128_TILE_COUNT,
                sbm,
                gamma_sb=kv_norm_gamma_sb,
                name="kv_transpose",
            )
            kv_qtz_sb, kv_scale_sb = _quantize_mx(
                kv_transposed_sb, KV_LORA_128_TILE_COUNT, s_tile_pad, sbm, name="kv_qtz"
            )

            kv_out_sb = _mx_matmul(
                kv_qtz_sb,
                kv_scale_sb,
                wkv_b_sb,
                wkv_b_scale_sb,
                KV_LORA_128_TILE_COUNT,
                s_tile_pad,
                kv_b_out_dim,
                sbm,
                name="kv_matmul",
            )

            # ============================================================
            # Apply RoPE to k_pe (shared across all heads)
            # ============================================================
            k_pe_rope_sb = _apply_rope_to_tensor(
                k_pe_sb, cos_sb, sin_sb, s_tile_sz, qk_rope_head_dim, sbm, name="k_pe_rope"
            )

            # ============================================================
            # Assemble Q, K, V and apply RoPE to Q's rope portion
            # ============================================================
            q_tile_sb = sbm.alloc_stack(
                (P_MAX, n_heads * qk_head_dim),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name="q_tile_out",
            )
            k_tile_sb = sbm.alloc_stack(
                (P_MAX, n_heads * qk_head_dim),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name="k_tile_out",
            )
            v_tile_sb = sbm.alloc_stack(
                (P_MAX, n_heads * v_head_dim),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name="v_tile_out",
            )

            rope_temp_sb = sbm.alloc_stack(
                (P_MAX, qk_rope_head_dim * 2),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name="q_rope_scratch",
            )

            for head_idx in nl.affine_range(n_heads):
                # ----- Q assembly for this head -----
                q_head_offset = head_idx * qk_head_dim

                nisa.tensor_copy(
                    dst=q_tile_sb[0:s_tile_sz, q_head_offset : q_head_offset + qk_nope_head_dim],
                    src=q_sb[0:s_tile_sz, q_head_offset : q_head_offset + qk_nope_head_dim],
                    engine=nisa.scalar_engine,
                )

                q_pe_offset = q_head_offset + qk_nope_head_dim
                nisa.tensor_copy(
                    dst=rope_temp_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                    src=q_sb[0:s_tile_sz, q_pe_offset : q_pe_offset + qk_rope_head_dim],
                    engine=nisa.scalar_engine,
                )

                _apply_rope_inplace(rope_temp_sb, cos_sb, sin_sb, s_tile_sz, qk_rope_head_dim)

                nisa.tensor_copy(
                    dst=q_tile_sb[0:s_tile_sz, q_pe_offset : q_pe_offset + qk_rope_head_dim],
                    src=rope_temp_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                    engine=nisa.scalar_engine,
                )

                # ----- K assembly for this head -----
                kv_head_stride = qk_nope_head_dim + v_head_dim
                kv_head_offset = head_idx * kv_head_stride
                k_head_offset = head_idx * qk_head_dim

                nisa.tensor_copy(
                    dst=k_tile_sb[0:s_tile_sz, k_head_offset : k_head_offset + qk_nope_head_dim],
                    src=kv_out_sb[0:s_tile_sz, kv_head_offset : kv_head_offset + qk_nope_head_dim],
                    engine=nisa.scalar_engine,
                )

                nisa.tensor_copy(
                    dst=k_tile_sb[
                        0:s_tile_sz,
                        k_head_offset + qk_nope_head_dim : k_head_offset + qk_head_dim,
                    ],
                    src=k_pe_rope_sb[0:s_tile_sz, 0:qk_rope_head_dim],
                    engine=nisa.scalar_engine,
                )

                # ----- V extraction for this head -----
                v_head_offset = head_idx * v_head_dim
                v_src_offset = kv_head_offset + qk_nope_head_dim

                nisa.tensor_copy(
                    dst=v_tile_sb[0:s_tile_sz, v_head_offset : v_head_offset + v_head_dim],
                    src=kv_out_sb[0:s_tile_sz, v_src_offset : v_src_offset + v_head_dim],
                    engine=nisa.scalar_engine,
                )

            # ============================================================
            # Store Q, K, V to HBM
            # ============================================================
            nisa.dma_copy(
                dst=q_hbm.ap(
                    pattern=[[n_heads * qk_head_dim, s_tile_sz], [1, n_heads * qk_head_dim]],
                    offset=s_tile_global_offset * n_heads * qk_head_dim,
                ),
                src=q_tile_sb[0:s_tile_sz, 0 : n_heads * qk_head_dim],
                dge_mode=dge_mode.swdge,
            )

            nisa.dma_copy(
                dst=k_hbm.ap(
                    pattern=[[n_heads * qk_head_dim, s_tile_sz], [1, n_heads * qk_head_dim]],
                    offset=s_tile_global_offset * n_heads * qk_head_dim,
                ),
                src=k_tile_sb[0:s_tile_sz, 0 : n_heads * qk_head_dim],
                dge_mode=dge_mode.swdge,
            )

            nisa.dma_copy(
                dst=v_hbm.ap(
                    pattern=[[n_heads * v_head_dim, s_tile_sz], [1, n_heads * v_head_dim]],
                    offset=s_tile_global_offset * n_heads * v_head_dim,
                ),
                src=v_tile_sb[0:s_tile_sz, 0 : n_heads * v_head_dim],
                dge_mode=dge_mode.swdge,
            )

            sbm.close_scope()

        sbm.close_scope()

    sbm.close_scope()
    return q_hbm, k_hbm, v_hbm


@nki.jit
def qkv_mla_mx_deepseek_v4(
    # Input
    x_hbm: nl.NkiTensor,
    # Fused first projection
    wqkv_hbm: nl.NkiTensor,
    wqkv_scale_hbm: nl.NkiTensor,
    # Q second projection
    wq_b_hbm: nl.NkiTensor,
    wq_b_scale_hbm: nl.NkiTensor,
    q_norm_gamma_hbm: nl.NkiTensor,
    # KV norm
    kv_norm_gamma_hbm: nl.NkiTensor,
    # RoPE caches
    cos_cache_hbm: nl.NkiTensor,
    sin_cache_hbm: nl.NkiTensor,
    # Dimensions
    n_heads: int,
    head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    qk_lora_rank: int,
    norm_eps: float = 1e-6,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """DeepSeek v4 MLA QKV projection with MX quantization.

    Variant of :func:`qkv_mla_mx` that fuses the first projection (single wqkv
    matmul instead of separate Q/KV first projections), drops the wkv_b second
    matmul (kv latent is passed through directly to the caller), applies a
    post-wq_b rsqrt norm per head on q, and applies RoPE in-place on the last
    rope_dim elements of both q and kv.

    Pipeline:
        Fused: x -> wqkv -> split [qr, kv]
        Q path: qr -> q_norm(gamma) -> wq_b -> unflatten -> rsqrt_norm -> RoPE(q[..., -rd:])
        KV path: kv -> kv_norm(gamma) -> RoPE(kv[..., -rd:])

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension (input)
        qk_lora_rank: Q latent dimension
        kv_lora_rank: KV latent dimension
        kv_dim: kv_lora_rank + qk_rope_head_dim (full kv output width)
        head_dim: full Q/K head dimension (nope + rope)
        qk_rope_head_dim: RoPE portion of Q/K head dimension
        n_heads: Number of attention heads

    Args:
        x_hbm: [B, S, H] bf16 input hidden states
        wqkv_hbm: [H//4, qk_lora_rank + kv_dim] fp8x4 fused Q/KV first projection
        wqkv_scale_hbm: [H//128, ceil((qk_lora_rank + kv_dim)/128)] uint8 DeepSeek block-128 compact scales
        wq_b_hbm: [qk_lora_rank//4, n_heads * head_dim] fp8x4 second Q projection
        wq_b_scale_hbm: [qk_lora_rank//128, ceil(n_heads * head_dim / 128)] uint8 DeepSeek block-128 compact scales
        q_norm_gamma_hbm: [1, qk_lora_rank] bf16 RMSNorm gamma for Q intermediate
        kv_norm_gamma_hbm: [1, kv_dim] bf16 RMSNorm gamma for KV
        cos_cache_hbm: [B, S, qk_rope_head_dim] bf16
        sin_cache_hbm: [B, S, qk_rope_head_dim] bf16
        n_heads: number of attention heads
        head_dim: full Q/K head dimension (nope + rope)
        qk_rope_head_dim: RoPE dimension
        kv_lora_rank: KV latent dimension
        qk_lora_rank: Q latent dimension
        norm_eps: RMSNorm epsilon

    Returns:
        q: [B, S, n_heads, head_dim] bf16 with RoPE on last rope_dim
        kv: [B, S, kv_dim] bf16 with RoPE on last rope_dim

    Notes:
        - Matmul shapes:
          Fused first projection:
            x[B,S,H] @ wqkv[H, qk_lora_rank + kv_dim]
                -> qkv_a_out[B,S,qk_lora_rank + kv_dim]
            Split: qr[B,S,qk_lora_rank], kv[B,S,kv_dim]
          Q Path Stage 2:
            norm(qr)[B,S,qk_lora_rank] @ wq_b[qk_lora_rank, n_heads*head_dim]
                -> q[B,S,n_heads*head_dim]
          Final assembly:
            q -> per-head rsqrt norm -> RoPE on q[..., -rope_dim:]
            kv -> RoPE on kv[..., -rope_dim:] (kv latent returned directly)
    """
    _validate_mla_v4_inputs(
        x_hbm=x_hbm,
        wqkv_hbm=wqkv_hbm,
        wqkv_scale_hbm=wqkv_scale_hbm,
        wq_b_hbm=wq_b_hbm,
        wq_b_scale_hbm=wq_b_scale_hbm,
        q_norm_gamma_hbm=q_norm_gamma_hbm,
        kv_norm_gamma_hbm=kv_norm_gamma_hbm,
        cos_cache_hbm=cos_cache_hbm,
        sin_cache_hbm=sin_cache_hbm,
        n_heads=n_heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_lora_rank=qk_lora_rank,
        norm_eps=norm_eps,
    )

    B, S, H = x_hbm.shape
    kv_dim = kv_lora_rank + qk_rope_head_dim
    q_out_dim = n_heads * head_dim
    qkv_out_dim = qk_lora_rank + kv_dim

    # LNC sharding on S
    _, num_shards, shard_id = get_program_sharding_info()
    S_shard_base = S // num_shards
    S_shard = S_shard_base
    if S % num_shards != 0 and shard_id == num_shards - 1:
        S_shard = S_shard_base + (S % num_shards)
    S_shard_offset = shard_id * S_shard_base

    P_MAX = nl.tile_size.pmax
    H_TILE_SIZE = P_MAX
    H_TILE_COUNT = H // H_TILE_SIZE
    H_PACK = 4

    q_hbm = nl.ndarray((B, S, n_heads, head_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    kv_hbm = nl.ndarray((B, S, kv_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    _CALIBRATED_SBUF_BUDGET_BYTES = 232 * 1024
    _sbuf_budget = min(int(nl.tile_size.total_available_sbuf_size), _CALIBRATED_SBUF_BUDGET_BYTES)
    # Unified dispatch: returns (heads_per_chunk, k_slab_size_512). The
    # predicate prefers fast > N-only > K-only > both, picking max-SBUF among
    # ties. Fast path corresponds to (n_heads, H_512_TILE_COUNT).
    _H_512 = H // (P_MAX * H_PACK)
    heads_per_chunk, K_SLAB_SIZE_512 = _v4_compute_dispatch(
        n_heads=n_heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_lora_rank=qk_lora_rank,
        kv_lora_rank=kv_lora_rank,
        H=H,
        sbuf_budget_bytes=_sbuf_budget,
    )
    num_head_chunks = n_heads // heads_per_chunk
    chunk_q_out_dim = heads_per_chunk * head_dim
    NUM_K_SLABS = _H_512 // K_SLAB_SIZE_512
    USE_K_SLAB = NUM_K_SLABS > 1
    _NUM_WQKV_SLAB_BUFFERS = 1

    sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=nl.tile_size.total_available_sbuf_size,
        use_auto_alloc=False,
        logger=get_logger("mla_v4"),
    )
    sbm.open_scope()

    # Tiling
    S_TILE_SIZE = s_tile_pad = P_MAX
    num_S_tiles = div_ceil(S_shard, S_TILE_SIZE)
    H_512_TILE_COUNT = H // (P_MAX * H_PACK)
    QK_LORA_128_TILE_COUNT = qk_lora_rank // (P_MAX * H_PACK)

    # Load norm weights
    q_norm_gamma_sb = _load_norm_weights_for_mx(q_norm_gamma_hbm, qk_lora_rank, sbm, name="q_norm_gamma")

    # Pre-load KV gamma once (reused across all tiles)
    kv_gamma_sb = sbm.alloc_stack((P_MAX, kv_dim), dtype=nl.bfloat16, buffer=nl.sbuf, name="kv_gamma")
    nisa.dma_copy(
        dst=kv_gamma_sb[0:P_MAX, 0:kv_dim],
        src=kv_norm_gamma_hbm.reshape((kv_dim,)).ap(
            pattern=[[0, P_MAX], [1, kv_dim]],
            offset=0,
        ),
        dge_mode=dge_mode.swdge,
    )

    norm_eps_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="norm_eps")
    nisa.memset(dst=norm_eps_sb, value=norm_eps)
    zero_bias_sb = sbm.alloc_stack((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.sbuf, name="zero_bias")
    nisa.memset(dst=zero_bias_sb, value=0.0)

    # Load weights: fused wqkv (qr columns pre-swizzled). wq_b is loaded inside
    # the head-chunk loop when chunking; in the single-chunk fast path the load
    # still happens once at full width with no extra DMAs.
    # K-slab dispatch: when USE_K_SLAB, wqkv is loaded per-slab inside the
    # per-S-tile loop instead of in the prologue.
    if not USE_K_SLAB:
        wqkv_sb, wqkv_scale_sb = _load_mx_weights(wqkv_hbm, wqkv_scale_hbm, H, qkv_out_dim, sbm, name="wqkv")
    if num_head_chunks == 1:
        wq_b_sb, wq_b_scale_sb = _load_mx_weights(wq_b_hbm, wq_b_scale_hbm, qk_lora_rank, q_out_dim, sbm, name="wq_b")

    SCALE_BLOCK = 128
    qkv_out_dim_padded = ((qkv_out_dim + SCALE_BLOCK - 1) // SCALE_BLOCK) * SCALE_BLOCK
    if USE_K_SLAB:
        # Slabbed path: allocate slab-sized buffers ONCE up front and reuse
        # them across all S-tiles. Each S-tile drives a slab loop that
        # DMA-loads a K-slice of wqkv into these buffers and runs that slab's
        # matmul; PSUM accumulates implicitly across slab iterations.
        wqkv_slab_bufs = []
        wqkv_scale_slab_bufs = []
        for buf_idx in range(_NUM_WQKV_SLAB_BUFFERS):
            wqkv_slab_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, K_SLAB_SIZE_512, qkv_out_dim),
                    dtype=nl.float8_e4m3fn_x4,
                    buffer=nl.sbuf,
                    name=f"wqkv_slab_buf_{buf_idx}",
                )
            )
            wqkv_scale_slab_bufs.append(
                sbm.alloc_stack(
                    (P_MAX, K_SLAB_SIZE_512, qkv_out_dim_padded),
                    dtype=nl.uint8,
                    buffer=nl.sbuf,
                    name=f"wqkv_scale_slab_buf_{buf_idx}",
                )
            )

    # Double buffering: allocate 2 input buffers so DMA for tile N+1
    # overlaps with compute on tile N.
    NUM_S_BUFFERS = 2
    x_sb_bufs = []
    for buf_idx in range(NUM_S_BUFFERS):
        x_sb_bufs.append(
            sbm.alloc_stack(
                (H_TILE_SIZE, s_tile_pad, H_TILE_COUNT),
                dtype=x_hbm.dtype,
                buffer=nl.sbuf,
                align=32,
                name=f"x_sb_buf_{buf_idx}",
            )
        )

    # Double-buffer quantized input buffers so quantize_mx for tile N+1
    # can write to one buffer while nc_matmul_mx for tile N still reads
    # from the other.
    NUM_QTZ_BUFFERS = 2
    x_qtz_sb_bufs = []
    x_scale_sb_bufs = []
    for buf_idx in range(NUM_QTZ_BUFFERS):
        x_qtz_sb_bufs.append(
            sbm.alloc_stack(
                (P_MAX, H_512_TILE_COUNT, s_tile_pad),
                dtype=nl.float8_e4m3fn_x4,
                buffer=nl.sbuf,
                name=f"x_qtz_buf_{buf_idx}",
            )
        )
        x_scale_sb_bufs.append(
            sbm.alloc_stack(
                (P_MAX, H_512_TILE_COUNT, s_tile_pad),
                dtype=nl.uint8,
                buffer=nl.sbuf,
                name=f"x_scale_buf_{buf_idx}",
            )
        )
    hidden_for_transpose = x_hbm.reshape((S * H_TILE_COUNT, 1, 1, H_TILE_SIZE))

    for s_tile_idx in nl.affine_range(num_S_tiles):
        sbm.open_scope()
        sbm.set_name_prefix(f"t{s_tile_idx}_")

        s_tile_local_offset = s_tile_idx * S_TILE_SIZE
        s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)
        s_tile_global_offset = S_shard_offset + s_tile_local_offset

        # ============================================================
        # STEP 1: Load & quantize input (double-buffered)
        # ============================================================
        buf_idx = s_tile_idx % NUM_S_BUFFERS
        x_sb = x_sb_bufs[buf_idx]
        hidden_sb_flat = x_sb.reshape((H_TILE_SIZE, 1, 1, s_tile_pad * H_TILE_COUNT))
        nisa.dma_transpose(
            dst=hidden_sb_flat[0:H_TILE_SIZE, 0:1, 0:1, 0 : s_tile_sz * H_TILE_COUNT],
            src=hidden_for_transpose[
                s_tile_global_offset * H_TILE_COUNT : (s_tile_global_offset + s_tile_sz) * H_TILE_COUNT,
                0:1,
                0:1,
                0:H_TILE_SIZE,
            ],
        )
        x_transposed_sb = _layout_adapter_sb(src=x_sb, n_prgs=1, prg_id=0).reshape(
            (P_MAX, H_512_TILE_COUNT, s_tile_pad * H_PACK)
        )
        qtz_buf_idx = s_tile_idx % NUM_QTZ_BUFFERS
        x_qtz_sb = x_qtz_sb_bufs[qtz_buf_idx]
        x_scale_sb = x_scale_sb_bufs[qtz_buf_idx]
        nisa.quantize_mx(
            src=x_transposed_sb[0:P_MAX, 0:H_512_TILE_COUNT, 0 : s_tile_pad * H_PACK],
            dst=x_qtz_sb[0:P_MAX, 0:H_512_TILE_COUNT, 0:s_tile_pad],
            dst_scale=x_scale_sb[0:P_MAX, 0:H_512_TILE_COUNT, 0:s_tile_pad],
        )

        # ============================================================
        # STEP 2: Fused first projection - x @ wqkv -> split [qr, kv]
        # Fast path: single matmul, split during PSUM->SBUF copy.
        # K-slab path: stream wqkv from HBM in slabs; PSUM accumulates
        # across slabs; drain PSUM into qr_sb/kv_sb after the slab loop.
        # ============================================================
        if not USE_K_SLAB:
            qr_sb, kv_sb = _mx_matmul_split(
                x_qtz_sb,
                x_scale_sb,
                wqkv_sb,
                wqkv_scale_sb,
                H_512_TILE_COUNT,
                s_tile_pad,
                qkv_out_dim,
                [qk_lora_rank],
                sbm,
            )
        else:
            # Slabbed path: allocate PSUM banks, then loop over slabs
            # streaming wqkv from HBM. PSUM accumulates implicitly across
            # slabs (each call targets the same dst banks).
            F_MAX = 512
            PSUM_BANK_SIZE = _get_psum_bank_size()
            num_n_tiles = (qkv_out_dim + F_MAX - 1) // F_MAX
            output_psum = []
            for bank_id in range(num_n_tiles):
                output_psum.append(
                    nl.ndarray(
                        (P_MAX, F_MAX),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                        address=(0, bank_id * PSUM_BANK_SIZE),
                    )
                )

            for slab_id in range(NUM_K_SLABS):
                cur_buf_idx = slab_id % _NUM_WQKV_SLAB_BUFFERS
                slab_buf = wqkv_slab_bufs[cur_buf_idx]
                slab_scale_buf = wqkv_scale_slab_bufs[cur_buf_idx]
                _load_mx_weights_k_slab(
                    wqkv_hbm,
                    wqkv_scale_hbm,
                    slab_buf,
                    slab_scale_buf,
                    in_dim_full=H,
                    out_dim=qkv_out_dim,
                    k_tile_start=slab_id * K_SLAB_SIZE_512,
                    k_tile_count=K_SLAB_SIZE_512,
                    sbm=sbm,
                    name=f"v4_slab{slab_id}",
                )
                _mx_matmul_split_k_range(
                    input_qtz_sb=x_qtz_sb,
                    input_scale_sb=x_scale_sb,
                    weights_slab_sb=slab_buf,
                    weights_slab_scale_sb=slab_scale_buf,
                    k_tile_start=slab_id * K_SLAB_SIZE_512,
                    k_tile_count=K_SLAB_SIZE_512,
                    m_dim=s_tile_pad,
                    n_dim=qkv_out_dim,
                    output_psum=output_psum,
                )

            # Drain PSUM into qr_sb (qk_lora_rank) and kv_sb (kv_dim).
            qr_sb = sbm.alloc_stack(
                (P_MAX, qk_lora_rank),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name="matmul_split_out_0",
            )
            kv_sb = sbm.alloc_stack(
                (P_MAX, kv_dim),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name="matmul_split_out_1",
            )
            # Split-0: PSUM[0 : qk_lora_rank] -> qr_sb
            width = qk_lora_rank
            col = 0
            while col < width:
                bank_idx, bank_offset = divmod(col, F_MAX)
                copy_width = min(F_MAX - bank_offset, width - col)
                nisa.tensor_copy(
                    dst=qr_sb[0:s_tile_pad, nl.ds(col, copy_width)],
                    src=output_psum[bank_idx][0:s_tile_pad, nl.ds(bank_offset, copy_width)],
                    engine=nisa.scalar_engine,
                )
                col += copy_width
            # Split-1: PSUM[qk_lora_rank : qkv_out_dim] -> kv_sb
            col = 0
            while col < kv_dim:
                global_col = qk_lora_rank + col
                bank_idx, bank_offset = divmod(global_col, F_MAX)
                copy_width = min(F_MAX - bank_offset, kv_dim - col)
                nisa.tensor_copy(
                    dst=kv_sb[0:s_tile_pad, nl.ds(col, copy_width)],
                    src=output_psum[bank_idx][0:s_tile_pad, nl.ds(bank_offset, copy_width)],
                    engine=nisa.scalar_engine,
                )
                col += copy_width

        # Pre-wq_b RMSNorm. rsqrt must be applied row-wise to qr_sb (row = seq
        # position) BEFORE transpose. Fusing rsqrt into the gamma multiply
        # inside the transpose would scale by the wrong axis: after nc_transpose
        # the partition dim is h-element, not seq, so a per-partition rsqrt
        # scalar would mix sequence positions into hidden elements.
        _apply_rms_norm_inplace(
            qr_sb,
            zero_bias_sb,
            norm_eps_sb,
            s_tile_pad,
            qk_lora_rank,
            sbm,
            name="qr_rms",
        )

        qr_transposed_sb = _transpose_preswizzled_for_mx(
            qr_sb,
            s_tile_pad,
            qk_lora_rank,
            QK_LORA_128_TILE_COUNT,
            sbm,
            gamma_sb=q_norm_gamma_sb,
            name="qr_transpose",
        )
        qr_qtz_sb, qr_scale_sb = _quantize_mx(qr_transposed_sb, QK_LORA_128_TILE_COUNT, s_tile_pad, sbm, name="qr_qtz")

        # ============================================================
        # Post-projection work. `num_head_chunks` is a compile-time
        # Python int, so this branch is resolved at trace time and only
        # one path is emitted. The fast path keeps everything in a single
        # linear sequence (no inner sbm scope, no split KV body) so the
        # compiler can interleave Q stage 2 with KV norm/RoPE; the
        # chunked path uses a per-chunk loop with its own scoping for
        # SBUF reuse, at the cost of breaking that interleaving.
        # ============================================================
        if num_head_chunks == 1:
            q_sb = _mx_matmul(
                qr_qtz_sb,
                qr_scale_sb,
                wq_b_sb,
                wq_b_scale_sb,
                QK_LORA_128_TILE_COUNT,
                s_tile_pad,
                q_out_dim,
                sbm,
                name="q_matmul",
            )
            _v4_apply_q_post_rsqrt_norm(
                q_sb,
                zero_bias_sb,
                norm_eps_sb,
                sbm,
                s_tile_pad,
                n_heads,
                head_dim,
            )
            # KV norm interleaved between Q stage 2 and Q RoPE so the
            # compiler can overlap them; order matters for the schedule.
            _v4_apply_kv_norm_inplace(
                kv_sb,
                kv_gamma_sb,
                zero_bias_sb,
                norm_eps_sb,
                sbm,
                s_tile_sz,
                kv_dim,
            )
            rope_offset = s_tile_global_offset * qk_rope_head_dim
            cos_sb, sin_sb = _v4_load_rope_caches(
                cos_cache_hbm,
                sin_cache_hbm,
                sbm,
                s_tile_sz,
                qk_rope_head_dim,
                rope_offset,
            )
            _v4_apply_q_rope_inplace(q_sb, cos_sb, sin_sb, sbm, n_heads, head_dim, qk_rope_head_dim)
            _v4_apply_kv_rope_inplace(kv_sb, cos_sb, sin_sb, sbm, kv_dim, qk_rope_head_dim)

            # ----- Fast-path Q DMA-out (single big copy) -----
            nisa.dma_copy(
                dst=q_hbm.ap(
                    pattern=[[n_heads * head_dim, s_tile_sz], [1, n_heads * head_dim]],
                    offset=s_tile_global_offset * n_heads * head_dim,
                ),
                src=q_sb[0:s_tile_sz, 0 : n_heads * head_dim],
                dge_mode=dge_mode.swdge,
            )

        else:
            # ----- Chunked path: load RoPE caches early, run KV first
            # so its scratch lives at a low SBUF address, then loop over
            # head chunks streaming wq_b slabs. -----
            rope_offset = s_tile_global_offset * qk_rope_head_dim
            cos_sb, sin_sb = _v4_load_rope_caches(
                cos_cache_hbm,
                sin_cache_hbm,
                sbm,
                s_tile_sz,
                qk_rope_head_dim,
                rope_offset,
            )
            _v4_apply_kv_norm_inplace(
                kv_sb,
                kv_gamma_sb,
                zero_bias_sb,
                norm_eps_sb,
                sbm,
                s_tile_sz,
                kv_dim,
            )
            _v4_apply_kv_rope_inplace(kv_sb, cos_sb, sin_sb, sbm, kv_dim, qk_rope_head_dim)

            # ============================================================
            # Q Stage 2 + post-rsqrt-norm + RoPE, tiled by head chunks.
            # wq_b weights, q matmul output, and Q rope scratch all scale
            # with heads_per_chunk instead of n_heads.
            # ============================================================
            for i_chunk in range(num_head_chunks):
                sbm.open_scope()
                sbm.set_name_prefix(f"t{s_tile_idx}_qc{i_chunk}_")
                head_chunk_offset = i_chunk * heads_per_chunk
                col_offset = head_chunk_offset * head_dim

                chunk_wq_b_sb, chunk_wq_b_scale_sb = _load_mx_weights(
                    wq_b_hbm,
                    wq_b_scale_hbm,
                    in_dim=qk_lora_rank,
                    out_dim=chunk_q_out_dim,
                    sbm=sbm,
                    name=f"wq_b_c{i_chunk}",
                    full_out_dim=q_out_dim,
                    out_col_offset=col_offset,
                )

                q_chunk_sb = _mx_matmul(
                    qr_qtz_sb,
                    qr_scale_sb,
                    chunk_wq_b_sb,
                    chunk_wq_b_scale_sb,
                    QK_LORA_128_TILE_COUNT,
                    s_tile_pad,
                    chunk_q_out_dim,
                    sbm,
                    name="q_matmul",
                )
                _v4_apply_q_post_rsqrt_norm(
                    q_chunk_sb,
                    zero_bias_sb,
                    norm_eps_sb,
                    sbm,
                    s_tile_pad,
                    heads_per_chunk,
                    head_dim,
                )
                _v4_apply_q_rope_inplace(
                    q_chunk_sb,
                    cos_sb,
                    sin_sb,
                    sbm,
                    heads_per_chunk,
                    head_dim,
                    qk_rope_head_dim,
                )

                # ----- Store this head-chunk slice of Q to HBM -----
                nisa.dma_copy(
                    dst=q_hbm.ap(
                        pattern=[[n_heads * head_dim, s_tile_sz], [1, chunk_q_out_dim]],
                        offset=s_tile_global_offset * n_heads * head_dim + col_offset,
                    ),
                    src=q_chunk_sb[0:s_tile_sz, 0:chunk_q_out_dim],
                    dge_mode=dge_mode.swdge,
                )

                sbm.close_scope()

        # ============================================================
        # Store KV to HBM. Q is already DMA'd: fast path issues a single
        # big copy inside the fast-path block; chunked path issues a
        # per-chunk DMA inside the loop.
        # ============================================================
        nisa.dma_copy(
            dst=kv_hbm.ap(
                pattern=[[kv_dim, s_tile_sz], [1, kv_dim]],
                offset=s_tile_global_offset * kv_dim,
            ),
            src=kv_sb[0:s_tile_sz, 0:kv_dim],
            dge_mode=dge_mode.swdge,
        )

        sbm.close_scope()

    sbm.close_scope()
    return q_hbm, kv_hbm
