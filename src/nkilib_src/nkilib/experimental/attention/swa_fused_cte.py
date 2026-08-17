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

"""Fused GPT-OSS sliding-window-attention (SWA) block kernel.

Fuses, for a single SWA layer of a TP-sharded GPT-OSS model, the chain
QKV projection -> RoPE + Q-scale -> sliding-window attention (per-head sink +
block KV cache prior) -> output projection, into one kernel.

Head-parallel across the two LNC cores: each core owns half of the rank's q-heads
and its matching kv-head(s), runs QKV/RoPE/SWA fully locally, and produces a
partial [B, S, H] output projection (contracting over only its heads). The two
partials are summed across cores with a single sendrecv + add reduction.

All SBUF tensors are allocated through a ModularAllocator; PSUM tiles use raw
nl.ndarray(buffer=nl.psum). The allocator address is saved/restored per batch so
per-batch scratch is freed between iterations.

See swa_fused_cte_torch.py for the executable spec and the exact layout conventions.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa import reduce_cmd, sendrecv

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.modular_allocator import ModularAllocator
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

_FLOAT32_MIN = -3.4028235e38  # exact np.finfo(np.float32).min (range_select requires this exact value)
_PMAX = 128
_FP8_DTYPES = (nl.float8_e4m3, nl.float8_e4m3fn, nl.float8_e5m2)
# Width of the wide projection+RoPE tile: Q/K projection + RoPE run at this free width to amortize
# their per-op fixed overhead (fewer, wider vector/scalar ops); V-proj + attention + OP + reduce stay
# per-128 sub-tile (tokens on the partition axis, capped at 128). 128 = no widening.
# 256 is a good balance to have the overlapping perf gain with small sbuf pressure
_PROJ_TILE_W = 256


def _fp8_max(dtype):
    """Max representable positive value for the FP8 quantize clamp. e4m3fn (OCP) tops out at 448;
    the e4m3 (non-OCP) and e5m2 variants at 240 (the safe bound across trn2/trn3)."""
    return 448.0 if dtype == nl.float8_e4m3fn else 240.0


def _proj_tile_width(S):
    """Wide-tile width, reduced to the largest power-of-2 that still divides S (so the wide tile splits
    into whole 128-subtiles and tiles S evenly). Falls back to 128 (no widening) for short S like the
    tiny sim config."""
    tw = _PROJ_TILE_W
    while tw > _PMAX and S % tw != 0:
        tw //= 2
    return tw if S % tw == 0 else _PMAX


def _h_tile_size(H, ht):
    """Size of H-contraction tile ``ht`` (last tile clamped when H is not a multiple of 128)."""
    return min(_PMAX, H - ht * _PMAX)


def _log2_int(n):
    """log2 of a power-of-2 int, in pure Python (parser-safe; avoids method calls on traced args)."""
    s = 0
    while n > 1:
        n //= 2
        s += 1
    return s


def _qkv_head_split(num_q_heads, num_kv_heads, num_shard, shard_id):
    """Split q/kv heads across the LNC cores.

    Returns (q_start, q_count, kv_start, kv_count) for this core.

    TP4 rank: (16 q, 2 kv) -> 8 q + 1 kv per core (distinct kv head per core).
    TP8 rank: (8 q, 1 kv)  -> 4 q per core, kv head 0 replicated to both cores.
    Each core's q-heads attend only its local kv head(s) (GQA mapping preserved).
    """
    kernel_assert(num_q_heads % num_shard == 0, "num_q_heads must divide across shards")
    q_count = num_q_heads // num_shard
    q_start = shard_id * q_count
    if num_kv_heads >= num_shard:
        kernel_assert(num_kv_heads % num_shard == 0, "num_kv_heads must divide across shards")
        kv_count = num_kv_heads // num_shard
        kv_start = shard_id * kv_count
    else:
        # Fewer kv heads than cores (e.g. TP8: 1 kv, 2 cores): replicate kv head 0.
        kv_count = num_kv_heads
        kv_start = 0
    return q_start, q_count, kv_start, kv_count


@nki.jit(experimental_flags="skip-non-top-level-shared-hbm-check")
def swa_fused_cte(
    hidden_states: nl.NkiTensor,  # [B, S, H] bf16
    qkv_weight: nl.NkiTensor,  # [H, I]  I=(num_q_heads + 2*num_kv_heads)*d_head
    op_weight: nl.NkiTensor,  # [num_q_heads*d_head, H]
    k_cache: nl.NkiTensor,  # bf16: [num_blocks, num_kv_heads, block_size, d_head] (post-RoPE K);
    #                       FP8:  [num_blocks, num_kv_heads, block_size // 2, d_head, 2] (packed)
    v_cache: nl.NkiTensor,  # bf16: [num_blocks, num_kv_heads, block_size, d_head] (plain V);
    #                       FP8:  [num_blocks, num_kv_heads, block_size, d_head] fp8 (UNPACKED, token-major)
    block_tables: nl.NkiTensor,  # [B, max_blocks_per_seq] int32 (logical->physical block map)
    cos_cache: nl.NkiTensor,  # [B, S, d_head] fp32 (active-token RoPE cos, pre-duplicated halves)
    sin_cache: nl.NkiTensor,  # [B, S, d_head] fp32
    sink: nl.NkiTensor,  # [B, num_q_heads] fp32 per-head sink logit
    prior_tokens: nl.NkiTensor,  # [1, 1] int32 runtime valid-prior length (0..sliding_window)
    qkv_bias: nl.NkiTensor,  # [1, I] QKV projection bias (added before RoPE/scale)
    op_bias: nl.NkiTensor,  # [1, H] output-projection bias (added once to the cross-core-summed output)
    scale: float = 1.0,
    sliding_window: int = 128,
    block_size: int = 128,
    num_q_heads: int = 16,
    num_kv_heads: int = 2,
    d_head: int = 64,
    k_scale: nl.NkiTensor = None,  # [1,1] fp32 per-tensor K dequant scale; required iff k_cache is FP8
    v_scale: nl.NkiTensor = None,  # [1,1] fp32 per-tensor V dequant scale; required iff v_cache is FP8
):
    """Fused GPT-OSS SWA block. Returns (out [B,S,H], k_cache, v_cache) with caches updated in place.

    Packed-FP8 KV cache: when k_cache/v_cache are an FP8 dtype, the K cache is stored PACKED as
    (num_blocks, num_kv_heads, block_size // 2, d_head, 2) -- two consecutive tokens in the trailing
    length-2 axis so the K cache views as bf16 (2 fp8 = 1 bf16 width) for DMA. The V cache is UNPACKED,
    token-major (num_blocks, num_kv_heads, block_size, d_head) fp8 -- same layout as the bf16 V cache,
    only the dtype differs (so V is loaded straight as fp8 with no implicit cast, then dequantized). The
    prior window is dequantized to bf16 on load (carry stays bf16, attention math unchanged);
    freshly-computed K is quantized + packed and V is quantized (token-major, no pack) on the write-back
    scatter. Dequant/quant use the per-tensor static k_scale/v_scale.
    """
    B, S, H = hidden_states.shape

    kernel_assert(d_head <= _PMAX, "d_head must be <= 128")
    kernel_assert(S % _PMAX == 0, "S must be a multiple of 128")
    kernel_assert(H % _PMAX == 0 or H % d_head == 0, "H must be a multiple of 128 or d_head")
    # The attention tile and SWA window are both _PMAX (=128) tokens. The paged KV cache may use a
    # SMALLER block_size (32/64/128): a 128-token tile/window then spans bpt = _PMAX//block_size
    # physical blocks. block_size must divide _PMAX and the window must be exactly one tile (W==_PMAX
    # so the carry is one 128-row tile, spanning bpt cache blocks).
    kernel_assert(_PMAX % block_size == 0, "block_size must divide 128")
    kernel_assert(sliding_window == _PMAX, "v1 assumes sliding_window == 128 (one-tile carry)")

    # Packed-FP8 KV cache, inferred from the cache dtype. The pack stores 2 tokens per bf16-width
    # element; a sub-tile block_size (32/64) is supported the same way as the bf16 path -- a 128-token
    # tile/window spans bpt = 128//block_size physical blocks, each block_size//2 packed rows. block_size
    # must be even (>=2) so the token-pair pack divides the block.
    fp8_packed = k_cache.dtype in _FP8_DTYPES
    if fp8_packed:
        kernel_assert(v_cache.dtype in _FP8_DTYPES, "FP8 packed cache requires both k_cache and v_cache FP8")
        kernel_assert(block_size % 2 == 0, "FP8 packed cache requires an even block_size")

    _, num_shard, shard_id = get_verified_program_sharding_info("swa_fused_cte", max_sharding=2)
    q_start, q_count, kv_start, kv_count = _qkv_head_split(num_q_heads, num_kv_heads, num_shard, shard_id)

    out = nl.ndarray(shape=(B, S, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)

    cfg = dict(
        H=H,
        S=S,
        d_head=d_head,
        half_d=d_head // 2,
        q_dim=num_q_heads * d_head,
        kv_dim=num_kv_heads * d_head,
        group_size=q_count // kv_count if kv_count > 0 else q_count,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        num_s_tiles=S // _PMAX,
        num_h_tiles=(H + _PMAX - 1) // _PMAX,
        block_size=block_size,
        sliding_window=sliding_window,
        scale=scale,
        bpt=_PMAX // block_size,  # physical cache blocks per 128-token tile/window
        fp8_packed=fp8_packed,
        num_shard=num_shard,
        shard_id=shard_id,
        q_start=q_start,
        q_count=q_count,
        kv_start=kv_start,
        kv_count=kv_count,
    )

    allocator = ModularAllocator(initial_address=0)
    for b in range(B):
        base_addr = allocator.get_current_address()
        _swa_fused_one_batch(
            b,
            hidden_states,
            qkv_weight,
            op_weight,
            k_cache,
            v_cache,
            block_tables,
            cos_cache,
            sin_cache,
            sink,
            prior_tokens,
            qkv_bias,
            op_bias,
            out,
            k_scale,
            v_scale,
            allocator,
            cfg,
        )
        allocator.set_current_address(base_addr)

    # Return the attention output plus the in-place-updated caches (verified outputs).
    return out, k_cache, v_cache


def _swa_fused_one_batch(
    b,
    hidden_states,
    qkv_weight,
    op_weight,
    k_cache,
    v_cache,
    block_tables,
    cos_cache,
    sin_cache,
    sink,
    prior_tokens,
    qkv_bias,
    op_bias,
    out,
    k_scale,
    v_scale,
    alloc,
    cfg,
):
    """Process one batch item for this core's heads, tiling the whole pipeline over S.

    Per 128-token S-tile: load+transpose hidden -> project Q/K/V -> RoPE+Q-scale -> scatter K/V
    to cache -> SWA attend (current tile + carried previous-tile K/V for the W<=128 window, or the
    loaded prior block at st==0) -> output projection -> cross-core reduce -> write out.

    Only one S-tile of activations is live at a time, so SBUF stays within budget at large S.
    Projection weights (small) stay resident across all S-tiles.
    """
    dt = hidden_states.dtype
    fp32 = nl.float32
    H, S, d_head, half_d = cfg["H"], cfg["S"], cfg["d_head"], cfg["half_d"]
    q_dim, kv_dim = cfg["q_dim"], cfg["kv_dim"]
    num_s_tiles, num_h_tiles = cfg["num_s_tiles"], cfg["num_h_tiles"]
    q_start, q_count, kv_start, kv_count = cfg["q_start"], cfg["q_count"], cfg["kv_start"], cfg["kv_count"]
    group_size = cfg["group_size"]
    block_size, sliding_window = cfg["block_size"], cfg["sliding_window"]
    bpt = cfg["bpt"]  # physical cache blocks spanned by one 128-token tile/window
    fp8_packed = cfg["fp8_packed"]
    num_shard, shard_id, scale = cfg["num_shard"], cfg["shard_id"], cfg["scale"]
    # Block layout is fully runtime: prior occupies logical blocks [0, n_prior_blocks); the SWA
    # window is the LAST prior block; active tile st -> logical block n_prior_blocks + st. All block
    # indices are resolved at runtime via block_tables, so prior_tokens (incl. > W) is handled
    # without compile-time baking.

    # ---- Resident weights (loaded once, reused across all S-tiles) ---------
    q_col0 = q_start * d_head
    k_col0 = q_dim + kv_start * d_head
    v_col0 = q_dim + kv_dim + kv_start * d_head
    qw_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, num_h_tiles, q_count * d_head), dtype=dt, align_to=32)
    kw_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, num_h_tiles, kv_count * d_head), dtype=dt, align_to=32)
    vw_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, num_h_tiles, kv_count * d_head), dtype=dt, align_to=32)
    # Resident QKV-weight load, ordered + queue-split to unblock the first tile's projection ASAP:
    #  - PROJECTION-CONSUMPTION ORDER: the Q-projection is the first Tensor consumer and accumulates
    #    over h-tiles in ht order (consuming qw[0],qw[1],...), and q-weight is ~80% of the bytes on
    #    TP4. So load ALL qw h-tiles FIRST, then kw, then vw -- the q-proj no longer waits behind the
    #    k/v slices it doesn't need yet (the old ht-major q,k,v,q,k,v order stalled qw[ht] behind
    #    kw/vw[ht-1]).
    #  - TWO PARALLEL DMA QUEUES: alternate each h-tile between hwdge (Sync -> qSyncDynamicHW) and
    #    swdge (GpSimd -> qGpSimdDynamic); the default single swDGE queue serializes all 23 h-tiles
    #    (~45us) and Tensor starves ~25% for the first ~90us.
    for ht in range(num_h_tiles):
        hsz = _h_tile_size(H, ht)
        dge = nisa.dge_mode.hwdge if (ht % 2 == 0) else nisa.dge_mode.swdge
        nisa.dma_copy(
            dst=qw_sb[:hsz, ht, :],
            src=qkv_weight[ht * _PMAX : ht * _PMAX + hsz, q_col0 : q_col0 + q_count * d_head],
            dge_mode=dge,
        )
    for ht in range(num_h_tiles):
        hsz = _h_tile_size(H, ht)
        dge = nisa.dge_mode.hwdge if (ht % 2 == 0) else nisa.dge_mode.swdge
        nisa.dma_copy(
            dst=kw_sb[:hsz, ht, :],
            src=qkv_weight[ht * _PMAX : ht * _PMAX + hsz, k_col0 : k_col0 + kv_count * d_head],
            dge_mode=dge,
        )
        nisa.dma_copy(
            dst=vw_sb[:hsz, ht, :],
            src=qkv_weight[ht * _PMAX : ht * _PMAX + hsz, v_col0 : v_col0 + kv_count * d_head],
            dge_mode=dge,
        )
    # OP weight stored stacked-by-pair: pair p holds head 2p in partition rows [0:d_head] and head
    # 2p+1 in [d_head:2*d_head]. The two heads' attention outputs land stacked the same way, so the
    # output projection is ONE d=128 matmul per pair (the cross-head sum happens inside the
    # contraction) instead of two d=64 matmuls + a cross-head add. q_count is even (TP4: 8, TP8: 4).
    kernel_assert(q_count % 2 == 0, "stacked-pair OP needs an even q_count (head pairing)")
    n_pairs = q_count // 2
    opw_sb = alloc.alloc_sbuf_tensor(shape=(2 * d_head, n_pairs, H), dtype=dt, align_to=32)
    for p in range(n_pairs):
        row_lo = (q_start + 2 * p) * d_head
        row_hi = (q_start + 2 * p + 1) * d_head
        nisa.dma_copy(dst=opw_sb[0:d_head, p, :], src=op_weight[row_lo : row_lo + d_head, :])
        nisa.dma_copy(dst=opw_sb[d_head : 2 * d_head, p, :], src=op_weight[row_hi : row_hi + d_head, :])

    # ---- Resident QKV bias (GPT-OSS attention_bias=True), sliced per core ----
    # qkv_bias [I] = [Q | K | V], added to the projection BEFORE RoPE/scale. Q/K are d-major so their
    # bias is partition-major: load d-MAJOR via a strided DMA (partition stride 1 over d, free stride d
    # over heads) -> [d, n_heads], avoiding the 128-wide nc_transpose cap. qb is then remapped to the
    # pair-stacked layout (head 2p -> qb_sb[0:d,p], 2p+1 -> qb_sb[d:2d,p]). V is S-major (d on free),
    # so vb stays a [1, kv*d] row broadcast over the token partitions.
    qb_sb = alloc.alloc_sbuf_tensor(shape=(2 * d_head, n_pairs), dtype=fp32, align_to=32)
    kb_sb = alloc.alloc_sbuf_tensor(shape=(d_head, kv_count), dtype=fp32, align_to=32)
    vb_row = alloc.alloc_sbuf_tensor(shape=(1, kv_count * d_head), dtype=fp32, align_to=32)
    # V is token-major (d on free); its bias is added with a tensor_tensor whose rhs must match the
    # 128-token partition dim (HW rejects partition-dim-1 broadcast), so broadcast vb to [128, kv*d].
    vb_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, kv_count * d_head), dtype=fp32, align_to=32)
    qb_dmaj = alloc.alloc_sbuf_tensor(shape=(d_head, q_count), dtype=fp32, align_to=32)
    nisa.dma_copy(dst=qb_dmaj[:, :], src=qkv_bias.ap(pattern=[[1, d_head], [d_head, q_count]], offset=q_col0))
    # Pre-scale the Q-bias by `scale` during the pair-stacked remap (folded into the copy at zero extra
    # op cost). This lets the projection PSUM->SBUF fold use the ACT-legal (multiply, add) op order:
    # (psum + qb) * scale == psum * scale + qb * scale (fp32-identical). qb_sb is built once and read
    # only at the two projection sites, so scaling here cannot double-apply.
    for p in range(n_pairs):
        nisa.tensor_scalar(
            dst=qb_sb[0:d_head, p : p + 1],
            data=qb_dmaj[:, 2 * p : 2 * p + 1],
            op0=nl.multiply,
            operand0=float(scale),
            engine=nisa.scalar_engine,
        )
        nisa.tensor_scalar(
            dst=qb_sb[d_head : 2 * d_head, p : p + 1],
            data=qb_dmaj[:, 2 * p + 1 : 2 * p + 2],
            op0=nl.multiply,
            operand0=float(scale),
            engine=nisa.scalar_engine,
        )
    nisa.dma_copy(dst=kb_sb[:, :], src=qkv_bias.ap(pattern=[[1, d_head], [d_head, kv_count]], offset=k_col0))
    nisa.dma_copy(dst=vb_row[0:1, :], src=qkv_bias[0:1, v_col0 : v_col0 + kv_count * d_head])
    stream_shuffle_broadcast(src=vb_row[0:1, :], dst=vb_sb[:, :])

    # ---- Resident output-projection bias [H], broadcast to [128, H] once per batch ----
    # OP is head-parallel (reduce-scatter): each core sums only its heads, so op_bias must be added
    # EXACTLY ONCE to the final cross-core-summed output (in _reduce_and_write_tile), NOT per-core/
    # per-pair (that would double-count). Broadcast to [128, H] so it can be a dma_compute addend
    # alongside the two partials (dma_compute needs matching [128, half_H] operands).
    opb_sb = alloc.alloc_sbuf_tensor(shape=(1, H), dtype=fp32, align_to=32)
    nisa.dma_copy(dst=opb_sb[0:1, :], src=op_bias[0:1, 0:H])
    opb_bc = alloc.alloc_sbuf_tensor(shape=(_PMAX, H), dtype=fp32, align_to=32)
    stream_shuffle_broadcast(src=opb_sb[0:1, :], dst=opb_bc[:, :])

    # ---- FP8 KV-cache scales (resident): dequant scale on the prior load, inverse for write-back
    # quantize. k_scale/v_scale are per-tensor [1,1] fp32. A per-partition tensor_scalar needs the
    # scalar operand broadcast over the destination's partitions, so build [_PMAX, 1] columns (slice
    # [:P] per op). ----
    k_scale_sb = v_scale_sb = inv_k_scale_sb = inv_v_scale_sb = None
    k_fp8_max = _fp8_max(k_cache.dtype) if fp8_packed else 0.0
    if fp8_packed:
        k_scale_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
        v_scale_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
        nisa.dma_copy(dst=k_scale_sb[0:1, 0:1], src=k_scale[0:1, 0:1])
        nisa.dma_copy(dst=v_scale_sb[0:1, 0:1], src=v_scale[0:1, 0:1])
        stream_shuffle_broadcast(src=k_scale_sb[0:1, :], dst=k_scale_sb[:, :])
        stream_shuffle_broadcast(src=v_scale_sb[0:1, :], dst=v_scale_sb[:, :])
        inv_k_scale_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
        inv_v_scale_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
        nisa.reciprocal(inv_k_scale_sb[:, 0:1], k_scale_sb[:, 0:1])
        nisa.reciprocal(inv_v_scale_sb[:, 0:1], v_scale_sb[:, 0:1])

    # ---- Runtime prior geometry -------------------------------------------
    # prior_tokens is block-aligned (multiple of block_size, possibly > W). The full prior occupies
    # logical blocks [0, n_prior_blocks); SWA needs its LAST W (=128) tokens, which span the last
    # bpt = 128/block_size logical blocks [n_prior_blocks - bpt, n_prior_blocks). Active tokens follow
    # at logical blocks n_prior_blocks .. (active tile st -> blocks [n_prior_blocks + st*bpt, ...+bpt)).
    # n_prior_blocks = ceil(prior_tokens / block_size); with block-aligned prior = prior_tokens/block_size.
    pt_sb = alloc.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32, align_to=32)
    nisa.dma_copy(dst=pt_sb[0:1, 0:1], src=prior_tokens[0:1, 0:1])
    # n_prior_blocks = ceil(prior_tokens / block_size) = (prior + bs - 1) >> log2(bs).
    # Compute with integer add then right_shift (separate ops; can't mix arithmetic+bitvec). A
    # multiply-by-1/bs produces an fp32 intermediate that corrupts the value when later used as a
    # DGE scalar_offset index (proven via swa_scatter_probe on HW) — keep the index integer.
    block_size_shift = _log2_int(block_size)
    n_prior_blocks_sb = alloc.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32, align_to=32)
    nisa.tensor_scalar(dst=n_prior_blocks_sb[0:1, 0:1], data=pt_sb[0:1, 0:1], op0=nl.add, operand0=block_size - 1)
    nisa.tensor_scalar(
        dst=n_prior_blocks_sb[0:1, 0:1], data=n_prior_blocks_sb[0:1, 0:1], op0=nl.right_shift, operand0=block_size_shift
    )
    # win_lblk0 = max(n_prior_blocks - bpt, 0): FIRST of the bpt logical blocks holding the last W
    # prior tokens (the carry-load reads win_lblk0 .. win_lblk0+bpt-1 into the 128-row window).
    win_lblk_sb = alloc.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32, align_to=32)
    nisa.tensor_scalar(
        dst=win_lblk_sb[0:1, 0:1],
        data=n_prior_blocks_sb[0:1, 0:1],
        op0=nl.add,
        operand0=float(-bpt),
        op1=nl.maximum,
        operand1=0.0,
    )

    # ---- Previous-tile K/V carry (the sliding-window left context) ---------
    # prev_k is d-major [d_head, kv_count, 128]; prev_v is token-major [128, kv_count, d_head].
    prev_k = alloc.alloc_sbuf_tensor(shape=(d_head, kv_count, _PMAX), dtype=dt, align_to=32)
    prev_v = alloc.alloc_sbuf_tensor(shape=(_PMAX, kv_count, d_head), dtype=dt, align_to=32)
    # ALWAYS load the last prior block (via block_tables[b, win_lblk]) into the carry. When
    # prior_tokens < W, the leading (W-prior_tokens) slots are invalid and masked at runtime on tile
    # st==0; when prior_tokens==0 the whole window is masked off (-> no prior contribution).
    _load_prior_block_indirect(
        prev_k,
        prev_v,
        k_cache,
        v_cache,
        block_tables,
        b,
        win_lblk_sb,
        kv_start,
        kv_count,
        d_head,
        block_size,
        bpt,
        fp8_packed,
        k_scale_sb,
        v_scale_sb,
        alloc,
    )

    # Runtime window lower bound for the tile-0 prior mask: a prior window slot j (= absolute prior
    # position prior_tokens - W + j) is valid iff j >= (W - prior_tokens). For prior >= W the bound
    # is <= 0 so the full block is kept; for prior < W it masks the leading invalid slots.
    prior_lb = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.tensor_scalar(
        dst=prior_lb[0:1, 0:1], data=pt_sb[0:1, 0:1], op0=nl.subtract, operand0=float(sliding_window), reverse0=True
    )
    stream_shuffle_broadcast(src=prior_lb[0:1, :], dst=prior_lb[:, :])

    # ---- Per-head sink, broadcast ONCE per batch (resident across all S-tiles) ----
    # The sink is a per-(batch, head) constant; broadcast this core's q_count sinks to [128, q_count]
    # once so each head reads its column with no per-tile STREAM_SHUFFLE on the bottlenecked vector engine.
    sink_bcast = alloc.alloc_sbuf_tensor(shape=(_PMAX, q_count), dtype=fp32, align_to=32)
    nisa.dma_copy(dst=sink_bcast[0:1, :], src=sink[b : b + 1, q_start : q_start + q_count])
    stream_shuffle_broadcast(src=sink_bcast[0:1, :], dst=sink_bcast[:, :])
    # Pre-negated sink (resident): lets the per-tile sink-fold + negate collapse into one
    # scalar_tensor_tensor (neg_max = min(-row_max, -sink) = -max(row_max, sink)); see _attend_one_head.
    neg_sink_bcast = alloc.alloc_sbuf_tensor(shape=(_PMAX, q_count), dtype=fp32, align_to=32)
    nisa.tensor_scalar(dst=neg_sink_bcast[:, :], data=sink_bcast[:, :], op0=nl.multiply, operand0=-1.0)

    # ---- range_select band bounds for the combined mid-tile mask (built ONCE per batch) ----
    # _attend_combined masks a [128, 256] = [left | curr] score band. Laying the concat in key-time
    # order (left block first, then current) collapses the kept set into a SINGLE contiguous band:
    # query partition i attends to the 128 keys at concat columns [i+1, i+W], so ONE range_select
    # (Vector) masks it (vs two for the disjoint [curr|left] band) AND fuses the row-max reduce --
    # replacing the two affine_selects (GpSimd) + tensor_reduce(max), and (since it reads PSUM
    # directly) the score PSUM->SBUF copy. Why [i+1, i+W] in the 0..255 local column index:
    #   left col j (0..127): window keep j >= i+1            -> j in [i+1, 127]
    #   curr col 128+c     : causal keep c <= i (c=col-128)  -> col in [128, i+W]
    # which together are exactly [i+1, i+W]. Bounds are pure functions of the partition index ->
    # constant across pairs/tiles, built once.
    rs_band_lb = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)  # = i+1
    rs_band_ub = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)  # = i+W
    nisa.iota(rs_band_lb[:, :], pattern=[[0, 1]], offset=1, channel_multiplier=1)
    nisa.iota(rs_band_ub[:, :], pattern=[[0, 1]], offset=sliding_window, channel_multiplier=1)
    rs_bounds = (rs_band_lb, rs_band_ub)

    # kt_dma: K-scatter transpose on the DMA engine (TP4) vs nc_transpose (Tensor) + Scalar copy (TP8).
    # The only shard-dependent path left: TP4 is Tensor-bound so offloading the transpose to the idle
    # Sync engine wins, but the DMA/MBU-sensitive 4-head TP8 shard regresses ~+15% -> gate to TP4.
    # (The deferred-normalize, softmax reduce_regs fold, and bf16 RoPE cos/sin are unconditional: each
    # was re-measured to win on BOTH shards on the current Tensor-ized base.)
    kt_dma = q_count >= 8

    # Signed-swap matrix (resident): signed_swap(x) = swap_mat @ x = [-x_b1,+x_b0,-x_b3,+x_b2] (half_d
    # blocks, rotate's sign baked in) in ONE matmul vs 4 partition copies; sin then becomes UNSIGNED.
    swap_mat = _build_signed_swap_mat(d_head, dt, alloc)

    # WIDE PROJECTION+RoPE TILING: project Q/K and run their RoPE at width `tw` (>=128) to amortize the
    # per-op fixed overhead, then loop the inner 128-subtiles for V-proj + attention + OP + reduce
    # (those have tokens on the partition axis, capped at 128).
    tw = _proj_tile_width(S)
    n_sub = tw // _PMAX
    num_w_tiles = S // tw

    # ---- Double-buffered per-WIDE-TILE OP partial for software-pipelined cross-core reduction ----
    # Accumulate all n_sub subtiles of a wide tile into ONE [128, n_sub, H] buffer so the reduce-scatter
    # splits on the SUBTILE axis -> a SINGLE cross-core sendrecv per wide tile (2 barriers) instead of
    # one per 128-subtile. Deferred by one wide tile: op_wide[(wt-1)%2] reduces while wide tile wt
    # computes, so its sendrecv overlaps wt's V-proj+attention; the two (wt%2) buffers never clobber.
    op_wide_db = [
        alloc.alloc_sbuf_tensor(shape=(_PMAX, n_sub, H), dtype=fp32, align_to=32),
        alloc.alloc_sbuf_tensor(shape=(_PMAX, n_sub, H), dtype=fp32, align_to=32),
    ]
    # Persistent reduce scratch (one in-flight deferred reduce at a time): the peer's contribution to
    # THIS core's owned subtiles, [128, n_sub//2, H] (>= [128, H] so the n_sub==1 H-split path fits).
    recv_buf = alloc.alloc_sbuf_tensor(shape=(_PMAX, max(1, n_sub // 2), H), dtype=fp32, align_to=32)
    summed_buf = alloc.alloc_sbuf_tensor(shape=(_PMAX, H), dtype=out.dtype, align_to=32)

    # ---- TWO-TILE PIPELINE with EXPLICIT INTERLEAVE: project(wt+1) woven through attend(wt) ----------
    # Projection+RoPE is Tensor-bound; attention is Vector/Scalar-heavy and leaves Tensor idle. A
    # monolithic project-before-attend is neutral (the near-program-order scheduler won't hoist it into
    # attention's gaps), so instead EMIT the next tile's projection steps (one Q head-pair or the K step)
    # INTERLEAVED between the current tile's attention pairs, forcing each projection matmul into a
    # softmax gap. Double-buffered: attend reads (wt%2) while project writes (wt+1)%2.
    hidden_t_db = [
        alloc.alloc_sbuf_tensor(shape=(_PMAX, num_h_tiles, tw), dtype=dt, align_to=32),
        alloc.alloc_sbuf_tensor(shape=(_PMAX, num_h_tiles, tw), dtype=dt, align_to=32),
    ]
    cos_f_db = [
        alloc.alloc_sbuf_tensor(shape=(2 * d_head, tw), dtype=dt, align_to=32),
        alloc.alloc_sbuf_tensor(shape=(2 * d_head, tw), dtype=dt, align_to=32),
    ]
    sin_s_db = [
        alloc.alloc_sbuf_tensor(shape=(2 * d_head, tw), dtype=dt, align_to=32),
        alloc.alloc_sbuf_tensor(shape=(2 * d_head, tw), dtype=dt, align_to=32),
    ]
    q_packed_db = [
        alloc.alloc_sbuf_tensor(shape=(2 * d_head, n_pairs, tw), dtype=dt, align_to=32),
        alloc.alloc_sbuf_tensor(shape=(2 * d_head, n_pairs, tw), dtype=dt, align_to=32),
    ]
    k_tile_db = [
        alloc.alloc_sbuf_tensor(shape=(d_head, kv_count, tw), dtype=dt, align_to=32),
        alloc.alloc_sbuf_tensor(shape=(d_head, kv_count, tw), dtype=dt, align_to=32),
    ]

    # Prologue: load+project tile 0 fully before the loop attends it.
    _load_wide_inputs(
        0,
        hidden_t_db[0],
        cos_f_db[0],
        sin_s_db[0],
        hidden_states,
        cos_cache,
        sin_cache,
        b,
        tw,
        n_sub,
        num_h_tiles,
        H,
        d_head,
        dt,
        alloc,
    )
    _project_rope_from_inputs(
        q_packed_db[0],
        k_tile_db[0],
        hidden_t_db[0],
        cos_f_db[0],
        sin_s_db[0],
        qw_sb,
        kw_sb,
        qb_sb,
        kb_sb,
        swap_mat,
        tw,
        num_h_tiles,
        H,
        d_head,
        n_pairs,
        kv_count,
        scale,
        alloc,
    )

    # The next tile's projection has n_pairs Q-pair steps + 1 K step = n_proj_steps units, to be
    # spread across the current tile's n_sub*n_pairs attention-pair iterations (one proj step emitted
    # before roughly every (n_sub*n_pairs / n_proj_steps)-th attention pair).
    n_proj_steps = n_pairs + 1

    for wt in range(num_w_tiles):
        wt_save = alloc.get_current_address()
        op_wide = op_wide_db[wt % 2]
        hidden_t = hidden_t_db[wt % 2]
        q_packed = q_packed_db[wt % 2]
        k_tile = k_tile_db[wt % 2]
        nx = wt + 1
        do_proj = nx < num_w_tiles
        nq_packed = q_packed_db[nx % 2]
        nk_tile = k_tile_db[nx % 2]
        nhidden_t = hidden_t_db[nx % 2]
        ncos_f = cos_f_db[nx % 2]
        nsin_s = sin_s_db[nx % 2]
        proj_step = [0]  # mutable counter of next-tile projection steps emitted so far this tile
        # Bundle the next-tile projection params so _attend_combined can emit one step INSIDE the
        # softmax wait (between QK and P^T) via _emit_next_proj_step. None when there's no next tile.
        proj_ctx = (
            (
                proj_step,
                n_pairs,
                nq_packed,
                nhidden_t,
                qw_sb,
                qb_sb,
                ncos_f,
                nsin_s,
                swap_mat,
                nk_tile,
                kw_sb,
                kb_sb,
                d_head,
                num_h_tiles,
                H,
                tw,
                kv_count,
                scale,
                alloc,
            )
            if do_proj
            else None
        )

        # Deferred reduce+write of the PREVIOUS wide tile's OP partials, overlapping this wide tile's
        # compute. Deferred by one wide tile (double-buffered op_wide_db[wt%2]).
        if wt > 0:
            _reduce_and_write_wide(
                op_wide_db[(wt - 1) % 2],
                out,
                b,
                wt - 1,
                n_sub,
                H,
                num_shard,
                shard_id,
                out.dtype,
                alloc,
                recv_buf,
                summed_buf,
                opb_bc,
            )

        # PREFETCH tile wt+1's hidden inputs (Sync hidden-load, runs in this tile's idle Sync). The
        # projection MATMULS for wt+1 are NOT done here -- they are interleaved into the attend loop
        # below (see _emit_next_proj_step) so the scheduler places them in attention's Tensor gaps.
        if do_proj:
            _load_wide_inputs(
                nx,
                nhidden_t,
                ncos_f,
                nsin_s,
                hidden_states,
                cos_cache,
                sin_cache,
                b,
                tw,
                n_sub,
                num_h_tiles,
                H,
                d_head,
                dt,
                alloc,
            )

        # Persistent per-wide-tile buffers for BOTH subtiles' write-back. The cache scatter WRITES are
        # deferred + batched after the subtile loop, so the whole attention region stays DMA-free --
        # freeing those Sync slots for next-tile prefetch. bf16 path: token-major K (kt_wide) + plain V
        # (v_wide). FP8 path: K is quantized + PACKED before the subtile loop into k_pk_wide ([64, ...]
        # bf16 view of the fp8 pack); V is UNPACKED (token-major fp8, same layout as bf16 V) so it is just
        # quantized in place INSIDE the subtile loop into v_fp8_wide (a [128, ...] fp8 tile, no transpose/
        # pack), overlapping attention's Tensor-idle softmax gaps. The deferred scatter is DMA-only. v_wide
        # (bf16) is still needed as the fp8 V quantize source; kt_wide is bf16-only.
        v_wide = alloc.alloc_sbuf_tensor(shape=(_PMAX, n_sub, kv_count, d_head), dtype=dt, align_to=32)
        if fp8_packed:
            n_pk = _PMAX // 2
            k_pk_wide = alloc.alloc_sbuf_tensor(shape=(n_pk, n_sub, kv_count, d_head), dtype=dt, align_to=32)
            v_fp8_wide = alloc.alloc_sbuf_tensor(
                shape=(_PMAX, n_sub, kv_count, d_head), dtype=v_cache.dtype, align_to=32
            )
            # K is fully projected+RoPE'd (d-major [d, tw]) before the subtile loop, so quantize + pack
            # the WHOLE wide K at once -- one nc_transpose per <=256-token chunk vs one per 128-subtile.
            _quantize_pack_k_wide(
                k_tile, k_pk_wide, n_sub, kv_count, d_head, inv_k_scale_sb, k_fp8_max, k_cache.dtype, alloc
            )
        else:
            kt_wide = alloc.alloc_sbuf_tensor(shape=(_PMAX, n_sub, kv_count, d_head), dtype=k_cache.dtype, align_to=32)

        # ---- Inner 128-subtile loop: V-proj + K-transpose + attention + OP + reduce + carry ----
        for sub in range(n_sub):
            gst = wt * n_sub + sub  # global 128-tile index (drives carry, scatter, deferred reduce)
            cs = sub * _PMAX
            op_partial = op_wide[:, sub, :]  # this subtile's slot in the wide-tile OP buffer
            tile_save = alloc.get_current_address()

            # V projection for this 128-subtile (S-major) -> this subtile's slot in v_wide.
            v_tile = v_wide[:, sub, :, :]
            hidden_sub = hidden_t[:, :, cs : cs + _PMAX]
            _project_tile_smajor(v_tile, hidden_sub, vw_sb, kv_count, d_head, num_h_tiles, H, bias_sb=vb_sb)

            # This subtile's K view (already projected+RoPE'd in the wide tile above).
            k_sub = k_tile[:, :, cs : cs + _PMAX]
            if fp8_packed:
                # K was already quantized + packed wide before the loop; here quantize only this subtile's
                # V into v_fp8_wide for the deferred DMA-only scatter. V is token-major fp8 (UNPACKED,
                # like the bf16 cache), so this is a single in-place quantize -- no transpose/pack.
                # Emitted here so it overlaps attention's Tensor-idle softmax gaps.
                _quantize_v_subtile(
                    v_wide, v_fp8_wide, sub, kv_count, d_head, inv_v_scale_sb, k_fp8_max, v_cache.dtype, alloc
                )
            else:
                # Transpose post-RoPE K (d-major [d,128]) -> token-major [128,d] into kt_wide for the
                # deferred batched scatter. GATED by kt_dma (TP4): TP4 (Tensor-bound) uses dma_transpose --
                # the attention region is DMA-free so the idle Sync engine absorbs it, freeing Tensor; it
                # feeds only the deferred scatter (not a matmul) so no cross-engine semaphore. TP8 (DMA/MBU-
                # sensitive) keeps nc_transpose (dma_transpose regresses it ~+16%).
                for g in range(kv_count):
                    if kt_dma:
                        nisa.dma_transpose(dst=kt_wide[:, sub, g, :], src=k_sub[:, g, :])
                    else:
                        ks = alloc.get_current_address()
                        kt_psum = nl.ndarray((_PMAX, d_head), dtype=k_cache.dtype, buffer=nl.psum)
                        nisa.nc_transpose(dst=kt_psum, data=k_sub[:, g, :])
                        nisa.tensor_copy(dst=kt_wide[:, sub, g, :], src=kt_psum, engine=nisa.scalar_engine)
                        alloc.set_current_address(ks)

            lb_for_tile = prior_lb if gst == 0 else None
            attn_all = alloc.alloc_sbuf_tensor(shape=(2 * d_head, n_pairs, _PMAX), dtype=dt, align_to=32)
            # Mid-tile (gst>0, single kv head shared by all q-pairs): ROW-TILE the QK^T so a pair's two
            # heads' Q^T*K run CONCURRENTLY in the disjoint 64-row PE halves (d_head=64 each). Q is already
            # packed [2d, p, S] (head 2p in [0:d], 2p+1 in [d:2d]) -> no q_odd copy. K must be duplicated
            # to [K;K] across the two row-halves so each head's contraction reads its K -- built ONCE per
            # subtile here and reused by all pairs (kv_count==1).
            rowtile = (kv_count == 1) and (gst > 0)
            if rowtile:
                ksave = alloc.get_current_address()
                # k_dup[2d, 256] = [[k_left|k_curr] ; [k_left|k_curr]] (both 64-row halves hold combined K).
                # Key-time order (left block first) makes the kept softmax band a single contiguous
                # interval [i+1, i+W] per query i -> one range_select downstream (see _attend_combined).
                k_dup = alloc.alloc_sbuf_tensor(shape=(2 * d_head, 2 * _PMAX), dtype=dt, align_to=32)
                nisa.tensor_copy(dst=k_dup[0:d_head, 0:_PMAX], src=prev_k[:, 0, :], engine=nisa.scalar_engine)
                nisa.tensor_copy(dst=k_dup[0:d_head, _PMAX : 2 * _PMAX], src=k_sub[:, 0, :], engine=nisa.scalar_engine)
                nisa.tensor_copy(dst=k_dup[d_head : 2 * d_head, :], src=k_dup[0:d_head, :], engine=nisa.scalar_engine)
                for p in range(n_pairs):
                    save = alloc.get_current_address()
                    # Row-tiled QK: head0 (q rows [0:d], K rows [0:d]) -> sc_ps[:,0:256]; head1 (q rows
                    # [d:2d], K rows [d:2d]) -> a 2nd PSUM, concurrent in the bottom 64-row tile.
                    qp = q_packed[:, p, cs : cs + _PMAX]
                    sc0 = nl.ndarray((_PMAX, 2 * _PMAX), dtype=fp32, buffer=nl.psum)
                    sc1 = nl.ndarray((_PMAX, 2 * _PMAX), dtype=fp32, buffer=nl.psum)
                    nisa.nc_matmul(
                        sc0[:, :], qp[0:d_head, :], k_dup[0:d_head, :], tile_position=(0, 0), tile_size=(d_head, _PMAX)
                    )
                    nisa.nc_matmul(
                        sc1[:, :],
                        qp[d_head : 2 * d_head, :],
                        k_dup[d_head : 2 * d_head, :],
                        tile_position=(d_head, 0),
                        tile_size=(d_head, _PMAX),
                    )
                    for half in range(2):
                        qh = 2 * p + half
                        # proj_ctx lets _attend_combined emit one next-tile projection step in the
                        # softmax wait (between QK and P^T), filling the intra-pair Tensor gap. The
                        # emitter self-limits to n_pairs+1 steps; any leftover flushes at tile end.
                        _attend_combined(
                            attn_all[half * d_head : (half + 1) * d_head, p, :],
                            None,
                            None,
                            v_tile[:, 0, :],
                            None,
                            prev_v[:, 0, :],
                            sink_bcast[:, qh : qh + 1],
                            neg_sink_bcast[:, qh : qh + 1],
                            sliding_window,
                            block_size,
                            d_head,
                            alloc,
                            rs_bounds,
                            sc_psum=(sc0 if half == 0 else sc1),
                            proj_ctx=proj_ctx,
                        )
                    alloc.set_current_address(save)
                alloc.set_current_address(ksave)
            else:
                for p in range(n_pairs):
                    save = alloc.get_current_address()
                    # Q for this pair/subtile: even head at q_packed[0:d, p, cs:], odd at [d:2d, p, cs:].
                    q_even = q_packed[0:d_head, p, cs : cs + _PMAX]
                    q_odd = alloc.alloc_sbuf_tensor(shape=(d_head, _PMAX), dtype=dt, align_to=32)
                    nisa.tensor_copy(dst=q_odd[:, :], src=q_packed[d_head : 2 * d_head, p, cs : cs + _PMAX])
                    for half in range(2):
                        qh = 2 * p + half
                        g = min(qh // group_size, kv_count - 1)
                        q_h = q_even if half == 0 else q_odd[:, :]
                        _attend_one_head(
                            attn_all[half * d_head : (half + 1) * d_head, p, :],
                            q_h,
                            k_sub[:, g, :],
                            v_tile[:, g, :],
                            prev_k[:, g, :],
                            prev_v[:, g, :],
                            (gst == 0),
                            sink_bcast[:, qh : qh + 1],
                            neg_sink_bcast[:, qh : qh + 1],
                            sliding_window,
                            block_size,
                            d_head,
                            alloc,
                            lb_for_tile,
                            rs_bounds,
                        )
                    alloc.set_current_address(save)
            # Emit one next-tile projection step before the OP matmul to fill the OP-waits-for-last-PV
            # Tensor-idle gap (the OP can't start until the last pair's PV->softmax completes). Also a
            # backstop: if the intra-pair emission didn't reach this subtile's step (e.g. the tile-0
            # non-rowtile path emits none), this still fills the gap; otherwise it's a no-op.
            _emit_next_proj_step(proj_ctx)
            _op_project_stacked(op_partial, attn_all, opw_sb, n_pairs, d_head, H, alloc)

            # Update the carry with this subtile's K/V for the next 128-tile.
            for g in range(kv_count):
                nisa.tensor_copy(dst=prev_k[:, g, :], src=k_sub[:, g, :])
                nisa.tensor_copy(dst=prev_v[:, g, :], src=v_tile[:, g, :])
            alloc.set_current_address(tile_save)

        # ---- BATCHED KV-cache scatter for ALL n_sub subtiles (co-issued after the subtile loop) ----
        # Keeps the attention region DMA-free; all Sync-DMA writes fire here in one burst. Active tile
        # gst (128 tokens) -> logical blocks [n_prior_blocks + gst*bpt, ...+bpt); the 128-row kt_wide/
        # v_wide split into bpt block_size-row sub-blocks (sub-block i -> logical block ..+i), each
        # gathered to its physical block via block_tables. bpt=1 reduces to one write per subtile.
        sc_save = alloc.get_current_address()
        if fp8_packed:
            # DMA-only: K quant+pack ran before the loop (k_pk_wide), V quant ran in the loop (v_fp8_wide).
            _scatter_wide_fp8(
                k_pk_wide,
                v_fp8_wide,
                k_cache,
                v_cache,
                block_tables,
                b,
                n_prior_blocks_sb,
                n_sub,
                wt,
                kv_start,
                kv_count,
                d_head,
                block_size,
                bpt,
                alloc,
            )
            alloc.set_current_address(sc_save)
        else:
            for sub in range(n_sub):
                gst = wt * n_sub + sub
                for i in range(bpt):
                    active_lblk = alloc.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32, align_to=32)
                    nisa.tensor_scalar(
                        dst=active_lblk[0:1, 0:1],
                        data=n_prior_blocks_sb[0:1, 0:1],
                        op0=nl.add,
                        operand0=float(gst * bpt + i),
                    )
                    scatter_phys = _phys_blk(block_tables, b, active_lblk, alloc)
                    r0 = i * block_size
                    for g in range(kv_count):
                        head_off = (kv_start + g) * block_size * d_head
                        nisa.dma_copy(
                            dst=k_cache.ap(
                                pattern=[[d_head, block_size], [1, d_head]],
                                offset=head_off,
                                scalar_offset=scatter_phys,
                                indirect_dim=0,
                            ),
                            src=kt_wide[r0 : r0 + block_size, sub, g, :],
                            dge_mode=nisa.dge_mode.hwdge,
                            oob_mode=nisa.oob_mode.skip,
                        )
                        nisa.dma_copy(
                            dst=v_cache.ap(
                                pattern=[[d_head, block_size], [1, d_head]],
                                offset=head_off,
                                scalar_offset=scatter_phys,
                                indirect_dim=0,
                            ),
                            src=v_wide[r0 : r0 + block_size, sub, g, :],
                            dge_mode=nisa.dge_mode.hwdge,
                            oob_mode=nisa.oob_mode.skip,
                        )
            alloc.set_current_address(sc_save)

        # Flush any next-tile projection steps the interleave cadence didn't reach (e.g. the tile-0
        # non-rowtile path emits none; rounding may leave a tail). Ensures wt+1's q_packed/k_tile are
        # fully projected before the next iteration attends them.
        if do_proj:
            while proj_step[0] < n_proj_steps:
                ps = proj_step[0]
                if ps < n_pairs:
                    _project_one_qpair(
                        nq_packed,
                        nhidden_t,
                        qw_sb,
                        qb_sb,
                        ncos_f,
                        nsin_s,
                        swap_mat,
                        ps,
                        d_head,
                        num_h_tiles,
                        H,
                        tw,
                        scale,
                        alloc,
                    )
                else:
                    _project_tile_dmajor(nk_tile, nhidden_t, kw_sb, kv_count, d_head, num_h_tiles, H, bias_sb=kb_sb)
                    _rope_k_packed(
                        nk_tile,
                        ncos_f[0:d_head, :],
                        nsin_s[0:d_head, :],
                        swap_mat[0:d_head, 0:d_head],
                        kv_count,
                        d_head,
                        tw,
                        alloc,
                    )
                proj_step[0] += 1
        alloc.set_current_address(wt_save)

    # Flush the final wide tile's deferred cross-core reduce (no next iteration to overlap it with).
    _reduce_and_write_wide(
        op_wide_db[(num_w_tiles - 1) % 2],
        out,
        b,
        num_w_tiles - 1,
        n_sub,
        H,
        num_shard,
        shard_id,
        out.dtype,
        alloc,
        recv_buf,
        summed_buf,
        opb_bc,
    )


def _project_tile_dmajor(dst_sb, hidden_t, w_sb, n_heads, d_head, num_h_tiles, H, bias_sb=None):
    """QKV projection for one S-tile, emitting d-major [d_head, n_heads, 128] (weights-stationary).

    out[d, head, s] = sum_h W[h, head*d + d] * hidden[h, s]. hidden_t is [128, num_h_tiles, 128]
    (H on partition, this tile's 128 tokens on free). Successive matmuls into the same PSUM tile
    accumulate over the h tiles (partition dim clamped on the last tile). When bias_sb is given
    (d-major [d_head, n_heads]), it is added per-head in the PSUM->SBUF copy (tensor_scalar).
    """
    tw = dst_sb.shape[-1]
    for head in range(n_heads):
        psum = nl.ndarray((d_head, tw), dtype=nl.float32, buffer=nl.psum)
        for ht in range(num_h_tiles):
            hsz = _h_tile_size(H, ht)
            nisa.nc_matmul(
                psum[:, :],
                w_sb[:hsz, ht, head * d_head : (head + 1) * d_head],
                hidden_t[:hsz, ht, :],
            )
        if bias_sb is None:
            nisa.tensor_copy(dst=dst_sb[:, head, :], src=psum[:, :])
        else:
            nisa.tensor_scalar(
                dst=dst_sb[:, head, :], data=psum[:, :], op0=nl.add, operand0=bias_sb[:, head : head + 1]
            )


def _build_signed_swap_mat(d_head, dt, alloc):
    """[2*d_head, 2*d_head] signed-swap matrix M: signed_swap(x)=M@x = [-x_b1,+x_b0,-x_b3,+x_b2]
    (blocks of half_d, rotate's sign baked in). nc_matmul out[i,s]=sum_j M[j,i] x[j,s], so M's nonzero
    half_d-blocks (block-row j -> block-col i): M[b0,b1]=+I, M[b1,b0]=-I, M[b2,b3]=+I, M[b3,b2]=-I.
    Built ONCE per batch from a shared identity: memset 0, then copy +-I_half into the four blocks."""
    half = d_head // 2
    two_d = 2 * d_head
    ident = nl.shared_identity_matrix(half, dtype=dt)
    M = alloc.alloc_sbuf_tensor(shape=(two_d, two_d), dtype=dt, align_to=32)
    nisa.memset(M[:, :], 0.0)
    nisa.tensor_copy(dst=M[0:half, half : 2 * half], src=ident[:, :], engine=nisa.scalar_engine)
    nisa.tensor_copy(dst=M[2 * half : 3 * half, 3 * half : 4 * half], src=ident[:, :], engine=nisa.scalar_engine)
    nisa.tensor_scalar(dst=M[half : 2 * half, 0:half], data=ident[:, :], op0=nl.multiply, operand0=-1.0)
    nisa.tensor_scalar(
        dst=M[3 * half : 4 * half, 2 * half : 3 * half], data=ident[:, :], op0=nl.multiply, operand0=-1.0
    )
    return M


def _project_rope_pairs_fused(
    dst_sb, hidden_t, w_sb, qb_sb, cos_f, sin_s, swap_mat, n_pairs, d_head, num_h_tiles, H, S, scale, alloc
):
    """Fused per-pair Q projection + sign-folded RoPE. dst_sb is [2*d_head, n_pairs, S] (head 2p in
    partition rows [0:d], 2p+1 in [d:2d]). For each pair p: project the two heads in one [hsz, 2d=128]
    matmul (full PE fill), fold the QKV bias + Q-scale into the PSUM->SBUF copy, then apply the
    sign-folded rotate on this pair's [2*d_head, S] column: out = x*cos_f + swap(x)*sin_s, where
    swap(x) swaps each head's even/odd half_d-blocks (the +/- of the rotate is baked into sin_s).
    """
    half = d_head // 2
    two_d = 2 * d_head
    for p in range(n_pairs):
        psum = nl.ndarray((two_d, S), dtype=nl.float32, buffer=nl.psum)
        for ht in range(num_h_tiles):
            hsz = _h_tile_size(H, ht)
            nisa.nc_matmul(psum[:, :], w_sb[:hsz, ht, 2 * p * d_head : (2 * p + 2) * d_head], hidden_t[:hsz, ht, :])
        # Fold the Q-scale AND QKV bias into the projection PSUM->SBUF copy: dst = psum*scale + qb_scaled,
        # where qb_scaled = qb*scale was pre-computed at the qb_sb build. This is the identity
        # (psum + qb)*scale == psum*scale + qb*scale, expressed as the ACT-legal (multiply, add) op order
        # (the trn2 Scalar/ACT engine does not support add-then-multiply). Bias is folded into the linear
        # projection BEFORE RoPE; RoPE is linear so the scaled+biased x flows through swap/cos/sin and the
        # result is already biased+scaled -- no trailing op. qb_sb[:, p] is the per-partition [2d,1] bias
        # for pair p (head 2p in [0:d], 2p+1 in [d:2d]).
        nisa.tensor_scalar(
            dst=dst_sb[:, p, :],
            data=psum[:, :],
            op0=nl.multiply,
            operand0=float(scale),
            op1=nl.add,
            operand1=qb_sb[:, p : p + 1],
            engine=nisa.scalar_engine,
        )
        # Per-pair RoPE. signed_swap(x) = swap_mat @ x in ONE matmul (Tensor) -> PSUM, vs 4 partition
        # copies; sign baked into swap_mat so the sin multiply uses the UNSIGNED sin_s. The swap matmul
        # result feeds the sin multiply straight from PSUM (PSUM x SBUF).
        save = alloc.get_current_address()
        xsw_ps = nl.ndarray((two_d, S), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(xsw_ps[:, :], swap_mat[:, :], dst_sb[:, p, :])
        xsw = alloc.alloc_sbuf_tensor(shape=(two_d, S), dtype=dst_sb.dtype, align_to=32)
        A = alloc.alloc_sbuf_tensor(shape=(two_d, S), dtype=cos_f.dtype, align_to=32)
        nisa.tensor_tensor(A[:, :], dst_sb[:, p, :], cos_f[:, :], nl.multiply)
        nisa.tensor_tensor(xsw[:, :], xsw_ps[:, :], sin_s[:, :], nl.multiply)
        nisa.tensor_tensor(dst_sb[:, p, :], A[:, :], xsw[:, :], nl.add)
        # (Q-scale already folded into the projection copy above -- no trailing scale op.)
        alloc.set_current_address(save)


def _rope_k_packed(k_sb, cos_k, sin_k, swap_k, kv_count, d_head, S, alloc):
    """K RoPE via the swap-matmul scheme (moves the swap onto the Tensor engine, like Q's RoPE).

    k_sb is [d_head, kv_count, S] (one head/block = [even|odd] on d_head partitions). Per kv head g:
    signed_swap(k) = swap_k @ k[:, g, :] in ONE matmul (Tensor) -> PSUM, then k = k*cos_k + swap*sin_k
    with cos_k/sin_k = [c;c]/[s;s] ([d_head, S]) and sin_k UNSIGNED (sign baked into swap_k): 1 matmul
    + 3 vector ops per head (the swap runs on Tensor instead of as vector copies).
    """
    for g in range(kv_count):
        save = alloc.get_current_address()
        sw_ps = nl.ndarray((d_head, S), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(sw_ps[:, :], swap_k[:, :], k_sb[:, g, :])
        sw = alloc.alloc_sbuf_tensor(shape=(d_head, S), dtype=k_sb.dtype, align_to=32)
        A = alloc.alloc_sbuf_tensor(shape=(d_head, S), dtype=cos_k.dtype, align_to=32)
        nisa.tensor_tensor(A[:, :], k_sb[:, g, :], cos_k[:, :], nl.multiply)
        nisa.tensor_tensor(sw[:, :], sw_ps[:, :], sin_k[:, :], nl.multiply)
        nisa.tensor_tensor(k_sb[:, g, :], A[:, :], sw[:, :], nl.add)
        alloc.set_current_address(save)


def _load_wide_inputs(
    wt,
    hidden_t,
    cos_f,
    sin_s,
    hidden_states,
    cos_cache,
    sin_cache,
    b,
    tw,
    n_sub,
    num_h_tiles,
    H,
    d_head,
    dt,
    alloc,
):
    """PREFETCH a wide tile's projection INPUTS into caller-owned hidden_t/cos_f/sin_s buffers.

    Does the Sync/DMA-engine work (hidden [S,H]->[H,S] dma_transpose + cos/sin dma_copy) plus the cheap
    cos/sin nc_transpose+replicate that builds cos_f=[c;c;c;c]/sin_s=[s;s;s;s]. Split out from the
    projection matmuls so the caller can issue it 2 tiles AHEAD: the projection matmul of tile wt+1
    has a true FLOW dependency on these loads, and if they share the Sync engine with tile wt's
    attention (P^T dma_transpose) the matmul stalls. Prefetching de-congests that path -- the inputs are
    resident before the matmul needs them. The tiny per-sub c_tile/s_tile scratch is save/restored.
    """
    save = alloc.get_current_address()
    # Emit the cos/sin load+build FIRST, then the hidden load. cos/sin and hidden are INDEPENDENT
    # (no data dep), but cos/sin's chain contains an nc_transpose (Tensor) while the hidden load is
    # pure Sync DMA. Emitting cos/sin first lets its nc_transpose (the only Tensor work in this load
    # phase) issue early / overlap the hidden Sync DMA, putting some Tensor activity into what was an
    # otherwise Tensor-idle boundary load burst (profile: ~5-6us Tensor-idle per tile).
    for sub in range(n_sub):
        gst = wt * n_sub + sub
        cs = sub * _PMAX
        c_tile = alloc.alloc_sbuf_tensor(shape=(_PMAX, d_head), dtype=dt, align_to=32)
        nisa.dma_copy(dst=c_tile[:, :], src=cos_cache[b, gst * _PMAX : (gst + 1) * _PMAX, 0:d_head])
        ctp = nl.ndarray((d_head, _PMAX), dtype=dt, buffer=nl.psum)
        nisa.nc_transpose(dst=ctp, data=c_tile[:, :])
        nisa.tensor_copy(dst=cos_f[0:d_head, cs : cs + _PMAX], src=ctp, engine=nisa.scalar_engine)
        s_tile = alloc.alloc_sbuf_tensor(shape=(_PMAX, d_head), dtype=dt, align_to=32)
        nisa.dma_copy(dst=s_tile[:, :], src=sin_cache[b, gst * _PMAX : (gst + 1) * _PMAX, 0:d_head])
        stp = nl.ndarray((d_head, _PMAX), dtype=dt, buffer=nl.psum)
        nisa.nc_transpose(dst=stp, data=s_tile[:, :])
        nisa.tensor_copy(dst=sin_s[0:d_head, cs : cs + _PMAX], src=stp, engine=nisa.scalar_engine)
    nisa.tensor_copy(dst=cos_f[d_head : 2 * d_head, :], src=cos_f[0:d_head, :], engine=nisa.scalar_engine)
    nisa.tensor_copy(dst=sin_s[d_head : 2 * d_head, :], src=sin_s[0:d_head, :], engine=nisa.scalar_engine)
    for ht in range(num_h_tiles):
        hsz = _h_tile_size(H, ht)
        nisa.dma_transpose(
            dst=hidden_t[:hsz, ht, :],
            src=hidden_states[b, wt * tw : (wt + 1) * tw, ht * _PMAX : ht * _PMAX + hsz],
        )
    alloc.set_current_address(save)


def _project_rope_from_inputs(
    q_packed,
    k_tile,
    hidden_t,
    cos_f,
    sin_s,
    qw_sb,
    kw_sb,
    qb_sb,
    kb_sb,
    swap_mat,
    tw,
    num_h_tiles,
    H,
    d_head,
    n_pairs,
    kv_count,
    scale,
    alloc,
):
    """Q/K projection + sign-folded RoPE from PREFETCHED hidden_t/cos_f/sin_s -> q_packed/k_tile.

    The Tensor-heavy matmul phase, split from _load_wide_inputs so the loads can be prefetched ahead.
    """
    save = alloc.get_current_address()
    _project_rope_pairs_fused(
        q_packed, hidden_t, qw_sb, qb_sb, cos_f, sin_s, swap_mat, n_pairs, d_head, num_h_tiles, H, tw, scale, alloc
    )
    _project_tile_dmajor(k_tile, hidden_t, kw_sb, kv_count, d_head, num_h_tiles, H, bias_sb=kb_sb)
    _rope_k_packed(
        k_tile, cos_f[0:d_head, :], sin_s[0:d_head, :], swap_mat[0:d_head, 0:d_head], kv_count, d_head, tw, alloc
    )
    alloc.set_current_address(save)


def _project_one_qpair(
    q_packed, hidden_t, qw_sb, qb_sb, cos_f, sin_s, swap_mat, p, d_head, num_h_tiles, H, S, scale, alloc
):
    """ONE Q head-pair's projection + sign-folded RoPE -> q_packed[:, p, :]. Extracted from
    _project_rope_pairs_fused so the caller can EXPLICITLY interleave these per-pair projection
    matmuls (Tensor) between the current tile's attention pairs in program order -- forcing the
    near-program-order scheduler to place the next tile's projection into attention's Tensor idle
    (it does not hoist across a monolithic projection block on its own). Matches the per-pair body
    of _project_rope_pairs_fused exactly.
    """
    two_d = 2 * d_head
    psum = nl.ndarray((two_d, S), dtype=nl.float32, buffer=nl.psum)
    for ht in range(num_h_tiles):
        hsz = _h_tile_size(H, ht)
        nisa.nc_matmul(psum[:, :], qw_sb[:hsz, ht, 2 * p * d_head : (2 * p + 2) * d_head], hidden_t[:hsz, ht, :])
    # dst = psum*scale + qb_scaled (qb pre-scaled at build); ACT-legal (multiply, add) form of
    # (psum + qb)*scale. See _project_rope_pairs_fused for the full rationale.
    nisa.tensor_scalar(
        dst=q_packed[:, p, :],
        data=psum[:, :],
        op0=nl.multiply,
        operand0=float(scale),
        op1=nl.add,
        operand1=qb_sb[:, p : p + 1],
        engine=nisa.scalar_engine,
    )
    save = alloc.get_current_address()
    xsw_ps = nl.ndarray((two_d, S), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(xsw_ps[:, :], swap_mat[:, :], q_packed[:, p, :])
    xsw = alloc.alloc_sbuf_tensor(shape=(two_d, S), dtype=q_packed.dtype, align_to=32)
    A = alloc.alloc_sbuf_tensor(shape=(two_d, S), dtype=cos_f.dtype, align_to=32)
    nisa.tensor_tensor(A[:, :], q_packed[:, p, :], cos_f[:, :], nl.multiply)
    nisa.tensor_tensor(xsw[:, :], xsw_ps[:, :], sin_s[:, :], nl.multiply)
    nisa.tensor_tensor(q_packed[:, p, :], A[:, :], xsw[:, :], nl.add)
    alloc.set_current_address(save)


def _emit_next_proj_step(proj_ctx):
    """Emit ONE next-tile projection step (a Q head-pair, or the K proj+RoPE) from the bundled
    proj_ctx, advancing the mutable step counter. Used to inject projection (Tensor) into the
    softmax-wait Tensor gap INSIDE _attend_combined (between the QK and the P^T transpose) -- the
    scheduler runs Tensor near program order and won't hoist projection into that intra-pair gap on
    its own. proj_ctx is None on paths with no next tile (tile-0 / TP8 / last wide tile) -> no-op.

    proj_ctx = (step[mutable [int]], n_pairs, nq_packed, nhidden_t, qw_sb, qb_sb, ncos_f, nsin_s,
                swap_mat, nk_tile, kw_sb, kb_sb, d_head, num_h_tiles, H, tw, kv_count, scale, alloc)
    """
    if proj_ctx is None:
        return
    (
        step,
        n_pairs,
        nq_packed,
        nhidden_t,
        qw_sb,
        qb_sb,
        ncos_f,
        nsin_s,
        swap_mat,
        nk_tile,
        kw_sb,
        kb_sb,
        d_head,
        num_h_tiles,
        H,
        tw,
        kv_count,
        scale,
        alloc,
    ) = proj_ctx
    ps = step[0]
    if ps >= n_pairs + 1:
        return  # all steps for this tile already emitted
    if ps < n_pairs:
        _project_one_qpair(
            nq_packed, nhidden_t, qw_sb, qb_sb, ncos_f, nsin_s, swap_mat, ps, d_head, num_h_tiles, H, tw, scale, alloc
        )
    else:
        _project_tile_dmajor(nk_tile, nhidden_t, kw_sb, kv_count, d_head, num_h_tiles, H, bias_sb=kb_sb)
        _rope_k_packed(
            nk_tile, ncos_f[0:d_head, :], nsin_s[0:d_head, :], swap_mat[0:d_head, 0:d_head], kv_count, d_head, tw, alloc
        )
    step[0] += 1


def _project_tile_smajor(dst_sb, hidden_t, w_sb, n_heads, d_head, num_h_tiles, H, bias_sb=None):
    """V projection for one S-tile, emitting S-major [128, n_heads, d_head] (input-stationary).
    V is token-major (d on the free axis), so its bias [d] lives on free: when bias_sb is given
    ([128, n_heads*d], pre-broadcast over the token partitions since HW tensor_tensor rejects a
    partition-dim-1 operand), head g's [128, d] bias slice is added via tensor_tensor."""
    for head in range(n_heads):
        psum = nl.ndarray((_PMAX, d_head), dtype=nl.float32, buffer=nl.psum)
        for ht in range(num_h_tiles):
            hsz = _h_tile_size(H, ht)
            nisa.nc_matmul(
                psum[:, :],
                hidden_t[:hsz, ht, :],
                w_sb[:hsz, ht, head * d_head : (head + 1) * d_head],
            )
        if bias_sb is None:
            nisa.tensor_copy(dst=dst_sb[:, head, :], src=psum[:, :])
        else:
            # bias_sb is [128, n_heads*d] (broadcast over token partitions); add head's [128, d] slice.
            nisa.tensor_tensor(dst_sb[:, head, :], psum[:, :], bias_sb[:, head * d_head : (head + 1) * d_head], nl.add)


def _phys_blk(block_tables, b, logical_blk_sb, alloc):
    """Gather the RAW physical block index = block_tables[b, logical_blk] (runtime) into SBUF [1,1].

    logical_blk_sb is a runtime SBUF [1,1] int32 logical block index. The returned raw index is
    used as a dma_copy scalar_offset onto a cache whose leading (block) dim stride the access
    pattern applies via indirect_dim=0 (so the index must NOT be pre-scaled).
    """
    max_blocks = block_tables.shape[1]
    phys = alloc.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32, align_to=32)
    nisa.dma_copy(
        dst=phys[0:1, 0:1],
        src=block_tables.ap(
            pattern=[[1, 1], [1, 1]], offset=b * max_blocks, scalar_offset=logical_blk_sb, indirect_dim=1
        ),
        dge_mode=nisa.dge_mode.hwdge,
        oob_mode=nisa.oob_mode.skip,
    )
    return phys


def _load_prior_block_indirect(
    prior_k,
    prior_v,
    k_cache,
    v_cache,
    block_tables,
    b,
    win_lblk_sb,
    kv_start,
    kv_count,
    d_head,
    block_size,
    bpt,
    fp8_packed,
    k_scale_sb,
    v_scale_sb,
    alloc,
):
    """Load the prior sliding-window (last W==128 tokens) into the 128-row carry for this core's kv
    head(s). The window spans ``bpt`` = 128//block_size physical cache blocks, logical indices
    [win_lblk0, win_lblk0+bpt) where win_lblk0 = ``win_lblk_sb``; sub-block i (block_size tokens) lands
    in carry rows [i*block_size, (i+1)*block_size) -- so carry row r is the r-th window token in
    natural order (the convention the tile-0 prior_lb mask relies on, identical to the bpt=1 case).

    prior_k is d-major [d_head, kv_count, 128]; prior_v is [128, kv_count, d_head]. K is transposed
    to d-major; V copied token-major as-is. Always called (prior may be empty -> tile-0 mask drops it).

    When fp8_packed, the cache is packed-FP8 and the unpack+dequant runs in _load_prior_block_fp8.
    """
    if fp8_packed:
        _load_prior_block_fp8(
            prior_k,
            prior_v,
            k_cache,
            v_cache,
            block_tables,
            b,
            win_lblk_sb,
            kv_start,
            kv_count,
            d_head,
            block_size,
            bpt,
            k_scale_sb,
            v_scale_sb,
            alloc,
        )
        return
    for i in range(bpt):
        # logical block win_lblk0 + i -> physical via block_tables.
        lsave = alloc.get_current_address()
        lblk_i = alloc.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32, align_to=32)
        nisa.tensor_scalar(dst=lblk_i[0:1, 0:1], data=win_lblk_sb[0:1, 0:1], op0=nl.add, operand0=float(i))
        blk_phys = _phys_blk(block_tables, b, lblk_i, alloc)
        r0 = i * block_size
        for g in range(kv_count):
            save = alloc.get_current_address()
            head_off = (kv_start + g) * block_size * d_head
            nisa.dma_copy(
                dst=prior_v[r0 : r0 + block_size, g, :],
                src=v_cache.ap(
                    pattern=[[d_head, block_size], [1, d_head]], offset=head_off, scalar_offset=blk_phys, indirect_dim=0
                ),
                dge_mode=nisa.dge_mode.hwdge,
                oob_mode=nisa.oob_mode.skip,
            )
            ktmp = alloc.alloc_sbuf_tensor(shape=(block_size, d_head), dtype=prior_k.dtype, align_to=32)
            nisa.dma_copy(
                dst=ktmp[:, :],
                src=k_cache.ap(
                    pattern=[[d_head, block_size], [1, d_head]], offset=head_off, scalar_offset=blk_phys, indirect_dim=0
                ),
                dge_mode=nisa.dge_mode.hwdge,
                oob_mode=nisa.oob_mode.skip,
            )
            kt_psum = nl.ndarray((d_head, block_size), dtype=prior_k.dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=kt_psum, data=ktmp[:, :])
            nisa.tensor_copy(dst=prior_k[:, g, r0 : r0 + block_size], src=kt_psum)
            alloc.set_current_address(save)
        alloc.set_current_address(lsave)


def _load_prior_block_fp8(
    prior_k,
    prior_v,
    k_cache,
    v_cache,
    block_tables,
    b,
    win_lblk_sb,
    kv_start,
    kv_count,
    d_head,
    block_size,
    bpt,
    k_scale_sb,
    v_scale_sb,
    alloc,
):
    """Packed-FP8 prior load. The W=128 window spans bpt = 128//block_size physical cache blocks.

    K cache is PACKED (num_blocks, num_kv_heads, block_size//2, d_head, 2) fp8: each physical block holds
    block_size//2 packed rows (2 tokens per row in the trailing length-2 axis). V cache is UNPACKED,
    token-major (num_blocks, num_kv_heads, block_size, d_head) fp8 -- same layout as the bf16 cache, only
    the dtype differs -- so V is DMA'd straight into an fp8 SBUF tile (NO implicit cast) then dequantized
    to bf16, exactly like the bf16 path. num_blocks is the leading axis on both, so the indirect DMA uses
    scalar_offset=phys + indirect_dim=0 directly; the head is a static offset.

    K unpack: view the packed cache as bf16 (num_blocks, num_kv_heads*block_size//2*d_head) -- per
    (block, head) a [bs/2, d_head] bf16 tile whose element [r, c] is the byte-pair (token 2r, token 2r+1)
    of dim c. DMA the [bs/2, d] bf16 tile, nc_transpose to [d, bs/2] bf16 (packed pair on the FREE axis),
    reinterpret bf16->fp8 to expand free to [d, block_size] (free pos t = token t), dequant. For sub-block
    i the dequantized block_size tokens land in carry rows [i*block_size, (i+1)*block_size): K d-major
    (free axis), V token-major (partition axis).
    """
    bsz_half = block_size // 2  # packed rows per physical K block
    num_kv_total = k_cache.shape[1]
    k_flat = k_cache.reshape((k_cache.shape[0], num_kv_total * bsz_half * d_head * 2))
    k_bf16 = k_flat.view(nl.bfloat16)  # (num_blocks, num_kv*bs/2*d)
    k_head_stride = bsz_half * d_head  # bf16 elements per head within a packed K block
    v_head_stride = block_size * d_head  # fp8 elements per head within a token-major V block
    n_pk = _PMAX // 2  # total packed rows for the 128-token K window (bpt * bsz_half)
    for g in range(kv_count):
        k_col = (kv_start + g) * k_head_stride  # K head's static offset in the bf16 view
        v_head_off = (kv_start + g) * v_head_stride  # V head's static offset (fp8 token-major)
        gsave = alloc.get_current_address()
        # DMA all bpt physical blocks' K into ONE packed [64, d] bf16 buffer (block i -> rows
        # [i*bsz_half, (i+1)*bsz_half)), then unpack+dequant the FULL K window ONCE (transpose + dequant
        # are bpt-independent). V is loaded token-major per sub-block straight into an fp8 tile and
        # dequanted in place -- no transpose (it is already token-major in the cache, like bf16 V).
        k_pk_full = alloc.alloc_sbuf_tensor(shape=(n_pk, d_head), dtype=nl.bfloat16, align_to=32)
        for i in range(bpt):
            save = alloc.get_current_address()
            lblk_i = alloc.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32, align_to=32)
            nisa.tensor_scalar(dst=lblk_i[0:1, 0:1], data=win_lblk_sb[0:1, 0:1], op0=nl.add, operand0=float(i))
            blk_phys = _phys_blk(block_tables, b, lblk_i, alloc)
            pr0 = i * bsz_half  # packed-row offset for physical sub-block i (K)
            nisa.dma_copy(
                dst=k_pk_full[pr0 : pr0 + bsz_half, :],
                src=k_bf16.ap(
                    pattern=[[d_head, bsz_half], [1, d_head]], offset=k_col, scalar_offset=blk_phys, indirect_dim=0
                ),
                dge_mode=nisa.dge_mode.hwdge,
                oob_mode=nisa.oob_mode.skip,
            )
            # V: token-major fp8, DMA straight into an fp8 tile (NO implicit cast), then dequant to bf16.
            r0 = i * block_size  # sub-block i's carry rows [r0, r0+block_size)
            v_fp8 = alloc.alloc_sbuf_tensor(shape=(block_size, d_head), dtype=v_cache.dtype, align_to=32)
            nisa.dma_copy(
                dst=v_fp8[:, :],
                src=v_cache.ap(
                    pattern=[[d_head, block_size], [1, d_head]],
                    offset=v_head_off,
                    scalar_offset=blk_phys,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                oob_mode=nisa.oob_mode.skip,
            )
            nisa.tensor_scalar(
                dst=prior_v[r0 : r0 + block_size, g, :],
                data=v_fp8[:, :],
                op0=nl.multiply,
                operand0=v_scale_sb[0:block_size, 0:1],
            )
            alloc.set_current_address(save)
        # ---- K: [64, d] bf16 -> nc_transpose [d, 64] bf16 -> reinterpret [d, 128] fp8 -> dequant ----
        kt_ps = nl.ndarray((d_head, n_pk), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=kt_ps, data=k_pk_full[:, :])
        kt_sb = alloc.alloc_sbuf_tensor(shape=(d_head, n_pk), dtype=nl.bfloat16, align_to=32)
        nisa.tensor_copy(dst=kt_sb[:, :], src=kt_ps)
        kt_fp8 = kt_sb.view(k_cache.dtype)  # [d, 128] fp8, free=token
        nisa.tensor_scalar(
            dst=prior_k[:, g, 0:_PMAX],
            data=kt_fp8[:, :],
            op0=nl.multiply,
            operand0=k_scale_sb[0:d_head, 0:1],
        )
        alloc.set_current_address(gsave)


def _quantize_pack_k_wide(k_tile, k_pk_wide, n_sub, kv_count, d_head, inv_k_scale_sb, k_fp8_max, k_dtype, alloc):
    """Quantize + pack the WHOLE wide tile's K at once (K is d-major [d, n_sub*128] post-RoPE, fully
    available before the subtile loop). The quantize + pack-transpose are independent of the per-subtile
    attention, so batch them: quantize a 256-token chunk [d, 256] -> reinterpret [d, 128] bf16 ->
    ONE nc_transpose [d, 128] -> [128, d] (the partition cap is 128 packed rows = 256 tokens) ->
    split-copy each 128-token subtile's 64 packed rows into k_pk_wide[:, sub, g, :]. Halves K's
    quantize + pack-transpose op count vs per-subtile (TP4 n_sub=2: 1 chunk; TP8 n_sub=4: 2 chunks)."""
    sub_per_chunk = min(2, n_sub)  # 256 tokens / 128 = 2 subtiles per <=128-packed-row transpose
    for g in range(kv_count):
        for c0 in range(0, n_sub, sub_per_chunk):
            csave = alloc.get_current_address()
            nc = min(sub_per_chunk, n_sub - c0)  # subtiles in this chunk
            ctok = nc * _PMAX
            crows = ctok // 2
            t0 = c0 * _PMAX  # token offset of this chunk in the wide K tile
            q_fp8 = alloc.alloc_sbuf_tensor(shape=(d_head, ctok), dtype=k_dtype, align_to=32)
            qf = alloc.alloc_sbuf_tensor(shape=(d_head, ctok), dtype=nl.float32, align_to=32)
            nisa.tensor_scalar(
                dst=qf[:, :],
                data=k_tile[:, g, t0 : t0 + ctok],
                op0=nl.multiply,
                operand0=inv_k_scale_sb[0:d_head, 0:1],
                op1=nl.minimum,
                operand1=k_fp8_max,
            )
            nisa.tensor_scalar(dst=q_fp8[:, :], data=qf[:, :], op0=nl.maximum, operand0=-k_fp8_max)
            qb16 = q_fp8.view(nl.bfloat16)  # [d, ctok//2] bf16
            pk_ps = nl.ndarray((crows, d_head), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(dst=pk_ps, data=qb16[:, :])
            n_pk = _PMAX // 2  # 64 packed rows per subtile
            for sj in range(nc):
                nisa.tensor_copy(dst=k_pk_wide[:, c0 + sj, g, :], src=pk_ps[sj * n_pk : (sj + 1) * n_pk, :])
            alloc.set_current_address(csave)


def _quantize_v_subtile(v_wide, v_fp8_wide, sub, kv_count, d_head, inv_v_scale_sb, v_fp8_max, v_dtype, alloc):
    """Quantize this 128-token subtile's V into v_fp8_wide[:, sub, g, :] (token-major fp8). The V cache is
    UNPACKED (token-major [num_blocks, num_kv_heads, block_size, d_head] fp8, same layout as the bf16
    cache), so no transpose/pack is needed: V is already token-major [128, d] in v_wide, so quantize
    straight into the fp8 tile. quantize: q = clamp(x * (1/scale), -fp8_max, +fp8_max) -> fp8. The scale
    operand is per-token (partition axis), so use the [128, 1] broadcast column. Per-subtile (V is
    projected per-subtile, unlike the wide K)."""
    for g in range(kv_count):
        gsave = alloc.get_current_address()
        qf = alloc.alloc_sbuf_tensor(shape=(_PMAX, d_head), dtype=nl.float32, align_to=32)
        nisa.tensor_scalar(
            dst=qf[:, :],
            data=v_wide[0:_PMAX, sub, g, :],
            op0=nl.multiply,
            operand0=inv_v_scale_sb[0:_PMAX, 0:1],
            op1=nl.minimum,
            operand1=v_fp8_max,
        )
        nisa.tensor_scalar(
            dst=v_fp8_wide[:, sub, g, :], data=qf[:, :], op0=nl.maximum, operand0=-v_fp8_max
        )  # cast to fp8 on write
        alloc.set_current_address(gsave)


def _scatter_wide_fp8(
    k_pk_wide,
    v_fp8_wide,
    k_cache,
    v_cache,
    block_tables,
    b,
    n_prior_blocks_sb,
    n_sub,
    wt,
    kv_start,
    kv_count,
    d_head,
    block_size,
    bpt,
    alloc,
):
    """DMA-only FP8 write-back. The quantize (+ pack for K) already ran before/inside the subtile loop
    (_quantize_pack_k_wide -> k_pk_wide [64, n_sub, kv_count, d_head] bf16 packed rows;
    _quantize_v_subtile -> v_fp8_wide [128, n_sub, kv_count, d_head] token-major fp8).
    Each 128-token subtile spans bpt = 128//block_size physical blocks; sub-block i writes to logical
    block n_prior_blocks + gst*bpt + i, gathered to its physical block. num_blocks is the leading axis on
    both caches so the indirect DMA uses scalar_offset=phys + indirect_dim=0.

    K is PACKED: DMA into the bf16 view of (num_blocks, num_kv_heads, block_size//2, d_head, 2); sub-block
    i is packed rows [i*bsz_half, (i+1)*bsz_half). V is UNPACKED token-major fp8
    (num_blocks, num_kv_heads, block_size, d_head): DMA the fp8 tile straight in, sub-block i is token
    rows [i*block_size, (i+1)*block_size) -- identical to the bf16 V scatter, only the dtype differs."""
    bsz_half = block_size // 2
    num_kv_total = k_cache.shape[1]
    k_flat = k_cache.reshape((k_cache.shape[0], num_kv_total * bsz_half * d_head * 2))
    k_bf16 = k_flat.view(nl.bfloat16)  # (num_blocks, num_kv*bs/2*d) packed-K bf16 view
    k_head_stride = bsz_half * d_head  # bf16 elements per head within a packed K block
    v_head_stride = block_size * d_head  # fp8 elements per head within a token-major V block
    for g in range(kv_count):
        k_col_g = (kv_start + g) * k_head_stride
        v_head_off = (kv_start + g) * v_head_stride
        for sub in range(n_sub):
            gst = wt * n_sub + sub
            for i in range(bpt):
                save = alloc.get_current_address()
                pr0 = i * bsz_half  # packed-row offset for physical sub-block i (K)
                r0 = i * block_size  # token-row offset for physical sub-block i (V)
                active_lblk = alloc.alloc_sbuf_tensor(shape=(1, 1), dtype=nl.int32, align_to=32)
                nisa.tensor_scalar(
                    dst=active_lblk[0:1, 0:1],
                    data=n_prior_blocks_sb[0:1, 0:1],
                    op0=nl.add,
                    operand0=float(gst * bpt + i),
                )
                scatter_phys = _phys_blk(block_tables, b, active_lblk, alloc)
                nisa.dma_copy(
                    dst=k_bf16.ap(
                        pattern=[[d_head, bsz_half], [1, d_head]],
                        offset=k_col_g,
                        scalar_offset=scatter_phys,
                        indirect_dim=0,
                    ),
                    src=k_pk_wide[pr0 : pr0 + bsz_half, sub, g, :],
                    dge_mode=nisa.dge_mode.hwdge,
                    oob_mode=nisa.oob_mode.skip,
                )
                nisa.dma_copy(
                    dst=v_cache.ap(
                        pattern=[[d_head, block_size], [1, d_head]],
                        offset=v_head_off,
                        scalar_offset=scatter_phys,
                        indirect_dim=0,
                    ),
                    src=v_fp8_wide[r0 : r0 + block_size, sub, g, :],
                    dge_mode=nisa.dge_mode.hwdge,
                    oob_mode=nisa.oob_mode.skip,
                )
                alloc.set_current_address(save)


def _attend_one_head(
    out_dst,
    q_tile,
    k_curr,
    v_curr,
    k_left,
    v_left,
    left_is_prior_block,
    sink_b,
    neg_sink_b,
    sliding_window,
    block_size,
    d_head,
    alloc,
    prior_lb=None,
    rs_bounds=None,
):
    """Single-head SWA over [left | curr] window with causal+window+per-head sink.

    Writes the softmax-normalized attention output (d-major [d_head, 128]) into ``out_dst`` (a
    caller-provided SBUF view) so the caller can place the two heads of a pair into the two
    partition halves of a stacked [2*d_head, 128] tile -- no extra copy.
    ``k_left``/``v_left`` are None when this tile has no preceding window.
    ``sink_b`` is this head's per-partition sink column [128, 1], already DMA-loaded and broadcast
    once per batch by the caller (constant across S-tiles), so there is no per-tile sink shuffle here.
    ``prior_lb`` (per-partition fp32 [128,1] = W - prior_tokens) is non-None only on the first
    tile, where the left context is the prior block: a window slot j is valid iff j >= prior_lb,
    so the leading (W - prior_tokens) invalid slots are masked out at runtime via range_select.
    """
    fp32 = nl.float32
    has_left = k_left is not None
    # The prior/left window is always W==_PMAX (128) tokens: on tile 0 the prior carry is loaded as
    # 128 tokens spanning bpt=128//block_size physical cache blocks into the 128-row prev_k/prev_v
    # (leading invalid slots masked via prior_lb), and mid-tile the left context is the full previous
    # 128-tile. So the left score width is _PMAX regardless of block_size.
    Wkv = _PMAX

    # Mid-sequence tile (left is the full previous 128-tile, no runtime prior mask): do ONE combined
    # matmul + softmax over [left | curr] instead of two separate ones. The whole score/softmax chain
    # (PSUM->SBUF copy, mask, max-reduce, exp, normalize) then runs once over [128, 2*128] instead of
    # twice + two combines. Only the tile-0 prior path (runtime prior_lb mask) keeps the split form.
    if has_left and not left_is_prior_block and prior_lb is None:
        _attend_combined(
            out_dst,
            q_tile,
            k_curr,
            v_curr,
            k_left,
            v_left,
            sink_b,
            neg_sink_b,
            sliding_window,
            block_size,
            d_head,
            alloc,
            rs_bounds,
        )
        return

    # ---- MM1: scores. stationary=Q[d,128], moving=K[d,*] -> [128q, *] ------
    sc_curr = nl.ndarray((_PMAX, _PMAX), dtype=fp32, buffer=nl.psum)
    nisa.nc_matmul(sc_curr[:, :], q_tile, k_curr)

    sc_curr_m = alloc.alloc_sbuf_tensor(shape=(_PMAX, _PMAX), dtype=fp32, align_to=32)
    nisa.tensor_copy(dst=sc_curr_m[:, :], src=sc_curr[:, :])  # PSUM -> SBUF (affine_select needs SBUF)
    # causal: keep i - j >= 0
    nisa.affine_select(
        sc_curr_m[:, :],
        pattern=[[-1, _PMAX]],
        offset=0,
        channel_multiplier=1,
        cmp_op=nl.greater_equal,
        on_true_tile=sc_curr_m[:, :],
        on_false_value=_FLOAT32_MIN,
    )
    # window: keep i - j <= W-1  ->  (W-1) - i + j >= 0
    nisa.affine_select(
        sc_curr_m[:, :],
        pattern=[[1, _PMAX]],
        offset=(sliding_window - 1),
        channel_multiplier=-1,
        cmp_op=nl.greater_equal,
        on_true_tile=sc_curr_m[:, :],
        on_false_value=_FLOAT32_MIN,
    )

    row_max = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.tensor_reduce(row_max[:, 0:1], nl.maximum, sc_curr_m[:, :], [1])

    sc_left_m = None
    if has_left:
        sc_left = nl.ndarray((_PMAX, Wkv), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(sc_left[:, :], q_tile, k_left)
        sc_left_m = alloc.alloc_sbuf_tensor(shape=(_PMAX, Wkv), dtype=fp32, align_to=32)
        nisa.tensor_copy(dst=sc_left_m[:, :], src=sc_left[:, :])  # PSUM -> SBUF
        # left key j is at distance Wkv + i - j; causal always holds; window: Wkv+i-j <= W-1
        #   -> (W-1-Wkv) - i + j >= 0
        nisa.affine_select(
            sc_left_m[:, :],
            pattern=[[1, Wkv]],
            offset=(sliding_window - 1 - Wkv),
            channel_multiplier=-1,
            cmp_op=nl.greater_equal,
            on_true_tile=sc_left_m[:, :],
            on_false_value=_FLOAT32_MIN,
        )
        if prior_lb is not None:
            # Runtime prior-validity mask: keep window slot j iff j >= (W - prior_tokens) = prior_lb.
            # range_select compares (free_idx + range_start) against per-partition bounds; we need a
            # lower bound (>=) and a permissive upper bound (< Wkv keeps everything below the top).
            ub = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
            nisa.memset(ub[:, :], float(Wkv))
            nisa.range_select(
                sc_left_m[:, :],
                on_true_tile=sc_left_m[:, :],
                comp_op0=nl.greater_equal,
                comp_op1=nl.less,
                bound0=prior_lb[:, 0:1],
                bound1=ub[:, 0:1],
                on_false_value=_FLOAT32_MIN,
            )
        max_left = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
        nisa.tensor_reduce(max_left[:, 0:1], nl.maximum, sc_left_m[:, :], [1])
        nisa.tensor_tensor(row_max[:, 0:1], row_max[:, 0:1], max_left[:, 0:1], nl.maximum)

    # Fold the per-head sink into the row max AND negate, in a single vector op:
    #   neg_max = -max(row_max, sink) = min(-row_max, -sink)
    # via scalar_tensor_tensor (data*op0 then op1 operand1): (row_max * -1) min (-sink). neg_sink_b is
    # the pre-negated sink (resident, built once per batch). Replaces the separate max + multiply.
    neg_max = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.scalar_tensor_tensor(
        dst=neg_max[:, 0:1],
        data=row_max[:, 0:1],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.minimum,
        operand1=neg_sink_b[:, 0:1],
    )

    # exp(scores - max) with per-row running sums. denom is the running softmax denominator: seed it
    # directly with the current-tile exp-sum (the activation_reduce writes it), then accumulate the
    # left and sink terms in place -- no separate copy from a sum_curr scratch.
    p_curr = alloc.alloc_sbuf_tensor(shape=(_PMAX, _PMAX), dtype=fp32, align_to=32)
    denom = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.activation_reduce(
        p_curr[:, :], op=nl.exp, data=sc_curr_m[:, :], reduce_op=nl.add, reduce_res=denom[:, 0:1], bias=neg_max[:, 0:1]
    )

    p_left = None
    if has_left:
        p_left = alloc.alloc_sbuf_tensor(shape=(_PMAX, Wkv), dtype=fp32, align_to=32)
        sum_left = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
        nisa.activation_reduce(
            p_left[:, :],
            op=nl.exp,
            data=sc_left_m[:, :],
            reduce_op=nl.add,
            reduce_res=sum_left[:, 0:1],
            bias=neg_max[:, 0:1],
        )
        nisa.tensor_tensor(denom[:, 0:1], denom[:, 0:1], sum_left[:, 0:1], nl.add)

    # sink term: exp(sink - max) added to the denominator only. neg_max = -max(row_max, sink), so
    # sink - max = sink + neg_max; fold the subtract into the exp's bias (scalar-engine activation
    # computes exp(data*scale + bias)), removing the separate vector tensor_tensor(subtract).
    sink_exp = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.activation(sink_exp[:, 0:1], op=nl.exp, data=sink_b[:, 0:1], bias=neg_max[:, 0:1])
    nisa.tensor_tensor(denom[:, 0:1], denom[:, 0:1], sink_exp[:, 0:1], nl.add)

    recip = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.reciprocal(recip[:, 0:1], denom[:, 0:1])

    # ---- MM2: out[d, 128q] = sum over {curr, left} of V[k,d]^T @ P^T[k,q] ----
    # Both PV matmuls accumulate into ONE PSUM bank (mm); the cross-window sum is done in the matmul
    # accumulator (hardware), then a single PSUM->SBUF copy writes out_dst.
    save = alloc.get_current_address()
    mm = nl.ndarray((d_head, _PMAX), dtype=nl.float32, buffer=nl.psum)
    # Normalize in place (fp32), then nc_transpose + vector cast copy per window. (P^T via dma_transpose
    # was tried and regresses: it feeds the PV matmul, a cross-engine producer->consumer that needs a
    # per-window semaphore handshake; nc_transpose runs on the same Tensor engine as the matmul.)
    nisa.tensor_scalar(dst=p_curr[:, :], data=p_curr[:, :], op0=nl.multiply, operand0=recip[:, 0:1])
    pt_curr = nl.ndarray((_PMAX, _PMAX), dtype=p_curr.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=pt_curr, data=p_curr[:, :])
    ptc_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, _PMAX), dtype=v_curr.dtype, align_to=32)
    nisa.tensor_copy(dst=ptc_sb[:, :], src=pt_curr)
    nisa.nc_matmul(mm[:, :], v_curr[:, :], ptc_sb[:, :])
    if has_left:
        nisa.tensor_scalar(dst=p_left[:, :], data=p_left[:, :], op0=nl.multiply, operand0=recip[:, 0:1])
        pt_left = nl.ndarray((Wkv, _PMAX), dtype=p_left.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=pt_left, data=p_left[:, :])
        ptl_sb = alloc.alloc_sbuf_tensor(shape=(Wkv, _PMAX), dtype=v_left.dtype, align_to=32)
        nisa.tensor_copy(dst=ptl_sb[:, :], src=pt_left)
        nisa.nc_matmul(mm[:, :], v_left[:, :], ptl_sb[:, :])
    nisa.tensor_copy(dst=out_dst[:, :], src=mm[:, :], engine=nisa.scalar_engine)  # one PSUM -> SBUF for the summed PV
    alloc.set_current_address(save)


def _attend_combined(
    out_dst,
    q_tile,
    k_curr,
    v_curr,
    k_left,
    v_left,
    sink_b,
    neg_sink_b,
    sliding_window,
    block_size,
    d_head,
    alloc,
    rs_bounds=None,
    sc_psum=None,
    proj_ctx=None,
):
    """Mid-sequence SWA: ONE combined [left | curr] score + softmax pass (W == tile == 128).

    The left context is the full previous 128-tile (no runtime prior mask). Concatenating the two
    K blocks in key-time order [left(0:128) | curr(128:256)], query partition i keeps:
      - left key j (=col):       window keep j >= i+1            -> cols [i+1, 127]
      - curr key c (=col-128):   causal keep c <= i              -> cols [128, 128+i]
    whose UNION is the single contiguous band [i+1, i+W] (W=128) -- so ONE range_select masks it,
    vs the two needed for the old disjoint [curr|left] layout. The whole score/softmax chain
    (PSUM->SBUF copy, mask, max-reduce, exp+sum, normalize) runs ONCE over [128, 256]. PV uses two
    matmuls (contraction capped at 128) but they accumulate in one PSUM bank, and (since both windows'
    P^T transpose into one PSUM tile) a SINGLE PSUM->SBUF copy feeds both.

    sc_psum (optional [128, 256] PSUM scores): if given, the QK^T is precomputed by the caller (e.g.
    the row-tiled pair path that runs both heads' QK concurrently) and we skip the internal QK.
    """
    fp32 = nl.float32
    save0 = alloc.get_current_address()

    # MM1: scores for left and curr in key-time order. Both QK matmuls write into ONE [128, 256] PSUM
    # bank (left -> [:,0:128], curr -> [:,128:256]; 256 fp32 fits one 512-elem bank). Skipped when the
    # caller passes precomputed sc_psum (row-tiled pair QK, already [left|curr]).
    sc = alloc.alloc_sbuf_tensor(shape=(_PMAX, 2 * _PMAX), dtype=fp32, align_to=32)
    if sc_psum is None:
        sc_psum = nl.ndarray((_PMAX, 2 * _PMAX), dtype=fp32, buffer=nl.psum)
        nisa.nc_matmul(sc_psum[:, 0:_PMAX], q_tile, k_left)
        nisa.nc_matmul(sc_psum[:, _PMAX : 2 * _PMAX], q_tile, k_curr)

    # Band mask via range_select (Vector engine) instead of affine_select (GpSimd, the #1 GpSimd cost).
    # range_select reads the PSUM scores DIRECTLY (no PSUM->SBUF copy needed), masks to the band by
    # comparing the per-column free index (0..255) against per-partition [bound0, bound1], writes the
    # masked scores to sc (SBUF), AND fuses the row-max reduce via reduce_cmd/reduce_res -- so the
    # separate tensor_reduce(max) is gone too. With the [left|curr] layout the kept band is a single
    # contiguous interval [i+1, i+W] per query i, so ONE select (full reset_reduce) covers the row.
    rs_band_lb, rs_band_ub = rs_bounds
    row_max = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.range_select(
        sc[:, :],
        on_true_tile=sc_psum[:, :],
        comp_op0=nl.greater_equal,
        comp_op1=nl.less_equal,
        bound0=rs_band_lb[:, 0:1],
        bound1=rs_band_ub[:, 0:1],
        on_false_value=_FLOAT32_MIN,
        reduce_op=nl.maximum,
        reduce_res=row_max[:, 0:1],
        reduce_cmd=reduce_cmd.reset_reduce,
    )
    neg_max = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.scalar_tensor_tensor(
        dst=neg_max[:, 0:1],
        data=row_max[:, 0:1],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.minimum,
        operand1=neg_sink_b[:, 0:1],
    )
    # p (softmax exp) written bf16 directly: the activation casts on write (free), and the PV-path P^T
    # PSUM->SBUF copy becomes bf16->bf16 (0.5 cyc/elem) instead of a fp32->bf16 CAST (1.0). denom still
    # accumulates fp32 internally.
    p = alloc.alloc_sbuf_tensor(shape=(_PMAX, 2 * _PMAX), dtype=v_curr.dtype, align_to=32)
    denom = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    # denom = sum_j exp(sc-max) + exp(sink-max), accumulated on the Scalar engine's per-lane reduce_regs
    # ACROSS the two exp activations: the wide p exp seeds reduce_regs (reset_reduce, no read-out), the
    # sink exp reduce-accumulates exp(sink-max) and reads the FINAL denom into reduce_res -- removing a
    # separate Vector tensor_tensor(add) from the serial softmax chain. reduce_regs persists across the
    # two (both Scalar) with no eviction.
    sink_exp = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.activation(
        p[:, :], op=nl.exp, data=sc[:, :], bias=neg_max[:, 0:1], reduce_op=nl.add, reduce_cmd=reduce_cmd.reset_reduce
    )
    nisa.activation(
        sink_exp[:, 0:1],
        op=nl.exp,
        data=sink_b[:, 0:1],
        bias=neg_max[:, 0:1],
        reduce_op=nl.add,
        reduce_cmd=reduce_cmd.reduce,
        reduce_res=denom[:, 0:1],
    )
    recip = alloc.alloc_sbuf_tensor(shape=(_PMAX, 1), dtype=fp32, align_to=32)
    nisa.reciprocal(recip[:, 0:1], denom[:, 0:1])

    # Fill the softmax-wait Tensor gap: the range_select/exp/reciprocal above run on Vector/Scalar
    # while Tensor is idle until the P^T transpose below (which depends on p). Emit one next-tile
    # projection step (independent Tensor work) HERE, in program order between the QK and the P^T, so
    # the near-program-order scheduler places it in that intra-pair gap.
    _emit_next_proj_step(proj_ctx)

    # MM2: out = V_curr^T @ P_curr^T + V_left^T @ P_left^T, both into one PSUM bank.
    # DEFER the per-query 1/denom normalize past the PV matmul: since recip is per-query and PV is
    # linear, out[d,q] = recip[q] * sum_k V[d,k]*p[q,k]. So transpose the UNNORMALIZED p, run PV on it,
    # and apply recip once on the small [d_head, 128] output (folded into the out_dst write) -- removing
    # the wide [128, 256] normalize tensor_scalar at the cost of transposing recip [128q,1] ->
    # [d_head,128q] (query is on the free axis after the tp_out PV; mirrors attention_cte tp_out).
    # Both windows' P transpose into ONE [128,256] PSUM tile, then a SINGLE PSUM->SBUF copy feeds both
    # PV matmuls. P^T stays on nc_transpose (NOT dma_transpose): feeding the PV matmul cross-engine
    # needs a per-window semaphore handshake -> dma_transpose regresses ~45%.
    mm = nl.ndarray((d_head, _PMAX), dtype=nl.float32, buffer=nl.psum)
    pt = nl.ndarray((_PMAX, 2 * _PMAX), dtype=p.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=pt[:, 0:_PMAX], data=p[:, 0:_PMAX])
    nisa.nc_transpose(dst=pt[:, _PMAX : 2 * _PMAX], data=p[:, _PMAX : 2 * _PMAX])
    pt_sb = alloc.alloc_sbuf_tensor(shape=(_PMAX, 2 * _PMAX), dtype=v_curr.dtype, align_to=32)
    nisa.tensor_copy(dst=pt_sb[:, :], src=pt, engine=nisa.scalar_engine)
    nisa.nc_matmul(mm[:, :], v_left[:, :], pt_sb[:, 0:_PMAX])
    nisa.nc_matmul(mm[:, :], v_curr[:, :], pt_sb[:, _PMAX : 2 * _PMAX])
    # recip [128q,1] -> recip_t [d_head, 128q] (broadcast over d in the transpose source), then the
    # normalize fuses into the PSUM->SBUF out write: out_dst = mm * recip_t (one vector op, no copy).
    recip_tp = nl.ndarray((d_head, _PMAX), dtype=fp32, buffer=nl.psum)
    recip_bc = recip.broadcast(dim=1, size=d_head)  # [128q, d_head]
    nisa.nc_transpose(dst=recip_tp, data=recip_bc)
    recip_t = alloc.alloc_sbuf_tensor(shape=(d_head, _PMAX), dtype=fp32, align_to=32)
    nisa.tensor_copy(dst=recip_t[:, :], src=recip_tp, engine=nisa.scalar_engine)
    nisa.tensor_tensor(out_dst[:, :], mm[:, :], recip_t[:, :], nl.multiply)
    alloc.set_current_address(save0)


def _op_project_stacked(op_partial, attn_all, opw_sb, n_pairs, d_head, H, alloc):
    """op_partial[s, :H] = sum_p attn_all[2d, p, s]^T @ opw_sb[2d, p, :H] over all head-pairs.

    attn_all [2*d_head, n_pairs, 128] holds pair p's two heads stacked on the partition halves
    [0:d]/[d:2d]; opw_sb [2*d_head, n_pairs, H] is stacked the same way. For each H-block, ALL pairs'
    OP matmuls accumulate into one PSUM bank (the matmul accumulator sums the cross-pair contribution
    in hardware -- the OP is a sum over this core's heads), then a single PSUM->SBUF copy writes the
    block. This replaces the previous per-pair vector add into op_partial (n_pairs * n_hblocks vector
    ops) with just n_hblocks copies -- removing (n_pairs-1) * n_hblocks vector ops per tile.
    H is tiled in <=512-wide blocks (one PSUM bank's free dim).
    """
    h_block = 512
    num_blocks = (H + h_block - 1) // h_block
    for hb in range(num_blocks):
        h0 = hb * h_block
        hsz = min(h_block, H - h0)
        save = alloc.get_current_address()
        psum = nl.ndarray((_PMAX, h_block), dtype=nl.float32, buffer=nl.psum)
        for p in range(n_pairs):
            # Successive matmuls into the same PSUM tile accumulate (matmul start=p==0 semantics are
            # implicit: the first write initializes, later writes add in the accumulator).
            nisa.nc_matmul(psum[:, :hsz], attn_all[:, p, :], opw_sb[:, p, h0 : h0 + hsz])
        nisa.tensor_copy(dst=op_partial[:, h0 : h0 + hsz], src=psum[:, :hsz], engine=nisa.scalar_engine)
        alloc.set_current_address(save)


def _reduce_and_write_tile(op_partial, out, b, st, H, num_shard, shard_id, dt, alloc, recv, summed, opb_bc):
    """Sum the two cores' [128, H] OP partials for this S-tile, add the OP bias ONCE, write out[...].

    Single-core: write directly. Multi-core: exchange the partial with the peer core and add.
    ``recv``/``summed`` are caller-owned persistent buffers (one in-flight reduce at a time), so the
    exchange scratch does not alias per-tile stack scratch when this reduce is software-pipelined
    into the next tile's compute. ``opb_bc`` is the [128, H] broadcast output-projection bias.
    """
    if num_shard == 1:
        # out = op_partial + op_bias (single core owns all of H): fold the bias into the write.
        nisa.tensor_tensor(summed[:, :], op_partial[:, :], opb_bc[:, :], nl.add)
        nisa.dma_copy(dst=out[b, st * _PMAX : (st + 1) * _PMAX, :], src=summed[:, :])
        return
    # Reduce-SCATTER (not all-reduce): each core owns HALF of the H output. Core `shard_id` keeps the
    # half [my0:my0+half_H], sends its OTHER half to the peer, receives the peer's contribution to ITS
    # half, adds only that half, and writes only that half to out. (The peer writes the complementary
    # half.) This halves the cross-core DMA, the reduce add (vector), and the out DMA vs all-reduce,
    # where both cores redundantly summed and wrote the full [128, H]. Symmetric per core -> preserves
    # basic-block symmetry. NOTE: requires both cores' out writes to be disjoint and cover all of H.
    half_H = H // 2
    my0 = shard_id * half_H  # this core's owned half: [my0, my0+half_H)
    peer0 = (1 - shard_id) * half_H  # the half this core sends to the peer
    # Send my peer-half to the peer; receive the peer's contribution to MY half into recv[:, my0:].
    sendrecv(
        src=op_partial[:, peer0 : peer0 + half_H],
        dst=recv[:, my0 : my0 + half_H],
        send_to_rank=(1 - shard_id),
        recv_from_rank=(1 - shard_id),
        pipe_id=0,
    )
    # Fuse the reduce-add and the out-write into one DMA-engine op: dma_compute sums the two SBUF
    # halves (this core's own partial + the peer's contribution) in fp32 inside the DMA engine and
    # writes the result straight to out (HBM), casting to out.dtype in flight. This removes the
    # [128, half_H] vector add (the kernel's widest vector op, the reduce's critical-path tail) AND
    # the separate out dma_copy, trading two instructions on the Vector+DMA engines for one DMA op.
    # Add the OP bias as a third dma_compute addend (this core's owned half), so out = self_half +
    # peer_half + op_bias -- the bias is added exactly once (each core writes its disjoint half).
    nisa.dma_compute(
        dst=out[b, st * _PMAX : (st + 1) * _PMAX, my0 : my0 + half_H],
        srcs=[op_partial[:, my0 : my0 + half_H], recv[:, my0 : my0 + half_H], opb_bc[:, my0 : my0 + half_H]],
        reduce_op=nl.add,
    )


def _reduce_and_write_wide(op_wide, out, b, wt, n_sub, H, num_shard, shard_id, dt, alloc, recv, summed, opb_bc):
    """Coalesced cross-core reduce of a whole wide tile's n_sub OP partials with ONE sendrecv.

    ``op_wide`` is [128, n_sub, H]: subtile ``sub``'s [128, H] OP partial at op_wide[:, sub, :], for
    the wide tile ``wt`` (global 128-tile gst = wt*n_sub + sub). Instead of one sendrecv PER subtile
    (the per-subtile reduce-scatter, 2*n_sub cross-core barriers per wide tile), split the reduce-
    scatter on the SUBTILE axis: each core OWNS half the subtiles (whole [128, H] blocks, contiguous on
    the free dim), exchanges the peer-owned half in ONE sendrecv, then does a full-H reduce+write for
    each of its owned subtiles. One exchange => ONE pair of core barriers per wide tile.

    Owned subtiles: core ``shard_id`` keeps subtiles [my0, my0+n_own); sends the peer's [peer0, peer0+
    n_own) block; receives the peer's contribution to ITS subtiles into recv[:, 0:n_own, :]. The two
    cores write disjoint token ranges of ``out`` (each owns whole 128-token subtiles), and the OP bias
    is added exactly once per write. Symmetric per core -> preserves basic-block symmetry.
    """
    if num_shard == 1:
        for sub in range(n_sub):
            gst = wt * n_sub + sub
            nisa.tensor_tensor(summed[:, :], op_wide[:, sub, :], opb_bc[:, :], nl.add)
            nisa.dma_copy(dst=out[b, gst * _PMAX : (gst + 1) * _PMAX, :], src=summed[:, :])
        return
    if n_sub == 1:
        # Can't split a single subtile across cores on the subtile axis -> fall back to the H-split
        # reduce-scatter for this (gst == wt) tile (recv reused as the [128, H] H-split scratch).
        _reduce_and_write_tile(
            op_wide[:, 0, :], out, b, wt, H, num_shard, shard_id, dt, alloc, recv[:, 0, :], summed, opb_bc
        )
        return
    n_own = n_sub // 2
    my0 = shard_id * n_own  # this core's owned subtiles: [my0, my0+n_own)
    peer0 = (1 - shard_id) * n_own  # the subtiles this core sends to the peer
    # ONE sendrecv: ship the peer-owned subtile block (contiguous [128, n_own, H]); receive the peer's
    # contribution to MY subtiles into recv[:, 0:n_own, :]. This is the only cross-core sync per wide tile.
    sendrecv(
        src=op_wide[:, peer0 : peer0 + n_own, :],
        dst=recv[:, 0:n_own, :],
        send_to_rank=(1 - shard_id),
        recv_from_rank=(1 - shard_id),
        pipe_id=0,
    )
    # Full-H reduce+write for each owned subtile: out = self + peer + bias, fused in the DMA engine.
    for j in range(n_own):
        sub = my0 + j
        gst = wt * n_sub + sub
        nisa.dma_compute(
            dst=out[b, gst * _PMAX : (gst + 1) * _PMAX, :],
            srcs=[op_wide[:, sub, :], recv[:, j, :], opb_bc[:, :]],
            reduce_op=nl.add,
        )
