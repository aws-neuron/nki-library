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

"""BF16 score-path helpers for the SAI bf16-score variant (no Hadamard rotation and no FP8/MX
quantization of Q or K; the K cache stores bf16 and the per-head score matmul uses full-partition bf16
``nc_matmul``). The chunk-loop body applies per-chunk batched RoPE to Q, then dispatches a per-head bf16
score matmul against the loaded K cache."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_assert import kernel_assert
from ..mla.deepseek.mla_common_cte import RopeLayout
from .sparse_attention_indexer_utils import CACHE_TILE, P_MAX, TOPK_QUERY_CHUNKS

_UNIQ = [0]


def _u():
    """Monotonic suffix for alloc_stack names.

    ``alloc_stack`` names must be unique across the WHOLE trace -- ``close_scope`` releases a
    scope's addresses but not its tensor names -- and every loop in this file is Python-unrolled,
    so a bare literal name collides on the second iteration. Appending a trace-order counter makes
    that structurally impossible instead of relying on hand-written index suffixes.
    """
    _UNIQ[0] += 1
    return f"_{_UNIQ[0]}"


def transpose_k_for_cache(sbm: SbufManager, k_bf16_sb: nl.NkiTensor, S: int, head_dim: int) -> nl.NkiTensor:
    """Transpose K from [S P, head_dim F] to [head_dim P, S F] via DMA.

    Returns the transposed SBUF tile, reused by BOTH ``persist_k_bf16``
    (DMA out to cache) and ``score_against_cache_bf16`` (direct use as
    score matmul moving operand for the current S-tile). Using
    ``dma_transpose`` (DMA engine) instead of ``nc_transpose`` (PE) frees
    PE for the score matmul, which is the wall-clock binder.
    """
    k_transposed_sb = sbm.alloc_stack((P_MAX, S), nl.bfloat16)
    nisa.dma_transpose(
        dst=k_transposed_sb[:head_dim, :S],
        src=k_bf16_sb[:S, :head_dim],
    )
    return k_transposed_sb


def persist_k_bf16(
    sbm: SbufManager,
    k_transposed_sb: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    b_idx: int,
    cache_pos: int,
    S: int,
    head_dim: int,
) -> None:
    """DMA the pre-transposed [head_dim P, S F] K to the bf16 K cache.

    The transpose is done once by ``transpose_k_for_cache`` and reused by
    ``score_against_cache_bf16`` to avoid a second nc_transpose.
    """
    sbm.open_scope(name=f"persist_k_bf16{_u()}")
    nisa.dma_copy(
        dst=k_cache[b_idx, :head_dim, cache_pos : cache_pos + S],
        src=k_transposed_sb[:head_dim, :S],
    )
    sbm.close_scope()


def _load_k_cache_bf16(
    sbm: SbufManager,
    k_cache: nl.NkiTensor,
    b_idx: int,
    end_pos: int,
    head_dim: int,
    k_full_bf16_sb: nl.NkiTensor,
    k_current_T_sb: Optional[nl.NkiTensor] = None,
    current_start: Optional[int] = None,
    current_size: Optional[int] = None,
) -> None:
    """Load the bf16 K cache slab for the score matmul.

    HBM source: ``k_cache [B, head_dim, max_seq_len]`` bf16.
    SBUF dest:  ``k_full_bf16_sb [head_dim partition, end_pos free]`` bf16.

    If ``k_current_T_sb`` is provided, it must be the **already-transposed**
    current K (``[head_dim P, S F]`` bf16, produced by
    ``transpose_k_for_cache``); we copy it into the [current_start, current_start+S)
    slice. Older (committed) cache positions are loaded via DMA. This
    avoids both the HBM-write-commit-before-read race on TRN3 AND the
    redundant nc_transpose that was previously done here.
    """
    sbm.open_scope(name=f"load_k_cache_bf16{_u()}")

    """
    Zero the entire buffer; the loops below overwrite each valid range.
    Without this, partitions outside the loaded range hold stale bytes
    that decode as bf16 NaN/Inf and propagate through nc_matmul into
    the score, then NaN when masked. Pattern from _load_k_cache_mx.
    """
    nisa.memset(dst=k_full_bf16_sb, value=0, engine=nisa.engine.gpsimd)

    if k_current_T_sb != None and current_start != None:
        if current_start > 0:
            nisa.dma_copy(
                dst=k_full_bf16_sb[:head_dim, 0:current_start],
                src=k_cache[b_idx, 0:head_dim, 0:current_start],
            )
        # Forward the pre-transposed current K (no nc_transpose here).
        nisa.tensor_copy(
            dst=k_full_bf16_sb[:head_dim, current_start : current_start + current_size],
            src=k_current_T_sb[:head_dim, :current_size],
        )
    else:
        nisa.dma_copy(
            dst=k_full_bf16_sb[:head_dim, 0:end_pos],
            src=k_cache[b_idx, 0:head_dim, 0:end_pos],
        )

    sbm.close_scope()


def score_against_cache_bf16(
    sbm: SbufManager,
    q_full_hbm_view: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    score_row_sb: nl.NkiTensor,
    b_idx: int,
    S: int,
    end_pos: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    k_current_T_sb: Optional[nl.NkiTensor] = None,
    current_start: Optional[int] = None,
    current_size: Optional[int] = None,
    rope_layout: RopeLayout = RopeLayout.HALF_SPLIT,
) -> None:
    """BF16 score path: no Hadamard, no MX quantization.

    Per chunk of HEAD_CHUNK heads:
      1. DMA Q chunk from HBM (bf16).
      2. Batched RoPE on Q (DVE).
      3. For each head: dma_transpose Q[S, head_dim] -> [head_dim, S],
         then nc_matmul (bf16) against k_full_bf16_sb -> logits PSUM.
      4. Per-head: matmul into a wide bf16 logits PSUM (capped at PSUM_MAX_W and
         chunked when end_pos exceeds it), relu-evict to a full-width buffer, then a
         second pass of scalar_tensor_tensor weighted accumulate (DVE) per cache tile.

    Args:
        sbm (SbufManager): SBUF stack allocator used for all scratch tiles.
        q_full_hbm_view (nl.NkiTensor): [S, n_heads * head_dim], bf16 Q for the current S-tile on HBM.
        cos_sb (nl.NkiTensor): [S, rope_head_dim // 2], RoPE cosine table in SBUF.
        sin_sb (nl.NkiTensor): [S, rope_head_dim // 2], RoPE sine table in SBUF.
        k_cache (nl.NkiTensor): [B, head_dim, max_seq_len], bf16 K cache on HBM.
        weights_sb (nl.NkiTensor): [S, n_heads], per-head score weights in SBUF.
        score_row_sb (nl.NkiTensor): [S, end_pos], output score accumulator in SBUF (written in place).
        b_idx (int): Batch index into k_cache.
        S (int): Query sequence length of the current S-tile.
        end_pos (int): Total number of cache positions to score against (current + prior).
        n_heads (int): Number of attention heads.
        head_dim (int): Per-head dimension.
        rope_head_dim (int): Number of leading head-dim elements RoPE is applied to (0 disables RoPE).
        k_current_T_sb (Optional[nl.NkiTensor]): Pre-transposed [head_dim, S] current K, forwarded to
            _load_k_cache_bf16 to avoid a redundant transpose. Supply together with current_start and
            current_size.
        current_start (Optional[int]): Cache offset where the current K slab starts.
        current_size (Optional[int]): Number of current-K positions.
        rope_layout (RopeLayout): Q RoPE element pairing. ``HALF_SPLIT`` (default) matches
            DeepSeek-V3.2's indexer; ``INTERLEAVED`` matches GLM-MoE-DSA's. Must match the K
            side (``fused_layernorm_rope_k``) — Q and K are contracted together, so mixing the
            two pairings silently corrupts the scores.

    Returns:
        None: score_row_sb is updated in place with the accumulated per-head scores.
    """
    sbm.open_scope(name=f"score_full_bf16{_u()}")

    # Load full K cache: [head_dim partition, end_pos free] bf16.
    k_full_bf16_sb = sbm.alloc_stack((P_MAX, end_pos), nl.bfloat16)
    _load_k_cache_bf16(
        sbm,
        k_cache,
        b_idx,
        end_pos,
        head_dim,
        k_full_bf16_sb,
        k_current_T_sb=k_current_T_sb,
        current_start=current_start,
        current_size=current_size,
    )

    nisa.memset(dst=score_row_sb[:S, :end_pos], value=0.0, engine=nisa.engine.gpsimd)

    HEAD_CHUNK = 8 if n_heads >= 8 else n_heads
    n_head_chunks = (n_heads + HEAD_CHUNK - 1) // HEAD_CHUNK
    rope_half = rope_head_dim // 2
    num_cache_tiles = (end_pos + CACHE_TILE - 1) // CACHE_TILE

    sbm.open_scope(interleave_degree=2, name=f"chunk_loop_bf16{_u()}")
    for chunk_idx in nl.sequential_range(n_head_chunks):
        h_start = chunk_idx * HEAD_CHUNK
        h_count = min(HEAD_CHUNK, n_heads - h_start)
        chunk_off = h_start * head_dim
        chunk_width = HEAD_CHUNK * head_dim

        """
        q_full_hbm is bf16 (Q-projection wrote bf16). Land in bf16 SBUF
        directly so RoPE consumes bf16 inputs and we skip the explicit
        f32->bf16 cast that was previously on Scalar (~1.2 ms occupancy).
        """
        q_chunk_sb = sbm.alloc_stack((P_MAX, chunk_width), nl.bfloat16)
        nisa.dma_copy(
            dst=q_chunk_sb[:S, : h_count * head_dim],
            src=q_full_hbm_view[:S, chunk_off : chunk_off + h_count * head_dim],
        )

        # Batched RoPE: process all h_count heads in one op set.
        if rope_head_dim > 0:
            q_view = q_chunk_sb[:S, :].reshape_dim(1, [HEAD_CHUNK, head_dim])
            if rope_layout.is_interleaved():
                """
                INTERLEAVED pairing: the two rotated lanes are the EVEN and ODD columns of each
                head's rope slice, reached with a stride-2 inner access pattern instead of the
                half-split's two contiguous halves. Free pattern
                [[chunk_width, S], [head_dim, HEAD_CHUNK], [2, rope_half]] walks
                (query, head, pair); offset 0/1 picks the lane. Everything below is identical,
                so this is a re-addressing of the same 6 ops, not extra work.
                """
                x_lo_view = q_chunk_sb.ap(pattern=[[chunk_width, S], [head_dim, HEAD_CHUNK], [2, rope_half]], offset=0)
                x_hi_view = q_chunk_sb.ap(pattern=[[chunk_width, S], [head_dim, HEAD_CHUNK], [2, rope_half]], offset=1)
            else:
                x_lo_view = q_view.slice(dim=2, start=0, end=rope_half)
                x_hi_view = q_view.slice(dim=2, start=rope_half, end=rope_head_dim)
            cos_bcast = cos_sb[:S, :].expand_dim(1).broadcast(dim=1, size=HEAD_CHUNK)
            sin_bcast = sin_sb[:S, :].expand_dim(1).broadcast(dim=1, size=HEAD_CHUNK)

            t_lo_cos = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_hi_cos = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_lo_sin = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_hi_sin = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)

            nisa.tensor_tensor(
                dst=t_lo_cos[:S, :HEAD_CHUNK, :rope_half],
                data1=x_lo_view,
                data2=cos_bcast,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=t_hi_cos[:S, :HEAD_CHUNK, :rope_half],
                data1=x_hi_view,
                data2=cos_bcast,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=t_lo_sin[:S, :HEAD_CHUNK, :rope_half],
                data1=x_lo_view,
                data2=sin_bcast,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=t_hi_sin[:S, :HEAD_CHUNK, :rope_half],
                data1=x_hi_view,
                data2=sin_bcast,
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=x_lo_view,
                data1=t_lo_cos[:S, :HEAD_CHUNK, :rope_half],
                data2=t_hi_sin[:S, :HEAD_CHUNK, :rope_half],
                op=nl.subtract,
            )
            nisa.tensor_tensor(
                dst=x_hi_view,
                data1=t_lo_sin[:S, :HEAD_CHUNK, :rope_half],
                data2=t_hi_cos[:S, :HEAD_CHUNK, :rope_half],
                op=nl.add,
            )

        # q_chunk_sb is already bf16 (loaded from bf16 q_full_hbm). The
        # explicit f32->bf16 cast that was here is no longer needed.

        """
        Batched Q transpose: SBUF→SBUF dma_transpose with axes=(2, 1, 0)
        produces all h_count heads' transposed Q in ONE DMA op. View
        q_chunk_sb as [S, HEAD_CHUNK, head_dim] and transpose to
        [head_dim, HEAD_CHUNK, S]. Each head's stationary slice is then
        q_T_chunk[:, h, :] = [head_dim P, S F] — ready for nc_matmul.
        """
        q_T_chunk_sb = sbm.alloc_stack(
            (P_MAX, HEAD_CHUNK, S),
            nl.bfloat16,
        )
        q_chunk_bf16_view = q_chunk_sb[:S, :].reshape_dim(
            1,
            [HEAD_CHUNK, head_dim],
        )
        nisa.dma_transpose(
            dst=q_T_chunk_sb[:head_dim, :HEAD_CHUNK, :S],
            src=q_chunk_bf16_view,
            axes=(2, 1, 0),
        )

        """
        Per-head bf16 score matmul + post-processing.
        interleave_degree=2 double-buffers logits_psum so head N+1's
        matmul can overlap with head N's relu+STT.
        """
        sbm.open_scope(interleave_degree=2, name=f"per_head_score_bf16{_u()}")
        for h_in_chunk in range(HEAD_CHUNK):
            if h_in_chunk >= h_count:
                continue
            h_idx = h_start + h_in_chunk

            # PSUM width: a full-[S, end_pos] logits PSUM lets the relu evict the whole
            # row in ONE Scalar activation (fastest), but at large end_pos it exceeds the
            # 8 hardware banks (NCC_IGCA088). So cap the PSUM tile at PSUM_MAX_W (<=4 banks
            # bf16 = 4096 elems); when end_pos fits, psum_w == end_pos (one wide PSUM, one
            # relu). When it doesn't (e.g. 8k gathered K), matmul in psum_w chunks, evict
            # each to the full-width relu buffer, then one wide relu over it.
            PSUM_MAX_W = 4096
            psum_w = min(end_pos, PSUM_MAX_W)
            relu_logits = sbm.alloc_stack((P_MAX, end_pos), nl.bfloat16)
            for ps in range(0, end_pos, psum_w):
                pz = min(psum_w, end_pos - ps)
                logits_psum = nl.ndarray((P_MAX, psum_w), dtype=nl.bfloat16, buffer=nl.psum)
                for cs in range(ps, ps + pz, CACHE_TILE):
                    cz = min(CACHE_TILE, ps + pz - cs)
                    nisa.nc_matmul(
                        dst=logits_psum[:S, cs - ps : cs - ps + cz],
                        stationary=q_T_chunk_sb[:head_dim, h_in_chunk, :S],
                        moving=k_full_bf16_sb[:head_dim, cs : cs + cz],
                    )
                # relu this PSUM chunk into the full-width relu buffer (Scalar).
                nisa.activation(
                    dst=relu_logits[:S, ps : ps + pz],
                    op=nl.relu,
                    data=logits_psum[:S, :pz],
                )
            # Per-tile fused weighted accumulate on Vector (interleave_degree=2 keeps head
            # N+1's matmul overlapping head N's Vector work).
            for ct in range(num_cache_tiles):
                cs = ct * CACHE_TILE
                cz = min(CACHE_TILE, end_pos - cs)
                nisa.scalar_tensor_tensor(
                    dst=score_row_sb[:S, cs : cs + cz],
                    data=relu_logits[:S, cs : cs + cz],
                    op0=nl.multiply,
                    operand0=weights_sb[:S, h_idx : h_idx + 1],
                    op1=nl.add,
                    operand1=score_row_sb[:S, cs : cs + cz],
                )
            sbm.increment_section()
        sbm.close_scope()  # per_head_score_bf16

        sbm.increment_section()
    sbm.close_scope()  # chunk_loop_bf16

    sbm.close_scope()  # score_full_bf16


# Position-block width for the streamed score path. Multiple of CACHE_TILE (512) so the
# matmul tiling divides evenly, and of TOPK_QUERY_CHUNKS (16) so each block's HBM write
# lands on a snake-chunk boundary.
POS_BLOCK = 4096


def _load_k_block_bf16(k_cache, b_idx, head_dim, k_blk_sb, pb, blk):
    """Load COMMITTED cache positions ``[pb, pb + blk)`` into ``k_blk_sb[:head_dim, :blk]``.

    Committed only: the current (uncommitted) K is handled as its own segment straight out of
    SBUF by the caller, so this never has to splice the two and never reads a cache range whose
    HBM write may not have landed.

    The memset covers the ``[blk, POS_BLOCK)`` tail of the buffer for a short final segment --
    stale bytes there would decode as bf16 NaN/Inf and reach the score through nc_matmul.
    """
    # Only a SHORT final segment needs zeroing, and only its tail [blk, POS_BLOCK): the DMA below
    # overwrites [0, blk) in full. Memsetting the whole buffer on every block was a full-width
    # write per block per S-tile for nothing.
    if blk < POS_BLOCK:
        nisa.memset(dst=k_blk_sb[:head_dim, blk:POS_BLOCK], value=0, engine=nisa.engine.gpsimd)
    nisa.dma_copy(
        dst=k_blk_sb[:head_dim, 0:blk],
        src=k_cache[b_idx, 0:head_dim, pb : pb + blk],
    )


def _prep_q_transposed_all_heads(
    sbm: SbufManager,
    q_full_hbm_view: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    S: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    rope_layout: RopeLayout,
    tag: str = "",
) -> nl.NkiTensor:
    """RoPE + transpose Q for ALL heads once, into ``[head_dim P, n_heads, S]`` bf16.

    The streamed score loop is position-major, so Q preparation must be hoisted above it
    -- doing it per position block would repeat the RoPE and transpose ``end_pos /
    POS_BLOCK`` times (32x at 128k). Cost is ``n_heads * S * 2`` B/partition (8 KB at
    GLM's 32 heads, 32 KB at DeepSeek's 128), which is fixed in ``end_pos``.

    Body is the head-chunk loop of :func:`score_against_cache_bf16` writing into a
    wider destination; see there for the RoPE pairing details.
    """
    rope_half = rope_head_dim // 2
    HEAD_CHUNK = 8 if n_heads >= 8 else n_heads
    n_head_chunks = (n_heads + HEAD_CHUNK - 1) // HEAD_CHUNK

    q_T_all_sb = sbm.alloc_stack((P_MAX, n_heads, S), nl.bfloat16, name=f"q_T_all{_u()}")

    sbm.open_scope(interleave_degree=2, name=f"q_prep_all_heads{_u()}")
    for chunk_idx in nl.sequential_range(n_head_chunks):
        h_start = chunk_idx * HEAD_CHUNK
        h_count = min(HEAD_CHUNK, n_heads - h_start)
        chunk_off = h_start * head_dim
        chunk_width = HEAD_CHUNK * head_dim

        q_chunk_sb = sbm.alloc_stack((P_MAX, chunk_width), nl.bfloat16)
        nisa.dma_copy(
            dst=q_chunk_sb[:S, : h_count * head_dim],
            src=q_full_hbm_view[:S, chunk_off : chunk_off + h_count * head_dim],
        )

        if rope_head_dim > 0:
            q_view = q_chunk_sb[:S, :].reshape_dim(1, [HEAD_CHUNK, head_dim])
            if rope_layout.is_interleaved():
                x_lo_view = q_chunk_sb.ap(pattern=[[chunk_width, S], [head_dim, HEAD_CHUNK], [2, rope_half]], offset=0)
                x_hi_view = q_chunk_sb.ap(pattern=[[chunk_width, S], [head_dim, HEAD_CHUNK], [2, rope_half]], offset=1)
            else:
                x_lo_view = q_view.slice(dim=2, start=0, end=rope_half)
                x_hi_view = q_view.slice(dim=2, start=rope_half, end=rope_head_dim)
            cos_bcast = cos_sb[:S, :].expand_dim(1).broadcast(dim=1, size=HEAD_CHUNK)
            sin_bcast = sin_sb[:S, :].expand_dim(1).broadcast(dim=1, size=HEAD_CHUNK)

            t_lo_cos = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_hi_cos = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_lo_sin = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_hi_sin = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)

            nisa.tensor_tensor(
                dst=t_lo_cos[:S, :HEAD_CHUNK, :rope_half], data1=x_lo_view, data2=cos_bcast, op=nl.multiply
            )
            nisa.tensor_tensor(
                dst=t_hi_cos[:S, :HEAD_CHUNK, :rope_half], data1=x_hi_view, data2=cos_bcast, op=nl.multiply
            )
            nisa.tensor_tensor(
                dst=t_lo_sin[:S, :HEAD_CHUNK, :rope_half], data1=x_lo_view, data2=sin_bcast, op=nl.multiply
            )
            nisa.tensor_tensor(
                dst=t_hi_sin[:S, :HEAD_CHUNK, :rope_half], data1=x_hi_view, data2=sin_bcast, op=nl.multiply
            )
            nisa.tensor_tensor(
                dst=x_lo_view,
                data1=t_lo_cos[:S, :HEAD_CHUNK, :rope_half],
                data2=t_hi_sin[:S, :HEAD_CHUNK, :rope_half],
                op=nl.subtract,
            )
            nisa.tensor_tensor(
                dst=x_hi_view,
                data1=t_lo_sin[:S, :HEAD_CHUNK, :rope_half],
                data2=t_hi_cos[:S, :HEAD_CHUNK, :rope_half],
                op=nl.add,
            )

        q_chunk_bf16_view = q_chunk_sb[:S, :].reshape_dim(1, [HEAD_CHUNK, head_dim])
        nisa.dma_transpose(
            dst=q_T_all_sb[:head_dim, h_start : h_start + HEAD_CHUNK, :S],
            src=q_chunk_bf16_view,
            axes=(2, 1, 0),
        )
        sbm.increment_section()
    sbm.close_scope()

    return q_T_all_sb


def score_and_persist_blocked_bf16(
    sbm: SbufManager,
    q_full_hbm_view: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    k_cache: nl.NkiTensor,
    weights_sb: nl.NkiTensor,
    mask_hbm: nl.NkiTensor,
    mask_row_start: int,
    score_padded_hbm: nl.NkiTensor,
    s_tile_idx: int,
    T: int,
    b_idx: int,
    S: int,
    end_pos: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    index_score_hbm: Optional[nl.NkiTensor] = None,
    index_score_row_start: int = 0,
    persist_buf: Optional[nl.NkiTensor] = None,
    part_offset: int = 0,
    k_current_T_sb: Optional[nl.NkiTensor] = None,
    current_start: Optional[int] = None,
    current_size: Optional[int] = None,
    rope_layout: RopeLayout = RopeLayout.HALF_SPLIT,
) -> None:
    """Streamed score + mask + persist, with SBUF cost independent of ``end_pos``.

    Replaces the ``score_against_cache_bf16`` -> mask-add -> ``_pad_and_persist_score``
    sequence, which held FIVE ``end_pos``-wide SBUF buffers live at once (12 B/partition
    per position: k_full 2 + relu_logits 2 + score_row 2 + mask 2 + the fp32 padded-score
    staging 4). At ``end_pos`` 131072 that is 1.57 MB/partition against a ~245 KB budget,
    so 128k could not be traced at all.

    The fix is loop order, not tiling: the original nest is head-major / position-minor,
    and ``score_row`` accumulates ACROSS heads, which forces it (and everything feeding it)
    to full width. Inverting to position-major makes every buffer ``POS_BLOCK``-wide::

        for each position block:            # outer
            load K block                    # 2 B * POS_BLOCK
            for each head:                  # inner
                matmul -> relu -> weighted-accumulate into the block's score
            add the mask block, write the block straight to score_padded_hbm

    Q's RoPE + transpose is hoisted above the position loop (see
    :func:`_prep_q_transposed_all_heads`) so it is not repeated per block. Peak SBUF is
    then ``~6 B * POS_BLOCK`` plus ``n_heads * S * 2`` for Q^T -- ~32 KB at GLM dims,
    flat in ``end_pos``.

    The per-block HBM write also removes the separate fp32 pad-and-persist staging: each
    block is cast to fp32 and DMA'd into its slice of ``score_padded_hbm``, and only the
    ``[end_pos, T)`` remainder (< 16 columns) is filled with -inf so padded snake slots
    never win the top-k -- the invariant :func:`_pad_and_persist_score` provided.

    NOTE: this lifts the SBUF ceiling only. ``nisa.topk`` still reduces at most
    ``n < 65536`` per call, so ``T >= 65536`` needs the two-stage (chunked) top-k; this
    function alone gets the kernel to just under 64k, not to 128k.
    """
    kernel_assert(T % TOPK_QUERY_CHUNKS == 0, f"T must be a multiple of {TOPK_QUERY_CHUNKS}, got T={T}")
    kernel_assert(POS_BLOCK % CACHE_TILE == 0, "POS_BLOCK must be a multiple of CACHE_TILE")

    sbm.open_scope(name=f"score_blocked_bf16{_u()}")

    # Hoisted above the position loop: fixed cost in end_pos.
    # Unique per (batch, query-row) so the per-S-tile allocations do not collide: alloc_stack
    # names must be unique across the whole trace, and this is called once per S-tile per core.
    _tag = f"_b{b_idx}_p{index_score_row_start}"
    q_T_all_sb = _prep_q_transposed_all_heads(
        sbm, q_full_hbm_view, cos_sb, sin_sb, S, n_heads, head_dim, rope_head_dim, rope_layout, tag=_tag
    )

    """
    Position segments, as concrete Python values.

    Two things are deliberate. First, the CURRENT (uncommitted) K gets its OWN final segment and
    is fed to the matmul directly from ``k_current_T_sb`` -- it is never copied into the block
    buffer. Copying it in was a read-before-write: the copy sits inside the block loop, and the
    matmuls consuming that range could be scheduled ahead of it, which zeroed exactly the
    current-K columns of the score (the committed blocks stayed bit-exact). Using the SBUF
    tensor as the moving operand removes both the copy and the hazard.

    Second, the loop is a plain Python loop over precomputed starts rather than
    ``nl.sequential_range``: the committed range ends at ``current_start``, which is not a
    multiple of POS_BLOCK, so the final committed segment is short. Computing that size from a
    traced index would mean ``min()`` on a tracer.
    """
    has_current = k_current_T_sb != None and current_start != None
    n_committed = current_start if has_current else end_pos
    segments = []  # (start, size, kind) with kind in {"hbm", "sbuf", "zero"}
    for st in range(0, n_committed, POS_BLOCK):
        segments.append((st, min(POS_BLOCK, n_committed - st), "hbm"))
    if has_current:
        segments.append((current_start, current_size, "sbuf"))
        """
        Trailing range [current_start + current_size, end_pos): positions this S-tile does not
        own yet. Every column in [0, end_pos) must still be WRITTEN -- the mask sets these to
        -inf, and the top-k reads the whole padded row -- so leaving them out hands the top-k
        stale SBUF. The old full-width loader got this for free by memsetting k over the entire
        row; the per-segment version has to be explicit. K is treated as zero here (exactly what
        that memset did), so the score contribution is 0 and the result is just the mask: the
        per-head matmuls are skipped entirely rather than multiplying by a zero buffer.

        Only S-tiles before the last have such a range (for the final tile it is empty), which is
        why single-S-tile shapes passed while S=256/512/1024 did not.
        """
        cur_end = current_start + current_size
        for st in range(cur_end, end_pos, POS_BLOCK):
            segments.append((st, min(POS_BLOCK, end_pos - st), "zero"))

    """
    fp32 persist staging comes from the CALLER, not from this scope.

    Correctness, not economy. Allocated here it would be released the moment the score phase's
    scope closes, and the topk phase that follows reuses that SBUF -- while the final segment's
    DMA to score_padded_hbm / index_score could still be in flight. Measured: with the chunked
    topk running, 7939 of 8256 live score cells were wrong; with it stubbed out, 195. Allocating
    in the caller's scope (which encloses BOTH phases) keeps the address off-limits until the
    s-tile is finished.
    """
    kernel_assert(persist_buf != None, "score_and_persist_blocked_bf16 needs a caller-owned persist_buf")
    blk_f32_sb = persist_buf

    sbm.open_scope(interleave_degree=2, name=f"pos_block_loop{_u()}")
    for _seg_i, (pb, blk, kind) in enumerate(segments):
        if kind == "sbuf":
            # Current K already sits transposed in SBUF; use it as the moving operand directly.
            k_src = k_current_T_sb
        elif kind == "hbm":
            k_blk_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"k_blk_s{_seg_i}{_u()}")
            _load_k_block_bf16(k_cache, b_idx, head_dim, k_blk_sb, pb, blk)
            k_src = k_blk_sb
        else:
            k_src = None  # zero-K trailing range: mask only

        score_blk_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"score_blk_s{_seg_i}{_u()}")
        nisa.memset(dst=score_blk_sb[:S, :blk], value=0.0, engine=nisa.engine.gpsimd)

        # Per-head accumulate over just this block's positions (nothing to accumulate when K
        # is zero -- score_blk_sb is already memset, so only the mask add below applies).
        sbm.open_scope(interleave_degree=2, name=f"per_head_blocked{_u()}")
        for h_idx in range(n_heads if k_src != None else 0):
            psum_w = min(blk, 4096)
            # Unique per unrolled head: alloc_stack names must be unique across the whole trace
            # (close_scope frees addresses, not names).
            relu_blk_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"relu_blk_s{_seg_i}_h{h_idx}{_u()}")
            for ps in range(0, blk, psum_w):
                pz = min(psum_w, blk - ps)
                logits_psum = nl.ndarray((P_MAX, psum_w), dtype=nl.bfloat16, buffer=nl.psum)
                for cs in range(ps, ps + pz, CACHE_TILE):
                    cz = min(CACHE_TILE, ps + pz - cs)
                    nisa.nc_matmul(
                        dst=logits_psum[:S, cs - ps : cs - ps + cz],
                        stationary=q_T_all_sb[:head_dim, h_idx, :S],
                        moving=k_src[:head_dim, cs : cs + cz],
                    )
                nisa.activation(
                    dst=relu_blk_sb[:S, ps : ps + pz],
                    op=nl.relu,
                    data=logits_psum[:S, :pz],
                )
            for ct in range(0, blk, CACHE_TILE):
                cz = min(CACHE_TILE, blk - ct)
                nisa.scalar_tensor_tensor(
                    dst=score_blk_sb[:S, ct : ct + cz],
                    data=relu_blk_sb[:S, ct : ct + cz],
                    op0=nl.multiply,
                    operand0=weights_sb[:S, h_idx : h_idx + 1],
                    op1=nl.add,
                    operand1=score_blk_sb[:S, ct : ct + cz],
                )
            sbm.increment_section()
        sbm.close_scope()  # per_head_blocked

        # Additive mask for this block only.
        mask_blk_sb = sbm.alloc_stack((P_MAX, POS_BLOCK), nl.bfloat16, name=f"mask_blk_s{_seg_i}{_u()}")
        nisa.dma_copy(
            dst=mask_blk_sb[:S, :blk],
            src=mask_hbm[mask_row_start : mask_row_start + S, pb : pb + blk],
        )
        nisa.tensor_tensor(
            dst=score_blk_sb[:S, :blk],
            data1=score_blk_sb[:S, :blk],
            data2=mask_blk_sb[:S, :blk],
            op=nl.add,
        )

        # Persist this block: bf16 -> fp32 then one DMA into the padded-score slot.
        nisa.tensor_copy(dst=blk_f32_sb[:S, :blk], src=score_blk_sb[:S, :blk])
        nisa.dma_copy(
            dst=score_padded_hbm[s_tile_idx, part_offset : part_offset + S, pb : pb + blk],
            src=blk_f32_sb[:S, :blk],
            dge_mode=dge_mode.swdge,
        )
        if index_score_hbm != None:
            # Source from the caller-owned fp32 staging, NOT the loop-scoped bf16 score_blk_sb.
            # score_blk_sb dies with the position-block scope, so the topk phase reuses its
            # address while this DMA can still be in flight -- measured as an 88.8% max relative
            # diff on output_0, worst in the final segment's columns (the last DMA issued).
            # blk_f32_sb also already matches index_score_hbm's fp32 dtype.
            nisa.dma_copy(
                dst=index_score_hbm[index_score_row_start : index_score_row_start + S, pb : pb + blk],
                src=blk_f32_sb[:S, :blk],
            )
        sbm.increment_section()
    sbm.close_scope()  # pos_block_loop

    # -inf tail so padded snake slots never win the top-k (the [end_pos, T) remainder
    # is < TOPK_QUERY_CHUNKS wide, so one small block covers it).
    if T > end_pos:
        tail = T - end_pos
        tail_sb = sbm.alloc_stack((P_MAX, TOPK_QUERY_CHUNKS), nl.float32, name=f"score_tail{_u()}")
        nisa.memset(dst=tail_sb[:S, :tail], value=float("-inf"))
        nisa.dma_copy(
            dst=score_padded_hbm[s_tile_idx, part_offset : part_offset + S, end_pos:T],
            src=tail_sb[:S, :tail],
            dge_mode=dge_mode.swdge,
        )

    sbm.close_scope()  # score_blocked_bf16
