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

from ...core.utils.allocator import SbufManager
from ...core.utils.tensor_view import TensorView
from .sparse_attention_indexer_utils import CACHE_TILE, P_MAX


def transpose_k_for_cache(sbm: SbufManager, k_bf16_sb: nl.ndarray, S: int, head_dim: int) -> nl.ndarray:
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
    k_transposed_sb: nl.ndarray,
    k_cache: nl.ndarray,
    b_idx: int,
    cache_pos: int,
    S: int,
    head_dim: int,
) -> None:
    """DMA the pre-transposed [head_dim P, S F] K to the bf16 K cache.

    The transpose is done once by ``transpose_k_for_cache`` and reused by
    ``score_against_cache_bf16`` to avoid a second nc_transpose.
    """
    sbm.open_scope(name="persist_k_bf16")
    nisa.dma_copy(
        dst=k_cache[b_idx, :head_dim, cache_pos : cache_pos + S],
        src=k_transposed_sb[:head_dim, :S],
    )
    sbm.close_scope()


def _load_k_cache_bf16(
    sbm: SbufManager,
    k_cache: nl.ndarray,
    b_idx: int,
    end_pos: int,
    head_dim: int,
    k_full_bf16_sb: nl.ndarray,
    k_current_T_sb: Optional[nl.ndarray] = None,
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
    sbm.open_scope(name="load_k_cache_bf16")

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
    q_full_hbm_view: nl.ndarray,
    cos_sb: nl.ndarray,
    sin_sb: nl.ndarray,
    k_cache: nl.ndarray,
    weights_sb: nl.ndarray,
    score_row_sb: nl.ndarray,
    b_idx: int,
    S: int,
    end_pos: int,
    n_heads: int,
    head_dim: int,
    rope_head_dim: int,
    k_current_T_sb: Optional[nl.ndarray] = None,
    current_start: Optional[int] = None,
    current_size: Optional[int] = None,
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
        q_full_hbm_view (nl.ndarray): [S, n_heads * head_dim], bf16 Q for the current S-tile on HBM.
        cos_sb (nl.ndarray): [S, rope_head_dim // 2], RoPE cosine table in SBUF.
        sin_sb (nl.ndarray): [S, rope_head_dim // 2], RoPE sine table in SBUF.
        k_cache (nl.ndarray): [B, head_dim, max_seq_len], bf16 K cache on HBM.
        weights_sb (nl.ndarray): [S, n_heads], per-head score weights in SBUF.
        score_row_sb (nl.ndarray): [S, end_pos], output score accumulator in SBUF (written in place).
        b_idx (int): Batch index into k_cache.
        S (int): Query sequence length of the current S-tile.
        end_pos (int): Total number of cache positions to score against (current + prior).
        n_heads (int): Number of attention heads.
        head_dim (int): Per-head dimension.
        rope_head_dim (int): Number of leading head-dim elements RoPE is applied to (0 disables RoPE).
        k_current_T_sb (Optional[nl.ndarray]): Pre-transposed [head_dim, S] current K, forwarded to
            _load_k_cache_bf16 to avoid a redundant transpose. Supply together with current_start and
            current_size.
        current_start (Optional[int]): Cache offset where the current K slab starts.
        current_size (Optional[int]): Number of current-K positions.

    Returns:
        None: score_row_sb is updated in place with the accumulated per-head scores.
    """
    sbm.open_scope(name="score_full_bf16")

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

    sbm.open_scope(interleave_degree=2, name="chunk_loop_bf16")
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
            q_view = TensorView(q_chunk_sb[:S, :]).reshape_dim(1, [HEAD_CHUNK, head_dim])
            x_lo_view = q_view.slice(dim=2, start=0, end=rope_half)
            x_hi_view = q_view.slice(dim=2, start=rope_half, end=rope_head_dim)
            cos_bcast = TensorView(cos_sb[:S, :]).expand_dim(1).broadcast(dim=1, size=HEAD_CHUNK)
            sin_bcast = TensorView(sin_sb[:S, :]).expand_dim(1).broadcast(dim=1, size=HEAD_CHUNK)

            t_lo_cos = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_hi_cos = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_lo_sin = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)
            t_hi_sin = sbm.alloc_stack((P_MAX, HEAD_CHUNK, rope_half), nl.float32)

            nisa.tensor_tensor(
                dst=t_lo_cos[:S, :HEAD_CHUNK, :rope_half],
                data1=x_lo_view.get_view(),
                data2=cos_bcast.get_view(),
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=t_hi_cos[:S, :HEAD_CHUNK, :rope_half],
                data1=x_hi_view.get_view(),
                data2=cos_bcast.get_view(),
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=t_lo_sin[:S, :HEAD_CHUNK, :rope_half],
                data1=x_lo_view.get_view(),
                data2=sin_bcast.get_view(),
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=t_hi_sin[:S, :HEAD_CHUNK, :rope_half],
                data1=x_hi_view.get_view(),
                data2=sin_bcast.get_view(),
                op=nl.multiply,
            )
            nisa.tensor_tensor(
                dst=x_lo_view.get_view(),
                data1=t_lo_cos[:S, :HEAD_CHUNK, :rope_half],
                data2=t_hi_sin[:S, :HEAD_CHUNK, :rope_half],
                op=nl.subtract,
            )
            nisa.tensor_tensor(
                dst=x_hi_view.get_view(),
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
        q_chunk_bf16_view = TensorView(q_chunk_sb[:S, :]).reshape_dim(
            1,
            [HEAD_CHUNK, head_dim],
        )
        nisa.dma_transpose(
            dst=q_T_chunk_sb[:head_dim, :HEAD_CHUNK, :S],
            src=q_chunk_bf16_view.get_view(),
            axes=(2, 1, 0),
        )

        """
        Per-head bf16 score matmul + post-processing.
        interleave_degree=2 double-buffers logits_psum so head N+1's
        matmul can overlap with head N's relu+STT.
        """
        sbm.open_scope(interleave_degree=2, name="per_head_score_bf16")
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
