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

"""High-level FlashAttention-2 forward entry point for Trainium 3.

Owns the HBM boundary -- the Q/O block views and the K/V streams -- and drives
:class:`~.flash_attention_fwd.FlashAttentionFwd` over them.
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.allocator import sizeinbytes
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import is_trn3_b1
from ... import neurotile as nt
from .config import D_TILE, SQ_TILE, SV_TILE, FlashAttentionConfig
from .flash_attention_fwd import FlashAttentionFwd
from .remainder_stream import remainder_block_stream


@nki.jit
def flash_attention_2(
    q: nl.NkiTensor,
    k: nl.NkiTensor,
    v: nl.NkiTensor,
    scale: float,
    config: Optional[FlashAttentionConfig] = None,
) -> nl.NkiTensor:
    """FlashAttention-2 forward: ``O = softmax(Q @ K^T * scale) @ V``.

    Unmasked bidirectional attention for long key sequences -- the regime where Sk dominates and the
    softmax must stay online.

    Set ``sq_tiles_per_blk`` to ``ceil(Sq / 128)`` when all of Q fits SBUF: K and V are then streamed
    once for the whole of Q instead of once per Sq block, which is a significant win once Sk is long.

    Dimensions:
        * ``B`` -- batch x heads, collapsed into one axis.
        * ``Bkv`` -- key/value groups; must divide ``B``. ``B // Bkv`` query heads share each KV head,
          which is GQA computed natively rather than by materializing repeated K/V.
        * ``Sq``, ``Sk`` -- query and key sequence lengths. Arbitrary; neither needs to be a multiple
          of any tile size.
        * ``D`` -- head dimension. Must be a positive multiple of 128, at most 512.

    Pseudocode:
        Q's Sq axis is padded to a whole 128-row tile and Sk is cut so that no DMA block mixes
        whole V tiles with a partial one. Six stages run as a software pipeline over a flat walk of
        ``(sk_block, sq_tile, sk_tile)``, each offset a fixed number of steps behind MM1::

            for each (batch, sq_block):
                running_max, running_sum, running_out = -inf, 0, 0
                for each (sk_block, sq_tile, sk_tile) step:
                    scores      = Q_tile @ K_tile * scale        # MM1, fp32 in PSUM
                    block_max   = max(block_max, max(scores))    # chained reduce, Scalar
                    probs       = exp(scores - block_max)        # Vector, bf16 out
                    probs_t     = transpose(probs)               # PE, for MM2's contraction
                    block_out  += probs_t @ V_tile               # MM2, fp32 in PSUM
                    # once per (sk_block, sq_tile), fold the block into the running state:
                    rescale        = exp(running_max - max(running_max, block_max))
                    running_out    = running_out * rescale + block_out
                    running_sum    = running_sum * rescale + sum(probs)
                    running_max    = max(running_max, block_max)
                O_block = running_out / running_sum

    Args:
        q (nl.NkiTensor): ``[B, Sq, D]`` in HBM. Must be a 2-byte dtype (transposed load uses
            ``dma_transpose``).
        k (nl.NkiTensor): ``[Bkv, D, Sk]`` in HBM.
        v (nl.NkiTensor): ``[Bkv, Sk, D]`` in HBM.
        scale (float): Attention-score scale.
        config (Optional[FlashAttentionConfig]): Tuning knobs; defaults are used if omitted.

    Returns:
        nl.NkiTensor: ``[B, Sq, D]`` in shared HBM.

    Notes:
        * **Unmasked.** Computes full bidirectional attention. There is no causal or padding mask, so
          this is not a drop-in prefill kernel for decoder models.
        * **Trn3 B1 only** (NeuronCore-v4 sub-version 1). ``nisa.exponential`` is a VectorE
          instruction that does not exist on Trn3 A0 (``trn3pre``, sub-version 0) or on any earlier
          generation, so ``trn3_a0`` and ``trn3_pds_a0`` are not supported targets.
        * ``Sq`` and ``Sk`` are arbitrary. Only ``D`` is constrained (a positive multiple of 128, at
          most 512, because ``D`` bounds MM2's moving free size).
        * **Compile with** ``--internal-skip-backend-allocation-opt-nki``. The kernel hand-places its
          PSUM banks, and the backend allocation optimization rearranges them, which measurably
          regresses the kernel. This is a leaked implementation detail and should become unnecessary
          once the underlying pass is fixed.
    """
    kernel_assert(
        is_trn3_b1(),
        "flash_attention_2 requires Trn3 B1 (NeuronCore-v4 sub-version 1) for nisa.exponential",
    )
    B, Sq, D = q.shape
    Bkv, d_k, Sk = k.shape
    Sv, d_v = v.shape[1], v.shape[2]

    kernel_assert(d_k == D and d_v == D, f"K/V head dim must match Q, got {d_k=}, {d_v=}, {D=}")
    kernel_assert(Sv == Sk, f"V seq length must match K, got {Sv=}, {Sk=}")
    # dma_transpose on the Q load requires a 2-byte element.
    kernel_assert(sizeinbytes(q.dtype) == 2, f"q must be a 2-byte dtype for the transposed load, got {q.dtype}")
    kernel_assert(B % Bkv == 0, f"B={B} must be a multiple of Bkv={Bkv} for GQA")

    if config == None:
        config = FlashAttentionConfig()
    geometry = config.geometry(D=D, Sq=Sq, Sk=Sk)

    out = nl.ndarray(shape=(B, Sq, D), dtype=q.dtype, buffer=nl.shared_hbm)
    q_heads_per_kv_head = B // Bkv

    fwd = FlashAttentionFwd(scale=scale, geometry=geometry)

    for batch_idx in range(B):
        kv_batch_idx = batch_idx // q_heads_per_kv_head

        q_blks = nt.blocks(
            q[batch_idx], tile_size=(SQ_TILE, D_TILE), block_size=(geometry.sq_tiles_per_blk, geometry.d_tiles)
        )
        o_blks = nt.blocks(out[batch_idx], tile_size=(SQ_TILE, D), block_size=(geometry.sq_tiles_per_blk, 1))
        # K and V are cut at the same SV_TILE granule.
        # Required because V maps Sk on partitions, where a block mixing whole and partial tiles is not allowed.
        k_stream = remainder_block_stream(
            k[kv_batch_idx],
            block_size=(geometry.d_tiles, geometry.sk_tiles_per_blk),
            tile_size=(D_TILE, geometry.sk_tile),
            dim=1,
            buffer_count=geometry.kv_buffer_count,
            align=SV_TILE,
        )
        v_stream = remainder_block_stream(
            v[kv_batch_idx],
            block_size=(geometry.sk_blk // SV_TILE, 1),
            tile_size=(SV_TILE, geometry.D),
            dim=0,
            buffer_count=geometry.kv_buffer_count,
            align=SV_TILE,
        )

        for sq_blk_idx in range(q_blks.shape[0]):
            # Both SBUF buffers are sized to the block being computed, not to the tiling, so a trailing
            # block that is short on tile count needs no narrowing on either the load or the store.
            o_hbm_blk = o_blks[sq_blk_idx, 0]
            o_sbuf = nt.alloc_tiles(
                tile_size=(SQ_TILE, D), grid=(o_hbm_blk.shape[0], 1), buffer_type=nl.sbuf, dtype=out.dtype
            )

            # MM1 contracts over D, so Q is transposed on load: [Sq, d_tiles, D_TILE] -> [D_TILE, d_tiles, Sq].
            q_src_blk = q_blks[sq_blk_idx, 0]
            q_block = nt.alloc_tiles(
                tile_size=(D_TILE, SQ_TILE),
                grid=(q_src_blk.shape[1], q_src_blk.shape[0]),
                buffer_type=nl.sbuf,
                dtype=q.dtype,
            )
            sq_valid = q_src_blk.element_shape[0]
            nisa.dma_transpose(
                dst=q_block.data.reshape_dim(1, (geometry.d_tiles, -1))[:, :, 0:sq_valid],
                src=q_src_blk.rearrange(("p", ("d", "c")), ("p", "d", "c"), {"c": D_TILE}).data,
                axes=(2, 1, 0),
            )

            fwd.run(
                o_block=o_sbuf,
                q_block=q_block,
                k_stream=k_stream,
                v_stream=v_stream,
            )
            # One store per block; a short trailing Sq tile goes separately, as O has Sq on partitions.
            if not o_hbm_blk.is_remainder:
                o_hbm_blk.store(o_sbuf.data)
            else:
                last_sq_tile = o_hbm_blk.shape[0] - 1
                if last_sq_tile > 0:
                    o_hbm_blk[0:last_sq_tile, 0].store(o_sbuf[0:last_sq_tile, 0].data)
                o_rows = o_hbm_blk[last_sq_tile, 0].element_shape[0]
                o_hbm_blk[last_sq_tile, 0].store(o_sbuf[last_sq_tile, 0].data[0:o_rows, :])

    return out
