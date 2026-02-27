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

"""
KV Data Parallel (KVDP) helpers for attention_block_tkg.

KVDP partitions the KV cache across ranks along the batch dimension. Each rank holds
B/KVDP batches of the KV cache. Before attention: all_gather Q heads, slice Q/K/V batch.
After attention: all_gather output batch, slice heads.

See attention_block_tkg_sharding_design_spec.md for detailed design documentation.
"""

import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.tensor_view import TensorView


def _KVDP_attention_input_collectives(
    Q_tkg_sb: nl.ndarray,
    K_tkg_sb: nl.ndarray,
    V_tkg_hbm: nl.ndarray,
    q_heads: int,
    kv_heads: int,
    d_head: int,
    KVDP: int,
    B: int,
    B_attn: int,
    S_tkg: int,
    replica_group: ReplicaGroup,
    sbm,
):
    """Input collectives for KV data parallelism.

    Transforms Q/K/V from full-batch layout to per-rank layout for attention:
    - Q: all_gather heads across ranks, slice batch → each rank has all Q heads for B/KVDP batches
    - K/V: slice batch by rank_id → each rank has K/V for its B/KVDP batches

    Pseudocode (q_heads > 1, general path)::

        def _KVDP_input_gather(Q_sb, K_sb, V_hbm, KVDP, B, ...):
            # 1. Transpose Q from SBUF to HBM (need q_heads on dim=0 for all_gather)
            Q_hbm = transpose(Q_sb)  # (d_head, B*q_heads*S_tkg) → (q_heads, B, S_tkg, d_head)

            # 2. all_gather Q heads across KVDP ranks
            Q_gathered = all_gather(Q_hbm, dim=0)  # (q_heads, B, S_tkg, d_head) → (KVDP*q_heads, B, S_tkg, d_head)

            # 3. Slice batch for this rank using dynamic rank_id
            Q_local = Q_gathered[:, rank_id*B_attn:(rank_id+1)*B_attn, :, :]  # (KVDP*q_heads, B_attn, S_tkg, d_head)

            # 4. Transpose Q back to SBUF for attention
            Q_sb_out = transpose(Q_local)  # (d_head, B_attn*KVDP*q_heads*S_tkg)

            # 5. K: SBUF → HBM, slice batch, HBM → SBUF (dynamic slice requires HBM)
            K_hbm = dma_copy(K_sb)  # (d_head, B*S_tkg)
            K_local = K_hbm[:, rank_id*B_attn*S_tkg:(rank_id+1)*B_attn*S_tkg]  # (d_head, B_attn*S_tkg)
            K_sb_out = dma_copy(K_local)

            # 6. V: slice batch in HBM
            V_local = V_hbm[rank_id*B_attn:(rank_id+1)*B_attn, :, :, :]  # (B_attn, kv_heads, S_tkg, d_head)

            return Q_sb_out, K_sb_out, V_local

    When q_heads == 1, the transpose can be skipped: all_gather on d_head dim directly,
    then rearrange to get the correct SBUF layout.

    Example: TP64 QKV projection → TP8 KVDP8 attention for GPT-OSS (64 q_heads, 8 k_heads) for B=16
        - 64 ranks compute QKV projection, each with q_heads=1, B=16
        - all_gather Q heads within each KVDP group: 1 q_head x KVDP8 → 8 q_heads
        - rank_id slice Q on batch dim: 16 B / KVDP8 → 2 B
        - Each rank now has 8 Q heads for 2 batches for 1 K head (TP8)

    Args:
        Q_tkg_sb (nl.ndarray): [d_head, B * q_heads * S_tkg] @ SBUF - Q after RoPE
        K_tkg_sb (nl.ndarray): [d_head, B * S_tkg] @ SBUF - K after RoPE
        V_tkg_hbm (nl.ndarray): [B, kv_heads, S_tkg, d_head] @ HBM - V from QKV extraction
        q_heads (int): Number of query heads per rank (before gather)
        kv_heads (int): Number of KV heads (always 1 for GQA)
        d_head (int): Head dimension
        KVDP (int): KV data parallelism degree (number of ranks)
        B (int): Total batch size across all ranks
        B_attn (int): Batch size per rank for attention (B / KVDP)
        S_tkg (int): Token generation sequence length
        replica_group (ReplicaGroup): Replica group for collective ops
        sbm: SBUF memory manager

    Returns:
        Q_tkg_sb (nl.ndarray): [d_head, B_attn * q_heads * KVDP * S_tkg] @ SBUF - gathered Q
        K_tkg_sb (nl.ndarray): [d_head, B_attn * S_tkg] @ SBUF - sliced K
        V_tkg_hbm (nl.ndarray): [B_attn, kv_heads, S_tkg, d_head] @ HBM - sliced V for attention_tkg

    Notes:
        - V is returned in HBM because attention_tkg loads V tile-by-tile during P*V matmul
        - Batch selection uses reshape to (KVDP, rest) + dynamic select with rank_id
    """
    dtype = Q_tkg_sb.dtype

    # Shape assertions
    kernel_assert(
        Q_tkg_sb.shape == (d_head, B * q_heads * S_tkg),
        f"Q_tkg_sb shape mismatch: {Q_tkg_sb.shape} != {(d_head, B * q_heads * S_tkg)}",
    )
    kernel_assert(
        K_tkg_sb.shape == (d_head, B * S_tkg),
        f"K_tkg_sb shape mismatch: {K_tkg_sb.shape} != {(d_head, B * S_tkg)}",
    )
    kernel_assert(
        V_tkg_hbm.shape == (B, kv_heads, S_tkg, d_head),
        f"V_tkg_hbm shape mismatch: {V_tkg_hbm.shape} != {(B, kv_heads, S_tkg, d_head)}",
    )

    # Get dynamic rank_id for batch selection
    dynamic_rank_id = ncc.rank_id()

    # ========== Q: all_gather heads, slice batch ==========
    q_heads_attn = q_heads * KVDP

    if q_heads == 1:
        # Optimized path for q_heads=1: no transpose needed
        # SBUF -> HBM for collectives
        Q_hbm = nl.ndarray((d_head, B * q_heads * S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="Q_hbm")
        nisa.dma_copy(Q_hbm, Q_tkg_sb)
        # all_gather on dim=0: (d_head, B*S_tkg) -> (q_heads_attn*d_head, B*S_tkg) where q_heads_attn=KVDP*1
        Q_gathered_hbm = nl.ndarray(
            (q_heads_attn * d_head, B * S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="Q_gathered_hbm"
        )
        ncc.all_gather(dsts=[Q_gathered_hbm], srcs=[Q_hbm], replica_group=replica_group, collective_dim=0)

        # Slice Q batch
        # Reshape to (q_heads_attn, d_head, KVDP, B_attn*S_tkg)
        #                                    ^---- select batch using rank_id on dim=2
        Q_gathered_view = TensorView(Q_gathered_hbm.reshape((q_heads_attn, d_head, KVDP, B_attn * S_tkg))).select(
            dim=2, index=dynamic_rank_id
        )
        Q_sliced_hbm = nl.ndarray(Q_gathered_view.shape, dtype=dtype, buffer=nl.shared_hbm, name="Q_sliced_hbm")
        nisa.dma_copy(dst=Q_sliced_hbm, src=Q_gathered_view.get_view())

        # See explanation below on why we can't combine this dma_copy with the slice dma_copy above
        # DMA to SBUF and rearrange (q_heads_attn, d_head, B_attn, S_tkg) -> (d_head, B_attn, q_heads_attn, S_tkg)
        Q_sliced_view = TensorView(Q_sliced_hbm.reshape((q_heads_attn, d_head, B_attn, S_tkg))).rearrange(
            ("H", "d", "B", "S"), ("d", "B", "H", "S"), {}
        )
        Q_tkg_sb_out = sbm.alloc_stack((d_head, B_attn * q_heads_attn * S_tkg), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(Q_tkg_sb_out.reshape((d_head, B_attn, q_heads_attn, S_tkg)), Q_sliced_view.get_view())
    else:
        # General path: transpose to get q_heads on dim=0 for all_gather
        # Transpose Q to HBM: (d_head, B*q_heads*S) -> (q_heads, B, S_tkg, d_head)
        # Need q_heads on dim=0 for all_gather on Q heads
        kernel_assert(d_head <= nl.tile_size.pmax, f"d_head must be <= {nl.tile_size.pmax}, got {d_head}")
        kernel_assert(
            q_heads * B * S_tkg <= nl.tile_size.psum_fmax,
            f"B * q_heads * S_tkg must be <= {nl.tile_size.psum_fmax}, got {B} * {q_heads} * {S_tkg} = {q_heads * B * S_tkg}",
        )
        # Rearrange q_heads to first dim of free dim before transpose:
        #   (d_head, B, q_heads, S) -> (d_head, q_heads, B, S)
        Q_tkg_sb_rearranged = nl.ndarray((d_head, q_heads * B * S_tkg), dtype=dtype, buffer=nl.sbuf)
        Q_tkg_sb_view = TensorView(Q_tkg_sb.reshape((d_head, B, q_heads, S_tkg))).rearrange(
            ("d", "B", "H", "S"), ("d", "H", "B", "S"), {}
        )
        nisa.tensor_copy(Q_tkg_sb_rearranged, Q_tkg_sb_view.get_view())
        # Transpose: (d_head, q_heads*B*S) -> (q_heads*B*S, d_head)
        Q_psum = nl.ndarray((q_heads * B * S_tkg, d_head), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(Q_psum, Q_tkg_sb_rearranged)
        # Copy PSUM -> SBUF
        Q_transposed = nl.ndarray((q_heads * B * S_tkg, d_head), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(Q_transposed, Q_psum)
        # SBUF -> HBM with target shape (q_heads, B, S_tkg, d_head) for collectives
        Q_hbm = nl.ndarray((q_heads, B, S_tkg, d_head), dtype=dtype, buffer=nl.shared_hbm, name="Q_hbm")
        nisa.dma_copy(Q_hbm.reshape((q_heads * B * S_tkg, d_head)), Q_transposed)

        # all_gather Q on head dim: (q_heads, B, S_tkg, d_head) -> (KVDP*q_heads, B, S_tkg, d_head)
        Q_gathered_hbm = nl.ndarray(
            (q_heads_attn, B, S_tkg, d_head), dtype=dtype, buffer=nl.shared_hbm, name="Q_gathered_hbm"
        )
        ncc.all_gather(dsts=[Q_gathered_hbm], srcs=[Q_hbm], replica_group=replica_group, collective_dim=0)

        # Slice Q batch
        # Reshape to (q_heads_attn, KVDP, B_attn, S_tkg, d_head)
        #                            ^---- select batch on dim=1
        #
        # Why we need two dma_copy below: 1) HBM->HBM batch slice 2) HBM->SBUF transpose
        #
        # Step  Tensor              Shape                    Physical Strides                          Contiguous?
        # ----  ------              -----                    ----------------                          -----------
        # 1     Q_gathered_hbm      (H, B, S, d)             (B·S·d, S·d, d, 1)                        Yes
        # 2     reshape             (H, KVDP, B_attn, S, d)  (KVDP·B_attn·S·d, B_attn·S·d, S·d, d, 1)  Yes
        # 3     select(dim=1)       (H, B_attn, S, d)        (KVDP·B_attn·S·d, S·d, d, 1)              No - stride[0] has KVDP gap
        # 4     Q_sliced_hbm (DMA)  (H, B_attn, S, d)        (B_attn·S·d, S·d, d, 1)                   Yes
        # 5     reshape for SBUF    (H·B_attn·S, d)          (d, 1)                                    Yes
        #
        # After select (step 3), stride[0] still has the KVDP factor, creating gaps where other ranks' data lives.
        # We can't flatten this non-contiguous view to 2D for SBUF load.
        # Additionally, dynamic DMA requires src/dst to have same number of dimensions (4D view -> 2D SBUF fails).
        # The DMA to Q_sliced_hbm materializes the slice into contiguous memory, enabling the reshape in step 5.
        Q_gathered_hbm_batch_slice_view = TensorView(
            Q_gathered_hbm.reshape((q_heads_attn, KVDP, B_attn, S_tkg, d_head))
        ).select(dim=1, index=dynamic_rank_id)
        Q_sliced_hbm = nl.ndarray(
            Q_gathered_hbm_batch_slice_view.shape, dtype=dtype, buffer=nl.shared_hbm, name="Q_sliced_hbm"
        )
        nisa.dma_copy(dst=Q_sliced_hbm, src=Q_gathered_hbm_batch_slice_view.get_view())

        # Load Q to SBUF and transpose: (q_heads_attn, B_attn, S_tkg, d_head) -> (d_head, B_attn*q_heads_attn*S_tkg)
        Q_tkg_sb_out = sbm.alloc_stack((d_head, B_attn * q_heads_attn * S_tkg), dtype=dtype, buffer=nl.sbuf)
        # HBM -> SBUF: flatten to (H*B*S, d) for transpose
        Q_load = nl.ndarray((q_heads_attn * B_attn * S_tkg, d_head), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(Q_load, Q_sliced_hbm.reshape((q_heads_attn * B_attn * S_tkg, d_head)))
        # Transpose: (q_heads_attn*B_attn*S_tkg, d_head) -> (d_head, q_heads_attn*B_attn*S_tkg)
        # Note: q_heads_attn * B_attn * S_tkg <= psum_fmax due to assertion above, so single transpose works
        Q_psum = nl.ndarray((d_head, q_heads_attn * B_attn * S_tkg), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(Q_psum, Q_load)
        # PSUM -> SBUF: (d_head, q_heads_attn, B_attn, S_tkg) -> (d_head, B_attn, q_heads_attn, S_tkg)
        Q_psum_view = TensorView(Q_psum.reshape((d_head, q_heads_attn, B_attn, S_tkg))).rearrange(
            ('d', 'H', 'B', 'S'), ('d', 'B', 'H', 'S'), {}
        )
        nisa.tensor_copy(Q_tkg_sb_out, Q_psum_view.get_view())

    # ========== K: slice batch ==========
    # Dynamic DMA only supports DRAM (HBM), so: SBUF -> HBM -> slice -> SBUF
    K_full = nl.ndarray((d_head, B * S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="K_full")
    nisa.dma_copy(K_full, K_tkg_sb)

    # Slice K batch: (d_head, B*S_tkg) -> (d_head, KVDP, B_attn*S_tkg) -> select -> (d_head, B_attn*S_tkg)
    K_full_view = TensorView(K_full.reshape((d_head, KVDP, B_attn * S_tkg))).select(dim=1, index=dynamic_rank_id)
    K_local = nl.ndarray(K_full_view.shape, dtype=dtype, buffer=nl.shared_hbm, name="K_local")
    nisa.dma_copy(dst=K_local, src=K_full_view.get_view())

    # DMA back to SBUF
    K_tkg_sb_out = sbm.alloc_stack((d_head, B_attn * S_tkg), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(K_tkg_sb_out, K_local)

    # ========== V: slice batch ==========
    # (B, kv_heads=1, S_tkg, d_head) -> (KVDP, B_attn, kv_heads=1, S_tkg, d_head) -> select -> (B_attn, kv_heads=1, S_tkg, d_head)
    V_tkg_hbm_batch_slice_view = TensorView(V_tkg_hbm.reshape((KVDP, B_attn, kv_heads, S_tkg, d_head))).select(
        dim=0, index=dynamic_rank_id
    )
    V_tkg_hbm_out = nl.ndarray(
        V_tkg_hbm_batch_slice_view.shape, dtype=dtype, buffer=nl.shared_hbm, name="V_tkg_hbm_out"
    )
    nisa.dma_copy(dst=V_tkg_hbm_out, src=V_tkg_hbm_batch_slice_view.get_view())

    return Q_tkg_sb_out, K_tkg_sb_out, V_tkg_hbm_out


def _KVDP_attention_output_collectives(
    attn_sb: nl.ndarray,
    V_tkg_hbm: nl.ndarray,
    KVDP: int,
    B_attn: int,
    q_heads: int,
    d_head: int,
    S_tkg: int,
    replica_group: ReplicaGroup,
    sbm,
):
    """Output collectives for KV data parallelism.

    Transforms attention output from per-rank layout back to full-batch layout:
    - Attention output: all_gather batch across ranks, slice heads → each rank has its q_heads for all B batches
    - V: copy from HBM to SBUF for KV cache update

    Pseudocode::

        def _KVDP_output_gather(attn_sb, KVDP, q_heads, B, ...):
            q_heads_attn = q_heads * KVDP

            # 1. Transpose attn from SBUF to HBM (need B_attn on dim=0 for all_gather)
            attn_transposed = transpose(attn_sb)  # (d_head, B_attn*q_heads_attn*S_tkg) → (B_attn*q_heads_attn*S_tkg, d_head)
            attn_hbm = dma_copy(attn_transposed)  # reshape to (B_attn, q_heads_attn, d_head, S_tkg)

            # 2. all_gather on batch dimension
            attn_gathered = all_gather(attn_hbm, dim=0)  # (B_attn, q_heads_attn, d_head, S_tkg) → (B, q_heads_attn, d_head, S_tkg)

            # 3. Slice heads for this rank using dynamic rank_id
            # reshape q_heads_attn to (KVDP, q_heads) and select with rank_id
            attn_local = attn_gathered[:, rank_id*q_heads:(rank_id+1)*q_heads, :, :]  # (B, q_heads, d_head, S_tkg)

            # 4. Transpose back to SBUF for output projection
            attn_load = dma_copy(attn_local)  # flatten to (B*q_heads*S_tkg, d_head)
            attn_sb_out = transpose(attn_load)  # (d_head, B*q_heads*S_tkg)

            # 5. V: copy from HBM to SBUF for KV cache update
            V_sb = dma_copy(V_hbm)  # (B_attn*S_tkg, d_head)

            return attn_sb_out, V_sb

    Example: TP8 KVDP8 attention → TP64 output projection for GPT-OSS (64 q_heads, 8 k_heads) for B=16
        - 8 ranks compute attention, each with q_heads*KVDP=8 heads, B_attn=2 batches
        - all_gather attention output on batch dim: 2 B_attn x KVDP8 → 16 B
        - rank_id slice attention output on head dim: 8 q_heads / KVDP8 → 1 q_head
        - Each rank now has 1 Q head for 16 batches (TP64)

    Args:

        attn_sb (nl.ndarray): [d_head, B_attn * q_heads * KVDP * S_tkg] @ SBUF - attention output
        V_tkg_hbm (nl.ndarray): [B_attn, kv_heads, S_tkg, d_head] @ HBM - V for this rank's batch slice
        KVDP (int): KV data parallelism degree (number of ranks)
        B_attn (int): Batch size per rank for attention (B / KVDP)
        q_heads (int): Number of query heads per rank (after slice)
        d_head (int): Head dimension
        S_tkg (int): Token generation sequence length
        replica_group (ReplicaGroup): Replica group for collective ops
        sbm: SBUF memory manager

    Returns:
        attn_out (nl.ndarray): [d_head, B * q_heads * S_tkg] @ SBUF - gathered attention output
        V_tkg_sb (nl.ndarray): [B_attn * S_tkg, d_head] @ SBUF - V for KV cache update

    Notes:
        - V is copied back to SBUF for KV cache update which requires SBUF input
        - Each rank gets its own q_heads slice of the gathered attention output
    """
    q_heads_attn = q_heads * KVDP
    B = B_attn * KVDP
    dtype = attn_sb.dtype

    # Shape assertions
    kernel_assert(
        attn_sb.shape == (d_head, B_attn * q_heads_attn * S_tkg),
        f"attn_sb shape mismatch: {attn_sb.shape} != {(d_head, B_attn * q_heads_attn * S_tkg)}",
    )
    kernel_assert(
        V_tkg_hbm.shape[0] == B_attn and V_tkg_hbm.shape[2] == S_tkg and V_tkg_hbm.shape[3] == d_head,
        f"V_tkg_hbm shape mismatch: {V_tkg_hbm.shape} != (B_attn={B_attn}, kv_heads, S_tkg={S_tkg}, d_head={d_head})",
    )

    # ========== Attention output: all_gather batch, slice heads ==========
    # Transpose attn to HBM: (d_head, B_attn, q_heads_attn, S_tkg) -> (B_attn, q_heads_attn, d_head, S_tkg)
    # Need B_attn on dim=0 for all_gather on batch dimension
    # TODO: If B_attn==1, can skip transpose (same trick as q_heads==1 in input_collectives)
    kernel_assert(
        B_attn * q_heads_attn * S_tkg <= nl.tile_size.psum_fmax,
        f"B * q_heads * S_tkg must be <= {nl.tile_size.psum_fmax}, got {B} * {q_heads} * {S_tkg} = {B * q_heads * S_tkg}",
    )
    attn_transposed = nl.ndarray((B_attn * q_heads_attn * S_tkg, d_head), dtype=dtype, buffer=nl.sbuf)
    psum_t = nl.ndarray((B_attn * q_heads_attn * S_tkg, d_head), dtype=dtype, buffer=nl.psum)
    nisa.nc_transpose(psum_t, attn_sb)
    nisa.tensor_copy(attn_transposed, psum_t)
    attn_hbm = nl.ndarray(
        (B_attn, q_heads_attn, d_head, S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="attn_pre_gather"
    )
    nisa.dma_copy(attn_hbm.reshape((B_attn * q_heads_attn * S_tkg, d_head)), attn_transposed)

    # all_gather on batch dim: (B_attn, q_heads_attn, d_head, S_tkg) -> (B, q_heads_attn, d_head, S_tkg)
    attn_gathered = nl.ndarray(
        (B, q_heads_attn, d_head, S_tkg), dtype=dtype, buffer=nl.shared_hbm, name="attn_gathered"
    )
    ncc.all_gather(dsts=[attn_gathered], srcs=[attn_hbm], replica_group=replica_group, collective_dim=0)

    # Slice heads: reshape q_heads_attn to (KVDP, q_heads) and select with rank_id
    # (B, q_heads_attn, d_head, S_tkg) -> (B, KVDP, q_heads, d_head, S_tkg) -> select -> (B, q_heads, d_head, S_tkg)
    dynamic_rank_id = ncc.rank_id()
    attn_gathered_head_slice_view = TensorView(attn_gathered.reshape((B, KVDP, q_heads, d_head, S_tkg))).select(
        dim=1, index=dynamic_rank_id
    )
    attn_sliced = nl.ndarray(attn_gathered_head_slice_view.shape, dtype=dtype, buffer=nl.shared_hbm, name="attn_sliced")
    nisa.dma_copy(dst=attn_sliced, src=attn_gathered_head_slice_view.get_view())

    # Transpose back to SBUF: (B, q_heads, d_head, S_tkg) -> (d_head, B*q_heads*S_tkg)
    # B * q_heads * S_tkg = B_attn * q_heads_attn * S_tkg <= psum_fmax (same assertion as above)
    attn_final_sb = sbm.alloc_stack((d_head, B * q_heads * S_tkg), dtype=dtype, buffer=nl.sbuf)
    attn_load = nl.ndarray((B * q_heads * S_tkg, d_head), dtype=dtype, buffer=nl.sbuf)
    nisa.dma_copy(attn_load, attn_sliced.reshape((B * q_heads * S_tkg, d_head)))
    psum = nl.ndarray((d_head, B * q_heads * S_tkg), dtype=dtype, buffer=nl.psum)
    nisa.nc_transpose(psum, attn_load)
    nisa.tensor_copy(attn_final_sb, psum)

    # ========== V: copy from HBM to SBUF for KV cache update ==========
    V_tkg_sb = nl.ndarray((B_attn * S_tkg, d_head), dtype=V_tkg_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(V_tkg_sb, V_tkg_hbm.reshape((B_attn * S_tkg, d_head)))

    return attn_final_sb, V_tkg_sb
