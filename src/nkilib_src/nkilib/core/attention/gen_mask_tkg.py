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
Standalone mask generation kernel for attention TKG.

This kernel generates attention masks with support for:
- Flat KV cache (block_len = 0)
- Block KV cache (block_len > 0)
- Strided and non-strided MM1 layouts
- Cascaded attention with active mask loading

The mask generation matches the K cache layout used by attention_tkg kernel.

Design spec: gen_mask_tkg_design_spec.md
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import SbufManager
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..utils.logging import get_logger
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .attention_tkg_utils import (
    AttnTKGConfig,
    resize_cache_block_len_for_attention_tkg_kernel,
)
from .attention_tkg_utils import (
    is_batch_sharded as is_batch_sharded_fn,
)
from .attention_tkg_utils import (
    is_s_prior_sharded as is_s_prior_sharded_fn,
)

# Free-axis tile width for the swap-layout SWA compute. Bounds the fp32 scratch footprint so the
# full-s_prior free axis (up to the FA tile size) can't overrun SBUF.
_SWA_FREE_TILE = 512


def gen_mask_tkg(
    pos_ids: nl.NkiTensor,
    mask_out: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    is_s_prior_sharded: bool,
    s_prior_per_shard: int,
    start_pos: Optional[nl.NkiTensor] = None,
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
    active_mask: Optional[nl.NkiTensor] = None,
    sbm: Optional[SbufManager] = None,
    is_batch_sharded: bool = False,
    batch_offset: int = 0,
    n_sprior_tile_total: int = 0,
    dynamic_s_prior_offset: Optional[nl.NkiTensor] = None,
    transposed_out: bool = False,
    cp_seq_offset: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """
    Generate attention mask for TKG kernel.

    This function generates prior masks from position IDs with support for both
    flat KV cache and block KV cache. For block KV cache, the mask indices are
    shuffled to match the K cache block layout used by the attention kernel.
    Constraints are the same as the attention_tkg kernel.

    Dimensions:
        bs: Batch size
        q_head: Number of query heads
        s_active: Active sequence length
        n_sprior_tile: Number of prior sequence tiles (derived from mask_out shape)
        P_MAX: Hardware partition dimension (128)

    Block KV Cache Support (block_len > 0):
        When using block KV cache, the K cache has a specific layout where tokens are grouped
        into blocks and distributed across partitions. The mask generation must match this layout
        so that mask[i] corresponds to the correct token at position K_cache[..., i].

        The shuffling formula:
            token_idx = fold_idx * block_len * P_MAX + partition * block_len + blk_offset

    Args:
        pos_ids (nl.NkiTensor): Position IDs tensor in SBUF.
            - transposed_out=False (default): [P_MAX, bs * s_active] with values broadcasted.
            - transposed_out=True (QK-swap): [1, bs * s_active] or [P_MAX, bs * s_active] for consistency.
        mask_out (nl.NkiTensor): Output mask buffer in SBUF.
            - transposed_out=False (default): [P_MAX, n_sprior_tile, bs, q_head, s_active]
            - transposed_out=True (QK-swap): [P_MAX, n_bsq_tiles, s_prior]
        bs (int): Batch size.
        q_head (int): Number of query heads.
        s_active (int): Active sequence length.
        is_s_prior_sharded (bool): Whether s_prior dimension is sharded across LNCs.
        s_prior_per_shard (int): Total s_prior per shard (NC's full s_prior, used for NC offset calculation).
        start_pos (Optional[nl.NkiTensor]): [P_MAX, bs * s_active], Per-query SWA window start (inclusive).
            When None, standard attention mask is generated (iota < pos_ids).
            When provided, per-query banded SWA mask is generated.
        s_prior_offset (int): Offset within current shard (for flash attention tiling). Default: 0.
        block_len (int): Block length for block KV cache (0 = flat cache). Default: 0.
        strided_mm1 (bool): Whether to use strided MM1 layout. Default: True.
        active_mask (Optional[nl.NkiTensor]): [s_active, bs_full, q_head, s_active], Optional active mask
            tensor in HBM. If provided, loaded onto the last section of the mask.
            The batch slice starts at the NC's shard offset plus batch_offset.
        sbm (Optional[SbufManager]): SBUF memory manager. If None, creates a new one.
        is_batch_sharded (bool): Whether batch dimension is sharded across LNCs. Default: False.
            When True: NC0 loads batches [0, bs), NC1 loads batches [bs, bs_full).
            When False: both NCs load batches [0, bs) where bs == bs_full.
        batch_offset (int): Shard-local batch offset into active_mask. Default: 0.
            Added on top of the NC sharding offset (analogous to s_prior_offset for the
            sequence dimension). Used for batch tiling when the full per-NC batch is
            processed in multiple tiles.
        n_sprior_tile_total (int): Total s_prior tiles across all FA tiles (for strided
            iota channel_multiplier). 0 = use n_sprior_tile (no FA tiling). Default: 0.
        dynamic_s_prior_offset (Optional[nl.NkiTensor]): [P_MAX, 1] float32 runtime s_prior offset
            added to the iota, for dynamic FA tiling. None for static tiling. Default: None.
        transposed_out (bool): Emit the transposed mask_out. The caller decides this via
            attention_tkg helper function is_qk_swapped. Default: False.
        cp_seq_offset (Optional[nl.NkiTensor]): [P_MAX, 1] context-parallel global sequence
            offset added to the shard-local iota before the causal compare (same mechanism
            as dynamic_s_prior_offset). None for the non-CP path. Not supported with
            transposed_out=True. Default: None.

    Returns:
        mask_out (nl.NkiTensor): Generated mask tensor.

    Notes:
        - For block KV cache, indices are shuffled to match K cache block layout
        - Supports LNC sharding (lnc=1 and lnc=2 configurations)
        - When batch-sharded with LNC=2, both shards load the active mask
          (each shard places it at the last s_active positions of its own tile space)
        - The mask is initialized to zeros before generation

    Pseudocode:
        # Initialize mask to zeros
        mask_out = zeros()

        # Step 1: Generate index tensor based on cache layout
        if block_len > 0:
            # Block KV: generate shuffled indices
            for fold_idx in range(num_folds):
                iota[p, f] = fold_base + p * block_len + f
        else:
            # Flat KV: generate sequential or strided indices
            iota = generate_iota(strided=strided_mm1)

        # Step 2: Create masks by comparing indices with position IDs
        if start_pos is not None:
            # SWA: per-query banded mask with wrap-around support
            for batch_idx, sa_idx:
                mask = branchless_select(iota, start_pos, pos_ids)
        else:
            # Standard: uniform causal mask
            for batch_idx:
                mask[batch_idx] = (iota < pos_ids[batch_idx])

        # Step 3: Optionally load active mask
        if active_mask is not None:
            load_active_mask_to_last_section(mask_out, active_mask)
    """
    # Hardware partition dim constraint
    P_MAX = nl.tile_size.pmax

    # Determine sharding configuration
    _, lnc, shard_id = get_verified_program_sharding_info("gen_mask_tkg", (0, 1))

    # sprior_prg_id selects which s_prior portion this shard processes:
    # batch-sharded → 0 (both shards see full s_prior), sprior-sharded → shard_id
    sprior_prg_id = shard_id if is_s_prior_sharded else 0

    if sbm is None:
        sbm = SbufManager(0, P_MAX * 128 * 4, get_logger("gen_mask_tkg"), use_auto_alloc=True)

    kernel_assert(
        not (transposed_out and cp_seq_offset is not None),
        "cp_seq_offset is not supported with the transposed_out (QK-swap) mask layout",
    )

    sbm.open_scope(name="gen_mask_tkg")

    # Initialize mask to zeros
    nisa.memset(mask_out, value=0)

    if transposed_out:
        # QK-swap layout: mask_out is [P_MAX (=s_active_bqh row), n_bsq_tiles, s_prior]. s_prior on the
        # free axis, s_active_bqh (bqh row) on partitions.
        _gen_mask_s_active_bqh_partition_layout(
            pos_ids=pos_ids,
            mask_out=mask_out,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            sprior_prg_id=sprior_prg_id,
            s_prior_per_shard=s_prior_per_shard,
            shard_id=shard_id,
            start_pos=start_pos,
            s_prior_offset=s_prior_offset,
            block_len=block_len,
            active_mask=active_mask,
            sbm=sbm,
            is_batch_sharded=is_batch_sharded,
            batch_offset=batch_offset,
        )
    else:
        # Default layout: mask_out is [P_MAX (=s_prior within fold), n_sprior_tile, bs, q_head, s_active].
        _gen_mask_sprior_partition_layout(
            pos_ids=pos_ids,
            mask_out=mask_out,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            sprior_prg_id=sprior_prg_id,
            s_prior_per_shard=s_prior_per_shard,
            shard_id=shard_id,
            start_pos=start_pos,
            s_prior_offset=s_prior_offset,
            block_len=block_len,
            strided_mm1=strided_mm1,
            active_mask=active_mask,
            sbm=sbm,
            is_batch_sharded=is_batch_sharded,
            batch_offset=batch_offset,
            n_sprior_tile_total=n_sprior_tile_total,
            dynamic_s_prior_offset=dynamic_s_prior_offset,
            cp_seq_offset=cp_seq_offset,
        )

    sbm.close_scope()
    return mask_out


def _gen_mask_sprior_partition_layout(
    pos_ids: nl.NkiTensor,
    mask_out: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    sprior_prg_id: int,
    s_prior_per_shard: int,
    shard_id: int,
    start_pos: Optional[nl.NkiTensor] = None,
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
    active_mask: Optional[nl.NkiTensor] = None,
    sbm: Optional[SbufManager] = None,
    is_batch_sharded: bool = False,
    batch_offset: int = 0,
    n_sprior_tile_total: int = 0,
    dynamic_s_prior_offset: Optional[nl.NkiTensor] = None,
    cp_seq_offset: Optional[nl.NkiTensor] = None,
) -> None:
    """
    Default (non-swap) mask layout: mask_out is [P_MAX, n_sprior_tile, bs, q_head, s_active] with
    s_prior on the partition axis (via the iota) and (bs, q_head, s_active) on the free axis
    """
    P_MAX = nl.tile_size.pmax

    kernel_assert(
        len(mask_out.shape) == 5,
        "gen_mask_tkg (default layout) expects a 5D tensor of shape (P_MAX, n_sprior_tile, bs, q_head, "
        f"s_active). Allocate or reshape to a 5D tensor. Got shape {mask_out.shape}",
    )

    kernel_assert(
        not (strided_mm1 and block_len > 0),
        f"strided_mm1=True is incompatible with block KV cache (block_len={block_len}). "
        "Block KV always uses non-strided layout.",
    )

    # Extract and validate dimensions from mask_out shape
    _, n_sprior_tile, _bs, _q_head, _s_active = mask_out.shape

    kernel_assert(_bs == bs, f"mask_out bs dimension {_bs} does not match provided bs {bs}")
    kernel_assert(_q_head == q_head, f"mask_out q_head dimension {_q_head} does not match provided q_head {q_head}")
    kernel_assert(
        _s_active == s_active, f"mask_out s_active dimension {_s_active} does not match provided s_active {s_active}"
    )

    # Create index tensor
    tmp_iota = sbm.alloc_stack(
        (P_MAX, n_sprior_tile), dtype=pos_ids.dtype, buffer=nl.sbuf, name=f"tmp_iota_{s_prior_offset}_{batch_offset}"
    )
    nisa.memset(tmp_iota, value=0)

    # Step 1: Generate index tensor based on cache layout
    _generate_iota_tensor(
        tmp_iota=tmp_iota,
        n_sprior_tile=n_sprior_tile,
        s_prior_per_shard=s_prior_per_shard,
        sprior_prg_id=sprior_prg_id,
        s_prior_offset=0 if dynamic_s_prior_offset is not None else s_prior_offset,
        block_len=block_len,
        strided_mm1=strided_mm1,
        n_sprior_tile_total=n_sprior_tile_total,
    )

    # For dynamic FA tiling, add the runtime s_prior_offset to the iota.
    # dynamic_s_prior_offset is (P_MAX, 1) float32 SBUF tensor, pre-broadcast on partition dim.
    if dynamic_s_prior_offset is not None:
        nisa.activation(dst=tmp_iota, op=nl.copy, data=tmp_iota, bias=dynamic_s_prior_offset, scale=1.0)

    # Context-parallel global sequence offset. When a CP rank holds a disjoint
    # global slice [cp_seq_offset, cp_seq_offset + s_prior) of the prior context
    # but pos_ids are global query positions, shift the shard-local iota into
    # global coordinates so the causal iota < pos_ids compare is correct for
    # ranks past the active context. (P_MAX, 1) bias, same mechanism as
    # dynamic_s_prior_offset. None for the non-CP path (no-op).
    if cp_seq_offset is not None:
        nisa.activation(dst=tmp_iota, op=nl.copy, data=tmp_iota, bias=cp_seq_offset, scale=1.0)

    # Step 2: Create prior masks by per-batch comparison
    # Trace-time routing: SWA path when start_pos is provided, standard path otherwise
    if start_pos is not None:
        # SWA path: each query is processed one at a time, no replication needed
        _create_batch_masks_swa(
            iota=tmp_iota,
            mask_out=mask_out,
            pos_ids=pos_ids,
            start_pos=start_pos,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            n_sprior_tile=n_sprior_tile,
            s_prior_offset=s_prior_offset,
            sbm=sbm,
        )
    else:
        # Standard (non-SWA) path: compare at narrow n_sprior_tile width and broadcast in
        # _create_batch_masks (see its docstring), instead of the ×s_active_qh-wide compare.
        _create_batch_masks(
            iota=tmp_iota,
            mask_out=mask_out,
            pos_ids=pos_ids,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            s_prior_offset=s_prior_offset,
            sbm=sbm,
            batch_offset=batch_offset,
        )

    # Step 3: Optionally load active mask onto the last section of mask_out.
    # Skip if this FA tile doesn't reach the active region (last s_active positions).
    tile_end = s_prior_offset + n_sprior_tile * P_MAX
    if active_mask is not None and (s_prior_per_shard <= 0 or tile_end > s_prior_per_shard - s_active):
        _load_active_mask(
            mask_out=mask_out,
            active_mask=active_mask,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            n_sprior_tile=n_sprior_tile,
            block_len=block_len,
            strided_mm1=strided_mm1,
            shard_id=shard_id,
            is_batch_sharded=is_batch_sharded,
            s_prior_offset=s_prior_offset,
            s_prior_per_shard=s_prior_per_shard,
            batch_offset=batch_offset,
            n_sprior_tile_total=n_sprior_tile_total,
        )


def _gen_mask_s_active_bqh_partition_layout(
    pos_ids: nl.NkiTensor,
    mask_out: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    sprior_prg_id: int,
    s_prior_per_shard: int,
    shard_id: int,
    start_pos: Optional[nl.NkiTensor] = None,
    s_prior_offset: int = 0,
    block_len: int = 0,
    active_mask: Optional[nl.NkiTensor] = None,
    sbm: Optional[SbufManager] = None,
    is_batch_sharded: bool = False,
    batch_offset: int = 0,
) -> None:
    """
    QK-swap mask layout: mask_out is [P_MAX, n_bsq_tiles, s_prior] with s_active_bqh (bqh row) on the partition axis.

    When start_pos is provided, generates a per-query banded SWA mask (transposed counterpart of
    _create_batch_masks_swa): the window start (inclusive) and end (exclusive) sit on the partition axis and
    the token index on the free axis, with branchless wrap-around selection.
    """
    P_MAX = nl.tile_size.pmax

    kernel_assert(
        len(mask_out.shape) == 3,
        "gen_mask_tkg (swap layout) expects a 3D tensor of shape (P_MAX, n_bsq_tiles, s_prior). "
        f"Got shape {mask_out.shape}",
    )
    _, n_bsq_tiles, s_prior_this_tile = mask_out.shape
    s_active_qh = q_head * s_active
    s_active_bqh = n_bsq_tiles * P_MAX
    kernel_assert(
        bs * s_active_qh == s_active_bqh,
        f"mask_out grps {n_bsq_tiles} (s_active_bqh={s_active_bqh}) does not match bs*q_head*s_active "
        f"({bs}*{q_head}*{s_active}={bs * s_active_qh})",
    )

    # Step 1: iota on the FREE axis with the global s_prior token at each free position, identical across
    # partitions (channel_multiplier=0).
    iota_base = sprior_prg_id * s_prior_per_shard + s_prior_offset
    iota_free = sbm.alloc_stack(
        (P_MAX, s_prior_this_tile), dtype=pos_ids.dtype, buffer=nl.sbuf, name=f"iota_free_{s_prior_offset}"
    )
    if block_len > 0:
        # One iota per fold (mirrors the default-layout _generate_iota_tensor) so the folds pipeline
        # instead of a single long op. Within a fold the free order is [f_within, p_slot]: value =
        # fold_base + f_within*1 + p_slot*block_len (the swapaxes'd token order).
        num_folds = s_prior_this_tile // (P_MAX * block_len)
        iota_folds = iota_free.reshape_dim(1, [num_folds, block_len * P_MAX])
        for fold_idx in range(num_folds):
            fold_base = iota_base + fold_idx * P_MAX * block_len
            nisa.iota(
                dst=iota_folds[:, fold_idx, :].reshape_dim(1, [block_len, P_MAX]),
                pattern=[[1, block_len], [block_len, P_MAX]],
                offset=fold_base,
                channel_multiplier=0,
            )
    else:
        # Flat KV: contiguous s_prior on the free axis.
        nisa.iota(dst=iota_free, pattern=[[1, s_prior_this_tile]], offset=iota_base, channel_multiplier=0)

    # Step 2: place each batch's base position (cache_len) on the partition axis. Build a
    # [1, s_active_bqh] row where partition-row order is (batch, q_head, s_active) then PE-transpose onto partition.
    # Allocate as fp32 for precision (bf16 rounds ints > 256).
    pos_row = sbm.alloc_stack((1, s_active_bqh), dtype=nl.float32, buffer=nl.sbuf, name=f"pos_row_{s_prior_offset}")
    base_pos = pos_ids[:1].reshape_dim(1, [bs, s_active])[:, :, 0:1]  # [1, bs, 1] base position per batch
    nisa.tensor_copy(dst=pos_row.reshape_dim(1, [bs, s_active_qh]), src=base_pos.broadcast(2, s_active_qh))

    # pos_partition matches iota_free's dtype so the tensor_tensor(less) operands agree
    pos_partition = sbm.alloc_stack(
        (P_MAX, n_bsq_tiles), dtype=pos_ids.dtype, buffer=nl.sbuf, name=f"pos_partition_{s_prior_offset}"
    )
    for grp in range(n_bsq_tiles):
        tp_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(tp_psum, pos_row[:1, grp * P_MAX : (grp + 1) * P_MAX])
        nisa.tensor_copy(pos_partition[:, grp : grp + 1], tp_psum)

    if start_pos is None:
        # Step 3 (standard causal): mask[p, grp, sp] = iota_free[p, sp] < pos_partition[p, grp] (bcast on free).
        for grp in range(n_bsq_tiles):
            nisa.tensor_tensor(
                mask_out[:, grp, :],
                iota_free,
                pos_partition[:, grp : grp + 1].broadcast(1, s_prior_this_tile),
                op=nl.less,
            )
    else:
        # Step 3 (SWA): per-query banded mask, transposed counterpart of _create_batch_masks_swa. The window
        # start (inclusive) sits on the partition axis alongside the end (=cache_len), and the branchless
        # wrap-around select runs vectorized over the free (s_prior) axis:
        #   normal (start<=end): (iota >= start) AND (iota < end)
        #   wrap   (start> end): (iota >= start) OR  (iota < end)
        #   final = normal + is_wrap * (wrap - normal)
        # The end boundary is pos_partition (=pos_ids[b,0]=cache_len) for every query in a batch, matching
        # _create_batch_masks_swa: positions >= cache_len are active tokens loaded separately in Step 4.

        # Place each query's window start on the partition axis, same (batch, q_head, s_active) row order as
        # pos_partition. start_pos is per-(batch, s_active); broadcast it across q_head.
        start_row = sbm.alloc_stack(
            (1, s_active_bqh), dtype=nl.float32, buffer=nl.sbuf, name=f"start_row_{s_prior_offset}"
        )
        start_src = start_pos[:1].reshape_dim(1, [bs, s_active]).expand_dim(2).broadcast(2, q_head)  # [1,bs,qh,sa]
        nisa.tensor_copy(dst=start_row.reshape_dim(1, [bs, q_head, s_active]), src=start_src)
        start_partition = sbm.alloc_stack(
            (P_MAX, n_bsq_tiles), dtype=pos_ids.dtype, buffer=nl.sbuf, name=f"start_partition_{s_prior_offset}"
        )
        for grp in range(n_bsq_tiles):
            tp_psum = nl.ndarray((P_MAX, 1), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(tp_psum, start_row[:1, grp * P_MAX : (grp + 1) * P_MAX])
            nisa.tensor_copy(start_partition[:, grp : grp + 1], tp_psum)

        # Scratch buffers (float32 for tensor_scalar/tensor_tensor arithmetic; final op casts to mask dtype).
        # Unlike the default layout the swap layout carries the full s_prior on the free axis.
        # Process the free axis in bounded tiles.
        chunk_w = min(s_prior_this_tile, _SWA_FREE_TILE)
        buf_ge = sbm.alloc_stack((P_MAX, chunk_w), dtype=nl.float32, buffer=nl.sbuf, name=f"swa_ge_{s_prior_offset}")
        buf_lt = sbm.alloc_stack((P_MAX, chunk_w), dtype=nl.float32, buffer=nl.sbuf, name=f"swa_lt_{s_prior_offset}")
        buf_normal = sbm.alloc_stack(
            (P_MAX, chunk_w), dtype=nl.float32, buffer=nl.sbuf, name=f"swa_normal_{s_prior_offset}"
        )
        for grp in range(n_bsq_tiles):
            for c0 in range(0, s_prior_this_tile, chunk_w):
                cw = min(chunk_w, s_prior_this_tile - c0)
                iota_c = iota_free[:, c0 : c0 + cw]
                ge, lt, normal = buf_ge[:, :cw], buf_lt[:, :cw], buf_normal[:, :cw]
                # ge = (iota >= start)
                nisa.tensor_scalar(
                    dst=ge, data=iota_c, op0=nl.greater_equal, operand0=start_partition[:, grp : grp + 1]
                )
                # lt = (iota < end), end = pos_partition (cache_len)
                nisa.tensor_scalar(dst=lt, data=iota_c, op0=nl.less, operand0=pos_partition[:, grp : grp + 1])
                # normal = ge AND lt
                nisa.tensor_tensor(normal, ge, lt, op=nl.multiply)
                # wrap = ge OR lt  (reuse ge)
                nisa.tensor_tensor(ge, ge, lt, op=nl.maximum)
                # diff = wrap - normal  (reuse ge)
                nisa.tensor_tensor(ge, ge, normal, op=nl.subtract)
                # is_wrap = (start > end); reuse buf_lt's first column now that the lt mask is consumed.
                nisa.tensor_tensor(
                    buf_lt[:, 0:1], start_partition[:, grp : grp + 1], pos_partition[:, grp : grp + 1], op=nl.greater
                )
                # scaled_diff = diff * is_wrap  (reuse ge)
                nisa.tensor_scalar(dst=ge, data=ge, op0=nl.multiply, operand0=buf_lt[:, 0:1])
                # final = normal + scaled_diff
                nisa.tensor_tensor(mask_out[:, grp, c0 : c0 + cw], normal, ge, op=nl.add)

    # Step 4: active mask placement (only when this tile reaches the active region).
    tile_end = s_prior_offset + s_prior_this_tile
    if active_mask is not None and (s_prior_per_shard <= 0 or tile_end > s_prior_per_shard - s_active):
        _load_active_mask_block_kv_swap(
            mask_out=mask_out,
            active_mask=active_mask,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            n_bsq_tiles=n_bsq_tiles,
            block_len=block_len,
            shard_id=shard_id,
            is_batch_sharded=is_batch_sharded,
            s_prior_offset=s_prior_offset,
            s_prior_per_shard=s_prior_per_shard,
            batch_offset=batch_offset,
            sbm=sbm,
        )


# ============================================================================
# Helper Functions
# ============================================================================


def _generate_iota_tensor(
    tmp_iota: nl.NkiTensor,
    n_sprior_tile: int,
    s_prior_per_shard: int,
    sprior_prg_id: int,
    s_prior_offset: int,
    block_len: int,
    strided_mm1: bool,
    n_sprior_tile_total: int = 0,
) -> None:
    """
    Generate index tensor based on cache layout.

    For block KV cache (block_len > 0), generates shuffled indices to match
    the K cache block layout used by the attention kernel.

    For flat KV cache (block_len = 0), generates sequential or strided indices
    based on the strided_mm1 setting.

    Args:
        tmp_iota: Output tensor to store generated indices. Shape [P_MAX, n_sprior_tile].
        n_sprior_tile: Number of s_prior tiles.
        s_prior_per_shard: Total s_prior per shard (for NC offset calculation in LNC sharding).
        sprior_prg_id: Shard ID (0 or 1 for LNC=2).
        s_prior_offset: Offset within current shard (for flash attention tiling).
        block_len: Block length for block KV cache (0 = flat cache).
        strided_mm1: Whether to use strided MM1 layout.
        n_sprior_tile_total: Total s_prior tiles across all FA tiles (for strided
            iota channel_multiplier). 0 = use n_sprior_tile (no FA tiling). Default: 0.
    """
    P_MAX = nl.tile_size.pmax
    iota_base = sprior_prg_id * s_prior_per_shard + s_prior_offset

    if block_len > 0:
        # Block KV: generate shuffled indices to match K cache block layout.
        #
        # The golden does .swapaxes(-1, -2) on (P_MAX, block_len) dims.
        # After swapaxes, linear index i = fold * block_len * P_MAX + f * P_MAX + p
        # maps to original token position = fold * P_MAX * block_len + p * block_len + f.
        #
        # So kernel needs: iota[p, f] = fold_base + p * block_len + f
        #
        # Using iota pattern=[[1, block_len]] with channel_multiplier=block_len:
        #     For partition p, free dim f: value = offset + f * 1 + p * block_len
        # This gives: fold_base + p * block_len + f (correct!)
        num_folds = n_sprior_tile // block_len

        for fold_idx in range(num_folds):
            fold_base = iota_base + fold_idx * P_MAX * block_len
            nisa.iota(
                dst=tmp_iota[:, nl.ds(fold_idx * block_len, block_len)],
                pattern=[[1, block_len]],
                offset=fold_base,
                channel_multiplier=block_len,
            )
    else:
        # Flat KV cache: for FA-tiled strided layout, channel_multiplier uses
        # the global tile count and iota_base is in tile-index space.
        iota_ch_mul = n_sprior_tile
        if strided_mm1 and n_sprior_tile_total > 0:
            iota_ch_mul = n_sprior_tile_total
            iota_base = sprior_prg_id * (s_prior_per_shard // P_MAX) + s_prior_offset // P_MAX

        iota_pattern = [[1, n_sprior_tile]] if strided_mm1 else [[P_MAX, n_sprior_tile]]
        iota_multiplier = iota_ch_mul if strided_mm1 else 1

        nisa.iota(
            dst=tmp_iota[...],
            pattern=iota_pattern,
            offset=iota_base,
            channel_multiplier=iota_multiplier,
        )


def _create_batch_masks(
    iota: nl.NkiTensor,
    mask_out: nl.NkiTensor,
    pos_ids: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior_offset: int,
    sbm: SbufManager,
    batch_offset: int = 0,
) -> None:
    """
    Create prior masks by per-batch comparison with position IDs.

    For each batch, generates a mask by comparing the index tensor (iota) against the corresponding
    position ID. Because the causal predicate ``iota < pos_ids[batch]`` is head- and
    active-token-independent, the compare runs once at the narrow n_sprior_tile width and the boolean
    result is broadcast across (q_head, s_active) when copied into mask_out, avoiding the redundant
    ×s_active_qh-wide compare.

    Args:
        iota: Index tensor for comparison. Shape [P_MAX, n_sprior_tile].
        mask_out: Output mask buffer. Shape [P_MAX, n_sprior_tile, bs, q_head, s_active].
        pos_ids: Position IDs tensor. Shape [P_MAX, bs * s_active].
        bs: Batch size.
        q_head: Number of query heads.
        s_active: Active sequence length.
        s_prior_offset (int): Offset within current shard (for flash attention tiling, used here for tensor naming). Default: 0.
        sbm: SBUF memory manager.
        batch_offset (int): Batch offset for tensor naming uniqueness. Default: 0.
    """
    for batch_idx in range(bs):
        cur_mask = sbm.alloc_stack(
            iota.shape,
            dtype=mask_out.dtype,
            buffer=nl.sbuf,
            name=f"cur_mask_{batch_idx}_{s_prior_offset}_{batch_offset}",
        )
        nisa.tensor_scalar(
            dst=cur_mask[...],
            data=iota[...],
            op0=nl.less,
            operand0=pos_ids[:, nl.ds(batch_idx * s_active, 1)],
        )

        # Copy the [P_MAX, n_sprior_tile] mask for this batch to mask_out, broadcasting the
        # head-independent result across (q_head, s_active) via free-axis broadcast on the source.
        mask_out_pat = mask_out.select(2, batch_idx)
        cur_mask_bcast = cur_mask.expand_dim(2).broadcast(2, q_head).expand_dim(3).broadcast(3, s_active)

        # Alternate between scalar and vector engines for better performance
        if batch_idx % 2 == 0:
            nisa.tensor_copy(mask_out_pat, cur_mask_bcast, engine=nisa.scalar_engine)
        else:
            nisa.tensor_copy(mask_out_pat, cur_mask_bcast, engine=nisa.vector_engine)


def _create_batch_masks_swa(
    iota: nl.NkiTensor,
    mask_out: nl.NkiTensor,
    pos_ids: nl.NkiTensor,
    start_pos: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    n_sprior_tile: int,
    s_prior_offset: int,
    sbm: SbufManager,
) -> None:
    """
    Create per-query banded SWA masks with branchless wrap-around selection.

    For each query (batch, s_active_idx), the mask is:
      - Normal (start <= end): (iota >= start) AND (iota < end)
      - Wrap-around (start > end): (iota >= start) OR (iota < end)

    Branchless selection: final = normal + is_wrap * (wrap - normal)

    Processes each (batch, sa_idx) independently using n_sprior_tile-sized
    scratch buffers, then copies the result directly to mask_out for each
    q_head. Uses the raw iota tensor directly since each query is processed
    one at a time.

    The prior mask end boundary is always pos_ids[b, 0] (the base position =
    cache_lens[b]) for every query in a batch, NOT pos_ids[b, i].  Positions
    cache_lens[b] .. cache_lens[b]+i-1 are active tokens that live in the
    active KV buffer, not the prior cache.  Using pos_ids[b, i] would
    incorrectly mark stale prior-cache slots as attended.

    Args:
        iota: Raw index tensor. Shape [P_MAX, n_sprior_tile].
        mask_out: Output mask buffer. Shape [P_MAX, n_sprior_tile, bs, q_head, s_active].
        pos_ids: End position IDs (exclusive). Shape [P_MAX, bs * s_active].
            pos_ids[:, b*s_active + i] = cache_lens[b] + i.  Only the first
            element per batch (i=0) is used as the prior mask end.
        start_pos: Start position IDs (inclusive). Shape [P_MAX, bs * s_active].
        bs: Batch size.
        q_head: Number of query heads.
        s_active: Active sequence length.
        n_sprior_tile: Number of s_prior tiles.
        s_prior_offset: Offset within current shard (for tensor naming).
        sbm: SBUF memory manager.
    """
    P_MAX = nl.tile_size.pmax
    tile_shape = (P_MAX, n_sprior_tile)

    for batch_idx in range(bs):
        # Per-query scratch buffers (n_sprior_tile wide)
        # Must be float32 for nisa.tensor_scalar arithmetic (hardware requirement).
        # The final nisa.tensor_copy to mask_out handles the implicit dtype cast.
        buf_ge = sbm.alloc_stack(
            tile_shape,
            dtype=nl.float32,
            buffer=nl.sbuf,
            name=f"swa_ge_{batch_idx}_{s_prior_offset}",
        )
        buf_lt = sbm.alloc_stack(
            tile_shape,
            dtype=nl.float32,
            buffer=nl.sbuf,
            name=f"swa_lt_{batch_idx}_{s_prior_offset}",
        )
        buf_scratch = sbm.alloc_stack(
            tile_shape,
            dtype=nl.float32,
            buffer=nl.sbuf,
            name=f"swa_scratch_{batch_idx}_{s_prior_offset}",
        )

        # Prior mask end = pos_ids[b, 0] = cache_lens[b] for all queries in this batch.
        # Positions >= cache_lens[b] are active tokens (not in the prior cache).
        base_col_idx = batch_idx * s_active

        for sa_idx in range(s_active):
            col_idx = batch_idx * s_active + sa_idx

            # Step 1: ge = (iota >= start)
            nisa.tensor_scalar(
                dst=buf_ge[...],
                data=iota[...],
                op0=nl.greater_equal,
                operand0=start_pos[:, nl.ds(col_idx, 1)],
            )

            # Step 2: lt = (iota < end), end = pos_ids[b, 0] (base position)
            nisa.tensor_scalar(
                dst=buf_lt[...],
                data=iota[...],
                op0=nl.less,
                operand0=pos_ids[:, nl.ds(base_col_idx, 1)],
            )

            # Step 3: normal = ge AND lt (multiply for binary)
            nisa.tensor_tensor(buf_scratch[...], buf_ge[...], buf_lt[...], op=nl.multiply)

            # Step 4: wrap = ge OR lt (max for binary)
            nisa.tensor_tensor(buf_ge[...], buf_ge[...], buf_lt[...], op=nl.maximum)

            # Step 5: diff = wrap - normal
            nisa.tensor_tensor(buf_ge[...], buf_ge[...], buf_scratch[...], op=nl.subtract)

            # Step 6: is_wrap = (start > end), end = pos_ids[b, 0]
            nisa.tensor_tensor(
                buf_lt[:, nl.ds(0, 1)],
                start_pos[:, nl.ds(col_idx, 1)],
                pos_ids[:, nl.ds(base_col_idx, 1)],
                op=nl.greater,
            )

            # Step 7: scaled_diff = diff * is_wrap
            nisa.tensor_scalar(
                dst=buf_ge[...],
                data=buf_ge[...],
                op0=nl.multiply,
                operand0=buf_lt[:, nl.ds(0, 1)],
            )

            # Step 8: final = normal + scaled_diff
            nisa.tensor_tensor(buf_ge[...], buf_scratch[...], buf_ge[...], op=nl.add)

            # Step 9: Copy result to mask_out for each q_head
            for qh_idx in range(q_head):
                out_view = (
                    mask_out.select(2, batch_idx)  # [P_MAX, n_sprior_tile, q_head, s_active]
                    .select(2, qh_idx)  # [P_MAX, n_sprior_tile, s_active]
                    .select(2, sa_idx)  # [P_MAX, n_sprior_tile]
                )
                if (batch_idx + qh_idx + sa_idx) % 2 == 0:
                    nisa.tensor_copy(out_view, buf_ge[...], engine=nisa.scalar_engine)
                else:
                    nisa.tensor_copy(out_view, buf_ge[...], engine=nisa.vector_engine)


def _load_active_mask(
    mask_out: nl.NkiTensor,
    active_mask: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    n_sprior_tile: int,
    block_len: int,
    strided_mm1: bool,
    shard_id: int,
    is_batch_sharded: bool,
    s_prior_offset: int = 0,
    s_prior_per_shard: int = 0,
    batch_offset: int = 0,
    n_sprior_tile_total: int = 0,
) -> None:
    """
    Load active mask onto the last section of mask_out.

    Handles three cases:
    1. Block KV: Load active mask with shuffled block layout.
    2. Strided MM1: Load active mask in strided manner across partitions.
    3. Non-strided: Load to bottom right chunk of mask_out.

    Callers must ensure the current tile overlaps the active region before
    calling this function.

    Batch offset logic:
    - batch_offset specifies a shard-local offset into the batch dimension.
    - The effective batch start is: (shard_id * bs if batch-sharded else 0) + batch_offset.
    - Analogous to s_prior_offset for the sequence dimension.

    For LNC=2 with sprior-sharding:
    - Shard 0 processes s_prior positions [0, s_prior/2)
    - Shard 1 processes s_prior positions [s_prior/2, s_prior)
    Both shards load the active mask into the last s_active positions of their
    own tile space.

    Args:
        mask_out: Output mask buffer. Shape [P_MAX, n_sprior_tile, bs, q_head, s_active].
        active_mask: Active mask tensor in HBM. Shape [s_active, bs_full, q_head, s_active].
        bs: Batch size for this tile (may be less than full per-NC batch when batch tiling).
        q_head: Number of query heads.
        s_active: Active sequence length.
        n_sprior_tile: Number of s_prior tiles.
        block_len: Block length for block KV cache (0 = flat cache).
        strided_mm1: Whether to use strided MM1 layout.
        shard_id: Current shard ID (0 or 1 for LNC=2).
        is_batch_sharded: Whether sharding is on batch dimension.
        s_prior_offset: Offset within current shard (for FA tiling). Default: 0.
        s_prior_per_shard: Total s_prior per shard. Default: 0.
        batch_offset: Shard-local batch offset into active_mask. Default: 0.
        n_sprior_tile_total: Total s_prior tiles across all FA tiles (for strided
            partition mapping). 0 = use n_sprior_tile. Default: 0.
    """
    P_MAX = nl.tile_size.pmax

    # batch_start = NC sharding offset + batch tiling offset (analogous to s_prior_offset)
    # When batch-sharded, each NC owns half of active_mask's batch dimension.
    # Use active_mask.shape[1] to derive the per-NC size (bs may be a smaller batch tile).
    nc_batch_offset = shard_id * (active_mask.shape[1] // 2) if is_batch_sharded else 0
    batch_start = nc_batch_offset + batch_offset

    if block_len > 0:
        _load_active_mask_block_kv(
            mask_out=mask_out,
            active_mask=active_mask,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            n_sprior_tile=n_sprior_tile,
            block_len=block_len,
            batch_start=batch_start,
            s_prior_offset=s_prior_offset,
            s_prior_per_shard=s_prior_per_shard,
        )
    elif strided_mm1:
        # Strided MM1: load active mask using per-position coordinate mapping.
        # For strided layout, position pos maps to (p=pos//N, f=pos%N) where
        # N is the total tile count.  When FA-tiled, n_sprior_tile is the
        # tile-local count but coordinates must use the global count so
        # that active positions land at the correct (p, f) coordinates.
        n_tile_for_stride = n_sprior_tile_total if n_sprior_tile_total > 0 else n_sprior_tile

        # Active positions occupy the last s_active slots of the strided space.
        # Use n_tile_for_stride * P_MAX (the full strided capacity) so that
        # p stays within [0, P_MAX) and f_global spans the tile columns.
        active_start = n_tile_for_stride * P_MAX - s_active

        for k_idx in range(s_active):
            linear_pos = active_start + k_idx
            p = linear_pos // n_tile_for_stride
            f_global = linear_pos % n_tile_for_stride

            # When FA-tiled, skip positions outside this tile's range
            if n_sprior_tile_total > 0 and (
                f_global < s_prior_offset // P_MAX or f_global >= s_prior_offset // P_MAX + n_sprior_tile
            ):
                continue

            f_local = f_global - s_prior_offset // P_MAX if n_sprior_tile_total > 0 else f_global

            active_mask_view = (
                active_mask.select(0, k_idx)  # [bs_full, q_head, s_active]
                .slice(0, batch_start, batch_start + bs)  # [bs, q_head, s_active]
                .expand_dim(0)  # [1, bs, q_head, s_active]
                .expand_dim(0)  # [1, 1, bs, q_head, s_active]
            )
            nisa.dma_copy(
                mask_out[p : p + 1, f_local : f_local + 1, :, :, :],
                active_mask_view,
                name=f"active_mask_strided_{k_idx}_bo{batch_offset}_sp{s_prior_offset}",
            )
    else:
        # Non-strided: load to bottom-right chunk of size [s_active, 1, bs, q_head, s_active]
        active_mask_view = (
            active_mask.slice(1, batch_start, batch_start + bs).expand_dim(  # [s_active, bs, q_head, s_active]
                1
            )  # [s_active, 1, bs, q_head, s_active]
        )
        nisa.dma_copy(
            mask_out[P_MAX - s_active :, n_sprior_tile - 1 : n_sprior_tile, :, :, :],
            active_mask_view,
            name=f"active_mask_sequential_bo{batch_offset}_sp{s_prior_offset}",
        )


def _load_active_mask_block_kv(
    mask_out: nl.NkiTensor,
    active_mask: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    n_sprior_tile: int,
    block_len: int,
    batch_start: int,
    s_prior_offset: int = 0,
    s_prior_per_shard: int = 0,
) -> None:
    """
    Load active mask for block KV layout.

    For block KV cache, the active tokens occupy the last s_active positions
    in the sequence. These positions are shuffled according to the block layout.

    Args:
        mask_out: Output mask buffer. Shape [P_MAX, n_sprior_tile, bs, q_head, s_active].
        active_mask: Active mask tensor in HBM. Shape [s_active, bs_full, q_head, s_active].
        bs: Batch size per NC.
        q_head: Number of query heads.
        s_active: Active sequence length.
        n_sprior_tile: Number of s_prior tiles for this FA tile.
        block_len: Block length for block KV cache.
        batch_start: Starting batch index in active_mask.
        s_prior_offset: Offset within current shard (for FA tiling).
        s_prior_per_shard: Total s_prior per shard.
    """
    P_MAX = nl.tile_size.pmax

    # Use s_prior_per_shard if provided, otherwise compute from tile dimensions
    if s_prior_per_shard == 0:
        num_folds = n_sprior_tile // block_len
        s_prior_per_shard = P_MAX * block_len * num_folds

    # Tile boundaries within this shard
    tile_size = n_sprior_tile * P_MAX
    tile_end = s_prior_offset + tile_size

    # Active positions are at the END of s_prior_per_shard
    active_start_linear = s_prior_per_shard - s_active

    # Load each active position that falls within this tile (s_active is typically 1-8)
    for k_idx in range(s_active):
        linear_pos = active_start_linear + k_idx

        # Skip positions outside this tile
        if linear_pos < s_prior_offset or linear_pos >= tile_end:
            continue

        # Reverse the iota formula to get (p, f) coordinates
        pos_in_tile = linear_pos - s_prior_offset
        fold_within_tile = pos_in_tile // (P_MAX * block_len)
        within_fold = pos_in_tile % (P_MAX * block_len)
        p = within_fold // block_len
        f = fold_within_tile * block_len + (within_fold % block_len)

        # Copy active_mask[k_idx, batch_start:batch_start+bs] → mask_out[p, f]
        active_mask_view = (
            active_mask.select(0, k_idx)  # [s_active, bs_full, q_head, s_active]  # [bs_full, q_head, s_active]
            .slice(0, batch_start, batch_start + bs)  # [bs, q_head, s_active]
            .expand_dim(0)  # [1, bs, q_head, s_active]
            .expand_dim(0)  # [1, 1, bs, q_head, s_active]
        )
        nisa.dma_copy(
            mask_out[p : p + 1, f : f + 1, :, :, :],
            active_mask_view,
            name=f"active_mask_block_kv_{k_idx}_bo{batch_start}_sp{s_prior_offset}",
        )


def _load_active_mask_block_kv_swap(
    mask_out: nl.NkiTensor,
    active_mask: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    n_bsq_tiles: int,
    block_len: int,
    shard_id: int,
    is_batch_sharded: bool,
    s_prior_offset: int,
    s_prior_per_shard: int,
    batch_offset: int,
    sbm: SbufManager,
) -> None:
    """
    Load active mask for the QK-swap (s_active_bqh-partition) block KV layout.

    Swap counterpart to _load_active_mask_block_kv. The active tokens occupy the last s_active positions
    of s_prior_per_shard. For each active token, its global s_prior position maps to one free column
    (via the same block-fold shuffle the prior-mask iota uses), and the per-query causal pattern lands
    on the partition axis: partition row p of column-tile group grp is s_active_bqh index grp*P_MAX+p.
    active_mask's (batch, q_head, s_active) free order matches that partition-row order, so each active
    token is DMA'd into a [1, s_active_bqh] row and PE-transposed onto the partition axis at its column.

    Args:
        mask_out: Output mask buffer. Shape [P_MAX, n_bsq_tiles, s_prior] (s_active_bqh on partition).
        active_mask: Active mask tensor in HBM. Shape [s_active, bs_full, q_head, s_active].
        bs: Batch size per NC.
        q_head: Number of query heads.
        s_active: Active sequence length.
        n_bsq_tiles: Number of column-tile (s_active_bqh) groups in mask_out.
        block_len: Block length for block KV cache.
        shard_id: This shard's NeuronCore ID (0 or 1), for the batch-sharded active_mask offset.
        is_batch_sharded: Whether batch is sharded across LNCs.
        s_prior_offset: Offset within current shard (for FA tiling).
        s_prior_per_shard: Total s_prior per shard.
        batch_offset: Shard-local batch offset into active_mask (for batch tiling).
        sbm: SBUF memory manager (for the staging row + transpose).
    """
    P_MAX = nl.tile_size.pmax
    s_active_qh = q_head * s_active

    # batch_start = NC sharding offset + batch tiling offset (mirrors _load_active_mask).
    nc_batch_offset = shard_id * (active_mask.shape[1] // 2) if is_batch_sharded else 0
    batch_start = nc_batch_offset + batch_offset

    if s_prior_per_shard == 0:
        s_prior_per_shard = s_prior_offset + n_bsq_tiles * P_MAX  # single-tile fallback
    tile_size = s_prior_this_tile = mask_out.shape[2]
    tile_end = s_prior_offset + tile_size
    active_start_linear = s_prior_per_shard - s_active

    for k in range(s_active):
        linear_pos = active_start_linear + k
        # Skip active positions outside this FA tile.
        if linear_pos < s_prior_offset or linear_pos >= tile_end:
            continue

        # Free column of this token within the tile: invert the prior-mask fold shuffle.
        # global-token t (within tile) = fold*(P_MAX*block_len) + p_slot*block_len + f_within;
        # free index = fold*(P_MAX*block_len) + f_within*P_MAX + p_slot.
        pos_in_tile = linear_pos - s_prior_offset
        fold, rem = divmod(pos_in_tile, P_MAX * block_len)
        p_slot, f_within = divmod(rem, block_len)
        free_col = fold * (P_MAX * block_len) + f_within * P_MAX + p_slot

        # active_mask[k, batch_start:batch_start+bs, :, :] -> [1, s_active_bqh] free vector in
        # (batch, q_head, sa) order, matching the partition-row order.
        # load as bf16 row for PE transpose then cast back to uint8 mask_out.
        act_row = sbm.alloc_stack(
            (1, bs * s_active_qh), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"act_row_{k}_{s_prior_offset}"
        )
        nisa.dma_copy(
            act_row.reshape_dim(1, [bs, q_head, s_active]),
            active_mask.select(0, k).slice(0, batch_start, batch_start + bs).expand_dim(0),
        )
        # Transpose each 128-wide bqh tile onto the partition axis at free_col.
        for grp in range(n_bsq_tiles):
            tp_psum = nl.ndarray((P_MAX, 1), dtype=nl.bfloat16, buffer=nl.psum)
            nisa.nc_transpose(tp_psum, act_row[:1, grp * P_MAX : (grp + 1) * P_MAX])
            nisa.tensor_copy(mask_out[:, grp, free_col : free_col + 1], tp_psum)


# ============================================================================
# HBM Wrapper
# ============================================================================


def _load_active_mask_hbm(
    mask_out: nl.NkiTensor,
    active_mask: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    n_sprior_tile: int,
    block_len: int,
    strided_mm1: bool,
    shard_id: int,
    s_prior_offset: int,
    s_prior_per_shard: int,
    batch_start: int,
    n_sprior_tile_total: int,
) -> None:
    """Load active mask from HBM into the SBUF tile for the HBM wrapper.

    Active positions occupy the last ``s_active`` positions of the full
    s_prior.  For s_prior-sharded configs only the owning shard loads them.
    Strided layouts use per-position coordinate mapping; non-strided and
    block-KV layouts delegate to ``_load_active_mask``.

    Callers must ensure the current tile overlaps the active region before
    calling this function.
    """
    P_MAX = nl.tile_size.pmax
    n_sprior_tile_per_shard = s_prior_per_shard // P_MAX
    is_sprior_sharded = n_sprior_tile_per_shard < n_sprior_tile_total

    if strided_mm1 and block_len == 0:
        # Strided layout: map each active position to (p, f_global) and load
        # only those falling within this FA tile's range.
        n_tile_for_stride = n_sprior_tile_total if n_sprior_tile_total > 0 else n_sprior_tile

        if is_sprior_sharded:
            shard_tile_base = shard_id * n_sprior_tile_per_shard
            fa_tile_base = shard_tile_base + s_prior_offset // P_MAX
            active_start = n_sprior_tile_total * P_MAX - s_active
        else:
            fa_tile_base = s_prior_offset // P_MAX
            active_start = s_prior_per_shard - s_active if s_prior_per_shard > 0 else n_sprior_tile * P_MAX - s_active

        for k_idx in range(s_active):
            linear_pos = active_start + k_idx
            p = linear_pos // n_tile_for_stride
            f_global = linear_pos % n_tile_for_stride

            if f_global < fa_tile_base or f_global >= fa_tile_base + n_sprior_tile:
                continue

            f_local = f_global - fa_tile_base

            active_mask_view = (
                active_mask.select(0, k_idx).slice(0, batch_start, batch_start + bs).expand_dim(0).expand_dim(0)
            )
            nisa.dma_copy(
                mask_out[p : p + 1, f_local : f_local + 1, :, :, :],
                active_mask_view,
                name=f"active_mask_hbm_strided_{k_idx}_sp{s_prior_offset}_b{batch_start}",
            )
    else:
        # Non-strided / block-KV: delegate to _load_active_mask.
        # With s_prior sharding, only the last shard owns active positions.
        if is_sprior_sharded:
            is_last_shard = (shard_id + 1) * n_sprior_tile_per_shard >= n_sprior_tile_total
            if not is_last_shard:
                return
            fa_n_tile_total = 0
        else:
            fa_n_tile_total = n_sprior_tile_total if n_sprior_tile < n_sprior_tile_total else 0

        _load_active_mask(
            mask_out=mask_out,
            active_mask=active_mask,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            n_sprior_tile=n_sprior_tile,
            block_len=block_len,
            strided_mm1=strided_mm1,
            shard_id=shard_id,
            is_batch_sharded=False,
            s_prior_offset=s_prior_offset,
            s_prior_per_shard=s_prior_per_shard,
            batch_offset=batch_start,
            n_sprior_tile_total=fa_n_tile_total,
        )


@nki.jit
def gen_mask_tkg_hbm(
    pos_ids_hbm: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos_hbm: Optional[nl.NkiTensor] = None,
    block_len: int = 0,
    active_mask: Optional[nl.NkiTensor] = None,
    enable_fa_s_prior_tiling: bool = True,
    fuse_rope: bool = False,
    transposed_out: bool = False,
    cp_seq_offset: int = 0,
) -> nl.NkiTensor:
    """HBM wrapper for gen_mask_tkg.

    Accepts HBM-resident tensors, manages SBUF allocation, DMA transfers,
    P_MAX broadcast, LNC sharding, and tiling internally.

    Tiles over s_prior and batch to stay within SBUF capacity.  The tiling
    is orthogonal to layout (strided / block KV) — the inner kernel handles
    all layout complexity.

    Args:
        pos_ids_hbm: [1, bs * s_active] position IDs in HBM.
        bs: Batch size.
        q_head: Number of query heads.
        s_active: Active sequence length.
        s_prior: Total prior sequence length (must be divisible by P_MAX).
        start_pos_hbm: Optional [1, bs * s_active] SWA start positions in HBM.
        block_len: Block length for block KV cache (0 = flat). Default: 0.
        active_mask: Optional [s_active, bs, q_head, s_active] active mask in HBM.
        enable_fa_s_prior_tiling: Whether flash attention tiling is enabled. Must match
            the value passed to attention_tkg / attention_block_tkg. Default: True.
        fuse_rope: Whether RoPE is fused (impacts LNC sharding decision).
        transposed_out: Emit the QK-swap (s_active_bqh-major) layout instead of the default
            s_prior-major layout. The caller decides this via attention tkg utility helper
            is_qk_swapped. Default: False.
        cp_seq_offset: Context-parallel global sequence offset added to the shard-local
            iota before the causal comparison. When a CP rank holds a disjoint global
            slice [cp_seq_offset, cp_seq_offset + s_prior) of the prior context but
            pos_ids are global query positions, this shifts the local indices into
            global coordinates so ``iota < pos_ids`` is correct for ranks past the
            active context. 0 = no offset (non-CP path). Default: 0.

    Returns:
        mask_out_hbm: generated uint8 mask in HBM. Layout depends on transposed_out:
            - transposed_out=False (default): [s_prior, bs, q_head, s_active]
            - transposed_out=True (QK-swap):  [bs, q_head, s_active, s_prior]

    Example:
        The mask layout MUST match the layout the attention kernel expects for this config.
        Use attention_tkg_utils::is_qk_swapped to derive transposed_out from the same logic
        attention tkg kernel uses for the decision.

            transposed_out = is_qk_swapped(
                bs=bs,
                q_head=q_head,
                d_head=d_head,
                s_active=s_active,
                curr_sprior=s_prior,
                lnc=nl.num_programs(0),
                p_max=nl.tile_size.pmax,
                is_block_kv=block_len > 0,
                is_2byte_kv=sizeinbytes(k_prior.dtype) == 2,
                fp8_packed=fp8_packed,
                fuse_rope=fuse_rope,
                kv_heads=kv_heads, # optional, calculates q_head//kv_heads internally
            )
            mask = gen_mask_tkg_hbm(
                pos_ids_hbm,
                bs,
                q_head,
                s_active,
                s_prior,
                block_len=block_len,
                fuse_rope=fuse_rope,
                transposed_out=transposed_out,
            )
            # mask now matches the layout attention_tkg reads for this config.
    """
    P_MAX = nl.tile_size.pmax

    kernel_assert(
        not (transposed_out and cp_seq_offset != 0),
        "cp_seq_offset is not supported with the transposed_out (QK-swap) mask layout",
    )

    strided_mm1 = block_len == 0

    # LNC sharding
    _, lnc, nc_id = get_verified_program_sharding_info("gen_mask_tkg_hbm", (0, 1))

    cfg = AttnTKGConfig(bs=bs, q_head=q_head, s_active=s_active, curr_sprior=s_prior, fuse_rope=fuse_rope)

    if lnc == 2 and is_s_prior_sharded_fn(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, P_MAX, cfg.fuse_rope):
        sprior_sharded = True
        batch_sharded = False
        s_prior_per_shard = s_prior // lnc
    elif lnc == 2 and is_batch_sharded_fn(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, P_MAX, cfg.fuse_rope):
        sprior_sharded = False
        batch_sharded = True
        s_prior_per_shard = s_prior
    else:
        sprior_sharded = False
        batch_sharded = False
        s_prior_per_shard = s_prior

    # Adjust block_len for hardware constraints (same resize the attention kernel applies).
    if block_len > 0:
        num_blocks_per_batch = s_prior // block_len
        block_len, _ = resize_cache_block_len_for_attention_tkg_kernel(
            num_blocks_per_batch,
            block_len,
            lnc,
            P_MAX,
            bs,
            q_head,
            s_active,
            enable_fa_s_prior_tiling=enable_fa_s_prior_tiling,
            fuse_rope=fuse_rope,
        )

    if transposed_out:
        return _gen_mask_tkg_hbm_s_active_bqh_partition_layout(
            pos_ids_hbm=pos_ids_hbm,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            s_prior=s_prior,
            start_pos_hbm=start_pos_hbm,
            block_len=block_len,
            active_mask=active_mask,
            lnc=lnc,
            nc_id=nc_id,
            sprior_sharded=sprior_sharded,
            batch_sharded=batch_sharded,
            s_prior_per_shard=s_prior_per_shard,
        )
    else:
        return _gen_mask_tkg_hbm_sprior_partition_layout(
            pos_ids_hbm=pos_ids_hbm,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            s_prior=s_prior,
            start_pos_hbm=start_pos_hbm,
            block_len=block_len,
            active_mask=active_mask,
            strided_mm1=strided_mm1,
            lnc=lnc,
            nc_id=nc_id,
            sprior_sharded=sprior_sharded,
            batch_sharded=batch_sharded,
            s_prior_per_shard=s_prior_per_shard,
            cp_seq_offset=cp_seq_offset,
        )


def _gen_mask_tkg_hbm_sprior_partition_layout(
    pos_ids_hbm: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos_hbm: Optional[nl.NkiTensor],
    block_len: int,
    active_mask: Optional[nl.NkiTensor],
    strided_mm1: bool,
    lnc: int,
    nc_id: int,
    sprior_sharded: bool,
    batch_sharded: bool,
    s_prior_per_shard: int,
    cp_seq_offset: int = 0,
) -> nl.NkiTensor:
    """
    Generate the default s_prior-major HBM mask (the non-transposed gen_mask_tkg_hbm output path).

    Because batch is a free dimension in this layout, the SBUF mask tile grows with bs, so this tiles
    over BOTH s_prior and batch to stay within SBUF capacity. It calls the inner gen_mask_tkg for each
    tile and DMA-stores each result. For flat KV, mask row k belongs to KV token k. Block KV keeps its
    existing folded [tile, partition] order. Supports SWA (start_pos_hbm) and both LNC sharding modes.

    Args:
        pos_ids_hbm (nl.NkiTensor): [1, bs * s_active], Position IDs in HBM.
        bs (int): Batch size.
        q_head (int): Number of query heads.
        s_active (int): Active sequence length.
        s_prior (int): Total prior sequence length (must be divisible by P_MAX).
        start_pos_hbm (Optional[nl.NkiTensor]): [1, bs * s_active], SWA window start positions in HBM.
            None for standard (non-SWA) masks.
        block_len (int): Block length for block KV cache (0 = flat cache), already hardware-resized.
        active_mask (Optional[nl.NkiTensor]): [s_active, bs, q_head, s_active], Optional active mask in HBM.
        strided_mm1 (bool): Whether MM1 uses the strided (flat-KV) layout.
        lnc (int): Number of LNC shards (1 or 2).
        nc_id (int): This shard's NeuronCore ID (0 or 1).
        sprior_sharded (bool): Whether s_prior is sharded across LNCs.
        batch_sharded (bool): Whether batch is sharded across LNCs.
        s_prior_per_shard (int): s_prior owned by this shard (s_prior // lnc when s_prior-sharded, else s_prior).
        cp_seq_offset (int): Context-parallel global sequence offset added to the shard-local
            iota before the causal compare. 0 = no offset (non-CP path). Default: 0.

    Returns:
        mask_out_hbm (nl.NkiTensor): [s_prior, bs, q_head, s_active] mask in HBM.
    """
    P_MAX = nl.tile_size.pmax

    n_sprior_tile_total = s_prior // P_MAX
    n_sprior_tile_per_shard = s_prior_per_shard // P_MAX
    elem_size = 8

    # Tile sizes for s_prior and batch dimensions
    SBUF_BUDGET = 16 * 1024 * 1024

    per_sprior_tile_per_batch = P_MAX * q_head * s_active * elem_size

    # Block KV: each tile must hold at least one full fold (block_len tiles).
    min_sprior_tiles = block_len if block_len > 0 else 1

    max_sprior_tiles = max(min_sprior_tiles, SBUF_BUDGET // (per_sprior_tile_per_batch * bs))

    # Block KV alignment: tile must be divisible by block_len
    if block_len > 0 and max_sprior_tiles >= block_len:
        max_sprior_tiles = (max_sprior_tiles // block_len) * block_len
    if block_len > 0 and 0 < max_sprior_tiles < n_sprior_tile_per_shard:
        while n_sprior_tile_per_shard % max_sprior_tiles != 0:
            max_sprior_tiles -= 1
            while max_sprior_tiles > min_sprior_tiles and max_sprior_tiles % block_len != 0:
                max_sprior_tiles -= 1

    if max_sprior_tiles > n_sprior_tile_per_shard:
        max_sprior_tiles = n_sprior_tile_per_shard

    # Even-tile adjustment: all tiles same shape for trace reuse.
    if 0 < max_sprior_tiles < n_sprior_tile_per_shard:
        while n_sprior_tile_per_shard % max_sprior_tiles != 0 and max_sprior_tiles > min_sprior_tiles:
            max_sprior_tiles -= 1

    tile_mask_size = P_MAX * max_sprior_tiles * q_head * s_active * elem_size
    max_bs_tile = max(1, SBUF_BUDGET // tile_mask_size)
    if max_bs_tile > bs:
        max_bs_tile = bs

    n_sprior_tiles = div_ceil(n_sprior_tile_per_shard, max_sprior_tiles)
    n_batch_tiles = div_ceil(bs, max_bs_tile)

    # SbufManager budget
    mask_buf_size = P_MAX * max_sprior_tiles * max_bs_tile * q_head * s_active * elem_size
    input_buf_size = P_MAX * bs * s_active * elem_size
    if start_pos_hbm is not None:
        input_buf_size *= 2
    sbm_budget = mask_buf_size + input_buf_size
    sbm = SbufManager(0, sbm_budget, get_logger("gen_mask_tkg_hbm"), use_auto_alloc=True)
    sbm.open_scope(name="gen_mask_tkg_hbm")

    # nisa.tensor_scalar requires fp32 operands (hardware constraint), so all
    # SBUF mask compute stays fp32.  The HBM output, however, is emitted as
    # uint8 (mask values are binary 0/1): the final DMA store casts fp32->uint8,
    # matching attention_tkg's uint8 `mask_sb` load buffer.  This quarters the
    # mask's HBM footprint and DMA bytes vs. fp32.
    compute_dtype = nl.float32
    out_dtype = nl.uint8

    # Allocate HBM output in the order expected by the attention kernel.
    # Each SBUF output tile has shape [P_MAX, cur_sprior_tiles, ...].
    if block_len == 0:
        # Flat KV HBM layout: [P_MAX, n_sprior_tile_total, bs, q_head, s_active].
        # After flattening the first two dimensions, row k is at
        # [partition=k // n_sprior_tile_total, tile=k % n_sprior_tile_total].
        # Therefore, HBM mask row k belongs to KV token k.
        mask_out_result = nl.ndarray(
            (P_MAX, n_sprior_tile_total, bs, q_head, s_active),
            dtype=out_dtype,
            buffer=nl.shared_hbm,
            name="mask_out_result",
        )
    else:
        # Block KV keeps the existing folded [tile, partition, ...] HBM layout:
        # [n_sprior_tile_total, P_MAX, bs, q_head, s_active] (n_sprior_tile-major).
        # Row width is P_MAX = 128. LNC shard boundaries are multiples of
        # s_prior_per_shard = n_sprior_tile_per_shard * P_MAX, and FA tile
        # boundaries are multiples of fa_tile_size = fa_tile_n_sprior * P_MAX.
        # Both therefore fall between rows.
        # The attention kernel restores the SBUF dimension order with
        # reshape_dim(0, [n_sprior_tile, P_MAX]).permute([1, 0, ...]).
        mask_out_result = nl.ndarray(
            (n_sprior_tile_total, P_MAX, bs, q_head, s_active),
            dtype=out_dtype,
            buffer=nl.shared_hbm,
            name="mask_out_result",
        )

    # Load pos_ids into SBUF with P_MAX broadcast (cast to fp32 if needed)
    pos_ids_sbuf = sbm.alloc_stack((P_MAX, bs * s_active), dtype=compute_dtype, buffer=nl.sbuf, name="pos_ids_sbuf")
    nisa.dma_copy(dst=pos_ids_sbuf[0:1, :], src=pos_ids_hbm)
    stream_shuffle_broadcast(src=pos_ids_sbuf[0:1, :], dst=pos_ids_sbuf)

    start_pos_sbuf = None
    if start_pos_hbm is not None:
        start_pos_sbuf = sbm.alloc_stack(
            (P_MAX, bs * s_active), dtype=compute_dtype, buffer=nl.sbuf, name="start_pos_sbuf"
        )
        nisa.dma_copy(dst=start_pos_sbuf[0:1, :], src=start_pos_hbm)
        stream_shuffle_broadcast(src=start_pos_sbuf[0:1, :], dst=start_pos_sbuf)

    # Context-parallel global sequence offset. The inner kernel applies it as a
    # (P_MAX, 1) bias broadcast on the partition dim onto the shard-local iota
    # (same shape/mechanism as dynamic_s_prior_offset). Built once and reused
    # across all tiles. None when cp_seq_offset == 0 (no-op, non-CP path).
    cp_seq_offset_sbuf = None
    if cp_seq_offset != 0:
        cp_seq_offset_sbuf = sbm.alloc_stack((P_MAX, 1), dtype=compute_dtype, buffer=nl.sbuf, name="cp_seq_offset_sbuf")
        nisa.memset(cp_seq_offset_sbuf, value=float(cp_seq_offset))

    # Tile loop
    hbm_shard_base = nc_id * n_sprior_tile_per_shard if sprior_sharded else 0

    nc_batch_start = 0
    if batch_sharded:
        bs_per_nc = bs // lnc
        nc_batch_start = nc_id * bs_per_nc

    for sp_idx in range(n_sprior_tiles):
        sp_offset = sp_idx * max_sprior_tiles * P_MAX
        cur_sprior_tiles = min(max_sprior_tiles, n_sprior_tile_per_shard - sp_idx * max_sprior_tiles)
        hbm_sp_offset = hbm_shard_base + sp_idx * max_sprior_tiles

        for bt_idx in range(n_batch_tiles):
            b_start = bt_idx * max_bs_tile
            bs_tile = min(max_bs_tile, bs - b_start)

            sbm.open_scope(name=f"tile_sp{sp_idx}_bt{bt_idx}")

            mask_out_sbuf = sbm.alloc_stack(
                (P_MAX, cur_sprior_tiles, bs_tile, q_head, s_active),
                dtype=compute_dtype,
                buffer=nl.sbuf,
                name=f"mask_out_sbuf_sp{sp_idx}_bt{bt_idx}",
            )

            pos_ids_tile = pos_ids_sbuf[:, b_start * s_active : (b_start + bs_tile) * s_active]
            start_pos_tile = None
            if start_pos_sbuf is not None:
                start_pos_tile = start_pos_sbuf[:, b_start * s_active : (b_start + bs_tile) * s_active]

            gen_mask_tkg(
                pos_ids=pos_ids_tile,
                mask_out=mask_out_sbuf,
                bs=bs_tile,
                q_head=q_head,
                s_active=s_active,
                is_s_prior_sharded=sprior_sharded,
                s_prior_per_shard=s_prior_per_shard,
                start_pos=start_pos_tile,
                s_prior_offset=sp_offset,
                block_len=block_len,
                strided_mm1=strided_mm1,
                active_mask=None,
                sbm=sbm,
                is_batch_sharded=False,
                batch_offset=b_start,
                n_sprior_tile_total=n_sprior_tile_total if strided_mm1 and block_len == 0 else 0,
                transposed_out=False,
                cp_seq_offset=cp_seq_offset_sbuf,
            )

            # Batch-sharded overlap: only the shard's portion is valid.
            if batch_sharded:
                tile_batch_end = b_start + bs_tile
                overlap_start = max(b_start, nc_batch_start)
                overlap_end = min(tile_batch_end, nc_batch_start + bs_per_nc)
                has_overlap = overlap_start < overlap_end
                overlap_local = overlap_start - b_start
                overlap_bs = overlap_end - overlap_start if has_overlap else 0
            else:
                has_overlap = True
                overlap_start = b_start
                overlap_local = 0
                overlap_bs = bs_tile

            # Active mask (wrapper handles offsets; inner kernel path bypassed)
            sp_tile_end = sp_offset + cur_sprior_tiles * P_MAX
            if (
                active_mask is not None
                and has_overlap
                and (s_prior_per_shard <= 0 or sp_tile_end > s_prior_per_shard - s_active)
            ):
                _load_active_mask_hbm(
                    mask_out=mask_out_sbuf[:, :, overlap_local : overlap_local + overlap_bs, :, :],
                    active_mask=active_mask,
                    bs=overlap_bs,
                    q_head=q_head,
                    s_active=s_active,
                    n_sprior_tile=cur_sprior_tiles,
                    block_len=block_len,
                    strided_mm1=strided_mm1,
                    shard_id=nc_id,
                    s_prior_offset=sp_offset,
                    s_prior_per_shard=s_prior_per_shard,
                    batch_start=overlap_start,
                    n_sprior_tile_total=n_sprior_tile_total,
                )

            # SBUF: [P_MAX, cur_sprior_tiles, bs_tile, q_head, s_active]
            if has_overlap:
                if block_len == 0:
                    # Flat KV HBM is [P_MAX, n_sprior_tile_total, ...], so its
                    # [partition, tile, ...] order already matches SBUF.
                    dst_view = mask_out_result.slice(
                        dim=1, start=hbm_sp_offset, end=hbm_sp_offset + cur_sprior_tiles
                    ).slice(dim=2, start=overlap_start, end=overlap_start + overlap_bs)
                else:
                    # Block KV HBM is [n_sprior_tile_total, P_MAX, ...], so permute
                    # its [tile, partition, ...] view to match SBUF for the DMA.
                    dst_view = (
                        mask_out_result.permute([1, 0, 2, 3, 4])
                        .slice(dim=1, start=hbm_sp_offset, end=hbm_sp_offset + cur_sprior_tiles)
                        .slice(dim=2, start=overlap_start, end=overlap_start + overlap_bs)
                    )
                nisa.dma_copy(
                    dst=dst_view,
                    src=mask_out_sbuf[:, :, overlap_local : overlap_local + overlap_bs, :, :],
                )

            sbm.close_scope()

    sbm.close_scope()

    return mask_out_result.reshape((s_prior, bs, q_head, s_active))


def _gen_mask_tkg_hbm_s_active_bqh_partition_layout(
    pos_ids_hbm: nl.NkiTensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos_hbm: Optional[nl.NkiTensor],
    block_len: int,
    active_mask: Optional[nl.NkiTensor],
    lnc: int,
    nc_id: int,
    sprior_sharded: bool,
    batch_sharded: bool,
    s_prior_per_shard: int,
) -> nl.NkiTensor:
    """
    Generate the transposed s_active_bqh-major HBM mask (the QK-swap gen_mask_tkg_hbm output path).

    Batch (folded into s_active_bqh) lives on the partition axis, so it never needs tiling; only
    s_prior (the free axis) is tiled in whole-fold chunks to stay within SBUF capacity, calling the
    inner gen_mask_tkg (transposed_out=True) per tile and storing each into the HBM result.
    Supports SWA (start_pos_hbm) and both LNC sharding modes.

    Args:
        pos_ids_hbm (nl.NkiTensor): [1, bs * s_active], Position IDs in HBM.
        bs (int): Batch size.
        q_head (int): Number of query heads.
        s_active (int): Active sequence length.
        s_prior (int): Total prior sequence length (must be divisible by P_MAX).
        start_pos_hbm (Optional[nl.NkiTensor]): [1, bs * s_active], SWA window start positions in HBM.
            None for standard (non-SWA) masks.
        block_len (int): Block length for block KV cache, already hardware-resized (swap is block-KV only).
        active_mask (Optional[nl.NkiTensor]): [s_active, bs, q_head, s_active], Optional active mask in HBM.
        lnc (int): Number of LNC shards (1 or 2).
        nc_id (int): This shard's NeuronCore ID (0 or 1).
        sprior_sharded (bool): Whether s_prior is sharded across LNCs.
        batch_sharded (bool): Whether batch is sharded across LNCs.
        s_prior_per_shard (int): s_prior owned by this shard (s_prior // lnc when s_prior-sharded, else s_prior).

    Returns:
        mask_out_hbm (nl.NkiTensor): [bs, q_head, s_active, s_prior] mask in HBM.
    """
    P_MAX = nl.tile_size.pmax
    s_active_qh = q_head * s_active

    bs_per_nc = bs
    nc_batch_start = 0
    if batch_sharded:
        bs_per_nc = bs // lnc
        nc_batch_start = nc_id * bs_per_nc

    n_bsq_tiles_total = bs * s_active_qh // P_MAX
    n_bsq_tiles_shard = bs_per_nc * s_active_qh // P_MAX
    # grp base of this shard's batch (multiple of P_MAX, 0 unless batch-sharded).
    grp_base = nc_batch_start * s_active_qh // P_MAX
    # This NC's base offset on the global s_prior axis (0 unless s_prior-sharded).
    hbm_shard_base = nc_id * s_prior_per_shard if sprior_sharded else 0

    # nisa.tensor_scalar requires fp32 operands (hardware constraint), so all SBUF mask compute
    # stays fp32.  The HBM output is emitted as uint8 (binary 0/1 mask); the final DMA store casts
    # fp32->uint8, matching attention_tkg's uint8 `mask_sb` load buffer.
    compute_dtype = nl.float32
    out_dtype = nl.uint8

    # Free-axis (s_prior) tiling for SBUF budget. A full [P_MAX, n_bsq_tiles_shard, s_prior_per_shard]
    # float32 tile can exceed the 16MB SBUF budget, so tile the shard's s_prior on the free axis in
    # whole-fold chunks (fold_size = P_MAX * block_len for block KV, or a single P_MAX tile for flat KV).
    SBUF_BUDGET = 16 * 1024 * 1024
    elem_size = 4  # float32
    fold_size = P_MAX * block_len if block_len > 0 else P_MAX

    s_prior_tile = SBUF_BUDGET // (P_MAX * max(1, n_bsq_tiles_shard) * elem_size)
    s_prior_tile = (s_prior_tile // fold_size) * fold_size
    if s_prior_tile < fold_size:
        s_prior_tile = fold_size
    if s_prior_tile > s_prior_per_shard:
        s_prior_tile = s_prior_per_shard
    # Even tiles (all same shape) for trace reuse.
    while s_prior_per_shard % s_prior_tile != 0 and s_prior_tile > fold_size:
        s_prior_tile -= fold_size
    n_sprior_tiles = div_ceil(s_prior_per_shard, s_prior_tile)

    # SbufManager budget: mask_out tile + inner temps (iota_free, pos rows, active-mask row). SWA adds
    # three P_MAX x min(s_prior_tile, _SWA_FREE_TILE) scratch buffers (buf_ge/buf_lt/buf_normal) plus start_pos_sbuf.
    mask_buf_size = P_MAX * n_bsq_tiles_shard * s_prior_tile * elem_size
    swa_scratch_size = 3 * P_MAX * min(s_prior_tile, _SWA_FREE_TILE) * elem_size if start_pos_hbm is not None else 0
    input_buf_size = P_MAX * bs * s_active * elem_size
    sbm_budget = mask_buf_size + swa_scratch_size + 5 * input_buf_size
    sbm = SbufManager(0, sbm_budget, get_logger("gen_mask_tkg_hbm_swap"), use_auto_alloc=True)
    sbm.open_scope(name="gen_mask_tkg_hbm_swap")

    # HBM output laid out [n_bsq_tiles_total, P_MAX, s_prior] so that flattening gives
    # [s_active_bqh_total, s_prior] = [bs, q_head, s_active, s_prior].
    mask_out_result = nl.ndarray(
        (n_bsq_tiles_total, P_MAX, s_prior),
        dtype=out_dtype,
        buffer=nl.shared_hbm,
        name="mask_out_result",
    )

    pos_ids_sbuf = sbm.alloc_stack((1, bs * s_active), dtype=compute_dtype, buffer=nl.sbuf, name="pos_ids_sbuf")
    nisa.dma_copy(dst=pos_ids_sbuf, src=pos_ids_hbm)

    # Slice pos_ids to this shard's batch range.
    pos_ids_shard = pos_ids_sbuf[:, nc_batch_start * s_active : (nc_batch_start + bs_per_nc) * s_active]

    start_pos_shard = None
    if start_pos_hbm is not None:
        start_pos_sbuf = sbm.alloc_stack((1, bs * s_active), dtype=compute_dtype, buffer=nl.sbuf, name="start_pos_sbuf")
        nisa.dma_copy(dst=start_pos_sbuf, src=start_pos_hbm)
        start_pos_shard = start_pos_sbuf[:, nc_batch_start * s_active : (nc_batch_start + bs_per_nc) * s_active]

    for sp_idx in range(n_sprior_tiles):
        sp_offset = sp_idx * s_prior_tile  # offset within this shard's s_prior range
        cur_sprior = min(s_prior_tile, s_prior_per_shard - sp_offset)

        sbm.open_scope(name=f"swap_sp{sp_idx}")

        mask_out_sbuf = sbm.alloc_stack(
            (P_MAX, n_bsq_tiles_shard, cur_sprior),
            dtype=compute_dtype,
            buffer=nl.sbuf,
            name=f"mask_out_sbuf_swap_sp{sp_idx}",
        )

        gen_mask_tkg(
            pos_ids=pos_ids_shard,
            mask_out=mask_out_sbuf,
            bs=bs_per_nc,
            q_head=q_head,
            s_active=s_active,
            is_s_prior_sharded=sprior_sharded,
            s_prior_per_shard=s_prior_per_shard,
            start_pos=start_pos_shard,
            s_prior_offset=sp_offset,
            block_len=block_len,
            strided_mm1=False,
            active_mask=active_mask,
            sbm=sbm,
            is_batch_sharded=batch_sharded,
            batch_offset=0,
            transposed_out=True,
        )

        # Store into this shard's grp (partition) and global s_prior (free) of the HBM output.
        hbm_sp_offset = hbm_shard_base + sp_offset
        dst_view = (
            mask_out_result.slice(dim=0, start=grp_base, end=grp_base + n_bsq_tiles_shard)
            .slice(dim=2, start=hbm_sp_offset, end=hbm_sp_offset + cur_sprior)
            .permute([1, 0, 2])
        )
        nisa.dma_copy(dst=dst_view, src=mask_out_sbuf)

        sbm.close_scope()

    sbm.close_scope()

    return mask_out_result.reshape((bs, q_head, s_active, s_prior))
