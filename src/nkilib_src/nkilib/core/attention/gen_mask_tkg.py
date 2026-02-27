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
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa import dge_mode

from ..utils.allocator import SbufManager
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.logging import get_logger
from ..utils.tensor_view import TensorView


def gen_mask_tkg(
    pos_ids: nl.ndarray,
    mask_out: nl.ndarray,
    bs: int,
    q_head: int,
    s_active: int,
    is_s_prior_sharded: bool,
    s_prior_per_shard: int,
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
    active_mask: Optional[nl.ndarray] = None,
    sbm: Optional[SbufManager] = None,
    is_batch_sharded: bool = False,
) -> nl.ndarray:
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
        pos_ids (nl.ndarray): [P_MAX, bs * s_active], Position IDs tensor in SBUF. P_MAX is broadcasted.
        mask_out (nl.ndarray): [P_MAX, n_sprior_tile, bs, q_head, s_active], Output mask buffer in SBUF.
        bs (int): Batch size.
        q_head (int): Number of query heads.
        s_active (int): Active sequence length.
        is_s_prior_sharded (bool): Whether s_prior dimension is sharded across LNCs.
        s_prior_per_shard (int): Total s_prior per shard (NC's full s_prior, used for NC offset calculation).
        s_prior_offset (int): Offset within current shard (for flash attention tiling). Default: 0.
        block_len (int): Block length for block KV cache (0 = flat cache). Default: 0.
        strided_mm1 (bool): Whether to use strided MM1 layout. Default: True.
        active_mask (Optional[nl.ndarray]): [s_active, bs_full, q_head, s_active], Optional active mask
            tensor in HBM. If provided, loaded onto the last section of the mask.
            When batch-sharded, each NC loads its portion [shard_id * bs, (shard_id + 1) * bs).
        sbm (Optional[SbufManager]): SBUF memory manager. If None, creates a new one.
        is_batch_sharded (bool): Whether batch dimension is sharded across LNCs. Default: False.
            When True: NC0 loads batches [0, bs), NC1 loads batches [bs, bs_full).
            When False: both NCs load batches [0, bs) where bs == bs_full.

    Returns:
        mask_out (nl.ndarray): [P_MAX, n_sprior_tile, bs, q_head, s_active], Generated mask tensor.

    Notes:
        - For block KV cache, indices are shuffled to match K cache block layout
        - Supports LNC sharding (lnc=1 and lnc=2 configurations)
        - When sprior-sharded with LNC=2, only shard 1 loads the active mask
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
        for batch_idx in range(bs):
            mask[batch_idx] = (iota < pos_ids[batch_idx])

        # Step 3: Optionally load active mask
        if active_mask is not None:
            load_active_mask_to_last_section(mask_out, active_mask)
    """
    # Hardware partition dim constraint
    P_MAX = nl.tile_size.pmax

    # Determine sharding configuration
    _, lnc, shard_id = get_verified_program_sharding_info("gen_mask_tkg", (0, 1))

    # Determine which dimension is sharded:
    # - When batch-sharded: both shards process full s_prior, so sprior_prg_id = 0
    # - When sprior-sharded: each shard processes different s_prior portion, so sprior_prg_id = shard_id
    # - When neither: sprior_prg_id = 0 (no sharding on s_prior dimension)
    sprior_prg_id = shard_id if is_s_prior_sharded else 0

    # Initialize SBUF manager if not provided
    if sbm is None:
        sbm = SbufManager(0, P_MAX * 128 * 4, get_logger("gen_mask_tkg"), use_auto_alloc=True)

    # Open SBUF memory scope
    sbm.open_scope(name="gen_mask_tkg")

    # Initialize mask to zeros
    nisa.memset(mask_out, value=0)

    kernel_assert(
        len(mask_out.shape) == 5,
        "gen_mask_tkg expects a 5D tensor of shape (P_MAX, n_sprior_tile, bs, q_head, s_active). "
        f"Allocate or reshape to a 5D tensor. Got shape {mask_out.shape}",
    )

    # Extract dimensions from mask_out shape
    _, n_sprior_tile, _bs, _q_head, _s_active = mask_out.shape
    s_active_qh = q_head * s_active

    # Validate dimensions
    kernel_assert(_bs == bs, f"mask_out bs dimension {_bs} does not match provided bs {bs}")
    kernel_assert(_q_head == q_head, f"mask_out q_head dimension {_q_head} does not match provided q_head {q_head}")
    kernel_assert(
        _s_active == s_active, f"mask_out s_active dimension {_s_active} does not match provided s_active {s_active}"
    )

    # Create a tensor with row indices and initialize to zero
    tmp_iota = sbm.alloc_stack(
        (P_MAX, n_sprior_tile), dtype=pos_ids.dtype, buffer=nl.sbuf, name=f"tmp_iota_{s_prior_offset}"
    )
    nisa.memset(tmp_iota, value=0)

    # Step 1: Generate index tensor based on cache layout
    _generate_iota_tensor(
        tmp_iota=tmp_iota,
        n_sprior_tile=n_sprior_tile,
        s_prior_per_shard=s_prior_per_shard,
        sprior_prg_id=sprior_prg_id,
        s_prior_offset=s_prior_offset,
        block_len=block_len,
        strided_mm1=strided_mm1,
    )

    # Repeat mask_iota s_active_qh times for each element
    mask_iota = sbm.alloc_stack(
        (P_MAX, n_sprior_tile * s_active_qh), dtype=tmp_iota.dtype, buffer=nl.sbuf, name=f"mask_iota_{s_prior_offset}"
    )
    nisa.tensor_copy(
        dst=mask_iota.ap(
            pattern=[
                [n_sprior_tile * s_active_qh, P_MAX],
                [s_active_qh, n_sprior_tile],
                [1, s_active_qh],
            ]
        ),
        src=tmp_iota.ap(
            pattern=[[n_sprior_tile, P_MAX], [1, n_sprior_tile], [0, s_active_qh]],
            offset=0,
        ),
        engine=nisa.scalar_engine,
    )

    # Step 2: Create prior masks by per-batch comparison
    _create_batch_masks(
        mask_iota=mask_iota,
        mask_out=mask_out,
        pos_ids=pos_ids,
        bs=bs,
        q_head=q_head,
        s_active=s_active,
        n_sprior_tile=n_sprior_tile,
        s_prior_offset=s_prior_offset,
        sbm=sbm,
    )

    # Step 3: Optionally load active mask onto the last section of mask_out
    if active_mask is not None:
        _load_active_mask(
            mask_out=mask_out,
            active_mask=active_mask,
            bs=bs,
            q_head=q_head,
            s_active=s_active,
            n_sprior_tile=n_sprior_tile,
            block_len=block_len,
            strided_mm1=strided_mm1,
            lnc=lnc,
            shard_id=shard_id,
            is_sprior_sharded=is_s_prior_sharded,
            is_batch_sharded=is_batch_sharded,
            s_prior_offset=s_prior_offset,
            s_prior_per_shard=s_prior_per_shard,
        )

    # Close SBUF memory scope
    sbm.close_scope()

    return mask_out


# ============================================================================
# Helper Functions
# ============================================================================


def _generate_iota_tensor(
    tmp_iota: nl.ndarray,
    n_sprior_tile: int,
    s_prior_per_shard: int,
    sprior_prg_id: int,
    s_prior_offset: int,
    block_len: int,
    strided_mm1: bool,
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
    """
    P_MAX = nl.tile_size.pmax
    iota_base = sprior_prg_id * s_prior_per_shard + s_prior_offset

    if block_len > 0:
        """
        Block KV: generate shuffled indices to match K cache block layout.
        
        The golden does .swapaxes(-1, -2) on (P_MAX, block_len) dims.
        After swapaxes, linear index i = fold * block_len * P_MAX + f * P_MAX + p
        maps to original token position = fold * P_MAX * block_len + p * block_len + f.
        
        So kernel needs: iota[p, f] = fold_base + p * block_len + f
        
        Using iota pattern=[[1, block_len]] with channel_multiplier=block_len:
            For partition p, free dim f: value = offset + f * 1 + p * block_len
        This gives: fold_base + p * block_len + f (correct!)
        """
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
        # Flat KV cache
        iota_pattern = [[1, n_sprior_tile]] if strided_mm1 else [[P_MAX, n_sprior_tile]]
        iota_multiplier = n_sprior_tile if strided_mm1 else 1
        nisa.iota(
            dst=tmp_iota[...],
            pattern=iota_pattern,
            offset=iota_base,
            channel_multiplier=iota_multiplier,
        )


def _create_batch_masks(
    mask_iota: nl.ndarray,
    mask_out: nl.ndarray,
    pos_ids: nl.ndarray,
    bs: int,
    q_head: int,
    s_active: int,
    n_sprior_tile: int,
    s_prior_offset: int,
    sbm: SbufManager,
) -> None:
    """
    Create prior masks by per-batch comparison with position IDs.

    For each batch, generates a mask by comparing the index tensor (mask_iota)
    against the corresponding position ID. The mask is then copied to the
    output tensor with proper batch interleaving.

    Args:
        mask_iota: Index tensor for comparison. Shape [P_MAX, n_sprior_tile * q_head * s_active].
        mask_out: Output mask buffer. Shape [P_MAX, n_sprior_tile, bs, q_head, s_active].
        pos_ids: Position IDs tensor. Shape [P_MAX, bs * s_active].
        bs: Batch size.
        q_head: Number of query heads.
        s_active: Active sequence length.
        n_sprior_tile: Number of s_prior tiles.
        s_prior_offset (int): Offset within current shard (for flash attention tiling, used here for tensor naming). Default: 0.
        sbm: SBUF memory manager.
    """
    P_MAX = nl.tile_size.pmax
    s_active_qh = q_head * s_active

    for batch_idx in range(bs):
        cur_mask = sbm.alloc_stack(
            mask_iota.shape, dtype=mask_out.dtype, buffer=nl.sbuf, name=f"cur_mask_{batch_idx}_{s_prior_offset}"
        )
        nisa.tensor_scalar(
            dst=cur_mask[...],
            data=mask_iota[...],
            op0=nl.less,
            operand0=pos_ids[:, nl.ds(batch_idx * s_active, 1)],
        )

        # Copy mask for this batch to mask_out, where batch dim is interleaved on fdim
        cur_mask = cur_mask.reshape((P_MAX, n_sprior_tile, q_head, s_active))
        mask_out_pat = mask_out.ap(
            [
                [n_sprior_tile * bs * q_head * s_active, P_MAX],
                [bs * q_head * s_active, n_sprior_tile],
                [1, q_head * s_active],
            ],
            offset=batch_idx * q_head * s_active,
        )

        # Alternate between scalar and vector engines for better performance
        if batch_idx % 2 == 0:
            nisa.tensor_copy(mask_out_pat, cur_mask[...], engine=nisa.scalar_engine)
        else:
            nisa.tensor_copy(mask_out_pat, cur_mask[...], engine=nisa.vector_engine)


def _load_active_mask(
    mask_out: nl.ndarray,
    active_mask: nl.ndarray,
    bs: int,
    q_head: int,
    s_active: int,
    n_sprior_tile: int,
    block_len: int,
    strided_mm1: bool,
    lnc: int,
    shard_id: int,
    is_sprior_sharded: bool,
    is_batch_sharded: bool,
    s_prior_offset: int = 0,
    s_prior_per_shard: int = 0,
) -> None:
    """
    Load active mask onto the last section of mask_out.

    Handles three cases:
    1. Block KV: Load active mask with shuffled block layout.
    2. Strided MM1: Load active mask in strided manner across partitions.
    3. Non-strided: Load to bottom right chunk of mask_out.

    Batch sharding logic:
    - Batch-sharded (is_batch_sharded=True): NC0 loads [0, bs), NC1 loads [bs, bs_full)
    - Sprior-sharded (is_batch_sharded=False): Both NCs load [0, bs) where bs == bs_full

    For LNC=2 with sprior-sharding:
    - Shard 0 processes s_prior positions [0, s_prior/2) - no active mask
    - Shard 1 processes s_prior positions [s_prior/2, s_prior) - has active mask
    Only shard 1 should load the active mask when sprior-sharded.

    Args:
        mask_out: Output mask buffer. Shape [P_MAX, n_sprior_tile, bs, q_head, s_active].
        active_mask: Active mask tensor in HBM. Shape [s_active, bs_full, q_head, s_active].
        bs: Batch size per NC.
        q_head: Number of query heads.
        s_active: Active sequence length.
        n_sprior_tile: Number of s_prior tiles.
        block_len: Block length for block KV cache (0 = flat cache).
        strided_mm1: Whether to use strided MM1 layout.
        is_sprior_sharded: Whether sharding is on s_prior dimension.
        is_batch_sharded: Whether sharding is on batch dimension.
        shard_id: Current shard ID (0 or 1 for LNC=2).
        lnc: Number of LNC shards (1 or 2).
        s_prior_offset: Offset within current shard (for FA tiling). Default: 0.
        s_prior_per_shard: Total s_prior per shard. Default: 0.
    """
    P_MAX = nl.tile_size.pmax

    # Compute batch start index based on sharding type:
    # - Batch-sharded: NC0 loads [0, bs), NC1 loads [bs, bs_full)
    # - Sprior-sharded: Both NCs load [0, bs) where bs == bs_full
    batch_start = shard_id * bs if is_batch_sharded else 0
    s_active_bqh = bs * q_head * s_active

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
        # Strided MM1: load active mask in strided manner
        # active_mask shape: [s_active, bs_full, q_head, s_active]
        # When batch-sharded, we need to offset into the correct batch portion
        # bs_full = bs * lnc for batch-sharded, bs_full = bs for sprior-sharded
        # batch_start gives us the starting batch index for this NC

        load1_nrows = s_active % n_sprior_tile
        load2_nrows = s_active - load1_nrows

        # Compute the source offset to account for batch sharding
        # active_mask is [s_active, bs_full, q_head, s_active]
        # We need to skip batch_start * q_head * s_active elements in the flattened view
        batch_offset = batch_start * q_head * s_active

        # The stride between rows in active_mask is bs_full * q_head * s_active,
        # where bs_full = bs * lnc for batch-sharded, bs for sprior-sharded/lnc=1.
        # This differs from n_sprior_tile * s_active_bqh when batch-sharded and
        # n_sprior_tile != lnc. We must use the actual source stride for the
        # active_mask DMA patterns, while the destination uses the mask_out stride.
        bs_full = bs * lnc if is_batch_sharded else bs
        s_active_bqh_full = bs_full * q_head * s_active  # stride between active_mask rows

        # Load first portion of active mask onto one partition
        if load1_nrows > 0:
            load1_pidx = P_MAX - (load2_nrows // n_sprior_tile) - 1

            dst_offset = load1_pidx * (n_sprior_tile * s_active_bqh) + (n_sprior_tile - load1_nrows) * s_active_bqh
            nisa.dma_copy(
                dst=mask_out.ap(
                    pattern=[
                        [n_sprior_tile * s_active_bqh, 1],
                        [s_active_bqh, load1_nrows],
                        [1, s_active_bqh],
                    ],
                    offset=dst_offset,
                ),
                src=active_mask.ap(
                    pattern=[
                        [n_sprior_tile * s_active_bqh_full, 1],
                        [s_active_bqh_full, load1_nrows],
                        [1, s_active_bqh],
                    ],
                    offset=batch_offset,
                ),
                dge_mode=dge_mode.none,
                name="active_mask_strided_load_partial",
            )

        # Load remaining active mask onto the last few partitions
        if load2_nrows > 0:
            load2_pidx = P_MAX - (load2_nrows // n_sprior_tile)

            dst_offset = load2_pidx * s_active_bqh * n_sprior_tile
            src_offset = batch_offset + load1_nrows * s_active_bqh_full
            nisa.dma_copy(
                mask_out.ap(
                    pattern=[
                        [
                            n_sprior_tile * s_active_bqh,
                            load2_nrows // n_sprior_tile,
                        ],
                        [s_active_bqh, n_sprior_tile],
                        [1, s_active_bqh],
                    ],
                    offset=dst_offset,
                ),
                active_mask.ap(
                    pattern=[
                        [
                            n_sprior_tile * s_active_bqh_full,
                            load2_nrows // n_sprior_tile,
                        ],
                        [s_active_bqh_full, n_sprior_tile],
                        [1, s_active_bqh],
                    ],
                    offset=src_offset,
                ),
                dge_mode=dge_mode.none,
                name="active_mask_strided_load_remaining",
            )
    else:
        # Non-strided: load to bottom right chunk of size [s_active, 1, bs, q_head, s_active]
        # active_mask shape: [s_active, bs_full, q_head, s_active]
        # Use TensorView to slice batch portion and expand dim for n_sprior_tile
        active_mask_view = (
            TensorView(active_mask)
            .slice(1, batch_start, batch_start + bs)  # [s_active, bs, q_head, s_active]
            .expand_dim(1)  # [s_active, 1, bs, q_head, s_active]
            .get_view()
        )
        nisa.dma_copy(
            mask_out[P_MAX - s_active :, n_sprior_tile - 1 : n_sprior_tile, :, :, :],
            active_mask_view,
            dge_mode=dge_mode.none,
            name="active_mask_sequential",
        )


def _load_active_mask_block_kv(
    mask_out: nl.ndarray,
    active_mask: nl.ndarray,
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

    # Only load if this tile contains any active positions
    if tile_end <= active_start_linear:
        return

    # Load each active position that falls within this tile.
    # Note: Minimal overhead since s_active is typically 1-8 for TKG.
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

        # Copy active_mask[k_idx, batch_start:batch_start+bs, :, :] to mask_out[p, f, :, :, :]
        # Use TensorView to slice and reshape the HBM tensor
        active_mask_view = (
            TensorView(active_mask)  # [s_active, bs_full, q_head, s_active]
            .select(0, k_idx)  # [bs_full, q_head, s_active]
            .slice(0, batch_start, batch_start + bs)  # [bs, q_head, s_active]
            .expand_dim(0)  # [1, bs, q_head, s_active]
            .expand_dim(0)  # [1, 1, bs, q_head, s_active]
            .get_view()
        )
        nisa.dma_copy(
            mask_out[p : p + 1, f : f + 1, :, :, :],
            active_mask_view,
            dge_mode=dge_mode.none,
            name=f"active_mask_block_kv_{k_idx}",
        )
