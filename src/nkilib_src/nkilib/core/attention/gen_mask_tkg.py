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
from ..utils.logging import Logger
from .attention_tkg_utils import AttnTKGConfig, is_s_prior_sharded


def gen_mask_tkg(
    pos_ids: nl.ndarray,
    mask_out: nl.ndarray,
    bs: int,
    q_head: int,
    s_active: int,
    block_len: int = 0,
    strided_mm1: bool = True,
    active_mask: Optional[nl.ndarray] = None,
    sbm: Optional[SbufManager] = None,
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
        s_prior: Prior sequence length (derived from mask_out shape)
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
        block_len (int): Block length for block KV cache (0 = flat cache). Default: 0.
        strided_mm1 (bool): Whether to use strided MM1 layout. Default: True.
        active_mask (Optional[nl.ndarray]): [s_active, bs, q_head, s_active], Optional active mask
            tensor in HBM. If provided, loaded onto the last section of the mask.
        sbm (Optional[SbufManager]): SBUF memory manager. If None, creates a new one.

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

    # Extract s_prior from mask_out shape for sharding decision
    _, n_sprior_tile_for_sharding, _, _, _ = mask_out.shape
    s_prior_for_sharding = n_sprior_tile_for_sharding * P_MAX

    # Create minimal config to determine sharding mode
    cfg = AttnTKGConfig(bs=bs, q_head=q_head, s_active=s_active, curr_sprior=s_prior_for_sharding)

    # Determine which dimension is sharded:
    # - When batch-sharded: both shards process full s_prior, so sprior_prg_id = 0
    # - When sprior-sharded: each shard processes different s_prior portion, so sprior_prg_id = shard_id
    # - When neither: sprior_prg_id = 0 (no sharding on s_prior dimension)
    if is_s_prior_sharded(cfg, P_MAX):
        sprior_prg_id = shard_id
    else:
        # Either batch-sharded or no sharding at all
        sprior_prg_id = 0

    # Initialize SBUF manager if not provided
    if sbm is None:
        sbm = SbufManager(0, P_MAX * 128 * 4, Logger("gen_mask_tkg"), use_auto_alloc=True)

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
    s_prior = n_sprior_tile * P_MAX
    s_active_qh = q_head * s_active

    # Validate dimensions
    kernel_assert(_bs == bs, f"mask_out bs dimension {_bs} does not match provided bs {bs}")
    kernel_assert(_q_head == q_head, f"mask_out q_head dimension {_q_head} does not match provided q_head {q_head}")
    kernel_assert(
        _s_active == s_active, f"mask_out s_active dimension {_s_active} does not match provided s_active {s_active}"
    )

    # Create a tensor with row indices and initialize to zero
    tmp_iota = sbm.alloc_stack((P_MAX, n_sprior_tile), dtype=pos_ids.dtype, buffer=nl.sbuf, name="tmp_iota")
    nisa.memset(tmp_iota, value=0)

    # Step 1: Generate index tensor based on cache layout
    _generate_iota_tensor(
        tmp_iota=tmp_iota,
        n_sprior_tile=n_sprior_tile,
        s_prior=s_prior,
        sprior_prg_id=sprior_prg_id,
        block_len=block_len,
        strided_mm1=strided_mm1,
    )

    # Repeat mask_iota s_active_qh times for each element
    mask_iota = sbm.alloc_stack(
        (P_MAX, n_sprior_tile * s_active_qh), dtype=tmp_iota.dtype, buffer=nl.sbuf, name="mask_iota"
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
        sbm=sbm,
    )

    # Determine if we're sprior-sharded for active mask loading
    is_sprior_sharded_mode = is_s_prior_sharded(cfg, P_MAX)

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
            is_sprior_sharded=is_sprior_sharded_mode,
            shard_id=shard_id,
            lnc=lnc,
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
    s_prior: int,
    sprior_prg_id: int,
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
        s_prior: Prior sequence length for this shard.
        sprior_prg_id: Shard ID (0 or 1 for LNC=2).
        block_len: Block length for block KV cache (0 = flat cache).
        strided_mm1: Whether to use strided MM1 layout.
    """
    P_MAX = nl.tile_size.pmax
    iota_base = sprior_prg_id * s_prior

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
        sbm: SBUF memory manager.
    """
    P_MAX = nl.tile_size.pmax
    s_active_qh = q_head * s_active

    for batch_idx in range(bs):
        cur_mask = sbm.alloc_stack(mask_iota.shape, dtype=mask_out.dtype, buffer=nl.sbuf, name=f"cur_mask_{batch_idx}")
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
    is_sprior_sharded: bool,
    shard_id: int,
    lnc: int,
) -> None:
    """
    Load active mask onto the last section of mask_out.

    Handles three cases:
    1. Block KV: Active mask is already incorporated via pos_ids comparison.
    2. Strided MM1: Load active mask in strided manner across partitions.
    3. Non-strided: Load to bottom right chunk of mask_out.

    For LNC=2 with sprior-sharding:
    - Shard 0 processes s_prior positions [0, s_prior/2) - no active mask
    - Shard 1 processes s_prior positions [s_prior/2, s_prior) - has active mask
    Only shard 1 should load the active mask when sprior-sharded.

    For LNC=1: Always load active mask (no sharding).

    Args:
        mask_out: Output mask buffer. Shape [P_MAX, n_sprior_tile, bs, q_head, s_active].
        active_mask: Active mask tensor in HBM. Shape [s_active, bs, q_head, s_active].
        bs: Batch size.
        q_head: Number of query heads.
        s_active: Active sequence length.
        n_sprior_tile: Number of s_prior tiles.
        block_len: Block length for block KV cache (0 = flat cache).
        strided_mm1: Whether to use strided MM1 layout.
        is_sprior_sharded: Whether sharding is on s_prior dimension.
        shard_id: Current shard ID (0 or 1 for LNC=2).
        lnc: Number of LNC shards (1 or 2).
    """
    # For sprior-sharded mode with LNC > 1, only shard 1 should load the active mask
    # because the active mask corresponds to the last s_active positions
    # of the full s_prior sequence, which only shard 1 processes.
    # For LNC=1, always load the active mask since there's no sharding.
    if lnc > 1 and is_sprior_sharded and shard_id == 0:
        return
    P_MAX = nl.tile_size.pmax
    s_active_bqh = bs * q_head * s_active

    if block_len > 0:
        # Block KV active mask: already incorporated in prior mask via pos_ids
        # The causal relationship is encoded in pos_ids comparison
        pass
    elif strided_mm1:
        # Strided MM1: load active mask in strided manner
        load1_nrows = s_active % n_sprior_tile
        load2_nrows = s_active - load1_nrows

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
                        [n_sprior_tile * s_active_bqh, 1],
                        [s_active_bqh, load1_nrows],
                        [1, s_active_bqh],
                    ],
                    offset=0,
                ),
                dge_mode=dge_mode.none,
                name="active_mask_strided_load_partial",
            )

        # Load remaining active mask onto the last few partitions
        if load2_nrows > 0:
            load2_pidx = P_MAX - (load2_nrows // n_sprior_tile)

            dst_offset = load2_pidx * s_active_bqh * n_sprior_tile
            src_offset = load1_nrows * s_active_bqh
            nisa.dma_copy(
                mask_out.ap(
                    pattern=[
                        [
                            n_sprior_tile * s_active_bqh,
                            load2_nrows // n_sprior_tile,
                        ],
                        [1, n_sprior_tile * s_active_bqh],
                    ],
                    offset=dst_offset,
                ),
                active_mask.ap(
                    pattern=[
                        [
                            n_sprior_tile * s_active_bqh,
                            load2_nrows // n_sprior_tile,
                        ],
                        [1, n_sprior_tile * s_active_bqh],
                    ],
                    offset=src_offset,
                ),
                dge_mode=dge_mode.none,
                name="active_mask_strided_load_remaining",
            )
    else:
        # Non-strided: load to bottom right chunk of size [s_active, s_active_bqh]
        active_mask_reshaped = active_mask.reshape((s_active, 1, bs, q_head, s_active))
        nisa.dma_copy(
            mask_out[P_MAX - s_active :, n_sprior_tile - 1 : n_sprior_tile, :, :, :],
            active_mask_reshaped,
            dge_mode=dge_mode.none,
            name="active_mask_load_sequential",
        )
