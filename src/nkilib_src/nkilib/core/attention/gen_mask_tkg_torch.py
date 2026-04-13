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
PyTorch reference for gen_mask_tkg kernel.

Generates attention masks for TKG attention with support for flat/block KV cache,
strided/non-strided layouts, LNC sharding, and active mask loading.

Usage:
    gen_mask_tkg_torch_ref.shard_id = 0
    gen_mask_tkg_torch_ref[lnc](pos_ids=..., mask_out=..., ...)
"""

from typing import Optional, Protocol

import torch

from ..utils.lnc_subscriptable import LncSubscriptable
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

P_MAX = 128


gen_mask_tkg_torch_ref: "LncSubscriptable[_GenMaskTkgTorchRefFn]"
"""
PyTorch reference for NKI kernel gen_mask_tkg.

Usage:
    gen_mask_tkg_torch_ref.shard_id = 0
    gen_mask_tkg_torch_ref[lnc](pos_ids=..., mask_out=..., ...)

Args:
    pos_ids: [P_MAX, bs * s_active] position IDs. P_MAX dim is broadcasted.
    mask_out: [P_MAX, n_sprior_tile, bs, q_head, s_active] output mask buffer.
    bs: Batch size.
    q_head: Number of query heads.
    s_active: Active sequence length.
    is_s_prior_sharded: Whether s_prior is sharded across LNCs.
    s_prior_per_shard: Total s_prior per shard.
    start_pos: Optional [P_MAX, bs * s_active] SWA window start (inclusive).
    s_prior_offset: Offset within shard (for FA tiling). Default: 0.
    block_len: Block length for block KV cache (0 = flat). Default: 0.
    strided_mm1: Whether to use strided MM1 layout. Default: True.
    active_mask: Optional [s_active, bs_full, q_head, s_active] active mask.
    is_batch_sharded: Whether batch is sharded across LNCs. Default: False.
    batch_offset: Shard-local batch offset into active_mask. Default: 0.

Returns:
    mask_out with generated mask.
"""


def _gen_mask_tkg_torch_ref_impl(
    pos_ids: torch.Tensor,
    mask_out: torch.Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    is_s_prior_sharded: bool,
    s_prior_per_shard: int,
    start_pos: Optional[torch.Tensor] = None,
    s_prior_offset: int = 0,
    block_len: int = 0,
    strided_mm1: bool = True,
    active_mask: Optional[torch.Tensor] = None,
    is_batch_sharded: bool = False,
    batch_offset: int = 0,
) -> torch.Tensor:
    """Generate mask matching gen_mask_tkg kernel output format.

    Unpacks kernel inputs, calls build_attention_mask (standard) or
    build_swa_attention_mask (SWA), then reshapes to the kernel's tiled layout.
    """
    LNC = gen_mask_tkg_torch_ref.lnc
    shard_id = gen_mask_tkg_torch_ref.shard_id

    if shard_id < 0 or shard_id >= LNC:
        raise ValueError(f"shard_id {shard_id} must be in range [0, {LNC})")

    _, n_sprior_tile, _bs, _q_head, _s_active = mask_out.shape
    mask_out.zero_()

    # Determine the batch slice this shard is responsible for
    # batch_offset is shard-local, added on top of NC sharding offset (like s_prior_offset)
    batch_start = (shard_id * bs if is_batch_sharded else 0) + batch_offset

    # Active mask is placed at the tail of each shard's s_prior_per_shard
    active_standard = None
    if active_mask is not None:
        # [s_active, bs_full, q_head, s_active] → [bs, q_head, s_active, s_active]
        active_standard = active_mask[:, batch_start : batch_start + bs, :, :].permute(1, 2, 3, 0).float()

    # When s_prior is sharded, each shard covers a different global range
    shard_offset = shard_id * s_prior_per_shard if is_s_prior_sharded else 0

    if start_pos is not None:
        # SWA path: per-query windowed mask
        # Extract per-query start and end values from kernel's packed format
        start_vals = start_pos[0, : bs * s_active].to(torch.float32)
        end_vals = pos_ids[0, : bs * s_active].to(torch.float32)
        mask = build_swa_attention_mask(
            start_vals=start_vals,
            end_vals=end_vals,
            batch=bs,
            num_heads=q_head,
            s_active=s_active,
            s_ctx=s_prior_per_shard,
            active_mask=active_standard,
            s_prior_start_offset=shard_offset,
        )
    else:
        # Standard path: all positions < cache_len are valid
        cache_lens = pos_ids[0, ::s_active][:bs].to(torch.float32)
        mask = build_attention_mask(
            cache_lens=cache_lens,
            batch=bs,
            num_heads=q_head,
            s_active=s_active,
            s_ctx=s_prior_per_shard,
            active_mask=active_standard,
            s_prior_start_offset=shard_offset,
        )
    # mask shape: [bs, q_head, s_active, s_prior_per_shard]

    # Slice the FA tile window from the shard's s_prior
    tile_size = n_sprior_tile * P_MAX
    tile_mask = mask[:, :, :, s_prior_offset : s_prior_offset + tile_size]
    # tile_mask shape: [bs, q_head, s_active, tile_size]

    # Reshape tile_size -> [P_MAX, n_sprior_tile] based on layout
    mask_out.copy_(_reshape_to_kernel_format(tile_mask, bs, q_head, s_active, n_sprior_tile, block_len, strided_mm1))

    return mask_out


def build_attention_mask(
    cache_lens: torch.Tensor,
    batch: int,
    num_heads: int,
    s_active: int,
    s_ctx: int,
    active_mask: Optional[torch.Tensor] = None,
    s_prior_start_offset: int = 0,
) -> torch.Tensor:
    """Core attention mask: prior mask + optional active mask placement.

    Args:
        cache_lens: [batch] cache lengths per batch element.
        batch: Batch size.
        num_heads: Number of attention heads.
        s_active: Active sequence length.
        s_ctx: Total context length (prior + active).
        active_mask: Optional [batch, num_heads, s_active, s_active] mask to place
            at the tail of s_ctx. If None, the last s_active positions are left as
            prior mask values.
        s_prior_start_offset: Global offset for this shard's k positions (default 0).

    Returns:
        mask [batch, num_heads, s_active, s_ctx].
    """
    if cache_lens.dim() == 2:
        cache_lens = cache_lens.squeeze(-1)
    cache_lens = cache_lens.to(torch.float32)

    # Prior mask: mask[b, :, :, k] = 1 if (k + offset) < cache_len[b]
    k_indices = torch.arange(s_ctx, dtype=torch.float32).view(1, 1, 1, s_ctx) + s_prior_start_offset
    mask = (k_indices < cache_lens.view(batch, 1, 1, 1)).float().expand(batch, num_heads, s_active, s_ctx).clone()

    # Place active mask at the tail of s_ctx
    if active_mask is not None:
        mask[:, :, :, s_ctx - s_active :] = active_mask

    return mask


def build_swa_attention_mask(
    start_vals: torch.Tensor,
    end_vals: torch.Tensor,
    batch: int,
    num_heads: int,
    s_active: int,
    s_ctx: int,
    active_mask: Optional[torch.Tensor] = None,
    s_prior_start_offset: int = 0,
) -> torch.Tensor:
    """Per-query SWA mask: each query has its own [start, end) window.

    Args:
        start_vals: [bs * s_active] per-query SWA window start (inclusive).
        end_vals: [bs * s_active] per-query end positions (exclusive).
        batch: Batch size.
        num_heads: Number of attention heads.
        s_active: Active sequence length.
        s_ctx: Total context length (prior + active).
        active_mask: Optional [batch, num_heads, s_active, s_active] mask to place
            at the tail of s_ctx.
        s_prior_start_offset: Global offset for this shard's k positions (default 0).

    Returns:
        mask [batch, num_heads, s_active, s_ctx].
    """
    # k_indices: [1, 1, s_ctx] global position of each cache slot
    k_indices = torch.arange(s_ctx, dtype=torch.float32).view(1, 1, s_ctx) + s_prior_start_offset

    # Reshape start/end from [bs * s_active] to [batch, s_active, 1] for broadcasting
    starts = start_vals.view(batch, s_active, 1).float()
    ends = end_vals.view(batch, s_active, 1).float()

    # Vectorized mask: normal (start <= end) uses AND, wrap-around uses OR
    normal = starts <= ends
    mask_normal = (k_indices >= starts) & (k_indices < ends)
    mask_wrap = (k_indices >= starts) | (k_indices < ends)
    mask = torch.where(normal, mask_normal, mask_wrap).float()

    # Broadcast across heads: [batch, s_active, s_ctx] -> [batch, num_heads, s_active, s_ctx]
    mask = mask.unsqueeze(1).expand(batch, num_heads, s_active, s_ctx)

    # Place active mask at the tail of s_ctx
    if active_mask is not None:
        mask = mask.clone()
        mask[:, :, :, s_ctx - s_active :] = active_mask

    return mask


def build_full_attention_mask(
    cache_lens: torch.Tensor,
    batch: int,
    num_heads: int,
    s_active: int,
    s_ctx: int,
    lnc: int = 1,
    block_len: int = 0,
    include_active_mask: bool = False,
    transposed: bool = False,
    enable_fa_s_prior_tiling: bool = True,
    KVDP: int = 1,
) -> torch.Tensor:
    """Convenience wrapper: generates causal active mask, calls build_attention_mask,
    then applies block KV reshaping and transpose.

    Args:
        enable_fa_s_prior_tiling: Whether flash attention tiling is enabled. Must match
            the value passed to attention_tkg / attention_block_tkg. Default: True.
        KVDP: KV data parallelism factor. When > 1, the kernel runs per-rank with
            batch // KVDP, so sharding decisions use that reduced batch size. Default: 1.

    Returns [batch, num_heads, s_active, s_ctx] or transposed [s_ctx, batch, num_heads, s_active].
    """
    kernel_bs = batch // KVDP

    # Optionally generate causal active mask (lower triangular)
    active_mask = None
    if include_active_mask:
        active_mask = torch.tril(torch.ones(s_active, s_active, dtype=torch.float32))

    mask = build_attention_mask(
        cache_lens=cache_lens,
        batch=batch,
        num_heads=num_heads,
        s_active=s_active,
        s_ctx=s_ctx,
        active_mask=active_mask,
    )

    # Block KV reshaping: reorder to [num_folds, block_len, P_MAX] so that
    # the flat s_ctx dimension is in [n_sprior_tile, P_MAX] row-major order
    # (n_sprior_tile-major), matching the layout produced by gen_mask_tkg_hbm.
    if block_len > 0 and mask.numel() > 0:
        sprior_n_prgs = lnc if lnc > 1 and is_s_prior_sharded_fn(kernel_bs, num_heads, s_active, s_ctx, P_MAX) else 1
        reduced_block_len, _ = resize_cache_block_len_for_attention_tkg_kernel(
            s_ctx // block_len,
            block_len,
            lnc,
            P_MAX,
            kernel_bs,
            num_heads,
            s_active,
            enable_fa_s_prior_tiling=enable_fa_s_prior_tiling,
        )
        mask = mask.reshape(batch, num_heads, s_active, sprior_n_prgs, -1, P_MAX, reduced_block_len)
        # Permute to (..., num_folds, block_len, P_MAX) then flatten
        mask = mask.permute(0, 1, 2, 3, 4, 6, 5)
        mask = mask.reshape(batch, num_heads, s_active, s_ctx)

    if transposed:
        mask = mask.permute(3, 0, 1, 2)

    return mask


def _reshape_to_kernel_format(
    mask: torch.Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    n_sprior_tile: int,
    block_len: int,
    strided_mm1: bool,
) -> torch.Tensor:
    """Reshape mask from [bs, q_head, s_active, tile_size] to kernel format [P_MAX, n_sprior_tile, bs, q_head, s_active].

    The mapping from linear position k in the tile to (p, f) coordinates:
    - Strided MM1:    k -> (p=k // n_sprior_tile, f=k % n_sprior_tile)
    - Non-strided:    k -> (p=k % P_MAX,          f=k // P_MAX)
    - Block KV:       k -> fold/block shuffle matching K cache block layout
    """
    if block_len > 0:
        num_folds = n_sprior_tile // block_len
        # Linear positions are organized as: fold * (block_len * P_MAX) + p * block_len + blk_offset
        # Reshape to [bs, q_head, s_active, num_folds, P_MAX, block_len]
        result = mask.reshape(bs, q_head, s_active, num_folds, P_MAX, block_len)
        # Reorder to [P_MAX, num_folds, block_len, bs, q_head, s_active]
        result = result.permute(4, 3, 5, 0, 1, 2)
        # Merge num_folds and block_len -> n_sprior_tile
        result = result.reshape(P_MAX, n_sprior_tile, bs, q_head, s_active)
    elif strided_mm1:
        # Linear pos k -> (p=k // n_sprior_tile, f=k % n_sprior_tile)
        # Reshape tile_size -> [P_MAX, n_sprior_tile]
        result = mask.reshape(bs, q_head, s_active, P_MAX, n_sprior_tile)
        result = result.permute(3, 4, 0, 1, 2)  # [P_MAX, n_sprior_tile, bs, q_head, s_active]
    else:
        # Linear pos k -> (p=k % P_MAX, f=k // P_MAX)
        # Reshape tile_size -> [n_sprior_tile, P_MAX]
        result = mask.reshape(bs, q_head, s_active, n_sprior_tile, P_MAX)
        result = result.permute(4, 3, 0, 1, 2)  # [P_MAX, n_sprior_tile, bs, q_head, s_active]

    return result.contiguous()


class _GenMaskTkgTorchRefFn(Protocol):
    def __call__(
        self,
        pos_ids: torch.Tensor,
        mask_out: torch.Tensor,
        bs: int,
        q_head: int,
        s_active: int,
        is_s_prior_sharded: bool,
        s_prior_per_shard: int,
        start_pos: Optional[torch.Tensor] = None,
        s_prior_offset: int = 0,
        block_len: int = 0,
        strided_mm1: bool = True,
        active_mask: Optional[torch.Tensor] = None,
        is_batch_sharded: bool = False,
        batch_offset: int = 0,
    ) -> torch.Tensor:
        """
        PyTorch reference for NKI kernel gen_mask_tkg.

        Usage:
            gen_mask_tkg_torch_ref.shard_id = 0
            gen_mask_tkg_torch_ref[lnc](pos_ids=..., mask_out=..., ...)

        Args:
            pos_ids: [P_MAX, bs * s_active] position IDs. P_MAX dim is broadcasted.
            mask_out: [P_MAX, n_sprior_tile, bs, q_head, s_active] output mask buffer.
            bs: Batch size.
            q_head: Number of query heads.
            s_active: Active sequence length.
            is_s_prior_sharded: Whether s_prior is sharded across LNCs.
            s_prior_per_shard: Total s_prior per shard.
            start_pos: Optional [P_MAX, bs * s_active] SWA window start (inclusive).
            s_prior_offset: Offset within shard (for FA tiling). Default: 0.
            block_len: Block length for block KV cache (0 = flat). Default: 0.
            strided_mm1: Whether to use strided MM1 layout. Default: True.
            active_mask: Optional [s_active, bs_full, q_head, s_active] active mask.
            is_batch_sharded: Whether batch is sharded across LNCs. Default: False.
            batch_offset: Shard-local batch offset into active_mask. Default: 0.

        Returns:
            mask_out with generated mask.
        """
        ...


gen_mask_tkg_torch_ref = LncSubscriptable(_gen_mask_tkg_torch_ref_impl, _GenMaskTkgTorchRefFn)


# ---------------------------------------------------------------------------
# HBM wrapper torch reference
# ---------------------------------------------------------------------------

gen_mask_tkg_hbm_torch_ref: "LncSubscriptable[_GenMaskTkgHbmTorchRefFn]"
"""
PyTorch reference for NKI kernel gen_mask_tkg_hbm.

Usage:
    gen_mask_tkg_hbm_torch_ref[lnc](pos_ids_hbm=..., bs=..., ...)

Args:
    pos_ids_hbm: [1, bs * s_active] position IDs.
    bs: Batch size.
    q_head: Number of query heads.
    s_active: Active sequence length.
    s_prior: Total prior sequence length (must be divisible by P_MAX).
    start_pos_hbm: Optional [1, bs * s_active] SWA start positions.
    block_len: Block length for block KV cache (0 = flat). Default: 0.
    active_mask: Optional [s_active, bs, q_head, s_active] active mask.

Returns:
    [s_prior, bs, q_head, s_active] generated mask.
"""


def _gen_mask_tkg_hbm_torch_ref_impl(
    pos_ids_hbm: torch.Tensor,
    bs: int,
    q_head: int,
    s_active: int,
    s_prior: int,
    start_pos_hbm: Optional[torch.Tensor] = None,
    block_len: int = 0,
    active_mask: Optional[torch.Tensor] = None,
    enable_fa_s_prior_tiling: bool = True,
) -> torch.Tensor:
    """Generate mask matching gen_mask_tkg_hbm kernel output format.

    Builds the full mask over all s_prior positions, reshapes to the kernel's
    5D tiled layout, then handles batch-sharded LNC by zeroing non-owned
    batch slices.
    """
    lnc = gen_mask_tkg_hbm_torch_ref.lnc

    strided_mm1 = block_len == 0

    # LNC sharding decision (mirrors gen_mask_tkg_hbm kernel logic)
    cfg = AttnTKGConfig(bs=bs, q_head=q_head, s_active=s_active, curr_sprior=s_prior)

    if lnc == 2 and is_s_prior_sharded_fn(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, P_MAX):
        batch_sharded = False
    elif lnc == 2 and is_batch_sharded_fn(cfg.bs, cfg.q_head, cfg.s_active, cfg.curr_sprior, P_MAX):
        batch_sharded = True
    else:
        batch_sharded = False

    # Adjust block_len for hardware constraints
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
        )

    n_sprior_tile = s_prior // P_MAX

    # Build full mask over all s_prior positions per batch element.
    # Active mask placement at the tail of s_prior is handled by
    # build_attention_mask / build_swa_attention_mask.
    active_standard = None
    if active_mask is not None:
        # [s_active, bs, q_head, s_active] → [bs, q_head, s_active, s_active]
        active_standard = active_mask[:, :bs, :, :].permute(1, 2, 3, 0).float()

    if start_pos_hbm is not None:
        start_vals = start_pos_hbm[0, : bs * s_active].to(torch.float32)
        end_vals = pos_ids_hbm[0, : bs * s_active].to(torch.float32)
        full_mask = build_swa_attention_mask(
            start_vals=start_vals,
            end_vals=end_vals,
            batch=bs,
            num_heads=q_head,
            s_active=s_active,
            s_ctx=s_prior,
            active_mask=active_standard,
        )
    else:
        cache_lens = pos_ids_hbm[0, ::s_active][:bs].to(torch.float32)
        full_mask = build_attention_mask(
            cache_lens=cache_lens,
            batch=bs,
            num_heads=q_head,
            s_active=s_active,
            s_ctx=s_prior,
            active_mask=active_standard,
        )
    # full_mask: [bs, q_head, s_active, s_prior]

    # Reshape to 5D tiled kernel format using the global tile count
    full_mask_5d = _reshape_to_kernel_format(full_mask, bs, q_head, s_active, n_sprior_tile, block_len, strided_mm1)
    # full_mask_5d: [P_MAX, n_sprior_tile, bs, q_head, s_active]

    # For batch-sharded LNC=2, each shard only writes its own batch slice;
    # the other slice stays zero. Combine both shards' contributions.
    if batch_sharded:
        bs_per_shard = bs // lnc
        combined = torch.zeros_like(full_mask_5d)
        for shard_id in range(lnc):
            b_start = shard_id * bs_per_shard
            b_end = b_start + bs_per_shard
            combined[:, :, b_start:b_end, :, :] = full_mask_5d[:, :, b_start:b_end, :, :]
        full_mask_5d = combined

    # Transpose to n_sprior_tile-major: [n_sprior_tile, P_MAX, bs, q_head, s_active]
    # The kernel stores in n_sprior_tile-major order (row width = P_MAX) so that
    # flat slicing at multiples of P_MAX always lands on row boundaries.
    full_mask_5d = full_mask_5d.permute(1, 0, 2, 3, 4).contiguous()

    # Flatten to HBM output format: [s_prior, bs, q_head, s_active]
    return full_mask_5d.reshape(s_prior, bs, q_head, s_active)


class _GenMaskTkgHbmTorchRefFn(Protocol):
    def __call__(
        self,
        pos_ids_hbm: torch.Tensor,
        bs: int,
        q_head: int,
        s_active: int,
        s_prior: int,
        start_pos_hbm: Optional[torch.Tensor] = None,
        block_len: int = 0,
        active_mask: Optional[torch.Tensor] = None,
        enable_fa_s_prior_tiling: bool = True,
    ) -> torch.Tensor:
        """
        PyTorch reference for NKI kernel gen_mask_tkg_hbm.

        Usage:
            gen_mask_tkg_hbm_torch_ref[lnc](pos_ids_hbm=..., bs=..., ...)

        Args:
            pos_ids_hbm: [1, bs * s_active] position IDs.
            bs: Batch size.
            q_head: Number of query heads.
            s_active: Active sequence length.
            s_prior: Total prior sequence length (must be divisible by P_MAX).
            start_pos_hbm: Optional [1, bs * s_active] SWA start positions.
            block_len: Block length for block KV cache (0 = flat). Default: 0.
            active_mask: Optional [s_active, bs, q_head, s_active] active mask.

        Returns:
            [s_prior, bs, q_head, s_active] generated mask.
        """
        ...


gen_mask_tkg_hbm_torch_ref = LncSubscriptable(_gen_mask_tkg_hbm_torch_ref_impl, _GenMaskTkgHbmTorchRefFn)
