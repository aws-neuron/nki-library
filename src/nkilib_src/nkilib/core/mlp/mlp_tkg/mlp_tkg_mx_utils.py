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

"""Utility functions for MLP TKG MX kernel — weight loading, bias loading, dummy scale init."""

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_helpers import div_ceil

# Dummy MX scale value: uint32 representation of 4 × uint8(127) = 0x7F7F7F7F
_DUMMY_SCALE_U32 = 2139062143


def alloc_dummy_scale(sbm, p_dim, d1, d2):
    """
    Allocate a dummy MX scale buffer via sbm, memset to all-127, and return a uint8 view.

    Args:
        sbm: SbufManager for scoped allocation.
        p_dim (int): Partition dimension (128).
        d1 (int): First free dimension of the uint32 buffer.
        d2 (int): Second free dimension of the uint32 buffer (the last dim is 4x larger
            in uint8 after reinterpret).

    Returns:
        uint8 view of the memset buffer.
    """
    u32_buf = sbm.alloc_stack((p_dim, d1, d2), dtype=nl.uint32, buffer=nl.sbuf, align=32)
    nisa.memset(dst=u32_buf, value=_DUMMY_SCALE_U32)
    return u32_buf.view(nl.uint8)


def alloc_dummy_scale_tile(sbm, _pmax=128):
    """
    Allocate a single dummy MX scale tile (all-127) for reuse across all matmul calls.

    Allocates a ``[_pmax, 32]`` uint32 buffer (equivalent to ``[_pmax, 128]`` uint8 when
    reinterpreted; 128 = 32 * 4 elements per uint32), memsets it to 0x7F7F7F7F, and
    returns a uint8 view.

    Args:
        sbm: SbufManager for scoped allocation.
        _pmax (int): Partition dimension (default 128).

    Returns:
        uint8 ``[_pmax, 128]`` view of the memset buffer.
    """
    u32_buf = sbm.alloc_stack((_pmax, 32), dtype=nl.uint32, buffer=nl.sbuf, align=32)
    nisa.memset(dst=u32_buf, value=_DUMMY_SCALE_U32)
    return u32_buf.view(nl.uint8)


def load_gate_up_weight(sbm, w_hbm, i_tile, n_H512_tile_sharded, num_shards, shard_id, _pmax):
    """
    Allocate SBUF and load one gate or up weight slice for an I-tile.

    HBM layout: ``[128, n_H512_tile, I]``. Loads the shard's H512 slice and I-tile range.

    Args:
        sbm: SbufManager for scoped allocation.
        w_hbm: Gate or up weight tensor in HBM with layout ``[128, n_H512_tile, I]``.
        i_tile: Tile descriptor providing ``start_offset``, ``end_offset``, and ``size``
            along the I dimension.
        n_H512_tile_sharded (int): Number of H512 tiles assigned to this shard.
        num_shards (int): Total number of shards.
        shard_id (int): Index of the current shard.
        _pmax (int): Partition dimension (128).

    Returns:
        SBUF tensor ``[_pmax, n_H512_tile_sharded, i_tile.size]``.
    """
    _q_width = 4
    is_scalar_fp8 = w_hbm.dtype in (nl.float8_e4m3, nl.float8_e4m3fn)

    if is_scalar_fp8:
        # Scalar fp8: HBM has I*4 elements on last dim. Load slice, then reinterpret to x4.
        load_size = i_tile.size * _q_width
        i_start = i_tile.start_offset * _q_width
        i_end = i_tile.end_offset * _q_width
        w_sb = sbm.alloc_stack((_pmax, n_H512_tile_sharded, load_size), dtype=w_hbm.dtype, buffer=nl.sbuf, align=32)
        if num_shards > 1:
            nisa.dma_copy(
                dst=w_sb,
                src=w_hbm[:, shard_id * n_H512_tile_sharded : (shard_id + 1) * n_H512_tile_sharded, i_start:i_end],
            )
        else:
            nisa.dma_copy(dst=w_sb, src=w_hbm[:, :, i_start:i_end])
        return w_sb.view(nl.float8_e4m3fn_x4)

    else:
        w_sb = sbm.alloc_stack((_pmax, n_H512_tile_sharded, i_tile.size), dtype=w_hbm.dtype, buffer=nl.sbuf, align=32)
        if num_shards > 1:
            nisa.dma_copy(
                dst=w_sb,
                src=w_hbm[
                    :,
                    shard_id * n_H512_tile_sharded : (shard_id + 1) * n_H512_tile_sharded,
                    i_tile.start_offset : i_tile.end_offset,
                ],
                dge_mode=nisa.dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(dst=w_sb, src=w_hbm[:, :, i_tile.start_offset : i_tile.end_offset])
        return w_sb


def load_down_weight(sbm, down_w_hbm, i_tile, H_sharded, num_shards, shard_id, original_I, _pmax, _q_width):
    """
    Allocate SBUF and load down weight slice for an I-tile, handling partial last I512 tile.

    HBM layout: ``[p_I, n_I512_total, H]``. Loads the I-tile's I512 range and shard's H slice.

    Args:
        sbm: SbufManager for scoped allocation.
        down_w_hbm: Down weight tensor in HBM with layout ``[p_I, n_I512_total, H]``.
        i_tile: Tile descriptor providing ``start_offset``, ``end_offset``, and ``size``
            along the I dimension.
        H_sharded (int): H slice size assigned to this shard.
        num_shards (int): Total number of shards.
        shard_id (int): Index of the current shard.
        original_I (int): Original (unsharded) I dimension, used to compute the partition
            size when I <= 512.
        _pmax (int): Partition dimension (128).
        _q_width (int): Quantization width (elements per quantization group on the free
            dimension).

    Returns:
        SBUF tensor ``[_pmax, n_I512_cur, H_sharded]``.
    """
    is_scalar_fp8 = down_w_hbm.dtype in (nl.float8_e4m3, nl.float8_e4m3fn)
    H_sharded_load = H_sharded * _q_width if is_scalar_fp8 else H_sharded

    n_I512_cur = div_ceil(i_tile.size, _pmax * _q_width)
    i512_start = i_tile.start_offset // (_pmax * _q_width)
    i512_end = i512_start + n_I512_cur
    down_w_sb = sbm.alloc_stack((_pmax, n_I512_cur, H_sharded_load), dtype=down_w_hbm.dtype, buffer=nl.sbuf, align=32)
    p_I_full = _pmax if original_I > 512 else original_I // _q_width
    last_i512_size = i_tile.size - (n_I512_cur - 1) * (_pmax * _q_width)
    p_I_last = _pmax if last_i512_size >= _pmax * _q_width else last_i512_size // _q_width

    if p_I_last != _pmax:
        nisa.memset(dst=down_w_sb[:, n_I512_cur - 1, :], value=0.0)
        if n_I512_cur > 1:
            if num_shards > 1:
                nisa.dma_copy(
                    src=down_w_hbm[
                        :p_I_full,
                        i512_start : i512_end - 1,
                        shard_id * H_sharded_load : (shard_id + 1) * H_sharded_load,
                    ],
                    dst=down_w_sb[:p_I_full, : n_I512_cur - 1, :],
                    dge_mode=nisa.dge_mode.hwdge,
                )
            else:
                nisa.dma_copy(
                    dst=down_w_sb[:p_I_full, : n_I512_cur - 1, :],
                    src=down_w_hbm[:p_I_full, i512_start : i512_end - 1, :],
                )
        if num_shards > 1:
            nisa.dma_copy(
                src=down_w_hbm[
                    :p_I_last, i512_end - 1 : i512_end, shard_id * H_sharded_load : (shard_id + 1) * H_sharded_load
                ],
                dst=down_w_sb[:p_I_last, n_I512_cur - 1 : n_I512_cur, :],
                dge_mode=nisa.dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(
                dst=down_w_sb[:p_I_last, n_I512_cur - 1 : n_I512_cur, :],
                src=down_w_hbm[:p_I_last, i512_end - 1 : i512_end, :],
            )
    else:
        if num_shards > 1:
            nisa.dma_copy(
                src=down_w_hbm[
                    :p_I_full, i512_start:i512_end, shard_id * H_sharded_load : (shard_id + 1) * H_sharded_load
                ],
                dst=down_w_sb[:p_I_full, :, :],
                dge_mode=nisa.dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(dst=down_w_sb[:p_I_full, :, :], src=down_w_hbm[:p_I_full, i512_start:i512_end, :])

    if is_scalar_fp8:
        return down_w_sb.view(nl.float8_e4m3fn_x4)

    return down_w_sb


def load_gate_up_bias_mx(bias_hbm, I, _pmax, n_I512_tile, _q_width):
    """
    Allocate SBUF buffer and load gate or up projection bias from HBM.

    Args:
        bias_hbm: Bias tensor in HBM, or None.
        I (int): Intermediate dimension.
        _pmax (int): Partition dimension (128).
        n_I512_tile (int): Number of I512 tiles on the free dimension.
        _q_width (int): Quantization width (elements per quantization group on the free
            dimension).

    Returns:
        bias_sb: ``[_pmax, n_I512_tile, _q_width]`` bf16 SBUF tensor, or None if
            ``bias_hbm`` is None.
    """
    if bias_hbm == None:
        return None
    bias_sb = nl.ndarray((_pmax, n_I512_tile, _q_width), dtype=nl.bfloat16, buffer=nl.sbuf)
    if I < 512:
        nisa.memset(dst=bias_sb[:, 0, :], value=0.0)
        nisa.dma_copy(dst=bias_sb[: I // _q_width, :, :], src=bias_hbm)
    else:
        nisa.dma_copy(dst=bias_sb, src=bias_hbm)
    return bias_sb


def load_down_bias_mx(bias_hbm, num_shards, H1_shard, H0, shard_id):
    """
    Allocate SBUF buffer and load down projection bias from HBM via ``dma_transpose``.

    Args:
        bias_hbm: Down bias tensor in HBM with layout implied by
            ``[num_shards, H1_shard, H0]``, or None.
        num_shards (int): Total number of shards.
        H1_shard (int): Size of the H1 slice assigned to this shard.
        H0 (int): Size of the H0 (partition) dimension.
        shard_id (int): Index of the current shard.

    Returns:
        bias_sb: ``[H0, H1_shard]`` bf16 SBUF tensor, or None if ``bias_hbm`` is None.
    """
    if bias_hbm == None:
        return None
    bias_reshaped = bias_hbm.reshape((num_shards, H1_shard, H0))
    sharded_view = bias_reshaped.select(dim=0, index=shard_id)
    bias_sb = nl.ndarray((H0, H1_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
    bias_sb_view = bias_sb
    # dma_transpose requires 4D access patterns
    while sharded_view.ndim < 4:
        sharded_view = sharded_view.expand_dim(1)
    while bias_sb_view.ndim < 4:
        bias_sb_view = bias_sb_view.expand_dim(1)
    nisa.dma_transpose(dst=bias_sb_view, src=sharded_view)
    return bias_sb
