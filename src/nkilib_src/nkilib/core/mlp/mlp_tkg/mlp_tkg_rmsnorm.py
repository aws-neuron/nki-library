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

"""RMSNorm kernel optimized for token generation (decoding) phase with efficient sharding and memory management."""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.common_types import HiddenLayout
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_verified_program_sharding_info
from ...utils.logging import get_logger
from ...utils.tiled_range import TiledRange

# Tile size for T (token) dimension processing in HT layout
T_FULL_TILE_SIZE = 512

# DMA engine mode
_DGE_MODE_NONE = 3

# Heuristic threshold: when T > threshold, use shard_on_h=False to avoid sendrecv cost
SHARDING_THRESHOLD = 18


def _transpose_th_fused(
    src: nl.NkiTensor,
    dst: nl.NkiTensor,
    gamma: nl.NkiTensor,
    tile_T: int,
    H0: int,
    H2: int,
):
    """
    Transpose + gamma multiply: [tile_T, H0, H2] → [H0, H2, tile_T] * gamma.

    Transposes each H2 slice [tile_T, H0] → [H0, tile_T] via nc_transpose into PSUM,
    packing multiple tiles per PSUM bank. Drains PSUM with fused gamma multiply
    in a single tensor_tensor instruction.

    Args:
        src: [tile_T, H0, H2] in SBUF (source data, original H0-major layout).
        dst: [H0, H2, tile_T] in SBUF (output buffer).
        gamma: [H0, H2] in SBUF (gamma weights for this shard).
        tile_T: number of tokens in this tile.
        H0: partition dimension size (128).
        H2: number of H2 tiles.
    """
    _psum_fmax = nl.tile_size.psum_fmax

    padded_tile_T = ((tile_T * 2 + 3) // 4) * 4 // 2
    tiles_per_psum = _psum_fmax // padded_tile_T

    for batch_start in range(0, H2, tiles_per_psum):
        batch_end = min(batch_start + tiles_per_psum, H2)
        batch_size = batch_end - batch_start

        tp_psum = nl.ndarray((H0, batch_size * padded_tile_T), dtype=src.dtype, buffer=nl.psum)

        for i in range(batch_size):
            h2_idx = batch_start + i
            tile_src = src.slice(dim=2, start=h2_idx, end=h2_idx + 1).squeeze_dim(dim=2)
            col_offset = i * padded_tile_T
            nisa.nc_transpose(
                dst=tp_psum[0:H0, col_offset : col_offset + tile_T],
                data=tile_src,
            )

        tp_psum_view = tp_psum.reshape_dim(dim=1, shape=[batch_size, padded_tile_T]).slice(dim=2, start=0, end=tile_T)
        dst_batch = dst.slice(dim=1, start=batch_start, end=batch_end)
        gamma_batch = gamma.slice(dim=1, start=batch_start, end=batch_end)
        gamma_broadcast = gamma_batch.expand_dim(dim=2).broadcast(dim=2, size=tile_T)
        nisa.tensor_tensor(
            dst_batch,
            tp_psum_view,
            gamma_broadcast,
            nl.multiply,
        )


def rmsnorm_tkg(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    hidden_scale: float,
    eps: float = 1e-6,
    hidden_dim_tp: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm implementation optimized for inference token generation (decoding) phase.

    Args:
        input (nl.NkiTensor): [B, S, H] when in HBM or [H0, T, H//128] when in SBUF.
        gamma (nl.NkiTensor): [1, H], Gamma tensor in HBM.
        output (nl.NkiTensor): Output tensor in SBUF.
            _th path: [H0, shard_H1, T]. _ht path: [H0, T, shard_H1].
        hidden_scale (float): 1 / H for mean calculation.
        eps (float): Epsilon for numerical stability. Default is 1e-6.
        hidden_dim_tp (bool): If True, input H dimension view is (H/128, 128). Default is False.
        sbm (Optional[SbufManager]): SBUF memory manager. Default is None.

    Returns:
        Tuple of (output, HiddenLayout) indicating the output layout.
    """

    # Validate output shape
    _H0 = nl.tile_size.pmax
    kernel_assert(len(output.shape) in (2, 3), f"output must be 2D or 3D, got {len(output.shape)}D")
    kernel_assert(output.shape[0] == _H0, f"output partition dim must be {_H0}, got {output.shape[0]}")
    kernel_assert((output.buffer == nl.sbuf), "output should be in sbuf")

    if not sbm:
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=nl.tile_size.total_available_sbuf_size,
            logger=get_logger("rmsnorm_tkg"),
            use_auto_alloc=True,
        )

    sbm.open_scope(name="rmsnorm_tkg")

    _, _lnc, _ = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    # Early dispatch: SBUF input is already per-core sharded, always goes to _ht
    if input.buffer == nl.sbuf:
        _T = input.shape[1]
        _shard_H1 = input.shape[2]
        shard_on_h = _lnc > 1

        if len(output.shape) == 2:
            output = output.reshape_dim(dim=1, shape=[_T, _shard_H1])
        out_layout = HiddenLayout.H0_T_H1
        _rmsnorm_tkg_ht(
            input=input,
            gamma=gamma,
            output=output,
            hidden_scale=hidden_scale,
            eps=eps,
            sbm=sbm,
            shard_on_h=shard_on_h,
            hidden_dim_tp=hidden_dim_tp,
        )
    else:
        # HBM input path
        _T = input.shape[0] * input.shape[1] if len(input.shape) == 3 else input.shape[0]
        _H = input.shape[-1]
        _H1 = _H // _H0

        if _lnc == 1:
            shard_on_h = False
        else:
            shard_on_h = (_H1 % _lnc == 0) and (SHARDING_THRESHOLD <= _T)

        # Determine shard_H1 from output shape
        if len(output.shape) == 2:
            _shard_H1 = output.shape[1] // _T
        else:
            _shard_H1 = output.shape[1] if output.shape[2] == _T else output.shape[2]

        # T-sharding: output covers full H, no H-sharding needed
        if _shard_H1 == _H1:
            shard_on_h = False

        if _T >= 128:
            # _th path: output is [H0, H1_shard, T]
            if len(output.shape) == 2:
                output = output.reshape_dim(dim=1, shape=[_shard_H1, _T])
            out_layout = HiddenLayout.H0_H1_T
            _rmsnorm_tkg_th(
                input=input,
                gamma=gamma,
                output=output,
                hidden_scale=hidden_scale,
                eps=eps,
                sbm=sbm,
                shard_on_h=shard_on_h,
                hidden_dim_tp=hidden_dim_tp,
            )
        else:
            # _ht path: output is [H0, T, H1_shard]
            if len(output.shape) == 2:
                output = output.reshape_dim(dim=1, shape=[_T, _shard_H1])
            out_layout = HiddenLayout.H0_T_H1
            _rmsnorm_tkg_ht(
                input=input,
                gamma=gamma,
                output=output,
                hidden_scale=hidden_scale,
                eps=eps,
                sbm=sbm,
                shard_on_h=shard_on_h,
                hidden_dim_tp=hidden_dim_tp,
            )

    sbm.close_scope()

    return output, out_layout


def _rmsnorm_tkg_th(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    hidden_scale: float,
    eps: float = 1e-6,
    hidden_dim_tp: bool = False,
    shard_on_h: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm in [T, H] layout (T on partition dim, H on free dim).

    Computation is done entirely in [T, H] layout. The reduction along H is a
    single fused activation_reduce op. The final result is transposed to the
    output layout [H0, H1_shard, T].

    Args:
        input (NkiTensor): [B, S, H] or [T, H] in HBM (always full H). Flattened to [T, H] if 3D.
        gamma (NkiTensor): [1, H] in HBM (full H).
        output (NkiTensor): [H0, H1_shard, T] in SBUF.
        hidden_scale (float): 1 / H for mean calculation.
        eps (float): Epsilon for numerical stability.
        hidden_dim_tp (bool): If True, use TP-sharded hidden dim layout.
        shard_on_h (bool): If True, each NC loads only its H-shard from input
            and exchanges partial sum(x^2) via sendrecv.
        sbm (Optional[SbufManager]): SBUF memory manager instance.
    """
    # [B, S, H] -> [T, H]
    if len(input.shape) == 3:
        input = input.flatten_dims(start_dim=0, end_dim=1)

    T, H = input.shape
    T0 = min(T, nl.tile_size.pmax)
    H0 = nl.tile_size.pmax
    H1 = H // H0
    inter_dtype = nl.float32

    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    H2 = output.shape[1]  # shard_H1 from output (handles TP != LNC)
    # When output covers full H (T-sharding case), no H-sharding needed
    if H2 == H1:
        lnc = 1
        shard_id = 0
    H_per_shard = H0 * H2
    H_shard_start = shard_id * H_per_shard
    # Size of H loaded per tile: full H or just this NC's shard
    H_load = H_per_shard if shard_on_h else H

    alloc_tensor = sbm.alloc_heap
    num_allocs = 0

    # Pre-allocate tensors (reused across T-tiles)
    gamma_align = 32 if hidden_dim_tp else None
    gamma_sb = alloc_tensor(
        shape=(H0, H2), dtype=gamma.dtype, buffer=nl.sbuf, name="rmsnorm_th_gamma", align=gamma_align
    )
    num_allocs += 1
    input_buf = alloc_tensor(
        shape=(T0, H_load), dtype=input.dtype, buffer=nl.sbuf, name="rmsnorm_th_input_buf", align=32
    )
    num_allocs += 1
    reduced_sq = alloc_tensor(shape=(T0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_th_reduced_sq")
    num_allocs += 1
    square_buf = alloc_tensor(
        shape=(T0, H_load), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_th_square_buf", align=32
    )
    sbm.pop_heap()  # trick to reuse start addr of square_buf

    if shard_on_h:
        remote_reduced_sq = alloc_tensor(
            shape=(T0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_th_remote_reduced_sq"
        )
        num_allocs += 1

    # Load gamma once (always this core's shard only)
    if hidden_dim_tp:
        gamma_hbm_view = gamma.flatten_dims(start_dim=0, end_dim=1).reshape_dim(dim=0, shape=[lnc, H2, H0])
        gamma_hbm_view = gamma_hbm_view.select(dim=0, index=shard_id)
        nisa.dma_transpose(dst=gamma_sb, src=gamma_hbm_view, dge_mode=_DGE_MODE_NONE)
    else:
        gamma_hbm_view = gamma.flatten_dims(start_dim=0, end_dim=1).reshape_dim(dim=0, shape=[lnc, H0, H2])
        gamma_hbm_view = gamma_hbm_view.select(dim=0, index=shard_id)
        nisa.dma_copy(dst=gamma_sb, src=gamma_hbm_view, dge_mode=_DGE_MODE_NONE)

    # Slice input to this NC's H-shard for DMA loads when shard_on_h
    if shard_on_h:
        input_shard = input.slice(dim=1, start=H_shard_start, end=H_shard_start + H_per_shard)
    else:
        input_shard = input

    for t_tile in TiledRange(T, T0):
        tile_T = t_tile.size

        # Slice per-tile views from pre-allocated buffers
        tile_input = input_buf.slice(dim=0, start=0, end=tile_T)
        tile_sq = square_buf.slice(dim=0, start=0, end=tile_T)
        tile_red = reduced_sq.slice(dim=0, start=0, end=tile_T)

        # --- Step 1: Load input tile from HBM → SBUF [tile_T, H_load] ---
        input_tile = input_shard.slice(dim=0, start=t_tile.start_offset, end=t_tile.end_offset)
        nisa.dma_copy(dst=tile_input, src=input_tile, dge_mode=_DGE_MODE_NONE)

        # --- Step 2: Compute x^2 and reduce along H: [tile_T, H_load] → [tile_T, 1] ---
        nisa.activation_reduce(
            tile_sq,
            op=nl.square,
            data=tile_input,
            reduce_op=nl.add,
            reduce_res=tile_red,
        )

        # --- Step 3: Cross-core exchange (shard_on_h only) ---
        if shard_on_h:
            tile_remote_red = remote_reduced_sq.slice(dim=0, start=0, end=tile_T)
            nisa.sendrecv(
                dst=tile_remote_red,
                src=tile_red,
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
                dma_engine=nisa.dma_engine.gpsimd_dma,
            )
            nisa.tensor_tensor(
                tile_red,
                tile_red,
                tile_remote_red,
                nl.add,
            )

        # --- Step 4: Normalization factor: 1/sqrt(mean(x^2) + eps) ---
        nisa.activation(tile_red, op=nl.rsqrt, data=tile_red, scale=hidden_scale, bias=eps)

        # --- Step 5: Normalize input shard: input_shard * norm_factor ---
        if shard_on_h:
            norm_result_shard = tile_input
        else:
            norm_result_shard = tile_input.slice(dim=1, start=H_shard_start, end=H_shard_start + H_per_shard)
        nisa.tensor_scalar(
            norm_result_shard,
            (tile_input if shard_on_h else norm_result_shard),
            op0=nl.multiply,
            operand0=tile_red,
        )

        # --- Step 6+7: Transpose [tile_T, H_per_shard] → [H0, H2, tile_T] and gamma multiply ---
        dst = output.slice(dim=2, start=t_tile.start_offset, end=t_tile.end_offset)
        if hidden_dim_tp:
            # hidden_dim_tp: physical layout is [tile_T, H2, H0], dma_transpose reverses to [H0, H2, tile_T]
            src = norm_result_shard.reshape_dim(dim=1, shape=[H2, H0])
            nisa.dma_transpose(src=src, dst=dst, dge_mode=_DGE_MODE_NONE)
            # Gamma multiply (separate for TP path)
            gamma_broadcast = gamma_sb.expand_dim(dim=2).broadcast(dim=2, size=tile_T)
            nisa.tensor_tensor(
                dst,
                dst,
                gamma_broadcast,
                nl.multiply,
            )
        else:
            # non-TP: data in original [H0, H2] order, use nc_transpose fallback
            src = norm_result_shard.reshape_dim(dim=1, shape=[H0, H2])
            _transpose_th_fused(
                src=src,
                dst=dst,
                gamma=gamma_sb,
                tile_T=tile_T,
                H0=H0,
                H2=H2,
            )

    for _ in range(num_allocs):
        sbm.pop_heap()


def _rmsnorm_tkg_ht(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    hidden_scale: float,
    eps: float = 1e-6,
    hidden_dim_tp: bool = False,
    shard_on_h: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    RMSNorm in [H, T] layout (H on partition dim, T on free dim).

    Used when input is already in SBUF or T < 128. Requires a matmul with
    all-ones to complete the reduction across the H0 partition dimension.
    Input is [H0, T, H1], output is [H0, T, shard_H1].

    Args:
        input (NkiTensor): [T, H] when in HBM or [H0, T, H1] when in SBUF.
        gamma (NkiTensor): [1, H], Gamma tensor in HBM.
        output (NkiTensor): [H0, T, shard_H1] in SBUF, Output tensor.
        hidden_scale (float): 1 / H for mean calculation.
        eps (float): Epsilon for numerical stability.
        hidden_dim_tp (bool): If True, use TP-sharded hidden dim layout.
        shard_on_h (bool): If True, input is already per-core sharded and needs cross-core reduction.
        sbm (Optional[SbufManager]): SBUF memory manager instance.
    """
    output_sb_view = output

    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_tkg", (0, 1))

    # Extract dimensions from input
    H0 = nl.tile_size.pmax
    # Derive shard dimensions from output shape
    shard_H1_dim = output.shape[2]  # output is [H0, T, shard_H1]
    shard_H = H0 * shard_H1_dim

    if input.buffer == nl.sbuf:
        # SBUF input is always already sharded: [H0, T, H1_shard]
        H0, T, H1 = input.shape
        # When shard_on_h=False and output matches input H1, no cross-core reduction needed
        if shard_H1_dim == H1 and not shard_on_h:
            lnc = 1
            shard_id = 0
        # Gamma slice: use shard_H to get this core's portion
        gamma = gamma.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)
    else:
        if len(input.shape) == 3:
            T = input.shape[0] * input.shape[1]
            H = input.shape[2]
        else:
            T, H = input.shape
        H1 = H // H0
        # When output covers full H (T-sharding case), no H-sharding needed
        if shard_H1_dim == H1 and not shard_on_h:
            lnc = 1
            shard_id = 0
        # Gamma slice
        gamma = gamma.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)
        # Input slice (when shard_on_h and not hidden_dim_tp, load only this core's shard from HBM)
        # For hidden_dim_tp, we load full H and slice in SBUF to avoid non-contiguous flatten_dims
        if shard_on_h and not hidden_dim_tp:
            input_view_flat = input.flatten_dims(start_dim=0, end_dim=1) if len(input.shape) == 3 else input
            input = input_view_flat.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    # Re-extract T and H1 after potential slicing
    if not (input.buffer == nl.sbuf):
        if len(input.shape) == 2:
            T, H = input.shape
        else:
            T = input.shape[0] * input.shape[1]
            H = input.shape[2]
        H1 = H // H0

    inter_dtype = nl.float32
    T0 = min(T, T_FULL_TILE_SIZE)

    alloc_tensor = sbm.alloc_heap
    num_allocs = 0

    # Load input into SBUF (or use existing SBUF input)
    if input.buffer == nl.sbuf:
        input_sb = input
    else:
        # Input is [T, H] in HBM (already sliced if shard_on_h), load to [H0, T, H1] in SBUF
        input_align = 32 if hidden_dim_tp else None
        input_sb = alloc_tensor(
            shape=(H0, T, H1), dtype=input.dtype, buffer=nl.sbuf, name="rmsnorm_ht_input", align=input_align
        )
        num_allocs += 1
        # Load [T, H] from HBM to [H0, T, H1] in SBUF
        input_flat = input.flatten_dims(start_dim=0, end_dim=1) if len(input.shape) == 3 else input
        if hidden_dim_tp:
            # HBM layout is [T, H] viewed as [T*H1, H0] → 2D dma_transpose to [H0, T*H1] → reshape [H0, T, H1]
            input_hbm_2d = input_flat.reshape_dim(dim=1, shape=[H1, H0]).flatten_dims(start_dim=0, end_dim=1)
            nisa.dma_transpose(dst=input_sb.reshape((H0, T * H1)), src=input_hbm_2d, dge_mode=_DGE_MODE_NONE)
            # When shard_on_h, slice SBUF to this core's shard after loading full H
            if shard_on_h:
                input_sb = input_sb.slice(dim=2, start=shard_id * shard_H1_dim, end=(shard_id + 1) * shard_H1_dim)
                H1 = shard_H1_dim
        else:
            # HBM layout [T, H] = [T, num_shards, H0, H2] when hidden_dim_tp=False
            # Load to [H0, T, num_shards, H2] = [H0, T, H1] via dma_copy
            num_shards = H1 // shard_H1_dim
            H2 = shard_H1_dim
            input_hbm_view = input_flat.reshape_dim(dim=1, shape=[num_shards, H0, H2]).permute(dims=[2, 0, 1, 3])
            input_sb_reshaped = input_sb.reshape_dim(dim=2, shape=[num_shards, H2])
            nisa.dma_copy(dst=input_sb_reshaped, src=input_hbm_view, dge_mode=_DGE_MODE_NONE)

    # Load gamma (always this core's shard — gamma was sliced above)
    gamma_align = 32 if hidden_dim_tp else None
    gamma_sb = alloc_tensor(
        shape=(H0, shard_H1_dim), dtype=gamma.dtype, buffer=nl.sbuf, name="rmsnorm_gamma", align=gamma_align
    )
    num_allocs += 1
    # Gamma is [1, shard_H] → reshape and load to [H0, shard_H1]
    if hidden_dim_tp:
        # gamma viewed as [1, shard_H1, H0] → select dim 0 → [shard_H1, H0] → dma_transpose to [H0, shard_H1]
        gamma_reshaped = gamma.reshape_dim(dim=1, shape=[shard_H1_dim, H0]).select(dim=0, index=0)
        nisa.dma_transpose(dst=gamma_sb, src=gamma_reshaped, dge_mode=_DGE_MODE_NONE)
    else:
        # gamma viewed as [1, H0, shard_H1] → select dim 0 → [H0, shard_H1] → dma_copy
        gamma_reshaped = gamma.reshape_dim(dim=1, shape=[H0, shard_H1_dim]).select(dim=0, index=0)
        nisa.dma_copy(dst=gamma_sb, src=gamma_reshaped, dge_mode=_DGE_MODE_NONE)

    # Allocate eps and matmul reduction constant
    eps_sb = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_eps")
    num_allocs += 1
    nisa.memset(eps_sb, value=eps)

    matmul_reduction_const = alloc_tensor(
        shape=(H0, H0), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_mm_reduced_const"
    )
    num_allocs += 1
    nisa.memset(matmul_reduction_const, value=1.0)

    # Pre-allocate per-tile buffers at max tile size (reused across iterations)
    square_buf = alloc_tensor(shape=(H0, T0, H1), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_ht_square_buf")
    num_allocs += 1
    reduced_sq = alloc_tensor(shape=(H0, T0), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_ht_reduced_sq")
    num_allocs += 1
    gamma_mult_buf = alloc_tensor(
        shape=(H0, T0, shard_H1_dim), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_ht_gamma_mult"
    )
    num_allocs += 1

    if shard_on_h:
        remote_reduced_sq = alloc_tensor(
            shape=(H0, T0), dtype=inter_dtype, buffer=nl.sbuf, name="rmsnorm_ht_remote_reduced_sq"
        )
        num_allocs += 1

    for t_tile in TiledRange(T, T_FULL_TILE_SIZE):
        tile_T = t_tile.size

        input_sb_view_tile = input_sb.slice(dim=1, start=t_tile.start_offset, end=t_tile.start_offset + tile_T)
        gamma_sb_view_tile = gamma_sb.expand_dim(dim=1).broadcast(dim=1, size=tile_T)
        output_sb_view_tile = output_sb_view.slice(dim=1, start=t_tile.start_offset, end=t_tile.start_offset + tile_T)

        # Slice pre-allocated buffers to current tile size
        tile_sq = square_buf.slice(dim=1, start=0, end=tile_T)
        tile_red = reduced_sq.slice(dim=1, start=0, end=tile_T)
        tile_gamma_mult = gamma_mult_buf.slice(dim=1, start=0, end=tile_T)

        # --- Step 1: Compute x^2 [H0, tile_T, H1] ---
        nisa.activation(tile_sq, op=nl.square, data=input_sb_view_tile)

        # --- Step 2: Reduce along H1 (free dim): [H0, tile_T, H1] → [H0, tile_T] ---
        nisa.tensor_reduce(tile_red, nl.add, tile_sq, axis=2)

        # --- Step 3: Gamma multiply on this core's H-shard ---
        # Moved before sendrecv/matmul to overlap with norm factor computation
        if shard_on_h:
            input_shard_tile = input_sb_view_tile
        else:
            input_shard_tile = input_sb_view_tile.slice(
                dim=2, start=shard_id * shard_H1_dim, end=(shard_id + 1) * shard_H1_dim
            )

        nisa.tensor_tensor(
            tile_gamma_mult,
            input_shard_tile,
            gamma_sb_view_tile,
            nl.multiply,
        )

        # --- Step 4: Cross-core exchange (shard_on_h only) ---
        if shard_on_h:
            tile_remote_red = remote_reduced_sq.slice(dim=1, start=0, end=tile_T)
            nisa.sendrecv(
                dst=tile_remote_red,
                src=tile_red,
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
                dma_engine=nisa.dma_engine.gpsimd_dma,
            )
            nisa.tensor_tensor(
                tile_red,
                tile_red,
                tile_remote_red,
                nl.add,
            )

        # --- Step 5: Reduce across H0 (partition dim) via matmul ---
        # ones[H0,H0] @ reduced[H0,tile_T] → [H0,tile_T]
        final_reduced = nl.ndarray((H0, tile_T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            stationary=matmul_reduction_const,
            moving=tile_red,
            dst=final_reduced,
        )

        # --- Step 6: Normalization factor: 1/sqrt(mean(x^2) + eps) ---
        nisa.activation(
            tile_red,
            op=nl.rsqrt,
            data=final_reduced[...],
            scale=hidden_scale,
            bias=eps_sb,
        )

        # --- Step 7: Final multiply: gamma_mult * norm_factor → output ---
        # gamma_mult is [H0, tile_T, shard_H1], norm_factor is [H0, tile_T]
        # Write result directly to output [H0, tile_T, shard_H1]
        reduced_view = tile_red.expand_dim(dim=2).broadcast(dim=2, size=shard_H1_dim)
        nisa.tensor_tensor(
            output_sb_view_tile,
            tile_gamma_mult,
            reduced_view,
            nl.multiply,
        )

    for _ in range(num_allocs):
        sbm.pop_heap()
