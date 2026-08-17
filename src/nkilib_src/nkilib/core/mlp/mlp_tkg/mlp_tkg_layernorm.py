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

"""LayerNorm kernel optimized for token generation (decoding) phase with efficient sharding and memory management."""

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


def layernorm_tkg(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    hidden_scale: float,
    beta: Optional[nl.NkiTensor] = None,
    eps: float = 1e-6,
    hidden_dim_tp: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    LayerNorm implementation optimized for inference token generation (decoding) phase.

    Args:
        input (nl.NkiTensor): [B, S, H] when in HBM or [H0, T, H//128] when in SBUF.
        gamma (nl.NkiTensor): [1, H], Gamma tensor in HBM.
        output (nl.NkiTensor): [H0, shard_H1, T] in SBUF, Output tensor.
        hidden_scale (float): 1 / H for mean calculation.
        beta (Optional[nl.NkiTensor]): [1, H], Beta tensor in HBM. Default is None.
        eps (float): Epsilon for numerical stability. Default is 1e-6.
        hidden_dim_tp (bool): If True, input H dimension view is (H/128, 128). Default is False.
        sbm (Optional[SbufManager]): SBUF memory manager. Default is None.

    Returns:
        Output tensor with LayerNorm applied, [H0, shard_H1, T] in SBUF.

    Notes:
        - H must be divisible by 128 (partition dimension).
        - shard_on_h is determined internally based on H divisibility by LNC.
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
            logger=get_logger("layernorm_tkg"),
            use_auto_alloc=True,
        )

    sbm.open_scope(name="layernorm_tkg")

    _, _lnc, _ = get_verified_program_sharding_info("layernorm_tkg", (0, 1))

    # Early dispatch: SBUF input is already per-core sharded, always goes to _ht
    if input.buffer == nl.sbuf:
        _T = input.shape[1]
        _shard_H1 = input.shape[2]
        shard_on_h = _lnc > 1

        if len(output.shape) == 2:
            output = output.reshape_dim(dim=1, shape=[_T, _shard_H1])
        out_layout = HiddenLayout.H0_T_H1
        _layernorm_tkg_ht(
            input=input,
            gamma=gamma,
            beta=beta,
            output=output,
            hidden_scale=hidden_scale,
            eps=eps,
            hidden_dim_tp=hidden_dim_tp,
            shard_on_h=shard_on_h,
            sbm=sbm,
        )
    else:
        # HBM input path
        _T = input.shape[0] * input.shape[1] if len(input.shape) == 3 else input.shape[0]

        # Reduce over the full hidden dim locally on every core instead of exchanging partial
        # sum(x) and sum(x^2) across cores. The cross-core exchange (two nisa.sendrecv plus a
        # per-partial reduction matmul into an auto-allocated PSUM tile) carries a latent
        # non-determinism that the backend's PSUM address-rotation can expose under certain
        # schedules, producing intermittent divergence in the top tokens of the output. Reducing
        # locally is numerically equivalent and race-free, and the extra reduction work is
        # negligible relative to the MLP projections.
        shard_on_h = False

        # Determine shard_H1 from output shape
        if len(output.shape) == 2:
            _shard_H1 = output.shape[1] // _T
        else:
            _shard_H1 = output.shape[1] if output.shape[2] == _T else output.shape[2]

        if _T >= _H0:
            # _th path: output is [H0, H1_shard, T]
            if len(output.shape) == 2:
                output = output.reshape_dim(dim=1, shape=[_shard_H1, _T])
            out_layout = HiddenLayout.H0_H1_T
            _layernorm_tkg_th(
                input=input,
                gamma=gamma,
                beta=beta,
                output=output,
                hidden_scale=hidden_scale,
                eps=eps,
                hidden_dim_tp=hidden_dim_tp,
                shard_on_h=shard_on_h,
                sbm=sbm,
            )
        else:
            # _ht path: output is [H0, T, H1_shard]
            if len(output.shape) == 2:
                output = output.reshape_dim(dim=1, shape=[_T, _shard_H1])
            out_layout = HiddenLayout.H0_T_H1
            _layernorm_tkg_ht(
                input=input,
                gamma=gamma,
                beta=beta,
                output=output,
                hidden_scale=hidden_scale,
                eps=eps,
                hidden_dim_tp=hidden_dim_tp,
                shard_on_h=shard_on_h,
                sbm=sbm,
            )

    sbm.close_scope()

    return output, out_layout


def _layernorm_tkg_th(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: Optional[nl.NkiTensor],
    output: nl.NkiTensor,
    hidden_scale: float,
    eps: float = 1e-6,
    hidden_dim_tp: bool = False,
    shard_on_h: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    LayerNorm in [T, H] layout (T on partition dim, H on free dim).

    Computation is done entirely in [T, H] layout. The reduction along H uses
    activation_reduce. The final result is transposed to output layout [H0, H1_shard, T].

    Args:
        input (NkiTensor): [B, S, H] or [T, H] in HBM. Flattened to [T, H] if 3D.
        gamma (NkiTensor): [1, H] in HBM.
        beta (Optional[nl.NkiTensor]): [1, H] in HBM, or None.
        output (NkiTensor): [H0, H1_shard, T] in SBUF.
        hidden_scale (float): 1 / H for mean calculation.
        eps (float): Epsilon for numerical stability.
        hidden_dim_tp (bool): If True, use TP-sharded hidden dim layout.
        shard_on_h (bool): If True, each NC loads only its H-shard.
        sbm (Optional[SbufManager]): SBUF memory manager instance.
    """
    # [B, S, H] -> [T, H]
    if len(input.shape) == 3:
        input = input.flatten_dims(start_dim=0, end_dim=1)

    T, H = input.shape
    T0 = nl.tile_size.pmax
    H0 = nl.tile_size.pmax
    H1 = H // H0
    inter_dtype = nl.float32

    _, lnc, shard_id = get_verified_program_sharding_info("layernorm_tkg", (0, 1))

    H2 = output.shape[1]  # shard_H1 from output (handles TP != LNC and T-sharding)
    # When output covers full H (T-sharding case), no H-sharding needed
    if H2 == H1:
        lnc = 1
        shard_id = 0
    H_per_shard = H0 * H2
    H_shard_start = shard_id * H_per_shard
    H_load = H_per_shard if shard_on_h else H

    alloc_tensor = sbm.alloc_heap
    num_allocs = 0

    # Pre-allocate tensors (reused across T-tiles)
    gamma_align = 32 if hidden_dim_tp else None
    gamma_sb = alloc_tensor(
        shape=(H0, H2), dtype=gamma.dtype, buffer=nl.sbuf, name="layernorm_th_gamma", align=gamma_align
    )
    num_allocs += 1
    beta_sb = None
    if beta is not None:
        beta_sb = alloc_tensor(
            shape=(H0, H2), dtype=beta.dtype, buffer=nl.sbuf, name="layernorm_th_beta", align=gamma_align
        )
        num_allocs += 1
    input_buf = alloc_tensor(shape=(T0, H_load), dtype=input.dtype, buffer=nl.sbuf, name="layernorm_th_input_buf")
    num_allocs += 1
    reduced_sum = alloc_tensor(shape=(T0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_th_reduced_sum")
    num_allocs += 1
    reduced_sq = alloc_tensor(shape=(T0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_th_reduced_sq")
    num_allocs += 1
    square_buf = alloc_tensor(shape=(T0, H_load), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_th_square_buf")
    num_allocs += 1

    if shard_on_h:
        remote_reduced_sum = alloc_tensor(
            shape=(T0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_th_remote_sum"
        )
        num_allocs += 1
        remote_reduced_sq = alloc_tensor(
            shape=(T0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_th_remote_sq"
        )
        num_allocs += 1

    # Reuse square_buf as permutation temp in Step 6 (unused after Step 2)
    # square_buf is (T0, H_load) float32 → reinterpret as bf16 → (T0, H_load*2), slice (T0, H_per_shard)
    perm_buf = None
    if not hidden_dim_tp:
        perm_buf = square_buf.view(input.dtype).slice(dim=1, start=0, end=H_per_shard)

    # Load gamma (always this core's shard)
    if hidden_dim_tp:
        gamma_hbm_view = gamma.flatten_dims(start_dim=0, end_dim=1).reshape_dim(dim=0, shape=[lnc, H2, H0])
        gamma_hbm_view = gamma_hbm_view.select(dim=0, index=shard_id)
        nisa.dma_transpose(dst=gamma_sb, src=gamma_hbm_view, dge_mode=_DGE_MODE_NONE)
    else:
        gamma_hbm_view = gamma.flatten_dims(start_dim=0, end_dim=1).reshape_dim(dim=0, shape=[lnc, H0, H2])
        gamma_hbm_view = gamma_hbm_view.select(dim=0, index=shard_id)
        nisa.dma_copy(dst=gamma_sb, src=gamma_hbm_view, dge_mode=_DGE_MODE_NONE)

    # Load beta (if present)
    if beta is not None:
        if hidden_dim_tp:
            beta_hbm_view = beta.flatten_dims(start_dim=0, end_dim=1).reshape_dim(dim=0, shape=[lnc, H2, H0])
            beta_hbm_view = beta_hbm_view.select(dim=0, index=shard_id)
            nisa.dma_transpose(dst=beta_sb, src=beta_hbm_view, dge_mode=_DGE_MODE_NONE)
        else:
            beta_hbm_view = beta.flatten_dims(start_dim=0, end_dim=1).reshape_dim(dim=0, shape=[lnc, H0, H2])
            beta_hbm_view = beta_hbm_view.select(dim=0, index=shard_id)
            nisa.dma_copy(dst=beta_sb, src=beta_hbm_view, dge_mode=_DGE_MODE_NONE)

    # Slice input to this NC's H-shard
    if shard_on_h:
        input_shard = input.slice(dim=1, start=H_shard_start, end=H_shard_start + H_per_shard)
    else:
        input_shard = input

    for t_tile in TiledRange(T, T0):
        tile_T = t_tile.size

        tile_input = input_buf.slice(dim=0, start=0, end=tile_T)
        tile_sq = square_buf.slice(dim=0, start=0, end=tile_T)
        tile_sum = reduced_sum.slice(dim=0, start=0, end=tile_T)
        tile_var = reduced_sq.slice(dim=0, start=0, end=tile_T)

        # --- Step 1: Load input tile from HBM → SBUF [tile_T, H_load] ---
        input_tile = input_shard.slice(dim=0, start=t_tile.start_offset, end=t_tile.end_offset)
        nisa.dma_copy(dst=tile_input, src=input_tile, dge_mode=_DGE_MODE_NONE)

        # --- Step 2: Compute sum(x) and sum(x^2) along H ---
        # sum(x): [tile_T, H_load] → [tile_T, 1]
        nisa.activation_reduce(
            tile_sq,
            op=nl.square,
            data=tile_input,
            reduce_op=nl.add,
            reduce_res=tile_var,
        )
        # For mean, we need sum(x) separately
        nisa.tensor_reduce(tile_sum, nl.add, tile_input, axis=1)

        # --- Step 3: Cross-core exchange (shard_on_h only) ---
        if shard_on_h:
            tile_remote_sum = remote_reduced_sum.slice(dim=0, start=0, end=tile_T)
            tile_remote_var = remote_reduced_sq.slice(dim=0, start=0, end=tile_T)
            nisa.sendrecv(
                dst=tile_remote_sum,
                src=tile_sum,
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
                dma_engine=nisa.dma_engine.gpsimd_dma,
            )
            nisa.sendrecv(
                dst=tile_remote_var,
                src=tile_var,
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=1,
                dma_engine=nisa.dma_engine.gpsimd_dma,
            )
            nisa.tensor_tensor(tile_sum, tile_sum, tile_remote_sum, nl.add)
            nisa.tensor_tensor(tile_var, tile_var, tile_remote_var, nl.add)

        # --- Step 4: Compute mean and variance ---
        # mean = sum(x) * hidden_scale
        nisa.tensor_scalar(tile_sum, tile_sum, op0=nl.multiply, operand0=hidden_scale)
        # var = sum(x^2) * hidden_scale - mean^2
        # rsqrt(var + eps)
        nisa.activation(tile_var, op=nl.rsqrt, data=tile_var, scale=hidden_scale, bias=eps)

        # --- Step 5: Normalize: (input - mean) * rsqrt(var + eps) ---
        if shard_on_h:
            norm_result_shard = tile_input
        else:
            norm_result_shard = tile_input.slice(dim=1, start=H_shard_start, end=H_shard_start + H_per_shard)
        # input - mean
        nisa.tensor_scalar(
            norm_result_shard,
            (tile_input if shard_on_h else norm_result_shard),
            op0=nl.subtract,
            operand0=tile_sum,
        )
        # * rsqrt(var + eps)
        nisa.tensor_scalar(
            norm_result_shard,
            norm_result_shard,
            op0=nl.multiply,
            operand0=tile_var,
        )

        # --- Step 6: Transpose [tile_T, H_per_shard] → [H0, H2, tile_T] ---
        # dma_transpose swaps partition dim with last free dim:
        #   [tile_T, H2, H0] → [H0, H2, tile_T]
        # When hidden_dim_tp=True, data is already in [H2, H0] order in the free dim.
        # When hidden_dim_tp=False, data is in [H0, H2] order, so we need tensor_copy
        # to reorder to [H2, H0] before transposing.
        dst = output.slice(dim=2, start=t_tile.start_offset, end=t_tile.end_offset)
        if hidden_dim_tp:
            src = norm_result_shard.reshape_dim(dim=1, shape=[H2, H0])
            nisa.dma_transpose(src=src, dst=dst)
        else:
            # Copy [tile_T, H0, H2] → perm_buf [tile_T, H2, H0], then dma_transpose → [H0, H2, tile_T]
            src = norm_result_shard.reshape_dim(dim=1, shape=[H0, H2])
            perm_tile = perm_buf.slice(dim=0, start=0, end=tile_T).reshape_dim(dim=1, shape=[H2, H0])
            nisa.tensor_copy(src=src, dst=perm_tile)
            nisa.dma_transpose(src=perm_tile, dst=dst)

        # --- Step 7: Gamma (and beta) multiplication ---
        gamma_broadcast = gamma_sb.expand_dim(dim=2).broadcast(dim=2, size=tile_T)
        nisa.tensor_tensor(dst, dst, gamma_broadcast, nl.multiply)
        if beta_sb is not None:
            beta_broadcast = beta_sb.expand_dim(dim=2).broadcast(dim=2, size=tile_T)
            nisa.tensor_tensor(dst, dst, beta_broadcast, nl.add)

    for _ in range(num_allocs):
        sbm.pop_heap()


def _layernorm_tkg_ht(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    beta: Optional[nl.NkiTensor],
    output: nl.NkiTensor,
    hidden_scale: float,
    eps: float = 1e-6,
    hidden_dim_tp: bool = False,
    shard_on_h: bool = False,
    sbm: Optional[SbufManager] = None,
):
    """
    LayerNorm in [H, T] layout (H on partition dim, T on free dim).

    Used when input is already in SBUF or T < 128. Requires a matmul with
    all-ones to complete the reduction across the H0 partition dimension.
    Input is [H0, T, H1], output is [H0, T, shard_H1].

    Args:
        input (NkiTensor): [T, H] when in HBM or [H0, T, H1] when in SBUF.
        gamma (NkiTensor): [1, H], Gamma tensor in HBM.
        beta (Optional[nl.NkiTensor]): [1, H], Beta tensor in HBM, or None.
        output (NkiTensor): [H0, T, shard_H1] in SBUF, Output tensor.
        hidden_scale (float): 1 / H for mean calculation.
        eps (float): Epsilon for numerical stability.
        hidden_dim_tp (bool): If True, use TP-sharded hidden dim layout.
        shard_on_h (bool): If True, input is already per-core sharded and needs cross-core reduction.
        sbm (Optional[SbufManager]): SBUF memory manager instance.
    """
    # Shape extraction
    _, lnc, shard_id = get_verified_program_sharding_info("layernorm_tkg", (0, 1))

    H0 = nl.tile_size.pmax
    if input.buffer == nl.sbuf:
        # SBUF input is always already sharded: [H0, T, H1_shard]
        H0, T, H1 = input.shape
        H = H0 * H1
        shard_H = H0 * H1
        shard_H1_dim = H1
    else:
        if len(input.shape) == 3:
            input = input.flatten_dims(start_dim=0, end_dim=1)
        T, H = input.shape
        H1 = H // H0
        shard_H = H // lnc
        shard_H1_dim = shard_H // H0

    # Use output shape to determine actual shard size
    shard_H1_dim = output.shape[2]
    shard_H = H0 * shard_H1_dim

    # When output covers full H (T-sharding case), no H-sharding needed
    if shard_H1_dim == H1 and not shard_on_h:
        lnc = 1
        shard_id = 0

    T0 = min(T, T_FULL_TILE_SIZE)
    inter_dtype = nl.float32

    # Slice gamma/beta to this core's shard
    gamma = gamma.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)
    if beta is not None:
        beta = beta.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)

    if shard_on_h:
        if not (input.buffer == nl.sbuf):
            input = input.slice(dim=1, start=shard_id * shard_H, end=(shard_id + 1) * shard_H)
            H1 = shard_H // H0

    alloc_tensor = sbm.alloc_heap
    num_allocs = 0

    # Load input into SBUF
    if input.buffer == nl.sbuf:
        input_sb = input
    else:
        input_align = 32 if hidden_dim_tp else None
        input_sb = alloc_tensor(
            shape=(H0, T, H1), dtype=input.dtype, buffer=nl.sbuf, name="layernorm_ht_input", align=input_align
        )
        num_allocs += 1
        input_flat = input.flatten_dims(start_dim=0, end_dim=1) if len(input.shape) == 3 else input
        if hidden_dim_tp:
            input_reshaped = input_flat.reshape_dim(dim=1, shape=[H1, H0])
            nisa.dma_transpose(dst=input_sb, src=input_reshaped, dge_mode=_DGE_MODE_NONE)
        else:
            # HBM layout [T, H] = [T, num_shards, H0, H2] when hidden_dim_tp=False
            num_shards = H1 // shard_H1_dim
            H2 = shard_H1_dim
            input_hbm_view = input_flat.reshape_dim(dim=1, shape=[num_shards, H0, H2]).permute(dims=[2, 0, 1, 3])
            input_sb_reshaped = input_sb.reshape_dim(dim=2, shape=[num_shards, H2])
            nisa.dma_copy(dst=input_sb_reshaped, src=input_hbm_view, dge_mode=_DGE_MODE_NONE)

    # Load gamma
    gamma_align = 32 if hidden_dim_tp else None
    gamma_sb = alloc_tensor(
        shape=(H0, shard_H1_dim), dtype=gamma.dtype, buffer=nl.sbuf, name="layernorm_gamma", align=gamma_align
    )
    num_allocs += 1
    if hidden_dim_tp:
        gamma_reshaped = gamma.reshape_dim(dim=1, shape=[shard_H1_dim, H0]).select(dim=0, index=0)
        nisa.dma_transpose(dst=gamma_sb, src=gamma_reshaped, dge_mode=_DGE_MODE_NONE)
    else:
        gamma_reshaped = gamma.reshape_dim(dim=1, shape=[H0, shard_H1_dim]).select(dim=0, index=0)
        nisa.dma_copy(dst=gamma_sb, src=gamma_reshaped, dge_mode=_DGE_MODE_NONE)

    # Load beta (if present)
    beta_sb = None
    if beta is not None:
        beta_sb = alloc_tensor(
            shape=(H0, shard_H1_dim), dtype=beta.dtype, buffer=nl.sbuf, name="layernorm_beta", align=gamma_align
        )
        num_allocs += 1
        if hidden_dim_tp:
            beta_reshaped = beta.reshape_dim(dim=1, shape=[shard_H1_dim, H0]).select(dim=0, index=0)
            nisa.dma_transpose(dst=beta_sb, src=beta_reshaped, dge_mode=_DGE_MODE_NONE)
        else:
            beta_reshaped = beta.reshape_dim(dim=1, shape=[H0, shard_H1_dim]).select(dim=0, index=0)
            nisa.dma_copy(dst=beta_sb, src=beta_reshaped, dge_mode=_DGE_MODE_NONE)

    # Allocate eps and matmul reduction constant
    eps_sb = alloc_tensor(shape=(H0, 1), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_eps")
    num_allocs += 1
    nisa.memset(eps_sb, value=eps)

    matmul_reduction_const = alloc_tensor(shape=(H0, H0), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_mm_const")
    num_allocs += 1
    nisa.memset(matmul_reduction_const, value=1.0)

    # Pre-allocate per-tile buffers
    square_buf = alloc_tensor(shape=(H0, T0, H1), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_ht_square_buf")
    num_allocs += 1
    reduced_sum = alloc_tensor(shape=(H0, T0), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_ht_reduced_sum")
    num_allocs += 1
    reduced_sq = alloc_tensor(shape=(H0, T0), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_ht_reduced_sq")
    num_allocs += 1
    mean_sq_buf = alloc_tensor(shape=(H0, T0), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_ht_mean_sq")
    num_allocs += 1
    gamma_mult_buf = alloc_tensor(
        shape=(H0, T0, shard_H1_dim), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_ht_gamma_mult"
    )
    num_allocs += 1

    if shard_on_h:
        remote_reduced_sum = alloc_tensor(
            shape=(H0, T0), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_ht_remote_sum"
        )
        num_allocs += 1
        remote_reduced_sq = alloc_tensor(
            shape=(H0, T0), dtype=inter_dtype, buffer=nl.sbuf, name="layernorm_ht_remote_sq"
        )
        num_allocs += 1

    output_sb_view = output

    for t_tile in TiledRange(T, T_FULL_TILE_SIZE):
        tile_T = t_tile.size

        input_sb_view_tile = input_sb.slice(dim=1, start=t_tile.start_offset, end=t_tile.start_offset + tile_T)
        gamma_sb_view_tile = gamma_sb.expand_dim(dim=1).broadcast(dim=1, size=tile_T)
        output_sb_view_tile = output_sb_view.slice(dim=1, start=t_tile.start_offset, end=t_tile.start_offset + tile_T)

        # Slice pre-allocated buffers
        tile_sq = square_buf.slice(dim=1, start=0, end=tile_T)
        tile_sum = reduced_sum.slice(dim=1, start=0, end=tile_T)
        tile_var = reduced_sq.slice(dim=1, start=0, end=tile_T)
        tile_mean_sq = mean_sq_buf.slice(dim=1, start=0, end=tile_T)
        tile_gamma_mult = gamma_mult_buf.slice(dim=1, start=0, end=tile_T)

        H0, tile_T, H1_input = input_sb_view_tile.shape

        # --- Step 1: Compute x^2 and reduce sum(x^2) along H1 ---
        nisa.activation(tile_sq, op=nl.square, data=input_sb_view_tile)
        nisa.tensor_reduce(tile_var, nl.add, tile_sq, axis=2)

        # --- Step 2: Reduce sum(x) along H1 ---
        nisa.tensor_reduce(tile_sum, nl.add, input_sb_view_tile, axis=2)

        # --- Step 3: Gamma multiply (early, overlaps with norm factor computation) ---
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
            tile_remote_sum = remote_reduced_sum.slice(dim=1, start=0, end=tile_T)
            tile_remote_var = remote_reduced_sq.slice(dim=1, start=0, end=tile_T)
            nisa.sendrecv(
                dst=tile_remote_sum,
                src=tile_sum,
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=0,
                dma_engine=nisa.dma_engine.gpsimd_dma,
            )
            nisa.sendrecv(
                dst=tile_remote_var,
                src=tile_var,
                send_to_rank=1 - shard_id,
                recv_from_rank=1 - shard_id,
                pipe_id=1,
                dma_engine=nisa.dma_engine.gpsimd_dma,
            )
            nisa.tensor_tensor(tile_sum, tile_sum, tile_remote_sum, nl.add)
            nisa.tensor_tensor(tile_var, tile_var, tile_remote_var, nl.add)

        # --- Step 5: Reduce across H0 via matmul ---
        final_sum = nl.ndarray((H0, tile_T), dtype=nl.float32, buffer=nl.psum)
        final_var = nl.ndarray((H0, tile_T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(stationary=matmul_reduction_const, moving=tile_sum, dst=final_sum)
        nisa.nc_matmul(stationary=matmul_reduction_const, moving=tile_var, dst=final_var)

        # --- Step 6: Compute mean and rsqrt(var + eps) ---
        # mean = sum(x) / H → tile_sum [H0, tile_T]
        nisa.activation(tile_sum, op=nl.copy, data=final_sum[...], scale=hidden_scale)
        # E[x²] = sum(x²) / H → tile_var [H0, tile_T]
        nisa.activation(tile_var, op=nl.copy, data=final_var[...], scale=hidden_scale)
        # mean² → tile_mean_sq [H0, tile_T]
        nisa.tensor_tensor(tile_mean_sq, tile_sum, tile_sum, nl.multiply)
        # var = E[x²] - mean² → tile_var
        nisa.tensor_tensor(tile_var, tile_var, tile_mean_sq, nl.subtract)
        # var + eps → tile_var
        nisa.tensor_scalar(tile_var, tile_var, op0=nl.add, operand0=eps)
        # rsqrt(var + eps) → tile_var
        nisa.activation(tile_var, op=nl.rsqrt, data=tile_var)

        # --- Step 7: output = gamma * (input - mean) * rsqrt_var + beta ---
        # gamma_mult = gamma * input (from step 3)
        # We want: gamma * (input - mean) * rsqrt_var = (gamma*input - gamma*mean) * rsqrt_var
        # = gamma_mult * rsqrt_var - gamma * mean * rsqrt_var
        #
        # Step 7a: gamma_mult *= rsqrt_var
        var_broadcast = tile_var.expand_dim(dim=2).broadcast(dim=2, size=shard_H1_dim)
        nisa.tensor_tensor(tile_gamma_mult, tile_gamma_mult, var_broadcast, nl.multiply)

        # Step 7b: Compute mean * rsqrt_var → tile_sum (reuse, mean no longer needed separately)
        nisa.tensor_tensor(tile_sum, tile_sum, tile_var, nl.multiply)
        # Broadcast mean*rsqrt_var [H0, tile_T] → [H0, tile_T, shard_H1]
        mean_rsqrt_broadcast = tile_sum.expand_dim(dim=2).broadcast(dim=2, size=shard_H1_dim)
        # gamma * mean * rsqrt_var = gamma_broadcast * mean_rsqrt_broadcast → use output as temp
        nisa.tensor_tensor(output_sb_view_tile, mean_rsqrt_broadcast, gamma_sb_view_tile, nl.multiply)
        # gamma_mult -= gamma * mean * rsqrt_var
        nisa.tensor_tensor(tile_gamma_mult, tile_gamma_mult, output_sb_view_tile, nl.subtract)

        # Step 7c: Add beta and write to output
        if beta_sb is None:
            nisa.tensor_copy(src=tile_gamma_mult, dst=output_sb_view_tile)
        else:
            beta_sb_view_tile = beta_sb.expand_dim(dim=1).broadcast(dim=1, size=tile_T)
            nisa.tensor_tensor(output_sb_view_tile, tile_gamma_mult, beta_sb_view_tile, nl.add)

    for _ in range(num_allocs):
        sbm.pop_heap()
