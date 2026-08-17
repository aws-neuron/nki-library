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

"""Fused optional residual add + RMSNorm + MX quantization kernel optimized for token generation (decoding) phase."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ..mlp.mlp_tkg.projection_mx_constants import _q_width
from ..quantization.fp8_quantize import row_quantization, static_quantization
from ..utils.kernel_helpers import div_ceil
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .norm_tkg_utils import _1B_XPOSE_PSUM_STEP, _MX_SCALE_DTYPE, _UINT8_TP_VIEW_DTYPE, validate_rmsnorm_mx_quantize_tkg


@nki.jit
def rmsnorm_mx_quantize_tkg(
    input: nl.NkiTensor,
    gamma: nl.NkiTensor,
    output: nl.NkiTensor,
    output_quant: Optional[nl.NkiTensor] = None,
    output_scale: Optional[nl.NkiTensor] = None,
    residual: Optional[nl.NkiTensor] = None,
    output_residual: Optional[nl.NkiTensor] = None,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
    hidden_dim_tp: bool = True,
    gate_up_in_scale: Optional[nl.NkiTensor] = None,
    output_input_dequant_scale: Optional[nl.NkiTensor] = None,
    is_row_quant: bool = False,
    is_static_mx: bool = False,
    output_row_dequant_scale: Optional[nl.NkiTensor] = None,
    skip_output_gather: bool = False,
):
    """
    Fused residual add (optional) + RMSNorm + optional MX/static/row quantization for token generation.

    When output_quant=None, performs RMSNorm (+ optional residual add) only, producing bf16 output
    without any quantization. This mode is used by STATIC_MX all-expert to defer quantization
    to the per-expert loop downstream.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size
        H0: Partition dimension (128)
        H1: H // H0

    Args:
        input (nl.NkiTensor): [B, S, H], Input tensor on HBM.
        gamma (nl.NkiTensor): [1, H], RMSNorm scaling weights on HBM.
        output (nl.NkiTensor): [H0, B*S, H1], FP16/BF16 output tensor in SBUF.
        output_quant (Optional[nl.NkiTensor]): [H0, H/512, B*S], FP8x4 quantized output tensor in HBM or SBUF,
            or [B*S, H + H/4] FP8 quantized output tensor in HBM (packed layout).
            When None, quantization is skipped entirely (bf16-only output mode).
        output_scale (Optional[nl.NkiTensor]): [H0, H/512, B*S], MX scale output tensor in SBUF.
            Expected to be None when output_quant is None or when packed layout is used.
        residual (Optional[nl.NkiTensor]): [B, S, H], Optional residual tensor for fused add on HBM.
        output_residual (Optional[nl.NkiTensor]): [B*S, H], Optional output for residual add result on HBM.
        eps (float): Epsilon for numerical stability. Default is 1e-6.
        hidden_actual (Optional[int]): Actual hidden dimension for padded inputs. Default is None (uses H).
        hidden_dim_tp (bool): If True, H dimension view is (H/128, 128). Default is True.
        gate_up_in_scale (Optional[nl.NkiTensor]): [E_L, 1], Per-tensor FP8 dequant scale for STATIC_MX.
            When provided (with output_quant != None), uses software static_quantization instead of
            hardware nisa.quantize_mx. Produces dummy MX scales (127 = 1.0) and FP8 quantized output.
        output_input_dequant_scale (Optional[nl.NkiTensor]): [pmax, 1], Pre-allocated SBUF buffer for
            the broadcast input dequant scale. Required when gate_up_in_scale is provided.
        is_row_quant (bool): If True, uses per-token row_quantization instead of static_quantization.
            Produces per-token dequant scales and dummy MX scales (127 = 1.0). Default is False.
        is_static_mx (bool): If True, indicates STATIC_MX quantization mode. When True with
            gate_up_in_scale provided, uses software static_quantization. Default is False.
        output_row_dequant_scale (Optional[nl.NkiTensor]): [pmax, B*S, 1], Pre-allocated SBUF buffer for
            per-token dequant scales. Required when is_row_quant is True. Populated with per-token
            scales from row_quantization, with LNC sendrecv exchange across cores.

    Returns:
        None. Results are written to pre-allocated output tensors:
            - output: always populated with bf16 RMSNorm result
            - output_quant/output_scale: populated when output_quant is not None
            - output_residual: populated when residual is provided

    Notes:
        - Requires LNC=2 sharding configuration
        - Output tensors must be pre-allocated
        - When output_quant=None, only bf16 output and optional residual are produced
        - When output_scale is not provided (packed layout), QMX output and scales are transposed
          from [H0, num_H512_tiles, shard_size] to [pmax, n_tiles, H] and [pmax, n_tiles, H/4] and concatenated
          into output_quant [B*S, H + H/4] in HBM.

    Pseudocode:
        # For each token tile:
        hidden = input + residual  # if residual provided
        squared = hidden ** 2
        rms = sqrt(mean(squared) + eps)
        normalized = (hidden * gamma) / rms
        output = normalized

        # Quantization (skipped when output_quant=None):
        if output_quant is None:
            pass  # bf16-only mode
        elif is_row_quant:
            output_quant = row_quantize(output)       # per-token FP8 + dequant scales
        elif is_static_mx:
            output_quant = static_quantize(output)    # per-tensor FP8 + dummy MX scales
        else:
            output_quant, output_scale = quantize_mx(output)  # hardware MX quantization

        # Gather results across LNC cores
        # If packed layout: transpose and concatenate [quant | scales] -> [B*S, H + H/4] in HBM
    """

    # Step 1: Configuration, validation
    dims, cfg = validate_rmsnorm_mx_quantize_tkg(
        input_shape=input.shape,
        gamma_shape=gamma.shape,
        output_shape=output.shape,
        output_quant_dtype=output_quant.dtype if output_quant != None else None,
        output_quant_shape=output_quant.shape if output_quant != None else None,
        output_quant_buffer=output_quant.buffer if output_quant != None else None,
        output_scale_shape=output_scale.shape if output_scale != None else None,
        output_scale_buffer=output_scale.buffer if output_scale != None else None,
        output_dtype=output.dtype,
        eps=eps,
        hidden_actual=hidden_actual,
        hidden_dim_tp=hidden_dim_tp,
        has_residual=residual != None,
        residual_shape=residual.shape if residual != None else None,
        has_output_residual=output_residual != None,
        output_residual_shape=output_residual.shape if output_residual != None else None,
        is_static_mx=is_static_mx,
        has_output_input_dequant_scale=output_input_dequant_scale != None,
    )

    # Step 2: Allocate buffers, load weights/constants
    residual_sb = (
        nl.ndarray((dims.H0, cfg.shard_size, dims.H1), dtype=residual.dtype, buffer=nl.sbuf)
        if cfg.is_residual_add
        else None
    )
    gamma_sb = nl.ndarray((dims.H0, dims.H1), dtype=gamma.dtype, buffer=nl.sbuf)
    zero_bias = nl.ndarray((dims.H0, 1), dtype=dims.inter_dtype, buffer=nl.sbuf)
    reduction_const_matrix = nl.ndarray((dims.H0, dims.H0), dtype=dims.inter_dtype, buffer=nl.sbuf)
    eps_loaded = nl.ndarray((dims.H0, 1), dtype=dims.inter_dtype, buffer=nl.sbuf)

    if output_quant == None:
        output_quant_sb = None
        output_scale_sb = None
    elif cfg.is_output_quant_in_sbuf:
        output_quant_sb = output_quant
        output_scale_sb = output_scale
    else:
        output_quant_sb = nl.ndarray(
            (dims.H0, dims.num_H512_tiles, cfg.shard_size), dtype=cfg.qmx_output_dtype, buffer=nl.sbuf
        )
        output_scale_sb = nl.ndarray(
            (dims.H0, dims.num_H512_tiles, cfg.shard_size), dtype=_MX_SCALE_DTYPE, buffer=nl.sbuf
        )
        # nisa.quantize_mx writes only 4 of 32 scale partitions per quadrant; zero-init so the
        # unwritten padding reads as 0 (matching the golden reference) rather than an uninitialized value.
        nisa.memset(output_scale_sb, value=0)

    nisa.memset(zero_bias, value=0.0)
    nisa.memset(reduction_const_matrix, value=1.0)
    nisa.memset(eps_loaded, value=cfg.eps)

    # Load gamma with transpose: (1, H) -> (H) -> (H1, H0) -> (H0, H1)
    gamma_hbm = gamma.flatten_dims(start_dim=0, end_dim=1)
    gamma_hbm_view = gamma_hbm.reshape_dim(dim=0, shape=[dims.H1, dims.H0]).expand_dim(dim=1).expand_dim(dim=1)
    gamma_sb_view = gamma_sb.expand_dim(dim=1).expand_dim(dim=1)
    nisa.dma_transpose(dst=gamma_sb_view, src=gamma_hbm_view)

    # STATIC_MX pre-quantization for E_L=1 case (currently not used): Load and broadcast input dequant scale once before tile loop.
    # When gate_up_in_scale=None for STATIC_MX, quantization is skipped (handled per-expert downstream in expert MLP loop).
    if cfg.is_static_mx and gate_up_in_scale != None:
        nisa.dma_copy(dst=output_input_dequant_scale[:1, :], src=gate_up_in_scale[0:1, :])
        stream_shuffle_broadcast(src=output_input_dequant_scale, dst=output_input_dequant_scale)

    # Reshape HBM views for loading with NkiTensor
    residual_hbm_view = residual.reshape((dims.B * dims.S * dims.H1, dims.H0)) if cfg.is_residual_add else None
    input_hbm_view = input.reshape((dims.B * dims.S * dims.H1, dims.H0))

    # Step 3: Process tiles - Residual Add + RMSNorm + MX Quantization
    for bxs_tile_idx in range(cfg.num_BxS_tiles):
        # Step 3.1: Indexing, allocations
        tile_BxS_start_idx = bxs_tile_idx * cfg.BxS_tile_size
        tile_BxS_offset = cfg.BxS_offset + tile_BxS_start_idx
        hbm_tile_offset = tile_BxS_offset * dims.H1
        tile_BxS_slice_local = nl.ds(tile_BxS_start_idx, cfg.BxS_tile_size)
        tile_BxS_slice_with_offset = nl.ds(tile_BxS_offset, cfg.BxS_tile_size)

        input_tile_sb = nl.ndarray((dims.H0, cfg.BxS_tile_size, dims.H1), dtype=input.dtype, buffer=nl.sbuf)
        output_tile_swizzled = nl.ndarray(
            (dims.H0, dims.num_H512_tiles, cfg.BxS_tile_size, _q_width), dtype=output.dtype, buffer=nl.sbuf
        )

        # Step 3.2: Load input, optionally compute residual
        hidden_sb = _load_hidden_compute_residual(
            dims=dims,
            cfg=cfg,
            input_hbm_view=input_hbm_view,
            residual_hbm_view=residual_hbm_view,
            input_tile_sb=input_tile_sb,
            residual_sb=residual_sb,
            hbm_tile_offset=hbm_tile_offset,
            tile_BxS_start_idx=tile_BxS_start_idx,
            residual_tile_BxS_slice=tile_BxS_slice_local,
        )

        # Step 3.3: Compute RMSNorm
        _compute_rmsnorm(
            dims=dims,
            cfg=cfg,
            hidden_sb=hidden_sb,
            input_tile_sb=input_tile_sb,
            output=output,
            gamma_sb=gamma_sb,
            zero_bias=zero_bias,
            reduction_const_matrix=reduction_const_matrix,
            eps_loaded=eps_loaded,
            output_tile_BxS_slice=tile_BxS_slice_with_offset,
        )

        # Step 3.4: Swizzle and MX quantize (skipped when output_quant=None for STATIC_MX bf16-only output)
        if output_quant != None:
            _swizzle_quantize_mx(
                dims=dims,
                cfg=cfg,
                output=output,
                output_tile_swizzled=output_tile_swizzled,
                output_quant=output_quant,
                output_quant_sb=output_quant_sb,
                output_scale=output_scale,
                output_scale_sb=output_scale_sb,
                output_input_dequant_scale=output_input_dequant_scale,
                output_tile_BxS_slice=tile_BxS_slice_with_offset,
                quant_tile_BxS_slice=tile_BxS_slice_with_offset
                if cfg.is_output_quant_in_sbuf
                else tile_BxS_slice_local,
                is_row_quant=is_row_quant,
                output_row_dequant_scale=output_row_dequant_scale,
                tile_BxS_offset=tile_BxS_offset,
            )

    # Step 4: Gather unquantized output across LNC cores
    # Skip when downstream router_topk also shards on BxS — each NC only needs its own shard.
    if cfg.do_shard and not skip_output_gather:
        send_to_rank = recv_from_rank = 1 - cfg.shard_id
        # Exchange the non-quantized output so both NCs have complete data for router_topk
        nisa.sendrecv(
            send_to_rank=send_to_rank,
            recv_from_rank=recv_from_rank,
            src=output[0 : dims.H0, nl.ds(cfg.BxS_offset, cfg.shard_size), 0 : dims.H1],
            dst=output[0 : dims.H0, nl.ds((1 - cfg.shard_id) * cfg.shard_size, cfg.shard_size), 0 : dims.H1],
            pipe_id=0,
        )

    # Step 5: Gather output_quant and output_scale across NCs
    # When output is in SBUF, gather with sendrecv (skip when output_quant=None)
    if output_quant != None and cfg.is_output_quant_in_sbuf and cfg.do_shard:
        send_to_rank = recv_from_rank = 1 - cfg.shard_id
        nisa.sendrecv(
            send_to_rank=send_to_rank,
            recv_from_rank=recv_from_rank,
            src=output_quant[0 : dims.H0, 0 : dims.num_H512_tiles, nl.ds(cfg.BxS_offset, cfg.shard_size)],
            dst=output_quant[
                0 : dims.H0, 0 : dims.num_H512_tiles, nl.ds((1 - cfg.shard_id) * cfg.shard_size, cfg.shard_size)
            ],
            pipe_id=1,
        )
        nisa.sendrecv(
            send_to_rank=send_to_rank,
            recv_from_rank=recv_from_rank,
            src=output_scale[0 : dims.H0, 0 : dims.num_H512_tiles, nl.ds(cfg.BxS_offset, cfg.shard_size)],
            dst=output_scale[
                0 : dims.H0, 0 : dims.num_H512_tiles, nl.ds((1 - cfg.shard_id) * cfg.shard_size, cfg.shard_size)
            ],
            pipe_id=2,
        )
        # ROW_MX: exchange per-token dequant scales across LNC cores
        if is_row_quant and output_row_dequant_scale is not None:
            dequant_flat = output_row_dequant_scale.reshape((dims.pmax, dims.BxS))
            nisa.sendrecv(
                send_to_rank=send_to_rank,
                recv_from_rank=recv_from_rank,
                src=dequant_flat[:, nl.ds(cfg.BxS_offset, cfg.shard_size)],
                dst=dequant_flat[:, nl.ds((1 - cfg.shard_id) * cfg.shard_size, cfg.shard_size)],
                pipe_id=0,
            )

    # When output is packed in HBM, transpose and concatenate [quant | scales] -> [B*S, H + H/4] for A2A-v
    elif output_quant != None and cfg.is_output_quant_packed:
        n_BxS_tiles = div_ceil(cfg.shard_size, dims.pmax)
        H_packed = dims.num_H512_tiles * dims.H0

        # Reinterpret f8x4 -> f32 for transpose, since PE does not allow transpose with f8x4
        output_quant_sb = output_quant_sb.view(nl.float32)
        output_quant_transposed_sb = nl.ndarray((dims.pmax, n_BxS_tiles, H_packed), dtype=nl.float32, buffer=nl.sbuf)
        _transpose_qmx_output(dims=dims, cfg=cfg, src_sb=output_quant_sb, dst_sb=output_quant_transposed_sb)

        # Reinterpret f32 -> f8 (unpacks x4 to individual f8 elements), then spill quant data to [B*S, 0:H]
        output_quant_transposed_view = output_quant_transposed_sb.view(cfg.output_quant_dtype)
        _spill_tiled_sb_to_hbm(
            src_sb=output_quant_transposed_view,
            dst_hbm=output_quant,
            shard_size=cfg.shard_size,
            BxS_offset=cfg.BxS_offset,
            total_free=dims.H,
            dst_free_offset=0,
        )

        # Reinterpret u8 -> f8 for transpose, since PE does not allow transpose with u8
        output_scale_sb = output_scale_sb.view(_UINT8_TP_VIEW_DTYPE)
        output_scale_transposed_sb = nl.ndarray(
            (dims.pmax, n_BxS_tiles, H_packed), dtype=_UINT8_TP_VIEW_DTYPE, buffer=nl.sbuf
        )
        _transpose_qmx_output(dims=dims, cfg=cfg, src_sb=output_scale_sb, dst_sb=output_scale_transposed_sb)

        # Reinterpret f8 -> f8 used by packed output_quant, then spill scales to [B*S, H:H+H/4]
        output_scale_transposed_sb = output_scale_transposed_sb.view(cfg.output_quant_dtype)
        _spill_tiled_sb_to_hbm(
            src_sb=output_scale_transposed_sb,
            dst_hbm=output_quant,
            shard_size=cfg.shard_size,
            BxS_offset=cfg.BxS_offset,
            total_free=H_packed,
            dst_free_offset=dims.H,
        )

    # When output is untransposed in HBM, gather output_quant and output_scale separately in HBM
    elif output_quant != None:
        nisa.dma_copy(
            src=output_quant_sb[0 : dims.H0, 0 : dims.num_H512_tiles, 0 : cfg.shard_size],
            dst=output_quant[0 : dims.H0, 0 : dims.num_H512_tiles, nl.ds(cfg.BxS_offset, cfg.shard_size)],
        )
        nisa.dma_copy(
            src=output_scale_sb[0 : dims.H0, 0 : dims.num_H512_tiles, 0 : cfg.shard_size],
            dst=output_scale[0 : dims.H0, 0 : dims.num_H512_tiles, nl.ds(cfg.BxS_offset, cfg.shard_size)],
        )

    # Step 6: Spill residual result to HBM
    if cfg.is_residual_add:
        residual_transposed_sb = _transpose_residual(dims=dims, cfg=cfg, residual_sb=residual_sb)
        _spill_tiled_sb_to_hbm(
            src_sb=residual_transposed_sb,
            dst_hbm=output_residual,
            shard_size=cfg.shard_size,
            BxS_offset=cfg.BxS_offset,
            total_free=dims.H,
        )

    # Return
    outputs = [output, output_quant]
    if not cfg.is_output_quant_packed:
        outputs.append(output_scale)
    if cfg.is_residual_add:
        outputs.append(residual)

    return outputs


def _load_hidden_compute_residual(
    dims,
    cfg,
    input_hbm_view,
    residual_hbm_view,
    input_tile_sb,
    residual_sb,
    hbm_tile_offset,
    tile_BxS_start_idx,
    residual_tile_BxS_slice,
):
    """Load input (+ optional residual) from HBM with DMA transpose and compute residual add."""
    input_src_view = (
        input_hbm_view.slice(dim=0, start=hbm_tile_offset, end=hbm_tile_offset + cfg.BxS_tile_size * dims.H1)
        .expand_dim(dim=1)
        .expand_dim(dim=1)
    )
    input_dst_view = input_tile_sb.flatten_dims(start_dim=1, end_dim=2).expand_dim(dim=1).expand_dim(dim=1)

    if cfg.is_residual_add:
        residual_src_view = (
            residual_hbm_view.slice(dim=0, start=hbm_tile_offset, end=hbm_tile_offset + cfg.BxS_tile_size * dims.H1)
            .expand_dim(dim=1)
            .expand_dim(dim=1)
        )
        residual_dst_view = (
            residual_sb.slice(dim=1, start=tile_BxS_start_idx, end=tile_BxS_start_idx + cfg.BxS_tile_size)
            .flatten_dims(start_dim=1, end_dim=2)
            .expand_dim(dim=1)
            .expand_dim(dim=1)
        )

        # Transpose load residual and input
        nisa.dma_transpose(dst=residual_dst_view, src=residual_src_view)
        nisa.dma_transpose(dst=input_dst_view, src=input_src_view)

        # Residual add: hidden = input + residual
        nisa.tensor_tensor(
            residual_sb[:, residual_tile_BxS_slice, :],
            input_tile_sb,
            residual_sb[:, residual_tile_BxS_slice, :],
            nl.add,
        )
        hidden_sb = residual_sb[:, residual_tile_BxS_slice, :]
    else:
        # Transpose load input
        nisa.dma_transpose(dst=input_dst_view, src=input_src_view)
        hidden_sb = input_tile_sb

    return hidden_sb


def _compute_rmsnorm(
    dims,
    cfg,
    hidden_sb,
    input_tile_sb,
    output,
    gamma_sb,
    zero_bias,
    reduction_const_matrix,
    eps_loaded,
    output_tile_BxS_slice,
):
    """Compute RMSNorm on tile: output = (hidden * gamma) / RMS(hidden)."""

    # Allocations
    square = nl.ndarray((dims.H0, cfg.BxS_tile_size, dims.H1), dtype=dims.inter_dtype, buffer=nl.sbuf)
    reduced = nl.ndarray((dims.H0, cfg.BxS_tile_size), dtype=dims.inter_dtype, buffer=nl.sbuf)
    final_reduced = nl.ndarray((dims.H0, cfg.BxS_tile_size), dtype=nl.float32, buffer=nl.psum)
    sqrt = nl.ndarray((dims.H0, cfg.BxS_tile_size), dtype=dims.inter_dtype, buffer=nl.sbuf)

    # Input ^2
    nisa.activation(dst=square, op=nl.square, data=hidden_sb, bias=zero_bias)

    # Input * gamma (broadcast gamma across BxS_tile_size)
    gamma_sb_view = gamma_sb.expand_dim(dim=1).broadcast(dim=1, size=cfg.BxS_tile_size)
    nisa.tensor_tensor(input_tile_sb, hidden_sb, gamma_sb_view, nl.multiply)

    # Reduce squared input along H1 dimension (last free dimension)
    nisa.tensor_reduce(dst=reduced, op=nl.add, data=square, axis=2)

    # Complete reduction across H0 dimension using matmul
    nisa.nc_matmul(dst=final_reduced, stationary=reduction_const_matrix, moving=reduced)

    # Compute 1/sqrt(mean(x^2) + eps)
    nisa.activation(dst=sqrt, op=nl.rsqrt, data=final_reduced, scale=(1.0 / dims.hidden_actual), bias=eps_loaded)

    # Compute input * 1/RMS(input)
    sqrt_view = sqrt.expand_dim(dim=2).broadcast(dim=2, size=dims.H1)
    nisa.tensor_tensor(output[:, output_tile_BxS_slice, :], input_tile_sb, sqrt_view, nl.multiply)


def _swizzle_quantize_mx(
    dims,
    cfg,
    output,
    output_tile_swizzled,
    output_quant,
    output_quant_sb,
    output_scale,
    output_scale_sb,
    output_input_dequant_scale,
    output_tile_BxS_slice,
    quant_tile_BxS_slice,
    is_row_quant,
    output_row_dequant_scale,
    tile_BxS_offset,
):
    """Swizzle normalized output and quantize to MX format."""

    # Swizzle from [H0, BxS_tile_size, H1] to [H0, num_H512_tiles, BxS_tile_size, _q_width]
    if is_row_quant:
        # ROW_QUANT: quantize first, then swizzle the quantized data
        output_tile_fresh = nl.ndarray((dims.H0, cfg.BxS_tile_size, dims.H1), dtype=output.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=output_tile_fresh, src=output[0 : dims.H0, output_tile_BxS_slice, 0 : dims.H1])
        quantized_tile_bf16, tile_dequant_scale = row_quantization(output_tile_fresh)
        nisa.tensor_copy(
            dst=output_row_dequant_scale[0 : dims.H0, tile_BxS_offset : tile_BxS_offset + cfg.BxS_tile_size, 0:1],
            src=tile_dequant_scale,
        )
        for h512_tile_idx in nl.affine_range(dims.num_H512_tiles):
            for q_idx in nl.affine_range(_q_width):
                nisa.tensor_copy(
                    dst=output_tile_swizzled[0 : dims.H0, h512_tile_idx, 0 : cfg.BxS_tile_size, q_idx],
                    src=quantized_tile_bf16[
                        0 : dims.H0, 0 : cfg.BxS_tile_size, q_idx * dims.num_H512_tiles + h512_tile_idx
                    ],
                )
    else:
        # STATIC_MX / HW_MX: swizzle directly from normalized output
        for h512_tile_idx in nl.affine_range(dims.num_H512_tiles):
            for q_idx in nl.affine_range(_q_width):
                nisa.tensor_copy(
                    dst=output_tile_swizzled[0 : dims.H0, h512_tile_idx, 0 : cfg.BxS_tile_size, q_idx],
                    src=output[0 : dims.H0, output_tile_BxS_slice, q_idx * dims.num_H512_tiles + h512_tile_idx],
                )

    # Quantize
    if is_row_quant:
        # ROW_QUANT: cast swizzled bf16 → fp8 → fp8_x4, dummy 127 MX scales
        total_free = dims.num_H512_tiles * cfg.BxS_tile_size * _q_width
        swizzled_flat = output_tile_swizzled.reshape((dims.H0, total_free))
        quantized_fp8 = nl.ndarray(swizzled_flat.shape, dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        nisa.tensor_copy(dst=quantized_fp8, src=swizzled_flat)
        total_x4 = dims.num_H512_tiles * cfg.BxS_tile_size
        temp_quant = nl.ndarray((dims.H0, total_x4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=temp_quant.view(nl.uint32),
            src=quantized_fp8.view(nl.uint32),
            engine=nisa.vector_engine,
        )
        temp_quant_3d = temp_quant.reshape((dims.H0, dims.num_H512_tiles, cfg.BxS_tile_size))
        for h512_tile_idx in nl.sequential_range(dims.num_H512_tiles):
            nisa.tensor_copy(
                dst=output_quant[0 : dims.H0, h512_tile_idx : h512_tile_idx + 1, quant_tile_BxS_slice],
                src=temp_quant_3d[:, h512_tile_idx : h512_tile_idx + 1, :],
            )
            nisa.memset(
                dst=output_scale[0 : dims.H0, h512_tile_idx : h512_tile_idx + 1, quant_tile_BxS_slice], value=127
            )
    elif cfg.is_static_mx:
        # STATIC_MX: software static quantization + dummy 127 MX scales
        total_free = dims.num_H512_tiles * cfg.BxS_tile_size * _q_width
        swizzled_flat = output_tile_swizzled.reshape((dims.H0, total_free))
        quantized_flat, _ = static_quantization(swizzled_flat, output_input_dequant_scale)
        # Cast bf16 → fp8, then reinterpret as fp8_x4
        quantized_fp8 = nl.ndarray(quantized_flat.shape, dtype=nl.float8_e4m3fn, buffer=nl.sbuf)
        nisa.tensor_copy(dst=quantized_fp8, src=quantized_flat)
        total_x4 = dims.num_H512_tiles * cfg.BxS_tile_size
        temp_quant = nl.ndarray((dims.H0, total_x4), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        nisa.tensor_copy(dst=temp_quant.view(nl.uint32), src=quantized_fp8.view(nl.uint32), engine=nisa.vector_engine)
        # Copy to output per H512 tile
        temp_quant_3d = temp_quant.reshape((dims.H0, dims.num_H512_tiles, cfg.BxS_tile_size))
        for h512_tile_idx in nl.sequential_range(dims.num_H512_tiles):
            nisa.tensor_copy(
                dst=output_quant[0 : dims.H0, h512_tile_idx : h512_tile_idx + 1, quant_tile_BxS_slice],
                src=temp_quant_3d[:, h512_tile_idx : h512_tile_idx + 1, :],
            )
            nisa.memset(
                dst=output_scale[0 : dims.H0, h512_tile_idx : h512_tile_idx + 1, quant_tile_BxS_slice], value=127
            )
    else:
        # Hardware MX quantization
        # NOTE: since QMX cannot have strided output AP, QMX is called on each contiguous region of swizzled tile
        for h512_tile_idx in nl.sequential_range(dims.num_H512_tiles):
            nisa.quantize_mx(
                src=output_tile_swizzled[
                    0 : dims.H0, h512_tile_idx : h512_tile_idx + 1, 0 : cfg.BxS_tile_size, 0:_q_width
                ],
                dst=output_quant_sb[0 : dims.H0, h512_tile_idx : h512_tile_idx + 1, quant_tile_BxS_slice],
                dst_scale=output_scale_sb[0 : dims.H0, h512_tile_idx : h512_tile_idx + 1, quant_tile_BxS_slice],
            )


def _transpose_qmx_output(dims, cfg, src_sb, dst_sb):
    """Transpose from [H0, num_H512_tiles, shard_size] to [pmax, shard_size/pmax, num_H512_tiles * H0].

    Determines PSUM sizing and strided transpose based on src dtype:
    - 4-byte types (f32): non-strided PSUM, tile_H = psum_fmax
    - 1-byte types (f8): strided PSUM with _1B_XPOSE_PSUM_STEP, tile_H = psum_fmax * 2
    """

    # Tiling strategy
    n_BxS_tiles = div_ceil(cfg.shard_size, dims.pmax)
    src_dtype = src_sb.dtype
    is_1byte = src_dtype == _UINT8_TP_VIEW_DTYPE
    transpose_tile_H = dims.psum_fmax * 2 if is_1byte else dims.psum_fmax
    num_H_tiles_per_psum = transpose_tile_H // dims.H0
    num_H_tile_groups = div_ceil(dims.num_H512_tiles, num_H_tiles_per_psum)

    # Transpose loop
    for bxs_tile_idx in nl.affine_range(n_BxS_tiles):
        tile_tokens = min(dims.pmax, cfg.shard_size - bxs_tile_idx * dims.pmax)
        bxs_start = bxs_tile_idx * dims.pmax
        for H_group_idx in nl.affine_range(num_H_tile_groups):
            H_tiles_in_group = min(num_H_tiles_per_psum, dims.num_H512_tiles - H_group_idx * num_H_tiles_per_psum)
            psum_tile_H = H_tiles_in_group * dims.H0
            psum_shape = (
                (dims.pmax, transpose_tile_H, _1B_XPOSE_PSUM_STEP) if is_1byte else (dims.pmax, transpose_tile_H)
            )
            transposed_psum = nl.ndarray(psum_shape, dtype=src_dtype, buffer=nl.psum)
            for H_tile_grp_idx in nl.affine_range(H_tiles_in_group):
                H_tile_idx = H_group_idx * num_H_tiles_per_psum + H_tile_grp_idx
                dst_tile = (
                    transposed_psum[0:tile_tokens, nl.ds(H_tile_grp_idx * dims.H0, dims.H0), 0]
                    if is_1byte
                    else transposed_psum[0:tile_tokens, nl.ds(H_tile_grp_idx * dims.H0, dims.H0)]
                )
                nisa.nc_transpose(
                    data=src_sb[0 : dims.H0, H_tile_idx, nl.ds(bxs_start, tile_tokens)],
                    dst=dst_tile,
                )
            dst_H_offset = H_group_idx * num_H_tiles_per_psum * dims.H0
            src_tile = (
                transposed_psum[0:tile_tokens, 0:psum_tile_H, 0]
                if is_1byte
                else transposed_psum[0:tile_tokens, 0:psum_tile_H]
            )
            nisa.tensor_copy(
                dst=dst_sb[0:tile_tokens, bxs_tile_idx, nl.ds(dst_H_offset, psum_tile_H)],
                src=src_tile,
                engine=nisa.vector_engine if H_group_idx % 2 == 0 else nisa.scalar_engine,
            )


def _transpose_residual(dims, cfg, residual_sb):
    """
    PE transpose residual_sb from [H0, shard_size, H1] to [pmax, num_pmax_token_tiles, H].

    residual_sb actual layout: (H0, shard_size * H1)
    To access [0:H0, pmax_token_tile_idx*pmax : pmax_token_tile_idx*pmax+tile_tokens_actual, H1_idx]
    as (H0, tile_tokens_actual): element [p, t, h1] is at position p * (shard_size * H1) + t * H1 + h1
    AP: [[shard_size*H1, H0], [H1, tile_tokens_actual]], offset = pmax_token_tile_idx * pmax * H1 + H1_idx
    """

    # Tiling strategy
    num_pmax_token_tiles = div_ceil(cfg.shard_size, dims.pmax)
    residual_dtype = residual_sb.dtype
    is_16bit = residual_dtype in [nl.float16, nl.bfloat16]
    transpose_tile_H = dims.psum_fmax * 2 if is_16bit else dims.psum_fmax
    num_transpose_tiles = dims.num_H512_tiles // 2 if is_16bit else dims.num_H512_tiles
    num_H1_per_transpose_tile = transpose_tile_H // dims.H0

    residual_transposed_sb = nl.ndarray((dims.pmax, num_pmax_token_tiles, dims.H), dtype=residual_dtype, buffer=nl.sbuf)
    for pmax_token_tile_idx in nl.affine_range(num_pmax_token_tiles):
        tile_tokens_actual = min(dims.pmax, cfg.shard_size - pmax_token_tile_idx * dims.pmax)
        for transpose_tile_idx in nl.affine_range(num_transpose_tiles):
            residual_transposed_tile_psum = nl.ndarray(
                (dims.pmax, transpose_tile_H), dtype=residual_dtype, buffer=nl.psum
            )
            for h1_in_tile_idx in nl.affine_range(num_H1_per_transpose_tile):
                H1_idx = transpose_tile_idx * num_H1_per_transpose_tile + h1_in_tile_idx
                nisa.nc_transpose(
                    dst=residual_transposed_tile_psum[0:tile_tokens_actual, nl.ds(h1_in_tile_idx * dims.H0, dims.H0)],
                    data=residual_sb[
                        :,
                        pmax_token_tile_idx * dims.pmax : pmax_token_tile_idx * dims.pmax + tile_tokens_actual,
                        H1_idx,
                    ],
                )
            nisa.tensor_copy(
                dst=residual_transposed_sb[
                    0:tile_tokens_actual,
                    pmax_token_tile_idx,
                    nl.ds(transpose_tile_idx * transpose_tile_H, transpose_tile_H),
                ],
                src=residual_transposed_tile_psum[0:tile_tokens_actual, 0:transpose_tile_H],
                engine=nisa.vector_engine if transpose_tile_idx % 2 == 0 else nisa.scalar_engine,
            )

    return residual_transposed_sb


def _spill_tiled_sb_to_hbm(src_sb, dst_hbm, shard_size, BxS_offset, total_free, dst_free_offset=0):
    """DMA spill from tiled SBUF [pmax, n_tiles, total_free] to a region of HBM [..., dst_total_free].

    Args:
        src_sb: Source SBUF tensor or NkiTensor of shape [pmax, n_tiles, total_free].
        dst_hbm: Destination HBM tensor or NkiTensor.
        shard_size: Total number of token rows to spill.
        BxS_offset: Row offset in dst_hbm.
        total_free: Number of elements in the free dimension per tile.
        dst_free_offset: Starting offset along the free (last) dimension of dst_hbm.
    """
    pmax = nl.tile_size.pmax
    num_full_tiles = shard_size // pmax
    remainder = shard_size % pmax
    src_view = src_sb
    dst_view = dst_hbm
    dst_total_free = dst_view.shape[-1]

    if remainder == 0 and num_full_tiles > 1:
        # All tiles are full and multiple tiles - one vectorized DMA.
        # Reshape dst region [N*pmax, total_free] -> [N, pmax, total_free] -> [pmax, N, total_free]
        src_3d = src_view.slice(dim=1, start=0, end=num_full_tiles)
        dst_tile = (
            dst_view.slice(dim=0, start=BxS_offset, end=BxS_offset + shard_size)
            .slice(dim=1, start=dst_free_offset, end=dst_free_offset + total_free)
            .reshape_dim(dim=0, shape=(num_full_tiles, pmax))
            .permute((1, 0, 2))
        )
        nisa.dma_copy(
            src=src_3d,
            dst=dst_tile,
        )
    elif remainder == 0:
        # Single full tile
        src_tile = src_view.select(dim=1, index=0)
        dst_tile = dst_view.slice(dim=0, start=BxS_offset, end=BxS_offset + pmax).slice(
            dim=1, start=dst_free_offset, end=dst_free_offset + total_free
        )
        nisa.dma_copy(
            src=src_tile,
            dst=dst_tile,
        )
    else:
        # Has partial last tile
        if num_full_tiles > 1:
            # Vectorized DMA for full tiles
            src_3d = src_view.slice(dim=1, start=0, end=num_full_tiles)
            full_size = num_full_tiles * pmax
            dst_tile = (
                dst_view.slice(dim=0, start=BxS_offset, end=BxS_offset + full_size)
                .slice(dim=1, start=dst_free_offset, end=dst_free_offset + total_free)
                .reshape_dim(dim=0, shape=(num_full_tiles, pmax))
                .permute((1, 0, 2))
            )
            nisa.dma_copy(
                src=src_3d,
                dst=dst_tile,
            )
        elif num_full_tiles == 1:
            # Single full tile
            src_tile = src_view.select(dim=1, index=0)
            dst_tile = dst_view.slice(dim=0, start=BxS_offset, end=BxS_offset + pmax).slice(
                dim=1, start=dst_free_offset, end=dst_free_offset + total_free
            )
            nisa.dma_copy(
                src=src_tile,
                dst=dst_tile,
            )
        # Single DMA for partial last tile
        partial_offset = num_full_tiles * pmax
        src_partial = src_view.slice(dim=0, start=0, end=remainder).select(dim=1, index=num_full_tiles)
        dst_partial = dst_view.slice(
            dim=0, start=BxS_offset + partial_offset, end=BxS_offset + partial_offset + remainder
        ).slice(dim=1, start=dst_free_offset, end=dst_free_offset + total_free)
        nisa.dma_copy(
            src=src_partial,
            dst=dst_partial,
        )
