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

"""Utility functions for MLP TKG kernel including input loading, normalization, and output transpose."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...subkernels.layernorm_tkg import layernorm_tkg
from ...subkernels.norm_tkg_utils import _CONTIGUOUS_LOAD_H_THRESHOLD, contiguous_load_transpose
from ...subkernels.rmsnorm_tkg import rmsnorm_tkg, rmsnorm_tkg_th
from ...utils.allocator import SbufManager
from ...utils.common_types import HiddenLayout, NormType
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from .mlp_parameters import (
    _Q_WIDTH,
    MLPParameters,
    mlpp_has_normalization,
    mlpp_has_rms_normalization,
)
from .mlp_tkg_constants import MLPTKGConstantsDimensionSizes

_DGE_MODE_UNKNOWN = 0  # Compiler decides best DMA mode internally
_DGE_MODE_NONE = 3  # Use STATIC DMA mode
_DGE_MODE_SWDGE = 1  # Use SWDGE mode (required inside dynamic while loops)


def convert_params_to_views(params: MLPParameters):
    """Convert all weight, bias, fused-add, and quantization scale tensors in params to NkiTensor."""
    params.hidden_tensor = params.hidden_tensor
    if params.gate_proj_weights_tensor is not None:
        params.gate_proj_weights_tensor = params.gate_proj_weights_tensor
    params.up_proj_weights_tensor = params.up_proj_weights_tensor
    params.down_proj_weights_tensor = params.down_proj_weights_tensor
    params.bias_params.convert_to_view()
    params.fused_add_params.convert_to_view()
    if params.quant_params.is_quant():
        params.quant_params.convert_to_view()


def prepare_gate_up_bias_and_scale(
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
):
    """
    Pre-shard gate/up bias and w_scale tensors based on column tiling mode and quantization type.

    Output shapes by mode:
        Bias:
            Column tiling:  [1, I]   (2D, sliced per I tile by caller)
            LHS/RHS swap:   [I]      (1D, squeezed here, sliced per I tile by caller)

        W_scale (static quantization):
            Column tiling:  [T, 1]   (no sharding needed, loaded as-is)
            LHS/RHS swap:   [128, 1] (no sharding needed, loaded as-is)

        W_scale (row quantization):
            Column tiling:  [128, I]  (passed through as-is, I-sliced per tile by caller)
            LHS/RHS swap:   [128, I]  (passed through as-is, I-sliced per tile by caller)

    Args:
        params (MLPParameters): MLP parameters with bias and quant scale tensors.
        dims (MLPTKGConstantsDimensionSizes): Dimension and sharding metadata.

    Returns:
        tuple: (gate_b, up_b, gate_w_scale, up_w_scale) — pre-shaped NkiTensors or None.
    """
    gate_b = params.bias_params.gate_proj_bias_tensor
    up_b = params.bias_params.up_proj_bias_tensor
    gate_w_scale = params.quant_params.gate_w_scale
    up_w_scale = params.quant_params.up_w_scale

    # Pre-squeeze bias for LHS/RHS swap path (needs 1D)
    if not params.use_tkg_gate_up_proj_column_tiling:
        if gate_b is not None and gate_b.ndim > 1:
            gate_b = gate_b.squeeze_dim(dim=0)
        if up_b is not None and up_b.ndim > 1:
            up_b = up_b.squeeze_dim(dim=0)

    return gate_b, up_b, gate_w_scale, up_w_scale


def prepare_down_bias_and_scale(
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
):
    """
    Pre-shard down bias and w_scale tensors based on column tiling mode and quantization type.

    Output shapes by mode:
        Bias:
            Column tiling:  [T, H_per_shard]   (2D, H-sharded and broadcast T here)
            LHS/RHS swap:   [H0, H1_shard]     (H-sharded, squeezed, reshaped here)

        W_scale (static quantization):
            Column tiling:  [T, 1]            (sliced dim=0 to T here)
            LHS/RHS swap:   [H0, 1]           (sliced dim=0 to H0 here)

        W_scale (row quantization):
            Column tiling:  [T, H_per_shard]   (slice dim=0 to T, H-shard dim=1 here)
            LHS/RHS swap:   [H0, H1_shard]     (select row 0, H-shard, reshape here)

    Args:
        params (MLPParameters): MLP parameters with bias and quant scale tensors.
        dims (MLPTKGConstantsDimensionSizes): Dimension and sharding metadata.

    Returns:
        tuple: (down_b, down_w_scale) — pre-shaped NkiTensors or None.
    """
    down_b = params.bias_params.down_proj_bias_tensor
    down_w_scale = params.quant_params.down_w_scale

    # Pre-shard and reshape bias on H dimension
    if down_b is not None:
        if params.use_tkg_down_proj_column_tiling:
            # [1, H] → [1, H_per_shard] → broadcast to [T, H_per_shard]
            down_b = down_b.slice(
                dim=1, start=dims.H1_offset * dims.H0, end=dims.H1_offset * dims.H0 + dims.H_per_shard
            ).broadcast(dim=0, size=dims.T)
        else:
            # [1, H] or [H] → [H_per_shard] → [H0, H1_shard]
            bias_dim = 0 if down_b.ndim == 1 else 1
            down_b = down_b.slice(
                dim=bias_dim,
                start=dims.H1_offset * dims.H0,
                end=dims.H1_offset * dims.H0 + dims.H_per_shard,
            )
            if down_b.ndim > 1:
                down_b = down_b.squeeze_dim(dim=0)
            down_b = down_b.reshape_dim(dim=0, shape=(dims.H0, dims.H1_shard))

    # Pre-shape w_scale based on quantization type
    # w_scale input shape: [128, H] for row quant, [128, 1] or [T, 1] for static quant
    if down_w_scale is not None:
        if params.quant_params.is_quant_static():
            par_dims = dims.T if params.use_tkg_down_proj_column_tiling else dims.H0
            down_w_scale = down_w_scale.slice(dim=0, start=0, end=par_dims)
        elif params.quant_params.is_quant_row():
            if params.use_tkg_down_proj_column_tiling:
                # [128, H] → [T, H_per_shard]
                down_w_scale = down_w_scale.slice(dim=0, start=0, end=dims.T).slice(
                    dim=1,
                    start=dims.H1_offset * dims.H0,
                    end=dims.H1_offset * dims.H0 + dims.H_per_shard,
                )
            else:
                # [128, H] → [1, H] → squeeze → [H] → [H_per_shard] → [H0, H1_shard]
                down_w_scale = (
                    down_w_scale.slice(dim=0, start=0, end=1)
                    .squeeze_dim(dim=0)
                    .slice(dim=0, start=dims.H1_offset * dims.H0, end=dims.H1_offset * dims.H0 + dims.H_per_shard)
                    .reshape_dim(dim=0, shape=(dims.H0, dims.H1_shard))
                )

    return down_b, down_w_scale


def _layout_adapter_hbm(src: nl.NkiTensor, n_prgs: int, prg_id: int):
    """
    Load and transpose input tensor from HBM to SBUF with swizzled layout.

    Performs the following transformations:
    1. Input layout in HBM: [T, H] with internally shuffled layout [T, 4_H, H/512, 16_H, 8_H]
    2. Load input to SBUF: [4_T * 4_H (P), T/4, H/512, 16_H * 8_H]
    3. Perform T/4 * H/512 transpose operations to swap outermost and innermost dims
    4. Obtain swizzle layout: [16_H * 8_H(P), H/512, T/4, 4_T * 4_H]

    Args:
        src (nl.NkiTensor): [T, H], 5D tensor in HBM with internally shuffled layout [T, 4_H, H/512, 16_H, 8_H].
        n_prgs (int): Number of programs.
        prg_id (int): Program ID.

    Returns:
        result (nl.NkiTensor): [16_H * 8_H(P), H/512, ceil(T/4) * 4, 4_H], 4D tensor in SBUF with swizzled layout.
    """
    _q_width = _Q_WIDTH
    _pmax = nl.tile_size.pmax

    # Check for shape
    kernel_assert(len(src.shape) == 2, f"expect input to be of the shape [T, H], got {src.shape}")
    T, H = src.shape
    kernel_assert(H % 512 == 0, f"Expect H to be a multiple of 512, got {H}")
    H_div_512 = H // 512 // n_prgs
    T_div_4 = div_ceil(T, 4)

    # [16_H * 8_H(P), H/512, T/4, 4_T * 4_H]
    result = nl.ndarray((_pmax, H_div_512, T_div_4, 4 * _q_width), dtype=src.dtype, buffer=nl.sbuf)
    # [T, 4_H, H/512, 16_H * 8_H]
    src = src.reshape((T, _q_width, H_div_512 * n_prgs, 16 * 8))
    # [4_T * 4_H (P), T/4, H/512, 16_H * 8_H]
    src_sbuf = nl.ndarray((4 * _q_width, T_div_4, H_div_512, 16 * 8), dtype=src.dtype, buffer=nl.sbuf)
    nisa.memset(dst=src_sbuf, value=0.0)

    """
    Load [4_T * 4_H (P), T/4, H/512, 16_H * 8_H]@SBUF
    from [T, 4_H, H/512, 16_H, 8_H]@HBM.
    """
    for T_div_4_idx in range(T_div_4):
        for H_4_idx in range(4):
            for token_grp_idx in range(4):
                actual_token_idx = token_grp_idx + T_div_4_idx * 4
                if actual_token_idx < T:
                    nisa.dma_copy(
                        dst=src_sbuf[
                            token_grp_idx * 4 + H_4_idx : token_grp_idx * 4 + H_4_idx + 1,
                            T_div_4_idx : T_div_4_idx + 1,
                            0:H_div_512,
                            0 : 16 * 8,
                        ],
                        src=src[
                            actual_token_idx : actual_token_idx + 1,
                            H_4_idx : H_4_idx + 1,
                            prg_id * H_div_512 : (prg_id + 1) * H_div_512,
                            0 : 16 * 8,
                        ],
                    )

    for T_div_4_idx in range(T_div_4):
        for H_div_512_idx in range(H_div_512):
            # transpose [4_T * 4_H, 16_H*8_H] -> [16_H*8_H, 4_T * 4_H]
            tile_transposed = nl.ndarray((_pmax, 4 * _q_width), buffer=nl.psum, dtype=src_sbuf.dtype)
            nisa.nc_transpose(
                dst=tile_transposed, data=src_sbuf[0 : (4 * _q_width), T_div_4_idx, H_div_512_idx, 0:_pmax]
            )
            nisa.tensor_copy(dst=result[0:_pmax, H_div_512_idx, T_div_4_idx, 0 : (4 * _q_width)], src=tile_transposed)

    T_padded = T_div_4 * 4
    return result.reshape((_pmax, H_div_512, T_padded, _q_width))


def _layout_adapter_sb(src: nl.NkiTensor, n_prgs: int, prg_id: int):
    """
    SBUF version of the layout adapter.

    Args:
        src (nl.NkiTensor): [_pmax, T, _q_width, n_H512_tiles], Input tensor in SBUF.
        n_prgs (int): Number of programs.
        prg_id (int): Program ID.

    Returns:
        shfl_sb (nl.NkiTensor): [_pmax, n_H512_tile_sharded, ceil_div(T, 4) * 4, _q_width], Shuffled tensor in SBUF.
    """
    _q_width = _Q_WIDTH
    _pmax = nl.tile_size.pmax

    kernel_assert(len(src.shape) == 3, f"expect input to have shape [_pmax, T, H//_pmax], got {src.shape}")
    P, T, H_div_P = src.shape
    kernel_assert(
        P == _pmax and H_div_P % _q_width == 0,
        f'Expect input SBUF shape to be ({_pmax}, T, <multiple-of-{_q_width}>), got {src.shape}',
    )
    n_H512_tiles = H_div_P // _q_width

    src = src.reshape((P, T, _q_width, n_H512_tiles))

    """
    Shuffle input SBUF layout from [H_128, T, H_4 * n_H512_tiles] to [H_128, n_H512_tiles, T, H_4],
    shard on n_H512_tiles between NCs, and pad T to a multiple of 4 for quantization AP restrictions.
    """

    n_H512_tile_sharded = n_H512_tiles // n_prgs
    T_padded = div_ceil(T, 4) * 4

    shfl_sb = nl.ndarray((P, n_H512_tile_sharded, T_padded, _q_width), dtype=src.dtype, buffer=nl.sbuf)
    nisa.memset(dst=shfl_sb, value=0.0)

    src_view = (
        src.slice(dim=3, start=prg_id * n_H512_tile_sharded, end=(prg_id + 1) * n_H512_tile_sharded).permute(
            dims=[0, 3, 1, 2]
        )  # [P, n_H512_tiles_sharded, T, _q_width]
    )

    dst_view = shfl_sb.slice(dim=2, start=0, end=T)

    nisa.tensor_copy(
        dst=dst_view,
        src=src_view,
    )

    return shfl_sb


def input_fused_add(
    input: nl.NkiTensor,
    fused_add_tensor: nl.NkiTensor,
    fused_output: nl.NkiTensor,
    normtype: NormType,
    sbm: SbufManager,
    dims: MLPTKGConstantsDimensionSizes,
) -> nl.NkiTensor:
    """
    Add fused_add_tensor to input hidden tensor (fused add).

    Depending on sharding:
    - Batch sharding (`do_norm_batch_sharding`): shard along T.
    - Hidden sharding: shard along H.
        A core barrier is inserted when normalization follows to ensure all shards complete before use.

    Flattens 3D inputs to 2D, shards along T (batch) or H (hidden), and performs
    element-wise add on HBM using dma_compute.

    Args:
        input (NkiTensor): Input hidden state [B, S, H] in HBM.
        fused_add_tensor (NkiTensor): Tensor to add [B, S, H] in HBM.
        fused_output (NkiTensor): Output buffer, modified in-place.
        normtype (NormType): Normalization type (controls barrier usage).
        sbm (SbufManager): SBUF allocation manager.
        dims (MLPTKGConstantsDimensionSizes): Dimension and sharding metadata.

    Returns:
        NkiTensor: fused_output in HBM.
    """
    shard_id = dims.shard_id
    num_shards = dims.num_shards
    H_per_shard = dims.H_per_shard

    input_2d = input.flatten_dims(start_dim=0, end_dim=1) if len(input.shape) == 3 else input
    fused_add_2d = (
        fused_add_tensor.flatten_dims(start_dim=0, end_dim=1) if len(fused_add_tensor.shape) == 3 else fused_add_tensor
    )
    fused_out_2d = fused_output.flatten_dims(start_dim=0, end_dim=1) if len(fused_output.shape) == 3 else fused_output

    if num_shards > 1 and dims.do_norm_batch_sharding:
        # Batch-sharded
        T_shard = dims.T // num_shards
        input_nd = input_2d.reshape_dim(dim=0, shape=[num_shards, T_shard]).select(dim=0, index=shard_id)
        fused_add_nd = fused_add_2d.reshape_dim(dim=0, shape=[num_shards, T_shard]).select(dim=0, index=shard_id)
        fused_out_nd = fused_out_2d.reshape_dim(dim=0, shape=[num_shards, T_shard]).select(dim=0, index=shard_id)
    else:
        # Hidden-sharded
        input_nd = input_2d.reshape_dim(dim=1, shape=[num_shards, H_per_shard]).select(dim=1, index=shard_id)
        fused_add_nd = fused_add_2d.reshape_dim(dim=1, shape=[num_shards, H_per_shard]).select(dim=1, index=shard_id)
        fused_out_nd = fused_out_2d.reshape_dim(dim=1, shape=[num_shards, H_per_shard]).select(dim=1, index=shard_id)

    nisa.dma_compute(
        dst=fused_out_nd,
        srcs=[input_nd, fused_add_nd],
        scales=[1.0, 1.0],
        reduce_op=nl.add,
    )

    if num_shards > 1 and normtype.value != NormType.NO_NORM.value:
        nisa.core_barrier(fused_output, cores=[0, 1])

    return fused_output


def input_norm_load(
    input: nl.NkiTensor,
    output: nl.NkiTensor,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
    T_offset: int = 0,
) -> nl.NkiTensor:
    """
    Load input activations and optionally apply normalization.

    Args:
        input (NkiTensor): Input hidden state [B, S, H] in HBM, [H0, T, H1] in SBUF, or [H0, n_prgs, H1_shard, BxS] when transposed_in=True.
        output (NkiTensor): SBUF tensor [H0, T, H1_shard] for normalized or loaded output.
        params (MLPParameters): Normalization parameters and settings.
        dims (MLPTKGConstantsDimensionSizes): Dimension data.
        sbm (SbufManager): SBUF allocation manager.
        T_offset (int): Offset into the T dimension for T-tiling. Only used in no-norm HBM path.

    Returns:
        NkiTensor: SBUF [H0, T, H1_shard].

    Notes:
        - MLP weight tensors are stack-allocated.
        - Normalization intermediates are heap-allocated to avoid address
          reuse and thereby prevent anti-dependencies when prefetching MLP weight tensors.
        - Supports RMSNorm and LayerNorm.
    """
    H0 = dims.H0
    T = dims.T
    H1 = dims.H1
    H1_shard = dims.H1_shard
    shard_id = dims.shard_id
    num_shards = dims.num_shards
    norm_weights = params.norm_params.normalization_weights_tensor
    norm_bias = params.norm_params.normalization_bias_tensor
    eps = params.eps

    # ------------------------- Norm + Input Load -------------------------
    if mlpp_has_normalization(params):
        if params.transposed_in:
            # Transposed input: [H0, n_prgs, H1_shard, T] in HBM
            # Contiguous load per-NC shard, permute in SBUF, run shard_on_h rmsnorm
            nc_size = H1_shard * T
            flat_size = num_shards * nc_size
            nc_offset = shard_id * nc_size

            # Step 1: Contiguous load [H0, H1_shard*T] from HBM
            input_raw_sb = sbm.alloc_heap(
                (H0, H1_shard, T), dtype=input.dtype, buffer=nl.sbuf, name="transposed_in_raw"
            )
            nisa.dma_copy(
                dst=input_raw_sb.reshape((H0, nc_size)),
                src=input.reshape((H0, flat_size))[:, nc_offset : nc_offset + nc_size],
            )

            # Step 2: Permute in SBUF: [H0, H1_shard, T] -> [H0, T, H1_shard]
            input_nc_sb = sbm.alloc_heap(
                (H0, T, H1_shard), dtype=input.dtype, buffer=nl.sbuf, name="transposed_in_perm"
            )
            nisa.tensor_copy(
                dst=input_nc_sb,
                src=input_raw_sb.permute(dims=[0, 2, 1]),
            )

            # Step 3: Run shard_on_h rmsnorm
            if mlpp_has_rms_normalization(params):
                rmsnorm_tkg(
                    input=input_nc_sb,
                    gamma=norm_weights,
                    output=output,
                    eps=eps,
                    hidden_actual=H0 * H1,
                    shard_on_h=True,
                    use_heap_memory=True,
                    sbm=sbm,
                )
            else:
                layernorm_tkg(
                    input=input_nc_sb,
                    gamma=norm_weights,
                    beta=norm_bias,
                    output=output,
                    eps=eps,
                    use_heap_memory=True,
                    sbm=sbm,
                )
            sbm.pop_heap()  # deallocate input_nc_sb
            sbm.pop_heap()  # deallocate input_raw_sb
        else:
            use_th_layout = mlpp_has_rms_normalization(params) and not (input.buffer == nl.sbuf) and T >= H0

            if use_th_layout:
                # Contiguous load [T, H] + norm in [T, H] layout + transpose to [H0, H1_shard, T]
                kernel_assert(
                    dims.hidden_layout == HiddenLayout.H0_H1_T,
                    "For using (T, H) layout rmsnorm, the hidden layout should be set to (H0, H1, T)",
                )
                hidden_flat = input.flatten_dims(start_dim=0, end_dim=1) if len(input.shape) == 3 else input
                rmsnorm_tkg_th(
                    input_hbm=hidden_flat,
                    gamma=norm_weights,
                    output=output,
                    num_H_shards=num_shards,
                    hidden_actual=dims.H,
                    eps=eps,
                    sbm=sbm,
                )
            else:
                norm_out = output
                if num_shards > 1:
                    norm_out = sbm.alloc_heap((H0, T, H1), dtype=input.dtype, buffer=nl.sbuf, name="norm_out_tensor")

                # Select normalization kernel (RMSNorm or LayerNorm)
                if mlpp_has_rms_normalization(params):
                    rmsnorm_tkg(
                        input=input,
                        gamma=norm_weights,
                        output=norm_out,
                        eps=eps,
                        use_heap_memory=True,
                        sbm=sbm,
                    )
                else:
                    layernorm_tkg(
                        input=input,
                        gamma=norm_weights,
                        beta=norm_bias,
                        output=norm_out,
                        eps=eps,
                        use_heap_memory=True,
                        sbm=sbm,
                    )

                # Slice normalized output per shard
                if num_shards > 1:
                    norm_out = norm_out.reshape(((H0, T, num_shards, H1_shard)))
                    nisa.tensor_copy(dst=output, src=norm_out[:, :, shard_id, :])

                    # deallocate norm_out
                    sbm.pop_heap()

    # --------------------------- No-Norm Path ----------------------------
    else:
        if params.transposed_in:
            # Transposed input: [H0, n_prgs, H1_shard, T] in HBM
            # Contiguous load per-NC shard, permute in SBUF
            nc_size = H1_shard * T
            flat_size = num_shards * nc_size
            nc_offset = shard_id * nc_size

            raw_sb = sbm.alloc_heap(
                (H0, H1_shard, T), dtype=input.dtype, buffer=nl.sbuf, name="transposed_in_raw_nonorm"
            )
            nisa.dma_copy(
                dst=raw_sb.reshape((H0, nc_size)),
                src=input.reshape((H0, flat_size))[:, nc_offset : nc_offset + nc_size],
            )
            # Permute in SBUF: [H0, H1_shard, T] -> [H0, T, H1_shard]
            nisa.tensor_copy(
                dst=output,
                src=raw_sb.permute(dims=[0, 2, 1]),
            )
            sbm.pop_heap()  # deallocate raw_sb
        else:
            input_view = input
            if len(input_view.shape) == 3:
                input_view = input_view.flatten_dims(start_dim=0, end_dim=1)

            # Use contiguous load + on-chip transpose for small H
            if dims.H_per_shard <= _CONTIGUOUS_LOAD_H_THRESHOLD:
                kernel_assert(
                    dims.H % H0 == 0,
                    f"H ({dims.H}) must be divisible by {H0}",
                )
                # Apply T_offset slicing for T-tiling
                if T_offset > 0 or T < input_view.shape[0]:
                    input_view = input_view.slice(dim=0, start=T_offset, end=T_offset + T)
                # [T, H] -> [T, H_per_shard]
                input_view = input_view.slice(
                    dim=1,
                    start=dims.H1_offset * dims.H0,
                    end=dims.H1_offset * dims.H0 + dims.H_per_shard,
                )
                # [T, H_per_shard] -> [H0, T, H1_shard]
                prev_prefix = sbm.get_name_prefix()
                sbm.set_name_prefix(f"{prev_prefix}nonorm_t{T_offset}_")
                contiguous_load_transpose(input_view, output, 1, sbm)
                sbm.set_name_prefix(prev_prefix)
            else:
                input_view = input
                if len(input_view.shape) == 3:
                    # (B, S, H)
                    input_view = input_view.flatten_dims(start_dim=0, end_dim=1)
                # Expecting (T_total, H) otherwise

                # Apply T_offset slicing for T-tiling
                if T_offset > 0 or T < input_view.shape[0]:
                    input_view = input_view.slice(dim=0, start=T_offset, end=T_offset + T)

                input_view = (
                    input_view.reshape_dim(dim=1, shape=[num_shards, H0, H1_shard])  # T, num_shards, H0:128, H1_shard
                    .permute(dims=[2, 0, 1, 3])  # 128, T, num_shards, H1_shard
                    .select(dim=2, index=shard_id)  # 128, T, H1_shard
                )

                # Load input[T, H] to [H0, T, H1_shard]
                nisa.dma_copy(
                    src=input_view,
                    dst=output,
                    dge_mode=_DGE_MODE_NONE,
                )

    return output


def _load_transposed_tile(
    src_tensor: nl.NkiTensor,
    dst_tile: nl.NkiTensor,
    total_size: int,
    tile_size: int,
    op_name: str,
    sbm: SbufManager,
    use_dge: bool = False,
) -> None:
    """
    Load a 1D tensor into a transposed 2D SBUF tile [tile_size, num_tiles].

    Handles both aligned (full tile_size) and residual elements. Uses dma_transpose
    when possible, falls back to dma_copy + nc_transpose for dynamic access patterns.

    Args:
        src_tensor (NkiTensor): 1D source tensor of size total_size.
        dst_tile (NkiTensor): 2D destination tile [tile_size, num_total_tiles] in SBUF.
        total_size (int): Total number of elements in source.
        tile_size (int): Size of each tile (typically I0=128).
        op_name (str): Operation name for buffer naming.
        sbm (SbufManager): SBUF allocation manager.
        use_dge (bool): If True, force SWDGE mode for all DMA operations.
    """
    src_dim = 0 if src_tensor.ndim == 1 else 1
    num_full_tiles = total_size // tile_size
    res_elements = total_size % tile_size

    # When use_dge=True (inside dynamic loops), avoid the residual path which triggers
    # NCC_IGCA108 (loop-carried dependency). Instead, load full tiles and residual into
    # a single transposed buffer to eliminate separate writes to dst_tile.
    if use_dge and res_elements > 0:
        num_padded_tiles = num_full_tiles + 1
        tmp_sbuf = sbm.alloc_stack(
            shape=(num_padded_tiles, tile_size),
            dtype=src_tensor.dtype,
            buffer=nl.sbuf,
            name=f"{op_name}_{sbm.get_name_prefix()}_transpose_sbuf",
        )
        nisa.memset(tmp_sbuf, 0)
        tmp_psum = nl.ndarray((tile_size, num_padded_tiles), dtype=src_tensor.dtype, buffer=nl.psum)
        if num_full_tiles > 0:
            full_src = src_tensor.slice(dim=src_dim, start=0, end=num_full_tiles * tile_size).reshape_dim(
                dim=src_dim, shape=(num_full_tiles, tile_size)
            )
            if full_src._scalar_offset is not None and full_src._indirect_dim is not None:
                nisa.dma_copy(
                    src=full_src,
                    dst=tmp_sbuf.ap(
                        pattern=[[tile_size, num_full_tiles], [1, tile_size]],
                        offset=0,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                )
            else:
                while full_src.ndim < 4:
                    full_src = full_src.expand_dim(1)
                nisa.dma_copy(src=full_src, dst=tmp_sbuf[:num_full_tiles, :tile_size], dge_mode=nisa.dge_mode.hwdge)
        res_src = src_tensor.slice(dim=src_dim, start=num_full_tiles * tile_size, end=total_size).expand_dim(1)
        nisa.dma_copy(
            src=res_src,
            dst=tmp_sbuf[num_full_tiles : num_full_tiles + 1, :res_elements],
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.nc_transpose(dst=tmp_psum, data=tmp_sbuf)
        nisa.tensor_copy(dst=dst_tile.slice(dim=1, start=0, end=num_padded_tiles), src=tmp_psum)
        return

    if num_full_tiles > 0:
        src_view = src_tensor.slice(dim=src_dim, start=0, end=total_size - res_elements).reshape_dim(
            dim=src_dim, shape=(num_full_tiles, tile_size)
        )
        if not (src_view._scalar_offset is not None and src_view._indirect_dim is not None) and not use_dge:
            while src_view.ndim < 4:
                src_view = src_view.expand_dim(1)
            nisa.dma_transpose(
                src=src_view,
                dst=dst_tile.slice(dim=1, start=0, end=num_full_tiles).expand_dim(1).expand_dim(1),
                dge_mode=_DGE_MODE_UNKNOWN if use_dge else _DGE_MODE_NONE,
            )
        else:
            # Workaround for dynamic access not supported by dma_transpose (NKI-415)
            tmp_sbuf = sbm.alloc_stack(
                shape=(num_full_tiles, tile_size),
                dtype=src_tensor.dtype,
                buffer=nl.sbuf,
                name=f"{op_name}_{sbm.get_name_prefix()}_transpose_sbuf",
            )
            tmp_psum = nl.ndarray((tile_size, num_full_tiles), dtype=src_tensor.dtype, buffer=nl.psum)
            if src_view._scalar_offset is not None and src_view._indirect_dim is not None:
                nisa.dma_copy(
                    src=src_view,
                    dst=tmp_sbuf.ap(
                        pattern=[[tile_size, num_full_tiles], [1, tile_size]],
                        offset=0,
                    ),
                    dge_mode=nisa.dge_mode.hwdge if use_dge else adaptive_dge_mode(src_view),
                )
            else:
                while src_view.ndim < 4:
                    src_view = src_view.expand_dim(1)
                nisa.dma_copy(
                    src=src_view,
                    dst=tmp_sbuf,
                    dge_mode=nisa.dge_mode.hwdge if use_dge else _DGE_MODE_NONE,
                )
            nisa.nc_transpose(dst=tmp_psum, data=tmp_sbuf)
            nisa.tensor_copy(dst=dst_tile.slice(dim=1, start=0, end=num_full_tiles), src=tmp_psum)

    if res_elements > 0:
        src_view = src_tensor.slice(dim=src_dim, start=total_size - res_elements, end=total_size).expand_dim(1)
        nisa.dma_copy(
            src=src_view,
            dst=dst_tile.slice(dim=0, start=0, end=res_elements).slice(
                dim=1, start=num_full_tiles, end=num_full_tiles + 1
            ),
            dge_mode=nisa.dge_mode.hwdge if use_dge else adaptive_dge_mode(src_view),
        )


def _clamp_lower_upper_limit(tensor: nl.NkiTensor, lower_limit, upper_limit) -> None:
    """Apply optional lower and upper clamping to a tensor in-place."""
    if upper_limit is not None:
        nisa.tensor_scalar(data=tensor, dst=tensor, op0=nl.minimum, operand0=upper_limit)
    if lower_limit is not None:
        nisa.tensor_scalar(data=tensor, dst=tensor, op0=nl.maximum, operand0=lower_limit)


def transpose_store(
    output_temp: nl.NkiTensor,
    output: nl.NkiTensor,
    dims: MLPTKGConstantsDimensionSizes,
    output_dtype: nki.dtype,
    sbm: SbufManager,
    T_offset: int = 0,
) -> None:
    """
    Transpose temporary output SBUF tensor and store to final HBM tensor.

    This function handles the storage of the output tensor from temporary SBUF tensor
    to the final output tensor, taking into account the hardware-specific requirements
    and data layout.

    Args:
        output_temp (nl.NkiTensor): Temporary output tensor storage [H0, H1, T] in SBUF.
        output (nl.NkiTensor): Final output tensor [T, H] in HBM.
        dims (MLPTKGConstantsDimensionSizes): Dimension sizes object.
        output_dtype (nki.dtype): Data type of the output tensor.
        sbm (SbufManager): SbufManager for buffer allocation.
        T_offset (int): Offset into the T dimension for storing output. Default 0.
    """

    output_sb = sbm.alloc_stack(
        (dims.T, dims.H_per_shard),
        dtype=output_dtype,
        buffer=nl.sbuf,
        name="tkg_moe_output_sb",
    )

    # Transpose output[H0, H1, T] to [T, H], only required in LHS/RHS swap projection
    H0, H1, T = output_temp.shape
    for h1_tile_idx in range(H1):
        psum_idx = h1_tile_idx % dims._psum_bmax
        tp_psum = nl.ndarray(
            (T, H0),
            dtype=output_dtype,
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, psum_idx * dims._psum_fmax * 4),
        )
        nisa.nc_transpose(dst=tp_psum[0:T, 0:H0], data=output_temp[0:H0, h1_tile_idx, 0:T])
        interleave_copy(
            dst=output_sb.ap(
                pattern=[[dims.H_per_shard, T], [H1, H0]],
                offset=h1_tile_idx,
            ),
            src=tp_psum[0:T, 0:H0],
            index=h1_tile_idx,
        )

    nisa.dma_copy(
        dst=output[nl.ds(T_offset, T), nl.ds(dims.H1_offset * dims.H0, dims.H_per_shard)],
        src=output_sb[:, 0 : dims.H_per_shard],
    )


def transpose_store_sbuf_copy(
    output_temp: nl.NkiTensor,
    output: nl.NkiTensor,
    dims: MLPTKGConstantsDimensionSizes,
    output_dtype: nki.dtype,
    sbm: SbufManager,
    T_offset: int = 0,
) -> None:
    """
    Transpose temporary output SBUF tensor and store to final HBM tensor using
    SBUF-to-SBUF copies instead of PSUM transpose.

    Rearranges output_temp [H0, H1_shard, T] to [H0, T, H1_shard] in SBUF using
    interleaved scalar/vector engine copies, then stores to HBM [T, H] with
    a single DMA using access patterns.

    Args:
        output_temp (nl.NkiTensor): Temporary output tensor storage [H0, H1_shard, T] in SBUF.
        output (nl.NkiTensor): Final output tensor [T, H] in HBM.
        dims (MLPTKGConstantsDimensionSizes): Dimension sizes object.
        output_dtype (nki.dtype): Data type of the output tensor.
        sbm (SbufManager): SbufManager for buffer allocation.
        T_offset (int): Offset along the T dimension for the output store.
    """
    H0, H1_shard, T = output_temp.shape
    H = dims.H

    output_reshape = sbm.alloc_stack(
        (H0, T, H1_shard),
        dtype=output_dtype,
        buffer=nl.sbuf,
        name="tkg_output_reshape",
    )

    # SBUF copy: rearrange [H0, H1_shard, T] -> [H0, T, H1_shard] with engine interleaving
    src_copy_pat = [[T * H1_shard, H0], [T, H1_shard]]
    dst_copy_pat = [[T * H1_shard, H0], [1, H1_shard]]
    for token in range(T):
        src_ap = output_temp.ap(pattern=src_copy_pat, offset=token)
        dst_ap = output_reshape.ap(pattern=dst_copy_pat, offset=token * H1_shard)
        if token % 2 == 0:
            nisa.tensor_copy(dst=dst_ap, src=src_ap, engine=nisa.scalar_engine)
        else:
            nisa.tensor_copy(dst=dst_ap, src=src_ap, engine=nisa.vector_engine)

    # Store [H0, T, H1_shard] -> HBM [T, H] via AP-based transpose
    store_src_pat = [[T * H1_shard, H0], [H1_shard, T], [1, H1_shard]]
    store_dst_pat = [[H1_shard, H0], [H, T], [1, H1_shard]]
    store_dst_offset = T_offset * H + dims.H1_offset * dims.H0
    nisa.dma_copy(
        dst=output.ap(pattern=store_dst_pat, offset=store_dst_offset),
        src=output_reshape.ap(pattern=store_src_pat, offset=0),
    )


def adaptive_dge_mode(tensor: nl.NkiTensor) -> int:
    """
    Determine DGE mode based on tensor access pattern.

    Args:
        tensor: nl.NkiTensor to check for dynamic access.

    Returns:
        int: _DGE_MODE_UNKNOWN if dynamic access (compiler decides), _DGE_MODE_NONE (static) otherwise.
    """
    if not (tensor._scalar_offset is not None and tensor._indirect_dim is not None):
        return _DGE_MODE_NONE
    else:
        return _DGE_MODE_UNKNOWN
