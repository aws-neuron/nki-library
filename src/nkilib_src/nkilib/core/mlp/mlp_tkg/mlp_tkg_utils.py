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
from ...subkernels.rmsnorm_tkg import rmsnorm_tkg
from ...utils.allocator import SbufManager
from ...utils.common_types import NormType
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.tensor_view import TensorView
from ..mlp_parameters import (
    MLPBiasParameters,
    MLPParameters,
    mlpp_has_normalization,
    mlpp_has_rms_normalization,
)
from .mlp_tkg_constants import MLPTKGConstantsDimensionSizes

_DGE_MODE_UNKNOWN = 0  # Compiler decides best DMA mode internally
_DGE_MODE_NONE = 3  # Use STATIC DMA mode


def input_fused_add(
    input: nl.ndarray,
    fused_add_tensor: nl.ndarray,
    fused_output: nl.ndarray,
    normtype: NormType,
    store_fused_add_result: bool,
    sbm: SbufManager,
    dims: MLPTKGConstantsDimensionSizes,
) -> nl.ndarray:
    """
    Add fused_add_tensor to input hidden tensor (fused add).

    Depending on sharding:
    - Batch sharding (`do_norm_batch_sharding`): shard along T.
        Input layout must be [num_shards, T/num_shards, H].
    - Hidden sharding: shard along H.
        Input layout must be [T, num_shards, H/num_shards].
        A core barrier is inserted when normalization follows to ensure all shards complete before use.

    Args:
        input (nl.ndarray): Input hidden state.
            Expected layouts:
                - HBM: [B, S, H]
                - SBUF: [128, BxS, H//128]
        fused_add_tensor (nl.ndarray): Tensor to add.
        fused_output (nl.ndarray): Output buffer, modified in-place.
        normtype (NormType): Normalization type (controls barrier usage).
        store_fused_add_result (bool): Whether to store result to HBM.
        sbm (SbufManager): SBUF allocation manager.
        dims (MLPTKGConstantsDimensionSizes): Dimension and sharding metadata.

    Returns:
        nl.ndarray:
            When store_fused_add_result = True:
                - HBM: [BxS, H]
            When store_fused_add_result = False:
                - RMSNORM / LAYERNORM:
                    SBUF: [128, BxS, H//128]
                - NONORM:
                    SBUF: [128, BxS, H//128//LNC_SIZE]
    """
    H0 = dims.H0
    T = dims.T
    H = dims.H
    H1 = dims.H1
    H1_shard = dims.H1_shard
    shard_id = dims.shard_id
    num_shards = dims.num_shards
    H_per_shard = dims.H_per_shard

    is_input_in_sbuf = input.buffer == nl.sbuf

    if is_input_in_sbuf:
        kernel_assert(not store_fused_add_result, "Storing fused_add_result is not supported when input is in SBUF")

    if store_fused_add_result:
        # Batch-sharded
        if num_shards > 1 and dims.do_norm_batch_sharding:
            input = input.reshape((num_shards, T // num_shards, H))
            fused_add_tensor = fused_add_tensor.reshape((num_shards, T // num_shards, H))
            fused_output = fused_output.reshape((num_shards, T // num_shards, H))
            nisa.dma_compute(
                dst=fused_output[shard_id, 0 : T // num_shards, 0:H],
                srcs=[
                    input[shard_id, 0 : T // num_shards, 0:H],
                    fused_add_tensor[shard_id, 0 : T // num_shards, 0:H],
                ],
                scales=[1.0, 1.0],
                reduce_op=nl.add,
            )
        # Hidden-sharded
        else:
            input = input.reshape((T, num_shards, H_per_shard))
            fused_add_tensor = fused_add_tensor.reshape((T, num_shards, H_per_shard))
            fused_output = fused_output.reshape((T, num_shards, H_per_shard))
            nisa.dma_compute(
                dst=fused_output[0:T, shard_id, 0:H_per_shard],
                srcs=[
                    input[0:T, shard_id, 0:H_per_shard],
                    fused_add_tensor[0:T, shard_id, 0:H_per_shard],
                ],
                scales=[1.0, 1.0],
                reduce_op=nl.add,
            )
        if num_shards > 1 and normtype.value != NormType.NO_NORM.value:
            nisa.core_barrier(fused_output, cores=[0, 1])

        fused_output = fused_output.reshape((T, H))

    else:
        # If NO_NORM, load only the sharded portion
        if normtype == NormType.NO_NORM:
            load_shape = (H0, T, H1_shard)
        else:
            load_shape = (H0, T, num_shards, H1_shard)
            fused_output = fused_output.reshape(load_shape)

        # ---------------- Load input ----------------
        if is_input_in_sbuf:
            input_view = TensorView(input)

            # (H0,BxS,H1) -> (H0,BxS,num_shards,H1_shard) -> (H0,BxS,H1_shard)
            if normtype == NormType.NO_NORM:
                input_view = (
                    input_view.reshape_dim(dim=2, shape=[num_shards, H1_shard])
                    .slice(start_dim=2, start=shard_id, end=shard_id + 1)
                    .squeeze_dim(dim=2)
                )
        else:
            input_buf = sbm.alloc_heap(load_shape, dtype=input.dtype, buffer=nl.sbuf)

            # Transform input: (B,S,H) -> (B,S,num_shards,H0,H1_shard) -> (BxS,num_shards,H0,H1_shard) -> (H0,BxS,num_shards,H1_shard)
            input_view = (
                TensorView(input)
                .reshape_dim(dim=2, shape=[num_shards, H0, H1_shard])
                .flatten_dims(start_dim=0, end_dim=1)
                .permute(dims=[2, 0, 1, 3])
            )

            # (H0,BxS,num_shards,H1_shard) -> (H0,BxS,H1_shard)
            if normtype == NormType.NO_NORM:
                input_view = input_view.slice(dim=2, start=shard_id, end=shard_id + 1).squeeze_dim(dim=2)

            # Load input tensors
            nisa.dma_copy(
                src=input_view.get_view(),
                dst=input_buf[...],
                dge_mode=_DGE_MODE_NONE,
            )
            input_view = TensorView(input_buf)

        # ---------------- Load fused_add_tensor ----------------
        fused_add_tensor_buf = sbm.alloc_heap(load_shape, dtype=fused_add_tensor.dtype, buffer=nl.sbuf)

        # Transform input: (B,S,H) -> (B,S,num_shards,H0,H1_shard) -> (BxS,num_shards,H0,H1_shard) -> (H0,BxS,num_shards,H1_shard)
        fused_add_tensor_view = (
            TensorView(fused_add_tensor)
            .reshape_dim(dim=2, shape=[num_shards, H0, H1_shard])
            .flatten_dims(start_dim=0, end_dim=1)
            .permute(dims=[2, 0, 1, 3])
        )

        # (H0,BxS,num_shards,H1_shard) -> (H0,BxS,H1_shard)
        if normtype == NormType.NO_NORM:
            fused_add_tensor_view = fused_add_tensor_view.slice(dim=2, start=shard_id, end=shard_id + 1).squeeze_dim(
                dim=2
            )

        # Load fused add tensor
        nisa.dma_copy(
            src=fused_add_tensor_view.get_view(),
            dst=fused_add_tensor_buf[...],
            dge_mode=_DGE_MODE_NONE,
        )

        # fused_output = input + fused_add_tensor
        nisa.tensor_tensor(dst=fused_output, data1=input_view.get_view(), data2=fused_add_tensor_buf, op=nl.add)

        if normtype != NormType.NO_NORM:
            fused_output = fused_output.reshape((H0, T, H1))

        if not is_input_in_sbuf:
            sbm.pop_heap()  # deallocate input_buf
        sbm.pop_heap()  # deallocate fused_add_tensor_buf

    return fused_output


def input_norm_load(
    input: nl.ndarray,
    output: nl.ndarray,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Load input activations and optionally apply normalization.

    Args:
        input (nl.ndarray): Input hidden state.
            Expected layouts:
                When input is in HBM:
                    - [B, S, H]
                When input is in SBUF:
                    - [128, B×S, H//128]
        output (nl.ndarray): SBUF tensor of shape [128, B×S, H//128//LNC_SIZE], used to
            store the normalized output or the loaded input.
        params (MLPParameters): Normalization parameters and settings.
        dims (MLPTKGConstantsDimensionSizes): Dimension data.
        sbm (SbufManager): SBUF allocation manager.

    Returns:
        nl.ndarray: SBUF [128, B×S, H//128//LNC_SIZE].

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
        # Transform input: (B,S,H) -> (B,S,num_shards,H0,H1_shard) -> (BxS,num_shards,H0,H1_shard) -> (H0,BxS,num_shards,H1_shard)
        input_view = TensorView(input)
        if len(input_view.shape) == 3:
            # (B, S, H)
            input_view = input_view.flatten_dims(start_dim=0, end_dim=1)
        # Expecting (T, H) otherwise

        input_view = (
            input_view.reshape_dim(dim=1, shape=[num_shards, H0, H1_shard])  # T, num_shards, H0:128, H1_shard
            .permute(dims=[2, 0, 1, 3])  # 128, T, num_shards, H1_shard
            .slice(dim=2, start=shard_id, end=shard_id + 1)  # 128, T, H1_shard
            .squeeze_dim(dim=2)
        )

        # Load input[T, H] to [H0, T, H1_shard]
        nisa.dma_copy(
            src=input_view.get_view(),
            dst=output[0:H0, 0:T, 0:H1_shard],
            dge_mode=_DGE_MODE_NONE,
        )

    return output


def convert_weight_scale_params_to_views(params: MLPParameters) -> None:
    """
    Convert weights, bias and quantization scale vectors into TensorViews in-place.

    Args:
        params (MLPParameters): Parameter object, will be modified in-place.
    """
    gate_bias_view, up_bias_view, down_bias_view = None, None, None
    if not params.use_tkg_gate_up_proj_column_tiling:
        params.gate_proj_weights_tensor = (
            TensorView(params.gate_proj_weights_tensor) if not params.skip_gate_proj else None
        )
        if params.bias_params.gate_proj_bias_tensor is not None:
            gate_bias_shape = (params.bias_params.gate_proj_bias_tensor.shape[1],)
            gate_bias_view = TensorView(params.bias_params.gate_proj_bias_tensor.reshape(gate_bias_shape))
        params.up_proj_weights_tensor = TensorView(params.up_proj_weights_tensor)
        if params.bias_params.up_proj_bias_tensor is not None:
            up_bias_shape = (params.bias_params.up_proj_bias_tensor.shape[1],)
            up_bias_view = TensorView(params.bias_params.up_proj_bias_tensor.reshape(up_bias_shape))
    if not params.use_tkg_down_proj_column_tiling:
        params.down_proj_weights_tensor = TensorView(params.down_proj_weights_tensor)
        if params.bias_params.down_proj_bias_tensor is not None:
            down_bias_shape = (params.bias_params.down_proj_bias_tensor.shape[1],)
            down_bias_view = TensorView(params.bias_params.down_proj_bias_tensor.reshape(down_bias_shape))

    if gate_bias_view is not None or up_bias_view is not None or down_bias_view is not None:
        params.bias_params = MLPBiasParameters(
            gate_proj_bias_tensor=gate_bias_view,
            up_proj_bias_tensor=up_bias_view,
            down_proj_bias_tensor=down_bias_view,
        )

    if params.quant_params.is_quant():
        params.quant_params.convert_to_view()


def transpose_store(
    output_temp: nl.ndarray,
    output: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    output_dtype: nki.dtype,
    sbm: SbufManager,
) -> None:
    """
    Transpose temporary output SBUF tensor and store to final HBM tensor.

    This function handles the storage of the output tensor from temporary SBUF tensor
    to the final output tensor, taking into account the hardware-specific requirements
    and data layout.

    Args:
        output_temp (nl.ndarray): Temporary output tensor storage [H0, H1, T] in SBUF.
        output (nl.ndarray): Final output tensor [T, H] in HBM.
        dims (MLPTKGConstantsDimensionSizes): Dimension sizes object.
        output_dtype (nki.dtype): Data type of the output tensor.
        sbm (SbufManager): SbufManager for buffer allocation.
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
            name=f"transpose_output_{h1_tile_idx}",
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
        dst=output[:, nl.ds(dims.shard_id * dims.H_per_shard, dims.H_per_shard)],
        src=output_sb[:, : dims.H_per_shard],
    )


def adaptive_dge_mode(tensor: TensorView) -> int:
    """
    Determine DGE mode based on tensor access pattern.

    Args:
        tensor: TensorView to check for dynamic access.

    Returns:
        int: _DGE_MODE_UNKNOWN if dynamic access (compiler decides), _DGE_MODE_NONE (static) otherwise.
    """
    if not isinstance(tensor, TensorView) or not tensor.has_dynamic_access():
        return _DGE_MODE_NONE
    else:
        return _DGE_MODE_UNKNOWN
