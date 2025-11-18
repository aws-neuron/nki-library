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
kernels - high performance MLP kernels

"""

import nki.isa as nisa
import nki.language as nl

# common utils
from ...utils.common_types import NormType
from ...utils.allocator import SbufManager
from ...utils.tensor_view import TensorView

# subkernels utils
from ...subkernels.layernorm_tkg import layernorm_tkg
from ...subkernels.rmsnorm_tkg import rmsnorm_tkg

# MLP utils
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_layer_normalization,
    mlpp_has_normalization,
    mlpp_has_rms_normalization,
)
from .mlp_tkg_constants import MLPTKGConstantsDimensionSizes


def input_fused_add(
    input_hbm: nl.ndarray,
    fused_add_tensor: nl.ndarray,
    fused_output: nl.ndarray,
    normtype: NormType,
    store_fused_add_result: bool,
    sbm: SbufManager,
    dims: MLPTKGConstantsDimensionSizes,
) -> nl.ndarray:
    """
    Add fused_add_tensor to input hidden tensor (fused add)

    Depending on sharding:
    - Batch sharding (`do_norm_batch_sharding`): shard along T.
        Input layout must be [num_shards, T/num_shards, H].
    - Hidden sharding: shard along H.
        Input layout must be [T, num_shards, H/num_shards].
        A core barrier is inserted when normalization follows to ensure all shards complete before use.

    Args:
        input_hbm (nl.ndarray): Input hidden states from HBM [T, H].
        fused_add_tensor (nl.ndarray): Tensor to add.
        fused_output (nl.ndarray): Output buffer, modified in-place.
        normtype (NormType): Normalization type (controls barrier usage).
        sbm (SbufManager): SBUF allocation manager.
        dims (MLPTKGConstantsDimensionSizes): Dimension and sharding metadata.

    Returns:
        nl.ndarray: Fused hidden tensor [T, H].
    """

    # Parse constants
    H0 = dims.H0
    T = dims.T
    H = dims.H
    H1 = dims.H1
    H1_shard = dims.H1_shard
    shard_id = dims.shard_id
    num_shards = dims.num_shards
    H_per_shard = dims.H_per_shard

    if store_fused_add_result:
        # Batch-sharded
        if num_shards > 1 and dims.do_norm_batch_sharding:
            input_hbm = input_hbm.reshape((num_shards, T // num_shards, H))
            fused_add_tensor = fused_add_tensor.reshape((num_shards, T // num_shards, H))
            fused_output = fused_output.reshape((num_shards, T // num_shards, H))
            nisa.dma_compute(
                dst=fused_output[shard_id, 0 : T // num_shards, 0:H],
                srcs=[
                    input_hbm[shard_id, 0 : T // num_shards, 0:H],
                    fused_add_tensor[shard_id, 0 : T // num_shards, 0:H],
                ],
                scales=[1.0, 1.0],
                reduce_op=nl.add,
            )
        # Hidden-sharded
        else:
            input_hbm = input_hbm.reshape((T, num_shards, H_per_shard))
            fused_add_tensor = fused_add_tensor.reshape((T, num_shards, H_per_shard))
            fused_output = fused_output.reshape((T, num_shards, H_per_shard))
            nisa.dma_compute(
                dst=fused_output[0:T, shard_id, 0:H_per_shard],
                srcs=[
                    input_hbm[0:T, shard_id, 0:H_per_shard],
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

        # Allocate input_hbm_buf and fused_add_tensor_buf
        input_hbm_buf = sbm.alloc_heap(load_shape, dtype=input_hbm.dtype, buffer=nl.sbuf)
        fused_add_tensor_buf = sbm.alloc_heap(load_shape, dtype=fused_add_tensor.dtype, buffer=nl.sbuf)

        # Transform input: (B,S,H) -> (B,S,num_shards,H0,H1_shard) -> (BxS,num_shards,H0,H1_shard) -> (H0,BxS,num_shards,H1_shard)
        input_view = (
            TensorView(input_hbm)
            .reshape_dim(dim=2, sizes=[num_shards, H0, H1_shard])
            .flatten_dims(start_dim=0, end_dim=1)
            .permute(dims=[2, 0, 1, 3])
        )
        fused_add_tensor_view = (
            TensorView(fused_add_tensor)
            .reshape_dim(dim=2, sizes=[num_shards, H0, H1_shard])
            .flatten_dims(start_dim=0, end_dim=1)
            .permute(dims=[2, 0, 1, 3])
        )

        # (H0,BxS,num_shards,H1_shard) -> (H0,BxS,H1_shard)
        if normtype == NormType.NO_NORM:
            input_view = input_view.slice(dim=2, start=shard_id, end=shard_id + 1).squeeze_dim(dim=2)
            fused_add_tensor_view = fused_add_tensor_view.slice(dim=2, start=shard_id, end=shard_id + 1).squeeze_dim(
                dim=2
            )

        # Load input tensors
        nisa.dma_copy(
            src=input_view.get_view(),
            dst=input_hbm_buf[...],
            dge_mode=nisa.dge_mode.none,
        )
        nisa.dma_copy(
            src=fused_add_tensor_view.get_view(),
            dst=fused_add_tensor_buf[...],
            dge_mode=nisa.dge_mode.none,
        )

        # fused_output = input_hbm + fused_add_tensor
        nisa.tensor_tensor(dst=fused_output, data1=input_hbm_buf, data2=fused_add_tensor_buf, op=nl.add)

        if normtype != NormType.NO_NORM:
            fused_output = fused_output.reshape((H0, T, H1))

        sbm.pop_heap()  # deallocate input_hbm_buf
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
        input (nl.ndarray): Input hidden states. Could be in either SBUF[H0, T, H1] and HBM[T, H].
        output (nl.ndarray): SBUF tensor to store norm output or loaded input into [H0, T, H1_shard].
        is_input_in_sbuf (bool) : Indicate if input is already loaded into SBUF
        params (MLPParameters): Normalization parameters and settings.
        dims (MLPTKGConstantsDimensionSizes): Dimension data.
        sbm (SbufManager): SBUF allocation manager.

    Returns:
        nl.ndarray: SBUF tensor containing loaded (and optionally normalized) input.

    Notes:
        - MLP weight tensors are stack-allocated.
        - Normalization intermediates are heap-allocated to avoid address
        reuse and thereby prevent anti-dependencies when prefetching MLP weight tensors.
        - Supports RMSNorm and LayerNorm.
    """
    # Parse constants
    H0 = dims.H0
    T = dims.T
    H1_shard = dims.H1_shard
    shard_id = dims.shard_id
    num_shards = dims.num_shards
    norm_weights = params.norm_params.normalization_weights_tensor
    norm_bias = params.norm_params.normalization_bias_tensor
    eps = params.eps

    is_input_in_sbuf = input.buffer == nl.sbuf

    # ------------------------- Norm + Input Load -------------------------
    if mlpp_has_normalization(params):
        # Select normalization kernel (RMSNorm or LayerNorm)
        if mlpp_has_rms_normalization(params):
            norm_out = rmsnorm_tkg(
                input,
                norm_weights,
                eps,
                output_in_sbuf=True,
                use_heap_memory=True,
                sbm=sbm,
            )
        else:
            norm_out = layernorm_tkg(
                input,
                norm_weights,
                norm_bias,
                eps,
                output_in_sbuf=True,
                use_heap_memory=True,
                sbm=sbm,
            )

        # Slice normalized output per shard
        if num_shards > 1:
            norm_out = norm_out.reshape(((H0, T, num_shards, H1_shard)))
            nisa.tensor_copy(dst=output, src=norm_out[:, :, shard_id, :])
        else:
            nisa.tensor_copy(dst=output, src=norm_out)

        # deallocate norm_out
        sbm.pop_heap()

    # --------------------------- No-Norm Path ----------------------------
    else:
        if is_input_in_sbuf:
            nisa.tensor_copy(dst=output, src=input)
        else:
            # Transform input: (B,S,H) -> (B,S,num_shards,H0,H1_shard) -> (BxS,num_shards,H0,H1_shard) -> (H0,BxS,num_shards,H1_shard)
            input_view = (
                TensorView(input)
                .reshape_dim(dim=2, sizes=[num_shards, H0, H1_shard])
                .flatten_dims(start_dim=0, end_dim=1)
                .permute(dims=[2, 0, 1, 3])
                .slice(dim=2, start=shard_id, end=shard_id + 1)
                .squeeze_dim(dim=2)
            )

            # Load input[T, H] to [H0, T, H1_shard]
            nisa.dma_copy(
                src=input_view.get_view(),
                dst=output[0:H0, 0:T, 0:H1_shard],
                dge_mode=nisa.dge_mode.none,
            )

    return output
