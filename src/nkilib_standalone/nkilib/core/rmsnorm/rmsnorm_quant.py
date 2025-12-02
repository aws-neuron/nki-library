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


import logging
from dataclasses import dataclass
from typing import Optional

import nki
import nki.compiler as ncc
import nki.isa as nisa
import nki.language as nl

from ..utils.common_types import NormType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import (
    get_program_sharding_info,
    get_verified_program_sharding_info,
    is_launched_as_spmd,
    is_rms_normalization,
)
from .rmsnorm_quant_constants import (
    RMSNormQuantConstants,
    build_rms_norm_quant_constants,
)
from .rmsnorm_quant_tile_info import (
    RMSNormQuantTileInfo,
    build_rms_norm_quant_tile_info,
)

"""
8-bit Quantization and RMS Normalization Kernel

This kernel performs quantization down to 8 bits along the last dimension of the input tensor.  What this means is that for each
given last dimension vector, the maximum absolute value is found and that is used in conjunction with the maximum values that can be
represented in the 8-bit data type to scale the values of that vector to fit within the range of the 8-bit data type.  The scale
factors for converting each vector back to 16 bits (i.e. dequantizing) are provided in the output tensor along with the quantized
vectors.

RMS normalization is an optional step when calling this kernel.  If it is desired, the RMS normalization is computed along each
of the last dimension vectors.  This happens prior to quantization.

This kernel accepts an input tensor of at minimum 2 dimensions.  We establish 2 terms here:
processing dimension - The dimension along which the kernel calculates normalization and quantization.
                       This is the last dimension of the input tensor.
outer dimension      - This dimension is comprised of all input tensor dimensions excluding the last dimension.

For example, if our input tensor has shape [W, X, Y, Z], we reshape the tensor to be [W * X * Y, Z] where
the processing dimension is Z and the outer dimension is W * X * Y.  This is done for a few reasons:
1. Since the code processes along the last dimension, there is no reason to require specific criteria for the other
   dimensions other than that there has to be at least one other dimension (i.e. the input tensor itself must be at least 2D).
   We then naturally support input tensors like ones of shape [batch size, sequence length, hidden size] without requiring
   such specific shapes.
2. When we collapse all major dimensions into this outer dimension and have one loop, we only have one boundary condition at the
   very end of the loop where compute is less efficient if the outer dimension is not a multiple of the tile size for that dimension.
   Contrast that with an example where we have a loop for batch size and a nested loop for sequence length.  If the batch size is
   greater than 1 and the sequence length is not a multiple of the tile size, we end up with the inefficient boundary condition
   for each batch loop.

To keep comments more abbreviated, we establish the following acronyms when describing things like shapes:
  OD  - Outer dimension
  PD  - Processing dimension
  ODT - Outer dimension tile
  PDT - Processing dimension tile
  PDS - Processing dimension with dequantizing scale
"""


@dataclass()
class RmsNormQuantKernelArgs(nl.NKIObject):
    """RMS Norm Quantization Kernel arguments.

    Args:
      lower_bound (float): Non-negative float used for clipping input values and scale.
      norm_type (NormType): Normalization type to use [RMS_NORM, NO_NORM]
      eps (float): Epsilon value for numerical stability, model hyperparameter

    """

    lower_bound: float
    norm_type: NormType = NormType.RMS_NORM
    eps: float = 1e-6

    def __post_init__(self):
        kernel_assert(
            self.norm_type == NormType.NO_NORM or self.norm_type == NormType.RMS_NORM,
            f"{self.norm_type.name} normalization is not supported",
        )
        kernel_assert(
            self.lower_bound >= 0,
            f"Lower bound must be positive but got {self.lower_bound}",
        )
        kernel_assert(self.eps >= 0, f"Epsilon must be positive but got {self.eps}")

    def needs_rms_normalization(self) -> bool:
        return is_rms_normalization(self.norm_type)

    def has_lower_bound(self) -> bool:
        return self.lower_bound is not None and self.lower_bound > 0.000001


@nki.jit
def rmsnorm_quant_kernel(hidden, ln_w, kargs: RmsNormQuantKernelArgs):
    """Entrypoint NKI kernel that performs one of the following:
        (1) perform RMSNorm and quantize the normalized hidden over the hidden dimension (H, or axis=-1).
        (2) quantize hidden over dimension H.

    The kernel supports no specialization, or specialization along 1 dimension (1D SPMD grid).

    Args:
      hidden (nl.ndarray): Input hidden states in [B, S, H] layout
      ln_w (nl.ndarray): Gamma multiplicative bias vector with [H] or [1, H] layout
      kargs (RmsNormQuantKernelArgs): See docstring for arguments

    Returns:
      Output tensor of shape [B, S, H + 4] on HBM. out[:, :, :H] of shape [B, S, H] stores the possibly
        normalized, and quantized tensor. out[:, :, H:] of shape [B, S, 4] stores 4 fp8 floats (for each unique
        batch and sequence length index) which can be reinterpreted as a fp32 dequantization scale.

    NOTE:
        The autocast argument may NOT be respected properly.
    """
    # Either we aren't sharding (grid_ndim == 0) or we can shard on a single dimension (grid_ndim == 1)
    get_verified_program_sharding_info("rmsnorm_quant", (0, 1))

    # We can handle any input by processing along its last dimension and reshaping to collapse all other dimensions into one.
    # This way, we don't have to be so specific that we require inputs with shape [B, S, H] for example.
    tsr_proc_shape = _collapse_shape_major_dimensions(hidden.shape)
    # Build data structures with info that we need throughout the kernel
    tile_info = build_rms_norm_quant_tile_info(tsr_proc_shape)
    constants = build_rms_norm_quant_constants(tile_info, kargs.eps, tsr_proc_shape)

    _validate_kernel_input(hidden, ln_w, kargs, constants)

    # Create the output tensor with the same shape as the input tensor but with the
    # innermost dimension extended to hold the dequantizing scale factors.
    out_tsr_shape = tuple(list(hidden.shape)[:-1] + [hidden.shape[-1] + constants.dequant_scale_size])
    out_tsr_proc_shape = _collapse_shape_major_dimensions(out_tsr_shape)
    out_tsr_hbm = nl.ndarray(out_tsr_shape, dtype=constants.quant_data_type, buffer=nl.shared_hbm)

    # Set the input and output shapes up based on an outer dimension and the dimension that we process along
    in_tsr_hbm_view = hidden.reshape(tsr_proc_shape)
    out_tsr_hbm_view = out_tsr_hbm.reshape(out_tsr_proc_shape)

    # Check if we were launched for single program multiple data and if so, call the code to do the appropriate sharding.
    # Otherwise, just do a single invocation of the kernel to process all the data.
    if is_launched_as_spmd():
        _rmsnorm_quant_sharded_kernel(kargs, in_tsr_hbm_view, ln_w, out_tsr_hbm_view)
    else:
        _rmsnorm_quant_single_core_kernel(kargs, tile_info, constants, in_tsr_hbm_view, ln_w, out_tsr_hbm_view)

    return out_tsr_hbm


def _validate_kernel_input(
    hidden,
    ln_w,
    kargs: RmsNormQuantKernelArgs,
    constants: RMSNormQuantConstants,
):
    # We need at least 2 dimensions on the input.  If there are N dimensions, the outer dimensions are considered to be
    # the first N - 1 dimensions and the dimension we process on is the Nth dimension (i.e. the last dimension).
    kernel_assert(
        len(hidden.shape) >= 2,
        f"Rank of hidden must be at least 2 but got {len(hidden.shape)}",
    )

    if kargs.needs_rms_normalization():
        # The shape of the RMS norm gamma tensor needs to either be [N] or [1, N] where
        # N is the size of the processing dimension.
        kernel_assert(
            len(ln_w.shape) == 1 or len(ln_w.shape) == 2,
            f"Rank of ln_w must be 1 or 2 but got {len(ln_w.shape)}",
        )
        kernel_assert(
            ln_w.shape[-1] == constants.proc_dim_size,
            f"ln_w vector length must equal {constants.proc_dim_size} but got {ln_w.shape[-1]}",
        )

    # Validate hidden tensor dimensions
    # Check that input tensor has at least 3 dimensions for [B, S, H] layout
    if len(hidden.shape) >= 3:
        # Validate batch dimension
        kernel_assert(
            hidden.shape[0] <= constants.MAX_B,
            f"Batch dimension {hidden.shape[0]} exceeds maximum {constants.MAX_B}",
        )
        # Validate sequence length dimension
        kernel_assert(
            hidden.shape[1] <= constants.MAX_S,
            f"Sequence length {hidden.shape[1]} exceeds maximum {constants.MAX_S}",
        )
        # Validate hidden dimension
        kernel_assert(
            hidden.shape[2] <= constants.MAX_H,
            f"Hidden dimension {hidden.shape[2]} exceeds maximum {constants.MAX_H}",
        )
    elif len(hidden.shape) == 2:
        # For 2D inputs, validate as [outer_dim, processing_dim]
        # The outer dimension is the product of all dimensions except the last
        outer_dim_size = hidden.shape[0]
        processing_dim_size = hidden.shape[1]

        kernel_assert(
            processing_dim_size <= constants.MAX_H,
            f"Processing dimension {processing_dim_size} exceeds maximum {constants.MAX_H}",
        )

        max_outer_dim = constants.MAX_B * constants.MAX_S
        kernel_assert(
            outer_dim_size <= max_outer_dim,
            f"Outer dimension {outer_dim_size} exceeds maximum {max_outer_dim} (MAX_B * MAX_S)",
        )

    # Validate ln_w tensor dimensions when RMS normalization is needed
    if kargs.needs_rms_normalization():
        if len(ln_w.shape) == 2:
            kernel_assert(
                ln_w.shape[0] <= 1,
                f"ln_w first dimension {ln_w.shape[0]} must be at most 1",
            )
            kernel_assert(
                ln_w.shape[1] <= constants.MAX_H,
                f"ln_w last dimension {ln_w.shape[1]} exceeds maximum {constants.MAX_H}",
            )
        elif len(ln_w.shape) == 1:
            kernel_assert(
                ln_w.shape[0] <= constants.MAX_H,
                f"ln_w dimension {ln_w.shape[0]} exceeds maximum {constants.MAX_H}",
            )


# Return the product of elements in x
def _local_prod(x: tuple[int, ...]) -> int:
    result = x[0]
    for i in range(len(x) - 1):
        result = result * x[i + 1]
    return result


def _collapse_shape_major_dimensions(shape: tuple[int, ...]) -> tuple[int, int]:
    """
    We process along the last dimension only so just multiply all dimensions leading up
    to the last one together.  Our processing loop can just traverse this resulting outer
    dimension and process each vector (i.e. the last dimension of the input).
    """
    return (_local_prod(shape[:-1]), shape[-1])


def _load_input_tensor_tile(
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    in_tsr_hbm,
    outer_dim_tile_num: int,
    output_tile_sbuf,
):
    """
    Load a single tile from the input tensor in HBM into SBUF ensuring that we handle the
    case where the outer dimension is not a multiple of the tile size.  The shapes are:
      in_tsr_hbm       - [OD, PD]
      output_tile_sbuf - [ODT, PD]
    NOTE: We don't need special handling for the processing dimension here because we aren't tiling on that
          dimension when loading.
    """
    outer_dim_offset = tile_info.outer_dim_tile.tile_size * outer_dim_tile_num
    num_p = min(tile_info.outer_dim_tile.tile_size, constants.outer_dim_size - outer_dim_offset)

    nisa.dma_copy(
        dst=output_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
        src=in_tsr_hbm[outer_dim_offset : outer_dim_offset + num_p, 0 : constants.proc_dim_size],
    )


def _store_tensor_tile_and_dequant_scales(
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    outer_dim_tile_num: int,
    in_tile_sbuf,
    dequant_scales_tile_sbuf,
    out_tsr_hbm,
):
    """
    Store a single computed tile into the output tensor in HBM from SBUF ensuring that we handle the
    case where the outer dimension is not a multiple of the tile size.  Each of the last dimensions in the
    output tensor is structured like this:
    -----------------------------------
    | ... Computed elements ... | DQS |
    -----------------------------------
    | ... Computed elements ... | DQS |
    -----------------------------------
    |              .                  |
                   .
    |              .                  |
    -----------------------------------
    | ... Computed elements ... | DQS |
    -----------------------------------

    Where DQS is the dequantizing scale for the computed elements in a given last dimension vector.

    The shapes are:
    in_tile_sbuf             - [ODT, PD]
    dequant_scales_tile_sbuf - [ODT, 1]
    out_tsr_hbm              - [OD, PDS]

    NOTE: We don't need special handling for the processing dimension here because we aren't tiling on that
          dimension when storing.
    """
    outer_dim_offset = tile_info.outer_dim_tile.tile_size * outer_dim_tile_num
    num_p = min(tile_info.outer_dim_tile.tile_size, constants.outer_dim_size - outer_dim_offset)

    # Store the quantized data
    nisa.dma_copy(
        dst=out_tsr_hbm[outer_dim_offset : outer_dim_offset + num_p, 0 : constants.proc_dim_size],
        src=in_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
    )

    # Store the dequantization weights
    nisa.dma_copy(
        dst=out_tsr_hbm[
            outer_dim_offset : outer_dim_offset + num_p,
            constants.proc_dim_size : constants.proc_dim_size + constants.dequant_scale_size,
        ],
        src=dequant_scales_tile_sbuf.ap(
            pattern=[
                [constants.dequant_scale_size, num_p],
                [1, constants.dequant_scale_size],
            ],
            dtype=nl.float8_e4m3,
        ),
    )


def _rms_normalize_tile(
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    outer_dim_tile_num: int,
    in_tile_sbuf,
    gamma_hbm,
    squared_in_tsr_sbuf,
    inverse_rms_scale_sbuf,
):
    """
    Compute RMS normalization along the last dimension for a tile.  The shapes are:
      in_tile_sbuf           - [ODT, PD]
      gamma_hbm              - [1, PD]
      squared_in_tsr_sbuf    - [ODT, PD]
      inverse_rms_scale_sbuf - [ODT, 1]

    NOTE: The computation is stored in place in in_tile_sbuf.
    """
    # Alias these to cut down on the verbosity in the code
    outer_dim_tile_size = tile_info.outer_dim_tile.tile_size
    proc_dim_tile_size = tile_info.proc_dim_tile.tile_size
    outer_dim_offset = tile_info.outer_dim_tile.tile_size * outer_dim_tile_num
    num_p = min(outer_dim_tile_size, constants.outer_dim_size - outer_dim_offset)

    # Load the gamma values
    gamma_loaded_sbuf = nl.ndarray((gamma_hbm.shape[0], gamma_hbm.shape[1]), gamma_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=gamma_loaded_sbuf[0 : gamma_hbm.shape[0], 0 : gamma_hbm.shape[1]],
        src=gamma_hbm[0 : gamma_hbm.shape[0], 0 : gamma_hbm.shape[1]],
    )

    # Find the sum of the squares of all elements in the processing dimension and store that in inverse_rms_scale_sbuf.
    # NOTE: squared_in_tsr_sbuf is each individual element squared.  We don't use that result data but have to provide storage for it.
    nisa.activation_reduce(
        dst=squared_in_tsr_sbuf[0:num_p, 0 : constants.proc_dim_size],
        op=nl.square,
        data=in_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
        reduce_op=nl.add,
        reduce_res=inverse_rms_scale_sbuf[0:num_p, 0:1],
        bias=constants.outer_dim_tile_zero_bias_vector_sbuf[0:num_p, 0:1],
        scale=1.0,
    )
    # Calculate the reciprocal of the RMS being sure to add epsilon for numerical stability.  Store the result back in inverse_rms_scale_sbuf.
    nisa.activation(
        dst=inverse_rms_scale_sbuf[0:num_p, 0:1],
        op=nl.rsqrt,
        data=inverse_rms_scale_sbuf[0:num_p, 0:1],
        bias=constants.rmsn_eps_bias_sbuf[0:num_p, 0:1],
        scale=1 / constants.proc_dim_size,
    )

    # Optimized processing: divide iterations into batches of 8
    n_iters = tile_info.proc_dim_tile.tile_count
    q, k = divmod(n_iters, 8)

    for i_q in range(q):
        psum_batch = nl.ndarray(shape=(num_p, proc_dim_tile_size * 8), dtype=nl.float32, buffer=nl.psum)
        for i_8 in range(8):
            proc_dim_tile_num = i_q * 8 + i_8
            proc_dim_offset = tile_info.proc_dim_tile.tile_size * proc_dim_tile_num
            num_f = min(proc_dim_tile_size, constants.proc_dim_size - proc_dim_offset)

            # Get the slice for this specific buffer within the batch
            broadcasted_gamma_psum = psum_batch[0:num_p, i_8 * proc_dim_tile_size : (i_8 + 1) * proc_dim_tile_size]

            # Matrix multiply the ones vector by the gamma values to get the values broadcast across all partitions
            nisa.nc_matmul(
                dst=broadcasted_gamma_psum[0:num_p, 0:num_f],
                stationary=constants.pe_broadcast_ones_vector_sbuf[
                    0 : constants.pe_broadcast_ones_vector_sbuf.shape[0], 0:num_p
                ],
                moving=gamma_loaded_sbuf[0:1, proc_dim_offset : proc_dim_offset + num_f],
            )

            # Apply gamma and inverse_rms_scale
            nisa.scalar_tensor_tensor(
                dst=in_tile_sbuf[0:num_p, proc_dim_offset : proc_dim_offset + num_f],
                data=in_tile_sbuf[0:num_p, proc_dim_offset : proc_dim_offset + num_f],
                op0=nl.multiply,
                operand0=inverse_rms_scale_sbuf[0:num_p, 0:1],
                op1=nl.multiply,
                operand1=broadcasted_gamma_psum[0:num_p, 0:num_f],
            )

    # Handle remainder tiles (if k > 0)
    if k > 0:
        # For remainder, we still need a psum buffer but only for k tiles
        psum_remainder = nl.ndarray(shape=(num_p, proc_dim_tile_size * k), dtype=nl.float32, buffer=nl.psum)

        for i_k in range(k):
            proc_dim_tile_num = q * 8 + i_k
            proc_dim_offset = tile_info.proc_dim_tile.tile_size * proc_dim_tile_num
            num_f = min(proc_dim_tile_size, constants.proc_dim_size - proc_dim_offset)

            # Get the slice for this specific buffer within the remainder
            broadcasted_gamma_psum = psum_remainder[0:num_p, i_k * proc_dim_tile_size : (i_k + 1) * proc_dim_tile_size]

            # Matrix multiply the ones vector by the gamma values
            nisa.nc_matmul(
                dst=broadcasted_gamma_psum[0:num_p, 0:num_f],
                stationary=constants.pe_broadcast_ones_vector_sbuf[
                    0 : constants.pe_broadcast_ones_vector_sbuf.shape[0], 0:num_p
                ],
                moving=gamma_loaded_sbuf[0:1, proc_dim_offset : proc_dim_offset + num_f],
            )

            # Apply gamma and inverse_rms_scale
            nisa.scalar_tensor_tensor(
                dst=in_tile_sbuf[0:num_p, proc_dim_offset : proc_dim_offset + num_f],
                data=in_tile_sbuf[0:num_p, proc_dim_offset : proc_dim_offset + num_f],
                op0=nl.multiply,
                operand0=inverse_rms_scale_sbuf[0:num_p, 0:1],
                op1=nl.multiply,
                operand1=broadcasted_gamma_psum[0:num_p, 0:num_f],
            )


def _quantize_tile(
    kargs: RmsNormQuantKernelArgs,
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    outer_dim_tile_num: int,
    in_tile_sbuf,
    out_tile_sbuf,
    out_dequant_scales_sbuf,
):
    """
    Compute quantization and dequantization scales along the last dimension for a tile.  The shapes are:
      in_tile_sbuf            - [ODT, PD]
      out_tile_sbuf           - [ODT, PD]
      out_dequant_scales_sbuf - [ODT, 1]
    """
    # Alias these to cut down on the verbosity in the code
    outer_dim_tile_size = tile_info.outer_dim_tile.tile_size
    outer_dim_offset = tile_info.outer_dim_tile.tile_size * outer_dim_tile_num
    num_p = min(outer_dim_tile_size, constants.outer_dim_size - outer_dim_offset)

    # abs_tile_sbuf stores the absolute values of the tile being processed.  We don't end up using these values but need a place to store them.
    # out_dequant_scales_sbuf stores abs_group reduced over abs_group's last dimension (i.e. the processing dimension)
    # to get the maximum absolute value.
    abs_tile_sbuf = nl.ndarray(
        (num_p, constants.proc_dim_size),
        dtype=constants.compute_data_type,
        buffer=nl.sbuf,
    )
    nisa.tensor_scalar_reduce(
        dst=abs_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
        data=in_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
        op0=nl.abs,
        operand0=0.0,
        reduce_op=nl.maximum,
        reduce_res=out_dequant_scales_sbuf[0:num_p, 0:1],
    )

    if kargs.has_lower_bound():
        # Clip out_dequant_scales_sbuf to the range [0, lower_bound].
        nisa.tensor_scalar(
            dst=out_dequant_scales_sbuf[0:num_p, 0:1],
            data=out_dequant_scales_sbuf[0:num_p, 0:1],
            op0=nl.minimum,
            operand0=kargs.lower_bound,
        )

        # Clip the tile being processed in range [-lower_bound, lower_bound].
        nisa.tensor_scalar(
            dst=in_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
            data=in_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
            op0=nl.minimum,
            operand0=kargs.lower_bound,
            op1=nl.maximum,
            operand1=-kargs.lower_bound,
        )

    # Compute absolute maximum / _FP8_RANGE along each processing dimension to get the dequantization scales.
    nisa.activation(
        dst=out_dequant_scales_sbuf[0:num_p, 0:1],
        op=nl.copy,
        data=out_dequant_scales_sbuf[0:num_p, 0:1],
        scale=1 / constants.quant_data_type_range,
        bias=constants.outer_dim_tile_zero_bias_vector_sbuf[0:num_p, 0:1],
    )

    # Clamp out_dequant_scales_sbuf to _MIN_DEQUANT_SCALE_VAL for numerical stability reasons.  Basically, it keeps tiny
    # values from exploding in size when we take the reciprocal of them.
    nisa.tensor_scalar(
        dst=out_dequant_scales_sbuf[0:num_p, 0:1],
        data=out_dequant_scales_sbuf[0:num_p, 0:1],
        op0=nl.maximum,
        operand0=constants.min_dequant_scale_value,
    )

    # Get the reciprocol of our dequantization scales to get the quantization scales
    quant_scales_sbuf = nl.ndarray((num_p, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(
        dst=quant_scales_sbuf[0:num_p, 0:1],
        data=out_dequant_scales_sbuf[0:num_p, 0:1],
    )

    # Apply quantization scales to get the quantized result.
    nisa.tensor_scalar(
        dst=out_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
        data=in_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
        op0=nl.multiply,
        operand0=quant_scales_sbuf[0:num_p, 0:1],
        engine=nisa.vector_engine,
    )


def _rmsnorm_quant_single_core_kernel(
    kargs: RmsNormQuantKernelArgs,
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    in_tsr_hbm,
    rmsn_gamma_hbm,
    out_tsr_hbm,
):
    """
    Process all tiles along the outer dimension by applying optional RMS normalization, quantizing the data, and storing the results.
    The shapes are:
      in_tsr_hbm     - [OD, PD]
      rmsn_gamma_hbm - [1, PD] or [PD]
      out_tsr_hbm    - [OD, PDS]

      Where DIM1 - DIMN are the major dimensions that ultimately are combined to be the outer dimension in this kernel.
    """
    if kargs.needs_rms_normalization():
        # Ensure the shape that the code requires
        rmsn_gamma_hbm_view = rmsn_gamma_hbm.reshape((1, constants.proc_dim_size))

    # Loop over all the tiles in the outer dimension and process them one at a time
    for outer_tile_num in range(tile_info.outer_dim_tile.tile_count):
        if kargs.needs_rms_normalization():
            # Declare some allocations required throughout RMS norm calculations
            squared_in_tsr_sbuf = nl.ndarray(
                (tile_info.outer_dim_tile.tile_size, constants.proc_dim_size),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            inverse_rms_scale_sbuf = nl.ndarray(
                (tile_info.outer_dim_tile.tile_size, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )

        quant_tile_sbuf = nl.ndarray(
            (tile_info.outer_dim_tile.tile_size, constants.proc_dim_size),
            dtype=constants.quant_data_type,
            buffer=nl.sbuf,
        )

        in_tile_sbuf = nl.ndarray(
            (tile_info.outer_dim_tile.tile_size, constants.proc_dim_size),
            dtype=constants.compute_data_type,
            buffer=nl.sbuf,
        )
        dequant_scales_tile_sbuf = nl.ndarray((tile_info.outer_dim_tile.tile_size, 1), dtype=nl.float32, buffer=nl.sbuf)

        # Conceptually, our loop is simple:
        #   - Load a tile into SBUF
        #   - Apply RMS normalization if the caller has elected to do so
        #   - Quantize the tile
        #   - Store the resulting quantized tile data along with the associated dequantization scales into HBM
        _load_input_tensor_tile(tile_info, constants, in_tsr_hbm, outer_tile_num, in_tile_sbuf)

        if kargs.needs_rms_normalization():
            _rms_normalize_tile(
                tile_info,
                constants,
                outer_tile_num,
                in_tile_sbuf,
                rmsn_gamma_hbm_view,
                squared_in_tsr_sbuf,
                inverse_rms_scale_sbuf,
            )

        _quantize_tile(
            kargs,
            tile_info,
            constants,
            outer_tile_num,
            in_tile_sbuf,
            quant_tile_sbuf,
            dequant_scales_tile_sbuf,
        )

        _store_tensor_tile_and_dequant_scales(
            tile_info,
            constants,
            outer_tile_num,
            quant_tile_sbuf,
            dequant_scales_tile_sbuf,
            out_tsr_hbm,
        )


def _rmsnorm_quant_sharded_kernel(kargs: RmsNormQuantKernelArgs, in_tsr_hbm, rmsn_gamma_hbm, out_tsr_hbm):
    """
    We support a 1D launch grid only (or no launch grid at all).  If there is a dimension in the launch grid then however many programs
    there are in that dimension dictates our number of shards since we will launch the kernel once per program and each launch of the
    kernel needs a shard to process.

    We shard along the outer dimension since each processing dimension vector can be processed independently.  We calculate references
    to the original input and output tensors based on the sharding and then pass those references into the core of the kernel.
    """
    outer_dim, _ = in_tsr_hbm.shape
    _, num_shards, shard_id = get_program_sharding_info()
    nominal_shard_size = outer_dim // num_shards

    kernel_assert(
        nominal_shard_size > 0,
        f"Outer dimension of size {outer_dim} is not big enough to distribute work among {num_shards} programs. Reduce SPMD launch grid, so that each program gets at least a single shard to work on",
    )

    # Allow outer dimensions that are not a multiple of the number of shards.  For the last shard,
    # we have to calculate the remaining size for the shard.  Otherwise, it is just the
    # nominal size.
    if shard_id == num_shards - 1:
        shard_size = outer_dim - nominal_shard_size * (num_shards - 1)
    else:
        shard_size = nominal_shard_size
    shard_offset = shard_id * nominal_shard_size

    outer_dim_idx = nl.ds(shard_offset, shard_size)
    in_tile_shard_hbm = in_tsr_hbm[outer_dim_idx, :]
    out_tile_shard_hbm = out_tsr_hbm[outer_dim_idx, :]
    # Figure out the shape in terms of [outer dim, processing dim]
    tsr_proc_shape = _collapse_shape_major_dimensions(in_tile_shard_hbm.shape)
    # Build data structures with info that we need throughout the kernel
    tile_info = build_rms_norm_quant_tile_info(tsr_proc_shape)
    constants = build_rms_norm_quant_constants(tile_info, kargs.eps, tsr_proc_shape)

    return _rmsnorm_quant_single_core_kernel(
        kargs,
        tile_info,
        constants,
        in_tile_shard_hbm,
        rmsn_gamma_hbm,
        out_tile_shard_hbm,
    )
