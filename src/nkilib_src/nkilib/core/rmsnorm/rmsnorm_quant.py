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

"""8-bit Quantization and RMS Normalization Kernel for NKI. Performs optional RMS normalization followed by FP8 quantization along the last dimension."""

from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.common_types import NormType, QuantizationType
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


@dataclass()
class RmsNormQuantKernelArgs(nl.NKIObject):
    """
    RMS Norm Quantization Kernel arguments.

    Args:
        lower_bound (float): Non-negative float for clipping input values and scale (row quant)
        norm_type (NormType): Normalization type [RMS_NORM, NO_NORM]
        quantization_type (QuantizationType): Quantization type [ROW, STATIC]
        eps (float): Epsilon value for numerical stability
    """

    lower_bound: float = 0.0
    norm_type: NormType = NormType.RMS_NORM
    quantization_type: QuantizationType = QuantizationType.ROW
    eps: float = 1e-6

    def __post_init__(self):
        kernel_assert(
            self.quantization_type in (QuantizationType.ROW, QuantizationType.STATIC),
            f"{self.quantization_type.name} quantization is not supported",
        )
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
        return self.lower_bound != None and self.lower_bound > 0.000001

    def is_row_quant(self) -> bool:
        return self.quantization_type == QuantizationType.ROW

    def is_static_quant(self) -> bool:
        return self.quantization_type == QuantizationType.STATIC


@nki.jit
def rmsnorm_quant_kernel(
    hidden: nl.ndarray,
    ln_w: nl.ndarray,
    kargs: RmsNormQuantKernelArgs,
    input_dequant_scale: nl.ndarray = None,
) -> nl.ndarray:
    """
    Perform optional RMS normalization followed by FP8 quantization along the last dimension.

    This kernel performs quantization down to 8 bits along the last dimension of the input tensor.
    For each last dimension vector, the maximum absolute value is found and used with the FP8 range
    to scale values. RMS normalization is optional and computed prior to quantization.
    TODO: Specify intended usage range (e.g., sequence length, batch size).

    Dimensions:
        B: Batch size (first dimension of input)
        S: Sequence length (second dimension of input)
        H: Hidden dimension (last dimension, processing dimension)
        OD: Outer dimension (B * S when collapsed)
        PD: Processing dimension (H)

    Args:
        hidden (nl.ndarray): [B, S, H], Input hidden states tensor on HBM
        ln_w (nl.ndarray): [H] or [1, H], Gamma multiplicative bias vector for RMS norm
        kargs (RmsNormQuantKernelArgs): Kernel configuration arguments
        input_dequant_scale (nl.ndarray): [128, 1], Input dequantization scale for static quant

    Returns:
        output (nl.ndarray): [B, S, H+4] for row quant or [B, S, H] for static quant,
            Quantized output tensor on HBM. For row quant, last 4 elements per row
            store the fp32 dequantization scale as 4 fp8 values.

    Notes:
        - Supports no specialization or 1D SPMD grid sharding
        - The autocast argument may NOT be respected properly
        - Input tensor must have at least 2 dimensions
        - For row quant: dequant scale stored as 4 fp8 values reinterpretable as fp32

    Pseudocode:
        # Reshape input to [OD, PD] where OD = B*S, PD = H
        for outer_tile_idx in range(num_outer_tiles):
            tile = load_tile(hidden, outer_tile_idx)
            if needs_rms_norm:
                tile = rms_normalize(tile, ln_w, eps)
            if row_quant:
                quant_tile, dequant_scale = row_quantize(tile)
            else:
                quant_tile = static_quantize(tile, input_scale)
            store_tile(output, quant_tile, dequant_scale)
    """
    # Either we aren't sharding (grid_ndim == 0) or we can shard on a single dimension (grid_ndim == 1)
    get_verified_program_sharding_info("rmsnorm_quant", (0, 1))

    # We can handle any input by processing along its last dimension and reshaping to collapse all other dimensions into one.
    # This way, we don't have to be so specific that we require inputs with shape [B, S, H] for example.
    tsr_proc_shape = _collapse_shape_major_dimensions(hidden.shape)
    # Build data structures with info that we need throughout the kernel
    tile_info = build_rms_norm_quant_tile_info(tsr_proc_shape)
    constants = build_rms_norm_quant_constants(tile_info, kargs.eps, tsr_proc_shape)

    _validate_kernel_input(hidden, ln_w, input_dequant_scale, kargs, constants)

    if kargs.is_row_quant():
        # Create the output tensor with the same shape as the input tensor but with the
        # innermost dimension extended to hold the dequantizing scale factors.
        out_tsr_shape = tuple(list(hidden.shape)[:-1] + [hidden.shape[-1] + constants.dequant_scale_size])
    elif kargs.is_static_quant():
        out_tsr_shape = hidden.shape
    out_tsr_proc_shape = _collapse_shape_major_dimensions(out_tsr_shape)
    out_tsr_hbm = nl.ndarray(out_tsr_shape, dtype=constants.quant_data_type, buffer=nl.shared_hbm)

    # Set the input and output shapes up based on an outer dimension and the dimension that we process along
    in_tsr_hbm_view = hidden.reshape(tsr_proc_shape)
    out_tsr_hbm_view = out_tsr_hbm.reshape(out_tsr_proc_shape)

    # Check if we were launched for single program multiple data and if so, call the code to do the appropriate sharding.
    # Otherwise, just do a single invocation of the kernel to process all the data.
    if is_launched_as_spmd():
        _rmsnorm_quant_sharded_kernel(kargs, in_tsr_hbm_view, ln_w, input_dequant_scale, out_tsr_hbm_view)
    else:
        _rmsnorm_quant_single_core_kernel(
            kargs, tile_info, constants, in_tsr_hbm_view, ln_w, input_dequant_scale, out_tsr_hbm_view
        )

    return out_tsr_hbm


def _validate_kernel_input(
    hidden: nl.ndarray,
    ln_w: nl.ndarray,
    input_sc: nl.ndarray,
    kargs: RmsNormQuantKernelArgs,
    constants: RMSNormQuantConstants,
) -> None:
    """
    Validate all input parameters for the RMS norm quantization kernel.

    Args:
        hidden (nl.ndarray): Input hidden states tensor
        ln_w (nl.ndarray): Gamma multiplicative bias vector
        input_sc (nl.ndarray): Input dequantization scale for static quant
        kargs (RmsNormQuantKernelArgs): Kernel configuration arguments
        constants (RMSNormQuantConstants): Kernel constants

    Notes:
        - Input tensor must have at least 2 dimensions
        - For static quant, input_sc must be provided with shape (128, 1)
        - For RMS norm, ln_w shape must be [H] or [1, H]
    """
    # We need at least 2 dimensions on the input.  If there are N dimensions, the outer dimensions are considered to be
    # the first N - 1 dimensions and the dimension we process on is the Nth dimension (i.e. the last dimension).
    kernel_assert(
        len(hidden.shape) >= 2,
        f"Rank of hidden must be at least 2 but got {len(hidden.shape)}",
    )

    if kargs.is_static_quant():
        kernel_assert(input_sc != None, "input_dequant_scale must be provided for static quantization")
        kernel_assert(
            input_sc.shape == (nl.tile_size.pmax, 1),
            f"static input dequant scale shape must be broadcasted to (128, 1) but got {input_sc.shape}",
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


def _local_prod(x: tuple[int, ...]) -> int:
    """Return the product of all elements in the tuple."""
    result = x[0]
    for idx in range(len(x) - 1):
        result = result * x[idx + 1]
    return result


def _collapse_shape_major_dimensions(shape: tuple[int, ...]) -> tuple[int, int]:
    """
    Collapse all dimensions except the last into a single outer dimension.

    Multiplies all dimensions leading up to the last one together. The processing
    loop can traverse this resulting outer dimension and process each vector.
    """
    return (_local_prod(shape[:-1]), shape[-1])


def _load_and_invert_static_dequant_scale(
    in_scale_hbm: nl.ndarray,
    in_scale_sbuf: nl.ndarray,
) -> None:
    """
    Load input dequantization scale from HBM and compute its reciprocal.

    The reciprocal of the dequantization scale is the quantization scale.
    Input is a scalar value broadcasted to [128, 1] on HBM.

    Args:
        in_scale_hbm (nl.ndarray): [128, 1], Input dequant scale on HBM
        in_scale_sbuf (nl.ndarray): [128, 1], Output buffer in SBUF for quant scale
    """
    nisa.dma_copy(src=in_scale_hbm[: nl.tile_size.pmax, :1], dst=in_scale_sbuf[: nl.tile_size.pmax, :1])
    nisa.reciprocal(data=in_scale_sbuf[: nl.tile_size.pmax, :1], dst=in_scale_sbuf[: nl.tile_size.pmax, :1])


def _load_input_tensor_tile(
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    in_tsr_hbm: nl.ndarray,
    outer_dim_tile_num: int,
    output_tile_sbuf: nl.ndarray,
) -> None:
    """
    Load a single tile from the input tensor in HBM into SBUF.

    Handles the case where the outer dimension is not a multiple of the tile size.

    Args:
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        constants (RMSNormQuantConstants): Kernel constants
        in_tsr_hbm (nl.ndarray): [OD, PD], Input tensor on HBM
        outer_dim_tile_num (int): Current outer dimension tile index
        output_tile_sbuf (nl.ndarray): [ODT, PD], Output buffer in SBUF

    Notes:
        - No special handling needed for processing dimension (not tiled when loading)
    """
    outer_dim_offset = tile_info.outer_dim_tile.tile_size * outer_dim_tile_num
    num_p = min(tile_info.outer_dim_tile.tile_size, constants.outer_dim_size - outer_dim_offset)

    nisa.dma_copy(
        dst=output_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
        src=in_tsr_hbm[outer_dim_offset : outer_dim_offset + num_p, 0 : constants.proc_dim_size],
    )


def _store_tensor_tile_and_dequant_scales(
    kargs: RmsNormQuantKernelArgs,
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    outer_dim_tile_num: int,
    in_tile_sbuf: nl.ndarray,
    dequant_scales_tile_sbuf: nl.ndarray,
    out_tsr_hbm: nl.ndarray,
) -> None:
    """
    Store computed tile and dequantization scales to output tensor in HBM.

    Each last dimension in the output tensor is structured as:
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
    where DQS is the dequantizing scale.

    Args:
        kargs (RmsNormQuantKernelArgs): Kernel configuration arguments
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        constants (RMSNormQuantConstants): Kernel constants
        outer_dim_tile_num (int): Current outer dimension tile index
        in_tile_sbuf (nl.ndarray): [ODT, PD], Quantized tile in SBUF
        dequant_scales_tile_sbuf (nl.ndarray): [ODT, 1], Dequant scales in SBUF
        out_tsr_hbm (nl.ndarray): [OD, PDS], Output tensor on HBM

    Notes:
        - No special handling needed for processing dimension (not tiled when storing)
    """
    outer_dim_offset = tile_info.outer_dim_tile.tile_size * outer_dim_tile_num
    num_p = min(tile_info.outer_dim_tile.tile_size, constants.outer_dim_size - outer_dim_offset)

    # Store the quantized data
    nisa.dma_copy(
        dst=out_tsr_hbm[outer_dim_offset : outer_dim_offset + num_p, 0 : constants.proc_dim_size],
        src=in_tile_sbuf[0:num_p, 0 : constants.proc_dim_size],
    )

    if kargs.is_row_quant():
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
    in_tile_sbuf: nl.ndarray,
    gamma_hbm: nl.ndarray,
    squared_in_tsr_sbuf: nl.ndarray,
    inverse_rms_scale_sbuf: nl.ndarray,
) -> None:
    """
    Compute RMS normalization along the last dimension for a tile.

    Args:
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        constants (RMSNormQuantConstants): Kernel constants
        outer_dim_tile_num (int): Current outer dimension tile index
        in_tile_sbuf (nl.ndarray): [ODT, PD], Input tile in SBUF (modified in place)
        gamma_hbm (nl.ndarray): [1, PD], Gamma values on HBM
        squared_in_tsr_sbuf (nl.ndarray): [ODT, PD], Scratch buffer for squared values
        inverse_rms_scale_sbuf (nl.ndarray): [ODT, 1], Buffer for inverse RMS scale

    Notes:
        - Computation is stored in place in in_tile_sbuf
        - Uses PE to broadcast gamma across partition dimension
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
    batch_count, remainder_count = divmod(n_iters, 8)

    for batch_idx in range(batch_count):
        for inner_idx in range(8):
            proc_dim_tile_num = batch_idx * 8 + inner_idx
            proc_dim_offset = tile_info.proc_dim_tile.tile_size * proc_dim_tile_num
            num_f = min(proc_dim_tile_size, constants.proc_dim_size - proc_dim_offset)

            # Allocate PSUM buffer for this tile only
            broadcasted_gamma_psum = nl.ndarray(shape=(num_p, proc_dim_tile_size), dtype=nl.float32, buffer=nl.psum)

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

    # Handle remainder tiles (if remainder_count > 0)
    if remainder_count > 0:
        for remainder_idx in range(remainder_count):
            proc_dim_tile_num = batch_count * 8 + remainder_idx
            proc_dim_offset = tile_info.proc_dim_tile.tile_size * proc_dim_tile_num
            num_f = min(proc_dim_tile_size, constants.proc_dim_size - proc_dim_offset)

            # Allocate PSUM buffer for this tile only
            broadcasted_gamma_psum = nl.ndarray(shape=(num_p, proc_dim_tile_size), dtype=nl.float32, buffer=nl.psum)

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


def _row_quantize_tile(
    kargs: RmsNormQuantKernelArgs,
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    outer_dim_tile_num: int,
    in_tile_sbuf: nl.ndarray,
    out_tile_sbuf: nl.ndarray,
    out_dequant_scales_sbuf: nl.ndarray,
) -> None:
    """
    Compute row quantization and dequantization scales along the last dimension.

    Args:
        kargs (RmsNormQuantKernelArgs): Kernel configuration arguments
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        constants (RMSNormQuantConstants): Kernel constants
        outer_dim_tile_num (int): Current outer dimension tile index
        in_tile_sbuf (nl.ndarray): [ODT, PD], Input tile in SBUF
        out_tile_sbuf (nl.ndarray): [ODT, PD], Output quantized tile in SBUF
        out_dequant_scales_sbuf (nl.ndarray): [ODT, 1], Output dequant scales in SBUF
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


def _static_quantize_tile(
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    outer_dim_tile_num: int,
    in_tile_sbuf: nl.ndarray,
    in_static_quant_scales_sbuf: nl.ndarray,
    out_tile_sbuf: nl.ndarray,
) -> None:
    """
    Compute static quantization for a tile using pre-computed scale.

    Args:
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        constants (RMSNormQuantConstants): Kernel constants
        outer_dim_tile_num (int): Current outer dimension tile index
        in_tile_sbuf (nl.ndarray): [ODT, PD], Input tile in SBUF
        in_static_quant_scales_sbuf (nl.ndarray): [128, 1], Static quant scale in SBUF
        out_tile_sbuf (nl.ndarray): [ODT, PD], Output quantized tile in SBUF
    """
    # Alias these to cut down on the verbosity in the code
    outer_dim_tile_size = tile_info.outer_dim_tile.tile_size
    outer_dim_offset = tile_info.outer_dim_tile.tile_size * outer_dim_tile_num
    num_p = min(outer_dim_tile_size, constants.outer_dim_size - outer_dim_offset)

    nisa.tensor_scalar(
        dst=in_tile_sbuf[:num_p, : constants.proc_dim_size],
        data=in_tile_sbuf[:num_p, : constants.proc_dim_size],
        op0=nl.multiply,
        operand0=in_static_quant_scales_sbuf[:num_p, :1],
        op1=nl.add,
        operand1=constants.outer_dim_tile_zero_bias_vector_sbuf[:num_p, :1],
    )
    nisa.tensor_scalar(
        dst=out_tile_sbuf[:num_p, : constants.proc_dim_size],
        data=in_tile_sbuf[:num_p, : constants.proc_dim_size],
        op0=nl.minimum,
        operand0=constants.quant_data_type_range,
        op1=nl.maximum,
        operand1=-constants.quant_data_type_range,
    )


def _quantize_tile(
    kargs: RmsNormQuantKernelArgs,
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    outer_dim_tile_num: int,
    in_tile_sbuf: nl.ndarray,
    in_static_quant_scales_sbuf: nl.ndarray,
    out_tile_sbuf: nl.ndarray,
    out_row_dequant_scales_sbuf: nl.ndarray,
) -> None:
    """
    Dispatch to row or static quantization based on kernel arguments.

    Args:
        kargs (RmsNormQuantKernelArgs): Kernel configuration arguments
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        constants (RMSNormQuantConstants): Kernel constants
        outer_dim_tile_num (int): Current outer dimension tile index
        in_tile_sbuf (nl.ndarray): [ODT, PD], Input tile in SBUF
        in_static_quant_scales_sbuf (nl.ndarray): [128, 1], Static quant scale (or None)
        out_tile_sbuf (nl.ndarray): [ODT, PD], Output quantized tile in SBUF
        out_row_dequant_scales_sbuf (nl.ndarray): [ODT, 1], Row dequant scales (or None)
    """
    if kargs.is_row_quant():
        _row_quantize_tile(
            kargs, tile_info, constants, outer_dim_tile_num, in_tile_sbuf, out_tile_sbuf, out_row_dequant_scales_sbuf
        )
    elif kargs.is_static_quant():
        _static_quantize_tile(
            tile_info,
            constants,
            outer_dim_tile_num,
            in_tile_sbuf,
            in_static_quant_scales_sbuf,
            out_tile_sbuf,
        )


def _rmsnorm_quant_single_core_kernel(
    kargs: RmsNormQuantKernelArgs,
    tile_info: RMSNormQuantTileInfo,
    constants: RMSNormQuantConstants,
    in_tsr_hbm: nl.ndarray,
    rmsn_gamma_hbm: nl.ndarray,
    in_sc_hbm: nl.ndarray,
    out_tsr_hbm: nl.ndarray,
) -> None:
    """
    Process all tiles along the outer dimension with optional RMS norm and quantization.

    Args:
        kargs (RmsNormQuantKernelArgs): Kernel configuration arguments
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        constants (RMSNormQuantConstants): Kernel constants
        in_tsr_hbm (nl.ndarray): [OD, PD], Input tensor on HBM
        rmsn_gamma_hbm (nl.ndarray): [1, PD] or [PD], RMS norm gamma on HBM
        in_sc_hbm (nl.ndarray): [128, 1], Static dequant scale on HBM (or None)
        out_tsr_hbm (nl.ndarray): [OD, PDS], Output tensor on HBM
    """
    if kargs.needs_rms_normalization():
        # Ensure the shape that the code requires
        rmsn_gamma_hbm_view = rmsn_gamma_hbm.reshape((1, constants.proc_dim_size))

    if kargs.is_static_quant():
        static_quant_scale_sbuf = nl.ndarray((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        _load_and_invert_static_dequant_scale(in_sc_hbm, static_quant_scale_sbuf)
    else:
        static_quant_scale_sbuf = None

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
        if kargs.is_row_quant():
            row_dequant_scales_tile_sbuf = nl.ndarray(
                (tile_info.outer_dim_tile.tile_size, 1), dtype=nl.float32, buffer=nl.sbuf
            )
        else:
            row_dequant_scales_tile_sbuf = None

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
            static_quant_scale_sbuf,
            quant_tile_sbuf,
            row_dequant_scales_tile_sbuf,
        )

        _store_tensor_tile_and_dequant_scales(
            kargs,
            tile_info,
            constants,
            outer_tile_num,
            quant_tile_sbuf,
            row_dequant_scales_tile_sbuf,
            out_tsr_hbm,
        )


def _rmsnorm_quant_sharded_kernel(
    kargs: RmsNormQuantKernelArgs,
    in_tsr_hbm: nl.ndarray,
    rmsn_gamma_hbm: nl.ndarray,
    in_sc_hbm: nl.ndarray,
    out_tsr_hbm: nl.ndarray,
) -> None:
    """
    Handle 1D SPMD sharding for the RMS norm quantization kernel.

    Shards along the outer dimension since each processing dimension vector can be
    processed independently. Calculates references to input/output tensors based on
    sharding and passes them to the single core kernel.

    Args:
        kargs (RmsNormQuantKernelArgs): Kernel configuration arguments
        in_tsr_hbm (nl.ndarray): [OD, PD], Input tensor on HBM
        rmsn_gamma_hbm (nl.ndarray): [1, PD] or [PD], RMS norm gamma on HBM
        in_sc_hbm (nl.ndarray): [128, 1], Static dequant scale on HBM (or None)
        out_tsr_hbm (nl.ndarray): [OD, PDS], Output tensor on HBM

    Notes:
        - Supports 1D launch grid only (or no launch grid)
        - Allows outer dimensions not divisible by number of shards
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
        in_sc_hbm,
        out_tile_shard_hbm,
    )
