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


import neuron_dtypes as dt
import nki.language as nl
import numpy as np
from neuron_dtypes import static_cast

# Mapping from nki dtype strings to neuron_dtypes dtype objects
NL_TO_DT_DTYPE = {
    nl.float8_e4m3fn_x4: dt.float8_e4m3fn_x4,
    nl.float8_e5m2_x4: dt.float8_e5m2_x4,
    nl.float4_e2m1fn_x4: dt.float4_e2m1fn_x4,
}


def is_mx_quantize(quantize):
    return quantize in [nl.float8_e5m2_x4, nl.float8_e4m3fn_x4, nl.float4_e2m1fn_x4]


# Get exponent for float32 in IEEE 754 standard
def get_float32_exp(float_data):
    man_nbits, exp_nbits = 23, 8
    return (float_data.astype(np.float32).view(np.uint32) >> man_nbits) & ((1 << exp_nbits) - 1)


# max normal
# float8_e5m2: S 11110 11 = ± 2^15 × 1.75 = ± 57,344
# float8_e4m3fn: S 1111 110 = ± 2^8 × 1.75 = ± 448
# float4_e2m1fn: S 11 1 = ± 2^2 × 1.5 = ± 6
def get_mx_fp_max(dst_dtype):
    max_values = {nl.float8_e5m2_x4: 57344, nl.float8_e4m3fn_x4: 448, nl.float4_e2m1fn_x4: 6}
    assert dst_dtype in max_values, f'no max value provided for {dst_dtype}'
    return max_values.get(dst_dtype)


def get_mx_max_exp(dst_dtype):
    max_exp_values = {nl.float8_e5m2_x4: 15, nl.float8_e4m3fn_x4: 8, nl.float4_e2m1fn_x4: 2}
    assert dst_dtype in max_exp_values, f"no max exp value provided for {dst_dtype}"
    return max_exp_values.get(dst_dtype)


def quantize_mx_golden(in_tensor, out_x4_dtype, ocp_saturation=True, reverse_dst_fdim_group=0, custom_mx_max_exp=None):
    max_exp = custom_mx_max_exp(out_x4_dtype) if custom_mx_max_exp else get_mx_max_exp(out_x4_dtype)
    max_val = get_mx_fp_max(out_x4_dtype)
    float32_exp_bias = 127

    P, F = in_tensor.shape
    SP, SF = P // 8, F // 4

    in_tensor_ = np.copy(in_tensor)

    RG = reverse_dst_fdim_group
    # reverse free dimension by a group of RG elements (keep the order within each group)
    if RG > 0:
        assert F % RG == 0
        in_tensor_ = in_tensor_.reshape(P, F // RG, RG)[:, ::-1, :].reshape(P, F)

    exp = get_float32_exp(in_tensor_)

    # Reshape exponent tensor to group by 8x4 blocks for max computation
    exp_reshaped = exp.reshape(SP, 8, SF, 4)

    # Compute max exponent for each 8x4 block using vectorized operations
    # Take max over the 8x4 dimensions (axes 1 and 3)
    mx_scale_golden = np.max(exp_reshaped, axis=(1, 3)).astype(np.uint8) - max_exp

    # Convert scale exponents to scale factors
    scale_exp = mx_scale_golden.astype(np.int32) - float32_exp_bias
    scale_factors = 2.0**scale_exp  # Shape: [SP, SF]

    # Expand scale factors to match input tensor shape using vectorized operations
    # Each scale factor applies to an 8x4 block
    scale_expanded_p = np.repeat(scale_factors, 8, axis=0)  # Shape: [P, SF]
    scale = np.repeat(scale_expanded_p, 4, axis=1)  # Shape: [P, F]

    # Quantize: divide by scale
    mx_data_golden = in_tensor_ / scale
    if ocp_saturation:
        mx_data_golden = np.clip(mx_data_golden, -max_val, max_val)
    mx_data_golden = static_cast(mx_data_golden.astype(np.float32), NL_TO_DT_DTYPE.get(out_x4_dtype, out_x4_dtype))

    return mx_data_golden, mx_scale_golden


def get_p_contiguous_scale(hw_scale, data_p_size, p_offset=0):
    if data_p_size <= 32:
        return hw_scale[p_offset : p_offset + data_p_size]

    scale = np.zeros((data_p_size // 8,) + tuple(hw_scale.shape[1:]), hw_scale.dtype)
    for i in range(data_p_size // 8):
        scale[i] = hw_scale[i // 4 * 32 + i % 4 + p_offset]

    return scale


def nc_matmul_mx_golden(
    stationary_x4,
    moving_x4,
    stationary_scale,
    moving_scale,
    use_contiguous_scale=True,
    stationary_scale_p_offset=0,
    moving_scale_p_offset=0,
):
    # Process moving tensor
    moving = static_cast(moving_x4, np.float32)
    new_shape = moving.shape[:-1] + (moving.shape[-1] // 4, 4)
    moving = moving.reshape(new_shape)
    MP, MF0, MF1 = moving.shape
    assert MF1 == 4
    moving_scale = moving_scale.astype(np.float32)
    if not use_contiguous_scale:
        # if scale follows hw layout, make it contiguous at partition dimension
        moving_scale = get_p_contiguous_scale(moving_scale, MP, moving_scale_p_offset)

    MSP, MSF0 = moving_scale.shape

    # The scale tensor may have more columns than needed (e.g., when stationary and moving scales are packed together).
    moving_scale_relevant = moving_scale[:, :MF0]

    # Convert scale exponents to scale factors
    moving_scale_factors = 2.0 ** (moving_scale_relevant - 127)  # Shape: [MSP, MF0]

    # Expand scale factors to match moving tensor shape
    # Each scale factor applies to an 8x1x4 block
    moving_scale_expanded = np.repeat(moving_scale_factors[:, :, np.newaxis], 4, axis=2)  # Shape: [MSP, MF0, 4]
    moving_scale_expanded = np.repeat(moving_scale_expanded[:, np.newaxis, :, :], 8, axis=1)  # Shape: [MSP, 8, MF0, 4]
    moving_scale_expanded = moving_scale_expanded.reshape(MSP * 8, MF0, 4)  # Shape: [MP, MF0, 4]

    # Apply scaling
    moving *= moving_scale_expanded

    # Process stationary tensor
    stationary = static_cast(stationary_x4, np.float32)
    new_shape = stationary.shape[:-1] + (stationary.shape[-1] // 4, 4)
    stationary = stationary.reshape(new_shape)
    SP, SF0, SF1 = stationary.shape
    assert SF1 == 4
    stationary = stationary.astype(np.float32)
    stationary_scale = stationary_scale.astype(np.float32)
    if not use_contiguous_scale:
        # if scale follows hw layout, make it contiguous at partition dimension
        stationary_scale = get_p_contiguous_scale(stationary_scale, SP, stationary_scale_p_offset)

    SSP, SSF0 = stationary_scale.shape

    # The scale tensor may have more columns than needed (e.g., when stationary and moving scales are packed together).
    stationary_scale_relevant = stationary_scale[:, :SF0]

    # Convert scale exponents to scale factors
    stationary_scale_factors = 2.0 ** (stationary_scale_relevant - 127)  # Shape: [SSP, SF0]

    # Expand scale factors to match stationary tensor shape
    # Each scale factor applies to an 8x1x4 block
    stationary_scale_expanded = np.repeat(stationary_scale_factors[:, :, np.newaxis], 4, axis=2)  # Shape: [SSP, SF0, 4]
    stationary_scale_expanded = np.repeat(
        stationary_scale_expanded[:, np.newaxis, :, :], 8, axis=1
    )  # Shape: [SSP, 8, SF0, 4]
    stationary_scale_expanded = stationary_scale_expanded.reshape(SSP * 8, SF0, 4)  # Shape: [SP, SF0, 4]

    # Apply scaling
    stationary *= stationary_scale_expanded

    golden = np.einsum("kiq,kjq->ij", stationary, moving)
    return golden


def nc_matmul_mx_golden_physical_scale(
    stationary_x4, moving_x4, stationary_scale, moving_scale, stationary_scale_p_offset=0, moving_scale_p_offset=0
):
    return nc_matmul_mx_golden(
        stationary_x4,
        moving_x4,
        stationary_scale,
        moving_scale,
        use_contiguous_scale=False,
        stationary_scale_p_offset=stationary_scale_p_offset,
        moving_scale_p_offset=moving_scale_p_offset,
    )


def dequantize_mx_golden(mx_data_x4, mx_scale):
    """
    Dequantize MX data back to float32, reversing quantize_mx_golden.

    This is the exact reverse of quantize_mx_golden:
    - quantize: out_data = in_data / scale, then clip, then static_cast to MX format
    - dequantize: static_cast to float32, then out_data = in_data * scale
    where scale = 2^(mx_scale - float32_exp_bias)

    Args:
            mx_data_x4: np.ndarray [P, F//4] in MxFP_x4 format - quantized data
            mx_scale: np.ndarray [SP, SF] in uint8 - scale tensor where SP=P//8, SF=F//4

    Returns:
            np.ndarray [P, F] in float32 - dequantized data (same shape as original input to quantize)
    """
    float32_exp_bias = 127

    P, F_packed = mx_data_x4.shape  # F_packed = F//4 from quantize_mx_golden
    SP, SF = mx_scale.shape  # SP = P//8, SF = F//4

    # Verify expected relationships
    assert SP == P // 8, f"Scale tensor P dimension mismatch: expected {P // 8}, got {SP}"
    assert SF == F_packed, f"Scale tensor F dimension mismatch: expected {F_packed}, got {SF}"

    # Convert quantized data to float32
    # static_cast expands the data: MxFP_x4 [P, F//4] -> float32 [P, F]
    data_float = static_cast(mx_data_x4, np.float32)

    # Get the actual expanded shape after static_cast
    P_expanded, F_expanded = data_float.shape

    # The expanded F dimension should be F = F_packed * 4
    assert F_expanded == F_packed * 4, f"Unexpected expansion: {F_packed} * 4 != {F_expanded}"

    # Convert scale exponents to scale factors in a vectorized manner
    scale_exp = mx_scale.astype(np.int32) - float32_exp_bias
    scale_exp = np.clip(scale_exp, -127, 127)  # Prevent overflow/underflow
    scale_factors = 2.0**scale_exp  # Shape: [SP, SF]

    # Use numpy's repeat and tile to expand scale factors to match data shape
    # Each scale factor needs to be applied to an 8x4 block
    # First expand along P dimension: repeat each row 8 times
    scale_expanded_p = np.repeat(scale_factors, 8, axis=0)  # Shape: [P_expanded, SF]

    # Then expand along F dimension: repeat each column 4 times
    scale_expanded = np.repeat(scale_expanded_p, 4, axis=1)  # Shape: [P_expanded, F_expanded]

    # Dequantize: multiply by scale (reverse of quantize division)
    dequantized_data = data_float * scale_expanded

    return dequantized_data
