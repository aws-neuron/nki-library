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

# Re-export of utilities that have been moved to src/.
from nkilib_src.nkilib.core.utils.mx_torch_common import (
    get_p_contiguous_scale as get_p_contiguous_scale,
)
from nkilib_src.nkilib.core.utils.mx_torch_common import (
    nc_matmul_mx_golden as nc_matmul_mx_golden,
)

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


def get_mx_max_exp(dst_dtype, mx_alt_emax=True):
    """Return the max biased exponent for a given MX x4 dtype.

    Args:
        dst_dtype: One of nl.float8_e5m2_x4, nl.float8_e4m3fn_x4, nl.float4_e2m1fn_x4.
        mx_alt_emax: When True (default), returns the alternative emax MX scale
            computation that avoids an OCP rounding bias where large values in a
            scale group are clamped down rather than rounded up to the next
            representable value, which can degrade training convergence and
            inference accuracy. When False, returns the original OCP-compliant
            max exponent.
    """
    if mx_alt_emax:
        max_exp_values = {nl.float8_e5m2_x4: 14, nl.float8_e4m3fn_x4: 7, nl.float4_e2m1fn_x4: 2}
    else:
        max_exp_values = {nl.float8_e5m2_x4: 15, nl.float8_e4m3fn_x4: 8, nl.float4_e2m1fn_x4: 2}
    assert dst_dtype in max_exp_values, f"no max exp value provided for {dst_dtype}"
    return max_exp_values[dst_dtype]


def quantize_mx_golden(
    in_tensor,
    out_x4_dtype,
    ocp_saturation=True,
    reverse_dst_fdim_group=0,
    custom_mx_max_exp=None,
    mx_alt_emax=True,
):
    """Quantize a float32/float16 tensor to MX x4 format.

    Args:
        in_tensor: (P, F) float input tensor.
        out_x4_dtype: Target MX x4 dtype (e.g., nl.float8_e5m2_x4).
        ocp_saturation: Whether to clip to the MX format's max representable value.
        reverse_dst_fdim_group: If > 0, reverse free dimension by groups of this size.
        custom_mx_max_exp: Optional callable(out_x4_dtype) -> max_exp. When
            provided, overrides mx_alt_emax.
        mx_alt_emax: When True (default), uses the alternative emax, matching the
            neuronxcc compiler default. Set to False to restore the original
            OCP-compliant behavior.
    """
    max_exp = (
        custom_mx_max_exp(out_x4_dtype) if custom_mx_max_exp else get_mx_max_exp(out_x4_dtype, mx_alt_emax=mx_alt_emax)
    )
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
