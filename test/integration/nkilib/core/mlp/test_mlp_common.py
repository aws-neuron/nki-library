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
import math
from collections import namedtuple
from test.integration.nkilib.utils.tensor_generators import (
    TensorTemplate,
    gaussian_tensor_generator,
    generate_stabilized_mx_data,
    update_func_str,
)
from test.integration.nkilib.utils.test_kernel_common import (
    act_fn_type2func,
    is_dtype_mx,
    norm_name2func,
)
from test.utils.mx_utils import nc_matmul_mx_golden
from typing import Callable

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    NormType,
    QuantizationType,
)
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert


def build_fused_norm_mlp(
    batch,
    seqlen,
    hidden,
    intermediate,
    dtype,
    quant_dtype=None,
    quantization_type=QuantizationType.NONE,
    is_input_quantized=False,
    eps=1e-6,
    fused_add=False,
    norm_type=NormType.RMS_NORM,
    store_add=False,
    lnc_degree=1,
    tiling_degree=None,
    skip_gate=False,
    act_fn_type=ActFnType.SiLU,
    gate_bias=False,
    up_bias=False,
    down_bias=False,
    norm_bias=False,
    use_tkg_gate_up_proj_column_tiling=True,
    use_tkg_down_proj_column_tiling=True,
    use_tkg_down_proj_optimized_layout=False,
    gate_clamp_lower_limit=None,
    gate_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    tensor_generator: Callable = gaussian_tensor_generator(),
):
    np.random.seed(42)
    rng = np.random.default_rng(42)

    if isinstance(norm_type, bool):
        norm_type = NormType.RMS_NORM if norm_type else NormType.NO_NORM

    fused_add_tensor = (
        tensor_generator(shape=(batch, seqlen, hidden), dtype=dtype, name="fused_add_tensor") if fused_add else None
    )

    if quantization_type == QuantizationType.MX:
        _pmax = 128
        _q_width = 4
        tokens = batch * seqlen
        n_H512_tile = hidden // 512
        n_I512_tile = math.ceil(intermediate / (_pmax * _q_width))

        # Generate hidden_input as stabilized MX data
        hidden_states, _, _ = generate_stabilized_mx_data(
            mx_dtype=nl.float8_e4m3fn_x4, shape=(tokens * n_H512_tile * _pmax, _q_width), val_range=1.0
        )
        hidden_states = (
            hidden_states.reshape(tokens, n_H512_tile, _pmax, _q_width).transpose(0, 3, 1, 2).reshape(tokens, hidden)
        )
        hidden_input = dt.static_cast(hidden_states, dtype).reshape(batch, seqlen, hidden)
    elif is_input_quantized:
        if quantization_type == QuantizationType.ROW:
            hidden_input = tensor_generator(
                shape=(batch, seqlen, hidden + 4), dtype=quant_dtype, name='hidden'
            )  # +4 to allow space for an fp32 dequant value
        elif quantization_type in [QuantizationType.STATIC, QuantizationType.STATIC_MX]:
            hidden_input = tensor_generator(shape=(batch, seqlen, hidden), dtype=quant_dtype, name='hidden')
    else:
        hidden_input = tensor_generator(shape=(batch, seqlen, hidden), dtype=dtype, name="hidden")

    gamma = None
    norm_b = None
    if norm_type != NormType.NO_NORM and norm_type != NormType.RMS_NORM_SKIP_GAMMA:
        gamma = dt.static_cast(rng.uniform(low=-0.1, high=0.1, size=(1, hidden)), dtype)
        if quantization_type == QuantizationType.MX:
            gamma = (
                gamma.reshape(1, hidden // (_pmax * _q_width), _pmax, _q_width).transpose(0, 3, 1, 2).reshape(1, hidden)
            )
        if norm_bias:
            norm_b = tensor_generator(shape=(1, hidden), dtype=dtype, name="norm_b")

    # Pre-generate MX weights if using MX quantization
    if quantization_type == QuantizationType.MX:
        mx_weights = gen_mlp_mxfp_weights(hidden, intermediate, quant_dtype)

    weight_dtype = quant_dtype if quant_dtype is not None else dtype

    # Use pre-generated MX weights or generate regular weights
    if quantization_type == QuantizationType.MX:
        gate_w = mx_weights.gate_w_qtz
        up_w = mx_weights.up_w_qtz
        down_w = mx_weights.down_w_qtz
    else:
        gate_w = tensor_generator(shape=(hidden, intermediate), dtype=weight_dtype, name="gate_w")
        up_w = tensor_generator(shape=(hidden, intermediate), dtype=weight_dtype, name="up_w")
        down_w = tensor_generator(shape=(intermediate, hidden), dtype=weight_dtype, name="down_w")

    # Generate Bias
    if quantization_type == QuantizationType.MX:
        i_p = intermediate // 4 if intermediate <= 512 else _pmax
        gate_b = tensor_generator(shape=(i_p, n_I512_tile, _q_width), dtype=dtype, name="gate_b") if gate_bias else None
        up_b = tensor_generator(shape=(i_p, n_I512_tile, _q_width), dtype=dtype, name="up_b") if up_bias else None
        down_b = tensor_generator(shape=(1, hidden), dtype=dtype, name="down_b") if down_bias else None
    else:
        gate_b = tensor_generator(shape=(1, intermediate), dtype=dtype, name="gate_b") if gate_bias else None
        up_b = tensor_generator(shape=(1, intermediate), dtype=dtype, name="up_b") if up_bias else None
        down_b = tensor_generator(shape=(1, hidden), dtype=dtype, name="down_b") if down_bias else None

    if quantization_type == QuantizationType.MX:
        # MX quantization uses uint8 scales
        gate_w_scale = mx_weights.gate_w_scale
        up_w_scale = mx_weights.up_w_scale
        down_w_scale = mx_weights.down_w_scale
        gate_up_in_scale = None
        down_in_scale = None
    elif quantization_type in [QuantizationType.STATIC, QuantizationType.STATIC_MX]:
        gate_w_scale = tensor_generator(shape=(128, 1), dtype=np.float32, name="gate_w_scale")
        up_w_scale = tensor_generator(shape=(128, 1), dtype=np.float32, name="up_w_scale")
        down_w_scale = tensor_generator(shape=(128, 1), dtype=np.float32, name="down_w_scale")
        gate_up_in_scale = tensor_generator(shape=(128, 1), dtype=np.float32, name="gate_up_in_scale")
        down_in_scale = tensor_generator(shape=(128, 1), dtype=np.float32, name="down_in_scale")
    elif quantization_type == QuantizationType.ROW or quant_dtype is not None:
        gate_w_scale = tensor_generator(shape=(128, intermediate), dtype=np.float32, name="gate_w_scale")
        up_w_scale = tensor_generator(shape=(128, intermediate), dtype=np.float32, name="up_w_scale")
        down_w_scale = tensor_generator(shape=(128, hidden), dtype=np.float32, name="down_w_scale")
        gate_up_in_scale = None
        down_in_scale = None
    else:
        gate_w_scale = None
        up_w_scale = None
        down_w_scale = None
        gate_up_in_scale = None
        down_in_scale = None

    kernel_input = {
        "hidden_tensor": hidden_input,
        "gate_proj_weights_tensor": gate_w,
        "up_proj_weights_tensor": up_w,
        "down_proj_weights_tensor": down_w,
        "normalization_weights_tensor": gamma,
        "gate_proj_bias_tensor": gate_b,
        "up_proj_bias_tensor": up_b,
        "down_proj_bias_tensor": down_b,
        "normalization_bias_tensor": norm_b,
        "fused_add_tensor": fused_add_tensor,
        "store_fused_add_result": store_add,
        "activation_fn": act_fn_type,
        "normalization_type": norm_type,
        "quantization_type": quantization_type,
        "gate_w_scale": gate_w_scale,
        "up_w_scale": up_w_scale,
        "down_w_scale": down_w_scale,
        "gate_up_in_scale": gate_up_in_scale,
        "down_in_scale": down_in_scale,
        "output_dtype": nl.bfloat16,
        "store_output_in_sbuf": False,
        "eps": eps,
        "skip_gate_proj": skip_gate,
        "use_tkg_gate_up_proj_column_tiling": use_tkg_gate_up_proj_column_tiling,
        "use_tkg_down_proj_column_tiling": use_tkg_down_proj_column_tiling,
        "use_tkg_down_proj_optimized_layout": use_tkg_down_proj_optimized_layout,
        "gate_clamp_upper_limit": gate_clamp_upper_limit,
        "gate_clamp_lower_limit": gate_clamp_lower_limit,
        "up_clamp_upper_limit": up_clamp_upper_limit,
        "up_clamp_lower_limit": up_clamp_lower_limit,
    }
    return kernel_input


float8_e5m2_x4 = nl.float8_e5m2_x4
float8_e4m3fn_x4 = nl.float8_e4m3fn_x4
float4_e2m1fn_x4 = nl.float4_e2m1fn_x4


# max normal
# float8_e5m2: S 11110 11 = ± 2^15 × 1.75 = ± 57,344
# float8_e4m3fn: S 1111 110 = ± 2^8 × 1.75 = ± 448
# float4_e2m1fn: S 11 1 = ± 2^2 × 1.5 = ± 6
def get_mx_fp_max(dst_dtype):
    max_values = {float8_e5m2_x4: 57344, float8_e4m3fn_x4: 448, float4_e2m1fn_x4: 6}
    assert dst_dtype in max_values, f"no max value provided for {dst_dtype}"
    return max_values.get(dst_dtype)


def get_mx_max_exp(dst_dtype):
    max_exp_values = {float8_e5m2_x4: 15, float8_e4m3fn_x4: 8, float4_e2m1fn_x4: 2}
    assert dst_dtype in max_exp_values, f"no max exp value provided for {dst_dtype}"
    return max_exp_values.get(dst_dtype)


# Get exponent for float32 in IEEE 754 standard
def get_float32_exp(float_data):
    man_nbits, exp_nbits = 23, 8
    return (float_data.astype(np.float32).view(np.uint32) >> man_nbits) & ((1 << exp_nbits) - 1)


MlpMxWeights = namedtuple(
    'MlpMxWeights',
    [
        'gate_w_qtz',
        'gate_w_scale',
        'up_w_qtz',
        'up_w_scale',
        'down_w_qtz',
        'down_w_scale',
    ],
)


def gen_mlp_mxfp_weights(hidden, intermediate, mx_dtype):
    """Generate MX quantized weights and scales for MLP gate, up, and down projections.

    Args:
        hidden: Hidden dimension size (must be divisible by 512)
        intermediate: Intermediate dimension size
        mx_dtype: MX quantization dtype (float4_e2m1fn_x4 or float8_e4m3fn_x2)

    Returns:
        MlpMxWeights: Named tuple containing quantized weights and scales
    """

    def split_last_dim(X, extra_last_dim):
        return X.reshape(*X.shape[:-1], -1, extra_last_dim)

    n_H512_tile = hidden // 512
    n_I512_tile = math.ceil(intermediate / 512)
    p_I = (intermediate // 4) if intermediate < 512 else 128  # do not pad I's pdim if I<512

    # Initialize tensors for gate projection
    gate_w_qtz = np.zeros((128, n_H512_tile, intermediate), dtype=mx_dtype)
    gate_w_scale = np.zeros((16, n_H512_tile, intermediate), dtype=np.uint8)

    # Initialize tensors for up projection
    up_w_qtz = np.zeros((128, n_H512_tile, intermediate), dtype=mx_dtype)
    up_w_scale = np.zeros((16, n_H512_tile, intermediate), dtype=np.uint8)

    # Initialize tensors for down projection
    down_w_qtz = np.zeros((p_I, n_I512_tile, hidden), dtype=mx_dtype)
    down_w_scale = np.zeros((p_I // 8, n_I512_tile, hidden), dtype=np.uint8)

    # Generate gate projection weights
    _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (128, hidden // 128 * intermediate))
    gate_w_qtz[:, :, :] = split_last_dim(tmp_w_qtz, intermediate)  # [128, n_H512_tile, intermediate]
    gate_w_scale[:, :, :] = split_last_dim(tmp_w_scale, intermediate)  # [16, n_H512_tile, intermediate]

    # Generate up projection weights
    _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (128, hidden // 128 * intermediate))
    up_w_qtz[:, :, :] = split_last_dim(tmp_w_qtz, intermediate)  # [128, n_H512_tile, intermediate]
    up_w_scale[:, :, :] = split_last_dim(tmp_w_scale, intermediate)  # [16, n_H512_tile, intermediate]

    # Generate down projection weights
    _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (intermediate // 4, hidden * 4))
    for i_I512_tile in range(n_I512_tile):
        n_rows_qtz = min(128, intermediate // 4 - i_I512_tile * 128)  # every 512 tile has at most 512/4=128 x4 values
        n_rows_scale = min(16, intermediate // 32 - i_I512_tile * 16)  # every 512 tile has at most 512/32=16 scales
        down_w_qtz[:n_rows_qtz, i_I512_tile, :] = tmp_w_qtz[i_I512_tile * 128 : i_I512_tile * 128 + n_rows_qtz, :]
        down_w_scale[:n_rows_scale, i_I512_tile, :] = tmp_w_scale[i_I512_tile * 16 : i_I512_tile * 16 + n_rows_scale, :]

    return MlpMxWeights(gate_w_qtz, gate_w_scale, up_w_qtz, up_w_scale, down_w_qtz, down_w_scale)


MxAllTokensWeights = namedtuple(
    'MxAllTokensWeights',
    [
        'gate_up_w_qtz',
        'gate_up_w_scale',
        'down_w_qtz',
        'down_w_scale',
    ],
)


def gen_moe_mx_weights(hidden, intermediate, expert, mx_dtype=float4_e2m1fn_x4):
    """Generate MX weights and scales for all tokens gate/up (fused into one tensor) and down projection.

    Args:
        hidden: Hidden dimension size
        intermediate: Intermediate dimension size
        expert: Number of experts
        mx_dtype: MX quantization dtype (float4_e2m1fn_x4 or float8_e4m3fn_x4)
    """
    # x4 types pack 4 elements, so static_cast divides F-dim by 4 for both MXFP4 and MXFP8
    x4_pack_size = 4

    def split_last_dim(X, extra_last_dim):
        return X.reshape(*X.shape[:-1], -1, extra_last_dim)

    n_H512_tile = hidden // 512
    n_I512_tile = math.ceil(intermediate / 512)
    p_I = math.ceil(intermediate / x4_pack_size / 8) * 8 if intermediate < 512 else 128
    gate_up_w_qtz = np.zeros((expert, 128, 2, n_H512_tile, intermediate), dtype=mx_dtype)
    gate_up_w_scale = np.zeros((expert, 16, 2, n_H512_tile, intermediate), dtype=np.uint8)
    down_w_qtz = np.zeros((expert, p_I, n_I512_tile, hidden), dtype=mx_dtype)
    down_w_scale = np.zeros((expert, math.ceil(p_I / 8), n_I512_tile, hidden), dtype=np.uint8)

    for e in range(expert):
        # Gen weight for gate and up proj
        for i in range(2):
            _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (128, hidden // 128 * intermediate))

            # Copy to full weights and scales
            gate_up_w_qtz[e, :, i, :, :] = split_last_dim(tmp_w_qtz, intermediate)  # [128, n_H512_tile, intermediate]
            gate_up_w_scale[e, :, i, :, :] = split_last_dim(
                tmp_w_scale, intermediate
            )  # [128, n_H512_tile, intermediate]

        # Gen weight for down proj - generate logical shape, static_cast handles x4 packing
        # Pad to multiple of 8 if needed
        I_p_actual = math.ceil(intermediate / x4_pack_size)
        I_p_padded = math.ceil(I_p_actual / 8) * 8
        _, tmp_w_qtz, tmp_w_scale = generate_stabilized_mx_data(mx_dtype, (I_p_padded, hidden * x4_pack_size))
        tmp_w_qtz = tmp_w_qtz[:I_p_actual, :]
        tmp_w_scale = tmp_w_scale[: math.ceil(I_p_actual / 8), :]
        for i_I512_tile in range(n_I512_tile):
            n_rows_qtz = min(128, intermediate // x4_pack_size - i_I512_tile * 128)
            n_rows_scale = min(16, math.ceil(intermediate / 32) - i_I512_tile * 16)
            down_w_qtz[e, :n_rows_qtz, i_I512_tile, :] = tmp_w_qtz[
                i_I512_tile * 128 : i_I512_tile * 128 + n_rows_qtz, :
            ]
            down_w_scale[e, :n_rows_scale, i_I512_tile, :] = tmp_w_scale[
                i_I512_tile * 16 : i_I512_tile * 16 + n_rows_scale, :
            ]

    return MxAllTokensWeights(gate_up_w_qtz, gate_up_w_scale, down_w_qtz, down_w_scale)


def quantize_mx_golden(
    in_tensor,
    out_x4_dtype,
    ocp_saturation=True,
    reverse_dst_fdim_group=0,
    custom_mx_max_exp=None,
):
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
    mx_data_golden = dt.static_cast(mx_data_golden, out_x4_dtype)

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
    data_float = dt.static_cast(mx_data_x4, np.float32)

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


def norm_mlp_ref(
    hiddens,
    gamma,
    gate,
    up,
    down,
    dtype,
    norm_type=NormType.NO_NORM,
    store_add=False,
    skip_gate=False,
    act_fn_type=ActFnType.SiLU,
    gate_b=None,
    up_b=None,
    down_b=None,
    norm_b=None,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    down_proj_lhs_rhs_swap_optimized_layout=False,
    lnc=None,
    is_matching_mxfp4=False,
):
    hidden_sum = np.zeros(hiddens[0].shape)
    for hidden in hiddens:
        hidden_sum += hidden
    sum_out = hidden_sum
    if norm_b is not None:
        hidden_sum = norm_name2func[norm_type](hidden_sum, gamma, norm_b=norm_b)
    else:
        hidden_sum = norm_name2func[norm_type](hidden_sum, gamma)

    if is_matching_mxfp4:
        hidden_sum = dequantize_mx_golden(*quantize_mx_golden(hidden_sum.reshape(-1, 4), float8_e4m3fn_x4)).reshape(
            hidden_sum.shape
        )

    up_out = hidden_sum @ up
    if up_b is not None:
        up_out += up_b

    if down_proj_lhs_rhs_swap_optimized_layout:
        assert lnc is not None
        # Have layout friendly shape here, need to put it back for golden calculation
        I, H = down.shape
        down = down.reshape((I, lnc, H // 128 // lnc, 128)).transpose((0, 1, 3, 2)).reshape((I, H))

    # clamp on up projection results
    if up_clamp_upper_limit is not None:
        up_out = np.minimum(up_out, up_clamp_upper_limit)
    if up_clamp_lower_limit is not None:
        up_out = np.maximum(up_out, up_clamp_lower_limit)

    act_fn = act_fn_type2func[act_fn_type]

    if not skip_gate:
        gate_out = hidden_sum @ gate
        if gate_b is not None:
            gate_out += gate_b
        if gate_clamp_upper_limit is not None:
            gate_out = np.minimum(gate_out, gate_clamp_upper_limit)
        if gate_clamp_lower_limit is not None:
            gate_out = np.maximum(gate_out, gate_clamp_lower_limit)
        gate_act_fn_out = act_fn(gate_out)
        mult = gate_act_fn_out * up_out
    else:
        mult = act_fn(up_out)

    if is_matching_mxfp4:
        mult = dequantize_mx_golden(*quantize_mx_golden(mult.reshape(-1, 4), float8_e4m3fn_x4)).reshape(mult.shape)

    output = mult @ down

    if down_b is not None:
        output += down_b

    return output.astype(dtype), sum_out.astype(dtype)


_q_width = 4  # quantization width
_q_height = 8  # quantization height
_pmax = 128  # sbuf max partition dim


def _gate_up_proj_golden_mx(hidden, hidden_scale, weight, weight_scale, bias, cfg):
    """
    Golden reference for gate/up projection kernel.
    Performs: 1. hidden (moving) [H, BxS] @ weight (stationary) [H, I] → [I, BxS]
                2. fold I 4 times onto the fastest dim [I, BxS] → [I//4, BxS, 4]

    Weight layout:
        - [16_H, 8_H, H/512, I, 4_H], 4_H is packed in mxfp4
        - I dimension has layout order of ⌈I/512⌉, 4_I, 16_I, 8_I
    Bias layout:
        - [_pmax, ⌈I/512⌉, 4_I]
        - if I >= 512: [_pmax, ⌈I/512⌉, 4_I]
        - if I < 512:  [I / 4, ⌈I/512⌉, 4_I]
    Hidden layout:
        - if already quantized: [16_H, 8_H, H/512, BxS, 4_H], 4_H is packed in mxfp4
        - if not yet quantized: [BxS, 4_H, H/512, 16_H, 8_H]
    Output layout:
        - [_pmax, ⌈I/512⌉, BxS, 4_I]
        - Note that if I is not a multiple of 512, the output will be zero-padded
    """
    do_hidden_quantization = hidden_scale is None

    H, I, BxS = cfg.H, cfg.I, cfg.BxS
    assert H % (_pmax * _q_width) == 0

    if do_hidden_quantization:
        assert hidden.shape == (BxS, H)
    else:
        assert hidden.shape == (_pmax, H // _pmax // _q_width, BxS)
        assert hidden_scale.shape == (_pmax // _q_height, H // _pmax // _q_width, BxS)

    assert weight.shape == (_pmax, H // _pmax // _q_width, I)
    assert weight_scale.shape == (_pmax // _q_height, H // _pmax // _q_width, I)

    # Transpose and quantize input hidden if needed:
    # [T, 4_H, H/512, 16_H, 8_H] -> [16_H * 8_H (P), H/512, T, 4_H]
    if do_hidden_quantization:
        hidden = hidden.reshape(BxS, _q_width, H // _pmax // _q_width, _pmax).transpose(3, 2, 0, 1).reshape(_pmax, -1)
        hidden_mx, hidden_scale = quantize_mx_golden(in_tensor=hidden, out_x4_dtype=float8_e4m3fn_x4)
    else:
        hidden_mx, hidden_scale = hidden, hidden_scale

    # Weight layout: [16_H (P), 8_H (P), H/512, I, 4_H]
    # Need to make sure _pmax is fastest moving in H
    weight = weight.transpose(1, 0, 2)
    weight_scale = weight_scale.transpose(1, 0, 2)
    hidden_mx = hidden_mx.reshape(_pmax, H // _pmax // _q_width, BxS).transpose(1, 0, 2)
    hidden_scale = hidden_scale.reshape(_pmax // _q_height, H // _pmax // _q_width, BxS).transpose(1, 0, 2)

    # [I, BxS]
    result = nc_matmul_mx_golden(
        stationary_x4=weight.reshape((H // _q_width, I)),
        moving_x4=hidden_mx.reshape((H // _q_width, BxS)),
        stationary_scale=weight_scale.reshape((H // _q_height // _q_width, I)),
        moving_scale=hidden_scale.reshape((H // _q_height // _q_width, BxS)),
    )

    # Shuffle I dim with a stride of four, this mimics kernel behaviour which aligns with quantization
    n_I512_tiles = math.ceil(I / 512)

    res_shfl = np.zeros((_pmax, n_I512_tiles, BxS, 4), dtype=result.dtype)
    for i in range(n_I512_tiles):
        rows_filled = 512
        # Last tile, may have less filled rows in the 512 tile
        if (i == n_I512_tiles - 1) and (I % 512 != 0):
            rows_filled = I % 512

        # Fold and transpose the current tile and fill into res_shfl
        cur_tile = result[i * 512 : i * 512 + rows_filled, :]  # [rows_filled, BxS]
        rows_padded = math.ceil(rows_filled / 8) * 8
        if rows_padded > rows_filled:
            cur_tile = np.pad(cur_tile, ((0, rows_padded - rows_filled), (0, 0)))
        cur_tile = cur_tile.reshape(4, rows_padded // 4, BxS)
        cur_tile = cur_tile.transpose(1, 2, 0)  # [rows_padded//4, BxS, 4]
        res_shfl[: rows_padded // 4, i, :, :] = cur_tile

    if bias is not None:
        if I < 512:  # Pad first dim to _pmax
            bias = np.pad(bias, ((0, _pmax - bias.shape[0]), *((0, 0),) * (bias.ndim - 1)))
        assert bias.shape == (_pmax, math.ceil(I / (_pmax * _q_width)), _q_width)
        res_shfl += bias[:, :, np.newaxis, :]

    return res_shfl


def _down_proj_golden_mx(inter_sb, weight, weight_scale, bias, cfg):
    """
    Golden reference for down projection kernel.
    Performs: weight (moving) [I, H] @ inter_sb (stationary) [I, BxS] → [BxS, H]

    inter_sb layout:
        - [_pmax, ⌈I/512⌉, BxS, 4_I]
    weight layout:
        - if I >= 512: [_pmax, ⌈I/512⌉, H, 4_I], 4_I is packed in mxfp4
        - if I < 512:  [I / 4,       1, H, 4_I]
    bias layout:
        - [1, H]
    output layout:
        - [BxS, H]
    """
    H, I, BxS = cfg.H, cfg.I, cfg.BxS

    I_padded_q = math.ceil(I / _q_width)

    # Original weight shape is [128, ceil(I/512), H], scale shape is [16, ceil(I/512), H].
    # Both of them have zero filled values for last I tile. Reshape and truncate.
    weight = weight.transpose(1, 0, 2).reshape(-1, H)
    weight_scale = weight_scale.transpose(1, 0, 2).reshape(-1, H)

    if I < 512:
        inter_sb = inter_sb[:I_padded_q, :, :, :]
    inter_sb = inter_sb.transpose(1, 0, 2, 3).reshape(-1, BxS * _q_width)

    # Align partition dim to multiple of 8 for MX quantization block alignment.
    P = inter_sb.shape[0]
    SP = weight_scale.shape[0]
    P_aligned = SP * _q_height
    if P < P_aligned:
        inter_sb = np.pad(inter_sb, ((0, P_aligned - P), (0, 0)))
    if weight.shape[0] < P_aligned:
        pad_rows = P_aligned - weight.shape[0]
        weight = np.concatenate([weight, np.zeros((pad_rows, weight.shape[1]), dtype=weight.dtype)], axis=0)
    else:
        weight = weight[:P_aligned, :]

    # Quantize input to mxfp8.
    input_mx_data, input_mx_scale = quantize_mx_golden(inter_sb, float8_e4m3fn_x4)
    result = nc_matmul_mx_golden(input_mx_data, weight, input_mx_scale, weight_scale)

    if bias is not None:
        result += bias.reshape((1, H))

    return result


def norm_mlp_ref_mx(
    hiddens,
    gamma,
    gate,
    gate_scale,
    up,
    up_scale,
    down,
    down_scale,
    dtype,
    norm_type=NormType.NO_NORM,
    skip_gate=False,
    act_fn_type=ActFnType.SiLU,
    gate_b=None,
    up_b=None,
    down_b=None,
    norm_b=None,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
):
    # Create a simple config object with the required attributes
    class SimpleConfig:
        def __init__(self, H, I, BxS):
            self.H = H
            self.I = I
            self.BxS = BxS

    for hidden in hiddens:
        if len(hiddens[0].shape) == 3:
            B, S, H = hiddens[0].shape
            hiddens[0] = hiddens[0].reshape((B * S, H))
    cfg = SimpleConfig(hiddens[0].shape[1], gate.shape[-1], hiddens[0].shape[0])

    _pmax = 128
    _q_width = 4

    hidden_sum = np.zeros(hiddens[0].shape)
    assert hidden_sum.shape == (cfg.BxS, cfg.H)
    for hidden in hiddens:
        hidden_sum += hidden
    sum_out = hidden_sum

    if gamma is not None:
        hidden_sum = (
            hidden_sum.reshape(cfg.BxS, _q_width, cfg.H // (_pmax * _q_width), _pmax)
            .transpose(0, 2, 3, 1)
            .reshape(cfg.BxS, cfg.H)
        )
        gamma = gamma.reshape(1, _q_width, cfg.H // (_pmax * _q_width), _pmax).transpose(0, 2, 3, 1).reshape(1, cfg.H)
        if norm_b is not None:
            hidden_sum = norm_name2func[norm_type](hidden_sum, gamma, norm_b=norm_b)
        else:
            hidden_sum = norm_name2func[norm_type](hidden_sum, gamma)
        hidden_sum = (
            hidden_sum.reshape(cfg.BxS, cfg.H // (_pmax * _q_width), _pmax, _q_width)
            .transpose(0, 3, 1, 2)
            .reshape(cfg.BxS, cfg.H)
        )

    up_out = _gate_up_proj_golden_mx(
        hidden=hidden_sum,
        hidden_scale=None,
        weight=up,
        weight_scale=up_scale,
        bias=up_b,
        cfg=cfg,
    )

    # clamp on up projection results
    if up_clamp_upper_limit is not None:
        up_out = np.minimum(up_out, up_clamp_upper_limit)
    if up_clamp_lower_limit is not None:
        up_out = np.maximum(up_out, up_clamp_lower_limit)

    act_fn = act_fn_type2func[act_fn_type]

    if not skip_gate:
        gate_out = _gate_up_proj_golden_mx(
            hidden=hidden_sum,
            hidden_scale=None,
            weight=gate,
            weight_scale=gate_scale,
            bias=gate_b,
            cfg=cfg,
        )
        if gate_clamp_upper_limit is not None:
            gate_out = np.minimum(gate_out, gate_clamp_upper_limit)
        if gate_clamp_lower_limit is not None:
            gate_out = np.maximum(gate_out, gate_clamp_lower_limit)
        gate_act_fn_out = act_fn(gate_out)
        mult = gate_act_fn_out * up_out
    else:
        mult = act_fn(up_out)

    output = _down_proj_golden_mx(
        inter_sb=mult,
        weight=down,
        weight_scale=down_scale,
        bias=down_b,
        cfg=cfg,
    )

    return output.astype(dtype), sum_out.astype(dtype)


def golden_mlp(
    inp_np,
    norm_type,
    fused_add,
    store_add,
    dtype,
    skip_gate=False,
    act_fn_type=ActFnType.SiLU,
    lnc=None,
    down_proj_lhs_rhs_swap_optimized_layout=False,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
):
    if fused_add:
        fused_add_tensor = inp_np["fused_add_tensor"]

    hidden = inp_np["hidden_tensor"]
    gamma = inp_np["normalization_weights_tensor"] if "normalization_weights_tensor" in inp_np else None
    gate_w = inp_np["gate_proj_weights_tensor"]
    up_w = inp_np["up_proj_weights_tensor"]
    down_w = inp_np["down_proj_weights_tensor"]
    gate_b = inp_np["gate_proj_bias_tensor"] if "gate_proj_bias_tensor" in inp_np else None
    up_b = inp_np["up_proj_bias_tensor"] if "up_proj_bias_tensor" in inp_np else None
    down_b = inp_np["down_proj_bias_tensor"] if "down_proj_bias_tensor" in inp_np else None
    norm_b = inp_np["normalization_bias_tensor"] if "normalization_bias_tensor" in inp_np else None

    hiddens = [fused_add_tensor, hidden] if fused_add else [hidden]
    store_add_valid = fused_add and store_add
    mlp_out, add_out = norm_mlp_ref(
        hiddens,
        gamma,
        gate_w,
        up_w,
        down_w,
        dtype,
        norm_type=norm_type,
        store_add=store_add_valid,
        skip_gate=skip_gate,
        act_fn_type=act_fn_type,
        gate_b=gate_b,
        up_b=up_b,
        down_b=down_b,
        norm_b=norm_b,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        down_proj_lhs_rhs_swap_optimized_layout=down_proj_lhs_rhs_swap_optimized_layout,
        lnc=lnc,
    )
    out_dict = {"out": dt.static_cast(mlp_out, dtype)}
    if store_add_valid:
        out_dict["add_out"] = dt.static_cast(add_out, dtype)

    return out_dict


def norm_quant_mlp_ref(
    hiddens,
    gamma,
    gate,
    up,
    down,
    quantization_type,
    gate_w_scale,
    up_w_scale,
    down_w_scale,
    gate_in_scale,
    up_in_scale,
    down_in_scale,
    dtype,
    quant_dtype,
    norm_type,
    store_add=False,
    clip=None,
    skip_gate=False,
    act_fn_type=ActFnType.SiLU,
    gate_b=None,
    up_b=None,
    down_b=None,
    down_proj_lhs_rhs_swap_optimized_layout=False,
    lnc=None,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    quantize_activation=False,
):
    """
    Reference implementation for MLP with optional RMSNorm and quantization.

    This function supports both TKG (Token Generation) and CTE (Context Encoding) modes:

    quantize_activation=True:
        - Input remains in original dtype (BF16/FP32), no input quantization
        - Intermediate outputs (gate*up) remain unquantized
        - Only weights are quantized (FP8) and dequantized during projection
        - Computation: Hidden_BF16 @ Weight_FP8 * weight_scale

    quantize_activation=False:
        - Computation: Hidden_FP8 @ Weight_FP8 * weight_scale * input_scale
    """

    def scale_with_broadcast(weight, scale):
        if len(weight.shape) == 3 and len(scale.shape) == 2:
            # TG kernel is triggered when batch * seqlen in less than 16
            if weight.shape[1] * weight.shape[0] < TKG_BS_SEQLEN_THRESHOLD:
                # this is a special case for TG. the scale tensor is bigger than the
                # weight tensor, so there is not need to broadcast
                return dt.static_cast(weight, dtype) * scale[: weight.shape[1], :]

            # broadcast scale to the size of weight tensor and perform tensorXtensor
            return dt.static_cast(weight, dtype) * np.tile(
                scale,
                (
                    weight.shape[0],
                    math.ceil(weight.shape[1] / scale.shape[0]),
                    math.ceil(weight.shape[2] / scale.shape[1]),
                ),
            )
        elif len(weight.shape) == 3 and len(scale.shape) == 3:
            return dt.static_cast(weight, dtype) * scale
        else:
            return dt.static_cast(weight, dtype) * np.tile(
                scale,
                (
                    math.ceil(weight.shape[0] / scale.shape[0]),
                    math.ceil(weight.shape[1] / scale.shape[1]),
                ),
            )

    def get_max_value_for_dtype(dtype):
        if str(dtype) == "uint8":
            return 127.5
        elif dtype == nl.float8_e5m2:
            return 57344.0
        elif dtype == nl.float8_e4m3:
            return 240.0
        else:
            return 1

    def get_zero_point_for_dtype(dtype):
        if str(dtype) == "uint8":
            return 127
        else:
            return 0

    is_static_quant = quantization_type in [QuantizationType.STATIC, QuantizationType.STATIC_MX]

    def undo_mx_gate_up_w_reshape(weight):
        H, I = weight.shape[0], weight.shape[1]
        H_OVER_512 = H // 512
        H_128 = (H // H_OVER_512) // 4
        H_2 = 2
        I_OVER_512 = I // 512
        I_128 = (I // I_OVER_512) // 4
        I_4 = 4
        return (
            weight.reshape(
                (H_128, H_OVER_512, I_OVER_512, I_4, I_128, H_2, H_2),
            )
            .transpose(
                # (H_2, H_OVER_512, H_128, H_2, I_OVER_512, I_128, I_4)
                (5, 1, 0, 6, 2, 4, 3),
            )
            .reshape(H, I)
        )

    def undo_mx_down_w_reshape(weight):
        I, H = weight.shape[0], weight.shape[1]
        I_OVER_512 = I // 512
        I_128 = (I // I_OVER_512) // 4
        I_4 = 4
        return (
            weight.reshape(
                (I_128, I_OVER_512, H, I_4),
            )
            .transpose(
                # (I_OVER_512, I_128, I_4, H)
                (1, 0, 3, 2),
            )
            .reshape(I, H)
        )

    def perform_row_quant(input_tensor, clipping_boundary):
        abs_max = np.max(np.absolute(input_tensor), axis=-1, keepdims=True)
        if clipping_boundary is not None and clipping_boundary != 0:
            np.clip(abs_max, -clipping_boundary, clipping_boundary, out=abs_max)
            # don't modify reference to the original input tensor
            input_tensor = np.clip(input_tensor, -clipping_boundary, clipping_boundary)

        quant_scale = np.maximum(abs_max / get_max_value_for_dtype(quant_dtype), 1e-05)
        return input_tensor / quant_scale + get_zero_point_for_dtype(dtype), quant_scale

    def perform_static_quant(input_tensor, quant_scale):
        max_val = get_max_value_for_dtype(quant_dtype)
        scaled_tensor = scale_with_broadcast(input_tensor, 1 / quant_scale)
        np.clip(scaled_tensor, -max_val, max_val, out=scaled_tensor)
        return scaled_tensor, None

    def perform_tensor_quant(input_tensor, clipping_boundary, quant_scale):
        if quantization_type == QuantizationType.ROW:
            return perform_row_quant(input_tensor, clipping_boundary)
        elif is_static_quant:
            return perform_static_quant(input_tensor, quant_scale)

    def perform_projection(input, proj_w, proj_w_scale, static_in_scale, row_in_scale):
        """
        Perform matrix multiplication with optional quantization scaling.

        TKG case (no input quantization):
            - Input remains in BF16/FP32
            - Computation: Hidden_BF16 @ Weight_FP8 * proj_w_scale
            - Both static_in_scale and row_in_scale are None

        CTE case (with input quantization):
            - Input is quantized to FP8
            - ROW quantization: Hidden_FP8 @ Weight_FP8 * proj_w_scale * row_in_scale
            - STATIC quantization: Hidden_FP8 @ Weight_FP8 * proj_w_scale * static_in_scale
        """
        if row_in_scale is None and static_in_scale is None:
            # TKG: No input quantization, only weight dequantization
            return scale_with_broadcast(input @ proj_w, proj_w_scale)

        # CTE: Input is quantized, apply both weight and input dequantization
        if quantization_type == QuantizationType.ROW:
            return scale_with_broadcast(scale_with_broadcast(input @ proj_w, proj_w_scale), row_in_scale)
        elif is_static_quant:
            return scale_with_broadcast(input @ proj_w, proj_w_scale * static_in_scale)

    def perform_fused_add_norm(hiddens, gamma, norm_b=None):
        sum = np.zeros(hiddens[0].shape)
        for hidden in hiddens:
            sum += hidden
        sum_out = sum
        if norm_b:
            sum = norm_name2func[norm_type](sum, gamma, norm_b=norm_b)
        else:
            sum = norm_name2func[norm_type](sum, gamma)
        return sum, sum_out

    if down_proj_lhs_rhs_swap_optimized_layout:
        assert lnc is not None
        # Have layout friendly shape here, need to put it back for golden calculation
        I, H = down.shape
        down = down.reshape((I, lnc, H // 128 // lnc, 128)).transpose((0, 1, 3, 2)).reshape((I, H))

    if quantization_type == QuantizationType.STATIC_MX:
        gate = undo_mx_gate_up_w_reshape(gate)
        up = undo_mx_gate_up_w_reshape(up)
        down = undo_mx_down_w_reshape(down)

    is_input_quantized = any(hidden.dtype == quant_dtype for hidden in hiddens)
    # Handle input quantization based on execution mode
    if quantize_activation:
        # No input or intermediate quantization
        # - Input remains in original dtype (BF16/FP32)
        # - Only weights are quantized (FP8)
        # - Intermediate results (gate*up output) remain unquantized
        input, sum_out = perform_fused_add_norm(hiddens, gamma)
        input_quant_scale = None
        gate_in_scale = up_in_scale = down_in_scale = None
    else:
        # Full quantization pipeline
        # - Input is quantized to FP8 (ROW or STATIC)
        # - Intermediate results are quantized to FP8
        # - Both input and weight scales are applied during projection
        if quantization_type == QuantizationType.ROW and (hiddens[0].shape[-1] - up.shape[0] == 4):
            # input is quantized
            b, s, h = hiddens[0].shape
            input = np.zeros((b, s, h - 4))
            input_quant_scale = np.zeros((b, s, 1))
            for hidden in hiddens:
                input += hidden[..., :-4]
                input_quant_scale += hidden.astype(quant_dtype).view(np.float32)[..., -1, None]
            sum_out = input_quant_scale
        else:
            norm_out, sum_out = perform_fused_add_norm(hiddens, gamma)
            if is_input_quantized:
                input = norm_out
                input_quant_scale = None
            else:
                input, input_quant_scale = perform_tensor_quant(norm_out, clip, gate_in_scale)
            assert not is_static_quant or np.all(gate_in_scale == up_in_scale)

    act_fn = act_fn_type2func[act_fn_type]

    if not skip_gate:
        gate_out = perform_projection(input, gate, gate_w_scale, gate_in_scale, input_quant_scale)
        if gate_b is not None:
            gate_out += gate_b
        if gate_clamp_upper_limit is not None:
            gate_out = np.minimum(gate_out, gate_clamp_upper_limit)
        if gate_clamp_lower_limit is not None:
            gate_out = np.maximum(gate_out, gate_clamp_lower_limit)
        gate_act_fn_out = act_fn(gate_out)
        up_out = perform_projection(input, up, up_w_scale, up_in_scale, input_quant_scale)
        if up_b is not None:
            up_out += up_b
        if up_clamp_upper_limit is not None:
            up_out = np.minimum(up_out, up_clamp_upper_limit)
        if up_clamp_lower_limit is not None:
            up_out = np.maximum(up_out, up_clamp_lower_limit)
        intermediate_out = gate_act_fn_out * up_out
    else:
        up_out = perform_projection(input, up, up_w_scale, up_in_scale, input_quant_scale)
        if up_b is not None:
            up_out += up_b
        if up_clamp_upper_limit is not None:
            up_out = np.minimum(up_out, up_clamp_upper_limit)
        if up_clamp_lower_limit is not None:
            up_out = np.maximum(up_out, up_clamp_lower_limit)
        intermediate_out = act_fn(up_out)

    # Handle down projection based on execution mode
    if quantize_activation:
        # Intermediate output is not quantized, only weight dequantization
        output = perform_projection(intermediate_out, down, down_w_scale, down_in_scale, None)
    else:
        # Quantize intermediate output before down projection
        quantized_inter_out, inter_quant_scale = perform_tensor_quant(intermediate_out, clip, down_in_scale)
        quantized_inter_out = quantized_inter_out.astype(quant_dtype).astype(nl.float32)
        output = perform_projection(quantized_inter_out, down, down_w_scale, down_in_scale, inter_quant_scale)

    if down_b is not None:
        output += down_b

    return output.astype(dtype), sum_out.astype(dtype)


def golden_quant_mlp(
    inp_np,
    norm_type,
    fused_add,
    store_add,
    dtype,
    quant_dtype,
    quantization_type=QuantizationType.ROW,
    skip_gate=False,
    act_fn_type=ActFnType.SiLU,
    lnc=None,
    down_proj_lhs_rhs_swap_optimized_layout=False,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    quantize_activation=False,
):
    if fused_add:
        fused_add_tensor = inp_np["fused_add_tensor"]
    hidden = inp_np["hidden_tensor"]
    gamma = inp_np["normalization_weights_tensor"] if "normalization_weights_tensor" in inp_np else None
    gate_w = inp_np["gate_proj_weights_tensor"]
    up_w = inp_np["up_proj_weights_tensor"]
    down_w = inp_np["down_proj_weights_tensor"]
    gate_b = inp_np["gate_proj_bias_tensor"] if "gate_proj_bias_tensor" in inp_np else None
    up_b = inp_np["up_proj_bias_tensor"] if "up_proj_bias_tensor" in inp_np else None
    down_b = inp_np["down_proj_bias_tensor"] if "down_proj_bias_tensor" in inp_np else None
    norm_b = inp_np["norm_b"] if "norm_b" in inp_np else None
    gate_in_scale = up_in_scale = down_in_scale = None
    if quantization_type in [QuantizationType.STATIC, QuantizationType.STATIC_MX]:
        if "gate_up_in_scale" in inp_np:
            gate_in_scale = inp_np["gate_up_in_scale"]
            up_in_scale = inp_np["gate_up_in_scale"]
        else:
            gate_in_scale = inp_np["gate_in_scale"]
            up_in_scale = inp_np["up_in_scale"]
        down_in_scale = inp_np["down_in_scale"]
    gate_w_scale = inp_np["gate_w_scale"]
    up_w_scale = inp_np["up_w_scale"]
    down_w_scale = inp_np["down_w_scale"]
    hiddens = [fused_add_tensor, hidden] if fused_add else [hidden]
    store_add_valid = fused_add and store_add

    is_mx_quant = is_dtype_mx(quant_dtype) if quant_dtype is not None else False
    if is_mx_quant:
        mlp_out, add_out = norm_mlp_ref_mx(
            hiddens,
            gamma,
            gate_w,
            gate_w_scale,
            up_w,
            up_w_scale,
            down_w,
            down_w_scale,
            dtype,
            norm_type=norm_type,
            skip_gate=skip_gate,
            act_fn_type=act_fn_type,
            gate_b=gate_b,
            up_b=up_b,
            down_b=down_b,
            norm_b=norm_b,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
        )
    else:
        mlp_out, add_out = norm_quant_mlp_ref(
            hiddens,
            gamma,
            gate_w,
            up_w,
            down_w,
            quantization_type,
            gate_w_scale,
            up_w_scale,
            down_w_scale,
            gate_in_scale,
            up_in_scale,
            down_in_scale,
            dtype,
            quant_dtype,
            norm_type=norm_type,
            store_add=store_add_valid,
            skip_gate=skip_gate,
            act_fn_type=act_fn_type,
            gate_b=gate_b,
            up_b=up_b,
            down_b=down_b,
            down_proj_lhs_rhs_swap_optimized_layout=down_proj_lhs_rhs_swap_optimized_layout,
            lnc=lnc,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            quantize_activation=quantize_activation,
        )
    out_dict = {"out": dt.static_cast(mlp_out, dtype)}
    if store_add_valid:
        out_dict["add_out"] = dt.static_cast(add_out, dtype)

    return out_dict


def modify_down_proj_lhs_rhs_swap_unit_stride_layout(tensor_template, tensor, lnc):
    if tensor_template.name == "down_w":
        I, H = tensor.shape
        kernel_assert(
            H // 128 >= lnc,
            f"Hidden dimension {H} must be at least {128 * lnc} to avoid zero dimension in reshape",
        )
        tensor = tensor.reshape((I, lnc, 128, H // 128 // lnc)).transpose((0, 1, 3, 2)).reshape((I, H))
    return tensor


def modify_for_row_quant(tensor_template, tensor, lnc):
    rng = np.random.default_rng(0)
    if tensor_template.name == "hidden":
        B, S, H_PLUS_4 = tensor.shape
        single_row = rng.normal(size=(1, 1, 1)) * 10.0
        single_row_split = single_row.astype(np.float32).view(nl.float8_e4m3).astype(tensor_template.dtype)
        while not np.isfinite(single_row_split).all():
            single_row = rng.normal(size=(1, 1, 1)) * 10.0
            single_row_split = single_row.astype(np.float32).view(nl.float8_e4m3).astype(tensor_template.dtype)
        full_scale = single_row_split.repeat(B, axis=0).repeat(S, axis=1)
        tensor[:, :, -4:] = full_scale
    elif "scale" in tensor_template.name:
        P, F = tensor.shape
        single_row = rng.normal(size=(1, 1)) * 10.0
        tensor = single_row.repeat(P, axis=0).repeat(F, axis=1).astype(tensor_template.dtype)
    return tensor


def modify_fp8_static_scale(tensor_template, tensor, lnc):
    rng = np.random.default_rng(0)
    if "scale" in tensor_template.name:
        scale = rng.normal() * 0.5
        return np.full(tensor_template.shape, scale, dtype=tensor_template.dtype)
    else:
        return tensor


def random_lhs_and_random_bound_weight_tensor_generator(weight_lower, weight_upper, modifier_fn=None, lnc=None):
    # make tests generate stable tensors
    rng = np.random.default_rng(42)

    @update_func_str()
    def tensor_generator(shape, dtype, name):
        """Generate tensor with specified shape, dtype, and name.

        Args:
            shape: Tuple specifying tensor dimensions
            dtype: Data type for the tensor
            name: Name for the tensor
        """
        if name in ("down_w", "gate_w", "up_w"):
            tensor = rng.uniform(weight_lower, weight_upper, shape).astype(dtype)
            if modifier_fn is not None:
                assert lnc is not None
                tensor = modifier_fn(
                    tensor_template=TensorTemplate(name=name, shape=shape, dtype=dtype),
                    tensor=tensor,
                    lnc=lnc,
                )
            return tensor
        elif name == "hidden":
            single_row = rng.uniform(weight_lower, weight_upper, size=(1, 1, shape[-1])).astype(dtype)
            full_tensor = single_row.repeat(shape[0], axis=0).repeat(shape[1], axis=1)
            return full_tensor
        elif name == "fused_add_tensor":
            return rng.uniform(weight_lower, weight_upper, shape).astype(dtype)
        elif name in ("fused_add_tensor, gate_up_in_scale", "down_w_scale"):
            return np.full(
                shape=shape,
                fill_value=rng.random() * 0.01,
                dtype=dtype,
            )
        else:
            return np.full(
                shape=shape,
                fill_value=rng.random(),
                dtype=dtype,
            )

    return tensor_generator
