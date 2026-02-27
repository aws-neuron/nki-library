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
from test.integration.nkilib.core.mlp.test_mlp_common import dequantize_mx_golden, quantize_mx_golden
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
    generate_stabilized_mx_data,
    update_func_str,
)
from test.integration.nkilib.utils.test_kernel_common import norm_name2func
from test.utils.mx_utils import nc_matmul_mx_golden
from typing import Optional

import nki.dtype as nt
import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.utils.common_types import (
    NormType,
    QKVOutputLayout,
    QuantizationType,
)

DUMMY_TENSOR_NAME = "dummy"

VNC_DEGREE_DIM_NAME = "vnc_degree"
BATCH_DIM_NAME = "batch"
SEQUENCE_LEN_DIM_NAME = "seqlen"
HIDDEN_DIM_NAME = "hidden_dim"
N_Q_HEADS_DIM_NAME = "n_q_heads"
N_KV_HEADS_DIM_NAME = "n_kv_heads"
D_HEAD_DIM_NAME = "d_head"
NORM_TYPE_DIM_NAME = "norm_type"
FUSED_ADD_DIM_NAME = "fused_add"
OUTPUT_LAYOUT_DIM_NAME = "output_layout"
QUANTIZATION_TYPE_DIM_NAME = "quantization_type"

_q_width = 4
p_max = 128


def norm_qkv_ref(
    hidden,
    gamma,
    qkv_weights,
    dtype,
    norm_type=NormType.RMS_NORM,
    quantization_type=QuantizationType.NONE,
    qkv_w_scale=None,
    qkv_in_scale=None,
    eps=1e-6,
    bias=None,
    norm_b=None,
    hidden_actual=None,
    head_dim=None,
    n_kv_heads=None,
    n_q_heads=None,
):
    if norm_type == NormType.RMS_NORM:
        hidden = norm_name2func[norm_type](hidden, gamma, eps=eps, norm_b=norm_b, hidden_actual=hidden_actual)
    else:
        hidden = norm_name2func[norm_type](hidden, gamma, eps=eps, norm_b=norm_b)

    if quantization_type == QuantizationType.STATIC:
        hidden /= qkv_in_scale
        hidden = hidden.clip(-240, 240)
    qkv_out = hidden @ qkv_weights.astype(dtype)
    if quantization_type == QuantizationType.STATIC:
        # qkv_out shape is B, S, n_heads * d
        qkv_idx = [
            0,
            n_q_heads * head_dim,
            (n_q_heads + n_kv_heads) * head_dim,
            (n_q_heads * 2 + n_kv_heads) * head_dim,
        ]
        qkv_out[:, :, qkv_idx[0] : qkv_idx[1]] *= qkv_w_scale[0]
        qkv_out[:, :, qkv_idx[1] : qkv_idx[2]] *= qkv_w_scale[1]
        qkv_out[:, :, qkv_idx[2] : qkv_idx[3]] *= qkv_w_scale[2]
        qkv_out *= qkv_in_scale
    if bias is not None:
        qkv_out += bias
    return qkv_out


def norm_qkv_ref_mx(
    hidden,
    gamma,
    qkv_weights,
    norm_type=NormType.RMS_NORM,
    qkv_w_scale=None,
    eps=1e-6,
    bias=None,
    norm_b=None,
    hidden_actual=None,
    is_input_swizzled=False,
):
    b, s, H = hidden.shape

    if is_input_swizzled:
        hidden = hidden.reshape(b * s, _q_width, H // (p_max * _q_width), p_max).transpose(0, 2, 3, 1).reshape(b, s, H)

    if norm_type == NormType.RMS_NORM:
        hidden = norm_name2func[norm_type](hidden, gamma, eps=eps, norm_b=norm_b, hidden_actual=hidden_actual)
    else:
        hidden = norm_name2func[norm_type](hidden, gamma, eps=eps, norm_b=norm_b)

    _, f_qkv = qkv_w_scale.shape
    hidden_reshape = (
        hidden.reshape(b * s, H)
        .T.reshape(H // _q_width, _q_width, b * s)
        .transpose(0, 2, 1)
        .reshape(H // _q_width, _q_width * b * s)
    )

    hidden_mx, hidden_scale = quantize_mx_golden(in_tensor=hidden_reshape, out_x4_dtype=nl.float8_e4m3fn_x4)

    qkv_out = nc_matmul_mx_golden(
        stationary_x4=hidden_mx,
        moving_x4=qkv_weights,
        stationary_scale=hidden_scale,
        moving_scale=qkv_w_scale,
    )
    qkv_out = qkv_out.reshape(b, s, f_qkv)
    if bias is not None:
        qkv_out += bias
    return qkv_out


def build_qkv_input(
    batch: int,
    seqlen: int,
    hidden_dim: int,
    fused_qkv_dim: int,
    dtype,
    d_head: Optional[int] = None,
    eps: Optional[float] = None,
    norm_type: NormType = NormType.RMS_NORM,
    fused_add: bool = True,
    lnc_degree: int = 1,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    quantization_type: QuantizationType = QuantizationType.NONE,
    use_dma_transpose: bool = True,
    qkv_bias: Optional[bool] = False,
    norm_bias: Optional[bool] = False,
    hidden_actual: Optional[int] = None,
    fused_rope: Optional[bool] = False,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    tensor_gen=gaussian_tensor_generator(),
    fp8_kv_cache: bool = False,
    max_seq_len: Optional[int] = None,
    k_scale_val: Optional[float] = None,
    v_scale_val: Optional[float] = None,
    fp8_max: float = 240.0,
    fp8_min: float = -240.0,
    use_block_kv: bool = False,
    num_blocks: Optional[int] = None,
    block_size: Optional[int] = None,
    slot_mapping: Optional[np.ndarray] = None,
    is_input_swizzled: bool = False,
):
    qkv_w_scale = None
    qkv_in_scale = None
    if quantization_type == QuantizationType.MX:
        np.random.seed(0)
        dequant_weights, fused_qkv_weights, qkv_w_scale = generate_stabilized_mx_data(
            nl.float8_e4m3fn_x4, (hidden_dim // _q_width, fused_qkv_dim * _q_width), val_range=5
        )

        qkv_w_scale = qkv_w_scale.reshape(-1, fused_qkv_dim)
        input_tensor = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="input")
        if is_input_swizzled:
            input_tensor = (
                input_tensor.reshape(batch, seqlen, hidden_dim // (p_max * _q_width), p_max, _q_width)
                .transpose(0, 1, 4, 2, 3)
                .reshape(batch, seqlen, hidden_dim)
            )
    else:
        input_tensor = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="input")
        weight_dtype = dtype if quantization_type == QuantizationType.NONE else nt.float8_e4m3
        fused_qkv_weights = tensor_gen(shape=(hidden_dim, fused_qkv_dim), dtype=weight_dtype, name="fused_qkv_weights")
        if quantization_type == QuantizationType.STATIC:
            # Clip input and weights to [-1, 1] for proper quantization
            input_tensor = np.clip(input_tensor, -1.0, 1.0).astype(input_tensor.dtype)
            fused_qkv_weights = np.clip(fused_qkv_weights, -1.0, 1.0).astype(fused_qkv_weights.dtype)
            # Set scales to 1/240 with small deviation
            base_scale = 1.0 / 240.0
            np.random.seed(42)  # For reproducibility
            # Add small per-element deviation (±0.5%)
            qkv_w_scale = base_scale * np.random.uniform(0.995, 1.005, (128, 3)).astype(np.float32)
            qkv_in_scale = base_scale * np.random.uniform(0.995, 1.005, (128, 1)).astype(np.float32)

    mlp_prev = tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="mlp_prev") if fused_add else None
    attention_prev = (
        tensor_gen(shape=(batch, seqlen, hidden_dim), dtype=dtype, name="attention_prev") if fused_add else None
    )
    gamma_norm_weights = (
        tensor_gen(shape=(1, hidden_dim), dtype=dtype, name="gamma_norm_weights")
        if norm_type in [NormType.RMS_NORM, NormType.LAYER_NORM]
        else None
    )
    bias = tensor_gen(shape=(1, fused_qkv_dim), dtype=dtype, name="bias") if qkv_bias else None
    layer_norm_bias = tensor_gen(shape=(1, hidden_dim), dtype=dtype, name="layer_norm_bias") if norm_bias else None

    cos_cache_t = tensor_gen(shape=(batch, seqlen, d_head), dtype=dtype, name="cos_cache") if fused_rope else None
    sin_cache_t = tensor_gen(shape=(batch, seqlen, d_head), dtype=dtype, name="sin_cache") if fused_rope else None

    k_cache = None
    v_cache = None
    k_scale = None
    v_scale = None
    kv_dtype = None
    kv_dim = num_kv_heads * d_head if num_kv_heads and d_head else None
    if fp8_kv_cache:
        kv_dtype = nt.float8_e4m3
        k_scale = np.full((128, 1), k_scale_val, dtype=np.float32)
        v_scale = np.full((128, 1), v_scale_val, dtype=np.float32)
        if use_block_kv:
            k_cache = np.zeros((num_blocks, block_size, kv_dim), dtype=nt.float8_e4m3)
            v_cache = np.zeros((num_blocks, block_size, kv_dim), dtype=nt.float8_e4m3)
        else:
            k_cache = np.zeros((batch, max_seq_len, kv_dim), dtype=nt.float8_e4m3)
            v_cache = np.zeros((batch, max_seq_len, kv_dim), dtype=nt.float8_e4m3)

    result = {
        "input": input_tensor,
        "fused_qkv_weights": fused_qkv_weights,
        "output_layout": output_layout,
        "bias": bias,
        "quantization_type": quantization_type,
        "qkv_w_scale": qkv_w_scale,
        "qkv_in_scale": qkv_in_scale,
        "fused_residual_add": fused_add,
        "mlp_prev": mlp_prev,
        "attention_prev": attention_prev,
        "fused_norm_type": norm_type,
        "gamma_norm_weights": gamma_norm_weights,
        "layer_norm_bias": layer_norm_bias,
        "norm_eps": eps,
        "hidden_actual": hidden_actual,
        "fused_rope": fused_rope,
        "cos_cache": cos_cache_t,
        "sin_cache": sin_cache_t,
        "d_head": d_head,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "store_output_in_sbuf": False,
        "sbm": None,
        "use_auto_allocation": False,
        "load_input_with_DMA_transpose": use_dma_transpose,
    }

    if fp8_kv_cache:
        result["k_cache.must_alias_input"] = k_cache
        result["v_cache.must_alias_input"] = v_cache
        result["k_scale"] = k_scale
        result["v_scale"] = v_scale
        result["fp8_max"] = fp8_max
        result["fp8_min"] = fp8_min
        result["kv_dtype"] = kv_dtype
        if use_block_kv:
            result["use_block_kv"] = True
            result["block_size"] = block_size
            result["slot_mapping"] = slot_mapping

    return result


def rope_gaussian_tensor_generator(mean=0.0, std=1.0):
    """Create a Gaussian tensor generator with special handling for RoPE cache tensors.

    Args:
        mean (float, optional): The mean (center) of the Gaussian distribution.
                               Defaults to 0.0.
        std (float, optional): The standard deviation (spread) of the Gaussian
                              distribution. Defaults to 1.0.

    Returns:
        callable: A tensor generator function with signature (shape, dtype, name) -> np.ndarray
                 that generates Gaussian-distributed tensors with special RoPE cache handling.

    Behavior:
        - For cos_cache and sin_cache tensors:
          * Generates tensor with last dimension halved
          * Tiles the tensor to duplicate values (required for RoPE implementation)
          * Final shape matches requested shape
        - For all other tensors:
          * Standard Gaussian distribution with specified mean/std

    Example:
        >>> generator = rope_gaussian_tensor_generator(mean=0.0, std=1.0)
        >>> cos_cache = generator(shape=(128, 64), dtype=np.float32, name="cos_cache")
        >>> # Generates (128, 32) tensor, then tiles to (128, 64)
    """
    rng = np.random.default_rng(0)

    @update_func_str(mean=mean, std=std)
    def tensor_generator(shape, dtype, name):
        guessed_dtype = dtype
        if name in ["cos_cache", "sin_cache"]:
            tensor_template_shape = list(shape)
            tensor_template_shape[-1] = tensor_template_shape[-1] // 2
            tensor = (rng.normal(size=tensor_template_shape) * std + mean).astype(guessed_dtype)
            tensor = np.tile(tensor, 2)
            return tensor
        else:
            return (rng.normal(size=shape) * std + mean).astype(guessed_dtype)

    return tensor_generator
