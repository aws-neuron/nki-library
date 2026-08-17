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

"""Integration tests for the output projection CTE kernel using UnitTestFramework."""

import functools
from typing import Optional, TypedDict, final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.output_projection.output_projection_cte import output_projection_cte
from nkilib_src.nkilib.core.output_projection.output_projection_cte.output_projection_cte_torch import (
    output_projection_cte_mx_torch_ref,
    output_projection_cte_torch_ref,
)
from nkilib_src.nkilib.core.utils.common_types import DtypeMode, OProjAttentionLayout, QuantizationType

try:
    from test.integration.nkilib.core.output_projection.test_output_proj_cte_model_config import (
        get_mx_weight_dtype,
        output_proj_cte_model_configs,
    )
except ImportError:
    output_proj_cte_model_configs = {}

    def get_mx_weight_dtype(configs=None):
        return "fp4"


from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
    generate_stabilized_mx_data,
    np_random_sample,
    np_random_sample_static_quantize_inp,
)
from test.integration.nkilib.utils.test_kernel_common import resolve_dtype_mode_for_torch_ref
from test.utils.common_dataclasses import (
    CompilerArgs,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Map the model-config MX weight-dtype token to the nl dtype the input generator expects.
_MX_WEIGHT_DTYPE_BY_TOKEN = {
    "fp4": nl.float4_e2m1fn_x4,
    "fp8": nl.float8_e4m3fn_x4,
}
_MODEL_MX_WEIGHT_DTYPE = _MX_WEIGHT_DTYPE_BY_TOKEN[get_mx_weight_dtype()]


def generate_output_proj_cte_inputs(
    batch: int,
    seqlen: int,
    hidden: int,
    n_head: int,
    d_head: int,
    test_bias: bool,
    quantization_type: QuantizationType = QuantizationType.NONE,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    attention_layout: Optional[OProjAttentionLayout] = None,
) -> dict:
    """Generate inputs for output projection CTE test.

    ``attention_layout=None`` is passed through to the kernel unchanged so tests exercise
    the pre-tag default; the generated tensor follows the layout that resolves to.
    """
    dtype = nl.bfloat16
    input_scales = None
    weight_scales = None

    random_gen = np_random_sample()
    attention = random_gen(shape=(batch, n_head, d_head, seqlen), dtype=dtype)
    bias = gaussian_tensor_generator(std=1)(shape=(1, hidden), dtype=dtype, name="bias") if test_bias else None

    if quantization_type == QuantizationType.NONE:
        weight = random_gen(shape=(n_head * d_head, hidden), dtype=dtype)
    else:
        # Pick E4M3 dtype based on dtype_mode for STATIC/ROW. MX-family always OCP.
        if quantization_type in (QuantizationType.STATIC_MX, QuantizationType.ROW_MX):
            quant_dtype = nl.float8_e4m3fn
        elif dtype_mode == DtypeMode.OCP:
            quant_dtype = nl.float8_e4m3fn
        else:
            quant_dtype = nl.float8_e4m3
        static_quant_gen = np_random_sample_static_quantize_inp()
        weight, weight_scale_val, input_scale_val = static_quant_gen(shape=(n_head * d_head, hidden), dtype=quant_dtype)

        if quantization_type == QuantizationType.ROW_MX:
            # Per-row weight dequant scale: [128, H], no input scales
            weight_scales = np.broadcast_to(
                np.random.random_sample((1, hidden)).astype(np.float32) / 512.0, (128, hidden)
            ).copy()
        elif quantization_type == QuantizationType.ROW:
            # Per-row weight dequant scale: [128, H], no input scales
            weight_scales = np.broadcast_to(
                np.random.random_sample((1, hidden)).astype(np.float32) / 256.0, (128, hidden)
            ).copy()
        else:
            input_scales = np.full(shape=(128, 1), fill_value=input_scale_val, dtype=np.float32)
            weight_scales = np.full(shape=(128, 1), fill_value=weight_scale_val, dtype=np.float32)

        if quantization_type in (QuantizationType.STATIC_MX, QuantizationType.ROW_MX):
            nd = n_head * d_head
            if nd % 4 == 0:
                w = weight.reshape(nd // 4, 4, hidden)
                weight = np.ascontiguousarray(np.transpose(w, (0, 2, 1))).reshape(nd, hidden)

    # Mirror the kernel's None resolution so the tensor matches the layout it will read.
    resolved_layout = attention_layout
    if resolved_layout is None:
        resolved_layout = (
            OProjAttentionLayout.BSNd if quantization_type == QuantizationType.ROW else OProjAttentionLayout.BNdS
        )

    if resolved_layout == OProjAttentionLayout.BSNd:
        # [B, N, D, S] -> [B, S, N, D]
        attention = np.ascontiguousarray(attention.transpose(0, 3, 1, 2))
    elif resolved_layout == OProjAttentionLayout.BNSd:
        # [B, N, D, S] -> [B, N, S, D]
        attention = np.ascontiguousarray(attention.transpose(0, 1, 3, 2))

    return {
        "attention": attention,
        "weight": weight,
        "bias": bias,
        "quantization_type": quantization_type,
        "input_scales": input_scales,
        "weight_scales": weight_scales,
        "dtype_mode": dtype_mode,
        "attention_layout": attention_layout,
    }


def generate_output_proj_cte_mx_inputs(
    batch: int,
    seqlen: int,
    hidden: int,
    n_head: int,
    d_head: int,
    input_prequantized: bool = False,
    test_bias: bool = False,
    weight_dtype=nl.float4_e2m1fn_x4,
) -> dict:
    """Generate inputs for MX output projection CTE test (fp4 default, fp8 via weight_dtype)."""
    dtype = nl.bfloat16
    np.random.seed(42)

    bias = gaussian_tensor_generator(std=100)(shape=(1, hidden), dtype=dtype, name="bias") if test_bias else None

    weight_logical_shape = (n_head * d_head // 4, hidden * 4)
    # Skip MX data generation for invalid shapes - use zeros instead
    if weight_logical_shape[0] % 8 != 0:
        weight_quantized = np.zeros((n_head * d_head // 4, hidden), dtype=np.uint8)
        weight_scale = np.zeros((n_head * d_head // 32, hidden), dtype=np.uint8)
    else:
        _, weight_quantized, weight_scale = generate_stabilized_mx_data(
            weight_dtype, weight_logical_shape, val_range=1.0
        )

    if input_prequantized:
        # Pre-quantized input path: generate float8_e4m3fn_x4 attention with scales
        # Input shape is [B, 1, D_packed, S] where D_packed = n_head * d_head // 4
        # Kernel uses contraction_dim = N * D_packed = 1 * D_packed (no additional //4 division)
        packed_d = n_head * d_head // 4
        attention_logical_shape = (packed_d, seqlen * 4)
        attention_list, scale_list = [], []
        for _ in range(batch):
            if attention_logical_shape[0] % 8 != 0:
                attn_q = np.zeros((packed_d, seqlen), dtype=np.uint8)
                attn_s = np.zeros((packed_d // 8, seqlen), dtype=np.uint8)
            else:
                _, attn_q, attn_s = generate_stabilized_mx_data(
                    nl.float8_e4m3fn_x4, attention_logical_shape, val_range=1.0
                )
            attention_list.append(attn_q.reshape(1, 1, packed_d, seqlen))
            scale_list.append(attn_s.reshape(1, packed_d // 8, seqlen))
        attention = np.concatenate(attention_list, axis=0)
        input_scales = np.concatenate(scale_list, axis=0)
    else:
        attention = (np.random.randn(batch, n_head, d_head, seqlen) * 0.1).astype(np.float32)
        attention = attention.astype(np.float16).view(np.uint16).view(np.float16).astype(nl.bfloat16)
        input_scales = None

    return {
        "attention": attention,
        "weight": weight_quantized,
        "bias": bias,
        "quantization_type": QuantizationType.MX,
        "input_scales": input_scales,
        "weight_scales": weight_scale,
    }


def generate_output_proj_cte_mx_compact_inputs(
    batch: int,
    seqlen: int,
    hidden: int,
    n_head: int,
    d_head: int,
    test_bias: bool = False,
) -> dict:
    """Generate inputs for MX FP8 output projection CTE with block-128 compact scales.

    Weights are float8_e4m3fn_x4 with one uint8 scale per 128x128 block. Input
    is bf16 (online quantization). Tests the compact scale path.

    The kernel and torch reference both interpret the compact scale identically
    (broadcast to dense block-32 layout), so we don't need to stabilize the
    weight bytes — random fp8x4 bytes plus random compact scales produce
    matching results in both paths.
    """
    dtype = nl.bfloat16
    np.random.seed(42)
    nd = n_head * d_head
    DS_SCALE_BLOCK = 128

    bias = gaussian_tensor_generator(std=100)(shape=(1, hidden), dtype=dtype, name="bias") if test_bias else None

    is_invalid = (nd % DS_SCALE_BLOCK != 0) or (hidden % DS_SCALE_BLOCK != 0)

    if is_invalid:
        weight_quantized = np.zeros((nd // 4, hidden), dtype=np.uint8)
        compact_scales = np.zeros((max(nd // DS_SCALE_BLOCK, 1), max(hidden // DS_SCALE_BLOCK, 1)), dtype=np.uint8)
    else:
        # quantize to fp8x4 with the standard MX block-32 pipeline, then collapse the dense scales to
        # compact block-128 by taking the per-block max.
        from nkilib_src.nkilib.core.utils.mx_torch_common import quantize_to_mx

        # generate values within fp8 representable range (max=448).
        fp_weight = (np.random.random((nd // 4, hidden * 4)) * 2 - 1) * 16.0
        weight_quantized, dense_scale = quantize_to_mx(fp_weight.astype(np.float32), nl.float8_e4m3fn_x4)

        ds_view = dense_scale.reshape(nd // DS_SCALE_BLOCK, 4, hidden // DS_SCALE_BLOCK, DS_SCALE_BLOCK)
        compact_scales = ds_view.max(axis=(1, 3)).astype(np.uint8)
        weight_quantized = np.asarray(weight_quantized)

    attention = (np.random.randn(batch, n_head, d_head, seqlen) * 0.1).astype(np.float32)
    attention = attention.astype(np.float16).view(np.uint16).view(np.float16).astype(nl.bfloat16)

    return {
        "attention": attention,
        "weight": weight_quantized,
        "bias": bias,
        "quantization_type": QuantizationType.MX,
        "input_scales": None,
        "weight_scales": compact_scales,
        "compact_weight_scales": True,
    }


# Manual test cases: (batch, seqlen, hidden, n_head, d_head, test_bias)
OUTPUT_PROJ_CTE_UNIT_CASES = [
    # New model, 2025-Jul
    (1, 16, 5120, 10, 128, True),
    (1, 128, 3072, 16, 64, True),
    (1, 1024, 3072, 16, 64, True),
    (1, 2048, 3072, 16, 64, True),
    (1, 10240, 3072, 8, 64, True),
    # Test cases to verify folding n_head into d_head
    (1, 128, 3072, 16, 10, True),  # group_size of 8
    (1, 128, 3072, 17, 10, True),  # Cannot reshape
    (1, 128, 3072, 8, 32, True),  # group_size of 4
    # 70B & 76B
    (1, 128, 8192, 1, 128, False),
    (1, 256, 8192, 1, 128, False),
    (1, 512, 8192, 1, 128, False),
    (1, 2048, 8192, 1, 128, False),
    (1, 4096, 8192, 1, 128, False),
    (1, 8192, 8192, 1, 128, False),
    (1, 16384, 8192, 1, 128, False),
    (1, 512, 8192, 2, 128, False),
    (1, 1024, 8192, 2, 128, False),
    (1, 2048, 8192, 2, 128, False),
    (1, 4096, 8192, 2, 128, False),
    (1, 8192, 8192, 2, 128, False),
    (1, 10240, 8192, 2, 128, False),
    (1, 16384, 8192, 2, 128, False),
    # 405B
    (1, 512, 16384, 1, 128, False),
    (1, 1024, 16384, 1, 128, False),
    (1, 2048, 16384, 1, 128, False),
    (1, 4096, 16384, 1, 128, False),
    (1, 8192, 16384, 1, 128, False),
    (1, 10240, 16384, 1, 128, False),
    (1, 16384, 16384, 1, 128, False),
    # Draft model
    (1, 1024, 1024, 1, 64, False),
    (1, 1024, 2048, 1, 64, False),
    (1, 1024, 3072, 1, 64, False),
    (1, 1024, 8192, 1, 64, False),
    # 470B model
    (1, 1024, 20480, 3, 128, False),
    # Text
    (4, 256, 7168, 4, 128, False),
    (4, 256, 7168, 1, 128, False),
    (1, 256, 7168, 1, 128, False),
    # arbitrary seqlen not aligned by 128
    (1, 128 + 64, 1024, 1, 128, False),
    (1, 256 + 64, 2048, 2, 128, False),
    (1, 512 + 64, 3072, 3, 128, False),
    (1, 1024 + 64, 7168, 4, 128, False),
    (1, 2048 + 120, 8192, 5, 128, False),
    (1, 4096 + 1000, 16384, 6, 128, False),
    # Test cases for d_head > 128 (D folded back into N)
    (1, 1024, 8192, 4, 256, False),
    (1, 1024, 3072, 8, 192, False),
    (1, 512, 8192, 2, 384, False),
    (1, 512, 3072, 4, 160, False),
    (1, 1024, 8192, 4, 256, True),
    (1, 1024, 3072, 8, 192, True),
    (1, 1024, 8192, 3, 256, False),
    (1, 512, 3072, 5, 192, True),
    (1, 128 + 64, 1024, 1, 256, False),
    (1, 256 + 64, 2048, 2, 256, False),
    (1, 512 + 64, 3072, 3, 256, False),
    (1, 1024 + 64, 7168, 4, 256, False),
    (1, 2048 + 120, 8192, 5, 256, False),
    (1, 4096 + 1000, 16384, 6, 256, False),
]

# BSNd cases: attention arrives untransposed as [B, S, N, D]
OUTPUT_PROJ_CTE_UNTRANSPOSED_ATTENTION_CASES = [
    (1, 8192, 3072, 1, 64, True),
    (1, 8192, 3072, 2, 64, True),
    (1, 8192, 3072, 4, 64, True),
    (1, 8192, 3072, 8, 64, True),
    (1, 8192, 3072, 16, 64, True),
    (1, 1024, 3072, 1, 64, True),
    (1, 1024, 3072, 8, 64, True),
    (1, 2048, 3072, 8, 64, True),
    (1, 16384, 3072, 8, 64, True),
    (1, 32768, 3072, 1, 64, True),
    (1, 576, 8192, 3, 128, False),
    (4, 512, 7168, 4, 128, False),
    (1, 4096, 8192, 2, 128, False),
    (1, 512, 3072, 16, 10, True),
    (1, 1024, 3072, 8, 192, True),
    (1, 1024, 8192, 4, 256, False),
]

# BNSd cases: attention arrives untransposed heads-outer as [B, N, S, D], i.e. the
# attention CTE output with heads folded into batch and tp_out=False.
OUTPUT_PROJ_CTE_HEADS_OUTER_ATTENTION_CASES = [
    (1, 8192, 3072, 1, 64, True),
    (1, 8192, 3072, 2, 64, True),
    (1, 8192, 3072, 8, 64, True),
    (1, 8192, 3072, 16, 64, True),
    (1, 1024, 3072, 1, 64, True),
    (1, 2048, 3072, 8, 64, True),
    (1, 16384, 3072, 8, 64, True),
    (1, 576, 8192, 3, 128, False),
    (4, 512, 7168, 4, 128, False),
    (1, 4096, 8192, 2, 128, False),
    # group_size > 1: N folds into D, so a packed tile spans several heads.
    (1, 512, 3072, 16, 10, True),
    (1, 128, 3072, 8, 32, True),
    # D > 128: D folds back into N, so a packed tile is a column band of one head.
    (1, 1024, 3072, 8, 192, True),
    (1, 1024, 8192, 4, 256, False),
]

# Slow unit cases (>2min compile time), run in full pipeline but not dry-run
OUTPUT_PROJ_CTE_SLOW_UNIT_CASES = [
    (1, 16384 + 4321, 16384, 9, 256, False),
    (1, 16384 + 4321, 16384, 9, 128, False),
    (1, 8192 + 1120, 16384, 7, 256, False),
    (1, 8192 + 1120, 16384, 7, 128, False),
    (1, 10240 + 1234, 16384, 8, 256, False),
    (1, 10240 + 1234, 16384, 8, 128, False),
]

OUTPUT_PROJ_CTE_UNIT_PARAMS = "batch, seqlen, hidden, n_head, d_head, test_bias"
_ABBREVS = {"batch": "b", "seqlen": "s", "hidden": "h", "n_head": "nh", "d_head": "dh", "test_bias": "bias"}


# Each method that consumes OUTPUT_PROJ_CTE_UNIT_CASES has its own set of fast keys
# `(batch, seqlen, hidden, n_head, d_head, test_bias)`. Robust to row reordering.
def _output_proj_cte_with_fast_keys(fast_keys):
    """Return OUTPUT_PROJ_CTE_UNIT_CASES with marks=fast on rows whose tuple
    matches one in fast_keys.
    """
    fk = frozenset(fast_keys)
    out = []
    for c in OUTPUT_PROJ_CTE_UNIT_CASES:
        if tuple(c) in fk:
            out.append(pytest.param(*c, marks=pytest.mark.fast))
        else:
            out.append(pytest.param(*c))
    return out


_OPROJ_CTE_UNIT_BF16_FAST = _output_proj_cte_with_fast_keys(
    {
        (1, 128, 3072, 16, 10, True),
        (1, 8192, 16384, 1, 128, False),
    }
)
_OPROJ_CTE_UNIT_STATIC_FP8_FAST = _output_proj_cte_with_fast_keys(
    {
        (1, 128, 3072, 16, 64, True),
        (1, 4096, 16384, 1, 128, False),
        (1, 1024, 8192, 4, 256, False),
        (1, 1024, 3072, 8, 192, False),
        (1, 1024, 8192, 3, 256, False),
        (1, 5096, 16384, 6, 256, False),
    }
)
_OPROJ_CTE_UNIT_MXFP4_FAST = _output_proj_cte_with_fast_keys(
    {
        (1, 1024, 3072, 16, 64, True),
        (1, 128, 3072, 16, 10, True),
        (1, 16384, 16384, 1, 128, False),
        (1, 1024, 2048, 1, 64, False),
        (4, 256, 7168, 1, 128, False),
        (1, 5096, 16384, 6, 128, False),
    }
)
_OPROJ_CTE_UNIT_MXFP4_PREQ_FAST = _output_proj_cte_with_fast_keys(
    {
        (1, 128, 3072, 16, 10, True),
        (1, 128, 3072, 17, 10, True),
        (1, 10240, 16384, 1, 128, False),
        (1, 1024, 3072, 1, 64, False),
        (1, 1024, 8192, 1, 64, False),
        (1, 1024, 20480, 3, 128, False),
    }
)
_OPROJ_CTE_UNIT_STATIC_MXFP8_FAST = _output_proj_cte_with_fast_keys(
    {
        (1, 128, 3072, 16, 10, True),
        (1, 128, 3072, 8, 32, True),
        (1, 4096, 16384, 1, 128, False),
        (1, 1024, 3072, 1, 64, False),
        (1, 1024, 20480, 3, 128, False),
        (1, 1024, 8192, 3, 256, False),
        (1, 5096, 16384, 6, 256, False),
    }
)
_OPROJ_CTE_UNIT_ROW_MXFP8_FAST = _output_proj_cte_with_fast_keys(
    {
        (1, 128, 3072, 16, 64, True),
        (1, 128, 3072, 16, 10, True),
        (1, 8192, 16384, 1, 128, False),
        (1, 1024, 8192, 1, 64, False),
        (1, 1024, 8192, 3, 256, False),
        (1, 5096, 16384, 6, 256, False),
    }
)
_OPROJ_CTE_UNIT_ROW_FP8_FAST = _output_proj_cte_with_fast_keys(
    {
        (1, 1024, 3072, 16, 64, True),
        (1, 128, 8192, 1, 128, False),
        (1, 1024, 3072, 8, 192, False),
        (1, 1024, 8192, 3, 256, False),
    }
)

_OPROJ_CTE_UNIT_MX_COMPACT_FAST = _output_proj_cte_with_fast_keys(
    {
        (1, 128, 8192, 1, 128, False),
        (1, 512, 8192, 1, 128, False),
        (1, 1024, 8192, 1, 128, False),
        (1, 2048, 8192, 1, 128, False),
        (1, 4096, 8192, 1, 128, False),
        (1, 8192, 8192, 1, 128, False),
        (1, 16384, 8192, 1, 128, False),
        (1, 128, 3072, 16, 64, True),
        (1, 1024, 16384, 1, 128, False),
        (1, 16384, 16384, 1, 128, False),
        (1, 1024, 8192, 4, 256, False),
        (1, 1024, 3072, 8, 192, False),
        (1, 1024, 8192, 3, 256, False),
        (4, 256, 7168, 4, 128, False),
        (1, 256, 7168, 1, 128, False),
        (1, 1024, 8192, 1, 64, False),
    }
)

# Kernel constraints
_MAX_B_TIMES_S = 128 * 1024
_MAX_H = 20705
_MAX_N = 17
_MAX_D = 256
_MAX_D_ROW = 128  # ROW quantization uses [B, S, N, D] layout, no D-folding support

# Max sizes to run validation on (to avoid OOM during testing)
_MAX_BxS_VALIDATE = 64 * 1024
_MAX_H_VALIDATE = 16384

_INT32_MAX = 2**31 - 1


def _exceeds_int32_tensor_elements(batch, seqlen, hidden, n_head, d_head):
    """Check if any tensor would exceed int32 element count, causing compiler overflow."""
    if batch * n_head * d_head * seqlen > _INT32_MAX:
        return True
    if batch * seqlen * hidden > _INT32_MAX:
        return True
    return False


def filter_output_proj_combinations(batch, seqlen, hidden, n_head, d_head, test_bias=None):
    """Filter out invalid parameter combinations for output projection kernel."""
    # Skip configs that overflow compiler int32 limits — can't even run as negative tests
    if _exceeds_int32_tensor_elements(batch, seqlen, hidden, n_head, d_head):
        return FilterResult.REDUNDANT
    if batch * seqlen > _MAX_B_TIMES_S:
        return FilterResult.INVALID
    if hidden > _MAX_H:
        return FilterResult.INVALID
    if n_head > _MAX_N:
        return FilterResult.INVALID
    if d_head > _MAX_D and d_head % 2 != 0:
        return FilterResult.INVALID
    return FilterResult.VALID


def filter_output_proj_mx_combinations(batch, seqlen, hidden, n_head, d_head, test_bias=None):
    """Filter out invalid parameter combinations for MX output projection kernel."""
    # Skip configs that overflow compiler int32 limits — can't even run as negative tests
    if _exceeds_int32_tensor_elements(batch, seqlen, hidden, n_head, d_head):
        return FilterResult.REDUNDANT
    if batch * seqlen > _MAX_B_TIMES_S:
        return FilterResult.INVALID
    if (n_head * d_head < 128) or (n_head * d_head % 128 != 0):
        return FilterResult.INVALID
    return FilterResult.VALID


# Seeded RNG for deterministic sweep test generation
_sweep_rng = np.random.default_rng(42)

# Sweep parameters
_SWEEP_BATCH = sorted(_sweep_rng.choice(range(1, 129), size=4, replace=False).tolist())
_SWEEP_SEQLEN = sorted(
    _sweep_rng.choice(range(16, 1024), size=4, replace=False).tolist()
    + _sweep_rng.choice(range(1024, 16 * 1024 + 1), size=4, replace=False).tolist()
)
_SWEEP_HIDDEN = BoundedRange(
    values=sorted(
        _sweep_rng.choice(range(128, 2048, 2), size=2, replace=False).tolist()
        + _sweep_rng.choice(range(2048, 16 * 1024 + 1, 2), size=2, replace=False).tolist()
    ),
    boundary_values=[_MAX_H + 2],
)
_SWEEP_N_HEAD = BoundedRange(
    values=sorted(_sweep_rng.choice(range(1, 18), size=4, replace=False).tolist()),
    boundary_values=[_MAX_N + 1],
)
_SWEEP_D_HEAD = BoundedRange(
    values=sorted(
        _sweep_rng.choice(range(1, 64), size=2, replace=False).tolist()
        + _sweep_rng.choice(range(64, 129), size=2, replace=False).tolist()
        + _sweep_rng.choice(range(128, 257, 2), size=2, replace=False).tolist()
    ),
    boundary_values=[_MAX_D + 1],
)


class OutputProjCteDtypeModeConfig(TypedDict):
    """Shapes and fusion settings shared by every dtype_mode canary case."""

    batch: int
    seqlen: int
    hidden: int
    n_head: int
    d_head: int
    test_bias: bool


@pytest_test_metadata(name="Output Projection CTE", tags=["model"])
@pytest_marks(["output_projection", "cte", "mx"])
@final
class TestOutputProjCteKernel:
    """Test class for output_projection_cte using UnitTestFramework."""

    # ============================================================================
    # Float (Non-Quantized) Tests
    # ============================================================================

    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, _OPROJ_CTE_UNIT_BF16_FAST, abbrevs=_ABBREVS)
    def test_output_proj_cte_bf16_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        platform_target: Platforms,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch, seqlen=seqlen, hidden=hidden, n_head=n_head, d_head=d_head, test_bias=test_bias
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_UNTRANSPOSED_ATTENTION_CASES, abbrevs=_ABBREVS)
    def test_output_proj_cte_bf16_untransposed_attention_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        platform_target: Platforms,
    ):
        """bf16 projection reading attention in the untransposed [B, S, N, D] layout."""
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                attention_layout=OProjAttentionLayout.BSNd,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest.mark.fast
    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_HEADS_OUTER_ATTENTION_CASES, abbrevs=_ABBREVS)
    def test_output_proj_cte_bf16_heads_outer_attention_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        platform_target: Platforms,
    ):
        """bf16 projection reading attention in the untransposed heads-outer [B, N, S, D] layout."""
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                attention_layout=OProjAttentionLayout.BNSd,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )

    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_SLOW_UNIT_CASES, abbrevs=_ABBREVS)
    def test_output_proj_cte_bf16_slow_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        platform_target: Platforms,
    ):
        self.test_output_proj_cte_bf16_unit(
            test_manager, collector, batch, seqlen, hidden, n_head, d_head, test_bias, platform_target
        )

    # ============================================================================
    # FP8 Static Quantization Tests
    # ============================================================================

    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, _OPROJ_CTE_UNIT_STATIC_FP8_FAST, abbrevs=_ABBREVS)
    def test_output_proj_cte_static_fp8_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        platform_target: Platforms,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                quantization_type=QuantizationType.STATIC,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.036,
            atol=1e-5,
        )

    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_SLOW_UNIT_CASES, abbrevs=_ABBREVS)
    def test_output_proj_cte_static_fp8_slow_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        platform_target: Platforms,
    ):
        self.test_output_proj_cte_static_fp8_unit(
            test_manager, collector, batch, seqlen, hidden, n_head, d_head, test_bias, platform_target
        )

    # ============================================================================
    # Sweep Tests
    # ============================================================================

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_SWEEP_BATCH,
        seqlen=_SWEEP_SEQLEN,
        hidden=_SWEEP_HIDDEN,
        n_head=_SWEEP_N_HEAD,
        d_head=_SWEEP_D_HEAD,
        test_bias=[True, False],
        filter=filter_output_proj_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_output_proj_cte_bf16_sweep(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        is_negative_test_case: bool,
        platform_target: Platforms,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch, seqlen=seqlen, hidden=hidden, n_head=n_head, d_head=d_head, test_bias=test_bias
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_SWEEP_BATCH,
        seqlen=_SWEEP_SEQLEN,
        hidden=_SWEEP_HIDDEN,
        n_head=_SWEEP_N_HEAD,
        d_head=_SWEEP_D_HEAD,
        test_bias=[True, False],
        filter=filter_output_proj_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_output_proj_cte_static_fp8_sweep(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        is_negative_test_case: bool,
        platform_target: Platforms,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                quantization_type=QuantizationType.STATIC,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.036,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # MX Quantization Tests
    # ============================================================================

    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, _OPROJ_CTE_UNIT_MXFP4_FAST, abbrevs=_ABBREVS)
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_output_proj_cte_mxfp4_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for MX FP4 output projection CTE kernel."""
        is_negative_test_case = (n_head * d_head < 128) or (n_head * d_head % 128 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_mx_inputs(batch, seqlen, hidden, n_head, d_head, test_bias=test_bias)

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=output_projection_cte_mx_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
            ),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_SWEEP_BATCH,
        seqlen=_SWEEP_SEQLEN,
        hidden=_SWEEP_HIDDEN,
        n_head=_SWEEP_N_HEAD,
        d_head=_SWEEP_D_HEAD,
        test_bias=[True, False],
        filter=filter_output_proj_mx_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_output_proj_cte_mxfp4_sweep(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        is_negative_test_case: bool,
    ):
        """Sweep test for MX FP4 output projection CTE kernel."""
        is_negative_test_case = (n_head * d_head < 128) or (n_head * d_head % 128 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_mx_inputs(batch, seqlen, hidden, n_head, d_head, test_bias=test_bias)

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=output_projection_cte_mx_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # MX Pre-Quantized Input Tests
    # ============================================================================

    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, _OPROJ_CTE_UNIT_MXFP4_PREQ_FAST, abbrevs=_ABBREVS)
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_output_proj_cte_mxfp4_prequantized_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for MX FP4 output projection with pre-quantized input."""
        n_d = n_head * d_head // 4
        is_negative_test_case = (n_d < 32) or (n_d % 32 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_mx_inputs(
                batch, seqlen, hidden, n_head, d_head, input_prequantized=True, test_bias=test_bias
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=output_projection_cte_mx_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # MX FP8 Compact (block-128) Tests
    # ============================================================================

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, _OPROJ_CTE_UNIT_MX_COMPACT_FAST, abbrevs=_ABBREVS)
    def test_output_proj_cte_mx_compact_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for MX FP8 output projection CTE with block-128 compact scales.

        Mirrors the qkv_cte_mla compact-scale path: weights are float8_e4m3fn_x4
        with one uint8 scale per 128x128 block, input is bf16 (online MX
        quantization on-device).
        """
        DS_SCALE_BLOCK = 128
        n_d = n_head * d_head
        is_negative_test_case = (n_d % DS_SCALE_BLOCK != 0) or (hidden % DS_SCALE_BLOCK != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_mx_compact_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=output_projection_cte_mx_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.parametrize(
        "alt_dtype, canonical_dtype",
        [(np.uint32, nl.float8_e4m3fn_x4), (np.uint16, nl.float4_e2m1fn_x4)],
        ids=["u32_fp8", "u16_fp4"],
    )
    @pytest.mark.parametrize(
        "hidden, n_head",
        # h512: single SBUF tile; h5120: forces H-block tiling + exercises the byte-budget guard.
        [(512, 1), (5120, 8)],
        ids=["h512", "h5120"],
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_output_proj_cte_mx_alt_dtype_weight_repro(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        alt_dtype,
        canonical_dtype,
        hidden,
        n_head,
    ):
        """Online MX accepts weights labeled with the same-width torch container dtype.

        Frameworks store x4-packed MX weights as uint32 (mxfp8) / uint16 (mxfp4)
        since torch has no packed MXFP dtypes. Same bytes, different label. The h5120
        case guards the SBUF byte-budget: the container label must be sized at its true
        width (uint32 = 4B) for the tiler, not the 2B default fallback.
        """
        batch, seqlen, d_head = 1, 128, 128
        dtype = nl.bfloat16

        def input_generator(test_config):
            kernel_input = generate_output_proj_cte_mx_inputs(
                batch, seqlen, hidden, n_head, d_head, test_bias=False, weight_dtype=canonical_dtype
            )
            kernel_input["weight"] = kernel_input["weight"].view(alt_dtype)
            return kernel_input

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=output_projection_cte_mx_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
        )

    # ============================================================================
    # STATIC_MX Quantization Tests
    # ============================================================================

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, _OPROJ_CTE_UNIT_STATIC_MXFP8_FAST, abbrevs=_ABBREVS)
    def test_output_proj_cte_static_mxfp8_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for STATIC_MX FP8 output projection."""
        n_d = n_head * d_head
        is_negative_test_case = (n_d < 128) or (n_d % 128 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                quantization_type=QuantizationType.STATIC_MX,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.036,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # ROW_MX Quantization Tests
    # ============================================================================

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, _OPROJ_CTE_UNIT_ROW_MXFP8_FAST, abbrevs=_ABBREVS)
    def test_output_proj_cte_row_mxfp8_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for ROW_MX FP8 output projection."""
        n_d = n_head * d_head
        is_negative_test_case = (n_d < 128) or (n_d % 128 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                quantization_type=QuantizationType.ROW_MX,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
            ),
            rtol=0.036,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # ROW FP8 Quantization Tests (TRN2)
    # ============================================================================

    @pytest_parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, _OPROJ_CTE_UNIT_ROW_FP8_FAST, abbrevs=_ABBREVS)
    def test_output_proj_cte_row_fp8_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for ROW FP8 output projection (TRN2). Input is [B, S, N, D]."""
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                quantization_type=QuantizationType.ROW,
                attention_layout=OProjAttentionLayout.BSNd,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.036,
            atol=1e-5,
            is_negative_test=d_head > _MAX_D_ROW,
        )

    # ============================================================================
    # Opt-in FP8 E4M3 canary (dtype_mode).
    #
    # Parametrized over STATIC and ROW × [NON_OCP, OCP, AUTO]. Both reach
    # build_quantization_config and allocate the internal quantized attention
    # SBUF with the resolved FP8 dtype. OCP is TRN3-gated via pytest.skip;
    # NON_OCP and AUTO run on any platform.
    # ============================================================================
    _OUTPUT_PROJ_CTE_BY_DTYPE_MODE_CONFIG: OutputProjCteDtypeModeConfig = {
        "batch": 1,
        "seqlen": 512,
        "hidden": 3072,
        "n_head": 8,
        "d_head": 128,
        "test_bias": True,
    }

    @pytest.mark.fast
    @pytest.mark.parametrize("dtype_mode", [DtypeMode.NON_OCP, DtypeMode.OCP, DtypeMode.AUTO])
    @pytest.mark.parametrize("quantization_type", [QuantizationType.STATIC, QuantizationType.ROW])
    def test_output_proj_cte_by_dtype_mode(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        quantization_type: QuantizationType,
        dtype_mode: DtypeMode,
    ):
        """Smoke-test each DtypeMode through output_projection_cte STATIC/ROW.

        NON_OCP → ``nl.float8_e4m3`` (240), any platform.
        OCP     → ``nl.float8_e4m3fn`` (448), TRN3 only.
        AUTO    → ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3`` elsewhere.
        """
        if dtype_mode == DtypeMode.OCP and not platform_target.is_trn3():
            pytest.skip("dtype_mode=DtypeMode.OCP only exercises the OCP path on TRN3")
        dtype = nl.bfloat16
        cfg = self._OUTPUT_PROJ_CTE_BY_DTYPE_MODE_CONFIG
        batch, seqlen, hidden = cfg["batch"], cfg["seqlen"], cfg["hidden"]

        # Pre-resolve AUTO so the generated weight dtype (kernel-side) and torch-ref
        # clip agree with the platform the kernel actually traces on.
        resolved_dtype_mode = resolve_dtype_mode_for_torch_ref(dtype_mode, platform_target)

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                quantization_type=quantization_type,
                dtype_mode=resolved_dtype_mode,
                **cfg,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        @functools.wraps(output_projection_cte_torch_ref)
        def _torch_ref_with_resolved_dtype_mode(**kwargs):
            kwargs["dtype_mode"] = resolved_dtype_mode
            return output_projection_cte_torch_ref(**kwargs)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(_torch_ref_with_resolved_dtype_mode),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=5e-2,
            atol=1e-3,
        )


@pytest_marks(["output_projection", "cte", "model", "mx"])
@final
class TestOutputProjCteModel:
    """Model-driven tests for Output Projection CTE kernel, organized by tier."""

    _OPROJ_MODEL_PARAMS = f"{OUTPUT_PROJ_CTE_UNIT_PARAMS}, quant_type"

    _OPTIMAL_PARAMS, _OPTIMAL_IDS = (
        prepare_model_parametrize({ModelTestType.OPTIMAL: output_proj_cte_model_configs.get(ModelTestType.OPTIMAL, [])})
        if output_proj_cte_model_configs
        else ([], [])
    )

    def _run_model_test(self, **kwargs):
        """Common test logic for model tiers."""
        test_manager = kwargs["test_manager"]
        platform_target = kwargs["platform_target"]
        batch = kwargs["batch"]
        seqlen = kwargs["seqlen"]
        hidden = kwargs["hidden"]
        n_head = kwargs["n_head"]
        d_head = kwargs["d_head"]
        test_bias = kwargs["test_bias"]
        quantization_type = kwargs["quant_type"]

        if (
            quantization_type in (QuantizationType.MX, QuantizationType.STATIC_MX, QuantizationType.ROW_MX)
            and not platform_target.is_trn3()
        ):
            pytest.skip("MX/STATIC_MX/ROW_MX only supported on TRN3")

        n_d = n_head * d_head
        is_negative_test_case = quantization_type in (
            QuantizationType.MX,
            QuantizationType.STATIC_MX,
            QuantizationType.ROW_MX,
        ) and (n_d < 128 or n_d % 128 != 0)
        dtype = nl.bfloat16

        if quantization_type == QuantizationType.MX:
            rtol = 5e-2
            torch_ref = torch_ref_wrapper(output_projection_cte_mx_torch_ref)

            def input_generator(test_config):
                return generate_output_proj_cte_mx_inputs(
                    batch,
                    seqlen,
                    hidden,
                    n_head,
                    d_head,
                    test_bias=test_bias,
                    weight_dtype=_MODEL_MX_WEIGHT_DTYPE,
                )
        else:
            rtol = 2e-2 if quantization_type == QuantizationType.NONE else 0.036
            torch_ref = torch_ref_wrapper(output_projection_cte_torch_ref)

            def input_generator(test_config):
                return generate_output_proj_cte_inputs(
                    batch=batch,
                    seqlen=seqlen,
                    hidden=hidden,
                    n_head=n_head,
                    d_head=d_head,
                    test_bias=test_bias,
                    quantization_type=quantization_type,
                )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=rtol,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.optimal
    @pytest.mark.parametrize(_OPROJ_MODEL_PARAMS, _OPTIMAL_PARAMS, ids=_OPTIMAL_IDS)
    def test_optimal(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        quant_type: QuantizationType,
    ):
        """OPTIMAL: Performance-optimized model configs."""
        kwargs = {k: v for k, v in locals().items() if k != "self"}
        self._run_model_test(**kwargs)
