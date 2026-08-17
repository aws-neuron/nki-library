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

"""Shared utilities for MLP MXFP8 checkpoint tests (forward and backward)."""

import itertools
from dataclasses import dataclass

import neuron_dtypes as ndtype
import numpy as np
import numpy.typing as npt
from neuronxcc.nki._private.test import mx_util  # ty: ignore[unresolved-import]
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import _get_mx_max_exp
from nkilib_src.nkilib.experimental.mlp_mxfp8.common_utils import DGT_MIN_K

from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.integration.nkilib.experimental.matmul_mxfp8.utils import (
    resize_scales_compact_to_oversized_2d,
)
from test.integration.nkilib.experimental.quantize_mxfp8.test_quantize_mxfp8_utils import (
    Q_TILE_K,
    generate_golden_packed_scales,
)
from test.utils.rng import NKITestsRNG

_rng = NKITestsRNG()

# ============================================================================
# Correctness thresholds — for MXFP8 kernel vs FP32 golden comparison.
# ============================================================================

MXFP8_ATOL_GOLDEN_ABSMAX_PERCENTAGE_TOLERANCE = 0.5
MXFP8_COSINE_SIMILARITY_THRESHOLD = 0.99
MXFP8_NORMALIZED_EUCLIDEAN_THRESHOLD = 0.70
NUMERICAL_STABILITY_EPSILON = 1e-12
DEFAULT_RTOL = 1e-3
DEFAULT_SEED = 42
BWD_GOLDEN_SEED = 123
NUM_CHECKPOINT_FLAGS = 4
ALL_CHECKPOINTS_ENABLED = (True, True, True, True)

LNC = 2

# ============================================================================
# Model configurations
# ============================================================================


@dataclass(frozen=True)
class ModelConfig:
    """Model shape configuration for MLP checkpoint tests."""

    name: str
    seq_len: int
    hidden_size: int
    intermediate_size: int  # per-shard (already divided by TP)
    lnc: int = 2

    def __repr__(self) -> str:
        return f"{self.name}"


# Qwen3 8B: H=4096, base I=12288; Qwen3 32B: H=5120, base I=25600.
# All dimensions must be divisible by 512 (DGT hardware requirement).

MODEL_CONFIGS = [
    ModelConfig("qwen3_8b_tp1", 4096, 4096, 12288),
    ModelConfig("qwen3_8b_tp2", 4096, 4096, 6144),
    ModelConfig("qwen3_8b_tp4", 4096, 4096, 3072),
    ModelConfig("qwen3_8b_tp8", 4096, 4096, 1536),
    ModelConfig("qwen3_32b_tp4", 4096, 5120, 6400),
    ModelConfig("qwen3_32b_tp8", 4096, 5120, 3200),
    ModelConfig("remainder_128", 2048 + 128, 1024 + 128, 1024 + 128),
    ModelConfig("remainder_256", 2048 + 256, 1024 + 256, 1024 + 256),
    ModelConfig("remainder_384", 2048 + 384, 1024 + 384, 1024 + 384),
]

# Randomly generated configs with S, H, I divisible by DGT_MIN_K
NUM_RANDOM_SHAPES = 10
_shape_rng = np.random.default_rng(seed=DEFAULT_SEED)
RANDOM_MODEL_CONFIGS = [
    ModelConfig(
        f"random_{i}",
        int(_shape_rng.integers(1, 64) * DGT_MIN_K),
        int(_shape_rng.integers(1, 64) * DGT_MIN_K),
        int(_shape_rng.integers(1, 64) * DGT_MIN_K),
    )
    for i in range(NUM_RANDOM_SHAPES)
]

PERF_BENCH_MODEL_CONFIG = MODEL_CONFIGS[2]  # qwen3_8b_tp4: S=4096, H=4096, I=3072

# Dense-model shapes for the backward perf bench. qwen3_32b_tp1 (I=FFN/TP1=25600)
# is defined here rather than in MODEL_CONFIGS so the fwd/bwd sweeps don't pick up
# its large shape.
PERF_BENCH_MODEL_CONFIGS = [
    MODEL_CONFIGS[2],  # qwen3_8b_tp4:  S=4096, H=4096, I=3072
    MODEL_CONFIGS[4],  # qwen3_32b_tp4: S=4096, H=5120, I=6400
    MODEL_CONFIGS[0],  # qwen3_8b_tp1:  S=4096, H=4096, I=12288
    ModelConfig("qwen3_32b_tp1", 4096, 5120, 25600),
]

# ============================================================================
# Checkpoint flag combinations
# ============================================================================

ALL_CHECKPOINT_COMBOS = list(itertools.product([True, False], repeat=NUM_CHECKPOINT_FLAGS))
ALL_FALSE_CHECKPOINT = (False, False, False, False)

# ============================================================================
# Input mode helpers
# ============================================================================

_INPUT_MODES = ["raw", "preswizzled", "prequantized"]

# ============================================================================
# Correctness utilities
# ============================================================================


def cosine_sim(a: npt.NDArray, b: npt.NDArray) -> float:
    """Cosine similarity between two flattened arrays."""
    a_flat = a.flatten().astype(np.float32)
    b_flat = b.flatten().astype(np.float32)
    dot = np.dot(a_flat, b_flat)
    return dot / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + NUMERICAL_STABILITY_EPSILON)


def check_correctness(kernel_result: npt.NDArray, golden_result: npt.NDArray, rtol: float = DEFAULT_RTOL) -> tuple:
    """Compare kernel output to golden using matmul_mxfp8 criteria.

    Returns (passed, metrics_dict).
    """
    k_flat = kernel_result.flatten().astype(np.float32)
    g_flat = golden_result.flatten().astype(np.float32)

    cos = cosine_sim(k_flat, g_flat)
    norm_sum = np.linalg.norm(k_flat) + np.linalg.norm(g_flat) + NUMERICAL_STABILITY_EPSILON
    euclid = np.linalg.norm(k_flat - g_flat) / norm_sum
    is_close = np.allclose(
        k_flat,
        g_flat,
        atol=np.abs(g_flat).max() * MXFP8_ATOL_GOLDEN_ABSMAX_PERCENTAGE_TOLERANCE,
        rtol=rtol,
    )
    cos_ok = cos >= MXFP8_COSINE_SIMILARITY_THRESHOLD
    euclid_ok = euclid <= MXFP8_NORMALIZED_EUCLIDEAN_THRESHOLD
    passed = is_close and cos_ok and euclid_ok

    abs_diff = np.abs(k_flat - g_flat)
    max_abs_idx = int(np.argmax(abs_diff))
    max_abs_loc = np.unravel_index(max_abs_idx, kernel_result.shape)

    return passed, {
        "cosine_similarity": float(cos),
        "normalized_euclidean_distance": float(euclid),
        "all_close": is_close,
        "max_abs_diff": float(abs_diff[max_abs_idx]),
        "max_abs_loc": max_abs_loc,
        "max_abs_kernel": float(k_flat[max_abs_idx]),
        "max_abs_golden": float(g_flat[max_abs_idx]),
        "atol_used": float(np.abs(g_flat).max() * MXFP8_ATOL_GOLDEN_ABSMAX_PERCENTAGE_TOLERANCE),
    }


# ============================================================================
# PyTorch-free golden reference (BF16 SwiGLU MLP)
# ============================================================================


def silu_np(x: npt.NDArray) -> npt.NDArray:
    """SiLU activation in float32, numerically stable."""
    x32 = x.astype(np.float32)
    sig = np.where(x32 >= 0, 1.0 / (1.0 + np.exp(-x32)), np.exp(x32) / (1.0 + np.exp(x32)))
    return x32 * sig


def golden_mlp_fwd(hidden_np: npt.NDArray, gate_up_np: npt.NDArray, down_np: npt.NDArray) -> tuple:
    """BF16 SwiGLU MLP forward golden reference.

    Returns (output, gate_pre, gate_act, up_proj, intermediate).
    All computation in float32 for golden accuracy.
    """
    H = hidden_np.astype(np.float32)
    I = gate_up_np.shape[0] // 2

    W_gate = gate_up_np[:I, :].astype(np.float32)
    W_up = gate_up_np[I:, :].astype(np.float32)
    W_down = down_np.astype(np.float32)

    gate_pre = H @ W_gate.T
    gate_act = silu_np(gate_pre)
    up_proj = H @ W_up.T
    intermediate = gate_act * up_proj
    output = intermediate @ W_down.T

    return output, gate_pre, gate_act, up_proj, intermediate


# ============================================================================
# Input generation
# ============================================================================


def swizzle_for_mlp(tensor_np: npt.NDArray) -> npt.NDArray:
    """Swizzle a [F, K] BF16 tensor to [K/4, F*4] layout for pre-swizzled input.

    The MLP kernel expects inputs in [F, K] layout (is_f_by_k=True).
    Swizzling converts [K, F] -> [K/4, F*4], so we first transpose to [K, F].
    """
    return matmul_utils.swizzle_tensor(tensor_np.T)


def prequantize_for_mlp(tensor_np: npt.NDArray, use_scale_packing: bool = False) -> tuple:
    """Pre-quantize a [F, K] BF16 tensor to MXFP8 x4 format.

    Steps:
      1. Transpose to [K, F] and swizzle to [K/4, F*4]
      2. Quantize via mx_util.quantize_mx_golden -> (x4_data, compact_scales)
      3. Format scales: packed if use_scale_packing, else oversized 2D

    Returns:
        (data, scales) tuple ready to pass as kernel input.
    """
    F, K = tensor_np.shape
    swizzled = matmul_utils.swizzle_tensor(tensor_np.T)
    fp8_x4_dtype = ndtype.float8_e4m3fn_x4
    data, compact_scales = mx_util.quantize_mx_golden(swizzled, fp8_x4_dtype, custom_mx_max_exp=_get_mx_max_exp)
    if use_scale_packing:
        scales = generate_golden_packed_scales(compact_scales, K, F, Q_TILE_K)
    else:
        scales = resize_scales_compact_to_oversized_2d(compact_scales)
    return data, scales


def generate_inputs(S: int, H: int, I: int, seed: int = DEFAULT_SEED) -> tuple:
    """Generate deterministic BF16 inputs for given MLP dimensions.

    Uses kaiming_normal initialization (same as matmul_mxfp8 tests) to produce
    well-scaled values that avoid overflow in MXFP8 matmul accumulations.
    """
    import torch

    _rng.reset()
    try:
        import ml_dtypes

        bf16 = ml_dtypes.bfloat16
    except ImportError:
        bf16 = np.float16

    hidden = _rng.kaiming_normal_(torch.empty(S, H)).numpy().astype(bf16)
    gate_up = _rng.kaiming_normal_(torch.empty(2 * I, H)).numpy().astype(bf16)
    down = _rng.kaiming_normal_(torch.empty(H, I)).numpy().astype(bf16)
    return hidden, gate_up, down


def apply_input_mode(
    kernel_input: dict, key: str, mode: str, use_scale_packing: bool, scales_key: str, swizzled_key: str
) -> None:
    """Replace a kernel_input tensor with pre-swizzled or pre-quantized version in-place."""
    if mode == "raw":
        return
    bf16_tensor = kernel_input[key]
    if mode == "preswizzled":
        kernel_input[key] = swizzle_for_mlp(bf16_tensor)
        kernel_input[swizzled_key] = True
    elif mode == "prequantized":
        data, scales = prequantize_for_mlp(bf16_tensor, use_scale_packing)
        kernel_input[key] = data
        kernel_input[scales_key] = scales
