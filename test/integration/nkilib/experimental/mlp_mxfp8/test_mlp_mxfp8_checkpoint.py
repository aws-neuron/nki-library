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

"""Integration tests for MLP MXFP8 forward and backward kernels with activation checkpointing."""

import itertools
from dataclasses import dataclass
from typing import Any, final

import neuron_dtypes as ndtype
import numpy as np
import numpy.typing as npt
import pytest
from neuronxcc.nki._private.test import mx_util
from typing_extensions import override

from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import _get_mx_max_exp
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8.config import (
    get_config_for_shape as get_bwd_config,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8.mlp_bwd_mxfp8_kernel import (
    mlp_backward_mxfp8_nki,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8.config import (
    get_config_for_shape as get_fwd_config,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8.mlp_fwd_mxfp8_kernel import (
    mlp_forward_mxfp8_nki,
)
from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.integration.nkilib.experimental.matmul_mxfp8.utils import (
    resize_scales_compact_to_oversized_2d,
)
from test.integration.nkilib.experimental.quantize_mxfp8.test_quantize_mxfp8_utils import (
    Q_TILE_K,
    generate_golden_packed_scales,
)
from test.utils import common_dataclasses, coverage_parametrized_tests, test_orchestrator
from test.utils.pseudo_rng import NKITestsPseudoRNG
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework

_rng = NKITestsPseudoRNG(seed=42)
_bwd_rng = NKITestsPseudoRNG(seed=123)

# ============================================================================
# Correctness thresholds — for MXFP8 kernel vs FP32 golden comparison.
# ============================================================================

# MXFP8 quantization introduces error compared to FP32, especially after
# chained matmuls. These thresholds account for quantization noise.
MXFP8_ATOL_GOLDEN_ABSMAX_PERCENTAGE_TOLERANCE = 0.5
MXFP8_COSINE_SIMILARITY_THRESHOLD = 0.99
MXFP8_NORMALIZED_EUCLIDEAN_THRESHOLD = 0.70
NUMERICAL_STABILITY_EPSILON = 1e-12
DEFAULT_RTOL = 1e-3
DEFAULT_SEED = 42
BWD_GOLDEN_SEED = 123
NUM_CHECKPOINT_FLAGS = 4
ALL_CHECKPOINTS_ENABLED = (True, True, True, True)

# Qwen3 model configs: (name, seq_len, hidden_size, base_intermediate_size)
# I is divided by TP degree to get per-shard intermediate size


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

PERF_BENCH_MODEL_CONFIG = MODEL_CONFIGS[2]  # qwen3_8b_tp4: S=4096, H=4096, I=3072

LNC = 2

# ============================================================================
# Correctness utilities — reused from matmul_mxfp8 pattern
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
# Custom comparators for MXFP8 validation
# ============================================================================


def _fwd_comparator(golden_dict, output_tensors):
    """custom_comparator for forward tests."""
    golden = golden_dict["output"]
    dtype = golden.dtype

    class _FwdValidator(common_dataclasses.CustomValidator):
        @override
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            """Validate forward output against golden reference."""
            reshaped = inference_output.view(dtype=dtype).reshape(golden.shape)
            passed, metrics = check_correctness(reshaped, golden.astype(dtype))
            if not passed:
                self._print_with_log("[fwd output] Validation FAILED")
                self._print_with_log(f"  metrics: {metrics}")
                self._print_with_log(f"  kernel[0,:5]: {reshaped.flatten()[:5]}")
                self._print_with_log(f"  golden[0,:5]: {golden.flatten()[:5]}")
            return passed

    return {
        "output": common_dataclasses.CustomValidatorWithOutputTensorData(
            validator=_FwdValidator,
            output_ndarray=np.ndarray(golden.shape, dtype=dtype),
        )
    }


def _bwd_comparator(golden_dict, output_tensors):
    """custom_comparator for backward tests."""
    result = {}
    for name, golden in golden_dict.items():
        dtype = golden.dtype

        class _BwdValidator(common_dataclasses.CustomValidator):
            _golden = golden
            _label = name

            @override
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                """Validate backward output against golden reference."""
                reshaped = inference_output.view(dtype=self._golden.dtype).reshape(self._golden.shape)
                passed, metrics = check_correctness(reshaped, self._golden.astype(self._golden.dtype))
                if not passed:
                    self._print_with_log(f"[{self._label}] Validation FAILED")
                    self._print_with_log(f"  metrics: {metrics}")
                    self._print_with_log(f"  kernel[0,:5]: {reshaped.flatten()[:5]}")
                    self._print_with_log(f"  golden[0,:5]: {self._golden.flatten()[:5]}")
                return passed

        result[name] = common_dataclasses.CustomValidatorWithOutputTensorData(
            validator=_BwdValidator,
            output_ndarray=np.ndarray(golden.shape, dtype=dtype),
        )
    return result


# ============================================================================
# Torch ref — signature must match kernel; golden via custom_comparator
# ============================================================================


def _make_fwd_torch_ref(original_hidden, original_gate_up, original_down):
    """Create a forward torch_ref that always computes real golden.

    Captures original BF16 tensors via closure so golden computation works
    even after apply_input_mode mutates kernel_input in-place.
    """

    def torch_ref(
        hidden,
        gate_up_weights,
        down_weights,
        intermediate_hbm,
        run_with_lnc2=True,
        gate_up_tiles_m=8,
        gate_up_tiles_n=1,
        gate_up_tiles_k=8,
        down_tiles_m=8,
        down_tiles_n=1,
        down_tiles_k=8,
        fp8_x4_dtype=None,
        save_gate_pre=None,
        save_gate_act=None,
        save_up=None,
        save_hidden=None,
        dtype=None,
        spill_reload=False,
        use_scale_packing=False,
        hidden_scales=None,
        gate_up_scales=None,
        down_scales=None,
        hidden_is_swizzled=False,
        gate_up_is_swizzled=False,
        down_is_swizzled=False,
    ):
        """Compute forward golden output from original BF16 tensors."""
        if hidden_scales is not None or hidden_is_swizzled:
            h, g, d = original_hidden, original_gate_up, original_down
        else:
            h, g, d = hidden, gate_up_weights, down_weights
        golden_out, *_ = golden_mlp_fwd(h, g, d)
        return {"output": golden_out.astype(h.dtype)}

    return torch_ref


def _make_bwd_torch_ref(original_hidden, original_gate_up, original_down):
    """Create a backward torch_ref that always computes real golden.

    Captures original BF16 tensors via closure so golden computation works
    even after apply_input_mode mutates kernel_input in-place.
    """

    def torch_ref(
        output_hidden_states_grad,
        hidden_states,
        gate_proj_weight_T,
        up_proj_weight_T,
        down_proj_weight_T,
        gate_up_weights,
        d_gate_scratch,
        d_up_scratch,
        hidden_states_T,
        output_grad_T,
        hidden_T,
        silu_up_mul_gate_grad_T_scratch,
        gate_pre_scratch,
        gate_act_scratch,
        up_scratch,
        hidden_scratch,
        gate_pre=None,
        gate_act=None,
        up=None,
        hidden=None,
        run_with_lnc2=True,
        phase1_tiles_m=8,
        phase1_tiles_n=1,
        phase1_tiles_k=8,
        phase2_tiles_m=8,
        phase2_tiles_n=1,
        phase2_tiles_k=8,
        phase3_tiles_m=4,
        phase3_tiles_n=1,
        phase3_tiles_k=8,
        phase4_tiles_m=4,
        phase4_tiles_n=1,
        phase4_tiles_k=8,
        recompute_tiles_m=8,
        recompute_tiles_n=1,
        recompute_tiles_k=8,
        fp8_x4_dtype=None,
        spill_reload=False,
        use_scale_packing=False,
        output_grad_T_scales=None,
        output_grad_T_is_swizzled=False,
        hidden_T_scales=None,
        hidden_T_is_swizzled=False,
        output_grad_scales=None,
        output_grad_is_swizzled=False,
        down_weight_scales=None,
        down_weight_is_swizzled=False,
        gate_weight_scales=None,
        gate_weight_is_swizzled=False,
        up_weight_scales=None,
        up_weight_is_swizzled=False,
        hidden_states_T_scales=None,
        hidden_states_T_is_swizzled=False,
        hidden_states_scales=None,
        hidden_states_is_swizzled=False,
        gate_up_weights_scales=None,
        gate_up_weights_is_swizzled=False,
    ):
        """Compute backward golden gradients from original BF16 tensors."""
        S, H = original_hidden.shape
        I = original_gate_up.shape[0] // 2
        bf16 = original_hidden.dtype

        _, golden_gp, golden_ga, golden_up, golden_hidden = golden_mlp_fwd(
            original_hidden, original_gate_up, original_down
        )
        _, golden_hs_grad, golden_gate_wgrad, golden_up_wgrad, golden_dw_wgrad = compute_bwd_golden(
            S,
            H,
            I,
            original_hidden,
            original_gate_up,
            original_down,
            golden_gp,
            golden_ga,
            golden_up,
            golden_hidden,
        )
        return {
            "hidden_states_grad": golden_hs_grad.astype(bf16),
            "gate_up_weight_grad": np.concatenate([golden_gate_wgrad, golden_up_wgrad], axis=0).astype(bf16),
            "down_proj_weight_grad": golden_dw_wgrad.astype(bf16),
        }

    return torch_ref


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

    try:
        import ml_dtypes

        bf16 = ml_dtypes.bfloat16
    except ImportError:
        bf16 = np.float16

    hidden = _rng.kaiming_normal_(torch.empty(S, H)).numpy().astype(bf16)
    gate_up = _rng.kaiming_normal_(torch.empty(2 * I, H)).numpy().astype(bf16)
    down = _rng.kaiming_normal_(torch.empty(H, I)).numpy().astype(bf16)
    return hidden, gate_up, down


# ============================================================================
# Checkpoint flag combinations
# ============================================================================

# All 16 combinations of (save_gate_pre, save_gate_act, save_up, save_hidden)
ALL_CHECKPOINT_COMBOS = list(itertools.product([True, False], repeat=NUM_CHECKPOINT_FLAGS))


# ============================================================================
# Forward test — validates output + checkpointed intermediates
# ============================================================================


def compute_bwd_golden(
    S: int, H: int, I: int, hidden_np, gate_up_np, down_np, golden_gp, golden_ga, golden_up, golden_hidden
) -> tuple:
    """Compute backward golden reference tensors.

    Returns (output_grad, golden_hs_grad, golden_gate_wgrad, golden_up_wgrad, golden_dw_wgrad).
    """
    import torch

    bf16 = hidden_np.dtype
    output_grad = _bwd_rng.kaiming_normal_(torch.empty(S, H)).numpy().astype(bf16)
    og32 = output_grad.astype(np.float32)
    W_gate = gate_up_np[:I, :].astype(np.float32)
    W_up = gate_up_np[I:, :].astype(np.float32)
    W_down = down_np.astype(np.float32)

    d_intermediate = og32 @ W_down
    sig = 1.0 / (1.0 + np.exp(-golden_gp.astype(np.float32)))
    silu_deriv = sig * (1.0 + golden_gp.astype(np.float32) * (1.0 - sig))
    d_gate = d_intermediate * silu_deriv * golden_up.astype(np.float32)
    d_up = d_intermediate * golden_ga.astype(np.float32)

    d_gate_up = np.concatenate([d_gate, d_up], axis=1)
    W_gate_up = np.concatenate([W_gate, W_up], axis=0)
    golden_hs_grad = d_gate_up @ W_gate_up
    golden_gate_wgrad = d_gate.T @ hidden_np.astype(np.float32)
    golden_up_wgrad = d_up.T @ hidden_np.astype(np.float32)
    golden_dw_wgrad = og32.T @ golden_hidden.astype(np.float32)

    return output_grad, golden_hs_grad, golden_gate_wgrad, golden_up_wgrad, golden_dw_wgrad


def build_bwd_kernel_input(
    S: int,
    H: int,
    I: int,
    hidden_np,
    gate_up_np,
    down_np,
    golden_hidden,
    output_grad,
    spill_reload: bool,
    use_scale_packing: bool,
    checkpoint_tensors: dict | None = None,
) -> dict:
    """Build the kernel_input dict for backward tests.

    checkpoint_tensors: optional dict with keys from {gate_pre, gate_act, up, hidden}.
    """
    bf16 = hidden_np.dtype
    gate_T = np.ascontiguousarray(gate_up_np[:I, :].astype(bf16).T)
    up_T = np.ascontiguousarray(gate_up_np[I:, :].astype(bf16).T)
    down_T = np.ascontiguousarray(down_np.astype(bf16).T)
    hs_T = np.ascontiguousarray(hidden_np.astype(bf16).T)
    og_T = np.ascontiguousarray(output_grad.astype(bf16).T)
    hidden_act_T = np.ascontiguousarray(golden_hidden.astype(bf16).T)

    bwd_cfg = get_bwd_config(S, H, I, LNC)

    kernel_input = {
        "output_hidden_states_grad": output_grad,
        "hidden_states": hidden_np,
        "gate_proj_weight_T": gate_T,
        "up_proj_weight_T": up_T,
        "down_proj_weight_T": down_T,
        "gate_up_weights": gate_up_np,
        "d_gate_scratch": np.zeros((S, I), dtype=bf16),
        "d_up_scratch": np.zeros((S, I), dtype=bf16),
        "hidden_states_T": hs_T,
        "output_grad_T": og_T,
        "hidden_T": hidden_act_T,
        "silu_up_mul_gate_grad_T_scratch": np.zeros((2 * I, S), dtype=bf16),
        "gate_pre_scratch": np.zeros((S, I), dtype=bf16),
        "gate_act_scratch": np.zeros((S, I), dtype=bf16),
        "up_scratch": np.zeros((S, I), dtype=bf16),
        "hidden_scratch": np.zeros((S, I), dtype=bf16),
        "phase1_tiles_m": bwd_cfg.phase1_down_proj_mm_grad.TILES_IN_BLOCK_M,
        "phase1_tiles_n": bwd_cfg.phase1_down_proj_mm_grad.TILES_IN_BLOCK_N,
        "phase1_tiles_k": bwd_cfg.phase1_down_proj_mm_grad.TILES_IN_BLOCK_K,
        "phase2_tiles_m": bwd_cfg.phase2_hidden_states_grad.TILES_IN_BLOCK_M,
        "phase2_tiles_n": bwd_cfg.phase2_hidden_states_grad.TILES_IN_BLOCK_N,
        "phase2_tiles_k": bwd_cfg.phase2_hidden_states_grad.TILES_IN_BLOCK_K,
        "phase3_tiles_m": bwd_cfg.phase3_gate_up_weight_grad.TILES_IN_BLOCK_M,
        "phase3_tiles_n": bwd_cfg.phase3_gate_up_weight_grad.TILES_IN_BLOCK_N,
        "phase3_tiles_k": bwd_cfg.phase3_gate_up_weight_grad.TILES_IN_BLOCK_K,
        "phase4_tiles_m": bwd_cfg.phase4_down_weight_grad.TILES_IN_BLOCK_M,
        "phase4_tiles_n": bwd_cfg.phase4_down_weight_grad.TILES_IN_BLOCK_N,
        "phase4_tiles_k": bwd_cfg.phase4_down_weight_grad.TILES_IN_BLOCK_K,
        "recompute_tiles_m": bwd_cfg.recompute_gate_up.TILES_IN_BLOCK_M,
        "recompute_tiles_n": bwd_cfg.recompute_gate_up.TILES_IN_BLOCK_N,
        "recompute_tiles_k": bwd_cfg.recompute_gate_up.TILES_IN_BLOCK_K,
        "spill_reload": spill_reload,
        "use_scale_packing": use_scale_packing,
    }

    if checkpoint_tensors:
        for key, val in checkpoint_tensors.items():
            kernel_input[key] = val

    return kernel_input


# ============================================================================
# Input mode helpers
# ============================================================================

_INPUT_MODES = ["raw", "preswizzled", "prequantized"]


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


# ============================================================================
# Coverage parametrize abbreviations
# ============================================================================

ALL_FALSE_CHECKPOINT = (False, False, False, False)

_FWD_ABBREVS = {
    "model_cfg": "cfg",
    "input_mode": "in",
    "checkpoint_combo": "ckpt",
    "spill_reload": "sr",
    "use_scale_packing": "sp",
}

_BWD_ABBREVS = {
    "model_cfg": "cfg",
    "checkpoint_combo": "ckpt",
    "phase1_mode": "p1",
    "phase2_mode": "p2",
    "phase3_mode": "p3",
    "phase4_mode": "p4",
    "recompute_mode": "rc",
    "spill_reload": "sr",
    "use_scale_packing": "sp",
}


# ============================================================================
# Coverage parametrize filters
# ============================================================================


def _fwd_filter(
    model_cfg=None,
    input_mode=None,
    checkpoint_combo=None,
    spill_reload=None,
    use_scale_packing=None,
) -> coverage_parametrized_tests.FilterResult:
    """Filter for forward sweep combinations."""
    return coverage_parametrized_tests.FilterResult.VALID


def _bwd_filter(
    model_cfg=None,
    checkpoint_combo=None,
    phase1_mode=None,
    phase2_mode=None,
    phase3_mode=None,
    phase4_mode=None,
    recompute_mode=None,
    spill_reload=None,
    use_scale_packing=None,
) -> coverage_parametrized_tests.FilterResult:
    """Filter for backward sweep combinations."""
    return coverage_parametrized_tests.FilterResult.VALID


@pytest_test_metadata(name="MLP MXFP8 Checkpoint")
@pytest_marks(["mlp_mxfp8_checkpoint", "mx"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMlpMxfp8Checkpoint:
    """Tests for MLP MXFP8 forward and backward kernels with activation checkpointing."""

    # TODO: Add more comprehensive sweep once all functional features are in

    def run_fwd_test(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
        S: int,
        H: int,
        I: int,
        spill_reload: bool,
        use_scale_packing: bool,
        input_mode: str = "raw",
        checkpoint_combo: tuple = (False, False, False, False),
    ) -> None:
        """Run a forward MLP test end-to-end.

        Args:
            input_mode: "raw" (BF16), "preswizzled", or "prequantized".
            checkpoint_combo: (save_gate_pre, save_gate_act, save_up, save_hidden).
        """
        hidden_np, gate_up_np, down_np = generate_inputs(S, H, I)
        bf16 = hidden_np.dtype

        fwd_cfg = get_fwd_config(S, H, I, LNC)
        kernel_input = {
            "hidden": hidden_np,
            "gate_up_weights": gate_up_np,
            "down_weights": down_np,
            "intermediate_hbm": np.zeros((S, I), dtype=bf16),
            "gate_up_tiles_m": fwd_cfg.gate_up.TILES_IN_BLOCK_M,
            "gate_up_tiles_n": fwd_cfg.gate_up.TILES_IN_BLOCK_N,
            "gate_up_tiles_k": fwd_cfg.gate_up.TILES_IN_BLOCK_K,
            "down_tiles_m": fwd_cfg.down.TILES_IN_BLOCK_M,
            "down_tiles_n": fwd_cfg.down.TILES_IN_BLOCK_N,
            "down_tiles_k": fwd_cfg.down.TILES_IN_BLOCK_K,
            "spill_reload": spill_reload,
            "use_scale_packing": use_scale_packing,
        }

        # Create torch_ref before apply_input_mode mutates kernel_input tensors.
        fwd_torch_ref = _make_fwd_torch_ref(hidden_np, gate_up_np, down_np)

        apply_input_mode(kernel_input, "hidden", input_mode, use_scale_packing, "hidden_scales", "hidden_is_swizzled")
        apply_input_mode(
            kernel_input, "gate_up_weights", input_mode, use_scale_packing, "gate_up_scales", "gate_up_is_swizzled"
        )
        apply_input_mode(kernel_input, "down_weights", input_mode, use_scale_packing, "down_scales", "down_is_swizzled")

        # Checkpoint save buffers
        save_gate_pre, save_gate_act, save_up, save_hidden = checkpoint_combo
        if save_gate_pre:
            kernel_input["save_gate_pre"] = np.zeros((S, I), dtype=bf16)
        if save_gate_act:
            kernel_input["save_gate_act"] = np.zeros((S, I), dtype=bf16)
        if save_up:
            kernel_input["save_up"] = np.zeros((S, I), dtype=bf16)
        if save_hidden:
            kernel_input["save_hidden"] = np.zeros((S, I), dtype=bf16)

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=LNC,
            platform_target=platform_target,
            additional_cmd_args=[
                "--internal-backend-options=--enable-mx-alternative-emax --skip-pass=address_rotation_sb"
            ],
        )
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mlp_forward_mxfp8_nki,
            torch_ref=fwd_torch_ref,
            kernel_input_generator=lambda _: kernel_input,
            output_tensor_descriptor=lambda _: {"output": np.zeros((S, H), dtype=bf16)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_fwd_comparator,
        )

    def run_bwd_test(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
        S: int,
        H: int,
        I: int,
        spill_reload: bool,
        use_scale_packing: bool,
        checkpoint_combo: tuple = (False, False, False, False),
        phase1_mode: str = "raw",
        phase2_mode: str = "raw",
        phase3_mode: str = "raw",
        phase4_mode: str = "raw",
        recompute_mode: str = "raw",
    ) -> None:
        """Run a backward MLP test end-to-end with per-phase input mode control.

        Args:
            phase1_mode: Input mode for phase 1 (output_grad, down_weight).
            phase2_mode: Input mode for phase 2 (gate_weight, up_weight).
            phase3_mode: Input mode for phase 3 (hidden_states_T).
            phase4_mode: Input mode for phase 4 (output_grad_T, hidden_T).
            recompute_mode: Input mode for recompute (hidden_states, gate_up_weights).
        """
        hidden_np, gate_up_np, down_np = generate_inputs(S, H, I)
        bf16 = hidden_np.dtype

        _, golden_gp, golden_ga, golden_up, golden_hidden = golden_mlp_fwd(hidden_np, gate_up_np, down_np)
        output_grad, *_ = compute_bwd_golden(
            S, H, I, hidden_np, gate_up_np, down_np, golden_gp, golden_ga, golden_up, golden_hidden
        )

        # Build checkpoint tensors from combo
        save_gate_pre, save_gate_act, save_up, save_hidden = checkpoint_combo
        checkpoint_tensors = {}
        if save_gate_pre:
            checkpoint_tensors["gate_pre"] = golden_gp.astype(bf16)
        if save_gate_act:
            checkpoint_tensors["gate_act"] = golden_ga.astype(bf16)
        if save_up:
            checkpoint_tensors["up"] = golden_up.astype(bf16)
        if save_hidden:
            checkpoint_tensors["hidden"] = golden_hidden.astype(bf16)

        kernel_input = build_bwd_kernel_input(
            S,
            H,
            I,
            hidden_np,
            gate_up_np,
            down_np,
            golden_hidden,
            output_grad,
            spill_reload,
            use_scale_packing,
            checkpoint_tensors,
        )

        # Create torch_ref before apply_input_mode mutates kernel_input tensors.
        bwd_torch_ref = _make_bwd_torch_ref(hidden_np, gate_up_np, down_np)

        # Phase 1: output_grad [S, H], down_weight [I, H]
        apply_input_mode(
            kernel_input,
            "output_hidden_states_grad",
            phase1_mode,
            use_scale_packing,
            "output_grad_scales",
            "output_grad_is_swizzled",
        )
        apply_input_mode(
            kernel_input,
            "down_proj_weight_T",
            phase1_mode,
            use_scale_packing,
            "down_weight_scales",
            "down_weight_is_swizzled",
        )

        # Phase 2: gate_weight [H, I], up_weight [H, I]
        apply_input_mode(
            kernel_input,
            "gate_proj_weight_T",
            phase2_mode,
            use_scale_packing,
            "gate_weight_scales",
            "gate_weight_is_swizzled",
        )
        apply_input_mode(
            kernel_input,
            "up_proj_weight_T",
            phase2_mode,
            use_scale_packing,
            "up_weight_scales",
            "up_weight_is_swizzled",
        )

        # Phase 3: hidden_states_T [H, S]
        apply_input_mode(
            kernel_input,
            "hidden_states_T",
            phase3_mode,
            use_scale_packing,
            "hidden_states_T_scales",
            "hidden_states_T_is_swizzled",
        )

        # Phase 4: output_grad_T [H, S], hidden_T [I, S]
        apply_input_mode(
            kernel_input,
            "output_grad_T",
            phase4_mode,
            use_scale_packing,
            "output_grad_T_scales",
            "output_grad_T_is_swizzled",
        )
        apply_input_mode(
            kernel_input, "hidden_T", phase4_mode, use_scale_packing, "hidden_T_scales", "hidden_T_is_swizzled"
        )

        # Recompute: hidden_states [S, H], gate_up_weights [2I, H]
        apply_input_mode(
            kernel_input,
            "hidden_states",
            recompute_mode,
            use_scale_packing,
            "hidden_states_scales",
            "hidden_states_is_swizzled",
        )
        apply_input_mode(
            kernel_input,
            "gate_up_weights",
            recompute_mode,
            use_scale_packing,
            "gate_up_weights_scales",
            "gate_up_weights_is_swizzled",
        )

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=LNC,
            platform_target=platform_target,
            # Skipping address_rotation_sb is a temporary workaround as we switch to latest nki, remove once KTK-151 resolved
            additional_cmd_args=[
                "--internal-backend-options=--enable-mx-alternative-emax --skip-pass=address_rotation_sb"
            ],
        )
        output_ndarray = {
            "hidden_states_grad": np.zeros((S, H), dtype=bf16),
            "gate_up_weight_grad": np.zeros((2 * I, H), dtype=bf16),
            "down_proj_weight_grad": np.zeros((H, I), dtype=bf16),
        }
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mlp_backward_mxfp8_nki,
            torch_ref=bwd_torch_ref,
            kernel_input_generator=lambda _: kernel_input,
            output_tensor_descriptor=lambda _: output_ndarray.copy(),
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_bwd_comparator,
        )

    @pytest.mark.coverage_parametrize(
        model_cfg=MODEL_CONFIGS,
        input_mode=["raw", "preswizzled", "prequantized"],
        checkpoint_combo=ALL_CHECKPOINT_COMBOS,
        spill_reload=[False, True],
        use_scale_packing=[False, True],
        filter=_fwd_filter,
        coverage="pairs",
        abbrev=_FWD_ABBREVS,
        enable_automatic_boundary_tests=True,
        enable_invalid_combination_tests=True,
    )
    def test_fwd_sweep(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
        model_cfg: ModelConfig,
        input_mode: str,
        checkpoint_combo: tuple,
        spill_reload: bool,
        use_scale_packing: bool,
        is_negative_test_case: bool,
    ) -> None:
        """Forward sweep: model configs × input modes × checkpoint combos × spill/pack."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        if is_negative_test_case:
            pytest.skip("Negative test cases not yet supported for MLP fwd")
        self.run_fwd_test(
            test_manager,
            platform_target,
            model_cfg.seq_len,
            model_cfg.hidden_size,
            model_cfg.intermediate_size,
            spill_reload,
            use_scale_packing,
            input_mode=input_mode,
            checkpoint_combo=checkpoint_combo,
        )

    @pytest.mark.coverage_parametrize(
        model_cfg=MODEL_CONFIGS,
        checkpoint_combo=ALL_CHECKPOINT_COMBOS,
        phase1_mode=_INPUT_MODES,
        phase2_mode=_INPUT_MODES,
        phase3_mode=_INPUT_MODES,
        phase4_mode=_INPUT_MODES,
        recompute_mode=_INPUT_MODES,
        spill_reload=[False, True],
        use_scale_packing=[False, True],
        filter=_bwd_filter,
        coverage="pairs",
        abbrev=_BWD_ABBREVS,
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_bwd_sweep(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
        model_cfg: ModelConfig,
        checkpoint_combo: tuple,
        phase1_mode: str,
        phase2_mode: str,
        phase3_mode: str,
        phase4_mode: str,
        recompute_mode: str,
        spill_reload: bool,
        use_scale_packing: bool,
        is_negative_test_case: bool,
    ) -> None:
        """Backward sweep: model configs × checkpoint combos × per-phase input modes × spill/pack."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        if is_negative_test_case:
            pytest.skip("Negative test cases not yet supported for MLP bwd")
        self.run_bwd_test(
            test_manager,
            platform_target,
            model_cfg.seq_len,
            model_cfg.hidden_size,
            model_cfg.intermediate_size,
            spill_reload,
            use_scale_packing,
            checkpoint_combo,
            phase1_mode=phase1_mode,
            phase2_mode=phase2_mode,
            phase3_mode=phase3_mode,
            phase4_mode=phase4_mode,
            recompute_mode=recompute_mode,
        )

    # ================================================================
    # Perf bench tests — qwen3_8b_tp4, all checkpoints enabled
    # ================================================================

    @pytest.mark.parametrize("input_mode", _INPUT_MODES, ids=_INPUT_MODES)
    @pytest.mark.parametrize("spill_reload", [False, True], ids=["no_spill", "spill"])
    @pytest.mark.parametrize("use_scale_packing", [False, True], ids=["no_pack", "pack"])
    def test_fwd_perf_bench(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
        input_mode: str,
        spill_reload: bool,
        use_scale_packing: bool,
    ) -> None:
        """Forward perf bench: qwen3_8b_tp4 with all checkpoints."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_fwd_test(
            test_manager,
            platform_target,
            PERF_BENCH_MODEL_CONFIG.seq_len,
            PERF_BENCH_MODEL_CONFIG.hidden_size,
            PERF_BENCH_MODEL_CONFIG.intermediate_size,
            spill_reload,
            use_scale_packing,
            input_mode=input_mode,
            checkpoint_combo=ALL_CHECKPOINTS_ENABLED,
        )

    @pytest.mark.fast
    def test_fwd_perf_bench_fast(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
    ) -> None:
        """Forward perf bench: qwen3_8b_tp4 with all checkpoints."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_fwd_test(
            test_manager,
            platform_target,
            PERF_BENCH_MODEL_CONFIG.seq_len,
            PERF_BENCH_MODEL_CONFIG.hidden_size,
            PERF_BENCH_MODEL_CONFIG.intermediate_size,
            True,
            True,
            input_mode="raw",
            checkpoint_combo=ALL_CHECKPOINTS_ENABLED,
        )

    @pytest.mark.parametrize("input_mode", _INPUT_MODES, ids=_INPUT_MODES)
    @pytest.mark.parametrize("spill_reload", [False, True], ids=["no_spill", "spill"])
    @pytest.mark.parametrize("use_scale_packing", [False, True], ids=["no_pack", "pack"])
    def test_bwd_perf_bench(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
        input_mode: str,
        spill_reload: bool,
        use_scale_packing: bool,
    ) -> None:
        """Backward perf bench: qwen3_8b_tp4 with all save options enabled."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_bwd_test(
            test_manager,
            platform_target,
            PERF_BENCH_MODEL_CONFIG.seq_len,
            PERF_BENCH_MODEL_CONFIG.hidden_size,
            PERF_BENCH_MODEL_CONFIG.intermediate_size,
            spill_reload,
            use_scale_packing,
            checkpoint_combo=ALL_CHECKPOINTS_ENABLED,
            phase1_mode=input_mode,
            phase2_mode=input_mode,
            phase3_mode=input_mode,
            phase4_mode=input_mode,
            recompute_mode=input_mode,
        )

    @pytest.mark.fast
    def test_bwd_perf_bench_fast(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
    ) -> None:
        """Backward perf bench: qwen3_8b_tp4 with all save options enabled."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_bwd_test(
            test_manager,
            platform_target,
            PERF_BENCH_MODEL_CONFIG.seq_len,
            PERF_BENCH_MODEL_CONFIG.hidden_size,
            PERF_BENCH_MODEL_CONFIG.intermediate_size,
            True,
            True,
            checkpoint_combo=ALL_CHECKPOINTS_ENABLED,
            phase1_mode="raw",
            phase2_mode="raw",
            phase3_mode="raw",
            phase4_mode="raw",
            recompute_mode="raw",
        )
