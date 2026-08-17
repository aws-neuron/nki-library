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

"""Shared helpers for the MXFP8 MoE forward/backward validated test suites.

The forward (``test_blockwise_mm_forward_mxfp8_validated``) and backward
(``test_blockwise_mm_backward_mxfp8_validated``) suites share the same
three-metric MXFP8 correctness check, the custom-comparator factory, the pytest
param-marking helpers, and the shared abbreviations / param-name string. Those
live here so both suites stay in lockstep. The parts that genuinely differ
(per-phase blocking math — 2 phases fwd vs 4 bwd, the feature/shape grids, the
random sweep, and the test classes themselves) stay in each suite's file.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typing_extensions import override

from test.utils import common_dataclasses

# ============================================================================
# Correctness thresholds (same as MLP MXFP8 checkpoint tests)
# ============================================================================

MXFP8_ATOL_GOLDEN_ABSMAX_PERCENTAGE_TOLERANCE = 0.5
MXFP8_COSINE_SIMILARITY_THRESHOLD = 0.99
MXFP8_NORMALIZED_EUCLIDEAN_THRESHOLD = 0.70
NUMERICAL_STABILITY_EPSILON = 1e-12
DEFAULT_RTOL = 1e-3

# ============================================================================
# Shared pytest param-marking helpers
# ============================================================================

# Test-id abbreviations shared by both suites (parametrized on the same dims).
ABBREVS = {
    "hidden": "hid",
    "tokens": "tok",
    "expert": "exp",
    "block_size": "bs",
    "top_k": "k",
    "intermediate": "int",
}

# Parametrize name string for the shape grid (H, T, E, B, TopK, I_TP).
PARAM_NAMES = "hidden, tokens, expert, block_size, top_k, intermediate"


def clamp_tiles(desired: int, num_tiles: int) -> int:
    """Clamp TILES_IN_BLOCK to not exceed available tiles, minimum 1."""
    return max(1, min(desired, num_tiles))


def build_params(params_list, fast_keys=None, skip_params=None, xfail_params=None):
    """Build pytest params with fast, xfail, and skip marks applied.

    Args:
        params_list: list of param tuples/lists ([H, T, E, B, TOPK, I_TP]).
        fast_keys: set of tuples that should get pytest.mark.fast.
        skip_params: dict {param_tuple: reason} for pytest.mark.skip.
        xfail_params: dict {param_tuple: reason} for pytest.mark.xfail(strict=False).
    """
    fast_keys = fast_keys or set()
    skip_params = skip_params or {}
    xfail_params = xfail_params or {}
    result = []
    for c in params_list:
        key = tuple(c)
        marks = []
        if key in fast_keys:
            marks.append(pytest.mark.fast)
        if key in skip_params:
            marks.append(pytest.mark.skip(reason=skip_params[key]))
        elif key in xfail_params:
            marks.append(pytest.mark.xfail(reason=xfail_params[key], strict=False))
        result.append(pytest.param(*c, marks=marks) if marks else c)
    return result


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
    """Compare kernel output to golden using three-metric MXFP8 criteria.

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
# Custom comparator for MXFP8 MoE validation (fwd and bwd)
# ============================================================================


def make_moe_validated_comparator(output_shapes):
    """Create a custom_comparator closure for the MXFP8 MoE validated tests.

    Each named output is validated against the golden with the three-metric
    MXFP8 criterion. Shared by the forward and backward suites (the per-output
    label already distinguishes them at runtime).

    Args:
        output_shapes: dict mapping output names to (shape, kernel_dtype) tuples.
            kernel_dtype is the actual dtype the kernel writes (bfloat16), which
            may differ from the golden's dtype (float32 from torch_ref_wrapper).
    """

    def comparator(golden_dict, output_tensors):
        result = {}
        for name, golden in golden_dict.items():
            shape, kernel_dtype = output_shapes[name]

            class _MoeValidator(common_dataclasses.CustomValidator):
                _golden = golden
                _label = name
                # Bound at class-creation time (like _golden/_label above) so each
                # validator closes over the value from its own loop iteration, rather
                # than the loop variable `kernel_dtype` from the last iteration.
                _kernel_dtype = kernel_dtype

                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    """Validate a kernel output against the golden reference."""
                    reshaped = (
                        inference_output.view(dtype=self._kernel_dtype).astype(np.float32).reshape(self._golden.shape)
                    )
                    golden_f32 = self._golden.astype(np.float32)
                    if np.linalg.norm(reshaped) == 0 and np.linalg.norm(golden_f32) == 0:
                        return True
                    passed, metrics = check_correctness(reshaped, golden_f32)
                    if not passed:
                        self._print_with_log(f"[{self._label}] Validation FAILED")
                        self._print_with_log(f"  metrics: {metrics}")
                        self._print_with_log(f"  kernel[0,:5]: {reshaped.flatten()[:5]}")
                        self._print_with_log(f"  golden[0,:5]: {golden_f32.flatten()[:5]}")
                    return passed

            result[name] = common_dataclasses.CustomValidatorWithOutputTensorData(
                validator=_MoeValidator,
                output_ndarray=np.ndarray(shape, dtype=kernel_dtype),
            )
        return result

    return comparator
