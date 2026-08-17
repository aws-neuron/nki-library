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

"""Unit tests for MLP MXFP8 gradient clamping.

Uses MXFP8 golden matmul (swizzle + quantize + nc_matmul_mx) to compute the
reference, matching the kernel's quantization error so that clamping boundary
decisions are consistent between kernel and golden.
"""

from typing import Any, final

import numpy as np
import numpy.typing as npt
import pytest
from neuronxcc.nki._private.private_api import float8_e4m3fn_x4
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import (
    _swizzle,
    golden_matmul,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8.config import (
    ClampLimits,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8.mlp_bwd_mxfp8_kernel import (
    mlp_backward_mxfp8_nki,
)
from typing_extensions import override

from test.integration.nkilib.experimental.mlp_mxfp8.mlp_mxfp8_checkpoint_utils import (
    check_correctness,
    generate_inputs,
    silu_np,
)
from test.integration.nkilib.experimental.mlp_mxfp8.test_mlp_mxfp8_bwd_checkpoint import (
    build_bwd_kernel_input,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.rng import NKITestsRNG
from test.utils.unit_test_framework import UnitTestFramework

# Constants
LNC = 2
COMPUTE_DTYPE_X4 = float8_e4m3fn_x4
BWD_GOLDEN_SEED = 123

_bwd_rng = NKITestsRNG(seed=BWD_GOLDEN_SEED)


# ============================================================================
# MXFP8 golden MLP backward
# ============================================================================


def mxfp8_golden_matmul(a_bf16: npt.NDArray, b_bf16: npt.NDArray) -> npt.NDArray:
    """Compute A @ B.T using MXFP8 golden (swizzle + quantize + nc_matmul_mx).

    Args:
        a_bf16: [M, K] — left operand.
        b_bf16: [N, K] — right operand (transposed internally).

    Returns:
        [M, N] result matching hardware MXFP8 matmul behavior.
    """
    a_sw = _swizzle(a_bf16.astype(np.float32).T.copy())
    b_sw = _swizzle(b_bf16.astype(np.float32).T.copy())
    return golden_matmul(a_sw, b_sw, COMPUTE_DTYPE_X4)


def compute_bwd_golden_mxfp8(
    S: int,
    H: int,
    I: int,
    hidden_np: npt.NDArray,
    gate_up_np: npt.NDArray,
    down_np: npt.NDArray,
    clamp_limits: ClampLimits,
) -> tuple:
    """Compute backward golden using MXFP8 golden matmuls for recompute.

    Uses MXFP8 golden for the recompute matmuls (gate_pre, up) and phase 1
    (d_intermediate) so that quantization error matches the kernel. Downstream
    matmuls (phases 2-4) use FP32 since the test focuses on clamping correctness.

    Returns:
        (output_grad, golden_hs_grad, golden_gate_up_wgrad, golden_dw_wgrad)
    """
    import torch

    bf16 = hidden_np.dtype
    W_gate = gate_up_np[:I, :]  # [I, H]
    W_up = gate_up_np[I:, :]  # [I, H]
    W_down = down_np  # [H, I]

    # Recompute forward using MXFP8 golden (matches kernel recompute phase)
    gate_pre = mxfp8_golden_matmul(hidden_np, W_gate).astype(np.float32)
    up = mxfp8_golden_matmul(hidden_np, W_up).astype(np.float32)

    # Apply activation clamping
    if clamp_limits.non_linear_clamp_upper_limit is not None:
        gate_pre = np.minimum(gate_pre, clamp_limits.non_linear_clamp_upper_limit)
    if clamp_limits.non_linear_clamp_lower_limit is not None:
        gate_pre = np.maximum(gate_pre, clamp_limits.non_linear_clamp_lower_limit)
    if clamp_limits.linear_clamp_upper_limit is not None:
        up = np.minimum(up, clamp_limits.linear_clamp_upper_limit)
    if clamp_limits.linear_clamp_lower_limit is not None:
        up = np.maximum(up, clamp_limits.linear_clamp_lower_limit)

    gate_act = silu_np(gate_pre)
    intermediate_act = gate_act * up

    # Phase 1: d_intermediate = output_grad @ W_down (W_down is [H, I])
    _bwd_rng.reset()
    output_grad = _bwd_rng.kaiming_normal_(torch.empty(S, H)).numpy().astype(bf16)
    # Kernel computes output_grad[S,H] @ W_down.T[I,H] -> [S, I]
    d_intermediate = mxfp8_golden_matmul(output_grad, W_down.T.copy()).astype(np.float32)

    # Compute d_gate and d_up (element-wise, no quantization error)
    sig = 1.0 / (1.0 + np.exp(-gate_pre))
    silu_deriv = sig * (1.0 + gate_pre * (1.0 - sig))
    d_gate = d_intermediate * silu_deriv * up
    d_up = d_intermediate * gate_act

    # Apply gradient clamping
    gate_mask = np.ones_like(d_gate)
    if clamp_limits.non_linear_clamp_upper_limit is not None:
        gate_mask *= (gate_pre < clamp_limits.non_linear_clamp_upper_limit).astype(np.float32)
    if clamp_limits.non_linear_clamp_lower_limit is not None:
        gate_mask *= (gate_pre > clamp_limits.non_linear_clamp_lower_limit).astype(np.float32)
    d_gate = d_gate * gate_mask

    up_mask = np.ones_like(d_up)
    if clamp_limits.linear_clamp_upper_limit is not None:
        up_mask *= (up < clamp_limits.linear_clamp_upper_limit).astype(np.float32)
    if clamp_limits.linear_clamp_lower_limit is not None:
        up_mask *= (up > clamp_limits.linear_clamp_lower_limit).astype(np.float32)
    d_up = d_up * up_mask

    # Downstream matmuls (FP32 — not the focus of this test)
    d_gate_up = np.concatenate([d_gate, d_up], axis=1)
    W_gate_up = np.concatenate([W_gate, W_up], axis=0)
    golden_hs_grad = d_gate_up @ W_gate_up
    golden_gate_wgrad = d_gate.T @ hidden_np.astype(np.float32)
    golden_up_wgrad = d_up.T @ hidden_np.astype(np.float32)
    golden_dw_wgrad = output_grad.astype(np.float32).T @ intermediate_act

    return (
        output_grad,
        golden_hs_grad.astype(np.float32),
        np.concatenate([golden_gate_wgrad, golden_up_wgrad], axis=0).astype(np.float32),
        golden_dw_wgrad.astype(np.float32),
    )


# ============================================================================
# Comparator
# ============================================================================


def _grad_clamp_comparator(golden_dict, output_tensors):
    """Custom comparator for gradient clamp tests."""
    result = {}
    for name, golden in golden_dict.items():
        dtype = golden.dtype

        class _Validator(CustomValidator):
            _golden = golden
            _label = name

            @override
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                reshaped = inference_output.view(dtype=self._golden.dtype).reshape(self._golden.shape)
                passed, metrics = check_correctness(reshaped, self._golden.astype(self._golden.dtype))
                if not passed:
                    self._print_with_log(f"[{self._label}] Validation FAILED")
                    self._print_with_log(f"  metrics: {metrics}")
                    self._print_with_log(f"  kernel[0,:5]: {reshaped.flatten()[:5]}")
                    self._print_with_log(f"  golden[0,:5]: {self._golden.flatten()[:5]}")
                return passed

        result[name] = CustomValidatorWithOutputTensorData(
            validator=_Validator,
            output_ndarray=np.ndarray(golden.shape, dtype=dtype),
        )
    return result


# ============================================================================
# Test configs
# ============================================================================

_CLAMP_TEST_CONFIGS = [
    (2048, 1024, 1024, "small_aligned"),
    (2048 + 256, 1024 + 256, 1024 + 256, "remainder_256"),
]

_CLAMP_LIMITS = [
    ClampLimits(0.05, -0.05, 0.05, -0.05),
    ClampLimits(0.05, -0.05, None, None),
    ClampLimits(None, None, 0.05, -0.05),
    ClampLimits(0.05, None, 0.05, None),
    ClampLimits(None, -0.05, None, -0.05),
]


def _clamp_id(cl):
    parts = []
    if cl.non_linear_clamp_upper_limit is not None:
        parts.append(f"nl_u{cl.non_linear_clamp_upper_limit}")
    if cl.non_linear_clamp_lower_limit is not None:
        parts.append(f"nl_l{cl.non_linear_clamp_lower_limit}")
    if cl.linear_clamp_upper_limit is not None:
        parts.append(f"lin_u{cl.linear_clamp_upper_limit}")
    if cl.linear_clamp_lower_limit is not None:
        parts.append(f"lin_l{cl.linear_clamp_lower_limit}")
    return "_".join(parts)


# ============================================================================
# Test class
# ============================================================================


@pytest_test_metadata(name="MLP MXFP8 Grad Clamp")
@pytest_marks(["mlp_mxfp8_grad_clamp", "mx"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@final
class TestMlpMxfp8GradClamp:
    """Tests for MLP MXFP8 gradient clamping using MXFP8 golden reference."""

    def _run_clamp_test(
        self,
        test_manager,
        platform_target,
        S: int,
        H: int,
        I: int,
        clamp_limits: ClampLimits,
        provide_checkpoints: bool,
    ) -> None:
        """Helper: run a backward clamp test with or without checkpoints."""
        hidden_np, gate_up_np, down_np = generate_inputs(S, H, I)
        bf16 = hidden_np.dtype

        # Compute MXFP8-aware golden
        output_grad, golden_hs_grad, golden_gate_up_wgrad, golden_dw_wgrad = compute_bwd_golden_mxfp8(
            S, H, I, hidden_np, gate_up_np, down_np, clamp_limits
        )

        # Compute clamped intermediates using MXFP8 golden
        gate_pre = mxfp8_golden_matmul(hidden_np, gate_up_np[:I, :]).astype(np.float32)
        up = mxfp8_golden_matmul(hidden_np, gate_up_np[I:, :]).astype(np.float32)
        if clamp_limits.non_linear_clamp_upper_limit is not None:
            gate_pre = np.minimum(gate_pre, clamp_limits.non_linear_clamp_upper_limit)
        if clamp_limits.non_linear_clamp_lower_limit is not None:
            gate_pre = np.maximum(gate_pre, clamp_limits.non_linear_clamp_lower_limit)
        if clamp_limits.linear_clamp_upper_limit is not None:
            up = np.minimum(up, clamp_limits.linear_clamp_upper_limit)
        if clamp_limits.linear_clamp_lower_limit is not None:
            up = np.maximum(up, clamp_limits.linear_clamp_lower_limit)
        intermediate_act = (silu_np(gate_pre) * up).astype(bf16)

        checkpoint_tensors = None
        if provide_checkpoints:
            checkpoint_tensors = {
                "gate_pre": gate_pre.astype(bf16),
                "gate_act": silu_np(gate_pre).astype(bf16),
                "up": up.astype(bf16),
                "intermediate": intermediate_act,
            }

        kernel_input = build_bwd_kernel_input(
            S,
            H,
            I,
            hidden_np,
            gate_up_np,
            down_np,
            intermediate_act,
            output_grad,
            spill_reload=False,
            use_scale_packing=False,
            checkpoint_tensors=checkpoint_tensors,
            clamp_limits=clamp_limits,
        )

        golden_dict = {
            "hidden_states_grad": golden_hs_grad.astype(bf16),
            "gate_up_weight_grad": golden_gate_up_wgrad.astype(bf16),
            "down_proj_weight_grad": golden_dw_wgrad.astype(bf16),
        }

        def torch_ref(
            output_grad=None,
            hidden_states=None,
            down_proj_weight=None,
            gate_up_weights=None,
            gate_up_weight_T=None,
            gate_up_weight_T_scales=None,
            gate_up_weights_scales=None,
            down_weight_T=None,
            down_weight_T_scales=None,
            output_grad_T=None,
            output_grad_T_scales=None,
            hidden_states_T=None,
            hidden_states_T_scales=None,
            gate_pre=None,
            gate_act=None,
            up=None,
            intermediate=None,
            run_with_lnc2=True,
            matmul_config=None,
            fp8_x4_dtype=None,
            spill_reload=False,
            use_scale_packing=False,
            clamp_limits=None,
        ):
            return golden_dict

        compiler_args = CompilerArgs(
            logical_nc_config=LNC,
            platform_target=platform_target,
        )

        output_ndarray = {
            "hidden_states_grad": np.zeros((S, H), dtype=bf16),
            "gate_up_weight_grad": np.zeros((2 * I, H), dtype=bf16),
            "down_proj_weight_grad": np.zeros((H, I), dtype=bf16),
        }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=mlp_backward_mxfp8_nki,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: kernel_input,
            output_tensor_descriptor=lambda _: output_ndarray.copy(),
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_grad_clamp_comparator,
        )

    @pytest.mark.parametrize(
        "seqlen,hidden,intermediate",
        [(c[0], c[1], c[2]) for c in _CLAMP_TEST_CONFIGS],
        ids=[c[3] for c in _CLAMP_TEST_CONFIGS],
    )
    @pytest.mark.parametrize("clamp_limits", _CLAMP_LIMITS, ids=[_clamp_id(c) for c in _CLAMP_LIMITS])
    def test_grad_clamp(
        self, test_manager, platform_target: Any, seqlen: int, hidden: int, intermediate: int, clamp_limits: ClampLimits
    ):
        """Test gradient clamping with all checkpoints provided (isolates clamp logic)."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self._run_clamp_test(
            test_manager, platform_target, seqlen, hidden, intermediate, clamp_limits, provide_checkpoints=True
        )

    @pytest.mark.parametrize(
        "seqlen,hidden,intermediate",
        [(c[0], c[1], c[2]) for c in _CLAMP_TEST_CONFIGS],
        ids=[c[3] for c in _CLAMP_TEST_CONFIGS],
    )
    @pytest.mark.parametrize("clamp_limits", _CLAMP_LIMITS, ids=[_clamp_id(c) for c in _CLAMP_LIMITS])
    def test_activation_clamp_recompute(
        self, test_manager, platform_target: Any, seqlen: int, hidden: int, intermediate: int, clamp_limits: ClampLimits
    ):
        """Test activation clamping in recompute stages (no checkpoints, kernel recomputes)."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self._run_clamp_test(
            test_manager, platform_target, seqlen, hidden, intermediate, clamp_limits, provide_checkpoints=False
        )
