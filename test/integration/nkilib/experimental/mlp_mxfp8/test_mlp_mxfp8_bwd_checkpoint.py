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

"""Integration tests for MLP MXFP8 backward kernel with activation checkpointing."""

from typing import Any, final

import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8.mlp_bwd_mxfp8_kernel import (
    mlp_backward_mxfp8_nki,
)
from typing_extensions import override

from test.integration.nkilib.experimental.mlp_mxfp8.mlp_mxfp8_checkpoint_utils import (
    ALL_CHECKPOINT_COMBOS,
    ALL_CHECKPOINTS_ENABLED,
    BWD_GOLDEN_SEED,
    LNC,
    MODEL_CONFIGS,
    PERF_BENCH_MODEL_CONFIG,
    PERF_BENCH_MODEL_CONFIGS,
    RANDOM_MODEL_CONFIGS,
    ModelConfig,
    check_correctness,
    generate_inputs,
    golden_mlp_fwd,
)
from test.utils import common_dataclasses, coverage_parametrized_tests, test_orchestrator
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.rng import NKITestsRNG
from test.utils.unit_test_framework import UnitTestFramework

_bwd_rng = NKITestsRNG(seed=BWD_GOLDEN_SEED)

# ============================================================================
# Custom comparator for MXFP8 backward validation
# ============================================================================


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
# Backward golden computation
# ============================================================================


def compute_bwd_golden(
    S: int,
    H: int,
    I: int,
    hidden_np,
    gate_up_np,
    down_np,
    golden_gp,
    golden_ga,
    golden_up,
    golden_intermediate,
) -> tuple:
    """Compute backward golden reference tensors.

    Returns (output_grad, golden_hs_grad, golden_gate_wgrad, golden_up_wgrad, golden_dw_wgrad).
    """
    import torch

    bf16 = hidden_np.dtype
    _bwd_rng.reset()
    output_grad = _bwd_rng.kaiming_normal_(torch.empty(S, H)).numpy().astype(bf16)
    og32 = output_grad.astype(np.float32)
    W_gate = gate_up_np[:I, :].astype(np.float32)
    W_up = gate_up_np[I:, :].astype(np.float32)
    W_down = down_np.astype(np.float32)

    d_intermediate = og32 @ W_down

    gp = golden_gp.astype(np.float32)
    up = golden_up.astype(np.float32)
    ga = golden_ga.astype(np.float32)
    intermediate_f32 = golden_intermediate.astype(np.float32)

    sig = 1.0 / (1.0 + np.exp(-gp))
    silu_deriv = sig * (1.0 + gp * (1.0 - sig))
    d_gate = d_intermediate * silu_deriv * up
    d_up = d_intermediate * ga

    d_gate_up = np.concatenate([d_gate, d_up], axis=1)
    W_gate_up = np.concatenate([W_gate, W_up], axis=0)
    golden_hs_grad = d_gate_up @ W_gate_up
    golden_gate_wgrad = d_gate.T @ hidden_np.astype(np.float32)
    golden_up_wgrad = d_up.T @ hidden_np.astype(np.float32)
    golden_dw_wgrad = og32.T @ intermediate_f32

    return output_grad, golden_hs_grad, golden_gate_wgrad, golden_up_wgrad, golden_dw_wgrad


def build_bwd_kernel_input(
    S: int,
    H: int,
    I: int,
    hidden_np,
    gate_up_np,
    down_np,
    golden_intermediate,
    output_grad,
    spill_reload: bool,
    use_scale_packing: bool,
    checkpoint_tensors: dict | None = None,
    clamp_limits=None,
) -> dict:
    """Build the kernel_input dict for backward tests.

    checkpoint_tensors: optional dict with keys from {gate_pre, gate_act, up, intermediate}.
    clamp_limits: optional ClampLimits for gradient clamping.
    """
    bf16 = hidden_np.dtype

    kernel_input = {
        "output_grad": output_grad,
        "hidden_states": hidden_np,
        "down_proj_weight": down_np.astype(bf16),
        "gate_up_weights": gate_up_np,
        "spill_reload": spill_reload,
        "use_scale_packing": use_scale_packing,
    }

    if checkpoint_tensors:
        for key, val in checkpoint_tensors.items():
            kernel_input[key] = val

    if clamp_limits is not None:
        kernel_input["clamp_limits"] = clamp_limits

    return kernel_input


# ============================================================================
# Torch ref — signature must match kernel; golden via custom_comparator
# ============================================================================


def _make_bwd_torch_ref(original_hidden, original_gate_up, original_down):
    """Create a backward torch_ref that always computes real golden.

    Captures original BF16 tensors via closure so golden computation works
    even after apply_input_mode mutates kernel_input in-place.
    """

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
        """Compute backward golden gradients from original BF16 tensors."""
        S, H = original_hidden.shape
        I = original_gate_up.shape[0] // 2
        bf16 = original_hidden.dtype

        _, golden_gp, golden_ga, golden_up, golden_intermediate = golden_mlp_fwd(
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
            golden_intermediate,
        )
        return {
            "hidden_states_grad": golden_hs_grad.astype(bf16),
            "gate_up_weight_grad": np.concatenate([golden_gate_wgrad, golden_up_wgrad], axis=0).astype(bf16),
            "down_proj_weight_grad": golden_dw_wgrad.astype(bf16),
        }

    return torch_ref


# ============================================================================
# Coverage parametrize abbreviations and filters
# ============================================================================

_BWD_ABBREVS = {
    "model_cfg": "cfg",
    "checkpoint_combo": "ckpt",
    "spill_reload": "sr",
    "use_scale_packing": "sp",
}


def _bwd_filter(
    model_cfg=None,
    checkpoint_combo=None,
    spill_reload=None,
    use_scale_packing=None,
) -> coverage_parametrized_tests.FilterResult:
    """Filter for backward sweep combinations."""
    return coverage_parametrized_tests.FilterResult.VALID


# ============================================================================
# Backward test class
# ============================================================================


@pytest_test_metadata(name="MLP MXFP8 Bwd Checkpoint")
@pytest_marks(["mlp_mxfp8_checkpoint", "mx"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMlpMxfp8BwdCheckpoint:
    """Tests for MLP MXFP8 backward kernel with activation checkpointing."""

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
    ) -> None:
        """Run a backward MLP test end-to-end."""
        hidden_np, gate_up_np, down_np = generate_inputs(S, H, I)
        bf16 = hidden_np.dtype

        _, golden_gp, golden_ga, golden_up, golden_intermediate = golden_mlp_fwd(hidden_np, gate_up_np, down_np)
        output_grad, *_ = compute_bwd_golden(
            S, H, I, hidden_np, gate_up_np, down_np, golden_gp, golden_ga, golden_up, golden_intermediate
        )

        # Build checkpoint tensors from combo
        save_gate_pre, save_gate_act, save_up, save_hidden = checkpoint_combo
        checkpoint_tensors = {}

        ckpt_gp = golden_gp.astype(np.float32)
        ckpt_up = golden_up.astype(np.float32)
        ckpt_ga = golden_ga.astype(np.float32)
        ckpt_intermediate = golden_intermediate.astype(np.float32)

        if save_gate_pre:
            checkpoint_tensors["gate_pre"] = ckpt_gp.astype(bf16)
        if save_gate_act:
            checkpoint_tensors["gate_act"] = ckpt_ga.astype(bf16)
        if save_up:
            checkpoint_tensors["up"] = ckpt_up.astype(bf16)
        if save_hidden:
            checkpoint_tensors["intermediate"] = ckpt_intermediate.astype(bf16)

        kernel_input = build_bwd_kernel_input(
            S,
            H,
            I,
            hidden_np,
            gate_up_np,
            down_np,
            golden_intermediate,
            output_grad,
            spill_reload,
            use_scale_packing,
            checkpoint_tensors,
        )

        bwd_torch_ref = _make_bwd_torch_ref(hidden_np, gate_up_np, down_np)

        compiler_args = common_dataclasses.CompilerArgs(
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
        model_cfg=MODEL_CONFIGS + RANDOM_MODEL_CONFIGS,
        checkpoint_combo=ALL_CHECKPOINT_COMBOS,
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
        spill_reload: bool,
        use_scale_packing: bool,
        is_negative_test_case: bool,
    ) -> None:
        """Backward sweep: model configs x checkpoint combos x spill/pack."""
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
        )

    # ================================================================
    # Perf bench tests — dense models, scale packing + spill reload +
    # all checkpoints enabled
    # ================================================================

    @pytest.mark.parametrize("test_name", ["test_bwd_perf_bench"])
    @pytest.mark.parametrize("model_cfg", PERF_BENCH_MODEL_CONFIGS, ids=[c.name for c in PERF_BENCH_MODEL_CONFIGS])
    def test_bwd_perf_bench(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
        model_cfg: ModelConfig,
        test_name: str,
    ) -> None:
        """Backward perf bench: dense models with all save options enabled."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_bwd_test(
            test_manager,
            platform_target,
            model_cfg.seq_len,
            model_cfg.hidden_size,
            model_cfg.intermediate_size,
            True,
            True,
            checkpoint_combo=ALL_CHECKPOINTS_ENABLED,
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
        )
