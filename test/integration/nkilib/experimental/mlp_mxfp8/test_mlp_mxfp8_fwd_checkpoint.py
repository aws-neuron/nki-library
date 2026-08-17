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

"""Integration tests for MLP MXFP8 forward kernel with activation checkpointing."""

from typing import Any, final

import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8.config import (
    get_config_for_shape as get_fwd_config,
)
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_fwd_mxfp8.mlp_fwd_mxfp8_kernel import (
    mlp_forward_mxfp8_nki,
)
from typing_extensions import override

from test.integration.nkilib.experimental.mlp_mxfp8.mlp_mxfp8_checkpoint_utils import (
    _INPUT_MODES,
    ALL_CHECKPOINT_COMBOS,
    ALL_CHECKPOINTS_ENABLED,
    LNC,
    MODEL_CONFIGS,
    PERF_BENCH_MODEL_CONFIG,
    ModelConfig,
    apply_input_mode,
    check_correctness,
    generate_inputs,
    golden_mlp_fwd,
)
from test.utils import common_dataclasses, coverage_parametrized_tests, test_orchestrator
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework

# ============================================================================
# Custom comparator for MXFP8 forward validation
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


# ============================================================================
# Coverage parametrize abbreviations and filters
# ============================================================================

_FWD_ABBREVS = {
    "model_cfg": "cfg",
    "input_mode": "in",
    "checkpoint_combo": "ckpt",
    "spill_reload": "sr",
    "use_scale_packing": "sp",
}


def _fwd_filter(
    model_cfg=None,
    input_mode=None,
    checkpoint_combo=None,
    spill_reload=None,
    use_scale_packing=None,
) -> coverage_parametrized_tests.FilterResult:
    """Filter for forward sweep combinations."""
    return coverage_parametrized_tests.FilterResult.VALID


# ============================================================================
# Forward test class
# ============================================================================


@pytest_test_metadata(name="MLP MXFP8 Fwd Checkpoint")
@pytest_marks(["mlp_mxfp8_checkpoint", "mx"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMlpMxfp8FwdCheckpoint:
    """Tests for MLP MXFP8 forward kernel with activation checkpointing."""

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
        """Forward sweep: model configs x input modes x checkpoint combos x spill/pack."""
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

    # ================================================================
    # Perf bench tests — qwen3_8b_tp4, all checkpoints enabled
    # ================================================================

    @pytest.mark.parametrize("input_mode", _INPUT_MODES, ids=_INPUT_MODES)
    def test_fwd_perf_bench(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target: Any,
        input_mode: str,
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
