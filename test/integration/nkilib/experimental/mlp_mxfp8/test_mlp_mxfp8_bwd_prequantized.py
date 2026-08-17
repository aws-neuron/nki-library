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

"""Integration tests for MLP MXFP8 backward kernels with pre-quantized inputs.

Tests both modes:
- Mode 2 (weights prequantized): MXFP8 weights, BF16 activations, 4 internal transposes
- Mode 3 (all prequantized): MXFP8 weights + hidden_T + output_grad + output_grad_T, BF16 intermediate, 2 internal transposes
"""

from typing import Any, final

import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.mlp_mxfp8.mlp_bwd_mxfp8.mlp_bwd_mxfp8_kernel import (
    mlp_backward_mxfp8_nki,
)
from typing_extensions import override

from test.integration.nkilib.experimental.mlp_mxfp8.mlp_mxfp8_checkpoint_utils import (
    ALL_CHECKPOINTS_ENABLED,
    LNC,
    MODEL_CONFIGS,
    PERF_BENCH_MODEL_CONFIG,
    PERF_BENCH_MODEL_CONFIGS,
    check_correctness,
    generate_inputs,
    golden_mlp_fwd,
    prequantize_for_mlp,
)
from test.integration.nkilib.experimental.mlp_mxfp8.test_mlp_mxfp8_bwd_checkpoint import (
    compute_bwd_golden,
)
from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework

# ============================================================================
# Custom comparator
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
# Mode 2: Weights pre-quantized (4 internal transposes)
# ============================================================================


def _make_weights_pq_torch_ref(original_hidden, original_gate_up, original_down):
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


@pytest_test_metadata(name="MLP MXFP8 Bwd Prequantized")
@pytest_marks(["mlp_mxfp8_prequantized", "mx"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMlpMxfp8BwdWeightsPrequantized:
    """Tests for MLP MXFP8 backward kernel with pre-quantized/pre-swizzled weights (Mode 2)."""

    def _run_framework(self, test_manager, platform_target, S, H, I, kernel_input, hidden_np, gate_up_np, down_np):
        bf16 = hidden_np.dtype
        bwd_torch_ref = _make_weights_pq_torch_ref(hidden_np, gate_up_np, down_np)
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
        framework.run_test(test_config=None, compiler_args=compiler_args, custom_comparator=_bwd_comparator)

    def run_prequantized_test(
        self,
        test_manager,
        platform_target,
        S,
        H,
        I,
        spill_reload=True,
        use_scale_packing=True,
        checkpoint_combo=ALL_CHECKPOINTS_ENABLED,
    ):
        hidden_np, gate_up_np, down_np = generate_inputs(S, H, I)
        bf16 = hidden_np.dtype

        _, golden_gp, golden_ga, golden_up, golden_intermediate = golden_mlp_fwd(hidden_np, gate_up_np, down_np)
        output_grad, *_ = compute_bwd_golden(
            S, H, I, hidden_np, gate_up_np, down_np, golden_gp, golden_ga, golden_up, golden_intermediate
        )

        gate_up_weight_T_data, gate_up_weight_T_scales = prequantize_for_mlp(gate_up_np.T, use_scale_packing)
        gate_up_weights_data, gate_up_weights_scales = prequantize_for_mlp(gate_up_np, use_scale_packing)
        down_weight_T_data, down_weight_T_scales = prequantize_for_mlp(down_np.astype(bf16).T, use_scale_packing)

        save_gate_pre, save_gate_act, save_up, save_intermediate = checkpoint_combo
        checkpoint_kwargs = {}
        if save_gate_pre:
            checkpoint_kwargs["gate_pre"] = golden_gp.astype(bf16)
        if save_gate_act:
            checkpoint_kwargs["gate_act"] = golden_ga.astype(bf16)
        if save_up:
            checkpoint_kwargs["up"] = golden_up.astype(bf16)
        if save_intermediate:
            checkpoint_kwargs["intermediate"] = golden_intermediate.astype(bf16)

        kernel_input = {
            "output_grad": output_grad,
            "hidden_states": hidden_np,
            "gate_up_weight_T": gate_up_weight_T_data,
            "gate_up_weights": gate_up_weights_data,
            "down_weight_T": down_weight_T_data,
            "gate_up_weight_T_scales": gate_up_weight_T_scales,
            "gate_up_weights_scales": gate_up_weights_scales,
            "down_weight_T_scales": down_weight_T_scales,
            "spill_reload": spill_reload,
            "use_scale_packing": use_scale_packing,
            **checkpoint_kwargs,
        }
        self._run_framework(test_manager, platform_target, S, H, I, kernel_input, hidden_np, gate_up_np, down_np)

    @pytest.mark.parametrize("model_cfg", MODEL_CONFIGS[:4], ids=[c.name for c in MODEL_CONFIGS[:4]])
    def test_weights_pq(self, test_manager, platform_target, model_cfg):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager, platform_target, model_cfg.seq_len, model_cfg.hidden_size, model_cfg.intermediate_size
        )

    @pytest.mark.parametrize("model_cfg", MODEL_CONFIGS[:4], ids=[c.name for c in MODEL_CONFIGS[:4]])
    def test_weights_pq_recompute(self, test_manager, platform_target, model_cfg):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager,
            platform_target,
            model_cfg.seq_len,
            model_cfg.hidden_size,
            model_cfg.intermediate_size,
            checkpoint_combo=(False, False, False, False),
        )

    def test_weights_pq_fast(self, test_manager, platform_target):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager,
            platform_target,
            PERF_BENCH_MODEL_CONFIG.seq_len,
            PERF_BENCH_MODEL_CONFIG.hidden_size,
            PERF_BENCH_MODEL_CONFIG.intermediate_size,
        )

    @pytest.mark.parametrize("test_name", ["test_weights_pq_perf_bench"])
    @pytest.mark.parametrize("model_cfg", PERF_BENCH_MODEL_CONFIGS, ids=[c.name for c in PERF_BENCH_MODEL_CONFIGS])
    def test_weights_pq_perf_bench(self, test_manager, platform_target, model_cfg, test_name):
        """Perf bench: dense models with scale packing, spill reload, and all checkpoints enabled."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager,
            platform_target,
            model_cfg.seq_len,
            model_cfg.hidden_size,
            model_cfg.intermediate_size,
            spill_reload=True,
            use_scale_packing=True,
            checkpoint_combo=ALL_CHECKPOINTS_ENABLED,
        )

    @pytest.mark.parametrize("model_cfg", MODEL_CONFIGS[6:9], ids=[c.name for c in MODEL_CONFIGS[6:9]])
    def test_weights_pq_remainder(self, test_manager, platform_target, model_cfg):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager, platform_target, model_cfg.seq_len, model_cfg.hidden_size, model_cfg.intermediate_size
        )


# ============================================================================
# Mode 3: All pre-quantized (2 internal transposes)
# ============================================================================


def _make_all_pq_torch_ref(original_hidden, original_gate_up, original_down):
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


@pytest_marks(["mlp_mxfp8_prequantized", "mx"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMlpMxfp8BwdAllPrequantized:
    """Tests for MLP MXFP8 backward kernel with all pre-quantized/pre-swizzled inputs (Mode 3)."""

    def _run_framework(self, test_manager, platform_target, S, H, I, kernel_input, hidden_np, gate_up_np, down_np):
        bf16 = hidden_np.dtype
        bwd_torch_ref = _make_all_pq_torch_ref(hidden_np, gate_up_np, down_np)
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
        framework.run_test(test_config=None, compiler_args=compiler_args, custom_comparator=_bwd_comparator)

    def run_prequantized_test(
        self,
        test_manager,
        platform_target,
        S,
        H,
        I,
        spill_reload=True,
        use_scale_packing=True,
        checkpoint_combo=ALL_CHECKPOINTS_ENABLED,
        prequantize_output_grad_T=True,
        prequantize_hidden_states_T=True,
    ):
        hidden_np, gate_up_np, down_np = generate_inputs(S, H, I)
        bf16 = hidden_np.dtype

        _, golden_gp, golden_ga, golden_up, golden_intermediate = golden_mlp_fwd(hidden_np, gate_up_np, down_np)
        output_grad, *_ = compute_bwd_golden(
            S, H, I, hidden_np, gate_up_np, down_np, golden_gp, golden_ga, golden_up, golden_intermediate
        )

        gate_up_weight_T_data, gate_up_weight_T_scales = prequantize_for_mlp(gate_up_np.T, use_scale_packing)
        gate_up_weights_data, gate_up_weights_scales = prequantize_for_mlp(gate_up_np, use_scale_packing)
        down_weight_T_data, down_weight_T_scales = prequantize_for_mlp(down_np.astype(bf16).T, use_scale_packing)

        activation_kwargs = {}
        if prequantize_output_grad_T:
            output_grad_T_data, output_grad_T_scales = prequantize_for_mlp(output_grad.T, use_scale_packing)
            activation_kwargs["output_grad_T"] = output_grad_T_data
            activation_kwargs["output_grad_T_scales"] = output_grad_T_scales
        if prequantize_hidden_states_T:
            hidden_states_T_data, hidden_states_T_scales = prequantize_for_mlp(hidden_np.T, use_scale_packing)
            activation_kwargs["hidden_states_T"] = hidden_states_T_data
            activation_kwargs["hidden_states_T_scales"] = hidden_states_T_scales

        save_gate_pre, save_gate_act, save_up, save_intermediate = checkpoint_combo
        checkpoint_kwargs = {}
        if save_gate_pre:
            checkpoint_kwargs["gate_pre"] = golden_gp.astype(bf16)
        if save_gate_act:
            checkpoint_kwargs["gate_act"] = golden_ga.astype(bf16)
        if save_up:
            checkpoint_kwargs["up"] = golden_up.astype(bf16)
        checkpoint_kwargs["intermediate"] = golden_intermediate.astype(bf16)

        kernel_input = {
            "output_grad": output_grad,
            "hidden_states": hidden_np,
            "gate_up_weights": gate_up_weights_data,
            "gate_up_weight_T": gate_up_weight_T_data,
            "gate_up_weight_T_scales": gate_up_weight_T_scales,
            "gate_up_weights_scales": gate_up_weights_scales,
            "down_weight_T": down_weight_T_data,
            "down_weight_T_scales": down_weight_T_scales,
            "spill_reload": spill_reload,
            "use_scale_packing": use_scale_packing,
            **activation_kwargs,
            **checkpoint_kwargs,
        }
        self._run_framework(test_manager, platform_target, S, H, I, kernel_input, hidden_np, gate_up_np, down_np)

    @pytest.mark.parametrize("model_cfg", MODEL_CONFIGS[:4], ids=[c.name for c in MODEL_CONFIGS[:4]])
    def test_all_pq(self, test_manager, platform_target, model_cfg):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager, platform_target, model_cfg.seq_len, model_cfg.hidden_size, model_cfg.intermediate_size
        )

    @pytest.mark.parametrize("model_cfg", MODEL_CONFIGS[:4], ids=[c.name for c in MODEL_CONFIGS[:4]])
    def test_all_pq_recompute(self, test_manager, platform_target, model_cfg):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager,
            platform_target,
            model_cfg.seq_len,
            model_cfg.hidden_size,
            model_cfg.intermediate_size,
            checkpoint_combo=(False, False, False, False),
        )

    def test_all_pq_fast(self, test_manager, platform_target):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager,
            platform_target,
            PERF_BENCH_MODEL_CONFIG.seq_len,
            PERF_BENCH_MODEL_CONFIG.hidden_size,
            PERF_BENCH_MODEL_CONFIG.intermediate_size,
        )

    @pytest.mark.parametrize("model_cfg", MODEL_CONFIGS[6:9], ids=[c.name for c in MODEL_CONFIGS[6:9]])
    def test_all_pq_remainder(self, test_manager, platform_target, model_cfg):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        self.run_prequantized_test(
            test_manager, platform_target, model_cfg.seq_len, model_cfg.hidden_size, model_cfg.intermediate_size
        )

    @pytest.mark.parametrize("use_scale_packing", [True, False], ids=["scale_packed", "scale_unpacked"])
    @pytest.mark.parametrize("spill_reload", [True, False], ids=["spill", "no_spill"])
    def test_all_pq_scale_packing_spill(self, test_manager, platform_target, use_scale_packing, spill_reload):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        cfg = MODEL_CONFIGS[2]
        self.run_prequantized_test(
            test_manager,
            platform_target,
            cfg.seq_len,
            cfg.hidden_size,
            cfg.intermediate_size,
            use_scale_packing=use_scale_packing,
            spill_reload=spill_reload,
        )

    @pytest.mark.parametrize(
        "pq_output_grad_T,pq_hidden_states_T",
        [(True, False), (False, True), (False, False)],
        ids=["output_grad_T_only", "hidden_states_T_only", "none_prequantized"],
    )
    def test_mixed_pq(self, test_manager, platform_target, pq_output_grad_T, pq_hidden_states_T):
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")
        cfg = MODEL_CONFIGS[2]
        self.run_prequantized_test(
            test_manager,
            platform_target,
            cfg.seq_len,
            cfg.hidden_size,
            cfg.intermediate_size,
            prequantize_output_grad_T=pq_output_grad_T,
            prequantize_hidden_states_T=pq_hidden_states_T,
        )
