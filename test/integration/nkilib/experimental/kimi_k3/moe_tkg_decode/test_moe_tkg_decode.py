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

"""Integration tests for Kimi-K3's routed-expert MXFP4 decode kernel."""

from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
)
from nkilib_src.nkilib.experimental.kimi_k3.moe_tkg_decode import (
    moe_tkg_decode,
    moe_tkg_decode_torch_ref,
)

from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_utils import build_moe_tkg
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_BETA = 4.0
_LINEAR_BETA = 25.0
# Trn3 MX matmul and Scalar Engine transcendental results can differ from the
# host oracle by one BF16 ULP while remaining deterministic.
_DEVICE_RTOL = 2**-7
_DEVICE_ATOL = 2**-8
_TAIL_RTOL = 2**-8
_TAIL_ATOL = 2**-8

# The fast canaries cover decode T=1 and a multi-token H multiple. Production
# cases cover Kimi's D=3584 tail and both expected local-expert counts.
_TEST_CASES = [
    pytest.param(1, 512, 512, 3, 2, marks=pytest.mark.fast, id="t1-aligned"),
    pytest.param(4, 1024, 512, 3, 2, marks=pytest.mark.fast, id="multi-h-multiple"),
    pytest.param(1, 3584, 3072, 7, 4, id="production-t1-e7"),
    pytest.param(4, 3584, 3072, 16, 8, id="production-multi-e16"),
]


def _generate_inputs(tokens: int, hidden: int, intermediate: int, experts: int, top_k: int):
    np.random.seed(42)
    raw = build_moe_tkg(
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        expert=experts,
        top_k=top_k,
        act_fn=ActFnType.SiLU,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        is_all_expert=True,
        quant_dtype=nl.float4_e2m1fn_x4,
        quant_type=QuantizationType.MX,
        in_dtype=nl.bfloat16,
        out_dtype=nl.bfloat16,
        expert_affinities_dtype=nl.bfloat16,
        bias=False,
        clamp=False,
    )
    return {
        # build_moe_tkg emits the legacy TKG H shuffle. Kimi's entry point
        # consumes the natural model layout.
        "latent": (
            raw["hidden_input"].reshape(tokens, 4, hidden // 4).transpose(0, 2, 1).reshape(tokens, hidden).copy()
        ),
        "local_affinities": raw["expert_affinities"],
        "expert_gate_up_weights": raw["expert_gate_up_weights"],
        "expert_down_weights": raw["expert_down_weights"],
        "expert_gate_up_weights_scale": raw["expert_gate_up_weights_scale"],
        "expert_down_weights_scale": raw["expert_down_weights_scale"],
        "beta": _BETA,
        "linear_beta": _LINEAR_BETA,
    }


def _generate_tail_perturbation_inputs():
    inputs = _generate_inputs(tokens=2, hidden=3584, intermediate=512, experts=1, top_k=1)
    inputs["latent"][:] = 0.0
    inputs["latent"][1, 0] = 1.0
    inputs["local_affinities"][:] = 1.0
    # Every packed nibble is +1.0 (E2M1 code 0b0010), with unit E8M0
    # scales. All output columns must therefore have the same nonzero delta.
    inputs["expert_gate_up_weights"].view(np.uint16)[:] = 0x2222
    inputs["expert_down_weights"].view(np.uint16)[:] = 0x2222
    inputs["expert_gate_up_weights_scale"][:] = 127
    inputs["expert_down_weights_scale"][:] = 127
    return inputs


def _tail_response_comparator(golden_dict, output_tensors):
    golden = golden_dict["out"].astype(np.float32)

    class _TailResponseValidator(CustomValidator):
        def validate(self, inference_output: npt.NDArray[Any]) -> bool:
            actual = np.frombuffer(inference_output, dtype=ml_dtypes.bfloat16).astype(np.float32).reshape(golden.shape)
            abs_error = np.abs(actual - golden)
            max_abs_error = float(np.max(abs_error))
            relative_error = abs_error / np.maximum(np.abs(golden), _TAIL_ATOL)
            max_relative_error = float(np.max(relative_error))
            close = bool(np.allclose(actual, golden, rtol=_TAIL_RTOL, atol=_TAIL_ATOL))

            expected_response = np.abs(golden[1] - golden[0]) > 0.0
            actual_response = np.abs(actual[1] - actual[0]) > 0.0
            expected_count = int(np.count_nonzero(expected_response))
            actual_count = int(np.count_nonzero(actual_response))
            tail_expected_count = int(np.count_nonzero(expected_response[-512:]))
            tail_actual_count = int(np.count_nonzero(actual_response[-512:]))
            self._print_with_log(
                "Kimi decode accuracy: "
                f"max_abs={max_abs_error:.6e}, max_rel={max_relative_error:.6e}; "
                f"response={actual_count}/3584, tail={tail_actual_count}/512"
            )
            return (
                close
                and expected_count == 3584
                and actual_count == 3584
                and tail_expected_count == 512
                and tail_actual_count == 512
            )

    return {
        "out": CustomValidatorWithOutputTensorData(
            validator=_TailResponseValidator,
            output_ndarray=output_tensors["out"],
        )
    }


@pytest_test_metadata(name="Kimi-K3 MoE MXFP4 Decode", tags=["model"])
@pytest_marks(["moe", "decode", "mx"])
@final
class TestMoeTkgDecode:
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("tokens, hidden, intermediate, experts, top_k", _TEST_CASES)
    def test_moe_tkg_decode(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        tokens: int,
        hidden: int,
        intermediate: int,
        experts: int,
        top_k: int,
    ):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_tkg_decode,
            torch_ref=torch_ref_wrapper(moe_tkg_decode_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(tokens, hidden, intermediate, experts, top_k),
            output_tensor_descriptor=lambda _: {"out": np.zeros((tokens, hidden), dtype=nl.bfloat16)},
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            rtol=_DEVICE_RTOL,
            atol=_DEVICE_ATOL,
            inference_args=TKG_INFERENCE_ARGS,
        )

    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_all_hidden_columns_respond_to_input_perturbation(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
    ):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=moe_tkg_decode,
            torch_ref=torch_ref_wrapper(moe_tkg_decode_torch_ref),
            kernel_input_generator=lambda _: _generate_tail_perturbation_inputs(),
            output_tensor_descriptor=lambda _: {"out": np.zeros((2, 3584), dtype=nl.bfloat16)},
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            inference_args=TKG_INFERENCE_ARGS,
            custom_comparator=_tail_response_comparator,
        )
