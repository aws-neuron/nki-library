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

"""Integration tests for the Kimi-K3 exact two-sided SiTU MLP."""

from copy import deepcopy
from inspect import Parameter, signature
from typing import final

import ml_dtypes
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.kimi_k3.situ_mlp import (
    kimi_k3_situ_mlp,
    kimi_k3_situ_mlp_torch_ref,
)

from test.utils.common_dataclasses import TKG_INFERENCE_ARGS, CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_BETA = 4.0
_LINEAR_BETA = 25.0
_RTOL = 2**-7
_ATOL = 2**-8

_EXECUTION_CASES = [
    pytest.param(1, 512, 512, marks=pytest.mark.fast, id="decode-t1"),
    pytest.param(64, 512, 512, marks=pytest.mark.fast, id="prefill-t64"),
    pytest.param(4, 224, 192, marks=pytest.mark.fast, id="tp32-tail"),
    pytest.param(4, 224, 1056, marks=pytest.mark.fast, id="tp32-dense-tail"),
]

_PRODUCTION_COMPILE_CASES = [
    pytest.param(1, 7168, 6144, id="shared-decode"),
    pytest.param(64, 7168, 6144, id="shared-prefill"),
    pytest.param(1, 7168, 33792, id="dense-decode"),
    pytest.param(64, 7168, 33792, id="dense-prefill"),
]


def _generate_inputs(
    tokens: int, hidden: int, intermediate: int, beta: float = _BETA, linear_beta: float = _LINEAR_BETA
):
    rng = np.random.default_rng(42)

    def normal(shape, scale):
        values = rng.standard_normal(shape, dtype=np.float32)
        return (values * scale).astype(ml_dtypes.bfloat16)

    return {
        "hidden_states": normal((tokens, hidden), 0.5),
        "gate_proj_weight": normal((intermediate, hidden), hidden**-0.5),
        "up_proj_weight": normal((intermediate, hidden), 8.0 * hidden**-0.5),
        "down_proj_weight": normal((hidden, intermediate), 0.25 * intermediate**-0.5),
        "beta": beta,
        "linear_beta": linear_beta,
    }


def _output_tensors(kernel_input):
    tokens, hidden = kernel_input["hidden_states"].shape
    return {"output": np.zeros((tokens, hidden), dtype=ml_dtypes.bfloat16)}


def _run_kernel(
    test_manager: Orchestrator,
    platform_target: Platforms,
    collector: IMetricsCollector,
    tokens: int,
    hidden: int,
    intermediate: int,
    beta: float = _BETA,
    linear_beta: float = _LINEAR_BETA,
):
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kimi_k3_situ_mlp,
        torch_ref=torch_ref_wrapper(kimi_k3_situ_mlp_torch_ref),
        kernel_input_generator=lambda _: _generate_inputs(
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
            beta=beta,
            linear_beta=linear_beta,
        ),
        output_tensor_descriptor=_output_tensors,
        collector=collector,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
        inference_args=TKG_INFERENCE_ARGS,
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest_test_metadata(name="Kimi-K3 Exact SiTU MLP", tags=["model"])
@pytest_marks(["mlp", "decode", "prefill"])
@final
class TestKimiK3SituMlp:
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("tokens, hidden, intermediate", _EXECUTION_CASES)
    def test_matches_model_oracle(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        tokens: int,
        hidden: int,
        intermediate: int,
    ):
        _run_kernel(
            test_manager=test_manager,
            platform_target=platform_target,
            collector=collector,
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
        )

    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "beta, linear_beta",
        [
            pytest.param(2.0, _LINEAR_BETA, id="beta"),
            pytest.param(_BETA, 3.0, id="linear-beta"),
        ],
    )
    def test_runtime_parameters(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        beta: float,
        linear_beta: float,
    ):
        _run_kernel(
            test_manager=test_manager,
            platform_target=platform_target,
            collector=collector,
            tokens=4,
            hidden=512,
            intermediate=512,
            beta=beta,
            linear_beta=linear_beta,
        )

    @pytest.mark.compile_only
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("tokens, hidden, intermediate", _PRODUCTION_COMPILE_CASES)
    def test_production_shapes_compile(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        tokens: int,
        hidden: int,
        intermediate: int,
    ):
        _run_kernel(
            test_manager=test_manager,
            platform_target=platform_target,
            collector=collector,
            tokens=tokens,
            hidden=hidden,
            intermediate=intermediate,
        )


_MUTATIONS = [
    "hidden_states",
    "gate_proj_weight",
    "up_proj_weight",
    "down_proj_weight",
    "beta",
    "linear_beta",
]


@pytest.mark.parametrize("input_name", _MUTATIONS)
def test_model_oracle_responds_to_each_input(input_name: str):
    """Keep every kernel argument live in the executable comparison oracle."""
    base = _generate_inputs(tokens=4, hidden=8, intermediate=6)
    baseline = kimi_k3_situ_mlp_torch_ref(**base)["output"].float()
    changed = deepcopy(base)

    if input_name == "beta":
        changed[input_name] = 0.5
    elif input_name == "linear_beta":
        changed[input_name] = 0.75
    else:
        changed[input_name][:] = -changed[input_name]

    mutated = kimi_k3_situ_mlp_torch_ref(**changed)["output"].float()
    assert torch.count_nonzero(mutated - baseline) > 0


def test_parameters_are_required():
    """Fail closed instead of silently selecting a one-sided activation."""
    for function in (kimi_k3_situ_mlp, kimi_k3_situ_mlp_torch_ref):
        parameters = signature(function).parameters
        assert parameters["beta"].default == Parameter.empty
        assert parameters["linear_beta"].default == Parameter.empty

    inputs = _generate_inputs(tokens=1, hidden=8, intermediate=6)
    inputs.pop("linear_beta")
    with pytest.raises(TypeError):
        kimi_k3_situ_mlp_torch_ref(**inputs)


def test_hard_clamped_silu_is_not_exact_situ():
    """Quantify the semantic gap that this kernel removes."""
    generator = torch.Generator().manual_seed(7)
    gate = torch.randn(200_000, generator=generator) * 3.0
    up = torch.randn(200_000, generator=generator) * 10.0
    gate_term = _BETA * torch.tanh(gate / _BETA) * torch.sigmoid(gate)
    up_term = _LINEAR_BETA * torch.tanh(up / _LINEAR_BETA)
    exact = gate_term * up_term

    gate_clamped = torch.clamp(gate, -_BETA, _BETA)
    up_clamped = torch.clamp(up, -_LINEAR_BETA, _LINEAR_BETA)
    hard_clamped = gate_clamped * torch.sigmoid(gate_clamped) * up_clamped
    relative_rms = (hard_clamped - exact).square().mean().sqrt() / exact.square().mean().sqrt()
    assert float(relative_rms) > 0.3
