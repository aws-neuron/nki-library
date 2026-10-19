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

"""Integration tests for the latent-MoE MXFP4 prefill kernel."""

from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
)
from nkilib_src.nkilib.experimental.kimi_k3.latent_moe_mx_prefill import (
    latent_moe_mx_prefill,
    latent_moe_mx_prefill_torch_ref,
)

from test.integration.nkilib.core.moe.moe_tkg.test_moe_tkg_utils import build_moe_tkg
from test.utils.common_dataclasses import TKG_INFERENCE_ARGS, CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_BETA = 4.0
_LINEAR_BETA = 25.0

# The small aligned shape is the simulator/fast canary. The production shapes
# exercise Kimi-K3's D=3584 tail and both expected local-expert counts.
_TEST_CASES = [
    pytest.param(4, 512, 512, 3, 2, marks=pytest.mark.fast, id="component-aligned"),
    pytest.param(4, 3584, 3072, 7, 4, id="production-e7"),
    pytest.param(4, 3584, 3072, 16, 8, id="production-e16"),
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

    # Model the case where every selected expert is remote for one token. The
    # local contribution for this row must remain exactly zero.
    local_affinities = raw["expert_affinities"].copy()
    local_affinities[0, :] = 0

    return {
        # build_moe_tkg emits the legacy TKG H shuffle. This standalone kernel
        # consumes Kimi's natural latent order directly.
        "latent": (
            raw["hidden_input"].reshape(tokens, 4, hidden // 4).transpose(0, 2, 1).reshape(tokens, hidden).copy()
        ),
        "local_affinities": local_affinities,
        "expert_gate_up_weights": raw["expert_gate_up_weights"],
        "expert_down_weights": raw["expert_down_weights"],
        "expert_gate_up_weights_scale": raw["expert_gate_up_weights_scale"],
        "expert_down_weights_scale": raw["expert_down_weights_scale"],
        "beta": _BETA,
        "linear_beta": _LINEAR_BETA,
    }


@pytest_test_metadata(name="Latent MoE MXFP4 Prefill", tags=["model"])
@pytest_marks(["moe", "prefill", "mx"])
@final
class TestLatentMoeMxPrefill:
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("tokens, hidden, intermediate, experts, top_k", _TEST_CASES)
    def test_latent_moe_mx_prefill(
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
            kernel_entry=latent_moe_mx_prefill,
            torch_ref=torch_ref_wrapper(latent_moe_mx_prefill_torch_ref),
            kernel_input_generator=lambda _: _generate_inputs(tokens, hidden, intermediate, experts, top_k),
            output_tensor_descriptor=lambda _: {"out": np.zeros((tokens, hidden), dtype=nl.bfloat16)},
            collector=collector,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=1, platform_target=platform_target),
            rtol=2**-8,
            atol=2**-8,
            inference_args=TKG_INFERENCE_ARGS,
        )
