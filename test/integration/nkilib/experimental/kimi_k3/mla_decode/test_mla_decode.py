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

"""Integration tests for Kimi-K3 dense gated MLA decode."""

from copy import deepcopy
from typing import final

import ml_dtypes
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.kimi_k3.mla_decode import (
    kimi_k3_mla_decode,
    kimi_k3_mla_decode_torch_ref,
)

from test.utils.common_dataclasses import TKG_INFERENCE_ARGS, CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_C = 512
_R = 64
_V = 128
_H = 7168
_SCALE = 192**-0.5
_RTOL = 2**-8
_ATOL = 2**-8

_CASES = [
    pytest.param(1, 2, 128, marks=pytest.mark.fast, id="component-b1-nh2-s128"),
    pytest.param(2, 3, 1024, marks=pytest.mark.fast, id="production-heads-b2-s1024"),
]
_MAX_DENSE_PRIOR = 17152


def _generate_inputs(batch: int, heads: int, prior_len: int, gate_mode: str = "random"):
    rng = np.random.default_rng(42)

    def normal(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(ml_dtypes.bfloat16)

    positions = np.full((batch,), prior_len, dtype=np.int32)
    if batch > 1:
        positions[-1] = max(1, prior_len - 17)

    if gate_mode == "random":
        gate = rng.uniform(0.05, 0.95, size=(batch, heads * _V)).astype(ml_dtypes.bfloat16)
    elif gate_mode == "zeros":
        gate = np.zeros((batch, heads * _V), dtype=ml_dtypes.bfloat16)
    elif gate_mode == "ones":
        gate = np.ones((batch, heads * _V), dtype=ml_dtypes.bfloat16)
    else:
        raise ValueError(f"Unsupported gate_mode={gate_mode}")

    return {
        "q_nope_absorbed_T": normal(_C, batch, heads),
        "q_pe_T": normal(_R, batch, heads),
        "ckv_prior_T": normal(_C, batch, prior_len),
        "kpe_prior_T": normal(_R, batch, prior_len),
        "ckv_cur_T": normal(_C, batch, 1),
        "kpe_cur_T": normal(_R, batch, 1),
        "positions": positions,
        "out_absorb_w": normal(heads, _V, _C, scale=_C**-0.5),
        "gate": gate,
        "o_proj_w": normal(heads * _V, _H, scale=(heads * _V) ** -0.5),
        "softmax_scale": _SCALE,
        # Kimi uses dense sequential slots and has no DSA additive mask.
        "key_pos": None,
        "sparse_add": None,
    }


def _output_tensors(kernel_input):
    batch = kernel_input["q_nope_absorbed_T"].shape[1]
    return {"output": np.zeros((batch, _H), dtype=ml_dtypes.bfloat16)}


def _run_kernel(
    test_manager: Orchestrator,
    platform_target: Platforms,
    collector: IMetricsCollector,
    batch: int,
    heads: int,
    prior_len: int,
    gate_mode: str,
):
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kimi_k3_mla_decode,
        torch_ref=torch_ref_wrapper(kimi_k3_mla_decode_torch_ref),
        kernel_input_generator=lambda _: _generate_inputs(batch, heads, prior_len, gate_mode),
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


@pytest_test_metadata(name="Kimi-K3 Dense Gated MLA Decode", tags=["model"])
@pytest_marks(["attention", "decode", "mla"])
@final
class TestKimiK3MlaDecode:
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("batch, heads, prior_len", _CASES)
    def test_matches_model_oracle(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        batch: int,
        heads: int,
        prior_len: int,
    ):
        _run_kernel(
            test_manager,
            platform_target,
            collector,
            batch,
            heads,
            prior_len,
            gate_mode="random",
        )

    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize("gate_mode", ["zeros", "ones"])
    def test_gate_boundary(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        gate_mode: str,
    ):
        # Zero proves the gate is before o-proj and semantically active. One is
        # the ungated identity and must preserve the existing attention tail.
        _run_kernel(
            test_manager,
            platform_target,
            collector,
            batch=1,
            heads=3,
            prior_len=128,
            gate_mode=gate_mode,
        )

    @pytest.mark.compile_only
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_context_capacity(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
    ):
        # Trn3 LNC1 allocator measurement for B=2, Nh=3: this length has zero
        # SBUF spills; the next 128-token step (17280) spills 3072 bytes/part.
        _run_kernel(
            test_manager,
            platform_target,
            collector,
            batch=2,
            heads=3,
            prior_len=_MAX_DENSE_PRIOR,
            gate_mode="random",
        )


_MUTATIONS = [
    "q_nope_absorbed_T",
    "q_pe_T",
    "ckv_prior_T",
    "kpe_prior_T",
    "ckv_cur_T",
    "kpe_cur_T",
    "positions",
    "out_absorb_w",
    "gate",
    "o_proj_w",
    "softmax_scale",
]


@pytest.mark.parametrize("input_name", _MUTATIONS)
def test_model_oracle_responds_to_each_input(input_name: str):
    """Keep every kernel argument live in the executable comparison oracle."""
    base = _generate_inputs(batch=1, heads=3, prior_len=32)
    baseline = kimi_k3_mla_decode_torch_ref(**base)["output"].float()
    changed = deepcopy(base)

    if input_name == "positions":
        changed[input_name][0] = 1
    elif input_name == "gate":
        changed[input_name][:] = 0.0
    elif input_name == "softmax_scale":
        changed[input_name] *= 4.0
    else:
        changed[input_name][:] = -changed[input_name]

    mutated = kimi_k3_mla_decode_torch_ref(**changed)["output"].float()
    assert torch.count_nonzero(mutated - baseline) > 0
