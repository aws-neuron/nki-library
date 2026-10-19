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

"""Integration tests for Kimi-K3 dense causal MLA prefill."""

from copy import deepcopy
from typing import final

import ml_dtypes
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.kimi_k3.mla_prefill import (
    kimi_k3_mla_prefill,
    kimi_k3_mla_prefill_torch_ref,
)

from test.utils.common_dataclasses import TKG_INFERENCE_ARGS, CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_L = 512
_R = 64
_SCALE = 192**-0.5
_RTOL = 2**-8
_ATOL = 2**-8

_EXECUTION_CASES = [
    pytest.param(128, 128, 3, None, marks=pytest.mark.fast, id="h3-component"),
    pytest.param(128, 256, 3, 64, marks=pytest.mark.fast, id="h3-q-offset"),
    pytest.param(256, 256, 16, None, marks=pytest.mark.fast, id="h16-control"),
]


def _generate_inputs(
    seqlen: int,
    kv_seqlen: int,
    heads: int,
    q_pos_offset: int | None,
):
    rng = np.random.default_rng(42)

    def normal(shape):
        return (rng.standard_normal(shape) * 0.05).astype(ml_dtypes.bfloat16)

    inputs = {
        "q_lift_hbm": normal((1, seqlen, heads, _L)),
        "q_pe_hbm": normal((1, seqlen, heads, _R)),
        "c_kv_hbm": normal((1, kv_seqlen, _L)),
        "k_pe_hbm": normal((1, kv_seqlen, _R)),
        "softmax_scale": _SCALE,
        "topk_indices_hbm": None,
        "dense": True,
    }
    if q_pos_offset is not None:
        inputs["q_pos_offset_hbm"] = np.array(
            [[q_pos_offset]],
            dtype=np.int32,
        )
    return inputs


def _output_tensors(kernel_input):
    batch, seqlen, heads, latent_width = kernel_input["q_lift_hbm"].shape
    return {
        "out_attn": np.zeros(
            (batch, seqlen, heads * latent_width),
            dtype=ml_dtypes.bfloat16,
        )
    }


def _run_kernel(
    test_manager: Orchestrator,
    platform_target: Platforms,
    collector: IMetricsCollector,
    seqlen: int,
    kv_seqlen: int,
    heads: int,
    q_pos_offset: int | None,
):
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kimi_k3_mla_prefill,
        torch_ref=torch_ref_wrapper(kimi_k3_mla_prefill_torch_ref),
        kernel_input_generator=lambda _: _generate_inputs(
            seqlen=seqlen,
            kv_seqlen=kv_seqlen,
            heads=heads,
            q_pos_offset=q_pos_offset,
        ),
        output_tensor_descriptor=_output_tensors,
        collector=collector,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(
            logical_nc_config=1,
            platform_target=platform_target,
        ),
        inference_args=TKG_INFERENCE_ARGS,
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest_test_metadata(name="Kimi-K3 Dense Causal MLA Prefill", tags=["model"])
@pytest_marks(["attention", "mla", "prefill"])
@final
class TestKimiK3MlaPrefill:
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "seqlen, kv_seqlen, heads, q_pos_offset",
        _EXECUTION_CASES,
    )
    def test_matches_model_oracle(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
        seqlen: int,
        kv_seqlen: int,
        heads: int,
        q_pos_offset: int | None,
    ):
        _run_kernel(
            test_manager=test_manager,
            platform_target=platform_target,
            collector=collector,
            seqlen=seqlen,
            kv_seqlen=kv_seqlen,
            heads=heads,
            q_pos_offset=q_pos_offset,
        )

    @pytest.mark.compile_only
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_max_sequence_compile(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        collector: IMetricsCollector,
    ):
        _run_kernel(
            test_manager=test_manager,
            platform_target=platform_target,
            collector=collector,
            seqlen=2048,
            kv_seqlen=2048,
            heads=3,
            q_pos_offset=None,
        )


_MUTATIONS = [
    "q_lift_hbm",
    "q_pe_hbm",
    "c_kv_hbm",
    "k_pe_hbm",
    "softmax_scale",
    "q_pos_offset_hbm",
]


@pytest.mark.parametrize("input_name", _MUTATIONS)
def test_model_oracle_responds_to_each_input(input_name: str):
    """Keep every semantic kernel input live in the executable oracle."""
    base = _generate_inputs(
        seqlen=8,
        kv_seqlen=16,
        heads=3,
        q_pos_offset=4,
    )
    baseline = kimi_k3_mla_prefill_torch_ref(**base)["out_attn"].float()
    changed = deepcopy(base)

    if input_name == "softmax_scale":
        changed[input_name] *= 4.0
    elif input_name == "q_pos_offset_hbm":
        changed[input_name][0, 0] = 0
    else:
        changed[input_name][:] = -changed[input_name]

    mutated = kimi_k3_mla_prefill_torch_ref(**changed)["out_attn"].float()
    assert torch.count_nonzero(mutated - baseline) > 0


def test_future_latent_does_not_move_query_zero():
    """A future latent row must not affect query zero under the causal mask."""
    inputs = _generate_inputs(
        seqlen=8,
        kv_seqlen=16,
        heads=3,
        q_pos_offset=0,
    )
    baseline = kimi_k3_mla_prefill_torch_ref(**inputs)["out_attn"].float()
    changed = deepcopy(inputs)
    changed["c_kv_hbm"][:, 6, :] = (changed["c_kv_hbm"][:, 6, :].astype(np.float32) + 10.0).astype(ml_dtypes.bfloat16)
    mutated = kimi_k3_mla_prefill_torch_ref(**changed)["out_attn"].float()

    assert torch.equal(mutated[:, 0, :], baseline[:, 0, :])
    assert torch.count_nonzero(mutated[:, -1, :] - baseline[:, -1, :]) > 0


def test_dense_ignores_topk_indices():
    """Dense mode must not reinterpret an omitted index tensor as an empty gather."""
    inputs = _generate_inputs(
        seqlen=8,
        kv_seqlen=16,
        heads=3,
        q_pos_offset=0,
    )
    baseline = kimi_k3_mla_prefill_torch_ref(**inputs)["out_attn"]
    inputs["topk_indices_hbm"] = np.full(
        (1, 8, 7),
        2**30,
        dtype=np.int32,
    )
    with_junk_indices = kimi_k3_mla_prefill_torch_ref(**inputs)["out_attn"]
    assert torch.equal(with_junk_indices, baseline)


def test_sparse_mode_fails_closed():
    """Kimi prefill is dense-only and must reject a sparse-mode request."""
    inputs = _generate_inputs(
        seqlen=8,
        kv_seqlen=16,
        heads=3,
        q_pos_offset=0,
    )
    inputs["dense"] = False
    with pytest.raises(ValueError, match="dense=True"):
        kimi_k3_mla_prefill_torch_ref(**inputs)


def test_output_is_natural_head_major_order():
    """Flattened columns must map exactly to h*L+l, without MX 4-pack swizzle."""
    inputs = _generate_inputs(
        seqlen=8,
        kv_seqlen=16,
        heads=3,
        q_pos_offset=0,
    )
    output = kimi_k3_mla_prefill_torch_ref(**inputs)["out_attn"]
    restored = output.reshape(1, 8, 3, _L)

    q_lift = torch.from_numpy(np.asarray(inputs["q_lift_hbm"], dtype=np.float32)).to(torch.bfloat16)
    q_pe = torch.from_numpy(np.asarray(inputs["q_pe_hbm"], dtype=np.float32)).to(torch.bfloat16)
    c_kv = torch.from_numpy(np.asarray(inputs["c_kv_hbm"], dtype=np.float32)).to(torch.bfloat16)
    k_pe = torch.from_numpy(np.asarray(inputs["k_pe_hbm"], dtype=np.float32)).to(torch.bfloat16)
    scores = torch.matmul(
        q_pe.permute(0, 2, 1, 3),
        k_pe.unsqueeze(1).transpose(2, 3),
    )
    scores = scores + torch.einsum(
        "bhqc,bsc->bhqs",
        q_lift.permute(0, 2, 1, 3),
        c_kv,
    )
    scores = scores * _SCALE
    causal = torch.tril(torch.ones(8, 16, dtype=torch.bool))
    scores = torch.where(
        causal[None, None, :, :],
        scores,
        torch.finfo(scores.dtype).min,
    )
    probabilities = torch.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    expected = torch.einsum("bhqs,bsc->bhqc", probabilities, c_kv).permute(
        0,
        2,
        1,
        3,
    )
    assert torch.equal(restored, expected)
