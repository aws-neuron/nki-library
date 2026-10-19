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
"""Tests for the gdn_conv1d depthwise causal conv1d kernels (prefill + decode)."""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.gdn.gdn_conv1d import gdn_conv1d_decode, gdn_conv1d_prefill
from nkilib_src.nkilib.experimental.gdn.gdn_conv1d_torch import (
    gdn_conv1d_decode_torch_ref,
    gdn_conv1d_prefill_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_prefill_inputs(batch: int, conv_dim: int, seqlen: int, kernel_width: int, dtype) -> dict:
    rng = np.random.RandomState(42)
    mixed = dt.static_cast(rng.randn(batch, conv_dim, seqlen).astype(np.float32) * 0.5, dtype)
    conv_weight = rng.randn(conv_dim, kernel_width).astype(np.float32) * 0.3
    return {"mixed": mixed, "conv_weight": conv_weight}


def _generate_decode_inputs(batch: int, conv_dim: int, kernel_width: int, dtype) -> dict:
    rng = np.random.RandomState(7)
    x = dt.static_cast(rng.randn(batch, conv_dim, 1).astype(np.float32) * 0.5, dtype)
    conv_state = dt.static_cast(rng.randn(batch, conv_dim, kernel_width).astype(np.float32) * 0.5, dtype)
    conv_weight = rng.randn(conv_dim, kernel_width).astype(np.float32) * 0.3
    return {"x": x, "conv_state": conv_state, "conv_weight": conv_weight}


@torch_ref_wrapper
def _prefill_torch_ref(mixed, conv_weight):
    import torch

    # as_tensor, not from_numpy: torch_ref_wrapper has already converted every numpy
    # kwarg to a Tensor, and from_numpy rejects one. as_tensor accepts both, so the ref
    # stays callable with raw numpy for direct debugging.
    conv_out, new_conv_state = gdn_conv1d_prefill_torch_ref(
        torch.as_tensor(mixed).float(), torch.as_tensor(conv_weight).float()
    )
    return {"conv_out": conv_out.numpy(), "new_conv_state": new_conv_state.numpy()}


@torch_ref_wrapper
def _decode_torch_ref(x, conv_state, conv_weight):
    import torch

    out, new_conv_state = gdn_conv1d_decode_torch_ref(
        torch.as_tensor(x).float(),
        torch.as_tensor(conv_state).float(),
        torch.as_tensor(conv_weight).float(),
    )
    return {"out": out.numpy(), "new_conv_state": new_conv_state.numpy()}


@pytest_test_metadata(name="GDN conv1d")
@pytest_marks(["attention", "gdn", "conv1d"])
class TestGdnConv1d:
    _PREFILL_PARAMS = "batch, conv_dim, seqlen, kernel_width, dtype"
    _DECODE_PARAMS = "batch, conv_dim, kernel_width, dtype"
    _PREFILL_ABBREVS = {"batch": "b", "conv_dim": "c", "seqlen": "s", "kernel_width": "k", "dtype": "dt"}
    _DECODE_ABBREVS = {"batch": "b", "conv_dim": "c", "kernel_width": "k", "dtype": "dt"}

    prefill_cases = [
        (1, 256, 8, 4, nl.bfloat16),
        (2, 1024, 16, 4, nl.bfloat16),
    ]
    decode_cases = [
        (1, 256, 4, nl.bfloat16),
        (2, 1024, 4, nl.bfloat16),
    ]

    @pytest.mark.fast
    @pytest_parametrize(_PREFILL_PARAMS, prefill_cases, abbrevs=_PREFILL_ABBREVS)
    def test_prefill(
        self, test_manager: Orchestrator, platform_target: Platforms, batch, conv_dim, seqlen, kernel_width, dtype
    ):
        def inputs(test_config, input_tensor_def=None):
            return _generate_prefill_inputs(batch, conv_dim, seqlen, kernel_width, dtype)

        def outputs(kernel_input):
            mixed = kernel_input["mixed"]
            return {
                "conv_out": np.zeros(mixed.shape, mixed.dtype),
                "new_conv_state": np.zeros((batch, conv_dim, kernel_width), mixed.dtype),
            }

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gdn_conv1d_prefill,
            torch_ref=_prefill_torch_ref,
            kernel_input_generator=inputs,
            output_tensor_descriptor=outputs,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=2e-2, atol=2e-2)

    @pytest.mark.fast
    @pytest_parametrize(_DECODE_PARAMS, decode_cases, abbrevs=_DECODE_ABBREVS)
    def test_decode(self, test_manager: Orchestrator, platform_target: Platforms, batch, conv_dim, kernel_width, dtype):
        def inputs(test_config, input_tensor_def=None):
            return _generate_decode_inputs(batch, conv_dim, kernel_width, dtype)

        def outputs(kernel_input):
            x = kernel_input["x"]
            return {
                "out": np.zeros(x.shape, x.dtype),
                "new_conv_state": np.zeros((batch, conv_dim, kernel_width), x.dtype),
            }

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gdn_conv1d_decode,
            torch_ref=_decode_torch_ref,
            kernel_input_generator=inputs,
            output_tensor_descriptor=outputs,
        ).run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=2e-2, atol=2e-2)
