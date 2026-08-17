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

"""Tests for permute_routed_tokens subkernel using UnitTestFramework."""

from typing import final

import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental.subkernels.permute_routed_tokens import (
    _SUPPORTED_HIDDEN_DTYPES,
    permute_routed_tokens,
)
from nkilib_src.nkilib.experimental.subkernels.permute_routed_tokens_torch import permute_routed_tokens_torch_ref

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Dtype converters: preserve fp8 and bf16 dtypes through the torch ref
_NP_TO_TORCH_FP8 = {
    'float8_e4m3': torch.float8_e4m3fn,  # not supported in torch, use fp8_e4m3fn instead for torch ref
    'float8_e4m3fn': torch.float8_e4m3fn,
    'float8_e5m2': torch.float8_e5m2,
}


def _input_dtype_converter(value):
    dtype_str = str(value.dtype)
    torch_dtype = _NP_TO_TORCH_FP8.get(dtype_str)
    if torch_dtype is not None:
        return torch.from_numpy(value.astype(np.float32)).to(torch_dtype)
    if 'bfloat16' in dtype_str:
        return torch.from_numpy(value.astype(np.float32)).to(torch.bfloat16)
    return None


def _make_output_dtype_converter(target_np_dtype_str):
    """Create an output converter that maps torch fp8 tensors back to the given numpy fp8 dtype."""

    def _output_dtype_converter(tensor):
        if tensor.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Use uint8 view to preserve raw bytes when converting back to numpy.
            # This avoids lossy f8→f32→f8 round-trip for bitcast affinities / token indices.
            return tensor.view(torch.uint8).numpy().view(target_np_dtype_str)
        return None

    return _output_dtype_converter


def _run_test(
    test_manager: Orchestrator,
    platform_target: Platforms,
    lnc_degree: int,
    T: int,
    H: int,
    K: int,
    E: int,
    hidden_dtype: str,
):
    np.random.seed(42)

    assert hidden_dtype in _SUPPORTED_HIDDEN_DTYPES, (
        f"Kernel only supports hidden_dtype in {_SUPPORTED_HIDDEN_DTYPES} but got {hidden_dtype=}"
    )

    # I/O shapes based on hidden_dtype
    input_cols = H if hidden_dtype == nl.bfloat16 else H + H // 4
    output_cols = H + 3 if hidden_dtype == nl.bfloat16 else H + H // 4 + 6

    # Build inputs
    hidden_input = np.random.randn(T, input_cols).astype(hidden_dtype)
    expert_index = np.stack([np.random.choice(E, size=K, replace=False) for _ in range(T)]).astype(np.int32)
    random_matrix = np.random.rand(T, K)
    normalized_matrix = random_matrix / random_matrix.sum(axis=1, keepdims=True)
    expert_affinities_masked = np.zeros((T, E), dtype=nl.bfloat16)
    for token_idx in range(T):
        for topk_idx in range(K):
            expert_id = expert_index[token_idx, topk_idx]
            expert_affinities_masked[token_idx, expert_id] = normalized_matrix[token_idx, topk_idx]

    def input_generator(test_config):
        return {
            "hidden_input": hidden_input,
            "expert_index": expert_index,
            "expert_affinities_masked": expert_affinities_masked,
        }

    def output_tensors(kernel_input):
        return {"out": np.zeros((T * K, output_cols), dtype=hidden_dtype)}

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=permute_routed_tokens,
        torch_ref=torch_ref_wrapper(
            permute_routed_tokens_torch_ref,
            preserve_lower_precision=True,
            input_dtype_converter=_input_dtype_converter,
            output_dtype_converter=_make_output_dtype_converter(str(hidden_dtype)),
        ),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=output_tensors,
    )

    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=lnc_degree),
        inference_args=InferenceArgs(enable_determinism_check=True, num_runs=10),
        rtol=0,
        atol=0,
        # nan and inf are considered "passing" so long as they match between output and golden.
        # When we reinterpret bf16/int32 to a smaller dtype, it is possible for nan/inf to get created
        # in the smaller dtype. For example, [127] in int32 is [0, 0, 0, nan] when reinterpreted to fp8_e4m3fn.
        equal_nan_inf=True,
    )


# fmt: off
PERMUTE_ROUTED_TOKENS_PARAM_NAMES = "lnc_degree, T, H, K, E, hidden_dtype"
# hidden_dtype is a dtype name such as nl.bfloat16, which the framework defines as a string.

PERMUTE_ROUTED_TOKENS_BF16_E5M2_PARAMS = [
    # bf16 hidden states
    (2, 8,  1024, 1, 16, nl.bfloat16),
    (2, 16, 1024, 1, 8,  nl.bfloat16),
    (2, 8,  1024, 2, 16, nl.bfloat16),
    (2, 16, 1024, 2, 8,  nl.bfloat16),
    (2, 8,  1024, 4, 16, nl.bfloat16),
    (2, 16, 1024, 4, 32, nl.bfloat16),
    (2, 32, 1024, 4, 16, nl.bfloat16),
    (2, 1,  5120, 8, 256, nl.bfloat16),
    # fp8 non-FN hidden states (supported on Trn2 and Trn3)
    (2, 8,  1024, 1, 16, nl.float8_e4m3),
    (2, 32, 1024, 1, 16, nl.float8_e4m3),
    (2, 8,  1024, 1, 16, nl.float8_e5m2),
    (2, 32, 1024, 1, 16, nl.float8_e5m2),
]

PERMUTE_ROUTED_TOKENS_E4M3FN_PARAMS = [
    # fp8 e4m3fn hidden states (Trn3 only)
    (2, 16, 1024, 1, 8,  nl.float8_e4m3fn),
    (2, 8,  1024, 2, 16, nl.float8_e4m3fn),
    (2, 16, 1024, 2, 8,  nl.float8_e4m3fn),
    (2, 8,  1024, 4, 16, nl.float8_e4m3fn),
    (2, 16, 1024, 4, 32, nl.float8_e4m3fn),
    (2, 32, 1024, 4, 16, nl.float8_e4m3fn),
    (2, 1,  5120, 8, 256, nl.float8_e4m3fn),
]
# fmt: on


@pytest_test_metadata(
    name="PermuteRoutedTokens",
    pytest_marks=["permute_routed_tokens", "subkernels"],
)
@final
class TestPermuteRoutedTokensKernel:
    """Test class for permute_routed_tokens subkernel (bf16 and fp8_e5m2)."""

    @pytest.mark.fast
    @pytest.mark.parametrize(PERMUTE_ROUTED_TOKENS_PARAM_NAMES, PERMUTE_ROUTED_TOKENS_BF16_E5M2_PARAMS)
    def test_permute_routed_tokens(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree: int,
        T: int,
        H: int,
        K: int,
        E: int,
        hidden_dtype: str,
    ) -> None:
        _run_test(test_manager, platform_target, lnc_degree, T, H, K, E, hidden_dtype)


@pytest_marks(["permute_routed_tokens", "subkernels"])
@final
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
class TestPermuteRoutedTokensFP8E4M3Kernel:
    """Test class for permute_routed_tokens subkernel (fp8_e4m3fn, Trn3+ only)."""

    @pytest.mark.fast
    @pytest.mark.parametrize(PERMUTE_ROUTED_TOKENS_PARAM_NAMES, PERMUTE_ROUTED_TOKENS_E4M3FN_PARAMS)
    def test_permute_routed_tokens_fp8_e4m3fn(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree: int,
        T: int,
        H: int,
        K: int,
        E: int,
        hidden_dtype: str,
    ) -> None:
        _run_test(test_manager, platform_target, lnc_degree, T, H, K, E, hidden_dtype)
