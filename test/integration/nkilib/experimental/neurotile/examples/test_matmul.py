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
"""Integration tests for tutorials in test/docs/neurotile/examples/02_matmul/."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples._02_matmul import (
    _01_matmul_patterns as patterns_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._02_matmul import (
    _01_matmul_patterns_torch as patterns_refs,
)
from nkilib_src.nkilib.experimental.neurotile.examples._02_matmul import (
    _02_matmul_coalesced as coalesced_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples._02_matmul import (
    _02_matmul_coalesced_torch as coalesced_refs,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Patterns tutorial uses M=256, K=256, N=512 (lhsT shape [K,M] = [256,256], rhs [K,N] = [256,512]).
_PATTERNS_M, _PATTERNS_K, _PATTERNS_N = 256, 256, 512


def _patterns_inputs(_):
    np.random.seed(42)
    return {
        "lhsT_hbm": np.random.rand(_PATTERNS_K, _PATTERNS_M).astype(ml_dtypes.bfloat16),
        "rhs_hbm": np.random.rand(_PATTERNS_K, _PATTERNS_N).astype(ml_dtypes.bfloat16),
    }


def _patterns_outputs(_kernel_input):
    return {"out": np.zeros((_PATTERNS_M, _PATTERNS_N), dtype=ml_dtypes.bfloat16)}


# Coalesced tutorial uses M=512, K=512, N=1024.
_COAL_M, _COAL_K, _COAL_N = 512, 512, 1024


def _coalesced_inputs(_):
    np.random.seed(42)
    return {
        "lhsT": np.random.rand(_COAL_K, _COAL_M).astype(ml_dtypes.bfloat16),
        "rhs": np.random.rand(_COAL_K, _COAL_N).astype(ml_dtypes.bfloat16),
    }


def _coalesced_outputs(_kernel_input):
    return {"out": np.zeros((_COAL_M, _COAL_N), dtype=ml_dtypes.bfloat16)}


@pytest_marks(["neurotile"])
class TestNeurotileMatmulPatterns:
    """Tutorials in 01_matmul_patterns.py — five matmul tile-iteration variants."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel,ref",
        [
            (patterns_mod.matmul_baseline, patterns_refs.matmul_baseline_torch_ref),
            (patterns_mod.matmul_hoist_lhs, patterns_refs.matmul_hoist_lhs_torch_ref),
            (patterns_mod.matmul_hoist_rhs, patterns_refs.matmul_hoist_rhs_torch_ref),
            (patterns_mod.matmul_hoist_both, patterns_refs.matmul_hoist_both_torch_ref),
            (patterns_mod.matmul_hoist_stream, patterns_refs.matmul_hoist_stream_torch_ref),
        ],
    )
    def test_pattern(self, test_manager: Orchestrator, platform_target: Platforms, kernel, ref):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel,
            torch_ref=torch_ref_wrapper(ref),
            kernel_input_generator=_patterns_inputs,
            output_tensor_descriptor=_patterns_outputs,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-2,
            atol=1e-2,
        )


@pytest_marks(["neurotile"])
class TestNeurotileMatmulCoalesced:
    """Tutorial in 02_matmul_coalesced.py — block-coalesced matmul."""

    @pytest.mark.fast
    def test_matmul_coalesced(self, test_manager, platform_target):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=coalesced_mod.matmul_coalesced,
            torch_ref=torch_ref_wrapper(coalesced_refs.matmul_coalesced_torch_ref),
            kernel_input_generator=_coalesced_inputs,
            output_tensor_descriptor=_coalesced_outputs,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-2,
            atol=1e-2,
        )


# K-streamed coalesced variant: block span is TILES_IN_BLOCK_M*128 on M and
# TILES_IN_BLOCK_N*512 on N. The "remainder" shape (M=384, N=1792) is not a
# multiple of either span, so the trailing M-block holds one 128-row tile and
# the trailing N-block holds one 512 tile + a 256 partial.
_STREAMED_BLOCK = {"TILES_IN_BLOCK_M": 2, "TILES_IN_BLOCK_N": 2, "TILES_IN_BLOCK_K": 4}
_STREAMED_SHAPES = {
    "aligned": (512, 1024, 2048),
    "remainder": (384, 1024, 1792),
}


def _streamed_input_generator(shape):
    def _gen(_):
        m, k, n = shape
        np.random.seed(42)
        return {
            "lhsT": np.random.rand(k, m).astype(ml_dtypes.bfloat16),
            "rhs": np.random.rand(k, n).astype(ml_dtypes.bfloat16),
            **_STREAMED_BLOCK,
        }

    return _gen


def _streamed_output_descriptor(shape):
    def _desc(_kernel_input):
        m, _k, n = shape
        return {"out": np.zeros((m, n), dtype=ml_dtypes.bfloat16)}

    return _desc


@pytest_marks(["neurotile"])
class TestNeurotileMatmulCoalescedStreamed:
    """K-streamed coalesced matmul (matmul_coalesced_streamed), block-aligned
    and remainder (non-block-aligned M and N) shapes."""

    @pytest.mark.fast
    @pytest.mark.parametrize("variant", ["aligned", "remainder"])
    def test_matmul_coalesced_streamed(self, test_manager, platform_target, variant):
        shape = _STREAMED_SHAPES[variant]
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=coalesced_mod.matmul_coalesced_streamed,
            torch_ref=torch_ref_wrapper(coalesced_refs.matmul_coalesced_torch_ref),
            kernel_input_generator=_streamed_input_generator(shape),
            output_tensor_descriptor=_streamed_output_descriptor(shape),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=2.5,
        )
