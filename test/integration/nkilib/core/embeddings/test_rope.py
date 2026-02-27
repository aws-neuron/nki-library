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

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import nki.language as nl
import nkilib_src.nkilib.core.embeddings.rope as rope
import numpy as np
import pytest
from nkilib_src.nkilib.core.embeddings.rope_torch import rope_torch_ref

_BF16_EPS = 2**-7
_RTOL = _BF16_EPS
_ATOL = 1e-3


def filter_rope_combinations(d_head, B, n_heads, S, contiguous_layout=None, relayout_in_sbuf=None):
    """Filter invalid RoPE parameter combinations.

    Constraints:
    - d_head must be 64 or 128
    - S must be even
    - SBUF limit: B * n_heads * S <= 73728 (skip, not negative test)
    """
    if d_head not in (64, 128):
        return FilterResult.INVALID
    if S is not None and S % 2 != 0:
        return FilterResult.INVALID
    # SBUF limit - skip these configs (OOM, not kernel assertion)
    if B is not None and n_heads is not None and S is not None:
        if B * n_heads * S > 73728:
            return FilterResult.REDUNDANT
    return FilterResult.VALID


def generate_kernel_inputs(d_head, B, n_heads, S, contiguous_layout, relayout_in_sbuf):
    """Generate RoPE kernel inputs from parameters."""
    dtype = nl.bfloat16
    half_d = d_head // 2
    generate_tensor = gaussian_tensor_generator()
    return {
        "x_in": generate_tensor(name="x_in", shape=(d_head, B, n_heads, S), dtype=dtype),
        "cos": generate_tensor(name="cos", shape=(half_d, B, S), dtype=dtype),
        "sin": generate_tensor(name="sin", shape=(half_d, B, S), dtype=dtype),
        "lnc_shard": True,
        "contiguous_layout": contiguous_layout,
        "relayout_in_sbuf": relayout_in_sbuf,
    }


@pytest_test_metadata(
    name="RoPE",
    pytest_marks=["embeddings", "rope"],
)
@final
class TestRopeKernel:
    """Test class for RoPE kernel."""

    @staticmethod
    def _output_tensors(kernel_input):
        x_in = kernel_input["x_in"]
        return {"x_out": np.zeros(x_in.shape, dtype=x_in.dtype)}

    def _run_rope_test(
        self, test_manager, d_head, B, n_heads, S, contiguous_layout, relayout_in_sbuf, is_negative_test=False
    ):
        def input_generator(test_config):
            return generate_kernel_inputs(d_head, B, n_heads, S, contiguous_layout, relayout_in_sbuf)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rope.RoPE,
            torch_ref=torch_ref_wrapper(rope_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self._output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(),
            rtol=_RTOL,
            atol=_ATOL,
            is_negative_test=is_negative_test,
        )

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        d_head=[64, 128],
        B=[1, 8, 64],
        n_heads=[1, 8],
        S=[16, 128],
        contiguous_layout=[True, False],
        relayout_in_sbuf=[True, False],
        filter=filter_rope_combinations,
        coverage="singles",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_rope_fast(
        self,
        test_manager: Orchestrator,
        d_head,
        B,
        n_heads,
        S,
        contiguous_layout,
        relayout_in_sbuf,
        is_negative_test_case,
    ):
        """Fast tests with minimal coverage for quick validation."""
        self._run_rope_test(
            test_manager,
            d_head,
            B,
            n_heads,
            S,
            contiguous_layout,
            relayout_in_sbuf,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.coverage_parametrize(
        d_head=BoundedRange([64, 128], boundary_values=[32, 63, 129, 256]),
        B=[1, 3, 8, 32, 100],
        n_heads=[1, 5, 8, 16],
        S=BoundedRange([16, 50, 128, 300, 512], boundary_values=[15, 17, 51]),  # odd values
        contiguous_layout=[True, False],
        relayout_in_sbuf=[True, False],
        filter=filter_rope_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_rope_sweep(
        self,
        test_manager: Orchestrator,
        d_head,
        B,
        n_heads,
        S,
        contiguous_layout,
        relayout_in_sbuf,
        is_negative_test_case,
    ):
        """Full sweep tests with pairwise coverage."""
        self._run_rope_test(
            test_manager,
            d_head,
            B,
            n_heads,
            S,
            contiguous_layout,
            relayout_in_sbuf,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.parametrize(
        "d_head,B,n_heads,S,contiguous_layout,relayout_in_sbuf",
        [
            (128, 8192, 1, 8, True, False),
            (64, 8192, 1, 8, True, False),
            (128, 1, 1, 8192, True, False),
            (64, 1, 1, 8192, True, False),
            (128, 1, 512, 128, True, False),
            (64, 1, 512, 128, True, False),
            (128, 64, 8, 128, True, False),
            (64, 64, 8, 128, True, False),
            (128, 32, 4, 64, False, False),
            (64, 32, 4, 64, False, False),
            (128, 1, 32, 2048, True, False),
            (128, 4, 8, 1024, True, False),
            (128, 2, 128, 256, True, False),
            (64, 2, 128, 256, True, False),
        ],
    )
    def test_rope_manual(self, test_manager: Orchestrator, d_head, B, n_heads, S, contiguous_layout, relayout_in_sbuf):
        """Manual test cases for QoR tracking and deterministic pipeline runs."""
        self._run_rope_test(test_manager, d_head, B, n_heads, S, contiguous_layout, relayout_in_sbuf)
