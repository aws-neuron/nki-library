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
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult, assert_negative_test_case
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import final

import neuronxcc.nki.language as nl
import nkilib_src.nkilib.core.embeddings.rope as rope
import numpy as np
import pytest

# BF16 tolerance: 7 mantissa bits → ~1% relative precision
_RTOL = 0.01
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


def rope_single_head(x_in, cos, sin, contiguous_layout):
    """Apply RoPE to single head: [d_head, S]."""
    d_head, S = x_in.shape
    x = x_in.transpose(1, 0)  # [d_head, S] -> [S, d_head]

    # Convert contiguous layout to interleaved if needed
    if contiguous_layout:
        new_x = np.empty_like(x)
        new_x[:, ::2] = x[:, : d_head // 2]  # even positions <- first half
        new_x[:, 1::2] = x[:, d_head // 2 :]  # odd positions <- second half
        x = new_x

    # Prepare frequencies
    freqs_cos = cos.transpose(1, 0)  # [half_d, S] -> [S, half_d]
    freqs_sin = sin.transpose(1, 0)

    # Split into real/imaginary pairs: [S, d_head] -> [S, d_head//2, 2]
    xri = x.reshape(x.shape[:-1] + (-1, 2))
    x_r, x_i = xri[..., 0], xri[..., 1]  # [S, d_head//2] each

    # Apply RoPE rotation
    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    # Recombine: [S, d_head//2, 2] -> [S, d_head]
    x_out = np.stack([x_out_r, x_out_i], axis=-1).reshape(x.shape)

    # Convert interleaved back to contiguous if needed
    if contiguous_layout:
        x_out = np.concatenate((x_out[:, 0::2], x_out[:, 1::2]), axis=1)

    return x_out.transpose(1, 0)  # [S, d_head] -> [d_head, S]


def golden_rope_np(kernel_input, contiguous_layout):
    """NumPy reference implementation of RoPE."""
    x_in, cos, sin = kernel_input["x_in"], kernel_input["cos"], kernel_input["sin"]
    d_head, B, n_heads, S = x_in.shape

    # Apply to all batch and head dimensions
    x_out = np.empty_like(x_in)
    for b in range(B):
        for h in range(n_heads):
            x_out[:, b, h, :] = rope_single_head(x_in[:, b, h, :], cos[:, b, :], sin[:, b, :], contiguous_layout)

    return {"x_out": nl.static_cast(x_out, x_in.dtype)}


@pytest_test_metadata(
    name="RoPE",
    pytest_marks=["embeddings", "rope"],
)
@final
class TestRopeKernel:
    """Test class for RoPE kernel."""

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
        with assert_negative_test_case(is_negative_test_case):
            kernel_input = generate_kernel_inputs(d_head, B, n_heads, S, contiguous_layout, relayout_in_sbuf)
            x_in = kernel_input["x_in"]

            def create_golden():
                return golden_rope_np(kernel_input, contiguous_layout)

            test_manager.execute(
                KernelArgs(
                    kernel_func=rope.RoPE,
                    compiler_input=CompilerArgs(enable_birsim=True),
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=create_golden,
                            output_ndarray={"x_out": np.zeros(x_in.shape, dtype=x_in.dtype)},
                        ),
                        absolute_accuracy=_ATOL,
                        relative_accuracy=_RTOL,
                    ),
                )
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
        with assert_negative_test_case(is_negative_test_case):
            kernel_input = generate_kernel_inputs(d_head, B, n_heads, S, contiguous_layout, relayout_in_sbuf)
            x_in = kernel_input["x_in"]

            def create_golden():
                return golden_rope_np(kernel_input, contiguous_layout)

            # Disable birsim for negative tests (golden would crash on invalid params)
            test_manager.execute(
                KernelArgs(
                    kernel_func=rope.RoPE,
                    compiler_input=CompilerArgs(enable_birsim=not is_negative_test_case),
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=create_golden,
                            output_ndarray={"x_out": np.zeros(x_in.shape, dtype=x_in.dtype)},
                        ),
                        absolute_accuracy=_ATOL,
                        relative_accuracy=_RTOL,
                    ),
                )
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
        kernel_input = generate_kernel_inputs(d_head, B, n_heads, S, contiguous_layout, relayout_in_sbuf)
        x_in = kernel_input["x_in"]

        def create_golden():
            return golden_rope_np(kernel_input, contiguous_layout)

        test_manager.execute(
            KernelArgs(
                kernel_func=rope.RoPE,
                compiler_input=CompilerArgs(enable_birsim=True),
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_golden,
                        output_ndarray={"x_out": np.zeros(x_in.shape, dtype=x_in.dtype)},
                    ),
                    absolute_accuracy=_ATOL,
                    relative_accuracy=_RTOL,
                ),
            )
        )
