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

"""
Tests for rope_hf kernel (HuggingFace format) implementation
with various batch sizes, head configurations, and dtypes.
"""

from test.utils.common_dataclasses import CompilerArgs
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.embeddings.rope_hf import rope_hf
from nkilib_src.nkilib.core.embeddings.rope_hf_torch import rope_hf_torch_ref

# Constants
PMAX = 128  # nl.tile_size.pmax

# fmt: off
ROPE_HF_PARAMS = \
    "batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, backward"
ROPE_HF_TEST_CASES = [
    # batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, backward
    (1, 32, 8, 256, 128, nl.bfloat16, False),  # LNC-2 compatible, bfloat16
    (1, 32, 8, 256, 128, nl.float32, False),   # float32
    (2, 16, 4, 512, 128, nl.bfloat16, False),  # BS>1, bfloat16
    (4, 1, 1, 512, 128, nl.bfloat16, False),   # No GQA, bfloat16
    (1, 32, 8, 512, 64, nl.float32, False),    # Small head_dim (GPT-style), float32
    (1, 32, 8, 256, 128, nl.float32, True),    # Backward pass, float32
    (1, 32, 8, 64, 128, nl.float32, False),    # NEGATIVE TEST CASE: seq_len < PMAX * lnc_count
]
# fmt: on

_ABBREVS = {
    "batch_size": "bs",
    "num_q_heads": "qh",
    "num_kv_heads": "kvh",
    "seq_len": "sl",
    "head_dim": "hd",
}


def _is_valid_seq_len(seq_len: int, lnc_count: int) -> bool:
    """Check if seq_len is valid (must be multiple of PMAX * lnc_count)."""
    return seq_len % (PMAX * lnc_count) == 0


def _generate_inputs(batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, backward, use_packed_format):
    """Generate kernel inputs for RoPE computation."""
    np.random.seed(42)
    q = np.random.randn(batch_size, num_q_heads, seq_len, head_dim).astype(dtype)
    k = np.random.randn(batch_size, num_kv_heads, seq_len, head_dim).astype(dtype)

    result = {
        "q": q,
        "k": k,
        "q_out.must_alias_input": np.zeros_like(q),
        "k_out.must_alias_input": np.zeros_like(k),
        "backward": backward,
    }

    if use_packed_format:
        result["rope_cache"] = np.random.randn(seq_len, head_dim * 2).astype(dtype)
    else:
        result["cos"] = np.random.randn(batch_size, seq_len, head_dim).astype(dtype)
        result["sin"] = np.random.randn(batch_size, seq_len, head_dim).astype(dtype)

    return result


@final
@pytest_test_metadata(
    name="RoPE HF Format",
    pytest_marks=["embeddings", "rope", "hf_format"],
)
class TestRopeHFKernel:
    """Test class for rope kernel with HuggingFace format."""

    @pytest.mark.fast
    @pytest.mark.parametrize("lnc_count", [1, 2])
    @pytest_parametrize(ROPE_HF_PARAMS, ROPE_HF_TEST_CASES, abbrevs=_ABBREVS)
    def test_rope_hf_lnc(
        self,
        test_manager: Orchestrator,
        batch_size,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        dtype,
        backward,
        lnc_count,
    ):
        """Test rope kernel with separate cos/sin format."""
        is_negative = not _is_valid_seq_len(seq_len, lnc_count)

        def input_generator(test_config):
            return _generate_inputs(
                batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, backward, use_packed_format=False
            )

        def output_tensors(kernel_input):
            return {
                "q_out": np.zeros_like(kernel_input["q"]),
                "k_out": np.zeros_like(kernel_input["k"]),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rope_hf,
            torch_ref=torch_ref_wrapper(rope_hf_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count),
            rtol=1e-6,
            atol=1e-6,
            is_negative_test=is_negative,
        )

    @pytest_parametrize(ROPE_HF_PARAMS, ROPE_HF_TEST_CASES, abbrevs=_ABBREVS)
    def test_rope_hf_cache(
        self,
        test_manager: Orchestrator,
        batch_size,
        num_q_heads,
        num_kv_heads,
        seq_len,
        head_dim,
        dtype,
        backward,
    ):
        """Test rope kernel with packed cos/sin format."""
        lnc_count = 2
        is_negative = not _is_valid_seq_len(seq_len, lnc_count)

        def input_generator(test_config):
            return _generate_inputs(
                batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, backward, use_packed_format=True
            )

        def output_tensors(kernel_input):
            return {
                "q_out": np.zeros_like(kernel_input["q"]),
                "k_out": np.zeros_like(kernel_input["k"]),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rope_hf,
            torch_ref=torch_ref_wrapper(rope_hf_torch_ref, preserve_lower_precision=True),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count),
            rtol=1e-6,
            atol=1e-6,
            is_negative_test=is_negative,
        )
