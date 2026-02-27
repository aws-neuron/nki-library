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

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, KernelArgs, LazyGoldenGenerator, ValidationArgs
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import assert_negative_test_case
from test.utils.test_orchestrator import Orchestrator
from typing import final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.embeddings.rope_hf import rope_hf

# Constants
PMAX = 128  # nl.tile_size.pmax


def rotate_half_numpy(x):
    """Numpy implementation of rotate_half function."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return np.concatenate((-x2, x1), axis=-1)


def rotate_half_backward_numpy(x):
    """Numpy implementation of rotate_half backward function."""
    half_dim = x.shape[-1] // 2
    x1 = x[..., :half_dim]
    x2 = x[..., half_dim:]
    return np.concatenate((x2, -x1), axis=-1)


def rope_forward_numpy(q, k, cos, sin, unsqueeze_dim=1):
    """Numpy RoPE forward implementation."""
    cos = np.expand_dims(cos, axis=unsqueeze_dim).astype(q.dtype)
    sin = np.expand_dims(sin, axis=unsqueeze_dim).astype(q.dtype)
    q_embed = (q * cos) + (rotate_half_numpy(q) * sin)
    k_embed = (k * cos) + (rotate_half_numpy(k) * sin)
    return q_embed, k_embed


def rope_backward_numpy(q, k, cos, sin, unsqueeze_dim=1):
    """Numpy RoPE backward implementation."""
    cos = np.expand_dims(cos, axis=unsqueeze_dim).astype(q.dtype)
    sin = np.expand_dims(sin, axis=unsqueeze_dim).astype(q.dtype)
    q_out = q * cos + rotate_half_backward_numpy(q * sin)
    k_out = k * cos + rotate_half_backward_numpy(k * sin)
    return q_out, k_out


def generate_kernel_inputs(
    batch_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    seq_len: int,
    head_dim: int,
    dtype,
    backward: bool,
    use_packed_format: bool = False,
):
    """Generate kernel inputs for RoPE computation.

    Args:
        ...
        use_packed_format: If True, generate packed rope_cache tensor.
                           If False, generate separate cos/sin tensors.
    """
    generate_tensor = gaussian_tensor_generator()

    # Common tensors
    q = generate_tensor(name="q", shape=(batch_size, num_q_heads, seq_len, head_dim), dtype=dtype)
    k = generate_tensor(name="k", shape=(batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype)
    q_out = np.zeros((batch_size, num_q_heads, seq_len, head_dim), dtype=dtype)
    k_out = np.zeros((batch_size, num_kv_heads, seq_len, head_dim), dtype=dtype)

    result = {
        "q": q,
        "k": k,
        "q_out.must_alias_input": q_out,
        "k_out.must_alias_input": k_out,
        "backward": backward,
    }

    # Position encoding tensors - format-specific
    if use_packed_format:
        result["rope_cache"] = generate_tensor(name="rope_cache", shape=(seq_len, head_dim * 2), dtype=dtype)
    else:
        result["cos"] = generate_tensor(name="cos", shape=(batch_size, seq_len, head_dim), dtype=dtype)
        result["sin"] = generate_tensor(name="sin", shape=(batch_size, seq_len, head_dim), dtype=dtype)

    return result


def golden_rope(inp_np, backward=False):
    """Golden function for RoPE computation.

    Supports both separate cos/sin format and packed rope_cache format.
    """
    q = inp_np["q"]
    k = inp_np["k"]
    dtype = q.dtype

    # Extract cos/sin - auto-detect format based on input keys
    if "rope_cache" in inp_np:
        # Packed format: [seq_len, head_dim*2] -> cos, sin
        rope_cache = inp_np["rope_cache"]
        cos = rope_cache[..., : rope_cache.shape[-1] // 2]
        sin = rope_cache[..., rope_cache.shape[-1] // 2 :]
        # Add batch dimension if needed
        if cos.ndim == 2:
            cos = np.expand_dims(cos, axis=0)
            sin = np.expand_dims(sin, axis=0)
    else:
        # Separate format
        cos = inp_np["cos"]
        sin = inp_np["sin"]

    # Compute RoPE
    rope_fn = rope_backward_numpy if backward else rope_forward_numpy
    q_golden, k_golden = rope_fn(q, k, cos, sin, unsqueeze_dim=1)

    return {
        "q_out": dt.static_cast(q_golden, dtype),
        "k_out": dt.static_cast(k_golden, dtype),
    }


def is_valid_seq_len(seq_len: int, lnc_count: int) -> bool:
    """Check if seq_len is valid (must be multiple of PMAX * lnc_count)."""
    return seq_len % (PMAX * lnc_count) == 0


@final
@pytest_test_metadata(
    name="RoPE HF Format",
    pytest_marks=["embeddings", "rope", "hf_format"],
)
class TestRopeHFKernel:
    """Test class for rope kernel with HuggingFace format."""

    # fmt: off
    rope_hf_params = "batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, backward"
    rope_hf_test_cases = [
        # batch_size, num_q_heads, num_kv_heads, seq_len, head_dim, dtype, backward
        [1, 32, 8, 256, 128, nl.bfloat16, False],  # LNC-2 compatible, bfloat16
        [1, 32, 8, 256, 128, nl.float32, False],   # float32
        [2, 16, 4, 512, 128, nl.bfloat16, False],  # BS>1, bfloat16
        [4, 1, 1, 512, 128, nl.bfloat16, False],   # No GQA, bfloat16
        [1, 32, 8, 512, 64, nl.float32, False],    # Small head_dim (GPT-style), float32
        [1, 32, 8, 256, 128, nl.float32, True],    # Backward pass, float32
        [1, 32, 8, 64, 128, nl.float32, False],    # NEGATIVE TEST CASE: seq_len < PMAX * lnc_count
    ]
    # fmt: on

    def run_rope_hf_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        batch_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        seq_len: int,
        head_dim: int,
        dtype,
        backward: bool,
        lnc_count: int,
        use_packed_format: bool,
    ):
        """Common test runner for rope HF format."""
        is_negative_test_case = not is_valid_seq_len(seq_len, lnc_count)
        with assert_negative_test_case(is_negative_test_case):
            kernel_input = generate_kernel_inputs(
                batch_size=batch_size,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                seq_len=seq_len,
                head_dim=head_dim,
                dtype=dtype,
                backward=backward,
                use_packed_format=use_packed_format,
            )

            q, k = kernel_input["q"], kernel_input["k"]
            output_placeholder = {
                "q_out": np.zeros(q.shape, q.dtype),
                "k_out": np.zeros(k.shape, k.dtype),
            }

            def create_golden():
                return golden_rope(inp_np=kernel_input, backward=backward)

            test_manager.execute(
                KernelArgs(
                    kernel_func=rope_hf,
                    compiler_input=compiler_args,
                    kernel_input=kernel_input,
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            lazy_golden_generator=create_golden,
                            output_ndarray=output_placeholder,
                        ),
                        absolute_accuracy=1e-6,
                        relative_accuracy=1e-6,  # NOTE: currently, we cannot set rtol to 0
                    ),
                )
            )

    @pytest.mark.fast
    @pytest.mark.parametrize("lnc_count", [1, 2])
    @pytest.mark.parametrize(rope_hf_params, rope_hf_test_cases)
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
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=lnc_count)
        self.run_rope_hf_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype,
            backward=backward,
            lnc_count=compiler_args.logical_nc_config,
            use_packed_format=False,
        )

    @pytest.mark.parametrize(rope_hf_params, rope_hf_test_cases)
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
        compiler_args = CompilerArgs(enable_birsim=False, logical_nc_config=2)
        self.run_rope_hf_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            dtype=dtype,
            backward=backward,
            lnc_count=compiler_args.logical_nc_config,
            use_packed_format=True,
        )
