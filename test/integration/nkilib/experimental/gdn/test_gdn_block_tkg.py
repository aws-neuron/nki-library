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
"""Tests for the gdn_block_tkg fused decode megakernel."""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.gdn.gdn_block_tkg import gdn_block_tkg
from nkilib_src.nkilib.experimental.gdn.gdn_block_tkg_torch import gdn_block_decode_fused_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_inputs(batch, hidden, num_k_heads, num_v_heads, head_dim, kernel_width, dtype) -> dict:
    rng = np.random.RandomState(42)
    key_dim = num_k_heads * head_dim
    value_dim = num_v_heads * head_dim
    conv_dim = 2 * key_dim + value_dim

    def bf16(x):
        return dt.static_cast(x.astype(np.float32), dtype)

    return {
        "hidden": bf16(rng.randn(batch, 1, hidden) * 0.5),
        "W_in_qkv": bf16(rng.randn(hidden, conv_dim) * 0.05),
        "W_in_z": bf16(rng.randn(hidden, value_dim) * 0.05),
        "W_in_a": bf16(rng.randn(hidden, num_v_heads) * 0.05),
        "W_in_b": bf16(rng.randn(hidden, num_v_heads) * 0.05),
        "conv_weight": (rng.randn(conv_dim, kernel_width) * 0.3).astype(np.float32),
        "conv_state": bf16(rng.randn(batch, conv_dim, kernel_width) * 0.5),
        "A_log": (rng.randn(num_v_heads) * 0.5).astype(np.float32),
        "dt_bias": (rng.randn(num_v_heads) * 0.1).astype(np.float32),
        "norm_weight": (rng.randn(head_dim) * 0.2 + 1.0).astype(np.float32),
        "recurrent_state": (rng.randn(batch, num_v_heads, head_dim, head_dim) * 0.1).astype(np.float32),
    }


@torch_ref_wrapper
def _torch_ref(
    hidden, W_in_qkv, W_in_z, W_in_a, W_in_b, conv_weight, conv_state, A_log, dt_bias, norm_weight, recurrent_state
):
    import torch

    # as_tensor, not from_numpy: torch_ref_wrapper has already converted every numpy
    # kwarg to a Tensor, and from_numpy rejects one. as_tensor accepts both, so the ref
    # stays callable with raw numpy for direct debugging.
    def tt(x):
        return torch.as_tensor(x).float()

    # Derive head layout from tensor shapes (same as the kernel).
    conv_dim = W_in_qkv.shape[1]
    value_dim = W_in_z.shape[1]
    num_v_heads = A_log.shape[0]
    head_v_dim = value_dim // num_v_heads
    key_dim = (conv_dim - value_dim) // 2
    head_k_dim = recurrent_state.shape[2]
    num_k_heads = key_dim // head_k_dim

    core_out, new_conv_state, new_recurrent_state = gdn_block_decode_fused_torch_ref(
        tt(hidden),
        tt(W_in_qkv),
        tt(W_in_z),
        tt(W_in_a),
        tt(W_in_b),
        tt(conv_weight),
        tt(conv_state),
        tt(A_log),
        tt(dt_bias),
        tt(norm_weight),
        tt(recurrent_state),
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )
    return {
        "core_out": core_out.numpy(),
        "new_conv_state": new_conv_state.numpy(),
        "new_recurrent_state": new_recurrent_state.numpy(),
    }


@pytest_test_metadata(name="GDN block TKG fused decode")
@pytest_marks(["attention", "gdn", "tkg"])
class TestGdnBlockTkg:
    params = "batch, hidden, num_k_heads, num_v_heads, head_dim, kernel_width, dtype"
    _ABBREVS = {
        "batch": "b",
        "hidden": "h",
        "num_k_heads": "kh",
        "num_v_heads": "vh",
        "head_dim": "d",
        "kernel_width": "k",
        "dtype": "dt",
    }
    # (B, H, num_k_heads, num_v_heads, head_dim, K_win, dtype); H must be a multiple of 128.
    test_cases_basic = [
        (1, 128, 2, 4, 128, 4, nl.bfloat16),
    ]

    @pytest.mark.fast
    @pytest_parametrize(params, test_cases_basic, abbrevs=_ABBREVS)
    def test_basic(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        hidden,
        num_k_heads,
        num_v_heads,
        head_dim,
        kernel_width,
        dtype,
    ):
        key_dim = num_k_heads * head_dim
        value_dim = num_v_heads * head_dim
        conv_dim = 2 * key_dim + value_dim

        def inputs(test_config, input_tensor_def=None):
            return _generate_inputs(batch, hidden, num_k_heads, num_v_heads, head_dim, kernel_width, dtype)

        def outputs(kernel_input):
            return {
                "core_out": np.zeros((batch, 1, value_dim), dtype=np.dtype("bfloat16")),
                "new_conv_state": np.zeros((batch, conv_dim, kernel_width), dtype=np.dtype("bfloat16")),
                "new_recurrent_state": np.zeros((batch, num_v_heads, head_dim, head_dim), dtype=np.float32),
            }

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gdn_block_tkg,
            torch_ref=_torch_ref,
            kernel_input_generator=inputs,
            output_tensor_descriptor=outputs,
        ).run_test(
            test_config=None,
            # gdn_block_tkg is single-core (no LNC sharding); force grid size 1 (platform default is 2).
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            rtol=3e-2,
            atol=3e-2,
        )
