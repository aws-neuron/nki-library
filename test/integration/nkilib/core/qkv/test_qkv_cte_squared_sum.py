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

"""Integration tests for the optional squared-sum outputs of qkv_cte.

qkv_cte computes per-segment sum-of-squares (q/k/v_squared_sum_out, each [B, S, 1])
and writes them in place into caller-provided HBM buffers; it does not return them
(matching how the LTX-2 megakernel consumes them as internal buffers). To validate
the values directly, a thin @nki.jit wrapper kernel returns the in-place buffers so
the test framework can compare them against the torch reference. The wrapper and the
reference share an identical signature (required by the framework's signature-parity
check).
"""

from typing import final

import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.qkv.qkv_cte import qkv_cte
from nkilib_src.nkilib.core.qkv.qkv_cte_torch import qkv_cte_torch_ref
from nkilib_src.nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType

from test.integration.nkilib.core.qkv.test_qkv_common import build_qkv_input, rope_gaussian_tensor_generator
from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Keys taken from build_qkv_input and forwarded to the wrapper kernel / reference.
_FORWARDED_KEYS = (
    "input",
    "fused_qkv_weights",
    "output_layout",
    "bias",
    "fused_norm_type",
    "gamma_norm_weights",
    "norm_eps",
    "fused_rope",
    "cos_cache",
    "sin_cache",
    "d_head",
    "num_q_heads",
    "num_kv_heads",
)


@nki.jit
def _qkv_cte_squared_sum_wrapper_kernel(
    input,
    fused_qkv_weights,
    q_squared_sum_out,
    k_squared_sum_out,
    v_squared_sum_out,
    output_layout=QKVOutputLayout.BSD,
    bias=None,
    fused_norm_type=NormType.NO_NORM,
    gamma_norm_weights=None,
    norm_eps=1e-6,
    fused_rope=False,
    cos_cache=None,
    sin_cache=None,
    d_head=None,
    num_q_heads=None,
    num_kv_heads=None,
):
    """Call qkv_cte and return the in-place squared-sum buffers so they are
    observable as kernel outputs for numeric validation."""
    out = qkv_cte(
        input=input,
        fused_qkv_weights=fused_qkv_weights,
        output_layout=output_layout,
        bias=bias,
        fused_norm_type=fused_norm_type,
        gamma_norm_weights=gamma_norm_weights,
        norm_eps=norm_eps,
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        use_auto_allocation=True,
        q_squared_sum_out=q_squared_sum_out,
        k_squared_sum_out=k_squared_sum_out,
        v_squared_sum_out=v_squared_sum_out,
    )
    return out, q_squared_sum_out, k_squared_sum_out, v_squared_sum_out


def _qkv_cte_squared_sum_wrapper_ref(
    input,
    fused_qkv_weights,
    q_squared_sum_out,
    k_squared_sum_out,
    v_squared_sum_out,
    output_layout=QKVOutputLayout.BSD,
    bias=None,
    fused_norm_type=NormType.NO_NORM,
    gamma_norm_weights=None,
    norm_eps=1e-6,
    fused_rope=False,
    cos_cache=None,
    sin_cache=None,
    d_head=None,
    num_q_heads=None,
    num_kv_heads=None,
):
    """Torch reference matching _qkv_cte_squared_sum_wrapper_kernel's signature."""
    result = qkv_cte_torch_ref(
        input=input,
        fused_qkv_weights=fused_qkv_weights,
        output_layout=output_layout,
        bias=bias,
        fused_norm_type=fused_norm_type,
        gamma_norm_weights=gamma_norm_weights,
        norm_eps=norm_eps,
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_squared_sum_out=q_squared_sum_out,
        k_squared_sum_out=k_squared_sum_out,
        v_squared_sum_out=v_squared_sum_out,
    )
    return {
        "out": result["out"],
        "q_squared_sum_out": result["q_squared_sum_out"],
        "k_squared_sum_out": result["k_squared_sum_out"],
        "v_squared_sum_out": result["v_squared_sum_out"],
    }


@pytest_test_metadata(name="QKV CTE Squared Sum", tags=["qkv", "cte", "squared_sum"])
@pytest_marks(["qkv", "cte"])
@final
class TestQkvCteSquaredSum:
    """Numerically validates qkv_cte's optional q/k/v_squared_sum_out outputs."""

    # fmt: off
    squared_sum_test_params = \
        "vnc_degree, batch, seqlen, n_q_heads, n_kv_heads, d_head, norm_type, fused_rope, output_layout"
    squared_sum_test_perms = [
        pytest.param(1, 1, 128, 8, 1, 128, NormType.NO_NORM, False, QKVOutputLayout.BSD, marks=pytest.mark.fast),
        [1, 1, 128, 8, 1, 128, NormType.RMS_NORM, False, QKVOutputLayout.BSD],
        [1, 2, 256, 4, 2, 64,  NormType.NO_NORM,  False, QKVOutputLayout.BSD],
        [2, 1, 128, 8, 1, 128, NormType.NO_NORM,  True,  QKVOutputLayout.BSD],
        [2, 1, 128, 8, 1, 128, NormType.RMS_NORM, True,  QKVOutputLayout.BSD],
        # NBSd is the layout the LTX-2 megakernel uses; d_head must be 128.
        [1, 1, 128, 8, 1, 128, NormType.NO_NORM,  False, QKVOutputLayout.NBSd],
        [2, 1, 128, 8, 1, 128, NormType.NO_NORM,  True,  QKVOutputLayout.NBSd],
    ]
    # fmt: on

    @pytest_parametrize(squared_sum_test_params, squared_sum_test_perms)
    def test_qkv_cte_squared_sum_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        vnc_degree,
        batch,
        seqlen,
        n_q_heads,
        n_kv_heads,
        d_head,
        norm_type,
        fused_rope,
        output_layout,
    ):
        np.random.seed(42)
        compiler_args = CompilerArgs(logical_nc_config=vnc_degree, platform_target=platform_target)
        fused_qkv_dim = (n_q_heads + 2 * n_kv_heads) * d_head
        num_heads = n_q_heads + 2 * n_kv_heads
        dtype = nl.bfloat16

        def input_generator(test_config):
            full_input = build_qkv_input(
                batch=batch,
                seqlen=seqlen,
                hidden_dim=8192,
                fused_qkv_dim=fused_qkv_dim,
                dtype=dtype,
                eps=1e-6,
                d_head=d_head,
                norm_type=norm_type,
                fused_add=False,
                output_layout=output_layout,
                lnc_degree=vnc_degree,
                use_dma_transpose=False,
                qkv_bias=False,
                norm_bias=False,
                hidden_actual=None,
                fused_rope=fused_rope,
                num_q_heads=n_q_heads,
                num_kv_heads=n_kv_heads,
                quantization_type=QuantizationType.NONE,
                tensor_gen=rope_gaussian_tensor_generator() if fused_rope else gaussian_tensor_generator(),
            )
            kernel_input = {key: full_input[key] for key in _FORWARDED_KEYS if key in full_input}
            # In-place outputs: passed as inputs the wrapper returns, so the
            # framework treats them as aliased kernel outputs to validate.
            kernel_input["q_squared_sum_out.must_alias_input"] = np.zeros((batch, seqlen, 1), dtype=np.float32)
            kernel_input["k_squared_sum_out.must_alias_input"] = np.zeros((batch, seqlen, 1), dtype=np.float32)
            kernel_input["v_squared_sum_out.must_alias_input"] = np.zeros((batch, seqlen, 1), dtype=np.float32)
            return kernel_input

        def output_tensor_descriptor(kernel_input):
            if output_layout == QKVOutputLayout.NBSd:
                out = np.zeros((num_heads, batch, seqlen, d_head), dtype=dtype)
            else:
                out = np.zeros((batch, seqlen, fused_qkv_dim), dtype=dtype)
            return {
                "out": out,
                "q_squared_sum_out": np.zeros((batch, seqlen, 1), dtype=np.float32),
                "k_squared_sum_out": np.zeros((batch, seqlen, 1), dtype=np.float32),
                "v_squared_sum_out": np.zeros((batch, seqlen, 1), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_qkv_cte_squared_sum_wrapper_kernel,
            torch_ref=torch_ref_wrapper(_qkv_cte_squared_sum_wrapper_ref, preserve_lower_precision=True),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            # Squaring amplifies bf16 rounding error in the projection output, so use
            # a slightly looser tolerance than the plain projection-output tests.
            rtol=5e-2,
            atol=1e-1,
        )
