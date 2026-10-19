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

from typing import final

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.attention.kda_block_tkg import kda_block_tkg
from nkilib_src.nkilib.experimental.attention.kda_block_tkg_torch import kda_block_tkg_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

_CONV_STREAMS = 3


def _generate_block_inputs(
    batch, num_head, dk, dv, hidden_size, conv_taps, lower_bound=-5.0, weight_dtype=ml_dtypes.bfloat16
):
    """Generate inputs for kda_block_tkg.

    Weight magnitudes are kept small so the fused projections do not blow up the
    activations before the norms bring them back; the recurrence itself is validated
    over its full range by test_kda_tkg.

    weight_dtype selects the dtype of every projection/filter weight, and defaults to
    bf16 -- weights are never fp32 in practice (fp8 and MX are the future options). It matters for more than
    bandwidth: input_projection_tkg stages the fused weight as
    NUM_TILES_PER_H_BLOCK(16) x min(I, 4096) elements per partition, which is 256 KB at
    fp32 and overflows the 192 KB partition budget for any real KDA width. bf16 halves
    it to 128 KB, and is what the model actually ships as.
    """
    np.random.seed(42)

    i_qk = dk * num_head
    i_v = dv * num_head
    proj_width = 2 * i_qk + i_v + num_head + 2 * dk  # q|k|v|g_a|beta|f_a
    conv_channels = i_qk

    def _w(*shape, s=0.05):
        return (np.random.randn(*shape) * s).astype(np.float32)

    return {
        "hidden": np.random.randn(batch, 1, hidden_size).astype(weight_dtype),
        "gamma_in": np.ones((1, hidden_size), dtype=np.float32),
        "W_in": _w(hidden_size, proj_width).astype(weight_dtype),
        "W_fb": _w(dk, i_qk).astype(weight_dtype),
        "W_gb": _w(dk, i_v).astype(weight_dtype),
        "conv_w": _w(_CONV_STREAMS, conv_taps, conv_channels, s=0.5).astype(weight_dtype),
        "A_log": np.random.rand(1, num_head).astype(np.float32),
        "dt_bias": _w(dk, num_head, s=0.5),
        "gamma_o": np.ones((1, dv), dtype=np.float32),
        "W_o": _w(i_v, hidden_size).astype(weight_dtype),
        "out.must_alias_input": np.zeros((batch, 1, hidden_size), dtype=weight_dtype),
        "S.must_alias_input": (np.random.randn(batch, num_head, dk, dv) * 0.01).astype(np.float32),
        "conv_state.must_alias_input": _w(batch, _CONV_STREAMS, conv_taps - 1, conv_channels, s=0.1).astype(
            weight_dtype
        ),
        "scale": np.array([[1.0 / np.sqrt(dk)]], dtype=np.float32),
        "lower_bound": np.array([[lower_bound]], dtype=np.float32),
    }


@pytest_marks(["attention", "kda", "tkg"])
@final
class TestKdaBlockTkg:
    def _run(
        self,
        *,
        test_manager,
        platform_target,
        batch,
        num_head,
        dk,
        dv,
        hidden_size,
        conv_taps,
        lower_bound=-5.0,
        weight_dtype=ml_dtypes.bfloat16,
        rtol=2e-2,
        atol=1e-2,
        logical_nc_config=None,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return _generate_block_inputs(
                batch=batch,
                num_head=num_head,
                dk=dk,
                dv=dv,
                hidden_size=hidden_size,
                conv_taps=conv_taps,
                lower_bound=lower_bound,
                weight_dtype=weight_dtype,
            )

        def output_tensor_descriptor(kernel_input):  # noqa: ARG001
            return {
                "out": np.zeros((batch, 1, hidden_size), dtype=weight_dtype),
                "S": np.zeros((batch, num_head, dk, dv), dtype=np.float32),
                "conv_state": np.zeros((batch, _CONV_STREAMS, conv_taps - 1, dk * num_head), dtype=weight_dtype),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kda_block_tkg,
            torch_ref=torch_ref_wrapper(kda_block_tkg_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensor_descriptor,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
                logical_nc_config=logical_nc_config,
            ),
            rtol=rtol,
            atol=atol,
        )

    @pytest_marks(["model"])
    @pytest.mark.parametrize(
        "batch, num_head, dk, dv, hidden, conv_taps",
        [
            # Kimi K3 KDA layer: hidden_size 7168, linear_attn_config head_dim 128 /
            # num_heads 96 / short_conv_kernel_size 4, gate_lower_bound -5. num_head is
            # PER RANK; this is the unsharded (TP=1) layer. Weights are bf16, matching
            # the model's own dtype -- see _generate_block_inputs on why fp32 weights
            # cannot fit the fused projection's SBUF staging at this width.
            pytest.param(1, 96, 128, 128, 7168, 4, id="tp1"),
        ],
    )
    def test_kda_block_tkg_hw_model(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch,
        num_head,
        dk,
        dv,
        hidden,
        conv_taps,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            num_head=num_head,
            dk=dk,
            dv=dv,
            hidden_size=hidden,
            conv_taps=conv_taps,
            weight_dtype=ml_dtypes.bfloat16,
            rtol=2e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    @pytest_marks(["model"])
    @pytest.mark.parametrize(
        "batch, num_head, dk, dv, hidden, conv_taps",
        [
            pytest.param(1, 8, 64, 64, 512, 4),
        ],
    )
    def test_kda_block_tkg_hw(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch,
        num_head,
        dk,
        dv,
        hidden,
        conv_taps,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            num_head=num_head,
            dk=dk,
            dv=dv,
            hidden_size=hidden,
            conv_taps=conv_taps,
        )
