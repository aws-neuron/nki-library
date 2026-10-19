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

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.kimi_k3.kda.kda_tkg import kda_tkg
from nkilib_src.nkilib.experimental.kimi_k3.kda.kda_tkg_torch import kda_tkg_torch_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_tkg_inputs(batch, num_head, dk, dv, lower_bound=-5.0, saturate_gate=False):
    """Generate inputs for kda_tkg.

    Args:
        batch: Number of requests in the batch
        num_head: Number of value heads
        dk: Key/query head dimension (<= 128)
        dv: Value head dimension (<= 128)
        lower_bound: Gate lower bound, in [-5, 0]
        saturate_gate: If True, drive g_raw very positive so the activated gate
            saturates at ``lower_bound`` on every channel -- the worst case for
            the exponential range of the decay.
    """
    np.random.seed(42)

    # q/k are L2-normalized in-kernel, so the input scale is irrelevant for them.
    q = np.random.randn(batch, num_head, dk).astype(np.float32)
    k = np.random.randn(batch, num_head, dk).astype(np.float32)
    v = (np.random.randn(batch, num_head, dv) * 0.1).astype(np.float32)

    if saturate_gate:
        # sigmoid(large) -> 1, so g -> lower_bound on every channel.
        g_raw = np.full((batch, num_head, dk), 20.0, dtype=np.float32)
        dt_bias = np.zeros((num_head, dk), dtype=np.float32)
        a_log = np.zeros((num_head, 1), dtype=np.float32)
    else:
        # Spread the activated gate across the full (lower_bound, 0) range.
        g_raw = (np.random.randn(batch, num_head, dk) * 3.0).astype(np.float32)
        dt_bias = (np.random.randn(num_head, dk) * 0.5).astype(np.float32)
        a_log = np.random.rand(num_head, 1).astype(np.float32)

    # q/k/g are CHANNEL-MAJOR [batch, dk, num_head]; v/out/state stay head-major.
    return {
        "q": np.ascontiguousarray(q.transpose(0, 2, 1)),
        "k": np.ascontiguousarray(k.transpose(0, 2, 1)),
        "v": v,
        "g": np.ascontiguousarray(g_raw.transpose(0, 2, 1)),
        "beta": np.random.randn(batch, 1, num_head).astype(np.float32),
        "a_log": a_log.reshape(1, num_head),
        "dt_bias": np.ascontiguousarray(dt_bias.T),
        "out.must_alias_input": np.zeros((batch, num_head, dv), dtype=np.float32),
        "S.must_alias_input": (np.random.randn(batch, num_head, dk, dv) * 0.01).astype(np.float32),
        "scale": np.array([[1.0 / np.sqrt(dk)]], dtype=np.float32),
        "lower_bound": np.array([[lower_bound]], dtype=np.float32),
    }


@pytest_marks(["attention", "kda", "tkg"])
@final
class TestKdaTkg:
    def _run(
        self,
        *,
        test_manager,
        platform_target,
        batch,
        num_head,
        dk,
        dv,
        lower_bound=-5.0,
        saturate_gate=False,
        rtol=1e-4,
        atol=1e-5,
        logical_nc_config=None,
    ):
        def input_generator(test_config, input_tensor_def=None):
            return _generate_tkg_inputs(
                batch=batch,
                num_head=num_head,
                dk=dk,
                dv=dv,
                lower_bound=lower_bound,
                saturate_gate=saturate_gate,
            )

        def output_tensor_descriptor(kernel_input):
            return {
                "out": np.zeros((batch, num_head, dv), dtype=np.float32),
                "S": np.zeros((batch, num_head, dk, dv), dtype=np.float32),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kda_tkg,
            torch_ref=torch_ref_wrapper(kda_tkg_torch_ref),
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
        "batch, num_head, dk, dv",
        [
            pytest.param(4, 96, 128, 128),
        ],
    )
    def test_kda_tkg_hw(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch,
        num_head,
        dk,
        dv,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            num_head=num_head,
            dk=dk,
            dv=dv,
            rtol=1e-3,
            atol=1e-4,
        )

    @pytest_marks(["model"])
    @pytest.mark.parametrize(
        "batch, num_head, dk, dv",
        [
            pytest.param(1, 96, 128, 128),
        ],
    )
    def test_kda_tkg_hw_lnc1(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch,
        num_head,
        dk,
        dv,
    ):
        self._run(
            test_manager=test_manager,
            platform_target=platform_target,
            batch=batch,
            num_head=num_head,
            dk=dk,
            dv=dv,
            rtol=1e-3,
            atol=1e-4,
            logical_nc_config=1,
        )
