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
"""Tests for gdn_cte kernel."""

from typing import Any, final

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.gdn.gdn_cte import gdn_cte
from nkilib_src.nkilib.experimental.gdn.gdn_cte_torch import gdn_cte_torch_nki_ref

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _generate_inputs(batch_size: int, seq_len: int, head_dim: int, dtype: Any) -> dict:
    rng = np.random.RandomState(42)
    q = rng.randn(batch_size, seq_len, head_dim).astype(np.float32)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    k = rng.randn(batch_size, seq_len, head_dim).astype(np.float32)
    k = k / np.linalg.norm(k, axis=-1, keepdims=True)
    q = dt.static_cast(q, dtype)
    k = dt.static_cast(k, dtype)
    v = dt.static_cast(rng.randn(batch_size, seq_len, head_dim).astype(np.float32) * 0.23, dtype)
    beta = rng.rand(batch_size, seq_len).astype(np.float32) * 0.95 + 0.05
    gate = -rng.exponential(scale=1.8, size=(batch_size, seq_len)).astype(np.float32)
    gate = dt.static_cast(gate, dtype)
    scale = 1.0 / np.sqrt(head_dim)
    return {"q": q, "k": k, "v": v, "beta": beta, "gate": gate, "scale": scale}


@torch_ref_wrapper
def _torch_ref(q, k, v, beta, gate, scale):
    import torch

    def to_torch(x):
        return torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.tensor(x).float()

    out, final_state = gdn_cte_torch_nki_ref(
        to_torch(q), to_torch(k), to_torch(v), to_torch(beta), gate=to_torch(gate), scale=scale
    )
    return {"out": out.numpy(), "state_output": final_state.numpy()}


@final
@pytest_test_metadata(name="GDN CTE V3")
@pytest_marks(["attention", "gdn"])
class TestGdnCteV3Kernel:
    gdn_cte_params = "batch_size, seq_len, head_dim, dtype"
    _ABBREVS = {"batch_size": "b", "seq_len": "s", "head_dim": "d", "dtype": "dt"}

    test_cases_basic = [
        (1, 64, 128, nl.bfloat16),
        (1, 128, 128, nl.bfloat16),
        (32, 64, 128, nl.bfloat16),
        (32, 128, 128, nl.bfloat16),
    ]

    test_cases_large = [
        (1, 256, 128, nl.bfloat16),
        (32, 2304, 128, nl.bfloat16),
    ]

    def _run_test(self, test_manager, platform_target, batch_size, seq_len, head_dim, dtype, atol=5e-2, rtol=5e-2):
        def input_generator(test_config, input_tensor_def=None):
            return _generate_inputs(batch_size, seq_len, head_dim, dtype)

        def output_tensors(kernel_input):
            q = kernel_input["q"]
            B, S, D = q.shape
            return {"out": np.zeros(q.shape, q.dtype), "state_output": np.zeros((B, D, D), dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=gdn_cte,
            torch_ref=_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=rtol, atol=atol
        )

    @pytest.mark.fast
    @pytest_parametrize(gdn_cte_params, test_cases_basic, abbrevs=_ABBREVS)
    def test_gdn_cte_basic(
        self, test_manager: Orchestrator, platform_target: Platforms, batch_size, seq_len, head_dim, dtype
    ):
        self._run_test(test_manager, platform_target, batch_size, seq_len, head_dim, dtype)

    @pytest_parametrize(gdn_cte_params, test_cases_large, abbrevs=_ABBREVS)
    def test_gdn_cte_large(
        self, test_manager: Orchestrator, platform_target: Platforms, batch_size, seq_len, head_dim, dtype
    ):
        self._run_test(test_manager, platform_target, batch_size, seq_len, head_dim, dtype)
