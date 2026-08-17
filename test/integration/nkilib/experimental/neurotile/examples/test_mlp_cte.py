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
"""Integration test for the neurotile mlp_cte_nt example kernel."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.neurotile.examples.kernels.mlp_cte import (
    mlp_cte_nt as kernel_mod,
)
from nkilib_src.nkilib.experimental.neurotile.examples.kernels.mlp_cte import (
    mlp_cte_nt_torch as refs,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Smallest config from run_mlp.py CONFIGS["small"].
_M, _K, _I, _H = 1024, 512, 512, 512


def _mlp_inputs(_):
    np.random.seed(42)
    return {
        "x": (np.random.randn(1, _M, _K) * 0.1).astype(ml_dtypes.bfloat16),
        "gate_proj": (np.random.randn(_K, _I) * 0.1).astype(ml_dtypes.bfloat16),
        "up_proj": (np.random.randn(_K, _I) * 0.1).astype(ml_dtypes.bfloat16),
        "down_proj": (np.random.randn(_I, _H) * 0.1).astype(ml_dtypes.bfloat16),
        "config": kernel_mod.MLPConfig(K=_K, I=_I, H=_H, bxs_subtile_count=4),
    }


def _mlp_outputs(_kernel_input):
    return {"out": np.zeros((1, _M, _H), dtype=ml_dtypes.bfloat16)}


@pytest_marks(["neurotile"])
class TestNeurotileMLPCTE:
    @pytest.mark.fast
    def test_mlp_cte(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_mod.mlp_cte,
            torch_ref=torch_ref_wrapper(refs.mlp_cte_torch_ref),
            kernel_input_generator=_mlp_inputs,
            output_tensor_descriptor=_mlp_outputs,
        )
        framework.run_test(
            test_config=None,
            # Tutorial invokes mlp_cte[2](...) — runs at LNC=2.
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            # MLP has multiple bf16 matmul accumulations; tolerance per run_mlp.py "small" config.
            rtol=0.0625,
            atol=0.0625,
        )
