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
from nkilib_src.nkilib.experimental.subkernels.rmsnorm_residual_prefill import rmsnorm_residual_prefill
from nkilib_src.nkilib.experimental.subkernels.rmsnorm_residual_prefill_torch import (
    rmsnorm_residual_prefill_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# fmt: off
RMSNORM_PARAM_NAMES = "lnc_degree, tokens, hidden, hidden_actual, has_residual, eps"
RMSNORM_PARAMS = [
    # normal case, microbenchmark shape
    (1, 1024, 3072, None, True, 1e-6),
    # H below the H tile width, so a single H tile
    pytest.param(1, 128, 512, None, True, 1e-6, marks=pytest.mark.fast),
    # padded H: mean divides by hidden_actual, not H
    pytest.param(1, 1024, 3072, 2880, True, 1e-6, marks=pytest.mark.fast),
    # H above the tile width and not a multiple of it, so an uneven tail
    (1, 128, 2048, None, True, 1e-6),
    # fewer tokens than one tile
    (1, 64, 3072, None, True, 1e-6),
    # token count not a multiple of the tile, so a partial final tile
    (1, 1000, 3072, None, True, 1e-6),
    # more token tiles than the buffer rotation degree
    (1, 2048, 3072, None, True, 1e-6),
    # LNC2 with a token count the shards cannot divide evenly
    (2, 1023, 3072, None, True, 1e-6),
    # the LNC1 cases above, resharded across two cores
    (2, 1024, 3072, None, True, 1e-6),
    (2, 128, 512, None, True, 1e-6),
    (2, 1024, 3072, 2880, True, 1e-6),
    (2, 128, 2048, None, True, 1e-6),
    (2, 64, 3072, None, True, 1e-6),
    (2, 1000, 3072, None, True, 1e-6),
    (2, 2048, 3072, None, True, 1e-6),
    (2, 2048, 3072, 2880, True, 1e-6),
    # plain RMSNorm, no residual add
    (1, 1024, 3072, None, False, 1e-6),
    # H below one PSUM bank and not a multiple of it, so the gamma broadcast is a short matmul
    (1, 128, 256, None, True, 1e-6),
    # a single token at LNC2, so one shard gets no tokens at all
    (2, 1, 3072, None, True, 1e-6),
]
# fmt: on


@pytest_test_metadata(name="RmsNormResidualPrefill")
@pytest_marks(["rmsnorm_residual_prefill", "subkernels"])
@final
class TestRmsNormResidualPrefillKernel:
    @pytest.mark.parametrize(RMSNORM_PARAM_NAMES, RMSNORM_PARAMS)
    def test_rmsnorm(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        lnc_degree: int,
        tokens: int,
        hidden: int,
        hidden_actual: int,
        has_residual: bool,
        eps: float,
    ) -> None:
        np.random.seed(42)
        hidden_np = np.random.randn(tokens, hidden).astype(np.dtype("bfloat16"))
        residual_np = np.random.randn(tokens, hidden).astype(np.dtype("bfloat16"))
        gamma_np = np.random.randn(1, hidden).astype(np.dtype("bfloat16"))

        def input_generator(test_config):
            kernel_input = {
                "hidden": hidden_np,
                "gamma": gamma_np,
                "eps": eps,
                "hidden_actual": hidden_actual,
            }
            if has_residual:
                kernel_input["residual"] = residual_np
            return kernel_input

        def output_tensors(kernel_input):
            tensors = {"out": np.zeros((tokens, hidden), dtype=np.dtype("bfloat16"))}
            if has_residual:
                tensors["residual_out"] = np.zeros((tokens, hidden), dtype=np.dtype("bfloat16"))
            return tensors

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_residual_prefill,
            torch_ref=torch_ref_wrapper(rmsnorm_residual_prefill_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=lnc_degree),
            inference_args=InferenceArgs(enable_determinism_check=True, num_runs=10),
            rtol=5e-2,
            atol=5e-2,
        )

    @pytest.mark.parametrize("batch, seqlen, hidden", [(2, 512, 3072)])
    def test_rmsnorm_3d(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
    ) -> None:
        """[B, S, H] input, the layout the model code passes. Output stays [B*S, H]."""
        np.random.seed(42)
        hidden_np = np.random.randn(batch, seqlen, hidden).astype(np.dtype("bfloat16"))
        residual_np = np.random.randn(batch, seqlen, hidden).astype(np.dtype("bfloat16"))
        gamma_np = np.random.randn(1, hidden).astype(np.dtype("bfloat16"))

        def input_generator(test_config):
            return {"hidden": hidden_np, "gamma": gamma_np, "residual": residual_np, "eps": 1e-6}

        def output_tensors(kernel_input):
            shape = (batch * seqlen, hidden)
            return {
                "out": np.zeros(shape, dtype=np.dtype("bfloat16")),
                "residual_out": np.zeros(shape, dtype=np.dtype("bfloat16")),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_residual_prefill,
            torch_ref=torch_ref_wrapper(rmsnorm_residual_prefill_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            inference_args=InferenceArgs(enable_determinism_check=True, num_runs=10),
            rtol=5e-2,
            atol=5e-2,
        )

    @pytest.mark.parametrize("dtype", ["float32", "float16"])
    def test_rmsnorm_gamma_dtype(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        dtype: str,
    ) -> None:
        """Gamma wider or narrower than the hidden states, which selects the PSUM dtype."""
        tokens, hidden = 128, 3072
        np.random.seed(42)
        hidden_np = np.random.randn(tokens, hidden).astype(np.dtype("bfloat16"))
        residual_np = np.random.randn(tokens, hidden).astype(np.dtype("bfloat16"))
        gamma_np = np.random.randn(1, hidden).astype(np.dtype(dtype))

        def input_generator(test_config):
            return {"hidden": hidden_np, "gamma": gamma_np, "residual": residual_np, "eps": 1e-6}

        def output_tensors(kernel_input):
            shape = (tokens, hidden)
            return {
                "out": np.zeros(shape, dtype=np.dtype("bfloat16")),
                "residual_out": np.zeros(shape, dtype=np.dtype("bfloat16")),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rmsnorm_residual_prefill,
            torch_ref=torch_ref_wrapper(rmsnorm_residual_prefill_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            inference_args=InferenceArgs(enable_determinism_check=True, num_runs=10),
            rtol=5e-2,
            atol=5e-2,
        )
