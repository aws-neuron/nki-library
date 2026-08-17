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

"""Integration tests for fused Adam/AdamW kernel."""

import ml_dtypes
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.optimizer.fused_adam import adam_kernel, adamw_kernel
from nkilib_src.nkilib.experimental.optimizer.fused_adam_torch import (
    adam_torch_ref,
    adamw_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

P_SIZE = 128
LR = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPS = 1e-8
WEIGHT_DECAY = 0.01

_adamw_torch_ref = torch_ref_wrapper(adamw_torch_ref)
_adam_torch_ref = torch_ref_wrapper(adam_torch_ref)


def _generate_inputs(numel, dtype, decoupled_wd, amsgrad=True):
    """Generate kernel inputs."""
    np.random.seed(42)
    param = np.random.randn(numel).astype(dtype)
    grad = np.random.randn(numel).astype(dtype)
    exp_avg = np.zeros(numel, dtype=dtype)
    exp_avg_sq = np.zeros(numel, dtype=dtype)
    max_exp_avg_sq = np.zeros(numel, dtype=dtype)

    bc1 = 1.0 - BETA1
    bc2_sqrt = np.sqrt(1.0 - BETA2)
    step_size = np.full((P_SIZE, 1), LR / bc1, dtype=np.float32)
    inv_bc2_sqrt = np.full((P_SIZE, 1), 1.0 / bc2_sqrt, dtype=np.float32)
    wd_factor = np.full((P_SIZE, 1), (1.0 - LR * WEIGHT_DECAY) if decoupled_wd else WEIGHT_DECAY, dtype=np.float32)

    return {
        "param_ptr": param,
        "grad_ptr": grad,
        "exp_avg_ptr": exp_avg,
        "exp_avg_sq_ptr": exp_avg_sq,
        "max_exp_avg_sq_ptr": max_exp_avg_sq,
        "step_size_ptr": step_size,
        "inv_bc2_sqrt_ptr": inv_bc2_sqrt,
        "wd_factor_ptr": wd_factor,
        "numel": numel,
        "beta1": BETA1,
        "beta2": BETA2,
        "eps": EPS,
        "amsgrad": amsgrad,
    }


def _output_tensors(kernel_input):
    """Generate output tensor descriptors."""
    numel = kernel_input["numel"]
    dtype = kernel_input["param_ptr"].dtype
    out = {
        "param_out": np.zeros(numel, dtype=dtype),
        "exp_avg_out": np.zeros(numel, dtype=dtype),
        "exp_avg_sq_out": np.zeros(numel, dtype=dtype),
    }
    if kernel_input["amsgrad"]:
        out["max_exp_avg_sq_out"] = np.zeros(numel, dtype=dtype)
    return out


PARAM_NAMES = "numel, dtype"
BF16_PARAMS = [
    (128, ml_dtypes.bfloat16),
    (256, ml_dtypes.bfloat16),
    (512, ml_dtypes.bfloat16),
    (1024, ml_dtypes.bfloat16),
    (4096, ml_dtypes.bfloat16),
    (16384, ml_dtypes.bfloat16),
]
F32_PARAMS = [
    (128, np.float32),
    (256, np.float32),
    (512, np.float32),
    (1024, np.float32),
    (4096, np.float32),
]
BOUNDARY_PARAMS = [
    (127, ml_dtypes.bfloat16),
    (129, ml_dtypes.bfloat16),
    (255, ml_dtypes.bfloat16),
    (257, ml_dtypes.bfloat16),
]


@pytest_test_metadata(
    name="FusedAdam",
    pytest_marks=["fused_adam"],
)
class TestFusedAdam:
    """Integration tests for the fused Adam/AdamW NKI kernel."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_torch_ref, True),
            (adam_kernel, _adam_torch_ref, False),
        ],
    )
    @pytest_parametrize(PARAM_NAMES, BF16_PARAMS)
    def test_bf16(
        self, test_manager: Orchestrator, platform_target: Platforms, kernel_func, torch_ref, decoupled_wd, numel, dtype
    ):
        """Correctness sweep over tensor sizes (bf16)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_torch_ref, True),
            (adam_kernel, _adam_torch_ref, False),
        ],
    )
    @pytest_parametrize(PARAM_NAMES, F32_PARAMS)
    def test_f32(
        self, test_manager: Orchestrator, platform_target: Platforms, kernel_func, torch_ref, decoupled_wd, numel, dtype
    ):
        """Correctness sweep over tensor sizes (f32)."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-4, atol=1e-5
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_torch_ref, True),
            (adam_kernel, _adam_torch_ref, False),
        ],
    )
    @pytest_parametrize(PARAM_NAMES, BOUNDARY_PARAMS)
    def test_boundary_sizes(
        self, test_manager: Orchestrator, platform_target: Platforms, kernel_func, torch_ref, decoupled_wd, numel, dtype
    ):
        """Boundary sizes: non-multiples of P_SIZE to exercise tail path."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_torch_ref, True),
            (adam_kernel, _adam_torch_ref, False),
        ],
    )
    @pytest_parametrize(
        PARAM_NAMES, [(128, ml_dtypes.bfloat16), (1024, ml_dtypes.bfloat16), (4096, ml_dtypes.bfloat16)]
    )
    def test_no_amsgrad(
        self, test_manager: Orchestrator, platform_target: Platforms, kernel_func, torch_ref, decoupled_wd, numel, dtype
    ):
        """amsgrad=False path: 3 outputs, no max_exp_avg_sq tracking."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd, amsgrad=False),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_torch_ref, True),
            (adam_kernel, _adam_torch_ref, False),
        ],
    )
    @pytest_parametrize(PARAM_NAMES, [(262144, ml_dtypes.bfloat16), (1048576, ml_dtypes.bfloat16)])
    def test_large(
        self, test_manager: Orchestrator, platform_target: Platforms, kernel_func, torch_ref, decoupled_wd, numel, dtype
    ):
        """Large tensors to exercise multi-tile and SPMD."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )

    # --- Performance: roofline comparison ---

    @pytest.mark.parametrize(
        "numel,dtype,label",
        [
            (823_186_880, ml_dtypes.bfloat16, "gpt_oss_bf16"),
            (165_117_392, np.float32, "qwen_f32"),
        ],
    )
    def test_perf_roofline(self, test_manager: Orchestrator, platform_target: Platforms, numel, dtype, label):
        """Performance test: compare against roofline. Requires --target-host for hardware execution.

        5 VectorE + 6 ScalarE ops/element.
        Memory-bound on Trn2 LNC2 (7 accesses per element).

        Manual profiling results:
          GPT-OSS bf16 (823M elems): 25.8ms measured vs 18.0ms memory roofline (1.43x)
          QWen f32 (165M elems): 11.95ms measured vs 7.22ms memory roofline (1.66x)
        """
        bytes_per_elem = 4 if dtype == np.float32 else 2
        hbm_bw = 640e9
        accesses = 7
        t_memory_ms = numel * bytes_per_elem * accesses / hbm_bw * 1000
        print(f"\n[{label}] Memory roofline: {t_memory_ms:.2f}ms")

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=adamw_kernel,
            torch_ref=_adamw_torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd=True, amsgrad=False),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-2, atol=1e-2
        )
