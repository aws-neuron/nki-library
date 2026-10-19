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
from nkilib_src.nkilib.experimental.optimizer.adam_kernel_torch import adam_kernel_torch_ref
from nkilib_src.nkilib.experimental.optimizer.adamw_kernel_torch import adamw_kernel_torch_ref
from nkilib_src.nkilib.experimental.optimizer.fused_adam import adam_kernel, adamw_kernel

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.negative_test_helpers import assert_negative_test_case
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

_adamw_torch_ref = torch_ref_wrapper(adamw_kernel_torch_ref)
_adam_torch_ref = torch_ref_wrapper(adam_kernel_torch_ref)
_adamw_fp16_torch_ref = torch_ref_wrapper(adamw_kernel_torch_ref, preserve_lower_precision=True)
_adam_fp16_torch_ref = torch_ref_wrapper(adam_kernel_torch_ref, preserve_lower_precision=True)


def _generate_inputs(numel, dtype, decoupled_wd, amsgrad=True, shape=None):
    """Generate kernel inputs."""
    np.random.seed(42)
    shape = (numel,) if shape is None else shape
    param = np.random.randn(*shape).astype(dtype)
    grad = np.random.randn(*shape).astype(dtype)
    exp_avg = np.zeros(shape, dtype=dtype)
    exp_avg_sq = np.zeros(shape, dtype=dtype)
    max_exp_avg_sq = np.zeros(shape, dtype=dtype)

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


def _generate_fp16_small_gradient_inputs(numel, decoupled_wd, amsgrad):
    """Exercise zero and tiny gradients with Adam's default epsilon."""
    inputs = _generate_inputs(numel, np.float16, decoupled_wd, amsgrad=amsgrad)
    inputs["param_ptr"].fill(1.0)
    inputs["grad_ptr"].fill(0.0)
    inputs["grad_ptr"][1] = 1e-5
    return inputs


def _output_tensors(kernel_input):
    """Generate output tensor descriptors."""
    shape = kernel_input["param_ptr"].shape
    dtype = kernel_input["param_ptr"].dtype
    out = {
        "param_out": np.zeros(shape, dtype=dtype),
        "exp_avg_out": np.zeros(shape, dtype=dtype),
        "exp_avg_sq_out": np.zeros(shape, dtype=dtype),
    }
    if kernel_input["amsgrad"]:
        out["max_exp_avg_sq_out"] = np.zeros(shape, dtype=dtype)
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
FP16_PARAMS = [
    (128, np.float16),
    (256, np.float16),
    (512, np.float16),
    (1024, np.float16),
    (4096, np.float16),
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

    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_torch_ref, True),
            (adam_kernel, _adam_torch_ref, False),
        ],
    )
    @pytest.mark.coverage_parametrize(case=BF16_PARAMS, coverage="full", enable_automatic_boundary_tests=False)
    def test_bf16(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        kernel_func,
        torch_ref,
        decoupled_wd,
        case,
        is_negative_test_case,
    ):
        """Correctness sweep over tensor sizes (bf16)."""
        numel, dtype = case
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-2,
            atol=1e-2,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_torch_ref, True),
            (adam_kernel, _adam_torch_ref, False),
        ],
    )
    @pytest.mark.coverage_parametrize(case=F32_PARAMS, coverage="full", enable_automatic_boundary_tests=False)
    def test_f32(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        kernel_func,
        torch_ref,
        decoupled_wd,
        case,
        is_negative_test_case,
    ):
        """Correctness sweep over tensor sizes (f32)."""
        numel, dtype = case
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-4,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_fp16_torch_ref, True),
            (adam_kernel, _adam_fp16_torch_ref, False),
        ],
    )
    @pytest_parametrize("numel,amsgrad", [(128, False), (257, False), (128, True)])
    def test_fp16_zero_and_small_gradients(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        kernel_func,
        torch_ref,
        decoupled_wd,
        numel,
        amsgrad,
    ):
        """The fp16 update must follow PyTorch's fp32 opmath and stay finite."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_fp16_small_gradient_inputs(numel, decoupled_wd, amsgrad),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-3, atol=1e-5
        )

    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_fp16_torch_ref, True),
            (adam_kernel, _adam_fp16_torch_ref, False),
        ],
    )
    @pytest.mark.coverage_parametrize(case=FP16_PARAMS, coverage="full", enable_automatic_boundary_tests=False)
    def test_fp16_size_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        kernel_func,
        torch_ref,
        decoupled_wd,
        case,
        is_negative_test_case,
    ):
        """FP16 Adam variants match the rounded PyTorch reference across sizes."""
        numel, dtype = case
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(numel, dtype, decoupled_wd),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=1e-3,
            atol=3e-4,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_fp16_torch_ref, True),
            (adam_kernel, _adam_fp16_torch_ref, False),
        ],
    )
    def test_fp16_random_small_gradients(
        self, test_manager: Orchestrator, platform_target: Platforms, kernel_func, torch_ref, decoupled_wd
    ):
        """Step-one gradients around 1e-6 must keep the update finite."""

        def inputs(_):
            kernel_input = _generate_inputs(4096, np.float16, decoupled_wd, amsgrad=False)
            kernel_input["grad_ptr"] = np.random.default_rng(42).uniform(-1e-6, 1e-6, 4096).astype(np.float16)
            return kernel_input

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=inputs,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-3, atol=3e-4
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

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "kernel_func,torch_ref,decoupled_wd",
        [
            (adamw_kernel, _adamw_torch_ref, True),
            (adam_kernel, _adam_torch_ref, False),
        ],
    )
    @pytest_parametrize("shape", [(2, 8), (3, 5, 17), (4, 128, 128)])
    @pytest_parametrize("amsgrad", [False, True])
    def test_multidimensional(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        kernel_func,
        torch_ref,
        decoupled_wd,
        shape,
        amsgrad,
    ):
        """Flat access preserves the original shape for small, tail, and multi-tile buffers."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: _generate_inputs(
                int(np.prod(shape)), np.float32, decoupled_wd, amsgrad=amsgrad, shape=shape
            ),
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None, compiler_args=CompilerArgs(platform_target=platform_target), rtol=1e-4, atol=1e-5
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("kernel_func", [adam_kernel, adamw_kernel])
    @pytest_parametrize("numel", [2, 15, 17])
    def test_invalid_numel(self, test_manager: Orchestrator, platform_target: Platforms, kernel_func, numel):
        """Reject partial updates and out-of-bounds access, including the old shape[0] check."""
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_func,
            kernel_input_generator=lambda _: _generate_inputs(numel, np.float32, False, shape=(2, 8)),
            trace_only=True,
        )
        with assert_negative_test_case(True, f"param_ptr size 16 must equal numel {numel}"):
            framework.run_test(test_config=None, compiler_args=CompilerArgs(platform_target=platform_target))

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
