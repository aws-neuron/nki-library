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

"""Integration tests for MXFP8 matrix multiplication backward kernel."""

from typing import Any, final

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_backward_kernel
from typing_extensions import override

from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.utils import common_dataclasses
from test.utils.unit_test_framework import UnitTestFramework

# ============================================================================
# Custom comparator — matches MLP backward test pattern
# ============================================================================


def _bwd_comparator(golden_dict, output_tensors):
    """custom_comparator for matmul backward tests."""
    result = {}
    for name, golden in golden_dict.items():
        dtype = golden.dtype

        class _BwdValidator(common_dataclasses.CustomValidator):
            _golden = golden
            _label = name

            @override
            def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                reshaped = inference_output.view(dtype=self._golden.dtype).reshape(self._golden.shape)
                passed, metrics = matmul_utils.check_correctness(reshaped, self._golden.astype(self._golden.dtype))
                if not passed:
                    self._print_with_log(f"[{self._label}] Validation FAILED")
                    self._print_with_log(f"  metrics: {metrics}")
                    self._print_with_log(f"  kernel[0,:5]: {reshaped.flatten()[:5]}")
                    self._print_with_log(f"  golden[0,:5]: {self._golden.flatten()[:5]}")
                return passed

        result[name] = common_dataclasses.CustomValidatorWithOutputTensorData(
            validator=_BwdValidator,
            output_ndarray=np.ndarray(golden.shape, dtype=dtype),
        )
    return result


# ============================================================================
# Torch reference — signature matches kernel exactly
# ============================================================================


def matmul_mxfp8_backward_torch_ref(
    output_grad,
    weights,
    input_activation,
    input_grad_config=None,
    weight_grad_config=None,
    tile_loop_order="mnk",
    float8_dtype="float8_e5m2",
    output_dtype=None,
    run_with_lnc2=True,
    lnc_2_shard_rhs=True,
    output_grad_scales=None,
    weight_scales=None,
    input_scales=None,
    use_scale_packing=False,
    spill_reload=False,
    output_grad_is_swizzled=False,
    weights_is_swizzled=False,
    input_is_swizzled=False,
):
    """Compute golden backward pass: dX = dY @ W, dW = dY^T @ X."""
    dy = output_grad.astype(np.float32)
    w = weights.astype(np.float32)
    x = input_activation.astype(np.float32)

    return {
        "input_grad": (dy @ w).astype(np.float32),
        "weight_grad": (dy.T @ x).astype(np.float32),
    }


# ============================================================================
# Test configurations
# ============================================================================

BACKWARD_CONFIGS = [
    (512, 512, 512, "Square_small"),
    (1024, 1024, 1024, "Square_medium"),
    (2048, 1024, 512, "Nonsquare_M_gt_N"),
    (512, 1024, 2048, "Nonsquare_N_gt_M"),
]


# ============================================================================
# Test class
# ============================================================================


@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
@final
class TestMatmulMxfp8GenericBackwardKernel:
    """Tests for the MXFP8 matmul backward kernel."""

    def _run_backward_test(self, test_manager, platform_target, M, K, N, seed=42):
        """Run a single backward kernel test with given dimensions."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        rng = np.random.default_rng(seed)
        output_grad = rng.standard_normal((M, N)).astype(np.float32).astype(nl.bfloat16)
        weights = rng.standard_normal((N, K)).astype(np.float32).astype(nl.bfloat16)
        input_activation = rng.standard_normal((M, K)).astype(np.float32).astype(nl.bfloat16)

        kernel_input = {
            "output_grad": output_grad,
            "weights": weights,
            "input_activation": input_activation,
            "output_dtype": nl.float32,
            "run_with_lnc2": False,
            "output_grad_is_swizzled": False,
            "weights_is_swizzled": False,
            "input_is_swizzled": False,
        }

        output_ndarray = {
            "input_grad": np.zeros((M, K), dtype=np.float32),
            "weight_grad": np.zeros((N, K), dtype=np.float32),
        }

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=1,
            platform_target=platform_target,
        )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=matmul_mxfp8_generic_backward_kernel.matmul_mxfp8_backward,
            torch_ref=matmul_mxfp8_backward_torch_ref,
            kernel_input_generator=lambda _: kernel_input,
            output_tensor_descriptor=lambda _: output_ndarray.copy(),
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_bwd_comparator,
        )

    @pytest.mark.parametrize(
        "M,K,N,description",
        BACKWARD_CONFIGS,
        ids=[desc for _, _, _, desc in BACKWARD_CONFIGS],
    )
    def test_matmul_mxfp8_backward(self, test_manager, platform_target, M, K, N, description):
        """Test backward pass computes correct input and weight gradients."""
        self._run_backward_test(test_manager, platform_target, M, K, N)
