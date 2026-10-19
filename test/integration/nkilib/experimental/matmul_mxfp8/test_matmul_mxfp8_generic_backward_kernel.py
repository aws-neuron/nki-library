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
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation
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
    fast_dma_transpose=False,
    quant_scheme="wrapX",
    disable_dma_transpose=False,
    weights_is_f_by_k=False,
    input_is_f_by_k=False,
    psum_copy_in_ratio=None,
    strided_copy_swizzle_1x32_ratio=None,
):
    """Compute golden backward pass: dX = dY @ W, dW = dY^T @ X.

    disable_dma_transpose is a load-mechanism choice with identical numerics, so this
    reference accepts it for signature parity and ignores it.

    Delegates each phase to the forward matmul golden so the MXFP8 quantization
    error is modelled the same way the kernel produces it. Computing the two
    matmuls in FP32 here instead would make every quantized arm fail on
    quantization error rather than on a real defect.
    """
    from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import matmul_mxfp8_torch_ref

    def _phase(lhs, rhs, lhs_scales, rhs_scales, lhs_swizzled, rhs_swizzled, lhs_f_by_k, rhs_f_by_k, config):
        return matmul_mxfp8_torch_ref(
            lhs=lhs,
            rhs=rhs,
            lhs_scales=lhs_scales,
            rhs_scales=rhs_scales,
            lhs_is_swizzled=lhs_swizzled,
            rhs_is_swizzled=rhs_swizzled,
            lhs_orientation=None if lhs_f_by_k else TensorOrientation.K_BY_F,
            rhs_orientation=None if rhs_f_by_k else TensorOrientation.K_BY_F,
            TILES_IN_BLOCK_M=config.TILES_IN_BLOCK_M if config else None,
            TILES_IN_BLOCK_N=config.TILES_IN_BLOCK_N if config else None,
            TILES_IN_BLOCK_K=config.TILES_IN_BLOCK_K if config else None,
            TILES_IN_LOAD_M=config.TILES_IN_LOAD_M if config else None,
            TILES_IN_LOAD_N=config.TILES_IN_LOAD_N if config else None,
            tile_loop_order=tile_loop_order,
            float8_dtype=float8_dtype,
            output_dtype=output_dtype,
            run_with_lnc2=run_with_lnc2,
            lnc_2_shard_rhs=lnc_2_shard_rhs,
            use_scale_packing=use_scale_packing,
            spill_reload=spill_reload,
            fast_dma_transpose=fast_dma_transpose,
            quant_scheme=quant_scheme,
        )["out"]

    # Layout flags mirror matmul_mxfp8_backward exactly: a swizzled operand is
    # already stored K-by-F and must not also claim an explicit K_BY_F orientation.
    input_grad = _phase(
        output_grad,
        weights,
        output_grad_scales,
        weight_scales,
        output_grad_is_swizzled,
        weights_is_swizzled,
        True,
        weights_is_swizzled or weights_is_f_by_k,
        input_grad_config,
    )
    weight_grad = _phase(
        output_grad,
        input_activation,
        output_grad_scales,
        input_scales,
        output_grad_is_swizzled,
        input_is_swizzled,
        output_grad_is_swizzled,
        input_is_swizzled or input_is_f_by_k,
        weight_grad_config,
    )
    return {"input_grad": input_grad, "weight_grad": weight_grad}


# ============================================================================
# Test configurations
# ============================================================================

BACKWARD_CONFIGS = [
    (512, 512, 512, "Square_small"),
    (1024, 1024, 1024, "Square_medium"),
    (2048, 1024, 512, "Nonsquare_M_gt_N"),
    (512, 1024, 2048, "Nonsquare_N_gt_M"),
    # Partial-F regression for the K-by-F loader. N_2560 (%512 but partial vs LOAD_TILE_F) caught
    # the original OOB; N_768/N_2304 were rejected by the old %512 guard.
    (512, 512, 768, "N_768_ctrl"),
    (512, 768, 512, "K_768_ctrl"),
    (512, 512, 2304, "N_2304_partial"),
    (512, 512, 2560, "N_2560_partial"),
    # Sub-128 partial-F via K (a free/output dim of dW/dX). N is not used here: in backward N is
    # the dX contraction dim and independently needs %128; sub-128 free dims are covered forward.
    (512, 520, 512, "K_520_subtile"),
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


@pytest.mark.fast
@pytest.mark.parametrize("ratio", [0.0, 0.25, 0.5, 1.0], ids=["off", "quarter", "half", "all"])
def test_backward_feature_ratios(test_manager, platform_target, ratio):
    """Exercise both standalone backward phases, including a split K tail."""
    from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
    from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import LncShardingMode, QuantScheme

    if not platform_target.is_trn3():
        pytest.skip("MXFP8 requires TRN3.")
    M, K, N = 896, 128, 1536
    rng = np.random.RandomState(42)

    def phase_config(m, k, n):
        return MatmulMxfp8KernelConfig(
            m,
            k,
            n,
            tile_m=128,
            tile_k=512,
            tile_n=128,
            TILES_IN_BLOCK_M=1,
            TILES_IN_BLOCK_N=1,
            TILES_IN_BLOCK_K=1,
            TILES_IN_LOAD_M=1,
            TILES_IN_LOAD_N=1,
            quant_scheme=QuantScheme._1x32,
            lnc_sharding=LncShardingMode.OFF,
        )

    kwargs = {
        "output_grad": rng.normal(size=(M, N)).astype(nl.bfloat16),
        "weights": rng.normal(size=(N, K)).astype(nl.bfloat16),
        "input_activation": rng.normal(size=(M, K)).astype(nl.bfloat16),
        "input_grad_config": phase_config(M, N, K),
        "weight_grad_config": phase_config(N, M, K),
        "output_dtype": nl.float32,
        "run_with_lnc2": False,
        "quant_scheme": "1x32",
        "float8_dtype": "float8_e4m3fn",
        "disable_dma_transpose": True,
        "psum_copy_in_ratio": ratio,
        "strided_copy_swizzle_1x32_ratio": ratio,
    }
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=matmul_mxfp8_generic_backward_kernel.matmul_mxfp8_backward,
        torch_ref=matmul_mxfp8_backward_torch_ref,
        kernel_input_generator=lambda _: kwargs,
        output_tensor_descriptor=lambda _: {
            "input_grad": np.zeros((M, K), dtype=np.float32),
            "weight_grad": np.zeros((N, K), dtype=np.float32),
        },
    )
    framework.run_test(
        None,
        compiler_args=common_dataclasses.CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        custom_comparator=_bwd_comparator,
    )
