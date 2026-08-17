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

"""Integration tests for quantize_mxfp8 -> matmul_mxfp8 end-to-end pipeline.

Validates that the output of the quantize kernel can be consumed directly
by the matmul kernel and produces correct results. Each test runs a single
combined kernel that:
  1. Quantizes one or both BF16 operands using quantize_block_mxfp8_kernel.
  2. Feeds the actual quantize kernel HBM outputs (data + scales) directly
     into matmul_mxfp8 as pre-quantized inputs.
  3. Returns the matmul result for validation against the CPU golden.

This proves true format compatibility between the kernels, including
scale packing layouts that differ between the kernel output and the
CPU golden reference.
"""

from typing import Any

import nki
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.experimental.matmul_mxfp8 import matmul_mxfp8_generic_kernel
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_torch import (
    quantize_lhs_matmul_pipeline_torch_ref,
    quantize_rhs_matmul_pipeline_torch_ref,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_utils import (
    create_and_set_active_sbm,
    get_active_sbm,
    with_active_sbm,
)
from nkilib_src.nkilib.experimental.quantize_mxfp8.quantize_mxfp8 import (
    quantize_block_mxfp8_kernel,
)
from typing_extensions import override

from test.integration.nkilib.experimental.matmul_mxfp8 import config_helper
from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.utils import common_dataclasses, test_orchestrator
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework


@nki.jit
@with_active_sbm
def quantize_lhs_matmul_pipeline_kernel(
    lhs_bf16,
    rhs_sw,
    return_fp8_dtype: str,
    run_with_lnc2: bool,
    enable_scale_packing: bool,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_N: int,
    TILES_IN_BLOCK_K: int,
    TILES_IN_LOAD_M: int,
    TILES_IN_LOAD_N: int,
    lhs_matmul_tile_shape_logical: tuple,
    rhs_matmul_tile_shape_logical: tuple,
    block_loop_order: str,
    tile_loop_order: str,
    output_dtype,
    float8_dtype: str,
    use_scale_packing: bool,
    spill_reload: bool,
):
    """Combined kernel: quantize LHS only, RHS is BF16 swizzled (matmul quantizes internally)."""

    if get_active_sbm() is None:
        create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST QUANT + MM")

    lhs_scales_hbm, lhs_data_hbm = quantize_block_mxfp8_kernel(
        lhs_bf16, return_fp8_dtype, run_with_lnc2, enable_scale_packing
    )

    output = matmul_mxfp8_generic_kernel.matmul_mxfp8(
        lhs=lhs_data_hbm,
        rhs=rhs_sw,
        lhs_scales=lhs_scales_hbm,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_logical=lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical=rhs_matmul_tile_shape_logical,
        block_loop_order=block_loop_order,
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        output_dtype=output_dtype,
        run_with_lnc2=run_with_lnc2,
        use_scale_packing=use_scale_packing,
        spill_reload=spill_reload,
    )

    sbm.close_scope()
    return output


@nki.jit
@with_active_sbm
def quantize_rhs_matmul_pipeline_kernel(
    lhs_sw,
    rhs_bf16,
    return_fp8_dtype: str,
    run_with_lnc2: bool,
    enable_scale_packing: bool,
    TILES_IN_BLOCK_M: int,
    TILES_IN_BLOCK_N: int,
    TILES_IN_BLOCK_K: int,
    TILES_IN_LOAD_M: int,
    TILES_IN_LOAD_N: int,
    lhs_matmul_tile_shape_logical: tuple,
    rhs_matmul_tile_shape_logical: tuple,
    block_loop_order: str,
    tile_loop_order: str,
    output_dtype,
    float8_dtype: str,
    use_scale_packing: bool,
    spill_reload: bool,
):
    """Combined kernel: quantize RHS only, LHS is BF16 swizzled (matmul quantizes internally)."""
    if get_active_sbm() is None:
        create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST QUANT + MM")

    rhs_scales_hbm, rhs_data_hbm = quantize_block_mxfp8_kernel(
        rhs_bf16, return_fp8_dtype, run_with_lnc2, enable_scale_packing
    )

    output = matmul_mxfp8_generic_kernel.matmul_mxfp8(
        lhs=lhs_sw,
        rhs=rhs_data_hbm,
        rhs_scales=rhs_scales_hbm,
        TILES_IN_BLOCK_M=TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N=TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K=TILES_IN_BLOCK_K,
        TILES_IN_LOAD_M=TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_logical=lhs_matmul_tile_shape_logical,
        rhs_matmul_tile_shape_logical=rhs_matmul_tile_shape_logical,
        block_loop_order=block_loop_order,
        tile_loop_order=tile_loop_order,
        float8_dtype=float8_dtype,
        output_dtype=output_dtype,
        run_with_lnc2=run_with_lnc2,
        use_scale_packing=use_scale_packing,
        spill_reload=spill_reload,
    )

    sbm.close_scope()
    return output


# ============================================================================
# Input construction helpers
# ============================================================================


def _build_pipeline_inputs(
    M: int,
    K: int,
    N: int,
    fp8_str: str,
    quantize_lhs: bool,
    quantize_rhs: bool,
    enable_scale_packing: bool,
    lnc_degree: int,
    use_scale_packing: bool,
    seed: int = 42,
    spill_reload: bool = False,
):
    """Return the combined pipeline kernel function and its inputs.

    Returns
    -------
    pipeline_kernel : callable
        The combined @nki.jit kernel to execute.
    kernel_input : dict
        Arguments for the combined kernel.
    output_dtype : nl.dtype
    M, N : int
        Output dimensions.
    """
    rng = np.random.RandomState(seed)
    lhs_bf16 = rng.uniform(-1, 1, (M, K)).astype(nl.bfloat16)
    rhs_bf16 = rng.uniform(-1, 1, (K, N)).astype(nl.bfloat16)

    # Swizzled BF16 (same layout the matmul golden uses)
    lhs_sw = matmul_utils.swizzle_tensor(lhs_bf16.T.copy())  # (K, M)
    rhs_sw = matmul_utils.swizzle_tensor(rhs_bf16.copy())  # (K, N)

    output_dtype = nl.float32

    # ---- Matmul tile config ----
    conf = config_helper.TestConfig(M=M, K=K, N=N, tile_k=512, seed=seed, run_with_lnc2=lnc_degree > 1)
    conf = conf.autoGenerateRandomSubset(1)[0]

    run_with_lnc2 = lnc_degree > 1

    # Common matmul config args shared by all pipeline kernel variants
    matmul_config = {
        "return_fp8_dtype": fp8_str,
        "run_with_lnc2": run_with_lnc2,
        "enable_scale_packing": enable_scale_packing,
        "TILES_IN_BLOCK_M": conf.TILES_IN_BLOCK_M,
        "TILES_IN_BLOCK_N": conf.TILES_IN_BLOCK_N,
        "TILES_IN_BLOCK_K": conf.TILES_IN_BLOCK_K,
        "TILES_IN_LOAD_M": conf.TILES_IN_LOAD_M,
        "TILES_IN_LOAD_N": conf.TILES_IN_LOAD_N,
        "lhs_matmul_tile_shape_logical": (conf.tile_k, conf.tile_m),
        "rhs_matmul_tile_shape_logical": (conf.tile_k, conf.tile_n),
        "block_loop_order": conf.block_loop_order,
        "tile_loop_order": conf.tile_loop_order,
        "output_dtype": output_dtype,
        "float8_dtype": fp8_str,
        "use_scale_packing": use_scale_packing,
        "spill_reload": spill_reload,
    }

    if quantize_lhs and not quantize_rhs:
        # Only LHS quantized, RHS is BF16 swizzled
        pipeline_kernel = quantize_lhs_matmul_pipeline_kernel
        kernel_input = {
            "lhs_bf16": lhs_bf16.copy(),  # (M, K)
            "rhs_sw": rhs_sw,  # (K, N) swizzled BF16
            **matmul_config,
        }
    elif not quantize_lhs and quantize_rhs:
        # Only RHS quantized, LHS is BF16 swizzled
        pipeline_kernel = quantize_rhs_matmul_pipeline_kernel
        kernel_input = {
            "lhs_sw": lhs_sw,  # (K, M) swizzled BF16
            "rhs_bf16": rhs_bf16.T.copy(),  # (N, K)
            **matmul_config,
        }
    else:
        raise ValueError("At least one operand must be quantized for the pipeline test.")

    return pipeline_kernel, kernel_input, output_dtype, M, N


# ============================================================================
# Test parameters
# ============================================================================

# (M, K, N) triples – kept small for fast CI turnaround
_SMALL_SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
]

_MEDIUM_SHAPES = [
    (1024, 2048, 512),
    (2048, 2048, 2048),
]

_LARGE_SHAPES = [(1024, 3072, 3072)]


# ============================================================================
# Test class
# ============================================================================


@pytest_test_metadata(name="Quantize - Matmul MXFP8 Pipeline")
@pytest_marks(["quantize_matmul_mxfp8_pipeline"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestQuantizeMatmulPipeline:
    """End-to-end pipeline: quantize_mxfp8 ➜ matmul_mxfp8.

    Each test runs a single combined kernel that quantizes operand(s) and
    feeds the actual HW quantize outputs directly into the matmul kernel.
    """

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------
    def _run_pipeline(
        self,
        test_manager: test_orchestrator.Orchestrator,
        platform_target,
        M: int,
        K: int,
        N: int,
        quantize_lhs: bool,
        quantize_rhs: bool,
        enable_scale_packing: bool,
        use_scale_packing: bool,
        fp8_str: str = "float8_e4m3fn",
        lnc_degree: int = 1,
        seed: int = 42,
        spill_reload: bool = False,
    ):
        """Execute the combined quantize->matmul pipeline and validate."""
        if not platform_target.is_trn3():
            pytest.skip("MX is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            logical_nc_config=lnc_degree,
            platform_target=platform_target,
        )

        (
            pipeline_kernel,
            kernel_input,
            output_dtype,
            out_M,
            out_N,
        ) = _build_pipeline_inputs(
            M,
            K,
            N,
            fp8_str,
            quantize_lhs,
            quantize_rhs,
            enable_scale_packing,
            lnc_degree,
            use_scale_packing,
            seed,
            spill_reload,
        )

        torch_ref = quantize_lhs_matmul_pipeline_torch_ref if quantize_lhs else quantize_rhs_matmul_pipeline_torch_ref

        def _mxfp8_comparator(golden_dict, output_tensors):
            """Wrap torch ref golden with MXFP8 check_correctness validation."""
            golden = golden_dict["out"]

            class _PipelineValidator(common_dataclasses.CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]) -> bool:
                    hw = inference_output.view(dtype=output_dtype).astype(output_dtype).reshape(golden.shape)
                    passed, metrics = matmul_utils.check_correctness(hw, golden.astype(output_dtype))
                    if not passed:
                        self._print_with_log("Pipeline matmul validation FAILED")
                        self._print_with_log(f"  metrics: {metrics}")
                        self._print_with_log(f"  HW[0,:5]     = {hw[0, :5]}")
                        self._print_with_log(f"  Golden[0,:5]  = {golden[0, :5]}")
                    return passed

            return {
                "out": common_dataclasses.CustomValidatorWithOutputTensorData(
                    validator=_PipelineValidator,
                    output_ndarray=np.ndarray((out_M, out_N), dtype=output_dtype),
                )
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=pipeline_kernel,
            torch_ref=torch_ref,
            kernel_input_generator=lambda _: kernel_input,
            output_tensor_descriptor=lambda _: {"out": np.zeros((out_M, out_N), dtype=output_dtype)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            custom_comparator=_mxfp8_comparator,
        )

    # ------------------------------------------------------------------
    # Case 1: LHS pre-quantized, RHS BF16 → quantize RHS → matmul
    #         (with and without scale packing)
    # ------------------------------------------------------------------
    @pytest.mark.fast
    @pytest.mark.parametrize("M,K,N", _SMALL_SHAPES + [_MEDIUM_SHAPES[0]] + _LARGE_SHAPES)
    @pytest.mark.parametrize("enable_scale_packing", [True, False], ids=["packed", "unpacked"])
    @pytest.mark.parametrize("spill_reload", [True, False])
    def test_lhs_prequantized_rhs_bf16(
        self, test_manager, platform_target, M, K, N, enable_scale_packing, spill_reload
    ):
        """LHS is BF16 swizzled (matmul quantizes internally) + RHS(BF16) → quantize RHS → matmul."""
        self._run_pipeline(
            test_manager,
            platform_target,
            M=M,
            K=K,
            N=N,
            quantize_lhs=False,
            quantize_rhs=True,
            enable_scale_packing=enable_scale_packing,
            use_scale_packing=enable_scale_packing,
            spill_reload=spill_reload,
        )

    # ------------------------------------------------------------------
    # Case 2: RHS pre-quantized, LHS BF16 → quantize LHS → matmul
    #         (with and without scale packing)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("M,K,N", _SMALL_SHAPES + _MEDIUM_SHAPES + _LARGE_SHAPES)
    @pytest.mark.parametrize("enable_scale_packing", [True, False], ids=["packed", "unpacked"])
    @pytest.mark.parametrize("spill_reload", [True, False])
    def test_rhs_prequantized_lhs_bf16(
        self, test_manager, platform_target, M, K, N, enable_scale_packing, spill_reload
    ):
        """RHS is BF16 swizzled (matmul quantizes internally) + LHS(BF16) → quantize LHS → matmul."""
        self._run_pipeline(
            test_manager,
            platform_target,
            M=M,
            K=K,
            N=N,
            quantize_lhs=True,
            quantize_rhs=False,
            enable_scale_packing=enable_scale_packing,
            use_scale_packing=enable_scale_packing,
            spill_reload=spill_reload,
        )
