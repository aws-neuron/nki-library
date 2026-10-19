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
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._common_utils import (
    create_and_set_active_sbm,
    get_active_sbm,
    with_active_sbm,
)
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils._quantize_mxfp8_utils import (
    QUANT_SCHEME_1X32,
    QUANT_SCHEME_WRAPX,
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
    quant_scheme: str,
):
    """Combined kernel: quantize LHS only, RHS is BF16 swizzled (matmul quantizes internally)."""

    if get_active_sbm() is None:
        create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST QUANT + MM")

    lhs_scales_hbm, lhs_data_hbm = quantize_block_mxfp8_kernel(lhs_bf16, return_fp8_dtype, run_with_lnc2, quant_scheme)

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
        quant_scheme=quant_scheme,
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
    quant_scheme: str,
):
    """Combined kernel: quantize RHS only, LHS is BF16 swizzled (matmul quantizes internally)."""
    if get_active_sbm() is None:
        create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST QUANT + MM")

    rhs_scales_hbm, rhs_data_hbm = quantize_block_mxfp8_kernel(rhs_bf16, return_fp8_dtype, run_with_lnc2, quant_scheme)

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
        quant_scheme=quant_scheme,
    )

    sbm.close_scope()
    return output


@nki.jit
@with_active_sbm
def quantize_matmul_fast_dgt_bf16_pipeline_kernel(
    lhs_f_by_k,
    rhs_f_by_k,
    quantize_lhs: bool,
    return_fp8_dtype: str,
    run_with_lnc2: bool,
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
    quant_scheme: str,
):
    """Quantize one operand; the matmul loads the other as unswizzled BF16 via the fast DMA transpose.

    Both inputs are F-by-K ([M, K] and [N, K]), as the quantize kernel and the DGT loader take them.
    """
    if get_active_sbm() is None:
        create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("TEST QUANT + FAST DGT MM")

    quantized_src = lhs_f_by_k if quantize_lhs else rhs_f_by_k
    scales_hbm, data_hbm = quantize_block_mxfp8_kernel(quantized_src, return_fp8_dtype, run_with_lnc2, quant_scheme)
    if quantize_lhs:
        operands = {"lhs": data_hbm, "lhs_scales": scales_hbm, "rhs": rhs_f_by_k, "rhs_is_swizzled": False}
    else:
        operands = {"lhs": lhs_f_by_k, "rhs": data_hbm, "rhs_scales": scales_hbm, "lhs_is_swizzled": False}

    output = matmul_mxfp8_generic_kernel.matmul_mxfp8(
        **operands,
        fast_dma_transpose=True,
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
        quant_scheme=quant_scheme,
    )

    sbm.close_scope()
    return output


def quantize_matmul_fast_dgt_bf16_pipeline_torch_ref(
    lhs_f_by_k,
    rhs_f_by_k,
    quantize_lhs,
    return_fp8_dtype,
    run_with_lnc2,
    TILES_IN_BLOCK_M,
    TILES_IN_BLOCK_N,
    TILES_IN_BLOCK_K,
    TILES_IN_LOAD_M,
    TILES_IN_LOAD_N,
    lhs_matmul_tile_shape_logical,
    rhs_matmul_tile_shape_logical,
    block_loop_order,
    tile_loop_order,
    output_dtype,
    float8_dtype,
    use_scale_packing,
    spill_reload,
    quant_scheme,
):
    """Golden: both operands in the shared wrapX layout (decompose_k_remainder), then MX matmul."""
    kwargs = {
        "return_fp8_dtype": return_fp8_dtype,
        "output_dtype": output_dtype,
        "float8_dtype": float8_dtype,
        "quant_scheme": quant_scheme,
    }
    if quantize_lhs:
        return quantize_lhs_matmul_pipeline_torch_ref(lhs_f_by_k, matmul_utils.swizzle_tensor(rhs_f_by_k.T), **kwargs)
    return quantize_rhs_matmul_pipeline_torch_ref(matmul_utils.swizzle_tensor(lhs_f_by_k.T), rhs_f_by_k, **kwargs)


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
    lnc_degree: int,
    use_scale_packing: bool,
    seed: int = 42,
    spill_reload: bool = False,
    quant_scheme: str = QUANT_SCHEME_WRAPX,
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

    # Swizzled BF16 in the quantize kernel's layout, down to its K-remainder tiling (see
    # _K_DIV_32_WRAPX_NOTE).
    swizzle = matmul_utils.swizzle_tensor_1x32 if quant_scheme == QUANT_SCHEME_1X32 else matmul_utils.swizzle_tensor

    lhs_sw = swizzle(lhs_bf16.T.copy())  # (K, M)
    rhs_sw = swizzle(rhs_bf16.copy())  # (K, N)

    output_dtype = nl.float32

    # ---- Matmul tile config ----
    conf = config_helper.TestConfig(M=M, K=K, N=N, tile_k=512, seed=seed, run_with_lnc2=lnc_degree > 1)
    conf = conf.autoGenerateRandomSubset(1)[0]

    run_with_lnc2 = lnc_degree > 1

    # Common matmul config args shared by all pipeline kernel variants
    matmul_config = {
        "return_fp8_dtype": fp8_str,
        "run_with_lnc2": run_with_lnc2,
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
        "quant_scheme": quant_scheme,
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

# K aligned to MX_PARTITION_SIZE (32) but not to L_TILE_K (512), so the quantize kernel emits a
# 256 / 128 / sub-128 remainder decomposition. The comment gives that decomposition.
_K_DIV_32_SHAPES = [
    pytest.param(512, 544, 512, marks=pytest.mark.fast),  # 512 + 32
    (512, 640, 512),  # 512 + 128
    (1024, 1696, 512),  # 3x512 + 128 + 32
    (512, 2016, 1024),  # 3x512 + 256 + 128 + 96
]

_K_DIV_32_WRAPX_NOTE = """A wrapX K remainder has more than one valid layout, because wrapX ties a
(partition, slot) to a logical K through the *producer's* tile size:

    k = k_offset + p * (tile_k / 4) + i

A 160-element remainder can therefore be laid out as one 160-wide tile (sub-tile stride 40) or as
128 + 32 (strides 32 and 8). Both are self-consistent; they are not interchangeable.
quantize_block_mxfp8_kernel picks the second via decompose_k_remainder, since quantize_mx caps a
tile at 128 partitions and needs the count to be a multiple of 32.

nc_matmul_mx contracts its two operands by (partition, slot) position, so every operand paired
against a quantize-kernel output has to use the same decomposition: swizzle_tensor for a
pre-swizzled operand, and the matmul's BF16 loaders (legacy and fast DMA transpose) for an
unswizzled one. Getting it wrong is not a subtle numerical drift: it silently contracts mismatched
K and the result is ~100% off.

1x32 is immune, since k = 4 * row + p holds regardless of tiling."""

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
        use_scale_packing: bool,
        fp8_str: str = "float8_e4m3fn",
        lnc_degree: int = 1,
        seed: int = 42,
        spill_reload: bool = False,
        quant_scheme: str = QUANT_SCHEME_WRAPX,
        bf16_fast_dgt: bool = False,
    ):
        """Execute the combined quantize->matmul pipeline and validate.

        bf16_fast_dgt passes the non-quantized operand unswizzled, loaded by the matmul's fast DMA
        transpose, instead of pre-swizzled.
        """
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
            lnc_degree,
            use_scale_packing,
            seed,
            spill_reload,
            quant_scheme,
        )

        torch_ref = quantize_lhs_matmul_pipeline_torch_ref if quantize_lhs else quantize_rhs_matmul_pipeline_torch_ref
        if bf16_fast_dgt:
            # Same values as _build_pipeline_inputs, as F-by-K BF16 for the quantize kernel and DGT.
            rng = np.random.RandomState(seed)
            lhs_bf16 = rng.uniform(-1, 1, (M, K)).astype(nl.bfloat16)
            rhs_bf16 = rng.uniform(-1, 1, (K, N)).astype(nl.bfloat16)
            operand_keys = ("lhs_bf16", "rhs_sw", "lhs_sw", "rhs_bf16")
            kernel_input = {
                "lhs_f_by_k": lhs_bf16,
                "rhs_f_by_k": rhs_bf16.T.copy(),
                "quantize_lhs": quantize_lhs,
                **{key: value for key, value in kernel_input.items() if key not in operand_keys},
            }
            pipeline_kernel = quantize_matmul_fast_dgt_bf16_pipeline_kernel
            torch_ref = quantize_matmul_fast_dgt_bf16_pipeline_torch_ref

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
    @pytest.mark.parametrize("use_scale_packing", [True, False], ids=["packed", "unpacked"])
    @pytest.mark.parametrize("spill_reload", [True, False])
    def test_lhs_prequantized_rhs_bf16(self, test_manager, platform_target, M, K, N, use_scale_packing, spill_reload):
        """LHS is BF16 swizzled (matmul quantizes internally) + RHS(BF16) → quantize RHS → matmul."""
        self._run_pipeline(
            test_manager,
            platform_target,
            M=M,
            K=K,
            N=N,
            quantize_lhs=False,
            quantize_rhs=True,
            use_scale_packing=use_scale_packing,
            spill_reload=spill_reload,
        )

    # ------------------------------------------------------------------
    # Case 2: RHS pre-quantized, LHS BF16 → quantize LHS → matmul
    #         (with and without scale packing)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("M,K,N", _SMALL_SHAPES + _MEDIUM_SHAPES + _LARGE_SHAPES)
    @pytest.mark.parametrize("use_scale_packing", [True, False], ids=["packed", "unpacked"])
    @pytest.mark.parametrize("spill_reload", [True, False])
    def test_rhs_prequantized_lhs_bf16(self, test_manager, platform_target, M, K, N, use_scale_packing, spill_reload):
        """RHS is BF16 swizzled (matmul quantizes internally) + LHS(BF16) → quantize LHS → matmul."""
        self._run_pipeline(
            test_manager,
            platform_target,
            M=M,
            K=K,
            N=N,
            quantize_lhs=True,
            quantize_rhs=False,
            use_scale_packing=use_scale_packing,
            spill_reload=spill_reload,
        )

    # ------------------------------------------------------------------
    # Case 3: the same two directions under the 1x32 layout
    # ------------------------------------------------------------------
    @pytest.mark.fast
    @pytest.mark.parametrize("M,K,N", _SMALL_SHAPES + [_MEDIUM_SHAPES[0]] + _LARGE_SHAPES)
    @pytest.mark.parametrize("use_scale_packing", [True, False], ids=["packed", "unpacked"])
    @pytest.mark.parametrize("quantize_lhs", [True, False], ids=["quant_lhs", "quant_rhs"])
    def test_1x32_pipeline(self, test_manager, platform_target, M, K, N, use_scale_packing, quantize_lhs):
        """quantize(1x32) → matmul(1x32). Both operands carry the 1x32 layout."""
        self._run_pipeline(
            test_manager,
            platform_target,
            M=M,
            K=K,
            N=N,
            quantize_lhs=quantize_lhs,
            quantize_rhs=not quantize_lhs,
            use_scale_packing=use_scale_packing,
            quant_scheme=QUANT_SCHEME_1X32,
        )

    # ------------------------------------------------------------------
    # Case 4: K % 32 (but not % 512) end-to-end under 1x32
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("M,K,N", _K_DIV_32_SHAPES)
    @pytest.mark.parametrize("use_scale_packing", [True, False], ids=["packed", "unpacked"])
    @pytest.mark.parametrize("quantize_lhs", [True, False], ids=["quant_lhs", "quant_rhs"])
    def test_1x32_pipeline_k_div_32(self, test_manager, platform_target, M, K, N, use_scale_packing, quantize_lhs):
        """quantize(1x32) → matmul(1x32) for K % 32 != 0 mod 512.

        The quantize kernel splits K into 512 + 256/128/sub-128 tiles and quantizes any sub-128
        tile as a padded 32-partition tile. Because the 1x32 mapping k = 4*row + p is independent
        of tile_k, the matmul reads every one of those tiles back at the right K.
        """
        self._run_pipeline(
            test_manager,
            platform_target,
            M=M,
            K=K,
            N=N,
            quantize_lhs=quantize_lhs,
            quantize_rhs=not quantize_lhs,
            use_scale_packing=use_scale_packing,
            quant_scheme=QUANT_SCHEME_1X32,
        )

    # ------------------------------------------------------------------
    # Case 5: K % 32 end-to-end under wrapX
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("M,K,N", _K_DIV_32_SHAPES)
    @pytest.mark.parametrize("use_scale_packing", [True, False], ids=["packed", "unpacked"])
    @pytest.mark.parametrize("quantize_lhs", [True, False], ids=["quant_lhs", "quant_rhs"])
    def test_wrapx_pipeline_k_div_32(self, test_manager, platform_target, M, K, N, use_scale_packing, quantize_lhs):
        """quantize(wrapX) → matmul(wrapX) for K % 32 != 0 mod 512.

        See _K_DIV_32_WRAPX_NOTE: wrapX admits more than one valid layout for a K remainder, so
        the pre-swizzled operand is built with the quantize kernel's own decomposition.
        """
        self._run_pipeline(
            test_manager,
            platform_target,
            M=M,
            K=K,
            N=N,
            quantize_lhs=quantize_lhs,
            quantize_rhs=not quantize_lhs,
            use_scale_packing=use_scale_packing,
            quant_scheme=QUANT_SCHEME_WRAPX,
        )

    # ------------------------------------------------------------------
    # Case 6: K % 32 under wrapX against an unswizzled BF16 operand on the fast DMA transpose
    # ------------------------------------------------------------------
    @pytest.mark.parametrize(
        "M,K,N",
        [pytest.param(128, 672, 512, marks=pytest.mark.fast)] + _K_DIV_32_SHAPES,  # 512 + 128 + 32
    )
    @pytest.mark.parametrize("quantize_lhs", [True, False], ids=["quant_lhs", "quant_rhs"])
    @pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN3_A0, common_dataclasses.Platforms.TRN3_PDS_A0])
    def test_wrapx_k_div_32_fast_dgt_bf16(self, test_manager, platform_target, M, K, N, quantize_lhs):
        """quantize(wrapX) -> matmul with the other operand as unswizzled BF16 on the fast DMA transpose.

        The fast DMA loader must split the K remainder like the quantize kernel (see
        _K_DIV_32_WRAPX_NOTE); a single remainder tile pairs mismatched K. Fast DMA transpose is
        B1-only, so A0 is excluded.
        """
        self._run_pipeline(
            test_manager,
            platform_target,
            M=M,
            K=K,
            N=N,
            quantize_lhs=quantize_lhs,
            quantize_rhs=not quantize_lhs,
            use_scale_packing=False,
            quant_scheme=QUANT_SCHEME_WRAPX,
            bf16_fast_dgt=True,
        )
