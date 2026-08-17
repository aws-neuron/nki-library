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

import enum
import functools
from typing import Optional, TypedDict

import neuron_dtypes as dt
import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.output_projection.output_projection_tkg_torch import (
    output_projection_tkg_torch_ref,
)
from nkilib_src.nkilib.core.utils.allocator import BufferManager, Logger
from nkilib_src.nkilib.core.utils.common_types import DtypeMode, QuantizationType
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert
from nkilib_src.nkilib.experimental.primitives.output_projection.output_projection_tkg import (
    output_projection_tkg,
)

from test.integration.nkilib.utils.tensor_generators import (
    FP8_E4M3_MAX,
    gaussian_tensor_generator,
    generate_stabilized_mx_data,
    np_random_sample,
    static_cast,
)
from test.integration.nkilib.utils.test_kernel_common import resolve_dtype_mode_for_torch_ref
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
    Platforms,
)
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Hardware constants - must match values in output_projection_tkg.py
F_MAX = 512  # Free dimension size for PSUM/GEMM operations
P_MAX = 128  # Partition dimension size


class TkgOutputProjClassification(enum.Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    @staticmethod
    def classify(B: int, n_heads: int, S_tkg: int, d_head: int, H: int):
        flops_estimate = B * n_heads * S_tkg * d_head * H

        # TODO: choose proper classification thresholds.
        if flops_estimate <= 130000000000:
            return TkgOutputProjClassification.SMALL
        elif flops_estimate <= 626000000000:
            return TkgOutputProjClassification.MEDIUM
        else:
            return TkgOutputProjClassification.LARGE

    def __str__(self):
        return self.name


@nki.jit
def output_proj_tkg_wrapper(
    attention: nl.ndarray,
    weight: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[nl.ndarray] = None,
    input_scale: Optional[nl.ndarray] = None,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
    # Placeholder param to match torch-ref/kernel signature
    sbm: Optional[BufferManager] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> nl.ndarray:
    sbm = BufferManager(
        0,
        nl.tile_size.total_available_sbuf_size,
        logger=Logger("test_output_projection_tkg"),
        use_auto_alloc=False,
    )

    return output_projection_tkg(
        attention,
        weight,
        bias,
        quantization_type,
        weight_scale,
        input_scale,
        TRANSPOSE_OUT,
        OUT_IN_SB,
        sbm,
        dtype_mode=dtype_mode,
    )


def build_output_proj_tkg_input(
    lnc_degree,
    d_head,
    B,
    n_heads,
    S_tkg,
    quantization_type,
    H,
    test_bias,
    transpose_out,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
):
    dtype = nl.bfloat16
    _q_width = 4

    # Pick E4M3 dtype + clip according to the declared dtype_mode. STATIC_MX is always OCP.
    _fp8_e4m3_dtype = nl.float8_e4m3fn if dtype_mode == DtypeMode.OCP else nl.float8_e4m3
    _fp8_e4m3_max = 448.0 if dtype_mode == DtypeMode.OCP else FP8_E4M3_MAX

    if quantization_type == QuantizationType.STATIC_MX:
        FP8_E4M3FN_MAX = 448.0
        N_D = d_head * n_heads
        random_gen = np_random_sample()

        def convert_to_range(max_val, unif_tensor):
            return (2 * max_val * unif_tensor - max_val).astype(unif_tensor.dtype)

        attention = convert_to_range(1, random_gen(shape=(d_head, B, n_heads, S_tkg), dtype=dtype, name="attention"))
        in_scale = (np.abs(attention).max() / FP8_E4M3FN_MAX) * np.random.uniform(0.995, 1.005)

        # Generate scalar fp8 weights, then repack to x4 matching generate_stabilized_mx_data convention.
        weight_bf16 = convert_to_range(1, random_gen(shape=(N_D, H), dtype=dtype, name="weight"))
        w_scale = np.abs(weight_bf16).max() / FP8_E4M3FN_MAX
        weight_scalar_fp8 = static_cast(weight_bf16 / w_scale, nl.float8_e4m3fn)
        # Pack: [N_D, H] → [N_D//4, 4, H] → transpose → [N_D//4, H, 4] → flatten → [N_D//4, H*4] → x4 → [N_D//4, H]
        flat = (
            weight_scalar_fp8.reshape(N_D // _q_width, _q_width, H)
            .transpose(0, 2, 1)
            .reshape(N_D // _q_width, H * _q_width)
        )
        weight = dt.static_cast(flat.astype(np.float32), nl.float8_e4m3fn_x4)

        '''
        # Alternative: Using generate_stabilized_mx_data directly (commented out) ---
        # Chose to generate BF16 tensor and pack it manually to compare golden accuracy to the regualar STATIC golden.

         _, weights_qtz, _ = generate_stabilized_mx_data(
             mx_dtype=nl.float8_e4m3fn_x4,
             shape=(N_D // _q_width, H * _q_width),
        )
        base_scale = 1.0 / FP8_E4M3FN_MAX
        w_scale = base_scale * np.random.uniform(0.995, 1.005).astype(np.float32)
        weight = weights_qtz
        '''

        weight_scale = np.broadcast_to(np.array([[w_scale]], dtype=np.float32), (128, 1))
        input_scale = np.broadcast_to(np.array([[in_scale]], dtype=np.float32), (128, 1))
    elif quantization_type == QuantizationType.STATIC:
        random_gen = np_random_sample()

        def convert_to_range(max_val, unif_tensor):
            return (2 * max_val * unif_tensor - max_val).astype(unif_tensor.dtype)

        attention = convert_to_range(1, random_gen(shape=(d_head, B, n_heads, S_tkg), dtype=dtype, name="attention"))
        # Compute input_scale from actual max with small perturbation (simulates calibration)
        in_scale = (np.abs(attention).max() / _fp8_e4m3_max) * np.random.uniform(0.995, 1.005)

        weight_bf16 = convert_to_range(1, random_gen(shape=(d_head * n_heads, H), dtype=dtype, name="weight"))
        w_scale = np.abs(weight_bf16).max() / _fp8_e4m3_max
        weight = static_cast(weight_bf16 / w_scale, _fp8_e4m3_dtype)

        weight_scale = np.broadcast_to(np.array([[w_scale]], dtype=np.float32), (128, 1))
        input_scale = np.broadcast_to(np.array([[in_scale]], dtype=np.float32), (128, 1))
    elif quantization_type == QuantizationType.ROW:
        random_gen = np_random_sample()

        def convert_to_range(max_val, unif_tensor):
            return (2 * max_val * unif_tensor - max_val).astype(unif_tensor.dtype)

        attention = convert_to_range(1, random_gen(shape=(d_head, B, n_heads, S_tkg), dtype=dtype, name="attention"))

        weight_bf16 = convert_to_range(1, random_gen(shape=(d_head * n_heads, H), dtype=dtype, name="weight"))
        w_scale = np.abs(weight_bf16).max(axis=0, keepdims=True) / _fp8_e4m3_max
        weight = static_cast(weight_bf16 / w_scale, _fp8_e4m3_dtype)

        weight_scale = np.broadcast_to(w_scale.astype(np.float32), (128, H))
        input_scale = None
    else:  # QuantizationType.NONE
        random_gen = gaussian_tensor_generator()
        attention = random_gen(shape=(d_head, B, n_heads, S_tkg), dtype=dtype, name="attention")
        weight = random_gen(shape=(d_head * n_heads, H), dtype=dtype, name="weight")
        weight_scale = None
        input_scale = None

    bias = gaussian_tensor_generator(std=100)(shape=(1, H), dtype=dtype, name="bias") if test_bias else None
    if transpose_out:
        H0 = 128
        H1_sharded = H // H0 // lnc_degree
        kernel_assert(
            H1_sharded * lnc_degree * H0 == H,
            f"Output projection in attention block token-gen kernel requires the hidden dimension (H) to be a multiple of {H0} * LNC, where LNC = {lnc_degree}; but H = {H}.",
        )

    return {
        "attention": attention,
        "weight": weight,
        "bias": bias,
        "quantization_type": quantization_type,
        "weight_scale": weight_scale,
        "input_scale": input_scale,
        "TRANSPOSE_OUT": transpose_out,
        "OUT_IN_SB": False,
        "dtype_mode": dtype_mode,
    }


# Params in order:
#    B, n_heads, S_tkg, d_head, H, quantization_type, test_bias, transpose_out
OUTPUT_PROJ_TKG_TEST_CASES = (
    [512, 8, 4, 32, 3072, QuantizationType.NONE, True, True],
    [4, 8, 4, 32, 3072, QuantizationType.NONE, True, True],
    [4, 8, 4, 64, 3072, QuantizationType.NONE, False, False],
    [4, 8, 4, 64, 5376, QuantizationType.NONE, False, False],
    [4, 8, 4, 64, 5378, QuantizationType.NONE, False, False],
    [4, 2, 4, 64, 7168, QuantizationType.NONE, False, True],
    [4, 8, 4, 64, 3072, QuantizationType.NONE, True, False],
    [4, 2, 4, 64, 8192, QuantizationType.NONE, True, False],
    [4, 2, 4, 64, 16384, QuantizationType.NONE, True, False],
    [4, 2, 4, 128, 8192, QuantizationType.NONE, True, False],
    [4, 2, 4, 128, 16384, QuantizationType.NONE, True, False],
    [4, 4, 8, 128, 8192, QuantizationType.NONE, True, False],
    [4, 4, 8, 128, 16384, QuantizationType.NONE, True, False],
    [4, 8, 8, 128, 8192, QuantizationType.NONE, True, False],
    [4, 8, 8, 128, 16384, QuantizationType.NONE, True, False],
    [4, 10, 8, 128, 8192, QuantizationType.NONE, True, False],
    [4, 16, 8, 128, 16384, QuantizationType.NONE, True, False],
    [4, 8, 4, 64, 3072, QuantizationType.NONE, True, True],
    [4, 8, 4, 64, 8192, QuantizationType.NONE, True, True],
    [4, 8, 4, 64, 16384, QuantizationType.NONE, True, True],
    [4, 8, 4, 128, 16384, QuantizationType.NONE, True, True],
    [4, 8, 8, 128, 16384, QuantizationType.NONE, True, True],
    [4, 8, 5, 128, 10240, QuantizationType.NONE, True, True],
    [4, 8, 5, 128, 16384, QuantizationType.NONE, True, True],
    [16, 8, 1, 128, 8192, QuantizationType.STATIC, True, True],
    [16, 8, 1, 128, 8192, QuantizationType.STATIC, False, True],
    [16, 8, 1, 128, 8192, QuantizationType.STATIC, True, False],
    [8, 8, 1, 128, 8192, QuantizationType.STATIC, False, False],
    [64, 8, 5, 128, 8192, QuantizationType.STATIC, True, True],
    [32, 8, 3, 128, 3072, QuantizationType.STATIC, False, False],
    [128, 8, 3, 128, 3072, QuantizationType.STATIC, True, False],
    [192, 8, 3, 128, 3072, QuantizationType.NONE, True, False],
    [512, 8, 7, 128, 3072, QuantizationType.STATIC, True, False],
    [128, 8, 3, 128, 3072, QuantizationType.NONE, True, True],
    # double_row STATIC quantization tests (even n_heads >= 4, B*S >= 64)
    [32, 6, 3, 128, 3072, QuantizationType.STATIC, False, False],
    # double_row + transpose_out=True coverage
    [64, 8, 5, 128, 8192, QuantizationType.STATIC, False, True],  # no bias
    [32, 6, 3, 128, 3072, QuantizationType.STATIC, False, True],  # n_heads=6
    [128, 6, 1, 128, 3072, QuantizationType.STATIC, True, True],  # larger B, n_heads=6
    [16, 8, 1, 128, 3072, QuantizationType.ROW, True, False],
    [16, 8, 1, 128, 3072, QuantizationType.ROW, False, False],
    [16, 8, 1, 128, 3072, QuantizationType.ROW, True, True],
    [16, 8, 1, 128, 3072, QuantizationType.ROW, False, True],
    [128, 8, 1, 128, 3072, QuantizationType.ROW, True, False],
    [128, 8, 1, 128, 3072, QuantizationType.ROW, True, True],
    [256, 8, 1, 128, 3072, QuantizationType.ROW, True, False],
    [256, 8, 1, 128, 3072, QuantizationType.ROW, True, True],
    [512, 8, 1, 128, 3072, QuantizationType.ROW, True, False],
    [1024, 8, 1, 128, 3072, QuantizationType.ROW, True, False],
    [1024, 8, 1, 128, 3072, QuantizationType.ROW, True, True],
    [64, 8, 1, 128, 8192, QuantizationType.ROW, True, False],
    [64, 8, 1, 128, 8192, QuantizationType.ROW, True, True],
    # STATIC_MX quantization tests (requires H % 512 == 0, N*D % 4 == 0, TRN3 only)
    [32, 3, 4, 128, 4096, QuantizationType.STATIC_MX, True, False],
    [32, 1, 2, 128, 2048, QuantizationType.STATIC_MX, True, False],
    [64, 4, 2, 128, 3072, QuantizationType.STATIC_MX, True, False],
    [64, 4, 2, 128, 8192, QuantizationType.STATIC_MX, True, False],
    [64, 2, 1, 128, 4096, QuantizationType.STATIC_MX, False, False],
    [128, 8, 1, 128, 8192, QuantizationType.STATIC_MX, False, False],
    [128, 2, 1, 128, 8192, QuantizationType.STATIC_MX, False, False],
)

# Manual sweep cases: H=602 (not divisible by 128) for negative test coverage
# Old framework: output_proj_tkg_sweep_manual_config() generated these 2 cases
# with test_bias=True (default) and quantization_type=NONE (default)
MANUAL_PARAM_NAMES = "B, n_heads, S_tkg, d_head, H, quantization_type, test_bias, transpose_out"
MANUAL_TEST_CASES = [
    (4, 10, 8, 128, 602, QuantizationType.NONE, True, False),
    (4, 10, 8, 128, 602, QuantizationType.NONE, True, True),
]

PARAM_NAMES = "B, n_heads, S_tkg, d_head, H, quantization_type, test_bias, transpose_out"
_ABBREVS = {
    "n_heads": "nh",
    "S_tkg": "S",
    "d_head": "dh",
    "quantization_type": "qt",
    "test_bias": "bias",
    "transpose_out": "tp",
}


def _filter_sweep_params(B, n_heads=None, S_tkg=None, d_head=None, H=None, transpose_out=None):
    """Filter for coverage_parametrize sweep tests.

    Replicates the old RangeProductConstraintMonotonicStrategy configs which used
    separate strategies with most dims fixed. A combo is valid if it fits at least
    one of the old strategies:

    Random/Monotonic: B<=4, n_heads<=10, S_tkg<=8, d_head<=128, H mult 128
    Strategy 1 (tp=F): S_tkg*B<=4096, B<=512, S_tkg<=8, n_heads<=10, d_head<=128, H mult 128
    Strategy 2 (tp=F): n_heads*d_head<=4096, n_heads<=64, d_head<=128, B<=4, S_tkg<=8, H mult 128
    Strategy 3 (tp=T): S_tkg*B<=512, B<=128, S_tkg<=8, n_heads<=10, d_head<=128, H mult 128
    """
    if n_heads is None or S_tkg is None or d_head is None or H is None or transpose_out is None:
        return FilterResult.VALID

    if H % 128 != 0:
        return FilterResult.INVALID

    # Random/Monotonic strategy: all dims small
    if B <= 4 and n_heads <= 10:
        return FilterResult.VALID

    if not transpose_out:
        # Strategy 1: sweep S*B, fix N<=10, D<=128
        # Old config had D=128 fixed; we relax but require d_head >= 32 to avoid
        # compiler issues with very small d_head at large B (never tested in old config)
        if n_heads <= 10 and S_tkg * B <= 4096 and (B <= 4 or d_head >= 32):
            return FilterResult.VALID
        # Strategy 2: sweep N*D, fix B<=4, S<=8
        if B <= 4 and n_heads * d_head <= 4096:
            return FilterResult.VALID
    else:
        # Strategy 3: sweep S*B, fix N<=10, D<=128
        if n_heads <= 10 and S_tkg * B <= 512 and (B <= 4 or d_head >= 32):
            return FilterResult.VALID

    return FilterResult.INVALID


class OutputProjTkgDtypeModeConfig(TypedDict):
    """Shapes and fusion settings shared by every dtype_mode canary case."""

    B: int
    H: int
    S_tkg: int
    d_head: int
    dtype: str
    n_heads: int
    test_bias: bool
    transpose_out: bool


class TestOutputProjTkgKernel:
    def run_output_proj_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        B: int,
        H: int,
        S_tkg: int,
        d_head: int,
        quantization_type,
        dtype,
        lnc_degree,
        n_heads: int,
        test_bias: int | bool,
        transpose_out: bool,
        is_negative_test: bool = False,
        dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    ):
        if B * n_heads * d_head * H > 128 * 1024 * 1024:
            import pytest

            pytest.skip("Config too large for shared fleet SSH transfer (weight > 120MB causes timeout)")

        # Pre-resolve DtypeMode.AUTO to the concrete OCP/NON_OCP variant so the
        # weight allocated by build_output_proj_tkg_input matches the platform.
        # The kernel's internal FP8 allocations follow weight.dtype.
        resolved_dtype_mode = resolve_dtype_mode_for_torch_ref(dtype_mode, compiler_args.platform_target)

        def input_generator(test_config):
            return build_output_proj_tkg_input(
                lnc_degree=lnc_degree,
                d_head=d_head,
                B=B,
                n_heads=n_heads,
                S_tkg=S_tkg,
                quantization_type=quantization_type,
                H=H,
                test_bias=test_bias,
                transpose_out=transpose_out,
                dtype_mode=resolved_dtype_mode,
            )

        def output_tensors(kernel_input):
            # Shape depends on transpose_out flag
            if transpose_out:
                H0 = 128
                output_shape = (H0, lnc_degree, H // lnc_degree // H0, B * S_tkg)
            else:
                output_shape = (B * S_tkg, H)
            return {"out": np.zeros(output_shape, dtype=dtype)}

        # Pre-resolve DtypeMode.AUTO for the torch ref; the kernel receives the
        # original dtype_mode (via build_output_proj_tkg_input) and resolves at
        # trace time. The torch ref runs on CPU and can't query hardware.
        torch_ref_dtype_mode = resolved_dtype_mode

        @functools.wraps(output_projection_tkg_torch_ref)
        def _torch_ref_with_resolved_dtype_mode(**kwargs):
            kwargs["dtype_mode"] = torch_ref_dtype_mode
            return output_projection_tkg_torch_ref(**kwargs)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_proj_tkg_wrapper,
            torch_ref=torch_ref_wrapper(_torch_ref_with_resolved_dtype_mode),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=2e-2 if quantization_type == QuantizationType.NONE else 5e-2,
            atol=1e-5,
            inference_args=TKG_INFERENCE_ARGS,
            is_negative_test=is_negative_test,
        )

    @pytest_parametrize(PARAM_NAMES, OUTPUT_PROJ_TKG_TEST_CASES, abbrevs=_ABBREVS)
    def test_output_proj_tkg_unit(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        B: int,
        n_heads: int,
        S_tkg: int,
        d_head: int,
        H: int,
        quantization_type: QuantizationType,
        test_bias: bool,
        transpose_out: bool,
    ):
        if quantization_type == QuantizationType.STATIC_MX and not platform_target.is_trn3():
            pytest.skip("STATIC_MX quantization is only supported on TRN3.")

        compiler_args = CompilerArgs(platform_target=platform_target)
        self.run_output_proj_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=B,
            H=H,
            S_tkg=S_tkg,
            d_head=d_head,
            quantization_type=quantization_type,
            dtype=nl.bfloat16,
            lnc_degree=compiler_args.logical_nc_config,
            n_heads=n_heads,
            test_bias=test_bias,
            transpose_out=transpose_out,
        )

    @pytest_parametrize(MANUAL_PARAM_NAMES, MANUAL_TEST_CASES, abbrevs=_ABBREVS, prefix="manual")
    def test_output_proj_tkg_sweep_manual(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        B: int,
        n_heads: int,
        S_tkg: int,
        d_head: int,
        H: int,
        quantization_type: QuantizationType,
        test_bias: bool,
        transpose_out: bool,
    ):
        compiler_args = CompilerArgs(platform_target=platform_target)
        lnc_degree = compiler_args.logical_nc_config

        # H=602 is not divisible by 128*lnc_degree, so this is a negative test
        # when transpose_out=True (requires H % (P_MAX * lnc_degree) == 0)
        H_divis_requirement = P_MAX if transpose_out else 1
        is_negative = H % (lnc_degree * H_divis_requirement) != 0

        self.run_output_proj_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=B,
            H=H,
            S_tkg=S_tkg,
            d_head=d_head,
            quantization_type=quantization_type,
            dtype=nl.bfloat16,
            lnc_degree=lnc_degree,
            n_heads=n_heads,
            test_bias=test_bias,
            transpose_out=transpose_out,
            is_negative_test=is_negative,
        )

    # Sweep test: coverage_parametrize with "pairs" coverage generates deterministic
    # 2-way interaction coverage. The value lists below cover the full parameter space.
    #
    # Dimension ranges:
    #   d_head: 1..128
    #   B:      1..512 (constraint: B*S_tkg <= 4096 for tp=0, <= 512 for tp=1)
    #   n_heads: 1..64 (constraint: n_heads*d_head <= 4096)
    #   S_tkg:  1..8
    #   H:      128..16384 (multiple_of=128, plus 602 for negative test)
    #   transpose_out: 0 or 1
    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        B=[1, 4, 8, 16, 32, 64, 128, 192, 310, 413, 512],
        n_heads=[1, 2, 4, 8, 10, 14, 27, 40, 53, 64],
        S_tkg=[1, 2, 3, 4, 5, 6, 7, 8],
        d_head=[1, 13, 26, 39, 52, 65, 78, 91, 104, 117, 128],
        H=[128, 602, 1024, 3072, 5376, 7168, 8192, 10240, 12288, 14336, 16384],
        transpose_out=[False, True],
        filter=_filter_sweep_params,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_output_proj_tkg_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        B: int,
        n_heads: int,
        S_tkg: int,
        d_head: int,
        H: int,
        transpose_out: bool,
        is_negative_test_case: bool,
    ):
        compiler_args = CompilerArgs(platform_target=platform_target)
        lnc_degree = compiler_args.logical_nc_config

        # Additional negative test check: H divisibility by lnc_degree * P_MAX for transpose_out
        H_divis_requirement = P_MAX if transpose_out else 1
        if H % (lnc_degree * H_divis_requirement) != 0:
            is_negative_test_case = True

        self.run_output_proj_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=B,
            H=H,
            S_tkg=S_tkg,
            d_head=d_head,
            quantization_type=QuantizationType.NONE,
            dtype=nl.bfloat16,
            lnc_degree=lnc_degree,
            n_heads=n_heads,
            test_bias=True,
            transpose_out=transpose_out,
            is_negative_test=is_negative_test_case,
        )

    # FP8 sweep: same parameter space as sweep but with STATIC quantization.
    # Old test_output_proj_tkg_fp8_sweep forced quantization_type=STATIC on the same sweep config.
    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        B=[1, 4, 8, 16, 32, 64, 128, 192, 310, 413, 512],
        n_heads=[1, 2, 4, 8, 10, 14, 27, 40, 53, 64],
        S_tkg=[1, 2, 3, 4, 5, 6, 7, 8],
        d_head=[1, 13, 26, 39, 52, 65, 78, 91, 104, 117, 128],
        H=[128, 602, 1024, 3072, 5376, 7168, 8192, 10240, 12288, 14336, 16384],
        transpose_out=[False, True],
        filter=_filter_sweep_params,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_output_proj_tkg_fp8_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        B: int,
        n_heads: int,
        S_tkg: int,
        d_head: int,
        H: int,
        transpose_out: bool,
        is_negative_test_case: bool,
    ):
        compiler_args = CompilerArgs(platform_target=platform_target)
        lnc_degree = compiler_args.logical_nc_config

        H_divis_requirement = P_MAX if transpose_out else 1
        if H % (lnc_degree * H_divis_requirement) != 0:
            is_negative_test_case = True

        self.run_output_proj_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            B=B,
            H=H,
            S_tkg=S_tkg,
            d_head=d_head,
            quantization_type=QuantizationType.STATIC,
            dtype=nl.bfloat16,
            lnc_degree=lnc_degree,
            n_heads=n_heads,
            test_bias=True,
            transpose_out=transpose_out,
            is_negative_test=is_negative_test_case,
        )

    # MX quantization test cases
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(
        "batch, n_heads, seqlen, d_head, hidden, add_bias, transpose_out, quantization_type",
        [
            [32, 3, 4, 128, 4096, True, False, QuantizationType.MX],
            [32, 1, 2, 128, 2048, True, False, QuantizationType.MX],
            [64, 4, 2, 128, 3072, True, False, QuantizationType.MX],
            [64, 4, 2, 128, 8192, True, False, QuantizationType.MX],
            [64, 2, 1, 128, 4096, False, False, QuantizationType.MX],
            [128, 8, 1, 128, 8192, False, False, QuantizationType.MX],
            [128, 2, 1, 128, 8192, False, False, QuantizationType.MX],
            [128, 32, 1, 64, 6144, True, False, QuantizationType.MX],
        ],
    )
    def test_output_proj_tkg_mxfp(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch: int,
        n_heads: int,
        seqlen: int,
        d_head: int,
        hidden: int,
        add_bias: bool,
        transpose_out: bool,
        quantization_type: QuantizationType,
    ):
        ################## Build kernel input tensors ####################

        kernel_assert(quantization_type == QuantizationType.MX, "This input test generator assumes MX dtype.")
        kernel_assert((batch * seqlen % 4) == 0, "[test_output_proj_tkg_mxfp] requires BxS to be divisible by 4.")
        kernel_assert((hidden % 512) == 0, "")
        dtype = nl.bfloat16
        np.random.seed(42)

        # Generate input/attention (BF16 dtype), to be quantized online inside kernel.
        # TKG out_proj kernel assumes [D,B,N,S] input layout.
        attention = gaussian_tensor_generator()(shape=(d_head, batch, n_heads, seqlen), dtype=dtype, name="attention")

        # Generate pre-quantized weights using stabilized MX data
        # weights_qtz has [n_heads * d_head // 4, hidden] shape.
        _, weights_qtz, weight_scales = generate_stabilized_mx_data(
            mx_dtype=nl.float8_e4m3fn_x4,
            shape=((n_heads * d_head) // 4, hidden * 4),
        )
        weight_scales = weight_scales.reshape((n_heads * d_head) // 32, hidden)

        bias = gaussian_tensor_generator(std=100)(shape=(1, hidden), dtype=dtype, name="bias") if add_bias else None

        # Dictionary of inputs to be passed to both torch reference and the kernel
        # Names match kernel parameters exactly
        kernel_inputs = {
            "attention": attention,
            "weight": weights_qtz,
            "bias": bias,
            "quantization_type": quantization_type,
            "weight_scale": weight_scales,
            "input_scale": None,
            "TRANSPOSE_OUT": transpose_out,
            "OUT_IN_SB": False,
        }

        def input_generator(test_config):
            return kernel_inputs

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch * seqlen, hidden), dtype=dtype)}

        compiler_args = CompilerArgs(platform_target=platform_target)
        # input_generator (=kernel_inputs) passed to both torch_ref and the kernel.
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_tkg,
            torch_ref=torch_ref_wrapper(output_projection_tkg_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=compiler_args,
            rtol=5e-2,
            atol=1e-3,
            inference_args=TKG_INFERENCE_ARGS,
        )

    # ------------------------------------------------------------------
    # Opt-in FP8 E4M3 canary (dtype_mode).
    #
    # Kernel: STATIC attention-quant SBUF dtype follows ``weight.dtype``.
    # Torch ref: STATIC input clip is derived from dtype_mode (OCP → 448,
    # NON_OCP → 240) so goldens match.
    # ------------------------------------------------------------------
    _OUTPUT_PROJ_TKG_BY_DTYPE_MODE_CONFIG: OutputProjTkgDtypeModeConfig = {
        "B": 4,
        "H": 3072,
        "S_tkg": 4,
        "d_head": 128,
        "dtype": nl.bfloat16,
        "n_heads": 8,
        "test_bias": True,
        "transpose_out": False,
    }

    @pytest.mark.parametrize("dtype_mode", [DtypeMode.NON_OCP, DtypeMode.OCP, DtypeMode.AUTO])
    @pytest.mark.parametrize("quantization_type", [QuantizationType.STATIC, QuantizationType.ROW])
    def test_output_proj_tkg_by_dtype_mode(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        quantization_type: QuantizationType,
        dtype_mode: DtypeMode,
    ):
        """Smoke-test each DtypeMode through output_projection_tkg STATIC/ROW.

        NON_OCP → ``nl.float8_e4m3`` (240), any platform.
        OCP     → ``nl.float8_e4m3fn`` (448), TRN3 only.
        AUTO    → ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3`` elsewhere.
        """
        if dtype_mode == DtypeMode.OCP and not platform_target.is_trn3():
            pytest.skip("dtype_mode=DtypeMode.OCP only exercises the OCP path on TRN3")
        compiler_args = CompilerArgs(platform_target=platform_target)
        self.run_output_proj_tkg_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            quantization_type=quantization_type,
            lnc_degree=compiler_args.logical_nc_config,
            dtype_mode=dtype_mode,
            **self._OUTPUT_PROJ_TKG_BY_DTYPE_MODE_CONFIG,
        )
