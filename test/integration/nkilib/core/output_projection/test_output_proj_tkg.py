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

# LEGACY SWEEP TEST FRAMEWORK - Uses @range_test_config / RangeTestHarness
# New tests should use @pytest.mark.coverage_parametrize instead
import enum
from test.integration.nkilib.utils.tensor_generators import (
    FP8_E4M3_MAX,
    gaussian_tensor_generator,
    np_random_sample,
    static_cast,
)
from test.utils.common_dataclasses import (
    TKG_INFERENCE_ARGS,
    CompilerArgs,
)
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
    RangeManualGeneratorStrategy,
    RangeMonotonicGeneratorStrategy,
    RangeProductConstraintMonotonicStrategy,
    RangeRandomGeneratorStrategy,
    RangeTestCase,
    RangeTestConfig,
    TensorConfig,
    TensorRangeConfig,
    assert_negative_test_case,
    range_test_config,
)
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.output_projection.output_projection_tkg import (
    output_projection_tkg,
)
from nkilib_src.nkilib.core.output_projection.output_projection_tkg_torch import (
    output_projection_tkg_torch_ref,
)
from nkilib_src.nkilib.core.utils.common_types import QuantizationType
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert

# Hardware constants - must match values in output_projection_tkg.py
F_MAX = 512  # Free dimension size for PSUM/GEMM operations
P_MAX = 128  # Partition dimension size


# Dimension names for test configuration
DUMMY_TENSOR_NAME = "dummy"
BATCH_DIM_NAME = "batch"
NUM_HEADS_DIM_NAME = "n_heads"
SEQUENCE_LEN_DIM_NAME = "S_tkg"
ATTN_DIM_NAME = "d_head"
HIDDEN_DIM_NAME = "H"
TEST_BIAS_NAME = "test_bias"
DO_TRANSPOSE_OUT_NAME = "transpose_out"
QUANTIZATION_TYPE_NAME = "quantization_type"


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


def build_output_proj_tkg_input(lnc_degree, d_head, B, n_heads, S_tkg, quantization_type, H, test_bias, transpose_out):
    dtype = nl.bfloat16

    if quantization_type == QuantizationType.STATIC:
        random_gen = np_random_sample()

        def convert_to_range(max_val, unif_tensor):
            return (2 * max_val * unif_tensor - max_val).astype(unif_tensor.dtype)

        attention = convert_to_range(1, random_gen(shape=(d_head, B, n_heads, S_tkg), dtype=dtype, name="attention"))
        # Compute input_scale from actual max with small perturbation (simulates calibration)
        in_scale = (np.abs(attention).max() / FP8_E4M3_MAX) * np.random.uniform(0.995, 1.005)

        weight_bf16 = convert_to_range(1, random_gen(shape=(d_head * n_heads, H), dtype=dtype, name="weight"))
        w_scale = np.abs(weight_bf16).max() / FP8_E4M3_MAX
        weight = static_cast(weight_bf16 / w_scale, nl.float8_e4m3)

        weight_scale = np.broadcast_to(np.array([[w_scale]], dtype=np.float32), (128, 1))
        input_scale = np.broadcast_to(np.array([[in_scale]], dtype=np.float32), (128, 1))
    else:
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
)


@pytest_test_metadata(
    name="Output Projection TKG",
    pytest_marks=["output_projection", "tkg"],
)
class TestOutputProjTkgKernel:
    def run_range_output_proj_tkg_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        lnc_degree,
        dtype,
        collector: IMetricsCollector,
    ):
        dummy_tensor = test_options.tensors[DUMMY_TENSOR_NAME]

        B = dummy_tensor[BATCH_DIM_NAME]
        n_heads = dummy_tensor[NUM_HEADS_DIM_NAME]
        S_tkg = dummy_tensor[SEQUENCE_LEN_DIM_NAME]
        d_head = dummy_tensor[ATTN_DIM_NAME]
        H = dummy_tensor[HIDDEN_DIM_NAME]
        transpose_out = dummy_tensor[DO_TRANSPOSE_OUT_NAME] == 1
        test_bias = dummy_tensor[TEST_BIAS_NAME] if TEST_BIAS_NAME in dummy_tensor else True
        quantization_type = (
            dummy_tensor[QUANTIZATION_TYPE_NAME] if QUANTIZATION_TYPE_NAME in dummy_tensor else QuantizationType.NONE
        )
        is_negative_test_case = test_options.is_negative_test_case

        # When `transpose_out` is False, the tkg kernel requires H to be divisible by
        # lnc_degree * nl.tile_size.gemm_moving_fmax. When `transpose_out` is True,
        # the kernel requires divisibility by lnc_degree * P_MAX.
        H_divis_requirement = P_MAX if transpose_out else 1
        if H % (lnc_degree * H_divis_requirement) != 0:
            is_negative_test_case = True

        test_size_classification = TkgOutputProjClassification.classify(
            B=B, n_heads=n_heads, S_tkg=S_tkg, d_head=d_head, H=H
        )

        with assert_negative_test_case(is_negative_test_case):
            self.run_output_proj_tkg_test(
                test_manager,
                compiler_args,
                B,
                H,
                S_tkg,
                d_head,
                quantization_type,
                dtype,
                lnc_degree,
                n_heads,
                test_bias,
                transpose_out,
            )

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
    ):
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
            )

        def output_tensors(kernel_input):
            # Shape depends on transpose_out flag
            if transpose_out:
                H0 = 128
                output_shape = (H0, lnc_degree, H // lnc_degree // H0, B * S_tkg)
            else:
                output_shape = (B * S_tkg, H)
            return {"out": np.zeros(output_shape, dtype=dtype)}

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
            rtol=2e-2 if quantization_type == QuantizationType.NONE else 5e-2,
            atol=1e-5,
            inference_args=TKG_INFERENCE_ARGS,
        )

    @staticmethod
    def output_proj_tkg_sweep_config() -> RangeTestConfig:
        """Generate sweep configuration for TKG output projection kernel"""
        S = 8  # Matches Perf Card limits
        H = 16384  # Matches Perf Card limits
        B = 4  # Note: B up to 128 is handled by RangeProductConstraintMonotonicStrategy
        D = 128  # Matches Perf Card limits
        N = 10  # Note: N up to 64 is handled by RangeProductConstraintMonotonicStrategy

        tc = TensorRangeConfig(
            tensor_configs={
                DUMMY_TENSOR_NAME: TensorConfig(
                    [
                        DimensionRangeConfig(max=D, name=ATTN_DIM_NAME),
                        DimensionRangeConfig(max=B, name=BATCH_DIM_NAME),
                        DimensionRangeConfig(max=N, name=NUM_HEADS_DIM_NAME),
                        DimensionRangeConfig(max=S, name=SEQUENCE_LEN_DIM_NAME),
                        DimensionRangeConfig(max=H, min=128, multiple_of=128, name=HIDDEN_DIM_NAME),
                        DimensionRangeConfig(min=0, max=1, name=DO_TRANSPOSE_OUT_NAME),
                    ]
                )
            },
            random_sample_size=6,
            monotonic_step_percent=10,
        )

        tc.custom_generators = [
            RangeRandomGeneratorStrategy(
                tc.random_sample_size,
            ),
            RangeMonotonicGeneratorStrategy(
                tc.monotonic_step_size,
                tc.monotonic_step_percent,
            ),
            RangeManualGeneratorStrategy(
                test_cases=[
                    {
                        DUMMY_TENSOR_NAME: {
                            ATTN_DIM_NAME: D,
                            BATCH_DIM_NAME: B,
                            NUM_HEADS_DIM_NAME: N,
                            SEQUENCE_LEN_DIM_NAME: S,
                            # Here we test for H not cleanly divisible:
                            HIDDEN_DIM_NAME: 602,
                            DO_TRANSPOSE_OUT_NAME: do_transpose,
                        }
                    }
                    for do_transpose in [0, 1]
                ]
            ),
            # transpose_out=False cases:
            RangeProductConstraintMonotonicStrategy(
                fixed_dims={
                    DUMMY_TENSOR_NAME: {
                        NUM_HEADS_DIM_NAME: N,
                        ATTN_DIM_NAME: D,
                        HIDDEN_DIM_NAME: H,
                        DO_TRANSPOSE_OUT_NAME: 0,
                    }
                },
                product_dims=(SEQUENCE_LEN_DIM_NAME, BATCH_DIM_NAME),
                product_limit=4096,
                step_percent=20,
                dim_max={SEQUENCE_LEN_DIM_NAME: 8, BATCH_DIM_NAME: 512},
            ),
            RangeProductConstraintMonotonicStrategy(
                fixed_dims={
                    DUMMY_TENSOR_NAME: {
                        BATCH_DIM_NAME: B,
                        SEQUENCE_LEN_DIM_NAME: S,
                        HIDDEN_DIM_NAME: H,
                        DO_TRANSPOSE_OUT_NAME: 0,
                    }
                },
                product_dims=(NUM_HEADS_DIM_NAME, ATTN_DIM_NAME),
                product_limit=4096,  # N * D <= 4096
                step_percent=20,
                dim_max={NUM_HEADS_DIM_NAME: 64, ATTN_DIM_NAME: 128},
            ),
            # transpose_out=True cases
            RangeProductConstraintMonotonicStrategy(
                fixed_dims={
                    DUMMY_TENSOR_NAME: {
                        NUM_HEADS_DIM_NAME: N,
                        ATTN_DIM_NAME: D,
                        HIDDEN_DIM_NAME: H,
                        DO_TRANSPOSE_OUT_NAME: 1,
                    }
                },
                product_dims=(SEQUENCE_LEN_DIM_NAME, BATCH_DIM_NAME),
                product_limit=512,
                step_percent=20,
                dim_max={SEQUENCE_LEN_DIM_NAME: 8, BATCH_DIM_NAME: 128},
            ),
        ]

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=tc,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "B,n_heads,S_tkg,d_head,H,quantization_type,test_bias,transpose_out",
        OUTPUT_PROJ_TKG_TEST_CASES,
    )
    def test_output_proj_tkg_unit(
        self,
        test_manager: Orchestrator,
        B: int,
        n_heads: int,
        S_tkg: int,
        d_head: int,
        H: int,
        quantization_type: QuantizationType,
        test_bias: bool,
        transpose_out: bool,
    ):
        compiler_args = CompilerArgs()
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

    @range_test_config(output_proj_tkg_sweep_config())
    def test_output_proj_tkg_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
    ):
        compiler_args = CompilerArgs()
        self.run_range_output_proj_tkg_test(
            test_manager=test_manager,
            dtype=nl.bfloat16,
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            collector=collector,
        )

    @range_test_config(output_proj_tkg_sweep_config())
    def test_output_proj_tkg_fp8_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
    ):
        compiler_args = CompilerArgs()
        range_test_options.tensors[DUMMY_TENSOR_NAME][QUANTIZATION_TYPE_NAME] = QuantizationType.STATIC
        self.run_range_output_proj_tkg_test(
            test_manager=test_manager,
            dtype=nl.bfloat16,
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            collector=collector,
        )

    @staticmethod
    def output_proj_tkg_sweep_manual_config() -> RangeTestConfig:
        """Generate manual-only sweep configuration for TKG output projection kernel.

        This config extracts only the manual test cases from output_proj_tkg_sweep_config
        for fast test execution.
        """
        S = 8
        H = 16384
        B = 4
        D = 128
        N = 10

        tc = TensorRangeConfig(
            tensor_configs={
                DUMMY_TENSOR_NAME: TensorConfig(
                    [
                        DimensionRangeConfig(max=D, name=ATTN_DIM_NAME),
                        DimensionRangeConfig(max=B, name=BATCH_DIM_NAME),
                        DimensionRangeConfig(max=N, name=NUM_HEADS_DIM_NAME),
                        DimensionRangeConfig(max=S, name=SEQUENCE_LEN_DIM_NAME),
                        DimensionRangeConfig(max=H, min=128, multiple_of=128, name=HIDDEN_DIM_NAME),
                        DimensionRangeConfig(min=0, max=1, name=DO_TRANSPOSE_OUT_NAME),
                    ]
                )
            },
            monotonic_step_size=1,
            custom_generators=[
                RangeManualGeneratorStrategy(
                    test_cases=[
                        {
                            DUMMY_TENSOR_NAME: {
                                ATTN_DIM_NAME: D,
                                BATCH_DIM_NAME: B,
                                NUM_HEADS_DIM_NAME: N,
                                SEQUENCE_LEN_DIM_NAME: S,
                                # Here we test for H not cleanly divisible:
                                HIDDEN_DIM_NAME: 602,
                                DO_TRANSPOSE_OUT_NAME: do_transpose,
                            }
                        }
                        for do_transpose in [0, 1]
                    ]
                ),
            ],
        )

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=tc,
        )

    @pytest.mark.fast
    @range_test_config(output_proj_tkg_sweep_manual_config())
    def test_output_proj_tkg_sweep_manual(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: IMetricsCollector,
    ):
        compiler_args = CompilerArgs()
        self.run_range_output_proj_tkg_test(
            test_manager=test_manager,
            dtype=nl.bfloat16,
            test_options=range_test_options,
            lnc_degree=compiler_args.logical_nc_config,
            compiler_args=compiler_args,
            collector=collector,
        )
