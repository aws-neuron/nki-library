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

"""Integration tests for the output projection CTE kernel using UnitTestFramework."""

from test.integration.nkilib.core.output_projection.test_output_proj_cte_model_config import (
    OUTPUT_PROJ_CTE_MODEL_CONFIGS,
)
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
    generate_stabilized_mx_data,
    np_random_sample,
    np_random_sample_static_quantize_inp,
)
from test.utils.common_dataclasses import (
    CompilerArgs,
    Platforms,
)
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import final

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.output_projection.output_projection_cte import output_projection_cte
from nkilib_src.nkilib.core.output_projection.output_projection_cte.output_projection_cte_torch import (
    output_projection_cte_mx_torch_ref,
    output_projection_cte_torch_ref,
)
from nkilib_src.nkilib.core.utils.common_types import QuantizationType


def generate_output_proj_cte_inputs(
    batch: int,
    seqlen: int,
    hidden: int,
    n_head: int,
    d_head: int,
    test_bias: bool,
    quantization_type: QuantizationType = QuantizationType.NONE,
) -> dict:
    """Generate inputs for output projection CTE test."""
    dtype = nl.bfloat16
    input_scales = None
    weight_scales = None

    random_gen = np_random_sample()
    attention = random_gen(shape=(batch, n_head, d_head, seqlen), dtype=dtype)
    bias = gaussian_tensor_generator(std=100)(shape=(1, hidden), dtype=dtype, name="bias") if test_bias else None

    if quantization_type == QuantizationType.NONE:
        weight = random_gen(shape=(n_head * d_head, hidden), dtype=dtype)
    else:
        quant_dtype = nl.float8_e4m3fn if quantization_type == QuantizationType.STATIC_MX else nl.float8_e4m3
        static_quant_gen = np_random_sample_static_quantize_inp()
        weight, weight_scale_val, input_scale_val = static_quant_gen(shape=(n_head * d_head, hidden), dtype=quant_dtype)
        input_scales = np.full(shape=(128, 1), fill_value=input_scale_val, dtype=np.float32)
        weight_scales = np.full(shape=(128, 1), fill_value=weight_scale_val, dtype=np.float32)

        if quantization_type == QuantizationType.STATIC_MX:
            nd = n_head * d_head
            if nd % 4 == 0:
                w = weight.reshape(nd // 4, 4, hidden)
                weight = np.ascontiguousarray(np.transpose(w, (0, 2, 1))).reshape(nd, hidden)

    return {
        "attention": attention,
        "weight": weight,
        "bias": bias,
        "quantization_type": quantization_type,
        "input_scales": input_scales,
        "weight_scales": weight_scales,
    }


def generate_output_proj_cte_mx_inputs(
    batch: int,
    seqlen: int,
    hidden: int,
    n_head: int,
    d_head: int,
    input_prequantized: bool = False,
    test_bias: bool = False,
) -> dict:
    """Generate inputs for MX FP4 output projection CTE test."""
    dtype = nl.bfloat16
    np.random.seed(42)

    bias = gaussian_tensor_generator(std=100)(shape=(1, hidden), dtype=dtype, name="bias") if test_bias else None

    weight_logical_shape = (n_head * d_head // 4, hidden * 4)
    # Skip MX data generation for invalid shapes - use zeros instead
    if weight_logical_shape[0] % 8 != 0:
        weight_quantized = np.zeros((n_head * d_head // 4, hidden), dtype=np.uint8)
        weight_scale = np.zeros((n_head * d_head // 32, hidden), dtype=np.uint8)
    else:
        _, weight_quantized, weight_scale = generate_stabilized_mx_data(
            nl.float4_e2m1fn_x4, weight_logical_shape, val_range=1.0
        )

    if input_prequantized:
        # Pre-quantized input path: generate float8_e4m3fn_x4 attention with scales
        # Input shape is [B, 1, D_packed, S] where D_packed = n_head * d_head // 4
        # Kernel uses contraction_dim = N * D_packed = 1 * D_packed (no additional //4 division)
        packed_d = n_head * d_head // 4
        attention_logical_shape = (packed_d, seqlen * 4)
        attention_list, scale_list = [], []
        for _ in range(batch):
            if attention_logical_shape[0] % 8 != 0:
                attn_q = np.zeros((packed_d, seqlen), dtype=np.uint8)
                attn_s = np.zeros((packed_d // 8, seqlen), dtype=np.uint8)
            else:
                _, attn_q, attn_s = generate_stabilized_mx_data(
                    nl.float8_e4m3fn_x4, attention_logical_shape, val_range=1.0
                )
            attention_list.append(attn_q.reshape(1, 1, packed_d, seqlen))
            scale_list.append(attn_s.reshape(1, packed_d // 8, seqlen))
        attention = np.concatenate(attention_list, axis=0)
        input_scales = np.concatenate(scale_list, axis=0)
    else:
        attention = (np.random.randn(batch, n_head, d_head, seqlen) * 0.1).astype(np.float32)
        attention = attention.astype(np.float16).view(np.uint16).view(np.float16).astype(nl.bfloat16)
        input_scales = None

    return {
        "attention": attention,
        "weight": weight_quantized,
        "bias": bias,
        "quantization_type": QuantizationType.MX,
        "input_scales": input_scales,
        "weight_scales": weight_scale,
    }


# Manual test cases: (batch, seqlen, hidden, n_head, d_head, test_bias)
OUTPUT_PROJ_CTE_UNIT_CASES = [
    # New model, 2025-Jul
    (1, 16, 5120, 10, 128, True),
    (1, 128, 3072, 16, 64, True),
    (1, 1024, 3072, 16, 64, True),
    (1, 2048, 3072, 16, 64, True),
    (1, 10240, 3072, 8, 64, True),
    # Test cases to verify folding n_head into d_head
    (1, 128, 3072, 16, 10, True),  # group_size of 8
    (1, 128, 3072, 17, 10, True),  # Cannot reshape
    (1, 128, 3072, 8, 32, True),  # group_size of 4
    # 70B & 76B
    (1, 128, 8192, 1, 128, False),
    (1, 256, 8192, 1, 128, False),
    (1, 512, 8192, 1, 128, False),
    (1, 2048, 8192, 1, 128, False),
    (1, 4096, 8192, 1, 128, False),
    (1, 8192, 8192, 1, 128, False),
    (1, 16384, 8192, 1, 128, False),
    (1, 512, 8192, 2, 128, False),
    (1, 1024, 8192, 2, 128, False),
    (1, 2048, 8192, 2, 128, False),
    (1, 4096, 8192, 2, 128, False),
    (1, 8192, 8192, 2, 128, False),
    (1, 10240, 8192, 2, 128, False),
    (1, 16384, 8192, 2, 128, False),
    # 405B
    (1, 512, 16384, 1, 128, False),
    (1, 1024, 16384, 1, 128, False),
    (1, 2048, 16384, 1, 128, False),
    (1, 4096, 16384, 1, 128, False),
    (1, 8192, 16384, 1, 128, False),
    (1, 10240, 16384, 1, 128, False),
    (1, 16384, 16384, 1, 128, False),
    # Draft model
    (1, 1024, 1024, 1, 64, False),
    (1, 1024, 2048, 1, 64, False),
    (1, 1024, 3072, 1, 64, False),
    (1, 1024, 8192, 1, 64, False),
    # 470B model
    (1, 1024, 20480, 3, 128, False),
    # Text
    (4, 256, 7168, 4, 128, False),
    (4, 256, 7168, 1, 128, False),
    (1, 256, 7168, 1, 128, False),
    # arbitrary seqlen not aligned by 128
    (1, 128 + 64, 1024, 1, 128, False),
    (1, 256 + 64, 2048, 2, 128, False),
    (1, 512 + 64, 3072, 3, 128, False),
    (1, 1024 + 64, 7168, 4, 128, False),
    (1, 2048 + 120, 8192, 5, 128, False),
    (1, 4096 + 1000, 16384, 6, 128, False),
    (1, 8192 + 1120, 16384, 7, 128, False),
    (1, 10240 + 1234, 16384, 8, 128, False),
    (1, 16384 + 4321, 16384, 9, 128, False),
]

OUTPUT_PROJ_CTE_UNIT_PARAMS = "batch, seqlen, hidden, n_head, d_head, test_bias"

# Combined unit + model configs (deduplicated)
OUTPUT_PROJ_CTE_UNIT_AND_MODEL_CASES = list(dict.fromkeys(OUTPUT_PROJ_CTE_UNIT_CASES + OUTPUT_PROJ_CTE_MODEL_CONFIGS))

# Kernel constraints
_MAX_B_TIMES_S = 128 * 1024
_MAX_H = 20705
_MAX_N = 17
_MAX_D = 128

# Max sizes to run validation on (to avoid OOM during testing)
_MAX_BxS_VALIDATE = 64 * 1024
_MAX_H_VALIDATE = 16384


def filter_output_proj_combinations(batch, seqlen, hidden, n_head, d_head, test_bias=None):
    """Filter out invalid parameter combinations for output projection kernel."""
    if batch * seqlen > _MAX_B_TIMES_S:
        return FilterResult.INVALID
    return FilterResult.VALID


def filter_output_proj_mx_combinations(batch, seqlen, hidden, n_head, d_head, test_bias=None):
    """Filter out invalid parameter combinations for MX output projection kernel."""
    if batch * seqlen > _MAX_B_TIMES_S:
        return FilterResult.INVALID
    if (n_head * d_head < 128) or (n_head * d_head % 128 != 0):
        return FilterResult.INVALID
    return FilterResult.VALID


# Seeded RNG for deterministic sweep test generation
_sweep_rng = np.random.default_rng(42)

# Sweep parameters
_SWEEP_BATCH = sorted(_sweep_rng.choice(range(1, 129), size=4, replace=False).tolist())
_SWEEP_SEQLEN = sorted(
    _sweep_rng.choice(range(16, 1024), size=4, replace=False).tolist()
    + _sweep_rng.choice(range(1024, 16 * 1024 + 1), size=4, replace=False).tolist()
)
_SWEEP_HIDDEN = BoundedRange(
    values=sorted(
        _sweep_rng.choice(range(128, 2048, 2), size=2, replace=False).tolist()
        + _sweep_rng.choice(range(2048, 16 * 1024 + 1, 2), size=2, replace=False).tolist()
    ),
    boundary_values=[_MAX_H + 2],
)
_SWEEP_N_HEAD = BoundedRange(
    values=sorted(_sweep_rng.choice(range(1, 18), size=4, replace=False).tolist()),
    boundary_values=[_MAX_N + 1],
)
_SWEEP_D_HEAD = BoundedRange(
    values=sorted(
        _sweep_rng.choice(range(1, 64), size=2, replace=False).tolist()
        + _sweep_rng.choice(range(64, 129), size=2, replace=False).tolist()
    ),
    boundary_values=[_MAX_D + 1],
)


@pytest_test_metadata(
    name="Output Projection CTE",
    pytest_marks=["output_projection", "cte"],
)
@final
class TestOutputProjCteKernel:
    """Test class for output_projection_cte using UnitTestFramework."""

    # ============================================================================
    # Float (Non-Quantized) Tests
    # ============================================================================

    @pytest.mark.fast
    @pytest.mark.parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_UNIT_CASES)
    def test_output_proj_cte_bf16_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        platform_target: Platforms,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch, seqlen=seqlen, hidden=hidden, n_head=n_head, d_head=d_head, test_bias=test_bias
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
        )

    # ============================================================================
    # FP8 Static Quantization Tests
    # ============================================================================

    @pytest.mark.fast
    @pytest.mark.parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_UNIT_CASES)
    def test_output_proj_cte_static_fp8_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        platform_target: Platforms,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                quantization_type=QuantizationType.STATIC,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.036,
            atol=1e-5,
        )

    # ============================================================================
    # Sweep Tests
    # ============================================================================

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_SWEEP_BATCH,
        seqlen=_SWEEP_SEQLEN,
        hidden=_SWEEP_HIDDEN,
        n_head=_SWEEP_N_HEAD,
        d_head=_SWEEP_D_HEAD,
        test_bias=[True, False],
        filter=filter_output_proj_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_output_proj_cte_bf16_sweep(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        is_negative_test_case: bool,
        platform_target: Platforms,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch, seqlen=seqlen, hidden=hidden, n_head=n_head, d_head=d_head, test_bias=test_bias
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=2e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_SWEEP_BATCH,
        seqlen=_SWEEP_SEQLEN,
        hidden=_SWEEP_HIDDEN,
        n_head=_SWEEP_N_HEAD,
        d_head=_SWEEP_D_HEAD,
        test_bias=[True, False],
        filter=filter_output_proj_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_output_proj_cte_static_fp8_sweep(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        is_negative_test_case: bool,
        platform_target: Platforms,
    ):
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                quantization_type=QuantizationType.STATIC,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.036,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # MX Quantization Tests
    # ============================================================================

    @pytest.mark.fast
    @pytest.mark.parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_UNIT_CASES)
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_output_proj_cte_mxfp4_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for MX FP4 output projection CTE kernel."""
        is_negative_test_case = (n_head * d_head < 128) or (n_head * d_head % 128 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_mx_inputs(batch, seqlen, hidden, n_head, d_head, test_bias=test_bias)

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=output_projection_cte_mx_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # @IGNORE_FAST
    @pytest.mark.coverage_parametrize(
        batch=_SWEEP_BATCH,
        seqlen=_SWEEP_SEQLEN,
        hidden=_SWEEP_HIDDEN,
        n_head=_SWEEP_N_HEAD,
        d_head=_SWEEP_D_HEAD,
        test_bias=[True, False],
        filter=filter_output_proj_mx_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_output_proj_cte_mxfp4_sweep(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        is_negative_test_case: bool,
    ):
        """Sweep test for MX FP4 output projection CTE kernel."""
        is_negative_test_case = (n_head * d_head < 128) or (n_head * d_head % 128 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_mx_inputs(batch, seqlen, hidden, n_head, d_head, test_bias=test_bias)

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=output_projection_cte_mx_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # MX Pre-Quantized Input Tests
    # ============================================================================

    @pytest.mark.fast
    @pytest.mark.parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_UNIT_CASES)
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    def test_output_proj_cte_mxfp4_prequantized_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for MX FP4 output projection with pre-quantized input."""
        n_d = n_head * d_head // 4
        is_negative_test_case = (n_d < 32) or (n_d % 32 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_mx_inputs(
                batch, seqlen, hidden, n_head, d_head, input_prequantized=True, test_bias=test_bias
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=output_projection_cte_mx_torch_ref,
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=5e-2,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )

    # ============================================================================
    # STATIC_MX Quantization Tests
    # ============================================================================

    @pytest.mark.fast
    @pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
    @pytest.mark.parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_MODEL_CONFIGS)
    def test_output_proj_cte_static_mxfp8_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Unit test for STATIC_MX FP8 output projection."""
        n_d = n_head * d_head
        is_negative_test_case = (n_d < 128) or (n_d % 128 != 0)
        dtype = nl.bfloat16

        def input_generator(test_config):
            return generate_output_proj_cte_inputs(
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
                quantization_type=QuantizationType.STATIC_MX,
            )

        def output_tensors(kernel_input):
            return {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=output_projection_cte,
            torch_ref=torch_ref_wrapper(output_projection_cte_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            rtol=0.036,
            atol=1e-5,
            is_negative_test=is_negative_test_case,
        )
