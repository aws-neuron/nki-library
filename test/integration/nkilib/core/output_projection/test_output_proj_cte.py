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

from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.tensor_generators import (
    gaussian_tensor_generator,
    np_random_sample,
)
from test.integration.nkilib.utils.test_kernel_common import convert_to_torch
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult, assert_negative_test_case
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import final

import nki.dtype as nt
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.output_projection.output_projection_cte import (
    output_projection_cte,
)
from nkilib_src.nkilib.core.output_projection.output_projection_cte.output_projection_cte_torch import (
    output_projection_cte_torch_ref,
)
from nkilib_src.nkilib.core.utils.common_types import QuantizationType


def build_output_proj_cte_input(batch, seqlen, hidden, n_head, d_head, test_bias):
    dtype = nl.bfloat16

    random_gen = np_random_sample()
    attention = random_gen(shape=(batch, n_head, d_head, seqlen), dtype=dtype)
    weight = random_gen(shape=(n_head * d_head, hidden), dtype=dtype)
    bias = gaussian_tensor_generator(std=100)(shape=(1, hidden), dtype=dtype, name="bias") if test_bias else None

    return {
        "attention": attention,
        "weight": weight,
        "bias": bias,
    }


def build_output_proj_cte_fp8_input(batch, seqlen, hidden, n_head, d_head, test_bias):
    """Build input tensors for FP8 static quantization tests"""
    dtype = nl.bfloat16
    quant_dtype = nt.float8_e4m3

    random_gen = np_random_sample()
    attention = random_gen(shape=(batch, n_head, d_head, seqlen), dtype=dtype)
    weight = random_gen(shape=(n_head * d_head, hidden), dtype=quant_dtype)
    bias = gaussian_tensor_generator(std=100)(shape=(1, hidden), dtype=dtype, name="bias") if test_bias else None

    # Generate scale tensors for static quantization
    rng = np.random.default_rng(0)
    static_quant_input_scale = np.full(shape=(128, 1), fill_value=rng.normal(), dtype=np.float32)
    static_quant_weight_scale = np.full(shape=(128, 1), fill_value=rng.normal(), dtype=np.float32)

    return {
        "attention": attention,
        "weight": weight,
        "bias": bias,
        "quantization_type": QuantizationType.STATIC,
        "input_scales": static_quant_input_scale,
        "weight_scales": static_quant_weight_scale,
    }


# fmt: off
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
    (1, 128, 3072, 8, 32, True),   # group_size of 4
    # 70B & 76B
    (1, 128, 8192, 1, 128, False),
    (1, 256, 8192, 1, 128, False),
    (1, 512, 8192, 1, 128, False),
    (1, 1024, 8192, 1, 128, False),
    (1, 2048, 8192, 1, 128, False),
    (1, 4096, 8192, 1, 128, False),
    (1, 8192, 8192, 1, 128, False),
    (1, 10240, 8192, 1, 128, False),
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
# fmt: on

OUTPUT_PROJ_CTE_UNIT_PARAMS = "batch, seqlen, hidden, n_head, d_head, test_bias"

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


# Seeded RNG for deterministic sweep test generation
_sweep_rng = np.random.default_rng(42)

# Sweep parameters - sample from both small and large ranges to ensure coverage
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
    boundary_values=[_MAX_H + 2],  # Test kernel fails gracefully above max H
)
_SWEEP_N_HEAD = BoundedRange(
    values=sorted(_sweep_rng.choice(range(1, 18), size=4, replace=False).tolist()),
    boundary_values=[_MAX_N + 1],  # Test kernel fails gracefully above max N
)
_SWEEP_D_HEAD = BoundedRange(
    values=sorted(
        _sweep_rng.choice(range(1, 64), size=2, replace=False).tolist()
        + _sweep_rng.choice(range(64, 129), size=2, replace=False).tolist()
    ),
    boundary_values=[_MAX_D + 1],  # Test kernel fails gracefully above max D
)


@pytest_test_metadata(
    name="Output Projection CTE",
    pytest_marks=["output_projection", "cte"],
)
@final
class TestOutputProjCteKernel:
    def run_output_proj_cte_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        dtype=nl.bfloat16,
    ):
        # Skip validation for large sizes to avoid OOM during testing
        skip_validation = batch * seqlen > _MAX_BxS_VALIDATE or hidden > _MAX_H_VALIDATE

        kernel_input = build_output_proj_cte_input(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            n_head=n_head,
            d_head=d_head,
            test_bias=test_bias,
        )

        def create_lazy_golden():
            out = output_projection_cte_torch_ref(
                attention=convert_to_torch(kernel_input["attention"]),
                weight=convert_to_torch(kernel_input["weight"]),
                bias=convert_to_torch(kernel_input.get("bias")),
            )
            return {"out": dt.static_cast(out.numpy(), dtype)}

        output_placeholder = {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        test_manager.execute(
            KernelArgs(
                kernel_func=output_projection_cte,
                kernel_input=kernel_input,
                compiler_input=compiler_args,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden if not skip_validation else None,
                        output_ndarray=output_placeholder,
                    ),
                    relative_accuracy=2e-2,
                    absolute_accuracy=1e-5,
                ),
            ),
        )

    def run_output_proj_cte_fp8_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
        dtype=nl.bfloat16,
    ):
        kernel_input = build_output_proj_cte_fp8_input(
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            n_head=n_head,
            d_head=d_head,
            test_bias=test_bias,
        )

        def create_lazy_golden():
            out = output_projection_cte_torch_ref(
                attention=convert_to_torch(kernel_input["attention"]),
                weight=convert_to_torch(kernel_input["weight"]),
                bias=convert_to_torch(kernel_input.get("bias")),
                input_scale=convert_to_torch(kernel_input["input_scales"]),
                weight_scale=convert_to_torch(kernel_input["weight_scales"]),
                quantization_type=QuantizationType.STATIC,
            )
            return {"out": dt.static_cast(out.numpy(), dtype)}

        output_placeholder = {"out": np.zeros((batch, seqlen, hidden), dtype=dtype)}

        test_manager.execute(
            KernelArgs(
                kernel_func=output_projection_cte,
                kernel_input=kernel_input,
                compiler_input=compiler_args,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=create_lazy_golden,
                        output_ndarray=output_placeholder,
                    ),
                    relative_accuracy=0.036,  # Higher tolerance for FP8 quantization
                    absolute_accuracy=1e-5,
                ),
            ),
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_UNIT_CASES)
    def test_output_proj_cte_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        compiler_args = CompilerArgs()
        self.run_output_proj_cte_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            n_head=n_head,
            d_head=d_head,
            test_bias=test_bias,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(OUTPUT_PROJ_CTE_UNIT_PARAMS, OUTPUT_PROJ_CTE_UNIT_CASES)
    def test_output_proj_cte_fp8_unit(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        batch: int,
        seqlen: int,
        hidden: int,
        n_head: int,
        d_head: int,
        test_bias: bool,
    ):
        """Test FP8 static quantization output projection using same test cases as regular unit tests"""
        compiler_args = CompilerArgs()
        self.run_output_proj_cte_fp8_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            batch=batch,
            seqlen=seqlen,
            hidden=hidden,
            n_head=n_head,
            d_head=d_head,
            test_bias=test_bias,
        )

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
    def test_output_proj_cte_sweep(
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
    ):
        compiler_args = CompilerArgs()
        with assert_negative_test_case(is_negative_test_case):
            self.run_output_proj_cte_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
            )

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
    def test_output_proj_cte_fp8_sweep(
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
    ):
        """Test FP8 static quantization output projection sweep tests"""
        compiler_args = CompilerArgs()
        with assert_negative_test_case(is_negative_test_case):
            self.run_output_proj_cte_fp8_test(
                test_manager=test_manager,
                compiler_args=compiler_args,
                collector=collector,
                batch=batch,
                seqlen=seqlen,
                hidden=hidden,
                n_head=n_head,
                d_head=d_head,
                test_bias=test_bias,
            )
