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

"""
Test suite for cascaded max kernel following RMSNorm test patterns.

Key Features:
- Test Structure: Mirrors the RMSNorm test with similar imports, class structure, and test organization
- Reference Implementation: cascaded_max_ref() provides golden reference using numpy's argmax and max
- Validation: golden_output_validator() compares hardware output against reference implementation
- Test Classification: MaxClassification enum to categorize test sizes (Small/Medium/Large)
- Parameterized Tests: Unit tests with various batch sizes, sequence lengths, and vocabulary sizes
- Range Testing: Sweep configuration for comprehensive testing across parameter ranges
- Multiple Data Types: Support for both float32 and bfloat16

Test Coverage:
- Unit Tests: 12 different parameter combinations covering various tensor shapes and data types
- Sweep Tests: Configurable range testing with random and monotonic generation strategies
- Edge Cases: Single batch/sequence scenarios and larger vocabulary sizes up to 32K

The test follows the exact same patterns as the RMSNorm test but adapted for the cascaded max
kernel's specific input/output requirements (single input tensor, max values + indices output).
"""

import enum
from test.integration.nkilib.utils.comparators import maxAllClose
from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, CustomValidator, KernelArgs, LazyGoldenGenerator, ValidationArgs
from test.utils.metrics_collector import MetricName, MetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.ranged_test_harness import (
    DimensionRangeConfig,
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
from typing import Any, final

import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.max.cascaded_max import cascaded_max
from typing_extensions import override

INPUT_TENSOR_NAME = "input_tensor"
BATCH_DIM_NAME = "batch"
VOCAB_DIM_NAME = "vocab"
SEQUENCE_LEN_DIM_NAME = "seqlen"


class MaxClassification(enum.Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

    @staticmethod
    def classify(batch_size: int, seqlen: int, vocab_size: int):
        total_elements = batch_size * seqlen * vocab_size

        if total_elements <= 100000:
            return MaxClassification.SMALL
        elif total_elements <= 1000000:
            return MaxClassification.MEDIUM
        else:
            return MaxClassification.LARGE

    @override
    def __str__(self):
        return self.name


def cascaded_max_ref(inp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cascaded max reference implementation.

    Args:
        inp: Input tensor of shape [..., vocab_size]

    Returns:
        tuple: (max_values, max_indices) where:
            - max_values: shape [..., 1] containing max values
            - max_indices: shape [..., 1] containing indices of max values
    """
    # Find max values and indices along last dimension
    max_indices = np.argmax(inp, axis=-1, keepdims=True).astype(np.uint32)
    max_values = np.max(inp, axis=-1, keepdims=True)

    return {"max_values": max_values, "max_indices": max_indices}


def golden_output_generator(inp: dict[str, Any]):
    def generate(inp: npt.NDArray[Any]):
        input_tensor = inp[INPUT_TENSOR_NAME]
        output = cascaded_max_ref(input_tensor)
        return output

    out = lambda: generate(inp)
    return out


def build_cascaded_max_kernel_input(lnc_degree: int, batch: int, seqlen: int, vocab_size: int, dtype, tensor_gen):
    """Build input for cascaded max kernel."""
    input_tensor = tensor_gen(shape=(batch, seqlen, vocab_size), dtype=dtype, name=INPUT_TENSOR_NAME)
    return {INPUT_TENSOR_NAME: input_tensor}


@pytest_test_metadata(
    name="Cascaded Max",
    pytest_marks=["max", "cascaded"],
)
@final
class TestCascadedMaxKernel:
    def run_range_cascaded_max_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        test_options: RangeTestCase,
        dtype,
        collector: MetricsCollector,
    ):
        input_tensor = test_options.tensors[INPUT_TENSOR_NAME]

        batch_size = input_tensor[BATCH_DIM_NAME]
        seqlen = input_tensor[SEQUENCE_LEN_DIM_NAME]
        vocab_size = input_tensor[VOCAB_DIM_NAME]
        lnc_degree = compiler_args.logical_nc_config
        test_size_classification = MaxClassification.classify(
            batch_size=batch_size,
            seqlen=seqlen,
            vocab_size=vocab_size,
        )

        is_negative_test_case = test_options.is_negative_test_case
        with assert_negative_test_case(is_negative_test_case):
            self.run_cascaded_max_test(
                test_manager,
                compiler_args,
                collector,
                lnc_degree,
                batch_size,
                seqlen,
                vocab_size,
                dtype,
            )

    def run_cascaded_max_test(
        self,
        test_manager: Orchestrator,
        compiler_args: CompilerArgs,
        collector: MetricsCollector,
        lnc_degree: int,
        batch_size: int,
        seqlen: int,
        vocab_size: int,
        dtype,
        tensor_gen=gaussian_tensor_generator(),
    ):
        kernel_input = build_cascaded_max_kernel_input(
            lnc_degree=lnc_degree,
            batch=batch_size,
            seqlen=seqlen,
            vocab_size=vocab_size,
            dtype=dtype,
            tensor_gen=tensor_gen,
        )
        placeholder_output = {
            "max_values": np.ndarray(shape=(batch_size, seqlen, 1), dtype=dtype),
            "max_indices": np.ndarray(shape=(batch_size, seqlen, 1), dtype=np.uint32),
        }

        test_manager.execute(
            KernelArgs(
                kernel_func=cascaded_max,
                compiler_input=compiler_args,
                kernel_input=kernel_input,
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(
                        lazy_golden_generator=golden_output_generator(kernel_input),
                        output_ndarray=placeholder_output,
                    )
                ),
            )
        )

    # fmt: off
    cascaded_max_unit_params = "lnc_degree, batch, seqlen, vocab_size, dtype"

    cascaded_max_unit_perms = [
        # Llama 3 76B before global gather
        [1, 8, 5, 4058, nl.float32],
        [2, 8, 5, 4058, nl.float32],
        [1, 4, 5, 4058, nl.float32],
        [2, 5, 5, 4058, nl.float32],

        # # Llama 3 76B after global gather
        [1, 4, 5, 8192, nl.float32],
        [1, 8, 5, 8192, nl.float32],
        [2, 8, 5, 8192, nl.float32],
        [2, 5, 5, 8192, nl.float32],

        # Functionality tests
        # nominal
        [1, 1, 1, 3168, nl.float32],
        [2, 1, 1, 3168, nl.float32],

        # Vocab size generalization
        [2, 1, 1, 256, nl.float32],
        [2, 1, 1, 16000, nl.float32],

        # Max stage num batch sizes
        [2, 3, 1, 3168, nl.float32],
        [2, 7, 1, 3168, nl.float32],
        [2, 8, 1, 3168, nl.float32],

        # Medium stage num batch sizes
        [2, 10, 1, 3168, nl.float32],
        [2, 16, 1, 3168, nl.float32],
        [2, 32, 1, 3168, nl.float32],
        [2, 63, 1, 3168, nl.float32],
        [2, 65, 1, 3168, nl.float32],
        [2, 99, 1, 3168, nl.float32],
        [2, 128, 1, 3168, nl.float32],
        [2, 256, 1, 3168, nl.float32],
        [2, 1, 1, 3168, nl.float32],

        # Mixed tests
        [1, 1, 7, 3999,  nl.float32],
        [1, 1, 63, 3999, nl.float32],
        [2, 1, 127, 3999, nl.float32],
        [1, 1, 127, 3999, nl.float32],
    ]
    # fmt: on
    @pytest.mark.fast
    @pytest.mark.parametrize(cascaded_max_unit_params, cascaded_max_unit_perms)
    def test_cascaded_max_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        dtype,
    ):
        compiler_args = CompilerArgs(logical_nc_config=lnc_degree)
        self.run_cascaded_max_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            collector=collector,
            lnc_degree=lnc_degree,
            batch_size=batch,
            seqlen=seqlen,
            vocab_size=vocab_size,
            dtype=dtype,
            tensor_gen=gaussian_tensor_generator(),
        )

    @staticmethod
    def cascaded_max_sweep_config() -> RangeTestConfig:
        # Test-specific dimension values
        B = 128
        S = 1
        V = 2**14

        tc = TensorRangeConfig(
            tensor_configs={
                INPUT_TENSOR_NAME: TensorConfig(
                    [
                        DimensionRangeConfig(max=B, name=BATCH_DIM_NAME),
                        DimensionRangeConfig(max=S, name=SEQUENCE_LEN_DIM_NAME),
                        DimensionRangeConfig(max=V, name=VOCAB_DIM_NAME),
                    ]
                ),
            },
            monotonic_step_percent=10,
        )

        # Add generators: random, monotonic
        tc.custom_generators = [
            RangeRandomGeneratorStrategy(
                tc.random_sample_size,
            ),
            RangeMonotonicGeneratorStrategy(
                tc.monotonic_step_size,
                tc.monotonic_step_percent,
            ),
        ]

        return RangeTestConfig(
            additional_params={},
            global_tensor_configs=tc,
        )

    @range_test_config(cascaded_max_sweep_config())
    def test_cascaded_max_sweep(
        self,
        test_manager: Orchestrator,
        range_test_options: RangeTestCase,
        collector: MetricsCollector,
    ):
        compiler_args = CompilerArgs()
        self.run_range_cascaded_max_test(
            test_manager=test_manager,
            compiler_args=compiler_args,
            dtype=nl.float32,
            test_options=range_test_options,
            collector=collector,
        )
