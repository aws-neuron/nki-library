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

import math
from test.integration.nkilib.utils.comparators import maxAllClose
from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ValidationArgs,
)
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult, assert_negative_test_case
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.topk.rotational_topk import (
    RotationalTopkConfig,
    TopkConfig,
    cleanup_rotational_constants,
    prepare_rotational_constants,
    rotational_topk,
)
from nkilib_src.nkilib.core.topk.torch_ref import topk_torch_ref
from typing_extensions import override

INPUT_TENSOR_NAME = "input_tensor"
BATCH_DIM_NAME = "batch"
VOCAB_DIM_NAME = "vocab"
SEQUENCE_LEN_DIM_NAME = "seqlen"


def golden_output_validator_values(
    inp: dict[str, Any],
):
    class RotationalTopkValueValidator(CustomValidator):
        @override
        def validate(self, actual_raw_output: npt.NDArray[Any]):
            input_tensor = inp["inp"]
            BxS, _ = input_tensor.shape
            config = inp["config"]
            k = config.topk_config.k
            output = np.frombuffer(actual_raw_output, dtype=input_tensor.dtype).reshape(BxS, k)

            output_ref = torch_ref_wrapper(topk_torch_ref)(inp=input_tensor, config=config)

            passed = maxAllClose(np.sort(output, axis=-1), np.sort(output_ref["topk_values"], axis=-1), verbose=1)
            return passed

    return RotationalTopkValueValidator


def golden_output_validator_indices(
    inp: dict[str, Any],
):
    class RotationalTopkIndicesValidator(CustomValidator):
        @override
        def validate(self, inference_output: npt.NDArray[Any]):
            input_tensor = inp["inp"]
            config = inp["config"]
            BxS, _ = input_tensor.shape
            k = config.topk_config.k
            output = np.frombuffer(inference_output, dtype=np.uint32).reshape(BxS, k)

            output_ref = torch_ref_wrapper(topk_torch_ref)(inp=input_tensor, config=config)
            val = np.take_along_axis(input_tensor, output.astype(np.uint64), axis=-1)

            passed = maxAllClose(np.sort(output_ref["topk_values"], axis=-1), np.sort(val, axis=-1), verbose=1)

            return passed

    return RotationalTopkIndicesValidator


def _get_np_dtype(dtype):
    """Convert NKI dtype to numpy dtype."""
    if dtype == nl.bfloat16:
        return ml_dtypes.bfloat16
    return dtype


@pytest_test_metadata(
    name="Rotational TopK",
    pytest_marks=["topk", "rotational"],
)
@final
class TestTopKKernel:
    @staticmethod
    def generate_inputs(batch: int, seqlen: int, vocab: int, k: int, dtype, sorted: bool = True):
        """Generate input tensors for topk test."""
        # set seed so that inputs to kernel match
        np.random.seed(seed=42)
        np_dtype = _get_np_dtype(dtype)
        BxS = batch * seqlen
        inp = np.random.randn(BxS, vocab).astype(np_dtype)
        return {"inp": inp, "k": k, "sorted": sorted, "batch": batch, "seqlen": seqlen, "vocab": vocab, "dtype": dtype}

    @staticmethod
    def output_tensors(kernel_input):
        """Define output tensor shapes for validation.

        Note:
            topk_indices are included but not validated for exact match due to
            tie-breaking. Validation passes if topk_values are correct.
        """
        inp = kernel_input["inp"]
        config = kernel_input["config"]
        k = config.topk_config.k
        BxS, _ = inp.shape
        dtype = config.topk_config.inp_dtype
        np_dtype = _get_np_dtype(dtype)

        return {
            "topk_values": np.zeros((BxS, k), dtype=np_dtype),
            "topk_indices": np.zeros((BxS, k), dtype=np.uint32),
        }

    def run_topk_test(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        vocab: int,
        k: int,
        dtype,
        sorted: bool = True,
    ):
        """Run a single topk test case.

        Note:
            Only topk_values are validated. Indices are implicitly correct: if the
            kernel returns correct top-k values, the indices must point to those
            values in the input. This handles tie-breaking where PyTorch and NKI
            may return different valid indices for equal values.
        """
        """Run topk test using UnitTestFramework."""

        def input_generator(test_config, input_tensor_def=None):
            inputs = self.generate_inputs(batch, seqlen, vocab, k, dtype, sorted)

            # Build config
            inp_3d = inputs["inp"].reshape((batch, seqlen, vocab))
            topk_config = TopkConfig(
                inp_shape=inp_3d.shape,
                inp_dtype=dtype,
                k=k,
                sorted=sorted,
                num_programs=lnc_degree,
            )
            inp_reshaped = inp_3d.reshape((topk_config.BxS, topk_config.vocab_size))
            config = RotationalTopkConfig(inp_shape=inp_reshaped.shape, topk_config=topk_config)

            # Setup constants before kernel execution
            prepare_rotational_constants(config)
            config.log_strategy()

            return {"inp": inp_reshaped, "config": config}

        def cleanup_fn():
            """Cleanup function called after test execution."""
            cleanup_rotational_constants()

        inputs = input_generator(test_config=None)
        golden_validator_values = golden_output_validator_values(inputs)
        golden_validator_indices = golden_output_validator_indices(inputs)
        placeholder_output_values = self.output_tensors(kernel_input=inputs)["topk_values"]
        placeholder_output_indices = self.output_tensors(kernel_input=inputs)["topk_indices"]

        validation_args = ValidationArgs(
            golden_output={
                "topk_values": CustomValidatorWithOutputTensorData(
                    validator=golden_validator_values,
                    output_ndarray=placeholder_output_values,
                ),
                "topk_indices": CustomValidatorWithOutputTensorData(
                    validator=golden_validator_indices,
                    output_ndarray=placeholder_output_indices,
                ),
            }
        )
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rotational_topk,
            torch_ref=torch_ref_wrapper(topk_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=lnc_degree),
            rtol=1e-3,
            atol=1e-5,
            custom_validation_args=validation_args,
        )
        cleanup_fn()

    # fmt: off
    topk_unit_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"

    large_batch_perms  = [
            [2, 150, 1, 1000, 50, nl.float32],
            [2, 200, 1, 2000, 64, nl.float32],
            [2, 1024, 1, 5000, 128, nl.float32],
        ]

    topk_unit_perms = [
        # Llama 3 76B before global gather
        [2, 8, 5, 4058, 256, nl.float32],
        [2, 5, 5, 4058, 256, nl.float32],

        # Llama 3 76B after global gather
        [1, 4, 5, 8192, 256, nl.float32],
        [1, 8, 5, 8192, 256, nl.float32],
        [2, 8, 5, 8192, 256, nl.float32],
        [2, 5, 5, 8192, 256, nl.float32],

        # Functionality tests
        [2, 1, 1, 3168, 256, nl.float32],

        # Vocab size generalization
        [2, 1, 1, 16000, 256, nl.float32],

        # Max stage num batch sizes
        [2, 3, 1, 3168, 256, nl.float32],
        [2, 7, 1, 3168, 256, nl.float32],
        [2, 8, 1, 3168, 256, nl.float32],

        # Medium stage num batch sizes
        [2, 10, 1, 3168, 256, nl.float32],
        [2, 16, 1, 3168, 256, nl.float32],
        [2, 32, 1, 3168, 256, nl.float32],
        [2, 63, 1, 3168, 256, nl.float32],

        # Scanning approach batch sizes
        [2, 65, 1, 3168, 256, nl.float32],
        [2, 99, 1, 3168, 256, nl.float32],
        [2, 128, 1, 3168, 256, nl.float32],
        [2, 256, 1, 3168, 256, nl.float32],

        # K generalization nominal
        [2, 1, 1, 3168, 8, nl.float32],
        [2, 1, 1, 3168, 64, nl.float32],
        [2, 1, 1, 3168, 192, nl.float32],

        # K generalization hard
        [2, 1, 1, 3168, 1, nl.float32],
        [2, 1, 1, 3168, 7, nl.float32],
        [2, 1, 1, 3168, 60, nl.float32],
        [2, 1, 1, 3168, 99, nl.float32],

        # Mixed tests
        [1, 1, 7, 3999, 256, nl.float32],
        [1, 1, 63, 3999, 20, nl.float32],
        [2, 1, 127, 3999, 256, nl.float32],
        [2, 1, 127, 3999, 1, nl.float32],
        [1, 1, 127, 3999, 20, nl.float32],

        # bfloat16 tests
        [2, 1, 1, 3168, 256, nl.bfloat16],
        [2, 8, 5, 4058, 256, nl.bfloat16],
        [2, 1, 1, 16000, 256, nl.bfloat16],
        [2, 32, 1, 3168, 256, nl.bfloat16],

        # Large vocab test 
        [2, 1, 1, 25600, 256, nl.float32],
        [2, 8, 1, 25600, 256, nl.float32],
        [2, 1, 1, 2048, 256, nl.float32],

        # Large K tests
        [2, 1, 1, 8192, 2048, nl.float32],
        [2, 8, 1, 8192, 2048, nl.float32],
        [1, 4, 5, 8192, 2048, nl.float32],
        [2, 1, 1, 25600, 2048, nl.float32],

        # Large BxS + large vocab (Qwen-class)
        [2, 2, 1, 151936, 256, nl.float32],
        [2, 1024, 1, 8192, 2048, nl.float32],
        [2, 1024, 1, 3568, 2048, nl.float32],
    ]
    # fmt: on
    topk_unit_perms.extend(large_batch_perms)

    @pytest.mark.fast
    @pytest.mark.parametrize(topk_unit_params, topk_unit_perms)
    def test_topk_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        is_negative_test_case = False
        with assert_negative_test_case(is_negative_test_case):
            self.run_topk_test(
                test_manager=test_manager,
                lnc_degree=lnc_degree,
                batch=batch,
                seqlen=seqlen,
                vocab=vocab_size,
                k=K,
                dtype=dtype,
            )

    topk_unsorted_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"
    topk_unsorted_perms = [
        [2, 1, 1, 25136, 256, nl.float32],
        [2, 8, 5, 4058, 256, nl.float32],
        [2, 5, 5, 4058, 256, nl.float32],
    ]

    @pytest.mark.fast
    @pytest.mark.parametrize(topk_unsorted_params, topk_unsorted_perms)
    def test_topk_unsorted(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        self.run_topk_test(
            test_manager=test_manager,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            sorted=False,
        )


# ===================== Edge Case Validation =====================


class TopkEdgeCaseValidator:
    """Validates topk configuration constraints and classifies test cases."""

    PMAX = 128
    MAX_FREE_DIM = 2**14
    DVE_MAX_ALUS = 8

    @staticmethod
    def is_valid_config(batch: int, seqlen: int, vocab_size: int, k: int, lnc_degree: int) -> FilterResult:
        """Return FilterResult for a given topk configuration."""
        BxS = batch * seqlen

        if vocab_size < k:
            return FilterResult.INVALID
        if k < 1:
            return FilterResult.INVALID
        if BxS < 1:
            return FilterResult.INVALID

        n_prgs = 1 if BxS == 1 else lnc_degree
        per_lnc_BxS = (BxS + n_prgs - 1) // n_prgs

        max_n_stages = TopkEdgeCaseValidator.PMAX
        if max_n_stages < 1:
            return FilterResult.INVALID

        ideal_n_stages = math.ceil(min(k, vocab_size) / TopkEdgeCaseValidator.DVE_MAX_ALUS)
        min_n_stages_hw = math.ceil(vocab_size / TopkEdgeCaseValidator.MAX_FREE_DIM)

        n_stages = min(max_n_stages, ideal_n_stages)
        n_stages = max(n_stages, min_n_stages_hw)

        if n_stages > max_n_stages:
            return FilterResult.INVALID

        stage_free_size = math.ceil(vocab_size / n_stages)
        if stage_free_size > TopkEdgeCaseValidator.MAX_FREE_DIM:
            return FilterResult.INVALID

        local_k = (
            (math.ceil(k / n_stages) + TopkEdgeCaseValidator.DVE_MAX_ALUS - 1) // TopkEdgeCaseValidator.DVE_MAX_ALUS
        ) * TopkEdgeCaseValidator.DVE_MAX_ALUS
        if stage_free_size + n_stages * local_k > TopkEdgeCaseValidator.MAX_FREE_DIM:
            return FilterResult.INVALID

        return FilterResult.VALID


# ===================== Sweep Config =====================


def sweep_topk_config():
    """Returns BoundedRange objects for topk sweep parameters."""
    return {
        "batch": BoundedRange([1, 8, 16, 32, 128, 512, 1024], boundary_values=[]),
        "seqlen": BoundedRange([1, 5, 7], boundary_values=[]),
        "vocab": BoundedRange([256, 3168, 4058, 8192, 16000, 25600], boundary_values=[]),
        "K": BoundedRange([1, 8, 64, 128, 256, 2048], boundary_values=[]),
        "lnc_degree": BoundedRange([1, 2], boundary_values=[]),
    }


def filter_topk_combinations(batch, seqlen, vocab, K, lnc_degree):
    """Filter function for coverage_parametrize: validates HW constraints."""
    return TopkEdgeCaseValidator.is_valid_config(batch, seqlen, vocab, K, lnc_degree=lnc_degree)


# ===================== Sweep Tests =====================


@pytest_test_metadata(
    name="TopK Sweep",
    pytest_marks=["topk"],
)
class TestTopKSweep:
    @pytest.mark.coverage_parametrize(
        **sweep_topk_config(),
        filter=filter_topk_combinations,
        coverage="pairs",
    )
    def test_topk_sweep(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        lnc_degree,
        batch,
        seqlen,
        vocab,
        K,
        is_negative_test_case,
    ):
        test_cls = TestTopKKernel()

        with assert_negative_test_case(is_negative_test_case):
            test_cls.run_topk_test(
                test_manager=test_manager,
                lnc_degree=lnc_degree,
                batch=batch,
                seqlen=seqlen,
                vocab=vocab,
                k=K,
                dtype=nl.float32,
            )
