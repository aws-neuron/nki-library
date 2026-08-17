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
from typing import Any, final

import ml_dtypes
import nki.language as nl
import numpy as np
import numpy.typing as npt
import pytest
from nkilib_src.nkilib.core.topk.rotational_topk import (
    create_rotational_topk_config,
    create_topk_config,
    rotational_topk,
)
from nkilib_src.nkilib.core.topk.rotational_topk_torch import rotational_topk_torch_ref
from typing_extensions import override

try:
    from test.integration.nkilib.core.topk.test_topk_model_config import rotational_topk_model_configs
except ImportError:
    rotational_topk_model_configs = {}

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    ModelTestType,
    Platforms,
    prepare_model_parametrize,
)
from test.utils.comparators import maxAllClose
from test.utils.coverage_parametrized_tests import BoundedRange, FilterResult, assert_negative_test_case
from test.utils.metrics_collector import MetricsCollector
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _get_np_dtype(dtype):
    """Convert NKI dtype to numpy dtype."""
    if dtype == nl.bfloat16:
        return ml_dtypes.bfloat16
    return dtype


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
        if vocab_size == k:
            return FilterResult.INVALID

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


@pytest_test_metadata(name="Rotational TopK")
@pytest_marks(["topk", "rotational"])
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
        inp = kernel_input.get("inp", kernel_input.get("inp.must_alias_input"))
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
        platform_target: Platforms,
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

            # Build config outside kernel — no nl API calls needed
            inp_3d = inputs["inp"].reshape((batch, seqlen, vocab))
            topk_config = create_topk_config(
                inp_shape=inp_3d.shape,
                inp_dtype=dtype,
                k=k,
                sorted=sorted,
                num_programs=lnc_degree,
            )
            inp_reshaped = inp_3d.reshape((topk_config.BxS, topk_config.vocab_size))
            config = create_rotational_topk_config(inp_shape=inp_reshaped.shape, topk_config=topk_config)

            config.log_strategy()

            # When k == vocab_size, the kernel returns inp directly (trivial case),
            # creating a must-alias relationship that Beta 3 runtime requires.
            inp_key = "inp.must_alias_input" if k == vocab else "inp"
            return {inp_key: inp_reshaped, "config": config}

        inputs = input_generator(test_config=None)

        def topk_comparator(golden_dict, output_tensors):
            input_tensor = inputs.get("inp", inputs.get("inp.must_alias_input"))
            golden_values = golden_dict["topk_values"]

            class ValuesValidator(CustomValidator):
                @override
                def validate(self, actual_raw_output: npt.NDArray[Any]):
                    BxS, _ = input_tensor.shape
                    k = inputs["config"].topk_config.k
                    output = np.frombuffer(actual_raw_output, dtype=input_tensor.dtype).reshape(BxS, k)
                    self._print_with_log("Results for topk_values:")
                    values_correct = maxAllClose(
                        np.sort(output, axis=-1),
                        np.sort(golden_values, axis=-1),
                        verbose=1,
                        logfile=self.logfile,
                    )
                    if not values_correct:
                        return False
                    vocab_size = inputs["config"].vocab_size
                    if sorted and k > 1 and k < vocab_size:
                        output_f32 = output.astype(np.float32)
                        diffs = np.diff(output_f32, axis=-1)
                        # diffs > 0 means value increased (violates descending).
                        # Allow ties (diffs == 0) — only flag strict increases.
                        non_descending = int(np.sum(diffs > 0))
                        if non_descending > 0:
                            worst_row = int(np.argmax(np.sum(diffs > 0, axis=-1)))
                            self._print_with_log(
                                f"FAIL: output not in descending order. "
                                f"{non_descending} violations. "
                                f"Worst row {worst_row}: {output_f32[worst_row, :8]}..."
                            )
                            return False
                    return True

            class IndicesValidator(CustomValidator):
                @override
                def validate(self, inference_output: npt.NDArray[Any]):
                    BxS, _ = input_tensor.shape
                    k = inputs["config"].topk_config.k
                    output = np.frombuffer(inference_output, dtype=np.uint32).reshape(BxS, k)
                    val = np.take_along_axis(input_tensor, output.astype(np.uint64), axis=-1)
                    self._print_with_log("Results for topk_indices:")
                    return maxAllClose(
                        np.sort(golden_values, axis=-1),
                        np.sort(val, axis=-1),
                        verbose=1,
                        logfile=self.logfile,
                    )

            return {
                "topk_values": CustomValidatorWithOutputTensorData(
                    validator=ValuesValidator,
                    output_ndarray=output_tensors["topk_values"],
                ),
                "topk_indices": CustomValidatorWithOutputTensorData(
                    validator=IndicesValidator,
                    output_ndarray=output_tensors["topk_indices"],
                ),
            }

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rotational_topk,
            torch_ref=torch_ref_wrapper(rotational_topk_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=self.output_tensors,
        )

        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=lnc_degree,
                platform_target=platform_target,
            ),
            rtol=1e-3,
            atol=1e-5,
            custom_comparator=topk_comparator,
        )

    # fmt: off
    topk_unit_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"
    _ABBREVS = {"lnc_degree": "lnc", "batch": "b", "seqlen": "s", "vocab_size": "v", "K": "K", "dtype": "dt"}

    large_batch_perms  = [
            pytest.param(2, 150, 1, 1000, 50, nl.float32, marks=pytest.mark.fast),
            [2, 200, 1, 2000, 64, nl.float32],
            [2, 1024, 1, 5000, 128, nl.float32],
        ]

    topk_unit_perms = [
        # Llama 3 76B before global gather
        pytest.param(2, 8, 5, 4058, 256, nl.float32, marks=pytest.mark.fast),
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

        # High batch and vocab
        [2, 256, 1, 16384, 256, nl.float32],
        pytest.param(2, 256, 1, 2374, 256, nl.float32, marks=pytest.mark.fast),

        # K generalization nominal
        [2, 1, 1, 3168, 8, nl.float32],
        [2, 1, 1, 3168, 64, nl.float32],
        [2, 1, 1, 3168, 192, nl.float32],

        # K generalization hard
        pytest.param(2, 1, 1, 3168, 1, nl.float32, marks=pytest.mark.fast),
        [2, 1, 1, 3168, 7, nl.float32],
        [2, 1, 1, 3168, 60, nl.float32],
        [2, 1, 1, 3168, 99, nl.float32],

        # Mixed tests
        [1, 1, 7, 3999, 256, nl.float32],
        [1, 1, 63, 3999, 20, nl.float32],
        pytest.param(2, 1, 127, 3999, 256, nl.float32, marks=pytest.mark.fast),
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

    # Full-only: large batch + large vocab (>30s compile)
    topk_full_only_perms = [
        [2, 1024, 1, 8192, 2048, nl.float32],
    ]

    @pytest_parametrize(topk_unit_params, topk_unit_perms, abbrevs=_ABBREVS)
    def test_topk_unit(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
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
                platform_target=platform_target,
                lnc_degree=lnc_degree,
                batch=batch,
                seqlen=seqlen,
                vocab=vocab_size,
                k=K,
                dtype=dtype,
            )

    @pytest_parametrize(topk_unit_params, topk_full_only_perms, abbrevs=_ABBREVS)
    def test_topk_unit_slow(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        self.test_topk_unit(test_manager, collector, platform_target, lnc_degree, batch, seqlen, vocab_size, K, dtype)

    topk_unsorted_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"
    topk_unsorted_perms = [
        [2, 1, 1, 25136, 256, nl.float32],
        [2, 8, 5, 4058, 256, nl.float32],
        pytest.param(2, 5, 5, 4058, 256, nl.float32, marks=pytest.mark.fast),
    ]

    @pytest_parametrize(topk_unsorted_params, topk_unsorted_perms, abbrevs=_ABBREVS)
    def test_topk_unsorted(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            sorted=False,
        )

    # Trivial k == vocab_size fast-return path: the kernel returns the input unchanged
    # with sequential indices, tiling index generation over the partition limit (128).
    # Two configs cover both tiling branches: BxS <= 128 (single full tile, no remainder)
    # and BxS > 128 (full 128-row tile + partial remainder DMA). sorted=False is required
    # (the kernel asserts sorted output is unsupported when k == vocab_size).
    topk_trivial_perms = [
        pytest.param(2, 8, 1, 8, 8, nl.float32, marks=pytest.mark.fast),
        pytest.param(2, 129, 1, 8, 8, nl.float32, marks=pytest.mark.fast),
    ]

    @pytest_parametrize(topk_unit_params, topk_trivial_perms, abbrevs=_ABBREVS)
    def test_topk_trivial_k_equals_vocab(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """k == vocab_size returns the input with sequential indices."""
        self.run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
            sorted=False,
        )

    @pytest.mark.coverage_parametrize(
        **sweep_topk_config(),
        filter=filter_topk_combinations,
        coverage="pairs",
    )
    def test_topk_sweep(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
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
                platform_target=platform_target,
                lnc_degree=lnc_degree,
                batch=batch,
                seqlen=seqlen,
                vocab=vocab,
                k=K,
                dtype=nl.float32,
            )


def _make_model_id(params):
    """Generate a test ID string from a model config parameter list."""
    abbrevs = ["lnc", "b", "s", "v", "K", "dt"]

    def fmt(v):
        if hasattr(v, "value"):
            return v.value
        return v

    return "_".join(f"{k}-{fmt(v)}" for k, v in zip(abbrevs, params, strict=True))


# MODEL TESTING ENTRY POINT
@pytest_marks(["topk", "model"])
@final
class TestTopKModel:
    """Model regression tests for rotational_topk kernel."""

    _TIER0_PARAMS, _TIER0_IDS = (
        prepare_model_parametrize(
            {ModelTestType.TIER0: rotational_topk_model_configs.get(ModelTestType.TIER0, [])},
            id_formatter=_make_model_id,
        )
        if rotational_topk_model_configs
        else ([], [])
    )

    topk_model_params = "lnc_degree, batch, seqlen, vocab_size, K, dtype"

    @pytest.mark.tier0
    @pytest.mark.parametrize(topk_model_params, _TIER0_PARAMS, ids=_TIER0_IDS)
    def test_tier0(
        self,
        test_manager: Orchestrator,
        collector: MetricsCollector,
        platform_target: Platforms,
        lnc_degree,
        batch,
        seqlen,
        vocab_size,
        K,
        dtype,
    ):
        """TIER0: Critical model configs for rotational_topk."""
        TestTopKKernel().run_topk_test(
            test_manager=test_manager,
            platform_target=platform_target,
            lnc_degree=lnc_degree,
            batch=batch,
            seqlen=seqlen,
            vocab=vocab_size,
            k=K,
            dtype=dtype,
        )
