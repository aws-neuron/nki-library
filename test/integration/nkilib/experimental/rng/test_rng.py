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

"""Integration tests for RNG kernels (get/set state, generate random)."""

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.rng import (
    generate_random,
    generate_random_fast,
    get_rng_state_gpsimd,
    set_rng_state_gpsimd,
)
from nkilib_src.nkilib.experimental.rng.rng import NUM_RNG_SEEDS
from nkilib_src.nkilib.experimental.rng.rng_torch import (
    generate_random_fast_torch_ref,
    generate_random_torch_ref,
    get_rng_state_gpsimd_torch_ref,
    set_rng_state_gpsimd_torch_ref,
)
from scipy.stats import chi2 as chi2_dist

from test.utils.common_dataclasses import (
    CompilerArgs,
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    Platforms,
    ValidationArgs,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _state_output(kernel_input):
    """Return expected output shape for state kernels: zeros matching input tensor_state."""
    return {"output_0": np.zeros_like(kernel_input["tensor_state"])}


def _set_state_output(kernel_input):
    """Return expected output shape for set_rng_state: int32 view of the echoed seeds."""
    return {"output_0": np.zeros(kernel_input["tensor_state"].shape, dtype=np.int32)}


def _generate_set_state_input(**kwargs):
    """Generate a (1, NUM_RNG_SEEDS) uint32 tensor with nonzero seeds."""
    np.random.seed(42)
    seeds = np.random.randint(1, 2**31, size=(1, NUM_RNG_SEEDS), dtype=np.uint32)
    return {"tensor_state": seeds}


def _generate_random_input(n_elements, **kwargs):
    """Generate input for generate_random kernel."""
    return {
        "output.must_alias_input": np.zeros((1, n_elements), dtype=np.int32),
        "n_elements": n_elements,
    }


def _random_output(kernel_input):
    """Return expected output shape for generate_random: zeros of shape (1, n_elements)."""
    return {"output_0": np.zeros((1, kernel_input["n_elements"]), dtype=np.int32)}


# fmt: off
FAST_RANDOM_PARAMS = "n_elements"
FAST_RANDOM_VALUES = [
    pytest.param(64,   id="64"),
    pytest.param(1024, id="1024"),
    pytest.param(4096, id="4096"),
]

STATISTICAL_RANDOM_PARAMS = "n_elements"
STATISTICAL_RANDOM_VALUES = [
    pytest.param(1024, id="1024"),
    pytest.param(4096, id="4096"),
]
# fmt: on


def _uniform_int32_validator(n_elements):
    """Factory returning a CustomValidator that checks uniform int32 distribution via chi-squared test."""
    MAX_BINS = 20
    MIN_SAMPLES_PER_BIN = 5
    CHI2_P_VALUE_THRESHOLD = 0.001

    class UniformInt32Validator(CustomValidator):
        def validate(self, inference_output):
            data = np.frombuffer(inference_output, dtype=np.int32).flatten()[:n_elements].astype(np.float64)

            # Normalize to [0, 1]
            int32_min = np.float64(-(2**31))
            int32_max = np.float64(2**31 - 1)
            normalized = (data - int32_min) / (int32_max - int32_min)

            # Chi-squared goodness-of-fit against uniform distribution
            num_bins = min(MAX_BINS, n_elements // MIN_SAMPLES_PER_BIN)
            observed, _ = np.histogram(normalized, bins=num_bins, range=(0, 1))
            expected_count = n_elements / num_bins
            chi2_stat = np.sum((observed - expected_count) ** 2 / expected_count)
            p_value = 1 - chi2_dist.cdf(chi2_stat, df=num_bins - 1)

            self._print_with_log(f"Chi2={chi2_stat:.2f}, p={p_value:.6f}, bins={num_bins}, n={n_elements}")

            if p_value < CHI2_P_VALUE_THRESHOLD:
                self._print_with_log(f"FAIL: Chi-squared rejected uniformity (p={p_value:.6f})")
                return False

            return True

    return UniformInt32Validator


def _generate_get_state_input(**kwargs):
    """Generate a (1, NUM_RNG_SEEDS) uint32 zero tensor for get_rng_state_gpsimd."""
    return {"tensor_state": np.zeros((1, NUM_RNG_SEEDS), dtype=np.uint32)}


def _get_state_validator():
    """Factory returning a CustomValidator that checks get_rng_state output is valid uint32 seeds."""

    class GetStateValidator(CustomValidator):
        def validate(self, inference_output):
            data = np.frombuffer(inference_output, dtype=np.uint32).reshape(1, NUM_RNG_SEEDS)
            self._print_with_log(f"RNG state: {data}")
            self._print_with_log(f"Shape: {data.shape}, dtype: {data.dtype}")
            if data.shape != (1, NUM_RNG_SEEDS):
                self._print_with_log(f"FAIL: Expected shape (1, {NUM_RNG_SEEDS}), got {data.shape}")
                return False
            return True

    return GetStateValidator


@pytest_test_metadata(
    name="RNG",
    pytest_marks=["rng"],
)
class TestRngKernels:
    """Test class for RNG kernels."""

    @pytest.mark.fast
    def test_get_rng_state_gpsimd_fast(self, test_manager: Orchestrator, platform_target: Platforms):
        """Test that get_rng_state_gpsimd returns a valid (1, 6) uint32 state tensor."""

        def input_generator(test_config):
            return _generate_get_state_input()

        validation_args = ValidationArgs(
            golden_output={
                "output_0": CustomValidatorWithOutputTensorData(
                    validator=_get_state_validator(),
                    output_ndarray=np.zeros((1, NUM_RNG_SEEDS), dtype=np.uint32),
                ),
            },
        )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=get_rng_state_gpsimd,
            torch_ref=torch_ref_wrapper(get_rng_state_gpsimd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_state_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            custom_validation_args=validation_args,
        )

    @pytest.mark.fast
    def test_set_rng_state_gpsimd_fast(self, test_manager: Orchestrator, platform_target: Platforms):
        """Test that set_rng_state_gpsimd zeros the input tensor."""

        def input_generator(test_config):
            return _generate_set_state_input()

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=set_rng_state_gpsimd,
            torch_ref=torch_ref_wrapper(set_rng_state_gpsimd_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_set_state_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=0,
            rtol=0,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(FAST_RANDOM_PARAMS, FAST_RANDOM_VALUES)
    def test_generate_random_fast(self, test_manager: Orchestrator, platform_target: Platforms, n_elements):
        """Test that generate_random produces output of the correct shape (values are non-deterministic)."""

        def input_generator(test_config):
            return _generate_random_input(n_elements)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=generate_random,
            torch_ref=torch_ref_wrapper(generate_random_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_random_output,
        )
        # Random values won't match torch ref — use very loose tolerance
        # This primarily validates compilation and correct output shape
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=2**31,
            rtol=1.0,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(STATISTICAL_RANDOM_PARAMS, STATISTICAL_RANDOM_VALUES)
    def test_generate_random_statistical(self, test_manager: Orchestrator, platform_target: Platforms, n_elements):
        """Validate that generate_random produces uniformly distributed int32 values."""

        def input_generator(test_config):
            return _generate_random_input(n_elements)

        validation_args = ValidationArgs(
            golden_output={
                "output_0": CustomValidatorWithOutputTensorData(
                    validator=_uniform_int32_validator(n_elements),
                    output_ndarray=np.zeros((1, n_elements), dtype=np.int32),
                ),
            },
        )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=generate_random,
            torch_ref=torch_ref_wrapper(generate_random_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_random_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            custom_validation_args=validation_args,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(FAST_RANDOM_PARAMS, FAST_RANDOM_VALUES)
    def test_generate_random_fast_multilane_fast(
        self, test_manager: Orchestrator, platform_target: Platforms, n_elements
    ):
        """Test that generate_random_fast (all-128-lane) produces output of the correct shape."""

        def input_generator(test_config):
            return _generate_random_input(n_elements)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=generate_random_fast,
            torch_ref=torch_ref_wrapper(generate_random_fast_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_random_output,
        )
        # Random values won't match torch ref — use very loose tolerance.
        # This primarily validates compilation and correct output shape.
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            atol=2**31,
            rtol=1.0,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(STATISTICAL_RANDOM_PARAMS, STATISTICAL_RANDOM_VALUES)
    def test_generate_random_fast_multilane_statistical(
        self, test_manager: Orchestrator, platform_target: Platforms, n_elements
    ):
        """Validate that generate_random_fast produces uniformly distributed int32 values.

        This exercises the per-lane seeding: if the 128 GPSIMD lanes were not seeded
        distinctly the output would be 128-periodic and fail the chi-squared uniformity
        check.
        """

        def input_generator(test_config):
            return _generate_random_input(n_elements)

        validation_args = ValidationArgs(
            golden_output={
                "output_0": CustomValidatorWithOutputTensorData(
                    validator=_uniform_int32_validator(n_elements),
                    output_ndarray=np.zeros((1, n_elements), dtype=np.int32),
                ),
            },
        )

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=generate_random_fast,
            torch_ref=torch_ref_wrapper(generate_random_fast_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_random_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target),
            custom_validation_args=validation_args,
        )
