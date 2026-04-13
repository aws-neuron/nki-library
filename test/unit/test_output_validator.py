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
"""Unit tests for output validator and terminal visualizer."""

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import torch

from ..utils.common_dataclasses import (
    CustomValidator,
    CustomValidatorWithOutputTensorData,
    KernelArgs,
    ValidationArgs,
)
from ..utils.output_validator import OutputValidator
from ..utils.tensor_histogram import TensorHistogram


class TestTensorHistogram:
    """Test cases for TensorHistogram."""

    def test_visualizer_initialization(self):
        """Test that visualizer initializes with correct defaults."""
        viz = TensorHistogram()
        assert viz.histogram_width == 120
        assert viz.num_bins == 100

    def test_visualizer_custom_params(self):
        """Test that visualizer accepts custom parameters."""
        viz = TensorHistogram(histogram_width=80, num_bins=20)
        assert viz.histogram_width == 80
        assert viz.num_bins == 20

    def test_print_histogram_with_valid_data(self):
        """Test histogram printing with valid data."""
        viz = TensorHistogram()
        data = torch.randn(1000, dtype=torch.float32)

        # Should not raise any exceptions
        viz.print_histogram(data, "Test Histogram")

    def test_print_histogram_with_empty_data(self):
        """Test histogram printing with empty data."""
        viz = TensorHistogram()
        data = torch.tensor([])

        # Should handle empty data gracefully
        viz.print_histogram(data, "Empty Histogram")

    def test_print_histogram_with_nan_inf(self):
        """Test histogram printing with NaN and Inf values."""
        viz = TensorHistogram()
        data = torch.tensor([1.0, 2.0, float('nan'), float('inf'), float('-inf'), 3.0])

        # Should filter out NaN/Inf and plot valid data
        viz.print_histogram(data, "NaN/Inf Histogram")

    def test_print_comparison_histogram(self):
        """Test overlaid comparison histogram."""
        viz = TensorHistogram()
        actual = torch.randn(1000, dtype=torch.float32)
        expected = actual + torch.randn(1000, dtype=torch.float32) * 0.1

        # Should not raise any exceptions
        viz.print_comparison_histogram(actual, expected, "Comparison Test")

    def test_print_comparison_stats(self):
        """Test statistics table printing."""
        viz = TensorHistogram()
        actual = torch.randn(100, 100, dtype=torch.float32)
        expected = actual + torch.randn(100, 100, dtype=torch.float32) * 0.01

        # Should not raise any exceptions
        viz.print_comparison_stats(actual, expected, atol=1e-5, rtol=1e-3)

    def test_print_full_comparison_report(self):
        """Test full comparison report."""
        viz = TensorHistogram()
        actual = torch.randn(100, 100, dtype=torch.float32)
        expected = actual + torch.randn(100, 100, dtype=torch.float32) * 0.01

        # Should not raise any exceptions
        viz.print_full_comparison_report(actual, expected, "test_output", atol=1e-5, rtol=1e-3, passed=True)

    def test_print_to_logfile(self):
        """Test that output is written to logfile."""
        viz = TensorHistogram()
        actual = torch.randn(100, dtype=torch.float32)
        expected = actual + torch.randn(100, dtype=torch.float32) * 0.01

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            logfile_path = f.name

        try:
            with open(logfile_path, 'w') as logfile:
                viz.print_full_comparison_report(
                    actual, expected, "test_output", atol=1e-5, rtol=1e-3, passed=True, logfile=logfile
                )

            # Verify logfile was written
            assert os.path.exists(logfile_path)
            with open(logfile_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                assert "COMPARISON: test_output" in content
        finally:
            if os.path.exists(logfile_path):
                os.remove(logfile_path)

    def test_print_full_comparison_report_quantile_too_large(self):
        """Test full comparison report."""
        viz = TensorHistogram()
        actual = torch.randn(5000, 5000, dtype=torch.float32)
        expected = actual + torch.randn(5000, 5000, dtype=torch.float32) * 0.01

        # Should not raise any exceptions
        viz.print_full_comparison_report(actual, expected, "test_output", atol=1e-5, rtol=1e-3, passed=True)


class FailingCustomValidator(CustomValidator):
    """A custom validator that always fails."""

    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
        return False


class PassingCustomValidator(CustomValidator):
    """A custom validator that always passes."""

    def validate(self, inference_output: npt.NDArray[Any]) -> bool:
        return True


class TestOutputValidatorCustomValidator:
    """Tests for OutputValidator with custom validators."""

    def _create_mock_emitter(self):
        """Create a mock emitter with metrics collector."""
        collector = MagicMock()
        collector.timer.return_value.__enter__ = MagicMock()
        collector.timer.return_value.__exit__ = MagicMock()
        emitter = MagicMock()
        emitter.get_collector.return_value = collector
        return emitter

    def test_custom_validator_failure_does_not_raise_name_error(self, tmp_path):
        """
        Test that a failing custom validator raises AssertionError, not NameError.

        This is a regression test for a bug where failing custom validators would
        cause a NameError because 'expected_output' was not defined in the custom
        validator code path, but was referenced in the error handling code.
        """
        # Create a temporary output file
        output_file = tmp_path / "test_output.bin"
        test_data = np.array([1, 2, 3], dtype=np.uint8)
        test_data.tofile(output_file)

        # Create kernel args with a failing custom validator
        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            emitter=self._create_mock_emitter(),
            validation_args=ValidationArgs(
                golden_output={
                    "test_output": CustomValidatorWithOutputTensorData(
                        validator=FailingCustomValidator,
                        output_ndarray=np.zeros(3, dtype=np.uint8),
                    )
                },
            ),
        )

        validator = OutputValidator(
            kernels_args=kernel_args,
            output_file_list=[str(output_file)],
        )

        # Create a logfile path for the test
        logfile_path = str(tmp_path / "log-validate.txt")

        # The validator should raise AssertionError for failed validation,
        # NOT NameError for undefined 'expected_output'
        try:
            validator.validate(logfile_path=logfile_path)
            assert False, "Expected AssertionError to be raised"
        except AssertionError as e:
            assert "Validation failed" in str(e), f"Expected 'Validation failed' error, got: {e}"
        except NameError as e:
            assert False, f"Got NameError instead of AssertionError - this is the bug: {e}"

        # Verify that no golden-*.bin file was created (since custom validators
        # don't have expected_output to save)
        golden_files = list(tmp_path.glob("golden-*.bin"))
        assert len(golden_files) == 0, "No golden file should be saved for custom validators"

    def test_custom_validator_success(self, tmp_path):
        """Test that a passing custom validator completes without error."""
        # Create a temporary output file
        output_file = tmp_path / "test_output.bin"
        test_data = np.array([1, 2, 3], dtype=np.uint8)
        test_data.tofile(output_file)

        # Create kernel args with a passing custom validator
        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            emitter=self._create_mock_emitter(),
            validation_args=ValidationArgs(
                golden_output={
                    "test_output": CustomValidatorWithOutputTensorData(
                        validator=PassingCustomValidator,
                        output_ndarray=np.zeros(3, dtype=np.uint8),
                    )
                },
            ),
        )

        validator = OutputValidator(
            kernels_args=kernel_args,
            output_file_list=[str(output_file)],
        )

        # Should complete without error
        validator.validate()
