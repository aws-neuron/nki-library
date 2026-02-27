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
Unit tests for DeterminismChecker.
"""

import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from ..utils.common_dataclasses import InferenceArgs, KernelArgs
from ..utils.determinism_checker import DeterminismChecker
from ..utils.exceptions import ValidationException


def create_mock_collector():
    """Create a mock metrics collector with timer context manager."""
    mock = Mock()
    mock.timer = Mock(return_value=nullcontext())
    return mock


def create_output_files(artifact_dir: Path, num_runs: int, file_name: str = "out", make_run_differ: int = -1):
    """
    Helper to create output files following neuron-profile naming convention.

    Args:
        artifact_dir: Directory to create files in
        num_runs: Number of runs to create
        file_name: Base file name (default: "out")
        make_run_differ: If >= 0, make this run's output different (default: -1 = all same)
    """
    # Create identical data for all runs
    reference_data = np.random.rand(100, 200).astype(np.float32)

    for run_idx in range(num_runs):
        if run_idx == 0:
            # First run: base name
            file_path = artifact_dir / file_name
        else:
            # Subsequent runs: base_name.N
            file_path = artifact_dir / f"{file_name}.{run_idx + 1}"

        # Write data (different if specified)
        if run_idx == make_run_differ:
            # Create different data for this run
            different_data = np.random.rand(100, 200).astype(np.float32)
            different_data.tofile(file_path)
        else:
            reference_data.tofile(file_path)


def test_determinism_check_success_two_runs():
    """Test that determinism check passes when 2 runs produce identical output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create 2 identical outputs
        create_output_files(artifact_dir, num_runs=2)

        # Create dummy KernelArgs
        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=2, enable_determinism_check=True),
        )

        # Should not raise
        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=2, collector=create_mock_collector())
        checker.check()


def test_determinism_check_success_five_runs():
    """Test that determinism check passes when 5 runs produce identical output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create 5 identical outputs
        create_output_files(artifact_dir, num_runs=5)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=5, enable_determinism_check=True),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=5, collector=create_mock_collector())
        checker.check()


def test_determinism_check_failure_second_run_differs():
    """Test that determinism check fails when run 1 (out.2) differs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create 3 outputs, make run 1 different
        create_output_files(artifact_dir, num_runs=3, make_run_differ=1)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=3, enable_determinism_check=True),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=3, collector=create_mock_collector())

        with pytest.raises(ValidationException) as exc_info:
            checker.check()

        error_msg = str(exc_info.value)
        assert "Determinism check FAILED" in error_msg
        assert "Tensor: 'out'" in error_msg
        assert "Run 1" in error_msg


def test_determinism_check_failure_middle_run_differs():
    """Test that determinism check fails when run 2 (out.3) differs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create 5 outputs, make run 2 (out.3) different
        create_output_files(artifact_dir, num_runs=5, make_run_differ=2)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=5, enable_determinism_check=True),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=5, collector=create_mock_collector())

        with pytest.raises(ValidationException) as exc_info:
            checker.check()

        error_msg = str(exc_info.value)
        assert "Determinism check FAILED" in error_msg
        assert "Tensor: 'out'" in error_msg
        assert "Run 2" in error_msg


def test_determinism_check_failure_last_run_differs():
    """Test that determinism check fails when last run differs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create 4 outputs, make last run different
        create_output_files(artifact_dir, num_runs=4, make_run_differ=3)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=4, enable_determinism_check=True),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=4, collector=create_mock_collector())

        with pytest.raises(ValidationException) as exc_info:
            checker.check()

        error_msg = str(exc_info.value)
        assert "Determinism check FAILED" in error_msg
        assert "Run 3" in error_msg


def test_determinism_check_multiple_output_tensors_all_match():
    """Test determinism check with multiple output tensors that all match."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create multiple outputs for 3 runs
        create_output_files(artifact_dir, num_runs=3, file_name="out")
        create_output_files(artifact_dir, num_runs=3, file_name="cached_max")

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=3, enable_determinism_check=True),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=3, collector=create_mock_collector())
        checker.check()


def test_determinism_check_multiple_outputs_one_differs():
    """Test that determinism check fails when one of multiple outputs differs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create multiple outputs, make cached_max differ in run 1
        create_output_files(artifact_dir, num_runs=3, file_name="out")
        create_output_files(artifact_dir, num_runs=3, file_name="cached_max", make_run_differ=1)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=3, enable_determinism_check=True),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=3, collector=create_mock_collector())

        with pytest.raises(ValidationException) as exc_info:
            checker.check()

        error_msg = str(exc_info.value)
        assert "Determinism check FAILED" in error_msg
        assert "cached_max" in error_msg
        # Should NOT mention 'out' since that one matches
        assert "Tensor: 'out'" not in error_msg or "✓" in error_msg


def test_determinism_check_ignores_log_files():
    """Test that determinism check ignores log files and other non-output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create outputs
        create_output_files(artifact_dir, num_runs=3)

        # Create log files and ntff files (only once, no .2/.3 versions)
        (artifact_dir / "log-infer.txt").write_text("some log content")
        (artifact_dir / "log-validate.txt").write_text("validation log")
        (artifact_dir / "profile_exec_3.ntff").write_bytes(b"binary ntff data")
        (artifact_dir / "allclose_summary.txt").write_text("allclose results")

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=3, enable_determinism_check=True),
        )

        # Should succeed - only compares 'out' files, ignores logs
        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=3, collector=create_mock_collector())
        checker.check()


def test_determinism_check_different_sizes():
    """Test that determinism check fails when output sizes differ."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create outputs with different sizes
        reference_data = np.random.rand(100, 200).astype(np.float32)
        different_size_data = np.random.rand(100, 300).astype(np.float32)  # Different size

        reference_data.tofile(artifact_dir / "out")
        reference_data.tofile(artifact_dir / "out.2")
        different_size_data.tofile(artifact_dir / "out.3")  # Different!

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=3, enable_determinism_check=True),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=3, collector=create_mock_collector())

        with pytest.raises(ValidationException) as exc_info:
            checker.check()

        error_msg = str(exc_info.value)
        assert "Determinism check FAILED" in error_msg
        assert "Run 2" in error_msg


def test_determinism_check_byte_offset_displayed_on_failure():
    """Test that byte offset and values are displayed in error message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create different outputs
        create_output_files(artifact_dir, num_runs=2, make_run_differ=1)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(num_runs=2, enable_determinism_check=True),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=2, collector=create_mock_collector())

        with pytest.raises(ValidationException) as exc_info:
            checker.check()

        error_msg = str(exc_info.value)
        # Should contain byte offset and values
        assert "First mismatch at byte offset" in error_msg
        assert "Reference: 0x" in error_msg
        assert "Run 1: 0x" in error_msg


def test_determinism_check_with_profile_all_runs_true():
    """Test determinism check works with profile_all_runs=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create 3 identical outputs
        create_output_files(artifact_dir, num_runs=3)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(
                num_runs=3,
                profile_all_runs=True,  # Profile all runs
                enable_determinism_check=True,
            ),
        )

        # Should succeed
        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=3, collector=create_mock_collector())
        checker.check()


def test_determinism_check_with_profile_all_runs_false():
    """Test determinism check works with profile_all_runs=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create 4 identical outputs
        create_output_files(artifact_dir, num_runs=4)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(
                num_runs=4,
                profile_all_runs=False,  # Profile only last run
                enable_determinism_check=True,
            ),
        )

        # Should succeed - determinism check works regardless of profiling config
        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=4, collector=create_mock_collector())
        checker.check()


def test_determinism_check_failure_with_profile_all_false():
    """Test determinism check fails correctly when profile_all_runs=False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create 3 outputs, make run 2 different
        create_output_files(artifact_dir, num_runs=3, make_run_differ=2)

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(
                num_runs=3,
                profile_all_runs=False,  # Profile only last run
                enable_determinism_check=True,
            ),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=3, collector=create_mock_collector())

        with pytest.raises(ValidationException) as exc_info:
            checker.check()

        error_msg = str(exc_info.value)
        assert "Determinism check FAILED" in error_msg
        assert "Run 2" in error_msg


def test_determinism_checker_validates_insufficient_runs():
    """Test that determinism checker raises ValidationException when num_runs < 2."""
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_dir = Path(tmpdir)

        # Create just one output
        create_output_files(artifact_dir, num_runs=1, file_name="out")

        kernel_args = KernelArgs(
            kernel_func=lambda: None,
            inference_args=InferenceArgs(
                num_runs=1,  # Not enough for determinism check!
                profile_all_runs=False,
                enable_determinism_check=True,
            ),
        )

        checker = DeterminismChecker(kernel_args, str(artifact_dir), num_runs=1, collector=create_mock_collector())

        with pytest.raises(ValidationException) as exc_info:
            checker.check()

        error_msg = str(exc_info.value)
        assert "Determinism check requires at least 2 runs" in error_msg
        assert "num_runs=1" in error_msg
