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
Test to ensure all Python files are properly formatted with ruff.

This test runs `ruff format --check` to verify that all Python files
in the project conform to the configured formatting standards.
"""

import subprocess
from pathlib import Path


def test_ruff_format_check():
    # Get the project root directory (two levels up from this test file)
    test_file_path = Path(__file__)
    project_root = test_file_path.parent.parent.parent

    result = subprocess.run(
        ["ruff", "format", "--check", "."],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    ruff_diff = subprocess.run(
        ["ruff", "format", "--diff", "."],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # If ruff format --check returns non-zero, files need formatting
    if result.returncode != 0:
        error_message = (
            "ruff format check failed. The following files need formatting:\n\n"
            f"{result.stdout}\n"
            f"{result.stderr}\n\n"
            "The specific diffs are:\n"
            f"{ruff_diff.stdout}\n\n"
            "To fix formatting issues, run: ruff format ."
        )
        assert False, error_message

    assert result.returncode == 0, "ruff format --check should return 0 for properly formatted code"


def test_ruff_check():
    """
    Test to ensure all Python files pass ruff linting checks.

    This test runs `ruff check` to verify that all Python files
    in the project conform to the configured linting standards.
    """
    # Get the project root directory (two levels up from this test file)
    test_file_path = Path(__file__)
    project_root = test_file_path.parent.parent.parent

    result = subprocess.run(
        ["ruff", "check", "."],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    # If ruff check returns non-zero, there are linting errors
    if result.returncode != 0:
        error_message = (
            "ruff check failed. The following linting errors were found:\n\n"
            f"{result.stdout}\n"
            f"{result.stderr}\n\n"
            "To fix auto-fixable issues, run: ruff check --fix ."
        )
        assert False, error_message

    assert result.returncode == 0, "ruff check should return 0 for code that passes all linting rules"
