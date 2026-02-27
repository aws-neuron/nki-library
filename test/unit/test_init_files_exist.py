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
Test to ensure all directories under src/nkilib and test/ have __init__.py files.

This test walks all subdirectories of src/nkilib and test/ and verifies that each
directory containing Python files has an associated __init__.py file.
"""

import os
from pathlib import Path

SRC_DIR = "src"
TEST_DIR = "test"
NKILIB_NAMESPACE = "nkilib_src"
NKILIB_DIR = "nkilib"

# Directories to exclude from __init__.py checks
EXCLUDED_DIR_PATTERNS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "build",
}


def test_all_nkilib_directories_have_init_files():
    """
    Verify that all directories under src/nkilib_src/nkilib and test/ have __init__.py files.

    This ensures proper Python package structure and prevents import issues.
    """
    test_dir = Path(__file__).parent.parent
    package_root = test_dir.parent

    # Assert we're targeting the expected directories
    src_nkilib_dir = package_root / SRC_DIR / NKILIB_NAMESPACE / NKILIB_DIR
    test_dir_path = package_root / TEST_DIR

    assert src_nkilib_dir.exists(), (
        f"Expected src/nkilib directory does not exist at {src_nkilib_dir}. "
        "This test may have been moved or the package structure has changed."
    )
    assert test_dir_path.exists(), (
        f"Expected test directory does not exist at {test_dir_path}. "
        "This test may have been moved or the package structure has changed."
    )

    # Directories to check
    dirs_to_check = [src_nkilib_dir, test_dir_path]

    missing_init_files = []
    for check_dir in dirs_to_check:
        if not check_dir.exists():
            continue

        # Walk through all directories
        for root, dirs, files in os.walk(check_dir):
            root_path = Path(root)

            # Remove excluded directories from dirs in-place to prevent os.walk
            # from descending into them (this is the documented way to prune the walk)
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_PATTERNS]

            # Skip if this directory itself is excluded
            if any(part in EXCLUDED_DIR_PATTERNS for part in root_path.parts):
                continue

            # Check if this directory has any Python files (excluding __init__.py)
            has_python_files = [f for f in files if f.endswith(".py") and f != "__init__.py"]

            # If there are Python files or subdirectories, we need an __init__.py
            if has_python_files or len(dirs) > 0:
                init_file = root_path / "__init__.py"
                if not init_file.exists():
                    # Record the path relative to package root for error message
                    relative_path = root_path.relative_to(package_root)
                    missing_init_files.append(str(relative_path))

    # Assert that no directories are missing __init__.py files
    if missing_init_files:
        error_message = f"Found {len(missing_init_files)} directories " f"missing __init__.py files:\n"
        for path in sorted(missing_init_files):
            error_message += f"  - {path}\n"
        error_message += (
            "\nEach directory containing Python files or subdirectories "
            "must have an __init__.py file to be treated as a Python package."
        )
        assert False, error_message
