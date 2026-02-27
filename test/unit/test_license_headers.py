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
Test to ensure all Python files contain the required Apache 2.0 license header.

This test verifies that all .py files in src/ and test/ directories contain
the Apache 2.0 license disclaimer at the top of the file, accounting for
different comment styles and formatting variations.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Expected license text (without comment markers or extra whitespace)
EXPECTED_LICENSE = """
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License").
You may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


def normalize_license_text(text: str) -> str:
    """
    Normalize license text by removing comment characters and extra whitespace.

    Args:
        text: Raw text that may contain comment markers and varied whitespace

    Returns:
        Normalized text with single spaces and no comment markers
    """
    # Remove common Python comment markers (# and """)
    text = re.sub(r"^#\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r'"""', "", text)
    text = re.sub(r"'''", "", text)

    # Normalize whitespace: collapse multiple spaces/newlines to single space
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing whitespace
    text = text.strip()

    return text


def extract_file_header(file_path: Path, num_lines: int = 50) -> str:
    """
    Extract the first N lines from a file for license checking.

    Args:
        file_path: Path to the Python file
        num_lines: Number of lines to extract from the top

    Returns:
        The extracted header text
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = []
            for _ in range(num_lines):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            return "".join(lines)
    except Exception as e:
        return f"Error reading file: {e}"


def check_license_in_file(file_path: Path) -> Tuple[bool, str]:
    """
    Check if a file contains the expected license header.

    Args:
        file_path: Path to the Python file to check

    Returns:
        Tuple of (has_license, error_message)
    """
    # Extract the top portion of the file
    header = extract_file_header(file_path)

    # Normalize both the header and expected license
    normalized_header = normalize_license_text(header)
    normalized_expected = normalize_license_text(EXPECTED_LICENSE)

    # Check if the normalized expected license is in the normalized header
    if normalized_expected in normalized_header:
        return True, ""
    else:
        return False, f"Missing or incorrect license header in: {file_path}"


def get_python_files(root_dir: Path) -> List[Path]:
    """
    Recursively find all Python files in a directory.

    Args:
        root_dir: Root directory to search

    Returns:
        List of Python file paths
    """
    return sorted(root_dir.rglob("*.py"))


def test_license_headers():
    """
    Test that all Python files in src/ and test/ contain the Apache 2.0 license header.

    This test:
    - Checks all .py files in src/ and test/ directories
    - Verifies the license appears at the top of each file
    - Ignores comment character variations and formatting differences
    - Reports all files with missing/incorrect licenses before failing
    """
    # Get the project root directory (three levels up from this test file)
    test_file_path = Path(__file__)
    project_root = test_file_path.parent.parent.parent

    # Directories to check
    directories_to_check = [
        project_root / "src",
        project_root / "test",
    ]

    # Collect all Python files
    all_files = []
    for directory in directories_to_check:
        if directory.exists():
            all_files.extend(get_python_files(directory))

    # Check each file and collect failures
    failures = []
    for file_path in all_files:
        if str(file_path).endswith("version/__init__.py"):
            continue
        has_license, error_msg = check_license_in_file(file_path)
        if not has_license:
            failures.append(error_msg)

    # Report all failures at once
    if failures:
        failure_report = "\n".join(failures)
        error_message = (
            f"\n{'='*70}\n"
            f"License header check failed for {len(failures)} file(s):\n"
            f"{'='*70}\n\n"
            f"{failure_report}\n\n"
            f"{'='*70}\n"
            f"All Python files must include the Apache 2.0 license header\n"
            f"at the top of the file.\n"
            f"{'='*70}\n"
        )
        assert False, error_message

    # Success message
    assert len(all_files) > 0, "No Python files found to check"
    print(f"✓ License header check passed for {len(all_files)} Python file(s)")
