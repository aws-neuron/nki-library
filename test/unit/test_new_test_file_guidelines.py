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
Unit test to enforce test framework migration rules for NEW test files.

Rules enforced (only for new files not yet committed to mainline):
1. Must use UnitTestFramework (not raw pytest)
2. Must use torch_ref instead of golden_* functions
3. Must NOT use deprecated RangeTestConfig
4. Must use @pytest.mark.coverage_parametrize for sweeping tests

Files can opt-out by including @IGNORE_NEW_TEST_GUIDELINES comment.
"""

import re
import subprocess
from pathlib import Path
from typing import List, Set

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
INTEGRATION_TEST_DIR = REPO_ROOT / "test" / "integration"
IGNORE_MARKER = "@IGNORE_NEW_TEST_GUIDELINES"

# Patterns to detect violations
RANGE_TEST_CONFIG_IMPORT = re.compile(r"from\s+test\.utils\.ranged_test_harness\s+import.*RangeTestConfig")
RANGE_TEST_CONFIG_DECORATOR = re.compile(r"@range_test_config")
GOLDEN_FUNCTION_DEF = re.compile(r"def\s+golden_\w+\s*\(")

FIX_INSTRUCTIONS = [
    "New test files must follow the updated test framework guidelines.",
    "",
    "Rule 1: Use UnitTestFramework",
    "   from test.utils.unit_test_framework import UnitTestFramework",
    "",
    "Rule 2: Use torch_ref instead of golden_* functions",
    "   # BAD:  def golden_my_kernel(...): ...",
    "   # GOOD: torch_ref=my_torch_reference_function",
    "",
    "Rule 3: Do NOT use deprecated RangeTestConfig",
    "   # BAD:  from test.utils.ranged_test_harness import RangeTestConfig",
    "   # GOOD: Use @pytest.mark.coverage_parametrize instead",
    "",
    "Rule 4: Use coverage_parametrize for sweeping tests",
    "   @pytest.mark.coverage_parametrize(...)",
    "   def test_my_kernel(self, ...):",
    "",
    "See test/docs/llm_coding_guidelines_review/nki-hard-guidelines.md for details.",
    "",
    "To opt-out, add comment: # @IGNORE_NEW_TEST_GUIDELINES",
]


def _get_new_test_files() -> Set[Path]:
    """Get test files that are new (not in mainline)."""
    new_files = set()
    try:
        # Method 1: Check staged/uncommitted new files via git status
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                # A = staged new, ?? = untracked
                if line.startswith("A ") or line.startswith("A  ") or line.startswith("?? "):
                    file_path = line[3:].strip() if line.startswith("?? ") else line[2:].strip()
                    if file_path.startswith("test/integration/") and file_path.endswith(".py") and "test_" in file_path:
                        new_files.add(REPO_ROOT / file_path)

        # Method 2: Check files added since origin/mainline (for committed but not pushed)
        result = subprocess.run(
            ["git", "diff", "--name-only", "--diff-filter=A", "origin/mainline", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.startswith("test/integration/") and line.endswith(".py") and "test_" in line:
                    new_files.add(REPO_ROOT / line)

        return new_files
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return new_files


def _check_file_violations(file_path: Path) -> List[str]:
    """Check a file for guideline violations. Returns list of violation messages."""
    content = file_path.read_text()

    # Skip if opt-out marker present
    if IGNORE_MARKER in content:
        return []

    violations = []

    # Rule 1: Must import UnitTestFramework
    if "from test.utils.unit_test_framework import" not in content:
        if "UnitTestFramework" not in content:
            violations.append("Missing UnitTestFramework import")

    # Rule 2: No golden_* function definitions
    if GOLDEN_FUNCTION_DEF.search(content):
        violations.append("Uses golden_* function (use torch_ref instead)")

    # Rule 3: No RangeTestConfig
    if RANGE_TEST_CONFIG_IMPORT.search(content) or RANGE_TEST_CONFIG_DECORATOR.search(content):
        violations.append("Uses deprecated RangeTestConfig (use coverage_parametrize)")

    # Rule 4: Must use coverage_parametrize for sweep tests
    if "sweep" in file_path.name.lower() or "_sweep" in content.lower():
        if "coverage_parametrize" not in content:
            violations.append("Sweep test missing @pytest.mark.coverage_parametrize")

    return violations


def test_new_test_files_follow_guidelines():
    """
    Verify that NEW test files follow the updated test framework guidelines.

    This check only applies to files that are new (not yet in mainline).
    Existing files are grandfathered and exempt from these rules.
    """
    new_files = _get_new_test_files()

    if not new_files:
        pytest.skip("No new test files to check")

    all_violations = []
    for file_path in sorted(new_files):
        if not file_path.exists():
            continue
        violations = _check_file_violations(file_path)
        if violations:
            rel_path = file_path.relative_to(REPO_ROOT)
            all_violations.append(f"  • {rel_path}")
            for v in violations:
                all_violations.append(f"      - {v}")

    if all_violations:
        msg = [
            "",
            "=" * 80,
            "ERROR: New test files violate test framework guidelines",
            "=" * 80,
            "",
        ]
        msg.extend(all_violations)
        msg.append("")
        msg.append("-" * 80)
        msg.append("HOW TO FIX:")
        msg.append("-" * 80)
        msg.extend(FIX_INSTRUCTIONS)
        msg.append("=" * 80)
        pytest.fail("\n".join(msg))
