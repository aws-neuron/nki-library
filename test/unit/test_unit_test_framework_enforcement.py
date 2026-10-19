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
Enforce UnitTestFramework usage for ALL integration test files.

Rules enforced:
1. Must NOT call test_manager.execute() directly — use UnitTestFramework.run_test()
2. Must use torch_ref instead of golden_* functions (unless file also uses UnitTestFramework,
   where golden_* helpers inside custom validators are acceptable)

Legacy files that predate the migration are in LEGACY_ALLOWLIST.
This allowlist can ONLY shrink — new files may never be added.
"""

import ast
from pathlib import Path
from typing import List, Set

from test.utils.test_validation_utils import fail_with_report

REPO_ROOT = Path(__file__).parent.parent.parent
INTEGRATION_TEST_DIR = REPO_ROOT / "test" / "integration"

# ── Legacy allowlist: files that still use deprecated patterns ──
# Remove entries as they are migrated.
# Don't add new entries unless you have good reason.
# Comments indicate which class/function still needs migration.
LEGACY_ALLOWLIST: Set[str] = set()


def _check_file_violations(file_path: Path) -> List[str]:
    """Return list of violation descriptions using AST analysis."""
    source = file_path.read_text()
    tree = ast.parse(source, filename=str(file_path))

    violations = []
    has_framework = False
    has_golden_def = False
    golden_lines: list[int] = []

    for node in ast.walk(tree):
        # Detect UnitTestFramework(...) instantiation
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "UnitTestFramework":
                has_framework = True
            elif isinstance(func, ast.Attribute) and func.attr == "UnitTestFramework":
                has_framework = True

        # Rule 1: no test_manager.execute() calls
        if isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "execute"
                and isinstance(func.value, ast.Name)
                and func.value.id == "test_manager"
            ):
                violations.append(
                    f"L{node.lineno}: Calls test_manager.execute() directly — use UnitTestFramework.run_test()"
                )

        # Rule 2: no golden_* function definitions
        if isinstance(node, ast.FunctionDef) and node.name.startswith("golden_"):
            has_golden_def = True
            golden_lines.append(node.lineno)

    # golden_* is only a violation if the file does NOT use UnitTestFramework
    if has_golden_def and not has_framework:
        for lineno in golden_lines:
            violations.append(f"L{lineno}: Defines golden_* function without UnitTestFramework — use torch_ref instead")

    return violations


def test_all_integration_tests_use_unit_test_framework():
    """
    Every integration test file must follow UnitTestFramework conventions
    unless it is in LEGACY_ALLOWLIST.
    """
    all_violations = []
    for file_path in sorted(INTEGRATION_TEST_DIR.rglob("test_*.py")):
        rel = str(file_path.relative_to(REPO_ROOT))
        if rel in LEGACY_ALLOWLIST:
            continue
        violations = _check_file_violations(file_path)
        if violations:
            all_violations.append(f"  {rel}")
            for v in violations:
                all_violations.append(f"    - {v}")

    if all_violations:
        fail_with_report(
            "\n\nFiles violating UnitTestFramework guidelines:\n"
            + "\n".join(all_violations)
            + "\n\nFix: migrate to UnitTestFramework (see test/docs/unit_test_guide.md)\n"
        )


def test_legacy_allowlist_no_stale_entries():
    """
    If a legacy file has been migrated (no more violations), it should
    be removed from LEGACY_ALLOWLIST to keep the list shrinking.
    """
    stale = []
    for rel in sorted(LEGACY_ALLOWLIST):
        file_path = REPO_ROOT / rel
        if not file_path.exists():
            stale.append(f"  {rel}  (file deleted)")
            continue
        if not _check_file_violations(file_path):
            stale.append(f"  {rel}  (no violations remaining)")

    if stale:
        fail_with_report("\n\nStale LEGACY_ALLOWLIST entries — remove these:\n" + "\n".join(stale) + "\n")
