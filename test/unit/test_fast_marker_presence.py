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
Unit test to validate that all integration test files have at least one @fast annotated test.

This test ensures that all test files (test_*.py) in test/integration/nkilib/core have
at least one test method, test class, or test vector annotated with @pytest.mark.fast
for quick CI validation.

Files can opt-out of this check by including the comment @IGNORE_FAST anywhere in the file.
"""

import ast
from test.utils.test_validation_utils import (
    IntegrationFileCollector,
    ParsedFileInfo,
    ValidationErrorReporter,
    ValidationViolation,
    get_decorator_kwargs,
    get_decorator_name,
)
from typing import List

import pytest

# Comment marker to exclude a file from @fast marker validation
IGNORE_FAST_MARKER = "@IGNORE_FAST"

# Fix instructions for @pytest.mark.fast annotation
FAST_MARKER_FIX_INSTRUCTIONS = [
    "Add @pytest.mark.fast to at least one test in each file listed above.",
    "",
    "Option 1: Annotate a specific test method:",
    "",
    "   import pytest",
    "",
    "   class TestMyKernel:",
    "       @pytest.mark.fast",
    "       def test_my_kernel_fast(self):",
    "           # Fast test with minimal test cases",
    "           ...",
    "",
    "Option 2: Annotate the entire test class:",
    "",
    "   import pytest",
    "",
    "   @pytest.mark.fast",
    "   class TestMyKernelFast:",
    "       def test_case_1(self):",
    "           ...",
    "",
    "Option 3: Include 'fast' in @pytest_test_metadata pytest_marks:",
    "",
    "   @pytest_test_metadata(",
    "       name=\"My Test\",",
    "       pytest_marks=[\"category\", \"fast\"],",
    "   )",
    "   class TestMyKernel:",
    "       ...",
    "",
    "Option 4: Add @IGNORE_FAST comment to opt-out of this check:",
    "",
    "   # @IGNORE_FAST - This test file intentionally has no fast tests",
    "   class TestMyKernel:",
    "       ...",
    "",
    "NOTE: @fast tests should be quick to run and cover basic functionality.",
    "      They are used for fast CI validation before running full sweeps.",
]


def _has_fast_marker_on_decorator(decorator: ast.expr) -> bool:
    """
    Check if a decorator is @pytest.mark.fast or @fast.

    Handles various decorator forms:
    - @pytest.mark.fast
    - @pytest.mark.fast()
    - @fast (if imported as 'fast = pytest.mark.fast')
    """
    # Handle @pytest.mark.fast (Attribute form)
    if isinstance(decorator, ast.Attribute):
        if decorator.attr == "fast":
            # Verify it's pytest.mark.fast
            if isinstance(decorator.value, ast.Attribute):
                if decorator.value.attr == "mark":
                    if isinstance(decorator.value.value, ast.Name):
                        return decorator.value.value.id == "pytest"
        return False

    # Handle @pytest.mark.fast() (Call form)
    if isinstance(decorator, ast.Call):
        return _has_fast_marker_on_decorator(decorator.func)

    # Handle @fast (simple Name form, if imported directly)
    if isinstance(decorator, ast.Name):
        return decorator.id == "fast"

    return False


def _check_pytest_test_metadata_for_fast(decorator: ast.expr) -> bool:
    """
    Check if @pytest_test_metadata decorator contains 'fast' in pytest_marks.

    Example:
        @pytest_test_metadata(
            name="Test Name",
            pytest_marks=["attention", "fast"],
        )
    """
    decorator_name = get_decorator_name(decorator)
    if decorator_name != "pytest_test_metadata":
        return False

    kwargs = get_decorator_kwargs(decorator)
    pytest_marks = kwargs.get("pytest_marks", [])

    return isinstance(pytest_marks, list) and "fast" in pytest_marks


def _has_fast_marker(decorators: List[ast.expr]) -> bool:
    """Check if any decorator in the list indicates @fast marking."""
    for decorator in decorators:
        if _has_fast_marker_on_decorator(decorator):
            return True
        if _check_pytest_test_metadata_for_fast(decorator):
            return True
    return False


def _extract_fast_marker_info(file_info: ParsedFileInfo) -> dict:
    """
    Extract information about @fast markers in a test file.

    Returns a dict with:
    - has_fast_class: True if any Test* class has @fast decorator
    - has_fast_method: True if any test_* method has @fast decorator
    - has_fast_in_metadata: True if any @pytest_test_metadata has 'fast' in pytest_marks
    - test_class_names: Set of test class names found
    - test_method_names: Set of test method names found
    """
    result = {
        "has_fast_class": False,
        "has_fast_method": False,
        "has_fast_in_metadata": False,
        "test_class_names": set(),
        "test_method_names": set(),
    }

    # Check class-level decorators
    for test_class in file_info.test_classes:
        result["test_class_names"].add(test_class.name)

        # Check class decorators for @fast
        if _has_fast_marker(test_class.decorator_list):
            result["has_fast_class"] = True

        # Check @pytest_test_metadata for 'fast' in pytest_marks
        for decorator in test_class.decorator_list:
            if _check_pytest_test_metadata_for_fast(decorator):
                result["has_fast_in_metadata"] = True

        # Check methods within the class
        for item in test_class.body:
            if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                result["test_method_names"].add(f"{test_class.name}.{item.name}")

                if _has_fast_marker(item.decorator_list):
                    result["has_fast_method"] = True

    # Check top-level test functions
    for test_func in file_info.test_functions:
        result["test_method_names"].add(test_func.name)

        if _has_fast_marker(test_func.decorator_list):
            result["has_fast_method"] = True

    return result


def test_all_integration_test_files_have_fast_marker():
    """
    Verify that all test files in test/integration/nkilib/core have at least one @fast marker.

    This test scans all Python test files in test/integration/nkilib/core and ensures
    that each file has at least one test annotated with @pytest.mark.fast. This can be
    achieved through:
    - Method-level annotation: @pytest.mark.fast on a test_* method
    - Class-level annotation: @pytest.mark.fast on a Test* class
    - Metadata annotation: 'fast' in pytest_marks of @pytest_test_metadata

    The test will report ALL missing files rather than failing on the first one,
    allowing developers to fix all issues in a single pass.

    If this test fails, add the @pytest.mark.fast decorator to at least one test in
    the reported files.
    """
    # Collect test files using shared collector, filtering out files with @IGNORE_FAST
    integration_core_dir = IntegrationFileCollector.get_integration_core_dir()
    collector = IntegrationFileCollector(integration_core_dir)
    test_files = collector.collect(ignore_marker=IGNORE_FAST_MARKER)

    # Ensure we found test files
    assert len(test_files) > 0, (
        f"No test files found in {integration_core_dir}. "
        "This might indicate the test is looking in the wrong directory."
    )

    # Build error reporter
    reporter = ValidationErrorReporter(
        error_title="Found test files missing @pytest.mark.fast annotation",
    )
    reporter.add_fix_instructions(FAST_MARKER_FIX_INSTRUCTIONS)

    # Scan each test file
    for file_info in test_files:
        info = _extract_fast_marker_info(file_info)

        # Check if the file has any @fast marker
        has_fast = info["has_fast_class"] or info["has_fast_method"] or info["has_fast_in_metadata"]

        if not has_fast:
            reporter.add_violation(
                ValidationViolation(
                    file_path=file_info.file_path,
                    message=f"Classes: {', '.join(sorted(info['test_class_names']))}"
                    if info["test_class_names"]
                    else "",
                    method_names=info["test_method_names"] if info["test_method_names"] else None,
                )
            )

    # If there are violations, fail with detailed message
    if reporter.has_violations():
        pytest.fail(reporter.build())

    # Report success
    print(f"\n✓ All {len(test_files)} test files in integration/nkilib/core have @pytest.mark.fast annotations")
