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
Unit test to validate that all integration test classes have @pytest_test_metadata decorators.

This test ensures that all test classes (classes named Test*) in the integration test
suite have the required @pytest_test_metadata decorator for proper test discovery and metadata
tracking.
"""

import ast
from pathlib import Path

from test.utils.pytest_test_metadata import extract_pytest_test_metadata_from_file
from test.utils.test_validation_utils import (
    IntegrationFileCollector,
    ValidationErrorReporter,
    ValidationViolation,
    fail_with_report,
    get_decorator_name,
)

# Fix instructions for @pytest_test_metadata decorator
METADATA_FIX_INSTRUCTIONS = [
    "Add the @pytest_test_metadata decorator to each test class listed above.",
    "",
    "1. Import the decorator at the top of your test file:",
    "   from test.utils.pytest_test_metadata import pytest_test_metadata",
    "",
    "2. Add the decorator above your test class:",
    "",
    "   @pytest_test_metadata(",
    "       name=\"Descriptive Test Name\",",
    "   )",
    "   @pytest_marks([\"category\", \"subcategory\"])",
    "   class TestYourKernel:",
    "       ...",
    "",
    "3. Choose appropriate pytest marks that describe your test:",
    "   - Kernel type: attention, mlp, qkv, rope, rmsnorm, output_projection",
    "   - Test type: cte, tkg, common",
    "   - Other: slow, fast, integration",
    "",
    "Example from test/integration/nkilib/core/attention/test_attention_cte.py:",
    "",
    "   @pytest_test_metadata(",
    "       name=\"Attention CTE\",",
    "   )",
    "   @pytest_marks([\"attention\", \"cte\"])",
    "   @final",
    "   class TestRangedAttentionCTEKernels:",
    "       ...",
]


def test_all_integration_test_classes_have_metadata():
    """
    Verify that all Test* classes in test/integration/nkilib/core have @pytest_test_metadata.

    This test scans all Python test files in test/integration/nkilib/core and ensures
    that every test class (classes starting with "Test") has the @pytest_test_metadata decorator.

    The test will report ALL missing decorators rather than failing on the first one,
    allowing developers to fix all issues in a single pass.

    If this test fails, add the @pytest_test_metadata decorator to the reported test classes.
    Example:

        from test.utils.pytest_test_metadata import pytest_test_metadata

        @pytest_test_metadata(
            name="Your Test Name",
        )
        class TestYourKernel:
            pass

    For more examples, see:
        - test/integration/nkilib/core/attention/test_attention_cte.py
        - test/integration/nkilib/core/attention/test_attention_tkg.py
    """
    # Collect test files using shared collector
    integration_core_dir = IntegrationFileCollector.get_integration_core_dir()
    collector = IntegrationFileCollector(integration_core_dir)
    test_files = collector.collect()

    # Ensure we found test files
    assert len(test_files) > 0, (
        f"No test files found in {integration_core_dir}. "
        "This might indicate the test is looking in the wrong directory."
    )

    # Build error reporter
    reporter = ValidationErrorReporter(
        error_title="Found test classes missing @pytest_test_metadata decorator",
    )
    reporter.add_fix_instructions(METADATA_FIX_INSTRUCTIONS)

    # Scan each test file
    for file_info in test_files:
        # Extract test class metadata from the file
        test_classes = extract_pytest_test_metadata_from_file(file_info.file_path)
        has_metadata = any(tc.metadata is not None for tc in test_classes)

        # Each file with test classes must have at least one @pytest_test_metadata
        if test_classes and not has_metadata:
            reporter.add_violation(
                ValidationViolation(
                    file_path=file_info.file_path,
                    message=f"No @pytest_test_metadata found ({len(test_classes)} test class(es))",
                )
            )

    # If there are violations, fail with detailed message
    if reporter.has_violations():
        fail_with_report(reporter.build())

    # Report success
    total_classes = sum(len(extract_pytest_test_metadata_from_file(f.file_path)) for f in test_files)
    print(f"\n✓ All {total_classes} test classes in {len(test_files)} files have @pytest_test_metadata decorators")


def _count_pytest_test_metadata(file_path: Path) -> int:
    """Count @pytest_test_metadata decorators in a file (on any class or function).

    Each @pytest_test_metadata occurrence is discovered as a separate test entry, regardless
    of whether it decorates a class or a test function, so this counts both.
    AST-based to avoid matching the name in docstrings/imports/strings.
    """
    tree = ast.parse(Path(file_path).read_text(encoding="utf-8"), filename=str(file_path))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        for decorator in node.decorator_list
        if get_decorator_name(decorator) == "pytest_test_metadata"
    )


def test_one_metadata_per_file():
    """
    Enforce that each test file under integration/nkilib has at most one
    @pytest_test_metadata annotation.

    Each @pytest_test_metadata annotation is discovered as a separate test entry across the
    whole integration/nkilib tree (not just core). Multiple annotations in the same file
    cause duplicate entries that re-run the same file. Use @pytest_marks for additional
    test classes/functions instead.

    Files with no @pytest_test_metadata (e.g. experimental/not-ready tests) are unaffected.
    """
    # Scan all of integration/nkilib (core + experimental + private), matching test discovery.
    nkilib_dir = IntegrationFileCollector.get_integration_core_dir().parent
    collector = IntegrationFileCollector(nkilib_dir)
    test_files = collector.collect()

    assert len(test_files) > 0, (
        f"No test files found in {nkilib_dir}. This might indicate the test is looking in the wrong directory."
    )

    reporter = ValidationErrorReporter(
        error_title="Found test files with multiple @pytest_test_metadata annotations",
    )
    reporter.add_fix_instructions(
        [
            "Each test file must have exactly ONE @pytest_test_metadata decorator.",
            "",
            "Only the first test class/function should have @pytest_test_metadata.",
            "Other test classes/functions should only use @pytest_marks for marking:",
            "",
            "   @pytest_test_metadata(",
            '       name="My Kernel",',
            "   )",
            '   @pytest_marks(["kernel", "core"])',
            "   class TestMyKernelBasic:",
            "       ...",
            "",
            '   @pytest_marks(["kernel", "core", "advanced"])',
            "   class TestMyKernelAdvanced:",
            "       ...",
        ]
    )

    for file_info in test_files:
        count = _count_pytest_test_metadata(file_info.file_path)
        if count > 1:
            reporter.add_violation(
                ValidationViolation(
                    file_path=file_info.file_path,
                    message=f"Found {count} @pytest_test_metadata annotations (expected at most 1)",
                )
            )

    if reporter.has_violations():
        fail_with_report(reporter.build())

    print(f"\n✓ All {len(test_files)} test files have at most one @pytest_test_metadata annotation")
