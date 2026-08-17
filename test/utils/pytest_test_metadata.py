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
Utilities for adding metadata to test classes.

This module provides a decorator for annotating test classes with structured metadata
that can be used for test discovery, filtering, and reporting.

Example usage:
    @pytest_test_metadata(
        name="Attention CTE Kernels",
        tags=["attention", "cte", "ranged"],
    )
    @pytest_marks(["attention", "cte"])
    class TestRangedAttentionCTEKernels:
        def test_something(self):
            pass

The decorator:
- Stores metadata as a class attribute (__pytest_test_metadata__)
- Supports regex parsing for external tools

The pytest_marks decorator:
- Dynamically applies pytest marks to enable filtering with pytest -m
"""

from __future__ import annotations

import ast
import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pytest

from .test_validation_utils import (
    IntegrationFileCollector,
    get_decorator_args,
    get_decorator_kwargs,
    get_decorator_name,
)


def pytest_marks(marks: List[str]):
    """
    Decorator to apply pytest marks to a test class.

    This decorator is the recommended way to apply dynamic pytest marks to test
    classes. It should be used alongside @pytest_test_metadata (which handles
    metadata only) to maintain clean separation of concerns.

    Args:
        marks: List of pytest marker names to apply to the class.

    Returns:
        Decorated class with pytest marks applied.

    Example:
        @pytest_test_metadata(
            name="MM Test",
            tags=["core"],
        )
        @pytest_marks(["matmul", "slow"])
        class TestMatMul:
            pass

        # Run tests with: pytest -m matmul
    """

    def decorator(cls):
        for mark_name in marks:
            mark = getattr(pytest.mark, mark_name)
            cls = mark(cls)
        return cls

    return decorator


def pytest_test_metadata(name: str, **kwargs: Any):
    """
    Decorator to add metadata to test classes.

    Stores structured metadata as a class attribute for programmatic access.
    Use @pytest_marks for applying pytest marks.

    Args:
        name: Human-readable name of the test
        **kwargs: Additional custom metadata fields

    Returns:
        Decorated class with metadata applied

    Example:
        @pytest_test_metadata(
            name="MM Test",
            category="core",
            owners=["team-a"]
        )
        @pytest_marks(["matmul", "slow"])
        class TestMatMul:
            pass

        # Access metadata programmatically
        print(TestMatMul.__pytest_test_metadata__)
    """

    def decorator(cls):
        cls.__pytest_test_metadata__ = {'name': name, **kwargs}
        return cls

    return decorator


_TEST_PACKAGE_SEGMENT = ("integration", "nkilib")
_CORE_FAMILY = "core"


def derive_labeled_kernel_name(test_path: Path | str, metadata_name: str | None) -> str | None:
    """Prepend the capitalized family folder to non-core kernels, to match the pipeline's KernelName."""
    if not metadata_name:
        return None
    parts = Path(test_path).parts
    try:
        integration_idx = parts.index(_TEST_PACKAGE_SEGMENT[0])
    except ValueError:
        return metadata_name
    family_idx = integration_idx + len(_TEST_PACKAGE_SEGMENT)
    if family_idx + 1 >= len(parts) or parts[integration_idx + 1] != _TEST_PACKAGE_SEGMENT[-1]:
        return metadata_name
    family = parts[family_idx]
    if family == _CORE_FAMILY:
        return metadata_name
    return f"{family.capitalize()} {metadata_name}"


@dataclass
class TestMetadataInfo:
    """Information about a test class and its metadata."""

    file_path: Path
    class_name: str
    line_number: int
    metadata: Optional[Dict[str, Any]]


def extract_pytest_test_metadata_from_file(file_path: Union[str, Path]) -> List[TestMetadataInfo]:
    """
    Extract test metadata from a single test file.

    Scans a Python file for test classes (classes named Test*) and extracts
    their @pytest_test_metadata decorator information if present.

    Args:
        file_path: Path to the Python test file to analyze

    Returns:
        List of TestMetadataInfo objects, one for each Test* class found.
        If a class has @pytest_test_metadata, the metadata field will be a dict
        containing the decorator arguments (name, etc.). Otherwise, metadata will be None.

    Example:
        >>> results = extract_pytest_test_metadata_from_file('test_example.py')
        >>> for info in results:
        ...     if info.metadata is None:
        ...         print(f"{info.class_name} is missing @pytest_test_metadata")
    """
    file_path = Path(file_path)
    results = []

    # Use TestFileCollector to parse the file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))

        # Walk through all top-level nodes
        for node in ast.iter_child_nodes(tree):
            # Look for class definitions that start with "Test"
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                metadata = None

                # Check decorators on the class
                for decorator in node.decorator_list:
                    decorator_name = get_decorator_name(decorator)
                    if decorator_name == 'pytest_test_metadata':
                        # Extract keyword arguments using shared utility
                        metadata = get_decorator_kwargs(decorator)

                results.append(
                    TestMetadataInfo(
                        file_path=file_path, class_name=node.name, line_number=node.lineno, metadata=metadata
                    )
                )

    except (SyntaxError, OSError, UnicodeDecodeError) as e:
        # OSError covers a missing file, a permission error, or a directory
        # being passed instead of a file (IsADirectoryError). Any unreadable
        # path degrades to "no metadata" rather than aborting the caller.
        logging.debug(f"Could not parse {file_path} for test metadata: {e}")

    return results


@functools.lru_cache(maxsize=None)
def resolve_file_kernel_name(test_path: Path | str) -> str | None:
    """Resolve the KernelName for a test file from its single @pytest_test_metadata.

    Metadata is defined once per file because it drives the per-file approval
    step, so the runtime KernelName must be file-scoped, not class-scoped —
    otherwise a second-or-later class in a multi-class file emits no KernelName.
    Cached because it re-parses the file.
    """
    metadata_name = next(
        (tc.metadata["name"] for tc in extract_pytest_test_metadata_from_file(test_path) if tc.metadata),
        None,
    )
    return derive_labeled_kernel_name(test_path, metadata_name)


def discover_pytest_test_metadata_marks(test_root: Path) -> dict[str, str]:
    """
    Auto-discover pytest marks from @pytest_marks decorators in test files.

    Scans all test_*.py files recursively and extracts marks from
    @pytest_marks decorators using AST parsing.

    Args:
        test_root: Root directory to search for test files

    Returns:
        Dictionary mapping mark names to auto-generated descriptions
    """
    marks = {}

    # Use TestFileCollector to find all test files
    collector = IntegrationFileCollector(test_root)
    for file_info in collector.collect():
        # Extract pytest_marks from each file
        for mark_name in _extract_pytest_marks_from_file(file_info.file_path):
            if mark_name not in marks:
                marks[mark_name] = "Auto-discovered mark from @pytest_marks"

    return marks


def _extract_pytest_marks_from_file(file_path: Path) -> list[str]:
    """Extract all mark names from @pytest_marks decorators in a file."""
    try:
        tree = ast.parse(file_path.read_text(encoding='utf-8'), filename=str(file_path))
    except (SyntaxError, FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
        logging.debug(f"Could not parse {file_path} for pytest marks: {e}")
        return []

    result = []
    for node in ast.iter_child_nodes(tree):
        if not (isinstance(node, ast.ClassDef) and node.name.startswith('Test')):
            continue
        for decorator in node.decorator_list:
            if get_decorator_name(decorator) != 'pytest_marks':
                continue
            args = get_decorator_args(decorator)
            if args and isinstance(args[0], list):
                result.extend(m for m in args[0] if isinstance(m, str))
    return result
