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
        description="Attention CTE Kernels",
        tags=["attention", "cte", "ranged"],
        pytest_marks=["attention", "cte"]
    )
    class TestRangedAttentionCTEKernels:
        def test_something(self):
            pass

The decorator:
- Stores metadata as a class attribute (__pytest_test_metadata__)
- Dynamically applies pytest marks to enable filtering with pytest -m
- Supports regex parsing for external tools
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from test.utils.test_validation_utils import (
    IntegrationFileCollector,
    get_decorator_kwargs,
    get_decorator_name,
)
from typing import Any, Dict, List, Optional, Union

import pytest


def pytest_test_metadata(name: str, pytest_marks: Optional[List[str]] = None, **kwargs: Any):
    """
    Decorator to add metadata to test classes and apply pytest marks.

    This decorator serves two purposes:
    1. Stores structured metadata as a class attribute for programmatic access
    2. Dynamically applies pytest marks to enable test filtering

    Args:
        name: Human-readable name of the test
        description: Human-readable description of what the test class covers
        pytest_marks: List of pytest marker names to dynamically apply to the class
        **kwargs: Additional custom metadata fields

    Returns:
        Decorated class with metadata and pytest marks applied

    Example:
        @pytest_test_metadata(
            name="MM Test"
            pytest_marks=["matmul", "slow"],
            category="core",
            owners=["team-a"]
        )
        class TestMatMul:
            pass

        # Access metadata programmatically
        print(TestMatMul.__pytest_test_metadata__)

        # Run tests with: pytest -m matmul
    """

    def decorator(cls):
        # Store metadata as class attribute for programmatic access
        cls.__pytest_test_metadata__ = {'name': name, 'pytest_marks': pytest_marks or [], **kwargs}

        # Dynamically apply pytest marks to the class
        # This allows using pytest -m <mark> to filter tests
        if pytest_marks:
            for mark_name in pytest_marks:
                # Apply the mark using pytest's marker system
                mark = getattr(pytest.mark, mark_name)
                cls = mark(cls)

        return cls

    return decorator


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
        containing the decorator arguments (name, pytest_marks, etc.).
        decorator arguments. Otherwise, metadata will be None.

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

        import ast

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

    except (SyntaxError, FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
        logging.debug(f"Could not parse {file_path} for test metadata: {e}")

    return results


def discover_pytest_test_metadata_marks(test_root: Path) -> dict[str, str]:
    """
    Auto-discover pytest marks from @pytest_test_metadata decorators in test files.

    Scans all test_*.py files recursively and extracts pytest_marks from
    @pytest_test_metadata decorators using AST parsing.

    Args:
        test_root: Root directory to search for test files

    Returns:
        Dictionary mapping mark names to auto-generated descriptions
    """
    marks = {}

    # Use TestFileCollector to find all test files
    collector = IntegrationFileCollector(test_root)
    for file_info in collector.collect():
        # Extract test class metadata from each file
        test_classes = extract_pytest_test_metadata_from_file(file_info.file_path)

        # Process each test class that has metadata
        for test_class in test_classes:
            if test_class.metadata and 'pytest_marks' in test_class.metadata:
                pytest_marks = test_class.metadata['pytest_marks']

                # pytest_marks should be a list of strings
                if isinstance(pytest_marks, list):
                    for mark_name in pytest_marks:
                        if isinstance(mark_name, str) and mark_name not in marks:
                            marks[mark_name] = "Auto-discovered mark from @pytest_test_metadata"

    return marks
