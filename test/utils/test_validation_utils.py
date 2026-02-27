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
Shared utilities for test validation and annotation checking.

This module provides common functionality for validating test files, including:
- File discovery and filtering
- AST-based decorator/marker extraction
- Standardized error reporting for validation tests

Usage:
    from test.utils.test_validation_utils import (
        IntegrationFileCollector,
        ValidationErrorReporter,
        extract_ast_value,
    )
"""

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

# =============================================================================
# AST Utilities
# =============================================================================


def extract_ast_value(node: ast.expr) -> Any:
    """
    Extract a Python value from an AST node.

    Handles common literal types used in decorator arguments:
    - Constants (strings, numbers, booleans, None)
    - Lists of constants
    - Tuples of constants
    - Dicts of constants

    Args:
        node: AST expression node to extract value from

    Returns:
        The Python value represented by the node, or None if it can't be extracted
    """
    if isinstance(node, ast.Constant):
        return node.value
    elif isinstance(node, ast.List):
        return [extract_ast_value(elt) for elt in node.elts]
    elif isinstance(node, ast.Tuple):
        return tuple(extract_ast_value(elt) for elt in node.elts)
    elif isinstance(node, ast.Dict):
        return {extract_ast_value(k): extract_ast_value(v) for k, v in zip(node.keys, node.values)}
    else:
        return None


def get_decorator_name(decorator: ast.expr) -> Optional[str]:
    """
    Extract the name of a decorator from an AST node.

    Handles various decorator forms:
    - @decorator_name
    - @decorator_name()
    - @module.decorator_name
    - @module.decorator_name()

    Args:
        decorator: AST decorator node

    Returns:
        The decorator name as a string, or None if it can't be determined
    """
    if isinstance(decorator, ast.Call):
        return get_decorator_name(decorator.func)
    elif isinstance(decorator, ast.Name):
        return decorator.id
    elif isinstance(decorator, ast.Attribute):
        return decorator.attr
    return None


def get_decorator_kwargs(decorator: ast.expr) -> Dict[str, Any]:
    """
    Extract keyword arguments from a decorator call.

    Args:
        decorator: AST decorator node (must be ast.Call)

    Returns:
        Dictionary mapping argument names to their values
    """
    if not isinstance(decorator, ast.Call):
        return {}

    result = {}
    for keyword in decorator.keywords:
        if keyword.arg:
            value = extract_ast_value(keyword.value)
            if value is not None:
                result[keyword.arg] = value
    return result


# =============================================================================
# File Collection
# =============================================================================


# Comment marker to exclude a file from validation
IGNORE_MARKER_PREFIX = "@IGNORE_"


@dataclass
class ParsedFileInfo:
    """Information about a test file for validation purposes."""

    file_path: Path
    content: str = ""
    tree: Optional[ast.Module] = None
    test_classes: List[ast.ClassDef] = field(default_factory=list)
    test_functions: List[ast.FunctionDef] = field(default_factory=list)

    def has_marker(self, marker: str) -> bool:
        """Check if the file contains a specific marker comment."""
        return marker in self.content


class IntegrationFileCollector:
    """
    Collects and filters test files for validation.

    This class provides a unified way to discover test files and apply
    common filtering rules (e.g., ignore markers, file patterns).

    Example:
        collector = IntegrationFileCollector(repo_root / "test" / "integration" / "nkilib" / "core")
        for file_info in collector.collect(ignore_marker="@IGNORE_FAST"):
            # Process file_info.test_classes, file_info.test_functions, etc.
            pass
    """

    def __init__(self, test_dir: Path, pattern: str = "test_*.py"):
        """
        Initialize the collector.

        Args:
            test_dir: Root directory to search for test files
            pattern: Glob pattern for test files (default: "test_*.py")
        """
        self.test_dir = test_dir
        self.pattern = pattern

    def collect(
        self,
        ignore_marker: Optional[str] = None,
        file_filter: Optional[Callable[[Path], bool]] = None,
    ) -> List[ParsedFileInfo]:
        """
        Collect test files from the configured directory.

        Args:
            ignore_marker: Optional marker string (e.g., "@IGNORE_FAST"). Files
                          containing this marker will be excluded.
            file_filter: Optional callable to filter files. Should return True
                        to include the file.

        Returns:
            List of TestFileInfo objects for files that pass all filters
        """
        if not self.test_dir.exists():
            logging.warning(f"Test directory not found: {self.test_dir}")
            return []

        results = []
        for file_path in sorted(self.test_dir.rglob(self.pattern)):
            try:
                file_info = self._parse_file(file_path)

                # Apply ignore marker filter
                if ignore_marker and file_info.has_marker(ignore_marker):
                    continue

                # Apply custom filter
                if file_filter and not file_filter(file_path):
                    continue

                # Skip files with no test content
                if not file_info.test_classes and not file_info.test_functions:
                    continue

                results.append(file_info)

            except (SyntaxError, FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
                logging.debug(f"Could not parse {file_path}: {e}")
                continue

        return results

    def _parse_file(self, file_path: Path) -> ParsedFileInfo:
        """Parse a test file and extract test classes and functions."""
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))

        test_classes = []
        test_functions = []

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                test_classes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                test_functions.append(node)

        return ParsedFileInfo(
            file_path=file_path,
            content=content,
            tree=tree,
            test_classes=test_classes,
            test_functions=test_functions,
        )

    @staticmethod
    def get_repo_root() -> Path:
        """
        Get the repository root from the current file location.

        Assumes this file is at test/utils/test_validation_utils.py
        """
        return Path(__file__).parent.parent.parent

    @staticmethod
    def get_integration_core_dir() -> Path:
        """Get the integration test core directory."""
        return IntegrationFileCollector.get_repo_root() / "test" / "integration" / "nkilib" / "core"


# =============================================================================
# Error Reporting
# =============================================================================


@dataclass
class ValidationViolation:
    """Represents a single validation violation."""

    file_path: Path
    message: str
    line_number: Optional[int] = None
    class_name: Optional[str] = None
    method_names: Optional[Set[str]] = None
    additional_info: Optional[Dict[str, Any]] = None


class ValidationErrorReporter:
    """
    Builds standardized error messages for validation test failures.

    This class provides a consistent format for reporting validation
    violations across different test validation checks.

    Example:
        reporter = ValidationErrorReporter(
            error_title="Found test classes missing @pytest_test_metadata decorator",
            repo_root=repo_root,
        )
        reporter.add_violation(violation)
        reporter.add_fix_instructions([
            "Add the @pytest_test_metadata decorator to each class.",
            "",
            "Example:",
            "   @pytest_test_metadata(name='Test Name')",
            "   class TestMyKernel:",
            "       ...",
        ])
        error_message = reporter.build()
    """

    def __init__(
        self,
        error_title: str,
        repo_root: Optional[Path] = None,
    ):
        """
        Initialize the reporter.

        Args:
            error_title: Title describing the validation error
            repo_root: Repository root for making paths relative (optional)
        """
        self.error_title = error_title
        self.repo_root = repo_root or IntegrationFileCollector.get_repo_root()
        self.violations: List[ValidationViolation] = []
        self.fix_instructions: List[str] = []

    def add_violation(self, violation: ValidationViolation) -> None:
        """Add a violation to the report."""
        self.violations.append(violation)

    def add_violations(self, violations: List[ValidationViolation]) -> None:
        """Add multiple violations to the report."""
        self.violations.extend(violations)

    def add_fix_instructions(self, instructions: List[str]) -> None:
        """Add fix instructions to the report."""
        self.fix_instructions = instructions

    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0

    def _relative_path(self, path: Path) -> Path:
        """Make a path relative to repo root if possible."""
        try:
            return path.relative_to(self.repo_root)
        except ValueError:
            return path

    def build(self) -> str:
        """
        Build the complete error message.

        Returns:
            Formatted error message string
        """
        if not self.violations:
            return ""

        lines = [
            "",
            "=" * 80,
            f"ERROR: {self.error_title}",
            "=" * 80,
            "",
            f"Total violations: {len(self.violations)}",
            "",
        ]

        # Add violation details
        for violation in self.violations:
            relative_path = self._relative_path(violation.file_path)

            if violation.line_number:
                lines.append(f"  • {relative_path}:{violation.line_number}")
            else:
                lines.append(f"  • {relative_path}")

            if violation.class_name:
                lines.append(f"    Class: {violation.class_name}")

            if violation.method_names:
                method_list = sorted(violation.method_names)
                if len(method_list) <= 5:
                    lines.append(f"    Methods: {', '.join(method_list)}")
                else:
                    lines.append(f"    Methods: {', '.join(method_list[:5])}")
                    lines.append(f"             ... and {len(method_list) - 5} more")

            if violation.message:
                lines.append(f"    {violation.message}")

            lines.append("")

        # Add fix instructions
        if self.fix_instructions:
            lines.extend(
                [
                    "-" * 80,
                    "HOW TO FIX:",
                    "-" * 80,
                    "",
                ]
            )
            lines.extend(self.fix_instructions)
            lines.append("")
            lines.append("=" * 80)

        return "\n".join(lines)
