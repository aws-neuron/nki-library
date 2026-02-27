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
Test to ensure all intra-package imports use relative imports instead of absolute imports.

This test walks through all Python files in src/nkilib_src/nkilib and verifies
that imports referencing modules within the same package use relative import syntax
(e.g., `from .module import foo`) instead of absolute imports
(e.g., `from nkilib.module import foo`).
"""

import ast
from pathlib import Path
from typing import List, NamedTuple, Union


class AbsoluteImportViolation(NamedTuple):
    """Represents a single absolute import violation."""

    filepath: Path
    lineno: int
    import_statement: str
    suggestion: str


# Package prefixes that should use relative imports when imported from within the package
INTRA_PACKAGE_PREFIXES = (
    "nkilib.",
    "nkilib_src.",
)

# Directories to exclude from checks
EXCLUDED_DIR_PATTERNS = {
    "__pycache__/*",
    ".pytest_cache/*",
    ".mypy_cache/*",
    ".venv/*",
    "build/*",
    "*.egg-info/",
    "nkilib/__init__.py",
}

SRC_DIR = "src"
NKILIB_NAMESPACE = "nkilib_src"
NKILIB_DIR = "nkilib"


def should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped based on excluded patterns."""
    return any(path.match(pattern) for pattern in EXCLUDED_DIR_PATTERNS)


def get_import_statement(node: Union[ast.ImportFrom, ast.Import]) -> str:
    """Reconstruct the import statement from an AST node."""
    if isinstance(node, ast.ImportFrom):
        names = ", ".join(alias.asname if alias.asname else alias.name for alias in node.names)
        return f"from {node.module} import {names}"
    elif isinstance(node, ast.Import):
        names = ", ".join(alias.name if not alias.asname else f"{alias.name} as {alias.asname}" for alias in node.names)
        return f"import {names}"


def compute_relative_import_suggestion(module: str, file_path: Path, package_root: Path) -> str:
    """
    Compute a suggested relative import for a given absolute import.

    Args:
        module: The absolute module path (e.g., 'nkilib.core.utils')
        file_path: Path to the file containing the import
        package_root: Root path of the package (e.g., src/nkilib_src/nkilib)

    Returns:
        A suggestion string for the relative import
    """
    # Get the directory containing the importing file, relative to package root
    try:
        file_relative = file_path.parent.relative_to(package_root)
    except ValueError:
        return "from .<module> import ..."

    # Count how many levels up we need to go
    file_parts = list(file_relative.parts)
    file_depth = len(file_parts)

    # Parse the module path - strip the package prefix
    module_parts = module.split(".")

    # Remove the package prefix (nkilib or nkilib_src.nkilib)
    if module.startswith("nkilib_src.nkilib."):
        module_parts = module_parts[2:]  # Remove 'nkilib_src', 'nkilib'
    elif module.startswith("nkilib_src."):
        module_parts = module_parts[1:]  # Remove 'nkilib_src'
    elif module.startswith("nkilib."):
        module_parts = module_parts[1:]  # Remove 'nkilib'

    # Find common prefix between file path and module path
    common_depth = 0
    for i, (file_part, mod_part) in enumerate(zip(file_parts, module_parts)):
        if file_part == mod_part:
            common_depth = i + 1
        else:
            break

    # Calculate levels up needed
    levels_up = file_depth - common_depth

    # Build the relative import path
    if levels_up == 0:
        # Same directory or subdirectory
        remaining_module = ".".join(module_parts[common_depth:])
        if remaining_module:
            return f"from .{remaining_module} import ..."
        else:
            return "from . import ..."
    else:
        # Need to go up directories
        dots = "." * (levels_up + 1)
        remaining_module = ".".join(module_parts[common_depth:])
        if remaining_module:
            return f"from {dots}{remaining_module} import ..."
        else:
            return f"from {dots} import ..."


def check_file_for_absolute_imports(file_path: Path, package_root: Path) -> List[AbsoluteImportViolation]:
    """
    Check a single Python file for absolute imports that should be relative.

    Args:
        file_path: Path to the Python file to check
        package_root: Root path of the package

    Returns:
        List of AbsoluteImportViolation for each violation found
    """
    violations = []

    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError) as e:
        # Skip files that can't be parsed
        print(f"Warning: Could not parse {file_path}: {e}")
        return violations

    for node in ast.walk(tree):
        # check "import xxx as xxx"
        if isinstance(node, ast.Import):
            for alias in node.names:
                if any((alias.name + ".").endswith(prefix) for prefix in INTRA_PACKAGE_PREFIXES):
                    violations.append(
                        AbsoluteImportViolation(
                            filepath=file_path,
                            lineno=node.lineno,
                            import_statement=get_import_statement(node),
                            suggestion="remove importing top-level module",
                        )
                    )
                elif any(alias.name.startswith(prefix) for prefix in INTRA_PACKAGE_PREFIXES):
                    parent_alias = ".".join(alias.name.split(".")[:-1])
                    violations.append(
                        AbsoluteImportViolation(
                            filepath=file_path,
                            lineno=node.lineno,
                            import_statement=get_import_statement(node),
                            suggestion=compute_relative_import_suggestion(parent_alias, file_path, package_root),
                        )
                    )
        # check "from xxx import xxx"
        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (level > 0 means relative)
            if node.level > 0:
                continue

            # Skip if no module (e.g., `from . import foo`)
            if node.module is None:
                continue

            # Check if this is an intra-package absolute import
            if any(node.module.startswith(prefix) for prefix in INTRA_PACKAGE_PREFIXES):
                import_stmt = get_import_statement(node)
                suggestion = compute_relative_import_suggestion(node.module, file_path, package_root)
                violations.append(
                    AbsoluteImportViolation(
                        filepath=file_path,
                        lineno=node.lineno,
                        import_statement=import_stmt,
                        suggestion=suggestion,
                    )
                )

    return violations


def test_relative_imports():
    """
    Test that all intra-package imports use relative import syntax.

    This test:
    - Checks all .py files in src/nkilib_src/nkilib/
    - Identifies absolute imports that reference modules within the same package
    - Reports all violations with line numbers and suggested fixes
    - Fails if any absolute intra-package imports are found
    """
    # Get the project root directory
    test_file_path = Path(__file__)
    project_root = test_file_path.parent.parent.parent

    # Directory to check
    package_root = project_root / SRC_DIR / NKILIB_NAMESPACE / NKILIB_DIR

    assert package_root.exists(), (
        f"Expected package directory does not exist at {package_root}. "
        "This test may have been moved or the package structure has changed."
    )

    # Collect all Python files
    all_files = sorted(package_root.rglob("*.py"))
    all_files = [f for f in all_files if not should_skip_path(f)]

    # Check each file and collect violations
    all_violations: List[AbsoluteImportViolation] = []
    for file_path in all_files:
        violations = check_file_for_absolute_imports(file_path, package_root)
        all_violations.extend(violations)

    # Report all violations at once
    if all_violations:
        # Group violations by file for better readability
        violations_by_file: dict[Path, List[AbsoluteImportViolation]] = {}
        for v in all_violations:
            violations_by_file.setdefault(v.filepath, []).append(v)

        error_lines = [
            f"\n{'=' * 70}",
            f"Relative import check failed: {len(all_violations)} violation(s) "
            f"in {len(violations_by_file)} file(s)",
            f"{'=' * 70}\n",
        ]

        for filepath, violations in sorted(violations_by_file.items()):
            relative_path = filepath.relative_to(project_root)
            error_lines.append(f"\n{relative_path}:")
            for v in sorted(violations, key=lambda x: x.lineno):
                error_lines.append(f"  Line {v.lineno}: {v.import_statement}")
                error_lines.append(f"    Suggestion: {v.suggestion}")

        error_lines.extend(
            [
                f"\n{'=' * 70}",
                "All imports within the package should use relative import syntax.",
                "Example: `from .module import foo` instead of `from nkilib.module import foo`",
                f"{'=' * 70}\n",
            ]
        )

        assert False, "\n".join(error_lines)

    # Success message
    assert len(all_files) > 0, "No Python files found to check"
    print(f"✓ Relative import check passed for {len(all_files)} Python file(s)")
