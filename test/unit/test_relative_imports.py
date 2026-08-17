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

Walks through all Python files in two package trees and verifies that imports
referencing modules within the same package use relative import syntax
(e.g., ``from .module import foo``) instead of absolute imports
(e.g., ``from nkilib.module import foo``).

Checked packages:
  - ``src/nkilib_src/nkilib/`` — shipped as nki-library wheel
  - ``test/utils/``            — shipped as nki-library-testing wheel (remapped to nkilib_testing)
"""

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Union

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Package prefixes that should use relative imports from within each package.
# Ordered longest-first so the most-specific prefix matches first.
NKILIB_PREFIXES = ("nkilib_src.nkilib.", "nkilib_src.", "nkilib.")
TEST_UTILS_PREFIXES = ("test.utils.", "test.")

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

# Path-substring patterns matched on the posix path string. Used for deeply-nested
# carve-outs that Path.match cannot express because Path.match has no ** glob support.
EXCLUDED_PATH_SUBSTRINGS = (
    # neurotile tutorials use absolute imports for standalone runnability
    # (they're scripts under examples/, not importable package modules).
    "/neurotile/examples/",
)


class AbsoluteImportViolation(NamedTuple):
    """Represents a single absolute import violation."""

    filepath: Path
    lineno: int
    import_statement: str
    suggestion: str


def should_skip_path(path: Path) -> bool:
    """Check if a path should be skipped based on excluded patterns."""
    if any(path.match(pattern) for pattern in EXCLUDED_DIR_PATTERNS):
        return True
    posix = path.as_posix()
    return any(needle in posix for needle in EXCLUDED_PATH_SUBSTRINGS)


def get_import_statement(node: Union[ast.ImportFrom, ast.Import]) -> str:
    """Reconstruct the import statement from an AST node."""
    if isinstance(node, ast.ImportFrom):
        names = ", ".join(alias.asname if alias.asname else alias.name for alias in node.names)
        return f"from {node.module} import {names}"
    elif isinstance(node, ast.Import):
        names = ", ".join(alias.name if not alias.asname else f"{alias.name} as {alias.asname}" for alias in node.names)
        return f"import {names}"


def _strip_prefix_parts(module: str, prefixes: tuple[str, ...]) -> list[str]:
    """Strip the longest matching prefix and return remaining module parts."""
    parts = module.split(".")
    for prefix in sorted(prefixes, key=len, reverse=True):
        if module.startswith(prefix):
            n = len(prefix.rstrip(".").split("."))
            return parts[n:]
    return parts


def compute_relative_import_suggestion(
    module: str, file_path: Path, package_root: Path, prefixes: tuple[str, ...]
) -> str:
    """
    Compute a suggested relative import for a given absolute import.

    Args:
        module: The absolute module path (e.g., 'nkilib.core.utils')
        file_path: Path to the file containing the import
        package_root: Root path of the package (e.g., src/nkilib_src/nkilib)
        prefixes: Absolute prefixes to strip (e.g., ("nkilib.",))

    Returns:
        A suggestion string for the relative import
    """
    try:
        file_relative = file_path.parent.relative_to(package_root)
    except ValueError:
        return "from .<module> import ..."

    file_parts = list(file_relative.parts)
    file_depth = len(file_parts)
    module_parts = _strip_prefix_parts(module, prefixes)

    common_depth = 0
    for i, (file_part, mod_part) in enumerate(zip(file_parts, module_parts, strict=False)):
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


def check_file_for_absolute_imports(
    file_path: Path, package_root: Path, prefixes: tuple[str, ...]
) -> List[AbsoluteImportViolation]:
    """
    Check a single Python file for absolute imports that should be relative.

    Args:
        file_path: Path to the Python file to check
        package_root: Root path of the package
        prefixes: Absolute prefixes that indicate an intra-package import

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
                if any(alias.name == prefix.rstrip(".") for prefix in prefixes):
                    # Bare top-level import like ``import nkilib`` or ``import test``
                    violations.append(
                        AbsoluteImportViolation(
                            filepath=file_path,
                            lineno=node.lineno,
                            import_statement=get_import_statement(node),
                            suggestion="remove importing top-level module",
                        )
                    )
                elif any(alias.name.startswith(prefix) for prefix in prefixes):
                    parent_alias = ".".join(alias.name.split(".")[:-1])
                    violations.append(
                        AbsoluteImportViolation(
                            filepath=file_path,
                            lineno=node.lineno,
                            import_statement=get_import_statement(node),
                            suggestion=compute_relative_import_suggestion(
                                parent_alias, file_path, package_root, prefixes
                            ),
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
            if any(node.module.startswith(prefix) for prefix in prefixes):
                import_stmt = get_import_statement(node)
                suggestion = compute_relative_import_suggestion(node.module, file_path, package_root, prefixes)
                violations.append(
                    AbsoluteImportViolation(
                        filepath=file_path,
                        lineno=node.lineno,
                        import_statement=import_stmt,
                        suggestion=suggestion,
                    )
                )

    return violations


def _run_check(label: str, package_root: Path, prefixes: tuple[str, ...]):
    """Scan *package_root* for absolute intra-package imports and fail with a report."""
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
        violations = check_file_for_absolute_imports(file_path, package_root, prefixes)
        all_violations.extend(violations)

    # Report all violations at once
    if all_violations:
        # Group violations by file for better readability
        violations_by_file: dict[Path, List[AbsoluteImportViolation]] = {}
        for v in all_violations:
            violations_by_file.setdefault(v.filepath, []).append(v)

        error_lines = [
            f"\n{'=' * 70}",
            f"Relative import check failed ({label}): "
            f"{len(all_violations)} violation(s) in {len(violations_by_file)} file(s)",
            f"{'=' * 70}\n",
        ]

        for filepath, violations in sorted(violations_by_file.items()):
            relative_path = filepath.relative_to(PROJECT_ROOT)
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

        raise AssertionError("\n".join(error_lines))

    # Success message
    assert len(all_files) > 0, "No Python files found to check"
    print(f"✓ Relative import check passed for {len(all_files)} file(s) in {label}")


@dataclass(frozen=True)
class PackageConfig:
    label: str
    package_root: Path
    prefixes: tuple[str, ...]

    def __str__(self):
        return self.label


PACKAGES = [
    PackageConfig("nkilib", PROJECT_ROOT / "src" / "nkilib_src" / "nkilib", NKILIB_PREFIXES),
    PackageConfig("test_utils", PROJECT_ROOT / "test" / "utils", TEST_UTILS_PREFIXES),
]


@pytest.mark.parametrize("pkg", PACKAGES, ids=str)
def test_relative_imports(pkg: PackageConfig):
    """Ensure intra-package imports use relative syntax.

    Checked packages:
      - src/nkilib_src/nkilib/ — shipped as nki-library wheel
      - test/utils/            — shipped as nki-library-testing wheel (remapped to nkilib_testing)
    """
    _run_check(label=pkg.label, package_root=pkg.package_root, prefixes=pkg.prefixes)
