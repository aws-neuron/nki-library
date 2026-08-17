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
"""Determine which integration tests are relevant to a given commit's changes.

Uses AST-based reverse import analysis to find transitive dependents of changed
source and test files, then maps them to test directories.

Handles two kinds of cross-file dependencies:
1. Source-to-source: ``from nkilib_src.nkilib.core.x import y`` or relative imports
   within ``src/nkilib_src/nkilib/``.
2. Test-to-test: ``from test.integration.nkilib.core.x import y`` between test files
   under ``test/integration/nkilib/``.
"""

from __future__ import annotations

import ast
import logging
import subprocess
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)

# Source prefix relative to repo root
_SRC_PREFIX = Path("src/nkilib_src/nkilib")
# Test prefix relative to repo root
_TEST_PREFIX = Path("test/integration/nkilib")

# Absolute import prefix used within the nkilib package
_ABSOLUTE_IMPORT_PREFIX = "nkilib_src.nkilib."

# Absolute import prefix used for cross-test imports
_TEST_IMPORT_PREFIX = "test.integration.nkilib."

# Paths under _SRC_PREFIX whose changes should trigger a full test run
_RUN_ALL_SRC_DIRS = {"utils"}

# Non-source paths whose changes should trigger a full test run
_RUN_ALL_PATTERNS = [
    "test/conftest.py",
    "test/utils/",
    "build-tools/",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
]


class RelevantTestFinder:
    """Finds integration test directories relevant to a commit's changes."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self.src_root = self.repo_root / _SRC_PREFIX
        self.test_root = self.repo_root / _TEST_PREFIX
        self._reverse_deps: dict[str, set[str]] | None = None

    def get_relevant_test_dirs(self, commit_ids: str = "HEAD") -> set[str] | None:
        """Get test file paths relevant to one or more commits' changes.

        Args:
            commit_ids: Comma-separated commit IDs or a git range
                (e.g., "abc123,def456", "HEAD", or "abc123..def456").

        Returns:
            set[str]: Absolute paths of relevant test files.
            None: If all tests should run (infrastructure change detected).

        Raises:
            subprocess.CalledProcessError: If git diff fails for a commit ID.
            ValueError: If the commit ID format is invalid.
            RuntimeError: If no changed files or no relevant test paths are found.
        """
        changed_files: list[str] = []
        for commit_id in self._parse_commit_ids(commit_ids):
            changed_files.extend(self._get_changed_files(commit_id))

        # Deduplicate while preserving order
        changed_files = list(dict.fromkeys(changed_files))

        if not changed_files:
            raise RuntimeError(f"No changed files found for commit(s) {commit_ids}")

        logger.info("Changed files in %s: %s", commit_ids, changed_files)

        test_dirs: set[str] = set()
        changed_src_files: list[str] = []
        changed_test_files: list[str] = []

        for f in changed_files:
            fpath = Path(f)

            # Check if this is an infrastructure file that should trigger full run
            if self._is_run_all_file(f):
                logger.info("Infrastructure file changed (%s), running all tests", f)
                return None

            # Check if this is a source file under nkilib
            if self._is_source_file(fpath):
                # Check if it's in a "run all" source directory (e.g., core/utils/)
                rel_to_src = fpath.relative_to(_SRC_PREFIX)
                parts = rel_to_src.parts
                # parts[0] is module (core/experimental/private), parts[1] is kernel group
                if len(parts) >= 2 and parts[1] in _RUN_ALL_SRC_DIRS:
                    logger.info("Shared utility changed (%s), running all tests", f)
                    return None
                changed_src_files.append(f)

            # Check if this is a test file
            elif self._is_test_file(fpath):
                test_dir = self._test_file_to_test_dir(fpath)
                if test_dir:
                    test_dirs.add(test_dir)
                changed_test_files.append(f)

            # Non-Python or unrelated files are ignored

        # Build dependency graph and find affected test dirs for source changes
        if changed_src_files or changed_test_files:
            self._build_reverse_import_index()

        if changed_src_files:
            affected = self._get_transitive_dependents(changed_src_files)
            src_test_dirs = self._map_to_test_dirs(affected)
            test_dirs.update(src_test_dirs)

        # Trace test-to-test dependencies for changed test files
        if changed_test_files:
            affected_tests = self._get_transitive_dependents(changed_test_files)
            test_test_dirs = self._map_to_test_dirs(affected_tests)
            test_dirs.update(test_test_dirs)

        if not test_dirs:
            logger.warning(
                "No relevant test files found for commit(s) %s (changed files: %s), running all tests",
                commit_ids,
                changed_files,
            )
            return None

        logger.info("Relevant test paths: %s", test_dirs)
        return test_dirs

    @staticmethod
    def _parse_commit_ids(commit_ids: str) -> list[str]:
        """Parse and validate commit ID input.

        Accepts:
            - Single commit: "abc123" or "HEAD"
            - Comma-separated: "abc123,def456"
            - Git range: "abc123..def456"
        """
        raw = commit_ids.strip()
        if not raw:
            raise ValueError("Empty commit ID string")

        # Git range syntax: commit_1..commit_2
        if ".." in raw:
            if "," in raw:
                raise ValueError(f"Cannot mix range (..) and comma-separated syntax: {raw}")
            return [raw]  # Pass the range directly to git diff

        # Comma-separated list
        parsed = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if " " in part:
                raise ValueError(
                    f"Invalid commit ID '{part}' (contains spaces). "
                    f"Use comma-separated IDs (e.g., 'abc123,def456') or a range (e.g., 'abc123..def456')."
                )
            parsed.append(part)
        if not parsed:
            raise ValueError(f"No valid commit IDs found in: {raw}")
        return parsed

    def _get_changed_files(self, commit_id: str) -> list[str]:
        """Get files changed in a commit or range, relative to repo root."""
        # If it's a range (abc..def), use it directly; otherwise diff against parent
        if ".." in commit_id:
            diff_spec = commit_id
        else:
            diff_spec = f"{commit_id}~1..{commit_id}"
        result = subprocess.run(
            ["git", "diff", "--name-only", diff_spec],
            capture_output=True,
            text=True,
            cwd=self.repo_root,
            check=True,
        )
        return [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]

    def _is_run_all_file(self, filepath: str) -> bool:
        return any(filepath.startswith(pattern) or filepath == pattern for pattern in _RUN_ALL_PATTERNS)

    def _is_source_file(self, fpath: Path) -> bool:
        return fpath.suffix == ".py" and _is_under(fpath, _SRC_PREFIX)

    def _is_test_file(self, fpath: Path) -> bool:
        return fpath.suffix == ".py" and _is_under(fpath, _TEST_PREFIX)

    def _test_file_to_test_dir(self, fpath: Path) -> str | None:
        """Map a test file path to its absolute file path for filtering.

        Returns the absolute path of the test file itself (not the directory),
        enabling file-level precision in test selection.
        """
        abs_path = self.repo_root / fpath
        if abs_path.is_file():
            return str(abs_path)
        return None

    def _build_reverse_import_index(self) -> dict[str, set[str]]:
        """Scan all source and test files and build a reverse dependency map.

        For each file F that is imported by file G, we record:
        reverse_deps[F] contains G.

        Handles all Python import styles:
        - ``from .foo import bar`` (relative ImportFrom, level > 0)
        - ``from .foo import bar as baz`` (same AST node, alias ignored)
        - ``from nkilib_src.nkilib.core.x import y`` (absolute ImportFrom, level == 0)
        - ``import nkilib_src.nkilib.core.x`` (absolute Import)
        - ``from test.integration.nkilib.core.x import y`` (cross-test ImportFrom)

        Returns:
            The reverse dependency map, built on the first call and cached after.
        """
        if self._reverse_deps is not None:
            return self._reverse_deps

        reverse_deps: dict[str, set[str]] = {}

        # Scan source files for source-to-source imports. We include
        # __init__.py: kernel packages commonly re-export their implementation
        # via ``from .kernel import kernel`` in __init__.py, and tests import
        # the package (``from nkilib_src.nkilib.core.x import ...``). Skipping
        # __init__.py drops the ``__init__.py -> kernel.py`` edge, so an edit
        # to the implementation file resolves to no tests and falls back to a
        # full run. Scanning __init__.py restores that edge.
        for py_file in self.src_root.rglob("*.py"):
            self._scan_file_imports(py_file, reverse_deps)

        # Scan test files for test-to-test imports
        for py_file in self.test_root.rglob("*.py"):
            if py_file.name == "__init__.py" or py_file.name == "conftest.py":
                continue
            self._scan_file_imports(py_file, reverse_deps)

        # Published only once fully built, so a failed scan leaves no partial map.
        self._reverse_deps = reverse_deps
        return reverse_deps

    def _scan_file_imports(self, py_file: Path, reverse_deps: dict[str, set[str]]) -> None:
        """Parse a single file and record its imports in the given reverse dependency map."""
        rel_path = str(py_file.relative_to(self.repo_root))
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (SyntaxError, UnicodeDecodeError):
            logger.warning("Failed to parse %s, skipping", rel_path)
            return

        for node in ast.walk(tree):
            imported_file = None

            if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                imported_file = self._resolve_relative_import(py_file, node.module or "", node.level)

            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(_ABSOLUTE_IMPORT_PREFIX):
                    imported_file = self._resolve_absolute_import(node.module)
                elif node.module.startswith(_TEST_IMPORT_PREFIX):
                    imported_file = self._resolve_test_import(node.module)

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    resolved = None
                    if alias.name.startswith(_ABSOLUTE_IMPORT_PREFIX):
                        resolved = self._resolve_absolute_import(alias.name)
                    elif alias.name.startswith(_TEST_IMPORT_PREFIX):
                        resolved = self._resolve_test_import(alias.name)
                    if resolved:
                        imp_rel = str(resolved.relative_to(self.repo_root))
                        reverse_deps.setdefault(imp_rel, set()).add(rel_path)
                continue

            if imported_file:
                imported_rel = str(imported_file.relative_to(self.repo_root))
                reverse_deps.setdefault(imported_rel, set()).add(rel_path)

    def _resolve_relative_import(self, importing_file: Path, module: str, level: int) -> Path | None:
        """Resolve a relative import to an absolute file path.

        Args:
            importing_file: The file containing the import statement.
            module: The module path after the dots (e.g., "mlp.mlp_parameters").
            level: Number of dots (1=current package, 2=parent, etc.).
        """
        # Start from the importing file's directory
        base = importing_file.parent
        # Go up (level - 1) directories. level=1 means same package (no extra up),
        # level=2 means parent package (up 1), etc.
        for _ in range(level - 1):
            base = base.parent

        if not module:
            return None

        # Convert module path to directory path
        parts = module.split(".")
        target = base / "/".join(parts)

        # Try as a file first (module.py)
        if target.with_suffix(".py").is_file():
            return target.with_suffix(".py")

        # Try as a package (__init__.py)
        if (target / "__init__.py").is_file():
            return target / "__init__.py"

        return None

    def _resolve_absolute_import(self, module: str) -> Path | None:
        """Resolve an absolute import like 'nkilib_src.nkilib.core.x.y' to a file path."""
        # Strip the prefix that maps to src/nkilib_src/nkilib/
        suffix = module[len(_ABSOLUTE_IMPORT_PREFIX) :]
        if not suffix:
            return None
        parts = suffix.split(".")
        target = self.src_root / "/".join(parts)

        if target.with_suffix(".py").is_file():
            return target.with_suffix(".py")
        if (target / "__init__.py").is_file():
            return target / "__init__.py"
        return None

    def _resolve_test_import(self, module: str) -> Path | None:
        """Resolve a test import like 'test.integration.nkilib.core.x.y' to a file path."""
        suffix = module[len(_TEST_IMPORT_PREFIX) :]
        if not suffix:
            return None
        parts = suffix.split(".")
        target = self.test_root / "/".join(parts)

        if target.with_suffix(".py").is_file():
            return target.with_suffix(".py")
        if (target / "__init__.py").is_file():
            return target / "__init__.py"
        return None

    def _get_transitive_dependents(self, changed_files: list[str]) -> set[str]:
        """BFS over reverse dependency graph to find all transitive dependents.

        Stops propagation at test-to-test edges: when a test file is reached
        via the import graph, it is included in the result but its own dependents
        (other test files that import from it) are NOT traversed. This prevents
        shared test utilities from causing unrelated test files to be selected.

        Test-to-test propagation only happens when the test file itself is in
        the initial changed_files list (i.e., it was directly modified).
        """
        if not self._reverse_deps:
            return set(changed_files)

        changed_set = set(changed_files)
        visited: set[str] = set()
        queue = deque(changed_files)

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            # If current is a test file that was NOT directly changed,
            # don't propagate further (test→test edge stop)
            current_is_test = _is_under(Path(current), _TEST_PREFIX)
            if current_is_test and current not in changed_set:
                continue

            for dependent in self._reverse_deps.get(current, set()):
                if dependent not in visited:
                    queue.append(dependent)

        return visited

    def _map_to_test_dirs(self, affected_files: set[str]) -> set[str]:
        """Map affected files to relevant test file paths.

        Instead of mapping to kernel-group directories (which over-selects),
        returns only the test files that are actually in the affected set.
        The BFS already traced imports transitively, so any test file that
        imports (directly or transitively) from a changed source file will
        already be in affected_files.

        Returns absolute paths of individual test files, compatible with the
        startswith() filter in pytest_collection_modifyitems.
        """
        test_paths: set[str] = set()

        for f in affected_files:
            fpath = Path(f)
            if _is_under(fpath, _TEST_PREFIX) and fpath.suffix == ".py":
                abs_path = self.repo_root / fpath
                if abs_path.is_file():
                    test_paths.add(str(abs_path))

        return test_paths


def _is_under(path: Path, prefix: Path) -> bool:
    """Check if path starts with prefix (works for relative paths)."""
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False
