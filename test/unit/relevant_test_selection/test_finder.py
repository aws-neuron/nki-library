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
"""Unit tests for the relevant test finder module.

Tests the dependency graph, file classification, transitive dependency
resolution, and source-to-test directory mapping using the live repo.
"""

import copy
from pathlib import Path
from unittest.mock import patch

import pytest

from test.utils.relevant_test_selection.finder import (
    _SRC_PREFIX,
    _TEST_PREFIX,
    RelevantTestFinder,
    _is_under,
)

# Repo root (package root = four levels up from test/unit/relevant_test_selection/)
REPO_ROOT = Path(__file__).parent.parent.parent.parent


def _rel_test_dirs(test_dirs):
    """Convert absolute test paths to relative directory paths after 'test/integration/nkilib/'.

    Handles both file-level paths (extracts parent directory at kernel-group level)
    and directory-level paths (legacy fallback).
    """
    result = set()
    for d in test_dirs:
        suffix = d.split("test/integration/nkilib/")[1]
        # If it's a file path, extract the kernel-group directory (first 2 components)
        parts = Path(suffix).parts
        if len(parts) >= 2 and parts[-1].endswith(".py"):
            result.add(str(Path(parts[0]) / parts[1]))
        else:
            # Directory path (2 components like "core/mlp")
            result.add(suffix.rstrip("/"))
    return sorted(result)


def _new_finder_with_cached_index(cached_deps):
    """Return a fresh RelevantTestFinder with _reverse_deps pre-populated (deep-copied).

    Avoids the ~1.3s AST rebuild while keeping per-instance isolation: each test
    gets its own dict copy, so mutations cannot leak across tests.
    """
    finder = RelevantTestFinder(repo_root=REPO_ROOT)
    finder._reverse_deps = copy.deepcopy(cached_deps)
    return finder


def _make_reexport_repo(root: Path, kernel: str = "widget") -> dict[str, str]:
    """Build a minimal synthetic repo exercising the package re-export pattern.

    Layout (mirrors how real kernels are wired, without depending on any of
    them): a kernel package whose __init__.py re-exports the implementation,
    and a test that imports the *package* rather than the impl module::

        src/.../core/<kernel>/__init__.py   ->  from .<kernel> import <kernel>
        src/.../core/<kernel>/<kernel>.py   ->  the implementation
        test/.../core/<kernel>/test_<kernel>.py  ->  from ...<kernel> import <kernel>

    Returns the repo-relative paths of the three files.
    """
    src_pkg = root / _SRC_PREFIX / "core" / kernel
    test_pkg = root / _TEST_PREFIX / "core" / kernel
    src_pkg.mkdir(parents=True)
    test_pkg.mkdir(parents=True)

    init_py = src_pkg / "__init__.py"
    impl_py = src_pkg / f"{kernel}.py"
    test_py = test_pkg / f"test_{kernel}.py"

    init_py.write_text(f"from .{kernel} import {kernel}\n", encoding="utf-8")
    impl_py.write_text(f"def {kernel}():\n    return None\n", encoding="utf-8")
    test_py.write_text(
        f"from nkilib_src.nkilib.core.{kernel} import {kernel}\n\n"
        f"def test_{kernel}():\n    assert {kernel}() is None\n",
        encoding="utf-8",
    )

    return {
        "init": str(init_py.relative_to(root)),
        "impl": str(impl_py.relative_to(root)),
        "test": str(test_py.relative_to(root)),
    }


class TestIsUnder:
    def test_path_under_prefix(self):
        assert _is_under(Path("src/nkilib_src/nkilib/core/mlp/mlp.py"), _SRC_PREFIX)

    def test_path_not_under_prefix(self):
        assert not _is_under(Path("test/integration/nkilib/core/mlp/test.py"), _SRC_PREFIX)

    def test_test_path_under_test_prefix(self):
        assert _is_under(Path("test/integration/nkilib/core/attention/test_attn.py"), _TEST_PREFIX)


class TestFileClassification:
    def setup_method(self):
        self.finder = RelevantTestFinder(repo_root=REPO_ROOT)

    def test_source_file_detected(self):
        assert self.finder._is_source_file(Path("src/nkilib_src/nkilib/core/mlp/mlp.py"))

    def test_test_file_not_detected_as_source(self):
        assert not self.finder._is_source_file(Path("test/integration/nkilib/core/mlp/test_mlp.py"))

    def test_test_file_detected(self):
        assert self.finder._is_test_file(Path("test/integration/nkilib/core/mlp/test_mlp.py"))

    def test_non_python_file_not_source(self):
        assert not self.finder._is_source_file(Path("src/nkilib_src/nkilib/core/mlp/README.md"))

    def test_infrastructure_file_conftest(self):
        assert self.finder._is_run_all_file("test/conftest.py")

    def test_infrastructure_file_test_utils(self):
        assert self.finder._is_run_all_file("test/utils/some_helper.py")

    def test_infrastructure_file_build_tools(self):
        assert self.finder._is_run_all_file("build-tools/bin/run_test.sh")

    def test_infrastructure_file_setup_py(self):
        assert self.finder._is_run_all_file("setup.py")

    def test_non_infrastructure_source(self):
        assert not self.finder._is_run_all_file("src/nkilib_src/nkilib/core/mlp/mlp.py")


class TestReverseImportIndex:
    """Tests that the reverse import index correctly captures known dependencies."""

    @pytest.fixture(autouse=True)
    def _inject_finder(self, relevant_test_finder_index):
        self.finder = _new_finder_with_cached_index(relevant_test_finder_index)

    def test_index_is_built(self):
        assert self.finder._reverse_deps is not None
        assert len(self.finder._reverse_deps) > 0

    def test_down_projection_imported_by_mlp_tkg(self):
        down_proj = "src/nkilib_src/nkilib/core/mlp/mlp_tkg/mlp_tkg_down_projection.py"
        deps = self.finder._reverse_deps.get(down_proj, set())
        mlp_tkg_files = [d for d in deps if "mlp_tkg/mlp_tkg.py" in d]
        assert len(mlp_tkg_files) > 0, f"mlp_tkg.py should import down_projection, got: {deps}"

    def test_aliased_import_tracked(self):
        """'from ..moe.moe_tkg.moe_tkg import moe_tkg as _moe_tkg' should still track the dependency."""
        moe_tkg = "src/nkilib_src/nkilib/core/moe/moe_tkg/moe_tkg.py"
        deps = self.finder._reverse_deps.get(moe_tkg, set())
        moe_block_files = [d for d in deps if "moe_block" in d]
        assert len(moe_block_files) > 0, f"moe_block should import moe_tkg (via alias), got: {deps}"

    def test_index_cached_on_rebuild(self):
        first_id = id(self.finder._reverse_deps)
        self.finder._build_reverse_import_index()
        assert id(self.finder._reverse_deps) == first_id

    def test_package_init_reexport_edge_is_indexed(self, tmp_path):
        """A kernel __init__.py that re-exports its implementation
        (``from .widget import widget``) must create an
        __init__.py -> widget.py reverse-dep edge. Without it, editing the
        implementation file resolves to no tests and falls back to a full
        run (see test_impl_file_edit_scopes_to_its_test).

        Uses a synthetic repo so the test does not depend on any real kernel."""
        files = _make_reexport_repo(tmp_path)
        finder = RelevantTestFinder(repo_root=tmp_path)
        finder._build_reverse_import_index()

        deps = finder._reverse_deps.get(files["impl"], set())
        init_files = [d for d in deps if d == files["init"]]
        assert init_files, f"__init__.py should import the impl module, got: {deps}"


class TestResolveAbsoluteImport:
    """Tests resolution of absolute imports like 'nkilib_src.nkilib.core.x.y'."""

    def setup_method(self):
        self.finder = RelevantTestFinder(repo_root=REPO_ROOT)

    def test_resolves_existing_module(self):
        result = self.finder._resolve_absolute_import("nkilib_src.nkilib.core.mlp.mlp_parameters")
        assert result is not None
        assert result.name == "mlp_parameters.py"

    def test_resolves_package_init(self):
        result = self.finder._resolve_absolute_import("nkilib_src.nkilib.core.quantization")
        assert result is not None
        assert result.name == "__init__.py"

    def test_returns_none_for_nonexistent_module(self):
        result = self.finder._resolve_absolute_import("nkilib_src.nkilib.core.nonexistent_module")
        assert result is None

    def test_returns_none_for_prefix_only(self):
        result = self.finder._resolve_absolute_import("nkilib_src.nkilib.")
        assert result is None


class TestTransitiveDependents:
    """Tests BFS traversal of the reverse dependency graph."""

    @pytest.fixture(autouse=True)
    def _inject_finder(self, relevant_test_finder_index):
        self.finder = _new_finder_with_cached_index(relevant_test_finder_index)

    def test_changed_file_included_in_result(self):
        changed = ["src/nkilib_src/nkilib/core/mlp/mlp_tkg/mlp_tkg_down_projection.py"]
        affected = self.finder._get_transitive_dependents(changed)
        assert changed[0] in affected

    def test_leaf_file_only_returns_itself(self):
        # A file that nothing imports should only return itself
        changed = ["src/nkilib_src/nkilib/core/moe/moe_tkg/moe_tkg_torch.py"]
        affected = self.finder._get_transitive_dependents(changed)
        assert changed[0] in affected


class TestMapToTestDirs:
    """Tests affected file to test file mapping.

    _map_to_test_dirs now returns only test files that are already in the
    affected set (placed there by BFS). Source-only inputs return empty
    because the BFS step is what discovers dependent test files.
    """

    def setup_method(self):
        self.finder = RelevantTestFinder(repo_root=REPO_ROOT)

    def test_source_only_returns_empty(self):
        """Source files alone produce no test paths (BFS adds test dependents)."""
        affected = {"src/nkilib_src/nkilib/core/mlp/mlp_tkg/mlp_tkg.py"}
        result = self.finder._map_to_test_dirs(affected)
        assert len(result) == 0

    def test_test_file_in_affected_set_is_returned(self):
        """Test files present in affected set are returned directly."""
        affected = {"test/integration/nkilib/core/mlp/test_mlp_tkg.py"}
        result = self.finder._map_to_test_dirs(affected)
        rel_dirs = _rel_test_dirs(result)
        assert "core/mlp" in rel_dirs

    def test_mixed_source_and_test_returns_only_tests(self):
        """Only test files from the affected set are returned, source files ignored."""
        affected = {
            "src/nkilib_src/nkilib/core/mlp/mlp_tkg/mlp_tkg.py",
            "test/integration/nkilib/core/mlp/test_mlp_tkg.py",
            "test/integration/nkilib/core/moe/moe_tkg/test_moe_tkg.py",
        }
        result = self.finder._map_to_test_dirs(affected)
        rel_dirs = _rel_test_dirs(result)
        assert "core/mlp" in rel_dirs
        assert "core/moe" in rel_dirs

    def test_non_python_files_ignored(self):
        affected = {"README.md", "docs/guide.rst"}
        dirs = self.finder._map_to_test_dirs(affected)
        assert len(dirs) == 0

    def test_nonexistent_test_file_ignored(self):
        """Test file paths that don't exist on disk are not returned."""
        affected = {"test/integration/nkilib/core/mlp/test_nonexistent.py"}
        result = self.finder._map_to_test_dirs(affected)
        assert len(result) == 0


class TestGetRelevantTestDirsEndToEnd:
    """End-to-end tests with mocked git output to avoid git history dependency."""

    @pytest.fixture(autouse=True)
    def _inject_finder(self, relevant_test_finder_index):
        self.finder = _new_finder_with_cached_index(relevant_test_finder_index)

    def _run_with_files(self, changed_files):
        """Run get_relevant_test_dirs with mocked changed files."""
        with patch.object(self.finder, "_get_changed_files", return_value=changed_files):
            return self.finder.get_relevant_test_dirs("mocked")

    def test_single_attention_cte_change(self):
        """Single file change to attention_cte.py should affect attention tests."""
        result = self._run_with_files(
            [
                "src/nkilib_src/nkilib/core/attention/attention_cte.py",
            ]
        )
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert rel_dirs == ["core/attention", "experimental/attention"]

    def test_single_mlp_change(self):
        """Single file change to mlp.py (top-level) should affect mlp, moe, moe_block, and transformer."""
        result = self._run_with_files(
            [
                "src/nkilib_src/nkilib/core/mlp/mlp.py",
            ]
        )
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/mlp" in rel_dirs
        assert "experimental/transformer" in rel_dirs

    def test_single_qkv_cte_change(self):
        """Single file change to qkv_cte.py should affect qkv and transformer.

        qkv_cte is imported by qkv.py, which is imported by transformer_tkg.
        Note: experimental/qkv test was previously reached via test-to-test imports
        from core/qkv test utilities, but test-to-test propagation is now stopped
        unless the test file itself is directly changed.
        """
        result = self._run_with_files(
            [
                "src/nkilib_src/nkilib/core/qkv/qkv_cte.py",
            ]
        )
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/qkv" in rel_dirs
        assert "experimental/transformer" in rel_dirs

    def test_experimental_moe_bwd_change(self):
        """Single file in experimental/moe/bwd/ should only affect experimental/moe."""
        result = self._run_with_files(
            [
                "src/nkilib_src/nkilib/experimental/moe/bwd/bwmm_bwd_dropless.py",
            ]
        )
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert rel_dirs == ["experimental/moe", "experimental/moe_mxfp8"]

    def test_impl_file_edit_scopes_to_its_test(self, tmp_path):
        """Editing a kernel's implementation file must scope to that kernel's
        test, NOT fall back to a full run. The synthetic kernel's test imports
        the package (``from nkilib_src.nkilib.core.widget import ...``), which
        __init__.py re-exports from widget.py. Regression guard for the dropped
        __init__.py -> widget.py edge that previously returned None here.

        Uses a synthetic repo so the test does not depend on any real kernel."""
        files = _make_reexport_repo(tmp_path)
        finder = RelevantTestFinder(repo_root=tmp_path)
        with patch.object(finder, "_get_changed_files", return_value=[files["impl"]]):
            result = finder.get_relevant_test_dirs("mocked")

        assert result is not None, "impl-file edit must not trigger a full run"
        rel_dirs = _rel_test_dirs(result)
        assert "core/widget" in rel_dirs

    def test_infrastructure_change_conftest_returns_none(self):
        """Changes to test/conftest.py should run all tests."""
        result = self._run_with_files(
            [
                "test/conftest.py",
                "test/utils/common_dataclasses.py",
            ]
        )
        assert result is None

    def test_mlp_parameters_change_has_wide_blast_radius(self):
        """Changes to mlp_parameters.py (shared by many kernels) should have wide blast radius.

        mlp_parameters is imported by many source files across mlp, qkv, moe, and transformer.
        Only tests that directly import affected source files are selected (no test-to-test propagation).
        """
        result = self._run_with_files(
            [
                "src/nkilib_src/nkilib/core/mlp/mlp_parameters.py",
            ]
        )
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/mlp" in rel_dirs
        assert "core/qkv" in rel_dirs
        assert "experimental/moe" in rel_dirs
        assert "experimental/transformer" in rel_dirs

    def test_core_utils_change_returns_none(self):
        """Changes to core/utils/ should trigger a full test run."""
        result = self._run_with_files(
            [
                "src/nkilib_src/nkilib/core/utils/allocator.py",
            ]
        )
        assert result is None

    def test_non_python_changes_runs_all(self):
        """Non-Python file changes should fall back to running all tests."""
        result = self._run_with_files(
            [
                "README.md",
                "docs/guide.rst",
            ]
        )
        assert result is None

    def test_test_only_change(self):
        """Only a test file changed in moe_cte should run core/moe tests."""
        result = self._run_with_files(
            [
                "test/integration/nkilib/core/moe/moe_cte/test_moe_bwmm_mx_cte.py",
            ]
        )
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert rel_dirs == ["core/moe"]

    def test_down_projection_affects_mlp_and_moe(self):
        """Changes to mlp_tkg_down_projection.py should affect mlp and transformer.

        The source-level import chain is: down_projection → mlp_tkg → mlp → transformer_tkg.
        core/moe and core/moe_block are NOT reached via source imports (they were
        previously reached only via test-to-test utility sharing).
        """
        result = self._run_with_files(
            [
                "src/nkilib_src/nkilib/core/mlp/mlp_tkg/mlp_tkg_down_projection.py",
            ]
        )
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/mlp" in rel_dirs
        assert "experimental/transformer" in rel_dirs

    def test_comma_separated_commit_ids(self):
        """Comma-separated commit IDs should union changed files from all commits."""
        # Simulate two commits: one touching attention, one touching qkv
        call_count = 0

        def mock_get_changed_files(commit_id):
            nonlocal call_count
            call_count += 1
            if commit_id == "commit_a":
                return ["src/nkilib_src/nkilib/core/attention/attention_cte.py"]
            elif commit_id == "commit_b":
                return ["src/nkilib_src/nkilib/core/qkv/qkv_cte.py"]
            return []

        with patch.object(self.finder, "_get_changed_files", side_effect=mock_get_changed_files):
            result = self.finder.get_relevant_test_dirs("commit_a,commit_b")

        assert call_count == 2
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/attention" in rel_dirs
        assert "core/qkv" in rel_dirs

    def test_git_range_syntax(self):
        """Git range 'commit_a..commit_b' should be passed directly to git diff."""
        with patch.object(
            self.finder,
            "_get_changed_files",
            return_value=["src/nkilib_src/nkilib/core/attention/attention_cte.py"],
        ) as mock:
            result = self.finder.get_relevant_test_dirs("abc123..def456")
        mock.assert_called_once_with("abc123..def456")
        assert result is not None

    def test_malformed_commit_ids_with_spaces_raises(self):
        """Space-separated commit IDs should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="contains spaces"):
            self.finder.get_relevant_test_dirs("commit_1 commit_2")

    def test_empty_commit_ids_raises(self):
        """Empty string should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Empty commit ID"):
            self.finder.get_relevant_test_dirs("")

    def test_mixed_range_and_comma_raises(self):
        """Mixing range and comma syntax should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Cannot mix"):
            self.finder.get_relevant_test_dirs("abc..def,ghi")


class TestResolveTestImport:
    """Tests resolution of test imports like 'test.integration.nkilib.core.x.y'."""

    def setup_method(self):
        self.finder = RelevantTestFinder(repo_root=REPO_ROOT)

    def test_resolves_existing_test_module(self):
        result = self.finder._resolve_test_import("test.integration.nkilib.core.moe_block.test_moe_block_tkg")
        assert result is not None
        assert result.name == "test_moe_block_tkg.py"

    def test_returns_none_for_nonexistent_test_module(self):
        result = self.finder._resolve_test_import("test.integration.nkilib.core.nonexistent.test_foo")
        assert result is None

    def test_returns_none_for_prefix_only(self):
        result = self.finder._resolve_test_import("test.integration.nkilib.")
        assert result is None


class TestTestToTestReverseIndex:
    """Tests that the reverse import index captures cross-test dependencies."""

    def setup_method(self):
        self.finder = RelevantTestFinder(repo_root=REPO_ROOT)
        self.finder._build_reverse_import_index()

    def test_experimental_moe_block_imports_core_moe_block(self):
        """test_mx_moe_block_tkg_wrapper.py imports generate_inputs from test_moe_block_tkg.py."""
        core_test = "test/integration/nkilib/core/moe_block/test_moe_block_tkg.py"
        deps = self.finder._reverse_deps.get(core_test, set())
        exp_files = [d for d in deps if "experimental/moe_block" in d]
        assert len(exp_files) > 0, f"experimental/moe_block should import from core test, got: {deps}"

    def test_test_index_includes_test_files(self):
        """The reverse index should contain test file paths as keys or values."""
        test_paths = [k for k in self.finder._reverse_deps if k.startswith("test/integration/")]
        assert len(test_paths) > 0, "Reverse index should include test file entries"


class TestTestToTestEndToEnd:
    """End-to-end tests verifying test-to-test dependency tracing."""

    def setup_method(self):
        self.finder = RelevantTestFinder(repo_root=REPO_ROOT)

    def _run_with_files(self, changed_files):
        with patch.object(self.finder, "_get_changed_files", return_value=changed_files):
            return self.finder.get_relevant_test_dirs("mocked")

    def test_core_moe_block_test_change_selects_experimental_moe_block(self):
        """Changing test_moe_block_tkg.py should also select experimental/moe_block.

        This is the exact scenario from commit b83cbfd2: generate_inputs() in the
        core test was modified, breaking test_mx_moe_block_tkg_wrapper.py which
        imports it.
        """
        result = self._run_with_files(["test/integration/nkilib/core/moe_block/test_moe_block_tkg.py"])
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/moe_block" in rel_dirs
        assert "experimental/moe_block" in rel_dirs

    def test_test_only_change_still_includes_own_dir(self):
        """A test-only change should always include its own test directory."""
        result = self._run_with_files(["test/integration/nkilib/core/moe/moe_cte/test_moe_bwmm_mx_cte.py"])
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/moe" in rel_dirs

    def test_core_attention_tkg_utils_change_selects_experimental_transformer(self):
        """Changing test_attention_tkg_utils.py should select experimental/transformer.

        test_transformer_tkg.py imports generate_cache_lens from test_attention_tkg_utils.
        """
        result = self._run_with_files(["test/integration/nkilib/core/attention/test_attention_tkg_utils.py"])
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/attention" in rel_dirs
        assert "experimental/transformer" in rel_dirs


class TestTestToTestPropagationStop:
    """Tests verifying that test-to-test propagation is stopped for source changes.

    When a source file changes, BFS should NOT propagate through test utility files
    to reach unrelated test suites. Only test files that directly or transitively
    import the changed source (via source-level imports) should be selected.
    """

    def setup_method(self):
        self.finder = RelevantTestFinder(repo_root=REPO_ROOT)

    def _run_with_files(self, changed_files):
        with patch.object(self.finder, "_get_changed_files", return_value=changed_files):
            return self.finder.get_relevant_test_dirs("mocked")

    def test_gate_up_projection_does_not_reach_moe_via_test_utils(self):
        """mlp_tkg_gate_up_projection.py should NOT select moe tests.

        Previously, the chain was: gate_up_projection → mlp_tkg → mlp →
        test_mlp_common.py → test_moe_tkg_utils.py → test_moe_tkg.py.
        With test-to-test propagation stopped, test_mlp_common.py is a dead end.
        """
        result = self._run_with_files(["src/nkilib_src/nkilib/core/mlp/mlp_tkg/mlp_tkg_gate_up_projection.py"])
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        assert "core/mlp" in rel_dirs
        assert "core/moe" not in rel_dirs
        assert "core/moe_block" not in rel_dirs

    def test_rmsnorm_mx_quantize_does_not_select_layernorm(self):
        """rmsnorm_mx_quantize_tkg.py should NOT select test_layernorm_tkg.py.

        Both live in core/subkernels/ but layernorm does not import rmsnorm_mx_quantize.
        File-level precision ensures only directly dependent tests are selected.
        """
        result = self._run_with_files(["src/nkilib_src/nkilib/core/subkernels/rmsnorm_mx_quantize_tkg.py"])
        assert result is not None
        # Should select rmsnorm_mx_quantize and moe_block (which imports it at source level)
        test_files = sorted(p.split("test/integration/nkilib/")[1] for p in result)
        assert any("test_rmsnorm_mx_quantize_tkg" in f for f in test_files)
        assert any("test_moe_block_tkg" in f for f in test_files)
        # Should NOT select unrelated subkernels tests
        assert not any("test_layernorm_tkg" in f for f in test_files)
        assert not any("test_find_nonzero_indices" in f for f in test_files)
        assert not any("test_rmsnorm_tkg" in f for f in test_files)

    def test_attention_tkg_does_not_select_attention_cte(self):
        """attention_tkg.py change should NOT select test_attention_cte.py.

        attention_cte tests import from attention_cte source, not attention_tkg.
        """
        result = self._run_with_files(["src/nkilib_src/nkilib/core/attention/attention_tkg.py"])
        assert result is not None
        test_files = sorted(p.split("test/integration/nkilib/")[1] for p in result)
        assert any("test_attention_tkg" in f for f in test_files)
        assert not any("test_attention_cte" in f for f in test_files)

    def test_directly_changed_test_still_propagates(self):
        """When a test file itself is changed, test-to-test propagation still works.

        Changing test_mlp_common.py should select tests that import from it,
        because the test file is directly in the changed set.
        """
        result = self._run_with_files(["test/integration/nkilib/core/mlp/test_mlp_common.py"])
        assert result is not None
        rel_dirs = _rel_test_dirs(result)
        # test_mlp_common is imported by test_mlp_cte and others
        assert "core/mlp" in rel_dirs
        # Should propagate to moe tests that import from test_mlp_common
        test_files = sorted(p.split("test/integration/nkilib/")[1] for p in result)
        assert any("test_moe_block_tkg" in f for f in test_files)
