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
"""Unit tests for pytest_test_metadata helpers."""

from unittest.mock import patch

from ..utils.pytest_test_metadata import (
    TestMetadataInfo as MetadataInfo,  # alias avoids pytest collecting the dataclass as a test
)
from ..utils.pytest_test_metadata import (
    derive_labeled_kernel_name,
    resolve_file_kernel_name,
)


def test_core_kernel_returns_bare_name():
    """Core kernels keep the unqualified metadata name (matches pipeline)."""
    path = "/repo/test/integration/nkilib/core/mlp/test_mlp_tkg.py"
    assert derive_labeled_kernel_name(path, "MLP TKG") == "MLP TKG"


def test_experimental_kernel_is_prefixed():
    """Non-core families get the folder capitalized and prepended."""
    path = "/repo/test/integration/nkilib/experimental/moe/moe_tkg/test_x.py"
    assert derive_labeled_kernel_name(path, "MoE TKG") == "Experimental MoE TKG"


def test_missing_metadata_returns_none():
    path = "/repo/test/integration/nkilib/core/mlp/test_mlp_tkg.py"
    assert derive_labeled_kernel_name(path, None) is None


def test_path_outside_test_package_falls_back_to_bare_name():
    """Paths that don't live under integration/nkilib skip the prefix rather than guessing."""
    path = "/some/other/tree/test_mlp.py"
    assert derive_labeled_kernel_name(path, "MLP TKG") == "MLP TKG"


def test_integration_without_nkilib_falls_back_to_bare_name():
    """The marker is specifically integration/nkilib — a lone 'integration' is not enough."""
    path = "/repo/test/integration/other/test_x.py"
    assert derive_labeled_kernel_name(path, "X") == "X"


def test_file_directly_under_nkilib_falls_back_to_bare_name():
    """CDK treats this case (pathParts length 1) as having no family prefix; mirror that."""
    path = "/repo/test/integration/nkilib/test_x.py"
    assert derive_labeled_kernel_name(path, "X") == "X"


# ── resolve_file_kernel_name: file-scoped resolution ──
# Metadata is defined once per file; the KernelName must resolve from the file,
# not the running class, so a second-or-later class in a multi-class file still
# gets named. The file's classes are stubbed so these stay pure-logic unit tests
# (no filesystem, no real kernel names).


def _classes(path, *metadata_names):
    """Build the extractor's return for a file: one TestMetadataInfo per class.

    A None entry models a class without @pytest_test_metadata.
    """
    return [
        MetadataInfo(
            file_path=path,
            class_name=f"TestClass{i}",
            line_number=i,
            metadata={"name": name} if name is not None else None,
        )
        for i, name in enumerate(metadata_names)
    ]


@patch("test.utils.pytest_test_metadata.extract_pytest_test_metadata_from_file")
def test_file_scoped_name_resolves_for_all_classes(mock_extract):
    """The file's single @pytest_test_metadata names the file regardless of running class.

    The multi-class file case that previously dropped KernelName for every class
    after the first: the annotated class is first, a metadata-less class second.
    """
    path = "/repo/test/integration/nkilib/core/kernel/test_x.py"
    mock_extract.return_value = _classes(path, "Widget", None)
    resolve_file_kernel_name.cache_clear()
    assert resolve_file_kernel_name(path) == "Widget"


@patch("test.utils.pytest_test_metadata.extract_pytest_test_metadata_from_file")
def test_file_without_metadata_returns_none(mock_extract):
    """A file whose classes carry no @pytest_test_metadata resolves to None."""
    path = "/repo/test/integration/nkilib/core/kernel/test_x.py"
    mock_extract.return_value = _classes(path, None)
    resolve_file_kernel_name.cache_clear()
    assert resolve_file_kernel_name(path) is None


def test_directory_path_resolves_to_none(tmp_path):
    """A directory path degrades to None instead of raising IsADirectoryError.

    Exercises the real extractor (not the mock): opening a directory raises
    IsADirectoryError, which must be swallowed so a bad path never aborts the
    metadata lookup.
    """
    resolve_file_kernel_name.cache_clear()
    assert resolve_file_kernel_name(tmp_path) is None
