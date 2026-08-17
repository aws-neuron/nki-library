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
"""Shared fixtures for unit tests."""

from pathlib import Path

import pytest

from test.utils.relevant_test_selection.finder import RelevantTestFinder

_REPO_ROOT = Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def relevant_test_finder_index() -> dict[str, set[str]]:
    """Build the reverse-import index once per session and return the raw dict.

    Consuming tests deep-copy it into a fresh RelevantTestFinder to avoid
    cross-test mutation while skipping the ~1.3s-per-test AST rebuild.
    """
    finder = RelevantTestFinder(repo_root=_REPO_ROOT)
    return finder._build_reverse_import_index()
