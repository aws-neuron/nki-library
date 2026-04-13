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

"""Unit tests for FileClassifier."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "build-tools" / "bin"))

from file_classifier import FileClassifier


class TestFileClassifierDefaults:
    """Tests for default behavior when sync_ignore is missing."""

    def test_falls_back_to_default_when_sync_ignore_missing(self):
        classifier = FileClassifier(sync_ignore_path=Path("/nonexistent/sync_ignore"))
        assert classifier.is_mirror_excluded("src/private/secret.py")
        assert not classifier.is_mirror_excluded("src/public/api.py")

    def test_default_pattern_is_case_insensitive(self):
        classifier = FileClassifier(sync_ignore_path=Path("/nonexistent/sync_ignore"))
        assert classifier.is_mirror_excluded("PRIVATE/file.py")
        assert classifier.is_mirror_excluded("Private/file.py")


class TestFileClassifierSyncIgnore:
    """Tests for loading patterns from sync_ignore files."""

    def test_loads_patterns_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(".*secret.*\n.*config.*\n")
            f.flush()
            classifier = FileClassifier(sync_ignore_path=Path(f.name))

        assert classifier.is_mirror_excluded("my_secret.py")
        assert classifier.is_mirror_excluded("app_config.yaml")
        assert not classifier.is_mirror_excluded("public.py")
        Path(f.name).unlink()

    def test_ignores_comments_and_blank_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# comment\n\n.*secret.*\n# another comment\n.*private.*\n")
            f.flush()
            classifier = FileClassifier(sync_ignore_path=Path(f.name))

        assert classifier.is_mirror_excluded("secret.py")
        assert classifier.is_mirror_excluded("private.py")
        assert not classifier.is_mirror_excluded("# comment")
        Path(f.name).unlink()

    def test_uses_real_sync_ignore(self):
        """Verify the actual sync_ignore file loads without error."""
        classifier = FileClassifier()
        assert classifier.is_mirror_excluded("src/nkilib/private/kernel.py")
        assert classifier.is_mirror_excluded("aws_lambda/handler.py")
        assert not classifier.is_mirror_excluded("src/nkilib/core/attention.py")


class TestFileClassifierPackages:
    """Tests for is_private_package (dot-separated package names)."""

    def test_package_classification(self):
        classifier = FileClassifier()
        assert classifier.is_private_package("nkilib_src.nkilib.private")
        assert classifier.is_private_package("nkilib_src.nkilib.private.matmul")
        assert not classifier.is_private_package("nkilib_src.nkilib.core")
        assert not classifier.is_private_package("nkilib_src.nkilib.core.attention")
        assert not classifier.is_private_package("nkilib_src.nkilib.experimental")

    def test_package_converts_dots_to_slashes(self):
        """Verify package names are converted to paths before matching."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(".*secret.*\n")
            f.flush()
            classifier = FileClassifier(sync_ignore_path=Path(f.name))

        assert classifier.is_private_package("my.secret.module")
        assert not classifier.is_private_package("my.public.module")
        Path(f.name).unlink()
