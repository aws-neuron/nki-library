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
"""Unit tests for bir_neff_cache module."""

import json
import os
import tempfile
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from ..utils.bir_neff_cache import (
    CachedCompiledKernel,
    _s3_keys,
    compute_cache_key,
    lookup_cache,
    store_cache,
)
from ..utils.s3_utils import S3Uri


@patch("test.utils.bir_neff_cache.neuronxcc")
class TestS3Keys:
    def test_with_prefix(self, mock_neuronxcc):
        mock_neuronxcc.__version__ = "2.0.123"
        neff_key, meta_key = _s3_keys(S3Uri(bucket="my-bucket", prefix="my-prefix"), "abc123")
        assert neff_key == "my-prefix/2.0.123/abc123/file.neff"
        assert meta_key == "my-prefix/2.0.123/abc123/metadata.json"

    def test_without_prefix(self, mock_neuronxcc):
        mock_neuronxcc.__version__ = "2.0.123"
        neff_key, meta_key = _s3_keys(S3Uri(bucket="my-bucket", prefix=""), "abc123")
        assert neff_key == "2.0.123/abc123/file.neff"
        assert meta_key == "2.0.123/abc123/metadata.json"

    def test_version_in_path(self, mock_neuronxcc):
        mock_neuronxcc.__version__ = "2.0.253977.0a0+2ba785af"
        neff_key, _ = _s3_keys(S3Uri(bucket="my-bucket", prefix="cache"), "deadbeef")
        assert "2.0.253977.0a0+2ba785af" in neff_key


class TestComputeCacheKey:
    def _make_bir_result(self, tmpdir, kernel_json_content="{}"):
        kernel_json_path = os.path.join(tmpdir, "kernel.json")
        with open(kernel_json_path, "w") as f:
            f.write(kernel_json_content)

        @dataclass
        class MockDescriptor:
            kernel_json_path: str

        @dataclass
        class MockBirResult:
            descriptor: MockDescriptor

        return MockBirResult(descriptor=MockDescriptor(kernel_json_path=kernel_json_path))

    def _make_compile_opts(self):
        @dataclass
        class MockCompileOpts:
            target: str = "trn2"
            lnc: int = 2
            neuronx_cc_args: tuple = ()
            enable_device_dump: bool = False

        return MockCompileOpts()

    def test_returns_hex_digest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bir_result = self._make_bir_result(tmpdir)
            compile_opts = self._make_compile_opts()
            key = compute_cache_key(bir_result, compile_opts)
            assert key is not None
            assert len(key) == 64  # SHA-256 hex
            assert all(c in "0123456789abcdef" for c in key)

    def test_same_inputs_same_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bir_result = self._make_bir_result(tmpdir, '{"ops": [1, 2, 3]}')
            compile_opts = self._make_compile_opts()
            key1 = compute_cache_key(bir_result, compile_opts)
            key2 = compute_cache_key(bir_result, compile_opts)
            assert key1 == key2

    def test_different_bir_different_key(self):
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            bir1 = self._make_bir_result(tmpdir1, '{"ops": [1]}')
            bir2 = self._make_bir_result(tmpdir2, '{"ops": [2]}')
            opts = self._make_compile_opts()
            assert compute_cache_key(bir1, opts) != compute_cache_key(bir2, opts)

    def test_different_target_different_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bir_result = self._make_bir_result(tmpdir)
            opts1 = self._make_compile_opts()
            opts2 = self._make_compile_opts()
            opts2.target = "trn3"
            assert compute_cache_key(bir_result, opts1) != compute_cache_key(bir_result, opts2)

    def test_different_lnc_different_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bir_result = self._make_bir_result(tmpdir)
            opts1 = self._make_compile_opts()
            opts2 = self._make_compile_opts()
            opts2.lnc = 1
            assert compute_cache_key(bir_result, opts1) != compute_cache_key(bir_result, opts2)

    def test_different_cc_args_different_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bir_result = self._make_bir_result(tmpdir)
            opts1 = self._make_compile_opts()
            opts2 = self._make_compile_opts()
            opts2.neuronx_cc_args = ("--jobs=4",)
            assert compute_cache_key(bir_result, opts1) != compute_cache_key(bir_result, opts2)

    def test_npy_files_affect_key(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            bir1 = self._make_bir_result(tmpdir1)
            bir2 = self._make_bir_result(tmpdir2)
            opts = self._make_compile_opts()

            # Add a .npy constant to tmpdir2
            np.save(os.path.join(tmpdir2, "const.npy"), np.array([1.0, 2.0]))

            assert compute_cache_key(bir1, opts) != compute_cache_key(bir2, opts)

    def test_non_npy_siblings_ignored(self):
        # Sibling files that are not .npy constants (e.g. path-laden debug files)
        # must not affect the key, so it stays stable across build hosts.
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            bir1 = self._make_bir_result(tmpdir1, '{"ops": [1]}')
            bir2 = self._make_bir_result(tmpdir2, '{"ops": [1]}')
            opts = self._make_compile_opts()
            with open(os.path.join(tmpdir2, "debug_info.dbg"), "w") as f:
                f.write("/home/userB/ws/site-packages/nkilib/core/mlp.py")
            with open(os.path.join(tmpdir2, "module.mlir"), "w") as f:
                f.write("compiled at 2026-06-03T11:00:00")
            assert compute_cache_key(bir1, opts) == compute_cache_key(bir2, opts)

    def test_host_path_prefix_normalized(self):
        # Same BIR differing only in the host-specific path prefix before
        # /site-packages/ must produce the same key (cross-device stability).
        json_a = '{"loc": "/home/userA/ws/proj/env/lib/python3.10/site-packages/nkilib/core/mlp.py"}'
        json_b = '{"loc": "/local/home/userB/some/other/root/site-packages/nkilib/core/mlp.py"}'
        with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
            bir1 = self._make_bir_result(tmpdir1, json_a)
            bir2 = self._make_bir_result(tmpdir2, json_b)
            opts = self._make_compile_opts()
            assert compute_cache_key(bir1, opts) == compute_cache_key(bir2, opts)

    def test_missing_kernel_json_returns_none(self):
        @dataclass
        class MockDescriptor:
            kernel_json_path: str = "/nonexistent/path.json"

        @dataclass
        class MockBirResult:
            descriptor: MockDescriptor | None = None

            def __post_init__(self):
                self.descriptor = MockDescriptor()

        bir_result = MockBirResult()
        opts = self._make_compile_opts()
        assert compute_cache_key(bir_result, opts) is None


@patch("test.utils.bir_neff_cache.get_s3_client_and_session")
@patch("test.utils.bir_neff_cache.neuronxcc")
class TestLookupCache:
    def test_cache_hit(self, mock_neuronxcc, mock_get_s3):
        mock_neuronxcc.__version__ = "2.0.123"
        mock_s3 = MagicMock()
        mock_get_s3.return_value = (mock_s3, MagicMock())
        collector = MagicMock()

        mock_body = MagicMock()
        mock_body.read.return_value = json.dumps({"input_output_aliases": {0: "x"}}).encode()
        mock_s3.get_object.return_value = {"Body": mock_body}

        with tempfile.TemporaryDirectory() as tmpdir:
            local_neff = os.path.join(tmpdir, "file.neff")
            result = lookup_cache("s3://bucket/prefix", "abc123", local_neff, collector)

        assert result is not None
        assert isinstance(result, CachedCompiledKernel)
        assert result.neff_path == local_neff
        # JSON round-trips int keys to strings; lookup_cache must coerce them
        # back so the framework's `i in aliases` rename lookup (where i is int)
        # finds the entry. Otherwise the rename is silently skipped and the
        # validator fails with "<name> was not emitted by neuron-explorer
        # capture".
        assert result.input_output_aliases == {0: "x"}
        assert result.mlir_time == 0.0
        assert result.neuronx_cc_time == 0.0
        mock_s3.download_file.assert_called_once_with("bucket", "prefix/2.0.123/abc123/file.neff", local_neff)
        collector.record_timer.assert_called()
        collector.record_metric.assert_called_with("NeffCacheHit", 1, "Count")

    def test_cache_miss(self, mock_neuronxcc, mock_get_s3):
        mock_neuronxcc.__version__ = "2.0.123"
        mock_s3 = MagicMock()
        mock_get_s3.return_value = (mock_s3, MagicMock())
        collector = MagicMock()

        mock_s3.download_file.side_effect = ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "GetObject")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_neff = os.path.join(tmpdir, "file.neff")
            result = lookup_cache("s3://bucket/prefix", "abc123", local_neff, collector)

        assert result is None
        collector.record_metric.assert_called_with("NeffCacheHit", 0, "Count")

    def test_s3_error_returns_none(self, mock_neuronxcc, mock_get_s3):
        mock_neuronxcc.__version__ = "2.0.123"
        mock_s3 = MagicMock()
        mock_get_s3.return_value = (mock_s3, MagicMock())
        collector = MagicMock()

        mock_s3.download_file.side_effect = Exception("network timeout")

        with tempfile.TemporaryDirectory() as tmpdir:
            local_neff = os.path.join(tmpdir, "file.neff")
            result = lookup_cache("s3://bucket/prefix", "abc123", local_neff, collector)

        assert result is None
        collector.record_metric.assert_called_with("NeffCacheHit", 0, "Count")


@patch("test.utils.bir_neff_cache.get_s3_client_and_session")
@patch("test.utils.bir_neff_cache.neuronxcc")
class TestStoreCache:
    def test_store_success(self, mock_neuronxcc, mock_get_s3):
        mock_neuronxcc.__version__ = "2.0.123"
        mock_s3 = MagicMock()
        mock_get_s3.return_value = (mock_s3, MagicMock())
        collector = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            neff_path = os.path.join(tmpdir, "file.neff")
            with open(neff_path, "wb") as f:
                f.write(b"fake neff content")

            compiled = MagicMock()
            compiled.neff_path = neff_path
            compiled.input_output_aliases = {0: "x"}

            store_cache("s3://bucket/prefix", "abc123", compiled, collector)

        mock_s3.upload_file.assert_called_once_with(neff_path, "bucket", "prefix/2.0.123/abc123/file.neff")
        mock_s3.put_object.assert_called_once()
        put_kwargs = mock_s3.put_object.call_args[1]
        assert put_kwargs["Bucket"] == "bucket"
        assert put_kwargs["Key"] == "prefix/2.0.123/abc123/metadata.json"
        meta = json.loads(put_kwargs["Body"])
        assert meta["input_output_aliases"] == {"0": "x"}
        collector.record_timer.assert_called()

    def test_store_rejects_non_int_alias_keys(self, mock_neuronxcc, mock_get_s3):
        """Producer contract is dict[int, str]; non-int keys must raise."""
        mock_neuronxcc.__version__ = "2.0.123"
        mock_get_s3.return_value = (MagicMock(), MagicMock())
        collector = MagicMock()

        compiled = MagicMock()
        compiled.neff_path = "/tmp/unused.neff"
        compiled.input_output_aliases = {"0": "x"}

        with pytest.raises(AssertionError, match="key must be int"):
            store_cache("s3://bucket/prefix", "abc123", compiled, collector)

    def test_store_rejects_non_str_alias_values(self, mock_neuronxcc, mock_get_s3):
        """Producer contract is dict[int, str]; non-str values must raise."""
        mock_neuronxcc.__version__ = "2.0.123"
        mock_get_s3.return_value = (MagicMock(), MagicMock())
        collector = MagicMock()

        compiled = MagicMock()
        compiled.neff_path = "/tmp/unused.neff"
        compiled.input_output_aliases = {0: 42}

        with pytest.raises(AssertionError, match="value must be str"):
            store_cache("s3://bucket/prefix", "abc123", compiled, collector)

    def test_store_s3_error_does_not_raise(self, mock_neuronxcc, mock_get_s3):
        mock_neuronxcc.__version__ = "2.0.123"
        mock_s3 = MagicMock()
        mock_get_s3.return_value = (mock_s3, MagicMock())
        mock_s3.upload_file.side_effect = Exception("access denied")
        collector = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            neff_path = os.path.join(tmpdir, "file.neff")
            with open(neff_path, "wb") as f:
                f.write(b"fake neff")

            compiled = MagicMock()
            compiled.neff_path = neff_path
            compiled.input_output_aliases = {}

            # Should not raise
            store_cache("s3://bucket/prefix", "abc123", compiled, collector)

        collector.record_timer.assert_called()
