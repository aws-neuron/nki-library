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
"""Unit tests for artifact upload / --force-local-cleanup interaction.

Regression coverage for the "neuron-explorer capture ... open inp-*.bin: no such
file or directory" failure: under fleet contention the host-assignment loop can
re-enter ``prepare_host`` (a host rotation) and upload the *same* once-dumped
artifact directory a second time. If ``--force-local-cleanup`` deleted the input
``*.bin`` files right after the first upload's tarball was built, the second
upload ships an archive that is missing the inputs, and ``neuron-explorer`` on the
second host fails to open them.

These tests exercise the mechanism deterministically (no hardware, no fleet
contention) by uploading the same directory twice with cleanup enabled and
asserting the inputs are still present in what gets shipped on the retry.
"""

import hashlib
import io
import os
import tarfile
import tempfile
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from test.utils.exceptions import RemoteFileTransferException, ValidationException
from test.utils.host_io import (
    RemoteParamikoIO,
    RemoteS3IO,
    ScpIO,
    _parse_sha256sum_output,
    _sha256_of_file,
)


class RecordingTarPipeIO(RemoteParamikoIO):
    """Tar-pipe IO whose network write records shipped tar members instead of sending them.

    ``upload()`` streams a gzipped tar of the local path into ``write()``; this
    override unpacks it and snapshots the member names — exactly the set of
    files shipped this attempt.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.shipped_snapshots: list[set[str]] = []

    def write(self, data: Iterator[bytes], path: str) -> None:
        buf = b"".join(data)
        with tarfile.open(fileobj=io.BytesIO(buf), mode="r:gz") as tf:
            self.shipped_snapshots.append({os.path.normpath(name) for name in tf.getnames()})


@pytest.fixture
def source_dir():
    """A dumped artifact directory: two large input .bin tensors + a scalar arg."""
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "inp-q-000.bin"), "wb") as f:
            f.write(b"\0" * 4096)
        with open(os.path.join(tmp, "inp-k_cache-000.bin"), "wb") as f:
            f.write(b"\0" * 4096)
        # Scalar args are pickled without a .bin extension; they are NOT cleaned up.
        with open(os.path.join(tmp, "inp-scale-000"), "wb") as f:
            f.write(b"x")
        # The compiled neff also lives in the dir and is not a .bin.
        with open(os.path.join(tmp, "file.neff"), "wb") as f:
            f.write(b"\0" * 1024)
        yield tmp


def _make_tar_pipe_io():
    """A tar-pipe HostDirectory whose network write snapshots what it ships."""
    remote = RecordingTarPipeIO(MagicMock(), "/tmp/neuronx-cc/tests/out-x_pid1", MagicMock())
    return remote, remote.shipped_snapshots


def test_upload_excluded_paths_are_not_shipped(source_dir):
    """excluded_paths (compiler artifacts/, metrics/) must not reach the host."""
    os.makedirs(os.path.join(source_dir, "artifacts", "nc00"))
    with open(os.path.join(source_dir, "artifacts", "bir.json"), "wb") as f:
        f.write(b"{}" * 2048)
    with open(os.path.join(source_dir, "artifacts", "nc00", "penguin.py"), "wb") as f:
        f.write(b"#" * 1024)
    os.makedirs(os.path.join(source_dir, "metrics"))
    with open(os.path.join(source_dir, "metrics", "test.json"), "wb") as f:
        f.write(b"{}")

    remote, shipped = _make_tar_pipe_io()
    remote.upload(source_dir, excluded_paths=["artifacts", "metrics"])

    assert len(shipped) == 1
    archive = shipped[0]
    excluded_shipped = {name for name in archive if name.split(os.sep)[0] in ("artifacts", "metrics")}
    assert not excluded_shipped, f"excluded paths leaked into the upload archive: {sorted(excluded_shipped)}"
    assert {"inp-q-000.bin", "inp-k_cache-000.bin", "file.neff"} <= archive, (
        f"needed files missing from archive: {sorted(archive)}"
    )


def test_retry_reupload_still_ships_input_bins(source_dir):
    """A rotation re-uploads the same dir; the retry's archive must keep the inputs.

    This is the exact failure path: attempt #1 uploads + (with cleanup) may drop
    the local .bin, then the host-assignment loop rotates and attempt #2 uploads
    the SAME dir. The second archive must still contain inp-*.bin, otherwise
    neuron-explorer on the second host cannot open them.
    """
    remote, shipped = _make_tar_pipe_io()

    # Attempt #1: upload with cleanup enabled (as the pipeline runs it).
    remote.upload(source_dir, force_local_cleanup=True)
    # Attempt #2: a host rotation re-uploads the same once-dumped directory.
    remote.upload(source_dir, force_local_cleanup=True)

    assert len(shipped) == 2, "expected two uploads (initial + rotation retry)"
    retry_archive = shipped[1]
    missing = {"inp-q-000.bin", "inp-k_cache-000.bin"} - retry_archive
    assert not missing, (
        f"retry upload shipped an archive missing input tensors {missing}; "
        f"neuron-explorer would fail with 'open <file>: no such file or directory'. "
        f"Archive contained: {sorted(retry_archive)}"
    )


class TestScpIO:
    """Native-ssh tar-pipe IO: transfer failures surface, no silent overwrites."""

    def _make_scp_io(self) -> ScpIO:
        return ScpIO("/tmp/fake_ssh_config", "fake-host", "/tmp/neuronx-cc/tests/out-x", MagicMock())

    def test_upload_failure_raises_remote_file_transfer_exception(self, source_dir) -> None:
        scp = self._make_scp_io()
        failed_proc = MagicMock()
        failed_proc.returncode = 255
        with patch("test.utils.host_io.subprocess.Popen", return_value=failed_proc):
            with pytest.raises(RemoteFileTransferException):
                scp.upload(source_dir)

    def test_download_refuses_to_overwrite_existing_destination(self, source_dir) -> None:
        scp = self._make_scp_io()
        with pytest.raises(AssertionError):
            scp.download(source_dir)  # destination already exists, no force_clean_destination


class TestTransferChecksumHelpers:
    """The sha256 helpers underpinning the transfer-integrity check."""

    def test_sha256_of_file_matches_hashlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blob")
            payload = os.urandom(200000)  # > _STREAM_BUFFER_SIZE to exercise chunked read
            with open(path, "wb") as f:
                f.write(payload)
            assert _sha256_of_file(path) == hashlib.sha256(payload).hexdigest()

    def test_parse_sha256sum_output_extracts_digest(self) -> None:
        digest = "a" * 64
        assert _parse_sha256sum_output(f"{digest}  /tmp/archive.tar.gz\n") == digest  # sha256sum: "<hex>  <path>"

    @pytest.mark.parametrize("bad", ["", "not-a-hash  file", "abc  file", "z" * 64 + "  file"])
    def test_parse_sha256sum_output_rejects_garbage(self, bad) -> None:
        assert _parse_sha256sum_output(bad) is None


class TestRemoteS3IODownloadIntegrity:
    """End-to-end sha256 verification on RemoteS3IO.download.

    We compute the archive's sha256 on the host, then compare after download. A
    mismatch (truncation/corruption in transit) re-downloads the same S3 object in
    place; a mismatch that persists across all attempts raises a RETRYABLE
    RemoteFileTransferException -- not a ValidationException, which --rerun-except
    excludes from the pipeline's auto-reruns.
    """

    def _make_collector(self):
        collector = MagicMock()
        timer_cm = MagicMock()
        timer_cm.__enter__.return_value = MagicMock(duration=1.0)
        collector.timer.return_value = timer_cm
        return collector

    def _make_s3_io(self, remote_stdout: str):
        """RemoteS3IO whose remote sha256sum returns ``remote_stdout``; other runs succeed."""
        client = MagicMock()

        def run(command, *args, **kwargs):
            stdout = remote_stdout if command.startswith("sha256sum") else ""
            return MagicMock(failed=False, stdout=stdout, stderr="")

        client.run.side_effect = run
        s3_config = MagicMock()
        s3_config.bucket = "bucket"
        s3_config.profile = None
        return RemoteS3IO(client, "/tmp/neuronx-cc/tests/out-x", s3_config, self._make_collector())

    def _run_download(self, remote_stdout: str, local_payload: bytes, dest: str):
        """Run download() where every S3->local attempt writes the same ``local_payload``."""
        return self._run_download_seq(remote_stdout, [local_payload], dest)

    def _run_download_seq(self, remote_stdout: str, payloads_per_attempt: list[bytes], dest: str):
        """Run download() writing one payload per S3->local attempt (last payload repeats if
        attempts exceed the list). Returns the mock S3 client so callers can assert call count."""
        io_obj = self._make_s3_io(remote_stdout)
        s3_client = MagicMock()
        attempts = {"n": 0}

        def fake_download_file(bucket, key, local_path):
            i = min(attempts["n"], len(payloads_per_attempt) - 1)
            attempts["n"] += 1
            with open(local_path, "wb") as f:
                f.write(payloads_per_attempt[i])

        s3_client.download_file.side_effect = fake_download_file
        with (
            patch("test.utils.host_io.get_s3_client_and_session", return_value=(s3_client, MagicMock())),
            patch("test.utils.host_io.build_remote_s3_cli_command", return_value="aws s3 cp ..."),
            patch("test.utils.host_io.generate_s3_key", return_value="out/archive.tar.gz"),
            patch("test.utils.host_io.subprocess.run", return_value=MagicMock(returncode=0)),
            patch("test.utils.host_io.__cleanup_local_paths__"),
            patch("test.utils.host_io.__cleanup_remote_paths__"),
        ):
            io_obj.download(dest, force_clean_destination=True)
        return s3_client

    @staticmethod
    def _sha_line(payload: bytes) -> str:
        return f"{hashlib.sha256(payload).hexdigest()}  archive.tar.gz\n"

    def test_mismatch_raises_retryable_not_validation(self) -> None:
        full = b"the-full-original-archive-bytes"
        truncated = full[: len(full) // 2]  # what actually lands after a truncated transfer
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RemoteFileTransferException) as excinfo:
                self._run_download(self._sha_line(full), truncated, os.path.join(tmp, "dl"))
            assert not isinstance(excinfo.value, ValidationException)
            assert "checksum mismatch" in str(excinfo.value).lower()

    def test_transient_truncation_recovers_on_redownload(self) -> None:
        """A first-attempt truncation is fixed by re-downloading the same object, no rerun."""
        full = b"the-full-original-archive-bytes"
        truncated = full[: len(full) // 2]
        with tempfile.TemporaryDirectory() as tmp:
            # attempt 1 lands truncated, attempt 2 lands whole -> download() succeeds
            s3_client = self._run_download_seq(self._sha_line(full), [truncated, full], os.path.join(tmp, "dl"))
            assert s3_client.download_file.call_count == 2  # re-downloaded once, no test-level rerun

    def test_persistent_truncation_exhausts_attempts_then_raises(self) -> None:
        """Every attempt truncated -> retryable error after exhausting the local attempts."""
        full = b"the-full-original-archive-bytes"
        truncated = full[: len(full) // 2]
        with tempfile.TemporaryDirectory() as tmp:
            with pytest.raises(RemoteFileTransferException) as excinfo:
                self._run_download_seq(self._sha_line(full), [truncated], os.path.join(tmp, "dl"))
            assert not isinstance(excinfo.value, ValidationException)
            assert "checksum mismatch" in str(excinfo.value).lower()

    def test_matching_checksum_passes(self) -> None:
        payload = b"the-full-original-archive-bytes"
        with tempfile.TemporaryDirectory() as tmp:
            self._run_download(self._sha_line(payload), payload, os.path.join(tmp, "dl"))  # no exception

    def test_unavailable_remote_checksum_skips_check(self) -> None:
        """If sha256sum is unavailable/unparseable, skip integrity (best-effort), don't fail."""
        with tempfile.TemporaryDirectory() as tmp:
            self._run_download("sha256sum: command not found\n", b"whatever", os.path.join(tmp, "dl"))
