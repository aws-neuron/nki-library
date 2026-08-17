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
"""Host file I/O.

A ``HostDirectory`` uploads/downloads a directory between the local box and a
host, abstracting over the transfer mechanism:

- ``RemoteParamikoIO`` — tar-pipe over a paramiko channel.
- ``ScpIO``            — tar-pipe over native ssh (direct subprocess pipes).
- ``RemoteS3IO``       — via an S3 intermediary (boto3 + remote ``aws s3 cp``).
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterator

from .exceptions import (
    LocalExecutionException,
    RemoteExecutionException,
    RemoteFileTransferException,
)
from .metrics_collector import IMetricsCollector, MetricName
from .s3_utils import (
    S3ArtifactUploadConfig,
    S3TransferDirection,
    build_remote_s3_cli_command,
    generate_s3_key,
    get_bucket_region,
    get_s3_client_and_session,
)

if TYPE_CHECKING:
    from .host_communication import HostCommunication, SshChannel

_STREAM_BUFFER_SIZE = 65536

# Budget for remote transfer commands (aws s3 cp / tar of multi-GB artifact
# archives), which can far exceed the transport's default workload budget.
# Bounded (unlike the pre-refactor code) so a wedged transfer still aborts.
_S3_TRANSFER_TIMEOUT_SECONDS = 900

# Re-attempts for the S3->local download within a single download() call, so a
# transient truncation re-transfers the archive rather than failing the test.
_S3_DOWNLOAD_MAX_ATTEMPTS = 3


def cleanup_input_bins(local_path: str) -> None:
    """Delete input ``*.bin`` tensors from a local artifact directory to free disk.

    Call this ONLY after inference has fully succeeded. Deleting inputs while a
    host rotation could still re-upload the directory makes the retry ship an
    incomplete archive (missing ``inp-*.bin``), which neuron-explorer reports as
    "open inp-*.bin: no such file or directory".
    """
    import pathlib

    for bin_file in pathlib.Path(local_path).glob("*.bin"):
        bin_file.unlink(missing_ok=True)


def _sha256_of_file(path: str) -> str:
    """Hex SHA-256 of a local file, read in bounded chunks (~1.2 GiB/s, CPU-bound)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_STREAM_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sha256sum_output(stdout: str) -> str | None:
    """Extract the 64-char hex digest from ``sha256sum`` stdout (``<hex>  <path>``), else None."""
    token = stdout.split(maxsplit=1)[0] if stdout.strip() else ""
    return token if len(token) == 64 and all(c in "0123456789abcdef" for c in token) else None


def create_archive_command(
    source_path: str,
    destination_path: str,
    files_to_include: list[str] | None = None,
    excluded_paths: list[str] | None = None,
) -> str:
    if files_to_include is None:
        files_to_include = ["."]

    if not destination_path.__contains__(".tar"):
        destination_path += ".tar"

    files_inside_archive = " ".join(file_name for file_name in files_to_include)
    exclude_args = "".join(f"--exclude={pattern} " for pattern in (excluded_paths or []))

    return f"tar --ignore-failed-read {exclude_args}-cf {destination_path} -C {source_path} {files_inside_archive}"


def defalate_archive_command(source_path: str, destination_path: str) -> str:
    return f"tar -xf {source_path} -C {destination_path}"


def parent(path: str) -> str:
    return os.path.dirname(path)


def to_archive_name_in_parent(path: str) -> str:
    return os.path.join(parent(path), os.path.basename(path)) + ".tar"


def is_archive(path: str) -> bool:
    return path.__contains__(".tar")


def __cleanup_remote_paths__(
    client: "HostCommunication",
    *remote_path_list: str,
    base_exception: Exception | None = None,
    ignore_exceptions: bool = False,
) -> None:
    exceptions: list[Exception] = [base_exception] if base_exception else []
    for remote_path in remote_path_list:
        try:
            client.run(f"rm -rf {remote_path}")
        except Exception as e:
            exceptions.append(e)
    if len(exceptions) > 0 and not ignore_exceptions:
        raise Exception(*exceptions)


def __cleanup_local_paths__(
    *local_path_list: str,
    base_exception: Exception | None = None,
    ignore_exceptions: bool = False,
) -> None:
    exceptions: list[Exception] = [base_exception] if base_exception else []
    for local_path in local_path_list:
        try:
            if os.path.isfile(local_path):
                os.remove(local_path)
            elif os.path.isdir(local_path):
                shutil.rmtree(local_path)
        except Exception as e:
            exceptions.append(e)
    if len(exceptions) > 0 and not ignore_exceptions:
        raise Exception(*exceptions)


class HostDirectory(ABC):
    """A directory on a host. ``upload``/``download`` move a tree to/from the
    local box; ``cleanup`` removes the remote tree."""

    def __init__(self, path: str, collector: IMetricsCollector) -> None:
        self.target_path = path
        self._collector = collector
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def upload(
        self,
        local_path: str,
        force_local_cleanup: bool = False,
        excluded_paths: list[str] | None = None,
    ) -> None:
        """``excluded_paths`` are tar glob patterns; matching artifacts are not uploaded."""
        raise NotImplementedError

    @abstractmethod
    def download(
        self,
        destination_dir_path: str,
        list_of_files: list[str] | None = None,
        force_clean_destination: bool = False,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        raise NotImplementedError


class RemoteParamikoIO(HostDirectory):
    """Tar-pipe transfer over a paramiko channel."""

    def __init__(self, client: "HostCommunication", path: str, collector: IMetricsCollector) -> None:
        super().__init__(path, collector)
        self._client = client

    def _open_exec_channel(self, command: str) -> "SshChannel":
        from .host_communication import SshChannelOpenError  # noqa: PLC0415

        try:
            return self._client.open_channel(command)
        except SshChannelOpenError as e:
            raise RemoteFileTransferException(f"Failed to open remote channel: {command}", e) from e

    def write(self, data: Iterator[bytes], path: str) -> None:
        """Stream ``data`` (tar bytes) into a remote directory rooted at ``path``."""
        channel = self._open_exec_channel(f"mkdir -p -m 777 {path} && tar -xzf - -C {path}")
        try:
            for chunk in data:
                channel.sendall(chunk)
            channel.shutdown_write()
        finally:
            exit_status = channel.recv_exit_status()
            channel.close()
        if exit_status != 0:
            raise RemoteFileTransferException(
                f"Remote command failed with exit code {exit_status}", Exception(f"exit code {exit_status}")
            )

    def read(self, path: str, files: list[str] | None = None) -> Iterator[bytes]:
        """Stream a compressed tar of ``path`` (optionally only ``files``)."""
        files_arg = " ".join(files) if files else "."
        channel = self._open_exec_channel(f"tar --ignore-failed-read -czf - -C {path} {files_arg}")
        try:
            while True:
                data = channel.recv(_STREAM_BUFFER_SIZE)
                if not data:
                    break
                yield data
        finally:
            exit_status = channel.recv_exit_status()
            channel.close()
            if exit_status != 0:
                raise RemoteFileTransferException(
                    f"Remote command failed with exit code {exit_status}", Exception(f"exit code {exit_status}")
                )

    def upload(
        self,
        local_path: str,
        force_local_cleanup: bool = False,
        excluded_paths: list[str] | None = None,
    ) -> None:
        # Record uncompressed upload bytes
        uncompressed_bytes = sum(
            os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(local_path) for f in files
        )
        self._collector.record_metric(MetricName.FILE_TRANSFER_BYTES_UNCOMPRESSED, uncompressed_bytes, "Bytes")
        assert os.path.exists(local_path)

        exclude_args = [f"--exclude={pattern}" for pattern in (excluded_paths or [])]
        with self._collector.timer(MetricName.SFTP_UPLOAD_TIME):
            local_tar = subprocess.Popen(
                ["tar", *exclude_args, "-czf", "-", "-C", local_path, "."], stdout=subprocess.PIPE
            )
            tar_stdout = local_tar.stdout
            assert tar_stdout is not None, "stdout=PIPE was requested"
            try:
                self.write(iter(lambda: tar_stdout.read(_STREAM_BUFFER_SIZE), b""), self.target_path)
            finally:
                tar_stdout.close()
                local_tar.wait()

        # NOTE: input .bin files are intentionally NOT deleted here. A host
        # rotation re-enters prepare_host and re-uploads this same directory; if
        # the inputs were deleted after the first upload, the retry would ship an
        # archive missing them and neuron-explorer on the new host would fail with
        # "open inp-*.bin: no such file or directory". Local cleanup happens once,
        # after inference succeeds (see Orchestrator._run_inference).

    def download(
        self,
        destination_dir_path: str,
        list_of_files: list[str] | None = None,
        force_clean_destination: bool = False,
    ) -> str:
        if not force_clean_destination:
            assert not os.path.exists(destination_dir_path), (
                f"{destination_dir_path} already exists locally. Can't download artifacts into it for the fear of overwriting content"
            )
        else:
            shutil.rmtree(os.path.join(destination_dir_path, "*"), ignore_errors=True)
        os.makedirs(destination_dir_path, exist_ok=True)

        with self._collector.timer(MetricName.SFTP_DOWNLOAD_TIME):
            local_tar = subprocess.Popen(["tar", "-xzf", "-", "-C", destination_dir_path], stdin=subprocess.PIPE)
            tar_stdin = local_tar.stdin
            assert tar_stdin is not None, "stdin=PIPE was requested"
            try:
                for chunk in self.read(self.target_path, list_of_files):
                    tar_stdin.write(chunk)
            finally:
                tar_stdin.close()
                local_tar.wait()
            if local_tar.returncode != 0:
                raise LocalExecutionException(
                    "Local tar extract failed",
                    subprocess.CompletedProcess(local_tar.args, local_tar.returncode),
                )

        uncompressed_bytes = sum(
            os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(destination_dir_path) for f in files
        )
        self._collector.record_metric(MetricName.FILE_TRANSFER_BYTES_UNCOMPRESSED, uncompressed_bytes, "Bytes")
        return destination_dir_path

    def cleanup(self) -> None:
        result = self._client.run(f"rm -rf {self.target_path}")
        if result.failed:
            raise RemoteExecutionException(f"Unable to cleanup remote folder {self.target_path}", result)


class ScpIO(HostDirectory):
    """Tar-pipe file I/O over native ssh subprocess pipes."""

    _KEEPALIVE_OPTS = ("-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2")

    def __init__(self, ssh_config_path: str, alias: str, path: str, collector: IMetricsCollector) -> None:
        super().__init__(path, collector)
        self._cfg = ssh_config_path
        self._alias = alias
        # Per-process ControlPath mirrors SshCommunication: each xdist worker
        # multiplexes over its own SSH connection for parallel uploads.
        self._control_path = f"/tmp/nkilib-cm-{os.getpid()}-%C"

    def _ssh_cmd(self, remote_command: str) -> list[str]:
        return [
            "ssh",
            "-F",
            self._cfg,
            *self._KEEPALIVE_OPTS,
            "-o",
            f"ControlPath={self._control_path}",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=30",
            "-T",
            self._alias,
            remote_command,
        ]

    def upload(
        self,
        local_path: str,
        force_local_cleanup: bool = False,
        excluded_paths: list[str] | None = None,
    ) -> None:
        uncompressed_bytes = sum(
            os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(local_path) for f in files
        )
        self._collector.record_metric(MetricName.FILE_TRANSFER_BYTES_UNCOMPRESSED, uncompressed_bytes, "Bytes")
        assert os.path.exists(local_path)

        # Uncompressed tar: the payload is dominated by float tensors (~1.1x
        # gzip ratio) and the wire outpaces gzip, so compression only adds CPU
        # to the critical path.
        exclude_args = [f"--exclude={pattern}" for pattern in (excluded_paths or [])]
        with self._collector.timer(MetricName.SFTP_UPLOAD_TIME):
            tar_proc = subprocess.Popen(
                ["tar", *exclude_args, "-cf", "-", "-C", local_path, "."], stdout=subprocess.PIPE
            )
            tar_stdout = tar_proc.stdout
            assert tar_stdout is not None, "stdout=PIPE was requested"
            ssh_proc = subprocess.Popen(
                self._ssh_cmd(f"mkdir -p -m 777 {self.target_path} && tar -xf - -C {self.target_path}"),
                stdin=tar_stdout,
            )
            tar_stdout.close()
            ssh_proc.wait()
            tar_proc.wait()
        if ssh_proc.returncode != 0:
            raise RemoteFileTransferException(
                f"Upload to {self.target_path} failed (ssh exit {ssh_proc.returncode})",
                Exception(f"exit code {ssh_proc.returncode}"),
            )

    def download(
        self,
        destination_dir_path: str,
        list_of_files: list[str] | None = None,
        force_clean_destination: bool = False,
    ) -> str:
        if not force_clean_destination:
            assert not os.path.exists(destination_dir_path), (
                f"{destination_dir_path} already exists locally. Can't download artifacts into it for the fear of overwriting content"
            )
        else:
            shutil.rmtree(os.path.join(destination_dir_path, "*"), ignore_errors=True)
        os.makedirs(destination_dir_path, exist_ok=True)

        files_arg = " ".join(list_of_files) if list_of_files else "."
        with self._collector.timer(MetricName.SFTP_DOWNLOAD_TIME):
            ssh_proc = subprocess.Popen(
                self._ssh_cmd(f"tar --ignore-failed-read -cf - -C {self.target_path} {files_arg}"),
                stdout=subprocess.PIPE,
            )
            ssh_stdout = ssh_proc.stdout
            assert ssh_stdout is not None, "stdout=PIPE was requested"
            tar_proc = subprocess.Popen(["tar", "-xf", "-", "-C", destination_dir_path], stdin=ssh_stdout)
            ssh_stdout.close()
            tar_proc.wait()
            ssh_proc.wait()
        if tar_proc.returncode != 0:
            raise RemoteFileTransferException(
                f"Local tar extract failed (exit {tar_proc.returncode})",
                Exception(f"exit code {tar_proc.returncode}"),
            )
        if ssh_proc.returncode != 0:
            raise RemoteFileTransferException(
                f"Download from {self.target_path} failed (ssh exit {ssh_proc.returncode})",
                Exception(f"exit code {ssh_proc.returncode}"),
            )

        uncompressed_bytes = sum(
            os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(destination_dir_path) for f in files
        )
        self._collector.record_metric(MetricName.FILE_TRANSFER_BYTES_UNCOMPRESSED, uncompressed_bytes, "Bytes")
        return destination_dir_path

    def cleanup(self) -> None:
        result = subprocess.run(self._ssh_cmd(f"rm -rf {self.target_path}"), capture_output=True, text=True)
        if result.returncode != 0:
            raise RemoteExecutionException(
                f"Unable to cleanup remote folder {self.target_path}",
                result,
            )


class RemoteS3IO(HostDirectory):
    """Transfer via an S3 intermediary: boto3 locally, ``aws s3 cp`` on the host."""

    def __init__(
        self, client: "HostCommunication", path: str, s3_config: S3ArtifactUploadConfig, collector: IMetricsCollector
    ) -> None:
        super().__init__(path, collector)
        self._client = client
        self._s3_config = s3_config
        if s3_config.bucket is None:
            raise ValueError("S3-based artifact transfer requires a bucket in the S3 configuration")
        self._bucket = s3_config.bucket
        self._resolved_region: str | None = None

    def _bucket_region(self) -> str:
        """Region to sign remote ``aws s3 cp`` for: the bucket's region, never the
        host's. Uses the explicit config value if set, else resolves once from the
        bucket. Memoized per instance (and process-wide via ``get_bucket_region``).
        """
        if self._resolved_region is None:
            self._resolved_region = self._s3_config.region or get_bucket_region(self._bucket, self._s3_config.profile)
        return self._resolved_region

    def upload(
        self,
        local_path: str,
        force_local_cleanup: bool = False,
        excluded_paths: list[str] | None = None,
    ) -> None:
        s3_config = self._s3_config
        s3_client, session = get_s3_client_and_session(s3_config.profile)
        assert os.path.exists(local_path), f"Local path {local_path} does not exist"

        s3_key = generate_s3_key(s3_config, S3TransferDirection.INPUTS)
        archive_name = os.path.basename(s3_key)

        local_archive_location = None
        if not is_archive(local_path):
            local_archive_location = to_archive_name_in_parent(local_path)
            with self._collector.timer(MetricName.S3_UPLOAD_LOCAL_ARCHIVE_TIME):
                command_result = subprocess.run(
                    create_archive_command(local_path, local_archive_location, excluded_paths=excluded_paths).split(" ")
                )
            if command_result.returncode != 0:
                raise LocalExecutionException(f"Unable to create tarball of {local_path}", command_result)
            local_path = local_archive_location

            # NOTE: input .bin files are intentionally NOT deleted here. A host
            # rotation re-enters prepare_host and re-uploads original_source_path;
            # deleting the inputs after the first upload would make the retry ship
            # an archive missing them, and neuron-explorer on the new host would
            # fail with "open inp-*.bin: no such file or directory". Local cleanup
            # happens once, after inference succeeds (see Orchestrator._run_inference).

        try:
            self.logger.info(f"Uploading {local_path} to s3://{s3_config.bucket}/{s3_key}...")
            upload_bytes = os.path.getsize(local_path)
            self._collector.record_metric(MetricName.S3_UPLOAD_BYTES, upload_bytes, "Bytes")
            with self._collector.timer(MetricName.S3_UPLOAD_LOCAL_TO_S3_TIME) as t:
                s3_client.upload_file(local_path, s3_config.bucket, s3_key)
            rate = (upload_bytes / 1024) / t.duration if t.duration > 0 else 1e9
            self._collector.record_metric(MetricName.S3_UPLOAD_LOCAL_TO_S3_RATE, rate, "Kilobytes/Second")
            self.logger.info(f"Successfully uploaded to s3://{s3_config.bucket}/{s3_key}")
        finally:
            if local_archive_location and os.path.exists(local_archive_location):
                __cleanup_local_paths__(local_archive_location, ignore_exceptions=True)

        remote_archive_location = os.path.join(self.target_path, archive_name)
        deflate_exception = None
        try:
            self._client.run(f"mkdir -p -m 777 {self.target_path}")
            creds = session.get_credentials().get_frozen_credentials()
            self.logger.info(f"Downloading s3://{s3_config.bucket}/{s3_key} on remote device...")
            download_cmd = build_remote_s3_cli_command(
                self._bucket, s3_key, remote_archive_location, creds, S3TransferDirection.INPUTS, self._bucket_region()
            )
            with self._collector.timer(MetricName.S3_UPLOAD_S3_TO_REMOTE_TIME) as t:
                result = self._client.run(download_cmd, hide=True, timeout=_S3_TRANSFER_TIMEOUT_SECONDS)
            if result.failed:
                raise RemoteExecutionException(f"Failed to download from S3 on remote: {result.stderr}", result)
            rate = (upload_bytes / 1024) / t.duration if t.duration > 0 else 1e9
            self._collector.record_metric(MetricName.S3_UPLOAD_S3_TO_REMOTE_RATE, rate, "Kilobytes/Second")
            self.logger.info(f"Successfully downloaded to remote: {remote_archive_location}")

            with self._collector.timer(MetricName.S3_UPLOAD_REMOTE_EXTRACT_TIME):
                result = self._client.run(
                    defalate_archive_command(remote_archive_location, self.target_path),
                    timeout=_S3_TRANSFER_TIMEOUT_SECONDS,
                )
            if result.failed:
                deflate_exception = RemoteExecutionException(
                    f"Unable to deflate archive in {self.target_path} on remote", result
                )
        finally:
            __cleanup_remote_paths__(
                self._client, remote_archive_location, base_exception=deflate_exception, ignore_exceptions=True
            )

    def download(
        self,
        destination_dir_path: str,
        list_of_files: list[str] | None = None,
        force_clean_destination: bool = False,
    ) -> str:
        if not force_clean_destination:
            assert not os.path.exists(destination_dir_path), (
                f"{destination_dir_path} already exists locally. Can't download artifacts into it for the fear of overwriting content"
            )
        else:
            shutil.rmtree(os.path.join(destination_dir_path, "*"), ignore_errors=True)

        s3_config = self._s3_config
        s3_client, session = get_s3_client_and_session(s3_config.profile)
        creds = session.get_credentials().get_frozen_credentials()
        s3_key = generate_s3_key(s3_config, S3TransferDirection.OUTPUTS)
        remote_archive_location = to_archive_name_in_parent(self.target_path)

        try:
            with self._collector.timer(MetricName.S3_DOWNLOAD_REMOTE_ARCHIVE_TIME):
                result = self._client.run(
                    create_archive_command(self.target_path, remote_archive_location, list_of_files),
                    timeout=_S3_TRANSFER_TIMEOUT_SECONDS,
                )
            if result.failed:
                raise RemoteExecutionException(
                    f"Unable to create tarball in {remote_archive_location} on remote", result
                )

            # Fingerprint the archive on the host before it leaves. boto3 does not
            # validate the object checksum on a ranged (multipart) download, so we
            # compare the digest ourselves. Best-effort: skip if sha256sum is unavailable.
            checksum_result = self._client.run(
                f"sha256sum '{remote_archive_location}'",
                hide=True,
                warn=True,
                timeout=_S3_TRANSFER_TIMEOUT_SECONDS,
            )
            remote_sha256 = None if checksum_result.failed else _parse_sha256sum_output(checksum_result.stdout)
            if remote_sha256 is None:
                self.logger.warning("Could not compute remote archive sha256; skipping transfer-integrity check")

            self.logger.info(f"Uploading from remote to s3://{s3_config.bucket}/{s3_key}...")
            upload_cmd = build_remote_s3_cli_command(
                self._bucket, s3_key, remote_archive_location, creds, S3TransferDirection.OUTPUTS, self._bucket_region()
            )
            with self._collector.timer(MetricName.S3_DOWNLOAD_REMOTE_TO_S3_TIME) as t_remote_to_s3:
                result = self._client.run(upload_cmd, hide=True, timeout=_S3_TRANSFER_TIMEOUT_SECONDS)
            if result.failed:
                raise RemoteExecutionException(f"Failed to upload to S3 from remote: {result.stderr}", result)
            self.logger.info(f"Successfully uploaded to s3://{s3_config.bucket}/{s3_key}")

            archive_name = os.path.basename(s3_key)
            local_archive_location = os.path.join(destination_dir_path, archive_name)
            os.makedirs(destination_dir_path, exist_ok=True)

            # Retry the S3->local download in place. The uploaded object is whole
            # (multipart completion is atomic), so re-downloading the same key fixes a
            # truncated GET without rebuilding the tarball, re-uploading, or re-running
            # the test.
            download_bytes = 0
            for attempt in range(_S3_DOWNLOAD_MAX_ATTEMPTS):
                self.logger.info(f"Downloading s3://{s3_config.bucket}/{s3_key} to local...")
                with self._collector.timer(MetricName.S3_DOWNLOAD_S3_TO_LOCAL_TIME) as t_s3_to_local:
                    s3_client.download_file(s3_config.bucket, s3_key, local_archive_location)
                download_bytes = os.path.getsize(local_archive_location)

                # Transfer integrity only, blind to whether the kernel produced the right
                # shape. A wrong-size kernel still ships a whole, matching archive.
                if remote_sha256 is None or _sha256_of_file(local_archive_location) == remote_sha256:
                    self.logger.info(f"Successfully downloaded to: {local_archive_location}")
                    break

                # A mismatch that persists across all attempts raises a retryable error, not
                # a ValidationException (which --rerun-except excludes from the auto-reruns).
                __cleanup_local_paths__(local_archive_location, ignore_exceptions=True)
                if attempt + 1 >= _S3_DOWNLOAD_MAX_ATTEMPTS:
                    raise RemoteFileTransferException(
                        f"Archive checksum mismatch after {_S3_DOWNLOAD_MAX_ATTEMPTS} download attempts for "
                        f"s3://{s3_config.bucket}/{s3_key} (remote={remote_sha256}, local_size={download_bytes}) "
                        f"— archive was truncated or corrupted in transit",
                        Exception("sha256 mismatch"),
                    )
                self.logger.warning(
                    f"Archive checksum mismatch on download attempt {attempt + 1}/{_S3_DOWNLOAD_MAX_ATTEMPTS} "
                    f"for s3://{s3_config.bucket}/{s3_key}; re-downloading"
                )

            self._collector.record_metric(MetricName.S3_DOWNLOAD_BYTES, download_bytes, "Bytes")

            rate_remote_to_s3 = (
                (download_bytes / 1024) / t_remote_to_s3.duration if t_remote_to_s3.duration > 0 else 1e9
            )
            self._collector.record_metric(
                MetricName.S3_DOWNLOAD_REMOTE_TO_S3_RATE, rate_remote_to_s3, "Kilobytes/Second"
            )
            rate_s3_to_local = (download_bytes / 1024) / t_s3_to_local.duration if t_s3_to_local.duration > 0 else 1e9
            self._collector.record_metric(MetricName.S3_DOWNLOAD_S3_TO_LOCAL_RATE, rate_s3_to_local, "Kilobytes/Second")

            with self._collector.timer(MetricName.S3_DOWNLOAD_LOCAL_EXTRACT_TIME):
                command_result = subprocess.run(
                    defalate_archive_command(local_archive_location, destination_dir_path).split(" ")
                )
            if command_result.returncode != 0:
                raise LocalExecutionException(f"Unable to unarchive {local_archive_location}", command_result)

            __cleanup_local_paths__(local_archive_location, ignore_exceptions=True)
        finally:
            __cleanup_remote_paths__(self._client, remote_archive_location, ignore_exceptions=True)

        return destination_dir_path

    def cleanup(self) -> None:
        result = self._client.run(f"rm -rf {self.target_path}")
        if result.failed:
            raise RemoteExecutionException(f"Unable to cleanup remote folder {self.target_path}", result)
