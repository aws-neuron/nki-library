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
import logging
import os
import pathlib
import shutil
import subprocess
from abc import ABC

import fabric2

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
    get_s3_client_and_session,
)


def create_archive_command(source_path: str, destination_path: str, files_to_include: list[str] | None = None) -> str:
    if files_to_include is None:
        files_to_include = ["."]

    if not destination_path.__contains__(".tar"):
        destination_path += ".tar"

    files_inside_archive = " ".join(file_name for file_name in files_to_include)

    return f"tar --ignore-failed-read -cf {destination_path} -C {source_path} {files_inside_archive}"


def defalate_archive_command(source_path: str, destination_path: str):
    return f"tar -xf {source_path} -C {destination_path}"


def parent(path: str):
    return os.path.dirname(path)


def to_archive_name_in_parent(path: str):
    return os.path.join(parent(path), os.path.basename(path)) + ".tar"


def is_archive(path: str):
    return path.__contains__(".tar")


def __cleanup_remote_paths__(
    connection: fabric2.Connection,
    *remote_path_list: str,
    base_exception: Exception | None = None,
    ignore_exceptions: bool = False,
):
    exceptions: list[Exception] = [base_exception] if base_exception else []
    for remote_path in remote_path_list:
        try:
            connection.run(f"rm -rf {remote_path}")
        except Exception as e:
            exceptions.append(e)

    if len(exceptions) > 0 and not ignore_exceptions:
        raise Exception(*exceptions)


def __cleanup_local_paths__(
    *local_path_list: str,
    base_exception: Exception | None = None,
    ignore_exceptions: bool = False,
):
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


class RemoteDirectory(ABC):
    def __init__(
        self,
        path: str,
        connection: fabric2.Connection,
    ) -> None:
        super().__init__()
        self.remote_path = path
        self.connection = connection
        self.logger = logging.getLogger(__name__)

    def download(
        self,
        destination_dir_path: str,
        collector: IMetricsCollector,
        s3_config: S3ArtifactUploadConfig,
        list_of_files: list[str] | None = None,
        force_clean_destination: bool = False,
    ):
        """Dispatch to S3 or SFTP download based on S3 config."""
        if s3_config.is_enabled():
            return self.download_s3(
                destination_dir_path,
                collector,
                s3_config,
                list_of_files,
                force_clean_destination,
            )
        return self.download_sftp(destination_dir_path, collector, list_of_files, force_clean_destination)

    def download_sftp(
        self,
        destination_dir_path: str,
        collector: IMetricsCollector,
        list_of_files: list[str] | None = None,
        force_clean_destination: bool = False,
    ):
        """Download files from remote device via SFTP."""
        if not force_clean_destination:
            assert not os.path.exists(
                destination_dir_path
            ), f"{destination_dir_path} already exists locally. Can't download artifacts into it for the fear of overwriting content"
        else:
            shutil.rmtree(os.path.join(destination_dir_path, "*"), ignore_errors=True)

        remote_archive_location = to_archive_name_in_parent(self.remote_path)

        # Time remote compression
        with collector.timer(MetricName.FILE_TRANSFER_COMPRESSION_TIME):
            result: fabric2.Result = self.connection.run(
                create_archive_command(self.remote_path, remote_archive_location, list_of_files)
            )
            if result.failed:
                raise RemoteExecutionException(
                    f"Unable to create tarball in {remote_archive_location} on remote",
                    result,
                )

        # download the archive
        local_archive_location = os.path.join(destination_dir_path, os.path.basename(remote_archive_location))
        download_exception = None

        with collector.timer(MetricName.FILE_TRANSFER_NETWORK_TIME):
            _ = self.connection.get(remote=remote_archive_location, local=local_archive_location)

        # Record compressed bytes
        try:
            compressed_bytes = os.path.getsize(local_archive_location)
            collector.record_metric(MetricName.FILE_TRANSFER_BYTES_COMPRESSED, compressed_bytes, "Bytes")
        except Exception as e:
            download_exception = RemoteFileTransferException(
                f"Unable to download {remote_archive_location} from remote", e
            )
        finally:
            __cleanup_remote_paths__(
                self.connection,
                remote_archive_location,
                base_exception=download_exception,
            )

        # Time local decompression
        with collector.timer(MetricName.FILE_TRANSFER_COMPRESSION_TIME):
            deflate_exception = None
            command_result = subprocess.run(
                defalate_archive_command(local_archive_location, destination_dir_path).split(" ")
            )
            if command_result.returncode != 0:
                deflate_exception = LocalExecutionException(
                    f"Unable to unarchive {local_archive_location}", command_result
                )
        __cleanup_local_paths__(local_archive_location, base_exception=deflate_exception)

        # Record uncompressed bytes
        uncompressed_bytes = 0
        for root, dirs, files in os.walk(destination_dir_path):
            for file in files:
                file_path = os.path.join(root, file)
                uncompressed_bytes += os.path.getsize(file_path)
        collector.record_metric(MetricName.FILE_TRANSFER_BYTES_UNCOMPRESSED, uncompressed_bytes, "Bytes")

        # return where archive was unpacked
        return destination_dir_path

    def download_s3(
        self,
        destination_dir_path: str,
        collector: IMetricsCollector,
        s3_config: S3ArtifactUploadConfig,
        list_of_files: list[str] | None = None,
        force_clean_destination: bool = False,
    ):
        """Download files from remote device via S3 intermediary."""
        if not force_clean_destination:
            assert not os.path.exists(
                destination_dir_path
            ), f"{destination_dir_path} already exists locally. Can't download artifacts into it for the fear of overwriting content"
        else:
            shutil.rmtree(os.path.join(destination_dir_path, "*"), ignore_errors=True)

        s3_client, session = get_s3_client_and_session(s3_config.profile)
        creds = session.get_credentials().get_frozen_credentials()
        s3_key = generate_s3_key(s3_config, S3TransferDirection.OUTPUTS)
        remote_archive_location = to_archive_name_in_parent(self.remote_path)

        # Create archive on remote and upload to S3
        try:
            result: fabric2.Result = self.connection.run(
                create_archive_command(self.remote_path, remote_archive_location, list_of_files)
            )
            if result.failed:
                raise RemoteExecutionException(
                    f"Unable to create tarball in {remote_archive_location} on remote",
                    result,
                )

            # Upload archive from remote to S3
            self.logger.info(f"Uploading from remote to s3://{s3_config.bucket}/{s3_key}...")
            upload_cmd = build_remote_s3_cli_command(
                s3_config.bucket, s3_key, remote_archive_location, creds, S3TransferDirection.OUTPUTS
            )
            with collector.timer(MetricName.FILE_TRANSFER_NETWORK_TIME):
                result: fabric2.Result = self.connection.run(upload_cmd, hide=True)
            if result.failed:
                raise RemoteExecutionException(f"Failed to upload to S3 from remote: {result.stderr}", result)
            self.logger.info(f"Successfully uploaded to s3://{s3_config.bucket}/{s3_key}")

            # Download from S3 to local
            archive_name = os.path.basename(s3_key)
            local_archive_location = os.path.join(destination_dir_path, archive_name)
            os.makedirs(destination_dir_path, exist_ok=True)
            self.logger.info(f"Downloading s3://{s3_config.bucket}/{s3_key} to local...")
            with collector.timer(MetricName.FILE_TRANSFER_NETWORK_TIME):
                s3_client.download_file(s3_config.bucket, s3_key, local_archive_location)
            self.logger.info(f"Successfully downloaded to: {local_archive_location}")

            # Unpack locally
            command_result = subprocess.run(
                defalate_archive_command(local_archive_location, destination_dir_path).split(" ")
            )
            if command_result.returncode != 0:
                raise LocalExecutionException(f"Unable to unarchive {local_archive_location}", command_result)

            __cleanup_local_paths__(local_archive_location, ignore_exceptions=True)
        finally:
            __cleanup_remote_paths__(self.connection, remote_archive_location, ignore_exceptions=True)

        return destination_dir_path

    def upload(
        self,
        local_path: str,
        collector: IMetricsCollector,
        s3_config: S3ArtifactUploadConfig,
        force_local_cleanup: bool = False,
    ):
        """Dispatch to S3 or SFTP upload based on S3 config."""
        if s3_config.is_enabled():
            return self.upload_s3(local_path, s3_config, collector, force_local_cleanup=force_local_cleanup)
        return self.upload_sftp(local_path, collector, force_local_cleanup=force_local_cleanup)

    def upload_sftp(self, local_path: str, collector: IMetricsCollector, force_local_cleanup: bool = False):
        # Record uncompressed upload bytes
        uncompressed_bytes = sum(
            os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(local_path) for f in files
        )
        collector.record_metric(MetricName.FILE_TRANSFER_BYTES_UNCOMPRESSED, uncompressed_bytes, "Bytes")

        assert os.path.exists(local_path)
        original_source_path = local_path  # Save for cleanup

        # Time local compression
        with collector.timer(MetricName.FILE_TRANSFER_COMPRESSION_TIME):
            if not is_archive(local_path):
                # create a tarball if it's not already one
                local_archive_location = to_archive_name_in_parent(local_path)
                command_result = subprocess.run(create_archive_command(local_path, local_archive_location).split(" "))
                if command_result.returncode != 0:
                    raise LocalExecutionException(f"Unable to create tarball of {local_path}", command_result)
                local_path = local_archive_location

                # Delete .bin files immediately after tarball creation to free disk space
                if force_local_cleanup:
                    for bin_file in pathlib.Path(original_source_path).glob("*.bin"):
                        bin_file.unlink(missing_ok=True)

        remote_archive_location = os.path.join(self.remote_path, os.path.basename(local_path))
        upload_exception = None

        # Time network transfer only
        with collector.timer(MetricName.FILE_TRANSFER_NETWORK_TIME):
            # Create with global permissions so any user can use the shared test directory
            _ = self.connection.run(f"mkdir -p -m 777 {self.remote_path}")
            _ = self.connection.put(remote=remote_archive_location, local=local_path)

        # Record bytes (after timer exits)
        try:
            compressed_bytes = os.path.getsize(local_path)
            collector.record_metric(MetricName.FILE_TRANSFER_BYTES_COMPRESSED, compressed_bytes, "Bytes")
        except Exception as e:
            upload_exception = RemoteFileTransferException(
                f"Unable to upload {local_path} to {remote_archive_location} to remote",
                e,
            )
            # only attempt clean up remote path in case we failed to upload because we need it
            # later on
            __cleanup_remote_paths__(self.connection, remote_archive_location, ignore_exceptions=True)
        finally:
            __cleanup_local_paths__(local_path, base_exception=upload_exception)

        # unpack uploaded tarball
        deflate_exception = None
        try:
            with collector.timer(MetricName.FILE_TRANSFER_COMPRESSION_TIME):
                result: fabric2.Result = self.connection.run(
                    defalate_archive_command(remote_archive_location, self.remote_path)
                )
            if result.failed:
                deflate_exception = RemoteExecutionException(
                    f"Unable to create tarball in {self.remote_path} on remote",
                    result,
                )
        finally:
            __cleanup_local_paths__(local_path, ignore_exceptions=True)
            __cleanup_remote_paths__(
                self.connection,
                remote_archive_location,
                base_exception=deflate_exception,
            )

    def upload_s3(
        self,
        local_path: str,
        s3_config: S3ArtifactUploadConfig,
        collector: IMetricsCollector,
        force_local_cleanup: bool = False,
    ):
        """Upload files to remote device via S3 intermediary."""
        s3_client, session = get_s3_client_and_session(s3_config.profile)

        assert os.path.exists(local_path), f"Local path {local_path} does not exist"
        original_source_path = local_path  # Save for cleanup

        s3_key = generate_s3_key(s3_config, S3TransferDirection.INPUTS)
        archive_name = os.path.basename(s3_key)

        # Compress if needed
        local_archive_location = None
        if not is_archive(local_path):
            local_archive_location = to_archive_name_in_parent(local_path)
            command_result = subprocess.run(create_archive_command(local_path, local_archive_location).split(" "))
            if command_result.returncode != 0:
                raise LocalExecutionException(f"Unable to create tarball of {local_path}", command_result)
            local_path = local_archive_location

            # Delete .bin files immediately after tarball creation to free disk space
            if force_local_cleanup:
                for bin_file in pathlib.Path(original_source_path).glob("*.bin"):
                    bin_file.unlink(missing_ok=True)

        try:
            self.logger.info(f"Uploading {local_path} to s3://{s3_config.bucket}/{s3_key}...")
            with collector.timer(MetricName.FILE_TRANSFER_NETWORK_TIME):
                s3_client.upload_file(local_path, s3_config.bucket, s3_key)
            self.logger.info(f"Successfully uploaded to s3://{s3_config.bucket}/{s3_key}")
        finally:
            if local_archive_location and os.path.exists(local_archive_location):
                __cleanup_local_paths__(local_archive_location, ignore_exceptions=True)

        # Download and deflate on remote device
        remote_archive_location = os.path.join(self.remote_path, archive_name)
        deflate_exception = None

        try:
            # Create with global permissions so any user can use the shared test directory
            self.connection.run(f"mkdir -p -m 777 {self.remote_path}")

            # Get credentials from boto3 session for remote download
            creds = session.get_credentials().get_frozen_credentials()
            self.logger.info(f"Downloading s3://{s3_config.bucket}/{s3_key} on remote device...")
            download_cmd = build_remote_s3_cli_command(
                s3_config.bucket, s3_key, remote_archive_location, creds, S3TransferDirection.INPUTS
            )
            with collector.timer(MetricName.FILE_TRANSFER_NETWORK_TIME):
                result: fabric2.Result = self.connection.run(download_cmd, hide=True)
            if result.failed:
                raise RemoteExecutionException(f"Failed to download from S3 on remote: {result.stderr}", result)
            self.logger.info(f"Successfully downloaded to remote: {remote_archive_location}")

            # Deflate archive on remote
            result: fabric2.Result = self.connection.run(
                defalate_archive_command(remote_archive_location, self.remote_path)
            )
            if result.failed:
                deflate_exception = RemoteExecutionException(
                    f"Unable to deflate archive in {self.remote_path} on remote", result
                )
        finally:
            __cleanup_remote_paths__(
                self.connection, remote_archive_location, base_exception=deflate_exception, ignore_exceptions=True
            )

    def cleanup(self):
        result: fabric2.Result = self.connection.run(f"rm -rf {self.remote_path}")
        if result.failed:
            raise RemoteExecutionException(f"Unable to cleanup remote folder {self.remote_path}", result)
