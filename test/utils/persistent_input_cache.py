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
"""Persistent raw-array cache for expensive per-rank test inputs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import shutil
import tempfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

log = logging.getLogger(__name__)

_CACHE_FORMAT_VERSION = 1
_MANIFEST_FILE = "manifest.pkl"


def link_raw_memmap(value: np.ndarray, destination: str | os.PathLike[str]) -> bool:
    """Hard-link an exact raw-file memmap into an artifact directory."""
    if not isinstance(value, np.memmap) or value.offset != 0 or not value.flags.c_contiguous:
        return False
    source = os.path.realpath(os.fspath(value.filename))  # ty: ignore — memmap.filename is str here
    destination = os.fspath(destination)
    if not os.path.isfile(source) or os.path.getsize(source) != value.nbytes:
        return False
    try:
        if os.path.lexists(destination):
            os.unlink(destination)
        os.link(source, destination)
        return True
    except OSError:
        log.debug("Could not hard-link cached input %s to %s", source, destination, exc_info=True)
        return False


def build_input_cache_key(payload: Mapping[str, Any], source_files: Iterable[str | os.PathLike[str]]) -> str:
    """Fingerprint cache configuration and the source that generates its arrays."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"format_version": _CACHE_FORMAT_VERSION, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    for source_file in sorted({os.path.realpath(os.fspath(path)) for path in source_files}):
        digest.update(source_file.encode())
        with open(source_file, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()[:32]


class PersistentPerRankInputCache:
    """Store ndarray inputs as raw files and return read-only memory maps.

    The cache owns only ndarray values. Callers should reconstruct scalar and
    object-valued kernel arguments from source on every load.
    """

    def __init__(self, root: str | os.PathLike[str], key: str) -> None:
        self.directory = Path(root).expanduser().resolve() / key

    def get_or_create(
        self,
        rank_id: int,
        generator: Callable[[], Mapping[str, np.ndarray]],
    ) -> dict[str, np.ndarray]:
        rank_directory = self.directory / f"rank-{rank_id:03d}"
        cached = self._load(rank_directory)
        if cached is not None:
            log.info("Persistent input cache hit: %s", rank_directory)
            return cached

        self.directory.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(rank_directory) + ".lock")
        with lock:
            cached = self._load(rank_directory)
            if cached is not None:
                log.info("Persistent input cache hit after lock: %s", rank_directory)
                return cached

            log.info("Persistent input cache miss: generating %s", rank_directory)
            arrays = dict(generator())
            invalid = {
                name: type(value).__name__ for name, value in arrays.items() if not isinstance(value, np.ndarray)
            }
            if invalid:
                raise TypeError(f"Persistent input cache accepts only numpy arrays, got {invalid}")
            self._store(rank_directory, arrays)

        cached = self._load(rank_directory)
        if cached is None:
            raise RuntimeError(f"Persistent input cache write did not produce a valid entry: {rank_directory}")
        return cached

    @staticmethod
    def _load(rank_directory: Path) -> dict[str, np.ndarray] | None:
        manifest_path = rank_directory / _MANIFEST_FILE
        if not manifest_path.is_file():
            return None
        try:
            with open(manifest_path, "rb") as f:
                manifest = pickle.load(f)

            arrays: dict[str, np.ndarray] = {}
            for name, spec in manifest.items():
                file_path = rank_directory / spec["file"]
                dtype = spec["dtype"]
                shape = tuple(spec["shape"])
                expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if file_path.stat().st_size != expected_bytes:
                    return None
                if expected_bytes == 0:
                    arrays[name] = np.empty(shape, dtype=dtype)
                else:
                    arrays[name] = np.memmap(file_path, dtype=dtype, mode="r", shape=shape, order="C")
            return arrays
        except (OSError, pickle.PickleError, TypeError, ValueError, EOFError, KeyError):
            log.warning("Ignoring invalid persistent input cache entry: %s", rank_directory, exc_info=True)
            return None

    @staticmethod
    def _store(rank_directory: Path, arrays: Mapping[str, np.ndarray]) -> None:
        rank_directory.parent.mkdir(parents=True, exist_ok=True)
        temp_directory = Path(tempfile.mkdtemp(prefix=f".{rank_directory.name}-", dir=rank_directory.parent))
        try:
            manifest = {}
            for index, name in enumerate(sorted(arrays)):
                value = np.ascontiguousarray(arrays[name])
                file_name = f"{index:03d}.bin"
                value.tofile(temp_directory / file_name)
                manifest[name] = {
                    "file": file_name,
                    "dtype": value.dtype,
                    "shape": value.shape,
                }

            with open(temp_directory / _MANIFEST_FILE, "wb") as f:
                pickle.dump(manifest, f, protocol=pickle.HIGHEST_PROTOCOL)

            if rank_directory.exists():
                shutil.rmtree(rank_directory)
            os.replace(temp_directory, rank_directory)
        finally:
            shutil.rmtree(temp_directory, ignore_errors=True)
