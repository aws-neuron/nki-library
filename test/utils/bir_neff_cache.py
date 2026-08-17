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
"""S3-backed cache for BIR-to-NEFF compilation outputs."""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

import neuronxcc

from .metrics_collector import IMetricsCollector, MetricName
from .s3_utils import S3Uri, get_s3_client_and_session, parse_s3_uri

log = logging.getLogger(__name__)


@dataclass
class CachedCompiledKernel:
    """Drop-in replacement for compile_bir_to_neff return value on cache hit."""

    neff_path: str
    mlir_time: float = 0.0
    neuronx_cc_time: float = 0.0
    input_output_aliases: dict = field(default_factory=dict)


def _s3_keys(parsed_uri: S3Uri, cache_key: str) -> tuple[str, str]:
    """Return (neff_key, meta_key) under prefix/{version}/{cache_key}/."""
    version = neuronxcc.__version__
    prefix = parsed_uri.prefix
    base = f"{prefix}/{version}/{cache_key}" if prefix else f"{version}/{cache_key}"
    return f"{base}/file.neff", f"{base}/metadata.json"


def _hash_file(h, path: str) -> None:
    """Stream file contents into a hashlib object in 64KB chunks."""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)


# Compiler artifacts may have hard coded source locations, need to canonicalize
# to keep cache key stable across platforms.
_PATH_PREFIX_RE = re.compile(rb'[^"\s]*/site-packages/')


def _hash_normalized_json(h, path: str) -> None:
    """Hash a BIR JSON file with host-specific source-path prefixes stripped."""
    with open(path, "rb") as f:
        data = f.read()
    h.update(_PATH_PREFIX_RE.sub(b"site-packages/", data))


def compute_cache_key(bir_result: Any, compile_opts: Any) -> str | None:
    """Return a SHA-256 hex digest over BIR kernel JSON, constant files, and compile options.

    The kernel JSON at bir_result.descriptor.kernel_json_path is the actual input
    to neuronx-cc. Only the .npy constant files in the same directory are also
    hashed; other sibling files are compiler outputs/debug artifacts that may embed
    host-specific paths or timestamps and must not influence the key.

    The neuronxcc version is encoded in the S3 path rather than the key.
    Returns None if the kernel JSON is missing.
    """
    kernel_json_path = bir_result.descriptor.kernel_json_path
    if not os.path.exists(kernel_json_path):
        log.warning("BIR-to-NEFF cache: kernel JSON not found at %s; skipping cache", kernel_json_path)
        return None

    h = hashlib.sha256()

    _hash_normalized_json(h, kernel_json_path)

    # Hash only the path-independent constant inputs (.npy) alongside the kernel
    # JSON.
    kernel_dir = os.path.dirname(kernel_json_path)
    for aux_file in sorted(os.listdir(kernel_dir)):
        if aux_file.endswith(".npy"):
            _hash_file(h, os.path.join(kernel_dir, aux_file))

    opts = {
        "target": compile_opts.target,
        "lnc": compile_opts.lnc,
        "neuronx_cc_args": " ".join(sorted(compile_opts.neuronx_cc_args)),
        # Device dump mode adds instrumentation to the NEFF, producing different output
        "enable_device_dump": compile_opts.enable_device_dump,
    }
    h.update(json.dumps(opts, sort_keys=True).encode())

    return h.hexdigest()


def lookup_cache(
    cache_path: str, cache_key: str, local_neff_path: str, collector: IMetricsCollector
) -> CachedCompiledKernel | None:
    """Download NEFF from S3 cache if present. Records timing via collector."""
    parsed_uri = parse_s3_uri(cache_path)
    neff_key, meta_key = _s3_keys(parsed_uri, cache_key)

    start = time.monotonic()
    try:
        s3, _ = get_s3_client_and_session()
        os.makedirs(os.path.dirname(local_neff_path), exist_ok=True)
        s3.download_file(parsed_uri.bucket, neff_key, local_neff_path)
        meta = json.loads(s3.get_object(Bucket=parsed_uri.bucket, Key=meta_key)["Body"].read())
        # JSON only allows string keys, so the int output indices stored at
        # cache-write time come back as strings. The framework's rename loop
        # in test_orchestrator.py looks them up by int (`i in aliases`), so
        # without this coercion every cache hit produces a missing-rename and
        # the validator fails with "<name> was not emitted by neuron-explorer
        # capture". Coerce on read so existing entries stay usable.
        raw_aliases = meta.get("input_output_aliases") or {}
        aliases = {int(k): v for k, v in raw_aliases.items()}
        elapsed = time.monotonic() - start
        collector.record_timer(MetricName.NEFF_CACHE_LOOKUP_TIME, elapsed)
        collector.record_metric(MetricName.NEFF_CACHE_HIT, 1, "Count")
        log.info("BIR-to-NEFF cache HIT: s3://%s/%s (%.2fs)", parsed_uri.bucket, neff_key, elapsed)
        return CachedCompiledKernel(
            neff_path=local_neff_path,
            input_output_aliases=aliases,
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        collector.record_timer(MetricName.NEFF_CACHE_LOOKUP_TIME, elapsed)
        collector.record_metric(MetricName.NEFF_CACHE_HIT, 0, "Count")
        log.debug("BIR-to-NEFF cache miss: s3://%s/%s (%s)", parsed_uri.bucket, neff_key, e)
        return None


def store_cache(cache_path: str, cache_key: str, compiled: Any, collector: IMetricsCollector) -> None:
    """Upload NEFF and metadata to S3 cache. Records timing via collector."""
    parsed_uri = parse_s3_uri(cache_path)
    neff_key, meta_key = _s3_keys(parsed_uri, cache_key)

    # Validate the producer contract before serializing. NKI declares
    # input_output_aliases as dict[int, str] in nki.compiler.ncc_driver, and
    # the framework rename loop in test_orchestrator.py looks up entries via
    # `i in aliases` with int keys. If NKI ever changes the type, JSON
    # round-tripping plus our int-coercion-on-read would silently break. Fail
    # loudly here so contract drift surfaces at the boundary instead of as
    # downstream validation failures.
    aliases = compiled.input_output_aliases or {}
    for k, v in aliases.items():
        assert isinstance(k, int), (
            f"BIR-to-NEFF cache: input_output_aliases key must be int "
            f"(NKI declares dict[int, str] in nki.compiler.ncc_driver), "
            f"got {type(k).__name__}={k!r}"
        )
        assert isinstance(v, str), (
            f"BIR-to-NEFF cache: input_output_aliases value must be str, got {type(v).__name__}={v!r}"
        )

    start = time.monotonic()
    try:
        s3, _ = get_s3_client_and_session()
        s3.upload_file(compiled.neff_path, parsed_uri.bucket, neff_key)
        meta = json.dumps({"input_output_aliases": aliases})
        s3.put_object(Bucket=parsed_uri.bucket, Key=meta_key, Body=meta.encode())
        elapsed = time.monotonic() - start
        collector.record_timer(MetricName.NEFF_CACHE_STORE_TIME, elapsed)
        log.info("BIR-to-NEFF cache STORED: s3://%s/%s (%.2fs)", parsed_uri.bucket, neff_key, elapsed)
    except Exception as e:
        elapsed = time.monotonic() - start
        collector.record_timer(MetricName.NEFF_CACHE_STORE_TIME, elapsed)
        log.warning("BIR-to-NEFF cache: failed to store s3://%s/%s: %s", parsed_uri.bucket, neff_key, e)
