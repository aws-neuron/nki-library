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
"""S3-backed cache for torch-reference (golden) outputs.

For validation-bound kernels, where the torch reference dominates elapsed time,
recomputing the golden every run is wasted work. This cache stores the reference
output keyed by (reference dependency hash, input hash) and serves it from S3 on
a hit, replacing the recompute with a download.

Cache key layout (mirrors the NEFF cache's version-in-path convention):
    s3://<cache_path>/<version_hash>/<ref_qualname>/<dep_hash>/<input_hash>/golden.bin

The golden is stored as raw per-array bytes plus a small JSON manifest of
{name: [dtype, shape]} (see _serialize_golden). This carries the dtype
out-of-band and reapplies it on load — the same mechanic the hardware-output
reader uses (np.fromfile with an explicit dtype) — so ml_dtypes extension dtypes
(bfloat16, float8_*) round-trip losslessly, which np.savez/np.load cannot do
(they reload as opaque void).

  - version_hash fingerprints third-party deps whose change alters golden
              numerics (neuron_dtypes C-extension .so bytes + ml_dtypes/numpy/
              torch versions) — a dep bump or C-ext rebuild namespaces the whole
              cache afresh (clean miss), the way NEFF encodes neuronxcc version.

  - dep_hash  invalidates on ANY edit to the reference's transitive in-package
              import graph (see torch_ref_invalidation.ref_dependency_hash) —
              so a helper-function change never serves a stale golden.
  - input_hash content-hashes the kwargs (numpy arrays by raw bytes — safe for
              custom MX x4 dtypes, which np.array_equal cannot compare).

Correctness contract:
  - A cached reference MUST be a pure function of (kwargs, in-package source,
    pinned third-party deps). No external data-file reads, no RNG, no env/time
    dependence, no first-party imports outside the hashed prefixes — any of these
    silently serves a stale golden. See torch_ref_invalidation's "REFERENCE
    PURITY CONTRACT" for the full enumeration and the current-reference audit.

Safety:
  - Fail-open: any S3/serialization error, or any error building the cache key
    (unfingerprintable input, malformed URI), falls back to recomputing. The
    cache can only skip work, never cause a failure.
  - Only plain ndarray / tensor dict goldens are cached; anything else (custom
    validators, non-array values) bypasses the cache. All numpy dtypes are
    supported, including ml_dtypes extension types.
  - Goldens above MAX_GOLDEN_BYTES are not stored: past ~1 GiB the S3 round-trip
    costs more than recomputing, so caching would be net negative.
  - No cross-worker locking on a cold cache: if two xdist workers miss the same
    key at once, each recomputes and stores the golden (last writer wins; the
    data is identical, since the golden is deterministic). Benign in practice —
    workers shard distinct configs, so they rarely collide — and a distributed
    lock isn't worth the complexity.
"""

import enum
import hashlib
import io
import json
import logging
from typing import Any

import ml_dtypes
import numpy as np
import numpy.typing as npt
import torch

from .metrics_collector import IMetricsCollector, MetricName
from .s3_utils import parse_s3_uri
from .torch_ref_invalidation import dependency_version_hash

log = logging.getLogger(__name__)

# Don't cache goldens larger than this (serialized size). Past ~1 GiB the S3
# round-trip costs more than recomputing: empirically a ~2 GiB golden downloads in
# ~24s (median) vs ~12s to recompute, so caching it is net negative. Declining
# them here lets those configs recompute (identical to the no-cache path).
MAX_GOLDEN_BYTES = 1 * 1024**3


def _array_to_bytes(arr: npt.NDArray[Any]) -> bytes:
    """Flatten an ndarray to its raw bytes in canonical (C/row-major) order.

    ascontiguousarray normalizes non-contiguous inputs (transposes, strided slices)
    so identical values always yield identical bytes; the uint8 view works for any
    dtype (incl. custom x4/bf16 that == can't compare); reshape(-1) handles 0-d
    scalars (a 0-d array can't be re-dtyped directly).
    """
    return np.ascontiguousarray(arr).view(np.uint8).reshape(-1).tobytes()


def _hash_value(h: "hashlib._Hash", value: Any) -> bool:
    """Mix one kwarg value into the hash. Returns False if not fingerprintable.

    Each supported type has a distinct tag prefix so different types can't collide.
    """
    if isinstance(value, np.ndarray):
        h.update(b"\x00ndarray\x00")
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(_array_to_bytes(value))
        return True
    if value is None:
        h.update(b"\x00none\x00")
        return True
    if isinstance(value, (int, float, bool, str)):
        h.update(b"\x00scalar\x00")
        h.update(json.dumps(value).encode())
        return True
    if isinstance(value, enum.Enum):
        h.update(b"\x00enum\x00")
        h.update(type(value).__name__.encode())
        h.update(json.dumps(value.value).encode())
        return True
    if isinstance(value, torch.Tensor):
        # .numpy() raises on a tensor that requires grad (needs detach first) or
        # lives on a non-CPU device (needs a host copy). detach().cpu() normalizes
        # both so any input tensor is hashable; the values are identical to what the
        # reference would see, so the key is unaffected.
        arr = value.detach().cpu().numpy()
        h.update(b"\x00tensor\x00")
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(_array_to_bytes(arr))
        return True
    if isinstance(value, (tuple, list)):
        # sequence of values (e.g. a tile shape tuple): hash each element in order
        h.update(b"\x00seq\x00")
        h.update(str(len(value)).encode())
        for item in value:
            if not _hash_value(h, item):
                return False
        return True
    if isinstance(value, dict):
        # mapping (e.g. a config kwarg): hash each (key, value) in a stable key
        # order so dict insertion order never changes the digest. repr the keys so
        # mixed key types stay comparable to sort and type-distinct ('1' vs 1).
        h.update(b"\x00dict\x00")
        h.update(str(len(value)).encode())
        for k in sorted(value, key=repr):
            h.update(repr(k).encode())
            h.update(b"\x00")
            if not _hash_value(h, value[k]):
                return False
        return True
    if hasattr(value, "__dict__") and value.__dict__:
        # simple data-holder (e.g. SkipMode): hash attrs if all fingerprintable
        h.update(b"\x00obj\x00")
        h.update(type(value).__name__.encode())
        for attr in sorted(value.__dict__):
            h.update(attr.encode())
            if not _hash_value(h, value.__dict__[attr]):
                return False
        return True
    return False


def _input_hash(kwargs: dict[str, Any]) -> str | None:
    h = hashlib.sha256()
    for key in sorted(kwargs):
        h.update(key.encode())
        h.update(b"\x00")  # explicit key/value boundary (provably collision-free)
        if not _hash_value(h, kwargs[key]):
            log.debug(
                "torch-ref cache: kwarg %r (%s) not fingerprintable; skipping cache", key, type(kwargs[key]).__name__
            )
            return None
    return h.hexdigest()[:32]


def _golden_to_arrays(result: Any) -> dict[str, npt.NDArray[Any]] | None:
    """Coerce a golden dict to {name: ndarray}, or None if not a plain array dict.

    Custom dtypes (ml_dtypes bfloat16/float8_*, native-packed MX x4) are kept — the
    raw-bytes-plus-metadata format below round-trips them losslessly, so no
    dtype-based rejection here. Returns None only when the golden is not a dict of
    ndarray/tensor values (e.g. custom validators), which the cache can't store.
    """
    if not isinstance(result, dict):
        return None
    out = {}
    for k, v in result.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        if not isinstance(v, np.ndarray):
            return None
        out[k] = v
    return out


def _resolve_dtype(name: str) -> np.dtype:
    """Resolve a dtype name string back to a numpy dtype, supporting ml_dtypes
    extension types (bfloat16, float8_*). Native names resolve via np.dtype; the
    rest fall back to an ml_dtypes attribute. Never evals — an unknown name raises.
    """
    try:
        return np.dtype(name)
    except TypeError as e:
        dt = getattr(ml_dtypes, name, None)
        if dt is None:
            raise TypeError(f"unresolvable golden dtype {name!r}") from e
        return np.dtype(dt)


def _serialize_golden(arrays: dict[str, npt.NDArray[Any]]) -> bytes:
    """Serialize {name: ndarray} as raw uint8 bytes per array plus an out-of-band
    {name: [dtype_str, shape]} manifest.

    This is numpy's standard raw-bytes serialization (the tofile/frombuffer +
    explicit-dtype pattern), which is exactly how the repo already round-trips
    hardware output tensors (see load_output_tensor_as_bytes / the .tofile dumps) —
    so this reuses an established, tested idiom rather than a bespoke format. The
    dtype is carried in the manifest and reapplied on load, which is required
    because ml_dtypes extension dtypes (bfloat16, float8_*) do NOT survive np.save/
    np.savez (they reload as opaque void).
    """
    manifest: dict[str, Any] = {}
    payload = io.BytesIO()
    for name in sorted(arrays):
        orig = arrays[name]
        # record dtype/shape from the ORIGINAL array (not the contiguous copy, whose
        # reshape would lose a 0-d scalar's () shape).
        manifest[name] = [str(orig.dtype), list(orig.shape)]
        payload.write(_array_to_bytes(orig))
    manifest_bytes = json.dumps(manifest).encode()
    # framing: [4-byte big-endian manifest length][manifest json][concatenated raw arrays]
    return len(manifest_bytes).to_bytes(4, "big") + manifest_bytes + payload.getvalue()


def _deserialize_golden(body: bytes) -> dict[str, npt.NDArray[Any]]:
    """Inverse of _serialize_golden: rebuild {name: ndarray} from raw bytes +
    manifest, restoring each array's original (possibly ml_dtypes) dtype/shape."""
    mlen = int.from_bytes(body[:4], "big")
    manifest = json.loads(body[4 : 4 + mlen].decode())
    data = body[4 + mlen :]
    out: dict[str, npt.NDArray[Any]] = {}
    offset = 0
    for name in sorted(manifest):
        dtype_str, shape = manifest[name]
        dt = _resolve_dtype(dtype_str)
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * dt.itemsize
        # .copy(): np.frombuffer over immutable bytes yields a read-only array, so
        # a cache-hit golden would be non-writable while a freshly computed (miss)
        # golden is writable. Copy so hits and misses are both writable, owned
        # arrays — downstream in-place mutation must not depend on hit/miss.
        arr = np.frombuffer(data[offset : offset + nbytes], dtype=dt).reshape(shape).copy()
        out[name] = arr
        offset += nbytes
    return out


def _s3_key(prefix: str, version_hash: str, ref_name: str, dep_hash: str, input_hash: str) -> str:
    safe = ref_name.replace("/", "_")
    # version_hash namespaces the whole cache by third-party dep versions/C-ext
    # bytes (mirrors NEFF's neuronxcc-version-in-path), so a dep bump = clean miss.
    base = f"{version_hash}/{safe}/{dep_hash}/{input_hash}/golden.bin"
    return f"{prefix}/{base}" if prefix else base


class GoldenCache:
    """A passive S3-backed store for golden outputs: load a golden by key, or
    store one. It computes nothing — the caller owns "what to do on a miss".

    This keeps caching a bolt-on: an orchestrator (GoldenProvider) tries load();
    on None it computes the golden its own way and calls store(). Both methods
    fail open — a load never raises (returns None), a store never raises — so
    caching can only skip work, never cause a failure.

    Overhead accounting (so a hit/miss can be compared against the cost of just
    recomputing — important for cheap goldens where the cache may not pay off):
      - TorchRefCacheLookupTime: all load-side work — input hashing, key
        construction, the S3 GET, and (on a hit) deserialization. Timed from
        load() entry so the input-hash cost (a full raw-byte pass over every
        kwarg array — non-trivial for large MX tensors) is included, not hidden.
        Recorded on every load path, including the fail-open early-returns.
      - TorchRefCacheStoreTime: the store-side serialization plus the S3 PUT.
    """

    def __init__(self, cache_path: str, collector: IMetricsCollector, s3_client: Any):
        self._cache_path = cache_path
        self._collector = collector
        # Injected by the caller so this class holds no S3 setup and tests can pass a mock.
        self._s3_client = s3_client

    def _key(self, ref_qualname: str, dep_hash: str, inputs: dict[str, Any]) -> str | None:
        """Build the S3 key for these inputs, or None if inputs aren't
        fingerprintable. May raise on a malformed cache path (callers fail open)."""
        ih = _input_hash(inputs)
        if ih is None:
            return None
        parsed = parse_s3_uri(self._cache_path)
        return _s3_key(parsed.prefix, dependency_version_hash(), ref_qualname, dep_hash, ih)

    def load(self, ref_qualname: str, dep_hash: str, inputs: dict[str, Any]) -> Any | None:
        """Return the cached golden dict, or None on miss / uncacheable input /
        any error. Never raises."""
        # Times every path (miss/hit/error) incl. input hashing.
        with self._collector.timer(MetricName.TORCH_REF_CACHE_LOOKUP_TIME):
            try:
                key = self._key(ref_qualname, dep_hash, inputs)
                if key is None:
                    return None
                parsed = parse_s3_uri(self._cache_path)
            except Exception as e:
                log.debug("torch-ref cache: key construction failed (%s); treating as miss", e)
                return None

            try:
                s3 = self._s3_client
                body = s3.get_object(Bucket=parsed.bucket, Key=key)["Body"].read()
                result = _deserialize_golden(body)
                self._collector.record_metric(MetricName.TORCH_REF_CACHE_HIT, 1, "Count")
                log.info("torch-ref cache HIT: s3://%s/%s", parsed.bucket, key)
                return result
            except Exception as e:
                self._collector.record_metric(MetricName.TORCH_REF_CACHE_HIT, 0, "Count")
                log.debug("torch-ref cache miss: s3://%s/%s (%s)", parsed.bucket, key, e)
                return None

    def store(self, ref_qualname: str, dep_hash: str, inputs: dict[str, Any], golden: Any) -> None:
        """Best-effort store of a freshly computed golden. Never raises. No-op if
        the golden isn't a plain array dict, exceeds the size cap, or the key
        can't be built."""
        arrays = _golden_to_arrays(golden)
        if arrays is None:
            return
        try:
            key = self._key(ref_qualname, dep_hash, inputs)
            if key is None:
                return
            parsed = parse_s3_uri(self._cache_path)
        except Exception as e:
            log.debug("torch-ref cache: key construction failed (%s); not storing", e)
            return

        try:
            payload = _serialize_golden(arrays)
            if len(payload) > MAX_GOLDEN_BYTES:
                log.info("torch-ref cache: golden %d bytes > cap %d; not storing", len(payload), MAX_GOLDEN_BYTES)
                return
            # STORE_TIME covers the S3 PUT (the dominant store cost); recorded only
            # when a store actually happens, not on the oversize/decline path. (The
            # serialize above must run first to size-gate against MAX_GOLDEN_BYTES.)
            with self._collector.timer(MetricName.TORCH_REF_CACHE_STORE_TIME):
                s3 = self._s3_client
                s3.put_object(Bucket=parsed.bucket, Key=key, Body=payload)
            log.info("torch-ref cache STORED: s3://%s/%s (%d bytes)", parsed.bucket, key, len(payload))
        except Exception as e:
            log.warning("torch-ref cache: failed to store s3://%s/%s: %s", parsed.bucket, key, e)
