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
"""Golden provider.

Owns the "how do I produce a golden" strategy so callers (unit_test_framework)
need not know whether or how it is cached. Session context — the metrics
collector and the S3 torch-ref cache path — is held once on the provider, so the
golden lifecycle (including caching) is decided in one place rather than wired
into every caller.

Scope: this serves the standard (non-custom-validator) golden path, where a
reference is a pure function of its inputs and its output is consumed as plain
arrays. Custom-validator kernels that compute a golden inside their own
validate() currently manage caching themselves via torch_ref_cache; migrating
them onto this provider is future work.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, runtime_checkable

import numpy.typing as npt

from .metrics_collector import IMetricsCollector
from .s3_utils import get_s3_client_and_session
from .torch_ref_cache import GoldenCache
from .torch_ref_invalidation import ref_dependency_hash

GoldenDict = dict[str, npt.NDArray[Any]]


def mark_uncacheable(reference: Callable) -> Callable:
    """Opt a torch reference out of the golden cache.

    Use for a reference whose comparator depends on a SIDE EFFECT of running the
    reference — typically the ref stashes an extra value in a closure holder dict
    that the comparator later reads (e.g. rmsnorm-quant's dequant_scale,
    rmsnorm-mx-prefill's expert_affinities). The cache serves the golden dict on a
    HIT without invoking the reference, so that holder stays empty and the
    comparator raises KeyError. Marking the ref uncacheable forces a recompute
    every run, so the side effect always fires. Returns the reference for use as a
    decorator (applied outside @torch_ref_wrapper).
    """
    ref: Any = reference
    ref._torch_ref_uncacheable = True
    return reference


@runtime_checkable
class CachedTorchRef(Protocol):
    """A torch reference that carries its own golden-affecting construction options.

    A reference wrapped for numpy<->torch conversion closes over options (e.g.
    output precision) that change the golden but are absent from its kwargs, so it
    publishes them as an attribute for the cache key to fold in. A plain callable
    makes no such promise, which is why this is a separate, narrower contract.
    """

    _torch_ref_cache_options: dict[str, Any]

    def __call__(self, **kwargs: Any) -> Any: ...


class GoldenProvider:
    """Provides goldens for the standard path, transparently caching when enabled.

    Holds session context (collector + cache path) once. Callers hand it a
    reference and inputs and get back a zero-arg callable that yields the golden;
    whether that callable hits S3 or recomputes is the provider's concern, not the
    caller's.

    Caching is a bolt-on here: the provider owns the compute, and treats the
    GoldenCache as an optional keyed store — try load(), and on a miss run the
    compute and store() the result. When no cache path is set, no cache is built
    and the compute runs directly.
    """

    def __init__(self, collector: IMetricsCollector, torch_ref_cache_path: str | None):
        self._collector = collector
        self._cache_path = torch_ref_cache_path

    def get(
        self,
        compute: Callable[[], GoldenDict],
        reference: Callable,
        inputs: dict[str, Any],
    ) -> Callable[[], GoldenDict]:
        """Return a zero-arg callable that yields the golden.

        compute: runs the reference and returns the raw golden dict.
        reference: the reference function, used to key the cache on its
            transitive source (so a helper edit invalidates).
        inputs: the reference's kwargs, content-hashed for the cache key.

        Caching is applied only when a cache path is set and the reference is
        cacheable (its source can be hashed); otherwise the returned callable is
        just `compute`. All of that is internal — the caller only sees "a thing
        that yields a golden".
        """
        if not self._cache_path:
            return compute
        if getattr(reference, "_torch_ref_uncacheable", False):
            # The reference has a comparator-visible side effect (it stashes a value
            # in a closure holder that the comparator later reads). A cache HIT would
            # skip compute() and leave that holder empty, so the comparator raises
            # KeyError. Refs marked this way always recompute — see mark_uncacheable.
            return compute
        dep_hash = ref_dependency_hash(reference)
        if dep_hash is None:
            # Uncacheable reference (source unobtainable): run uncached.
            return compute
        ref_qualname = getattr(reference, "__qualname__", "torch_ref")
        cache = GoldenCache(self._cache_path, self._collector, get_s3_client_and_session()[0])

        # Fold a wrapper's golden-affecting options into the key so variants over the
        # same ref don't collide. An unfingerprintable option makes the input hash
        # None -> load/store no-op (recompute): fail closed, never a wrong hit.
        options = reference._torch_ref_cache_options if isinstance(reference, CachedTorchRef) else None
        key_inputs = {**inputs, "__torch_ref_cache_options__": options} if options else inputs

        def cached() -> GoldenDict:
            hit = cache.load(ref_qualname, dep_hash, key_inputs)
            if hit is not None:
                return hit
            golden = compute()
            cache.store(ref_qualname, dep_hash, key_inputs, golden)
            return golden

        return cached


class GoldenProducer(ABC):
    """Produces the golden the OutputValidator compares against, as a pluggable stage
    separate from the validator. Isolating production is what lets caching
    (GoldenProvider) attach transparently, so the same cached golden feeds the plain
    and custom_comparator paths alike."""

    @abstractmethod
    def output_spec(self) -> GoldenDict:
        """Output placeholders (shapes/dtypes) the compiler needs, without computing
        the golden — keeps compile/trace-only paths valid."""

    @abstractmethod
    def produce(self):
        """Return the golden the validator consumes (from cache when enabled)."""


class TorchRefProducer(GoldenProducer):
    """Produces the golden by running a torch reference through GoldenProvider, so the
    result is cached whenever a cache path is set."""

    def __init__(
        self,
        compute: Callable[[], GoldenDict],
        reference: Callable,
        inputs: dict[str, Any],
        output_spec: GoldenDict,
        collector: IMetricsCollector,
        torch_ref_cache_path: str | None,
    ):
        self._compute = compute
        self._reference = reference
        self._inputs = inputs
        self._output_spec = output_spec
        self._provider = GoldenProvider(collector, torch_ref_cache_path)

    def output_spec(self) -> GoldenDict:
        return self._output_spec

    def produce(self) -> GoldenDict:
        return self._provider.get(self._compute, self._reference, self._inputs)()


class CustomComparatorProducer(GoldenProducer):
    """Wraps a base producer and applies a custom_comparator to its (cached) golden,
    yielding the per-output validators the OutputValidator dispatches on. Because the
    base produces and caches the golden first, custom_comparator kernels cache too."""

    def __init__(self, base: GoldenProducer, custom_comparator: Callable, output_tensors: GoldenDict):
        self._base = base
        self._custom_comparator = custom_comparator
        self._output_tensors = output_tensors

    def output_spec(self) -> GoldenDict:
        return self._base.output_spec()

    def produce(self):
        return self._custom_comparator(self._base.produce(), self._output_tensors)
