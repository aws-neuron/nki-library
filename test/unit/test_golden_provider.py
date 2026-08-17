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
"""Unit tests for GoldenProvider — the caching decision is internal, so these
assert the observable contract: when caching applies vs. when compute is returned
unwrapped."""

from unittest.mock import MagicMock, patch

import numpy as np

from ..utils.golden_provider import CustomComparatorProducer, GoldenProvider, TorchRefProducer
from ..utils.torch_ref_cache import GoldenCache


def _ref(**kwargs):
    return {"out": np.ones((2,))}


class TestGoldenProvider:
    def test_no_cache_path_returns_compute_unwrapped(self):
        compute = MagicMock(return_value={"out": np.ones((2,))})
        provider = GoldenProvider(collector=MagicMock(), torch_ref_cache_path=None)
        assert provider.get(compute, _ref, {"x": 1}) is compute

    def test_empty_cache_path_returns_compute_unwrapped(self):
        compute = MagicMock()
        provider = GoldenProvider(collector=MagicMock(), torch_ref_cache_path="")
        assert provider.get(compute, _ref, {"x": 1}) is compute

    @patch("test.utils.golden_provider.ref_dependency_hash", return_value=None)
    def test_uncacheable_reference_returns_compute_unwrapped(self, _mock_dep):
        # dep_hash None (source unobtainable) -> caching would risk a stale golden,
        # so get() returns compute unchanged even with a cache path.
        compute = MagicMock()
        provider = GoldenProvider(collector=MagicMock(), torch_ref_cache_path="s3://b/p")
        assert provider.get(compute, _ref, {"x": 1}) is compute

    @patch("test.utils.golden_provider.ref_dependency_hash", return_value="DEPHASH")
    def test_cacheable_reference_serves_hit_without_computing(self, _mock_dep):
        # on a hit the provider returns the cached golden and never runs compute.
        with patch.object(GoldenCache, "load", return_value={"out": np.ones((2,))}) as mock_load:
            compute = MagicMock(return_value={"out": np.zeros((2,))})
            provider = GoldenProvider(collector=MagicMock(), torch_ref_cache_path="s3://b/p")
            wrapped = provider.get(compute, _ref, {"x": 1})
            assert wrapped is not compute  # cache path -> a new callable
            result = wrapped()

        compute.assert_not_called()
        assert np.array_equal(result["out"], np.ones((2,)))
        ref_qualname, dep_hash, inputs = mock_load.call_args.args
        assert ref_qualname.endswith("_ref")
        assert dep_hash == "DEPHASH"
        assert inputs == {"x": 1}

    @patch("test.utils.golden_provider.ref_dependency_hash", return_value="DEPHASH")
    def test_cacheable_reference_computes_and_stores_on_miss(self, _mock_dep):
        # on a miss the provider runs compute and stores the result under the key.
        with (
            patch.object(GoldenCache, "load", return_value=None),
            patch.object(GoldenCache, "store") as mock_store,
        ):
            compute = MagicMock(return_value={"out": np.zeros((2,))})
            provider = GoldenProvider(collector=MagicMock(), torch_ref_cache_path="s3://b/p")
            result = provider.get(compute, _ref, {"x": 1})()

        compute.assert_called_once()
        assert np.array_equal(result["out"], np.zeros((2,)))
        mock_store.assert_called_once()
        ref_qualname, dep_hash, inputs, golden = mock_store.call_args.args
        assert dep_hash == "DEPHASH"
        assert np.array_equal(golden["out"], np.zeros((2,)))

    @patch("test.utils.golden_provider.ref_dependency_hash", return_value="DEPHASH")
    def test_wrapper_options_folded_into_cache_key(self, _mock_dep):
        # A wrapper exposes its construction options; the provider folds them into the
        # hashed inputs so the key accounts for golden-affecting closure state.
        def ref(**kwargs):
            return {"out": np.ones((2,))}

        ref._torch_ref_cache_options = {"preserve_lower_precision": True}
        captured = []
        with (
            patch.object(GoldenCache, "load", return_value=None),
            patch.object(GoldenCache, "store", side_effect=lambda q, d, inp, g: captured.append(inp)),
        ):
            provider = GoldenProvider(collector=MagicMock(), torch_ref_cache_path="s3://b/p")
            provider.get(MagicMock(return_value={"out": np.ones((2,))}), ref, {"x": 1})()

        assert captured[0]["__torch_ref_cache_options__"] == {"preserve_lower_precision": True}

    @patch("test.utils.golden_provider.ref_dependency_hash", return_value="DEPHASH")
    def test_wrapper_options_disambiguate_variants(self, _mock_dep):
        # Two wrappers over the same reference + inputs but different golden-affecting
        # options (e.g. preserve_lower_precision) must key differently.
        def make_ref(preserve):
            def ref(**kwargs):
                return {"out": np.ones((2,))}

            ref._torch_ref_cache_options = {"preserve_lower_precision": preserve}
            return ref

        captured = []
        with (
            patch.object(GoldenCache, "load", return_value=None),
            patch.object(GoldenCache, "store", side_effect=lambda q, d, inp, g: captured.append(inp)),
        ):
            provider = GoldenProvider(collector=MagicMock(), torch_ref_cache_path="s3://b/p")
            provider.get(MagicMock(return_value={"out": np.ones((2,))}), make_ref(False), {"x": 1})()
            provider.get(MagicMock(return_value={"out": np.ones((2,))}), make_ref(True), {"x": 1})()

        assert len(captured) == 2
        # Same kwargs, but the folded options differ -> the hashed input dicts differ.
        assert captured[0] != captured[1]
        assert captured[0]["__torch_ref_cache_options__"] == {"preserve_lower_precision": False}
        assert captured[1]["__torch_ref_cache_options__"] == {"preserve_lower_precision": True}


class TestTorchRefProducer:
    """TorchRefProducer runs the reference through GoldenProvider: output_spec is the
    pre-golden placeholder dict; produce() returns the (possibly cached) golden."""

    def test_output_spec_returns_placeholders_without_producing(self):
        compute = MagicMock(return_value={"out": np.ones((2,))})
        spec = {"out": np.zeros((2,), np.float32)}
        producer = TorchRefProducer(compute, _ref, {"x": 1}, spec, MagicMock(), None)
        assert producer.output_spec() is spec
        compute.assert_not_called()  # output_spec must not trigger the golden

    def test_produce_runs_reference_uncached(self):
        compute = MagicMock(return_value={"out": np.full((2,), 5.0)})
        producer = TorchRefProducer(compute, _ref, {"x": 1}, {"out": np.zeros((2,))}, MagicMock(), None)
        result = producer.produce()
        compute.assert_called_once()
        assert np.array_equal(result["out"], np.full((2,), 5.0))

    @patch("test.utils.golden_provider.ref_dependency_hash", return_value="DEP")
    def test_produce_caches_cold_miss_then_warm_hit(self, _mock_dep):
        spec = {"out": np.zeros((2,), np.float32)}
        # cold: load miss -> compute + store
        with (
            patch.object(GoldenCache, "load", return_value=None),
            patch.object(GoldenCache, "store") as mock_store,
        ):
            compute = MagicMock(return_value={"out": np.arange(2, dtype=np.float32)})
            cold = TorchRefProducer(compute, _ref, {"x": 1}, spec, MagicMock(), "s3://b/p").produce()
            compute.assert_called_once()  # miss -> computed
            mock_store.assert_called_once()  # miss -> stored
            assert np.array_equal(cold["out"], np.arange(2, dtype=np.float32))
        # warm: load hit -> compute never runs
        with patch.object(GoldenCache, "load", return_value={"out": np.arange(2, dtype=np.float32)}):
            warm_compute = MagicMock(return_value={"out": np.full((2,), 99.0)})
            warm = TorchRefProducer(warm_compute, _ref, {"x": 1}, spec, MagicMock(), "s3://b/p").produce()
            warm_compute.assert_not_called()  # hit -> served from cache
            assert np.array_equal(warm["out"], np.arange(2, dtype=np.float32))


class TestCustomComparatorProducer:
    """CustomComparatorProducer applies a custom_comparator to a base producer's golden;
    output_spec delegates to the base. Because the base produces (and caches) the
    golden first, custom_comparator kernels get the cached golden too."""

    def test_applies_comparator_to_base_golden(self):
        base_golden = {"out": np.arange(3, dtype=np.float32)}
        base = MagicMock(spec=TorchRefProducer)
        base.produce.return_value = base_golden
        base.output_spec.return_value = {"out": np.zeros((3,))}
        received = {}

        def comparator(golden, output_tensors):
            received["golden"], received["outs"] = golden, output_tensors
            return {"out": "validator-object"}

        outs = {"out": np.zeros((3,))}
        producer = CustomComparatorProducer(base, comparator, outs)
        assert producer.output_spec() is base.output_spec.return_value  # delegates to base
        assert producer.produce() == {"out": "validator-object"}
        assert received["golden"] is base_golden  # comparator got the base golden
        assert received["outs"] is outs

    @patch("test.utils.golden_provider.ref_dependency_hash", return_value="DEP")
    def test_custom_comparator_golden_is_served_from_cache(self, _mock_dep):
        # The key benefit: wrapping a caching TorchRefProducer means the golden fed to
        # custom_comparator comes from the cache (base compute never runs on a hit).
        spec = {"out": np.ones((2,), np.float32)}
        seen = {}

        def comparator(golden, _outs):
            seen["golden"] = golden
            return {"out": "v"}

        with patch.object(GoldenCache, "load", return_value={"out": np.ones((2,))}):
            compute = MagicMock(return_value={"out": np.zeros((2,))})
            base = TorchRefProducer(compute, _ref, {"x": 1}, spec, MagicMock(), "s3://b/p")
            out = CustomComparatorProducer(base, comparator, spec).produce()

        compute.assert_not_called()  # golden served from cache, not recomputed
        assert out == {"out": "v"}
        assert np.array_equal(seen["golden"]["out"], np.ones((2,)))
