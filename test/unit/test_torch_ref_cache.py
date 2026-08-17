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
"""Unit tests for the torch-ref golden cache and its invalidation logic.

S3 is mocked (no network); invalidation uses throwaway on-disk packages so the
real AST-walk / file-read code paths are exercised without touching real source.
"""

import ast
import enum
import glob
import importlib
import os
import re
import sys
import tempfile
import textwrap
from unittest.mock import MagicMock, patch

import ml_dtypes
import numpy as np
import pytest
import torch

from ..utils import torch_ref_cache as trc
from ..utils import torch_ref_invalidation as inval
from ..utils.golden_provider import CachedTorchRef


# ===========================================================================
# Invalidation: input hashing
# ===========================================================================
class TestInputHash:
    def test_same_inputs_same_hash(self):
        a = {"x": np.arange(12, dtype=np.float32).reshape(3, 4), "k": 2}
        b = {"x": np.arange(12, dtype=np.float32).reshape(3, 4), "k": 2}
        assert trc._input_hash(a) == trc._input_hash(b)

    def test_different_array_bytes_different_hash(self):
        a = {"x": np.zeros((4,), np.float32)}
        b = {"x": np.ones((4,), np.float32)}
        assert trc._input_hash(a) != trc._input_hash(b)

    def test_different_scalar_different_hash(self):
        assert trc._input_hash({"k": 2}) != trc._input_hash({"k": 3})

    def test_key_order_independent(self):
        assert trc._input_hash({"a": 1, "b": 2}) == trc._input_hash({"b": 2, "a": 1})

    def test_custom_x4_dtype_hashed_by_bytes(self):
        # raw-byte view must work even for dtypes that np.array_equal can't compare
        x = np.frombuffer(b"\x01\x02\x03\x04", dtype=np.uint8)
        assert trc._input_hash({"w": x}) is not None

    def test_data_holder_object_hashed(self):
        class SkipMode:
            def __init__(self, t, w):
                self.skip_token = t
                self.skip_weight = w

        h1 = trc._input_hash({"s": SkipMode(True, False)})
        h2 = trc._input_hash({"s": SkipMode(True, False)})
        h3 = trc._input_hash({"s": SkipMode(False, False)})
        assert h1 == h2 and h1 != h3

    def test_enum_hashed_by_value(self):
        class Quant(enum.Enum):
            NONE = 0
            MX = 3

        assert trc._input_hash({"q": Quant.MX}) == trc._input_hash({"q": Quant.MX})
        assert trc._input_hash({"q": Quant.MX}) != trc._input_hash({"q": Quant.NONE})

    def test_non_enum_with_value_attr_not_treated_as_enum(self):
        # a plain object that merely has a `.value` must route through __dict__,
        # not the enum branch (the old hasattr-based check mis-caught these).
        class HasValue:
            def __init__(self, v, extra):
                self.value = v
                self.extra = extra

        # if it were mis-hashed as an enum, `extra` would be ignored and these two
        # would collide; via __dict__ they must differ.
        assert trc._input_hash({"o": HasValue(1, "a")}) != trc._input_hash({"o": HasValue(1, "b")})

    def test_tuple_and_list_fingerprintable(self):
        # config values like tile-shape tuples must be fingerprintable, else the
        # whole input is unfingerprintable and the kernel never caches.
        assert trc._input_hash({"shape": (128, 256)}) is not None
        assert trc._input_hash({"dists": ["normal", "normal"]}) is not None
        # same content -> same hash; different -> different
        assert trc._input_hash({"s": (1, 2)}) == trc._input_hash({"s": (1, 2)})
        assert trc._input_hash({"s": (1, 2)}) != trc._input_hash({"s": (1, 3)})
        # order matters; nesting works
        assert trc._input_hash({"s": (1, 2)}) != trc._input_hash({"s": (2, 1)})
        assert trc._input_hash({"s": [(1, 2), (3, 4)]}) is not None

    def test_unfingerprintable_returns_none(self):
        assert trc._input_hash({"fn": lambda: 1}) is None

    def test_dict_kwarg_fingerprintable(self):
        # a dict-valued kwarg (e.g. a config) must be fingerprintable, else the
        # whole input is unfingerprintable and the kernel silently never caches.
        assert trc._input_hash({"cfg": {"a": 1, "b": 2}}) is not None
        # insertion order of the inner dict must not change the hash
        assert trc._input_hash({"cfg": {"a": 1, "b": 2}}) == trc._input_hash({"cfg": {"b": 2, "a": 1}})
        # value change -> different; nested arrays work
        assert trc._input_hash({"cfg": {"a": 1}}) != trc._input_hash({"cfg": {"a": 2}})
        assert trc._input_hash({"cfg": {"w": np.zeros((4,), np.float32)}}) is not None
        # an unfingerprintable value inside the dict propagates None
        assert trc._input_hash({"cfg": {"fn": lambda: 1}}) is None

    def test_torch_ref_wrapper_options_disambiguate_and_fail_closed(self):
        # torch_ref_wrapper exposes its golden-affecting options; the cache folds them
        # into the input hash. Fingerprintable options (preserve_lower_precision) must
        # yield distinct hashes; a converter callable makes the whole input
        # unfingerprintable (None) so the cache declines rather than serve a wrong hit.
        from nkilib_src.nkilib.core.utils.torch_ref_wrapper import torch_ref_wrapper

        def _ref(input_a):
            return input_a

        def _options(*args, **kwargs) -> dict:
            wrapped = torch_ref_wrapper(*args, **kwargs)
            assert isinstance(wrapped, CachedTorchRef), "torch_ref_wrapper must publish its cache options"
            return wrapped._torch_ref_cache_options

        opts_default = _options(_ref)
        opts_preserve = _options(_ref, preserve_lower_precision=True)
        opts_converter = _options(_ref, output_dtype_converter=lambda t: t.numpy())

        h_default = trc._input_hash({"__torch_ref_cache_options__": opts_default})
        h_preserve = trc._input_hash({"__torch_ref_cache_options__": opts_preserve})
        assert h_default is not None and h_preserve is not None
        assert h_default != h_preserve
        assert trc._input_hash({"__torch_ref_cache_options__": opts_converter}) is None


# ===========================================================================
# Invalidation: S3 key layout
# ===========================================================================
class TestS3Key:
    def test_layout_has_all_segments(self):
        key = trc._s3_key("pre", "VERH", "ref", "DEPH", "INPH")
        assert key == "pre/VERH/ref/DEPH/INPH/golden.bin"

    def test_no_prefix(self):
        assert trc._s3_key("", "VERH", "ref", "DEPH", "INPH") == "VERH/ref/DEPH/INPH/golden.bin"

    def test_version_change_changes_key(self):
        k1 = trc._s3_key("p", "V1", "ref", "D", "I")
        k2 = trc._s3_key("p", "V2", "ref", "D", "I")
        assert k1 != k2

    def test_dep_change_changes_key(self):
        assert trc._s3_key("p", "V", "ref", "D1", "I") != trc._s3_key("p", "V", "ref", "D2", "I")


# ===========================================================================
# Invalidation: transitive dependency hash (real AST walk + file edit)
# ===========================================================================
class TestDependencyHash:
    def _make_pkg(self, tmp, leaf_body):
        pkg = "_trc_probe_pkg"
        pkg_dir = os.path.join(tmp, pkg)
        os.makedirs(pkg_dir, exist_ok=True)
        open(os.path.join(pkg_dir, "__init__.py"), "w").close()
        with open(os.path.join(pkg_dir, "leaf.py"), "w") as f:
            f.write(leaf_body)
        with open(os.path.join(pkg_dir, "root.py"), "w") as f:
            f.write(
                textwrap.dedent(f"""
                from {pkg}.leaf import helper
                def ref(**kw):
                    return {{"out": helper(kw.get('x', 0))}}
            """)
            )
        return pkg

    def test_reaches_transitive_helper_and_invalidates_on_edit(self, monkeypatch):
        tmp = tempfile.mkdtemp(prefix="trc_inval_")
        pkg = self._make_pkg(tmp, "def helper(x):\n    return x + 1\n")
        sys.path.insert(0, tmp)
        monkeypatch.setattr(inval, "_PKG_PREFIXES", (pkg,))
        try:
            mod = importlib.import_module(f"{pkg}.root")
            reached = inval._transitive_module_files(f"{pkg}.root")
            assert f"{pkg}.leaf" in reached  # walk recursed into the helper

            inval.dependency_hash.cache_clear()
            h1 = inval.ref_dependency_hash(mod.ref)

            # edit the transitive helper on disk
            with open(os.path.join(tmp, pkg, "leaf.py"), "w") as f:
                f.write("def helper(x):\n    return x + 999\n")
            inval.dependency_hash.cache_clear()
            h2 = inval.ref_dependency_hash(mod.ref)

            assert h1 != h2, "helper edit must change dep hash (else stale golden)"
            assert h1 is not None, "a resolvable in-package module must hash"
            assert len(h1) == 16
        finally:
            sys.path.remove(tmp)
            for m in list(sys.modules):
                if m.startswith(pkg):
                    del sys.modules[m]
            inval.dependency_hash.cache_clear()

    def test_sourceless_ref_returns_none(self):
        # A builtin has no module .py file and no obtainable source, so we cannot
        # produce a hash that tracks its content — must signal uncacheable (None),
        # not a stable placeholder that would freeze the hash.
        assert inval.ref_dependency_hash(len) is None

    def test_unresolvable_module_ref_declines(self):
        # A ref whose module can't be resolved to a .py file is fail-closed to
        # uncacheable (None): a body-only hash would miss transitive helper edits
        # AND skip the purity scan, risking a stale golden. Declining is safe.
        def ref():  # pragma: no cover - not executed, only its (unresolvable) module matters
            return {}

        ref.__module__ = "no.such.module.that.exists"
        assert inval.ref_dependency_hash(ref) is None

    def test_unreadable_module_source_returns_none(self, monkeypatch):
        # If a reached module's source can't be read, dependency_hash returns None
        # rather than hashing a placeholder (which would freeze the hash).
        tmp = tempfile.mkdtemp(prefix="trc_unread_")
        pkg = self._make_pkg(tmp, "def helper(x):\n    return x + 1\n")
        sys.path.insert(0, tmp)
        monkeypatch.setattr(inval, "_PKG_PREFIXES", (pkg,))
        try:
            importlib.import_module(f"{pkg}.root")
            inval.dependency_hash.cache_clear()

            real_open = open

            def raising_open(path, *a, **k):
                if str(path).endswith("leaf.py"):
                    raise OSError("unreadable")
                return real_open(path, *a, **k)

            monkeypatch.setattr("builtins.open", raising_open)
            assert inval.dependency_hash(f"{pkg}.root") is None
        finally:
            sys.path.remove(tmp)
            for m in list(sys.modules):
                if m.startswith(pkg):
                    del sys.modules[m]
            inval.dependency_hash.cache_clear()


class TestIsNumericSource:
    """_is_numeric_source decides which modules the purity scan runs on. A prefix
    collision here would either falsely decline (numeric module missed) or, worse,
    falsely skip the scan for a numeric module — so the exact/dotted-boundary match
    matters."""

    def test_exact_package_names(self):
        assert inval._is_numeric_source("nkilib_src")
        assert inval._is_numeric_source("nkilib")

    def test_submodules(self):
        assert inval._is_numeric_source("nkilib_src.nkilib.core.moe.moe_cte.moe_cte_torch")
        assert inval._is_numeric_source("nkilib.core.attention.attention_torch")

    def test_test_modules_are_not_numeric_source(self):
        assert not inval._is_numeric_source("test.integration.nkilib.core.moe.test_moe_cte_common")
        assert not inval._is_numeric_source("test.utils.common_dataclasses")

    def test_prefix_collisions_rejected(self):
        # Must match only on the dotted boundary, not a bare string prefix. The
        # nkilib_testing runtime package is the real-world trap the "." guards against.
        assert not inval._is_numeric_source("nkilibtesting")
        assert not inval._is_numeric_source("nkilib_testing.foo")
        assert not inval._is_numeric_source("nkilib_src_extra")
        assert not inval._is_numeric_source("nkilibextra")
        assert not inval._is_numeric_source("")
        assert not inval._is_numeric_source("torch")


class TestPurityGuard:
    """Static purity scan (_purity_violation): decline to cache a reference whose
    transitive source imports an unpinned package, imports dynamically, or reads a
    file — cases the cache key can't capture (contract points 1-3). Point 4 (RNG /
    env / clock / global state) is out of scope here."""

    def _scan(self, src: str, module: str = "nkilib_src.probe") -> str | None:
        # module defaults to a numeric-source name so the unpinned-import rule (which
        # applies only to numeric source) is exercised. Pass a test.* name to check
        # the test-scaffolding relaxation.
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            path = f.name
        try:
            return inval._purity_violation({module: path})
        finally:
            os.unlink(path)

    def _assert_declined(self, src: str, expected_reason: str) -> None:
        violation = self._scan(src)
        assert violation is not None, "expected the purity scan to decline this source, but it was accepted"
        assert expected_reason in violation, f"expected reason {expected_reason!r} in {violation!r}"

    def test_clean_ref_is_cacheable(self):
        # in-package + pinned (torch/numpy) + stdlib imports are all allowed.
        assert self._scan("import torch\nimport numpy as np\nimport math\ndef ref():\n    return {}\n") is None

    def test_pinned_neuronxcc_and_nki_allowed(self):
        # MX refs import these; they are version-pinned, so cacheable.
        src = "import nki.language as nl\nfrom neuronxcc.nki._private.test import mx_util\ndef ref():\n    return {}\n"
        assert self._scan(src) is None

    def test_test_module_scaffolding_not_scanned(self):
        # A torch-ref can live in / wrap through a test module that pulls in the test
        # framework: unpinned imports (pytest, fabric2, ...) and file reads (model-config
        # loaders). None of that computes the golden, so a test module is not purity-
        # scanned and must not decline caching. The numeric golden code lives in
        # nkilib_src, which IS scanned.
        src = (
            "import pytest\n"
            "from test.utils.common_dataclasses import CustomValidator\n"
            "def ref():\n"
            "    open('model_config.json')\n"
            "    return {}\n"
        )
        assert self._scan(src, module="test.integration.foo.test_bar_common") is None

    def test_unpinned_import_declines(self):
        self._assert_declined("import pandas\ndef ref():\n    return {}\n", "unpinned")

    def test_unpinned_from_import_declines(self):
        self._assert_declined("from sklearn import metrics\ndef ref():\n    return {}\n", "unpinned")

    def test_dynamic_import_declines(self):
        src = "import importlib\ndef ref():\n    importlib.import_module('x')\n    return {}\n"
        self._assert_declined(src, "dynamic import")

    def test_open_declines(self):
        self._assert_declined("def ref():\n    open('t.bin')\n    return {}\n", "reads a file")

    def test_np_load_and_fromfile_decline(self):
        self._assert_declined("import numpy as np\ndef ref():\n    return np.load('x.npy')\n", "reads a file")
        self._assert_declined("import numpy as np\ndef ref():\n    return np.fromfile('x')\n", "reads a file")

    def test_all_real_refs_remain_cacheable(self):
        # Guard against a too-strict allowlist: every shipped torch-ref module must
        # still pass the single-file purity scan (no false decline of the population).
        declined = []
        for path in glob.glob("src/nkilib_src/nkilib/**/*_torch.py", recursive=True):
            modname = path.replace("src/", "").replace("/", ".")[:-3]
            if inval._purity_violation({modname: path}) is not None:
                declined.append(path)
        assert not declined, f"purity guard falsely declined real refs: {declined}"

    def test_torch_ref_test_modules_import_only_known_safe_top_level(self):
        # The scan skips test.* modules, so allowlist what a ref-defining test file may
        # import: in-package, version-pinned, stdlib, or known test-infra. Any other
        # top-level import fails closed, forcing a decision (pin it, or move the numeric
        # code to nkilib_src) — a new numeric lib can't silently slip past the skip.
        test_infra_ok = {"pytest", "typing_extensions"}
        allowed = inval._PINNED_TOP_LEVEL | test_infra_ok | set(sys.stdlib_module_names)
        defines_ref = re.compile(r"def\s+\w*(torch_ref|torch_wrapper)\w*\s*\(")
        offenders = {}
        for path in glob.glob("test/**/*.py", recursive=True):
            src = open(path, encoding="utf-8").read()
            if not defines_ref.search(src):
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    tops = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    tops = [node.module.split(".")[0]]
                else:
                    continue
                for top in tops:
                    if not (inval._in_pkg(top) or top in allowed):
                        offenders.setdefault(path, set()).add(top)
        assert not offenders, f"ref-defining test files import non-allowlisted top-level packages: {offenders}"


# ===========================================================================
# Invalidation: third-party version hash
# ===========================================================================
class TestVersionHash:
    def test_stable_and_nonempty(self):
        inval.dependency_version_hash.cache_clear()
        h1 = inval.dependency_version_hash()
        h2 = inval.dependency_version_hash()
        assert h1 == h2 and len(h1) == 12

    def test_changes_when_lib_version_changes(self, monkeypatch):
        inval.dependency_version_hash.cache_clear()
        base = inval.dependency_version_hash()

        real_import = importlib.import_module

        def fake_import(name):
            m = real_import(name)
            if name == "numpy":
                fake = MagicMock()
                fake.__version__ = "9.9.9-bumped"
                return fake
            return m

        monkeypatch.setattr(inval.importlib, "import_module", fake_import)
        inval.dependency_version_hash.cache_clear()
        bumped = inval.dependency_version_hash()
        inval.dependency_version_hash.cache_clear()
        assert base != bumped, "numpy version bump must change the version hash"

    def test_changes_when_python_minor_version_changes(self, monkeypatch):
        # A Python interpreter minor bump must re-namespace the cache: hash()/dict
        # ordering and RNG-adjacent semantics can shift across versions.
        inval.dependency_version_hash.cache_clear()
        base = inval.dependency_version_hash()

        faked = (sys.version_info[0], sys.version_info[1] + 1, 0)
        monkeypatch.setattr(inval.sys, "version_info", faked)
        inval.dependency_version_hash.cache_clear()
        bumped = inval.dependency_version_hash()
        inval.dependency_version_hash.cache_clear()
        assert base != bumped, "Python minor version bump must change the version hash"


# ===========================================================================
# Golden serialization round-trip (raw bytes + dtype manifest)
# ===========================================================================
class TestGoldenSerialization:
    def _roundtrip(self, golden):
        out = trc._deserialize_golden(trc._serialize_golden(golden))
        assert set(out) == set(golden)
        for k, a in golden.items():
            assert out[k].dtype == a.dtype
            assert out[k].shape == a.shape
            # a cache-hit golden must be writable, matching a freshly computed
            # (miss) one — np.frombuffer over immutable bytes is read-only, so the
            # deserializer must copy. Downstream in-place mutation must not depend
            # on hit vs miss.
            assert out[k].flags.writeable
            # reshape(-1) before the uint8 view: a 0-d array can't be re-dtyped directly
            assert (
                out[k].reshape(-1).view(np.uint8).tobytes()
                == np.ascontiguousarray(a).reshape(-1).view(np.uint8).tobytes()
            )

    def test_native_dtypes(self):
        self._roundtrip(
            {
                "f32": np.arange(6, dtype=np.float32).reshape(2, 3),
                "i64": np.array([[-1, 2], [3, 4]], dtype=np.int64),
                "u8": np.arange(4, dtype=np.uint8),
            }
        )

    def test_ml_dtypes(self):
        self._roundtrip(
            {
                "bf16": np.arange(6, dtype=ml_dtypes.bfloat16).reshape(3, 2),
                "fp8": np.arange(4, dtype=ml_dtypes.float8_e4m3fn),
            }
        )

    def test_scalar_and_empty_shapes(self):
        self._roundtrip({"scalar": np.array(3.5, dtype=np.float32), "empty": np.zeros((0,), np.float32)})

    def test_deserialized_golden_is_writable_in_place(self):
        # A cache hit deserializes the golden; it must accept in-place mutation
        # (rescale, nan_to_num, byteswap in a custom comparator) exactly like a
        # freshly computed miss. Without a copy, np.frombuffer's read-only array
        # would raise "assignment destination is read-only" only once a key is warm.
        out = trc._deserialize_golden(trc._serialize_golden({"out": np.arange(6, dtype=np.float32)}))
        out["out"][0] = 42.0  # must not raise
        assert out["out"][0] == 42.0

    def test_resolve_dtype_unknown_raises(self):
        with pytest.raises(TypeError):
            trc._resolve_dtype("definitely_not_a_dtype")


# ===========================================================================
# Cache: lookup / store with mocked S3
# ===========================================================================
def _golden_bytes(d):
    # serialize a golden the way the cache stores it (raw bytes + manifest)
    return trc._serialize_golden(d)


class TestGoldenCacheLoad:
    """GoldenCache.load(): serve a stored golden, or return None (never raise) on
    miss / uncacheable input / any error. The load/compute/store orchestration
    that sits on top lives in GoldenProvider and is tested there — these exercise
    the passive cache directly. The S3 client is injected (no module patching)."""

    REF = "myref"
    DEP = "deadbeefdeadbeef"
    CACHE = "s3://bucket/prefix"

    def _cache(self, collector=None, s3=None, cache_path=None):
        return trc.GoldenCache(cache_path or self.CACHE, collector or MagicMock(), s3 or MagicMock())

    def _load(self, cache, inputs=None):
        return cache.load(self.REF, self.DEP, {"x": np.zeros((4,), np.float32)} if inputs is None else inputs)

    @patch("test.utils.torch_ref_cache.dependency_version_hash", return_value="VERHASH")
    def test_hit_returns_golden_and_records_hit(self, _mock_ver):
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: _golden_bytes({"out": np.ones((2,))}))}
        collector = MagicMock()
        result = self._load(self._cache(collector, s3=mock_s3))
        assert np.array_equal(result["out"], np.ones((2,)))
        collector.record_metric.assert_any_call("TorchRefCacheHit", 1, "Count")

    @patch("test.utils.torch_ref_cache.dependency_version_hash", return_value="VERHASH")
    def test_miss_returns_none_and_records_miss(self, _mock_ver):
        mock_s3 = MagicMock()
        mock_s3.get_object.side_effect = Exception("NoSuchKey")
        collector = MagicMock()
        assert self._load(self._cache(collector, s3=mock_s3)) is None
        collector.record_metric.assert_any_call("TorchRefCacheHit", 0, "Count")

    @patch("test.utils.torch_ref_cache.dependency_version_hash", return_value="VERHASH")
    def test_unfingerprintable_input_returns_none_without_s3(self, _mock_ver):
        mock_s3 = MagicMock()
        assert self._load(self._cache(s3=mock_s3), inputs={"fn": lambda: 1}) is None  # not fingerprintable
        mock_s3.get_object.assert_not_called()

    @patch("test.utils.torch_ref_cache.dependency_version_hash", return_value="VERHASH")
    def test_object_dtype_input_returns_none_without_raising(self, _mock_ver):
        # object-dtype ndarray makes _input_hash raise (.view(np.uint8) fails);
        # load must catch it and treat it as a miss, never propagate.
        mock_s3 = MagicMock()
        assert self._load(self._cache(s3=mock_s3), inputs={"x": np.array([object()], dtype=object)}) is None
        mock_s3.get_object.assert_not_called()

    @patch("test.utils.torch_ref_cache.dependency_version_hash", return_value="VERHASH")
    def test_malformed_cache_path_returns_none_without_raising(self, _mock_ver):
        # parse_s3_uri raising on a malformed URI must fail open, not crash.
        cache = self._cache(cache_path="not-an-s3-uri")
        assert cache.load(self.REF, self.DEP, {"x": np.zeros((4,), np.float32)}) is None


@patch("test.utils.torch_ref_cache.dependency_version_hash", return_value="VERHASH")
class TestGoldenCacheStore:
    """GoldenCache.store(): PUT a golden (best-effort, never raises), respecting
    the key layout, the size cap, and round-trip fidelity. S3 client is injected."""

    REF = "myref"
    DEP = "deadbeefdeadbeef"
    CACHE = "s3://bucket/prefix"

    def _store(self, mock_s3, golden, collector=None, inputs=None):
        cache = trc.GoldenCache(self.CACHE, collector or MagicMock(), mock_s3)
        cache.store(self.REF, self.DEP, {"x": np.zeros((4,), np.float32)} if inputs is None else inputs, golden)

    def test_stores_with_versioned_key(self, _ver):
        mock_s3 = MagicMock()
        self._store(mock_s3, {"out": np.ones((3,))})
        mock_s3.put_object.assert_called_once()
        key = mock_s3.put_object.call_args[1]["Key"]
        assert key == "prefix/VERHASH/myref/deadbeefdeadbeef/" + key.split("/")[-2] + "/golden.bin"

    def test_put_failure_does_not_raise(self, _ver):
        mock_s3 = MagicMock()
        mock_s3.put_object.side_effect = Exception("AccessDenied")
        self._store(mock_s3, {"out": np.ones((3,))})  # must not raise (best-effort)

    def test_oversize_golden_not_stored(self, _ver):
        mock_s3 = MagicMock()
        with patch.object(trc, "MAX_GOLDEN_BYTES", 1):  # force over-cap
            self._store(mock_s3, {"out": np.ones((100,))})
        mock_s3.put_object.assert_not_called()

    def test_unfingerprintable_input_not_stored(self, _ver):
        mock_s3 = MagicMock()
        self._store(mock_s3, {"out": np.ones((2,))}, inputs={"fn": lambda: 1})  # unhashable key
        mock_s3.put_object.assert_not_called()

    def test_grad_tensor_input_stores_normally(self, _ver):
        # grad-requiring tensor input must hash (via detach/cpu), not raise from a
        # bare .numpy() call.
        mock_s3 = MagicMock()
        self._store(mock_s3, {"out": np.ones((3,))}, inputs={"x": torch.ones(4, requires_grad=True)})
        mock_s3.put_object.assert_called_once()

    def test_float32_golden_round_trips(self, _ver):
        # the real MX CTE golden is float32 — what we store must reload identically.
        mock_s3 = MagicMock()
        golden = np.arange(6, dtype=np.float32).reshape(2, 3)
        self._store(mock_s3, {"out": golden})
        reloaded = trc._deserialize_golden(mock_s3.put_object.call_args[1]["Body"])["out"]
        assert reloaded.dtype == np.float32
        assert np.array_equal(reloaded, golden)

    def test_ml_dtypes_golden_round_trips(self, _ver):
        # raw-bytes + dtype-manifest format preserves ml_dtypes extension dtypes
        # (bfloat16) that np.savez cannot — stored bytes must reload bit-identical.
        mock_s3 = MagicMock()
        bf16_golden = np.arange(6, dtype=ml_dtypes.bfloat16).reshape(2, 3)
        self._store(mock_s3, {"out": bf16_golden})
        mock_s3.put_object.assert_called_once()
        reloaded = trc._deserialize_golden(mock_s3.put_object.call_args[1]["Body"])["out"]
        assert reloaded.dtype == np.dtype(ml_dtypes.bfloat16)
        assert reloaded.shape == (2, 3)
        assert reloaded.view(np.uint8).tobytes() == bf16_golden.view(np.uint8).tobytes()


def _timer_names(collector):
    """Timer metric names the cache recorded — via the timer() context manager
    (the current path) or a direct record_timer() call (legacy). On a MagicMock
    collector, timer() is the observable call; a real collector routes timer()
    through record_timer(), so checking both covers either collector type."""
    return [c.args[0] for c in collector.timer.call_args_list] + [
        c.args[0] for c in collector.record_timer.call_args_list
    ]


@patch("test.utils.torch_ref_cache.dependency_version_hash", return_value="VERHASH")
class TestOverheadMetrics:
    """The cache must record its own overhead so a cheap golden can be checked
    for whether caching actually pays off vs. just recomputing."""

    REF = "myref"
    DEP = "deadbeefdeadbeef"
    CACHE = "s3://bucket/prefix"

    def test_load_hit_records_lookup_time(self, _ver):
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {"Body": MagicMock(read=lambda: _golden_bytes({"out": np.ones((2,))}))}
        collector = MagicMock()
        trc.GoldenCache(self.CACHE, collector, mock_s3).load(self.REF, self.DEP, {"x": np.zeros((4,), np.float32)})
        assert "TorchRefCacheLookupTime" in _timer_names(collector)

    def test_store_records_store_time(self, _ver):
        collector = MagicMock()
        trc.GoldenCache(self.CACHE, collector, MagicMock()).store(
            self.REF, self.DEP, {"x": np.zeros((4,), np.float32)}, {"out": np.ones((3,))}
        )
        assert "TorchRefCacheStoreTime" in _timer_names(collector)

    def test_load_unfingerprintable_still_records_lookup_overhead(self, _ver):
        # the input-hashing work happened even though we bail before S3 — its cost
        # must be visible, else cheap-golden overhead looks free.
        collector = MagicMock()
        trc.GoldenCache(self.CACHE, collector, MagicMock()).load(self.REF, self.DEP, {"fn": lambda: 1})
        assert "TorchRefCacheLookupTime" in _timer_names(collector)

    def test_load_malformed_uri_still_records_lookup_overhead(self, _ver):
        collector = MagicMock()
        trc.GoldenCache("not-an-s3-uri", collector, MagicMock()).load(
            self.REF, self.DEP, {"x": np.zeros((4,), np.float32)}
        )
        assert "TorchRefCacheLookupTime" in _timer_names(collector)
