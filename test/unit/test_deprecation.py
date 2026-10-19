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
"""Unit tests for the nkilib @deprecated / @deprecated_unshipped markers.

The markers warn at import / decoration time (not on use) and return the decorated
object unchanged, so an @nki.jit kernel stays compilable by both the tracer and the
parser frontends.
"""

import tempfile
import warnings

import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib._deprecation import deprecated, deprecated_unshipped

from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    NKICompilationMode,
    Platforms,
    TraceMode,
)
from test.utils.kernel_tracer import trace_kernel
from test.utils.metrics_collector import NoopMetricsCollector


class TestDeprecated:
    """@deprecated — shipped API; `since` is required; warns at decoration time."""

    def test_decoration_warns_and_leaves_callable_working(self):
        with pytest.warns(DeprecationWarning) as record:

            @deprecated(since="2.33", use_instead="mlp_v2")
            def mlp(a, b=1):
                return a + b

        msg = str(record[0].message)
        assert "2.33" in msg
        assert "mlp_v2" in msg
        # Object is returned unchanged: still works, and calling it does not warn.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert mlp(2, 3) == 5

    def test_metadata_recorded(self):
        with pytest.warns(DeprecationWarning):

            @deprecated(since="2.33", use_instead="mlp_v2")
            def mlp():
                return None

        assert mlp.__dict__["__nki_deprecated__"] == {"since": "2.33", "use_instead": "mlp_v2"}
        assert mlp.__name__ == "mlp"  # object unchanged, identity preserved

    def test_since_is_required(self):
        # `since` has no default; omitting it is a TypeError (before any warning).
        kwargs = {"use_instead": "mlp_v2"}
        with pytest.raises(TypeError):
            deprecated(**kwargs)

    @pytest.mark.parametrize("since", ["next", "2.x", "v2", "", "2", "2.33.2.1"])
    def test_invalid_since_rejected(self, since):
        with pytest.raises(ValueError):
            deprecated(since=since)

    @pytest.mark.parametrize("since", ["2.33", "2.33.2"])
    def test_valid_since_accepted(self, since):
        # MAJOR.MINOR and MAJOR.MINOR.PATCH (patch release) are both valid.
        assert callable(deprecated(since=since, use_instead="mlp_v2"))

    def test_class_decoration_warns_not_instantiation(self):
        with pytest.warns(DeprecationWarning):

            @deprecated(since="2.33", use_instead="Foo2")
            class Foo:
                def __init__(self, x):
                    self.x = x

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            f = Foo(9)  # instantiation does not warn
        assert f.x == 9
        assert Foo.__dict__["__nki_deprecated__"]["use_instead"] == "Foo2"


class TestDeprecatedUnshipped:
    """@deprecated_unshipped — tight loop; no release to record."""

    def test_decoration_warns_with_downstream_phrasing(self):
        with pytest.warns(DeprecationWarning) as record:

            @deprecated_unshipped(use_instead="new_kernel")
            def old():
                return 7

        msg = str(record[0].message)
        assert "downstream" in msg
        assert "since release" not in msg

    def test_metadata_has_no_since(self):
        with pytest.warns(DeprecationWarning):

            @deprecated_unshipped(use_instead="new_kernel")
            def old():
                return None

        assert old.__dict__["__nki_deprecated__"] == {"since": None, "use_instead": "new_kernel"}


# --- @nki.jit compatibility -------------------------------------------------
# Regression guard: applying either decorator to an @nki.jit kernel must not break
# compilation under EITHER frontend. Because the decorators return the kernel
# unchanged, both the parser and the tracer compile the real kernel. The import-time
# DeprecationWarning is suppressed here so it doesn't pollute test output.

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)

    @deprecated(since="2.33", use_instead="noop_v2")
    @nki.jit
    def _shipped_noop(input_a):
        out = nl.ndarray(input_a.shape, dtype=input_a.dtype, buffer=nl.shared_hbm)
        nl.store(out, nl.load(input_a))
        return out

    @deprecated_unshipped(use_instead="noop_v2")
    @nki.jit
    def _unshipped_noop(input_a):
        out = nl.ndarray(input_a.shape, dtype=input_a.dtype, buffer=nl.shared_hbm)
        nl.store(out, nl.load(input_a))
        return out


class TestDecoratedNkiKernelCompiles:
    @pytest.mark.parametrize("frontend", [NKICompilationMode.parser, NKICompilationMode.tracer])
    @pytest.mark.parametrize("kernel_func", [_shipped_noop, _unshipped_noop])
    def test_decorated_kernel_traces(self, kernel_func, frontend):
        input_a = np.zeros((1, 8), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_kernel(
                KernelArgs(
                    kernel_func=kernel_func,
                    compiler_input=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=1),
                    kernel_input={"input_a": input_a},
                ),
                mode=TraceMode.TraceOnly,
                output_directory=tmpdir,
                frontendMode=frontend,
                collector=NoopMetricsCollector(),
            )
