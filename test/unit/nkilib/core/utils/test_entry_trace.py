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
"""Unit tests for core/utils/entry_trace, traced through a real @nki.jit kernel."""

import tempfile

import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils import entry_trace
from nkilib_src.nkilib.core.utils.allocator import SbufManager

from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    NKICompilationMode,
    Platforms,
    TraceMode,
)
from test.utils.kernel_tracer import trace_kernel
from test.utils.metrics_collector import NoopMetricsCollector


@nki.jit
def kernel(input_a, input_b, bias=None):
    entry_trace.trace_kernel_entry("kernel", locals())
    out = nl.ndarray(input_a.shape, dtype=input_a.dtype, buffer=nl.shared_hbm)
    nl.store(out, nl.load(input_a))
    return out


@nki.jit
def kernel_with_sbm(input_a, sbm):
    entry_trace.trace_kernel_entry("kernel_with_sbm", locals())
    out = nl.ndarray(input_a.shape, dtype=input_a.dtype, buffer=nl.shared_hbm)
    nl.store(out, nl.load(input_a))
    return out


def _trace(kernel_func, **kernel_input):
    """Trace kernel_func with the tracer frontend, stopping before neuronx-cc.

    entry_trace is tracer-only, and TraceOnly returns once the frontend has run
    the body -- which is all that has to happen for an entry to print.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_kernel(
            KernelArgs(
                kernel_func=kernel_func,
                compiler_input=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=1),
                kernel_input=kernel_input,
            ),
            mode=TraceMode.TraceOnly,
            output_directory=tmpdir,
            frontendMode=NKICompilationMode.tracer,
            collector=NoopMetricsCollector(),
        )


@pytest.fixture
def enabled(monkeypatch):
    """Force the flag on; it is otherwise resolved from the env at import."""
    monkeypatch.setattr(entry_trace, "_ENABLED", True)


@pytest.fixture
def input_a():
    return np.zeros((1, 8), dtype=np.float32)


def test_silent_by_default(input_a, capsys):
    _trace(kernel, input_a=input_a, input_b=16)
    assert "[nki-entry" not in capsys.readouterr().out


def test_prints_entry(enabled, input_a, capsys):
    _trace(kernel, input_a=input_a, input_b=16)
    out = capsys.readouterr().out
    assert "[nki-entry #1] kernel" in out
    assert "input_a" in out and "(1, 8)" in out and "float32" in out
    assert "input_b" in out and "16" in out
    assert "none: bias" in out


def test_prints_note(enabled, capsys):
    entry_trace.trace_kernel_note("kernel", "lnc1 body selected")
    assert capsys.readouterr().out.strip() == "[nki-entry] kernel: lnc1 body selected"


def test_prints_sbm_fields(enabled, input_a, capsys):
    """A real SBM covers both expansions: its own __dict__ and a nested dataclass."""
    sbm = SbufManager(0, 128 * 1024)
    sbm.open_scope(name="outer")

    _trace(kernel_with_sbm, input_a=input_a, sbm=sbm)

    out = capsys.readouterr().out
    assert "sbm" in out and "BufferManager:" in out
    assert "upper_bound" in out and "131072" in out
    # scopes is a list of Scope dataclasses, expanded per element and per field
    assert "list[1]" in out
    assert "Scope:" in out and "num_sections" in out and "'outer'" in out
