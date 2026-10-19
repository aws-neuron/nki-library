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
"""Unit tests for kernel_tracer to verify trace-only mode catches kernel errors."""

import json
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import nki
import numpy as np
import pytest
from nki.compiler.ncc_driver import KernelDescriptor, NirResult  # ty: ignore[unresolved-import]

from test.utils import kernel_tracer
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    NKICompilationMode,
    Platforms,
    SeparationPassMode,
    TraceMode,
    ValidationArgs,
)
from test.utils.kernel_tracer import trace_kernel
from test.utils.metrics_collector import MetricsCollector, NoopMetricsCollector
from test.utils.metrics_emitter import MetricsEmitter, OutputMode


def test_collect_neuronx_cc_flags_enables_birsim_with_validation_tolerance():
    compiler_args = CompilerArgs(platform_target=Platforms.TRN2, enable_birsim=True)
    validation_args = ValidationArgs(golden_output={}, relative_accuracy=1e-2, absolute_accuracy=1e-5)

    compiler_flags = kernel_tracer._collect_neuronx_cc_flags(compiler_args, validation_args)

    assert compiler_flags == [
        "--internal-backend-options=--enable-birsim=True "
        "--enable-birsim-at-begin=False "
        "--enable-birsim-after-all=False "
        "--enable-birsim-at-end=True "
        "--birsim-output-tolerance 1.0,1e-05"
    ]


@pytest.mark.parametrize(
    "mode, expected",
    [
        (SeparationPassMode.DEFAULT, "--enable-experimental-separate-load-and-compute=default"),
        (SeparationPassMode.SERIALIZE_LOADS, "--enable-experimental-separate-load-and-compute=serialize-loads"),
    ],
)
def test_collect_neuronx_cc_flags_separation_public_spelling(monkeypatch, mode, expected):
    monkeypatch.setattr(kernel_tracer, "_has_public_separation_flag", lambda: True)
    compiler_args = CompilerArgs(platform_target=Platforms.TRN2, separation_pass_mode=mode)

    assert kernel_tracer._collect_neuronx_cc_flags(compiler_args) == [expected]


@pytest.mark.parametrize(
    "mode, expected",
    [
        (SeparationPassMode.DEFAULT, "--internal-enable-separate-load-and-compute=indirect"),
        (SeparationPassMode.SERIALIZE_LOADS, "--internal-enable-separate-load-and-compute=indirect-serialize"),
    ],
)
def test_collect_neuronx_cc_flags_separation_legacy_spelling(monkeypatch, mode, expected):
    """A pre-NCC-9599 compiler only accepts the internal flag and the old mode names."""
    monkeypatch.setattr(kernel_tracer, "_has_public_separation_flag", lambda: False)
    compiler_args = CompilerArgs(platform_target=Platforms.TRN2, separation_pass_mode=mode)

    assert kernel_tracer._collect_neuronx_cc_flags(compiler_args) == [expected]


def test_collect_neuronx_cc_flags_omits_separation_when_disabled(monkeypatch):
    monkeypatch.setattr(kernel_tracer, "_has_public_separation_flag", lambda: True)
    compiler_args = CompilerArgs(platform_target=Platforms.TRN2, separation_pass_mode=SeparationPassMode.NONE)

    assert kernel_tracer._collect_neuronx_cc_flags(compiler_args) == []


@pytest.mark.parametrize(
    "help_text, expected",
    [
        ("  --enable-experimental-separate-load-and-compute {none,default,serialize-loads}", True),
        ("  --enable-fast-loading-neuron-binaries", False),
    ],
)
def test_has_public_separation_flag_reads_compiler_help(monkeypatch, help_text, expected):
    monkeypatch.setattr(kernel_tracer.shutil, "which", lambda _: "/usr/bin/neuronx-cc")
    monkeypatch.setattr(
        kernel_tracer.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=help_text, stderr=""),
    )
    kernel_tracer._has_public_separation_flag.cache_clear()

    try:
        assert kernel_tracer._has_public_separation_flag() is expected
    finally:
        kernel_tracer._has_public_separation_flag.cache_clear()


@nki.jit
def kernel_with_assert_false(dummy_out):
    """Kernel that always fails with assert False."""
    assert False, "This kernel should fail"  # noqa: B011 -- NKI kernels can only use `assert`, not `raise`


class TestKernelTracerFailsOnAssert:
    """Verify that trace-only mode catches assertion errors in kernels."""

    def test_trace_only_fails_on_assert_false(self):
        """Trace-only mode should raise an error when kernel contains assert False."""
        dummy = np.zeros((1,), dtype=np.float32)
        kernel_args = KernelArgs(
            kernel_func=kernel_with_assert_false,
            compiler_input=CompilerArgs(platform_target=Platforms.TRN2, logical_nc_config=1),
            kernel_input={"dummy_out.must_alias_input": dummy},
            validation_args=ValidationArgs(
                golden_output=LazyGoldenGenerator(output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None)
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(Exception, match="This kernel should fail"):
                trace_kernel(
                    kernel_args,
                    mode=TraceMode.CompileOnly,
                    output_directory=tmpdir,
                    frontendMode=NKICompilationMode.parser,
                    collector=NoopMetricsCollector(),
                )


class TestRecordActualCompileFlow:
    """Verify the allocator that actually ran is recorded for querying."""

    def test_records_allocation_dimension(self):
        """A fallback shows up as Allocation=neuronx-cc even when linear was requested."""
        collector = MetricsCollector()

        kernel_tracer._record_actual_compile_flow(collector, SimpleNamespace(allocation="neuronx-cc"))

        assert collector.dimensions["Allocation"] == "neuronx-cc"

    def test_omits_dimension_missing_from_nir_result(self):
        """nki versions predating the field must not emit an empty dimension."""
        collector = MetricsCollector()

        kernel_tracer._record_actual_compile_flow(collector, SimpleNamespace())

        assert "Allocation" not in collector.dimensions

    def test_nir_result_owns_the_recorded_attribute(self):
        """Pin the contract to nki: a stub cannot catch reading the wrong object.

        The field lives on NirResult, not on its KernelDescriptor, so recording from
        result.descriptor would silently emit nothing forever.
        """
        nir_result_fields = getattr(NirResult, "__dataclass_fields__", {})
        if "allocation" not in nir_result_fields:
            pytest.skip("installed nki predates NirResult.allocation")

        descriptor_fields = getattr(KernelDescriptor, "__dataclass_fields__", {})
        assert "allocation" not in descriptor_fields, "expected allocation on NirResult, not KernelDescriptor"

    def test_dimension_reaches_emitted_metrics_record(self):
        """The dimension must survive emission so OpenSearch can aggregate it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector()
            collector.set_namespace("NeuronCompiler")
            collector.test_name = "test_example"
            collector.set_output_dir(tmpdir)
            collector.start_test()

            kernel_tracer._record_actual_compile_flow(collector, SimpleNamespace(allocation="nki"))
            MetricsEmitter(output_mode=OutputMode.FILE).emit(collector)

            record_files = list(Path(tmpdir).joinpath("metrics").glob("*.json"))
            assert len(record_files) == 1
            with open(record_files[0], "r") as record_file:
                record = json.load(record_file)

            assert record["Allocation"] == "nki"
