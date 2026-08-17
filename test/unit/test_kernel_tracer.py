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

import tempfile

import nki
import numpy as np
import pytest

from test.utils import kernel_tracer
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    NKICompilationMode,
    Platforms,
    TraceMode,
    ValidationArgs,
)
from test.utils.kernel_tracer import trace_kernel
from test.utils.metrics_collector import NoopMetricsCollector


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
