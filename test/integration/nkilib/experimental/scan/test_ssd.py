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

"""Integration tests for SSD (Mamba-2) chunk-wise parallel scan kernel."""

from typing import final

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.scan.ssd import ssd
from nkilib_src.nkilib.experimental.scan.ssd_torch import ssd_torch_ref

from test.integration.nkilib.utils.tensor_generators import gaussian_tensor_generator
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


def _ssd_torch_ref_wrapper(x, dt, A, B, C, chunk_size=128, D=None, initial_state=None, causal_mask=None):
    """Torch reference wrapper that accepts and ignores causal_mask.

    The NKI kernel requires causal_mask as an explicit input tensor, but the
    torch reference computes its own lower-triangular mask internally.
    """
    return ssd_torch_ref(x, dt, A, B, C, chunk_size=chunk_size, D=D, initial_state=initial_state)


def generate_ssd_inputs(batch, nheads, seqlen, headdim, dstate, chunk_size, use_D, use_initial_state):
    """Generate SSD kernel inputs from parameters."""
    gen = gaussian_tensor_generator()

    x = gen(name="x", shape=(batch, nheads, seqlen, headdim), dtype=np.float32)
    x = x * 0.1

    # dt should be positive and small for numerical stability
    dt_raw = gen(name="dt_raw", shape=(batch, nheads, seqlen), dtype=np.float32)
    dt = np.abs(dt_raw) * 0.05

    # A should be negative for stable dynamics
    A_raw = gen(name="A", shape=(nheads,), dtype=np.float32)
    A = -np.abs(A_raw) * 0.5

    B = gen(name="B", shape=(batch, seqlen, dstate), dtype=np.float32) * 0.1
    C = gen(name="C", shape=(batch, seqlen, dstate), dtype=np.float32) * 0.1

    causal_mask = np.tril(np.ones((chunk_size, chunk_size), dtype=np.float32))

    inputs = {
        "x": x,
        "dt": dt,
        "A": A,
        "B": B,
        "C": C,
        "chunk_size": chunk_size,
        "causal_mask": causal_mask,
    }

    if use_D:
        inputs["D"] = gen(name="D", shape=(nheads,), dtype=np.float32) * 0.1
    if use_initial_state:
        inputs["initial_state"] = (
            gen(name="initial_state", shape=(batch, nheads, dstate, headdim), dtype=np.float32) * 0.01
        )

    return inputs


def filter_invalid_combinations(batch, nheads, seqlen, headdim, dstate, chunk_size, use_D, use_initial_state):
    """Filter out invalid parameter combinations."""
    if seqlen % chunk_size != 0:
        return FilterResult.INVALID
    if dstate > 128:
        return FilterResult.INVALID
    if chunk_size > 128:
        return FilterResult.INVALID
    return FilterResult.VALID


def _run_ssd_test(
    test_manager,
    platform_target,
    batch,
    nheads,
    seqlen,
    headdim,
    dstate,
    chunk_size,
    use_D,
    use_initial_state,
    atol=2e-1,
    rtol=5e-2,
    is_negative_test=False,
):
    """Run a single SSD test with the given parameters."""

    def input_generator(test_config):
        return generate_ssd_inputs(batch, nheads, seqlen, headdim, dstate, chunk_size, use_D, use_initial_state)

    def output_tensors(kernel_input):
        return {
            "y": np.zeros((batch, nheads, seqlen, headdim), dtype=np.float32),
            "final_state": np.zeros((batch, nheads, dstate, headdim), dtype=np.float32),
        }

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=ssd,
        torch_ref=torch_ref_wrapper(_ssd_torch_ref_wrapper),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=output_tensors,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target),
        atol=atol,
        rtol=rtol,
        is_negative_test=is_negative_test,
    )


@pytest_test_metadata(
    name="SSD",
    pytest_marks=["ssd"],
)
@final
class TestSSDKernel:
    """Test class for SSD (Mamba-2) kernel."""

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        batch=[1, 2],
        nheads=[1, 2],
        seqlen=[128, 256, 512],
        headdim=[64, 128],
        dstate=[16, 64],
        chunk_size=[64, 128],
        use_D=[True, False],
        use_initial_state=[True, False],
        filter=filter_invalid_combinations,
        coverage="singles",
        enable_automatic_boundary_tests=False,
    )
    def test_ssd_fast(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        nheads,
        seqlen,
        headdim,
        dstate,
        chunk_size,
        use_D,
        use_initial_state,
        is_negative_test_case,
    ):
        """Fast compile-only tests with minimal coverage."""
        _run_ssd_test(
            test_manager,
            platform_target,
            batch,
            nheads,
            seqlen,
            headdim,
            dstate,
            chunk_size,
            use_D,
            use_initial_state,
            atol=1e-1,
            rtol=5e-2,
            is_negative_test=is_negative_test_case,
        )

    @pytest.mark.coverage_parametrize(
        batch=[1, 2],
        nheads=[1, 2, 4],
        seqlen=[128, 256, 512, 1024],
        headdim=[64, 128, 256],
        dstate=[16, 32, 64, 128],
        chunk_size=[64, 128],
        use_D=[True, False],
        use_initial_state=[True, False],
        filter=filter_invalid_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_ssd_sweep(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        nheads,
        seqlen,
        headdim,
        dstate,
        chunk_size,
        use_D,
        use_initial_state,
        is_negative_test_case,
    ):
        """Full sweep tests with pairwise coverage."""
        _run_ssd_test(
            test_manager,
            platform_target,
            batch,
            nheads,
            seqlen,
            headdim,
            dstate,
            chunk_size,
            use_D,
            use_initial_state,
            atol=2e-1,
            rtol=5e-2,
            is_negative_test=is_negative_test_case,
        )

    # fmt: off
    _ssd_model_test_params = \
        "batch, nheads, seqlen, headdim, dstate, chunk_size, use_D, use_initial_state"
    _ssd_model_test_vectors = [
        # mamba2_130m: nheads=24, headdim=24, dstate=64
        [1, 24, 1024, 24, 64, 128, False, False],
        [1, 24, 2048, 24, 64, 128, False, False],
        [1, 24, 4096, 24, 64, 128, False, False],
        # mamba2_370m: nheads=48, headdim=32, dstate=64
        [1, 48, 1024, 32, 64, 128, False, False],
        [1, 48, 2048, 32, 64, 128, False, False],
        [1, 48, 4096, 32, 64, 128, False, False],
        # mamba2_1.3b / mamba2_2.7b: nheads=64, headdim=64, dstate=128
        [1, 64, 1024, 64, 128, 128, False, False],
        [1, 64, 2048, 64, 128, 128, False, False],
        [1, 64, 4096, 64, 128, 128, False, False],
        # large headdim (>512, exercises headdim tiling)
        [1, 4, 512, 1024, 64, 128, False, False],
        [1, 4, 512, 1024, 128, 128, False, False],
    ]
    # fmt: on

    @pytest_parametrize(_ssd_model_test_params, _ssd_model_test_vectors)
    def test_ssd_model(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        batch,
        nheads,
        seqlen,
        headdim,
        dstate,
        chunk_size,
        use_D,
        use_initial_state,
    ):
        """Model-driven tests with real Mamba-2 architecture dimensions."""
        _run_ssd_test(
            test_manager,
            platform_target,
            batch,
            nheads,
            seqlen,
            headdim,
            dstate,
            chunk_size,
            use_D,
            use_initial_state,
            atol=2e-1,
            rtol=5e-2,
        )
