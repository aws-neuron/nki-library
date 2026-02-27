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

"""Integration tests for the LayerNorm TKG kernel with various LNC configurations and batch sizes."""

from test.utils.common_dataclasses import CompilerArgs
from test.utils.coverage_parametrized_tests import FilterResult, assert_negative_test_case
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.subkernels.layernorm_tkg import layernorm_tkg
from nkilib_src.nkilib.core.subkernels.layernorm_torch import (
    layernorm_tkg_torch_ref,
    layernorm_tkg_torch_ref_lnc1,
)


def generate_inputs(batch: int, seqlen: int, hidden: int, dtype: np.dtype) -> dict:
    """
    Generate input tensors for layernorm_tkg test.

    Args:
        batch: Batch size
        seqlen: Sequence length
        hidden: Hidden dimension size
        dtype: Data type for tensors

    Returns:
        dict: Dictionary containing input, gamma, output, and beta tensors
    """
    np.random.seed(42)
    H0 = 128  # Partition dimension (nl.tile_size.pmax)
    return {
        "input": np.random.randn(batch, seqlen, hidden).astype(dtype),
        "gamma": np.random.randn(1, hidden).astype(dtype),
        "output.must_alias_input": np.zeros((H0, batch * seqlen, hidden // H0), dtype=dtype),
        "beta": np.random.randn(1, hidden).astype(dtype),
    }


# fmt: off
LAYERNORM_TKG_PARAMS = "lnc_degree, batch, seqlen, hidden, dtype"
LAYERNORM_TKG_TEST_CASES = [
    # LNC1 / Trn1
    (1, 1, 1, 8192, np.float16),
    (1, 1, 8, 8192, np.float16),
    # LNC2 batch 1
    (2, 1, 1, 5120, np.float16),
    (2, 1, 1, 8192, np.float16),
    # LNC2 higher batch
    (2, 2, 8, 8192, np.float16),
    (2, 4, 8, 8192, np.float16),
    (2, 128, 1, 8192, np.float16),
    (2, 4, 1, 5120, np.float16),
    # LNC2 higher hidden
    (2, 1, 1, 16384, np.float16),
    # Sharding threshold
    (2, 2, 5, 16384, np.float16),
]
# fmt: on


def _run_layernorm_tkg_test(
    test_manager: Orchestrator,
    lnc_degree: int,
    batch: int,
    seqlen: int,
    hidden: int,
    dtype: np.dtype,
    is_negative_test: bool = False,
) -> None:
    """Run layernorm_tkg test with given parameters."""

    def input_generator(test_config, input_tensor_def=None):
        return generate_inputs(batch, seqlen, hidden, dtype)

    def output_tensors(kernel_input):
        H0 = 128  # Partition dimension
        return {"out": np.zeros((H0, batch * seqlen, hidden // H0), dtype=dtype)}

    torch_ref = layernorm_tkg_torch_ref if lnc_degree == 2 else layernorm_tkg_torch_ref_lnc1

    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=layernorm_tkg,
        torch_ref=torch_ref_wrapper(torch_ref),
        kernel_input_generator=input_generator,
        output_tensor_descriptor=output_tensors,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=lnc_degree),
        rtol=2e-2,
        atol=1e-5,
        is_negative_test=is_negative_test,
    )


def filter_layernorm_tkg_combinations(lnc_degree, batch=None, seqlen=None, hidden=None, dtype=None):
    """Filter invalid LayerNorm TKG parameter combinations."""
    # H must be divisible by 128 (H0 = nl.tile_size.pmax)
    if hidden is not None and hidden % 128 != 0:
        return FilterResult.INVALID
    # H1 = H // 128 must be divisible by lnc_degree (for LNC sharding: H2 = H1 // lnc)
    if hidden is not None and (hidden // 128) % lnc_degree != 0:
        return FilterResult.INVALID
    return FilterResult.VALID


@pytest_test_metadata(
    name="LayerNorm TKG",
    pytest_marks=["layernorm", "tkg"],
)
class TestLayerNormTKGKernel:
    """Test class for LayerNorm TKG kernel using UnitTestFramework."""

    @pytest.mark.fast
    @pytest.mark.parametrize(LAYERNORM_TKG_PARAMS, LAYERNORM_TKG_TEST_CASES)
    def test_layernorm_tkg_unit(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        hidden: int,
        dtype,
    ):
        """Test layernorm_tkg using UnitTestFramework."""
        _run_layernorm_tkg_test(test_manager, lnc_degree, batch, seqlen, hidden, dtype)

    @pytest.mark.coverage_parametrize(
        lnc_degree=[1, 2],
        batch=[1, 2, 4, 8, 16, 32, 64],
        seqlen=[1, 2],
        hidden=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
        dtype=[nl.float16, nl.bfloat16, nl.float32],
        filter=filter_layernorm_tkg_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_layernorm_tkg_sweep(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        hidden: int,
        dtype,
        is_negative_test_case,
    ):
        """Run LayerNorm TKG sweep test across multiple dimension configurations."""
        with assert_negative_test_case(is_negative_test_case):
            _run_layernorm_tkg_test(
                test_manager, lnc_degree, batch, seqlen, hidden, dtype, is_negative_test=is_negative_test_case
            )

    @pytest.mark.coverage_parametrize(
        lnc_degree=[2],
        batch=[128, 256, 384, 512, 640],
        seqlen=[1],
        hidden=[3072, 4096, 6144, 8192],
        dtype=[nl.float16],
        filter=filter_layernorm_tkg_combinations,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_layernorm_tkg_sweep_large_batch(
        self,
        test_manager: Orchestrator,
        lnc_degree: int,
        batch: int,
        seqlen: int,
        hidden: int,
        dtype,
        is_negative_test_case,
    ):
        """Run LayerNorm TKG sweep test with large batch configurations."""
        with assert_negative_test_case(is_negative_test_case):
            _run_layernorm_tkg_test(
                test_manager, lnc_degree, batch, seqlen, hidden, dtype, is_negative_test=is_negative_test_case
            )
