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

"""Integration tests for hbm_dma_transpose utility.

Validates that hbm_dma_transpose correctly transposes 2D tensors of arbitrary
shape by tiling into (512, 128) chunks, transposing each via nisa.dma_transpose,
and copying to the destination HBM buffer.
"""

import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.mlp_mxfp8.common_utils import hbm_dma_transpose
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_utils import (
    create_and_set_active_sbm,
    get_active_sbm,
    with_active_sbm,
)

from test.utils import common_dataclasses
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework


@nki.jit
@with_active_sbm
def hbm_dma_transpose_kernel(src_hbm, M: int, N: int):
    """Wrapper kernel that calls hbm_dma_transpose and returns the transposed tensor."""
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("hbm_dma_transpose")
    dst_hbm = nl.ndarray((N, M), dtype=src_hbm.dtype, buffer=nl.shared_hbm)
    hbm_dma_transpose(src_hbm, dst_hbm)
    sbm.close_scope()
    return dst_hbm


def transpose_torch_ref(src_hbm, M: int, N: int):
    """Numpy reference: simple transpose."""
    return {"out": src_hbm.T.copy()}


# (M, N) test shapes — covers aligned, partial row tiles, partial col tiles, both partial
_TEST_SHAPES = [
    # Aligned to (512, 128)
    (512, 128),
    (1024, 256),
    (512, 512),
    (1024, 1024),
    # Partial column tiles (N not multiple of 128)
    (512, 64),
    (512, 192),
    (1024, 96),
    # Partial row tiles (M not multiple of 512)
    (256, 128),
    (768, 128),
    (384, 256),
    # Both partial
    (256, 64),
    (768, 192),
    (384, 96),
    # Large
    (2048, 512),
    (2048, 384),
]


@nki.jit
@with_active_sbm
def hbm_dma_transpose_src_slice_kernel(src_hbm, M: int, N: int, row_offset: int, row_size: int):
    """Transpose a row-slice of the source: src_hbm[row_offset:row_offset+row_size, :]."""
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("hbm_dma_transpose_src_slice")
    dst_hbm = nl.ndarray((N, row_size), dtype=src_hbm.dtype, buffer=nl.shared_hbm)
    hbm_dma_transpose(src_hbm, dst_hbm, M=row_size, src_row_offset=row_offset)
    sbm.close_scope()
    return dst_hbm


def transpose_src_slice_torch_ref(src_hbm, M: int, N: int, row_offset: int, row_size: int):
    """Numpy reference: transpose a row-slice."""
    return {"out": src_hbm[row_offset : row_offset + row_size, :].T.copy()}


# (M, N, row_offset, row_size) for source slice tests
_SRC_SLICE_SHAPES = [
    (1024, 256, 0, 512),
    (1024, 256, 512, 512),
    (2048, 512, 0, 1024),
    (2048, 512, 1024, 1024),
    (1024, 384, 256, 512),
]

# (M, N, col_offset, col_size) for destination slice tests
_DST_SLICE_SHAPES = [
    (1024, 256, 0, 128),
    (1024, 256, 128, 128),
    (2048, 512, 0, 256),
    (2048, 512, 256, 256),
    (1024, 384, 128, 256),
]


@nki.jit
@with_active_sbm
def hbm_dma_transpose_dst_slice_kernel(src_hbm, M: int, N: int, col_offset: int, col_size: int):
    """Transpose src_hbm[:, col_offset:col_offset+col_size] into a row-slice of the destination."""
    create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope("hbm_dma_transpose_dst_slice")
    # Only allocate the slice size — avoids comparing uninitialized regions
    dst_hbm = nl.ndarray((col_size, M), dtype=src_hbm.dtype, buffer=nl.shared_hbm)
    hbm_dma_transpose(
        src_hbm,
        dst_hbm,
        N=col_size,
        src_col_offset=col_offset,
    )
    sbm.close_scope()
    return dst_hbm


def transpose_dst_slice_torch_ref(src_hbm, M: int, N: int, col_offset: int, col_size: int):
    """Numpy reference: transpose a col-slice."""
    return {"out": src_hbm[:, col_offset : col_offset + col_size].T.copy()}


@pytest_test_metadata(name="HBM DMA Transpose")
@pytest_marks(["hbm_dma_transpose"])
@pytest.mark.platforms(exclude=[common_dataclasses.Platforms.TRN1, common_dataclasses.Platforms.TRN2])
class TestHbmDmaTranspose:
    """Correctness tests for hbm_dma_transpose utility."""

    @pytest.mark.fast
    @pytest.mark.parametrize("M,N", _TEST_SHAPES)
    def test_transpose(self, test_manager, platform_target, M, N):
        """Verify hbm_dma_transpose produces correct transposed output."""
        if not platform_target.is_trn3():
            pytest.skip("DMA transpose is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        rng = np.random.RandomState(42)
        src = rng.uniform(-1, 1, (M, N)).astype(np.float32).astype(nl.bfloat16)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=hbm_dma_transpose_kernel,
            torch_ref=transpose_torch_ref,
            kernel_input_generator=lambda _: {
                "src_hbm": src,
                "M": M,
                "N": N,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((N, M), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args)

    @pytest.mark.fast
    @pytest.mark.parametrize("M,N,row_offset,row_size", _SRC_SLICE_SHAPES)
    def test_transpose_src_slice(self, test_manager, platform_target, M, N, row_offset, row_size):
        """Verify hbm_dma_transpose works when source is a row-slice of a larger tensor."""
        if not platform_target.is_trn3():
            pytest.skip("DMA transpose is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        rng = np.random.RandomState(42)
        src = rng.uniform(-1, 1, (M, N)).astype(np.float32).astype(nl.bfloat16)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=hbm_dma_transpose_src_slice_kernel,
            torch_ref=transpose_src_slice_torch_ref,
            kernel_input_generator=lambda _: {
                "src_hbm": src,
                "M": M,
                "N": N,
                "row_offset": row_offset,
                "row_size": row_size,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((N, row_size), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args)

    @pytest.mark.fast
    @pytest.mark.parametrize("M,N,col_offset,col_size", _DST_SLICE_SHAPES)
    def test_transpose_dst_slice(self, test_manager, platform_target, M, N, col_offset, col_size):
        """Verify hbm_dma_transpose works when writing to a row-slice of the destination."""
        if not platform_target.is_trn3():
            pytest.skip("DMA transpose is only supported on TRN3.")

        compiler_args = common_dataclasses.CompilerArgs(
            platform_target=platform_target,
            logical_nc_config=1,
        )

        rng = np.random.RandomState(42)
        src = rng.uniform(-1, 1, (M, N)).astype(np.float32).astype(nl.bfloat16)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=hbm_dma_transpose_dst_slice_kernel,
            torch_ref=transpose_dst_slice_torch_ref,
            kernel_input_generator=lambda _: {
                "src_hbm": src,
                "M": M,
                "N": N,
                "col_offset": col_offset,
                "col_size": col_size,
            },
            output_tensor_descriptor=lambda _: {
                "out": np.zeros((col_size, M), dtype=nl.bfloat16),
            },
        )
        framework.run_test(test_config=None, compiler_args=compiler_args)
