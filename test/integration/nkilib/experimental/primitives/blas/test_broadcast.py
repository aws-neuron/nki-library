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

"""Integration tests for the broadcast BLAS primitive.

Replicates a single source partition row to all P rows of the destination:
dst[p, :] = src[src_partition, :] for every p (a partition broadcast). The source
tile has SRC_ROWS rows; src_partition selects which one is broadcast.
"""

import ml_dtypes
import nki
import nki.language as nl
import numpy as np
import pytest
import torch
from nki.language import tile_size

from nkilib_src.nkilib.experimental.primitives import blas, dma, tile_stream
from nkilib_src.nkilib.experimental.primitives.iter_order import RowMajor
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Test Kernels (thin @nki.jit wrappers around the primitive under test)
# =============================================================================


def _make_broadcast_kernel(P: int, src_rows: int, src_partition: int):
    """Build a wrapper that broadcasts src row `src_partition` to P destination rows.
    P, src_rows, and src_partition are closed over because they set the tensor shapes:
    src is (src_rows, F), dst is (P, F)."""

    @nki.jit
    def kernel_broadcast(src: nl.ndarray) -> nl.ndarray:
        _, f = src.shape  # src is (src_rows, F)
        y = nl.ndarray((P, f), dtype=src.dtype, buffer=nl.shared_hbm)

        src_sb = tile_stream.alloc_logical((src_rows, f), src_rows, src.dtype, "src")
        dst_sb = tile_stream.alloc_logical((P, f), P, src.dtype, "dst")

        dma.load(src_sb, src)
        blas.broadcast(dst_sb, src_sb, src_partition=src_partition)
        dma.store(y, dst_sb)
        return y

    return kernel_broadcast


def _make_broadcast_class_kernel(P: int, src_rows: int, src_partition: int):
    """Build a class-API (blas.Broadcast) wrapper covering both the valid single-tile case
    and the rejected multi-tile case with the same code.

    Broadcast is a single-tile op by design: it spreads one per-feature source row across
    <= pmax destination rows. For a larger destination the correct pattern is to broadcast
    once into a <= pmax buffer and reuse it across the tile loop, so multi-tile streams are
    rejected rather than silently doing an unintended per-tile broadcast."""

    @nki.jit
    def kernel_broadcast_class(src: nl.ndarray) -> nl.ndarray:
        _, f = src.shape
        p_tile = tile_size.pmax
        y = nl.ndarray((P, f), dtype=src.dtype, buffer=nl.shared_hbm)

        src_sb = tile_stream.alloc_logical((src_rows, f), src_rows, src.dtype, "src")
        dst_sb = tile_stream.alloc_logical((P, f), p_tile, src.dtype, "dst")

        dma.load(src_sb, src)
        blas.Broadcast(
            dst=tile_stream.tile(dst_sb, (p_tile, f), iter_order=RowMajor(), logical_p=P),
            src=tile_stream.tile(src_sb, (src_rows, f), iter_order=RowMajor()),
            src_partition=src_partition,
        ).execute()
        dma.Store(
            tile_stream.tile_hbm(y, (p_tile, f), iter_order=RowMajor()),
            tile_stream.tile(dst_sb, (p_tile, f), iter_order=RowMajor(), logical_p=P),
        ).execute()
        return y

    return kernel_broadcast_class


# =============================================================================
# Torch Reference
# =============================================================================


def _make_broadcast_torch_ref(P: int, src_partition: int):
    """Reference: replicate source row `src_partition` to P rows (exact copy)."""

    def broadcast_torch_ref(src: torch.Tensor) -> torch.Tensor:
        return src[src_partition : src_partition + 1, :].expand(P, src.shape[1]).clone()

    return broadcast_torch_ref


# =============================================================================
# Inputs
# =============================================================================


def _generate_inputs(src_rows, F, dtype):
    """Generate a source tile (src_rows, F). Broadcast is exact, so any values work."""
    np.random.seed(42)
    src = np.random.randn(src_rows, F).astype(dtype)
    return {"src": src.astype(dtype)}


def _output_tensors(P):
    def output_tensors(kernel_input):
        return {"out": np.zeros((P, kernel_input["src"].shape[1]), dtype=kernel_input["src"].dtype)}

    return output_tensors


def _filter_src_partition(P=None, F=None, dtype=None, SRC_ROWS=None, SP=None):
    """The broadcast row SP is a 0-based index into the SRC_ROWS-row source tile, so only
    SP < SRC_ROWS is meaningful. Drop the rest (REDUNDANT = skip, not a negative test)."""
    if SRC_ROWS is not None and SP is not None and SP >= SRC_ROWS:
        return FilterResult.REDUNDANT
    return FilterResult.VALID


# fmt: off
# P = destination partition-row count (1..pmax); F = free dim; SRC_ROWS = rows in the
# source tile; SP = which source row (< SRC_ROWS) to broadcast.
FAST_PARAM_NAMES = "P, F, dtype, SRC_ROWS, SP"
FAST_TEST_PARAMS = [
    pytest.param(1,   128,  np.float32,         1, 0, id="1_128_float32"),
    pytest.param(128, 512,  np.float32,         1, 0, id="128_512_float32"),
    pytest.param(64,  256,  ml_dtypes.bfloat16, 1, 0, id="64_256_bfloat16"),
    pytest.param(32,  1,    np.float32,         1, 0, id="32_1_float32"),          # F=1 edge
    pytest.param(128, 64,   np.float32,         4, 2, id="128_64_float32_sp2"),    # non-zero src_partition
    pytest.param(64,  32,   ml_dtypes.bfloat16, 8, 5, id="64_32_bfloat16_sp5"),    # non-zero src_partition
]

# Single-tile class-API cases, chosen at the edges. P=pmax is the single-tile boundary
# (one more row -> 2 tiles -> guard fires); F=1 min free dim; SP = last valid source row.
CLASS_SINGLE_TILE_PARAMS = [
    pytest.param(128, 1,    np.float32,         128, 127, id="Ppmax_F1_srcfull_splast"),
    pytest.param(1,   2048, ml_dtypes.bfloat16, 16,  15,  id="P1_Fmax_bfloat16_splast"),
]
# fmt: on


# =============================================================================
# Tests
# =============================================================================


@pytest_marks(["broadcast"])
class TestBroadcastPrimitive:
    """Tests class for the broadcast BLAS primitive kernel."""

    def _run(self, test_manager, platform_target, P, F, dtype, src_rows, src_partition):
        # Broadcast is an exact copy of the source row, so tolerances are 0.
        def input_generator(test_config):
            return _generate_inputs(src_rows, F, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_make_broadcast_kernel(P, src_rows, src_partition),
            torch_ref=torch_ref_wrapper(_make_broadcast_torch_ref(P, src_partition)),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors(P),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(FAST_PARAM_NAMES, FAST_TEST_PARAMS)
    def test_broadcast_fast(self, test_manager: Orchestrator, platform_target: Platforms, P, F, dtype, SRC_ROWS, SP):
        """Broadcast row SP to P rows: dst[p, :] = src[SP, :]."""
        self._run(test_manager, platform_target, P, F, dtype, SRC_ROWS, SP)

    @pytest.mark.coverage_parametrize(
        P=[1, 8, 32, 64, 96, 128],
        F=[1, 64, 128, 512, 1024, 2048],
        dtype=[np.float32, ml_dtypes.bfloat16],
        SRC_ROWS=[1, 4, 8, 16],
        SP=[0, 1, 5, 15],
        filter=_filter_src_partition,
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_broadcast_sweep(
        self, test_manager: Orchestrator, platform_target: Platforms, P, F, dtype, SRC_ROWS, SP, is_negative_test_case
    ):
        """Sweep with pairwise coverage over dst row count P, free dim F, dtype, source
        tile row count SRC_ROWS, and broadcast row SP (filtered to SP < SRC_ROWS)."""
        self._run(test_manager, platform_target, P, F, dtype, src_rows=SRC_ROWS, src_partition=SP)

    @pytest.mark.fast
    @pytest.mark.parametrize(FAST_PARAM_NAMES, CLASS_SINGLE_TILE_PARAMS)
    def test_broadcast_class_single_tile(
        self, test_manager: Orchestrator, platform_target: Platforms, P, F, dtype, SRC_ROWS, SP
    ):
        """Single-tile broadcast via the class API (blas.Broadcast) at the edges. Confirms the
        single-tile guard passes the valid case (each stream has one tile) -- notably at P=pmax,
        the boundary just below where the guard starts rejecting -- and produces correct output."""
        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return _generate_inputs(SRC_ROWS, F, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_make_broadcast_class_kernel(P, SRC_ROWS, SP),
            torch_ref=torch_ref_wrapper(_make_broadcast_torch_ref(P, SP)),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors(P),
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            atol=0.0,
            rtol=0.0,
        )

    @pytest.mark.fast
    def test_broadcast_multitile_rejected(self, test_manager: Orchestrator, platform_target: Platforms):
        """Broadcast is single-tile by design (one per-feature row -> <= pmax rows). Feeding
        the class API multi-tile src/dst streams (P > pmax) must be rejected by the
        single-tile guard rather than silently doing a per-tile broadcast. This pins that
        contract; the correct pattern for a larger destination is broadcast-once-and-reuse."""
        P, F, dtype = 256, 8, np.float32  # P > pmax -> dst has 2 tiles -> guard fires

        def input_generator(test_config):
            return _generate_inputs(1, F, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_make_broadcast_class_kernel(P, src_rows=1, src_partition=0),
            torch_ref=torch_ref_wrapper(_make_broadcast_torch_ref(P, 0)),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors(P),
        )
        with pytest.raises(Exception, match="src and dst must each be a single tile"):
            framework.run_test(
                test_config=None,
                compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                atol=0.0,
                rtol=0.0,
            )
