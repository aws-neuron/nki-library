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

"""Integration tests for the tensor_tensor BLAS primitive.

Element-wise binary op between two tensors: dst = op(src1, src2), where op is one of
multiply / add / subtract / maximum / minimum. Class-only primitive (no compact fn), so
there is a single invocation surface (blas.TensorTensor). One tiled kernel covers both
single-tile (P <= pmax) and multi-tile (P > pmax) inputs: the class API loops over
dst.get_num_tiles(), which is 1 for small P and > 1 (with a partial last tile) for large P.
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
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# Ops, one per operation named in the TensorTensor docstring. Each maps the kernel-side
# nl op to its torch reference (both operands are full tensors of the same shape). The refs
# use named (x1, x2) params so they match the kernel's input keys directly
OPS = {
    "multiply": {"nl_op": nl.multiply, "ref": lambda x1, x2: torch.multiply(x1, x2)},
    "add": {"nl_op": nl.add, "ref": lambda x1, x2: torch.add(x1, x2)},
    "subtract": {"nl_op": nl.subtract, "ref": lambda x1, x2: torch.subtract(x1, x2)},
    "maximum": {"nl_op": nl.maximum, "ref": lambda x1, x2: torch.maximum(x1, x2)},
    "minimum": {"nl_op": nl.minimum, "ref": lambda x1, x2: torch.minimum(x1, x2)},
}


# =============================================================================
# Test Kernels (thin @nki.jit wrappers around the primitive under test)
# =============================================================================


def _make_tensor_tensor_kernel(op_name):
    """Build a TensorTensor wrapper: dst = op(src1, src2). `op_name` (a key of OPS) selects
    the nl op, which is closed over -- so the kernel keeps a clean (x1, x2) tensor signature.
    """
    op = OPS[op_name]["nl_op"]

    @nki.jit
    def kernel_tensor_tensor(x1: nl.ndarray, x2: nl.ndarray) -> nl.ndarray:
        p, f = x1.shape
        p_tile = tile_size.pmax
        y = nl.ndarray((p, f), dtype=x1.dtype, buffer=nl.shared_hbm)

        src1_sb = tile_stream.alloc_logical((p, f), p_tile, x1.dtype, "src1")
        src2_sb = tile_stream.alloc_logical((p, f), p_tile, x2.dtype, "src2")
        dst_sb = tile_stream.alloc_logical((p, f), p_tile, x1.dtype, "dst")

        dma.Load(
            tile_stream.tile(src1_sb, (p_tile, f), iter_order=RowMajor(), logical_p=p),
            tile_stream.tile_hbm(x1, (p_tile, f), iter_order=RowMajor()),
        ).execute()
        dma.Load(
            tile_stream.tile(src2_sb, (p_tile, f), iter_order=RowMajor(), logical_p=p),
            tile_stream.tile_hbm(x2, (p_tile, f), iter_order=RowMajor()),
        ).execute()
        blas.TensorTensor(
            dst=tile_stream.tile(dst_sb, (p_tile, f), iter_order=RowMajor(), logical_p=p),
            src1=tile_stream.tile(src1_sb, (p_tile, f), iter_order=RowMajor(), logical_p=p),
            src2=tile_stream.tile(src2_sb, (p_tile, f), iter_order=RowMajor(), logical_p=p),
            op=op,
        ).execute()
        dma.Store(
            tile_stream.tile_hbm(y, (p_tile, f), iter_order=RowMajor()),
            tile_stream.tile(dst_sb, (p_tile, f), iter_order=RowMajor(), logical_p=p),
        ).execute()
        return y

    return kernel_tensor_tensor


# =============================================================================
# Inputs
# =============================================================================


def _generate_inputs(P, F, dtype):
    """Generate two independent source tiles (P, F). Distinct seeds so src1 != src2,
    which makes maximum/minimum select from both operands (not trivially one side)."""
    x1 = np.random.RandomState(42).randn(P, F).astype(dtype)
    x2 = np.random.RandomState(7).randn(P, F).astype(dtype)
    return {"x1": x1.astype(dtype), "x2": x2.astype(dtype)}


def _output_tensors(kernel_input):
    return {"out": np.zeros_like(kernel_input["x1"])}


# fmt: off
# op is orthogonal to shape (each op is the same nisa.tensor_tensor through the same tile
# loop), so the fast tier pairs each op 1:1 with a distinct shape -- every op AND every
# shape runs once, without the redundant op x shape cross-product. Shapes span single-tile
# (P <= 128) and multi-tile (P > 128, incl. a partial last tile). The sweep below does the
# thorough pairwise op x P x F x dtype coverage.
FAST_PARAM_NAMES = "P, F, dtype, op_name"
FAST_TEST_PARAMS = [
    pytest.param(1,   128,  np.float32,         "multiply", id="1_128_float32_multiply"),             # single tile
    pytest.param(128, 512,  np.float32,         "add",      id="128_512_float32_add"),                # single tile, full P
    pytest.param(64,  256,  ml_dtypes.bfloat16, "subtract", id="64_256_bfloat16_subtract"),           # single tile, bf16
    pytest.param(256, 64,   np.float32,         "maximum",  id="256_64_float32_maximum"),             # 2 full tiles
    pytest.param(300, 128,  ml_dtypes.bfloat16, "minimum",  id="300_128_bfloat16_minimum_partial"),   # 128 + 128 + 44
]
# fmt: on


# =============================================================================
# Tests
# =============================================================================


@pytest_marks(["tensor_tensor"])
class TestTensorTensorPrimitive:
    """Tests class for the tensor_tensor BLAS primitive kernel."""

    def _run(self, test_manager, platform_target, kernel_entry, torch_ref, P, F, dtype):
        is_bf16 = dtype == ml_dtypes.bfloat16

        def input_generator(test_config):
            return _generate_inputs(P, F, dtype)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_entry,
            torch_ref=torch_ref_wrapper(torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=_output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            atol=1e-2 if is_bf16 else 1e-4,
            rtol=1e-2 if is_bf16 else 1e-4,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(FAST_PARAM_NAMES, FAST_TEST_PARAMS)
    def test_tensor_tensor_fast(self, test_manager: Orchestrator, platform_target: Platforms, P, F, dtype, op_name):
        """TensorTensor over every op, across single- and multi-tile shapes: dst = op(src1, src2)."""
        self._run(
            test_manager,
            platform_target,
            _make_tensor_tensor_kernel(op_name),
            OPS[op_name]["ref"],
            P,
            F,
            dtype,
        )

    # One tiled kernel, so the sweep spans single-tile (P <= 128) and multi-tile
    # (P > 128, incl. partial last tiles) in a single pairwise sweep.
    @pytest.mark.coverage_parametrize(
        op_name=list(OPS.keys()),
        P=[1, 32, 128, 200, 300],
        F=[1, 64, 128, 512, 1024, 2048],
        dtype=[np.float32, ml_dtypes.bfloat16],
        coverage="pairs",
        enable_automatic_boundary_tests=False,
    )
    def test_tensor_tensor_sweep(
        self, test_manager: Orchestrator, platform_target: Platforms, op_name, P, F, dtype, is_negative_test_case
    ):
        """Pairwise sweep over op, partition P (single- and multi-tile), free dim F, dtype."""
        self._run(
            test_manager,
            platform_target,
            _make_tensor_tensor_kernel(op_name),
            OPS[op_name]["ref"],
            P,
            F,
            dtype,
        )
