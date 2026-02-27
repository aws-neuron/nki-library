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

"""Unit tests for TensorView module - compile-time validation only."""

from dataclasses import dataclass
from test.utils.common_dataclasses import (
    CompilerArgs,
    KernelArgs,
    LazyGoldenGenerator,
    Platforms,
    ValidationArgs,
)
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from typing import Tuple, final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert
from nkilib_src.nkilib.core.utils.tensor_view import TensorView


@dataclass
class ViewResult:
    """Expected result from a TensorView operation."""

    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    offset: int = 0


# =============================================================================
# PyTorch Reference
# =============================================================================


def apply_pytorch_op(t, op):
    """Apply a single operation to a PyTorch tensor, return (tensor, offset_delta)."""
    name, args = op
    if name == "slice":
        dim, start, end, step = args
        slices = [slice(None)] * t.dim()
        slices[dim] = slice(start, end, step)
        return t[tuple(slices)], t.stride(dim) * start
    elif name == "permute":
        (dims,) = args
        return t.permute(dims), 0
    elif name == "broadcast":
        dim, size = args
        new_shape = list(t.shape)
        new_shape[dim] = size
        return t.expand(new_shape), 0
    elif name == "reshape_dim":
        dim, new_shape = args
        full_shape = list(t.shape[:dim]) + list(new_shape) + list(t.shape[dim + 1 :])
        return t.view(*full_shape), 0
    elif name == "flatten_dims":
        start_dim, end_dim = args
        return t.flatten(start_dim, end_dim), 0
    elif name == "expand_dim":
        (dim,) = args
        return t.unsqueeze(dim), 0
    elif name == "squeeze_dim":
        (dim,) = args
        return t.squeeze(dim), 0
    elif name == "select":
        dim, index = args
        return t.select(dim, index), t.stride(dim) * index
    elif name == "select_dynamic":
        dim, index = args
        return t.select(dim, index), t.stride(dim) * index
    raise ValueError(f"Unknown op: {name}")


def pytorch_ref_ops(shape, ops) -> ViewResult:
    """Apply operations using PyTorch and return expected result."""
    t = torch.empty(shape)
    total_offset = 0
    for op in ops:
        t, offset = apply_pytorch_op(t, op)
        total_offset += offset
    return ViewResult(tuple(t.shape), tuple(t.stride()), total_offset)


# =============================================================================
# Test Kernel
# =============================================================================


def apply_view_op(t: TensorView, op_name: str, args: tuple) -> TensorView:
    """Apply a single view operation to TensorView."""
    if op_name == "slice":
        dim, start, end, step = args
        return t.slice(dim, start, end, step)
    elif op_name == "permute":
        (dims,) = args
        return t.permute(dims)
    elif op_name == "broadcast":
        dim, size = args
        return t.broadcast(dim, size)
    elif op_name == "reshape_dim":
        dim, new_shape = args
        return t.reshape_dim(dim, new_shape)
    elif op_name == "flatten_dims":
        start_dim, end_dim = args
        return t.flatten_dims(start_dim, end_dim)
    elif op_name == "expand_dim":
        (dim,) = args
        return t.expand_dim(dim)
    elif op_name == "squeeze_dim":
        (dim,) = args
        return t.squeeze_dim(dim)
    elif op_name == "select":
        dim, index = args
        return t.select(dim, index)
    elif op_name == "select_dynamic":
        dim, index = args
        return t.select(dim, index)
    kernel_assert(False, f"Unknown op: {op_name}")


@nki.jit
def kernel_test_view_ops(
    dummy_out,
    shape: tuple,
    buffer: str,
    ops: tuple,
    expected_shape: tuple,
    expected_strides: tuple,
    expected_offset: int,
):
    """Test kernel that applies view ops and validates shape/strides/offset.

    Note: dummy_out is required because NKI kernels must have at least one output.
    """
    t = TensorView(nl.ndarray(shape, nl.float32, buffer))
    for i in range(len(ops)):
        t = apply_view_op(t, ops[i][0], ops[i][1])
    for i in range(len(expected_shape)):
        kernel_assert(t.shape[i] == expected_shape[i], f"shape[{i}]")
        kernel_assert(t.strides[i] == expected_strides[i], f"strides[{i}]")
    kernel_assert(t.offset == expected_offset, "offset")


# =============================================================================
# Helper
# =============================================================================


def run_test(test_manager: Orchestrator, platform_target: Platforms, shape, buffer, ops):
    expected = pytorch_ref_ops(shape, ops)
    # Dummy output required because NKI kernels must have at least one output.
    # The .must_alias_input suffix tells the framework this output aliases the input.
    dummy = np.zeros((1,), dtype=np.float32)
    test_manager.execute(
        KernelArgs(
            kernel_func=kernel_test_view_ops,
            compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
            kernel_input={
                "dummy_out.must_alias_input": dummy,
                "shape": shape,
                "buffer": buffer,
                "ops": ops,
                "expected_shape": expected.shape,
                "expected_strides": expected.strides,
                "expected_offset": expected.offset,
            },
            validation_args=ValidationArgs(
                golden_output=LazyGoldenGenerator(output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None)
            ),
        )
    )


# =============================================================================
# Tests
# =============================================================================


@pytest_test_metadata(
    name="TensorView",
    pytest_marks=["tensor_view"],
)
@final
class TestTensorView:
    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize("shape", [(128, 64), (128, 32, 16)])
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    @pytest.mark.parametrize(
        "dim,start,end,step",
        [(0, 0, 64, 1), (0, 0, 64, 2), (0, 64, 256, 32), (1, 0, 32, 1), (1, 4, 16, 2), (1, 0, 100, 1)],
    )
    def test_slice(self, test_manager, platform_target, shape, buffer, dim, start, end, step):
        if dim >= len(shape):
            pytest.skip("Invalid params")
        run_test(test_manager, platform_target, shape, buffer, (("slice", (dim, start, end, step)),))

    @pytest.mark.trace_only
    @pytest.mark.parametrize(
        "shape,dims",
        [
            ((128, 64), (0, 1)),
            ((128, 64), (1, 0)),
            ((128, 32, 16), (0, 1, 2)),
            ((128, 32, 16), (0, 2, 1)),
            ((128, 32, 16), (1, 0, 2)),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    def test_permute(self, test_manager, platform_target, shape, dims, buffer):
        if buffer == nl.sbuf and dims[0] != 0:
            pytest.skip("SBUF partition dim must stay at 0")
        run_test(test_manager, platform_target, shape, buffer, (("permute", (dims,)),))

    @pytest.mark.trace_only
    @pytest.mark.parametrize("shape,dim", [((128, 1, 64), 1), ((128, 1), 1), ((128, 64, 1), 2)])
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    @pytest.mark.parametrize("size", [8, 16])
    def test_broadcast(self, test_manager, platform_target, shape, dim, buffer, size):
        run_test(test_manager, platform_target, shape, buffer, (("broadcast", (dim, size)),))

    @pytest.mark.trace_only
    @pytest.mark.parametrize(
        "shape,dim,new_shape",
        [
            ((128, 24), 1, (4, 6)),
            ((128, 24), 1, (2, 3, 4)),
            ((128, 24, 8), 1, (4, 6)),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    def test_reshape_dim(self, test_manager, platform_target, shape, dim, new_shape, buffer):
        run_test(test_manager, platform_target, shape, buffer, (("reshape_dim", (dim, new_shape)),))

    @pytest.mark.trace_only
    @pytest.mark.parametrize("shape,start_dim,end_dim", [((128, 2, 3, 4), 1, 2), ((128, 2, 3, 4), 2, 3)])
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    def test_flatten_dims(self, test_manager, platform_target, shape, start_dim, end_dim, buffer):
        run_test(test_manager, platform_target, shape, buffer, (("flatten_dims", (start_dim, end_dim)),))

    @pytest.mark.trace_only
    @pytest.mark.parametrize("shape", [(128, 64), (128, 32, 16)])
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    @pytest.mark.parametrize("dim", [1, 2])
    def test_expand_dim(self, test_manager, platform_target, shape, buffer, dim):
        if dim > len(shape):
            pytest.skip("Invalid dim")
        run_test(test_manager, platform_target, shape, buffer, (("expand_dim", (dim,)),))

    @pytest.mark.trace_only
    @pytest.mark.parametrize("shape,dim", [((128, 1, 64), 1), ((128, 64, 1), 2), ((128, 1, 32, 1), 1)])
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    def test_squeeze_dim(self, test_manager, platform_target, shape, dim, buffer):
        run_test(test_manager, platform_target, shape, buffer, (("squeeze_dim", (dim,)),))

    @pytest.mark.trace_only
    @pytest.mark.parametrize("shape", [(128, 8, 64), (128, 16, 32)])
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    @pytest.mark.parametrize("dim,index", [(1, 0), (1, 4), (2, 0), (2, 8)])
    def test_select(self, test_manager, platform_target, shape, buffer, dim, index):
        if index >= shape[dim]:
            pytest.skip("Invalid index")
        run_test(test_manager, platform_target, shape, buffer, (("select", (dim, index)),))

    @pytest.mark.trace_only
    @pytest.mark.parametrize(
        "shape,ops",
        [
            # slice -> expand_dim
            ((128, 64), (("slice", (1, 0, 32, 1)), ("expand_dim", (1,)))),
            # expand_dim -> squeeze_dim (roundtrip)
            ((128, 64), (("expand_dim", (1,)), ("squeeze_dim", (1,)))),
            # reshape_dim -> flatten_dims (roundtrip)
            ((128, 24), (("reshape_dim", (1, (4, 6))), ("flatten_dims", (1, 2)))),
            # slice -> permute
            ((128, 32, 16), (("slice", (1, 0, 16, 1)), ("permute", ((0, 2, 1),)))),
            # reshape_dim -> permute
            ((128, 24), (("reshape_dim", (1, (4, 6))), ("permute", ((0, 2, 1),)))),
            # slice -> broadcast
            ((128, 1, 64), (("slice", (2, 0, 32, 1)), ("broadcast", (1, 8)))),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    def test_chain(self, test_manager, platform_target, shape, ops, buffer):
        run_test(test_manager, platform_target, shape, buffer, ops)

    @pytest.mark.trace_only
    @pytest.mark.parametrize(
        "shape,ops",
        [
            # NKI-796: select -> select -> expand_dim -> broadcast (HBM partition broadcast pattern)
            ((8, 2, 1), (("select", (0, 3)), ("select", (0, 0)), ("expand_dim", (0,)), ("broadcast", (0, 128)))),
            ((8, 2, 1), (("slice", (0, 3, 4, 1)), ("select", (1, 0)), ("broadcast", (0, 128)))),
        ],
    )
    def test_chain_hbm(self, test_manager, platform_target, shape, ops):
        run_test(test_manager, platform_target, shape, nl.hbm, ops)


# =============================================================================
# Dynamic TensorView Tests - Full Compilation with Data Validation
# =============================================================================


def dynamic_ref_ops(input_data: np.ndarray, ops) -> np.ndarray:
    """Apply operations to numpy array and return expected output data."""
    t = torch.from_numpy(input_data)
    for op in ops:
        t, _ = apply_pytorch_op(t, op)
    return t.contiguous().numpy()


@nki.jit
def kernel_dynamic_ops(input_tensor, dynamic_idx, ops: tuple, out_shape: tuple):
    """Kernel that applies view ops (including dynamic select) and copies data."""
    t = TensorView(input_tensor)
    for i in range(len(ops)):
        op_name = ops[i][0]
        args = ops[i][1]
        if op_name == "select_dynamic":
            dynamic_idx_sbuf = nl.ndarray((1,), dtype=dynamic_idx.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=dynamic_idx_sbuf, src=dynamic_idx)
            t = apply_view_op(t, op_name, (args[0], dynamic_idx_sbuf))
        else:
            t = apply_view_op(t, op_name, args)
    # Copy through SBUF to force actual DMA
    sbuf = nl.ndarray(shape=out_shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf, src=t.get_view())

    output = nl.ndarray(shape=out_shape, dtype=nl.float32, buffer=nl.hbm)
    nisa.dma_copy(dst=output, src=sbuf)
    return output


def run_dynamic_test(test_manager: Orchestrator, platform_target: Platforms, input_shape, ops):
    """Run a dynamic view test with data validation."""
    np.random.seed(42)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Extract dynamic index value from ops (first select_dynamic found)
    dynamic_idx_value = next((args[1] for name, args in ops if name == "select_dynamic"), 0)
    dynamic_idx = np.array([[dynamic_idx_value]], dtype=np.int32)

    # Compute reference output using ops as-is (PT uses int index)
    expected_output = dynamic_ref_ops(input_data, ops)
    out_shape = expected_output.shape
    test_manager.execute(
        KernelArgs(
            kernel_func=kernel_dynamic_ops,
            compiler_input=CompilerArgs(platform_target=platform_target),
            kernel_input={
                "input_tensor": input_data,
                "dynamic_idx": dynamic_idx,
                "ops": ops,
                "out_shape": out_shape,
            },
            validation_args=ValidationArgs(
                golden_output=LazyGoldenGenerator(
                    output_ndarray={"out": expected_output}, lazy_golden_generator=lambda: {"out": expected_output}
                )
            ),
        )
    )


@final
@pytest_test_metadata(name="DynamicTensorView", pytest_marks=["tensor_view"])
class TestDynamicTensorView:
    """Tests for TensorView with dynamic indexing - requires full compilation."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "input_shape,ops",
        [
            # Static select only (baseline)
            ((4, 2, 8), (("select", (0, 0)),)),
            # Dynamic select only
            ((4, 2, 8), (("select_dynamic", (0, 1)),)),
            # dynamic_select → permute
            ((4, 2, 8), (("select_dynamic", (0, 1)), ("permute", ((1, 0),)))),
            # permute → dynamic_select
            ((4, 2, 8), (("permute", ((1, 0, 2),)), ("select_dynamic", (0, 1)))),
            # NKI-798: dynamic_select → static_select → expand_dim → broadcast [TODO]
            # ((4, 2, 1), (("select_dynamic", (0, 1)), ("select", (0, 0)), ("expand_dim", (0,)), ("broadcast", (0, 128)))),
            # NKI-831: dynamic select when requested stride doesn't exist in base tensor [TODO]
            # ((4, 8), (("reshape_dim"), (0, (2, 2)), ("select_dynamic", (0, 1)))),
            # ((8, 4), (("slice"), (0, 0, 8, 2), ("select_dynamic", (0, 1)))),
        ],
    )
    def test_dynamic_select(self, test_manager, platform_target, input_shape, ops):
        run_dynamic_test(test_manager, platform_target, input_shape, ops)
