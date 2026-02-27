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
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper
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
    elif name == "reshape":
        (new_shape,) = args
        return t.view(*new_shape), 0
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
    elif name == "vector_select":
        # args is (dim, offsets_np) where offsets_np is a numpy array of shape (P, 1)
        dim, offsets_np = args
        return torch.from_numpy(t.numpy()[offsets_np.flatten()]), 0
    elif name == "reinterpret_cast":
        (nki_dtype,) = args
        torch_dtype = _NKI_TO_TORCH_DTYPE[nki_dtype]
        return t.view(torch_dtype), 0
    raise ValueError(f"Unknown op: {name}")


# Mapping from NKI dtypes to torch dtypes (only dtypes torch supports)
_NKI_TO_TORCH_DTYPE = {
    nl.float32: torch.float32,
    nl.bfloat16: torch.bfloat16,
    nl.float16: torch.float16,
    nl.int32: torch.int32,
    nl.int16: torch.int16,
    nl.int8: torch.int8,
    nl.uint8: torch.uint8,
}


def pytorch_ref_ops(shape, ops, src_dtype=nl.float32) -> ViewResult:
    """Apply operations using PyTorch and return expected result."""
    t = torch.empty(shape, dtype=_NKI_TO_TORCH_DTYPE[src_dtype])
    total_offset = 0
    for op in ops:
        name = op[0]
        if name == "reinterpret_cast":
            # Offset units change across dtype boundaries
            nki_dtype = op[1][0]
            torch_dtype = _NKI_TO_TORCH_DTYPE[nki_dtype]
            old_size = t.element_size()
            new_size = torch.tensor([], dtype=torch_dtype).element_size()
            total_offset = total_offset * old_size // new_size
            t, _ = apply_pytorch_op(t, op)
        else:
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
    elif op_name == "reshape":
        (new_shape,) = args
        return t.reshape(new_shape)
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
    elif op_name == "vector_select":
        dim, vector_offset = args
        return t.vector_select(dim, vector_offset)
    elif op_name == "reinterpret_cast":
        (nki_dtype,) = args
        return t.reinterpret_cast(nki_dtype)
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
    src_dtype=nl.float32,
):
    """Test kernel that applies view ops and validates shape/strides/offset.

    Note: dummy_out is required because NKI kernels must have at least one output.
    """
    t = TensorView(nl.ndarray(shape, src_dtype, buffer))
    for i in range(len(ops)):
        t = apply_view_op(t, ops[i][0], ops[i][1])
    for i in range(len(expected_shape)):
        kernel_assert(t.shape[i] == expected_shape[i], f"shape[{i}]")
        kernel_assert(t.strides[i] == expected_strides[i], f"strides[{i}]")
    kernel_assert(t.offset == expected_offset, "offset")


# =============================================================================
# Helper
# =============================================================================


def run_test(test_manager: Orchestrator, platform_target: Platforms, shape, buffer, ops, nki_dtype=nl.float32):
    expected = pytorch_ref_ops(shape, ops, src_dtype=nki_dtype)
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
                "src_dtype": nki_dtype,
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
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "shape,new_shape",
        [
            # Flatten all non-partition dims
            ((128, 4, 6), (128, 24)),
            # Split a dim
            ((128, 24), (128, 4, 6)),
            # Split + merge across contiguous dims
            ((128, 24, 8), (128, 4, 48)),
            # Multi-dim reshape
            ((128, 2, 3, 4), (128, 6, 4)),
            # Identity reshape
            ((128, 64), (128, 64)),
            # Size-1 dims: squeeze
            ((128, 1, 64), (128, 64)),
            ((128, 64, 1), (128, 64)),
            ((128, 1, 1, 64), (128, 64)),
            # Size-1 dims: insert
            ((128, 64), (128, 1, 64)),
            ((128, 64), (128, 64, 1)),
            # Size-1 dims: both sides
            ((128, 1, 64), (128, 64, 1)),
            # Size-1 between real dims + reshape
            ((128, 1, 4, 6), (128, 24)),
            # Dim-0 is 1 (HBM only, filtered by parametrize skip below)
            ((1, 128, 64), (1, 8192)),
            # All-1 except one
            ((1, 1, 64, 1), (64,)),
            ((64,), (1, 64, 1)),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    def test_reshape(self, test_manager, platform_target, shape, new_shape, buffer):
        if buffer == nl.sbuf and (shape[0] != new_shape[0]):
            pytest.skip("SBUF partition dim must match")
        run_test(test_manager, platform_target, shape, buffer, (("reshape", (new_shape,)),))

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
            # reshape -> permute
            ((128, 4, 6), (("reshape", ((128, 2, 12),)), ("permute", ((0, 2, 1),)))),
            # slice -> reshape
            ((128, 32, 16), (("slice", (1, 0, 16, 1)), ("reshape", ((128, 4, 4, 16),)))),
            # reshape -> slice
            ((128, 24), (("reshape", ((128, 4, 6),)), ("slice", (1, 0, 2, 1)))),
            # reshape_dim -> reshape (roundtrip flatten)
            ((128, 24), (("reshape_dim", (1, (4, 6))), ("reshape", ((128, 24),)))),
            # expand_dim -> reshape (squeeze the 1-dim via reshape)
            ((128, 64), (("expand_dim", (1,)), ("reshape", ((128, 64),)))),
            # broadcast -> reshape within non-broadcast dims
            ((128, 1, 24), (("broadcast", (1, 8)), ("reshape", ((128, 8, 4, 6),)))),
            # reshape squeezing 1-dim -> slice
            ((128, 1, 64), (("reshape", ((128, 64),)), ("slice", (1, 0, 32, 1)))),
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

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "shape,ops,buffer,match",
        [
            # Size mismatch
            ((128, 64), (("reshape", ((128, 32),)),), nl.hbm, "size mismatch"),
            # Reshape across partition dim on SBUF
            ((128, 64), (("reshape", ((64, 128),)),), nl.sbuf, "partition dim"),
            # Non-contiguous (permute) then reshape that crosses the permuted boundary
            ((128, 4, 6), (("permute", ((0, 2, 1),)), ("reshape", ((128, 24),))), nl.hbm, "non-contiguous layout"),
            # Broadcast then reshape merging broadcast dim with real dim
            (
                (128, 1, 64),
                (("broadcast", (1, 8)), ("reshape", ((128, 512),))),
                nl.hbm,
                "non-contiguous layout",
            ),
            # SBUF: reshape splitting partition dim into multiple dims
            ((128, 64), (("reshape", ((64, 2, 64),)),), nl.sbuf, "partition dim"),
            # Indivisible split
            ((128, 64), (("reshape", ((128, 5, 13),)),), nl.hbm, "size mismatch"),
        ],
    )
    def test_negative_chain(self, test_manager, platform_target, shape, ops, buffer, match):
        dummy = np.zeros((1,), dtype=np.float32)
        with pytest.raises(Exception, match=match):
            test_manager.execute(
                KernelArgs(
                    kernel_func=kernel_test_view_ops,
                    compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                    kernel_input={
                        "dummy_out.must_alias_input": dummy,
                        "shape": shape,
                        "buffer": buffer,
                        "ops": ops,
                        "expected_shape": (1,),
                        "expected_strides": (1,),
                        "expected_offset": 0,
                    },
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None
                        )
                    ),
                )
            )

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    @pytest.mark.parametrize(
        "src_nki,dst_nki",
        [
            (nl.float32, nl.uint8),
            (nl.uint8, nl.float32),
            (nl.float32, nl.bfloat16),
            (nl.float32, nl.int32),
        ],
    )
    def test_reinterpret_cast(self, test_manager, platform_target, buffer, src_nki, dst_nki):
        ops = (("reinterpret_cast", (dst_nki,)),)
        run_test(test_manager, platform_target, (128, 512), buffer, ops, nki_dtype=src_nki)

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "shape,ops",
        [
            # slice -> cast -> reshape_dim
            (
                (128, 512),
                (("slice", (1, 0, 256, 1)), ("reinterpret_cast", (nl.uint8,)), ("reshape_dim", (1, (4, 256)))),
            ),
            # cast -> slice
            ((128, 512), (("reinterpret_cast", (nl.bfloat16,)), ("slice", (1, 0, 128, 1)))),
            # slice with offset -> cast
            ((128, 512), (("slice", (1, 128, 256, 1)), ("reinterpret_cast", (nl.bfloat16,)))),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.sbuf, nl.hbm])
    def test_chain_reinterpret_cast(self, test_manager, platform_target, shape, ops, buffer):
        run_test(test_manager, platform_target, shape, buffer, ops)


# =============================================================================
# Dynamic TensorView Tests - Full Compilation with Data Validation
# =============================================================================


@nki.jit
def kernel_dynamic_ops(input_tensor, dynamic_idx, vector_offsets_hbm, ops: tuple, out_shape: tuple):
    """Kernel that applies view ops (including dynamic select and vector_select) and copies data."""
    t = TensorView(input_tensor)
    for i in range(len(ops)):
        op_name = ops[i][0]
        args = ops[i][1]
        if op_name == "select_dynamic":
            dynamic_idx_sbuf = nl.ndarray((1, 1), dtype=dynamic_idx.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=dynamic_idx_sbuf, src=dynamic_idx)
            t = apply_view_op(t, op_name, (args[0], dynamic_idx_sbuf))
        elif op_name == "vector_select":
            vector_offsets_sbuf = nl.ndarray((128, 1), dtype=vector_offsets_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=vector_offsets_sbuf, src=vector_offsets_hbm)
            t = apply_view_op(t, op_name, (args[0], vector_offsets_sbuf))
        else:
            t = apply_view_op(t, op_name, args)
    # Copy through SBUF to force actual DMA
    sbuf = nl.ndarray(shape=out_shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf, src=t.get_view())

    output = nl.ndarray(shape=out_shape, dtype=nl.float32, buffer=nl.hbm)
    nisa.dma_copy(dst=output, src=sbuf)
    return output


def _apply_ref_ops(tensor, ops, dynamic_idx_value, vector_offsets):
    """Apply view ops to a tensor using PyTorch reference, resolving dynamic indices."""
    t = tensor
    for name, args in ops:
        if name == "select_dynamic":
            op = (name, (args[0], dynamic_idx_value))
        elif name == "vector_select":
            op = (name, (args[0], vector_offsets))
        else:
            op = (name, args)
        t, _ = apply_pytorch_op(t, op)
    return t.contiguous()


def dynamic_ops_torch_ref(input_tensor, dynamic_idx, vector_offsets_hbm, ops, out_shape):
    """Torch reference for dynamic view ops: apply ops to input and return expected output."""
    dynamic_idx_value = int(dynamic_idx[0, 0])
    vector_offsets = vector_offsets_hbm.numpy() if hasattr(vector_offsets_hbm, 'numpy') else vector_offsets_hbm
    return {"out": _apply_ref_ops(input_tensor, ops, dynamic_idx_value, vector_offsets)}


def generate_dynamic_ops_inputs(input_shape, ops):
    """Generate inputs for dynamic view ops test."""
    np.random.seed(42)
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Extract dynamic index value from ops (first select_dynamic found)
    dynamic_idx_value = next((args[1] for name, args in ops if name == "select_dynamic"), 0)
    dynamic_idx = np.array([[dynamic_idx_value]], dtype=np.int32)

    # For vector_select: compute dim 0 size at the point of vector_select to determine valid offset range
    has_vector_select = any(name == "vector_select" for name, _ in ops)
    if has_vector_select:
        t = torch.from_numpy(input_data)
        for name, args in ops:
            if name == "vector_select":
                break
            t, _ = apply_pytorch_op(t, (name, args))
        dim0_size = t.shape[0]
        vector_offsets = np.random.randint(0, dim0_size, size=(128, 1)).astype(np.int32)
    else:
        vector_offsets = np.zeros((128, 1), dtype=np.int32)

    out_shape = tuple(_apply_ref_ops(torch.from_numpy(input_data), ops, dynamic_idx_value, vector_offsets).shape)

    return {
        "input_tensor": input_data,
        "dynamic_idx": dynamic_idx,
        "vector_offsets_hbm": vector_offsets,
        "ops": ops,
        "out_shape": out_shape,
    }


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
            # NKI-831: dynamic select when requested stride doesn't exist in base tensor
            ((4, 8), (("reshape_dim", (0, (2, 2))), ("select_dynamic", (0, 1)))),
            # NKI-831: slice(step>1) then dynamic select on 3D tensor (2D AP after select)
            ((8, 2, 4), (("slice", (0, 0, 8, 2)), ("select_dynamic", (0, 1)))),
            # NKI-1180: basic vector select on 2D tensor
            ((256, 8), (("vector_select", (0,)),)),
            # NKI-1180: basic vector select on 3D tensor
            ((256, 2, 8), (("vector_select", (0,)),)),
            # NKI-1180: reshape_dim → vector_select (stride doesn't exist in base tensor)
            ((256, 8), (("reshape_dim", (0, (128, 2))), ("vector_select", (0,)))),
            # NKI-1180: slice(step>1) → vector_select
            ((512, 2, 4), (("slice", (0, 0, 512, 2)), ("vector_select", (0,)))),
            # NKI-1180: vector_select → permute (indirect dim stays at position 0)
            ((256, 2, 8), (("vector_select", (0,)), ("permute", ((0, 2, 1),)))),
        ],
    )
    def test_dynamic_select(self, test_manager, platform_target, input_shape, ops):
        def input_generator(test_config):
            return generate_dynamic_ops_inputs(input_shape, ops)

        def output_tensors(kernel_input):
            out_shape = kernel_input["out_shape"]
            return {"out": np.zeros(out_shape, dtype=np.float32)}

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_dynamic_ops,
            torch_ref=torch_ref_wrapper(dynamic_ops_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )


# =============================================================================
# TensorView reinterpret_cast Test
# =============================================================================


@nki.jit
def kernel_test_reinterpret_cast(dummy_out, shape: tuple):
    """Test reinterpret_cast preserves view state but changes dtype."""
    base = nl.ndarray(shape, nl.float16, nl.hbm)
    tv1 = TensorView(base).slice(dim=1, start=0, end=16)
    tv2 = tv1.reinterpret_cast(nl.uint16)

    # Shape, strides, offset should be preserved
    kernel_assert(tv2.shape == tv1.shape, "shape mismatch")
    kernel_assert(tv2.strides == tv1.strides, "strides mismatch")
    kernel_assert(tv2.offset == tv1.offset, "offset mismatch")
    # Dtype should change
    kernel_assert(tv2.dtype == nl.uint16, "dtype not changed")
    kernel_assert(tv1.dtype == nl.float16, "original dtype changed")


@nki.jit
def kernel_test_reinterpret_cast_mxfp4(dummy_out, shape: tuple):
    """Test reinterpret_cast uint16 -> float4_e2m1fn_x4 (MXFP4)."""
    base = nl.ndarray(shape, nl.uint16, nl.hbm)
    tv = TensorView(base).reinterpret_cast(nl.float4_e2m1fn_x4)
    kernel_assert(tv.dtype == nl.float4_e2m1fn_x4, "dtype mismatch")


@nki.jit
def kernel_test_reinterpret_cast_mxfp8(dummy_out, shape: tuple):
    """Test reinterpret_cast uint32 -> float8_e4m3fn_x4 (MXFP8)."""
    base = nl.ndarray(shape, nl.uint32, nl.hbm)
    tv = TensorView(base).reinterpret_cast(nl.float8_e4m3fn_x4)
    kernel_assert(tv.dtype == nl.float8_e4m3fn_x4, "dtype mismatch")


@nki.jit
def kernel_test_reinterpret_cast_bf16_fp16(dummy_out, shape: tuple):
    """Test reinterpret_cast bfloat16 <-> float16."""
    base_bf16 = nl.ndarray(shape, nl.bfloat16, nl.hbm)
    tv_fp16 = TensorView(base_bf16).reinterpret_cast(nl.float16)
    kernel_assert(tv_fp16.dtype == nl.float16, "bf16->fp16 failed")

    base_fp16 = nl.ndarray(shape, nl.float16, nl.hbm)
    tv_bf16 = TensorView(base_fp16).reinterpret_cast(nl.bfloat16)
    kernel_assert(tv_bf16.dtype == nl.bfloat16, "fp16->bf16 failed")


@nki.jit
def kernel_test_reinterpret_cast_cross_size_indirect(dummy_out, shape: tuple):
    """Test reinterpret_cast fails with cross-size cast after dynamic select."""
    base = nl.ndarray(shape, nl.float32, nl.hbm)
    tv1 = TensorView(base)
    # Manually set indirect_dim to simulate post-dynamic-select state
    tv1.indirect_dim = 0
    tv1.reinterpret_cast(nl.uint8)  # cross-size with indirect_dim - should fail


@nki.jit
def kernel_test_tensorview_from_tensorview(dummy_out, shape: tuple):
    """Test that TensorView can be constructed from another TensorView."""
    base = nl.ndarray(shape, nl.float32, nl.hbm)
    tv1 = TensorView(base).slice(dim=1, start=0, end=16)
    tv2 = TensorView(tv1)  # Construct from TensorView

    # tv2 should have same state as tv1
    kernel_assert(tv2.shape == tv1.shape, "shape mismatch")
    kernel_assert(tv2.strides == tv1.strides, "strides mismatch")
    kernel_assert(tv2.offset == tv1.offset, "offset mismatch")
    kernel_assert(tv2.base_tensor is tv1.base_tensor, "base_tensor mismatch")


@nki.jit
def kernel_test_chained_ops_on_tensorview_input(dummy_out, shape: tuple):
    """Test that ops can be chained on a TensorView constructed from TensorView."""
    base = nl.ndarray(shape, nl.float32, nl.hbm)
    tv1 = TensorView(base).slice(dim=1, start=0, end=32)
    tv2 = TensorView(tv1).slice(dim=1, start=0, end=16)  # Chain on TensorView input

    kernel_assert(tv2.shape[1] == 16, "chained slice failed")


@final
@pytest_test_metadata(name="TensorViewConstructor", pytest_marks=["tensor_view"])
class TestTensorViewConstructor:
    """Tests for TensorView constructor accepting TensorView as input."""

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_tensorview_from_tensorview(self, test_manager, platform_target):
        """Test TensorView can be constructed from another TensorView."""
        dummy = np.zeros((1,), dtype=np.float32)
        test_manager.execute(
            KernelArgs(
                kernel_func=kernel_test_tensorview_from_tensorview,
                compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                kernel_input={
                    "dummy_out.must_alias_input": dummy,
                    "shape": (128, 64),
                },
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None)
                ),
            )
        )

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_chained_ops_on_tensorview_input(self, test_manager, platform_target):
        """Test ops can be chained on TensorView constructed from TensorView."""
        dummy = np.zeros((1,), dtype=np.float32)
        test_manager.execute(
            KernelArgs(
                kernel_func=kernel_test_chained_ops_on_tensorview_input,
                compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                kernel_input={
                    "dummy_out.must_alias_input": dummy,
                    "shape": (128, 64),
                },
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None)
                ),
            )
        )


@final
@pytest_test_metadata(name="TensorViewReinterpretCast", pytest_marks=["tensor_view"])
class TestTensorViewReinterpretCast:
    """Tests for TensorView.reinterpret_cast method."""

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_reinterpret_cast(self, test_manager, platform_target):
        """Test reinterpret_cast changes dtype but preserves view state."""
        dummy = np.zeros((1,), dtype=np.float32)
        test_manager.execute(
            KernelArgs(
                kernel_func=kernel_test_reinterpret_cast,
                compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                kernel_input={
                    "dummy_out.must_alias_input": dummy,
                    "shape": (128, 64),
                },
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None)
                ),
            )
        )

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_reinterpret_cast_cross_size_indirect_blocked(self, test_manager, platform_target):
        """Test cross-size reinterpret_cast is blocked when indirect_dim is set."""
        dummy = np.zeros((1,), dtype=np.float32)
        with pytest.raises(
            Exception,
            match="reinterpret_cast with different element sizes is not supported after dynamic/vector select",
        ):
            test_manager.execute(
                KernelArgs(
                    kernel_func=kernel_test_reinterpret_cast_cross_size_indirect,
                    compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                    kernel_input={
                        "dummy_out.must_alias_input": dummy,
                        "shape": (128, 64),
                    },
                    validation_args=ValidationArgs(
                        golden_output=LazyGoldenGenerator(
                            output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None
                        )
                    ),
                )
            )

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_reinterpret_cast_mxfp4(self, test_manager, platform_target):
        """Test reinterpret_cast uint16 -> float4_e2m1fn_x4 (MXFP4)."""
        dummy = np.zeros((1,), dtype=np.float32)
        test_manager.execute(
            KernelArgs(
                kernel_func=kernel_test_reinterpret_cast_mxfp4,
                compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                kernel_input={"dummy_out.must_alias_input": dummy, "shape": (128, 64)},
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None)
                ),
            )
        )

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_reinterpret_cast_mxfp8(self, test_manager, platform_target):
        """Test reinterpret_cast uint32 -> float8_e4m3fn_x4 (MXFP8)."""
        dummy = np.zeros((1,), dtype=np.float32)
        test_manager.execute(
            KernelArgs(
                kernel_func=kernel_test_reinterpret_cast_mxfp8,
                compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                kernel_input={"dummy_out.must_alias_input": dummy, "shape": (128, 64)},
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None)
                ),
            )
        )

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_reinterpret_cast_bf16_fp16(self, test_manager, platform_target):
        """Test reinterpret_cast bfloat16 <-> float16."""
        dummy = np.zeros((1,), dtype=np.float32)
        test_manager.execute(
            KernelArgs(
                kernel_func=kernel_test_reinterpret_cast_bf16_fp16,
                compiler_input=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
                kernel_input={"dummy_out.must_alias_input": dummy, "shape": (128, 64)},
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray={"dummy_out": dummy}, lazy_golden_generator=None)
                ),
            )
        )

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_reinterpret_cast_cross_size(self, test_manager, platform_target):
        """Test cross-size reinterpret_cast: float32 → uint8 and back."""
        ops_down = (("reinterpret_cast", (nl.uint8,)),)
        run_test(test_manager, platform_target, (128, 512), nl.hbm, ops_down, nki_dtype=nl.float32)
        ops_up = (("reinterpret_cast", (nl.float32,)),)
        run_test(test_manager, platform_target, (128, 2048), nl.hbm, ops_up, nki_dtype=nl.uint8)

    @pytest.mark.trace_only
    @pytest.mark.fast
    def test_reinterpret_cast_dynamic_select_same_size(self, test_manager, platform_target):
        """Test same-size reinterpret_cast after dynamic_select (scalar_offset + indirect_dim)."""
        ops = (("select_dynamic", (0, 1)), ("reinterpret_cast", (nl.int32,)))
        run_test(test_manager, platform_target, (4, 2, 8), nl.hbm, ops)
