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

"""Unit tests for TiledTensor module - compile-time validation only."""

from typing import final

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert
from nkilib_src.nkilib.core.utils.tiled_tensor import TiledTensor

from test.utils.common_dataclasses import (
    CompilerArgs,
)
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper

# =============================================================================
# Test Kernels
# =============================================================================


def apply_tiled_op(tt, op_name, args):
    """Apply a single operation to a TiledTensor."""
    if op_name == "select":
        dim, idx = args
        return tt.select(dim, idx)
    elif op_name == "slice":
        dim, start, end = args
        return tt.slice(dim, start, end)
    elif op_name == "reshape_dim":
        dim, shape = args
        return tt.reshape_dim(dim, shape)
    elif op_name == "flatten_dims":
        start_dim, end_dim = args
        return tt.flatten_dims(start_dim, end_dim)
    elif op_name == "expand_dim":
        (dim,) = args
        return tt.expand_dim(dim)
    elif op_name == "squeeze_dim":
        (dim,) = args
        return tt.squeeze_dim(dim)
    elif op_name == "reshape":
        (new_shape,) = args
        return tt.reshape(new_shape)
    kernel_assert(False, f"Unknown op: {op_name}")


@nki.jit
def kernel_test_tiled_tensor_shape(
    dummy_out,
    source_shape: tuple,
    tile_size: tuple,
    buffer: str,
    ops: tuple,
    expected_grid_shape: tuple,
    expected_view_shape: tuple,
    expected_view_strides: tuple,
    expected_view_offset: int,
):
    """Test kernel that creates a TiledTensor, applies ops, validates grid shape,
    then calls get_view() and validates the resulting view's shape, strides, and offset."""
    src = nl.ndarray(source_shape, nl.bfloat16, buffer)
    tt = TiledTensor(src, tile_size)
    for i in range(len(ops)):
        tt = apply_tiled_op(tt, ops[i][0], ops[i][1])
    grid_shape = tt.get_shape()
    for i in range(len(expected_grid_shape)):
        kernel_assert(
            grid_shape[i] == expected_grid_shape[i],
            f"grid shape[{i}]: expected {expected_grid_shape[i]}, got {grid_shape[i]}",
        )
    kernel_assert(
        len(grid_shape) == len(expected_grid_shape),
        f"grid ndim: expected {len(expected_grid_shape)}, got {len(grid_shape)}",
    )
    view = tt.get_view()
    kernel_assert(
        len(view.shape) == len(expected_view_shape),
        f"view ndim: expected {len(expected_view_shape)}, got {len(view.shape)}",
    )
    for i in range(len(expected_view_shape)):
        kernel_assert(
            view.shape[i] == expected_view_shape[i],
            f"view shape[{i}]: expected {expected_view_shape[i]}, got {view.shape[i]}",
        )
    for i in range(len(expected_view_strides)):
        kernel_assert(
            view.strides[i] == expected_view_strides[i],
            f"view strides[{i}]: expected {expected_view_strides[i]}, got {view.strides[i]}",
        )
    kernel_assert(
        view.offset == expected_view_offset,
        f"view offset: expected {expected_view_offset}, got {view.offset}",
    )
    return dummy_out


@nki.jit
def kernel_test_single_tile(
    dummy_out,
    source_shape: tuple,
    tile_size: tuple,
    buffer: str,
    grid_coords: tuple,
    expected_tile_shape: tuple,
    expected_tile_strides: tuple,
    expected_tile_offset: int,
):
    """Test kernel that accesses a single tile and validates its shape, strides, offset."""
    src = nl.ndarray(source_shape, nl.bfloat16, buffer)
    tt = TiledTensor(src, tile_size)
    tile_view = tt.get_tile(grid_coords)
    kernel_assert(
        len(tile_view.shape) == len(expected_tile_shape),
        f"tile ndim: expected {len(expected_tile_shape)}, got {len(tile_view.shape)}",
    )
    for i in range(len(expected_tile_shape)):
        kernel_assert(
            tile_view.shape[i] == expected_tile_shape[i],
            f"tile shape[{i}]: expected {expected_tile_shape[i]}, got {tile_view.shape[i]}",
        )
    for i in range(len(expected_tile_strides)):
        kernel_assert(
            tile_view.strides[i] == expected_tile_strides[i],
            f"tile strides[{i}]: expected {expected_tile_strides[i]}, got {tile_view.strides[i]}",
        )
    kernel_assert(
        tile_view.offset == expected_tile_offset,
        f"tile offset: expected {expected_tile_offset}, got {tile_view.offset}",
    )
    return dummy_out


@nki.jit
def kernel_test_get_view(
    dummy_out,
    source_shape: tuple,
    tile_size: tuple,
    buffer: str,
    ops: tuple,
    expected_view_shape: tuple,
    expected_view_strides: tuple,
    expected_view_offset: int,
):
    """Test kernel that creates TiledTensor, applies ops, calls get_view,
    and validates shape, strides, and offset."""
    src = nl.ndarray(source_shape, nl.bfloat16, buffer)
    tt = TiledTensor(src, tile_size)
    for i in range(len(ops)):
        tt = apply_tiled_op(tt, ops[i][0], ops[i][1])
    view = tt.get_view()
    kernel_assert(
        len(view.shape) == len(expected_view_shape),
        f"get_view ndim: expected {len(expected_view_shape)}, got {len(view.shape)}",
    )
    for i in range(len(expected_view_shape)):
        kernel_assert(
            view.shape[i] == expected_view_shape[i],
            f"get_view shape[{i}]: expected {expected_view_shape[i]}, got {view.shape[i]}",
        )
    for i in range(len(expected_view_strides)):
        kernel_assert(
            view.strides[i] == expected_view_strides[i],
            f"get_view strides[{i}]: expected {expected_view_strides[i]}, got {view.strides[i]}",
        )
    kernel_assert(
        view.offset == expected_view_offset,
        f"get_view offset: expected {expected_view_offset}, got {view.offset}",
    )
    return dummy_out


@nki.jit
def kernel_test_force_get_view(
    dummy_out,
    source_shape: tuple,
    tile_size: tuple,
    buffer: str,
    ops: tuple,
    expected_view_shape: tuple,
    expected_view_strides: tuple,
    expected_view_offset: int,
):
    """Test kernel that creates TiledTensor, applies ops, calls force_get_view,
    and validates shape, strides, and offset."""
    src = nl.ndarray(source_shape, nl.bfloat16, buffer)
    tt = TiledTensor(src, tile_size)
    for i in range(len(ops)):
        tt = apply_tiled_op(tt, ops[i][0], ops[i][1])
    view = tt.force_get_view()
    kernel_assert(
        len(view.shape) == len(expected_view_shape),
        f"force_get_view ndim: expected {len(expected_view_shape)}, got {len(view.shape)}",
    )
    for i in range(len(expected_view_shape)):
        kernel_assert(
            view.shape[i] == expected_view_shape[i],
            f"force_get_view shape[{i}]: expected {expected_view_shape[i]}, got {view.shape[i]}",
        )
    for i in range(len(expected_view_strides)):
        kernel_assert(
            view.strides[i] == expected_view_strides[i],
            f"force_get_view strides[{i}]: expected {expected_view_strides[i]}, got {view.strides[i]}",
        )
    kernel_assert(
        view.offset == expected_view_offset,
        f"force_get_view offset: expected {expected_view_offset}, got {view.offset}",
    )
    return dummy_out


@nki.jit
def kernel_test_get_view_negative(
    dummy_out,
    source_shape: tuple,
    tile_size: tuple,
    buffer: str,
    ops: tuple,
):
    """Test kernel that creates TiledTensor, applies ops, and calls get_view (expected to fail)."""
    src = nl.ndarray(source_shape, nl.bfloat16, buffer)
    tt = TiledTensor(src, tile_size)
    for i in range(len(ops)):
        tt = apply_tiled_op(tt, ops[i][0], ops[i][1])
    tt.get_view()
    return dummy_out


@nki.jit
def kernel_test_tiled_tensor_ops_negative(
    dummy_out,
    source_shape: tuple,
    tile_size: tuple,
    buffer: str,
    ops: tuple,
):
    """Test kernel that creates TiledTensor and applies ops (expected to fail during ops)."""
    src = nl.ndarray(source_shape, nl.bfloat16, buffer)
    tt = TiledTensor(src, tile_size)
    for i in range(len(ops)):
        tt = apply_tiled_op(tt, ops[i][0], ops[i][1])
    return dummy_out


@nki.jit
def kernel_test_construction_negative(
    dummy_out,
    source_shape: tuple,
    tile_size: tuple,
    buffer: str,
):
    """Test kernel for construction failures."""
    src = nl.ndarray(source_shape, nl.bfloat16, buffer)
    TiledTensor(src, tile_size)
    return dummy_out


@nki.jit
def kernel_test_from_tensor_view(
    dummy_out,
    source_shape: tuple,
    tile_size: tuple,
    buffer: str,
    view_ops: tuple,
    expected_grid_shape: tuple,
):
    """Test kernel that creates TiledTensor from a pre-processed NkiTensor."""
    src = nl.ndarray(source_shape, nl.bfloat16, buffer)
    for i in range(len(view_ops)):
        op_name = view_ops[i][0]
        args = view_ops[i][1]
        if op_name == "reshape_dim":
            dim, shape = args
            src = src.reshape_dim(dim, shape)
        elif op_name == "permute":
            (dims,) = args
            src = src.permute(dims)
        elif op_name == "slice":
            dim, start, end, step = args
            src = src.slice(dim, start, end, step)
    tt = TiledTensor(src, tile_size)
    grid_shape = tt.get_shape()
    for i in range(len(expected_grid_shape)):
        kernel_assert(
            grid_shape[i] == expected_grid_shape[i],
            f"grid shape[{i}]: expected {expected_grid_shape[i]}, got {grid_shape[i]}",
        )
    return dummy_out


# =============================================================================
# Helpers
# =============================================================================


def _make_dummy():
    return np.zeros((1,), dtype=np.float32)


def _run_trace_only(test_manager, platform_target, kernel_func, kernel_input):
    """Run a trace-only test using UnitTestFramework."""
    framework = UnitTestFramework(
        test_manager=test_manager,
        kernel_entry=kernel_func,
        kernel_input_generator=lambda _: kernel_input,
        trace_only=True,
    )
    framework.run_test(
        test_config=None,
        compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
    )


def run_shape_test(
    test_manager,
    platform_target,
    source_shape,
    tile_size,
    buffer,
    ops,
    expected_grid_shape,
    expected_view_shape,
    expected_view_strides,
    expected_view_offset,
):
    _run_trace_only(
        test_manager,
        platform_target,
        kernel_test_tiled_tensor_shape,
        {
            "dummy_out.must_alias_input": _make_dummy(),
            "source_shape": source_shape,
            "tile_size": tile_size,
            "buffer": buffer,
            "ops": ops,
            "expected_grid_shape": expected_grid_shape,
            "expected_view_shape": expected_view_shape,
            "expected_view_strides": expected_view_strides,
            "expected_view_offset": expected_view_offset,
        },
    )


def run_single_tile_test(
    test_manager,
    platform_target,
    source_shape,
    tile_size,
    buffer,
    grid_coords,
    expected_tile_shape,
    expected_tile_strides,
    expected_tile_offset,
):
    _run_trace_only(
        test_manager,
        platform_target,
        kernel_test_single_tile,
        {
            "dummy_out.must_alias_input": _make_dummy(),
            "source_shape": source_shape,
            "tile_size": tile_size,
            "buffer": buffer,
            "grid_coords": grid_coords,
            "expected_tile_shape": expected_tile_shape,
            "expected_tile_strides": expected_tile_strides,
            "expected_tile_offset": expected_tile_offset,
        },
    )


def run_get_view_test(
    test_manager,
    platform_target,
    source_shape,
    tile_size,
    buffer,
    ops,
    expected_view_shape,
    expected_view_strides,
    expected_view_offset,
):
    _run_trace_only(
        test_manager,
        platform_target,
        kernel_test_get_view,
        {
            "dummy_out.must_alias_input": _make_dummy(),
            "source_shape": source_shape,
            "tile_size": tile_size,
            "buffer": buffer,
            "ops": ops,
            "expected_view_shape": expected_view_shape,
            "expected_view_strides": expected_view_strides,
            "expected_view_offset": expected_view_offset,
        },
    )


def run_force_get_view_test(
    test_manager,
    platform_target,
    source_shape,
    tile_size,
    buffer,
    ops,
    expected_view_shape,
    expected_view_strides,
    expected_view_offset,
):
    _run_trace_only(
        test_manager,
        platform_target,
        kernel_test_force_get_view,
        {
            "dummy_out.must_alias_input": _make_dummy(),
            "source_shape": source_shape,
            "tile_size": tile_size,
            "buffer": buffer,
            "ops": ops,
            "expected_view_shape": expected_view_shape,
            "expected_view_strides": expected_view_strides,
            "expected_view_offset": expected_view_offset,
        },
    )


def run_negative_test(
    test_manager,
    platform_target,
    kernel_func,
    kernel_input,
    match,
):
    kernel_input["dummy_out.must_alias_input"] = _make_dummy()
    with pytest.raises(Exception, match=match):
        _run_trace_only(test_manager, platform_target, kernel_func, kernel_input)


# =============================================================================
# Positive Tests
# =============================================================================


@pytest_test_metadata(name="TiledTensor")
@pytest_marks(["tiled_tensor"])
@final
class TestTiledTensor:
    """Tests for TiledTensor compile-time validation."""

    # ----- Construction -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        # (source_shape, tile_size, expected_grid, view_shape, view_strides, view_offset)
        # No ops: get_view returns full source
        "source_shape,tile_size,expected_grid_shape,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # Exact division
            ((512, 1024), (128, 256), (4, 4), (512, 1024), (1024, 1), 0),
            ((128, 64), (128, 64), (1, 1), (128, 64), (64, 1), 0),
            ((256, 256, 256), (128, 128, 128), (2, 2, 2), (256, 256, 256), (256 * 256, 256, 1), 0),
            # Non-divisible (ceil) - get_view returns full source (clipped to actual shape)
            ((511, 1023), (128, 256), (4, 4), (511, 1023), (1023, 1), 0),
            ((129, 65), (128, 64), (2, 2), (129, 65), (65, 1), 0),
            ((1, 1), (128, 256), (1, 1), (1, 1), (1, 1), 0),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_construction(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        expected_grid_shape,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_shape_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            (),
            expected_grid_shape,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- Construction from NkiTensor -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,view_ops,tile_size,expected_grid_shape",
        [
            # NkiTensor with reshape_dim: (512, 1024) -> reshape_dim(1, (512, 2)) -> (512, 512, 2)
            ((512, 1024), (("reshape_dim", (1, (512, 2))),), (128, 256, 1), (4, 2, 2)),
            # NkiTensor with slice: (512, 1024) -> slice(0, 0, 256) -> (256, 1024)
            ((512, 1024), (("slice", (0, 0, 256, 1)),), (128, 256), (2, 4)),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_construction_from_tensor_view(
        self, test_manager, platform_target, source_shape, view_ops, tile_size, expected_grid_shape, buffer
    ):
        _run_trace_only(
            test_manager,
            platform_target,
            kernel_test_from_tensor_view,
            {
                "dummy_out.must_alias_input": _make_dummy(),
                "source_shape": source_shape,
                "tile_size": tile_size,
                "buffer": buffer,
                "view_ops": view_ops,
                "expected_grid_shape": expected_grid_shape,
            },
        )

    # ----- Single Tile Access -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        # (source_shape, tile_size, grid_coords, expected_shape, expected_strides, expected_offset)
        "source_shape,tile_size,grid_coords,expected_tile_shape,expected_tile_strides,expected_tile_offset",
        [
            # Source (512, 1024), strides (1024, 1)
            # Exact division: all tiles full size
            ((512, 1024), (128, 256), (0, 0), (128, 256), (1024, 1), 0),
            ((512, 1024), (128, 256), (3, 3), (128, 256), (1024, 1), 3 * 128 * 1024 + 3 * 256),
            ((512, 1024), (128, 256), (2, 1), (128, 256), (1024, 1), 2 * 128 * 1024 + 1 * 256),
            # Source (511, 1023), strides (1023, 1)
            # Non-divisible: last row/col tiles are clipped
            ((511, 1023), (128, 256), (0, 0), (128, 256), (1023, 1), 0),
            ((511, 1023), (128, 256), (3, 0), (127, 256), (1023, 1), 3 * 128 * 1023),  # last row
            ((511, 1023), (128, 256), (0, 3), (128, 255), (1023, 1), 3 * 256),  # last col
            ((511, 1023), (128, 256), (3, 3), (127, 255), (1023, 1), 3 * 128 * 1023 + 3 * 256),  # last row & col
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_single_tile(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        grid_coords,
        expected_tile_shape,
        expected_tile_strides,
        expected_tile_offset,
        buffer,
    ):
        run_single_tile_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            grid_coords,
            expected_tile_shape,
            expected_tile_strides,
            expected_tile_offset,
        )

    # ----- Select -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,expected_grid_shape,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # Select row 0: view (128, 1024), offset 0
            ((512, 1024), (128, 256), (("select", (0, 0)),), (4,), (128, 1024), (1024, 1), 0),
            # Select row 3: view (128, 1024), offset = 3*128*1024
            ((512, 1024), (128, 256), (("select", (0, 3)),), (4,), (128, 1024), (1024, 1), 3 * 128 * 1024),
            # Select col 2: view (512, 256), offset = 2*256
            ((512, 1024), (128, 256), (("select", (1, 2)),), (4,), (512, 256), (1024, 1), 2 * 256),
            # 3D: select dim 1 idx 0 -> view (256, 128, 256), offset 0
            ((256, 256, 256), (128, 128, 128), (("select", (1, 0)),), (2, 2), (256, 128, 256), (65536, 256, 1), 0),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_select(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_grid_shape,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_shape_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_grid_shape,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- Slice -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,expected_grid_shape,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # Slice first 2 rows: view (256, 1024), offset 0
            ((512, 1024), (128, 256), (("slice", (0, 0, 2)),), (2, 4), (256, 1024), (1024, 1), 0),
            # Slice middle columns [1,3): view (512, 512), offset = 1*256
            ((512, 1024), (128, 256), (("slice", (1, 1, 3)),), (4, 2), (512, 512), (1024, 1), 256),
            # Full range slice: view (512, 1024), offset 0
            ((512, 1024), (128, 256), (("slice", (0, 0, 4)),), (4, 4), (512, 1024), (1024, 1), 0),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_slice(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_grid_shape,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_shape_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_grid_shape,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- reshape_dim -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,expected_grid_shape,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # Split grid dim 0: still all tiles, view = full source
            ((512, 1024), (128, 256), (("reshape_dim", (0, (2, 2))),), (2, 2, 4), (512, 1024), (1024, 1), 0),
            # Split grid dim 1: still all tiles, view = full source
            ((512, 1024), (128, 256), (("reshape_dim", (1, (2, 2))),), (4, 2, 2), (512, 1024), (1024, 1), 0),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_reshape_dim(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_grid_shape,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_shape_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_grid_shape,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- flatten_dims -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,expected_grid_shape,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # Flatten (4, 4) -> (16,): still all tiles, view = full source
            ((512, 1024), (128, 256), (("flatten_dims", (0, 1)),), (16,), (512, 1024), (1024, 1), 0),
            # Reshape then flatten roundtrip: still all tiles
            (
                (512, 1024),
                (128, 256),
                (("reshape_dim", (0, (2, 2))), ("flatten_dims", (0, 1))),
                (4, 4),
                (512, 1024),
                (1024, 1),
                0,
            ),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_flatten_dims(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_grid_shape,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_shape_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_grid_shape,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- expand_dim / squeeze_dim -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,expected_grid_shape,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # expand at beginning: still all tiles
            ((512, 1024), (128, 256), (("expand_dim", (0,)),), (1, 4, 4), (512, 1024), (1024, 1), 0),
            # expand at end: still all tiles
            ((512, 1024), (128, 256), (("expand_dim", (2,)),), (4, 4, 1), (512, 1024), (1024, 1), 0),
            # expand then squeeze roundtrip: still all tiles
            ((512, 1024), (128, 256), (("expand_dim", (1,)), ("squeeze_dim", (1,))), (4, 4), (512, 1024), (1024, 1), 0),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_expand_squeeze(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_grid_shape,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_shape_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_grid_shape,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- reshape -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,expected_grid_shape,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # Flatten via reshape: still all tiles
            ((512, 1024), (128, 256), (("reshape", ((16,),)),), (16,), (512, 1024), (1024, 1), 0),
            # Reshape (4, 4) -> (2, 8): still all tiles
            ((512, 1024), (128, 256), (("reshape", ((2, 8),)),), (2, 8), (512, 1024), (1024, 1), 0),
            # Identity reshape: still all tiles
            ((512, 1024), (128, 256), (("reshape", ((4, 4),)),), (4, 4), (512, 1024), (1024, 1), 0),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_reshape(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_grid_shape,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_shape_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_grid_shape,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- get_view contiguous -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        # Source (512, 1024) has strides (1024, 1). Source (511, 1023) has strides (1023, 1).
        "source_shape,tile_size,ops,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # Full grid -> full source
            ((512, 1024), (128, 256), (), (512, 1024), (1024, 1), 0),
            # Select row 0 -> (128, 1024), offset 0
            ((512, 1024), (128, 256), (("select", (0, 0)),), (128, 1024), (1024, 1), 0),
            # Select row 2 -> (128, 1024), offset = 2*128*1024
            ((512, 1024), (128, 256), (("select", (0, 2)),), (128, 1024), (1024, 1), 2 * 128 * 1024),
            # Select col 0 -> (512, 256), offset 0
            ((512, 1024), (128, 256), (("select", (1, 0)),), (512, 256), (1024, 1), 0),
            # Slice first 2 rows -> (256, 1024), offset 0
            ((512, 1024), (128, 256), (("slice", (0, 0, 2)),), (256, 1024), (1024, 1), 0),
            # Select row 0 then slice cols [0,2) -> (128, 512), offset 0
            (
                (512, 1024),
                (128, 256),
                (("select", (0, 0)), ("slice", (0, 0, 2))),
                (128, 512),
                (1024, 1),
                0,
            ),
            # reshape + slice + reshape + slice -> squeeze -> get_view
            # Top-left quadrant: tile rows [0,1], tile cols [0,1] -> (256, 512), offset 0
            (
                (512, 1024),
                (128, 256),
                (
                    ("reshape_dim", (0, (2, 2))),
                    ("slice", (0, 0, 1)),
                    ("reshape_dim", (2, (2, 2))),
                    ("slice", (2, 0, 1)),
                    ("squeeze_dim", (0,)),
                    ("squeeze_dim", (1,)),
                ),
                (256, 512),
                (1024, 1),
                0,
            ),
            # reshape grid -> still full source, dim_map is -1 but memory is contiguous
            ((512, 1024), (128, 256), (("reshape", ((2, 8),)),), (512, 1024), (1024, 1), 0),
            # expand_dim -> inserts dim_map=-1 size-1 dim, still full source
            ((512, 1024), (128, 256), (("expand_dim", (1,)),), (512, 1024), (1024, 1), 0),
            # Non-divisible source: full grid
            ((511, 1023), (128, 256), (), (511, 1023), (1023, 1), 0),
            # Non-divisible: select last row -> offset = 3*128*1023
            ((511, 1023), (128, 256), (("select", (0, 3)),), (127, 1023), (1023, 1), 3 * 128 * 1023),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_get_view_contiguous(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_get_view_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- force_get_view -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # The spec example: reshape_dim + select creates a gap.
            # (4, 4) -> reshape_dim(0, (2, 2)) -> (2, 2, 4) -> select(1, 0) -> (2, 4)
            # Selects tile-rows 0 and 2 (rows [0,128) and [256,384)), skipping row 1.
            # force_get_view returns shape (2, 128, 1024):
            #   dim 0: 2 tile-rows with stride 256*1024 (= 2 tiles * 128 rows * 1024 cols)
            #   dim 1: 128 rows within each tile, stride 1024
            #   dim 2: 1024 cols, stride 1
            # offset = 0 (starts at beginning of source)
            (
                (512, 1024),
                (128, 256),
                (
                    ("reshape_dim", (0, (2, 2))),
                    ("select", (1, 0)),
                ),
                (2, 128, 1024),
                (256 * 1024, 1024, 1),
                0,
            ),
            # Contiguous case: select row 0 -> (128, 1024)
            # strides = (1024, 1), offset = 0
            (
                (512, 1024),
                (128, 256),
                (("select", (0, 0)),),
                (128, 1024),
                (1024, 1),
                0,
            ),
            # Contiguous case: select row 2 -> (128, 1024)
            # strides = (1024, 1), offset = 2*128*1024 = 262144
            (
                (512, 1024),
                (128, 256),
                (("select", (0, 2)),),
                (128, 1024),
                (1024, 1),
                2 * 128 * 1024,
            ),
            # Full grid -> (512, 1024)
            # strides = (1024, 1), offset = 0
            (
                (512, 1024),
                (128, 256),
                (),
                (512, 1024),
                (1024, 1),
                0,
            ),
            # reshape + select + reshape + select (checkerboard on 2D grid)
            # Selects tiles (0,0), (0,2), (2,0), (2,2) from a (4,4) grid.
            # Grid (2, 2) with strides (8, 2), dim_map (0, 1).
            # Dim 0: non-contiguous, tile_step=2, elem_step=256.
            #   reshape source dim 0 into (2, 256), slice inner to 128 -> (2, 128)
            # Dim 1: non-contiguous, tile_step=2, elem_step=512.
            #   reshape source dim 1 into (2, 512), slice inner to 256 -> (2, 256)
            # Result shape: (2, 128, 2, 256)
            # Source strides: (1024, 1). After reshape(0, (2, 256)): (256*1024, 1024, 1)
            # After slice(1, 0, 128): (256*1024, 1024, 1) with shape (2, 128, 1024)
            # After reshape(2, (2, 512)): (256*1024, 1024, 512, 1)
            # After slice(3, 0, 256): (256*1024, 1024, 512, 1) with shape (2, 128, 2, 256)
            (
                (512, 1024),
                (128, 256),
                (
                    ("reshape_dim", (0, (2, 2))),
                    ("select", (1, 0)),
                    ("reshape_dim", (1, (2, 2))),
                    ("select", (2, 0)),
                ),
                (2, 128, 2, 256),
                (256 * 1024, 1024, 512, 1),
                0,
            ),
            # 4D grid with strided selection on dims 0 and 2:
            # [0:4:2, :, 0:4:2, :] via reshape_dim + select on each strided dim.
            # Source (512, 512, 512, 512), tile (128, 128, 128, 128), grid (4, 4, 4, 4).
            #
            # reshape_dim(0, (2,2)) -> (2,2,4,4,4), select(1,0) -> (2,4,4,4) [stride-2 in dim 0]
            # reshape_dim(2, (2,2)) -> (2,4,2,2,4), select(3,0) -> (2,4,2,4) [stride-2 in dim 2]
            #
            # force_get_view builds:
            #   dim 0: non-contiguous -> reshape to (2, 128) -> view dims (2, 128)
            #   dim 1: contiguous 4 tiles -> view dim (512)
            #   dim 2: non-contiguous -> reshape to (2, 128) -> view dims (2, 128)
            #   dim 3: contiguous 4 tiles -> view dim (512)
            # Result shape: (2, 128, 512, 2, 128, 512)
            #
            # Source trivial strides: (512^3, 512^2, 512, 1) = (134217728, 262144, 512, 1)
            # After reshape_dim(0, (2, 256)): (256*134217728, 134217728, 262144, 512, 1)
            # After slice(1, 0, 128): shape (2, 128, 512, 512, 512)
            #   strides: (256*134217728, 134217728, 262144, 512, 1)
            # After slice(2, 0, 512): no-op
            # After slice(3, 0, 512): no-op
            # After reshape_dim(3, (2, 256)): (2, 128, 512, 2, 256, 512)
            #   strides: (256*134217728, 134217728, 262144, 256*512, 512, 1)
            # After slice(3, 0, 2): no-op
            # After slice(4, 0, 128): (2, 128, 512, 2, 128, 512)
            #   strides: (256*134217728, 134217728, 262144, 256*512, 512, 1)
            # After slice(5, 0, 512): no-op
            (
                (512, 512, 512, 512),
                (128, 128, 128, 128),
                (
                    ("reshape_dim", (0, (2, 2))),
                    ("select", (1, 0)),
                    ("reshape_dim", (2, (2, 2))),
                    ("select", (3, 0)),
                ),
                (2, 128, 512, 2, 128, 512),
                (256 * 134217728, 134217728, 262144, 256 * 512, 512, 1),
                0,
            ),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_force_get_view(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_force_get_view_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )

    # ----- Chained operations -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,expected_grid_shape,expected_view_shape,expected_view_strides,expected_view_offset",
        [
            # Select row 0 then slice cols [1,3): view (128, 512), offset = 256
            (
                (512, 1024),
                (128, 256),
                (("select", (0, 0)), ("slice", (0, 1, 3))),
                (2,),
                (128, 512),
                (1024, 1),
                256,
            ),
            # reshape_dim then select first half: view (256, 1024), offset 0
            (
                (512, 1024),
                (128, 256),
                (("reshape_dim", (0, (2, 2))), ("select", (0, 0))),
                (2, 4),
                (256, 1024),
                (1024, 1),
                0,
            ),
            # expand_dim then select then squeeze: same as select(0,0) -> (128, 1024)
            (
                (512, 1024),
                (128, 256),
                (("expand_dim", (0,)), ("select", (1, 0)), ("squeeze_dim", (0,))),
                (4,),
                (128, 1024),
                (1024, 1),
                0,
            ),
            # reshape + slice + reshape + slice (contiguous: top-left quadrant)
            # Tiles [0,1] x [0,1] -> view (256, 512), offset 0
            (
                (512, 1024),
                (128, 256),
                (
                    ("reshape_dim", (0, (2, 2))),
                    ("slice", (0, 0, 1)),
                    ("reshape_dim", (2, (2, 2))),
                    ("slice", (2, 0, 1)),
                ),
                (1, 2, 1, 2),
                (256, 512),
                (1024, 1),
                0,
            ),
        ],
    )
    @pytest.mark.parametrize("buffer", [nl.hbm])
    def test_chained_ops(
        self,
        test_manager,
        platform_target,
        source_shape,
        tile_size,
        ops,
        expected_grid_shape,
        expected_view_shape,
        expected_view_strides,
        expected_view_offset,
        buffer,
    ):
        run_shape_test(
            test_manager,
            platform_target,
            source_shape,
            tile_size,
            buffer,
            ops,
            expected_grid_shape,
            expected_view_shape,
            expected_view_strides,
            expected_view_offset,
        )


# =============================================================================
# Negative Tests
# =============================================================================


@pytest_marks(["tiled_tensor"])
@final
class TestTiledTensorNegative:
    """Negative tests for TiledTensor - expected to fail with clear error messages."""

    # ----- Construction failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,match",
        [
            # Wrong number of dims
            ((512, 1024), (128,), "dimensions"),
            ((512, 1024), (128, 256, 64), "dimensions"),
            # Zero tile size
            ((512, 1024), (0, 256), "positive"),
            ((512, 1024), (128, 0), "positive"),
            # Negative tile size
            ((512, 1024), (-1, 256), "positive"),
            ((512, 1024), (128, -128), "positive"),
        ],
    )
    def test_construction_negative(self, test_manager, platform_target, source_shape, tile_size, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_construction_negative,
            {
                "source_shape": source_shape,
                "tile_size": tile_size,
                "buffer": nl.hbm,
            },
            match,
        )

    # ----- Select failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "ops,match",
        [
            # Out of bounds index
            (
                (("select", (0, 5)),),
                "out of range",
            ),
            # Negative index
            (
                (("select", (0, -1)),),
                "non-negative",
            ),
            # Dim out of range
            (
                (("select", (5, 0)),),
                "out of range",
            ),
        ],
    )
    def test_select_negative(self, test_manager, platform_target, ops, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_tiled_tensor_ops_negative,
            {
                "source_shape": (512, 1024),
                "tile_size": (128, 256),
                "buffer": nl.hbm,
                "ops": ops,
            },
            match,
        )

    # ----- Slice failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "ops,match",
        [
            # start >= end
            (
                (("slice", (0, 3, 2)),),
                "greater than start",
            ),
            # start == end
            (
                (("slice", (0, 2, 2)),),
                "greater than start",
            ),
            # Out of bounds end
            (
                (("slice", (0, 0, 10)),),
                "out of range",
            ),
            # Negative start
            (
                (("slice", (0, -1, 2)),),
                "non-negative",
            ),
            # Dim out of range
            (
                (("slice", (5, 0, 1)),),
                "out of range",
            ),
        ],
    )
    def test_slice_negative(self, test_manager, platform_target, ops, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_tiled_tensor_ops_negative,
            {
                "source_shape": (512, 1024),
                "tile_size": (128, 256),
                "buffer": nl.hbm,
                "ops": ops,
            },
            match,
        )

    # ----- reshape_dim failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "ops,match",
        [
            # Product mismatch: grid dim 0 has size 4, but 2*3=6
            (
                (("reshape_dim", (0, (2, 3))),),
                "Product",
            ),
            # Dim out of range
            (
                (("reshape_dim", (5, (2, 2))),),
                "out of range",
            ),
        ],
    )
    def test_reshape_dim_negative(self, test_manager, platform_target, ops, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_tiled_tensor_ops_negative,
            {
                "source_shape": (512, 1024),
                "tile_size": (128, 256),
                "buffer": nl.hbm,
                "ops": ops,
            },
            match,
        )

    # ----- flatten_dims failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,match",
        [
            # start >= end
            (
                (512, 1024),
                (128, 256),
                (("flatten_dims", (1, 0)),),
                "less than end",
            ),
            # Dim out of range
            (
                (512, 1024),
                (128, 256),
                (("flatten_dims", (0, 5)),),
                "out of range",
            ),
            # Non-contiguous strides after reshape_dim + select that creates gaps
            (
                (512, 1024),
                (128, 256),
                (
                    ("reshape_dim", (0, (2, 2))),
                    ("select", (1, 0)),
                    ("flatten_dims", (0, 1)),
                ),
                "not contiguous",
            ),
        ],
    )
    def test_flatten_dims_negative(self, test_manager, platform_target, source_shape, tile_size, ops, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_tiled_tensor_ops_negative,
            {
                "source_shape": source_shape,
                "tile_size": tile_size,
                "buffer": nl.hbm,
                "ops": ops,
            },
            match,
        )

    # ----- squeeze_dim failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "ops,match",
        [
            # Not size 1
            (
                (("squeeze_dim", (0,)),),
                "size-1",
            ),
            # Dim out of range
            (
                (("squeeze_dim", (5,)),),
                "out of range",
            ),
        ],
    )
    def test_squeeze_dim_negative(self, test_manager, platform_target, ops, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_tiled_tensor_ops_negative,
            {
                "source_shape": (512, 1024),
                "tile_size": (128, 256),
                "buffer": nl.hbm,
                "ops": ops,
            },
            match,
        )

    # ----- expand_dim failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "ops,match",
        [
            # Dim way out of range
            (
                (("expand_dim", (10,)),),
                "out of range",
            ),
        ],
    )
    def test_expand_dim_negative(self, test_manager, platform_target, ops, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_tiled_tensor_ops_negative,
            {
                "source_shape": (512, 1024),
                "tile_size": (128, 256),
                "buffer": nl.hbm,
                "ops": ops,
            },
            match,
        )

    # ----- reshape failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "ops,match",
        [
            # Product mismatch
            (
                (("reshape", ((2, 4),)),),
                "Cannot reshape",
            ),
        ],
    )
    def test_reshape_negative(self, test_manager, platform_target, ops, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_tiled_tensor_ops_negative,
            {
                "source_shape": (512, 1024),
                "tile_size": (128, 256),
                "buffer": nl.hbm,
                "ops": ops,
            },
            match,
        )

    # ----- get_view non-contiguous failures -----

    @pytest.mark.trace_only
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "source_shape,tile_size,ops,match",
        [
            # The key negative case from spec: reshape_dim + select creates gap
            # (4, 4) -> reshape_dim(0, (2,2)) -> (2, 2, 4) -> select(1, 0) -> (2, 4)
            # Covers rows [0,128) and [256,384) — gap at [128,256)
            (
                (512, 1024),
                (128, 256),
                (
                    ("reshape_dim", (0, (2, 2))),
                    ("select", (1, 0)),
                ),
                "not contiguous|force_get_view",
            ),
        ],
    )
    def test_get_view_non_contiguous(self, test_manager, platform_target, source_shape, tile_size, ops, match):
        run_negative_test(
            test_manager,
            platform_target,
            kernel_test_get_view_negative,
            {
                "source_shape": source_shape,
                "tile_size": tile_size,
                "buffer": nl.hbm,
                "ops": ops,
            },
            match,
        )


# =============================================================================
# Simple Kernel Usage Examples
# =============================================================================


@nki.jit
def kernel_tiled_dma_copy(src_hbm):
    """Simple kernel: tile HBM src, DMA each tile through SBUF to a new output.

    Demonstrates the basic TiledTensor usage pattern:
      1. Tile the source
      2. Loop over tile grid
      3. Access individual tiles via get_tile
      4. Use tile NkiTensor with nisa.dma_copy
    """
    src_tiles = TiledTensor(src_hbm, tile_size=(128, 512))
    grid = src_tiles.get_shape()

    out = nl.ndarray(src_hbm.shape, dtype=src_hbm.dtype, buffer=nl.shared_hbm)
    dst_tiles = TiledTensor(out, tile_size=(128, 512))

    for i in range(grid[0]):
        for j in range(grid[1]):
            src_tile = src_tiles.get_tile((i, j))
            dst_tile = dst_tiles.get_tile((i, j))
            sbuf = nl.ndarray((128, 512), dtype=src_hbm.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sbuf, src=src_tile)
            nisa.dma_copy(dst=dst_tile, src=sbuf)

    return out


@nki.jit
def kernel_tiled_row_select_copy(src_hbm, row_idx: int):
    """Kernel: select a tile row via get_view and DMA the whole row at once.

    Demonstrates coalesced access: select a row of tiles and use get_view()
    to get a single contiguous NkiTensor covering all tiles in that row.
    Returns a (128, 1024) tensor containing that row.
    """
    src_tiles = TiledTensor(src_hbm, tile_size=(128, 256))
    row = src_tiles.select(0, row_idx)
    row_view = row.get_view()

    out = nl.ndarray((128, 1024), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    sbuf = nl.ndarray((128, 1024), dtype=src_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf, src=row_view)
    nisa.dma_copy(dst=out, src=sbuf)

    return out


@nki.jit
def kernel_tiled_reshape_select(src_hbm):
    """Kernel: reshape grid, select contiguous first half, copy first row.

    Demonstrates reshape_dim + select producing a contiguous tile block:
      (4, 4) -> reshape_dim(0, (2, 2)) -> (2, 2, 4) -> select(0, 0) -> (2, 4)
    This selects the first half of rows (rows 0-1), which is contiguous.
    Copies the first tile-row (128, 1024) of the selected half to output.
    """
    src_tiles = TiledTensor(src_hbm, tile_size=(128, 256))

    # Split row dim: (4, 4) -> (2, 2, 4)
    reshaped = src_tiles.reshape_dim(0, (2, 2))
    # Select first half of rows -> (2, 4), contiguous
    first_half = reshaped.select(0, 0)
    view = first_half.get_view()

    kernel_assert(view.shape[0] == 256, f"Expected 256, got {view.shape[0]}")
    kernel_assert(view.shape[1] == 1024, f"Expected 1024, got {view.shape[1]}")

    out = nl.ndarray((128, 1024), dtype=src_hbm.dtype, buffer=nl.shared_hbm)

    # DMA first tile-row of the selected half
    first_row_tiles = first_half.select(0, 0)
    first_row_view = first_row_tiles.get_view()
    sbuf = nl.ndarray((128, 1024), dtype=src_hbm.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=sbuf, src=first_row_view)
    nisa.dma_copy(dst=out, src=sbuf)

    return out


# =============================================================================
# Torch Reference Functions
# =============================================================================


@torch_ref_wrapper
def tiled_dma_copy_torch_ref(src_hbm):
    """Torch ref: full copy is identity."""
    return {"out": src_hbm}


@torch_ref_wrapper
def tiled_row_select_copy_torch_ref(src_hbm, row_idx):
    """Torch ref: extract tile row row_idx (each tile row is 128 elements high)."""
    start = row_idx * 128
    end = start + 128
    return {"out": src_hbm[start:end, :].contiguous()}


@torch_ref_wrapper
def tiled_reshape_select_torch_ref(src_hbm):
    """Torch ref: first tile-row of the first half = rows [0, 128)."""
    return {"out": src_hbm[0:128, 0:1024].contiguous()}


# =============================================================================
# Kernel Tests
# =============================================================================


@pytest_marks(["tiled_tensor"])
@final
class TestTiledTensorKernels:
    """Simple kernel tests using TiledTensor with DMA ops (compile-only)."""

    @pytest.mark.fast
    def test_tiled_dma_copy(self, test_manager, platform_target):
        np.random.seed(42)
        src = np.random.randn(512, 1024).astype(np.float32)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_tiled_dma_copy,
            torch_ref=tiled_dma_copy_torch_ref,
            kernel_input_generator=lambda _: {"src_hbm": src},
            output_tensor_descriptor=lambda _: {"out": np.zeros(src.shape, dtype=np.float32)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("row_idx", [0, 2, 3])
    def test_tiled_row_select_copy(self, test_manager, platform_target, row_idx):
        np.random.seed(42)
        src = np.random.randn(512, 1024).astype(np.float32)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_tiled_row_select_copy,
            torch_ref=tiled_row_select_copy_torch_ref,
            kernel_input_generator=lambda _: {"src_hbm": src, "row_idx": row_idx},
            output_tensor_descriptor=lambda _: {"out": np.zeros((128, 1024), dtype=np.float32)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )

    @pytest.mark.fast
    def test_tiled_reshape_select(self, test_manager, platform_target):
        np.random.seed(42)
        src = np.random.randn(512, 1024).astype(np.float32)

        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=kernel_tiled_reshape_select,
            torch_ref=tiled_reshape_select_torch_ref,
            kernel_input_generator=lambda _: {"src_hbm": src},
            output_tensor_descriptor=lambda _: {"out": np.zeros((128, 1024), dtype=np.float32)},
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=1),
        )
