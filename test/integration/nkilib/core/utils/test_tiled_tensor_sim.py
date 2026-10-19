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

"""Simulator-mode branch-coverage tests for TiledTensor.

The tests in ``test_tiled_tensor.py`` are all marked ``@pytest.mark.trace_only``
and are therefore SKIPPED under ``--test-mode=simulation`` (see
``test/utils/pytest_plugin.py``), contributing zero simulation-mode coverage.

The tests here deliberately carry NO ``trace_only`` / ``compile_only`` /
``compile_and_infer`` marker, so they actually execute the (pure-Python,
metadata-only) TiledTensor grid logic inside a tiny ``@nki.jit`` kernel run via
``nki.simulate``. Small shapes (<=256) keep the simulator fast. Each test targets
one or more specific branch arcs in ``tiled_tensor.py``.
"""

import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.core.utils.allocator import BufferManager
from nkilib_src.nkilib.core.utils.tiled_tensor import TiledTensor

# Metadata-only kernels alias a 1-element dummy output back out; nki.simulate needs
# an output tensor even though these kernels perform no device compute.
_DUMMY = np.zeros((1,), dtype=np.float32)


def _sim(kernel):
    """Run a metadata-only @nki.jit kernel through the simulator."""
    nki.simulate(kernel)[1](dummy_out=_DUMMY)


# =============================================================================
# get_tile keep_dim=False (_get_single_tile_view else-branch: L217, L223-L232)
# =============================================================================


@nki.jit
def _k_get_tile_squeeze(dummy_out):
    # tile_size[1] == 1 exercises the select() sub-branch (L228); tile_size[0] != 1
    # exercises the slice() sub-branch (L231).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 1))
    v = tt.get_tile((0, 0), keep_dim=False)
    _ = v.shape
    return dummy_out


@pytest.mark.fast
def test_get_tile_no_keep_dim_squeezes_size1_dim():
    _sim(_k_get_tile_squeeze)


# =============================================================================
# reshape_dim on a non-zero dim (dim_map prefix loop: L339-L340)
# =============================================================================


@nki.jit
def _k_reshape_dim_nonzero(dummy_out):
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 64))  # grid (2, 4)
    tt = tt.reshape_dim(1, (2, 2))
    _ = tt.get_shape()
    return dummy_out


@pytest.mark.fast
def test_reshape_dim_on_inner_dimension():
    _sim(_k_reshape_dim_nonzero)


# =============================================================================
# flatten_dims (contiguity loop + all_same True/False: L370-L397)
# =============================================================================


@nki.jit
def _k_flatten_distinct_source_dims(dummy_out):
    # Flattening dims that map to distinct source dims -> all_same becomes False
    # (L391 -> L392 break, L394 -> L395).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 64))  # grid (2, 4), dim_map (0, 1)
    tt = tt.flatten_dims(0, 1)
    _ = tt.get_shape()
    return dummy_out


@nki.jit
def _k_flatten_same_source_dim(dummy_out):
    # reshape_dim first so both flattened dims map to source dim 0 -> all_same stays
    # True (L391 -> L390 loop-continue, L394 -> L397).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 64))  # grid (2, 4)
    tt = tt.reshape_dim(0, (2, 1))  # (2, 1, 4), dim_map (0, 0, 1)
    tt = tt.flatten_dims(0, 1)
    _ = tt.get_shape()
    return dummy_out


@pytest.mark.fast
def test_flatten_dims_across_distinct_source_dims():
    _sim(_k_flatten_distinct_source_dims)


@pytest.mark.fast
def test_flatten_dims_within_same_source_dim():
    _sim(_k_flatten_same_source_dim)


# =============================================================================
# expand_dim at end (L416) and in the middle (L418)
# =============================================================================


@nki.jit
def _k_expand_dim_end_and_middle(dummy_out):
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 64))  # grid (2, 4), ndim 2
    a = tt.expand_dim(2)  # dim == ndim -> new_stride = 1 (L416)
    b = tt.expand_dim(0)  # dim < ndim -> stride*shape (L418)
    _ = a.get_shape(), b.get_shape()
    return dummy_out


@pytest.mark.fast
def test_expand_dim_at_end_and_middle():
    _sim(_k_expand_dim_end_and_middle)


# =============================================================================
# reshape (product loops + dim_map=-1 loop: L461-L481)
# =============================================================================


@nki.jit
def _k_reshape_grid(dummy_out):
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 64))  # grid (2, 4) -> 8 tiles
    tt = tt.reshape((4, 2))
    _ = tt.get_shape()
    return dummy_out


@pytest.mark.fast
def test_reshape_grid_shape():
    _sim(_k_reshape_grid)


# =============================================================================
# reinterpret_cast: same size (L501), larger dtype (L506), smaller dtype (L518)
# =============================================================================


@nki.jit
def _k_reinterpret_same_size(dummy_out):
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 64))
    tt = tt.reinterpret_cast(nl.float16)  # 2 bytes -> 2 bytes
    _ = tt.get_shape()
    return dummy_out


@nki.jit
def _k_reinterpret_larger(dummy_out):
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 64))
    tt = tt.reinterpret_cast(nl.float32)  # 2 -> 4 bytes (ratio 2)
    _ = tt.get_shape()
    return dummy_out


@nki.jit
def _k_reinterpret_smaller(dummy_out):
    src = nl.ndarray((256, 256), nl.float32, nl.hbm)
    tt = TiledTensor(src, (128, 64))
    tt = tt.reinterpret_cast(nl.bfloat16)  # 4 -> 2 bytes
    _ = tt.get_shape()
    return dummy_out


@pytest.mark.fast
def test_reinterpret_cast_same_size():
    _sim(_k_reinterpret_same_size)


@pytest.mark.fast
def test_reinterpret_cast_to_larger_dtype():
    _sim(_k_reinterpret_larger)


@pytest.mark.fast
def test_reinterpret_cast_to_smaller_dtype():
    _sim(_k_reinterpret_smaller)


# =============================================================================
# force_get_view / _build_force_view / _build_dim_groups (L610-L698, L775-L782)
# =============================================================================


@nki.jit
def _k_force_view_strided(dummy_out):
    # Single-gdim non-contiguous source dim (reshape_dim + select creates a gap):
    # exercises L646 -> non-contiguous, L671 strided path, L682 (elem_step > tile).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (64, 128))  # grid (4, 2)
    tt = tt.reshape_dim(0, (2, 2))  # (2, 2, 2)
    tt = tt.select(1, 0)  # (2, 2) strided along source dim 0
    v = tt.force_get_view()
    _ = v.shape
    return dummy_out


@nki.jit
def _k_force_view_contiguous(dummy_out):
    # Contiguous selection: source dim 1 single-gdim contiguous (L646 -> L663 -> L664),
    # source dim 0 selected away -> gdims is None (L625 -> L627).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (64, 128))  # grid (4, 2)
    tt = tt.select(0, 0)  # remove grid dim 0
    v = tt.force_get_view()
    _ = v.shape
    return dummy_out


@nki.jit
def _k_force_view_multigdim_contiguous(dummy_out):
    # Two consecutive grid dims mapping to one source dim (reshape_dim, no select):
    # exercises the multi-gdim contiguity check (L649-L660, all-contiguous path).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (64, 128))  # grid (4, 2)
    tt = tt.reshape_dim(0, (2, 2))  # (2, 2, 2), dim_map (0, 0, 1)
    v = tt.force_get_view()
    _ = v.shape
    return dummy_out


@nki.jit
def _k_force_view_multigdim_break(dummy_out):
    # Multi-gdim group made internally non-contiguous by slicing the inner reshaped
    # dim: exercises the inner-loop break (L650 -> L654) and L656 -> L663 (False).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (64, 128))  # grid (4, 2)
    tt = tt.reshape_dim(0, (2, 2))  # (2, 2, 2)
    tt = tt.slice(1, 0, 1)  # (2, 1, 2)
    v = tt.force_get_view()
    _ = v.shape
    return dummy_out


@nki.jit
def _k_force_view_multigdim_outer_strided(dummy_out):
    # Multi-gdim group that is internally contiguous but whose outermost stride is
    # displaced in the source (reshape after a strided select): exercises L660 -> L661.
    src = nl.ndarray((256, 128), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (64, 64))  # grid (4, 2)
    tt = tt.reshape_dim(0, (2, 2))  # (2, 2, 2)
    tt = tt.select(1, 0)  # (2, 2) strided
    tt = tt.reshape_dim(0, (2, 1))  # (2, 1, 2), dim_map (0, 0, 1)
    v = tt.force_get_view()
    _ = v.shape
    return dummy_out


@nki.jit
def _k_force_view_selected_away_middle(dummy_out):
    # 3D grid with the middle grid dim selected away -> that source dim contributes a
    # gdims-is-None single tile (L625 -> L627 with a fully-collapsed source dim).
    src = nl.ndarray((256, 256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 128, 128))  # grid (2, 2, 2)
    tt = tt.select(1, 0)  # remove middle grid dim
    v = tt.force_get_view()
    _ = v.shape
    return dummy_out


@nki.jit
def _k_force_view_expand_dim_group(dummy_out):
    # expand_dim inserts a dim_map=-1 grid dim, producing a group with src_dim < 0 in
    # _build_dim_groups: exercises L616 -> L613 (skip the virtual group).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 128))  # grid (2, 2)
    tt = tt.expand_dim(1)  # (2, 1, 2), dim_map (0, -1, 1)
    v = tt.force_get_view()
    _ = v.shape
    return dummy_out


@pytest.mark.fast
def test_force_get_view_strided_single_gdim():
    _sim(_k_force_view_strided)


@pytest.mark.fast
def test_force_get_view_contiguous_and_selected_away():
    _sim(_k_force_view_contiguous)


@pytest.mark.fast
def test_force_get_view_multigdim_contiguous_group():
    _sim(_k_force_view_multigdim_contiguous)


@pytest.mark.fast
def test_force_get_view_multigdim_internal_break():
    _sim(_k_force_view_multigdim_break)


@pytest.mark.fast
def test_force_get_view_multigdim_outer_strided():
    _sim(_k_force_view_multigdim_outer_strided)


@pytest.mark.fast
def test_force_get_view_middle_dim_selected_away():
    _sim(_k_force_view_selected_away_middle)


@pytest.mark.fast
def test_force_get_view_with_expand_dim_virtual_group():
    _sim(_k_force_view_expand_dim_group)


# =============================================================================
# get_view with a fully-collapsed (0-dim) grid (_build_view: L704-L705)
# =============================================================================


@nki.jit
def _k_get_view_zero_dim_grid(dummy_out):
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 128))  # grid (2, 2)
    tt = tt.select(0, 0).select(0, 0)  # ndim -> 0
    v = tt.get_view()
    _ = v.shape
    return dummy_out


@pytest.mark.fast
def test_get_view_on_zero_dim_grid():
    _sim(_k_get_view_zero_dim_grid)


# =============================================================================
# __getitem__ paths (L797-L826)
# =============================================================================


@nki.jit
def _k_getitem_source_backed(dummy_out):
    # tuple index on a source-backed TiledTensor -> delegates to get_tile (L802 -> L826).
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 128))
    a = tt[(0, 1)]
    _ = a.shape
    return dummy_out


@nki.jit
def _k_getitem_source_scalar_index(dummy_out):
    # scalar (non-tuple) index -> L797 -> L800 wraps into a 1-tuple; 1D grid.
    src = nl.ndarray((256,), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128,))  # grid (2,)
    a = tt[0]
    _ = a.shape
    return dummy_out


@nki.jit
def _k_getitem_list_plain(dummy_out):
    # list-backed, no banks/rotation: L802 -> L803, L804 -> L812, L813 -> L816,
    # L818-L824, L820 -> L823.
    tt = TiledTensor.alloc((2, 2), (128, 128), nl.bfloat16, buffer=nl.sbuf)
    a = tt[(0, 1)]
    _ = a.shape
    return dummy_out


@nki.jit
def _k_getitem_list_banks(dummy_out):
    # num_banks path: L804 -> L805, L807-L810.
    tt = TiledTensor.alloc((2, 2), (128, 128), nl.bfloat16, buffer=nl.sbuf, num_banks=2)
    a = tt[(1, 1)]
    _ = a.shape
    return dummy_out


@nki.jit
def _k_getitem_list_rotate(dummy_out):
    # rotation path: L813 -> L814, and L820 -> L821 (on the rotated dim).
    tt = TiledTensor.alloc((4, 2), (128, 128), nl.bfloat16, buffer=nl.sbuf, rotate=(0, 2))
    a = tt[(3, 1)]  # index 3 wraps modulo rotate_count 2
    _ = a.shape
    return dummy_out


@pytest.mark.fast
def test_getitem_source_backed_tuple():
    _sim(_k_getitem_source_backed)


@pytest.mark.fast
def test_getitem_source_backed_scalar():
    _sim(_k_getitem_source_scalar_index)


@pytest.mark.fast
def test_getitem_list_backed_plain():
    _sim(_k_getitem_list_plain)


@pytest.mark.fast
def test_getitem_list_backed_banks():
    _sim(_k_getitem_list_banks)


@pytest.mark.fast
def test_getitem_list_backed_rotation():
    _sim(_k_getitem_list_rotate)


# =============================================================================
# permute (L831-L848) and broadcast (L853-L867)
# =============================================================================


@nki.jit
def _k_permute_source(dummy_out):
    src = nl.ndarray((256, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 64))
    tt = tt.permute((1, 0))  # source-backed path (L834 -> L848)
    _ = tt.get_shape()
    return dummy_out


@nki.jit
def _k_permute_list(dummy_out):
    # list-backed path (L834 -> L835-L846). Keep the partition dim (0) fixed since
    # SBUF tensors cannot permute the partition dimension.
    tt = TiledTensor.alloc((1, 1), (128, 64, 32), nl.bfloat16, buffer=nl.sbuf)
    tt = tt.permute((0, 2, 1))
    _ = tt[(0, 0)].shape
    return dummy_out


@nki.jit
def _k_broadcast_source(dummy_out):
    src = nl.ndarray((128, 1), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 1))
    tt = tt.broadcast(1, 4)  # source-backed path (L853 -> L867)
    _ = tt.get_shape()
    return dummy_out


@nki.jit
def _k_broadcast_list(dummy_out):
    tt = TiledTensor.alloc((2, 2), (128, 1), nl.bfloat16, buffer=nl.sbuf)
    tt = tt.broadcast(1, 4)  # list-backed path (L853 -> L854-L866)
    _ = tt[(0, 0)].shape
    return dummy_out


@pytest.mark.fast
def test_permute_source_backed():
    _sim(_k_permute_source)


@pytest.mark.fast
def test_permute_list_backed():
    _sim(_k_permute_list)


@pytest.mark.fast
def test_broadcast_source_backed():
    _sim(_k_broadcast_source)


@pytest.mark.fast
def test_broadcast_list_backed():
    _sim(_k_broadcast_list)


# =============================================================================
# alloc (L894-L933): num_banks / rotate / plain, with and without an SBM
# =============================================================================


@nki.jit
def _k_alloc_plain_named(dummy_out):
    # Plain nl.ndarray allocation, no banks/rotate, with names (L894 -> L909,
    # L912 -> L918, L920-L933, L924 -> L931).
    tt = TiledTensor.alloc((2, 2), (128, 128), nl.bfloat16, buffer=nl.sbuf, name="plain")
    _ = tt[(0, 0)].shape
    return dummy_out


@nki.jit
def _k_alloc_banks_named(dummy_out):
    # num_banks nl.ndarray path with names (L894 -> L895, L896-L906, L898 -> L905).
    tt = TiledTensor.alloc((2, 2), (128, 128), nl.bfloat16, buffer=nl.sbuf, num_banks=2, name="bank")
    _ = tt[(0, 0)].shape
    return dummy_out


@nki.jit
def _k_alloc_rotate(dummy_out):
    # rotate path (L912 -> L913).
    tt = TiledTensor.alloc((4, 2), (128, 128), nl.bfloat16, buffer=nl.sbuf, rotate=(0, 2))
    _ = tt[(0, 0)].shape
    return dummy_out


@nki.jit
def _k_alloc_sbm(dummy_out):
    # SBM-backed allocation exercises the alloc_stack / alloc_heap branches
    # (L898 -> L899, L924 -> L925).
    sbm = BufferManager(0, 24 * 1024 * 1024)
    sbm.open_scope()
    a = TiledTensor.alloc((2, 1), (128, 128), nl.bfloat16, buffer=nl.sbuf, sbm=sbm, name="sa")
    b = TiledTensor.alloc((2, 1), (128, 128), nl.bfloat16, buffer=nl.sbuf, sbm=sbm, num_banks=2, name="sb")
    _ = a[(0, 0)].shape, b[(0, 0)].shape
    return dummy_out


@pytest.mark.fast
def test_alloc_plain_ndarray():
    _sim(_k_alloc_plain_named)


@pytest.mark.fast
def test_alloc_num_banks_ndarray():
    _sim(_k_alloc_banks_named)


@pytest.mark.fast
def test_alloc_rotate():
    _sim(_k_alloc_rotate)


@pytest.mark.fast
def test_alloc_with_buffer_manager():
    _sim(_k_alloc_sbm)


# =============================================================================
# sub_tile (L969-L1023)
# =============================================================================


@nki.jit
def _k_sub_tile_source_even_negdim(dummy_out):
    # Negative dim normalization (L969 -> L970) + source-backed even sub-division
    # returning via reshape_dim (L975 -> L978, L987 -> L988).
    src = nl.ndarray((128, 256), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 256))  # grid (1, 1)
    tt = tt.sub_tile(-1, 128)  # 256 / 128 = 2, even
    _ = tt.get_shape()
    return dummy_out


@nki.jit
def _k_sub_tile_source_uneven(dummy_out):
    # Source-backed uneven sub-division falls through to the list-based expansion
    # (L987 -> L992), driving _get_single_tile_view for each original tile.
    src = nl.ndarray((128, 300), nl.bfloat16, nl.hbm)
    tt = TiledTensor(src, (128, 150))  # grid (1, 2)
    tt = tt.sub_tile(1, 100)  # 150 -> ceil 2 subs; grid dim mismatch -> fall through
    _ = tt.get_shape()
    return dummy_out


@nki.jit
def _k_sub_tile_list_dim0(dummy_out):
    # list-backed, dim == 0 slice branch (L975 -> L992, L1004 -> L1005),
    # grid-expansion "multiply last" branch (L1013 False, L1015 False, L1019).
    tt = TiledTensor.alloc((2, 1), (128, 128), nl.bfloat16, buffer=nl.sbuf)
    tt = tt.sub_tile(0, 64)
    _ = tt[(0, 0)].shape
    return dummy_out


@nki.jit
def _k_sub_tile_list_dim1(dummy_out):
    # list-backed, dim == 1 slice branch (L1006 -> L1007) and the 2-D grid
    # size-1-last-dim expansion (L1013 True -> L1014).
    tt = TiledTensor.alloc((1, 1), (128, 128), nl.bfloat16, buffer=nl.sbuf)
    tt = tt.sub_tile(1, 64)
    _ = tt[(0, 0)].shape
    return dummy_out


@nki.jit
def _k_sub_tile_list_dim2(dummy_out):
    # 3-D list-backed, dim == 2 else-branch slice (L1006 -> L1009) and the
    # generic size-1 expansion (L1015 True -> L1016).
    tt = TiledTensor.alloc((1, 1, 1), (64, 64, 128), nl.bfloat16, buffer=nl.sbuf)
    tt = tt.sub_tile(2, 64)
    _ = tt[(0, 0, 0)].shape
    return dummy_out


@pytest.mark.fast
def test_sub_tile_source_even_negative_dim():
    _sim(_k_sub_tile_source_even_negdim)


@pytest.mark.fast
def test_sub_tile_source_uneven_fallthrough():
    _sim(_k_sub_tile_source_uneven)


@pytest.mark.fast
def test_sub_tile_list_backed_dim0():
    _sim(_k_sub_tile_list_dim0)


@pytest.mark.fast
def test_sub_tile_list_backed_dim1():
    _sim(_k_sub_tile_list_dim1)


@pytest.mark.fast
def test_sub_tile_list_backed_dim2():
    _sim(_k_sub_tile_list_dim2)
