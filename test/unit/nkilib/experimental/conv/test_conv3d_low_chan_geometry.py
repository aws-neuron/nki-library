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

"""Low-channel Conv3D geometry tests."""

from dataclasses import replace
from typing import cast

import nki.language as nl
import pytest
from nkilib_src.nkilib.experimental.conv.conv3d import (
    BatchNormMode,
    ResidualAddLoc,
)
from nkilib_src.nkilib.experimental.conv.conv3d_bwd import (
    _dx_low_chan_supported,
    _finalize_dx_method,
    _low_chan_dx_padding,
)
from nkilib_src.nkilib.experimental.conv.conv3d_bwd_helpers import (
    BwdStrategy,
    Conv3dBwdConfig,
    DwFastVariant,
    DwUnitVariant,
    DwXLoadMethod,
    DxMethod,
)
from nkilib_src.nkilib.experimental.conv.conv3d_helpers import _build_conv3d_config
from nkilib_src.nkilib.experimental.conv.conv3d_kernels import (
    _build_low_chan_output_regions,
    _build_low_chan_output_tile_groups,
    _build_low_chan_spatial_runs,
    _compute_low_chan_unwritten_free_ranges,
    _get_low_chan_batch_capacity,
    _get_low_chan_input_window,
    _get_low_chan_input_window_reuse,
    _get_low_chan_psum_banks,
    _get_low_chan_replication,
    _get_low_chan_spatial_tile_limit,
    _get_low_chan_tap_access,
    _iter_low_chan_batch_groups,
    _iter_low_chan_spatial_tiles,
    _low_chan_result_arena_shape,
    _LowChanSpatialRun,
    _LowChanSpatialTile,
    _max_low_chan_output_region_size,
    _maximize_low_chan_working_sets,
    _use_low_chan_merged_dma,
    _use_low_chan_merged_input_dma,
)


class _ShapeOnlyTensor:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape
        self.dtype = nl.bfloat16


def _as_tensor(tensor: _ShapeOnlyTensor) -> nl.NkiTensor:
    return cast(nl.NkiTensor, tensor)


def _cfg(
    x_shape: tuple[int, int, int, int, int],
    filter_shape: tuple[int, int, int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    transpose_taps: bool = False,
):
    return _build_conv3d_config(
        x_in=_as_tensor(_ShapeOnlyTensor(x_shape)),
        filters=_as_tensor(_ShapeOnlyTensor(filter_shape)),
        bias=None,
        stride=stride,
        padding=padding,
        dilation=dilation,
        activation_fn=None,
        batch_norm_mode=BatchNormMode.NONE,
        output_pre_norm=False,
        residual_add_loc=ResidualAddLoc.NONE,
        transpose_taps=transpose_taps,
    )


def _bwd_cfg(
    *,
    c_out: int = 64,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (1, 3, 0, 2, 2, 1),
    dilation: tuple[int, int, int] = (2, 1, 2),
    strategy: BwdStrategy = BwdStrategy.PER_GRADIENT,
    dx_method: DxMethod = DxMethod.COL2IM,
) -> Conv3dBwdConfig:
    stride_d, stride_h, stride_w = stride
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    dilation_d, dilation_h, dilation_w = dilation
    return Conv3dBwdConfig(
        B=1,
        C_in=32,
        C_out=c_out,
        D=5,
        H=5,
        W=5,
        K_d=3,
        K_h=3,
        K_w=3,
        D_out=5,
        H_out=5,
        W_out=5,
        stride_d=stride_d,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_d_left=pad_d_left,
        pad_d_right=pad_d_right,
        pad_h_top=pad_h_top,
        pad_h_bottom=pad_h_bottom,
        pad_w_left=pad_w_left,
        pad_w_right=pad_w_right,
        dilation_d=dilation_d,
        dilation_h=dilation_h,
        dilation_w=dilation_w,
        compute_dx=True,
        compute_dw=False,
        compute_db=False,
        grid_ndim=1,
        n_prgs=1,
        prg_id=0,
        dtype_size=2,
        interleave={},
        dw_x_load_method=DwXLoadMethod.GATHER_TRANSPOSE,
        dx_method=dx_method,
        strategy=strategy,
        dw_fast_variant=DwFastVariant.GENERAL,
        dw_unit_variant=DwUnitVariant.BASELINE,
    )


def test_transpose_taps_swaps_logical_filter_channels():
    cfg = _cfg(
        x_shape=(2, 32, 4, 8, 8),
        filter_shape=(3, 3, 3, 96, 32),
        padding=(2, 0, 2, 0, 2, 0),
        transpose_taps=True,
    )

    assert cfg.C_in == 32
    assert cfg.C_out == 96
    assert (cfg.D_out, cfg.H_out, cfg.W_out) == (4, 8, 8)


def test_low_channel_dx_geometry_and_fallback_boundaries():
    cfg = _bwd_cfg()
    assert _low_chan_dx_padding(cfg) == (3, 1, 2, 0, 2, 3)
    assert _dx_low_chan_supported(cfg)

    assert not _dx_low_chan_supported(replace(cfg, C_out=65))
    assert not _dx_low_chan_supported(replace(cfg, stride_h=2))
    assert not _dx_low_chan_supported(replace(cfg, pad_h_top=3))


def test_low_channel_dx_method_requires_per_gradient_strategy():
    cfg = _bwd_cfg(
        c_out=32,
        padding=(1, 1, 1, 1, 1, 1),
        dilation=(1, 1, 1),
    )
    assert _finalize_dx_method(cfg) == DxMethod.LOW_CHAN
    assert _finalize_dx_method(replace(cfg, strategy=BwdStrategy.SHARE_DY_POINTWISE)) == DxMethod.COL2IM
    assert _finalize_dx_method(replace(cfg, C_out=65)) == DxMethod.COL2IM


@pytest.mark.parametrize(
    "channel_count,expected",
    [
        (1, (4, 32)),
        (32, (4, 32)),
        (33, (2, 64)),
        (64, (2, 64)),
        (65, (1, 128)),
        (128, (1, 128)),
    ],
)
def test_replication_uses_all_128_array_rows(channel_count, expected):
    assert _get_low_chan_replication(channel_count) == expected


@pytest.mark.parametrize(
    "c_in_repl,batch_per_program,expected",
    [
        (4, 6, 4),
        (4, 3, 3),
        (2, 3, 2),
        (1, 6, 1),
    ],
)
def test_batch_capacity_is_bounded_by_replication_and_local_batch(c_in_repl, batch_per_program, expected):
    assert _get_low_chan_batch_capacity(c_in_repl, batch_per_program) == expected


def test_result_arena_packs_all_local_batches_into_one_allocation():
    assert _low_chan_result_arena_shape(batch_capacity=3, result_band_size=2048) == (128, 3, 2048)


def test_batch_groups_preserve_ragged_final_group_for_per_batch_dma_fallback():
    groups = _iter_low_chan_batch_groups(batch_start=2, batch_end=8, batch_capacity=4)

    assert groups == [(2, 4), (6, 2)]
    assert _use_low_chan_merged_dma(c_out_rep=4, batch_count=4, batch_capacity=4)
    assert not _use_low_chan_merged_dma(c_out_rep=4, batch_count=2, batch_capacity=4)


@pytest.mark.parametrize("c_out_rep", [2, 4])
def test_merged_dma_requires_full_multi_batch_column_replication(c_out_rep):
    assert _use_low_chan_merged_dma(c_out_rep=c_out_rep, batch_count=3, batch_capacity=3)
    assert not _use_low_chan_merged_dma(c_out_rep=c_out_rep, batch_count=1, batch_capacity=1)


def test_replication_one_preserves_per_batch_dma_fallback():
    assert not _use_low_chan_merged_dma(c_out_rep=1, batch_count=4, batch_capacity=4)


@pytest.mark.parametrize(
    "c_in_repl,expected",
    [
        (4, 12 * 1024),
        (2, 8 * 1024),
        (1, 8 * 1024),
    ],
)
def test_spatial_tile_limit_is_tuned_per_input_replication(c_in_repl, expected):
    assert _get_low_chan_spatial_tile_limit(c_in_repl) == expected


@pytest.mark.parametrize(
    "c_in,c_in_stride,c_in_repl,batch_count",
    [
        (32, 32, 4, 4),
        (32, 32, 4, 2),
    ],
)
def test_merged_input_dma_requires_rep4_multiple_batches_with_unpadded_channels(
    c_in,
    c_in_stride,
    c_in_repl,
    batch_count,
):
    assert _use_low_chan_merged_input_dma(c_in, c_in_stride, c_in_repl, batch_count)


@pytest.mark.parametrize(
    "c_in,c_in_stride,c_in_repl,batch_count",
    [
        (32, 32, 4, 1),
        (16, 32, 4, 4),
        (33, 64, 2, 2),
        (64, 64, 2, 2),
    ],
)
def test_merged_input_dma_preserves_rep2_single_batch_and_padded_stride_fallbacks(
    c_in,
    c_in_stride,
    c_in_repl,
    batch_count,
):
    assert not _use_low_chan_merged_input_dma(c_in, c_in_stride, c_in_repl, batch_count)


def test_flat_spatial_tiles_cover_output_once_with_8k_cap():
    tiles = _iter_low_chan_spatial_tiles(total_spatial=17_013, spatial_tile_size=8 * 1024)

    assert [(tile.spatial_start, tile.spatial_end) for tile in tiles] == [
        (0, 8192),
        (8192, 16384),
        (16384, 17013),
    ]
    covered = [position for tile in tiles for position in range(tile.spatial_start, tile.spatial_end)]
    assert covered == list(range(17_013))


def test_spatial_runs_split_at_height_and_depth_boundaries():
    cfg = _cfg(
        x_shape=(1, 32, 2, 3, 5),
        filter_shape=(1, 1, 1, 32, 32),
    )
    runs = _build_low_chan_spatial_runs(cfg, spatial_start=13, spatial_end=23)

    assert [(run.d_out, run.h_out, run.w_start, run.w_end, run.free_start) for run in runs] == [
        (0, 2, 3, 5, 0),
        (1, 0, 0, 5, 2),
        (1, 1, 0, 3, 7),
    ]


def test_tap_access_clips_width_padding_and_preserves_stride():
    cfg = _cfg(
        x_shape=(1, 33, 7, 12, 17),
        filter_shape=(3, 3, 3, 33, 64),
        stride=(2, 2, 2),
        padding=(2, 3, 3, 2, 4, 1),
        dilation=(2, 2, 2),
    )
    run = _LowChanSpatialRun(d_out=0, h_out=0, w_start=0, w_end=9, free_start=0)

    assert _get_low_chan_tap_access(cfg, run, k_d=0, k_h=2, k_w=0) is None
    left = _get_low_chan_tap_access(cfg, run, k_d=1, k_h=2, k_w=0)
    right = _get_low_chan_tap_access(cfg, run, k_d=1, k_h=2, k_w=2)

    assert left is not None
    assert (left.input_flat_start, left.input_flat_end) == (17, 30)
    assert (left.output_free_start, left.output_free_end) == (2, 9)

    assert right is not None
    assert (right.input_flat_start, right.input_flat_end) == (17, 34)
    assert (right.output_free_start, right.output_free_end) == (0, 9)


def test_input_window_is_one_contiguous_flattened_span():
    cfg = _cfg(
        x_shape=(1, 32, 2, 4, 5),
        filter_shape=(1, 3, 3, 32, 32),
        padding=(0, 0, 1, 1, 1, 1),
    )
    spatial_tile = _LowChanSpatialTile(spatial_start=3, spatial_end=27)

    input_start, input_end = _get_low_chan_input_window(cfg, spatial_tile)

    assert (input_start, input_end) == (0, 33)


def test_consecutive_input_windows_reuse_the_overlapping_prefix():
    reuse = _get_low_chan_input_window_reuse(
        previous_input_start=100,
        previous_input_end=300,
        current_input_start=250,
        current_input_end=400,
        previous_working_set_idx=0,
        current_working_set_idx=1,
        working_set_count=2,
        batch_count=4,
        c_in_stride=32,
    )

    assert reuse is not None
    assert (
        reuse.previous_buffer_offset,
        reuse.prefix_length,
        reuse.fresh_input_start,
        reuse.fresh_input_end,
        reuse.active_partitions,
    ) == (150, 50, 300, 400, 128)


def test_consecutive_input_windows_skip_reuse_without_overlap():
    reuse = _get_low_chan_input_window_reuse(
        previous_input_start=0,
        previous_input_end=100,
        current_input_start=100,
        current_input_end=200,
        previous_working_set_idx=0,
        current_working_set_idx=1,
        working_set_count=2,
        batch_count=4,
        c_in_stride=32,
    )

    assert reuse is None


def test_consecutive_input_windows_skip_non_prefix_overlap():
    reuse = _get_low_chan_input_window_reuse(
        previous_input_start=100,
        previous_input_end=300,
        current_input_start=50,
        current_input_end=150,
        previous_working_set_idx=0,
        current_working_set_idx=1,
        working_set_count=2,
        batch_count=4,
        c_in_stride=32,
    )

    assert reuse is None


def test_consecutive_input_windows_support_full_containment_without_fresh_dma():
    reuse = _get_low_chan_input_window_reuse(
        previous_input_start=0,
        previous_input_end=200,
        current_input_start=50,
        current_input_end=150,
        previous_working_set_idx=0,
        current_working_set_idx=1,
        working_set_count=2,
        batch_count=2,
        c_in_stride=64,
    )

    assert reuse is not None
    assert (
        reuse.previous_buffer_offset,
        reuse.prefix_length,
        reuse.fresh_input_start,
        reuse.fresh_input_end,
        reuse.active_partitions,
    ) == (50, 100, 150, 150, 128)


def test_consecutive_input_windows_clip_reuse_for_ragged_final_window():
    reuse = _get_low_chan_input_window_reuse(
        previous_input_start=0,
        previous_input_end=9000,
        current_input_start=8192,
        current_input_end=9300,
        previous_working_set_idx=3,
        current_working_set_idx=4,
        working_set_count=5,
        batch_count=3,
        c_in_stride=32,
    )

    assert reuse is not None
    assert (
        reuse.previous_buffer_offset,
        reuse.prefix_length,
        reuse.fresh_input_start,
        reuse.fresh_input_end,
        reuse.active_partitions,
    ) == (8192, 808, 9000, 9300, 96)


@pytest.mark.parametrize(
    "batch_count,c_in_stride,expected_active_partitions",
    [
        (4, 32, 128),
        (3, 32, 96),
        (2, 64, 128),
        (1, 64, 64),
    ],
)
def test_consecutive_input_window_reuse_tracks_active_packed_batches(
    batch_count,
    c_in_stride,
    expected_active_partitions,
):
    reuse = _get_low_chan_input_window_reuse(
        previous_input_start=0,
        previous_input_end=100,
        current_input_start=50,
        current_input_end=150,
        previous_working_set_idx=0,
        current_working_set_idx=1,
        working_set_count=2,
        batch_count=batch_count,
        c_in_stride=c_in_stride,
    )

    assert reuse is not None
    assert reuse.active_partitions == expected_active_partitions


@pytest.mark.parametrize(
    "working_set_count,previous_working_set_idx,current_working_set_idx",
    [
        (1, 0, 0),
        (2, 1, 1),
    ],
)
def test_consecutive_input_window_reuse_requires_distinct_working_sets(
    working_set_count,
    previous_working_set_idx,
    current_working_set_idx,
):
    reuse = _get_low_chan_input_window_reuse(
        previous_input_start=0,
        previous_input_end=100,
        current_input_start=50,
        current_input_end=150,
        previous_working_set_idx=previous_working_set_idx,
        current_working_set_idx=current_working_set_idx,
        working_set_count=working_set_count,
        batch_count=4,
        c_in_stride=32,
    )

    assert reuse is None


def test_four_way_column_groups_interleave_0_4_8_12():
    spatial_tile = _LowChanSpatialTile(spatial_start=0, spatial_end=8192)
    regions = _build_low_chan_output_regions(spatial_tile, c_out_rep=4)
    groups = _build_low_chan_output_tile_groups(spatial_tile, c_out_rep=4)

    assert [(region.spatial_start, region.spatial_end) for region in regions] == [
        (0, 2048),
        (2048, 4096),
        (4096, 6144),
        (6144, 8192),
    ]
    assert [[chunk.spatial_start // 512 for chunk in group] for group in groups] == [
        [0, 4, 8, 12],
        [1, 5, 9, 13],
        [2, 6, 10, 14],
        [3, 7, 11, 15],
    ]
    assert [[chunk.free_offset for chunk in group] for group in groups] == [
        [0, 0, 0, 0],
        [512, 512, 512, 512],
        [1024, 1024, 1024, 1024],
        [1536, 1536, 1536, 1536],
    ]


def test_two_way_column_groups_interleave_0_and_8():
    spatial_tile = _LowChanSpatialTile(spatial_start=0, spatial_end=8192)
    groups = _build_low_chan_output_tile_groups(spatial_tile, c_out_rep=2)

    assert [chunk.spatial_start // 512 for chunk in groups[0]] == [0, 8]
    assert [chunk.spatial_start // 512 for chunk in groups[7]] == [7, 15]
    assert _max_low_chan_output_region_size(8192, 2) == 4096


def test_ragged_spatial_tail_keeps_regions_contiguous():
    spatial_tile = _LowChanSpatialTile(spatial_start=100, spatial_end=1400)
    regions = _build_low_chan_output_regions(spatial_tile, c_out_rep=4)
    groups = _build_low_chan_output_tile_groups(spatial_tile, c_out_rep=4)

    assert [(region.band_idx, region.spatial_start, region.spatial_end) for region in regions] == [
        (0, 100, 612),
        (1, 612, 1124),
        (2, 1124, 1400),
    ]
    assert len(groups) == 1
    assert [chunk.size() for chunk in groups[0]] == [512, 512, 276]
    assert _use_low_chan_merged_dma(c_out_rep=4, batch_count=3, batch_capacity=3)


def test_unwritten_ranges_track_fully_padded_positions_across_bands():
    cfg = _cfg(
        x_shape=(1, 32, 1, 32, 32),
        filter_shape=(1, 3, 3, 32, 32),
        padding=(0, 0, 3, 3, 3, 3),
    )
    spatial_tile = _LowChanSpatialTile(spatial_start=0, spatial_end=cfg.D_out * cfg.H_out * cfg.W_out)
    tile_group = _build_low_chan_output_tile_groups(spatial_tile, c_out_rep=4)[0]
    chunk_runs = [_build_low_chan_spatial_runs(cfg, chunk.spatial_start, chunk.spatial_end) for chunk in tile_group]

    unwritten_ranges = _compute_low_chan_unwritten_free_ranges(cfg, tile_group, chunk_runs)

    assert [(chunk.band_idx, chunk.spatial_start, chunk.spatial_end) for chunk in tile_group] == [
        (0, 0, 512),
        (1, 512, 1024),
        (2, 1024, 1296),
    ]
    unwritten_positions = {
        position for range_start, range_end in unwritten_ranges for position in range(range_start, range_end)
    }
    expected_positions = {
        h_out * cfg.W_out + w_out
        for h_out in range(cfg.H_out)
        for w_out in range(cfg.W_out)
        if h_out == 0 or h_out == cfg.H_out - 1 or w_out == 0 or w_out == cfg.W_out - 1
    }
    assert unwritten_positions == expected_positions


@pytest.mark.parametrize(
    "batch_count,expected",
    [
        (4, [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3]]),
        # A three-bank step visits every bank before repeating
        (3, [[0, 1, 2], [3, 4, 5], [6, 7, 0], [1, 2, 3], [4, 5, 6], [7, 0, 1], [2, 3, 4], [5, 6, 7], [0, 1, 2]]),
        (2, [[0, 1], [2, 3], [4, 5], [6, 7], [0, 1]]),
        (1, [[0], [1], [2], [3], [4], [5], [6], [7], [0]]),
    ],
)
def test_psum_bank_rotation_strides_by_batch_count(batch_count, expected):
    assert [
        _get_low_chan_psum_banks(rotation_idx=rotation_idx, batch_count=batch_count)
        for rotation_idx in range(len(expected))
    ] == expected


@pytest.mark.parametrize("batch_count", [1, 2, 3, 4])
def test_psum_bank_rotation_never_aliases_within_a_group(batch_count):
    for rotation_idx in range(16):
        banks = _get_low_chan_psum_banks(rotation_idx=rotation_idx, batch_count=batch_count)
        assert len(banks) == batch_count
        assert len(set(banks)) == batch_count


@pytest.mark.parametrize("batch_count", [1, 2, 3, 4])
def test_psum_bank_rotation_keeps_consecutive_groups_disjoint(batch_count):
    for rotation_idx in range(16):
        current = set(_get_low_chan_psum_banks(rotation_idx=rotation_idx, batch_count=batch_count))
        following = set(_get_low_chan_psum_banks(rotation_idx=rotation_idx + 1, batch_count=batch_count))
        assert current & following == set()


def test_working_set_planner_maximizes_complete_sets():
    working_sets = _maximize_low_chan_working_sets(
        budget=200,
        fixed_cost=10,
        filter_cost=30,
        working_set_cost=40,
        max_working_sets=10,
    )

    assert working_sets == 4


def test_working_set_planner_caps_at_iteration_count():
    working_sets = _maximize_low_chan_working_sets(
        budget=1000,
        fixed_cost=10,
        filter_cost=10,
        working_set_cost=10,
        max_working_sets=3,
    )

    assert working_sets == 3
