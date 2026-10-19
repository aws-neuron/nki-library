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

"""Dense HBM layout for MXFP8 scales, and the DMAs that move it to and from SBUF.

One scale covers DATA_ROWS_PER_SCALE (8) consecutive data rows (partitions) of a feature.

SBUF: quantize_mx and matmul_mx keep the scale of data rows [8r, 8r + 8) of a tile at partition
``32 * (r // 4) + r % 4`` -- the first SCALE_ROWS_PER_QUADRANT partitions of each 32-partition
quadrant -- plus a ``4 * slot`` offset when SBUF scale packing shares one buffer between tiles.
The other partitions are unused.

HBM: scales are stored dense, one row per scale row, with no unused rows::

    scales_hbm[data_row // 8, f]          shape [K_PHYSICAL // 8, F] == [K // MX_PARTITION_SIZE, F]

A scale's HBM row depends only on its global data row, never on how a kernel tiled K. With the
1x32 layout, row r holds the scales of logical K [32 r, 32 r + 32), so the outputs of quantizing
any MX_PARTITION_SIZE-aligned K-shards concatenate along dim 0 into the output of quantizing the
whole tensor.

Loading writes every partition of the SBUF scale tile (the unused ones with don't-care rows);
storing takes one contiguous DMA per quadrant.
"""

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ....core.utils.kernel_assert import kernel_assert
from ._quantize_mxfp8_utils import MX_PARTITION_SIZE

DATA_ROWS_PER_SCALE = 8  # data partitions covered by one scale
SCALE_ROWS_PER_QUADRANT = MX_PARTITION_SIZE // DATA_ROWS_PER_SCALE  # 4 used partitions per quadrant


def get_dense_scale_shape(K: int, F: int) -> tuple[int, int]:
    """HBM scale shape for a quantized tensor of logical [K, F]: [K // MX_PARTITION_SIZE, F]."""
    kernel_assert(K % MX_PARTITION_SIZE == 0, f"K must be divisible by {MX_PARTITION_SIZE}, got {K=}")
    return (K // MX_PARTITION_SIZE, F)


def get_dense_scale_rows(num_data_rows: int) -> int:
    """Number of dense HBM scale rows covering num_data_rows physical data rows."""
    kernel_assert(
        num_data_rows % DATA_ROWS_PER_SCALE == 0,
        f"Data row count must be divisible by {DATA_ROWS_PER_SCALE}, got {num_data_rows=}",
    )
    return num_data_rows // DATA_ROWS_PER_SCALE


def assert_dense_scale_shape(data_shape: tuple, scales_shape: tuple, name: str = "operand") -> None:
    """Assert a 2D quantized HBM operand ([K_PHYSICAL, F'] data) carries dense [K_PHYSICAL // 8, F] scales."""
    expected_rows = get_dense_scale_rows(data_shape[0])
    kernel_assert(
        scales_shape[0] == expected_rows,
        f"{name}: MXFP8 scales in HBM must be dense [K_PHYSICAL // {DATA_ROWS_PER_SCALE}, F] = "
        f"[K // {MX_PARTITION_SIZE}, F], i.e. {expected_rows} rows for data shape {data_shape}; "
        f"got scales shape {scales_shape}",
    )


def _sbuf_flat_stride(sbuf_tensor: nl.NkiTensor) -> int:
    """Elements per partition of an SBUF tensor of any rank (all dims after the first)."""
    stride = 1
    for dim in sbuf_tensor.shape[1:]:
        stride *= dim
    return stride


def _quadrant_runs(num_data_rows: int) -> list[tuple[int, int]]:
    """(quadrant, scale rows in it) for a tile whose first num_data_rows data partitions are valid."""
    num_scale_rows = get_dense_scale_rows(num_data_rows)
    runs = []
    quadrant = 0
    while quadrant * SCALE_ROWS_PER_QUADRANT < num_scale_rows:
        runs.append((quadrant, min(SCALE_ROWS_PER_QUADRANT, num_scale_rows - quadrant * SCALE_ROWS_PER_QUADRANT)))
        quadrant += 1
    return runs


def load_dense_scales(
    scales_hbm: nl.NkiTensor,
    hbm_row: int,
    hbm_f: int,
    num_f: int,
    tile_rows: int,
    num_data_rows: int,
    scales_sbuf: nl.NkiTensor,
    sbuf_free_offset: int = 0,
    num_packed_tiles: int = 1,
    scalar_offset=None,
) -> None:
    """
    DMA the scales of consecutive data tiles from the dense HBM layout into the SBUF scale layout.

    Loads num_packed_tiles consecutive tiles of tile_rows data rows each into one SBUF scale
    tile, tile s at partition offset 4 * s (the scale-packing slot; num_packed_tiles=1 when SBUF
    scales are not packed).

    The matmul reads every partition of an SBUF scale tile, not only the used ones, so a full
    load defines all of them: a partial write inside a loop would make the buffer loop-carried.
    Per quadrant, each tile writes its 4 used partitions, and the last tile also runs on to the
    end of the quadrant with whatever dense rows follow (they are never used). Where those rows
    would leave the tensor -- or cannot be bounded at compile time because of a runtime
    scalar_offset -- the rest of the quadrant repeats one row instead (a zero-stride source).

    Args:
        scales_hbm (nl.NkiTensor): [K_PHYSICAL // 8, F] dense scales on HBM.
        hbm_row (int): Dense row of the first tile's first scale (its first data row // 8).
        hbm_f (int): First F column on HBM.
        num_f (int): F columns to move.
        tile_rows (int): Data rows (SBUF partitions) per tile; a multiple of 32, <= 128.
        num_data_rows (int): Valid data rows across the tiles; less than
            num_packed_tiles * tile_rows only at the tensor end.
        scales_sbuf (nl.NkiTensor): SBUF scale buffer of any rank; dim 0 is the partition.
        sbuf_free_offset (int): Flat free-dim offset of the SBUF scale tile (e.g.
            tile_or_group_idx * F + f for a [P, NUM_TILES, F] buffer).
        num_packed_tiles (int): Tiles sharing the SBUF scale tile (1 to 4).
        scalar_offset: Optional runtime row offset on HBM (e.g. per-expert indexing); applied
            with indirect_dim=0 on the hardware DGE.

    Notes:
        - One DMA per tile and quadrant (plus one where a quadrant tail is repeated), on the
          hardware DGE. A single DMA cannot do it: the dense row -> scale partition mapping is not
          affine, and dma_copy needs the HBM outer dimension to map affinely onto partitions.
        - At the tensor end (num_data_rows below a full load) the unused partitions repeat the
          last valid row instead.
    """
    kernel_assert(
        tile_rows % MX_PARTITION_SIZE == 0 and tile_rows <= MX_PARTITION_SIZE * SCALE_ROWS_PER_QUADRANT,
        f"tile_rows must be a multiple of {MX_PARTITION_SIZE} and <= 128, got {tile_rows=}",
    )
    kernel_assert(
        1 <= num_packed_tiles <= SCALE_ROWS_PER_QUADRANT,
        f"num_packed_tiles must be in [1, {SCALE_ROWS_PER_QUADRANT}], got {num_packed_tiles=}",
    )
    sbuf_stride = _sbuf_flat_stride(scales_sbuf)
    hbm_rows, hbm_f_dim = scales_hbm.shape
    rows_per_tile = get_dense_scale_rows(tile_rows)
    full_load = num_data_rows == num_packed_tiles * tile_rows
    num_quadrants = tile_rows // MX_PARTITION_SIZE

    def _copy(dst_partition, dst_partitions, src_row, src_row_step):
        # src_row_step is 1 for consecutive rows, 0 to repeat src_row on every partition.
        dst = scales_sbuf.ap(
            pattern=[[sbuf_stride, dst_partitions], [1, num_f]],
            offset=dst_partition * sbuf_stride + sbuf_free_offset,
        )
        src_pattern = [[src_row_step * hbm_f_dim, dst_partitions], [1, num_f]]
        src_offset = src_row * hbm_f_dim + hbm_f
        if scalar_offset is None:
            src = scales_hbm.ap(pattern=src_pattern, offset=src_offset)
        else:
            src = scales_hbm.ap(pattern=src_pattern, offset=src_offset, scalar_offset=scalar_offset, indirect_dim=0)
        # Required for performance: a scale tile takes several small DMAs, and the default
        # (software) DGE builds their descriptors on GpSimd, measured up to 30% slower on
        # pre-quantized matmuls. dge_mode.none was no better (up to 13% slower on fast-DGT operands).
        nisa.dma_copy(dst=dst, src=src, dge_mode=dge_mode.hwdge)

    if full_load:
        for tile in range(num_packed_tiles):
            tile_row = hbm_row + tile * rows_per_tile
            slot_partition = tile * SCALE_ROWS_PER_QUADRANT
            # The last tile also runs on to the end of each quadrant.
            last = tile == num_packed_tiles - 1
            for quadrant in range(num_quadrants):
                dst_partition = quadrant * MX_PARTITION_SIZE + slot_partition
                src_row = tile_row + quadrant * SCALE_ROWS_PER_QUADRANT
                width = MX_PARTITION_SIZE - slot_partition if last else SCALE_ROWS_PER_QUADRANT
                # Consecutive rows as far as they are known to stay in the tensor.
                readable = SCALE_ROWS_PER_QUADRANT if scalar_offset is not None else min(width, hbm_rows - src_row)
                _copy(dst_partition, readable, src_row, 1)
                if width > readable:
                    _copy(dst_partition + readable, width - readable, src_row, 0)
        return

    # Tensor end: the used partitions of the valid rows, then every remaining partition of the
    # scale tile repeats the last valid row. Adjacent repeated spans share one DMA.
    num_scale_rows = get_dense_scale_rows(num_data_rows)
    last_row = hbm_row + num_scale_rows - 1
    fill_start = 0
    for quadrant in range(num_quadrants):
        for tile in range(num_packed_tiles):
            rows = min(
                SCALE_ROWS_PER_QUADRANT,
                num_scale_rows - tile * rows_per_tile - quadrant * SCALE_ROWS_PER_QUADRANT,
                rows_per_tile - quadrant * SCALE_ROWS_PER_QUADRANT,
            )
            if rows <= 0:
                continue
            dst_partition = quadrant * MX_PARTITION_SIZE + tile * SCALE_ROWS_PER_QUADRANT
            if dst_partition > fill_start:
                _copy(fill_start, dst_partition - fill_start, last_row, 0)
            _copy(dst_partition, rows, hbm_row + tile * rows_per_tile + quadrant * SCALE_ROWS_PER_QUADRANT, 1)
            fill_start = dst_partition + rows
    if tile_rows > fill_start:
        _copy(fill_start, tile_rows - fill_start, last_row, 0)


def store_dense_scales(
    scales_hbm: nl.NkiTensor,
    hbm_row: int,
    hbm_f: int,
    num_f: int,
    num_data_rows: int,
    scales_sbuf: nl.NkiTensor,
    sbuf_partition: int = 0,
    sbuf_free_offset: int = 0,
    num_tiles: int = 1,
    dma_engines=None,
) -> None:
    """
    DMA one tile's scales from the SBUF scale layout into the dense HBM layout.

    The inverse of load_dense_scales; see it for the arguments. Only the used partitions are
    read, so the HBM tensor never receives the unused (uninitialised) SBUF partitions.

    num_tiles > 1 stores that many full (P_MAX data row) tiles held at consecutive slots of dim 1
    of a [P, slots, F] scales_sbuf, starting at sbuf_free_offset. In quadrant q, slot t's rows go
    to HBM rows hbm_row + 16 t + 4 q + [0, 4), so it is still one DMA per quadrant.
    dma_engines: optional engines whose hardware DGE queues issue the quadrant DMAs in turn.
    None keeps the default DGE. Required for num_tiles > 1.
    """
    sbuf_stride = _sbuf_flat_stride(scales_sbuf)
    if num_tiles > 1:
        kernel_assert(
            len(scales_sbuf.shape) == 3 and num_data_rows == nl.tile_size.pmax and dma_engines is not None,
            "num_tiles > 1 needs a [P, slots, F] buffer of full tiles and dma_engines",
        )
    hbm_f_dim = scales_hbm.shape[1]
    rows_per_tile = get_dense_scale_rows(num_data_rows)
    for quadrant, rows in _quadrant_runs(num_data_rows):
        hbm_start = hbm_row + quadrant * SCALE_ROWS_PER_QUADRANT
        sbuf_offset = (sbuf_partition + quadrant * MX_PARTITION_SIZE) * sbuf_stride + sbuf_free_offset
        if num_tiles == 1:
            dst = scales_hbm[nl.ds(hbm_start, rows), nl.ds(hbm_f, num_f)]
            src = scales_sbuf.ap(pattern=[[sbuf_stride, rows], [1, num_f]], offset=sbuf_offset)
        else:
            dst = scales_hbm.ap(
                pattern=[[hbm_f_dim, rows], [rows_per_tile * hbm_f_dim, num_tiles], [1, num_f]],
                offset=hbm_start * hbm_f_dim + hbm_f,
            )
            src = scales_sbuf.ap(
                pattern=[[sbuf_stride, rows], [scales_sbuf.shape[2], num_tiles], [1, num_f]], offset=sbuf_offset
            )
        if dma_engines is None:
            nisa.dma_copy(dst=dst, src=src)
        else:
            nisa.dma_copy(dst=dst, src=src, dge_mode=dge_mode.hwdge, engine=dma_engines[quadrant % len(dma_engines)])
