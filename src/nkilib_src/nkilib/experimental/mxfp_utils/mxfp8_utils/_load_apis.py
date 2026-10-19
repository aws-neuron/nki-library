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

"""Helper functions for loading tensor blocks from HBM to SBUF for MXFP8 kernels.

Provides four interleaving strategies for preparing data for MXFP8 quantization:

1. DGT (DMA Gather Transpose) — ``load_tile_dgt``
   Loads unswizzled [F, K] bf16 from HBM directly into interleaved SBUF layout
   via hardware gather-transpose.

2. FP32 reinterpret transpose — ``load_tile_PE_Swizzle_1x32``
   Loads [F, K] bf16 from HBM, reinterprets as fp32 to halve the partition count,
   then uses 2× nc_transpose + scalar-engine tensor_copy to interleave.
   Best for data coming from HBM (weights, activations).

3. BF16 xbar transpose — ``load_tile_bf16_xbar_transpose``
   Interleaves a [P_MAX, tile_f] bf16 tile already resident in SBUF using
   4× nc_transpose + tensor_copy with unscramble AP.
   Best for on-chip intermediates that should not be spilled to HBM.

4. K-by-F tensor-copy interleave — ``load_tile_tensor_copy_1x32``
   Loads consecutive K groups with DMA copies and interleaves their free axes
   using tensor_copy, preserving consecutive-K 1x32 quantization groups.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ....core.utils.kernel_assert import kernel_assert
from ._common_utils import get_active_sbm
from ._dense_scales import DATA_ROWS_PER_SCALE, load_dense_scales
from ._quantize_mxfp8_utils import (
    INTERLEAVE_FACTOR,
    L_TILE_K,
    MIN_F_FOR_QUANTIZATION,
    MIN_K_FOR_QUANTIZATION,
    MX_PARTITION_SIZE,
    Q_TILE_K,
    get_fp8_dtype_x4,
)
from .common_dataclasses import QuantScheme, TensorDescriptor, TensorOrientation, _TileLocation

P_MAX = 128  # 128, hardware partition dimension size
TRANSPOSE_CHUNK_SIZE = 32  # Element count per nc_transpose chunk
NUM_TRANSPOSE_CHUNKS = P_MAX // TRANSPOSE_CHUNK_SIZE  # Number of chunks in one partition
SUB_TILE = 128  # Max rows processed per sub-tile (matches P_MAX for partition alignment)
L_TILE_K_FP32 = L_TILE_K // 2  # 256, fp32 reinterpret halves the K dimension
BF16_TO_FP32_RATIO = 2  # Two bf16 elements pack into one fp32 element
# Sub-tiles folded into one bulk DMA. More than this at a nonzero offset trips the
# compiler's bounds checker (NCC_IBIR243), so we split into ceil(NUM_SUB_TILES / 4) DMAs.
MAX_SUBTILES_PER_DMA = 4

"""
FP32 PE swizzle constants.

Reinterpreting bf16 as fp32 halves the partition count, so stride-2 nc_transpose
with 2 passes is sufficient to interleave all partitions.
"""
FP32_INTERLEAVE_STRIDE = 2  # nc_transpose stride in fp32 space (reads/writes every other partition)
FP32_NUM_TRANSPOSE_PASSES = 2  # Number of nc_transpose passes needed in fp32 space

"""
BF16 xbar PE swizzle constants.

Working in native bf16 requires stride-4 reads and a 2-step shuffle
(nc_transpose + tensor_copy with unscramble) to achieve the full interleave.
"""
BF16_SRC_STRIDE = 4  # nc_transpose source stride (reads every 4th partition)
BF16_NUM_TRANSPOSE_PASSES = 4  # Number of nc_transpose passes needed in bf16 space
BF16_DST_STRIDE = 8  # nc_transpose destination stride (places into every 8th slot)
BF16_DST_GROUP_SIZE = 64  # Number of elements per destination group (P_MAX / BF16_TO_FP32_RATIO)
BF16_DST_PAIR_SIZE = 2  # Adjacent element pair grouping in destination AP
BF16_DST_OFFSET_MULTIPLIER = 2  # Offset step between passes (transpose_pass_idx * 2) to fill alternating pairs
BF16_UNSCRAMBLE_STRIDE = 2  # tensor_copy unscramble: stride between interleaved groups
BF16_UNSCRAMBLE_GROUP_SIZE = 4  # tensor_copy unscramble: elements per group (INTERLEAVE_FACTOR)


def load_tile(
    load_loc: _TileLocation,
    data_store_loc: Optional[_TileLocation] = None,
    scale_store_loc: Optional[_TileLocation] = None,
    num_packed_scale_tiles: int = 1,
    k_end: Optional[int] = None,
    use_strided_copy_swizzle_1x32: bool = False,
    dma_engine=None,
) -> None:
    """
    Load a tile from HBM to SBUF, dispatching by tensor format.

    Routes to the appropriate loader based on tensor metadata:
    1. MXFP8 (has scales): load data via dma_copy and the dense HBM scales into the SBUF
       scale layout (see dense_scales.py)
    2. Pre-swizzled bf16/fp16: load data via dma_copy
    3. Unswizzled 1x32: PE swizzle or K-by-F tensor-copy interleave.
    4. Other unswizzled bf16/fp16: PE swizzle or DMA gather transpose (DGT).

    Args:
        load_loc (_TileLocation): Source tile location in HBM.
        data_store_loc (Optional[_TileLocation]): Destination in SBUF for data. For MXFP8,
            None skips the data load.
        scale_store_loc (Optional[_TileLocation]): Destination in SBUF for MXFP8 scales; its
            f_offset is the flat free-dim offset of the SBUF scale tile. None skips the scale load.
        num_packed_scale_tiles (int): MXFP8 only. Load the scales of this many consecutive
            tiles (starting at load_loc) into one packed SBUF scale tile, tile s in slot s.
            Default 1 (unpacked).
        k_end (Optional[int]): Swizzled/MXFP8 only. Absolute K row (in load_loc.k_offset units)
            at which data and scales stop being read, e.g. a K-slice end. None reads to the
            tensor (or per-expert) end.
        use_strided_copy_swizzle_1x32: The caller's selection for this logical tile.
            Applies only to unswizzled 1x32 K-by-F BF16/FP16 inputs.
        dma_engine: Unswizzled 1x32 only (F-by-K PE loader, K-by-F tensor-copy loader).
            Issue the HBM loads on the hardware DGE of this engine (nisa.engine.sync or
            nisa.engine.scalar) instead of letting the compiler pick the DGE, which may build
            the descriptors on GpSimd. None keeps the default; any other loader rejects it.

    Returns:
        None

    Pseudocode:
        if tensor has scales (MXFP8):
            dma_copy data to data_store_loc
            dma_copy dense scale rows into the SBUF scale partitions of scale_store_loc
        elif tensor is pre-swizzled:
            dma_copy data to data_store_loc
        elif tensor uses 1x32:
            select PE swizzle or K-by-F tensor-copy interleave
        else:
            load using PE swizzle or DMA gather transpose
    """
    is_unswizzled_1x32 = (
        not load_loc.tensor.is_quantized
        and not load_loc.tensor.is_swizzled
        and load_loc.tensor.quant_scheme == QuantScheme._1x32
    )
    kernel_assert(
        dma_engine is None or is_unswizzled_1x32,
        "dma_engine is only supported for unswizzled 1x32 loads.",
    )
    # MXFP8, preswizzled. Load data and scales separately using dma_copy
    if load_loc.tensor.is_quantized:
        kernel_assert(not load_loc.tensor.is_f_by_k, "The tensor needs to be K by F.")
        if data_store_loc != None:
            _load_tile_dma_copy(load_loc, data_store_loc, k_end)
        if scale_store_loc != None:
            _load_tile_dense_scales(load_loc, scale_store_loc, num_packed_scale_tiles, k_end)
    # bf16, preswizzled. Load as is using dma_copy
    elif load_loc.tensor.is_swizzled:
        kernel_assert(not load_loc.tensor.is_f_by_k, "The tensor needs to be K by F.")
        kernel_assert(load_loc.tensor.data.dtype == nl.bfloat16, "Input tensor must be bfloat16.")
        _load_tile_dma_copy(load_loc, data_store_loc, k_end)
    # Unswizzled 1x32: choose the PE or tensor-copy loader here.
    elif load_loc.tensor.quant_scheme == QuantScheme._1x32:
        use_tensor_copy = not load_loc.tensor.is_f_by_k and use_strided_copy_swizzle_1x32
        if use_tensor_copy:
            load_tile_tensor_copy_1x32(load_loc, data_store_loc, dma_engine=dma_engine)
        else:
            load_tile_PE_Swizzle_1x32(load_loc, data_store_loc, dma_engine=dma_engine)
    # bf16/fp16, unswizzled with PE swizzle: use PE transpose (indirect or direct
    # determined by vector_offset presence inside load_tile_PE_swizzle_wrapX).
    elif load_loc.tensor.uses_pe_swizzle:
        load_tile_PE_swizzle_wrapX(load_loc, data_store_loc)
    # bf16/fp16, unswizzled with direct dma loads use DGT.
    else:
        load_tile_dgt(load_loc, data_store_loc)


def _get_dma_copy_ap(
    tensor: nl.NkiTensor,
    k_offset: int,
    f_offset: int,
    num_k: int,
    num_f: int,
) -> nl.NkiTensor:
    """
    Compute a BIR access pattern and flat offset for a dma_copy destination.

    Uses .ap() addressing so the tensor can be 2D or 3D (or any shape).
    Flattens all dimensions after the first into a single stride, then
    applies a linear offset from (k_offset, f_offset).

    Args:
        tensor (nl.NkiTensor): Destination SBUF tensor.
        k_offset (int): Offset in the K (first) dimension.
        f_offset (int): Offset in the flattened remaining dimensions.
        num_k (int): Number of K elements to write.
        num_f (int): Number of F elements to write.

    Returns:
        nl.NkiTensor: The tensor with .ap() applied.

    Pseudocode:
        stride_k = product(tensor.shape[1:])
        offset = k_offset * stride_k + f_offset
        return tensor.ap(pattern=[[stride_k, num_k], [1, num_f]], offset=offset)
    """
    stride_k = 1
    for dim in tensor.shape[1:]:
        stride_k *= dim
    offset = k_offset * stride_k + f_offset
    return tensor.ap(pattern=[[stride_k, num_k], [1, num_f]], offset=offset)


def _load_tile_dma_copy(
    load_loc: _TileLocation,
    store_loc: Optional[_TileLocation] = None,
    k_end: Optional[int] = None,
) -> _TileLocation:
    """
    Load a data tile from HBM to SBUF using dma_copy.

    Unified loader for MXFP8 data and pre-swizzled bf16 tiles (MXFP8 scales go through
    _load_tile_dense_scales).
    For non-x4 MXFP8 data, uses BIR AP to reinterpret as x4 dtype during load.
    Handles boundary tiles by clamping to tensor bounds.

    Args:
        load_loc (_TileLocation): Source tile location with tensor, tile dimensions, and offsets.
        store_loc (Optional[_TileLocation]): Destination in SBUF. If None, allocated automatically.
        k_end (Optional[int]): Absolute K row at which the read stops (see load_tile).

    Returns:
        store_loc (_TileLocation): The store location with loaded tensor in SBUF.

    Pseudocode:
        determine src tensor and its x4-packed width
        clamp num_k, num_f to tensor boundary
        if out of bounds: return store_loc unchanged
        allocate dst SBUF if store_loc is None
        if non-x4 MXFP8 data:
            dma_copy with BIR AP reinterpretation to x4 dtype
        else:
            dma_copy with direct slice indexing
        return store_loc
    """
    sbm = get_active_sbm()

    is_x4 = load_loc.tensor.is_x4

    kernel_assert(not load_loc.tensor.is_f_by_k, "The source tensor should be F by K.")
    kernel_assert(
        load_loc.tensor.f_scalar_offset is None,
        "f_scalar_offset is only supported for unswizzled F-by-K operands, not the "
        "pre-swizzled/quantized K-by-F dma_copy path",
    )

    src = load_loc.tensor.data
    K = src.shape[0]
    # For non-x4 MXFP8, use scales' F dimension (x4-packed width)
    F = load_loc.tensor.scales.shape[1] if (load_loc.tensor.is_quantized and not is_x4) else src.shape[1]
    dst_dtype = src.dtype if (not load_loc.tensor.is_quantized or is_x4) else get_fp8_dtype_x4(str(src.dtype))

    # Clamp to boundary for partial tiles.
    # When effective_f_dim is set (stacked-expert tensors with scalar_offset),
    # clamp K to the per-expert boundary instead of the full tensor K.
    if load_loc.tensor.effective_f_dim is not None:
        K_effective = load_loc.tensor.effective_f_dim
    else:
        K_effective = K
    if k_end is not None:
        K_effective = min(K_effective, k_end)
    num_k = min(load_loc.tile_k, K_effective - load_loc.k_offset)
    num_f = min(load_loc.tile_f, F - load_loc.f_offset)
    # Skip entirely out-of-bounds tiles (masking case where SBUF is pre-zeroed)
    if num_k <= 0 or num_f <= 0:
        return store_loc

    # Allocate destination SBUF tensor if not provided
    if store_loc == None:
        dst_sbuf = sbm.alloc_stack(shape=(num_k, num_f), dtype=dst_dtype, buffer=nl.sbuf)
        if load_loc.tensor.is_quantized:
            store_loc = _TileLocation(
                tensor=TensorDescriptor(data=dst_sbuf, is_x4=is_x4, orientation=TensorOrientation.K_BY_F),
                tile_k=num_k,
                tile_f=num_f,
            )
        else:
            store_loc = _TileLocation(
                tensor=TensorDescriptor(data=dst_sbuf, is_swizzled=True, orientation=TensorOrientation.K_BY_F),
                tile_k=num_k,
                tile_f=num_f,
            )

    dst_ap = _get_dma_copy_ap(store_loc.tensor.data, store_loc.k_offset, store_loc.f_offset, num_k, num_f)

    # Runtime scalar_offset (e.g., per-expert indexing for MoE)
    active_scalar_offset = load_loc.tensor.scalar_offset
    use_scalar_offset = active_scalar_offset is not None

    # Non-x4 MXFP8 data: use BIR AP to reinterpret as x4 during load
    if load_loc.tensor.is_quantized and not is_x4:
        x4_f_dim = load_loc.tensor.scales.shape[1]
        global_offset = load_loc.k_offset * x4_f_dim + load_loc.f_offset
        if use_scalar_offset:
            nisa.dma_copy(
                src=src.ap(
                    pattern=[[x4_f_dim, num_k], [1, num_f]],
                    offset=global_offset,
                    dtype=dst_dtype,
                    scalar_offset=active_scalar_offset,
                    indirect_dim=0,
                ),
                dst=dst_ap,
                dge_mode=dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(
                src=src.ap(
                    pattern=[[x4_f_dim, num_k], [1, num_f]],
                    offset=global_offset,
                    dtype=dst_dtype,
                ),
                dst=dst_ap,
            )
    else:
        if use_scalar_offset:
            # Cannot use nl.ds slicing with scalar_offset; must use .ap() addressing.
            # src shape is [K, F]; stride for dim 0 (K) is F.
            src_f_dim = src.shape[1]
            static_offset = load_loc.k_offset * src_f_dim + load_loc.f_offset
            nisa.dma_copy(
                src=src.ap(
                    pattern=[[src_f_dim, num_k], [1, num_f]],
                    offset=static_offset,
                    scalar_offset=active_scalar_offset,
                    indirect_dim=0,
                ),
                dst=dst_ap,
                dge_mode=dge_mode.hwdge,
            )
        else:
            nisa.dma_copy(
                src=src[nl.ds(load_loc.k_offset, num_k), nl.ds(load_loc.f_offset, num_f)],
                dst=dst_ap,
            )

    return store_loc


def _load_tile_dense_scales(
    load_loc: _TileLocation, store_loc: _TileLocation, num_packed_tiles: int = 1, k_end: Optional[int] = None
) -> None:
    """
    Load the scales of MXFP8 tiles from the dense HBM layout into the SBUF scale layout.

    The tiles cover physical data rows [k_offset, k_offset + num_packed_tiles * tile_k) of
    load_loc.tensor; their scales are the dense rows from k_offset // 8. Rows past the tensor (or
    past the per-expert effective_f_dim) are skipped like the data load, leaving the pre-zeroed SBUF.

    Args:
        load_loc (_TileLocation): Source tile; tensor is a quantized HBM TensorDescriptor with
            dense [K_PHYSICAL // 8, F] scales. k_offset / tile_k are in physical data rows.
        store_loc (_TileLocation): SBUF destination; tensor.scales is the SBUF scale buffer and
            f_offset the flat free offset of the scale tile.
        num_packed_tiles (int): Consecutive tiles packed into the scale tile (1 = unpacked).
        k_end (Optional[int]): Absolute data row at which the read stops, so the scales stop with
            the data (see load_tile).

    Returns:
        None
    """
    td = load_loc.tensor
    K_data = td.effective_f_dim if td.effective_f_dim is not None else td.data.shape[0]
    if k_end is not None:
        K_data = min(K_data, k_end)
    num_k = min(num_packed_tiles * load_loc.tile_k, K_data - load_loc.k_offset)
    num_f = min(load_loc.tile_f, td.scales.shape[1] - load_loc.f_offset)
    if num_k <= 0 or num_f <= 0:
        return

    kernel_assert(
        td.scalar_offset is None or td.scalar_offset_scales is not None,
        "A quantized operand with a runtime scalar_offset needs scalar_offset_scales: its dense scale "
        "rows are offset by data_row // 8, not by the data row offset",
    )
    kernel_assert(
        load_loc.k_offset % DATA_ROWS_PER_SCALE == 0,
        f"A quantized tile must start on a {DATA_ROWS_PER_SCALE}-row scale boundary, got k_offset={load_loc.k_offset}",
    )
    load_dense_scales(
        td.scales,
        hbm_row=load_loc.k_offset // DATA_ROWS_PER_SCALE,
        hbm_f=load_loc.f_offset,
        num_f=num_f,
        tile_rows=load_loc.tile_k,
        num_data_rows=num_k,
        scales_sbuf=store_loc.tensor.scales,
        sbuf_free_offset=store_loc.f_offset,
        num_packed_tiles=num_packed_tiles,
        scalar_offset=td.scalar_offset_scales,
    )


def load_tile_dgt(
    load_loc: _TileLocation,
    store_loc: Optional[_TileLocation] = None,
) -> _TileLocation:
    """
    Load an unswizzled bf16 tile from HBM to SBUF using DMA Gather Transpose.

    Flattens the source [F, K] tensor, applies vector offsets to gather the tile,
    and transposes into interleaved SBUF layout [K/4, 1, 1, tile_f*4].

    Dimensions:
        F: Source tensor first dimension (feature/free dimension)
        K: Source tensor second dimension (partition dimfension)

    Args:
        load_loc (_TileLocation): Source tile location with tensor, tile dimensions, and offsets.
            If access_pattern and vector_offset are None, they are generated automatically.
        store_loc (Optional[_TileLocation]): Destination in SBUF. If None, allocated automatically.

    Returns:
        store_loc (_TileLocation): The store location with the loaded data tensor in SBUF.

    Pseudocode:
        validate input is F-by-K and bf16/fp16
        generate access_pattern and vector_offset if not provided
        allocate dst SBUF [tile_k/INTERLEAVE_FACTOR, 1, 1, tile_f*INTERLEAVE_FACTOR] if needed
        flatten src [F, K] -> [F*K/P_MAX, 1, 1, P_MAX]
        dma_transpose(dst, src.ap(access_pattern, vector_offset), axes=(3,1,2,0))
    """

    sbm = get_active_sbm()

    src_data = load_loc.tensor.data
    F, K = src_data.shape

    # DGT (fast or legacy) is a DMA gather-transpose, so it cannot honor disable_dma_transpose.
    kernel_assert(
        not load_loc.tensor.disable_dma_transpose,
        "disable_dma_transpose is unsupported for the DGT loader; DGT inherently uses a DMA transpose.",
    )

    # Ensure input is F-by-K and 16-bit dtype
    kernel_assert(load_loc.tensor.is_f_by_k, "Transpose not implemented. DGT input tensor needs to be F by K.")
    kernel_assert(
        load_loc.tensor.quant_scheme == QuantScheme.WRAPX,
        "load_tile_dgt only supports wrapX quantization scheme.",
    )
    kernel_assert(
        src_data.dtype in (nl.bfloat16, nl.float16),
        f"DGT input tensor must be bfloat16 or float16, got {src_data.dtype}.",
    )
    # The fast DMA transpose path gathers any tile_k that is a multiple of
    # MX_PARTITION_SIZE (the quantization scale granularity) in a single op, so it only
    # needs K aligned to MX_PARTITION_SIZE. The legacy vector-offset path still requires
    # the coarser MIN_K_FOR_QUANTIZATION alignment.
    min_k_alignment = MX_PARTITION_SIZE if load_loc.tensor.fast_dma_transpose else MIN_K_FOR_QUANTIZATION
    kernel_assert(
        K % min_k_alignment == 0,
        f"K dimension must be divisible by {min_k_alignment} for mxfp8 quantization, got {K=}",
    )
    kernel_assert(
        F % MIN_F_FOR_QUANTIZATION == 0,
        f"F dim must be divisible by {MIN_F_FOR_QUANTIZATION} for quantization, got {F=}",
    )

    # Allocate destination SBUF tensor if store_loc not provided
    if store_loc == None:
        dst_sbuf = sbm.alloc_stack(
            shape=(load_loc.tile_k // INTERLEAVE_FACTOR, 1, 1, load_loc.tile_f * INTERLEAVE_FACTOR),
            dtype=nl.bfloat16,
            buffer=nl.sbuf,
        )
        store_loc = _TileLocation(
            tensor=TensorDescriptor(data=dst_sbuf, is_swizzled=True, orientation=TensorOrientation.K_BY_F),
            tile_k=load_loc.tile_k,
            tile_f=load_loc.tile_f,
        )

    # Compute actual tile_f for partial tiles (boundary handling).
    # Use effective_f_dim (per-expert F) when set, otherwise full tensor F.
    F_effective = load_loc.tensor.effective_f_dim if load_loc.tensor.effective_f_dim is not None else F
    actual_tile_f = min(load_loc.tile_f, F_effective - load_loc.f_offset)

    # Slice destination for the correct tile position
    # store_loc.k_offset indexes into the tiles dimension (dim 1)
    tile_idx = store_loc.k_offset
    f_start = store_loc.f_offset * INTERLEAVE_FACTOR
    f_end = f_start + actual_tile_f * INTERLEAVE_FACTOR
    actual_physical_k = load_loc.tile_k // INTERLEAVE_FACTOR
    dst_slice = store_loc.tensor.data[nl.ds(0, actual_physical_k), nl.ds(tile_idx, 1), :, f_start:f_end]

    # Fast DMA transpose path: use direct access pattern on the source tensor
    # instead of flattening + vector offsets
    if load_loc.tensor.fast_dma_transpose:
        vector_size = load_loc.tile_k // INTERLEAVE_FACTOR
        # f_scalar_offset is in F rows; dim-0 of the [F, K] source strides by K, so it
        # lands as-is (None reproduces the plain static address).
        src_ap = src_data.ap(
            pattern=[[vector_size, INTERLEAVE_FACTOR], [1, 1], [K, actual_tile_f], [1, vector_size]],
            offset=load_loc.f_offset * K + load_loc.k_offset,
            scalar_offset=load_loc.tensor.f_scalar_offset,
            indirect_dim=0,
        )
        nisa.dma_transpose(
            dst=dst_slice.reshape((vector_size, 1, actual_tile_f, INTERLEAVE_FACTOR)),
            src=src_ap,
            axes=(3, 1, 2, 0),
            dge_mode=dge_mode.hwdge,
        )
    else:
        # Legacy path: flatten source and use vector offsets
        # Flatten source: [F, K] -> [F*K/vector_size, 1, 1, vector_size]
        vector_size = load_loc.tile_k // INTERLEAVE_FACTOR
        src_flattened = src_data.reshape((F * K // vector_size, 1, 1, vector_size))

        kernel_assert(
            load_loc.access_pattern != None and load_loc.vector_offset != None,
            "Must set load location access pattern and vector offset before ",
        )

        # Slice vector offsets for remainder tiles. The access pattern gathers
        # actual_tile_f * INTERLEAVE_FACTOR rows and each offset column holds P_MAX
        # entries, so round the column count up - DGT requires the offsets to cover
        # every gathered row.
        flattened_rows = actual_tile_f * INTERLEAVE_FACTOR
        num_active_partitions = (flattened_rows + P_MAX - 1) // P_MAX
        vector_offset = load_loc.vector_offset[:, nl.ds(0, num_active_partitions)]

        # Perform DMA gather transpose
        nisa.dma_transpose(
            dst=dst_slice,
            src=src_flattened.ap(
                pattern=load_loc.access_pattern,
                vector_offset=vector_offset,
            ),
            axes=(3, 1, 2, 0),
        )

    return store_loc


def _validate_1x32_load(load_loc: _TileLocation) -> None:
    """Validate the common input contract for the two 1x32 loaders."""
    tensor = load_loc.tensor
    kernel_assert(
        not tensor.is_swizzled and not tensor.is_quantized and len(tensor.data.shape) == 2,
        "1x32 swizzle loading requires an unswizzled, unquantized 2D tensor.",
    )
    kernel_assert(
        tensor.data.dtype in (nl.bfloat16, nl.float16),
        "1x32 swizzle loading requires BF16 or FP16 input.",
    )
    kernel_assert(
        load_loc.tile_k > 0
        and load_loc.tile_k % MX_PARTITION_SIZE == 0
        and (load_loc.tile_k <= L_TILE_K or load_loc.tile_k % L_TILE_K == 0),
        "tile_k must be 32-aligned up to 512, or a multiple of 512.",
    )
    kernel_assert(
        load_loc.tile_f > 0 and load_loc.tile_f % MIN_F_FOR_QUANTIZATION == 0,
        f"tile_f must be a positive multiple of {MIN_F_FOR_QUANTIZATION}.",
    )
    K, F = tensor.logical_shape
    kernel_assert(
        K % MX_PARTITION_SIZE == 0,
        f"K dimension ({K}) must be divisible by {MX_PARTITION_SIZE} for MXFP8 quantization.",
    )
    kernel_assert(
        F % MIN_F_FOR_QUANTIZATION == 0,
        f"F dimension ({F}) must be divisible by {MIN_F_FOR_QUANTIZATION} for MXFP8 quantization.",
    )
    kernel_assert(
        tensor.f_scalar_offset is None or tensor.is_f_by_k,
        "f_scalar_offset requires an F-by-K operand; on a K-by-F tensor F is not the dim-0 axis",
    )


def _allocate_1x32_store_loc(load_loc: _TileLocation) -> _TileLocation:
    """Allocate the shared swizzled destination layout."""
    inner_tile_k = min(load_loc.tile_k, L_TILE_K)
    physical_k = inner_tile_k // INTERLEAVE_FACTOR
    num_k_slots = load_loc.tile_k // inner_tile_k
    data = get_active_sbm().alloc_stack(
        shape=(physical_k, num_k_slots, 1, load_loc.tile_f * INTERLEAVE_FACTOR),
        dtype=load_loc.tensor.data.dtype,
        buffer=nl.sbuf,
    )
    return _TileLocation(
        tensor=TensorDescriptor(
            data=data, is_swizzled=True, orientation=TensorOrientation.K_BY_F, quant_scheme=QuantScheme._1x32
        ),
        tile_k=physical_k,
        tile_f=load_loc.tile_f,
    )


def _get_1x32_destination(store_loc: _TileLocation, physical_k: int, num_k_slots: int, tile_f: int) -> nl.NkiTensor:
    """View a 2D, 3D or 4D destination with identical K-slot and F offsets."""
    destination = store_loc.tensor.data
    if len(destination.shape) == 2:
        destination = destination.reshape((destination.shape[0], 1, 1, destination.shape[1]))
    elif len(destination.shape) == 3:
        destination = destination.reshape((destination.shape[0], destination.shape[1], 1, destination.shape[2]))
    kernel_assert(
        len(destination.shape) == 4 and destination.shape[2] == 1,
        "1x32 swizzle requires a 2D, 3D, or [P, K-slots, 1, F] destination.",
    )
    kernel_assert(
        physical_k <= destination.shape[0]
        and 0 <= store_loc.k_offset
        and store_loc.k_offset + num_k_slots <= destination.shape[1]
        and 0 <= store_loc.f_offset
        and (store_loc.f_offset + tile_f) * INTERLEAVE_FACTOR <= destination.shape[3],
        "The swizzled tile exceeds the destination extent.",
    )
    return destination


def _zero_sub_tile_tail_2d(buf, num_full_sub_tiles: int, total_sub_tiles: int, tile_k: int) -> None:
    """Zero the free-axis columns from the first partial sub-tile onward.

    memset must start at partition 0, so this spans all partitions; the DMAs that follow
    overwrite the valid rows. `buf` is [P_MAX, total_sub_tiles * tile_k]; `tile_k` is in the
    buffer's element units (fp32 columns for the 1x32 loader, bf16 columns for wrapX).
    """
    if num_full_sub_tiles < total_sub_tiles:
        nisa.memset(buf[:, num_full_sub_tiles * tile_k :], value=0.0)


def _zero_sub_tile_tail_4d(buf, num_full_sub_tiles: int, total_sub_tiles: int) -> None:
    """4D variant of _zero_sub_tile_tail_2d: zero whole sub-tiles from the first partial one on."""
    if num_full_sub_tiles < total_sub_tiles:
        nisa.memset(buf[:, num_full_sub_tiles:, :, :], value=0.0)


def load_tile_tensor_copy_1x32(
    load_loc: _TileLocation,
    data_store_loc: Optional[_TileLocation] = None,
    dma_engine=None,
) -> None:
    """Swizzle an unswizzled K-by-F BF16/FP16 tile using DMA copy and tensor-copy interleave.

    Consecutive groups of four K elements are folded onto SBUF free axes, then
    interleaved so each 1x32 quantization group contains consecutive K at a fixed F.
    Uses the same input validation, allocation, destination layouts and offsets
    as load_tile_PE_Swizzle_1x32 for K-by-F inputs. K tiles are 32-aligned up to
    512, or a multiple of 512; F tiles are multiples of 8. K padding is zeroed
    within valid F columns. F padding remains unwritten and must not be consumed.
    A missing destination is allocated in SBUF. Both loaders write into
    data_store_loc and return None. dma_engine optionally pins the
    load's hardware DGE queue (see load_tile).
    """
    dma_kwargs = {} if dma_engine is None else {"dge_mode": dge_mode.hwdge, "engine": dma_engine}
    sbm = get_active_sbm()
    src_data = load_loc.tensor.data
    kernel_assert(not load_loc.tensor.is_f_by_k, "The tensor-copy 1x32 loader requires K-by-F input.")
    _validate_1x32_load(load_loc)
    K, F = load_loc.tensor.logical_shape
    inner_tile_k = min(load_loc.tile_k, L_TILE_K)
    physical_k = inner_tile_k // INTERLEAVE_FACTOR
    num_inner_k = load_loc.tile_k // inner_tile_k
    if data_store_loc is None:
        data_store_loc = _allocate_1x32_store_loc(load_loc)
    destination = _get_1x32_destination(data_store_loc, physical_k, num_inner_k, load_loc.tile_f)
    F_effective = load_loc.tensor.effective_f_dim if load_loc.tensor.effective_f_dim is not None else F
    actual_tile_f = max(0, min(load_loc.tile_f, F - load_loc.f_offset, F_effective - load_loc.f_offset))
    tile_stride = destination.shape[2] * destination.shape[3]
    row_stride = destination.shape[1] * tile_stride

    sbm.open_scope("LOAD_1X32_TENSOR_COPY")
    for inner_k_idx in range(num_inner_k):
        k_start = load_loc.k_offset + inner_k_idx * inner_tile_k
        actual_tile_k = max(0, min(inner_tile_k, K - k_start))
        kernel_assert(actual_tile_k % INTERLEAVE_FACTOR == 0, "The readable K extent must be x4 aligned.")
        actual_physical_k = actual_tile_k // INTERLEAVE_FACTOR
        destination_offset = (
            data_store_loc.k_offset + inner_k_idx
        ) * tile_stride + data_store_loc.f_offset * INTERLEAVE_FACTOR
        if actual_tile_k < inner_tile_k and actual_tile_f > 0:
            # Matmul still reduces these padded K rows in the final tile.
            nisa.memset(
                dst=destination.ap(
                    pattern=[[row_stride, physical_k], [1, actual_tile_f * INTERLEAVE_FACTOR]],
                    offset=destination_offset,
                ),
                value=0,
            )
        if actual_physical_k > 0 and actual_tile_f > 0:
            lane_buffer = sbm.alloc_stack(
                shape=(physical_k, INTERLEAVE_FACTOR, actual_tile_f),
                dtype=src_data.dtype,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=lane_buffer[:actual_physical_k, :, :],
                src=src_data.ap(
                    pattern=[
                        [INTERLEAVE_FACTOR * F, actual_physical_k],
                        [F, INTERLEAVE_FACTOR],
                        [1, actual_tile_f],
                    ],
                    offset=k_start * F + load_loc.f_offset,
                ),
                **dma_kwargs,
            )
            nisa.tensor_copy(
                dst=destination.ap(
                    pattern=[
                        [row_stride, actual_physical_k],
                        [INTERLEAVE_FACTOR, actual_tile_f],
                        [1, INTERLEAVE_FACTOR],
                    ],
                    offset=destination_offset,
                ),
                src=lane_buffer.ap(
                    pattern=[
                        [INTERLEAVE_FACTOR * actual_tile_f, actual_physical_k],
                        [1, actual_tile_f],
                        [actual_tile_f, INTERLEAVE_FACTOR],
                    ],
                ),
            )
    sbm.close_scope()


def _load_k_by_f_pe_chunk(
    src_data: nl.NkiTensor,
    sbuf_fp32: nl.NkiTensor,
    chunk_k: int,
    actual_tile_f: int,
    num_sub_tiles: int,
    num_full_sub_tiles: int,
    remainder_f: int,
    row_stride_bf16: int,
    tile_k: int,
    k_sub: int,
    f_offset: int,
    k_chunk_offset: int,
) -> None:
    """DMA-copy one K strip, then PE-transpose its feature sub-tiles."""
    if actual_tile_f == 0:
        return
    load_tmp = get_active_sbm().alloc_stack((chunk_k, actual_tile_f), dtype=src_data.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=load_tmp, src=src_data[nl.ds(k_sub, chunk_k), nl.ds(f_offset, actual_tile_f)])
    for subtile_idx in range(num_sub_tiles):
        sub_tile_f = SUB_TILE if subtile_idx < num_full_sub_tiles else remainder_f
        psum_t = nl.ndarray((sub_tile_f, chunk_k), dtype=src_data.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=psum_t, data=load_tmp[:, nl.ds(subtile_idx * SUB_TILE, sub_tile_f)])
        nisa.tensor_copy(
            dst=sbuf_fp32.ap(
                pattern=[[row_stride_bf16, sub_tile_f], [1, chunk_k]],
                offset=subtile_idx * tile_k + k_chunk_offset,
                dtype=src_data.dtype,
            ),
            src=psum_t,
        )


def load_tile_PE_Swizzle_1x32(
    load_loc: _TileLocation,
    data_store_loc: Optional[_TileLocation] = None,
    dma_engine=None,
) -> None:
    """
    Load an unswizzled BF16/FP16 tile from HBM and interleave into the
    swizzled [K//4, F*4] layout in SBUF using FP32 PE transpose.

    Supports both [F, K] (F-by-K) and [K, F] (K-by-F) input layouts.

    Uses the FP32 PE swizzle approach: reinterprets bf16 as fp32 to
    halve the partition count, then 2x nc_transpose with stride-2 APs interleaves
    the partitions. Suitable for data coming from HBM (weights, activations).

    The output is a swizzled bf16 buffer — NOT quantized. Quantization (if needed)
    should be done separately downstream via nisa.quantize_mx on the output.

    Both layouts gather into an identical [F-rows, K(fp32-packed)] SBUF buffer:
    F-by-K via reinterpret dma_copy, K-by-F via one dma_transpose per 128-F
    sub-tile. The fp32 transpose loop is then shared.

    With DMA transpose disabled, K-by-F loads use regular DMA copies and
    PE transposes. Loader selection is handled by load_tile. F padding may be
    quantized, but downstream matmul must exclude those features.

    Args:
        load_loc (_TileLocation): Source tile location in HBM. Tensor must be
            unswizzled bf16 [F, K] (is_f_by_k=True) or [K, F] (is_f_by_k=False).
        data_store_loc (Optional[_TileLocation]): Destination for swizzled bf16 data.
            If None, allocated as [K//4, K-slots, 1, F*4] in SBUF.
        dma_engine: F-by-K only (K-by-F rejects it). Issue the HBM loads on this engine's
            hardware DGE (see load_tile). None lets the compiler pick the DGE.

    Returns:
        None (writes swizzled bf16 data into data_store_loc)
    """

    sbm = get_active_sbm()

    src_data = load_loc.tensor.data
    is_f_by_k = load_loc.tensor.is_f_by_k
    # (K, F) logical extent regardless of physical layout ([F, K] vs [K, F]).
    K, F = load_loc.tensor.logical_shape

    _validate_1x32_load(load_loc)
    kernel_assert(dma_engine is None or is_f_by_k, "dma_engine is only supported for F-by-K input.")

    tile_f = load_loc.tile_f
    f_offset = load_loc.f_offset
    k_offset = load_loc.k_offset
    NUM_SUB_TILES = (tile_f + SUB_TILE - 1) // SUB_TILE

    # Runtime F-row offset, in F rows. Only the F-by-K gathers below index dim 0 by F, so
    # the K-by-F layout (where F is dim 1) cannot carry it.
    f_scalar_offset = load_loc.tensor.f_scalar_offset

    # K extents. tile_k is 32-aligned and may span several L_TILE_K with coalescing.
    tile_k = load_loc.tile_k
    tile_k_fp32 = tile_k // BF16_TO_FP32_RATIO

    # Coalesced load: tile_k may span several L_TILE_K worth of K so the DMA can read the full
    # contiguous K row in one burst. The PE transpose still runs per INNER_TILE_K (<= L_TILE_K)
    # slice, since its output puts INNER_TILE_K/INTERLEAVE_FACTOR (<= 128) K-groups on partitions.
    INNER_TILE_K = min(tile_k, L_TILE_K)
    NUM_INNER_K = tile_k // INNER_TILE_K
    inner_tile_k_fp32 = INNER_TILE_K // BF16_TO_FP32_RATIO
    physical_inner_k = INNER_TILE_K // INTERLEAVE_FACTOR

    # Allocate output SBUF tensors if not provided.
    if data_store_loc is None:
        data_store_loc = _allocate_1x32_store_loc(load_loc)

    K_fp32 = K // BF16_TO_FP32_RATIO
    k_fp32_offset = k_offset // BF16_TO_FP32_RATIO

    # Clamp to the in-bounds F (effective_f_dim handles stacked-expert tensors). Only the full
    # 128-wide sub-tiles plus an optional <128 remainder are read. Staging padding is
    # zeroed for the fixed-size PE transposes; only valid F columns reach the destination.
    # remainder_f stays x4-pack aligned so the swizzle stays correct.
    F_effective = load_loc.tensor.effective_f_dim if load_loc.tensor.effective_f_dim is not None else F
    actual_tile_f = max(0, min(tile_f, F - f_offset, F_effective - f_offset))
    num_full_sub_tiles = actual_tile_f // SUB_TILE
    remainder_f = actual_tile_f - num_full_sub_tiles * SUB_TILE
    has_partial_f = actual_tile_f < NUM_SUB_TILES * SUB_TILE

    destination = _get_1x32_destination(data_store_loc, physical_inner_k, NUM_INNER_K, tile_f)

    # Shared buffer: both layouts gather here (K-by-F via a bf16-reinterpret AP), sub-tile i at fp32 offset i * tile_k_fp32.
    row_stride_fp32 = NUM_SUB_TILES * tile_k_fp32
    sbuf_fp32 = sbm.alloc_stack(shape=(P_MAX, row_stride_fp32), dtype=nl.float32, buffer=nl.sbuf)

    num_sub_tiles = num_full_sub_tiles + (remainder_f > 0)

    dma_kwargs = {} if dma_engine is None else {"dge_mode": dge_mode.hwdge, "engine": dma_engine}

    if is_f_by_k:
        if not has_partial_f:
            # Tile fits: grab up to MAX_SUBTILES_PER_DMA sub-tiles per DMA with one folded gather.
            # More than that at a nonzero offset trips the compiler's bounds checker (NCC_IBIR243).
            for chunk_start in range(0, NUM_SUB_TILES, MAX_SUBTILES_PER_DMA):
                chunk_subtiles = min(MAX_SUBTILES_PER_DMA, NUM_SUB_TILES - chunk_start)
                nisa.dma_copy(
                    dst=sbuf_fp32.ap(
                        pattern=[[row_stride_fp32, P_MAX], [1, chunk_subtiles * tile_k_fp32]],
                        offset=chunk_start * tile_k_fp32,
                    ),
                    src=src_data.ap(
                        pattern=[[K_fp32, P_MAX], [K_fp32 * SUB_TILE, chunk_subtiles], [1, tile_k_fp32]],
                        offset=(f_offset + chunk_start * SUB_TILE) * K_fp32 + k_fp32_offset,
                        dtype=nl.float32,
                        scalar_offset=f_scalar_offset,
                        indirect_dim=0,
                    ),
                    **dma_kwargs,
                )
        else:
            # Partial F: folded gather would over-read, so one DMA per sub-tile; last loads sub_tile_f (<128) rows into the pre-zeroed tail.
            _zero_sub_tile_tail_2d(sbuf_fp32, num_full_sub_tiles, NUM_SUB_TILES, tile_k_fp32)
            for subtile_idx in range(num_sub_tiles):
                sub_tile_f = SUB_TILE if subtile_idx < num_full_sub_tiles else remainder_f
                f_sub = f_offset + subtile_idx * SUB_TILE
                nisa.dma_copy(
                    dst=sbuf_fp32.ap(
                        pattern=[[row_stride_fp32, sub_tile_f], [1, tile_k_fp32]],
                        offset=subtile_idx * tile_k_fp32,
                    ),
                    src=src_data.ap(
                        pattern=[[K_fp32, sub_tile_f], [1, tile_k_fp32]],
                        offset=f_sub * K_fp32 + k_fp32_offset,
                        dtype=nl.float32,
                        scalar_offset=f_scalar_offset,
                        indirect_dim=0,
                    ),
                    **dma_kwargs,
                )
    elif not load_loc.tensor.disable_dma_transpose:
        # K-by-F: one dma_transpose per 128-F sub-tile reads [tile_k, SUB_TILE] from [K, F] and
        # transposes to [128P, tile_k], written via a bf16 reinterpret AP (stride in bf16 units).
        row_stride_bf16 = NUM_SUB_TILES * tile_k
        if has_partial_f:
            _zero_sub_tile_tail_2d(sbuf_fp32, num_full_sub_tiles, NUM_SUB_TILES, tile_k_fp32)
        # One dma_transpose per sub-tile; last transposes sub_tile_f (<128) F-cols into the pre-zeroed tail.
        for subtile_idx in range(num_sub_tiles):
            sub_tile_f = SUB_TILE if subtile_idx < num_full_sub_tiles else remainder_f
            f_sub = f_offset + subtile_idx * SUB_TILE
            nisa.dma_transpose(
                dst=sbuf_fp32.ap(
                    pattern=[[row_stride_bf16, sub_tile_f], [1, 1], [1, 1], [1, tile_k]],
                    offset=subtile_idx * tile_k,
                    dtype=src_data.dtype,
                ),
                src=src_data.ap(
                    pattern=[[F, tile_k], [1, 1], [1, 1], [1, sub_tile_f]],
                    offset=k_offset * F + f_sub,
                ),
                axes=(3, 1, 2, 0),
            )
    else:
        # K-by-F, DMA transpose disabled: read the full contiguous F-strip once per 128-K chunk
        # (one wide burst, not a narrow strided read per sub-tile), then PE-transpose each 128-F
        # slice. affine_range over K-chunks lets the compiler double-buffer the load vs the transpose.
        row_stride_bf16 = NUM_SUB_TILES * tile_k
        num_k_chunks = tile_k // P_MAX
        if has_partial_f:
            _zero_sub_tile_tail_2d(sbuf_fp32, num_full_sub_tiles, NUM_SUB_TILES, tile_k_fp32)
        for k_chunk_idx in nl.affine_range(num_k_chunks):
            _load_k_by_f_pe_chunk(
                src_data,
                sbuf_fp32,
                P_MAX,
                actual_tile_f,
                num_sub_tiles,
                num_full_sub_tiles,
                remainder_f,
                row_stride_bf16,
                tile_k,
                k_offset + k_chunk_idx * P_MAX,
                f_offset,
                k_chunk_idx * P_MAX,
            )
        if tile_k % P_MAX:
            # A 32-aligned K tile can end with a 32/64/96-row strip.
            _load_k_by_f_pe_chunk(
                src_data,
                sbuf_fp32,
                tile_k % P_MAX,
                actual_tile_f,
                num_sub_tiles,
                num_full_sub_tiles,
                remainder_f,
                row_stride_bf16,
                tile_k,
                k_offset + num_k_chunks * P_MAX,
                f_offset,
                num_k_chunks * P_MAX,
            )

    for subtile_idx in nl.affine_range(num_sub_tiles):
        for inner_k_idx in nl.affine_range(NUM_INNER_K):
            # Step 2: fp32 transpose (2 stride-2 passes) -> psum [K=physical_inner_k, F sub-tile x4].
            psum_fp32 = nl.ndarray((physical_inner_k, SUB_TILE * BF16_TO_FP32_RATIO), dtype=nl.float32, buffer=nl.psum)
            for interleave_pass_idx in nl.affine_range(FP32_NUM_TRANSPOSE_PASSES):
                nisa.nc_transpose(
                    dst=psum_fp32.ap(
                        pattern=[[SUB_TILE * BF16_TO_FP32_RATIO, physical_inner_k], [FP32_INTERLEAVE_STRIDE, SUB_TILE]],
                        offset=interleave_pass_idx,
                    ),
                    data=sbuf_fp32.ap(
                        pattern=[[row_stride_fp32, SUB_TILE], [FP32_INTERLEAVE_STRIDE, physical_inner_k]],
                        offset=subtile_idx * tile_k_fp32 + inner_k_idx * inner_tile_k_fp32 + interleave_pass_idx,
                    ),
                )

            # Step 3: copy PSUM -> destination K-slot (k_offset + inner_k_idx), reinterpreting fp32
            # as the bf16 swizzled [K//4, F*4] layout.
            tile_idx = data_store_loc.k_offset + inner_k_idx
            f_start = (data_store_loc.f_offset + subtile_idx * SUB_TILE) * INTERLEAVE_FACTOR
            sub_tile_f = min(SUB_TILE, actual_tile_f - subtile_idx * SUB_TILE)
            f_end = f_start + sub_tile_f * INTERLEAVE_FACTOR
            dst_slice = destination[nl.ds(0, physical_inner_k), nl.ds(tile_idx, 1), :, f_start:f_end]

            nisa.tensor_copy(
                dst=dst_slice,
                src=psum_fp32.ap(
                    pattern=[[SUB_TILE * INTERLEAVE_FACTOR, physical_inner_k], [1, sub_tile_f * INTERLEAVE_FACTOR]],
                    dtype=src_data.dtype,
                ),
            )


def load_tile_PE_swizzle_wrapX(
    load_loc: _TileLocation,
    data_store_loc: _TileLocation,
) -> None:
    """
    Load an unswizzled bf16 tile from HBM and interleave into the
    swizzled [K//4, F*4] layout in SBUF using BF16 PE transpose.

    Supports both [F, K] (F-by-K) and [K, F] (K-by-F) input layouts.
    This produces the same swizzled layout as DGT (load_tile_dgt) but uses
    a different mechanism: DMA loads followed by strided nc_transpose
    and tensor_copy, rather than hardware gather-transpose with interleaving.

    Loading strategy depends on input layout:
      - F-by-K direct: single dma_copy with 3-pair AP for the full tile.
      - F-by-K indirect: per-sub-tile dma_copy with vector_offset (128
        partition HW limit on dma_copy gather).
      - K-by-F direct: 4x dma_transpose per 128-F sub-tile, each transposes
        [tile_k, 128] from HBM directly into [128P, tile_k F] SBUF.
      - K-by-F indirect: not supported (kernel_assert).

    After loading, the PE swizzle loop (4x nc_transpose + tensor_copy)
    interleaves partitions into the final [K//4, F*4] layout.

    The output is a swizzled bf16 buffer — NOT quantized. Quantization (if needed)
    should be done separately via nisa.quantize_mx on the output.

    Args:
        load_loc (_TileLocation): Source tile location in HBM.
            - tensor: Must be [F, K] bf16/fp16, is_f_by_k=True, not swizzled, not quantized.
            - tile_k: Must equal L_TILE_K (512).
            - tile_f: Must be divisible by SUB_TILE (128).
            - k_offset: Offset in K dimension for the tile start.
            - f_offset: Offset in F dimension (used in direct mode only).
            - vector_offset: If None, direct DMA. If set, must be an SBUF tensor
              of shape (P_MAX, NUM_SUB_TILES) where NUM_SUB_TILES = tile_f // SUB_TILE.
              Each column contains SUB_TILE (128) int32 global F-row indices for
              one sub-tile's gather operation. This layout respects SBUF's 128
              partition limit.

        data_store_loc (Optional[_TileLocation]): Destination for swizzled data in SBUF.
            If None, allocated as [tile_k // INTERLEAVE_FACTOR, tile_f * INTERLEAVE_FACTOR]
            bf16 in SBUF. The _TileLocation.tensor must have is_swizzled=True.

    Returns:
        None (writes swizzled bf16 data into data_store_loc)

    Examples:
        # Direct mode (contiguous rows 128..255):
        load_loc = _TileLocation(
            tensor=td, tile_k=512, tile_f=128,
            k_offset=0, f_offset=128,
            vector_offset=None,  # direct DMA
        )
        load_tile_PE_swizzle_wrapX(load_loc)

        # Indirect mode (gather scattered tokens, tile_f=256 → 2 sub-tiles):
        # Shape (P_MAX=128, NUM_SUB_TILES=2): column 0 has first 128 row indices,
        # column 1 has next 128 row indices.
        token_indices = sbm.alloc_stack((128, 2), dtype=nl.int32)
        # ... fill token_indices with global F-row IDs ...
        load_loc = _TileLocation(
            tensor=td, tile_k=512, tile_f=256,
            k_offset=0, f_offset=0,
            vector_offset=token_indices,  # indirect DMA
        )
        load_tile_PE_swizzle_wrapX(load_loc)
    """

    sbm = get_active_sbm()

    src_data = load_loc.tensor.data
    original_dtype = src_data.dtype

    is_f_by_k = load_loc.tensor.is_f_by_k
    # (K, F) logical extent regardless of physical layout ([F, K] vs [K, F]) — the
    # TensorDescriptor encapsulates that mapping via logical_shape.
    K, F = load_loc.tensor.logical_shape

    kernel_assert(not load_loc.tensor.is_swizzled, "BF16 transpose load requires unswizzled input.")
    kernel_assert(not load_loc.tensor.is_quantized, "BF16 transpose load requires non-quantized input.")
    kernel_assert(
        load_loc.tensor.f_scalar_offset is None,
        "f_scalar_offset is not supported by the wrapX PE-swizzle loader; use the DGT, "
        "DGT_FAST or 1x32 PE path, or fold the offset into a slice view",
    )
    kernel_assert(
        load_loc.tensor.quant_scheme == QuantScheme.WRAPX,
        "load_tile_PE_swizzle_wrapX only supports wrapX quantization scheme.",
    )
    kernel_assert(
        src_data.dtype in (nl.bfloat16, nl.float16),
        f"BF16 transpose load requires bfloat16 or float16 input, got {src_data.dtype}.",
    )

    # The swizzle loop only needs tile_k split into INTERLEAVE_FACTOR equal K quarters whose
    # size is a whole number of partitions, so MX_PARTITION_SIZE alignment suffices -- the
    # tighter Q_TILE_K bound would exclude the 32/64/96 K tiles a K % 32 shape decomposes into.
    kernel_assert(
        load_loc.tile_k % MX_PARTITION_SIZE == 0,
        f"BF16 transpose load requires tile_k a multiple of {MX_PARTITION_SIZE}, got {load_loc.tile_k}.",
    )
    kernel_assert(load_loc.tile_f % SUB_TILE == 0, f"Currently tile_f must be divisible by {SUB_TILE}.")
    kernel_assert(
        K % MX_PARTITION_SIZE == 0,
        f"K dimension ({K}) must be divisible by {MX_PARTITION_SIZE} for MXFP8 quantization.",
    )
    kernel_assert(
        F % MIN_F_FOR_QUANTIZATION == 0,
        f"F dimension ({F}) must be divisible by {MIN_F_FOR_QUANTIZATION} for MXFP8 quantization.",
    )

    tile_f = load_loc.tile_f
    tile_k = load_loc.tile_k
    f_offset = load_loc.f_offset
    k_offset = load_loc.k_offset
    NUM_SUB_TILES = tile_f // SUB_TILE
    physical_tile_k = tile_k // INTERLEAVE_FACTOR

    # Partial final F load-tile (load_block always passes the full LOAD_TILE_F): load the full
    # 128-wide sub-tiles that fit plus a partial (<128) tail of remainder_f cols, masking the
    # pre-zeroed SBUF remainder. remainder_f stays x4-pack aligned, so the swizzle is correct.
    F_effective = load_loc.tensor.effective_f_dim if load_loc.tensor.effective_f_dim is not None else F
    actual_tile_f = min(tile_f, F_effective - f_offset)
    num_full_sub_tiles = actual_tile_f // SUB_TILE
    remainder_f = actual_tile_f - num_full_sub_tiles * SUB_TILE
    has_partial_f = actual_tile_f < tile_f
    num_sub_tiles = num_full_sub_tiles + (remainder_f > 0)

    # The partial-F paths below zero the sub-tile tail via _zero_sub_tile_tail_2d/_4d before the
    # masked DMAs. (skip_token over-reads unpredictably and zeroes the whole buffer instead --
    # handled by the caller.)

    # Determine DMA mode: indirect if vector_offset is provided, direct otherwise.
    use_indirect_dma = load_loc.vector_offset is not None

    kernel_assert(data_store_loc is not None, "data_store_loc is None")
    kernel_assert(
        len(data_store_loc.tensor.data.shape) == 4,
        f"expected 4D SBUF destination tensor, found tensor with shape {data_store_loc.tensor.data.shape}",
    )

    skip_token = False

    if load_loc.tensor.skip_dma is not None:
        skip_token = load_loc.tensor.skip_dma.skip_token

    # ========================================================================
    # Step 1: DMA load — issue bulk DMA(s) BEFORE the sub-tile PE swizzle loop.
    # Each path produces sbuf_tile [P_MAX, NUM_SUB_TILES * tile_k] where sub-tile
    # `i` occupies free-axis columns [i*tile_k : (i+1)*tile_k].
    # ========================================================================

    if is_f_by_k and not use_indirect_dma:
        # --- F-by-K direct: single DMA for the full tile ---
        # Load [tile_f, tile_k] from HBM into [P_MAX, NUM_SUB_TILES * tile_k] SBUF.
        # Groups of P_MAX=128 consecutive F-rows map to partitions; successive groups
        # are placed at increasing free-axis offsets (tile_k apart).
        sbuf_tile = sbm.alloc_stack(shape=(P_MAX, NUM_SUB_TILES * tile_k), dtype=original_dtype, buffer=nl.sbuf)
        if skip_token:
            nisa.memset(sbuf_tile, value=0.0)
        elif has_partial_f:
            _zero_sub_tile_tail_2d(sbuf_tile, num_full_sub_tiles, NUM_SUB_TILES, tile_k)
        # Full 128-row sub-tile groups in one DMA.
        if num_full_sub_tiles > 0:
            nisa.dma_copy(
                dst=sbuf_tile[:, nl.ds(0, num_full_sub_tiles * tile_k)],
                src=src_data.ap(
                    pattern=[[K, P_MAX], [K * P_MAX, num_full_sub_tiles], [1, tile_k]],
                    offset=f_offset * K + k_offset,
                    dtype=original_dtype,
                ),
            )
        # Partial final sub-tile: remainder_f F-rows into the first remainder_f partitions.
        if remainder_f > 0:
            f_sub = f_offset + num_full_sub_tiles * SUB_TILE
            nisa.dma_copy(
                dst=sbuf_tile[nl.ds(0, remainder_f), nl.ds(num_full_sub_tiles * tile_k, tile_k)],
                src=src_data.ap(
                    pattern=[[K, remainder_f], [1, tile_k]],
                    offset=f_sub * K + k_offset,
                    dtype=original_dtype,
                ),
            )

    elif is_f_by_k and use_indirect_dma:
        # --- F-by-K indirect: one DMA per sub-tile (128 partition limit) ---
        sbuf_tile = sbm.alloc_stack(shape=(P_MAX, NUM_SUB_TILES * tile_k), dtype=original_dtype, buffer=nl.sbuf)
        if skip_token:
            nisa.memset(sbuf_tile, value=0.0)
        elif has_partial_f:
            _zero_sub_tile_tail_2d(sbuf_tile, num_full_sub_tiles, NUM_SUB_TILES, tile_k)
        # Gather is whole-128-sub-tile granular; partial-F is not expressible and routes through
        # the direct branch (this is the MoE token-gather path).
        kernel_assert(
            skip_token or remainder_f == 0,
            f"Indirect PE-swizzle load requires F aligned to {SUB_TILE}, got partial {remainder_f}.",
        )
        # skip_token over-reads and relies on oob_mode.skip; otherwise clamp to the in-bounds sub-tiles.
        n_sub_tiles = NUM_SUB_TILES if skip_token else num_full_sub_tiles
        for subtile_idx in nl.affine_range(n_sub_tiles):
            nisa.dma_copy(
                dst=sbuf_tile[:, nl.ds(subtile_idx * tile_k, tile_k)],
                src=src_data.ap(
                    pattern=[[K, SUB_TILE], [1, tile_k]],
                    offset=k_offset,
                    vector_offset=load_loc.vector_offset[:, nl.ds(subtile_idx, 1)],
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_token else oob_mode.error,
            )

    elif not is_f_by_k and not use_indirect_dma:
        # --- K-by-F direct: 4x fast DMA direct transpose (one per 128-F sub-tile) ---
        # Each reads [tile_k, SUB_TILE] from HBM and transposes to [SUB_TILE=128P, tile_k F].
        # This path is a DMA transpose; disable_dma_transpose is only implemented for 1x32.
        kernel_assert(
            not load_loc.tensor.disable_dma_transpose,
            "disable_dma_transpose is only implemented for the 1x32 loader; wrapX K-by-F uses a DMA transpose.",
        )
        sbuf_tile_4d = sbm.alloc_stack(shape=(P_MAX, NUM_SUB_TILES, 1, tile_k), dtype=original_dtype, buffer=nl.sbuf)
        if skip_token:
            nisa.memset(sbuf_tile_4d, value=0.0)
        elif has_partial_f:
            _zero_sub_tile_tail_4d(sbuf_tile_4d, num_full_sub_tiles, NUM_SUB_TILES)

        # Full 128-wide sub-tiles within F; a sub-tile past F would OOB-read HBM.
        for subtile_idx in nl.affine_range(num_full_sub_tiles):
            f_sub = f_offset + subtile_idx * SUB_TILE
            nisa.dma_transpose(
                dst=sbuf_tile_4d[:, nl.ds(subtile_idx, 1), :, :],
                src=src_data.ap(
                    pattern=[[F, tile_k], [1, 1], [1, 1], [1, SUB_TILE]],
                    offset=k_offset * F + f_sub,
                ),
                axes=(3, 1, 2, 0),
            )

        # Partial final sub-tile: transpose remainder_f F-columns into the first remainder_f partitions.
        if remainder_f > 0:
            f_sub = f_offset + num_full_sub_tiles * SUB_TILE
            nisa.dma_transpose(
                dst=sbuf_tile_4d[nl.ds(0, remainder_f), nl.ds(num_full_sub_tiles, 1), :, :],
                src=src_data.ap(
                    pattern=[[F, tile_k], [1, 1], [1, 1], [1, remainder_f]],
                    offset=k_offset * F + f_sub,
                ),
                axes=(3, 1, 2, 0),
            )

        sbuf_tile = sbuf_tile_4d.reshape((P_MAX, NUM_SUB_TILES * tile_k))

    elif not is_f_by_k and use_indirect_dma:
        # --- K-by-F indirect: one DMA per sub-tile (gather token rows from [K, F]) ---
        sbuf_tile = sbm.alloc_stack(shape=(P_MAX, NUM_SUB_TILES * tile_k), dtype=original_dtype, buffer=nl.sbuf)
        if skip_token:
            nisa.memset(sbuf_tile, value=0.0)
        elif has_partial_f:
            _zero_sub_tile_tail_2d(sbuf_tile, num_full_sub_tiles, NUM_SUB_TILES, tile_k)
        # See F-by-K indirect: whole-128-sub-tile granular; partial-F uses the direct branch.
        kernel_assert(
            skip_token or remainder_f == 0,
            f"Indirect PE-swizzle load requires F aligned to {SUB_TILE}, got partial {remainder_f}.",
        )
        n_sub_tiles = NUM_SUB_TILES if skip_token else num_full_sub_tiles
        for subtile_idx in nl.affine_range(n_sub_tiles):
            nisa.dma_copy(
                dst=sbuf_tile[:, nl.ds(subtile_idx * tile_k, tile_k)],
                src=src_data.ap(
                    pattern=[[F, SUB_TILE], [1, tile_k]],
                    offset=k_offset,
                    vector_offset=load_loc.vector_offset[:, nl.ds(subtile_idx, 1)],
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_token else oob_mode.error,
            )

    else:
        kernel_assert(False, "Unsupported DMA load configuration for load_tile_PE_swizzle_wrapX.")

    # Drains are assigned per repeating group of drain_group_size sub-tiles: the first
    # num_vector_drains go to Vector, the rest to Scalar.
    num_scalar_drains, num_vector_drains = load_loc.tensor.psum_drain_engine_ratio
    drain_group_size = num_scalar_drains + num_vector_drains

    # ========================================================================
    # Step 2-3: PE swizzle loop — operates on sbuf_tile sub-tile slices.
    # ========================================================================
    for subtile_idx in range(NUM_SUB_TILES):
        # Step 2: nc_transpose in bf16 space (4 passes)
        # Source reads from sbuf_tile at free-axis offset subtile_idx * tile_k.
        psum_interleaved = nl.ndarray(
            (physical_tile_k, SUB_TILE * INTERLEAVE_FACTOR), dtype=original_dtype, buffer=nl.psum
        )
        for interleave_pass_idx in nl.affine_range(BF16_NUM_TRANSPOSE_PASSES):
            nisa.nc_transpose(
                dst=psum_interleaved.ap(
                    pattern=[[SUB_TILE * INTERLEAVE_FACTOR, physical_tile_k], [8, SUB_TILE // 2], [1, 2]],
                    offset=interleave_pass_idx * 2,
                ),
                data=sbuf_tile.ap(
                    pattern=[[NUM_SUB_TILES * tile_k, SUB_TILE], [1, physical_tile_k]],
                    offset=subtile_idx * tile_k + interleave_pass_idx * physical_tile_k,
                ),
            )

        # Step 3: tensor_copy from PSUM to final SBUF destination with reordering AP
        tile_idx = data_store_loc.k_offset
        f_start = (data_store_loc.f_offset + subtile_idx * SUB_TILE) * INTERLEAVE_FACTOR
        f_end = f_start + SUB_TILE * INTERLEAVE_FACTOR
        dst_slice = data_store_loc.tensor.data[nl.ds(0, physical_tile_k), nl.ds(tile_idx, 1), :, f_start:f_end]

        src_psum_ap = psum_interleaved.ap(
            pattern=[[SUB_TILE * INTERLEAVE_FACTOR, physical_tile_k], [8, SUB_TILE // 2], [1, 2], [2, 4]],
            dtype=original_dtype,
        )

        # Alternate using Vector/Scalar engine to drain PSUM to avoid bottleneck caused by all of them being scheduled on vector engine
        drain_engine = (
            nisa.vector_engine if (subtile_idx % drain_group_size) < num_vector_drains else nisa.scalar_engine
        )
        nisa.tensor_copy(dst=dst_slice, src=src_psum_ap, engine=drain_engine)


def load_tile_bf16_xbar_transpose(
    load_loc: _TileLocation,
    fp8_x4_dtype: Optional[type] = None,
) -> TensorDescriptor:
    """
    Interleave a [P_MAX, tile_f] bf16 tile already in SBUF via xbar transpose,
    then quantize to MXFP8.

    Uses the BF16 PE swizzle approach: 4x nc_transpose with stride-4
    src AP and stride-8/2-element dst AP, followed by tensor_copy PSUM -> SBUF
    with an unscramble pattern. Suitable for on-chip intermediates (e.g.,
    SiLU(gate) * up) that should not be spilled to HBM.

    Steps:
      1. 4x nc_transpose: stride-4 read from SBUF, stride-8 + 2-element
         grouping write to PSUM
      2. tensor_copy PSUM -> SBUF with unscramble AP
      3. quantize_mx from the interleaved SBUF buffer

    Args:
        load_loc (_TileLocation): Source tile location in SBUF. Tensor data must be
            [P_MAX, tile_f] bf16, standalone (not a slice of a larger buffer).
        fp8_x4_dtype: Target MXFP8 quantized dtype. If None, defaults to float8_e4m3fn_x4.

    Returns:
        TensorDescriptor: Quantized tensor with data (fp8_x4) and scales (uint8),
                          shape [Q_TILE_K, tile_f // INTERLEAVE_FACTOR]. Access via .data and .scales.
    """
    sbm = get_active_sbm()
    src_sbuf = load_loc.tensor.data
    kernel_assert(src_sbuf.dtype == nl.bfloat16, "BF16 xbar transpose requires bfloat16 input.")
    kernel_assert(src_sbuf.shape[0] == P_MAX, f"Source partition dim must be {P_MAX}.")
    kernel_assert(not load_loc.tensor.is_f_by_k, "BF16 xbar transpose expects K-by-F layout (SBUF-resident).")
    kernel_assert(not load_loc.tensor.is_swizzled, "BF16 xbar transpose requires unswizzled input.")
    kernel_assert(not load_loc.tensor.is_quantized, "BF16 xbar transpose requires non-quantized input.")
    kernel_assert(
        load_loc.tile_f % INTERLEAVE_FACTOR == 0,
        f"tile_f ({load_loc.tile_f}) must be divisible by INTERLEAVE_FACTOR ({INTERLEAVE_FACTOR}).",
    )

    tile_f = load_loc.tile_f
    if fp8_x4_dtype == None:
        fp8_x4_dtype = get_fp8_dtype_x4("float8_e4m3fn")

    q_tile_f = tile_f // INTERLEAVE_FACTOR
    data = sbm.alloc_stack(shape=(Q_TILE_K, q_tile_f), dtype=fp8_x4_dtype, buffer=nl.sbuf)
    scale = sbm.alloc_stack(shape=(Q_TILE_K, q_tile_f), dtype=nl.float8_e8m0fnu, buffer=nl.sbuf)

    # Step 1: 4-pass nc_transpose in bf16 space
    """
    Source AP: stride-4 reads every 4th partition from the [P_MAX, tile_f] input.
    Dest AP: stride-8 with 64-element groups and 2-element pairing places data
    into interleaved positions in PSUM.
    """
    psum_bf16 = nl.ndarray((P_MAX, tile_f), dtype=nl.bfloat16, buffer=nl.psum)
    for transpose_pass_idx in nl.affine_range(BF16_NUM_TRANSPOSE_PASSES):
        nisa.nc_transpose(
            dst=psum_bf16.ap(
                pattern=[[tile_f, P_MAX], [BF16_DST_STRIDE, BF16_DST_GROUP_SIZE], [1, BF16_DST_PAIR_SIZE]],
                offset=transpose_pass_idx * BF16_DST_OFFSET_MULTIPLIER,
                dtype=nl.bfloat16,
            ),
            data=src_sbuf.ap(
                pattern=[[tile_f, P_MAX], [BF16_SRC_STRIDE, P_MAX]],
                offset=transpose_pass_idx,
                dtype=nl.bfloat16,
            ),
        )

    # Step 2: Copy PSUM -> SBUF with unscramble pattern
    """
    The unscramble AP reorders the interleaved PSUM data into the final
    contiguous layout expected by quantize_mx.
    """
    swizzled_bf16 = sbm.alloc_stack(shape=(P_MAX, tile_f), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=swizzled_bf16,
        src=psum_bf16.ap(
            pattern=[
                [tile_f, P_MAX],
                [BF16_DST_STRIDE, BF16_DST_GROUP_SIZE],
                [1, BF16_DST_PAIR_SIZE],
                [BF16_UNSCRAMBLE_STRIDE, BF16_UNSCRAMBLE_GROUP_SIZE],
            ],
            dtype=nl.bfloat16,
        ),
    )

    # Step 3: Quantize from interleaved SBUF
    nisa.quantize_mx(dst=data, src=swizzled_bf16, dst_scale=scale)

    return TensorDescriptor(
        data=data,
        scales=scale,
        is_quantized=True,
        is_swizzled=True,
        orientation=TensorOrientation.K_BY_F,
    )
