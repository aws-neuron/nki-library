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

"""Kernel for quantizing BF16 tensors to MXFP8 format with block-wise quantization."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.logging import get_logger
from ..matmul_mxfp8.matmul_mxfp8_constants import QuantScheme
from ..mxfp_utils.mxfp8_utils import _quantize_mxfp8_block, _quantize_mxfp8_utils
from ..mxfp_utils.mxfp8_utils._common_utils import create_and_set_active_sbm, get_active_sbm, with_active_sbm
from ..mxfp_utils.mxfp8_utils._dense_scales import (
    get_dense_scale_shape,
    store_dense_scales,
)
from ..mxfp_utils.mxfp8_utils._load_apis import P_MAX, SUB_TILE, load_tile
from ..mxfp_utils.mxfp8_utils._quantize_mxfp8_utils import (
    MX_PARTITION_SIZE,
    QUANT_SCHEME_1X32,
    QUANT_SCHEME_MAP,
    QUANT_SCHEME_WRAPX,
    decompose_k_remainder,
)
from ..mxfp_utils.mxfp8_utils.common_dataclasses import (
    SwizzleMode,
    TensorDescriptor,
    TensorOrientation,
    _TileLocation,
)

logger = get_logger("quantize_mxfp8")

# Per-partition SBUF bytes for the scales the kernel holds back for one batched store. Large enough
# that a core's whole F range is one group for K x F up to 2048 x 8192 (LNC1), small next to the
# ~240 KB usable SBUF.
_SCALE_SBUF_BUDGET_BYTES = 48 * 1024


def _process_tile(
    src_td,
    tile_k_size,
    current_tile_f,
    k_offset,
    f_offset,
    INTERLEAVE_FACTOR,
    fp8_dtype,
    return_fp8_dtype,
    quantized_data_hbm,
    dma_engine,
    scales_sbuf,
    tile_f_alloc=None,
    uniform_dest_height=False,
    use_tensor_copy_swizzle=False,
):
    """
    Load, quantize, and store the data of a single tile of any MX_PARTITION_SIZE-aligned K size.
    Its scales are left in scales_sbuf for the caller to store.

    Args:
        src_td: Source TensorDescriptor
        tile_k_size: K dimension size for this tile (512, 256, 128, 96, 64 or 32)
        current_tile_f: Number of valid F columns in this tile
        k_offset: Offset in K dimension in the source tensor
        f_offset: Offset in F dimension in the source tensor
        INTERLEAVE_FACTOR: Interleave factor (4)
        fp8_dtype: FP8 dtype for output
        return_fp8_dtype: FP8 dtype string
        quantized_data_hbm: Output data tensor on HBM
        dma_engine: Engine whose hardware DGE issues the data store (and, for 1x32, the loads).
            Keeps the descriptor generation off GpSimd.
        scales_sbuf: [P_MAX, 1, tile_f_alloc] SBUF destination of the tile's scales (a slice of
            the kernel's scale buffer, see quantize_block_mxfp8_kernel).
        tile_f_alloc: SBUF F width to allocate; >= current_tile_f when the loader needs a
            padded (SUB_TILE-aligned) F tile. Defaults to current_tile_f.
        uniform_dest_height: Allocate the load destination at P_MAX partitions for every K tile
            (see the loader-selection comment in quantize_block_mxfp8_kernel).
        use_tensor_copy_swizzle: K-by-F 1x32 only. Swizzle this tile with the DMA +
            tensor_copy loader instead of the PE loader (see load_tile).

    Returns:
        None
    """

    sbm = get_active_sbm()

    if tile_f_alloc == None:
        tile_f_alloc = current_tile_f

    tile_k_interleaved = tile_k_size // INTERLEAVE_FACTOR
    # quantize_mx needs a src partition count that is a non-zero multiple of MX_PARTITION_SIZE,
    # so a sub-128 K tile (8/16/24 partitions) is quantized as a padded MX_PARTITION_SIZE tile.
    # Padding is safe because a scale covers exactly 8 partitions and tile_k_interleaved is a
    # multiple of 8, so the pad never shares a scale with valid data; only the valid
    # tile_k_interleaved rows are stored.
    p_quant = ((tile_k_interleaved + MX_PARTITION_SIZE - 1) // MX_PARTITION_SIZE) * MX_PARTITION_SIZE

    # TODO call load and quantize API when its merged
    # Allocate destination SBUF tensor and load. uniform_dest_height makes every K tile's load
    # destination the same shape regardless of tile_k; quantize still runs at p_quant via a slice.
    dest_parts = P_MAX if uniform_dest_height else p_quant
    src_tensor_sbuf = sbm.alloc_stack(
        shape=(dest_parts, 1, 1, tile_f_alloc * INTERLEAVE_FACTOR),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    # Zero the K pad so the discarded scale rows are deterministic rather than stale SBUF. Two
    # bounds keep this as small as the ISA allows:
    #   when -- only if quantize would read past what the load writes, i.e. p_quant differs from
    #           tile_k_interleaved. A 128/256/512 tile needs nothing.
    #   how much -- partitions [0, p_quant) only. quantize never reads above p_quant, so a taller
    #           destination (the fast arm's uniform height) must not widen the memset. memset
    #           cannot start at a nonzero partition, so [0, p_quant) is the tightest span that
    #           covers the real gap at [tile_k_interleaved, p_quant).
    if p_quant != tile_k_interleaved:
        nisa.memset(src_tensor_sbuf[nl.ds(0, p_quant)], value=0.0)

    load_loc = _TileLocation(
        tensor=src_td,
        tile_k=tile_k_size,
        tile_f=tile_f_alloc,
        k_offset=k_offset,
        f_offset=f_offset,
    )
    store_loc = _TileLocation(
        tensor=TensorDescriptor(data=src_tensor_sbuf, is_swizzled=True, orientation=TensorOrientation.K_BY_F),
        tile_k=tile_k_size,
        tile_f=tile_f_alloc,
    )

    # Only the 1x32 loaders take a DMA queue; the wrapX DGT and PE loaders keep their own.
    load_tile(
        load_loc,
        data_store_loc=store_loc,
        dma_engine=dma_engine if src_td.quant_scheme == QuantScheme._1x32 else None,
        use_strided_copy_swizzle_1x32=use_tensor_copy_swizzle,
    )

    if dest_parts == p_quant:
        quant_src = src_tensor_sbuf.reshape((p_quant, 1, tile_f_alloc * INTERLEAVE_FACTOR))
    else:
        quant_src = src_tensor_sbuf[nl.ds(0, p_quant), :, :, :].reshape((p_quant, 1, tile_f_alloc * INTERLEAVE_FACTOR))

    # Quantize: scales land in SBUF at the first 4 partitions of each 32-partition quadrant.
    quantized_data_x4, _ = _quantize_mxfp8_block.quantize_mxfp8_block(
        quant_src,
        (p_quant, tile_f_alloc),
        True,
        return_fp8_dtype,
        quantized_scales=scales_sbuf,
    )

    # Store quantized data. Offset is flat element offset: partition * elements_per_partition.
    # The row pitch is the allocated width tile_f_alloc, while only current_tile_f columns are valid.
    output_k_start = k_offset // INTERLEAVE_FACTOR
    output_f_start_data = f_offset * INTERLEAVE_FACTOR
    nisa.dma_copy(
        dst=quantized_data_hbm[
            nl.ds(output_k_start, tile_k_interleaved),
            nl.ds(output_f_start_data, current_tile_f * INTERLEAVE_FACTOR),
        ],
        src=quantized_data_x4.ap(
            pattern=[
                [tile_f_alloc * INTERLEAVE_FACTOR, tile_k_interleaved],
                [1, current_tile_f * INTERLEAVE_FACTOR],
            ],
            dtype=fp8_dtype,
            offset=0,
        ),
        dge_mode=dge_mode.hwdge,
        engine=dma_engine,
    )


@nki.jit
@with_active_sbm
def quantize_block_mxfp8_kernel(
    src_tensor: nl.NkiTensor,
    return_fp8_dtype: str,
    run_with_lnc2: bool = False,
    quant_scheme: str = None,
    use_fast_dma_transpose: bool = False,
    src_orientation: Optional[TensorOrientation] = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Kernel for quantizing BF16 tensor to MXFP8 format with block-wise quantization.

    This kernel performs block-wise quantization of BF16 tensors to MXFP8 format,
    producing both quantized data and scaling factors. Supports LNC2 parallelization
    for improved performance on large tensors.

    Scales are written dense, one row per 32 K (see dense_scales.py): every row holds a real
    scale, and a scale's row depends only on its K position. With the 1x32 layout the outputs
    of quantizing MX_PARTITION_SIZE-aligned K-shards therefore concatenate along dim 0 (data and
    scales alike) into the output of quantizing the whole tensor.

    Dimensions:
        F: Feature dimension (rows in input tensor)
        K: K dimension (columns in input tensor)

    Args:
        src_tensor (nl.NkiTensor): [F, K] (or [K, F], see src_orientation), Input tensor in BF16
            format on HBM
        return_fp8_dtype (str): FP8 dtype string like "float8_e4m3fn" or "float8_e5m2"
        run_with_lnc2 (bool): Enable LNC2 parallelization along F dimension (default: False)
        quant_scheme (str): Required. Swizzle layout of the quantized output, "1x32" or "wrapX".
            "wrapX" scatters a load tile's K across the four output columns of a feature and is
            loaded with a DMA gather-transpose; "1x32" packs INTERLEAVE_FACTOR consecutive K
            values of a feature into its four adjacent output columns and is loaded with the FP32
            PE transpose. The two produce different data orderings and are not interchangeable --
            the consuming matmul must use the same scheme. Prefer 1x32: its layout is independent of
            the K tiling, so any matmul tile_k can consume it, and K-shards quantized separately
            concatenate into the whole tensor's output. A wrapX output is only correct consumed whole by
            a matmul that tiles K exactly as this kernel does; composing wrapX outputs along K
            (concatenating, sharding, all-gathering) is illegal and silently gives wrong results, so
            wrapX logs a warning.
        use_fast_dma_transpose (bool): For wrapX, load every K tile with the fast (direct access
            pattern) DMA gather-transpose instead of the legacy vector-offset gather. Honoured for
            any K. When False the loader falls back to legacy DGT where legacy can express the shape
            and to the wrapX PE transpose where it cannot. Ignored for 1x32, which always uses its
            own PE loader. All three produce the identical output layout.
            Correct for any supported K: the fast path needed a uniform-height load destination
            to be reliable, which this kernel now gives it (see the loader-selection comment).
        src_orientation (Optional[TensorOrientation]): Layout of src_tensor relative to the
            contraction dim K. None or F_BY_K (default): src_tensor is [F, K]. K_BY_F: src_tensor
            is [K, F] -- e.g. a [O, I] weight quantized for the GEMM that contracts O -- and is
            read as is, so the caller need not materialize its transpose. The outputs are
            identical either way. K_BY_F requires quant_scheme='1x32'; it loads with the
            DMA + tensor_copy 1x32 swizzle.

    Returns:
        quantized_scales_hbm (nl.NkiTensor): [K // 32, F], Dense scales in float8_e8m0fnu format on HBM
        quantized_data_hbm (nl.NkiTensor): [K // 4, F * INTERLEAVE_FACTOR], Quantized data in FP8 format on HBM

    Notes:
        - K dimension must be divisible by MX_PARTITION_SIZE (32), the MX quantization block size
        - F dimension must be divisible by 8 for quantization
        - K not divisible by L_TILE_K is processed as full 512 tiles plus the remainder: for wrapX
          a 256/128/sub-128 decomposition (see decompose_k_remainder), for 1x32 one tile of the
          whole remainder. A tile whose K / 4 is not a multiple of MX_PARTITION_SIZE is quantized as
          a padded tile; only its valid rows reach HBM.
        - wrapX with K % 128 != 0 is loaded with the wrapX PE transpose rather than the DMA
          gather-transpose: DGT cannot express the sub-128 K tiles (see the loader-selection
          comment in the body). The output layout is identical either way.
        - LNC2 splits F in halves at a SUB_TILE boundary; a core whose F range fits in one
          L_TILE_F tile processes it as two half-width tiles so load, quantize and store overlap
        - Scales are kept in SBUF for a group of F tiles (the core's whole F range unless that
          exceeds _SCALE_SBUF_BUDGET_BYTES) and stored after the group's tiles: one DMA per
          quadrant for all full K tiles, and per quadrant for each K remainder tile
        - TODO: Add support for splitting along K dimension for LNC2
        - TODO: .view() is not supported for x4 to non-x4 conversion, so BIR ap() is used instead

    Pseudocode:
        # Allocate output tensors
        quantized_data_hbm = allocate([K // 4, F * INTERLEAVE_FACTOR], fp8_dtype)
        quantized_scales_hbm = allocate([K // 32, F], float8_e8m0fnu)

        # Process this core's F range in groups of F tiles
        for f_group in f_tile_groups(f_begin, f_end):
            scales_sbuf = allocate([P_MAX, num_k_tiles, f_group.width], float8_e8m0fnu)
            for tile_f_idx in f_group:
                for tile_k_size, k_offset in k_tiles(K):
                    # Load and interleave tile (DGT for wrapX, FP32 PE transpose / tensor copy for 1x32)
                    src_tile = load_tile(src_tensor, tile_k_size, k_offset, tile_f_idx)

                    # Quantize the tile; its scales land in scales_sbuf
                    quantized_data = quantize_mxfp8_block(src_tile, tile_size, return_fp8_dtype, scales_sbuf)

                    # Store quantized data
                    store(quantized_data_hbm, quantized_data, k_offset, tile_f_idx)

            # Store the group's scales, one DMA per quadrant per K-tile kind
            store_dense_scales(quantized_scales_hbm, scales_sbuf, f_group)

        return quantized_scales_hbm, quantized_data_hbm
    """

    if get_active_sbm() is None:
        create_and_set_active_sbm()
        sbm = get_active_sbm()
        sbm.open_scope(name="QUANTIZE KERNEL")

    sbm = get_active_sbm()

    # The input is [F, K] unless the caller says it is [K, F]; the output layout is the same.
    is_k_by_f = src_orientation == TensorOrientation.K_BY_F
    if is_k_by_f:
        K, F = src_tensor.shape
    else:
        F, K = src_tensor.shape

    L_TILE_K = _quantize_mxfp8_utils.L_TILE_K
    L_TILE_F = _quantize_mxfp8_utils.L_TILE_F

    # Required, with no default: the default layout changed from wrapX to 1x32, so a caller that
    # relied on it must choose explicitly rather than silently get the other layout.
    kernel_assert(
        quant_scheme is not None,
        f"quant_scheme is required: pass '{QUANT_SCHEME_1X32}' (recommended) or '{QUANT_SCHEME_WRAPX}'.",
    )
    kernel_assert(
        quant_scheme in QUANT_SCHEME_MAP,
        f"Invalid quant_scheme '{quant_scheme}', must be '{QUANT_SCHEME_WRAPX}' or '{QUANT_SCHEME_1X32}'.",
    )
    resolved_quant_scheme = QUANT_SCHEME_MAP[quant_scheme]
    is_1x32 = resolved_quant_scheme == QuantScheme._1x32
    kernel_assert(
        not is_k_by_f or is_1x32,
        f"src_orientation=K_BY_F is only supported with quant_scheme='{QUANT_SCHEME_1X32}'; "
        f"wrapX loads F-by-K input only.",
    )
    if not is_1x32:
        logger.warn(
            f"quantize_block_mxfp8_kernel: quant_scheme='{QUANT_SCHEME_WRAPX}' output must NOT be composed along "
            f"K. The wrapX layout depends on how this call tiled K (including its K remainder), so it is illegal "
            f"to concatenate, shard, or all-gather wrapX outputs along K (e.g. FSDP K-shards), or to pass one to "
            f"a consumer that tiles K differently: either silently contracts mismatched K and returns wrong "
            f"results, with no error. Only consume it whole, in one matmul that loads it with this kernel's K "
            f"tiling. Use quant_scheme='{QUANT_SCHEME_1X32}', whose layout does not depend on the K tiling, for "
            f"anything else. Got {K=}.",
        )

    kernel_assert(
        K % MX_PARTITION_SIZE == 0,
        f"K dimension must be divisible by {MX_PARTITION_SIZE} for mxfp8 quantization, got {K=}",
    )
    kernel_assert(
        F % _quantize_mxfp8_utils.MIN_F_FOR_QUANTIZATION == 0,
        f"F dim must be divisible by {_quantize_mxfp8_utils.MIN_F_FOR_QUANTIZATION} for quantization, got {F=}",
    )

    fp8_dtype = _quantize_mxfp8_utils.get_fp8_dtype(return_fp8_dtype)
    INTERLEAVE_FACTOR = _quantize_mxfp8_utils.INTERLEAVE_FACTOR
    quantized_data_hbm = nl.ndarray(
        (K // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR),
        dtype=fp8_dtype,
        buffer=nl.shared_hbm,
    )

    K_scales, F_scales = get_dense_scale_shape(K, F)
    quantized_scales_hbm = nl.ndarray((K_scales, F_scales), dtype=nl.float8_e8m0fnu, buffer=nl.shared_hbm)

    # LNC2 halves F at a SUB_TILE boundary (not at an L_TILE_F tile boundary, which leaves one
    # core idle for a tile whenever F spans an odd number of tiles); each core then tiles its own
    # F range by TILE_F (below).
    if run_with_lnc2:
        LNC_ID = nl.program_id(axis=0)
        kernel_assert(
            F >= _quantize_mxfp8_utils.MIN_F_FOR_LNC2,
            f"LNC2 requires F >= {_quantize_mxfp8_utils.MIN_F_FOR_LNC2}, but got {F=}",
        )
        f_split = ((F + 2 * SUB_TILE - 1) // (2 * SUB_TILE)) * SUB_TILE
        if LNC_ID == 0:
            f_begin = 0
            f_end = f_split
        else:
            f_begin = f_split
            f_end = F
    else:
        f_begin = 0
        f_end = F
    # A core whose F range fits in one L_TILE_F tile would run load -> quantize -> store strictly in
    # sequence; halving the F tile lets the second half's load overlap the first half's quantize and
    # store. Larger F ranges already pipeline across tiles, and smaller tiles there only add DMAs.
    # The legacy DGT arm keeps L_TILE_F: its vector-offset patterns are generated for that width.
    TILE_F = L_TILE_F
    if (is_1x32 or use_fast_dma_transpose) and f_end - f_begin <= L_TILE_F:
        TILE_F = L_TILE_F // 2
    NUM_TILES_IN_F = (f_end - f_begin + TILE_F - 1) // TILE_F

    # Full L_TILE_K tiles, then the K remainder. wrapX splits the remainder into at most one 256, one
    # 128 and one sub-128 tile, because its layout depends on the K tiling. The 1x32 layout does not,
    # so 1x32 loads the whole remainder as one tile (a 32-aligned tile_k below L_TILE_K, its K / 4 rows
    # quantized padded up to a multiple of MX_PARTITION_SIZE): one load, swizzle and store instead of
    # up to three.
    NUM_FULL_K_TILES = K // L_TILE_K
    if is_1x32:
        K_REMAINDER_TILES = [K % L_TILE_K] if K % L_TILE_K else []
    else:
        K_REMAINDER_TILES = decompose_k_remainder(K % L_TILE_K)

    # Loader selection. The caller's use_fast_dma_transpose wins when set; K % 128 only picks the
    # fallback for when it is not. SwizzleMode.PE additionally keeps the descriptor off the DGT
    # vector-offset setup path, which a PE load does not need.
    #   1x32                        -> load_tile dispatches on quant_scheme to the FP32 PE 1x32 loader
    #                                  (F-by-K) or the tensor-copy 1x32 loader (K-by-F).
    #   wrapX, fast DMA transpose   -> DGT_FAST for every tile, remainders included.
    #   wrapX, flag off, K % 128    -> legacy DGT, unchanged.
    #   wrapX, flag off, K % 128 != 0 -> wrapX PE transpose. Legacy DGT cannot express these shapes
    #                                  at all: it flattens K by tile_k // INTERLEAVE_FACTOR, so it
    #                                  needs K % (tile_k / 4) == 0, which the 512 tiles already fail
    #                                  (K=1696 -> 1696 % 128 = 32), and set_vector_offset_patterns
    #                                  asserts K % 128 == 0 outright. PE produces the identical
    #                                  layout with no tiling bound.
    # The fast DMA transpose needs every K tile's load destination to have the same height: with
    # destinations sized to each tile (64, then 32, then 32 partitions for a multi-tile remainder)
    # it writes wrong data. So on that path every tile, remainder tiles included, loads into a
    # P_MAX-partition destination and quantizes from its first p_quant partitions. Only the fast
    # path pays for it (a memset on the tiles that need one); the legacy and PE paths are unchanged.
    uniform_dest_height = use_fast_dma_transpose and not is_1x32

    if is_1x32:
        swizzle_mode = SwizzleMode.PE
    elif use_fast_dma_transpose:
        swizzle_mode = SwizzleMode.DGT_FAST
    elif K % _quantize_mxfp8_utils.MIN_K_FOR_QUANTIZATION != 0:
        swizzle_mode = SwizzleMode.PE
    else:
        swizzle_mode = SwizzleMode.DGT

    src_td = TensorDescriptor(
        data=src_tensor,
        is_swizzled=False,
        orientation=TensorOrientation.K_BY_F if is_k_by_f else TensorOrientation.F_BY_K,
        quant_scheme=resolved_quant_scheme,
        swizzle_mode=swizzle_mode,
    )
    if swizzle_mode == SwizzleMode.DGT:
        src_td.set_vector_offset_patterns(L_TILE_K, L_TILE_F)

    # Hardware DGE queue for the stores (and the 1x32 loads). The DMA transpose loads are issued
    # from Sync, so the DGT arms store from Scalar; the 1x32 arm keeps Scalar for its PSUM drains.
    dma_engine = nisa.engine.sync if is_1x32 else nisa.engine.scalar

    # Scales stay in SBUF for a group of F tiles and are stored after the group in a few wide
    # DMAs: per K tile and F tile they are four tiny per-quadrant DMAs, which the compiler
    # schedules after all the data stores, where their latency serializes into a long tail.
    # A group is the core's whole F range unless its scales would exceed _SCALE_SBUF_BUDGET_BYTES
    # per partition, which bounds SBUF use for large K x F.
    # Full K tile t is free-dim slot t of full_tile_scales, remainder tile r is remainder_scales[r],
    # and F tile i of the group is at columns i * TILE_F. A padded last F tile allocates at most
    # TILE_F columns, so the group needs no extra columns.
    num_scale_slots = NUM_FULL_K_TILES + len(K_REMAINDER_TILES)
    TILES_PER_GROUP = max(1, _SCALE_SBUF_BUDGET_BYTES // (num_scale_slots * TILE_F))
    NUM_GROUPS = (NUM_TILES_IN_F + TILES_PER_GROUP - 1) // TILES_PER_GROUP
    scale_engines = (nisa.engine.sync, nisa.engine.scalar)

    for group_idx in range(NUM_GROUPS):
        group_tile_begin = group_idx * TILES_PER_GROUP
        group_num_tiles = min(TILES_PER_GROUP, NUM_TILES_IN_F - group_tile_begin)
        group_f_begin = f_begin + group_tile_begin * TILE_F
        group_f_end = min(f_end, group_f_begin + group_num_tiles * TILE_F)
        group_cols = group_num_tiles * TILE_F

        sbm.open_scope(name="QUANTIZE SCALE GROUP")
        if NUM_FULL_K_TILES > 0:
            full_tile_scales = sbm.alloc_stack(
                shape=(P_MAX, NUM_FULL_K_TILES, group_cols), dtype=nl.float8_e8m0fnu, buffer=nl.sbuf
            )
        remainder_scales = [
            sbm.alloc_stack(shape=(P_MAX, 1, group_cols), dtype=nl.float8_e8m0fnu, buffer=nl.sbuf)
            for _ in K_REMAINDER_TILES
        ]

        for tile_f_local_idx in nl.affine_range(group_num_tiles):
            f_local = tile_f_local_idx * TILE_F
            f_offset = group_f_begin + f_local
            current_tile_f = min(TILE_F, group_f_end - f_offset)
            # Both PE loaders gather whole SUB_TILE-wide F sub-tiles, so a short final F tile is
            # loaded into a padded buffer (the loader zeroes the tail) and stored back clamped.
            # The K-by-F tensor-copy loader takes any multiple of 8, so it loads the exact width.
            if swizzle_mode == SwizzleMode.PE and not is_k_by_f:
                tile_f_alloc = ((current_tile_f + SUB_TILE - 1) // SUB_TILE) * SUB_TILE
            else:
                tile_f_alloc = current_tile_f

            tile_kwargs = dict(
                current_tile_f=current_tile_f,
                f_offset=f_offset,
                INTERLEAVE_FACTOR=INTERLEAVE_FACTOR,
                fp8_dtype=fp8_dtype,
                return_fp8_dtype=return_fp8_dtype,
                quantized_data_hbm=quantized_data_hbm,
                dma_engine=dma_engine,
                tile_f_alloc=tile_f_alloc,
                uniform_dest_height=uniform_dest_height,
            )
            # K-by-F 1x32 tiles take the DMA + tensor_copy swizzle: each partition reads its
            # 4 contiguous K rows and one SBUF copy interleaves them, with no PE transposes
            # and no PSUM round trip. Measured at parity with the F-by-K PE loader (the
            # K-by-F PE loader, which DMA-transposes first, was up to 1.5x slower).
            for k_tile_idx in range(NUM_FULL_K_TILES):
                _process_tile(
                    src_td,
                    L_TILE_K,
                    k_offset=k_tile_idx * L_TILE_K,
                    use_tensor_copy_swizzle=is_k_by_f,
                    scales_sbuf=full_tile_scales[:, nl.ds(k_tile_idx, 1), nl.ds(f_local, tile_f_alloc)],
                    **tile_kwargs,
                )
            k_offset = NUM_FULL_K_TILES * L_TILE_K
            for rem_idx, tile_k_size in enumerate(K_REMAINDER_TILES):
                _process_tile(
                    src_td,
                    tile_k_size,
                    k_offset=k_offset,
                    use_tensor_copy_swizzle=is_k_by_f,
                    scales_sbuf=remainder_scales[rem_idx][:, :, nl.ds(f_local, tile_f_alloc)],
                    **tile_kwargs,
                )
                k_offset += tile_k_size

        # Store the group's scales: the full tiles in one DMA per quadrant, then each remainder tile.
        if NUM_FULL_K_TILES > 0:
            store_dense_scales(
                quantized_scales_hbm,
                hbm_row=0,
                hbm_f=group_f_begin,
                num_f=group_f_end - group_f_begin,
                num_data_rows=P_MAX,
                scales_sbuf=full_tile_scales,
                num_tiles=NUM_FULL_K_TILES,
                dma_engines=scale_engines,
            )
        k_offset = NUM_FULL_K_TILES * L_TILE_K
        for rem_idx, tile_k_size in enumerate(K_REMAINDER_TILES):
            store_dense_scales(
                quantized_scales_hbm,
                hbm_row=k_offset // MX_PARTITION_SIZE,
                hbm_f=group_f_begin,
                num_f=group_f_end - group_f_begin,
                num_data_rows=tile_k_size // INTERLEAVE_FACTOR,
                scales_sbuf=remainder_scales[rem_idx],
                dma_engines=scale_engines,
            )
            k_offset += tile_k_size
        sbm.close_scope()

    return quantized_scales_hbm, quantized_data_hbm
