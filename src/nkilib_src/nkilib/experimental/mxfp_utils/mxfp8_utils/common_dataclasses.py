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

"""Tensor descriptor for managing quantized tensor metadata and layout information."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class QuantScheme(Enum):
    """Quantization scheme for MXFP8 PE swizzle layout."""

    WRAPX = "wrapX"
    _1x32 = "1x32"


class SwizzleMode(Enum):
    """Mechanism used to convert one unswizzled tensor to matmul layout."""

    DGT = "dgt"
    PE = "pe"


import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ...moe.bwd.moe_bwd_parameters import SkipMode
from .common_utils import get_active_sbm

P_MAX = 128  # Hardware partition dimension size
TRANSPOSE_CHUNK_SIZE = 32  # Element count per nc_transpose chunk
NUM_TRANSPOSE_CHUNKS = 4  # Number of chunks in one partition (P_MAX // TRANSPOSE_CHUNK_SIZE)
from .quantize_mxfp8_utils import INTERLEAVE_FACTOR, L_TILE_K, are_scales_packed


@dataclass
class TensorDescriptor(nl.NKIObject):
    """
    Descriptor for managing quantized tensor data and metadata.

    Encapsulates tensor data, quantization scales, and layout flags to track
    various transformations applied to tensors during computation.

    Can be constructed with data and optional scales:
        TensorDescriptor(data=tensor, scales=scales, is_swizzled=True, ...)
        If scales are provided, the tensor is treated as quantized and is_x4/is_swizzled
        are auto-detected. If unswizzled BF16, is_f_by_k is set automatically.

    Args:
        data (Optional[nl.ndarray]): Primary tensor data, None if not specified.
        scales (Optional[nl.ndarray]): Quantization scale factors, None if unquantized.
        is_swizzled (bool): True if tensor is in [P/4, F*4] swizzled format.
        is_f_by_k (bool): True if tensor is in the [F, P] ([F, K]) orientation.
        is_x4 (bool): True if data is in _x4 packed format.
        scales_are_packed (bool): True if scales are packed.
        is_col_parallel_sharded (bool): True if tensor is sharded across 2 cores (LNC2).
        swizzle_mode (SwizzleMode): Conversion mechanism for unswizzled BF16.
        load_with_PE_swizzle (bool): Legacy PE transpose selector retained for
            compatibility. When True, use PE transpose
            (load_tile_PE_swizzle_wrapX) instead of DGT.
            Supports both direct (contiguous) and indirect (scattered) DMA modes;
            indirect mode is activated when indirect_dma_vector_offset is set.
        indirect_dma_vector_offset (Optional[nl.ndarray]): SBUF
            int32 tensor of shape (P_MAX, NUM_SUB_TILES) holding global F-row indices
            for indirect DMA gather when load_with_PE_swizzle=True.
        psum_drain_engine_ratio (Tuple[int, int]): (num_scalar, num_vector) split of the
            PE swizzle PSUM copy-outs across the Scalar and Vector engines. Only applies
            when load_with_PE_swizzle=True. Defaults to (1, 1) (alternate). Both entries
            must be >= 0 and at least one must be non-zero.
        scalar_offset (Optional[nl.ndarray]): Currently unsupported. Runtime scalar
            offset added to every DGT vector_offset entry for per-expert weight slicing.
            Must be float32 dtype, pre-scaled into vector_size units.
        effective_f_dim (Optional[int]): Currently unsupported. Per-expert logical F
            dimension for stacked-expert tensors to clamp DGT tile_f at expert boundaries.
        skip_dma (Optional[SkipMode]): Currently unsupported. Controls OOB handling for
            DMA operations. When skip_dma.skip_token is True, out-of-bounds indices are
            skipped via oob_mode.skip.

    Notes:
        - Layout flags track cumulative transformations
        - Multiple flags can be True simultaneously
        - physical_shape and logical_shape are computed automatically from data
    """

    data: Optional[nl.ndarray] = None
    scales: Optional[nl.ndarray] = None
    is_swizzled: bool = False
    is_f_by_k: Optional[bool] = None
    is_x4: bool = False
    scales_are_packed: bool = False
    is_col_parallel_sharded: bool = False
    is_quantized: bool = False
    is_unswizzled_bf16: bool = False
    physical_shape: Optional[Tuple[int, int]] = None
    logical_shape: Optional[Tuple[int, int]] = None
    sharded_physical_shape: Optional[Tuple[int, int]] = None
    sharded_logical_shape: Optional[Tuple[int, int]] = None
    vector_offset_pattern_512: Optional[nl.ndarray] = None
    vector_offset_pattern_256: Optional[nl.ndarray] = None
    vector_offset_pattern_128: Optional[nl.ndarray] = None

    # Quantization scheme for PE swizzle layout.
    # WRAPX (default): 4-partition interleave (DGT / load_tile_PE_swizzle_wrapX)
    # _1x32: 1x32 contiguous block layout (load_tile_PE_Swizzle_1x32 / load_tile_bf16_xbar_transpose)
    quant_scheme: QuantScheme = QuantScheme.WRAPX

    # Tensor-local conversion instruction for unswizzled BF16.
    swizzle_mode: SwizzleMode = SwizzleMode.DGT

    # Legacy selector retained during migration to swizzle_mode.
    load_with_PE_swizzle: bool = False

    # PE swizzle supports both:
    #   - Direct DMA (contiguous rows): when indirect_dma_vector_offset is None
    #   - Indirect DMA (scattered token gather): when indirect_dma_vector_offset
    #     is set to an SBUF int32 tensor of shape (P_MAX, NUM_SUB_TILES) holding
    #     global F-row indices.
    # Whether indirect or direct is determined by the presence of vector_offset
    # on the TileLocation at load time.
    # When True, use a direct access pattern on the source tensor for DMA
    # gather-transpose instead of flattening + vector offsets. This avoids
    # generating vector_offset_pattern buffers in SBUF and simplifies the DGT
    # path by specifying a 4D access pattern directly on the [F, K] source:
    #   pattern=[[VECTOR_SIZE, INTERLEAVE], [1, 1], [K, TILE_F], [1, VECTOR_SIZE]]
    #   offset=f_offset * K + k_offset
    fast_dma_transpose: bool = False
    indirect_dma_vector_offset: Optional[nl.ndarray] = None

    # How to split the PE swizzle PSUM copy-outs (drains) between the Scalar and
    # Vector engines, as (num_scalar, num_vector) per repeating group of drains.
    # Draining entirely on Vector saturates it and stalls the transpose, so the
    # work is spread across both engines. Examples:
    #   (1, 1) -> alternate, half on each engine (default)
    #   (2, 1) -> two thirds on Scalar
    #   (0, 1) -> everything on Vector
    psum_drain_engine_ratio: Tuple[int, int] = (1, 1)

    # Runtime scalar offset added uniformly to every DGT vector_offset entry
    # at TileLocation construction time. Used to select per-expert weight
    # slices when the data tensor is a 2D reshape view of a stacked-experts
    # tensor (e.g. [E*I_TP, H] view of a [E, I_TP, H] tensor). The caller is
    # responsible for pre-scaling the runtime expert index into vector_size
    # units (vector_size = tile_k // INTERLEAVE_FACTOR), since vector_offset
    # values index the flattened (F*K/vector_size, ..., vector_size) DGT view.
    scalar_offset: Optional[nl.ndarray] = None
    scalar_offset_scales: Optional[nl.ndarray] = None
    effective_f_dim_scales: Optional[int] = None
    # Per-expert logical F dimension for stacked-expert tensors. When a tensor
    # is reshaped from [E, F_per_expert, K] to [E*F_per_expert, K] and scalar_offset
    # is used to index experts, the DGT access pattern must clamp tile_f to the
    # per-expert F boundary (not the full tensor F). Without this, the last F-tile
    # within an expert overflows into the next expert's rows (or past the tensor end).
    effective_f_dim: Optional[int] = None

    # OOB mode for indirect DMA loads. When set to oob_mode.skip, out-of-bounds
    # indices (e.g., -1 padding slots in MoE skip_token mode) are skipped by the
    # DMA engine instead of triggering a runtime error. The destination SBUF is
    # pre-zeroed before the DMA so skipped partitions read as zero.
    skip_dma: SkipMode = None

    def _get_physical_shape(self, data, is_quantized, is_x4, is_swizzled):
        """Compute physical (K, second_dim) shape from raw tensor data and layout flags."""
        if is_quantized:
            if is_x4:
                return (data.shape[0], data.shape[1])
            else:
                return (data.shape[0], data.shape[1] // INTERLEAVE_FACTOR)
        else:
            # Swizzled is stored (K, F), and K-by-F is already (K, F); both are in
            # (K, second_dim) order, so return as-is. Only F-by-K ([F, K]) needs the swap.
            if is_swizzled or not self.is_f_by_k:
                return (data.shape[0], data.shape[1])
            else:
                F, K = data.shape
                return (K, F)

    def _get_logical_shape(self, physical_shape, is_quantized, is_swizzled):
        """Convert physical (K, second_dim) shape to logical shape, accounting for interleave factor."""
        K_physical, second_physical = physical_shape
        K_logical = K_physical * INTERLEAVE_FACTOR if is_swizzled else K_physical
        second_logical = second_physical if is_quantized or not is_swizzled else second_physical // INTERLEAVE_FACTOR
        return (K_logical, second_logical)

    def __post_init__(self):
        if not isinstance(self.swizzle_mode, SwizzleMode):
            raise TypeError("swizzle_mode must be a SwizzleMode")

        self.is_quantized = self.scales != None
        # Auto-detect x4 format from dtype when quantized
        if self.is_quantized and not self.is_x4 and self.data != None:
            self.is_x4 = 'x4' in str(self.data.dtype)
        # Quantized inputs are always swizzled
        if self.is_quantized:
            self.is_swizzled = True
            # Auto-detect packed scales from data/scales shape mismatch (2D HBM tensors only)
            if self.data != None and not self.scales_are_packed and len(self.data.shape) == 2:
                self.scales_are_packed = are_scales_packed(self.data.shape, self.scales.shape)
        self.is_unswizzled_bf16 = not self.is_quantized and not self.is_swizzled
        # Resolve is_f_by_k (it may be None = "unset/auto" here):
        #   - unswizzled BF16 + explicit K-by-F (is_f_by_k == False) -> load via PE swizzle
        #   - unswizzled BF16 + unset(None)/F-by-K               -> F-by-K (set True below)
        #   - swizzled/quantized (layout irrelevant) + unset      -> coerced to False at the end
        # (None == False is False, so None never takes the K-by-F branch.)
        if self.is_unswizzled_bf16 and self.is_f_by_k == False:
            # K-by-F unswizzled input: must use PE swizzle (DGT requires F-by-K)
            self.load_with_PE_swizzle = True
        elif self.is_unswizzled_bf16:
            # Default (None) or explicit True: F-by-K layout
            self.is_f_by_k = True
        # For non-unswizzled-bf16 tensors (quantized/swizzled), ensure is_f_by_k is a concrete bool
        if self.is_f_by_k is None:
            self.is_f_by_k = False

        num_scalar, num_vector = self.psum_drain_engine_ratio
        kernel_assert(
            num_scalar >= 0 and num_vector >= 0 and (num_scalar + num_vector) > 0,
            f"psum_drain_engine_ratio must be non-negative and not both zero, got {self.psum_drain_engine_ratio}",
        )

        # Compute shapes
        if self.data != None and len(self.data.shape) == 2:
            self.physical_shape = self._get_physical_shape(self.data, self.is_quantized, self.is_x4, self.is_swizzled)
            self.logical_shape = self._get_logical_shape(self.physical_shape, self.is_quantized, self.is_swizzled)

            if self.is_col_parallel_sharded:
                self.sharded_physical_shape = (self.physical_shape[0], self.physical_shape[1] // 2)
                self.sharded_logical_shape = (self.logical_shape[0], self.logical_shape[1] // 2)
            else:
                self.sharded_physical_shape = self.physical_shape
                self.sharded_logical_shape = self.logical_shape

    @property
    def uses_pe_swizzle(self):
        """Whether this tensor uses PE swizzle conversion."""
        return self.load_with_PE_swizzle or self.swizzle_mode == SwizzleMode.PE

    def shard_col_parallel(self):
        """Enable column-parallel sharding, halving the second (F/M/N) dimension."""
        self.is_col_parallel_sharded = True
        self.sharded_physical_shape = (self.physical_shape[0], self.physical_shape[1] // 2)
        self.sharded_logical_shape = (self.logical_shape[0], self.logical_shape[1] // 2)

    def _generate_vector_offset_pattern(self, tile_k: int, tile_f: int) -> nl.ndarray:
        """
        Generate vector offset pattern for DMA gather transpose operations.

        Creates an index pattern used to gather and transpose data from a flattened
        [tile_f * tile_k / vector_size, vector_size] tensor into an interleaved and transposed
        [P_MAX, tile_f * tile_k / P_MAX] tensor, where vector_size = tile_k // INTERLEAVE_FACTOR.

        Args:
            tile_k (int): Tile size in K dimension for loading (512, 256, or 128)
            tile_f (int): Tile size in F dimension for loading (typically 512)

        Returns:
            nl.ndarray: Shape (P_MAX, NUM_PARTITIONS) containing uint32 indices for DMA gather operations.

        Notes:
            Intermediate calculations:
                vector_size: Flatten chunk size (tile_k // INTERLEAVE_FACTOR)
                COUNT_K: Number of vector_size-element chunks in tile_k (always INTERLEAVE_FACTOR)
                SKIP_K: Stride between F rows in flattened layout (K // vector_size)
                NUM_TILE_INDICES: Total indices needed (COUNT_K * tile_f)
                NUM_PARTITIONS: Number of P_MAX-element partitions (NUM_TILE_INDICES // P_MAX)
                COUNT_F_PER_PARTITION: F elements per partition (P_MAX // COUNT_K)
                CHANNEL_MULTIPLIER: Offset between partitions (SKIP_K * COUNT_F_PER_PARTITION)

            Example:
                K = 2048, tile_k = 512, tile_f = 512

                Derived values:
                    vector_size = 512 // 4 = 128
                    COUNT_K = 512 // 128 = 4
                    SKIP_K = 2048 // 128 = 16
                    NUM_TILE_INDICES = 4 * 512 = 2048
                    NUM_PARTITIONS = 2048 // 128 = 16
                    COUNT_F_PER_PARTITION = 128 // 4 = 32
                    CHANNEL_MULTIPLIER = 16 * 32 = 512

                The iota pattern [[SKIP_K, COUNT_F_PER_PARTITION], [1, COUNT_K]] = [[16, 32], [1, 4]]
                generates indices that interleave K-chunks with F-stride.

                After transpose - vector_offsets_sbuf shape (128, 16):
                    Row 0:   [0,    512,  1024, 1536, ...]  <- 1st K-chunk for F=0, 32, 64, 96, ...
                    Row 1:   [1,    513,  1025, 1537, ...]  <- 2nd K-chunk for F=0, 32, 64, 96, ...
                    Row 2:   [2,    514,  1026, 1538, ...]  <- 3rd K-chunk for F=0, 32, 64, 96, ...
                    Row 3:   [3,    515,  1027, 1539, ...]  <- 4th K-chunk for F=0, 32, 64, 96, ...

                These indices enable gather-transpose that converts (F, K) data into
                (K, F) interleaved tiles for MXFP8 quantization.
        """
        _, K = self.data.shape

        vector_size = tile_k // INTERLEAVE_FACTOR
        COUNT_K = tile_k // vector_size
        SKIP_K = K // vector_size
        NUM_TILE_INDICES = COUNT_K * tile_f
        NUM_PARTITIONS = NUM_TILE_INDICES // P_MAX
        COUNT_F_PER_PARTITION = P_MAX // COUNT_K
        CHANNEL_MULTIPLIER = SKIP_K * COUNT_F_PER_PARTITION

        kernel_assert(
            NUM_PARTITIONS <= TRANSPOSE_CHUNK_SIZE,
            f"_generate_vector_offset_pattern: nc_transpose requires NUM_PARTITIONS <= {TRANSPOSE_CHUNK_SIZE}, "
            f"got NUM_PARTITIONS={NUM_PARTITIONS} (tile_k={tile_k}, tile_f={tile_f})",
        )

        sbm = get_active_sbm()
        vector_offsets_sbuf = sbm.alloc_stack((P_MAX, NUM_PARTITIONS), dtype=nl.uint32, buffer=nl.sbuf)
        vector_offsets_tmp = sbm.alloc_stack((NUM_PARTITIONS, P_MAX), dtype=nl.uint32, buffer=nl.sbuf)

        nisa.iota(
            dst=vector_offsets_tmp,
            pattern=[[SKIP_K, COUNT_F_PER_PARTITION], [1, COUNT_K]],
            offset=0,
            channel_multiplier=CHANNEL_MULTIPLIER,
        )

        for transpose_chunk_idx in range(NUM_TRANSPOSE_CHUNKS):
            chunk_start = transpose_chunk_idx * TRANSPOSE_CHUNK_SIZE
            nisa.nc_transpose(
                vector_offsets_sbuf[nl.ds(chunk_start, TRANSPOSE_CHUNK_SIZE), 0:NUM_PARTITIONS],
                vector_offsets_tmp[0:NUM_PARTITIONS, nl.ds(chunk_start, TRANSPOSE_CHUNK_SIZE)],
            )

        return vector_offsets_sbuf

    def set_vector_offset_patterns(self, tile_k, tile_f):
        """
        Generate and store vector offset patterns for DMA gather transpose operations.

        Generates the main pattern for tile_k, plus remainder patterns (256 and/or 128)
        when K is not divisible by tile_k.

        Args:
            tile_k (int): Tile size in K dimension for loading (typically 512)
            tile_f (int): Tile size in F dimension for loading (typically 512)
        """
        kernel_assert(
            not self.is_swizzled and self.is_f_by_k,
            "Tensor must not be swizzled and must be f_by_k to set vector offset pattern",
        )
        K = self.data.shape[1]
        kernel_assert(K % 128 == 0, f"K must be divisible by 128, got {K}")
        remainder = K % tile_k
        if K >= tile_k:
            self.vector_offset_pattern_512 = self._generate_vector_offset_pattern(tile_k, tile_f)
        if remainder >= 256:
            self.vector_offset_pattern_256 = self._generate_vector_offset_pattern(256, tile_f)
        if remainder % 256 >= 128:
            self.vector_offset_pattern_128 = self._generate_vector_offset_pattern(128, tile_f)

    def get_vector_offset_pattern(self, tile_k: int = None):
        """
        Retrieve the pre-computed vector offset pattern for DMA gather transpose.

        Args:
            tile_k (int): Tile K size to select the correct pattern. If None, returns the main pattern.

        Returns:
            nl.ndarray: Shape (P_MAX, NUM_PARTITIONS) containing uint32 indices for DMA gather operations.
        """
        if tile_k == 256 and self.vector_offset_pattern_256 != None:
            return self.vector_offset_pattern_256
        if tile_k == 128 and self.vector_offset_pattern_128 != None:
            return self.vector_offset_pattern_128
        kernel_assert(self.vector_offset_pattern_512 != None, "Must call set_vector_offset_pattern first")
        return self.vector_offset_pattern_512


@dataclass
class BlockDescriptor(nl.NKIObject):
    """
    Descriptor for block dimensions used in tiled matrix multiplication.

    Computes logical and physical block sizes from tile shapes and tile counts.
    Also provides spill/reload-adjusted physical sizes.

    Args:
        TILES_IN_BLOCK_M (int): Number of tiles per block in M dimension
        TILES_IN_BLOCK_N (int): Number of tiles per block in N dimension
        TILES_IN_BLOCK_K (int): Number of tiles per block in K dimension
        lhs_matmul_tile_shape_logical (tuple): (TILE_K, TILE_M) logical tile shape for LHS
        rhs_matmul_tile_shape_logical (tuple): (TILE_K, TILE_N) logical tile shape for RHS
        lhs_load_tile_shape (tuple): (TILE_K, TILE_M) physical load tile shape for LHS
        rhs_load_tile_shape (tuple): (TILE_K, TILE_N) physical load tile shape for RHS
    """

    TILES_IN_BLOCK_M: int = 0
    TILES_IN_BLOCK_N: int = 0
    TILES_IN_BLOCK_K: int = 0
    lhs_matmul_tile_shape_logical: tuple = (0, 0)
    rhs_matmul_tile_shape_logical: tuple = (0, 0)
    lhs_load_tile_shape: tuple = (0, 0)
    rhs_load_tile_shape: tuple = (0, 0)

    # Computed fields
    BLOCK_M_LOGICAL: int = 0
    BLOCK_N_LOGICAL: int = 0
    BLOCK_K_LOGICAL: int = 0
    BLOCK_M_PHYSICAL: int = 0
    BLOCK_N_PHYSICAL: int = 0
    BLOCK_K_PHYSICAL_LHS: int = 0
    BLOCK_K_PHYSICAL_RHS: int = 0

    def __post_init__(self):
        if self.BLOCK_M_LOGICAL == 0:
            self.BLOCK_M_LOGICAL = self.TILES_IN_BLOCK_M * self.lhs_matmul_tile_shape_logical[1]
        if self.BLOCK_N_LOGICAL == 0:
            self.BLOCK_N_LOGICAL = self.TILES_IN_BLOCK_N * self.rhs_matmul_tile_shape_logical[1]
        if self.BLOCK_K_LOGICAL == 0:
            self.BLOCK_K_LOGICAL = self.TILES_IN_BLOCK_K * self.lhs_matmul_tile_shape_logical[0]
        if self.BLOCK_M_PHYSICAL == 0:
            self.BLOCK_M_PHYSICAL = self.TILES_IN_BLOCK_M * self.lhs_load_tile_shape[1]
        if self.BLOCK_N_PHYSICAL == 0:
            self.BLOCK_N_PHYSICAL = self.TILES_IN_BLOCK_N * self.rhs_load_tile_shape[1]
        if self.BLOCK_K_PHYSICAL_LHS == 0:
            self.BLOCK_K_PHYSICAL_LHS = self.TILES_IN_BLOCK_K * self.lhs_load_tile_shape[0]
        if self.BLOCK_K_PHYSICAL_RHS == 0:
            self.BLOCK_K_PHYSICAL_RHS = self.TILES_IN_BLOCK_K * self.rhs_load_tile_shape[0]

    def get_spill_reload_bd(self, lhs_is_swizzled, rhs_is_swizzled, lhs_no_spill_reload, rhs_no_spill_reload):
        """
        Get a new BlockDescriptor with physical sizes adjusted for spill/reload.

        When spill-reloading, data is in quantized x4 format (logical layout),
        so physical sizes collapse to logical. For K dimension, unswizzled tensors
        also need INTERLEAVE_FACTOR adjustment.

        Returns:
            BlockDescriptor with adjusted physical sizes.
        """
        return BlockDescriptor(
            BLOCK_M_PHYSICAL=self.BLOCK_M_PHYSICAL if lhs_no_spill_reload else self.BLOCK_M_LOGICAL,
            BLOCK_K_PHYSICAL_LHS=(
                self.BLOCK_K_PHYSICAL_LHS // INTERLEAVE_FACTOR
                if not lhs_is_swizzled and not lhs_no_spill_reload
                else self.BLOCK_K_PHYSICAL_LHS
            ),
            BLOCK_K_PHYSICAL_RHS=(
                self.BLOCK_K_PHYSICAL_RHS // INTERLEAVE_FACTOR
                if not rhs_is_swizzled and not rhs_no_spill_reload
                else self.BLOCK_K_PHYSICAL_RHS
            ),
            BLOCK_N_PHYSICAL=self.BLOCK_N_PHYSICAL if rhs_no_spill_reload else self.BLOCK_N_LOGICAL,
        )


@dataclass
class TileLocation(nl.NKIObject):
    """
    Represents the location of a tile within a tensor.

    Either specify the location with tile size + offset
    or with an access pattern and vector offset.

    Args:
        tensor (TensorDescriptor): The tensor containing this tile
        tile_k: Size of tile in K dimension
        tile_f: Size of tile in F dimension
        k_offset: Offset in K dimension
        f_offset: Offset in F dimension
        access_pattern (Optional[List]): Access pattern for the tile (used in DGT)
        vector_offset (Optional[nl.ndarray]): Vector offset for memory access (used in DGT)
    """

    tensor: TensorDescriptor
    tile_k: int
    tile_f: int
    k_offset: int = 0
    f_offset: int = 0
    access_pattern: Optional[List] = None
    vector_offset: Optional[nl.ndarray] = None

    def __post_init__(self):
        """Auto-generate vector_offset and access_pattern for unswizzled F-by-K tensors.

        Skipped when PE swizzle is selected, since PE transpose uses either
        user-provided row indices (indirect) or contiguous DMA (direct), and
        does not need DGT vector offsets.

        Skipped when fast_dma_transpose=True, since the fast path uses a direct
        access pattern on the source tensor without vector offsets.
        """
        if self.tensor.uses_pe_swizzle:
            return

        if self.tensor.fast_dma_transpose:
            return

        if not self.tensor.is_swizzled and self.tensor.is_f_by_k:
            self.set_vector_offset()
            self.generate_ap()

    def set_vector_offset(self):
        """
        Compute tile-specific vector offset by adding base offset to the pattern.

        Applies the tile's position offset to the pre-computed vector offset pattern
        from the parent TensorDescriptor. The base offset accounts for the tile's
        k_offset and f_offset in the flattened tensor layout.

        Args:
            None (uses self.tensor, self.tile_k, self.tile_f, self.k_offset, self.f_offset)

        Notes:
            Requires tensor.vector_offset_pattern_512 to be set via set_vector_offset_pattern.

            Base offset calculation:
                base_offset = k_offset // P_MAX + f_offset * SKIP_K

            Where SKIP_K = K // P_MAX is the stride between F rows in flattened layout.
        """
        if self.tensor.vector_offset_pattern_512 == None:
            self.tensor.set_vector_offset_patterns(self.tile_k, self.tile_f)

        sbm = get_active_sbm()
        pattern = self.tensor.get_vector_offset_pattern(self.tile_k)
        F, K = self.tensor.data.shape

        vector_size = self.tile_k // INTERLEAVE_FACTOR
        SKIP_K = K // vector_size
        base_offset = self.k_offset // vector_size + self.f_offset * SKIP_K
        vector_offsets_k = sbm.alloc_stack(pattern.shape, dtype=nl.uint32, buffer=nl.sbuf)

        nisa.tensor_scalar(dst=vector_offsets_k, data=pattern, op0=nl.add, operand0=base_offset)

        # Runtime expert offset (in vector_size units) for stacked-experts views.
        # The caller computes scalar_offset assuming vector_size = L_TILE_K // INTERLEAVE_FACTOR
        # (the full-tile vector_size). For remainder tiles (tile_k < L_TILE_K),
        # vector_size is smaller, so each index unit covers fewer elements.
        # We must scale the scalar_offset by (L_TILE_K / tile_k) to convert from
        # full-tile vector_size units to the current tile's vector_size units.
        if self.tensor.scalar_offset is not None:
            scalar_offset_to_add = self.tensor.scalar_offset
            if self.tile_k < L_TILE_K:
                scale_factor = L_TILE_K // self.tile_k
                scaled_offset = sbm.alloc_stack(
                    self.tensor.scalar_offset.shape,
                    dtype=nl.float32,
                    buffer=nl.sbuf,
                )
                nisa.tensor_scalar(
                    dst=scaled_offset,
                    data=self.tensor.scalar_offset,
                    op0=nl.multiply,
                    operand0=scale_factor,
                )
                scalar_offset_to_add = scaled_offset

            vector_offsets_with_expert = sbm.alloc_stack(
                pattern.shape,
                dtype=nl.uint32,
                buffer=nl.sbuf,
            )
            nisa.tensor_scalar(
                dst=vector_offsets_with_expert,
                data=vector_offsets_k,
                op0=nl.add,
                operand0=scalar_offset_to_add,
            )
            vector_offsets_k = vector_offsets_with_expert

        self.vector_offset = vector_offsets_k

    def generate_ap(self):
        """
        Generate DGT access pattern for the tile's flattened source layout.

        Computes the access pattern needed by nisa.dma_transpose for gather-transpose
        loading. The pattern describes how to read from the flattened [F*K/P_MAX, P_MAX]
        source tensor.

        Args:
            None (uses self.tensor, self.tile_k, self.tile_f, self.f_offset)

        Notes:
            Access pattern format: [[P_MAX, flattened_rows], [1, 1], [1, 1], [1, P_MAX]]

            Where flattened_rows = min(tile_f, F - f_offset) * tile_k // P_MAX
            When effective_f_dim is set (e.g. stacked-expert tensors), F_effective is the
            per-expert F range. Otherwise F_effective is the full tensor F dimension.
        """
        if self.access_pattern != None:
            return

        F, _ = self.tensor.data.shape
        F_effective = self.tensor.effective_f_dim if self.tensor.effective_f_dim is not None else F
        vector_size = self.tile_k // INTERLEAVE_FACTOR
        flattened_rows = min(self.tile_f, F_effective - self.f_offset) * self.tile_k // vector_size
        access_pattern = [[vector_size, flattened_rows], [1, 1], [1, 1], [1, vector_size]]

        self.access_pattern = access_pattern
