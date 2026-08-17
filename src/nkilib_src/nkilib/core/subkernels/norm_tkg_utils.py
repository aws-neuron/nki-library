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

"""Utility functions for normalization kernels in token generation mode."""

from dataclasses import dataclass
from typing import Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.interleave_copy import interleave_copy
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..utils.tiled_range import TiledRange

# DMA engine mode
_DGE_MODE_NONE = 3

# PSUM bank count for cycling allocations
_PSUM_BANK_COUNT = 8

# Threshold for using contiguous load + on-chip transpose
_CONTIGUOUS_LOAD_H_THRESHOLD = 2048

# Alignment constants for nc_transpose
_PSUM_ALIGNMENT_BYTES = 4

# MX config
_SUPPORTED_QMX_OUTPUT_DTYPES = [nl.float8_e4m3fn_x4, nl.float8_e5m2_x4]
_QMX_UNPACKED_OUTPUT_DTYPES = [nl.float8_e4m3fn, nl.float8_e5m2]
_MX_SCALE_DTYPE = nl.uint8
_MX_UNPACKED_PACKED_MAP = {
    nl.float8_e4m3fn: nl.float8_e4m3fn_x4,
    nl.float8_e5m2: nl.float8_e5m2_x4,
}

# Transpose config
_UINT8_TP_VIEW_DTYPE = nl.float8_e5m2
_1B_XPOSE_PSUM_STEP = 2

# rmsnorm_mx_quantize_tkg BxS sharding threshold
_RMSNORM_QMX_SHARDING_THRESHOLD = 4


def pe_transpose(
    src,
    dst,
    tile_size: int,
    dtype,
    sbm: SbufManager,
    psum_bank_base: int = 0,
):
    """
    Transpose 3D [P, tile_size, num_tiles] → 3D [tile_size, P, num_tiles] through PSUM.
    Ex: [T, H0, H2] -> [H0, T, H2]

    For each tile index i, extracts src[:, :, i] of shape [P, tile_size],
    transposes it to [tile_size, P] via nc_transpose, and writes to dst[:, :, i].

    Packs multiple tiles per PSUM bank when P is small, cycling across
    banks for scalar/vector engine interleaving.


    Args:
        src: 3D SBUF tensor or NkiTensor [P, tile_size, num_tiles].
        dst: 3D SBUF NkiTensor [tile_size, P, num_tiles].
        tile_size: partition dimension size of each transposed tile.
        dtype: tensor data type.
        sbm: SbufManager for PSUM address control.
        psum_bank_base: starting PSUM bank index offset, cycled modulo _PSUM_BANK_COUNT.
    """
    _psum_fmax = nl.tile_size.psum_fmax

    P = src.shape[0]
    num_tiles = src.shape[2]

    dtype_size = sizeinbytes(dtype)
    # PSUM alignment: compute padded size for 4-byte alignment
    padded_tile_size = div_ceil(P * dtype_size, _PSUM_ALIGNMENT_BYTES) * _PSUM_ALIGNMENT_BYTES // dtype_size
    tiles_per_psum = _psum_fmax // padded_tile_size

    # dst is [tile_size, P, num_tiles] → permute to [tile_size, num_tiles, P] for contiguous tile writes
    dst_permuted = dst.permute(dims=[0, 2, 1])

    for psum_tile in TiledRange(num_tiles, tiles_per_psum):
        psum_bank_idx = (psum_tile.index + psum_bank_base) % _PSUM_BANK_COUNT
        cur_tiles_per_psum = psum_tile.size

        tp_psum = nl.ndarray(
            (tile_size, cur_tiles_per_psum * padded_tile_size),
            dtype=dtype,
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, psum_bank_idx * _psum_fmax * 4),
        )

        for i in range(cur_tiles_per_psum):
            tile_idx = psum_tile.start_offset + i
            col_offset = i * padded_tile_size
            tile_src = src.slice(dim=2, start=tile_idx, end=tile_idx + 1).squeeze_dim(dim=2)  # [P, tile_size]
            nisa.nc_transpose(
                dst=tp_psum[0:tile_size, col_offset : col_offset + P],
                data=tile_src,
            )

        tp_psum_view = tp_psum.reshape_dim(dim=1, shape=[cur_tiles_per_psum, padded_tile_size]).slice(
            dim=2, start=0, end=P
        )
        interleave_copy(
            index=psum_bank_idx,
            dst=dst_permuted.slice(dim=1, start=psum_tile.start_offset, end=psum_tile.end_offset),
            src=tp_psum_view,
        )


def validate_shapes(
    input_view: nl.NkiTensor,
    gamma_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
) -> Tuple[int, int, int, int]:
    """
    Validate tensor shapes for normalization operations.

    Args:
        input_view (NkiTensor): Input tensor view
        gamma_view (NkiTensor): Gamma tensor view
        output_view (NkiTensor): Output tensor view

    Returns:
        Tuple[int, int, int, int]: (BxS, H, H0, H1) dimensions

    Notes:
        - H0 must equal nl.tile_size.pmax (128)
        - H must be divisible by H0
        - Output shape must be [H0, BxS, H1]
        - Gamma shape must be [1, H]
    """
    H0 = nl.tile_size.pmax
    if input_view.buffer == nl.sbuf:
        _H0, BxS, H1 = input_view.shape
        kernel_assert(
            _H0 == H0,
            f"Input tensor in SBUF does not have partition dimension H0 of {H0}, got {_H0}",
        )
        H = _H0 * H1
    else:
        B, S, H = input_view.shape
        BxS = B * S
        kernel_assert(H % H0 == 0, f"Input tensor H dimension must be divisible by {H0}, got {H}")
        H1 = H // H0

    kernel_assert(
        tuple(output_view.shape) == (H0, BxS, H1),
        f"Output shape expected is (H0, BxS, H1): {(H0, BxS, H1)}, got {tuple(output_view.shape)}",
    )

    kernel_assert(
        gamma_view.shape == (1, H),
        f"Malformed shape of gamma expected [1, {H}], got {gamma_view.shape}",
    )
    return BxS, H, H0, H1


def validate_shapes_shard_on_h(
    input_view: nl.NkiTensor,
    gamma_view: nl.NkiTensor,
    output_view: nl.NkiTensor,
) -> Tuple[int, int, int, int, int, int]:
    """
    Validate tensor shapes for normalization with H-dimension sharding.

    Args:
        input_view (NkiTensor): Input tensor view
        gamma_view (NkiTensor): Gamma tensor view
        output_view (NkiTensor): Output tensor view

    Returns:
        Tuple[int, int, int, int, int, int]: (BxS, H, H0, H1, sharded_H, sharded_H1) dimensions

    Notes:
        - Requires LNC=2 for H-dimension sharding
        - H must be divisible by H0 * lnc
        - If input is in SBUF, it is expected to be pre-sharded
    """
    H0 = nl.tile_size.pmax
    _, lnc, shard_id = get_verified_program_sharding_info("norm_tkg", (0, 1))

    # if input and output in sbuf, expected to be pre-sharded.
    if input_view.buffer == nl.sbuf:
        _H0, BxS, sharded_H1 = input_view.shape
        kernel_assert(
            _H0 == H0,
            f"Input tensor in SBUF does not have partition dimension H0 of {H0}, got {_H0}",
        )
        H = sharded_H1 * _H0 * lnc
    else:
        B, S, H = input_view.shape
        BxS = B * S
        kernel_assert(
            H % (H0 * lnc) == 0,
            f"Input tensor H dimension must be divisible by {H0} * {lnc}, got {H}",
        )
        sharded_H1 = H // H0 // lnc
    sharded_H = sharded_H1 * H0
    H1 = sharded_H1 * lnc

    if output_view.buffer == nl.sbuf:
        kernel_assert(
            tuple(output_view.shape) == (H0, BxS, sharded_H1),
            f"Output shape expected is {(H0, BxS, sharded_H1)}, got {tuple(output_view.shape)}",
        )
    else:
        kernel_assert(
            tuple(output_view.shape) == (H0, BxS, H1),
            f"Output shape expected is {(H0, BxS, H1)}, got {tuple(output_view.shape)}",
        )

    kernel_assert(
        gamma_view.shape == (1, H),
        f"Malformed shape of gamma expected [1, {H}], got {gamma_view.shape}",
    )
    return BxS, H, H0, H1, sharded_H, sharded_H1


def contiguous_load_transpose(
    input_hbm: nl.NkiTensor,
    input_sb: nl.NkiTensor,
    num_H_shards: int,
    sbm: SbufManager,
) -> None:
    """
    Load input using contiguous DMA + on-chip nc_transpose.

    More efficient than dma_copy for small H dimensions. Loads data contiguously
    to SBUF, then uses nc_transpose to rearrange into the target layout.

    Supports unbalanced sharding: when H1 is not evenly divisible by num_H_shards,
    earlier shards get floor(H1/num_H_shards) tiles and the last shard gets the remainder.

    Args:
        input_hbm (NkiTensor): [BxS, H], Input tensor view in HBM
        input_sb (NkiTensor): [H0, BxS, H1], Output buffer in SBUF
        num_H_shards (int): Number of shards along H dimension
        sbm (SbufManager): SBUF memory manager

    Returns:
        None: Data is written directly into input_sb

    Notes:
        Data Layout:
            HBM input:  [BxS, H] where H = sum of per-shard H0 * H2_shard
                        Memory order: for each bxs, data is [shard0{H0*H2_0}, shard1{H0*H2_1}, ...]

            SBUF output: [H0, BxS, H1] where H1 = sum of H2_shard
                         Logical view: [H0, BxS, shard0_H2 | shard1_H2 | ...]
    """
    H0 = nl.tile_size.pmax

    BxS, H = input_hbm.shape
    H1 = H // H0
    H2_base = H1 // num_H_shards

    output_H1 = input_sb.shape[2]
    output_H2_base = output_H1 // num_H_shards

    for bxs_tile in TiledRange(BxS, H0):
        # Load [bxs_tile.size, H] from HBM to SBUF
        input_sbuf_temp = sbm.alloc_heap(
            (bxs_tile.size, H),
            dtype=input_hbm.dtype,
            buffer=nl.sbuf,
            name=f"{sbm.get_name_prefix()}cont_load_transpose_buff_{bxs_tile.index}",
        )
        input_hbm_tile = input_hbm.slice(
            dim=0, start=bxs_tile.start_offset, end=bxs_tile.end_offset
        )  # [bxs_tile.size, H]
        nisa.dma_copy(src=input_hbm_tile, dst=input_sbuf_temp, dge_mode=_DGE_MODE_NONE)

        input_temp_view = input_sbuf_temp

        src_h_offset = 0
        dst_h_offset = 0
        for shard_idx in range(num_H_shards):
            # Unbalanced sharding: last shard gets the remainder
            shard_H2 = H2_base if shard_idx < num_H_shards - 1 else H1 - H2_base * (num_H_shards - 1)
            shard_output_H2 = (
                output_H2_base if shard_idx < num_H_shards - 1 else output_H1 - output_H2_base * (num_H_shards - 1)
            )

            # src: slice [bxs_tile.size, H0 * shard_H2] from flat H, reshape to [bxs_tile.size, H0, shard_H2]
            src_view = input_temp_view.slice(dim=1, start=src_h_offset, end=src_h_offset + H0 * shard_H2).reshape_dim(
                dim=1, shape=[H0, shard_H2]
            )

            # dst: [H0, bxs_tile.size, shard_output_H2]
            dst_view = input_sb.slice(dim=1, start=bxs_tile.start_offset, end=bxs_tile.end_offset).slice(
                dim=2, start=dst_h_offset, end=dst_h_offset + shard_output_H2
            )

            pe_transpose(
                src=src_view,
                dst=dst_view,
                tile_size=H0,
                dtype=input_hbm.dtype,
                sbm=sbm,
            )

            src_h_offset += H0 * shard_H2
            dst_h_offset += shard_output_H2

        sbm.pop_heap()


def load_input_to_sbuf(
    input_hbm: nl.NkiTensor,
    input_sb: nl.NkiTensor,
    num_H_shards: int,
    hidden_dim_tp: bool = False,
    shard_on_h: bool = False,
    sbm: Optional[SbufManager] = None,
) -> nl.NkiTensor:
    """
    Load input data from HBM to SBUF with appropriate layout transformation.

    Args:
        input_hbm (NkiTensor): [BxS, H], Input tensor view in HBM
        input_sb (NkiTensor): [H0, BxS, H1], Input buffer in SBUF
        num_H_shards (int): Number of shards along H dimension
        hidden_dim_tp (bool): If True, use transpose load for (H/128, 128) layout
        sbm (Optional[SbufManager]): SBUF manager, required for contiguous load path

    Returns:
        NkiTensor: Input tensor view in SBUF with shape [H0, BxS, H1]

    Notes:
        - hidden_dim_tp=True: Transpose load (BxS, H) -> (BxS*H1, H0) -> (H0, BxS, H1)
        - hidden_dim_tp=False: Standard layout (BxS, H) -> (BxS, num_H_shards, H0, H2) -> (H0, BxS, num_H_shards, H2)
        - Contiguous load: Contiguous DMA + on-chip nc_transpose (more efficient for small H)
    """
    H0 = nl.tile_size.pmax
    BxS, H = input_hbm.shape
    H1 = H // H0
    H2 = H1 // num_H_shards

    use_contiguous_load = H <= _CONTIGUOUS_LOAD_H_THRESHOLD

    if hidden_dim_tp:
        # (BxS, H) -> (BxS*H1, H0) -> (H0, BxS, H1)
        input_hbm_view = (
            input_hbm.reshape_dim(dim=1, shape=[H1, H0])
            .flatten_dims(start_dim=0, end_dim=1)
            .expand_dim(dim=1)
            .expand_dim(dim=1)
        )
        input_sb_view = input_sb.flatten_dims(start_dim=1, end_dim=2).expand_dim(dim=1).expand_dim(dim=1)
        nisa.dma_transpose(dst=input_sb_view, src=input_hbm_view)
    elif shard_on_h:
        if use_contiguous_load:
            kernel_assert(sbm != None, "sbm required for contiguous load path")
            contiguous_load_transpose(input_hbm, input_sb, 1, sbm)
        else:
            # (BxS, sharded_H) -> (BxS, H0, H1) -> (H0, BxS, H1)
            input_hbm_view = input_hbm.reshape_dim(dim=1, shape=[H0, H1]).permute(dims=[1, 0, 2])
            nisa.dma_copy(
                dst=input_sb,
                src=input_hbm_view,
                dge_mode=_DGE_MODE_NONE,
            )
    else:
        # (BxS, H) -> (BxS, num_H_shards, H0, H2) -> (H0, BxS, num_H_shards, H2)
        if use_contiguous_load:
            kernel_assert(sbm != None, "sbm required for contiguous load path")
            contiguous_load_transpose(input_hbm, input_sb, num_H_shards, sbm)
        else:
            if H1 % num_H_shards == 0:
                # Balanced sharding: single vectorized reshape
                input_hbm_view = input_hbm.reshape_dim(dim=1, shape=[num_H_shards, H0, H2]).permute(dims=[2, 0, 1, 3])
                input_sb_view = input_sb.reshape_dim(dim=2, shape=[num_H_shards, H2])  # (H0, BxS, num_H_shards, H2)
                nisa.dma_copy(
                    dst=input_sb_view,
                    src=input_hbm_view,
                    dge_mode=_DGE_MODE_NONE,
                )
            else:
                # Unbalanced sharding: load each shard separately
                src_h_offset = 0
                dst_h_offset = 0
                for shard_idx in range(num_H_shards):
                    shard_H2 = H2 if shard_idx < num_H_shards - 1 else H1 - H2 * (num_H_shards - 1)
                    # [BxS, shard_H2 * H0] -> [BxS, H0, shard_H2] -> [H0, BxS, shard_H2]
                    src_view = (
                        input_hbm.slice(dim=1, start=src_h_offset, end=src_h_offset + H0 * shard_H2)
                        .reshape_dim(dim=1, shape=[H0, shard_H2])
                        .permute(dims=[1, 0, 2])
                    )
                    dst_view = input_sb.slice(dim=2, start=dst_h_offset, end=dst_h_offset + shard_H2)
                    nisa.dma_copy(
                        dst=dst_view,
                        src=src_view,
                        dge_mode=_DGE_MODE_NONE,
                    )
                    src_h_offset += H0 * shard_H2
                    dst_h_offset += shard_H2
    return input_sb


def load_gamma_to_sbuf(
    gamma_hbm: nl.NkiTensor,
    gamma_sb: nl.NkiTensor,
    num_H_shards: int,
    hidden_dim_tp: bool = False,
    shard_on_h: bool = False,
) -> nl.NkiTensor:
    """
    Load gamma weights from HBM to SBUF with appropriate layout transformation.

    Args:
        gamma_hbm (NkiTensor): [1, H], Gamma tensor view in HBM
        gamma_sb (NkiTensor): [H0, H1], Gamma buffer in SBUF
        num_H_shards (int): Number of shards along H dimension
        hidden_dim_tp (bool): If True, use transpose load for (H/128, 128) layout
        shard_on_h (bool): If True, load gamma for H-dimension sharding layout

    Returns:
        NkiTensor: Gamma tensor view in SBUF with shape [H0, H1]

    Notes:
        - hidden_dim_tp=True: Transpose load (H) -> (H1, H0) -> (H0, H1)
        - hidden_dim_tp=False: Standard layout (H) -> (num_H_shards, H0, H2) -> (H0, num_H_shards, H2)
    """
    H0 = nl.tile_size.pmax
    H = gamma_hbm.shape[1]
    H1 = H // H0
    H2 = H1 // num_H_shards

    if hidden_dim_tp:
        # (1, H) -> (H)
        gamma_hbm = gamma_hbm.flatten_dims(start_dim=0, end_dim=1)
        # Transpose load: (H) -> (H1, H0) -> (H0, H1)
        gamma_hbm_view = gamma_hbm.reshape_dim(dim=0, shape=[H1, H0]).expand_dim(dim=1).expand_dim(dim=1)
        gamma_sb_dst_view = gamma_sb.expand_dim(dim=1).expand_dim(dim=1)
        nisa.dma_transpose(dst=gamma_sb_dst_view, src=gamma_hbm_view)
    elif shard_on_h:
        # (shared_H) -> (H0, shared_H1)
        gamma_hbm_view = gamma_hbm.reshape_dim(dim=1, shape=[H0, H1]).select(dim=0, index=0)
        nisa.dma_copy(
            dst=gamma_sb,
            src=gamma_hbm_view,
            dge_mode=_DGE_MODE_NONE,
        )
    else:
        # (1, H) -> (H)
        gamma_hbm = gamma_hbm.flatten_dims(start_dim=0, end_dim=1)
        # Standard layout: (H) -> (num_H_shards, H0, H2) -> (H0, num_H_shards, H2)
        if H1 % num_H_shards == 0:
            gamma_hbm_view = gamma_hbm.reshape_dim(dim=0, shape=[num_H_shards, H0, H2]).permute(dims=[1, 0, 2])
            gamma_sb_view_reshaped = gamma_sb.reshape_dim(dim=1, shape=[num_H_shards, H2])
            nisa.dma_copy(
                dst=gamma_sb_view_reshaped,
                src=gamma_hbm_view,
                dge_mode=_DGE_MODE_NONE,
            )
        else:
            # Unbalanced sharding: load each shard separately
            src_h_offset = 0
            dst_h_offset = 0
            for shard_idx in range(num_H_shards):
                shard_H2 = H2 if shard_idx < num_H_shards - 1 else H1 - H2 * (num_H_shards - 1)
                # [shard_H2 * H0] -> [H0, shard_H2] -> [H0, shard_H2]
                src_view = gamma_hbm.slice(dim=0, start=src_h_offset, end=src_h_offset + H0 * shard_H2).reshape_dim(
                    dim=0, shape=[H0, shard_H2]
                )
                dst_view = gamma_sb.slice(dim=1, start=dst_h_offset, end=dst_h_offset + shard_H2)
                nisa.dma_copy(
                    dst=dst_view,
                    src=src_view,
                    dge_mode=_DGE_MODE_NONE,
                )
                src_h_offset += H0 * shard_H2
                dst_h_offset += shard_H2
    return gamma_sb


def get_token_tile_size(num_tokens: int) -> int:
    """
    Determine tile size for processing tokens in RMSNorm quantize MX TKG kernel.

    Finds the largest power-of-2 tile size (8-64) that evenly divides num_tokens.
    Used to tile the token (batch * sequence) dimension for efficient processing.

    Args:
        num_tokens (int): Number of tokens to process (typically BxS // num_shards)

    Returns:
        int: Tile size for iterating over tokens

    Notes:
        - Falls back to num_tokens if no power-of-2 tile size in [8, 64] divides evenly
    """
    MIN_TILE_SIZE = 8
    MAX_TILE_SIZE = 64

    tile_size = MAX_TILE_SIZE
    while tile_size >= MIN_TILE_SIZE:
        if num_tokens % tile_size == 0:
            return tile_size
        tile_size //= 2

    return num_tokens


@dataclass
class RmsNormMXQuantizeDims(nl.NKIObject):
    """Dimension and hardware parameters derived from input shapes."""

    input_shape: tuple
    hidden_actual: int = None

    # Computed fields
    B: int = None
    S: int = None
    H: int = None
    H0: int = None
    H1: int = None
    BxS: int = None
    num_H512_tiles: int = None
    pmax: int = None
    psum_fmax: int = None
    inter_dtype: object = None

    def __post_init__(self):
        self.pmax = nl.tile_size.pmax
        self.H0 = self.pmax
        self.psum_fmax = nl.tile_size.gemm_moving_fmax
        self.inter_dtype = nl.float32

        # NOTE: NKI does not support unpacking complex var into multiple class var assignments
        B, S, H = self.input_shape
        self.B = B
        self.S = S
        self.H = H
        self.H1 = self.H // self.H0
        self.BxS = self.B * self.S
        self.num_H512_tiles = self.H // self.psum_fmax
        if self.hidden_actual == None:
            self.hidden_actual = self.H


@dataclass
class RmsNormMXQuantizeConfig(nl.NKIObject):
    """Hyperparameter configuration for RMSNorm + MX quantization TKG kernel."""

    eps: float
    is_residual_add: bool
    is_static_mx: bool
    is_output_quant_in_sbuf: bool
    is_output_quant_packed: bool
    qmx_output_dtype: object
    output_quant_dtype: object
    do_shard: bool
    shard_id: int
    shard_size: int
    BxS_offset: int
    BxS_tile_size: int
    num_BxS_tiles: int


def validate_rmsnorm_mx_quantize_tkg(
    input_shape,
    gamma_shape,
    output_shape,
    output_quant_dtype,
    output_quant_shape,
    output_quant_buffer,
    output_scale_shape,
    output_scale_buffer,
    output_dtype,
    eps=1e-6,
    hidden_actual=None,
    hidden_dim_tp=True,
    has_residual=False,
    residual_shape=None,
    has_output_residual=False,
    output_residual_shape=None,
    is_static_mx=False,
    has_output_input_dequant_scale=False,
):
    """
    Validate inputs and build RmsNormMXQuantizeDims and RmsNormMXQuantizeConfig.

    Returns:
        Tuple[RmsNormMXQuantizeDims, RmsNormMXQuantizeConfig]: (dims, cfg)
    """
    dims = RmsNormMXQuantizeDims(input_shape=input_shape, hidden_actual=hidden_actual)
    H0, H1, BxS, H = dims.H0, dims.H1, dims.BxS, dims.H

    # HW validation
    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen4,
        f"rmsnorm_mx_quantize_tkg only supports gen4+ (Trn3+) but got {nisa.get_nc_version()=}",
    )

    # Shape validation
    kernel_assert(H % dims.psum_fmax == 0, f"Expected H divisible by {dims.psum_fmax}, got H={H}")
    kernel_assert(H % H0 == 0, f"H must be divisible by {H0}")
    kernel_assert(gamma_shape == (1, H), f"Malformed shape of gamma {gamma_shape}")
    kernel_assert(hidden_dim_tp, "Only hidden_dim_tp=True is supported")
    kernel_assert(output_shape == (H0, BxS, H1), f"Expected output.shape = {(H0, BxS, H1)}")
    kernel_assert(output_dtype in [nl.float16, nl.bfloat16], "output.dtype must be float16 or bfloat16")

    # Output quantization config (skip when output_quant=None for bf16-only mode)
    is_output_quant_in_sbuf = output_quant_buffer == nl.sbuf if output_quant_buffer != None else False
    is_output_quant_packed = output_quant_shape == (BxS, H + H // 4) if output_quant_shape != None else False

    qmx_output_dtype = None
    if output_quant_dtype == None:
        pass
    elif output_scale_shape != None:
        kernel_assert(
            output_quant_shape == output_scale_shape,
            f"Expected same shape for output_quant and output_scale, but got {output_quant_shape=}, {output_scale_shape=}",
        )
        kernel_assert(
            output_quant_buffer == output_scale_buffer,
            f"Expected output_quant and output_scale both in HBM or SBUF, but got {output_quant_buffer=}, {output_scale_buffer=}",
        )
        kernel_assert(
            output_quant_dtype in _SUPPORTED_QMX_OUTPUT_DTYPES,
            f"Expected output_quant.dtype in {_SUPPORTED_QMX_OUTPUT_DTYPES}, but got {output_quant_dtype=}",
        )
        qmx_output_dtype = output_quant_dtype
    else:
        kernel_assert(
            is_output_quant_packed,
            f"Expected packed output_quant with shape (B*S, H+H/4), but got {output_quant_shape=}",
        )
        kernel_assert(
            output_quant_dtype in _QMX_UNPACKED_OUTPUT_DTYPES,
            f"Expected output_quant.dtype in {_QMX_UNPACKED_OUTPUT_DTYPES}, but got {output_quant_dtype=}",
        )
        qmx_output_dtype = _MX_UNPACKED_PACKED_MAP[output_quant_dtype]

    # Residual validation
    is_residual_add = has_residual
    if is_residual_add:
        kernel_assert(input_shape == residual_shape, "input and residual shapes must match")
        kernel_assert(has_output_residual, "output_residual required when residual provided")
        kernel_assert(output_residual_shape == (BxS, H), f"expected output_residual shape (B*S, H)={(BxS, H)}")
        kernel_assert(H1 % 8 == 0, f"Expected H1 divisible by 8 with fused residual add")
        kernel_assert(BxS >= 256, f"Residual add requires BxS >= 256 (got {BxS})")
    else:
        kernel_assert(H1 % 4 == 0, f"Expected H1 divisible by 4")

    # LNC sharding strategy
    _, lnc, shard_id = get_verified_program_sharding_info("rmsnorm_mx_quantize_tkg", (0, 1))
    kernel_assert(lnc == 2, "rmsnorm_mx_quantize_tkg kernel only supports LNC=2")
    do_shard = BxS % lnc == 0 and BxS >= _RMSNORM_QMX_SHARDING_THRESHOLD
    shard_size = BxS // lnc if do_shard else BxS
    BxS_offset = shard_id * shard_size if do_shard else 0

    # Tiling strategy
    BxS_tile_size = get_token_tile_size(shard_size)
    num_BxS_tiles = shard_size // BxS_tile_size
    kernel_assert(shard_size % BxS_tile_size == 0, "shard_size must be divisible by BxS_tile_size")

    # STATIC_MX validation
    if is_static_mx and output_quant_shape != None:
        kernel_assert(has_output_input_dequant_scale, "output_input_dequant_scale required for STATIC_MX")
        kernel_assert(is_output_quant_in_sbuf, "STATIC_MX mode requires SBUF output")

    # Build cfg
    cfg = RmsNormMXQuantizeConfig(
        eps=eps,
        is_residual_add=is_residual_add,
        is_static_mx=is_static_mx,
        is_output_quant_in_sbuf=is_output_quant_in_sbuf,
        is_output_quant_packed=is_output_quant_packed,
        qmx_output_dtype=qmx_output_dtype,
        output_quant_dtype=output_quant_dtype,
        do_shard=do_shard,
        shard_id=shard_id,
        shard_size=shard_size,
        BxS_offset=BxS_offset,
        BxS_tile_size=BxS_tile_size,
        num_BxS_tiles=num_BxS_tiles,
    )

    return dims, cfg
