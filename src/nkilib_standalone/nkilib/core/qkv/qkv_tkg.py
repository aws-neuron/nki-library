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

"""
QKV Projection TKG Kernel

This kernel implements the fused QKV (Query-Key-Value) projection operation with
optional residual addition and normalization (RMSNorm/LayerNorm), commonly used
before the attention block in transformer models. The kernel is specifically optimized
for Token Generation (TKG, also known as Decoding) scenarios where batch_size * seqlen
is small.

This kernel is designed with LNC support. When LNC>1, the H dimension is sharded across cores.
Multiple output layouts (BSD, NBSd) are supported to match downstream kernel requirements.

"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import (
    SbufManager,
    create_auto_alloc_manager,
    sizeinbytes,
)
from ..utils.common_types import QKVOutputLayout, NormType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info
from ..utils.logging import Logger, LogLevel

from ..subkernels.rmsnorm_tkg import rmsnorm_tkg as _rmsnorm_tkg
from ..subkernels.rmsnorm_tkg import SHARDING_THRESHOLD as rmsnorm_sharding_threshold
from ..subkernels.layernorm_tkg import layernorm_tkg as _layernorm_tkg
from ..subkernels.layernorm_tkg import (
    SHARDING_THRESHOLD as layernorm_sharding_threshold,
)

# I_TILE_SIZE is chosen to match the maximum free dimension of a matmul instruction
I_TILE_SIZE = 512
# I_SHARD_SIZE is chosen to be I_TILE_SIZE * 8 for 8 available PSUM banks
I_SHARD_SIZE = 4096
# Tile size for H dimension weight loading
H_TILE_SIZE = 2048


def qkv_tkg(
    hidden: nl.ndarray,
    qkv_w: nl.ndarray,
    norm_w: Optional[nl.ndarray] = None,
    fused_add: bool = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attn_prev: Optional[nl.ndarray] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.RMS_NORM,
    output_in_sbuf: bool = False,
    qkv_bias: Optional[nl.ndarray] = None,
    norm_bias: Optional[nl.ndarray] = None,
    hidden_actual: Optional[int] = None,
    sbm: Optional[SbufManager] = None,
) -> nl.ndarray | Tuple[nl.ndarray, nl.ndarray]:
    """
    QKV Projection Kernel for Token Generation

    This kernel computes the fused QKV projection operation:
        hidden' = norm(hidden + attn_prev + mlp_prev)  # optional fused add and norm
        output = hidden' @ qkv_w + qkv_bias
    typically used before the attention block in transformer models.

    This kernel is optimized for Token Generation (aka Decoding) use cases where
    batch_size * seqlen is small. Using this kernel with B*S > 128 may result in
    degraded performance - use the CTE variant for large sequence lengths.

    The kernel supports optional fused residual addition and normalization (RMSNorm/LayerNorm)
    to reduce HBM traffic and improve performance.

    Data Types:
        This kernel supports nl.float32, nl.float16, and nl.bfloat16 data types.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size
        I: Fused QKV output dimension ((Q + K + V) * N * D)
        N: Number of heads (for NBSd output layout)
        D: Head dimension size (for NBSd output layout)

    Args:
        hidden (nl.ndarray):
            Input hidden states tensor in HBM.
            Shape:    [B, S, H]
        qkv_w (nl.ndarray):
            QKV projection weight tensor in HBM.
            Shape:    [H, I]
        norm_w (nl.ndarray, optional):
            Normalization weight tensor in HBM. Required when norm_type is RMS_NORM or LAYER_NORM.
            Shape:    [1, H]
        fused_add (bool):
            Enable fused residual addition (hidden + attn_prev + mlp_prev). Default: False.
        mlp_prev (nl.ndarray, optional):
            Previous MLP residual tensor in HBM. Required when fused_add is True.
            Shape:    [B, S, H]
        attn_prev (nl.ndarray, optional):
            Previous attention residual tensor in HBM. Required when fused_add is True.
            Shape:    [B, S, H]
        d_head (int, optional):
            Head dimension size D. Required for NBSd output layout.
        output_layout (QKVOutputLayout):
            Output tensor layout format. BSD: [B, S, I] or NBSd: [N, B, S, D]. Default: QKVOutputLayout.BSD.
        eps (float):
            Epsilon value to maintain numerical stability in normalization. Default: 1e-6.
        norm_type (NormType):
            Type of normalization to apply (NO_NORM, RMS_NORM, or LAYER_NORM). Default: NormType.RMS_NORM.
        output_in_sbuf (bool):
            If True, output is kept in SBUF; otherwise stored to HBM. Default: False.
            Only supports single I-shard when True.
        qkv_bias (nl.ndarray, optional):
            Bias tensor in HBM for QKV projection.
            Shape:    [1, I]
        norm_bias (nl.ndarray, optional):
            LayerNorm beta parameter tensor in HBM. Required when norm_type is LAYER_NORM.
            Shape:    [1, H]
        hidden_actual (int, optional):
            Actual hidden dimension for padded input tensors. If specified, normalization
            uses this value instead of H for mean calculation.
        sbm (SbufManager, optional):
            Instance of SbufManager responsible for handling SBUF allocation.
            If None, auto-allocation manager is created.

    Returns:
        output (nl.ndarray | Tuple[nl.ndarray, nl.ndarray]):
            QKV projection output tensor. The tensor can reside in either SBUF or HBM.
            Shape:    [B, S, I] for BSD layout, [N, B, S, D] for NBSd layout.
            When fused_add is True, returns tuple (output, fused_hidden) where
            fused_hidden is the result of the fused residual addition.

    Restrictions:
        H must be divisible by 128 (nl.tile_size.pmax).
        H1 (H//128) must be divisible by number of shards for multi-core execution.
        output_in_sbuf only supports single I-shard (I < 4096).
    """

    if not sbm:
        sbm = create_auto_alloc_manager(logger=Logger("qkv-sbm"))

    kernel_assert(
        sbm.is_auto_alloc(),
        "QKV TKG only supports auto allocation but got non-auto allocation SBM",
    )

    sbm.open_scope(name="qkv_tkg")

    # Check program dimensionality
    _, num_shards, shard_id = get_verified_program_sharding_info("qkv_tkg", (0, 1))

    # Get input shapes
    B, S, H = hidden.shape
    _H, I = qkv_w.shape
    BxS = B * S
    H0 = nl.tile_size.pmax
    H1 = H // H0

    if hidden_actual == None:
        hidden_actual = H

    # Validate kernel inputs
    kernel_assert(
        H % nl.tile_size.pmax == 0,
        f"H must be divisible by {nl.tile_size.pmax}, got {H} % {nl.tile_size.pmax}={H % nl.tile_size.pmax}",
    )
    kernel_assert(
        _H == H,
        f"Weight tensor reduction dimension must match hidden dimension, got weight H={_H}, hidden H={H}",
    )
    if qkv_bias != None:
        kernel_assert(
            qkv_bias.shape == (1, I),
            f"Bias shape must be [1, I], got {qkv_bias.shape}, expected {(1, I)}",
        )
    kernel_assert(
        norm_type == NormType.LAYER_NORM or not norm_bias,
        f"norm_bias only supported for LAYER_NORM, got norm_type={norm_type} with norm_bias={norm_bias != None}",
    )
    kernel_assert(
        output_layout != QKVOutputLayout.NBdS,
        f"NBdS output layout is not supported in QKV TKG, got output_layout={output_layout}",
    )
    kernel_assert(
        H1 % num_shards == 0,
        f"H1 must be divisible by num_shards, got H={H}, H1={H1}, num_shards={num_shards}, H1 % num_shards={H1 % num_shards}",
    )
    kernel_assert(
        not fused_add or (attn_prev != None and mlp_prev != None),
        f"attn_prev and mlp_prev must be provided when fused_add=True, got fused_add={fused_add}, "
        f"attn_prev provided={attn_prev != None}, mlp_prev provided={mlp_prev != None}",
    )

    # Explicit Enum checks at top of function
    if norm_type != NormType.NO_NORM and norm_type != NormType.RMS_NORM and norm_type != NormType.LAYER_NORM:
        kernel_assert(False, f"NormType {norm_type} is not supported")
    if output_layout != QKVOutputLayout.BSD and output_layout != QKVOutputLayout.NBSd:
        kernel_assert(False, f"OutputLayout {output_layout} is not supported")

    # Calculate all constants
    dims, tiles = _calculate_constants(B, S, H, I, num_shards, shard_id)

    if output_in_sbuf:
        kernel_assert(
            tiles.num_I_shards == 1,
            f"output_in_sbuf=True requires I < {I_SHARD_SIZE}, got I={I}",
        )

    # Perform optional fused residual add in HBM
    # hidden_hbm: (B, S, H)
    if fused_add:
        hidden_hbm = _fused_residual_add_hbm2hbm(hidden, attn_prev, mlp_prev, dims, norm_type)
    else:
        hidden_hbm = hidden

    # Perform optional fused norm and load - dispatches based on norm_type
    # hidden_sb: (H0, BxS, H1_sharded) if NO_NORM, (H0, BxS, H1) if RMS/LAYER_NORM
    hidden_sb, hidden_base_offset = _fused_norm_and_load(
        hidden_hbm=hidden_hbm,
        norm_type=norm_type,
        norm_w=norm_w,
        norm_bias=norm_bias,
        eps=eps,
        hidden_actual=hidden_actual,
        dims=dims,
        sbm=sbm,
    )

    # Dispatch to appropriate projection path based output buffer
    if output_in_sbuf:
        output = _qkv_projection_sbuf_output(
            hidden_sb=hidden_sb,
            qkv_w=qkv_w,
            qkv_bias=qkv_bias,
            dims=dims,
            tiles=tiles,
            hidden_base_offset=hidden_base_offset,
            sbm=sbm,
        )
    else:
        output = _qkv_projection_hbm_output(
            hidden_sb=hidden_sb,
            qkv_w=qkv_w,
            qkv_bias=qkv_bias,
            dims=dims,
            tiles=tiles,
            hidden_base_offset=hidden_base_offset,
            output_layout=output_layout,
            d_head=d_head,
            sbm=sbm,
        )

    sbm.close_scope()

    if fused_add:
        return output, hidden_hbm
    else:
        return output


@dataclass
class DimensionSizes(nl.NKIObject):
    B: int
    S: int
    BxS: int
    H: int
    I: int
    H0: int
    H1: int
    num_shards: int
    shard_id: int
    H_shard: int
    H1_shard: int
    H1_offset: int
    H1_per_shard_max: int
    H_per_shard: int
    array_tiling_dim: int
    array_tiling_factor: int
    remainder_array_tiling_dim: int
    remainder_array_tiling_factor: int


@dataclass
class TileCounts(nl.NKIObject):
    H_tile: int
    num_H_tiles_per_H: int
    remainder_H_tile: int
    num_128_tiles_per_H_tile: int
    num_128_tiles_per_remainder_H_tile: int
    num_I_tiles_per_I_shard: int
    remainder_I_tiles: int
    I_tile: int
    array_tiled_H1: int
    remainder_array_tiled_H1: int
    num_I_shards: int


def _calculate_constants(
    B: int,
    S: int,
    H: int,
    I: int,
    num_shards: int,
    shard_id: int,
) -> Tuple[DimensionSizes, TileCounts]:
    """
    Calculate dimension sizes and tile counts for QKV TKG kernel.

    Args:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        I: Output dimension (fused QKV dimension = 3 * num_heads * d_head)
        num_shards: Number of neuron core shards
        shard_id: Current shard ID

    Returns:
        Tuple of (DimensionSizes, TileCounts) with all derived constants
    """
    H0 = nl.tile_size.pmax
    kernel_assert(H % H0 == 0, f"H must be divisible by {H0}, got {H} % {H0} == {H % H0}")
    H1 = H // H0
    BxS = B * S

    # Sharding calculations
    H1_sharded = H1 // num_shards
    H1_remainder = H1 % num_shards
    H1_per_shard_max = H1_sharded + (1 if H1_remainder > 0 else 0)

    kernel_assert(
        H1_remainder == 0,
        f"H1 must be evenly divisible by num_shards, got H={H}, H1={H1}, num_shards={num_shards}, "
        f"H1 % num_shards = {H1_remainder}",
    )
    if H1_remainder == 0:
        # Even split
        H1_shard = H1_sharded
        H1_offset = shard_id * H1_sharded
    else:
        # Uneven split
        if shard_id < H1_remainder:
            H1_shard = H1_sharded + 1
            H1_offset = shard_id * (H1_sharded + 1)
        else:
            H1_shard = H1_sharded
            H1_offset = H1_remainder * (H1_sharded + 1) + (shard_id - H1_remainder) * H1_sharded

    H_shard = H1_shard * H0
    H_per_shard = H1_sharded * H0

    # Intermediate dimension tiling
    H_tile = H_TILE_SIZE
    num_H_tiles_per_H = H_per_shard // H_tile
    remainder_H_tile = H_per_shard % H_tile
    num_128_tiles_per_H_tile = H_tile // 128
    num_128_tiles_per_remainder_H_tile = remainder_H_tile // 128
    # When I is larger than I_SHARD_SIZE, we will shard them
    # and compute each I_SHARD_SIZE tile on SBUF
    num_I_tiles_per_I_shard = (I // I_TILE_SIZE) % 8
    remainder_I_tiles = I % I_TILE_SIZE
    I_tile = min(I, I_TILE_SIZE)
    num_I_shards = (I + I_SHARD_SIZE - 1) // I_SHARD_SIZE

    # Choose Array tiling strategy depending on BxS
    if BxS <= 32:  # do 4x 128P*32F PE array tiling
        array_tiling_dim = 32
    elif BxS <= 64:  # do 2x 128P*64F PE array tiling
        array_tiling_dim = 64
    else:
        array_tiling_dim = 128

    # Adjust hardware-specific logic for column tiling on NeuronCore-v2
    if nisa.get_nc_version() == nisa.nc_version.gen2:
        # Both the row and column sizes in tile_size cannot be 32
        array_tiling_dim = 64

    array_tiling_factor = 128 // array_tiling_dim
    # QKV matmult [BxS, H] @ [H, I] = [BxS, I]
    # H is tiled with H0, resulting in matmul instructions : H1 x [H0(P), I(F)] @ [H0(P), BxS(F)] = [BxS(P), I(F)]
    # In addition, to apply array tiling optmization, H1 dimension is further tiled with array_tiling_factor
    array_tiled_H1 = num_128_tiles_per_H_tile // array_tiling_factor

    # If H is not multiple of H_TILE_SIZE and num_128_tiles_per_remainder_H_tile is not multiple of array_tiling_factor,
    # kernel won't use array tiling
    remainder_array_tiling_dim = array_tiling_dim
    remainder_array_tiling_factor = array_tiling_factor
    if num_128_tiles_per_remainder_H_tile % array_tiling_factor != 0:
        remainder_array_tiling_dim = 128
        remainder_array_tiling_factor = 1
    remainder_array_tiled_H1 = num_128_tiles_per_remainder_H_tile // remainder_array_tiling_factor

    dim_sizes = DimensionSizes(
        B=B,
        S=S,
        BxS=BxS,
        H=H,
        I=I,
        H0=H0,
        H1=H1,
        num_shards=num_shards,
        shard_id=shard_id,
        H_shard=H_shard,
        H1_shard=H1_shard,
        H1_offset=H1_offset,
        H1_per_shard_max=H1_per_shard_max,
        H_per_shard=H_per_shard,
        array_tiling_dim=array_tiling_dim,
        array_tiling_factor=array_tiling_factor,
        remainder_array_tiling_dim=remainder_array_tiling_dim,
        remainder_array_tiling_factor=remainder_array_tiling_factor,
    )

    tile_counts = TileCounts(
        H_tile=H_tile,
        num_H_tiles_per_H=num_H_tiles_per_H,
        remainder_H_tile=remainder_H_tile,
        num_128_tiles_per_H_tile=num_128_tiles_per_H_tile,
        num_128_tiles_per_remainder_H_tile=num_128_tiles_per_remainder_H_tile,
        num_I_tiles_per_I_shard=num_I_tiles_per_I_shard,
        remainder_I_tiles=remainder_I_tiles,
        I_tile=I_tile,
        array_tiled_H1=array_tiled_H1,
        remainder_array_tiled_H1=remainder_array_tiled_H1,
        num_I_shards=num_I_shards,
    )

    return dim_sizes, tile_counts


def _fused_residual_add_hbm2hbm(
    hidden_hbm: nl.ndarray,
    attn_prev_hbm: nl.ndarray,
    mlp_prev_hbm: nl.ndarray,
    dims: DimensionSizes,
    norm_type: NormType,
) -> nl.ndarray:
    """
    Perform fused residual addition in HBM: hidden + attn_prev + mlp_prev.

    Args:
        hidden_hbm: Hidden states in HBM. Shape: (B, S, H)
        attn_prev_hbm: Previous attention residual in HBM. Shape: (B, S, H)
        mlp_prev_hbm: Previous MLP residual in HBM. Shape: (B, S, H)
        dims: Dimension sizes dataclass
        norm_type: Type of normalization (affects sharding strategy)

    Returns:
        Fused hidden states in HBM. Shape: (B, S, H)
    """

    sharding_threshold = rmsnorm_sharding_threshold if norm_type == NormType.RMS_NORM else layernorm_sharding_threshold

    BxS = dims.BxS
    B = dims.B
    S = dims.S
    H = dims.H
    shard_id = dims.shard_id
    num_shards = dims.num_shards

    # Allocate Fused hidden hbm tensor
    if num_shards > 1 and (
        norm_type == NormType.NO_NORM or B * S > sharding_threshold and norm_type != NormType.NO_NORM
    ):
        fused_hidden = nl.ndarray(
            (BxS, H),
            dtype=hidden_hbm.dtype,
            buffer=nl.shared_hbm,
            name="fused_hidden_shared_hbm",
        )
    else:
        fused_hidden = nl.ndarray((BxS, H), dtype=hidden_hbm.dtype, buffer=nl.hbm, name="fused_hidden_hbm")

    # To prevent non-determinism, a different access pattern is needed for offloaded FMA instruction
    if num_shards > 1 and norm_type == NormType.NO_NORM:
        hidden_hbm = hidden_hbm.reshape((BxS, 2, H // 2))
        attn_prev_hbm = attn_prev_hbm.reshape((BxS, 2, H // 2))
        mlp_prev_hbm = mlp_prev_hbm.reshape((BxS, 2, H // 2))
        fused_hidden = fused_hidden.reshape((BxS, 2, H // 2))
        nisa.dma_compute(
            fused_hidden[0:BxS, shard_id, 0 : (H // 2)],
            (
                hidden_hbm[0:BxS, shard_id, 0 : (H // 2)],
                attn_prev_hbm[0:BxS, shard_id, 0 : (H // 2)],
                mlp_prev_hbm[0:BxS, shard_id, 0 : (H // 2)],
            ),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )
    elif norm_type != NormType.NO_NORM and num_shards > 1 and B * S > sharding_threshold:
        kernel_assert(
            (B * S) % 2 == 0,
            f"expected B*S divisible by 2 when B*S={B*S} > {sharding_threshold}",
        )
        hidden_hbm = hidden_hbm.reshape((2, B * S // 2, H))
        attn_prev_hbm = attn_prev_hbm.reshape((2, B * S // 2, H))
        mlp_prev_hbm = mlp_prev_hbm.reshape((2, B * S // 2, H))
        fused_hidden = fused_hidden.reshape((2, B * S // 2, H))
        nisa.dma_compute(
            fused_hidden[shard_id, 0 : (B * S // 2), 0:H],
            (
                hidden_hbm[shard_id, 0 : (B * S // 2), 0:H],
                attn_prev_hbm[shard_id, 0 : (B * S // 2), 0:H],
                mlp_prev_hbm[shard_id, 0 : (B * S // 2), 0:H],
            ),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )
    else:
        hidden_hbm = hidden_hbm.reshape((1, BxS * H))
        attn_prev_hbm = attn_prev_hbm.reshape((1, BxS * H))
        mlp_prev_hbm = mlp_prev_hbm.reshape((1, BxS * H))
        fused_hidden = fused_hidden.reshape((1, BxS * H))
        nisa.dma_compute(
            fused_hidden,
            (hidden_hbm, attn_prev_hbm, mlp_prev_hbm),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )

    return fused_hidden.reshape((B, S, H))


def _fused_norm_and_load(
    hidden_hbm: nl.ndarray,
    norm_type: NormType,
    norm_w: Optional[nl.ndarray],
    norm_bias: Optional[nl.ndarray],
    eps: float,
    hidden_actual: int,
    dims: DimensionSizes,
    sbm: SbufManager,
) -> Tuple[nl.ndarray, int]:
    """
    Perform fused normalization and load from HBM to SBUF.

    Handles three normalization paths:
    - NO_NORM: Direct load with shape (H0, BxS, H1_sharded)
    - RMS_NORM: RMSNorm + load with shape (H0, BxS, H1)
    - LAYER_NORM: LayerNorm + load with shape (H0, BxS, H1)

    Args:
        hidden_hbm: Input hidden states in HBM. Shape: (B, S, H)
        norm_type: Type of normalization (NO_NORM, RMS_NORM, or LAYER_NORM)
        norm_w: Normalization weights (required for RMS/LAYER_NORM). Shape: (1, H)
        norm_bias: LayerNorm bias (required for LAYER_NORM). Shape: (1, H)
        eps: Epsilon for numerical stability
        hidden_actual: Actual hidden dimension for padded tensors
        dims: Dimension sizes dataclass
        sbm: SbufManager object for SBUF allocation

    Returns:
        Tuple of (hidden_sb, hidden_base_offset):
        - hidden_sb: Hidden states in SBUF
          Shape: (H0, BxS, H1_sharded) if NO_NORM, (H0, BxS, H1) if RMS/LAYER_NORM
        - hidden_base_offset: Offset for access patterns
          Value: 0 if single-shard or with-norm, shard_id * H1_sharded if NO_NORM multi-shard
    """

    BxS = dims.BxS
    H0 = dims.H0
    H1 = dims.H1
    shard_id = dims.shard_id
    num_shards = dims.num_shards
    H1_sharded = dims.H1_shard

    hidden_base_offset = 0

    if norm_type == NormType.NO_NORM:
        # Perform direct input load with no norm
        # hidden_sb: (H0, BxS, H1_sharded)
        hidden_sb = sbm.alloc_stack((H0, BxS, H1_sharded), dtype=hidden_hbm.dtype, buffer=nl.sbuf)
        hidden_sb = _input_load(hidden_hbm, hidden_sb, dims, sbm)
    elif norm_type == NormType.RMS_NORM or norm_type == NormType.LAYER_NORM:
        # Perform norm with load
        hidden_sb = sbm.alloc_stack((H0, BxS, H1), dtype=hidden_hbm.dtype, buffer=nl.sbuf)
        if norm_type == NormType.RMS_NORM:
            # Perform rmsnorm with load
            # hidden_sb: (H0, BxS, H1)
            hidden_sb = _rmsnorm_tkg(
                input=hidden_hbm,
                gamma=norm_w,
                output=hidden_sb,
                eps=eps,
                hidden_actual=hidden_actual,
                sbm=sbm,
            )
        elif norm_type == NormType.LAYER_NORM:
            # Perform layernorm with load
            # hidden_sb: (H0, BxS, H1)
            hidden_sb = _layernorm_tkg(
                input=hidden_hbm,
                gamma=norm_w,
                beta=norm_bias,
                output=hidden_sb,
                eps=eps,
                sbm=sbm,
            )
        # Add an offset to account for no norm path having shape (H0, BxS, H1_sharded)
        if num_shards > 1:
            hidden_base_offset = shard_id * H1_sharded

    return hidden_sb, hidden_base_offset


def _input_load(
    hidden_hbm: nl.ndarray,
    hidden_sb: nl.ndarray,
    dims: DimensionSizes,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Load hidden states from HBM to SBUF without normalization.

    Args:
        hidden_hbm: Hidden states in HBM. Shape: (B, S, H)
        hidden_sbuf: Hidden states in SBUF to be loaded to. Shape: (H0, BxS, H1_sharded)
        dims: Dimension sizes dataclass
        sbm: SbufManager object for SBUF allocation

    Returns:
        Hidden states in SBUF. Shape: (H0, BxS, H1_sharded)
    """
    # parse constants
    H0 = dims.H0
    BxS = dims.BxS
    H1 = dims.H1
    H = dims.H
    shard_id = dims.shard_id
    num_shards = dims.num_shards
    H1_sharded = dims.H1_shard
    if num_shards > 1:
        hidden_hbm = hidden_hbm.reshape((BxS, num_shards, H0, H1_sharded))
        hidden_hbm_pattern = [[H1_sharded, H0], [H, BxS], [1, H1_sharded]]
        hidden_hbm_offset = shard_id * H0 * H1_sharded
        nisa.dma_copy(
            hidden_sb,
            hidden_hbm.ap(pattern=hidden_hbm_pattern, offset=hidden_hbm_offset),
        )
    else:
        hidden_hbm = hidden_hbm.reshape((BxS, H0, H1))
        hidden_hbm_pattern = [[H1, H0], [H, BxS], [1, H1]]
        hidden_hbm_offset = 0
        nisa.dma_copy(
            hidden_sb,
            hidden_hbm.ap(pattern=hidden_hbm_pattern, offset=hidden_hbm_offset),
        )

    return hidden_sb


def _initialize_qkv_out_with_bias(
    qkv_out_sb: nl.ndarray,
    qkv_bias: nl.ndarray,
    dims: DimensionSizes,
    shard_idx: int,
    sbm: SbufManager,
) -> None:
    """
    Initialize QKV output buffer with bias or zeros.

    When bias is provided and this is shard_id 0, loads the bias into SBUF
    and broadcasts it across all BxS partitions. Otherwise initializes to zeros.

    Args:
        qkv_out_sb: Output buffer to initialize. Shape: (BxS, I_shard_size)
        qkv_bias: Optional bias tensor. Shape: (1, I_shard_size) or None
        dims: Dimension sizes dataclass
        sbm: SbufManager for SBUF allocation
        shard_idx: Shard index for scope/buffer naming
    """
    BxS = dims.BxS

    if qkv_bias != None and dims.shard_id == 0:
        scope_name = f"qkv_bias_init_shard_{shard_idx}" if shard_idx is not None else "qkv_bias_init_shard"
        sbm.open_scope(name=scope_name)

        buffer_name = f"qkv_bias_sb_{shard_idx}" if shard_idx is not None else "qkv_bias_sb"
        qkv_bias_sb = sbm.alloc_stack(
            qkv_bias.shape,
            dtype=qkv_bias.dtype,
            buffer=nl.sbuf,
            name=buffer_name,
        )
        nisa.dma_copy(qkv_bias_sb, qkv_bias)
        # Broadcast bias to all BxS partitions
        shuffle_mask = [0] * BxS + [255] * (32 - BxS)
        nisa.nc_stream_shuffle(
            src=qkv_bias_sb[0, 0 : qkv_bias.shape[1]],
            dst=qkv_out_sb[0:BxS, 0 : qkv_bias.shape[1]],
            shuffle_mask=shuffle_mask,
        )
        sbm.close_scope()
    else:
        nisa.memset(qkv_out_sb, value=0)


def _qkv_projection_sbuf_output(
    hidden_sb: nl.ndarray,
    qkv_w: nl.ndarray,
    qkv_bias: nl.ndarray,
    dims: DimensionSizes,
    tiles: TileCounts,
    hidden_base_offset: int,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    QKV projection with SBUF output (output_in_sbuf=True path).

    Computes single I-shard (I < I_SHARD_SIZE) with output kept in SBUF.
    Includes neuron core cross-communication when sharded.

    Args:
        hidden_sb: Input hidden states in SBUF. Shape: (H0, BxS, H1_sharded) or (H0, BxS, H1)
        qkv_w: QKV projection weights in HBM. Shape: (H, I)
        qkv_bias: Optional bias in HBM. Shape: (1, I)
        dims: Dimension sizes dataclass
        tiles: Tile counts dataclass
        hidden_base_offset: Offset for hidden access patterns
        sbm: SbufManager for SBUF allocation

    Returns:
        QKV projection output in SBUF. Shape: (BxS, I)
    """

    I = dims.I
    num_shards = dims.num_shards
    shard_id = dims.shard_id
    BxS = dims.BxS
    io_type = hidden_sb.dtype

    # Calculate shapes for sharded weights
    h_offset = dims.H1_offset * dims.H0

    # Allocate qkv_out_sb in heap for now, in future pass in from top level
    qkv_out_sb = sbm.alloc_heap((BxS, I), dtype=io_type, buffer=nl.sbuf)

    _initialize_qkv_out_with_bias(qkv_out_sb, qkv_bias, dims, 0, sbm)

    # output_sb: (BxS, I)
    output_sb = _qkv_projection(
        hidden_sb=hidden_sb,
        qkv_w_hbm=qkv_w,
        qkv_bias_hbm=qkv_bias,
        qkv_out_sb=qkv_out_sb,
        dims=dims,
        tiles=tiles,
        shard_idx=0,
        sbm=sbm,
        outer_h_offset=h_offset,
        outer_h_size=dims.H_per_shard,
        outer_i_offset=0,
        outer_i_size=I,
        hidden_base_offset=hidden_base_offset,
    )

    # Receive qkv projection output from the other neuron core when LNC > 1
    if num_shards > 1:
        sbm.open_scope(name="output_store_sendrecv")
        qkv_recv = sbm.alloc_stack((BxS, I), dtype=io_type, buffer=nl.sbuf)
        other_core = 1 - shard_id
        nisa.sendrecv(
            src=output_sb,
            dst=qkv_recv,
            send_to_rank=other_core,
            recv_from_rank=other_core,
            pipe_id=0,
        )
        nisa.tensor_tensor(output_sb, output_sb, qkv_recv, op=nl.add)
        sbm.close_scope()

    return output_sb


def _qkv_projection_hbm_output(
    hidden_sb: nl.ndarray,
    qkv_w: nl.ndarray,
    qkv_bias: nl.ndarray,
    dims: DimensionSizes,
    tiles: TileCounts,
    hidden_base_offset: int,
    output_layout: QKVOutputLayout,
    d_head: Optional[int],
    sbm: SbufManager,
) -> nl.ndarray:
    """
    QKV projection with HBM output (output_in_sbuf=False path).

    Handles multiple I-shards when I > I_SHARD_SIZE. Each shard is computed
    in SBUF then stored to HBM with layout-specific transformation.
    Includes neuron core cross-communication when sharded.

    Args:
        hidden_sb: Input hidden states in SBUF. Shape: (H0, BxS, H1_sharded) or (H0, BxS, H1)
        qkv_w: QKV projection weights in HBM. Shape: (H, I)
        qkv_bias: Optional bias in HBM. Shape: (1, I) or None
        dims: Dimension sizes dataclass
        tiles: Tile counts dataclass
        hidden_base_offset: Offset for hidden access patterns
        output_layout: Target layout (BSD or NBSd)
        d_head: Head dimension (required for NBSd layout)
        sbm: SbufManager for SBUF allocation

    Returns:
        QKV projection output tensor. Shape depends on output_layout:
        - BSD: (B, S, I)
        - NBSd: (num_heads, B, S, d_head)
    """

    B = dims.B
    S = dims.S
    BxS = dims.BxS
    I = dims.I
    num_shards = dims.num_shards
    shard_id = dims.shard_id
    io_type = hidden_sb.dtype

    # Allocate output tensor with layout-specific shape
    if output_layout == QKVOutputLayout.BSD:
        output = nl.ndarray((BxS, I), dtype=hidden_sb.dtype, buffer=nl.shared_hbm)
    elif output_layout == QKVOutputLayout.NBSd:
        kernel_assert(
            d_head != None,
            f"d_head must be specified for NBSd output layout, got output_layout={output_layout}, d_head={d_head}",
        )
        kernel_assert(
            I % d_head == 0,
            f"I must be divisible by d_head for NBSd output layout, got I={I}, d_head={d_head}, I % d_head = {I % d_head}",
        )
        nh = I // d_head
        output = nl.ndarray((nh, B * S, d_head), dtype=hidden_sb.dtype, buffer=nl.shared_hbm)

    # Process each I shard
    for i_shard_idx in range(tiles.num_I_shards):
        sbm.open_scope(name=f"qkv_hbm_output_i_shard_{i_shard_idx}")

        # Calculate tiles for this shard
        tiles_shard = tiles
        if i_shard_idx < tiles.num_I_shards - 1:
            # Full shard: create tiles_shard with 8 complete I_TILE_SIZE tiles
            tiles_shard = TileCounts(
                H_tile=tiles.H_tile,
                num_H_tiles_per_H=tiles.num_H_tiles_per_H,
                remainder_H_tile=tiles.remainder_H_tile,
                num_128_tiles_per_H_tile=tiles.num_128_tiles_per_H_tile,
                num_128_tiles_per_remainder_H_tile=tiles.num_128_tiles_per_remainder_H_tile,
                num_I_tiles_per_I_shard=8,
                remainder_I_tiles=0,
                I_tile=tiles.I_tile,
                array_tiled_H1=tiles.array_tiled_H1,
                remainder_array_tiled_H1=tiles.remainder_array_tiled_H1,
                num_I_shards=tiles.num_I_shards,
            )

        # Calculate H offset for this projection
        h_offset = dims.H1_offset * dims.H0

        # Calculate I shard range
        I_shard_start = i_shard_idx * I_SHARD_SIZE
        I_shard_end = min(I_shard_start + I_SHARD_SIZE, I)
        I_shard_size = I_shard_end - I_shard_start

        # Allocate output SB that gets accumulated in HBM
        qkv_out_sb = sbm.alloc_stack((BxS, I_shard_size), dtype=io_type, buffer=nl.sbuf)

        # Create bias slice for this shard if bias exists
        if qkv_bias != None:
            qkv_bias_pattern = [[I, 1], [1, I_shard_size]]
            qkv_bias_offset = I_shard_start
            qkv_bias_shard = qkv_bias.ap(pattern=qkv_bias_pattern, offset=qkv_bias_offset)
        else:
            qkv_bias_shard = None

        _initialize_qkv_out_with_bias(qkv_out_sb, qkv_bias_shard, dims, i_shard_idx, sbm)

        # Perform QKV projection for this I shard
        # output_sb: (BxS, I_shard_size)
        output_sb = _qkv_projection(
            hidden_sb=hidden_sb,
            qkv_w_hbm=qkv_w,
            qkv_bias_hbm=qkv_bias_shard,
            qkv_out_sb=qkv_out_sb,
            dims=dims,
            tiles=tiles_shard,
            shard_idx=i_shard_idx,
            sbm=sbm,
            outer_h_offset=h_offset,
            outer_h_size=dims.H_per_shard,
            outer_i_offset=I_shard_start,
            outer_i_size=I_shard_size,
            hidden_base_offset=hidden_base_offset,
        )

        # Receive qkv projection output from the other neuron core when LNC > 1
        if num_shards > 1:
            sbm.open_scope(name=f"output_store_sendrecv_shard_{i_shard_idx}")
            qkv_recv = sbm.alloc_stack((BxS, I_shard_size), dtype=io_type, buffer=nl.sbuf)
            other_core = 1 - shard_id
            nisa.sendrecv(
                src=output_sb,
                dst=qkv_recv,
                send_to_rank=other_core,
                recv_from_rank=other_core,
                pipe_id=0,
            )
            nisa.tensor_tensor(output_sb, output_sb, qkv_recv, op=nl.add)
            sbm.close_scope()

        # Store to HBM with layout-specific transformation
        _store_qkv_output_to_hbm(
            output_hbm=output,
            output_sb=output_sb,
            I_shard_start=I_shard_start,
            I_shard_end=I_shard_end,
            output_layout=output_layout,
            d_head=d_head,
            dims=dims,
        )

        sbm.close_scope()

    # Reshape to expected output shape
    if output_layout == QKVOutputLayout.BSD:
        output = output.reshape((B, S, I))
    elif output_layout == QKVOutputLayout.NBSd:
        n_heads = I // d_head
        output = output.reshape((n_heads, B, S, d_head))

    # Return output in HBM
    return output


def _store_qkv_output_to_hbm(
    output_hbm: nl.ndarray,
    output_sb: nl.ndarray,
    I_shard_start: int,
    I_shard_end: int,
    output_layout: QKVOutputLayout,
    d_head: Optional[int],
    dims: DimensionSizes,
) -> None:
    """
    Store QKV output from SBUF to HBM with layout-specific transformation.

    Handles two output layouts:
    - BSD: (Batch, Seqlen, fused_qkv_dim) -> direct 2D copy
    - NBSd: (num_heads, Batch, Seqlen, d_head) -> reshape with head dimension split

    Args:
        output_hbm: Destination HBM tensor
                    Shape: (BxS, I) for BSD layout
                           (num_heads, BxS, d_head) for NBSd layout
        output_sb: Source SBUF tensor with shape (BxS, I_shard_size)
        I_shard_start: Starting index in I dimension for this shard
        I_shard_end: Ending index in I dimension for this shard (exclusive)
        output_layout: Target layout (BSD or NBSd)
        d_head: Head dimension (required for NBSd layout, can be None for BSD)
        dims: Dimension sizes dataclass containing BxS, I, etc.
    """

    BxS = dims.BxS
    I = dims.I
    I_shard_size = I_shard_end - I_shard_start

    if output_layout == QKVOutputLayout.BSD:
        output_pattern = [[I, BxS], [1, I_shard_size]]
        output_offset = I_shard_start
        nisa.dma_copy(output_hbm.ap(pattern=output_pattern, offset=output_offset), output_sb)

    elif output_layout == QKVOutputLayout.NBSd:
        # NBSd layout: Reshape from (BxS, I_shard_size) to (num_heads, BxS, d_head)
        # Need to split I dimension into num_heads chunks of d_head size

        # Calculate which head indices this shard covers
        ns = I_shard_start // d_head
        ne = I_shard_end // d_head

        # Copy each head's data separately
        for i_n in range(ns, ne):
            output_pattern = [[d_head, BxS], [1, d_head]]
            output_offset = i_n * BxS * d_head

            output_sb_pattern = [[I_shard_size, BxS], [1, d_head]]
            output_sb_offset = (i_n - ns) * d_head

            output_dst = output_hbm.ap(pattern=output_pattern, offset=output_offset)
            output_sb_src = output_sb.ap(pattern=output_sb_pattern, offset=output_sb_offset)
            nisa.dma_copy(output_dst, output_sb_src)


def _qkv_projection(
    hidden_sb: nl.ndarray,
    qkv_w_hbm: nl.ndarray,
    qkv_bias_hbm: nl.ndarray,
    qkv_out_sb: nl.ndarray,
    dims: DimensionSizes,
    tiles: TileCounts,
    shard_idx: int,
    sbm: SbufManager,
    outer_h_offset: int = 0,
    outer_h_size: Optional[int] = None,
    outer_i_offset: int = 0,
    outer_i_size: Optional[int] = None,
    hidden_base_offset: Optional[int] = None,
) -> nl.ndarray:
    # Calculate derived values from dims dataclass
    H1_sharded = dims.H1_shard

    # Define configs
    H0, BxS, H1 = hidden_sb.shape
    H_full, I_full = qkv_w_hbm.shape
    output_dtype = hidden_sb.dtype
    weight_dtype = qkv_w_hbm.dtype

    # Use outer size parameters if provided, otherwise use full dimensions
    H = outer_h_size if outer_h_size != None else H_full
    I = outer_i_size if outer_i_size != None else I_full

    sbm.open_scope(name=f"qkv_projection_shard_{shard_idx}")

    # Allocate all temp buffers: weights, PSUMs
    qkv_w_sb, num_w_tile, result_psum = _allocate_qkv_buffers(
        BxS=BxS,
        I=I,
        H0=H0,
        H1_sharded=H1_sharded,
        qkv_bias_hbm=qkv_bias_hbm,
        output_dtype=output_dtype,
        weight_dtype=weight_dtype,
        dims=dims,
        tiles=tiles,
        shard_idx=shard_idx,
        sbm=sbm,
    )

    # Process all full H tiles
    for H_tile_idx in range(tiles.num_H_tiles_per_H):
        _process_h_tile(
            H_tile_idx=H_tile_idx,
            h_offset=H_tile_idx * tiles.num_128_tiles_per_H_tile,
            num_128_tiles=tiles.num_128_tiles_per_H_tile,
            array_tiled_H1=tiles.array_tiled_H1,
            array_tiling_dim=dims.array_tiling_dim,
            array_tiling_factor=dims.array_tiling_factor,
            qkv_w_hbm=qkv_w_hbm,
            qkv_w_sb=qkv_w_sb,
            hidden_sb=hidden_sb,
            result_psum=result_psum,
            dims=dims,
            tiles=tiles,
            num_w_tile=num_w_tile,
            outer_h_offset=outer_h_offset,
            outer_i_offset=outer_i_offset,
            hidden_base_offset=hidden_base_offset,
            H_full=H_full,
            I_full=I_full,
            H1_sharded=H1_sharded,
            I=I,
        )

    # Process remainder H tile if exists
    if tiles.remainder_H_tile != 0:
        _process_h_tile(
            H_tile_idx=tiles.num_H_tiles_per_H,
            h_offset=tiles.num_H_tiles_per_H * tiles.num_128_tiles_per_H_tile,
            num_128_tiles=tiles.num_128_tiles_per_remainder_H_tile,
            array_tiled_H1=tiles.remainder_array_tiled_H1,
            array_tiling_dim=dims.remainder_array_tiling_dim,
            array_tiling_factor=dims.remainder_array_tiling_factor,
            qkv_w_hbm=qkv_w_hbm,
            qkv_w_sb=qkv_w_sb,
            hidden_sb=hidden_sb,
            result_psum=result_psum,
            dims=dims,
            tiles=tiles,
            num_w_tile=num_w_tile,
            outer_h_offset=outer_h_offset,
            outer_i_offset=outer_i_offset,
            hidden_base_offset=hidden_base_offset,
            H_full=H_full,
            I_full=I_full,
            H1_sharded=H1_sharded,
            I=I,
        )

    # Accumulate PSUMs into output
    _accumulate_psum_to_output(qkv_out_sb=qkv_out_sb, result_psum=result_psum, dims=dims, tiles=tiles, BxS=BxS)

    sbm.close_scope()

    return qkv_out_sb


def _allocate_qkv_buffers(
    BxS: int,
    I: int,
    H0: int,
    H1_sharded: int,
    qkv_bias_hbm: Optional[nl.ndarray],
    output_dtype,
    weight_dtype,
    dims: DimensionSizes,
    tiles: TileCounts,
    shard_idx: int,
    sbm: SbufManager,
) -> Tuple[nl.ndarray, int, list]:
    """
    Allocate all buffers needed for QKV projection: weights, and PSUMs.

    Allocates:
    1. Weight tile buffer (qkv_w_sb) sized to fit remaining SBUF space
    2. PSUM tiles for accumulation (one per I tile)

    Args:
        BxS: Batch * seqlen dimension
        I: Output dimension (fused QKV dimension)
        H0: Partition dimension (nl.tile_size.pmax, typically 128)
        H1_sharded: Sharded H1 dimension (H1 / num_shards)
        qkv_bias_hbm: Optional bias tensor in HBM. Shape: (1, I)
        output_dtype: Data type for output tensor
        weight_dtype: Data type for weight tensor
        dims: Dimension sizes dataclass with shard_id for bias init
        tiles: Tile counts dataclass for PSUM allocation
        shard_idx: I-shard index for buffer naming
        sbm: SbufManager for SBUF allocation

    Returns:
        Tuple of (qkv_out_sb, qkv_w_sb, num_w_tile, result_psum):
        - qkv_w_sb: Weight tile buffer
        - num_w_tile: Number of weight tiles allocated
        - result_psum: List of PSUM tiles
    """

    # Calculate number of weight tiles that fit in remaining space
    remaining_space = sbm.get_free_space()
    size_of_qkv_w_tile = I * tiles.num_128_tiles_per_H_tile * sizeinbytes(weight_dtype)
    num_available_w_tile = remaining_space // size_of_qkv_w_tile
    num_w_tile = min(tiles.num_H_tiles_per_H + (tiles.remainder_H_tile != 0), num_available_w_tile)
    kernel_assert(
        num_w_tile > 0,
        f"Not enough SBUF space for qkv projection weight, need {size_of_qkv_w_tile}, got {remaining_space}",
    )

    # Allocate weight tiles
    qkv_w_sb = sbm.alloc_stack(
        (H0, num_w_tile, tiles.num_128_tiles_per_H_tile, I),
        name=f"qkv_w_sb_shard_{shard_idx}",
        dtype=weight_dtype,
        buffer=nl.sbuf,
    )

    # Allocate PSUM tiles
    n_psum = tiles.num_I_tiles_per_I_shard + (tiles.remainder_I_tiles != 0)
    result_psum = []
    for i in range(n_psum):
        psum_tensor = nl.ndarray(
            (128, tiles.I_tile),
            dtype=nl.float32,
            name=f"batch_result_psum_{shard_idx}_{i}",
            buffer=nl.psum,
        )
        result_psum.append(psum_tensor)

    return qkv_w_sb, num_w_tile, result_psum


def _process_h_tile(
    H_tile_idx: int,
    h_offset: int,
    num_128_tiles: int,
    array_tiled_H1: int,
    array_tiling_dim: int,
    array_tiling_factor: int,
    qkv_w_hbm: nl.ndarray,
    qkv_w_sb: nl.ndarray,
    hidden_sb: nl.ndarray,
    result_psum: list,
    dims: DimensionSizes,
    tiles: TileCounts,
    num_w_tile: int,
    outer_h_offset: int,
    outer_i_offset: int,
    hidden_base_offset: int,
    H_full: int,
    I_full: int,
    H1_sharded: int,
    I: int,
) -> None:
    """
    Process a single H tile: load weights and perform tiled matrix multiplication.

    Unified implementation for both full and remainder H tiles.
    Caller determines tile type and provides appropriate parameters.

    Steps:
    1. Load weight tile from HBM to SBUF using BIR access patterns
    2. Perform nested tiled matmul with array tiling optimization:
       - Outer loop: h1_tile (array tiling chunks)
       - Middle loop: factor (array tiling factor)
       - Inner loop: i_tile (process full I tiles + remainder)

    Args:
        H_tile_idx: Index of current H tile
        h_offset: Offset in H dimension for this tile (in 128-element units)
                  Full tile: H_tile_idx * tiles.num_128_tiles_per_H_tile
                  Remainder: tiles.num_H_tiles_per_H * tiles.num_128_tiles_per_H_tile
        num_128_tiles: Number of 128-element tiles in this H tile
                       Full: tiles.num_128_tiles_per_H_tile
                       Remainder: tiles.num_128_tiles_per_remainder_H_tile
        array_tiled_H1: Number of array-tiled H1 chunks for this tile
                        Full: tiles.array_tiled_H1
                        Remainder: tiles.remainder_array_tiled_H1
        array_tiling_dim: Array tiling dimension (32, 64, or 128)
                          Full: dims.array_tiling_dim
                          Remainder: dims.remainder_array_tiling_dim
        array_tiling_factor: Array tiling factor (128 / array_tiling_dim)
                             Full: dims.array_tiling_factor
                             Remainder: dims.remainder_array_tiling_factor
        qkv_w_hbm: QKV projection weights in HBM. Shape: (H_full, I_full)
        qkv_w_sb: Weight tile buffer in SBUF. Shape: (H0, num_w_tile, num_128_tiles_per_H_tile, I)
        hidden_sb: Hidden states in SBUF. Shape: (H0, BxS, H1) or (H0, BxS, H1_sharded)
        result_psum: List of PSUM tensors for accumulation
        dims: Dimension sizes dataclass
        tiles: Tile counts dataclass
        num_w_tile: Number of weight tiles allocated in SBUF
        outer_h_offset: Offset for H dimension access in full weight tensor
        outer_i_offset: Offset for I dimension access in full weight tensor
        hidden_base_offset: Base offset for hidden access patterns
        H_full: Full H dimension of weight tensor
        I_full: Full I dimension of weight tensor
        H1_sharded: Sharded H1 dimension
        I: Effective I dimension for this projection

    """

    H0, BxS, H1 = hidden_sb.shape

    # Load weight tile from HBM to SBUF
    pattern = [[I_full * H1_sharded, H0], [I_full, num_128_tiles], [1, I]]
    offset = (outer_h_offset + h_offset) * I_full + outer_i_offset
    qkv_w_src = qkv_w_hbm.ap(pattern=pattern, offset=offset)

    qkv_w_sb_pattern = [
        [num_w_tile * tiles.num_128_tiles_per_H_tile * I, H0],
        [I, num_128_tiles],
        [1, I],
    ]
    qkv_w_sb_offset = (H_tile_idx % num_w_tile) * tiles.num_128_tiles_per_H_tile * I
    nisa.dma_copy(qkv_w_sb.ap(pattern=qkv_w_sb_pattern, offset=qkv_w_sb_offset), qkv_w_src)

    # Perform tiled matrix multiplication with array tiling
    for h1_tile in range(array_tiled_H1):
        array_tile_offset = array_tiling_factor * h1_tile

        for factor in range(array_tiling_factor):
            # Process full I tiles
            for i_tile in range(tiles.num_I_tiles_per_I_shard):
                hidden_pattern = [[BxS * H1, H0], [H1, BxS]]
                hidden_offset = hidden_base_offset + h_offset + array_tile_offset + factor

                qkv_w_pattern = [
                    [num_w_tile * tiles.num_128_tiles_per_H_tile * I, H0],
                    [1, 512],
                ]
                qkv_w_offset = (
                    (H_tile_idx % num_w_tile) * tiles.num_128_tiles_per_H_tile * I
                    + (array_tile_offset + factor) * I
                    + i_tile * I_TILE_SIZE
                )

                nisa.nc_matmul(
                    result_psum[i_tile][
                        array_tiling_dim * factor : array_tiling_dim * factor + BxS,
                        0:512,
                    ],
                    hidden_sb.ap(pattern=hidden_pattern, offset=hidden_offset),
                    qkv_w_sb.ap(pattern=qkv_w_pattern, offset=qkv_w_offset),
                    tile_position=(0, array_tiling_dim * factor),
                    tile_size=(H0, array_tiling_dim),
                )

            # Process remainder I tile
            if tiles.remainder_I_tiles != 0:
                hidden_pattern = [[BxS * H1, H0], [H1, BxS]]
                hidden_offset = hidden_base_offset + h_offset + array_tile_offset + factor
                hidden_slice = hidden_sb.ap(pattern=hidden_pattern, offset=hidden_offset)

                qkv_w_pattern = [
                    [num_w_tile * tiles.num_128_tiles_per_H_tile * I, H0],
                    [1, tiles.remainder_I_tiles],
                ]
                qkv_w_offset = (
                    (H_tile_idx % num_w_tile) * tiles.num_128_tiles_per_H_tile * I
                    + (array_tile_offset + factor) * I
                    + tiles.num_I_tiles_per_I_shard * I_TILE_SIZE
                )
                qkv_w_slice = qkv_w_sb.ap(pattern=qkv_w_pattern, offset=qkv_w_offset)

                result_pattern = [[tiles.I_tile, BxS], [1, tiles.remainder_I_tiles]]
                result_offset = array_tiling_dim * factor * tiles.I_tile
                result_slice = result_psum[tiles.num_I_tiles_per_I_shard].ap(
                    pattern=result_pattern, offset=result_offset
                )

                nisa.nc_matmul(
                    result_slice,
                    hidden_slice,
                    qkv_w_slice,
                    tile_position=(0, array_tiling_dim * factor),
                    tile_size=(H0, array_tiling_dim),
                )


def _accumulate_psum_to_output(
    qkv_out_sb: nl.ndarray,
    result_psum: list,
    dims: DimensionSizes,
    tiles: TileCounts,
    BxS: int,
) -> None:
    """
    Accumulate PSUM tiles into final output tensor.

    Combines partial sums from array tiling factors into final output tensor.
    Handles both full I tiles and remainder I tile. Remainder uses simplified
    accumulation when no array tiling is needed.

    Args:
        qkv_out_sb: Output buffer to accumulate into. Shape: (BxS, I)
        result_psum: List of PSUM tensors. Shape: (128, I_tile) each
        dims: Dimension sizes dataclass with array_tiling_dim and array_tiling_factor
        tiles: Tile counts dataclass with num_I_tiles_per_I_shard and remainder_I_tiles
        BxS: Batch * seqlen dimension
    """

    # Accumulate full I tiles
    for factor in range(dims.array_tiling_factor):
        for i_tile in range(tiles.num_I_tiles_per_I_shard):
            qkv_out_sb_slice_start = i_tile * I_TILE_SIZE
            qkv_out_sb_slice_end = i_tile * I_TILE_SIZE + tiles.I_tile
            result_psum_slice_start = dims.array_tiling_dim * factor
            result_psum_slice_end = dims.array_tiling_dim * factor + BxS

            nisa.tensor_tensor(
                qkv_out_sb[0:BxS, qkv_out_sb_slice_start:qkv_out_sb_slice_end],
                qkv_out_sb[0:BxS, qkv_out_sb_slice_start:qkv_out_sb_slice_end],
                result_psum[i_tile][result_psum_slice_start:result_psum_slice_end, 0 : tiles.I_tile],
                op=nl.add,
            )

    # Accumulate remainder I tile
    if tiles.remainder_I_tiles != 0:
        qkv_out_sb_slice_start = tiles.num_I_tiles_per_I_shard * I_TILE_SIZE
        qkv_out_sb_slice_end = tiles.num_I_tiles_per_I_shard * I_TILE_SIZE + tiles.remainder_I_tiles

        if tiles.num_H_tiles_per_H == 0 and dims.remainder_array_tiling_dim == 128:
            # No array tiling for remainder
            nisa.tensor_tensor(
                qkv_out_sb[0:BxS, qkv_out_sb_slice_start:qkv_out_sb_slice_end],
                qkv_out_sb[0:BxS, qkv_out_sb_slice_start:qkv_out_sb_slice_end],
                result_psum[tiles.num_I_tiles_per_I_shard][0:BxS, 0 : tiles.remainder_I_tiles],
                op=nl.add,
            )
        else:
            # With array tiling for remainder
            for factor in range(dims.array_tiling_factor):
                result_psum_slice_start = dims.array_tiling_dim * factor
                result_psum_slice_end = dims.array_tiling_dim * factor + BxS
                nisa.tensor_tensor(
                    qkv_out_sb[0:BxS, qkv_out_sb_slice_start:qkv_out_sb_slice_end],
                    qkv_out_sb[0:BxS, qkv_out_sb_slice_start:qkv_out_sb_slice_end],
                    result_psum[tiles.num_I_tiles_per_I_shard][
                        result_psum_slice_start:result_psum_slice_end,
                        0 : tiles.remainder_I_tiles,
                    ],
                    op=nl.add,
                )
