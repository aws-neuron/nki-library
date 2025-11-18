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
Output Projection TKG Kernel

This kernel implements the output projection operation (attention @ weight +
bias) commonly used after attention blocks in transformer models. The kernel is
specifically optimized for Token Generation (TKG, also known as Decode) scenarios
where the sequence length S is small (often 1 or a small number for spec. decode).

Remark: The input layouts expected for this kernel are different from those for the
CTE kernel. The reason for this is the broader impacts of such layouts on performance
(not just on this kernel but also on other kernels).

In CTE workloads, where sequence length is large, we generally expect to have to reload
it from HBM more frequently. Placing the large S dimension at the end allows more efficient
HBM loads.

For TKG workloads, the S dimension is small, so placing the N dimension next to it
allows more efficient GQA implementations by loading multiple heads at once.

This kernel is designed with LNC support. When LNC>1, the H dimension is sharded
across the cores. We choose to shard on H as this avoids the need for any
inter-core collective operations, as each core produces part of the output tensor.

"""

from typing import Optional, Tuple, List
from dataclasses import dataclass
import nki.isa as nisa
import nki.language as nl
from nki.language import static_range, affine_range

# TODO: Fix this import once available in new FE.
import neuronxcc.nki.typing as nt

from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ..output_projection.output_projection_utils import calculate_head_packing
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_program_sharding_info
import nki

P_MAX = 128
F_MAX = 512

# Conservative limit assuming 2 bytes per scalar and LNC=2, ~20MB SBUF per core for projection weights
MAX_VALIDATED_N_TIMES_H_SIZE = 163840
MAX_VALIDATED_N_TIMES_H_SIZE_FP32 = MAX_VALIDATED_N_TIMES_H_SIZE // 2


@nki.jit
def output_projection_tkg(attention, weight, bias, TRANSPOSE_OUT=False, OUT_IN_SB=False):
    """
    Output Projection Kernel

    This kernel computes
      out = attention @ weight + bias
    typically used to project the output scores after an attention block in transformer models.

    This kernel is optimized for Token Generation (aka Decode) use cases where sequence length S is
    small.

    Data Types:
      This kernel supports `nl.float32`, `nl.float16` and `nl.bfloat16` data types.
      However, for `nl.float32`, large inputs may not fit in SBUF.

    Dimensions:
      B: Batch size
      N: Number of heads
      S: Sequence length
      H: Hidden dimension size
      D: Head dimension size

    Args:
      attention (nl.ndarray):
        Input tensor in HBM or SBUF, typically the scores output from an attention block.
        Shape:    [D, B, N, S]
        Indexing: [d, b, n, s]
      weight (nl.ndarray):
        Weight tensor in HBM
        Shape:    [N * D,     H]
        Indexing: [n * D + d, h]
      bias (nl.ndarray):
        Optional bias tensor in HBM
        Shape:    [1, H]
        Indexing: [1, h]
      TRANSPOSE_OUT (bool):
        Whether need to store the output in transposed shape.

        If False, the output tensor has the following shape and indexing:
          Shape:    [B * S,     H]
          Indexing: [b * S + s, h]

        If True, the output is instead kept in a different shape, which may be
        advantageous for other kernels' performance.
          Shape:    [H_1, H_0, H_2, B * S    ]
          Indexing: [h_1, h_0, h_2, b * S + s]
        where
          H_0 = logical core size (LNC = 1 or LNC = 2),
          H_1 = 128,
          H_2 = H // H_0 // H_1,
        such that h = h_0 * H_1 * H_2 + h_1 * H_2 + h_2.

      OUT_IN_SB (bool):
        If True, output is in SBUF. Else, it is written out to HBM.

    Returns:
      out (nl.ndarray): Output tensor in HBM. Shape depends on `TRANSPOSE_OUT` parameter.

    Restrictions:
      - The product B * S must not exceed 128.
      - Head dimension D must not exceed 128.
      - When TRANSPOSE_OUT is False: H must be divisible by 512 * LNC,
        where LNC is the logical neuron core count (1 or 2).
      - When TRANSPOSE_OUT is True: H must be divisible by 128 * LNC.
      - When TRANSPOSE_OUT is True with float32 dtype: N * H must not exceed 81920.
      - When TRANSPOSE_OUT is True with float16/bfloat16 dtype: N * H must not exceed 163840.
    """
    d_original_size, b_size, n_original_size, s_size = attention.shape
    h_size = weight.shape[1]

    # Logical Neuron Core sharding. n_prgs here corresponds to LNC, the number of actual
    # neuron cores to a single logical neuron core. prg_id is 0 ... n_prgs-1.
    _, n_prgs, prg_id = get_program_sharding_info()

    # ================================================================================
    # Validation
    # ================================================================================

    # If this ever changes we will need to rewrite the entire kernel anyways :)
    kernel_assert(nl.tile_size.pmax == nl.tile_size.gemm_stationary_fmax, "")
    kernel_assert(nl.tile_size.psum_fmax == nl.tile_size.gemm_moving_fmax, "")

    # Validate Input Shapes
    kernel_assert(
        h_size % n_prgs == 0,
        f"output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be divisible by logical core size of {n_prgs}.\n"
        f"Note: H inferred from weight shape: {weight.shape}.",
    )
    kernel_assert(
        weight.shape[0] == n_original_size * d_original_size,
        f"output_projection_tkg kernel requires weight in shape (N * D = {n_original_size * d_original_size}, H = {h_size}), but got {weight.shape}.\n"
        f"Note: N and D inferred from attention score shape: {attention.shape}.",
    )

    if bias != None:
        kernel_assert(
            bias.shape[0] == 1,
            f"output_projection_tkg kernel requires bias in shape (1, H = {h_size}), but got {bias.shape}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )
        kernel_assert(
            bias.shape[1] == h_size,
            f"output_projection_tkg kernel requires bias in shape (1, H = {h_size}), but got {bias.shape}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )

    kernel_assert(
        b_size * s_size <= P_MAX,
        f"output_projection_tkg kernel does not support (B * S = {b_size * s_size}) greater than {P_MAX}.\n"
        f"Note: B and S inferred from attention score shape: {attention.shape}.",
    )
    kernel_assert(
        d_original_size <= P_MAX,
        f"output_projection_tkg kernel does not support head dimension (D = {d_original_size}) greater than {P_MAX}.\n"
        f"Note: D inferred from attention score shape: {attention.shape}.",
    )

    if not TRANSPOSE_OUT:
        kernel_assert(
            h_size % (F_MAX * n_prgs) == 0,
            f"When `TRANSPOSE_OUT` is False, output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be a multiple of {F_MAX} * logical core size, where logical core size is {n_prgs}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )
    else:
        kernel_assert(
            h_size % (P_MAX * n_prgs) == 0,
            f"When `TRANSPOSE_OUT` is True, output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be a multiple of {P_MAX} * logical core size, where logical core size is {n_prgs}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )

        if weight.dtype == nl.float32:
            kernel_assert(
                n_original_size * h_size <= MAX_VALIDATED_N_TIMES_H_SIZE_FP32,
                f"When `TRANSPOSE_OUT` is True and using 32bit floats, output_projection_tkg kernel is not tested for (N * H = {n_original_size * h_size}) greater than {MAX_VALIDATED_N_TIMES_H_SIZE_FP32}.\n"
                f"Note: H inferred from weight shape: {weight.shape}, N inferred from attention score shape: {attention.shape}.",
            )
        else:
            kernel_assert(
                n_original_size * h_size <= MAX_VALIDATED_N_TIMES_H_SIZE,
                f"When `TRANSPOSE_OUT` is True, output_projection_tkg kernel is not tested for (N * H = {n_original_size * h_size}) greater than {MAX_VALIDATED_N_TIMES_H_SIZE}.\n"
                f"Note: H inferred from weight shape: {weight.shape}, N inferred from attention score shape: {attention.shape}.",
            )

    # ================================================================================
    # Configuration
    # ================================================================================

    # Optimize contraction dimension by packing multiple heads into the partition dimension D
    #
    # However, if d_original_size is not divisible by 32 (statebuf_partitions_per_bank),
    # attempting head packing would require copying data from partition D to
    # partition 0 when doing the shuffle operation from attn_sb to attn_shuffled.
    # This is not supported by the hardware, as SBUF->SBUF tensor_copy operations
    # must be 32-aligned in the partition dimension.
    if d_original_size % 32 == 0:
        n_size, d_size, group_size = calculate_head_packing(n_original_size, d_original_size, P_MAX)
    else:
        n_size, d_size, group_size = n_original_size, d_original_size, 1

    cfg = None
    if TRANSPOSE_OUT:
        cfg = generate_tiling_strategy_transpose_out(
            b_size=b_size,
            n_size=n_size,
            d_size=d_size,
            s_size=s_size,
            h_size=h_size,
            n_prgs=n_prgs,
        )
    else:
        cfg = generate_tiling_strategy_regular(
            b_size=b_size,
            n_size=n_size,
            d_size=d_size,
            s_size=s_size,
            h_size=h_size,
            n_prgs=n_prgs,
        )

    # ================================================================================
    # Execution
    # ================================================================================

    w_reshaped = weight.reshape((n_size, d_size, h_size))
    attn_shuffled = load_and_shuffle_attn(
        attention=attention,
        d_original_size=d_original_size,
        n_original_size=n_original_size,
        group_size=group_size,
        cfg=cfg,
    )

    if not TRANSPOSE_OUT:
        out_sb = output_projection_tkg_impl(
            bias=bias,
            prg_id=prg_id,
            w_reshaped=w_reshaped,
            attn_shuffled=attn_shuffled,
            cfg=cfg,
        )

        if not OUT_IN_SB:
            out = nl.ndarray(
                (b_size * s_size, h_size),
                dtype=attn_shuffled.dtype,
                buffer=nl.shared_hbm,
            )
            nisa.dma_copy(
                dst=out[:, nl.ds(prg_id * cfg.h_sharded, cfg.h_sharded)],
                src=out_sb,
            )

    else:  # TRANSPOSE_OUT == True
        # Notes on iteration order:
        #
        # cfg.h_0_size corresponds to the outermost logical iterator h_0 = prg_id from 0 to cfg.num_prgs - 1. This corresponds to LNC sharding.
        # cfg.h_1_size corresponds to the mid logical iterator h_1 from 0 to P_MAX - 1. This is placed in partition dim.
        # cfg.h_2_size corresponds to the innermost logical iterator h_2. This is placed in free dim.
        #
        # Check for h_size % (P_MAX * n_prgs) == 0 above should cover this
        kernel_assert(h_size == cfg.h_0_size * cfg.h_1_size * cfg.h_2_size, "")

        out_sb = output_projection_tkg_transpose_out_impl(
            bias=bias,
            prg_id=prg_id,
            w_reshaped=w_reshaped,
            attn_shuffled=attn_shuffled,
            cfg=cfg,
        )

        if not OUT_IN_SB:
            out = nl.ndarray(
                (cfg.h_1_size, cfg.h_0_size, cfg.h_2_size, b_size * s_size),
                dtype=attn_shuffled.dtype,
                buffer=nl.shared_hbm,
            )
            nisa.dma_copy(
                dst=out.ap(
                    pattern=[
                        [cfg.h_0_size * cfg.h_2_size * b_size * s_size, cfg.h_1_size],
                        [b_size * s_size, cfg.h_2_size],
                        [1, b_size * s_size],
                    ],
                    offset=prg_id * cfg.h_2_size * b_size * s_size,
                ),
                src=out_sb,
            )

    return out if not OUT_IN_SB else out_sb


@dataclass()
class OutputProjectionTkgTilingStrategy(nl.NKIObject):
    num_prgs: int

    b_size: int
    n_size: int
    d_size: int
    s_size: int
    h_size: int

    # For Transpose out = False path
    h_sharded: int

    # For Transpose out = True path
    h_0_size: int
    h_1_size: int
    h_2_size: int


def load_and_shuffle_attn(
    attention: nt.tensor,
    d_original_size: int,
    n_original_size: int,
    group_size: int,
    cfg: OutputProjectionTkgTilingStrategy,
):
    if attention.buffer == nl.sbuf:
        attn_sb = attention
    else:
        attn_sb = nl.ndarray(
            (d_original_size, cfg.b_size, n_original_size, cfg.s_size),
            dtype=attention.dtype,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=attn_sb[...], src=attention[...])

    # Shuffle from attn_sb[d_original_size, B, n_original_size, S] to attn_shuffled[D, N * B * S]
    # Indexing is attn_shuffled[d, n * B * S + b * S + s]
    # Combined reshape + shuffle when group_size > 1
    attn_shuffled = nl.ndarray(
        (cfg.d_size, cfg.n_size * cfg.b_size * cfg.s_size),
        dtype=attn_sb.dtype,
        buffer=nl.sbuf,
    )
    # Use original n_original_size before it was divided by group_siz
    for n_orig in static_range(n_original_size):
        # Map original n to new (n_group, n_offset) coordinates
        n_group = n_orig // group_size
        n_offset = n_orig % group_size
        for b in static_range(cfg.b_size):
            nisa.tensor_copy(
                dst=attn_shuffled[
                    nl.ds(n_offset * d_original_size, d_original_size),
                    nl.ds((n_group * cfg.b_size + b) * cfg.s_size, cfg.s_size),
                ],
                src=attn_sb[:, b, n_orig, :],
            )

    return attn_shuffled


def generate_tiling_strategy_regular(
    b_size: int, n_size: int, d_size: int, s_size: int, h_size: int, n_prgs: int
) -> OutputProjectionTkgTilingStrategy:
    """
    Create and return an OutputProjectionTkgTilingStrategy object containing the
    relevant dimension size limits and tiling decisions for this kernel, when
    `TRANSPOSE_OUT` is False.
    """
    return OutputProjectionTkgTilingStrategy(
        num_prgs=n_prgs,
        b_size=b_size,
        n_size=n_size,
        d_size=d_size,
        s_size=s_size,
        h_size=h_size,
        h_sharded=h_size // n_prgs,
        h_0_size=-1,  # Not used in this case
        h_1_size=-1,  # Not used in this case
        h_2_size=-1,  # Not used in this case
    )


def generate_tiling_strategy_transpose_out(
    b_size: int, n_size: int, d_size: int, s_size: int, h_size: int, n_prgs: int
) -> OutputProjectionTkgTilingStrategy:
    """
    Create and return an OutputProjectionTkgTilingStrategy object containing the
    relevant dimension size limits and tiling decisions for this kernel, when
    `TRANSPOSE_OUT` is True.
    """
    return OutputProjectionTkgTilingStrategy(
        num_prgs=n_prgs,
        b_size=b_size,
        n_size=n_size,
        d_size=d_size,
        s_size=s_size,
        h_size=h_size,
        h_sharded=h_size // n_prgs,
        # H is sharded across cores, then further broken up in P_MAX size chunks.
        h_0_size=n_prgs,
        h_1_size=P_MAX,
        h_2_size=h_size // n_prgs // P_MAX,
    )


def output_projection_tkg_impl(
    bias,
    prg_id,
    w_reshaped,
    attn_shuffled,
    cfg: OutputProjectionTkgTilingStrategy,
):
    if bias != None:
        # Load bias at once, this can be improved if cfg.h_size is large.
        bias_sb_1d = nl.ndarray((1, cfg.h_sharded), dtype=bias.dtype)
        nisa.dma_copy(
            src=bias.ap(
                pattern=[[cfg.h_sharded, 1], [1, cfg.h_sharded]],
                offset=prg_id * cfg.h_sharded,
            ),
            dst=bias_sb_1d,
        )
        bias_sb = nl.ndarray((cfg.b_size * cfg.s_size, cfg.h_sharded), dtype=bias.dtype, buffer=nl.sbuf)
        # Broadcast bias from [1, cfg.h_sharded] to [B*S, cfg.h_sharded] to match out_sb shape below.
        stream_shuffle_broadcast(bias_sb_1d, bias_sb)

    # Potentially load w_reshaped into separate blocks in sbuf to allow prefetching.
    # 2K block size, otherwise just 1 block.
    # TODO: Fine tune this number.
    h_block_size = 2048 if cfg.h_sharded % 2048 == 0 else cfg.h_sharded
    num_h_blocks_per_prg = cfg.h_sharded // h_block_size
    kernel_assert(num_h_blocks_per_prg * h_block_size * cfg.num_prgs == cfg.h_size, "")

    # By loading into separate sbuf tensors, we give the compiler finer control over pre-fetching.
    w_sbuf_blocks = []  # overall shape is [num_h_blocks_per_prg][cfg.n_size][cfg.d_size, h_block_size]
    for h_block in affine_range(num_h_blocks_per_prg):
        w_row = []
        for n in affine_range(cfg.n_size):
            w_tensor = nl.ndarray((cfg.d_size, h_block_size), dtype=w_reshaped.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                src=w_reshaped.ap(
                    pattern=[[cfg.h_size, cfg.d_size], [1, h_block_size]],
                    offset=n * cfg.d_size * cfg.h_size + (prg_id * num_h_blocks_per_prg + h_block) * h_block_size,
                ),
                dst=w_tensor,
            )
            w_row.append(w_tensor)
        w_sbuf_blocks.append(w_row)

    out_sb = nl.ndarray(
        (cfg.b_size * cfg.s_size, cfg.h_sharded),
        dtype=attn_shuffled.dtype,
        buffer=nl.sbuf,
    )

    num_free_dim_tiles_per_h_block = h_block_size // F_MAX

    # Compute and write out attention @ weight (+ bias) blocks
    for h_block in affine_range(num_h_blocks_per_prg):
        for h_block_f_tile in affine_range(num_free_dim_tiles_per_h_block):
            res_psum = nl.ndarray((cfg.b_size * cfg.s_size, F_MAX), dtype=nl.float32, buffer=nl.psum)

            # Accumulate (B*S, F_MAX) tiled attn @ weight blocks for all cfg.n_size heads
            for n in affine_range(cfg.n_size):
                stationary = attn_shuffled[:, nl.ds(n * cfg.b_size * cfg.s_size, cfg.b_size * cfg.s_size)]
                moving = w_sbuf_blocks[h_block][n][:, nl.ds(h_block_f_tile * F_MAX, F_MAX)]
                nisa.nc_matmul(res_psum, stationary, moving)

            # Read out from psum, possibly adding bias if present.
            h_offset = h_block * h_block_size + h_block_f_tile * F_MAX
            if bias != None:
                nisa.tensor_tensor(
                    dst=out_sb[:, nl.ds(h_offset, F_MAX)],
                    data1=res_psum[:, :F_MAX],
                    data2=bias_sb[:, nl.ds(h_offset, F_MAX)],
                    op=nl.add,
                )
            else:
                nisa.tensor_copy(dst=out_sb[:, nl.ds(h_offset, F_MAX)], src=res_psum[:, :F_MAX])
    return out_sb


def output_projection_tkg_transpose_out_impl(
    bias,
    prg_id,
    w_reshaped,
    attn_shuffled,
    cfg: OutputProjectionTkgTilingStrategy,
):
    out_sb = nl.ndarray(
        (cfg.h_1_size, cfg.h_2_size * cfg.b_size * cfg.s_size),
        dtype=attn_shuffled.dtype,
        buffer=nl.sbuf,
    )

    if bias != None:
        bias_sb = nl.ndarray((cfg.h_1_size, cfg.h_2_size), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bias_sb,
            src=bias.ap(
                pattern=[[cfg.h_2_size, cfg.h_1_size], [1, cfg.h_2_size]],
                offset=prg_id * cfg.h_1_size * cfg.h_2_size,
            ),
        )

    # Load weights at once.
    w_sbuf = nl.ndarray(
        (cfg.d_size, cfg.n_size, cfg.h_1_size * cfg.h_2_size),
        dtype=w_reshaped.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=w_sbuf.ap(
            pattern=[
                [cfg.n_size * (cfg.h_1_size * cfg.h_2_size), cfg.d_size],
                [cfg.h_1_size * cfg.h_2_size, cfg.n_size],
                [1, cfg.h_1_size * cfg.h_2_size],
            ]
        ),
        src=w_reshaped.ap(
            pattern=[
                [cfg.h_size, cfg.d_size],
                [cfg.d_size * cfg.h_size, cfg.n_size],
                [1, cfg.h_1_size * cfg.h_2_size],
            ],
            offset=(cfg.h_1_size * cfg.h_2_size) * prg_id,
        ),
    )

    # Weights stationary, input (attn_shuffled) moving in, with B*S on free-dim each time.
    # For a PSUM bank with 512 (F_MAX) entries, we can pack multiple B*S groups (i.e. different values of h_c)
    # in one bank and copy the whole bank to SBUF at once.
    NUM_BS_PER_PSUM_BANK = F_MAX // (cfg.b_size * cfg.s_size)
    NUM_PSUM_TILES = div_ceil(cfg.h_2_size, NUM_BS_PER_PSUM_BANK)

    for i in affine_range(NUM_PSUM_TILES):
        # Accumulate (cfg.h_1_size, F_MAX) sized attn @ weight blocks for all cfg.n_size heads,
        # for a total of NUM_PSUM_TILES different resulting psum tiles.
        # Note that each PSUM row will have NUM_BS_PER_PSUM_BANK many B*S groups.
        # In effect, we are packing part of cfg.h_2_size dimension together with B*S.
        res_psum = nl.ndarray((cfg.h_1_size, F_MAX), dtype=nl.float32, buffer=nl.psum)
        for j in affine_range(NUM_BS_PER_PSUM_BANK):
            h_c_offset = i * NUM_BS_PER_PSUM_BANK + j
            for n in affine_range(cfg.n_size):
                moving = attn_shuffled[:, nl.ds(n * cfg.b_size * cfg.s_size, cfg.b_size * cfg.s_size)]
                stationary = w_sbuf.ap(
                    pattern=[
                        [cfg.n_size * (cfg.h_1_size * cfg.h_2_size), cfg.d_size],
                        [cfg.h_2_size, cfg.h_1_size],
                    ],
                    offset=n * (cfg.h_1_size * cfg.h_2_size) + h_c_offset,
                )
                # if cfg.h_2_size is not divsible by NUM_BS_PER_PSUM_BANK,
                # the last psum tile may be "incomplete".
                if h_c_offset < cfg.h_2_size:
                    nisa.nc_matmul(
                        dst=res_psum[
                            :,
                            nl.ds(j * cfg.b_size * cfg.s_size, cfg.b_size * cfg.s_size),
                        ],
                        stationary=stationary,
                        moving=moving,
                    )

        # Last psum tile may be "incomplete" if cfg.h_2_size is not divisible by NUM_BS_PER_PSUM_BANK
        num_BS_for_current_psum_tile = min(NUM_BS_PER_PSUM_BANK, cfg.h_2_size - i * NUM_BS_PER_PSUM_BANK)

        # Read out from psum, possibly adding bias if present.ß
        dst_sb_access_pattern = [
            [cfg.h_2_size * cfg.b_size * cfg.s_size, cfg.h_1_size],
            [cfg.b_size * cfg.s_size, num_BS_for_current_psum_tile],
            [1, cfg.b_size * cfg.s_size],
        ]
        dst_sb_access_offset = i * NUM_BS_PER_PSUM_BANK * cfg.b_size * cfg.s_size
        psum_access_pattern = [
            [F_MAX, cfg.h_1_size],
            [cfg.b_size * cfg.s_size, num_BS_for_current_psum_tile],
            [1, cfg.b_size * cfg.s_size],
        ]
        bias_access_pattern = [
            [cfg.h_2_size, cfg.h_1_size],
            [1, num_BS_for_current_psum_tile],
            [0, cfg.b_size * cfg.s_size],
        ]
        bias_access_offset = i * NUM_BS_PER_PSUM_BANK
        if bias != None:
            nisa.tensor_tensor(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                data1=res_psum.ap(pattern=psum_access_pattern),
                data2=bias_sb.ap(
                    pattern=bias_access_pattern,
                    offset=bias_access_offset,
                ),
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                src=res_psum.ap(pattern=psum_access_pattern),
            )

    # Store out as transposed.
    out_sb = out_sb.reshape((cfg.h_1_size, cfg.h_2_size, cfg.b_size * cfg.s_size))
    return out_sb
