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
Output Projection CTE Kernel

This kernel implements the output projection operation (attention @ weight +
bias) commonly used after attention blocks in transformer models. The kernel is
specifically optimized for Context Encoding (CTE, also known as Prefill) scenarios
where the sequence length S is large.

This kernel is designed with LNC support. When LNC>1, the H dimension is sharded
across the cores. We choose to shard on H as this avoids the need for any
inter-core collective operations, as each core produces part of the output tensor.

"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import nki.language as nl
import nki.isa as nisa
import nki

# TODO: Fix this import once available in new FE.
import neuronxcc.nki.typing as nt

from ..utils.kernel_helpers import div_ceil, get_program_sharding_info
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ..output_projection.output_projection_utils import calculate_head_packing
from ..utils.kernel_assert import kernel_assert

P_MAX = 128
F_MAX = 512

MAX_VALIDATED_H_SIZE = 16384 + 4321  # For some reason we have a unit test like this?
MAX_VALIDATED_B_TIMES_S_SIZE = 128 * 1024
MAX_VALIDATED_N_SIZE = 17  # For some reason we have a unit test for 17.


@nki.jit
def output_projection_cte(
    attention,
    weight,
    bias=None,
):
    """
    Output Projection Kernel

    This kernel computes
      out = attention @ weight + bias
    typically used to project the output scores after an attention block in transformer models.

    This kernel is optimized for Context Encoding (aka Prefill) use cases where sequence length S is
    large. Using this kernel with S < 512 may result in degraded performance.

    This kernel uses a layout also used by other Context Encoding kernels to avoid need for
    transposes.

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
        Input tensor in HBM, typically the scores output from an attention block.
        Shape:    [B, N, D, S]
        Indexing: [b, n, d, s]
      weight (nl.ndarray):
        Weight tensor in HBM.
        Shape:    [N * D,     H]
        Indexing: [n * D + d, h]
      bias (nl.ndarray):
        Optional bias tensor in HBM.
        Shape:    [1, H]
        Indexing: [1, h]

    Returns:
      out (nl.ndarray):
        Output tensor in HBM.
        Shape:    [B, S, H].
        Indexing: [b, s, h].

    Restrictions:
      - The product B * S must not exceed 131072.
      - Head dimension D must not exceed 128.
      - Hidden dimension H must not exceed 20705 (not fully tested beyond this value).
      - Number of heads N must not exceed 17 (not fully tested beyond this value).
      - Hidden dimension H must be divisible by LNC * 512,
        where LNC is the logical neuron core count (1 or 2).
    """
    b_size, n_original_size, d_original_size, s_size = attention.shape
    _, h_size = weight.shape

    # Logical Neuron Core sharding. n_prgs here corresponds to LNC, the number of actual
    # neuron cores to a single logical neuron core. prg_id is 0 ... n_prgs-1.
    _, n_prgs, prg_id = get_program_sharding_info()

    # ================================================================================
    # Validation
    # ================================================================================

    # If this ever changes we will need to rewrite the entire kernel anyways :)
    kernel_assert(nl.tile_size.pmax == nl.tile_size.gemm_stationary_fmax, "")
    kernel_assert(nl.tile_size.psum_fmax == nl.tile_size.gemm_moving_fmax, "")

    kernel_assert(
        n_original_size * d_original_size == weight.shape[0],
        f"output_projection_cte kernel requires weight in shape (N * D = {n_original_size * d_original_size}, H = {h_size}), but got {weight.shape}.\n"
        f"Note: N and D inferred from attention score shape: {attention.shape}.",
    )

    # Remove these at your own risk. Overly large values may perform poorly or fail to compile.
    kernel_assert(
        h_size <= MAX_VALIDATED_H_SIZE,
        f"output_projection_cte kernel is not tested for (H = {h_size}) greater than {MAX_VALIDATED_H_SIZE}.\n"
        f"Note: H inferred from weight shape: {weight.shape}.",
    )
    kernel_assert(
        b_size * s_size <= MAX_VALIDATED_B_TIMES_S_SIZE,
        f"output_projection_cte kernel is not tested for (B * S = {b_size * s_size}) greater than {MAX_VALIDATED_B_TIMES_S_SIZE}.\n"
        f"Note: B and S inferred from attention score shape: {attention.shape}.",
    )
    kernel_assert(
        n_original_size <= MAX_VALIDATED_N_SIZE,
        f"output_projection_cte kernel is not tested for (N = {n_original_size}) greater than {MAX_VALIDATED_N_SIZE}.\n"
        f"Note: N inferred from attention score shape: {attention.shape}.",
    )

    # Note: would require padding to support h_size not divisible by n_prgs.
    kernel_assert(
        h_size % n_prgs == 0,
        f"output_projection_cte kernel requires hidden dimension (H = {h_size}) to be divisible by logical core size of {n_prgs}.\n"
        f"Note: H inferred from weight shape: {weight.shape}.",
    )

    if bias != None:
        kernel_assert(
            bias.shape[0] == 1,
            f"output_projection_cte kernel requires bias in shape (1, H={h_size}), but got {bias.shape}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )
        kernel_assert(
            bias.shape[1] == h_size,
            f"output_projection_cte kernel requires bias in shape (1, H={h_size}), but got {bias.shape}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )

    # Note: would have to tile D dimension to support this.
    kernel_assert(
        d_original_size <= P_MAX,
        f"output_projection_cte kernel does not support head dimension (D = {d_original_size}) greater than {P_MAX}.\n"
        f"Note: D inferred from attention score shape: {attention.shape}.",
    )

    # ================================================================================
    # Configuration
    # ================================================================================

    # If D <= 128 / 2, we can process multiple heads in a single tile by packing them into the partition dimension.
    # Here we take advantage of the fact that an output projection with N heads of size D each has a computationally
    # equivalent output projection with N/k heads of size D*k each. This maximizes use of the PE engine.
    n_size, d_size, group_size = calculate_head_packing(n_original_size, d_original_size, P_MAX)

    # Build a `cfg` object which specifies how to tile the computation.
    # We first shard H across the LNC cores (if n_prgs>1). Then for each LNC core
    # we can further break H and S into blocks.
    cfg = generate_tiling_strategy(b_size, n_size, d_size, s_size, h_size, n_prgs)

    # ================================================================================
    # Execution
    # ================================================================================

    out = nl.ndarray((b_size, s_size, h_size), dtype=attention.dtype, buffer=nl.shared_hbm)
    weight = weight.reshape((n_size, d_size, h_size))
    if group_size > 1:
        attention = attention.reshape((b_size, n_size, d_size, s_size))

    w_sbuf_blocks = load_weight_sbuf_blocks(weight_hbm=weight, prg_id=prg_id, cfg=cfg)
    bias_sb = load_and_broacast_bias(bias_hbm=bias, prg_id=prg_id, cfg=cfg) if bias != None else None

    for b in range(b_size):
        for s_block in range(cfg.num_s_blocks):
            process_batch_tile(
                attention=attention,
                w_sbuf_blocks=w_sbuf_blocks,
                bias_sb=bias_sb,
                out=out,
                curr_b=b,
                curr_s_block=s_block,
                prg_id=prg_id,
                cfg=cfg,
            )

    return out


@dataclass()
class OutputProjectionCteTilingStrategy(nl.NKIObject):
    num_prgs: int

    b_size: int
    n_size: int
    d_size: int
    s_size: int
    h_size: int

    h_sharded_size: int
    h_block_size: int
    num_h_blocks_per_prg: int

    s_block_size: int
    num_s_blocks: int

    def curr_s_block_size(self, curr_s_block_idx: int) -> int:
        return min(self.s_block_size, self.s_size - curr_s_block_idx * self.s_block_size)

    def curr_h_block_size(self, curr_h_block_idx: int) -> int:
        return min(
            self.h_block_size,
            self.h_sharded_size - curr_h_block_idx * self.h_block_size,
        )


def generate_tiling_strategy(
    b_size: int, n_size: int, d_size: int, s_size: int, h_size: int, n_prgs: int
) -> OutputProjectionCteTilingStrategy:
    """
    Create and return an OutputProjectionCteTilingStrategy object containing the
    relevant dimension size limits and tiling decisions for this kernel.
    """
    h_sharded_size = h_size // n_prgs

    h_block_size = h_sharded_size  # TODO: fine tune this number?
    num_h_blocks_per_prg = div_ceil(h_sharded_size, h_block_size)

    s_block_size = F_MAX  # TODO: fine tune this number?
    num_s_blocks = div_ceil(s_size, s_block_size)

    return OutputProjectionCteTilingStrategy(
        num_prgs=n_prgs,
        b_size=b_size,
        n_size=n_size,
        d_size=d_size,
        s_size=s_size,
        h_size=h_size,
        h_sharded_size=h_sharded_size,
        h_block_size=h_block_size,
        num_h_blocks_per_prg=num_h_blocks_per_prg,
        s_block_size=s_block_size,
        num_s_blocks=num_s_blocks,
    )


def load_and_broacast_bias(bias_hbm: nt.tensor, prg_id: int, cfg: OutputProjectionCteTilingStrategy) -> nl.ndarray:
    """
    Loads bias_hbm into sbuf and broadcasts it to [P_MAX, cfg.num_h_blocks_per_prg * cfg.h_block_size],
    returning the resulting tensor.

    Note the broadcasting allows adding each bias value to (up to) 128 elements of the resulting tensor at once,
    utilizing the full 128 vector engine lanes available.

    If using LNC sharding, prg_id specifies which part of the bias to load. Otherwise prg_id should be 0.

    Note that if cfg.num_h_blocks_per_prg * cfg.h_block_size > cfg.h_sharded_size, which happens when cfg.h_block_size does not exactly
    divide cfg.h_size, the resulting tensor will contain some garbage data at the end.
    """
    # Remark: If NUM_H_BLOCKS_PER_PRG * H_BLOCK_SIZE > H_SHARDED, bias_sb will contain some garbage data at the end.
    bias_sb_1d = nl.ndarray(
        (1, cfg.num_h_blocks_per_prg * cfg.h_block_size),
        dtype=bias_hbm.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        src=bias_hbm.ap(
            pattern=[[cfg.h_sharded_size, 1], [1, cfg.h_sharded_size]],
            offset=cfg.h_sharded_size * prg_id,
        ),
        dst=bias_sb_1d,
    )

    # Broadcast bias from [1, NUM_H_BLOCKS_PER_PRG * H_BLOCK_SIZE] to [b_size*s_size, NUM_H_BLOCKS_PER_PRG * H_BLOCK_SIZE] to match out_sb shape below.
    bias_sb = nl.ndarray(
        (P_MAX, cfg.num_h_blocks_per_prg * cfg.h_block_size),
        dtype=bias_hbm.dtype,
        buffer=nl.sbuf,
    )
    stream_shuffle_broadcast(bias_sb_1d, bias_sb)

    return bias_sb


def load_weight_sbuf_blocks(
    weight_hbm: nt.tensor, prg_id: int, cfg: OutputProjectionCteTilingStrategy
) -> List[List[nl.ndarray]]:
    """
    Loads weights into sbuf blocks.

    If using LNC sharding, prg_id specifies which part of the weights to load. Otherwise prg_id should be 0.

    Returns `w_sbuf_blocks` with overall shape [cfg.num_h_blocks_per_prg][cfg.n_size][cfg.d_size, cfg.h_block_size].

    If cfg.num_h_blocks_per_prg * cfg.h_block_size > cfg.h_size, which happens when cfg.h_block_size does not exactly
    divide cfg.h_size, the last row of blocks will contain some garbage data at the end.
    """
    w_sbuf_blocks = []
    for h_block in range(cfg.num_h_blocks_per_prg):
        curr_h_block_size = cfg.curr_h_block_size(h_block)
        w_row = []
        for n in range(cfg.n_size):
            # Remark: If curr_h_block_size < cfg.h_block_size, tensor will contain some garbage data at the end.
            w_tensor = nl.ndarray((cfg.d_size, cfg.h_block_size), dtype=weight_hbm.dtype, buffer=nl.sbuf)

            nisa.dma_copy(
                w_tensor,
                weight_hbm.ap(
                    pattern=[[cfg.h_size, cfg.d_size], [1, curr_h_block_size]],
                    offset=n * cfg.d_size * cfg.h_size + (cfg.h_sharded_size * prg_id + h_block * cfg.h_block_size),
                ),
            )
            w_row.append(w_tensor)
        w_sbuf_blocks.append(w_row)
    return w_sbuf_blocks


def process_batch_tile(
    # Tensors
    attention: nt.tensor,
    w_sbuf_blocks: List[List[nl.ndarray]],
    bias_sb: nl.ndarray,
    out: nt.tensor,
    # Current tile
    curr_b: int,
    curr_s_block: int,
    prg_id: int,
    # Cfg
    cfg: OutputProjectionCteTilingStrategy,
):
    """
    Calculates a tile of attention @ weight + bias and writes to `out`.

    Expects `w_sbuf_blocks` from `load_weight_sbuf_blocks` and
    `bias_sb` from `load_and_broacast_bias`.

    Specifically, this writes tile out[b, s, h] comprised by a single value of b = `curr_b`,
    a slice of s corresponding to the block `curr_s_block`, and a slice of h
    corresponding to this neuron core's shard (if using LNC > 1, specified by `prg_id`).
    """
    curr_s_tile_size = cfg.curr_s_block_size(curr_s_block)
    num_s_subtiles = div_ceil(curr_s_tile_size, P_MAX)

    # Load attention slices into SBUF. We expect the compiler to schedule these loads efficiently close to their uses.
    attention_sb = []
    for n in range(cfg.n_size):
        attention_tensor = nl.ndarray((cfg.d_size, cfg.s_block_size), dtype=attention.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            attention_tensor.ap([[cfg.s_block_size, cfg.d_size], [1, curr_s_tile_size]]),
            attention.ap(
                pattern=[[cfg.s_size, cfg.d_size], [1, curr_s_tile_size]],
                offset=(
                    curr_b * cfg.n_size * cfg.d_size * cfg.s_size
                    + n * cfg.d_size * cfg.s_size
                    + curr_s_block * cfg.s_block_size
                ),
            ),
        )
        attention_sb.append(attention_tensor)

    # Setup SBUF buffers to store results in.
    result_sb = []
    for s_subtile in range(num_s_subtiles):
        result_sb.append(nl.ndarray((P_MAX, cfg.h_sharded_size), dtype=attention.dtype, buffer=nl.sbuf))

    # Compute result slices by multiplying attention @ weight and possibly adding bias.
    for s_subtile in range(num_s_subtiles):
        for h_block in range(cfg.num_h_blocks_per_prg):
            cur_h_tile = cfg.h_block_size
            num_h_subtiles = div_ceil(cur_h_tile, F_MAX)
            kernel_assert(
                cur_h_tile % F_MAX == 0,
                f"The H dimension must be a multiple of n_prgs * F_MAX = {cfg.num_prgs*F_MAX}, but got {cfg.h_size}",
            )
            for h_subtile in range(num_h_subtiles):
                res_psum = nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum)

                for n in range(cfg.n_size):
                    attention_slice = attention_sb[n][0 : cfg.d_size, s_subtile * P_MAX : s_subtile * P_MAX + P_MAX]
                    weight_slice = w_sbuf_blocks[h_block][n][
                        0 : cfg.d_size, h_subtile * F_MAX : h_subtile * F_MAX + F_MAX
                    ]

                    nisa.nc_matmul(
                        res_psum,
                        attention_slice,
                        weight_slice,
                    )

                if bias_sb != None:
                    nisa.tensor_tensor(
                        result_sb[s_subtile][
                            0:P_MAX,
                            nl.ds(h_block * cfg.h_block_size + h_subtile * F_MAX, F_MAX),
                        ],
                        res_psum,
                        bias_sb[
                            0:P_MAX,
                            nl.ds(h_block * cfg.h_block_size + h_subtile * F_MAX, F_MAX),
                        ],
                        nl.add,
                    )
                else:
                    nisa.tensor_copy(
                        result_sb[s_subtile][
                            :P_MAX,
                            h_block * cfg.h_block_size + h_subtile * F_MAX : (
                                (h_block * cfg.h_block_size + h_subtile * F_MAX) + F_MAX
                            ),
                        ],
                        res_psum,
                        # Alternate between scalar and vector engine for tensor copy to allow better pipelining of the `h_subtile`s.
                        engine=nisa.scalar_engine if s_subtile % 2 == 0 else nisa.vector_engine,
                    )

    # Write results to output.
    for s_subtile in range(num_s_subtiles):
        curr_s_subtile_size = min(P_MAX, curr_s_tile_size - s_subtile * P_MAX)
        nisa.dma_copy(
            out.ap(
                pattern=[[cfg.h_size, curr_s_subtile_size], [1, cfg.h_sharded_size]],
                offset=(
                    curr_b * cfg.s_size * cfg.h_size
                    + (curr_s_block * cfg.s_block_size + s_subtile * P_MAX) * cfg.h_size
                    + cfg.h_sharded_size * prg_id
                ),
            ),
            result_sb[s_subtile][:curr_s_subtile_size, : cfg.h_sharded_size],
        )
