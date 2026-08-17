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

"""MXFP8 SwiGLU MLP forward kernels with optional activation checkpointing."""

import nki.isa as nisa
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ...matmul_mxfp8.matmul_mxfp8_generic_api import generic_matmul_mxfp8_api
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor
from ...mxfp_utils.mxfp8_utils.common_utils import create_and_set_active_sbm, get_active_sbm, with_active_sbm
from ...mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import INTERLEAVE_FACTOR
from ..common_utils import (
    DGT_MIN_K,
    MAX_TILES_IN_LOAD_M,
    NUM_GATE_UP_PROJECTIONS,
    NUM_LNC2_CORES,
    _allocate_spill_buffer,
    _build_matmul_params,
    _compute_load_tile_shape,
    get_tile_sizes,
)


def compute_fused_gate_up_down_mxfp8(
    hidden_td: TensorDescriptor,
    gate_up_td: TensorDescriptor,
    down_w_td: TensorDescriptor,
    int_td: TensorDescriptor,
    output_td: TensorDescriptor,
    s_base_offset: int,
    dtype,
    TILES_IN_BLOCK_M: int = 8,
    TILES_IN_BLOCK_N_GU: int = 1,
    TILES_IN_BLOCK_K_GU: int = 8,
    TILES_IN_BLOCK_M_DOWN: int = 8,
    TILES_IN_BLOCK_N_DOWN: int = 1,
    TILES_IN_BLOCK_K_DOWN: int = 8,
    save_gate_pre_td: TensorDescriptor = None,
    save_gate_act_td: TensorDescriptor = None,
    save_up_td: TensorDescriptor = None,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
):
    """Fused gate/up + SiLU + multiply + down projection using TensorDescriptors.

    Uses HBM for the intermediate tensor. All loads use load_and_quantize_tile
    with TileLocation/TensorDescriptor.

    For each M-block of TILES_IN_BLOCK_M tiles:
      Phase 1: gate/up matmul -> SiLU(gate) * up -> write intermediate to HBM
      Phase 2: read intermediate from HBM via DGT, matmul with down weights

    Dimensions:
        S: Sequence length (number of tokens).
        H: Hidden dimension size.
        I: Intermediate dimension size (per gate/up projection).

    Args:
        hidden_td (TensorDescriptor): [S, H], input hidden states (is_f_by_k=True).
        gate_up_td (TensorDescriptor): [2I, H], fused gate+up weight matrix (is_f_by_k=True).
        down_w_td (TensorDescriptor): [H, I], down projection weights (is_f_by_k=True).
        int_td (TensorDescriptor): [S, I], scratch buffer for gated intermediate activations (is_f_by_k=True).
        output_td (TensorDescriptor): [S_local, H], output buffer (may be a slice for LNC sharding).
        s_base_offset (int): Row offset into the full [S, ...] tensors for this LNC core.
        dtype: Output data type (e.g. nl.bfloat16).
        TILES_IN_BLOCK_M (int): Number of M tiles per block for gate/up phase.
        TILES_IN_BLOCK_N_GU (int): Number of N tiles per block for gate/up phase.
        TILES_IN_BLOCK_K_GU (int): Number of K tiles per block for gate/up phase.
        TILES_IN_BLOCK_M_DOWN (int): Number of M tiles per block for down phase.
        TILES_IN_BLOCK_N_DOWN (int): Number of N tiles per block for down phase.
        TILES_IN_BLOCK_K_DOWN (int): Number of K tiles per block for down phase.
        save_gate_pre_td (TensorDescriptor): [S, I], optional TD to checkpoint gate pre-activation, or None.
        save_gate_act_td (TensorDescriptor): [S, I], optional TD to checkpoint SiLU(gate_pre), or None.
        save_up_td (TensorDescriptor): [S, I], optional TD to checkpoint up projection, or None.

    Returns:
        None. Results are written directly to output_td.data and int_td.data.

    Pseudocode:
        for each m_block in S tiles:
            # Phase 1: Gate/Up projection
            for each n_block in I tiles:
                gate_acc, up_acc = zeros()
                for each k_block in H tiles:
                    load x, gate_w, up_w via DGT + quantize
                    gate_acc += x @ gate_w
                    up_acc += x @ up_w
                silu_out = SiLU(gate_acc)
                intermediate = silu_out * up_acc
                store intermediate to HBM (and optional checkpoints)

            # Phase 2: Down projection
            for each n_block in H tiles:
                down_acc = zeros()
                for each k_block in I tiles:
                    load intermediate, down_w via DGT + quantize
                    down_acc += intermediate @ down_w
                store down_acc to output
    """
    sbm = get_active_sbm()
    # Derive dimensions from input TensorDescriptors
    H = hidden_td.logical_shape[0]
    S = hidden_td.logical_shape[1]
    S_local = S // NUM_LNC2_CORES if run_with_lnc2 else S
    I = gate_up_td.logical_shape[1] // NUM_GATE_UP_PROJECTIONS

    # Compute dynamic tile sizes for each phase
    gu_tiles = get_tile_sizes(H, S_local, I)
    down_tiles = get_tile_sizes(I, S_local, H)

    # Compute per-tensor load tile shapes
    hidden_load_tile_shape = _compute_load_tile_shape(hidden_td, gu_tiles, gu_tiles['tile_m'])
    gate_up_load_tile_shape = _compute_load_tile_shape(gate_up_td, gu_tiles, gu_tiles['tile_n'])
    down_load_tile_shape = _compute_load_tile_shape(down_w_td, down_tiles, down_tiles['tile_n'])

    # Unpack gate/up tile sizes
    gu_tile_m = gu_tiles['tile_m']
    gu_tile_n = gu_tiles['tile_n']
    gu_l_tile_k = gu_tiles['l_tile_k']

    NUM_S_TILES_LOCAL = div_ceil(S_local, gu_tile_m)

    BLOCK_N_GU = TILES_IN_BLOCK_N_GU * gu_tile_n

    NUM_I_TILES = div_ceil(I, gu_tile_n)
    NUM_K_TILES_GU = div_ceil(H, gu_l_tile_k)
    NUM_M_BLOCKS = div_ceil(NUM_S_TILES_LOCAL, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS_GU = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_N_GU)
    NUM_K_BLOCKS_GU = div_ceil(NUM_K_TILES_GU, TILES_IN_BLOCK_K_GU)

    bd_gu = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N_GU,
        TILES_IN_BLOCK_K_GU,
        lhs_load_tile_shape=hidden_load_tile_shape,
        rhs_load_tile_shape=gate_up_load_tile_shape,
        tiles=gu_tiles,
    )

    # s_base_offset is the logical offset. Need physical
    lhs_m_offset = (
        s_base_offset * INTERLEAVE_FACTOR if (hidden_td.is_swizzled and not hidden_td.is_quantized) else s_base_offset
    )
    rhs_n_offset_up = I * INTERLEAVE_FACTOR if (gate_up_td.is_swizzled and not gate_up_td.is_quantized) else I

    # Phase 1 spill/reload buffers
    hiddenq_td = None
    gate_wq_td = None
    up_wq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm

        if not hidden_td.is_quantized:
            hiddenq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS_GU,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd_gu.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K_GU,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
        if not gate_up_td.is_quantized:
            gate_wq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS_GU,
                num_f_blocks=NUM_N_BLOCKS_GU,
                block_f_logical=bd_gu.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K_GU,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
            up_wq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS_GU,
                num_f_blocks=NUM_N_BLOCKS_GU,
                block_f_logical=bd_gu.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K_GU,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    for m_block_idx in nl.sequential_range(NUM_M_BLOCKS):
        m_block_start = m_block_idx * TILES_IN_BLOCK_M

        # Phase 1: Gate/up -> SiLU(gate) * up -> write to HBM
        for n_block_idx in range(NUM_N_BLOCKS_GU):
            # 2D SBUF accumulators for gate and up
            # Layout: (TILE_M, TILES_IN_BLOCK_M * BLOCK_N_GU), AP: [[..., TILE_M], [1, TILE_N]]
            acc_cols = TILES_IN_BLOCK_M * BLOCK_N_GU
            gate_sbuf = sbm.alloc_stack(shape=(gu_tile_m, acc_cols), dtype=nl.float32, buffer=nl.sbuf)
            up_sbuf = sbm.alloc_stack(shape=(gu_tile_m, acc_cols), dtype=nl.float32, buffer=nl.sbuf)

            gate_output_td = TensorDescriptor(data=gate_sbuf)
            up_output_td = TensorDescriptor(data=up_sbuf)

            for k_block_idx in nl.sequential_range(NUM_K_BLOCKS_GU):
                # Empty TD — gate call fills it, up call reuses it
                hidden_sbuf_td = TensorDescriptor(is_quantized=True)

                # Gate matmul — loads hidden, fills hidden_sbuf_td
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=hidden_td,
                    rhs_hbm_td=gate_up_td,
                    bd=bd_gu,
                    output_td=gate_output_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_sbuf_td=hidden_sbuf_td,
                    lhs_m_offset=lhs_m_offset,
                    rhs_n_offset=0,
                    TILES_IN_LOAD_M=min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M),
                    TILES_IN_LOAD_N=1,
                    lhs_matmul_tile_shape_physical=gu_tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=gu_tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=hidden_load_tile_shape or gu_tiles['lhs_load_tile'],
                    rhs_load_tile_shape=gate_up_load_tile_shape or gu_tiles['rhs_load_tile'],
                    lhs_quantize_tile_shape=gu_tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=gu_tiles['rhs_quantize_tile'],
                    spill_reload=spill_reload,
                    lhsq_td=hiddenq_td,
                    rhsq_td=gate_wq_td,
                    use_scale_packing=use_scale_packing,
                    initialize_accumulator=(k_block_idx == 0),
                )

                # Up matmul — reuses hidden from hidden_sbuf_td
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=hidden_td,
                    rhs_hbm_td=gate_up_td,
                    bd=bd_gu,
                    output_td=up_output_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_sbuf_td=hidden_sbuf_td,
                    lhs_m_offset=lhs_m_offset,
                    rhs_n_offset=rhs_n_offset_up,
                    TILES_IN_LOAD_M=min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M),
                    TILES_IN_LOAD_N=1,
                    lhs_matmul_tile_shape_physical=gu_tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=gu_tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=hidden_load_tile_shape or gu_tiles['lhs_load_tile'],
                    rhs_load_tile_shape=gate_up_load_tile_shape or gu_tiles['rhs_load_tile'],
                    lhs_quantize_tile_shape=gu_tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=gu_tiles['rhs_quantize_tile'],
                    spill_reload=spill_reload,
                    lhsq_td=hiddenq_td,
                    rhsq_td=up_wq_td,
                    use_scale_packing=use_scale_packing,
                    initialize_accumulator=(k_block_idx == 0),
                )

            # SiLU(gate) * up -> write to HBM
            sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N_GU
            n_block_start = n_block_idx * TILES_IN_BLOCK_N_GU
            num_m_tiles_in_block = min(TILES_IN_BLOCK_M, div_ceil(S_local - m_block_start * gu_tile_m, gu_tile_m))
            num_n_tiles_in_block = min(TILES_IN_BLOCK_N_GU, div_ceil(I - n_block_start * gu_tile_n, gu_tile_n))
            for m_tile_idx in range(num_m_tiles_in_block):
                for n_tile_idx in range(num_n_tiles_in_block):
                    m_off = m_block_start * gu_tile_m + m_tile_idx * gu_tile_m
                    i_off = (n_block_start + n_tile_idx) * gu_tile_n
                    actual_m = min(gu_tile_m, S_local - m_off)
                    actual_n = min(gu_tile_n, I - i_off)
                    sbuf_offset = m_tile_idx * BLOCK_N_GU + n_tile_idx * gu_tile_n

                    gate_tile = gate_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)
                    up_tile = up_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)

                    if save_gate_pre_td != None:
                        nisa.dma_copy(
                            dst=save_gate_pre_td.data[
                                s_base_offset + m_off : s_base_offset + m_off + actual_m, i_off : i_off + actual_n
                            ],
                            src=gate_tile,
                        )

                    silu_out = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=silu_out, op=nl.silu, data=gate_tile)

                    if save_gate_act_td != None:
                        nisa.dma_copy(
                            dst=save_gate_act_td.data[
                                s_base_offset + m_off : s_base_offset + m_off + actual_m, i_off : i_off + actual_n
                            ],
                            src=silu_out,
                        )

                    if save_up_td != None:
                        nisa.dma_copy(
                            dst=save_up_td.data[
                                s_base_offset + m_off : s_base_offset + m_off + actual_m, i_off : i_off + actual_n
                            ],
                            src=up_tile,
                        )

                    result = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=result, data1=silu_out, data2=up_tile, op=nl.multiply)

                    nisa.dma_copy(
                        dst=int_td.data[
                            s_base_offset + m_off : s_base_offset + m_off + actual_m, i_off : i_off + actual_n
                        ],
                        src=result,
                    )

    # Phase 2: Down projection — single API call handles full M×N×K
    NUM_M_BLOCKS_DOWN = div_ceil(NUM_S_TILES_LOCAL, TILES_IN_BLOCK_M_DOWN)

    # Unpack down tile sizes
    down_tile_n = down_tiles['tile_n']

    bd_down = _build_matmul_params(
        TILES_IN_BLOCK_M_DOWN,
        TILES_IN_BLOCK_N_DOWN,
        TILES_IN_BLOCK_K_DOWN,
        rhs_load_tile_shape=down_load_tile_shape,
        tiles=down_tiles,
    )

    # Spill/reload buffer allocation for Phase 2
    intq_td = None
    downq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm
        NUM_K_TILES_DOWN = div_ceil(int_td.physical_shape[0], down_tiles['matmul_tile_k_physical'])
        NUM_K_BLOCKS_DOWN = div_ceil(NUM_K_TILES_DOWN, TILES_IN_BLOCK_K_DOWN)
        NUM_N_BLOCKS_DOWN = div_ceil(H, TILES_IN_BLOCK_N_DOWN * down_tile_n)

        intq_td = _allocate_spill_buffer(
            num_k_blocks=NUM_K_BLOCKS_DOWN,
            num_f_blocks=NUM_M_BLOCKS_DOWN,
            block_f_logical=bd_down.BLOCK_M_LOGICAL,
            tiles_in_block_k=TILES_IN_BLOCK_K_DOWN,
            use_scale_packing=use_scale_packing,
            data_buffer=data_buffer,
        )

        if not down_w_td.is_quantized:
            downq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS_DOWN,
                num_f_blocks=NUM_N_BLOCKS_DOWN,
                block_f_logical=bd_down.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K_DOWN,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    generic_matmul_mxfp8_api(
        lhs_hbm_td=int_td,
        rhs_hbm_td=down_w_td,
        bd=bd_down,
        output_td=output_td,
        block_idx_m=(0, NUM_M_BLOCKS_DOWN),
        lhs_m_offset=s_base_offset,
        TILES_IN_LOAD_M=min(TILES_IN_BLOCK_M_DOWN, MAX_TILES_IN_LOAD_M),
        TILES_IN_LOAD_N=1,
        lhs_matmul_tile_shape_physical=down_tiles['lhs_matmul_tile_physical'],
        rhs_matmul_tile_shape_physical=down_tiles['rhs_matmul_tile_physical'],
        lhs_load_tile_shape=down_tiles['lhs_load_tile'],
        rhs_load_tile_shape=down_load_tile_shape or down_tiles['rhs_load_tile'],
        lhs_quantize_tile_shape=down_tiles['lhs_quantize_tile'],
        rhs_quantize_tile_shape=down_tiles['rhs_quantize_tile'],
        spill_reload=spill_reload,
        lhsq_td=intq_td,
        rhsq_td=downq_td,
        use_scale_packing=use_scale_packing,
    )


@with_active_sbm
def mlp_forward_mxfp8_nki(
    hidden: nl.ndarray,
    gate_up_weights: nl.ndarray,
    down_weights: nl.ndarray,
    intermediate_hbm: nl.ndarray,
    run_with_lnc2: bool = True,
    gate_up_tiles_m: int = 8,
    gate_up_tiles_n: int = 1,
    gate_up_tiles_k: int = 8,
    down_tiles_m: int = 8,
    down_tiles_n: int = 1,
    down_tiles_k: int = 8,
    fp8_x4_dtype=float8_e4m3fn_x4,
    save_gate_pre: nl.ndarray = None,
    save_gate_act: nl.ndarray = None,
    save_up: nl.ndarray = None,
    save_hidden: nl.ndarray = None,
    dtype=nl.bfloat16,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    hidden_scales: nl.ndarray = None,
    gate_up_scales: nl.ndarray = None,
    down_scales: nl.ndarray = None,
    hidden_is_swizzled: bool = False,
    gate_up_is_swizzled: bool = False,
    down_is_swizzled: bool = False,
) -> nl.ndarray:
    """MXFP8 SwiGLU MLP forward pass with optional activation checkpointing.

    Computes the SwiGLU MLP forward pass using MXFP8 quantized matmuls.
    TODO: Specify intended usage range (e.g., sequence length, hidden size constraints).

    SwiGLU forward:
        gate_pre     = hidden @ W_gate.T                  (gate linear projection)
        gate_act     = SiLU(gate_pre)                     (gate activation)
        up           = hidden @ W_up.T                    (up linear projection)
        intermediate = gate_act * up                      (element-wise gating)
        output       = intermediate @ W_down.T            (down projection)

    Dimensions:
        S: Sequence length (number of tokens).
        H: Hidden dimension size.
        I: Intermediate dimension size (per gate/up projection).

    Args:
        hidden (nl.ndarray): [S, H], input hidden states.
        gate_up_weights (nl.ndarray): [2I, H], fused weight matrix — rows [0:I] = W_gate, rows [I:2I] = W_up.
        down_weights (nl.ndarray): [H, I], down projection weights (W_down).
        intermediate_hbm (nl.ndarray): [S, I], scratch buffer for gated intermediate activations.
        run_with_lnc2 (bool): Whether to shard across 2 LNC cores.
        gate_up_tiles_m (int): Number of M tiles per block for gate/up phase.
        gate_up_tiles_n (int): Number of N tiles per block for gate/up phase.
        gate_up_tiles_k (int): Number of K tiles per block for gate/up phase.
        down_tiles_m (int): Number of M tiles per block for down phase.
        down_tiles_n (int): Number of N tiles per block for down phase.
        down_tiles_k (int): Number of K tiles per block for down phase.
        fp8_x4_dtype: MXFP8 quantized data type for nc_matmul_mx.
        save_gate_pre (nl.ndarray): [S, I], HBM buffer to checkpoint gate pre-activation, or None.
        save_gate_act (nl.ndarray): [S, I], HBM buffer to checkpoint SiLU(gate_pre), or None.
        save_up (nl.ndarray): [S, I], HBM buffer to checkpoint up projection, or None.
        save_hidden (nl.ndarray): [S, I], HBM buffer to checkpoint gate_act * up, or None
            (same data as intermediate_hbm but kept as a separate named output
            for clarity in the fwd/bwd contract).
        dtype: Output data type (e.g. nl.bfloat16).
        spill_reload (bool): Whether to spill quantized operands to HBM for reload across K-blocks.
        use_scale_packing (bool): Whether to pack MXFP8 scales into compact format.
        hidden_scales (nl.ndarray): MXFP8 scales for pre-quantized hidden, or None (raw BF16).
        gate_up_scales (nl.ndarray): MXFP8 scales for pre-quantized gate_up_weights, or None.
        down_scales (nl.ndarray): MXFP8 scales for pre-quantized down_weights, or None.
        hidden_is_swizzled (bool): True if hidden is pre-swizzled [K/4, F*4] BF16.
        gate_up_is_swizzled (bool): True if gate_up_weights is pre-swizzled.
        down_is_swizzled (bool): True if down_weights is pre-swizzled.

    Returns:
        output (nl.ndarray): [S, H], MLP output hidden states.

    Pseudocode:
        gate_pre     = hidden @ W_gate.T
        gate_act     = SiLU(gate_pre)
        up           = hidden @ W_up.T
        intermediate = gate_act * up
        output       = intermediate @ W_down.T
    """
    if get_active_sbm() == None:
        create_and_set_active_sbm()

    sbm = get_active_sbm()
    sbm.open_scope(name="MXFP8 MLP FWD ")

    # save_hidden optimization: write directly to save_hidden when provided
    # to avoid double copy through intermediate_hbm.
    effective_intermediate = save_hidden if save_hidden != None else intermediate_hbm

    # Build TensorDescriptors for all tensors
    hidden_td = TensorDescriptor(
        data=hidden, scales=hidden_scales, is_swizzled=hidden_is_swizzled, is_col_parallel_sharded=run_with_lnc2
    )
    gate_up_td = TensorDescriptor(
        data=gate_up_weights, scales=gate_up_scales, is_swizzled=gate_up_is_swizzled, is_col_parallel_sharded=True
    )
    down_w_td = TensorDescriptor(data=down_weights, scales=down_scales, is_swizzled=down_is_swizzled)
    int_td = TensorDescriptor(data=effective_intermediate, is_f_by_k=True, is_col_parallel_sharded=run_with_lnc2)

    # Derive dimensions from logical shapes (works for all input modes)
    H = hidden_td.logical_shape[0]  # K dimension of hidden = hidden size
    TWO_I = gate_up_td.logical_shape[1]  # F dimension of gate_up = 2*I
    I = TWO_I // NUM_GATE_UP_PROJECTIONS
    S = hidden_td.logical_shape[1]  # F dimension of hidden = sequence length

    kernel_assert(
        TWO_I == NUM_GATE_UP_PROJECTIONS * I,
        f"gate_up_weights dim0 ({TWO_I}) must be {NUM_GATE_UP_PROJECTIONS}*I ({NUM_GATE_UP_PROJECTIONS * I})",
    )
    if run_with_lnc2:
        kernel_assert(S % NUM_LNC2_CORES == 0, f"S ({S}) must be even for LNC2")

    # DGT requires K dimension divisible by DGT_MIN_K for unswizzled BF16 inputs
    kernel_assert(
        hidden_td.is_quantized or hidden_td.is_swizzled or H % DGT_MIN_K == 0,
        f"H ({H}) must be divisible by {DGT_MIN_K} for DGT when hidden is unswizzled BF16",
    )
    kernel_assert(
        gate_up_td.is_quantized or gate_up_td.is_swizzled or H % DGT_MIN_K == 0,
        f"H ({H}) must be divisible by {DGT_MIN_K} for DGT when gate_up_weights is unswizzled BF16",
    )
    kernel_assert(
        down_w_td.is_quantized or down_w_td.is_swizzled or I % DGT_MIN_K == 0,
        f"I ({I}) must be divisible by {DGT_MIN_K} for DGT when down_weights is unswizzled BF16",
    )

    # LNC2 sharding along M (sequence) dimension
    S_local = S // NUM_LNC2_CORES if run_with_lnc2 else S
    output = nl.ndarray((S, H), dtype=dtype, buffer=nl.shared_hbm if run_with_lnc2 else nl.hbm)
    if run_with_lnc2:
        LNC_ID = nl.program_id(axis=0)
        s_base_offset = LNC_ID * S_local
        output_local = output[s_base_offset : s_base_offset + S_local, :]
    else:
        s_base_offset = 0
        output_local = output

    output_td = TensorDescriptor(data=output_local)
    save_gate_pre_td = TensorDescriptor(data=save_gate_pre) if save_gate_pre != None else None
    save_gate_act_td = TensorDescriptor(data=save_gate_act) if save_gate_act != None else None
    save_up_td = TensorDescriptor(data=save_up) if save_up != None else None

    compute_fused_gate_up_down_mxfp8(
        hidden_td=hidden_td,
        gate_up_td=gate_up_td,
        down_w_td=down_w_td,
        int_td=int_td,
        output_td=output_td,
        s_base_offset=s_base_offset,
        dtype=dtype,
        TILES_IN_BLOCK_M=gate_up_tiles_m,
        TILES_IN_BLOCK_N_GU=gate_up_tiles_n,
        TILES_IN_BLOCK_K_GU=gate_up_tiles_k,
        TILES_IN_BLOCK_M_DOWN=down_tiles_m,
        TILES_IN_BLOCK_N_DOWN=down_tiles_n,
        TILES_IN_BLOCK_K_DOWN=down_tiles_k,
        save_gate_pre_td=save_gate_pre_td,
        save_gate_act_td=save_gate_act_td,
        save_up_td=save_up_td,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    sbm.close_scope()

    return output
