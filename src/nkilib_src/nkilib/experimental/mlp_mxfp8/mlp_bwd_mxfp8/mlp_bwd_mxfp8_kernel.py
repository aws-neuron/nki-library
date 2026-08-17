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

"""MXFP8 SwiGLU MLP backward kernels with activation checkpointing and recompute support."""

import nki.isa as nisa
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ...matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from ...matmul_mxfp8.matmul_mxfp8_generic_api import generic_matmul_mxfp8_api
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor
from ...mxfp_utils.mxfp8_utils.common_utils import create_and_set_active_sbm, get_active_sbm, with_active_sbm
from ...mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import INTERLEAVE_FACTOR
from ..common_utils import (
    DGT_MIN_K,
    NUM_LNC2_CORES,
    _allocate_spill_buffer,
    _build_matmul_params,
    _compute_load_tile_shape,
    apply_gradient_clamp,
    hbm_dma_transpose,
)
from .config import ClampLimits, MlpBwdMatmulConfig
from .recompute import recompute_gate_act, recompute_gate_up_projection, recompute_intermediate


def get_program_sharding_info(run_with_lnc2: bool) -> tuple:
    """Return (num_cores, shard_id) for LNC2 sharding."""
    if run_with_lnc2:
        return NUM_LNC2_CORES, nl.program_id(axis=0)
    return 1, 0


def compute_phase1_down_proj_mm_grad_mxfp8(
    output_grad_td: TensorDescriptor,
    gate_pre_td: TensorDescriptor,
    gate_act_td: TensorDescriptor,
    up_td: TensorDescriptor,
    d_gate_up_td: TensorDescriptor,
    scratch_td: TensorDescriptor,
    down_weight_td: TensorDescriptor,
    s_base: int,
    dtype: type,
    fp8_x4_dtype: type,
    config: MatmulMxfp8KernelConfig = None,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
    clamp_limits: ClampLimits = None,
) -> None:
    """Phase 1: Compute gradient through the down projection and SwiGLU gate.

    Step A — matmul: backprop through down projection
        d_intermediate = output_grad @ W_down_T.T       [S, I]

    Step B — SwiGLU backward: split d_intermediate into gate and up gradients
        Using checkpointed forward activations:
          gate_pre:  [S, I]  gate pre-activation  (hidden_states @ W_gate.T, before SiLU)
          gate_act:  [S, I]  gate post-activation (SiLU(gate_pre))
          up:        [S, I]  up projection        (hidden_states @ W_up.T)

        silu_dx       = SiLU'(gate_pre)                  — SiLU derivative
        d_gate_act    = silu_dx * up                      — chain rule through gating
        d_gate        = d_intermediate * d_gate_act       — gradient for gate path
        d_up          = d_intermediate * gate_act         — gradient for up path

    Also transposes d_gate/d_up into scratch for phase 3 weight grad computation.

    Args:
        output_grad_td (TensorDescriptor): [S, H], incoming gradient (is_f_by_k=True).
        gate_pre_td (TensorDescriptor): [S, I], checkpointed gate pre-activation.
        gate_act_td (TensorDescriptor): [S, I], checkpointed gate post-activation.
        up_td (TensorDescriptor): [S, I], checkpointed up projection.
        d_gate_up_td (TensorDescriptor): [S, 2I], output: combined gate/up gradient (is_f_by_k=True).
        scratch_td (TensorDescriptor): [2I, S], output: transposed d_gate || d_up.
        down_weight_td (TensorDescriptor): [I, H], transposed down projection weights (is_f_by_k=True).
        s_base (int): Row offset into the full [S, ...] tensors for this LNC core.
        dtype: Data type for computation (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized data type (e.g. float8_e4m3fn_x4).

    Returns:
        None. Results are written to d_gate_up_td.data and scratch_td.data.

    Pseudocode:
        for each m_block in S tiles:
            for each n_block in I tiles:
                acc = zeros()
                for each k_block in H tiles:
                    load output_grad, down_weight via DGT + quantize
                    acc += output_grad @ down_weight
                # SwiGLU backward
                silu_dx = SiLU'(gate_pre)
                d_gate = acc * (silu_dx * up)
                d_up = acc * gate_act
                store d_gate_up to HBM
                transpose d_gate, d_up into scratch
    """
    sbm = get_active_sbm()

    # Phase 1: output_grad[S,H] @ down_weight[I,H].T -> [S,I]
    # K=H, M=S_local, N=I
    H = output_grad_td.logical_shape[0]
    I = down_weight_td.logical_shape[1]
    S_local = output_grad_td.sharded_logical_shape[1]

    # Extract tile sizes from resolved config
    tile_m = config.tile_m
    tile_n = config.tile_n
    l_tile_k = config.tile_k
    TILES_IN_BLOCK_M = config.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = config.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = config.TILES_IN_BLOCK_K

    # Build tiles dict for _compute_load_tile_shape compatibility
    matmul_tile_k_physical = l_tile_k // INTERLEAVE_FACTOR
    tiles = {
        'tile_m': tile_m,
        'tile_n': tile_n,
        'l_tile_k': l_tile_k,
        'matmul_tile_k_physical': matmul_tile_k_physical,
        'lhs_matmul_tile_physical': config.lhs_matmul_tile_shape_physical,
        'rhs_matmul_tile_physical': config.rhs_matmul_tile_shape_physical,
        'lhs_load_tile': config.lhs_load_tile_shape,
        'rhs_load_tile': config.rhs_load_tile_shape,
        'lhs_quantize_tile': config.lhs_quantize_tile_shape,
        'rhs_quantize_tile': config.rhs_quantize_tile_shape,
    }

    NUM_S_TILES_LOCAL = div_ceil(S_local, tile_m)
    BLOCK_N = TILES_IN_BLOCK_N * tile_n
    NUM_K_TILES = div_ceil(H, l_tile_k)
    NUM_I_TILES = div_ceil(I, tile_n)
    NUM_M_BLOCKS = div_ceil(NUM_S_TILES_LOCAL, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    # Compute load tile shapes from TD state
    lhs_load_tile_shape = _compute_load_tile_shape(output_grad_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(down_weight_td, tiles, tile_n)

    # Convert s_base to physical offset for pre-swizzled inputs
    s_base_physical = (
        s_base * INTERLEAVE_FACTOR if (output_grad_td.is_swizzled and not output_grad_td.is_quantized) else s_base
    )

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffers (skip for pre-quantized inputs)
    output_gradq_td = None
    down_weightq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm

        if not output_grad_td.is_quantized:
            output_gradq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
        if not down_weight_td.is_quantized:
            down_weightq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N

    for m_block_idx in nl.sequential_range(NUM_M_BLOCKS):
        m_block_start = m_block_idx * TILES_IN_BLOCK_M

        for n_block_idx in range(NUM_N_BLOCKS):
            n_block_start = n_block_idx * TILES_IN_BLOCK_N

            # SBUF accumulator for matmul result — same layout as generic API produces
            acc_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
            acc_td = TensorDescriptor(data=acc_sbuf)

            for k_block_idx in nl.sequential_range(NUM_K_BLOCKS):
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=output_grad_td,
                    rhs_hbm_td=down_weight_td,
                    bd=bd,
                    output_td=acc_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_m_offset=s_base_physical,
                    TILES_IN_LOAD_M=config.TILES_IN_LOAD_M,
                    TILES_IN_LOAD_N=config.TILES_IN_LOAD_N,
                    lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=lhs_load_tile_shape,
                    rhs_load_tile_shape=rhs_load_tile_shape,
                    lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                    initialize_accumulator=(k_block_idx == 0),
                    spill_reload=spill_reload,
                    lhsq_td=output_gradq_td,
                    rhsq_td=down_weightq_td,
                    use_scale_packing=use_scale_packing,
                )

            # SwiGLU backward: load checkpoints, compute gradients, store to HBM + scratch
            num_m_tiles_in_block = min(TILES_IN_BLOCK_M, div_ceil(S_local - m_block_start * tile_m, tile_m))
            num_n_tiles_in_block = min(TILES_IN_BLOCK_N, div_ceil(I - n_block_start * tile_n, tile_n))
            for tile_m_idx in nl.affine_range(num_m_tiles_in_block):
                m_off = m_block_start * tile_m + tile_m_idx * tile_m
                global_s = s_base + m_off
                actual_m = min(tile_m, S_local - m_off)

                for tile_n_idx in nl.affine_range(num_n_tiles_in_block):
                    i_off = (n_block_start + tile_n_idx) * tile_n
                    actual_n = min(tile_n, I - i_off)

                    # Load forward activation checkpoints
                    up_checkpoint = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    gate_act_checkpoint = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    gate_pre_checkpoint = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=up_checkpoint, src=up_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n]
                    )
                    nisa.dma_copy(
                        dst=gate_act_checkpoint,
                        src=gate_act_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                    )
                    nisa.dma_copy(
                        dst=gate_pre_checkpoint,
                        src=gate_pre_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                    )

                    # SiLU derivative and d_gate_act = silu_dx * up
                    silu_dx = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.activation(dst=silu_dx, op=nl.silu_dx, data=gate_pre_checkpoint, bias=None, scale=1.0)

                    d_gate_act = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=d_gate_act, data1=silu_dx, data2=up_checkpoint, op=nl.multiply, engine=nisa.vector_engine
                    )

                    # Compute d_gate and d_up from matmul accumulator
                    sbuf_offset = tile_m_idx * BLOCK_N + tile_n_idx * tile_n
                    acc_tile = acc_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)

                    d_gate = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=d_gate, data1=acc_tile, data2=d_gate_act, op=nl.multiply)

                    d_up = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=d_up, data1=acc_tile, data2=gate_act_checkpoint, op=nl.multiply)

                    # Apply gradient clamping (no-op when clamp_limits is None or all limits are None)
                    if clamp_limits is not None:
                        apply_gradient_clamp(
                            d_gate,
                            gate_pre_checkpoint,
                            clamp_limits.non_linear_clamp_upper_limit,
                            clamp_limits.non_linear_clamp_lower_limit,
                            dtype,
                        )
                        apply_gradient_clamp(
                            d_up,
                            up_checkpoint,
                            clamp_limits.linear_clamp_upper_limit,
                            clamp_limits.linear_clamp_lower_limit,
                            dtype,
                        )

                    # Store to combined [S, 2I] buffer: d_gate in [0:I], d_up in [I:2I]
                    nisa.dma_copy(
                        dst=d_gate_up_td.data[global_s : global_s + actual_m, i_off : i_off + actual_n],
                        src=d_gate,
                    )
                    nisa.dma_copy(
                        dst=d_gate_up_td.data[global_s : global_s + actual_m, I + i_off : I + i_off + actual_n],
                        src=d_up,
                    )

    # Transpose d_gate_up [S_local, 2I] -> scratch [2I, S_local] for phase 3
    hbm_dma_transpose(
        d_gate_up_td.data,
        scratch_td.data,
        M=S_local,
        N=2 * I,
        src_row_offset=s_base,
        dst_col_offset=s_base,
    )


def compute_phase2_hidden_states_grad_mxfp8(
    hidden_states_grad_td: TensorDescriptor,
    gate_up_weight_td: TensorDescriptor,
    d_gate_up_td: TensorDescriptor,
    s_base: int,
    dtype: type,
    fp8_x4_dtype: type,
    config: MatmulMxfp8KernelConfig = None,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
) -> None:
    """Phase 2: Compute gradient w.r.t. input hidden states.

    Computes:
        hidden_states_grad[s_base:s_base+S_local, :] = d_gate_up[S, 2I] @ gate_up_weight_T[H, 2I].T

    This is equivalent to d_gate @ W_gate + d_up @ W_up since concatenation along K
    distributes through matmul.

    Args:
        hidden_states_grad_td (TensorDescriptor): [S, H], output: dL/d_hidden.
        gate_up_weight_td (TensorDescriptor): [H, 2I], transposed combined gate+up weights.
        d_gate_up_td (TensorDescriptor): [S, 2I], combined gate/up gradient (is_f_by_k=True).
        s_base (int): Row offset for this LNC core's shard.
        dtype: Data type for computation (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized data type (e.g. float8_e4m3fn_x4).
        config: MatmulMxfp8KernelConfig with K=2I.

    Returns:
        None. Results are written to hidden_states_grad_td.data[s_base:s_base+S_local, :].
    """
    S_local = d_gate_up_td.sharded_logical_shape[1]
    H = gate_up_weight_td.logical_shape[1]

    tile_m = config.tile_m
    tile_n = config.tile_n
    l_tile_k = config.tile_k
    TILES_IN_BLOCK_M = config.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = config.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = config.TILES_IN_BLOCK_K

    matmul_tile_k_physical = l_tile_k // INTERLEAVE_FACTOR
    tiles = {
        'tile_m': tile_m,
        'tile_n': tile_n,
        'l_tile_k': l_tile_k,
        'matmul_tile_k_physical': matmul_tile_k_physical,
        'lhs_matmul_tile_physical': config.lhs_matmul_tile_shape_physical,
        'rhs_matmul_tile_physical': config.rhs_matmul_tile_shape_physical,
        'lhs_load_tile': config.lhs_load_tile_shape,
        'rhs_load_tile': config.rhs_load_tile_shape,
        'lhs_quantize_tile': config.lhs_quantize_tile_shape,
        'rhs_quantize_tile': config.rhs_quantize_tile_shape,
    }

    rhs_load_tile_shape = _compute_load_tile_shape(gate_up_weight_td, tiles, tile_n)
    lhs_load_tile_shape = _compute_load_tile_shape(d_gate_up_td, tiles, tile_m)

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffers
    NUM_K_TILES = div_ceil(d_gate_up_td.logical_shape[0], l_tile_k)
    NUM_M_BLOCKS = div_ceil(div_ceil(S_local, tile_m), TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(div_ceil(H, tile_n), TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    d_gate_upq_td = None
    gate_up_weightq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm
        d_gate_upq_td = _allocate_spill_buffer(
            num_k_blocks=NUM_K_BLOCKS,
            num_f_blocks=NUM_M_BLOCKS,
            block_f_logical=bd.BLOCK_M_LOGICAL,
            tiles_in_block_k=TILES_IN_BLOCK_K,
            use_scale_packing=use_scale_packing,
            data_buffer=data_buffer,
        )
        if not gate_up_weight_td.is_quantized:
            gate_up_weightq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    hidden_grad_local = hidden_states_grad_td.data[s_base : s_base + S_local, :]
    output_td = TensorDescriptor(data=hidden_grad_local)

    generic_matmul_mxfp8_api(
        lhs_hbm_td=d_gate_up_td,
        rhs_hbm_td=gate_up_weight_td,
        bd=bd,
        output_td=output_td,
        lhs_m_offset=s_base,
        TILES_IN_LOAD_M=config.TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=config.TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
        rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
        rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
        spill_reload=spill_reload,
        lhsq_td=d_gate_upq_td,
        rhsq_td=gate_up_weightq_td,
        use_scale_packing=use_scale_packing,
    )


def compute_phase3_gate_up_weight_grad_mxfp8(
    weight_grad_td: TensorDescriptor,
    hidden_states_T_td: TensorDescriptor,
    grad_T_td: TensorDescriptor,
    dtype: type,
    fp8_x4_dtype: type,
    config: MatmulMxfp8KernelConfig = None,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
) -> None:
    """Phase 3: Compute gradient w.r.t. gate and up weight matrices as a single matmul.

    Computes:
        [dW_gate; dW_up] = grad_T[2I, S] @ hidden_states[S, H] -> [2I, H]

    Uses pre-transposed inputs:
        grad_T_td: [2I, S]  transposed [d_gate || d_up] from phase 1
            rows [0:I]  = d_gate.T
            rows [I:2I] = d_up.T
        hidden_states_T_td:  [H, S]   transposed input hidden states

    LNC2 sharding: core 0 computes rows [0:I] (gate grad),
                   core 1 computes rows [I:2I] (up grad).

    Dimensions:
        S: Sequence length.
        H: Hidden dimension size.
        I: Intermediate dimension size.

    Args:
        weight_grad_td (TensorDescriptor): [2I, H], output: [dW_gate; dW_up].
        hidden_states_T_td (TensorDescriptor): [H, S], transposed input hidden states (is_f_by_k=True).
        grad_T_td (TensorDescriptor): [2I, S], transposed gate+up gradients
            (is_f_by_k=True, is_col_parallel_sharded=True for LNC2).
        dtype: Data type for computation (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized data type.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles to accumulate in PSUM.

    Returns:
        None. Results are written to weight_grad_td.data.

    Pseudocode:
        # LNC2: core 0 handles rows [0:I], core 1 handles rows [I:2I]
        weight_grad_local = weight_grad[i_base : i_base + I_local, :]
        weight_grad_local = grad_T[i_base : i_base + I_local, :] @ hidden_states_T.T
    """
    # Derive dimensions and LNC shard offset
    S = hidden_states_T_td.logical_shape[0]
    H = hidden_states_T_td.logical_shape[1]
    I = grad_T_td.logical_shape[1] // 2
    _, shard_id = get_program_sharding_info(run_with_lnc2)
    i_base = shard_id * I if run_with_lnc2 else 0

    # M_LOCAL = I per core (2I / 2), or 2I without LNC2
    I_local = I if run_with_lnc2 else 2 * I

    tile_m = config.tile_m
    tile_n = config.tile_n
    l_tile_k = config.tile_k
    TILES_IN_BLOCK_M = config.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = config.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = config.TILES_IN_BLOCK_K

    matmul_tile_k_physical = l_tile_k // INTERLEAVE_FACTOR
    tiles = {
        'tile_m': tile_m,
        'tile_n': tile_n,
        'l_tile_k': l_tile_k,
        'matmul_tile_k_physical': matmul_tile_k_physical,
        'lhs_matmul_tile_physical': config.lhs_matmul_tile_shape_physical,
        'rhs_matmul_tile_physical': config.rhs_matmul_tile_shape_physical,
        'lhs_load_tile': config.lhs_load_tile_shape,
        'rhs_load_tile': config.rhs_load_tile_shape,
        'lhs_quantize_tile': config.lhs_quantize_tile_shape,
        'rhs_quantize_tile': config.rhs_quantize_tile_shape,
    }

    NUM_I_TILES_LOCAL = div_ceil(I_local, tile_m)
    NUM_K_TILES = div_ceil(S, l_tile_k)
    NUM_H_TILES = div_ceil(H, tile_n)
    NUM_M_BLOCKS = div_ceil(NUM_I_TILES_LOCAL, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_H_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    # RHS load tile shape from TD state
    rhs_load_tile_shape = _compute_load_tile_shape(hidden_states_T_td, tiles, tile_n)
    lhs_load_tile_shape = _compute_load_tile_shape(grad_T_td, tiles, tile_m)

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffers — single LHS buffer, single RHS buffer
    grad_Tq_td = None
    hidden_states_Tq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm

        grad_Tq_td = _allocate_spill_buffer(
            num_k_blocks=NUM_K_BLOCKS,
            num_f_blocks=NUM_M_BLOCKS,
            block_f_logical=bd.BLOCK_M_LOGICAL,
            tiles_in_block_k=TILES_IN_BLOCK_K,
            use_scale_packing=use_scale_packing,
            data_buffer=data_buffer,
        )
        if not hidden_states_T_td.is_quantized:
            hidden_states_Tq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    # Slice output for this core's shard and let the generic API handle all loops + HBM store
    weight_grad_local = weight_grad_td.data[i_base : i_base + I_local, :]
    output_td = TensorDescriptor(data=weight_grad_local)

    generic_matmul_mxfp8_api(
        lhs_hbm_td=grad_T_td,
        rhs_hbm_td=hidden_states_T_td,
        bd=bd,
        output_td=output_td,
        lhs_m_offset=i_base,
        TILES_IN_LOAD_M=config.TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=config.TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
        rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
        lhs_load_tile_shape=tiles['lhs_load_tile'],
        rhs_load_tile_shape=rhs_load_tile_shape,
        lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
        rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
        spill_reload=spill_reload,
        lhsq_td=grad_Tq_td,
        rhsq_td=hidden_states_Tq_td,
        use_scale_packing=use_scale_packing,
    )


def compute_phase4_down_weight_grad_mxfp8(
    down_weight_grad_td: TensorDescriptor,
    output_grad_T_td: TensorDescriptor,
    intermediate_T_td: TensorDescriptor,
    h_base: int,
    dtype: type,
    fp8_x4_dtype: type,
    config: MatmulMxfp8KernelConfig = None,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
) -> None:
    """Phase 4: Compute gradient w.r.t. down projection weight matrix.

    Computes:
        dW_down = output_grad.T @ intermediate     [H, I]

    Uses pre-transposed inputs:
        output_grad_T_td:  [H, S]  transposed incoming gradient
        intermediate_T_td: [I, S]  transposed gated intermediate activations

    H-sharded across LNC cores (each core computes a slice of H rows).

    Dimensions derived from tensor descriptors:
        output_grad_T_td.logical_shape = (S, H)  — K=S, F=H
        intermediate_T_td.logical_shape = (S, I)  — K=S, F=I
        H_local from output_grad_T_td.sharded_logical_shape[1]

    Args:
        down_weight_grad_td (TensorDescriptor): [H, I], output: dW_down.
        output_grad_T_td (TensorDescriptor): [H, S], transposed output gradient (is_f_by_k=True).
        intermediate_T_td (TensorDescriptor): [I, S], transposed intermediate activations (is_f_by_k=True).
        h_base (int): Row offset into the H dimension for this LNC core.
        dtype: Data type for computation (nl.bfloat16).
        fp8_x4_dtype: MXFP8 quantized data type.
        TILES_IN_BLOCK_M (int): Number of M tiles per block.
        TILES_IN_BLOCK_N (int): Number of N tiles per block.
        TILES_IN_BLOCK_K (int): Number of K tiles to accumulate in PSUM.

    Returns:
        None. Results are written to down_weight_grad_td.data.

    Pseudocode:
        for each m_block in H tiles (sharded):
            for each n_block in I tiles:
                acc = zeros()
                for each k_block in S tiles:
                    load output_grad.T, intermediate.T via DGT + quantize
                    acc += output_grad.T @ intermediate.T
                store acc to down_proj_weight_grad
    """
    # Phase 4: output_grad.T[H,S] @ intermediate.T[I,S].T -> [H,I]
    # K=S, M=H_local, N=I
    S = output_grad_T_td.logical_shape[0]
    H = output_grad_T_td.logical_shape[1]
    I = intermediate_T_td.logical_shape[1]
    H_local = H // NUM_LNC2_CORES if run_with_lnc2 else H

    tile_m = config.tile_m
    tile_n = config.tile_n
    l_tile_k = config.tile_k
    TILES_IN_BLOCK_M = config.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = config.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = config.TILES_IN_BLOCK_K

    matmul_tile_k_physical = l_tile_k // INTERLEAVE_FACTOR
    tiles = {
        'tile_m': tile_m,
        'tile_n': tile_n,
        'l_tile_k': l_tile_k,
        'matmul_tile_k_physical': matmul_tile_k_physical,
        'lhs_matmul_tile_physical': config.lhs_matmul_tile_shape_physical,
        'rhs_matmul_tile_physical': config.rhs_matmul_tile_shape_physical,
        'lhs_load_tile': config.lhs_load_tile_shape,
        'rhs_load_tile': config.rhs_load_tile_shape,
        'lhs_quantize_tile': config.lhs_quantize_tile_shape,
        'rhs_quantize_tile': config.rhs_quantize_tile_shape,
    }

    NUM_H_TILES_LOCAL = div_ceil(H_local, tile_m)
    NUM_K_TILES = div_ceil(S, l_tile_k)
    NUM_I_TILES = div_ceil(I, tile_n)
    NUM_M_BLOCKS = div_ceil(NUM_H_TILES_LOCAL, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_N)

    # Compute load tile shapes from TD state (supports unswizzled BF16, pre-swizzled, pre-quantized)
    lhs_load_tile_shape = _compute_load_tile_shape(output_grad_T_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(intermediate_T_td, tiles, tile_n)

    # Convert h_base to physical offset for pre-swizzled inputs
    h_base_physical = (
        h_base * INTERLEAVE_FACTOR if (output_grad_T_td.is_swizzled and not output_grad_T_td.is_quantized) else h_base
    )

    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffer allocation (skip for pre-quantized inputs)
    output_grad_Tq_td = None
    intermediate_Tq_td = None
    if spill_reload:
        data_buffer = nl.private_hbm if run_with_lnc2 else nl.hbm
        NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

        if not output_grad_T_td.is_quantized:
            output_grad_Tq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )
        if not intermediate_T_td.is_quantized:
            intermediate_Tq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=use_scale_packing,
                data_buffer=data_buffer,
            )

    down_wgrad_local = down_weight_grad_td.data[h_base : h_base + H_local, :]
    output_td = TensorDescriptor(data=down_wgrad_local)

    generic_matmul_mxfp8_api(
        lhs_hbm_td=output_grad_T_td,
        rhs_hbm_td=intermediate_T_td,
        bd=bd,
        output_td=output_td,
        lhs_m_offset=h_base_physical,
        TILES_IN_LOAD_M=config.TILES_IN_LOAD_M,
        TILES_IN_LOAD_N=config.TILES_IN_LOAD_N,
        lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
        rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
        rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
        spill_reload=spill_reload,
        lhsq_td=output_grad_Tq_td,
        rhsq_td=intermediate_Tq_td,
        use_scale_packing=use_scale_packing,
    )


def mlp_backward_mxfp8_base_nki(
    output_grad_td: TensorDescriptor,
    gate_pre_td: TensorDescriptor,
    gate_act_td: TensorDescriptor,
    up_td: TensorDescriptor,
    gate_up_weight_T_td: TensorDescriptor,
    down_weight_T_td: TensorDescriptor,
    d_gate_up_td: TensorDescriptor,
    hidden_states_T_td: TensorDescriptor,
    output_grad_T_td: TensorDescriptor,
    intermediate_T_td: TensorDescriptor,
    scratch_td: TensorDescriptor,
    hidden_states_grad_td: TensorDescriptor,
    weight_grad_td: TensorDescriptor,
    down_weight_grad_td: TensorDescriptor,
    run_with_lnc2: bool = True,
    matmul_config: MlpBwdMatmulConfig = None,
    fp8_x4_dtype: type = float8_e4m3fn_x4,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    clamp_limits: ClampLimits = None,
) -> tuple:
    """MXFP8 SwiGLU MLP backward pass (base kernel).

    Accepts combined gate_up_weight_T [H, 2I] and returns fused gate_up_proj_weight_grad [2I, H].

    All three SwiGLU checkpoint tensors (gate_pre, gate_act, up) are REQUIRED.
    Use the wrapper mlp_backward_mxfp8_nki for checkpoint/recompute support.

    Forward recap:
        gate_pre     = hidden_states @ W_gate.T
        gate_act     = SiLU(gate_pre)
        up           = hidden_states @ W_up.T
        intermediate = gate_act * up
        output       = intermediate @ W_down.T

    Backward phases:
        Phase 1: d_intermediate = output_grad @ W_down.T, then SwiGLU bwd -> d_gate_up [S, 2I]
        Phase 2: hidden_states_grad = d_gate_up[S, 2I] @ gate_up_weight_T[H, 2I].T
        Phase 3: [dW_gate; dW_up] = grad_T[2I,S] @ hidden_states[S,H] -> [2I,H]
        Phase 4: dW_down = output_grad.T @ intermediate

    Dimensions:
        S: Sequence length.
        H: Hidden dimension size.
        I: Intermediate dimension size.

    Args:
        output_grad_td (TensorDescriptor): [S, H], incoming gradient dL/d_output (is_f_by_k=True).
        gate_pre_td (TensorDescriptor): [S, I], gate pre-activation (before SiLU).
        gate_act_td (TensorDescriptor): [S, I], gate post-activation (SiLU(gate_pre)).
        up_td (TensorDescriptor): [S, I], up projection (hidden_states @ W_up.T).
        gate_up_weight_T_td (TensorDescriptor): [H, 2I], transposed fused gate+up weights.
        down_weight_T_td (TensorDescriptor): [I, H], transposed down projection weights.
        d_gate_up_td (TensorDescriptor): [S, 2I], buffer: combined gate/up gradient.
        hidden_states_T_td (TensorDescriptor): [H, S], pre-transposed input hidden states.
        output_grad_T_td (TensorDescriptor): [H, S], pre-transposed output gradient.
        intermediate_T_td (TensorDescriptor): [I, S], pre-transposed intermediate activations.
        scratch_td (TensorDescriptor): [2I, S], buffer: transposed d_gate || d_up.
        hidden_states_grad_td (TensorDescriptor): [S, H], output: dL/d_hidden_states.
        weight_grad_td (TensorDescriptor): [2I, H], output: fused [dW_gate; dW_up].
        down_weight_grad_td (TensorDescriptor): [H, I], output: dL/dW_down.
        run_with_lnc2 (bool): Whether to shard across 2 LNC cores.
        matmul_config (MlpBwdMatmulConfig): Tile sizes for each phase.
        fp8_x4_dtype: MXFP8 quantized data type.
        spill_reload (bool): Whether to use spill/reload for SBUF pressure.
        use_scale_packing (bool): Whether to pack scales.
        clamp_limits (ClampLimits): Optional gradient clamping limits.

    Returns:
        tuple: (hidden_states_grad [S, H], gate_up_weight_grad [2I, H], down_weight_grad [H, I]).

    Pseudocode:
        Phase 1: d_intermediate = output_grad @ W_down.T; SwiGLU bwd -> d_gate_up [S, 2I]
        Phase 2: hidden_states_grad = d_gate_up[S, 2I] @ gate_up_weight[2I, H]
        Phase 3: [dW_gate; dW_up] = grad_T[2I,S] @ hidden_states[S,H] -> [2I,H]
        Phase 4: dW_down = output_grad.T @ intermediate
    """
    H, S = output_grad_td.logical_shape
    I = gate_up_weight_T_td.logical_shape[0] // 2
    dtype = nl.bfloat16

    if run_with_lnc2:
        kernel_assert(S % NUM_LNC2_CORES == 0, f"S ({S}) must be even for LNC2")
        kernel_assert(I % NUM_LNC2_CORES == 0, f"I ({I}) must be even for LNC2 (phase 3 I-sharding)")
        kernel_assert(H % NUM_LNC2_CORES == 0, f"H ({H}) must be even for LNC2 (phase 4 H-sharding)")

    # DGT requires K dimension divisible by DGT_MIN_K for unswizzled BF16 inputs
    # Phase 1: K=H (output_grad, down_weight)
    kernel_assert(
        output_grad_td.is_quantized or output_grad_td.is_swizzled or H % DGT_MIN_K == 0,
        f"H ({H}) must be divisible by {DGT_MIN_K} for DGT when output_grad is unswizzled BF16",
    )
    kernel_assert(
        down_weight_T_td.is_quantized or down_weight_T_td.is_swizzled or H % DGT_MIN_K == 0,
        f"H ({H}) must be divisible by {DGT_MIN_K} for DGT when down_weight is unswizzled BF16",
    )
    # Phase 2: K=2I (d_gate_up LHS is always unswizzled BF16; gate_up_weight_T RHS)
    kernel_assert(
        I % DGT_MIN_K == 0,
        f"I ({I}) must be divisible by {DGT_MIN_K} for DGT (phase 2 d_gate_up is always unswizzled BF16)",
    )
    # Phase 3: K=S (scratch grad_T is always unswizzled BF16, hidden_states_T)
    kernel_assert(
        S % DGT_MIN_K == 0,
        f"S ({S}) must be divisible by {DGT_MIN_K} for DGT (phase 3 grad_T is always unswizzled BF16)",
    )
    # Phase 4: K=S (output_grad_T, intermediate_T)
    kernel_assert(
        output_grad_T_td.is_quantized or output_grad_T_td.is_swizzled or S % DGT_MIN_K == 0,
        f"S ({S}) must be divisible by {DGT_MIN_K} for DGT when output_grad_T is unswizzled BF16",
    )
    kernel_assert(
        intermediate_T_td.is_quantized or intermediate_T_td.is_swizzled or S % DGT_MIN_K == 0,
        f"S ({S}) must be divisible by {DGT_MIN_K} for DGT when intermediate_T is unswizzled BF16",
    )

    _, shard_id = get_program_sharding_info(run_with_lnc2)

    s_base = shard_id * (S // NUM_LNC2_CORES) if run_with_lnc2 else 0
    h_base = shard_id * (H // NUM_LNC2_CORES) if run_with_lnc2 else 0

    # Matmul config must be pre-resolved before entering the kernel
    if matmul_config is None:
        matmul_config = MlpBwdMatmulConfig()

    # Barrier: down_weight_T transpose must complete before phase 1 consumes it
    nisa.core_barrier(data=down_weight_T_td.data, cores=(0, 1))

    compute_phase1_down_proj_mm_grad_mxfp8(
        output_grad_td=output_grad_td,
        gate_pre_td=gate_pre_td,
        gate_act_td=gate_act_td,
        up_td=up_td,
        d_gate_up_td=d_gate_up_td,
        scratch_td=scratch_td,
        down_weight_td=down_weight_T_td,
        s_base=s_base,
        dtype=dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        config=matmul_config.phase1_down_proj,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
        clamp_limits=clamp_limits,
    )

    nisa.core_barrier(data=d_gate_up_td.data, cores=(0, 1))
    nisa.core_barrier(data=gate_up_weight_T_td.data, cores=(0, 1))

    compute_phase2_hidden_states_grad_mxfp8(
        hidden_states_grad_td=hidden_states_grad_td,
        gate_up_weight_td=gate_up_weight_T_td,
        d_gate_up_td=d_gate_up_td,
        s_base=s_base,
        dtype=dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        config=matmul_config.phase2_hidden_grad,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    nisa.core_barrier(data=scratch_td.data, cores=(0, 1))
    # Barrier: hidden_states_T transpose must complete before phase 3 consumes it
    nisa.core_barrier(data=hidden_states_T_td.data, cores=(0, 1))

    compute_phase3_gate_up_weight_grad_mxfp8(
        weight_grad_td=weight_grad_td,
        hidden_states_T_td=hidden_states_T_td,
        grad_T_td=scratch_td,
        dtype=dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        config=matmul_config.phase3_wgrad,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    # Barrier: output_grad_T and intermediate_T transposes must complete before phase 4 consumes them
    nisa.core_barrier(data=output_grad_T_td.data, cores=(0, 1))
    nisa.core_barrier(data=intermediate_T_td.data, cores=(0, 1))

    compute_phase4_down_weight_grad_mxfp8(
        down_weight_grad_td=down_weight_grad_td,
        output_grad_T_td=output_grad_T_td,
        intermediate_T_td=intermediate_T_td,
        h_base=h_base,
        dtype=dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        config=matmul_config.phase4_wgrad,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    return hidden_states_grad_td.data, weight_grad_td.data, down_weight_grad_td.data


@with_active_sbm
def mlp_backward_mxfp8_nki(
    # Output gradient (always BF16)
    output_grad: nl.ndarray,
    # Hidden states (always BF16, needed for recompute and/or transpose)
    hidden_states: nl.ndarray,
    # Weights — BF16 (required when corresponding scales not provided)
    down_proj_weight: nl.ndarray = None,
    gate_up_weights: nl.ndarray = None,
    # Weights — pre-quantized (used when scales provided)
    gate_up_weight_T: nl.ndarray = None,
    gate_up_weight_T_scales: nl.ndarray = None,
    gate_up_weights_scales: nl.ndarray = None,
    down_weight_T: nl.ndarray = None,
    down_weight_T_scales: nl.ndarray = None,
    # Activations — pre-quantized (optional, used when scales provided)
    output_grad_T: nl.ndarray = None,
    output_grad_T_scales: nl.ndarray = None,
    hidden_states_T: nl.ndarray = None,
    hidden_states_T_scales: nl.ndarray = None,
    # Optional checkpoints (always BF16)
    gate_pre: nl.ndarray = None,
    gate_act: nl.ndarray = None,
    up: nl.ndarray = None,
    intermediate: nl.ndarray = None,
    # Configuration
    run_with_lnc2: bool = True,
    matmul_config: MlpBwdMatmulConfig = None,
    fp8_x4_dtype: type = float8_e4m3fn_x4,
    spill_reload: bool = True,
    use_scale_packing: bool = True,
    clamp_limits: ClampLimits = None,
) -> tuple:
    """MXFP8 SwiGLU MLP backward pass with activation checkpointing support.

    The kernel infers the format of each tensor from whether its scale is provided:
      - scales provided → tensor is pre-quantized MXFP8 x4, pre-transposed by caller
      - scales=None → tensor is raw BF16, kernel transposes internally

    This applies per-tensor, allowing mixed configurations (e.g., weights
    pre-quantized but activations BF16).

    Tensors that are always BF16 (never pre-quantized):
      - output_grad: [S, H], incoming gradient (phase 1 LHS)
      - hidden_states: [S, H], for recompute
      - intermediate: [S, I], always transposed internally
      - gate_pre, gate_act, up: [S, I] optional checkpoints

    Tensors with optional pre-quantization:
      - gate_up_weight_T + gate_up_weight_T_scales: phase 2 RHS
      - gate_up_weights + gate_up_weights_scales: for recompute
      - down_weight_T + down_weight_T_scales: phase 1 RHS
      - output_grad_T + output_grad_T_scales: phase 4 LHS
      - hidden_states_T + hidden_states_T_scales: phase 3 RHS

    Forward recap:
        gate_pre     = hidden_states @ W_gate.T
        gate_act     = SiLU(gate_pre)
        up           = hidden_states @ W_up.T
        intermediate = gate_act * up
        output       = intermediate @ W_down.T

    Backward phases:
        Phase 0 (conditional): Recompute missing checkpointed activations.
        Phase 1: d_intermediate = output_grad @ down_weight_T; SwiGLU bwd → d_gate_up
        Phase 2: hidden_states_grad = d_gate_up @ gate_up_weight_T
        Phase 3: gate_up_weight_grad = grad_T @ hidden_states_T
        Phase 4: down_proj_weight_grad = output_grad_T @ intermediate_T

    Args:
        output_grad: [S, H] BF16, incoming gradient dL/d_output.
        hidden_states: [S, H] BF16, original input (for recompute + phase 3 transpose).
        down_proj_weight: [H, I] BF16, required when down_weight_T_scales is None.
        gate_up_weights: [2I, H] BF16 or MXFP8 x4. Always required (recompute + phase 2).
        gate_up_weight_T: [2I/4, H] MXFP8 x4, pre-transposed. Required when scales provided.
        gate_up_weight_T_scales: Scales for gate_up_weight_T.
        gate_up_weights_scales: Scales for gate_up_weights.
        down_weight_T: [H/4, I] MXFP8 x4, pre-transposed. Required when scales provided.
        down_weight_T_scales: Scales for down_weight_T.
        output_grad_T: [S/4, H] MXFP8 x4, pre-transposed. Required when scales provided.
        output_grad_T_scales: Scales for output_grad_T.
        hidden_states_T: [S/4, H] MXFP8 x4, pre-transposed. Required when scales provided.
        hidden_states_T_scales: Scales for hidden_states_T.
        gate_pre: [S, I] BF16 optional checkpoint.
        gate_act: [S, I] BF16 optional checkpoint.
        up: [S, I] BF16 optional checkpoint.
        intermediate: [S, I] BF16 optional checkpoint.

    Returns:
        tuple: (hidden_states_grad [S, H], gate_up_weight_grad [2I, H],
                down_proj_weight_grad [H, I]), all bf16.
    """
    if get_active_sbm() == None:
        create_and_set_active_sbm()

    sbm = get_active_sbm()
    sbm.open_scope(name="MXFP8 MLP BWD ")

    # --- Build TDs and infer format from scales ---
    output_grad_td = TensorDescriptor(
        data=output_grad,
        is_col_parallel_sharded=run_with_lnc2,
    )
    H, S = output_grad_td.logical_shape

    gate_up_weights_td = TensorDescriptor(
        data=gate_up_weights,
        scales=gate_up_weights_scales,
        is_col_parallel_sharded=True,
    )

    # Per-tensor: build TD if data provided (pre-quantized), otherwise mark for internal transpose
    weights_prequantized = gate_up_weight_T_scales != None
    down_prequantized = down_weight_T_scales != None
    hidden_T_prequantized = hidden_states_T_scales != None
    output_grad_T_prequantized = output_grad_T_scales != None

    if weights_prequantized:
        gate_up_weight_T_td = TensorDescriptor(data=gate_up_weight_T, scales=gate_up_weight_T_scales)
        I = gate_up_weight_T_td.logical_shape[0] // 2
    else:
        I = gate_up_weights_td.logical_shape[1] // 2

    if down_prequantized:
        down_weight_T_td = TensorDescriptor(data=down_weight_T, scales=down_weight_T_scales)

    if hidden_T_prequantized:
        hidden_states_T_td = TensorDescriptor(data=hidden_states_T, scales=hidden_states_T_scales)

    if output_grad_T_prequantized:
        output_grad_T_td = TensorDescriptor(
            data=output_grad_T, scales=output_grad_T_scales, is_col_parallel_sharded=run_with_lnc2
        )

    dtype = nl.bfloat16

    # --- Assertions ---
    if run_with_lnc2:
        kernel_assert(S % NUM_LNC2_CORES == 0, f"S ({S}) must be even for LNC2")
        kernel_assert(I % NUM_LNC2_CORES == 0, f"I ({I}) must be even for LNC2 (phase 3 I-sharding)")
        kernel_assert(H % NUM_LNC2_CORES == 0, f"H ({H}) must be even for LNC2 (phase 4 H-sharding)")

    # DGT constraints only apply to BF16 tensors that need internal transpose
    if not weights_prequantized:
        kernel_assert(H % DGT_MIN_K == 0, f"H ({H}) must be divisible by {DGT_MIN_K} for DGT (gate_up transpose)")
    if not hidden_T_prequantized:
        kernel_assert(S % DGT_MIN_K == 0, f"S ({S}) must be divisible by {DGT_MIN_K} for DGT (hidden_states transpose)")
    if not output_grad_T_prequantized:
        kernel_assert(H % DGT_MIN_K == 0, f"H ({H}) must be divisible by {DGT_MIN_K} for DGT (output_grad transpose)")
    # intermediate is always transposed internally
    kernel_assert(S % DGT_MIN_K == 0, f"S ({S}) must be divisible by {DGT_MIN_K} for DGT (intermediate transpose)")

    # --- Allocate output buffers ---
    hbm_buf = nl.shared_hbm if run_with_lnc2 else nl.hbm
    hidden_states_grad = nl.ndarray((S, H), dtype=dtype, buffer=hbm_buf, name="hidden_states_grad")
    gate_up_weight_grad = nl.ndarray((2 * I, H), dtype=dtype, buffer=hbm_buf, name="gate_up_weight_grad")
    down_proj_weight_grad = nl.ndarray((H, I), dtype=dtype, buffer=hbm_buf, name="down_proj_weight_grad")

    # --- Allocate internal buffers ---
    d_gate_up_buf = nl.ndarray((S, 2 * I), dtype=dtype, buffer=hbm_buf, name="d_gate_up_buf")
    scratch_buf = nl.ndarray((2 * I, S), dtype=dtype, buffer=hbm_buf, name="scratch_buf")
    gate_pre_buf = nl.ndarray((S, I), dtype=dtype, buffer=hbm_buf, name="gate_pre_buf")
    gate_act_buf = nl.ndarray((S, I), dtype=dtype, buffer=hbm_buf, name="gate_act_buf")
    up_buf = nl.ndarray((S, I), dtype=dtype, buffer=hbm_buf, name="up_buf")
    intermediate_buf = nl.ndarray((S, I), dtype=dtype, buffer=hbm_buf, name="intermediate_buf")
    intermediate_T_buf = nl.ndarray((I, S), dtype=dtype, buffer=hbm_buf, name="intermediate_T_buf")

    _, shard_id = get_program_sharding_info(run_with_lnc2)
    s_base = shard_id * (S // NUM_LNC2_CORES) if run_with_lnc2 else 0

    if matmul_config is None:
        matmul_config = MlpBwdMatmulConfig()

    # --- Phase 0: Recompute missing checkpoints ---
    eff_gate_pre = gate_pre if gate_pre != None else gate_pre_buf
    eff_up = up if up != None else up_buf
    eff_gate_act = gate_act if gate_act != None else gate_act_buf
    eff_intermediate = intermediate if intermediate != None else intermediate_buf

    gate_pre_buf_td = TensorDescriptor(data=gate_pre_buf)
    up_buf_td = TensorDescriptor(data=up_buf)
    gate_act_buf_td = TensorDescriptor(data=gate_act_buf)
    intermediate_buf_td = TensorDescriptor(data=intermediate_buf)

    eff_gate_pre_td = TensorDescriptor(data=eff_gate_pre)
    eff_up_td = TensorDescriptor(data=eff_up)
    eff_gate_act_td = TensorDescriptor(data=eff_gate_act)
    eff_intermediate_td = TensorDescriptor(data=eff_intermediate)

    need_recompute_projections = (gate_pre == None) or (up == None)
    if need_recompute_projections:
        hidden_states_td = TensorDescriptor(
            data=hidden_states,
            is_col_parallel_sharded=run_with_lnc2,
        )
        matmul_config.resolve_recompute_phases(
            hidden_states_td, gate_up_weights_td, run_with_lnc2, spill_reload, use_scale_packing
        )
        recompute_gate_up_projection(
            hidden_td=hidden_states_td,
            gate_up_td=gate_up_weights_td,
            gate_pre_td=gate_pre_buf_td,
            up_td=up_buf_td,
            s_base_offset=s_base,
            dtype=dtype,
            fp8_x4_dtype=fp8_x4_dtype,
            gate_config=matmul_config.recompute_gate,
            up_config=matmul_config.recompute_up,
            spill_reload=spill_reload,
            use_scale_packing=use_scale_packing,
            run_with_lnc2=run_with_lnc2,
            clamp_limits=clamp_limits,
        )
        nisa.core_barrier(gate_pre_buf_td.data, cores=(0, 1))

    if gate_act == None:
        recompute_gate_act(
            gate_pre_td=eff_gate_pre_td,
            gate_act_td=gate_act_buf_td,
            s_base_offset=s_base,
            dtype=dtype,
            run_with_lnc2=run_with_lnc2,
        )

    if need_recompute_projections:
        nisa.core_barrier(up_buf_td.data, cores=(0, 1))

    if intermediate == None:
        recompute_intermediate(
            gate_act_td=eff_gate_act_td,
            up_td=eff_up_td,
            intermediate_td=intermediate_buf_td,
            s_base_offset=s_base,
            dtype=dtype,
            run_with_lnc2=run_with_lnc2,
        )

    # --- Per-tensor transpose: only transpose BF16 tensors that are not pre-quantized ---
    S_half = S // NUM_LNC2_CORES if run_with_lnc2 else S
    H_half = H // NUM_LNC2_CORES if run_with_lnc2 else H
    src_s_offset = shard_id * S_half if run_with_lnc2 else 0
    src_h_offset = shard_id * H_half if run_with_lnc2 else 0

    if not weights_prequantized:
        gate_up_weight_T_buf = nl.ndarray((H, 2 * I), dtype=dtype, buffer=hbm_buf, name="gate_up_weight_T_buf")
        two_I_half = 2 * I // NUM_LNC2_CORES if run_with_lnc2 else 2 * I
        src_2i_offset = shard_id * two_I_half if run_with_lnc2 else 0
        hbm_dma_transpose(
            gate_up_weights,
            gate_up_weight_T_buf,
            M=two_I_half,
            src_row_offset=src_2i_offset,
            dst_col_offset=src_2i_offset,
        )
        gate_up_weight_T_td = TensorDescriptor(data=gate_up_weight_T_buf)

    if not down_prequantized:
        down_weight_T_buf = nl.ndarray((I, H), dtype=dtype, buffer=hbm_buf, name="down_weight_T_buf")
        hbm_dma_transpose(
            down_proj_weight,
            down_weight_T_buf,
            M=H_half,
            src_row_offset=src_h_offset,
            dst_col_offset=src_h_offset,
        )
        down_weight_T_td = TensorDescriptor(data=down_weight_T_buf)

    if not hidden_T_prequantized:
        hidden_states_T_buf = nl.ndarray((H, S), dtype=dtype, buffer=hbm_buf, name="hidden_states_T_buf")
        hbm_dma_transpose(
            hidden_states,
            hidden_states_T_buf,
            M=S_half,
            src_row_offset=src_s_offset,
            dst_col_offset=src_s_offset,
        )
        hidden_states_T_td = TensorDescriptor(data=hidden_states_T_buf)

    if not output_grad_T_prequantized:
        output_grad_T_buf = nl.ndarray((H, S), dtype=dtype, buffer=hbm_buf, name="output_grad_T_buf")
        hbm_dma_transpose(
            output_grad,
            output_grad_T_buf,
            N=H_half,
            src_col_offset=src_h_offset,
            dst_row_offset=src_h_offset,
        )
        output_grad_T_td = TensorDescriptor(data=output_grad_T_buf, is_col_parallel_sharded=run_with_lnc2)

    # Transpose intermediate [S, I] -> [I, S] (common to all modes)
    hbm_dma_transpose(
        eff_intermediate_td.data,
        intermediate_T_buf,
        M=S_half,
        src_row_offset=src_s_offset,
        dst_col_offset=src_s_offset,
    )
    intermediate_T_td = TensorDescriptor(data=intermediate_T_buf)

    # --- Build remaining TDs and delegate to base kernel ---
    d_gate_up_td = TensorDescriptor(data=d_gate_up_buf, is_f_by_k=True, is_col_parallel_sharded=run_with_lnc2)
    scratch_td = TensorDescriptor(data=scratch_buf, is_f_by_k=True, is_col_parallel_sharded=run_with_lnc2)
    hidden_states_grad_td = TensorDescriptor(data=hidden_states_grad)
    weight_grad_td = TensorDescriptor(data=gate_up_weight_grad)
    down_weight_grad_td = TensorDescriptor(data=down_proj_weight_grad)

    matmul_config.resolve_backward_phases(
        run_with_lnc2,
        spill_reload,
        use_scale_packing,
        output_grad_td=output_grad_td,
        down_weight_T_td=down_weight_T_td,
        d_gate_up_td=d_gate_up_td,
        gate_up_weight_T_td=gate_up_weight_T_td,
        scratch_td=scratch_td,
        hidden_states_T_td=hidden_states_T_td,
        output_grad_T_td=output_grad_T_td,
        intermediate_T_td=intermediate_T_td,
    )

    mlp_backward_mxfp8_base_nki(
        output_grad_td=output_grad_td,
        gate_pre_td=eff_gate_pre_td,
        gate_act_td=eff_gate_act_td,
        up_td=eff_up_td,
        gate_up_weight_T_td=gate_up_weight_T_td,
        down_weight_T_td=down_weight_T_td,
        d_gate_up_td=d_gate_up_td,
        hidden_states_T_td=hidden_states_T_td,
        output_grad_T_td=output_grad_T_td,
        intermediate_T_td=intermediate_T_td,
        scratch_td=scratch_td,
        hidden_states_grad_td=hidden_states_grad_td,
        weight_grad_td=weight_grad_td,
        down_weight_grad_td=down_weight_grad_td,
        run_with_lnc2=run_with_lnc2,
        matmul_config=matmul_config,
        fp8_x4_dtype=fp8_x4_dtype,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        clamp_limits=clamp_limits,
    )

    sbm.close_scope()

    return hidden_states_grad, gate_up_weight_grad, down_proj_weight_grad
