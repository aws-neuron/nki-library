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
This kernel implements blockwise matrix multiplication for Mixture of Experts (MoE) layers using MXFP4 or MXFP8 quantization with block-level sharding. The implementation shards gate/up projections over the intermediate dimension and block accumulation over the batch dimension, processing all blocks without distinguishing between padded and non-padded blocks.
"""

from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ...mlp.mlp_parameters import MLPQuantizationParameters
from ...utils.allocator import SbufManager, sizeinbytes
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import _sbm_alloc, get_nl_act_fn_from_type
from ...utils.logging import get_logger
from ...utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .bwmm_shard_on_I import OutputTensors
from .down_projection_mx import down_projection_mx
from .gate_up_projection_mx import gate_up_projection_mx_tp
from .moe_cte_mx_utils import (
    SBUF_QUADRANT_SIZE,
    BWMMMXConfigs,
    BWMMMXDimensionSizes,
    InputTensors,
    ProjConfig,
    SharedBuffers,
    _generate_expert_index_vector,
    _pmax,
    _q_height,
    _q_width,
    apply_clamp,
    compute_hidden_index_vector,
    convert_to_mxfp_dtype,
    load_and_quantize_hidden_states,
    load_fp8_hidden_states_mx,
    load_hidden_states_mx,
    quantize_block_hidden_state_T,
    quantize_block_hidden_state_T_static_mx,
    sbuf_layout_adapter,
    transpose_fp8_hidden_states,
)
from .moe_cte_utils import (
    PSUM_SIZE,
    SkipMode,
    calculate_expert_affinities,
    div_ceil,
    load_block_expert,
    load_token_indices,
    load_token_indices_dynamic_block,
    output_initialization,
    reduce_outputs,
)

USE_DMA_TRANSPOSE = False

# Packed-scale geometry: each MX scale tile occupies a 4-partition stripe per
# 32-partition quadrant; up to 4 such tiles (16 partitions of the top half) fit
# per packed buffer. Used when computing the packed SBUF buffer shape.
SLOTS_PER_PACKED_BUFFER = 4

# I-tile size for tiled gate/up projection (512 = _pmax * _q_width = one I512 tile)
_I_TILE_SZ = _pmax * _q_width

# Reserve scratchpad buffer space for internally created ops, ex: identity for hidden transpose
SBUF_SCRATCHPAD_RESERVE = 1024

# ---------------------------------------------------------------------------
# Chunked dynamic-loop parameters
# ---------------------------------------------------------------------------
# _DYN_INNER = number of consecutive blocks one core processes per outer iter.
#   - each outer iter covers_DYN_INNER * 2 blocks total (= _DYN_STEP).
#   - The outer loop runs n_active // _DYN_STEP times; leftover 0.._DYN_STEP-1
#     active blocks fall through to the remainder ping-pong loop.

_DYN_INNER = 8
_DYN_STEP = _DYN_INNER * 2  # blocks consumed per outer iter
_DYN_INNER_M1 = _DYN_INNER - 1
_DYN_STEP_LOG2 = _DYN_STEP.bit_length() - 1  # log2(_DYN_STEP); used as right-shift count log2(16) - 1 = 3

logger = get_logger("bwmm_shard_on_block_mx")


@nki.jit
def bwmm_shard_on_block_mx(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    # dynamic-loop variables
    conditions: nl.NkiTensor = None,
    gate_and_up_proj_bias: nl.NkiTensor = None,
    down_proj_bias: nl.NkiTensor = None,
    # quantize scales
    gate_up_proj_scale: nl.NkiTensor = None,
    down_proj_scale: nl.NkiTensor = None,
    # Non-tensor args
    block_size=None,
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = -1,
    # Routing shape for the auto-computed best-case static-block estimate.
    # top_k tokens per expert, sharded across ep_degree EP ranks.
    top_k: int = 1,
    ep_degree: int = 1,
    gate_up_activations_T=None,
    down_activations=None,
    # Meta parameters
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype=nl.bfloat16,
    weight_dtype: Any = None,  # Target dtype for weight conversion (e.g., nl.float8_e4m3fn_x4, nl.float8_e5m2_x4)
    is_tensor_update_accumulating=True,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    use_packed_scales: bool = False,
    # When quantization_type == QuantizationType.STATIC_MX, takes the STATIC_MX path.
    # Otherwise (default NONE / any non-STATIC_MX value), behaves as MX (existing path, bit-identical).
    # STATIC_MX reuses the existing gate_up_proj_scale / down_proj_scale tensors to carry the
    # per-expert weight scales (gate_up_proj_scale: fp32 [E, 2, 1], down_proj_scale: fp32 [E, 1]);
    # the per-tensor input scales come in via gate_up_in_scale / down_in_scale.
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_up_in_scale: Optional[nl.NkiTensor] = None,
    down_in_scale: Optional[nl.NkiTensor] = None,
):
    """
    Blockwise MXFP MoE kernel, decorated. Use as standalone kernel.

    The blockwise matrix multiplication (matmul) kernel implements a Mixture of Experts (MoE)
    layer at a block granularity, offering an alternative to token dropping approaches.
    This method assumes that tokens have already been assigned to blocks, as specified
    by the user through the token_position_to_id parameter. This kernel shards the gate/up projection
    over the I dimension, and shards the block accumulation over the B dimension.
    This kernel loops over all blocks, without considering they are padded or non-padded blocks.
    Supports both MXFP4 and MXFP8 weight quantization.

    Intended Usage:
        - Block size B: 128-1024 tokens
        - Total tokens T: 32k
        - Hidden dimension H: 512-8192
        - Intermediate dimension I_TP: 384-3072
        - Number of experts E: 8-128

    Dimensions:
        H: Hidden dimension size
        T: Total number of input tokens (after linearizing across the batch dimension)
        B: Number of tokens per block
        N: Total number of blocks
        E: Number of experts
        I: Intermediate size / tp degree

    Args:
        hidden_states (nl.NkiTensor): Tensor of input hidden states on HBM of size (T+1, H). The reason it is T+1 is because padding token position is set to T.
                                       with skip_dma, id will be set to -1, so this shape can be (T, H). Similarly for expert_affinities_masked, output
        expert_affinities_masked (nl.NkiTensor): Tensor of expert affinities corresponding to each token of size ((T+1) * E, 1).
                                        TODO: cannot refactor to (T+1, E) as we currently don't support dynamic slice on both axis.
        gate_up_proj_weight (nl.NkiTensor): Tensor of concatenated gate and up projection weights on HBM (E, H, 2, I).
                                          Supports MXFP4 (nl.float4_e2m1fn_x4) and MXFP8 (nl.float8_e4m3fn_x4, nl.float8_e5m2_x4).
        down_proj_weight (nl.NkiTensor): Tensor of down projection weights on HBM (E, I, H).
                                       Supports MXFP4 (nl.float4_e2m1fn_x4) and MXFP8 (nl.float8_e4m3fn_x4, nl.float8_e5m2_x4).
        block_size (int): Number of tokens per block
        token_position_to_id (nl.NkiTensor): Tensor of block index of the corresponding tokens on HBM (N * B,)
                                          Note that we include tokens included for padding purposes and N * B >= T.
                                          For padding token, id is set to T. with skip_dma, id will be set to -1.
        block_to_expert (nl.NkiTensor): Tensor of expert indices of corresponding blocks on HBM (N, 1)

        num_static_block (int): Optional. Number of non-padded blocks if known (default: -1).
        n_dynamic_blocks (int): Number of blocks to process with dynamic loop when n_static_blocks
            is not specified (default: 55, empirically tuned for GPT-OSS).
        gate_and_up_proj_bias: nl.NkiTensor = None, Optional. A tensor of shape [E, 2, I].
                              Note that if activation function is Swiglu, we expect up_bias = up_bias + 1
        down_proj_bias: nl.NkiTensor = None. Optional argument. A tensor of shape [E, H]

        # Arguments for quantization scales
        gate_up_proj_scale: nl.NkiTensor = None. uint8 MX scales of shape
                            [E, _pmax // _q_height, 2, n_H512_tile, I] (standard layout) or
                            [E, _pmax, n_packed_gup, 2, I] (use_packed_scales=True).
                            Under STATIC_MX it instead carries the per-expert gate/up weight
                            scales as fp32 [E, 2, 1] (idx 0 = gate, idx 1 = up).
        down_proj_scale: nl.NkiTensor = None. uint8 MX scales of shape
                            [E, p_I // _q_height, n_total_I512_tile, H] (standard layout) or
                            [E, _pmax, n_packed_down, H] (use_packed_scales=True).
                            Under STATIC_MX it instead carries the per-expert down weight
                            scale as fp32 [E, 1].

        # Unsupported output tensors. Please set to None.
        gate_up_activations_T: nl.NkiTensor = None. Currently not supported.
        down_activations: nl.NkiTensor = None. Currently not supported

        # meta parameters
        activation_function: one of the Enum in nkilib.core.utils.common_types.ActFnType.
                              Indicate what activation function to use in the MLP block
        skip_dma: SkipMode = SkipMode(False, False),
        compute_dtype=nl.bfloat16,
        weight_dtype: Target dtype for weight conversion when weights are passed as uint/int/float types.
                     For MXFP4: nl.float4_e2m1fn_x4
                     For MXFP8: nl.float8_e4m3fn_x4 or nl.float8_e5m2_x4
                     If None, auto-detects (defaults to e4m3fn for MXFP8)
        is_tensor_update_accumulating: bool. Indicate whether we need to accumulate the results over multiple blocks
        expert_affinities_scaling_mode: one of the Enum in nkilib.core.utils.common_types.ExpertAffinityScaleMode.
                                        Indicate if the kernel is doing post or pre scaling.
        n_block_per_iter: int. Currently unsupported

        #parameters for clipping the MLP projections
        gate_clamp_upper_limit: Optional[float] = None,
        gate_clamp_lower_limit: Optional[float] = None,
        up_clamp_lower_limit: Optional[float] = None,
        up_clamp_upper_limit: Optional[float] = None

        skip_dma (bool): Whether to skip DMA operations (default: False)

    Returns:
        output (nl.NkiTensor): Tensor of output hidden states on HBM of size (T+1, H).

    Notes:
        - All input/output tensors must have the same floating point dtype
        - token_position_to_id and block_to_expert must be np.int32 tensors

    Pseudocode:
        if expert_affinities_scaling_mode == PRE_SCALE_DELAYED:
            expert_affinities_scaling_mode = PRE_SCALE

        T, H = hidden_states.shape
        B = block_size
        E, _, _, _, I = gate_up_proj_weight.shape
        N = token_position_to_id.shape[0] // B
        dims = BWMMMXDimensionSizes(T, H, B, E, N, I, cond_vec_len)
        prj_cfg = ProjConfig(H, I, B, force_lnc1=True, n_prgs=1, prg_id=0)
        configs = BWMMMXConfigs(...)

        allocate reused buffers: p_gup_idx_vector, p_down_idx_vector, gup_scales_sb, activation_bias
        inps = InputTensors(...)

        check_kernel_compatibility(dims, configs)

        if is_tensor_update_accumulating:
            output = allocate [2, T, H] in HBM
            output_initialization(output, dims)
        else:
            output = allocate [T, H] in HBM

        allocate shared buffers: block_hidden_states, block_hidden_states_T, hidden_qtz_sb, hidden_scale_sb
        allocate down_weight_qtz, block_old, cond, index

        if use_dynamic_while:
            n_dynamic_blocks = N - n_static_blocks (padded to even)
            n_static_blocks = N - n_dynamic_blocks
            process_static_blocks(n_static_blocks)
            process_dynamic_blocks(n_dynamic_blocks)
        else:
            process_static_blocks(N)

        if num_shards == 2:
            core_barrier(output)

        if is_tensor_update_accumulating and num_shards > 1:
            reduce_outputs(output)

        return output
    """

    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE_DELAYED:
        expert_affinities_scaling_mode = ExpertAffinityScaleMode.PRE_SCALE

    # Convert weights to MXFP dtype first
    gate_up_proj_weight, target_dtype = convert_to_mxfp_dtype(gate_up_proj_weight, weight_dtype)
    down_proj_weight, _ = convert_to_mxfp_dtype(down_proj_weight, target_dtype)

    # Pre-quantized fp8 hidden states (real MX): hidden_states arrives as a concatenated
    # [T, H_concat] tensor = [hidden_quant (H fp8) | hidden_scale (H/4 uint8)] from a prior
    # QMX layer. We skip on-device quantization and gather+transpose both regions. The true
    # hidden dim H is recovered from the gate/up weight (concat dim hides it on the input).
    is_fp8_hidden = hidden_states.dtype in (nl.float8_e4m3fn, nl.float8_e5m2)

    T = hidden_states.shape[0]
    B = block_size
    E, _Hp, _, _n_H512, I = gate_up_proj_weight.shape
    # When the producer fused expert affinities into the row, the concat is
    # [hidden (H fp8) | scale (scale_region uint8) | affinities (E bf16) | pad]. Detect this by the
    # concat being wider than hidden+scale, and extract the affinity for this block. affinities_col_offset is the fp8 column
    # where the affinity region starts (= the end of the hidden+scale region).
    is_affinities_packed = False
    affinities_col_offset = 0
    if is_fp8_hidden:
        H = _Hp * _n_H512 * _q_width  # real hidden dimension that's not concatted with scales
        kernel_assert(
            quantization_type == QuantizationType.MX,
            f"fp8 pre-quantized hidden states only support QuantizationType.MX, got {quantization_type}",
        )
        hidden_scale_width = H + div_ceil(_n_H512, SLOTS_PER_PACKED_BUFFER) * _pmax
        is_affinities_packed = hidden_states.shape[-1] > hidden_scale_width
        affinities_col_offset = hidden_scale_width
    else:
        _, H = hidden_states.shape
    cond_vec_len = conditions.shape[0] if conditions != None else 0

    N = token_position_to_id.shape[0] // B
    dims = BWMMMXDimensionSizes(T=T, H=H, B=B, E=E, N=N, I=I, cond_vec_len=cond_vec_len)

    is_static_quant = quantization_type == QuantizationType.STATIC_MX

    quant_params = None
    if is_static_quant:
        # STATIC_MX folds the per-block MX scale tables into the dummy-127 path, so the
        # per-block packed-scale layout is unused here; reject it explicitly.
        kernel_assert(
            not use_packed_scales,
            "STATIC_MX does not support use_packed_scales; per-block scales are not used on this path.",
        )
        # Reuse the existing weight-scale tensors: gate_up_proj_scale carries the packed
        # [E, 2, 1] gate/up weight scales, down_proj_scale carries the [E, 1] down weight scale.
        # Split the packed gate/up tensor into separate [E, 1] gate and up views here so the
        # setup below consumes gate_w_scale / up_w_scale symmetrically (idx 0 = gate, idx 1 = up).
        gate_up_w_view = gate_up_proj_scale.reshape((dims.E, 2, 1))
        quant_params = MLPQuantizationParameters(
            quantization_type=quantization_type,
            gate_w_scale=gate_up_w_view.slice(dim=1, start=0, end=1).reshape((dims.E, 1)),
            up_w_scale=gate_up_w_view.slice(dim=1, start=1, end=2).reshape((dims.E, 1)),
            down_w_scale=down_proj_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            clipping_bound=0.0,
        )

    # When use_packed_scales is True, both scale tensors are in packed HBM layout:
    #   gate_up_proj_scale: uint8[E, _pmax, n_packed_gup, 2, I]
    #   down_proj_scale:    uint8[E, _pmax, n_packed_down, H]
    # Producers always emit both packed or both standard, so a single flag covers both.

    # Skip unused partition zeroing when both gate and up have at least one clamp,
    # since the clamp op writes all partitions (including unused ones).
    has_gate_clamp = gate_clamp_upper_limit != None or gate_clamp_lower_limit != None
    has_up_clamp = up_clamp_upper_limit != None or up_clamp_lower_limit != None
    zero_unused_partitions = not (has_gate_clamp and has_up_clamp)

    prj_cfg = ProjConfig(
        H=dims.H,
        I=dims.I,
        BxS=dims.B,
        force_lnc1=True,
        n_prgs=1,
        prg_id=0,
        use_stream_shuffle_broadcast=False,
        sharding_config="H",
        zero_unused_partitions=zero_unused_partitions,
    )

    # subtract 1024 for scratchpad buffer space for internally creates ops, ex: identity for hidden transpose
    sb_upper_bound = nl.tile_size.total_available_sbuf_size - SBUF_SCRATCHPAD_RESERVE
    sbm = SbufManager(0, sb_upper_bound, logger)
    sbm.open_scope(name="top_level_scope")

    # reused buffers
    p_gup_idx_vector = sbm.alloc_stack((_pmax, 1), dtype=nl.float32, name="p_idx_vector", align=SBUF_QUADRANT_SIZE)
    nisa.memset(dst=p_gup_idx_vector, value=-1.0)

    p_gup_idx_vector_int32 = sbm.alloc_stack(
        (_pmax, 1), dtype=nl.int32, name="p_idx_vector_int32", align=SBUF_QUADRANT_SIZE
    )

    p_down_idx_vector = sbm.alloc_stack(
        (_pmax, 1), dtype=nl.float32, name="p_down_idx_vector", align=SBUF_QUADRANT_SIZE
    )
    nisa.memset(dst=p_down_idx_vector, value=-1.0)

    """
    When packing weight scales, allocate a smaller [_pmax, n_packed, 2, I] buffer instead
    of [_pmax, 2, n_H512_tile_sharded, I]. Since n_packed = ceil(n_H512_tile / 4), we save
    roughly 4x SBUF on this buffer when n_H512_tile is a multiple of 4 (and 3x for n_H512_tile=6).
    Unpacked path: scale DMAs use vector_offset which forces SWDGE; SWDGE permits
    src/dst dtype mismatch, so we can allocate as uint32 (4x faster memset on the upfront
    zero-pad) and consume via .view(nl.uint8). The packed path uses scalar_offset HWDGE,
    which requires src/dst dtype match, so it stays on uint8.
    """
    _gup_alloc_as_u32 = (not use_packed_scales) and (dims.I % 4 == 0)
    if is_static_quant:
        # STATIC_MX: single shared [_pmax, max_free] all-127 dummy reused across
        # gup_scales_sb (cur_I128_tile_sz ≤ 128), hidden_scale_sb / down inter
        # (cur_BxS ≤ BxS_tile_sz), etc. BxS_tile_sz = min(B, _psum_fmax*2/_q_width)
        # = min(B, 256), so the buffer must fit the widest tile.
        _bxs_tile_sz = min(dims.B, PSUM_SIZE * 2 // _q_width)
        _static_dummy_free = max(_pmax, _bxs_tile_sz)
        n_packed_gup = 0
        static_dummy_scale_sb = sbm.alloc_stack(
            (_pmax, _static_dummy_free),
            dtype=nl.uint8,
            name="static_dummy_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        nisa.memset(static_dummy_scale_sb, value=127)
        gup_scales_sb = static_dummy_scale_sb
    elif use_packed_scales:
        n_packed_gup = (prj_cfg.n_H512_tile_sharded + SLOTS_PER_PACKED_BUFFER - 1) // SLOTS_PER_PACKED_BUFFER
        gup_scales_sb = sbm.alloc_stack(
            (_pmax, n_packed_gup, 2, dims.I),
            dtype=nl.uint8,
            name="gup_scales_packed_sb",
            align=SBUF_QUADRANT_SIZE,
        )
    elif _gup_alloc_as_u32:
        n_packed_gup = 0
        gup_scales_sb = sbm.alloc_stack(
            (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I // 4),
            dtype=nl.uint32,
            name="gup_scales_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        gup_scales_sb = gup_scales_sb.view(nl.uint8)
    else:
        n_packed_gup = 0
        gup_scales_sb = sbm.alloc_stack(
            (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
            dtype=nl.uint8,
            name="gup_scales_sb",
            align=SBUF_QUADRANT_SIZE,
        )

    activation_bias = sbm.alloc_stack((_pmax, 1), dtype=nl.float32, name="activation_bias", align=SBUF_QUADRANT_SIZE)
    nisa.memset(activation_bias, value=0)

    # ── STATIC_MX top-level setup ──
    # Pack per-expert quant constants into two SBUF tables, broadcast across all 128 partitions.
    # Per-block: one scalar_offset gather per table — gup pulls [_pmax, 3], down pulls [_pmax, 2].
    #   gup_scale_lut_sb [_pmax, E, 3]: [1/in_scale[e], in_scale[e]*gate_w_scale[e], in_scale[e]*up_w_scale[e]]
    #   down_scale_lut_sb [_pmax, E, 2]: [1/down_in_scale[e],   down_in_scale[e]*down_w_scale[e]]
    gup_scale_lut_sb = None
    down_scale_lut_sb = None
    if is_static_quant:
        # ── gup table: [_pmax, E, 3] ──
        gup_scale_lut_sb = sbm.alloc_stack(
            (_pmax, dims.E, 3), dtype=nl.float32, name="gup_scale_lut_sb", align=SBUF_QUADRANT_SIZE
        )
        gup_p0 = gup_scale_lut_sb.slice(dim=0, start=0, end=1)  # [1, E, 3]
        gup_slot0 = gup_p0.slice(dim=2, start=0, end=1)  # [1, E, 1] = in_scale per expert
        gup_slot1 = gup_p0.slice(dim=2, start=1, end=2)  # gate w_scale → gate combined (after fuse)
        gup_slot2 = gup_p0.slice(dim=2, start=2, end=3)  # up w_scale → up combined (after fuse)
        gup_slots12 = gup_p0.slice(dim=2, start=1, end=3)  # gate+up slots, fused together below
        gup_slot0_bcast2 = gup_p0.slice(dim=2, start=0, end=1).broadcast(dim=2, size=2)

        # Load per-expert in_scale into slot 0; gate_w_scale into slot 1, up_w_scale into slot 2.
        nisa.dma_copy(
            dst=gup_slot0,
            src=quant_params.gate_up_in_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=gup_slot1,
            src=quant_params.gate_w_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=gup_slot2,
            src=quant_params.up_w_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        # combined = w_scale * in_scale on slots 1 and 2, (broadcast slot 0 across the 2 inner slots).
        nisa.tensor_tensor(dst=gup_slots12, data1=gup_slots12, data2=gup_slot0_bcast2, op=nl.multiply)
        # Replace slot 0 with reciprocal (1 / in_scale[e]) for the bf16→fp8 quant tensor_scalar.
        nisa.reciprocal(dst=gup_slot0, data=gup_slot0)
        # Broadcast to all 128 partitions
        gup_table_flat = gup_scale_lut_sb.reshape((_pmax, dims.E * 3))
        stream_shuffle_broadcast(src=gup_table_flat, dst=gup_table_flat)

        # ── down table: [_pmax, E, 2] ──
        down_scale_lut_sb = sbm.alloc_stack(
            (_pmax, dims.E, 2), dtype=nl.float32, name="down_scale_lut_sb", align=SBUF_QUADRANT_SIZE
        )
        down_p0 = down_scale_lut_sb.slice(dim=0, start=0, end=1)  # [1, E, 2]
        down_slot0 = down_p0.slice(dim=2, start=0, end=1)
        down_slot1 = down_p0.slice(dim=2, start=1, end=2)

        nisa.dma_copy(
            dst=down_slot0,
            src=quant_params.down_in_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        nisa.dma_copy(
            dst=down_slot1,
            src=quant_params.down_w_scale.reshape((1, dims.E, 1)),
            dge_mode=nisa.dge_mode.hwdge,
        )
        # Slot 1 ← in_scale[e] * w_scale[e].
        nisa.tensor_tensor(dst=down_slot1, data1=down_slot1, data2=down_slot0, op=nl.multiply)
        # Slot 0 ← 1 / in_scale[e] for intermediate fp8 quant.
        nisa.reciprocal(dst=down_slot0, data=down_slot0)
        # Broadcast to all 128 partitions.
        down_table_flat = down_scale_lut_sb.reshape((_pmax, dims.E * 2))
        stream_shuffle_broadcast(src=down_table_flat, dst=down_table_flat)

    # Hoist [0, 1, 2, 3] H-fold offset vector once for all blocks. Used for calculated hidden state indices
    arange_4H = sbm.alloc_stack((1, _q_width), dtype=nl.float32, name="arange_4H", align=SBUF_QUADRANT_SIZE)
    nisa.iota(arange_4H, [[1, _q_width]], offset=0)

    # fp8 path keeps the raw concat [T, H_concat] view; the gather helper does its own .ap().
    # bf16 path reshapes to the H-folded layout consumed by load_hidden_states_mx.
    _hidden_states_view = (
        hidden_states if is_fp8_hidden else hidden_states.reshape((T, _q_width, prj_cfg.n_H512_tile, _pmax))
    )

    inps = InputTensors(
        hidden_states=_hidden_states_view,
        gate_up_proj_weight=gate_up_proj_weight,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        down_proj_weight=down_proj_weight,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
        p_gup_idx_vector=p_gup_idx_vector,
        p_gup_idx_vector_int32=p_gup_idx_vector_int32,
        p_down_idx_vector=p_down_idx_vector,
        gup_scales_sb=gup_scales_sb,
        activation_bias=activation_bias,
        conditions=conditions,
        arange_4H=arange_4H,
        gup_scale_lut_sb=gup_scale_lut_sb,
        down_scale_lut_sb=down_scale_lut_sb,
    )

    # Full-persistent gup scheme: only available if SBUF budget alllows.
    # In that case, the persistent gup buffer holds the full (gate+up, full I) weights across blocks, and the cross-block
    # prefetch OOB-skips on same-expert blocks. All other configs keep today's per-tile streaming.
    _gup_full_persistent = skip_dma.skip_weight and dims.I > _I_TILE_SZ

    # If the largest SBUF buffers — persistent (full-persistent gup, down weight,
    # hidden_qtz, hidden pre/post-transpose double buffers, block_old) plus the
    # biggest per-block coexisting buffer (dp_out_sb) would exceed 85% of
    # per-partition SBUF, fall back to per-tile streaming so the remaining allocs don't OOM.
    if _gup_full_persistent:
        _gup_full_bytes = 2 * prj_cfg.n_H512_tile_sharded * dims.I * sizeinbytes(gate_up_proj_weight.dtype)
        _down_w_bytes = prj_cfg.n_total_I512_tile * prj_cfg.H_sharded * sizeinbytes(down_proj_weight.dtype)
        _hidden_qtz_bytes = prj_cfg.n_H512_tile * dims.B * sizeinbytes(nl.float8_e4m3fn_x4)
        _block_hs_bytes = (dims.B // SBUF_QUADRANT_SIZE) * prj_cfg.n_H512_tile * _pmax * sizeinbytes(compute_dtype)
        _block_old_bytes = div_ceil(dims.B, _pmax) * dims.H * sizeinbytes(compute_dtype)
        _dp_out_bytes = 2 * dims.H * sizeinbytes(compute_dtype)
        # Persistent uint8 weight-scale buffers (gup_scales_sb + down_scale_sb). Their size varies by
        # path and, on the standard MX path, is the single largest term this estimate previously
        # omitted (it was calibrated for STATIC_MX only):
        #   - STATIC_MX: a tiny shared all-127 dummy is reused -> ~128 B.
        #   - packed:    4 H512/I512 tiles fold into one 128-wide block -> ~4x smaller.
        #   - standard:  full per-tile scales (uint8, 1 B/elt) co-resident with gup_full.
        # Count them so full-resident gup falls back to per-tile streaming when they don't fit.
        if is_static_quant:
            _scale_bytes = _pmax
        elif use_packed_scales:
            _n_packed_down = div_ceil(prj_cfg.n_total_I512_tile, SLOTS_PER_PACKED_BUFFER)
            _scale_bytes = n_packed_gup * 2 * dims.I + _n_packed_down * prj_cfg.H_sharded
        else:
            _scale_bytes = 2 * prj_cfg.n_H512_tile_sharded * dims.I + prj_cfg.n_total_I512_tile * prj_cfg.H_sharded
        _big_bufs_bytes = (
            _gup_full_bytes
            + _down_w_bytes
            + _hidden_qtz_bytes
            + 2 * _block_hs_bytes
            + _block_old_bytes
            + _dp_out_bytes
            + _scale_bytes
        )
        _sbuf_threshold = (nl.tile_size.total_available_sbuf_size * 85) // 100
        if _big_bufs_bytes > _sbuf_threshold:
            logger.info(
                f"Disabling gup_full_persistent: big buffers={_big_bufs_bytes} B > 85% per-partition SBUF "
                f"({_sbuf_threshold} B). Falling back to per-tile-0 weight skipping."
            )
            _gup_full_persistent = False

    configs = BWMMMXConfigs(
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        scaling_mode=expert_affinities_scaling_mode,
        weight_dtype=gate_up_proj_weight.dtype,
        io_dtype=hidden_states.dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        use_dynamic_while=conditions != None,
        n_static_blocks=n_static_blocks,
        linear_bias=(gate_and_up_proj_bias != None and down_proj_bias != None),
        activation_function=activation_function,
        fuse_gate_and_up_load=True,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        qtz_dtype=nl.float8_e4m3fn_x4,
        use_packed_scales=use_packed_scales,
        is_static_quant=is_static_quant,
        has_gate_clamp=has_gate_clamp,
        gup_full_persistent=_gup_full_persistent,
        is_fp8_hidden=is_fp8_hidden,
        is_affinities_packed=is_affinities_packed,
        affinities_col_offset=affinities_col_offset,
    )

    check_kernel_compatibility(dims, configs)

    # Output is the dequantized result, not fp8. For pre-quantized fp8 hidden, hidden_states.dtype
    # is fp8, so use compute_dtype (bf16) for the output instead of mirroring the input dtype.
    output_dtype = compute_dtype if is_fp8_hidden else hidden_states.dtype
    if is_tensor_update_accumulating:
        output = nl.ndarray((2, dims.T, dims.H), dtype=output_dtype, buffer=nl.shared_hbm)
    else:
        output = nl.ndarray((dims.T, dims.H), dtype=output_dtype, buffer=nl.shared_hbm)

    outs = OutputTensors(
        gate_up_activations_T=gate_up_activations_T,
        down_activations=down_activations,
        output=output,
    )
    """
    Allocate buffers for prefetching and current block processing.
    
    hidden_sbuf_expected_shape defines the expected layout for hidden states:
    (32_B * 4_H, dims.B // 32, mx4_prj_cfg.n_H512_tile, 128_H)
    """

    token_4_H_indices_on_p = sbm.alloc_stack(
        (_pmax, dims.B // SBUF_QUADRANT_SIZE),
        dtype=nl.int32,
        name="token_4_H_indices_on_p",
        align=SBUF_QUADRANT_SIZE,
    )

    hidden_qtz_sb = sbm.alloc_stack(
        (_pmax, prj_cfg.n_H512_tile, dims.B // SBUF_QUADRANT_SIZE, SBUF_QUADRANT_SIZE),
        dtype=configs.qtz_dtype,
        name="hidden_qtz_sb",
        align=SBUF_QUADRANT_SIZE,
    )
    if is_static_quant:
        # STATIC_MX: reuse the shared all-127 dummy [_pmax, _pmax] buffer.
        hidden_scale_sb = static_dummy_scale_sb
    else:
        # fp8 pre-quantized path: scales arrive packed (4 H512 tiles folded into one 128-wide block),
        # so the scale buffer holds n_packed blocks instead of n_H512_tile. The online-quant bf16 path
        # keeps one scale tile per H512 tile.
        n_scale_dim = div_ceil(prj_cfg.n_H512_tile, 4) if is_fp8_hidden else prj_cfg.n_H512_tile
        hidden_scale_sb = sbm.alloc_stack(
            (_pmax, n_scale_dim, dims.B // SBUF_QUADRANT_SIZE, SBUF_QUADRANT_SIZE),
            dtype=nl.uint8,
            name="hidden_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )

    # fp8 pre-quantized path: persistent buffer holding one block's gathered concat rows
    # ([hidden_quant (H fp8) | hidden_scale (packed: n_packed*128 uint8)] [| affinities (E bf16) | pad]
    # when affinities are fused) between the prefetch (DMA) and transpose (PE) stages. H and the packed
    # scale region are both multiples of 4, so the row is fp32-aligned for the transpose's fp8->fp32
    # reinterpret; the affinity tail (when present) is the full remaining concat width.
    block_hidden_concat = None
    if is_fp8_hidden:
        # Hold the FULL concat row so the gather pulls the (optional) affinity tail too.
        concat_free_size = hidden_states.shape[-1]
        block_hidden_concat = sbm.alloc_stack(
            (_pmax, dims.B // _pmax, concat_free_size),
            dtype=hidden_states.dtype,
            name="block_hidden_concat",
            align=SBUF_QUADRANT_SIZE,
        )

        if is_affinities_packed and skip_dma.skip_token:
            nisa.memset(block_hidden_concat[:, :, affinities_col_offset:concat_free_size], value=0)

    block_old = sbm.alloc_stack(
        (_pmax, dims.n_B128_tiles, dims.H), dtype=configs.compute_dtype, name="block_old", align=SBUF_QUADRANT_SIZE
    )
    if skip_dma.skip_token:
        nisa.memset(block_old[0:_pmax, : dims.n_B128_tiles, 0:H], value=0)

    down_weight_qtz = sbm.alloc_stack(
        (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded),
        dtype=inps.down_proj_weight.dtype,
        name="down_weight_qtz",
        align=SBUF_QUADRANT_SIZE,
    )

    # Memset weight if input weight HBM does not pad on par dim
    if dims.p_I != _pmax:
        nisa.memset(down_weight_qtz[:, prj_cfg.n_total_I512_tile - 1, :], value=0)

    if is_static_quant:
        # STATIC_MX: dummy 127 moving_scale for down matmul. Per-tile matmul
        # reads [_pmax, H_tile_size] per call. Caller's matmul site slices 2D under static_quant instead of 3D with n_I512 tiles.
        # n_packed_down only used by MX path; keep unset under STATIC_MX (use_packed_scales=False enforced).
        n_packed_down = 0
        down_scale_sb = sbm.alloc_stack(
            (_pmax, prj_cfg.H_tile_size),
            dtype=nl.uint8,
            name="down_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        nisa.memset(down_scale_sb, value=127)
    elif use_packed_scales:
        n_packed_down = (prj_cfg.n_total_I512_tile + SLOTS_PER_PACKED_BUFFER - 1) // SLOTS_PER_PACKED_BUFFER
        down_scale_sb = sbm.alloc_stack(
            (_pmax, n_packed_down, prj_cfg.H_sharded),
            dtype=nl.uint8,
            name="down_scale_packed_sb",
            align=SBUF_QUADRANT_SIZE,
        )
    else:
        n_packed_down = 0
        down_scale_sb = sbm.alloc_stack(
            (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded),
            dtype=nl.uint8,
            name="down_scale_sb",
            align=SBUF_QUADRANT_SIZE,
        )
        if dims.p_I != _pmax:
            nisa.memset(down_scale_sb[:, prj_cfg.n_total_I512_tile - 1, :], value=0)

    # STATIC_MX: dummy 127 stationary_scale for the down matmul intermediate.
    # Reuse the shared [_pmax, _pmax] all-127 dummy buffer (gup_scales_sb /
    # hidden_scale_sb point to the same backing). Matmul site reads as a 2D
    # per-tile view under static_quant.
    dummy_inter_scale_sb = static_dummy_scale_sb if is_static_quant else None

    # init counters
    # in shard-on-block we can move independently
    cond = (
        sbm.alloc_stack((1, 1), dtype=nl.int32, name="cond", align=SBUF_QUADRANT_SIZE)
        if configs.use_dynamic_while
        else None
    )
    index = (
        sbm.alloc_stack((1, 1), dtype=nl.int32, name="index", align=SBUF_QUADRANT_SIZE)
        if configs.use_dynamic_while
        else None
    )

    # Pre-allocate persistent gup tile buffer for cross-block prefetching.
    # Two layouts:
    #   - configs.gup_full_persistent: hold full gup weights (gate+up, full I)
    #     persistent across blocks. Cross-block prefetch loads the entire gup; same-expert
    #     blocks OOB-skip the prefetch Per-I-tile compute reads slices directly from this persistent buffer (no per-tile streaming).
    #   - Otherwise: small "tile 0 prefetch" buffer; per-I-tile compute streams
    #     subsequent tiles via gup_wt_a/gup_wt_b ping-pong.
    gup_tile_prefetch_buf = None
    if dims.I > _I_TILE_SZ:
        if configs.gup_full_persistent:
            gup_tile_prefetch_buf = sbm.alloc_stack(
                (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
                dtype=inps.gate_up_proj_weight.dtype,
                name="gup_full_persistent",
                align=SBUF_QUADRANT_SIZE,
            )
        else:
            gup_tile_prefetch_buf = sbm.alloc_stack(
                (_pmax, 2, prj_cfg.n_H512_tile_sharded, _I_TILE_SZ),
                dtype=inps.gate_up_proj_weight.dtype,
                name="gup_tile_prefetch",
                align=SBUF_QUADRANT_SIZE,
            )

    buffers = SharedBuffers(
        block_hidden_states=None,
        block_hidden_states_T=None,
        hidden_qtz_sb=hidden_qtz_sb,
        hidden_scale_sb=hidden_scale_sb,
        block_old=block_old,
        down_weight_qtz=down_weight_qtz,
        down_scale_sb=down_scale_sb,
        cond=cond,
        index=index,
        token_4_H_indices_on_p=token_4_H_indices_on_p,
        gup_tile_buf_a=gup_tile_prefetch_buf,
        dummy_inter_scale_sb=dummy_inter_scale_sb,
        block_hidden_concat=block_hidden_concat,
    )

    """
    END OF PREPARING DIMS, CONFIGS, SHARED_BUFFERS

    MAIN COMPUTATION STARTS
    """

    """Weight skipping: pre-compute skip mask and hoist buffers."""
    # When tiling is active, gup_tile_prefetch_buf is allocated. In tiled path we
    # don't hoist full weight buffers (weights are tile-loaded per I-tile) —
    # skip_weight hoists scales, bias, and down weights for OOB-skip reuse.
    _tiling_active = gup_tile_prefetch_buf != None
    if skip_dma.skip_weight:
        # Determine the actual number of static blocks that process_static_blocks will iterate over, so the skip mask matches the block-to-shard assignment.
        if configs.use_dynamic_while:
            if configs.n_static_blocks > 0:
                _n_dyn = dims.N - configs.n_static_blocks
                # make it even number of blocks
                _n_dyn = _n_dyn + 1 if _n_dyn % 2 == 1 else _n_dyn
                _num_static_for_mask = dims.N - _n_dyn
            else:
                if conditions != None and (n_dynamic_blocks < 0 or n_dynamic_blocks > dims.N):
                    # Must match the auto-computed n_static_blocks below so the skip
                    # mask aligns with the block-to-shard assignment.
                    _num_static_for_mask = max(1, div_ceil(div_ceil(dims.T * top_k, ep_degree), dims.B))
                else:
                    _num_static_for_mask = dims.N - n_dynamic_blocks
        else:
            _num_static_for_mask = N

        if _num_static_for_mask > 0:
            """
            "Remainder block" (odd-N only): when the number of static blocks is odd,
            blocks don't divide evenly across the 2 shards. After both shards finish
            their balanced loops, one block (index _num_static_for_mask - 1) is left
            over. Shard 1 processes it for real; shard 0 runs a dummy (output zeroed).

            Shard 1's last loop iteration speculatively prefetches weights for that
            remainder block. The skip-mask compares the remainder's expert against
            the previous block's expert — so we need the remainder expert in the +1
            tail slot for that specific case.

            All other cases (even-N, or shard 0 odd-N) never meaningfully consume
            the tail. We leave it as E (memset) so any speculative prefetch is
            skipped, saving a DMA.
            """
            n_blocks_this_shard = div_ceil(max(_num_static_for_mask - dims.shard_id, 0), dims.num_shards)
            first_block = (_num_static_for_mask // dims.num_shards) * dims.shard_id
            n_mask_len = n_blocks_this_shard + 1 if n_blocks_this_shard > 0 else 0
            n_blocks_per_shard_alloc = div_ceil(_num_static_for_mask, dims.num_shards) + 1

            all_experts = sbm.alloc_stack((1, n_blocks_per_shard_alloc), dtype=nl.int32, name="all_experts")
            nisa.memset(dst=all_experts, value=E)
            if n_blocks_this_shard > 0:
                nisa.dma_copy(
                    dst=all_experts[0:1, 0:n_blocks_this_shard],
                    src=block_to_expert.reshape((N, 1)).ap(
                        pattern=[[1, 1], [1, n_blocks_this_shard]], offset=first_block
                    ),
                )
                # Only odd-N shard 1 consumes the tail (for remainder-block prefetch).
                if dims.shard_id == 1 and _num_static_for_mask % 2 == 1:
                    nisa.dma_copy(
                        dst=all_experts[0:1, n_blocks_this_shard : n_blocks_this_shard + 1],
                        src=block_to_expert.reshape((N, 1)).ap(
                            pattern=[[1, 1], [1, 1]],
                            offset=_num_static_for_mask - 1,
                        ),
                    )

            # Build weight-expert array: E (skip) where same expert as previous block.
            # Non-tiling: modify all_experts in-place (original not needed after).
            # Tiling: need original mapping for gup weight loads
            if _tiling_active:
                all_experts_for_weights = sbm.alloc_stack(
                    (1, n_blocks_per_shard_alloc), dtype=nl.int32, name="all_experts_for_weights"
                )
                nisa.tensor_copy(dst=all_experts_for_weights, src=all_experts)
                if n_mask_len > 1:
                    _compute_weight_skip_mask(sbm, all_experts, all_experts_for_weights, n_mask_len, E)
            else:
                all_experts_for_weights = all_experts
                if n_mask_len > 1:
                    _compute_weight_skip_mask(sbm, all_experts, all_experts, n_mask_len, E)
        else:
            all_experts_for_weights = None

        # Hoist weight/bias buffers so they persist across iterations. In tiled path, full gup weights don't persist (tile-loaded per I-tile) so we
        # only hoist scales, bias, and down weights for OOB-skip reuse.
        if _tiling_active:
            logger.info("Weight skipping on tiling path: only skipping scales, bias, and down weight")
            hoisted_gup_weights = None
            # Hoist bias buffers so bias DMAs can be OOB-skipped on same-expert blocks
            # (weights stay in tile buffers and are reloaded per I-tile). Conditionally allocate based on if there is a bias a or not.
            hoisted_gup_bias = (
                sbm.alloc_stack(
                    (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                    dtype=inps.gate_and_up_proj_bias.dtype,
                    name="hoisted_gup_bias",
                    align=SBUF_QUADRANT_SIZE,
                )
                if gate_and_up_proj_bias != None
                else None
            )
            if hoisted_gup_bias != None and dims.I < _pmax * _q_width:
                # hoisted_gup_bias: [128_I, 2, n_total_I512, 4_I]
                nisa.memset(dst=hoisted_gup_bias[:, :, 0, :], value=0.0)

            hoisted_down_bias = (
                sbm.alloc_stack(
                    (1, dims.H),
                    dtype=inps.down_proj_bias.dtype,
                    name="hoisted_down_bias",
                    align=SBUF_QUADRANT_SIZE,
                )
                if down_proj_bias != None
                else None
            )
        else:
            hoisted_gup_weights = sbm.alloc_stack(
                (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
                dtype=inps.gate_up_proj_weight.dtype,
                name="hoisted_gup_weights",
                align=SBUF_QUADRANT_SIZE,
            )
            hoisted_gup_bias = (
                sbm.alloc_stack(
                    (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                    dtype=inps.gate_and_up_proj_bias.dtype,
                    name="hoisted_gup_bias",
                    align=SBUF_QUADRANT_SIZE,
                )
                if gate_and_up_proj_bias != None
                else None
            )
            if hoisted_gup_bias != None and dims.I < _pmax * _q_width:
                nisa.memset(dst=hoisted_gup_bias[:, :, 0, :], value=0.0)
            hoisted_down_bias = (
                sbm.alloc_stack(
                    (1, dims.H),
                    dtype=inps.down_proj_bias.dtype,
                    name="hoisted_down_bias",
                    align=SBUF_QUADRANT_SIZE,
                )
                if down_proj_bias != None
                else None
            )
    else:
        # not doing any skip_weight
        all_experts_for_weights = None
        hoisted_gup_weights = None
        hoisted_gup_bias = None
        hoisted_down_bias = None

    if configs.use_dynamic_while:
        if configs.n_static_blocks > 0:
            kernel_assert(
                configs.n_static_blocks < dims.N,
                f"Cannot have more static blocks than total number of blocks. Got ({configs.n_static_blocks}) > N = {dims.N}",
            )
            n_dynamic_blocks = dims.N - configs.n_static_blocks
            n_dynamic_blocks = n_dynamic_blocks + 1 if n_dynamic_blocks % 2 == 1 else n_dynamic_blocks
            n_static_blocks = dims.N - n_dynamic_blocks
            n_dynamic_blocks_local = n_dynamic_blocks
            auto_computed = False
        else:
            # If invalid n_dynamic_blocks is passed, auto-calculate best case combination.
            # Always process at least one static block
            if conditions != None and (n_dynamic_blocks < 0 or n_dynamic_blocks > dims.N):
                # Best case: tokens spread top_k ways, sharded across ep_degree EP ranks,
                # packed into B-sized blocks. ceil(ceil(T*top_k/ep_degree)/B).
                n_static_blocks = max(1, div_ceil(div_ceil(dims.T * top_k, ep_degree), dims.B))
                n_dynamic_blocks_local = dims.N - n_static_blocks
                auto_computed = True
            else:
                n_dynamic_blocks_local = n_dynamic_blocks
                n_static_blocks = dims.N - n_dynamic_blocks_local
                auto_computed = False

        # process_dynamic_blocks steps the block index by num_shards, so it cannot
        # process fewer than num_shards blocks. Push residual into the static path,
        # which has an odd-N branch that handles N < num_shards via is_dummy.
        if 0 < n_dynamic_blocks_local < dims.num_shards:
            n_static_blocks += n_dynamic_blocks_local
            n_dynamic_blocks_local = 0

        if configs.n_static_blocks <= 0 and auto_computed:
            logger.info(
                f"n_dynamic_blocks={n_dynamic_blocks} out of range, auto-computing from T={dims.T}, B={dims.B}: "
                f"{n_static_blocks} static, {n_dynamic_blocks_local} dynamic"
            )

        # base index for dynamic blocks to start
        nisa.memset(dst=buffers.index[0, 0], value=n_static_blocks)

        logger.info(f"Processing {n_static_blocks} static blocks, {n_dynamic_blocks_local} dynamic blocks")
        # When n_static_blocks==0, process_static_blocks is skipped but output_initialization
        # (which zeros the output tensor for accumulation) lives inside it. Initialize here
        # so dynamic-only paths don't read uninitialized shared DRAM.
        if n_static_blocks == 0 and configs.is_tensor_update_accumulating:
            H = dims.H
            zeros = sbm.alloc_heap(
                (_pmax, H), dtype=nl.bfloat16, name="output_init_zeros_dyn", align=SBUF_QUADRANT_SIZE
            )
            nisa.memset(zeros, value=0.0)
            output_initialization(outs.output, dims, sbm=sbm, zeros=zeros)
            sbm.pop_heap()  # free zeros

        if n_static_blocks > 0:
            process_static_blocks(
                dims=dims,
                configs=configs,
                prj_cfg=prj_cfg,
                inps=inps,
                outs=outs,
                buffers=buffers,
                num_static_blocks=n_static_blocks,
                sbm=sbm,
                all_experts_for_weights=all_experts_for_weights,
                hoisted_gup_weights=hoisted_gup_weights,
                hoisted_gup_bias=hoisted_gup_bias,
                hoisted_down_bias=hoisted_down_bias,
                is_tensor_update_accumulating=configs.is_tensor_update_accumulating,
            )

        if n_dynamic_blocks_local > 0:
            """
            Precompute dynamic-for iteration counts at top level
            
            Chunked scheme:
              n_outer_iters = n_active // _DYN_STEP   (each outer iter = _DYN_STEP blocks = _DYN_INNER/core)
              n_half        = (n_active + 1) >> 1     (what total ping-pong iters would be)
              n_rem_iters   = n_half - n_outer_iters * _DYN_INNER
            Each outer iter consumes _DYN_INNER ping-pong iters per shard; remainder
            loop handles the rest (0 .. _DYN_INNER-1 iters).
            """
            dyn_conds_sb = sbm.alloc_stack(
                (1, n_dynamic_blocks_local), dtype=nl.int32, name="dyn_conds", align=SBUF_QUADRANT_SIZE
            )
            nisa.dma_copy(
                dst=dyn_conds_sb,
                src=inps.conditions.reshape((inps.conditions.shape[0], 1)).ap(
                    pattern=[[1, 1], [1, n_dynamic_blocks_local]], offset=n_static_blocks
                ),
            )
            # number of active dynamic blocks at runtime
            n_active_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_active", align=SBUF_QUADRANT_SIZE)
            nisa.tensor_reduce(dst=n_active_sb, data=dyn_conds_sb, op=nl.add, axis=1)

            # n_outer_iters = n_active // _DYN_STEP (shift-by-log2 since _DYN_STEP is a power of 2)
            n_outer_iters_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_outer_iters", align=SBUF_QUADRANT_SIZE)

            # floor division by _DYN_STEP
            nisa.tensor_scalar(
                dst=n_outer_iters_sb,
                data=n_active_sb,
                op0=nl.right_shift,
                operand0=_DYN_STEP_LOG2,
            )

            # n_half = (n_active + 1) >> 1 (total number of ping pong iterations needed in dynamic blocks path)
            n_half_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_half", align=SBUF_QUADRANT_SIZE)
            nisa.tensor_scalar(dst=n_half_sb, data=n_active_sb, op0=nl.add, operand0=1)
            nisa.tensor_scalar(dst=n_half_sb, data=n_half_sb, op0=nl.right_shift, operand0=1)

            # n_rem_iters = n_half - n_outer_iters * _DYN_INNER
            n_outer_scaled_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_outer_scaled", align=SBUF_QUADRANT_SIZE)
            nisa.tensor_scalar(dst=n_outer_scaled_sb, data=n_outer_iters_sb, op0=nl.multiply, operand0=_DYN_INNER)
            n_rem_iters_sb = sbm.alloc_stack((1, 1), dtype=nl.int32, name="n_rem_iters", align=SBUF_QUADRANT_SIZE)
            nisa.tensor_tensor(dst=n_rem_iters_sb, data1=n_half_sb, data2=n_outer_scaled_sb, op=nl.subtract)

            outer_reg = nisa.register_alloc()
            nisa.register_load(outer_reg, n_outer_iters_sb)
            rem_reg = nisa.register_alloc()
            nisa.register_load(rem_reg, n_rem_iters_sb)

            """
            Weight skipping for dynamic blocks: build two independent shard-local
            masks, one for the chunked region and one for the remainder region.
            Each mask gets its own _compute_weight_skip_mask pass, so position 0
            of each mask holds the raw expert (no predecessor to compare against)
            and the first block of each region always loads weights fresh. The
            chunked->rem boundary is therefore handled implicitly without any
            runtime seam comparison.
            """
            _use_dyn_weight_skip = configs.skip_dma.skip_weight
            chunked_experts_mask = None
            rem_experts_mask = None
            dyn_weight_expert_sb = None
            chunked_shard_iter_sb = None
            rem_shard_iter_sb = None
            n_chunked_alloc = 0
            n_rem_alloc = 0
            if _use_dyn_weight_skip:
                # Worst-case sizing (compile-time).
                # Chunked: (n_dynamic_blocks_local // _DYN_STEP) * _DYN_INNER blocks per shard.
                n_chunks_max = n_dynamic_blocks_local // _DYN_STEP
                n_chunked_alloc = n_chunks_max * _DYN_INNER
                # Remainder: after chunking, at most (_DYN_STEP - 1) active blocks remain.
                # Per shard: ceil(min(n_dynamic_blocks_local, _DYN_STEP - 1) / num_shards).
                n_rem_alloc = div_ceil(min(n_dynamic_blocks_local, _DYN_STEP - 1), dims.num_shards)

                # --- Chunked mask ---
                if n_chunked_alloc > 0:
                    chunked_experts_mask = _sbm_alloc(
                        sbm,
                        (1, n_chunked_alloc),
                        dtype=nl.int32,
                        name="chunked_experts_mask",
                        align=SBUF_QUADRANT_SIZE,
                    )
                    nisa.memset(dst=chunked_experts_mask, value=dims.E)

                    sbm.open_scope(name="chunked_weight_skip_mask")
                    prev_prefix = sbm.get_name_prefix()
                    sbm.set_name_prefix(f"{prev_prefix}chunk_")

                    # Shard s's chunked block at position k (0..n_chunked_alloc-1) is:
                    #   ns + (k // _DYN_INNER) * _DYN_STEP + s * _DYN_INNER + (k % _DYN_INNER)
                    # DMA pattern: outer-chunk stride _DYN_STEP × inner stride 1.
                    nisa.dma_copy(
                        dst=chunked_experts_mask[0:1, 0:n_chunked_alloc],
                        src=inps.block_to_expert.reshape((1, dims.N)).ap(
                            pattern=[
                                [1, 1],
                                [_DYN_STEP, n_chunks_max],
                                [1, _DYN_INNER],
                            ],
                            offset=n_static_blocks + dims.shard_id * _DYN_INNER,
                        ),
                    )

                    if n_chunked_alloc > 1:
                        if _tiling_active:
                            chunked_all_experts = _sbm_alloc(
                                sbm,
                                (1, n_chunked_alloc),
                                dtype=nl.int32,
                                name="chunked_all_experts",
                                align=SBUF_QUADRANT_SIZE,
                            )
                            nisa.tensor_copy(dst=chunked_all_experts, src=chunked_experts_mask)
                            _compute_weight_skip_mask(
                                sbm, chunked_all_experts, chunked_experts_mask, n_chunked_alloc, dims.E
                            )
                        else:
                            _compute_weight_skip_mask(
                                sbm, chunked_experts_mask, chunked_experts_mask, n_chunked_alloc, dims.E
                            )

                    sbm.set_name_prefix(prev_prefix)
                    sbm.close_scope()

                # --- Remainder mask ---
                """
                The remainder region starts at a runtime-computed offset
                (n_static_blocks + n_outer_iters * _DYN_STEP). DMA-load using a
                scalar_offset derived from n_outer_iters at runtime.
                Allocate one extra sentinel slot at the tail so the prefetch's
                next-expert lookup at the last rem iter (rem_shard_iter+1
                walking past n_rem_alloc - 1) reads E (skip) safely. The
                prefetch result at the last rem iter is never consumed
                (next_block_idx is clamped to N-1 with no further compute), so
                the sentinel value is functionally a don't-care.
                """
                # Records how many rem iters the rem_base clamp consumed,
                # so the rem loop's iter counters start at that shift and
                # iter 0 indexes the first real rem-region block.
                rem_iter_shift_sb = None
                if n_rem_alloc > 0:
                    rem_experts_mask = _sbm_alloc(
                        sbm,
                        (1, n_rem_alloc + 1),
                        dtype=nl.int32,
                        name="rem_experts_mask",
                        align=SBUF_QUADRANT_SIZE,
                    )
                    nisa.memset(dst=rem_experts_mask, value=dims.E)
                    rem_iter_shift_sb = _sbm_alloc(
                        sbm,
                        (1, 1),
                        dtype=nl.uint32,
                        name="rem_iter_shift",
                        align=SBUF_QUADRANT_SIZE,
                    )

                    sbm.open_scope(name="rem_weight_skip_mask")
                    prev_prefix = sbm.get_name_prefix()
                    sbm.set_name_prefix(f"{prev_prefix}rem_")

                    # rem_base_sb (uint32) = n_static_blocks + shard_id + n_outer_iters * _DYN_STEP
                    rem_base_sb = _sbm_alloc(sbm, (1, 1), dtype=nl.uint32, name="rem_base", align=SBUF_QUADRANT_SIZE)
                    nisa.tensor_scalar(
                        dst=rem_base_sb,
                        data=n_outer_iters_sb,
                        op0=nl.multiply,
                        operand0=_DYN_STEP,
                        op1=nl.add,
                        operand1=n_static_blocks + dims.shard_id,
                    )

                    """
                    The mask DMA reads n_rem_alloc entries strided by num_shards starting at rem_base. If any lane goes past N-1, oob_mode.skip
                    aborts the whole DMA — so clamp rem_base down to a safe start. When clamping, we must land on a block this shard actually
                    processes (every num_shards-th block starting from n_static_blocks + shard_id), otherwise the mask records experts for the
                    wrong neighbor blocks and weight-skip decisions go stale.
                    """
                    _max_safe_base = dims.N - 1 - (n_rem_alloc - 1) * dims.num_shards
                    _rem_base_parity = (n_static_blocks + dims.shard_id) % dims.num_shards
                    _max_safe_base_aligned = _max_safe_base - ((_max_safe_base - _rem_base_parity) % dims.num_shards)
                    # rem_iter_shift_sb tracks how many iters the clamp consumed,
                    # so iter 0 of the rem loop still indexes the first real rem block.
                    rem_clamped_base_sb = _sbm_alloc(
                        sbm,
                        (1, 1),
                        dtype=nl.uint32,
                        name="rem_clamped_base",
                        align=SBUF_QUADRANT_SIZE,
                    )
                    nisa.tensor_scalar(
                        dst=rem_clamped_base_sb,
                        data=rem_base_sb,
                        op0=nl.minimum,
                        operand0=_max_safe_base_aligned,
                    )
                    # rem_iter_shift_sb = (rem_base - clamped_base) / num_shards
                    nisa.tensor_tensor(
                        dst=rem_iter_shift_sb,
                        data1=rem_base_sb,
                        data2=rem_clamped_base_sb,
                        op=nl.subtract,
                    )
                    # num_shards == 2; divide by 2 via right shift by 1.
                    nisa.tensor_scalar(
                        dst=rem_iter_shift_sb,
                        data=rem_iter_shift_sb,
                        op0=nl.right_shift,
                        operand0=1,
                    )

                    nisa.dma_copy(
                        dst=rem_experts_mask[0:1, 0:n_rem_alloc],
                        src=inps.block_to_expert.ap(
                            pattern=[
                                [1, 1],
                                [dims.num_shards, n_rem_alloc],
                            ],
                            offset=0,
                            scalar_offset=rem_clamped_base_sb,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip,
                    )

                    if n_rem_alloc > 1:
                        # Mirror the +1 sentinel slot in the tiling shadow
                        # buffer so tensor_copy shapes match. Skip-mask range
                        # stays n_rem_alloc; the sentinel slot keeps its E.
                        if _tiling_active:
                            rem_all_experts = _sbm_alloc(
                                sbm,
                                (1, n_rem_alloc + 1),
                                dtype=nl.int32,
                                name="rem_all_experts",
                                align=SBUF_QUADRANT_SIZE,
                            )
                            nisa.tensor_copy(dst=rem_all_experts, src=rem_experts_mask)
                            _compute_weight_skip_mask(sbm, rem_all_experts, rem_experts_mask, n_rem_alloc, dims.E)
                        else:
                            _compute_weight_skip_mask(sbm, rem_experts_mask, rem_experts_mask, n_rem_alloc, dims.E)

                    sbm.set_name_prefix(prev_prefix)
                    sbm.close_scope()

                # Per-block scratch for current weight-expert lookup.
                dyn_weight_expert_sb = _sbm_alloc(
                    sbm, (1, 1), dtype=nl.int32, name="dyn_weight_expert_sb", align=SBUF_QUADRANT_SIZE
                )
                # Shard-local iter counters for chunked and remainder masks.
                chunked_shard_iter_sb = _sbm_alloc(
                    sbm, (1, 1), dtype=nl.uint32, name="chunked_shard_iter", align=SBUF_QUADRANT_SIZE
                )
                nisa.memset(dst=chunked_shard_iter_sb, value=0)
                rem_shard_iter_sb = _sbm_alloc(
                    sbm, (1, 1), dtype=nl.uint32, name="rem_shard_iter", align=SBUF_QUADRANT_SIZE
                )
                if n_rem_alloc > 0:
                    # Init iter counter to the rem-base clamp shift so iter 0
                    # of the rem loop indexes the slot holding the actual
                    # first rem-region block's expert.
                    nisa.tensor_copy(dst=rem_shard_iter_sb, src=rem_iter_shift_sb)
                else:
                    nisa.memset(dst=rem_shard_iter_sb, value=0)

            process_dynamic_blocks(
                dims=dims,
                configs=configs,
                prj_cfg=prj_cfg,
                inps=inps,
                outs=outs,
                buffers=buffers,
                num_static_blocks=n_static_blocks,
                num_dynamic_blocks=n_dynamic_blocks_local,
                sbm=sbm,
                hoisted_gup_weights=hoisted_gup_weights,
                hoisted_gup_bias=hoisted_gup_bias,
                hoisted_down_bias=hoisted_down_bias,
                outer_reg=outer_reg,
                rem_reg=rem_reg,
                n_outer_iters_sb=n_outer_iters_sb,
                chunked_experts_mask=chunked_experts_mask,
                rem_experts_mask=rem_experts_mask,
                dyn_weight_expert_sb=dyn_weight_expert_sb,
                chunked_shard_iter_sb=chunked_shard_iter_sb,
                rem_shard_iter_sb=rem_shard_iter_sb,
                n_chunked_alloc=n_chunked_alloc,
                n_rem_alloc=n_rem_alloc,
            )

    else:
        """
        STATIC LOOP OVER ALL BLOCKS
        """
        process_static_blocks(
            dims=dims,
            configs=configs,
            prj_cfg=prj_cfg,
            inps=inps,
            outs=outs,
            buffers=buffers,
            num_static_blocks=dims.N,
            sbm=sbm,
            all_experts_for_weights=all_experts_for_weights,
            hoisted_gup_weights=hoisted_gup_weights,
            hoisted_gup_bias=hoisted_gup_bias,
            hoisted_down_bias=hoisted_down_bias,
            is_tensor_update_accumulating=configs.is_tensor_update_accumulating,
        )

    """
    Final collective to produce the final result
    """

    if dims.num_shards == 2:
        nisa.core_barrier(output, (0, 1))

    if is_tensor_update_accumulating and dims.num_shards > 1:
        kernel_assert(dims.num_shards == 2, "only support reducing data from 2 shards")
        reduce_tile_size = _pmax
        if skip_dma.skip_token:
            reduce_tiles = div_ceil(T, _pmax)
        else:
            reduce_tiles = div_ceil(T - 1, _pmax)

        nc0_tiles = reduce_tiles // dims.num_shards
        nc1_tiles = reduce_tiles - nc0_tiles

        if dims.num_shards == 2:
            nisa.core_barrier(output, (0, 1))

        if dims.shard_id == 0:
            reduce_outputs(output, nc0_tiles, reduce_tile_size, 0, H)

        if dims.shard_id == 1:
            reduce_outputs(output, nc1_tiles, reduce_tile_size, nc0_tiles, H)

    sbm.close_scope()

    return output


def load_prev_block(output, token_indices, block_old, NUM_TILES, dtype, shard_id, skip_dma: SkipMode):
    """
    Load previous block outputs for accumulation in tensor update mode.

    Retrieves existing output values for tokens in the current block to enable
    accumulation across multiple expert evaluations (topK > 1).

    Args:
        output (nl.NkiTensor): Output tensor of shape [num_shards, T, H] containing
            accumulated results from previous blocks.
        token_indices (nl.NkiTensor): Token indices for current block of shape [P_MAX, NUM_TILES].
        block_old (nl.NkiTensor): Buffer to store loaded values of shape [P_MAX, NUM_TILES, H].
        NUM_TILES (int): Number of tiles in the block (B // 128).
        dtype: Data type for loading.
        shard_id (int): Current shard identifier (0 or 1).
        skip_dma (SkipMode): DMA skip configuration for handling invalid tokens.

    Returns:
        block_old (nl.NkiTensor): Loaded previous output values for the block.

    Notes:
        - Uses indirect addressing via token_indices for gather operation
        - Skips DMA for invalid tokens when skip_dma.skip_token == True
        - Required for topK > 1 scenarios where multiple experts contribute to same token
        - Reshapes output tensor for efficient access pattern

    Pseudocode:
        H = output.shape[-1]
        T = output.shape[-2]
        num_shards = output.shape[0]
        shard_offset = shard_id * T * H
        output_reshaped = reshape output to [num_shards * T, 1, H]

        for n in range(NUM_TILES):
            block_token_mapping = token_indices[:, n]
            dma_copy output_reshaped[block_token_mapping + shard_offset, :, :] to block_old[:, n, :]

        return block_old
    """
    H = output.shape[-1]
    T = output.shape[-2]
    num_shards = output.shape[0]
    shard_offset = shard_id * T * H

    # Reshape output to (num_shards * T, 1, H) for proper AP pattern
    output_reshaped = output.reshape((num_shards * T, 1, H))

    for n in range(NUM_TILES):
        block_token_mapping = token_indices.ap(
            [[NUM_TILES, _pmax], [1, 1]],
            offset=n,
        )
        nisa.dma_copy(
            dst=block_old[:_pmax, n, :H],
            src=output_reshaped.ap(
                pattern=[[H, _pmax], [1, 1], [1, H]],
                offset=shard_offset,
                vector_offset=block_token_mapping,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
    return block_old


def _extract_block_affinity(block_hidden_concat, block_expert, dims, kernel_cfg, sbm, name_prefix="aff"):
    """Extract this block's expert-affinity column from the gathered fp8 concat rows (packed path).

    When the producer fused the dense [T, E] affinities into the row (is_affinities_packed), each
    gathered row is [hidden | scale | affinities (E bf16) | pad]. The affinity this block needs is the
    block_expert column of every token's affinity vector. This is a pure on-chip op (one tensor_copy
    with scalar_offset=block_expert) -- NOT a DMA -- so it replaces the separate indirect affinity
    gather (calculate_expert_affinities) and removes a SWDGE pass per B128 tile.

    block_hidden_concat is [_pmax, n_B_tiles, concat_free_size] fp8. The affinity region starts at fp8
    column kernel_cfg.affinities_col_offset; viewed as bf16 that is column affinities_col_offset // 2,
    and the per-token row stride in bf16 is concat_free_size // 2.

    Returns a list of n_B_tiles tensors, each [_pmax, 1] fp32 -- the same shape contract as
    calculate_expert_affinities, so the per-block compute consumes expert_affinity[n] identically.
    """
    _bf16_as_fp8 = 2  # bf16 occupies 2 fp8 columns
    n_B_tiles = dims.B // _pmax
    aff_col_bf16 = kernel_cfg.affinities_col_offset // _bf16_as_fp8

    # bf16 view of the fp8 concat: [_pmax, n_B_tiles, concat_free_size // 2]. Indexing through the
    # nl.NkiTensor makes the dynamic expert select operate in bf16 element units
    # and avoids the fp8-vs-bf16 scalar_offset unit

    concat_bf16 = block_hidden_concat.view(nl.bfloat16)

    # Dynamic select requires a uint32 index (TensorCopyDynamicSrc verifier). block_expert is int32;
    # experts are 0..E-1 (always positive) so the bit pattern is identical -- reinterpret in place.
    block_expert_u32 = block_expert.view(nl.uint32)

    expert_affinity = []
    for n in range(n_B_tiles):
        affinity_f32 = _sbm_alloc(
            sbm, (_pmax, 1), dtype=nl.float32, name=f"{name_prefix}_affinity_t{n}", align=SBUF_QUADRANT_SIZE
        )
        # [_pmax, n_B_tiles, free_bf16] -> pick B-tile n -> [_pmax, free_bf16]
        #   -> slice affinity cols [aff_col_bf16, +E] -> [_pmax, E]
        #   -> dynamic-select this block's expert column -> [_pmax]
        #   -> expand to [_pmax, 1] for the fp32 affinity contract.
        aff_tv = (
            concat_bf16.slice(1, n, n + 1)
            .squeeze_dim(1)
            .slice(1, aff_col_bf16, aff_col_bf16 + dims.E)
            .select(dim=1, index=block_expert_u32)
            .expand_dim(1)
        )
        # tensor_copy bf16 -> fp32 (compute consumes fp32 affinity).
        nisa.tensor_copy(dst=affinity_f32, src=aff_tv)
        expert_affinity.append(affinity_f32)
    return expert_affinity


def _gather_gup_in_quant_recip(inps, block_expert, dims, sbm, name_prefix=""):
    """STATIC_MX: pull 1/in_scale[block_expert] from slot 0 of gup_scale_lut_sb."""
    # scalar_offset requires uint32; reinterpret block_expert (int32 [1,1]) in place (same-size bitcast).
    block_expert_u32 = block_expert.view(nl.uint32)
    in_quant_recip = _sbm_alloc(
        sbm, (_pmax, 1), dtype=nl.float32, name=f"{name_prefix}in_quant_recip", align=SBUF_QUADRANT_SIZE
    )
    nisa.tensor_copy(
        dst=in_quant_recip,
        src=inps.gup_scale_lut_sb.ap(
            pattern=[[dims.E * 3, _pmax], [3, 1], [1, 1]],
            offset=0,
            scalar_offset=block_expert_u32,
            indirect_dim=1,
        ),
    )
    return in_quant_recip


def _compute_weight_skip_mask(sbm, all_experts, all_experts_for_weights, n_blocks_this_shard, E):
    """Compare consecutive block experts and set weight expert to E (OOB/skip) where same."""
    sbm.open_scope(name="weight_skip_mask")
    is_same = _sbm_alloc(sbm, (1, n_blocks_this_shard - 1), dtype=nl.uint8, name="is_same", align=SBUF_QUADRANT_SIZE)
    nisa.tensor_tensor(
        data1=all_experts[0:1, 1:n_blocks_this_shard],
        data2=all_experts[0:1, 0 : n_blocks_this_shard - 1],
        op=nl.equal,
        dst=is_same,
    )
    on_false = _sbm_alloc(sbm, (1, n_blocks_this_shard - 1), dtype=nl.int32, name="on_false", align=SBUF_QUADRANT_SIZE)
    nisa.memset(dst=on_false, value=E)
    nisa.tensor_copy_predicated(
        dst=all_experts_for_weights[0:1, 1:n_blocks_this_shard],
        src=on_false,
        predicate=is_same,
    )
    sbm.close_scope()


def _prefetch_gup_tile0(
    inps,
    expert,
    prj_cfg,
    buffers,
    dims,
    skip_dma,
    sbm,
    name_prefix="pf",
    scale_expert=None,
    use_packed_scales=False,
    skip_scales=False,
    full_I_load=False,
):
    """Prefetch gate/up weights and scales into persistent buffers.

    scale_expert: optional OOB-aware expert used only for the scale expert-index vector
        (enables scale-skip across same-expert blocks). Defaults to `expert`.
    use_packed_scales: caller-supplied flag selecting packed vs. standard HBM scale layout.
    skip_scales: STATIC_MX path skips scale DMAs entirely — inps.gup_scales_sb holds dummy 127.
    full_I_load: when True, prefetch the FULL I dim of gup weights (used by the
        full-resident gup scheme). When False, prefetch one I-tile worth.
    """
    scale_shape = inps.gate_up_proj_scale.shape if not skip_scales else None
    # Packed scales address via scalar_offset=block_expert, so the per-expert index vector is never consumed.
    if not skip_scales and not use_packed_scales:
        token_indices = _generate_expert_index_vector(
            expert_index=scale_expert if scale_expert != None else expert,
            dst_idx_vector=inps.p_gup_idx_vector,
            scale_factor=scale_shape[1],
            n_quadrants_needed=prj_cfg.H0 // SBUF_QUADRANT_SIZE,
            n_remaining_partition=0,
            name_prefix=f"{name_prefix}_gup_eiv",
            sbm=sbm,
            dst_int32=inps.p_gup_idx_vector_int32,
        )
    # Weight extent: full I for the full-resident scheme; one I-tile otherwise.
    load_I = dims.I if full_I_load else min(_I_TILE_SZ, dims.I)
    gup_weight_view = (
        inps.gate_up_proj_weight.select(dim=0, index=expert)
        .slice(dim=2, start=0, end=prj_cfg.n_H512_tile_sharded)
        .slice(dim=3, start=0, end=load_I)
    )
    nisa.dma_copy(
        dst=buffers.gup_tile_buf_a[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, :load_I],
        src=gup_weight_view,
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
    )
    if skip_scales:
        # STATIC_MX: scale operand is the constant all-127 dummy, nothing to load per block.
        return
    if use_packed_scales:
        # Packed prefetch: mirror production's gate/up split for parity.
        # When skip_weight: one combined DMA (single OOB-skip evaluation).
        # Otherwise: two DMAs (gate + up) for better DMA scheduling.
        n_packed_gup = scale_shape[2]
        _scale_expert = scale_expert if scale_expert != None else expert
        if skip_dma.skip_weight:
            nisa.dma_copy(
                dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, :2, : prj_cfg.I],
                src=inps.gate_up_proj_scale.ap(
                    pattern=[
                        [n_packed_gup * 2 * prj_cfg.I, _pmax],
                        [2 * prj_cfg.I, n_packed_gup],
                        [prj_cfg.I, 2],
                        [1, prj_cfg.I],
                    ],
                    offset=0,
                    scalar_offset=_scale_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip,
                dge_mode=dge_mode.hwdge,
            )
        else:
            # Gate scales
            nisa.dma_copy(
                dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, 0:1, : prj_cfg.I],
                src=inps.gate_up_proj_scale.ap(
                    pattern=[
                        [n_packed_gup * 2 * prj_cfg.I, _pmax],
                        [2 * prj_cfg.I, n_packed_gup],
                        [1, prj_cfg.I],
                    ],
                    offset=0,
                    scalar_offset=_scale_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )
            # Up scales
            nisa.dma_copy(
                dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, 1:2, : prj_cfg.I],
                src=inps.gate_up_proj_scale.ap(
                    pattern=[
                        [n_packed_gup * 2 * prj_cfg.I, _pmax],
                        [2 * prj_cfg.I, n_packed_gup],
                        [1, prj_cfg.I],
                    ],
                    offset=prj_cfg.I,
                    scalar_offset=_scale_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )
        return
    # Full scales — one trigger for weight_skipping (simpler OOB-skip), two triggers otherwise (better DMA scheduling)
    gup_scale_view = inps.gate_up_proj_scale.reshape(
        (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
    )
    full_n_H512 = scale_shape[3]
    stride_dim0 = 2 * full_n_H512 * prj_cfg.I
    if skip_dma.skip_weight:
        # Single trigger: load gate+up scales together (matches load_gup_weights_scales_mx pattern)
        nisa.dma_copy(
            src=gup_scale_view.ap(
                pattern=[
                    [stride_dim0, _pmax],
                    [full_n_H512 * prj_cfg.I, 2],
                    [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                    [1, prj_cfg.I],
                ],
                offset=0,
                vector_offset=token_indices.ap([[1, _pmax], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            dst=inps.gup_scales_sb[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
            oob_mode=oob_mode.skip,
        )
    else:
        # Two triggers: split gate and up for better DMA scheduling
        # Gate scales
        nisa.dma_copy(
            src=gup_scale_view.ap(
                pattern=[
                    [stride_dim0, _pmax],
                    [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                    [1, prj_cfg.I],
                ],
                offset=0,
                vector_offset=token_indices.ap([[1, _pmax], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            dst=inps.gup_scales_sb[:_pmax, 0:1, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
            oob_mode=oob_mode.skip,
        )
        # Up scales
        nisa.dma_copy(
            src=gup_scale_view.ap(
                pattern=[
                    [stride_dim0, _pmax],
                    [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                    [1, prj_cfg.I],
                ],
                offset=full_n_H512 * prj_cfg.I,
                vector_offset=token_indices.ap([[1, _pmax], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            dst=inps.gup_scales_sb[:_pmax, 1:2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
            oob_mode=oob_mode.skip,
        )


def check_kernel_compatibility(dims: BWMMMXDimensionSizes, configs: BWMMMXConfigs):
    """
    Validate kernel configuration and dimension compatibility.

    Performs comprehensive validation of kernel parameters to ensure they meet
    hardware constraints and implementation requirements before execution.

    Args:
        dims (BWMMMXDimensionSizes): Dimension configuration containing B, H, I, N,
            num_shards, and cond_vec_len.
        configs (BWMMMXConfigs): Kernel configuration containing is_tensor_update_accumulating
            and use_dynamic_while flags.

    Returns:
        None: Raises assertion errors if validation fails.

    Notes:
        - Block size (B) must be multiple of 128 for efficient tiling
        - Hidden dimension (H) must be in range [512, 8192] and divisible by PSUM_SIZE (512)
        - Intermediate dimension (I) must be divisible by 16 for quantization alignment
        - Currently only supports 2-shard execution
        - Dynamic loop requires condition vector of length N+2
        - Only supports topK > 1 (tensor update accumulating mode)

    Pseudocode:
        assert B % 128 == 0
        assert 512 <= H <= 8192
        assert H % PSUM_SIZE == 0
        assert I % 16 == 0
        assert is_tensor_update_accumulating == True
        assert num_shards == 2
        if use_dynamic_while:
            assert cond_vec_len == N + 2
    """
    kernel_assert(dims.B % _pmax == 0, f"Blocksize must be a multiple of 128")
    kernel_assert(512 <= dims.H <= 8192, f"Hidden dims must be between 512 and 8192, found {dims.H}")
    kernel_assert(dims.H % PSUM_SIZE == 0, f"Hidden dim size must be multiples of {PSUM_SIZE}, found {dims.H} ")

    kernel_assert(dims.I % 16 == 0, f"down_proj_weight I must be divisible by 16, found {dims.I} . Please pad it")
    kernel_assert(configs.is_tensor_update_accumulating, "Only support topK > 1 at the moment.")

    kernel_assert(dims.num_shards == 2, f"The kernel only support sharding on exactly 2 cores, got {dims.num_shards}")

    if configs.use_dynamic_while:
        kernel_assert(
            dims.cond_vec_len == dims.N + 2,
            f"condition vector must have exactly N+2 elements, got {dims.cond_vec_len} != N + 2 ({dims.N} + 2)",
        )


def load_gup_weights_scales_mx(
    inps: InputTensors,
    block_expert: nl.NkiTensor,
    dims: BWMMMXDimensionSizes,
    prj_cfg: ProjConfig,
    skip_dma: SkipMode,
    sbm=None,
    dst_weight=None,
    dst_bias=None,
    name_prefix="gup",
    use_packed_scales: bool = False,
    skip_scales: bool = False,  # STATIC_MX: skip per-block scale DMA (inps.gup_scales_sb holds dummy 127)
):
    """
    Load gate and up projection weights, scales, and biases for current expert.

    Loads MXFP4/MXFP8 quantized weights, uint8 scales, and biases for both gate and up
    projections from HBM to SBUF for the expert assigned to the current block.

    Args:
        inps (InputTensors): Input tensors containing gate_up_proj_weight of shape
            [E, 128, 2, n_H512_tile, I], gate_up_proj_scale, gate_and_up_proj_bias,
            and buffers for scales and index vectors.
        block_expert (nl.NkiTensor): Expert index for current block, shape [1, 1].
        dims (BWMMMXDimensionSizes): Dimension configuration with I, H.
        prj_cfg (ProjConfig): Projection configuration with n_H512_tile_sharded, I.
        skip_dma (SkipMode): DMA skip configuration for weight loading.

    Returns:
        tuple: (gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb)
            - gup_weights_qtz_sb (nl.NkiTensor): Quantized weights [128, 2, n_H512_tile_sharded, I]
            - gup_scales_sb (nl.NkiTensor): Dequantization scales [128, 2, n_H512_tile_sharded, I]
            - gup_bias_sb (nl.NkiTensor): Bias values [128, 2, n_total_I512_tile, 128]

    Notes:
        - Uses indirect DGE with block_expert for expert selection
        - Generates index vectors on-the-fly for scale loading
        - Pads bias to 512 when I < 512 for alignment
        - Scales are loaded with zero-padding for out-of-bounds partitions
        - Gate and up projections share weight buffer (dimension 1 has size 2)

    Pseudocode:
        gup_weights_qtz_sb = allocate [128, 2, n_H512_tile_sharded, I] in SBUF
        dma_copy gate_up_proj_weight[block_expert, :, :, :, :] to gup_weights_qtz_sb

        gup_scale_view = reshape gate_up_proj_scale to [E*16, 2, n_H512_tile, I]
        token_indices_on_p = generate_expert_index_vector(block_expert)
        dma_copy gup_scale_view[token_indices_on_p, :, :, :] to gup_scales_sb

        gup_bias_sb = allocate [128, 2, n_total_I512_tile, 128] in SBUF
        if I < 512:
            memset gup_bias_sb to 0
            dma_copy gate_and_up_proj_bias[block_expert, :I//4, :, :, :] to gup_bias_sb[:I//4, :, :, :]
        else:
            dma_copy gate_and_up_proj_bias[block_expert, :, :, :, :] to gup_bias_sb

        return gup_weights_qtz_sb, gup_scales_sb, gup_bias_sb
    """
    if dst_weight != None:
        gup_weights_qtz_sb = dst_weight
    else:
        gup_weights_qtz_sb = _sbm_alloc(
            sbm,
            (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
            dtype=inps.gate_up_proj_weight.dtype,
            name="gup_weights_qtz_sb",
            align=SBUF_QUADRANT_SIZE,
        )
    """
    Load gate/up weight for current expert.

    gate_up_proj_weight shape: (E, 128, 2, n_H512_tile, I)
    select expert -> (128, 2, n_H512_tile, I)
    slice H512 tiles -> (128, 2, n_H512_tile_sharded, I)
    """
    gup_weight_view = inps.gate_up_proj_weight.select(dim=0, index=block_expert).slice(
        dim=2, start=0, end=prj_cfg.n_H512_tile_sharded
    )
    nisa.dma_copy(
        dst=gup_weights_qtz_sb,
        src=gup_weight_view,
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
    )

    """
    GATE UP SCALES
    """
    # gup_n_quadrants_needed is returned to callers; depends only on H0 / quadrant
    # size, so it's the same regardless of scale layout.
    gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
    token_indices_on_p = None
    scale_shape = None

    # STATIC_MX skips entirely: inps.gup_scales_sb holds persistent dummy 127 from top-level memset.
    if not skip_scales:
        scale_shape = inps.gate_up_proj_scale.shape

        if use_packed_scales:
            # Packed scale path: HBM is [E, _pmax, n_packed_gup, 2, I]; SBUF is the
            # same shape. Single dma_copy with scalar_offset=block_expert.
            n_packed_gup = scale_shape[2]
            kernel_assert(
                scale_shape == (dims.E, _pmax, n_packed_gup, 2, prj_cfg.I),
                f"Packed gate_up_proj_scale shape mismatch: got {scale_shape}, "
                f"expected ({dims.E}, {_pmax}, n_packed_gup, 2, {prj_cfg.I})",
            )
            nisa.dma_copy(
                dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, :2, : prj_cfg.I],
                src=inps.gate_up_proj_scale.ap(
                    pattern=[
                        [n_packed_gup * 2 * prj_cfg.I, _pmax],
                        [2 * prj_cfg.I, n_packed_gup],
                        [prj_cfg.I, 2],
                        [1, prj_cfg.I],
                    ],
                    offset=0,
                    scalar_offset=block_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )
        else:
            # fold E * 16 together
            gup_scale_view = inps.gate_up_proj_scale.reshape(
                (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
            )

            """
            Construct a vector DGE index to index into E*16
                if block_expert == 0, we want something like this (tranposed to the P dimension)
                [0 1 2 3 -1 -1 -1 ..... 4 5 6 7 -1 -1 -1 .... 8 9 10 11 -1 -1 -1 .... 12 13 14 15 -1 -1 -1... -1]

                if block_expert == 3, we want something like this
                [48 49 50 51 -1 -1 -1 ..... 52 53 54 55 -1 -1 -1 .... 56 57 58 59 -1 -1 -1 .... 60 61 62 63 -1 -1 -1... -1]
                i.e, basically the same as above, with offset 16*3 = 48
            """
            token_indices_on_p = _generate_expert_index_vector(
                expert_index=block_expert,
                dst_idx_vector=inps.p_gup_idx_vector,
                scale_factor=scale_shape[1],
                n_quadrants_needed=gup_n_quadrants_needed,
                n_remaining_partition=0,
                name_prefix=f"{name_prefix}_expert_index_vector",
                sbm=sbm,
                dst_int32=inps.p_gup_idx_vector_int32,
            )
            # gup_scale_view shape: (E*16, 2, n_H512_tile, I) - use FULL source tensor dimensions for strides.
            # The source tensor has full n_H512_tile, we only load n_H512_tile_sharded elements.
            full_n_H512_tile_scale = scale_shape[3]
            stride_dim0 = 2 * full_n_H512_tile_scale * prj_cfg.I
            nisa.dma_copy(
                src=gup_scale_view.ap(
                    pattern=[
                        [stride_dim0, _pmax],
                        [full_n_H512_tile_scale * prj_cfg.I, 2],
                        [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                        [1, prj_cfg.I],
                    ],
                    offset=0,
                    vector_offset=token_indices_on_p.ap(
                        [[1, _pmax], [1, 1]],
                        offset=0,
                    ),
                    indirect_dim=0,
                ),
                dst=inps.gup_scales_sb[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
                oob_mode=oob_mode.skip,
            )

    """
    GATE UP BIAS
    """
    gup_bias_sb = None
    if inps.gate_and_up_proj_bias:
        if dst_bias != None:
            gup_bias_sb = dst_bias
        else:
            gup_bias_sb = _sbm_alloc(
                sbm,
                (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                dtype=inps.gate_and_up_proj_bias.dtype,
                name="gup_bias_sb",
                align=SBUF_QUADRANT_SIZE,
            )

        if dims.I < _pmax * _q_width:  # when I<512, gate/up bias HBM is not padded so pad it here
            if not (skip_dma.skip_weight and dst_bias != None):
                nisa.memset(dst=gup_bias_sb[:, :, 0, :], value=0.0)
            # gate_and_up_proj_bias shape: (E, I_par_dim, 2, n_total_I512_tile, _q_width) where I_par_dim = I//4
            I_par_dim = dims.I // 4
            bias_stride_dim0 = 2 * prj_cfg.n_total_I512_tile * _q_width  # stride for I_par_dim
            bias_stride_dim1 = prj_cfg.n_total_I512_tile * _q_width  # stride for gate/up (2)
            bias_stride_dim2 = _q_width  # stride for n_total_I512_tile
            nisa.dma_copy(
                dst=gup_bias_sb[:I_par_dim, :, :, :],
                src=inps.gate_and_up_proj_bias.ap(
                    pattern=[
                        [bias_stride_dim0, I_par_dim],
                        [bias_stride_dim1, 2],
                        [bias_stride_dim2, prj_cfg.n_total_I512_tile],
                        [1, _q_width],
                    ],
                    offset=0,
                    scalar_offset=block_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )
        else:
            # gate_and_up_proj_bias shape: (E, _pmax, 2, n_total_I512_tile, _q_width)
            # Strides: dim1=2*n_total_I512_tile*_q_width, dim2=n_total_I512_tile*_q_width, dim3=_q_width, dim4=1
            bias_stride_dim1 = 2 * prj_cfg.n_total_I512_tile * _q_width
            bias_stride_dim2 = prj_cfg.n_total_I512_tile * _q_width
            nisa.dma_copy(
                dst=gup_bias_sb,
                src=inps.gate_and_up_proj_bias.ap(
                    pattern=[
                        [bias_stride_dim1, _pmax],
                        [bias_stride_dim2, 2],
                        [_q_width, prj_cfg.n_total_I512_tile],
                        [1, _q_width],
                    ],
                    offset=0,
                    scalar_offset=block_expert,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
                dge_mode=dge_mode.hwdge,
            )

    return gup_weights_qtz_sb, inps.gup_scales_sb, gup_bias_sb, token_indices_on_p, gup_n_quadrants_needed


def _load_gup_weight_tile(
    inps,
    block_expert,
    prj_cfg,
    skip_dma,
    dst_weight,
    dst_scale,
    dst_bias,
    token_indices_on_p,
    I_offset,
    dims,
    sbm=None,
):
    """Load a single I-tile of gate/up weights, scales, and bias from HBM to SBUF.

    Args:
        dst_weight: SBUF buffer (_pmax, 2, n_H512_tile_sharded, _I_TILE_SZ).
        dst_scale: SBUF buffer (_pmax, 2, n_H512_tile_sharded, _I_TILE_SZ).
        dst_bias: SBUF buffer (_pmax, 2, 1, _q_width).
        token_indices_on_p: Pre-computed expert index vector for scale DGE.
        I_offset: Starting offset in I dimension.
    """
    cur_I_load_sz = min(_I_TILE_SZ, dims.I - I_offset)

    # --- WEIGHTS ---
    gup_weight_view = (
        inps.gate_up_proj_weight.select(dim=0, index=block_expert)
        .slice(dim=2, start=0, end=prj_cfg.n_H512_tile_sharded)
        .slice(dim=3, start=I_offset, end=I_offset + cur_I_load_sz)
    )
    nisa.dma_copy(
        dst=dst_weight[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, :cur_I_load_sz],
        src=gup_weight_view,
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
    )

    # --- SCALES ---
    if dst_scale != None:
        scale_shape = inps.gate_up_proj_scale.shape
        gup_scale_view = inps.gate_up_proj_scale.reshape(
            (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
        )
        full_n_H512_tile_scale = scale_shape[3]
        stride_dim0 = 2 * full_n_H512_tile_scale * dims.I
        nisa.dma_copy(
            src=gup_scale_view.ap(
                pattern=[
                    [stride_dim0, _pmax],
                    [full_n_H512_tile_scale * dims.I, 2],
                    [dims.I, prj_cfg.n_H512_tile_sharded],
                    [1, cur_I_load_sz],
                ],
                offset=I_offset,
                vector_offset=token_indices_on_p.ap(
                    [[1, _pmax], [1, 1]],
                    offset=0,
                ),
                indirect_dim=0,
            ),
            dst=dst_scale[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, :cur_I_load_sz],
            oob_mode=oob_mode.skip,
        )

    # --- BIAS ---
    if dst_bias == None:
        return

    i_I512_tile = I_offset // _I_TILE_SZ
    full_n_total_I512_tile = prj_cfg.n_total_I512_tile

    if dims.I < _I_TILE_SZ:
        # I < 512: bias HBM has I_par_dim = I//4 on p-dim due to QMX
        I_par_dim = dims.I // 4
        bias_stride_dim0 = 2 * full_n_total_I512_tile * _q_width
        bias_stride_dim1 = full_n_total_I512_tile * _q_width
        nisa.dma_copy(
            dst=dst_bias[:I_par_dim, :, :, :],
            src=inps.gate_and_up_proj_bias.ap(
                pattern=[
                    [bias_stride_dim0, I_par_dim],
                    [bias_stride_dim1, 2],
                    [_q_width, 1],
                    [1, _q_width],
                ],
                offset=i_I512_tile * _q_width,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )
    else:
        # I >= 512: bias HBM has _pmax on p-dim
        bias_stride_dim1 = 2 * full_n_total_I512_tile * _q_width
        bias_stride_dim2 = full_n_total_I512_tile * _q_width
        nisa.dma_copy(
            dst=dst_bias,
            src=inps.gate_and_up_proj_bias.ap(
                pattern=[
                    [bias_stride_dim1, _pmax],
                    [bias_stride_dim2, 2],
                    [_q_width, 1],
                    [1, _q_width],
                ],
                offset=i_I512_tile * _q_width,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )


def load_down_proj_weights_mx(
    inps: InputTensors,
    block_expert: nl.NkiTensor,
    dst_weight: nl.NkiTensor,
    dims: BWMMMXDimensionSizes,
    prj_cfg: ProjConfig,
    skip_dma: SkipMode,
    gup_token_indices_on_p: nl.NkiTensor = None,
    gup_n_quadrants_needed: int = None,
    dst_scale: nl.NkiTensor = None,
    sbm=None,
    dst_bias=None,
    use_packed_scales: bool = False,
    skip_scales: bool = False,  # STATIC_MX: skip per-block scale DMA (caller pre-memsets dummy 127)
):
    """
    Load down projection weights, scales, and biases for current expert.

    Loads MXFP4/MXFP8 quantized weights and uint8 scales for down projection from HBM
    to SBUF, constructing partition index vectors for proper expert selection.

    Args:
        inps (InputTensors): Input tensors containing down_proj_weight [E, p_I, n_total_I512_tile, H],
            down_proj_scale, down_proj_bias, and index vector buffer.
        block_expert (nl.NkiTensor): Expert index for current block, shape [1, 1].
        dst_weight (nl.NkiTensor): Destination buffer for weights in SBUF.
        dims (BWMMMXDimensionSizes): Dimension configuration with I, H, p_I.
        prj_cfg (ProjConfig): Projection configuration with sharding info.
        skip_dma (SkipMode): DMA skip configuration.

    Returns:
        tuple: (down_weight_hbm, down_scale_sb, down_bias_sb)
            - down_weight_hbm: Reference to weight tensor in HBM
            - down_scale_sb: Scales in SBUF [128, n_total_I512_tile, H_sharded]
            - down_bias_sb: Bias in SBUF [1, H]

    Notes:
        - Loads only sharded portion of H dimension per program
        - Constructs partition index with quadrant-based addressing
        - Handles remainder partitions when p_I not divisible by 32
        - Zero-pads scales when p_I < 128

    Pseudocode:
        dma_copy down_proj_weight[block_expert, :, :, :] to dst_weight

        down_scale_sb = allocate [128, n_total_I512_tile, H_sharded] in SBUF
        if p_I != 128:
            memset down_scale_sb[:, -1, :] to 0

        down_scale_view = reshape down_proj_scale to [E*16, n_total_I512_tile, H]
        construct p_down_idx_vector: [block_expert*16+0, ..., block_expert*16+15, -1, ...]
        dma_copy down_scale_view[p_down_idx_vector, :, :] to down_scale_sb

        down_bias_sb = allocate [1, H] in SBUF
        dma_copy down_proj_bias[block_expert, :] to down_bias_sb

        return down_scale_sb, down_bias_sb
    """
    """
    Load down projection weights from HBM to SBUF.
    
    down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
    Load directly into dst_weight with scalar AP.
    scalar_offset=block_expert with indirect_dim=0 means access starts at 
    block_expert * (p_I * n_total_I512_tile * H)
    """

    """
    Load down projection weights from HBM to SBUF.

    down_proj_weight shape: (E, p_I, n_total_I512_tile, H)
    select expert -> (p_I, n_total_I512_tile, H)
    slice H for sharding -> (p_I, n_total_I512_tile, H_sharded)
    """
    down_weight_view = inps.down_proj_weight.select(dim=0, index=block_expert).slice(
        dim=2, start=prj_cfg.prg_id * prj_cfg.H_sharded, end=(prj_cfg.prg_id + 1) * prj_cfg.H_sharded
    )
    nisa.dma_copy(
        src=down_weight_view,
        dst=dst_weight[: dims.p_I, :, :],
        oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
        dge_mode=dge_mode.hwdge,
    )

    """
    DOWN SCALES
    """
    # STATIC_MX skips entirely: caller pre-fills dst_scale (the shared [_pmax, 1] all-127 buffer
    # under STATIC_MX, with [p_I:_pmax] zeroed when p_I < _pmax).
    if skip_scales:
        kernel_assert(dst_scale is not None, "skip_scales=True requires caller-provided dst_scale (dummy 127 buffer)")
        down_scale_sb = dst_scale
        # Skip directly to bias load.
        scale_shape = None
        n_packed_down = None
    else:
        scale_shape = inps.down_proj_scale.shape

        # n_packed_down is needed by the packed DMA below regardless of whether the
        # buffer was allocated locally or passed in via dst_scale, so derive it once up front.
        n_packed_down = scale_shape[2] if use_packed_scales else None

        # Alloc and load weight scale, which needs zero padding in sbuf
        if dst_scale != None:
            down_scale_sb = dst_scale
        elif use_packed_scales:
            down_scale_sb = _sbm_alloc(
                sbm,
                (_pmax, n_packed_down, prj_cfg.H_sharded),
                dtype=nl.uint8,
                name="down_scale_packed_sb_local",
                align=SBUF_QUADRANT_SIZE,
            )
            if dims.p_I != _pmax:
                nisa.memset(down_scale_sb[:, n_packed_down - 1, :], value=0)
        else:
            down_scale_sb = _sbm_alloc(
                sbm,
                (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded),
                dtype=nl.uint8,
                name="down_scale_sb_local",
                align=SBUF_QUADRANT_SIZE,
            )
            # Memset weight scale if input weight scale HBM does not pad on par dim
            if dims.p_I != _pmax:
                nisa.memset(down_scale_sb[:, prj_cfg.n_total_I512_tile - 1, :], value=0)

        if not use_packed_scales:
            kernel_assert(
                down_scale_sb.shape == (_pmax, prj_cfg.n_total_I512_tile, prj_cfg.H_sharded),
                f"Got {down_scale_sb.shape}",
            )

    if skip_scales:
        pass  # No scale DMA under STATIC_MX
    elif use_packed_scales:
        # Packed scale path: HBM is [E, _pmax, n_packed_down, H]; SBUF strips the
        # leading E dim (one expert at a time, via scalar_offset=block_expert).
        kernel_assert(
            scale_shape == (dims.E, _pmax, n_packed_down, dims.H),
            f"Packed down_proj_scale shape mismatch: got {scale_shape}, "
            f"expected ({dims.E}, {_pmax}, n_packed_down, {dims.H})",
        )
        nisa.dma_copy(
            dst=down_scale_sb[:_pmax, :n_packed_down, : prj_cfg.H_sharded],
            src=inps.down_proj_scale.ap(
                pattern=[
                    [n_packed_down * dims.H, _pmax],
                    [dims.H, n_packed_down],
                    [1, prj_cfg.H_sharded],
                ],
                offset=prj_cfg.prg_id * prj_cfg.H_sharded,
                scalar_offset=block_expert,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )
    else:
        """
        Construct a vector DGE index to index into E*16
            if block_expert == 0, we want something like this (tranposed to the P dimension)
            [0 1 2 3 -1 -1 -1 ..... 4 5 6 7 -1 -1 -1 .... 8 9 10 11 -1 -1 -1 .... 12 13 14 15 -1 -1 -1... -1]

            if block_expert == 3, we want something like this
            [48 49 50 51 -1 -1 -1 ..... 52 53 54 55 -1 -1 -1 .... 56 57 58 59 -1 -1 -1 .... 60 61 62 63 -1 -1 -1... -1]
            i.e, basically the same as above, with offset 16*3 = 48
        """
        down_scale_view = inps.down_proj_scale.reshape(
            (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3])
        )

        down_n_quadrants_needed, n_remaining_partition = divmod(dims.p_I, SBUF_QUADRANT_SIZE)
        n_remaining_partition = n_remaining_partition // _q_height

        if gup_n_quadrants_needed != None and gup_n_quadrants_needed == down_n_quadrants_needed:
            token_indices_on_p = gup_token_indices_on_p
        else:
            token_indices_on_p = _generate_expert_index_vector(
                expert_index=block_expert,
                dst_idx_vector=inps.p_down_idx_vector,
                scale_factor=scale_shape[1],
                n_quadrants_needed=down_n_quadrants_needed,
                n_remaining_partition=n_remaining_partition,
                name_prefix="down_expert_index_vector",
                sbm=sbm,
            )

        # down_scale_view shape: (E*16, n_total_I512_tile, H)
        # accumulated shape to right of dim 0: n_total_I512_tile * H
        down_scale_stride_dim0 = prj_cfg.n_total_I512_tile * dims.H

        # Copy only p_I valid partitions (padding partitions already zeroed by memset outside loop)
        nisa.dma_copy(
            src=down_scale_view.ap(
                pattern=[
                    [down_scale_stride_dim0, dims.p_I],
                    [dims.H, prj_cfg.n_total_I512_tile],
                    [1, prj_cfg.H_sharded],
                ],
                offset=prj_cfg.prg_id * prj_cfg.H_sharded,
                vector_offset=token_indices_on_p.ap(
                    [[1, dims.p_I], [1, 1]],
                    offset=0,
                ),
                indirect_dim=0,
            ),
            dst=down_scale_sb[: dims.p_I, : prj_cfg.n_total_I512_tile, : prj_cfg.H_sharded],
            oob_mode=oob_mode.skip,
        )

    # load bias
    # down_proj_bias shape: (E, H)
    down_bias_sb = None
    if inps.down_proj_bias:
        if dst_bias != None:
            down_bias_sb = dst_bias
        else:
            down_bias_sb = _sbm_alloc(
                sbm,
                (1, dims.H),
                dtype=inps.down_proj_bias.dtype,
                name="down_bias_sb",
                align=SBUF_QUADRANT_SIZE,
            )
        nisa.dma_copy(
            src=inps.down_proj_bias.ap(
                pattern=[[dims.H, 1], [1, dims.H]], offset=0, scalar_offset=block_expert, indirect_dim=0
            ),
            dst=down_bias_sb,
            oob_mode=oob_mode.skip if skip_dma.skip_weight else oob_mode.error,
            dge_mode=dge_mode.hwdge,
        )

    return down_scale_sb, down_bias_sb


def _write_output_scatter(src_data, token_indices_2D, outs, dims, shard_id):
    """Write block results to output tensor via indirect scatter DMA.

    Args:
        src_data: SBUF tensor (_pmax, n_B128_tiles, H) with final accumulated results.
        token_indices_2D: Token index mapping (_pmax, n_B128_tiles).
        outs: OutputTensors with output HBM tensor.
        dims: Dimension sizes.
        shard_id: Current shard identifier.
    """
    T = outs.output.shape[-2]
    shard_offset = shard_id * T * dims.H
    num_shards = outs.output.shape[0]

    for n in range(dims.B // _pmax):
        block_token_mapping = token_indices_2D.ap(
            [[dims.n_B128_tiles, _pmax], [1, 1]],
            offset=n,
        )
        output_ap = outs.output.reshape((num_shards * T, 1, dims.H)).ap(
            pattern=[[dims.H, _pmax], [1, 1], [1, dims.H]],
            offset=shard_offset,
            vector_offset=block_token_mapping,
            indirect_dim=0,
        )
        nisa.dma_copy(
            dst=output_ap,
            src=src_data[0:_pmax, n, 0 : dims.H],
            oob_mode=oob_mode.skip,
        )


def compute_one_block(
    block_idx: int,
    next_block_idx: int,
    buffers: SharedBuffers,
    dims: BWMMMXDimensionSizes,
    inps: InputTensors,
    outs: OutputTensors,
    kernel_cfg: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    shard_id: Any,
    is_dummy: bool = False,
    is_dynamic: bool = False,
    is_first_block: bool = False,
    sbm=None,
    block_expert_for_weights=None,
    next_block_expert_for_weights=None,
    hoisted_gup_weights=None,
    hoisted_gup_bias=None,
    hoisted_down_bias=None,
    delay_output_write: bool = False,
    pending_token_indices=None,
    name_tag=None,
):
    """
    Process one block through complete MoE MLP pipeline.

    Executes gate projection, up projection, activation, and down projection
    for a single block with MXFP4 quantization and expert routing.

    Args:
        block_idx (int): Current block index.
        next_block_idx (int): Next block index for prefetching (None if last).
        buffers (SharedBuffers): Shared computation buffers.
        dims (BWMMMXDimensionSizes): Dimension configuration.
        inps (InputTensors): Input tensors.
        outs (OutputTensors): Output tensors.
        kernel_cfg (BWMMMXConfigs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        shard_id (Any): Current shard identifier.
        is_dummy (bool): Whether this is a dummy block (for load balancing).
        is_dynamic (bool): Whether from dynamic loop.

    Returns:
        None: Writes results to outs.output.

    Notes:
        - Loads expert weights and scales
        - Prefetches next block hidden states if next_block_idx provided
        - Applies gate/up projections with optional clamping
        - Applies activation function (SiLU or Swish)
        - Computes down projection
        - Scales by expert affinity and accumulates
        - Dummy blocks have zero affinity for load balancing

    Pseudocode:
        block_expert = load_block_expert(block_to_expert, block_idx)

        if next_block_idx != None:
            compute_hidden_index_vector(inps, buffers, next_block_idx, dims, skip_dma, is_dynamic)

        if not is_dynamic:
            quantize_block_hidden_state_T(buffers, prj_cfg, dims)

        reshape buffers.hidden_qtz_sb and hidden_scale_sb

        gate_and_up_weights, gate_and_up_scales, gup_bias = load_gup_weights_scales_mx4(inps, block_expert, dims, prj_cfg, skip_dma)
        down_scale_sb, down_bias_sb = load_down_proj_weights_mx4(inps, block_expert, buffers.down_weight_qtz, dims, prj_cfg, skip_dma)

        token_indices_2D = load_token_indices(token_position_to_id, block_idx, B, n_B128_tiles)
        expert_affinity = calculate_expert_affinities(expert_affinities_masked, token_indices_2D, block_expert, E, B//128, compute_dtype, skip_dma)
        block_old = load_prev_block(output, token_indices_2D, block_old, B//128, compute_dtype, shard_id, skip_dma)

        gate_proj_out = gate_up_proj_mxfp4_tp(hidden_qtz_sb, hidden_scale_sb, gate_weights, gate_scales, gate_bias, cfg)
        gate_proj_out = clamp(gate_proj_out, gate_clamp_lower_limit, gate_clamp_upper_limit)

        up_proj_out = gate_up_proj_mxfp4_tp(hidden_qtz_sb, hidden_scale_sb, up_weights, up_scales, up_bias, cfg)
        up_proj_out = clamp(up_proj_out, up_clamp_lower_limit, up_clamp_upper_limit)

        if next_block_idx != None:
            load_and_quantize_hidden_states(
                inps, next_block_idx, buffers, dims, kernel_cfg, prj_cfg, is_dynamic, USE_DMA_TRANSPOSE
            )

        if activation_function == SiLU:
            gate_proj_out = silu(gate_proj_out)
        elif activation_function == Swish:
            gate_proj_out = gelu_apprx_sigmoid(gate_proj_out)

        intermediate_state = gate_proj_out * up_proj_out
        block_new = down_proj_mxfp4(intermediate_state, down_weight, down_scale, down_bias, cfg)

        for n in range(B // 128):
            if is_dummy:
                expert_affinity[n] = 0
            block_new[:, n, :] *= expert_affinity[n]
            block_new[:, n, :] += block_old[:, n, :]
            dma_copy block_new[:, n, :] to output[shard_id, token_indices_2D[:, n], :]
    """
    # Per-block disambiguator for op/alloc names. In the dynamic path block_idx is a
    # runtime SBUF tensor (not a Python int), so f"b{block_idx}" is identical across
    # iterations; callers pass a distinct name_tag (e.g. "chunk_0", "rem") for uniqueness.
    tag = name_tag if name_tag else f"b{block_idx}"
    if sbm != None:
        sbm.open_scope(name="compute_block_scope")
        prev_prefix = sbm.get_name_prefix()
        sbm.set_name_prefix(f"{prev_prefix}{tag}_")

    block_expert = load_block_expert(inps.block_to_expert, block_idx, sbm=sbm)

    # Delayed output write: flush previous block's results at the beginning of this block
    # so the scatter DMA overlaps with the current block's compute. Dynamic Blocks Path
    if delay_output_write and pending_token_indices != None:
        _write_output_scatter(buffers.block_old, pending_token_indices, outs, dims, shard_id)

    # Use weight-skip expert if provided (set to E when same as previous block's expert)
    weight_expert = block_expert_for_weights if block_expert_for_weights != None else block_expert

    # ── STATIC_MX: per-block scale-LUT gather ──
    # One scalar_offset gather per LUT pulls all per-expert constants for this block:
    #   gup_scale_lut [_pmax, E, 3] → [_pmax, 3] = [in_recip, gate_combined, up_combined]
    #   down_scale_lut [_pmax, E, 2] → [_pmax, 2] = [down_in_recip, down_combined]
    gate_in_quant_recip = None
    gate_combined_dequant = None
    up_combined_dequant = None
    down_in_quant_recip = None
    down_combined_dequant_per_block = None
    if kernel_cfg.is_static_quant:
        # scalar_offset requires uint32; reinterpret block_expert (int32 [1,1]).
        block_expert_u32 = block_expert.view(nl.uint32)

        # gup gather: 3D pattern walks _pmax × 3-slot expert row.
        # indirect_dim=1 has stride 3, so scalar_offset=block_expert shifts by 3*expert (one expert's slot triple).
        gup_per_block = _sbm_alloc(sbm, (_pmax, 3), dtype=nl.float32, name="gup_per_block", align=SBUF_QUADRANT_SIZE)
        nisa.tensor_copy(
            dst=gup_per_block,
            src=inps.gup_scale_lut_sb.ap(
                pattern=[[dims.E * 3, _pmax], [3, 1], [1, 3]],
                offset=0,
                scalar_offset=block_expert_u32,
                indirect_dim=1,
            ),
        )
        gate_in_quant_recip = gup_per_block[:, 0:1]
        gate_combined_dequant = gup_per_block[:, 1:2]
        up_combined_dequant = gup_per_block[:, 2:3]

        # down gather: same shape, count=2 on the inner slot dim.
        down_per_block = _sbm_alloc(sbm, (_pmax, 2), dtype=nl.float32, name="down_per_block", align=SBUF_QUADRANT_SIZE)
        nisa.tensor_copy(
            dst=down_per_block,
            src=inps.down_scale_lut_sb.ap(
                pattern=[[dims.E * 2, _pmax], [2, 1], [1, 2]],
                offset=0,
                scalar_offset=block_expert_u32,
                indirect_dim=1,
            ),
        )
        down_in_quant_recip = down_per_block[:, 0:1]
        down_combined_dequant_per_block = down_per_block[:, 1:2]

    # fp8 pre-quantized hidden: the load/transpose happens in-block (after token indices are
    # available) with no online quantize and no bf16 prefetch buffers, so skip the bf16 pipeline.
    if not kernel_cfg.is_fp8_hidden:
        if next_block_idx != None:
            compute_hidden_index_vector(
                inps, buffers, next_block_idx, dims, kernel_cfg.skip_dma, is_block_idx_dynamic=is_dynamic, sbm=sbm
            )

        # quantize prefetched data. Note that online quantize can only quantize to fp8
        # only quantize here if it is a static block. for dynamic block we quantize immediately after fetching
        if not is_dynamic:
            if kernel_cfg.is_static_quant:
                quantize_block_hidden_state_T_static_mx(buffers, prj_cfg, dims, gate_in_quant_recip)
            else:
                quantize_block_hidden_state_T(buffers, prj_cfg, dims)

        _free_hidden_bufs(sbm, buffers.block_hidden_states, buffers.block_hidden_states_T)
        buffers.block_hidden_states_T = None
        buffers.block_hidden_states = None

    """
    Alloc block_hidden_states and start DMA load early to overlap with up proj.
    block_hidden_states_T alloc + transpose deferred to after activation+multiply
    to prevent compiler from scheduling nc_transpose during up proj.
    """

    buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))
    if not kernel_cfg.is_static_quant:
        # STATIC_MX: hidden_scale_sb is the small dummy [_pmax, _pmax] all-127 buffer; matmul
        # site reads it as a 2D view (per-tile shape), so we skip the per-block reshape.
        # fp8 path: scales are packed (n_packed scale blocks); bf16 path: one tile per H512 tile.
        n_scale_dim = div_ceil(prj_cfg.n_H512_tile, 4) if kernel_cfg.is_fp8_hidden else prj_cfg.n_H512_tile
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, n_scale_dim, dims.B))

    flatten_free_dim = prj_cfg.n_total_I512_tile * dims.B * _q_width
    # Tile by default when I > 512
    use_tiled_gup = dims.I > _I_TILE_SZ
    if use_tiled_gup:
        logger.debug(f"Tiling gate/up weights: I={dims.I} > {_I_TILE_SZ}")

    # will be used in non-tiled path if down projection output can't be created. We will re-use address
    # of the gup weight qtz sb which should be finished, but was loaded outside of gup proj scope.
    _gup_wt_addr = None

    if use_tiled_gup:
        # Skip index vector computation if tile 0 was prefetched during previous block's down proj
        _skip_tile0 = not is_first_block and buffers.gup_tile_buf_a != None
        # Packed scales address via scalar_offset=block_expert, so the per-expert index
        # vector is never consumed, static mx has fixed scales
        if _skip_tile0 or kernel_cfg.is_static_quant or kernel_cfg.use_packed_scales:
            # Tile 0 weights, scales, and index vector already cached from prefetch.
            gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
            gup_token_indices_on_p = inps.p_gup_idx_vector_int32
        else:
            scale_shape = inps.gate_up_proj_scale.shape
            gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
            # Able to reuse index vector calculated here because I_TILE_SIZE >= 512, which means we have the same layout as gup proj
            gup_token_indices_on_p = _generate_expert_index_vector(
                expert_index=weight_expert,
                dst_idx_vector=inps.p_gup_idx_vector,
                scale_factor=scale_shape[1],
                n_quadrants_needed=gup_n_quadrants_needed,
                n_remaining_partition=0,
                name_prefix="gup_expert_index_vector",
                sbm=sbm,
                dst_int32=inps.p_gup_idx_vector_int32,
            )
    else:
        # Skip gup weight/scale/bias load if prefetched during previous block's down proj.
        # Works for both static and dynamic paths: block N-1's down proj prefetches block N's
        # weights into hoisted_gup_weights, and block N can skip its main load.
        _skip_gup_load = not is_first_block and hoisted_gup_weights != None
        # Save stack addr before gup weight alloc for potential reuse by down proj output
        _gup_wt_addr = sbm.stack_curr_addr if sbm != None and hoisted_gup_weights == None else None
        if _skip_gup_load:
            # Weights/scales/bias and index vector already cached from prefetch.
            gate_and_up_weights = hoisted_gup_weights
            gup_bias = hoisted_gup_bias
            gup_n_quadrants_needed = prj_cfg.H0 // SBUF_QUADRANT_SIZE
            gup_token_indices_on_p = inps.p_gup_idx_vector_int32
            gate_and_up_scales = inps.gup_scales_sb
        else:
            gate_and_up_weights, gate_and_up_scales, gup_bias, gup_token_indices_on_p, gup_n_quadrants_needed = (
                load_gup_weights_scales_mx(
                    inps,
                    weight_expert,
                    dims,
                    prj_cfg=prj_cfg,
                    skip_dma=kernel_cfg.skip_dma,
                    sbm=sbm,
                    dst_weight=hoisted_gup_weights,
                    dst_bias=hoisted_gup_bias,
                    use_packed_scales=kernel_cfg.use_packed_scales,
                    skip_scales=kernel_cfg.is_static_quant,
                )
            )

    # For non-tiled path, load down weights early to overlap with gup compute
    # For tiled path, defer to after gup scope to avoid DMA contention with tile prefetches
    if not use_tiled_gup:
        down_scale_sb, down_bias_sb = load_down_proj_weights_mx(
            inps,
            weight_expert,
            buffers.down_weight_qtz,
            dims,
            prj_cfg,
            kernel_cfg.skip_dma,
            gup_token_indices_on_p,
            gup_n_quadrants_needed,
            dst_scale=buffers.down_scale_sb,
            sbm=sbm,
            dst_bias=hoisted_down_bias,
            use_packed_scales=kernel_cfg.use_packed_scales,
            skip_scales=kernel_cfg.is_static_quant,
        )
    down_weight_qtz_viewed = buffers.down_weight_qtz

    if is_dynamic:
        token_indices_2D = load_token_indices_dynamic_block(
            inps.token_position_to_id, block_idx, dims.B, dims.n_B128_tiles, skip_dma=kernel_cfg.skip_dma, sbm=sbm
        )
    else:
        token_indices_2D = load_token_indices(inps.token_position_to_id, block_idx, dims.B, dims.n_B128_tiles, sbm=sbm)

    kernel_assert(
        token_indices_2D.shape == (_pmax, dims.n_B128_tiles),
        f"Expect token_indices_2D to have shape (128, {dims.n_B128_tiles}), got {token_indices_2D.shape}",
    )

    # fp8 pre-quantized hidden: this block's hidden was already gathered + transposed into
    # hidden_qtz_sb / hidden_scale_sb during the previous block's deferred stage (or driver init for
    # the first block). The next block's gather (DMA) is issued below during gate/up, and its
    # transpose (PE) is deferred to the sbuf_layout_adapter site so it overlaps down projection.

    # load previous block for accumulation
    if not is_first_block:
        block_old = load_prev_block(
            outs.output,
            token_indices_2D,
            buffers.block_old,
            dims.B // _pmax,
            kernel_cfg.compute_dtype,
            shard_id,
            kernel_cfg.skip_dma,
        )

    if kernel_cfg.is_affinities_packed:
        # Extract expert affinities for this block expert
        expert_affinity = _extract_block_affinity(buffers.block_hidden_concat, block_expert, dims, kernel_cfg, sbm=sbm)
    else:
        expert_affinity = calculate_expert_affinities(
            inps.expert_affinities_masked,
            token_indices_2D,
            block_expert,
            dims.E,
            dims.B // _pmax,
            nl.float32,
            kernel_cfg.skip_dma,
            sbm=sbm,
        )

    if next_block_idx != None and not USE_DMA_TRANSPOSE and not kernel_cfg.is_fp8_hidden:
        _alloc_hidden_src_buf(sbm, buffers, dims, prj_cfg, kernel_cfg, tag=f"nb{next_block_idx}_")
        load_hidden_states_mx(
            inps,
            dims,
            kernel_cfg.skip_dma,
            token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
            block_hidden_states=buffers.block_hidden_states,
            use_dma_transpose=False,
            sbm=sbm,
        )

    # fp8: gather (DMA only) the next block's concat rows here so it overlaps gate/up compute.
    # The matching transpose (PE) is deferred to the sbuf_layout_adapter site below.
    if next_block_idx != None and kernel_cfg.is_fp8_hidden:
        _prev_pfx = sbm.get_name_prefix() if sbm != None else None
        if sbm != None:
            sbm.set_name_prefix(f"{_prev_pfx}fp8pf_")
        if is_dynamic:
            _nb_tok = load_token_indices_dynamic_block(
                inps.token_position_to_id,
                next_block_idx,
                dims.B,
                dims.n_B128_tiles,
                skip_dma=kernel_cfg.skip_dma,
                sbm=sbm,
            )
        else:
            _nb_tok = load_token_indices(inps.token_position_to_id, next_block_idx, dims.B, dims.n_B128_tiles, sbm=sbm)
        load_fp8_hidden_states_mx(
            inps,
            dims,
            kernel_cfg.skip_dma,
            token_indices_on_p=_nb_tok,
            block_hidden_concat=buffers.block_hidden_concat,
        )
        if sbm != None:
            sbm.set_name_prefix(_prev_pfx)
    """
    GATE/UP PROJECTIONS + ACTIVATION + MULTIPLY
    
    Scoped so that gate/up weights, bias, gate_proj_out_sb, up_proj_out_sb,
    and all internal projection allocations are freed after producing intermediate_state_sb.
    """
    # intermediate_state_sb is allocated outside the gate/up scope so it survives for down projection
    intermediate_state_sb = _sbm_alloc(
        sbm,
        (_pmax, flatten_free_dim),
        dtype=nl.bfloat16,
        name="intermediate_state_sb",
        align=SBUF_QUADRANT_SIZE,
    )

    if sbm != None:
        sbm.open_scope(name="gup_proj")

    if use_tiled_gup:
        """
        Tiled gate/up projection: tile along I dimension with double-buffering.

        When full gate/up weights don't fit in SBUF, we tile along the I (intermediate)
        dimension in chunks of _I_TILE_SZ (512). Two weight buffers (A, B) alternate
        so DMA load of the next tile overlaps with compute on the current tile.

        Timeline (3 I-tiles example):
            buf_A: [load T0]──────[compute T0 gate]─[compute T0 up]──────────────────[load T2]──[compute T2 gate]─[compute T2 up]
            buf_B: ───────────────[load T1]──────────────────────────[compute T1 gate]─[compute T1 up]

        Memory layout per tile:
            weight buf: (_pmax, 2, n_H512_tile_sharded, _I_TILE_SZ)  -- gate+up interleaved
            scales:     loaded once for full I, sliced per tile
            bias:       loaded once for full I, sliced per tile
            output:     (_pmax, n_I_tiles, B, _q_width) -- one slice per tile

        Per-tile pipeline:
            1. gate_proj  = hidden @ weight[gate_tile] + bias[tile]
            2. (prefetch next tile weight into alternate buffer)
            3. up_proj    = hidden @ weight[up_tile] + bias[tile]
            4. clamp → activate → gate * up → intermediate_state[tile]
        """
        n_I_tiles = div_ceil(dims.I, _I_TILE_SZ)
        wt_dtype = inps.gate_up_proj_weight.dtype
        last_tile_partial = (dims.I % _I_TILE_SZ) != 0
        tile_buf_shape = (_pmax, 2, prj_cfg.n_H512_tile_sharded, _I_TILE_SZ)

        """
        Use persistent tile buffer + one scope-local buffer for double buffering.
        If persistent buffer wasn't pre-allocated (SBUF budget too tight to keep it
        alive through down proj), allocate a scope-local buffer instead — loses
        cross-block prefetch but still enables double-buffering within the I-tile loop.       
        """

        gup_wt_a = (
            buffers.gup_tile_buf_a
            if buffers.gup_tile_buf_a != None
            else _sbm_alloc(
                sbm,
                tile_buf_shape,
                dtype=wt_dtype,
                name="gup_wt_a",
                align=SBUF_QUADRANT_SIZE,
            )
        )

        # Single-buffering aliases both ping-pong slots, which is only correct when the
        # regime-3 inter-tile prefetch (the sole nxt_buf write, gated `not _use_h_chunked`)
        # is disabled — i.e. the H-chunked regime (H >= 3072). Allocate the second buffer
        # unless we're both in that regime AND out of SBUF room.
        _h_chunked_active = dims.H >= 3072 and not kernel_cfg.gup_full_persistent
        _tile_buf_bytes = 2 * prj_cfg.n_H512_tile_sharded * _I_TILE_SZ * sizeinbytes(wt_dtype)
        _can_double_buffer = sbm == None or sbm.get_free_space() >= _tile_buf_bytes
        if _h_chunked_active and not _can_double_buffer:
            logger.info(
                f"Single-buffering gate/up tile weights (free={sbm.get_free_space()} B, "
                f"need={_tile_buf_bytes} B for gup_wt_b)."
            )
            gup_wt_bufs = [gup_wt_a, gup_wt_a]
        else:
            gup_wt_b = _sbm_alloc(
                sbm,
                (tile_buf_shape),
                dtype=wt_dtype,
                name="gup_wt_b",
                align=SBUF_QUADRANT_SIZE,
            )
            gup_wt_bufs = [gup_wt_a, gup_wt_b]

        # Load full scales once into pre-allocated inps.gup_scales_sb (skip if prefetched).
        # STATIC_MX skips entirely: inps.gup_scales_sb holds persistent dummy 127 from top-level memset.
        if not _skip_tile0 and not kernel_cfg.is_static_quant:
            if kernel_cfg.use_packed_scales:
                # Packed scale load: HBM is [E, _pmax, n_packed_gup, 2, I]; mirror
                # production's gate/up split per skip_weight.
                scale_shape = inps.gate_up_proj_scale.shape
                n_packed_gup = scale_shape[2]
                if kernel_cfg.skip_dma.skip_weight:
                    nisa.dma_copy(
                        dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, :2, : prj_cfg.I],
                        src=inps.gate_up_proj_scale.ap(
                            pattern=[
                                [n_packed_gup * 2 * prj_cfg.I, _pmax],
                                [2 * prj_cfg.I, n_packed_gup],
                                [prj_cfg.I, 2],
                                [1, prj_cfg.I],
                            ],
                            offset=0,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.skip,
                        dge_mode=dge_mode.hwdge,
                        name=f"dma_gup_scales_packed_tile0_{tag}",
                    )
                else:
                    nisa.dma_copy(
                        dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, 0:1, : prj_cfg.I],
                        src=inps.gate_up_proj_scale.ap(
                            pattern=[
                                [n_packed_gup * 2 * prj_cfg.I, _pmax],
                                [2 * prj_cfg.I, n_packed_gup],
                                [1, prj_cfg.I],
                            ],
                            offset=0,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.error,
                        dge_mode=dge_mode.hwdge,
                        name=f"dma_gate_scales_packed_tile0_{tag}",
                    )
                    nisa.dma_copy(
                        dst=inps.gup_scales_sb[:_pmax, :n_packed_gup, 1:2, : prj_cfg.I],
                        src=inps.gate_up_proj_scale.ap(
                            pattern=[
                                [n_packed_gup * 2 * prj_cfg.I, _pmax],
                                [2 * prj_cfg.I, n_packed_gup],
                                [1, prj_cfg.I],
                            ],
                            offset=prj_cfg.I,
                            scalar_offset=block_expert,
                            indirect_dim=0,
                        ),
                        oob_mode=oob_mode.error,
                        dge_mode=dge_mode.hwdge,
                        name=f"dma_up_scales_packed_tile0_{tag}",
                    )
            else:
                scale_shape = inps.gate_up_proj_scale.shape
                gup_scale_view = inps.gate_up_proj_scale.reshape(
                    (scale_shape[0] * scale_shape[1], scale_shape[2], scale_shape[3], scale_shape[4])
                )
                full_n_H512_tile_scale = scale_shape[3]
                stride_dim0 = 2 * full_n_H512_tile_scale * prj_cfg.I
                nisa.dma_copy(
                    src=gup_scale_view.ap(
                        pattern=[
                            [stride_dim0, _pmax],
                            [full_n_H512_tile_scale * prj_cfg.I, 2],
                            [prj_cfg.I, prj_cfg.n_H512_tile_sharded],
                            [1, prj_cfg.I],
                        ],
                        offset=0,
                        vector_offset=gup_token_indices_on_p.ap(
                            [[1, _pmax], [1, 1]],
                            offset=0,
                        ),
                        indirect_dim=0,
                    ),
                    dst=inps.gup_scales_sb[:_pmax, :2, : prj_cfg.n_H512_tile_sharded, : prj_cfg.I],
                    oob_mode=oob_mode.skip,
                )

        # Load full bias once. If hoisted_gup_bias is provided (skip_weight path), reuse
        # it across blocks and use weight_expert so OOB-skip preserves previous expert's bias.
        gup_bias_full = None
        if inps.gate_and_up_proj_bias:
            if hoisted_gup_bias != None:
                gup_bias_full = hoisted_gup_bias
                _bias_expert = weight_expert
            else:
                gup_bias_full = _sbm_alloc(
                    sbm,
                    (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                    dtype=inps.gate_and_up_proj_bias.dtype,
                    name="gup_bias_full",
                    align=SBUF_QUADRANT_SIZE,
                )
                _bias_expert = block_expert
            if dims.I < _pmax * _q_width:
                if hoisted_gup_bias == None:
                    nisa.memset(dst=gup_bias_full[:, :, 0, :], value=0.0)
                # This is due to needing 4_I for QMX
                I_par_dim = dims.I // 4
                bias_stride_dim0 = 2 * prj_cfg.n_total_I512_tile * _q_width
                bias_stride_dim1 = prj_cfg.n_total_I512_tile * _q_width
                nisa.dma_copy(
                    dst=gup_bias_full[:I_par_dim, :, :, :],
                    src=inps.gate_and_up_proj_bias.ap(
                        pattern=[
                            [bias_stride_dim0, I_par_dim],
                            [bias_stride_dim1, 2],
                            [_q_width, prj_cfg.n_total_I512_tile],
                            [1, _q_width],
                        ],
                        offset=0,
                        scalar_offset=_bias_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if kernel_cfg.skip_dma.skip_weight else oob_mode.error,
                    dge_mode=dge_mode.hwdge,
                )
            else:
                bias_stride_dim1 = 2 * prj_cfg.n_total_I512_tile * _q_width
                bias_stride_dim2 = prj_cfg.n_total_I512_tile * _q_width
                nisa.dma_copy(
                    dst=gup_bias_full,
                    src=inps.gate_and_up_proj_bias.ap(
                        pattern=[
                            [bias_stride_dim1, _pmax],
                            [bias_stride_dim2, 2],
                            [_q_width, prj_cfg.n_total_I512_tile],
                            [1, _q_width],
                        ],
                        offset=0,
                        scalar_offset=_bias_expert,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if kernel_cfg.skip_dma.skip_weight else oob_mode.error,
                    dge_mode=dge_mode.hwdge,
                )

        # Single scratch buffer for up projection output. Gate writes directly to
        # intermediate_state_sb, then gate * up overwrites it in place.
        up_tile_sb = _sbm_alloc(
            sbm,
            (_pmax, 1, dims.B, _q_width),
            dtype=nl.bfloat16,
            name="up_tile_sb",
            align=SBUF_QUADRANT_SIZE,
        )

        # Pre-build ProjConfig for full tiles; update I-fields for last tile if partial
        tile_prj_cfg = ProjConfig(
            H=dims.H,
            I=_I_TILE_SZ,
            BxS=dims.B,
            force_lnc1=True,
            n_prgs=1,
            prg_id=0,
            use_stream_shuffle_broadcast=False,
            sharding_config="H",
            zero_unused_partitions=prj_cfg.zero_unused_partitions,
        )

        # Initial weight load:
        #   - full_resident path: load FULL gup (gate+up, full I) into the persistent
        #     buffer when no prior cross-block prefetch supplied them (first block of a shard).
        #   - tile-streaming path: load just tile 0; subsequent tiles stream during the loop.
        if not _skip_tile0:
            if kernel_cfg.gup_full_persistent:
                _prefetch_gup_tile0(
                    inps,
                    block_expert,
                    prj_cfg,
                    buffers,
                    dims,
                    kernel_cfg.skip_dma,
                    sbm,
                    name_prefix=f"first_b{block_idx}",
                    use_packed_scales=kernel_cfg.use_packed_scales,
                    skip_scales=kernel_cfg.is_static_quant,
                    full_I_load=True,
                )
            else:
                _load_gup_weight_tile(
                    inps=inps,
                    block_expert=block_expert,
                    prj_cfg=prj_cfg,
                    skip_dma=kernel_cfg.skip_dma,
                    dst_weight=gup_wt_bufs[0],
                    dst_scale=None,
                    dst_bias=None,
                    token_indices_on_p=gup_token_indices_on_p,
                    I_offset=0,
                    dims=dims,
                    sbm=sbm,
                )

        # Pre-build a flat view of the full-resident gup buffer for slicing per I-tile.
        if kernel_cfg.gup_full_persistent:
            _gup_resident_flat = buffers.gup_tile_buf_a.flatten_dims(1, 2)

        # Use H-chunked weight loading for tiles 1+ when H is large (tile-streaming path only)
        _use_h_chunked = dims.H >= 3072 and not kernel_cfg.gup_full_persistent

        # STATIC_MX/SW-dequant path.
        # gup_scales layout: standard [_pmax, 2, n_H512_tile_sharded, I] flattened
        # to [_pmax, 2*n_H512, I] for gate/up slicing. Packed layout
        # [_pmax, n_packed_gup, 2, I] needs no flatten. STATIC_MX uses a shared
        # all-127 dummy buffer that the sub-kernel reads directly.
        if kernel_cfg.use_packed_scales or kernel_cfg.is_static_quant:
            gup_scales_flat = None
        else:
            gup_scales_flat = inps.gup_scales_sb.reshape((_pmax, 2 * prj_cfg.n_H512_tile_sharded, dims.I))
        gup_bias_flat = (
            gup_bias_full.reshape((_pmax, 2 * prj_cfg.n_total_I512_tile, _q_width)) if gup_bias_full != None else None
        )
        intermediate_state_tiled = intermediate_state_sb.reshape((_pmax, prj_cfg.n_total_I512_tile, dims.B, _q_width))

        # Loop-invariant: gate activation op depends only on kernel_cfg, not the tile.
        # STATIC_MX fuses silu into the gate projection, but only when there is no gate
        # clamp (the clamp must run between dequant+bias and the activation).
        _gate_act_op = (
            get_nl_act_fn_from_type(kernel_cfg.activation_function)
            if kernel_cfg.is_static_quant and not kernel_cfg.has_gate_clamp
            else None
        )

        # ── Per-I-tile weight sourcing: three regimes, selected by the if/elif/else below ──
        # Each iteration computes one 512-wide I-tile (we are here because I > _I_TILE_SZ).
        # The regimes differ only in WHERE this tile's gate/up weights come from:
        #   1. Full-resident (gup_full_persistent): the entire gate+up (all I-tiles) is
        #      resident in one big SBUF buffer that persists across blocks; each tile is a
        #      slice at cur_I_offset, no per-tile DMA. Enabled only for STATIC_MX +
        #      skip_weight + I>512 that fits the SBUF budget (see _gup_full_persistent).
        #   2. H-chunked (_use_h_chunked and i_tile > 0; requires H >= 3072): weights are
        #      streamed from HBM via gate_up_projection_mx_tp's own dst_weight_sb DMA,
        #      interleaved with matmul. Tile 0 still falls to regime 3.
        #   3. Ping-pong tile (else): the default streaming scheme — the buffer holds one
        #      I-tile (gup_wt_a/gup_wt_b alternate). Tile 0 is the cross-block prefetch
        #      already in SBUF; tiles 1+ are staged into the alternate buffer by the
        #      _load_gup_weight_tile call below (gated on not _use_h_chunked). Handles tile 0
        #      always, and every tile when H < 3072.
        for i_tile in nl.affine_range(n_I_tiles):
            cur_buf = i_tile % 2
            nxt_buf = 1 - cur_buf
            cur_I_offset = i_tile * _I_TILE_SZ
            cur_I_tile_sz = min(_I_TILE_SZ, dims.I - cur_I_offset)
            is_last_tile = i_tile == n_I_tiles - 1

            # Update ProjConfig I-fields for last partial tile
            if is_last_tile and last_tile_partial:
                tile_prj_cfg.I = cur_I_tile_sz
                tile_prj_cfg._generate_H_shard_config()

            gate_bias_view = None
            if inps.gate_and_up_proj_bias:
                gate_bias_view = gup_bias_flat.slice(dim=1, start=i_tile, end=i_tile + 1)
            up_bias_view = None
            if inps.gate_and_up_proj_bias:
                up_bias_view = gup_bias_flat.slice(
                    dim=1, start=prj_cfg.n_total_I512_tile + i_tile, end=prj_cfg.n_total_I512_tile + i_tile + 1
                )

            if kernel_cfg.gup_full_persistent:
                # Regime 1 (full-resident): slice this I-tile from the persistent buffer; no DMA.
                gate_wt_view = _gup_resident_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(
                    2, cur_I_offset, cur_I_offset + cur_I_tile_sz
                )
                up_wt_view = _gup_resident_flat.slice(
                    1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
                ).slice(2, cur_I_offset, cur_I_offset + cur_I_tile_sz)

                if kernel_cfg.is_static_quant:
                    gate_weight_scale_arg = inps.gup_scales_sb
                    up_weight_scale_arg = inps.gup_scales_sb
                elif kernel_cfg.use_packed_scales:
                    gate_weight_scale_arg = inps.gup_scales_sb[:, :, 0, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                    up_weight_scale_arg = inps.gup_scales_sb[:, :, 1, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                else:
                    gate_weight_scale_arg = gup_scales_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(
                        2, cur_I_offset, cur_I_offset + cur_I_tile_sz
                    )
                    up_weight_scale_arg = gup_scales_flat.slice(
                        1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
                    ).slice(2, cur_I_offset, cur_I_offset + cur_I_tile_sz)

                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=gate_wt_view,
                    weight_scale=gate_weight_scale_arg,
                    bias_sb=gate_bias_view,
                    cfg=tile_prj_cfg,
                    sbm=sbm,
                    psum_bank_offset=0 if i_tile % 2 == 0 else 1,
                    name_prefix=f"gate_t{i_tile}",
                    out_sb=intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=gate_combined_dequant,
                    activation_op=_gate_act_op,
                    is_static_quant=kernel_cfg.is_static_quant,
                )

                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=up_wt_view,
                    weight_scale=up_weight_scale_arg,
                    bias_sb=up_bias_view,
                    cfg=tile_prj_cfg,
                    sbm=sbm,
                    psum_bank_offset=2 if i_tile % 2 == 0 else 3,
                    name_prefix=f"up_t{i_tile}",
                    out_sb=up_tile_sb[:_pmax, 0:1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=up_combined_dequant,
                    activation_op=None,
                    is_static_quant=kernel_cfg.is_static_quant,
                )
            elif _use_h_chunked and i_tile > 0:
                # Regime 2 (H-chunked): stream this I-tile's weight from HBM in chunks,
                # interleaving DMA with matmul. HBM weight view (gate half, then up).
                gate_wt_hbm = (
                    inps.gate_up_proj_weight.select(dim=0, index=block_expert)
                    .select(dim=1, index=0)
                    .slice(dim=1, start=0, end=prj_cfg.n_H512_tile_sharded)
                    .slice(dim=2, start=cur_I_offset, end=cur_I_offset + cur_I_tile_sz)
                )
                up_wt_hbm = (
                    inps.gate_up_proj_weight.select(dim=0, index=block_expert)
                    .select(dim=1, index=1)
                    .slice(dim=1, start=0, end=prj_cfg.n_H512_tile_sharded)
                    .slice(dim=2, start=cur_I_offset, end=cur_I_offset + cur_I_tile_sz)
                )

                # Build flattened (gate/up × n_H512) gate/up dst views via
                # nl.NkiTensor so partition stride is preserved through to the AP with base tensor AP stride
                cur_wt_flat = gup_wt_bufs[cur_buf].flatten_dims(1, 2)
                gate_dst = cur_wt_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(2, 0, _I_TILE_SZ)
                up_dst = cur_wt_flat.slice(1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded).slice(
                    2, 0, _I_TILE_SZ
                )

                if kernel_cfg.is_static_quant:
                    gate_weight_scale_arg = inps.gup_scales_sb
                    up_weight_scale_arg = inps.gup_scales_sb
                elif kernel_cfg.use_packed_scales:
                    gate_weight_scale_arg = inps.gup_scales_sb[:, :, 0, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                    up_weight_scale_arg = inps.gup_scales_sb[:, :, 1, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                else:
                    gate_weight_scale_arg = gup_scales_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(
                        2, cur_I_offset, cur_I_offset + cur_I_tile_sz
                    )
                    up_weight_scale_arg = gup_scales_flat.slice(
                        1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
                    ).slice(2, cur_I_offset, cur_I_offset + cur_I_tile_sz)

                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=gate_wt_hbm,
                    weight_scale=gate_weight_scale_arg,
                    dst_weight_sb=gate_dst,
                    bias_sb=gate_bias_view,
                    cfg=tile_prj_cfg,
                    skip_dma=kernel_cfg.skip_dma,
                    sbm=sbm,
                    psum_bank_offset=0 if i_tile % 2 == 0 else 1,
                    name_prefix=f"gate_t{i_tile}",
                    out_sb=intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=gate_combined_dequant,
                    activation_op=_gate_act_op,
                    is_static_quant=kernel_cfg.is_static_quant,
                )

                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=up_wt_hbm,
                    weight_scale=up_weight_scale_arg,
                    dst_weight_sb=up_dst,
                    bias_sb=up_bias_view,
                    cfg=tile_prj_cfg,
                    skip_dma=kernel_cfg.skip_dma,
                    sbm=sbm,
                    psum_bank_offset=2 if i_tile % 2 == 0 else 3,
                    name_prefix=f"up_t{i_tile}",
                    out_sb=up_tile_sb[:_pmax, 0:1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=up_combined_dequant,
                    activation_op=None,
                    is_static_quant=kernel_cfg.is_static_quant,
                )
            else:
                # Regime 3 (ping-pong tile): weights already in the SBUF tile buffer —
                # tile 0 from the cross-block prefetch, or any tile when H < 3072.
                cur_wt_flat = gup_wt_bufs[cur_buf].flatten_dims(1, 2)

                if kernel_cfg.is_static_quant:
                    gate_weight_scale_arg = inps.gup_scales_sb
                elif kernel_cfg.use_packed_scales:
                    gate_weight_scale_arg = inps.gup_scales_sb[:, :, 0, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                else:
                    gate_weight_scale_arg = gup_scales_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(
                        2, cur_I_offset, cur_I_offset + cur_I_tile_sz
                    )
                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=cur_wt_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded).slice(2, 0, cur_I_tile_sz),
                    weight_scale=gate_weight_scale_arg,
                    bias_sb=gate_bias_view,
                    cfg=tile_prj_cfg,
                    sbm=sbm,
                    psum_bank_offset=0 if i_tile % 2 == 0 else 1,
                    name_prefix=f"gate_t{i_tile}",
                    out_sb=intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=gate_combined_dequant,
                    activation_op=_gate_act_op,
                    is_static_quant=kernel_cfg.is_static_quant,
                )

                # Prefetch next tile between gate and up (only for standard path)
                if i_tile < n_I_tiles - 1 and not _use_h_chunked:
                    nxt_I_offset = (i_tile + 1) * _I_TILE_SZ
                    _load_gup_weight_tile(
                        inps=inps,
                        block_expert=block_expert,
                        prj_cfg=prj_cfg,
                        skip_dma=kernel_cfg.skip_dma,
                        dst_weight=gup_wt_bufs[nxt_buf],
                        dst_scale=None,
                        dst_bias=None,
                        token_indices_on_p=gup_token_indices_on_p,
                        I_offset=nxt_I_offset,
                        dims=dims,
                        sbm=sbm,
                    )

                if kernel_cfg.is_static_quant:
                    up_weight_scale_arg = inps.gup_scales_sb
                elif kernel_cfg.use_packed_scales:
                    up_weight_scale_arg = inps.gup_scales_sb[:, :, 1, cur_I_offset : cur_I_offset + cur_I_tile_sz]
                else:
                    up_weight_scale_arg = gup_scales_flat.slice(
                        1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded
                    ).slice(2, cur_I_offset, cur_I_offset + cur_I_tile_sz)
                gate_up_projection_mx_tp(
                    hidden_qtz_sb=buffers.hidden_qtz_sb,
                    hidden_scale_sb=buffers.hidden_scale_sb,
                    weight_qtz=cur_wt_flat.slice(1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded).slice(
                        2, 0, cur_I_tile_sz
                    ),
                    weight_scale=up_weight_scale_arg,
                    bias_sb=up_bias_view,
                    cfg=tile_prj_cfg,
                    sbm=sbm,
                    psum_bank_offset=2 if i_tile % 2 == 0 else 3,
                    name_prefix=f"up_t{i_tile}",
                    out_sb=up_tile_sb[:_pmax, 0:1, : dims.B, :_q_width],
                    is_packed_scale=kernel_cfg.use_packed_scales,
                    is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
                    w_dequant_scale=up_combined_dequant,
                    activation_op=None,
                    is_static_quant=kernel_cfg.is_static_quant,
                )

            # Per-tile: clip, activate gate (in intermediate_state_sb), clip up, then gate * up → intermediate_state_sb
            gate_tile = intermediate_state_tiled[:_pmax, i_tile : i_tile + 1, : dims.B, :_q_width]
            up_tile = up_tile_sb[:_pmax, 0:1, : dims.B, :_q_width]

            apply_clamp(gate_tile, kernel_cfg.gate_clamp_upper_limit, kernel_cfg.gate_clamp_lower_limit)
            apply_clamp(up_tile, kernel_cfg.up_clamp_upper_limit, kernel_cfg.up_clamp_lower_limit)
            # STATIC_MX with no gate clamp: silu was already fused into the gate projection above.
            if not (kernel_cfg.is_static_quant and not kernel_cfg.has_gate_clamp):
                nisa.activation(
                    dst=gate_tile,
                    op=get_nl_act_fn_from_type(kernel_cfg.activation_function),
                    data=gate_tile,
                    scale=1.0,
                    bias=inps.activation_bias,
                )
            nisa.tensor_tensor(gate_tile, gate_tile, up_tile, op=nl.multiply)

    else:
        # ── Non-tiled path ──
        # Build the flattened (gate/up × n_H512) view via flatten_dims rather than reshape.
        gup_weights_flat = gate_and_up_weights.flatten_dims(1, 2)
        if kernel_cfg.use_packed_scales or kernel_cfg.is_static_quant:
            # STATIC_MX: gate_and_up_scales is the shared [_pmax, _pmax] all-127 dummy;
            # the flatten is unused (call sites pass the buffer directly).
            gup_scales_flat = None
        else:
            gup_scales_flat = gate_and_up_scales.flatten_dims(1, 2)
        gate_bias_view = None
        up_bias_view = None
        if gup_bias:
            gup_bias_flat = gup_bias.flatten_dims(1, 2)
            gate_bias_view = gup_bias_flat.slice(dim=1, start=0, end=prj_cfg.n_total_I512_tile)

        if kernel_cfg.is_static_quant:
            gate_weight_scale_arg = inps.gup_scales_sb
        elif kernel_cfg.use_packed_scales:
            gate_weight_scale_arg = inps.gup_scales_sb[:, :, 0, :]
        else:
            gate_weight_scale_arg = gup_scales_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded)
        # STATIC_MX silu fusion: only when no gate clamp.
        _gate_act_op = (
            get_nl_act_fn_from_type(kernel_cfg.activation_function)
            if kernel_cfg.is_static_quant and not kernel_cfg.has_gate_clamp
            else None
        )
        gate_proj_out_sb = gate_up_projection_mx_tp(
            hidden_qtz_sb=buffers.hidden_qtz_sb,
            hidden_scale_sb=buffers.hidden_scale_sb,
            weight_qtz=gup_weights_flat.slice(1, 0, prj_cfg.n_H512_tile_sharded),
            weight_scale=gate_weight_scale_arg,
            bias_sb=gate_bias_view,
            cfg=prj_cfg,
            sbm=sbm,
            psum_bank_offset=0,
            name_prefix="gate",
            is_packed_scale=kernel_cfg.use_packed_scales,
            is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
            w_dequant_scale=gate_combined_dequant,
            activation_op=_gate_act_op,
            is_static_quant=kernel_cfg.is_static_quant,
        )

        gate_proj_out_sb = gate_proj_out_sb.reshape((_pmax, flatten_free_dim))

        if gup_bias:
            up_bias_view = gup_bias_flat.slice(
                dim=1, start=prj_cfg.n_total_I512_tile, end=2 * prj_cfg.n_total_I512_tile
            )

        if kernel_cfg.is_static_quant:
            up_weight_scale_arg = inps.gup_scales_sb
        elif kernel_cfg.use_packed_scales:
            up_weight_scale_arg = inps.gup_scales_sb[:, :, 1, :]
        else:
            up_weight_scale_arg = gup_scales_flat.slice(1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded)
        up_proj_out_sb = gate_up_projection_mx_tp(
            hidden_qtz_sb=buffers.hidden_qtz_sb,
            hidden_scale_sb=buffers.hidden_scale_sb,
            weight_qtz=gup_weights_flat.slice(1, prj_cfg.n_H512_tile_sharded, 2 * prj_cfg.n_H512_tile_sharded),
            weight_scale=up_weight_scale_arg,
            bias_sb=up_bias_view,
            cfg=prj_cfg,
            sbm=sbm,
            psum_bank_offset=4,
            name_prefix="up",
            is_packed_scale=kernel_cfg.use_packed_scales,
            is_packed_moving_scale=kernel_cfg.is_fp8_hidden,
            w_dequant_scale=up_combined_dequant,
            activation_op=None,
            is_static_quant=kernel_cfg.is_static_quant,
        )

        up_proj_out_sb = up_proj_out_sb.reshape((_pmax, flatten_free_dim))

    # clipping gate (non-tiled path only; tiled path does this per-tile)
    if not use_tiled_gup:
        apply_clamp(
            gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
            kernel_cfg.gate_clamp_upper_limit,
            kernel_cfg.gate_clamp_lower_limit,
        )

    # clipping up (non-tiled path only)
    if not use_tiled_gup:
        apply_clamp(
            up_proj_out_sb[0:_pmax, 0:flatten_free_dim],
            kernel_cfg.up_clamp_upper_limit,
            kernel_cfg.up_clamp_lower_limit,
        )

    # bf16 path reshapes the persistent hidden buffers back to the quantize layout [.., B//32, 32]
    # for the next-block online quantize below. fp8 path has no online quantize (gather fills the
    # buffers in-block), so it leaves them in the [.., n_H512_tile, B] matmul layout.
    if not kernel_cfg.is_fp8_hidden:
        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        if not kernel_cfg.is_static_quant:
            buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))

    # activation and multiply (non-tiled path only; tiled path does this per-tile).
    # STATIC_MX with no gate clamp: silu was already fused into the gate projection.
    if not use_tiled_gup:
        if not (kernel_cfg.is_static_quant and not kernel_cfg.has_gate_clamp):
            nisa.activation(
                dst=gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
                op=get_nl_act_fn_from_type(kernel_cfg.activation_function),
                data=gate_proj_out_sb[0:_pmax, 0:flatten_free_dim],
                scale=1.0,
                bias=inps.activation_bias,
            )

        nisa.tensor_tensor(
            intermediate_state_sb[:_pmax, :flatten_free_dim],
            gate_proj_out_sb[:_pmax, :flatten_free_dim],
            up_proj_out_sb[:_pmax, :flatten_free_dim],
            op=nl.multiply,
        )

    if sbm != None:
        sbm.close_scope()  # frees gate/up weights, bias, gate_proj_out_sb, up_proj_out_sb, and projection internals

    intermediate_state_sb = intermediate_state_sb.reshape((_pmax, prj_cfg.n_total_I512_tile, dims.B, _q_width))

    """
    TRANSPOSE AND QUANTIZE NEXT BLOCK HIDDEN STATES
    DMA load was started earlier (during up proj). Now allocate block_hidden_states_T
    and run sbuf_layout_adapter. Deferred to here so nc_transpose doesn't contend
    with up projection on the tensor engine.
    """
    # Hoist next-block expert load: shared by static_mx prefetch quant (recip) and gup-weight prefetch below.
    _pf_expert = None
    if next_block_idx != None:
        _pf_expert = load_block_expert(inps.block_to_expert, next_block_idx, sbm=sbm, name="pf_block_expert")

        # fp8 pre-quantized hidden: the next block was gathered (DMA) during gate/up above. Transpose
        # it (PE) into hidden_qtz_sb / hidden_scale_sb now — deferred here so nc_transpose overlaps
        # down projection instead of contending with gate/up. No online quantize.
        if kernel_cfg.is_fp8_hidden:
            transpose_fp8_hidden_states(
                dims,
                prj_cfg,
                buffers.block_hidden_concat,
                hidden_qtz_sb=buffers.hidden_qtz_sb,
                hidden_scale_sb=buffers.hidden_scale_sb,
                sbm=sbm,
            )
        elif USE_DMA_TRANSPOSE:
            # DMA transpose path: only block_hidden_states_T is needed
            # (DMA transpose writes directly into the transposed layout).
            _alloc_hidden_T_buf(sbm, buffers, dims, prj_cfg, kernel_cfg, tag=f"nb{next_block_idx}_")
            load_hidden_states_mx(
                inps,
                dims,
                kernel_cfg.skip_dma,
                token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                block_hidden_states_T=buffers.block_hidden_states_T,
                use_dma_transpose=True,
                sbm=sbm,
            )
        else:
            # PE transpose path: block_hidden_states already loaded, now alloc _T and transpose
            _alloc_hidden_T_buf(sbm, buffers, dims, prj_cfg, kernel_cfg, tag=f"nb{next_block_idx}_")
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims, sbm=sbm)

        if is_dynamic and not kernel_cfg.is_fp8_hidden:
            if kernel_cfg.is_static_quant:
                # Reuse hoisted _pf_expert to gather next block's 1/in_scale[expert] from gup_scale_lut_sb.
                _nb_in_quant_recip = _gather_gup_in_quant_recip(inps, _pf_expert, dims, sbm, name_prefix="nb_")
                quantize_block_hidden_state_T_static_mx(buffers, prj_cfg, dims, _nb_in_quant_recip)
            else:
                quantize_block_hidden_state_T(buffers, prj_cfg, dims)
            # Quantized data is in persistent hidden_qtz_sb/hidden_scale_sb;
            # pop blk_hs_T (and blk_hs if present) off heap to reclaim space for down_proj.
            sbm.pop_heap()  # blk_hs_T
            buffers.block_hidden_states_T = None
            if not USE_DMA_TRANSPOSE:
                sbm.pop_heap()  # blk_hs (PE mode only; DMA mode doesn't allocate it)
                buffers.block_hidden_states = None

    """
    DOWN PROJECTION
    """
    # Tiled path: load down weights here (deferred from before gup to avoid DMA contention)
    if use_tiled_gup:
        down_scale_sb, down_bias_sb = load_down_proj_weights_mx(
            inps,
            weight_expert,  # skip-aware: OOB-skips weight/scale/bias DMAs for same-expert consecutive blocks (down_weight_qtz is persistent)
            buffers.down_weight_qtz,
            dims,
            prj_cfg,
            kernel_cfg.skip_dma,
            gup_token_indices_on_p,
            gup_n_quadrants_needed,
            dst_scale=buffers.down_scale_sb,
            sbm=sbm,
            dst_bias=hoisted_down_bias,
            use_packed_scales=kernel_cfg.use_packed_scales,
            skip_scales=kernel_cfg.is_static_quant,
        )

    # Prefetch next block's gup weights/scales into persistent buffers (overlaps with down proj)
    if next_block_idx != None:
        # Skip-aware expert for the prefetch must be precomputed by the caller.
        # Static and dynamic paths both pass `next_block_expert_for_weights`
        # built from per-region weight-skip masks; the dynamic chunked->rem
        # boundary hoists a 4-op compare in the loop body.
        _pf_skip_expert = next_block_expert_for_weights if kernel_cfg.skip_dma.skip_weight else None

        if use_tiled_gup and buffers.gup_tile_buf_a != None:
            # Full-resident scheme: pass skip-aware expert as `expert` so the WEIGHT
            # DMA OOB-skips on same-expert (preserves the resident buffer across blocks).
            # Today's tile-0-only scheme passes the real expert (the buffer is reloaded
            # per-block anyway, so the skip-aware expert only matters for scales).
            if kernel_cfg.gup_full_persistent:
                _pf_weight_expert = _pf_skip_expert if _pf_skip_expert is not None else _pf_expert
            else:
                _pf_weight_expert = _pf_expert
            _prefetch_gup_tile0(
                inps,
                _pf_weight_expert,
                prj_cfg,
                buffers,
                dims,
                kernel_cfg.skip_dma,
                sbm,
                name_prefix=f"pf_b{next_block_idx}",
                scale_expert=_pf_skip_expert,
                use_packed_scales=kernel_cfg.use_packed_scales,
                skip_scales=kernel_cfg.is_static_quant,
                full_I_load=kernel_cfg.gup_full_persistent,
            )
        elif not use_tiled_gup and hoisted_gup_weights != None:
            if sbm != None:
                sbm.open_scope(name="pf_gup_full")
            load_gup_weights_scales_mx(
                inps,
                _pf_skip_expert or _pf_expert,
                dims,
                prj_cfg=prj_cfg,
                skip_dma=kernel_cfg.skip_dma,
                sbm=sbm,
                dst_weight=hoisted_gup_weights,
                dst_bias=hoisted_gup_bias,
                name_prefix="pf_gup",
                use_packed_scales=kernel_cfg.use_packed_scales,
                skip_scales=kernel_cfg.is_static_quant,
            )
            if sbm != None:
                sbm.close_scope()

    if sbm != None:
        sbm.open_scope(name="down_proj")

    # If not enough space for dp_out_sb, reuse dead gup weight buffer at same address
    _dp_out_sb = None
    if _gup_wt_addr != None and sbm != None:
        n_BxS_tile = dims.B // _pmax
        dp_out_bytes = n_BxS_tile * dims.H * sizeinbytes(nl.bfloat16)
        # check if we can allocate dp_out_sb, otherwise reuse memory location
        if sbm.heap_curr_addr - sbm.stack_curr_addr < dp_out_bytes:
            _dp_out_sb = nl.ndarray(
                (_pmax, n_BxS_tile, dims.H),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"dp_out_sb_reuse_{tag}",
                address=(0, _gup_wt_addr),
            )

    block_new = down_projection_mx(
        inter_sb=intermediate_state_sb,
        weight=down_weight_qtz_viewed,
        weight_scale=down_scale_sb,
        bias_sb=down_bias_sb,
        cfg=prj_cfg,
        sbm=sbm,
        # Start past the banks the hidden transpose used: fp8 transpose
        # (transpose_fp8_hidden_states) occupies banks 0-3, the bf16 transpose
        # (sbuf_layout_adapter) occupies banks 0-1.
        psum_bank_offset=4 if kernel_cfg.is_fp8_hidden else 2,
        name_prefix="dp",
        out_sb=_dp_out_sb,
        is_packed_scale=kernel_cfg.use_packed_scales,
        w_dequant_scale=down_combined_dequant_per_block,
        inter_quant_recip=down_in_quant_recip if kernel_cfg.is_static_quant else None,
        dummy_inter_scale=buffers.dummy_inter_scale_sb if kernel_cfg.is_static_quant else None,
    )

    if sbm != None:
        sbm.close_scope()

    for n in range(dims.B // _pmax):
        if is_dummy:
            if sbm != None:
                sbm.open_scope(name=f"dummy_zeros_{n}")
            zeros = _sbm_alloc(sbm, (_pmax, 1), dtype=nl.float32, name=f"dummy_zeros_{n}", align=SBUF_QUADRANT_SIZE)
            nisa.memset(zeros, value=0.0)
            nisa.tensor_copy(dst=expert_affinity[n][:, :], src=zeros, engine=nisa.vector_engine)
            if sbm != None:
                sbm.close_scope()

        if delay_output_write:
            # Scale block_new in-place, then accumulate into block_old (persists across blocks).
            # block_old[n] = block_new[n] * affinity + block_old[n]
            nisa.tensor_scalar(
                dst=block_new[0:_pmax, n, 0 : dims.H],
                data=block_new[0:_pmax, n, 0 : dims.H],
                op0=nl.multiply,
                operand0=expert_affinity[n][0:_pmax, 0:1],
                engine=nisa.scalar_engine,
            )
            nisa.tensor_tensor(
                dst=buffers.block_old[0:_pmax, n, 0 : dims.H],
                data1=block_new[0:_pmax, n, 0 : dims.H],
                op=nl.add,
                data2=block_old[0:_pmax, n, 0 : dims.H],
            )
        else:
            nisa.tensor_scalar(
                dst=block_new[0:_pmax, n, 0 : dims.H],
                data=block_new[0:_pmax, n, 0 : dims.H],
                op0=nl.multiply,
                operand0=expert_affinity[n][0:_pmax, 0:1],
                engine=nisa.scalar_engine,
            )
            if not is_first_block:
                nisa.tensor_tensor(
                    dst=block_new[0:_pmax, n, 0 : dims.H],
                    data1=block_new[0:_pmax, n, 0 : dims.H],
                    op=nl.add,
                    data2=block_old[0:_pmax, n, 0 : dims.H],
                )

    if delay_output_write:
        # Save token indices for the deferred scatter write at the beginning of the next block.
        nisa.tensor_copy(
            dst=pending_token_indices[0:_pmax, 0 : dims.n_B128_tiles],
            src=token_indices_2D[0:_pmax, 0 : dims.n_B128_tiles],
        )
    else:
        for n in range(dims.B // _pmax):
            T = outs.output.shape[-2]
            shard_offset = shard_id * T * dims.H

            block_token_mapping = token_indices_2D.ap(
                [[dims.n_B128_tiles, _pmax], [1, 1]],
                offset=n,
            )

            num_shards = outs.output.shape[0]

            output_ap = outs.output.reshape((num_shards * T, 1, dims.H)).ap(
                pattern=[[dims.H, _pmax], [1, 1], [1, dims.H]],
                offset=shard_offset,
                vector_offset=block_token_mapping,
                indirect_dim=0,
            )

            nisa.dma_copy(
                dst=output_ap,
                src=block_new[0:_pmax, n, 0 : dims.H],
                oob_mode=oob_mode.skip,
            )

    if sbm != None:
        sbm.set_name_prefix(prev_prefix)
        sbm.close_scope()


def _alloc_hidden_bufs(sbm, buffers, dims, prj_cfg, configs, tag=""):
    """Heap-allocate hidden state buffers.

    - PE mode: allocates both block_hidden_states (pre-transpose) and block_hidden_states_T.
    - DMA mode: allocates only block_hidden_states_T; block_hidden_states is unused.
    """
    _alloc_hidden_src_buf(sbm, buffers, dims, prj_cfg, configs, tag=tag)
    _alloc_hidden_T_buf(sbm, buffers, dims, prj_cfg, configs, tag=tag)


def _alloc_hidden_src_buf(sbm, buffers, dims, prj_cfg, configs, tag=""):
    """Heap-allocate block_hidden_states (pre-transpose) on demand."""
    prev = sbm.get_name_prefix()
    sbm.set_name_prefix(f"{prev}{tag}")
    if not USE_DMA_TRANSPOSE:
        buffers.block_hidden_states = sbm.alloc_heap(
            (_pmax, dims.B // SBUF_QUADRANT_SIZE, prj_cfg.n_H512_tile, _pmax),
            dtype=configs.compute_dtype,
            name="blk_hs",
            align=SBUF_QUADRANT_SIZE,
        )
    sbm.set_name_prefix(prev)


def _alloc_hidden_T_buf(sbm, buffers, dims, prj_cfg, configs, tag=""):
    """Heap-allocate block_hidden_states_T (post-transpose) on demand."""
    prev = sbm.get_name_prefix()
    sbm.set_name_prefix(f"{prev}{tag}")
    buffers.block_hidden_states_T = sbm.alloc_heap(
        (_pmax, prj_cfg.n_H512_tile, dims.B // SBUF_QUADRANT_SIZE, SBUF_QUADRANT_SIZE * _q_width),
        dtype=configs.compute_dtype,
        name="blk_hs_T",
        align=SBUF_QUADRANT_SIZE,
    )
    sbm.set_name_prefix(prev)


def _free_hidden_bufs(sbm, block_hidden_states, block_hidden_states_T=None):
    """Free heap-allocated block_hidden_states_T (and block_hidden_states if present)."""
    if block_hidden_states_T != None:
        sbm.pop_heap()  # block_hidden_states_T (allocated last)
    if block_hidden_states != None:
        sbm.pop_heap()  # block_hidden_states


def process_static_blocks(
    dims: BWMMMXDimensionSizes,
    configs: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    inps: InputTensors,
    outs: OutputTensors,
    buffers: SharedBuffers,
    num_static_blocks: int,
    sbm=None,
    all_experts_for_weights=None,
    hoisted_gup_weights=None,
    hoisted_gup_bias=None,
    hoisted_down_bias=None,
    is_tensor_update_accumulating=True,
):
    """
    Process static (non-padded) blocks with prefetching optimization.

    Iterates through known non-padded blocks with double-buffering to overlap
    computation and data loading.

    Args:
        dims (BWMMMXDimensionSizes): Dimension configuration.
        configs (BWMMMXConfigs): Kernel configuration.
        prj_cfg (ProjConfig): Projection configuration.
        inps (InputTensors): Input tensors.
        outs (OutputTensors): Output tensors.
        buffers (SharedBuffers): Shared buffers.
        num_static_blocks (int): Number of static blocks to process.

    Returns:
        None: Processes blocks and writes to outs.output.

    Notes:
        - Distributes blocks across shards evenly
        - Prefetches next block while processing current
        - Handles odd/even block counts differently
        - Last block has no prefetch
        - Shard 0 processes dummy block when N is odd

    Pseudocode:
        n_blocks_per_shard = num_static_blocks // num_shards
        r_block = num_static_blocks % num_shards
        first_block_idx = n_blocks_per_shard * shard_id

        load_and_quantize_hidden_states(inps, first_block_idx, buffers, dims, configs, prj_cfg)

        if num_static_blocks % num_shards == 0:
            for per_shard_block_idx in range(n_blocks_per_shard - 1):
                block_idx = per_shard_block_idx + n_blocks_per_shard * shard_id
                compute_one_block(block_idx, block_idx+1, buffers, dims, inps, outs, configs, prj_cfg, shard_id)
            last_block_idx = n_blocks_per_shard * shard_id + n_blocks_per_shard - 1
            compute_one_block(last_block_idx, None, buffers, dims, inps, outs, configs, prj_cfg, shard_id)
        else:
            for per_shard_block_idx in range(n_blocks_per_shard):
                block_idx = per_shard_block_idx + n_blocks_per_shard * shard_id
                compute_one_block(block_idx, block_idx+1, buffers, dims, inps, outs, configs, prj_cfg, shard_id)
            is_dummy = (shard_id == 0)
            remainder_block_idx = num_static_blocks - 1
            compute_one_block(remainder_block_idx, None, buffers, dims, inps, outs, configs, prj_cfg, shard_id, is_dummy)
    """
    n_blocks_per_shard, r_block = divmod(num_static_blocks, dims.num_shards)
    # prefetch the first block of each core
    first_block_idx = n_blocks_per_shard * dims.shard_id

    # Allocate zeros on heap (on top of hidden bufs) for output init, then free
    if is_tensor_update_accumulating:
        H = dims.H
        zeros = sbm.alloc_heap((_pmax, H), dtype=nl.bfloat16, name="output_init_zeros", align=SBUF_QUADRANT_SIZE)
        nisa.memset(zeros, value=0.0)
        output_initialization(outs.output, dims, sbm=sbm, zeros=zeros)
        sbm.pop_heap()  # free zeros, hidden bufs remain

    # fp8 pre-quantized hidden gathers + transposes in-block (no bf16 prefetch buffers, no
    # init quantize). Only the persistent hidden_qtz_sb/hidden_scale_sb view shape is set up.
    if not configs.is_fp8_hidden:
        # Heap-allocate hidden state buffers for first block load
        _alloc_hidden_bufs(sbm, buffers, dims, prj_cfg, configs, tag=f"sb{first_block_idx}_")

    if configs.is_fp8_hidden:
        # fp8: persistent hidden buffers stay in the [_pmax, n_H512_tile, B] matmul layout.
        # Prefetch (gather) the first block here; compute_one_block transposes it and prefetches
        # the next block, so the gather DMA always overlaps the prior block's compute.
        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))
        # fp8 scales are packed: n_packed scale blocks (4 H512 tiles each), not n_H512_tile.
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, div_ceil(prj_cfg.n_H512_tile, 4), dims.B))
        # First block: gather + transpose now so hidden_qtz_sb is ready before the loop. Each
        # compute_one_block then gathers the next block (during gate/up) and transposes it (deferred
        # after gate/up), so steady-state gather/transpose always overlap compute.
        _fb_tok = load_token_indices(inps.token_position_to_id, first_block_idx, dims.B, dims.n_B128_tiles, sbm=sbm)
        load_fp8_hidden_states_mx(
            inps,
            dims,
            configs.skip_dma,
            token_indices_on_p=_fb_tok,
            block_hidden_concat=buffers.block_hidden_concat,
        )
        transpose_fp8_hidden_states(
            dims,
            prj_cfg,
            buffers.block_hidden_concat,
            hidden_qtz_sb=buffers.hidden_qtz_sb,
            hidden_scale_sb=buffers.hidden_scale_sb,
            sbm=sbm,
        )
    else:
        if USE_DMA_TRANSPOSE:
            sbm.open_scope(name="init_hiv")
            compute_hidden_index_vector(inps, buffers, first_block_idx, dims, configs.skip_dma, False, sbm=sbm)
            sbm.close_scope()
            load_hidden_states_mx(
                inps,
                dims,
                configs.skip_dma,
                token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                block_hidden_states_T=buffers.block_hidden_states_T,
                use_dma_transpose=True,
                sbm=sbm,
            )
        else:
            sbm.open_scope(name="init_hiv")
            compute_hidden_index_vector(inps, buffers, first_block_idx, dims, configs.skip_dma, False, sbm=sbm)
            sbm.close_scope()
            load_hidden_states_mx(
                inps,
                dims,
                configs.skip_dma,
                token_4_H_indices_on_p=buffers.token_4_H_indices_on_p,
                block_hidden_states=buffers.block_hidden_states,
                use_dma_transpose=False,
                sbm=sbm,
            )
            sbuf_layout_adapter(buffers.block_hidden_states, buffers.block_hidden_states_T, dims, sbm=sbm)

        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        if not configs.is_static_quant:
            buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B // 32, 32))
        # NOTE: we do not quantize here because we will do it in the beginning of each static block

    # 2 different code paths to handle N odd and N even to explicitly handle prefetching
    if num_static_blocks % dims.num_shards == 0:
        """
        N is even
        In this case, in each core we can only do prefetch in the first n_blocks_per_shard - 1 blocks
        """
        kernel_assert(r_block == 0, "Expected r_block to be 0 for even number of static blocks")
        for per_shard_block_idx in nl.sequential_range(n_blocks_per_shard - 1):
            block_idx = per_shard_block_idx + n_blocks_per_shard * dims.shard_id

            _block_expert_for_weights = None
            _next_block_expert_for_weights = None
            if all_experts_for_weights != None:
                _block_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=_block_expert_for_weights,
                    src=all_experts_for_weights[0:1, per_shard_block_idx : per_shard_block_idx + 1],
                )
                _next_block_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=_next_block_expert_for_weights,
                    src=all_experts_for_weights[0:1, per_shard_block_idx + 1 : per_shard_block_idx + 2],
                )

            compute_one_block(
                block_idx,
                block_idx + 1,
                buffers,
                dims,
                inps,
                outs,
                kernel_cfg=configs,
                prj_cfg=prj_cfg,
                shard_id=dims.shard_id,
                is_first_block=(per_shard_block_idx == 0),
                sbm=sbm,
                block_expert_for_weights=_block_expert_for_weights,
                next_block_expert_for_weights=_next_block_expert_for_weights,
                hoisted_gup_weights=hoisted_gup_weights,
                hoisted_gup_bias=hoisted_gup_bias,
                hoisted_down_bias=hoisted_down_bias,
            )

        last_block_idx = n_blocks_per_shard * dims.shard_id + n_blocks_per_shard - 1

        _last_expert_for_weights = None
        if all_experts_for_weights != None:
            _last_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=_last_expert_for_weights,
                src=all_experts_for_weights[0:1, n_blocks_per_shard - 1 : n_blocks_per_shard],
            )

        compute_one_block(
            last_block_idx,
            None,
            buffers,
            dims,
            inps,
            outs,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            shard_id=dims.shard_id,
            is_first_block=(n_blocks_per_shard == 1),
            sbm=sbm,
            block_expert_for_weights=_last_expert_for_weights,
            hoisted_gup_weights=hoisted_gup_weights,
            hoisted_gup_bias=hoisted_gup_bias,
            hoisted_down_bias=hoisted_down_bias,
        )

    else:
        """
        N is odd
        In this case, each core can do prefetch the first n_blocks_per_shard
        """
        kernel_assert(r_block == 1, "Expected r_block to be 1 for odd number of static blocks")
        for per_shard_block_idx in nl.sequential_range(n_blocks_per_shard):
            block_idx = per_shard_block_idx + n_blocks_per_shard * dims.shard_id

            _block_expert_for_weights = None
            _next_block_expert_for_weights = None
            if all_experts_for_weights != None:
                _block_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=_block_expert_for_weights,
                    src=all_experts_for_weights[0:1, per_shard_block_idx : per_shard_block_idx + 1],
                )
                # Next-block lookup is also from the precomputed table.
                # all_experts_for_weights is sized n_blocks_per_shard_alloc which
                # overallocates by 1, so [per_shard_block_idx + 1] is in range and
                # holds E for the OOB tail (memset upfront), matching the prefetch
                # OOB-skip semantics.
                _next_block_expert_for_weights = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=_next_block_expert_for_weights,
                    src=all_experts_for_weights[0:1, per_shard_block_idx + 1 : per_shard_block_idx + 2],
                )

            compute_one_block(
                block_idx,
                block_idx + 1,
                buffers,
                dims,
                inps,
                outs,
                kernel_cfg=configs,
                prj_cfg=prj_cfg,
                shard_id=dims.shard_id,
                is_first_block=(per_shard_block_idx == 0),
                sbm=sbm,
                block_expert_for_weights=_block_expert_for_weights,
                next_block_expert_for_weights=_next_block_expert_for_weights,
                hoisted_gup_weights=hoisted_gup_weights,
                hoisted_gup_bias=hoisted_gup_bias,
                hoisted_down_bias=hoisted_down_bias,
            )

        """
        Handle last remaining block when N is odd.

        Core 1 should have the data for this block prefetched in the previous loop.
        Core 0 processes a dummy block, signaled by memsetting expert affinity to 0.
        """
        is_dummy = dims.shard_id == 0
        remainder_block_idx = num_static_blocks - 1
        compute_one_block(
            remainder_block_idx,
            None,
            buffers,
            dims,
            inps,
            outs,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            shard_id=dims.shard_id,
            is_dummy=is_dummy,
            # When n_blocks_per_shard == 0, the for loop didn't run, so no prefetch
            # has filled gup_tile_buf_a. Force tile 0 to be loaded explicitly by
            # marking this as the first block.
            is_first_block=(n_blocks_per_shard == 0),
            sbm=sbm,
            hoisted_gup_weights=hoisted_gup_weights,
            hoisted_gup_bias=hoisted_gup_bias,
            hoisted_down_bias=hoisted_down_bias,
        )


def process_dynamic_blocks(
    dims: BWMMMXDimensionSizes,
    configs: BWMMMXConfigs,
    prj_cfg: ProjConfig,
    inps: InputTensors,
    outs: OutputTensors,
    buffers: SharedBuffers,
    num_static_blocks: int,
    num_dynamic_blocks: int,
    sbm=None,
    hoisted_gup_weights=None,
    hoisted_gup_bias=None,
    hoisted_down_bias=None,
    outer_reg=None,
    rem_reg=None,
    n_outer_iters_sb=None,
    chunked_experts_mask=None,
    rem_experts_mask=None,
    dyn_weight_expert_sb=None,
    chunked_shard_iter_sb=None,
    rem_shard_iter_sb=None,
    n_chunked_alloc: int = 0,
    n_rem_alloc: int = 0,
):
    """
    Process dynamic (potentially padded) blocks using condition vector with
    a chunked outer loop + ping-pong remainder loop.

    Structure:
      - Outer loop (step=8): each iter processes 4 contiguous blocks per core.
          Shard 0 takes index+0..+3, shard 1 takes index+4..+7.
          index += 8 per outer iter.
      - Remainder loop (step=2, existing ping-pong): handles 0..3 leftover
          iters per shard (0..6 remaining active blocks total after chunking).

    Iteration counts (computed at caller, passed via outer_reg / rem_reg):
        n_active      = sum(conditions[n_static_blocks : n_static_blocks + num_dynamic_blocks])
        n_outer_iters = n_active // _DYN_STEP
        n_rem_iters   = ((n_active + 1) >> 1) - n_outer_iters * _DYN_INNER

    Args:
        outer_reg: register holding n_outer_iters (drives chunked outer loop)
        rem_reg:   register holding n_rem_iters (drives ping-pong remainder loop)
        n_outer_iters_sb: SBUF scalar holding n_outer_iters (used to compute
            runtime first-block offset for shard 1: shard_id*4 if outer runs, else shard_id)
        chunked_experts_mask: weight-skip mask for chunked path (shard-ordered)
        rem_experts_mask: weight-skip mask for remainder path (shard-ordered)
        dyn_weight_expert_sb: per-block scratch for current weight-expert lookup.
        chunked_shard_iter_sb / rem_shard_iter_sb: per-shard mask iter counters.
            Next-block expert is read with the same counter and
            `offset=1` on the .ap() pattern.
        n_chunked_alloc / n_rem_alloc: compile-time mask lengths
    """
    kernel_assert(
        num_static_blocks + num_dynamic_blocks == dims.N,
        f"num_static_blocks + num_dynamic_blocks must equal N, got {num_static_blocks} + {num_dynamic_blocks}!= {dims.N} ",
    )

    logger.info(f"Start looping over dynamic blocks {num_static_blocks} to {dims.cond_vec_len} - 1")

    # Compile-time flags: does the outer / remainder loop possibly run at runtime?
    _outer_may_run = num_dynamic_blocks >= _DYN_STEP

    INNER = _DYN_INNER  # blocks per core per outer iter (module constant)
    STEP = _DYN_STEP  # num_shards * INNER; asserted num_shards == 2 elsewhere

    # --- First-block prefetch -------------------------------------------------
    # Shard s's first real block depends on whether outer runs at runtime:
    #   outer runs (n_outer_iters > 0): first = num_static_blocks + s * INNER
    #   outer skipped:                  first = num_static_blocks + s
    # Compute first_block_idx at runtime:
    #   mult = 1 + (INNER - 1) * min(n_outer_iters, 1)   (= INNER if outer>0, else 1)
    #   first_block_off = shard_id * mult
    #   first_block_idx = num_static_blocks + first_block_off
    logger.info("Prefetch first block for each core")
    first_block_idx_sb = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name="first_block_idx", align=SBUF_QUADRANT_SIZE)
    if dims.shard_id == 0:
        # Shard 0's first block is always num_static_blocks (both schemes agree).
        nisa.memset(dst=first_block_idx_sb, value=num_static_blocks)
    else:
        # Shard 1: mult = 1 + (INNER - 1) * min(n_outer_iters, 1)
        outer_gt0_sb = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name="outer_gt0", align=SBUF_QUADRANT_SIZE)
        nisa.tensor_scalar(dst=outer_gt0_sb, data=n_outer_iters_sb, op0=nl.minimum, operand0=1)
        # first = num_static_blocks + shard_id * (1 + (INNER-1) * outer_gt0)
        # Fuse: first = outer_gt0 * ((INNER-1) * shard_id) + (num_static_blocks + shard_id)
        nisa.tensor_scalar(
            dst=first_block_idx_sb,
            data=outer_gt0_sb,
            op0=nl.multiply,
            operand0=_DYN_INNER_M1 * dims.shard_id,
            op1=nl.add,
            operand1=num_static_blocks + dims.shard_id,
        )

    if configs.is_fp8_hidden:
        # fp8: persistent hidden buffers stay in the [_pmax, n_H512_tile, B] matmul layout.
        # Prefetch (gather) the first dynamic block; compute_one_block transposes it and prefetches
        # the next block so the gather DMA overlaps the prior block's compute.
        buffers.hidden_qtz_sb = buffers.hidden_qtz_sb.reshape((_pmax, prj_cfg.n_H512_tile, dims.B))
        # fp8 scales are packed: n_packed scale blocks (4 H512 tiles each), not n_H512_tile.
        buffers.hidden_scale_sb = buffers.hidden_scale_sb.reshape((_pmax, div_ceil(prj_cfg.n_H512_tile, 4), dims.B))
        # First block: gather + transpose now so hidden_qtz_sb is ready before the loop. Each
        # compute_one_block then gathers the next block (during gate/up) and transposes it (deferred).
        _prev_pfx = sbm.get_name_prefix() if sbm != None else None
        if sbm != None:
            sbm.set_name_prefix(f"{_prev_pfx}fp8pf_dyninit_")
        _fb_tok = load_token_indices_dynamic_block(
            inps.token_position_to_id, first_block_idx_sb, dims.B, dims.n_B128_tiles, skip_dma=configs.skip_dma, sbm=sbm
        )
        load_fp8_hidden_states_mx(
            inps,
            dims,
            configs.skip_dma,
            token_indices_on_p=_fb_tok,
            block_hidden_concat=buffers.block_hidden_concat,
        )
        if sbm != None:
            sbm.set_name_prefix(_prev_pfx)
        transpose_fp8_hidden_states(
            dims,
            prj_cfg,
            buffers.block_hidden_concat,
            hidden_qtz_sb=buffers.hidden_qtz_sb,
            hidden_scale_sb=buffers.hidden_scale_sb,
            sbm=sbm,
        )
    else:
        # Heap-allocate hidden state buffers for first dynamic block load
        _alloc_hidden_bufs(sbm, buffers, dims, prj_cfg, configs, tag="dyn_init_")

        if sbm != None:
            sbm.open_scope(name="dyn_block_load_hidden_quant")
        # STATIC_MX: gather first block's per-expert recip for the software quant.
        _dyn_init_in_quant_recip = None
        if configs.is_static_quant:
            _dyn_init_expert = load_block_expert(
                inps.block_to_expert, first_block_idx_sb, sbm=sbm, name="dyn_init_expert"
            )
            _dyn_init_in_quant_recip = _gather_gup_in_quant_recip(
                inps, _dyn_init_expert, dims, sbm, name_prefix="dyn_init_"
            )
        load_and_quantize_hidden_states(
            inps,
            first_block_idx_sb,
            buffers,
            dims,
            configs,
            prj_cfg,
            is_block_idx_dynamic=True,
            use_dma_transpose=USE_DMA_TRANSPOSE,
            sbm=sbm,
            in_quant_recip=_dyn_init_in_quant_recip,
        )
        if sbm != None:
            sbm.close_scope()

    # Prefetch first-dynamic-block's gup weights into persistent buffer.
    # Under full-resident scheme: load the entire gup. Otherwise: just tile 0.
    if buffers.gup_tile_buf_a != None:
        sbm.open_scope(name="pf_dyn_init")
        _pf_expert = load_block_expert(inps.block_to_expert, first_block_idx_sb, sbm=sbm)
        _prefetch_gup_tile0(
            inps,
            _pf_expert,
            prj_cfg,
            buffers,
            dims,
            configs.skip_dma,
            sbm,
            name_prefix="pf_dyn_init",
            use_packed_scales=configs.use_packed_scales,
            skip_scales=configs.is_static_quant,
            full_I_load=configs.gup_full_persistent,
        )
        sbm.close_scope()

    # For non-tiled dynamic path, allocate persistent gup weight/bias buffers and prefetch
    # the first block's weights so the load can be skipped inside compute_one_block.
    _nontiled_gup_prefetch = dims.I <= _I_TILE_SZ
    if _nontiled_gup_prefetch:
        if hoisted_gup_weights == None:
            hoisted_gup_weights = _sbm_alloc(
                sbm,
                (_pmax, 2, prj_cfg.n_H512_tile_sharded, dims.I),
                dtype=inps.gate_up_proj_weight.dtype,
                name="dyn_hoisted_gup_weights",
                align=SBUF_QUADRANT_SIZE,
            )
        if hoisted_gup_bias == None and inps.gate_and_up_proj_bias:
            hoisted_gup_bias = _sbm_alloc(
                sbm,
                (_pmax, 2, prj_cfg.n_total_I512_tile, _q_width),
                dtype=inps.gate_and_up_proj_bias.dtype,
                name="dyn_hoisted_gup_bias",
                align=SBUF_QUADRANT_SIZE,
            )
            if dims.I < _pmax * _q_width:
                nisa.memset(dst=hoisted_gup_bias[:, :, 0, :], value=0.0)
        # Prefetch first dynamic block's full gup weights/scales/bias
        sbm.open_scope(name="pf_dyn_gup_init")
        _pf_expert = load_block_expert(inps.block_to_expert, first_block_idx_sb, sbm=sbm)
        load_gup_weights_scales_mx(
            inps,
            _pf_expert,
            dims,
            prj_cfg=prj_cfg,
            skip_dma=configs.skip_dma,
            sbm=sbm,
            dst_weight=hoisted_gup_weights,
            dst_bias=hoisted_gup_bias,
            use_packed_scales=configs.use_packed_scales,
            skip_scales=configs.is_static_quant,
        )
        sbm.close_scope()

    # Persistent buffer for delayed output scatter write. process_static loops doesn't do the output write delayed. First dynamic iteration block
    # will do a no-op skipped output write.
    pending_token_indices = _sbm_alloc(
        sbm, (_pmax, dims.n_B128_tiles), dtype=nl.int32, name="pending_token_indices", align=SBUF_QUADRANT_SIZE
    )
    nisa.memset(dst=pending_token_indices, value=-1)

    # Ensure block_old has a defined value before the first scatter-flush read.
    if not configs.skip_dma.skip_token and num_static_blocks // dims.num_shards < 2:
        nisa.memset(dst=buffers.block_old, value=0)

    # Weight-skip flags (compile-time, driven by presence of precomputed masks)
    _use_chunked_skip = chunked_experts_mask != None and n_chunked_alloc > 0
    _use_rem_skip = rem_experts_mask != None and n_rem_alloc > 0

    # Block-idx scratch shared by chunked and rem loops; compute_one_block
    # gets a unique scope prefix from the name_tag passed by each caller.
    dyn_block_idx_sb = _sbm_alloc(
        sbm,
        (1, 1),
        dtype=nl.int32,
        name="dyn_block_idx",
        align=SBUF_QUADRANT_SIZE,
    )
    dyn_next_block_idx_sb = _sbm_alloc(
        sbm,
        (1, 1),
        dtype=nl.int32,
        name="dyn_next_block_idx",
        align=SBUF_QUADRANT_SIZE,
    )

    # --- Chunked outer loop (step=8) ----------------------------------------
    if _outer_may_run:
        """
        Countdown tracker for detecting the final outer iter at runtime.
        On the LAST outer iter, inner=INNER-1's prefetch target needs to switch
        from "next-chunk's shard block" to "first remainder block for this shard"
        so that the prefetched tile 0 matches the block actually processed next.
        For shard 0 these are the same block (both = index + STEP), so no fix
        is needed — the adjustment only matters for shard 1.
        """
        outer_countdown_sb = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name="outer_countdown", align=SBUF_QUADRANT_SIZE)
        nisa.tensor_copy(dst=outer_countdown_sb, src=n_outer_iters_sb)
        # Shard 1 only: scratch SBUFs for is_last and correction computations.
        if dims.shard_id == 1:
            is_last_outer_sb = _sbm_alloc(sbm, (1, 1), dtype=nl.int32, name="is_last_outer", align=SBUF_QUADRANT_SIZE)

        # Boundary scratch for chunked->rem skip-aware expert compute.
        # boundary_skip_oob_sb holds the constant E (set once); the others are
        # overwritten each boundary iter.
        boundary_skip_oob_sb = None
        boundary_skip_expert_sb = None
        boundary_is_same_sb = None
        if _use_chunked_skip:
            boundary_skip_oob_sb = _sbm_alloc(
                sbm, (1, 1), dtype=nl.int32, name="boundary_skip_oob", align=SBUF_QUADRANT_SIZE
            )
            nisa.memset(dst=boundary_skip_oob_sb, value=dims.E)
            boundary_skip_expert_sb = _sbm_alloc(
                sbm, (1, 1), dtype=nl.int32, name="boundary_skip_expert", align=SBUF_QUADRANT_SIZE
            )
            boundary_is_same_sb = _sbm_alloc(
                sbm, (1, 1), dtype=nl.uint8, name="boundary_is_same", align=SBUF_QUADRANT_SIZE
            )

        def _process_outer_block(_outer):
            for inner in nl.sequential_range(INNER):
                _bi_sb = dyn_block_idx_sb
                _nbi_sb = dyn_next_block_idx_sb

                # block_idx = index + shard_id * INNER + inner
                nisa.tensor_scalar(
                    dst=_bi_sb,
                    data=buffers.index,
                    op0=nl.add,
                    operand0=dims.shard_id * INNER + inner,
                )

                """
                next_block_idx:
                inner 0..INNER-2: block_idx + 1 (within-chunk)
                inner == INNER-1: target the block each shard processes NEXT.
                - Non-last outer iter: next chunk's shard block
                    = index + STEP + shard_id * INNER
                - Last outer iter:     first remainder block for this shard
                    = index + STEP + shard_id
                Shard 0: both formulas equal index + STEP, so no runtime
                adjustment is needed.
                Shard 1: on last iter subtract (INNER - 1) from the base.
                Clamped to N-1 for safety.          
                """
                if inner < INNER - 1:
                    nisa.tensor_scalar(
                        dst=_nbi_sb,
                        data=_bi_sb,
                        op0=nl.add,
                        operand0=1,
                    )
                elif dims.shard_id == 0:
                    nisa.tensor_scalar(
                        dst=_nbi_sb,
                        data=buffers.index,
                        op0=nl.add,
                        operand0=STEP + 0,
                        op1=nl.minimum,
                        operand1=dims.N - 1,
                    )
                else:
                    # shard_id == 1: adjust on last outer iter.
                    # is_last = (countdown == 1)  (1 on last outer iter, 0 otherwise)
                    nisa.tensor_scalar(
                        dst=is_last_outer_sb,
                        data=outer_countdown_sb,
                        op0=nl.equal,
                        operand0=1,
                    )
                    # shard-1 offset term collapses via is_last:
                    #   is_last * -(INNER-1) + INNER = INNER on non-last, 1 on last.
                    # Then + (index + STEP) gives the right block without branching.
                    nisa.tensor_scalar(
                        dst=_nbi_sb,
                        data=is_last_outer_sb,
                        op0=nl.multiply,
                        operand0=-(INNER - 1),
                        op1=nl.add,
                        operand1=STEP + INNER,
                    )
                    nisa.tensor_tensor(
                        dst=_nbi_sb,
                        data1=_nbi_sb,
                        data2=buffers.index,
                        op=nl.add,
                    )

                    nisa.tensor_scalar(
                        dst=_nbi_sb,
                        data=_nbi_sb,
                        op0=nl.minimum,
                        operand0=dims.N - 1,
                    )

                _dyn_expert_for_weights = None
                _dyn_next_expert_for_weights = None
                if _use_chunked_skip:
                    # chunked_shard_iter advances by 1 per inner block.
                    nisa.tensor_copy(
                        dst=dyn_weight_expert_sb,
                        src=chunked_experts_mask.ap(
                            pattern=[[n_chunked_alloc, 1], [1, 1]],
                            offset=0,
                            scalar_offset=chunked_shard_iter_sb,
                            indirect_dim=1,
                        ),
                    )
                    _dyn_expert_for_weights = dyn_weight_expert_sb
                    if inner < INNER - 1:
                        # Precompute the prefetch's skip-aware expert via
                        # indirect tensor_copy from chunked_experts_mask at
                        # chunked_shard_iter + 1 (compile-time offset).
                        _dyn_next_expert_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
                        nisa.tensor_copy(
                            dst=_dyn_next_expert_sb,
                            src=chunked_experts_mask.ap(
                                pattern=[[n_chunked_alloc, 1], [1, 1]],
                                offset=1,
                                scalar_offset=chunked_shard_iter_sb,
                                indirect_dim=1,
                            ),
                        )
                        _dyn_next_expert_for_weights = _dyn_next_expert_sb
                    else:
                        # Chunked->rem boundary: next block is runtime-variable,
                        # derive skip-aware expert from raw block_to_expert reads.
                        # Load next-expert directly into boundary_skip_expert_sb
                        boundary_curr_expert_sb = load_block_expert(
                            inps.block_to_expert, _bi_sb, sbm=sbm, name="boundary_curr_expert"
                        )
                        nisa.dma_copy(
                            dst=boundary_skip_expert_sb[0, 0],
                            src=inps.block_to_expert.ap(
                                pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=_nbi_sb, indirect_dim=0
                            ),
                        )
                        nisa.tensor_tensor(
                            data1=boundary_skip_expert_sb,
                            data2=boundary_curr_expert_sb,
                            op=nl.equal,
                            dst=boundary_is_same_sb,
                        )
                        nisa.tensor_copy_predicated(
                            dst=boundary_skip_expert_sb,
                            src=boundary_skip_oob_sb,
                            predicate=boundary_is_same_sb,
                        )
                        _dyn_next_expert_for_weights = boundary_skip_expert_sb

                compute_one_block(
                    _bi_sb,
                    _nbi_sb,
                    buffers,
                    dims,
                    inps,
                    outs,
                    kernel_cfg=configs,
                    prj_cfg=prj_cfg,
                    shard_id=dims.shard_id,
                    is_dynamic=True,
                    is_first_block=False,
                    sbm=sbm,
                    block_expert_for_weights=_dyn_expert_for_weights,
                    next_block_expert_for_weights=_dyn_next_expert_for_weights,
                    hoisted_gup_weights=hoisted_gup_weights,
                    hoisted_gup_bias=hoisted_gup_bias,
                    hoisted_down_bias=hoisted_down_bias,
                    delay_output_write=True,
                    pending_token_indices=pending_token_indices,
                    name_tag=f"chunk_{inner}",
                )

                if _use_chunked_skip:
                    nisa.tensor_scalar(
                        dst=chunked_shard_iter_sb,
                        data=chunked_shard_iter_sb,
                        op0=nl.add,
                        operand0=1,
                    )

            # End of outer iter: advance index by STEP (8)
            nisa.tensor_scalar(dst=buffers.index, data=buffers.index, op0=nl.add, operand0=STEP)
            # Decrement countdown so inner=INNER-1 of the next iter can detect
            # "is this the last outer iter?" via countdown == 1.
            nisa.tensor_scalar(
                dst=outer_countdown_sb,
                data=outer_countdown_sb,
                op0=nl.subtract,
                operand0=1,
            )

        nl.fori_loop(0, outer_reg, _process_outer_block)

        # Outer -> Remainder transition:
        # At inner=INNER-1 of the last outer iter, we now prefetch the correct
        # block for each shard (index + STEP + shard_id, matching remainder's
        # first block), so no corrective transition prefetch is needed here.

    def _process_remainder_block(_rem):
        # block_idx = index + shard_id
        nisa.tensor_scalar(
            dst=dyn_block_idx_sb,
            data=buffers.index,
            op0=nl.add,
            operand0=dims.shard_id,
        )
        # next_block_idx = min(block_idx + num_shards, N - 1)
        nisa.tensor_scalar(
            dst=dyn_next_block_idx_sb,
            data=dyn_block_idx_sb,
            op0=nl.add,
            operand0=dims.num_shards,
            op1=nl.minimum,
            operand1=dims.N - 1,
        )

        _dyn_expert_for_weights = None
        _dyn_next_expert_for_weights = None
        if _use_rem_skip:
            # Mask is sized n_rem_alloc + 1 with the +1 sentinel slot at E,
            # so curr/next lookups use _n_rem_mask_len as the AP pattern bound.
            _n_rem_mask_len = n_rem_alloc + 1
            nisa.tensor_copy(
                dst=dyn_weight_expert_sb,
                src=rem_experts_mask.ap(
                    pattern=[[_n_rem_mask_len, 1], [1, 1]],
                    offset=0,
                    scalar_offset=rem_shard_iter_sb,
                    indirect_dim=1,
                ),
            )
            _dyn_expert_for_weights = dyn_weight_expert_sb
            # Precompute prefetch's skip-aware expert via rem_shard_iter + 1.
            # On the last rem iter, this walks into the sentinel slot (= E)
            # which is functionally a don't-care since the prefetch's output
            # is never consumed (no compute follows the last rem iter).
            _dyn_next_expert_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(
                dst=_dyn_next_expert_sb,
                src=rem_experts_mask.ap(
                    pattern=[[_n_rem_mask_len, 1], [1, 1]],
                    offset=1,
                    scalar_offset=rem_shard_iter_sb,
                    indirect_dim=1,
                ),
            )
            _dyn_next_expert_for_weights = _dyn_next_expert_sb

        compute_one_block(
            dyn_block_idx_sb,
            dyn_next_block_idx_sb,
            buffers,
            dims,
            inps,
            outs,
            kernel_cfg=configs,
            prj_cfg=prj_cfg,
            shard_id=dims.shard_id,
            is_dynamic=True,
            is_first_block=False,
            sbm=sbm,
            block_expert_for_weights=_dyn_expert_for_weights,
            next_block_expert_for_weights=_dyn_next_expert_for_weights,
            hoisted_gup_weights=hoisted_gup_weights,
            hoisted_gup_bias=hoisted_gup_bias,
            hoisted_down_bias=hoisted_down_bias,
            delay_output_write=True,
            pending_token_indices=pending_token_indices,
            name_tag="rem",
        )

        # Advance index by num_shards; advance rem_shard_iter by 1.
        nisa.tensor_scalar(dst=buffers.index, data=buffers.index, op0=nl.add, operand0=dims.num_shards)
        if _use_rem_skip:
            nisa.tensor_scalar(dst=rem_shard_iter_sb, data=rem_shard_iter_sb, op0=nl.add, operand0=1)

    nl.fori_loop(0, rem_reg, _process_remainder_block)

    # Flush the last dynamic block's pending output
    _write_output_scatter(buffers.block_old, pending_token_indices, outs, dims, dims.shard_id)
