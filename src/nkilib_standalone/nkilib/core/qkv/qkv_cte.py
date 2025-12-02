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
QKV_CTE kernel.

"""

# Standard Library
import math
from typing import List, Optional, Tuple, cast

# Neuron Kernel Interface
import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ..utils.allocator import Logger, SbufManager, sizeinbytes

# NKI Library
from ..utils.common_types import NormType, QKVOutputLayout

# QKV CTE
from .qkv_cte_utils import (
    QKV_CTE_Config,
    QKV_CTE_Dims,
    QKV_CTE_UserInput,
    _build_config,
    _get_tensor_dimensions,
    _validate_user_inputs,
)

# HARDWARE CONSTANTS
P_MAX = 128
F_MAX = 512
PSUM_BANK_SIZE = sizeinbytes(nl.float32) * F_MAX
NUM_HW_PSUM_BANKS = 8


def qkv_cte(
    input: nl.ndarray,
    fused_qkv_weights: nl.ndarray,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[nl.ndarray] = None,
    # -- Fused Residual Add
    fused_residual_add: Optional[bool] = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attention_prev: Optional[nl.ndarray] = None,
    # --- Fused Norm Related
    fused_norm_type: NormType = NormType.NO_NORM,
    gamma_norm_weights: Optional[nl.ndarray] = None,
    layer_norm_bias: Optional[nl.ndarray] = None,
    norm_eps: Optional[float] = 1e-6,
    hidden_actual: Optional[int] = None,
    # --- Fused RoPE Related
    fused_rope: Optional[bool] = False,
    cos_cache: Optional[nl.ndarray] = None,
    sin_cache: Optional[nl.ndarray] = None,
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    # -----------------------------------------
    store_output_in_sbuf: bool = False,
    # -----------------------------------------
    # User can optionally PASS Sbuf manager
    # -----------------------------------------
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    # ----------------------------------------
    load_input_with_DMA_transpose: bool = True,
) -> nl.ndarray:
    """
    QKV (Query, Key, Value) projection kernel with multiple (optional) fused operations.
    This kernel is optimized for large B x S, which commonly appear in prefill/context-encoding.
    Ideally, use this kernel when B x S >= 128.

    Performs matrix multiplication between hidden states (input) and fused QKV weights matrix,
    with optional fused operations including:
    - Residual addition (input + mlp_prev + attention_prev)
    - Layer normalization (LayerNorm) or RMS normalization
    - Bias addition to QKV projection output
    - RoPE (Rotary Position Embedding) rotation applied to Query and Key heads

    Core operation:
    1. Optional residual addition: input = input + mlp_prev + attention_prev
    2. Optional normalization: input = norm(input)
    3. QKV projection: qkv = input @ fused_qkv_weights + bias
    4. Optional RoPE: apply rotary position embedding to Q and K heads in qkv

    Formulas for fused operators:
    -----------------------------
    RMS Norm:
        RMSNorm(x) = x * gamma / sqrt(mean(x²) + eps)

    Layer Norm:
        LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
        where var(x) = mean((x - mean(x))²)

    Both normalizations operate along the hidden dimension for each sequence position.

    RoPE (Rotary Position Embedding):
        For each Query/Key head X = [X1, X2] (where X1, X2 are first/second half of head):
            RoPE(X) = [X1, X2] * cos_cache + [-X2, X1] * sin_cache


    Parameters:
    -----------
    input : nl.ndarray
        Input hidden states tensor of shape [B, S, H] where B=batch, S=sequence_length, H=hidden_dim.
        We name it 'input' and not 'hidden' to avoid ambiguity with the size of "hidden dimension".
    fused_qkv_weights : nl.ndarray
        Fused QKV weight matrix of shape [H, I] where I=fused_qkv_dim=(num_q_heads + 2*num_kv_heads)*d_head
    output_layout : QKVOutputLayout, default=QKVOutputLayout.BSD
        Output tensor layout: QKVOutputLayout.BSD=[B, S, I] or QKVOutputLayout.NBSd=[num_heads, B, S, d_head]
    bias : Optional[nl.ndarray], default=None
        Bias tensor of shape [1, I] to add to QKV projection output
    fused_residual_add : Optional[bool], default=False
        Whether to perform residual addition: input = input + mlp_prev + attention_prev
    mlp_prev : Optional[nl.ndarray], default=None
        Previous MLP output tensor of shape [B, S, H] for residual addition
    attention_prev : Optional[nl.ndarray], default=None
        Previous attention output tensor of shape [B, S, H] for residual addition
    fused_norm_type : NormType, default=NormType.NO_NORM
        Type of normalization: NormType.NO_NORM, NormType.RMS_NORM, NormType.RMS_NORM_SKIP_GAMMA, or NormType.LAYER_NORM
        NormType.RMS_NORM_SKIP_GAMMA assumes fused_qkv_weights have been pre-multiplied with gamma vector, so its skipped here.
    gamma_norm_weights : Optional[nl.ndarray], default=None
        Normalization gamma/scale weights of shape [1, H] (required for NormType.RMS_NORM and NormType.LAYER_NORM)
    layer_norm_bias : Optional[nl.ndarray], default=None
        Layer normalization beta/bias weights of shape [1, H] (only for NormType.LAYER_NORM)
        Using layer norm bias is optional.
    norm_eps : Optional[float], default=1e-6
        Epsilon value for numerical stability in normalization
    hidden_actual : Optional[int], default=None
        Actual hidden dimension for padded tensors (if H contains padding)
    fused_rope : Optional[bool], default=False
        Whether to apply RoPE rotation to Query and Key heads after QKV projection
    cos_cache : Optional[nl.ndarray], default=None
        Cosine cache for RoPE of shape [B, S, d_head] (required if fused_rope=True)
    sin_cache : Optional[nl.ndarray], default=None
        Sine cache for RoPE of shape [B, S, d_head] (required if fused_rope=True)
    d_head : Optional[int], default=None
        Dimension per attention head (required for QKVOutputLayout.NBSd and RoPE)
    num_q_heads : Optional[int], default=None
        Number of query heads (required for RoPE)
    num_kv_heads : Optional[int], default=None
        Number of key/value heads (required for RoPE)
    store_output_in_sbuf : bool, default=False
        Whether to store output in SBUF (currently unsupported, must be False)
    sbm : Optional[SbufManager], default=None
        Optional SBUF manager for memory allocation control, with pre-specified bounds for SBUF usage.
        If sbm is not provided, kernel will by default be allocated and use all of the available SBUF space.
    use_auto_allocation : bool, default=False
        Whether to use automatic SBUF allocation, by default kernel is manually allocated and it creates its own SbufManager.
        If 'sbm' is provided by user, user has the responsibility to set use_auto_allocation=True in the provided SbufManager.
    load_input_with_DMA_transpose : bool, default=True
        Whether to use DMA transpose optimization

    Returns:
    --------
    nl.ndarray
        QKV projection output tensor:
        - If output_layout=QKVOutputLayout.BSD: shape [B, S, I]
        - If output_layout=QKVOutputLayout.NBSd: shape [num_heads, B, S, d_head]

    Constraints:
    ------------
    Tensor Shape Requirements:
    - H must be ≤ 24576 and divisible by 128
    - I must be ≤ 4096
    - For QKVOutputLayout.NBSd output: d_head must be specified and equal to 128

    Dimension Consistency:
    - input.shape[2] must equal fused_qkv_weights.shape[0] (H dimension)
    - If heads are specified: (num_q_heads + 2*num_kv_heads) * d_head must equal I

    Fused Operation Requirements:
    - fused_residual_add=True requires both mlp_prev and attention_prev tensors
    - NormType.RMS_NORM/NormType.LAYER_NORM require gamma_norm_weights and norm_eps
    - fused_rope=True requires cos_cache, sin_cache, num_q_heads, and num_kv_heads

    Hardware Compatibility:
    - Loading input with dma transpose may be ignored internally if current implementation
      or hardware does not allow it.

    Supported Data Types:
    - bf16, fp16, fp32 (fp32 inputs are internally converted to bf16 for computation)
    """

    # Build object of user inputs.
    user_inputs = QKV_CTE_UserInput(
        input=input,
        fused_qkv_weights=fused_qkv_weights,
        output_layout=output_layout,
        bias=bias,
        fused_residual_add=fused_residual_add,
        mlp_prev=mlp_prev,
        attention_prev=attention_prev,
        fused_norm_type=fused_norm_type,
        gamma_norm_weights=gamma_norm_weights,
        layer_norm_bias=layer_norm_bias,
        norm_eps=norm_eps,
        hidden_actual=hidden_actual,
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        store_output_in_sbuf=store_output_in_sbuf,
        sbm=sbm,
        use_auto_allocation=use_auto_allocation,
        load_input_with_DMA_transpose=load_input_with_DMA_transpose,
    )

    _validate_user_inputs(args=user_inputs)
    # Build 'cfg' object, to store kernel configuration.
    cfg = _build_config(args=user_inputs)
    # Build 'dims' object to store tensor dimensions used throughout the kernel.
    dims = _get_tensor_dimensions(args=user_inputs, cfg=cfg)

    # Create output tensor with original dimensions
    if cfg.output_layout == QKVOutputLayout.BSD:
        output_hbm = nl.ndarray((dims.B_orig, dims.S_orig, dims.I), dtype=input.dtype, buffer=nl.shared_hbm)
    else:  # QKVOutputLayout.NBSd
        output_hbm = nl.ndarray(
            (dims.num_heads, dims.B_orig, dims.S_orig, dims.d_head),
            dtype=input.dtype,
            buffer=nl.shared_hbm,
        )

    # Potentially reshape B,S to BxS, for performance benefits.
    if cfg.use_BxS_input_reshape:
        input = input.reshape((1, dims.BxS, dims.H))
        if cfg.fused_residual_add:
            mlp_prev = mlp_prev.reshape((1, dims.BxS, dims.H))
            attention_prev = attention_prev.reshape((1, dims.BxS, dims.H))
        if cfg.fused_rope:
            cos_cache = cos_cache.reshape((1, dims.BxS, dims.d_head))
            sin_cache = sin_cache.reshape((1, dims.BxS, dims.d_head))
        if cfg.output_layout == QKVOutputLayout.BSD:
            output_hbm = output_hbm.reshape((1, dims.BxS, dims.I))
        elif cfg.output_layout == QKVOutputLayout.NBSd:
            output_hbm = output_hbm.reshape((dims.num_heads, 1, dims.BxS, dims.d_head))

    # Pass values and directly, and keep 'cfg' and 'dims'
    # object separate for clarity.
    _qkv_cte_impl(
        input_hbm=input,
        fused_qkv_weights_hbm=fused_qkv_weights,
        output_hbm=output_hbm,
        cfg=cfg,
        dims=dims,
        sbm=sbm,
        bias_hbm=bias,
        mlp_prev_hbm=mlp_prev,
        attention_prev_hbm=attention_prev,
        gamma_norm_weights_hbm=gamma_norm_weights,
        layer_norm_bias_hbm=layer_norm_bias,
        norm_eps=norm_eps,
        cos_cache_hbm=cos_cache,
        sin_cache_hbm=sin_cache,
    )

    # Revert BxS to B,S as it is required by the user provided output_layout.
    if cfg.use_BxS_input_reshape:
        input = input.reshape((dims.B_orig, dims.S_orig, dims.H))
        if cfg.fused_residual_add:
            mlp_prev = mlp_prev.reshape((dims.B_orig, dims.S_orig, dims.H))
            attention_prev = attention_prev.reshape((dims.B_orig, dims.S_orig, dims.H))
        if cfg.fused_rope:
            cos_cache = cos_cache.reshape((dims.B_orig, dims.S_orig, dims.d_head))
            sin_cache = sin_cache.reshape((dims.B_orig, dims.S_orig, dims.d_head))
        if cfg.output_layout == QKVOutputLayout.BSD:
            output_hbm = output_hbm.reshape((dims.B_orig, dims.S_orig, dims.I))
        elif cfg.output_layout == QKVOutputLayout.NBSd:
            output_hbm = output_hbm.reshape((dims.num_heads, dims.B_orig, dims.S_orig, dims.d_head))

    return output_hbm


def _qkv_cte_impl(
    input_hbm,
    fused_qkv_weights_hbm,
    output_hbm,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    bias_hbm: Optional[nl.ndarray] = None,
    # Fused Residual Add Related
    mlp_prev_hbm: Optional[nl.ndarray] = None,
    attention_prev_hbm: Optional[nl.ndarray] = None,
    # Fused Normalization Related
    gamma_norm_weights_hbm: Optional[nl.ndarray] = None,
    layer_norm_bias_hbm: Optional[nl.ndarray] = None,
    norm_eps: Optional[float] = 1e-6,
    # Fused RoPE Related
    cos_cache_hbm: Optional[nl.ndarray] = None,
    sin_cache_hbm: Optional[nl.ndarray] = None,
):
    """
    Core QKV CTE kernel implementation.
    """
    # Uncomment for debug.
    # cfg.print()
    # dims.print()

    # input_hbm shape         = [dims.B, dims.S, dims.H]
    # fused_qkv_weights shape = [dims.H, dims.I]
    # We apply QKV projecion only on dims.S_shard part of input_hbm (with dims.S_shard_offset)

    S_shard = dims.S_shard
    H = dims.H
    I = dims.I
    if dims.S_shard == 0:
        return output_hbm

    # If user provided SbufManager (with more restricted sb_lower_bound and sb_upper_bound), use that (likely at the expense of performance).
    # Otherwise, use most sbuf space available.
    if sbm == None:
        sbm_logger = Logger(name="logger")
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=cfg.total_available_sbuf_space_to_this_kernel,
            use_auto_alloc=cfg.use_auto_allocation,
            logger=sbm_logger,
        )
    sbm.open_scope()

    ######################### Global SBUF Allocations ######################################
    # Allocate zero_bias_sb, norm_eps_sb, bias_sb, gamma_norm_weights_sb, layer_norm_bias_sb

    zero_bias_sb = sbm.alloc_stack((P_MAX, 1), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_bias_sb, value=0)

    norm_eps_sb = sbm.alloc_stack((P_MAX, 1), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
    nisa.memset(dst=norm_eps_sb, value=norm_eps)

    if cfg.add_bias:
        # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    # Load gamma_norm_weights_hbm (1,H) to sbuf.
    # Mathematically, we (later) need to apply elementwise multiplication: (input) [S, H] * (gamma) [1, H] for each row.
    # Note: NormType.RMS_NORM_SKIP_GAMMA skips this step.
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        # We load gamma weights into SBUF as a 2-D tensor of shape [128, ceil(H / P_MAX) ].
        gamma_norm_weights_sb = _load_norm_weights(norm_weights_hbm=gamma_norm_weights_hbm, cfg=cfg, dims=dims, sbm=sbm)

        # gamma_norm_weights_sb is used for both RMS_NORM and LAYER_NORM.
        # But, LAYER_NORM may have an additional Beta bias term provided: layer_norm_bias_hbm
        if cfg.add_layer_norm_bias:
            # Shape: [128, ceil(H / P_MAX)]
            layer_norm_bias_sb = _load_norm_weights(norm_weights_hbm=layer_norm_bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    ######################## Choose Multi-Buffering Degree ###################################

    # Multi-Buffering S: Choose max multi-buffering degree for sequence length, without spilling SBUF and PSUM space.
    # WARNING: This function needs to be updated if any new tensors get added to the kernel.
    #          It assumes current tensor shapes, and its "look-ahead", e.g. pre-calculates SBUF space ahead of time.
    # Note: In auto-allocation mode, sbuf space calculations do not make sense, but they do not break the kernel correctness.
    s_multi_buffer_degree, projected_sbuf_taken_space_after_multi_buffer = _multi_buffering_degree_for_seqlen(
        cfg=cfg, dims=dims, sbm=sbm
    )

    # Block is PMAX * multi_buffer_degree, e.g.  process [128 * 4, H] elements of S at once.
    S_BLOCK_SIZE = s_multi_buffer_degree * min(dims.S_shard, P_MAX)
    num_blocks_per_S_shard = math.ceil(dims.S_shard / S_BLOCK_SIZE) if S_BLOCK_SIZE > 0 else 0

    ######################## Weight Prefetching: Enough Space Left ?  #########################

    use_weight_prefetch = _use_weight_prefetch(
        projected_sbuf_taken_space_after_multi_buffer, cfg=cfg, dims=dims, sbm=sbm
    )

    if use_weight_prefetch:
        # If rhs (weights) is small enough, weights can be pre-loaded before QKV projection.

        # NOTE: In both prefetched and non-pretched case, we append allocated weight tensor to the same "weights_sb" list, to keep later changes to the minimum
        # To keep indexing differences in QKV projection to the minimum between the two cases, keep the shape of the allocated tensor the same:
        # weights_allocated = (128, num_allocated_H_subtiles_per_weight_load, I), and
        # weights_sb[...] may multi-buffer/allocate multiple of above tensors

        # In the case of weight prefetching, set the following variables:
        num_weight_buffers = 1  # Since we allocate all weights at once, so there is no need for multiple-buffers. We still need to set it to 1 as the kernel uses this constant later.
        weight_load_block_size_per_H = (
            H  # We pre-load the entire weight matrix. In non-prefetched case, we can load e.g 1024 H blocks at a time.
        )
        num_weight_load_blocks_per_H = 1  # In pre-fetch case, H / weight_load_block_size_per_H = 1
        max_num_128_H_subtiles_per_weight_block = math.ceil(weight_load_block_size_per_H / 128)  # = math.ceil(H / 128).

        weights_sb = []
        weights_prefetched_sb = sbm.alloc_stack(
            (P_MAX, max_num_128_H_subtiles_per_weight_block, I),
            dtype=cfg.compute_mm_dtype,
            buffer=nl.sbuf,
        )

        for i_tile_H in range(max_num_128_H_subtiles_per_weight_block):
            h_tile_sz = min(P_MAX, H - (i_tile_H * P_MAX))
            dst_pattern = [
                [max_num_128_H_subtiles_per_weight_block * I, h_tile_sz],
                [1, I],
            ]
            src_pattern = [[I, h_tile_sz], [1, I]]
            nisa.dma_copy(
                dst=weights_prefetched_sb.ap(pattern=dst_pattern, offset=i_tile_H * I),
                src=fused_qkv_weights_hbm.ap(pattern=src_pattern, offset=i_tile_H * P_MAX * I),
                dge_mode=dge_mode.swdge,
            )
        weights_sb.append(weights_prefetched_sb)

    for i_batch in range(dims.B):
        for i_block_S in nl.affine_range(num_blocks_per_S_shard):
            sbm.open_scope()
            # Adjust for the last loop iteration.
            s_block_sz = min(S_BLOCK_SIZE, S_shard - S_BLOCK_SIZE * i_block_S)
            num_S_tiles_in_block = math.ceil(s_block_sz / P_MAX)

            #################### Start of Allocations for Multi-Buffered tensors ##############################
            input_sb = []
            for _ in range(num_S_tiles_in_block):
                align = 32 if cfg.load_input_with_DMA_transpose else 1  # DMA_transpose requires align=32.
                input_sb.append(
                    sbm.alloc_stack(
                        (P_MAX, H),
                        dtype=cfg.compute_mm_dtype,
                        buffer=nl.sbuf,
                        align=align,
                    )
                )

            output_sb = []
            for _ in range(num_S_tiles_in_block):
                output_sb.append(sbm.alloc_stack((P_MAX, I), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))

            if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
                # Allocate rms_norm tensor outside of tiled S buffer loop, as square_sum has NUM_128_TILES_PER_S_BUFFER in its shape.
                # Write RMS and RMS Reciprocal tensors here, in-place.
                square_sum_sb = []
                for _ in range(num_S_tiles_in_block):
                    square_sum_sb.append(sbm.alloc_stack((P_MAX, 1), dtype=cfg.act_dtype, buffer=nl.sbuf))

            elif cfg.fused_norm_type == NormType.LAYER_NORM:
                # Used for the result of layer_norm, stores mean and variance.
                NUM_AGGR_STATS = 2  # mean and variance.
                bn_aggr_result_sb = []
                for _ in range(num_S_tiles_in_block):
                    bn_aggr_result_sb.append(
                        sbm.alloc_stack((P_MAX, NUM_AGGR_STATS), dtype=cfg.act_dtype, buffer=nl.sbuf)
                    )

            if cfg.fused_rope:
                # For the input head X = [X1, X2] , RoPE does the following:
                # X = [X1, X2] * cos_cache + [-X2, X1] * sin_cache
                # X = [X1, X2] * cos_cache + [-X2*sin_cache_1, X1*sin_cache_2]
                # sin_cache_1 = sin_cache_2. Therefore, we can keep only half of the sin_cache.
                cos_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    cos_buffer_sb.append(
                        sbm.alloc_stack(
                            (P_MAX, dims.d_head),
                            dtype=cfg.compute_mm_dtype,
                            buffer=nl.sbuf,
                        )
                    )

                sin_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    sin_buffer_sb.append(
                        sbm.alloc_stack(
                            (P_MAX, dims.d_head // 2),
                            dtype=cfg.compute_mm_dtype,
                            buffer=nl.sbuf,
                        )
                    )

                rope_intermediate_buffer_sb = []
                for _ in range(num_S_tiles_in_block):
                    rope_intermediate_buffer_sb.append(
                        sbm.alloc_stack(
                            (P_MAX, dims.d_head * 2),
                            dtype=cfg.compute_mm_dtype,
                            buffer=nl.sbuf,
                        )
                    )
            #########################  End of Allocations for Multi-Buffered tensors ##################################

            # In this case, we will transpose the input buffer using PE array.
            if not cfg.load_input_with_DMA_transpose:
                for i_tile_S in range(num_S_tiles_in_block):
                    sbm.open_scope()
                    # i_tile_S is used to index input_sb, e.g., a single tile of input row [128, H].
                    ######################################################################################################
                    # Step 1: Load the row of input tensor: [P_MAX, H]
                    #         (optionally apply fused_residual_add with mlp_prev, and attention_prev).
                    #          If load_input_with_DMA_transpose is applicable, this step is skipped, and moved later.
                    ######################################################################################################
                    S_TILE_SIZE = P_MAX
                    s_tile_local_offset = (
                        i_block_S * S_BLOCK_SIZE + i_tile_S * S_TILE_SIZE
                    )  # s_tile offset within S_shard.
                    s_tile_sz = min(
                        P_MAX, S_shard - s_tile_local_offset
                    )  # tile_size adjusted for the last loop iteration.
                    if cfg.fused_residual_add:
                        # Load row of input, and apply fused residual add.
                        s_tile_global_offset = i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                        nisa.dma_compute(
                            dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            srcs=[
                                input_hbm.ap(
                                    pattern=[[H, s_tile_sz], [1, H]],
                                    offset=s_tile_global_offset,
                                ),
                                mlp_prev_hbm.ap(
                                    pattern=[[H, s_tile_sz], [1, H]],
                                    offset=s_tile_global_offset,
                                ),
                                attention_prev_hbm.ap(
                                    pattern=[[H, s_tile_sz], [1, H]],
                                    offset=s_tile_global_offset,
                                ),
                            ],
                            scales=[1.0, 1.0, 1.0],
                            reduce_op=nl.add,
                        )
                    else:  # Do a regular input load without the fused residual add.
                        s_tile_global_offset = i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                        nisa.dma_copy(
                            dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            src=input_hbm.ap(
                                pattern=[[H, s_tile_sz], [1, H]],
                                offset=s_tile_global_offset,
                            ),
                            dge_mode=dge_mode.swdge,
                        )

                    ######################################################################################################
                    # Step 2: Apply (partial) RMS_NORM / LAYER_NORM to the input row.
                    ######################################################################################################
                    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
                        # Multiply input_sb[i_tile_S] by 1 / RMS(x) =  1 / sqrt(eps + (1 / hidden_actual) * (x1^2 + x2^2 + ... + xn^2)).
                        # Gamma weights multiply is yet to be done.
                        _apply_rms_normalization(
                            input_sb[i_tile_S],
                            square_sum_sb[i_tile_S],
                            zero_bias_sb,
                            norm_eps_sb,
                            s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                        )

                    elif cfg.fused_norm_type == NormType.LAYER_NORM:
                        """
                        Compute LayerNorm statistics for row of input_sb and store it bn_aggr_result_tile.
                            mean = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 0:1]
                            rvar = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 1:2] #rvar(var + eps)
                        """
                        _compute_layer_norm_stats(
                            input_sb[i_tile_S],
                            bn_aggr_result_sb[i_tile_S],
                            norm_eps_sb,
                            s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                        )

                    #######################################################################################################
                    # Step 3: Transpose input_sb using PE array, and apply already calculated NORM to each tile if needed.
                    #######################################################################################################
                    # Transposed tiles will be written back to input_sb.

                    # This is PE transpose loop, multiplying input with identity tensor.
                    # Also finalize applying normalization to the input tensor inside the loop.
                    for i_tile_H in range(dims.num_512_tiles_per_H):
                        # In this 512 loop we need a mask for H, we require H to be divisible by 128, but not 512.
                        H_TILE_SIZE = F_MAX
                        h_tile_offset = i_tile_H * H_TILE_SIZE
                        h_tile_sz = min(H_TILE_SIZE, H - h_tile_offset)

                        # Note: In some cases, applying RMSNorm might give us better pipelining if tensor_scalar was placed here.
                        if cfg.fused_norm_type == NormType.LAYER_NORM:
                            # Compute (x - mean) * rvar  (multiply with Gamma and add Beta later).
                            mean = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 0:1]
                            rvar = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 1:2]  # rvar(var + eps)

                            nisa.tensor_scalar(
                                dst=input_sb[i_tile_S][0:s_tile_sz, nl.ds(h_tile_offset, h_tile_sz)],
                                data=input_sb[i_tile_S][0:s_tile_sz, nl.ds(h_tile_offset, h_tile_sz)],
                                op0=nl.subtract,
                                operand0=mean,
                                op1=nl.multiply,
                                operand1=rvar,
                            )

                        # Transpose each [S (128), H (512)] tile of input buffer -> PSUM.
                        input_transposed_psum = []
                        for bank_id in range(NUM_HW_PSUM_BANKS):
                            # Note: PSUM tensors do not have "use_auto_allocation" flag like SbufManager to ignore the allocation.
                            if cfg.use_auto_allocation:
                                input_transposed_psum.append(
                                    nl.ndarray(
                                        (P_MAX, F_MAX),
                                        dtype=cfg.psum_transpose_dtype,
                                        buffer=nl.psum,
                                    )
                                )
                            else:
                                input_transposed_psum.append(
                                    nl.ndarray(
                                        (P_MAX, F_MAX),
                                        dtype=cfg.psum_transpose_dtype,
                                        buffer=nl.psum,
                                        address=(0, bank_id * PSUM_BANK_SIZE),
                                    )
                                )

                        tp_psum_bank_idx = (i_tile_S * dims.num_512_tiles_per_H + i_tile_H) % NUM_HW_PSUM_BANKS

                        # (Transpose) nisa.nc_matmul(...) returns result of [128, 128] shape, we need FOUR of these for a single [128, 512] tile.
                        num_128_subtiles_per_H_tile = math.ceil(h_tile_sz / 128)  # At most FMAX/PMAX = 4
                        for j_subtile_H in nl.affine_range(num_128_subtiles_per_H_tile):
                            H_SUBTILE_SIZE = P_MAX
                            h_subtile_offset_src = h_tile_offset + j_subtile_H * H_SUBTILE_SIZE  # src SBUF
                            h_subtile_offset_dst = j_subtile_H * H_SUBTILE_SIZE  # dst PSUM
                            h_subtile_sz = min(H_SUBTILE_SIZE, H - h_subtile_offset_src)

                            nisa.nc_transpose(
                                data=input_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(h_subtile_offset_src, h_subtile_sz),
                                ],
                                dst=input_transposed_psum[tp_psum_bank_idx][
                                    0:h_subtile_sz,
                                    nl.ds(h_subtile_offset_dst, s_tile_sz),
                                ],
                            )

                        # Copy transposed [128 (H), 512 (S)] tile from PSUM -> SBUF, and apply gamma_weights.
                        if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
                            # Note: NormType.RMS_NORM_SKIP_GAMMA skips this step.

                            # -- INPUT_TRANSPOSED * GAMMA_WEIGHTS elementwise-multiply ---
                            # Multiply each PSUM sub-tile [128(H), 128 (S)] with [128 (H), 1] column of gamma (with offsets).
                            # Recall that gamma_weights_hbm (H) was broadcasted to [128, H // P_MAX].
                            for j_subtile_H in nl.affine_range(num_128_subtiles_per_H_tile):
                                h_subtile_offset_sbuf = h_tile_offset + j_subtile_H * P_MAX
                                h_subtile_sz = min(P_MAX, H - h_subtile_offset_sbuf)

                                s_tile_offset_psum = j_subtile_H * P_MAX  # src
                                s_tile_offset_sbuf = h_subtile_offset_sbuf  # dst

                                # Index the right column of gamma_norm_weights_sb.
                                # i_tile_H is current 512th tile of H, we need current 128th tile of H for gamma column index.
                                gamma_tile_index = i_tile_H * (F_MAX // P_MAX) + j_subtile_H
                                if not cfg.add_layer_norm_bias:  # Multiply gamma weights only.
                                    nisa.tensor_scalar(
                                        dst=input_sb[i_tile_S][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_sbuf, s_tile_sz),
                                        ],
                                        data=input_transposed_psum[tp_psum_bank_idx][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_psum, s_tile_sz),
                                        ],
                                        op0=nl.multiply,
                                        operand0=gamma_norm_weights_sb[0:h_subtile_sz, nl.ds(gamma_tile_index, 1)],
                                    )
                                else:  # In addition to gamma weights, multiply with beta weights as well (layer_norm_bias).
                                    beta_tile_index = gamma_tile_index
                                    nisa.tensor_scalar(
                                        dst=input_sb[i_tile_S][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_sbuf, s_tile_sz),
                                        ],
                                        data=input_transposed_psum[tp_psum_bank_idx][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_psum, s_tile_sz),
                                        ],
                                        op0=nl.multiply,
                                        operand0=gamma_norm_weights_sb[0:h_subtile_sz, nl.ds(gamma_tile_index, 1)],
                                        op1=nl.add,
                                        operand1=layer_norm_bias_sb[0:h_subtile_sz, nl.ds(beta_tile_index, 1)],
                                    )
                        else:  # In NO_NORM, RMS_NORM_SKIP_GAMMA, we just copy the transposed PSUM tile to SBUF.
                            # We copy [128, 512] elements in single tensor copy for performance.
                            # -> for S % 128 !=0, we'll copy some garbage memory (to be masked later).
                            # Otherwise, we need extra F_MAX/P_MAX loop here.
                            nisa.tensor_copy(
                                dst=input_sb[i_tile_S][0:P_MAX, nl.ds(h_tile_offset, h_tile_sz)],
                                src=input_transposed_psum[tp_psum_bank_idx][0:P_MAX, 0:h_tile_sz],
                            )

                        # End of i_tile_H loop.
                    # End of i_tile_S loop.
                    sbm.close_scope()  # act and bn_stats_result are the only allocated tensors in this scope.
                # End of cfg.load_input_with_DMA_transpose conditional

            #######################################################################################################
            # Step 4: (QKV Projection) Multiply transposed input buffer (potentially with norm pre-applied) with weights.
            #######################################################################################################

            # PSUM accumulation buffer for QKV matmult results.
            # Each column of 512 size of rhs (weights) is accumulated to a distinct PSUM bank.
            # Since kernel assumes I <= 4096 (=psum_banks * 512), we have enough banks to accumulate all columns without COPYs of intermediate results.
            qkv_MM_num_psum_banks_needed = dims.num_512_tiles_per_I * num_S_tiles_in_block
            qkv_MM_output_psum = []
            for bank_id in nl.affine_range(qkv_MM_num_psum_banks_needed):
                if cfg.use_auto_allocation:
                    qkv_MM_output_psum.append(nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum))
                else:
                    qkv_MM_output_psum.append(
                        nl.ndarray(
                            (P_MAX, F_MAX),
                            dtype=nl.float32,
                            buffer=nl.psum,
                            address=(0, bank_id * PSUM_BANK_SIZE),
                        )
                    )

            """
            Multiply transposed input_sb @ weight tensor.
            
            The following loop reads from transposed input_sb(i_tile_S, 128, H),
            and outputs to psum_buffer(psum_banks_used, 128, 512).
            
            * Loop Structure of QKV Projection (in the case of non-prefetched weights):
            * Here,  weight_load_block_size_per_H = 1024.
            for each WEIGHT_BLOCK of 1024 size (along H):
                Load [1024, I] of weights to SBUF at once.  
                * We have 8 * [128, I] sub-tiles of H in a single load.           
                
                for each row of S buffer:                  ( e.g. 1, [128, H] sized rows)
                    for jth_subtile 0 to 8:                       (1024 / 128 = 8)
                        for each 512 column tile of weights (along I)
                            Multiply (weights) tile [128, 512] with the corresponding (input) tile in transposes_input_row [128, 128].
                            * Each of 512 tiles (columns in I) is accumulated to a different PSUM bank. 
            """

            # Allocate weights here, if not prefetched already.
            if not use_weight_prefetch:
                # Load weights [weight_load_block_size_per_H=1024, I] at a time.
                # Default weight constants are meant for non-prefetches case (they are over-written in case of prefetching)
                # Note: Projection uses num_weight_buffers and weight_load_block_size_per_H for indexing regardless of use_weight_prefetch.
                num_weight_buffers = dims.NUM_WEIGHT_BUFFERS_DEFAULT  # 4
                weight_load_block_size_per_H = dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT  # 1024
                num_weight_load_blocks_per_H = math.ceil(H / weight_load_block_size_per_H)
                max_num_128_H_subtiles_per_weight_block = math.ceil(
                    weight_load_block_size_per_H / 128
                )  # e.g 1024 / 128 = 8.

                # We load weights using a strided access pattern with only a single DMA ISA.
                weights_sb = []
                for _ in range(num_weight_buffers):
                    weights_sb.append(
                        sbm.alloc_stack(
                            (P_MAX, max_num_128_H_subtiles_per_weight_block, I),
                            dtype=cfg.compute_mm_dtype,
                            buffer=nl.sbuf,
                        )
                    )

            # If use_weight_prefetch, then NUM_WEIGHT_LOADS_PER_H == 1.
            for i_weight_load in nl.affine_range(num_weight_load_blocks_per_H):
                weight_load_offset = i_weight_load * weight_load_block_size_per_H
                curr_num_128_H_subtiles_per_weight_block = min(
                    max_num_128_H_subtiles_per_weight_block,
                    math.ceil((H - weight_load_offset) / 128),
                )

                if not use_weight_prefetch:
                    # If we did not prefetch weights on top, load a section of weights -> compute -> load a section of weights -> etc.
                    # Example of the strided weight load in the comment below.
                    weight_buf = weights_sb[i_weight_load % num_weight_buffers]
                    nisa.dma_copy(
                        dst=weight_buf.ap(
                            pattern=[
                                [max_num_128_H_subtiles_per_weight_block * I, P_MAX],
                                [I, curr_num_128_H_subtiles_per_weight_block],
                                [1, I],
                            ],
                            offset=0,
                        ),
                        src=fused_qkv_weights_hbm.ap(
                            pattern=[
                                [I, min(P_MAX, H - weight_load_offset)],
                                [128 * I, curr_num_128_H_subtiles_per_weight_block],
                                [1, I],
                            ],
                            offset=weight_load_offset * I,
                        ),
                        dge_mode=dge_mode.swdge,
                    )

                    """
                    Strided HBM->SBUF weights load example, if loading 1024 x I weights at a time.
                    Here, weight_load_block_size_per_H  = 1024.
                    
                    HBM Weights
                    ------------
                                        I
                             -----------------------------
                        128 |       H_1                  |
                    1024 128|       H_2                  | H
                        ...                            ...
                            |       H_8                  |
                             -----------------------------
                                        ....
                            
                    SBUF Weights
                    ------------
                                                8 * I
                            -------------------------------------------------
                        128|  H_1   |  H_2 |      ....              |  H_8   |
                            -------------------------------------------------
                        
                    Note: Access pattern on HBM side is strided, we are skipping 128 * I elements each time.
                        Order:
                            [0, 0:I], [128, 0:I], [256, 0:I], ...   ( 8 rows of I elements)
                            [1, 0:I], [129, 0:I], [257, 0:I], ...   ( 8 rows of I elements)
                        
                    On SBUF side,
                            1st row of H_1, and 1st row H_2 will be both partition=0, etc.                 
                    """

                for i_tile_S in nl.affine_range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                    s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                    if cfg.load_input_with_DMA_transpose:
                        # Recall we did not use PE Array to transpose input buffer in case of loadWithTranspose.
                        # Use nisa.dma_transpose(...) to load/transpose just enough of input for this round of matmult.
                        # Load/transpose only [128, 1024] elements of input.

                        # NOTE: To drop H divisible by 128 constraint, update AP below with valid "num_h" for the last iteration.
                        src_offset = (
                            i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H + weight_load_offset
                        )
                        nisa.dma_transpose(
                            dst=input_sb[i_tile_S].ap(
                                pattern=[
                                    [H, 128],
                                    [1, 1],
                                    [128, curr_num_128_H_subtiles_per_weight_block],
                                    [1, s_tile_sz],
                                ],
                                offset=weight_load_offset,
                            ),
                            src=input_hbm.ap(
                                pattern=[
                                    [H, s_tile_sz],
                                    [1, 1],
                                    [128, curr_num_128_H_subtiles_per_weight_block],
                                    [1, 128],
                                ],
                                offset=src_offset,
                            ),
                        )

                    for j_128_subtile_of_weight_load in nl.affine_range(curr_num_128_H_subtiles_per_weight_block):
                        for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                            # Each 512 column of I is accumulated to a different PSUM bank.
                            psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I

                            h_subtile_offset = (
                                weight_load_block_size_per_H * i_weight_load + 128 * j_128_subtile_of_weight_load
                            )
                            i_offset = 512 * k_tile_I

                            h_subtile_sz = min(P_MAX, H - h_subtile_offset)
                            i_tile_sz = min(512, I - i_offset)

                            # Stationary PSUM tile is input tile:  offset_in_transposed_input_row   + [128,128].
                            # Moving PSUM tile is weights_sb tile: offset_in_weights_sbuf           + [128,512].
                            nisa.nc_matmul(
                                stationary=input_sb[i_tile_S][
                                    0:h_subtile_sz, nl.ds(h_subtile_offset, s_tile_sz)
                                ],  # Use h_subtile_offset as [S,H]->[H,S] was transposed in-place.
                                moving=weights_sb[i_weight_load % num_weight_buffers][
                                    0:h_subtile_sz,
                                    j_128_subtile_of_weight_load,
                                    nl.ds(512 * k_tile_I, i_tile_sz),
                                ],
                                dst=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:i_tile_sz],
                                psumAccumulateFlag=3,
                            )
                # End of i_weight_load loop

            #######################################################################################################
            # Step 5: Copy PSUM results from matmult back to SBUF, and optionally apply fused RoPE.
            #######################################################################################################
            # Store results to SBUF before copying them to HBM output_tensor.

            # We have one matmult result per each 512 tile/column of weights stored in psum_buffer[bank_index].
            for i_tile_S in nl.affine_range(num_S_tiles_in_block):
                s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                # Copy results PSUM -> SBUF, apply RoPE fusion, (optionally) add_bias.
                if cfg.fused_rope:
                    _copy_psum_to_sbuf_apply_rope_and_bias(
                        qkv_MM_output_psum=qkv_MM_output_psum,
                        output_sb=output_sb,
                        cos_buffer_sb=cos_buffer_sb,
                        sin_buffer_sb=sin_buffer_sb,
                        rope_intermediate_buffer_sb=rope_intermediate_buffer_sb,
                        cos_cache_hbm=cos_cache_hbm,
                        sin_cache_hbm=sin_cache_hbm,
                        i_tile_S=i_tile_S,
                        s_tile_sz=s_tile_sz,
                        i_batch=i_batch,
                        s_tile_local_offset=s_tile_local_offset,
                        cfg=cfg,
                        dims=dims,
                        bias_sb=bias_sb if cfg.add_bias else None,
                    )
                # Copy results PSUM -> SBUF, (optionally) add_bias.
                else:
                    for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                        psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I
                        num_i = min(512, I - 512 * k_tile_I)

                        if cfg.add_bias:
                            nisa.tensor_tensor(
                                dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(512 * k_tile_I, num_i)],
                                data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                                data2=bias_sb[0:s_tile_sz, nl.ds(512 * k_tile_I, num_i)],
                                op=nl.add,
                            )
                        else:
                            nisa.tensor_copy(
                                dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(512 * k_tile_I, num_i)],
                                src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                            )

                # End of i_tile_S loop.

            #######################################################################################################
            # Step 6: Store SBUF results back to HBM, using given output layout.
            #######################################################################################################
            # This parts reads from output_matmult_sbuf and writes to out_tensor.

            if cfg.output_layout == QKVOutputLayout.BSD:
                # output_tensor shape: [B, S, I].
                # output_matmult_sbuf contains [128 (S), I].
                for i_tile_S in range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                    s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                    nisa.dma_copy(
                        dst=output_hbm.ap(
                            pattern=[[I, s_tile_sz], [1, I]],
                            offset=i_batch * dims.S * I + (dims.S_shard_offset + s_tile_local_offset) * I,
                        ),
                        src=output_sb[i_tile_S][0:s_tile_sz, 0:I],
                        dge_mode=dge_mode.swdge,
                    )

            else:  # NBSd = [heads, B, S, head_dim], I = heads * head_dim
                d_head = cast(int, dims.d_head)  # Safe due to validation
                for i_head in range(dims.num_heads):
                    for i_tile_S in range(num_S_tiles_in_block):
                        s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                        s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)
                        num_d = min(d_head, I - (i_head * d_head))

                        nisa.dma_copy(
                            dst=output_hbm.ap(
                                pattern=[[d_head, s_tile_sz], [1, num_d]],
                                offset=i_head * dims.B * dims.S * d_head
                                + i_batch * dims.S * d_head
                                + (dims.S_shard_offset + s_tile_local_offset) * d_head,
                            ),
                            src=output_sb[i_tile_S].ap(
                                pattern=[[I, s_tile_sz], [1, num_d]],
                                offset=i_head * d_head,
                            ),
                            dge_mode=dge_mode.swdge,
                        )
            sbm.close_scope()  # Deallocate all multi-buffered tensors.
            # End of i_buffer_s loop
        # End of batch loop
    sbm.close_scope()
    return output_hbm


def _load_and_broadcast_bias(
    bias_hbm: nl.ndarray, cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager
) -> nl.ndarray:
    """
    Loads bias with shape [1,I] to SBUF and broadcasts it to [P_MAX, I], using stream_shuffle.

    Returns allocated SBUF bias tensor.
    Note: User is responsible for deallocating SBUF tensor.
    """
    # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
    bias_sb = sbm.alloc_stack((P_MAX, dims.I), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=bias_sb[0:1, 0 : dims.I],
        src=bias_hbm[0:1, 0 : dims.I],
        dge_mode=dge_mode.swdge,
    )
    # Stream Shuffle works on 32 partitions only, apply it P_MAX // 32 = 4 times.
    MAX_STREAM_SHUFFLE_PARTITIONS = 32
    NUM_BROADCASTS = P_MAX // MAX_STREAM_SHUFFLE_PARTITIONS
    for i in nl.affine_range(NUM_BROADCASTS):
        nisa.nc_stream_shuffle(
            dst=bias_sb[
                nl.ds(i * MAX_STREAM_SHUFFLE_PARTITIONS, MAX_STREAM_SHUFFLE_PARTITIONS),
                0 : dims.I,
            ],
            src=bias_sb[0:1, 0 : dims.I],
            shuffle_mask=[0] * MAX_STREAM_SHUFFLE_PARTITIONS,
        )
    return bias_sb


def _load_norm_weights(
    norm_weights_hbm: nl.ndarray,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Loads norm_weights with shape [H] to SBUF as [P_MAX, H // P_MAX].

    Returns allocated SBUF norm_weights tensor.
    Note: User is responsible for deallocating SBUF tensor.

    Used by RMS_NORM and LAYER_NORM to load gamma_weights_hbm to SBUF.
    In addition, may be used in LAYER_NORM to load layer_norm_bias to SBUF.
    """
    # norm_weights_hbm have 1D shape [H], make it 2-D to make NKI loads easier.
    norm_weights_hbm = norm_weights_hbm.reshape((dims.H, 1))
    # We load norm_weights into SBUF as a 2-D tensor of shape [128, H // P_MAX].
    # Note: We later do the multiplication on transposed input [H, S], so the math works out.
    norm_elements_in_free_dim = dims.num_128_tiles_per_H
    norm_weights_sb = sbm.alloc_stack((P_MAX, norm_elements_in_free_dim), dtype=cfg.act_dtype, buffer=nl.sbuf)

    # Load in tiles of [128,1] to SBUF, now H is the first dimension (DMA broadcasted).
    # It is loaded in a way that a single norm tile uses all P_MAX partitions, and has 1 element per partition.
    for i_gamma_tile in range(norm_elements_in_free_dim):
        nisa.dma_copy(
            dst=norm_weights_sb[0:P_MAX, nl.ds(i_gamma_tile, 1)],
            src=norm_weights_hbm[nl.ds(i_gamma_tile * P_MAX, P_MAX), 0:1],
            dge_mode=dge_mode.swdge,
        )
    return norm_weights_sb


def _multi_buffering_degree_for_seqlen(cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager) -> Tuple[int, int]:
    """
    Compute maximum multi-buffering degree that we can use for SEQLEN without over-flowing SBUF or PSUM space.

    WARNING: This is not independently useful function, its correctness is based on the tensor allocation that comes after it.
    This is a 'lookahead' function.
    NOTE: If any additional tensors are added in the kernel, this function needs to be updated.

    Goal is to find the MAX "multi_buffer_degree" such that:
    (multi_buffer_degree * X) + Y < sbuf_space (per_partition), where
    X = sbuf_space_taken_by_tensors_about_to_be_multi_buffered (per_partition)
    Y = sbuf_space_taken_by_live_non_buffered_tensors (per_partition)

    Note: cfg.total_available_sbuf_space_to_this_kernel gives SBUF space PER_PARTITION.

    Assumes:
        * Weight prefetching decision is made after we choose multi-buffering degree.
        * For SBUF space calculations, we take into account the space taken by non-prefeched weights.
        * All globally allocated tensors have already been allocated, so that we can use sbm.get_free_space().
            Note: Still need to do look-ahead calculation for the tensors after call to this function is made.

    Returns: multi_buffer_degree, projected_total_sbuf_space_taken (including all tensors).
    """

    # Cannot multi-buffer more than dims.S_shard / P_MAX, e.g. if S_shard=256, best we can do is 2.
    s_multi_buffer_degree = 1
    s_multi_buffer_degree = min(math.ceil(dims.S_shard / P_MAX), dims.MAX_S_MULTI_BUFFER_DEGREE)

    # ------------------- Make sure multi-buffering does not cause SBUF overflow --------------------#

    # ------------------------ SBUF Space Taken by Non Buffered Tensors ----------------------------#
    #     This is the space that will be consumed by tensors we will not multi-buffer
    #     This calculation assumes we are not pre-fetching weights (this can be decided after buffering)
    #     These same constants are used in the allocation of weight tensor.

    # Sum up sizes of: # zero_bias_sb, norm_eps_sb, bias_sb, gamma_weights_sb, l
    #   layer_norm_bias_sb, act_reduce_sum, bn_stats_result, and weights_sb (non-prefetched)
    sbuf_tile_space_non_buffered = 0
    # zero_bias_sb, norm_eps_sb, bias_sb, gamma_weights_sb, layer_norm_bias_sb, act_reduce_sum, bn_stats_result.
    sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # zero_bias_sb (P_MAX, 1)
    sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # norm_eps_sb (P_MAX, 1)
    if cfg.add_bias:
        sbuf_tile_space_non_buffered += dims.I * sizeinbytes(cfg.compute_mm_dtype)  # bias_sb (P_MAX, dims.I)
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        # gamma_weights_sb (P_MAX, num_128_tiles_per_H)
        sbuf_tile_space_non_buffered += dims.num_128_tiles_per_H * sizeinbytes(cfg.act_dtype)
        if cfg.add_layer_norm_bias:
            # layer_norm_bias_sb (P_MAX, num_128_tiles_per_H)
            sbuf_tile_space_non_buffered += dims.num_128_tiles_per_H * sizeinbytes(cfg.act_dtype)

    # act_reduce_sum and bn_stats_result_sb appear in the loop.
    # sbuf_tile_space_non_buffered = nl.tile_size.total_available_sbuf_size - sbm.get_free_space() # space taken so far.
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.act_dtype)  # act_reduce_sum (P_MAX, 1)
    if cfg.fused_norm_type == NormType.LAYER_NORM:
        # bn_stats_result_sb (P_MAX, 6*NUM_512_BN_STATS_TILES_H)
        BN_STATS_FMAX = 512  # nl.tile_size.bn_stats_fmax  # 512
        NUM_512_BN_STATS_TILES_H = math.ceil(dims.H / BN_STATS_FMAX)
        sbuf_tile_space_non_buffered += 6 * NUM_512_BN_STATS_TILES_H * sizeinbytes(cfg.act_dtype)

    weights_space_per_partition = (
        dims.NUM_WEIGHT_BUFFERS_DEFAULT
        * (dims.I * math.ceil(dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT / 128))
        * sizeinbytes(cfg.compute_mm_dtype)
    )
    sbuf_tile_space_non_buffered += weights_space_per_partition

    # --------------------SBUF Space Taken By Tensors We Will be Multi-Buffering ---------------------#

    sbuf_tile_space_pre_buffering = _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
        cfg=cfg, dims=dims, sbm=sbm
    )

    # Note: cfg.total_available_sbuf_space_to_this_kernel is total_available_sbuf_space PER PARTITION.
    max_s_buffer_without_exceeding_sbuf = (
        cfg.total_available_sbuf_space_to_this_kernel - sbuf_tile_space_non_buffered
    ) // sbuf_tile_space_pre_buffering
    s_multi_buffer_degree = min(s_multi_buffer_degree, max_s_buffer_without_exceeding_sbuf)

    # Step (3) Ensure multi-buffering does not exceed number of PSUM banks.
    # Later we use NUM_512_TILES_PER_H * s_multi_buffer_degree for psum_banks. (NUM_512_TILES_PER_H <= 4, since I <= 4096)
    # Ensure NUM_512_TILES_PER_H * s_multi_buffer_degree <= 8
    MAX_PSUM_TILING_GROUPS = NUM_HW_PSUM_BANKS // dims.num_512_tiles_per_I
    s_multi_buffer_degree = min(s_multi_buffer_degree, MAX_PSUM_TILING_GROUPS)

    projected_sbuf_taken_space = s_multi_buffer_degree * sbuf_tile_space_pre_buffering + sbuf_tile_space_non_buffered
    return s_multi_buffer_degree, projected_sbuf_taken_space


def _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
    cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager
) -> int:
    """
    Compute the total SBUF space taken (per partition) by simultaneously live tensors that will be multi-buffered in the kernel.

    WARNING: This is not independently useful function, its correctness is based on the tensor allocation that comes after it.
    This is a 'lookahead' function.
    NOTE: If any additional tensors are added in the kernel, this function needs to be updated.

    Current tensors inside a loop that will get buffered are:
    'input_sb', 'output_sb'                                      (always)
    'square_sum_sb',                                             (if cfg.fused_norm_type.RMS_NORM or cfg.fused_norm_type.RMS_NORM_GAMMA)
    'bn_aggr_result_sb'                                          (if cfg.fused_norm_type.LAYER_NORM)
    'cos_buffer_sb', 'sin_buffer_sb', 'rope_intermediate_buffer' (if cfg.fused_rope)
    """

    pre_buffer_tile_space_per_partition = 0
    # 'input_sb  [P_MAX, H]'
    pre_buffer_tile_space_per_partition += dims.H * sizeinbytes(cfg.compute_mm_dtype)
    # 'output_sb [P_MAX, I]'
    pre_buffer_tile_space_per_partition += dims.I * sizeinbytes(cfg.compute_mm_dtype)

    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        pre_buffer_tile_space_per_partition += 1 * sizeinbytes(cfg.act_dtype)  # 'square_sum_sb [P_MAX, 1]'

    if cfg.fused_norm_type == NormType.LAYER_NORM:
        NUM_AGGR_STATS = 2
        # 'bn_aggr_result_sb [P_MAX, NUM_AGGR_STATS]'
        pre_buffer_tile_space_per_partition += NUM_AGGR_STATS * sizeinbytes(cfg.act_dtype)

    if cfg.fused_rope:
        # 'cos_buffer_sb [P_MAX, d_head]'
        pre_buffer_tile_space_per_partition += dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
        # 'sin_buffer_sb [P_MAX, d_head // 2]'
        pre_buffer_tile_space_per_partition += dims.d_head // 2 * sizeinbytes(cfg.compute_mm_dtype)
        # 'rope_intermediate_buffer [P_MAX, d_head * 2]'
        pre_buffer_tile_space_per_partition += dims.d_head * 2 * sizeinbytes(cfg.compute_mm_dtype)

    return pre_buffer_tile_space_per_partition


def _use_weight_prefetch(
    projected_sbuf_taken_space_after_multi_buffer: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> bool:
    """
    Returns True if we can afford weight prefetching, given projected space requirements post multi-buffering.
    """
    # This is how much space we need to prefetch weights, and keep them on SBUF through the entire kernel.
    weights_NEW_space_needed = (dims.I * dims.num_128_tiles_per_H) * sizeinbytes(cfg.compute_mm_dtype)
    # Subtract the weights_OLD_space (non-prefetched), which was taken into account by multi-buffering space calculation.
    weights_OLD_space_taken = (
        dims.NUM_WEIGHT_BUFFERS_DEFAULT
        * (dims.I * math.ceil(dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT / P_MAX))
        * sizeinbytes(cfg.compute_mm_dtype)
    )
    # Note: In auto-allocation mode, sbuf space calculations do not make sense, but they do not break kernel correctness.
    can_weight_prefetch = (
        projected_sbuf_taken_space_after_multi_buffer - weights_OLD_space_taken
    ) + weights_NEW_space_needed < cfg.total_available_sbuf_space_to_this_kernel

    # Note: S >= 1024 should be investigated further. For small S, prefetching causes degradation in some cases.
    weight_prefetch_heuristic = (dims.S_shard >= 1024) or (dims.I >= 1024)
    use_weight_prefetch = can_weight_prefetch and weight_prefetch_heuristic
    return use_weight_prefetch


def _apply_rms_normalization(
    input_row_sb: nl.ndarray,
    square_sum_row_sb: nl.ndarray,
    zero_bias: nl.ndarray,
    norm_eps: nl.ndarray,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> None:
    """
    Apply RMS normalization to input tile in-place (no gamma weights applied yet).
    Multiply input_row_sb by 1 / RMS(x) =  1 / sqrt(eps + (1 / hidden_actual) * (x1^2 + x2^2 + ... + xn^2)).
    Src tensors: input_row_sb (+ norm_eps) [s_tile_sz, dims.H]
                 square_sum_row_sb         [s_tile_sz, dims.H] , temporary buffer pre-allocated.
    Dst tensors: input_row_sb (+ norm_eps) [s_tile_sz, dims.H] (multiplied by 1/RMS(x) ).
    """

    # NOTE: "act" tensor is not used, we only use reduce_res=square_sum from activation_reduce output, but NKI ISA requires dst tensor.
    # nisa.activation_reduce(...) requires src and dst APs shapes to be of equal size, but (input_sb[i_tile_S] has shape [128, H]).
    # To get around this, we use 0-Step AccessPattern "dst=act.ap(pattern=[[1, num_s],[0,H]])".
    # Keeping "act" shape as [pmax, H] would waste valuable SBUF space and potentally limit multi-buffering and weight-prefetching.

    # Temporary tensor allocation
    act_reduce_sbm = sbm.alloc_stack((P_MAX, 1), dtype=cfg.act_dtype, buffer=nl.sbuf)

    # Sum of squares: x1^2 + x2^2 + ... + xn^2 ( sum of squares of the input row ).
    nisa.activation_reduce(
        dst=act_reduce_sbm.ap(pattern=[[1, s_tile_sz], [0, dims.H]]),
        op=nl.square,
        data=input_row_sb[0:s_tile_sz, 0 : dims.H],
        reduce_op=nl.add,
        reduce_res=square_sum_row_sb[0:s_tile_sz, 0:1],
        bias=zero_bias[0:s_tile_sz, 0:1],
    )

    # Reciprocal square root: 1 / RMS(x) =  1 / sqrt(eps + (1 / hidden_actual) * (x1^2 + x2^2 + ... + xn^2)).
    nisa.activation(
        dst=square_sum_row_sb[0:s_tile_sz, 0:1],
        op=nl.rsqrt,
        data=square_sum_row_sb[0:s_tile_sz, 0:1],
        bias=norm_eps[0:s_tile_sz, 0:1],
        scale=float(1.0 / dims.H_actual),
    )

    # Apply normalization: Multiply input_sb by 1 / RMS(x).
    nisa.tensor_scalar(
        dst=input_row_sb[0:s_tile_sz, :],
        data=input_row_sb[0:s_tile_sz, :],
        op0=nl.multiply,
        operand0=square_sum_row_sb[0:s_tile_sz, 0:1],
        engine=nisa.vector_engine,
    )


def _compute_layer_norm_stats(
    input_row_sb: nl.ndarray,
    bn_aggr_result_tile: nl.ndarray,
    norm_eps: nl.ndarray,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> None:
    """
    Compute LayerNorm statistics for row of input_sb and store it bn_aggr_result_tile.
        mean = bn_aggr_result_tile[0:s_tile_sz, 0:1]
        rvar = bn_aggr_result_tile[0:s_tile_sz, 1:2] #rvar(var + eps)

    Src tensor: input_row_sb (+ norm_eps) [s_tile_sz, dims.H]
    Dst tensor: bn_aggr_result_tile       [s_tile_sz, 2]
    """
    # LayerNorm constants
    BN_STATS_TILE_SIZE = 512
    BN_STATS_DST_SIZE = 6
    NUM_512_BN_STATS_TILES_H = math.ceil(dims.H / BN_STATS_TILE_SIZE)

    # Allocate temporary tensor for bn_stats results
    # nisa.bn_stats(...) outputs 6 different metrics per tile (to later be aggregated to mean/varaince by nisa.bn_aggr(...)).
    bn_stats_result_sb = sbm.alloc_stack(
        (P_MAX, BN_STATS_DST_SIZE * NUM_512_BN_STATS_TILES_H),
        dtype=cfg.act_dtype,
        buffer=nl.sbuf,
    )

    # Compute bn_stats for each 512-sized tile along H dimension
    # Note: All nisa.bn_stats(..) computation is done on float32.
    for i_bn_tile in nl.affine_range(NUM_512_BN_STATS_TILES_H):
        # Calculate valid H elements for this tile
        bn_tile_offset = i_bn_tile * BN_STATS_TILE_SIZE
        bn_tile_sz = min(BN_STATS_TILE_SIZE, dims.H - bn_tile_offset)

        nisa.bn_stats(
            dst=bn_stats_result_sb[0:s_tile_sz, nl.ds(i_bn_tile * BN_STATS_DST_SIZE, BN_STATS_DST_SIZE)],
            data=input_row_sb[0:s_tile_sz, nl.ds(bn_tile_offset, bn_tile_sz)],
        )

    # Aggregate 6 bn_stats metrics into mean and variance
    NUM_AGGR_STATS = 2
    nisa.bn_aggr(
        dst=bn_aggr_result_tile[0:s_tile_sz, 0:NUM_AGGR_STATS],
        data=bn_stats_result_sb[0:s_tile_sz, 0 : BN_STATS_DST_SIZE * NUM_512_BN_STATS_TILES_H],
    )

    # Compute reciprocal square root of variance for normalization
    nisa.activation(
        dst=bn_aggr_result_tile[0:s_tile_sz, 1:NUM_AGGR_STATS],
        data=bn_aggr_result_tile[0:s_tile_sz, 1:NUM_AGGR_STATS],
        bias=norm_eps[0:s_tile_sz, 0:1],
        op=nl.rsqrt,
    )


def _copy_psum_to_sbuf_apply_rope_and_bias(
    qkv_MM_output_psum: List[nl.ndarray],
    output_sb: List[nl.ndarray],
    cos_buffer_sb: List[nl.ndarray],
    sin_buffer_sb: List[nl.ndarray],
    rope_intermediate_buffer_sb: List[nl.ndarray],
    cos_cache_hbm: nl.ndarray,
    sin_cache_hbm: nl.ndarray,
    i_tile_S: int,
    s_tile_sz: int,
    i_batch: int,
    s_tile_local_offset: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    bias_sb: Optional[nl.ndarray],
) -> None:
    """
    Apply RoPE rotation to Q/K heads and copy V heads from PSUM matmul results to output buffer.
    Performs the copy only for "i_tile_S" contracted row.

    Src: * qkv_MM_output_psum (QKV Projection Results)
         * Pre-allocated RoPE buffers: cos_buffer_sb, sin_buffer_sb, rope_intermediate_buffer_sb.
            and corresponding HBM tensors: cos_buffer_hbm, sin_buffer_hbm

    Dst: Store results to output_matmult_sb[i_tile_S]

    - Each element is a PSUM bank [128, 512] storing results for specific (S_tile, I_tile)
    - Bank indexing: i_tile_S * dims.num_512_tiles_per_I + k_tile_I
    - Contains Q, K, V head data across different banks based on head_offset
    """

    d_head = dims.d_head
    d_head_half = d_head // 2

    NUM_HEADS_PER_PSUM_BANK = 512 // d_head

    # Load RoPE tensors if RoPE fusion is enabled.
    cos_src_offset = i_batch * dims.S * d_head + (dims.S_shard_offset + s_tile_local_offset) * d_head
    nisa.dma_copy(
        dst=cos_buffer_sb[i_tile_S].ap(pattern=[[d_head, s_tile_sz], [1, d_head]], offset=0),
        src=cos_cache_hbm.ap(pattern=[[d_head, s_tile_sz], [1, d_head]], offset=cos_src_offset),
        dge_mode=dge_mode.swdge,
    )

    sin_src_offset = i_batch * dims.S * d_head + (dims.S_shard_offset + s_tile_local_offset) * d_head
    nisa.dma_copy(
        dst=sin_buffer_sb[i_tile_S].ap(pattern=[[d_head_half, s_tile_sz], [1, d_head_half]], offset=0),
        src=sin_cache_hbm.ap(pattern=[[d_head, s_tile_sz], [1, d_head_half]], offset=sin_src_offset),
        dge_mode=dge_mode.swdge,
    )

    # For each head, RoPE([X1, X2]) = [X1, X2] * cos + [-X2 * sin, X1 * sin]
    for i_head in nl.sequential_range(dims.num_q_heads + dims.num_kv_heads):
        head_offset = i_head * d_head
        num_d = min(d_head, dims.I - head_offset)
        num_d_half = num_d // 2

        psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + i_head // NUM_HEADS_PER_PSUM_BANK
        psum_head_offset = (i_head % NUM_HEADS_PER_PSUM_BANK) * d_head

        # Copy the current head from psum to sbuf first. we maintain two copy of the head, the first copy is for cos * X and the second for sin * rotate_half(X)
        if cfg.add_bias:
            nisa.tensor_tensor(
                dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
                data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                data2=bias_sb[0:s_tile_sz, nl.ds(head_offset, num_d)],
                op=nl.add,
            )
        else:
            # Copy the current head from psum to sbuf first. we maintain two copy of the head, the first copy is for cos * X and the second for sin * rotate_half(X)
            nisa.tensor_copy(
                dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
                src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
            )

            # -X2 * sin
        nisa.tensor_tensor(
            dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head, num_d_half)],
            data1=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head_half, num_d_half)],
            data2=sin_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d_half],
            op=nl.multiply,
        )

        nisa.tensor_scalar(
            dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head, num_d_half)],
            data=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head, num_d_half)],
            op0=nl.multiply,
            operand0=-1,
        )

        # X1 * sin
        nisa.tensor_tensor(
            dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head + d_head_half, num_d_half)],
            data1=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d_half],
            data2=sin_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d_half],
            op=nl.multiply,
        )

        # X * cos
        nisa.tensor_tensor(
            dst=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
            data1=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
            data2=cos_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
            op=nl.multiply,
        )

        # Copy X * cos + [-X2 * sin, X1 * sin] to output sbuf
        nisa.tensor_tensor(
            dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
            data1=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, 0:num_d],
            data2=rope_intermediate_buffer_sb[i_tile_S][0:s_tile_sz, nl.ds(d_head, num_d)],
            op=nl.add,
        )

    # Copy V
    for i_head in range(dims.num_q_heads + dims.num_kv_heads, dims.num_q_heads + 2 * dims.num_kv_heads):
        head_offset = i_head * d_head
        num_d = min(d_head, dims.I - head_offset)
        psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + i_head // NUM_HEADS_PER_PSUM_BANK
        psum_head_offset = (i_head % NUM_HEADS_PER_PSUM_BANK) * d_head

        if cfg.add_bias:
            nisa.tensor_tensor(
                dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
                data2=bias_sb[0:s_tile_sz, nl.ds(head_offset, num_d)],
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
            )
