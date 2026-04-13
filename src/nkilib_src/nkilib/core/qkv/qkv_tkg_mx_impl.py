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

"""QKV TKG MXFP Projection Kernel for token generation with MX quantization."""

import math
from typing import Optional, Tuple

import nki.isa as nisa
import nki.language as nl

from ..mlp.mlp_tkg.mlp_tkg_utils import _layout_adapter_sb
from ..subkernels.rmsnorm_tkg import rmsnorm_tkg as _rmsnorm_tkg
from ..utils.allocator import SbufManager, sizeinbytes
from ..utils.common_types import NormType, QKVOutputLayout, QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.tiled_range import TiledRange
from .qkv_tkg_mx_utils import (
    QKV_TKG_MXFP_Config,
    QKV_TKG_MXFP_UserInput,
    _build_config,
    _validate_user_inputs,
)

# Tiling constants
I_BLOCK_SIZE = 4096  # I_TILE_SIZE * 8 for 8 available PSUM banks
I_TILE_SIZE = 512  # Maximum free dimension of a matmul instruction (one PSUM bank)
WEIGHT_LOAD_BLOCK_SIZE = 2048  # Number of rows to load per weight block
P_MAX = 128  # Partition dimension size (128)


def _qkv_tkg_mx_impl(
    hidden: nl.ndarray,
    weights_qtz_hbm: nl.ndarray,
    norm_w: Optional[nl.ndarray] = None,
    fused_add: bool = False,
    mlp_prev: Optional[nl.ndarray] = None,
    attn_prev: Optional[nl.ndarray] = None,
    d_head: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.RMS_NORM,
    quantization_type: QuantizationType = QuantizationType.MX,
    is_h_dim_4h_transposed: bool = False,
    weight_scales_hbm: Optional[nl.ndarray] = None,
    output_in_sbuf: bool = False,
    qkv_bias: Optional[nl.ndarray] = None,
    norm_bias: Optional[nl.ndarray] = None,
    hidden_actual: Optional[int] = None,
    sbm: Optional[SbufManager] = None,
) -> nl.ndarray:
    """
    QKV MXFP Projection Kernel for Token Generation
    MXFP specific additional assumptions (subset of features supported by non-quantized version):
        -> Assumes weights_hbm are already quantized and stored in nl.float8_e4m3fn_x4 dtype.
        -> Assumes weights_scales are passed in nl.uint8 dtype as well.
        -> is_h_dim_4h_transposed must be set to True.
        -> H = hidden.shape[2] must be divisible by 512.
        -> BxS must be divisible by 4.
        -> Input must be on HBM.

    This kernel computes the fused QKV projection operation:
        hidden' = norm(hidden + attn_prev + mlp_prev)  # optional fused add and norm
        output = hidden' @ qkv_w + qkv_bias
    typically used before the attention block in transformer models.

    This kernel is optimized for Token Generation (aka Decoding) use cases where
    batch_size * seqlen is small. This kernel only supports BxS <= 128.

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
            Input hidden states tensor in HBM or SBUF.
            Shape:
                [B, S, H]         when in HBM
                [H0=128, BxS, H1] when in SBUF
        weights_qtz_hbm (nl.ndarray):
            QKV projection weight tensor in HBM, quantized offline.
            Dtype: nl.float8_e4m3fn_x4
            Shape: [H_packed, I], can be viewed as [H0, H1_packed, I].
            Note: Here H_packed = H // 4, where H = hidden.shape[2].
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
            Head dimension size D. Required for NBSd layout.
        num_q_heads : Optional[int], default=None
            Number of query heads
        num_kv_heads : Optional[int], default=None
            Number of key/value heads
        output_layout (QKVOutputLayout):
            Output tensor layout format. BSD: [B, S, I] or NBSd: [N, B, S, D]. Default: QKVOutputLayout.BSD.
        eps (float):
            Epsilon value to maintain numerical stability in normalization. Default: 1e-6.
        norm_type (NormType):
            Type of normalization to apply (NO_NORM, RMS_NORM, or LAYER_NORM). Default: NormType.RMS_NORM.
        quantization_type (QuantizationType):
            Must be QuantizationType.MX.
        is_h_dim_4h_transposed: bool, default=False
            Whether the H-dim (in input and gamma) has been pre-transposed by 4 (only applicable with MX Quantization).
            If is_h_dim_4h_transposed = False,
                * input has typical shape [B, S, H], viewed as [B, S, H//512, 128_H, 4_H].
            If is_h_dim_4h_transposed = True,
                * input has shape [B, S, H] but is pre-shuffled from
                  [B, S, H//512, 128_H, 4_H] -> [B, S, 4_H, H//512, 128_H] and flattened to [B, S, H].
                * IMPORTANT: H-dim in both input and gamma weights (for RMSNorm) must be pre-shuffled.
                    * For input, this is achieved by offline pre-shuffling weights of upstream projection (in real model).
                    * For gamma, this is achieved by offline pre-shuffling of gamma tensor.
                Purpose: More efficent for obtaining the required swizzled layout for quantize_mx instruction.
        weight_scales_hbm (nl.ndarray):
            QKV weight quantization scales for MXFP in HBM.
            dtype: uint8
            Shape: [H // 32, I] == [H_packed // 8, I]
            Note: Since weights_qtz_hbm is already quantized, weight scales are 8x times smaller, not 32x.
        output_in_sbuf (bool):
            If True, output is kept in SBUF; otherwise stored to HBM. Default: False.
            Only supports single I-block when True.
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

    Notes:
        - H must be divisible by 128 (nl.tile_size.pmax).
        - H1 (H//128) must be divisible by number of shards for multi-core execution.
        - output_in_sbuf only supports single I-block (I < 4096).

    Pseudocode:
        # Step 1: Load input and optionally apply RMSNorm
        if norm_type == RMS_NORM:
            hidden_sb = rmsnorm(hidden, gamma, eps)
        else:
            hidden_sb = dma_transpose(hidden)  # [B*S, H] -> [H0, B*S, H1]

        # Step 2: Quantize input for MXFP
        hidden_swizzled = layout_adapter_sb(hidden_sb)  # [H0, B*S, H1] -> [H0, H1_packed, B*S, 4]
        hidden_qtz, hidden_scale = quantize_mx(hidden_swizzled)

        # Step 3: MXFP Projection with tiled matmul
        for i_block in range(NUM_I_BLOCKS):
            for h_block in range(NUM_H_BLOCKS):
                weights_qtz = load_weights(h_block, i_block)
                weight_scales = load_weight_scales(h_block, i_block)
                psum += nc_matmul_mx(hidden_qtz, weights_qtz, hidden_scale, weight_scales)
            output[i_block] = psum + bias  # if bias enabled
            if num_shards > 1:
                output[i_block] = sendrecv_reduce(output[i_block])
    """

    # Build user inputs and validate
    user_inputs = QKV_TKG_MXFP_UserInput(
        hidden=hidden,
        weights_qtz_hbm=weights_qtz_hbm,
        norm_w=norm_w,
        fused_add=fused_add,
        mlp_prev=mlp_prev,
        attn_prev=attn_prev,
        d_head=d_head,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        output_layout=output_layout,
        eps=eps,
        norm_type=norm_type,
        quantization_type=quantization_type,
        is_h_dim_4h_transposed=is_h_dim_4h_transposed,
        weight_scales_hbm=weight_scales_hbm,
        output_in_sbuf=output_in_sbuf,
        qkv_bias=qkv_bias,
        norm_bias=norm_bias,
        hidden_actual=hidden_actual,
        sbm=sbm,
    )
    _validate_user_inputs(user_inputs)

    # Build config
    cfg = _build_config(user_inputs)
    B, S, H = hidden.shape
    BxS = B * S
    H0 = P_MAX
    H1 = H // H0
    if hidden_actual is None:
        hidden_actual = H

    hidden_sb = nl.ndarray((H0, BxS, H1), dtype=cfg.hidden_orig_dtype, buffer=nl.sbuf)
    """
    High-Level Layout Path (ignoring sharding):
        HBM input               [BxS, H]
     -> dma_transpose to        [H0, BxS, H1] (+rms_norm), H1 is viewed as outer-dim now.
     -> view as                 [H0, BxS, 4_H, H1_packed] (is_h_dim_4h_transposed=True needed for this 4_H view)
     -> free-dim tensor copy to [H0, H1_packed, BxS, 4_H]
     -> quantize_mx to          [H0, H1_packed, BxS]
     -> ideal for matmul_mx
    """

    # Step 1: Load Input and optionally apply RMS_NORM
    """
    The key is to view H as H1 * H0 with H1 being outer-dimension, and use dma_transpose
    (in both NO_NORM or RMS_NORM). Along with is_h_dim_4h_transposed, this will allow
    efficient re-layout for quantize_mx later.
    """
    hidden_sb = _load_hidden_and_apply_rms_norm(
        hidden_hbm=hidden,
        output_sb=hidden_sb,
        cfg=cfg,
        gamma_hbm=norm_w,
        eps=eps,
        hidden_actual=hidden_actual,
    )

    # Step 2: Quantize_MX input
    # _quantize_mx_input(..) will re-layout input for quantization and return quantized result.
    # hidden_qtz_sb shape: [H0, H1_packed_shard, BxS]
    hidden_qtz_sb, hidden_scales_sb = _quantize_mx_input(hidden_sb=hidden_sb, cfg=cfg)

    # Step 3: MXFP Projection
    output_hbm = _qkv_tkg_projection_mxfp(
        hidden_qtz_sb=hidden_qtz_sb,
        hidden_scales_sb=hidden_scales_sb,
        weights_qtz_hbm=weights_qtz_hbm,
        weight_scales_hbm=weight_scales_hbm,
        cfg=cfg,
        bias_hbm=qkv_bias,
    )

    return output_hbm.reshape((cfg.B, cfg.S, cfg.I))


def _load_hidden_and_apply_rms_norm(
    hidden_hbm: nl.ndarray,
    output_sb: nl.ndarray,
    cfg: QKV_TKG_MXFP_Config,
    gamma_hbm: Optional[nl.ndarray],
    eps: Optional[float] = 1e-6,
    hidden_actual: Optional[int] = None,
):
    """
    Loads the input to hidden_sb and (optionally) applies RMSNorm.

    Args:
        hidden_hbm (nl.ndarray):
            Input hidden states in HBM.
            Shape: [B, S, H].

        output_sb (nl.ndarray):
            Input hidden states in SBUF (destination).
            Shape: [H0, BxS, H1]

        gamma_hbm (nl.ndarray):
            Normalization weight tensor in HBM. Required when norm_type is RMS_NORM or LAYER_NORM.
            Shape:    [1, H]

        cfg (QKV_TKG_MXFP_Config): QKV TKG MXFP configuration.

        hidden_actual (int, optional): Non-padded H for RMSNorm.


    Returns:
        Returns the result in output_sb, with H1 being the outer-dimension.
    """
    B, S, H = hidden_hbm.shape
    BxS = B * S
    H0 = P_MAX
    H1 = H // H0

    # Note: We cannot LNC2 shard this strided dma_transpose.
    if cfg.fused_norm_type == NormType.NO_NORM:
        # The key is to view H as H1 * H0 with H1 being outer-dimension.
        hidden_for_transpose = hidden_hbm.reshape((BxS * H1, H0)).reshape((BxS * H1, 1, 1, H0))
        hidden_sb_flat = output_sb.reshape((H0, BxS * H1)).reshape((H0, 1, 1, BxS * H1))
        nisa.dma_transpose(dst=hidden_sb_flat, src=hidden_for_transpose)
    else:  # cfg.fused_norm_type == NormType.RMS_NORM:
        # In case of RMSNorm we use hidden_dim_tp=True to indicate H1 needs to be viewed as the outer-dimension.
        output_sb = _rmsnorm_tkg(
            input=hidden_hbm,
            gamma=gamma_hbm,
            output=output_sb,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=True,  # IMPORTANT
            single_core_forced=True,
        )

    return output_sb


def _quantize_mx_input(
    hidden_sb: nl.ndarray,
    cfg: QKV_TKG_MXFP_Config,
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Quantize hidden states for MXFP matmul.

    Args:
        hidden_sb (nl.ndarray):
            Input hidden states tensor in SBUF.
            Shape:
                [H0, BxS, H1] where H0=128, and H1 = H // H0
            Indexing: H is viewed as H1*H0 with H1 being the outer dimension.
                    -> Input was loaded with transpose.
        cfg (QKV_TKG_MXFP_Config): Kernel configuration.

    Note on is_h_dim_4h_transposed:
        This function assumes is_h_dim_4h_transposed=True.
        If swizzled, input has shape [B, S, H] but is pre-shuffled from
        [B, S, H//512, 128_H, 4_H] -> [B, S, 4_H, H//512, 128_H] and flattened to [B, S, H].

        Post dma_transpose, hidden_sb can be viewed as:
            [H0, BxS, H1] = [H0, BxS, 4_H * H // 512]
        This assumption allows for more efficient re-layout for quantize_mx.

    Returns:
        hidden_qtz_sb (nl.ndarray): [H0, H1_packed_shard, BxS], where H1_packed = H // 512, with nl.float8_e4m3fn_x4 dtype.
        hidden_scales_sb (nl.ndarray): [H0, H1_packed_shard, BxS], where H1_packed = H // 512, with nl.uint8 dtype.
    """
    H0, BxS, H1 = hidden_sb.shape
    H1_packed = H1 // 4
    H1_packed_shard = H1_packed // cfg.num_shards

    # Obtain needed layout for quantize_mx
    if cfg.is_h_dim_4h_transposed:
        """
        Pre-shuffling of H assumption (4_H being at front pre dma_transpose) allows us to
        obtain H layout necessary for quantization efficiently.
        START layout: [H0, BxS, H1] viewed as [H0, BxS, 4_H * H1_packed]
        GOAL layout for quantize_mx: [H0, H1_packed, BxS, 4_H]
        This can be achieved efficiently using free-dimension tensor_copy transpose
        in "_layout_adapter_sb" function.
        """
        hidden_swizzled_sb = _layout_adapter_sb(src=hidden_sb, n_prgs=cfg.num_shards, prg_id=cfg.shard_id)
        # Shape: [H0, H1_packed_shard, BxS, 4_H] - ready for quantize_mx
    else:
        kernel_assert(False, "[QKV TKG MXFP] is_dram_H_shuffled_with_4H_at_front=False is not implemented.")

    # Apply quantize_mx (_layout_adapter_sb returned sharded tensor)
    hidden_qtz_sb = nl.ndarray((H0, H1_packed_shard, BxS), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
    hidden_scales_sb = nl.ndarray((H0, H1_packed_shard, BxS), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.quantize_mx(dst=hidden_qtz_sb, src=hidden_swizzled_sb, dst_scale=hidden_scales_sb)

    return hidden_qtz_sb, hidden_scales_sb


def _qkv_tkg_projection_mxfp(
    hidden_qtz_sb: nl.ndarray,
    hidden_scales_sb: nl.ndarray,
    weights_qtz_hbm: nl.ndarray,
    weight_scales_hbm: nl.ndarray,
    cfg: QKV_TKG_MXFP_Config,
    bias_hbm: Optional[nl.ndarray] = None,
) -> nl.ndarray:
    """
    QKV MXFP Projection:
        Computes: hidden_qtz_sb @ weights_qtz_hbm.

    Note: This version differs from the current qkv_tkg projection in a few ways:
        (A) hidden_qtz_sb (SBUF) is assumed to be in [H0, H1, BxS] layout (with H1 being the outer dimension in H).
            * This differs from the original [H0, BxS, H1] layout, which won't work with MXFP.
        (B) No Column Tiling: MXFP version cannot have column tiling support (hardware incompatibility)

    Note: Here H_packed stands for the original H // 4.
    Args:
        hidden_qtz_sb (nl.ndarray):
            Input hidden states tensor in SBUF (already quantized).
            Dtype: nl.float8_e4m3fn_x4
            Shape:
                [H0, H1_packed_shard, BxS] where H0=128, and H1_packed_shard = H_packed_shard // H0.
                 e.g., H1_packed = H // 512 (or H_packed // 128).
            Indexing: h = h1*H0 + h0, i.e., H_packed is viewed as H1_packed*H0 with H1_packed being the outer dimension.
                Note: This means HBM->SBUF load requires a transpose.
                Torch equivalent of going from [H, BxS] -> [H0, H1, BxS] would be:
                    hidden_sb = input.reshape(H1, H0, BxS).permute(1, 0, 2)

        hidden_scales_sb (nl.ndarray):
            Input quantization scales for MXFP in SBUF.
            Dtype:  uint8
            Shape: [H0, H1_packed_shard, BxS],
            Note: Same indexing assumptions as hidden_qtz_sb.

        weights_qtz_hbm (nl.ndarray):
            QKV projection weight tensor in HBM.
            Dtype: nl.float8_e4m3fn_x4
            Shape: [H_packed, I], can be viewed as [H0, H1_packed, I].

        weight_scales_hbm (nl.ndarray):
            QKV weight quantization scales for MXFP in HBM.
            dtype: uint8
            Shape: [H // 32, I] == [H_packed // 8, I]
            Note: Since weights_qtz_hbm is already quantized, weight scales are 8x times smaller, not 32x.

        cfg (QKV_TKG_MXFP_Config): Kernel configuration (used for sharding info and dtype).

        bias_hbm (Optional[nl.ndarray]): [1, I], Optional bias tensor on HBM.

    Returns:
        output_hbm (nl.ndarray): [BxS, I], QKV projection output in HBM.
    """

    # Get sharding info from cfg
    num_shards = cfg.num_shards
    shard_id = cfg.shard_id
    hidden_orig_dtype = cfg.hidden_orig_dtype

    # We are deriving dimensions directly from already quantized tensors, hence shapes are 4x smaller already.
    H0, H1_packed_shard, BxS = hidden_qtz_sb.shape
    H_packed_shard = H0 * H1_packed_shard
    H_packed, I = weights_qtz_hbm.shape
    # hidden_qtz comes in LNC2 sharded, weights do not. We shard weights at load time with shard_id.
    # This is because we load weights with strided dma_copy pattern which cannot be done on pre-sliced tensor.

    weight_qtz_dtype = weights_qtz_hbm.dtype
    # Weight and weight scales asserts are done at the top-level of the kernel.
    kernel_assert(
        hidden_qtz_sb.dtype == nl.float8_e4m3fn_x4,
        f"[QKV TKG MXFP Kernel] _qkv_tkg_projection_mxfp(...) function expects hidden_qtz_sb.dtype == nl.float8_e4m3fn_x4, but got {hidden_qtz_sb.dtype}.",
    )
    kernel_assert(
        hidden_scales_sb.dtype == nl.uint8,
        f"[QKV TKG MXFP Kernel] _qkv_tkg_projection_mxfp(...) function expects hidden_scales_sb.dtype == nl.uint8, but got {hidden_scales_sb.dtype}.",
    )

    # Calculate MXFP tiling constants
    NUM_WEIGHT_LOAD_BLOCKS = math.ceil(H_packed_shard / WEIGHT_LOAD_BLOCK_SIZE)
    # NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK is the same constant as in non-MXFP kernel, however we are
    # processing 4x tiles per WEIGHT_LOAD_BLOCK. WEIGHT_LOAD_BLOCK is number of rows of H we
    # load/process at once, e.g., load only [2048, I] and do the compute.
    NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK = math.ceil(WEIGHT_LOAD_BLOCK_SIZE / P_MAX)
    NUM_I_BLOCKS = math.ceil(I / I_BLOCK_SIZE)

    # Allocate output in HBM
    output_hbm = nl.ndarray((BxS, I), dtype=hidden_orig_dtype, buffer=nl.shared_hbm)

    if cfg.add_bias:
        # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg)

    # QKV Projection loop - process each I_BLOCK_SIZE=4096 block (independent columns accumulated separately)
    for i_block in TiledRange(I, I_BLOCK_SIZE):
        # Allocate qkv_out_sb to store results of current I_BLOCK chunk
        qkv_out_sb = nl.ndarray((BxS, i_block.size), dtype=hidden_orig_dtype, buffer=nl.sbuf)

        """
        SBUF can run out-of-space in unallocated kernels. Add "list block" dimensions here,
        and/or make the kernel allocated. Also for now, SBUF space is approximate, use hardware
        specific SBUF space. None of this should be necessary in unallocated kernels, but it
        seems to be going out-of-space.
        """

        # Allocate weight buffer with double-buffering to overlap DMA with compute (if space permits)
        weight_tile_size = H0 * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK * i_block.size * sizeinbytes(weight_qtz_dtype)
        weight_and_weight_scales_tile_size = 2 * weight_tile_size
        # Note: Quick heuristic for now, to be improved later. Since this is unallocated kernel we cannot know exact space taken.
        WEIGHT_MULTI_BUFFER_THERSHOLD_HERUISTIC = 16 * 1024 * 1024
        NUM_W_BUFFERS = 2 if weight_and_weight_scales_tile_size * 2 <= WEIGHT_MULTI_BUFFER_THERSHOLD_HERUISTIC else 1

        """
        If WEIGHT_LOAD_BLOCK_SIZE were maximum of H_packed, then weight_qtz_sb shape would be:
        (H0, NUM_W_BUFFERS, H1_packed = H // 512, i_block_sz).
        We use fixed block size, but since H in weights is packed, we process fewer H_BLOCKS
        than in non-quant version.
        """
        weights_qtz_sb = nl.ndarray(
            (H0, NUM_W_BUFFERS, NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK, i_block.size),
            dtype=nl.float8_e4m3fn_x4,
            buffer=nl.sbuf,
        )

        weight_scales_sb = nl.ndarray(
            (H0, NUM_W_BUFFERS, NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK, i_block.size),
            dtype=nl.uint8,
            buffer=nl.sbuf,
        )

        # Allocate PSUM banks for accumulation (num_i_tiles_per_i_block <= 8)
        mm_result_psum = []
        for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
            psum_tile = nl.ndarray(
                (P_MAX, i_tile.size),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            mm_result_psum.append(psum_tile)

        # Process WEIGHT_LOAD_BLOCK_SIZE at a time, e.g., load [WEIGHT_LOAD_BLOCK_SIZE, I] and do the compute.
        for h_weight_block in TiledRange(H_packed_shard, WEIGHT_LOAD_BLOCK_SIZE):
            # TODO: Remove this once below loops are changed to TiledRange.
            num_128_tiles_in_current_weight_load_block = math.ceil(h_weight_block.size / P_MAX)

            # Buffer index for double-buffering (cycles 0, 1, 0, 1, ...)
            weight_buffer_idx = h_weight_block.index % NUM_W_BUFFERS

            """
            Load weight tile with single DMA using access patterns.
            Contiguous H sharding: shard 0 gets H[0:H/2], shard 1 gets H[H/2:H].
            HBM source pattern: weights_qtz_hbm is [H_packed, I].
            Offset into this shard's contiguous H portion.
            """
            h_shard_base_offset = shard_id * H_packed_shard * I
            weights_hbm_load_pattern = [
                [I, P_MAX],
                [P_MAX * I, num_128_tiles_in_current_weight_load_block],
                [1, i_block.size],
            ]
            weights_hbm_load_offset = h_shard_base_offset + h_weight_block.index * I + i_block.start_offset

            # SBUF dest pattern: weights_qtz_sb is [H0, NUM_W_BUFFERS, NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK, i_block_sz]
            weights_sb_load_pattern = [
                [NUM_W_BUFFERS * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK * i_block.size, P_MAX],
                [i_block.size, num_128_tiles_in_current_weight_load_block],
                [1, i_block.size],
            ]
            weights_sb_load_offset = weight_buffer_idx * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK * i_block.size

            # Load weights
            nisa.dma_copy(
                dst=weights_qtz_sb.ap(pattern=weights_sb_load_pattern, offset=weights_sb_load_offset),
                src=weights_qtz_hbm.ap(pattern=weights_hbm_load_pattern, offset=weights_hbm_load_offset),
            )

            """
            Load packed weight scales.
            MX Scale layout constants:
            Quadrant placement: HBM has 16 contiguous scale rows per H512 tile.
            SBUF needs them spread across 4 quadrants (4 rows each at offsets 0, 32, 64, 96).
            """
            SCALE_GROUP_SIZE = 32  # One scale per 32 H elements
            SCALES_PER_H512_TILE = 512 // SCALE_GROUP_SIZE  # = 16 scale rows per H512 tile in HBM
            SBUF_QUADRANT_SIZE = 32
            NUM_QUADRANTS = H0 // SBUF_QUADRANT_SIZE  # = 4
            SCALES_PER_QUADRANT = SCALES_PER_H512_TILE // NUM_QUADRANTS  # = 4

            # weight_scales_hbm shape: [H // 32, I] = [H_packed // 8, I]
            # Contiguous sharding: shard's scale rows start at shard_id * (H_packed_shard // 8)
            H_packed_shard_scale_rows = H_packed_shard // 8  # Number of scale rows for this shard
            scale_shard_base_offset = shard_id * H_packed_shard_scale_rows * I

            # num_128_tiles_in_current_weight_load_block iterations
            # for h_tile_idx_in_block in range(num_128_tiles_in_current_weight_load_block):
            for h_tile_in_block in TiledRange(h_weight_block.size, P_MAX):
                h_tile_idx_in_h = h_weight_block.index * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK + h_tile_in_block.index

                for quad_idx in range(NUM_QUADRANTS):
                    # Local offset within this shard's scale rows
                    local_hbm_row_offset = (
                        h_tile_idx_in_h * SCALES_PER_H512_TILE + quad_idx * SCALES_PER_QUADRANT
                    ) * I + i_block.start_offset
                    hbm_row_offset = scale_shard_base_offset + local_hbm_row_offset

                    nisa.dma_copy(
                        dst=weight_scales_sb[
                            quad_idx * SBUF_QUADRANT_SIZE : quad_idx * SBUF_QUADRANT_SIZE + SCALES_PER_QUADRANT,
                            weight_buffer_idx,
                            h_tile_in_block.index,
                            : i_block.size,
                        ],
                        src=weight_scales_hbm.ap(
                            pattern=[[I, SCALES_PER_QUADRANT], [1, i_block.size]], offset=hbm_row_offset, dtype=nl.uint8
                        ),
                    )

            # Matmul for current weight load
            # num_128_tiles_in_current_weight_load_block iterations
            for h_tile_in_block in TiledRange(h_weight_block.size, P_MAX):
                h_tile_idx_in_h = h_weight_block.index * NUM_128_TILES_PER_WEIGHT_LOAD_BLOCK + h_tile_in_block.index

                for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
                    nisa.nc_matmul_mx(
                        dst=mm_result_psum[i_tile.index][0:BxS, 0 : i_tile.size],
                        stationary=hidden_qtz_sb[0:H0, h_tile_idx_in_h, 0:BxS],
                        moving=weights_qtz_sb[
                            0:H0,
                            weight_buffer_idx,
                            h_tile_in_block.index,
                            i_tile.start_offset : i_tile.start_offset + i_tile.size,
                        ],
                        stationary_scale=hidden_scales_sb[0:H0, h_tile_idx_in_h, 0:BxS],
                        moving_scale=weight_scales_sb[
                            0:H0,
                            weight_buffer_idx,
                            h_tile_in_block.index,
                            i_tile.start_offset : i_tile.start_offset + i_tile.size,
                        ],
                    )

        # Copy PSUM results to SBUF.
        for i_tile in TiledRange(i_block.size, I_TILE_SIZE):
            # So we don't double-add bias.
            if cfg.add_bias and cfg.shard_id == 0:
                nisa.tensor_tensor(
                    dst=qkv_out_sb[:BxS, i_tile.start_offset : i_tile.start_offset + i_tile.size],
                    data1=mm_result_psum[i_tile.index][:BxS, 0 : i_tile.size],
                    data2=bias_sb[
                        :BxS,
                        i_block.start_offset + i_tile.start_offset : i_block.start_offset
                        + i_tile.start_offset
                        + i_tile.size,
                    ],
                    op=nl.add,
                )
            else:
                nisa.tensor_copy(
                    dst=qkv_out_sb[:BxS, i_tile.start_offset : i_tile.start_offset + i_tile.size],
                    src=mm_result_psum[i_tile.index][:BxS, 0 : i_tile.size],
                )

        # Cross-core reduction via sendrecv when LNC > 1
        # Each core has partial sum from its H shard, need to sum across cores
        if num_shards > 1:
            qkv_recv_sb = nl.ndarray((BxS, i_block.size), dtype=hidden_orig_dtype, buffer=nl.sbuf)
            other_core = 1 - shard_id
            nisa.sendrecv(
                src=qkv_out_sb,
                dst=qkv_recv_sb,
                send_to_rank=other_core,
                recv_from_rank=other_core,
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=qkv_out_sb, data1=qkv_out_sb, data2=qkv_recv_sb, op=nl.add)

        # Store to HBM
        nisa.dma_copy(
            dst=output_hbm[:BxS, i_block.start_offset : i_block.start_offset + i_block.size],
            src=qkv_out_sb[:BxS, 0 : i_block.size],
        )

    return output_hbm


def _load_and_broadcast_bias(
    bias_hbm: nl.ndarray,
    cfg: QKV_TKG_MXFP_Config,
) -> nl.ndarray:
    """
    Load bias and broadcast to partition dimension.

    Loads bias with shape [1, I] to SBUF and broadcasts it to [nl.tile_size.pmax, I]
    using stream_shuffle.

    Args:
        bias_hbm (nl.ndarray): [1, I], Bias tensor in HBM.
        cfg (QKV_TKG_MXFP_Config): Kernel configuration.

    Returns:
        nl.ndarray: [nl.tile_size.pmax, I], Broadcasted bias tensor in SBUF.
    """
    _, I = bias_hbm.shape

    # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
    bias_sb = nl.ndarray((nl.tile_size.pmax, I), dtype=cfg.hidden_orig_dtype, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=bias_sb[0:1, 0:I],
        src=bias_hbm[0:1, 0:I],
    )

    # Stream Shuffle works on 32 partitions only, apply it nl.tile_size.pmax // 32 = 4 times.
    MAX_STREAM_SHUFFLE_PARTITIONS = 32
    NUM_BROADCASTS = nl.tile_size.pmax // MAX_STREAM_SHUFFLE_PARTITIONS
    for broadcast_idx in nl.affine_range(NUM_BROADCASTS):
        nisa.nc_stream_shuffle(
            dst=bias_sb[
                nl.ds(broadcast_idx * MAX_STREAM_SHUFFLE_PARTITIONS, MAX_STREAM_SHUFFLE_PARTITIONS),
                0:I,
            ],
            src=bias_sb[0:1, 0:I],
            shuffle_mask=[0] * MAX_STREAM_SHUFFLE_PARTITIONS,
        )
    return bias_sb
