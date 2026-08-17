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
This kernel implements QKV (Query, Key, Value) projection optimized for Context Encoding (CTE) with support for multiple fused operations including normalization, residual addition, bias, and RoPE.
"""

# Standard Library
import math
from typing import Callable, Dict, List, Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ..utils.allocator import SbufManager, sizeinbytes

# NKI Library
from ..utils.common_types import (
    DtypeMode,
    NormType,
    QKNormConfig,
    QKVOutputLayout,
    QKVWeightLayout,
    QuantizationType,
    StridedInputConfig,
)
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..utils.logging import get_logger
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast

# QKV CTE
from .qkv_cte_utils import (
    ROW_MX_TAIL_SCALE_BYTES,
    QKV_CTE_Config,
    QKV_CTE_Dims,
    QKV_CTE_UserInput,
    _build_config,
    _get_tensor_dimensions,
    _mx_partition_splits,
    _validate_strided_input_config,
    _validate_user_inputs,
)

# HARDWARE CONSTANTS
NUM_HW_PSUM_BANKS = 8
MAX_STREAM_SHUFFLE_PARTITIONS = 32
NUM_MX_WEIGHT_BUFFERS = 2
MX_NEUTRAL_SCALE = 127  # MX scale exponent bias: 2^(127-127) = 2^0 = 1.0 (no scaling)
NUM_QKV_SEGMENTS = 3  # Q, K, V


def _get_psum_bank_size() -> int:
    """
    Calculate PSUM bank size in bytes.

    Returns:
        int: Size of a single PSUM bank in bytes
    """
    return sizeinbytes(nl.float32) * nl.tile_size.psum_fmax


# FP8 QUANTIZATION CONSTANT
def _get_fp8_e4m3_max_pos_val() -> float:
    """
    Get the maximum positive value for FP8 E4M3 quantization based on hardware version.

    Returns:
        448.0 for trn3 (gen4+), 240.0 for earlier hardware (trn1/trn2)

    TODO: switch to nki.isa.get_nc_version() >= nki.isa.nc_version.gen4 after the comparison support
    """
    return 448.0 if nki.isa.get_nc_version() == nki.isa.nc_version.gen4 else 240.0


def _scalar_activation_bias_supported() -> bool:
    """nisa.activation supports scalar bias on NeuronCore gen4 (Trn3) and later.

    gen3 (Trn2) also supports scalar bias in the ISA, but using it changes the SBUF
    address layout and triggers a compiler sensitivity that causes accuracy regressions.
    """
    return nki.isa.get_nc_version() >= nki.isa.nc_version.gen4


def _make_activation_constant_bias(value: float, sbm: SbufManager, pmax: int, dtype=nl.float32):
    """Create a constant bias for nisa.activation, broadcast across all partitions."""
    if _scalar_activation_bias_supported():
        return value
    buf = sbm.alloc_stack((pmax, 1), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=buf, value=value)
    return buf


def _slice_activation_constant_bias(bias, s_tile_sz: int):
    """Slice a constant activation bias to [s_tile_sz, 1] if tensor, or pass through if scalar."""
    if isinstance(bias, (int, float)):
        return bias
    return bias[0:s_tile_sz, 0:1]


class QKNormBuffers(nl.NKIObject):
    """SBUF working buffers for per-head QK-norm, shared by MX and non-MX paths."""

    pre_rope_q_gamma_sb: Optional[nl.NkiTensor] = None
    pre_rope_k_gamma_sb: Optional[nl.NkiTensor] = None
    post_rope_q_gamma_sb: Optional[nl.NkiTensor] = None
    post_rope_k_gamma_sb: Optional[nl.NkiTensor] = None
    pre_rope_eps_sb = None
    post_rope_eps_sb = None
    scratch_sb: Optional[list] = None
    zero_bias_sb: Optional[nl.NkiTensor] = None


def _alloc_qk_norm_position_buffers(bufs, prefix, qk_norm_cfg, sbm, pmax, d_head, dtype):
    """Allocate eps and gamma SBUF buffers for a single QK-norm position (pre or post RoPE).

    Sets bufs.{prefix}_eps_sb, bufs.{prefix}_q_gamma_sb, bufs.{prefix}_k_gamma_sb.
    """
    eps_sb = _make_activation_constant_bias(qk_norm_cfg.eps, sbm, pmax)
    q_gamma_sb = None
    k_gamma_sb = None
    if qk_norm_cfg.q_gamma_norm_weights is not None:
        q_gamma_sb = sbm.alloc_stack((pmax, d_head), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_gamma_sb[0:1, :], src=qk_norm_cfg.q_gamma_norm_weights)
        stream_shuffle_broadcast(q_gamma_sb[0:1, :], q_gamma_sb)
    if qk_norm_cfg.k_gamma_norm_weights is not None:
        k_gamma_sb = sbm.alloc_stack((pmax, d_head), dtype=dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_gamma_sb[0:1, :], src=qk_norm_cfg.k_gamma_norm_weights)
        stream_shuffle_broadcast(k_gamma_sb[0:1, :], k_gamma_sb)
    if prefix == "pre_rope":
        bufs.pre_rope_eps_sb = eps_sb
        bufs.pre_rope_q_gamma_sb = q_gamma_sb
        bufs.pre_rope_k_gamma_sb = k_gamma_sb
    else:
        bufs.post_rope_eps_sb = eps_sb
        bufs.post_rope_q_gamma_sb = q_gamma_sb
        bufs.post_rope_k_gamma_sb = k_gamma_sb


def _allocate_qk_norm_buffers(cfg, dims, sbm, pmax):
    """Allocate SBUF buffers for QK-norm. Returns None if QK-norm is disabled."""
    if cfg.qk_norm_pre_rope is None and cfg.qk_norm_post_rope is None:
        return None

    bufs = QKNormBuffers()
    bufs.scratch_sb = [
        sbm.alloc_stack((pmax, 1), dtype=nl.float32, buffer=nl.sbuf),
        sbm.alloc_stack((pmax, 1), dtype=nl.float32, buffer=nl.sbuf),
        sbm.alloc_stack((pmax, 1), dtype=nl.float32, buffer=nl.sbuf),
    ]
    bufs.zero_bias_sb = sbm.alloc_stack((pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=bufs.zero_bias_sb, value=0)

    if cfg.qk_norm_pre_rope is not None:
        _alloc_qk_norm_position_buffers(
            bufs, "pre_rope", cfg.qk_norm_pre_rope, sbm, pmax, dims.d_head, cfg.compute_mm_dtype
        )
    if cfg.qk_norm_post_rope is not None:
        _alloc_qk_norm_position_buffers(
            bufs, "post_rope", cfg.qk_norm_post_rope, sbm, pmax, dims.d_head, cfg.compute_mm_dtype
        )
    return bufs


class RoPEBuffers(nl.NKIObject):
    """SBUF/HBM buffers for fused RoPE, shared by MX and non-MX paths."""

    cos_sb: Optional[List[nl.NkiTensor]] = None
    sin_sb: Optional[List[nl.NkiTensor]] = None
    scratch_sb: Optional[List[list]] = None
    cos_hbm: Optional[nl.NkiTensor] = None
    sin_hbm: Optional[nl.NkiTensor] = None
    # K-specific fused caches (only when gamma_fused_in_rope_caches=True)
    k_cos_hbm: Optional[nl.NkiTensor] = None
    k_sin_hbm: Optional[nl.NkiTensor] = None
    k_cos_sb: Optional[List[nl.NkiTensor]] = None
    k_sin_sb: Optional[List[nl.NkiTensor]] = None


def _get_num_rope_scratch_bufs(has_qk_norm: bool, H: int) -> int:
    """Number of rope scratch buffers per S-tile.

    Triple-buffered when QK-norm is active to break carried dependency across
    heads. Double-buffered otherwise, falling back to single buffer at large H
    to avoid reducing multi-buffering degree.
    """
    if has_qk_norm:
        return 3
    MAX_H_FOR_DOUBLE_BUFFER = 8192  # Llama3 70B hidden dim
    if H > MAX_H_FOR_DOUBLE_BUFFER:
        return 1
    return 2


def _allocate_rope_buffers(
    cfg,
    dims,
    sbm,
    pmax,
    num_s_tiles,
    has_qk_norm,
    cos_cache_hbm,
    sin_cache_hbm,
    k_cos_cache_hbm=None,
    k_sin_cache_hbm=None,
):
    """Allocate SBUF buffers for fused RoPE. Returns None if fused_rope is disabled."""
    if not cfg.fused_rope:
        return None

    _gamma_fused_in_rope = cfg.qk_norm_pre_rope is not None and cfg.qk_norm_pre_rope.gamma_fused_in_rope_caches
    # When gamma_fused_in_rope_caches: sin cache is [B, S, d_head] packed
    # (sin_lo in [0:d_half], sin_hi in [d_half:d_head])
    _sin_buf_fdim = dims.d_head if _gamma_fused_in_rope else dims.d_head // 2
    _has_separate_k_caches = _gamma_fused_in_rope and k_cos_cache_hbm is not None

    bufs = RoPEBuffers()
    bufs.cos_hbm = cos_cache_hbm
    bufs.sin_hbm = sin_cache_hbm
    bufs.k_cos_hbm = k_cos_cache_hbm if _has_separate_k_caches else None
    bufs.k_sin_hbm = k_sin_cache_hbm if _has_separate_k_caches else None
    bufs.cos_sb = []
    bufs.sin_sb = []
    bufs.scratch_sb = []
    bufs.k_cos_sb = [] if _has_separate_k_caches else None
    bufs.k_sin_sb = [] if _has_separate_k_caches else None
    _num_rope_bufs = _get_num_rope_scratch_bufs(has_qk_norm, dims.H)
    for _ in range(num_s_tiles):
        bufs.cos_sb.append(sbm.alloc_stack((pmax, dims.d_head), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))
        bufs.sin_sb.append(sbm.alloc_stack((pmax, _sin_buf_fdim), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))
        if _has_separate_k_caches:
            bufs.k_cos_sb.append(sbm.alloc_stack((pmax, dims.d_head), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))
            bufs.k_sin_sb.append(sbm.alloc_stack((pmax, _sin_buf_fdim), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))
        scratch = []
        for _ in range(_num_rope_bufs):
            scratch.append(sbm.alloc_stack((pmax, dims.d_head * 2), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))
        bufs.scratch_sb.append(scratch)
    return bufs


def qkv_cte(
    input: nl.NkiTensor,
    fused_qkv_weights: nl.NkiTensor,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    # -- Bias
    bias: Optional[nl.NkiTensor] = None,
    # -- Fused Residual Add
    fused_residual_add: Optional[bool] = False,
    mlp_prev: Optional[nl.NkiTensor] = None,
    attention_prev: Optional[nl.NkiTensor] = None,
    # --- Fused Norm Related
    fused_norm_type: NormType = NormType.NO_NORM,
    gamma_norm_weights: Optional[nl.NkiTensor] = None,
    layer_norm_bias: Optional[nl.NkiTensor] = None,
    norm_eps: Optional[float] = 1e-6,
    hidden_actual: Optional[int] = None,
    # --- Fused RoPE Related
    fused_rope: Optional[bool] = False,
    cos_cache: Optional[nl.NkiTensor] = None,
    sin_cache: Optional[nl.NkiTensor] = None,
    # Fused RoPE + QK-norm: K-specific gamma-fused caches
    k_cos_cache: Optional[nl.NkiTensor] = None,
    k_sin_cache: Optional[nl.NkiTensor] = None,
    d_head: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    # --- KV Cache Related
    k_cache: Optional[nl.NkiTensor] = None,
    v_cache: Optional[nl.NkiTensor] = None,
    k_scale: Optional[nl.NkiTensor] = None,
    v_scale: Optional[nl.NkiTensor] = None,
    fp8_max: Optional[float] = None,
    fp8_min: Optional[float] = None,
    kv_dtype: Optional[type] = None,
    # --- Block KV Cache Related
    use_block_kv: bool = False,
    transpose_k_cache: bool = False,
    fp8_packed: bool = False,
    block_size: Optional[int] = None,
    slot_mapping: Optional[nl.NkiTensor] = None,
    # -----------------------------------------
    store_output_in_sbuf: bool = False,
    # -----------------------------------------
    # User can optionally PASS Sbuf manager
    # -----------------------------------------
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    # ----------------------------------------
    load_input_with_DMA_transpose: bool = True,
    # --- Quantization Related
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[nl.NkiTensor] = None,
    qkv_in_scale: Optional[nl.NkiTensor] = None,
    # ----------------------------------------
    is_input_swizzled: bool = False,
    weight_layout: QKVWeightLayout = QKVWeightLayout.CONTIGUOUS,
    # --- QK-Norm Related
    qk_norm_pre_rope: Optional[QKNormConfig] = None,
    qk_norm_post_rope: Optional[QKNormConfig] = None,
    # --- Output
    output_hbm: Optional[nl.NkiTensor] = None,
    # --- Optional per-segment sum-of-squares outputs (each [B, S, 1]).
    # When provided, the kernel computes sum(x**2) over the head dimension for
    # the Q/K/V segment inline while the QKV tile is resident in SBUF (no HBM
    # reload). Supported for the BSD and NBSd output layouts. Default None -> the
    # existing path is byte-for-byte unchanged.
    q_squared_sum_out: Optional[nl.NkiTensor] = None,
    k_squared_sum_out: Optional[nl.NkiTensor] = None,
    v_squared_sum_out: Optional[nl.NkiTensor] = None,
    # --- Strided Input
    strided_input_config: Optional[StridedInputConfig] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> nl.NkiTensor:
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

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        I: Fused QKV dimension = (num_q_heads + 2*num_kv_heads) * d_head
        d_head: Dimension per attention head
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        num_heads: Total number of heads = num_q_heads + 2*num_kv_heads

    Args:
        input (nl.NkiTensor): [B, S, H], Input hidden states tensor where B=batch, S=sequence_length, H=hidden_dim.
            We name it 'input' and not 'hidden' to avoid ambiguity with the size of "hidden dimension".
        fused_qkv_weights (nl.NkiTensor): [H, I] for non-MX, or [H//4, I, 4] fp8 for MX/ROW_MX. Fused QKV weight matrix where I=fused_qkv_dim=(num_q_heads + 2*num_kv_heads)*d_head
        output_layout (QKVOutputLayout): Output tensor layout: QKVOutputLayout.BSD=[B, S, I] or QKVOutputLayout.NBSd=[num_heads, B, S, d_head]. Default: QKVOutputLayout.BSD
        bias (Optional[nl.NkiTensor]): [1, I], Bias tensor to add to QKV projection output. Default: None
        fused_residual_add (Optional[bool]): Whether to perform residual addition: input = input + mlp_prev + attention_prev. Default: False
        mlp_prev (Optional[nl.NkiTensor]): [B, S, H], Previous MLP output tensor for residual addition. Default: None
        attention_prev (Optional[nl.NkiTensor]): [B, S, H], Previous attention output tensor for residual addition. Default: None
        fused_norm_type (NormType): Type of normalization: NormType.NO_NORM, NormType.RMS_NORM, NormType.RMS_NORM_SKIP_GAMMA, or NormType.LAYER_NORM.
            NormType.RMS_NORM_SKIP_GAMMA assumes fused_qkv_weights have been pre-multiplied with gamma vector, so its skipped here. Default: NormType.NO_NORM
        gamma_norm_weights (Optional[nl.NkiTensor]): [1, H], Normalization gamma/scale weights (required for NormType.RMS_NORM and NormType.LAYER_NORM). Default: None
        layer_norm_bias (Optional[nl.NkiTensor]): [1, H], Layer normalization beta/bias weights (only for NormType.LAYER_NORM). Using layer norm bias is optional. Default: None
        norm_eps (Optional[float]): Epsilon value for numerical stability in normalization. Default: 1e-6
        hidden_actual (Optional[int]): Actual hidden dimension for padded tensors (if H contains padding). Default: None
        fused_rope (Optional[bool]): Whether to apply RoPE rotation to Query and Key heads after QKV projection. Default: False
        cos_cache (Optional[nl.NkiTensor]): [B, S, d_head], Cosine cache for RoPE (required if fused_rope=True). Default: None
        sin_cache (Optional[nl.NkiTensor]): [B, S, d_head], Sine cache for RoPE (required if fused_rope=True). Default: None
        d_head (Optional[int]): Dimension per attention head (required for QKVOutputLayout.NBSd and RoPE). Default: None
        num_q_heads (Optional[int]): Number of query heads (required for RoPE). Default: None
        num_kv_heads (Optional[int]): Number of key/value heads (required for RoPE). Default: None
        transpose_k_cache (bool): Whether to store K in transposed layout [num_blocks*num_kv_heads, d_head, block_size] in the block KV cache
            or [B, kv_dim, max_seq_len] for flat KV cache. Default: False
        fp8_packed (bool): Enable packed FP8 K cache layout for block KV. Packs 2 consecutive FP8 sequence
            positions into one row: k_cache shape [num_blocks, num_kv_heads, block_size // 2, d_head, 2] fp8, where
            dim 4 index 0 = even positions and index 1 = odd positions. Enables DMA transpose on decode.
            Requires block KV, FP8 quantization, even block_size, and d_head <= 128.
            Mutually exclusive with transpose_k_cache. Default: False
        store_output_in_sbuf (bool): Whether to store output in SBUF (currently unsupported, must be False). Default: False
        sbm (Optional[SbufManager]): Optional SBUF manager for memory allocation control, with pre-specified bounds for SBUF usage.
            If sbm is not provided, kernel will by default be allocated and use all of the available SBUF space. Default: None
        use_auto_allocation (bool): Whether to use automatic SBUF allocation, by default kernel is manually allocated and it creates its own SbufManager.
            If 'sbm' is provided by user, user has the responsibility to set use_auto_allocation=True in the provided SbufManager. Default: False
        load_input_with_DMA_transpose (bool): Whether to use DMA transpose optimization. Default: True
        quantization_type: QuantizationType, default=QuantizationType.NONE
        qkv_w_scale: Optional[nl.NkiTensor], default=None The weight quantization scale for qkv projection,
            Shape: [H//32, I] for MX
        qkv_in_scale: Optional[nl.NkiTensor], default=None The input quantization scale for qkv projection, currently assume the input quantization scales are the scale for q, k, v projections
        is_input_swizzled: bool, default=False
            Whether the input tensor is swizzled (only applicable with MX Quantization).
            If not swizzled, input has shape [B, S, H].
            If swizzled, input has shape [B, S, H] but is preswizzled from
            [B, S, H//512, 128, 4] -> [B, S, 4, H//512, 128] and flattened to [B, S, H].
        weight_layout (QKVWeightLayout): Layout of fused_qkv_weights. See QKVWeightLayout
            docstring for packing instructions. Default: QKVWeightLayout.CONTIGUOUS
        dtype_mode (DtypeMode): Quantization dtype policy for STATIC/ROW
            weight tiles. Also used for the FP8 KV cache when ``kv_dtype`` is
            the opaque ``"float8e4"`` sentinel; a concrete ``kv_dtype`` is
            honored as-is.
            - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
            - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only.
            - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, else ``nl.float8_e4m3``.

    Returns:
        output (nl.NkiTensor): QKV projection output tensor:
            - If output_layout=QKVOutputLayout.BSD: shape [B, S, I]
            - If output_layout=QKVOutputLayout.NBSd: shape [num_heads, B, S, d_head]

    Notes:
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
        - mxfp8/int32 weights for MX quantization

    Pseudocode:
        # Step 1: Optional fused residual addition
        if fused_residual_add:
            x = input + mlp_prev + attention_prev
        else:
            x = input

        # Step 2: Optional normalization
        if fused_norm_type == RMS_NORM:
            x = x * gamma / sqrt(mean(x^2) + eps)
        elif fused_norm_type == LAYER_NORM:
            x = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta

        # Step 3: QKV projection with optional bias
        qkv = x @ fused_qkv_weights
        if bias is not None:
            qkv = qkv + bias

        # Step 4: Optional RoPE rotation on Q and K heads
        if fused_rope:
            Q, K, V = split_qkv(qkv, num_q_heads, num_kv_heads, d_head)
            Q = apply_rope(Q, cos_cache, sin_cache)
            K = apply_rope(K, cos_cache, sin_cache)
            qkv = concat(Q, K, V)

        # Step 5: Reshape output based on layout
        if output_layout == NBSd:
            output = reshape(qkv, [num_heads, B, S, d_head])
        else:  # BSD
            output = qkv  # [B, S, I]

        return output
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
        # KV Cache
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_max=fp8_max,
        fp8_min=fp8_min,
        kv_dtype=kv_dtype,
        # Block KV Cache
        use_block_kv=use_block_kv,
        transpose_k_cache=transpose_k_cache,
        fp8_packed=fp8_packed,
        block_size=block_size,
        slot_mapping=slot_mapping,
        # Performance
        store_output_in_sbuf=store_output_in_sbuf,
        sbm=sbm,
        use_auto_allocation=use_auto_allocation,
        load_input_with_DMA_transpose=load_input_with_DMA_transpose,
        quantization_type=quantization_type,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        is_input_swizzled=is_input_swizzled,
        weight_layout=weight_layout,
        qk_norm_pre_rope=qk_norm_pre_rope,
        qk_norm_post_rope=qk_norm_post_rope,
        strided_input_config=strided_input_config,
        output_hbm=output_hbm,
        q_squared_sum_out=q_squared_sum_out,
        k_squared_sum_out=k_squared_sum_out,
        v_squared_sum_out=v_squared_sum_out,
        dtype_mode=dtype_mode,
    )

    _validate_user_inputs(args=user_inputs)
    # Build 'cfg' object, to store kernel configuration.
    cfg = _build_config(args=user_inputs)
    # Build 'dims' object to store tensor dimensions used throughout the kernel.
    dims = _get_tensor_dimensions(args=user_inputs, cfg=cfg)

    # Strided input: validate S-shard alignment (requires dims).
    if cfg.strided_input_config is not None:
        _validate_strided_input_config(user_inputs, dims=dims)

    # HBM tensor name prefix: use sbm prefix when provided to avoid name collisions.
    _hbm_name_prefix = (sbm.get_name_prefix() + "_") if sbm is not None and sbm.get_name_prefix() else ""

    # Create output tensor with original dimensions
    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
        # Keep the output in bf16 when using static quantization
        output_dtype = nl.bfloat16
    elif input.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]:
        # FP8 input produces bf16 output (matmul accumulates in f32, stores as bf16)
        output_dtype = nl.bfloat16
    else:
        output_dtype = input.dtype

    # Output S dimension matches kernel's processing S (= num_local_tokens for strided, S_orig otherwise).
    if cfg.use_kv_cache:
        # Output Q tensor: S matches kernel's processing S (= num_local_tokens for strided).
        # When q_dim == 0 (KV-only projection), skip allocation — no Q output.
        if dims.q_dim > 0:
            q_S = cfg.strided_input_config.num_local_tokens if cfg.strided_input_config != None else dims.S_orig
            q_tensor_hbm = nl.ndarray(
                (dims.B_orig, q_S, dims.q_dim),
                dtype=input.dtype,
                buffer=nl.shared_hbm,
                name=f"{_hbm_name_prefix}qkv_cte_output_hbm",
            )
        else:
            q_tensor_hbm = None
        output_hbm = None
    elif output_hbm is not None:
        # Caller provided output tensor — skip allocation.
        q_tensor_hbm = None
    elif cfg.output_layout == QKVOutputLayout.BSD:
        output_hbm = nl.ndarray(
            (dims.B_orig, dims.S_orig, dims.I),
            dtype=output_dtype,
            buffer=nl.shared_hbm,
            name=f"{_hbm_name_prefix}qkv_cte_output_hbm",
        )
        q_tensor_hbm = None
    else:  # QKVOutputLayout.NBSd
        output_hbm = nl.ndarray(
            (dims.num_heads, dims.B_orig, dims.S_orig, dims.d_head),
            dtype=output_dtype,
            buffer=nl.shared_hbm,
            name=f"{_hbm_name_prefix}qkv_cte_output_hbm",
        )
        q_tensor_hbm = None

    # Potentially reshape B,S to BxS, for performance benefits.
    if cfg.use_BxS_input_reshape:
        input = input.reshape((1, dims.BxS, input.shape[-1]))
        if cfg.fused_residual_add:
            mlp_prev = mlp_prev.reshape((1, dims.BxS, dims.H))
            attention_prev = attention_prev.reshape((1, dims.BxS, dims.H))
        if cfg.fused_rope:
            cos_cache = cos_cache.reshape((1, dims.BxS, dims.d_head))
            sin_cache = sin_cache.reshape((1, dims.BxS, dims.d_head))
        if cfg.use_kv_cache:
            if q_tensor_hbm is not None:
                q_tensor_hbm = q_tensor_hbm.reshape((1, dims.BxS, dims.q_dim))
            if not use_block_kv:
                if transpose_k_cache:
                    # Cache layout: [B, kv_dim, max_seq_len] -> [1, kv_dim, max_seq_len]
                    k_cache = k_cache.reshape((1, dims.kv_dim, k_cache.shape[2]))
                else:
                    # Cache layout: [B, max_seq_len, kv_dim] -> [1, max_seq_len, kv_dim] (B folded)
                    k_cache = k_cache.reshape((1, k_cache.shape[1], dims.kv_dim))
                v_cache = v_cache.reshape((1, v_cache.shape[1], dims.kv_dim))
            if use_block_kv:
                slot_mapping = slot_mapping.reshape((1, dims.BxS))
        elif cfg.output_layout == QKVOutputLayout.BSD:
            output_hbm = output_hbm.reshape((1, dims.BxS, dims.I))
        elif cfg.output_layout == QKVOutputLayout.NBSd:
            output_hbm = output_hbm.reshape((dims.num_heads, 1, dims.BxS, dims.d_head))

    # Pass values and directly, and keep 'cfg' and 'dims'
    # object separate for clarity.
    if quantization_type.is_mx():
        _qkv_cte_mx_impl(
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
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            k_cos_cache_hbm=k_cos_cache,
            k_sin_cache_hbm=k_sin_cache,
        )
    else:
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
            q_tensor_hbm=q_tensor_hbm,
            k_cache_hbm=k_cache,
            v_cache_hbm=v_cache,
            k_scale_hbm=k_scale,
            v_scale_hbm=v_scale,
            slot_mapping_hbm=slot_mapping,
            qkv_in_scale=qkv_in_scale,
            qkv_w_scale=qkv_w_scale,
            q_squared_sum_hbm=q_squared_sum_out,
            k_squared_sum_hbm=k_squared_sum_out,
            v_squared_sum_hbm=v_squared_sum_out,
        )

    # Revert BxS to B,S as it is required by the user provided output_layout.
    if cfg.use_BxS_input_reshape:
        input = input.reshape((dims.B_orig, dims.S_orig, input.shape[-1]))
        if cfg.fused_residual_add:
            mlp_prev = mlp_prev.reshape((dims.B_orig, dims.S_orig, dims.H))
            attention_prev = attention_prev.reshape((dims.B_orig, dims.S_orig, dims.H))
        if cfg.fused_rope:
            cos_cache = cos_cache.reshape((dims.B_orig, dims.S_orig, dims.d_head))
            sin_cache = sin_cache.reshape((dims.B_orig, dims.S_orig, dims.d_head))
        if cfg.use_kv_cache:
            if q_tensor_hbm is not None:
                q_tensor_hbm = q_tensor_hbm.reshape((dims.B_orig, dims.S_orig, dims.q_dim))
            if not use_block_kv:
                if transpose_k_cache:
                    # Cache layout: [1, kv_dim, max_seq_len] -> [B, kv_dim, max_seq_len]
                    k_cache = k_cache.reshape((dims.B_orig, dims.kv_dim, k_cache.shape[2]))
                else:
                    # Cache layout: [1, max_seq_len, kv_dim] -> [B, max_seq_len, kv_dim]
                    k_cache = k_cache.reshape((dims.B_orig, k_cache.shape[1], dims.kv_dim))
                v_cache = v_cache.reshape((dims.B_orig, v_cache.shape[1], dims.kv_dim))
            if use_block_kv:
                slot_mapping = slot_mapping.reshape((dims.B_orig, dims.S_orig))
        elif cfg.output_layout == QKVOutputLayout.BSD:
            output_hbm = output_hbm.reshape((dims.B_orig, dims.S_orig, dims.I))
        elif cfg.output_layout == QKVOutputLayout.NBSd:
            output_hbm = output_hbm.reshape((dims.num_heads, dims.B_orig, dims.S_orig, dims.d_head))

    if cfg.use_kv_cache:
        return q_tensor_hbm, k_cache, v_cache

    # Barrier: ensure both NCs finish writing output_hbm before any consumer reads it.
    # This is required when qkv_cte is called by the attention_block_tkg kernel
    _, _n_prgs, _ = get_verified_program_sharding_info("qkv_cte_barrier", (0, 1), 2)
    if _n_prgs == 2:
        nisa.core_barrier(output_hbm, (0, 1))

    return output_hbm


def _compute_and_store_squared_sum(segment_sb, squared_scratch_sb, squared_sum_sb, squared_sum_dst):
    """Reduce ``sum(segment_sb ** 2)`` over the free dimension into
    ``squared_sum_sb``, then store it to ``squared_sum_dst``.

    Args:
        segment_sb: [n, d] SBUF view of one QKV segment (reduced over the free dim d).
        squared_scratch_sb: [n, d] SBUF scratch for the element-wise squares. Required
            by ``activation_reduce`` (it materializes the full squared output); its
            contents are otherwise discarded.
        squared_sum_sb: [n, 1] SBUF view receiving the reduced result before the store.
        squared_sum_dst: HBM destination view receiving the [n, 1] result. The caller
            passes a slice of the output flattened to [B*S, 1], so a flat offset stays
            in bounds even when B*S is folded (use_BxS_input_reshape).
    """
    nisa.activation_reduce(
        squared_scratch_sb,
        op=nl.square,
        data=segment_sb,
        reduce_op=nl.add,
        reduce_res=squared_sum_sb,
    )
    nisa.dma_copy(dst=squared_sum_dst, src=squared_sum_sb, dge_mode=dge_mode.swdge)


def _quantize_and_store_kv(
    output_sb: nl.NkiTensor,
    scale_sb: Optional[nl.NkiTensor],
    cache_hbm: nl.NkiTensor,
    kv_offset: int,
    i_batch: int,
    s_tile_global_offset: int,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    slot_mapping_sb: Optional[nl.NkiTensor] = None,
):
    """
    Optionally quantize KV values, then store to cache.

    When scale_sb is provided (FP8 path):
        quantized_value = clamp(value / scale, fp8_min, fp8_max)
    When scale_sb is None (bf16 path):
        Values are stored directly without quantization.

    Uses NBSd-style .ap() pattern for multi-head output.
    """
    kv_dim = dims.kv_dim
    d_head = dims.d_head
    num_kv_heads = dims.num_kv_heads
    max_seq_len = cache_hbm.shape[1] if not cfg.use_block_kv else None

    if scale_sb is not None:
        # FP8 quantization path: scale, clamp, then store
        inv_scale_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=scale_sb.dtype, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_scale_sb[0:s_tile_sz, 0:1], data=scale_sb[0:s_tile_sz, 0:1])

        clamped_sb = sbm.alloc_stack((nl.tile_size.pmax, kv_dim), dtype=cfg.kv_dtype, buffer=nl.sbuf)

        nisa.tensor_scalar(
            dst=clamped_sb[0:s_tile_sz, 0:kv_dim],
            data=output_sb[0:s_tile_sz, kv_offset : kv_offset + kv_dim],
            op0=nl.multiply,
            operand0=inv_scale_sb[0:s_tile_sz, 0:1],
        )
        nisa.tensor_scalar(
            dst=clamped_sb[0:s_tile_sz, 0:kv_dim],
            data=clamped_sb[0:s_tile_sz, 0:kv_dim],
            op0=nl.maximum,
            operand0=cfg.fp8_min,
            op1=nl.minimum,
            operand1=cfg.fp8_max,
        )
        src_sb = clamped_sb[0:s_tile_sz, 0:kv_dim]
    else:
        # bf16 path: store directly without quantization
        src_sb = output_sb[0:s_tile_sz, kv_offset : kv_offset + kv_dim]

    if cfg.use_block_kv and cfg.fp8_packed:
        # Head-split V cache layout [num_blocks, num_kv_heads, block_size, d_head] (fp8_packed).
        # V is not packed/transposed (PV matmul reads [block_size, d_head] directly); only the
        # KV heads are pulled out to a leading dim, matching the head-major K layout the decode
        # kernel reads. Flatten to [num_blocks * num_kv_heads * block_size, d_head] for scatter.
        block_size = cfg.block_size
        num_blocks = cache_hbm.shape[0]
        total_rows = num_blocks * num_kv_heads * block_size
        cache_2d = cache_hbm.reshape((total_rows, d_head))

        # slot_mapping holds slot = block * block_size + pos (row in the old [nb*block_size, kv_dim]
        # layout). The head-major row for (block, head, pos) is
        #   block * num_kv_heads * block_size + head * block_size + pos.
        # The per-head term is a static offset below; promote the block term's stride by num_kv_heads.
        v_slot_sb = slot_mapping_sb
        if num_kv_heads > 1:
            kernel_assert(
                block_size & (block_size - 1) == 0,
                f"fp8_packed multi-head V store requires power-of-2 block_size, got {block_size}",
            )
            log2_block_size = int(math.log2(block_size))
            v_slot_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=v_slot_sb[0:s_tile_sz, 0:1],
                data=slot_mapping_sb[0:s_tile_sz, 0:1],
                op0=nl.right_shift,
                operand0=log2_block_size,
            )
            # v_slot = block_idx * (num_kv_heads - 1) * block_size + slot
            nisa.scalar_tensor_tensor(
                dst=v_slot_sb[0:s_tile_sz, 0:1],
                data=v_slot_sb[0:s_tile_sz, 0:1],
                op0=nl.multiply,
                operand0=float((num_kv_heads - 1) * block_size),
                op1=nl.add,
                operand1=slot_mapping_sb[0:s_tile_sz, 0:1],
            )

        for h in range(num_kv_heads):
            nisa.dma_copy(
                dst=cache_2d.ap(
                    pattern=[[d_head, s_tile_sz], [1, d_head]],
                    offset=h * block_size * d_head,
                    vector_offset=v_slot_sb.ap(pattern=[[1, s_tile_sz], [1, 1]], offset=0),
                    indirect_dim=0,
                ),
                src=src_sb[0:s_tile_sz, nl.ds(h * d_head, d_head)],
                dge_mode=dge_mode.swdge,
            )
    elif cfg.use_block_kv:
        # Block KV layout: [num_blocks, block_size, kv_dim]
        num_blocks = cache_hbm.shape[0]
        block_size = cfg.block_size
        total_slots = num_blocks * block_size
        cache_2d = cache_hbm.reshape((total_slots, kv_dim))

        nisa.dma_copy(
            dst=cache_2d.ap(
                pattern=[[kv_dim, s_tile_sz], [1, kv_dim]],
                offset=0,
                vector_offset=slot_mapping_sb.ap(pattern=[[1, s_tile_sz], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            src=src_sb,
            dge_mode=dge_mode.swdge,
        )
    else:
        # Cache layout: [B, max_seq_len, kv_dim] - same pattern as BSD output store
        max_seq_len = cache_hbm.shape[1]
        cache_row = i_batch * max_seq_len + s_tile_global_offset
        nisa.dma_copy(
            dst=cache_hbm.flatten_dims(start_dim=0, end_dim=1).slice(0, cache_row, cache_row + s_tile_sz),
            src=src_sb,
            dge_mode=dge_mode.swdge,
        )


def _quantize_and_store_k_transposed(
    output_sb: nl.NkiTensor,
    inv_scale_sb: Optional[nl.NkiTensor],
    cache_hbm: nl.NkiTensor,
    kv_offset: int,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    slot_mapping_sb: Optional[nl.NkiTensor] = None,
    i_batch: int = 0,
    s_tile_global_offset: int = 0,
):
    """
    Optionally quantize K values, then transpose and store in [d_head, seq] layout to cache.

    When scale_sb is provided (FP8 path):
        quantized_value = clamp(value / scale, fp8_min, fp8_max)
    When scale_sb is None (bf16 path):
        Values are transposed and stored directly without quantization.

    Transposes each KV head from [s_tile_sz, d_head] to [d_head, s_tile_sz] before writing,
    which is the layout expected by attention TKG kernels for efficient K access.

    Supports two cache layouts:
    - Block KV: cache shape [num_blocks * num_kv_heads, d_head, block_size], uses slot_mapping_sb
      for indirect addressing to map tokens to non-contiguous physical blocks.
    - Flat KV: cache shape [B, kv_dim, max_seq_len], writes contiguously per batch.

    Args:
        output_sb (nl.NkiTensor): [s_tile_sz, I], QKV projection output tile in SBUF.
        scale_sb (Optional[nl.NkiTensor]): [s_tile_sz, 1], Per-token quantization scale in SBUF.
            When None, values are stored directly without quantization (bf16 path).
        cache_hbm (nl.NkiTensor): K cache tensor in HBM. Shape depends on cache mode:
            - Block KV: [num_blocks * num_kv_heads, d_head, block_size]
            - Flat KV: [B, kv_dim, max_seq_len]
        kv_offset (int): Column offset into output_sb where K values start (typically q_dim).
        s_tile_sz (int): Number of active sequence positions in this tile.
        cfg (QKV_CTE_Config): Kernel configuration (fp8_min, fp8_max, kv_dtype, block_size, etc.).
        dims (QKV_CTE_Dims): Tensor dimensions (kv_dim, d_head, num_kv_heads).
        sbm (SbufManager): SBUF memory manager for scratch allocations.
        slot_mapping_sb (Optional[nl.NkiTensor]): [s_tile_sz, 1], Per-token slot indices for block KV
            cache. Each slot encodes block_idx and offset within block. Required when cfg.use_block_kv
            is True. Default: None
        i_batch (int): Batch index for flat KV cache addressing. Default: 0
        s_tile_global_offset (int): Global sequence offset for flat KV cache addressing. Default: 0
    """
    kv_dim = dims.kv_dim
    d_head = dims.d_head
    num_kv_heads = dims.num_kv_heads

    transposed_sb = sbm.alloc_stack(
        (nl.tile_size.pmax, num_kv_heads * nl.tile_size.pmax),
        dtype=cfg.kv_dtype,
        buffer=nl.sbuf,
    )
    _scale_and_transpose_per_head(
        output_sb,
        inv_scale_sb,
        kv_offset,
        s_tile_sz,
        cfg,
        dims,
        sbm,
        transposed_sb,
    )

    if cfg.use_block_kv:
        block_size = cfg.block_size

        # Divide by num_kv_heads to get the number of physical blocks.
        num_blocks = cache_hbm.shape[0] // num_kv_heads

        # block_size must be power-of-2
        log2_block_size = int(math.log2(block_size))
        block_size_mask = block_size - 1

        # Strides for linearized addressing into the flattened cache:
        #   Within a block: elements are [d_head, block_size] contiguous
        #   Across heads:   head_stride = d_head * block_size
        #   Across blocks:  block_stride = num_kv_heads * head_stride
        head_stride = d_head * block_size
        block_stride = num_kv_heads * head_stride

        # Flatten cache to 1D for indirect DMA addressing via offset tensors.
        total_elems = num_blocks * num_kv_heads * d_head * block_size
        cache_flat = cache_hbm.reshape((total_elems, 1))

        num_full_blocks = s_tile_sz // block_size
        remainder = s_tile_sz % block_size

        # Compute block base addresses for ALL tokens in partition dim:
        #   block_base_sb[t, 0] = slot[t] * (num_kv_heads * d_head)
        block_base_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=block_base_sb[0:s_tile_sz, 0:1],
            data=slot_mapping_sb[0:s_tile_sz, 0:1],
            op0=nl.multiply,
            operand0=num_kv_heads * d_head,
        )

        # Loop over full blocks: scalar_offset reads from partition token_start.
        # AP partition stride [[block_size, d_head]] adds p * block_size.
        for i_block in nl.affine_range(num_full_blocks):
            token_start = i_block * block_size
            nisa.dma_copy(
                dst=cache_flat.ap(
                    pattern=[
                        [block_size, d_head],
                        [head_stride, num_kv_heads],
                        [1, block_size],
                    ],
                    offset=0,
                    scalar_offset=block_base_sb[token_start : token_start + 1, 0:1],
                    indirect_dim=0,
                ),
                src=transposed_sb.reshape_dim(1, [num_kv_heads, nl.tile_size.pmax])
                .slice(0, 0, d_head)
                .slice(2, token_start, token_start + block_size),
                dge_mode=dge_mode.swdge,
            )

        # Handle remainder (partial last block) if any.
        if remainder > 0:
            token_start = num_full_blocks * block_size

            # slot_in_block from original slot_mapping (partition dim)
            slot_in_block_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=slot_in_block_sb[0:1, 0:1],
                data=slot_mapping_sb[token_start : token_start + 1, 0:1],
                op0=nl.bitwise_and,
                operand0=block_size_mask,
            )
            # Compute block_idx * block_stride for remainder (can't reuse block_base_sb
            # since it has slot * num_kv_heads * d_head which includes slot_in_block).
            base_offset_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=base_offset_sb[0:1, 0:1],
                data=slot_mapping_sb[token_start : token_start + 1, 0:1],
                op0=nl.right_shift,
                operand0=log2_block_size,
            )
            nisa.scalar_tensor_tensor(
                dst=base_offset_sb[0:1, 0:1],
                data=base_offset_sb[0:1, 0:1],
                op0=nl.multiply,
                operand0=float(block_stride),
                op1=nl.add,
                operand1=slot_in_block_sb[0:1, 0:1],
            )

            nisa.dma_copy(
                dst=cache_flat.ap(
                    pattern=[[block_size, d_head], [head_stride, num_kv_heads], [1, remainder]],
                    offset=0,
                    scalar_offset=base_offset_sb[0:1, 0:1],
                    indirect_dim=0,
                ),
                src=transposed_sb.reshape_dim(1, [num_kv_heads, nl.tile_size.pmax])
                .slice(0, 0, d_head)
                .slice(2, token_start, token_start + remainder),
                dge_mode=dge_mode.swdge,
            )

    else:
        # Flat KV cache: [B, kv_dim, max_seq_len] — no block indirection needed.
        max_seq_len = cache_hbm.shape[2]

        # Batched DMA: write all heads with a single strided DMA.
        # cache_hbm is [B, kv_dim, max_seq_len] = [B, num_kv_heads, d_head, max_seq_len];
        # store as [d_head, num_kv_heads, s] (heads-major in the free dims).
        nisa.dma_copy(
            dst=cache_hbm.reshape_dim(1, [num_kv_heads, d_head])
            .slice(0, i_batch, i_batch + 1)
            .slice(3, s_tile_global_offset, s_tile_global_offset + s_tile_sz)
            .permute([0, 2, 1, 3])
            .squeeze_dim(0),
            src=transposed_sb.reshape_dim(1, [num_kv_heads, nl.tile_size.pmax])
            .slice(0, 0, d_head)
            .slice(2, 0, s_tile_sz),
            dge_mode=dge_mode.swdge,
        )


def _quantize_and_store_k_fp8_packed(
    output_sb_tiles: List[nl.NkiTensor],
    inv_scale_sb: nl.NkiTensor,
    cache_hbm: nl.NkiTensor,
    kv_offset: int,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    packed_slot_sb: nl.NkiTensor,
):
    """
    Quantize K values to FP8, pack consecutive pairs, and store to packed block KV cache.

    Processes 1 or 2 tiles at once. When 2 tiles are provided, each tile is transposed
    (stage 1) into adjacent slots within a shared buffer so that per head, both tiles'
    fp8 data is contiguous on the free dimension. After bf16 reinterpretation, this gives
    a full [d_head, pmax] bf16 region per head — enabling a single full-partition
    nc_transpose (stage 2) and scatter DMA (stage 3).

    Two consecutive FP8 sequence positions are packed into one row:
    position 2i at [:, :, :, 0] and position 2i+1 at [:, :, :, 1].

    The packing requires two nc_transposes per head:
    1. [s_tile_sz, d_head] bf16 → [d_head, s_tile_sz] fp8 (per tile, with FP8 clamp)
       Tiles write into adjacent slots via _scale_and_transpose_per_head(tile_idx=...).
    2. Reinterpret as bf16 → single nc_transpose [d_head, num_packed_rows] → [num_packed_rows, d_head]

    Args:
        output_sb_tiles: List of 1 or 2 output tiles, each [s_tile_sz, I] in SBUF.
        inv_scale_sb: [pmax, 1], Pre-computed inverse quantization scale in SBUF.
        cache_hbm: K cache in HBM, shape [num_blocks, block_size // 2, kv_dim, 2] fp8.
        kv_offset: Column offset into output_sb where K values start.
        s_tile_sz: Number of active sequence positions per tile. Must be even.
        cfg: Kernel configuration.
        dims: Tensor dimensions (kv_dim, d_head, num_kv_heads).
        sbm: SBUF memory manager.
        packed_slot_sb: [num_tiles * s_tile_sz // 2, 1], Pre-computed packed row indices.
    """
    kv_dim = dims.kv_dim
    d_head = dims.d_head
    num_kv_heads = dims.num_kv_heads
    num_tiles = len(output_sb_tiles)
    kernel_assert(num_tiles in (1, 2), f"Expected 1 or 2 tiles, got {num_tiles}")
    packed_rows_per_tile = s_tile_sz // 2
    num_packed_rows = num_tiles * packed_rows_per_tile

    # Stage 1: scale + per-head transpose with FP8 clamp for each tile.
    # Uses _scale_and_transpose_per_head (shared with _quantize_and_store_k_transposed).
    # Layout: [pmax, num_kv_heads * num_tiles * pmax] fp8.
    # Per head, tiles are adjacent: head i occupies [i * num_tiles * pmax : (i+1) * num_tiles * pmax],
    # with tile t at sub-offset t * pmax. This adjacency enables a single transpose 2 per head.
    combined_fp8_sb = sbm.alloc_stack(
        (nl.tile_size.pmax, num_kv_heads * num_tiles * nl.tile_size.pmax),
        dtype=cfg.kv_dtype,
        buffer=nl.sbuf,
    )

    for t_idx in range(num_tiles):
        _scale_and_transpose_per_head(
            output_sb_tiles[t_idx],
            inv_scale_sb,
            kv_offset,
            s_tile_sz,
            cfg,
            dims,
            sbm,
            transposed_sb=combined_fp8_sb,
            num_tiles_per_head=num_tiles,
            tile_idx=t_idx,
        )

    # Stage 2: Per-head second transpose (bf16 reinterpretation → store orientation).
    # Reinterpret combined_fp8_sb [pmax, num_kv_heads * num_tiles * pmax] fp8
    # as [pmax, num_kv_heads * num_tiles * pmax // 2] bf16 (zero-cost, same physical memory).
    # For 2 tiles: each head has num_tiles * packed_rows_per_tile = pmax bf16 values → full 128-partition transpose.
    # Output buffer for all heads: [num_packed_rows, kv_dim] bf16.
    packed_sb = sbm.alloc_stack((num_packed_rows, kv_dim), dtype=nl.bfloat16, buffer=nl.sbuf)

    NUM_TRANSPOSE_PSUM_BUFS = min(num_kv_heads, NUM_HW_PSUM_BANKS)
    transpose_psum_bufs2 = []
    for _ in nl.affine_range(NUM_TRANSPOSE_PSUM_BUFS):
        transpose_psum_bufs2.append(
            nl.ndarray((nl.tile_size.pmax, nl.tile_size.pmax), dtype=nl.bfloat16, buffer=nl.psum)
        )

    packed_bf16_view = combined_fp8_sb.view(nl.bfloat16)

    num_chunks = math.ceil(num_kv_heads / NUM_TRANSPOSE_PSUM_BUFS)
    for i_chunk in nl.affine_range(num_chunks):
        chunk_start = i_chunk * NUM_TRANSPOSE_PSUM_BUFS
        chunk_size = min(NUM_TRANSPOSE_PSUM_BUFS, num_kv_heads - chunk_start)
        for j in nl.affine_range(chunk_size):
            i_head = chunk_start + j

            # Per head: bf16 data at offset i_head * num_tiles * pmax // 2, length num_packed_rows
            head_bf16_offset = i_head * num_tiles * (nl.tile_size.pmax // 2)

            # Transpose 2: [d_head, num_packed_rows] bf16 → PSUM [num_packed_rows, d_head] bf16
            nisa.nc_transpose(
                data=packed_bf16_view.slice(0, start=0, end=d_head).slice(
                    1, start=head_bf16_offset, end=head_bf16_offset + num_packed_rows
                ),
                dst=transpose_psum_bufs2[j][0:num_packed_rows, 0:d_head],
            )

            # Copy PSUM → packed_sb at the correct head offset
            nisa.tensor_copy(
                dst=packed_sb[0:num_packed_rows, nl.ds(i_head * d_head, d_head)],
                src=transpose_psum_bufs2[j][0:num_packed_rows, 0:d_head],
                engine=nisa.vector_engine if i_head % 2 == 0 else nisa.scalar_engine,
            )

    # Stage 3: Scatter DMA store to packed block KV cache.
    # cache_hbm is [num_blocks, n_kv_heads, block_size // 2, d_head, 2] fp8.
    # Flatten to [total_packed_rows, d_head * 2] fp8 for scatter addressing.
    # Reinterpret SBUF src as fp8 to match element count (same bytes, 2x elements).
    num_blocks = cache_hbm.shape[0]
    block_size_half = cache_hbm.shape[2]
    total_packed_rows = num_blocks * num_kv_heads * block_size_half
    cache_2d = cache_hbm.reshape((total_packed_rows, d_head * 2))
    packed_sb_fp8 = packed_sb.view(cfg.kv_dtype)
    for i in range(num_kv_heads):
        nisa.dma_copy(
            dst=cache_2d.ap(
                pattern=[[d_head * 2, num_packed_rows], [1, d_head * 2]],
                offset=i * block_size_half * d_head * 2,
                vector_offset=packed_slot_sb.ap(pattern=[[1, num_packed_rows], [1, 1]], offset=0),
                indirect_dim=0,
            ),
            src=packed_sb_fp8[:num_packed_rows, nl.ds(i * d_head * 2, d_head * 2)],
            dge_mode=dge_mode.swdge,
        )


def _scale_and_transpose_per_head(
    output_sb: nl.NkiTensor,
    inv_scale_sb: Optional[nl.NkiTensor],
    kv_offset: int,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    transposed_sb: nl.NkiTensor,
    num_tiles_per_head: int = 1,
    tile_idx: int = 0,
) -> nl.NkiTensor:
    """Optionally scale K values, nc_transpose per head, clamp to FP8 on PSUM eviction (FP8 path).

    Produces a per-head transposed SBUF buffer. Head `i_head` data (s_tile_sz elements)
    is written at free-dim offset: i_head * num_tiles_per_head * pmax + tile_idx * pmax.

    When num_tiles_per_head=1 (default): layout is [pmax, num_kv_heads * pmax], head i at i*pmax.
    When num_tiles_per_head=2: layout is [pmax, num_kv_heads * 2 * pmax], with tiles adjacent
    per head (enables a single full-partition transpose 2 across both tiles).

    When inv_scale_sb is provided (FP8 path):
      - K is multiplied by inv_scale (precomputed 1/scale) before transpose
      - PSUM→SBUF eviction applies clamp to [fp8_min, fp8_max] and casts to cfg.kv_dtype (fp8)
    When inv_scale_sb is None (bf16 path):
      - K is transposed directly without scaling/clamping; output is bf16

    Writes into transposed_sb (caller allocates the output buffer).

    Used by both `_quantize_and_store_k_transposed` and
    `_quantize_and_store_k_fp8_packed` to share the scale + transpose-1 path.
    """
    kv_dim = dims.kv_dim
    d_head = dims.d_head
    num_kv_heads = dims.num_kv_heads

    if inv_scale_sb is not None:
        # FP8 path: scale first, clamp after transpose
        scaled_sb = sbm.alloc_stack((nl.tile_size.pmax, kv_dim), dtype=output_sb.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=scaled_sb[0:s_tile_sz, 0:kv_dim],
            data=output_sb[0:s_tile_sz, kv_offset : kv_offset + kv_dim],
            op0=nl.multiply,
            operand0=inv_scale_sb[0:s_tile_sz, 0:1],
        )
        transpose_src_sb = scaled_sb
    else:
        # bf16 path: transpose directly from output_sb (no scaling)
        transpose_src_sb = output_sb[0:s_tile_sz, kv_offset : kv_offset + kv_dim]

    NUM_TRANSPOSE_PSUM_BUFS = min(num_kv_heads, NUM_HW_PSUM_BANKS)
    transpose_psum_bufs = []
    for _ in nl.affine_range(NUM_TRANSPOSE_PSUM_BUFS):
        transpose_psum_bufs.append(
            nl.ndarray((nl.tile_size.pmax, nl.tile_size.pmax), dtype=output_sb.dtype, buffer=nl.psum)
        )

    num_chunks = math.ceil(num_kv_heads / NUM_TRANSPOSE_PSUM_BUFS)
    for i_chunk in nl.affine_range(num_chunks):
        chunk_start = i_chunk * NUM_TRANSPOSE_PSUM_BUFS
        chunk_size = min(NUM_TRANSPOSE_PSUM_BUFS, num_kv_heads - chunk_start)
        for j in nl.affine_range(chunk_size):
            i_head = chunk_start + j
            nisa.nc_transpose(
                data=transpose_src_sb[0:s_tile_sz, nl.ds(i_head * d_head, d_head)],
                dst=transpose_psum_bufs[j][0:d_head, 0:s_tile_sz],
            )
            if inv_scale_sb is not None:
                # FP8 path: clamp during PSUM→SBUF eviction
                nisa.tensor_scalar(
                    dst=transposed_sb[
                        0:d_head,
                        nl.ds(
                            i_head * num_tiles_per_head * nl.tile_size.pmax + tile_idx * nl.tile_size.pmax, s_tile_sz
                        ),
                    ],
                    data=transpose_psum_bufs[j][0:d_head, 0:s_tile_sz],
                    op0=nl.maximum,
                    operand0=cfg.fp8_min,
                    op1=nl.minimum,
                    operand1=cfg.fp8_max,
                )
            else:
                # bf16 path: plain copy from PSUM→SBUF
                nisa.tensor_copy(
                    dst=transposed_sb[
                        0:d_head,
                        nl.ds(
                            i_head * num_tiles_per_head * nl.tile_size.pmax + tile_idx * nl.tile_size.pmax, s_tile_sz
                        ),
                    ],
                    src=transpose_psum_bufs[j][0:d_head, 0:s_tile_sz],
                    engine=nisa.vector_engine if (i_head + tile_idx) % 2 == 0 else nisa.scalar_engine,
                )

    return transposed_sb


def _qkv_cte_impl(
    input_hbm,
    fused_qkv_weights_hbm,
    output_hbm,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    bias_hbm: Optional[nl.NkiTensor] = None,
    # Fused Residual Add Related
    mlp_prev_hbm: Optional[nl.NkiTensor] = None,
    attention_prev_hbm: Optional[nl.NkiTensor] = None,
    # Fused Normalization Related
    gamma_norm_weights_hbm: Optional[nl.NkiTensor] = None,
    layer_norm_bias_hbm: Optional[nl.NkiTensor] = None,
    norm_eps: Optional[float] = 1e-6,
    # Fused RoPE Related
    cos_cache_hbm: Optional[nl.NkiTensor] = None,
    sin_cache_hbm: Optional[nl.NkiTensor] = None,
    # Fused RoPE + QK-norm: K-specific gamma-fused caches
    k_cos_cache_hbm: Optional[nl.NkiTensor] = None,
    k_sin_cache_hbm: Optional[nl.NkiTensor] = None,
    # KV Cache Related
    q_tensor_hbm: Optional[nl.NkiTensor] = None,
    k_cache_hbm: Optional[nl.NkiTensor] = None,
    v_cache_hbm: Optional[nl.NkiTensor] = None,
    k_scale_hbm: Optional[nl.NkiTensor] = None,
    v_scale_hbm: Optional[nl.NkiTensor] = None,
    # Block KV Cache Related
    slot_mapping_hbm: Optional[nl.NkiTensor] = None,
    # Quantization Related
    qkv_in_scale: Optional[nl.NkiTensor] = None,
    qkv_w_scale: Optional[nl.NkiTensor] = None,
    # Optional per-segment sum-of-squares outputs (each [B, S, 1])
    q_squared_sum_hbm: Optional[nl.NkiTensor] = None,
    k_squared_sum_hbm: Optional[nl.NkiTensor] = None,
    v_squared_sum_hbm: Optional[nl.NkiTensor] = None,
):
    """
    Core QKV CTE kernel implementation.

    Performs the main computation including optional normalization, QKV projection,
    and optional RoPE application. Handles memory allocation, tiling, and data movement
    between HBM, SBUF, and PSUM.

    Args:
        input_hbm (nl.NkiTensor): [dims.B, dims.S, dims.H], Input tensor on HBM
        fused_qkv_weights_hbm (nl.NkiTensor): [dims.H, dims.I], Weight matrix on HBM
        output_hbm (nl.NkiTensor): Output tensor on HBM with shape determined by cfg.output_layout
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager
        bias_hbm (Optional[nl.NkiTensor]): [1, I], Optional bias tensor on HBM
        mlp_prev_hbm (Optional[nl.NkiTensor]): [dims.B, dims.S, dims.H], Optional MLP residual on HBM
        attention_prev_hbm (Optional[nl.NkiTensor]): [dims.B, dims.S, dims.H], Optional attention residual on HBM
        gamma_norm_weights_hbm (Optional[nl.NkiTensor]): [1, H], Optional normalization weights on HBM
        layer_norm_bias_hbm (Optional[nl.NkiTensor]): [1, H], Optional layer norm bias on HBM
        norm_eps (Optional[float]): Epsilon for normalization stability
        cos_cache_hbm (Optional[nl.NkiTensor]): [B, S, d_head], Optional RoPE cosine cache on HBM
        sin_cache_hbm (Optional[nl.NkiTensor]): [B, S, d_head], Optional RoPE sine cache on HBM

    Returns:
        nl.NkiTensor: Output tensor (same as output_hbm parameter)

    Notes:
        - Processes only dims.S_shard portion of sequence dimension when sharded
        - Uses multi-buffering for sequence tiles to improve performance
        - Supports weight prefetching when SBUF space allows
    """
    # Uncomment for debug.
    # cfg.print()
    # dims.print()

    """
    Input tensor shape: [dims.B, dims.S, dims.H]
    Weight tensor shape: [dims.H, dims.I]
    
    We apply QKV projection only on dims.S_shard part of input_hbm (with dims.S_shard_offset).
    """

    S_shard = dims.S_shard
    H = dims.H
    I = dims.I
    if dims.S_shard == 0:
        return output_hbm
    _is_fp8_input = input_hbm.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]

    """
    If user provided SbufManager (with more restricted sb_lower_bound and sb_upper_bound),
    use that (likely at the expense of performance). Otherwise, use most sbuf space available.
    """
    if sbm == None:
        sbm_logger = get_logger(name="qkv_cte")
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=cfg.total_available_sbuf_space_to_this_kernel,
            use_auto_alloc=cfg.use_auto_allocation,
            logger=sbm_logger,
        )
    sbm.open_scope()

    ######################### Global SBUF Allocations ######################################
    # Allocate zero_bias_sb, norm_eps_sb, bias_sb, gamma_norm_weights_sb, layer_norm_bias_sb

    zero_bias_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_bias_sb, value=0)

    norm_eps_sb = _make_activation_constant_bias(norm_eps, sbm, nl.tile_size.pmax, dtype=cfg.compute_mm_dtype)

    if cfg.add_bias:
        # Load Bias (1, I) to SBUF as (1, I), and broadcast it to (128, I) using stream_shuffle.
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    """
    Load gamma_norm_weights_hbm (1,H) to sbuf.
    
    Mathematically, we (later) need to apply elementwise multiplication: (input) [S, H] * (gamma) [1, H] for each row.
    Note: NormType.RMS_NORM_SKIP_GAMMA skips this step.
    """
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        # We load gamma weights into SBUF as a 2-D tensor of shape [128, ceil(H / nl.tile_size.pmax) ].
        gamma_norm_weights_sb = _load_norm_weights(norm_weights_hbm=gamma_norm_weights_hbm, cfg=cfg, dims=dims, sbm=sbm)

        """
        gamma_norm_weights_sb is used for both RMS_NORM and LAYER_NORM.
        But, LAYER_NORM may have an additional Beta bias term provided: layer_norm_bias_hbm
        """
        if cfg.add_layer_norm_bias:
            # Shape: [128, ceil(H / nl.tile_size.pmax)]
            layer_norm_bias_sb = _load_norm_weights(norm_weights_hbm=layer_norm_bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    # Load KV quantization scales if enabled (not needed for bf16 KV cache)
    k_scale_sb = None
    v_scale_sb = None
    k_inv_scale_sb = None
    v_inv_scale_sb = None
    if cfg.use_kv_quantization:
        k_scale_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        v_scale_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=k_scale_sb[0 : nl.tile_size.pmax, 0:1],
            src=k_scale_hbm[0 : nl.tile_size.pmax, 0:1],
            dge_mode=dge_mode.swdge,
        )
        nisa.dma_copy(
            dst=v_scale_sb[0 : nl.tile_size.pmax, 0:1],
            src=v_scale_hbm[0 : nl.tile_size.pmax, 0:1],
            dge_mode=dge_mode.swdge,
        )
        # Precompute inverse scales (1/scale) once — avoids redundant reciprocal per tile.
        k_inv_scale_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        v_inv_scale_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=k_inv_scale_sb, data=k_scale_sb)
        nisa.reciprocal(dst=v_inv_scale_sb, data=v_scale_sb)

    # Load quantization scales
    w_scale_tile, in_scale_tile = None, None
    row_w_scale_sb = None
    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
        w_scale_tile = sbm.alloc_stack(shape=(nl.tile_size.pmax, 3), dtype=qkv_w_scale.dtype, buffer=nl.sbuf)
        # Load and broadcast weight scale
        if qkv_w_scale.shape[0] == 1:
            nisa.dma_copy(dst=w_scale_tile[0, :], src=qkv_w_scale[0, :])
            stream_shuffle_broadcast(w_scale_tile, w_scale_tile)
        else:
            nisa.dma_copy(dst=w_scale_tile, src=qkv_w_scale)

        in_scale_tile = sbm.alloc_stack(shape=(nl.tile_size.pmax, 1), dtype=qkv_in_scale.dtype)
        if qkv_in_scale.shape[0] == 1:
            nisa.dma_copy(dst=in_scale_tile[0, :], src=qkv_in_scale[0, :])
            stream_shuffle_broadcast(in_scale_tile, in_scale_tile)
        else:
            nisa.dma_copy(dst=in_scale_tile, src=qkv_in_scale)
        # pre-apply input scales onto the weight scaling
        nisa.activation(dst=w_scale_tile, op=nl.copy, data=w_scale_tile, scale=in_scale_tile)

        if not _is_fp8_input:
            # Compute reciprocal once for quantization: 1/scale
            nisa.reciprocal(dst=in_scale_tile, data=in_scale_tile)
    elif cfg.quantization_config.quantization_type == QuantizationType.ROW:
        # ROW quantization: per-column weight scale [P_MAX, I] or [1, I]
        row_w_scale_sb = sbm.alloc_stack(shape=(nl.tile_size.pmax, I), dtype=qkv_w_scale.dtype, buffer=nl.sbuf)
        if qkv_w_scale.shape[0] == 1:
            nisa.dma_copy(dst=row_w_scale_sb[0:1, :], src=qkv_w_scale[0:1, :I])
            stream_shuffle_broadcast(row_w_scale_sb, row_w_scale_sb)
        else:
            nisa.dma_copy(dst=row_w_scale_sb, src=qkv_w_scale[:, :I])

    ######################## QK-Norm SBUF Allocations ########################################
    qk_norm_bufs = _allocate_qk_norm_buffers(cfg, dims, sbm, nl.tile_size.pmax)
    _has_qk_norm = qk_norm_bufs is not None

    ######################## Choose Multi-Buffering Degree ###################################

    """
    Multi-Buffering S: Choose max multi-buffering degree for sequence length, without spilling SBUF and PSUM space.
    
    WARNING: This function needs to be updated if any new tensors get added to the kernel.
    It assumes current tensor shapes, and its "look-ahead", e.g. pre-calculates SBUF space ahead of time.
    Note: In auto-allocation mode, sbuf space calculations do not make sense, but they do not break the kernel correctness.
    """
    s_multi_buffer_degree, projected_sbuf_taken_space_after_multi_buffer = _multi_buffering_degree_for_seqlen(
        cfg=cfg, dims=dims, sbm=sbm, qkv_in_scale=qkv_in_scale
    )

    # Block is PMAX * multi_buffer_degree, e.g.  process [128 * 4, H] elements of S at once.
    S_BLOCK_SIZE = s_multi_buffer_degree * min(dims.S_shard, nl.tile_size.pmax)
    num_blocks_per_S_shard = math.ceil(dims.S_shard / S_BLOCK_SIZE) if S_BLOCK_SIZE > 0 else 0

    ######################## Weight Prefetching: Enough Space Left ?  #########################

    use_weight_prefetch = _use_weight_prefetch(
        projected_sbuf_taken_space_after_multi_buffer, cfg=cfg, dims=dims, sbm=sbm
    )

    if use_weight_prefetch:
        """
        If rhs (weights) is small enough, weights can be pre-loaded before QKV projection.

        NOTE: In both prefetched and non-pretched case, we append allocated weight tensor to the same "weights_sb" list,
        to keep later changes to the minimum. To keep indexing differences in QKV projection to the minimum between the two cases,
        keep the shape of the allocated tensor the same:
        weights_allocated = (128, num_allocated_H_subtiles_per_weight_load, I), and
        weights_sb[...] may multi-buffer/allocate multiple of above tensors

        In the case of weight prefetching, set the following variables:
        """
        num_weight_buffers = 1  # Since we allocate all weights at once, so there is no need for multiple-buffers. We still need to set it to 1 as the kernel uses this constant later.
        weight_load_block_size_per_H = (
            H  # We pre-load the entire weight matrix. In non-prefetched case, we can load e.g 1024 H blocks at a time.
        )
        num_weight_load_blocks_per_H = 1  # In pre-fetch case, H / weight_load_block_size_per_H = 1
        max_num_128_H_subtiles_per_weight_block = math.ceil(weight_load_block_size_per_H / 128)  # = math.ceil(H / 128).

        weights_sb = []
        weights_prefetched_sb = sbm.alloc_stack(
            (nl.tile_size.pmax, max_num_128_H_subtiles_per_weight_block, I),
            dtype=cfg.weights_dtype,
            buffer=nl.sbuf,
        )

        for i_tile_H in range(max_num_128_H_subtiles_per_weight_block):
            h_tile_sz = min(nl.tile_size.pmax, H - (i_tile_H * nl.tile_size.pmax))
            nisa.dma_copy(
                dst=weights_prefetched_sb.slice(0, 0, h_tile_sz)[:, i_tile_H, :],
                src=fused_qkv_weights_hbm.slice(
                    0, i_tile_H * nl.tile_size.pmax, i_tile_H * nl.tile_size.pmax + h_tile_sz
                ),
                dge_mode=dge_mode.swdge,
            )
        weights_sb.append(weights_prefetched_sb)

    for i_batch in range(dims.B):
        for i_block_S in nl.affine_range(num_blocks_per_S_shard):
            sbm.open_scope()
            # Adjust for the last loop iteration.
            s_block_sz = min(S_BLOCK_SIZE, S_shard - S_BLOCK_SIZE * i_block_S)
            num_S_tiles_in_block = math.ceil(s_block_sz / nl.tile_size.pmax)

            #################### Start of Allocations for Multi-Buffered tensors ##############################
            input_sb = []
            if cfg.quantization_config.quantization_type == QuantizationType.STATIC and not _is_fp8_input:
                input_quantized_sb = []
            for _ in range(num_S_tiles_in_block):
                align = 32 if cfg.load_input_with_DMA_transpose else 1  # DMA_transpose requires align=32.
                input_dtype = (
                    cfg.compute_mm_dtype
                    if cfg.quantization_config.quantization_type != QuantizationType.STATIC
                    else input_hbm.dtype
                )
                input_sb.append(
                    sbm.alloc_stack(
                        (nl.tile_size.pmax, H),
                        dtype=input_dtype,
                        buffer=nl.sbuf,
                        align=align,
                    )
                )
                if cfg.quantization_config.quantization_type == QuantizationType.STATIC and not _is_fp8_input:
                    # assume q, k, v projections have same input scale, therefore no need to have multiple input copies.
                    input_quantized_sb.append(
                        sbm.alloc_stack(
                            (nl.tile_size.pmax, H),
                            dtype=cfg.quantization_config.quant_dtype,
                            buffer=nl.sbuf,
                            align=align,
                        )
                    )

            output_sb = []
            for _ in range(num_S_tiles_in_block):
                output_dtype = (
                    cfg.compute_mm_dtype
                    if cfg.quantization_config.quantization_type != QuantizationType.STATIC
                    else nl.bfloat16
                )
                output_sb.append(sbm.alloc_stack((nl.tile_size.pmax, I), dtype=output_dtype, buffer=nl.sbuf))

            if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
                # Allocate rms_norm tensor outside of tiled S buffer loop, as square_sum has NUM_128_TILES_PER_S_BUFFER in its shape.
                # Write RMS and RMS Reciprocal tensors here, in-place.
                square_sum_sb = []
                for _ in range(num_S_tiles_in_block):
                    square_sum_sb.append(sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=cfg.act_dtype, buffer=nl.sbuf))

            elif cfg.fused_norm_type == NormType.LAYER_NORM:
                # Used for the result of layer_norm, stores mean and variance.
                NUM_AGGR_STATS = 2  # mean and variance.
                bn_aggr_result_sb = []
                for _ in range(num_S_tiles_in_block):
                    bn_aggr_result_sb.append(
                        sbm.alloc_stack((nl.tile_size.pmax, NUM_AGGR_STATS), dtype=cfg.act_dtype, buffer=nl.sbuf)
                    )

            # RoPE buffers: cos [pmax, d_head], sin [pmax, d_head//2], scratch [pmax, d_head*2]
            rope_bufs = _allocate_rope_buffers(
                cfg,
                dims,
                sbm,
                nl.tile_size.pmax,
                num_S_tiles_in_block,
                _has_qk_norm,
                cos_cache_hbm,
                sin_cache_hbm,
                k_cos_cache_hbm=k_cos_cache_hbm,
                k_sin_cache_hbm=k_sin_cache_hbm,
            )
            #########################  End of Allocations for Multi-Buffered tensors ##################################

            # In this case, we will transpose the input buffer using PE array.
            if not cfg.load_input_with_DMA_transpose:
                for i_tile_S in range(num_S_tiles_in_block):
                    sbm.open_scope()
                    # i_tile_S is used to index input_sb, e.g., a single tile of input row [128, H].
                    ######################################################################################################
                    # Step 1: Load the row of input tensor: [nl.tile_size.pmax, H]
                    #         (optionally apply fused_residual_add with mlp_prev, and attention_prev).
                    #          If load_input_with_DMA_transpose is applicable, this step is skipped, and moved later.
                    ######################################################################################################
                    S_TILE_SIZE = nl.tile_size.pmax
                    s_tile_local_offset = (
                        i_block_S * S_BLOCK_SIZE + i_tile_S * S_TILE_SIZE
                    )  # s_tile offset within S_shard.
                    s_tile_sz = min(
                        nl.tile_size.pmax, S_shard - s_tile_local_offset
                    )  # tile_size adjusted for the last loop iteration.
                    if cfg.fused_residual_add:
                        # Load row of input, and apply fused residual add.
                        s_tile_global_offset = i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                        s_tile_row = s_tile_global_offset // H
                        nisa.dma_compute(
                            dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            srcs=[
                                input_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                    0, s_tile_row, s_tile_row + s_tile_sz
                                ),
                                mlp_prev_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                    0, s_tile_row, s_tile_row + s_tile_sz
                                ),
                                attention_prev_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                    0, s_tile_row, s_tile_row + s_tile_sz
                                ),
                            ],
                            scales=[1.0, 1.0, 1.0],
                            reduce_op=nl.add,
                        )
                    else:  # Do a regular input load without the fused residual add.
                        s_tile_global_offset = i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                        nisa.dma_copy(
                            dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            src=input_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                0, s_tile_global_offset // H, s_tile_global_offset // H + s_tile_sz
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
                        H_TILE_SIZE = nl.tile_size.psum_fmax
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
                        # 1-byte dtype (fp8) PE transpose requires a step size of 2 in the free dim.
                        psum_step_size = 2 if _is_fp8_input else 1
                        psum_bank_size = _get_psum_bank_size()
                        input_transposed_psum = []
                        for bank_id in range(NUM_HW_PSUM_BANKS):
                            # Note: PSUM tensors do not have "use_auto_allocation" flag like SbufManager to ignore the allocation.
                            if cfg.use_auto_allocation:
                                input_transposed_psum.append(
                                    nl.ndarray(
                                        (nl.tile_size.pmax, nl.tile_size.psum_fmax, psum_step_size),
                                        dtype=cfg.psum_transpose_dtype,
                                        buffer=nl.psum,
                                    )
                                )
                            else:
                                input_transposed_psum.append(
                                    nl.ndarray(
                                        (nl.tile_size.pmax, nl.tile_size.psum_fmax, psum_step_size),
                                        dtype=cfg.psum_transpose_dtype,
                                        buffer=nl.psum,
                                        address=(0, bank_id * psum_bank_size),
                                    )
                                )

                        tp_psum_bank_idx = (i_tile_S * dims.num_512_tiles_per_H + i_tile_H) % NUM_HW_PSUM_BANKS

                        # (Transpose) nisa.nc_matmul(...) returns result of [128, 128] shape, we need FOUR of these for a single [128, 512] tile.
                        num_128_subtiles_per_H_tile = math.ceil(h_tile_sz / 128)  # At most FMAX/PMAX = 4
                        for j_subtile_H in nl.affine_range(num_128_subtiles_per_H_tile):
                            H_SUBTILE_SIZE = nl.tile_size.pmax
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
                                    0,
                                ],
                            )

                        # Copy transposed [128 (H), 512 (S)] tile from PSUM -> SBUF, and apply gamma_weights.
                        if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
                            # Note: NormType.RMS_NORM_SKIP_GAMMA skips this step.

                            # -- INPUT_TRANSPOSED * GAMMA_WEIGHTS elementwise-multiply ---
                            # Multiply each PSUM sub-tile [128(H), 128 (S)] with [128 (H), 1] column of gamma (with offsets).
                            # Recall that gamma_weights_hbm (H) was broadcasted to [128, H // nl.tile_size.pmax].
                            for j_subtile_H in nl.affine_range(num_128_subtiles_per_H_tile):
                                h_subtile_offset_sbuf = h_tile_offset + j_subtile_H * nl.tile_size.pmax
                                h_subtile_sz = min(nl.tile_size.pmax, H - h_subtile_offset_sbuf)

                                s_tile_offset_psum = j_subtile_H * nl.tile_size.pmax  # src
                                s_tile_offset_sbuf = h_subtile_offset_sbuf  # dst

                                # Index the right column of gamma_norm_weights_sb.
                                # i_tile_H is current 512th tile of H, we need current 128th tile of H for gamma column index.
                                gamma_tile_index = (
                                    i_tile_H * (nl.tile_size.psum_fmax // nl.tile_size.pmax) + j_subtile_H
                                )
                                if not cfg.add_layer_norm_bias:  # Multiply gamma weights only.
                                    nisa.tensor_scalar(
                                        dst=input_sb[i_tile_S][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_sbuf, s_tile_sz),
                                        ],
                                        data=input_transposed_psum[tp_psum_bank_idx][
                                            0:h_subtile_sz,
                                            nl.ds(s_tile_offset_psum, s_tile_sz),
                                            0,
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
                                            0,
                                        ],
                                        op0=nl.multiply,
                                        operand0=gamma_norm_weights_sb[0:h_subtile_sz, nl.ds(gamma_tile_index, 1)],
                                        op1=nl.add,
                                        operand1=layer_norm_bias_sb[0:h_subtile_sz, nl.ds(beta_tile_index, 1)],
                                    )

                                # Input quantization after norm
                                if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                                    nisa.tensor_scalar(
                                        dst=input_quantized_sb[i_tile_S][
                                            0:h_subtile_sz, nl.ds(s_tile_offset_sbuf, s_tile_sz)
                                        ],
                                        data=input_sb[i_tile_S][0:h_subtile_sz, nl.ds(s_tile_offset_sbuf, s_tile_sz)],
                                        op0=nl.multiply,
                                        operand0=in_scale_tile,
                                    )

                                    fp8_max_val = _get_fp8_e4m3_max_pos_val()
                                    nisa.tensor_scalar(
                                        dst=input_quantized_sb[i_tile_S][
                                            0:h_subtile_sz, nl.ds(s_tile_offset_sbuf, s_tile_sz)
                                        ],
                                        data=input_quantized_sb[i_tile_S][
                                            0:h_subtile_sz, nl.ds(s_tile_offset_sbuf, s_tile_sz)
                                        ],
                                        op0=nl.minimum,
                                        operand0=fp8_max_val,
                                        op1=nl.maximum,
                                        operand1=-fp8_max_val,
                                    )

                        else:  # In NO_NORM, RMS_NORM_SKIP_GAMMA, we just copy the transposed PSUM tile to SBUF.
                            # We copy [128, 512] elements in single tensor copy for performance.
                            # -> for S % 128 !=0, we'll copy some garbage memory (to be masked later).
                            # Otherwise, we need extra nl.tile_size.psum_fmax/nl.tile_size.pmax loop here.
                            nisa.tensor_copy(
                                dst=input_sb[i_tile_S][0 : nl.tile_size.pmax, nl.ds(h_tile_offset, h_tile_sz)],
                                src=input_transposed_psum[tp_psum_bank_idx][0 : nl.tile_size.pmax, 0:h_tile_sz, 0],
                            )

                            if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                                if _is_fp8_input:
                                    input_quantized_sb = input_sb
                                else:
                                    # perform quantization for the input
                                    # input_quantized = clip(input / in_scale_tile, _FP8_E4M3_MAX_POS_VAL, -_FP8_E4M3_MAX_POS_VAL)
                                    nisa.tensor_scalar(
                                        dst=input_quantized_sb[i_tile_S][
                                            0 : nl.tile_size.pmax, nl.ds(h_tile_offset, h_tile_sz)
                                        ],
                                        data=input_transposed_psum[tp_psum_bank_idx][
                                            0 : nl.tile_size.pmax, 0:h_tile_sz, 0
                                        ],
                                        op0=nl.multiply,
                                        operand0=in_scale_tile,
                                    )

                                    fp8_max_val = _get_fp8_e4m3_max_pos_val()
                                    nisa.tensor_scalar(
                                        dst=input_quantized_sb[i_tile_S][
                                            0 : nl.tile_size.pmax, nl.ds(h_tile_offset, h_tile_sz)
                                        ],
                                        data=input_quantized_sb[i_tile_S][
                                            0 : nl.tile_size.pmax, nl.ds(h_tile_offset, h_tile_sz)
                                        ],
                                        op0=nl.minimum,
                                        operand0=fp8_max_val,
                                        op1=nl.maximum,
                                        operand1=-fp8_max_val,
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
            psum_bank_size = _get_psum_bank_size()
            qkv_MM_num_psum_banks_needed = dims.num_512_tiles_per_I * num_S_tiles_in_block
            qkv_MM_output_psum = []
            for bank_id in nl.affine_range(qkv_MM_num_psum_banks_needed):
                if cfg.use_auto_allocation:
                    qkv_MM_output_psum.append(
                        nl.ndarray((nl.tile_size.pmax, nl.tile_size.psum_fmax), dtype=nl.float32, buffer=nl.psum)
                    )
                else:
                    qkv_MM_output_psum.append(
                        nl.ndarray(
                            (nl.tile_size.pmax, nl.tile_size.psum_fmax),
                            dtype=nl.float32,
                            buffer=nl.psum,
                            address=(0, bank_id * psum_bank_size),
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
                weight_load_block_size_per_H = dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT  # 1024
                num_weight_load_blocks_per_H = math.ceil(H / weight_load_block_size_per_H)
                # Reduce weight buffers when SBUF budget is exceeded (mirrors lookahead in _multi_buffering_degree_for_seqlen)
                if projected_sbuf_taken_space_after_multi_buffer > cfg.total_available_sbuf_space_to_this_kernel:
                    num_weight_buffers = min(dims.NUM_WEIGHT_BUFFERS_DEFAULT, num_weight_load_blocks_per_H)
                else:
                    num_weight_buffers = dims.NUM_WEIGHT_BUFFERS_DEFAULT  # 4
                max_num_128_H_subtiles_per_weight_block = math.ceil(
                    weight_load_block_size_per_H / 128
                )  # e.g 1024 / 128 = 8.

                # We load weights using a strided access pattern with only a single DMA ISA.
                weights_sb = []
                for _ in range(num_weight_buffers):
                    weights_sb.append(
                        sbm.alloc_stack(
                            (nl.tile_size.pmax, max_num_128_H_subtiles_per_weight_block, I),
                            dtype=cfg.weights_dtype,
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
                        dst=weight_buf.slice(1, 0, curr_num_128_H_subtiles_per_weight_block),
                        src=fused_qkv_weights_hbm.reshape_dim(0, [H // 128, 128])
                        .slice(
                            0,
                            weight_load_offset // 128,
                            weight_load_offset // 128 + curr_num_128_H_subtiles_per_weight_block,
                        )
                        .slice(1, 0, min(nl.tile_size.pmax, H - weight_load_offset))
                        .permute([1, 0, 2]),
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
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)

                    if cfg.load_input_with_DMA_transpose:
                        # Recall we did not use PE Array to transpose input buffer in case of loadWithTranspose.
                        # Use nisa.dma_transpose(...) to load/transpose just enough of input for this round of matmult.
                        # Load/transpose only [128, 1024] elements of input.

                        # NOTE: To drop H divisible by 128 constraint, update AP below with valid "num_h" for the last iteration.
                        if cfg.strided_input_config is None:
                            _dma_xpose_input_non_strided(
                                input_sb,
                                input_hbm,
                                i_tile_S,
                                dims,
                                i_batch,
                                weight_load_offset,
                                s_tile_local_offset,
                                s_tile_sz,
                                curr_num_128_H_subtiles_per_weight_block,
                                H,
                            )
                        elif cfg.strided_input_config.block_len <= nl.tile_size.pmax:
                            _dma_xpose_input_strided_packed(
                                input_sb,
                                input_hbm,
                                i_tile_S,
                                cfg.strided_input_config,
                                dims,
                                i_batch,
                                weight_load_offset,
                                s_tile_local_offset,
                                s_tile_sz,
                                curr_num_128_H_subtiles_per_weight_block,
                                H,
                            )
                        else:
                            _dma_xpose_input_strided_contiguous(
                                input_sb,
                                input_hbm,
                                i_tile_S,
                                cfg.strided_input_config,
                                dims,
                                i_batch,
                                weight_load_offset,
                                s_tile_local_offset,
                                s_tile_sz,
                                curr_num_128_H_subtiles_per_weight_block,
                                H,
                            )

                        if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                            # perform quantization for the input
                            # input_quantized = clip(input / in_scale_tile, _FP8_E4M3_MAX_POS_VAL, -_FP8_E4M3_MAX_POS_VAL)
                            # qkv proj input quantization

                            nisa.tensor_scalar(
                                dst=input_quantized_sb[i_tile_S]
                                .reshape_dim(1, [H // nl.tile_size.pmax, nl.tile_size.pmax])
                                .slice(
                                    1,
                                    weight_load_offset // nl.tile_size.pmax,
                                    weight_load_offset // nl.tile_size.pmax + curr_num_128_H_subtiles_per_weight_block,
                                )
                                .slice(2, 0, s_tile_sz),
                                data=input_sb[i_tile_S]
                                .reshape_dim(1, [H // nl.tile_size.pmax, nl.tile_size.pmax])
                                .slice(
                                    1,
                                    weight_load_offset // nl.tile_size.pmax,
                                    weight_load_offset // nl.tile_size.pmax + curr_num_128_H_subtiles_per_weight_block,
                                )
                                .slice(2, 0, s_tile_sz),
                                op0=nl.multiply,
                                operand0=in_scale_tile,
                            )

                            fp8_max_val = _get_fp8_e4m3_max_pos_val()
                            nisa.tensor_scalar(
                                dst=input_quantized_sb[i_tile_S]
                                .reshape_dim(1, [H // nl.tile_size.pmax, nl.tile_size.pmax])
                                .slice(
                                    1,
                                    weight_load_offset // nl.tile_size.pmax,
                                    weight_load_offset // nl.tile_size.pmax + curr_num_128_H_subtiles_per_weight_block,
                                )
                                .slice(2, 0, s_tile_sz),
                                data=input_quantized_sb[i_tile_S]
                                .reshape_dim(1, [H // nl.tile_size.pmax, nl.tile_size.pmax])
                                .slice(
                                    1,
                                    weight_load_offset // nl.tile_size.pmax,
                                    weight_load_offset // nl.tile_size.pmax + curr_num_128_H_subtiles_per_weight_block,
                                )
                                .slice(2, 0, s_tile_sz),
                                op0=nl.minimum,
                                operand0=fp8_max_val,
                                op1=nl.maximum,
                                operand1=-fp8_max_val,
                            )

                    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                        # NOTE: for fp8 input and weights, we need to run double row mode for matmul
                        curr_num_of_256_H_subtiles_per_weight_block = math.ceil(
                            curr_num_128_H_subtiles_per_weight_block / 2
                        )
                        for j_256_subtile_of_weight_load in nl.affine_range(
                            curr_num_of_256_H_subtiles_per_weight_block
                        ):
                            for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                                # Each nl.tile_size.psum_fmax column of I is accumulated to a different PSUM bank.
                                psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I

                                h_subtile_offset = (
                                    weight_load_block_size_per_H * i_weight_load
                                    + nl.tile_size.pmax * j_256_subtile_of_weight_load
                                )
                                i_offset = nl.tile_size.psum_fmax * k_tile_I

                                h_subtile_sz = min(nl.tile_size.pmax, H - h_subtile_offset)
                                i_tile_sz = min(nl.tile_size.psum_fmax, I - i_offset)

                                input_offset = (
                                    max_num_128_H_subtiles_per_weight_block * i_weight_load
                                    + j_256_subtile_of_weight_load * 2
                                ) * nl.tile_size.pmax
                                weight_offset = j_256_subtile_of_weight_load * 2 * I + k_tile_I * nl.tile_size.psum_fmax
                                # Stationary PSUM tile is input tile:  offset_in_transposed_input_row   + [nl.tile_size.pmax,nl.tile_size.pmax].
                                # Moving PSUM tile is weights_sb tile: offset_in_weights_sbuf           + [nl.tile_size.pmax,nl.tile_size.psum_fmax].

                                nisa.nc_matmul(
                                    stationary=input_quantized_sb[i_tile_S]
                                    .reshape_dim(1, [H // nl.tile_size.pmax, nl.tile_size.pmax])
                                    .slice(0, 0, h_subtile_sz)
                                    .slice(1, input_offset // nl.tile_size.pmax, input_offset // nl.tile_size.pmax + 2)
                                    .slice(2, 0, s_tile_sz),
                                    moving=weights_sb[i_weight_load % num_weight_buffers]
                                    .slice(0, 0, h_subtile_sz)
                                    .slice(1, j_256_subtile_of_weight_load * 2, j_256_subtile_of_weight_load * 2 + 2)
                                    .slice(
                                        2,
                                        k_tile_I * nl.tile_size.psum_fmax,
                                        k_tile_I * nl.tile_size.psum_fmax + i_tile_sz,
                                    ),
                                    dst=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:i_tile_sz],
                                    perf_mode="double_row",
                                )
                    else:
                        for j_128_subtile_of_weight_load in nl.affine_range(curr_num_128_H_subtiles_per_weight_block):
                            for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                                # Each nl.tile_size.psum_fmax column of I is accumulated to a different PSUM bank.
                                psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I

                                h_subtile_offset = (
                                    weight_load_block_size_per_H * i_weight_load
                                    + nl.tile_size.pmax * j_128_subtile_of_weight_load
                                )
                                i_offset = nl.tile_size.psum_fmax * k_tile_I

                                h_subtile_sz = min(nl.tile_size.pmax, H - h_subtile_offset)
                                i_tile_sz = min(nl.tile_size.psum_fmax, I - i_offset)

                                # Stationary PSUM tile is input tile:  offset_in_transposed_input_row   + [nl.tile_size.pmax,nl.tile_size.pmax].
                                # Moving PSUM tile is weights_sb tile: offset_in_weights_sbuf           + [nl.tile_size.pmax,nl.tile_size.psum_fmax].
                                nisa.nc_matmul(
                                    stationary=input_sb[i_tile_S][
                                        0:h_subtile_sz, nl.ds(h_subtile_offset, s_tile_sz)
                                    ],  # Use h_subtile_offset as [S,H]->[H,S] was transposed in-place.
                                    moving=weights_sb[i_weight_load % num_weight_buffers][
                                        0:h_subtile_sz,
                                        j_128_subtile_of_weight_load,
                                        nl.ds(nl.tile_size.psum_fmax * k_tile_I, i_tile_sz),
                                    ],
                                    dst=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:i_tile_sz],
                                )
                # End of i_weight_load loop

            #######################################################################################################
            # Step 5: Copy PSUM results from matmult back to SBUF, and optionally apply fused RoPE.
            #######################################################################################################
            # Store results to SBUF before copying them to HBM output_tensor.

            # We have one matmult result per each 512 tile/column of weights stored in psum_buffer[bank_index].
            for i_tile_S in nl.affine_range(num_S_tiles_in_block):
                s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)

                # Copy results PSUM -> SBUF, apply per-head processing (RoPE and/or QK-norm).
                if cfg.fused_rope or _has_qk_norm:
                    _evict_and_postprocess_per_head(
                        qkv_MM_output_psum=qkv_MM_output_psum,
                        output_sb=output_sb,
                        i_tile_S=i_tile_S,
                        s_tile_sz=s_tile_sz,
                        cfg=cfg,
                        dims=dims,
                        bias_sb=bias_sb if cfg.add_bias else None,
                        w_scale_tile=w_scale_tile,
                        rope_bufs=rope_bufs,
                        i_batch=i_batch,
                        s_tile_local_offset=s_tile_local_offset,
                        qk_norm_bufs=qk_norm_bufs,
                        row_w_scale_sb=row_w_scale_sb,
                    )
                # Copy results PSUM -> SBUF, (optionally) add_bias.
                else:
                    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
                        for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                            psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I
                            num_i = min(nl.tile_size.psum_fmax, I - nl.tile_size.psum_fmax * k_tile_I)

                            nisa.tensor_copy(
                                dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(nl.tile_size.psum_fmax * k_tile_I, num_i)],
                                src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                            )

                        if cfg.add_bias:
                            # dequantize the qkv output
                            nisa.scalar_tensor_tensor(
                                dst=output_sb[i_tile_S][0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                                data=output_sb[i_tile_S][0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 0],
                                op1=nl.add,
                                operand1=bias_sb[0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                            )

                            nisa.scalar_tensor_tensor(
                                dst=output_sb[i_tile_S][
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                                data=output_sb[i_tile_S][
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 1],
                                op1=nl.add,
                                operand1=bias_sb[
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                            )

                            nisa.scalar_tensor_tensor(
                                dst=output_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                                data=output_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 2],
                                op1=nl.add,
                                operand1=bias_sb[
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                            )

                        else:
                            # dequantize the qkv output
                            nisa.tensor_scalar(
                                dst=output_sb[i_tile_S][0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                                data=output_sb[i_tile_S][0:s_tile_sz, : dims.num_q_heads * dims.d_head],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 0],
                            )

                            nisa.tensor_scalar(
                                dst=output_sb[i_tile_S][
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                                data=output_sb[i_tile_S][
                                    0:s_tile_sz, nl.ds(dims.num_q_heads * dims.d_head, dims.num_kv_heads * dims.d_head)
                                ],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 1],
                            )

                            nisa.tensor_scalar(
                                dst=output_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                                data=output_sb[i_tile_S][
                                    0:s_tile_sz,
                                    nl.ds(
                                        (dims.num_q_heads + dims.num_kv_heads) * dims.d_head,
                                        dims.num_kv_heads * dims.d_head,
                                    ),
                                ],
                                op0=nl.multiply,
                                operand0=w_scale_tile[0:s_tile_sz, 2],
                            )

                    else:
                        for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                            psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I
                            num_i = min(nl.tile_size.psum_fmax, I - nl.tile_size.psum_fmax * k_tile_I)

                            if cfg.add_bias:
                                nisa.tensor_tensor(
                                    dst=output_sb[i_tile_S][
                                        0:s_tile_sz, nl.ds(nl.tile_size.psum_fmax * k_tile_I, num_i)
                                    ],
                                    data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                                    data2=bias_sb[0:s_tile_sz, nl.ds(nl.tile_size.psum_fmax * k_tile_I, num_i)],
                                    op=nl.add,
                                )
                            else:
                                nisa.tensor_copy(
                                    dst=output_sb[i_tile_S][
                                        0:s_tile_sz, nl.ds(nl.tile_size.psum_fmax * k_tile_I, num_i)
                                    ],
                                    src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                                )

                        # ROW dequantization: element-wise multiply with per-column weight scale
                        if row_w_scale_sb is not None:
                            nisa.tensor_tensor(
                                dst=output_sb[i_tile_S][0:s_tile_sz, :I],
                                data1=output_sb[i_tile_S][0:s_tile_sz, :I],
                                data2=row_w_scale_sb[0:s_tile_sz, :I],
                                op=nl.multiply,
                            )

                # End of i_tile_S loop.

            #######################################################################################################
            # Step 6: Store SBUF results back to HBM, using given output layout.
            #######################################################################################################
            # This parts reads from output_matmult_sbuf and writes to out_tensor.

            if cfg.use_kv_cache and cfg.output_layout == QKVOutputLayout.BSD:
                # KV cache mode: store Q separately, optionally quantize and store K/V to caches
                fp8_pack_pending_tiles = []  # accumulator for fp8_packed tile pairing
                fp8_pack_first_tile_offset = None
                # Disable two-tile optimization if any tile in this block is partial (< pmax).
                # This avoids mixing different s_tile_sz values in a single paired call.
                fp8_packed_two_tile_opt = cfg.fp8_packed and (s_block_sz % nl.tile_size.pmax == 0)
                # slot_mapping arrives 1-D [B*S] from vLLM. Fold (B,S)->(B*S) only when it is
                # actually 2-D; when use_BxS_input_reshape is off (large-S prefill) it is not
                # pre-reshaped and stays 1-D, so flatten_dims(0,1) would index a missing dim.
                slot_mapping_flat = None
                if slot_mapping_hbm is not None:
                    slot_mapping_flat = (
                        slot_mapping_hbm.flatten_dims(start_dim=0, end_dim=1)
                        if len(slot_mapping_hbm.shape) > 1
                        else slot_mapping_hbm
                    )
                for i_tile_S in range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)
                    s_tile_global_offset = dims.S_shard_offset + s_tile_local_offset

                    # Store Q output to HBM (skip when q_dim == 0, i.e. KV-only projection)
                    if dims.q_dim > 0:
                        nisa.dma_copy(
                            dst=q_tensor_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                0,
                                i_batch * dims.S + s_tile_global_offset,
                                i_batch * dims.S + s_tile_global_offset + s_tile_sz,
                            ),
                            src=output_sb[i_tile_S][0:s_tile_sz, 0 : dims.q_dim],
                            dge_mode=dge_mode.swdge,
                        )

                    # Load slot_mapping for block KV cache
                    if cfg.use_block_kv:
                        slot_mapping_tile_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
                        # slot_mapping shape: (batch, seqlen)
                        # Access continuous elements: slot_mapping[i_batch, s_tile_global_offset:s_tile_global_offset+s_tile_sz]
                        nisa.dma_copy(
                            dst=slot_mapping_tile_sb[0:s_tile_sz, 0:1],
                            src=slot_mapping_flat.slice(
                                0,
                                i_batch * dims.S + s_tile_global_offset,
                                i_batch * dims.S + s_tile_global_offset + s_tile_sz,
                            ),
                            dge_mode=dge_mode.none,  # to remove gpsimd contention with the cache write
                        )
                    if cfg.fp8_packed:
                        # Accumulate tiles for paired processing (full 128-partition transpose+DMA).
                        # Flush when we have 2 tiles, on the last tile, or when two-tile opt is disabled.
                        fp8_pack_pending_tiles.append(output_sb[i_tile_S])
                        if fp8_pack_first_tile_offset is None:
                            fp8_pack_first_tile_offset = s_tile_global_offset
                        if (
                            len(fp8_pack_pending_tiles) == 2
                            or i_tile_S == num_S_tiles_in_block - 1
                            or not fp8_packed_two_tile_opt
                        ):
                            pair_count = len(fp8_pack_pending_tiles)

                            # Load packed slot indices for all tiles in one DMA (stride 2 across
                            # contiguous tiles). Both tiles' even-indexed slots are contiguous in HBM.
                            num_packed_rows = pair_count * (s_tile_sz // 2)
                            packed_slot_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
                            nisa.dma_copy(
                                dst=packed_slot_sb[0:num_packed_rows, 0:1],
                                src=slot_mapping_flat.slice(
                                    0,
                                    i_batch * dims.S + fp8_pack_first_tile_offset,
                                    i_batch * dims.S + fp8_pack_first_tile_offset + 2 * num_packed_rows,
                                    step=2,
                                ),
                                dge_mode=dge_mode.none,  # to remove gpsimd contention with the cache write
                            )
                            # slot_mapping holds flat indices into [num_blocks * block_size].
                            # Right-shift by 1 (= divide by 2) converts to packed row indices
                            # into [num_blocks * block_size // 2], since each packed row holds
                            # two consecutive sequence positions. This yields
                            # block * block_size_half + pair.
                            nisa.tensor_scalar(
                                dst=packed_slot_sb[0:num_packed_rows, 0:1],
                                data=packed_slot_sb[0:num_packed_rows, 0:1],
                                op0=nl.right_shift,
                                operand0=1,
                            )
                            # Head-major cache layout [num_blocks, num_kv_heads, block_size//2, d_head, 2]:
                            # the row for (block, head, pair) is
                            #   block * num_kv_heads * block_size_half + head * block_size_half + pair.
                            # The per-head term is added via a static offset inside the store fn; here we
                            # promote the block term's stride from block_size_half to
                            # num_kv_heads * block_size_half by adding block_idx * (num_kv_heads-1) * block_size_half.
                            # No-op when num_kv_heads == 1 (matches the original single-row layout).
                            if dims.num_kv_heads > 1:
                                block_size_half = cfg.block_size // 2
                                log2_block_size_half = int(math.log2(block_size_half))
                                block_idx_sb = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=nl.int32, buffer=nl.sbuf)
                                nisa.tensor_scalar(
                                    dst=block_idx_sb[0:num_packed_rows, 0:1],
                                    data=packed_slot_sb[0:num_packed_rows, 0:1],
                                    op0=nl.right_shift,
                                    operand0=log2_block_size_half,
                                )
                                nisa.scalar_tensor_tensor(
                                    dst=packed_slot_sb[0:num_packed_rows, 0:1],
                                    data=block_idx_sb[0:num_packed_rows, 0:1],
                                    op0=nl.multiply,
                                    operand0=float(block_size_half * (dims.num_kv_heads - 1)),
                                    op1=nl.add,
                                    operand1=packed_slot_sb[0:num_packed_rows, 0:1],
                                )
                            _quantize_and_store_k_fp8_packed(
                                output_sb_tiles=fp8_pack_pending_tiles,
                                inv_scale_sb=k_inv_scale_sb,
                                cache_hbm=k_cache_hbm,
                                kv_offset=dims.q_dim,
                                s_tile_sz=s_tile_sz,
                                cfg=cfg,
                                dims=dims,
                                sbm=sbm,
                                packed_slot_sb=packed_slot_sb,
                            )
                            fp8_pack_pending_tiles = []
                            fp8_pack_first_tile_offset = None
                    elif cfg.transpose_k_cache:
                        _quantize_and_store_k_transposed(
                            output_sb=output_sb[i_tile_S],
                            inv_scale_sb=k_inv_scale_sb,
                            cache_hbm=k_cache_hbm,
                            kv_offset=dims.q_dim,
                            s_tile_sz=s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                            slot_mapping_sb=slot_mapping_tile_sb if cfg.use_block_kv else None,
                            i_batch=i_batch,
                            s_tile_global_offset=s_tile_global_offset,
                        )
                    else:
                        _quantize_and_store_kv(
                            output_sb=output_sb[i_tile_S],
                            scale_sb=k_scale_sb,
                            cache_hbm=k_cache_hbm,
                            kv_offset=dims.q_dim,
                            i_batch=i_batch,
                            s_tile_global_offset=s_tile_global_offset,
                            s_tile_sz=s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                            slot_mapping_sb=slot_mapping_tile_sb if cfg.use_block_kv else None,
                        )

                    _quantize_and_store_kv(
                        output_sb=output_sb[i_tile_S],
                        scale_sb=v_scale_sb,
                        cache_hbm=v_cache_hbm,
                        kv_offset=dims.q_dim + dims.kv_dim,
                        i_batch=i_batch,
                        s_tile_global_offset=s_tile_global_offset,
                        s_tile_sz=s_tile_sz,
                        cfg=cfg,
                        dims=dims,
                        sbm=sbm,
                        slot_mapping_sb=slot_mapping_tile_sb if cfg.use_block_kv else None,
                    )

            elif cfg.output_layout == QKVOutputLayout.BSD:
                # output_tensor shape: [B, S, I].
                # output_matmult_sbuf contains [128 (S), I].
                for i_tile_S in range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)

                    nisa.dma_copy(
                        dst=output_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                            0,
                            i_batch * dims.S + dims.S_shard_offset + s_tile_local_offset,
                            i_batch * dims.S + dims.S_shard_offset + s_tile_local_offset + s_tile_sz,
                        ),
                        src=output_sb[i_tile_S][0:s_tile_sz, 0:I],
                        dge_mode=dge_mode.swdge,
                    )

            else:  # NBSd = [heads, B, S, head_dim], I = heads * head_dim
                d_head = int(dims.d_head)  # Safe due to validation
                for i_head in range(dims.num_heads):
                    for i_tile_S in range(num_S_tiles_in_block):
                        s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                        s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)
                        num_d = min(d_head, I - (i_head * d_head))

                        nbsd_row = (
                            i_head * dims.B * dims.S + i_batch * dims.S + dims.S_shard_offset + s_tile_local_offset
                        )
                        nisa.dma_copy(
                            dst=output_hbm.flatten_dims(start_dim=0, end_dim=2).slice(
                                0, nbsd_row, nbsd_row + s_tile_sz
                            ),
                            src=output_sb[i_tile_S]
                            .slice(0, 0, s_tile_sz)
                            .slice(1, i_head * d_head, i_head * d_head + num_d),
                            dge_mode=dge_mode.swdge,
                        )

            # Optional: per-segment sum of squares over the head dimension,
            # computed inline while the QKV tile is still resident in SBUF.
            # output_sb[i_tile_S] holds [s_tile_sz, I] = [Q|K|V] for both the BSD
            # and NBSd layouts (the layout only changes the HBM store above), so
            # this is layout-independent. Skipped on the KV-cache path, where
            # output_sb may be quantized in place by the cache store.
            _want_squared_sum = (
                q_squared_sum_hbm is not None or k_squared_sum_hbm is not None or v_squared_sum_hbm is not None
            )
            if _want_squared_sum and not cfg.use_kv_cache:
                for i_tile_S in range(num_S_tiles_in_block):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * nl.tile_size.pmax
                    s_tile_sz = min(nl.tile_size.pmax, S_shard - s_tile_local_offset)
                    s_tile_global_offset = dims.S_shard_offset + s_tile_local_offset
                    # Flat element offset into the [B, S, 1] output viewed as [B * S, 1].
                    # Folding i_batch in here keeps the destination a plain numpy slice
                    # that stays in bounds even when B*S is folded (use_BxS_input_reshape).
                    sq_flat_offset = i_batch * dims.S + s_tile_global_offset
                    if q_squared_sum_hbm is not None:
                        q_sq_scratch_sb = sbm.alloc_stack(
                            (nl.tile_size.pmax, dims.q_dim), dtype=nl.float32, buffer=nl.sbuf
                        )
                        q_sq_reduced_sb = sbm.alloc_stack(
                            (nl.tile_size.pmax, 1), dtype=q_squared_sum_hbm.dtype, buffer=nl.sbuf
                        )
                        q_sq_flat = q_squared_sum_hbm.reshape((dims.B * dims.S, 1))
                        _compute_and_store_squared_sum(
                            segment_sb=output_sb[i_tile_S][0:s_tile_sz, nl.ds(0, dims.q_dim)],
                            squared_scratch_sb=q_sq_scratch_sb[0:s_tile_sz, 0 : dims.q_dim],
                            squared_sum_sb=q_sq_reduced_sb[0:s_tile_sz, 0:1],
                            squared_sum_dst=q_sq_flat[sq_flat_offset : sq_flat_offset + s_tile_sz, :],
                        )
                    if k_squared_sum_hbm is not None:
                        k_sq_scratch_sb = sbm.alloc_stack(
                            (nl.tile_size.pmax, dims.kv_dim), dtype=nl.float32, buffer=nl.sbuf
                        )
                        k_sq_reduced_sb = sbm.alloc_stack(
                            (nl.tile_size.pmax, 1), dtype=k_squared_sum_hbm.dtype, buffer=nl.sbuf
                        )
                        k_sq_flat = k_squared_sum_hbm.reshape((dims.B * dims.S, 1))
                        _compute_and_store_squared_sum(
                            segment_sb=output_sb[i_tile_S][0:s_tile_sz, nl.ds(dims.q_dim, dims.kv_dim)],
                            squared_scratch_sb=k_sq_scratch_sb[0:s_tile_sz, 0 : dims.kv_dim],
                            squared_sum_sb=k_sq_reduced_sb[0:s_tile_sz, 0:1],
                            squared_sum_dst=k_sq_flat[sq_flat_offset : sq_flat_offset + s_tile_sz, :],
                        )
                    if v_squared_sum_hbm is not None:
                        v_sq_scratch_sb = sbm.alloc_stack(
                            (nl.tile_size.pmax, dims.kv_dim), dtype=nl.float32, buffer=nl.sbuf
                        )
                        v_sq_reduced_sb = sbm.alloc_stack(
                            (nl.tile_size.pmax, 1), dtype=v_squared_sum_hbm.dtype, buffer=nl.sbuf
                        )
                        v_sq_flat = v_squared_sum_hbm.reshape((dims.B * dims.S, 1))
                        _compute_and_store_squared_sum(
                            segment_sb=output_sb[i_tile_S][0:s_tile_sz, nl.ds(dims.q_dim + dims.kv_dim, dims.kv_dim)],
                            squared_scratch_sb=v_sq_scratch_sb[0:s_tile_sz, 0 : dims.kv_dim],
                            squared_sum_sb=v_sq_reduced_sb[0:s_tile_sz, 0:1],
                            squared_sum_dst=v_sq_flat[sq_flat_offset : sq_flat_offset + s_tile_sz, :],
                        )

            sbm.close_scope()  # Deallocate all multi-buffered tensors.
            # End of i_buffer_s loop
        # End of batch loop
    sbm.close_scope()
    return output_hbm


def _qkv_cte_mx_impl(
    input_hbm: nl.NkiTensor,
    fused_qkv_weights_hbm: nl.NkiTensor,
    output_hbm: nl.NkiTensor,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    bias_hbm: Optional[nl.NkiTensor] = None,
    mlp_prev_hbm: Optional[nl.NkiTensor] = None,
    attention_prev_hbm: Optional[nl.NkiTensor] = None,
    gamma_norm_weights_hbm: Optional[nl.NkiTensor] = None,
    layer_norm_bias_hbm: Optional[nl.NkiTensor] = None,
    norm_eps: Optional[float] = 1e-6,
    cos_cache_hbm: Optional[nl.NkiTensor] = None,
    sin_cache_hbm: Optional[nl.NkiTensor] = None,
    qkv_w_scale: Optional[nl.NkiTensor] = None,
    qkv_in_scale: Optional[nl.NkiTensor] = None,
    k_cos_cache_hbm: Optional[nl.NkiTensor] = None,
    k_sin_cache_hbm: Optional[nl.NkiTensor] = None,
) -> nl.NkiTensor:
    """
    MX Quantization implementation of QKV CTE kernel.

    Performs QKV projection using MX (Microscaling) quantized weights and activations.
    Supports optional fused operations including normalization, residual addition,
    bias, and RoPE. Uses nisa.quantize_mx for activation quantization and
    nisa.nc_matmul_mx for the matrix multiplication.

    Optionally supports static-quant-via-MX: when a statically-quantized FP8 model is routed
    through the MX engine for quadrow performance, per-tensor dequantization scales
    (qkv_in_scale and qkv_w_scale) are applied post-matmul.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        I: Fused QKV dimension = (num_q_heads + 2*num_kv_heads) * d_head
        H_128_tiles: Number of 128-element tiles in H dimension (H // 512)
        H_pack: Packing factor for MX format (4)

    Args:
        input_hbm (nl.NkiTensor): [B, S, H], Input hidden states tensor on HBM
        fused_qkv_weights_hbm (nl.NkiTensor): [H//4, I, 4] fp8, MX-quantized fused QKV weights on HBM
        output_hbm (nl.NkiTensor): Output tensor on HBM, shape depends on cfg.output_layout
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager
        bias_hbm (Optional[nl.NkiTensor]): [1, I], Optional bias tensor on HBM
        mlp_prev_hbm (Optional[nl.NkiTensor]): [B, S, H], Optional MLP residual on HBM
        attention_prev_hbm (Optional[nl.NkiTensor]): [B, S, H], Optional attention residual on HBM
        gamma_norm_weights_hbm (Optional[nl.NkiTensor]): [1, H], Optional normalization weights on HBM
        layer_norm_bias_hbm (Optional[nl.NkiTensor]): [1, H], Optional layer norm bias on HBM
        norm_eps (Optional[float]): Epsilon for normalization stability
        cos_cache_hbm (Optional[nl.NkiTensor]): [B, S, d_head], Optional RoPE cosine cache on HBM
        sin_cache_hbm (Optional[nl.NkiTensor]): [B, S, d_head], Optional RoPE sine cache on HBM
        qkv_w_scale (Optional[nl.NkiTensor]): MX weight scales on HBM. Interpretation
            depends on the quantization path:
            - MX per-block scales: [H//32, I], used in nc_matmul_mx for per-block dequantization.
              If None, uses neutral scaling (MX_NEUTRAL_SCALE=127, i.e. 2^0=1.0).
            - MX static dequant: [1, 3] or [128, 3], per-tensor weight dequantization scale
              (one per Q/K/V), applied post-matmul. Used when qkv_in_scale is also provided.
        qkv_in_scale (Optional[nl.NkiTensor]): [1, 1] or [128, 1], Per-tensor input dequantization
            scale for static-quant FP8 models routed through MX engine. When provided,
            qkv_w_scale is interpreted as per-tensor dequant w_scale ([1,3] or [128,3])
            and the combined scale (in_scale * w_scale) is applied post-matmul.

    Returns:
        nl.NkiTensor: Output tensor (same as output_hbm parameter)

    Notes:
        - Input can be pre-swizzled (cfg.is_input_swizzled) for optimized loading
        - Uses S_TILE_SIZE=32 for swizzled input, P_MAX=128 otherwise
        - Weight prefetching is used when SBUF space allows
        - Supports both BSD and NBSd output layouts

    Pseudocode:
        # Step 1: Load input and apply optional normalization
        for each S_tile:
            load input_tile from HBM
            if fused_residual_add:
                input_tile += mlp_prev + attention_prev
            if fused_norm:
                input_tile = normalize(input_tile)

        # Step 2: Transpose and swizzle for MX format
        transposed = transpose_for_mx(input_tiles)

        # Step 3: Quantize activations to MX format
        hidden_qtz, hidden_scale = quantize_mx(transposed)

        # Step 4: MX matrix multiplication
        for each weight_block:
            output_psum += nc_matmul_mx(hidden_qtz, weights, scales)

        # Step 5: Apply optional RoPE and bias, copy to output
        for each S_tile:
            if fused_rope:
                apply_rope(output_tile)
            if add_bias:
                output_tile += bias
            store output_tile to HBM
    """

    S_shard = dims.S_shard
    H = dims.H
    H_PACKED = H // 4  # MX packs 4 elements, so weight H dimension is H/4
    I = dims.I

    if dims.S_shard == 0:
        return output_hbm

    P_MAX = nl.tile_size.pmax
    F_MAX = 512
    S_TILE_SIZE = 32 if cfg.is_input_swizzled else P_MAX
    PSUM_BANK_SIZE = _get_psum_bank_size()

    if sbm is None:
        sbm_logger = get_logger("qkv_cte")
        sbm = SbufManager(
            sb_lower_bound=0,
            sb_upper_bound=cfg.total_available_sbuf_space_to_this_kernel,
            use_auto_alloc=cfg.use_auto_allocation,
            logger=sbm_logger,
        )
    sbm.open_scope()

    if cfg.add_bias:
        bias_sb = _load_and_broadcast_bias(bias_hbm=bias_hbm, cfg=cfg, dims=dims, sbm=sbm)

    # MX constants
    SCALE_P_PER_QUAD = H_pack = 4
    H_128_tiles = div_ceil(H, P_MAX * H_pack)

    # Load norm weights only for non-swizzled path
    if not cfg.is_input_swizzled:
        zero_bias_sb = sbm.alloc_stack((P_MAX, 1), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
        nisa.memset(dst=zero_bias_sb, value=0)

        norm_eps_sb = _make_activation_constant_bias(norm_eps, sbm, P_MAX, dtype=cfg.compute_mm_dtype)
    else:
        input_view = input_hbm.reshape((dims.B, dims.S * H_pack, H_128_tiles, P_MAX))

    _is_fp8_input = input_hbm.dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
    _use_static_dequant = cfg.quantization_config.has_mx_static_dequant_scales
    _is_bf16_with_static_quant = input_hbm.dtype == nl.bfloat16 and _use_static_dequant
    _is_row_mx = cfg.quantization_config.has_row_mx_dequant
    _is_mx_native_row_input = cfg.quantization_config.has_mx_native_row_input_dequant
    _use_dma_xpose_mx_path = (
        (_is_fp8_input or _is_bf16_with_static_quant)
        and cfg.load_input_with_DMA_transpose
        and not _is_row_mx
        and not _is_mx_native_row_input
    )

    # When on static dequant path, qkv_w_scale carries the per-tensor dequant w_scale
    # ([1,3] or [128,3]), not per-block MX scales. Extract it and clear so per-block logic sees None.
    mx_static_dequant_w_scale = None
    if _use_static_dequant:
        mx_static_dequant_w_scale = qkv_w_scale
        qkv_w_scale = None

    # ROW_MX: extract per-channel weight scale and clear qkv_w_scale so per-block MX logic sees None.
    row_mx_w_channel_scale = None
    if _is_row_mx:
        row_mx_w_channel_scale = qkv_w_scale
        qkv_w_scale = None

    # DMA transpose path: pack pairs of elements into a wider type for x4-interleaved layout.
    # FP8 (1B)→FP16 (2B), BF16 (2B)→FP32 (4B); reshape [B, S, H] → [B*S*2, H//2] viewed as wider type.
    if _use_dma_xpose_mx_path:
        _dma_xpose_packed_2d = input_hbm.reshape((dims.B * dims.S * 2, H // 2))
        _dma_xpose_view_dtype = nl.float32 if _is_bf16_with_static_quant else nl.float16

    # Precompute DMA transpose parameters for the shared helper.
    # ROW_MX / native MX row input: float32 AP directly on [B, S, H+4], no reshape, seq_mult=1.
    # Existing MX: fp16/fp32 AP on reshaped [B*S*2, H//2], seq_mult=2.
    if _is_row_mx or _is_mx_native_row_input:
        _xpose_src = input_hbm
        _xpose_row_stride = (H + ROW_MX_TAIL_SCALE_BYTES) // 4
        _xpose_seq_mult = 1
        _xpose_dtype = nl.float32
    elif _use_dma_xpose_mx_path:
        if cfg.weight_layout == QKVWeightLayout.MX_INTERLEAVED:
            _xpose_src = _dma_xpose_packed_2d
            _xpose_row_stride = H_PACKED
            _xpose_seq_mult = 2
            _xpose_dtype = _dma_xpose_view_dtype
        elif cfg.weight_layout == QKVWeightLayout.MX_CONTIGUOUS:
            _xpose_src = input_hbm
            _xpose_row_stride = H_PACKED
            _xpose_seq_mult = 1
            _xpose_dtype = nl.float32

    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        gamma_norm_weights_sb = _load_norm_weights_mx(
            norm_weights_hbm=gamma_norm_weights_hbm, cfg=cfg, dims=dims, sbm=sbm
        )
        if cfg.add_layer_norm_bias:
            layer_norm_bias_sb = _load_norm_weights_mx(
                norm_weights_hbm=layer_norm_bias_hbm, cfg=cfg, dims=dims, sbm=sbm
            )

    # Replace the manual multi-buffering calculation with:
    s_multi_buffer_degree, projected_sbuf_taken_space = _multi_buffering_degree_for_seqlen_mx(
        cfg=cfg, dims=dims, sbm=sbm
    )

    S_BLOCK_SIZE = s_multi_buffer_degree * min(S_shard, P_MAX)
    num_blocks_per_S_shard = math.ceil(S_shard / S_BLOCK_SIZE) if S_BLOCK_SIZE > 0 else 0

    # Weight prefetch decision
    use_weight_prefetch_mx = _use_weight_prefetch_mx(projected_sbuf_taken_space, cfg=cfg, dims=dims)

    # Pre-allocate neutral MX scale tile: used wherever scales are all 1.0 (exponent=0)
    _needs_neutral_scale = _use_dma_xpose_mx_path or _is_mx_native_row_input or qkv_w_scale is None
    if _needs_neutral_scale:
        neutral_scale_sb = sbm.alloc_stack((P_MAX, 1, F_MAX), dtype=nl.uint8, buffer=nl.sbuf)
        nisa.memset(dst=neutral_scale_sb, value=MX_NEUTRAL_SCALE)

    # ROW_MX: load per-channel weight scale [P_MAX, I] float32 into SBUF (global allocation)
    row_mx_w_channel_scale_sb = None
    if _is_row_mx:
        row_mx_w_channel_scale_sb = sbm.alloc_stack((P_MAX, I), dtype=nl.float32, buffer=nl.sbuf)
        if row_mx_w_channel_scale.shape[0] == 1:
            nisa.dma_copy(dst=row_mx_w_channel_scale_sb[0:1, 0:I], src=row_mx_w_channel_scale[0:1, 0:I])
            stream_shuffle_broadcast(row_mx_w_channel_scale_sb, row_mx_w_channel_scale_sb)
        else:
            nisa.dma_copy(dst=row_mx_w_channel_scale_sb, src=row_mx_w_channel_scale)

    if use_weight_prefetch_mx:
        num_weight_load_blocks_mx = H_128_tiles
        h_tiles_per_block = H_128_tiles

        H_128_tiles_full = H // (P_MAX * H_pack)  # Count of complete 512-element tiles
        h_pack_last = (H % (P_MAX * H_pack)) // P_MAX if H % (P_MAX * H_pack) != 0 else H_pack

        weight_scale_sb = []
        if qkv_w_scale is not None:
            mx_weight_scale_sb = sbm.alloc_stack((P_MAX, H_128_tiles, I), dtype=nl.uint8, buffer=nl.sbuf)
            # Full tiles: all 4 quads
            for h_tile_idx in nl.affine_range(H_128_tiles_full):
                for quad_idx in nl.affine_range(4):
                    hbm_row_offset = (h_tile_idx * 16 + quad_idx * SCALE_P_PER_QUAD) * I
                    nisa.dma_copy(
                        dst=mx_weight_scale_sb[nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), h_tile_idx, :],
                        src=qkv_w_scale.view(nl.uint8).slice(
                            0, hbm_row_offset // I, hbm_row_offset // I + SCALE_P_PER_QUAD
                        ),
                        dge_mode=dge_mode.hwdge,
                    )
            # Partial last tile (if exists)
            _is_remainder = H_128_tiles > H_128_tiles_full
            if _is_remainder:
                for quad_idx in nl.affine_range(h_pack_last):
                    hbm_row_offset = (H_128_tiles_full * 16 + quad_idx * SCALE_P_PER_QUAD) * I
                    nisa.dma_copy(
                        dst=mx_weight_scale_sb[nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), H_128_tiles_full, :],
                        src=qkv_w_scale.view(nl.uint8).slice(
                            0, hbm_row_offset // I, hbm_row_offset // I + SCALE_P_PER_QUAD
                        ),
                        dge_mode=dge_mode.hwdge,
                    )
            weight_scale_sb.append(mx_weight_scale_sb)

        # Load MX weights from HBM [H//4, I, 4] fp8 into SBUF.
        # Allocate SBUF in HBM fp8 dtype for HWDGE type matching, then
        # view-cast to float8_e4m3fn_x4 for nc_matmul_mx.
        _hbm_weight_dtype = fused_qkv_weights_hbm.dtype
        weights_sb = []
        mx_weights_sb = sbm.alloc_stack(
            (P_MAX, dims.num_512_tiles_per_H, I * 4),
            dtype=_hbm_weight_dtype,
            align=4,
            buffer=nl.sbuf,
        )
        for h_tile_idx in nl.affine_range(H_128_tiles):
            h_tile_sz = min(P_MAX, H_PACKED - h_tile_idx * P_MAX)
            nisa.dma_copy(
                dst=mx_weights_sb[0:h_tile_sz, h_tile_idx, 0 : I * 4],
                src=fused_qkv_weights_hbm.flatten_dims(start_dim=1, end_dim=2).slice(
                    0, h_tile_idx * P_MAX, h_tile_idx * P_MAX + h_tile_sz
                ),
                dge_mode=dge_mode.hwdge,
            )
        mx_weights_sb = mx_weights_sb.view(nl.float8_e4m3fn_x4)
        weights_sb.append(mx_weights_sb)
    else:
        # Chunked loading: allocate smaller buffers in HBM fp8 dtype for
        # HWDGE type matching. View-cast to x4 after each DMA for matmul.
        num_weight_load_blocks_mx = H_128_tiles  # Load 1 H_128_tile at a time
        h_tiles_per_block = 1

        _hbm_weight_dtype = fused_qkv_weights_hbm.dtype
        weight_scale_sb = []
        weights_sb = []
        for _ in range(NUM_MX_WEIGHT_BUFFERS):
            if qkv_w_scale is not None:
                weight_scale_sb.append(sbm.alloc_stack((P_MAX, h_tiles_per_block, I), dtype=nl.uint8, buffer=nl.sbuf))
            weights_sb.append(
                sbm.alloc_stack(
                    (P_MAX, h_tiles_per_block, I * 4),
                    dtype=_hbm_weight_dtype,
                    align=4,
                    buffer=nl.sbuf,
                )
            )

    # Load and pre-fuse MX static dequantization scales: combined = w_scale * in_scale
    mx_dequant_sb = None
    if _use_static_dequant:
        mx_dequant_sb = sbm.alloc_stack(shape=(P_MAX, 3), dtype=qkv_in_scale.dtype, buffer=nl.sbuf)
        if mx_static_dequant_w_scale.shape[0] == 1:
            nisa.dma_copy(dst=mx_dequant_sb[0, :], src=mx_static_dequant_w_scale[0, :])
            stream_shuffle_broadcast(mx_dequant_sb, mx_dequant_sb)
        else:
            nisa.dma_copy(dst=mx_dequant_sb, src=mx_static_dequant_w_scale)

        in_scale_sb = sbm.alloc_stack(shape=(P_MAX, 1), dtype=qkv_in_scale.dtype, buffer=nl.sbuf)
        if qkv_in_scale.shape[0] == 1:
            nisa.dma_copy(dst=in_scale_sb[0, :], src=qkv_in_scale[0, :])
            stream_shuffle_broadcast(in_scale_sb, in_scale_sb)
        else:
            nisa.dma_copy(dst=in_scale_sb, src=qkv_in_scale)

        # Pre-fuse: mx_dequant_sb = w_scale * in_scale (single post-matmul multiply)
        nisa.activation(dst=mx_dequant_sb, op=nl.copy, data=mx_dequant_sb, scale=in_scale_sb)

        # BF16 path needs 1/in_scale for static quantization before quantize_mx
        inv_in_scale_sb = None
        if _is_bf16_with_static_quant:
            inv_in_scale_sb = sbm.alloc_stack(shape=(P_MAX, 1), dtype=qkv_in_scale.dtype, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv_in_scale_sb, data=in_scale_sb)
    # Native MX with row-quantized FP8 input: per-row scale will be extracted per-tile
    # in the main loop (same as ROW_MX). No global scale buffer needed here.

    # QK-norm buffers: gamma weights [pmax, d_head] broadcast via stream_shuffle + scratch [pmax, 1]
    qk_norm_bufs = _allocate_qk_norm_buffers(cfg, dims, sbm, P_MAX)
    _has_qk_norm = qk_norm_bufs is not None

    for i_batch in range(dims.B):
        for i_block_S in nl.affine_range(num_blocks_per_S_shard):
            sbm.open_scope()

            s_block_sz = min(S_BLOCK_SIZE, S_shard - S_BLOCK_SIZE * i_block_S)
            num_S_tiles_in_block = math.ceil(s_block_sz / S_TILE_SIZE)

            row_mx_input_scale_sb = None
            if _is_row_mx or _is_mx_native_row_input or _use_dma_xpose_mx_path:
                # DMA transpose path: allocate buffers, loading happens per-weight-block below
                quant_s_tiles = num_S_tiles_in_block * S_TILE_SIZE
                hidden_qtz_sb = sbm.alloc_stack(
                    (P_MAX, H_128_tiles, quant_s_tiles), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf, align=32
                )

                hidden_scale_sb = None
                # BF16 path: allocate staging buffer for DMA transpose output before quantize_mx
                if _is_bf16_with_static_quant:
                    hidden_scale_sb = sbm.alloc_stack(
                        (P_MAX, H_128_tiles, quant_s_tiles), dtype=nl.uint8, buffer=nl.sbuf
                    )
                    bf16_xpose_staging_sb = sbm.alloc_stack(
                        (P_MAX, H_128_tiles, quant_s_tiles * 2), dtype=nl.float32, buffer=nl.sbuf, align=32
                    )

                # ROW_MX / native MX row input: per-S-tile input row scale [P_MAX, 1] float32
                if _is_row_mx or _is_mx_native_row_input:
                    row_mx_input_scale_sb = []
                    for _ in range(num_S_tiles_in_block):
                        row_mx_input_scale_sb.append(sbm.alloc_stack((P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf))

            _use_bf16_mx_path = not _use_dma_xpose_mx_path and not _is_row_mx and not _is_mx_native_row_input

            # BF16/swizzled path: load → norm → transpose → swizzle → quantize_mx
            if _use_bf16_mx_path:
                # For non-swizzled path, pad H to next multiple of 512 so the transpose
                # loop iterates full H_pack sub-tiles uniformly. Padding is zero-filled.
                H_padded = H_128_tiles * P_MAX * H_pack  # H_padded == H when H % 512 == 0
                input_sb = []
                for _ in range(num_S_tiles_in_block):
                    if cfg.is_input_swizzled:
                        input_sb.append(
                            sbm.alloc_stack((P_MAX, H_128_tiles, P_MAX), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)
                        )
                    else:
                        input_sb.append(sbm.alloc_stack((P_MAX, H_padded), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))

                if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
                    square_sum_sb = []
                    for _ in range(num_S_tiles_in_block):
                        square_sum_sb.append(sbm.alloc_stack((P_MAX, 1), dtype=cfg.act_dtype, buffer=nl.sbuf))
                elif cfg.fused_norm_type == NormType.LAYER_NORM:
                    NUM_AGGR_STATS = 2
                    bn_aggr_result_sb = []
                    for _ in range(num_S_tiles_in_block):
                        bn_aggr_result_sb.append(
                            sbm.alloc_stack((P_MAX, NUM_AGGR_STATS), dtype=cfg.act_dtype, buffer=nl.sbuf)
                        )

                # Step 1: Load input and apply normalization
                for i_tile_S in range(num_S_tiles_in_block):
                    sbm.open_scope()

                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * S_TILE_SIZE
                    s_tile_sz = min(S_TILE_SIZE, S_shard - s_tile_local_offset)

                    if cfg.is_input_swizzled:
                        # Swizzled input loading
                        s_tile_global_offset = (
                            i_batch * dims.S * H_pack + (dims.S_shard_offset + s_tile_local_offset) * H_pack
                        )
                        nisa.dma_copy(
                            dst=input_sb[i_tile_S][0 : s_tile_sz * H_pack, 0:H_128_tiles, 0:P_MAX],
                            src=input_view.flatten_dims(start_dim=0, end_dim=1).slice(
                                0, s_tile_global_offset, s_tile_global_offset + s_tile_sz * H_pack
                            ),
                        )
                    else:
                        if cfg.fused_residual_add:
                            s_tile_global_offset = (
                                i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                            )
                            nisa.dma_compute(
                                dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                                srcs=[
                                    input_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                        0, s_tile_global_offset // H, s_tile_global_offset // H + s_tile_sz
                                    ),
                                    mlp_prev_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                        0, s_tile_global_offset // H, s_tile_global_offset // H + s_tile_sz
                                    ),
                                    attention_prev_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                        0, s_tile_global_offset // H, s_tile_global_offset // H + s_tile_sz
                                    ),
                                ],
                                scales=[1.0, 1.0, 1.0],
                                reduce_op=nl.add,
                            )
                        else:
                            s_tile_global_offset = (
                                i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H
                            )
                            nisa.dma_copy(
                                dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                                src=input_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                                    0, s_tile_global_offset // H, s_tile_global_offset // H + s_tile_sz
                                ),
                                dge_mode=dge_mode.swdge,
                            )

                    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
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
                        _compute_layer_norm_stats(
                            input_sb[i_tile_S],
                            bn_aggr_result_sb[i_tile_S],
                            norm_eps_sb,
                            s_tile_sz,
                            cfg=cfg,
                            dims=dims,
                            sbm=sbm,
                        )

                        # Apply (x - mean) * rvar * gamma [+ beta]
                        mean = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 0:1]
                        rvar = bn_aggr_result_sb[i_tile_S][0:s_tile_sz, 1:2]

                        # (x - mean) * rvar
                        nisa.tensor_scalar(
                            dst=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            data=input_sb[i_tile_S][0:s_tile_sz, 0:H],
                            op0=nl.subtract,
                            operand0=mean,
                            op1=nl.multiply,
                            operand1=rvar,
                        )

                    sbm.close_scope()

                # Step 2: Swizzle + Transpose for MX
                # Allocate transposed buffer with correct layout for quantize_mx
                if cfg.is_input_swizzled:
                    transposed_buffer_sb = sbm.alloc_stack(
                        (P_MAX, H_128_tiles, num_S_tiles_in_block * P_MAX), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf
                    )

                    n_psum_banks = NUM_HW_PSUM_BANKS  # Always use all 8 banks
                    n_transposes_per_bank = math.ceil(H_128_tiles / n_psum_banks)

                    # Each bank holds multiple [128,128] transposes concatenated in free dim
                    transpose_psum = []
                    for bank_id in range(n_psum_banks):
                        transpose_psum.append(
                            nl.ndarray(
                                (P_MAX, n_transposes_per_bank * P_MAX),
                                dtype=nl.bfloat16,
                                buffer=nl.psum,
                                address=(0, bank_id * PSUM_BANK_SIZE),
                            )
                        )

                    for i_tile_S in nl.affine_range(num_S_tiles_in_block):
                        for bank in range(n_psum_banks):
                            # Transpose all H tiles that map to this bank
                            for idx in range(n_transposes_per_bank):
                                h_tile = bank * n_transposes_per_bank + idx
                                if h_tile < H_128_tiles:
                                    nisa.nc_transpose(
                                        data=input_sb[i_tile_S][0:P_MAX, h_tile, 0:P_MAX],
                                        dst=transpose_psum[bank][0:P_MAX, idx * P_MAX : (idx + 1) * P_MAX],
                                    )

                            # Evict full bank - copy all transposes at once
                            for idx in range(n_transposes_per_bank):
                                h_tile = bank * n_transposes_per_bank + idx
                                if h_tile < H_128_tiles:
                                    nisa.tensor_copy(
                                        dst=transposed_buffer_sb[0:P_MAX, h_tile, nl.ds(i_tile_S * P_MAX, P_MAX)],
                                        src=transpose_psum[bank][0:P_MAX, idx * P_MAX : (idx + 1) * P_MAX],
                                    )

                else:
                    transposed_buffer_sb = sbm.alloc_stack(
                        (P_MAX, H_128_tiles, num_S_tiles_in_block * P_MAX * H_pack),
                        dtype=cfg.compute_mm_dtype,
                        buffer=nl.sbuf,
                    )

                    # Pre-allocate PSUM banks ONCE outside all loops
                    transpose_psum = []
                    for bank_id in range(H_pack):
                        if cfg.use_auto_allocation:
                            transpose_psum.append(nl.ndarray((P_MAX, P_MAX), dtype=nl.bfloat16, buffer=nl.psum))
                        else:
                            transpose_psum.append(
                                nl.ndarray(
                                    (P_MAX, P_MAX),
                                    dtype=nl.bfloat16,
                                    buffer=nl.psum,
                                    address=(0, bank_id * PSUM_BANK_SIZE),
                                )
                            )

                    for h_tile in nl.affine_range(H_128_tiles):
                        for s_tile in nl.affine_range(num_S_tiles_in_block):
                            # Transpose all 4 sub-tiles
                            for h_sub in nl.affine_range(H_pack):
                                src_h_base = h_tile * P_MAX * H_pack + h_sub
                                nisa.nc_transpose(
                                    data=input_sb[s_tile]
                                    .reshape_dim(1, [H_padded // H_pack, H_pack])
                                    .slice(1, h_tile * P_MAX, h_tile * P_MAX + P_MAX)[:, :, h_sub],
                                    dst=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                                )

                            # Copy to output with interleaved pattern, optionally applying gamma
                            for h_sub in nl.affine_range(H_pack):
                                out_base = s_tile * P_MAX * H_pack + h_sub
                                dst_ap = (
                                    transposed_buffer_sb.reshape_dim(2, [num_S_tiles_in_block * P_MAX, H_pack])
                                    .slice(1, h_tile, h_tile + 1)
                                    .slice(2, s_tile * P_MAX, s_tile * P_MAX + P_MAX)[:, :, :, h_sub]
                                    .squeeze_dim(1)
                                )

                                if (
                                    cfg.fused_norm_type == NormType.RMS_NORM
                                    or cfg.fused_norm_type == NormType.LAYER_NORM
                                ):
                                    gamma_tile_index = h_tile * H_pack + h_sub
                                    if not cfg.add_layer_norm_bias:
                                        nisa.tensor_scalar(
                                            dst=dst_ap,
                                            data=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                                            op0=nl.multiply,
                                            operand0=gamma_norm_weights_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                                        )
                                    else:
                                        nisa.tensor_scalar(
                                            dst=dst_ap,
                                            data=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                                            op0=nl.multiply,
                                            operand0=gamma_norm_weights_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                                            op1=nl.add,
                                            operand1=layer_norm_bias_sb[0:P_MAX, nl.ds(gamma_tile_index, 1)],
                                        )
                                else:  # NO_NORM or RMS_NORM_SKIP_GAMMA
                                    nisa.tensor_copy(
                                        dst=dst_ap,
                                        src=transpose_psum[h_sub][0:P_MAX, 0:P_MAX],
                                    )

                # Step 3: Quantize using nisa.quantize_mx
                quant_s_tiles = num_S_tiles_in_block * S_TILE_SIZE
                hidden_qtz_sb = sbm.alloc_stack(
                    (P_MAX, H_128_tiles, quant_s_tiles), dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf
                )
                hidden_scale_sb = sbm.alloc_stack((P_MAX, H_128_tiles, quant_s_tiles), dtype=nl.uint8, buffer=nl.sbuf)

                quant_src_size = quant_s_tiles * H_pack
                nisa.quantize_mx(
                    src=transposed_buffer_sb[0:P_MAX, 0:H_128_tiles, 0:quant_src_size],
                    dst=hidden_qtz_sb[0:P_MAX, 0:H_128_tiles, 0:quant_s_tiles],
                    dst_scale=hidden_scale_sb[0:P_MAX, 0:H_128_tiles, 0:quant_s_tiles],
                )

            # RoPE buffers: allocated for both BF16 and FP8 paths (used in Step 5 post-matmul)
            rope_bufs = _allocate_rope_buffers(
                cfg,
                dims,
                sbm,
                P_MAX,
                num_S_tiles_in_block,
                _has_qk_norm,
                cos_cache_hbm,
                sin_cache_hbm,
                k_cos_cache_hbm=k_cos_cache_hbm,
                k_sin_cache_hbm=k_sin_cache_hbm,
            )

            # Step 4: MX Matrix Multiplication
            num_output_s_tiles = max(1, num_S_tiles_in_block // 4) if cfg.is_input_swizzled else num_S_tiles_in_block
            output_sb = []
            for _ in range(num_output_s_tiles):
                output_sb.append(sbm.alloc_stack((P_MAX, I), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf))

            qkv_MM_num_psum_banks_needed = dims.num_512_tiles_per_I * num_output_s_tiles
            qkv_MM_output_psum = []
            for bank_id in nl.affine_range(qkv_MM_num_psum_banks_needed):
                if cfg.use_auto_allocation:
                    qkv_MM_output_psum.append(nl.ndarray((P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum))
                else:
                    qkv_MM_output_psum.append(
                        nl.ndarray(
                            (P_MAX, F_MAX), dtype=nl.float32, buffer=nl.psum, address=(0, bank_id * PSUM_BANK_SIZE)
                        )
                    )

            for i_weight_block in nl.affine_range(num_weight_load_blocks_mx):
                buf_idx = i_weight_block % NUM_MX_WEIGHT_BUFFERS if not use_weight_prefetch_mx else 0
                h_packed_current_tile = min(P_MAX, H_PACKED - i_weight_block * P_MAX)

                if not use_weight_prefetch_mx:
                    # Load this chunk: weights and scales for H_128_tile = i_weight_block
                    h_tile_idx = i_weight_block
                    h_tile_sz = min(P_MAX, H_PACKED - h_tile_idx * P_MAX)
                    num_quads_current_tile = h_packed_current_tile // 32

                    if qkv_w_scale is not None:
                        for quad_idx in nl.affine_range(num_quads_current_tile):
                            hbm_row_offset = (h_tile_idx * 16 + quad_idx * SCALE_P_PER_QUAD) * I
                            nisa.dma_copy(
                                dst=weight_scale_sb[buf_idx][nl.ds(quad_idx * 32, SCALE_P_PER_QUAD), 0, :],
                                src=qkv_w_scale.view(nl.uint8).slice(
                                    0, hbm_row_offset // I, hbm_row_offset // I + SCALE_P_PER_QUAD
                                ),
                                dge_mode=dge_mode.hwdge,
                            )

                    # Load weights for this h_tile from HBM [H//4, I, 4] fp8.
                    # Buffer is allocated in fp8 for HWDGE matching; view-cast
                    # to x4 after load for nc_matmul_mx.
                    weights_sb_fp8 = weights_sb[buf_idx].view(_hbm_weight_dtype)
                    nisa.dma_copy(
                        dst=weights_sb_fp8[0:h_tile_sz, 0, 0 : I * 4],
                        src=fused_qkv_weights_hbm.flatten_dims(start_dim=1, end_dim=2).slice(
                            0, h_tile_idx * P_MAX, h_tile_idx * P_MAX + h_tile_sz
                        ),
                        dge_mode=dge_mode.hwdge,
                    )
                    weights_sb[buf_idx] = weights_sb_fp8.view(nl.float8_e4m3fn_x4)

                # DMA transpose: load one H_128_tile for each S tile
                for i_tile_S in nl.affine_range(num_output_s_tiles):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                    s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                    if _is_row_mx or _is_mx_native_row_input or _use_dma_xpose_mx_path:
                        s_global = dims.S_shard_offset + s_tile_local_offset

                        # FP8: transpose directly into hidden_qtz_sb (no quantize_mx needed)
                        # BF16: transpose into staging buffer (quantize_mx runs after all H tiles)
                        xpose_dst_buf = bf16_xpose_staging_sb if _is_bf16_with_static_quant else hidden_qtz_sb
                        _dma_transpose_mx_tile(
                            src_tensor=_xpose_src,
                            dst_buf=xpose_dst_buf,
                            row_stride=_xpose_row_stride,
                            seq_mult=_xpose_seq_mult,
                            view_dtype=_xpose_dtype,
                            h_packed_current_tile=h_packed_current_tile,
                            i_batch=i_batch,
                            s_global=s_global,
                            i_weight_block=i_weight_block,
                            i_tile_S=i_tile_S,
                            s_tile_sz=s_tile_sz,
                            S_TILE_SIZE=S_TILE_SIZE,
                            quant_s_tiles=quant_s_tiles,
                            H_128_tiles=H_128_tiles,
                            P_MAX=P_MAX,
                            S=dims.S,
                        )

                        # ROW_MX / native MX row input: extract per-row float32 scale from tail (once per S-tile)
                        if (_is_row_mx or _is_mx_native_row_input) and i_weight_block == 0:
                            nisa.dma_copy(
                                dst=row_mx_input_scale_sb[i_tile_S][0:s_tile_sz, 0:1],
                                src=input_hbm.ap(
                                    dtype=nl.float32,
                                    pattern=[[_xpose_row_stride, s_tile_sz], [1, 1]],
                                    offset=i_batch * dims.S * _xpose_row_stride
                                    + s_global * _xpose_row_stride
                                    + _xpose_row_stride
                                    - 1,
                                ),
                            )

                        # BF16: quantize the transposed staging data for this (H_128_tile, S_tile)
                        if _is_bf16_with_static_quant:
                            _static_mx_quantize_bf16_hidden_tile(
                                bf16_xpose_staging_sb=bf16_xpose_staging_sb,
                                hidden_qtz_sb=hidden_qtz_sb,
                                hidden_scale_sb=hidden_scale_sb,
                                inv_in_scale_sb=inv_in_scale_sb,
                                i_weight_block=i_weight_block,
                                i_tile_S=i_tile_S,
                                s_tile_sz=s_tile_sz,
                                quant_s_tiles=quant_s_tiles,
                                H_128_tiles=H_128_tiles,
                                H_pack=H_pack,
                                S_TILE_SIZE=S_TILE_SIZE,
                            )

                    for k_tile_I in nl.affine_range(dims.num_512_tiles_per_I):
                        i_tile_sz = min(F_MAX, I - k_tile_I * F_MAX)
                        psum_accumulation_bank_id = i_tile_S * dims.num_512_tiles_per_I + k_tile_I

                        # Key mapping: h_idx in buffer depends on prefetch mode
                        h_idx_in_buf = i_weight_block if use_weight_prefetch_mx else 0

                        # Split partition dim into valid nc_matmul_mx sizes (32, 64, 128).
                        # E.g. 96 partitions → 64+32 since 96 is not a valid size.
                        partition_starts, partition_sizes = _mx_partition_splits(h_packed_current_tile)
                        for partition_idx in range(len(partition_starts)):
                            partition_start = partition_starts[partition_idx]
                            partition_size = partition_sizes[partition_idx]
                            # Stationary scale: neutral (pre-quantized FP8) or computed (BF16 quantize_mx)
                            if _is_fp8_input:
                                stat_scale = neutral_scale_sb[nl.ds(partition_start, partition_size), 0, 0:s_tile_sz]
                            else:
                                stat_scale = hidden_scale_sb[
                                    nl.ds(partition_start, partition_size),
                                    i_weight_block,
                                    nl.ds(i_tile_S * P_MAX, s_tile_sz),
                                ]

                            # Moving scale: neutral (no weight scales) or loaded from HBM
                            if qkv_w_scale is None:
                                mov_scale = neutral_scale_sb[nl.ds(partition_start, partition_size), 0, 0:i_tile_sz]
                            else:
                                mov_scale = weight_scale_sb[buf_idx][
                                    nl.ds(partition_start, partition_size),
                                    h_idx_in_buf,
                                    nl.ds(k_tile_I * F_MAX, i_tile_sz),
                                ]

                            nisa.nc_matmul_mx(
                                dst=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:i_tile_sz],
                                stationary=hidden_qtz_sb[
                                    nl.ds(partition_start, partition_size),
                                    i_weight_block,
                                    nl.ds(i_tile_S * P_MAX, s_tile_sz),
                                ],
                                moving=weights_sb[buf_idx][
                                    nl.ds(partition_start, partition_size),
                                    h_idx_in_buf,
                                    nl.ds(k_tile_I * F_MAX, i_tile_sz),
                                ],
                                stationary_scale=stat_scale,
                                moving_scale=mov_scale,
                            )

            # Step 5: Copy PSUM to SBUF, apply RoPE/bias
            for i_tile_S in nl.affine_range(num_output_s_tiles):
                s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                _evict_psum_to_sbuf(
                    qkv_MM_output_psum=qkv_MM_output_psum,
                    output_sb=output_sb,
                    i_tile_S=i_tile_S,
                    s_tile_sz=s_tile_sz,
                    cfg=cfg,
                    dims=dims,
                    bias_sb=bias_sb if cfg.add_bias else None,
                    I=I,
                    use_static_dequant=_use_static_dequant,
                    mx_dequant_sb=mx_dequant_sb,
                    rope_bufs=rope_bufs,
                    i_batch=i_batch,
                    s_tile_local_offset=s_tile_local_offset,
                    qk_norm_bufs=qk_norm_bufs,
                    row_mx_input_scale=row_mx_input_scale_sb[i_tile_S] if row_mx_input_scale_sb is not None else None,
                    row_mx_w_channel_scale_sb=row_mx_w_channel_scale_sb,
                )

            # Step 6: Store output to HBM
            if cfg.output_layout == QKVOutputLayout.BSD:
                for i_tile_S in nl.affine_range(num_output_s_tiles):
                    s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                    s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)

                    nisa.dma_copy(
                        dst=output_hbm.flatten_dims(start_dim=0, end_dim=1).slice(
                            0,
                            i_batch * dims.S + dims.S_shard_offset + s_tile_local_offset,
                            i_batch * dims.S + dims.S_shard_offset + s_tile_local_offset + s_tile_sz,
                        ),
                        src=output_sb[i_tile_S][0:s_tile_sz, 0:I],
                        dge_mode=dge_mode.hwdge,
                    )
            else:  # NBSd = [heads, B, S, head_dim]
                d_head = int(dims.d_head)
                for i_head in range(dims.num_heads):
                    for i_tile_S in range(num_output_s_tiles):
                        s_tile_local_offset = i_block_S * S_BLOCK_SIZE + i_tile_S * P_MAX
                        s_tile_sz = min(P_MAX, S_shard - s_tile_local_offset)
                        num_d = min(d_head, I - (i_head * d_head))

                        nbsd_row = (
                            i_head * dims.B * dims.S + i_batch * dims.S + dims.S_shard_offset + s_tile_local_offset
                        )
                        nisa.dma_copy(
                            dst=output_hbm.flatten_dims(start_dim=0, end_dim=2).slice(
                                0, nbsd_row, nbsd_row + s_tile_sz
                            ),
                            src=output_sb[i_tile_S]
                            .slice(0, 0, s_tile_sz)
                            .slice(1, i_head * d_head, i_head * d_head + num_d),
                            dge_mode=dge_mode.hwdge,
                        )
            sbm.close_scope()
    sbm.close_scope()
    return output_hbm


def _dequant_row_mx(
    dst: nl.NkiTensor,
    psum_src: nl.NkiTensor,
    row_scale: nl.NkiTensor,
    channel_scale: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
) -> None:
    """Apply ROW_MX two-scale dequant: dst = psum * row_scale * channel_scale [+ bias].

    Args:
        dst (nl.NkiTensor): Destination slice in SBUF.
        psum_src (nl.NkiTensor): Source slice in PSUM.
        row_scale (nl.NkiTensor): [s_tile_sz, 1] per-row input scale.
        channel_scale (nl.NkiTensor): [s_tile_sz, num_d] per-channel weight scale.
        bias (Optional[nl.NkiTensor]): Bias slice in SBUF, or None.
    """
    nisa.scalar_tensor_tensor(
        dst=dst,
        data=psum_src,
        op0=nl.multiply,
        operand0=row_scale,
        op1=nl.multiply,
        operand1=channel_scale,
    )
    if bias is not None:
        nisa.tensor_tensor(dst=dst, data1=dst, data2=bias, op=nl.add)


def _dequant_row_input_only(
    dst: nl.ndarray,
    psum_src: nl.ndarray,
    row_scale: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
) -> None:
    """Apply row-input-only dequant: dst = psum * row_scale [+ bias].

    Used for native MX weights where per-block weight scales are already applied
    inside nc_matmul_mx; only the per-row input dequant remains.

    Args:
        dst (nl.ndarray): Destination slice in SBUF.
        psum_src (nl.ndarray): Source slice in PSUM.
        row_scale (nl.ndarray): [s_tile_sz, 1] per-row input scale.
        bias (Optional[nl.ndarray]): Bias slice in SBUF, or None.
    """
    if bias is not None:
        nisa.scalar_tensor_tensor(
            dst=dst,
            data=psum_src,
            op0=nl.multiply,
            operand0=row_scale,
            op1=nl.add,
            operand1=bias,
        )
    else:
        nisa.tensor_scalar(dst=dst, data=psum_src, op0=nl.multiply, operand0=row_scale)


def _evict_psum_to_sbuf(
    qkv_MM_output_psum: list,
    output_sb: List[nl.NkiTensor],
    i_tile_S: int,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    bias_sb: Optional[nl.NkiTensor],
    I: int,
    use_static_dequant: bool,
    mx_dequant_sb: Optional[nl.NkiTensor],
    # RoPE buffers (None when fused_rope=False)
    rope_bufs: Optional[RoPEBuffers],
    i_batch: Optional[int],
    s_tile_local_offset: Optional[int],
    # QK-norm buffers (None when qk_norm is disabled)
    qk_norm_bufs: Optional[QKNormBuffers] = None,
    # ROW_MX dequant buffers (None when not ROW_MX)
    row_mx_input_scale: Optional[nl.NkiTensor] = None,
    row_mx_w_channel_scale_sb: Optional[nl.NkiTensor] = None,
) -> None:
    """Evict PSUM matmul results to SBUF, applying optional post-matmul operations.

    Routes to the appropriate path based on which features are enabled:
    - per_head: RoPE and/or QK-norm (Q/K heads processed individually)
    - per_segment: static dequant with per-Q/K/V scale
    - bulk: straight copy ± bias
    """
    _has_qk_norm = cfg.qk_norm_pre_rope is not None or cfg.qk_norm_post_rope is not None
    if cfg.fused_rope or _has_qk_norm:
        _evict_and_postprocess_per_head(
            qkv_MM_output_psum=qkv_MM_output_psum,
            output_sb=output_sb,
            i_tile_S=i_tile_S,
            s_tile_sz=s_tile_sz,
            cfg=cfg,
            dims=dims,
            bias_sb=bias_sb,
            w_scale_tile=mx_dequant_sb,
            rope_bufs=rope_bufs,
            i_batch=i_batch,
            s_tile_local_offset=s_tile_local_offset,
            qk_norm_bufs=qk_norm_bufs,
            row_mx_input_scale=row_mx_input_scale,
            row_mx_w_channel_scale_sb=row_mx_w_channel_scale_sb,
        )
    elif use_static_dequant:
        _evict_psum_to_sbuf_per_segment(
            output_sb=output_sb[i_tile_S],
            qkv_MM_output_psum=qkv_MM_output_psum,
            s_tile_sz=s_tile_sz,
            i_tile_S=i_tile_S,
            mx_dequant_sb=mx_dequant_sb,
            bias_sb=bias_sb,
            q_dim=dims.num_q_heads * dims.d_head,
            kv_dim=dims.num_kv_heads * dims.d_head,
            I=I,
            num_512_tiles_per_I=dims.num_512_tiles_per_I,
        )
    elif row_mx_input_scale is not None and row_mx_w_channel_scale_sb is None:
        _evict_psum_to_sbuf_bulk_row_input_only(
            output_sb=output_sb[i_tile_S],
            qkv_MM_output_psum=qkv_MM_output_psum,
            s_tile_sz=s_tile_sz,
            i_tile_S=i_tile_S,
            bias_sb=bias_sb,
            row_mx_input_scale=row_mx_input_scale,
            I=I,
            num_512_tiles_per_I=dims.num_512_tiles_per_I,
        )
    elif row_mx_w_channel_scale_sb is not None:
        _evict_psum_to_sbuf_bulk_row_mx(
            output_sb=output_sb[i_tile_S],
            qkv_MM_output_psum=qkv_MM_output_psum,
            s_tile_sz=s_tile_sz,
            i_tile_S=i_tile_S,
            bias_sb=bias_sb,
            row_mx_input_scale=row_mx_input_scale,
            row_mx_w_channel_scale_sb=row_mx_w_channel_scale_sb,
            I=I,
            num_512_tiles_per_I=dims.num_512_tiles_per_I,
        )
    else:
        _evict_psum_to_sbuf_bulk(
            output_sb=output_sb[i_tile_S],
            qkv_MM_output_psum=qkv_MM_output_psum,
            s_tile_sz=s_tile_sz,
            i_tile_S=i_tile_S,
            bias_sb=bias_sb,
            I=I,
            num_512_tiles_per_I=dims.num_512_tiles_per_I,
        )


def _evict_psum_to_sbuf_bulk(
    output_sb: nl.NkiTensor,
    qkv_MM_output_psum: list,
    s_tile_sz: int,
    i_tile_S: int,
    bias_sb: Optional[nl.NkiTensor],
    I: int,
    num_512_tiles_per_I: int,
) -> None:
    """Plain PSUM eviction: copy PSUM banks to SBUF, optionally fusing bias add.

    No dequant scaling, no RoPE — just a straight copy (± bias) across all banks.

    Args:
        output_sb: Destination buffer in SBUF for this S tile
        qkv_MM_output_psum: PSUM bank list from matmul accumulation
        s_tile_sz: Active rows in the S tile
        i_tile_S: S tile index (for PSUM bank mapping)
        bias_sb: [P_MAX, I] bias in SBUF, or None
        I: Total fused QKV dimension
        num_512_tiles_per_I: Number of PSUM banks per S tile
    """
    F_MAX = 512
    for k_tile_I in nl.affine_range(num_512_tiles_per_I):
        psum_accumulation_bank_id = i_tile_S * num_512_tiles_per_I + k_tile_I
        num_i = min(F_MAX, I - k_tile_I * F_MAX)

        if bias_sb is not None:
            nisa.tensor_tensor(
                dst=output_sb[0:s_tile_sz, nl.ds(k_tile_I * F_MAX, num_i)],
                data1=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
                data2=bias_sb[0:s_tile_sz, nl.ds(k_tile_I * F_MAX, num_i)],
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=output_sb[0:s_tile_sz, nl.ds(k_tile_I * F_MAX, num_i)],
                src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
            )


def _evict_psum_to_sbuf_bulk_row_mx(
    output_sb: nl.NkiTensor,
    qkv_MM_output_psum: list,
    s_tile_sz: int,
    i_tile_S: int,
    bias_sb: Optional[nl.NkiTensor],
    row_mx_input_scale: nl.NkiTensor,
    row_mx_w_channel_scale_sb: nl.NkiTensor,
    I: int,
    num_512_tiles_per_I: int,
) -> None:
    """ROW_MX PSUM eviction: apply two-scale dequant (± bias) across all banks.

    Args:
        output_sb (nl.NkiTensor): Destination SBUF tile for this S-tile.
        qkv_MM_output_psum (list): List of PSUM bank tensors from matmul.
        s_tile_sz (int): Active rows in the S tile.
        i_tile_S (int): S-tile index within the current S-block.
        bias_sb (Optional[nl.NkiTensor]): Bias tensor in SBUF, or None.
        row_mx_input_scale (nl.NkiTensor): [P_MAX, 1] per-row input scale.
        row_mx_w_channel_scale_sb (nl.NkiTensor): [P_MAX, I] per-channel weight scale.
        I (int): Total output dimension (Q+K+V heads * d_head).
        num_512_tiles_per_I (int): Number of 512-wide tiles spanning I.
    """
    F_MAX = 512
    for k_tile_I in nl.affine_range(num_512_tiles_per_I):
        psum_accumulation_bank_id = i_tile_S * num_512_tiles_per_I + k_tile_I
        num_i = min(F_MAX, I - k_tile_I * F_MAX)
        bank_start = k_tile_I * F_MAX
        _dequant_row_mx(
            dst=output_sb[0:s_tile_sz, nl.ds(bank_start, num_i)],
            psum_src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
            row_scale=row_mx_input_scale[0:s_tile_sz, 0:1],
            channel_scale=row_mx_w_channel_scale_sb[0:s_tile_sz, nl.ds(bank_start, num_i)],
            bias=bias_sb[0:s_tile_sz, nl.ds(bank_start, num_i)] if bias_sb is not None else None,
        )


def _evict_psum_to_sbuf_bulk_row_input_only(
    output_sb: nl.ndarray,
    qkv_MM_output_psum: list,
    s_tile_sz: int,
    i_tile_S: int,
    bias_sb: Optional[nl.ndarray],
    row_mx_input_scale: nl.ndarray,
    I: int,
    num_512_tiles_per_I: int,
) -> None:
    """PSUM eviction with per-row input dequant only (no per-channel weight scale).

    Used for native MX weights (which already have per-block scales applied in nc_matmul_mx)
    with row-quantized FP8 input. Post-matmul correction is: output = psum * row_input_scale.

    Args:
        output_sb (nl.ndarray): Destination SBUF tile for this S-tile.
        qkv_MM_output_psum (list): List of PSUM bank tensors from matmul.
        s_tile_sz (int): Active rows in the S tile.
        i_tile_S (int): S-tile index within the current S-block.
        bias_sb (Optional[nl.ndarray]): Bias tensor in SBUF, or None.
        row_mx_input_scale (nl.ndarray): [P_MAX, 1] per-row input dequant scale.
        I (int): Total output dimension (Q+K+V heads * d_head).
        num_512_tiles_per_I (int): Number of 512-wide tiles spanning I.
    """
    F_MAX = 512
    for k_tile_I in nl.affine_range(num_512_tiles_per_I):
        psum_accumulation_bank_id = i_tile_S * num_512_tiles_per_I + k_tile_I
        num_i = min(F_MAX, I - k_tile_I * F_MAX)
        bank_start = k_tile_I * F_MAX
        _dequant_row_input_only(
            dst=output_sb[0:s_tile_sz, nl.ds(bank_start, num_i)],
            psum_src=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, 0:num_i],
            row_scale=row_mx_input_scale[0:s_tile_sz, 0:1],
            bias=bias_sb[0:s_tile_sz, nl.ds(bank_start, num_i)] if bias_sb is not None else None,
        )


def _evict_psum_to_sbuf_per_segment(
    output_sb: nl.NkiTensor,
    qkv_MM_output_psum: list,
    s_tile_sz: int,
    i_tile_S: int,
    mx_dequant_sb: nl.NkiTensor,
    bias_sb: Optional[nl.NkiTensor],
    q_dim: int,
    kv_dim: int,
    I: int,
    num_512_tiles_per_I: int,
) -> None:
    """Single-pass PSUM eviction with fused dequant scale (± bias) per Q/K/V segment.

    Banks straddling segment boundaries are split to apply the correct per-segment scale.
    Fuses the dequant multiply (and optional bias add) into the PSUM read, eliminating
    a separate copy + scale pass.

    Args:
        output_sb: Destination buffer in SBUF for this S tile
        qkv_MM_output_psum: PSUM bank list from matmul accumulation
        s_tile_sz: Active rows in the S tile
        i_tile_S: S tile index (for PSUM bank mapping)
        mx_dequant_sb: [P_MAX, 3], Pre-fused combined dequant scale per Q/K/V
        bias_sb: [P_MAX, I] bias in SBUF, or None
        q_dim: Q segment size (num_q_heads * d_head)
        kv_dim: KV segment size (num_kv_heads * d_head)
        I: Total fused QKV dimension
        num_512_tiles_per_I: Number of PSUM banks per S tile
    """
    F_MAX = 512
    seg_starts = [0, q_dim, q_dim + kv_dim]
    seg_ends = [q_dim, q_dim + kv_dim, I]

    for seg_idx in range(NUM_QKV_SEGMENTS):
        seg_start = seg_starts[seg_idx]
        seg_end = seg_ends[seg_idx]
        first_bank = seg_start // F_MAX
        last_bank = (seg_end - 1) // F_MAX
        for k_tile_I in range(first_bank, last_bank + 1):
            bank_start = k_tile_I * F_MAX
            col_start = max(seg_start, bank_start)
            col_end = min(seg_end, min(bank_start + F_MAX, I))
            num_cols = col_end - col_start
            psum_offset = col_start - bank_start
            psum_accumulation_bank_id = i_tile_S * num_512_tiles_per_I + k_tile_I

            if bias_sb is not None:
                nisa.scalar_tensor_tensor(
                    dst=output_sb[0:s_tile_sz, nl.ds(col_start, num_cols)],
                    data=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_offset, num_cols)],
                    op0=nl.multiply,
                    operand0=mx_dequant_sb[0:s_tile_sz, seg_idx],
                    op1=nl.add,
                    operand1=bias_sb[0:s_tile_sz, nl.ds(col_start, num_cols)],
                )
            else:
                nisa.tensor_scalar(
                    dst=output_sb[0:s_tile_sz, nl.ds(col_start, num_cols)],
                    data=qkv_MM_output_psum[psum_accumulation_bank_id][0:s_tile_sz, nl.ds(psum_offset, num_cols)],
                    op0=nl.multiply,
                    operand0=mx_dequant_sb[0:s_tile_sz, seg_idx],
                )


def _dma_xpose_input_non_strided(
    input_sb: nl.NkiTensor,
    input_hbm: nl.NkiTensor,
    i_tile_S: int,
    dims: QKV_CTE_Dims,
    i_batch: int,
    weight_load_offset: int,
    s_tile_local_offset: int,
    s_tile_sz: int,
    curr_num_128_H_subtiles_per_weight_block: int,
    H: int,
) -> None:
    """DMA-transpose one contiguous S tile of input from HBM into SBUF."""
    pmax = nl.tile_size.pmax
    H_subtiles = curr_num_128_H_subtiles_per_weight_block
    src_offset = i_batch * dims.S * H + (dims.S_shard_offset + s_tile_local_offset) * H + weight_load_offset
    nisa.dma_transpose(
        dst=input_sb[i_tile_S].ap(
            pattern=[[H, pmax], [1, 1], [pmax, H_subtiles], [1, s_tile_sz]],
            offset=weight_load_offset,
        ),
        src=input_hbm.ap(
            pattern=[[H, s_tile_sz], [1, 1], [pmax, H_subtiles], [1, pmax]],
            offset=src_offset,
        ),
    )


def _dma_xpose_input_strided_packed(
    input_sb: nl.NkiTensor,
    input_hbm: nl.NkiTensor,
    i_tile_S: int,
    si: StridedInputConfig,
    dims: QKV_CTE_Dims,
    i_batch: int,
    weight_load_offset: int,
    s_tile_local_offset: int,
    s_tile_sz: int,
    curr_num_128_H_subtiles_per_weight_block: int,
    H: int,
) -> None:
    """Strided gather, packed regime (block_len <= pmax): multiple blocks per DMA tile."""
    pmax = nl.tile_size.pmax
    H_subtiles = curr_num_128_H_subtiles_per_weight_block
    local_token_start = dims.S_shard_offset + s_tile_local_offset
    local_block = local_token_start // si.block_len
    local_pos_in_block = local_token_start % si.block_len
    global_token_start = si.block_offset + local_block * si.block_stride + local_pos_in_block
    src_offset = i_batch * dims.S_orig * H + global_token_start * H + weight_load_offset

    blocks_per_tile = s_tile_sz // si.block_len
    stride_el = si.block_stride * H
    # Workaround: on gen4 (Trn3), dge_mode.unknown produces incorrect results for this 4D AP.
    packed_dge_mode = dge_mode.none if nki.isa.get_nc_version() == nki.isa.nc_version.gen4 else dge_mode.unknown
    nisa.dma_transpose(
        dst=input_sb[i_tile_S].ap(
            pattern=[[H, pmax], [si.block_len, blocks_per_tile], [pmax, H_subtiles], [1, si.block_len]],
            offset=weight_load_offset,
        ),
        src=input_hbm.ap(
            pattern=[[H, si.block_len], [stride_el, blocks_per_tile], [pmax, H_subtiles], [1, pmax]],
            offset=src_offset,
        ),
        dge_mode=packed_dge_mode,
    )


def _dma_xpose_input_strided_contiguous(
    input_sb: nl.NkiTensor,
    input_hbm: nl.NkiTensor,
    i_tile_S: int,
    si: StridedInputConfig,
    dims: QKV_CTE_Dims,
    i_batch: int,
    weight_load_offset: int,
    s_tile_local_offset: int,
    s_tile_sz: int,
    curr_num_128_H_subtiles_per_weight_block: int,
    H: int,
) -> None:
    """Strided gather, contiguous-tile regime (block_len > pmax): one block spans multiple DMA tiles;
    inter-block jumps folded into src_offset, AP identical to non-strided."""
    pmax = nl.tile_size.pmax
    H_subtiles = curr_num_128_H_subtiles_per_weight_block
    local_token_start = dims.S_shard_offset + s_tile_local_offset
    local_block = local_token_start // si.block_len
    local_pos_in_block = local_token_start % si.block_len
    global_token_start = si.block_offset + local_block * si.block_stride + local_pos_in_block
    src_offset = i_batch * dims.S_orig * H + global_token_start * H + weight_load_offset
    nisa.dma_transpose(
        dst=input_sb[i_tile_S].ap(
            pattern=[[H, pmax], [1, 1], [pmax, H_subtiles], [1, s_tile_sz]],
            offset=weight_load_offset,
        ),
        src=input_hbm.ap(
            pattern=[[H, s_tile_sz], [1, 1], [pmax, H_subtiles], [1, pmax]],
            offset=src_offset,
        ),
    )


def _dma_transpose_mx_tile(
    src_tensor: nl.NkiTensor,
    dst_buf: nl.NkiTensor,
    row_stride: int,
    seq_mult: int,
    view_dtype: nki.dtype,
    h_packed_current_tile: int,
    i_batch: int,
    s_global: int,
    i_weight_block: int,
    i_tile_S: int,
    s_tile_sz: int,
    S_TILE_SIZE: int,
    quant_s_tiles: int,
    H_128_tiles: int,
    P_MAX: int,
    S: int,
) -> None:
    """DMA transpose one (H_128_tile, S_tile) from HBM into SBUF.

    Shared by the existing fp16/bf16 MX path and the ROW_MX float32 path.
    The caller controls the difference via row_stride, seq_mult, and view_dtype.

    Args:
        src_tensor (nl.NkiTensor): Source tensor on HBM (packed 2D or raw input).
        dst_buf (nl.NkiTensor): Destination buffer in SBUF.
        row_stride (int): Stride between rows in the source tensor (in elements of view_dtype).
        seq_mult (int): Sequence multiplier (2 for fp16/bf16 packed, 1 for ROW_MX float32).
        view_dtype (nki.dtype): DMA view dtype (nl.float16, nl.float32).
        h_packed_current_tile (int): Number of packed H elements in the current tile.
        i_batch (int): Batch index.
        s_global (int): Global S offset for the current tile.
        i_weight_block (int): Weight block (H_128_tile) index.
        i_tile_S (int): S-tile index within the current S-block.
        s_tile_sz (int): Active rows in the S tile.
        S_TILE_SIZE (int): Full S tile size.
        quant_s_tiles (int): Number of quantized S tiles in the block.
        H_128_tiles (int): Number of 128-wide H tiles.
        P_MAX (int): Partition dimension size.
        S (int): Total sequence length.
    """
    src_offset = i_batch * S * seq_mult * row_stride + s_global * seq_mult * row_stride + i_weight_block * P_MAX
    dst_offset = i_weight_block * quant_s_tiles * seq_mult + i_tile_S * S_TILE_SIZE * seq_mult
    seq_count = s_tile_sz * seq_mult

    nisa.dma_transpose(
        src=src_tensor.ap(
            pattern=[[row_stride, seq_count], [1, 1], [1, 1], [1, h_packed_current_tile]],
            offset=src_offset,
            dtype=view_dtype,
        ),
        dst=dst_buf.ap(
            pattern=[
                [quant_s_tiles * seq_mult * H_128_tiles, h_packed_current_tile],
                [1, 1],
                [1, 1],
                [1, seq_count],
            ],
            offset=dst_offset,
            dtype=view_dtype,
        ),
    )


def _static_mx_quantize_bf16_hidden_tile(
    bf16_xpose_staging_sb: nl.NkiTensor,
    hidden_qtz_sb: nl.NkiTensor,
    hidden_scale_sb: nl.NkiTensor,
    inv_in_scale_sb: nl.NkiTensor,
    i_weight_block: int,
    i_tile_S: int,
    s_tile_sz: int,
    quant_s_tiles: int,
    H_128_tiles: int,
    H_pack: int,
    S_TILE_SIZE: int,
) -> None:
    """Static-quantize and MX-quantize one (H_128_tile, S_tile) from the BF16 DMA transpose staging buffer.

    Applies static quantization in-place on the bfloat16 AP view (divide by in_scale,
    clamp to FP8 range), then runs quantize_mx to produce fp8_x4 data and uint8 scales
    for nc_matmul_mx.

    Always uses full P_MAX partitions and H_pack=4 for the free dim so that quantize_mx
    output aligns to s_tile_sz. For partial H tiles, unused partitions contain garbage
    that gets quantized but is never consumed by nc_matmul_mx (partition splits skip them).

    Args:
        bf16_xpose_staging_sb: Float32 staging buffer holding transposed BF16 data
        hidden_qtz_sb: Destination for quantized fp8_x4 data
        hidden_scale_sb: Destination for MX uint8 scales
        inv_in_scale_sb: [P_MAX, 1], reciprocal of static input dequant scale
        i_weight_block: Current H_128_tile index
        i_tile_S: Current S tile index
        s_tile_sz: Active rows in the S tile
        quant_s_tiles: Total S tiles allocated in buffers
        H_128_tiles: Number of H_128 tiles
        H_pack: Packing factor (4)
        S_TILE_SIZE: S tile size constant
    """
    P_MAX = nl.tile_size.pmax
    quant_src_size = s_tile_sz * H_pack

    src_ap = bf16_xpose_staging_sb.ap(
        pattern=[
            [quant_s_tiles * H_pack * H_128_tiles, P_MAX],
            [1, 1],
            [1, quant_src_size],
        ],
        offset=i_weight_block * quant_s_tiles * H_pack + i_tile_S * S_TILE_SIZE * H_pack,
        dtype=nl.bfloat16,
    )

    # Static quantize in-place: clamp(data / in_scale, ±fp8_max)
    fp8_max = _get_fp8_e4m3_max_pos_val()
    nisa.tensor_scalar(
        dst=src_ap,
        data=src_ap,
        op0=nl.multiply,
        operand0=inv_in_scale_sb[0:P_MAX, 0:1],
        op1=nl.minimum,
        operand1=fp8_max,
    )
    nisa.tensor_scalar(
        dst=src_ap,
        data=src_ap,
        op0=nl.maximum,
        operand0=-fp8_max,
    )

    nisa.quantize_mx(
        src=src_ap,
        dst=hidden_qtz_sb[0:P_MAX, i_weight_block, nl.ds(i_tile_S * P_MAX, s_tile_sz)],
        dst_scale=hidden_scale_sb[0:P_MAX, i_weight_block, nl.ds(i_tile_S * P_MAX, s_tile_sz)],
    )


def _load_and_broadcast_bias(
    bias_hbm: nl.NkiTensor, cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager
) -> nl.NkiTensor:
    """
    Loads bias to SBUF and broadcasts it to [nl.tile_size.pmax, I], using stream_shuffle.

    Accepts bias with shape [1, I] (broadcast required) or [pmax, I] (already broadcast,
    skip stream_shuffle).

    Returns allocated SBUF bias tensor.
    Note: User is responsible for deallocating SBUF tensor.
    """
    bias_sb = sbm.alloc_stack((nl.tile_size.pmax, dims.I), dtype=cfg.compute_mm_dtype, buffer=nl.sbuf)

    if bias_hbm.shape[0] == nl.tile_size.pmax:
        # Bias is already [pmax, I] — load directly, no broadcast needed.
        nisa.dma_copy(
            dst=bias_sb[0 : nl.tile_size.pmax, 0 : dims.I],
            src=bias_hbm[0 : nl.tile_size.pmax, 0 : dims.I],
            dge_mode=dge_mode.swdge,
        )
    else:
        # Bias is [1, I] — load and broadcast to [pmax, I] using stream_shuffle.
        nisa.dma_copy(
            dst=bias_sb[0:1, 0 : dims.I],
            src=bias_hbm[0:1, 0 : dims.I],
            dge_mode=dge_mode.swdge,
        )
        # Stream Shuffle works on 32 partitions only, apply it nl.tile_size.pmax // 32 = 4 times.
        NUM_BROADCASTS = nl.tile_size.pmax // MAX_STREAM_SHUFFLE_PARTITIONS
        for broadcast_idx in nl.affine_range(NUM_BROADCASTS):
            nisa.nc_stream_shuffle(
                dst=bias_sb[
                    nl.ds(broadcast_idx * MAX_STREAM_SHUFFLE_PARTITIONS, MAX_STREAM_SHUFFLE_PARTITIONS),
                    0 : dims.I,
                ],
                src=bias_sb[0:1, 0 : dims.I],
                shuffle_mask=[0] * MAX_STREAM_SHUFFLE_PARTITIONS,
            )
    return bias_sb


def _load_norm_weights(
    norm_weights_hbm: nl.NkiTensor,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> nl.NkiTensor:
    """
    Loads norm_weights with shape [H] to SBUF as [nl.tile_size.pmax, H // nl.tile_size.pmax].

    Returns allocated SBUF norm_weights tensor.
    Note: User is responsible for deallocating SBUF tensor.

    Used by RMS_NORM and LAYER_NORM to load gamma_weights_hbm to SBUF.
    In addition, may be used in LAYER_NORM to load layer_norm_bias to SBUF.
    """
    # norm_weights_hbm have 1D shape [H], make it 2-D to make NKI loads easier.
    norm_weights_hbm = norm_weights_hbm.reshape((dims.H, 1))
    # We load norm_weights into SBUF as a 2-D tensor of shape [128, H // nl.tile_size.pmax].
    # Note: We later do the multiplication on transposed input [H, S], so the math works out.
    norm_elements_in_free_dim = dims.num_128_tiles_per_H
    norm_weights_sb = sbm.alloc_stack(
        (nl.tile_size.pmax, norm_elements_in_free_dim), dtype=cfg.act_dtype, buffer=nl.sbuf
    )

    # Load in tiles of [128,1] to SBUF, now H is the first dimension (DMA broadcasted).
    # It is loaded in a way that a single norm tile uses all nl.tile_size.pmax partitions, and has 1 element per partition.
    for i_gamma_tile in range(norm_elements_in_free_dim):
        nisa.dma_copy(
            dst=norm_weights_sb[0 : nl.tile_size.pmax, nl.ds(i_gamma_tile, 1)],
            src=norm_weights_hbm[nl.ds(i_gamma_tile * nl.tile_size.pmax, nl.tile_size.pmax), 0:1],
            dge_mode=dge_mode.swdge,
        )
    return norm_weights_sb


def _load_norm_weights_mx(
    norm_weights_hbm: nl.NkiTensor,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
) -> nl.NkiTensor:
    """
    Load normalization weights to SBUF in swizzled format for MX path.

    Loads norm_weights from HBM with shape [1, H] and rearranges them into
    [P_MAX, H_128_tiles * H_pack] format suitable for MX quantized computation.
    Uses stride-4 gather pattern to match MX data layout.

    Args:
        norm_weights_hbm (nl.NkiTensor): [1, H], Normalization weights on HBM
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager

    Returns:
        nl.NkiTensor: [P_MAX, H_128_tiles * H_pack], Swizzled norm weights in SBUF

    Notes:
        - H_pack = 4 for MX format
        - H_128_tiles = ceil(H / (P_MAX * H_pack))
        - For partial last tile, only valid sub-tiles are loaded; remaining
          columns stay zero (from memset) so padded input zeros stay zero
          after gamma multiply.
        - User is responsible for deallocating SBUF tensor
    """
    P_MAX = nl.tile_size.pmax
    H_pack = 4
    H_128_tiles = div_ceil(dims.H, P_MAX * H_pack)

    # Reshape to [1, H] so stride-4 gather works on free dimension
    norm_weights_hbm = norm_weights_hbm.reshape((1, dims.H))
    gamma_sb = sbm.alloc_stack((P_MAX, H_128_tiles * H_pack), dtype=cfg.act_dtype, buffer=nl.sbuf)

    # Zero-init so unused sub-tile columns in the last partial tile are zero.
    # This ensures padded zeros in input_sb remain zero after gamma multiply.
    if dims.H % (P_MAX * H_pack) != 0:
        nisa.memset(dst=gamma_sb, value=0)

    H_128_tiles_full = dims.H // (P_MAX * H_pack)

    # For partial last tile, compute valid partitions per sub-tile.
    # H % 128 == 0 guarantees remaining is divisible by H_pack.
    if H_128_tiles > H_128_tiles_full:
        remaining = dims.H - H_128_tiles_full * P_MAX * H_pack
        p_count_last = div_ceil(remaining, H_pack)
    else:
        p_count_last = P_MAX

    for h_tile_idx in range(H_128_tiles):
        p_count = P_MAX if h_tile_idx < H_128_tiles_full else p_count_last
        for h_sub_idx in range(H_pack):
            src_offset = h_tile_idx * P_MAX * H_pack + h_sub_idx
            dst_col = h_tile_idx * H_pack + h_sub_idx
            nisa.dma_copy(
                dst=gamma_sb[0:p_count, nl.ds(dst_col, 1)],
                src=norm_weights_hbm.reshape((norm_weights_hbm.shape[1], 1)).slice(
                    # stepped slice selects p_count elements (last at
                    # src_offset + (p_count-1)*H_pack); tight exclusive end avoids
                    # overshooting H (NkiTensor.slice asserts stop <= dim_size,
                    # unlike the original .ap which did not bounds-check).
                    0,
                    src_offset,
                    src_offset + (p_count - 1) * H_pack + 1,
                    step=H_pack,
                ),
                dge_mode=dge_mode.swdge,
            )
    return gamma_sb


def _multi_buffering_degree_for_seqlen(
    cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager, qkv_in_scale: Optional[nl.NkiTensor] = None
) -> Tuple[int, int]:
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

    # Cannot multi-buffer more than dims.S_shard / nl.tile_size.pmax, e.g. if S_shard=256, best we can do is 2.
    s_multi_buffer_degree = 1
    s_multi_buffer_degree = min(math.ceil(dims.S_shard / nl.tile_size.pmax), dims.MAX_S_MULTI_BUFFER_DEGREE)

    # ------------------- Make sure multi-buffering does not cause SBUF overflow --------------------#

    # ------------------------ SBUF Space Taken by Non Buffered Tensors ----------------------------#
    #     This is the space that will be consumed by tensors we will not multi-buffer
    #     This calculation assumes we are not pre-fetching weights (this can be decided after buffering)
    #     These same constants are used in the allocation of weight tensor.

    # Sum up sizes of: # zero_bias_sb, norm_eps_sb, bias_sb, gamma_weights_sb, l
    #   layer_norm_bias_sb, act_reduce_sum, bn_stats_result, and weights_sb (non-prefetched)
    sbuf_tile_space_non_buffered = 0
    # zero_bias_sb, norm_eps_sb, bias_sb, gamma_weights_sb, layer_norm_bias_sb, act_reduce_sum, bn_stats_result.
    sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # zero_bias_sb (nl.tile_size.pmax, 1)
    if not _scalar_activation_bias_supported():
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # norm_eps_sb (nl.tile_size.pmax, 1)
    if cfg.add_bias:
        sbuf_tile_space_non_buffered += dims.I * sizeinbytes(
            cfg.compute_mm_dtype
        )  # bias_sb (nl.tile_size.pmax, dims.I)
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        # gamma_weights_sb (nl.tile_size.pmax, num_128_tiles_per_H)
        sbuf_tile_space_non_buffered += dims.num_128_tiles_per_H * sizeinbytes(cfg.act_dtype)
        if cfg.add_layer_norm_bias:
            # layer_norm_bias_sb (nl.tile_size.pmax, num_128_tiles_per_H)
            sbuf_tile_space_non_buffered += dims.num_128_tiles_per_H * sizeinbytes(cfg.act_dtype)

    # act_reduce_sum and bn_stats_result_sb appear in the loop.
    # sbuf_tile_space_non_buffered = nl.tile_size.total_available_sbuf_size - sbm.get_free_space() # space taken so far.
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.act_dtype)  # act_reduce_sum (nl.tile_size.pmax, 1)
    if cfg.fused_norm_type == NormType.LAYER_NORM:
        # bn_stats_result_sb (nl.tile_size.pmax, 6*NUM_512_BN_STATS_TILES_H)
        BN_STATS_FMAX = 512  # nl.tile_size.bn_stats_fmax  # 512
        NUM_512_BN_STATS_TILES_H = math.ceil(dims.H / BN_STATS_FMAX)
        sbuf_tile_space_non_buffered += 6 * NUM_512_BN_STATS_TILES_H * sizeinbytes(cfg.act_dtype)

    single_weight_buf_space = (dims.I * math.ceil(dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT / 128)) * sizeinbytes(
        cfg.compute_mm_dtype
    )
    weights_space_per_partition = dims.NUM_WEIGHT_BUFFERS_DEFAULT * single_weight_buf_space
    sbuf_tile_space_non_buffered += weights_space_per_partition

    # If SBUF budget is exceeded, reduce weight buffers to the actual number of load iterations needed for H.
    # With small H (e.g. H=1280, block=1024), only ceil(H/1024) loads are needed so extra buffers are unused.
    if cfg.total_available_sbuf_space_to_this_kernel - sbuf_tile_space_non_buffered < 0:
        num_weight_load_blocks = math.ceil(dims.H / dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT)
        effective_num_weight_buffers = min(dims.NUM_WEIGHT_BUFFERS_DEFAULT, num_weight_load_blocks)
        sbuf_tile_space_non_buffered -= (
            dims.NUM_WEIGHT_BUFFERS_DEFAULT - effective_num_weight_buffers
        ) * single_weight_buf_space
        kernel_assert(
            sbuf_tile_space_non_buffered < cfg.total_available_sbuf_space_to_this_kernel,
            f"SBUF budget exceeded even after reducing weight buffers: "
            f"sbuf_tile_space_non_buffered={sbuf_tile_space_non_buffered}, "
            f"available={cfg.total_available_sbuf_space_to_this_kernel}",
        )

    # QK-norm: gamma weights [pmax, d_head] broadcast + scratch [pmax, 1]
    _has_qk_norm = cfg.qk_norm_pre_rope is not None or cfg.qk_norm_post_rope is not None
    if _has_qk_norm:
        sbuf_tile_space_non_buffered += 3 * sizeinbytes(nl.float32)  # norm_scratch_sb [pmax, 1] × 3 (triple-buffered)
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(nl.float32)  # norm_zero_bias_sb [pmax, 1]
        if cfg.qk_norm_pre_rope is not None:
            if not _scalar_activation_bias_supported():
                sbuf_tile_space_non_buffered += 1 * sizeinbytes(nl.float32)  # qk_norm_pre_rope_eps_sb [pmax, 1]
            if cfg.qk_norm_pre_rope.q_gamma_norm_weights is not None:
                sbuf_tile_space_non_buffered += dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
            if cfg.qk_norm_pre_rope.k_gamma_norm_weights is not None:
                sbuf_tile_space_non_buffered += dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
        if cfg.qk_norm_post_rope is not None:
            if not _scalar_activation_bias_supported():
                sbuf_tile_space_non_buffered += 1 * sizeinbytes(nl.float32)  # qk_norm_post_rope_eps_sb [pmax, 1]
            if cfg.qk_norm_post_rope.q_gamma_norm_weights is not None:
                sbuf_tile_space_non_buffered += dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
            if cfg.qk_norm_post_rope.k_gamma_norm_weights is not None:
                sbuf_tile_space_non_buffered += dims.d_head * sizeinbytes(cfg.compute_mm_dtype)

    # --------------------SBUF Space Taken By Tensors We Will be Multi-Buffering ---------------------#

    sbuf_tile_space_pre_buffering = _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
        cfg=cfg, dims=dims, sbm=sbm, qkv_in_scale=qkv_in_scale
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

    # For fp8_packed, round down to even degree so tiles pair cleanly for the
    # two-tile transpose optimization (full 128-partition transpose 2 and DMA).
    if cfg.fp8_packed:
        s_multi_buffer_degree = max(2 * (s_multi_buffer_degree // 2), 1)

    projected_sbuf_taken_space = s_multi_buffer_degree * sbuf_tile_space_pre_buffering + sbuf_tile_space_non_buffered
    return s_multi_buffer_degree, projected_sbuf_taken_space


def _multi_buffering_degree_for_seqlen_mx(cfg: QKV_CTE_Config, dims: QKV_CTE_Dims, sbm: SbufManager) -> Tuple[int, int]:
    """
    Compute multi-buffering degree for MX path, accounting for MX-specific buffers.

    Calculates the maximum number of sequence tiles that can be processed simultaneously
    without exceeding SBUF or PSUM space constraints. Accounts for MX-specific buffers
    including transposed input, quantized activations, and scales.

    Args:
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object
        sbm (SbufManager): SBUF memory manager

    Returns:
        Tuple[int, int]: (multi_buffer_degree, projected_sbuf_space)
            - multi_buffer_degree: Number of S tiles to process simultaneously
            - projected_sbuf_space: Estimated total SBUF space usage

    Notes:
        - Considers MX weight buffers in chunked mode
        - Accounts for transposed_buffer, hidden_qtz, and hidden_scale per S tile
        - Respects PSUM bank constraints (NUM_HW_PSUM_BANKS // num_512_tiles_per_I)
    """
    P_MAX = nl.tile_size.pmax
    H_pack = 4
    H_128_tiles = div_ceil(dims.H, P_MAX * H_pack)

    s_multi_buffer_degree = min(math.ceil(dims.S_shard / P_MAX), dims.MAX_S_MULTI_BUFFER_DEGREE)

    # Non-buffered space: global allocations + MX weight buffers (chunked)
    NUM_MX_WEIGHT_BUFFERS_LOCAL = 2
    sbuf_tile_space_non_buffered = 0
    if not cfg.is_input_swizzled:
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # zero_bias_sb
        if not _scalar_activation_bias_supported():
            sbuf_tile_space_non_buffered += 1 * sizeinbytes(cfg.compute_mm_dtype)  # norm_eps_sb

    if cfg.add_bias:
        sbuf_tile_space_non_buffered += dims.I * sizeinbytes(cfg.compute_mm_dtype)
    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.LAYER_NORM:
        sbuf_tile_space_non_buffered += (
            H_128_tiles * H_pack * sizeinbytes(cfg.act_dtype)
        )  # gamma_sb [P_MAX, H_128_tiles * H_pack]
        if cfg.add_layer_norm_bias:
            sbuf_tile_space_non_buffered += H_128_tiles * H_pack * sizeinbytes(cfg.act_dtype)

    # MX weight buffers (chunked mode - worst case)
    sbuf_tile_space_non_buffered += NUM_MX_WEIGHT_BUFFERS_LOCAL * 1 * dims.I * sizeinbytes(nl.float8_e4m3fn_x4)
    sbuf_tile_space_non_buffered += NUM_MX_WEIGHT_BUFFERS_LOCAL * 1 * dims.I * sizeinbytes(nl.uint8)

    # QK-norm: gamma weights [pmax, d_head] broadcast + scratch [pmax, 1]
    _has_qk_norm = cfg.qk_norm_pre_rope is not None or cfg.qk_norm_post_rope is not None
    if _has_qk_norm:
        sbuf_tile_space_non_buffered += 3 * sizeinbytes(
            nl.float32
        )  # QKNormBuffers.norm_scratch_sb [pmax, 1] × 3 (triple-buffered)
        sbuf_tile_space_non_buffered += 1 * sizeinbytes(nl.float32)  # QKNormBuffers.norm_zero_bias_sb [pmax, 1]
        if cfg.qk_norm_pre_rope is not None:
            if not _scalar_activation_bias_supported():
                sbuf_tile_space_non_buffered += 1 * sizeinbytes(nl.float32)  # qk_norm_pre_rope_eps_sb [pmax, 1]
            if cfg.qk_norm_pre_rope.q_gamma_norm_weights is not None:
                sbuf_tile_space_non_buffered += P_MAX * dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
            if cfg.qk_norm_pre_rope.k_gamma_norm_weights is not None:
                sbuf_tile_space_non_buffered += P_MAX * dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
        if cfg.qk_norm_post_rope is not None:
            if not _scalar_activation_bias_supported():
                sbuf_tile_space_non_buffered += 1 * sizeinbytes(nl.float32)  # qk_norm_post_rope_eps_sb [pmax, 1]
            if cfg.qk_norm_post_rope.q_gamma_norm_weights is not None:
                sbuf_tile_space_non_buffered += P_MAX * dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
            if cfg.qk_norm_post_rope.k_gamma_norm_weights is not None:
                sbuf_tile_space_non_buffered += P_MAX * dims.d_head * sizeinbytes(cfg.compute_mm_dtype)

    _is_fp8_input = cfg.input_dtype in [nl.float8_e4m3, nl.float8_e4m3fn]
    _is_row_mx = cfg.quantization_config.has_row_mx_dequant
    _is_dma_xpose_mx = (
        (_is_fp8_input or (cfg.input_dtype == nl.bfloat16 and cfg.quantization_config.has_mx_static_dequant_scales))
        and cfg.load_input_with_DMA_transpose
    ) or _is_row_mx

    # ROW_MX: global weight channel scale [P_MAX, I] float32
    if _is_row_mx:
        sbuf_tile_space_non_buffered += dims.I * sizeinbytes(nl.float32)  # row_mx_w_channel_scale_sb

    # Per-S-tile space: input_sb + output_sb + norm buffers + rope buffers
    sbuf_tile_space_per_s_tile = _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
        cfg=cfg, dims=dims, sbm=sbm, is_fp8_dma_xpose=_is_dma_xpose_mx
    )

    # Non-DMA-xpose BF16 MX path pads input_sb from H to H_padded (next multiple of 512).
    # Account for the extra space in the per-S-tile budget.
    H_padded = H_128_tiles * P_MAX * H_pack
    if not _is_dma_xpose_mx and H_padded != dims.H:
        sbuf_tile_space_per_s_tile += (H_padded - dims.H) * sizeinbytes(cfg.compute_mm_dtype)

    # MX-specific per-S-tile space (scales with num_s_tiles)
    mx_space_per_s_tile = H_128_tiles * P_MAX * sizeinbytes(nl.float8_e4m3fn_x4)  # hidden_qtz
    if not (_is_fp8_input and _is_dma_xpose_mx):
        mx_space_per_s_tile += (
            H_128_tiles * P_MAX * sizeinbytes(nl.uint8)
        )  # hidden_scale (not needed for FP8 DMA xpose)
    if not _is_dma_xpose_mx:
        mx_space_per_s_tile += H_128_tiles * P_MAX * H_pack * sizeinbytes(cfg.compute_mm_dtype)  # transposed_buffer

    # BF16 DMA transpose path: staging buffer for DMA transpose output before quantize_mx
    _is_bf16_dma_xpose_mx = (
        cfg.input_dtype == nl.bfloat16
        and cfg.quantization_config.has_mx_static_dequant_scales
        and cfg.load_input_with_DMA_transpose
    )
    if _is_bf16_dma_xpose_mx:
        mx_space_per_s_tile += H_128_tiles * P_MAX * 2 * sizeinbytes(nl.float32)  # bf16_xpose_staging_sb

    # ROW_MX: per-S-tile input row scale [P_MAX, 1] float32
    if _is_row_mx:
        mx_space_per_s_tile += 1 * sizeinbytes(nl.float32)  # row_mx_input_scale_sb

    total_space_per_s_tile = sbuf_tile_space_per_s_tile + mx_space_per_s_tile

    # When input is swizzled, S_TILE_SIZE=32 instead of P_MAX=128, so each
    # multi-buffer degree unit (P_MAX rows) contains P_MAX/S_TILE_SIZE actual
    # S-tiles. Scale the per-unit cost accordingly.
    S_TILE_SIZE = 32 if cfg.is_input_swizzled else P_MAX
    tiles_per_unit = P_MAX // S_TILE_SIZE
    total_space_per_unit = total_space_per_s_tile * tiles_per_unit

    max_s_buffer = (
        cfg.total_available_sbuf_space_to_this_kernel - sbuf_tile_space_non_buffered
    ) // total_space_per_unit
    s_multi_buffer_degree = min(s_multi_buffer_degree, max(1, max_s_buffer))

    # PSUM constraint
    MAX_PSUM_TILING_GROUPS = NUM_HW_PSUM_BANKS // dims.num_512_tiles_per_I
    s_multi_buffer_degree = min(s_multi_buffer_degree, MAX_PSUM_TILING_GROUPS)

    projected_space = s_multi_buffer_degree * total_space_per_unit + sbuf_tile_space_non_buffered
    return s_multi_buffer_degree, projected_space


def _use_weight_prefetch_mx(
    projected_sbuf_taken_space: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
) -> bool:
    """
    Determine if MX path can afford weight prefetching.

    Compares the space needed for full weight prefetching against available SBUF space
    after accounting for other allocations. Prefetching loads all weights upfront
    instead of chunked loading during computation.

    Args:
        projected_sbuf_taken_space (int): Estimated SBUF space already allocated
        cfg (QKV_CTE_Config): Kernel configuration object
        dims (QKV_CTE_Dims): Tensor dimensions object

    Returns:
        bool: True if weight prefetching is feasible, False otherwise

    Notes:
        - Prefetch space = H_128_tiles * I * (sizeof(float8) + sizeof(uint8))
        - Chunked space = NUM_MX_WEIGHT_BUFFERS * 1 * I * (sizeof(float8) + sizeof(uint8))
    """
    P_MAX = nl.tile_size.pmax
    H_pack = 4
    H_128_tiles = div_ceil(dims.H, P_MAX * H_pack)
    NUM_MX_WEIGHT_BUFFERS_LOCAL = 2

    # Prefetch space needed
    weights_prefetch_space = H_128_tiles * dims.I * sizeinbytes(nl.float8_e4m3fn_x4)
    scales_prefetch_space = H_128_tiles * dims.I * sizeinbytes(nl.uint8)

    # Chunked space (already accounted in projected_space)
    weights_chunk_space = NUM_MX_WEIGHT_BUFFERS_LOCAL * 1 * dims.I * sizeinbytes(nl.float8_e4m3fn_x4)
    scales_chunk_space = NUM_MX_WEIGHT_BUFFERS_LOCAL * 1 * dims.I * sizeinbytes(nl.uint8)

    can_prefetch = (
        projected_sbuf_taken_space
        - weights_chunk_space
        - scales_chunk_space
        + weights_prefetch_space
        + scales_prefetch_space
    ) < cfg.total_available_sbuf_space_to_this_kernel

    return can_prefetch


def _get_sbuf_space_taken_by_tensors_about_to_be_multi_buffered(
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    sbm: SbufManager,
    is_fp8_dma_xpose: bool = False,
    qkv_in_scale: Optional[nl.NkiTensor] = None,
) -> int:
    """
    Compute the total SBUF space taken (per partition) by simultaneously live tensors that will be multi-buffered in the kernel.

    WARNING: This is not independently useful function, its correctness is based on the tensor allocation that comes after it.
    This is a 'lookahead' function.
    NOTE: If any additional tensors are added in the kernel, this function needs to be updated.

    Current tensors inside a loop that will get buffered are:
    'input_sb', 'output_sb'                                      (unless FP8 DMA transpose: no input_sb)
    'square_sum_sb',                                             (if cfg.fused_norm_type.RMS_NORM or cfg.fused_norm_type.RMS_NORM_GAMMA)
    'bn_aggr_result_sb'                                          (if cfg.fused_norm_type.LAYER_NORM)
    'cos_buffer_sb', 'sin_buffer_sb', 'rope_scratch_sb'          (if cfg.fused_rope, via RoPEBuffers)
    """

    pre_buffer_tile_space_per_partition = 0

    if cfg.quantization_config.quantization_type == QuantizationType.STATIC:
        # 'input_sb  [nl.tile_size.pmax, H]' + 'input_quantized_sb [nl.tile_size.pmax, H]'
        pre_buffer_tile_space_per_partition += dims.H * sizeinbytes(cfg.input_dtype)
        pre_buffer_tile_space_per_partition += dims.H * sizeinbytes(cfg.quantization_config.quant_dtype)

        # 'in_scale_tile' + 'w_scale_tile'
        pre_buffer_tile_space_per_partition += 3 * sizeinbytes(cfg.quantization_config.qkv_w_scale.dtype) + sizeinbytes(
            qkv_in_scale.dtype
        )
    else:
        # 'input_sb  [nl.tile_size.pmax, H]' — not allocated on FP8 DMA transpose path
        if not is_fp8_dma_xpose:
            pre_buffer_tile_space_per_partition += dims.H * sizeinbytes(cfg.compute_mm_dtype)

    # 'output_sb [nl.tile_size.pmax, I]'
    pre_buffer_tile_space_per_partition += dims.I * sizeinbytes(cfg.compute_mm_dtype)

    if cfg.fused_norm_type == NormType.RMS_NORM or cfg.fused_norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        pre_buffer_tile_space_per_partition += 1 * sizeinbytes(cfg.act_dtype)  # 'square_sum_sb [nl.tile_size.pmax, 1]'

    if cfg.fused_norm_type == NormType.LAYER_NORM:
        NUM_AGGR_STATS = 2
        # 'bn_aggr_result_sb [nl.tile_size.pmax, NUM_AGGR_STATS]'
        pre_buffer_tile_space_per_partition += NUM_AGGR_STATS * sizeinbytes(cfg.act_dtype)

    if cfg.fused_rope:
        # RoPEBuffers.cos_sb [pmax, d_head]
        pre_buffer_tile_space_per_partition += dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
        # RoPEBuffers.sin_sb [pmax, d_head // 2] or [pmax, d_head] when gamma fused
        _gamma_fused_in_rope = cfg.qk_norm_pre_rope is not None and cfg.qk_norm_pre_rope.gamma_fused_in_rope_caches
        _sin_fdim = dims.d_head if _gamma_fused_in_rope else dims.d_head // 2
        pre_buffer_tile_space_per_partition += _sin_fdim * sizeinbytes(cfg.compute_mm_dtype)
        # RoPEBuffers.k_cos_sb + k_sin_sb (when separate K caches provided)
        if _gamma_fused_in_rope:
            pre_buffer_tile_space_per_partition += dims.d_head * sizeinbytes(cfg.compute_mm_dtype)
            pre_buffer_tile_space_per_partition += _sin_fdim * sizeinbytes(cfg.compute_mm_dtype)
        # RoPEBuffers.scratch_sb [pmax, d_head * 2] × N
        # Triple-buffered (N=3) only when QK-norm is active to break carried dependency
        _has_qk_norm = cfg.qk_norm_pre_rope is not None or cfg.qk_norm_post_rope is not None
        _num_rope_bufs = _get_num_rope_scratch_bufs(_has_qk_norm, dims.H)
        pre_buffer_tile_space_per_partition += dims.d_head * 2 * _num_rope_bufs * sizeinbytes(cfg.compute_mm_dtype)

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
        * (dims.I * math.ceil(dims.WEIGHT_LOAD_BLOCK_SIZE_PER_H_DEFAULT / nl.tile_size.pmax))
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
    input_row_sb: nl.NkiTensor,
    square_sum_row_sb: nl.NkiTensor,
    zero_bias: nl.NkiTensor,
    norm_eps: nl.NkiTensor,
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
    # To get around this, we broadcast the [pmax, 1] dst over H with a 0-stride view (dst[..].broadcast(dim=1, size=H)).
    # Keeping "act" shape as [pmax, H] would waste valuable SBUF space and potentally limit multi-buffering and weight-prefetching.

    # Temporary tensor allocation
    act_reduce_sbm = sbm.alloc_stack((nl.tile_size.pmax, 1), dtype=cfg.act_dtype, buffer=nl.sbuf)

    # Sum of squares: x1^2 + x2^2 + ... + xn^2 ( sum of squares of the input row ).
    nisa.activation_reduce(
        dst=act_reduce_sbm[0:s_tile_sz, 0:1].broadcast(dim=1, size=dims.H),
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
        bias=_slice_activation_constant_bias(norm_eps, s_tile_sz),
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
    input_row_sb: nl.NkiTensor,
    bn_aggr_result_tile: nl.NkiTensor,
    norm_eps: nl.NkiTensor,
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
        (nl.tile_size.pmax, BN_STATS_DST_SIZE * NUM_512_BN_STATS_TILES_H),
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
        bias=_slice_activation_constant_bias(norm_eps, s_tile_sz),
        op=nl.rsqrt,
    )


# Per-head norm registry, keyed on NormType.name strings due to compiler incompatibility with enum lookups.
_QK_NORM_REGISTRY: Dict[str, Callable] = {}


def register_norm(norm_type: NormType):
    """Decorator to register a per-head norm implementation.

    Args:
        norm_type: NormType enum value.
    """

    def decorator(fn):
        _QK_NORM_REGISTRY[norm_type.name] = fn
        return fn

    return decorator


@register_norm(NormType.RMS_NORM)  # Qwen3.5, Gemma 4
def _rms_norm_impl(
    head_sb: nl.NkiTensor,
    i_head: int,
    num_q_heads: int,
    is_pre_rope: bool,
    qk_norm_bufs: QKNormBuffers,
    norm_config: QKNormConfig,
) -> None:
    """RMSNorm: x / sqrt(mean(x²) + eps) * gamma.

    Derives all buffers from qk_norm_bufs. Reads s_tile_sz and d_head from head_sb.shape.
    """
    s_tile_sz = head_sb.shape[0]
    d_head = head_sb.shape[1]
    is_q_head = i_head < num_q_heads
    norm_scratch_sb = qk_norm_bufs.scratch_sb[i_head % 3]
    eps_sb = qk_norm_bufs.pre_rope_eps_sb if is_pre_rope else qk_norm_bufs.post_rope_eps_sb
    if is_q_head:
        gamma_sb = qk_norm_bufs.pre_rope_q_gamma_sb if is_pre_rope else qk_norm_bufs.post_rope_q_gamma_sb
    else:
        gamma_sb = qk_norm_bufs.pre_rope_k_gamma_sb if is_pre_rope else qk_norm_bufs.post_rope_k_gamma_sb
    norm_zero_bias_sb = qk_norm_bufs.zero_bias_sb
    # sum(x²) -> [s_tile_sz, 1]
    nisa.activation_reduce(
        dst=norm_scratch_sb[0:s_tile_sz, 0:1].broadcast(dim=1, size=d_head),
        op=nl.square,
        data=head_sb,
        reduce_op=nl.add,
        reduce_res=norm_scratch_sb[0:s_tile_sz, 0:1],
        bias=norm_zero_bias_sb[0:s_tile_sz, 0:1],
    )

    # 1/sqrt(mean(x²) + eps) -> [s_tile_sz, 1]
    nisa.activation(
        dst=norm_scratch_sb[0:s_tile_sz, 0:1],
        op=nl.rsqrt,
        data=norm_scratch_sb[0:s_tile_sz, 0:1],
        bias=_slice_activation_constant_bias(eps_sb, s_tile_sz),
        scale=1.0 / d_head,
    )

    # x *= 1/sqrt(mean(x²) + eps), optionally fused with gamma multiply
    if gamma_sb is not None:
        nisa.scalar_tensor_tensor(
            dst=head_sb,
            data=head_sb,
            op0=nl.multiply,
            operand0=norm_scratch_sb[0:s_tile_sz, 0:1],
            op1=nl.multiply,
            operand1=gamma_sb[0:s_tile_sz, 0:d_head],
        )
    else:
        nisa.tensor_scalar(
            dst=head_sb,
            data=head_sb,
            op0=nl.multiply,
            operand0=norm_scratch_sb[0:s_tile_sz, 0:1],
        )


def _apply_qk_norm_to_head(
    head_sb: nl.NkiTensor,
    i_head: int,
    num_q_heads: int,
    is_pre_rope: bool,
    norm_func: str,
    qk_norm_bufs: QKNormBuffers,
    norm_config: QKNormConfig,
) -> None:
    """Dispatch per-head norm to the registered implementation.

    Args:
        head_sb: Pre-sliced [s_tile_sz, d_head] SBUF region to normalize in-place.
            The impl reads dimensions from head_sb.shape.
        i_head: Head index in the Q+K range. Determines Q vs K (i_head < num_q_heads)
            and scratch buffer selection.
        num_q_heads: Number of Q heads.
        is_pre_rope: True for pre-RoPE norm, False for post-RoPE.
        norm_func: NormType.name string (e.g. "RMS_NORM").
        qk_norm_bufs: Shared SBUF buffers (eps, gamma, scratch, zero_bias).
        norm_config: The QKNormConfig for this position (pre or post RoPE).
    """
    kernel_assert(
        norm_func in _QK_NORM_REGISTRY,
        f"[QKV CTE Kernel] QK norm type '{norm_func}' is not registered. Registered: {list(_QK_NORM_REGISTRY.keys())}.",
    )
    impl = _QK_NORM_REGISTRY[norm_func]
    impl(head_sb, i_head, num_q_heads, is_pre_rope, qk_norm_bufs, norm_config)


def _apply_rope_to_head(
    src_sb: nl.NkiTensor,
    dst_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    s_tile_sz: int,
    d_head: int,
    num_d: int,
):
    """Apply RoPE rotation to a single head.

    Reads from src_sb [s_tile_sz, d_head*2] scratch buffer (first d_head holds the head data),
    writes the rotated result to dst_sb [s_tile_sz, num_d].

    Implements: output = X * cos + [-X2, X1] * sin
    where X = [X1, X2] split at d_head // 2.
    """
    d_head_half = d_head // 2
    num_d_half = num_d // 2

    # -X2 * sin
    nisa.scalar_tensor_tensor(
        dst=src_sb[0:s_tile_sz, nl.ds(d_head, num_d_half)],
        data=src_sb[0:s_tile_sz, nl.ds(d_head_half, num_d_half)],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.multiply,
        operand1=sin_sb[0:s_tile_sz, 0:num_d_half],
    )

    # X1 * sin
    nisa.tensor_tensor(
        dst=src_sb[0:s_tile_sz, nl.ds(d_head + d_head_half, num_d_half)],
        data1=src_sb[0:s_tile_sz, 0:num_d_half],
        data2=sin_sb[0:s_tile_sz, 0:num_d_half],
        op=nl.multiply,
    )

    # X * cos
    nisa.tensor_tensor(
        dst=src_sb[0:s_tile_sz, 0:num_d],
        data1=src_sb[0:s_tile_sz, 0:num_d],
        data2=cos_sb[0:s_tile_sz, 0:num_d],
        op=nl.multiply,
    )

    # X * cos + [-X2 * sin, X1 * sin] → dst
    nisa.tensor_tensor(
        dst=dst_sb,
        data1=src_sb[0:s_tile_sz, 0:num_d],
        data2=src_sb[0:s_tile_sz, nl.ds(d_head, num_d)],
        op=nl.add,
    )


def _evict_and_postprocess_per_head(
    qkv_MM_output_psum: List[nl.NkiTensor],
    output_sb: List[nl.NkiTensor],
    i_tile_S: int,
    s_tile_sz: int,
    cfg: QKV_CTE_Config,
    dims: QKV_CTE_Dims,
    bias_sb: Optional[nl.NkiTensor],
    w_scale_tile: Optional[nl.NkiTensor],
    rope_bufs: Optional[RoPEBuffers],
    i_batch: Optional[int],
    s_tile_local_offset: Optional[int],
    qk_norm_bufs: Optional[QKNormBuffers] = None,
    row_mx_input_scale: Optional[nl.NkiTensor] = None,
    row_mx_w_channel_scale_sb: Optional[nl.NkiTensor] = None,
    row_w_scale_sb: Optional[nl.NkiTensor] = None,
) -> None:
    """Per-head PSUM eviction + postprocessing (QK-norm, RoPE).

    Handles all per-head eviction paths:
    - Gamma fusion (gamma_fused_in_rope_caches): evict to output_sb, fused rsqrt-in-RoPE in-place
    - Non-fusion with RoPE: evict to rope scratch, QK-norm, out-of-place RoPE to output_sb
    - Non-fusion without RoPE: bulk evict to output_sb, then QK-norm in-place per head

    V heads are always evicted directly to output_sb with optional dequant/bias.

    Args:
        qkv_MM_output_psum: PSUM banks from matmul, indexed [i_tile_S * num_512_tiles_per_I + k].
        output_sb: Multi-buffered SBUF output tiles, indexed [i_tile_S].
        i_tile_S: Current S-tile index within the block.
        s_tile_sz: Active rows in this S tile.
        cfg: Kernel configuration.
        dims: Kernel dimension parameters.
        bias_sb: [pmax, I] bias in SBUF, or None.
        w_scale_tile: [pmax, 3] per-Q/K/V dequant scales, or None.
        rope_bufs: RoPE cos/sin caches and scratch buffers, or None.
        i_batch: Current batch index.
        s_tile_local_offset: S offset of this tile within the current block.
        qk_norm_bufs: QK-norm scratch and weight buffers, or None.
    """
    d_head = dims.d_head
    d_head_half = d_head // 2
    NUM_HEADS_PER_PSUM_BANK = 512 // d_head
    _has_qk_norm = cfg.qk_norm_pre_rope is not None or cfg.qk_norm_post_rope is not None
    _gamma_fused_in_rope = (
        cfg.qk_norm_pre_rope is not None and cfg.qk_norm_pre_rope.gamma_fused_in_rope_caches and cfg.fused_rope
    )

    # No-RoPE fast path: bulk evict all banks to output_sb, then per-head QK-norm in-place.
    # This avoids per-head PSUM reads when only QK-norm is needed (no RoPE scratch buffer required).
    if not cfg.fused_rope and _has_qk_norm:
        # Step 1: Bulk evict PSUM → output_sb (per-bank, not per-head)
        if row_mx_w_channel_scale_sb is not None:
            _evict_psum_to_sbuf_bulk_row_mx(
                output_sb=output_sb[i_tile_S],
                qkv_MM_output_psum=qkv_MM_output_psum,
                s_tile_sz=s_tile_sz,
                i_tile_S=i_tile_S,
                bias_sb=bias_sb,
                row_mx_input_scale=row_mx_input_scale,
                row_mx_w_channel_scale_sb=row_mx_w_channel_scale_sb,
                I=dims.I,
                num_512_tiles_per_I=dims.num_512_tiles_per_I,
            )
        elif row_mx_input_scale is not None and row_mx_w_channel_scale_sb is None:
            _evict_psum_to_sbuf_bulk_row_input_only(
                output_sb=output_sb[i_tile_S],
                qkv_MM_output_psum=qkv_MM_output_psum,
                s_tile_sz=s_tile_sz,
                i_tile_S=i_tile_S,
                bias_sb=bias_sb,
                row_mx_input_scale=row_mx_input_scale,
                I=dims.I,
                num_512_tiles_per_I=dims.num_512_tiles_per_I,
            )
        elif w_scale_tile is not None:
            _evict_psum_to_sbuf_per_segment(
                output_sb=output_sb[i_tile_S],
                qkv_MM_output_psum=qkv_MM_output_psum,
                s_tile_sz=s_tile_sz,
                i_tile_S=i_tile_S,
                mx_dequant_sb=w_scale_tile,
                bias_sb=bias_sb,
                q_dim=dims.num_q_heads * d_head,
                kv_dim=dims.num_kv_heads * d_head,
                I=dims.I,
                num_512_tiles_per_I=dims.num_512_tiles_per_I,
            )
        else:
            _evict_psum_to_sbuf_bulk(
                output_sb=output_sb[i_tile_S],
                qkv_MM_output_psum=qkv_MM_output_psum,
                s_tile_sz=s_tile_sz,
                i_tile_S=i_tile_S,
                bias_sb=bias_sb,
                I=dims.I,
                num_512_tiles_per_I=dims.num_512_tiles_per_I,
            )

        # ROW dequantization after bulk eviction (before per-head processing)
        if row_w_scale_sb is not None:
            nisa.tensor_tensor(
                dst=output_sb[i_tile_S][0:s_tile_sz, : dims.I],
                data1=output_sb[i_tile_S][0:s_tile_sz, : dims.I],
                data2=row_w_scale_sb[0:s_tile_sz, : dims.I],
                op=nl.multiply,
            )

        # Step 2: Per-head QK-norm in-place on already-evicted output_sb
        for i_head in nl.affine_range(dims.num_q_heads + dims.num_kv_heads):
            head_offset = i_head * d_head
            num_d = min(d_head, dims.I - head_offset)

            if cfg.qk_norm_pre_rope is not None:
                _pre_norm_func = (
                    cfg.qk_norm_pre_rope.q_norm if i_head < dims.num_q_heads else cfg.qk_norm_pre_rope.k_norm
                )
                if _pre_norm_func is not None:
                    _apply_qk_norm_to_head(
                        head_sb=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                        i_head=i_head,
                        num_q_heads=dims.num_q_heads,
                        is_pre_rope=True,
                        norm_func=_pre_norm_func.name,
                        qk_norm_bufs=qk_norm_bufs,
                        norm_config=cfg.qk_norm_pre_rope,
                    )

            if cfg.qk_norm_post_rope is not None:
                _post_norm_func = (
                    cfg.qk_norm_post_rope.q_norm if i_head < dims.num_q_heads else cfg.qk_norm_post_rope.k_norm
                )
                if _post_norm_func is not None:
                    _apply_qk_norm_to_head(
                        head_sb=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                        i_head=i_head,
                        num_q_heads=dims.num_q_heads,
                        is_pre_rope=False,
                        norm_func=_post_norm_func.name,
                        qk_norm_bufs=qk_norm_bufs,
                        norm_config=cfg.qk_norm_post_rope,
                    )
        return

    # RoPE path: per-head eviction required (RoPE needs scratch buffer for out-of-place rotation)
    _load_rope_caches(rope_bufs, cfg, dims, i_tile_S, s_tile_sz, i_batch, s_tile_local_offset)

    # Q/K heads: evict + postprocess
    for i_head in nl.affine_range(dims.num_q_heads + dims.num_kv_heads):
        head_offset = i_head * d_head
        num_d = min(d_head, dims.I - head_offset)
        psum_bank_id = i_tile_S * dims.num_512_tiles_per_I + i_head // NUM_HEADS_PER_PSUM_BANK
        psum_head_offset = (i_head % NUM_HEADS_PER_PSUM_BANK) * d_head

        _num_rope_bufs = len(rope_bufs.scratch_sb[i_tile_S]) if cfg.fused_rope else 0
        _rope_buf = rope_bufs.scratch_sb[i_tile_S][i_head % _num_rope_bufs] if cfg.fused_rope else None
        _norm_scratch = qk_norm_bufs.scratch_sb[i_head % 3] if qk_norm_bufs is not None else None

        if w_scale_tile is not None:
            current_scale = w_scale_tile[0:s_tile_sz, 0] if i_head < dims.num_q_heads else w_scale_tile[0:s_tile_sz, 1]

        # Gamma fusion: evict to output_sb
        # Non-fusion with RoPE: evict to rope scratch (out-of-place RoPE needs it)
        if cfg.fused_rope and not _gamma_fused_in_rope:
            evict_dst = _rope_buf[0:s_tile_sz, 0:num_d]
        else:
            evict_dst = output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)]

        _row_mx_in_scale = row_mx_input_scale[0:s_tile_sz, 0:1] if row_mx_input_scale is not None else None
        _row_mx_w_scale = (
            row_mx_w_channel_scale_sb[0:s_tile_sz, nl.ds(head_offset, num_d)]
            if row_mx_w_channel_scale_sb is not None
            else None
        )

        _evict_one_head(
            dst=evict_dst,
            psum_src=qkv_MM_output_psum[psum_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
            bias_slice=bias_sb[0:s_tile_sz, nl.ds(head_offset, num_d)]
            if cfg.add_bias
            else None,  # None when qkv_bias=False
            w_scale=current_scale if w_scale_tile is not None else None,  # None for non-MX (no static dequant)
            s_tile_sz=s_tile_sz,
            use_activation_for_scale=_has_qk_norm,
            row_mx_input_scale=_row_mx_in_scale,
            row_mx_w_scale_slice=_row_mx_w_scale,
        )

        # ROW dequantization: per-element weight scale applied after eviction
        if row_w_scale_sb is not None:
            nisa.tensor_tensor(
                dst=evict_dst,
                data1=evict_dst,
                data2=row_w_scale_sb[0:s_tile_sz, nl.ds(head_offset, num_d)],
                op=nl.multiply,
            )

        if _gamma_fused_in_rope:
            # Gamma fusion: postprocess in-place on output_sb (fused rsqrt-in-RoPE)
            _postprocess_one_qk_head(
                output_sb=output_sb,
                i_tile_S=i_tile_S,
                s_tile_sz=s_tile_sz,
                head_offset=head_offset,
                num_d=num_d,
                i_head=i_head,
                cfg=cfg,
                dims=dims,
                rope_bufs=rope_bufs,
                qk_norm_bufs=qk_norm_bufs,
                _rope_buf=_rope_buf,
                _norm_scratch=_norm_scratch,
            )
        else:
            # Non-fusion: QK-norm on evict_dst, then out-of-place RoPE, then post-RoPE QK-norm
            if cfg.qk_norm_pre_rope is not None:
                _pre_norm_func = (
                    cfg.qk_norm_pre_rope.q_norm if i_head < dims.num_q_heads else cfg.qk_norm_pre_rope.k_norm
                )
                if _pre_norm_func is not None:
                    _apply_qk_norm_to_head(
                        head_sb=evict_dst,
                        i_head=i_head,
                        num_q_heads=dims.num_q_heads,
                        is_pre_rope=True,
                        norm_func=_pre_norm_func.name,
                        qk_norm_bufs=qk_norm_bufs,
                        norm_config=cfg.qk_norm_pre_rope,
                    )

            if cfg.fused_rope:
                _apply_rope_to_head(
                    src_sb=_rope_buf,
                    dst_sb=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                    cos_sb=rope_bufs.cos_sb[i_tile_S],
                    sin_sb=rope_bufs.sin_sb[i_tile_S],
                    s_tile_sz=s_tile_sz,
                    d_head=d_head,
                    num_d=num_d,
                )

            if cfg.qk_norm_post_rope is not None:
                _post_norm_func = (
                    cfg.qk_norm_post_rope.q_norm if i_head < dims.num_q_heads else cfg.qk_norm_post_rope.k_norm
                )
                if _post_norm_func is not None:
                    _apply_qk_norm_to_head(
                        head_sb=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                        i_head=i_head,
                        num_q_heads=dims.num_q_heads,
                        is_pre_rope=False,
                        norm_func=_post_norm_func.name,
                        qk_norm_bufs=qk_norm_bufs,
                        norm_config=cfg.qk_norm_post_rope,
                    )

    # V heads: evict only
    for i_head in range(dims.num_q_heads + dims.num_kv_heads, dims.num_q_heads + 2 * dims.num_kv_heads):
        head_offset = i_head * d_head
        num_d = min(d_head, dims.I - head_offset)
        psum_bank_id = i_tile_S * dims.num_512_tiles_per_I + i_head // NUM_HEADS_PER_PSUM_BANK
        psum_head_offset = (i_head % NUM_HEADS_PER_PSUM_BANK) * d_head

        _row_mx_in_scale = row_mx_input_scale[0:s_tile_sz, 0:1] if row_mx_input_scale is not None else None
        _row_mx_w_scale = (
            row_mx_w_channel_scale_sb[0:s_tile_sz, nl.ds(head_offset, num_d)]
            if row_mx_w_channel_scale_sb is not None
            else None
        )

        _evict_one_head(
            dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
            psum_src=qkv_MM_output_psum[psum_bank_id][0:s_tile_sz, nl.ds(psum_head_offset, num_d)],
            bias_slice=bias_sb[0:s_tile_sz, nl.ds(head_offset, num_d)]
            if cfg.add_bias
            else None,  # None when qkv_bias=False
            w_scale=w_scale_tile[0:s_tile_sz, 2]
            if w_scale_tile is not None
            else None,  # None for non-MX (no static dequant)
            s_tile_sz=s_tile_sz,
            use_activation_for_scale=_has_qk_norm,
            row_mx_input_scale=_row_mx_in_scale,
            row_mx_w_scale_slice=_row_mx_w_scale,
        )

        # ROW dequantization: per-element weight scale applied after eviction
        if row_w_scale_sb is not None:
            nisa.tensor_tensor(
                dst=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                data1=output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)],
                data2=row_w_scale_sb[0:s_tile_sz, nl.ds(head_offset, num_d)],
                op=nl.multiply,
            )


def _load_rope_caches(rope_bufs, cfg, dims, i_tile_S, s_tile_sz, i_batch, s_tile_local_offset):
    """DMA load cos/sin RoPE caches for the current S tile.

    When gamma_fused_in_rope_caches=True, sin is full d_head width (sin_lo_fused and
    sin_hi_fused packed) and separate K cos/sin caches are loaded if present.
    """
    if not cfg.fused_rope:
        return
    d_head = dims.d_head
    _gamma_fused_in_rope = cfg.qk_norm_pre_rope is not None and cfg.qk_norm_pre_rope.gamma_fused_in_rope_caches
    sin_fdim = d_head if _gamma_fused_in_rope else d_head // 2

    cos_src_offset = i_batch * dims.S * d_head + (dims.S_shard_offset + s_tile_local_offset) * d_head
    cos_src_row = cos_src_offset // d_head
    nisa.dma_copy(
        dst=rope_bufs.cos_sb[i_tile_S].slice(0, 0, s_tile_sz),
        src=rope_bufs.cos_hbm.flatten_dims(start_dim=0, end_dim=1).slice(0, cos_src_row, cos_src_row + s_tile_sz),
        dge_mode=dge_mode.swdge,
    )
    sin_src_offset = i_batch * dims.S * d_head + (dims.S_shard_offset + s_tile_local_offset) * d_head
    nisa.dma_copy(
        dst=rope_bufs.sin_sb[i_tile_S].slice(0, 0, s_tile_sz),
        src=rope_bufs.sin_hbm.flatten_dims(start_dim=0, end_dim=1)
        .slice(0, sin_src_offset // d_head, sin_src_offset // d_head + s_tile_sz)
        .slice(1, 0, sin_fdim),
        dge_mode=dge_mode.swdge,
    )

    if rope_bufs.k_cos_hbm is not None:
        nisa.dma_copy(
            dst=rope_bufs.k_cos_sb[i_tile_S].slice(0, 0, s_tile_sz),
            src=rope_bufs.k_cos_hbm.flatten_dims(start_dim=0, end_dim=1).slice(0, cos_src_row, cos_src_row + s_tile_sz),
            dge_mode=dge_mode.swdge,
        )
        nisa.dma_copy(
            dst=rope_bufs.k_sin_sb[i_tile_S].slice(0, 0, s_tile_sz),
            src=rope_bufs.k_sin_hbm.flatten_dims(start_dim=0, end_dim=1)
            .slice(0, sin_src_offset // d_head, sin_src_offset // d_head + s_tile_sz)
            .slice(1, 0, sin_fdim),
            dge_mode=dge_mode.swdge,
        )


def _postprocess_one_qk_head(
    output_sb,
    i_tile_S,
    s_tile_sz,
    head_offset,
    num_d,
    i_head,
    cfg,
    dims,
    rope_bufs,
    qk_norm_bufs,
    _rope_buf,
    _norm_scratch,
) -> None:
    """Fused rsqrt-in-RoPE for a single Q/K head (gamma_fused_in_rope_caches path).

    Uses 6 instructions: rsqrt is folded into the cos/sin multiplies, gamma is
    pre-baked into the caches by the caller. Data is in output_sb.
    """
    head_sb = output_sb[i_tile_S][0:s_tile_sz, nl.ds(head_offset, num_d)]
    d_head = dims.d_head
    d_head_half = d_head // 2
    num_d_half = num_d // 2

    # Select Q or K fused cos/sin
    if rope_bufs.k_cos_hbm is not None and i_head >= dims.num_q_heads:
        _fused_cos = rope_bufs.k_cos_sb[i_tile_S]
        _fused_sin = rope_bufs.k_sin_sb[i_tile_S]
    else:
        _fused_cos = rope_bufs.cos_sb[i_tile_S]
        _fused_sin = rope_bufs.sin_sb[i_tile_S]

    # 1. sum(x²)
    nisa.activation_reduce(
        dst=_norm_scratch[0:s_tile_sz, 0:1].broadcast(dim=1, size=num_d),
        op=nl.square,
        data=head_sb,
        reduce_op=nl.add,
        reduce_res=_norm_scratch[0:s_tile_sz, 0:1],
        bias=_slice_activation_constant_bias(qk_norm_bufs.zero_bias_sb, s_tile_sz),
    )

    # 2. rsqrt(mean(x²) + eps)
    nisa.activation(
        dst=_norm_scratch[0:s_tile_sz, 0:1],
        op=nl.rsqrt,
        data=_norm_scratch[0:s_tile_sz, 0:1],
        bias=_slice_activation_constant_bias(qk_norm_bufs.pre_rope_eps_sb, s_tile_sz),
        scale=1.0 / num_d,
    )
    _rsqrt = _norm_scratch[0:s_tile_sz, 0:1]

    # 3. -x2 * sin_hi_fused → scratch[0:d_half]
    nisa.scalar_tensor_tensor(
        dst=_rope_buf[0:s_tile_sz, 0:num_d_half],
        data=head_sb[0:s_tile_sz, nl.ds(d_head_half, num_d_half)],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.multiply,
        operand1=_fused_sin[0:s_tile_sz, nl.ds(d_head_half, num_d_half)],
    )

    # 4. x1 * sin_lo_fused → scratch[d_half:d]
    nisa.tensor_tensor(
        dst=_rope_buf[0:s_tile_sz, nl.ds(d_head_half, num_d_half)],
        data1=head_sb[0:s_tile_sz, 0:num_d_half],
        data2=_fused_sin[0:s_tile_sz, 0:num_d_half],
        op=nl.multiply,
    )

    # 5. x * rsqrt * cos_fused → head_sb
    nisa.scalar_tensor_tensor(
        dst=head_sb,
        data=head_sb,
        op0=nl.multiply,
        operand0=_rsqrt,
        op1=nl.multiply,
        operand1=_fused_cos[0:s_tile_sz, 0:num_d],
    )

    # 6. sin_terms * rsqrt + cos_result → head_sb
    nisa.scalar_tensor_tensor(
        dst=head_sb,
        data=_rope_buf[0:s_tile_sz, 0:num_d],
        op0=nl.multiply,
        operand0=_rsqrt,
        op1=nl.add,
        operand1=head_sb,
    )


def _evict_one_head(
    dst,
    psum_src,
    bias_slice,
    w_scale,
    s_tile_sz: int,
    use_activation_for_scale: bool,
    row_mx_input_scale=None,
    row_mx_w_scale_slice=None,
) -> None:
    """Evict a single head from PSUM to SBUF with optional dequant and bias.

    Selects the appropriate instruction based on which optional operations are active:
    - ROW_MX dequant: scalar_tensor_tensor (psum * row_scale * channel_scale) [+ bias]
    - dequant + bias: scalar_tensor_tensor (fused multiply + add)
    - bias only: tensor_tensor add
    - dequant with QK-norm: activation copy with scale (ScalarE, reduces DVE pressure)
    - dequant without QK-norm: tensor_scalar multiply (VectorE)
    - neither: tensor_copy

    Args:
        dst: Destination slice in SBUF.
        psum_src: Source slice in PSUM.
        bias_slice: Bias slice in SBUF, or None.
        w_scale: Per-segment dequant scale [s_tile_sz, 1], or None.
        s_tile_sz: Active rows in the S tile.
        use_activation_for_scale: When True, use ScalarE activation for dequant
            to reduce DVE pressure (typically set when QK-norm is active).
        row_mx_input_scale: [s_tile_sz, 1] per-row input scale, or None.
        row_mx_w_scale_slice: [s_tile_sz, d_head] per-element weight scale, or None.
    """
    if row_mx_w_scale_slice is not None:
        _dequant_row_mx(
            dst=dst,
            psum_src=psum_src,
            row_scale=row_mx_input_scale,
            channel_scale=row_mx_w_scale_slice,
            bias=bias_slice,
        )
    elif row_mx_input_scale is not None and row_mx_w_scale_slice is None:
        _dequant_row_input_only(
            dst=dst,
            psum_src=psum_src,
            row_scale=row_mx_input_scale,
            bias=bias_slice,
        )
    elif bias_slice is not None:
        if w_scale is not None:
            nisa.scalar_tensor_tensor(
                dst=dst,
                data=psum_src,
                op0=nl.multiply,
                operand0=w_scale,
                op1=nl.add,
                operand1=bias_slice,
            )
        else:
            nisa.tensor_tensor(dst=dst, data1=psum_src, data2=bias_slice, op=nl.add)
    elif w_scale is not None:
        if use_activation_for_scale:
            nisa.activation(dst=dst, op=nl.copy, data=psum_src, scale=w_scale)
        else:
            nisa.tensor_scalar(dst=dst, data=psum_src, op0=nl.multiply, operand0=w_scale)
    else:
        nisa.tensor_copy(dst=dst, src=psum_src)
