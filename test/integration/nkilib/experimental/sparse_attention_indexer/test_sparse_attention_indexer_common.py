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

"""Shared test utilities for Sparse Attention Indexer tests.

Contains: input generator, MX-weight quantization helpers used by both the
fixture (to produce packed weights) and the torch ref (to dequantize back to
fp32 so the torch math reproduces the same MX-weight rounding noise).
"""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np

from test.utils.mx_utils import dequantize_mx_golden, quantize_mx_golden


def precompute_freqs(dim, seq_len, theta=10000.0):
    """Precompute cos/sin for RoPE."""
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    t = np.arange(seq_len, dtype=np.float32)
    freqs = np.outer(t, freqs)
    return np.cos(freqs).astype(np.float32), np.sin(freqs).astype(np.float32)


def create_causal_mask(S, cache_len, start_pos, batch_size=1):
    """Create causal attention mask."""
    M = batch_size * S
    mask = np.full((M, cache_len), float('-inf'), dtype=np.float32)
    for b in range(batch_size):
        for i in range(S):
            row = b * S + i
            valid_end = start_pos + i + 1
            mask[row, :valid_end] = 0.0
    return mask


def _swizzle_and_quantize_x_for_mx(x_f32, num_S_tiles, S_tile, num_K_tiles, P_MAX=128, H_PACK=4):
    """Replicate the kernel's K-side ``_quantize_activation_for_mx`` host-side.

    Produces the SBUF data + scale buffers per S-tile so the kernel can
    ``dma_copy`` them directly and skip the in-kernel HBM load + 4-pass
    nc_transpose swizzle + quantize_mx.

    Args:
        x_f32: ``[B*S, dim]`` float32 activation. dim == num_K_tiles*P_MAX*H_PACK.
        num_S_tiles, S_tile, num_K_tiles: tiling sizes (S_tile == P_MAX in v3).

    Returns:
        x_mx_data: ``[num_S_tiles, P_MAX, num_K_tiles*P_MAX]`` fp8_e4m3fn_x4.
        x_mx_scale: ``[num_S_tiles, P_MAX, num_K_tiles*P_MAX]`` uint8 in HW-
            quadrant layout (valid rows at partitions [0..3, 32..35, 64..67,
            96..99] per K-tile, rest zero).
    """
    K_TILE_FREE = P_MAX * H_PACK  # 512
    assert x_f32.shape[1] == num_K_tiles * K_TILE_FREE
    BxS, dim = x_f32.shape
    # num_S_tiles is the CEIL tile count; the last tile may be partial (BxS not a
    # multiple of S_tile, e.g. a CP shard with S_local < P_MAX). Zero-pad the tail.
    assert BxS <= num_S_tiles * S_tile

    x_mx_data = np.zeros(
        (num_S_tiles, P_MAX, num_K_tiles * P_MAX),
        dtype=dt.float8_e4m3fn_x4,
    )
    x_mx_scale = np.zeros(
        (num_S_tiles, P_MAX, num_K_tiles * P_MAX),
        dtype=np.uint8,
    )

    for s_tile in range(num_S_tiles):
        s_start = s_tile * S_tile
        s_end = min(s_start + S_tile, BxS)  # clamp partial last tile (no OOB read)
        rows = s_end - s_start

        # Step 1: replicate the kernel's swizzle on x[s_start:s_end] (bf16).
        # Result: transposed [P_MAX, num_K_tiles, K_TILE_FREE] bf16 where
        #   transposed[j, k_tile, 4*s + h_sub] = x[s_start+s, k_tile*K_TILE_FREE + 4*j + h_sub]
        x_tile_bf16 = x_f32[s_start:s_end, :].astype(nl.bfloat16).astype(np.float32)
        # Pad partition to P_MAX (partial last tile -> only `rows` valid, rest zero).
        x_pad = np.zeros((P_MAX, dim), dtype=np.float32)
        x_pad[:rows, :] = x_tile_bf16

        transposed = np.zeros((P_MAX, num_K_tiles, K_TILE_FREE), dtype=np.float32)
        for k_tile in range(num_K_tiles):
            # Slice the K-tile's contiguous K_TILE_FREE region of x first.
            x_k_tile = x_pad[:, k_tile * K_TILE_FREE : (k_tile + 1) * K_TILE_FREE]
            for h_sub in range(H_PACK):
                # j ∈ [0, P_MAX), s ∈ [0, P_MAX) → output (j, k_tile, 4s+h_sub).
                # AP reads x[s, k_tile*K_TILE_FREE + 4*j + h_sub] for j ∈ [0,P_MAX).
                src = x_k_tile[:, h_sub::H_PACK]  # [P_MAX, P_MAX]
                # transposed[j, k_tile, 4*s + h_sub] = src[s, j]
                transposed[:, k_tile, h_sub::H_PACK] = src.T

        # Step 2: quantize_mx_golden expects [P, F]; reshape transposed.
        flat = transposed.reshape(P_MAX, num_K_tiles * K_TILE_FREE).astype(nl.bfloat16).astype(np.float32)
        # quantize_mx_golden returns mx_data [P, F/4] x4-packed and mx_scale
        # [P/8, F/4] uint8.
        mx_data, mx_scale_compact = quantize_mx_golden(flat, nl.float8_e4m3fn_x4)
        # mx_data shape: [P_MAX=128, num_K_tiles*P_MAX] fp8x4
        x_mx_data[s_tile] = mx_data

        # Step 3: expand compact scale [16, num_K_tiles*P_MAX] to HW-quadrant
        # [P_MAX=128, num_K_tiles*P_MAX] (rows [0..3, 32..35, 64..67, 96..99]).
        #
        # mx_scale_compact[i, f] for i in [0..16) corresponds to a group of 8
        # partitions [8i, 8i+1, ..., 8i+7] in the original input. For HW
        # quantize_mx, the output scale at compact row i lands at HW
        # partition row (i // 4) * SCALE_QUADRANT_SIZE + (i % 4).
        SCALE_QUADRANT_SIZE = 32
        for i in range(P_MAX // 8):  # 16
            hw_row = (i // 4) * SCALE_QUADRANT_SIZE + (i % 4)
            x_mx_scale[s_tile, hw_row, :] = mx_scale_compact[i, :]

    return x_mx_data, x_mx_scale


def _quantize_weight_for_mx(weight_f32):
    """Quantize a weight tensor for nc_matmul_mx.

    Args:
        weight_f32: [K, F] float32 weight where K is the contraction dim and F is
            the free (output) dim. The MX kernel will read this as [K // 4, F]
            fp8_x4 (4 consecutive K elements packed into one x4 word) plus a
            [K // 32, F] uint8 e8m0 scale per 32 K elements at each F position.

    Returns:
        (packed_x4 numpy [K // 4, F] of nl.float8_e4m3fn_x4 dtype,
         scale_uint8 numpy [K // 32, F],
         dequantized_f32 numpy [K, F] for the torch reference).

    The reshape/transpose places 4 consecutive K rows at the same F position into
    4 consecutive cols of the input to quantize_mx_golden. The MX 8×4 block then
    covers 8×4 = 32 K elements at one F position, matching nc_matmul_mx's micro-
    scaling group size.
    """
    K, F = weight_f32.shape
    assert K % 32 == 0, f"K must be a multiple of 32 for MX, got {K}"
    # Reshape so that the LAST dim is 4-wide along K (hits the x4 packing along K)
    # and the new "row" dim has K // 4 entries. Final shape passed to quantize_mx_golden:
    # [K // 4, F * 4], with each 4 consecutive cols at one row coming from 4 consecutive K rows.
    expanded = weight_f32.reshape(K // 4, 4, F).transpose(0, 2, 1).reshape(K // 4, F * 4)
    packed_x4, scale = quantize_mx_golden(expanded, nl.float8_e4m3fn_x4)
    # packed_x4: [K // 4, F] (numpy with x4 dtype after static_cast)
    # scale: [(K // 4) // 8, (F * 4) // 4] = [K // 32, F]

    # Dequantize back to f32 for the torch reference using the same scale layout.
    dequantized_expanded = dequantize_mx_golden(packed_x4, scale)  # [K // 4, F * 4] float32
    dequantized_f32 = dequantized_expanded.reshape(K // 4, F, 4).transpose(0, 2, 1).reshape(K, F)
    return packed_x4, scale, dequantized_f32


def generate_inputs(
    n_heads,
    head_dim,
    dim,
    q_lora_rank,
    rope_head_dim,
    S,
    batch_size,
    start_pos,
    use_hadamard,
    index_topk,
    max_seq_len=None,
    phase_d_kv_layout=False,
    bf16_score_kv_layout=False,
):
    """Generate kernel inputs including scalar parameters.

    The indexer always runs MX (block-32 fp8) projections, so the MX weight/qr
    quantization below is unconditional.
    """
    np.random.seed(42)
    M = batch_size * S
    end_pos = start_pos + S
    if max_seq_len is None:
        max_seq_len = max(256, end_pos + 128)

    x = np.random.randn(M, dim).astype(np.float32) * 0.1
    # Pre-cast bf16 view of x for W-projection. Keeps the [B*S, dim] layout
    # so the W-side can keep its dma_transpose primitive (HW-optimized for
    # transposed loads) and just skip the f32->bf16 cast on Scalar.
    x_non_mx = x.astype(nl.bfloat16)
    qr = np.random.randn(M, q_lora_rank).astype(np.float32) * 0.1
    wq_b_f32 = np.random.randn(q_lora_rank, n_heads * head_dim).astype(np.float32) * 0.1
    # wk is [head_dim, dim] in the BF16 path (so wk @ x.T uses the contiguous dim
    # of x as contraction). For the MX path we need the contraction dim (= dim)
    # on the partition axis of nc_matmul_mx, so re-order to [dim, head_dim].
    wk_f32 = np.random.randn(head_dim, dim).astype(np.float32) * 0.1
    k_norm_gamma = np.ones(head_dim, dtype=np.float32)
    k_norm_beta = np.zeros(head_dim, dtype=np.float32)
    weights_proj_f32 = np.random.randn(n_heads, dim).astype(np.float32) * 0.1
    # Pass weights_proj as bf16 so the kernel's hoist DMA can land directly
    # into a bf16 SBUF buffer (skip f32->bf16 tensor_copy cast on Scalar).
    weights_proj = weights_proj_f32.astype(nl.bfloat16)
    cos, sin = precompute_freqs(rope_head_dim, S)
    mask = create_causal_mask(S, end_pos, start_pos, batch_size)

    if bf16_score_kv_layout:
        # bf16-score variant: K cache is bf16 [B, head_dim, max_seq_len].
        # No k_scale_cache (no MX quant on score path).
        k_cache = np.zeros((batch_size, head_dim, max_seq_len), dtype=nl.bfloat16)
        k_scale_cache = None
    elif phase_d_kv_layout:
        # Phase D: K cache stores fp8_e4m3fn_x4. Each x4 word packs 4 head_dim
        # rows, matching the qkv_cte weight layout: [B, head_dim // 4, max_seq_len]
        # x4 = head_dim*max_seq_len fp8 values total (one per (b, head_dim, s)).
        # On SBUF the kernel sees this as [head_dim P, max_seq_len F] fp8_x4
        # because the HBM->SBUF DMA implicitly unpacks the x4 partition.
        # k_scale_cache is dense uint8 [B, head_dim // 32, max_seq_len] (one
        # uint8 e8m0 scale per 32-element block of head_dim per s).
        import neuron_dtypes as ndt

        k_cache = np.zeros((batch_size, head_dim // 4, max_seq_len), dtype=ndt.float8_e4m3fn_x4)
        k_scale_cache = np.zeros((batch_size, head_dim // 32, max_seq_len), dtype=np.uint8)
    else:
        # Phase B: k_cache stores fp8_e4m3fn alongside a per-token fp32 dequant scale cache.
        k_cache = np.zeros((batch_size, max_seq_len, head_dim), dtype=nl.float8_e4m3fn)
        k_scale_cache = np.zeros((batch_size, max_seq_len, 1), dtype=np.float32)

    inputs = {
        "x": x,
        "qr": qr,
        "k_norm_gamma": k_norm_gamma,
        "k_norm_beta": k_norm_beta,
        "weights_proj": weights_proj,
        "cos": cos,
        "sin": sin,
        "k_cache": k_cache,
        "mask": mask,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "rope_head_dim": rope_head_dim,
        "index_topk": index_topk,
        "start_pos": start_pos,
        "use_hadamard": use_hadamard,
        "batch_size": batch_size,
        "x_non_mx": x_non_mx,
    }

    wq_b_packed, wq_b_scale, _ = _quantize_weight_for_mx(wq_b_f32)
    wk_dim_first = wk_f32.T  # [dim, head_dim]
    wk_packed, wk_scale, _ = _quantize_weight_for_mx(wk_dim_first)
    # Pre-quantized x for K-projection (skip in-kernel swizzle + quantize_mx).
    # K-side reads partition slabs of these per S-tile.
    P_MAX_CONST = 128
    H_PACK_CONST = 4
    num_K_tiles_const = dim // (P_MAX_CONST * H_PACK_CONST)
    S_tile_const = P_MAX_CONST
    # CEIL: a CP shard with S < P_MAX (e.g. T_local=64 at ws64) still needs one padded
    # S-tile. The quantizer zero-pads the partial tail (matches the QKV kernel's ceil export).
    num_S_tiles_const = (M + S_tile_const - 1) // S_tile_const
    x_mx_data, x_mx_scale = _swizzle_and_quantize_x_for_mx(
        x,
        num_S_tiles_const,
        S_tile_const,
        num_K_tiles_const,
        P_MAX=P_MAX_CONST,
        H_PACK=H_PACK_CONST,
    )
    # Pre-quantized qr latent (qr_qtz_hbm/qr_scale_hbm): the MX entry always
    # consumes qr pre-quantized from an upstream QKV kernel (it derives
    # q_lora_rank from qr_qtz_hbm.shape[2]), so the standalone test must
    # supply it. Same swizzle+quantize as x, but over the q_lora contraction:
    # -> [num_S_tiles, P_MAX, num_K_tiles_qr*P_MAX], then reshape to the
    # kernel's [num_s_tiles, P_MAX, num_K_tiles_qr, P_MAX] and view fp8x4 as
    # uint32 (the kernel DMAs it as uint32 -> fp8x4 SBUF, no MX cast).
    num_K_tiles_qr = q_lora_rank // (P_MAX_CONST * H_PACK_CONST)
    qr_mx_data, qr_mx_scale = _swizzle_and_quantize_x_for_mx(
        qr,
        num_S_tiles_const,
        S_tile_const,
        num_K_tiles_qr,
        P_MAX=P_MAX_CONST,
        H_PACK=H_PACK_CONST,
    )
    qr_qtz = qr_mx_data.reshape(num_S_tiles_const, P_MAX_CONST, num_K_tiles_qr, P_MAX_CONST).view(np.uint32)
    qr_scale = qr_mx_scale.reshape(num_S_tiles_const, P_MAX_CONST, num_K_tiles_qr, P_MAX_CONST).astype(np.uint8)
    inputs["wq_b"] = wq_b_packed
    inputs["wq_b_scale"] = wq_b_scale
    inputs["wk"] = wk_packed
    inputs["wk_scale"] = wk_scale
    inputs["k_scale_cache"] = k_scale_cache
    inputs["x_mx_data"] = x_mx_data
    inputs["x_mx_scale"] = x_mx_scale
    inputs["qr_qtz_hbm"] = qr_qtz
    inputs["qr_scale_hbm"] = qr_scale

    return inputs


def _dequant_swizzled_qr_for_torch_ref(qr_qtz, qr_scale, M, q_lora_rank, P_MAX=128, H_PACK=4):
    """Inverse of the qr swizzle+quantize in ``generate_inputs`` (MX path).

    Recovers the ``[M, q_lora_rank]`` activation the kernel actually consumes — i.e.
    the raw ``qr`` after block-32 MX quant rounding — so the torch ref replays the
    Q projection against the same rounded values the kernel does (rather than the
    pristine raw ``qr``). Mirrors ``_swizzle_and_quantize_x_for_mx`` in reverse:
    dequant each S-tile's ``[P_MAX, num_K_tiles*P_MAX]`` block, then undo the swizzle
    ``transposed[j, k_tile, 4s+h] = x[s, k_tile*512 + 4j + h]``.

    Args:
        qr_qtz: ``[num_S_tiles, P_MAX, num_K_tiles, P_MAX]`` uint32 (fp8x4 viewed as u32).
        qr_scale: ``[num_S_tiles, P_MAX, num_K_tiles, P_MAX]`` uint8 (HW-quadrant scale).
    Returns: ``[M, q_lora_rank]`` float32.
    """
    K_TILE_FREE = P_MAX * H_PACK  # 512
    num_K_tiles = q_lora_rank // K_TILE_FREE
    S_tile = P_MAX
    num_S_tiles = M // S_tile
    SCALE_QUADRANT_SIZE = 32

    qr_qtz_x4 = qr_qtz.reshape(num_S_tiles, P_MAX, num_K_tiles * P_MAX).view(dt.float8_e4m3fn_x4)
    qr_scale_hw = qr_scale.reshape(num_S_tiles, P_MAX, num_K_tiles * P_MAX).astype(np.uint8)

    out = np.zeros((M, q_lora_rank), dtype=np.float32)
    for s_tile in range(num_S_tiles):
        # Rebuild the compact [16, num_K_tiles*P_MAX] scale from the HW-quadrant rows
        # ([0..3, 32..35, 64..67, 96..99]) that _swizzle_and_quantize_x_for_mx wrote.
        compact = np.zeros((P_MAX // 8, num_K_tiles * P_MAX), dtype=np.uint8)
        for i in range(P_MAX // 8):
            hw_row = (i // 4) * SCALE_QUADRANT_SIZE + (i % 4)
            compact[i, :] = qr_scale_hw[s_tile, hw_row, :]
        # Dequant -> transposed [P_MAX, num_K_tiles*K_TILE_FREE] float32.
        transposed = dequantize_mx_golden(qr_qtz_x4[s_tile], compact).reshape(P_MAX, num_K_tiles, K_TILE_FREE)
        # Undo swizzle: x[s, k_tile*512 + 4j + h] = transposed[j, k_tile, 4s+h].
        for k_tile in range(num_K_tiles):
            base = k_tile * K_TILE_FREE
            for h_sub in range(H_PACK):
                # transposed[:, k_tile, h_sub::H_PACK] is src.T with src[s, j].
                src_T = transposed[:, k_tile, h_sub::H_PACK]  # [P_MAX(j), P_MAX(s)]
                # Write only this k_tile's 512-wide region (base .. base+512), stride 4.
                out[s_tile * S_tile : (s_tile + 1) * S_tile, base + h_sub : base + K_TILE_FREE : H_PACK] = src_T.T
    return out


def _dequantize_weight_for_torch_ref(packed_x4, scale, K, F):
    """Inverse of _quantize_weight_for_mx for the torch reference.

    packed_x4: numpy [K // 4, F] in nl.float8_e4m3fn_x4 dtype.
    scale: numpy [K // 32, F] uint8.
    Returns: numpy [K, F] float32.
    """
    deq_expanded = dequantize_mx_golden(packed_x4, scale)  # [K // 4, F * 4] float32
    return deq_expanded.reshape(K // 4, F, 4).transpose(0, 2, 1).reshape(K, F)
