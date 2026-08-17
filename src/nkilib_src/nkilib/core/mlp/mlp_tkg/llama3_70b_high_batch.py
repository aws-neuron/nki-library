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
Specialized MLP TKG kernel for Llama3-70B high-batch static FP8 configs.

Isolated entry point with all projection logic inlined for independent tuning.
Config: H=8192, STATIC FP8, RMS_NORM, SiLU, column tiling, no fused_add, no bias, no skip_gate.

Optimization vs. the per-chunk baseline: the BxS dimension (T_full=256) is forced into
two 128-token chunks because T is the matmul stationary free dim (cap 128). The FP8
gate/up/down weights, however, are identical across chunks. Instead of running the full
MLP per chunk (which re-DMAs the gate, up and down weights from HBM for every chunk and
doubles weight DMA traffic on the critical path), this implementation hoists the weight
loads OUT of the per-chunk path: each weight tile is loaded from HBM exactly once and the
matmuls for BOTH chunks reuse the resident SBUF weight tile before it is overwritten/freed.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl
from nki.isa import matmul_perf_mode

from ...utils.allocator import BufferManager, SbufManager
from ...utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_verified_program_sharding_info
from ...utils.logging import get_logger
from ...utils.tiled_range import TiledRange

# ---------- Hardcoded constants ----------
_DGE_MODE_NONE = 3
BS_TILE_SIZE = 128  # tokens per BxS chunk

T = 128  # tokens per chunk
H = 8192
I = 3584
LNC = 2
H0 = 128  # nl.tile_size.pmax
I0 = 128
H1 = H // H0  # 64
H1_SHARD = H1 // LNC  # 32
H_PER_SHARD = H // LNC  # 4096
H2 = H1 // LNC  # 32  (per-shard H1 used inside RMSNorm load)
PSUM_FMAX = 512  # nl.tile_size.psum_fmax
PSUM_BMAX = 8  # PSUM bank count
COLUMN_TILING_DIM = 128
COLUMN_TILING_FACTOR = 1

# Gate/Up (column tiling, FP8 weights, fp32 partial sums).
GATE_UP_HTILE = 4096
GATE_UP_NUM_HTILES = H_PER_SHARD // GATE_UP_HTILE  # 1
GATE_UP_NUM_128_PER_HTILE = GATE_UP_HTILE // H0  # 32
GATE_UP_NUM_DR_PAIRS = GATE_UP_NUM_128_PER_HTILE // 2  # 16 (double_row pairs of H0-blocks)
GATE_UP_NUM_W_TILES = 1  # gate and up sequentially share one slot
NUM_GATE_UP_PSUMS = div_ceil(I, PSUM_FMAX)  # 7

# RMSNorm BxS sharding: each core handles half of T, then sendrecv exchanges shards.
RMSNORM_SHARDING_THRESHOLD = 18
RMSNORM_SHARD_SIZE = T // LNC  # 64

# Down (FP8 double_row: both operands FP8, contraction over I packed two I0-blocks
# at a time -> half the Tensor-Engine matmuls vs. 1x column tiling).
DOWN_HTILE = 4096
DOWN_NUM_HTILES = H_PER_SHARD // DOWN_HTILE  # 1
NUM_I_TILES = I // I0  # 28
DOWN_NUM_DR_PAIRS = NUM_I_TILES // 2  # 14 (double_row pairs of I0-blocks)
NUM_DOWN_PSUMS = DOWN_HTILE // PSUM_FMAX  # 8
DOWN_NUM_W_TILES = DOWN_NUM_DR_PAIRS * DOWN_NUM_HTILES  # 14 (one resident slot per I0-block pair)


# ============================================================================
# Phase 1: RMSNorm + input load for a single chunk -> resident SBUF [H0, T, H1_SHARD]
# ============================================================================


def _rmsnorm_chunk(params, sbm, shard_id, input_sb):
    """RMSNorm + input load for one 128-token chunk into the provided resident SBUF tile.

    H-shard with a tiny scale exchange (replaces the old BxS-shard + full-H broadcast):
    each core loads ONLY its own H1_SHARD half of the hidden vector for ALL T tokens
    (= H_PER_SHARD per token, the same data volume the matmul needs anyway). The
    sum-of-squares is split as ssq_full = ssq_half0 + ssq_half1, so each core reduces its
    own H-half locally to a per-token PARTIAL sum [H0, T], then a single tiny sendrecv
    exchanges that [H0, T] partial (≈64 KB, vs the old ≈1 MB full-H broadcast) and the two
    partials are summed to recover the exact full-H sum of squares. The norm is then applied
    in place on the resident H-half, writing ``input_sb`` directly — so the old broadcast
    sendrecv at line 174 AND the post-broadcast H1_SHARD select-copy both vanish, while the
    per-core norm compute and input DMA stay at their original (halved) volume. Math is
    bit-equivalent to the single-core full-H RMSNorm up to fp32 reduction order.
    All scratch buffers are scope/heap managed and fully freed before returning, so the
    only buffer that survives is ``input_sb``.
    """
    io_dtype = params.hidden_tensor.dtype

    sbm.open_scope()

    # 1a. Load this core's H-half (contiguous HBM block [shard*H_PER_SHARD, ...]) for ALL T
    # tokens directly into the resident input_sb [H0, T, H1_SHARD]. The H-half block of
    # H_PER_SHARD=4096 maps to [H0=128, H1_SHARD=32] (h_within = h0*H1_SHARD + h1s), matching
    # the gate/up weight H-shard (gate_w[shard*H_PER_SHARD : (shard+1)*H_PER_SHARD]).
    input_flat = params.hidden_tensor.flatten_dims(start_dim=0, end_dim=1)
    input_half = input_flat.slice(dim=1, start=shard_id * H_PER_SHARD, end=(shard_id + 1) * H_PER_SHARD)
    input_hbm_view = input_half.reshape_dim(dim=1, shape=[H0, H1_SHARD]).permute(dims=[1, 0, 2])  # [H0, T, H1_SHARD]
    nisa.dma_copy(dst=input_sb, src=input_hbm_view, dge_mode=_DGE_MODE_NONE)

    # 1b. Load this core's gamma H-half -> [H0, H1_SHARD].
    gamma_view = params.norm_params.normalization_weights_tensor
    gamma_sb = sbm.alloc_heap(shape=(H0, H1_SHARD), dtype=gamma_view.dtype, name="rmsnorm_gamma")
    gamma_flat = gamma_view.flatten_dims(start_dim=0, end_dim=1)
    gamma_half = gamma_flat.slice(dim=0, start=shard_id * H_PER_SHARD, end=(shard_id + 1) * H_PER_SHARD)
    gamma_hbm_reshaped = gamma_half.reshape_dim(dim=0, shape=[H0, H1_SHARD])  # [H0, H1_SHARD]
    nisa.dma_copy(dst=gamma_sb, src=gamma_hbm_reshaped, dge_mode=_DGE_MODE_NONE)

    # 1c. Constants for rsqrt(mean + eps) and matmul-reduce across H0 partitions.
    eps_sb = sbm.alloc_heap(shape=(H0, 1), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_eps")
    nisa.memset(eps_sb, value=params.eps)
    mm_reduce_const = sbm.alloc_heap(shape=(H0, H0), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_mm_ones")
    nisa.memset(mm_reduce_const, value=1.0)

    # 1d. Per-token PARTIAL sum-of-squares over this core's H-half -> [H0, T].
    ssq_partial = sbm.alloc_heap(shape=(H0, T), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_ssq_partial")
    ssq_other = sbm.alloc_heap(shape=(H0, T), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_ssq_other")

    sbm.open_scope()

    # x^2 -> [H0, T, H1_SHARD], reduce over the H1_SHARD free dim -> [H0, T].
    rmsnorm_square = sbm.alloc_heap(shape=(H0, T, H1_SHARD), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_square")
    nisa.activation(rmsnorm_square, op=nl.square, data=input_sb)
    rmsnorm_reduced = sbm.alloc_heap(shape=(H0, T), dtype=nl.float32, buffer=nl.sbuf, name="rmsnorm_reduced")
    nisa.tensor_reduce(rmsnorm_reduced, nl.add, rmsnorm_square, axis=2)

    # Sum across the H0 partitions via all-ones matmul -> [H0, T] (replicated across H0),
    # giving the per-token partial sum over this core's full H-half (H_PER_SHARD elements).
    final_reduced = nl.ndarray((H0, T), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
    nisa.nc_matmul(stationary=mm_reduce_const, moving=rmsnorm_reduced, dst=final_reduced)
    nisa.tensor_copy(dst=ssq_partial, src=final_reduced[...])

    sbm.pop_heap()  # rmsnorm_reduced
    sbm.pop_heap()  # rmsnorm_square
    sbm.close_scope()

    # 1e. Tiny cross-shard sum-reduce of the per-token partial ssq ([H0, T] ≈ 64 KB).
    nisa.sendrecv(
        dst=ssq_other,
        src=ssq_partial,
        send_to_rank=1 - shard_id,
        recv_from_rank=1 - shard_id,
        pipe_id=0,
    )
    # full ssq = partial_0 + partial_1 (exact full-H sum of squares).
    nisa.tensor_tensor(ssq_partial, ssq_partial, ssq_other, nl.add)

    # 1f. rsqrt(mean + eps) -> [H0, T].
    nisa.activation(
        ssq_partial,
        op=nl.rsqrt,
        data=ssq_partial,
        scale=1.0 / H,
        bias=eps_sb,
    )

    # 1g. input_sb = gamma_half * x_half * rsqrt_scale (scale broadcast over H1_SHARD).
    gamma_broadcast = gamma_sb.expand_dim(dim=1).broadcast(dim=1, size=T)
    scale_broadcast = ssq_partial.expand_dim(dim=2).broadcast(dim=2, size=H1_SHARD)
    nisa.tensor_tensor(input_sb, input_sb, gamma_broadcast, nl.multiply)
    nisa.tensor_tensor(input_sb, input_sb, scale_broadcast, nl.multiply)

    sbm.pop_heap()  # ssq_other
    sbm.pop_heap()  # ssq_partial
    sbm.pop_heap()  # mm_reduce_const
    sbm.pop_heap()  # eps_sb
    sbm.pop_heap()  # gamma_sb

    sbm.close_scope()


# Gate/Up weights are loaded once per I-split (the full weight tile would not fit in SBUF
# alongside both chunks' fp32 partials; splitting along I keeps the resident tile small
# while still loading each weight byte from HBM exactly once and reusing it for all chunks).
GATE_UP_I_SPLIT = 2
GATE_UP_I_PER_SPLIT = I // GATE_UP_I_SPLIT  # 1792 (I=3584 divisible by 2)


# ============================================================================
# Phase 2 helper: column-tiled gate/up matmul of one chunk against a resident
# FP8 weight (sub-)tile, draining into the provided fp32 SBUF buffer with dequant.
# ============================================================================


def _gate_up_matmul_chunk(
    input_fp8_pairs, weight_pairs_split, num_pairs, i_split_start, i_split_size, dst_fp32, dequant, psum_prefix, sbm
):
    """Run the FP8 double_row gate/up matmul for one chunk over the I range
    [i_split_start, i_split_start + i_split_size) into freshly-cycled PSUM banks, then
    drain that range into ``dst_fp32`` ([T, I]) applying the combined (weight*input)
    dequant scale.

    Both operands are FP8, so the matmul runs in ``matmul_perf_mode.double_row``: the
    contraction over the full H_PER_SHARD is packed two H0-blocks at a time. The stationary
    activation view is [H0, num_pairs, 2, T] and the moving weight view is
    [H0, num_pairs, 2, I_per_split]; ``select(dim=1, index=pair)`` yields the [H0, 2, M]/
    [H0, 2, N] tiles double_row expects (partition=K/2, free-dim-1=2, then M/N). double_row
    is mutually exclusive with column tiling, so tile_position/tile_size are NOT set.

    ``weight_pairs_split`` is the resident FP8 weight sub-tile for this I range (already
    DMA'd from HBM once and shared across chunks); this function performs NO weight DMA.
    """
    psums = []
    num_psums = div_ceil(i_split_size, PSUM_FMAX)
    for psum_idx in range(num_psums):
        psums.append(
            nl.ndarray(
                (H0, PSUM_FMAX),
                dtype=nl.float32,
                buffer=nl.psum,
                name=f"{psum_prefix}_psum_{psum_idx}",
                address=(0, psum_idx * PSUM_FMAX * 4),
            )
        )

    for pair in range(num_pairs):
        stationary = input_fp8_pairs.select(dim=1, index=pair)  # [H0, 2, T]
        weight_pair = weight_pairs_split.select(dim=1, index=pair)  # [H0, 2, I_per_split]
        for i_tile in TiledRange(i_split_size, PSUM_FMAX):
            nisa.nc_matmul(
                dst=psums[i_tile.index][nl.ds(0, T), 0 : i_tile.size],
                stationary=stationary,
                moving=weight_pair.slice(dim=2, start=i_tile.start_offset, end=i_tile.end_offset),
                perf_mode=matmul_perf_mode.double_row,
            )

    for i_tile in TiledRange(i_split_size, PSUM_FMAX):
        dst_start = i_split_start + i_tile.start_offset
        # Fuse the combined (weight*input) dequant scale into the PSUM drain (scale is
        # [T,1], broadcast over the I free dim) instead of a separate full-buffer pass.
        nisa.activation(
            dst=dst_fp32.slice(dim=1, start=dst_start, end=dst_start + i_tile.size),
            data=psums[i_tile.index][0:T, 0 : i_tile.size],
            scale=dequant,
            op=nl.copy,
        )


# ============================================================================
# Top-level entry point (BxS-tiled in BS_TILE_SIZE chunks, weights loaded once)
# ============================================================================


def mlp_tkg_llama3_70b_high_batch(
    params,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
    sbm: Optional[BufferManager] = None,
) -> list[nl.NkiTensor]:
    """T = batch*seq is tiled in 128-token chunks; the full MLP runs per LNC core.

    Each FP8 weight tile is DMA'd from HBM exactly once and reused by every BxS chunk.
    """
    if sbm is None:
        sbm = SbufManager(0, 200 * 1024, get_logger("mlp_tkg"))
        sbm.set_name_prefix("mlp_")

    # Resolve LNC shard id.
    _, _lnc, shard_id = get_verified_program_sharding_info("mlp_tkg", (0, 1))

    io_dtype = params.hidden_tensor.dtype
    fp8_w_dtype = (
        nl.float8_e4m3
        if str(params.up_proj_weights_tensor.dtype) == "float8e4"
        else params.up_proj_weights_tensor.dtype
    )

    T_full = params.batch_size * params.sequence_len
    H_in = params.hidden_size
    tile_size = min(BS_TILE_SIZE, T_full)
    num_chunks = div_ceil(T_full, tile_size)

    original_hidden = params.hidden_tensor
    hidden_2d = original_hidden.flatten_dims(start_dim=0, end_dim=1)

    B_out, S_out, _ = output_tensor_hbm.shape
    output_hbm_view = output_tensor_hbm.reshape((B_out * S_out, H_in))

    base_prefix = sbm.get_name_prefix()

    sbm.open_scope()

    # Resident transposed activation outputs for down (one per chunk). Allocated FIRST
    # (deepest on the heap) so the RMSNorm input tiles can be freed before the down phase.
    # The down stationary operand is FP8 (not bf16) and laid out pair-major
    # [I0, DOWN_NUM_DR_PAIRS, 2, T] so the down nc_matmul can run in double_row mode:
    # select(dim=1, pair) yields the [I0, 2, T] tile (partition=I/2-block, free-dim-1=2)
    # that double_row expects, packing two I0 contraction blocks per matmul.
    gate_up_sb_list = []
    for chunk in range(num_chunks):
        gate_up_sb_list.append(
            sbm.alloc_heap(
                shape=(I0, DOWN_NUM_DR_PAIRS, 2, T),
                dtype=fp8_w_dtype,
                buffer=nl.sbuf,
                name=f"gate_up_sbuf_chunk_{chunk}",
            )
        )

    # Down-projection activation FP8 (de)quant scales — resident across the gate/up and down
    # scopes. The down matmul runs FP8 double_row, so the down activations (SiLU(gate)*up) must
    # be FP8. For this specialized config the golden reference (_is_llama3_70b_specialized_config
    # branch, quantize_activations=True) quantizes the down activation to FP8 with the per-tensor
    # STATIC down_in_scale and dequantizes the matmul output by down_w_scale * down_in_scale, i.e.
    #   output = fp8(clamp(intermediate / down_in_scale)) @ down_w * (down_w_scale * down_in_scale).
    # We mirror that exactly: each per-chunk "act scale" buffer holds the static down_in_scale
    # (identical across chunks) and its reciprocal is the activation quant scale (1/down_in_scale).
    # A DYNAMIC per-token row-max scale would NOT reproduce the golden's static quant and fails
    # the accuracy comparator.
    down_w_scale_t = params.quant_params.down_w_scale
    down_in_scale_t = params.quant_params.down_in_scale
    down_w_scale_sb = sbm.alloc_heap((T, 1), dtype=nl.float32, buffer=nl.sbuf, name="down_w_scale_sb", align=4)
    nisa.dma_copy(
        dst=down_w_scale_sb,
        src=down_w_scale_t.slice(dim=0, start=0, end=T),
        dge_mode=_DGE_MODE_NONE,
    )
    # Static down-activation quant scale (= down_in_scale) and its reciprocal (= 1/down_in_scale),
    # resident so the dequant survives into the down phase. Loaded once per chunk slot up front.
    down_act_scale_list = []
    down_act_inv_list = []
    for chunk in range(num_chunks):
        down_act_scale = sbm.alloc_heap(
            (T, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"down_act_scale_{chunk}", align=4
        )
        down_act_inv = sbm.alloc_heap((T, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"down_act_inv_{chunk}", align=4)
        nisa.dma_copy(
            dst=down_act_scale,
            src=down_in_scale_t.slice(dim=0, start=0, end=T),
            dge_mode=_DGE_MODE_NONE,
        )
        nisa.reciprocal(dst=down_act_inv, data=down_act_scale)
        down_act_scale_list.append(down_act_scale)
        down_act_inv_list.append(down_act_inv)
    fp8_act_max_down = get_max_positive_value_for_dtype(fp8_w_dtype)

    # ------------------------------------------------------------------
    # Phase 1: RMSNorm + input load for every chunk into resident SBUF.
    # ------------------------------------------------------------------
    input_sb_list = []
    for chunk in range(num_chunks):
        chunk_start = chunk * tile_size
        chunk_size = min(tile_size, T_full - chunk_start)
        # This specialization assumes full 128-token chunks (T_full multiple of 128).
        params.batch_size = 1
        params.sequence_len = chunk_size
        params.hidden_tensor = hidden_2d.slice(dim=0, start=chunk_start, end=chunk_start + chunk_size).expand_dim(dim=0)

        input_sb = sbm.alloc_heap((H0, T, H1_SHARD), dtype=io_dtype, buffer=nl.sbuf, name=f"input_sbuf_chunk_{chunk}")
        sbm.set_name_prefix(f"{base_prefix}bxs_{chunk}_")
        _rmsnorm_chunk(params, sbm, shard_id, input_sb)
        sbm.set_name_prefix(base_prefix)
        input_sb_list.append(input_sb)

    # ------------------------------------------------------------------
    # Phase 2: Gate/Up projection. Load gate weights once, run both chunks;
    # load up weights once (overwriting slot), run both chunks; then per chunk
    # cross-shard reduce + SiLU(gate)*up + transpose into resident gate_up_sb.
    # ------------------------------------------------------------------
    gate_w = params.gate_proj_weights_tensor
    up_w = params.up_proj_weights_tensor
    gate_w_scale = params.quant_params.gate_w_scale
    up_w_scale = params.quant_params.up_w_scale
    gate_up_in_scale = params.quant_params.gate_up_in_scale  # static [128,1] activation dequant scale

    # H-shard the weights along the partition dim.
    h_offset = shard_id * H1_SHARD * H0
    gate_w_h = gate_w.slice(dim=0, start=h_offset, end=h_offset + H_PER_SHARD)
    up_w_h = up_w.slice(dim=0, start=h_offset, end=h_offset + H_PER_SHARD)

    sbm.open_scope()

    # Per-token weight dequant scales [T, 1] (FP32) — identical across chunks. Activations are
    # now FP8-quantized (X_fp8 = clamp(X / gate_up_in_scale)) so the matmul output must also be
    # multiplied by gate_up_in_scale to undo that scaling. Pre-combine both static scalars into
    # one fused drain scale: combined = w_scale * gate_up_in_scale.
    in_scale_sb = sbm.alloc_stack((T, 1), dtype=nl.float32, buffer=nl.sbuf, name="gate_up_in_scale_sb", align=4)
    nisa.dma_copy(
        dst=in_scale_sb,
        src=gate_up_in_scale.slice(dim=0, start=0, end=T),
        dge_mode=_DGE_MODE_NONE,
    )
    # quant_scale = 1 / gate_up_in_scale (used to scale the activation before clamping to FP8).
    in_quant_scale = sbm.alloc_stack(
        (T, 1), dtype=nl.float32, buffer=nl.sbuf, name="gate_up_in_quant_scale_sb", align=4
    )
    nisa.reciprocal(dst=in_quant_scale, data=in_scale_sb)

    gate_dequant = sbm.alloc_stack((T, 1), dtype=nl.float32, buffer=nl.sbuf, name="gate_w_scale_sb", align=4)
    up_dequant = sbm.alloc_stack((T, 1), dtype=nl.float32, buffer=nl.sbuf, name="up_w_scale_sb", align=4)
    nisa.dma_copy(dst=gate_dequant, src=gate_w_scale.slice(dim=0, start=0, end=T), dge_mode=_DGE_MODE_NONE)
    nisa.dma_copy(dst=up_dequant, src=up_w_scale.slice(dim=0, start=0, end=T), dge_mode=_DGE_MODE_NONE)
    # combined = w_scale * gate_up_in_scale (nisa.activation computes data * scale).
    nisa.activation(dst=gate_dequant, op=nl.copy, data=gate_dequant, scale=in_scale_sb)
    nisa.activation(dst=up_dequant, op=nl.copy, data=up_dequant, scale=in_scale_sb)

    # FP8-quantize each chunk's RMSNorm activation once (shared by gate AND up). The FP8 tile is
    # laid out [H0, 16, 2, T] (pair-major) so the double_row "2" dim has stride T (a multiple of
    # 16, as the HW AP requires). The scale is applied in-place on the bf16 ``input_sb`` (no longer
    # needed after this) to avoid a large fp32 scratch buffer, then clamped + cast to FP8.
    fp8_act_max = get_max_positive_value_for_dtype(fp8_w_dtype)
    input_fp8_list = []
    for chunk in range(num_chunks):
        # Contiguous [H0, 16, 2, T] FP8 destination -> select(pair) yields [H0, 2, T] with
        # 2-dim stride = T = 128 (128 % 16 == 0).
        input_fp8 = sbm.alloc_stack(
            (H0, GATE_UP_NUM_DR_PAIRS, 2, T),
            dtype=fp8_w_dtype,
            buffer=nl.sbuf,
            name=f"input_fp8_chunk_{chunk}",
            align=4,
        )
        src = input_sb_list[chunk]
        # scaled (bf16, in-place) = X * (1/gate_up_in_scale).
        nisa.activation(dst=src, op=nl.copy, data=src, scale=in_quant_scale)
        # Permuted source view matching the [H0, 16, 2, T] dst element order:
        #   [H0, T, 32] -> [H0, T, 16, 2] -> [H0, 16, 2, T].
        src_pairs = input_sb_list[chunk].reshape_dim(dim=2, shape=(GATE_UP_NUM_DR_PAIRS, 2)).permute(dims=[0, 2, 3, 1])
        # clamp to [-MAX, MAX] and cast to FP8 in pair-major layout.
        nisa.tensor_scalar(
            dst=input_fp8,
            data=src_pairs,
            op0=nl.minimum,
            operand0=fp8_act_max,
            op1=nl.maximum,
            operand1=-fp8_act_max,
        )
        input_fp8_list.append(input_fp8)

    # The bf16 ``input_sb`` heap tiles have been consumed into ``input_fp8`` and are no longer
    # needed; free them (LIFO) now to reclaim SBUF for the gate/up fp32 partials below.
    for chunk in range(num_chunks):
        sbm.pop_heap()  # input_sb_list[num_chunks-1-chunk]

    # Resident fp32 gate/up partials per chunk (kept through the cross-shard reduce).
    gate_sb_fp32_list = []
    up_sb_fp32_list = []
    for chunk in range(num_chunks):
        gate_sb_fp32_list.append(
            sbm.alloc_stack((T, I), dtype=nl.float32, name=f"gate_sbuf_fp32_{chunk}", buffer=nl.sbuf, align=4)
        )
        up_sb_fp32_list.append(
            sbm.alloc_stack((T, I), dtype=nl.float32, name=f"up_sbuf_fp32_{chunk}", buffer=nl.sbuf, align=4)
        )

    # --- Gate + Up matmuls: weights loaded once per I-split, reused by all chunks. ---
    # weight_tile lives in an inner scope so it is freed before the cross-shard recv
    # buffer is allocated (they never need to coexist), keeping peak SBUF in budget.
    # GATE_UP_NUM_HTILES == 1, so the single htile covers the full H_PER_SHARD.
    h1_tiles = GATE_UP_NUM_128_PER_HTILE  # 32
    gate_w_2d = gate_w_h.reshape_dim(dim=0, shape=(H0, H1_SHARD)).slice(dim=1, start=0, end=h1_tiles)
    up_w_2d = up_w_h.reshape_dim(dim=0, shape=(H0, H1_SHARD)).slice(dim=1, start=0, end=h1_tiles)

    sbm.open_scope()
    # Double-buffered FP8 weight sub-tiles (one I-split worth each — shared across all
    # chunks). Two physical slots let the next (is_up, I-split) weight DMA prefetch into
    # the alternate slot while the current split's matmuls still read its slot, breaking
    # the load<->compute serialization (anti-dep + flow-dep) that previously left the PE
    # engine idle for the full ~21us weight-DMA latency at each of the 4 transitions.
    GATE_UP_NUM_W_SLOTS = 2
    weight_tiles = []
    for slot in range(GATE_UP_NUM_W_SLOTS):
        weight_tiles.append(
            sbm.alloc_stack(
                shape=(H0, GATE_UP_NUM_128_PER_HTILE, GATE_UP_I_PER_SPLIT),
                dtype=fp8_w_dtype,
                buffer=nl.sbuf,
                name=f"gate_up_w_tile_{slot}",
            )
        )

    # Iterate (is_up, I-split) as a single flattened sequence so we can round-robin over
    # the two resident slots; the scheduler then overlaps load N+1 (into the other slot)
    # with the matmuls of load N.
    for load_idx in range(2 * GATE_UP_I_SPLIT):
        is_up = load_idx // GATE_UP_I_SPLIT
        split = load_idx % GATE_UP_I_SPLIT
        weight_2d = up_w_2d if is_up == 1 else gate_w_2d
        dst_list = up_sb_fp32_list if is_up == 1 else gate_sb_fp32_list
        dequant = up_dequant if is_up == 1 else gate_dequant
        tag = "up" if is_up == 1 else "gate"

        i_start = split * GATE_UP_I_PER_SPLIT
        slot = load_idx % GATE_UP_NUM_W_SLOTS
        # Load this weight's I-split sub-tile once into its slot: [H0, h1_tiles, I_per_split].
        weight_hbm_split = weight_2d.slice(dim=2, start=i_start, end=i_start + GATE_UP_I_PER_SPLIT)
        weight_sb = (
            weight_tiles[slot].slice(dim=1, start=0, end=h1_tiles).slice(dim=2, start=0, end=GATE_UP_I_PER_SPLIT)
        )
        nisa.dma_copy(dst=weight_sb, src=weight_hbm_split, dge_mode=_DGE_MODE_NONE)
        # Pair view for double_row: [H0, 32, I_per_split] -> [H0, 16, 2, I_per_split].
        weight_pairs = weight_sb.reshape_dim(dim=1, shape=(GATE_UP_NUM_DR_PAIRS, 2))

        for chunk in range(num_chunks):
            _gate_up_matmul_chunk(
                input_fp8_list[chunk],
                weight_pairs,
                GATE_UP_NUM_DR_PAIRS,
                i_start,
                GATE_UP_I_PER_SPLIT,
                dst_list[chunk],
                dequant,
                f"{tag}_{sbm.get_name_prefix()}_c{chunk}_s{split}",
                sbm,
            )

    sbm.close_scope()  # weight_tiles freed

    # Cross-shard recv buffer (allocated after weight_tile freed). DOUBLE-BUFFERED per chunk:
    # the SiLU(gate)*up + dynamic-quant chain is a long Vector/Scalar-Engine sequence, while the
    # subsequent per-I0-block transpose runs on the PE. With a single shared recv/mul scratch the
    # two chunks serialize on an anti-dependency (chunk 1 cannot overwrite the scratch until
    # chunk 0's transpose has finished READING it), so chunk 0's PE transpose and chunk 1's
    # Vector SiLU chain ran back-to-back. Giving each chunk its own scratch lets the scheduler
    # overlap chunk 0's transpose (PE) with chunk 1's SiLU/quant (Vector/Scalar), hiding the
    # transpose under Vector work that was previously stalled behind it.
    gate_up_recv_list = []
    gate_up_mul_list = []
    for chunk in range(num_chunks):
        gate_up_recv_list.append(
            sbm.alloc_stack((T, I), dtype=nl.float32, buffer=nl.sbuf, name=f"gate_up_recv_fp32_{chunk}")
        )
        # SiLU(gate)*up result (scaled bf16). The down matmul runs FP8 double_row, so the activation
        # must be FP8. The static activation quant scale (1/down_in_scale) is applied here while
        # T is still on the partition dim, BEFORE the bf16 transpose; the transposed bf16 tile is
        # then just clamped+cast to FP8 on the PSUM->SBUF drain (no per-element scale needed there).
        gate_up_mul_list.append(
            sbm.alloc_stack((T, I), dtype=io_dtype, buffer=nl.sbuf, name=f"gate_up_mul_bf16_{chunk}")
        )

    # --- Per chunk: cross-shard reduce gate & up, SiLU(gate)*up, scale, transpose, FP8-cast ---
    for chunk in range(num_chunks):
        gate_sb_fp32 = gate_sb_fp32_list[chunk]
        up_sb_fp32 = up_sb_fp32_list[chunk]
        gate_up_recv = gate_up_recv_list[chunk]
        gate_up_mul = gate_up_mul_list[chunk]

        # Cross-shard reduce gate, then SiLU.
        nisa.sendrecv(
            src=gate_sb_fp32, dst=gate_up_recv, send_to_rank=1 - shard_id, recv_from_rank=1 - shard_id, pipe_id=0
        )
        nisa.tensor_tensor(dst=gate_sb_fp32, data1=gate_sb_fp32, data2=gate_up_recv, op=nl.add)
        nisa.activation(dst=gate_sb_fp32[:, :], op=nl.silu, data=gate_sb_fp32, scale=1.0)

        # Cross-shard reduce up, then SiLU(gate) * up.
        nisa.sendrecv(
            src=up_sb_fp32, dst=gate_up_recv, send_to_rank=1 - shard_id, recv_from_rank=1 - shard_id, pipe_id=0
        )
        nisa.tensor_tensor(dst=up_sb_fp32, data1=up_sb_fp32, data2=gate_up_recv, op=nl.add)
        # gate_up_recv (now free) reused as fp32 product scratch for the SiLU(gate)*up product.
        nisa.tensor_tensor(dst=gate_up_recv, data1=gate_sb_fp32, data2=up_sb_fp32, op=nl.multiply)

        # STATIC FP8 quant of the down activation, matching the golden reference:
        # scaled (bf16) = product / down_in_scale = product * down_act_inv (per-tensor static scale,
        # broadcast over the I free dim with T on the partition). The down matmul output is later
        # dequantized by down_w_scale * down_in_scale = down_w_scale * down_act_scale.
        down_act_inv = down_act_inv_list[chunk]
        nisa.activation(dst=gate_up_mul, op=nl.copy, data=gate_up_recv, scale=down_act_inv)

        # Transpose bf16 [T, I] per I0-block, then clamp+cast to FP8 into the pair-major
        # gate_up_sb [I0, DOWN_NUM_DR_PAIRS, 2, T]. Each I0-block i_tile maps to
        # (pair = i_tile // 2, block = i_tile % 2). PSUM banks are partitioned by chunk
        # (4 banks each of the 8) so the two chunks' transposes can be in flight concurrently
        # without a PSUM anti-dependency between them.
        gate_up_sb = gate_up_sb_list[chunk]
        psum_bank_base = (chunk % 2) * (PSUM_BMAX // 2)
        for i_tile in TiledRange(I, I0):
            psum_idx = psum_bank_base + (i_tile.index % (PSUM_BMAX // 2))
            tp_psum = nl.ndarray(
                (i_tile.size, T),
                dtype=gate_up_mul.dtype,
                buffer=nl.psum,
                name=f"{sbm.get_name_prefix()}transpose_psum_c{chunk}_{i_tile.index}",
                address=(0, psum_idx * PSUM_FMAX * 4),
            )
            nisa.nc_transpose(dst=tp_psum, data=gate_up_mul[0:T, nl.ds(i_tile.index * I0, i_tile.size)])
            nisa.tensor_scalar(
                dst=gate_up_sb.slice(dim=0, start=0, end=i_tile.size)
                .slice(dim=1, start=i_tile.index // 2, end=i_tile.index // 2 + 1)
                .slice(dim=2, start=i_tile.index % 2, end=i_tile.index % 2 + 1)
                .slice(dim=3, start=0, end=T),
                data=tp_psum,
                op0=nl.minimum,
                operand0=fp8_act_max_down,
                op1=nl.maximum,
                operand1=-fp8_act_max_down,
            )

    sbm.close_scope()  # gate/up scratch (weight_tile, fp32 partials, recv, dequant)

    # ------------------------------------------------------------------
    # Phase 3: Down projection (FP8 double_row). Load each FP8 down weight pair-tile once,
    # run both chunks against it, drain (fused dequant) and store each chunk's slab.
    # ------------------------------------------------------------------
    down_w = params.down_proj_weights_tensor
    down_w_view = down_w.slice(dim=1, start=shard_id * H_PER_SHARD, end=(shard_id + 1) * H_PER_SHARD)

    sbm.open_scope()

    # Output SBUF, reused per chunk (stored to HBM before the next chunk drains).
    down_sb = sbm.alloc_stack((T, H_PER_SHARD), dtype=io_dtype, buffer=nl.sbuf, name="down_sbuf")
    # Per-chunk fused drain scale = down_w_scale * down_act_scale[chunk] (undoes the activation
    # quant per token, then applies the down weight dequant). Recomputed per chunk below.
    down_dequant = sbm.alloc_stack((T, 1), dtype=nl.float32, buffer=nl.sbuf, name="down_dequant_sb", align=4)

    # FP8 weight ring buffer — one pair-tile per (I0-block-pair × HTile), all resident across
    # chunks. Each pair-tile is [I0, 2, DOWN_HTILE]: the size-2 free-dim-1 selects the two
    # consecutive I0 contraction blocks that double_row packs (stride DOWN_HTILE, a multiple of 16).
    down_weight_tiles = []
    for w_idx in range(DOWN_NUM_W_TILES):
        down_weight_tiles.append(
            sbm.alloc_stack(
                shape=(I0, 2, DOWN_HTILE),
                dtype=fp8_w_dtype,
                buffer=nl.sbuf,
                name=f"down_w_tile_{w_idx}",
            )
        )

    for hidden_tiles in TiledRange(H_PER_SHARD, DOWN_HTILE):
        h_off = hidden_tiles.start_offset

        # Load all down weight pair-tiles once (shared across chunks). Pair p covers I0-blocks
        # 2p and 2p+1 -> HBM rows [2p*I0 : (2p+2)*I0], viewed as [I0, 2, htile].
        for pair in range(DOWN_NUM_DR_PAIRS):
            weight_idx = (hidden_tiles.index * DOWN_NUM_DR_PAIRS + pair) % DOWN_NUM_W_TILES
            weight_sb = down_weight_tiles[weight_idx].slice(dim=2, start=0, end=hidden_tiles.size)
            weight_view = (
                down_w_view.slice(dim=0, start=pair * 2 * I0, end=(pair + 1) * 2 * I0)
                .slice(dim=1, start=h_off, end=h_off + hidden_tiles.size)
                .reshape_dim(dim=0, shape=(2, I0))  # [2, I0, htile]
                .permute(dims=[1, 0, 2])  # [I0, 2, htile]
            )
            nisa.dma_copy(dst=weight_sb, src=weight_view, dge_mode=_DGE_MODE_NONE)

        # For each chunk, matmul against the resident weights, drain and store.
        for chunk in range(num_chunks):
            gate_up_sb = gate_up_sb_list[chunk]

            # Fused drain scale: down_w_scale * down_act_scale[chunk] = down_w_scale * down_in_scale.
            nisa.activation(
                dst=down_dequant,
                op=nl.copy,
                data=down_w_scale_sb,
                scale=down_act_scale_list[chunk],
            )

            result_psums = []
            for psum_idx in range(NUM_DOWN_PSUMS):
                result_psums.append(
                    nl.ndarray(
                        shape=(I0, PSUM_FMAX),
                        dtype=nl.float32,
                        buffer=nl.psum,
                        name=f"down_psum_{sbm.get_name_prefix()}_c{chunk}_{hidden_tiles.index}_{psum_idx}",
                        address=(0, psum_idx * PSUM_FMAX * 4),
                    )
                )

            # FP8 double_row: stationary [I0, 2, T], moving [I0, 2, compute_size]; the size-2
            # free-dim-1 packs two I0 contraction blocks per matmul (DOWN_NUM_DR_PAIRS instead
            # of NUM_I_TILES matmuls). double_row is mutually exclusive with column tiling, so
            # tile_position/tile_size are NOT set.
            for pair in range(DOWN_NUM_DR_PAIRS):
                weight_idx = (hidden_tiles.index * DOWN_NUM_DR_PAIRS + pair) % DOWN_NUM_W_TILES
                hidden_pair = gate_up_sb.select(dim=1, index=pair)  # [I0, 2, T]
                weight_pair = down_weight_tiles[weight_idx]  # [I0, 2, DOWN_HTILE]
                for compute_tile in TiledRange(hidden_tiles.size, PSUM_FMAX):
                    nisa.nc_matmul(
                        dst=result_psums[compute_tile.index][nl.ds(0, T), 0 : compute_tile.size],
                        stationary=hidden_pair,
                        moving=weight_pair.slice(dim=2, start=compute_tile.start_offset, end=compute_tile.end_offset),
                        perf_mode=matmul_perf_mode.double_row,
                    )

            # Drain PSUM -> SBUF with fused dequant scale.
            for compute_tile in TiledRange(hidden_tiles.size, PSUM_FMAX):
                dst_offset = h_off + compute_tile.index * PSUM_FMAX
                nisa.activation(
                    dst=down_sb.slice(dim=1, start=dst_offset, end=dst_offset + compute_tile.size),
                    data=result_psums[compute_tile.index][nl.ds(0, T), 0 : compute_tile.size],
                    scale=down_dequant,
                    op=nl.copy,
                )

            # Store this chunk's H_PER_SHARD slab to HBM.
            chunk_start = chunk * tile_size
            chunk_size = min(tile_size, T_full - chunk_start)
            out_chunk = output_hbm_view.slice(dim=0, start=chunk_start, end=chunk_start + chunk_size)
            nisa.dma_copy(
                dst=out_chunk.slice(dim=1, start=shard_id * H_PER_SHARD, end=(shard_id + 1) * H_PER_SHARD),
                src=down_sb.slice(dim=0, start=0, end=chunk_size).slice(dim=1, start=0, end=H_PER_SHARD),
            )

    sbm.close_scope()  # down scratch

    # Free resident heap tiles (LIFO): per-chunk down_act_inv/down_act_scale (reverse chunk
    # order), down_w_scale_sb, then the per-chunk gate_up_sb activation outputs.
    for chunk in range(num_chunks):
        sbm.pop_heap()  # down_act_inv_list[num_chunks-1-chunk]
        sbm.pop_heap()  # down_act_scale_list[num_chunks-1-chunk]
    sbm.pop_heap()  # down_w_scale_sb
    for chunk in range(num_chunks):
        sbm.pop_heap()  # gate_up_sb_list[num_chunks-1-chunk]

    sbm.close_scope()  # outer

    return [output_hbm_view.reshape((B_out, S_out, H_in))]
