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
Constant-Max Attention Kernel

Computes: Output = softmax(Q @ K.T * scale - const_max) @ V

The two matmuls are referred to throughout as:
  - MM1: Q @ K.T (scores)
  - MM2: P @ V   (weighted value accumulation)

By accepting a pre-known constant max, this kernel eliminates:
  1. The DMA transpose between MM1 and MM2, which serializes with collective
     communication (CC) and blocks CC overlap.
  2. The online softmax max-reduction pass.

This yields a simpler pipeline and better performance when a constant max is available.
See the attention_const_max docstring for how to choose the max value.
"""

from typing import Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from .. import neurotile as nt

# Tiling constants
_P = nl.tile_size.pmax  # 128: partition dimension
_SK_BLOCK = 2048  # KV streaming block size (16 tiles of 128)
_O_TILE_SIZE = nl.tile_size.gemm_stationary_fmax  # 128: output tile size along Sq (MM2 stationary)
_KV_TILES_PER_BLOCK = _SK_BLOCK // _P  # 16: number of K/V tiles per KV block
_EPS = 1e-9  # Floor for sum_exp to prevent division by zero

# PSUM bank allocation:
#   2 banks for MM1 scores, up to min(psum_num_banks - 2, 8) banks for O_aug accumulation.
#
# Q and O blocks share the same Sq-dimension size. O tiles are staged in PSUM and
# accumulate across all KV blocks. The number of available O_aug banks is derived at
# trace time from nl.tile_size.psum_num_banks. Since probabilities (MM1 output / MM2 stationary)
# have 128 on the free axis, the maximum O block is _MAX_O_AUG_BANKS × gemm_stationary_fmax Sq elements.
#
# On trn2 (gen3), MM1 produces f32 PSUM and the moving operand (Q) is limited to
# gemm_moving_fmax = 512, so we choose max Q/O block = 512. On trn3 (gen4), MM1
# produces bf16 PSUM with no such limit, so the full O_aug capacity is used.
#
# Q stays resident in SBUF while we stream all KV blocks, prefetching the next block
# behind the current block's compute.


@nki.jit
def attention_const_max(
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    v_hbm: nl.NkiTensor,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: Optional[float] = None,
) -> nl.NkiTensor:
    """Constant-max attention: softmax(Q@K.T * softmax_scale - softmax_max) @ V.

    Computes attention assuming a pre-known global softmax max.

    Dimensions:
        N: Number of attention heads (batch of heads)
        d: Head dimension (must be pmax=128)
        Sq: Query sequence length
        Sk: Key/Value sequence length (must be multiple of pmax=128)

    Args:
        q_hbm (nl.NkiTensor): [N, d, Sq] bf16 @ HBM, Query
        k_hbm (nl.NkiTensor): [N, d, Sk] bf16 @ HBM, Key
        v_hbm (nl.NkiTensor): [N, Sk, d] bf16 @ HBM, Value
        softmax_max (Union[float, nl.NkiTensor]): Upper bound on (scores * scale). Output is
            mathematically invariant to this value; it only affects numerical precision. A
            reasonable estimate suffices. May be either:
              - a Python scalar (compile-time constant, broadcast to all partitions), or
              - a [128, 1] f32 HBM tensor holding a single broadcast max replicated across
                the 128 partitions. Use the tensor form to supply a data-dependent max
                computed at runtime rather than baked in at trace time.
        softmax_scale (Optional[float]): Scaling factor applied to Q. Defaults to 1/sqrt(d).

    Returns:
        output (nl.NkiTensor): [N, Sq, d] bf16 @ HBM, Normalized attention output

    Notes:
        - Requires std(Q) ≤ 1 and std(K) ≤ 1 for bf16 accuracy with softmax_max=4–5.
          This holds when QK-RMSNorm is applied after projection.
        - A softmax_max that is too low causes exp() overflow (inf/NaN in output).
          A softmax_max that is too high clips all scores toward zero, losing precision.
        - For inputs with significantly larger std use a standard online-softmax kernel.

    Choosing softmax_max from QK-RMSNorm weights:
        When QK-RMSNorm is applied, the max can be derived from the gamma weights:

            σ = sqrt(sum(γ_q² * γ_k²) / d)
            softmax_max ≈ 0.88 * σ * sqrt(2 * ln(Sk))     # Gumbel expected row-max

        As a safety check, compute the hard bound (Cauchy-Schwarz worst case):

            hard_bound = sqrt(d) * max|γ_q| * max|γ_k|

        If hard_bound - softmax_max > ~80, the score range is too wide for constant-max
        and the block should fall back to online-softmax attention.

    Pseudocode:
        for each head:
            for each Q-block:
                Q = load(q_hbm[head, Q-block]) * softmax_scale
                for each KV-block:
                    for each kv tile:
                        S = Q @ K_tile.T             # MM1: scores
                        P = exp(S - max)             # fused eviction + exp
                        O_aug += P @ [V_tile | 1]    # MM2: output + sum_exp
                O = O_aug[:, :d] / O_aug[:, d]       # normalize
                store(O)
    """
    Nq, d, Sq = q_hbm.shape
    _, d_k, Sk = k_hbm.shape
    Sv, d_v = v_hbm.shape[1], v_hbm.shape[2]

    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale

    kernel_assert(d == _P, f"d must be pmax (128), got {d}")
    kernel_assert(d_k == d, f"K head dim must match Q, got {d_k=}, {d=}")
    kernel_assert(d_v == d, f"V head dim must match Q, got {d_v=}, {d=}")
    kernel_assert(Sv == Sk, f"V seq length must match K, got {Sv=}, {Sk=}")
    kernel_assert(Sk % _P == 0, f"Sk must be multiple of pmax, got {Sk}")
    kernel_assert(Sq >= 1, f"Sq must be at least 1, got {Sq}")

    # Prepare the constant max for the exp path once, up front. A scalar passes through as a
    # (pre-signed) float; a runtime HBM tensor is loaded into SBUF. See _prepare_softmax_max.
    prepared_softmax_max = _prepare_softmax_max(softmax_max)

    o_hbm = nl.ndarray(shape=(Nq, Sq, d), dtype=q_hbm.dtype, buffer=nl.shared_hbm)

    # Shard the independent (N, Sq) work across LNC cores by flattening N*Sq and splitting.
    # Q is [N, d, Sq] and O is [N, Sq, d] — both have N and Sq as concurrent dims.
    # K/V are fully replicated (all cores need all KV tokens).
    q_local, o_local, k_local, v_local = _shard_inputs(q_hbm, k_hbm, v_hbm, o_hbm)

    _attention_const_max_compute(
        q_local,
        k_local,
        v_local,
        o_local,
        softmax_max=prepared_softmax_max,
        softmax_scale=softmax_scale,
    )

    return o_hbm


def _prepare_softmax_max(softmax_max):
    """Prepare the constant softmax max for the exp path, pre-signing it once.

    The exp path needs a specific sign convention:
      - vector_engine exponential subtracts ``max_value`` → needs positive max.
      - activation adds ``bias`` → needs negated max, so exp(scores * scale + (-max)).

    Returns:
        float if softmax_max is a scalar, else a [_P, 1] f32 SBUF tensor.
    """
    if isinstance(softmax_max, (int, float)):
        return softmax_max if _has_vector_engine_exp() else -softmax_max

    max_sbuf = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=max_sbuf, src=softmax_max[:_P, 0:1])
    if _has_vector_engine_exp():
        return max_sbuf

    neg_max_sbuf = nl.ndarray(shape=(_P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg_max_sbuf, data=max_sbuf, op0=nl.multiply, operand0=-1.0)
    return neg_max_sbuf


def _shard_inputs(q_hbm, k_hbm, v_hbm, o_hbm):
    """Shard Q and O across LNC cores; K/V are replicated.

    Both N (heads) and Sq (query sequence) are fully concurrent — each (head, sq_position)
    is independent. Since these dims are contiguous in Q [N, d, Sq] and O [N, Sq, d],
    we shard on heads first (if evenly divisible), otherwise on Sq.

    Returns:
        (q_local, o_local, k_local, v_local) — sliced views for this core.
    """
    Nq, _, Sq = q_hbm.shape
    num_cores = nl.num_programs()
    core_id = nl.program_id(0)

    if Nq % num_cores == 0:
        # Shard on heads: each core gets a contiguous slice of heads
        heads_per_core = Nq // num_cores
        head_start = core_id * heads_per_core
        head_end = head_start + heads_per_core
        q_local = q_hbm[head_start:head_end, :, :]
        o_local = o_hbm[head_start:head_end, :, :]
        k_local = k_hbm[head_start:head_end, :, :]
        v_local = v_hbm[head_start:head_end, :, :]
    else:
        # Shard on Sq: each core gets a contiguous slice of query positions
        sq_per_core = Sq // num_cores
        sq_start = core_id * sq_per_core
        # Last core takes the remainder
        sq_end = Sq if (core_id == num_cores - 1) else (sq_start + sq_per_core)
        q_local = q_hbm[:, :, sq_start:sq_end]
        o_local = o_hbm[:, sq_start:sq_end, :]
        k_local = k_hbm
        v_local = v_hbm

    return q_local, o_local, k_local, v_local


def _attention_const_max_compute(
    q_hbm,
    k_hbm,
    v_hbm,
    o_hbm,
    softmax_max,
    softmax_scale,
):
    """Core attention computation — agnostic to LNC sharding.

    Operates on the local slice of Q/O assigned to this core.
    K/V are the full (or head-sliced) tensors.

    Args:
        q_hbm: [N_local, d, Sq_local] bf16 @ HBM
        k_hbm: [N_local, d, Sk] bf16 @ HBM
        v_hbm: [N_local, Sk, d] bf16 @ HBM
        o_hbm: [N_local, Sq_local, d] bf16 @ HBM (output)
        softmax_max: pre-signed max for the exp path (float or [_P, 1] SBUF tensor)
        softmax_scale: scaling factor for Q
    """
    Nq, d, _ = q_hbm.shape
    Sk = k_hbm.shape[2]

    kernel_assert(nl.tile_size.psum_num_banks >= 6, f"need at least 6 PSUM banks, have {nl.tile_size.psum_num_banks}")
    _MM1_BUFFER_BANKS = 2
    _MAX_O_AUG_BANKS = min(nl.tile_size.psum_num_banks - _MM1_BUFFER_BANKS, 8)

    has_bf16_psum = nisa.get_nc_version() >= nisa.nc_version.gen4
    max_sq_block_size = (
        _MAX_O_AUG_BANKS * nl.tile_size.gemm_stationary_fmax if has_bf16_psum else nl.tile_size.gemm_moving_fmax
    )

    num_full_kv_blocks = Sk // _SK_BLOCK
    kv_remainder = Sk % _SK_BLOCK
    kv_remainder_tiles = kv_remainder // _P

    for head_idx in range(Nq):
        # Q is [d, Sq] per head, partitioned into one or more Q blocks of size
        # [d, max_sq_block_size]. Each Q block is loaded once into SBUF and stays
        # resident while we stream all KV blocks through it.
        q_block_grid = nt.tiles(q_hbm[head_idx], tile_size=(_P, max_sq_block_size))
        # O: [Sq, d] per head — tile along Sq at _O_TILE_SIZE for per-tile normalization/store
        o_tiles = nt.tiles(o_hbm[head_idx], tile_size=(_O_TILE_SIZE, d))

        for q_block_idx in range(q_block_grid.shape[1]):
            # Load Q block into SBUF — stays resident for all KV iterations.
            # On the activation path, softmax_scale is fused into the exp eviction
            # (scale param). On the exponential path (no scale param), pre-scale Q here.
            q_loaded = q_block_grid[0, q_block_idx].load()
            q_block = q_loaded.data
            if _has_vector_engine_exp():
                nisa.tensor_scalar(dst=q_block, data=q_block, op0=nl.multiply, operand0=softmax_scale)

            sq_block_size = q_loaded.element_shape[1]

            # Allocate output accumulator in PSUM — one tile per Sq output slice, each on
            # its own bank. psum_pool handles the bank placement; we accumulate across all
            # KV blocks then normalize and store.
            num_output_tiles = (sq_block_size + _O_TILE_SIZE - 1) // _O_TILE_SIZE
            kernel_assert(
                num_output_tiles <= _MAX_O_AUG_BANKS,
                "Q block too large: O_aug tiles exceed available PSUM banks.",
            )
            # Output accumulator width is d+1: columns 0..d-1 hold the weighted output
            # (P @ V), and column d accumulates sum_exp (P @ 1) for normalization.
            o_aug_psums = nt.psum_pool(
                tile_size=(_O_TILE_SIZE, d + 1),
                element_shape=(sq_block_size, d + 1),
            )
            # Stream full KV blocks with double-buffered prefetch
            if num_full_kv_blocks > 0:
                k_view = nt.blocks(
                    k_hbm[head_idx, :, : num_full_kv_blocks * _SK_BLOCK],
                    tile_size=(_P, d),
                    block_size=(1, _KV_TILES_PER_BLOCK),
                )
                v_view = nt.blocks(
                    v_hbm[head_idx, : num_full_kv_blocks * _SK_BLOCK, :],
                    tile_size=(_P, d),
                    block_size=(_KV_TILES_PER_BLOCK, 1),
                )
                # Both K and V have a singleton block dim (K: row=1, V: col=1).
                # Select it away before streaming so .load() returns the tile grid directly.
                k_stream = k_view[0, :].stream(buffer_count=2)
                v_stream = v_view[:, 0].stream(buffer_count=2)

                for kv_block_idx in nl.sequential_range(num_full_kv_blocks):
                    k_loaded = k_stream.load(kv_block_idx)
                    v_loaded = v_stream.load(kv_block_idx)
                    # k_loaded has tile grid (1, 16) — select the single tile-row;
                    # v_loaded has tile grid (16, 1) — passed directly.
                    _process_kv_block(
                        k_block=k_loaded[0],
                        v_block=v_loaded,
                        q_block=q_block,
                        o_aug_psums=o_aug_psums,
                        softmax_max=softmax_max,
                        softmax_scale=softmax_scale,
                        is_first_block=(kv_block_idx == 0),
                    )

            # Process remaining KV tokens that don't fill a complete block
            if kv_remainder_tiles > 0:
                rem_start = num_full_kv_blocks * _SK_BLOCK
                k_rem_view = nt.tiles(
                    k_hbm[head_idx, :, rem_start : rem_start + kv_remainder],
                    tile_size=(_P, d),
                )
                v_rem_view = nt.tiles(
                    v_hbm[head_idx, rem_start : rem_start + kv_remainder, :],
                    tile_size=(_P, d),
                )
                k_rem_loaded = k_rem_view[0, :].load()
                v_rem_loaded = v_rem_view[:, 0].load()
                _process_kv_block(
                    k_block=k_rem_loaded,
                    v_block=v_rem_loaded,
                    q_block=q_block,
                    o_aug_psums=o_aug_psums,
                    softmax_max=softmax_max,
                    softmax_scale=softmax_scale,
                    is_first_block=(num_full_kv_blocks == 0),
                )

            # Normalize each output tile by its accumulated sum_exp, then store to HBM
            o_block_start = q_block_idx * (max_sq_block_size // _O_TILE_SIZE)
            for output_tile_idx in range(num_output_tiles):
                o_out = _normalize(o_aug_psums[output_tile_idx].data)
                o_tiles[o_block_start + output_tile_idx, 0].store(o_out)


def _process_kv_block(
    k_block,
    v_block,
    q_block,
    o_aug_psums,
    softmax_max,
    softmax_scale,
    is_first_block,
):
    """Process one KV block: scores (MM1) → exp → weighted accumulation (MM2).

    V is augmented with a ones column [V|1] so that MM2 simultaneously accumulates
    the weighted output and sum_exp into o_aug_psums.

    Args:
        k_block: NDSlice, 1-D tile grid of K tiles in SBUF — each tile is (128, 128)
        v_block: NDSlice, (num_kv_tiles, 1) tile grid in SBUF — each tile is (128, d)
        q_block: [d, sq_block_size] in SBUF — pre-scaled Q block
        o_aug_psums: NDSlice (psum_pool) — grid of [_O_TILE_SIZE, d+1] PSUM accumulators
        softmax_max: pre-signed max for the exp path — a float, or a per-partition [_P, 1]
            SBUF tensor. Positive for the vector_engine exponential's max_value, negated for the
            activation bias (see _prepare_softmax_max).
        is_first_block: if True, first write to o_aug (no accumulate)
    """
    d = _P
    num_kv_tiles = k_block.shape[0]
    num_output_tiles = o_aug_psums.shape[0]

    scores_dtype = nl.bfloat16 if nisa.get_nc_version() >= nisa.nc_version.gen4 else nl.float32
    sq_block_size = o_aug_psums.element_shape[0]

    copy_engine = nisa.engine.scalar if _has_vector_engine_exp() else nisa.engine.vector

    # Triple-buffer V_aug only when the copy runs on the Vector engine (gen3): the extra
    # rotating buffer relaxes the write-after-read hazard on buffer reuse, so the next tile's
    # copy starts earlier and MM2 (tensor engine) stalls less. When the copy is on the Scalar
    # engine (gen4) a third buffer perturbs an already well-packed schedule, so double-buffer.
    #
    # V tiles are DMA-loaded from HBM into a temporary SBUF tensor (v_block), then copied
    # into the augmented buffer (v_aug_all) via tensor_copy. This two-step pattern gives
    # better DMA performance than loading directly into the augmented layout, and the
    # tensor_copy overlaps well with TensorE compute so it doesn't stall MM2.
    num_v_aug_buffers = 3 if copy_engine == nisa.engine.vector else 2
    # Single allocation for all rotating buffers; ones column memset in one instruction.
    v_aug_all = nl.ndarray(shape=(_P, num_v_aug_buffers, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(v_aug_all[:, :, d : d + 1], 1.0)

    nisa.tensor_copy(dst=v_aug_all[:, 0, :d], src=v_block[0, 0].data, engine=copy_engine)

    for kv_tile_idx in range(num_kv_tiles):
        cur_v_aug = v_aug_all[:, kv_tile_idx % num_v_aug_buffers, :]

        # Copy next V tile into the next rotating buffer
        if kv_tile_idx < num_kv_tiles - 1:
            nisa.tensor_copy(
                dst=v_aug_all[:, (kv_tile_idx + 1) % num_v_aug_buffers, :d],
                src=v_block[kv_tile_idx + 1, 0].data,
                engine=copy_engine,
            )

        # MM1: Q @ K_tile.T → scores [128 Sk tokens, sq_block_size] in PSUM
        scores = nl.ndarray(shape=(_P, sq_block_size), dtype=scores_dtype, buffer=nl.psum)
        nisa.nc_matmul(dst=scores, stationary=k_block[kv_tile_idx].data, moving=q_block)

        # Fused eviction + exp: P = exp(scores * scale - max).
        # On the exponential path, Q was pre-scaled so scores are already scaled;
        # on the activation path, scale is fused here (one fewer instruction per Q block).
        probabilities = nl.ndarray(shape=(_P, sq_block_size), dtype=nl.bfloat16, buffer=nl.sbuf)
        if _has_vector_engine_exp():
            nisa.exponential(dst=probabilities, src=scores, max_value=softmax_max)
        else:
            nisa.activation(dst=probabilities, data=scores, op=nl.exp, scale=softmax_scale, bias=softmax_max)

        # MM2: P @ [V|1] → accumulate into O_aug [sq_tile_size, d+1] (tiled along Sq)
        should_accumulate = not (is_first_block and kv_tile_idx == 0)
        for output_tile_idx in range(num_output_tiles):
            sq_tile_size = o_aug_psums[output_tile_idx].data.shape[0]
            nisa.nc_matmul(
                dst=o_aug_psums[output_tile_idx].data,
                stationary=probabilities[:, nl.ds(output_tile_idx * _O_TILE_SIZE, sq_tile_size)],
                moving=cur_v_aug,
                accumulate=should_accumulate,
            )


def _normalize(o_aug_psum):
    """Normalize directly from PSUM: O = O_aug[:, :d] * (1 / max(O_aug[:, d], eps)).

    o_aug_psum: [sq_tile_size, d+1] in PSUM — columns 0..d-1 are weighted output, column d is sum_exp.
    Returns: [sq_tile_size, d] bf16 in SBUF.
    """
    sq_tile_size = o_aug_psum.shape[0]
    d = o_aug_psum.shape[1] - 1

    # Clamp sum_exp to avoid division by zero when all scores underflow
    sum_exp = nl.ndarray(shape=(sq_tile_size, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=sum_exp, data=o_aug_psum[:, d : d + 1], op0=nl.maximum, operand0=_EPS)

    inv_sum_exp = nl.ndarray(shape=(sq_tile_size, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=inv_sum_exp, data=sum_exp)

    o_out = nl.ndarray(shape=(sq_tile_size, d), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=o_out, data=o_aug_psum[:, :d], op0=nl.multiply, operand0=inv_sum_exp)
    return o_out


def _has_vector_engine_exp():
    """True when nisa.exponential is available (gen4-sub1+), using the Vector engine for exp."""
    return (nisa.get_nc_version() > nisa.nc_version.gen4) or (
        nisa.get_nc_version() == nisa.nc_version.gen4 and nisa.get_nc_sub_version() == 1
    )
