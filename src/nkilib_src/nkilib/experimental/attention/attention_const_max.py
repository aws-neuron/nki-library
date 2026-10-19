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

from typing import Optional, Tuple, Union

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.common_types import AttentionLayout
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
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
    q_layout: AttentionLayout = AttentionLayout.HEAD_MAJOR,
    k_layout: AttentionLayout = AttentionLayout.HEAD_MAJOR,
) -> nl.NkiTensor:
    """Constant-max attention: softmax(Q@K.T * softmax_scale - softmax_max) @ V.

    Computes attention assuming a pre-known global softmax max.

    Dimensions:
        N: Number of attention heads (batch of heads)
        d: Head dimension (must be pmax=128)
        Sq: Query sequence length
        Sk: Key/Value sequence length

    Args:
        q_hbm (nl.NkiTensor): ``[N, d, Sq]`` or ``[N, Sq, d]`` bf16 @ HBM, Query.
        k_hbm (nl.NkiTensor): ``[N, d, Sk]`` or ``[N, Sk, d]`` bf16 @ HBM, Key.
        v_hbm (nl.NkiTensor): [N, Sk, d] bf16 @ HBM, Value
        softmax_max (Union[float, nl.NkiTensor]): Upper bound on (scores * scale). Output is
            mathematically invariant to this value; it only affects numerical precision. A
            reasonable estimate suffices. May be either:
              - a Python scalar (compile-time constant, broadcast to all partitions), or
              - a [128, 1] f32 HBM tensor holding a single broadcast max replicated across
                the 128 partitions. Use the tensor form to supply a data-dependent max
                computed at runtime rather than baked in at trace time.
        softmax_scale (Optional[float]): Scaling factor applied to Q. Defaults to 1/sqrt(d).
        q_layout (AttentionLayout): Q layout. ``SEQ_MAJOR`` for ``[N, Sq, d]``,
            ``HEAD_MAJOR`` for ``[N, d, Sq]``. Defaults to ``HEAD_MAJOR``.
        k_layout (AttentionLayout): K layout. ``SEQ_MAJOR`` for ``[N, Sk, d]``,
            ``HEAD_MAJOR`` for ``[N, d, Sk]``. Defaults to ``HEAD_MAJOR``.

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
    Nq, Sq, d = _extract_dims(q_hbm, q_layout)
    _, Sk, d_k = _extract_dims(k_hbm, k_layout)
    Sv, d_v = v_hbm.shape[1], v_hbm.shape[2]

    softmax_scale = d ** (-0.5) if softmax_scale == None else softmax_scale
    _validate_inputs(Nq, Sq, d, Sk, d_k, Sv, d_v, q_hbm.dtype, k_hbm.dtype, v_hbm.dtype, softmax_max)

    # Prepare the constant max for the exp path once, up front. A scalar passes through as a
    # (pre-signed) float; a runtime HBM tensor is loaded into SBUF. See _prepare_softmax_max.
    prepared_softmax_max = _prepare_softmax_max(softmax_max)

    o_hbm = nl.ndarray(shape=(Nq, Sq, d), dtype=q_hbm.dtype, buffer=nl.shared_hbm)

    # Shard the independent (N, Sq) work across LNC cores by flattening N*Sq and splitting.
    # Q is [N, d, Sq] or [N, Sq, d] and O is [N, Sq, d] — both have N and Sq as concurrent dims.
    # K/V are fully replicated (all cores need all KV tokens).
    q_local, o_local, k_local, v_local = _shard_inputs(
        q_hbm,
        k_hbm,
        v_hbm,
        o_hbm,
        q_layout=q_layout,
    )

    _attention_const_max_compute(
        q_local,
        k_local,
        v_local,
        o_local,
        q_layout=q_layout,
        k_layout=k_layout,
        softmax_max=prepared_softmax_max,
        softmax_scale=softmax_scale,
    )

    return o_hbm


def _prepare_softmax_max(softmax_max: Union[float, nl.NkiTensor]) -> Union[float, nl.NkiTensor]:
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


def _validate_inputs(Nq, Sq, d, Sk, d_k, Sv, d_v, q_dtype, k_dtype, v_dtype, softmax_max):
    """Validate tensor shapes, dtypes, and softmax_max at kernel entry."""
    kernel_assert(d == _P, f"d must be pmax (128), got {d}")
    kernel_assert(d_k == d, f"K head dim must match Q, got {d_k=}, {d=}")
    kernel_assert(d_v == d, f"V head dim must match Q, got {d_v=}, {d=}")
    kernel_assert(Sv == Sk, f"V seq length must match K, got {Sv=}, {Sk=}")
    kernel_assert(Sk >= 1, f"Sk must be at least 1, got {Sk}")
    kernel_assert(Sq >= 1, f"Sq must be at least 1, got {Sq}")
    kernel_assert(q_dtype == nl.bfloat16, f"Q must be bf16, got {q_dtype}")
    kernel_assert(k_dtype == nl.bfloat16, f"K must be bf16, got {k_dtype}")
    kernel_assert(v_dtype == nl.bfloat16, f"V must be bf16, got {v_dtype}")
    if not isinstance(softmax_max, (int, float)):
        kernel_assert(softmax_max.shape[0] == _P, f"softmax_max partition dim must be {_P}, got {softmax_max.shape[0]}")
        kernel_assert(softmax_max.shape[1] == 1, f"softmax_max free dim must be 1, got {softmax_max.shape[1]}")


def _shard_inputs(
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    v_hbm: nl.NkiTensor,
    o_hbm: nl.NkiTensor,
    q_layout: AttentionLayout,
) -> Tuple[nl.NkiTensor, ...]:
    """Shard Q/O across heads when possible, otherwise across Q's sequence axis."""
    Nq, Sq, _ = _extract_dims(q_hbm, q_layout)
    num_cores = nl.num_programs()
    core_id = nl.program_id(0)

    if Nq % num_cores == 0:
        heads_per_core = Nq // num_cores
        head_start = core_id * heads_per_core
        head_end = head_start + heads_per_core
        q_local = q_hbm[head_start:head_end, :, :]
        o_local = o_hbm[head_start:head_end, :, :]
        k_local = k_hbm[head_start:head_end, :, :]
        v_local = v_hbm[head_start:head_end, :, :]
    else:
        sq_per_core = Sq // num_cores
        sq_start = core_id * sq_per_core
        sq_end = Sq if (core_id == num_cores - 1) else (sq_start + sq_per_core)
        q_local = (
            q_hbm[:, sq_start:sq_end, :] if q_layout == AttentionLayout.SEQ_MAJOR else q_hbm[:, :, sq_start:sq_end]
        )
        o_local = o_hbm[:, sq_start:sq_end, :]
        k_local = k_hbm
        v_local = v_hbm

    return q_local, o_local, k_local, v_local


def _attention_const_max_compute(
    q_hbm: nl.NkiTensor,
    k_hbm: nl.NkiTensor,
    v_hbm: nl.NkiTensor,
    o_hbm: nl.NkiTensor,
    q_layout: AttentionLayout,
    k_layout: AttentionLayout,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: float,
) -> None:
    """Core attention computation — agnostic to LNC sharding.

    Operates on the local slice of Q/O assigned to this core.
    K/V are the full (or head-sliced) tensors.

    Args:
        q_hbm: [N_local, d, Sq_local] or [N_local, Sq_local, d] bf16 @ HBM
        k_hbm: [N_local, d, Sk] or [N_local, Sk, d] bf16 @ HBM
        v_hbm: [N_local, Sk, d] bf16 @ HBM
        o_hbm: [N_local, Sq_local, d] bf16 @ HBM (output)
        softmax_max: pre-signed max for the exp path (float or [_P, 1] SBUF tensor)
        softmax_scale: scaling factor for Q
    """
    Nq, _, d = _extract_dims(q_hbm, q_layout)

    kernel_assert(nl.tile_size.psum_num_banks >= 6, f"need at least 6 PSUM banks, have {nl.tile_size.psum_num_banks}")
    _MM1_BUFFER_BANKS = 2
    _MAX_O_AUG_BANKS = min(nl.tile_size.psum_num_banks - _MM1_BUFFER_BANKS, 8)

    has_bf16_psum = nisa.get_nc_version() >= nisa.nc_version.gen4
    max_sq_block_size = (
        _MAX_O_AUG_BANKS * nl.tile_size.gemm_stationary_fmax if has_bf16_psum else nl.tile_size.gemm_moving_fmax
    )

    for head_idx in range(Nq):
        q_loader = _make_q_loader(
            q_hbm[head_idx],
            q_layout=q_layout,
            max_sq_block_size=max_sq_block_size,
            softmax_scale=softmax_scale if _has_vector_engine_exp() else None,
        )
        o_tiles = nt.tiles(o_hbm[head_idx], tile_size=(_O_TILE_SIZE, d))
        o_block_start = 0

        for q_block_idx in range(q_loader.num_blocks):
            q_block, sq_block_size = q_loader.load(q_block_idx)

            # O accumulator: one [Sq_tile, d+1] PSUM tile per Sq slice.
            # Columns 0..d-1 hold P @ V, column d accumulates sum_exp for normalization.
            num_output_tiles = div_ceil(sq_block_size, _O_TILE_SIZE)
            kernel_assert(
                num_output_tiles <= _MAX_O_AUG_BANKS,
                "Q block too large: O_aug tiles exceed available PSUM banks.",
            )
            o_aug_psums = nt.psum_pool(
                tile_size=(_O_TILE_SIZE, d + 1),
                element_shape=(sq_block_size, d + 1),
            )
            k_stream = _KVBlockStream(k_hbm[head_idx], layout=k_layout)
            v_stream = _KVBlockStream(v_hbm[head_idx], layout=AttentionLayout.SEQ_MAJOR, transpose=False)

            attention_const_max_core(
                k_stream=k_stream,
                v_stream=v_stream,
                q_block=q_block,
                o_aug_psums=o_aug_psums,
                softmax_max=softmax_max,
                softmax_scale=softmax_scale,
            )

            # Normalize each output tile by its accumulated sum_exp, then store to HBM.
            for output_tile_idx in range(num_output_tiles):
                o_out = _normalize(o_aug_psums[output_tile_idx].data)
                o_tiles[o_block_start + output_tile_idx, 0].store(o_out)
            o_block_start += num_output_tiles


def attention_const_max_core(
    k_stream: "_KVBlockStream",
    v_stream: "_KVBlockStream",
    q_block: nl.NkiTensor,
    o_aug_psums: nt.NDSlice,
    softmax_max: Union[float, nl.NkiTensor],
    softmax_scale: Optional[float] = None,
) -> None:
    """Compute scaled dot-product attention for one query block over all KV tiles.

    Args:
        k_stream (_KVBlockStream): Key tiles to attend over.
        v_stream (_KVBlockStream): Value tiles to attend over.
        q_block (nl.NkiTensor): [d, sq_block_size] query block.
        o_aug_psums (nt.NDSlice): Output accumulator, one [sq_tile_size, d + 1]
            tile per query slice, holding the unnormalized output and the softmax
            denominator. Written in place.
        softmax_max (Union[float, nl.NkiTensor]): Max subtracted for numerical
            stability. Scalar or [P_MAX, 1] tensor.
        softmax_scale (Optional[float]): Query-key scaling factor.

    Returns:
        None: Results are written in place into ``o_aug_psums``.

    Pipeline (P = prologue_depth):
        Prologue  (kv_tile_idx 0..P-1):        v_copy, MM1, exp
        Steady-state (kv_tile_idx P..total-1):  v_copy, MM1, exp, MM2(i-P)
        Epilogue  (drain last P):               MM2

    """
    d = _P
    kernel_assert(
        (k_stream.num_blocks, k_stream.num_tiles, k_stream.last_tile_extent)
        == (v_stream.num_blocks, v_stream.num_tiles, v_stream.last_tile_extent),
        "K/V stream geometry must match",
    )
    num_blocks = k_stream.num_blocks
    sq_block_size = q_block.shape[1]
    num_output_tiles = o_aug_psums.shape[0]
    total_tiles = k_stream.num_tiles
    last_tile_idx = total_tiles - 1
    last_tile_extent = k_stream.last_tile_extent

    scores_dtype = nl.bfloat16 if nisa.get_nc_version() >= nisa.nc_version.gen4 else nl.float32
    copy_engine = nisa.engine.scalar if _has_vector_engine_exp() else nisa.engine.vector

    # Number of MM1+exp tiles computed before first MM2 fires. Gives TensorE
    # a buffer of ready probabilities so it can always issue MM2 without waiting.
    prologue_depth = min(8, total_tiles)

    # Rotating V_aug buffers: need prologue_depth+1 (v_copy runs at MM1 time,
    # MM2 consumes prologue_depth iterations later; +1 to avoid WAR)
    num_v_aug_buffers = prologue_depth + 1
    v_aug_all = nl.ndarray(shape=(_P, num_v_aug_buffers, d + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(v_aug_all[:, :, d : d + 1], 1.0)

    num_prob_buffers = prologue_depth + 1
    probs_all = nl.ndarray(shape=(_P, num_prob_buffers, sq_block_size), dtype=nl.bfloat16, buffer=nl.sbuf)

    kv_tile_idx = 0  # flat index across all KV blocks
    softmax_max_is_scalar = isinstance(softmax_max, (int, float))

    for block_idx in range(num_blocks):
        k_block = k_stream.load(block_idx)[0]
        v_block = v_stream.load(block_idx)
        tiles_in_block = k_block.shape[0]

        with nl.no_reorder():
            for local_tile in range(tiles_in_block):
                v_buf = kv_tile_idx % num_v_aug_buffers
                prob_buf = kv_tile_idx % num_prob_buffers
                tile_extent = last_tile_extent if kv_tile_idx == last_tile_idx else _P

                # v_copy(i) — capture V tile now while block buffer is valid
                nisa.tensor_copy(
                    dst=v_aug_all[0:tile_extent, v_buf, :d],
                    src=v_block[local_tile, 0].data[0:tile_extent, :],
                    engine=copy_engine,
                )

                # MM1(i): K[D, r]^T @ Q[D, Sq] -> scores[r, Sq]. The free axis of
                # stationary K becomes the PSUM partition axis, so a final r < 128
                # naturally produces a partial-partition score tile.
                scores = nl.ndarray(shape=(_P, sq_block_size), dtype=scores_dtype, buffer=nl.psum)
                nisa.nc_matmul(
                    dst=scores[0:tile_extent, :],
                    stationary=k_block[local_tile].data[:, 0:tile_extent],
                    moving=q_block,
                )

                # exp(i) — fused evict + scale + subtract-max + exp
                tile_softmax_max = softmax_max if softmax_max_is_scalar else softmax_max[0:tile_extent, :]
                if _has_vector_engine_exp():
                    nisa.exponential(
                        dst=probs_all[0:tile_extent, prob_buf, :],
                        src=scores[0:tile_extent, :],
                        max_value=tile_softmax_max,
                    )
                else:
                    nisa.activation(
                        dst=probs_all[0:tile_extent, prob_buf, :],
                        data=scores[0:tile_extent, :],
                        op=nl.exp,
                        scale=softmax_scale,
                        bias=tile_softmax_max,
                    )

                # MM2(i-P) — only after prologue is filled
                prologue_filled = kv_tile_idx >= prologue_depth
                if prologue_filled:
                    consume_idx = kv_tile_idx - prologue_depth
                    consume_v_buf = consume_idx % num_v_aug_buffers
                    consume_prob_buf = consume_idx % num_prob_buffers
                    should_accumulate = consume_idx != 0
                    cur_v_aug = v_aug_all[:, consume_v_buf, :]
                    consume_extent = last_tile_extent if consume_idx == last_tile_idx else _P
                    for ot_idx in range(num_output_tiles):
                        sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
                        nisa.nc_matmul(
                            dst=o_aug_psums[ot_idx].data,
                            stationary=probs_all[
                                0:consume_extent,
                                consume_prob_buf,
                                nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size),
                            ],
                            moving=cur_v_aug[0:consume_extent, :],
                            accumulate=should_accumulate,
                        )

                kv_tile_idx += 1

    # === Epilogue: drain remaining prologue_depth MM2s ===
    kernel_assert(kv_tile_idx == total_tiles, f"stream yielded {kv_tile_idx} KV tiles, expected {total_tiles}")
    with nl.no_reorder():
        for i in range(prologue_depth):
            drain_idx = total_tiles - prologue_depth + i
            drain_v_buf = drain_idx % num_v_aug_buffers
            drain_prob_buf = drain_idx % num_prob_buffers
            should_accumulate = drain_idx != 0
            drain_v_aug = v_aug_all[:, drain_v_buf, :]
            drain_extent = last_tile_extent if drain_idx == last_tile_idx else _P
            for ot_idx in range(num_output_tiles):
                sq_tile_size = o_aug_psums[ot_idx].data.shape[0]
                nisa.nc_matmul(
                    dst=o_aug_psums[ot_idx].data,
                    stationary=probs_all[
                        0:drain_extent,
                        drain_prob_buf,
                        nl.ds(ot_idx * _O_TILE_SIZE, sq_tile_size),
                    ],
                    moving=drain_v_aug[0:drain_extent, :],
                    accumulate=should_accumulate,
                )


def _normalize(o_aug_psum: nl.NkiTensor) -> nl.NkiTensor:
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


def _has_vector_engine_exp() -> bool:
    """True when nisa.exponential is available (gen4-sub1+), using the Vector engine for exp."""
    return (nisa.get_nc_version() > nisa.nc_version.gen4) or (
        nisa.get_nc_version() == nisa.nc_version.gen4 and nisa.get_nc_sub_version() == 1
    )


def _extract_dims(tensor: nl.NkiTensor, layout: AttentionLayout) -> tuple:
    """Extract (N, S, d) from [..., d, S] or [..., S, d] tensor shape."""
    if layout == AttentionLayout.SEQ_MAJOR:
        N, S, d = tensor.shape
    else:
        N, d, S = tensor.shape
    return N, S, d


def _make_q_loader(q_head_hbm, q_layout, max_sq_block_size, softmax_scale):
    """Create the appropriate Q block loader based on input layout."""
    loader_cls = _QBlockLoaderSeqMajor if q_layout == AttentionLayout.SEQ_MAJOR else _QBlockLoaderHeadMajor
    return loader_cls(q_head_hbm, max_sq_block_size=max_sq_block_size, softmax_scale=softmax_scale)


class _KVBlockStream(nl.NKIObject):
    """Stream KV tile blocks with optional transpose, handling a sub-128 last tile.

    Tiles are 128×128: K tiles are [d, Sk_tile] where d=pmax and Sk_tile=pmax (Sk
    lands on the partition axis for scores), V tiles are [Sk_tile, d] for the same
    reason. Both dimensions are pmax, so tiling is always at _P.

    neurotile's BlockStream does not support a partial last tile within a block.
    This class splits it into a separate tail block so all tiles within each
    streamed block have uniform extent.

    TODO: Replace with neurotile's BlockStream once it supports partial last tiles.

    Attributes:
        num_blocks: Number of blocks to iterate over.
        num_tiles: Total number of tiles across all blocks.
        last_tile_extent: Number of active elements in the last tile (≤ _P).
    """

    def __init__(
        self,
        source: nl.NkiTensor,
        layout: AttentionLayout = AttentionLayout.HEAD_MAJOR,
        transpose: Optional[bool] = None,
    ):
        # layout controls tiling: dim 0 for SEQ_MAJOR [Sk, d], dim 1 for HEAD_MAJOR [d, Sk].
        # transpose defaults to True for SEQ_MAJOR (K needs transpose), False for HEAD_MAJOR.
        seq_major = layout == AttentionLayout.SEQ_MAJOR
        tiling_dim = 0 if seq_major else 1
        block_size = (_KV_TILES_PER_BLOCK, 1) if seq_major else (1, _KV_TILES_PER_BLOCK)
        transpose = seq_major if transpose == None else transpose
        seq_len = source.shape[tiling_dim]
        tail_extent = seq_len % _P
        body_len = seq_len - tail_extent  # 128-aligned portion
        self.num_tiles = div_ceil(seq_len, _P)
        self.last_tile_extent = tail_extent if tail_extent != 0 else _P
        self._transpose = transpose

        def blocks_over(lo, hi):
            """Create a 1D block grid over source[lo:hi] along tiling_dim."""
            idx = [slice(None)] * len(source.shape)
            idx[tiling_dim] = slice(lo, hi)
            block_grid = nt.blocks(source[tuple(idx)], tile_size=(_P, _P), block_size=block_size)
            stream_idx = [0] * len(block_size)
            stream_idx[tiling_dim] = slice(None)
            return block_grid[tuple(stream_idx)]

        tiles_per_block = block_size[tiling_dim]  # number of tiles in one block
        block_len = tiles_per_block * _P  # elements per block along tiling_dim
        self._body_block_count = div_ceil(body_len, block_len)

        no_tail = tail_extent == 0
        tail_only = body_len == 0

        if no_tail:
            # All tiles are full 128-element tiles.
            self._body_stream = blocks_over(0, seq_len).stream(buffer_count=2, transpose=transpose)
            self._tail_block = None
        elif tail_only:
            # Entire sequence is sub-128 (e.g., Sk < 128).
            self._body_stream = None
            self._body_block_count = 0
            self._tail_block = blocks_over(0, seq_len)[0]
        else:
            # Body + tail: full tiles streamed, partial last tile loaded separately.
            self._body_stream = blocks_over(0, body_len).stream(buffer_count=2, transpose=transpose)
            self._tail_block = blocks_over(body_len, seq_len)[0]

        self.num_blocks = self._body_block_count + (self._tail_block != None)

    def load(self, index: int, dge_mode=None, engine=None):
        """Load block at ``index``, returning the tile grid for this block."""
        kernel_assert(index < self.num_blocks, f"block index {index} out of range (num_blocks={self.num_blocks})")
        if index < self._body_block_count:
            return self._body_stream.load(index, transpose=self._transpose, dge_mode=dge_mode, engine=engine)
        return self._tail_block.load(transpose=self._transpose, dge_mode=dge_mode, engine=engine)


class _QBlockLoaderHeadMajor(nl.NKIObject):
    """Load Q blocks from [d, Sq] layout as scaled [d, Sq_block] SBUF tensors.

    Single bulk DMA per block via neurotile tile load.
    When ``softmax_scale`` is not None, it is fused into the load.

    Attributes:
        num_blocks: Number of Q blocks along the Sq axis.
    """

    def __init__(
        self,
        q_head_hbm: nl.NkiTensor,
        max_sq_block_size: int,  # max Sq elements per block
        softmax_scale: Optional[float] = None,
    ):
        self._softmax_scale = softmax_scale
        self._q_tile_grid = nt.tiles(q_head_hbm, tile_size=(_P, max_sq_block_size))
        self.num_blocks = self._q_tile_grid.shape[1]

    def load(self, block_idx: int) -> tuple:
        """Return ``(q_block, sq_block_size)`` with q_block as [d, sq_block_size] in SBUF."""
        loaded = self._q_tile_grid[0, block_idx].load()
        q_block = loaded.data
        sq_block_size = loaded.element_shape[1]  # actual Sq elements in this block
        if self._softmax_scale != None:
            nisa.tensor_scalar(dst=q_block, data=q_block, op0=nl.multiply, operand0=self._softmax_scale)
        return q_block, sq_block_size


class _QBlockLoaderSeqMajor(nl.NKIObject):
    """Load Q blocks from [Sq, d] layout as scaled [d, Sq_block] SBUF tensors.

    For the 128-aligned body, a single DMA transpose loads [Sq_body, d] HBM directly
    into [d, Sq_body] SBUF. A sub-128 tail (if any) uses a separate DMA + nc_transpose.
    When ``softmax_scale`` is not None, it is fused into the load.

    Attributes:
        num_blocks: Number of Q blocks along the Sq axis.
    """

    def __init__(
        self,
        q_head_hbm: nl.NkiTensor,
        max_sq_block_size: int,  # max Sq elements per block
        softmax_scale: Optional[float] = None,
    ):
        self._softmax_scale = softmax_scale
        self._dtype = q_head_hbm.dtype
        self._q_head_hbm = q_head_hbm
        self._sq_length = q_head_hbm.shape[0]  # total Sq elements
        self._tiles_per_block = max_sq_block_size // _P  # number of 128-element tiles per block
        num_tiles = div_ceil(self._sq_length, _P)
        self.num_blocks = div_ceil(num_tiles, self._tiles_per_block)

    def load(self, block_idx: int) -> tuple:
        """Return ``(q_block, sq_block_size)`` with q_block as [d, sq_block_size] in SBUF."""
        sq_start = block_idx * self._tiles_per_block * _P  # element offset into Sq
        sq_block_size = min(self._tiles_per_block * _P, self._sq_length - sq_start)  # elements in this block
        tail_extent = sq_block_size % _P  # sub-128 remainder (0 if aligned)
        body_extent = sq_block_size - tail_extent  # 128-aligned portion
        scale = self._softmax_scale if self._softmax_scale != None else 1.0

        q_block = nl.ndarray(shape=(_P, sq_block_size), dtype=self._dtype, buffer=nl.sbuf)

        # DMA transpose for the 128-aligned body: [body_extent, d] HBM -> [d, body_extent] SBUF
        if body_extent > 0:
            nisa.dma_transpose(
                dst=q_block[:, :body_extent],
                src=self._q_head_hbm[sq_start : sq_start + body_extent, :],
            )
            nisa.tensor_scalar(
                dst=q_block[:, :body_extent], data=q_block[:, :body_extent], op0=nl.multiply, operand0=scale
            )

        # Partial tail: DMA load + nc_transpose for the sub-128 remainder
        if tail_extent > 0:
            tail_sbuf = nl.ndarray(shape=(tail_extent, _P), dtype=self._dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=tail_sbuf, src=self._q_head_hbm[sq_start + body_extent : sq_start + sq_block_size, :])
            tail_psum = nl.ndarray(shape=(_P, tail_extent), dtype=self._dtype, buffer=nl.psum)
            nisa.nc_transpose(dst=tail_psum, data=tail_sbuf)
            nisa.tensor_scalar(
                dst=q_block[:, body_extent:sq_block_size],
                data=tail_psum,
                op0=nl.multiply,
                operand0=scale,
            )

        return q_block, sq_block_size
