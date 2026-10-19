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

"""Kimi-K3 dense causal MLA prefill with natural-order latent output.

The alignment-sensitive DeepSeek path transposes all heads from HBM at once,
which requires ``H * sizeof(bf16)`` to be 32-byte aligned. Kimi-K3 has three
heads per TP32 rank. This kernel instead loads one head's query rows in their
natural contiguous layout and performs the transpose on-core. No head padding
is needed and matmul work is performed for exactly the logical head count.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast

_P_MAX = 128
_Q_TILE = 128
_K_TILE = 128
_SCORE_TILE = 128
_L = 512
_R = 64
_MAX_SEQUENCE = 2048
_FLOAT32_MIN = -3.4028235e38


@nki.jit
def kimi_k3_mla_prefill(
    q_lift_hbm: nl.NkiTensor,
    q_pe_hbm: nl.NkiTensor,
    c_kv_hbm: nl.NkiTensor,
    k_pe_hbm: nl.NkiTensor,
    softmax_scale: float,
    topk_indices_hbm: nl.NkiTensor = None,
    dense: bool = True,
    q_pos_offset_hbm: nl.NkiTensor = None,
) -> nl.NkiTensor:
    """Compute Kimi-K3 dense causal MLA prefill on Trn3 LNC1.

    Dimensions:
        B: Batch size, fixed to 1.
        S: Active query length, a multiple of 128.
        S_kv: Active latent-cache length, a multiple of 128.
        H: Heads per rank, including Kimi TP32's H=3.
        L: Latent width, fixed to 512.
        R: Extra NoPE width, fixed to 64.

    Args:
        q_lift_hbm: BF16 absorbed latent query ``[B, S, H, L]``.
        q_pe_hbm: BF16 unrotated NoPE query ``[B, S, H, R]``.
        c_kv_hbm: BF16 latent keys/values ``[B, S_kv, L]``.
        k_pe_hbm: BF16 shared single-head NoPE keys ``[B, S_kv, R]``.
        softmax_scale: Score scale, ``192**-0.5`` for Kimi-K3.
        topk_indices_hbm: Ignored in dense mode and may be ``None``.
        dense: Must be ``True``. Kimi prefill has no sparse indexer.
        q_pos_offset_hbm: Optional INT32 ``[1, 1]`` absolute position of
            local query zero. The causal mask keeps key ``j`` when
            ``j <= q_pos_offset + local_query``.

    Returns:
        BF16 latent attention output ``[B, S, H * L]`` in natural order:
        physical column ``h * L + l`` stores logical head ``h``, latent ``l``.

    Notes:
        - Inputs are pure and unaliased.
        - ``k_pe_hbm`` is shared across heads; it is loaded once and reused.
        - The caller must pass exact active ``S`` and ``S_kv`` lengths. Bucket
          padding is not inferred or approximately masked by this kernel.
        - Query and key tensors are already in Kimi's NoPE representation. No
          rotary transform or cos/sin cache is consumed.
    """
    _validate_inputs(
        q_lift_hbm=q_lift_hbm,
        q_pe_hbm=q_pe_hbm,
        c_kv_hbm=c_kv_hbm,
        k_pe_hbm=k_pe_hbm,
        softmax_scale=softmax_scale,
        dense=dense,
        q_pos_offset_hbm=q_pos_offset_hbm,
    )

    _, seqlen, heads, latent_width = q_lift_hbm.shape
    kv_seqlen = c_kv_hbm.shape[1]
    latent_tiles = latent_width // _P_MAX
    key_chunks = kv_seqlen // _K_TILE
    score_tiles = kv_seqlen // _SCORE_TILE
    query_tiles = seqlen // _Q_TILE

    output = nl.ndarray(
        (1, seqlen, heads * latent_width),
        dtype=nl.bfloat16,
        buffer=nl.shared_hbm,
    )
    output_2d = output.reshape((seqlen, heads * latent_width))

    # Query-independent KV layouts. MM1 consumes feature-major tiles while
    # MM2 consumes token-major 128-row chunks.
    c_feature_major = nl.ndarray(
        (_P_MAX, latent_tiles, kv_seqlen),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    for latent_tile in range(latent_tiles):
        latent_offset = latent_tile * _P_MAX
        nisa.dma_transpose(
            dst=c_feature_major[:, latent_tile, :],
            src=c_kv_hbm.ap(
                pattern=[[latent_width, kv_seqlen], [1, _P_MAX]],
                offset=latent_offset,
            ),
        )

    k_pe_feature_major = nl.ndarray(
        (_R, kv_seqlen),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    nisa.dma_transpose(
        dst=k_pe_feature_major,
        src=k_pe_hbm.ap(pattern=[[_R, kv_seqlen], [1, _R]], offset=0),
    )

    c_token_major = nl.ndarray(
        (_K_TILE, key_chunks, latent_width),
        dtype=nl.bfloat16,
        buffer=nl.sbuf,
    )
    for key_chunk in range(key_chunks):
        key_offset = key_chunk * _K_TILE
        c_token_psum = nl.ndarray(
            (_K_TILE, latent_width),
            dtype=nl.bfloat16,
            buffer=nl.psum,
        )
        for latent_tile in range(latent_tiles):
            latent_offset = latent_tile * _P_MAX
            nisa.nc_transpose(
                dst=c_token_psum[:, latent_offset : latent_offset + _P_MAX],
                data=c_feature_major[:, latent_tile, key_offset : key_offset + _K_TILE],
            )
        nisa.tensor_copy(
            dst=c_token_major[:, key_chunk, :],
            src=c_token_psum,
        )

    q_pos_offset = _load_q_pos_offset(q_pos_offset_hbm)
    zero_bound = nl.ndarray(
        (_Q_TILE, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    nisa.memset(dst=zero_bound, value=0.0)

    q_lift_flat = q_lift_hbm.reshape((seqlen * heads * latent_width,))
    q_pe_flat = q_pe_hbm.reshape((seqlen * heads * _R,))

    for query_tile in range(query_tiles):
        query_offset = query_tile * _Q_TILE

        query_index_i32 = nl.ndarray(
            (_Q_TILE, 1),
            dtype=nl.int32,
            buffer=nl.sbuf,
        )
        nisa.iota(
            dst=query_index_i32,
            pattern=[[0, 1]],
            offset=query_offset,
            channel_multiplier=1,
        )
        query_global = nl.ndarray(
            (_Q_TILE, 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
        nisa.tensor_copy(dst=query_global, src=query_index_i32)
        nisa.tensor_scalar(
            dst=query_global,
            data=query_global,
            op0=nl.add,
            operand0=q_pos_offset,
        )

        for head in range(heads):
            q_lift_natural = nl.ndarray(
                (_Q_TILE, latent_width),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=q_lift_natural,
                src=q_lift_flat.ap(
                    pattern=[
                        [heads * latent_width, _Q_TILE],
                        [1, latent_width],
                    ],
                    offset=(query_offset * heads + head) * latent_width,
                ),
            )

            q_lift_feature_major = nl.ndarray(
                (_P_MAX, latent_tiles, _Q_TILE),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            q_transpose_psum = nl.ndarray(
                (_P_MAX, _Q_TILE),
                dtype=nl.bfloat16,
                buffer=nl.psum,
            )
            for latent_tile in range(latent_tiles):
                latent_offset = latent_tile * _P_MAX
                nisa.nc_transpose(
                    dst=q_transpose_psum,
                    data=q_lift_natural[:, latent_offset : latent_offset + _P_MAX],
                )
                nisa.tensor_copy(
                    dst=q_lift_feature_major[:, latent_tile, :],
                    src=q_transpose_psum,
                )

            q_pe_natural = nl.ndarray(
                (_Q_TILE, _R),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            nisa.dma_copy(
                dst=q_pe_natural,
                src=q_pe_flat.ap(
                    pattern=[
                        [heads * _R, _Q_TILE],
                        [1, _R],
                    ],
                    offset=(query_offset * heads + head) * _R,
                ),
            )
            q_pe_feature_major = nl.ndarray(
                (_R, _Q_TILE),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            nisa.nc_transpose(
                dst=q_transpose_psum[0:_R, :],
                data=q_pe_natural,
            )
            nisa.tensor_copy(
                dst=q_pe_feature_major,
                src=q_transpose_psum[0:_R, :],
            )

            # Match the model's BF16 score boundaries: each latent and NoPE
            # matmul rounds to BF16, their sum rounds to BF16, then scaling
            # rounds to BF16 before the FP32 softmax.
            latent_scores_bf16 = nl.ndarray(
                (_Q_TILE, kv_seqlen),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            scores_scaled_bf16 = nl.ndarray(
                (_Q_TILE, kv_seqlen),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            q_pe_scores_bf16 = nl.ndarray(
                (_Q_TILE, _SCORE_TILE),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )

            for score_tile in range(score_tiles):
                key_offset = score_tile * _SCORE_TILE
                latent_scores_psum = nl.ndarray(
                    (_Q_TILE, _SCORE_TILE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                for latent_tile in range(latent_tiles):
                    nisa.nc_matmul(
                        dst=latent_scores_psum,
                        stationary=q_lift_feature_major[:, latent_tile, :],
                        moving=c_feature_major[
                            :,
                            latent_tile,
                            key_offset : key_offset + _SCORE_TILE,
                        ],
                        accumulate=latent_tile != 0,
                    )
                nisa.tensor_copy(
                    dst=latent_scores_bf16[:, key_offset : key_offset + _SCORE_TILE],
                    src=latent_scores_psum,
                )

                q_pe_scores_psum = nl.ndarray(
                    (_Q_TILE, _SCORE_TILE),
                    dtype=nl.float32,
                    buffer=nl.psum,
                )
                nisa.nc_matmul(
                    dst=q_pe_scores_psum,
                    stationary=q_pe_feature_major,
                    moving=k_pe_feature_major[:, key_offset : key_offset + _SCORE_TILE],
                )
                nisa.tensor_copy(
                    dst=q_pe_scores_bf16,
                    src=q_pe_scores_psum,
                )
                nisa.tensor_tensor(
                    dst=scores_scaled_bf16[:, key_offset : key_offset + _SCORE_TILE],
                    data1=latent_scores_bf16[:, key_offset : key_offset + _SCORE_TILE],
                    data2=q_pe_scores_bf16,
                    op=nl.add,
                )
                nisa.tensor_scalar(
                    dst=scores_scaled_bf16[:, key_offset : key_offset + _SCORE_TILE],
                    data=scores_scaled_bf16[:, key_offset : key_offset + _SCORE_TILE],
                    op0=nl.multiply,
                    operand0=softmax_scale,
                )

            scores_f32 = nl.ndarray(
                (_Q_TILE, kv_seqlen),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            scores_masked = nl.ndarray(
                (_Q_TILE, kv_seqlen),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            row_max = nl.ndarray(
                (_Q_TILE, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.tensor_copy(dst=scores_f32, src=scores_scaled_bf16)
            nisa.range_select(
                dst=scores_masked,
                on_true_tile=scores_f32,
                on_false_value=_FLOAT32_MIN,
                comp_op0=nl.greater_equal,
                comp_op1=nl.less_equal,
                bound0=zero_bound,
                bound1=query_global,
                reduce_op=nl.maximum,
                reduce_res=row_max,
            )

            negative_row_max = nl.ndarray(
                (_Q_TILE, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.tensor_scalar(
                dst=negative_row_max,
                data=row_max,
                op0=nl.multiply,
                operand0=-1.0,
            )
            probabilities = nl.ndarray(
                (_Q_TILE, kv_seqlen),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            row_sum = nl.ndarray(
                (_Q_TILE, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            for score_tile in range(score_tiles):
                key_offset = score_tile * _SCORE_TILE
                nisa.activation(
                    dst=probabilities[:, key_offset : key_offset + _SCORE_TILE],
                    op=nl.exp,
                    data=scores_masked[:, key_offset : key_offset + _SCORE_TILE],
                    bias=negative_row_max,
                    reduce_op=nl.add,
                    reduce_res=row_sum,
                    reduce_cmd=(nisa.reduce_cmd.reset_reduce if score_tile == 0 else nisa.reduce_cmd.reduce),
                )

            reciprocal = nl.ndarray(
                (_Q_TILE, 1),
                dtype=nl.float32,
                buffer=nl.sbuf,
            )
            nisa.reciprocal(dst=reciprocal, data=row_sum)

            latent_output_psum = nl.ndarray(
                (_Q_TILE, latent_width),
                dtype=nl.float32,
                buffer=nl.psum,
            )
            probability_transpose_psum = nl.ndarray(
                (_K_TILE, _Q_TILE),
                dtype=nl.bfloat16,
                buffer=nl.psum,
            )
            probability_transpose = nl.ndarray(
                (_K_TILE, _Q_TILE),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            for key_chunk in range(key_chunks):
                key_offset = key_chunk * _K_TILE
                nisa.nc_transpose(
                    dst=probability_transpose_psum,
                    data=probabilities[:, key_offset : key_offset + _K_TILE],
                )
                nisa.tensor_copy(
                    dst=probability_transpose,
                    src=probability_transpose_psum,
                )
                nisa.nc_matmul(
                    dst=latent_output_psum,
                    stationary=probability_transpose,
                    moving=c_token_major[:, key_chunk, :],
                    accumulate=key_chunk != 0,
                )

            latent_output = nl.ndarray(
                (_Q_TILE, latent_width),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
            )
            nisa.tensor_scalar(
                dst=latent_output,
                data=latent_output_psum,
                op0=nl.multiply,
                operand0=reciprocal,
            )
            nisa.dma_copy(
                dst=output_2d[
                    query_offset : query_offset + _Q_TILE,
                    head * latent_width : (head + 1) * latent_width,
                ],
                src=latent_output,
            )

    return output


def _load_q_pos_offset(q_pos_offset_hbm: nl.NkiTensor | None) -> nl.NkiTensor:
    """Return the runtime absolute-position offset on all query partitions."""
    q_pos_offset = nl.ndarray(
        (_Q_TILE, 1),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    if q_pos_offset_hbm is None:
        nisa.memset(dst=q_pos_offset, value=0.0)
    else:
        q_pos_offset_in = nl.ndarray(
            (1, 1),
            dtype=nl.int32,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=q_pos_offset_in, src=q_pos_offset_hbm)
        nisa.tensor_copy(
            dst=q_pos_offset[0:1, :],
            src=q_pos_offset_in,
        )
        stream_shuffle_broadcast(
            src=q_pos_offset[0:1, :],
            dst=q_pos_offset,
        )
    return q_pos_offset


def _validate_inputs(
    q_lift_hbm: nl.NkiTensor,
    q_pe_hbm: nl.NkiTensor,
    c_kv_hbm: nl.NkiTensor,
    k_pe_hbm: nl.NkiTensor,
    softmax_scale: float,
    dense: bool,
    q_pos_offset_hbm: nl.NkiTensor | None,
) -> None:
    """Validate the public Kimi-K3 MLA prefill contract."""
    kernel_assert(
        nl.program_ndim() == 0 or (nl.program_ndim() == 1 and nl.num_programs(0) == 1),
        "kimi_k3_mla_prefill supports LNC1 only",
    )
    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen4,
        "kimi_k3_mla_prefill requires Trn3 or newer",
    )
    kernel_assert(dense is True, "Kimi-K3 MLA prefill requires dense=True")
    kernel_assert(softmax_scale > 0.0, "softmax_scale must be positive")
    kernel_assert(
        len(q_lift_hbm.shape) == 4,
        f"Expected q_lift_hbm [B, S, H, L], got {q_lift_hbm.shape}",
    )
    batch, seqlen, heads, latent_width = q_lift_hbm.shape
    kernel_assert(batch == 1, f"Expected B=1, got B={batch}")
    kernel_assert(
        0 < heads <= _P_MAX,
        f"Expected 1 <= H <= {_P_MAX}, got H={heads}",
    )
    kernel_assert(
        latent_width == _L,
        f"Expected L={_L}, got L={latent_width}",
    )
    kernel_assert(
        _Q_TILE <= seqlen <= _MAX_SEQUENCE and seqlen % _Q_TILE == 0,
        f"Expected {_Q_TILE} <= S <= {_MAX_SEQUENCE} and S % {_Q_TILE} == 0, got S={seqlen}",
    )
    kernel_assert(
        q_pe_hbm.shape == (batch, seqlen, heads, _R),
        f"Expected q_pe_hbm {(batch, seqlen, heads, _R)}, got {q_pe_hbm.shape}",
    )
    kernel_assert(
        len(c_kv_hbm.shape) == 3 and c_kv_hbm.shape[0] == batch and c_kv_hbm.shape[2] == latent_width,
        f"Expected c_kv_hbm [B, S_kv, L] with B={batch}, L={latent_width}, got {c_kv_hbm.shape}",
    )
    kv_seqlen = c_kv_hbm.shape[1]
    kernel_assert(
        _K_TILE <= kv_seqlen <= _MAX_SEQUENCE and kv_seqlen % _SCORE_TILE == 0,
        f"Expected {_K_TILE} <= S_kv <= {_MAX_SEQUENCE} and S_kv % {_SCORE_TILE} == 0, got S_kv={kv_seqlen}",
    )
    kernel_assert(
        k_pe_hbm.shape == (batch, kv_seqlen, _R),
        f"Expected k_pe_hbm {(batch, kv_seqlen, _R)}, got {k_pe_hbm.shape}",
    )
    kernel_assert(
        q_lift_hbm.dtype == nl.bfloat16,
        f"q_lift_hbm must be BF16, got {q_lift_hbm.dtype}",
    )
    kernel_assert(
        q_pe_hbm.dtype == nl.bfloat16,
        f"q_pe_hbm must be BF16, got {q_pe_hbm.dtype}",
    )
    kernel_assert(
        c_kv_hbm.dtype == nl.bfloat16,
        f"c_kv_hbm must be BF16, got {c_kv_hbm.dtype}",
    )
    kernel_assert(
        k_pe_hbm.dtype == nl.bfloat16,
        f"k_pe_hbm must be BF16, got {k_pe_hbm.dtype}",
    )
    if q_pos_offset_hbm is not None:
        kernel_assert(
            q_pos_offset_hbm.shape == (1, 1),
            f"Expected q_pos_offset_hbm [1, 1], got {q_pos_offset_hbm.shape}",
        )
        kernel_assert(
            q_pos_offset_hbm.dtype == nl.int32,
            f"q_pos_offset_hbm must be INT32, got {q_pos_offset_hbm.dtype}",
        )
