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
"""AttnRes --- Kimi-K3's cross-layer residual mixture, as a NKI kernel.

Replaces ``vllm_neuron.model.kimi_k3.attn_res.apply_attn_res``. Every decoder layer,
twice per layer, the plain residual add is replaced by a softmax mixture over the
current residual stream plus ``C`` snapshots of it banked at earlier layers. That torch
function was ~40% of the compiled graph's per-rank instructions (the ``cat`` of the
whole ``[T, C, H]`` bank and the per-token bmm); this kernel does the same math with a
fixed, small instruction count per token tile.

The operator, per token ``t``, over candidates ``v_c = bank[t, c]`` (``c < C``) and
``v_C = stream[t]``::

    ss_c  = sum_h v_c[h] ** 2
    dot_c = sum_h v_c[h] * score_w[h]
    s_c   = dot_c * rsqrt(ss_c / H + eps)      # == rmsnorm(v_c) . score_w
    p     = softmax_c(s)                        # over the C+1 candidates
    out   = sum_c p_c * v_c                     # mixes the RAW candidates

Two facts drive the layout and the low instruction count:

1. ``rsqrt(ss_c / H + eps)`` is one scalar per (token, candidate), so it factors out of
   the dot product: only two free-axis reductions per candidate are needed
   (``sum v^2`` and ``sum v * score_w``), never a normalised copy of ``v``.
2. The mixture is over the RAW candidates, so the ``v_c`` loaded for scoring are the
   same tensors reused (re-loaded) for the weighted sum --- there is no second,
   normalised tensor in play.

Layout: tokens on the partition axis (tiles of up to 128), ``H`` on the free axis. The
whole ``H`` row of one candidate is loaded contiguously and reduced in one instruction;
``C`` and ``H`` never sit on partitions. Two passes over the candidates per token tile:
pass 1 builds the ``[tok, C+1]`` score tile and its softmax, pass 2 re-loads each
candidate and accumulates ``p_c * v_c`` into the output row. Keeping all ``C+1``
candidates resident would be a single pass, but at ``H = 7168`` that is
``(C+1) * 7168 * 2B`` per partition; the re-load keeps the live set to three ``[128, H]``
tiles regardless of ``C`` and never risks the SBUF ceiling.

LNC: a 1-D SPMD grid shards the token axis across the programs (``T`` must divide the
grid size), exactly like ``rope_hf`` / the MoE kernels. Works with 1 program too. Every
candidate is independent per token, so the shards never communicate.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info

# Partition (token) tile size and the free-axis chunk contract.
_PMAX = 128
_HCHUNK = 512
# Practical bank width is 1..8; the kernel supports up to this many snapshots.
_MAX_C = 15


@nki.jit
def attn_res_cte(
    bank: nl.NkiTensor,
    stream: nl.NkiTensor,
    score_w: nl.NkiTensor,
    eps: float,
):
    """Softmax mixture of the residual stream with its banked snapshots.

    Args:
        bank: ``[T, C, H]`` bf16 --- ``C`` snapshot candidates per token
            (``1 <= C <= 15``). Row-major, ``H`` contiguous.
        stream: ``[T, H]`` bf16 --- the current residual stream (the
            ``(C+1)``-th candidate).
        score_w: ``[H]`` fp32 --- the fused score vector
            (``norm.weight * proj.weight``), precomputed by the caller.
        eps: RMSNorm epsilon.

    Returns:
        ``out: [T, H]`` in ``stream``'s dtype.

    Notes:
        - All reductions and the softmax run in fp32; only the loaded candidates and
          the stored output are bf16.
        - ``T`` must be a multiple of 128 and of the SPMD grid size; ``H`` a multiple
          of 512. Both are checked with ``kernel_assert``.
    """
    T, C, H = bank.shape
    Cp1 = C + 1
    dtype = stream.dtype

    kernel_assert(T % _PMAX == 0, f"attn_res_cte requires T % {_PMAX} == 0, got T={T}")
    kernel_assert(H % _HCHUNK == 0, f"attn_res_cte requires H % {_HCHUNK} == 0, got H={H}")
    kernel_assert(C <= _MAX_C, f"attn_res_cte supports C <= {_MAX_C} candidates, got C={C}")
    kernel_assert(
        bank.dtype == dtype,
        f"attn_res_cte requires bank and stream to share a dtype, got {bank.dtype} vs {dtype}",
    )
    kernel_assert(
        score_w.dtype == nl.float32,
        f"attn_res_cte requires score_w in float32, got {score_w.dtype}",
    )

    _, num_shards, shard_id = get_verified_program_sharding_info("attn_res_cte", (0, 1))
    kernel_assert(
        T % num_shards == 0,
        f"attn_res_cte requires T % num_shards == 0, got T={T}, num_shards={num_shards}",
    )
    seq_per_shard = T // num_shards
    tile = min(_PMAX, seq_per_shard)
    n_tiles = div_ceil(seq_per_shard, tile)
    shard_off = shard_id * seq_per_shard

    out = nl.ndarray((T, H), dtype=dtype, buffer=nl.shared_hbm)

    # score_w broadcast to every partition once (it does not depend on the token).
    # Partition stride 0 in the access pattern replicates the same H-vector to all
    # 128 partitions.
    sw = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=sw, src=score_w.ap(pattern=[[0, _PMAX], [1, H]], offset=0))

    inv_h = 1.0 / float(H)

    for i in range(n_tiles):
        tok = min(tile, seq_per_shard - i * tile)
        t_off = shard_off + i * tile

        # ---------- pass 1: scores over the C+1 candidates ----------
        scores = nl.ndarray((_PMAX, Cp1), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(Cp1):
            v_c = nl.ndarray((_PMAX, H), dtype=dtype, buffer=nl.sbuf)
            if c < C:
                nisa.dma_copy(
                    dst=v_c[0:tok, 0:H],
                    src=bank.ap(
                        pattern=[[C * H, tok], [1, H]],
                        offset=t_off * C * H + c * H,
                    ),
                )
            else:
                nisa.dma_copy(
                    dst=v_c[0:tok, 0:H],
                    src=stream.ap(pattern=[[H, tok], [1, H]], offset=t_off * H),
                )

            scratch = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
            # ss_c = sum_h v_c ** 2
            nisa.tensor_tensor(
                dst=scratch[0:tok, 0:H],
                data1=v_c[0:tok, 0:H],
                data2=v_c[0:tok, 0:H],
                op=nl.multiply,
            )
            ss = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=ss[0:tok, 0:1], data=scratch[0:tok, 0:H], op=nl.add, axis=(1,))
            # dot_c = sum_h v_c * score_w
            nisa.tensor_tensor(
                dst=scratch[0:tok, 0:H],
                data1=v_c[0:tok, 0:H],
                data2=sw[0:tok, 0:H],
                op=nl.multiply,
            )
            dot = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_reduce(dst=dot[0:tok, 0:1], data=scratch[0:tok, 0:H], op=nl.add, axis=(1,))
            # r = rsqrt(ss / H + eps), fused: activation computes op(scale*data + bias).
            # s_c = dot * r.
            r = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(
                dst=r[0:tok, 0:1],
                data=ss[0:tok, 0:1],
                op=nl.rsqrt,
                scale=inv_h,
                bias=eps,
            )
            nisa.tensor_tensor(
                dst=scores[0:tok, nl.ds(c, 1)],
                data1=dot[0:tok, 0:1],
                data2=r[0:tok, 0:1],
                op=nl.multiply,
            )

        # ---------- softmax over the C+1 candidates (free axis) ----------
        row_max = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(
            dst=row_max[0:tok, 0:1],
            data=scores[0:tok, 0:Cp1],
            op=nl.maximum,
            axis=(1,),
        )
        neg_max = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=neg_max[0:tok, 0:1],
            data=row_max[0:tok, 0:1],
            op0=nl.multiply,
            operand0=-1.0,
        )
        expv = nl.ndarray((_PMAX, Cp1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(
            dst=expv[0:tok, 0:Cp1],
            data=scores[0:tok, 0:Cp1],
            op=nl.exp,
            bias=neg_max[0:tok, 0:1],
        )
        denom = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=denom[0:tok, 0:1], data=expv[0:tok, 0:Cp1], op=nl.add, axis=(1,))
        inv_denom = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=inv_denom[0:tok, 0:1], data=denom[0:tok, 0:1])
        probs = nl.ndarray((_PMAX, Cp1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=probs[0:tok, 0:Cp1],
            data=expv[0:tok, 0:Cp1],
            op0=nl.multiply,
            operand0=inv_denom[0:tok, 0:1],
        )

        # ---------- pass 2: out = sum_c p_c * v_c ----------
        out_acc = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
        for c in range(Cp1):
            v_c = nl.ndarray((_PMAX, H), dtype=dtype, buffer=nl.sbuf)
            if c < C:
                nisa.dma_copy(
                    dst=v_c[0:tok, 0:H],
                    src=bank.ap(
                        pattern=[[C * H, tok], [1, H]],
                        offset=t_off * C * H + c * H,
                    ),
                )
            else:
                nisa.dma_copy(
                    dst=v_c[0:tok, 0:H],
                    src=stream.ap(pattern=[[H, tok], [1, H]], offset=t_off * H),
                )
            p_c = probs[0:tok, nl.ds(c, 1)]
            if c == 0:
                nisa.tensor_scalar(
                    dst=out_acc[0:tok, 0:H],
                    data=v_c[0:tok, 0:H],
                    op0=nl.multiply,
                    operand0=p_c,
                )
            else:
                weighted = nl.ndarray((_PMAX, H), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=weighted[0:tok, 0:H],
                    data=v_c[0:tok, 0:H],
                    op0=nl.multiply,
                    operand0=p_c,
                )
                nisa.tensor_tensor(
                    dst=out_acc[0:tok, 0:H],
                    data1=out_acc[0:tok, 0:H],
                    data2=weighted[0:tok, 0:H],
                    op=nl.add,
                )

        out_bf = nl.ndarray((_PMAX, H), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_bf[0:tok, 0:H], src=out_acc[0:tok, 0:H])
        nisa.dma_copy(
            dst=out.ap(pattern=[[H, tok], [1, H]], offset=t_off * H),
            src=out_bf[0:tok, 0:H],
        )

    return out
