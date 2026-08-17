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

"""BF16 sub-kernels for the Sparse Attention Indexer.

Contains the LayerNorm + RoPE helpers used to process K: ``_layernorm``,
``_rope_non_interleaved``, and the fused ``fused_layernorm_rope_k``.
"""

import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_helpers import div_ceil
from .sparse_attention_indexer_utils import P_MAX


# LayerNorm using bn_stats + bn_aggr (matches QKV CTE pattern).
def _layernorm(
    sbm: SbufManager,
    x_sb: nl.ndarray,
    gamma_sb: nl.ndarray,
    beta_sb: nl.ndarray,
    out_sb: nl.ndarray,
    S: int,
    H: int,
    eps: float = 1e-6,
) -> None:
    """LayerNorm with fused gamma/beta application.

    Uses nisa.bn_stats + nisa.bn_aggr to compute mean/var in one Vector Engine pass,
    then a fused scalar_tensor_tensor for ((x - mean) * rvar * gamma) + beta.

    gamma_sb/beta_sb are [P_MAX, H] (pre-broadcast by caller).
    """
    BN_STATS_TILE_SIZE = 512
    BN_STATS_DST_SIZE = 6
    NUM_BN_TILES = div_ceil(H, BN_STATS_TILE_SIZE)

    sbm.open_scope(name="layernorm")

    bn_stats_sb = sbm.alloc_stack((P_MAX, BN_STATS_DST_SIZE * NUM_BN_TILES), nl.float32)
    aggr_sb = sbm.alloc_stack((P_MAX, 2), nl.float32)  # [mean, rvar]

    for bn_tile_idx in nl.affine_range(NUM_BN_TILES):
        bn_off = bn_tile_idx * BN_STATS_TILE_SIZE
        bn_sz = min(BN_STATS_TILE_SIZE, H - bn_off)
        nisa.bn_stats(
            dst=bn_stats_sb[:S, nl.ds(bn_tile_idx * BN_STATS_DST_SIZE, BN_STATS_DST_SIZE)],
            data=x_sb[:S, nl.ds(bn_off, bn_sz)],
        )

    nisa.bn_aggr(
        dst=aggr_sb[:S, 0:2],
        data=bn_stats_sb[:S, 0 : BN_STATS_DST_SIZE * NUM_BN_TILES],
    )

    # rvar = 1 / sqrt(var + eps) in place.
    nisa.activation(
        dst=aggr_sb[:S, 1:2],
        op=nl.rsqrt,
        data=aggr_sb[:S, 1:2],
        bias=eps,
    )

    # x_norm = (x - mean) * rvar  (fused subtract + multiply via tensor_scalar two-op)
    x_norm = sbm.alloc_stack((P_MAX, H), nl.float32)
    nisa.tensor_scalar(
        dst=x_norm[:S, :H],
        data=x_sb[:S, :H],
        op0=nl.subtract,
        operand0=aggr_sb[:S, 0:1],
        op1=nl.multiply,
        operand1=aggr_sb[:S, 1:2],
    )

    # out = x_norm * gamma, then + beta below (op0 identity multiply by 1.0, op1 multiplies gamma)
    nisa.scalar_tensor_tensor(
        dst=out_sb[:S, :H],
        data=x_norm[:S, :H],
        op0=nl.multiply,
        operand0=1.0,  # identity scalar; gamma multiply is applied via op1
        op1=nl.multiply,
        operand1=gamma_sb[:S, :H],
    )
    nisa.tensor_tensor(
        dst=out_sb[:S, :H],
        data1=out_sb[:S, :H],
        data2=beta_sb[:S, :H],
        op=nl.add,
    )

    sbm.close_scope()


# RoPE — non-interleaved, applied once per S-tile to all Q heads or to K.
def _rope_non_interleaved(
    sbm: SbufManager,
    x_sb: nl.ndarray,
    cos_sb: nl.ndarray,
    sin_sb: nl.ndarray,
    out_sb: nl.ndarray,
    S: int,
    D: int,
) -> None:
    """In-place non-interleaved RoPE on a [S, D] slab using minimum ops.

    out = [X1*cos - X2*sin, X1*sin + X2*cos]
    """
    D_half = D // 2
    sbm.open_scope(name="rope")

    t_neg_x2_sin = sbm.alloc_stack((P_MAX, D_half), nl.float32)
    t_x1_sin = sbm.alloc_stack((P_MAX, D_half), nl.float32)
    t_x_cos = sbm.alloc_stack((P_MAX, D), nl.float32)

    # -X2 * sin
    nisa.scalar_tensor_tensor(
        dst=t_neg_x2_sin[:S, :D_half],
        data=x_sb[:S, D_half:D],
        op0=nl.multiply,
        operand0=-1.0,
        op1=nl.multiply,
        operand1=sin_sb[:S, :D_half],
    )
    # X1 * sin
    nisa.tensor_tensor(
        dst=t_x1_sin[:S, :D_half],
        data1=x_sb[:S, :D_half],
        data2=sin_sb[:S, :D_half],
        op=nl.multiply,
    )
    # X * cos (full slab)
    # cos_sb is [S, D_half]; broadcast across the two halves.
    nisa.tensor_tensor(
        dst=t_x_cos[:S, :D_half],
        data1=x_sb[:S, :D_half],
        data2=cos_sb[:S, :D_half],
        op=nl.multiply,
    )
    nisa.tensor_tensor(
        dst=t_x_cos[:S, D_half:D],
        data1=x_sb[:S, D_half:D],
        data2=cos_sb[:S, :D_half],
        op=nl.multiply,
    )
    # First half of output: X1*cos - X2*sin = t_x_cos[:D/2] + t_neg_x2_sin
    nisa.tensor_tensor(
        dst=out_sb[:S, :D_half],
        data1=t_x_cos[:S, :D_half],
        data2=t_neg_x2_sin[:S, :D_half],
        op=nl.add,
    )
    # Second half: X1*sin + X2*cos = t_x1_sin + t_x_cos[D/2:]
    nisa.tensor_tensor(
        dst=out_sb[:S, D_half:D],
        data1=t_x1_sin[:S, :D_half],
        data2=t_x_cos[:S, D_half:D],
        op=nl.add,
    )

    sbm.close_scope()


def fused_layernorm_rope_k(
    sbm: SbufManager,
    k_sb: nl.ndarray,
    gamma_sb: nl.ndarray,
    beta_sb: nl.ndarray,
    cos_sb: nl.ndarray,
    sin_sb: nl.ndarray,
    k_out: nl.ndarray,
    S: int,
    head_dim: int,
    rope_head_dim: int,
) -> None:
    """LayerNorm K, then RoPE the rope-slice in place, copying non-rope dims through.

    Applies LayerNorm over the full head_dim, then non-interleaved RoPE on the leading
    rope_head_dim columns. Any remaining (nope) tail columns are copied through unchanged.

    Args:
        sbm (SbufManager): SBUF stack allocator used for scratch buffers.
        k_sb (nl.ndarray): [S, head_dim], Input K in SBUF.
        gamma_sb (nl.ndarray): [P_MAX, head_dim], LayerNorm scale (pre-broadcast by caller).
        beta_sb (nl.ndarray): [P_MAX, head_dim], LayerNorm bias (pre-broadcast by caller).
        cos_sb (nl.ndarray): [S, rope_head_dim // 2], RoPE cosine frequencies in SBUF.
        sin_sb (nl.ndarray): [S, rope_head_dim // 2], RoPE sine frequencies in SBUF.
        k_out (nl.ndarray): [S, head_dim], Output buffer for LayerNorm+RoPE result in SBUF.
        S (int): Active sequence length (partition rows) for this tile.
        head_dim (int): Total head dimension.
        rope_head_dim (int): Leading number of columns to apply RoPE to.

    Returns:
        None: Result is written in place to k_out.

    Notes:
        - When head_dim > rope_head_dim, the trailing nope columns are copied through unchanged.
    """
    sbm.open_scope(name="ln_rope_k")

    k_normed = sbm.alloc_stack((P_MAX, head_dim), nl.float32)
    _layernorm(sbm, k_sb, gamma_sb, beta_sb, k_normed, S, head_dim)

    # RoPE on the rope_head_dim slice; copy nope tail through.
    _rope_non_interleaved(sbm, k_normed[:, :rope_head_dim], cos_sb, sin_sb, k_out[:, :rope_head_dim], S, rope_head_dim)
    if head_dim > rope_head_dim:
        nisa.tensor_copy(dst=k_out[:, rope_head_dim:head_dim], src=k_normed[:, rope_head_dim:head_dim])

    sbm.close_scope()
