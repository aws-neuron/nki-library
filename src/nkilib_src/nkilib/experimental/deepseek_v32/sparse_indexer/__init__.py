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

"""Public API for the DeepSeek-V3.2 sparse indexer kernel."""

from dataclasses import dataclass

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

P_MAX = nl.tile_size.pmax
"""Maximum partition dimension of a tile."""

F_MAX = nl.tile_size.psum_fmax
"""Maximum free dimension per PSUM bank per partition, in FP32 elements."""

MASK_BOUND = 1.0e30
"""
Bound written into masked score slots. Finite rather than ``-inf``: the top-k consumer reads
bfloat16, where ``-inf`` makes the hardware max-select produce NaN.
"""

K_PACK = 4
"""
FP8 values packed into one ``fp8x4`` word along the contraction dimension, per the OCP
microscaling spec. One MX K-tile is therefore ``P_MAX * K_PACK`` contraction elements, and the
query projection requires ``q_lora_rank`` to be a multiple of it.
"""

MX_BLOCK = 32
"""Contraction elements sharing one ``uint8`` e8m0 MX scale."""

SCALE_QUADRANT_SIZE = 32
"""Partitions per hardware MX-scale quadrant."""

SCALES_PER_QUADRANT = 4
"""Valid scale rows within each MX-scale quadrant."""

TOPK_GROUP = 16
"""Partitions that ``nki.isa.topk`` ranks as one snake group."""

TOPK_GROUP_SHIFT = 4
"""``log2(TOPK_GROUP)``, used to split a snake position into its partition and column."""

TOPK_GROUPS_PER_CALL = P_MAX // TOPK_GROUP
"""Sequences ranked per ``nki.isa.topk`` call, one snake group each."""


@dataclass
class DeepseekV32SparseIndexerShape(nl.NKIObject):
    """Shapes and derived tiling constants for one decode step of the sparse indexer."""

    n_tokens: int
    """Decode tokens in the batch, one per sequence."""

    max_seq_len: int
    """Key-cache capacity per sequence, in slots."""

    hidden: int
    """Model hidden width. The contraction dimension of the key and head-weight projections."""

    q_lora_rank: int
    """LoRA rank of the query path. The contraction dimension of the query projection."""

    n_heads: int
    """Indexer heads."""

    head_dim: int
    """Per-head feature width. Must equal ``P_MAX``."""

    rope_dim: int
    """Leading feature span that RoPE rotates. Must be ``<= head_dim`` and even."""

    index_topk: int
    """Slots selected per token, clamped to ``max_seq_len``."""

    dtype: nl.DType
    """Kernel compute dtype."""

    valid_len: int
    """
    Highest occupied slot plus one, clamped to ``max_seq_len``. Only the tiles covering it are
    scored.
    """

    @property
    def rope_dim_half(self) -> int:
        """Feature ``i`` pairs with ``i + rope_dim_half``: RoPE here is non-interleaved."""
        return self.rope_dim // 2

    @property
    def n_tokens_across_heads(self) -> int:
        """Flat free extent of the query tile: column ``head * n_tokens + token``."""
        return self.n_heads * self.n_tokens

    @property
    def n_key_tiles(self) -> int:
        """``P_MAX``-slot tiles spanning the whole cache."""
        return div_ceil(self.max_seq_len, P_MAX)

    @property
    def n_scored_tiles(self) -> int:
        """``P_MAX``-slot tiles spanning ``valid_len``. The tail is written as the mask bound."""
        return div_ceil(self.valid_len, P_MAX)

    @property
    def n_heads_scale(self) -> float:
        """Head-count normalization folded into the head weights."""
        return self.n_heads**-0.5

    @property
    def softmax_scale(self) -> float:
        """Attention scale folded into the head weights."""
        return self.head_dim**-0.5


def get_deepseek_v32_sparse_indexer_shape(
    hidden_states,
    qr,
    wk,
    weights_proj,
    cos,
    key_cache,
    index_topk: int,
    valid_len: int | None = None,
) -> DeepseekV32SparseIndexerShape:
    """
    Derive the shape object from the kernel's own tensors, so that callers pass only what no tensor
    carries: the selection width and the highest occupied slot.

    The clamping below belongs here rather than in ``DeepseekV32SparseIndexerShape.__init__``. The
    ``nl.NKIObject`` metaclass treats an ``__init__`` whose parameters match the dataclass fields as
    the generated one and bypasses it whenever a field is omitted, which silently discards any
    derived value it computes.
    """

    max_seq_len = key_cache.shape[1]

    shapes = DeepseekV32SparseIndexerShape(
        n_tokens=hidden_states.shape[0],
        max_seq_len=max_seq_len,
        hidden=hidden_states.shape[1],
        q_lora_rank=qr.shape[1],
        n_heads=weights_proj.shape[1],
        head_dim=wk.shape[1],
        rope_dim=2 * cos.shape[1],
        # nisa.topk requires k <= n, and selecting more slots than the cache holds is meaningless,
        # so clamp to match the reference's min(index_topk, max_seq_len).
        index_topk=min(index_topk, max_seq_len),
        dtype=hidden_states.dtype,
        # No bound from the caller means every slot may be occupied, so the whole cache is scored.
        valid_len=max_seq_len if valid_len is None else min(valid_len, max_seq_len),
    )

    kernel_assert(qr.shape[0] == shapes.n_tokens, f"qr rows {qr.shape[0]} must equal n_tokens {shapes.n_tokens}")
    kernel_assert(wk.shape[0] == shapes.hidden, f"wk rows {wk.shape[0]} must equal hidden {shapes.hidden}")
    kernel_assert(
        weights_proj.shape[0] == shapes.hidden,
        f"weights_proj rows {weights_proj.shape[0]} must equal hidden {shapes.hidden}",
    )
    kernel_assert(
        key_cache.shape[0] == shapes.n_tokens,
        f"key_cache rows {key_cache.shape[0]} must equal n_tokens {shapes.n_tokens}",
    )
    kernel_assert(
        key_cache.shape[2] == shapes.head_dim,
        f"key_cache last dim {key_cache.shape[2]} must equal head_dim {shapes.head_dim}",
    )
    return shapes


from .sparse_indexer import DeepseekV32SparseIndexerOutput, deepseek_v32_sparse_indexer
from .sparse_indexer_torch import deepseek_v32_sparse_indexer_torch_ref

__all__ = [
    "DeepseekV32SparseIndexerOutput",
    "DeepseekV32SparseIndexerShape",
    "get_deepseek_v32_sparse_indexer_shape",
    "deepseek_v32_sparse_indexer",
    "deepseek_v32_sparse_indexer_torch_ref",
]
