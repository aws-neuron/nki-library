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

"""Public API for the DeepSeek-V3.2 MLA decode attention core over indexer-selected keys."""

from dataclasses import dataclass

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

P_MAX = nl.tile_size.pmax
"""Maximum partition dimension of a tile."""

F_MAX = nl.tile_size.psum_fmax
"""Maximum free dimension per PSUM bank per partition, in FP32 elements."""

_DMA_TRANSPOSE_ALIGN_ELEMS = 16
"""
Destination alignment a DGE crossbar transpose requires, in 2-byte elements.

Each transposed tile must start on a 32-byte boundary, which is 16 elements at bf16.
"""


@dataclass
class MlaDecodeAttentionShape(nl.NKIObject):
    """Shapes and derived widths for one decode step of the MLA attention core."""

    n_tokens: int
    """Decode tokens in the batch, one per sequence."""

    n_heads: int
    """Attention heads on this rank."""

    n_prior: int
    """Prior keys scored per token, which is the indexer's top-k selection count."""

    kv_lora_rank: int
    """LoRA rank of the KV path. Also the width of the latent field in a cache row."""

    qk_rope_dim: int
    """Per-head query/key width that RoPE rotates. Also the width of a cache row's ``k_pe`` field."""

    v_head_dim: int
    """Per-head width the out-absorb projects the latent attention output into."""

    hidden: int
    """Model hidden width, which the o-projection emits."""

    n_blocks: int
    """Blocks in the paged KV cache."""

    block_size: int
    """Cache rows per block."""

    dtype: nl.DType
    """Kernel compute dtype."""

    @property
    def n_keys(self) -> int:
        """Key positions scored per token: the selected prior plus the current token."""
        return self.n_prior + 1

    @property
    def kv_row_k_pe_dim(self) -> int:
        """Width of the ``k_pe`` field, which occupies the start of a cache row."""
        return self.qk_rope_dim

    @property
    def kv_row_latent_dim(self) -> int:
        """Width of the ``kv_latent`` field, which follows ``k_pe`` in a cache row."""
        return self.kv_lora_rank

    @property
    def kv_row_dim(self) -> int:
        """
        Width of one cached KV vector.

        A row is ``cat([k_pe, kv_latent])``, so this is ``kv_row_k_pe_dim + kv_row_latent_dim``.
        """
        return self.kv_row_k_pe_dim + self.kv_row_latent_dim

    @property
    def n_cache_rows(self) -> int:
        """
        Rows in the flattened paged cache.

        Flattening ``[n_blocks, 1, block_size, kv_row_dim]`` to ``[n_blocks * block_size,
        kv_row_dim]`` is what lets a resolved row index address a cache row directly.
        """
        return self.n_blocks * self.block_size

    @property
    def attn_packed_dim(self) -> int:
        """Width the o-projection contracts: every head's ``v_head_dim`` output concatenated."""
        return self.n_heads * self.v_head_dim

    @property
    def n_latent_tiles(self) -> int:
        """Partition tiles the latent field spans, which is the score matmul's contraction count."""
        return div_ceil(self.kv_lora_rank, P_MAX)

    @property
    def n_key_tiles(self) -> int:
        """
        Partition tiles all ``n_keys`` keys span, key ``tile * P_MAX + p`` living on partition ``p``.

        This is the tile count of the score, weight and gathered-row buffers.
        """
        return div_ceil(self.n_keys, P_MAX)

    @property
    def n_prior_tiles(self) -> int:
        """
        Partition tiles the selected prior keys span, which is the index tables' tile count.

        This equals ``n_key_tiles`` when ``n_prior`` is ragged, because the current token's key then
        shares the last tile with prior keys; it is one less when ``n_prior`` is a multiple of
        ``P_MAX`` and the current key gets a tile of its own.
        """
        return div_ceil(self.n_prior, P_MAX)

    @property
    def latent_chunk_stride(self) -> int:
        """
        Stride between latent chunks in the feature-major key buffer.

        ``n_keys`` is padded up to ``_DMA_TRANSPOSE_ALIGN_ELEMS`` so that every chunk's destination
        starts on the 32-byte boundary the DGE crossbar transpose requires. At the production
        ``n_keys`` of 2049 the unpadded stride would put odd chunks on an odd byte offset.
        """
        align = _DMA_TRANSPOSE_ALIGN_ELEMS
        return div_ceil(self.n_keys, align) * align


def get_mla_decode_attention_shape(
    kv_cache: nl.NkiTensor,
    positions: nl.NkiTensor,
    out_absorb_w: nl.NkiTensor,
    o_proj_w: nl.NkiTensor,
    n_prior: int,
) -> MlaDecodeAttentionShape:
    """
    Derive the shape object from the kernel's own tensors, so that callers pass only ``n_prior``.

    Unlike the QKV front-fold, every extent here is recoverable from an operand: ``out_absorb_w``
    carries the head count and both of its widths, ``o_proj_w`` carries ``hidden``, and a cache row's
    width fixes ``qk_rope_dim`` once ``kv_lora_rank`` is known. Only ``n_prior`` is not, because the
    index tables are already tiled onto partitions and a ragged selection count cannot be recovered
    from its tile count.
    """

    kv_lora_rank = out_absorb_w.shape[2]
    kv_row_dim = kv_cache.shape[3]

    shapes = MlaDecodeAttentionShape(
        n_tokens=positions.shape[0],
        n_heads=out_absorb_w.shape[0],
        n_prior=n_prior,
        kv_lora_rank=kv_lora_rank,
        qk_rope_dim=kv_row_dim - kv_lora_rank,
        v_head_dim=out_absorb_w.shape[1],
        hidden=o_proj_w.shape[1],
        n_blocks=kv_cache.shape[0],
        block_size=kv_cache.shape[2],
        dtype=out_absorb_w.dtype,
    )

    kernel_assert(shapes.n_tokens <= P_MAX, f"n_tokens={shapes.n_tokens} must be <= {P_MAX}, one token per partition")
    kernel_assert(shapes.n_heads <= P_MAX, f"n_heads={shapes.n_heads} must be <= {P_MAX}, it is a matmul free axis")
    kernel_assert(shapes.n_prior >= 1, f"n_prior={shapes.n_prior} must select at least one prior key")
    kernel_assert(shapes.kv_lora_rank % P_MAX == 0, f"kv_lora_rank={shapes.kv_lora_rank} must be a multiple of {P_MAX}")
    kernel_assert(
        0 < shapes.qk_rope_dim <= P_MAX,
        f"qk_rope_dim={shapes.qk_rope_dim}, derived from kv_row_dim={kv_row_dim} less "
        f"kv_lora_rank={shapes.kv_lora_rank}, must be in (0, {P_MAX}]",
    )
    kernel_assert(
        shapes.v_head_dim <= P_MAX,
        f"v_head_dim={shapes.v_head_dim} must be <= {P_MAX}, it is the o-projection contraction",
    )
    kernel_assert(
        shapes.kv_lora_rank <= F_MAX,
        f"kv_lora_rank={shapes.kv_lora_rank} must be <= {F_MAX} so the latent output fits one PSUM bank",
    )
    kernel_assert(
        shapes.block_size & (shapes.block_size - 1) == 0,
        f"block_size={shapes.block_size} must be a power of two, the row resolve shifts and masks by it",
    )
    kernel_assert(
        o_proj_w.shape[0] == shapes.attn_packed_dim,
        f"o_proj_w.shape[0]={o_proj_w.shape[0]} must equal attn_packed_dim={shapes.attn_packed_dim}",
    )
    return shapes


from .mla_decode_attention import mla_decode_attention  # noqa: E402
from .mla_decode_attention_torch import mla_decode_attention_torch_ref  # noqa: E402
from .prior_index import PriorIndex, build_prior_index  # noqa: E402

__all__ = [
    "F_MAX",
    "P_MAX",
    "MlaDecodeAttentionShape",
    "PriorIndex",
    "build_prior_index",
    "get_mla_decode_attention_shape",
    "mla_decode_attention",
    "mla_decode_attention_torch_ref",
]
