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

"""Public API for the DeepSeek-V3.2 MLA decode QKV front-fold kernel."""

from dataclasses import dataclass

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert

P_MAX = nl.tile_size.pmax
"""Maximum partition dimension of a tile."""

F_MAX = nl.tile_size.psum_fmax
"""Maximum free dimension per PSUM bank per partition, in FP32 elements."""


@dataclass
class MlaDecodeQkvShape(nl.NKIObject):
    """Shapes and derived widths for one decode step of the MLA QKV front-fold."""

    n_tokens: int
    """Decode tokens in the batch, one per sequence."""

    n_heads: int
    """Attention heads."""

    hidden: int
    """Model hidden width. The contraction dimension of the query and KV down-projections."""

    kv_lora_rank: int
    """LoRA rank of the KV path. Also the width of the latent field in a cache row."""

    qk_nope_dim: int
    """Per-head query/key width that RoPE does not rotate."""

    qk_rope_dim: int
    """Per-head query/key width that RoPE rotates. Must be even."""

    n_blocks: int
    """Blocks in the paged KV cache."""

    block_size: int
    """Cache rows per block."""

    dtype: nl.DType
    """Kernel compute dtype."""

    @property
    def qk_head_dim(self) -> int:
        """One head's QK width: nope plus rope."""
        return self.qk_nope_dim + self.qk_rope_dim

    @property
    def qk_packed_dim(self) -> int:
        """
        Width that ``wq_b`` emits: every head's QK concatenated.

        Element ``off`` of head ``head`` lives at column ``head * qk_head_dim + off``.
        """
        return self.n_heads * self.qk_head_dim

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
        kv_row_dim]`` is what lets a slot index address a row directly.
        """
        return self.n_blocks * self.block_size


def get_mla_decode_qkv_shape(
    hidden_states: nl.NkiTensor,
    wq_a: nl.NkiTensor,
    wq_b: nl.NkiTensor,
    wkv_a: nl.NkiTensor,
    kv_cache: nl.NkiTensor,
    n_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
) -> MlaDecodeQkvShape:
    """
    Derive the shape object from the kernel's own tensors, so that callers pass only the per-head
    widths that no tensor carries.

    The head count and the three per-head widths cannot be recovered from any operand, because
    ``wq_b`` emits every head's QK concatenated into one flat axis and ``wkv_a`` emits ``k_pe`` and
    ``kv_latent`` concatenated into one cache row.
    """

    shapes = MlaDecodeQkvShape(
        n_tokens=hidden_states.shape[0],
        n_heads=n_heads,
        hidden=hidden_states.shape[1],
        kv_lora_rank=kv_lora_rank,
        qk_nope_dim=qk_nope_head_dim,
        qk_rope_dim=qk_rope_head_dim,
        n_blocks=kv_cache.shape[0],
        block_size=kv_cache.shape[2],
        dtype=hidden_states.dtype,
    )

    kernel_assert(shapes.n_tokens <= P_MAX, f"n_tokens={shapes.n_tokens} must be <= {P_MAX}, one token per partition")
    kernel_assert(
        shapes.qk_nope_dim <= P_MAX,
        f"qk_nope_dim={shapes.qk_nope_dim} must be <= {P_MAX}, it is the absorb contraction",
    )
    kernel_assert(shapes.kv_lora_rank % P_MAX == 0, f"kv_lora_rank={shapes.kv_lora_rank} must be a multiple of {P_MAX}")
    kernel_assert(shapes.qk_rope_dim % 2 == 0, f"qk_rope_dim={shapes.qk_rope_dim} must be even for interleaved RoPE")
    kernel_assert(wq_a.shape[0] == shapes.hidden, f"wq_a rows {wq_a.shape[0]} must equal hidden {shapes.hidden}")
    kernel_assert(wkv_a.shape[0] == shapes.hidden, f"wkv_a rows {wkv_a.shape[0]} must equal hidden {shapes.hidden}")
    kernel_assert(
        wq_b.shape[1] == shapes.qk_packed_dim,
        f"wq_b.shape[1]={wq_b.shape[1]} must equal qk_packed_dim={shapes.qk_packed_dim}",
    )
    kernel_assert(
        wkv_a.shape[1] == shapes.kv_row_dim,
        f"wkv_a.shape[1]={wkv_a.shape[1]} must equal kv_row_dim={shapes.kv_row_dim}",
    )
    kernel_assert(
        kv_cache.shape[3] == shapes.kv_row_dim,
        f"kv_cache.shape[3]={kv_cache.shape[3]} must equal kv_row_dim={shapes.kv_row_dim}",
    )
    return shapes


from .mla_decode_qkv import MlaDecodeQkvResult, mla_decode_qkv  # noqa: E402
from .mla_decode_qkv_torch import mla_decode_qkv_torch_ref  # noqa: E402

__all__ = [
    "F_MAX",
    "P_MAX",
    "MlaDecodeQkvResult",
    "MlaDecodeQkvShape",
    "get_mla_decode_qkv_shape",
    "mla_decode_qkv",
    "mla_decode_qkv_torch_ref",
]
