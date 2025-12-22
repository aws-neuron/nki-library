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

from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info

NUM_COALESCE_TILES = 8


@nki.jit
def rope_hf(
    q: nl.ndarray,
    k: nl.ndarray,
    q_out: nl.ndarray,
    k_out: nl.ndarray,
    cos: Optional[nl.ndarray] = None,
    sin: Optional[nl.ndarray] = None,
    rope_cache: Optional[nl.ndarray] = None,
    backward: bool = False,
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Apply Rotary Position Embedding (RoPE) to query and key tensors using HuggingFace layout.

    This kernel uses following tensor layout: [batch, heads, seq, head_dim],
    which is common in HuggingFace models (e.g., Llama). Use the `rope` kernel
    instead for following tensor layout: [head_dim, batch, heads, seq].

    RoPE encodes positional information by rotating pairs of embedding dimensions
    using precomputed sine/cosine frequencies, enabling position-aware attention
    without absolute position embeddings.

    Forward pass formula (split-half rotation):
        q_out[..., :half] = q[..., :half] * cos[..., :half] - q[..., half:] * sin[..., :half]
        q_out[..., half:] = q[..., half:] * cos[..., half:] + q[..., :half] * sin[..., half:]
        (same for k)

    Backward pass reverses the rotation direction for gradient computation.

    Grid: (lnc)

    Args:
        q: Query tensor [batch_size, q_heads, seq_len, head_dim] in HBM.
        k: Key tensor [batch_size, k_heads, seq_len, head_dim] in HBM.
        q_out: Output query tensor [batch_size, q_heads, seq_len, head_dim] in HBM.
        k_out: Output key tensor [batch_size, k_heads, seq_len, head_dim] in HBM.
        cos: Cosine embeddings [optional(batch_size), seq_len, head_dim] or [seq_len, head_dim] in HBM.
            Required if rope_cache is None.
        sin: Sine embeddings [optional(batch_size), seq_len, head_dim] or [seq_len, head_dim] in HBM.
            Required if rope_cache is None.
        rope_cache: Packed cos/sin tensor [optional(batch_size), seq_len, head_dim*2] or
            [seq_len, head_dim*2] in HBM. First half contains cos, second half contains sin.
            Alternative to providing separate cos/sin tensors.
        backward: If True, compute backward pass (gradient w.r.t. inputs).
            Default is False (forward pass).

    Returns:
        Tuple[q_out, k_out]: Query and key tensors with rotary embeddings applied.

    Note:
        - Either (cos, sin) or rope_cache must be provided, not both.
        - seq_len must be divisible by (128 * num_lnc_shards).

    See Also:
        rope: RoPE kernel using input tensor layout [head_dim, batch, heads, seq_len].
    """
    _, num_shards, shard_id = get_verified_program_sharding_info()
    _validate_apply_rope_inputs(q, k, cos, sin, rope_cache, num_shards)

    batch_size, _, seq_len, head_dim = q.shape

    # how much of total seq_len dim per shard (half for lnc 2)
    seq_per_shard = seq_len // num_shards
    seq_shard_offset = shard_id * seq_per_shard
    seq_tile_size = min(nl.tile_size.pmax, seq_per_shard)

    # how many tiles on shard dim
    num_seq_tiles_per_shard = div_ceil(seq_per_shard, seq_tile_size)
    for batch_id in range(batch_size):
        for seq_tile_idx in range(0, num_seq_tiles_per_shard, NUM_COALESCE_TILES):
            # Number of tiles to process at a time
            num_tiles = min(NUM_COALESCE_TILES, num_seq_tiles_per_shard - seq_tile_idx)
            seq_start = seq_shard_offset + seq_tile_idx * seq_tile_size

            cos_tile = nl.ndarray((seq_tile_size, num_tiles, head_dim), dtype=q.dtype, buffer=nl.sbuf)
            sin_tile = nl.ndarray((seq_tile_size, num_tiles, head_dim), dtype=q.dtype, buffer=nl.sbuf)

            # Slice cos, sin tiles using access pattern depending on input shapes
            rope_shape = (rope_cache or cos).shape
            rope_seq_len, rope_dim = rope_shape[-2:]
            if len(rope_shape) == 3:
                rope_batch_offset = batch_id
            else:
                rope_batch_offset = 0

            if rope_cache is not None:
                # Packed rope_cache format
                # rope_cache: [..., head_dim*2]
                # Access cos portion (first half of last dimension)
                cos_ap_src = rope_cache.ap(
                    pattern=[[rope_dim, seq_tile_size], [seq_tile_size * rope_dim, num_tiles], [1, head_dim]],
                    offset=rope_batch_offset * rope_seq_len * rope_dim + seq_start * rope_dim,
                )
                # Access sin portion (second half of last dimension)
                sin_ap_src = rope_cache.ap(
                    pattern=[[rope_dim, seq_tile_size], [seq_tile_size * rope_dim, num_tiles], [1, head_dim]],
                    offset=rope_batch_offset * rope_seq_len * rope_dim + seq_start * rope_dim + head_dim,
                )
            else:
                # cos, sin format
                # cos: [..., head_dim]
                cos_ap_src = cos.ap(
                    pattern=[[head_dim, seq_tile_size], [seq_tile_size * head_dim, num_tiles], [1, head_dim]],
                    offset=rope_batch_offset * rope_seq_len * rope_dim + seq_start * head_dim,
                )
                sin_ap_src = sin.ap(
                    pattern=[[head_dim, seq_tile_size], [seq_tile_size * head_dim, num_tiles], [1, head_dim]],
                    offset=rope_batch_offset * rope_seq_len * rope_dim + seq_start * head_dim,
                )

            nisa.dma_copy(cos_tile, cos_ap_src)
            nisa.dma_copy(sin_tile, sin_ap_src)

            _apply_rope_all_heads(q, q_out, cos_tile, sin_tile, batch_id, seq_start, backward)
            _apply_rope_all_heads(k, k_out, cos_tile, sin_tile, batch_id, seq_start, backward)

    return q_out, k_out


def _validate_apply_rope_inputs(
    q: nl.ndarray,
    k: nl.ndarray,
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_cache: Optional[nl.ndarray],
    num_shards: int,
) -> None:
    """
    Validate inputs for apply_rope operation.

    Args:
      q: Query tensor [batch_size, q_heads, seq_len, head_dim]
      k: Key tensor [batch_size, k_heads, seq_len, head_dim]
      cos: Cosine embeddings [batch_size, seq_len, head_dim] or [batch_size, seq_len, head_dim*2] or [seq_len, head_dim*2]
      sin: Sine embeddings [batch_size, seq_len, head_dim] or None (when cos contains packed rope_cache)
      num_shards: Number of LNC shards
    """
    kernel_assert(len(q.shape) == 4, f"apply_rope only supports 4D tensors, got input shape {q.shape}")
    kernel_assert(q.shape[-1] == k.shape[-1], f"Head dim mismatch: q={q.shape[-1]}, k={k.shape[-1]}")

    head_dim = q.shape[-1]

    if rope_cache is not None:
        # Packed rope_cache format: contains both cos and sin
        kernel_assert(
            rope_cache.shape[-1] == head_dim * 2,
            f"Packed rope_cache expects head_dim*2={head_dim * 2}, got {rope_cache.shape[-1]}",
        )
        kernel_assert(cos is None and sin is None, "Expected either rope_cache or separate cos/sin tensors, got both")
        rope_tensor = rope_cache
    else:
        # Separate cos/sin tensors
        kernel_assert(
            cos is not None and sin is not None, "Expected either rope_cache or separate cos/sin tensors, got none"
        )
        kernel_assert(
            cos.shape == sin.shape, f"Shape of cos Tensor: {cos.shape} doesn't match shape of sin Tensor: {sin.shape}"
        )
        kernel_assert(cos.shape[-1] == head_dim, f"Head dim mismatch: q={head_dim}, cos={cos.shape[-1]}")
        rope_tensor = cos

    # Validate batch dimension (optional for rope cache)
    kernel_assert(
        len(rope_tensor.shape) == 2 or (len(rope_tensor.shape) == 3 and rope_tensor.shape[0] == q.shape[0]),
        f"rope_cache or cos tensor batch size {rope_tensor.shape} doesn't match q batch size {q.shape}",
    )
    kernel_assert(
        rope_tensor.shape[-2] >= q.shape[2],
        f"rope_cache or cos tensor seq len {rope_tensor.shape[-2]} smaller than q seq len {q.shape[2]}",
    )
    kernel_assert(
        q.shape[2] % (nl.tile_size.pmax * num_shards) == 0,
        f"seq len must be multiple of 128 * LNC={num_shards}, got {q.shape[2]}",
    )


def _apply_rope_all_heads(
    x: nl.ndarray,
    x_out: nl.ndarray,
    cos_tile: nl.ndarray,
    sin_tile: nl.ndarray,
    batch_id: int,
    seq_start: int,
    backward: bool = False,
) -> None:
    """
    Apply rotary embedding to a single tensor tile for each attention head using
    preloaded cos/sin tiles.

    Args:
        x: Input tensor [batch_size, num_heads, seq_len, head_dim] in HBM
        x_out: Output tensor [batch_size, num_heads, seq_len, head_dim] in HBM
        cos_tile: Cosine embedding tile [seq_tile_size, num_tiles, head_dim] in SBUF
        sin_tile: Sine embedding tile [seq_tile_size, num_tiles, head_dim] in SBUF
        batch_id: Current batch index
        seq_start: Starting sequence position
        backward: If True, compute backward pass rotation. Default False.
    """
    seq_tile_size, num_tiles, head_dim = cos_tile.shape
    num_heads = x.shape[1]

    # Process all heads sequentially
    for head_id in range(num_heads):
        x_tile = nl.ndarray((seq_tile_size, num_tiles, head_dim), dtype=x.dtype, buffer=nl.sbuf)
        x_ap_src = x.ap(
            pattern=[[head_dim, seq_tile_size], [seq_tile_size * head_dim, num_tiles], [1, head_dim]],
            offset=batch_id * x.shape[1] * x.shape[2] * x.shape[3]
            + head_id * x.shape[2] * x.shape[3]
            + seq_start * head_dim,
        )
        nisa.dma_copy(x_tile, x_ap_src)

        x_rotated = _apply_rope_single(x_tile, cos_tile, sin_tile, backward)
        x_out_ap = x_out.ap(
            pattern=[[head_dim, seq_tile_size], [seq_tile_size * head_dim, num_tiles], [1, head_dim]],
            offset=batch_id * x_out.shape[1] * x_out.shape[2] * x_out.shape[3]
            + head_id * x_out.shape[2] * x_out.shape[3]
            + seq_start * head_dim,
        )
        nisa.dma_copy(x_out_ap, x_rotated)


def _apply_rope_single(
    x_tile: nl.ndarray,
    cos_tile: nl.ndarray,
    sin_tile: nl.ndarray,
    backward: bool = False,
) -> nl.ndarray:
    """
    Apply rotary embedding to a single tensor tile in SBUF.

    Computes the RoPE rotation for one tile of Q or K tensor using
    preloaded cos/sin tiles.

    Args:
        x_tile: Input tensor tile [seq_tile_size, num_tiles, head_dim] in SBUF
        cos_tile: Cosine embedding tile [seq_tile_size, num_tiles, head_dim] in SBUF
        sin_tile: Sine embedding tile [seq_tile_size, num_tiles, head_dim] in SBUF
        backward: If True, compute backward pass rotation. Default False.

    Returns:
        Rotated tensor tile [seq_tile_size, num_tiles, head_dim] in SBUF.
    """
    seq_size, num_tiles, head_dim = x_tile.shape
    result = nl.ndarray((seq_size, num_tiles, head_dim), dtype=x_tile.dtype, buffer=nl.sbuf)

    # Create temporary tensors for intermediate calculations
    temp1 = nl.ndarray((seq_size, num_tiles, head_dim // 2), dtype=x_tile.dtype, buffer=nl.sbuf)
    temp2 = nl.ndarray((seq_size, num_tiles, head_dim // 2), dtype=x_tile.dtype, buffer=nl.sbuf)

    if not backward:
        # Forward: y = [x1, x2] * [cos1, cos2] + [-x2, x1] * [sin1, sin2]
        # result = x_tile * cos_tile
        nisa.tensor_tensor(dst=result, data1=x_tile, data2=cos_tile, op=nl.multiply)

        # temp1 = x_tile[..., head_dim//2:] * sin_tile[..., :head_dim//2]
        nisa.tensor_tensor(
            dst=temp1, data1=x_tile[:, :, head_dim // 2 :], data2=sin_tile[:, :, : head_dim // 2], op=nl.multiply
        )

        # result[..., :head_dim//2] -= temp1
        nisa.tensor_tensor(
            dst=result[:, :, : head_dim // 2], data1=result[:, :, : head_dim // 2], data2=temp1, op=nl.subtract
        )

        # temp2 = x_tile[..., :head_dim//2] * sin_tile[..., head_dim//2:]
        nisa.tensor_tensor(
            dst=temp2, data1=x_tile[:, :, : head_dim // 2], data2=sin_tile[:, :, head_dim // 2 :], op=nl.multiply
        )

        # result[..., head_dim//2:] += temp2
        nisa.tensor_tensor(
            dst=result[:, :, head_dim // 2 :], data1=result[:, :, head_dim // 2 :], data2=temp2, op=nl.add
        )
    else:
        # Backward: dy = [dx1, dx2] * [cos1, cos2] + [dx2, -dx1] * [sin2, sin1]
        # result = x_tile * cos_tile
        nisa.tensor_tensor(dst=result, data1=x_tile, data2=cos_tile, op=nl.multiply)

        # temp1 = x_tile[..., head_dim//2:] * sin_tile[..., head_dim//2:]
        nisa.tensor_tensor(
            dst=temp1, data1=x_tile[:, :, head_dim // 2 :], data2=sin_tile[:, :, head_dim // 2 :], op=nl.multiply
        )

        # result[..., :head_dim//2] += temp1
        nisa.tensor_tensor(
            dst=result[:, :, : head_dim // 2], data1=result[:, :, : head_dim // 2], data2=temp1, op=nl.add
        )

        # temp2 = x_tile[..., :head_dim//2] * sin_tile[..., :head_dim//2]
        nisa.tensor_tensor(
            dst=temp2, data1=x_tile[:, :, : head_dim // 2], data2=sin_tile[:, :, : head_dim // 2], op=nl.multiply
        )

        # result[..., head_dim//2:] -= temp2
        nisa.tensor_tensor(
            dst=result[:, :, head_dim // 2 :], data1=result[:, :, head_dim // 2 :], data2=temp2, op=nl.subtract
        )

    return result
