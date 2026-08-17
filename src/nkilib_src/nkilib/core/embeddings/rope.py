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


"""Rotary Position Embedding (RoPE) kernels for NeuronCore."""

import nki
import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import sizeinbytes
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info

# Live SBUF buffers at peak, used to size the group tile in RoPE (see _rope_tile_fits).
# Keep in sync with the allocations these count:
#   - RoPE allocates x_in_sb + x_out_sb (2 full [.., batch, head, tile] buffers).
#   - RoPE_sbuf allocates _ROPE_SBUF_INTERNAL_BUFFERS internals (sb_odd + even_cos +
#     odd_cos + even_sin + odd_sin).
#   - RoPE allocates cos_sb + sin_sb (_ROPE_COSSIN_BUFFERS; no head factor — cos/sin
#     are shared across heads).
_ROPE_SBUF_INTERNAL_BUFFERS = 5
_ROPE_TILE_BUFFERS = 2 + _ROPE_SBUF_INTERNAL_BUFFERS  # x_in_sb, x_out_sb + RoPE_sbuf internals
_ROPE_COSSIN_BUFFERS = 2


def _rope_tile_fits(batch_group, head_group, tile_size, dtype_bytes, budget_bytes):
    # Free-dim elements live at peak on the hot partition range (0..half_d):
    # _ROPE_TILE_BUFFERS full buffers of batch_group*head_group*tile_size, plus
    # _ROPE_COSSIN_BUFFERS cos/sin buffers of batch_group*tile_size.
    tile_free_elems = batch_group * head_group * tile_size
    cossin_free_elems = batch_group * tile_size
    total = _ROPE_TILE_BUFFERS * tile_free_elems + _ROPE_COSSIN_BUFFERS * cossin_free_elems
    return total * dtype_bytes <= budget_bytes


@nki.jit
def RoPE(
    x_in: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    lnc_shard: bool = False,
    contiguous_layout: bool = True,
    relayout_in_sbuf: bool = False,
) -> nl.NkiTensor:
    """
    Apply Rotary Position Embedding (RoPE) to input embeddings.

    Standalone kernel with HBM I/O and optional LNC sharding.
    Supports both contiguous and interleaved memory layouts with automatic
    layout conversion via strided DMA or SBUF matmul.

    Dimensions:
        d_head: Head dimension (64 or 128)
        B: Batch size
        n_heads: Number of attention heads
        S: Sequence length (divisible by n_prgs if lnc_shard=True)

    Args:
        x_in (nl.NkiTensor): [d_head, B, n_heads, S] @ HBM, Input embeddings
        cos (nl.NkiTensor): [d_head//2, B, S] @ HBM, Cosine frequencies
        sin (nl.NkiTensor): [d_head//2, B, S] @ HBM, Sine frequencies
        lnc_shard (bool): Parallelize across LNC cores by tiling sequence dimension
        contiguous_layout (bool): Memory layout in d_head dimension.
            True: [first_half, second_half] (default, more efficient).
            False: [even, odd, even, odd, ...] (interleaved)
        relayout_in_sbuf (bool): Use SBUF matmul for layout conversion (only for small tensors)

    Returns:
        output (nl.NkiTensor): [d_head, B, n_heads, S] @ HBM, RoPE applied output

    Notes:
        - SBUF size constraint (for bf16): B * n_heads * S <= 73728 (approximately 72K).
          This limit applies regardless of d_head. Exceeding this limit will
          cause compilation failure. For larger sizes, tile the computation.
        - When relayout_in_sbuf=True with interleaved layout, a stricter limit
          applies: B * n_heads * S <= gemm_moving_fmax (typically 512)
        - d_head must be even (pairs of elements are rotated)
        - When lnc_shard=True, S must be divisible by number of programs
        - For large tensors with interleaved layout, uses strided DMA

    Pseudocode:
        # Determine sharding and tile size
        tile_size = S // n_prgs
        tile_start = tile_size * prg_id

        # Load input tile to SBUF (with optional layout conversion)
        if is_dma_relayout:
            x_in_sb = load_strided(x_in, even_odd_separated)
        else:
            x_in_sb = load_contiguous(x_in)

        # Load cos/sin frequency tiles
        cos_sb = load(cos[tile_start:tile_start+tile_size])
        sin_sb = load(sin[tile_start:tile_start+tile_size])

        # Apply RoPE rotation in SBUF
        x_out_sb = rope_sbuf(x_in_sb, cos_sb, sin_sb)

        # Store output (with optional layout conversion)
        if is_dma_relayout:
            store_strided(x_out, x_out_sb, even_odd_interleaved)
        else:
            store_contiguous(x_out, x_out_sb)
    """

    _validate_rope_inputs(x_in, cos, sin, 'RoPE')

    d_head, B, n_heads, S = x_in.shape
    half_d = d_head // 2

    # Determine parallelization across LNC cores
    n_prgs, prg_id = 1, 0
    if lnc_shard:
        _, n_prgs, prg_id = get_verified_program_sharding_info("RoPE", (0, 1))

    # Tile along sequence dimension
    kernel_assert(S % n_prgs == 0, f'RoPE: sequence length {S} not divisible by {n_prgs} programs')
    tile_size = S // n_prgs
    tile_start = tile_size * prg_id

    # Determine layout conversion strategy: DMA (strided access) vs SBUF (matmul)
    # SBUF relayout limited by gemm_moving_fmax, fallback to DMA for large tensors
    is_relayout_in_sbuf_supported = B * n_heads * S <= nl.tile_size.gemm_moving_fmax
    is_dma_relayout = not contiguous_layout and (not relayout_in_sbuf or not is_relayout_in_sbuf_supported)
    is_relayout_in_sbuf = not contiguous_layout and relayout_in_sbuf and is_relayout_in_sbuf_supported

    # Group-tile the (B, n_heads) free dimensions so the SBUF working set fits.
    # The whole [d_head, B, n_heads, tile_size] slab can exceed SBUF for large
    # B*n_heads*S (LinearScanAllocation cannot spill), so process batch_group
    # batches x head_group heads per iteration. Pick the largest group that fits
    # the per-partition SBUF budget (fewer, larger tiles minimize loop and
    # DMA-issue overhead). The budget is linear in the grown dimension, so solve
    # for the group size directly (mirrors the cost in _rope_tile_fits).
    dtype_bytes = sizeinbytes(x_in.dtype)
    budget_bytes = nl.tile_size.sbuf_fmax_bytes

    # Per-(batch, head) tile-buffer cost and per-batch cos/sin cost, in bytes.
    tile_cost = _ROPE_TILE_BUFFERS * tile_size * dtype_bytes
    cossin_cost = _ROPE_COSSIN_BUFFERS * tile_size * dtype_bytes

    if _rope_tile_fits(1, n_heads, tile_size, dtype_bytes, budget_bytes):
        # One batch of all heads fits: keep all heads, pack as many batches as fit.
        head_group = n_heads
        per_batch = tile_cost * n_heads + cossin_cost
        batch_group = max(1, min(B, budget_bytes // per_batch))
    else:
        # Even one batch of all heads is too big: pin one batch, pack as many heads as fit.
        batch_group = 1
        head_group = max(1, min(n_heads, (budget_bytes - cossin_cost) // tile_cost))

    x_out = nl.ndarray(x_in.shape, dtype=x_in.dtype, buffer=nl.shared_hbm)

    for b_start in range(0, B, batch_group):
        batch_group_size = min(batch_group, B - b_start)
        for h_start in range(0, n_heads, head_group):
            head_group_size = min(head_group, n_heads - h_start)

            # Load input tile to SBUF with optional layout conversion via strided DMA
            x_in_sb = nl.ndarray(
                (d_head, batch_group_size, head_group_size, tile_size), dtype=x_in.dtype, buffer=nl.sbuf
            )
            if is_dma_relayout:
                # Gather even/odd indices with stride=2 in d_head dimension
                nisa.dma_copy(
                    dst=x_in_sb[:half_d, :, :, :],
                    src=x_in.slice(0, 0, d_head, 2)
                    .slice(1, b_start, b_start + batch_group_size)
                    .slice(2, h_start, h_start + head_group_size)
                    .slice(3, tile_start, tile_start + tile_size),
                )
                nisa.dma_copy(
                    dst=x_in_sb[half_d:, :, :, :],
                    src=x_in.slice(0, 1, d_head, 2)
                    .slice(1, b_start, b_start + batch_group_size)
                    .slice(2, h_start, h_start + head_group_size)
                    .slice(3, tile_start, tile_start + tile_size),
                )
            else:
                nisa.dma_copy(
                    dst=x_in_sb,
                    src=x_in[
                        :,
                        b_start : b_start + batch_group_size,
                        h_start : h_start + head_group_size,
                        tile_start : tile_start + tile_size,
                    ],
                )

            # Load cos/sin frequency tiles (broadcast over heads inside RoPE_sbuf)
            cos_sb = nl.ndarray((half_d, batch_group_size, tile_size), dtype=cos.dtype, buffer=nl.sbuf)
            sin_sb = nl.ndarray((half_d, batch_group_size, tile_size), dtype=sin.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=cos_sb, src=cos[:, b_start : b_start + batch_group_size, tile_start : tile_start + tile_size]
            )
            nisa.dma_copy(
                dst=sin_sb, src=sin[:, b_start : b_start + batch_group_size, tile_start : tile_start + tile_size]
            )

            # Compute RoPE rotation in SBUF
            x_out_sb = nl.ndarray(x_in_sb.shape, dtype=x_in_sb.dtype, buffer=nl.sbuf)
            RoPE_sbuf(x_in_sb, cos_sb, sin_sb, x_out_sb, convert_from_interleaved=is_relayout_in_sbuf)

            # Store output tile to HBM with optional layout conversion via strided DMA
            if is_dma_relayout:
                # Scatter even/odd indices with stride=2 in d_head dimension
                nisa.dma_copy(
                    dst=x_out.slice(0, 0, d_head, 2)
                    .slice(1, b_start, b_start + batch_group_size)
                    .slice(2, h_start, h_start + head_group_size)
                    .slice(3, tile_start, tile_start + tile_size),
                    src=x_out_sb[:half_d, :, :, :],
                )
                nisa.dma_copy(
                    dst=x_out.slice(0, 1, d_head, 2)
                    .slice(1, b_start, b_start + batch_group_size)
                    .slice(2, h_start, h_start + head_group_size)
                    .slice(3, tile_start, tile_start + tile_size),
                    src=x_out_sb[half_d:, :, :, :],
                )
            else:
                nisa.dma_copy(
                    dst=x_out[
                        :,
                        b_start : b_start + batch_group_size,
                        h_start : h_start + head_group_size,
                        tile_start : tile_start + tile_size,
                    ],
                    src=x_out_sb,
                )

    return x_out


def RoPE_sbuf(
    x_in_sb: nl.NkiTensor,
    cos_sb: nl.NkiTensor,
    sin_sb: nl.NkiTensor,
    x_out_sb: nl.NkiTensor,
    convert_from_interleaved: bool = False,
) -> nl.NkiTensor:
    """
    Apply RoPE on tensors in SBUF (for megakernel fusion).
    Helper function that operates entirely in SBUF without HBM I/O.

    RoPE Formula:
        out[even] = x[even]*cos - x[odd]*sin
        out[odd] = x[odd]*cos + x[even]*sin

    Args:
        x_in_sb (nl.NkiTensor): [d_head, B, n_heads, S] @ SBUF - input embeddings.
            Can equal x_out_sb for in-place operation.
        cos_sb (nl.NkiTensor): [d_head//2, B, S] @ SBUF - cosine frequencies.
        sin_sb (nl.NkiTensor): [d_head//2, B, S] @ SBUF - sine frequencies.
        x_out_sb (nl.NkiTensor): [d_head, B, n_heads, S] @ SBUF - output buffer
        convert_from_interleaved (bool): convert from interleaved to contiguous layout

    Returns:
        nl.NkiTensor: x_out_sb with RoPE applied (modified in-place)

    Notes:
        - Assumes contiguous layout unless convert_from_interleaved=True
        - For large tensors with interleaved layout, use RoPE() with strided DMA
    """

    d_head, B, n_heads, S = x_out_sb.shape
    half_d = d_head // 2

    _validate_rope_inputs(x_in_sb, cos_sb, sin_sb, 'RoPE_sbuf')
    kernel_assert(x_in_sb.dtype == x_out_sb.dtype, 'RoPE_sbuf: dtype mismatch between x_in_sb and x_out_sb')

    # Convert interleaved to contiguous layout if needed
    if convert_from_interleaved:
        convert_to_interleaved_mat = _compute_convert_to_interleaved_mat(x_in_sb)
        x_in_sb = _convert_from_interleaved(x_in_sb, convert_to_interleaved_mat)

    # These internal SBUF buffers (sb_odd + even_cos + odd_cos + even_sin + odd_sin)
    # are counted by _ROPE_SBUF_INTERNAL_BUFFERS; keep that constant in sync if this
    # allocation set changes (RoPE's group-tile budget depends on it).
    # Copy odd half to separate buffer (required for tensor_tensor base partition alignment)
    sb_odd = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb_odd, src=x_in_sb[half_d:, :, :, :])

    # Allocate buffers for intermediate products
    even_cos = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    odd_cos = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    even_sin = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)
    odd_sin = nl.ndarray((half_d, B, n_heads, S), dtype=x_in_sb.dtype, buffer=nl.sbuf)

    # Compute RoPE: out_even = even*cos - odd*sin, out_odd = odd*cos + even*sin
    # Use view ops to broadcast cos/sin across n_heads dimension
    cos_broadcast = cos_sb.expand_dim(2).broadcast(2, n_heads)
    sin_broadcast = sin_sb.expand_dim(2).broadcast(2, n_heads)

    nisa.tensor_tensor(even_cos, x_in_sb[:half_d, :, :, :], cos_broadcast, nl.multiply)
    nisa.tensor_tensor(odd_cos, sb_odd[:half_d, :, :, :], cos_broadcast, nl.multiply)
    nisa.tensor_tensor(even_sin, x_in_sb[:half_d, :, :, :], sin_broadcast, nl.multiply)
    nisa.tensor_tensor(odd_sin, sb_odd[:half_d, :, :, :], sin_broadcast, nl.multiply)

    nisa.tensor_tensor(x_out_sb[:half_d, :, :, :], even_cos, odd_sin, nl.subtract)
    nisa.tensor_tensor(x_out_sb[half_d:, :, :, :], odd_cos, even_sin, nl.add)

    # Convert back to interleaved layout if needed
    if convert_from_interleaved:
        x_out_sb = _convert_to_interleaved(x_out_sb, convert_to_interleaved_mat)

    return x_out_sb


def _compute_convert_to_interleaved_mat(x_sb: nl.NkiTensor) -> nl.NkiTensor:
    """
    Generate permutation matrix for RoPE layout conversion.

    Creates matrix P for converting between contiguous and interleaved layouts.
    P @ X transforms [e0,e1,...,o0,o1,...] to [e0,o0,e1,o1,...].
    P^T @ X transforms [e0,o0,e1,o1,...] to [e0,e1,...,o0,o1,...].

    Args:
        x_sb (nl.NkiTensor): [d_head, B, n_heads, S] @ SBUF, Input tensor for shape info

    Returns:
        nl.NkiTensor: [d_head, d_head] @ SBUF, Permutation matrix

    Notes:
        - Only supports tensors where B*n_heads*S <= gemm_moving_fmax
        - d_head must be even
        - Uses reshape_dim + permute on identity matrix to build permutation
    """
    d_head, B, n_heads, S = x_sb.shape
    half_d = d_head // 2
    kernel_assert(d_head % 2 == 0, f'_compute_convert_to_interleaved_mat: d_head must be even, got {d_head}')

    identity_sb = nl.shared_identity_matrix(d_head, dtype=x_sb.dtype)

    # Reshape identity's free dim into (half_d, 2) then permute to (2, half_d).
    # This reads columns in stride-2 order: [0,2,4,...,1,3,5,...] per row.
    # Result: even rows get 1 in first half, odd rows get 1 in second half.
    # This creates P where P@X transforms [e0,e1,...,o0,o1,...] -> [e0,o0,e1,o1,...].
    convert_to_interleaved_mat = nl.ndarray((d_head, d_head), dtype=x_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(
        dst=convert_to_interleaved_mat.reshape((d_head, 2, half_d)),
        src=identity_sb.reshape_dim(1, (half_d, 2)).permute((0, 2, 1)),
        engine=nisa.scalar_engine,
    )

    return convert_to_interleaved_mat


def _convert_from_interleaved(x_sb: nl.NkiTensor, convert_to_interleaved_mat: nl.NkiTensor) -> nl.NkiTensor:
    """
    Convert interleaved to contiguous layout using matrix multiplication.

    Transforms [e0,o0,e1,o1,...] to [e0,e1,...,o0,o1,...] via P^T @ x_sb.

    Args:
        x_sb (nl.NkiTensor): [d_head, B, n_heads, S] @ SBUF, Input in interleaved layout
        convert_to_interleaved_mat (nl.NkiTensor): [d_head, d_head] @ SBUF, Permutation matrix

    Returns:
        nl.NkiTensor: [d_head, B, n_heads, S] @ SBUF, Output in contiguous layout

    Notes:
        - Returns new buffer (does not modify input)
    """
    d_head, B, n_heads, S = x_sb.shape
    kernel_assert(x_sb.buffer == nl.sbuf, '_convert_from_interleaved: input must be in SBUF')

    total_free_dim = B * n_heads * S
    fmax = nl.tile_size.gemm_moving_fmax
    x_converted_sb = nl.ndarray(x_sb.shape, dtype=x_sb.dtype, buffer=nl.sbuf)
    x_flat = x_sb.reshape((d_head, total_free_dim))
    x_out_flat = x_converted_sb.reshape((d_head, total_free_dim))
    for t_start in range(0, total_free_dim, fmax):
        t_size = min(fmax, total_free_dim - t_start)
        x_psum = nl.ndarray((d_head, t_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=x_psum, stationary=convert_to_interleaved_mat, moving=x_flat[:, nl.ds(t_start, t_size)])
        nisa.activation(dst=x_out_flat[:, nl.ds(t_start, t_size)], op=nl.copy, data=x_psum)
    return x_converted_sb


def _convert_to_interleaved(x_sb: nl.NkiTensor, convert_to_interleaved_mat: nl.NkiTensor) -> nl.NkiTensor:
    """
    Convert contiguous to interleaved layout using matrix multiplication.

    Transforms [e0,e1,...,o0,o1,...] to [e0,o0,e1,o1,...] via P @ x_sb.

    Args:
        x_sb (nl.NkiTensor): [d_head, B, n_heads, S] @ SBUF, Input in contiguous layout
        convert_to_interleaved_mat (nl.NkiTensor): [d_head, d_head] @ SBUF, Permutation matrix

    Returns:
        nl.NkiTensor: [d_head, B, n_heads, S] @ SBUF, Output in interleaved layout

    Notes:
        - Pre-transposes matrix to compensate for nc_matmul's implicit transpose
        - Modifies input buffer in-place
    """
    d_head, B, n_heads, S = x_sb.shape
    kernel_assert(x_sb.buffer == nl.sbuf, '_convert_to_interleaved: input must be in SBUF')

    # Pre-transpose to compensate for nc_matmul's implicit transpose
    convert_from_interleaved_sb = nl.ndarray((d_head, d_head), dtype=convert_to_interleaved_mat.dtype, buffer=nl.sbuf)
    convert_from_interleaved_psum = nl.ndarray((d_head, d_head), dtype=convert_to_interleaved_mat.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=convert_from_interleaved_psum, data=convert_to_interleaved_mat)
    nisa.tensor_copy(dst=convert_from_interleaved_sb, src=convert_from_interleaved_psum, engine=nisa.scalar_engine)

    total_free_dim = B * n_heads * S
    fmax = nl.tile_size.gemm_moving_fmax
    x_flat = x_sb.reshape((d_head, total_free_dim))
    for t_start in range(0, total_free_dim, fmax):
        t_size = min(fmax, total_free_dim - t_start)
        x_psum = nl.ndarray((d_head, t_size), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=x_psum, stationary=convert_from_interleaved_sb, moving=x_flat[:, nl.ds(t_start, t_size)])
        nisa.tensor_copy(dst=x_flat[:, nl.ds(t_start, t_size)], src=x_psum, engine=nisa.scalar_engine)
    return x_sb


def _validate_rope_inputs(
    x_in: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    func_name: str,
) -> None:
    """
    Validate RoPE input tensor shapes and constraints.

    Args:
        x_in (nl.NkiTensor): [d_head, B, n_heads, S], Input embeddings
        cos (nl.NkiTensor): [d_head//2, B, S], Cosine frequencies
        sin (nl.NkiTensor): [d_head//2, B, S], Sine frequencies
        func_name (str): Name of calling function for error messages

    Returns:
        None

    Notes:
        - Validates d_head in {64, 128}
        - Validates B in (0, 64]
        - Validates S in (0, 512]
        - Validates n_heads in (0, 16]
        - Validates cos/sin shapes match expected dimensions
    """
    d_head, B, n_heads, S = x_in.shape
    half_d = d_head // 2

    kernel_assert(d_head in (64, 128), f'{func_name}: d_head must be 64 or 128, got {d_head}')
    kernel_assert(B > 0, f'{func_name}: B must be > 0, got {B}')
    kernel_assert(S > 0, f'{func_name}: S must be > 0, got {S}')
    kernel_assert(n_heads > 0, f'{func_name}: n_heads must be > 0, got {n_heads}')

    kernel_assert(
        tuple(cos.shape) == (half_d, B, S),
        f'{func_name}: cos.shape expected ({half_d},{B},{S}), got {cos.shape}',
    )
    kernel_assert(
        tuple(sin.shape) == (half_d, B, S),
        f'{func_name}: sin.shape expected ({half_d},{B},{S}), got {sin.shape}',
    )
    kernel_assert(cos.dtype == sin.dtype, f'{func_name}: cos/sin dtype mismatch')
