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

"""Shared constants, the host-built mask pack and one tile helper for the gdn_cte family.

Who uses what: ``gdn_cte`` takes the algorithm selectors, ``gdn_cte_schur`` takes everything
(the pack, ``group_layout`` and ``transpose_tile``), and ``gdn_cte_neumann`` imports NOTHING
from here -- it builds its constants on device and tiles at its own private chunk length.
``gdn_cte_torch`` imports the selectors and ``MAX_HEAD_DIM``, but deliberately MIRRORS the
pack's plane layout rather than importing it, so that a reordering here fails its validator
instead of passing unnoticed.
"""

import math
from typing import Any, Optional

import nki.isa as nisa
import nki.language as nl
import numpy as np

ALGORITHM_SCHUR = "schur"
ALGORITHM_NEUMANN = "neumann"
DEFAULT_ALGORITHM = ALGORITHM_SCHUR
ALGORITHMS = (ALGORITHM_SCHUR, ALGORITHM_NEUMANN)
"""Algorithm selectors for `gdn_cte`, named after the intra-chunk triangular-inverse method
each uses. Kept here so the torch reference can validate the same set without importing a
kernel module."""

CHUNK = 128
"""Chunk length. CHUNK is the partition dim of the (CHUNK, CHUNK) decay/interaction tiles, so
`nl.tile_size.pmax` is both the maximum and the measured optimum. This is the SCHUR chunk;
gdn_cte_neumann tiles at 64 (its own private _CHUNK) and enforces its own constraint."""

MAX_HEAD_DIM = nl.tile_size.pmax
"""Head-dimension ceiling for both algorithms: the partition bound."""

COPY_ENGINE = nisa.engine.scalar
"""Engine for PSUM->SBUF evacuation copies. Only the scalar and vector engines can reach PSUM
at all, and the scalar engine measured best for both the format conversions (several thousand
per sequence) and the accumulator evacuations."""


def _merge_level_off_masks(chunk_len: int) -> list:
    """Per-level OFF-block masks for the recursive block-merge triangular inverse.

    At each level the merge is inv([[P, 0], [R, Q]]), and the only mask it needs selects the
    R (off-diagonal) block of every 2x2 pair: rows base+bs..base+2bs against cols
    base..base+bs.

    Args:
        chunk_len (int): Chunk length (the mask edge length).

    Returns:
        list: One fp32 (chunk_len, chunk_len) off mask per merge level, for block sizes
            1, 2, 4, ... chunk_len/2.
    """
    masks = []
    block_size = 1
    while block_size < chunk_len:
        merged = 2 * block_size
        off = np.zeros((chunk_len, chunk_len), dtype=np.float32)
        for base in range(0, chunk_len, merged):
            # The R block of this 2x2 pair: rows of Q against columns of P.
            p_start, q_start, pair_end = base, base + block_size, base + merged
            off[q_start:pair_end, p_start:q_start] = 1.0
        masks.append(off)
        block_size = merged
    return masks


N_MERGE_LEVELS = len(_merge_level_off_masks(CHUNK))

IDX_UPPER_NEG = 0
IDX_TRIL_STRICT = 1
IDX_EYE = 2
IDX_OFF0 = 3
IDX_NEG_OFF = 4
N_MASKS = IDX_NEG_OFF + (N_MERGE_LEVELS - 1)
"""Mask pack plane layout. Only what the mask-free merge actually reads:

    0 upper_neg, 1 tril_strict, 2 eye, 3 off mask at level 0,
    4 + i  ->  -off_masks[i + 1]   for i in [0, N_MERGE_LEVELS - 1)

There is no neg_off plane for level 0 by construction: level 0 is pure elementwise
(Inv == I there) and builds Inv from A directly via IDX_OFF0."""

_NEG_INF_MASK_VALUE = -1e9
"""The additive -inf stand-in that masks the strictly-upper triangle before exp().
exp(-1e9) underflows to 0 in fp32, which is what makes the later `W = M * QK` need no
separate lower-triangular mask."""


def build_mask_pack(chunk_len: int = CHUNK) -> np.ndarray:
    """Build the (N_MASKS, chunk_len, chunk_len) fp32 mask pack the kernel requires.

    Called on the HOST because the kernel cannot build these itself: every plane is a
    triangular or block-diagonal PATTERN, and the only on-device way to synthesize one is
    `affine_select` (or an iota compare) per plane per chunk -- instructions on the engines
    the merge is already bound by. Built once here, they arrive as a single DMA the caller
    pays for in HBM, not in instructions.

    Args:
        chunk_len (int): Chunk length. Must equal the kernel's CHUNK.

    Returns:
        np.ndarray: (N_MASKS, chunk_len, chunk_len) fp32, C-contiguous.

    Notes:
        The per-level off masks are stored ALREADY NEGATED, which removes a per-level
        ``* (-1.0)`` pass over Off_T inside the merge loop. Level 0's is stored un-negated
        and un-transposed: its consumer builds Inv (not Inv_T) elementwise, so that the
        transpose which follows hands the next merge level BOTH orientations for free.
    """
    if chunk_len != CHUNK:
        n_levels = len(_merge_level_off_masks(chunk_len))
        n_planes = IDX_NEG_OFF + (n_levels - 1)
    else:
        n_planes = N_MASKS
    planes = {}
    ones = np.ones((chunk_len, chunk_len), dtype=np.float32)
    planes[IDX_UPPER_NEG] = (ones - np.tril(ones, k=0)) * _NEG_INF_MASK_VALUE
    planes[IDX_TRIL_STRICT] = np.tril(ones, k=-1)
    planes[IDX_EYE] = np.eye(chunk_len, dtype=np.float32)
    off_masks = _merge_level_off_masks(chunk_len)
    # Level 0 is consumed un-negated (it builds Inv from A by subtracting from I); every
    # later level is consumed negated. All are in their natural orientation.
    planes[IDX_OFF0] = off_masks[0]
    for level_idx in range(1, len(off_masks)):
        planes[IDX_NEG_OFF + level_idx - 1] = -off_masks[level_idx]
    ordered = [planes[plane_idx] for plane_idx in range(n_planes)]
    # No cast: every plane above is constructed fp32, and `np.stack` preserves that.
    return np.ascontiguousarray(np.stack(ordered, axis=0))


def group_layout(n_chunks: int, group_size: int) -> tuple[int, int, int]:
    """Split ``n_chunks`` into full groups of ``group_size`` plus one residual group.

    The requested group size is honored rather than lowered: the kernel emits
    ``n_chunks // group`` groups of exactly ``group`` chunks and then, if anything is left
    over, ONE more group of ``n_chunks % group`` chunks.

    Args:
        n_chunks (int): Number of chunks in the sequence.
        group_size (int): Requested chunks per group.

    Returns:
        tuple[int, int, int]: ``(group, n_full_groups, residual)``. ``group`` is the
            requested size, clamped to ``n_chunks`` so a short sequence still runs one
            group; ``residual`` is 0 when the split is exact.
    """
    group = min(group_size, n_chunks)
    n_full_groups, residual = divmod(n_chunks, group)
    return group, n_full_groups, residual


def transpose_tile(
    tile: nl.NkiTensor,
    rows: int,
    cols: int,
    out_dtype: Optional[Any] = None,
    scale: Optional[Any] = None,
) -> nl.NkiTensor:
    """Transpose a (rows, cols) tile to (cols, rows) on the tensor engine.

    Args:
        tile (nl.NkiTensor): (rows, cols) SBUF tile.
        rows (int): Partition count of the input.
        cols (int): Free size of the input.
        out_dtype: SBUF result dtype. Defaults to ``tile.dtype``.
        scale: Optional factor applied ON THE EVACUATION, so it costs nothing: either a host
            float or a (cols, 1) tile, i.e. one value per OUTPUT partition. Callers that need
            a scaled transpose should always pass it here rather than emit a second pass.

    Returns:
        nl.NkiTensor: (cols, rows) SBUF tile.

    Notes:
        The PSUM tile keeps the INPUT dtype: on gen3+ a tensor-engine transpose
        requires dst dtype == input dtype. Any narrowing happens on the PSUM->SBUF
        copy instead, which runs either way.
    """
    if out_dtype is None:
        out_dtype = tile.dtype
    psum = nl.ndarray((cols, rows), dtype=tile.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=psum, data=tile, engine=nisa.tensor_engine)
    out = nl.ndarray((cols, rows), dtype=out_dtype, buffer=nl.sbuf)
    if scale is None:
        nisa.tensor_copy(dst=out, src=psum, engine=COPY_ENGINE)
    else:
        nisa.activation(dst=out, op=nl.copy, data=psum, scale=scale)
    return out


# log2(CHUNK) == N_MERGE_LEVELS is what makes the level list complete. Plain `assert`, not
# `kernel_assert`: a HOST-side import-time invariant, not kernel input validation.
assert N_MERGE_LEVELS == int(math.log2(CHUNK)), (  # noqa: S101
    f"CHUNK={CHUNK} must be a power of two; got {N_MERGE_LEVELS} merge levels"
)
