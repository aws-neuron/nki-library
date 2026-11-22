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

"""Unified SSD (State Space Duality) kernel for Mamba-2 prefill with dispatch."""

import nki
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from .ssd_block import ssd_block
from .ssd_head_outer import ssd_head_outer

P_MAX = nl.tile_size.pmax
F_MAX = nl.tile_size.psum_fmax

_CHUNK_OUTER_HEAD_THRESHOLD = 8


@nki.jit
def ssd(
    x: nl.NkiTensor,
    dt: nl.NkiTensor,
    A: nl.NkiTensor,
    B: nl.NkiTensor,
    C: nl.NkiTensor,
    chunk_size: int = 128,
    D: nl.NkiTensor = None,
    initial_state: nl.NkiTensor = None,
    causal_mask: nl.NkiTensor = None,
) -> tuple:
    """State Space Duality (SSD) scan for Mamba-2 prefill.

    Dispatches between two internal implementations based on the number of
    heads, which determines how effectively B/C projections amortize:

    - Head-outer (nheads < 8): State stays in SBUF across chunks. Better when
      few heads make B/C sharing savings small relative to state HBM traffic.
    - Chunk-outer (nheads >= 8): B/C loaded once per chunk, shared across heads.
      Better at real model head counts where sharing amortizes across many heads.

    The threshold reflects the crossover observed on Trainium hardware — below
    ~8 local heads (post-TP sharding), head-outer wins; above, chunk-outer wins.

    Dimensions:
        batch: Batch size.
        nheads: Number of SSM heads (post-TP sharding).
        seqlen: Sequence length (must be divisible by chunk_size).
        headdim: Per-head dimension. Tiled when > 512.
        dstate: SSM state dimension (<= 128 for partition dimension).
        Q: Chunk size (= chunk_size, <= 128).

    Args:
        x (nl.NkiTensor): [batch, nheads, seqlen, headdim], Input activations.
        dt (nl.NkiTensor): [batch, nheads, seqlen], Softplus'd timesteps. Must be positive.
        A (nl.NkiTensor): [nheads], State transition scalars. Typically negative.
        B (nl.NkiTensor): [batch, seqlen, dstate], Input projection.
        C (nl.NkiTensor): [batch, seqlen, dstate], Output projection.
        chunk_size (int): Chunk size for parallel scan. Must be <= 128.
        D (nl.NkiTensor, optional): [nheads], Skip connection weights.
        initial_state (nl.NkiTensor, optional): [batch, nheads, dstate, headdim], Initial
            SSM state. Default: None (zeros).
        causal_mask (nl.NkiTensor): [Q, Q], Lower-triangular mask. Required.
            Pass np.tril(np.ones((Q, Q), dtype=np.float32)).

    Returns:
        tuple: (y, final_state)
            - y (nl.NkiTensor): [batch, nheads, seqlen, headdim], Output, same dtype as x.
            - final_state (nl.NkiTensor): [batch, nheads, dstate, headdim], Final SSM state
              in float32.

    Notes:
        - chunk_size <= 128 (must fit in partition dimension)
        - dstate <= 128 (must fit in partition dimension for SBUF state)
        - seqlen must be divisible by chunk_size
        - ngroups=1 (B/C shared across all heads)
        - Uses float32 accumulation internally for numerical stability
        - A should be negative for stable dynamics (decay < 1)
        - dt should be positive; discretization computes exp(dt * A)
        - TODO: dstate > 128 support via tiled state through HBM (needed for
          Falcon-H1 1.5B+ which uses dstate=256)
        - TODO: ngroups > 1 support via per-group B/C projections (needed for
          Zamba2-7B ngroups=2, Falcon-H1 34B ngroups=2). Assign groups to
          NeuronCores so each core only loads its groups' B/C — avoids
          replication when ngroups divides evenly across LNC cores.

    Pseudocode:
        if nheads >= 8:
            chunk_outer_ssd(...)   # shared B/C projections across heads
        else:
            head_outer_ssd(...)    # state persists in SBUF across chunks
    """
    nheads = x.shape[1]
    headdim = x.shape[3]
    Q = chunk_size

    kernel_assert(x.shape[2] % Q == 0, "seqlen must be divisible by chunk_size")
    kernel_assert(Q <= P_MAX, f"chunk_size must be <= {P_MAX}, got {Q}")
    kernel_assert(B.shape[2] <= P_MAX, f"dstate must be <= {P_MAX}, got {B.shape[2]}")
    kernel_assert(causal_mask != None, "causal_mask is required")

    # Chunk-outer when many heads (B/C sharing wins) or large headdim (needs PSUM tiling)
    if nheads >= _CHUNK_OUTER_HEAD_THRESHOLD or headdim > F_MAX:
        return ssd_block(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=D,
            initial_state=initial_state,
            causal_mask=causal_mask,
        )
    else:
        return ssd_head_outer(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=D,
            initial_state=initial_state,
            causal_mask=causal_mask,
        )
