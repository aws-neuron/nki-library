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

"""Tensor-parallel output all-reduce for the CSA attention blocks.

This module provides the 2-LNC `ncc.all_reduce(op=add)` collective that sums the
head-parallel RowParallelLinear output PARTIALS across the `tp_size` CHIP-level
ranks. It is authored so it can be used TWO ways:

  1. MERGED into the attention block (`csa_block.py`): the block computes its
     rank-local partial and then calls `tp_all_reduce(...)` as the final op of its
     own forward, so a single `torch_neuronx.trace(block, ...)` emits ONE
     integrated lnc=2 NEFF that returns the full [B,S,dim] output.
  2. STANDALONE (`TPAllReduceNKI`): the same collective traced on its own, for
     profiling the collective in isolation.

The decode block traces at --logical-nc-config=2 (its attention/indexer kernels
use both logical cores). Earlier attempts to fold the collective in FAILED with
`[NCC_ILLC059] Could not find MemoryLocation ...:src on core 1` (neuronx-cc
status 70) because the collective kernel was launched on a `[1]` grid inside the
lnc=2 block graph — its src/dst were never materialized on logical core 1. The
fix is to launch the collective on the `[2]` grid (matching the block's lnc=2)
and make it 2-LNC-CORRECT:

  * At lnc=2 the collective is SHARDED across the 2 logical cores of each rank —
    core c reduces a DISJOINT free-axis slice of the tensor (via nl.program_id /
    nl.num_programs, exactly like nki_indexer_score_2core scores disjoint T_c
    halves). The two cores' slices
    together cover the whole tensor EXACTLY ONCE, in parallel — never the
    double-reduce a program_id-agnostic `[2]` launch would produce (both cores
    redundantly issuing the SAME full collective over the SAME shared buffer).
  * The all_reduce reduces across the `tp_size` CHIP-level ranks over NeuronLink;
    the framework maps logical core c across ranks (core c @ rank0..N-1 form one
    channel), so each core's disjoint-slice collective is independently correct.
"""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup
from torch import nn


@nki.jit
def nki_tp_all_reduce_kernel(input: nl.NkiTensor, replica_group: ReplicaGroup) -> nl.NkiTensor:
    """Sum `input` across the ranks of `replica_group` (op=nl.add) at lnc=2.

    `input` is a 2D [P, F] tile. This is the canonical nki-library
    all_reduce_hbm_kernel over the WHOLE tensor — collective src/dst must be
    freshly-allocated nl.shared_hbm WITH name= (else NCC_IBIR440 DRAM-alloc
    failure), and a collective cannot read/write IO tensors directly, so the
    input is staged in via dma_copy and the result copied back out.

    Launched on the `[2]` grid (`nki_tp_all_reduce_kernel[2]`) so it runs inside
    the block's lnc=2 context — a `[1]`-grid collective embedded in the lnc=2
    block graph fails with `[NCC_ILLC059] Could not find MemoryLocation ...:src
    on core 1` (its src/dst are never materialized on logical core 1). The single
    whole-tensor `ncc.all_reduce(replica_group=[[0..N-1]])` reduces across the N
    CHIP-level ranks; the lnc=2 lowering distributes that ONE collective across
    the rank's 2 logical cores automatically. This must NOT be hand-split into
    two per-core row-block collectives (`nl.program_id`-sliced src/dst): doing so
    wires only ONE of the two slice-collectives across the ranks over NeuronLink
    and leaves the other row-block at its unreduced local value (observed:
    rms_rel~0.71 == sqrt(0.5) on the full output, i.e. exactly half unreduced).
    """
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm)

    nisa.dma_copy(dst=src, src=input, priority=0)

    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=replica_group, priority=0)
    nisa.dma_copy(dst=out, src=dst, priority=1)
    return out


def tp_all_reduce(partial, replica_ranks):
    """All-reduce (sum) a [B, 1, dim] partial across `replica_ranks` on 2 LNC.

    Reshapes the partial to a balanced 2D [P, F] tile and launches the collective
    on the `[2]` grid (so it runs in the block's lnc=2 context). The all_reduce
    sum is element-wise (layout-invariant, value-EXACT), so the reshape does not
    affect the result; a [1, dim] single-partition input would instead leave a
    logical core empty. P is chosen so total is P*F with P even. Returns the full
    summed [B, 1, dim].
    """
    bsz, seqlen, dim = partial.shape
    total = bsz * seqlen * dim
    P = 128
    while P > 1 and (total % P != 0 or P % 2 != 0):
        P -= 2
    flat = partial.reshape(P, total // P).contiguous()
    replica_group = ReplicaGroup([list(replica_ranks)])
    summed = nki_tp_all_reduce_kernel[2](flat, replica_group)  # [P, total//P]
    return summed.reshape(bsz, seqlen, dim)


class TPAllReduceNKI(nn.Module):
    """Standalone module wrapping ONLY the 2-LNC ncc.all_reduce collective, so
    torch_neuronx.trace(..., compiler_args=["--logical-nc-config=2"]) produces a
    NEFF that is JUST the collective (for isolated profiling).

    replica_ranks defines the ReplicaGroup ([[0,1,2,3]] for a 4-rank world); on a
    genuine multi-worker launch every rank returns the full summed [B,1,dim], and
    with a single-rank group it is an identity passthrough.
    """

    def __init__(self, replica_ranks):
        super().__init__()
        self.replica_ranks = list(replica_ranks)

    def forward(self, partial):
        return tp_all_reduce(partial, self.replica_ranks)
