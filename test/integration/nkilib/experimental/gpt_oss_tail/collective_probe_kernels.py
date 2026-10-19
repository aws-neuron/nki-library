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

"""Minimal probe for the vocab-parallel -> batch-parallel collective, and nothing else.

WHY A PROBE. The GPT-OSS sampling tail wants the FULL vocab on each rank so the top-k can be
taken locally. The direct way to get it is one ``all_to_all`` over all VTP ranks, and at
VTP=128 that aborts in the Neuron runtime:

    encd_mesh_memcopy: Assertion `tx_descs_n[j] == tx_descs_n[j-1]
                                  && rx_descs_n[j] == rx_descs_n[j-1]' failed

MEASURED RESULTS. What an all_to_all replica group may look like is set by its DEVICE LAYOUT,
not by its size. At LNC=1 a trn3 device holds 8 cores, and a group must be either entirely
within ONE device, or hold exactly ONE core per device. A MIXED group -- several devices with
several cores each -- has no mesh subtype that implements all-to-all:

    group                          | device layout          | reserved 16-device trn3 host
    -------------------------------+------------------------+---------------------------------
    8 contiguous  {0..7}           | 1 device, 8 cores      | pass
    4 stride 8    {0,8,16,24}      | 4 devices, 1 core each | pass
    16 stride 8   {0,8,...,120}    | 16 devices, 1 core each| pass
    16 contiguous {0..15}          | 2 devices, 8 cores each| FAIL: op_type 7 unsupported
    128 all ranks                  | 16 devices, 8 cores    | FAIL: op_type 7 unsupported
    4 stride 4    {0,4,8,12}       | 2 devices, 2 cores each| FAIL: encd_mesh_memcopy
    all_gather over 128            | --                     | pass

The rejection is ``__select_mesh_subtype: not a supported operation for mesh (op_type 7)`` and it
is NOT payload driven: the 16-contiguous case carries ``total_sz 0x800`` and the 128-rank case
``0x62300``. Reading it as a size ceiling is the expensive mistake -- it makes 8 look like the
maximum and forces three hops, when ``(8, 16)`` is legal and is what the production sharded tail
already uses. The largest legal group is therefore max(8, 16) = 16, so VTP=128 needs at least TWO
hops, and 8 x 16 = 128 exactly.

Host matters: on the shared fleet's trn3_a0 a single mesh passes at 16 and 32 CONTIGUOUS ranks,
which reserved trn3 rejects. Validate a decomposition on the host that will run it.

TWO COMPILER RULES this module exists partly to record, because both are easy to trip and the
error text does not name the operand: a collective may NOT READ a kernel IO tensor, and may NOT
WRITE one either. Stage inputs into an internally-owned buffer first and copy results out. The
real tail kernel only avoids the read rule incidentally, because its collective source is the
lm_head output it just produced.

These kernels live in the TEST tree, not in src/: they compute nothing the library needs and
exist only to ascertain what the hardware will execute. Their reference is inherently
rank-dependent (a rank recovers its own id from the provenance encoded in its input), which is
not expressible as a convention-dispatch ``<name>_torch_ref``.

THE COMPOSITION. Each hop is an ``all_to_all`` followed by a rows->columns de-interleave, and
hops compose: a hop over ``g`` peers turns ``[rows, w]`` into ``[rows/g, g*w]``. So

    one hop over 128        : [gbs, V_s]  -> [gbs/128, 128*V_s]
    two hops over 8 then 16 : [gbs, V_s]  -> [gbs/8, 8*V_s] -> [gbs/128, 128*V_s]

reach the SAME shape, and both keep column index equal to global vocab id provided every hop's
group is listed in ascending rank order and the first hop's group is 8 CONTIGUOUS ranks. What
differs is row ownership: one hop leaves it the identity, while two hops leave rank ``r``
owning ``(r % 8)*(gbs/8) + (r // 8)*(gbs/128)`` -- a shuffle the caller must undo.

The probe input encodes the source rank in column 0 and the source row in column
1, both exactly representable in bfloat16 (integers up to 256), so ONE run recovers the entire
permutation and the test can check the mapping rather than assume it.
"""

from typing import Tuple

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert

_PMAX = nl.tile_size.pmax  # 128


@nki.jit
def collective_probe(x: nl.NkiTensor, hop_meshes: Tuple = ()) -> nl.NkiTensor:
    """Run the hop sequence on ``x`` and return the assembled ``[gbs/vtp, vtp*w]`` result.

    Args:
        x: ``[gbs, w]`` bfloat16 in HBM.
        hop_meshes: one replica-group mesh per hop, from ``compose_hops``.

    Returns:
        bfloat16 ``[gbs / prod(hop sizes), prod(hop sizes) * w]`` in HBM.
    """
    kernel_assert(len(hop_meshes) >= 1, "at least one hop is required")
    rows, width = x.shape

    # Stage the input into an internally-owned buffer before the first hop. A collective may
    # NOT read a kernel IO tensor -- the compiler's BIR verifier rejects it outright with
    # "Collective instruction cannot read IO tensors" -- and the real tail kernel only avoids
    # this because its collective source is the lm_head output it just produced.
    staged = nl.ndarray((rows, width), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    # ceil, not floor. `rows // _PMAX` leaves rows [floor*_PMAX, rows) NEVER WRITTEN whenever
    # rows > _PMAX and rows % _PMAX != 0 (rows=200 -> one tile, 72 rows unwritten), yet the
    # collective below reads ALL of `staged` -- so those rows carry whatever was last in that HBM
    # page and the probe silently reports a mesh verdict computed from garbage. Not hit today only
    # because every probe gbs is <= 128 or a multiple of 128; the all_gather copy-out below always
    # used ceil, which is what makes this a slip rather than a constraint.
    for t in range((rows + _PMAX - 1) // _PMAX):
        r0 = t * _PMAX
        n = min(_PMAX, rows - r0)
        tile = nl.ndarray((_PMAX, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile[0:n, :], src=x[r0 : r0 + n, :])
        nisa.dma_copy(dst=staged[r0 : r0 + n, :], src=tile[0:n, :])
    cur, cur_rows, cur_w = staged, rows, width

    for hop in range(len(hop_meshes)):
        groups = hop_meshes[hop]
        g = len(groups[0])
        kernel_assert(cur_rows % g == 0, f"hop {hop}: rows ({cur_rows}) must be divisible by group size ({g})")
        out_rows = cur_rows // g

        # all_to_all on HBM forces collective_dim=0, so it splits AND concatenates on rows:
        # peer p's contribution lands at rows [p*out_rows, (p+1)*out_rows).
        recv = nl.ndarray((cur_rows, cur_w), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        ncc.all_to_all(dsts=[recv], srcs=[cur], replica_group=ncc.ReplicaGroup(groups), collective_dim=0)

        # De-interleave rows -> columns so peer p's block becomes columns [p*cur_w, (p+1)*cur_w).
        # Ascending peer order is what preserves the global vocab ordering across hops.
        wide = nl.ndarray((out_rows, g * cur_w), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        for p in range(g):
            nisa.dma_copy(dst=wide[:, nl.ds(p * cur_w, cur_w)], src=recv[nl.ds(p * out_rows, out_rows), :])
        cur, cur_rows, cur_w = wide, out_rows, g * cur_w

    return cur


@nki.jit
def allgather_probe(x: nl.NkiTensor, groups: Tuple = ()) -> nl.NkiTensor:
    """One all_gather over ``groups`` (a single mesh). Returns ``[g*rows, w]`` bfloat16.

    Asks whether the >=64-participant limit that stops ``all_to_all`` is specific to
    all-to-all or applies to any mesh of that size. all_gather is also the only other way to
    put the full vocab on a rank, so its viability matters independently -- though note it
    hands every rank EVERY batch row, and extracting just this rank's rows needs a
    rank-dependent slice, which a single shared NEFF cannot express.
    """
    kernel_assert(len(groups) >= 1, "a replica group is required")
    rows, width = x.shape
    g = len(groups[0])

    # As in collective_probe: a collective may not read a kernel IO tensor.
    staged = nl.ndarray((rows, width), dtype=nl.bfloat16, buffer=nl.private_hbm)
    # ceil, not floor. `rows // _PMAX` leaves rows [floor*_PMAX, rows) NEVER WRITTEN whenever
    # rows > _PMAX and rows % _PMAX != 0 (rows=200 -> one tile, 72 rows unwritten), yet the
    # collective below reads ALL of `staged` -- so those rows carry whatever was last in that HBM
    # page and the probe silently reports a mesh verdict computed from garbage. Not hit today only
    # because every probe gbs is <= 128 or a multiple of 128; the all_gather copy-out below always
    # used ceil, which is what makes this a slip rather than a constraint.
    for t in range((rows + _PMAX - 1) // _PMAX):
        r0 = t * _PMAX
        n = min(_PMAX, rows - r0)
        tile = nl.ndarray((_PMAX, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile[0:n, :], src=x[r0 : r0 + n, :])
        nisa.dma_copy(dst=staged[r0 : r0 + n, :], src=tile[0:n, :])

    # A collective may not WRITE a kernel IO tensor either (the mirror of the read rule), so
    # it lands in an internal buffer that is then copied to the output.
    recv = nl.ndarray((g * rows, width), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    ncc.all_gather(dsts=[recv], srcs=[staged], replica_group=ncc.ReplicaGroup(groups), collective_dim=0)

    out = nl.ndarray((g * rows, width), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for t in range((g * rows + _PMAX - 1) // _PMAX):
        r0 = t * _PMAX
        n = min(_PMAX, g * rows - r0)
        tile = nl.ndarray((_PMAX, width), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=tile[0:n, :], src=recv[r0 : r0 + n, :])
        nisa.dma_copy(dst=out[r0 : r0 + n, :], src=tile[0:n, :])
    return out
