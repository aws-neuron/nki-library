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

"""Ring attention unpermute kernel: striped -> contiguous sequence reordering."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup


@nki.jit
def ring_attention_unpermute(
    x: nl.NkiTensor,
    replica_groups: tuple = None,
    num_workers: int = 1,
):
    """Unpermute striped ring attention output to contiguous sequence order.

    Each rank holds tokens at striped positions [rank, rank+nw, rank+2*nw, ...].
    After unpermute, each rank holds a contiguous chunk of the global sequence:
    rank 0 gets [0, spr), rank 1 gets [spr, 2*spr), etc.

    Algorithm: all_gather all ranks' data, then use strided DMA copies with
    rank-dependent offset to extract and interleave this rank's contiguous chunk.

    Requires spr % nw == 0 (seqlen_per_rank evenly divisible by num_workers).

    Args:
        x: [bs, seqlen_per_rank, d] — this rank's striped tokens (fp16/bf16).
        replica_groups: Replica group specification for collective communication.
        num_workers: Number of CP ranks in the ring.

    Returns:
        out_x: [bs, seqlen_per_rank, d] — this rank's contiguous chunk.
    """
    bs, spr, d = x.shape
    nw = num_workers
    chunk = spr // nw  # tokens from each source rank destined for this rank

    replica_group = ReplicaGroup(replica_groups)

    # Copy input to shared_hbm (collective src must be in shared_hbm)
    # name= required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it
    src = nl.ndarray((bs, spr, d), dtype=x.dtype, buffer=nl.shared_hbm, name="src")
    nisa.dma_copy(dst=src, src=x)

    # all_gather along dim 0: [bs, spr, d] per rank -> [nw*bs, spr, d]
    # After gather, gathered[w*bs:(w+1)*bs, :, :] contains rank w's striped data.
    gathered = nl.ndarray((nw * bs, spr, d), dtype=x.dtype, buffer=nl.shared_hbm, name="dst")
    ncc.all_gather(dsts=[gathered], srcs=[src], replica_group=replica_group, collective_dim=0)

    # Compute rank-dependent sequence offset for source pattern.
    # rank_id -> rank_id * chunk (the sequence offset within each source rank's data).
    # NKI doesn't support arithmetic on rank_id directly, so we store it to SBUF
    # via register_store and multiply by chunk using tensor_scalar.
    rank_id = ncc.rank_id()
    rank_int_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.register_store(dst=rank_int_sb, src=rank_id)

    rank_offset_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rank_offset_sb, data=rank_int_sb, op0=nl.multiply, operand0=chunk)

    out_x = nl.ndarray((bs, spr, d), dtype=x.dtype, buffer=nl.shared_hbm)

    # Strided rearrangement using DMA access patterns.
    #
    # Since spr % nw == 0, the mapping simplifies:
    #   out[b, j, :] = gathered[j%nw, b, my_rank*chunk + j//nw, :]
    #
    # For source rank w: out[:, w + k*nw, :] = gathered[w, :, my_rank*chunk + k, :]
    # where k = 0..chunk-1.
    #
    # Each iteration copies [bs, chunk, d] elements from source rank w's data
    # (at this rank's chunk offset) into strided output positions.
    for w in range(nw):
        # Source access pattern: read [bs, chunk, d] from gathered.
        # Flat layout of gathered is [nw*bs, spr, d] row-major.
        # We want gathered[w*bs + b, rank_offset + k, col] for b=0..bs-1, k=0..chunk-1, col=0..d-1.
        # Flat index: (w*bs + b)*spr*d + (rank_offset + k)*d + col
        #           = w*bs*spr*d + b*spr*d + rank_offset*d + k*d + col
        #
        # Pattern dims:
        #   dim0 (batch):     stride=spr*d, size=bs
        #   dim1 (seq chunk): stride=d,     size=chunk
        #   dim2 (hidden):    stride=1,     size=d
        # Static offset: w * bs * spr * d
        # Dynamic offset via scalar_offset on dim1: rank_offset_sb * d (stride of dim1)
        # rank_offset_sb = rank_id * chunk, so total dynamic = rank_id * chunk * d
        src_pat = [[spr * d, bs], [d, chunk], [1, d]]
        src_offset = w * bs * spr * d

        # Destination access pattern: write [bs, chunk, d] to output with stride nw.
        # out[b, w + k*nw, col] for b=0..bs-1, k=0..chunk-1, col=0..d-1.
        # Flat index: b*spr*d + (w + k*nw)*d + col = b*spr*d + w*d + k*nw*d + col
        #
        # Pattern dims:
        #   dim0 (batch):     stride=spr*d, size=bs
        #   dim1 (seq chunk): stride=nw*d,  size=chunk
        #   dim2 (hidden):    stride=1,     size=d
        # Static offset: w * d
        dst_pat = [[spr * d, bs], [nw * d, chunk], [1, d]]
        dst_offset = w * d

        nisa.dma_copy(
            dst=out_x.ap(pattern=dst_pat, offset=dst_offset),
            src=gathered.ap(pattern=src_pat, offset=src_offset, scalar_offset=rank_offset_sb, indirect_dim=1),
        )

    return out_x
