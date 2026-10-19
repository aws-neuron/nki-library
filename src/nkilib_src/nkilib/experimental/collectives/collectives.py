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
"""Basic collective operation kernels: all_reduce, all_gather, reduce_scatter, all_to_all, rank_id."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup


@nki.jit
def all_reduce_hbm_kernel(input: nl.NkiTensor, replica_group: ReplicaGroup) -> nl.NkiTensor:
    """Sum tensors across all ranks.

    Example with replica_group=[[0,1]], input shape (2, 3) -> output shape (2, 3):
      rank0: [[1,2,3], [4,5,6]] -> [[2,4,6], [8,10,12]]
      rank1: [[1,2,3], [4,5,6]] -> [[2,4,6], [8,10,12]]
    """
    # Keep both operands private_hbm for compatibility with old same-buffer validation and the
    # newer LNC1 rule that collective sources must not use shared_hbm.
    # name= is required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it.
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.private_hbm, name="src")
    dst = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.private_hbm, name="dst")
    out = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=replica_group)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def all_gather_hbm_kernel(input: nl.NkiTensor, replica_group: ReplicaGroup, num_ranks: int) -> nl.NkiTensor:
    """Gather tensors from all ranks along dim 0.

    Example with replica_group=[[0,1]], input shape (2, 3) -> output shape (4, 3):
      rank0: [[1,2,3], [4,5,6]]       -> [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
      rank1: [[7,8,9], [10,11,12]]    -> [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    """
    H, W = input.shape
    # Keep both operands private_hbm for compatibility with old same-buffer validation and the
    # newer LNC1 rule that collective sources must not use shared_hbm.
    # name= is required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it.
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.private_hbm, name="src")
    dst = nl.ndarray((H * num_ranks, W), dtype=input.dtype, buffer=nl.private_hbm, name="dst")
    out = nl.ndarray((H * num_ranks, W), dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.all_gather(dsts=[dst], srcs=[src], replica_group=replica_group, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def reduce_scatter_hbm_kernel(input: nl.NkiTensor, replica_group: ReplicaGroup, num_ranks: int) -> nl.NkiTensor:
    """Sum then scatter chunks along dim 0. Dim 0 is split into num_ranks chunks.

    Example with replica_group=[[0,1]], input shape (4, 3) -> output shape (2, 3):
      rank0: [[1,1,1], [2,2,2], [3,3,3], [4,4,4]] -> [[2,2,2], [4,4,4]]   (sum of inputs[:2,:])
      rank1: [[1,1,1], [2,2,2], [3,3,3], [4,4,4]] -> [[6,6,6], [8,8,8]]   (sum of inputs[2:,:])
    """
    H, W = input.shape
    # Keep both operands private_hbm for compatibility with old same-buffer validation and the
    # newer LNC1 rule that collective sources must not use shared_hbm.
    # name= is required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it.
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.private_hbm, name="src")
    dst = nl.ndarray((H // num_ranks, W), dtype=input.dtype, buffer=nl.private_hbm, name="dst")
    out = nl.ndarray((H // num_ranks, W), dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.reduce_scatter(dsts=[dst], srcs=[src], op=nl.add, replica_group=replica_group, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def all_to_all_hbm_kernel(input: nl.NkiTensor, replica_group: ReplicaGroup) -> nl.NkiTensor:
    """Exchange chunks across ranks along dim 0. Each rank sends input[i,:] to rank[i].

    Example with replica_group=[[0,1]], input shape (2, 3) -> output shape (2, 3):
      rank0: [[1,2,3], [4,5,6]] -> [[1,2,3], [7,8,9]]     (keeps input[0,:], gets rank1's input[0,:])
      rank1: [[7,8,9], [10,11,12]] -> [[4,5,6], [10,11,12]] (gets rank0's input[1,:], keeps input[1,:])
    """
    # Keep both operands private_hbm for compatibility with old same-buffer validation and the
    # newer LNC1 rule that collective sources must not use shared_hbm.
    # name= is required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it.
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.private_hbm, name="src")
    dst = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.private_hbm, name="dst")
    out = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.all_to_all(dsts=[dst], srcs=[src], replica_group=replica_group, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def rank_id_kernel(in_tensor: nl.NkiTensor) -> nl.NkiTensor:
    """Select per-rank slice using rank_id as scalar_offset.

    Example with 2 ranks, input shape (2, 2, 3) -> output shape (2, 3):
      in_tensor = [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]  (same for all ranks)
      rank0: rank_id=0 -> in_tensor[0] (scalar_offset=0) -> out = [[1,2,3], [4,5,6]]
      rank1: rank_id=1 -> in_tensor[1] (scalar_offset=1) -> out = [[7,8,9], [10,11,12]]
    """
    _, H, W = in_tensor.shape
    out = nl.ndarray((H, W), dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    rank_id = ncc.rank_id()
    nisa.dma_copy(
        dst=out,
        src=in_tensor.ap(pattern=[[W, H], [1, W]], scalar_offset=rank_id, indirect_dim=0),
    )
    return out


@nki.jit
def dma_copy_rank_id_kernel(in_tensor: nl.NkiTensor, rank_id_lookup: nl.NkiTensor) -> nl.NkiTensor:
    """Load rank_id into SBUF via lookup table, then use as scalar_offset.

    ncc.rank_id() returns a value in a register. Due to currently unsupported
    register_store, we cannot save it directly to SBUF. Instead, we use a lookup
    table with identity mapping [[0,1,...]] and scalar_offset to load the value.

    Example with 2 ranks, input shape (2, 2, 3) -> output shape (2, 3):
      rank_id_lookup = [[0, 1]]  # identity mapping: rank_id -> rank_id
      in_tensor = [[[1,2,3], [4,5,6]], [[7,8,9], [10,11,12]]]  (same for all ranks)
      rank0: rank_id=0 -> rank_id_lookup[0]=0 -> in_tensor[0] -> out = [[1,2,3], [4,5,6]]
      rank1: rank_id=1 -> rank_id_lookup[1]=1 -> in_tensor[1] -> out = [[7,8,9], [10,11,12]]
    """
    _, H, W = in_tensor.shape
    out = nl.ndarray([H, W], dtype=in_tensor.dtype, buffer=nl.shared_hbm)
    rank_id = ncc.rank_id()
    rank_id_sb = nl.ndarray([1, 1], dtype=nl.int32, buffer=nl.sbuf)
    # Load rank_id value into SBUF using lookup table
    nisa.dma_copy(
        dst=rank_id_sb, src=rank_id_lookup.ap(pattern=[[1, 1], [1, 1]], scalar_offset=rank_id, indirect_dim=1)
    )
    # Use SBUF value as scalar_offset for data access
    nisa.dma_copy(
        dst=out,
        src=in_tensor.ap(pattern=[[W, H], [1, W]], offset=0, scalar_offset=rank_id_sb, indirect_dim=0),
    )
    return out
