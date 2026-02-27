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
"""
Tests for NKI collective operations.
"""

from enum import Enum
from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    KernelArgs,
    LazyGoldenGenerator,
    PerRankLazyGoldenGenerator,
    PerRankLazyInputGenerator,
    ValidationArgs,
)
from test.utils.test_orchestrator import Orchestrator
from typing import Optional

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.core.utils.tensor_view import TensorView

# ==================== Basic Collective Kernels ====================
# Test fundamental collective operations: all_reduce, all_gather, reduce_scatter, all_to_all


@nki.jit
def all_reduce_hbm_kernel(input: nl.ndarray, replica_group: ReplicaGroup) -> nl.ndarray:
    """Sum tensors across all ranks.

    Example with replica_group=[[0,1]], input shape (2, 3) -> output shape (2, 3):
      rank0: [[1,2,3], [4,5,6]] -> [[2,4,6], [8,10,12]]
      rank1: [[1,2,3], [4,5,6]] -> [[2,4,6], [8,10,12]]
    """
    # name= required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.all_reduce(dsts=[dst], srcs=[src], op=nl.add, replica_group=replica_group)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def all_gather_hbm_kernel(input: nl.ndarray, replica_group: ReplicaGroup, num_ranks: int) -> nl.ndarray:
    """Gather tensors from all ranks along dim 0.

    Example with replica_group=[[0,1]], input shape (2, 3) -> output shape (4, 3):
      rank0: [[1,2,3], [4,5,6]]       -> [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
      rank1: [[7,8,9], [10,11,12]]    -> [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]
    """
    H, W = input.shape
    # name= required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray((H * num_ranks, W), dtype=input.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray((H * num_ranks, W), dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.all_gather(dsts=[dst], srcs=[src], replica_group=replica_group, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def reduce_scatter_hbm_kernel(input: nl.ndarray, replica_group: ReplicaGroup, num_ranks: int) -> nl.ndarray:
    """Sum then scatter chunks along dim 0. Dim 0 is split into num_ranks chunks.

    Example with replica_group=[[0,1]], input shape (4, 3) -> output shape (2, 3):
      rank0: [[1,1,1], [2,2,2], [3,3,3], [4,4,4]] -> [[2,2,2], [4,4,4]]   (sum of inputs[:2,:])
      rank1: [[1,1,1], [2,2,2], [3,3,3], [4,4,4]] -> [[6,6,6], [8,8,8]]   (sum of inputs[2:,:])
    """
    H, W = input.shape
    # name= required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray((H // num_ranks, W), dtype=input.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray((H // num_ranks, W), dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.reduce_scatter(dsts=[dst], srcs=[src], op=nl.add, replica_group=replica_group, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out


@nki.jit
def all_to_all_hbm_kernel(input: nl.ndarray, replica_group: ReplicaGroup) -> nl.ndarray:
    """Exchange chunks across ranks along dim 0. Each rank sends input[i,:] to rank[i].

    Example with replica_group=[[0,1]], input shape (2, 3) -> output shape (2, 3):
      rank0: [[1,2,3], [4,5,6]] -> [[1,2,3], [7,8,9]]     (keeps input[0,:], gets rank1's input[0,:])
      rank1: [[7,8,9], [10,11,12]] -> [[4,5,6], [10,11,12]] (gets rank0's input[1,:], keeps input[1,:])
    """
    # name= required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="src")
    dst = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="dst")
    out = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm)
    # dma_copy required: Collective instruction cannot read/write IO tensors
    nisa.dma_copy(dst=src, src=input)
    ncc.all_to_all(dsts=[dst], srcs=[src], replica_group=replica_group, collective_dim=0)
    nisa.dma_copy(dst=out, src=dst)
    return out


# ==================== rank_id + Indirect DMA Kernels ====================
# Test ncc.rank_id() with scalar_offset for per-rank data selection


@nki.jit
def rank_id_kernel(in_tensor: nl.ndarray) -> nl.ndarray:
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
def dma_copy_rank_id_kernel(in_tensor: nl.ndarray, rank_id_lookup: nl.ndarray) -> nl.ndarray:
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


# ==================== Batch Shard Kernels ====================
# Test TP64 -> TP8DP8 transitions for batch sharding attention


class AttnQBatchShardLayout(Enum):
    """Layout for QKV batch shard kernel output."""

    NBSd = 0  # (N_heads, B, S, d)
    dBnS = 1  # (d, B, n_heads, S)


@nki.jit
def attn_q_batch_shard(
    input: nl.ndarray,
    iota_workers: nl.ndarray,
    gathered_buf: nl.ndarray,
    gqa_group_size: int,
    replica_group: ReplicaGroup,
    layout: AttnQBatchShardLayout = AttnQBatchShardLayout.NBSd,
    rank_id_in: Optional[nl.ndarray] = None,
) -> nl.ndarray:
    """QKV batch shard kernel: all_gather on dim 0 + reshape to separate gqa_group_size heads + slice batch.

    Implements the Q projection transition from TP64 to TP8DP8 for batch sharding attention.
    Each rank starts with n Q heads for all B batches, ends with gqa_group_size*n Q heads for B/gqa_group_size batches.

    The pattern works when either: (1) heads are in dim=0 (NBSd), or (2) n=1 (any layout).
    all_gather on dim 0, reshape to (G, dim0, ...), slice batch (where G = gqa_group_size):

        Input (per rank)          all_gather dim 0           reshape                    slice batch              reshape back
        ────────────────          ────────────────           ───────                    ───────────              ────────────
        (n, B, S, d)         ->   (G*n, B, S, d)        ->   (G, n, B, S, d)       ->   (G, n, B/G, S, d)   ->   (G*n, B/G, S, d)
        (d, B, n=1, S)       ->   (G*d, B, n=1, S)      ->   (G, d, B, n=1, S)     ->   (G, d, B/G, n=1, S) ->   (n=G, d, B/G, S)

    Example with G=8, n=1, B=32, S=1, d=64:

        NBSd: (1,32,1,64) -> (8,32,1,64) -> (8,1,32,1,64) -> (8,1,4,1,64) -> (8,4,1,64)
        dBnS: (64,32,1,1) -> (512,32,1,1) -> (8,64,32,1,1) -> (8,64,4,1,1) -> (8,64,4,1)

    Args:
        x_in: Input Q tensor from TP64 projection. First dim is gathered across gqa_group_size ranks.
              NBSd: (n_heads, B, S, d)
              dBnS: (d, B, n_heads, S)
        iota_workers: Lookup table mapping rank_id -> batch_offset for scalar_offset DMA.
                      Shape: (1, collective_ranks), values: [(r % gqa_group_size) * B_per_rank for r in range(collective_ranks)]
                      Needed because NKI compiler doesn't support arithmetic on rank_id.
        gathered_buf: Workspace buffer for all_gather result (must be input tensor for scalar_offset)
        gqa_group_size: GQA group size (e.g., 8 for TP8DP8, 2 for TP2DP2)
        replica_group: ReplicaGroup defining the collective topology
        layout: Output layout - NBSd or dBnS
        rank_id_in: Optional rank_id as input tensor (1,1) int32. If None, uses ncc.rank_id().

    Returns:
        Output Q tensor with gqa_group_size*n heads for this rank's batch slice (4D, same layout as input)
        NBSd: (gqa_group_size*n_heads, B/gqa_group_size, S, d)
        dBnS: (n=gqa_group_size, d, B/gqa_group_size, S)
    """

    # Get rank_id
    if rank_id_in is not None:
        rank_id_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(dst=rank_id_sb, src=rank_id_in)
        rank_id = rank_id_sb
    else:
        rank_id = ncc.rank_id()

    # Layout-specific shapes and rearrange patterns
    G = gqa_group_size
    if layout == AttnQBatchShardLayout.NBSd:
        n, B, S, d = input.shape
        gathered_shape = (G * n, B, S, d)
        rearrange_src = (('G', 'n'), 'B', 'S', 'd')
        rearrange_dst = ('G', 'n', 'B', 'S', 'd')
        final_shape = (G * n, B // G, S, d)
    else:  # dBnS (requires n=1)
        d, B, n, S = input.shape
        assert n == 1, f"dBnS layout requires n=1, got n={n}"
        gathered_shape = (G * d, B, n, S)
        rearrange_src = (('G', 'd'), 'B', 'n', 'S')
        rearrange_dst = ('G', 'd', 'B', 'n', 'S')
        final_shape = (G, d, B // G, S)  # G becomes the new n dimension

    B_per_rank = B // G

    # WORKAROUND: Copy input to shared_hbm because all_gather requires src in shared_hbm.
    # Using x_in directly causes compiler error:
    #   "Error from .../inst_visitor.cpp:3377 in function 'checkCollective'"
    # name= required on collective src/dst: NCC_IBIR440 DRAM allocation failure without it
    src = nl.ndarray(input.shape, dtype=input.dtype, buffer=nl.shared_hbm, name="src")
    nisa.dma_copy(dst=src, src=input)

    # all_gather on dim 0
    gathered = nl.ndarray(gathered_shape, dtype=input.dtype, buffer=nl.shared_hbm, name="dst")
    ncc.all_gather(dsts=[gathered], srcs=[src], replica_group=replica_group, collective_dim=0)

    # reshape to separate G: (G*dim0, ...) -> (G, dim0, ...)
    gathered_view = TensorView(gathered).rearrange(rearrange_src, rearrange_dst, {'G': G})

    # WORKAROUND: Copy to input buffer because scalar_offset fails on internal tensors.
    # Using gathered directly causes compiler error:
    #   "Assertion `tensorId >= 0 && "Request tensorId must >= 0"' failed"
    # gathered_buf must be passed as kernel input for scalar_offset to work.
    nisa.dma_copy(dst=gathered_buf, src=gathered_view.get_view())

    # WORKAROUND: Convert rank_id to batch_offset using iota lookup table.
    # NKI compiler doesn't support arithmetic on rank_id (e.g., rank_id % gqa_group_size * B_per_rank),
    # so we pre-compute the mapping and use scalar_offset to look it up.
    # Error without workaround: "unimplemented operator 'mod'" or "unimplemented operator 'mul'"
    batch_offset_sb = nl.ndarray([1, 1], dtype=nl.int32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=batch_offset_sb, src=iota_workers.ap(pattern=[[1, 1], [1, 1]], scalar_offset=rank_id, indirect_dim=1)
    )

    # Extract this rank's batch slice using dynamic offset
    slice_view = TensorView(gathered_buf).slice(dim=2, start=0, end=B_per_rank)
    slice_pattern, slice_offset = slice_view._get_pattern_and_offset()
    q_out = nl.ndarray(slice_view.shape, dtype=input.dtype, buffer=nl.shared_hbm)
    nisa.dma_copy(
        dst=q_out,
        src=gathered_buf.ap(pattern=slice_pattern, offset=slice_offset, scalar_offset=batch_offset_sb, indirect_dim=2),
    )

    # Reshape back to 4D - just a view, no copy
    return q_out.reshape(final_shape)


# ==================== Test Class ====================

from test.utils.pytest_test_metadata import pytest_test_metadata
from typing import final


@pytest_test_metadata(
    name="Collectives",
    pytest_marks=["collectives"],
)
@final
class TestCollectives:
    """Test collective operations on multi-chip hardware."""

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "collective_ranks,logical_nc_config",
        [(2, 2), (2, 1)],
        ids=["2ranks_lnc2", "2ranks_lnc1"],
    )
    def test_all_reduce(self, test_manager: Orchestrator, collective_ranks: int, logical_nc_config: int):
        """Test all_reduce with determinism check (same input all ranks)."""
        np.random.seed(42)
        x_in = np.random.randn(128, 512).astype(np.float32)
        replica_group = ReplicaGroup([list(range(collective_ranks))])
        test_manager.execute(
            KernelArgs(
                kernel_func=all_reduce_hbm_kernel,
                compiler_input=CompilerArgs(logical_nc_config=logical_nc_config),
                kernel_input={"input": x_in, "replica_group": replica_group},
                inference_args=InferenceArgs(
                    collective_ranks=collective_ranks, enable_determinism_check=True, num_runs=10
                ),
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray={"out": x_in * collective_ranks}),
                    relative_accuracy=1e-3,
                    absolute_accuracy=1e-3,
                ),
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "collective_ranks,logical_nc_config",
        [(2, 2), (2, 1)],
        ids=["2ranks_lnc2", "2ranks_lnc1"],
    )
    def test_all_gather(self, test_manager: Orchestrator, collective_ranks: int, logical_nc_config: int):
        """Test all_gather with per-rank inputs and outputs."""
        np.random.seed(42)
        H, W = 128, 512
        # Each rank has different input data
        x_global = np.random.randn(collective_ranks, H, W).astype(np.float32)
        replica_group = ReplicaGroup([list(range(collective_ranks))])

        def create_inputs(rank_id: int):
            return {"input": x_global[rank_id], "replica_group": replica_group, "num_ranks": collective_ranks}

        def create_golden(rank_id: int):
            # all_gather concatenates all ranks' data
            return {"out": x_global.reshape(collective_ranks * H, W)}

        test_manager.execute(
            KernelArgs(
                kernel_func=all_gather_hbm_kernel,
                compiler_input=CompilerArgs(logical_nc_config=logical_nc_config),
                kernel_input=PerRankLazyInputGenerator(create_inputs),
                inference_args=InferenceArgs(collective_ranks=collective_ranks),
                validation_args=ValidationArgs(
                    golden_output=PerRankLazyGoldenGenerator(create_golden),
                    relative_accuracy=1e-3,
                    absolute_accuracy=1e-3,
                ),
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "collective_ranks,logical_nc_config",
        [(2, 2), (2, 1)],
        ids=["2ranks_lnc2", "2ranks_lnc1"],
    )
    def test_reduce_scatter(self, test_manager: Orchestrator, collective_ranks: int, logical_nc_config: int):
        """Test reduce_scatter with per-rank inputs and outputs."""
        np.random.seed(42)
        H, W = 128 * collective_ranks, 512
        chunk_size = H // collective_ranks
        # Each rank has different input (will be summed then scattered)
        x_global = np.random.randn(collective_ranks, H, W).astype(np.float32)
        replica_group = ReplicaGroup([list(range(collective_ranks))])

        def create_inputs(rank_id: int):
            return {"input": x_global[rank_id], "replica_group": replica_group, "num_ranks": collective_ranks}

        def create_golden(rank_id: int):
            # reduce_scatter: sum all inputs, then each rank gets its chunk
            summed = x_global.sum(axis=0)
            return {"out": summed[rank_id * chunk_size : (rank_id + 1) * chunk_size, :]}

        test_manager.execute(
            KernelArgs(
                kernel_func=reduce_scatter_hbm_kernel,
                compiler_input=CompilerArgs(logical_nc_config=logical_nc_config),
                kernel_input=PerRankLazyInputGenerator(create_inputs),
                inference_args=InferenceArgs(collective_ranks=collective_ranks),
                validation_args=ValidationArgs(
                    golden_output=PerRankLazyGoldenGenerator(create_golden),
                    relative_accuracy=1e-3,
                    absolute_accuracy=1e-3,
                ),
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "collective_ranks,logical_nc_config",
        # all_to_all requires Mesh algorithm: 4 ranks lnc2 or 8 ranks lnc1
        [(4, 2), (8, 1)],
        ids=["4ranks_lnc2", "8ranks_lnc1"],
    )
    def test_all_to_all(self, test_manager: Orchestrator, collective_ranks: int, logical_nc_config: int):
        """Test all_to_all (same input all ranks)."""
        np.random.seed(42)
        H, W = 128 * collective_ranks, 512
        x_in = np.random.randn(H, W).astype(np.float32)
        chunk_size = H // collective_ranks
        replica_group = ReplicaGroup([list(range(collective_ranks))])
        # all_to_all with same input: each rank gets chunk i from all ranks = tiled chunk 0
        golden = np.tile(x_in[:chunk_size, :], (collective_ranks, 1))
        test_manager.execute(
            KernelArgs(
                kernel_func=all_to_all_hbm_kernel,
                compiler_input=CompilerArgs(logical_nc_config=logical_nc_config),
                kernel_input={"input": x_in, "replica_group": replica_group},
                inference_args=InferenceArgs(collective_ranks=collective_ranks),
                validation_args=ValidationArgs(
                    golden_output=LazyGoldenGenerator(output_ndarray={"out": golden}),
                    relative_accuracy=1e-3,
                    absolute_accuracy=1e-3,
                ),
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "collective_ranks,logical_nc_config",
        [(2, 2), (2, 1)],
        ids=["2ranks_lnc2", "2ranks_lnc1"],
    )
    def test_rank_id(self, test_manager: Orchestrator, collective_ranks: int, logical_nc_config: int):
        """Test ncc.rank_id() as scalar_offset: each rank selects its slice."""
        np.random.seed(42)
        G, H, W = collective_ranks, 128, 512
        in_tensor = np.random.randn(G, H, W).astype(np.float32)

        def create_golden(rank_id: int):
            return {"out": in_tensor[rank_id]}

        test_manager.execute(
            KernelArgs(
                kernel_func=rank_id_kernel,
                compiler_input=CompilerArgs(logical_nc_config=logical_nc_config),
                kernel_input={"in_tensor": in_tensor},
                inference_args=InferenceArgs(collective_ranks=collective_ranks),
                validation_args=ValidationArgs(
                    golden_output=PerRankLazyGoldenGenerator(create_golden),
                    relative_accuracy=1e-3,
                    absolute_accuracy=1e-3,
                ),
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize(
        "collective_ranks,logical_nc_config",
        [(2, 2), (2, 1)],
        ids=["2ranks_lnc2", "2ranks_lnc1"],
    )
    def test_dma_copy_rank_id(self, test_manager: Orchestrator, collective_ranks: int, logical_nc_config: int):
        """Test rank_id loaded to SBUF via lookup table, then used as scalar_offset."""
        np.random.seed(42)
        G, H, W = collective_ranks, 128, 64
        in_tensor = np.random.randn(G, H, W).astype(np.float32)
        rank_id_lookup = np.arange(G, dtype=np.int32).reshape(1, G)

        def create_golden(rank_id: int):
            return {"out": in_tensor[rank_id]}

        test_manager.execute(
            KernelArgs(
                kernel_func=dma_copy_rank_id_kernel,
                compiler_input=CompilerArgs(logical_nc_config=logical_nc_config),
                kernel_input={"in_tensor": in_tensor, "rank_id_lookup": rank_id_lookup},
                inference_args=InferenceArgs(collective_ranks=collective_ranks),
                validation_args=ValidationArgs(
                    golden_output=PerRankLazyGoldenGenerator(create_golden),
                    relative_accuracy=1e-3,
                    absolute_accuracy=1e-3,
                ),
            )
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("layout", [AttnQBatchShardLayout.NBSd, AttnQBatchShardLayout.dBnS], ids=["NBSd", "dBnS"])
    @pytest.mark.parametrize("use_input_rank", [False, True], ids=["ncc_rank", "input_rank"])
    @pytest.mark.parametrize(
        "collective_ranks,logical_nc_config,gqa_group_size,q_heads,batch",
        [
            # TP4 -> TP2DP2 for shared fleet (trn2.3xlarge with 4 LNC2 cores)
            (4, 2, 2, 1, 8),
            # TP64 -> TP8DP8: Disabled by default - no trn2.48xlarge instances in shared fleet
            # (64, 2, 8, 1, 32),
        ],
        ids=["TP2DP2"],
    )
    def test_batch_shard_input(
        self,
        test_manager: Orchestrator,
        collective_ranks: int,
        logical_nc_config: int,
        layout: AttnQBatchShardLayout,
        use_input_rank: bool,
        gqa_group_size: int,
        q_heads: int,
        batch: int,
    ):
        """Test QKV batch shard: all_gather on heads + rank_id slice on batch."""
        np.random.seed(42)
        S_tkg, d_head = 1, 64
        batch_per_rank = batch // gqa_group_size

        # Create replica group dynamically: DP groups of gqa_group_size ranks each
        num_dp_groups = collective_ranks // gqa_group_size
        replica_group = ReplicaGroup(
            [[i * gqa_group_size + j for j in range(gqa_group_size)] for i in range(num_dp_groups)]
        )

        is_nbsd = layout == AttnQBatchShardLayout.NBSd
        # Q_global: all heads across all ranks
        Q_global = (
            np.random.randn(collective_ranks, batch, S_tkg, d_head).astype(np.float32)
            if is_nbsd
            else np.random.randn(collective_ranks, d_head, batch, S_tkg).astype(np.float32)
        )
        # gathered_buf is 5D after reshape:
        #   (gqa_group_size, q_heads, batch, S_tkg, d_head) or
        #   (gqa_group_size, d_head, batch, q_heads, S_tkg)
        gathered_buf_shape = (
            (gqa_group_size, q_heads, batch, S_tkg, d_head)
            if is_nbsd
            else (gqa_group_size, d_head, batch, q_heads, S_tkg)
        )

        def create_inputs(rank_id: int):
            x_in = (
                Q_global[rank_id : rank_id + 1]
                if is_nbsd
                else Q_global[rank_id : rank_id + 1].reshape(d_head, batch, q_heads, S_tkg)
            )
            inputs = {
                "input": x_in,
                "iota_workers": np.array(
                    [(r % gqa_group_size) * batch_per_rank for r in range(collective_ranks)], dtype=np.int32
                ).reshape(1, collective_ranks),
                "gathered_buf": np.zeros(gathered_buf_shape, dtype=np.float32),
                "gqa_group_size": gqa_group_size,
                "replica_group": replica_group,
                "layout": layout,
            }
            if use_input_rank:
                inputs["rank_id_in"] = np.array([[rank_id]], dtype=np.int32)
            return inputs

        def create_golden(rank_id: int):
            gqa_group, rank_in_group = rank_id // gqa_group_size, rank_id % gqa_group_size
            h0, h1 = gqa_group * gqa_group_size, (gqa_group + 1) * gqa_group_size
            b0, b1 = rank_in_group * batch_per_rank, (rank_in_group + 1) * batch_per_rank
            golden = Q_global[h0:h1, b0:b1, :, :] if is_nbsd else Q_global[h0:h1, :, b0:b1, :]
            return {"q_out": golden}

        test_manager.execute(
            KernelArgs(
                kernel_func=attn_q_batch_shard,
                compiler_input=CompilerArgs(logical_nc_config=logical_nc_config),
                kernel_input=PerRankLazyInputGenerator(create_inputs),
                inference_args=InferenceArgs(collective_ranks=collective_ranks),
                validation_args=ValidationArgs(
                    golden_output=PerRankLazyGoldenGenerator(create_golden),
                    relative_accuracy=1e-3,
                    absolute_accuracy=1e-3,
                ),
            )
        )
