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

from typing import final

import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.batch_shard import (
    AttnQBatchShardLayout,
    attn_q_batch_shard,
)
from nkilib_src.nkilib.experimental.collectives.batch_shard_torch import (
    attn_q_batch_shard_torch_ref,
)
from nkilib_src.nkilib.experimental.collectives.collectives import (
    all_gather_hbm_kernel,
    all_reduce_hbm_kernel,
    all_to_all_hbm_kernel,
    dma_copy_rank_id_kernel,
    rank_id_kernel,
    reduce_scatter_hbm_kernel,
)
from nkilib_src.nkilib.experimental.collectives.collectives_torch import (
    all_gather_hbm_torch_ref,
    all_reduce_hbm_torch_ref,
    all_to_all_hbm_torch_ref,
    dma_copy_rank_id_torch_ref,
    rank_id_torch_ref,
    reduce_scatter_hbm_torch_ref,
)

from test.utils.common_dataclasses import (
    CompilerArgs,
    InferenceArgs,
    Platforms,
)
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework

# ==================== Test Class ====================

RANKS_LNC_PARAM_NAMES = "collective_ranks, logical_nc_config"
RANKS_LNC_2RANK = [(2, 2), (2, 1)]
# all_to_all requires Mesh algorithm: 4 ranks lnc2 or 8 ranks lnc1
RANKS_LNC_A2A = [(4, 2), (8, 1)]
_RANKS_LNC_ABBREVS = {"collective_ranks": "ranks", "logical_nc_config": "lnc"}


@pytest_test_metadata(name="Collectives")
@pytest_marks(["collectives"])
@final
@pytest.mark.skip_simulation
@pytest.mark.high_rank
class TestCollectives:
    """Test collective operations on multi-chip hardware."""

    @pytest.mark.fast
    @pytest_parametrize(RANKS_LNC_PARAM_NAMES, RANKS_LNC_2RANK, abbrevs=_RANKS_LNC_ABBREVS)
    def test_all_reduce(
        self, test_manager: Orchestrator, platform_target: Platforms, collective_ranks: int, logical_nc_config: int
    ):
        """Test all_reduce with determinism check (same input all ranks)."""
        np.random.seed(42)
        x_in = np.random.randn(128, 512).astype(np.float32)
        replica_group = ReplicaGroup([list(range(collective_ranks))])

        def create_inputs(rank_id: int):
            return {"input": x_in, "replica_group": replica_group}

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=all_reduce_hbm_kernel,
            torch_ref=all_reduce_hbm_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=logical_nc_config, platform_target=platform_target),
            output_keys=["out"],
            rtol=1e-3,
            atol=1e-3,
            inference_args=InferenceArgs(collective_ranks=collective_ranks, enable_determinism_check=True, num_runs=10),
        )

    @pytest.mark.fast
    @pytest_parametrize(RANKS_LNC_PARAM_NAMES, RANKS_LNC_2RANK, abbrevs=_RANKS_LNC_ABBREVS)
    def test_all_gather(
        self, test_manager: Orchestrator, platform_target: Platforms, collective_ranks: int, logical_nc_config: int
    ):
        """Test all_gather with per-rank inputs and outputs."""
        np.random.seed(42)
        H, W = 128, 512
        # Each rank has different input data
        x_global = np.random.randn(collective_ranks, H, W).astype(np.float32)
        replica_group = ReplicaGroup([list(range(collective_ranks))])

        def create_inputs(rank_id: int):
            return {"input": x_global[rank_id], "replica_group": replica_group, "num_ranks": collective_ranks}

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=all_gather_hbm_kernel,
            torch_ref=all_gather_hbm_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=logical_nc_config, platform_target=platform_target),
            output_keys=["out"],
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.fast
    @pytest_parametrize(RANKS_LNC_PARAM_NAMES, RANKS_LNC_2RANK, abbrevs=_RANKS_LNC_ABBREVS)
    def test_reduce_scatter(
        self, test_manager: Orchestrator, platform_target: Platforms, collective_ranks: int, logical_nc_config: int
    ):
        """Test reduce_scatter with per-rank inputs and outputs."""
        np.random.seed(42)
        H, W = 128 * collective_ranks, 512
        # Each rank has different input (will be summed then scattered)
        x_global = np.random.randn(collective_ranks, H, W).astype(np.float32)
        replica_group = ReplicaGroup([list(range(collective_ranks))])

        def create_inputs(rank_id: int):
            return {"input": x_global[rank_id], "replica_group": replica_group, "num_ranks": collective_ranks}

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=reduce_scatter_hbm_kernel,
            torch_ref=reduce_scatter_hbm_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=logical_nc_config, platform_target=platform_target),
            output_keys=["out"],
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.fast
    @pytest_parametrize(RANKS_LNC_PARAM_NAMES, RANKS_LNC_A2A, abbrevs=_RANKS_LNC_ABBREVS)
    def test_all_to_all(
        self, test_manager: Orchestrator, platform_target: Platforms, collective_ranks: int, logical_nc_config: int
    ):
        """Test all_to_all (same input all ranks)."""
        np.random.seed(42)
        H, W = 128 * collective_ranks, 512
        x_in = np.random.randn(H, W).astype(np.float32)
        replica_group = ReplicaGroup([list(range(collective_ranks))])

        def create_inputs(rank_id: int):
            return {"input": x_in, "replica_group": replica_group}

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=all_to_all_hbm_kernel,
            torch_ref=all_to_all_hbm_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=logical_nc_config, platform_target=platform_target),
            output_keys=["out"],
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.fast
    @pytest_parametrize(RANKS_LNC_PARAM_NAMES, RANKS_LNC_2RANK, abbrevs=_RANKS_LNC_ABBREVS)
    def test_rank_id(
        self, test_manager: Orchestrator, platform_target: Platforms, collective_ranks: int, logical_nc_config: int
    ):
        """Test ncc.rank_id() as scalar_offset: each rank selects its slice."""
        np.random.seed(42)
        G, H, W = collective_ranks, 128, 512
        in_tensor = np.random.randn(G, H, W).astype(np.float32)

        def create_inputs(rank_id: int):
            return {"in_tensor": in_tensor}

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=rank_id_kernel,
            torch_ref=rank_id_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=logical_nc_config, platform_target=platform_target),
            output_keys=["out"],
            rtol=1e-3,
            atol=1e-3,
        )

    @pytest.mark.fast
    @pytest_parametrize(RANKS_LNC_PARAM_NAMES, RANKS_LNC_2RANK, abbrevs=_RANKS_LNC_ABBREVS)
    def test_dma_copy_rank_id(
        self, test_manager: Orchestrator, platform_target: Platforms, collective_ranks: int, logical_nc_config: int
    ):
        """Test rank_id loaded to SBUF via lookup table, then used as scalar_offset."""
        np.random.seed(42)
        G, H, W = collective_ranks, 128, 64
        in_tensor = np.random.randn(G, H, W).astype(np.float32)
        rank_id_lookup = np.arange(G, dtype=np.int32).reshape(1, G)

        def create_inputs(rank_id: int):
            return {"in_tensor": in_tensor, "rank_id_lookup": rank_id_lookup}

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=dma_copy_rank_id_kernel,
            torch_ref=dma_copy_rank_id_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=logical_nc_config, platform_target=platform_target),
            output_keys=["out"],
            rtol=1e-3,
            atol=1e-3,
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
        platform_target: Platforms,
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

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=attn_q_batch_shard,
            torch_ref=attn_q_batch_shard_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=collective_ranks,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=logical_nc_config, platform_target=platform_target),
            output_keys=["q_out"],
            rtol=1e-3,
            atol=1e-3,
        )
