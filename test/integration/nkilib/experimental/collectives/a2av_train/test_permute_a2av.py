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
"""Integration tests for the capacity-bounded permute_a2av kernel."""

import ml_dtypes
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.a2av_train import permute_a2av
from nkilib_src.nkilib.experimental.collectives.a2av_train.permute_a2av_torch import (
    permute_a2av_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework


def _generate_routing(
    rows: int,
    ep_size: int,
    *,
    seed: int,
    same_on_every_rank: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate exact-sum routing permutations with skewed per-peer counts."""
    all_indices = []
    all_counts = []
    for rank in range(ep_size):
        rng = np.random.RandomState(seed if same_on_every_rank else seed + rank)
        probabilities = rng.dirichlet(np.linspace(1.0, 2.0, ep_size))
        counts = rng.multinomial(rows, probabilities).astype(np.int32).reshape(1, ep_size)
        indices = rng.permutation(rows).astype(np.int32)
        all_indices.append(indices)
        all_counts.append(counts)
    return all_indices, all_counts


def _split_counts_by_local_expert(
    all_counts: list[np.ndarray],
    num_local_experts: int,
) -> list[np.ndarray]:
    """Split each destination-rank count across its local experts."""
    all_expert_counts = []
    for rank, rank_counts in enumerate(all_counts):
        expert_counts = np.zeros(
            (rank_counts.shape[1], num_local_experts),
            dtype=np.int32,
        )
        for peer, total in enumerate(rank_counts[0]):
            quotient, remainder = divmod(int(total), num_local_experts)
            expert_counts[peer, :] = quotient
            expert_order = np.roll(
                np.arange(num_local_experts),
                rank + peer,
            )
            expert_counts[peer, expert_order[:remainder]] += 1
        all_expert_counts.append(expert_counts)
    return all_expert_counts


def _single_local_expert_counts(rank_counts: np.ndarray) -> np.ndarray:
    """Convert ``[1, EP]`` rank counts to ``[EP, 1]`` expert counts."""
    return rank_counts.reshape(-1, 1)


def _prior_variable_matrix(tokens: int, hidden: int, ep_size: int, dtype) -> FilterResult:
    """Retain every variable-count configuration covered before this CR."""
    if tokens != 256 or hidden != 128:
        return FilterResult.REDUNDANT
    if ep_size == 8 and dtype in (np.float32, ml_dtypes.bfloat16):
        return FilterResult.VALID
    if ep_size in (32, 64) and dtype == ml_dtypes.bfloat16:
        return FilterResult.VALID
    return FilterResult.REDUNDANT


@pytest_test_metadata(name="PermuteA2av")
@pytest_marks(["collectives", "PermuteA2av"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
class TestPermuteA2av:
    """Validate packed dispatch across the prior and new coverage matrices."""

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        tokens=[256],
        hidden=[128],
        expert=[64],
        ep_size=[8],
        dtype=[ml_dtypes.bfloat16],
        coverage="full",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_permute_a2av(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        tokens: int,
        hidden: int,
        expert: int,
        ep_size: int,
        dtype,
        is_negative_test_case: bool,
    ):
        """Preserve the prior symmetric-routing configuration."""
        assert expert % ep_size == 0
        np.random.seed(42)
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices, all_counts = _generate_routing(
            tokens,
            ep_size,
            seed=42,
            same_on_every_rank=True,
        )
        rank_inputs = [np.random.randn(tokens, hidden).astype(dtype) for _ in range(ep_size)]

        def create_inputs(rank_id: int):
            return {
                "hidden_states": rank_inputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": _single_local_expert_counts(all_counts[rank_id]),
                "recv_capacity": ep_size * tokens,
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=permute_a2av,
            torch_ref=permute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["recv_data", "metadata", "recv_counts", "regroup_indices", "bwmm_counts"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    @pytest.mark.parametrize("num_local_experts", [1, 2, 4, 8])
    def test_permute_a2av_multiple_local_experts(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        num_local_experts: int,
    ):
        """Validate expert-count exchange and rank-total aggregation."""
        rows = 256
        hidden = 128
        ep_size = 8
        dtype = ml_dtypes.bfloat16
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices, all_counts = _generate_routing(
            rows,
            ep_size,
            seed=500,
        )
        all_expert_counts = _split_counts_by_local_expert(
            all_counts,
            num_local_experts,
        )
        rank_inputs = [np.random.RandomState(600 + rank).randn(rows, hidden).astype(dtype) for rank in range(ep_size)]

        def create_inputs(rank_id: int):
            return {
                "hidden_states": rank_inputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": all_expert_counts[rank_id],
                "recv_capacity": ep_size * rows,
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=permute_a2av,
            torch_ref=permute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                logical_nc_config=2,
                platform_target=platform_target,
            ),
            output_keys=[
                "recv_data",
                "metadata",
                "recv_counts",
                "regroup_indices",
                "bwmm_counts",
            ],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        tokens=[256],
        hidden=[128],
        ep_size=[8, 32, 64],
        dtype=[np.float32, ml_dtypes.bfloat16],
        filter=_prior_variable_matrix,
        coverage="full",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_permute_a2av_variable(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        tokens: int,
        hidden: int,
        ep_size: int,
        dtype,
        is_negative_test_case: bool,
    ):
        """Preserve prior EP8/EP32/EP64 variable-count coverage."""
        np.random.seed(42)
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices, all_counts = _generate_routing(tokens, ep_size, seed=42)
        rank_inputs = [np.random.randn(tokens, hidden).astype(dtype) for _ in range(ep_size)]

        def create_inputs(rank_id: int):
            return {
                "hidden_states": rank_inputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": _single_local_expert_counts(all_counts[rank_id]),
                "recv_capacity": ep_size * tokens,
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=permute_a2av,
            torch_ref=permute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["recv_data", "metadata", "recv_counts", "regroup_indices", "bwmm_counts"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        rows=[128, 131],
        hidden=[128, 256],
        ep_size=[8],
        dtype=[np.float32, ml_dtypes.bfloat16],
        coverage="pairs",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_permute_a2av_skewed_counts(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        rows: int,
        hidden: int,
        ep_size: int,
        dtype,
        is_negative_test_case: bool,
    ):
        """Exercise packed placement with non-tile-aligned skewed counts."""
        np.random.seed(42)
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices, all_counts = _generate_routing(rows, ep_size, seed=100)
        rank_inputs = [np.random.randn(rows, hidden).astype(dtype) for _ in range(ep_size)]

        def create_inputs(rank_id: int):
            return {
                "hidden_states": rank_inputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": _single_local_expert_counts(all_counts[rank_id]),
                "recv_capacity": ep_size * rows,
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=permute_a2av,
            torch_ref=permute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["recv_data", "metadata", "recv_counts", "regroup_indices", "bwmm_counts"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    def test_permute_handles_worst_case_unique_token_routing(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """One rank receives exactly the EP64/H4096 routed-token capacity."""
        tokens = 65
        top_k = 2
        rows = tokens * top_k
        hidden = 4096
        ep_size = 64
        recv_capacity = ep_size * tokens
        replica_group = ReplicaGroup([list(range(ep_size))])
        rank_inputs = [
            np.random.RandomState(200 + rank).randn(rows, hidden).astype(ml_dtypes.bfloat16) for rank in range(ep_size)
        ]

        def create_inputs(rank_id: int):
            counts = np.zeros((1, ep_size), dtype=np.int32)
            counts[0, 0] = tokens
            counts[0, 1 + rank_id % (ep_size - 1)] = tokens
            indices = np.arange(rows, dtype=np.int32)
            return {
                "hidden_states": rank_inputs[rank_id],
                "packed_source_indices": indices,
                "send_counts": _single_local_expert_counts(counts),
                "recv_capacity": recv_capacity,
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=permute_a2av,
            torch_ref=permute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["recv_data", "metadata", "recv_counts", "regroup_indices", "bwmm_counts"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    def test_permute_handles_zero_count_peers(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """Each rank communicates with one peer and leaves all other counts zero."""
        rows = 129
        hidden = 256
        ep_size = 8
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices = []
        all_counts = []
        rank_inputs = []
        for rank in range(ep_size):
            destination = (rank + 1) % ep_size
            counts = np.zeros((1, ep_size), dtype=np.int32)
            counts[0, destination] = rows
            indices = np.arange(rows, dtype=np.int32)
            all_counts.append(counts)
            all_indices.append(indices)
            rank_inputs.append(np.random.RandomState(300 + rank).randn(rows, hidden).astype(ml_dtypes.bfloat16))

        def create_inputs(rank_id: int):
            return {
                "hidden_states": rank_inputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": _single_local_expert_counts(all_counts[rank_id]),
                "recv_capacity": rows,
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=permute_a2av,
            torch_ref=permute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["recv_data", "metadata", "recv_counts", "regroup_indices", "bwmm_counts"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=0,
            atol=0,
        )

    @pytest.mark.fast
    def test_permute_preserves_final_capacity_fragment(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """Copy 129 received rows into capacity 131 and preserve two exact zero rows."""
        rows = 129
        recv_capacity = 131
        hidden = 128
        ep_size = 8
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices = []
        all_counts = []
        rank_inputs = []
        for rank in range(ep_size):
            destination = (rank + 1) % ep_size
            counts = np.zeros((1, ep_size), dtype=np.int32)
            counts[0, destination] = rows
            indices = np.arange(rows, dtype=np.int32)
            rank_values = np.arange(rows * hidden, dtype=np.float32).reshape(rows, hidden)
            rank_inputs.append(rank_values + np.float32(rank * rows * hidden))
            all_counts.append(counts)
            all_indices.append(indices)

        def create_inputs(rank_id: int):
            return {
                "hidden_states": rank_inputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": _single_local_expert_counts(all_counts[rank_id]),
                "recv_capacity": recv_capacity,
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=permute_a2av,
            torch_ref=permute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["recv_data", "metadata", "recv_counts", "regroup_indices", "bwmm_counts"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=0,
            atol=0,
        )


@pytest.mark.collectives
@pytest.mark.PermuteA2av
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
@pytest.mark.parametrize(
    "rows,ep_size,hidden,concentrated",
    [
        pytest.param(1, 8, 128, True, marks=pytest.mark.fast),
        pytest.param(129, 8, 128, False, marks=pytest.mark.fast),
        pytest.param(257, 8, 128, True, marks=pytest.mark.fast),
        # All-rank reference inputs exceed the fast suite's host-memory budget.
        (8192, 64, 4096, False),
    ],
)
def test_packed_source_index_exact(test_manager, platform_target, rows, ep_size, hidden, concentrated):
    """Check packed routing against the reference, including zero peers."""
    all_indices, all_counts = _generate_routing(rows, ep_size, seed=42)
    if concentrated:
        for indices, counts in zip(all_indices, all_counts, strict=True):
            counts[:] = 0
            counts[0, 0] = rows
            indices[:] = np.arange(rows, dtype=np.int32)[::-1]
    capacity = (
        rows * ep_size
        if concentrated
        else max(sum(int(counts[0, dst]) for counts in all_counts) for dst in range(ep_size))
    )
    capacity = max(128, capacity)
    if rows == 8192:
        capacity = 65536
    replica_group = ReplicaGroup([list(range(ep_size))])

    def create_inputs(rank_id):
        indices, counts = all_indices[rank_id], all_counts[rank_id]
        inputs = {
            "hidden_states": np.random.RandomState(42 + rank_id).randn(rows, hidden).astype(ml_dtypes.bfloat16),
            "packed_source_indices": indices,
            "send_counts": _single_local_expert_counts(counts),
            "recv_capacity": capacity,
            "replica_group": replica_group,
        }
        return inputs

    CollectiveUnitTestFramework(
        test_manager=test_manager,
        kernel_entry=permute_a2av,
        torch_ref=permute_a2av_torch_ref,
        per_rank_input_generator=create_inputs,
        collective_ranks=ep_size,
    ).run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
        output_keys=["recv_data", "metadata", "recv_counts", "regroup_indices", "bwmm_counts"],
        inference_args=InferenceArgs(collective_ranks=ep_size),
        rtol=0,
        atol=0,
    )
