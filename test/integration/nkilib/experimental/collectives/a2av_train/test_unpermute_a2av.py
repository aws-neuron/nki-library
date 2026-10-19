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
"""Integration tests for the capacity-bounded unpermute_a2av kernel."""

import ml_dtypes
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.a2av_train import unpermute_a2av
from nkilib_src.nkilib.experimental.collectives.a2av_train.unpermute_a2av_torch import (
    unpermute_a2av_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.coverage_parametrized_tests import FilterResult
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework


@pytest.mark.collectives
@pytest.mark.UnpermuteA2av
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
@pytest.mark.parametrize(
    "rows,ep_size,hidden,concentrated",
    [
        pytest.param(1, 8, 128, True, marks=pytest.mark.fast),
        pytest.param(129, 8, 128, False, marks=pytest.mark.fast),
        pytest.param(257, 8, 128, True, marks=pytest.mark.fast),
        (130, 64, 4096, True),
        (8192, 64, 4096, False),
    ],
)
def test_packed_unpermute_exact(test_manager, platform_target, rows, ep_size, hidden, concentrated):
    """Check unique scatter ownership, partial tiles, skew and small BF16 values."""
    all_indices, all_counts = _generate_routing(rows, ep_size, seed=42)
    if concentrated:
        for indices, counts in zip(all_indices, all_counts, strict=True):
            counts[:] = 0
            counts[0, 0] = rows
            indices[:] = np.arange(rows, dtype=np.int32)[::-1]
    recv_counts = [
        np.array([[all_counts[source][0, rank] for source in range(ep_size)]], dtype=np.int32)
        for rank in range(ep_size)
    ]
    capacity = max(1, max(int(counts.sum()) for counts in recv_counts))

    def create_inputs(rank_id):
        indices, counts = all_indices[rank_id], all_counts[rank_id]
        # Poison padding to expose accidental reads beyond the active prefix.
        output = np.full((capacity, hidden), np.nan, dtype=ml_dtypes.bfloat16)
        active = int(recv_counts[rank_id].sum())
        rng = np.random.RandomState(800 + rank_id)
        output[:active] = (rng.standard_normal((active, hidden)) * 2**-27).astype(ml_dtypes.bfloat16)
        inputs = {
            "output": output,
            "packed_source_indices": indices,
            "send_counts": counts,
            "recv_counts": recv_counts[rank_id],
            "replica_group": ReplicaGroup([list(range(ep_size))]),
        }
        return inputs

    CollectiveUnitTestFramework(
        test_manager=test_manager,
        kernel_entry=unpermute_a2av,
        torch_ref=unpermute_a2av_torch_ref,
        per_rank_input_generator=create_inputs,
        collective_ranks=ep_size,
    ).run_test(
        test_config=None,
        compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
        output_keys=["result"],
        inference_args=InferenceArgs(collective_ranks=ep_size),
        rtol=0,
        atol=0,
    )


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


def _prior_variable_matrix(tokens: int, hidden: int, ep_size: int, dtype) -> FilterResult:
    """Retain every variable-count configuration covered before this CR."""
    if hidden != 128:
        return FilterResult.REDUNDANT
    if tokens == 256 and ep_size == 8 and dtype in (np.float32, ml_dtypes.bfloat16):
        return FilterResult.VALID
    if tokens == 256 and ep_size in (32, 64) and dtype == ml_dtypes.bfloat16:
        return FilterResult.VALID
    if tokens in (64, 192) and ep_size == 8 and dtype == ml_dtypes.bfloat16:
        return FilterResult.VALID
    return FilterResult.REDUNDANT


def _make_rank_outputs(
    rank_recv_counts: list[np.ndarray],
    capacity: int,
    hidden: int,
    dtype,
    *,
    seed: int,
) -> list[np.ndarray]:
    """Create packed receive buffers with defined data in every active row."""
    rank_outputs = []
    for rank, recv_counts in enumerate(rank_recv_counts):
        output = np.zeros((capacity, hidden), dtype=dtype)
        active_rows = int(recv_counts.sum())
        output[:active_rows] = np.random.RandomState(seed + rank).randn(active_rows, hidden).astype(dtype)
        rank_outputs.append(output)
    return rank_outputs


@pytest_test_metadata(name="UnpermuteA2av")
@pytest_marks(["collectives", "UnpermuteA2av"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
class TestUnpermuteA2av:
    """Validate packed combine across the prior and new coverage matrices."""

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        tokens=[64, 192, 256],
        hidden=[128],
        expert=[64],
        ep_size=[8],
        coverage="full",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_unpermute_a2av(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        tokens: int,
        hidden: int,
        expert: int,
        ep_size: int,
        is_negative_test_case: bool,
    ):
        """Preserve prior symmetric and partial-tail configurations."""
        assert expert % ep_size == 0
        dtype = ml_dtypes.bfloat16
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices, all_counts = _generate_routing(
            tokens,
            ep_size,
            seed=42,
            same_on_every_rank=True,
        )
        rank_recv_counts = [
            np.array([[all_counts[source][0, rank] for source in range(ep_size)]], dtype=np.int32)
            for rank in range(ep_size)
        ]
        rank_outputs = _make_rank_outputs(
            rank_recv_counts,
            ep_size * tokens,
            hidden,
            dtype,
            seed=100,
        )

        def create_inputs(rank_id: int):
            return {
                "output": rank_outputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": all_counts[rank_id],
                "recv_counts": rank_recv_counts[rank_id],
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=unpermute_a2av,
            torch_ref=unpermute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["result"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    @pytest.mark.coverage_parametrize(
        tokens=[64, 192, 256],
        hidden=[128],
        ep_size=[8, 32, 64],
        dtype=[np.float32, ml_dtypes.bfloat16],
        filter=_prior_variable_matrix,
        coverage="full",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_unpermute_a2av_variable(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        tokens: int,
        hidden: int,
        ep_size: int,
        dtype,
        is_negative_test_case: bool,
    ):
        """Preserve prior EP8/EP32/EP64 and partial-tail coverage."""
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices, all_counts = _generate_routing(tokens, ep_size, seed=200)
        rank_recv_counts = [
            np.array([[all_counts[source][0, rank] for source in range(ep_size)]], dtype=np.int32)
            for rank in range(ep_size)
        ]
        rank_outputs = _make_rank_outputs(
            rank_recv_counts,
            ep_size * tokens,
            hidden,
            dtype,
            seed=300,
        )

        def create_inputs(rank_id: int):
            return {
                "output": rank_outputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": all_counts[rank_id],
                "recv_counts": rank_recv_counts[rank_id],
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=unpermute_a2av,
            torch_ref=unpermute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["result"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=1e-2,
            atol=1e-2,
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
    def test_unpermute_a2av_skewed_counts(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        rows: int,
        hidden: int,
        ep_size: int,
        dtype,
        is_negative_test_case: bool,
    ):
        """Exercise packed combine with non-tile-aligned skewed counts."""
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices, all_counts = _generate_routing(rows, ep_size, seed=400)
        rank_recv_counts = [
            np.array([[all_counts[source][0, rank] for source in range(ep_size)]], dtype=np.int32)
            for rank in range(ep_size)
        ]
        assert any(int(recv_counts.max()) > rows // ep_size for recv_counts in rank_recv_counts), (
            "skewed coverage must exceed the compact result's balanced N/EP window"
        )
        rank_outputs = _make_rank_outputs(
            rank_recv_counts,
            ep_size * rows,
            hidden,
            dtype,
            seed=500,
        )

        def create_inputs(rank_id: int):
            return {
                "output": rank_outputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": all_counts[rank_id],
                "recv_counts": rank_recv_counts[rank_id],
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=unpermute_a2av,
            torch_ref=unpermute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["result"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_unpermute_handles_exact_capacity(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """Reverse A2AV remains correct at exact EP64/H4096 capacity."""
        tokens = 65
        top_k = 2
        rows = tokens * top_k
        hidden = 4096
        ep_size = 64
        recv_capacity = ep_size * tokens
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices = []
        all_counts = []
        for rank in range(ep_size):
            counts = np.zeros((1, ep_size), dtype=np.int32)
            counts[0, 0] = tokens
            counts[0, 1 + rank % (ep_size - 1)] = tokens
            indices = np.arange(rows, dtype=np.int32)
            all_indices.append(indices)
            all_counts.append(counts)

        rank_recv_counts = [
            np.array([[all_counts[source][0, rank] for source in range(ep_size)]], dtype=np.int32)
            for rank in range(ep_size)
        ]
        rank_outputs = _make_rank_outputs(
            rank_recv_counts,
            recv_capacity,
            hidden,
            ml_dtypes.bfloat16,
            seed=600,
        )

        def create_inputs(rank_id: int):
            return {
                "output": rank_outputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": all_counts[rank_id],
                "recv_counts": rank_recv_counts[rank_id],
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=unpermute_a2av,
            torch_ref=unpermute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["result"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=1e-2,
            atol=1e-2,
        )

    @pytest.mark.fast
    def test_unpermute_handles_zero_count_peers(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """Combine remains correct when each rank communicates with one peer."""
        rows = 129
        hidden = 256
        ep_size = 8
        replica_group = ReplicaGroup([list(range(ep_size))])
        all_indices = []
        all_counts = []
        for rank in range(ep_size):
            destination = (rank + 1) % ep_size
            counts = np.zeros((1, ep_size), dtype=np.int32)
            counts[0, destination] = rows
            indices = np.arange(rows, dtype=np.int32)
            all_counts.append(counts)
            all_indices.append(indices)

        rank_recv_counts = [
            np.array([[all_counts[source][0, rank] for source in range(ep_size)]], dtype=np.int32)
            for rank in range(ep_size)
        ]
        rank_outputs = _make_rank_outputs(
            rank_recv_counts,
            rows,
            hidden,
            ml_dtypes.bfloat16,
            seed=700,
        )

        def create_inputs(rank_id: int):
            return {
                "output": rank_outputs[rank_id],
                "packed_source_indices": all_indices[rank_id],
                "send_counts": all_counts[rank_id],
                "recv_counts": rank_recv_counts[rank_id],
                "replica_group": replica_group,
            }

        CollectiveUnitTestFramework(
            test_manager=test_manager,
            kernel_entry=unpermute_a2av,
            torch_ref=unpermute_a2av_torch_ref,
            per_rank_input_generator=create_inputs,
            collective_ranks=ep_size,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            output_keys=["result"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=1e-2,
            atol=1e-2,
        )
