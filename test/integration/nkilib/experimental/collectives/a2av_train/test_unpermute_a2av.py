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
"""Tests for unpermute_a2av kernel."""

import ml_dtypes
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.a2av_train import unpermute_a2av
from nkilib_src.nkilib.experimental.collectives.a2av_train.unpermute_a2av_torch import unpermute_a2av_torch_ref

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_parametrize import pytest_parametrize
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework


def _generate_symmetric(tokens, expert, ep_size, multiples_of=32, seed=42):
    """Same send_indices/send_counts on every rank (symmetric MoE routing)."""
    np.random.seed(seed)
    expert_mask = np.zeros((tokens, expert), dtype=np.float32)
    for t_idx in range(tokens):
        selected = np.random.choice(expert, size=2, replace=False)
        expert_mask[t_idx, selected] = 1.0
    experts_per_rank = expert // ep_size
    ep_mask = expert_mask.reshape(tokens, ep_size, experts_per_rank)
    ep_mask = (ep_mask != 0).any(axis=2)

    send_indices = np.full((tokens, ep_size), tokens, dtype=np.int32)
    send_counts = np.zeros((1, ep_size), dtype=np.int32)
    for ep_dst in range(ep_size):
        idx = np.where(ep_mask[:, ep_dst])[0]
        n = (len(idx) // multiples_of) * multiples_of
        send_counts[0, ep_dst] = n
        send_indices[:n, ep_dst] = idx[:n]
    return send_indices, send_counts


def _generate_variable_per_rank(tokens, ep_size, multiples_of=32, seed=42):
    """Per-rank variable send_counts."""
    rng = np.random.RandomState(seed)
    choices = np.arange(0, tokens + 1, multiples_of)
    all_sc, all_si = [], []
    for _rank in range(ep_size):
        sc = rng.choice(choices, size=(1, ep_size)).astype(np.int32)
        while int(sc[0].sum()) > tokens:
            idx = int(sc[0].argmax())
            sc[0, idx] = max(0, sc[0, idx] - multiples_of)
        si = np.full((tokens, ep_size), tokens, dtype=np.int32)
        for d in range(ep_size):
            n = int(sc[0, d])
            si[:n, d] = np.arange(n)
        all_sc.append(sc)
        all_si.append(si)
    return all_si, all_sc


PARAM_NAMES_SYM = "tokens, hidden, expert, ep_size"
TEST_PARAMS_SYM = [
    (256, 128, 64, 8),
]
PARAM_NAMES_VAR = "tokens, hidden, ep_size, dtype"

TEST_PARAMS_VAR = [
    (256, 128, 8, np.float32),
    (256, 128, 8, ml_dtypes.bfloat16),
    (256, 128, 32, ml_dtypes.bfloat16),
    (256, 128, 64, ml_dtypes.bfloat16),
]


@pytest_test_metadata(name="UnpermuteA2av")
@pytest_marks(["collectives", "UnpermuteA2av"])
@pytest.mark.skip_simulation
@pytest.mark.high_rank
class TestUnpermuteA2av:
    """Test unpermute_a2av kernel."""

    @pytest.mark.fast
    @pytest_parametrize(PARAM_NAMES_SYM, TEST_PARAMS_SYM)
    def test_unpermute_a2av(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        tokens: int,
        hidden: int,
        expert: int,
        ep_size: int,
    ):
        """Symmetric routing: every rank has identical send_counts / send_indices."""
        np.random.seed(42)
        dtype = ml_dtypes.bfloat16
        replica_group = ReplicaGroup([list(range(ep_size))])
        elem_bytes = np.dtype(dtype).itemsize
        mo = max(1, 8192 // (elem_bytes * hidden))
        send_indices, send_counts = _generate_symmetric(tokens, expert, ep_size, multiples_of=mo)

        rank_outputs = []
        for rank in range(ep_size):
            out = np.zeros((ep_size * tokens, hidden), dtype=dtype)
            count = int(send_counts[0, rank])
            for src in range(ep_size):
                out[src * tokens : src * tokens + count, :] = np.random.randn(count, hidden).astype(dtype)
            rank_outputs.append(out)

        def create_inputs(rank_id: int):
            rc = np.full((1, ep_size), send_counts[0, rank_id], dtype=np.int32)
            return {
                "output": rank_outputs[rank_id],
                "send_indices": send_indices,
                "recv_counts": rc,
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
    @pytest_parametrize(PARAM_NAMES_VAR, TEST_PARAMS_VAR)
    def test_unpermute_a2av_variable(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        tokens: int,
        hidden: int,
        ep_size: int,
        dtype,
    ):
        """Per-rank variable counts (DGE-aligned per dtype)."""
        np.random.seed(42)
        replica_group = ReplicaGroup([list(range(ep_size))])
        elem_bytes = np.dtype(dtype).itemsize
        mo = max(1, 8192 // (elem_bytes * hidden))
        all_si, all_sc = _generate_variable_per_rank(tokens, ep_size, multiples_of=mo, seed=42)

        rank_recv_counts = [
            np.array([[all_sc[s][0, R] for s in range(ep_size)]], dtype=np.int32) for R in range(ep_size)
        ]

        rank_outputs = []
        for rank in range(ep_size):
            rc = rank_recv_counts[rank]
            out = np.zeros((ep_size * tokens, hidden), dtype=dtype)
            for src in range(ep_size):
                n = int(rc[0, src])
                if n > 0:
                    out[src * tokens : src * tokens + n, :] = np.random.randn(n, hidden).astype(dtype)
            rank_outputs.append(out)

        def create_inputs(rank_id: int):
            return {
                "output": rank_outputs[rank_id],
                "send_indices": all_si[rank_id],
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
