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
"""Scale-sensitive skewed-routing regression for packed unpermute on Trn3."""

import ml_dtypes
import numpy as np
import pytest
from nki.collectives import ReplicaGroup
from nkilib_src.nkilib.experimental.collectives.a2av_train import unpermute_a2av
from nkilib_src.nkilib.experimental.collectives.a2av_train.unpermute_a2av_torch import (
    unpermute_a2av_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, InferenceArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_collective_framework import CollectiveUnitTestFramework


@pytest_test_metadata(name="UnpermuteA2avSkewedScale")
@pytest_marks(["collectives", "UnpermuteA2av"])
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@pytest.mark.high_rank
@pytest.mark.skip_simulation
class TestUnpermuteA2avSkewedScale:
    @pytest.mark.fast
    def test_tiny_bfloat16_values_preserve_scale_ep64_h4096(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
    ):
        """Catch amplification hidden by order-one inputs and loose absolute tolerances."""
        tokens = 65
        rows = tokens * 2
        hidden = 4096
        ep_size = 64
        capacity = ep_size * tokens
        replica_group = ReplicaGroup([list(range(ep_size))])

        all_indices = []
        all_counts = []
        for rank in range(ep_size):
            second_destination = 1 + rank % (ep_size - 1)
            counts = np.zeros((1, ep_size), dtype=np.int32)
            counts[0, 0] = tokens
            counts[0, second_destination] = tokens
            indices = np.arange(rows, dtype=np.int32)
            all_indices.append(indices)
            all_counts.append(counts)

        rank_recv_counts = [
            np.array(
                [[all_counts[source][0, rank] for source in range(ep_size)]],
                dtype=np.int32,
            )
            for rank in range(ep_size)
        ]
        rank_outputs = []
        for rank, recv_counts in enumerate(rank_recv_counts):
            output = np.zeros((capacity, hidden), dtype=ml_dtypes.bfloat16)
            active_rows = int(recv_counts.sum())
            values = np.random.RandomState(800 + rank).choice(
                np.array([-4, -2, -1, 1, 2, 4], dtype=np.float32),
                size=(active_rows, hidden),
            )
            output[:active_rows] = (values * np.float32(2**-27)).astype(ml_dtypes.bfloat16)
            rank_outputs.append(output)

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
            compiler_args=CompilerArgs(
                logical_nc_config=2,
                platform_target=platform_target,
            ),
            output_keys=["result"],
            inference_args=InferenceArgs(collective_ranks=ep_size),
            rtol=0,
            atol=0,
        )
