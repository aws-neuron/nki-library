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
"""Direct kernel checks for packed-row metadata, including empty segments and tails."""

import nki
import nki.language as nl
import numpy as np
import pytest
from nkilib_src.nkilib.experimental.collectives.a2av_train._packed_metadata import _build_packed_regroup_indices
from nkilib_src.nkilib.experimental.collectives.a2av_train._packed_metadata_torch import (
    _a2av_packed_regroup_indices_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@nki.jit
def _packed_regroup_indices_kernel(
    recv_expert_counts: nl.NkiTensor,
    recv_capacity: int,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Single-program entry for the internal packed-metadata builder."""
    return _build_packed_regroup_indices(recv_expert_counts, recv_capacity, 1, 0)


@pytest_test_metadata(name="PackedMetadata")
@pytest_marks(["packed_metadata", "collectives"])
class TestPackedMetadata:
    @pytest.mark.fast
    @pytest.mark.parametrize(
        "recv_expert_counts,dtype,recv_capacity",
        [
            pytest.param([[0, 3], [2, 1]], np.int32, 6, id="zero-segment-no-tail"),
            pytest.param([[0, 0]] * 4, np.int32, 129, id="all-zero-counts"),
            pytest.param(
                [[0, 3, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 1, 0, 0]],
                np.uint32,
                17,
                id="empty-sources-and-experts",
            ),
            pytest.param([[30, 40], [20, 37]], np.int32, 127, id="partial-tile"),
            pytest.param([[31, 40], [20, 37]], np.uint32, 128, id="exact-tile"),
            pytest.param([[31, 40], [20, 37]], np.int32, 129, id="partial-second-tile"),
            pytest.param(
                ((np.arange(128).reshape(64, 2) * 7) % 11).tolist(),
                np.int32,
                4096,
                id="ep64-nle2",
            ),
            pytest.param(
                ((np.arange(64).reshape(8, 8) * 3) % 5).tolist(),
                np.uint32,
                257,
                id="nle8-multiple-tiles",
            ),
            pytest.param([[1, 1]] * 128, np.int32, 513, id="max-sources"),
            pytest.param([[1] * 128] * 2, np.uint32, 513, id="max-local-experts"),
            pytest.param([[512, 512]] * 64, np.int32, 65536, id="ep64-full"),
            pytest.param([[512, 512]] * 64, np.uint32, 131072, id="ep64-tail"),
            pytest.param([[0, 2, 1], [3, 0, 0], [1, 0, 4], [0, 0, 0]], np.int32, 128, id="nle3"),
        ],
    )
    def test_packed_metadata(
        self,
        test_manager: Orchestrator,
        platform_target: Platforms,
        recv_expert_counts: list[list[int]],
        dtype: type,
        recv_capacity: int,
    ) -> None:
        # Params stay plain lists so the metrics collector can hash them; build the array here.
        expert_counts = np.array(recv_expert_counts, dtype=dtype)

        def input_generator(_test_config):
            return {
                "recv_expert_counts": expert_counts,
                "recv_capacity": recv_capacity,
            }

        def output_tensors(_kernel_input):
            return {
                "regroup_indices": np.empty(recv_capacity, dtype=np.int32),
                "bwmm_counts": np.empty(expert_counts.shape[1], dtype=np.int32),
            }

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_packed_regroup_indices_kernel,
            torch_ref=torch_ref_wrapper(_a2av_packed_regroup_indices_torch_ref),
            kernel_input_generator=input_generator,
            output_tensor_descriptor=output_tensors,
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(
                platform_target=platform_target,
                # This builder is not sharded: match the consumer's grid (1,).
                logical_nc_config=1,
            ),
            rtol=0,
            atol=0,
        )
