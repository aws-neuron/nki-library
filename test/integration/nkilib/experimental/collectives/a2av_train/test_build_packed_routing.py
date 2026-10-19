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
"""Exact hardware checks for stable packed routing-vector construction."""

import numpy as np
import pytest
from nkilib_src.nkilib.experimental.collectives.a2av_train.build_packed_routing import build_packed_routing
from nkilib_src.nkilib.experimental.collectives.a2av_train.build_packed_routing_torch import (
    build_packed_routing_torch_ref,
)

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_test_metadata
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@pytest_test_metadata(name="PackedRouting")
@pytest.mark.platforms(exclude=[Platforms.TRN1, Platforms.TRN2])
@pytest.mark.skip_simulation
class TestPackedRouting:
    @pytest.mark.coverage_parametrize(
        rows=[1, 129, 257, 8192],
        ep=[8, 64, 128],
        pattern=["random", "concentrated", "empty_peers", "reverse"],
        coverage="pairs",
        enable_automatic_boundary_tests=False,
        enable_invalid_combination_tests=False,
    )
    def test_stable_routing(self, test_manager, platform_target, rows, ep, pattern, is_negative_test_case):
        def inputs(test_config):
            rng = np.random.default_rng(42)
            if pattern == "random":
                ids = rng.integers(0, ep, size=rows, dtype=np.int32)
            elif pattern == "concentrated":
                ids = np.full(rows, ep - 1, dtype=np.int32)
            elif pattern == "empty_peers":
                ids = (np.arange(rows, dtype=np.int32) % 2) * (ep - 1)
            else:
                ids = ep - 1 - np.arange(rows, dtype=np.int32) % ep
            counts = np.bincount(ids, minlength=ep).astype(np.int32).reshape(1, ep)
            return {"ids": ids, "counts": counts}

        UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=build_packed_routing,
            kernel_input_generator=inputs,
            torch_ref=torch_ref_wrapper(build_packed_routing_torch_ref),
            output_tensor_descriptor=lambda kernel_input: {"source_indices": np.zeros(rows, dtype=np.int32)},
        ).run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            atol=0,
            rtol=0,
            is_negative_test=is_negative_test_case,
        )
