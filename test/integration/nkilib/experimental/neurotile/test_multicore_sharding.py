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

"""Multi-core sharding integration tests (LNC=2).

Block-sharded and interleaved tensor_add kernels validated against a
CPU reference. The interleaved variant pre-refactor silently produced
wrong output (shard_stride didn't reach DMA); these tests pin the fix.
"""

import ml_dtypes
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
import pytest
import torch
from nkilib_src.nkilib.experimental import neurotile as nt

from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.pytest_test_metadata import pytest_marks
from test.utils.test_orchestrator import Orchestrator
from test.utils.unit_test_framework import UnitTestFramework, torch_ref_wrapper


@nki.jit
def _tensor_add_block_sharded(a, b):
    M, N = a.shape
    c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    own = nt.block_range(
        rank=nl.program_id(0),
        num_shards=nl.num_programs(0),
        total=nt.ceiling_div(M, 128),
    )
    A = nt.tiles(a, tile_size=(128, 512))[own, :]
    B = nt.tiles(b, tile_size=(128, 512))[own, :]
    C = nt.tiles(c, tile_size=(128, 512))[own, :]
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            at = A[i, j].load()
            bt = B[i, j].load()
            nisa.tensor_tensor(at.data, at.data, bt.data, op=nl.add)
            C[i, j].store(at.data)
    return c


@nki.jit
def _tensor_add_interleaved(a, b):
    M, N = a.shape
    c = nl.ndarray((M, N), dtype=a.dtype, buffer=nl.shared_hbm)
    own = nt.interleaved_range(
        rank=nl.program_id(0),
        num_shards=nl.num_programs(0),
        total=nt.ceiling_div(M, 128),
    )
    A = nt.tiles(a, tile_size=(128, 512))[own, :]
    B = nt.tiles(b, tile_size=(128, 512))[own, :]
    C = nt.tiles(c, tile_size=(128, 512))[own, :]
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            at = A[i, j].load()
            bt = B[i, j].load()
            nisa.tensor_tensor(at.data, at.data, bt.data, op=nl.add)
            C[i, j].store(at.data)
    return c


def _add_inputs(M, N, seed=1):
    def _gen(_):
        np.random.seed(seed)
        a = (np.random.randn(M, N) * 0.1).astype(ml_dtypes.bfloat16)
        b = (np.random.randn(M, N) * 0.1).astype(ml_dtypes.bfloat16)
        return {"a": a, "b": b}

    return _gen


def _add_output(kernel_input):
    return {"out": np.zeros_like(kernel_input["a"])}


def _add_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a + b


@pytest_marks(["neurotile"])
class TestTensorAddBlockMultiCore:
    """Block-sharded tensor_add on 2 cores."""

    @pytest.mark.fast
    def test_correctness(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_tensor_add_block_sharded,
            torch_ref=torch_ref_wrapper(_add_ref),
            kernel_input_generator=_add_inputs(512, 512, seed=1),
            output_tensor_descriptor=_add_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=0.02,
            atol=0.02,
        )


@pytest_marks(["neurotile"])
class TestTensorAddInterleavedMultiCore:
    """Interleaved tensor_add on 2 cores."""

    @pytest.mark.fast
    def test_correctness_256(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_tensor_add_interleaved,
            torch_ref=torch_ref_wrapper(_add_ref),
            kernel_input_generator=_add_inputs(256, 512, seed=2),
            output_tensor_descriptor=_add_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=0.02,
            atol=0.02,
        )

    def test_correctness_1024(self, test_manager: Orchestrator, platform_target: Platforms):
        framework = UnitTestFramework(
            test_manager=test_manager,
            kernel_entry=_tensor_add_interleaved,
            torch_ref=torch_ref_wrapper(_add_ref),
            kernel_input_generator=_add_inputs(1024, 512, seed=3),
            output_tensor_descriptor=_add_output,
        )
        framework.run_test(
            test_config=None,
            compiler_args=CompilerArgs(platform_target=platform_target, logical_nc_config=2),
            rtol=0.02,
            atol=0.02,
        )
